from __future__ import annotations

import asyncio
import atexit
import multiprocessing as mp
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import verifiers as vf
from verifiers.serve import ZMQEnvClient, ZMQEnvServer
from verifiers.utils.serve_utils import get_free_port

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.vf_utils import resolve_num_workers
from prime_rl.utils.logger import ProgressTracker, get_logger
from prime_rl.utils.utils import strip_env_version

REQUIRED_STATE_COLUMNS = ["trajectory", "sampling_args"]


class Env:
    """Wraps a verifiers Environment with its config and per-env state."""

    def __init__(self, config: EnvConfig):
        env_id = strip_env_version(config.id)
        self.config = config
        self._env: vf.Environment = vf.load_environment(env_id, **config.args)
        self.name: str = config.resolved_name
        self.uses_group_scoring: bool = any(
            self._env.rubric._is_group_func(func) for func in self._env.rubric._get_reward_funcs()
        )
        self.sampling_args: dict = {}
        self._process: mp.Process | None = None

    def get_dataset(self, seed: int | None = None):
        return self._env.get_dataset(seed=seed)

    def spawn(
        self,
        log_dir: Path,
        max_concurrent: int | None = None,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn an env server if no explicit address is configured."""
        if self.config.address is not None:
            return
        num_workers = resolve_num_workers(self.config.num_workers, max_concurrent)
        address = f"tcp://127.0.0.1:{get_free_port()}"
        process = mp.get_context("spawn").Process(
            target=ZMQEnvServer.run_server,
            args=(
                strip_env_version(self.config.id),
                self.config.args,
                self.config.extra_env_kwargs,
                log_level,
                (log_dir / self.name).as_posix(),
            ),
            kwargs=dict(
                address=address,
                json_logging=json_logging,
                console_logging=False,
                num_workers=num_workers,
            ),
            daemon=False,
        )
        process.start()
        self.config.address = address
        self._process = process
        get_logger().info(f"Spawned env server for {self.name} with {num_workers} worker(s)")

    async def connect(self) -> None:
        """Connect an env client to the server and assign it."""
        if self.config.address is None:
            raise RuntimeError(
                f"Env {self.name} has no address configured. Call spawn() first or set address in config."
            )
        get_logger().info(f"Connecting env {self.name} to server at {self.config.address}")
        client = ZMQEnvClient(address=self.config.address, name=self.name)
        await client.wait_for_server_startup()
        self._env.env_client = client

    async def run_rollout(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
    ) -> vf.RolloutOutput:
        rollout_input = vf.RolloutInput(**example)
        return await self._env.run_rollout(
            rollout_input,
            client=client,
            model=model_name,
            sampling_args=self.sampling_args,
            max_retries=self.config.max_retries,
            state_columns=REQUIRED_STATE_COLUMNS,
        )

    async def run_group(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        rollouts_per_example: int,
    ) -> list[vf.RolloutOutput]:
        group_inputs = [vf.RolloutInput(**example) for _ in range(rollouts_per_example)]
        return await self._env.run_group(
            group_inputs,
            client=client,
            model=model_name,
            sampling_args=self.sampling_args,
            max_retries=self.config.max_retries,
            state_columns=REQUIRED_STATE_COLUMNS,
        )

    def shutdown(self) -> None:
        if self._process is None:
            return
        logger = get_logger()
        self._process.terminate()
        self._process.join(timeout=25)
        if self._process.is_alive():
            logger.warning(f"Env server {self._process.pid} did not exit after 25s, force killing")
            self._process.kill()
            self._process.join(timeout=5)
        self._process = None


class TrainEnv(Env):
    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig):
        super().__init__(config)


class EvalEnv(Env):
    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)

    def get_dataset(self, seed: int | None = None):
        return self._env.get_eval_dataset(seed=seed)

    async def evaluate(
        self,
        model_name: str,
        get_client: Callable[[], Awaitable[vf.ClientConfig]],
    ) -> list[vf.RolloutOutput]:
        """Generate eval rollouts across all eval inputs, round-robining clients."""
        inputs = self._env._get_eval_inputs(self.config.num_examples, self.config.rollouts_per_example)
        total_rollouts = len(inputs)
        pbar = ProgressTracker(total=total_rollouts, desc=f"Evaluating {self.name}")

        async def _run_group(example: dict) -> list[vf.RolloutOutput] | None:
            try:
                client = await get_client()
                rollout_input = vf.RolloutInput(**example)
                # rollouts_per_example=1 because _get_eval_inputs already repeats examples
                result = await self._env.run_group(
                    [rollout_input],
                    client=client,
                    model=model_name,
                    sampling_args=self.sampling_args,
                    max_retries=self.config.max_retries,
                    state_columns=REQUIRED_STATE_COLUMNS,
                )
                pbar.update(1)
                return result
            except Exception as e:
                get_logger().warning(f"Group failed: {e}")
                pbar.update(1)
                return None

        try:
            group_outputs_list = await asyncio.gather(*[_run_group(example) for example in inputs])
        finally:
            pbar.close()

        failed_groups = sum(1 for g in group_outputs_list if g is None)
        if failed_groups:
            get_logger().warning(f"{failed_groups}/{len(group_outputs_list)} groups failed")

        return [output for group in group_outputs_list if group is not None for output in group]


class Envs:
    """Base container for a set of Env instances."""

    _envs: dict[str, Env]

    @property
    def names(self) -> list[str]:
        return list(self._envs.keys())

    @property
    def configs(self) -> list[EnvConfig]:
        return [env.config for env in self._envs.values()]

    def get(self, name: str) -> Env:
        return self._envs[name]

    def set_sampling_args(self, sampling_args: dict) -> None:
        for env in self:
            env.sampling_args = sampling_args

    def __iter__(self):
        return iter(self._envs.values())

    def __len__(self) -> int:
        return len(self._envs)

    def spawn(
        self,
        log_dir: Path,
        max_concurrent: int | None = None,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn env servers for all envs without an explicit address."""
        for env in self:
            env.spawn(
                log_dir=log_dir,
                max_concurrent=max_concurrent,
                log_level=log_level,
                json_logging=json_logging,
            )
        atexit.register(self.shutdown)

    async def connect(self) -> None:
        """Connect all env clients to their servers and wait for health (in parallel)."""
        await asyncio.gather(*(env.connect() for env in self))

    def shutdown(self) -> None:
        """Terminate all spawned env server processes."""
        processes = [env._process for env in self if env._process is not None]
        if not processes:
            return
        logger = get_logger()
        logger.info(f"Shutting down {len(processes)} env server(s), waiting for sandbox cleanup...")
        for env in self:
            env.shutdown()


class TrainEnvs(Envs):
    """Collection of training environments."""

    def __init__(self, configs: Sequence[TrainEnvConfig]):
        self._envs: dict[str, Env] = {}
        for config in configs:
            env = TrainEnv(config)
            self._envs[env.name] = env


class EvalEnvs(Envs):
    """Collection of evaluation environments."""

    def __init__(self, configs: Sequence[EvalEnvConfig]):
        self._envs: dict[str, Env] = {}
        for config in configs:
            env = EvalEnv(config)
            self._envs[env.name] = env
