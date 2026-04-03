from __future__ import annotations

import atexit
import multiprocessing as mp
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import verifiers as vf

from prime_rl.configs.orchestrator import EnvConfig, EvalEnvConfig, TrainEnvConfig
from prime_rl.orchestrator.vf_utils import (
    evaluate,
    resolve_num_workers,
    run_group,
    run_rollout,
    setup_env_client,
    spawn_env_server,
    task_uses_group_scoring,
    wait_for_env_servers,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import strip_env_version


class Env:
    """Wraps a verifiers Environment with its config and per-env state."""

    def __init__(self, config: EnvConfig):
        env_id = strip_env_version(config.id)
        self.config = config
        self.vf_env: vf.Environment = vf.load_environment(env_id, **config.args)
        self.name: str = config.resolved_name
        self.max_retries: int = config.max_retries
        self.ratio: float | None = config.ratio
        self.uses_group_scoring: bool = task_uses_group_scoring(self.vf_env, self.name)
        self.sampling_args: dict = {}
        self._process: mp.Process | None = None

    def get_dataset(self, seed: int | None = None):
        return self.vf_env.get_dataset(seed=seed)

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
        logger = get_logger()
        num_workers = resolve_num_workers(self.config.num_workers, max_concurrent)
        env_log_dir = (log_dir / self.name).as_posix()
        address, process = spawn_env_server(
            env_id=strip_env_version(self.config.id),
            env_args=self.config.args,
            extra_env_kwargs=self.config.extra_env_kwargs,
            num_workers=num_workers,
            log_level=log_level,
            log_dir=env_log_dir,
            json_logging=json_logging,
        )
        self.config.address = address
        self._process = process
        logger.info(f"Spawned env server for {self.name} with {num_workers} worker(s)")

    async def connect(self) -> None:
        """Connect an env client to the server and assign it."""
        if self.config.address is None:
            raise RuntimeError(
                f"Env {self.name} has no address configured. Call spawn() first or set address in config."
            )
        logger = get_logger()
        logger.info(f"Connecting env {self.name} to server at {self.config.address}")
        client = setup_env_client(address=self.config.address, name=self.name)
        await wait_for_env_servers([client])
        self.vf_env.env_client = client

    async def generate_rollout(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
    ) -> vf.RolloutOutput:
        return await run_rollout(
            env=self.vf_env,
            client=client,
            example=example,
            model_name=model_name,
            sampling_args=self.sampling_args,
            max_retries=self.max_retries,
        )

    async def generate_group(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        rollouts_per_example: int,
    ) -> list[vf.RolloutOutput]:
        return await run_group(
            env=self.vf_env,
            client=client,
            example=example,
            model_name=model_name,
            rollouts_per_example=rollouts_per_example,
            sampling_args=self.sampling_args,
            max_retries=self.max_retries,
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
    """Env for training."""

    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig):
        super().__init__(config)


class EvalEnv(Env):
    """Env for evaluation — dataset comes from the eval split."""

    config: EvalEnvConfig

    def __init__(self, config: EvalEnvConfig):
        super().__init__(config)
        self.num_examples: int = config.num_examples
        self.rollouts_per_example: int = config.rollouts_per_example

    def get_dataset(self, seed: int | None = None):
        return self.vf_env.get_eval_dataset(seed=seed)

    async def evaluate(
        self,
        model_name: str,
        sampling_args: dict,
        get_client: Callable[[], Awaitable[vf.ClientConfig]],
    ) -> list[vf.RolloutOutput]:
        return await evaluate(
            env=self.vf_env,
            model_name=model_name,
            sampling_args=sampling_args,
            num_examples=self.num_examples,
            rollouts_per_example=self.rollouts_per_example,
            get_client=get_client,
            max_retries=self.max_retries,
        )


class Envs:
    """Holds a set of Env instances."""

    def __init__(self, configs: Sequence[EnvConfig]):
        self._envs: dict[str, Env] = {}
        for config in configs:
            env = EvalEnv(config) if isinstance(config, EvalEnvConfig) else TrainEnv(config)
            self._envs[env.name] = env

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
        """Connect all env clients to their servers and wait for health."""
        clients = []
        logger = get_logger()
        for env in self:
            logger.info(f"Connecting env {env.name} to server at {env.config.address}")
            clients.append(setup_env_client(address=env.config.address, name=env.name))

        await wait_for_env_servers(clients)

        for env, client in zip(self, clients):
            env.vf_env.env_client = client

    def shutdown(self) -> None:
        """Terminate all spawned env server processes."""
        processes = [env._process for env in self if env._process is not None]
        if not processes:
            return
        logger = get_logger()
        logger.info(f"Shutting down {len(processes)} env server(s), waiting for sandbox cleanup...")
        for env in self:
            env.shutdown()
