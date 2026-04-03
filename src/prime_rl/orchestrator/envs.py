from __future__ import annotations

import atexit
import multiprocessing as mp
from collections.abc import Sequence
from pathlib import Path

import verifiers as vf

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.orchestrator.vf_utils import (
    resolve_num_workers,
    setup_env_client,
    spawn_env_server,
    wait_for_env_servers,
)
from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import strip_env_version


class Envs:
    """Holds a set of environments."""

    def __init__(self, configs: Sequence[EnvConfig]):
        def load_env(config: EnvConfig) -> vf.Environment:
            env_id = strip_env_version(config.id)
            return vf.load_environment(env_id, **config.args)

        self.configs = configs
        self.envs = {config.resolved_name: load_env(config) for config in configs}
        self._processes: list[mp.Process] = []
        self._addresses: list[str] = []

    @property
    def names(self) -> list[str]:
        return list(self.envs.keys())

    def get_env(self, name: str) -> vf.Environment:
        return self.envs[name]

    def items(self):
        return self.envs.items()

    def __len__(self) -> int:
        return len(self.envs)

    def spawn(
        self,
        log_dir: Path,
        max_concurrent: int | None = None,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn env servers for envs without an explicit address."""
        logger = get_logger()
        for cfg, name in zip(self.configs, self.names):
            if cfg.address is not None:
                self._addresses.append(cfg.address)
                continue
            logger.info(f"Spawning env server for {name} with {cfg.num_workers} worker(s)")
            env_id = strip_env_version(cfg.id)
            num_workers = resolve_num_workers(cfg.num_workers, max_concurrent)
            env_log_dir = (log_dir / name).as_posix()
            address, process = spawn_env_server(
                env_id=env_id,
                env_args=cfg.args,
                extra_env_kwargs=cfg.extra_env_kwargs,
                num_workers=num_workers,
                log_level=log_level,
                log_dir=env_log_dir,
                json_logging=json_logging,
            )
            cfg.address = address
            self._processes.append(process)
            self._addresses.append(address)
            logger.debug(f"Spawned env server for {name} with {num_workers} worker(s)")

        atexit.register(self.shutdown)

    async def connect(self) -> None:
        """Connect env clients to servers, wait for health, and assign to envs."""
        logger = get_logger()
        clients = []
        for address, name in zip(self._addresses, self.names):
            logger.info(f"Connecting env {name} to server at {address}")
            clients.append(setup_env_client(address=address, name=name))

        await wait_for_env_servers(clients)

        for name, client in zip(self.names, clients):
            self.get_env(name).env_client = client

    def shutdown(self) -> None:
        """Terminate all spawned env server processes."""
        if not self._processes:
            return
        logger = get_logger()
        logger.info(f"Shutting down {len(self._processes)} env server(s), waiting for sandbox cleanup...")
        for proc in self._processes:
            proc.terminate()
        for proc in self._processes:
            proc.join(timeout=25)
            if proc.is_alive():
                logger.warning(f"Env server {proc.pid} did not exit after 25s, force killing")
                proc.kill()
                proc.join(timeout=5)
        self._processes.clear()
