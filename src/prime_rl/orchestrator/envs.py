from __future__ import annotations

import verifiers as vf

from prime_rl.configs.orchestrator import EnvConfig
from prime_rl.utils.utils import strip_env_version


class Envs:
    """Holds a set of environments."""

    def __init__(self, configs: list[EnvConfig]):
        def load_env(config: EnvConfig) -> vf.Environment:
            env_id = strip_env_version(config.id)
            return vf.load_environment(env_id, **config.args)

        self.configs = configs
        self.envs = {config.resolved_name: load_env(config) for config in configs}

    @property
    def names(self) -> list[str]:
        return list(self.envs.keys())

    def get_env(self, name: str) -> vf.Environment:
        return self.envs[name]

    def items(self):
        return self.envs.items()

    def __len__(self) -> int:
        return len(self.envs)
