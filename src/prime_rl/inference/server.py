import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli

_INDEX_TOPK_FREQ_ENV_VAR = "PRIME_RL_INDEX_TOPK_FREQ"


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"

    if config.model.index_topk_freq is not None:
        os.environ[_INDEX_TOPK_FREQ_ENV_VAR] = str(config.model.index_topk_freq)
    else:
        os.environ.pop(_INDEX_TOPK_FREQ_ENV_VAR, None)


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
