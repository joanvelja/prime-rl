import getpass
import os
import tempfile

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    cache_root = os.environ.get("VLLM_CACHE_ROOT")
    if not cache_root:
        cache_dir = f".vllm_cache_{getpass.getuser()}"
        if job_id := os.environ.get("SLURM_JOB_ID"):
            cache_dir = f"{cache_dir}_{job_id}"
        cache_root = os.path.join(tempfile.gettempdir(), cache_dir)
        os.environ["VLLM_CACHE_ROOT"] = cache_root

    os.environ.setdefault("DG_JIT_CACHE_DIR", os.path.join(cache_root, "deep_gemm"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache_root, "triton"))
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
