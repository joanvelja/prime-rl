from types import SimpleNamespace

from prime_rl.inference.server import setup_vllm_env


def test_setup_vllm_env_creates_rpc_base_path(monkeypatch, tmp_path):
    rpc_base_path = tmp_path / "vllm-rpc"
    monkeypatch.setenv("VLLM_RPC_BASE_PATH", str(rpc_base_path))

    setup_vllm_env(SimpleNamespace(enable_lora=False, use_deep_gemm=False))

    assert rpc_base_path.is_dir()
