from __future__ import annotations

import argparse
from pathlib import Path

from prime_rl.entrypoints import launch


def test_rlvr_command_uses_existing_rl_entrypoint() -> None:
    args = argparse.Namespace(
        config=[Path("configs/omni_math2/run.toml")],
        output_dir=Path("/tmp/run"),
        dry_run=True,
        extra=[],
    )

    assert launch._build_rlvr_command(args) == [
        "uv",
        "run",
        "--no-sync",
        "rl",
        "@",
        "configs/omni_math2/run.toml",
        "--output-dir",
        "/tmp/run",
        "--dry-run",
    ]


def test_shell_join_keeps_slurm_job_id_expansion_live() -> None:
    rendered = launch._shell_join(["cmd", "--job", "${SLURM_JOB_ID}", "--literal", "a b"])

    assert '--job "$SLURM_JOB_ID"' in rendered
    assert "--literal 'a b'" in rendered


def test_offline_eval_args_default_to_routed_multinode() -> None:
    args = argparse.Namespace(
        arm="arm",
        run_root=Path("outputs/run/run_default"),
        weights_root=None,
        output_dir=Path("/tmp/eval"),
        base_model="model",
        num_examples=600,
        rollouts_per_example=8,
        max_concurrency=64,
        score_max_concurrency=1024,
        max_retries=3,
        ks=[1, 2, 8],
        steps=[25],
        step_interval=25,
        min_step=25,
        max_step=100,
        base_url=None,
        admin_url=[],
        nodes=8,
        gpus_per_node=4,
        dp_per_node=4,
        tp=1,
        api_server_count=4,
        dp_local=4,
        port=9800,
        router_port=None,
        backend_port=9900,
        gpu_memory_utilization=0.95,
        max_model_len=16384,
        max_num_seqs=192,
        max_num_batched_tokens=65536,
    )

    rendered = launch._offline_eval_args(args)

    assert "--launch-mode" in rendered
    assert "srun_multinode" in rendered
    assert "--launch-nodes" in rendered
    assert "8" in rendered
    assert "--launch-srun-job-id" in rendered
    assert "${SLURM_JOB_ID}" in rendered


def test_offline_eval_parser_defaults_to_high_concurrency(monkeypatch) -> None:
    monkeypatch.delenv("OFFLINE_EVAL_MAX_CONCURRENCY", raising=False)

    args = launch.build_parser().parse_args(
        [
            "offline-eval",
            "--arm",
            "arm",
            "--run-root",
            "outputs/run/run_default",
        ]
    )

    assert args.max_concurrency == 256


def test_offline_eval_env_script_preflights_requested_weights() -> None:
    args = argparse.Namespace(
        root=Path("/repo"),
        run_root=Path("outputs/run/run_default"),
        weights_root=Path("outputs/run/run_default/broadcasts"),
        output_dir=Path("/tmp/eval"),
        wait_step=None,
        base_url=None,
        driver_node_count=0,
        nodes=8,
        patched_verifiers=Path("/tmp/verifiers"),
        omni_env_path=Path("/repo/environments/omni_math2_singleturn"),
        disable_router=False,
        router_policy="round_robin",
        compare_output=Path("/tmp/compare.md"),
        steps=[75, 25],
    )

    rendered = launch._offline_eval_env_script(args, command=["uv", "run", "ok"])

    assert "offline eval weight preflight" in rendered
    assert "outputs/run/run_default/broadcasts/step_25" in rendered
    assert "outputs/run/run_default/broadcasts/step_75" in rendered
    assert "missing requested checkpoint directory" in rendered
    assert "compgen -G" in rendered
