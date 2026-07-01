from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATED = ROOT / "configs" / "debate" / "generated"
CALIBRATION = ROOT / "configs" / "calibration"
GPQA_OE_JUDGE_PROMPT = ROOT / "deps" / "verifiers" / "verifiers" / "utils" / "judge_prompts" / "gpqa_oe.yaml"


def test_generated_debate_envs_wire_gt_grader() -> None:
    for path in sorted(GENERATED.glob("*.toml")):
        config = tomllib.loads(path.read_text())
        train_envs = config["orchestrator"]["train"]["env"]
        eval_envs = config["orchestrator"]["eval"]["env"]
        debate_envs = [env for env in [*train_envs, *eval_envs] if env["id"] == "gpqa-open-ended-debate"]
        assert debate_envs, path

        for env in debate_envs:
            args = env["args"]
            assert args["judge_base_url"] == "https://openrouter.ai/api/v1", path
            assert args["judge_model"] == "deepseek/deepseek-v4-flash", path
            assert args["judge_api_key_var"] == "OPENROUTER_API_KEY", path


def test_generated_single_agent_eval_uses_deepseek_grader() -> None:
    # The grader sampling/provider policy lives in the shared gpqa_oe prompt pack
    # (single source of truth); the eval references it via ``judge_prompt_pack``.
    # The policy itself is asserted in
    # ``test_debate_prompt_pack_uses_same_deepseek_sampling_policy``.
    for path in sorted(GENERATED.glob("*.toml")):
        config = tomllib.loads(path.read_text())
        (single_eval,) = [env for env in config["orchestrator"]["eval"]["env"] if env["id"] == "hf-singleturn"]
        args = single_eval["args"]

        assert args["judge_prompt_pack"] == "gpqa_oe", path
        assert args["judge_model"] == "deepseek/deepseek-v4-flash", path
        assert args["judge_base_url"] == "https://openrouter.ai/api/v1", path
        assert args["judge_api_key_var"] == "OPENROUTER_API_KEY", path


def test_generated_debate_configs_use_64k_context_budget() -> None:
    for path in sorted(GENERATED.glob("*.toml")):
        config = tomllib.loads(path.read_text())
        train_sampling = config["orchestrator"]["train"]["sampling"]

        assert config["seq_len"] == 65536, path
        assert config["trainer"]["model"]["seq_len"] == 65536, path
        assert config["inference"]["model"]["max_model_len"] == 65536, path
        assert config["seq_len"] - train_sampling["max_completion_tokens"] >= 49152, path


def test_generated_debate_configs_use_filesystem_lora_broadcast_on_scratch() -> None:
    for path in sorted(GENERATED.glob("*.toml")):
        config = tomllib.loads(path.read_text())

        assert config["weight_broadcast"]["type"] == "filesystem", path
        assert config["output_dir"].startswith("/scratch/a6r/joanv.a6r/outputs/isambard/calibration/debate_"), path


def test_calibration_gpqa_debate_configs_use_64k_context_budget() -> None:
    paths = [
        *CALIBRATION.glob("gpqa*debate*.toml"),
        *CALIBRATION.glob("topology_*.toml"),
    ]
    for path in sorted(paths):
        config = tomllib.loads(path.read_text())
        train_sampling = config["orchestrator"]["train"]["sampling"]

        assert config["seq_len"] == 65536, path
        assert config["trainer"]["model"]["seq_len"] == 65536, path
        assert config["inference"]["model"]["max_model_len"] == 65536, path
        assert config["seq_len"] - train_sampling["max_completion_tokens"] >= 49152, path


def test_debate_prompt_pack_uses_same_deepseek_sampling_policy() -> None:
    text = GPQA_OE_JUDGE_PROMPT.read_text()

    assert text.count("model: deepseek/deepseek-v4-flash") == 2
    assert text.count("max_completion_tokens: 8192") == 2
    assert text.count("effort: high") == 2
    assert text.count("exclude: true") == 2
    # ZDR-constrained provider pool with fallbacks. zdr:true + data_collection:deny
    # filter every provider in the pool, so the privacy guarantee holds; fallbacks
    # stop one dead provider from failing every grade.
    assert text.count("allow_fallbacks: true") == 2
    # Both provider blocks omit require_parameters: combined with zdr + fp8 + a
    # pinned order it leaves no eligible OpenRouter route -> ModelError -> silent
    # 0.0 grade.
    assert text.count("require_parameters: true") == 0
    assert text.count("zdr: true") == 2
    assert text.count("data_collection: deny") == 2
    assert text.count("- AtlasCloud") == 2
