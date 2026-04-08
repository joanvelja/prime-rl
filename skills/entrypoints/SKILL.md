---
name: entrypoints
description: All available prime-rl entrypoints — what they do, how to launch them, and which config class they use. Use when running commands, launching training, or starting servers.
---

# Entrypoints

All entrypoints are run via `uv run <command>` and accept TOML configs via `@ path/to/config.toml` with CLI overrides. See the `config` skill for config system details.

## `rl` — RL training

Orchestrates the complete RL loop: launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/reverse_text/rl.toml
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml # with SLURM
uv run rl @ examples/reverse_text/rl.toml --dry-run # generate scripts without running
```

When running from a fresh worktree, include optional extras required by the config. For example, VLM color-codeword debugging needs FlashAttention and the env packages:

```bash
uv run --extra flash-attn --extra envs rl @ configs/multimodal/rl_color_codeword_test.toml
```

Short debug runs can write large filesystem weight-broadcast snapshots. If `/` is low on space, put the debug `--output-dir` on `/dev/shm` and clean it up afterward.

For hard-distillation with `orchestrator.teacher_rollout_model`, omit `[inference]`. That mode talks directly to the external OpenAI-compatible rollout endpoint and reconstructs training tokens from messages; it does not use the local renderer proxy or `/v1/generate`.

For LoRA RL smoke runs against local vLLM, set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` in the environment or adapter load/update requests will fail.

For ad-hoc `CliAgentEnv`/`ComposableEnv` sandbox smoke tests that call back through Prime Tunnel using Python stdlib `urllib`, set an explicit non-default `User-Agent` header such as `curl/7.81.0`. The tunnel can reject Python's default `urllib` user agent from inside Prime sandboxes with HTTP 403 / code 1010. Prefer uploading an agent script into the sandbox and running `python3 /tmp/agent.py` over passing a heredoc directly as a `start_background_job` command.

- **Config:** `RLConfig` (`src/prime_rl/configs/rl.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/rl.py`
- **SLURM:** yes — single-node and multi-node

## `sft` — SFT training

Trains a model on labeled data. Uses torchrun for distributed training.

```bash
uv run sft @ examples/reverse_text/sft.toml
uv run sft @ examples/reverse_text/sft.toml --slurm # with SLURM
uv run sft @ examples/reverse_text/sft.toml --dry-run # generate scripts without running
```

The entrypoint launches torchrun internally — no need to call torchrun directly.

- **Config:** `SFTConfig` (`src/prime_rl/configs/sft.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/sft.py`
- **SLURM:** yes — single-node and multi-node

## `inference` — Standalone inference server

Launches a vLLM-based inference server with OpenAI-compatible API.

```bash
uv run inference @ configs/debug/infer.toml
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Always use the `inference` entrypoint — never `vllm serve` directly.

Custom endpoints beyond standard OpenAI API:
- `/v1/generate` — accepts token IDs as prompt input, with optional images
- `/update_weights` — hot-reload model weights from the trainer
- `/load_lora_adapter` — load LoRA adapters at runtime
- `/init_broadcaster` — initialize weight broadcast for distributed training

Check health with:
```bash
curl http://<ip>:<port>/health
```

Check served models with:
```bash
curl http://<ip>:<port>/v1/models
```

Test chat completions with:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

- **Config:** `InferenceConfig` (`src/prime_rl/configs/inference.py`)
- **Entrypoint:** `src/prime_rl/entrypoints/inference.py`
- **SLURM:** yes — single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | SLURM | Typical use |
|---------|---------|-------|-------------|
| `rl` | Full RL pipeline | yes | Production RL training |
| `sft` | Supervised fine-tuning | yes | SFT training |
| `inference` | vLLM server | yes | Standalone inference or debugging |

## Key directories

- `src/prime_rl/entrypoints/` — top-level entrypoints (`rl`, `sft`, `inference`)
- `src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal configs for quick testing
- `examples/` — full example configs for various tasks
