"""Eval harness for the SFT sweep.

Structure:
  - `_server.py`: shared vLLM server boot + OpenAI client (bespoke evals)
  - `run_olmes.py`: wraps olmes CLI (per-invocation isolation via `uv run --with`)
  - `sycophancy.py`: Sharma et al. probes (feedback, are_you_sure, mimicry)
  - `mtbench.py`: MT-Bench 80 prompts + claude-sonnet-4-6 judge
  - `smoke.py`: qualitative base-vs-ckpt comparison (ported from scripts/smoke_eval_sft.py)
  - `run_all.py`: orchestrator

Each eval module exposes a `run(ckpt, output_dir, **kwargs) -> dict` function.
Configs live in `configs/evals/`.
"""
