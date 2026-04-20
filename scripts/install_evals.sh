#!/usr/bin/env bash
# Install olmes (Ai2's eval harness) + lm-eval + leaf deps into prime-rl's venv.
#
# Strategy: olmes pins old transformers (4.57), torch (2.8), vllm (0.11),
# numpy (<2), datasets (<4). Those conflict with prime-rl's core. However,
# olmes's runtime code is compatible with our newer versions. We install
# olmes + lm-eval with `--no-deps` so pip doesn't try to satisfy the old pins,
# then fill in the leaf deps olmes actually needs at runtime.
#
# We also patch oe_eval/utilities/model_utils.py to lazy-import OlmoCoreLM
# (avoids pulling the full ai2-olmo-core transitive tree; we never use the
# olmo_core model-type).
#
# Idempotent: safe to re-run after `uv sync --all-extras`.

set -euo pipefail

cd "$(dirname "$0")/.."

OLMES_REV="${OLMES_REV:-5a51f502}"
OLMES_DEP="ai2-olmes @ git+https://github.com/allenai/olmes.git@${OLMES_REV}"

log() { echo -e "\033[0;32m[install_evals]\033[0m $*"; }

log "Installing olmes at rev ${OLMES_REV} (--no-deps)..."
uv pip install --no-deps "${OLMES_DEP}"

log "Installing lm-eval (--no-deps)..."
# Pinned to olmes's 0.4.3 pin. Newer lm_eval versions have API drift that
# break oe_eval's internal imports (pad_and_concat, eval_logger, etc.).
# Instead we stay on 0.4.3 and apply two surgical patches below for vllm>=0.11:
#   - replace LLM.generate(prompt_token_ids=...) with TokensPrompt list
#   - coerce trust_remote_code to bool (vllm>=0.19 pydantic requires it)
uv pip install --no-deps "lm_eval==0.4.3"

log "Installing ai2-olmo-core (--no-deps; required for import chain but never executed)..."
uv pip install --no-deps "ai2-olmo-core"

log "Installing leaf runtime deps olmes actually calls..."
uv pip install \
    sacrebleu \
    portalocker \
    sqlitedict \
    pybind11 \
    rouge-score \
    zstandard \
    more-itertools \
    pytablewriter \
    evaluate \
    jsonlines \
    colorama \
    mecab-python3 \
    unidic-lite \
    lxml \
    scikit-learn \
    peft \
    alpaca-eval \
    langdetect \
    nltk \
    immutabledict \
    boto3 \
    gradio-client \
    tree-sitter-python \
    tree-sitter-languages \
    cached-path \
    omegaconf \
    "antlr4-python3-runtime==4.11" \
    litellm \
    smart-open \
    pytrec-eval \
    pygsheets \
    ray \
    sympy \
    math-verify

# Patch oe_eval to lazy-import olmo_core model backend (import would chain in
# bettermap, beaker, etc. transitively). Idempotent: re-runs are no-ops.
log "Patching oe_eval for lazy olmo_core import..."
MODEL_UTILS=$(uv run python -c 'import oe_eval.utilities.model_utils as m; print(m.__file__)')
if ! grep -q "LOCAL PATCH" "${MODEL_UTILS}"; then
    python3 - <<PYEOF
from pathlib import Path
p = Path("${MODEL_UTILS}")
src = p.read_text()
needle = "from oe_eval.models.eleuther_olmo_core import OlmoCoreLM"
replacement = '''# LOCAL PATCH (install_evals.sh): lazy-import OlmoCoreLM to avoid pulling the full
# ai2-olmo-core import chain at oe_eval load time. Only material when
# model_type == "olmo_core"; we use hf / vllm / litellm instead.
try:
    from oe_eval.models.eleuther_olmo_core import OlmoCoreLM
except Exception as _olmo_core_import_err:  # omegaconf ATN mismatch raises plain Exception
    class OlmoCoreLM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "OlmoCoreLM unavailable in this venv "
                f"(original error: {_olmo_core_import_err}). "
                "Use --model-type hf, vllm, or litellm."
            )'''
assert needle in src, f"Expected import line missing from {p}"
p.write_text(src.replace(needle, replacement, 1))
PYEOF
    log "Patched ${MODEL_UTILS}"
else
    log "Already patched; skipping."
fi

log "Patching lm_eval vllm backend for vllm>=0.11/0.19 (prompt_token_ids → TokensPrompt, trust_remote_code coercion)..."
VLLM_CL_FILE=$(uv run --no-sync python -c 'import lm_eval.models.vllm_causallms as m; print(m.__file__)')
if ! grep -q "LOCAL PATCH" "${VLLM_CL_FILE}"; then
    python3 - <<PYEOF
from pathlib import Path
p = Path("${VLLM_CL_FILE}")
src = p.read_text()
# Patch 1: generate(prompt_token_ids=requests) → TokensPrompt list
old1 = '''        if self.lora_request is not None:
            outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
        else:
            outputs = self.model.generate(
                prompt_token_ids=requests,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
            )'''
new1 = '''        # LOCAL PATCH (install_evals.sh): vllm>=0.11 requires TokensPrompt objects.
        from vllm import TokensPrompt as _TokensPrompt
        _prompts = [_TokensPrompt(prompt_token_ids=r) for r in requests]
        if self.lora_request is not None:
            outputs = self.model.generate(
                prompts=_prompts,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
                lora_request=self.lora_request,
            )
        else:
            outputs = self.model.generate(
                prompts=_prompts,
                sampling_params=sampling_params,
                use_tqdm=True if self.batch_size == "auto" else False,
            )'''
assert old1 in src, "vllm_causallms.py: generate() call block not matched"
src = src.replace(old1, new1, 1)
# Patch 2: trust_remote_code must be bool (not None) under vllm>=0.19 pydantic
old2 = '"trust_remote_code": trust_remote_code,'
new2 = '"trust_remote_code": bool(trust_remote_code),  # LOCAL PATCH: vllm>=0.19 requires bool'
assert old2 in src, "vllm_causallms.py: trust_remote_code line not matched"
src = src.replace(old2, new2, 1)
p.write_text(src)
PYEOF
    log "Patched ${VLLM_CL_FILE}"
else
    log "vllm_causallms already patched; skipping."
fi

log "Verifying imports..."
uv run python -c "
from oe_eval.tasks.oe_eval_tasks.mmlu import GenericMMLU_MC
from oe_eval.tasks.oe_eval_tasks.ifeval import IFEval
from oe_eval.metrics.metric import IFEvalMetric
from oe_eval.models.eleuther_huggingface import HFLM_Verbose
from oe_eval.models.eleuther_vllm_causallms import VLLM_Verbose
import anthropic
print('olmes + prime-rl coexist cleanly in this venv')
" 2>&1 | tail -3

log "Done."
