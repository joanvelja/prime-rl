#!/usr/bin/env bash
set -euo pipefail

root="${PRIME_RL_ROOT:-/lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl}"
run_root="${OFFLINE_EVAL_RUN_ROOT:?OFFLINE_EVAL_RUN_ROOT must point at the run_default directory}"
weights_root="${OFFLINE_EVAL_WEIGHTS_ROOT:-$run_root/broadcasts}"
offline_output="${OFFLINE_EVAL_OUTPUT_DIR:-$(dirname "$run_root")/offline_eval_600x8_all_ckpts}"
arm="${OFFLINE_EVAL_ARM:?OFFLINE_EVAL_ARM must name the eval arm}"
patched_verifiers="${PATCHED_VERIFIERS_PATH:-/lus/lfs1aip2/projects/a6r/joanv.a6r/tmp/verifiers-hf-task-envs}"
omni_env_path="$root/environments/omni_math2_singleturn"

cd "$root"
set -a
source .env
set +a

export PYTHONPATH="$patched_verifiers:$omni_env_path${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/projects/a6r/joanv.a6r/tmp/xdg-cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/projects/a6r/joanv.a6r/tmp/xdg-config}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/projects/a6r/joanv.a6r/tmp/vllm-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache-${SLURM_JOB_ID:-offline-eval}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/inductor-cache-${SLURM_JOB_ID:-offline-eval}}"
export INDUCTOR_CACHE_DIR="${INDUCTOR_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR}"
export VLLM_TORCH_COMPILE_CACHE_DIR="${VLLM_TORCH_COMPILE_CACHE_DIR:-/tmp/vllm-compile-${SLURM_JOB_ID:-offline-eval}}"
export VLLM_NO_USAGE_STATS=1
export PRIME_RL_DISABLE_VLLM_ROUTER="${PRIME_RL_DISABLE_VLLM_ROUTER:-1}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "SLURM_JOB_ID is required for fresh srun_multinode offline eval" >&2
    exit 1
fi

wait_step="${OFFLINE_EVAL_WAIT_STEP:-100}"
wait_path="$weights_root/step_${wait_step}/STABLE"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting for $wait_path"
while [[ ! -f "$wait_path" ]]; do
    sleep 30
done

log_dir="$offline_output/logs"
mkdir -p "$log_dir"
log_path="$log_dir/launcher_$(date -u '+%Y%m%dT%H%M%SZ').log"
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] logging to $log_path"

uv run --no-sync python -m scripts.evals.offline_omni_math2_ckpt_eval \
    --arm "$arm" \
    --run-root "$run_root" \
    --weights-root "$weights_root" \
    --output-dir "$offline_output" \
    --base-model "${OFFLINE_EVAL_BASE_MODEL:-allenai/Olmo-3-7B-Instruct-DPO}" \
    --steps "${OFFLINE_EVAL_STEPS:-25,50,75,100}" \
    --step-interval "${OFFLINE_EVAL_STEP_INTERVAL:-1}" \
    --num-examples "${OFFLINE_EVAL_NUM_EXAMPLES:-600}" \
    --rollouts-per-example "${OFFLINE_EVAL_ROLLOUTS_PER_EXAMPLE:-8}" \
    --max-concurrency "${OFFLINE_EVAL_MAX_CONCURRENCY:-64}" \
    --score-max-concurrency "${OFFLINE_EVAL_SCORE_MAX_CONCURRENCY:-1024}" \
    --max-retries "${OFFLINE_EVAL_MAX_RETRIES:-3}" \
    --ks "${OFFLINE_EVAL_KS:-1,2,3,4,5,6,8}" \
    --launch-mode srun_multinode \
    --launch-nodes "${OFFLINE_EVAL_NODES:-8}" \
    --launch-gpus-per-node "${OFFLINE_EVAL_GPUS_PER_NODE:-4}" \
    --launch-dp "${OFFLINE_EVAL_DP_PER_NODE:-4}" \
    --launch-tp "${OFFLINE_EVAL_TP:-1}" \
    --launch-api-server-count "${OFFLINE_EVAL_API_SERVER_COUNT:-4}" \
    --launch-data-parallel-size-local "${OFFLINE_EVAL_DP_LOCAL:-4}" \
    --launch-port "${OFFLINE_EVAL_PORT:-9800}" \
    --launch-backend-port "${OFFLINE_EVAL_BACKEND_PORT:-9900}" \
    --launch-srun-job-id "$SLURM_JOB_ID" \
    --launch-gpu-memory-utilization "${OFFLINE_EVAL_GPU_MEMORY_UTILIZATION:-0.95}" \
    --launch-max-model-len "${OFFLINE_EVAL_MAX_MODEL_LEN:-16384}" \
    --launch-max-num-seqs "${OFFLINE_EVAL_MAX_NUM_SEQS:-192}" \
    --launch-max-num-batched-tokens "${OFFLINE_EVAL_MAX_NUM_BATCHED_TOKENS:-65536}" \
    2>&1 | tee "$log_path"

status=${PIPESTATUS[0]}
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] offline eval exited with status $status"
echo "log: $log_path"
exit "$status"
