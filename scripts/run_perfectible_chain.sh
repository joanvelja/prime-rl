#!/usr/bin/env bash
# Chain the 3 perfectible-subset full runs sequentially on the same allocation.
# Each run owns the inference HOSTS (7 nodes — head excluded as orchestrator).
# Partial rollouts persist to raw_rollouts.partial.jsonl in the output dir;
# on clean exit it's unlinked.
set -uo pipefail
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl
source .env 2>/dev/null

# Required until the sbatch wrapper is rewritten to use salloc+use_interactive_step
# or --het-job. See memory: isambard_srun_pty_anti_pattern.md
export PRIME_RL_EXCLUDE_LOCAL_FROM_HOSTS=1

LOG=outputs/baselines/_chain-$(date +%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

run_one() {
    local config="$1"
    local outdir="$2"
    local label="$3"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') [$label] starting full run ==="
    uv run --no-sync python -m prime_rl.baselines.cli \
        --config "$config" \
        --output-dir "$outdir"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') [$label] exit=$? ==="
}

run_one configs/baselines/omni_math2_perfectible_gemma4_e4b_fp8kv.toml \
        outputs/baselines/omni-math2-gemma4-e4b-perfectible-40r \
        "1/3 gemma-4-E4B"

run_one configs/baselines/omni_math2_perfectible_rnj1_fp8kv.toml \
        outputs/baselines/omni-math2-rnj1-instruct-perfectible-40r \
        "2/3 rnj-1-instruct"

run_one configs/baselines/omni_math2_perfectible_gemma4_26b_a4b_fp8kv.toml \
        outputs/baselines/omni-math2-gemma4-26b-a4b-perfectible-40r \
        "3/3 gemma-4-26B-A4B"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') chain complete ==="
