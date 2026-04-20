#!/bin/bash
#SBATCH --job-name=zebra-probe
#SBATCH --partition=workq
#SBATCH --gpus=2
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Zebra-puzzles-v2 headroom probe on Isambard-AI Phase 2 (a6r).
# Budget: 2 GPU × 30 min = 1 GPU-h ≈ 0.002% of 50,000 GPU-h allocation.

set -euo pipefail

module load PrgEnv-gnu

export HF_HOME="${PROJECTDIR:-/projects/a6r}/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

# Use the existing prime-rl venv (uv sync'd in the repo).
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# Sanity: confirm vllm + pyarrow are importable.
python -c "import vllm, pyarrow, transformers" || {
    echo "ERROR: missing deps in .venv — run \`uv sync\` then retry."
    exit 1
}

# Dataset pre-staged from login node via:
#   hf download joanvelja/zebra-puzzles-v2 \
#     data/train_canonical-00000-of-00002.parquet \
#     --repo-type dataset --local-dir $PROJECTDIR/datasets/zebra-puzzles-v2
DATA_PATH="${PROJECTDIR:-/projects/a6r}/datasets/zebra-puzzles-v2/data/train_canonical-00000-of-00002.parquet"
if [[ ! -f "$DATA_PATH" ]]; then
    echo "ERROR: parquet not pre-staged at $DATA_PATH"
    echo "Run on login node:"
    echo "  hf download joanvelja/zebra-puzzles-v2 \\"
    echo "    data/train_canonical-00000-of-00002.parquet \\"
    echo "    --repo-type dataset \\"
    echo "    --local-dir \$PROJECTDIR/datasets/zebra-puzzles-v2"
    exit 1
fi

python tmp/zebra_probe/probe_pass_at_n.py --parquet "$DATA_PATH"
