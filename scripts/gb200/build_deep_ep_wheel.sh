#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd git

repo_dir="${WORKDIR_ROOT}/DeepEP"
nvshmem_dir="$(download_nvshmem)"

clone_checkout "${DEEPEP_REPO_URL}" "${repo_dir}" "${DEEPEP_REF}"

log "Applying GB200 DeepEP build-time edits"
(
    cd "${repo_dir}"
    sed -i 's/use_fabric: bool = False/use_fabric: bool = True/' deep_ep/buffer.py || true
    sed -i 's/#define NUM_CPU_TIMEOUT_SECS 100/#define NUM_CPU_TIMEOUT_SECS 10000/' csrc/kernels/configs.cuh
    sed -i 's/#define NUM_CPU_TIMEOUT_SECS 10/#define NUM_CPU_TIMEOUT_SECS 1000/' csrc/kernels/configs.cuh
    sed -i 's/#define NUM_TIMEOUT_CYCLES 200000000000ull/#define NUM_TIMEOUT_CYCLES 20000000000000ull/' csrc/kernels/configs.cuh
    sed -i 's/#define NUM_TIMEOUT_CYCLES 20000000000ull/#define NUM_TIMEOUT_CYCLES 2000000000000ull/' csrc/kernels/configs.cuh
)

log "Building DeepEP wheel from ${DEEPEP_REPO_URL}@${DEEPEP_REF}"
(
    cd "${repo_dir}"
    export NVSHMEM_DIR="${nvshmem_dir}"
    export CMAKE_PREFIX_PATH="${nvshmem_dir}/lib/cmake:${CMAKE_PREFIX_PATH:-}"
    "${PYTHON_BIN}" -m pip wheel . --no-deps --wheel-dir "${ARTIFACT_DIR}" --no-build-isolation
)
