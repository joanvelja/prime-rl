#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd git
require_cmd patch

repo_dir="${WORKDIR_ROOT}/DeepEP"
nvshmem_dir="$(download_nvshmem)"

clone_checkout "https://github.com/deepseek-ai/DeepEP.git" "${repo_dir}" "${DEEPEP_COMMIT}"

log "Applying GB200 DeepEP patch set"
patch -p1 -d "${repo_dir}" < "${PATCH_ROOT}/deep_ep-gb200.patch"

log "Building DeepEP wheel from ${DEEPEP_COMMIT}"
(
    cd "${repo_dir}"
    export NVSHMEM_DIR="${nvshmem_dir}"
    export CMAKE_PREFIX_PATH="${nvshmem_dir}/lib/cmake:${CMAKE_PREFIX_PATH:-}"
    "${PYTHON_BIN}" -m pip wheel . --no-deps --wheel-dir "${ARTIFACT_DIR}" --no-build-isolation
)
