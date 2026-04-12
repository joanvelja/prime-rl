#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd git
"${PYTHON_BIN}" -m pip --version >/dev/null

repo_dir="${WORKDIR_ROOT}/deepgemm"
clone_checkout "https://github.com/deepseek-ai/DeepGEMM.git" "${repo_dir}" "${DEEPGEMM_COMMIT}" true

log "Building DeepGEMM wheel from ${DEEPGEMM_COMMIT}"
(
    cd "${repo_dir}"
    "${PYTHON_BIN}" -m pip wheel . --no-deps --wheel-dir "${ARTIFACT_DIR}" --no-build-isolation
)
