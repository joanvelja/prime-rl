#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd git

repo_dir="${WORKDIR_ROOT}/nixl"
ucx_install="$(ensure_ucx)"

clone_checkout "https://github.com/ai-dynamo/nixl.git" "${repo_dir}" "${NIXL_GIT_REF}"

log "Building NIXL wheel from ${NIXL_GIT_REF}"
(
    cd "${repo_dir}"
    export PKG_CONFIG_PATH="${ucx_install}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="${ucx_install}/lib:${ucx_install}/lib/ucx:${LD_LIBRARY_PATH:-}"
    export MESONARGS="-Ducx_path=${ucx_install}"
    "${PYTHON_BIN}" -m pip wheel . --no-deps --wheel-dir "${ARTIFACT_DIR}"
)

log "Downloading pinned Python shim nixl==${NIXL_PYTHON_VERSION}"
"${PYTHON_BIN}" -m pip download --no-deps --dest "${ARTIFACT_DIR}" "nixl==${NIXL_PYTHON_VERSION}"
