#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

mkdir -p "${ARTIFACT_DIR}" "${WORKDIR_ROOT}"

log "Installing wheel build helpers"
"${PYTHON_BIN}" -m pip install --upgrade \
    pip \
    build \
    wheel \
    setuptools \
    packaging \
    meson \
    meson-python \
    pybind11 \
    patchelf \
    pyyaml

"${SCRIPT_DIR}/build_deep_gemm_wheel.sh"
"${SCRIPT_DIR}/build_deep_ep_wheel.sh"
"${SCRIPT_DIR}/build_nixl_wheel.sh"

log "Writing GB200 release manifest"
"${PYTHON_BIN}" "${SCRIPT_DIR}/write_release_manifest.py" \
    --artifact-dir "${ARTIFACT_DIR}" \
    --release-version "${RELEASE_VERSION}" \
    --patchset-rev "${PATCHSET_REV}" \
    --vllm-version "${VLLM_VERSION}" \
    --deepgemm-commit "${DEEPGEMM_COMMIT}" \
    --deepep-source "${DEEPEP_REPO_URL}@${DEEPEP_REF}" \
    --ucx-version "${UCX_VERSION}" \
    --nvshmem-version "${NVSHMEM_VERSION}" \
    --nixl-git-ref "${NIXL_GIT_REF}" \
    --nixl-python-version "${NIXL_PYTHON_VERSION}" \
    --cuda-toolkit-version "${CUDA_TOOLKIT_VERSION}" \
    --torch-cuda-arch-list "${TORCH_CUDA_ARCH_LIST}" \
    --omitted-patch "vllm/distributed/device_communicators/all2all.py DeepEP HT/MNNVL override intentionally not baked yet"

log "Wheelhouse contents"
ls -lh "${ARTIFACT_DIR}"
