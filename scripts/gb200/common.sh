#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_ROOT="${PATCH_ROOT:-$(cd "${SCRIPT_DIR}/../../patches/gb200" && pwd)}"

ARTIFACT_DIR="${ARTIFACT_DIR:-/out}"
WORKDIR_ROOT="${WORKDIR_ROOT:-/tmp/gb200-build}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8.1}"
CUDA_RUN_FILENAME="${CUDA_RUN_FILENAME:-cuda_12.8.1_570.124.06}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-477618c}"
DEEPEP_REPO_URL="${DEEPEP_REPO_URL:-https://github.com/tlrmchlsmth/DeepEP.git}"
DEEPEP_REF="${DEEPEP_REF:-sgl-gb200-blog-pt2}"
UCX_VERSION="${UCX_VERSION:-v1.19.x}"
NVSHMEM_VERSION="${NVSHMEM_VERSION:-3.3.24}"
NIXL_GIT_REF="${NIXL_GIT_REF:-v1.0.0}"
NIXL_PYTHON_VERSION="${NIXL_PYTHON_VERSION:-0.10.1}"
PATCHSET_REV="${PATCHSET_REV:-gb200.1}"
RELEASE_VERSION="${RELEASE_VERSION:-dev}"

log() {
    printf '[gb200] %s\n' "$*" >&2
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Required command not found: $1" >&2
        exit 1
    }
}

make_clean_dir() {
    rm -rf "$1"
    mkdir -p "$1"
}

clone_checkout() {
    local repo_url="$1"
    local target_dir="$2"
    local ref="$3"
    local recursive="${4:-false}"

    make_clean_dir "${target_dir}"
    if [ "${recursive}" = "true" ]; then
        git clone --recursive --shallow-submodules "${repo_url}" "${target_dir}"
    else
        git clone "${repo_url}" "${target_dir}"
    fi
    git -C "${target_dir}" checkout "${ref}"
    if [ "${recursive}" = "true" ]; then
        git -C "${target_dir}" submodule update --init --recursive --depth 1
    fi
}

nvshmem_subdir() {
    case "$(uname -m)" in
        x86_64|amd64)
            echo "linux-x86_64"
            ;;
        aarch64|arm64)
            echo "linux-sbsa"
            ;;
        *)
            echo "Unsupported architecture for NVSHMEM: $(uname -m)" >&2
            exit 1
            ;;
    esac
}

download_nvshmem() {
    local install_dir="${WORKDIR_ROOT}/nvshmem"
    local subdir file archive url

    if [ -d "${install_dir}/lib" ]; then
        echo "${install_dir}"
        return
    fi

    subdir="$(nvshmem_subdir)"
    file="libnvshmem-${subdir}-${NVSHMEM_VERSION}_cuda12-archive.tar.xz"
    archive="${WORKDIR_ROOT}/${file}"
    url="https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/${subdir}/${file}"

    log "Downloading NVSHMEM ${NVSHMEM_VERSION}"
    mkdir -p "${WORKDIR_ROOT}"
    curl -fsSL "${url}" -o "${archive}"
    tar -xf "${archive}" -C "${WORKDIR_ROOT}"
    mv "${WORKDIR_ROOT}/${file%.tar.xz}" "${install_dir}"
    rm -f "${archive}"
    echo "${install_dir}"
}

ensure_ucx() {
    local ucx_src="${WORKDIR_ROOT}/ucx_source"
    local ucx_install="${WORKDIR_ROOT}/ucx-install"

    if [ -f "${ucx_install}/lib/libucs.so" ]; then
        echo "${ucx_install}"
        return
    fi

    log "Building UCX ${UCX_VERSION}"
    clone_checkout "https://github.com/openucx/ucx.git" "${ucx_src}" "${UCX_VERSION}"
    (
        cd "${ucx_src}"
        ./autogen.sh
        ./configure \
            --prefix="${ucx_install}" \
            --enable-shared \
            --disable-static \
            --disable-doxygen-doc \
            --enable-optimizations \
            --enable-cma \
            --enable-devel-headers \
            --enable-mt \
            --with-verbs \
            --with-cuda="${CUDA_HOME}" \
            --with-ze=no
        make -j"$(nproc)"
        make install
    )
    echo "${ucx_install}"
}
