#!/usr/bin/env bash
# Build and install PrimeIntellect-ai/router (vLLM router) for linux-aarch64.
#
# PyPI ships only x86_64 wheels, so on aarch64 we build from source. This
# requires a Rust toolchain (setuptools-rust) and the protobuf compiler.
# When protoc is not in PATH the script fetches a pinned precompiled binary
# from protobuf releases into a build-local prefix.
#
# Usage:
#   bash scripts/install_router_aarch64.sh
#
# Options:
#   --ref REF         router git ref (default: v0.1.22)
#   --wheel-dir DIR   output wheel to DIR instead of installing
#   --keep-build      keep the temp build directory (default: cleanup)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

ROUTER_GIT_REPO="https://github.com/PrimeIntellect-ai/router.git"
ROUTER_GIT_REF="v0.1.22"
PROTOC_VERSION="28.3"
WHEEL_DIR=""
KEEP_BUILD=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)        ROUTER_GIT_REF="$2"; shift 2 ;;
        --wheel-dir)  WHEEL_DIR="$2";      shift 2 ;;
        --keep-build) KEEP_BUILD=1;        shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" ]]; then
    echo "ERROR: this script targets aarch64 (got $ARCH)." >&2
    echo "On x86_64, install the upstream wheel: 'uv pip install vllm-router'." >&2
    exit 1
fi

# Skip if already installed at the right version. ROUTER_GIT_REF can be a
# version tag ('v0.1.22') or a commit hash; for tags, match the leading 'v'.
TARGET_VER="${ROUTER_GIT_REF#v}"
INSTALLED_VER="$(uv pip show vllm-router 2>/dev/null | awk '/^Version:/{print $2}' || true)"
if [[ -n "$INSTALLED_VER" && "$INSTALLED_VER" == "$TARGET_VER" ]]; then
    echo "vllm-router ${INSTALLED_VER} already installed, skipping."
    exit 0
fi

# Pre-flight: rust toolchain
if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo not found. Install rust toolchain first:" >&2
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh" >&2
    echo "  source \$HOME/.cargo/env" >&2
    exit 1
fi

echo "================================================================"
echo " Building vllm-router (${ROUTER_GIT_REF})"
echo " arch: ${ARCH}, cargo: $(cargo --version)"
echo "================================================================"

BUILD_DIR="$(mktemp -d)"
if [[ $KEEP_BUILD -eq 0 ]]; then
    trap 'rm -rf "$BUILD_DIR"' EXIT
else
    echo "Build artifacts will be kept at: $BUILD_DIR"
fi

# Ensure protoc is available. The router's setuptools-rust build step shells
# out to protoc to compile the bundled .proto files.
if ! command -v protoc >/dev/null 2>&1; then
    echo "protoc not found; fetching v${PROTOC_VERSION} precompiled for linux-aarch_64..."
    PROTOC_DIR="$BUILD_DIR/protoc"
    PROTOC_ZIP="$BUILD_DIR/protoc.zip"
    PROTOC_URL="https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-aarch_64.zip"
    mkdir -p "$PROTOC_DIR"
    curl -fsSL -o "$PROTOC_ZIP" "$PROTOC_URL"
    (cd "$PROTOC_DIR" && unzip -q "$PROTOC_ZIP")
    export PATH="$PROTOC_DIR/bin:$PATH"
    echo "  using protoc: $(protoc --version)"
else
    echo "  using protoc: $(protoc --version)"
fi

git clone "$ROUTER_GIT_REPO" "$BUILD_DIR/router"
git -C "$BUILD_DIR/router" checkout "$ROUTER_GIT_REF"

cd "$REPO_ROOT"
if [[ -n "$WHEEL_DIR" ]]; then
    mkdir -p "$WHEEL_DIR"
    uv build --no-build-isolation --wheel --out-dir "$WHEEL_DIR" "$BUILD_DIR/router"
    echo ""
    echo "Wheel built:"
    ls -lh "$WHEEL_DIR"/vllm_router-*.whl
else
    uv pip install --no-build-isolation --reinstall "$BUILD_DIR/router"
    echo ""
    echo "Installed: $(uv pip show vllm-router | awk '/^Version:/{print $2}')"
fi

echo "================================================================"
echo " vllm-router build complete"
echo "================================================================"
