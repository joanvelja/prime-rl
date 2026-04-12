# GB200 Build Runbook

This runbook captures the current reproducible GB200 image state:

- arm64 wheels published to GitHub Releases
- standalone GB200 runtime image published to GHCR
- pinned inputs for building the final full image

It documents the intended path first. A short troubleshooting note at the end captures the Docker client push issue we hit locally.

## Current Published Artifacts

Wheel release:

- https://github.com/S1ro1/tmp-wheels/releases/tag/tmp-wheels-20260412-223722

Published wheel assets:

- `deep_gemm-2.3.0+477618c-cp312-cp312-linux_aarch64.whl`
- `deep_ep-1.2.1+9249c25-cp312-cp312-linux_aarch64.whl`
- `nixl_cu12-1.0.0-cp312-cp312-linux_aarch64.whl`
- `nixl-0.10.1-py3-none-any.whl`
- `gb200-release-manifest.json`
- `build.log`

Published runtime image:

- `ghcr.io/s1ro1/prime-rl-gb200-runtime:tmp-wheels-20260412-223722`
- manifest digest: `sha256:c4efd20cac5e861f46834e4b5d2b8b146884dd3ab3fadde6758cb98ea480b9d6`

## Inputs

Base Prime-RL image:

- `ghcr.io/primeintellect-ai/pi-rft/prime-rl:v0.5.1.dev38`

Pinned source/runtime versions:

- DeepGEMM: `477618c`
- DeepEP: `9249c25`
- UCX: `v1.19.x`
- NVSHMEM: `3.3.24`
- NIXL source ref: `v1.0.0`
- NIXL Python shim: `0.10.1`
- CUDA toolkit: `12.8.1`
- target arch: `linux/arm64`
- target SM: `10.0`

## 1. Build Wheels On GB200 Kubernetes

The wheel build requires a GB200-compatible arm64 environment.

Source inputs used by the job:

- [`scripts/gb200`](../scripts/gb200)
- [`patches/gb200`](../patches/gb200)

Builder pod shape:

- image: `ghcr.io/primeintellect-ai/pi-rft/prime-rl:v0.5.1.dev38-gb200`
- namespace: `dev-matej`
- node: a free GB200 worker
- GPU: `1`

The important environment/bootstrap bits:

```bash
export VIRTUAL_ENV=/app/.venv
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/app/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=10.0
export PYTHON_BIN=/app/.venv/bin/python
```

The job also needs:

- CUDA 12.8 toolkit installed from NVIDIA runfile
- RDMA/UCX build packages installed with `apt`
- [`patch_cuda_math_header.py`](../scripts/gb200/patch_cuda_math_header.py) applied to CUDA 12.8 on arm64 before DeepEP build

Build command inside the pod:

```bash
bash /workspace/scripts/gb200/build-wheelhouse.sh
```

Expected outputs in `/out`:

- wheel files
- `gb200-release-manifest.json`
- `build.log`
- `status` containing `BUILD_OK`

## 2. Publish Wheels To GitHub Releases

Create a release repo if needed:

```bash
gh repo create S1ro1/tmp-wheels --private
```

Initialize the repo once so releases are allowed:

```bash
gh repo clone S1ro1/tmp-wheels /tmp/tmp-wheels-repo
cd /tmp/tmp-wheels-repo
printf '# tmp-wheels\n\nTemporary GB200 wheel artifacts.\n' > README.md
git add README.md
git commit -m 'chore: initialize repo'
git push origin HEAD:main
```

Create the release:

```bash
gh release create tmp-wheels-20260412-223722 \
  --repo S1ro1/tmp-wheels \
  --target main \
  --title 'tmp-wheels-20260412-223722' \
  --notes 'GB200 arm64 wheel build from Kubernetes. Includes DeepEP, DeepGEMM, NIXL wheels and build manifest. all2all patch intentionally not baked.'
```

Upload assets:

```bash
gh release upload tmp-wheels-20260412-223722 \
  /path/to/deep_ep-1.2.1+9249c25-cp312-cp312-linux_aarch64.whl \
  /path/to/deep_gemm-2.3.0+477618c-cp312-cp312-linux_aarch64.whl \
  /path/to/nixl-0.10.1-py3-none-any.whl \
  /path/to/nixl_cu12-1.0.0-cp312-cp312-linux_aarch64.whl \
  /path/to/gb200-release-manifest.json \
  /path/to/build.log \
  --repo S1ro1/tmp-wheels
```

## 3. Build And Push The Runtime Image

Runtime image source:

- [`Dockerfile.gb200-runtime`](../Dockerfile.gb200-runtime)

Intended push path:

```bash
export GH_TOKEN=...
printf '%s' "$GH_TOKEN" | docker login ghcr.io -u S1ro1 --password-stdin

docker buildx build \
  --platform linux/arm64 \
  --file Dockerfile.gb200-runtime \
  --tag ghcr.io/s1ro1/prime-rl-gb200-runtime:tmp-wheels-20260412-223722 \
  --push .
```

What this image contains:

- UCX built from source into `/opt/ucx`
- NVSHMEM unpacked into `/opt/nvshmem`
- runtime env:
  - `PATH=/opt/ucx/bin:${PATH}`
  - `LD_LIBRARY_PATH=/opt/ucx/lib:/opt/ucx/lib/ucx:/opt/nvshmem/lib`
  - `NVSHMEM_DIR=/opt/nvshmem`

## 4. Build The Final Full Image

Full image source:

- [`Dockerfile.gb200`](../Dockerfile.gb200)

Inputs:

- base image: `ghcr.io/primeintellect-ai/pi-rft/prime-rl:v0.5.1.dev38`
- runtime image: `ghcr.io/s1ro1/prime-rl-gb200-runtime:tmp-wheels-20260412-223722`
- wheel base URL: `https://github.com/S1ro1/tmp-wheels/releases/download/tmp-wheels-20260412-223722`

Build command:

```bash
docker buildx build \
  --platform linux/arm64 \
  --file Dockerfile.gb200 \
  --build-arg BASE_IMAGE=ghcr.io/primeintellect-ai/pi-rft/prime-rl:v0.5.1.dev38 \
  --build-arg GB200_RUNTIME_IMAGE=ghcr.io/s1ro1/prime-rl-gb200-runtime:tmp-wheels-20260412-223722 \
  --build-arg WHEEL_BASE_URL=https://github.com/S1ro1/tmp-wheels/releases/download/tmp-wheels-20260412-223722 \
  --build-arg DEEP_GEMM_WHEEL=deep_gemm-2.3.0+477618c-cp312-cp312-linux_aarch64.whl \
  --build-arg DEEP_EP_WHEEL=deep_ep-1.2.1+9249c25-cp312-cp312-linux_aarch64.whl \
  --build-arg NIXL_WHEEL=nixl-0.10.1-py3-none-any.whl \
  --build-arg NIXL_CU12_WHEEL=nixl_cu12-1.0.0-cp312-cp312-linux_aarch64.whl \
  --tag ghcr.io/s1ro1/prime-rl:tmp-wheels-20260412-223722-gb200 \
  --push .
```

## Why DeepEP Needed Extra Build Fixes

DeepEP needed two GB200-specific build-time fixes:

- `use_fabric=True`
- extended timeout constants

Those are carried in:

- [`patches/gb200/deep_ep-gb200.patch`](../patches/gb200/deep_ep-gb200.patch)

DeepEP on arm64 + CUDA 12.8 also required patching `math_functions.h` before compilation:

- [`patch_cuda_math_header.py`](../scripts/gb200/patch_cuda_math_header.py)

## What We Learned About GHCR Push

Officially, there is no special extra “enable GHCR publishing” switch expected for a personal-account package push beyond a classic PAT with `write:packages`.

Relevant docs:

- https://docs.github.com/packages/guides/pushing-and-pulling-docker-images

What we observed:

- the token is valid and GitHub reports `write:packages`
- GHCR token exchange for `repository:s1ro1/prime-rl-gb200-runtime:push,pull` succeeds
- direct registry API upload start succeeds

So GHCR package creation itself was not the blocker.

The issue was client-side on this machine:

- `docker buildx build --push` built successfully
- Docker push/auth behavior was inconsistent during initial publish
- the runtime image was ultimately published successfully after bypassing Docker’s push path

For normal replication, keep the documented `docker login` + `docker buildx build --push` path above as the intended flow. Only fall back to client-level troubleshooting if Docker reproduces the same behavior again.
