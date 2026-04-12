# GB200 Image Build Pipeline

This repo now carries a dedicated GB200 image build path that avoids the old tarball overlay flow.

For the exact runbook used to reach the currently published wheel/runtime state, see [`gb200-runbook.md`](./gb200-runbook.md).

## What Gets Published

For each GB200 release tag, the pipeline publishes:

- a base `prime-rl:<tag>-arm64` image from [`Dockerfile.cuda`](../Dockerfile.cuda)
- a native runtime image from [`Dockerfile.gb200-runtime`](../Dockerfile.gb200-runtime) containing pinned `UCX` and `NVSHMEM`
- wheel assets built by [`Dockerfile.gb200-wheels`](../Dockerfile.gb200-wheels)
- a final `prime-rl:<tag>-gb200-arm64` image from [`Dockerfile.gb200`](../Dockerfile.gb200)

The final image installs only published wheels in its Dockerfile:

- `deep_ep`
- `deep_gemm`
- `nixl`
- `nixl_cu12`

There is no `site-packages` tar overlay and no runtime patch script in the image build.

## Carried Patches

The only baked source delta in this first production pass is [`patches/gb200/deep_ep-gb200.patch`](../patches/gb200/deep_ep-gb200.patch), which:

- flips `DeepEP` `use_fabric` default on for GB200 fabric handles
- extends the DeepEP timeout constants used by the current multi-node deployment

## Intentionally Omitted For Now

We are **not** baking the `vllm/distributed/device_communicators/all2all.py` high-throughput MNNVL override into the release artifacts yet.

That omission is intentional:

- the workflow release notes call it out explicitly
- the generated `gb200-release-manifest.json` records it under `omitted_patches`
- this lets us validate the cleaner wheel-only image first before carrying another `vllm` delta in the artifact pipeline

If that patch is still required after image validation, add a separate patched `vllm` wheel artifact rather than reintroducing runtime file edits.

## DeepGEMM Note

The DeepGEMM `ep_scatter` int64 fix is already applied by [`src/prime_rl/inference/patches.py`](../src/prime_rl/inference/patches.py) during `uv run inference`, so the image pipeline does not duplicate it.

## Workflow

The release workflow lives at [`.github/workflows/release-gb200.yaml`](../.github/workflows/release-gb200.yaml).

It performs these steps:

1. build and push the arm64 base image
2. build and push the GB200 runtime image
3. build the wheelhouse locally via Docker and upload it as release assets
4. build the final GB200 image by downloading those wheel assets in `Dockerfile.gb200`
5. smoke test that the image imports `deep_ep`, `deep_gemm`, `nixl`, and contains `/opt/ucx` and `/opt/nvshmem`
