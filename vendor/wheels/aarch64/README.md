# aarch64 wheels (vendored)

Pre-built wheels for packages that upstream only publishes for `x86_64`. Referenced from `pyproject.toml` via `[tool.uv.sources]` with a `platform_machine == 'aarch64'` marker, so `uv sync` picks them automatically on Isambard / other ARM hosts.

## `vllm_router-0.1.22-cp38-abi3-linux_aarch64.whl`

PrimeIntellect's `vllm-router` (Rust + pyo3) ships an x86_64-only manylinux wheel. Built locally for aarch64 on Isambard (GH200, Cray Slingshot, SLES 15).

### Rebuild recipe

Prereqs on Isambard: `rustup` + `cargo` (already at `~/.cargo/bin/`), and `uv`. No system protoc — torch ships an old one without the well-known type protos, so we pull official protoc from GitHub releases.

```bash
cd /lus/lfs1aip2/projects/a6r/joanv.a6r/work/prime-rl/tmp

# 1. Source
git clone --depth 1 --branch v0.1.22 \
  https://github.com/PrimeIntellect-ai/router vllm-router-aarch64

# 2. Bundled protoc + well-known type protos (timestamp.proto, struct.proto)
mkdir -p protoc-bundle && cd protoc-bundle
curl -sSL -o protoc.zip \
  https://github.com/protocolbuffers/protobuf/releases/download/v28.3/protoc-28.3-linux-aarch_64.zip
unzip -q -o protoc.zip
cd ..

# 3. Build wheel (setuptools-rust, uv-managed build env)
cd vllm-router-aarch64
PROTOC=$PWD/../protoc-bundle/bin/protoc \
PROTOC_INCLUDE=$PWD/../protoc-bundle/include \
  uv build --wheel

# 4. Vendor and install
cp dist/vllm_router-0.1.22-cp38-abi3-linux_aarch64.whl \
   ../../vendor/wheels/aarch64/
cd ../..
uv lock                        # refreshes the hash for the vendored path
uv sync --extra disagg         # or: uv pip install vendor/wheels/aarch64/vllm_router-*.whl
```

Build takes ~3 min on a GH200 login node. Most of the time is Rust dep compilation (axum, tower, hyper, tonic, kube, aws-lc-rs).

### Why this is needed

The router gives us cross-`ApiServer` load balancing on each inference node. Without it the sbatch template falls back to direct backend URLs — works but loses LB across the 4 ApiServers per node, costing ~5-10% on hot-prompt clusters.

### Notes / gotchas

- `cp38-abi3` means the wheel works on any CPython ≥ 3.8 (uses the stable ABI). Don't need to rebuild per Python minor version.
- `torch/bin/protoc` is too old (3.13.0) and ships no `.proto` includes — don't try to use it.
- The build emits `vllm_router_rs.abi3.so` plus a thin Python shim under `vllm_router/`. The `vllm-router` entry point is `vllm_router.launch_router:main`.
- Upstream tag `v0.1.22` still has `vllm_router/version.py` set to `0.1.12`. Trust the wheel metadata / lockfile package version (`0.1.22`) instead of that stale module constant.
- If `tonic-prost-build` ever needs a newer protoc, bump the protoc-bundle URL accordingly.
