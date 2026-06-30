#!/usr/bin/env bash
# Standing CXI net plugin: aws-ofi-nccl 1.20.0 (v12), overriding the brics module's 1.8.1 (v7).
#
# Mechanism: PREPEND the 1.20.0 lib so NCCL dlopens its libnccl-net.so first. The brics/nccl module
# APPENDS the 1.8.1 lib (module show -> append_path), so a prepend always wins — even after a later
# isambard-fabric.sh reloads brics/nccl. Do NOT `module unload` the 1.8.1 module: that also strips
# NCCL_NET="AWS Libfabric" + FI_CXI_* env the brics module set (verified), which NCCL needs to select
# the OFI plugin. Prepending keeps all that env intact while still binding 1.20.0 (v12).
#
# NCCL itself is 2.30.7 via the .venv wheel (pyproject [tool.uv] override-dependencies) — torch loads
# it by absolute path, so no LD change is needed here for NCCL. Validated torch-safe inter-node
# (job 5441435: torch nccl_runtime=23007, v12 binds, GB-scale collectives correct across 2 nodes).
# Idempotent in effect (first match wins; dup entries harmless); safe under set -euo pipefail;
# degrades to the stock brics 1.8.1 plugin if the artifact is absent. Source AFTER any brics/nccl load.
_OFI_120=/lus/lfs1aip2/projects/a6r/joanv.a6r/opt/aws-ofi-nccl-1.20.0/lib
if [ -d "$_OFI_120" ]; then
  # libfabric.so.1 (linked by the 1.20.0 plugin) is already on the path via the still-loaded 1.8.1
  # module; re-add defensively in case a future change unloads it.
  export LD_LIBRARY_PATH="$_OFI_120:/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH:-}"
fi
unset _OFI_120
