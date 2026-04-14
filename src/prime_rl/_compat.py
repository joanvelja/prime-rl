"""Compatibility shims for upstream regressions.

Imported early (before any model code) by trainer and orchestrator entrypoints.
Each shim documents the upstream issue and removal condition.
"""

# ---------------------------------------------------------------------------
# ring_flash_attn + transformers >= 5.4
#
# ring_flash_attn 0.1.8 imports `is_flash_attn_greater_or_equal_2_10` from
# `transformers.modeling_flash_attention_utils`, removed in transformers 5.4.
#
# Upstream fix: https://github.com/zhuzilin/ring-flash-attention/pull/85
# Remove once ring_flash_attn ships a fixed version.
# ---------------------------------------------------------------------------
import transformers.modeling_flash_attention_utils as _mfau

if not hasattr(_mfau, "is_flash_attn_greater_or_equal_2_10"):
    _mfau.is_flash_attn_greater_or_equal_2_10 = lambda: True


# ---------------------------------------------------------------------------
# transformers >= 5.5 hub_kernels offline regression
#
# lazy_load_kernel() resolves kernel versions via HfApi().list_repo_refs(),
# which raises OfflineModeIsEnabled when HF_HUB_OFFLINE=1 — even if the
# kernel is already cached locally. Only FileNotFoundError and AssertionError
# are caught; OfflineModeIsEnabled (a ConnectionError subclass) is not.
#
# This breaks Mamba-based models (NemotronH, Zamba2, Jamba, etc.) when
# loaded with HF_HUB_OFFLINE=1 — which our multi-node SLURM template
# (multi_node_rl.sbatch.j2) sets to prevent worker nodes from making
# network calls during training.
#
# Regression chain:
#   PR #41577 (Oct 2025) — lazy_load_kernel introduced
#   PR #43955 (Feb 2026) — version=1 integer versioning bypasses cache
#   PR #44176 (Feb 2026) — all Mamba models switched to lazy_load_kernel
#
# Fix: catch the exception and load kernels from the local HF Hub cache.
# Remove once huggingface/transformers fixes lazy_load_kernel offline handling.
# ---------------------------------------------------------------------------
try:
    import transformers.integrations.hub_kernels as _hub_kernels

    _original_lazy_load_kernel = _hub_kernels.lazy_load_kernel
    _default_mapping = _hub_kernels._KERNEL_MODULE_MAPPING

    def _patched_lazy_load_kernel(kernel_name, mapping=None):
        if mapping is None:
            mapping = _default_mapping
        try:
            return _original_lazy_load_kernel(kernel_name, mapping)
        except Exception:
            pass

        # Hub call failed (OfflineModeIsEnabled, ConnectionError, etc.).
        # Try loading from the local HF Hub cache.
        try:
            from pathlib import Path

            from huggingface_hub.constants import HF_HUB_CACHE
            from kernels.utils import _find_kernel_in_repo_path, _import_from_path, package_name_from_repo_id

            info = _hub_kernels._HUB_KERNEL_MAPPING.get(kernel_name)
            if info is not None:
                repo_id = info["repo_id"]
                cache_repo = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
                snapshots_dir = cache_repo / "snapshots"
                if snapshots_dir.exists():
                    package_name = package_name_from_repo_id(repo_id)
                    # Prefer the snapshot pinned by refs/main (latest downloaded revision)
                    snapshot_dirs = []
                    refs_main = cache_repo / "refs" / "main"
                    if refs_main.exists():
                        pinned = snapshots_dir / refs_main.read_text().strip()
                        if pinned.is_dir():
                            snapshot_dirs.append(pinned)
                    snapshot_dirs.extend(d for d in snapshots_dir.iterdir() if d.is_dir() and d not in snapshot_dirs)

                    for snapshot in snapshot_dirs:
                        try:
                            _, variant_path = _find_kernel_in_repo_path(snapshot, package_name)
                            kernel = _import_from_path(package_name, variant_path)
                            mapping[kernel_name] = kernel
                            return kernel
                        except (FileNotFoundError, ImportError, OSError):
                            continue
        except Exception:
            pass

        mapping[kernel_name] = None
        return None

    _hub_kernels.lazy_load_kernel = _patched_lazy_load_kernel

    # Patch references in model files that were already imported before this
    # patch ran. We don't eagerly import them — the _hub_kernels patch above
    # ensures any future imports get the patched version automatically.
    import sys

    for _mod_name in [
        "transformers.models.zamba2.modeling_zamba2",
        "transformers.models.nemotron_h.modeling_nemotron_h",
    ]:
        _mod = sys.modules.get(_mod_name)
        if _mod is not None and hasattr(_mod, "lazy_load_kernel"):
            _mod.lazy_load_kernel = _patched_lazy_load_kernel

except ImportError:
    pass
