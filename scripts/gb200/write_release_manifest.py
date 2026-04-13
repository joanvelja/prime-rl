from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(artifact_dir: Path, prefix: str) -> Path:
    matches = sorted(artifact_dir.glob(f"{prefix}*.whl"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one wheel for {prefix}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--patchset-rev", required=True)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--deepgemm-commit", required=True)
    parser.add_argument("--deepep-source", required=True)
    parser.add_argument("--ucx-version", required=True)
    parser.add_argument("--nvshmem-version", required=True)
    parser.add_argument("--nixl-git-ref", required=True)
    parser.add_argument("--nixl-python-version", required=True)
    parser.add_argument("--cuda-toolkit-version", required=True)
    parser.add_argument("--torch-cuda-arch-list", required=True)
    parser.add_argument("--omitted-patch", action="append", default=[])
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    wheels = {
        "deep_gemm": find_one(artifact_dir, "deep_gemm-"),
        "deep_ep": find_one(artifact_dir, "deep_ep-"),
        "nixl_cu12": find_one(artifact_dir, "nixl_cu12-"),
        "nixl": find_one(artifact_dir, "nixl-"),
    }

    manifest = {
        "schema_version": 1,
        "release_version": args.release_version,
        "patchset_rev": args.patchset_rev,
        "cuda_toolkit_version": args.cuda_toolkit_version,
        "torch_cuda_arch_list": args.torch_cuda_arch_list,
        "components": {
            "vllm": {
                "version": args.vllm_version,
                "notes": "Uses prime_rl runtime monkey patch for DeepGEMM ep_scatter; no baked all2all override.",
            },
            "deep_gemm": {"source_commit": args.deepgemm_commit},
            "deep_ep": {
                "source_ref": args.deepep_source,
                "patches": [
                    "use_fabric default forced to True",
                    "NUM_CPU_TIMEOUT_SECS 100 -> 10000",
                    "NUM_CPU_TIMEOUT_SECS 10 -> 1000",
                    "NUM_TIMEOUT_CYCLES 200000000000ull -> 20000000000000ull",
                    "NUM_TIMEOUT_CYCLES 20000000000ull -> 2000000000000ull",
                ],
            },
            "ucx": {"source_ref": args.ucx_version},
            "nvshmem": {"version": args.nvshmem_version},
            "nixl": {
                "source_ref": args.nixl_git_ref,
                "python_shim_version": args.nixl_python_version,
            },
        },
        "omitted_patches": args.omitted_patch,
        "wheels": {
            key: {"file": path.name, "sha256": sha256sum(path)}
            for key, path in wheels.items()
        },
    }

    (artifact_dir / "gb200-release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
