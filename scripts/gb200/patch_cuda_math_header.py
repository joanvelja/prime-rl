from __future__ import annotations

import sys
from pathlib import Path


def should_comment(line: str) -> bool:
    return (
        "extern __DEVICE_FUNCTIONS_DECL__" in line
        and "__device_builtin__" in line
        and any(token in line for token in (" sinpi", " sinpif", " cospi", " cospif"))
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_cuda_math_header.py <math_functions.h>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    original = path.read_text()
    lines = original.splitlines()

    changed = False
    patched_lines: list[str] = []
    for line in lines:
        if should_comment(line) and not line.lstrip().startswith("//"):
            patched_lines.append("//" + line)
            changed = True
        else:
            patched_lines.append(line)

    if changed:
        path.write_text("\n".join(patched_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
