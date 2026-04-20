#!/bin/bash
# Both `flash-attn` (FA2) and `flash-attn-cute` (FA4) ship a `flash_attn/cute/`
# sub-package. The one from `flash-attn` is a tiny stub; `flash-attn-cute`
# ships the real FA4 kernels (>1000 lines in interface.py). When both extras
# are installed, `uv sync` may install `flash-attn` *after* `flash-attn-cute`,
# causing the stub to overwrite the real module.
#
# This script is idempotent: it checks first, and only reinstalls FA4 if the
# stub has clobbered it. Safe to call unconditionally after `uv sync`.

set -e

CUTE_INTERFACE=$(uv run python -c 'import flash_attn.cute.interface as m; print(m.__file__)' 2>/dev/null || echo "")

if [ -n "$CUTE_INTERFACE" ]; then
    LINES=$(wc -l < "$CUTE_INTERFACE")
    if [ "$LINES" -gt 1000 ]; then
        echo "flash-attn-cute OK ($LINES lines at $CUTE_INTERFACE); no repair needed."
        exit 0
    fi
    echo "flash-attn-cute clobbered by FA2 stub ($LINES lines); reinstalling FA4..."
else
    echo "flash_attn.cute.interface not importable; attempting fresh FA4 install..."
fi

uv pip install --reinstall --no-deps \
    "flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@abd9943b#subdirectory=flash_attn/cute"

# Re-verify
CUTE_INTERFACE=$(uv run python -c 'import flash_attn.cute.interface as m; print(m.__file__)')
LINES=$(wc -l < "$CUTE_INTERFACE")
if [ "$LINES" -gt 1000 ]; then
    echo "Success: flash-attn-cute interface.py has $LINES lines (correct version)"
else
    echo "Error: flash-attn-cute interface.py still has only $LINES lines (wrong version)"
    exit 1
fi
