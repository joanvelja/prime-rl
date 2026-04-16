"""Compatibility shim: ring_flash_attn + transformers >= 5.4.

ring_flash_attn 0.1.8 imports `is_flash_attn_greater_or_equal_2_10` from
`transformers.modeling_flash_attention_utils`. This symbol was removed from
that module in transformers 5.4 (still available as a deprecated function
in `transformers.utils.import_utils`, scheduled for removal in 5.8).

ring_flash_attn's except-branch is a no-op (imports the same symbol again),
so the import crashes on transformers >= 5.4. We patch the symbol back in as
`True` — the check is dead code since no one uses flash_attn < 2.1.0 anymore.

`prime_rl/__init__.py` imports this unconditionally, so the shim has to be
robust to contexts that don't have transformers at all (the verifiers fork
venv used by orchestrator unit tests on Darwin — prime_sandboxes is there,
transformers is not). We gate with `find_spec` so a genuinely broken
transformers install (partial wheel, missing transitive) still fails loud
at import time; only a cleanly-absent transformers takes the skip path.

Upstream fix: https://github.com/zhuzilin/ring-flash-attention/pull/85
Remove this shim once ring_flash_attn ships a fixed version.
"""

import importlib.util

if importlib.util.find_spec("transformers") is not None:
    import transformers.modeling_flash_attention_utils as _mfau

    if not hasattr(_mfau, "is_flash_attn_greater_or_equal_2_10"):
        # ring_flash_attn uses this as a bare value (if x:), but other code
        # may call it as a function (if x():). Use a callable that is also
        # truthy.
        _mfau.is_flash_attn_greater_or_equal_2_10 = lambda: True
