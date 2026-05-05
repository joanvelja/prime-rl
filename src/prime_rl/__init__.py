import prime_rl._compat  # noqa: F401 — must run before ring_flash_attn is imported
from prime_rl.utils.hf_config import patch_transformers_config_rope_numeric_types

patch_transformers_config_rope_numeric_types()
