import logging

import prime_rl._compat  # noqa: F401 — must run before ring_flash_attn is imported
from prime_rl.utils.hf_config import patch_transformers_config_rope_numeric_types


class _SuppressHubKernelsDisabledWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "kernels hub usage is disabled through the environment USE_HUB_KERNELS=" not in record.getMessage()


logging.getLogger("transformers.integrations.hub_kernels").addFilter(_SuppressHubKernelsDisabledWarning())
patch_transformers_config_rope_numeric_types()
