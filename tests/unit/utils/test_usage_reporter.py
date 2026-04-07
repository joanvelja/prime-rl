import os
from unittest.mock import patch

from prime_rl.utils.usage_reporter import UsageReporter


def test_initializes_with_env_vars():
    with patch.dict(
        os.environ,
        {"PI_USAGE_BASE_URL": "http://localhost:8000/api/internal/rft", "PI_USAGE_API_KEY": "test-key"},
        clear=True,
    ):
        reporter = UsageReporter()
        try:
            assert reporter._base_url == "http://localhost:8000/api/internal/rft"
            assert reporter._api_key == "test-key"
        finally:
            reporter.close()


def test_close_is_idempotent():
    with patch.dict(
        os.environ,
        {"PI_USAGE_BASE_URL": "http://localhost:8000/api/internal/rft", "PI_USAGE_API_KEY": "test-key"},
        clear=True,
    ):
        reporter = UsageReporter()
        reporter.close()
        reporter.close()  # second call should be a no-op
