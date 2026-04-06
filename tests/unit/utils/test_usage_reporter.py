import os
from unittest.mock import patch

from prime_rl.utils.usage_reporter import UsageReporter


def test_disabled_when_no_env_var():
    with patch.dict(os.environ, {}, clear=True):
        reporter = UsageReporter()
        assert reporter.is_enabled is False


def test_disabled_when_api_key_missing():
    with patch.dict(os.environ, {"PI_USAGE_BASE_URL": "http://localhost:8000/api/internal/rft"}, clear=True):
        reporter = UsageReporter()
        assert reporter.is_enabled is False


def test_enabled_when_configured():
    with patch.dict(
        os.environ,
        {"PI_USAGE_BASE_URL": "http://localhost:8000/api/internal/rft", "PI_USAGE_API_KEY": "test-key"},
        clear=True,
    ):
        reporter = UsageReporter()
        assert reporter.is_enabled is True
        reporter.close()


def test_report_training_noop_when_disabled():
    with patch.dict(os.environ, {}, clear=True):
        reporter = UsageReporter()
        reporter.report_training_usage("run1", 0, 5000)


def test_close_noop_when_disabled():
    with patch.dict(os.environ, {}, clear=True):
        reporter = UsageReporter()
        reporter.close()
