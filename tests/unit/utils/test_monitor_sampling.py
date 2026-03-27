from prime_rl.utils.monitor.sampling import sample_rollouts_for_logging


def test_sample_rollouts_for_logging_keeps_all_when_ratio_is_none():
    rollouts = list(range(20))

    selected = sample_rollouts_for_logging(rollouts, None)

    assert selected == rollouts


def test_sample_rollouts_for_logging_returns_empty_when_ratio_is_zero():
    rollouts = list(range(20))

    selected = sample_rollouts_for_logging(rollouts, 0.0)

    assert selected == []


def test_sample_rollouts_for_logging_samples_expected_count():
    rollouts = list(range(20))

    selected = sample_rollouts_for_logging(rollouts, 0.25)

    assert len(selected) == 5
    assert set(selected).issubset(set(rollouts))
