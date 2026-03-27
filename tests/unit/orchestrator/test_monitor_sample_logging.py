from prime_rl.orchestrator.orchestrator import _select_rollouts_for_sample_logging


def test_select_rollouts_for_sample_logging_keeps_full_batch():
    rollouts = [{"example_id": i} for i in range(20)]

    selected = _select_rollouts_for_sample_logging(rollouts)

    assert selected == rollouts
    assert len(selected) == 20
