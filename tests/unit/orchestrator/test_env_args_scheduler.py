from unittest.mock import MagicMock

from prime_rl.orchestrator.env_args_scheduler import (
    EnvArgsEvent,
    EnvArgsState,
    build_env_args_metrics,
    build_env_args_table_rows,
    merge_arg_diff,
)


def test_merge_arg_diff_preserves_unspecified_keys():
    current_args = {"prompt_suffix": "old", "difficulty": 1}

    merged = merge_arg_diff(current_args, {"difficulty": 2})

    assert merged == {"prompt_suffix": "old", "difficulty": 2}
    assert current_args == {"prompt_suffix": "old", "difficulty": 1}


def test_request_change_updates_desired_args_but_not_active_args():
    state = EnvArgsState(
        env_name="env_a",
        active_version=0,
        desired_version=0,
        active_args={"prompt_suffix": "old", "difficulty": 1},
        desired_args={"prompt_suffix": "old", "difficulty": 1},
    )

    events = state.request_change(step=5, diff={"difficulty": 2})

    assert state.active_version == 0
    assert state.desired_version == 1
    assert state.active_args == {"prompt_suffix": "old", "difficulty": 1}
    assert state.desired_args == {"prompt_suffix": "old", "difficulty": 2}
    assert events == [
        EnvArgsEvent(
            step=5,
            env="env_a",
            event="request",
            version="0->1",
            args={"difficulty": 2},
        )
    ]


def test_request_change_is_noop_when_effective_args_do_not_change():
    state = EnvArgsState(
        env_name="env_a",
        active_version=0,
        desired_version=0,
        active_args={"difficulty": 1},
        desired_args={"difficulty": 1},
    )

    events = state.request_change(step=3, diff={"difficulty": 1})

    assert state.active_version == 0
    assert state.desired_version == 0
    assert state.active_args == {"difficulty": 1}
    assert state.desired_args == {"difficulty": 1}
    assert events == []


def test_request_change_emits_overlap_event_when_reload_is_in_progress():
    reload_task = MagicMock()
    reload_task.done.return_value = False
    state = EnvArgsState(
        env_name="env_a",
        active_version=0,
        desired_version=1,
        active_args={"difficulty": 1},
        desired_args={"difficulty": 2},
        reload_task=reload_task,
    )

    events = state.request_change(step=9, diff={"difficulty": 3})

    assert state.desired_version == 2
    assert state.desired_args == {"difficulty": 3}
    assert events == [
        EnvArgsEvent(step=9, env="env_a", event="request", version="1->2", args={"difficulty": 3}),
        EnvArgsEvent(step=9, env="env_a", event="overlap", version="active=0 desired=2", args={"difficulty": 3}),
    ]


def test_mark_reload_complete_updates_active_state():
    state = EnvArgsState(
        env_name="env_a",
        active_version=0,
        desired_version=1,
        active_args={"difficulty": 1},
        desired_args={"difficulty": 2},
    )

    event = state.mark_reload_complete(step=11, version=1, args={"difficulty": 2})

    assert state.active_version == 1
    assert state.desired_version == 1
    assert state.active_args == {"difficulty": 2}
    assert state.desired_args == {"difficulty": 2}
    assert event == EnvArgsEvent(step=11, env="env_a", event="activate", version="0->1", args={"difficulty": 2})


def test_mark_reload_complete_clears_pending_version_and_reload_timestamp():
    state = EnvArgsState(
        env_name="env_a",
        active_version=0,
        desired_version=1,
        active_args={"difficulty": 1},
        desired_args={"difficulty": 2},
        pending_version=1,
        reload_started_at=12.5,
    )

    event = state.mark_reload_complete(step=11, version=1, args={"difficulty": 2})

    assert event is not None
    assert state.pending_version is None
    assert state.reload_started_at is None


def test_build_env_args_metrics_counts_changes_reloads_and_pending_envs():
    states = {
        "env_a": EnvArgsState(
            env_name="env_a",
            active_version=1,
            desired_version=1,
            active_args={"difficulty": 2},
            desired_args={"difficulty": 2},
        ),
        "env_b": EnvArgsState(
            env_name="env_b",
            active_version=0,
            desired_version=1,
            active_args={"difficulty": 1},
            desired_args={"difficulty": 2},
        ),
    }
    events = [
        EnvArgsEvent(step=4, env="env_a", event="request", version="0->1", args={"difficulty": 2}),
        EnvArgsEvent(step=4, env="env_a", event="activate", version="0->1", args={"difficulty": 2}),
    ]

    metrics = build_env_args_metrics(states, events)

    assert metrics == {
        "env_args/changed": 1,
        "env_args/reloaded": 1,
        "env_args/pending": 1,
    }


def test_build_env_args_table_rows_keeps_only_event_columns():
    events = [
        EnvArgsEvent(step=7, env="env_a", event="request", version="0->1", args={"difficulty": 2}),
        EnvArgsEvent(step=8, env="env_a", event="activate", version="0->1", args={"difficulty": 2}),
    ]

    rows = build_env_args_table_rows(events)

    assert rows == [
        {
            "step": 7,
            "env": "env_a",
            "event": "request",
            "version": "0->1",
            "args": '{"difficulty": 2}',
        },
        {
            "step": 8,
            "env": "env_a",
            "event": "activate",
            "version": "0->1",
            "args": '{"difficulty": 2}',
        },
    ]
