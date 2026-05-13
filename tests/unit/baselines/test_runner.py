import json
from types import SimpleNamespace

from prime_rl.baselines.config import BaselineConfig
from prime_rl.baselines.runner import (
    _collect_judge_cache_stats,
    _eval_examples,
    _expected_rollout_keys,
    _merge_rollout_outputs,
    _read_partial_outputs,
    _sampling_args,
)


class _EvalEnv:
    def __init__(self):
        self.calls = []
        self.rows = [
            {"example_id": 2, "answer": "b"},
            {"example_id": 65, "answer": "loop"},
            {"example_id": 2641, "answer": "loop"},
        ]

    def get_eval_dataset(self, n=-1, seed=None):
        self.calls.append((n, seed))
        rows = self.rows if n < 0 else self.rows[:n]
        return SimpleNamespace(to_list=lambda: list(rows))


def test_collect_judge_cache_stats_walks_nested_rubrics_once():
    shared = SimpleNamespace(judge_cache_stats={"hits": 2, "misses": 1})
    env = SimpleNamespace(
        rubric=SimpleNamespace(
            rubrics={"answer": shared},
            grader=shared,
            matcher=SimpleNamespace(judge_cache_stats={"hits": 0, "misses": 3}),
            judge_rubric=SimpleNamespace(judge_cache_stats={"persistent_hits": 1}),
        )
    )

    stats = _collect_judge_cache_stats(env)

    assert stats == {
        "env.rubric.rubrics.answer": {"hits": 2, "misses": 1},
        "env.rubric.matcher": {"hits": 0, "misses": 3},
        "env.rubric.judge_rubric": {"persistent_hits": 1},
    }


def test_sampling_args_routes_vllm_only_keys_to_extra_body(tmp_path):
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path,
        sampling_args={
            "temperature": 0.7,
            "repetition_penalty": 1.12,
            "chat_template_kwargs": {"enable_thinking": False},
            "include_reasoning": False,
            "bad_words": ["<think>", "</think>"],
            "extra_body": {"top_k": 40},
        },
    )

    assert _sampling_args(config) == {
        "n": 1,
        "temperature": 0.7,
        "extra_body": {
            "repetition_penalty": 1.12,
            "chat_template_kwargs": {"enable_thinking": False},
            "include_reasoning": False,
            "bad_words": ["<think>", "</think>"],
            "top_k": 40,
        },
    }


def test_sampling_args_keeps_runner_owned_fanout_and_explicit_extra_body_precedence(tmp_path):
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path,
        sampling_args={
            "n": 4,
            "top_k": 40,
            "extra_body": {"top_k": 50},
        },
    )

    assert _sampling_args(config) == {
        "n": 1,
        "extra_body": {"top_k": 50},
    }


def test_sampling_args_prefers_max_completion_tokens_over_deprecated_max_tokens(tmp_path):
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path,
        sampling_args={
            "max_tokens": 8192,
            "max_completion_tokens": 512,
        },
    )

    assert _sampling_args(config) == {
        "n": 1,
        "max_completion_tokens": 512,
    }


def test_eval_examples_filters_record_ids_in_requested_order(tmp_path):
    env = _EvalEnv()
    config = BaselineConfig(
        env_id="hf_singleturn",
        model="model",
        output_dir=tmp_path,
        seed=123,
        num_examples=1,
        record_ids=["2641", "65"],
    )

    examples = _eval_examples(config, env)

    assert examples == [
        {"example_id": 2641, "answer": "loop"},
        {"example_id": 65, "answer": "loop"},
    ]
    assert env.calls == [(-1, 123)]


def test_read_partial_outputs_keeps_expected_latest_rows(tmp_path):
    partial_path = tmp_path / "raw_rollouts.partial.jsonl"
    rows = [
        {"trial_index": 0, "output": {"example_id": "a", "completion": "old"}},
        {"trial_index": 1, "output": {"example_id": "a", "completion": "keep"}},
        {"trial_index": 0, "output": {"example_id": "stale", "completion": "drop"}},
        {"trial_index": 0, "output": {"example_id": "a", "completion": "new"}},
    ]
    partial_path.write_text(
        "\n".join(json.dumps(row) for row in rows)
        + "\n"
        + '{"trial_index": 1, "output": {"example_id": "broken"',
    )

    outputs = _read_partial_outputs(partial_path, {("a", 0), ("a", 1)})

    assert outputs == {
        ("a", 0): (0, {"example_id": "a", "completion": "new"}),
        ("a", 1): (1, {"example_id": "a", "completion": "keep"}),
    }


def test_merge_rollout_outputs_respects_expected_order_and_newer_rows():
    expected = _expected_rollout_keys([{"example_id": "b"}, {"example_id": "a"}], 2)

    merged = _merge_rollout_outputs(
        expected_keys=expected,
        existing_outputs=[
            (0, {"example_id": "a", "completion": "a0-old"}),
            (1, {"example_id": "a", "completion": "a1"}),
            (0, {"example_id": "b", "completion": "b0"}),
        ],
        new_outputs=[
            (0, {"example_id": "a", "completion": "a0-new"}),
            (1, {"example_id": "b", "completion": "b1"}),
        ],
    )

    assert merged == [
        (0, {"example_id": "a", "completion": "a0-new"}),
        (1, {"example_id": "a", "completion": "a1"}),
        (0, {"example_id": "b", "completion": "b0"}),
        (1, {"example_id": "b", "completion": "b1"}),
    ]
