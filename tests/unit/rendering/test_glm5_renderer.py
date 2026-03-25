"""Tests for GLM-5 Renderer — every case compares against tokenizer.apply_chat_template()."""

import pytest
from transformers import AutoTokenizer

from prime_rl.rendering import GLM5Renderer, build_supervised_sample, build_trajectory_step


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("zai-org/GLM-5", trust_remote_code=True)


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return GLM5Renderer(tokenizer)


def _expected(tokenizer, messages, **kwargs):
    return tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]


# ── Basic rendering ──────────────────────────────────────────────────


def test_system_and_user(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_single_turn(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_no_system(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_multi_turn(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "D"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_empty_content(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": ""},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


# ── Generation prompt ────────────────────────────────────────────────


def test_generation_prompt_thinking_enabled(tokenizer, renderer):
    msgs = [{"role": "user", "content": "Hi"}]
    assert renderer.render_ids(msgs, add_generation_prompt=True) == _expected(
        tokenizer, msgs, add_generation_prompt=True, enable_thinking=True
    )


def test_generation_prompt_thinking_disabled(tokenizer):
    renderer = GLM5Renderer(tokenizer, enable_thinking=False)
    msgs = [{"role": "user", "content": "Hi"}]
    assert renderer.render_ids(msgs, add_generation_prompt=True) == _expected(
        tokenizer, msgs, add_generation_prompt=True, enable_thinking=False
    )


# ── Thinking ─────────────────────────────────────────────────────────


def test_reasoning_content_field(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Simple math", "content": "4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_thinking_in_content(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nMath.\n</think>\n\n4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_thinking_multi_turn(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "greeting", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "reasoning_content": "farewell", "content": "Goodbye!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


# ── Tools ────────────────────────────────────────────────────────────


def test_tools(tokenizer, renderer):
    msgs = [{"role": "user", "content": "Weather?"}]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_tool_call(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_tool_response(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {"role": "assistant", "content": "20 degrees."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_consecutive_tool_responses(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Two cities"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                {"function": {"name": "get_weather", "arguments": {"city": "London"}}},
            ],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {"role": "tool", "content": '{"temp": 15}'},
        {"role": "assistant", "content": "Paris 20, London 15."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_full_tool_cycle(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {"role": "assistant", "content": "20 degrees."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


# ── build helpers ────────────────────────────────────────────────────


def test_build_trajectory_step(tokenizer, renderer):
    prompt = [{"role": "user", "content": "Hi"}]
    completion = [{"role": "assistant", "content": "Hello!"}]
    step = build_trajectory_step(renderer, prompt, completion)
    assert step["prompt_ids"] + step["completion_ids"] == renderer.render_ids(prompt + completion)


def test_build_supervised_sample(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    ids, mask = build_supervised_sample(renderer, msgs, role_to_mask=lambda m: m["role"] == "assistant")
    assert ids == _expected(tokenizer, msgs)
    assert any(mask)
