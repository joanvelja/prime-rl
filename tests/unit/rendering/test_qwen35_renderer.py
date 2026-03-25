"""Tests for Qwen3.5 Renderer — every case compares against tokenizer.apply_chat_template()."""

import pytest
from transformers import AutoTokenizer

from prime_rl.rendering import Qwen35Renderer, build_supervised_sample, build_trajectory_step
from prime_rl.rendering.base import Renderer


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return Qwen35Renderer(tokenizer)


@pytest.fixture(scope="module")
def renderer_no_think(tokenizer):
    return Qwen35Renderer(tokenizer, enable_thinking=False)


def _expected(tokenizer, messages, *, tools=None, add_generation_prompt=False, **kwargs):
    """Get expected token IDs from tokenizer.apply_chat_template."""
    return tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=add_generation_prompt, return_dict=False, **kwargs
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]


# ── Protocol conformance ─────────────────────────────────────────────


def test_satisfies_protocol(renderer):
    assert isinstance(renderer, Renderer)


# ── Basic message rendering ──────────────────────────────────────────


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


def test_no_system_message(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_multi_turn(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_multi_turn_many_rounds(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "D"},
        {"role": "user", "content": "E"},
        {"role": "assistant", "content": "F"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_empty_content(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": ""},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_none_content(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": None},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_whitespace_content(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "  You are helpful.  "},
        {"role": "user", "content": "  Hello!  "},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


# ── Generation prompt ────────────────────────────────────────────────


def test_generation_prompt_thinking_enabled(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    assert renderer.render_ids(msgs, add_generation_prompt=True) == _expected(
        tokenizer, msgs, add_generation_prompt=True, enable_thinking=True
    )


def test_generation_prompt_thinking_disabled(tokenizer, renderer_no_think):
    msgs = [
        {"role": "user", "content": "Hi"},
    ]
    assert renderer_no_think.render_ids(msgs, add_generation_prompt=True) == _expected(
        tokenizer, msgs, add_generation_prompt=True, enable_thinking=False
    )


# ── Thinking / reasoning ────────────────────────────────────────────


def test_reasoning_content_field(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "reasoning_content": "Simple arithmetic: 2+2=4", "content": "4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_thinking_in_content(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nSimple math.\n</think>\n\n4"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_thinking_in_history_before_last_query(tokenizer, renderer):
    """Assistant messages BEFORE last_query_index should NOT get thinking blocks."""
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "reasoning_content": "greeting", "content": "Hi!"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "reasoning_content": "farewell", "content": "Goodbye!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_thinking_empty_reasoning(tokenizer, renderer):
    """Empty reasoning_content should produce empty thinking block."""
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "reasoning_content": "", "content": "Hello!"},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


# ── Tool definitions ─────────────────────────────────────────────────


def test_tools_with_system(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "What's the weather?"},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_tools_without_system(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "What's the weather?"},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_tools_with_generation_prompt(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Get weather for Paris"},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS, add_generation_prompt=True) == _expected(
        tokenizer, msgs, tools=TOOLS, add_generation_prompt=True
    )


# ── Tool calls ───────────────────────────────────────────────────────


def test_single_tool_call_with_content(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check the weather.",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}],
        },
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_single_tool_call_no_content(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_multiple_tool_calls(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris and London?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
                {"function": {"name": "get_weather", "arguments": {"city": "London"}}},
            ],
        },
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_tool_call_with_complex_arguments(tokenizer, renderer):
    """Arguments that are dicts/lists should be JSON-serialized."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "filters": {"type": "object"},
                        "tags": {"type": "array"},
                    },
                },
            },
        }
    ]
    msgs = [
        {"role": "user", "content": "Search for cats"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "search",
                        "arguments": {"query": "cats", "filters": {"type": "animal"}, "tags": ["cute", "fluffy"]},
                    }
                }
            ],
        },
    ]
    assert renderer.render_ids(msgs, tools=tools) == _expected(tokenizer, msgs, tools=tools)


# ── Tool responses ───────────────────────────────────────────────────


def test_single_tool_response(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {"role": "assistant", "content": "It's 20 degrees."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_consecutive_tool_responses(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Weather in Paris and London?"},
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
        {"role": "assistant", "content": "Paris: 20, London: 15."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


# ── Full tool cycle ──────────────────────────────────────────────────


def test_full_tool_cycle(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}],
        },
        {"role": "tool", "content": '{"temp": 20, "condition": "sunny"}'},
        {"role": "assistant", "content": "It is 20 degrees and sunny in Paris."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_multi_step_tool_cycle(tokenizer, renderer):
    """Two rounds of tool calling."""
    msgs = [
        {"role": "user", "content": "Compare weather in Paris and London"},
        {
            "role": "assistant",
            "content": "Let me check Paris first.",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
        {"role": "tool", "content": '{"temp": 20}'},
        {
            "role": "assistant",
            "content": "Now London.",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "London"}}}],
        },
        {"role": "tool", "content": '{"temp": 15}'},
        {"role": "assistant", "content": "Paris: 20, London: 15."},
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


# ── RenderedTokens message_indices ───────────────────────────────────


def test_message_indices_basic(renderer):
    msgs = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    assert len(rendered.token_ids) == len(rendered.message_indices)

    # All indices should reference valid messages or be -1
    for idx in rendered.message_indices:
        assert idx == -1 or 0 <= idx < len(msgs)

    # System tokens should have index 0
    assert rendered.message_indices[0] == 0

    unique = sorted(set(rendered.message_indices))
    assert unique == [0, 1, 2], f"Expected indices for 3 messages, got {unique}"


def test_message_indices_with_generation_prompt(renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
    ]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    # Last tokens (generation prompt) should have index -1
    assert rendered.message_indices[-1] == -1


# ── Stop tokens ──────────────────────────────────────────────────────


def test_stop_token_ids(renderer, tokenizer):
    stop_ids = renderer.get_stop_token_ids()
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") in stop_ids
    assert tokenizer.convert_tokens_to_ids("<|endoftext|>") in stop_ids


# ── build_supervised_sample ──────────────────────────────────────────


def test_build_supervised_sample_basic(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    def assistant_only(msg):
        return msg.get("role") == "assistant"

    ids, mask = build_supervised_sample(renderer, msgs, role_to_mask=assistant_only)

    # Token IDs must match apply_chat_template
    assert ids == _expected(tokenizer, msgs)

    # Mask: True only for assistant tokens
    rendered = renderer.render(msgs)
    for i, (m, idx) in enumerate(zip(mask, rendered.message_indices)):
        if idx >= 0 and msgs[idx]["role"] == "assistant":
            assert m is True, f"Expected True for assistant token at {i}"
        else:
            assert m is False, f"Expected False for non-assistant token at {i}"


def test_build_supervised_sample_multi_turn(tokenizer, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Goodbye!"},
    ]

    def assistant_only(msg):
        return msg.get("role") == "assistant"

    ids, mask = build_supervised_sample(renderer, msgs, role_to_mask=assistant_only)
    assert ids == _expected(tokenizer, msgs)

    # Count masked tokens — should cover both assistant messages
    assert sum(mask) > 0


# ── build_trajectory_step ────────────────────────────────────────────


def test_build_trajectory_step_basic(tokenizer, renderer):
    prompt = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    completion = [{"role": "assistant", "content": "Hello!"}]

    step = build_trajectory_step(renderer, prompt, completion)

    # Full reconstruction must equal the full render
    full_ids = renderer.render_ids(prompt + completion)
    assert step["prompt_ids"] + step["completion_ids"] == full_ids

    # Masks
    assert all(m is False for m in step["prompt_mask"])
    assert all(m is True for m in step["completion_mask"])
    assert len(step["completion_logprobs"]) == len(step["completion_ids"])


def test_build_trajectory_step_with_thinking(tokenizer, renderer):
    prompt = [{"role": "user", "content": "What is 2+2?"}]
    completion = [{"role": "assistant", "reasoning_content": "2+2=4", "content": "4"}]

    step = build_trajectory_step(renderer, prompt, completion)
    full_ids = renderer.render_ids(prompt + completion)
    assert step["prompt_ids"] + step["completion_ids"] == full_ids


def test_build_trajectory_step_with_tools(tokenizer, renderer):
    prompt = [
        {"role": "user", "content": "Weather?"},
    ]
    completion = [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        }
    ]

    step = build_trajectory_step(renderer, prompt, completion, tools=TOOLS)
    full_ids = renderer.render_ids(prompt + completion, tools=TOOLS)
    assert step["prompt_ids"] + step["completion_ids"] == full_ids


# ── Extension property ───────────────────────────────────────────────


def test_extension_property_holds_tool_cycle(tokenizer, renderer):
    """Extension holds in tool cycles because last_query_index stays the same."""
    step1_prompt = [
        {"role": "user", "content": "Weather in Paris?"},
    ]
    step1_completion = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}],
        },
    ]

    # Step 2: tool response comes in, model responds
    step2_prompt = step1_prompt + step1_completion + [{"role": "tool", "content": '{"temp": 20}'}]
    step2_completion = [{"role": "assistant", "content": "It's 20 degrees."}]

    s1 = build_trajectory_step(renderer, step1_prompt, step1_completion, tools=TOOLS)
    s2 = build_trajectory_step(renderer, step2_prompt, step2_completion, tools=TOOLS)

    step1_full = s1["prompt_ids"] + s1["completion_ids"]

    # Step 2's prompt should start with step 1's full sequence
    assert s2["prompt_ids"][: len(step1_full)] == step1_full


def test_extension_property_breaks_new_user_query(tokenizer, renderer):
    """Extension breaks when a new user query changes last_query_index,
    causing historical assistant messages to lose their thinking blocks.
    This is inherent to the Qwen3.5 template design.
    """
    step1_prompt = [
        {"role": "user", "content": "Hi"},
    ]
    step1_completion = [{"role": "assistant", "content": "Hello!"}]

    step2_prompt = step1_prompt + step1_completion + [{"role": "user", "content": "Bye"}]
    step2_completion = [{"role": "assistant", "content": "Goodbye!"}]

    s1 = build_trajectory_step(renderer, step1_prompt, step1_completion)
    s2 = build_trajectory_step(renderer, step2_prompt, step2_completion)

    step1_full = s1["prompt_ids"] + s1["completion_ids"]

    # Extension breaks because the assistant at index 1 had <think> block in step 1
    # but doesn't in step 2 (new user query at index 2 shifts last_query_index)
    assert s2["prompt_ids"][: len(step1_full)] != step1_full


# ── VLM (multimodal content) ─────────────────────────────────────────


def test_vlm_single_image(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You are a vision assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": "What is in this image?"},
            ],
        },
        {"role": "assistant", "content": "I see a cat."},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_vlm_multiple_images(tokenizer, renderer):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "First."},
                {"type": "image"},
                {"type": "text", "text": "Second. Compare."},
            ],
        },
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_vlm_image_url_format(tokenizer, renderer):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "http://example.com/cat.jpg"}},
                {"type": "text", "text": "Describe this."},
            ],
        },
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_vlm_video(tokenizer, renderer):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "placeholder"},
                {"type": "text", "text": "What happens in this video?"},
            ],
        },
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)


def test_vlm_with_tools(tokenizer, renderer):
    msgs = [
        {"role": "system", "content": "You can see and use tools."},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What city is this? Get the weather."},
            ],
        },
    ]
    assert renderer.render_ids(msgs, tools=TOOLS) == _expected(tokenizer, msgs, tools=TOOLS)


def test_vlm_multi_turn(tokenizer, renderer):
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is this?"},
            ],
        },
        {"role": "assistant", "content": "A cat."},
        {"role": "user", "content": "What color?"},
        {"role": "assistant", "content": "Orange."},
    ]
    assert renderer.render_ids(msgs) == _expected(tokenizer, msgs)
