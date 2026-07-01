from scripts.docent.ingest_prime_rollouts import (
    _messages_from_prompt_completion,
    _private_reasoning_chars,
    _step_metadata,
)


def test_assistant_reasoning_content_becomes_docent_reasoning_block():
    messages = _messages_from_prompt_completion(
        [{"role": "user", "content": "Question"}],
        [
            {
                "role": "assistant",
                "reasoning_content": "private scratchpad",
                "content": "public answer",
            }
        ],
        step_index=2,
    )

    assistant = messages[1]
    assert assistant["content"] == [
        {"type": "reasoning", "reasoning": "private scratchpad"},
        {"type": "text", "text": "public answer"},
    ]
    assert assistant["metadata"] == {
        "prime_rl_step_index": 2,
        "prime_rl_has_private_reasoning": True,
        "prime_rl_private_reasoning_sources": ["reasoning_content"],
    }
    assert _private_reasoning_chars(messages) == len("private scratchpad")


def test_assistant_without_reasoning_stays_plain_text():
    messages = _messages_from_prompt_completion(
        [{"role": "user", "content": "Question"}],
        [{"role": "assistant", "content": "public answer"}],
        step_index=0,
    )

    assert messages[1] == {
        "role": "assistant",
        "content": "public answer",
        "metadata": {"prime_rl_step_index": 0},
    }
    assert _private_reasoning_chars(messages) == 0


def test_step_metadata_exposes_channels_and_debate_fields():
    messages = _messages_from_prompt_completion(
        [{"role": "user", "content": "Question"}],
        [
            {
                "role": "assistant",
                "reasoning_content": "private scratchpad",
                "content": "public answer",
            }
        ],
        step_index=1,
    )
    step = {
        "reward": 1.0,
        "advantage": 0.25,
        "tokens": {"completion_ids": [1, 2]},
        "extras": {
            "member_id": "debater_a",
            "phase": "critique",
            "fields": {"answer": "C"},
        },
    }

    metadata = _step_metadata(step, messages, step_index=1)

    assert metadata["member_id"] == "debater_a"
    assert metadata["phase"] == "critique"
    assert metadata["fields"] == {"answer": "C"}
    assert metadata["channels"] == {
        "public": {"present": True, "visibility": "all"},
        "private": {"present": True, "visibility": "self"},
    }
    assert metadata["has_private_reasoning"] is True
    assert metadata["private_reasoning_chars"] == len("private scratchpad")
    assert metadata["public_text_chars"] == len("Question") + len("public answer")
