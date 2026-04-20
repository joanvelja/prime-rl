"""Targeted verification for worker-d's mmlu.py changes.

Covers:
- _call_with_retry retries on 5xx and transient httpx.RequestError
- 4xx raises immediately (no retry)
- exhausting retries raises the final error (no silent -inf)
- _score_all_async happy path produces correct per-row shape
- all-(-inf) row in per_row_scores triggers RuntimeError guard
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evals import mmlu  # noqa: E402


# Speed up retry delays for tests.
mmlu._RETRY_DELAYS = (0.0, 0.01, 0.01, 0.01)


def _make_completion_body(
    token_logprobs: list[float | None], text_offsets: list[int]
) -> dict:
    return {
        "choices": [
            {
                "logprobs": {
                    "token_logprobs": token_logprobs,
                    "text_offset": text_offsets,
                }
            }
        ]
    }


def _success_response(prompt_text: str) -> httpx.Response:
    # ctx = "ctx", continuation = " x" → full = "ctx x"
    # Token breakdown (pretend): ["ctx", " x", "<gen>"]
    ctx_len = 3  # len("ctx")
    token_logprobs = [None, -1.0, -99.0]  # first is None (no preceding), last is generated
    text_offsets = [0, 3, 5]  # second token starts at offset 3 (== ctx_len), first continuation
    return httpx.Response(200, json=_make_completion_body(token_logprobs, text_offsets))


async def _test_retry_succeeds_after_5xx():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(503, json={"error": "overloaded"})
        return _success_response("ctx x")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sem = asyncio.Semaphore(1)
        score = await mmlu._score_continuation(
            "http://fake", "key", "m", "ctx", " x", sem, client
        )
    assert attempts["count"] == 3
    # Per-char logprob: -1.0 / len(" x")=2 → -0.5
    assert score == pytest.approx(-0.5)


async def _test_4xx_raises_immediately_no_retry():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sem = asyncio.Semaphore(1)
        with pytest.raises(httpx.HTTPStatusError):
            await mmlu._score_continuation(
                "http://fake", "key", "m", "ctx", " x", sem, client
            )
    assert attempts["count"] == 1  # no retry on 4xx


async def _test_exhaust_retries_raises():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(503, json={"error": "overloaded"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sem = asyncio.Semaphore(1)
        with pytest.raises(httpx.HTTPStatusError):
            await mmlu._score_continuation(
                "http://fake", "key", "m", "ctx", " x", sem, client
            )
    assert attempts["count"] == len(mmlu._RETRY_DELAYS)  # bounded retries


async def _test_transient_request_error_retries():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise httpx.ConnectError("boom", request=request)
        return _success_response("ctx x")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sem = asyncio.Semaphore(1)
        score = await mmlu._score_continuation(
            "http://fake", "key", "m", "ctx", " x", sem, client
        )
    assert attempts["count"] == 2
    assert score == pytest.approx(-0.5)


async def _test_score_all_async_happy_path():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["prompt"]
        # Context format: lines ending in "Answer:". Shortest row here: use the
        # actual context produced by _format_context.
        ctx_len = prompt.rfind(" ")  # last space; continuation starts here
        # Token breakdown: pretend 3 tokens in prompt + 1 generated.
        token_logprobs = [None, -2.0, -1.0, -99.0]  # ctx, ctx, cont, gen
        text_offsets = [0, 1, ctx_len, len(prompt)]
        return httpx.Response(
            200, json=_make_completion_body(token_logprobs, text_offsets)
        )

    rows = [
        {
            "subject": "math",
            "question": "1+1=?",
            "choices": ["2", "3", "4", "5"],
            "answer": 0,
        }
    ]

    transport = httpx.MockTransport(handler)
    # Monkeypatch httpx.AsyncClient to inject our transport.
    orig_client = httpx.AsyncClient

    class _PatchedClient(orig_client):
        def __init__(self, **kw):
            kw["transport"] = transport
            super().__init__(**kw)

    mmlu.httpx.AsyncClient = _PatchedClient
    try:
        result = await mmlu._score_all_async(
            rows, base_url="http://fake", api_key="k", model_id="m", max_concurrency=4
        )
    finally:
        mmlu.httpx.AsyncClient = orig_client

    assert result.n == 1
    assert len(result.per_row) == 1
    # All 4 choices scored (none -inf).
    for s in result.per_row[0]["scores"]:
        assert s != float("-inf"), f"got unexpected -inf score: {result.per_row[0]['scores']}"


async def _test_all_inf_guard_raises():
    # Return a degenerate response where all continuation tokens are None,
    # forcing _score_continuation to return -inf for every (row, choice).
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["prompt"]
        # All token_logprobs are None → vals empty → -inf path.
        return httpx.Response(
            200,
            json=_make_completion_body(
                token_logprobs=[None] * 4,
                text_offsets=[0, 1, 2, len(prompt)],
            ),
        )

    rows = [
        {
            "subject": "math",
            "question": "1+1=?",
            "choices": ["2", "3", "4", "5"],
            "answer": 0,
        }
    ]

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient

    class _PatchedClient(orig_client):
        def __init__(self, **kw):
            kw["transport"] = transport
            super().__init__(**kw)

    mmlu.httpx.AsyncClient = _PatchedClient
    try:
        with pytest.raises(RuntimeError, match="all-\\(-inf\\) scores"):
            await mmlu._score_all_async(
                rows, base_url="http://fake", api_key="k", model_id="m", max_concurrency=4
            )
    finally:
        mmlu.httpx.AsyncClient = orig_client


def test_retry_succeeds_after_5xx():
    asyncio.run(_test_retry_succeeds_after_5xx())


def test_4xx_raises_immediately_no_retry():
    asyncio.run(_test_4xx_raises_immediately_no_retry())


def test_exhaust_retries_raises():
    asyncio.run(_test_exhaust_retries_raises())


def test_transient_request_error_retries():
    asyncio.run(_test_transient_request_error_retries())


def test_score_all_async_happy_path():
    asyncio.run(_test_score_all_async_happy_path())


def test_all_inf_guard_raises():
    asyncio.run(_test_all_inf_guard_raises())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
