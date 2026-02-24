import json
from unittest.mock import AsyncMock

import pytest

from ticket_analyzer.llm import PROMPT, analyze, retry
from ticket_analyzer.llm import LLMResult
from ticket_analyzer.http import TicketRequest

VALID_PAYLOAD = json.dumps({
    "category": "authentication",
    "priority": "high",
    "sentiment": "negative",
    "confidence": 0.92,
    "suggested_response": "We are sorry to hear you are having trouble logging in. Please try resetting your password.",
    "reasoning": "Customer cannot access their account and password reset is not working.",
})


def _llm_result(raw_json: str) -> LLMResult:
    return LLMResult(
        raw_json=raw_json,
        model="gpt-4o-mini",
        latency_ms=100,
        input_tokens=50,
        output_tokens=80,
    )


@pytest.fixture
def request_():
    return TicketRequest(
        text="I cannot log into my account. The password reset email never arrives.",
        ticket_id="TKT-001",
    )


# ── retry ─────────────────────────────────────────────────────────────────────

async def test_retry_succeeds_on_first_attempt():
    fn = AsyncMock(return_value="ok")

    result = await retry(fn, max_retries=3, retryable=(ValueError,))

    assert result == "ok"
    assert fn.call_count == 1


async def test_retry_succeeds_after_failures():
    fn = AsyncMock(side_effect=[ValueError(), ValueError(), "ok"])

    result = await retry(fn, max_retries=3, retryable=(ValueError,))

    assert result == "ok"
    assert fn.call_count == 3


async def test_retry_raises_after_exhaustion():
    fn = AsyncMock(side_effect=ValueError("bad"))

    with pytest.raises(RuntimeError, match="Failed after 3 attempts"):
        await retry(fn, max_retries=3, retryable=(ValueError,))

    assert fn.call_count == 3


async def test_retry_does_not_catch_non_retryable():
    fn = AsyncMock(side_effect=RuntimeError("fatal"))

    with pytest.raises(RuntimeError, match="fatal"):
        await retry(fn, max_retries=3, retryable=(ValueError,))

    assert fn.call_count == 1


async def test_retry_respects_max_retries():
    fn = AsyncMock(side_effect=ValueError())

    with pytest.raises(RuntimeError):
        await retry(fn, max_retries=1, retryable=(ValueError,))

    assert fn.call_count == 1


# ── analyze ───────────────────────────────────────────────────────────────────

async def test_successful_classification(request_):
    classify_fn = AsyncMock(return_value=_llm_result(VALID_PAYLOAD))

    result = await analyze(request_, classify_fn)

    assert result.analysis.category == "authentication"
    assert result.analysis.priority == "high"
    assert result.analysis.sentiment == "negative"
    assert result.analysis.confidence == 0.92
    assert result.llm_result.model == "gpt-4o-mini"


async def test_analyze_calls_classify_fn_with_correct_args(request_):
    classify_fn = AsyncMock(return_value=_llm_result(VALID_PAYLOAD))

    await analyze(request_, classify_fn)

    classify_fn.assert_called_once_with(PROMPT, request_.text)


async def test_analyze_returns_llm_metadata(request_):
    classify_fn = AsyncMock(return_value=_llm_result(VALID_PAYLOAD))

    result = await analyze(request_, classify_fn)

    assert result.llm_result.latency_ms == 100
    assert result.llm_result.input_tokens == 50
    assert result.llm_result.output_tokens == 80


async def test_retries_on_invalid_json(request_):
    classify_fn = AsyncMock(side_effect=[
        _llm_result("not valid json"),
        _llm_result("still not json"),
        _llm_result(VALID_PAYLOAD),
    ])

    result = await analyze(request_, classify_fn)

    assert result.analysis.category == "authentication"
    assert classify_fn.call_count == 3


async def test_retries_on_invalid_schema(request_):
    bad_payload = json.dumps({"category": "unknown_category", "priority": "high"})
    classify_fn = AsyncMock(side_effect=[
        _llm_result(bad_payload),
        _llm_result(VALID_PAYLOAD),
    ])

    result = await analyze(request_, classify_fn)

    assert result.analysis.category == "authentication"
    assert classify_fn.call_count == 2


async def test_raises_after_all_retries_exhausted(request_):
    classify_fn = AsyncMock(return_value=_llm_result("not valid json"))

    with pytest.raises(RuntimeError, match="Failed after"):
        await analyze(request_, classify_fn)


async def test_raises_on_empty_response(request_):
    classify_fn = AsyncMock(return_value=_llm_result(""))

    with pytest.raises(RuntimeError, match="Failed after"):
        await analyze(request_, classify_fn)
