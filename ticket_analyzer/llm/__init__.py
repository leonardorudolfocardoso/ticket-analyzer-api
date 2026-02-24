import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import ValidationError

from ticket_analyzer.config import settings
from ticket_analyzer.domain import TicketAnalysis
from ticket_analyzer.http import TicketRequest

logger = logging.getLogger(__name__)

T = TypeVar("T")

@dataclass
class LLMResult:
    """
    Metadata returned alongside the raw JSON from an LLM call.

    Attributes:
        raw_json:      The raw JSON string returned by the model.
        model:         The model identifier used (e.g. "gpt-4o-mini").
        latency_ms:    Round-trip latency in milliseconds.
        input_tokens:  Prompt token count, or None if unavailable.
        output_tokens: Completion token count, or None if unavailable.
    """

    raw_json: str
    model: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class ClassificationResult:
    """Wraps both the validated analysis and the raw LLM call metadata."""

    analysis: TicketAnalysis
    llm_result: LLMResult


class ClassifyFn(Protocol):
    """
    Protocol for any callable that sends a prompt to an LLM and returns an LLMResult.

    To add a new provider, implement a plain async function with this signature
    and register it in `get_provider` below.
    """

    async def __call__(self, system_prompt: str, user_text: str) -> LLMResult: ...

async def retry(
    fn: Callable[[], Awaitable[T]],
    max_retries: int,
    retryable: tuple[type[Exception], ...],
) -> T:
    """
    Retry an async callable up to `max_retries` times on expected failures.

    Each failed attempt is logged as a warning. Non-retryable exceptions
    propagate immediately without consuming retry budget.

    Args:
        fn:          Zero-argument async callable to invoke on each attempt.
        max_retries: Maximum number of attempts before giving up.
        retryable:   Tuple of exception types that trigger a retry.

    Returns:
        The return value of `fn` on the first successful attempt.

    Raises:
        RuntimeError: When all attempts are exhausted.
        Exception:    Any exception not listed in `retryable`, immediately.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await fn()
        except retryable as exc:
            logger.warning("Attempt %d/%d failed | error=%s", attempt, max_retries, exc)

    raise RuntimeError(f"Failed after {max_retries} attempts")


PROMPT = """\
    You are a support ticket classification system. Analyze the given ticket and \
    return a JSON object with exactly these fields:

    - category: one of "billing", "technical_issue", "authentication", \
    "feature_request", "general_question", "other"
    - priority: one of "low", "medium", "high", "urgent"
    - sentiment: one of "positive", "neutral", "negative"
    - confidence: a float between 0.0 and 1.0 representing your classification confidence
    - suggested_response: a professional draft reply the support agent can send (10–2000 chars)
    - reasoning: a brief explanation of why you chose this classification (10–1000 chars)

    Priority guide:
    - urgent: system down, data loss, security breach, complete blocker
    - high: major feature broken, significant business impact
    - medium: partial degradation, workaround exists
    - low: general questions, minor issues, feature requests

    Return ONLY the JSON object. No markdown, no code fences, no extra text.\
"""

RETRYABLE = (ValidationError, ValueError, json.JSONDecodeError)
async def analyze(request: TicketRequest, classify_fn: ClassifyFn) -> ClassificationResult:
    """
    Classify a support ticket with automatic retries on recoverable failures.

    Retries on `ValidationError`, `ValueError`, and `JSONDecodeError` up to
    `settings.max_retries` times. LLM-level errors (auth, rate limit, network)
    are not retryable and surface immediately.

    Args:
        request:     The incoming ticket request.
        classify_fn: A `ClassifyFn`-compatible callable to use for inference.

    Returns:
        A `ClassificationResult` with the validated analysis and LLM metadata.

    Raises:
        RuntimeError: If all retry attempts are exhausted.
    """
    logger.info("Analyzing ticket | ticket_id=%s text_length=%d", request.ticket_id, len(request.text))

    async def attempt() -> ClassificationResult:
        llm_result = await classify_fn(PROMPT, request.text)
        analysis = TicketAnalysis.model_validate_json(llm_result.raw_json)
        return ClassificationResult(analysis=analysis, llm_result=llm_result)

    result = await retry(attempt, max_retries=settings.max_retries, retryable=RETRYABLE)

    logger.info("Classification succeeded | ticket_id=%s", request.ticket_id)
    return result


def get_provider(provider: str = settings.llm_provider) -> ClassifyFn:
    """
    Return the LLM classify function for the given provider name.

    Defaults to the value of the LLM_PROVIDER environment variable.
    To add a new provider, implement the ClassifyFn protocol and add
    a branch here.

    Args:
        provider: Provider identifier (e.g. "openai"). Defaults to `settings.llm_provider`.

    Raises:
        ValueError: If `provider` is not a recognised value.
    """
    if provider == "openai":
        from ticket_analyzer.llm.openai import classify
        return classify
    raise ValueError(f"Unknown LLM provider: '{provider}'")
