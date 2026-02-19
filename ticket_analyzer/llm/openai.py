import logging
import time

from openai import AsyncOpenAI

from ticket_analyzer.config import settings

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    LLMProvider implementation backed by the OpenAI chat completions API.

    Uses `response_format={"type": "json_object"}` to enforce JSON output
    at the API level, and `temperature=0.0` for deterministic results.
    Token usage and API latency are logged on every call for cost observability.

    Configured via:
        LLM_MODEL       - model to use (default: gpt-4o-mini)
        OPENAI_API_KEY  - OpenAI API key
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def classify(self, system_prompt: str, user_text: str) -> str:
        """
        Call the OpenAI chat completions API and return the raw JSON response.

        Raises:
            ValueError: If the API returns an empty message content.
            openai.OpenAIError: On API-level failures (auth, rate limit, network).
        """
        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        latency_ms = round((time.perf_counter() - t0) * 1000)

        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("LLM returned an empty response")

        usage = response.usage
        logger.info(
            "LLM call completed | model=%s latency_ms=%d input_tokens=%s output_tokens=%s total_tokens=%s",
            response.model,
            latency_ms,
            usage.prompt_tokens if usage else None,
            usage.completion_tokens if usage else None,
            usage.total_tokens if usage else None,
        )

        return raw
