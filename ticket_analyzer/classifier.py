import json
import logging

from pydantic import ValidationError

from ticket_analyzer.config import settings
from ticket_analyzer.llm import LLMProvider
from ticket_analyzer.schemas import TicketAnalysis, TicketRequest

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
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


class ClassifierService:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def analyze(self, request: TicketRequest) -> TicketAnalysis:
        logger.info(
            "Analyzing ticket",
            extra={"ticket_id": request.ticket_id, "text_length": len(request.text)},
        )

        last_error: Exception | None = None

        for attempt in range(1, settings.max_retries + 1):
            try:
                raw = await self._provider.classify(SYSTEM_PROMPT, request.text)
                result = TicketAnalysis.model_validate_json(raw)
                logger.info(
                    "Classification succeeded",
                    extra={"ticket_id": request.ticket_id, "attempt": attempt},
                )
                return result
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "Classification attempt failed",
                    extra={
                        "ticket_id": request.ticket_id,
                        "attempt": attempt,
                        "error": str(exc),
                    },
                )

        raise RuntimeError(
            f"Classification failed after {settings.max_retries} attempts"
        ) from last_error
