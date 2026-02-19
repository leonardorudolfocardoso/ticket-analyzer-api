import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ticket_analyzer.classifier import ClassifierService
from ticket_analyzer.schemas import TicketRequest


def _make_llm_response(content: str) -> MagicMock:
    """Build a mock that looks like an openai ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = "gpt-4o-mini"

    return response


VALID_PAYLOAD = json.dumps({
    "category": "authentication",
    "priority": "high",
    "sentiment": "negative",
    "confidence": 0.92,
    "suggested_response": "We are sorry to hear you are having trouble logging in. Please try resetting your password.",
    "reasoning": "Customer cannot access their account and password reset is not working.",
})


@pytest.fixture
def service():
    with patch("ticket_analyzer.classifier.AsyncOpenAI"):
        svc = ClassifierService()
        svc._client = AsyncMock()
        return svc


@pytest.fixture
def request_():
    return TicketRequest(
        text="I cannot log into my account. The password reset email never arrives.",
        ticket_id="TKT-001",
    )


class TestClassifierService:
    async def test_successful_classification(self, service, request_):
        service._client.chat.completions.create = AsyncMock(
            return_value=_make_llm_response(VALID_PAYLOAD)
        )

        result = await service.analyze(request_)

        assert result.category == "authentication"
        assert result.priority == "high"
        assert result.sentiment == "negative"
        assert result.confidence == 0.92

    async def test_retries_on_invalid_json(self, service, request_):
        service._client.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response("not valid json"),
            _make_llm_response("still not json"),
            _make_llm_response(VALID_PAYLOAD),
        ])

        result = await service.analyze(request_)

        assert result.category == "authentication"
        assert service._client.chat.completions.create.call_count == 3

    async def test_retries_on_invalid_schema(self, service, request_):
        bad_payload = json.dumps({"category": "unknown_category", "priority": "high"})

        service._client.chat.completions.create = AsyncMock(side_effect=[
            _make_llm_response(bad_payload),
            _make_llm_response(VALID_PAYLOAD),
        ])

        result = await service.analyze(request_)

        assert result.category == "authentication"
        assert service._client.chat.completions.create.call_count == 2

    async def test_raises_after_all_retries_exhausted(self, service, request_):
        service._client.chat.completions.create = AsyncMock(
            return_value=_make_llm_response("not valid json")
        )

        with pytest.raises(RuntimeError, match="Classification failed after"):
            await service.analyze(request_)

    async def test_raises_on_empty_response(self, service, request_):
        service._client.chat.completions.create = AsyncMock(
            return_value=_make_llm_response("")
        )

        with pytest.raises(RuntimeError, match="Classification failed after"):
            await service.analyze(request_)

    async def test_llm_called_with_correct_message(self, service, request_):
        service._client.chat.completions.create = AsyncMock(
            return_value=_make_llm_response(VALID_PAYLOAD)
        )

        await service.analyze(request_)

        call_kwargs = service._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == request_.text
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["response_format"] == {"type": "json_object"}
