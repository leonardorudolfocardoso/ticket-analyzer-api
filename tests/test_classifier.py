import json
from unittest.mock import AsyncMock

import pytest

from ticket_analyzer.classifier import ClassifierService
from ticket_analyzer.schemas import TicketRequest

VALID_PAYLOAD = json.dumps({
    "category": "authentication",
    "priority": "high",
    "sentiment": "negative",
    "confidence": 0.92,
    "suggested_response": "We are sorry to hear you are having trouble logging in. Please try resetting your password.",
    "reasoning": "Customer cannot access their account and password reset is not working.",
})


@pytest.fixture
def provider():
    return AsyncMock()


@pytest.fixture
def service(provider):
    return ClassifierService(provider)


@pytest.fixture
def request_():
    return TicketRequest(
        text="I cannot log into my account. The password reset email never arrives.",
        ticket_id="TKT-001",
    )


class TestClassifierService:
    async def test_successful_classification(self, service, provider, request_):
        provider.classify = AsyncMock(return_value=VALID_PAYLOAD)

        result = await service.analyze(request_)

        assert result.category == "authentication"
        assert result.priority == "high"
        assert result.sentiment == "negative"
        assert result.confidence == 0.92

    async def test_retries_on_invalid_json(self, service, provider, request_):
        provider.classify = AsyncMock(side_effect=[
            "not valid json",
            "still not json",
            VALID_PAYLOAD,
        ])

        result = await service.analyze(request_)

        assert result.category == "authentication"
        assert provider.classify.call_count == 3

    async def test_retries_on_invalid_schema(self, service, provider, request_):
        bad_payload = json.dumps({"category": "unknown_category", "priority": "high"})
        provider.classify = AsyncMock(side_effect=[bad_payload, VALID_PAYLOAD])

        result = await service.analyze(request_)

        assert result.category == "authentication"
        assert provider.classify.call_count == 2

    async def test_raises_after_all_retries_exhausted(self, service, provider, request_):
        provider.classify = AsyncMock(return_value="not valid json")

        with pytest.raises(RuntimeError, match="Classification failed after"):
            await service.analyze(request_)

    async def test_raises_on_empty_response(self, service, provider, request_):
        provider.classify = AsyncMock(return_value="")

        with pytest.raises(RuntimeError, match="Classification failed after"):
            await service.analyze(request_)
