from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from ticket_analyzer.main import app
from ticket_analyzer.schemas import TicketAnalysis


def _valid_analysis() -> TicketAnalysis:
    return TicketAnalysis(
        category="billing",
        priority="medium",
        sentiment="negative",
        confidence=0.88,
        suggested_response="We are reviewing your billing issue and will respond within 24 hours.",
        reasoning="Customer reports an unexpected charge on their account.",
    )


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestAnalyzeRoute:
    async def test_successful_analysis(self, client):
        with patch(
            "ticket_analyzer.routes.ClassifierService.analyze",
            new=AsyncMock(return_value=_valid_analysis()),
        ):
            response = await client.post(
                "/api/v1/analyze",
                json={"text": "I was charged twice for my subscription this month."},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["analysis"]["category"] == "billing"
        assert body["analysis"]["priority"] == "medium"
        assert body["analysis"]["sentiment"] == "negative"
        assert body["ticket_id"] is None

    async def test_ticket_id_echoed(self, client):
        with patch(
            "ticket_analyzer.routes.ClassifierService.analyze",
            new=AsyncMock(return_value=_valid_analysis()),
        ):
            response = await client.post(
                "/api/v1/analyze",
                json={
                    "text": "I was charged twice for my subscription this month.",
                    "ticket_id": "TKT-99",
                },
            )

        assert response.status_code == 200
        assert response.json()["ticket_id"] == "TKT-99"

    async def test_text_too_short_returns_422(self, client):
        response = await client.post(
            "/api/v1/analyze",
            json={"text": "Help"},
        )
        assert response.status_code == 422

    async def test_missing_text_returns_422(self, client):
        response = await client.post("/api/v1/analyze", json={})
        assert response.status_code == 422

    async def test_classifier_failure_returns_502(self, client):
        with patch(
            "ticket_analyzer.routes.ClassifierService.analyze",
            new=AsyncMock(side_effect=RuntimeError("Classification failed after 3 attempts")),
        ):
            response = await client.post(
                "/api/v1/analyze",
                json={"text": "I was charged twice for my subscription this month."},
            )

        assert response.status_code == 502


class TestHealthRoute:
    async def test_health(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
