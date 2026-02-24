import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from ticket_analyzer.llm import ClassificationResult
from ticket_analyzer.llm import LLMResult
from ticket_analyzer.main import app
from ticket_analyzer.domain import TicketAnalysis
from ticket_analyzer.http import TicketClassificationRecord


def _valid_analysis() -> TicketAnalysis:
    return TicketAnalysis(
        category="billing",
        priority="medium",
        sentiment="negative",
        confidence=0.88,
        suggested_response="We are reviewing your billing issue and will respond within 24 hours.",
        reasoning="Customer reports an unexpected charge on their account.",
    )


def _valid_llm_result() -> LLMResult:
    return LLMResult(
        raw_json="{}",
        model="gpt-4o-mini",
        latency_ms=200,
        input_tokens=60,
        output_tokens=90,
    )


def _valid_classification_result() -> ClassificationResult:
    return ClassificationResult(
        analysis=_valid_analysis(),
        llm_result=_valid_llm_result(),
    )


def _valid_record(**overrides) -> TicketClassificationRecord:
    base = dict(
        id=uuid.uuid4(),
        ticket_id="TKT-99",
        text="I was charged twice for my subscription this month.",
        category="billing",
        priority="medium",
        sentiment="negative",
        confidence=0.88,
        suggested_response="We are reviewing your billing issue and will respond within 24 hours.",
        reasoning="Customer reports an unexpected charge on their account.",
        model="gpt-4o-mini",
        latency_ms=200,
        input_tokens=60,
        output_tokens=90,
        created_at=datetime.now(tz=timezone.utc),
    )
    return TicketClassificationRecord(**{**base, **overrides})


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── POST /api/v1/analyze ──────────────────────────────────────────────────────

async def test_successful_analysis(client):
    with (
        patch(
            "ticket_analyzer.routes.analyze",
            new=AsyncMock(return_value=_valid_classification_result()),
        ),
        patch(
            "ticket_analyzer.routes.queries.save",
            new=AsyncMock(),
        ),
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
    assert body["analysis"]["confidence"] == 0.88
    assert body["ticket_id"] is None


async def test_ticket_id_echoed(client):
    with (
        patch(
            "ticket_analyzer.routes.analyze",
            new=AsyncMock(return_value=_valid_classification_result()),
        ),
        patch(
            "ticket_analyzer.routes.queries.save",
            new=AsyncMock(),
        ),
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


async def test_repository_save_called_on_success(client):
    save_mock = AsyncMock()
    with (
        patch("ticket_analyzer.routes.analyze", new=AsyncMock(return_value=_valid_classification_result())),
        patch("ticket_analyzer.routes.queries.save", new=save_mock),
    ):
        await client.post(
            "/api/v1/analyze",
            json={"text": "I was charged twice for my subscription this month."},
        )

    save_mock.assert_called_once()


async def test_repository_save_not_called_on_failure(client):
    save_mock = AsyncMock()
    with (
        patch("ticket_analyzer.routes.analyze", new=AsyncMock(side_effect=RuntimeError("Failed after 3 attempts"))),
        patch("ticket_analyzer.routes.queries.save", new=save_mock),
    ):
        await client.post(
            "/api/v1/analyze",
            json={"text": "I was charged twice for my subscription this month."},
        )

    save_mock.assert_not_called()


async def test_text_too_short_returns_422(client):
    response = await client.post(
        "/api/v1/analyze",
        json={"text": "Help"},
    )
    assert response.status_code == 422


async def test_missing_text_returns_422(client):
    response = await client.post("/api/v1/analyze", json={})
    assert response.status_code == 422


async def test_classifier_failure_returns_502(client):
    with patch(
        "ticket_analyzer.routes.analyze",
        new=AsyncMock(side_effect=RuntimeError("Failed after 3 attempts")),
    ):
        response = await client.post(
            "/api/v1/analyze",
            json={"text": "I was charged twice for my subscription this month."},
        )

    assert response.status_code == 502


# ── GET /api/v1/tickets ───────────────────────────────────────────────────────

async def test_list_tickets(client):
    record = _valid_record()
    with patch(
        "ticket_analyzer.routes.queries.get_all",
        new=AsyncMock(return_value=[record]),
    ):
        response = await client.get("/api/v1/tickets")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["category"] == "billing"
    assert body[0]["ticket_id"] == "TKT-99"
    assert body[0]["confidence"] == 0.88


async def test_list_tickets_empty(client):
    with patch(
        "ticket_analyzer.routes.queries.get_all",
        new=AsyncMock(return_value=[]),
    ):
        response = await client.get("/api/v1/tickets")

    assert response.status_code == 200
    assert response.json() == []


async def test_list_tickets_returns_multiple(client):
    records = [_valid_record(id=uuid.uuid4()), _valid_record(id=uuid.uuid4())]
    with patch(
        "ticket_analyzer.routes.queries.get_all",
        new=AsyncMock(return_value=records),
    ):
        response = await client.get("/api/v1/tickets")

    assert response.status_code == 200
    assert len(response.json()) == 2


# ── GET /api/v1/tickets/{id} ──────────────────────────────────────────────────

async def test_get_ticket_found(client):
    record = _valid_record()
    with patch(
        "ticket_analyzer.routes.queries.get_by_id",
        new=AsyncMock(return_value=record),
    ):
        response = await client.get(f"/api/v1/tickets/{record.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "TKT-99"
    assert body["category"] == "billing"
    assert body["priority"] == "medium"
    assert body["sentiment"] == "negative"
    assert body["confidence"] == 0.88
    assert body["model"] == "gpt-4o-mini"


async def test_get_ticket_not_found(client):
    with patch(
        "ticket_analyzer.routes.queries.get_by_id",
        new=AsyncMock(return_value=None),
    ):
        response = await client.get(f"/api/v1/tickets/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_get_ticket_invalid_uuid(client):
    response = await client.get("/api/v1/tickets/not-a-uuid")
    assert response.status_code == 422


# ── GET /health ───────────────────────────────────────────────────────────────

async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
