import pytest
from pydantic import ValidationError

from ticket_analyzer.domain import TicketAnalysis
from ticket_analyzer.http import TicketRequest


def _valid_analysis(**overrides) -> dict:
    base = {
        "category": "billing",
        "priority": "high",
        "sentiment": "negative",
        "confidence": 0.95,
        "suggested_response": "We are looking into your billing issue right away.",
        "reasoning": "Customer mentions invoice and incorrect charge.",
    }
    return {**base, **overrides}


# ── TicketRequest ─────────────────────────────────────────────────────────────

def test_text_too_short():
    with pytest.raises(ValidationError):
        TicketRequest(text="Too short")


def test_text_too_long():
    with pytest.raises(ValidationError):
        TicketRequest(text="x" * 5001)


def test_text_at_min_length():
    request = TicketRequest(text="x" * 10)
    assert len(request.text) == 10


def test_text_at_max_length():
    request = TicketRequest(text="x" * 5000)
    assert len(request.text) == 5000


def test_ticket_id_is_optional():
    request = TicketRequest(text="x" * 10)
    assert request.ticket_id is None


def test_ticket_id_is_set_when_provided():
    request = TicketRequest(text="x" * 10, ticket_id="TKT-001")
    assert request.ticket_id == "TKT-001"


# ── TicketAnalysis ────────────────────────────────────────────────────────────

def test_invalid_category():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(category="unknown"))


def test_invalid_priority():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(priority="critical"))


def test_invalid_sentiment():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(sentiment="angry"))


def test_confidence_below_zero():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(confidence=-0.1))


def test_confidence_above_one():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(confidence=1.1))


def test_confidence_boundary_values():
    assert TicketAnalysis(**_valid_analysis(confidence=0.0)).confidence == 0.0
    assert TicketAnalysis(**_valid_analysis(confidence=1.0)).confidence == 1.0


def test_suggested_response_too_short():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(suggested_response="short"))


def test_suggested_response_too_long():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(suggested_response="x" * 2001))


def test_suggested_response_at_boundaries():
    assert TicketAnalysis(**_valid_analysis(suggested_response="x" * 10))
    assert TicketAnalysis(**_valid_analysis(suggested_response="x" * 2000))


def test_reasoning_too_short():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(reasoning="short"))


def test_reasoning_too_long():
    with pytest.raises(ValidationError):
        TicketAnalysis(**_valid_analysis(reasoning="x" * 1001))


def test_reasoning_at_boundaries():
    assert TicketAnalysis(**_valid_analysis(reasoning="x" * 10))
    assert TicketAnalysis(**_valid_analysis(reasoning="x" * 1000))


def test_missing_required_field():
    with pytest.raises(ValidationError):
        TicketAnalysis(**{k: v for k, v in _valid_analysis().items() if k != "suggested_response"})


def test_all_valid_categories():
    for category in ("billing", "technical_issue", "authentication", "feature_request", "general_question", "other"):
        assert TicketAnalysis(**_valid_analysis(category=category)).category == category


def test_all_valid_priorities():
    for priority in ("low", "medium", "high", "urgent"):
        assert TicketAnalysis(**_valid_analysis(priority=priority)).priority == priority


def test_all_valid_sentiments():
    for sentiment in ("positive", "neutral", "negative"):
        assert TicketAnalysis(**_valid_analysis(sentiment=sentiment)).sentiment == sentiment
