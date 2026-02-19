import pytest
from pydantic import ValidationError

from ticket_analyzer.schemas import TicketAnalysis, TicketRequest


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


class TestTicketRequest:
    def test_text_too_short(self):
        with pytest.raises(ValidationError):
            TicketRequest(text="Too short")

    def test_text_too_long(self):
        with pytest.raises(ValidationError):
            TicketRequest(text="x" * 5001)


class TestTicketAnalysis:
    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            TicketAnalysis(**_valid_analysis(category="unknown"))

    def test_invalid_priority(self):
        with pytest.raises(ValidationError):
            TicketAnalysis(**_valid_analysis(priority="critical"))

    def test_invalid_sentiment(self):
        with pytest.raises(ValidationError):
            TicketAnalysis(**_valid_analysis(sentiment="angry"))

    def test_confidence_below_zero(self):
        with pytest.raises(ValidationError):
            TicketAnalysis(**_valid_analysis(confidence=-0.1))

    def test_confidence_above_one(self):
        with pytest.raises(ValidationError):
            TicketAnalysis(**_valid_analysis(confidence=1.1))

    def test_confidence_boundary_values(self):
        assert TicketAnalysis(**_valid_analysis(confidence=0.0)).confidence == 0.0
        assert TicketAnalysis(**_valid_analysis(confidence=1.0)).confidence == 1.0
