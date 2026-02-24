import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ticket_analyzer.domain import Category, Priority, Sentiment, TicketAnalysis


class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Raw customer support ticket text to be analyzed.",
        examples=["I can't log into my account. The password reset email never arrives."],
    )
    ticket_id: str | None = Field(
        default=None,
        description="Optional external ticket identifier for correlation.",
        examples=["TKT-00123"],
    )


class AnalyzeResponse(BaseModel):
    ticket_id: str | None = Field(
        default=None,
        description="Echoed from the request for correlation.",
    )
    analysis: TicketAnalysis


class TicketClassificationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticket_id: str | None
    text: str
    category: Category
    priority: Priority
    sentiment: Sentiment
    confidence: float
    suggested_response: str
    reasoning: str
    model: str
    latency_ms: int
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime
