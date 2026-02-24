from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "billing",
    "technical_issue",
    "authentication",
    "feature_request",
    "general_question",
    "other",
]

Priority = Literal["low", "medium", "high", "urgent"]

Sentiment = Literal["positive", "neutral", "negative"]


class TicketAnalysis(BaseModel):
    category: Category = Field(
        ...,
        description="The primary category of the support ticket.",
    )
    priority: Priority = Field(
        ...,
        description="Urgency level inferred from ticket content and sentiment.",
    )
    sentiment: Sentiment = Field(
        ...,
        description="Overall emotional tone of the ticket.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the classification (0.0–1.0).",
    )
    suggested_response: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="A draft reply the support agent can send to the customer.",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Brief explanation of why this classification was chosen.",
    )
