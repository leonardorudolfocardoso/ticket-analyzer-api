import uuid
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column

from ticket_analyzer.database import Base


class TicketClassification(Base):
    __tablename__ = "ticket_classifications"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    ticket_id: Mapped[str | None]
    text: Mapped[str]
    category: Mapped[str]
    priority: Mapped[str]
    sentiment: Mapped[str]
    confidence: Mapped[float]
    suggested_response: Mapped[str]
    reasoning: Mapped[str]
    model: Mapped[str]
    latency_ms: Mapped[int]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
