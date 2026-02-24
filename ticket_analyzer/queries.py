import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ticket_analyzer.llm import LLMResult
from ticket_analyzer.orm import TicketClassification
from ticket_analyzer.domain import TicketAnalysis
from ticket_analyzer.http import TicketRequest


async def save(
    session: AsyncSession,
    request: TicketRequest,
    analysis: TicketAnalysis,
    llm_result: LLMResult,
) -> TicketClassification:
    record = TicketClassification(
        ticket_id=request.ticket_id,
        text=request.text,
        category=analysis.category,
        priority=analysis.priority,
        sentiment=analysis.sentiment,
        confidence=analysis.confidence,
        suggested_response=analysis.suggested_response,
        reasoning=analysis.reasoning,
        model=llm_result.model,
        latency_ms=llm_result.latency_ms,
        input_tokens=llm_result.input_tokens,
        output_tokens=llm_result.output_tokens,
    )
    session.add(record)
    await session.commit()
    return record


async def get_all(session: AsyncSession) -> list[TicketClassification]:
    result = await session.execute(
        select(TicketClassification).order_by(TicketClassification.created_at.desc())
    )
    return list(result.scalars().all())


async def get_by_id(session: AsyncSession, id: uuid.UUID) -> TicketClassification | None:
    return await session.get(TicketClassification, id)
