import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ticket_analyzer.database import get_db
from ticket_analyzer.llm import analyze, get_provider
from ticket_analyzer import queries
from ticket_analyzer.http import (
    AnalyzeResponse,
    TicketClassificationRecord,
    TicketRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse, status_code=200)
async def analyze_ticket(
    request: TicketRequest,
    session: AsyncSession = Depends(get_db),
) -> AnalyzeResponse:
    """
    Accepts raw support ticket text and returns a structured classification.

    - **category**: billing | technical_issue | authentication | feature_request | general_question | other
    - **priority**: low | medium | high | urgent
    - **sentiment**: positive | neutral | negative
    - **confidence**: 0.0–1.0
    - **suggested_response**: draft reply for the support agent
    - **reasoning**: explanation of the classification decision
    """
    logger.info("POST /analyze", extra={"ticket_id": request.ticket_id})
    try:
        result = await analyze(request, classifier=get_provider())
    except RuntimeError as exc:
        logger.error("Classification failed", extra={"error": str(exc)})
        raise HTTPException(status_code=502, detail="Classification failed. Please try again.") from exc

    await queries.save(session, request, result.analysis, result.llm_result)

    return AnalyzeResponse(ticket_id=request.ticket_id, analysis=result.analysis)


@router.get("/tickets", response_model=list[TicketClassificationRecord], tags=["tickets"])
async def list_tickets(session: AsyncSession = Depends(get_db)) -> list[TicketClassificationRecord]:
    """Return all ticket classification records, newest first."""
    records = await queries.get_all(session)
    return [TicketClassificationRecord.model_validate(r) for r in records]


@router.get("/tickets/{id}", response_model=TicketClassificationRecord, tags=["tickets"])
async def get_ticket(id: uuid.UUID, session: AsyncSession = Depends(get_db)) -> TicketClassificationRecord:
    """Return a single ticket classification record by UUID."""
    record = await queries.get_by_id(session, id)
    if record is None:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    return TicketClassificationRecord.model_validate(record)
