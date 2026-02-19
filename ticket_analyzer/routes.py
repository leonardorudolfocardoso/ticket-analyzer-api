import logging

from fastapi import APIRouter, HTTPException

from ticket_analyzer.classifier import ClassifierService
from ticket_analyzer.schemas import AnalyzeResponse, TicketRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse, status_code=200)
async def analyze_ticket(request: TicketRequest) -> AnalyzeResponse:
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
        analysis = await ClassifierService().analyze(request)
    except RuntimeError as exc:
        logger.error("Classification failed", extra={"error": str(exc)})
        raise HTTPException(status_code=502, detail="Classification failed. Please try again.") from exc
    return AnalyzeResponse(ticket_id=request.ticket_id, analysis=analysis)
