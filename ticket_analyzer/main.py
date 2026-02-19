import logging

from fastapi import FastAPI

from ticket_analyzer.config import settings
from ticket_analyzer.routes import router


def create_app() -> FastAPI:
    logging.basicConfig(level=settings.log_level.upper())

    app = FastAPI(
        title="Ticket Analyzer API",
        description="LLM-powered support ticket classification service.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(router)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
