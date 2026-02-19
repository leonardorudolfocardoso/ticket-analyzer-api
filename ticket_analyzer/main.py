import logging
import logging.config

from fastapi import FastAPI

from ticket_analyzer.config import settings
from ticket_analyzer.routes import router


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
            },
            "root": {
                "level": settings.log_level.upper(),
                "handlers": ["console"],
            },
        }
    )


def create_app() -> FastAPI:
    configure_logging()

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
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()
