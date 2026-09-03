from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api import admin, history, questions
from app.config import settings
from app.scheduler.jobs import start_scheduler, stop_scheduler
from app.utils.logging import configure_logging

configure_logging()

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("service_starting", app_name=settings.app_name, environment=settings.environment)
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("service_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Superforecaster",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.include_router(questions.router)
    app.include_router(history.router)
    app.include_router(admin.router)

    @app.get("/health", tags=["health"])
    async def health() -> JSONResponse:
        from sqlalchemy import text

        from app.persistence.database import AsyncSessionLocal

        db_status = "ok"
        try:
            async with AsyncSessionLocal() as db:
                await db.execute(text("SELECT 1"))
        except Exception:
            db_status = "error"

        return JSONResponse(
            content={"status": "ok", "db": db_status, "version": "0.1.0"},
            status_code=200 if db_status == "ok" else 503,
        )

    return app


app = create_app()
