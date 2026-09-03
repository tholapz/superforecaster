import asyncio

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings

logger = structlog.get_logger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _rerun_open_questions() -> None:
    from app.persistence.crud import list_questions
    from app.persistence.database import AsyncSessionLocal
    from app.services.forecast import run_forecast

    async with AsyncSessionLocal() as db:
        questions = await list_questions(db, status="open")

    logger.info("scheduler_rerun_start", count=len(questions))

    for question in questions:
        async with AsyncSessionLocal() as db:
            await run_forecast(question_id=question.id, db=db)
            logger.info("scheduler_rerun_done", question_id=question.id)


def start_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        return

    _scheduler = AsyncIOScheduler()
    _scheduler.add_job(
        _rerun_open_questions,
        trigger="interval",
        hours=settings.rerun_interval_hours,
        id="rerun_open_questions",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("scheduler_started", interval_hours=settings.rerun_interval_hours)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("scheduler_stopped")


# Entry point for worker mode: `python -m app.scheduler.jobs`
if __name__ == "__main__":
    from app.utils.logging import configure_logging

    configure_logging()

    async def main() -> None:
        start_scheduler()
        logger.info("worker_running")
        # Keep alive
        while True:
            await asyncio.sleep(60)

    asyncio.run(main())
