import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.models import ForecastSnapshot, PersonaOutput, Question, Resolution

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


async def upsert_question(
    db: AsyncSession,
    id: str,
    text: str,
    resolution_criteria: str,
    resolution_deadline: datetime | None,
    domain_tags: list[str] | None,
) -> Question:
    result = await db.execute(select(Question).where(Question.id == id))
    question = result.scalar_one_or_none()
    if question is None:
        question = Question(
            id=id,
            text=text,
            resolution_criteria=resolution_criteria,
            resolution_deadline=resolution_deadline,
            domain_tags=domain_tags,
            status="open",
        )
        db.add(question)
    await db.commit()
    await db.refresh(question)
    return question


async def get_question(db: AsyncSession, question_id: str) -> Question | None:
    result = await db.execute(select(Question).where(Question.id == question_id))
    return result.scalar_one_or_none()


async def list_questions(db: AsyncSession, status: str | None = None) -> list[Question]:
    stmt = select(Question)
    if status:
        stmt = stmt.where(Question.status == status)
    stmt = stmt.order_by(Question.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_question_status(db: AsyncSession, question_id: str, status: str) -> None:
    result = await db.execute(select(Question).where(Question.id == question_id))
    question = result.scalar_one_or_none()
    if question:
        question.status = status
        await db.commit()


# ---------------------------------------------------------------------------
# Forecast snapshots
# ---------------------------------------------------------------------------


async def create_snapshot(
    db: AsyncSession,
    question_id: str,
    today_forecast: int,
    raw_mean: float,
    extremized: float,
    spread: float,
    n_valid_personas: int,
    persona_estimates: list[float] | None,
    context_source_urls: list[str] | None,
    run_status: str = "ok",
) -> ForecastSnapshot:
    snapshot = ForecastSnapshot(
        id=uuid.uuid4(),
        question_id=question_id,
        today_forecast=today_forecast,
        raw_mean=raw_mean,
        extremized=extremized,
        spread=spread,
        n_valid_personas=n_valid_personas,
        persona_estimates=persona_estimates,
        context_source_urls=context_source_urls,
        run_status=run_status,
    )
    db.add(snapshot)
    await db.commit()
    await db.refresh(snapshot)
    return snapshot


async def get_latest_snapshot(db: AsyncSession, question_id: str) -> ForecastSnapshot | None:
    result = await db.execute(
        select(ForecastSnapshot)
        .where(ForecastSnapshot.question_id == question_id)
        .order_by(ForecastSnapshot.run_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_snapshots(
    db: AsyncSession,
    question_id: str,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
) -> list[ForecastSnapshot]:
    stmt = (
        select(ForecastSnapshot)
        .where(ForecastSnapshot.question_id == question_id)
        .order_by(ForecastSnapshot.run_at.asc())
    )
    if from_dt:
        stmt = stmt.where(ForecastSnapshot.run_at >= from_dt)
    if to_dt:
        stmt = stmt.where(ForecastSnapshot.run_at <= to_dt)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_snapshot_closest_to(
    db: AsyncSession, question_id: str, target_dt: datetime
) -> ForecastSnapshot | None:
    result = await db.execute(
        select(ForecastSnapshot)
        .where(ForecastSnapshot.question_id == question_id)
        .where(ForecastSnapshot.run_status == "ok")
        .order_by(
            # absolute difference from target date
            (ForecastSnapshot.run_at - target_dt).desc()
            if False
            else ForecastSnapshot.run_at.desc()
        )
        .limit(100)
    )
    snapshots = list(result.scalars().all())
    if not snapshots:
        return None
    return min(snapshots, key=lambda s: abs((s.run_at.replace(tzinfo=timezone.utc) if s.run_at.tzinfo is None else s.run_at) - target_dt))


# ---------------------------------------------------------------------------
# Persona outputs
# ---------------------------------------------------------------------------


async def create_persona_output(
    db: AsyncSession,
    snapshot_id: uuid.UUID,
    persona_id: str,
    point_estimate: float,
    confidence_interval: list[float] | None,
    reasoning_chain: list[str] | None,
    key_cruxes: list[str] | None,
    update_direction: str | None,
    tokens_used: int | None,
) -> PersonaOutput:
    output = PersonaOutput(
        id=uuid.uuid4(),
        snapshot_id=snapshot_id,
        persona_id=persona_id,
        point_estimate=point_estimate,
        confidence_interval=confidence_interval,
        reasoning_chain=reasoning_chain,
        key_cruxes=key_cruxes,
        update_direction=update_direction,
        tokens_used=tokens_used,
    )
    db.add(output)
    await db.commit()
    await db.refresh(output)
    return output


async def list_persona_outputs(
    db: AsyncSession, snapshot_id: uuid.UUID
) -> list[PersonaOutput]:
    result = await db.execute(
        select(PersonaOutput).where(PersonaOutput.snapshot_id == snapshot_id)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


async def create_resolution(
    db: AsyncSession,
    question_id: str,
    outcome: bool,
    aggregate_brier: float | None,
    persona_brier_scores: dict | None,
) -> Resolution:
    resolution = Resolution(
        question_id=question_id,
        outcome=outcome,
        aggregate_brier=aggregate_brier,
        persona_brier_scores=persona_brier_scores,
    )
    db.add(resolution)
    await db.commit()
    await db.refresh(resolution)
    return resolution
