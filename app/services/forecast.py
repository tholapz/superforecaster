import uuid
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.aggregation import engine as aggregation_engine
from app.config import settings
from app.context.retriever import fetch_context
from app.persistence import crud
from app.personas.orchestrator import run_personas

logger = structlog.get_logger(__name__)


async def run_forecast(question_id: str, db: AsyncSession) -> None:
    """Execute a full forecast run for the given question."""
    run_id = str(uuid.uuid4())
    log = logger.bind(question_id=question_id, run_id=run_id)

    question = await crud.get_question(db, question_id)
    if question is None:
        log.error("question_not_found")
        return

    log.info("forecast_run_start")

    # 1. Fetch web context
    context_digest, source_urls = await fetch_context(
        question.text, question.resolution_criteria
    )

    # 2. Build prior reasoning summaries for re-runs
    prior_reasoning_summaries: str | None = None
    latest_snapshot = await crud.get_latest_snapshot(db, question_id)
    if latest_snapshot is not None:
        prior_outputs = await crud.list_persona_outputs(db, latest_snapshot.id)
        if prior_outputs:
            summaries = []
            for po in prior_outputs:
                if po.reasoning_chain:
                    summaries.append(f"[{po.persona_id}]: {' → '.join(po.reasoning_chain[:3])}")
            if summaries:
                prior_reasoning_summaries = "\n".join(summaries)

    # 3. Run all personas in parallel
    valid_outputs = await run_personas(
        question_text=question.text,
        resolution_criteria=question.resolution_criteria,
        context_digest=context_digest,
        prior_reasoning_summaries=prior_reasoning_summaries,
        run_id=run_id,
    )

    # 4. Quorum check
    if len(valid_outputs) < settings.min_quorum:
        log.warning(
            "insufficient_quorum",
            valid_count=len(valid_outputs),
            min_quorum=settings.min_quorum,
        )
        await crud.create_snapshot(
            db=db,
            question_id=question_id,
            today_forecast=0,
            raw_mean=0.0,
            extremized=0.0,
            spread=0.0,
            n_valid_personas=len(valid_outputs),
            persona_estimates=None,
            context_source_urls=source_urls or None,
            run_status="insufficient_quorum",
        )
        return

    # 5. Compute change metrics
    now = datetime.now(timezone.utc)
    change_1w: int | None = None
    change_30d: int | None = None

    snap_1w = await crud.get_snapshot_closest_to(db, question_id, now - timedelta(days=7))
    if snap_1w is not None and snap_1w.run_status == "ok":
        change_1w = None  # will compute after we have today_forecast

    snap_30d = await crud.get_snapshot_closest_to(db, question_id, now - timedelta(days=30))

    # 6. Aggregate
    result = aggregation_engine.compute(valid_outputs)

    # Compute change metrics now that we have today_forecast
    if snap_1w is not None and snap_1w.run_status == "ok":
        change_1w = result.today_forecast - snap_1w.today_forecast
    if snap_30d is not None and snap_30d.run_status == "ok":
        change_30d = result.today_forecast - snap_30d.today_forecast

    result.change_1w = change_1w
    result.change_30d = change_30d

    # 7. Write snapshot
    snapshot = await crud.create_snapshot(
        db=db,
        question_id=question_id,
        today_forecast=result.today_forecast,
        raw_mean=result.raw_mean,
        extremized=result.extremized,
        spread=result.spread,
        n_valid_personas=result.n_valid_personas,
        persona_estimates=result.persona_estimates,
        context_source_urls=source_urls or None,
        run_status="ok",
    )

    # 8. Write persona outputs
    for output in valid_outputs:
        await crud.create_persona_output(
            db=db,
            snapshot_id=snapshot.id,
            persona_id=output.persona_id,
            point_estimate=output.point_estimate,
            confidence_interval=list(output.confidence_interval),
            reasoning_chain=output.reasoning_chain,
            key_cruxes=output.key_cruxes,
            update_direction=output.update_direction,
            tokens_used=None,
        )

    log.info(
        "forecast_run_complete",
        today_forecast=result.today_forecast,
        n_valid_personas=result.n_valid_personas,
        change_1w=change_1w,
        change_30d=change_30d,
    )
