import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import verify_api_key
from app.persistence import crud
from app.persistence.database import get_db
from app.questions import registry
from app.questions.schemas import QuestionCreate, QuestionListItem, QuestionResponse
from app.services.forecast import run_forecast

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/questions", tags=["questions"])


class CreateQuestionResponse(BaseModel):
    id: str
    status: str
    run_id: str


class RunResponse(BaseModel):
    run_id: str
    status: str


class ResolveRequest(BaseModel):
    outcome: bool


@router.post("", response_model=CreateQuestionResponse, dependencies=[Depends(verify_api_key)])
async def create_question(
    body: QuestionCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> CreateQuestionResponse:
    qid = registry.question_id(body.text)
    domain_tags = registry.tag_domains(body.text)

    await crud.upsert_question(
        db=db,
        id=qid,
        text=body.text,
        resolution_criteria=body.resolution_criteria,
        resolution_deadline=body.resolution_deadline,
        domain_tags=domain_tags,
    )

    run_id = str(uuid.uuid4())
    background_tasks.add_task(_run_forecast_bg, question_id=qid)

    logger.info("question_created", question_id=qid, run_id=run_id)
    return CreateQuestionResponse(id=qid, status="queued", run_id=run_id)


async def _run_forecast_bg(question_id: str) -> None:
    from app.persistence.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        await run_forecast(question_id=question_id, db=db)


@router.get("", response_model=list[QuestionListItem], dependencies=[Depends(verify_api_key)])
async def list_questions(
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[QuestionListItem]:
    questions = await crud.list_questions(db, status=status)
    result: list[QuestionListItem] = []
    for q in questions:
        snapshot = await crud.get_latest_snapshot(db, q.id)
        result.append(
            QuestionListItem(
                id=q.id,
                text=q.text,
                today_forecast=snapshot.today_forecast if snapshot else None,
                run_at=snapshot.run_at if snapshot else None,
                status=q.status,
            )
        )
    return result


class PersonaBreakdownItem(BaseModel):
    persona_id: str
    point_estimate: float
    confidence_interval: list[float] | None
    reasoning_chain: list[str] | None
    key_cruxes: list[str] | None
    update_direction: str | None


class QuestionDetail(BaseModel):
    id: str
    text: str
    resolution_criteria: str
    resolution_deadline: datetime | None
    domain_tags: list[str] | None
    status: str
    created_at: datetime
    today_forecast: int | None
    run_at: datetime | None
    run_status: str | None
    spread: float | None
    persona_breakdown: list[PersonaBreakdownItem]


@router.get("/{question_id}", response_model=QuestionDetail, dependencies=[Depends(verify_api_key)])
async def get_question(
    question_id: str,
    db: AsyncSession = Depends(get_db),
) -> QuestionDetail:
    question = await crud.get_question(db, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found", headers={"code": "not_found"})

    snapshot = await crud.get_latest_snapshot(db, question_id)
    persona_breakdown: list[PersonaBreakdownItem] = []
    if snapshot:
        outputs = await crud.list_persona_outputs(db, snapshot.id)
        persona_breakdown = [
            PersonaBreakdownItem(
                persona_id=po.persona_id,
                point_estimate=po.point_estimate,
                confidence_interval=po.confidence_interval,
                reasoning_chain=po.reasoning_chain,
                key_cruxes=po.key_cruxes,
                update_direction=po.update_direction,
            )
            for po in outputs
        ]

    return QuestionDetail(
        id=question.id,
        text=question.text,
        resolution_criteria=question.resolution_criteria,
        resolution_deadline=question.resolution_deadline,
        domain_tags=question.domain_tags,
        status=question.status,
        created_at=question.created_at,
        today_forecast=snapshot.today_forecast if snapshot else None,
        run_at=snapshot.run_at if snapshot else None,
        run_status=snapshot.run_status if snapshot else None,
        spread=snapshot.spread if snapshot else None,
        persona_breakdown=persona_breakdown,
    )


@router.post("/{question_id}/run", response_model=RunResponse, dependencies=[Depends(verify_api_key)])
async def trigger_run(
    question_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> RunResponse:
    question = await crud.get_question(db, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    run_id = str(uuid.uuid4())
    background_tasks.add_task(_run_forecast_bg, question_id=question_id)
    logger.info("manual_run_triggered", question_id=question_id, run_id=run_id)
    return RunResponse(run_id=run_id, status="queued")


@router.post("/{question_id}/resolve", dependencies=[Depends(verify_api_key)])
async def resolve_question(
    question_id: str,
    body: ResolveRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    question = await crud.get_question(db, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    snapshot = await crud.get_latest_snapshot(db, question_id)
    aggregate_brier: float | None = None
    persona_brier_scores: dict | None = None

    if snapshot and snapshot.run_status == "ok":
        # Brier score = (forecast/100 - outcome)^2
        forecast_p = snapshot.today_forecast / 100.0
        outcome_val = 1.0 if body.outcome else 0.0
        aggregate_brier = (forecast_p - outcome_val) ** 2

        outputs = await crud.list_persona_outputs(db, snapshot.id)
        persona_brier_scores = {
            po.persona_id: (po.point_estimate - outcome_val) ** 2 for po in outputs
        }

    await crud.create_resolution(
        db=db,
        question_id=question_id,
        outcome=body.outcome,
        aggregate_brier=aggregate_brier,
        persona_brier_scores=persona_brier_scores,
    )
    await crud.update_question_status(db, question_id, "resolved")

    logger.info("question_resolved", question_id=question_id, outcome=body.outcome)
    return {"status": "resolved", "aggregate_brier": aggregate_brier}
