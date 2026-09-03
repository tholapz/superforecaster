from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import verify_api_key
from app.persistence import crud
from app.persistence.database import get_db

router = APIRouter(prefix="/v1/questions", tags=["history"])


class HistoryItem(BaseModel):
    run_at: datetime
    today_forecast: int
    spread: float
    n_valid_personas: int


class ChangesResponse(BaseModel):
    today_forecast: int | None
    change_1w: int | None
    change_30d: int | None
    run_at: datetime | None


@router.get(
    "/{question_id}/history",
    response_model=list[HistoryItem],
    dependencies=[Depends(verify_api_key)],
)
async def get_history(
    question_id: str,
    from_: datetime | None = None,
    to: datetime | None = None,
    db: AsyncSession = Depends(get_db),
) -> list[HistoryItem]:
    question = await crud.get_question(db, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    snapshots = await crud.list_snapshots(db, question_id, from_dt=from_, to_dt=to)
    return [
        HistoryItem(
            run_at=s.run_at,
            today_forecast=s.today_forecast,
            spread=s.spread,
            n_valid_personas=s.n_valid_personas,
        )
        for s in snapshots
        if s.run_status == "ok"
    ]


@router.get(
    "/{question_id}/history/changes",
    response_model=ChangesResponse,
    dependencies=[Depends(verify_api_key)],
)
async def get_changes(
    question_id: str,
    db: AsyncSession = Depends(get_db),
) -> ChangesResponse:
    from datetime import timedelta

    question = await crud.get_question(db, question_id)
    if question is None:
        raise HTTPException(status_code=404, detail="Question not found")

    latest = await crud.get_latest_snapshot(db, question_id)
    if latest is None or latest.run_status != "ok":
        return ChangesResponse(today_forecast=None, change_1w=None, change_30d=None, run_at=None)

    now = datetime.now(UTC)
    change_1w: int | None = None
    change_30d: int | None = None

    snap_1w = await crud.get_snapshot_closest_to(db, question_id, now - timedelta(days=7))
    if snap_1w is not None and snap_1w.run_status == "ok" and snap_1w.id != latest.id:
        change_1w = latest.today_forecast - snap_1w.today_forecast

    snap_30d = await crud.get_snapshot_closest_to(db, question_id, now - timedelta(days=30))
    if snap_30d is not None and snap_30d.run_status == "ok" and snap_30d.id != latest.id:
        change_30d = latest.today_forecast - snap_30d.today_forecast

    return ChangesResponse(
        today_forecast=latest.today_forecast,
        change_1w=change_1w,
        change_30d=change_30d,
        run_at=latest.run_at,
    )
