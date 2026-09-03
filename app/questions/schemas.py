from datetime import datetime

from pydantic import BaseModel, Field


class QuestionCreate(BaseModel):
    text: str = Field(..., min_length=10)
    resolution_criteria: str = Field(..., min_length=10)
    resolution_deadline: datetime | None = None


class QuestionResponse(BaseModel):
    id: str
    text: str
    resolution_criteria: str
    resolution_deadline: datetime | None
    domain_tags: list[str] | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class QuestionListItem(BaseModel):
    id: str
    text: str
    today_forecast: int | None
    run_at: datetime | None
    status: str

    model_config = {"from_attributes": True}
