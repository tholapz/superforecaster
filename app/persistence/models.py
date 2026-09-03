import uuid
from datetime import datetime

from sqlalchemy import ARRAY, Boolean, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.persistence.database import Base


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    resolution_criteria: Mapped[str] = mapped_column(Text, nullable=False)
    resolution_deadline: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    domain_tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    status: Mapped[str] = mapped_column(Text, default="open", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ForecastSnapshot(Base):
    __tablename__ = "forecast_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id: Mapped[str] = mapped_column(Text, ForeignKey("questions.id"), nullable=False)
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    today_forecast: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_mean: Mapped[float] = mapped_column(Float, nullable=False)
    extremized: Mapped[float] = mapped_column(Float, nullable=False)
    spread: Mapped[float] = mapped_column(Float, nullable=False)
    n_valid_personas: Mapped[int] = mapped_column(Integer, nullable=False)
    persona_estimates: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    context_source_urls: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    run_status: Mapped[str] = mapped_column(Text, default="ok", nullable=False)


class PersonaOutput(Base):
    __tablename__ = "persona_outputs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("forecast_snapshots.id"), nullable=False
    )
    persona_id: Mapped[str] = mapped_column(Text, nullable=False)
    point_estimate: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_interval: Mapped[list[float] | None] = mapped_column(ARRAY(Float), nullable=True)
    reasoning_chain: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    key_cruxes: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    update_direction: Mapped[str | None] = mapped_column(Text, nullable=True)
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Resolution(Base):
    __tablename__ = "resolution"

    question_id: Mapped[str] = mapped_column(Text, ForeignKey("questions.id"), primary_key=True)
    outcome: Mapped[bool] = mapped_column(Boolean, nullable=False)
    resolved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    aggregate_brier: Mapped[float | None] = mapped_column(Float, nullable=True)
    persona_brier_scores: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
