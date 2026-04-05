"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-05

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "questions",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("resolution_criteria", sa.Text(), nullable=False),
        sa.Column("resolution_deadline", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("domain_tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("status", sa.Text(), server_default="open", nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "forecast_snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("question_id", sa.Text(), sa.ForeignKey("questions.id"), nullable=False),
        sa.Column(
            "run_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("today_forecast", sa.Integer(), nullable=False),
        sa.Column("raw_mean", sa.Float(), nullable=False),
        sa.Column("extremized", sa.Float(), nullable=False),
        sa.Column("spread", sa.Float(), nullable=False),
        sa.Column("n_valid_personas", sa.Integer(), nullable=False),
        sa.Column("persona_estimates", postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column("context_source_urls", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("run_status", sa.Text(), server_default="ok", nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "persona_outputs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column(
            "snapshot_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("forecast_snapshots.id"),
            nullable=False,
        ),
        sa.Column("persona_id", sa.Text(), nullable=False),
        sa.Column("point_estimate", sa.Float(), nullable=False),
        sa.Column("confidence_interval", postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column("reasoning_chain", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("key_cruxes", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("update_direction", sa.Text(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "resolution",
        sa.Column("question_id", sa.Text(), sa.ForeignKey("questions.id"), nullable=False),
        sa.Column("outcome", sa.Boolean(), nullable=False),
        sa.Column(
            "resolved_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("aggregate_brier", sa.Float(), nullable=True),
        sa.Column("persona_brier_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("question_id"),
    )

    # Indexes for common queries
    op.create_index("ix_forecast_snapshots_question_id", "forecast_snapshots", ["question_id"])
    op.create_index("ix_forecast_snapshots_run_at", "forecast_snapshots", ["run_at"])
    op.create_index("ix_persona_outputs_snapshot_id", "persona_outputs", ["snapshot_id"])


def downgrade() -> None:
    op.drop_index("ix_persona_outputs_snapshot_id", table_name="persona_outputs")
    op.drop_index("ix_forecast_snapshots_run_at", table_name="forecast_snapshots")
    op.drop_index("ix_forecast_snapshots_question_id", table_name="forecast_snapshots")
    op.drop_table("resolution")
    op.drop_table("persona_outputs")
    op.drop_table("forecast_snapshots")
    op.drop_table("questions")
