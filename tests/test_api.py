"""Integration tests for the API using httpx.AsyncClient against a real test DB."""

import os
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.persistence.database import Base

# ---------------------------------------------------------------------------
# Test DB setup
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://sf:sf@localhost:5432/superforecaster_test",
)

# NullPool: pytest-asyncio gives each test its own event loop, and a pooled
# asyncpg connection cannot be reused across loops.
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


# Function-scoped: pytest-asyncio runs each test in its own event loop, so a
# session-scoped fixture would tear the schema down after the first test.
@pytest_asyncio.fixture(autouse=True)
async def create_test_tables() -> AsyncGenerator[None, None]:
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    # Patch settings to use test DB and a known API key
    with (
        patch("app.config.settings.database_url", TEST_DATABASE_URL),
        patch("app.config.settings.api_key", "test-key"),
        patch("app.persistence.database.engine", test_engine),
        patch("app.persistence.database.AsyncSessionLocal", TestSessionLocal),
        patch("app.scheduler.jobs.start_scheduler"),
        patch("app.scheduler.jobs.stop_scheduler"),
    ):
        from app.main import create_app

        test_app = create_app()
        transport = ASGITransport(app=test_app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"Authorization": "Bearer test-key"},
        ) as ac:
            yield ac


AUTH_HEADERS = {"Authorization": "Bearer test-key"}

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code in (200, 503)  # 503 if test DB unavailable
        data = resp.json()
        assert "status" in data
        assert "version" in data


class TestPostQuestion:
    @pytest.mark.asyncio
    async def test_create_question_returns_queued(self, client: AsyncClient) -> None:
        with (
            patch("app.api.questions._run_forecast_bg", new_callable=AsyncMock),
        ):
            resp = await client.post(
                "/v1/questions",
                json={
                    "text": "Will the Federal Reserve cut interest rates before 1 Jan 2027?",
                    "resolution_criteria": "The Fed announces at least one rate cut in an official FOMC statement.",
                    "resolution_deadline": "2027-01-01T00:00:00Z",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "id" in data
        assert "run_id" in data

    @pytest.mark.asyncio
    async def test_create_question_requires_auth(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/questions",
            json={
                "text": "Will X happen before Y date?",
                "resolution_criteria": "X happens if Z is publicly confirmed.",
            },
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_create_question_short_text_rejected(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/questions",
            json={
                "text": "Short?",
                "resolution_criteria": "Some criteria here.",
            },
        )
        assert resp.status_code == 422


class TestGetQuestion:
    @pytest.mark.asyncio
    async def test_get_nonexistent_question_returns_404(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/questions/nonexistent-id-abc123")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_question_returns_detail(self, client: AsyncClient) -> None:
        # First create a question
        with patch("app.api.questions._run_forecast_bg", new_callable=AsyncMock):
            create_resp = await client.post(
                "/v1/questions",
                json={
                    "text": "Will AI surpass human-level performance on all benchmarks by 2026?",
                    "resolution_criteria": "A single AI system exceeds human average on a standard benchmark suite.",
                },
            )
        assert create_resp.status_code == 200
        qid = create_resp.json()["id"]

        # Retrieve it
        get_resp = await client.get(f"/v1/questions/{qid}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["id"] == qid
        assert "text" in data
        assert "status" in data
        assert "persona_breakdown" in data


class TestHistoryChanges:
    @pytest.mark.asyncio
    async def test_changes_returns_none_for_no_history(self, client: AsyncClient) -> None:
        # Create a fresh question with no forecasts
        with patch("app.api.questions._run_forecast_bg", new_callable=AsyncMock):
            create_resp = await client.post(
                "/v1/questions",
                json={
                    "text": "Will there be a major earthquake in Japan before July 2026?",
                    "resolution_criteria": "Magnitude 7.0+ earthquake reported by USGS in Japan.",
                },
            )
        assert create_resp.status_code == 200
        qid = create_resp.json()["id"]

        resp = await client.get(f"/v1/questions/{qid}/history/changes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["today_forecast"] is None
        assert data["change_1w"] is None
        assert data["change_30d"] is None

    @pytest.mark.asyncio
    async def test_history_returns_empty_list_for_no_snapshots(self, client: AsyncClient) -> None:
        with patch("app.api.questions._run_forecast_bg", new_callable=AsyncMock):
            create_resp = await client.post(
                "/v1/questions",
                json={
                    "text": "Will the S&P 500 exceed 6000 by end of 2026?",
                    "resolution_criteria": "S&P 500 index closes above 6000 on any trading day.",
                },
            )
        qid = create_resp.json()["id"]

        resp = await client.get(f"/v1/questions/{qid}/history")
        assert resp.status_code == 200
        assert resp.json() == []


class TestAdminPersonas:
    @pytest.mark.asyncio
    async def test_list_personas(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/admin/personas")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 6
        persona_ids = {p["persona_id"] for p in data}
        assert "geopolitical_analyst" in persona_ids
        assert "base_rate_statistician" in persona_ids

    @pytest.mark.asyncio
    async def test_update_persona_weight(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/admin/personas/geopolitical_analyst/weight",
            json={"weight": 1.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["weight"] == 1.5

    @pytest.mark.asyncio
    async def test_update_nonexistent_persona_returns_404(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/v1/admin/personas/nonexistent_persona/weight",
            json={"weight": 1.0},
        )
        assert resp.status_code == 404
