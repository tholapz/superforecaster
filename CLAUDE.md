# CLAUDE.md — AI Superforecaster System

This file is the authoritative project brief for Claude Code. Read it in full before making any changes.

---

## Project Overview

A Python microservice that emulates the multi-analyst ensemble methodology from Barbara Mellers and Philip Tetlock's Good Judgment Project (GJP). A panel of LLM-backed superforecaster personas independently estimates the probability of a given binary question, then an aggregation engine combines those estimates into a single calibrated forecast (0–100%).

The service tracks forecast history over time and exposes change metrics (e.g., 1-week Δ, 30-day Δ).

**Primary reference:** Mellers et al. (2014). *Psychological Strategies for Winning a Geopolitical Forecasting Tournament.* Psychological Science, 25(5), 1106–1115. DOI: 10.1177/0956797614524255

**Aggregation reference:** Satopää et al. (2014). *Combining Multiple Probability Predictions Using a Simple Logit Model.* Management Science, 60(2), 444–457. DOI: 10.1287/mnsc.2013.1761

---

## V1 Scope (Docker-deployable)

The following is the complete scope for the first deployable version. Do not implement anything outside this list without explicit instruction.

### In scope
- FastAPI HTTP service with Pydantic v2 validation
- 6 fixed superforecaster personas (see Persona Definitions below)
- Parallel async persona reasoning via `asyncio.gather()`
- Web context retrieval using Claude tool_use + web search before each forecast run
- Aggregation engine: trimmed weighted mean on log-odds scale + extremization
- PostgreSQL persistence: questions, forecast snapshots, persona outputs
- Scheduled re-runs of open questions via APScheduler (default: every 24 hours)
- REST API (see API Specification below)
- Docker image + docker-compose for local and production deployment
- Alembic migrations
- `.env`-based configuration

### Out of scope for V1
- Brier-score-based auto weight recalibration
- WebSocket streaming of persona reasoning
- Batch question submission
- Confidence interval propagation through aggregation
- Persona debate rounds
- Cost tracking endpoint
- Metaculus export

---

## Repository Layout

```
superforecaster/
├── CLAUDE.md
├── README.md
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── pyproject.toml
├── alembic.ini
├── alembic/
│   └── versions/
├── app/
│   ├── main.py               # FastAPI app factory
│   ├── config.py             # Settings (pydantic-settings)
│   ├── dependencies.py       # Shared FastAPI dependencies
│   ├── api/
│   │   ├── __init__.py
│   │   ├── questions.py      # Question CRUD + forecast trigger routes
│   │   ├── history.py        # Forecast history + change metrics routes
│   │   └── admin.py          # Persona weight management routes
│   ├── questions/
│   │   ├── __init__.py
│   │   ├── registry.py       # Question normalization, dedup, domain tagging
│   │   └── schemas.py        # Pydantic schemas for Question
│   ├── context/
│   │   ├── __init__.py
│   │   └── retriever.py      # Web search via Claude tool_use; returns digest
│   ├── personas/
│   │   ├── __init__.py
│   │   ├── definitions.py    # System prompts and weights for all 6 personas
│   │   ├── orchestrator.py   # Fan-out, gather, validate persona outputs
│   │   └── schemas.py        # Pydantic schema for PersonaOutput
│   ├── aggregation/
│   │   ├── __init__.py
│   │   └── engine.py         # Trim → weighted log-odds mean → extremize
│   ├── persistence/
│   │   ├── __init__.py
│   │   ├── database.py       # SQLAlchemy async engine + session factory
│   │   ├── models.py         # ORM models
│   │   └── crud.py           # DB read/write helpers
│   └── scheduler/
│       ├── __init__.py
│       └── jobs.py           # APScheduler setup; rerun_open_questions job
└── tests/
    ├── test_aggregation.py
    ├── test_personas.py
    └── test_api.py
```

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| HTTP framework | FastAPI | ≥0.111 |
| Validation | Pydantic v2 | ≥2.7 |
| ASGI server | Uvicorn | ≥0.29 |
| LLM client | anthropic (Python SDK) | ≥0.28 |
| ORM | SQLAlchemy (async) | ≥2.0 |
| Migrations | Alembic | ≥1.13 |
| Database | PostgreSQL | ≥15 |
| Scheduler | APScheduler | ≥3.10 |
| Math | numpy, scipy | latest stable |
| Settings | pydantic-settings | ≥2.2 |
| Testing | pytest, pytest-asyncio, httpx | latest stable |

Python version: **3.12**

---

## Configuration

All configuration is read from environment variables. Never hardcode secrets.

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/superforecaster

# LLM
LLM_MODEL=claude-sonnet-4-20250514
CONTEXT_MAX_TOKENS=1500

# Forecasting
PERSONA_COUNT=6
TRIM_ALPHA=0.10
EXTREMIZE_BETA=2.5
MIN_QUORUM=3

# Scheduler
RERUN_INTERVAL_HOURS=24

# API
API_KEY=changeme
```

`app/config.py` must use `pydantic-settings` `BaseSettings` to load these. Provide an `.env.example` with all keys and no values.

---

## Persona Definitions

Defined in `app/personas/definitions.py` as a list of `PersonaConfig` dataclasses. All 6 are always active in V1.

| ID | Name | Primary lens | Bias to counteract | Weight |
|---|---|---|---|---|
| `geopolitical_analyst` | Geopolitical Analyst | Actor incentives, historical precedent | Narrative fallacy, availability bias | 1.0 |
| `base_rate_statistician` | Base-Rate Statistician | Reference class frequency, outside view | Inside-view overconfidence | 1.2 |
| `scenario_planner` | Scenario Planner | Structural drivers, contingencies | Anchoring on current state | 0.9 |
| `bayesian_updater` | Bayesian Updater | Prior probability + evidence strength | Conservatism / under-updating | 1.1 |
| `devils_advocate` | Devil's Advocate | Best case for opposite outcome | Confirmation bias | 0.8 |
| `domain_specialist` | Domain Specialist | Technical domain facts | Domain overconfidence | 1.0 |

### Persona system prompt structure

Each persona system prompt must contain exactly these four sections:

1. **Role and epistemic identity** — who this forecaster is, what they prioritize
2. **Reasoning process** — decompose into sub-questions, apply base rates, identify cruxes, practice active open-minded thinking (AOT per Mellers)
3. **Output format** — must produce a valid JSON object matching the schema below, no prose outside the JSON
4. **Calibration instruction** — explicitly counteract the listed bias; do not anchor on the previous persona's estimate

### Persona output schema

```python
class PersonaOutput(BaseModel):
    persona_id: str
    reasoning_chain: list[str]  # ordered reasoning steps, minimum 3
    point_estimate: float  # 0.0–1.0 inclusive
    confidence_interval: tuple[float, float]
    confidence_level: Literal["low", "medium", "high"]
    key_cruxes: list[str]  # 2–4 items
    update_direction: Literal["up", "down", "unchanged"]
```

Outputs that fail Pydantic validation or have `point_estimate` outside [0.0, 1.0] are discarded and logged as warnings. If fewer than `MIN_QUORUM` valid outputs remain after validation, the run is marked `INSUFFICIENT_QUORUM` and no snapshot is written.

---

## Aggregation Engine

Implemented in `app/aggregation/engine.py`. Three sequential operations:

### Step 1 — Outlier trim

Remove the top and bottom `TRIM_ALPHA` fraction of `point_estimate` values. With 6 personas and α=0.10, this removes the single most extreme estimate on each tail.

### Step 2 — Weighted mean on log-odds scale

```python
import numpy as np


def logit(p):
    return np.log(p / (1 - p))


def sigmoid(l):
    return 1 / (1 + np.exp(-l))


weighted_logit = np.average([logit(p) for p in estimates], weights=weights)
raw_mean = sigmoid(weighted_logit)
```

Clamp `point_estimate` to [0.001, 0.999] before logit to avoid ±inf.

### Step 3 — Extremization

```python
def extremize(p, beta):
    return p**beta / (p**beta + (1 - p) ** beta)


extremized = extremize(raw_mean, EXTREMIZE_BETA)
```

### Final output

```python
today_forecast: int  # round(extremized * 100), 0–100
raw_mean: float
extremized: float
n_valid_personas: int
spread: float  # std dev of trimmed point_estimate inputs
persona_estimates: list[float]
change_1w: int | None  # pp vs snapshot closest to now - 7 days; None if no history
change_30d: int | None
```

---

## Database Models

Defined in `app/persistence/models.py` using SQLAlchemy 2.0 declarative style with async session.

### `questions`

```
id                  TEXT PRIMARY KEY     -- SHA-256 of normalized question text
text                TEXT NOT NULL
resolution_criteria TEXT NOT NULL
resolution_deadline TIMESTAMPTZ
domain_tags         TEXT[]
status              TEXT DEFAULT 'open'  -- open | resolved | expired
created_at          TIMESTAMPTZ DEFAULT now()
```

### `forecast_snapshots`

```
id                   UUID PRIMARY KEY DEFAULT gen_random_uuid()
question_id          TEXT REFERENCES questions(id)
run_at               TIMESTAMPTZ DEFAULT now()
today_forecast       INT NOT NULL        -- 0–100
raw_mean             FLOAT NOT NULL
extremized           FLOAT NOT NULL
spread               FLOAT NOT NULL
n_valid_personas     INT NOT NULL
persona_estimates    FLOAT[]
context_source_urls  TEXT[]
run_status           TEXT DEFAULT 'ok'   -- ok | insufficient_quorum | error
```

### `persona_outputs`

```
id                   UUID PRIMARY KEY DEFAULT gen_random_uuid()
snapshot_id          UUID REFERENCES forecast_snapshots(id)
persona_id           TEXT NOT NULL
point_estimate       FLOAT NOT NULL
confidence_interval  FLOAT[2]
reasoning_chain      TEXT[]
key_cruxes           TEXT[]
update_direction     TEXT
tokens_used          INT
created_at           TIMESTAMPTZ DEFAULT now()
```

### `resolution`

```
question_id          TEXT PRIMARY KEY REFERENCES questions(id)
outcome              BOOLEAN NOT NULL
resolved_at          TIMESTAMPTZ DEFAULT now()
aggregate_brier      FLOAT               -- (forecast - outcome)^2
persona_brier_scores JSONB               -- {persona_id: brier_score}
```

---

## API Specification

Base path: `/v1`. All responses are JSON. Authentication: `Authorization: Bearer <API_KEY>` header.

### Questions

```
POST   /v1/questions
       Body: {text, resolution_criteria, resolution_deadline?}
       Returns: {id, status: "queued", run_id}
       Side-effect: triggers async forecast run

GET    /v1/questions
       Query: ?status=open|resolved
       Returns: list of {id, text, today_forecast, run_at, status}

GET    /v1/questions/{id}
       Returns: full question record + latest forecast + persona breakdown

POST   /v1/questions/{id}/run
       Triggers immediate re-run. Returns: {run_id, status: "queued"}

POST   /v1/questions/{id}/resolve
       Body: {outcome: true|false}
       Marks resolved, computes Brier scores.
```

### History

```
GET    /v1/questions/{id}/history
       Query: ?from=ISO8601&to=ISO8601
       Returns: list of {run_at, today_forecast, spread, n_valid_personas}

GET    /v1/questions/{id}/history/changes
       Returns: {today_forecast, change_1w, change_30d, run_at}
       Primary display payload for dashboard widgets.
```

### Admin

```
GET    /v1/admin/personas
       Returns: list of {persona_id, name, weight, description}

POST   /v1/admin/personas/{persona_id}/weight
       Body: {weight: float}
       Updates persona weight in memory (not persisted across restarts in V1).
```

### Health

```
GET    /health
       Returns: {status: "ok", db: "ok"|"error", version: str}
       No auth required.
```

---

## Forecasting Workflow (Request Flow)

```
POST /v1/questions  →  question_registry.normalize()
                    →  db: upsert question row
                    →  background_task: run_forecast(question_id)

run_forecast():
  1. context_retriever.fetch(question)        # web search via Claude tool_use
  2. asyncio.gather(*[persona(q, ctx) for persona in personas])
  3. validate each PersonaOutput via Pydantic
  4. if valid_count < MIN_QUORUM → write snapshot with run_status=insufficient_quorum; return
  5. aggregation_engine.compute(valid_outputs)
  6. db: write forecast_snapshot + persona_outputs
```

On **re-runs**, each persona additionally receives the previous run's `reasoning_chain` summaries (one per persona, concatenated) appended to the context to allow incremental belief updating.

---

## Docker

### Dockerfile

Single image, two run modes controlled by `APP_MODE` env var.

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

ENV APP_MODE=api

CMD ["sh", "-c", "if [ \"$APP_MODE\" = 'worker' ]; then python -m app.scheduler.jobs; else uvicorn app.main:app --host 0.0.0.0 --port 8000; fi"]
```

### docker-compose.yml

Three services: `db` (PostgreSQL 15), `api` (`APP_MODE=api`), `worker` (`APP_MODE=worker`).

```yaml
version: "3.9"
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: superforecaster
      POSTGRES_USER: sf
      POSTGRES_PASSWORD: sf
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "sf"]
      interval: 5s
      retries: 5

  api:
    build: .
    env_file: .env
    environment:
      APP_MODE: api
      DATABASE_URL: postgresql+asyncpg://sf:sf@db:5432/superforecaster
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    command: >
      sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"

  worker:
    build: .
    env_file: .env
    environment:
      APP_MODE: worker
      DATABASE_URL: postgresql+asyncpg://sf:sf@db:5432/superforecaster
    depends_on:
      db:
        condition: service_healthy

volumes:
  pgdata:
```

The `api` container runs Alembic migrations on startup before binding. This is acceptable for V1; use an init container or separate migration step for production.

---

## Error Handling Conventions

- All API errors return `{"detail": "...", "code": "..."}` with appropriate HTTP status codes.
- LLM call failures per persona are caught, logged at WARNING level, and that persona's output is discarded (not a fatal error).
- If the LLM returns malformed JSON, attempt one retry with an explicit correction prompt before discarding.
- Database errors are fatal and return HTTP 503 with `code: "db_error"`.
- Log structured JSON to stdout (`structlog` or equivalent). Include `question_id`, `run_id`, `persona_id` in all log entries.

---

## Testing

Minimum test coverage required before marking V1 complete:

- `test_aggregation.py`: unit tests for trim, weighted logit mean, and extremization with known inputs and expected outputs
- `test_personas.py`: mock the Anthropic client; assert that invalid persona JSON is discarded; assert quorum check triggers `insufficient_quorum`
- `test_api.py`: integration tests using `httpx.AsyncClient` against a test PostgreSQL DB (use `pytest-asyncio`); cover `POST /v1/questions`, `GET /v1/questions/{id}`, and `GET /v1/questions/{id}/history/changes`

Run: `pytest tests/ -v`

---

## Coding Standards

- Type-annotate all function signatures.
- Use `async def` throughout; no synchronous blocking calls in the request path.
- No business logic in route handlers — handlers call service functions only.
- All database access goes through `app/persistence/crud.py`.
- Do not use `print()` — use the configured logger.
- Format with `ruff format`; lint with `ruff check`.
- Keep each module under 300 lines; split if necessary.

---

## Quick Start (local dev)

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env

docker compose up --build
# api available at http://localhost:8000
# docs at http://localhost:8000/docs

# Submit a question
curl -X POST http://localhost:8000/v1/questions \
  -H "Authorization: Bearer changeme" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Will the PRC or Taiwan publicly accuse the other of using a weapon against its national military before 1 July 2026?",
    "resolution_criteria": "Public accusation by an official government source attributed to a specific incident.",
    "resolution_deadline": "2026-07-01T00:00:00Z"
  }'
```

---

## Key Design Decisions (do not change without discussion)

1. **Personas do not see each other's estimates before submitting.** This is a core GJP finding. Do not add any cross-persona communication before the aggregation step.
2. **Aggregation operates on log-odds, not raw probabilities.** This is non-negotiable for correct behavior near 0 and 1.
3. **Extremization is applied after the weighted mean, not before.** Order matters.
4. **Forecast snapshots are immutable.** Never update a written snapshot row. Corrections are new rows.
5. **The scheduler re-runs questions; it does not delete or merge history.** History is append-only.
