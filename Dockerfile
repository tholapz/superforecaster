FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache .

COPY . .

ENV APP_MODE=api

CMD ["sh", "-c", "if [ \"$APP_MODE\" = 'worker' ]; then python -m app.scheduler.jobs; else uvicorn app.main:app --host 0.0.0.0 --port 8000; fi"]
