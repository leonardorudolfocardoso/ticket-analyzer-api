# ---------- Builder (main deps only) ----------
FROM python:3.14-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y curl build-essential

RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --no-root --only main

# ---------- Test (main + dev deps) ----------
FROM builder AS test

COPY . .

RUN poetry install --no-interaction --no-ansi --only dev

CMD ["pytest", "tests/", "-v"]

# ---------- Final (production) ----------
FROM python:3.14-slim AS final

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.14 /usr/local/lib/python3.14
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "ticket_analyzer.main:app", "--bind", "0.0.0.0:8000"]
