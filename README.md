# Ticket Analyzer API

AI-powered FastAPI service that classifies customer support tickets using LLMs. Returns structured JSON with category, priority, sentiment, confidence, a suggested response, and reasoning.

## Motivation

Customer support teams often struggle to triage large volumes of tickets quickly.
This project explores how LLM-based systems can assist support operations by automatically
classifying ticket urgency and routing them to the appropriate team.

## Features

- LLM-powered ticket classification via OpenAI
- Strict two-layer validation (API-level JSON enforcement + Pydantic)
- Automatic retry on malformed model output
- Persistent storage of all classifications in PostgreSQL
- REST endpoints to query historical results

## Requirements

- Python 3.10+
- Docker + Docker Compose (for containerised workflows)
- An OpenAI API key

## Quick start

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY and DATABASE_URL in .env

make dev-build   # build and start API + DB in development mode
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `DATABASE_URL` | — | Async SQLAlchemy URL, e.g. `postgresql+asyncpg://...` |
| `LLM_PROVIDER` | `openai` | Provider identifier |
| `LLM_MODEL` | `gpt-4o-mini` | Model to use for classification |
| `MAX_RETRIES` | `3` | Retry attempts on validation failure |
| `LOG_LEVEL` | `INFO` | Python log level |

## Examples

Classify a ticket:

```bash
curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice this month.", "ticket_id": "TKT-001"}' \
  | python3 -m json.tool
```

```json
{
  "ticket_id": "TKT-001",
  "analysis": {
    "category": "billing",
    "priority": "high",
    "sentiment": "negative",
    "confidence": 0.95,
    "suggested_response": "We're sorry to hear about the duplicate charge. We'll investigate and issue a refund within 3–5 business days.",
    "reasoning": "Customer reports an unexpected duplicate billing event, which is a high-priority billing issue requiring immediate attention."
  }
}
```

## Development

```bash
make dev-build   # start with live reload and source bind-mount
make dev         # start without rebuilding
make dev-down    # stop
make logs        # tail container logs
```

## Testing

```bash
make test-build        # run pytest in Docker (isolated test DB)
make test              # run without rebuilding
make test-watch        # watch mode in Docker (reruns on file changes)
make test-local        # run locally with .venv/bin/pytest
```

No real API calls are made in tests — the LLM provider is mocked.

## Project structure

```
ticket_analyzer/
├── main.py       # App factory and lifespan
├── config.py     # Settings (reads from .env)
├── routes.py     # HTTP route handlers
├── domain.py     # Core types: Category, Priority, Sentiment, TicketAnalysis
├── http.py       # HTTP shapes: TicketRequest, AnalyzeResponse, TicketClassificationRecord
├── orm.py        # SQLAlchemy ORM table definition
├── queries.py    # Database query functions
├── database.py   # Engine, session factory, get_db dependency
└── llm/
    ├── __init__.py  # analyze, retry, ClassifyFn protocol, get_provider
    └── openai.py    # OpenAI implementation
```

## Adding a new LLM provider

1. Create `ticket_analyzer/llm/<provider>.py` with a plain async function:
   ```python
   async def classify(system_prompt: str, user_text: str) -> LLMResult: ...
   ```
2. Add a branch in `get_classify_fn()` in `llm/__init__.py`.
3. Set `LLM_PROVIDER=<provider>` in your `.env`.
