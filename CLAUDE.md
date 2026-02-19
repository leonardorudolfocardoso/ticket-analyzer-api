# Ticket Analyzer API

AI-powered FastAPI service that classifies support tickets using LLMs and returns strictly validated JSON responses.

## Commands

```bash
# Run dev server
.venv/bin/fastapi dev ticket_analyzer/main.py

# Run tests
.venv/bin/pytest tests/ -v

# Test the API
curl -s -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice this month.", "ticket_id": "TKT-001"}' \
  | python3 -m json.tool
```

## Project Structure

```
ticket_analyzer/
├── main.py         # App factory, logging setup, health endpoint
├── config.py       # Pydantic Settings (reads from .env)
├── schemas.py      # All Pydantic models — the domain contract
├── classifier.py   # ClassifierService: retry loop, validation
├── routes.py       # POST /api/v1/analyze
└── llm/
    ├── __init__.py # LLMProvider protocol + get_provider()
    └── openai.py   # OpenAI implementation
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your key.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `LLM_PROVIDER` | `openai` | Provider to use |
| `LLM_MODEL` | `gpt-4o-mini` | Model identifier |
| `MAX_RETRIES` | `3` | Retry attempts on validation failure |
| `LOG_LEVEL` | `INFO` | Python log level |

## Architecture

### LLM Provider

Providers implement the `LLMProvider` protocol from `llm/__init__.py`:

```python
async def classify(self, system_prompt: str, user_text: str) -> str: ...
```

To add a new provider: create a class in `llm/` implementing `classify`, then add a branch in `get_provider()`.

### Retry Logic

`ClassifierService.analyze()` retries up to `MAX_RETRIES` times on `ValidationError`, `ValueError`, or `JSONDecodeError`. OpenAI-level errors (auth, rate limit, network) are not caught and surface immediately.

### Validation

Two layers of enforcement:
1. `response_format={"type": "json_object"}` at the OpenAI API level
2. `TicketAnalysis.model_validate_json()` via Pydantic — catches invalid enums, out-of-range confidence, missing fields

### Response Schema

| Field | Type | Values |
|---|---|---|
| `category` | Literal | `billing`, `technical_issue`, `authentication`, `feature_request`, `general_question`, `other` |
| `priority` | Literal | `low`, `medium`, `high`, `urgent` |
| `sentiment` | Literal | `positive`, `neutral`, `negative` |
| `confidence` | float | 0.0 – 1.0 |
| `suggested_response` | str | Draft reply for the support agent |
| `reasoning` | str | Explanation of the classification |

## Testing

```bash
.venv/bin/pytest tests/ -v
```

- `test_schemas.py` — Pydantic constraint validation
- `test_classifier.py` — Retry logic with a mocked provider (no real API calls)
- `test_routes.py` — Full request/response cycle via ASGI test client

## Conventions

- Python 3.10+ union syntax (`str | None`)
- Private attributes prefixed with `_`
- Log format: `"Action | key=value key2=value2"`
- HTTP 422 for input validation errors, 502 for exhausted retries
