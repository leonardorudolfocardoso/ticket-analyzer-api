DEV  = docker compose -f docker-compose.yml -f docker-compose.dev.yml
PROD = docker compose -f docker-compose.yml -f docker-compose.prod.yml
TEST = docker compose -f docker-compose.yml -f docker-compose.test.yml

# ── Development ───────────────────────────────────────────────────────────────

.PHONY: dev
dev:
	$(DEV) up

.PHONY: dev-build
dev-build:
	$(DEV) up --build

.PHONY: dev-down
dev-down:
	$(DEV) down

# ── Production ────────────────────────────────────────────────────────────────

.PHONY: prod
prod:
	$(PROD) up -d

.PHONY: prod-build
prod-build:
	$(PROD) up -d --build

.PHONY: prod-down
prod-down:
	$(PROD) down

# ── Test ──────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(TEST) run --rm api

.PHONY: test-build
test-build:
	$(TEST) run --rm --build api

.PHONY: test-local
test-local:
	.venv/bin/pytest tests/ -v

# ── Shared ────────────────────────────────────────────────────────────────────

.PHONY: logs
logs:
	$(DEV) logs -f
