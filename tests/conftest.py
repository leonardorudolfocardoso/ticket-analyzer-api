from unittest.mock import AsyncMock, MagicMock

import pytest

from ticket_analyzer.database import get_db
from ticket_analyzer.main import app


@pytest.fixture(autouse=True)
def override_get_db():
    """
    Replace the get_db dependency with a no-op mock for all route tests.
    Repository calls are mocked per-test, so no real session is needed.
    """
    mock_session = MagicMock()

    async def _fake_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = _fake_get_db
    yield
    app.dependency_overrides.clear()
