from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from ticket_analyzer.config import settings


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine = create_async_engine(settings.database_url)
_async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(_engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.

    The session is automatically closed when the request completes,
    whether it succeeds or raises an exception.
    """
    async with _async_session() as session:
        yield session


async def dispose_engine() -> None:
    """
    Dispose the engine connection pool.

    Called on app shutdown to cleanly close all open connections.
    """
    await _engine.dispose()
