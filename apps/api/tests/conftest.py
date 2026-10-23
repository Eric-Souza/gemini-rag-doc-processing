"""Test setup: point the app at a separate `<db>_test` database before anything imports app.db."""

import os

import asyncpg
import pytest
from sqlalchemy.engine import make_url

from app.config import Settings, get_settings

_base_url = make_url(Settings().database_url)
TEST_DB = f"{_base_url.database}_test"
os.environ["DATABASE_URL"] = _base_url.set(database=TEST_DB).render_as_string(hide_password=False)
get_settings.cache_clear()

EMBED_DIM = get_settings().embed_dim


async def _create_test_database() -> None:
    conn = await asyncpg.connect(
        user=_base_url.username,
        password=_base_url.password,
        host=_base_url.host,
        port=_base_url.port,
        database=_base_url.database,
    )
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session", autouse=True)
async def database():
    await _create_test_database()
    from app.db import Base, engine, init_db

    await init_db()
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def axis_vector(*weights: tuple[int, float]) -> list[float]:
    """A unit-ish embedding with weight on chosen axes, for predictable cosine similarity."""
    vec = [0.0] * EMBED_DIM
    for axis, weight in weights:
        vec[axis] = weight
    return vec
