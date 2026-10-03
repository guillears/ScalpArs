"""SCALPARS regression suite (Sep-7, operator-directed).

Scope: the SILENT-MATH money cores — the bug class every incident and review
finding in this project shares (code that runs clean but computes the wrong
number). Pure functions are tested directly; endpoint/DB logic runs against an
in-memory SQLite with the REAL models and monkeypatched exchange calls.

Run: venv/bin/pytest tests/ -q      (no network, no real DB, ~seconds)
Rule: green suite required before every commit (alongside compile + JS checks).
"""
import asyncio
import sys, os
os.environ.setdefault("SCALPARS_JOURNAL_OFF", "1")   # 📓 Sep-30: tests never write the local decision journal (journal tests opt back in)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

import models


@pytest_asyncio.fixture
async def db():
    """Fresh in-memory DB with the real schema per test."""
    eng = create_async_engine("sqlite+aiosqlite://", future=True)
    async with eng.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as session:
        yield session
    await eng.dispose()



@pytest.fixture(autouse=True)
def _no_orderbook_network(monkeypatch):
    """📖 Oct-3: manual opens read the order book (research stamp) — never the network in tests."""
    from services import binance_service as _bs

    async def _none(*a, **k):
        return None
    monkeypatch.setattr(_bs.binance_service, "fetch_orderbook_depth", _none, raising=False)
