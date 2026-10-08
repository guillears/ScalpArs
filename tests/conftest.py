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


@pytest.fixture(autouse=True)
def _no_display_price_network(monkeypatch):
    """💹 Oct-7 (DECISION_LOG 245): /api/pairs' live display price never reaches the network in tests (and no real map stays cached)."""
    from services import binance_service as _bs

    async def _none(*a, **k):
        return None
    monkeypatch.setattr(_bs.binance_service, "fetch_all_prices", _none)
    _m = sys.modules.get("main")
    if _m is not None and hasattr(_m, "_live_px"):
        _m._live_px.update(t=0.0, px={})
        _m._live_px["try"] = 0.0


@pytest.fixture(autouse=True)
def _willy_hold_free(request, monkeypatch):
    """🔒 Oct-8 (DECISION_LOG 251): the FRENZY_WILLY global hold reads the open-orders cache + the DB and REFUSES when it cannot (fail-closed).
    Older tests drive the open paths with STUB db objects that cannot answer that read — for those (and for db=None, which would reach the
    real AsyncSessionLocal) the hold reads 'no WILLY open'. A real SQLAlchemy AsyncSession (in-memory SQLite) always gets the REAL check, and
    tests marked @pytest.mark.willy_hold get the real check whatever the db (tests/test_frenzy_willy.py)."""
    if request.node.get_closest_marker("willy_hold"):
        return
    import services.trading_engine as _te
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    _real = _te.TradingEngine._willy_hold_state

    async def _narrow(self, db=None):
        if isinstance(db, _AS):
            return await _real(self, db)
        return False, None, None, None
    monkeypatch.setattr(_te.TradingEngine, "_willy_hold_state", _narrow)
