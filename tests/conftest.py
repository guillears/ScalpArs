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
def _fresh_process_locks(monkeypatch):
    """🔒 (DECISION_LOG 340 review) every process-wide asyncio.Lock gets a FRESH instance per test. Python ≥ 3.10 binds a Lock to the event
    loop of its first CONTENDED acquire; tests run many asyncio.run() loops in one process, so a lock contended in one test (e.g. concurrent
    background inserts through database.locked_commit) raised "bound to a different event loop" in the next — an order-dependent leak.
    Names imported by value elsewhere (main._cache_lock) are replaced alongside their source."""
    import database as _db
    import services.trading_engine as _te
    monkeypatch.setattr(_db, "db_write_lock", asyncio.Lock())
    _cl = asyncio.Lock()
    monkeypatch.setattr(_te, "_cache_lock", _cl)
    monkeypatch.setattr(_te, "_close_lock", asyncio.Lock())
    _m = sys.modules.get("main")
    if _m is not None:
        if hasattr(_m, "_cache_lock"):
            monkeypatch.setattr(_m, "_cache_lock", _cl)
        for _n in ("_scan_lock", "_investor_lock", "_live_px_lock", "_PAPER_BOOK_LOCK"):
            if isinstance(getattr(_m, _n, None), asyncio.Lock):
                monkeypatch.setattr(_m, _n, asyncio.Lock())
    from services import mcap_service as _mcs
    if getattr(_mcs, "_save_lock", None) is not None:
        monkeypatch.setattr(_mcs, "_save_lock", None)


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
    import services.trading_engine as _te
    if request.node.get_closest_marker("willy_hold"):
        # 🔓 (DECISION_LOG 340) the hold's own tests pin the switch ON (trading_config.json runs it OFF since 2026-10-21); a test that
        # checks the OFF path monkeypatches _willy_global_hold_on / frenzy_willy_global_hold itself
        monkeypatch.setattr(_te.TradingEngine, "_willy_global_hold_on", lambda self: True)
        return
    from sqlalchemy.ext.asyncio import AsyncSession as _AS
    _real = _te.TradingEngine._willy_hold_state

    async def _narrow(self, db=None):
        if isinstance(db, _AS):
            return await _real(self, db)
        return False, None, None, None
    monkeypatch.setattr(_te.TradingEngine, "_willy_hold_state", _narrow)


@pytest.fixture(autouse=True)
def _mcap_isolated(request, monkeypatch, tmp_path):
    """🗄 Oct-8: services.mcap_service never fetches from the network or writes the real mcap_cache.json (repo dir / /opt/scalpars-data) in a
    test — the persist path is ALWAYS a per-test tmp file; request_pair / ensure_refresh / fetch_pair_now (cache-only) are no-ops unless the test is @pytest.mark.mcap_real."""
    from services import mcap_service as _mcs
    _p = str(tmp_path / "mcap_cache.json")
    monkeypatch.setattr(_mcs, "_path", lambda: _p)
    if request.node.get_closest_marker("mcap_real"):
        return
    monkeypatch.setattr(_mcs, "request_pair", lambda pair: False)
    monkeypatch.setattr(_mcs, "ensure_refresh", lambda pairs: None)

    async def _no_fetch(pair, timeout_s=20.0):   # 🏷 (311) the post-fill one-pair backfill never touches the network in a test
        return _mcs.get(pair)
    monkeypatch.setattr(_mcs, "fetch_pair_now", _no_fetch)


@pytest.fixture(autouse=True)
def _export_archive_isolated(tmp_path_factory, monkeypatch):
    """🗄 (DECISION_LOG 409 review) no test reads or writes the real exports_archive/: the archive and its Downloads trigger point at an empty
    temp dir, so a test globbing the real ~/Downloads sees exactly the pre-409 behaviour (Downloads only). Archive tests set their own."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import export_glob as _EG
    d = tmp_path_factory.mktemp("export_archive")
    monkeypatch.setattr(_EG, "DOWNLOADS", str(d / "downloads"))
    monkeypatch.setattr(_EG, "ARCHIVE_DIR", str(d / "archive"))
