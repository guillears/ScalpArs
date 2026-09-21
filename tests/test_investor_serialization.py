"""🔒 Sep-21 — investor writes are serialized read→mutate→COMMIT.

Investor/NAV endpoints are read-modify-write across separate sessions: two concurrent requests
could both read shares=3000, both compute, and last-writer-wins. For a TRANSFER that MINTS shares
rather than merely misreporting one row — the sender is debited once, the recipient credited twice.

SQLite has no row locking, and `get_db` commits AFTER the handler returns — outside anything the
handler itself holds — so a lock inside the handler body would not have covered the commit and the
next waiter could still read its own stale snapshot. `_investor_serialized` holds the lock across
the whole handler AND commits inside it.

Lock order is always _investor_lock -> db_write_lock (inside locked_commit), never the reverse.
"""
import asyncio

import pytest
from fastapi.routing import APIRoute
from sqlalchemy import func, select

import main
import models


@pytest.fixture(autouse=True)
def _fresh_lock():
    """asyncio.Lock binds to the event loop on FIRST USE, and pytest-asyncio gives every test a
    new loop — so a module-level lock touched by one test raises "bound to a different event loop"
    in the next. Production has a single loop for the app's lifetime, so this is a test-isolation
    fixture, not a hint that the lock is wrong. (It also guarantees no test inherits a lock another
    test left held.)"""
    main._investor_lock = asyncio.Lock()
    yield


def _mutating_investor_routes():
    return [r for r in main.app.routes
            if isinstance(r, APIRoute) and r.path.startswith("/api/investors")
            and r.methods & {"POST", "PATCH", "DELETE"}]


def test_every_mutating_investor_route_is_serialized():
    """A SCAN, not an allowlist — a new /api/investors write route is caught the day it lands.

    Asserts the explicit __investor_serialized__ marker, not __wrapped__: any functools.wraps
    decorator sets __wrapped__, so that proved nothing (deep review).
    """
    routes = _mutating_investor_routes()
    assert len(routes) >= 8, "route scan found fewer endpoints than expected — did paths change?"
    missing = [f"{list(r.methods)[0]} {r.path}" for r in routes
               if not getattr(r.endpoint, "__investor_serialized__", False)]
    assert not missing, f"NOT serialized: {missing}"


def test_decorator_preserves_the_signature_fastapi_reads():
    """functools.wraps must expose __wrapped__ or FastAPI resolves the body/db params wrongly."""
    import inspect
    route = next(r for r in main.app.routes
                 if isinstance(r, APIRoute) and r.path == "/api/investors/transfer")
    names = [p.name for p in inspect.signature(route.endpoint).parameters.values()]
    assert names == ["body", "db"]
    assert [f.name for f in route.dependant.body_params] == ["body"]


async def test_the_lock_actually_excludes(db):
    """Two coroutines through the decorator must not interleave inside the critical section."""
    inside = 0
    overlaps = 0

    @main._investor_serialized
    async def _critical(db):
        nonlocal inside, overlaps
        inside += 1
        if inside > 1:
            overlaps += 1
        await asyncio.sleep(0)          # hand control back — unguarded, the other would enter here
        inside -= 1

    await asyncio.gather(*[_critical(db=db) for _ in range(8)])
    assert overlaps == 0


async def test_without_the_lock_the_same_probe_DOES_overlap():
    """Proves the test above is not vacuous — the bare coroutine interleaves."""
    inside = 0
    overlaps = 0

    async def _critical():
        nonlocal inside, overlaps
        inside += 1
        if inside > 1:
            overlaps += 1
        await asyncio.sleep(0)
        inside -= 1

    await asyncio.gather(*[_critical() for _ in range(8)])
    assert overlaps > 0


async def test_missing_session_is_a_hard_error_not_a_skipped_commit(db):
    """THE critical regression. Finding the session by NAME meant an endpoint calling it
    `session` would be serialized but never committed inside the lock — silently back to the
    mint race with a green suite. Missing session must raise, never quietly skip."""
    @main._investor_serialized
    async def _no_session():
        return "ok"

    with pytest.raises(RuntimeError, match="no AsyncSession"):
        await _no_session()


async def test_session_is_found_by_type_not_by_the_name_db(db):
    """An endpoint naming it `session` must still get the in-lock commit."""
    seen = {}

    @main._investor_serialized
    async def _named_session(session):
        seen["locked"] = main._investor_lock.locked()
        return "ok"

    assert await _named_session(session=db) == "ok"
    assert seen["locked"] is True


async def test_preview_calls_skip_the_lock_entirely(db):
    """A read-only modal preview must not queue behind — or block — a real transfer."""
    from types import SimpleNamespace
    seen = {}

    @main._investor_serialized
    async def _preview(body, db):
        seen["locked"] = main._investor_lock.locked()
        return "ok"

    await _preview(body=SimpleNamespace(preview=True), db=db)
    assert seen["locked"] is False


async def test_lock_is_released_when_the_handler_raises(db):
    """A refused transfer (HTTP 400) must not wedge every later investor request."""
    @main._investor_serialized
    async def _boom(db):
        raise main.HTTPException(400, "nope")

    with pytest.raises(main.HTTPException):
        await _boom(db=db)
    assert not main._investor_lock.locked()


async def test_concurrent_transfers_on_separate_sessions_conserve_shares(monkeypatch):
    """THE race, end to end: two REAL transfers on two REAL sessions, gathered.

    Unguarded, both read the sender's shares before either writes and the recipient is credited
    twice — shares are MINTED. Serialized, the second sees the first's committed state. Needs a
    shared in-memory DB (StaticPool), since the default fixture yields a single session.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool

    async def _pv(_db):
        return 3000.0
    monkeypatch.setattr(main, "_get_portfolio_value", _pv)

    eng = create_async_engine("sqlite+aiosqlite://", future=True, poolclass=StaticPool,
                              connect_args={"check_same_thread": False})
    async with eng.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    Session = async_sessionmaker(eng, expire_on_commit=False)

    async with Session() as setup:
        setup.add_all([
            models.Investor(name="Src", shares=3000.0, total_deposited=3000.0, total_withdrawn=0.0),
            models.Investor(name="Dst", shares=0.0, total_deposited=0.0, total_withdrawn=0.0)])
        await setup.commit()
        rows = (await setup.execute(select(models.Investor))).scalars().all()
        a_id = next(r.id for r in rows if r.name == "Src")
        b_id = next(r.id for r in rows if r.name == "Dst")

    async def _one():
        async with Session() as s:
            try:
                await main.investor_transfer(
                    main.InvestorTransfer(from_investor_id=a_id, to_investor_id=b_id, amount=500.0),
                    db=s)
            except main.HTTPException:
                pass

    await asyncio.gather(_one(), _one())

    async with Session() as check:
        rows = {r.name: r for r in (await check.execute(select(models.Investor))).scalars().all()}
        total = sum(r.shares for r in rows.values())
        dep = sum(r.total_deposited for r in rows.values())
        # conservation holds even under the race (both writers set ABSOLUTE values from their own
        # read, so the loser is overwritten, not added) — so conservation alone proves nothing.
        assert float(total) == pytest.approx(3000.0, abs=1e-6)
        assert float(dep) == pytest.approx(3000.0, abs=1e-6)
        # THE real failure mode is a LOST UPDATE: unguarded, both read shares=3000, both compute
        # from it, and the second commit silently erases the first — one $500 transfer vanishes
        # with an HTTP 200. Both must land.
        assert rows["Dst"].total_deposited == pytest.approx(1000.0, abs=1e-6), \
            "a transfer was silently lost — both $500 moves must apply"
        assert rows["Dst"].shares == pytest.approx(2 * (500.0 / 1.0), abs=1e-6)
        assert rows["Src"].total_deposited == pytest.approx(2000.0, abs=1e-6)
    await eng.dispose()
