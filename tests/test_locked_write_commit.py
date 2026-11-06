"""DECISION_LOG 492 — locked_write_commit takes the fair lock BEFORE any write (no priority inversion),
commits inside it, and rolls back inside it on failure. The Frenzy flag-list save uses it."""
import asyncio
import inspect

import pytest

import database
from database import locked_write_commit


def _lock():
    return database.db_write_lock          # conftest swaps a fresh lock in per test — always read it live


class _FakeSession:
    def __init__(self, log, fail_commit=False):
        self.log = log
        self.fail_commit = fail_commit

    async def execute(self, stmt):
        self.log.append(("execute", _lock().locked()))

    def add_all(self, rows):
        self.log.append(("add_all", _lock().locked()))

    async def commit(self):
        self.log.append(("commit", _lock().locked()))
        if self.fail_commit:
            raise RuntimeError("database is locked")

    async def rollback(self):
        self.log.append(("rollback", _lock().locked()))


async def _work(s):
    await s.execute("DELETE")
    s.add_all([1, 2])


def test_writes_only_inside_the_lock():
    log = []
    asyncio.run(locked_write_commit(_FakeSession(log), _work))
    assert [k for k, _ in log] == ["execute", "add_all", "commit"]
    assert all(held for _, held in log)
    assert not _lock().locked()


def test_waits_for_the_queue_before_the_first_write():
    log = []

    async def main():
        await _lock().acquire()                     # another writer is mid-commit
        t = asyncio.create_task(locked_write_commit(_FakeSession(log), _work))
        await asyncio.sleep(0.05)
        assert log == []                                  # nothing written while queued (the old inversion wrote here)
        _lock().release()
        await t

    asyncio.run(main())
    assert [k for k, _ in log] == ["execute", "add_all", "commit"]


def test_failure_rolls_back_inside_the_lock_and_reraises():
    log = []
    with pytest.raises(RuntimeError):
        asyncio.run(locked_write_commit(_FakeSession(log, fail_commit=True), _work))
    assert log[-1] == ("rollback", True)
    assert not _lock().locked()


def test_frenzy_flag_save_is_lock_first():
    from services.trading_engine import TradingEngine
    src = inspect.getsource(TradingEngine._frenzy_persist_flags)
    assert "locked_write_commit(" in src
    assert "await _ls.execute(" not in src                # no write before the lock
    assert "locked_commit(_ls)" not in src


def test_close_phase2_read_does_not_flush_early():
    """The close path's PairData read sits under no_autoflush, so the ~40 phase-2 attributes are not written
    (SQLite file lock not taken) before the Binance get_ohlcv call — they flush inside locked_commit."""
    from services.trading_engine import TradingEngine
    src = inspect.getsource(TradingEngine)
    i = src.index("PHASE 2: Optional metadata")
    seg = src[i:src.index("await locked_commit(db)", i)]
    nf = seg.index("with db.no_autoflush:")
    assert nf < seg.index("select(PairData).where(PairData.pair == order.pair)") < seg.index("binance_service.get_ohlcv(")


def test_no_autoflush_really_defers_the_write_on_async_sqlite():
    from sqlalchemy import Column, Integer, String, event, select
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import declarative_base

    B = declarative_base()

    class T(B):
        __tablename__ = "t"
        id = Column(Integer, primary_key=True)
        v = Column(String)

    class U(B):
        __tablename__ = "u"
        id = Column(Integer, primary_key=True)

    async def main():
        eng = create_async_engine("sqlite+aiosqlite://")
        seen = []
        event.listen(eng.sync_engine, "before_cursor_execute", lambda c, cur, st, *a: seen.append(st.split()[0]))
        async with eng.begin() as c:
            await c.run_sync(B.metadata.create_all)
        async with AsyncSession(eng) as s:
            s.add(T(id=1, v="a")); await s.commit()
            row = (await s.execute(select(T))).scalar_one()
            row.v = "b"
            seen.clear()
            with s.no_autoflush:
                await s.execute(select(U))
            assert "UPDATE" not in seen                   # no write before the commit
            await s.commit()
            assert "UPDATE" in seen
        await eng.dispose()

    asyncio.run(main())


def test_willy_flush_and_close_flag_are_lock_first():
    from services.trading_engine import TradingEngine
    wf = inspect.getsource(TradingEngine._willy_hold_flush)
    assert "locked_write_commit(_ws, _wh_write)" in wf and "locked_commit(_ws)" not in wf
    mc = inspect.getsource(TradingEngine._mark_close_in_progress)
    assert "await locked_execute_commit(db," in mc and "await db.execute(" not in mc and "locked_commit(db)" not in mc
