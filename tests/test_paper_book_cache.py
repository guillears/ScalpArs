"""⚡ Oct-5 refresh fix 3 (DECISION_LOG 212): the dashboard's shared paper-book read.
Concurrent dashboard reads share ONE recompute; fresh=True (money paths) always recomputes; the TTL and invalidate expire it."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402


class _DB:
    """the open-margin SUM query → 200."""
    async def execute(self, q):
        class _R:
            def scalar(self):
                return 200.0
        return _R()


DB = _DB()


def _stub(monkeypatch):
    calls = {"usdt": 0, "bnb": 0}

    async def usdt(db):
        calls["usdt"] += 1
        await asyncio.sleep(0.01)   # long enough for the concurrent readers to queue on the lock
        return 1000.0 + calls["usdt"]

    async def bnb(db):
        calls["bnb"] += 1
        return 50.0

    monkeypatch.setattr(main.trading_engine, "_recalculate_paper_balance", usdt)
    monkeypatch.setattr(main.trading_engine, "_recalculate_paper_bnb", bnb)
    main._paper_book_invalidate()
    main._PAPER_BOOK.update(free=None, bnb=None, margin=None)
    return calls


def test_concurrent_dashboard_reads_share_one_recompute(monkeypatch):
    calls = _stub(monkeypatch)

    async def run():
        return await asyncio.gather(*(main._paper_book(DB) for _ in range(3)))
    res = asyncio.run(run())
    assert calls == {"usdt": 1, "bnb": 1}
    assert res == [(1001.0, 50.0, 200.0)] * 3


def test_fresh_always_recomputes_and_refreshes_the_copy(monkeypatch):
    calls = _stub(monkeypatch)
    asyncio.run(main._paper_book(DB))
    assert asyncio.run(main._paper_book(DB, fresh=True)) == (1002.0, 50.0, 200.0)
    assert calls["usdt"] == 2
    assert asyncio.run(main._paper_book(DB)) == (1002.0, 50.0, 200.0)   # the cached copy is the fresh one
    assert calls["usdt"] == 2


def test_ttl_and_invalidate_expire_the_copy(monkeypatch):
    calls = _stub(monkeypatch)
    asyncio.run(main._paper_book(DB))
    main._PAPER_BOOK["t"] -= main._PAPER_BOOK_TTL_S + 0.1
    asyncio.run(main._paper_book(DB))
    assert calls["usdt"] == 2
    main._paper_book_invalidate()
    asyncio.run(main._paper_book(DB))
    assert calls["usdt"] == 3


def test_invalidate_during_an_inflight_recompute_does_not_recache_stale_values(monkeypatch):
    calls = _stub(monkeypatch)

    async def run():
        t = asyncio.create_task(main._paper_book(DB))
        await asyncio.sleep(0)            # the recompute is in flight (sleeping inside the stub)
        main._paper_book_invalidate()     # e.g. a paper reset lands now
        await t
    asyncio.run(run())
    asyncio.run(main._paper_book(DB))
    assert calls["usdt"] == 2             # the in-flight result was served once but not cached
