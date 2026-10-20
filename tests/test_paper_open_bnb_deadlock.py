"""🔓 DECISION_LOG 325: a PAPER bot open of a resting-TP sleeve (FRENZY_WILLY / FRENZY_LITE …) armed its resting TP while still holding
the account book lock — _rtp_arm waits on the global _close_lock — while a paper close holding _close_lock can run the emergency BNB
top-up (_deduct_fee_from_bnb → _execute_bnb_swap → the book lock). Lock-order cycle → every paper close and the monitor freeze.
Fix: paper arms AFTER its book release (as the manual open does); live is unchanged (its book is released at the commit)."""
import asyncio
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def _open_src():
    import services.trading_engine as T
    return inspect.getsource(T.TradingEngine.open_position)


def test_paper_arms_only_after_the_book_release():
    src = _open_src()
    i = src.index("_rtp_arm_after_release = False")
    site = src[i:i + 400]
    assert "if self.is_paper_mode:\n                _rtp_arm_after_release = True" in site   # paper defers
    assert "else:\n                await self._rtp_arm(db, order)" in site                     # live arms at once (unchanged)
    j = src.index("if _rtp_arm_after_release:")
    paper = src[src.rindex("_bnb_swap_owed = await self._deduct_fee_from_bnb(entry_fee, db, defer_swap=True)", 0, j):j]
    assert "_book_release()" in paper                                    # the book lock is released BEFORE the paper arm
    assert src.count("await self._rtp_arm(db, order)") == 2              # exactly the two sites (live now · paper after release)


def test_close_path_still_takes_the_book_under_the_close_lock():
    """the other half of the cycle is unchanged on purpose (a close's fee top-up may take the book lock) — which is why an open must
    never wait on _close_lock while holding the book."""
    import services.trading_engine as T
    assert "async with (self._book_ctx() if self.is_paper_mode else contextlib.nullcontext()):" in inspect.getsource(T.TradingEngine._execute_bnb_swap)


def test_lock_order_model_no_cycle():
    """the two flows with fresh locks: the NEW open order (book → release → close) and a close (close → book) both finish; the OLD
    open order (close taken while holding the book) deadlocks against the same close."""
    async def run(old_order):
        book, close = asyncio.Lock(), asyncio.Lock()
        in_close = asyncio.Event(); open_has_book = asyncio.Event()

        async def opener():
            await book.acquire()
            open_has_book.set()
            await in_close.wait()                    # the close is holding _close_lock now
            if old_order:
                async with close:                    # _rtp_arm under the book lock (pre-325)
                    pass
                book.release()
            else:
                book.release()                       # paper bookkeeping done → book released
                async with close:                    # then _rtp_arm
                    pass

        async def closer():
            await open_has_book.wait()
            async with close:
                in_close.set()
                async with book:                     # emergency BNB top-up
                    pass
        try:
            await asyncio.wait_for(asyncio.gather(opener(), closer()), 1.0)
            return True
        except asyncio.TimeoutError:
            return False
    assert asyncio.run(run(old_order=False)) is True
    assert asyncio.run(run(old_order=True)) is False


def test_real_engine_locks_new_order_finishes_old_order_deadlocks(monkeypatch):
    """the engine's OWN primitives (review M1): _book_hold / its release, the module _close_lock and _book_ctx (what the paper BNB
    top-up takes under _close_lock). New order: book → release → _close_lock. Old order: _close_lock while still holding the book."""
    import services.trading_engine as T

    async def run(old_order):
        monkeypatch.setattr(T, "_close_lock", asyncio.Lock())       # a fresh lock bound to this loop
        e = T.TradingEngine(); e.is_paper_mode = True
        has_book = asyncio.Event(); closing = asyncio.Event()

        async def opener():                                          # open_position's paper tail
            rel = await e._book_hold()
            has_book.set()
            await closing.wait()
            if not old_order:
                rel()                                                # paper book released after the bookkeeping …
            async with T._close_lock:                                # … then _rtp_arm
                pass
            if old_order:
                rel()

        async def closer():                                          # a paper close paying its fee → emergency top-up
            await has_book.wait()
            async with T._close_lock:
                closing.set()
                async with e._book_ctx():
                    pass
        try:
            await asyncio.wait_for(asyncio.gather(opener(), closer()), 1.0)
            return True
        except asyncio.TimeoutError:
            return False
    assert asyncio.run(run(old_order=False)) is True
    assert asyncio.run(run(old_order=True)) is False
