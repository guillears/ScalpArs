"""🔒 Sep-30 one open per pair at a time — a manual click landing inside the bot's own check→insert window (or the reverse)
created two OPEN rows for one pair (deep review, reproduced). Both open paths now hold the pair's lock from check to insert."""
import asyncio, inspect, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
import services.trading_engine as T


def _eng():
    return T.TradingEngine.__new__(T.TradingEngine)


def test_same_pair_waits_other_pairs_do_not():
    e = _eng(); log = []

    async def hold(pair, tag, secs):
        async with e._pair_open_guard(pair):
            log.append((tag, "in", round(time.monotonic() - t0, 2)))
            await asyncio.sleep(secs)
            log.append((tag, "out", round(time.monotonic() - t0, 2)))

    async def run():
        global t0
        t0 = time.monotonic()
        await asyncio.gather(hold("QNTUSDT", "a", 0.3), hold("QNTUSDT", "b", 0.0), hold("SUIUSDT", "c", 0.0))
    asyncio.run(run())
    order = [(t, w) for t, w, _ in log]
    assert order.index(("b", "in")) > order.index(("a", "out"))          # same pair: b waited for a
    assert order.index(("c", "in")) < order.index(("a", "out"))          # other pair: c did not wait


def test_the_task_already_opening_a_pair_can_re_enter_it():
    """open_position's matched-long flip calls open_position again for the SAME pair from inside the lock."""
    e = _eng()

    async def run():
        async with e._pair_open_guard("QNTUSDT"):
            async with e._pair_open_guard("QNTUSDT"):
                return "no deadlock"
    assert asyncio.run(asyncio.wait_for(run(), 2.0)) == "no deadlock"


def test_a_manual_click_behind_a_long_bot_open_gives_up_readably():
    e = _eng()

    async def run():
        async def bot():
            async with e._pair_open_guard("QNTUSDT"):
                await asyncio.sleep(0.5)
        async def manual():
            await asyncio.sleep(0.05)
            async with e._pair_open_guard("QNTUSDT", 0.1):
                return "entered"
        return await asyncio.gather(bot(), manual(), return_exceptions=True)
    _, m = asyncio.run(run())
    assert isinstance(m, ValueError) and "being opened right now" in str(m)


def test_both_open_paths_are_serialized_with_their_signatures_kept():
    for fn in (T.TradingEngine.open_position, T.TradingEngine.open_manual_position):
        assert hasattr(fn, "__wrapped__")
    params = inspect.signature(T.TradingEngine.open_position).parameters
    assert "pair" in params and "entry_gap_5_20_signed_pct" in params          # wraps() keeps the real signature
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "    @serialized_per_pair()\n    async def open_position(" in src
    assert "    @serialized_per_pair(normalize=normalize_manual_pair, wait_s=lambda: manual_open_wait_s(config.trading_config))\n    async def open_manual_position(" in src
    # the manual lock key is the order-row pair format, whatever the operator typed
    for raw in ("qnt", "QNT/USDT", "QNT/USDT:USDT", "QNTUSDT", " qntusdt "):
        assert T.normalize_manual_pair(raw) == "QNTUSDT"


def test_manual_open_reaches_the_lock_with_the_normalized_pair(monkeypatch):
    """The decorator must lock the SAME key the bot uses ('QNTUSDT'), not the raw text typed in the dashboard."""
    e = _eng(); e.is_paper_mode = True; seen = []
    real = T.TradingEngine._pair_open_guard

    def spy(self, pair, wait_s=None):
        seen.append((pair, wait_s)); return real(self, pair, wait_s)
    monkeypatch.setattr(T.TradingEngine, "_pair_open_guard", spy)
    try:
        asyncio.run(e.open_manual_position(None, pair="qnt/usdt", direction="UP", investment=100, leverage=20))
    except ValueError:
        pass
    assert seen == [("QNTUSDT", T.manual_open_wait_s(T.config.trading_config))]


def test_manual_wait_outlasts_a_bot_maker_entry():
    from types import SimpleNamespace as NS
    assert T.manual_open_wait_s(NS(maker_entry_enabled=True, maker_timeout_seconds=20)) == 40.0
    assert T.manual_open_wait_s(NS(maker_entry_enabled=True, maker_timeout_seconds=90)) == 45.0     # capped: the browser request must answer first
    assert T.manual_open_wait_s(NS(maker_entry_enabled=False, maker_timeout_seconds=90)) == 20.0
    assert T.manual_open_wait_s(NS()) == 20.0 and T.manual_open_wait_s(NS(maker_entry_enabled=True, maker_timeout_seconds="x")) == 20.0


def test_two_concurrent_opens_of_one_pair_leave_one_open_row():
    """The bug claim end to end, through the real decorator: each open checks the book, awaits (the exchange fill / stamp
    reads), then inserts. Without the lock both pass the check → 2 rows; with it the second sees the first's row."""
    class Book(T.TradingEngine):
        def __init__(self):
            self.rows = []

        async def _open(self, db, pair, who, secs):
            pair = T.normalize_manual_pair(pair)              # as open_manual_position does
            if any(r[0] == pair for r in self.rows):
                return None                                   # "pair already OPEN" → refused
            await asyncio.sleep(secs)                         # the window between the check and the insert
            self.rows.append((pair, who)); return who
        locked = T.serialized_per_pair(normalize=T.normalize_manual_pair)(_open)

    async def run(fn):
        b = Book()
        res = await asyncio.gather(fn(b, None, pair="QNTUSDT", who="bot", secs=0.3),
                                   fn(b, None, pair="qnt", who="manual", secs=0.0),
                                   fn(b, None, pair="SUIUSDT", who="bot2", secs=0.0))
        return b.rows, res
    rows, _ = asyncio.run(run(Book._open))
    assert sorted(r[1] for r in rows if r[0] == "QNTUSDT") == ["bot", "manual"]                # unlocked: the race — two rows
    rows, res = asyncio.run(run(Book.locked))
    assert [r for r in rows if r[0] == "QNTUSDT"] == [("QNTUSDT", "bot")] and res[1] is None    # manual waited, saw the bot's row
    assert ("SUIUSDT", "bot2") in rows                                                          # another pair was never held up


# ── 🔒 Sep-30 account book lock (position limits / funds race across DIFFERENT pairs) ──

def test_book_lock_serializes_final_sections_across_pairs_and_is_released_on_every_exit():
    e = _eng()

    async def run():
        order = []

        async def section(tag, secs):
            rel = await e._book_hold()
            order.append((tag, "in"))
            await asyncio.sleep(secs)
            order.append((tag, "out"))
            rel(); rel()                                  # idempotent
        await asyncio.gather(section("a", 0.2), section("b", 0.0))
        assert order == [("a", "in"), ("a", "out"), ("b", "in"), ("b", "out")]
        rel = await e._book_hold(); rel2 = await e._book_hold(); rel2(); rel()      # re-entrant for the holder
        # a book lock left held by an open that raises / returns early is released by the serialized_per_pair wrapper
        class X(T.TradingEngine):
            def __init__(self): pass
            @T.serialized_per_pair()
            async def boom(self, db, pair):
                await self._book_hold(); raise RuntimeError("fail after the book lock")
            @T.serialized_per_pair()
            async def early(self, db, pair):
                await self._book_hold(); return None
        x = X()
        try:
            await x.boom(None, pair="AUSDT")
        except RuntimeError:
            pass
        await x.early(None, pair="BUSDT")
        rel = await asyncio.wait_for(x._book_hold(), 1.0); rel()                   # free again
        try:
            async def hold_long():
                r = await e._book_hold(); await asyncio.sleep(0.5); r()
            async def impatient():
                await asyncio.sleep(0.05); await e._book_hold(0.1)
            await asyncio.gather(hold_long(), impatient())
            raise AssertionError("should have timed out")
        except ValueError as err:
            assert "being booked right now" in str(err)
    asyncio.run(run())


def test_two_manual_clicks_on_different_pairs_cannot_overshoot_the_cap():
    """The bug claim through the real decorator + book lock: cap 1, two clicks on different pairs, each checks the count and
    inserts after its stamp reads. Without the book lock both pass the check (2 rows); with it the second sees the first."""
    class Book(T.TradingEngine):
        def __init__(self, use_book):
            self.rows = []; self.use_book = use_book

        @T.serialized_per_pair(normalize=T.normalize_manual_pair)
        async def click(self, db, pair, cap=1):
            await asyncio.sleep(0.1)                                   # the stamp reads
            rel = (await self._book_hold()) if self.use_book else (lambda: None)
            if len(self.rows) >= cap:
                rel(); return "refused"
            await asyncio.sleep(0.05)                                  # build + flush + commit
            self.rows.append(pair); rel(); return "opened"

    async def run(use_book):
        b = Book(use_book)
        res = await asyncio.gather(b.click(None, pair="QNT"), b.click(None, pair="SUI"))
        return b.rows, sorted(res)
    rows, res = asyncio.run(run(False)); assert len(rows) == 2                            # the race: 2 / 1
    rows, res = asyncio.run(run(True)); assert len(rows) == 1 and res == ["opened", "refused"]


def test_both_open_paths_book_under_the_lock():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = src.index("    async def open_position(\n"); j = src.index("\n    async def ", i + 10); bot = src[i:j]
    assert "if book_changed_refusal(available, _avail_now, investment, self.is_paper_mode):" in bot
    a, b, c, d = (bot.index("_book_release = await self._book_hold()"), bot.index("[BOOK_CHANGED]"),
                  bot.index("        order = Order(\n            binance_order_id=binance_order_id,"), bot.index("_book_release()   # 🔒 live: the row is committed"))
    assert a < b < c < d and bot.index("await locked_commit(db)", c) < d
    # paper: released only after the bookkeeping writes (save_state), never at the commit (deep review 140)
    e = bot.index("await self.save_state(db)", d); f = bot.index("            _book_release()\n", e)
    assert e < f < bot.index("_snap = await db.execute(", e)
    i = src.index("    async def open_manual_position"); j = src.index("\n    async def ", i + 10); man = src[i:j]
    a = man.index("_book_release = await self._book_hold(10.0)")
    assert a < man.index("was opened by the bot while this manual entry") < man.index("manual positions cap reached while") \
        < man.index("exceeds the available balance after") < man.index("order = Order(") < man.index("await self.save_state(db)") \
        < man.index("_book_release()   # 🔒 after the bookkeeping writes")



def test_book_changed_refusal_only_when_free_usdt_fell_and_no_longer_covers():
    f = T.book_changed_refusal
    assert f(1000.0, 1000.0, 1200.0, True) is False          # unchanged balance: never refused (even if sized above it)
    assert f(1000.0, 1500.0, 900.0, True) is False           # balance rose (a close)
    assert f(1000.0, 950.0, 900.0, True) is False            # fell but still covers
    assert f(1000.0, 400.0, 900.0, True) is True             # fell and no longer covers → refuse
    assert f(1000.0, 400.0, 900.0, False) is False           # live: the exchange already funded the order
    assert f(None, 400.0, 900.0, True) is False and f(1000.0, "x", 900.0, True) is False


def test_book_hold_survives_a_flip_style_re_entry_and_is_released_by_the_right_wrapper():
    """open_position → _maybe_open_flip → open_position: the inner wrapper must neither release the outer open's book hold
    nor leave its own behind; after an exception the book is free for other tasks."""
    class X(T.TradingEngine):
        def __init__(self): pass

        @T.serialized_per_pair()
        async def outer(self, db, pair, fail=False):
            await self._book_hold()                      # the outer open is booking
            await self.inner(db, pair=pair)              # re-enters for the SAME pair (flip)
            st = self._book_lock_state
            assert st[1].locked() and st[2] is asyncio.current_task(), "inner wrapper released the outer hold"
            if fail:
                raise RuntimeError("outer fails while booking")
            return "ok"

        @T.serialized_per_pair()
        async def inner(self, db, pair):
            await self._book_hold()                      # re-entrant: no-op for the holder
            return "inner"

    async def run():
        x = X()
        assert await x.outer(None, pair="QNTUSDT") == "ok"
        assert not x._book_lock_state[1].locked()
        try:
            await x.outer(None, pair="QNTUSDT", fail=True)
        except RuntimeError:
            pass
        rel = await asyncio.wait_for(x._book_hold(), 1.0); rel()
    asyncio.run(run())


def test_paper_bnb_swaps_move_usdt_only_under_the_book_lock():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    for name in ("    async def _execute_bnb_swap(", "    async def _execute_bnb_sell(", "    async def _paper_bnb_credit("):
        i = src.index(name); body = src[i:src.index("\n    async def ", i + 10)]
        assert body.index("self._book_ctx()") < body.index("async with self._bnb_swap_lock():"), name   # book → bnb, never reverse
