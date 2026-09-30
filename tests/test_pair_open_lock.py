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
