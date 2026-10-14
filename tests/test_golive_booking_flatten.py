"""🛡 Go-live ③ (DECISION_LOG 292): a BOT entry that FILLED on Binance but whose order row was never committed is closed again at
market (reduce-only) at once — the mirror of the MANUAL open's flatten. Flatten failed → CRITICAL + the symbol is blocked (untracked).
Paper: never armed. All exchange calls mocked."""
import asyncio, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


class _DB:
    def __init__(self): self.rolled = 0
    async def rollback(self): self.rolled += 1


def _engine(monkeypatch, paper=False, close_ok=True, close_raises=False):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper
    log, blocks = [], []
    eng._record_filter_block = lambda name, d, had_room=True: blocks.append((name, d))
    async def _close(symbol, side, qty):
        log.append(("close", symbol, side, qty))
        if close_raises:
            raise RuntimeError("exchange down")
        return {"id": "c1"} if close_ok else None
    async def _cancel(pair, aid):
        log.append(("cancel_backstop", pair, aid)); return True
    monkeypatch.setattr(T.binance_service, "close_position", _close)
    monkeypatch.setattr(T.binance_service, "cancel_backstop_stop", _cancel)
    return T, eng, log, blocks


def _open_fn(T, *, fail=None, commit=False, fill=True, bk="ALG1"):
    """A stand-in for open_position's body: fills, records the backstop, then fails / returns / commits."""
    async def fn(self, db, pair="FOOUSDT"):
        if fill:
            self._note_live_fill(pair, "FOO/USDT:USDT", "LONG", 12.5, 2.0)
            st = T._LIVE_FILL_CV.get()
            if st is not None and st.get('filled'):
                st['bk'] = bk
        if commit:
            T._LIVE_FILL_CV.get()['committed'] = True
        if fail is not None:
            raise fail
        return "row" if commit else None
    return T.flatten_unbooked_live_fill(fn)


def test_failed_booking_is_flattened_and_its_stop_cancelled(monkeypatch):
    T, eng, log, blocks = _engine(monkeypatch)
    db = _DB()
    try:
        asyncio.run(_open_fn(T, fail=RuntimeError("database is locked"))(eng, db))
        raise AssertionError("the booking error must still propagate")
    except RuntimeError as e:
        assert "database is locked" in str(e)
    assert log == [("close", "FOO/USDT:USDT", "LONG", 12.5), ("cancel_backstop", "FOOUSDT", "ALG1")]
    assert db.rolled == 1 and blocks == [("BOOKING_FAILED_FLATTENED", "LONG")] and eng.untracked_block("FOOUSDT") is None


def test_a_return_without_booking_also_flattens(monkeypatch):
    T, eng, log, blocks = _engine(monkeypatch)
    assert asyncio.run(_open_fn(T)(eng, _DB())) is None
    assert log[0] == ("close", "FOO/USDT:USDT", "LONG", 12.5) and ("BOOKING_FAILED_FLATTENED", "LONG") in blocks


def test_flatten_failure_blocks_the_symbol_and_keeps_the_stop(monkeypatch):
    for kw in (dict(close_ok=False), dict(close_raises=True)):
        T, eng, log, blocks = _engine(monkeypatch, **kw)
        try:
            asyncio.run(_open_fn(T, fail=RuntimeError("insert failed"))(eng, _DB()))
        except RuntimeError:
            pass
        assert [x[0] for x in log] == ["close"]                                   # the safety stop stays in place
        assert blocks == [("BOOKING_FAILED_FLATTEN_FAILED", "LONG")]
        u = eng.untracked_block("FOOUSDT")
        assert u is not None and u["source"] == "BOOKING_FAILED" and u["qty"] == 12.5 and u["side"] == "LONG"


def test_committed_row_is_never_flattened(monkeypatch):
    T, eng, log, blocks = _engine(monkeypatch)
    try:
        asyncio.run(_open_fn(T, commit=True, fail=RuntimeError("refresh failed after commit"))(eng, _DB()))
    except RuntimeError:
        pass
    assert asyncio.run(_open_fn(T, commit=True)(eng, _DB())) == "row"
    assert log == [] and blocks == []


def test_no_fill_and_paper_are_noops(monkeypatch):
    T, eng, log, blocks = _engine(monkeypatch)
    try:
        asyncio.run(_open_fn(T, fill=False, fail=ValueError("refused before the order"))(eng, _DB()))
    except ValueError:
        pass
    T, eng, log2, blocks2 = _engine(monkeypatch, paper=True)
    try:
        asyncio.run(_open_fn(T, fail=RuntimeError("paper insert failed"))(eng, _DB()))
    except RuntimeError:
        pass
    assert log == [] and blocks == [] and log2 == [] and blocks2 == []


def test_a_cancelled_task_still_flattens(monkeypatch):
    T, eng, log, blocks = _engine(monkeypatch)
    try:
        asyncio.run(_open_fn(T, fail=asyncio.CancelledError())(eng, _DB()))
    except asyncio.CancelledError:
        pass
    assert log[0][0] == "close" and ("BOOKING_FAILED_FLATTENED", "LONG") in blocks


def test_open_position_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "    @serialized_per_pair(lane=BOT_OPEN_LANE)\n    @flatten_unbooked_live_fill\n    async def open_position(" in eng
    assert eng.count("self._note_live_fill(pair, symbol, direction, quantity, actual_price)") == 2     # maker and taker live fills
    i = eng.index("        db.add(transaction)\n\n        await locked_commit(db)\n")
    assert "_st_fill['committed'] = True" in eng[i:i + 400]                                           # marked right after the commit
    assert "_st_fill['bk'] = _bk_algo_id" in eng
