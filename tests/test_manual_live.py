"""🖐 MANUAL entry in LIVE mode (DECISION_LOG 181) — the order of operations is the safety design: nothing may refuse after the fill,
the exchange safety stop sits beyond the trade's own stop and inside liquidation, and a fill that cannot be booked is flattened."""
import asyncio, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def _bars(n, seed=7):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return [[1_700_000_000_000 + i * 300_000, float(c[i]), float(c[i] * 1.001), float(c[i] * 0.999), float(c[i]), 1000.0] for i in range(n)]


def test_safety_stop_distance_per_order():
    import services.trading_engine as T
    class TH: broker_backstop_pct = 2.5; frenzy_stop_pct = 3.0; broker_backstop_enabled = True
    f = T.order_backstop_pct
    assert f(TH, "MOMENTUM", None, None, 20) == 2.5 and f(TH, None, None, None, None) == 2.5 and f(TH, "FRENZY_LONG", "FRENZY", -3.0, 20) == 2.5   # every bot trade
    assert f(TH, "MANUAL", "FIXED", -1.5, 20) == 2.5                     # own stop 1.5 + 0.5 < the bot's distance → unchanged
    assert f(TH, "MANUAL", "FLOOR", -3.0, 20) == 3.5 and f(TH, "MANUAL", "FRENZY", None, 20) == 3.5   # beyond the 3 % stop, inside liquidation (4.5)
    assert f(TH, "MANUAL", "MOMENTUM", None, 20) == 2.5
    liq30 = T.manual_liquidation_distance_pct(30)
    assert abs(f(TH, "MANUAL", "FLOOR", T.manual_floor_for_leverage(TH, 30), 30) - round(liq30 * 0.92, 4)) < 1e-9     # capped inside liquidation
    assert f(TH, "MANUAL", "MOMENTUM", None, 50) < 2.5                   # the bot's 2.5 % is past liquidation at 50× → pulled inside it
    assert f(TH, "MANUAL", "FLOOR", "x", "y") == 2.5                     # bad input → the bot's distance
    g = T.manual_backstop_stop_floor
    assert g(True, TH, "FRENZY", None, 20) is None and abs(g(False, TH, "FRENZY", None, 20) - (-3.2)) < 1e-9
    class Off(TH): broker_backstop_enabled = False
    assert g(False, Off, "FRENZY", None, 20) is None


def _harness(monkeypatch, *, positions_before=(), order_result="ok", positions_after=(), commit_fails=False, close_ok=True, positions_fail=False,
             visible_from=2, filled=None, backstop_raises=None, cache_fails=False):
    """order_result: "ok" · None = the request left and errored (outcome unknown) · "notsent" = refused before the request."""
    # A live engine on in-memory SQLite with a faked exchange. Returns (run, log) — log records every exchange call in order.
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    import models, services.trading_engine as T
    log = []
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False; eng._last_pair_block_reason = {}
    eng._record_filter_block = lambda name, d, had_room=True: log.append(("block", name))
    async def _noop(*a, **k): return None
    async def _cache(*a, **k):
        if cache_fails:
            raise RuntimeError("cache boom")
    eng.update_orders_cache = _cache
    monkeypatch.setattr(T, "MANUAL_FILL_POLL_S", 0.0)
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False); monkeypatch.setattr(th, "broker_backstop_pct", 2.5, raising=False)
    monkeypatch.setattr(T.config.trading_config.investment, "leverage_bracket_cap_enabled", False, raising=False)
    B = T.binance_service
    async def _bal(): return {"usdt_free": 10_000.0}
    async def _px(symbol): return 2.0
    async def _ohlcv(symbol, tf, n): return _bars(260)[-n:]
    async def _fr(symbol): return 0.0001
    calls = {"pos": 0}
    async def _positions():
        calls["pos"] += 1; log.append(("positions", calls["pos"]))
        if positions_fail and calls["pos"] > 1:
            return None
        src = positions_before if calls["pos"] == 1 else (positions_after if calls["pos"] >= visible_from else ())
        return [dict(symbol="FOO/USDT:USDT", side=s, contracts=c, entry_price=p) for (s, c, p) in src]
    async def _order(symbol, side, amount, leverage=1, is_close=False, status=None):
        log.append(("order", side, round(amount, 4), leverage))
        if order_result == "notsent":
            return None
        if status is not None:
            status["sent"] = True
        if order_result is None:
            return None
        return {"id": "777", "price": 2.01, "amount": amount, "filled": (amount if filled is None else filled)}
    async def _close(symbol, side, amount):
        log.append(("flatten", side, round(amount, 4))); return {"id": "778"} if close_ok else None
    async def _bk(pair, direction, trigger):
        log.append(("backstop", direction, round(trigger, 5)))
        if backstop_raises is not None:
            raise backstop_raises
        return "algo-1"
    async def _cancel(pair, algo_id):
        log.append(("cancel_backstop", algo_id)); return True
    for name, fn in (("get_balance", _bal), ("get_current_price", _px), ("get_ohlcv", _ohlcv), ("fetch_funding_rate", _fr), ("get_open_positions", _positions),
                     ("create_market_order", _order), ("close_position", _close), ("place_backstop_stop", _bk), ("cancel_backstop_stop", _cancel)):
        monkeypatch.setattr(B, name, fn)
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: None)
    monkeypatch.setattr(T.websocket_tracker, "pair_silence_seconds", lambda p: None)
    monkeypatch.setattr(T.websocket_tracker, "force_reset_tracking", lambda *a, **k: None)
    monkeypatch.setattr(T.websocket_tracker, "subscribe_pair", _noop)
    if commit_fails:
        async def _boom(db): raise RuntimeError("database is locked")
        monkeypatch.setattr(T, "locked_commit", _boom)

    async def run(**kw):
        e = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        args = dict(pair="FOOUSDT", direction="LONG", investment=100.0, leverage=20, exit_mode="FLOOR")
        args.update(kw)
        try:
            async with async_sessionmaker(e, expire_on_commit=False)() as db:
                try:
                    o = await TE.open_manual_position(eng, db, **args)
                    return ("OK", o)
                except (ValueError, RuntimeError, asyncio.CancelledError) as err:
                    from sqlalchemy import select, func
                    await db.rollback()
                    n = (await db.execute(select(func.count(models.Order.id)))).scalar()
                    return (type(err).__name__, str(err), n)
        finally:
            await e.dispose()
    return run, log


def test_live_open_sends_the_order_last_and_places_the_safety_stop(monkeypatch):
    run, log = _harness(monkeypatch)
    kind, o = asyncio.run(run())
    assert kind == "OK" and o.is_paper is False and o.entry_strategy == "MANUAL" and o.binance_order_id == "777" and o.backstop_algo_id == "algo-1"
    assert o.entry_price == 2.01 and abs(o.quantity - 1000.0) < 1e-6 and o.pattern_fixed_sl_pct == -3.0
    kinds = [x[0] for x in log]
    assert kinds == ["positions", "order", "backstop"]                               # exchange position check → order → safety stop; nothing else
    assert log[1] == ("order", "buy", 1000.0, 20) and type(log[1][3]) is int and log[2] == ("backstop", "LONG", round(2.01 * (1 - 0.035), 5))   # FLOOR −3 % → safety stop 3.5 % below the fill
    run2, log2 = _harness(monkeypatch)
    kind, o = asyncio.run(run2(direction="SHORT", exit_mode="FIXED", sl_pct=1.0))
    assert kind == "OK" and o.direction == "SHORT" and log2[1][:2] == ("order", "sell") and log2[2] == ("backstop", "SHORT", round(2.01 * 1.025, 5))


def test_live_refusals_happen_before_any_order(monkeypatch):
    import services.trading_engine as T
    run, log = _harness(monkeypatch)
    r = asyncio.run(run(leverage=20.5)); assert r[0] == "ValueError" and "whole number" in r[1]
    r = asyncio.run(run(leverage=50, exit_mode="FLOOR")); assert r[0] == "ValueError" and "too close to this trade's own stop" in r[1]
    monkeypatch.setattr(T.config.trading_config.thresholds, "broker_backstop_enabled", False, raising=False)
    r = asyncio.run(run()); assert r[0] == "ValueError" and "safety stop" in r[1]
    assert not [x for x in log if x[0] in ("order", "backstop", "flatten")]          # nothing reached the exchange
    run, log = _harness(monkeypatch, positions_before=[("LONG", 5.0, 1.9)])          # an exchange position the bot does not know
    r = asyncio.run(run()); assert r[0] == "ValueError" and "does not know" in r[1] and r[2] == 0 and [x[0] for x in log] == ["positions"]


def test_live_order_failure_is_checked_on_the_exchange(monkeypatch):
    run, log = _harness(monkeypatch, order_result="notsent")                          # refused BEFORE the request left → nothing can be open
    r = asyncio.run(run()); assert r[0] == "ValueError" and "NOT sent" in r[1] and r[2] == 0
    assert [x[0] for x in log] == ["positions", "order"]
    run, log = _harness(monkeypatch, order_result=None)                               # the request left, errored, and nothing shows after 4 reads
    r = asyncio.run(run()); assert r[0] == "RuntimeError" and "outcome is unknown" in r[1] and "CHECK BINANCE NOW" in r[1] and r[2] == 0
    assert [x[0] for x in log] == ["positions", "order"] + ["positions"] * 4          # never "nothing opened" on a single early look
    run, log = _harness(monkeypatch, order_result=None, positions_after=[("LONG", 1000.0, 2.02)], visible_from=4)   # the fill shows on the 3rd re-read
    kind, o = asyncio.run(run())
    assert kind == "OK" and o.entry_price == 2.02 and o.quantity == 1000.0 and o.binance_order_id is None and o.backstop_algo_id == "algo-1"
    run, log = _harness(monkeypatch, order_result=None, positions_after=[("SHORT", 1000.0, 2.02)])   # a position on the OTHER side
    r = asyncio.run(run()); assert r[0] == "RuntimeError" and "shows a SHORT position" in r[1] and r[2] == 0
    run, log = _harness(monkeypatch, order_result=None, positions_fail=True)          # the request left and Binance cannot be read
    r = asyncio.run(run()); assert r[0] == "RuntimeError" and "CHECK BINANCE NOW" in r[1] and r[2] == 0


def test_live_fill_that_cannot_be_booked_is_closed_again(monkeypatch):
    run, log = _harness(monkeypatch, commit_fails=True)
    r = asyncio.run(run())
    assert r[0] == "RuntimeError" and "closed again at market" in r[1] and r[2] == 0
    assert [x[0] for x in log] == ["positions", "order", "backstop", "flatten", "cancel_backstop"] and log[3] == ("flatten", "LONG", 1000.0)
    run, log = _harness(monkeypatch, commit_fails=True, close_ok=False)
    r = asyncio.run(run())
    assert r[0] == "RuntimeError" and "could NOT be closed" in r[1] and "safety stop is in place" in r[1]
    assert "cancel_backstop" not in [x[0] for x in log]                              # the safety stop stays when the close failed


def test_live_books_the_filled_size_and_survives_a_bad_fill_report(monkeypatch):
    run, log = _harness(monkeypatch, filled=400.0)                                    # the book only took 400 of 1,000
    kind, o = asyncio.run(run())
    assert kind == "OK" and o.quantity == 400.0 and abs(o.notional_value - 400.0 * 2.01) < 1e-6 and abs(o.investment - 400.0 * 2.01 / 20) < 1e-6
    import services.trading_engine as T
    run, log = _harness(monkeypatch)
    async def _weird(symbol, side, amount, leverage=1, is_close=False, status=None):
        status["sent"] = True; return {"id": "9", "price": "x", "amount": None, "filled": "y"}
    monkeypatch.setattr(T.binance_service, "create_market_order", _weird)
    kind, o = asyncio.run(run())
    assert kind == "OK" and o.entry_price == 2.0 and o.quantity == 1000.0             # unreadable fill → the click price and the requested size


def test_live_safety_stop_failure_still_books_and_a_cancel_flattens(monkeypatch):
    run, log = _harness(monkeypatch, backstop_raises=RuntimeError("algo down"))
    kind, o = asyncio.run(run())
    assert kind == "OK" and o.backstop_algo_id is None and "flatten" not in [x[0] for x in log]      # booked; the monitor's sweep heals the stop
    run, log = _harness(monkeypatch, backstop_raises=asyncio.CancelledError())                     # the task is cancelled between the fill and the row
    r = asyncio.run(run())
    assert r[0] == "CancelledError" and r[2] == 0 and ("flatten", "LONG", 1000.0) in log


def test_live_failures_after_the_commit_never_undo_the_position(monkeypatch):
    from sqlalchemy.ext.asyncio import AsyncSession
    run, log = _harness(monkeypatch, cache_fails=True)                                # the price-stream / cache set-up fails AFTER the row exists
    kind, o = asyncio.run(run())
    assert kind == "OK" and o.backstop_algo_id == "algo-1" and "flatten" not in [x[0] for x in log]
    run, log = _harness(monkeypatch)
    async def _boom(self, obj, *a, **k): raise RuntimeError("refresh boom")
    monkeypatch.setattr(AsyncSession, "refresh", _boom)
    kind, o = asyncio.run(run())
    assert kind == "OK" and "flatten" not in [x[0] for x in log] and "cancel_backstop" not in [x[0] for x in log]


def test_wiring_of_the_review_fixes():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "except BaseException as _book_err:" in eng and "asyncio.shield(_flatten())" in eng and "status=_sent" in eng
    assert "else:   # FIXED / FLOOR: the operator's SL / TP stored on the row" in eng      # the candle-path backup covers every manual exit mode
    assert eng.count("order_backstop_pct(config.trading_config.thresholds, ") == 2          # the re-place after a failed close + the sweep
    bs = open(os.path.join(ROOT, "services", "binance_service.py"), encoding="utf-8").read()
    assert "status['sent'] = True" in bs and "'filled': float(order.get('filled') or 0)" in bs
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "Number(data.manual_max_open_positions) <= 0);" in html and "data.is_paper === false); }   // 🖐" not in html   # the panel shows in live
