"""🎯 RESTING TAKE-PROFIT (DECISION_LOG 296) + ⏰ WILLY LATE TP (DECISION_LOG 297).

Pure rules (services/resting_tp.py, services/frenzy.py) and the engine wiring against an in-memory DB with EVERY exchange call mocked — no
network. Covers: placement on open (live, enabled sleeves only) · failure path · fill detection books MAKER at the limit with NO market
order · a touch does nothing · the through + grace safety net · partial fill + stop → cancel, remainder, blended · every exit path cancels
first (source pins + functional close) · switch OFF = today (no exchange call, same paper P&L) · paper fill rule · heal / switch-off /
re-price / orphan sweeps · the late-minute swap · reduce-only on every close."""
import asyncio, os, sys, inspect, time
from datetime import datetime, timedelta
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import pytest
from services.resting_tp import (blend_exit, order_fill, ticks_through, paper_filled, safety_net, is_full, remaining_qty, config_errors,
                                 rtp_target, rtp_reason_for_limit, at_or_through)
from services.exec_shadow import resting_tp_price, net_pct_at
from services.indicators import calculate_pnl
from services.frenzy import frenzy_willy_exit_for, frenzy_exit_for, frenzy_willy_late_tp, frenzy_willy_tp_at

MK, TK = 0.00018, 0.00045


def _th(**kw):
    base = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0,
                frenzy_lock_arm_pct=0.0, frenzy_stop_pct=3.0, frenzy_trail_arm_pct=5.0, frenzy_trail_giveback_pct=1.5,
                resting_tp_willy_enabled=True, resting_tp_lite_enabled=True, resting_tp_grace_s=3.0, resting_tp_through_ticks=2,
                frenzy_willy_late_tp_minutes=30.0, frenzy_willy_late_tp_pct=0.25)
    base.update(kw)
    return SimpleNamespace(**base)


# ─────────────────────────────────────────────────────────────── pure rules

def test_blend_books_exactly_the_sum_of_the_parts():
    e, q, ef = 100.0, 10.0, 0.45   # entry fee $0.45 on $1000
    b = blend_exit(q, 4.0, 101.3, 99.0, MK, TK)
    assert b["order_type"] == "MAKER+TAKER" and b["filled_qty"] == 4.0 and b["rem_qty"] == 6.0
    pd = calculate_pnl("LONG", e, b["price"], q, 10, entry_fee=ef, exit_fee=b["fee"])
    parts = (101.3 - e) * 4 + (99.0 - e) * 6 - ef - (4 * 101.3 * MK + 6 * 99.0 * TK)
    assert abs(pd["pnl"] - parts) < 1e-9
    full = blend_exit(q, 10.0, 101.3, None, MK, TK)
    assert full["order_type"] == "MAKER" and abs(full["fee"] - 10 * 101.3 * MK) < 1e-12 and full["rem_qty"] == 0.0
    none = blend_exit(q, 0.0, None, 99.0, MK, TK)
    assert none["order_type"] == "TAKER" and abs(none["fee"] - 10 * 99.0 * TK) < 1e-12
    s = blend_exit(q, 4.0, 98.0, 101.0, MK, TK)   # SHORT: same convention (price is direction-agnostic)
    pd = calculate_pnl("SHORT", e, s["price"], q, 10, entry_fee=ef, exit_fee=s["fee"])
    assert abs(pd["pnl"] - ((e - 98.0) * 4 + (e - 101.0) * 6 - ef - (4 * 98 * MK + 6 * 101 * TK))) < 1e-9
    assert blend_exit(0, 1, 1, 1, MK, TK) is None and blend_exit(q, 4.0, None, 99.0, MK, TK) is None
    assert remaining_qty(10, 4) == 6 and remaining_qty(10, 12) == 0 and is_full(10, 10) and not is_full(10, 9.9)


def test_order_fill_normalises_binance_states():
    assert order_fill({"status": "FILLED", "filled": 10, "amount": 10, "average": 101.3}) == ("FILLED", 10.0, 101.3)
    assert order_fill({"status": "PARTIALLY_FILLED", "filled": 4, "amount": 10, "average": 101.3})[0] == "PARTIAL"
    assert order_fill({"status": "NEW", "filled": 0, "amount": 10})[0] == "OPEN"
    assert order_fill({"status": "CANCELED", "filled": 3, "amount": 10, "average": 101.3}) == ("GONE", 3.0, 101.3)
    assert order_fill(None) == (None, 0.0, None)


def test_touch_is_not_through_and_the_safety_net_needs_ticks_and_time():
    assert ticks_through("LONG", 101.0, 101.0, 0.01) == 0 and not paper_filled("LONG", 101.0, 101.0, 0.01)
    assert paper_filled("LONG", 101.01, 101.0, 0.01) and paper_filled("SHORT", 98.99, 99.0, 0.01) and not paper_filled("SHORT", 99.0, 99.0, 0.01)
    assert at_or_through("LONG", 101.0, 101.0) and not at_or_through("LONG", 100.99, 101.0)
    ft, fire = safety_net(None, 10.0, "LONG", 101.01, 101.0, 0.01, 2, 3.0)
    assert ft is None and not fire                                              # 1 tick through < 2
    ft, fire = safety_net(None, 10.0, "LONG", 101.02, 101.0, 0.01, 2, 3.0)
    assert ft == 10.0 and not fire
    assert safety_net(ft, 12.9, "LONG", 101.05, 101.0, 0.01, 2, 3.0) == (10.0, False)
    assert safety_net(ft, 13.0, "LONG", 101.05, 101.0, 0.01, 2, 3.0) == (10.0, True)
    assert safety_net(ft, 13.0, "LONG", 101.0, 101.0, 0.01, 2, 3.0) == (None, False)   # back to a touch → the clock resets


def test_config_errors_and_targets():
    assert config_errors(_th()) == []
    assert config_errors(_th(resting_tp_grace_s=-1)) and config_errors(_th(resting_tp_through_ticks=0)) and config_errors(_th(resting_tp_through_ticks=2.5))
    assert config_errors(_th(frenzy_willy_late_tp_pct=1.25)) and config_errors(_th(frenzy_willy_late_tp_minutes=-1)) and config_errors(_th(frenzy_willy_late_tp_pct=-0.1))
    assert config_errors(_th(frenzy_willy_late_tp_pct=0, frenzy_willy_late_tp_minutes=0)) == []
    th = _th()
    assert rtp_target(th, "FRENZY_WILLY", 0) == (1.25, "FRENZY_TP") and rtp_target(th, "FRENZY_WILLY", 30.0) == (0.25, "WILLY_LATE_TP")
    assert rtp_target(th, "FRENZY_LITE", 50) == (3.0, "FRENZY_TP")
    assert rtp_target(th, "FRENZY_LONG", 0) is None and rtp_target(th, "MOMENTUM", 0) is None and rtp_target(th, "MANUAL", 0) is None
    assert rtp_target(_th(resting_tp_willy_enabled=False), "FRENZY_WILLY", 0) is None
    assert rtp_target(_th(frenzy_lock_arm_pct=3.0), "FRENZY_LITE", 0) is None             # lock-then-trail: no fixed TP to rest
    ef = 0.045
    main_lim = resting_tp_price("LONG", 100.0, ef, 1.25, MK, 0.01)
    late_lim = resting_tp_price("LONG", 100.0, ef, 0.25, MK, 0.01)
    assert net_pct_at("LONG", 100.0, main_lim, ef, MK) >= 1.25 and late_lim < main_lim
    assert rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 100.0, ef, main_lim, MK) == "FRENZY_TP"
    assert rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 100.0, ef, late_lim, MK) == "WILLY_LATE_TP"
    assert rtp_reason_for_limit(th, "FRENZY_LITE", "LONG", 100.0, ef, late_lim, MK) == "FRENZY_TP"


def test_willy_late_tp_rule():
    th = _th()
    assert frenzy_willy_late_tp(th) == (30.0, 0.25) and frenzy_willy_tp_at(th, 29.9) == (1.25, "FRENZY_TP")
    assert frenzy_willy_tp_at(th, 30.0) == (0.25, "WILLY_LATE_TP") and frenzy_willy_tp_at(th, None) == (1.25, "FRENZY_TP")
    # before the late minute: nothing changes
    assert frenzy_willy_exit_for(0.5, 0.9, th, 29.0) == (False, "MAX_HOLD_TIME", -4.0)
    assert frenzy_willy_exit_for(1.3, 1.3, th, 29.0) == (True, "FRENZY_TP", 1.25)
    # from the late minute: +0.25 on the CURRENT P&L (a peak earlier is no fill)
    assert frenzy_willy_exit_for(0.26, 0.9, th, 30.0) == (True, "WILLY_LATE_TP", 0.25)
    assert frenzy_willy_exit_for(1.30, 1.30, th, 45.0) == (True, "FRENZY_TP", 1.25)         # deep E: a jump past +1.25 after minute 30 is the TP
    assert frenzy_willy_exit_for(0.10, 0.9, th, 45.0)[0] is False
    assert frenzy_willy_exit_for(-4.1, 0.9, th, 45.0)[:2] == (True, "STOP_LOSS")        # stop unchanged
    assert frenzy_willy_exit_for(-1.0, 0.1, th, 90.0)[:2] == (True, "MAX_HOLD_TIME")    # cap unchanged
    # off (either ≤ 0) → identical to today at every age
    for off in (_th(frenzy_willy_late_tp_pct=0), _th(frenzy_willy_late_tp_minutes=0), _th(frenzy_willy_late_tp_pct=1.25)):
        assert frenzy_willy_late_tp(off) is None
        for p, pk, age in ((0.26, 0.9, 45), (1.3, 1.3, 45), (0.5, 1.4, 45), (-4.1, 0.2, 45), (0.0, 0.0, 95)):
            ref = frenzy_willy_exit_for(p, pk, _th(frenzy_willy_late_tp_pct=0, frenzy_willy_late_tp_minutes=0), age)
            assert frenzy_willy_exit_for(p, pk, off, age) == ref
    # a resting TP alive: no TP branch fires (main or late); stop / cap still do
    assert frenzy_willy_exit_for(1.3, 1.3, th, 10, resting_tp=True)[0] is False
    assert frenzy_willy_exit_for(0.5, 1.4, th, 10, resting_tp=True)[0] is False
    assert frenzy_willy_exit_for(0.3, 0.3, th, 40, resting_tp=True)[0] is False
    assert frenzy_willy_exit_for(-4.0, 1.4, th, 10, resting_tp=True)[:2] == (True, "STOP_LOSS")
    assert frenzy_exit_for(3.1, 3.1, th, use_tp=True, sleeve="FRENZY_LITE") == (True, "FRENZY_TP", 3.0)
    assert frenzy_exit_for(3.1, 3.1, th, use_tp=True, sleeve="FRENZY_LITE", resting_tp=True)[0] is False
    assert frenzy_exit_for(-3.0, 3.1, th, use_tp=True, sleeve="FRENZY_LITE", resting_tp=True)[:2] == (True, "STOP_LOSS")


# ─────────────────────────────────────────────────────────────── engine harness

TICK = 0.01


def _setup(monkeypatch, paper=False, rows=(), place=None, fetch=None, cancel="canceled", ro_orders=(), close_px=99.0, th_kw=None, pos=None):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    import models, services.trading_engine as T
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    S = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(T, "AsyncSessionLocal", S)
    monkeypatch.setattr(T.binance_service, "_markets_loaded", False, raising=False)   # isolation: another test may leave markets "loaded"
    th = T.config.trading_config.thresholds
    _kw = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_lite_tp_pct=3.0,
               frenzy_lock_arm_pct=0.0, resting_tp_willy_enabled=True, resting_tp_lite_enabled=True, resting_tp_grace_s=3.0,
               resting_tp_through_ticks=2, frenzy_willy_late_tp_minutes=30.0, frenzy_willy_late_tp_pct=0.25, broker_backstop_enabled=True,
               resting_tp_frenzy_long_enabled=False, resting_tp_manual_enabled=False, resting_tp_override_enabled=False,   # 🎯 (310) the 296 scope; the 310 switches: tests/test_resting_tp_fixed_all.py
               resting_tp_lite_retry_enabled=False)   # 🔁 (316) its own switch: tests/test_frenzy_lite_retry.py
    _kw.update(th_kw or {})
    for k, v in _kw.items():
        monkeypatch.setattr(th, k, v, raising=False)
    monkeypatch.setattr(T.config.trading_config, "maker_fee", MK, raising=False)
    monkeypatch.setattr(T.config.trading_config, "taker_fee", TK, raising=False)
    monkeypatch.setattr(T.config.trading_config, "maker_exit_enabled", False, raising=False)
    T._rtp_rt.clear(); T._exit_retry_queue.clear(); T._open_orders_cache.clear()
    log, blocks = [], []

    def _no_paper(name):
        if paper:
            raise AssertionError(f"paper must never call the exchange ({name})")

    async def _place(pair, direction, qty, price):
        _no_paper("place"); log.append(("place", pair, direction, qty, round(price, 8)))
        return place(pair, price) if callable(place) else place
    async def _cancel(pair, oid):
        _no_paper("cancel"); log.append(("cancel", pair, oid)); return cancel
    async def _fetch(pair, oid):
        _no_paper("fetch"); log.append(("fetch", pair, oid)); return fetch(oid) if callable(fetch) else fetch
    async def _close(symbol, side, amount):
        _no_paper("close"); log.append(("market_close", symbol, side, round(amount, 8))); return {"id": "m1", "price": close_px}
    async def _bk_cancel(pair, aid):
        log.append(("bk_cancel", pair, aid)); return True
    async def _positions():
        return []
    async def _ro(pair=None):
        _no_paper("ro"); log.append(("ro_read", pair)); return list(ro_orders)
    async def _raise(*a, **k):
        raise RuntimeError("not in tests")
    _pos_seq = list(pos) if isinstance(pos, (list, tuple)) else None
    async def _pos(symbol):
        _no_paper("position"); log.append(("pos_read", symbol))
        v = (_pos_seq.pop(0) if _pos_seq else None) if _pos_seq is not None else (pos(symbol) if callable(pos) else pos)
        if isinstance(v, Exception):
            raise v
        return None if not v else {"symbol": symbol, "contracts": float(v)}
    async def _empty(*a, **k):
        return []
    for name, fn in (("place_resting_tp", _place), ("cancel_resting_tp", _cancel), ("fetch_resting_tp", _fetch), ("close_position", _close),
                     ("cancel_backstop_stop", _bk_cancel), ("get_open_positions", _positions), ("fetch_open_reduce_only_orders", _ro),
                     ("get_balance", _raise), ("get_bnb_price", _raise), ("get_ohlcv", _empty), ("load_markets", _empty),
                     ("get_position_for_symbol", _pos)):
        monkeypatch.setattr(T.binance_service, name, fn)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper
    eng._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
    eng._rtp_tick_cached = lambda pair, px: TICK
    async def _tick_for(pair, px):
        return TICK
    eng._rtp_tick_for = _tick_for
    eng.close_db_priority = 0; eng._bg_tasks = set(); eng._rsi3_history = {}; eng._post_exit_tracking = {}
    async def _gap():
        return None
    eng._get_exit_btc_trend_gap = _gap
    eng._register_post_exit_tracking = lambda o, r: None
    eng._wh_spawn = lambda coro: coro.close()
    async def _fund(*a):
        return None
    eng._stamp_funding_async = _fund
    async def _paper_noop(*a, **k):
        return None
    eng._recalculate_paper_balance = _paper_noop; eng._deduct_fee_from_bnb = _paper_noop; eng.save_state = _paper_noop
    eng.paper_balance = 0.0; eng.paper_bnb_balance_usd = 0.0
    monkeypatch.setattr(T.asyncio, "sleep", _fast_sleep)

    async def seed():
        async with engine.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as s:
            for kw in rows:
                base = dict(pair="WILUSDT", direction="LONG", status="OPEN", entry_price=100.0, current_price=100.0, investment=100.0, leverage=10.0,
                            notional_value=1000.0, quantity=10.0, entry_fee=0.45, confidence="STRONG_BUY", is_paper=paper,
                            opened_at=datetime.utcnow(), closing_in_progress=False, entry_strategy="FRENZY_WILLY", peak_pnl=0.0, trough_pnl=0.0)
                base.update(kw)
                s.add(models.Order(**base))
            await s.commit()

    async def read():
        from sqlalchemy import select
        async with S() as s:
            return {o.id: o for o in (await s.execute(select(models.Order))).scalars().all()}
    return T, eng, engine, S, seed, read, log, blocks


_real_sleep = asyncio.sleep
async def _fast_sleep(*a, **k):
    await _real_sleep(0)


EF = 0.045   # entry fee % (0.45 / 1000)
LIM = resting_tp_price("LONG", 100.0, EF, 1.25, MK, TICK)
LATE = resting_tp_price("LONG", 100.0, EF, 0.25, MK, TICK)


def _row_with_rtp(**kw):
    d = dict(protective_tp_order_id="T1", resting_tp_price=LIM, resting_tp_filled_qty=0.0, backstop_algo_id="B1")
    d.update(kw)
    return d


# ── placement on open

def test_arm_places_for_enabled_sleeves_only(monkeypatch):
    rows = [dict(pair="WILUSDT"), dict(pair="LITUSDT", entry_strategy="FRENZY_LITE"), dict(pair="LONUSDT", entry_strategy="FRENZY_LONG"),
            dict(pair="MOMUSDT", entry_strategy="MOMENTUM")]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place=lambda p, px: {"id": "O-" + p, "price": px, "amount": 10.0})
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            for o in (await s.execute(select(models.Order))).scalars().all():
                await eng._rtp_arm(s, o)
        r = await read(); await engine.dispose(); return r
    r = asyncio.run(go())
    lite_lim = resting_tp_price("LONG", 100.0, EF, 3.0, MK, TICK)
    assert [x for x in log if x[0] == "place"] == [("place", "WILUSDT", "LONG", 10.0, LIM), ("place", "LITUSDT", "LONG", 10.0, lite_lim)]
    assert r[1].protective_tp_order_id == "O-WILUSDT" and r[1].resting_tp_price == LIM and r[1].resting_tp_filled_qty == 0.0
    assert r[2].protective_tp_order_id == "O-LITUSDT" and r[3].resting_tp_price is None and r[4].resting_tp_price is None
    assert net_pct_at("LONG", 100.0, LIM, EF, MK) >= 1.25 and 1 in T._rtp_rt and blocks == []


def test_arm_failure_runs_the_software_tp(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict()], place={"rejected": True, "error": "-5022 would cross"})
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            o = (await s.execute(select(models.Order))).scalar_one()
            res = await eng._rtp_arm(s, o)
        r = await read(); await engine.dispose(); return res, r
    res, r = asyncio.run(go())
    assert res == "rejected" and blocks == ["RESTING_TP_FAILED"] and r[1].protective_tp_order_id is None and 1 not in T._rtp_rt


def test_switch_off_no_exchange_call_and_no_state(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict(), dict(pair="LITUSDT", entry_strategy="FRENZY_LITE")],
                                                        th_kw=dict(resting_tp_willy_enabled=False, resting_tp_lite_enabled=False), place={"id": "X"})
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            for o in (await s.execute(select(models.Order))).scalars().all():
                assert await eng._rtp_arm(s, o) is None
                assert eng._rtp_state(o.id, row=o) is None
        out = await eng.resting_tp_sweep(now=0.0)
        await engine.dispose(); return out
    out = asyncio.run(go())
    assert log == [] and out == [] and T._rtp_rt == {}


def test_paper_arm_records_without_exchange(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, paper=True, rows=[dict()])
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            o = (await s.execute(select(models.Order))).scalar_one()
            res = await eng._rtp_arm(s, o)
        r = await read(); await engine.dispose(); return res, r
    res, r = asyncio.run(go())
    assert res == "paper" and r[1].resting_tp_price == LIM and r[1].protective_tp_order_id is None and log == []


# ── the gate (monitor + realtime)

def test_gate_touch_nothing_fill_books_and_safety_net(monkeypatch):
    state = {"st": {"status": "NEW", "filled": 0.0, "amount": 10.0, "average": 0.0}}
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], fetch=lambda oid: state["st"])
    async def go():
        await seed()
        rows = await read(); eng._rtp_state(1, row=rows[1])
        out = []
        out.append(await eng._rtp_gate(1, LIM - TICK, now=0.0))           # below: no query, nothing
        out.append(await eng._rtp_gate(1, LIM, now=1.0))                  # TOUCH: queried (1), still NEW → nothing
        out.append(await eng._rtp_gate(1, LIM + 2 * TICK, now=1.5))      # 2 through: query throttled, the clock starts
        out.append(await eng._rtp_gate(1, LIM + 2 * TICK, now=3.0))      # queried (2), still NEW, 1.5 s < 3 s
        state["st"] = {"status": "PARTIALLY_FILLED", "filled": 4.0, "amount": 10.0, "average": LIM}
        out.append(await eng._rtp_gate(1, LIM + 2 * TICK, now=4.2))      # queried (3): partial recorded, 2.7 s < 3 s
        out.append(await eng._rtp_gate(1, LIM + 3 * TICK, now=4.6))      # ≥ 3 s through, no full fill → FALLBACK (query throttled)
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert out[:5] == [None, None, None, None, None] and out[5] == ("FALLBACK", "FRENZY_TP")
    assert [x[0] for x in log].count("fetch") == 3 and "RESTING_TP_FALLBACK" in blocks
    assert r[1].resting_tp_filled_qty == 4.0                                  # the partial fill was recorded
    assert not [x for x in log if x[0] in ("market_close", "cancel")]       # the gate never trades


def test_gate_filled_and_paper_rule(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()],
                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    async def go():
        await seed(); rows = await read(); eng._rtp_state(1, row=rows[1])
        g = await eng._rtp_gate(1, LIM, now=0.0); await engine.dispose(); return g
    assert asyncio.run(go()) == ("FILLED", "FRENZY_TP")
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, paper=True, rows=[dict(resting_tp_price=LIM, resting_tp_filled_qty=0.0)])
    async def go2():
        await seed(); rows = await read(); eng._rtp_state(1, row=rows[1])
        a = await eng._rtp_gate(1, LIM, now=0.0)                 # touch
        b = await eng._rtp_gate(1, LIM + TICK, now=10.0)         # 1 tick through
        await engine.dispose(); return a, b
    assert asyncio.run(go2()) == (None, ("PAPER_FILL", "FRENZY_TP")) and log == []


# ── the close path (cancel FIRST)

def _close(eng, T, S, reason, price, oid=1):
    async def run():
        from sqlalchemy import select
        import models
        async with S() as s:
            o = (await s.execute(select(models.Order).where(models.Order.id == oid))).scalar_one()
            return await eng._close_position_locked(s, o, price, reason)
    return run()


def test_full_fill_books_maker_with_no_market_order(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], cancel="gone",
                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 97.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    o = r[1]
    assert c is not None and o.status == "CLOSED" and o.close_reason == "FRENZY_TP" and o.exit_order_type == "MAKER"
    assert abs(o.exit_price - LIM) < 1e-12 and abs(o.exit_fee - 10 * LIM * MK) < 1e-9
    assert abs(o.pnl - ((LIM - 100.0) * 10 - 0.45 - 10 * LIM * MK)) < 1e-6
    assert not [x for x in log if x[0] == "market_close"]
    kinds = [x[0] for x in log]
    assert kinds.index("cancel") < kinds.index("bk_cancel")                      # the resting TP is cancelled FIRST
    assert o.protective_tp_order_id is None and o.resting_tp_filled_qty == 10.0 and 1 not in T._rtp_rt


def test_partial_fill_then_stop_closes_only_the_remainder_blended(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], close_px=96.0,
                                                        fetch={"status": "CANCELED", "filled": 4.0, "amount": 10.0, "average": LIM})
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    o = r[1]
    assert [x for x in log if x[0] == "market_close"] == [("market_close", "WIL/USDT:USDT", "LONG", 6.0)]   # ONLY the remainder
    kinds = [x[0] for x in log]
    assert kinds.index("cancel") < kinds.index("market_close")
    want_px = (4 * LIM + 6 * 96.0) / 10
    want_fee = 4 * LIM * MK + 6 * 96.0 * TK
    assert o.close_reason == "STOP_LOSS" and o.exit_order_type == "MAKER+TAKER" and abs(o.exit_price - want_px) < 1e-9
    assert abs(o.exit_fee - want_fee) < 1e-9 and o.resting_tp_filled_qty == 4.0
    assert abs(o.pnl - ((LIM - 100) * 4 + (96.0 - 100) * 6 - 0.45 - want_fee)) < 1e-6


def test_fallback_cancels_then_market_closes_the_rest(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], close_px=LIM - 0.05,
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0, "average": 0.0})
    async def go():
        await seed(); c = await _close(eng, T, S, "FRENZY_TP", LIM + 0.03); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert [x[0] for x in log][:2] == ["cancel", "fetch"] and ("market_close", "WIL/USDT:USDT", "LONG", 10.0) in log
    assert r[1].exit_order_type == "TAKER" and r[1].close_reason == "FRENZY_TP" and r[1].resting_tp_filled_qty == 0.0


def test_blocked_cancel_sends_no_market_order_and_stamps_the_reason(monkeypatch):
    # a non-stop exit (the safety-net FRENZY_TP) waits: no market order, backstop kept, the retry queue re-runs THIS reason (caveman #8)
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], cancel="error",
                                                        fetch={"status": "NEW", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed(); c = await _close(eng, T, S, "FRENZY_TP", 101.5); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert c is None and r[1].status == "OPEN" and 1 in T._exit_retry_queue and r[1].close_reason == "FRENZY_TP"
    assert not [x for x in log if x[0] in ("market_close", "bk_cancel")]   # backstop stays, no market order
    assert r[1].protective_tp_order_id == "T1"


def test_stop_class_close_goes_out_on_a_blocked_cancel_and_reconciles_after(monkeypatch):
    # caveman #6: STOP_LOSS never waits — reduce-only market for (qty − recorded fill), then the limit is re-read and booked blended
    reads = iter([{"status": "NEW", "filled": 2.0, "amount": 10.0, "average": LIM},          # settle: cancel errored, still resting
                  {"status": "CANCELED", "filled": 2.0, "amount": 10.0, "average": LIM}])   # after the close: cancelled, 2 filled
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp(resting_tp_filled_qty=2.0)], cancel="error",
                                                        fetch=lambda oid: next(reads), close_px=96.0, pos=[8.0, None])
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    o = r[1]
    assert c is not None and o.status == "CLOSED" and o.close_reason == "STOP_LOSS"
    assert ("market_close", "WIL/USDT:USDT", "LONG", 8.0) in log                   # ONLY the unfilled rest, reduce-only
    assert o.exit_order_type == "MAKER+TAKER" and abs(o.exit_price - (2 * LIM + 8 * 96.0) / 10) < 1e-9 and "RTP_FILL_UNVERIFIED" not in (o.notes or "")


def test_stop_class_close_unverified_fill_is_flagged(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], cancel="error", fetch=None, close_px=96.0, pos=[10.0, None])
    async def go():
        await seed(); c = await _close(eng, T, S, "MAX_HOLD_TIME", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert r[1].status == "CLOSED" and ("market_close", "WIL/USDT:USDT", "LONG", 10.0) in log and "RTP_FILL_UNVERIFIED" in (r[1].notes or "")


def test_switch_off_live_close_is_todays_close(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict(backstop_algo_id="B1")], close_px=99.0,
                                                        th_kw=dict(resting_tp_willy_enabled=False))
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 99.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert [x[0] for x in log] == ["bk_cancel", "market_close"] and r[1].exit_order_type == "TAKER"
    assert abs(r[1].pnl - ((99.0 - 100.0) * 10 - 0.45 - 10 * 99.0 * TK)) < 1e-6


def test_paper_close_parity(monkeypatch):
    # switch OFF / no record → today's paper close (taker at the price); with a record + 1 tick through → the limit, maker fee
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, paper=True, rows=[dict(), dict(pair="P2USDT", resting_tp_price=LIM),
                                                                                        dict(pair="P3USDT", resting_tp_price=LIM)])
    async def go():
        await seed()
        await _close(eng, T, S, "FRENZY_TP", 101.30, oid=1)
        await _close(eng, T, S, "FRENZY_TP", LIM + TICK, oid=2)
        await _close(eng, T, S, "FRENZY_TP", LIM, oid=3)            # a touch is not a fill → taker at the price
        r = await read(); await engine.dispose(); return r
    r = asyncio.run(go())
    assert log == []
    assert abs(r[1].pnl - ((101.30 - 100) * 10 - 0.45 - 10 * 101.30 * TK)) < 1e-6 and r[1].exit_order_type == "TAKER"
    assert abs(r[2].exit_price - LIM) < 1e-12 and r[2].exit_order_type == "MAKER" and abs(r[2].exit_fee - 10 * LIM * MK) < 1e-9
    assert r[2].resting_tp_filled_qty == 10.0
    assert r[3].exit_order_type == "TAKER" and abs(r[3].exit_price - LIM) < 1e-12


# ── sweeps

def test_sweep_heal_withdraw_reprice(monkeypatch):
    old = datetime.utcnow() - timedelta(minutes=5)
    rows = [dict(pair="HEAUSDT", opened_at=old),                                                              # 1 enabled, none → heal
            dict(pair="LITUSDT", entry_strategy="FRENZY_LITE", protective_tp_order_id="L1", resting_tp_price=LIM, resting_tp_filled_qty=0.0),   # 2 off → withdraw
            dict(pair="REPUSDT", protective_tp_order_id="R1", resting_tp_price=LIM + 0.5, resting_tp_filled_qty=0.0, opened_at=old),  # 3 TP moved → re-price
            dict(pair="PARUSDT", protective_tp_order_id=None, resting_tp_price=LIM, resting_tp_filled_qty=3.0)]   # 4 partial → never re-armed
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, th_kw=dict(resting_tp_lite_enabled=False),
                                                        place=lambda p, px: {"id": "N-" + p, "price": px, "amount": 10.0},
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed(); out = await eng.resting_tp_sweep(now=100.0); again = await eng.resting_tp_sweep(now=101.0)
        r = await read(); await engine.dispose(); return out, again, r
    out, again, r = asyncio.run(go())
    assert again == []                                                           # throttled
    assert sorted((k, i, v) for k, i, v in out) == [("heal", 1, "placed"), ("reprice", 3, "replaced"), ("withdraw", 2, "withdrawn")]
    assert r[1].protective_tp_order_id == "N-HEAUSDT" and r[1].resting_tp_price == LIM
    assert r[2].protective_tp_order_id is None and ("cancel", "LITUSDT", "L1") in log
    assert r[3].protective_tp_order_id == "N-REPUSDT" and r[3].resting_tp_price == LIM
    k = [x[:3] for x in log if x[0] in ("cancel", "place") and x[1] == "REPUSDT"]
    assert k == [("cancel", "REPUSDT", "R1"), ("place", "REPUSDT", "LONG")]       # cancel, then place
    assert r[4].protective_tp_order_id is None and not [x for x in log if x[1:2] == ("PARUSDT",)]


def test_heal_adopts_a_leftover_and_cancels_strays(monkeypatch):
    ro = [{"id": "LEFT", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": LIM, "amount": 10.0, "filled": 0.0, "timestamp": None},
          {"id": "STRAY", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": LIM + 1, "amount": 10.0, "filled": 0.0, "timestamp": None}]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict()], ro_orders=ro, place={"id": "NEW"})
    async def go():
        await seed(); res = await eng._rtp_heal(1); r = await read(); await engine.dispose(); return res, r
    res, r = asyncio.run(go())
    assert res == "placed" and r[1].protective_tp_order_id == "LEFT" and ("cancel", "WILUSDT", "STRAY") in log
    assert not [x for x in log if x[0] == "place"] and "RESTING_TP_ORPHAN" in blocks


def test_orphan_clean(monkeypatch):
    OLD_MS = (time.time() - 600) * 1000
    ro = [{"id": "O1", "pair": "GONEUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0, "timestamp": OLD_MS},
          {"id": "T1", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0, "timestamp": OLD_MS},
          {"id": "O4", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0, "timestamp": OLD_MS},
          {"id": "O5", "pair": "BSYUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0, "timestamp": OLD_MS},
          {"id": "O6", "pair": "NOTSUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0, "timestamp": None},
          {"id": "O7", "pair": "OPERUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0, "timestamp": OLD_MS},
          {"id": "O3", "pair": "NEWUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0,
           "timestamp": datetime.utcnow().timestamp() * 1000}]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp(), dict(pair="BSYUSDT")], ro_orders=ro)
    eng._rtp_enter("BSYUSDT")                                                    # an arm / heal in flight on that pair → never judged
    async def go():
        await seed()
        async with S() as s:
            out = await eng.resting_tp_orphan_clean(s, now=0.0, positions=[{"symbol": "OPER/USDT:USDT"}])
            again = await eng.resting_tp_orphan_clean(s, now=10.0)
        await engine.dispose(); return out, again
    out, again = asyncio.run(go())
    # S6: O6 (no timestamp = young) and O7 (a Binance position with no OPEN row — the operator's own) are never touched
    # by ORDER ID (caveman #5b / deep B): O1 (no row), O4 (a row, but not its recorded TP) cancelled; T1 (the row's TP), O5 (in flight),
    # O3 (younger than 60 s) kept
    assert out == [("GONEUSDT", "O1", "canceled"), ("WILUSDT", "O4", "canceled")] and again == [] and blocks == ["RESTING_TP_ORPHAN"] * 2


# ── ⏰ the WILLY late minute

def test_late_swap_cancel_then_place_and_market_now(monkeypatch):
    late_open = datetime.utcnow() - timedelta(minutes=31)
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp(opened_at=late_open), _row_with_rtp(pair="NOWUSDT", opened_at=late_open,
                                                        protective_tp_order_id="T2")],
                                                        place=lambda p, px: {"id": "LATE-" + p, "price": px, "amount": 10.0},
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed(); rows = await read()
        eng._rtp_state(1, row=rows[1]); eng._rtp_state(2, row=rows[2])
        a = await eng._rtp_gate(1, LATE - 5 * TICK, now=0.0)         # below the late limit → swap, no close
        b = await eng._rtp_gate(2, LATE + TICK, now=0.0)             # already through → market now
        r = await read(); await engine.dispose(); return a, b, r
    a, b, r = asyncio.run(go())
    assert a is None and r[1].protective_tp_order_id == "LATE-WILUSDT" and r[1].resting_tp_price == LATE
    seq = [x[:3] for x in log if x[1] == "WILUSDT" and x[0] in ("cancel", "place")]
    assert seq == [("cancel", "WILUSDT", "T1"), ("place", "WILUSDT", "LONG")]
    assert b == ("LATE_NOW", "WILLY_LATE_TP") and r[2].protective_tp_order_id is None and not [x for x in log if x[0] == "place" and x[1] == "NOWUSDT"]
    assert eng._rtp_reason(T._rtp_rt[1]) == "WILLY_LATE_TP"


def test_late_swap_cancel_reports_filled_books_the_tp(monkeypatch):
    late_open = datetime.utcnow() - timedelta(minutes=31)
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp(opened_at=late_open)], cancel="gone",
                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    async def go():
        await seed(); rows = await read(); eng._rtp_state(1, row=rows[1])
        g = await eng._rtp_gate(1, LIM - 0.2, now=0.0)
        c = await _close(eng, T, S, g[1], LIM - 0.2)
        r = await read(); await engine.dispose(); return g, r
    g, r = asyncio.run(go())
    assert g == ("FILLED", "FRENZY_TP") and r[1].close_reason == "FRENZY_TP" and r[1].exit_order_type == "MAKER"
    assert not [x for x in log if x[0] in ("place", "market_close")]


def test_late_paper_parity(monkeypatch):
    late_open = datetime.utcnow() - timedelta(minutes=31)
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, paper=True, rows=[dict(resting_tp_price=LIM, opened_at=late_open),
                                                                                        dict(pair="NOWUSDT", resting_tp_price=LIM, opened_at=late_open)])
    async def go():
        await seed(); rows = await read(); eng._rtp_state(1, row=rows[1]); eng._rtp_state(2, row=rows[2])
        a = await eng._rtp_gate(1, LATE - 5 * TICK, now=0.0)
        a2 = await eng._rtp_gate(1, LATE + TICK, now=1.0)
        b = await eng._rtp_gate(2, LATE + 3 * TICK, now=0.0)
        await _close(eng, T, S, a2[1], LATE + TICK, oid=1)
        await _close(eng, T, S, b[1], LATE + 3 * TICK, oid=2)
        r = await read(); await engine.dispose(); return a, a2, b, r
    a, a2, b, r = asyncio.run(go())
    assert log == [] and a is None and a2 == ("PAPER_FILL", "WILLY_LATE_TP") and b == ("LATE_NOW", "WILLY_LATE_TP")
    assert r[1].close_reason == "WILLY_LATE_TP" and r[1].exit_order_type == "MAKER" and abs(r[1].exit_price - LATE) < 1e-12
    assert r[2].close_reason == "WILLY_LATE_TP" and r[2].exit_order_type == "TAKER" and abs(r[2].exit_price - (LATE + 3 * TICK)) < 1e-12


# ── exchange layer: reduce-only everywhere

def test_reduce_only_on_every_close_and_gtx_on_the_limit(monkeypatch):
    from services.binance_service import BinanceService
    svc = BinanceService.__new__(BinanceService)
    calls = []
    class _X:
        markets = {}
        options = {}
        def price_to_precision(self, s, p): return f"{p:.2f}"
        def amount_to_precision(self, s, a): return f"{a:.3f}"
        def market(self, s): return {"limits": {}}
        async def create_order(self, symbol, type, side, amount, price=None, params=None):
            calls.append((type, side, amount, price, dict(params or {})))
            return {"id": "9", "status": "open", "price": price, "amount": amount, "info": {"status": "NEW"}, "average": 101.0, "filled": amount}
        async def fetch_ticker(self, s): return {"last": 100.0}
    svc.exchange = _X(); svc._markets_loaded = True
    async def _nb(): return None
    svc._check_ban = _nb
    async def _lm(): return None
    svc.load_markets = _lm
    r = asyncio.run(svc.place_resting_tp("WILUSDT", "LONG", 10.0, 101.234))
    assert r["id"] == "9" and calls[0][0] == "limit" and calls[0][1] == "sell" and calls[0][4] == {"reduceOnly": True, "timeInForce": "GTX"}
    asyncio.run(svc.close_position("WIL/USDT:USDT", "LONG", 6.0))
    assert calls[1][0] == "market" and calls[1][4] == {"reduceOnly": True}
    class _Exp(_X):
        async def create_order(self, **k): return {"id": "10", "status": "expired", "info": {"status": "EXPIRED"}}
    svc.exchange = _Exp()
    assert asyncio.run(svc.place_resting_tp("WILUSDT", "LONG", 10.0, 101.234))["rejected"] is True   # GTX would cross → rejected


# ── source pins: every exit path cancels first; both exit paths identical

def test_every_exit_path_cancels_first_source_pins():
    import services.trading_engine as T
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    close_src = inspect.getsource(T.TradingEngine._close_position_locked)
    i_settle, i_bk, i_mkt = close_src.index("await self._rtp_settle(order, allow_force="), close_src.index("cancel_backstop_stop(order.pair"), close_src.index("binance_service.close_position(")
    assert close_src.index("await db.refresh(order, attribute_names=['protective_tp_order_id'") < i_settle   # caveman #2: re-read first
    assert i_settle < i_bk < i_mkt                                               # RTP cancel → backstop cancel → market (remainder) order
    assert "amount=_close_qty" in close_src and "amount=order.quantity" not in close_src   # the market order closes ONLY the remainder
    # every close funnel goes through _close_position_locked (stop / time cap / manual / override / retry queue / safety net)
    assert eng.count("await self._close_position_locked(") == 2 and "closed_order = await trading_engine.close_position(db, order, current_price, \"MANUAL\")" in main
    # reconciler (EXTERNAL_CLOSE / BACKSTOP_STOP) settles the resting TP before booking; orphan clean runs there
    assert "_rtp_x = await trading_engine._rtp_settle(order)" in main and "await trading_engine.resting_tp_orphan_clean(db, positions=binance_positions)" in main
    # untracked flatten cancels reduce-only orders before the market close
    ut = inspect.getsource(T.TradingEngine.check_untracked_positions)
    assert ut.index("fetch_open_reduce_only_orders(p)") < ut.index("binance_service.close_position(")
    # exit override withdraws the limit; booking-failure flatten never meets one (placed only after the commit)
    assert "await self._rtp_withdraw(order.id, \"exit override\")" in inspect.getsource(T.TradingEngine.set_exit_override)
    op = eng[eng.index("_st_fill['committed'] = True   # 🛡 (292 ③) booked"):]
    assert op.index("await self._rtp_arm(db, order)") > 0
    # both exit paths: the same WILLY / FRENZY exit with the resting flag, then the same gate
    assert eng.count("_held_minutes(order.opened_at), None, resting_tp=_rtp_on_m)") == 1
    assert eng.count("_held_minutes(order_info.get('opened_at')), None, resting_tp=_rtp_on_rt)") == 1
    # 🎯 (310) + the operator-TP branch (MANUAL FIXED / override) of each path: the same gate, monitor awaits, realtime in the background
    assert eng.count("await self._rtp_gate(order.id, current_price)") == 2 and eng.count("await self._rtp_gate(order_id, current_price, background=True)") == 2
    assert "await self.resting_tp_sweep()" in eng
    # WILLY_LATE_TP whitelisted wherever FRENZY_TP is
    assert eng.count('startswith("FRENZY_TP") or _reason_base.startswith("WILLY_LATE_TP")') == 2
    assert "FRENZY_LATE_TP" not in eng and "FRENZY_LATE_TP" not in main                     # deep E: renamed WILLY_LATE_TP everywhere
    assert "self.rtp_clear_all()" in inspect.getsource(T.TradingEngine._set_paper_mode_locked) and main.count("trading_engine.rtp_clear_all()") == 2   # #7
    assert "await trading_engine.resting_tp_legacy_clear()" in main                         # deep G at startup


def test_model_columns_migration_config_and_ui_parity():
    import models
    cols = [c.name for c in models.Order.__table__.columns]
    assert {"protective_tp_order_id", "resting_tp_price", "resting_tp_filled_qty"} <= set(cols)
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "ADD COLUMN resting_tp_price FLOAT" in db and "ADD COLUMN resting_tp_filled_qty FLOAT" in db
    import json, config
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
    th = cfg.get("thresholds", cfg)
    for k, v in (("resting_tp_willy_enabled", True), ("resting_tp_lite_enabled", True), ("resting_tp_grace_s", 3.0), ("resting_tp_through_ticks", 2),
                 ("frenzy_willy_late_tp_minutes", 30.0), ("frenzy_willy_late_tp_pct", 0.25)):
        assert th[k] == v and getattr(config.SignalThresholds(), k) == v
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for i in ("config-rtp-willy-enabled", "config-rtp-lite-enabled", "config-rtp-grace", "config-rtp-through-ticks"):
        assert ui.count(f'id="{i}"') == 1 and ui.count(f"getElementById('{i}')") == 2   # input + load + save
    for i, k in (("config-fz-willy-late-tp-min", "frenzy_willy_late_tp_minutes"), ("config-fz-willy-late-tp-pct", "frenzy_willy_late_tp_pct")):
        assert ui.count(f'id="{i}"') == 1 and f"['{i}', '{k}'," in ui                    # FRENZY_NUM_FIELDS drives its load + save
    for k in ("resting_tp_willy_enabled", "resting_tp_through_ticks", "frenzy_willy_late_tp_minutes"):
        assert ui.count(f"_bt.{k}") >= 1                                         # the shared config-export builder (both text exports)
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_cfg_errs.extend(_rtp_cfg_errors(new_config.thresholds))" in main


# ─────────────────────────────────────────────────────────────── review round 1 (caveman #1–#12 + deep A–G)

def _seed_row_orm(S, oid=1):
    async def get():
        from sqlalchemy import select
        import models
        async with S() as s:
            return (await s.execute(select(models.Order).where(models.Order.id == oid))).scalar_one()
    return get()


def _reconcile_setup(monkeypatch, rows, **kw):
    import main
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, **kw)
    monkeypatch.setattr(T, "_close_lock", asyncio.Lock())                      # a fresh lock per test loop (contended below)
    async def _fill(order):
        return 101.0
    monkeypatch.setattr(main, "_get_actual_fill_price", _fill)
    async def _q(aid):
        return {"algoStatus": "CANCELED"}
    monkeypatch.setattr(T.binance_service, "query_backstop_stop", _q)
    monkeypatch.setattr(main.trading_engine, "_wh_spawn", lambda coro: coro.close())
    return main, T, eng, engine, S, seed, read, log, blocks


def test_reconciler_skips_a_row_the_gate_already_closed(monkeypatch):
    # caveman #1: the reconciler books under the close lock with a re-read — a gate close that won the race is never booked twice
    old = datetime.utcnow() - timedelta(minutes=5)
    main, T, eng, engine, S, seed, read, log, blocks = _reconcile_setup(monkeypatch, [_row_with_rtp(opened_at=old)],
                                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    async def go():
        await seed()
        await T._close_lock.acquire()                                            # the gate's close holds the lock
        async with S() as s:
            task = asyncio.create_task(main._close_orphan_orders(s, set()))
            await _real_sleep(0.05)
            from sqlalchemy import update
            import models
            async with S() as s2:                                                  # … and books the row CLOSED
                await s2.execute(update(models.Order).values(status="CLOSED")); await s2.commit()
            T._close_lock.release()
            out = await task
        from sqlalchemy import select, func
        import models
        async with S() as s3:
            ntx = (await s3.execute(select(func.count(models.Transaction.id)))).scalar()
        await engine.dispose(); return out, ntx
    out, ntx = asyncio.run(go())
    assert out == [] and ntx == 0 and not [x for x in log if x[0] == "cancel"]   # skipped before any settle / booking


def test_reconciler_books_a_full_fill_as_the_tp_and_flags_a_blocked_settle(monkeypatch):
    old = datetime.utcnow() - timedelta(minutes=5)
    main, T, eng, engine, S, seed, read, log, blocks = _reconcile_setup(monkeypatch, [_row_with_rtp(opened_at=old)], cancel="gone",
                                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    T._rtp_rt[1] = {"pair": "WILUSDT"}
    async def go():
        await seed()
        async with S() as s:
            out = await main._close_orphan_orders(s, set())
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    o = r[1]
    assert len(out) == 1 and o.close_reason == "FRENZY_TP" and o.exit_order_type == "MAKER" and abs(o.exit_price - LIM) < 1e-12
    assert abs(o.exit_fee - round(10 * LIM * MK, 4)) < 1e-9 and 1 not in T._rtp_rt               # state popped on the reconciler close
    # caveman #12: a blocked settle → EXTERNAL_CLOSE, flagged for the revert gate's accounting check
    main, T, eng, engine, S, seed, read, log, blocks = _reconcile_setup(monkeypatch, [_row_with_rtp(opened_at=old)], cancel="error", fetch=None)
    async def go2():
        await seed()
        async with S() as s:
            await main._close_orphan_orders(s, set())
        r = await read(); await engine.dispose(); return r
    r = asyncio.run(go2())
    assert r[1].close_reason == "EXTERNAL_CLOSE" and "RTP_SETTLE_BLOCKED" in (r[1].notes or "")


def test_close_rereads_the_resting_id_before_cancelling(monkeypatch):
    # caveman #2: a concurrent late swap wrote a NEWER id after this row object was loaded — the close cancels the NEW one
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp(protective_tp_order_id="OLD")], cancel="gone",
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed()
        from sqlalchemy import select, update
        import models
        async with S() as s:
            o = (await s.execute(select(models.Order))).scalar_one()
            async with S() as s2:
                await s2.execute(update(models.Order).values(protective_tp_order_id="NEW")); await s2.commit()
            await eng._close_position_locked(s, o, 96.0, "STOP_LOSS")
        r = await read(); await engine.dispose(); return r
    r = asyncio.run(go())
    assert ("cancel", "WILUSDT", "NEW") in log and ("cancel", "WILUSDT", "OLD") not in log and r[1].status == "CLOSED"


def test_exchange_timeouts_are_unknown_never_filled(monkeypatch):
    # caveman #3: a call that does not answer in RTP_IO_TIMEOUT_S is UNKNOWN → blocked (non-stop close), never a fill / a clean cancel
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()])
    monkeypatch.setattr(T, "RTP_IO_TIMEOUT_S", 0.05)
    async def _slow(*a, **k):
        await _real_sleep(1.0); return "canceled"
    monkeypatch.setattr(T.binance_service, "cancel_resting_tp", _slow)
    monkeypatch.setattr(T.binance_service, "fetch_resting_tp", _slow)
    monkeypatch.setattr(T.binance_service, "place_resting_tp", _slow)
    async def go():
        await seed()
        c = await _close(eng, T, S, "FRENZY_TP", 101.5)
        r = await read()
        from sqlalchemy import select
        import models
        async with S() as s:
            o = (await s.execute(select(models.Order))).scalar_one()
            o.protective_tp_order_id = None; await s.commit()
            placed = await eng._rtp_place_on(s, o, 1.25, "test")
        await engine.dispose(); return c, r, placed
    c, r, placed = asyncio.run(go())
    assert c is None and r[1].status == "OPEN" and not [x for x in log if x[0] == "market_close"] and placed == "unknown"


def test_realtime_gate_never_awaits_rest(monkeypatch):
    # caveman #4: the price callback schedules the query (in-flight flag) and returns at once; the job books its own fill
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()])
    closes = []
    async def _close_now(oid, reason):
        closes.append((oid, reason))
    eng._rtp_close_now = _close_now
    async def go():
        await seed(); rows = await read(); eng._rtp_state(1, row=rows[1])
        ev = asyncio.Event()
        async def _fetch(pair, oid):
            await ev.wait(); return {"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM}
        monkeypatch.setattr(T.binance_service, "fetch_resting_tp", _fetch)
        t0 = time.monotonic()
        g1 = await eng._rtp_gate(1, LIM + TICK, now=0.0, background=True)
        dt = time.monotonic() - t0
        busy = T._rtp_rt[1]["busy"]
        g2 = await eng._rtp_gate(1, LIM + TICK, now=5.0, background=True)     # in flight → no second query
        ev.set()
        for _ in range(20):
            await _real_sleep(0.01)
        g3 = await eng._rtp_gate(1, LIM + TICK, now=6.0, background=True)
        await engine.dispose(); return g1, dt, busy, g2, g3
    g1, dt, busy, g2, g3 = asyncio.run(go())
    assert g1 is None and dt < 0.5 and busy is True and g2 is None
    assert closes == [(1, "FRENZY_TP")] and g3 == ("FILLED", "FRENZY_TP") and T._rtp_rt[1]["busy"] is False


def test_close_with_no_recorded_id_cancels_the_symbols_reduce_only_orders_first(monkeypatch):
    # caveman #5a: a placement whose answer / commit was lost — the close cancels the symbol's reduce-only limits and books their fill
    ro = [{"id": "LOST", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": LIM, "amount": 10.0, "filled": 3.0, "timestamp": None}]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict(backstop_algo_id="B1", resting_tp_price=LIM, resting_tp_filled_qty=None)],
                                                        ro_orders=ro, close_px=96.0,
                                                        cancel={"result": "canceled", "status": "CANCELED", "filled": 3.0, "amount": 10.0, "average": LIM})
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    kinds = [x[0] for x in log]
    assert kinds.index("ro_read") < kinds.index("cancel") < kinds.index("market_close") and ("market_close", "WIL/USDT:USDT", "LONG", 7.0) in log
    assert r[1].exit_order_type == "MAKER+TAKER" and r[1].resting_tp_filled_qty == 3.0 and "fetch" not in kinds   # deep A: no follow-up read


def test_unknown_placement_is_adopted_by_the_next_heal(monkeypatch):
    # caveman #5c: place answered nothing → 'unknown'; the heal pass reads the symbol and ADOPTS the limit instead of placing a second
    ro = [{"id": "MAYBE", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": LIM, "amount": 10.0, "filled": 0.0, "timestamp": None}]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict()], place=None, ro_orders=ro)
    async def go():
        await seed()
        o = await _seed_row_orm(S)
        async with S() as s:
            from sqlalchemy import select
            import models
            o = (await s.execute(select(models.Order))).scalar_one()
            a = await eng._rtp_place_on(s, o, 1.25, "open")
        h = await eng._rtp_heal(1)
        r = await read(); await engine.dispose(); return a, h, r
    a, h, r = asyncio.run(go())
    assert a == "unknown" and h == "placed" and r[1].protective_tp_order_id == "MAYBE" and [x[0] for x in log].count("place") == 1


def test_state_is_validated_against_pair_open_time_and_mode(monkeypatch):
    # caveman #7: ids restart after a paper reset — a stale state never drives another order
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()])
    async def go():
        await seed(); rows = await read()
        st = eng._rtp_state(1, row=rows[1])
        a = eng._rtp_state(1, pair="WILUSDT", opened_at=rows[1].opened_at) is st
        b = eng._rtp_state(1, pair="OTHERUSDT") is None and 1 not in T._rtp_rt
        eng._rtp_state(1, row=rows[1])
        c = eng._rtp_state(1, pair="WILUSDT", opened_at=rows[1].opened_at + timedelta(hours=1)) is None
        eng._rtp_state(1, row=rows[1]); eng.is_paper_mode = True
        d = eng._rtp_state(1) is None
        eng.is_paper_mode = False; eng._rtp_state(1, row=rows[1]); eng.rtp_clear_all()
        await engine.dispose(); return a, b, c, d, dict(T._rtp_rt)
    a, b, c, d, left = asyncio.run(go())
    assert a and b and c and d and left == {}


def test_sweep_isolates_a_failing_row_and_closes_late_now(monkeypatch):
    # caveman #9 (one row's failure never stops the pass) + #10 (a sweep LATE_NOW is logged and closed at once)
    late_open = datetime.utcnow() - timedelta(minutes=31)
    rows = [_row_with_rtp(pair="BADUSDT"), dict(pair="HEAUSDT"), _row_with_rtp(pair="LATUSDT", protective_tp_order_id="T3", opened_at=late_open)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place={"id": "N1"},
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    orig = eng._rtp_limit_for
    async def _lim(o, tp):
        if o.pair == "BADUSDT":
            raise RuntimeError("boom")
        return await orig(o, tp)
    eng._rtp_limit_for = _lim
    eng._rtp_last_price = lambda pair: (LATE + 3 * TICK) if pair == "LATUSDT" else None
    closes = []
    async def _close_now(oid, reason):
        closes.append((oid, reason))
    eng._rtp_close_now = _close_now
    async def go():
        await seed(); out = await eng.resting_tp_sweep(now=0.0); await engine.dispose(); return out
    out = asyncio.run(go())
    assert ("heal", 2, "placed") in out and ("reprice", 3, "LATE_NOW") in out and closes == [(3, "WILLY_LATE_TP")]
    assert not [x for x in out if x[1] == 1]


def test_arm_cancels_stray_reduce_only_limits_first(monkeypatch):
    # deep B: a stale limit on the symbol can never ride into the new position
    ro = [{"id": "STALE", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": 50.0, "amount": 3.0, "filled": 0.0, "timestamp": None}]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict()], ro_orders=ro, place={"id": "NEW", "price": LIM})
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            o = (await s.execute(select(models.Order))).scalar_one()
            res = await eng._rtp_arm(s, o)
        await engine.dispose(); return res
    res = asyncio.run(go())
    kinds = [x[:3] for x in log if x[0] in ("cancel", "place")]
    assert res == "placed" and kinds[0] == ("cancel", "WILUSDT", "STALE") and kinds[1][0] == "place"


def test_gone_write_never_clears_a_newer_id(monkeypatch):
    # deep D: the gate's GONE record is conditional on the id it read
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed(); rows = await read(); eng._rtp_state(1, row=rows[1])
        from sqlalchemy import update
        import models
        async with S() as s2:
            await s2.execute(update(models.Order).values(protective_tp_order_id="NEWER")); await s2.commit()
        g = await eng._rtp_gate(1, LIM, now=0.0)
        r = await read(); await engine.dispose(); return g, r
    g, r = asyncio.run(go())
    assert g is None and r[1].protective_tp_order_id == "NEWER" and 1 not in T._rtp_rt


def test_legacy_protective_tp_id_cleared_on_startup(monkeypatch):
    # deep G: the retired April column — an id with no resting_tp_price on an OPEN row is cleared
    rows = [dict(pair="LEGUSDT", entry_strategy="MOMENTUM", protective_tp_order_id="APRIL"), _row_with_rtp(pair="RTPUSDT"),
            dict(pair="CLSUSDT", status="CLOSED", protective_tp_order_id="OLDCLOSED")]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows)
    async def go():
        await seed(); n = await eng.resting_tp_legacy_clear(); r = await read(); await engine.dispose(); return n, r
    n, r = asyncio.run(go())
    assert n == 1 and r[1].protective_tp_order_id is None and r[2].protective_tp_order_id == "T1" and r[3].protective_tp_order_id == "OLDCLOSED"


def test_blended_exit_type_on_every_surface():
    # deep C: the Exit Type table lists MAKER+TAKER; the fee saving counts the blended rows' maker part
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "const extOrder = ['MAKER', 'MAKER+TAKER'," in ui and "'MAKER+TAKER': 'text-cyan-400'" in ui
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert 'if etype in ("MAKER", "MAKER+TAKER"):' in main
    b = blend_exit(10, 4, 101.3, 99.0, MK, TK)
    hyp = b["price"] * 10 * TK
    assert abs((hyp - b["fee"]) - 4 * 101.3 * (TK - MK)) < 1e-12                  # = exactly the maker part's saving


def test_cancel_resting_tp_returns_binance_cancel_response():
    from services.binance_service import BinanceService
    svc = BinanceService.__new__(BinanceService)
    class _X:
        async def cancel_order(self, oid, sym):
            return {"id": oid, "status": "canceled", "filled": 2.0, "amount": 10.0, "average": 101.3, "info": {"status": "CANCELED"}}
    svc.exchange = _X(); svc._markets_loaded = True
    async def _nb(): return None
    svc._check_ban = _nb; svc.load_markets = _nb
    r = asyncio.run(svc.cancel_resting_tp("WILUSDT", "9"))
    assert r["result"] == "canceled" and r["status"] == "CANCELED" and r["filled"] == 2.0 and r["average"] == 101.3
    from services.resting_tp import cancel_view, stop_class
    assert cancel_view(r)[0] == "canceled" and cancel_view("gone") == ("gone", None) and cancel_view(None) == ("error", None)
    assert stop_class("STOP_LOSS") and stop_class("MAX_HOLD_TIME") and stop_class("MANUAL") and stop_class("BACKSTOP_STOP")
    assert not stop_class("FRENZY_TP") and not stop_class("WILLY_LATE_TP") and not stop_class("MANUAL_TP")


# ─────────────────────────────────────────────────────────────── review round 2 (caveman #1–#13 + deep B1 / S6 / S7)

def test_reconciler_slow_reads_happen_before_the_lock_and_switch_off_never_locks(monkeypatch):
    # #1: fill price + backstop status are read BEFORE the global close lock (the position is already gone); #5: a row with nothing recorded
    # and the switch OFF never takes the lock at all
    old = datetime.utcnow() - timedelta(minutes=5)
    main, T, eng, engine, S, seed, read, log, blocks = _reconcile_setup(monkeypatch, [_row_with_rtp(opened_at=old), dict(pair="OFFUSDT", opened_at=old,
                                                                        entry_strategy="FRENZY_LONG")], cancel="gone",
                                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    seen = []
    async def _fill(order):
        seen.append((order.pair, T._close_lock.locked())); return 101.0
    monkeypatch.setattr(main, "_get_actual_fill_price", _fill)
    async def go():
        await seed()
        await T._close_lock.acquire()
        async with S() as s:
            task = asyncio.create_task(main._close_orphan_orders(s, set()))
            await _real_sleep(0.05)
            pre = list(seen)
            T._close_lock.release()
            out = await task
        await engine.dispose(); return pre, out
    pre, out = asyncio.run(go())
    assert ("WILUSDT", True) in pre                      # read while the gate held the lock → i.e. outside / before it
    assert sorted(o["pair"] for o in out) == ["OFFUSDT", "WILUSDT"]
    # #5: a switch-OFF row with nothing recorded books while the close lock is HELD (it never takes it)
    main, T, eng, engine, S, seed, read, log, blocks = _reconcile_setup(monkeypatch, [dict(pair="OFFUSDT", opened_at=old, entry_strategy="FRENZY_LONG")])
    async def go2():
        await seed()
        await T._close_lock.acquire()
        async with S() as s:
            out = await asyncio.wait_for(main._close_orphan_orders(s, set()), 1.0)
        T._close_lock.release(); await engine.dispose(); return out
    assert [o["pair"] for o in asyncio.run(go2())] == ["OFFUSDT"] and log == []


def test_forced_close_full_limit_fill_is_the_tp_with_no_market_order(monkeypatch):
    # #2 + #3: STOP_LOSS forced on a blocked cancel; the live position is already flat → no market order; the reconcile finds the WHOLE
    # limit filled → booked FRENZY_TP, MAKER, at the limit — never STOP_LOSS
    reads = iter([{"status": "NEW", "filled": 0.0, "amount": 10.0}, {"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM}])
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], cancel="error", fetch=lambda oid: next(reads), pos=[0.0])
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    o = r[1]
    assert not [x for x in log if x[0] == "market_close"] and o.close_reason == "FRENZY_TP" and o.exit_order_type == "MAKER"
    assert abs(o.exit_price - LIM) < 1e-12 and abs(o.exit_fee - 10 * LIM * MK) < 1e-9 and (o.exit_slippage_pct or 0) == 0


def test_forced_close_is_sized_from_the_live_position(monkeypatch):
    # #3 (-2022): the limit filled more since the settle read → the reduce-only market order takes only what is LIVE
    reads = iter([{"status": "NEW", "filled": 2.0, "amount": 10.0, "average": LIM}, {"status": "CANCELED", "filled": 5.0, "amount": 10.0, "average": LIM}])
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp(resting_tp_filled_qty=2.0)], cancel="error",
                                                        fetch=lambda oid: next(reads), close_px=96.0, pos=[5.0, None])
    async def go():
        await seed(); c = await _close(eng, T, S, "MAX_HOLD_TIME", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert [x for x in log if x[0] == "market_close"] == [("market_close", "WIL/USDT:USDT", "LONG", 5.0)]
    assert r[1].resting_tp_filled_qty == 5.0 and abs(r[1].exit_price - (5 * LIM + 5 * 96.0) / 10) < 1e-9


def test_close_fallback_reconciles_the_limit_before_booking(monkeypatch):
    # #3: market close fails, the position is gone (CLOSE_FALLBACK) → the unverified limit is re-read: it FILLED → its TP, MAKER
    reads = iter([{"status": "NEW", "filled": 0.0, "amount": 10.0}, {"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM}])
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], cancel="error", fetch=lambda oid: next(reads),
                                                        pos=[RuntimeError("down")] * 5)
    async def _fail(symbol, side, amount):
        log.append(("market_close_fail", amount)); return None
    monkeypatch.setattr(T.binance_service, "close_position", _fail)
    async def _px(order, cp, exclude_ids=()):   # (321) the settled TP ids ride along
        return 96.0
    eng._fetch_actual_fill_price = _px
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert r[1].status == "CLOSED" and r[1].close_reason == "FRENZY_TP" and r[1].exit_order_type == "MAKER" and abs(r[1].exit_price - LIM) < 1e-9


def test_only_a_gtx_cross_turns_the_late_swap_into_market_now(monkeypatch):
    # #4: -5022 / EXPIRED = crossed → LATE_NOW; any other refusal (-2022 …) → 'failed' (software late TP + heal)
    late_open = datetime.utcnow() - timedelta(minutes=31)
    for res, want in (({"rejected": True, "crossed": True, "error": "-5022"}, "LATE_NOW"), ({"rejected": True, "error": "-2022 ReduceOnly"}, "failed")):
        T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp(opened_at=late_open)], place=res,
                                                            fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
        async def go():
            await seed(); r = await eng._rtp_reprice(1, late=True, price=LATE - 5 * TICK); await engine.dispose(); return r
        assert asyncio.run(go()) == want
    from services.binance_service import BinanceService
    svc = BinanceService.__new__(BinanceService)
    class _X:
        def __init__(self, err): self.err = err
        def price_to_precision(self, s, p): return f"{p:.2f}"
        def amount_to_precision(self, s, a): return f"{a:.3f}"
        async def create_order(self, **k): raise RuntimeError(self.err)
    async def _nb(): return None
    svc._check_ban = _nb; svc.load_markets = _nb; svc._markets_loaded = True
    svc.exchange = _X('binance {"code":-5022,"msg":"Due to the order could not be executed as maker"}')
    assert asyncio.run(svc.place_resting_tp("WILUSDT", "LONG", 1.0, 1.0)).get("crossed") is True
    svc.exchange = _X('binance {"code":-2022,"msg":"ReduceOnly Order is rejected."}')
    r = asyncio.run(svc.place_resting_tp("WILUSDT", "LONG", 1.0, 1.0))
    assert r.get("rejected") is True and not r.get("crossed")


def test_switch_off_orphan_clean_makes_no_exchange_call(monkeypatch):
    # #5: switch OFF and nothing recorded → the orphan clean never reads Binance
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict(entry_strategy="FRENZY_WILLY")],
                                                        th_kw=dict(resting_tp_willy_enabled=False, resting_tp_lite_enabled=False))
    async def go():
        await seed()
        async with S() as s:
            out = await eng.resting_tp_orphan_clean(s, now=0.0)
        await engine.dispose(); return out
    assert asyncio.run(go()) == [] and log == []


def test_probe_only_after_an_unknown_placement_and_never_blocks(monkeypatch):
    # #6: a cleanly cancelled / rejected placement (price recorded, id NULL, fill 0) never probes; an UNKNOWN one (fill NULL) does, and a
    # failed probe never blocks a non-stop close
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict(resting_tp_price=LIM, resting_tp_filled_qty=0.0)])
    async def go():
        await seed(); await _close(eng, T, S, "FRENZY_TP", 101.5); await engine.dispose()
    asyncio.run(go())
    assert "ro_read" not in [x[0] for x in log] and ("market_close", "WIL/USDT:USDT", "LONG", 10.0) in log
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict(resting_tp_price=LIM, resting_tp_filled_qty=None)])
    async def _ro_fail(pair=None):
        log.append(("ro_read", pair)); return None
    monkeypatch.setattr(T.binance_service, "fetch_open_reduce_only_orders", _ro_fail)
    async def go2():
        await seed(); c = await _close(eng, T, S, "FRENZY_TP", 101.5); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go2())
    assert ("ro_read", "WILUSDT") in log and r[1].status == "CLOSED" and "RTP_FILL_UNVERIFIED" in (r[1].notes or "")
    assert "WILUSDT" in eng.__dict__.get("_rtp_dirty_pairs", set())


def test_unknown_placement_is_stamped_for_the_probe(monkeypatch):
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict()], place=None)
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            o = (await s.execute(select(models.Order))).scalar_one()
            res = await eng._rtp_place_on(s, o, 1.25, "open")
        r = await read(); await engine.dispose(); return res, r
    res, r = asyncio.run(go())
    assert res == "unknown" and r[1].resting_tp_price == LIM and r[1].protective_tp_order_id is None and r[1].resting_tp_filled_qty is None


def test_gone_with_a_known_fill_needs_no_read(monkeypatch):
    # #7: the query job already saw FILLED; the cancel answers 'gone' → FILLED, no read
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], cancel="gone", fetch=None)
    async def go():
        await seed(); rows = await read(); st = eng._rtp_state(1, row=rows[1]); st.update(full=True, filled=10.0, avg=LIM)
        c = await _close(eng, T, S, "FRENZY_TP", LIM); r = await read(); await engine.dispose(); return r
    r = asyncio.run(go())
    assert "fetch" not in [x[0] for x in log] and r[1].exit_order_type == "MAKER" and r[1].close_reason == "FRENZY_TP"


def test_busy_flag_cleared_on_spawn_failure_and_by_the_watchdog(monkeypatch):
    # #8
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()])
    async def go():
        await seed(); rows = await read(); st = eng._rtp_state(1, row=rows[1])
        eng._rtp_spawn = lambda coro: (coro.close(), None)[1]
        await eng._rtp_gate(1, LIM, now=0.0, background=True)
        a = st["busy"]
        st.update(busy=True, busy_at=0.0, last_q=1e18)
        await eng._rtp_gate(1, LIM - TICK, now=7.0, background=True)
        b = st["busy"]
        await engine.dispose(); return a, b
    a, b = asyncio.run(go())
    assert a is False and b is False


def test_orphan_clean_rechecks_each_id_fresh(monkeypatch):
    # #9: a placement recorded AFTER the snapshot (here: committed while the open orders are read) is never cancelled
    OLD_MS = (time.time() - 600) * 1000
    ro = [{"id": "O4", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": 1.0, "amount": 5.0, "filled": 0.0, "timestamp": OLD_MS}]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[dict()], ro_orders=ro)
    async def _ro(pair=None):
        from sqlalchemy import update
        import models
        async with S() as s2:
            await s2.execute(update(models.Order).values(protective_tp_order_id="O4", resting_tp_price=LIM)); await s2.commit()
        return list(ro)
    monkeypatch.setattr(T.binance_service, "fetch_open_reduce_only_orders", _ro)
    async def go():
        await seed()
        async with S() as s:
            out = await eng.resting_tp_orphan_clean(s, now=0.0)
        await engine.dispose(); return out
    assert asyncio.run(go()) == [] and not [x for x in log if x[0] == "cancel"]


def test_dirty_pair_is_cleared_before_any_new_open(monkeypatch):
    # #10: after an unverified close, ANY new open on that pair first cancels the symbol's reduce-only orders; unreadable → refused
    ro = [{"id": "LEFT", "pair": "WILUSDT", "side": "sell", "type": "limit", "price": LIM, "amount": 10.0, "filled": 0.0, "timestamp": None}]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[], ro_orders=ro)
    eng._rtp_mark_dirty("WILUSDT")
    assert asyncio.run(eng.rtp_preopen_clear("OTHERUSDT")) is True and log == []
    assert asyncio.run(eng.rtp_preopen_clear("WILUSDT")) is True and ("cancel", "WILUSDT", "LEFT") in log
    assert "WILUSDT" not in eng.__dict__["_rtp_dirty_pairs"]
    eng._rtp_mark_dirty("WILUSDT")
    async def _none(pair=None):
        return None
    monkeypatch.setattr(T.binance_service, "fetch_open_reduce_only_orders", _none)
    assert asyncio.run(eng.rtp_preopen_clear("WILUSDT")) is False and "RESTING_TP_PREOPEN_BLOCK" in blocks
    eng_src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng_src.count("await self.rtp_preopen_clear(pair)") == 2                 # bot open AND manual open


def test_reconciler_failed_row_never_rides_the_next_commit(monkeypatch):
    # #12: a row whose booking raised is rolled back / expired — the next row's commit never books it half-way
    old = datetime.utcnow() - timedelta(minutes=5)
    main, T, eng, engine, S, seed, read, log, blocks = _reconcile_setup(monkeypatch, [_row_with_rtp(opened_at=old), dict(pair="MOMUSDT", opened_at=old,
                                                                        entry_strategy="MOMENTUM", backstop_algo_id=None)], cancel="gone",
                                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    real_commit = main.locked_commit
    calls = {"n": 0}
    async def _commit(session):
        calls["n"] += 1
        if calls["n"] == 2:   # 1 = the pre-lock commit, 2 = the resting-TP row's booking → fails
            raise RuntimeError("database is locked")
        return await real_commit(session)
    monkeypatch.setattr(main, "locked_commit", _commit)
    async def go():
        await seed()
        async with S() as s:
            out = await main._close_orphan_orders(s, set())
        r = await read()
        from sqlalchemy import select, func
        import models
        async with S() as s3:
            ntx = (await s3.execute(select(func.count(models.Transaction.id)))).scalar()
        await engine.dispose(); return out, r, ntx
    out, r, ntx = asyncio.run(go())
    assert [o["pair"] for o in out] == ["MOMUSDT"] and r[1].status == "OPEN" and r[2].status == "CLOSED" and ntx == 1


def test_reconciler_flags_a_likely_unrecorded_fill(monkeypatch):
    # S7: an UNKNOWN placement whose position vanished at ≈ the limit price
    old = datetime.utcnow() - timedelta(minutes=5)
    main, T, eng, engine, S, seed, read, log, blocks = _reconcile_setup(monkeypatch, [dict(opened_at=old, resting_tp_price=LIM, resting_tp_filled_qty=None)])
    async def _fill(order):
        return LIM
    monkeypatch.setattr(main, "_get_actual_fill_price", _fill)
    async def go():
        await seed()
        async with S() as s:
            await main._close_orphan_orders(s, set())
        r = await read(); await engine.dispose(); return r
    r = asyncio.run(go())
    assert r[1].close_reason == "EXTERNAL_CLOSE" and "RTP_UNRECORDED_FILL?" in (r[1].notes or "")


def test_remaining_quantity_precision_b1():
    from services.resting_tp import remaining_qty
    assert remaining_qty(0.3, 0.1) == 0.2 and remaining_qty(0.6, 0.4) == 0.2 and remaining_qty(0.3, 0.2) == 0.1 and remaining_qty(1.0, 1.2) == 0
    import services.trading_engine as T
    class _Ex:
        def __init__(self, step): self.step = step
        def amount_to_precision(self, s, a):   # ccxt TICK_SIZE + TRUNCATE
            import math
            n = math.floor(float(a) / self.step + 1e-12)
            return repr(round(n * self.step, 10))
    for step, q, f, want in ((0.1, 0.3, 0.1, 0.2), (0.01, 0.07, 0.04, 0.03), (0.001, 0.007, 0.004, 0.003), (0.1, 0.3, 0.2, 0.1)):
        bs = T.binance_service
        saved = (bs.exchange, bs._markets_loaded)
        try:
            bs.exchange, bs._markets_loaded = _Ex(step), True
            assert abs(T.TradingEngine._rtp_qty_snap("X/USDT:USDT", remaining_qty(q, f)) - want) < 1e-12, (step, q, f)
            raw = T.TradingEngine._rtp_qty_snap("X/USDT:USDT", q - f) if (q - f) != round(q - f, 10) else want
            assert abs(raw - want) < 1e-12                                     # even a raw 0.19999… is rounded before the truncation
        finally:
            bs.exchange, bs._markets_loaded = saved


def _fetch_by_id(oid):
    """(322) per-id reads: the market close 'm1' is final with 3 short of the 6 asked; anything else is the cancelled resting TP."""
    if str(oid) == "m1":
        return {"status": "FILLED", "filled": 3.0, "amount": 6.0, "average": 96.0}
    return {"status": "CANCELED", "filled": 4.0, "amount": 10.0, "average": LIM}


def _short_fill(monkeypatch, T, sweep_reports_fill=True):
    """(322) the market close reports it filled 3 less than asked — that 3 is the row's OWN residual (the sweep closes only that).
    sweep_reports_fill False: the residual sweep's own order reports no fill (judged on the live position only)."""
    _orig = T.binance_service.close_position
    n = {"i": 0}
    async def _c(symbol, side, amount):
        r = await _orig(symbol=symbol, side=side, amount=amount)
        n["i"] += 1
        return dict(r, filled=(amount - 3.0) if n["i"] == 1 else (amount if sweep_reports_fill else 0.0)) if r else r
    monkeypatch.setattr(T.binance_service, "close_position", _c)


def test_residual_after_a_partial_close_is_closed_or_guarded(monkeypatch):
    # B1: after a partial close the live position must be 0 — a residual is closed reduce-only; one that cannot be closed keeps the row
    # OPEN, re-places the backstop and retries (never booked CLOSED over a live residual)
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], close_px=96.0, pos=[3.0, None],
                                                        fetch=_fetch_by_id)
    _short_fill(monkeypatch, T)
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert [x[3] for x in log if x[0] == "market_close"] == [6.0, 3.0] and r[1].status == "CLOSED"
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], close_px=96.0, pos=[3.0, 3.0],
                                                        fetch=_fetch_by_id)
    async def _bk(pair, d, trig):
        log.append(("bk_place", pair)); return "B2"
    monkeypatch.setattr(T.binance_service, "place_backstop_stop", _bk)
    _short_fill(monkeypatch, T, sweep_reports_fill=False)
    async def go2():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go2())
    assert c is None and r[1].status == "OPEN" and ("bk_place", "WILUSDT") in log and 1 in T._exit_retry_queue and r[1].backstop_algo_id == "B2"


def test_residual_with_unknown_fill_is_still_swept(monkeypatch):
    """(322 review) the close did not report its fill → the whole residual is treated as ours (never left without a stop)."""
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], close_px=96.0, pos=[3.0, None],
                                                        fetch={"status": "CANCELED", "filled": 4.0, "amount": 10.0, "average": LIM})
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert [x[3] for x in log if x[0] == "market_close"] == [6.0, 3.0] and r[1].status == "CLOSED"


def test_unconfirmed_sweep_beside_an_operator_position_never_retries(monkeypatch):
    """(322 deep review) our close left 3 (confirmed), the operator holds 5 more; the sweep of our 3 is not confirmed (no fill reported,
    the operator added meanwhile) → the row is booked, never re-queued (a retry would re-send the full remainder into their coins)."""
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], close_px=96.0, pos=[8.0, 9.0],
                                                        fetch=_fetch_by_id)
    _short_fill(monkeypatch, T, sweep_reports_fill=False)
    T._exit_retry_queue.pop(1, None)
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert [x[3] for x in log if x[0] == "market_close"] == [6.0, 3.0] and r[1].status == "CLOSED" and 1 not in T._exit_retry_queue


def test_residual_that_is_not_ours_is_left_alone(monkeypatch):
    """(322) BTW #48: our close filled everything it was asked; 3 more on Binance belong to the operator → no sweep, the row books CLOSED."""
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[_row_with_rtp()], close_px=96.0, pos=[3.0, 3.0],
                                                        fetch={"status": "CANCELED", "filled": 4.0, "amount": 10.0, "average": LIM})
    _orig = T.binance_service.close_position
    async def _full(symbol, side, amount):
        r = await _orig(symbol=symbol, side=side, amount=amount)
        return dict(r, filled=amount) if r else r
    monkeypatch.setattr(T.binance_service, "close_position", _full)
    async def go():
        await seed(); c = await _close(eng, T, S, "STOP_LOSS", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    assert [x[3] for x in log if x[0] == "market_close"] == [6.0] and r[1].status == "CLOSED"


def test_residual_guard_writes_on_the_session_object_round3(monkeypatch):
    """Round-3 deep review: the residual guard must update the SAME identity-mapped Order the session holds (expire_on_commit=False) —
    a same-cycle exit-retry reads that object; a separate-session write would leave it with backstop_algo_id=None / the old reason."""
    import asyncio
    from datetime import datetime
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    from sqlalchemy import select
    import models, services.trading_engine as T
    eng_db = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    S = async_sessionmaker(eng_db, expire_on_commit=False)
    monkeypatch.setattr(T, "AsyncSessionLocal", S)
    monkeypatch.setattr(T.config.trading_config.thresholds, "broker_backstop_enabled", True, raising=False)
    async def _place(pair, direction, trig): return "BK2"
    monkeypatch.setattr(T.binance_service, "place_backstop_stop", _place)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    eng._record_filter_block = lambda *a, **k: None
    async def go():
        async with eng_db.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as s:
            o = models.Order(pair="RLCUSDT", direction="LONG", status="OPEN", entry_price=1.0, current_price=1.0, investment=10.0, leverage=10.0,
                             notional_value=100.0, quantity=100.0, confidence="STRONG_BUY", is_paper=False, opened_at=datetime.utcnow(),
                             entry_strategy="FRENZY_WILLY", backstop_algo_id=None, close_reason=None)
            s.add(o); await s.commit()
            await eng._rtp_guard_open_residual(s, o, "STOP_LOSS")
            same = (await s.execute(select(models.Order).where(models.Order.id == o.id))).scalar_one()   # identity-mapped, like the retry queue
            r = (same is o, same.backstop_algo_id, same.close_reason)
        await eng_db.dispose(); return r
    is_same, bk, why = asyncio.run(go())
    assert is_same and bk == "BK2" and why == "STOP_LOSS"


def test_qty_snap_failure_never_zeroes_a_stop_order(monkeypatch):
    """Final review: markets loaded but the symbol missing (amount_to_precision raises) → the rounded remainder, never 0.0."""
    import services.trading_engine as T
    class _X:
        def amount_to_precision(self, s, a): raise Exception("symbol not in markets")
    monkeypatch.setattr(T.binance_service, "_markets_loaded", True, raising=False)
    monkeypatch.setattr(T.binance_service, "exchange", _X(), raising=False)
    assert T.TradingEngine._rtp_qty_snap("NEW/USDT:USDT", 0.3 - 0.1) == 0.2
