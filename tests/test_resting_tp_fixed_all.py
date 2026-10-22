"""🎯 RESTING MAKER TP FOR EVERY FIXED TAKE-PROFIT (DECISION_LOG 310) — the 296 resting limit extended to: MANUAL trades in FIXED mode with a
TP > 0 (MANUAL_TP, incl. a mid-trade edit), any bot trade whose exit the operator overrode with a TP > 0 (OVERRIDE_TP), and FRENZY_LONG
(FRENZY_TP; none while frenzy_lock_arm_pct > 0). Pure target rules (services/resting_tp.rtp_row_target — read from the ROW, restart-proof) and
the engine wiring on the 296 harness (in-memory DB, every exchange call mocked — no network): open / manual open · sweep place / re-price /
withdraw · operator edit (set_exit_override) re-prices at once · the close path cancels FIRST and books blended · paper = live · the
software TP is the safety net on the operator-TP branch of both exit paths · switches OFF = the 296 behaviour · config / UI parity."""
import asyncio, os, sys, inspect, re
from datetime import datetime, timedelta
from types import SimpleNamespace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.resting_tp import (rtp_row_target, rtp_row_reason, rtp_row_enabled, rtp_fixed_reason, rtp_reason_for_limit, config_errors,
                                 RTP_SWITCHES, RTP_ROW_STRATEGIES, RTP_SLEEVES)
from services.exec_shadow import resting_tp_price
from test_resting_tp import _setup, _close, MK, TK, TICK, EF, LIM

ON = dict(resting_tp_manual_enabled=True, resting_tp_override_enabled=True, resting_tp_frenzy_long_enabled=True)


def _th(**kw):
    base = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0,
                frenzy_lock_arm_pct=0.0, resting_tp_willy_enabled=True, resting_tp_lite_enabled=True, resting_tp_grace_s=3.0,
                resting_tp_through_ticks=2, frenzy_willy_late_tp_minutes=30.0, frenzy_willy_late_tp_pct=0.25, **ON)
    base.update(kw)
    return SimpleNamespace(**base)


def _r(**kw):
    base = dict(entry_strategy="MANUAL", manual_exit_mode="FIXED", pattern_fixed_tp_pct=2.0, pattern_fixed_sl_pct=-3.0, exit_override_at=None,
                frenzy_lite_tp_used=None, direction="LONG", entry_price=100.0)
    base.update(kw)
    return SimpleNamespace(**base)


L2 = resting_tp_price("LONG", 100.0, EF, 2.0, MK, TICK)
L3 = resting_tp_price("LONG", 100.0, EF, 3.0, MK, TICK)
L15 = resting_tp_price("LONG", 100.0, EF, 1.5, MK, TICK)
NOW = datetime.utcnow


# ─────────────────────────────────────────────────────────────── pure target rules

def test_manual_fixed_is_the_only_manual_mode_with_a_resting_tp():
    th = _th()
    assert rtp_row_target(th, _r()) == (2.0, "MANUAL_TP")
    assert rtp_row_target(th, _r(manual_exit_mode=None)) == (2.0, "MANUAL_TP")             # None = FIXED (the exits' own default)
    assert rtp_row_target(th, _r(manual_exit_mode="FLOOR")) == (2.0, "MANUAL_TP")            # FLOOR's optional TP is a fixed line too (AGT #22)
    for m in ("MOMENTUM", "FRENZY", "BULLRUN", "BULLRUN_SL"):
        assert rtp_row_target(th, _r(manual_exit_mode=m)) is None, m
    for tp in (0, 0.0, None, -1.0, "x", float("nan")):
        assert rtp_row_target(th, _r(pattern_fixed_tp_pct=tp)) is None, tp
    assert rtp_row_target(_th(resting_tp_manual_enabled=False), _r()) is None
    # a MANUAL row is governed by the MANUAL switch even when overridden (AGT: FLOOR tp1 → FIXED tp2)
    assert rtp_row_target(_th(resting_tp_override_enabled=False), _r(exit_override_at=NOW())) == (2.0, "MANUAL_TP")


def test_operator_edit_moves_the_target():
    th = _th()
    before = _r(manual_exit_mode="FLOOR", pattern_fixed_tp_pct=1.0)                      # MANUAL:FLOOR sl=-3.0 tp=1.0
    after = _r(manual_exit_mode="FIXED", pattern_fixed_tp_pct=2.0, exit_override_at=NOW())
    assert rtp_row_target(th, before) == (1.0, "MANUAL_TP") and rtp_row_target(th, after) == (2.0, "MANUAL_TP")
    assert rtp_row_target(th, _r(pattern_fixed_tp_pct=3.0, exit_override_at=NOW())) == (3.0, "MANUAL_TP")


def test_override_on_a_bot_row():
    th = _th()
    for es in ("MOMENTUM", "FRENZY_WILLY", "FRENZY_LITE", "FLIP:X", None):
        assert rtp_row_target(th, _r(entry_strategy=es, manual_exit_mode=None, exit_override_at=NOW(), pattern_fixed_tp_pct=2.0)) == (2.0, "OVERRIDE_TP"), es
        assert rtp_row_target(th, _r(entry_strategy=es, manual_exit_mode=None, exit_override_at=NOW(), pattern_fixed_tp_pct=None)) is None, es
    # override switch OFF → none (the 296 behaviour: an override withdraws the sleeve's limit), even with the sleeve's own switch ON
    assert rtp_row_target(_th(resting_tp_override_enabled=False), _r(entry_strategy="FRENZY_WILLY", exit_override_at=NOW())) is None
    # a bot row's pattern_fixed_tp_pct WITHOUT an override is the bot's own pattern cap, never the resting TP
    assert rtp_row_target(th, _r(entry_strategy="MOMENTUM", exit_override_at=None, pattern_fixed_tp_pct=0.8)) is None
    assert rtp_row_target(th, _r(entry_strategy="FRENZY_WILLY", exit_override_at=None), held_minutes=0) == (1.25, "FRENZY_TP")


def test_frenzy_long_and_lock_arm():
    th = _th()
    assert rtp_row_target(th, _r(entry_strategy="FRENZY_LONG")) == (1.5, "FRENZY_TP")
    assert rtp_row_target(_th(frenzy_lock_arm_pct=3.0), _r(entry_strategy="FRENZY_LONG")) is None     # lock-then-trail: no fixed TP
    assert rtp_row_target(_th(resting_tp_frenzy_long_enabled=False), _r(entry_strategy="FRENZY_LONG")) is None
    assert rtp_row_target(_th(frenzy_tp_pct=0.0), _r(entry_strategy="FRENZY_LONG")) is None
    assert rtp_row_target(th, _r(entry_strategy="FRENZY_WIDE")) is None and "FRENZY_WIDE" not in RTP_SLEEVES
    assert rtp_row_target(th, _r(entry_strategy="FRENZY_LONG", frenzy_lite_tp_used=4.0)) == (1.5, "FRENZY_TP")   # LITE C TP: LITE only


def test_reasons_switches_and_config_errors():
    th = _th()
    assert rtp_fixed_reason(_r()) == "MANUAL_TP" and rtp_fixed_reason(_r(entry_strategy="FRENZY_LITE", exit_override_at=NOW())) == "OVERRIDE_TP"
    assert rtp_fixed_reason(_r(entry_strategy="FRENZY_LITE")) is None and rtp_fixed_reason(None) is None
    assert rtp_row_reason(th, _r(), L2, MK, EF) == "MANUAL_TP"
    assert rtp_row_reason(th, _r(entry_strategy="FRENZY_WILLY", exit_override_at=NOW()), L2, MK, EF) == "OVERRIDE_TP"
    late = resting_tp_price("LONG", 100.0, EF, 0.25, MK, TICK)
    assert rtp_row_reason(th, _r(entry_strategy="FRENZY_WILLY"), late, MK, EF) == "WILLY_LATE_TP" == rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 100.0, EF, late, MK)
    assert rtp_row_enabled(th, _r()) and not rtp_row_enabled(_th(resting_tp_manual_enabled=False), _r())
    assert rtp_row_enabled(th, _r(entry_strategy="MOMENTUM", exit_override_at=NOW())) and not rtp_row_enabled(th, _r(entry_strategy="MOMENTUM"))
    assert set(RTP_SWITCHES) == {"resting_tp_willy_enabled", "resting_tp_lite_enabled", "resting_tp_frenzy_long_enabled",
                                 "resting_tp_manual_enabled", "resting_tp_override_enabled", "resting_tp_lite_retry_enabled",   # 🔁 (316)
                                 "resting_tp_flatbase_enabled"}   # 🩳 (348)
    assert "MANUAL" in RTP_ROW_STRATEGIES and "FRENZY_LONG" in RTP_ROW_STRATEGIES
    assert config_errors(th) == []
    for k in ("resting_tp_manual_enabled", "resting_tp_override_enabled", "resting_tp_frenzy_long_enabled"):
        assert any(k in e for e in config_errors(_th(**{k: "yes"}))), k


# ─────────────────────────────────────────────────────────────── engine: open / sweep / edit

def _manual(**kw):
    d = dict(pair="AGTUSDT", entry_strategy="MANUAL", manual_exit_mode="FIXED", pattern_fixed_tp_pct=2.0, pattern_fixed_sl_pct=-3.0)
    d.update(kw)
    return d


def _place_ok(p, px):
    return {"id": "N-" + p, "price": px, "amount": 10.0}


def test_arm_places_for_manual_fixed_and_frenzy_long_only(monkeypatch):
    rows = [_manual(), _manual(pair="FLOUSDT", manual_exit_mode="MOMENTUM", pattern_fixed_tp_pct=1.0), _manual(pair="NOTUSDT", pattern_fixed_tp_pct=None),
            dict(pair="LONUSDT", entry_strategy="FRENZY_LONG"), dict(pair="WIDUSDT", entry_strategy="FRENZY_WIDE")]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place=_place_ok, th_kw=dict(ON, frenzy_tp_pct=1.5))
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            for o in (await s.execute(select(models.Order))).scalars().all():
                await eng._rtp_arm(s, o)
        r = await read(); await engine.dispose(); return r
    r = asyncio.run(go())
    assert [x for x in log if x[0] == "place"] == [("place", "AGTUSDT", "LONG", 10.0, L2), ("place", "LONUSDT", "LONG", 10.0, L15)]
    assert r[1].protective_tp_order_id == "N-AGTUSDT" and r[1].resting_tp_price == L2 and T._rtp_rt[1]["why"] == "MANUAL_TP"
    assert r[4].protective_tp_order_id == "N-LONUSDT" and T._rtp_rt[4]["why"] is None
    assert r[2].resting_tp_price is None and r[3].resting_tp_price is None and r[5].resting_tp_price is None and blocks == []


def test_switches_off_is_the_296_behaviour(monkeypatch):
    old = datetime.utcnow() - timedelta(minutes=5)
    rows = [_manual(), dict(pair="LONUSDT", entry_strategy="FRENZY_LONG"),
            _manual(pair="HASUSDT", protective_tp_order_id="H1", resting_tp_price=L2, resting_tp_filled_qty=0.0),          # manual OFF → withdraw
            dict(pair="OVRUSDT", exit_override_at=old, pattern_fixed_tp_pct=2.0, protective_tp_order_id="O1", resting_tp_price=LIM,
                 resting_tp_filled_qty=0.0)]                                                                               # override OFF → withdraw
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place=_place_ok,
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed()
        from sqlalchemy import select
        import models
        async with S() as s:
            for o in (await s.execute(select(models.Order).where(models.Order.id.in_((1, 2))))).scalars().all():
                assert await eng._rtp_arm(s, o) is None
        out = await eng.resting_tp_sweep(now=0.0)
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert not [x for x in log if x[0] == "place"]
    assert sorted(out) == [("withdraw", 3, "withdrawn"), ("withdraw", 4, "withdrawn")]
    assert r[3].protective_tp_order_id is None and r[4].protective_tp_order_id is None


def test_sweep_place_reprice_withdraw_for_operator_tps(monkeypatch):
    old = datetime.utcnow() - timedelta(minutes=45)                                     # past the WILLY late minute: an override never swaps
    rows = [_manual(),                                                                                                       # 1 FIXED, none → heal
            _manual(pair="EDTUSDT", pattern_fixed_tp_pct=3.0, protective_tp_order_id="E1", resting_tp_price=L2, resting_tp_filled_qty=0.0,
                    exit_override_at=old),                                                                                    # 2 edited 2 → 3: re-price
            _manual(pair="FLRUSDT", manual_exit_mode="MOMENTUM", protective_tp_order_id="F1", resting_tp_price=L2, resting_tp_filled_qty=0.0),  # 3 MOMENTUM → withdraw
            _manual(pair="TP0USDT", pattern_fixed_tp_pct=None, protective_tp_order_id="Z1", resting_tp_price=L2, resting_tp_filled_qty=0.0),  # 4 TP none → withdraw
            dict(pair="WOVUSDT", opened_at=old, exit_override_at=old, pattern_fixed_tp_pct=2.0, protective_tp_order_id="W1", resting_tp_price=LIM,
                 resting_tp_filled_qty=0.0),                                                                                  # 5 WILLY override tp 2 → re-price to 2
            dict(pair="WNOUSDT", exit_override_at=old, pattern_fixed_tp_pct=None, protective_tp_order_id="X1", resting_tp_price=LIM,
                 resting_tp_filled_qty=0.0),                                                                                  # 6 override without TP → withdraw
            dict(pair="MOMUSDT", entry_strategy="MOMENTUM", pattern_fixed_tp_pct=0.8)]                                       # 7 bot pattern cap → untouched
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place=_place_ok, th_kw=ON,
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    monkeypatch.setattr(T, "RTP_SWEEP_MAX_ACTIONS", 20)
    async def go():
        await seed(); out = await eng.resting_tp_sweep(now=100.0)
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert sorted(out) == [("heal", 1, "placed"), ("reprice", 2, "replaced"), ("reprice", 5, "replaced"),
                           ("withdraw", 3, "withdrawn"), ("withdraw", 4, "withdrawn"), ("withdraw", 6, "withdrawn")]
    assert r[1].resting_tp_price == L2 and r[1].protective_tp_order_id == "N-AGTUSDT"
    assert r[2].resting_tp_price == L3 and [x[:3] for x in log if x[1] == "EDTUSDT" and x[0] in ("cancel", "place")] == [("cancel", "EDTUSDT", "E1"), ("place", "EDTUSDT", "LONG")]
    assert r[5].resting_tp_price == L2 and r[5].protective_tp_order_id == "N-WOVUSDT" and T._rtp_rt[5]["why"] == "OVERRIDE_TP"
    assert r[3].protective_tp_order_id is None and r[4].protective_tp_order_id is None and r[6].protective_tp_order_id is None
    assert not [x for x in log if x[1:2] == ("MOMUSDT",)] and r[7].resting_tp_price is None
    assert T._rtp_rt[1]["why"] == "MANUAL_TP" and T._rtp_rt[2]["why"] == "MANUAL_TP"


def test_restart_reseeds_the_operator_reason_from_the_row(monkeypatch):
    rows = [_manual(protective_tp_order_id="A1", resting_tp_price=L2, resting_tp_filled_qty=0.0)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, th_kw=ON,
                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": L2})
    async def go():
        await seed()
        T._rtp_rt.clear()                                          # a restart: no memory
        out = await eng.resting_tp_sweep(now=0.0)
        g = await eng._rtp_gate(1, L2, now=1.0)
        await engine.dispose(); return out, g
    out, g = asyncio.run(go())
    assert out == [] and g == ("FILLED", "MANUAL_TP") and not [x for x in log if x[0] in ("place", "cancel")]


def test_operator_edit_places_and_reprices_at_once(monkeypatch):
    # AGT 2026-10-19: MANUAL FLOOR sl −3 tp 1 (no resting TP) → the operator switches to FIXED tp 2 → the limit rests at once; edit to tp 3 →
    # cancel + re-place; edit to no TP → withdrawn, nothing placed
    rows = [_manual(manual_exit_mode="FLOOR", pattern_fixed_tp_pct=1.0)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place=_place_ok, th_kw=dict(ON, broker_backstop_enabled=False),
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def _cache(db):
        return None
    eng.update_orders_cache = _cache
    async def go():
        await seed()
        out = []
        for sl, tp in ((None, 2.0), (None, 3.0), (-3.0, None)):
            async with S() as s:
                await eng.set_exit_override(s, 1, sl_pct=sl, tp_pct=tp)
            out.append((await read())[1])
        await engine.dispose(); return out
    a, b, c = asyncio.run(go())
    assert a.manual_exit_mode == "FIXED" and a.protective_tp_order_id == "N-AGTUSDT" and a.resting_tp_price == L2
    assert b.resting_tp_price == L3 and b.protective_tp_order_id == "N-AGTUSDT"
    assert c.protective_tp_order_id is None and c.pattern_fixed_tp_pct is None
    seq = [x[:3] for x in log if x[0] in ("cancel", "place")]
    assert seq == [("place", "AGTUSDT", "LONG"), ("cancel", "AGTUSDT", "N-AGTUSDT"), ("place", "AGTUSDT", "LONG"), ("cancel", "AGTUSDT", "N-AGTUSDT")]
    assert 1 not in T._rtp_rt


def test_override_on_a_willy_row_rests_at_the_operator_tp(monkeypatch):
    rows = [dict(pair="WILUSDT", protective_tp_order_id="T1", resting_tp_price=LIM, resting_tp_filled_qty=0.0)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place=_place_ok, th_kw=dict(ON, broker_backstop_enabled=False),
                                                        fetch={"status": "CANCELED", "filled": 0.0, "amount": 10.0})
    async def _cache(db):
        return None
    eng.update_orders_cache = _cache
    async def go():
        await seed()
        async with S() as s:
            await eng.set_exit_override(s, 1, sl_pct=-2.0, tp_pct=2.0)
        r = await read()
        g = dict(T._rtp_rt.get(1) or {})
        await engine.dispose(); return r, g
    r, g = asyncio.run(go())
    assert [x[:3] for x in log if x[0] in ("cancel", "place")] == [("cancel", "WILUSDT", "T1"), ("place", "WILUSDT", "LONG")]
    assert r[1].resting_tp_price == L2 and r[1].protective_tp_order_id == "N-WILUSDT" and g["why"] == "OVERRIDE_TP" and g["es"] == "FRENZY_WILLY"


def test_override_refused_when_the_sleeve_limit_already_filled(monkeypatch):
    """deep review M1: the limit is cancelled BEFORE the override is written — a WILLY limit that filled meanwhile keeps its own reason
    (FRENZY_TP, never OVERRIDE_TP) and the edit is refused (the position is flat)."""
    rows = [dict(pair="WILUSDT", protective_tp_order_id="T1", resting_tp_price=LIM, resting_tp_filled_qty=0.0)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, place=_place_ok, th_kw=dict(ON, broker_backstop_enabled=False),
                                                        fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": LIM})
    async def _cache(db):
        return None
    eng.update_orders_cache = _cache
    async def go():
        await seed()
        err = None
        async with S() as s:
            try:
                await eng.set_exit_override(s, 1, sl_pct=-2.0, tp_pct=2.0)
            except ValueError as e:
                err = str(e)
        r = await read()
        g = dict(T._rtp_rt.get(1) or {})
        await engine.dispose(); return err, r, g
    err, r, g = asyncio.run(go())
    assert err and "already filled" in err
    assert r[1].exit_override_at is None and r[1].pattern_fixed_tp_pct != 2.0
    assert g.get("why") is None and not [x for x in log if x[0] == "place"]


# ─────────────────────────────────────────────────────────────── engine: gate / close / paper

def test_gate_books_the_operator_reason(monkeypatch):
    rows = [_manual(protective_tp_order_id="A1", resting_tp_price=L2, resting_tp_filled_qty=0.0)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, th_kw=ON, fetch={"status": "NEW", "filled": 0.0, "amount": 10.0})
    async def go():
        await seed(); rows_ = await read(); eng._rtp_state(1, row=rows_[1])
        a = await eng._rtp_gate(1, L2 + 2 * TICK, now=0.0)
        b = await eng._rtp_gate(1, L2 + 2 * TICK, now=3.5)
        await engine.dispose(); return a, b
    assert asyncio.run(go()) == (None, ("FALLBACK", "MANUAL_TP"))
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, paper=True, th_kw=ON,
                                                        rows=[_manual(resting_tp_price=L2, resting_tp_filled_qty=0.0, exit_override_at=NOW())])
    async def go2():
        await seed(); rows_ = await read(); eng._rtp_state(1, row=rows_[1])
        a = await eng._rtp_gate(1, L2, now=0.0); b = await eng._rtp_gate(1, L2 + TICK, now=1.0)
        await engine.dispose(); return a, b
    assert asyncio.run(go2()) == (None, ("PAPER_FILL", "MANUAL_TP")) and log == []


def test_override_state_never_runs_the_willy_late_swap(monkeypatch):
    old = datetime.utcnow() - timedelta(minutes=45)
    rows = [dict(pair="WILUSDT", opened_at=old, exit_override_at=old, pattern_fixed_tp_pct=2.0, protective_tp_order_id="T1", resting_tp_price=L2,
                 resting_tp_filled_qty=0.0)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, th_kw=ON, place=_place_ok)
    async def go():
        await seed(); rows_ = await read(); eng._rtp_state(1, row=rows_[1])
        g = await eng._rtp_gate(1, 100.1, now=0.0)
        await engine.dispose(); return g
    assert asyncio.run(go()) is None and log == []


def test_manual_close_cancels_first_and_books_blended(monkeypatch):
    rows = [_manual(protective_tp_order_id="A1", resting_tp_price=L2, resting_tp_filled_qty=0.0, backstop_algo_id="B1")]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=rows, th_kw=ON, close_px=96.0,
                                                        fetch={"status": "CANCELED", "filled": 4.0, "amount": 10.0, "average": L2})
    async def go():
        await seed(); c = await _close(eng, T, S, "MANUAL_SL", 96.0); r = await read(); await engine.dispose(); return c, r
    c, r = asyncio.run(go())
    o = r[1]
    kinds = [x[0] for x in log]
    assert kinds.index("cancel") < kinds.index("bk_cancel") < kinds.index("market_close")
    assert [x for x in log if x[0] == "market_close"] == [("market_close", "AGT/USDT:USDT", "LONG", 6.0)]       # ONLY the remainder
    want_fee = 4 * L2 * MK + 6 * 96.0 * TK
    assert o.close_reason == "MANUAL_SL" and o.exit_order_type == "MAKER+TAKER" and abs(o.exit_fee - want_fee) < 1e-9
    assert abs(o.pnl - ((L2 - 100) * 4 + (96.0 - 100) * 6 - 0.45 - want_fee)) < 1e-6


def test_manual_full_fill_books_manual_tp_maker_and_override_tp(monkeypatch):
    for kw, want in ((_manual(), "MANUAL_TP"), (dict(pair="WILUSDT", exit_override_at=NOW(), pattern_fixed_tp_pct=2.0), "OVERRIDE_TP")):
        row = dict(kw, protective_tp_order_id="A1", resting_tp_price=L2, resting_tp_filled_qty=0.0, backstop_algo_id="B1")
        T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, rows=[row], th_kw=ON, cancel="gone",
                                                            fetch={"status": "FILLED", "filled": 10.0, "amount": 10.0, "average": L2})
        async def go():
            await seed(); c = await _close(eng, T, S, "MANUAL", 97.0); r = await read(); await engine.dispose(); return c, r
        c, r = asyncio.run(go())
        o = r[1]
        assert o.status == "CLOSED" and o.close_reason == want and o.exit_order_type == "MAKER" and abs(o.exit_price - L2) < 1e-12
        assert abs(o.exit_fee - 10 * L2 * MK) < 1e-9 and not [x for x in log if x[0] == "market_close"]


def test_paper_parity_for_a_manual_row(monkeypatch):
    rows = [_manual(resting_tp_price=L2), _manual(pair="P2USDT", resting_tp_price=L2)]
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, paper=True, rows=rows, th_kw=ON)
    async def go():
        await seed()
        await _close(eng, T, S, "MANUAL_TP", L2 + TICK, oid=1)        # ≥ 1 tick through → the limit, maker, MANUAL_TP
        await _close(eng, T, S, "MANUAL_SL", 96.0, oid=2)             # never through → taker at the price
        r = await read(); await engine.dispose(); return r
    r = asyncio.run(go())
    assert log == []
    assert r[1].close_reason == "MANUAL_TP" and r[1].exit_order_type == "MAKER" and abs(r[1].exit_price - L2) < 1e-12
    assert abs(r[1].exit_fee - 10 * L2 * MK) < 1e-9 and r[1].resting_tp_filled_qty == 10.0
    assert r[2].close_reason == "MANUAL_SL" and r[2].exit_order_type == "TAKER"


def test_realtime_operator_tp_is_the_safety_net_and_the_stop_still_fires(monkeypatch):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    closed, gated = [], []

    class _Res:
        def __init__(self, o): self.o = o
        def scalar_one_or_none(self): return self.o

    class _DB:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, *a, **k): return _Res(object())

    async def _close_(db, order, price, reason): closed.append(reason); return True
    async def _gate(oid, price, now=None, background=False):
        gated.append((oid, price, background)); return None
    monkeypatch.setattr(T, "AsyncSessionLocal", lambda: _DB())
    monkeypatch.setattr(eng, "close_position", _close_, raising=False)
    monkeypatch.setattr(eng, "_rtp_gate", _gate, raising=False)
    fee = float(getattr(T.config.trading_config, 'taker_fee', T.config.trading_config.trading_fee))
    opened = T.datetime.utcnow()

    async def drive(price, alive):
        closed.clear(); gated.clear(); T._rtp_rt.clear()
        if alive:
            T._rtp_rt[1] = dict(pair="ZZZUSDT", dir="LONG", oid="A1", px=103.1, qty=10.0, filled=0.0, full=False, avg=None, tick=0.01, first=None,
                                last_q=-1e18, es="MANUAL", opened_at=opened, okey=T.ts_key(opened), paper=False, ef=0.045, entry=100.0,
                                swap_next=-1e18, busy=False, why="MANUAL_TP")
        T._open_orders_cache["ZZZUSDT"] = [{'id': 1, 'direction': "LONG", 'entry_price': 100.0, 'quantity': 10.0, 'entry_fee': 1000.0 * fee,
                                            'confidence': 'STRONG_BUY', 'opened_at': opened, 'entry_strategy': 'MANUAL', 'manual_exit_mode': 'FIXED',
                                            'pattern_fixed_sl_pct': -1.5, 'pattern_fixed_tp_pct': 3.0, 'stop_loss': -0.7, 'peak_pnl': 0.0,
                                            'trough_pnl': 0.0, 'leverage': 20.0, 'notional_value': 1000.0, 'investment': 50.0}]
        await eng.check_realtime_stop_loss("ZZZUSDT", price)
        T._rtp_rt.clear(); T._open_orders_cache.clear()
        return list(closed), list(gated)
    assert asyncio.run(drive(104.0, False)) == (["MANUAL_TP"], [])                          # no resting TP: the software TP as before
    c, g = asyncio.run(drive(104.0, True))
    assert c == [] and g == [(1, 104.0, True)]                                              # alive: the gate decides, in the background
    assert asyncio.run(drive(98.0, True))[0] == ["MANUAL_SL"]                              # the stop never waits on the limit


# ─────────────────────────────────────────────────────────────── source pins + config / UI parity

def test_source_pins_every_path():
    import services.trading_engine as T
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    mo = inspect.getsource(T.TradingEngine.open_manual_position)
    assert "await self._rtp_arm(db, order)" in mo and mo.index("await self.update_orders_cache(db)") < mo.index("await self._rtp_arm(db, order)")
    ov = inspect.getsource(T.TradingEngine.set_exit_override)
    assert ov.index('await self._rtp_withdraw(order.id, "exit override")') < ov.index("_h = await self._rtp_heal(order.id)")
    sw = inspect.getsource(T.TradingEngine.resting_tp_sweep)
    assert "Order.entry_strategy.in_(RTP_ROW_STRATEGIES), Order.exit_override_at.isnot(None)" in sw and "rtp_row_target(th, o, " in sw
    assert "exit_override_at is not None" not in inspect.getsource(T.TradingEngine._rtp_heal)
    assert eng.count("_rtp_on_mf = _ovr_c is None and self._rtp_alive(order.id, row=order)") == 1
    assert eng.count("_rtp_on_mn = self._rtp_alive(order_id, pair=pair, opened_at=order_info.get('opened_at'))") == 1
    assert "not _rtp_on_mf and _mf_pnl >= float(_mf_tp)" in eng and "not _rtp_on_mn and pnl_pct >= float(_mn_tp)" in eng
    assert "rtp_reason_for_limit(config.trading_config.thresholds, order." not in eng                # every row reason goes through rtp_row_reason
    assert "rtp_row_enabled(config.trading_config.thresholds, o)" in inspect.getsource(T.TradingEngine.rtp_row_active)
    assert "for k in RTP_SWITCHES" in inspect.getsource(T.TradingEngine.rtp_any_switch_on)
    # the DL 308 guard (backstop heal skips the bot's own in-flight close) is untouched
    assert "test_backstop_heal_inflight" in " ".join(os.listdir(os.path.join(ROOT, "tests")))


def test_config_ui_parity():
    import json, config
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
    th = cfg.get("thresholds", cfg)
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for k, i in (("resting_tp_manual_enabled", "config-rtp-manual-enabled"), ("resting_tp_override_enabled", "config-rtp-override-enabled"),
                 ("resting_tp_frenzy_long_enabled", "config-rtp-frenzy-long-enabled")):
        assert th[k] is True and getattr(config.SignalThresholds(), k) is True, k
        assert ui.count(f'id="{i}"') == 1 and ui.count(f"getElementById('{i}')") == 2, i         # input + load + save
        assert f"{k}: document.getElementById('{i}')?.checked ?? true" in ui and f"config.thresholds.{k} !== false" in ui
        assert ui.count(f"_bt.{k}") >= 1                                                         # the shared config-export line
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_cfg_errs.extend(_rtp_cfg_errors(new_config.thresholds))" in main
