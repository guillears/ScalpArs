"""🔂 Nov-11 MOMENTUM LONG SECOND CHANCE (DECISION_LOG 532) — services/ml_second_chance.py + the engine's close-funnel intercept.

Invariants pinned here:
  · zone = MOMENTUM LONG with entry_adx_delta ≤ bound; a missing / garbage ADX stamp, any other sleeve, a short, or the switch off → today's exit.
  · only RUNNER_TRAIL / HARD_TP_LADDER (L-suffixed, RH_-prefixed) are intercepted; RUNNER_TRAIL_LATE, stops, SC_* never.
  · phase 2 = first of SC_FLOOR (L − floor) · SC_TRAIL (post-trigger high − N×ATR, the high never below L) · SC_CLOCK (age); no ATR = floor only.
  · the live phase-2 loop reproduces the study's pricing (scripts/study_ml_sc_refine_1111.sc, REF_d0.3_k1) tick for tick.
  · the engine intercept: first runner close → held (returns None, state set, cache flag released); a second runner close of a held
    trade → still held; SC_* / stop closes pass; out-of-zone / switch off → close goes through.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from services.ml_second_chance import (sc_params, sc_in_zone, sc_base_reason, sc_phase2_exit, net_pnl_pct, SC_REASONS, SC_URGENT)


def _th(**kw):
    d = dict(ml_second_chance_enabled=True, ml_second_chance_adx_delta_max=0.0, ml_second_chance_floor_pct=0.30,
             ml_second_chance_trail_atr_mult=1.0, ml_second_chance_clock_min=180.0)
    d.update(kw)
    return SimpleNamespace(**d)


P = sc_params(_th())


# ───────────────────────────────────────────── params / zone
def test_params_defaults_when_missing():
    p = sc_params(SimpleNamespace())
    assert p == dict(enabled=False, adx_delta_max=0.0, floor=0.30, trail_mult=1.0, clock_min=180.0)
    assert sc_params(_th(ml_second_chance_floor_pct="x"))['floor'] == 0.30


def test_zone():
    assert sc_in_zone(P, "MOMENTUM", "LONG", -0.4)
    assert sc_in_zone(P, "MOMENTUM", "LONG", 0.0)          # flat counts (≤ 0, the study's zone)
    assert not sc_in_zone(P, "MOMENTUM", "LONG", 0.39)       # ICP 11-11: rising ADX → today's exit
    for d in (None, "x", float("nan")):
        assert not sc_in_zone(P, "MOMENTUM", "LONG", d)     # fail closed
    for s in ("FRENZY_LITE", "FLIP:FAN", "MANUAL", "SPIKE_BOUNCE", "BULLRUN_LONG", None, ""):
        assert not sc_in_zone(P, s, "LONG", -1.0)
    assert not sc_in_zone(P, "MOMENTUM", "SHORT", -1.0)
    assert not sc_in_zone(sc_params(_th(ml_second_chance_enabled=False)), "MOMENTUM", "LONG", -1.0)
    assert sc_in_zone(sc_params(_th(ml_second_chance_adx_delta_max=0.5)), "MOMENTUM", "LONG", 0.39)


def test_intercepted_reasons():
    for r in ("RUNNER_TRAIL", "RUNNER_TRAIL L2", "RH_RUNNER_TRAIL", "RH_RUNNER_TRAIL L1"):
        assert sc_base_reason(r) == "RUNNER_TRAIL"
    for r in ("HARD_TP_LADDER L1", "RH_HARD_TP_LADDER L3"):
        assert sc_base_reason(r) == "HARD_TP_LADDER"
    for r in ("RUNNER_TRAIL_LATE", "STOP_LOSS L1", "STOP_LOSS_WIDE L1", "MAX_HOLD_TIME", "NO_EXPANSION", "MANUAL", "HARD_TP L1",
              "TRAILING_STOP L2", "RH_HARD_STOP", "EMA13_CROSS", None, "") + SC_REASONS:
        assert sc_base_reason(r) is None
    assert SC_URGENT == ("SC_FLOOR",)


# ───────────────────────────────────────────── phase-2 exit
def test_floor_trail_clock():
    # L = 0.55, ATR 0.45: right after the trigger the floor (0.25) binds before the trail (0.10)
    assert sc_phase2_exit(0.30, 0.55, 0.55, 0.45, 30, P) is None
    assert sc_phase2_exit(0.25, 0.55, 0.55, 0.45, 30, P) == "SC_FLOOR"
    # second leg to 1.80 → the trail sits at 1.35
    assert sc_phase2_exit(1.36, 0.55, 1.80, 0.45, 60, P) is None
    assert sc_phase2_exit(1.35, 0.55, 1.80, 0.45, 60, P) == "SC_TRAIL"
    # the clock
    assert sc_phase2_exit(1.00, 0.55, 1.10, 0.45, 179.9, P) is None
    assert sc_phase2_exit(1.00, 0.55, 1.10, 0.45, 180.0, P) == "SC_CLOCK"
    # a small ATR: the trail binds before the floor
    assert sc_phase2_exit(0.40, 0.55, 0.55, 0.15, 30, P) == "SC_TRAIL"
    # floor beats trail when both hit
    assert sc_phase2_exit(0.20, 0.55, 0.60, 0.10, 30, P) == "SC_FLOOR"


def test_no_atr_is_floor_only_and_high_never_below_L():
    for atr in (None, 0, -1, "x"):
        assert sc_phase2_exit(0.26, 0.55, 2.0, atr, 30, P) is None
        assert sc_phase2_exit(0.25, 0.55, 2.0, atr, 30, P) == "SC_FLOOR"


def test_floor_binds_with_missing_high():
    assert sc_phase2_exit(0.24, 0.55, None, 0.45, 30, P) == "SC_FLOOR"
    assert sc_phase2_exit(0.30, 0.55, 0.10, 0.45, 30, P) is None   # a high below L is treated as L
    assert sc_phase2_exit(None, 0.55, 0.55, 0.45, 30, P) is None
    assert sc_phase2_exit(0.0, None, 0.55, 0.45, 30, P) is None
    assert sc_phase2_exit(1.0, 0.55, 1.0, 0.45, None, P) is None   # no age → no clock


def test_net_pnl_pct_matches_engine_formula():
    e, c, q, fee, tk = 2.614, 2.623, 13987.18, 16.45, 0.0005
    exp = ((c - e) * q - fee - c * q * tk) / (e * q) * 100
    assert net_pnl_pct("LONG", e, c, q, fee, tk) == pytest.approx(exp)
    assert net_pnl_pct("LONG", 0, c, q, fee, tk) is None
    assert net_pnl_pct("LONG", e, c, 0, fee, tk) is None


# ───────────────────────────────────────────── parity with the study (REF_d0.3_k1)
def _live_loop(net, t, k0, atr, o_ms, prm):
    L = net[k0]; high = L
    for i in range(k0 + 1, len(net)):
        high = max(high, net[i])
        why = sc_phase2_exit(net[i], L, high, atr, (t[i] - o_ms) / 60_000, prm)
        if why:
            return i, why
    return len(net) - 1, "END_OF_DATA"


def test_parity_with_study_pricing():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    SR = pytest.importorskip("study_ml_sc_refine_1111")
    rng = np.random.default_rng(7)
    for trial in range(300):
        n = 400
        t = np.arange(n) * 15_000 + 1_000_000
        net = np.cumsum(rng.normal(0, 0.05, n)) + 0.5
        k0 = int(rng.integers(5, 100))
        atr = float(rng.choice([0.2, 0.35, 0.45, 0.8]))
        fx = dict(atr=atr, o_ms=int(t[0]) - int(rng.integers(0, 150)) * 60_000)
        Pp = dict(t=t, net=net)
        clock = 180 * 60_000
        exp = SR.sc(Pp, fx, k0, SR.G["REF_d0.3_k1"], clock)
        got = _live_loop(net, t, k0, atr, fx["o_ms"], P)
        assert got[0] == exp[0] and got[1] == exp[1], (trial, got, exp)


# ───────────────────────────────────────────── engine intercept (close funnel)
def _engine(monkeypatch, th):
    import services.trading_engine as TE
    eng = TE.TradingEngine.__new__(TE.TradingEngine)
    eng._bg_tasks = set()
    monkeypatch.setattr(TE.config.trading_config, "thresholds", th, raising=False)
    persisted = []

    async def _p(order_id, armed_at, level, base):
        persisted.append((order_id, level, base))
    monkeypatch.setattr(eng, "_persist_sc_async", _p, raising=False)
    TE._SC_STATE.clear()
    return TE, eng, persisted


def _order(**kw):
    d = dict(id=9001, pair="TESTUSDT", direction="LONG", entry_strategy="MOMENTUM", entry_adx_delta=-0.3, entry_price=1.0,
             quantity=1000.0, entry_fee=0.2, entry_atr_pct=0.45, sc_armed_at=None, sc_base_pnl=None, sc_base_reason=None, sc_high_pct=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_intercept_holds_then_keeps_holding(monkeypatch):
    TE, eng, persisted = _engine(monkeypatch, _th())
    o = _order()
    TE._open_orders_cache[o.pair] = [{'id': o.id, '_closing_in_progress': True}]
    try:
        async def run():
            assert await eng._sc_intercept(o, 1.006, "RUNNER_TRAIL") is True
            await asyncio.sleep(0)
            st = TE._SC_STATE[o.id]
            assert st['L'] == pytest.approx(net_pnl_pct("LONG", 1.0, 1.006, 1000.0, 0.2, TE.config.trading_config.taker_fee
                                                        if hasattr(TE.config.trading_config, 'taker_fee') else TE.config.trading_config.trading_fee))
            assert TE._open_orders_cache[o.pair][0]['_closing_in_progress'] is False
            assert TE._open_orders_cache[o.pair][0]['sc_base_pnl'] == st['L']
            # the monitor's runner close a scan later: still held, L unchanged
            assert await eng._sc_intercept(o, 1.002, "RUNNER_TRAIL L2") is True
            assert TE._SC_STATE[o.id]['L'] == st['L']
            # SC / stop / clock closes pass
            for r in ("SC_FLOOR", "SC_TRAIL", "SC_CLOCK", "STOP_LOSS L1", "MAX_HOLD_TIME", "MANUAL"):
                assert await eng._sc_intercept(o, 1.0, r) is False
        asyncio.run(run())
        assert len(persisted) == 1 and persisted[0][2] == "RUNNER_TRAIL"
    finally:
        TE._open_orders_cache.pop(o.pair, None)
        TE._SC_STATE.clear()


def test_intercept_passes_out_of_zone_and_switch_off(monkeypatch):
    TE, eng, _ = _engine(monkeypatch, _th())

    async def run(o, th=None):
        if th is not None:
            monkeypatch.setattr(TE.config.trading_config, "thresholds", th, raising=False)
        return await eng._sc_intercept(o, 1.006, "RUNNER_TRAIL")
    assert asyncio.run(run(_order(entry_adx_delta=0.39))) is False
    assert asyncio.run(run(_order(entry_adx_delta=None))) is False
    assert asyncio.run(run(_order(entry_strategy="FRENZY_LITE"))) is False
    assert asyncio.run(run(_order(direction="SHORT"))) is False
    assert asyncio.run(run(_order(), _th(ml_second_chance_enabled=False))) is False
    assert TE._SC_STATE == {}


def test_held_trade_resumes_after_restart_and_switch_off(monkeypatch):
    """A trade already held (sc_* stamped on its row) stays held even if the switch went off — it runs to its own exits."""
    TE, eng, persisted = _engine(monkeypatch, _th(ml_second_chance_enabled=False))
    o = _order(sc_armed_at=datetime.utcnow() - timedelta(minutes=5), sc_base_pnl=0.55, sc_base_reason="RUNNER_TRAIL", sc_high_pct=0.9)
    try:
        assert asyncio.run(eng._sc_intercept(o, 1.0, "RUNNER_TRAIL")) is True
        st = TE._SC_STATE[o.id]
        assert st['L'] == 0.55 and st['high'] == 0.9 and persisted == []
    finally:
        TE._SC_STATE.clear()


def test_state_seed_from_cache_dict():
    import services.trading_engine as TE
    eng = TE.TradingEngine.__new__(TE.TradingEngine)
    TE._SC_STATE.clear()
    assert eng._sc_state_for(1, {'sc_armed_at': None, 'sc_base_pnl': None}) is None
    st = eng._sc_state_for(1, {'sc_armed_at': datetime.utcnow(), 'sc_base_pnl': 0.4, 'sc_high_pct': None})
    assert st['L'] == 0.4 and st['high'] == 0.4
    TE._SC_STATE.clear()


def test_config_and_ui_wiring():
    """D11: config default · trading_config.json value · UI input · load + save handlers."""
    import json
    from config import SignalThresholds
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    th = json.load(open(os.path.join(root, "trading_config.json")))["thresholds"]
    d = SignalThresholds()
    keys = {"ml_second_chance_enabled": "config-sc-enabled", "ml_second_chance_adx_delta_max": "config-sc-adx-max",
            "ml_second_chance_floor_pct": "config-sc-floor", "ml_second_chance_trail_atr_mult": "config-sc-trail",
            "ml_second_chance_clock_min": "config-sc-clock"}
    html = open(os.path.join(root, "templates", "index.html")).read()
    for k, el in keys.items():
        assert hasattr(d, k) and k in th
        assert html.count(f'id="{el}"') == 1          # the input
        assert html.count(f"'{el}'") + html.count(f"('{el}')") >= 2   # load + save handlers
    assert d.ml_second_chance_enabled is False and th["ml_second_chance_enabled"] is True
    assert (th["ml_second_chance_floor_pct"], th["ml_second_chance_trail_atr_mult"], th["ml_second_chance_clock_min"],
            th["ml_second_chance_adx_delta_max"]) == (0.3, 1.0, 180.0, 0.0)


def test_order_columns_exist():
    from models import Order
    cols = {c.name for c in Order.__table__.columns}
    assert {"sc_armed_at", "sc_base_pnl", "sc_base_reason", "sc_high_pct"} <= cols


# ───────────────────────────────────────────── the revert-gate scout
def _scout():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import scout_ml_second_chance as M
    return M


def test_gate_rules():
    M = _scout()
    assert M.gate([])[0] == "OBSERVING"
    assert M.gate([-0.3] * 7)[0] == "OBSERVING"                       # early stop needs 8
    assert M.gate([-0.2] * 8)[0] == "REVERT"                          # Σ −1.6 < −1.5
    assert M.gate([-0.18] * 8)[0] == "OBSERVING"                      # Σ −1.44 ≥ −1.5
    assert M.gate([0.1] * 8 + [-0.2] * 7)[0] == "REVERT"              # Σ −0.6 ≤ 0 at 15
    assert M.gate([0.0] * 15)[0] == "REVERT"                          # ≤ 0 reverts
    assert M.gate([0.05] * 15 + [-9] * 5)[0] == "PASS"                # decided on the first 15
    assert M.gate([-0.2] * 8 + [5] * 7)[0] == "REVERT"                # the early stop fired first


def test_scout_reads_stamped_shadow(tmp_path):
    M = _scout()
    rows = [  # opened, pair, adx, status, pct, sc_armed_at, sc_base, close_reason, source
        ("2026-11-12T10:00:00", "AAAUSDT", -0.3, "CLOSED", 1.40, "2026-11-12T10:20:00", 0.55, "SC_TRAIL", "UNMATCHED"),
        ("2026-11-12T11:00:00", "BBBUSDT", -0.1, "CLOSED", 0.25, "2026-11-12T11:10:00", 0.55, "SC_FLOOR", "UNMATCHED"),
        ("2026-11-12T12:00:00", "CCCUSDT", 0.4, "CLOSED", 0.30, None, None, "RUNNER_TRAIL", "UNMATCHED"),
        ("2026-11-12T13:00:00", "DDDUSDT", -0.5, "CLOSED", 0.90, "2026-11-12T13:05:00", 0.40, "SC_TRAIL", "MAJORS_PROBE"),   # probe out
        ("2026-11-12T14:00:00", "EEEUSDT", -0.5, "CLOSED", 0.10, "2026-11-12T14:05:00", 0.40, "MANUAL", "UNMATCHED"),        # operator close out
        ("2026-11-10T14:00:00", "FFFUSDT", -0.5, "CLOSED", 2.00, "2026-11-10T14:05:00", 0.40, "SC_TRAIL", "UNMATCHED"),      # before the floor
    ]
    d = pd.DataFrame([dict(opened_at=o, closed_at=o, pair=p, direction="LONG", entry_strategy="MOMENTUM", status=st, pnl_percentage=pc,
                           close_reason=cr, cell_multiplier_source=src, entry_adx_delta=a, peak_pnl=1.0, sc_armed_at=sa, sc_base_pnl=sb,
                           sc_base_reason=("RUNNER_TRAIL" if sa else None), sc_high_pct=(1.9 if sa else None))
                      for o, p, a, st, pc, sa, sb, cr, src in rows])
    f = tmp_path / "scalpars_orders_paper_2026-11-12_20-00-00.csv"
    d.to_csv(f, index=False)
    out = "\n".join(M.run(paths=[str(f)], floor_ms=int(pd.Timestamp("2026-11-11T12:00:00").value // 10**6)))
    assert "3 momentum longs" in out and "2 exits changed" in out and "OBSERVING" in out
    assert "+1.10 → second chance +1.65 pts (Δ +0.55; 1 better / 1 worse" in out
    assert "DDDUSDT" not in out and "FFFUSDT" not in out
    assert "1 held trade(s) closed by the operator — NOT counted (Δ EEEUSDT -0.30)" in out   # listed apart, never counted


# ───────────────────────────────────────────── the replica pricer (blocked-trade scout) applies the same rule
def test_replica_apply_second_chance():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import scout_ml_matched_refusals as MM
    t = np.arange(0, 400) * 15_000 + 10_000_000
    net = np.r_[np.linspace(0, 1.0, 40), np.linspace(1.0, 0.55, 20), np.linspace(0.55, 1.9, 100), np.linspace(1.9, 1.0, 240)]
    RP = SimpleNamespace(HORIZON_MS=10**9, prep_path=lambda fx, hz: dict(t=t, net=net))
    fx = dict(o_ms=int(t[0]), atr=0.45)
    i0 = 59
    base = (i0, "RUNNER_TRAIL", float(net[i0]), int(t[i0]), "tick")
    k, why, pct, tex, src = MM.apply_second_chance(RP, fx, base, -0.2, P)
    assert why == "SC_TRAIL" and pct <= 1.9 - 0.45 + 1e-9 and pct > 1.4
    assert MM.apply_second_chance(RP, fx, base, 0.3, P) == base            # rising ADX
    assert MM.apply_second_chance(RP, fx, base, None, P) == base           # no stamp
    assert MM.apply_second_chance(RP, fx, base, -0.2, None) == base        # switch off
    stop = (i0, "STOP_LOSS", -0.7, int(t[i0]), "tick")
    assert MM.apply_second_chance(RP, fx, stop, -0.2, P) == stop           # only runner / ladder exits


# ───────────────────────────────────────────── review fixes (caveman + deep, 2026-11-11)
def test_review_guards(monkeypatch):
    TE, eng, persisted = _engine(monkeypatch, _th())
    t_open = datetime(2026, 11, 12, 10, 0, 0)
    try:
        async def run():
            # a stale monitor row of a trade no longer in the open cache never arms (deep #1)
            o = _order(opened_at=t_open)
            TE._open_orders_cache.pop(o.pair, None)
            assert await eng._sc_intercept(o, 1.006, "RUNNER_TRAIL") is False and TE._SC_STATE == {}
            TE._open_orders_cache[o.pair] = [{'id': o.id, '_closing_in_progress': True}]
            # a trigger below 0 % keeps today's exit
            assert await eng._sc_intercept(o, 0.995, "RUNNER_TRAIL") is False
            # a recovery-hold trade keeps today's exit
            assert await eng._sc_intercept(_order(opened_at=t_open, rh_triggered_at=t_open), 1.006, "RUNNER_TRAIL") is False
            # arms → flag released; a repeat intercept leaves a re-set flag alone (an SC close may be in flight)
            assert await eng._sc_intercept(o, 1.006, "RUNNER_TRAIL") is True
            assert TE._open_orders_cache[o.pair][0]['_closing_in_progress'] is False
            TE._open_orders_cache[o.pair][0]['_closing_in_progress'] = True
            assert await eng._sc_intercept(o, 1.004, "RUNNER_TRAIL L1") is True
            assert TE._open_orders_cache[o.pair][0]['_closing_in_progress'] is True
            # id reused by a NEW trade (paper reset): the old state is dropped, the new trade is not held
            o2 = _order(opened_at=t_open + timedelta(days=1), entry_adx_delta=0.5)
            assert await eng._sc_intercept(o2, 1.006, "RUNNER_TRAIL") is False
            assert o2.id not in TE._SC_STATE
        asyncio.run(run())
    finally:
        TE._open_orders_cache.pop("TESTUSDT", None)
        TE._SC_STATE.clear()
