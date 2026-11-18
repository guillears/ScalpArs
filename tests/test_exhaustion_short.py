"""🪫 EXHAUSTION_SHORT (DECISION_LOG 578 — operator-approved PAPER run, declared override) + ⛔ the momentum entry switches.

Pure rules (services/exhaustion.py) checked against the SPOT-SHARE DEEP study's own rows (tests/fixtures/exh_parity.npz, built by
scripts/exh_parity_fixture.py from reports/SPOT_SHARE_DEEP_data — the study's 5m series, spot volumes and stored features), the REAL engine
paths on SQLite with every exchange read stubbed (no network), exits through the realtime path, config / UI / both exports parity."""
import asyncio
import datetime as dt
import json
import math
import os
import random
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import exhaustion as X  # noqa: E402
from tests.test_frenzy_willy import _hold_db  # noqa: E402

FIX = np.load(os.path.join(ROOT, "tests", "fixtures", "exh_parity.npz"))
BAR, M15, HOUR = 300_000, 900_000, 3_600_000


def _th(**kw):
    d = dict(exhaustion_short_enabled=True, exhaustion_rsi_period=14, exhaustion_rsi_level=70.0, exhaustion_vol_decay_min=0.6366,
             exhaustion_spot_share_chg_max=-0.0066, exhaustion_tp_atr_mult=3.0, exhaustion_tp_cap_pct=5.0, exhaustion_sl_atr_mult=1.5,
             exhaustion_sl_cap_pct=5.0, exhaustion_max_hold_hours=24.0, exhaustion_max_hours_after_spike=72.0, exhaustion_min_vol_usd=20e6,
             exhaustion_invest_mult=0.5, exhaustion_lev_mult=0.25, resting_tp_exhaustion_enabled=True, broker_backstop_pct=2.5)
    d.update(kw)
    return NS(**d)


def _row(k):
    """fixture row k → (pair, T, on_ms, spike_ms, raw perp rows with qv at index 7, spot rows [open, qv], study meta, entry-bar open)."""
    m = FIX[f"r{k}_meta"]
    k5 = FIX[f"r{k}_k5"]
    rows = [[int(r[0]), r[1], r[2], r[3], r[4], r[5], int(r[0]) + BAR - 1, r[6]] for r in k5[:-1]]
    spot = [[int(a), b] for a, b in FIX[f"r{k}_spot"]]
    return str(FIX[f"r{k}_pair"]), int(m[0]), int(m[1]), int(m[2]), rows, spot, m, float(k5[-1][1])


# ── ① pure helpers equal the study's pandas code ─────────────────────────────────────────────────────────────────────────────────────

def test_wilder_rsi_atr_equal_the_study_pandas_code():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import frenzy_ta_build as TB
    rnd = random.Random(7)
    c = [100.0]
    for _ in range(400):
        c.append(c[-1] * (1 + rnd.gauss(0, 0.01)))
    h = [x * (1 + abs(rnd.gauss(0, 0.004))) for x in c]
    l = [x * (1 - abs(rnd.gauss(0, 0.004))) for x in c]
    r_ref = TB.rsi(np.array(c), 14)
    a_ref = TB.atr(np.array(h), np.array(l), np.array(c), 14)
    r, a = X.rsi(c, 14), X.atr(h, l, c, 14)
    for i in range(len(c)):
        if i < 14:
            assert r[i] is None and a[i] is None
        else:
            assert r[i] == pytest.approx(r_ref[i], abs=1e-9) and a[i] == pytest.approx(a_ref[i], rel=1e-9)
    assert X.rsi([1.0] * 30, 14)[20] == 100.0                                         # down = 0 → 100 (the study's rule)


def test_grid_fills_a_missing_bar_flat_and_aggregates_15m():
    t0 = 1_700_000_100_000 // M15 * M15
    rows = [[t0 + k * BAR, 1, 2, 0.5, 1 + k, 10, 0, 100 + k] for k in range(6) if k != 4]
    g = X.grid5(rows, t0 + 6 * BAR)
    t, o, h, l, c, q = g
    assert len(t) == 6 and c[4] == c[3] == 4 and h[4] == l[4] == 4 and q[4] == 0.0      # the study's flat fill, zero volume
    t15, o15, h15, l15, c15, q15 = X.agg15(t, o, h, l, c, q)
    assert t15 == [t0, t0 + M15] and c15 == [3, 6] and q15 == [303.0, 103 + 0 + 105]
    assert X.grid5(rows, t0 + 3 * BAR)[0] == [t0, t0 + BAR, t0 + 2 * BAR]                # only bars CLOSED by the decision time
    assert X.grid5([[t0 + 7, 1, 1, 1, 1, 1]], t0 + M15) is None and X.grid5([], 0) is None   # off grid / empty → fail-closed


# ── ② PARITY with the study's rows ────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", range(5))
def test_parity_with_the_study_rows(k):
    pair, T, on, sp, rows, spot, m, o_next = _row(k)
    f = X.features(rows, spot, T, on, True, _th())
    assert f['ok'] and sp == on + BAR
    assert f['rsi_prev'] == pytest.approx(m[7], abs=1e-6) and f['rsi'] == pytest.approx(m[8], abs=1e-6)   # the study's RSI (full history)
    assert f['rsi_prev'] > 70 >= f['rsi'] and f['cross']                                                   # every picked row is an RSI14X short
    assert f['atr'] / o_next * 100 == pytest.approx(m[3], rel=1e-5)                    # atr_pct (study: ÷ the entry bar's open; float32)
    assert f['atr_pct'] == pytest.approx(m[3], rel=0.02)                                # live: ÷ the decision close — the same price ± the gap
    assert f['vol_decay'] == pytest.approx(m[4], abs=1e-5) and f['spot_share_chg'] == pytest.approx(m[5], abs=1e-6)
    assert f['q5'] == pytest.approx(m[6], rel=1e-5) and f['vol24'] == pytest.approx(m[9], rel=1e-9)
    ok, code, why = X.decide(f, _th())
    assert ok and code is None, why                                                     # the study KEPT these rows (R4)
    # the engine's FRENZY window carries no quote volume: the RSI cross read there must agree
    sig = X.rsi_cross([r[:6] for r in rows], T, _th())
    assert sig['cross'] and sig['rsi'] == pytest.approx(m[8], abs=1e-6)


def test_decide_refusals_and_fail_closed():
    pair, T, on, sp, rows, spot, m, _ = _row(1)
    th = _th()
    base = X.features(rows, spot, T, on, True, th)
    assert X.decide(dict(base, vol_decay=0.5), th)[1] == X.B_VOL_DECAY
    assert X.decide(dict(base, spot_share_chg=0.0), th)[1] == X.B_SPOT_SHARE
    assert X.decide(dict(base, spot_share_chg=-0.0066), th)[1] == X.B_SPOT_SHARE          # strict <
    assert X.decide(dict(base, vol24=19e6), th)[1] == X.B_VOL24
    assert X.decide(dict(base, cross=False), th)[1] == X.B_NO_SIGNAL
    assert X.decide(X.features(rows, spot, T, on, False, th), th)[1] == X.B_NO_SPOT       # no spot market → unscored → no trade
    assert X.decide(X.features(rows, spot[-2:], T, on, True, th), th)[1] == X.B_NO_DATA    # spot window short → unscorable
    assert X.decide(X.features([r[:6] for r in rows], spot, T, on, True, th), th)[1] == X.B_NO_DATA   # no quote volume
    assert X.decide(X.features(rows, spot, T + M15, on, True, th), th)[1] == X.B_NO_DATA   # the window does not reach T
    assert X.decide(X.features(rows, spot, T, rows[0][0] - BAR, True, th), th)[1] == X.B_NO_DATA   # spike bar outside the window
    assert X.decide(None, th)[1] == X.B_NO_DATA and X.features(None, None, T, on, True, th)['ok'] is False
    assert X.in_window(sp, sp, th) and X.in_window(sp, sp + 72 * HOUR, th) and not X.in_window(sp, sp + 72 * HOUR + M15, th)
    assert not X.in_window(sp, sp - BAR, th)


# ── ③ exits ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_exit_levels_caps_and_row_levels():
    th = _th()
    assert X.exit_levels(1.0, th) == (3.0, 1.5)
    assert X.exit_levels(2.5, th) == (5.0, 3.75)                                         # TP capped at 5 %
    assert X.exit_levels(4.0, th) == (5.0, 5.0)                                          # both capped
    assert X.exit_levels(None, th) == (None, None) and X.exit_levels(-1, th) == (None, None)
    assert X.row_levels(dict(entry_exh_tp_pct=3.0, entry_exh_sl_pct=1.5), th) == (3.0, 1.5)
    assert X.row_levels(dict(), th) == (5.0, 5.0)                                        # missing stamps → the caps (never wider)
    assert X.row_levels(dict(entry_exh_tp_pct=9.0, entry_exh_sl_pct=-1), th) == (5.0, 5.0)


@pytest.mark.parametrize("price,held,rtp,want", [(0.969, 10, False, "FRENZY_TP"), (0.969, 10, True, None), (1.0151, 10, False, "STOP_LOSS"),
                                                  (1.0151, 10, True, "STOP_LOSS"), (1.01, 24 * 60, False, "MAX_HOLD_TIME"), (1.01, 60, False, None)])
def test_exit_rule(price, held, rtp, want):
    row = dict(entry_exh_tp_pct=3.0, entry_exh_sl_pct=1.5)
    c, why, _ = X.exhaustion_exit_for(1.0, price, held, _th(), row=row, resting_tp=rtp)
    assert (why if c else None) == want


def test_resting_tp_row_target_is_the_row_price_level():
    from services.resting_tp import rtp_row_target, RTP_SLEEVES, rtp_target
    th = _th(maker_fee=0.00018)
    row = NS(entry_strategy="EXHAUSTION_SHORT", direction="SHORT", entry_price=1.0, quantity=100.0, entry_fee=0.045, entry_exh_tp_pct=3.0,
             manual_exit_mode=None, exit_override_at=None)
    net, why = rtp_row_target(th, row)
    assert RTP_SLEEVES["EXHAUSTION_SHORT"] == "resting_tp_exhaustion_enabled" and why == "FRENZY_TP"
    assert net == pytest.approx(3.0 - 0.045 - 0.018 * 0.97, abs=1e-6)                       # 100·m − entry fee % − maker on the exit notional
    assert rtp_row_target(_th(resting_tp_exhaustion_enabled=False), row) is None
    assert rtp_target(th, "EXHAUSTION_SHORT") == (5.0, "FRENZY_TP")                          # nominal = the cap; the row's own level is placed


# ── ④ slippage model + revert gate ────────────────────────────────────────────────────────────────────────────────────────────────────

def test_slippage_model_and_gate():
    assert X.model_slip_pct(1000, 1.1e6) == 0.0                                              # ≈ 0 at $1k on these coins (floored)
    assert X.model_slip_pct(100_000, 1e5) == pytest.approx(-0.041640290342889685 + 0.27612727721342023, rel=1e-9)
    assert X.gate_net_pct(1.0, "FRENZY_TP", 0.1) == pytest.approx(0.9) and X.gate_net_pct(1.0, "STOP_LOSS", 0.1) == pytest.approx(0.8)
    t0 = dt.datetime(2026, 11, 18)
    win = [(t0 + dt.timedelta(hours=i), 1.0, "FRENZY_TP", 0.0) for i in range(20)]
    assert X.exhaustion_gate(win[:5])[0] == "COLLECTING"
    assert X.exhaustion_gate(win)[0] == "KEEP"
    lose = [(t0 + dt.timedelta(hours=i), -0.1, "STOP_LOSS", 0.0) for i in range(20)]
    assert X.exhaustion_gate(lose)[0] == "OFF"
    slip = [(t0 + dt.timedelta(hours=i), 1.0, "FRENZY_TP", 0.3) for i in range(20)]
    v, txt = X.exhaustion_gate(slip)
    assert v == "OFF" and "slippage" in txt                                                  # mean modelled slippage > 0.25 %
    edge = [(t0 + dt.timedelta(hours=i), 0.1, "STOP_LOSS", 0.05) for i in range(20)]         # 0.1 − 0.05 − 0.05 = 0 → ≤ 0 → OFF
    assert X.exhaustion_gate(edge)[0] == "OFF"
    late_bad = win + [(t0 + dt.timedelta(hours=30 + i), -9.0, "STOP_LOSS", 0.0) for i in range(5)]
    assert X.exhaustion_gate(late_bad)[0] == "KEEP"                                         # FROZEN: the first 20 by open time


def test_scout_reads_the_gate(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_exhaustion_short as SC
    rows = [dict(opened_at=f"2026-11-18 0{i}:00:00", closed_at="", pair=f"P{i}USDT", direction="SHORT", status="CLOSED",
                 entry_strategy="EXHAUSTION_SHORT", pnl=1, pnl_percentage=0.5, close_reason="FRENZY_TP", notional_value=500,
                 entry_exh_model_slip_pct=0.0, entry_exh_q5_usd=1e6) for i in range(3)]
    rows.append(dict(rows[0], pair="MANUSDT", close_reason="MANUAL"))                         # operator-closed → out
    f = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    fills = SC.load_fills([str(f)])
    assert len(fills) == 3
    out = SC.report(fills, pd.DataFrame())
    assert "COLLECTING" in out and "3/20" in out
    import observe_registry as OR
    assert "scout_exhaustion_short.py" in OR.classified()


# ── ⑤ the REAL engine path (tracking · signal · reads · rule · order · journal) ─────────────────────────────────────────────────────

def _engine(TE):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True; e.is_running = True
    e.blocks = []
    e._record_filter_block = lambda n, d, had_room=True: e.blocks.append(n)
    return e


def _patch_th(monkeypatch, TE, **kw):
    for k, v in dict(vars(_th()), **kw).items():
        monkeypatch.setattr(TE.config.trading_config.thresholds, k, v, raising=False)


def _run(monkeypatch, k=1, *, spot_syms=None, held=False, open_ok=True, perp_ok=True, th_kw=None, fs_bars=0, T_shift=0):
    """drives the REAL _exhaustion_eval → _exhaustion_judge_queued on fixture row k (the FRENZY window = the study's closes up to T);
    exchange reads + open_position stubbed. → (engine, opens, journal rows, episode rows, read calls)."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    pair, T, on, sp, rows, spot, m, _ = _row(k)
    T += T_shift
    _patch_th(monkeypatch, TE, **(th_kw or {}))
    calls = []

    async def kl(pid, interval, limit, end_ms=None, spot=False):
        calls.append((pid, interval, limit, end_ms, spot))
        if spot:
            return [[r[0], 0, 0, 0, 0, 0, r[0] + M15 - 1, r[1]] for r in _row(k)[5]]
        return [r for r in rows if r[0] <= end_ms][-limit:] if perp_ok else None

    async def syms():
        calls.append(("info",))
        return {pair} if spot_syms is None else spot_syms

    async def fr(sym, max_age_s=None):
        return 0.0001
    monkeypatch.setattr(TE.binance_service, "fetch_exh_klines", kl)
    monkeypatch.setattr(TE.binance_service, "fetch_exh_spot_symbols", syms)
    monkeypatch.setattr(TE.binance_service, "fetch_funding_rate", fr)
    monkeypatch.setattr(TE.binance_service, "exh_used_weight", 0, raising=False)
    opens = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        if held:
            async with SL() as s:
                s.add(models.Order(pair=pair, direction="LONG", status="OPEN", is_paper=True, entry_strategy="FRENZY_LONG", confidence="STRONG_BUY",
                                   entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0))
                await s.commit()
        e = _engine(TE)

        async def fake_open(db=None, **kw):
            opens.append(kw)
            return NS(id=88) if open_ok else None
        e.open_position = fake_open
        closed = [r[:6] for r in rows if r[0] + BAR <= T]
        ep = dict(spike_ts=sp, verified=True, hours=(T - sp) / HOUR)
        async with SL() as db:
            # first sighting one bar earlier (tracking), then the 15m close itself
            await e._exhaustion_eval(db, pair, ep, closed[:-1], {"symbol": pair.replace("USDT", "/USDT:USDT")}, T - BAR)
            await e._exhaustion_eval(db, pair, None, closed, {"symbol": pair.replace("USDT", "/USDT:USDT")}, T)
            await e._exhaustion_judge_queued(db)
            await e._exhaustion_eval(db, pair, None, closed, {}, T)          # a retry pass of the same bar never judges twice
            await e._exhaustion_judge_queued(db)
        async with SL() as s:
            sig = (await s.execute(select(models.ExhaustionSignal))).scalars().all()
            eps = (await s.execute(select(models.ExhaustionEpisode))).scalars().all()
        await eng.dispose()
        return e, sig, eps
    e, sig, eps = asyncio.run(go())
    return e, opens, sig, eps, calls


def test_engine_opens_on_a_study_row(monkeypatch):
    e, opens, sig, eps, calls = _run(monkeypatch, k=1)
    pair, T, on, sp, rows, spot, m, _ = _row(1)
    assert len(eps) == 1 and eps[0].status == "TRACKING" and eps[0].pair == pair
    assert len(opens) == 1 and len(sig) == 1 and sig[0].status == "OPENED" and sig[0].order_id == 88
    o = opens[0]
    assert o['exhaustion_short'] is True and o['direction'] == "SHORT" and o['exhaustion_bar_close_ms'] == T
    assert o['entry_exh_vol_decay'] == pytest.approx(m[4], abs=1e-5) and o['entry_exh_spot_share_chg'] == pytest.approx(m[5], abs=1e-5)
    tp, sl = X.exit_levels(o['entry_exh_atr_pct'], _th())
    assert (o['entry_exh_tp_pct'], o['entry_exh_sl_pct']) == (tp, sl) and o['entry_exh_decision_price'] == o['current_price'] == rows[-1][4]
    assert o['entry_frenzy_spike_at'] == dt.datetime.utcfromtimestamp(sp / 1000) and o['entry_exh_q5_usd'] == pytest.approx(m[6], rel=1e-5)
    perp = [c for c in calls if len(c) == 5 and not c[4]]; spt = [c for c in calls if len(c) == 5 and c[4]]
    assert perp == [(pair, '5m', 1500, T - 1, False)] and spt == [(pair, '15m', 104, T - 1, True)]   # one read each, only on the signal
    assert e.blocks == []


def test_engine_refusals_are_journaled_and_counted(monkeypatch):
    e, opens, sig, _, calls = _run(monkeypatch, k=2, th_kw=dict(exhaustion_vol_decay_min=0.9))   # ROSE: vol_decay 0.80
    assert opens == [] and sig[0].status == "BLOCKED" and sig[0].block == "EXH_VOL_DECAY" and "EXH_VOL_DECAY" in e.blocks
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, _, calls = _run(mp, k=1, spot_syms={"BTCUSDT"})
    finally:
        mp.undo()
    assert opens == [] and sig[0].block == "EXH_NO_SPOT" and not [c for c in calls if len(c) == 5 and c[4]]   # no spot read for a no-spot coin
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, _, calls = _run(mp, k=1, held=True)
    finally:
        mp.undo()
    assert opens == [] and sig[0].block == "EXH_PAIR_HELD" and calls == []                  # one position per coin, checked before any read
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, _, calls = _run(mp, k=1, perp_ok=False)
    finally:
        mp.undo()
    assert opens == [] and sig[0].block == "EXH_NO_DATA" and "EXH_NO_DATA" in e.blocks       # fail-closed
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, _, calls = _run(mp, k=1, open_ok=False)
    finally:
        mp.undo()
    assert len(opens) == 1 and sig[0].status == "REFUSED" and "EXH_OPEN_REFUSED" in e.blocks


def test_engine_no_signal_off_the_15m_grid_or_outside_the_window(monkeypatch):
    e, opens, sig, eps, calls = _run(monkeypatch, k=1, th_kw=dict(exhaustion_max_hours_after_spike=0.5))
    assert opens == [] and sig == [] and calls == []                                         # outside the window: never tracked / read
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, eps, calls = _run(mp, k=1, T_shift=BAR)                              # a 5m close that is not a 15m close
    finally:
        mp.undo()
    assert opens == [] and sig == [] and calls == []
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, eps, calls = _run(mp, k=1, th_kw=dict(exhaustion_short_enabled=False))
    finally:
        mp.undo()
    assert opens == [] and sig == [] and eps == []


def test_frenzy_pass_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    p = eng.index("    async def _update_frenzy_pass(self, db):"); body = eng[p:eng.index("\n    def _frenzy_wide_off_note(", p)]
    assert "self.__dict__.setdefault('_exh_eval_q', []).append((pair, ep, closed, by[pair], bar_open))" in body and "self._exhaustion_eval(" not in body
    assert body.index("await self._flatbase_sweep(") < body.index("await self._exhaustion_run_queue(db)") < body.index("await self._exhaustion_judge_queued(db)") \
        < body.index("await self._exhaustion_sweep(bar_open)")
    assert "_follow += await self._exhaustion_follow_seed(db)" in body and "EXH_EXTRA_READS_MAX" in body
    import services.trading_engine as TE
    assert TE._ema13_cross_exit_applies("EXHAUSTION_SHORT") is False
    assert TE.TradingEngine.stamp_class("entry_exh_tp_pct") == "sleeve" and TE.TradingEngine.stamp_class("entry_exh_model_slip_pct") == "sleeve"
    assert TE.pair_reason_stampable("EXH_VOL_DECAY") is False
    th = _th()
    assert TE.bot_own_stop_pct(th, "EXHAUSTION_SHORT") == 5.0 and TE.order_backstop_pct(th, "EXHAUSTION_SHORT", leverage=5) == 5.5


# ── ⑥ open_position: size, stamps, stop, backstop, resting TP, lateness (REAL open path, paper) ──────────────────────────────────────

def _e2e(monkeypatch, n_open=0, th_extra=None, **kw):
    from tests.test_flatbase_short import _e2e_open
    return _e2e_open(monkeypatch, n_open=n_open, th_kw=dict(vars(_th()), **(th_extra or {})), **kw)


def test_open_position_exhaustion_size_stamps_and_exit_record(monkeypatch):
    import services.trading_engine as TE
    now_ms = int(__import__("time").time() * 1000)
    r, blocks, n_slot, n_bot, ref, ref_m = _e2e(monkeypatch, n_open=2, exhaustion_short=True, exhaustion_bar_close_ms=now_ms,
                                                 entry_frenzy_spike_at=dt.datetime(2026, 11, 17, 8, 0), entry_exh_rsi_prev=72.0, entry_exh_rsi=68.0,
                                                 entry_exh_vol_decay=0.8, entry_exh_spot_share_chg=-0.02, entry_exh_atr_pct=1.2,
                                                 entry_exh_tp_pct=3.6, entry_exh_sl_pct=1.8, entry_exh_decision_price=1.0, entry_exh_q5_usd=2e6)
    assert r is not None and r.entry_strategy == "EXHAUSTION_SHORT" and r.direction == "SHORT" and r.cell_multiplier_source == "EXHAUSTION_SHORT"
    assert n_slot == 3 and n_bot == 3                                                       # it holds a bot slot (only the shared book binds)
    assert r.leverage == 5                                                                  # 20 × 0.25
    assert r.entry_exh_tp_pct == 3.6 and r.entry_exh_sl_pct == 1.8 and r.entry_exh_vol_decay == 0.8 and r.entry_exh_rsi_prev == 72.0
    assert r.entry_frenzy_spike_at == dt.datetime(2026, 11, 17, 8, 0) and r.entry_exh_q5_usd == 2e6
    assert r.entry_exh_model_slip_pct == pytest.approx(X.model_slip_pct(r.notional_value, 2e6))
    assert r.backstop_pct == 5.5 and r.resting_tp_price == pytest.approx(r.entry_price * (1 - 0.036), rel=2e-3)   # the row's own TP rests
    c = TE._open_orders_cache.get("FBSUSDT") or []
    assert c and c[-1]['entry_exh_tp_pct'] == 3.6 and c[-1]['stop_loss'] == -1.8
    assert "MOMENTUM_SHORT_OFF" not in blocks


def test_open_position_exhaustion_book_full_and_late(monkeypatch):
    now_ms = int(__import__("time").time() * 1000)
    r, blocks, *_ = _e2e(monkeypatch, n_open=4, exhaustion_short=True, exhaustion_bar_close_ms=now_ms)
    assert r is None and "EXH_BOOK_FULL" in blocks                                          # a full shared book refuses it — no own cap
    mp = pytest.MonkeyPatch()
    try:
        r, blocks, *_ = _e2e(mp, exhaustion_short=True, exhaustion_bar_close_ms=now_ms - 301_000)
    finally:
        mp.undo()
    assert r is None and "EXH_LATE" in blocks


# ── ⑦ the momentum entry switches (REAL open path) ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("direction,key,code", [("LONG", "momentum_long_entries_enabled", "MOMENTUM_LONG_OFF"),
                                                ("SHORT", "momentum_short_entries_enabled", "MOMENTUM_SHORT_OFF")])
def test_momentum_switch_refuses_only_momentum(monkeypatch, direction, key, code):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config.trading_config.thresholds, key, False, raising=False)   # overrides conftest's _momentum_entries_on
    assert getattr(TE.config.trading_config.thresholds, key) is False                     # (the autouse fixture set it True first — not vacuous)

    async def go(**kw):
        eng, SL = await _hold_db(monkeypatch)()
        e = TE.TradingEngine(); e.is_running = True; e.is_paper_mode = True
        blocks = []
        e._record_filter_block = lambda n, d, had_room=True: blocks.append(n)

        async def mb():   # the next check after the switch: a manual block ends every other path there (no reads, no network)
            return {"pair": "MANUSDT", "order_id": 1}
        e.manual_block_active = mb
        async with SL() as db:
            r = await e.open_position(db=db, pair="MOMUSDT", direction=direction, confidence="STRONG_BUY", current_price=1.0, **kw)
        await eng.dispose()
        return r, blocks
    r, blocks = asyncio.run(go())
    assert r is None and blocks[:1] == [code]                                                # the core momentum entry: refused first
    mp = pytest.MonkeyPatch()
    mp.setattr(TE.config.trading_config.thresholds, key, True, raising=False)
    try:
        r, blocks = asyncio.run(go())
    finally:
        mp.undo()
    assert code not in blocks and blocks[:1] == ["MANUAL_BLOCK"]                             # switch ON → the open path goes on
    r, blocks = asyncio.run(go(flip_source="FAN_RATIO_GATE"))
    assert code not in blocks                                                                # a flip is never refused by this switch
    if direction == "SHORT":
        r, blocks = asyncio.run(go(bearrun_short=True))
        assert code not in blocks                                                            # Bear-Run has its own switch
    r, blocks = asyncio.run(go(surge_dir=direction))
    assert code not in blocks


def test_momentum_switch_defaults_and_json():
    import config as C
    mf = C.SignalThresholds.model_fields
    assert mf["momentum_long_entries_enabled"].default is True and mf["momentum_short_entries_enabled"].default is True
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["momentum_long_entries_enabled"] is False and cfg["momentum_short_entries_enabled"] is False   # DECISION_LOG 578 paper run


# ── ⑧ config / UI / exports parity (D11 / D12) + the operator's sleeve set (DECISION_LOG 578) ──────────────────────────────────────

FIELDS = {"exhaustion_short_enabled": (False, True), "exhaustion_rsi_period": (14, 14), "exhaustion_rsi_level": (70.0, 70.0),
          "exhaustion_vol_decay_min": (0.6366, 0.6366), "exhaustion_spot_share_chg_max": (-0.0066, -0.0066), "exhaustion_tp_atr_mult": (3.0, 3.0),
          "exhaustion_tp_cap_pct": (5.0, 5.0), "exhaustion_sl_atr_mult": (1.5, 1.5), "exhaustion_sl_cap_pct": (5.0, 5.0),
          "exhaustion_max_hold_hours": (24.0, 24.0), "exhaustion_max_hours_after_spike": (72.0, 72.0), "exhaustion_min_vol_usd": (20e6, 20e6),
          "exhaustion_invest_mult": (0.5, 0.5), "exhaustion_lev_mult": (0.25, 0.25), "resting_tp_exhaustion_enabled": (True, True)}


def test_config_defaults_json_ui_and_exports():
    import config as C
    mf = C.SignalThresholds.model_fields
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, (d, j) in FIELDS.items():
        assert mf[k].default == d and cfg[k] == j, k
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 578" in src and "DECLARED OVERRIDE" in src
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    ids = ["config-exh-enabled", "config-rtp-exhaustion-enabled", "config-exh-rsi-period", "config-exh-rsi-level", "config-exh-vol-decay-min",
           "config-exh-ssc-max", "config-exh-min-vol", "config-exh-max-hours", "config-exh-tp-atr", "config-exh-tp-cap", "config-exh-sl-atr",
           "config-exh-sl-cap", "config-exh-max-hold", "config-exh-invest-mult", "config-exh-lev-mult", "config-momentum-long-entries",
           "config-momentum-short-entries"]
    for i in ids:
        assert html.count(f'id="{i}"') == 1, i
    for k in FIELDS:
        if k not in ("exhaustion_short_enabled", "resting_tp_exhaustion_enabled"):
            assert f"'exhaustion_" in html and f", '{k}', " in html, k                        # EXHAUSTION_NUM_FIELDS = load AND save
    assert "for (const [_id, _key, _def] of EXHAUSTION_NUM_FIELDS)" in html and "...Object.fromEntries(EXHAUSTION_NUM_FIELDS.map(" in html
    for k, i in (("exhaustion_short_enabled", "config-exh-enabled"), ("resting_tp_exhaustion_enabled", "config-rtp-exhaustion-enabled"),
                 ("momentum_long_entries_enabled", "config-momentum-long-entries"), ("momentum_short_entries_enabled", "config-momentum-short-entries")):
        assert f"config.thresholds.{k}" in html and f"{k}: document.getElementById('{i}')" in html, k
    a = html.index("<!-- ===== FLATBASE_SHORT BOX END ===== -->"); b = html.index("<!-- ===== 🪫 EXHAUSTION_SHORT BOX")
    assert 0 < b - a < 100
    assert '<option value="EXHAUSTION_SHORT">' in html and 'data-ss="exh"' in html and "S.exh = {" in html
    assert html.count("_stratRaw === 'EXHAUSTION_SHORT' ? 'EXH-S'") == 2
    # both text exports: the shared config-line builder + the FRENZY monitor line + the FRENZY Fills rows
    assert html.count("..._buildConfigLines(cfg, changelog, hr, hr2, status)") == 2 and "🪫 EXHAUSTION_SHORT (DECISION_LOG 578" in html
    assert "⛔ Momentum entries (DECISION_LOG 578)" in html and "+ exhaustionMonitorStr(m)" in html
    assert html.count("...frenzyReportLines(perf, hr2)") == 2
    assert html.count("FRENZY_WILLY_HT · FLATBASE_SHORT · EXHAUSTION_SHORT)") == 2
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '("EXHAUSTION_SHORT", "🪫 EXHAUSTION_SHORT (' in main and 'if _fz_es == "EXHAUSTION_SHORT":' in main
    assert "_exh_cfg_errors(new_config.thresholds, _sb_lev)" in main and "'EXHAUSTION_SHORT': ('exhaustion_short_enabled',)" in main
    assert 'e="EXHAUSTION"' in main and "_exh_cols" in main and "await _exhaustion_reset_state(db, is_paper)" in main


def test_operator_sleeve_set_dl578():
    """DECISION_LOG 578 (operator 2026-11-18): paper with FLATBASE + EXHAUSTION + FRENZY_LONG / WILLY / WILLY_HT; every other entry sleeve off;
    WILLY + HT sized like FRENZY_LONG (0.2 / 0.05)."""
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
    th = cfg["thresholds"]
    on = ("flatbase_short_enabled", "exhaustion_short_enabled", "frenzy_long_enabled", "frenzy_willy_enabled", "frenzy_willy_ht_enabled")
    off = ("momentum_long_entries_enabled", "momentum_short_entries_enabled", "frenzy_lite_enabled", "frenzy_lite_retry_enabled", "frenzy_wide_enabled",
           "surge_long_enabled", "surge_short_enabled", "spike_fade_enabled", "spike_chase_probe_enabled", "spike_bounce_enabled",
           "flip_entry_enabled", "bullrun_sleeve_enabled", "bearrun_sleeve_enabled", "bull_long_enabled", "bounce_long_enabled")
    for k in on:
        assert th[k] is True, k
    for k in off:
        assert th[k] is False, k
    assert cfg["paper_trading"] is True
    assert (th["frenzy_willy_invest_mult"], th["frenzy_willy_lev_mult"]) == (th["frenzy_long_invest_mult"], th["frenzy_long_lev_mult"]) == (0.2, 0.05)
    assert th["frenzy_willy_ht_size_frac"] == 1.0                                             # HT = WILLY's fields × this → the same size


def test_config_errors():
    import config
    th = config.trading_config.thresholds
    assert X.exhaustion_config_errors(th, 20.0) == []
    bad = NS(**dict(th.model_dump(), exhaustion_tp_cap_pct=0, exhaustion_max_hours_after_spike=150, exhaustion_short_enabled="yes",
                    momentum_long_entries_enabled="no", exhaustion_rsi_period=14.5))
    errs = " | ".join(X.exhaustion_config_errors(bad, 20.0))
    for k in ("exhaustion_tp_cap_pct", "exhaustion_max_hours_after_spike", "exhaustion_short_enabled", "momentum_long_entries_enabled", "exhaustion_rsi_period"):
        assert k in errs, k
    wide = NS(**dict(th.model_dump(), exhaustion_lev_mult=1.0, exhaustion_sl_cap_pct=10.0))      # 20×: the backstop cannot sit beyond +10 %
    assert "too wide" in " ".join(X.exhaustion_config_errors(wide, 20.0))
    p = X.exhaustion_params(NS(exhaustion_vol_decay_min=7, exhaustion_lev_mult=-1))
    assert p['vol_decay_min'] == 0.6366 and p['lev'] == 0.25                                   # a bad value never widens the rule
    import main
    from fastapi import HTTPException
    saved = []
    mp = pytest.MonkeyPatch()
    mp.setattr(main, "save_trading_config", lambda c: saved.append(c) or True)
    try:
        with pytest.raises(HTTPException) as ei:
            asyncio.run(main.update_config(main.ConfigUpdate(thresholds={"exhaustion_sl_cap_pct": -1})))
    finally:
        mp.undo()
    assert ei.value.status_code == 400 and "exhaustion_sl_cap_pct" in ei.value.detail and saved == []


def test_sleeve_rows_and_labels():
    import main as M
    o = NS(entry_strategy="EXHAUSTION_SHORT", direction="SHORT", pnl_percentage=1.5, pnl=3.0, opened_at=dt.datetime(2026, 11, 18, 8),
           closed_at=dt.datetime(2026, 11, 18, 9), investment=60.0, notional_value=300.0, leverage=5, peak_pnl=2.0, import_source=None)
    rows = M._compute_sleeve_performance([o], start_balance=1000.0)
    assert any(r.get('sleeve') == 'Exhaustion-Short' for r in rows)
    assert M._compute_strategy_performance([o])[0]["strategy"].startswith("EXHAUSTION_SHORT")
    assert M._SLEEVE_LABELS[("EXHAUSTION_SHORT", "SHORT")] == "EXHAUSTION Short"
    assert M._sleeve_enabled(NS(exhaustion_short_enabled=False), "EXHAUSTION_SHORT") is False
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import screen_pool as SP
    assert SP.sleeve(dict(status="CLOSED", pair="X", entry_strategy="EXHAUSTION_SHORT", direction="SHORT")) is None


# ── ⑨ the realtime exit path ──────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("price,held_min,want", [(0.9639, 10, "FRENZY_TP"), (1.0181, 10, "STOP_LOSS"), (1.01, 24 * 60 + 1, "MAX_HOLD_TIME"),
                                                  (1.01, 10, None)])
def test_realtime_exit_path(monkeypatch, price, held_min, want):
    import models
    import services.trading_engine as TE
    _patch_th(monkeypatch, TE, resting_tp_exhaustion_enabled=False)
    closed = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        opened = dt.datetime.utcnow() - dt.timedelta(minutes=held_min)
        async with SL() as s:
            o = models.Order(pair="EXHUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy="EXHAUSTION_SHORT", confidence="STRONG_BUY",
                             entry_price=1.0, quantity=100.0, investment=20.0, leverage=5, notional_value=100.0, entry_fee=0.045, opened_at=opened,
                             entry_exh_tp_pct=3.6, entry_exh_sl_pct=1.8)
            s.add(o); await s.commit(); oid = o.id
        e = _engine(TE)

        async def cp(db, order, px, reason):
            closed.append(reason); order.status = "CLOSED"; return order
        e.close_position = cp
        TE._open_orders_cache["EXHUSDT"] = [dict(id=oid, direction="SHORT", entry_price=1.0, quantity=100.0, entry_fee=0.045, stop_loss=-1.8,
                                                 entry_strategy="EXHAUSTION_SHORT", opened_at=opened, peak_pnl=0.0, trough_pnl=0.0, confidence="STRONG_BUY",
                                                 entry_exh_tp_pct=3.6, entry_exh_sl_pct=1.8)]
        await e.check_realtime_stop_loss("EXHUSDT", price)
        await eng.dispose()
    asyncio.run(go())
    assert closed == ([want] if want else [])


# ── ⑩ review additions (DECISION_LOG 578 deep review I5) ──────────────────────────────────────────────────────────────────────────────

def test_exhaustion_opens_with_both_momentum_switches_off(monkeypatch):
    now_ms = int(__import__("time").time() * 1000)
    r, blocks, *_ = _e2e(monkeypatch, th_extra=dict(momentum_long_entries_enabled=False, momentum_short_entries_enabled=False),
                         exhaustion_short=True, exhaustion_bar_close_ms=now_ms, entry_exh_tp_pct=3.0, entry_exh_sl_pct=1.5, entry_exh_q5_usd=1e6)
    import services.trading_engine as TE
    assert TE.config.trading_config.thresholds.momentum_short_entries_enabled is False
    assert r is not None and r.entry_strategy == "EXHAUSTION_SHORT" and "MOMENTUM_SHORT_OFF" not in blocks


@pytest.mark.parametrize("price,held_min,want", [(0.969, 10, "FRENZY_TP"), (1.016, 10, "STOP_LOSS"), (1.01, 24 * 60 + 2, "MAX_HOLD_TIME"),
                                                  (1.01, 10, None)])
def test_candle_monitor_exit_path(monkeypatch, price, held_min, want):
    """the REAL update_open_positions (the 1 s monitor / candle path) on an EXHAUSTION row: the row's own TP 3 / SL 1.5 / 24 h."""
    import models
    import services.trading_engine as TE
    _patch_th(monkeypatch, TE, resting_tp_exhaustion_enabled=False)
    closed = []

    async def px(pair):
        return price
    monkeypatch.setattr(TE.binance_service, "get_current_price", px)
    monkeypatch.setattr(TE.websocket_tracker, "pair_silence_seconds", lambda p: 1.0)
    monkeypatch.setattr(TE.websocket_tracker, "get_tracker", lambda p: NS(last_price=price))
    monkeypatch.setattr(TE.websocket_tracker, "get_high_low", lambda p: (None, None))
    monkeypatch.setattr(TE.websocket_tracker, "update_price", lambda *a, **k: None)

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        opened = dt.datetime.utcnow() - dt.timedelta(minutes=held_min)
        async with SL() as s:
            s.add(models.Order(pair="EXCUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy="EXHAUSTION_SHORT", confidence="STRONG_BUY",
                               entry_price=1.0, quantity=100.0, investment=20.0, leverage=5, notional_value=100.0, entry_fee=0.045, opened_at=opened,
                               entry_exh_tp_pct=3.0, entry_exh_sl_pct=1.5, peak_pnl=0.0, trough_pnl=0.0, current_tp_level=1))
            await s.commit()
        e = _engine(TE)

        async def cp(db, order, px_, reason):
            closed.append(reason); order.status = "CLOSED"; return order
        e.close_position = cp
        async with SL() as db:
            await e.update_open_positions(db)
        await eng.dispose()
    asyncio.run(go())
    assert [c.split(' ')[0] for c in closed] == ([want] if want else [])


def test_cache_rebuild_reloads_the_row_exit_levels(monkeypatch):
    import models
    import services.trading_engine as TE

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        async with SL() as s:
            s.add(models.Order(pair="EXRUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy="EXHAUSTION_SHORT", confidence="STRONG_BUY",
                               entry_price=1.0, quantity=100.0, investment=20.0, leverage=5, notional_value=100.0, entry_fee=0.045,
                               opened_at=dt.datetime.utcnow(), entry_exh_tp_pct=3.3, entry_exh_sl_pct=1.65))
            await s.commit()
        e = TE.TradingEngine(); e.is_paper_mode = True
        monkeypatch.setattr(TE, "_open_orders_cache", {})
        async with SL() as db:
            await e.update_orders_cache(db)
        c = TE._open_orders_cache["EXRUSDT"][0]
        await eng.dispose()
        return c
    c = asyncio.run(go())
    assert c['entry_exh_tp_pct'] == 3.3 and c['entry_exh_sl_pct'] == 1.65 and c['stop_loss'] == -1.65
    assert X.exhaustion_exit_for(1.0, 0.967, 5, _th(), row=c)[1] == "FRENZY_TP" and X.exhaustion_exit_for(1.0, 1.0166, 5, _th(), row=c)[1] == "STOP_LOSS"


def test_ready_reloads_tracking_rows_in_a_fresh_engine(monkeypatch):
    import models
    import services.trading_engine as TE
    _patch_th(monkeypatch, TE)
    sp = 1_763_000_100_000 // BAR * BAR

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        async with SL() as s:
            s.add(models.ExhaustionEpisode(pair="TRKUSDT", spike_at=dt.datetime.utcfromtimestamp(sp / 1000), is_paper=True, status="TRACKING",
                                           first_seen_at=dt.datetime.utcfromtimestamp((sp + BAR) / 1000)))
            s.add(models.ExhaustionEpisode(pair="OLDUSDT", spike_at=dt.datetime.utcfromtimestamp(sp / 1000), is_paper=True, status="ENDED",
                                           first_seen_at=dt.datetime.utcfromtimestamp(sp / 1000)))
            s.add(models.ExhaustionEpisode(pair="LIVUSDT", spike_at=dt.datetime.utcfromtimestamp(sp / 1000), is_paper=False, status="TRACKING",
                                           first_seen_at=dt.datetime.utcfromtimestamp(sp / 1000)))
            await s.commit()
        e = _engine(TE)
        async with SL() as db:
            ok = await e._exhaustion_ready(db)
            seed = await e._exhaustion_follow_seed(db)
        await eng.dispose()
        return ok, dict(e._exh_state()), seed, e._exhaustion_pairs()
    ok, st, seed, pairs = asyncio.run(go())
    assert ok and set(st) == {("TRKUSDT", sp)} and st[("TRKUSDT", sp)]['first_seen_ms'] == sp + BAR
    assert seed == ["TRKUSDT"] and pairs == {"TRKUSDT"}                                     # ENDED and the other mode stay out


def test_weight_stop_waits_once_then_fails_closed(monkeypatch):
    import services.trading_engine as TE
    e = _engine(TE)
    slept = []

    async def fake_sleep(s_):
        slept.append(s_)
    monkeypatch.setattr(TE.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(TE.binance_service, "exh_used_weight", 2000, raising=False)
    now = TE._leash_time.time()
    e._exh_last_read_t = now - 30
    T = int(now * 1000) - 10_000
    assert asyncio.run(e._exh_budget_wait(T)) is None and len(slept) == 1 and 25 <= slept[0] <= 35   # waits for the minute, then reads
    T_late = int(now * 1000) - 230_000
    assert "no time left" in asyncio.run(e._exh_budget_wait(T_late)) and len(slept) == 1          # entry budget gone → fail closed
    e._exh_last_read_t = now - 90
    assert asyncio.run(e._exh_budget_wait(T_late)) is None                                     # a header older than 60 s never blocks
    monkeypatch.setattr(TE.binance_service, "exh_spot_used_weight", 5000, raising=False)
    e._exh_last_spot_t = now - 10
    assert asyncio.run(e._exh_budget_wait(T, spot=True)) is None and len(slept) == 2           # the spot check has the same escape
    assert TE.EXH_WEIGHT_STOP == 1500


def test_extra_reads_on_15m_passes_only_and_cap_is_journaled():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    p = eng.index("    async def _update_frenzy_pass(self, db):"); body = eng[p:eng.index("\n    def _frenzy_wide_off_note(", p)]
    assert "if _exh and bar_open % 900_000 == 0:" in body and "_exh_follow_rot" not in eng                 # no rotation on 5m passes
    assert "await self._exh_read_cap_note(_xp, bar_open)" in body and '"EXH_READ_CAP"' in eng


def test_read_cap_note_counts_and_journals(monkeypatch):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _patch_th(monkeypatch, TE)
    sp = 1_763_000_100_000 // M15 * M15

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        e = _engine(TE)
        e._exh_state()[("CAPUSDT", sp)] = dict(id=None, status="TRACKING", first_seen_ms=sp)
        await e._exh_read_cap_note("CAPUSDT", sp + M15)
        await e._exh_read_cap_note("CAPUSDT", sp + M15)                                      # a retry pass of the same bar: once
        async with SL() as s:
            rows = (await s.execute(select(models.ExhaustionSignal))).scalars().all()
        await eng.dispose()
        return e.blocks, rows
    blocks, rows = asyncio.run(go())
    assert blocks == ["EXH_READ_CAP"] and len(rows) == 1 and rows[0].block == "EXH_READ_CAP" and rows[0].status == "BLOCKED"
