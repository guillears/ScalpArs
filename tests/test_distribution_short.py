"""📉 DISTRIBUTION_SHORT (DECISION_LOG 586 — operator-approved PAPER run, declared override).

Pure rules (services/distribution.py) checked against the UNIFIED_SHORT pool the L1 study read (tests/fixtures/dist_parity.npz, built by
scripts/dist_parity_fixture.py: the builder's 15m candles, 5m rows and metrics), the REAL engine paths on SQLite with every exchange read
stubbed (no network), exits through the realtime path, config / UI / both exports parity, the scout's frozen gate."""
import asyncio
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import distribution as D  # noqa: E402
from tests.test_frenzy_willy import _hold_db  # noqa: E402

FIX = np.load(os.path.join(ROOT, "tests", "fixtures", "dist_parity.npz"))
NFIX = len([k for k in FIX.files if k.endswith("_meta")])
M5, M15, HOUR = 300_000, 900_000, 3_600_000


def _th(**kw):
    d = dict(distribution_short_enabled=True, distribution_pos_chg_4h_max=-0.01097, distribution_vsw_vol_mult=5.0, distribution_min_vol_usd=20e6,
             distribution_max_hours_after_spike=72.0, distribution_tp_atr_mult=20.0, distribution_tp_cap_pct=10.0, distribution_sl_atr_mult=1.5,
             distribution_sl_cap_pct=6.0, distribution_max_hold_hours=24.0, distribution_invest_mult=0.5, distribution_lev_mult=0.25,
             distribution_rsi1h_min=0.0,   # 🔒 the RSI1h filter is off here: the test_rsi1h_* tests below arm it
             distribution_crash_funding_block_enabled=False,   # 📉🌡 (651) off here: tests/test_dist_crash_funding_651.py drives it
             resting_tp_distribution_enabled=True, broker_backstop_pct=2.5,
             exhaustion_short_enabled=False, exhaustion_wide_enabled=False, exhaustion_max_hours_after_spike=72.0)
    d.update(kw)
    return NS(**d)


def _row(k):
    """fixture row k → (pair, T, raw 15m klines, FRENZY-window 5m rows, API-shaped position rows, meta)."""
    m = FIX[f"r{k}_meta"]
    raw = [[int(r[0]), r[1], r[2], r[3], r[4], 0.0, int(r[0]) + M15 - 1, r[5]] for r in FIX[f"r{k}_k15"]]
    k5 = [[int(r[0])] + [float(x) for x in r[1:]] for r in FIX[f"r{k}_k5"]]
    pos = [{"timestamp": int(a), "longShortRatio": f"{b:.6f}"} for a, b in FIX[f"r{k}_mx"]]
    return str(FIX[f"r{k}_pair"]), int(m[0]), raw, k5, pos, m


# ── ① parity with the study's rows ────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", range(NFIX))
def test_parity_with_the_study_rows(k):
    pair, T, raw, k5, pos, m = _row(k)
    f = D.features(raw, T, _th())
    assert f['ok'] and f['vsw'] and f['red']
    assert f['vsw_mult'] == pytest.approx(m[1], rel=1e-6) and f['bar_ret_pct'] == pytest.approx(m[2], abs=1e-6)
    assert f['vol24'] == pytest.approx(m[5], rel=1e-6)
    assert f['atr_pct'] * f['close'] / m[4] == pytest.approx(m[3], rel=2e-3)          # study ÷ the entry bar's open, live ÷ the signal close
    chg, ratio, tc, t4 = D.pos_chg_4h(pos, T)
    assert chg == pytest.approx(m[6], abs=1e-6) and tc <= T - D.POS_LAG_MS and tc - t4 >= D.POS_WIN_MS
    ps = D.prescreen(k5, T, _th())
    assert ps is not None and ps['proxy_mult'] == pytest.approx(m[1], rel=0.05)       # the pre-screen keeps every study signal
    ok, code, _ = D.decide(f, chg, _th())
    assert ok == (m[6] <= -0.01097) and (ok or code == D.B_POS_CHG)
    # 📉 observe-only stamps (the DIST_WL RSI1h ≥ 74.07 line): equal the study's up to the 499-candle warm-up
    assert f['rsi1h'] == pytest.approx(m[8], abs=0.05) and f['di_diff1h'] == pytest.approx(m[9], abs=0.05) and f['gap15'] == pytest.approx(m[10], abs=1e-4)


def test_feature_refusals_and_fail_closed():
    pair, T, raw, k5, pos, m = _row(0)
    assert D.features(raw[:-1], T, _th())['ok'] is False                                  # the window does not reach the signal close
    assert D.features(raw[-15:], T, _th())['ok'] is False                                 # < 21 candles
    f = D.features(raw, T, _th(distribution_vsw_vol_mult=50.0))
    assert D.decide(f, -0.1, _th(distribution_vsw_vol_mult=50.0))[1] == D.B_NOT_VSW
    assert D.decide(D.features(raw, T, _th()), -0.1, _th(distribution_min_vol_usd=1e15))[1] == D.B_VOL24
    assert D.decide(D.features(raw, T, _th()), None, _th())[1] == D.B_POS_UNREAD
    assert D.decide({}, -0.1, _th())[1] == D.B_NO_DATA
    green = [r[:] for r in raw]; green[-1][4] = green[-1][1] * 1.01
    assert D.features(green, T, _th())['vsw'] is False


def test_position_change_window_edges():
    T = 1_763_000_000_000 // M15 * M15
    rows = lambda pts: [{"timestamp": t, "longShortRatio": str(v)} for t, v in pts]
    good = rows([(T - 5 * 60_000, 1.50), (T - 5 * 60_000 - 4 * HOUR, 1.52), (T, 9.9)])     # the row AT T is not yet published (≤ T − 5 min)
    assert D.pos_chg_4h(good, T)[0] == pytest.approx(-0.02)
    assert D.pos_chg_4h(rows([(T - 31 * 60_000, 1.5), (T - 31 * 60_000 - 4 * HOUR, 1.6)]), T)[0] is None   # stale (> 30 min old)
    assert D.pos_chg_4h(rows([(T - 5 * 60_000, 1.5), (T - 5 * 60_000 - 4 * HOUR - 20 * 60_000, 1.6)]), T)[0] is None   # 4 h row too old
    assert D.pos_chg_4h(rows([(T - 5 * 60_000, 1.5), (T - 5 * 60_000 - 4 * HOUR - 15 * 60_000, 1.6)]), T)[0] == pytest.approx(-0.1)
    assert D.pos_chg_4h(None, T)[0] is None and D.pos_chg_4h([{"x": 1}], T)[0] is None


def test_exit_levels_rows_and_rule():
    th = _th()
    assert D.exit_levels(0.4, th) == (8.0, 0.6) and D.exit_levels(2.0, th) == (10.0, 3.0) and D.exit_levels(5.0, th) == (10.0, 6.0)
    assert D.exit_levels(None, th) == (None, None)
    assert D.row_levels(dict(entry_dist_tp_pct=4.0, entry_dist_sl_pct=2.0), th) == (4.0, 2.0)
    assert D.row_levels(dict(entry_dist_tp_pct=40.0, entry_dist_sl_pct=None), th) == (10.0, 6.0)   # a bad stamp → the cap, never wider
    row = dict(entry_dist_tp_pct=4.0, entry_dist_sl_pct=2.0)
    assert D.distribution_exit_for(1.0, 0.959, 10, th, row=row)[:2] == (True, "FRENZY_TP")
    assert D.distribution_exit_for(1.0, 0.959, 10, th, row=row, resting_tp=True)[0] is False
    assert D.distribution_exit_for(1.0, 1.021, 10, th, row=row)[:2] == (True, "STOP_LOSS")
    assert D.distribution_exit_for(1.0, 1.0, 24 * 60, th, row=row)[:2] == (True, "MAX_HOLD_TIME")
    assert D.distribution_exit_for(1.0, 1.0, 10, th, row=row)[0] is False
    from services.resting_tp import rtp_row_target, rtp_target, RTP_SLEEVES
    assert RTP_SLEEVES["DISTRIBUTION_SHORT"] == "resting_tp_distribution_enabled"
    o = NS(entry_strategy="DISTRIBUTION_SHORT", direction="SHORT", entry_price=1.0, quantity=100.0, entry_fee=0.045, entry_dist_tp_pct=4.0,
           entry_dist_sl_pct=2.0, exit_override_at=None, manual_exit_mode=None)
    net, why = rtp_row_target(th, o)
    assert why == "FRENZY_TP" and net == pytest.approx(4.0 - 0.045 - 0.018 * 0.96, abs=0.01)
    assert rtp_row_target(_th(resting_tp_distribution_enabled=False), o) is None and rtp_target(th, "DISTRIBUTION_SHORT")[0] == 10.0


def test_upper_bound_gate():
    t0 = dt.datetime(2026, 11, 20)
    few = [(t0 + dt.timedelta(hours=i), 1.0) for i in range(10)]
    assert D.upper_bound_gate(few, "X", "k", 586)[0] == "COLLECTING"
    short = [(t0 + dt.timedelta(hours=i), 1.0) for i in range(40)]                       # 40 fills in < 2 days → the cohort keeps growing
    assert D.upper_bound_gate(short, "X", "k", 586)[0] == "COLLECTING"
    bad = [(t0 + dt.timedelta(hours=9 * i), -1.0 + (0.3 if i % 3 == 0 else 0.0)) for i in range(30)]
    v, txt = D.upper_bound_gate(bad, "X", "distribution_short_enabled", 586)
    assert v == "OFF" and "distribution_short_enabled" in txt
    good = [(t0 + dt.timedelta(hours=9 * i), 0.5 if i % 2 else -0.3) for i in range(30)]
    assert D.upper_bound_gate(good, "X", "k", 586)[0] == "KEEP"
    ub = D.day_block_upper([(-1.0, "a"), (-1.0, "b"), (2.0, "c")])
    assert ub is not None and ub >= -1.0 and D.day_block_upper([(1.0, "a")]) is None


# ── ② the REAL engine path (pre-screen in _exhaustion_eval · reads · rule · order · journal) ──────────────────────────────────────

def _engine(TE):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True; e.is_running = True
    e.blocks = []
    e._record_filter_block = lambda n, d, had_room=True: e.blocks.append(n)
    return e


def _patch_th(monkeypatch, TE, **kw):
    for k, v in dict(vars(_th()), **kw).items():
        monkeypatch.setattr(TE.config.trading_config.thresholds, k, v, raising=False)


def _run(monkeypatch, k=0, *, held=False, open_ok=True, k15_ok=True, pos_ok=True, th_kw=None, universe="FRENZY", T_shift=0, k15_tail=None, late_s=5.0):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    pair, T, raw, k5, pos, m = _row(k)
    T += T_shift
    sp = int(m[7])
    _patch_th(monkeypatch, TE, **(th_kw or {}))
    monkeypatch.setattr(TE, "_leash_time", NS(time=lambda: T / 1000.0 + late_s))   # the decision happens late_s after the candle close
    calls = []

    async def kl(pid, interval, limit, end_ms=None, spot=False):
        calls.append((pid, interval, limit, end_ms, spot))
        return [r for r in raw if r[0] <= end_ms][-(k15_tail or limit):] if k15_ok else None

    async def pr(pid, end_ms, limit=60):
        calls.append(("pos", pid, end_ms, limit))
        return pos if pos_ok else None

    async def fr(sym, max_age_s=None):
        return 0.0001
    monkeypatch.setattr(TE.binance_service, "fetch_exh_klines", kl)
    monkeypatch.setattr(TE.binance_service, "fetch_exh_top_pos_ratio", pr)
    monkeypatch.setattr(TE.binance_service, "fetch_funding_rate", fr)
    monkeypatch.setattr(TE.binance_service, "exh_used_weight", 0, raising=False)
    opens = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        if held:
            async with SL() as s:
                s.add(models.Order(pair=pair, direction="SHORT", status="OPEN", is_paper=True, entry_strategy="EXHAUSTION_SHORT", confidence="STRONG_BUY",
                                   entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0))
                await s.commit()
        e = _engine(TE)
        e._exh_loaded = True; e._exh_mode = True
        e._exh_eps = {(pair, sp): dict(id=None, status="TRACKING", first_seen_ms=sp, universe=universe)}

        async def fake_open(db=None, **kw):
            opens.append(kw)
            return NS(id=77) if open_ok else None
        e.open_position = fake_open
        closed = [r for r in k5 if r[0] + M5 <= T]
        async with SL() as db:
            await e._exhaustion_eval(db, pair, None, closed, {"symbol": pair.replace("USDT", "/USDT:USDT")}, T)
            await e._distribution_judge_queued(db)
            await e._exhaustion_eval(db, pair, None, closed, {}, T)        # a retry pass of the same bar never judges twice
            await e._distribution_judge_queued(db)
        async with SL() as s:
            sig = (await s.execute(select(models.DistributionSignal))).scalars().all()
        await eng.dispose()
        return e, sig
    e, sig = asyncio.run(go())
    return e, opens, sig, calls, (pair, T, sp, m)


def test_engine_opens_on_a_study_row(monkeypatch):
    e, opens, sig, calls, (pair, T, sp, m) = _run(monkeypatch, k=0)
    assert len(opens) == 1 and len(sig) == 1 and sig[0].status == "OPENED" and sig[0].order_id == 77
    o = opens[0]
    assert o['distribution_short'] is True and o['direction'] == "SHORT" and o['distribution_bar_close_ms'] == T
    assert o['entry_dist_vsw_mult'] == pytest.approx(m[1], rel=1e-5) and o['entry_dist_pos_chg_4h'] == pytest.approx(m[6], abs=1e-6)
    tp, sl = D.exit_levels(o['entry_dist_atr_pct'], _th())
    assert (o['entry_dist_tp_pct'], o['entry_dist_sl_pct']) == (tp, sl) and o['entry_frenzy_spike_at'] == dt.datetime.utcfromtimestamp(sp / 1000)
    assert [c for c in calls if c[0] != "pos"] == [(pair, '15m', D.K15_LIMIT, T - 1, False)] and [c for c in calls if c[0] == "pos"] == [("pos", pair, T, 60)]
    assert o['entry_dist_rsi1h'] == pytest.approx(m[8], abs=0.05) and o['entry_dist_gap15'] == pytest.approx(m[10], abs=1e-4)   # observe-only
    assert sig[0].rsi1h == pytest.approx(m[8], abs=0.05) and sig[0].di_diff1h == pytest.approx(m[9], abs=0.05)
    assert sig[0].pos_chg_4h == pytest.approx(m[6], abs=1e-6) and sig[0].vsw_mult == pytest.approx(m[1], rel=1e-5) and e.blocks == []


def test_engine_refusals_are_journaled_and_counted(monkeypatch):
    e, opens, sig, calls, _ = _run(monkeypatch, k=NFIX - 1)                                # a study row whose positions did NOT fall
    assert opens == [] and sig[0].block == "DIST_POS_CHG" and "DIST_POS_CHG" in e.blocks
    for kw, code in ((dict(held=True), "DIST_PAIR_HELD"), (dict(k15_ok=False), "DIST_NO_DATA"), (dict(pos_ok=False), "DIST_POS_UNREAD"),
                     (dict(th_kw=dict(distribution_min_vol_usd=1e15)), "DIST_VOL24_LOW")):
        mp = pytest.MonkeyPatch()
        try:
            e, opens, sig, calls, _ = _run(mp, k=0, **kw)
        finally:
            mp.undo()
        assert opens == [] and len(sig) == 1 and sig[0].block == code and code in e.blocks, code
        if code == "DIST_PAIR_HELD":
            assert calls == []                                                             # one position per coin, before any read
        if code in ("DIST_VOL24_LOW", "DIST_NO_DATA"):
            assert not [c for c in calls if c[0] == "pos"]                                 # no position read for a candle the cheap gates refuse
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, calls, _ = _run(mp, k=0, open_ok=False)
    finally:
        mp.undo()
    assert len(opens) == 1 and sig[0].status == "REFUSED" and "DIST_OPEN_REFUSED" in e.blocks


def test_engine_scope_switch_wide_and_window(monkeypatch):
    e, opens, sig, calls, _ = _run(monkeypatch, k=0, universe="WIDE")                     # a WIDE episode is never DISTRIBUTION's universe
    assert opens == [] and sig == [] and calls == []
    for kw in (dict(th_kw=dict(distribution_short_enabled=False)), dict(th_kw=dict(distribution_max_hours_after_spike=0.01)), dict(T_shift=M5)):
        mp = pytest.MonkeyPatch()
        try:
            e, opens, sig, calls, _ = _run(mp, k=0, **kw)
        finally:
            mp.undo()
        assert opens == [] and sig == [] and calls == [], kw


def test_frenzy_pass_wiring_and_engine_sites():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    p = eng.index("    async def _update_frenzy_pass(self, db):"); body = eng[p:eng.index("\n    def _frenzy_wide_off_note(", p)]
    assert "_exh_trk = _exh or _dist" in body and body.index("await self._exhaustion_judge_queued(db)") < body.index("await self._distribution_judge_queued(db)")
    assert body.index("await self._distribution_judge_queued(db, skip_pairs=") < body.index("await self._exhaustion_judge_queued(db)")   # (I4) the other coins first
    assert "if not (exhaustion_enabled(_th) or distribution_enabled(_th))" in eng                     # the follow seed keeps tracked pairs for DIST too
    import services.trading_engine as TE
    th = _th()
    assert TE._ema13_cross_exit_applies("DISTRIBUTION_SHORT") is False
    assert TE.bot_own_stop_pct(th, "DISTRIBUTION_SHORT") == 6.0 and TE.order_backstop_pct(th, "DISTRIBUTION_SHORT", leverage=5) == 6.5
    for c in ("entry_dist_tp_pct", "entry_dist_pos_chg_4h", "entry_dist_model_slip_pct", "entry_dist_rsi1h", "entry_dist_di_diff1h", "entry_dist_gap15"):
        assert TE.TradingEngine.stamp_class(c) == "sleeve"
    assert TE.TradingEngine.stamp_class("entry_dist_from_ema13_pct") == "universal"                  # the shared column keeps its class
    assert "DISTRIBUTION_SHORT" in TE.TradingEngine.STAMP_BACKFILL_NO_RANK and TE.pair_reason_stampable("DIST_POS_CHG") is False
    assert eng.count("distribution_exit_for(") == 2 and eng.count("in EXHAUSTION_CLASS + DISTRIBUTION_CLASS") >= 6


# ── ③ open_position: size, stamps, stop, backstop, resting TP, lateness, book (REAL open path, paper) ───────────────────────────────

def _e2e(monkeypatch, n_open=0, th_extra=None, **kw):
    from tests.test_flatbase_short import _e2e_open
    return _e2e_open(monkeypatch, n_open=n_open, th_kw=dict(vars(_th()), **(th_extra or {})), **kw)


def test_open_position_size_stamps_and_exit_record(monkeypatch):
    import services.trading_engine as TE
    from services.exhaustion import model_slip_pct
    now_ms = int(__import__("time").time() * 1000)
    r, blocks, n_slot, n_bot, ref, ref_m = _e2e(monkeypatch, n_open=2, distribution_short=True, distribution_bar_close_ms=now_ms,
                                                 entry_frenzy_spike_at=dt.datetime(2026, 11, 17, 8, 0), entry_dist_bar_ret_pct=-1.2,
                                                 entry_dist_vsw_mult=6.1, entry_dist_pos_ratio=1.8, entry_dist_pos_chg_4h=-0.03, entry_dist_atr_pct=1.2,
                                                 entry_dist_tp_pct=10.0, entry_dist_sl_pct=1.8, entry_dist_decision_price=1.0, entry_dist_q5_usd=2e6,
                                                 entry_dist_rsi1h=76.5, entry_dist_di_diff1h=31.2, entry_dist_gap15=3.4)
    assert r is not None and r.entry_strategy == "DISTRIBUTION_SHORT" and r.direction == "SHORT" and r.cell_multiplier_source == "DISTRIBUTION_SHORT"
    assert n_slot == 3 and n_bot == 3 and r.leverage == 5                                         # a bot slot · 20 × 0.25
    assert r.entry_dist_tp_pct == 10.0 and r.entry_dist_sl_pct == 1.8 and r.entry_dist_vsw_mult == 6.1 and r.entry_dist_pos_chg_4h == -0.03
    assert r.entry_frenzy_spike_at == dt.datetime(2026, 11, 17, 8, 0) and r.entry_exh_tp_pct is None
    assert (r.entry_dist_rsi1h, r.entry_dist_di_diff1h, r.entry_dist_gap15) == (76.5, 31.2, 3.4)                     # observe-only stamps ride the fill
    assert r.entry_dist_model_slip_pct == pytest.approx(model_slip_pct(r.notional_value, 2e6))
    assert r.backstop_pct == 6.5 and r.resting_tp_price == pytest.approx(r.entry_price * 0.90, rel=2e-3)
    c = TE._open_orders_cache.get("FBSUSDT") or []
    assert c and c[-1]['entry_dist_tp_pct'] == 10.0 and c[-1]['stop_loss'] == -1.8 and c[-1]['entry_strategy'] == "DISTRIBUTION_SHORT"


def test_size_is_the_exhaustion_size(monkeypatch):
    """📈 (DECISION_LOG 586, operator 11-18) 0.5 × 5× WITH the RSI1h filter — the same margin share and leverage as EXHAUSTION_SHORT."""
    now_ms = int(__import__("time").time() * 1000)
    rd, *_ = _e2e(monkeypatch, distribution_short=True, distribution_bar_close_ms=now_ms, entry_dist_tp_pct=5.0, entry_dist_sl_pct=2.0)
    mp = pytest.MonkeyPatch()
    try:
        from tests.test_flatbase_short import _e2e_open
        from tests.test_exhaustion_short import _th as _exth
        rx, *_ = _e2e_open(mp, n_open=0, th_kw=dict(vars(_exth())),
                           exhaustion_short=True, exhaustion_bar_close_ms=now_ms, entry_exh_tp_pct=5.0, entry_exh_sl_pct=2.0)
    finally:
        mp.undo()
    assert rd.investment == pytest.approx(rx.investment) and rd.leverage == rx.leverage == 5


def test_open_position_book_full_and_late(monkeypatch):
    now_ms = int(__import__("time").time() * 1000)
    r, blocks, *_ = _e2e(monkeypatch, n_open=4, distribution_short=True, distribution_bar_close_ms=now_ms)
    assert r is None and "DIST_BOOK_FULL" in blocks
    mp = pytest.MonkeyPatch()
    try:
        r, blocks, *_ = _e2e(mp, distribution_short=True, distribution_bar_close_ms=now_ms - 301_000)
    finally:
        mp.undo()
    assert r is None and "DIST_LATE" in blocks


@pytest.mark.parametrize("price,held_min,want", [(0.959, 10, "FRENZY_TP"), (1.0181, 10, "STOP_LOSS"), (1.01, 24 * 60 + 1, "MAX_HOLD_TIME"),
                                                  (1.01, 10, None)])
def test_realtime_exit_path(monkeypatch, price, held_min, want):
    import models
    import services.trading_engine as TE
    _patch_th(monkeypatch, TE, resting_tp_distribution_enabled=False)
    closed = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        opened = dt.datetime.utcnow() - dt.timedelta(minutes=held_min)
        async with SL() as s:
            o = models.Order(pair="DSTUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy="DISTRIBUTION_SHORT", confidence="STRONG_BUY",
                             entry_price=1.0, quantity=100.0, investment=20.0, leverage=5, notional_value=100.0, entry_fee=0.045, opened_at=opened,
                             entry_dist_tp_pct=4.0, entry_dist_sl_pct=1.8)
            s.add(o); await s.commit(); oid = o.id
        e = _engine(TE)

        async def cp(db, order, px, reason):
            closed.append(reason); order.status = "CLOSED"; return order
        e.close_position = cp
        TE._open_orders_cache["DSTUSDT"] = [dict(id=oid, direction="SHORT", entry_price=1.0, quantity=100.0, entry_fee=0.045, stop_loss=-1.8,
                                                 entry_strategy="DISTRIBUTION_SHORT", opened_at=opened, peak_pnl=0.0, trough_pnl=0.0, confidence="STRONG_BUY",
                                                 entry_dist_tp_pct=4.0, entry_dist_sl_pct=1.8)]
        await e.check_realtime_stop_loss("DSTUSDT", price)
        await eng.dispose()
    asyncio.run(go())
    assert closed == ([want] if want else [])


def test_cache_rebuild_reloads_the_row_exit_levels(monkeypatch):
    import models
    import services.trading_engine as TE

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        async with SL() as s:
            s.add(models.Order(pair="DSRUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy="DISTRIBUTION_SHORT", confidence="STRONG_BUY",
                               entry_price=1.0, quantity=100.0, investment=20.0, leverage=5, notional_value=100.0, entry_fee=0.045,
                               opened_at=dt.datetime.utcnow(), entry_dist_tp_pct=7.5, entry_dist_sl_pct=2.25))
            await s.commit()
        e = TE.TradingEngine(); e.is_paper_mode = True
        monkeypatch.setattr(TE, "_open_orders_cache", {})
        async with SL() as db:
            await e.update_orders_cache(db)
        c = TE._open_orders_cache["DSRUSDT"][0]
        await eng.dispose()
        return c
    c = asyncio.run(go())
    assert c['entry_dist_tp_pct'] == 7.5 and c['entry_dist_sl_pct'] == 2.25 and c['stop_loss'] == -2.25


# ── ④ config / UI / exports / scout / registry (D11 / D12) ─────────────────────────────────────────────────────────────────────────

FIELDS = {"distribution_short_enabled": (False, True), "distribution_pos_chg_4h_max": (-0.01097, -0.01097), "distribution_vsw_vol_mult": (5.0, 5.0),
          "distribution_min_vol_usd": (20e6, 20e6), "distribution_max_hours_after_spike": (72.0, 72.0), "distribution_tp_atr_mult": (20.0, 20.0),
          "distribution_tp_cap_pct": (10.0, 10.0), "distribution_sl_atr_mult": (1.5, 1.5), "distribution_sl_cap_pct": (6.0, 6.0),
          "distribution_max_hold_hours": (24.0, 24.0), "distribution_invest_mult": (0.5, 0.5), "distribution_lev_mult": (0.25, 0.25),
          "resting_tp_distribution_enabled": (True, True)}


def test_config_defaults_json_ui_and_exports():
    import config as C
    mf = C.SignalThresholds.model_fields
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, (d, j) in FIELDS.items():
        assert mf[k].default == d and cfg[k] == j, k
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 586" in src and "OPERATOR-DECLARED" in src and "Kelly at" in src
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    ids = ["config-dist-enabled", "config-rtp-distribution-enabled", "config-dist-pos-chg", "config-dist-vsw-mult", "config-dist-min-vol",
           "config-dist-max-hours", "config-dist-tp-atr", "config-dist-tp-cap", "config-dist-sl-atr", "config-dist-sl-cap", "config-dist-max-hold",
           "config-dist-invest-mult", "config-dist-lev-mult"]
    for i in ids:
        assert html.count(f'id="{i}"') == 1, i
    for k in FIELDS:
        if k not in ("distribution_short_enabled", "resting_tp_distribution_enabled"):
            assert f", '{k}', " in html, k
    assert "for (const [_id, _key, _def] of DISTRIBUTION_NUM_FIELDS)" in html and "...Object.fromEntries(DISTRIBUTION_NUM_FIELDS.map(" in html
    for k, i in (("distribution_short_enabled", "config-dist-enabled"), ("resting_tp_distribution_enabled", "config-rtp-distribution-enabled")):
        assert f"config.thresholds.{k}" in html and f"{k}: document.getElementById('{i}')" in html, k
    assert '<option value="DISTRIBUTION_SHORT">' in html and 'data-ss="dist"' in html and "S.dist = {" in html
    assert html.count("_stratRaw === 'DISTRIBUTION_SHORT' ? 'DIST-S'") == 2
    assert "📉 DISTRIBUTION_SHORT (DECISION_LOG 586" in html and html.count("..._buildConfigLines(cfg, changelog, hr, hr2, status)") == 2
    assert html.count("EXHAUSTION_SHORT · DISTRIBUTION_SHORT)") == 2                                  # the FRENZY Fills header: UI + the shared export block
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '("DISTRIBUTION_SHORT", "📉 DISTRIBUTION_SHORT (' in main and 'if _fz_es == "DISTRIBUTION_SHORT":' in main
    assert "_dist_cfg_errors(new_config.thresholds, _sb_lev)" in main and "'DISTRIBUTION_SHORT': ('distribution_short_enabled',)" in main
    assert 'e="DISTRIBUTION"' in main and "+ _exh_cols + _dist_cols +" in main and "delete(DistributionSignal)" in main


def test_config_errors():
    import config
    th = config.trading_config.thresholds
    assert D.distribution_config_errors(th, 20.0) == []
    bad = NS(**dict(th.model_dump(), distribution_pos_chg_4h_max=0.01, distribution_short_enabled="yes", distribution_vsw_vol_mult=0.5))
    errs = " | ".join(D.distribution_config_errors(bad, 20.0))
    for k in ("distribution_pos_chg_4h_max", "distribution_short_enabled", "distribution_vsw_vol_mult"):
        assert k in errs, k
    wide = NS(**dict(th.model_dump(), distribution_lev_mult=1.0, distribution_sl_cap_pct=10.0))
    assert "too wide" in " ".join(D.distribution_config_errors(wide, 20.0))
    assert D.distribution_params(NS(distribution_invest_mult=7, distribution_lev_mult=-1))['invest'] == 0.5


def test_sleeve_rows_and_labels():
    import main as M
    o = NS(entry_strategy="DISTRIBUTION_SHORT", direction="SHORT", pnl_percentage=1.5, pnl=3.0, opened_at=dt.datetime(2026, 11, 18, 8),
           closed_at=dt.datetime(2026, 11, 18, 9), investment=60.0, notional_value=300.0, leverage=5, peak_pnl=2.0, import_source=None)
    assert any(r.get('sleeve') == 'Distribution-Short' for r in M._compute_sleeve_performance([o], start_balance=1000.0))
    assert M._compute_strategy_performance([o])[0]["strategy"].startswith("DISTRIBUTION_SHORT")
    assert M._SLEEVE_LABELS[("DISTRIBUTION_SHORT", "SHORT")] == "DISTRIBUTION Short"
    assert M._sleeve_enabled(NS(distribution_short_enabled=False), "DISTRIBUTION_SHORT") is False
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import screen_pool as SP
    assert SP.sleeve(dict(status="CLOSED", pair="X", entry_strategy="DISTRIBUTION_SHORT", direction="SHORT")) is None


def test_scout_gate_frozen_once(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_distribution_586 as S
    t0 = pd.Timestamp("2026-11-20")
    rows = [dict(opened_at=str(t0 + pd.Timedelta(hours=9 * i)), pair=f"P{i}USDT", direction="SHORT", status="CLOSED",
                 entry_strategy="DISTRIBUTION_SHORT", pnl=-1.0, pnl_percentage=-1.0, close_reason="STOP_LOSS", notional_value=1000.0,
                 entry_dist_model_slip_pct=0.05, entry_dist_q5_usd=1e6, entry_depth_mode=None, entry_depth_walk_pct=None) for i in range(31)]
    fr = pd.DataFrame(rows); fr["o"] = pd.to_datetime(fr.opened_at)
    st = str(tmp_path / "s.json")
    txt = S.report(frame=fr, since_ms=int(t0.value // 10**6), state_path=st, tally={"OPENED": 31, "DIST_POS_CHG": 5})
    assert "FROZEN" in txt and "OFF" in txt and "distribution_short_enabled" in txt and "DIST_POS_CHG 5" in txt
    good = fr.assign(pnl_percentage=2.0, close_reason="FRENZY_TP")
    assert "FROZEN" in S.report(frame=good, since_ms=int(t0.value // 10**6), state_path=st, tally={}) and \
        json.load(open(st))["verdict"] == "OFF"                                                       # frozen once: a later read never re-decides


def test_registry_hook_and_gitignore():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import observe_registry as R
    assert "scout_distribution_586.py" in R.BOOKED
    assert "import scout_distribution_586 as _ds586" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "reports/SCOUT_DISTRIBUTION_586.json" in open(os.path.join(ROOT, ".gitignore")).read()


# ── ⑤ 🔒 the RSI1h ≥ 74.07 entry filter (DECISION_LOG 586, operator 11-18, DECLARED OVERRIDE) ─────────────────────────────────────────

def test_rsi1h_block_rule():
    on = _th(distribution_rsi1h_min=74.07)
    assert D.rsi1h_block(50.0, _th()) == (False, None, "")                                  # 0 = off (code default)
    assert D.rsi1h_block(None, _th())[0] is False
    assert D.rsi1h_block(74.07, on)[0] is False and D.rsi1h_block(80.0, on)[0] is False
    assert D.rsi1h_block(74.06, on)[:2] == (True, D.B_RSI1H)
    assert D.rsi1h_block(None, on)[:2] == (True, D.B_RSI1H_UNREAD) and D.rsi1h_block(float("nan"), on)[:2] == (True, D.B_RSI1H_UNREAD)
    assert D.distribution_params(NS(distribution_rsi1h_min=150))['rsi1h_min'] == 74.07         # (review I1) a bad value → the armed cut, never off
    assert D.distribution_params(NS())['rsi1h_min'] == 74.07 and D.rsi1h_block(50.0, NS())[1] == D.B_RSI1H   # a missing value keeps the cut
    assert D.distribution_params(NS(distribution_rsi1h_min=0))['rsi1h_min'] == 0.0             # OFF only by an explicit 0


def test_rsi1h_engine_pass_block_and_unread(monkeypatch):
    armed = dict(distribution_rsi1h_min=74.07)
    e, opens, sig, calls, (pair, T, sp, m) = _run(monkeypatch, k=1, th_kw=armed)             # HIFI: study RSI1h 74.82 ≥ 74.07, positions fell
    assert m[8] >= 74.07 and len(opens) == 1 and sig[0].status == "OPENED" and "RSI 74.8" in sig[0].reason.replace("1h RSI", "RSI")
    mp = pytest.MonkeyPatch()
    try:
        e, opens, sig, calls, (pair, T, sp, m) = _run(mp, k=0, th_kw=armed)                  # LPT: study RSI1h 50.4 → blocked, passed every L1 gate
    finally:
        mp.undo()
    assert opens == [] and sig[0].block == "DIST_RSI1H" and "DIST_RSI1H" in e.blocks
    assert sig[0].rsi1h == pytest.approx(m[8], abs=0.05) and sig[0].pos_chg_4h == pytest.approx(m[6], abs=1e-6)
    assert sig[0].tp_pct and sig[0].sl_pct and sig[0].decision_price                          # what the blocked-side gate prices with
    mp = pytest.MonkeyPatch()
    try:
        import services.trading_engine as TE                                                  # an unreadable 1h RSI (e.g. gapped / young coin)
        mp.setattr(TE._dist_rules, "observe_stamps", lambda t, h, l, c: dict(rsi1h=None, di_diff1h=None, gap15=None))
        e, opens, sig, calls, _ = _run(mp, k=1, th_kw=armed)
    finally:
        mp.undo()
    assert opens == [] and sig[0].block == "DIST_RSI1H_UNREAD" and "DIST_RSI1H_UNREAD" in e.blocks and sig[0].rsi1h is None


def test_rsi1h_config_json_ui_and_exports():
    import config as C
    assert C.SignalThresholds.model_fields["distribution_rsi1h_min"].default == 74.07                  # (review I1) armed by default
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["distribution_rsi1h_min"] == 74.07
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "RSI1h ENTRY FILTER (DECISION_LOG 586" in src and "t 3.24 vs 5.62" in src
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-dist-rsi1h-min"') == 1 and "'distribution_rsi1h_min'" not in html.split("const DISTRIBUTION_NUM_FIELDS")[1][:900]
    assert "distribution_rsi1h_min: (function(){ const _e = document.getElementById('config-dist-rsi1h-min');" in html     # own save: blank / invalid → stored or 74.07
    assert "_e.dataset.stored = _ok ? String(_v) : '74.07'" in html and "return (Number.isFinite(st) && st >= 0 && st < 100) ? st : 74.07;" in html
    assert "🔒 RSI1h filter (declared override)" in html and "🔒 RSI1h ≥ ${Number(d.rsi1h_min) > 0" in html   # config line + monitor line (both exports)
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "_dist_rules.rsi1h_block(f.get('rsi1h'), th)" in eng
    import config
    assert not D.distribution_config_errors(config.trading_config.thresholds, 20.0)
    bad = NS(**dict(config.trading_config.thresholds.model_dump(), distribution_rsi1h_min=-1))
    assert "distribution_rsi1h_min" in " ".join(D.distribution_config_errors(bad, 20.0))


def test_rsi1h_gates(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_distribution_586 as S
    t0 = dt.datetime(2026, 11, 20)
    assert S.kept_gate([(t0 + dt.timedelta(hours=i), 1.0) for i in range(80)])[0] == "COLLECTING"          # 80 fills in < 4 days
    v, txt, n = S.kept_gate([(t0 + dt.timedelta(hours=5 * i), 0.2) for i in range(70)])
    assert v == "DISARM" and n == 60 and "disarm" in txt
    assert S.kept_gate([(t0 + dt.timedelta(hours=5 * i), 0.9) for i in range(70)])[0] == "KEEP"
    day = lambda i: (t0 + dt.timedelta(hours=9 * i)).strftime("%Y-%m-%d")
    assert S.blocked_gate([(day(i), 2.0) for i in range(10)])[0] == "COLLECTING"
    assert S.blocked_gate([(day(i), 2.0 + (0.5 if i % 2 else -0.5)) for i in range(30)])[0] == "BLOCKS_WINNERS"
    assert S.blocked_gate([(day(i), 0.0 + (3.0 if i % 2 else -3.0)) for i in range(30)])[0] == "KEEP"
    # the report: kept side from the fills frame, blocked side from the journal through an injected pricer (no network)
    fills = pd.DataFrame([dict(opened_at=str(pd.Timestamp(t0) + pd.Timedelta(hours=5 * i)), pair=f"K{i}USDT", direction="SHORT", status="CLOSED",
                               entry_strategy="DISTRIBUTION_SHORT", pnl=1.0, pnl_percentage=0.2, close_reason="MAX_HOLD_TIME", notional_value=3000.0,
                               entry_dist_model_slip_pct=0.0, entry_dist_q5_usd=1e6, entry_depth_mode=None, entry_depth_walk_pct=None) for i in range(65)])
    fills["o"] = pd.to_datetime(fills.opened_at)
    jr = pd.DataFrame([dict(e="DISTRIBUTION", pair=f"B{i}USDT", dist_signal_at=(t0 + dt.timedelta(hours=9 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
                            dist_status="BLOCKED", dist_block="DIST_RSI1H", dist_decision_price=1.0, dist_tp_pct=8.0, dist_sl_pct=3.0,
                            dist_atr_pct=2.0, dist_q5_usd=1e4, dist_rsi1h=60.0) for i in range(32)])
    seen = []

    def pricer(pair, t_ms, entry, tp, sl, slip, cache, now_ms=None):
        seen.append((pair, tp, sl, slip))
        return (2.5 if len(seen) % 2 else 1.5), "FRENZY_TP"
    txt = S.report(frame=fills, since_ms=int(pd.Timestamp(t0).value // 10**6), state_path=str(tmp_path / "a.json"), tally={}, journal=jr,
                   pricer=pricer, state_kept=str(tmp_path / "k.json"), state_blocked=str(tmp_path / "b.json"), cache_path=None)
    assert "(a) KEPT: FROZEN" in txt and "DISARM" in txt and "(b) BLOCKED: FROZEN" in txt and "BLOCKS_WINNERS" in txt
    assert seen[0][1:3] == (8.0, 3.0) and seen[0][3] > 0                                          # journaled levels · slip at the median kept notional
    assert json.load(open(tmp_path / "k.json"))["verdict"] == "DISARM" and json.load(open(tmp_path / "b.json"))["verdict"] == "BLOCKS_WINNERS"
    import observe_registry as R
    assert R.BOOKED["scout_distribution_586.py"]["distribution_rsi1h_min"] == 74.07
    gi = open(os.path.join(ROOT, ".gitignore")).read()
    assert "reports/SCOUT_DIST_RSI1H_KEPT_586.json" in gi and "reports/SCOUT_DIST_RSI1H_BLOCKED_586.json" in gi and "reports/.dist_rsi1h_cache.json" in gi


# ── ⑥ review fixes (DECISION_LOG 586 review I4 / I5 / I6 / M1 / M2 / M3) ─────────────────────────────────────────────────────────────

def test_late_signal_is_skipped_before_any_read(monkeypatch):
    e, opens, sig, calls, _ = _run(monkeypatch, k=0, late_s=290.0)                           # 10 s left < the 20 s reserve
    assert opens == [] and calls == [] and sig[0].block == "DIST_LATE" and sig[0].status == "REFUSED" and "DIST_LATE" in e.blocks


def test_skip_pairs_split_keeps_the_rest_for_after_exhaustion(monkeypatch):
    import services.trading_engine as TE
    _patch_th(monkeypatch, TE)
    e = _engine(TE)
    seen = []

    async def judge(db, pair, sp, closed, info, T):
        seen.append(pair)
    e._distribution_judge = judge
    e._dist_cand_q = [("AUSDT", 1, [], {}, 900_000), ("BUSDT", 1, [], {}, 900_000)]
    asyncio.run(e._distribution_judge_queued(None, skip_pairs={"BUSDT"}))
    assert seen == ["AUSDT"] and [c[0] for c in e._dist_cand_q] == ["BUSDT"]
    asyncio.run(e._distribution_judge_queued(None))
    assert seen == ["AUSDT", "BUSDT"] and e._dist_cand_q == []


def test_judge_failure_is_counted_and_journaled(monkeypatch):
    import services.trading_engine as TE

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(TE._dist_rules, "features", boom)
    e, opens, sig, calls, _ = _run(monkeypatch, k=0)
    assert opens == [] and "DIST_FAILED" in e.blocks and sig[0].block == "DIST_FAILED" and sig[0].status == "BLOCKED"


def test_prescreen_failure_is_its_own_counter(monkeypatch):
    import services.trading_engine as TE

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(TE._dist_rules, "prescreen", boom)
    e, opens, sig, calls, _ = _run(monkeypatch, k=0)
    assert e.blocks.count("DIST_PRESCREEN_FAILED") == 1 and "EXH_FAILED" not in e.blocks and sig == []


def test_window_cannot_outlive_exhaustion():
    import config
    th = config.trading_config.thresholds
    bad = NS(**dict(th.model_dump(), distribution_max_hours_after_spike=80.0, exhaustion_max_hours_after_spike=72.0))
    assert "must be ≤ exhaustion_max_hours_after_spike" in " ".join(D.distribution_config_errors(bad, 20.0))
    ok = NS(**dict(th.model_dump(), distribution_max_hours_after_spike=48.0, exhaustion_max_hours_after_spike=72.0))
    assert D.distribution_config_errors(ok, 20.0) == []


def test_position_read_has_its_own_backoff():
    src = open(os.path.join(ROOT, "services", "binance_service.py"), encoding="utf-8").read()
    i = src.index("async def fetch_exh_top_pos_ratio"); body = src[i:src.index("async def fetch_exh_spot_symbols", i)]
    assert "exh_pos_limited_until" in body and "exh_oi_limited_until" not in body and "self.exh_pos_limited_until = 0.0" in src


def test_api_rounded_ratio_and_lag_stamps(monkeypatch):
    T = 1_763_000_000_000 // M15 * M15
    rows = lambda now, old: [{"timestamp": T - 5 * 60_000, "longShortRatio": now}, {"timestamp": T - 5 * 60_000 - 4 * HOUR, "longShortRatio": old}]
    chg, _, tc, t4 = D.pos_chg_4h(rows("1.5000", "1.5110"), T)
    assert chg == pytest.approx(-0.0110) and D.decide(dict(ok=True, vsw=True, vol24=1e9), chg, _th())[0] is True   # API 4 decimals: −0.0110 passes
    chg2 = D.pos_chg_4h(rows("1.5000", "1.5109"), T)[0]
    assert chg2 == pytest.approx(-0.0109) and D.decide(dict(ok=True, vsw=True, vol24=1e9), chg2, _th())[1] == D.B_POS_CHG   # effective cut ≤ −0.0110
    e, opens, sig, calls, (pair, T, sp, m) = _run(monkeypatch, k=0)
    o = opens[0]
    assert 5 <= o['entry_dist_pos_lag_min'] <= 30 and 240 <= o['entry_dist_pos_span_min'] <= 255
    assert sig[0].pos_lag_min == o['entry_dist_pos_lag_min'] and sig[0].pos_span_min == o['entry_dist_pos_span_min']
    import services.trading_engine as TE
    assert TE.TradingEngine.stamp_class("entry_dist_pos_lag_min") == "sleeve"


def test_blocked_gate_waits_for_a_kept_fill(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_distribution_586 as S
    t0 = dt.datetime(2026, 11, 20)
    empty = pd.DataFrame(columns=["opened_at", "o", "pnl_percentage", "close_reason", "notional_value", "entry_dist_model_slip_pct",
                                  "entry_dist_q5_usd", "entry_depth_mode", "entry_depth_walk_pct", "entry_strategy", "pair"])
    jr = pd.DataFrame([dict(e="DISTRIBUTION", pair=f"B{i}USDT", dist_signal_at=(t0 + dt.timedelta(hours=9 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
                            dist_status="BLOCKED", dist_block="DIST_RSI1H", dist_decision_price=1.0, dist_tp_pct=8.0, dist_sl_pct=3.0,
                            dist_atr_pct=2.0, dist_q5_usd=1e4, dist_rsi1h=60.0) for i in range(32)])
    txt = S.report(frame=empty, since_ms=int(pd.Timestamp(t0).value // 10**6), state_path=str(tmp_path / "a.json"), tally={}, journal=jr,
                   pricer=lambda *a, **k: (2.5, "FRENZY_TP"), state_kept=str(tmp_path / "k.json"), state_blocked=str(tmp_path / "b.json"), cache_path=None)
    assert "BLOCKS_WINNERS" in txt and "(b) BLOCKED: FROZEN" not in txt and not os.path.exists(tmp_path / "b.json")
    assert "DIST_RSI1H_UNREAD blocks are not priced" in txt and "does not freeze until" in txt


# ── ⚡ DECISION_LOG 599: the fast path (DISTRIBUTION judged at the start of a 15m-close pass) ─────────────────────────────────────────

def _run_fast(monkeypatch, k=0, *, k15_ok=True, allowed=True, th_kw=None, then_pass=True):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    pair, T, raw, k5, pos, m = _row(k)
    sp = int(m[7])
    _patch_th(monkeypatch, TE, **(th_kw or {}))
    monkeypatch.setattr(TE, "_leash_time", NS(time=lambda: T / 1000.0 + 5.0))
    calls = []

    async def kl(pid, interval, limit, end_ms=None, spot=False):
        calls.append((pid, interval, limit, end_ms))
        return [r for r in raw if r[0] <= end_ms][-limit:] if k15_ok else None

    async def pr(pid, end_ms, limit=60):
        calls.append(("pos", pid))
        return pos

    async def fr(sym, max_age_s=None):
        return 0.0001
    monkeypatch.setattr(TE.binance_service, "fetch_exh_klines", kl)
    monkeypatch.setattr(TE.binance_service, "fetch_exh_top_pos_ratio", pr)
    monkeypatch.setattr(TE.binance_service, "fetch_funding_rate", fr)
    monkeypatch.setattr(TE.binance_service, "exh_used_weight", 0, raising=False)
    opens = []

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        e = _engine(TE)
        e._exh_loaded = True; e._exh_mode = True
        e._exh_eps = {(pair, sp): dict(id=None, status="TRACKING", first_seen_ms=sp, universe="FRENZY")}

        async def fake_open(db=None, **kw):
            opens.append(kw)
            return NS(id=88)
        e.open_position = fake_open
        closed = [r for r in k5 if r[0] + M5 <= T]
        async with SL() as db:
            await e._distribution_fast(db, T, ({pair} if allowed else set()), {})
            claimed = (pair, T) in e.__dict__.get('_dist_judged', set())
            if then_pass:   # the same bar's normal pass (pre-screen + queue) afterwards
                await e._exhaustion_eval(db, pair, None, closed, {}, T)
                await e._distribution_judge_queued(db)
        async with SL() as s:
            sig = (await s.execute(select(models.DistributionSignal))).scalars().all()
        await eng.dispose()
        return e, sig, claimed
    e, sig, claimed = asyncio.run(go())
    return e, opens, sig, calls, claimed, (pair, T)


def test_fast_path_opens_once_and_the_pass_never_rejudges(monkeypatch):
    e, opens, sig, calls, claimed, (pair, T) = _run_fast(monkeypatch, k=0)
    assert len(opens) == 1 and opens[0]['distribution_short'] is True and opens[0]['distribution_bar_close_ms'] == T
    assert len(sig) == 1 and sig[0].status == "OPENED" and sig[0].order_id == 88 and claimed
    assert [c for c in calls if c[0] != "pos"] == [(pair, '15m', D.K15_LIMIT, T - 1), (pair, '5m', 301, T - 1)]   # the pass never re-reads
    assert opens[0]['entry_atr_pct'] is not None                                               # (review M1) the stamp from a FRESH closed 5m read


def test_fast_path_unread_coin_falls_back_to_the_pass(monkeypatch):
    e, opens, sig, calls, claimed, _ = _run_fast(monkeypatch, k=0, k15_ok=False)
    assert not claimed and opens == [] and len(sig) == 1 and sig[0].block == "DIST_NO_DATA"   # released → today's path judged it


def test_fast_path_respects_the_pass_universe(monkeypatch):
    e, opens, sig, calls, claimed, _ = _run_fast(monkeypatch, k=0, allowed=False, then_pass=False)
    assert not claimed and opens == [] and sig == [] and calls == []                          # blacklisted / unlisted → no read


def test_fast_path_below_prescreen_is_released(monkeypatch):
    e, opens, sig, calls, claimed, _ = _run_fast(monkeypatch, k=0, th_kw=dict(distribution_vsw_vol_mult=50.0), then_pass=False)
    assert not claimed and opens == [] and sig == [] and [c for c in calls if c[0] == "pos"] == []   # (review M2) the in-pass proxy decides


def test_fast_path_failure_releases_its_claims(monkeypatch):
    import services.trading_engine as TE

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(TE._dist_rules, "features", boom)
    e, opens, sig, calls, claimed, _ = _run_fast(monkeypatch, k=0, then_pass=False)
    assert not claimed and opens == []                                                        # (review L2) never claimed-but-unjudged


def test_fast_path_runs_first_in_the_pass():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = src.index("async def _update_frenzy_pass")
    body = src[i:src.index("\n    async def ", i + 40)]
    assert body.index("await self._distribution_fast(db, bar_open,") < body.index("_frenzy_kl_cache")   # before any FRENZY kline read
    assert "bar_open % _dist_rules.M15_MS == 0" in body


def test_dist_tp_shadow_605():
    """🎯👁 (DECISION_LOG 605) DIST_TP_SHADOW — observe-only TP 5 / 8 vs today's TP on real DIST fills: walk, read, freeze, holds."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_dist_tp_shadow as S
    import observe_registry as R
    m, te, E = 60_000, 1_800_000_000_000, 100.0
    bar = lambda k, o, h, l: [te // m * m + (k + 1) * m, o, h, l, o]
    assert S.walk([[te // m * m, 100, 200, 1, 100]], E, te, 10, 6)[0] == "UNPRICED"          # the fill minute is never read
    b = [bar(0, 100, 100, 94.9), bar(1, 95, 95, 89.9)]
    assert S.walk(b, E, te, 5, 6) == ("TP", 5 - S.FEE_IN - S.FEE_MK) and S.walk(b, E, te, 10, 6) == ("TP", 10 - S.FEE_IN - S.FEE_MK)
    assert S.walk([bar(0, 100, 100, 95.0)], E, te, 5, 6)[0] == "PENDING"                      # a touch is not a fill (needs 0.05 % through)
    w, v = S.walk([bar(0, 100, 106.5, 90)], E, te, 5, 6, slip=0.1)                           # stop first on a bar touching both
    assert w == "STOP" and abs(v - (-6 - S.FEE_IN - S.FEE_TK - 0.2)) < 1e-9
    w, v = S.walk([bar(0, 108, 109, 107)], E, te, 5, 6)                                      # a gap through the stop books the open
    assert w == "STOP" and abs(v - (-8 - S.FEE_IN - S.FEE_TK - S.EXIT_SLIP_MIN)) < 1e-9       # a taker exit pays ≥ 0.10 slip
    w, v = S.walk([bar(0, 100, 101, 99), [te + 24 * S.HOUR, 97.0, 97, 97, 97]], E, te, 5, 6)
    assert w == "TIME" and abs(v - (3 - S.FEE_IN - S.FEE_TK - S.EXIT_SLIP_MIN)) < 1e-9
    w, v = S.walk([bar(0, 100, 106.5, 90)], E, te, 5, 6, slip=0.3, entry_slip=0.1)         # impact fill: only the entry leg still owed
    assert abs(v - (-6 - S.FEE_IN - S.FEE_TK - 0.1 - 0.3)) < 1e-9
    assert S.prefix([(d,) for d in range(29)]) is None and S.prefix([(1,)] * 40) is None and S.prefix([(d,) for d in range(30)]) == 30
    assert S.arm_verdict([1.0] * 30, list(range(30)))[0] == "SHIP CANDIDATE"
    assert S.arm_verdict([-1.0] * 30, list(range(30)))[0] == "NO SHIP"
    assert S.arm_verdict([0.0] * 28 + [20.0, 20.0], list(range(30)))[0] == "NO SHIP"        # all gain in the best 5 %
    # report(): a stopped fetch holds and never freezes; NOT_EVALUABLE leaves; the read freezes ONCE
    t0 = pd.Timestamp("2026-12-01 00:00:20")
    fr = pd.DataFrame([dict(pair=f"P{i}USDT", o=t0 + pd.Timedelta(hours=10 * i), entry_price=100.0, entry_dist_tp_pct=10.0, entry_dist_sl_pct=6.0,
                            entry_dist_model_slip_pct=0.1, pnl_percentage=9.0, close_reason="TP") for i in range(35)])
    def bars_fn(p, t, now):
        if p == "P0USDT":
            return []
        return [[t // m * m + m, 100.0, 100.0, 89.0, 90.0]]                                  # straight through −10
    now = int((t0 + pd.Timedelta(days=40)).value // 1_000_000)
    fl = int(t0.value // 1_000_000) - 1
    import tempfile
    sp = os.path.join(tempfile.mkdtemp(), "s.json")
    o1 = "\n".join(S.report(frame=fr, since_ms=fl, state_path=sp, now_ms=now, newest=pd.Timestamp(now, unit="ms"),
                            bars_fn=lambda p, t, n: None if p == "P3USDT" else bars_fn(p, t, n)))
    assert "COLLECTING — 2/30" in o1                                                           # P0 out (no path), the read waits at the held P3
    # risk_read buckets each arm by its OWN close day; equal-drawdown n/a when an arm never loses
    rr = S.risk_read([((1, 0, 0), -6.0, 4.0, 7.0), ((2, 2, 2), 9.0, 4.0, 7.0)], 0, 2)
    assert rr["TODAY"]["worst"] < 0 and rr["TP5"]["worst"] == 0 and rr["TP5"]["eq_dcr"] is None and rr["TODAY"]["eq_dcr"] is not None
    assert not os.path.exists(sp)
    o2 = "\n".join(S.report(frame=fr, since_ms=fl, state_path=sp, now_ms=now, newest=pd.Timestamp(now, unit="ms"), bars_fn=bars_fn))
    assert "P0USDT" in o2 and "NOT_EVALUABLE" in o2 and "Frozen read" in o2 and "TP 5: NO SHIP" in o2
    st = json.load(open(sp))
    st[S.KEY]["arms"]["TP 5"]["state"] = "X"
    json.dump(st, open(sp, "w"))
    o3 = "\n".join(S.report(frame=fr, since_ms=fl, state_path=sp, now_ms=now, newest=pd.Timestamp(now, unit="ms"), bars_fn=bars_fn))
    assert "TP 5: X" in o3                                                                    # frozen once, never re-decided
    assert R.EXIT["scout_dist_tp_shadow.py"]["dl"] == "605 · 647" and S.DL == 605   # 🪢 (647) the trail line rides the file
    assert "import scout_dist_tp_shadow as _dtp605" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
