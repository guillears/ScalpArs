"""🟢 Scout trackers WIDE_BY_CODE + FRENZY_GREEN_CLOCK (observe-only, 2026-10-06) — the pure rules: refusal code, lock on ticks, the
pre-registered bars (reports/FRENZY_GREEN_AND_WIDE_ATR_FORMAL_2026-10-06.md, end of Study B), and one gc_run pass on stubbed data."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_frenzy_exits as X  # noqa: E402

T0 = 1_790_000_000_000 // 300_000 * 300_000


def test_selftest_passes():
    X.selftest()


def test_wide_code_mirrors_engine_order():
    # frenzy_long_status judges the ATR cap BEFORE the candle: ATR > cap is ATR_HIGH in the journal whatever the colour → BOTH when green
    assert X.wide_code(2.3775, 0.519, 2.5) == "GREEN_BAR"          # RLC 10-05 12:00 (the live stamps)
    assert X.wide_code(6.627, -1.759, 2.5) == "ATR_HIGH"           # AIN 10-03 23:25
    assert X.wide_code(2.6, 0.1, 2.5) == "BOTH"
    assert X.wide_code(2.4, 0.0, 2.5) == "NONE"                    # flat = red (bar_red is close ≤ open)


def test_walk_ticks_matches_1m_walk_on_flat_minutes():
    # one print per minute == a 1m bar with o = h = l = c → the tick walker and the 1m walker agree (no slippage)
    ps = [100, 101, 103.5, 106, 105, 103.0, 102.0]
    tt = np.array([T0 + i * X.MIN for i in range(len(ps))]); pp = np.array(ps, float)
    pt = X.walk_ticks(tt, pp, 100, T0)
    pm = X.walk([[T0 + i * X.MIN, p, p, p, p] for i, p in enumerate(ps)], 100, "LOCK2")
    assert pt[2] == pm[2] == "floor / trail"
    assert abs(pt[0] - pm[0]) < 1e-9


def test_entry_12s_and_slippage():
    # prints before close + 12 s are never the entry; slippage 0.10 is charged on top of the 0.09 fees, once
    tt = np.array([T0 + 500, T0 + 11_000, T0 + 12_400, T0 + 30_000, T0 + 90_000])
    pp = np.array([90.0, 95.0, 100.0, 96.0, 99.0])
    e, te = X.gc_entry(tt, pp, T0)
    assert (e, te) == (100.0, T0 + 12_400)
    pnl, _, how = X.walk_ticks(tt, pp, e, te, slip=X.SLIP)
    assert how == "stop" and abs(pnl - ((96.0 / 100 - 1) * 100 - X.FEE - X.SLIP)) < 1e-9
    assert (X.ENTRY_LAG_MS, X.SLIP, X.FEE) == (12_000, 0.10, 0.09)


def test_formal_bar_constants():
    # frozen from FRENZY_GREEN_AND_WIDE_ATR_FORMAL_2026-10-06.md (end of Study B) — never re-fit
    assert X.GC_STREAK == 12 and X.GC_FROM == "2026-10-07T00:00:00"
    assert (X.V2_N, X.V2_DAYS, X.V2_MEAN, X.V2_WR, X.V2_PAIR) == (30, 15, 0.30, 55.0, 25.0)
    assert (X.V2_HAIRCUT, X.V2_HAIRCUT_MIN, X.V2_REVERT_N) == (0.50, 0.15, 20)
    assert (X.W_ATR, X.W_N, X.W_WR, X.W_MEAN, X.W_PAIR, X.W_REVERT_N) == (1.5, 30, 70.0, 0.50, 25.0, 15)
    assert X.GC_RETIRE_N == 60   # addition beyond the pre-registration (labelled in the output)


def test_gc_run_counts_v2_after_v1_in_same_spike(tmp_path, monkeypatch):
    """one spike on XUSDT: a V1 green refusal (streak 12) at 01:00, then a V2 one (streak 20) at 03:00 → both counted, each first in its arm;
    a pre-floor V2 refusal of the same spike never hides them."""
    monkeypatch.setattr(X, "GC_CSV", str(tmp_path / "gc.csv"))
    monkeypatch.setattr(X, "JR_CSV", str(tmp_path / "jr.csv"))
    day0 = int(pd.Timestamp("2026-10-07", tz="UTC").value // 1_000_000)
    pre, s1, s2 = day0 - 2 * X.H, day0 + 1 * X.H, day0 + 3 * X.H          # signal closes
    streak = {pre: 30, s1: 12, s2: 20}
    spike = day0 - 5 * X.H

    def kl(sym, tf, start, end):
        step = {"1m": X.MIN, "5m": X.BAR, "1h": X.H}[tf]
        n = int((end - start) // step) + 1
        return [[start + i * step, 100.0, 100.5, 99.5, 100.0, 1000.0] for i in range(min(n, 1500))]

    def ticks(pair, t0, t1, now_ms, budget):
        tt = np.array([t0 + 13_000, t0 + 60_000, t0 + 120_000]); pp = np.array([100.0, 104.0, 101.5])   # +3 arms, back to the +2 floor
        return "ok", tt, pp

    monkeypatch.setattr(X, "_kl", kl)
    monkeypatch.setattr(X, "_ticks", ticks)
    monkeypatch.setattr(X, "normal_hour_usd", lambda h1, ms: 1.0)
    monkeypatch.setattr(X, "frenzy_walk", lambda bars, nh, th: dict(
        spike_ts=spike, hours=(bars[-1][0] + X.BAR - spike) / X.H, above_streak=streak[bars[-1][0] + X.BAR], above_share=80.0,
        vs_vwap_pct=2.0, vol_mult=200.0, bar_ret_pct=0.4, verified=True, in_state=True, fresh_on=True))
    monkeypatch.setattr(X, "frenzy_flagged", lambda ep, th: True)
    monkeypatch.setattr(X, "frenzy_long_status", lambda ep, atr, v, th: (False, "FRENZY_GREEN_BAR", "green"))
    J = pd.DataFrame([(X._iso(t), "BLOCK", "XUSDT", "FRENZY_GREEN_BAR", "") for t in (pre, s1, s2)]
                     + [(X._iso(t), "BLOCK", "XUSDT", "FRENZY_WIDE_CHOPPY", "") for t in (pre, s1, s2)],
                     columns=["t", "e", "pair", "gate", "strategy"])
    F = pd.DataFrame(columns=["opened_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "closed_at", "k"])
    th = SimpleNamespace(frenzy_max_slots=2, frenzy_max_entries_per_pair_day=3, frenzy_long_lev_mult=0.32, frenzy_wide_lev_mult=0.2)
    L = X.gc_run(day0 + 30 * 86_400_000, th, F, J, pd.DataFrame())
    g = pd.read_csv(X.GC_CSV)
    assert len(g) == 3 and g.final.all() and (g.px_src == "tick").all() and (g.gvol == "pass").all()
    by = g.set_index("k")
    assert not by.loc[X._iso(pre), "counted"]                           # reference row
    assert by.loc[X._iso(s1), "counted"] and not by.loc[X._iso(s1), "v2"]
    assert by.loc[X._iso(s2), "counted"] and by.loc[X._iso(s2), "v2"]   # V2 still counted after the V1 refusal of the same spike
    assert abs(by.loc[X._iso(s2), "LOCK"] - (1.5 - X.FEE - X.SLIP)) < 1e-9
    txt = "\n".join(L)
    assert "V2 bar (FORMAL bar 1" in txt and "1/30 signals" in txt and "V2 ∧ ATR ≤ 1.5" in txt


def test_aggtrades_parse_and_corrupt_cache(tmp_path, monkeypatch):
    raw_h = b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n1,0.4649,10,1,1,1790000000123,true\n2,0.4650,5,2,2,1790000000456,false\n"
    raw_n = b"1,0.4649,10,1,1,1790000000123,true\n"
    t, p = X._parse_aggtrades(raw_h)
    assert list(t) == [1790000000123, 1790000000456] and abs(p[1] - 0.4650) < 1e-12
    t, p = X._parse_aggtrades(raw_n)
    assert list(t) == [1790000000123] and t.dtype == np.int64
    monkeypatch.setattr(X, "TICK_CACHE", str(tmp_path))
    bad = tmp_path / "ticks" / "XUSDT" / "2026-10-05.npz"
    bad.parent.mkdir(parents=True); bad.write_bytes(b"not a zip")
    st, _, _ = X._tick_day("XUSDT", "2026-10-05", 1_900_000_000_000, {"dl": 0})
    assert st == "pending" and not bad.exists()                          # corrupt own-cache file removed, re-fetched later
    good = tmp_path / "ticks" / "XUSDT" / "2026-10-06.npz"
    np.savez_compressed(good, t=np.array([1, 2], dtype=np.int64), p=np.array([0.1, 0.2], dtype=np.float32))
    st, t, p = X._tick_day("XUSDT", "2026-10-06", 1_900_000_000_000, {"dl": 0})
    assert st == "ok" and p.dtype == np.float64 and p[0] == np.float64(np.float32(0.1))
