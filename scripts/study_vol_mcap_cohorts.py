#!/usr/bin/env python3
"""📊 Oct-8 research (vol24h / mcap study) — build the backtest cohorts (no network).

  FRENZY backtest (engine-parity fresh-ON cohort, reports/FRENZY_ENGINE_COHORT_2026-10-05.csv via the ATR-cap study's
  loader <scratch>/atr_cap/core.py — live-eligible, engine market volume U2 < 1.0, dislocation re-decided at 8 s):
    GATED   every gated signal with a role (red → FRENZY_LONG, green ∧ above_streak > 12 → WIDE), ANY ATR, unsequenced
    TODAY   ATR ≤ 3.0 (live cap), sequenced (2 slots / sleeve, one per pair, 3 per pair-day) on the FIXED +3 / −3 8 s ruler (FIX38)
    TODAY_BEAR  TODAY with the FRENZY_BEARISH_DAY block (BTC last closed UTC daily return < 0 ∧ BTC 5m EMA13−EMA50 gap < 0 at the
                signal bar), re-sequenced
  WILLY A  first-flag events (scratch flag_math/ev2.pkl, 2,320 engine-reachable) → t0 = flag close
  WILLY B  live-eligible priced fresh-ON bars NOT taken by TODAY_BEAR (what entry B would see), t0 = signal close
Writes <scratch>/vm/*.csv. Usage: S=<scratchpad> venv/bin/python scripts/study_vol_mcap_cohorts.py"""
import os, sys
import numpy as np, pandas as pd

S = os.environ["S"]; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(S, "atr_cap")); os.chdir(os.path.join(S, "atr_cap"))
import core  # noqa: E402
OUT = os.path.join(S, "vm"); os.makedirs(OUT, exist_ok=True)


def btc_bear(t_close_ms):
    b = pd.read_csv(os.path.join(ROOT, "reports/backtest_cache/k5m_full/BTCUSDT.csv"), usecols=["open_time", "c"]) \
        .drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    e13 = b.c.ewm(span=13, adjust=False).mean(); e50 = b.c.ewm(span=50, adjust=False).mean(); gap = (e13 - e50) / e50 * 100
    i = np.searchsorted(b.open_time.values, np.asarray(t_close_ms, dtype="int64") - 300_000, side="right") - 1   # signal bar (closed)
    day = pd.to_datetime(b.open_time, unit="ms").dt.floor("D")
    dc = b.groupby(day).c.last()                                     # UTC daily close
    r1d = (dc / dc.shift(1) - 1) * 100
    sig_day = pd.to_datetime(np.asarray(t_close_ms, dtype="int64") - 1, unit="ms").floor("D")
    last_closed = r1d.reindex(sig_day - pd.Timedelta(days=1)).to_numpy()   # the LAST CLOSED daily candle
    return pd.DataFrame(dict(btc_1d_ret=last_closed, btc_gap=gap.values[i]))


d = core.load()
bb = btc_bear(d.t_signal_close.values); d["btc_1d_ret"] = bb.btc_1d_ret.values; d["btc_gap"] = bb.btc_gap.values
d["bear"] = (d.btc_1d_ret < 0) & (d.btc_gap < 0)
G = d[d.gated].assign(sleeve=lambda x: x.role_any)
T = core.sequence(core.admit(d, 3.0), "FIX38")
TB = core.sequence(core.admit(d, 3.0).loc[lambda x: ~x.bear], "FIX38")
keep = ["key", "pair", "t_signal_close", "day", "month", "half", "sleeve", "atr", "bar_ret", "vol24", "U2", "gvol", "above_streak",
        "hours", "run_pct", "gain_pct", "vol_mult", "bear", "btc_1d_ret", "btc_gap", "episode", "FIX38", "FIX38_xt", "PRI"]
G[keep].to_csv(f"{OUT}/frenzy_gated.csv", index=False)
T[keep].to_csv(f"{OUT}/frenzy_today.csv", index=False)
TB[keep].to_csv(f"{OUT}/frenzy_today_bear.csv", index=False)
taken = set(TB.key)
B = d[d.live_elig.astype(bool) & d.outcome.isin(["tick", "disloc_refused", "1m", "1m_csv"]) & ~d.key.isin(taken)]
B[["key", "pair", "t_signal_close", "day", "sleeve", "code", "atr", "bar_ret", "vol24", "U2", "gvol", "bear", "btc_1d_ret", "btc_gap",
   "episode"]].rename(columns={"t_signal_close": "t0"}).to_csv(f"{OUT}/willyB_in.csv", index=False)
A = pd.read_pickle(os.path.join(S, "flag_math/ev2.pkl"))
A = pd.DataFrame(dict(pair=A.pair, t0=A.fc.astype("int64"), spike=A.spike.astype("int64")))
ab = btc_bear(A.t0.values); A["btc_1d_ret"] = ab.btc_1d_ret.values; A["btc_gap"] = ab.btc_gap.values
A["bear"] = (A.btc_1d_ret < 0) & (A.btc_gap < 0); A["day"] = pd.to_datetime(A.t0, unit="ms").dt.strftime("%Y-%m-%d")
A.to_csv(f"{OUT}/willyA_in.csv", index=False)
print(f"GATED {len(G)} · TODAY {len(T)} (avg FIX38 {T.FIX38.mean():+.3f}) · TODAY_BEAR {len(TB)} (avg {TB.FIX38.mean():+.3f}) · "
      f"bear share gated {G.bear.mean() * 100:.1f}% · WILLY B {len(B)} · WILLY A {len(A)}")
