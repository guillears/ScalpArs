#!/usr/bin/env python3
"""🔥 "Don't chase" with ORDER FLOW — does the pattern in the operator's 40 MOVR trades hold across the year?

Source of the hypothesis (2026-10-01, MOVR, AFTER the cache period → this whole dataset is out-of-sample for it): entries made
after a 15-second pause won 86–88 %, entries made on a burst (price up AND aggressive buying) won 45 %; on the MOVR tape the
same split gave 78 % vs 68 % target-first for a mechanical entry.

PRE-DECLARED (written before any flow result was read):
  entries   LONG at HOT seconds 30 s apart in every cached episode (as hot_entry_separator_screen.py)
  flow      taker-buy imbalance over the last N seconds = (2·taker-buy − volume) ÷ volume × 100, from 1-second SPOT klines
  classes   CHASE = last-15 s price up ∧ flow15 ≥ +10        PAUSE = last-15 s price flat/down ∧ flow15 < 0
            MILD  = flow15 in [−10, 0) (any price move)       REST  = everything else
  exits     +0.59 / −1.11 (the operator's) and +1.09 / −1.11 · 30-min limit · costs 0.09 % fees + 0.02 % slippage
  tests     H1 direction: target-first(PAUSE) > target-first(CHASE) in BOTH halves (Jan–Apr, May–Sep), by episode
            H2 tradable:  PAUSE net per trade > 0 with its 95 % day interval above 0
            H3 tradable:  MILD  net per trade > 0 with its 95 % day interval above 0
  screen    flow_5s / 15s / 60s / 300s, volume speed (last 15 s ÷ its 5-min pace), trades per second: quintile edges on Jan–Apr,
            target-first rate per quintile in both halves (same protocol as the price screen)
  units     episode means → UTC-day means → t-interval (EPISODE-weighted; "target first" = result > 0, so a positive time-out counts;
            MILD overlaps the other classes; entries in the first 5 minutes of a window are not scored)
Usage: venv/bin/python scripts/hot_flow_test.py → reports/HOT_FLOW_TEST_2026-10-01.md"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import hot_scalp_backtest as H  # noqa: E402
sys.argv = _a
FLOW = os.path.join(H.ROOT, "reports", "backtest_cache", "k1s_hot_flow")
OUT = os.path.join(H.ROOT, "reports", "HOT_FLOW_TEST_2026-10-01.md"); COST = 0.11


def rows_for(pair, d5, f):
    ff = os.path.join(FLOW, os.path.basename(f))
    if not os.path.exists(ff):
        return []
    z = np.load(f); y = np.load(ff); ts, h, l, c = z["t"], z["h"], z["l"], z["c"]; q, tb, n = y["q"], y["tb"], y["n"]
    if len(q) != len(c) or np.isnan(q).mean() > 0.02:
        return []
    q = np.nan_to_num(q); tb = np.nan_to_num(tb); n = np.clip(np.nan_to_num(n), 0, None)
    st = d5.reindex(ts // H.BAR * H.BAR); atr, rsi, ema5, q24 = st.atr.values, st.rsi.values, st.ema5.values, st.q24.values
    hot = (atr >= 2) & (rsi >= 70) & (c >= ema5 * 1.03) & (q24 >= 20e6); hi = np.nonzero(hot)[0]
    cq, cb, cn = np.cumsum(q), np.cumsum(tb), np.cumsum(n); out = []; last = -10**9

    def fl(i, k):
        v = cq[i] - cq[i - k]; return (2 * (cb[i] - cb[i - k]) - v) / v * 100 if v > 0 else 0.0
    for i in hi:
        if i - last < 30 or i < 305 or i > len(c) - 120:
            continue
        last = i; e = c[i]
        a, _, _ = H.walk(h, l, c, int(i), (0.59, 1.11, None), e); b, _, _ = H.walk(h, l, c, int(i), (1.09, 1.11, None), e)
        v15, v300 = cq[i] - cq[i - 15], (cq[i] - cq[i - 300]) / 20
        out.append(dict(pair=pair, eid=os.path.basename(f)[:-4], t=int(ts[i]), a=a, b=b, win=bool(a > 0), r15=(e / c[i - 15] - 1) * 100,
                        flow_5s=fl(i, 5), flow_15s=fl(i, 15), flow_60s=fl(i, 60), flow_300s=fl(i, 300),
                        vol_speed=v15 / v300 if v300 > 0 else np.nan, trades_per_s=(cn[i] - cn[i - 15]) / 15, usd_15s=v15))
    return out


def dayline(g, k):
    if g.eid.nunique() < 20:
        return "too few"
    e = g.groupby(["day", "eid"])[k].mean() - COST; dm = e.groupby(level=0).mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n); q = H.tq(0.975, n)
    return f"{dm.mean():+.3f} [{dm.mean() - q * se:+.3f}, {dm.mean() + q * se:+.3f}]"


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(H.RAW, "*.npz"))); R = []; cache = {}
    for n_, f in enumerate(files):
        pair = os.path.basename(f)[:-4].rsplit("_", 1)[0]
        if pair not in cache:
            cache = {pair: H.frame5(pair)[0]}
        R += rows_for(pair, cache[pair], f)
    T = pd.DataFrame(R).replace([np.inf, -np.inf], np.nan); T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d")
    T.to_csv(os.path.join(H.ROOT, "reports", "backtest_cache", "hot_flow_entries.csv"), index=False)
    cls = np.where((T.r15 > 0) & (T.flow_15s >= 10), "CHASE", np.where((T.r15 <= 0) & (T.flow_15s < 0), "PAUSE", "REST")); T["cls"] = cls
    A, B = T[T.day < H.SPLIT], T[T.day >= H.SPLIT]; er = lambda g: g.groupby("eid").win.mean().mean() * 100 if len(g) else np.nan
    L = ["# 🔥 Don't chase — the order-flow test across the year", "",
         f"{len(T):,} long entries (30 s apart) in {T.eid.nunique():,} episodes · {T.pair.nunique()} pairs · {T.day.nunique()} days, with per-second taker-buy volume. "
         f"Target-first rate (episode-weighted): Jan–Apr {er(A):.1f} % · May–Sep {er(B):.1f} % · needs 72 % to pay.", "",
         "| Class (frozen from the MOVR session) | entries | episodes | target first Jan–Apr | May–Sep | net % / trade +0.59 / −1.11, by day [95 %] | +1.09 / −1.11 |", "|---|---|---|---|---|---|---|"]
    groups = [("CHASE (price up ∧ flow ≥ +10)", T.cls == "CHASE"), ("PAUSE (price flat/down ∧ flow < 0)", T.cls == "PAUSE"),
              ("MILD (flow in [−10, 0))", (T.flow_15s >= -10) & (T.flow_15s < 0)), ("REST", T.cls == "REST"), ("ALL", T.a.notna())]
    for nm, m in groups:
        g = T[m]; L.append(f"| {nm} | {len(g):,} | {g.eid.nunique()} | {er(g[g.day < H.SPLIT]):.1f}% | {er(g[g.day >= H.SPLIT]):.1f}% | {dayline(g, 'a')} | {dayline(g, 'b')} |")
    pa, ch = T[T.cls == "PAUSE"], T[T.cls == "CHASE"]
    h1 = [er(pa[pa.day < H.SPLIT]) - er(ch[ch.day < H.SPLIT]), er(pa[pa.day >= H.SPLIT]) - er(ch[ch.day >= H.SPLIT])]
    L += ["", f"**H1 (direction): PAUSE − CHASE target-first = {h1[0]:+.1f} points Jan–Apr · {h1[1]:+.1f} May–Sep → {'HOLDS in both halves' if min(h1) > 0 else 'does NOT hold in both halves'}.**",
          "H2 / H3 (tradable): read the PAUSE and MILD rows — positive only if the bracket is entirely above 0.", "",
          "## Flow features by quintile (edges from Jan–Apr)", "", "| Feature | target-first % by quintile, Jan–Apr (low → high) | May–Sep |", "|---|---|---|"]
    for f in ("flow_5s", "flow_15s", "flow_60s", "flow_300s", "vol_speed", "trades_per_s", "usd_15s", "r15"):
        ed = np.nanquantile(A[f], [.2, .4, .6, .8])
        ra = [er(A[np.searchsorted(ed, A[f].values, side="right") == k]) for k in range(5)]; rb = [er(B[np.searchsorted(ed, B[f].values, side="right") == k]) for k in range(5)]
        L.append(f"| {f} | " + " · ".join(f"{x:.0f}" for x in ra) + " | " + " · ".join(f"{x:.0f}" for x in rb) + " |")
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
