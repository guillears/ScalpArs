#!/usr/bin/env python3
"""HT-only "hold at the TP" trigger screen (operator 2026-11-24: "i didn't say that the trigger to hold should be the same for each
sleeve"; research only). Cohort: the 118 HT TP hits of the today's-filters backtest (reports/TP_HOLD_2026-11-24_data/bt_tp_hits.csv,
features at the touch from 1m klines — speed, volume, taker share, BTC, ATR, theory: room to base, unwound share, close location,
absorbing wick, acceleration, OI, top-trader position) joined to HT's own runner outcome (dive_HT.csv: 4 × 15m ATR trail, floor +5, 24 h,
funding incl.). Outcomes: Δ vs TP +6 and RAN (the runner left the floor, why = trail / time). Median split per feature; family-wise null =
5,000 shuffles (max |t| over every feature × both outcomes); stability = the same sign in the older and newer half (time median)."""
import os
import numpy as np
import pandas as pd
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "TP_HOLD_2026-11-24_data")
rng = np.random.default_rng(6)


def t_(a, b):
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else 0.0


def split(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30 or np.nanstd(x[ok]) == 0:
        return None
    m = np.median(x[ok]); hi, lo = ok & (x > m), ok & (x <= m)
    if hi.sum() < 10 or lo.sum() < 10:
        return None
    return y[hi].mean(), y[lo].mean(), t_(y[hi], y[lo]), int(ok.sum())


B = pd.read_csv(os.path.join(D, "bt_tp_hits.csv")); B = B[B.sleeve == "FRENZY_WILLY_HT"]
O = pd.read_csv(os.path.join(D, "dive_HT.csv"))[["pair", "t", "4x|+5", "4x|+5_why"]].rename(columns={"t": "t_touch", "4x|+5": "delta", "4x|+5_why": "why"})
X = B.merge(O, on=["pair", "t_touch"], how="inner")
X["ran"] = (X.why != "floor").astype(float)
NOT = {"sleeve", "pair", "t_touch", "period", "E", "tp_px", "dirn", "tp_pct", "bars_after", "best_after", "delta", "why", "ran"}
F = [c for c in X.columns if c not in NOT and not c.startswith(("d1", "d2", "d3", "d5", "f0", "f1", "f2", "a1", "a2", "a3", "af", "m_", "c_"))
     and X[c].dtype != object]
print(f"HT TP hits joined: {len(X)} · ran {X.ran.mean():.0%} · mean Δ {X.delta.mean():+.2f} · features {len(F)}")
cov = {f: int(np.isfinite(X[f]).sum()) for f in F}
print("coverage < 118:", {k: v for k, v in cov.items() if v < len(X)})
older = (X.t_touch <= X.t_touch.median()).values
rows = []
for yc in ("delta", "ran"):
    y = X[yc].values
    for f in F:
        r = split(X[f].values, y)
        if r:
            a = split(X[f].values[older], y[older]); b = split(X[f].values[~older], y[~older])
            rows.append(dict(outcome=yc, feature=f, N=r[3], hi=r[0], lo=r[1], t=r[2],
                             older=(a[0] - a[1]) if a else np.nan, newer=(b[0] - b[1]) if b else np.nan))
R = pd.DataFrame(rows)
null = []
for _ in range(5000):
    p = rng.permutation(len(X)); best = 0.0
    for yc in ("delta", "ran"):
        y = X[yc].values[p]
        for f in R[R.outcome == yc].feature:
            r = split(X[f].values, y)
            if r: best = max(best, abs(r[2]))
    null.append(best)
null = np.array(null)
R["fw_p"] = [(null >= abs(t)).mean() for t in R.t]
R["stable"] = np.where(R.older.notna() & R.newer.notna(), np.sign(R.older) == np.sign(R.newer), np.nan)
R = R.sort_values("t", key=abs, ascending=False)
R.to_csv(os.path.join(os.path.dirname(D), "TP_RUNNER_HT_TRIGGER_2026-11-24_screen.csv"), index=False)
print(f"family-wise 95 % |t| {np.quantile(null, .95):.2f} over {len(R)} tests")
print(R.head(12).round(3).to_string(index=False))

# the top variable by calendar half-year (the numbers DL 638 quotes)
top = "vol5_60"; cut = X[top].median(); X["half"] = pd.to_datetime(X.t_touch, unit="ms").dt.year.astype(str) + "-H" + (
    (pd.to_datetime(X.t_touch, unit="ms").dt.month > 6) + 1).astype(str)
X["grp"] = np.where(X[top] > cut, "surge", "quiet")
print(f"\n{top} median {cut:.5f}")
for g, x in X.groupby("grp"):
    s_ = x.delta.sort_values(ascending=False)
    print(f"  {g}: N {len(x)} · ran {int(x.ran.sum())} · mean Δ {x.delta.mean():+.2f} · w/o top 3 {s_.iloc[3:].mean():+.2f} · by half "
          f"{x.groupby('half').delta.mean().round(2).to_dict()}")
