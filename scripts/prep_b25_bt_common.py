#!/usr/bin/env python3
"""B25 batch-review prep — shared stats helpers for the BACKTEST block (block ③). READ-ONLY, no network.

day-clustered bootstrap (resample UTC days, not trades), breakeven WR, loss concentration, summary rows.
"""
import numpy as np
import pandas as pd

NB, SEED = 4000, 7


def day_boot(x, days, n=NB, seed=SEED):
    """→ (lo95, hi95, P(mean < 0)) of the per-fill mean, resampling UTC days."""
    x = np.asarray(x, float); days = np.asarray(days)
    ok = np.isfinite(x); x, days = x[ok], days[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    d = pd.DataFrame({"d": days, "x": x}).groupby("d").x.agg(["sum", "count"])
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(d), (n, len(d)))
    m = d["sum"].values[k].sum(1) / d["count"].values[k].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m < 0).mean())


def breakeven_wr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    w, l = x[x > 0], x[x <= 0]
    if not len(w) or not len(l):
        return np.nan
    return 100 * abs(l.mean()) / (w.mean() + abs(l.mean()))


def loss_share(df, key, col="pct"):
    """largest single-key share of the cohort's loss: −(most negative key Σ) / −(Σ of all negative key sums)."""
    s = df.groupby(key)[col].sum()
    neg = s[s < 0]
    if not len(neg):
        return np.nan, None
    return float(neg.min() / neg.sum()), str(neg.idxmin())


def summ(g, col="pct", usd=None, day="day"):
    """N · WR · avg % · Σ % · $ · CI · P(<0) · days."""
    x = g[col].astype(float)
    lo, hi, pn = day_boot(x.values, g[day].values)
    out = dict(N=len(g), WR=round(100 * (x > 0).mean(), 1) if len(g) else np.nan, avg=round(x.mean(), 4) if len(g) else np.nan,
               sum_pct=round(x.sum(), 2), ci_lo=round(lo, 4) if np.isfinite(lo) else np.nan, ci_hi=round(hi, 4) if np.isfinite(hi) else np.nan,
               p_neg=round(pn, 3) if np.isfinite(pn) else np.nan, days=g[day].nunique())
    if usd is not None:
        for k, c in usd.items():
            out[k] = round(float(g[c].sum()), 0)
    return out


def fmt_row(d, usd_keys=()):
    s = f"{d['N']:>4} · {d['WR']:5.1f} % · {d['avg']:+.3f}"
    if np.isfinite(d.get("ci_lo", np.nan)):
        s += f" [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]"
    for k in usd_keys:
        s += f" · {k} ${d[k]:+,.0f}"
    return s
