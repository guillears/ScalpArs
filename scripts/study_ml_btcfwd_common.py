"""ML BTC-forward follow-up (2026-10-15) — shared stats + gate helpers (pre-registration reports/ML_BTCFWD_PREREG.txt)."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
MIN1, MIN5, DAY = 60_000, 300_000, 86_400_000


def day_boot(y, day, n=2000, seed=7):
    """day-clustered bootstrap of the mean -> (mean, lo95, hi95, P(mean > 0))."""
    y = np.asarray(y, float); day = np.asarray(day)
    if len(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=y); c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(y.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean())


def lomo_min(delta, month):
    """min over calendar months of mean(delta without that month) -> sign-stable iff > 0."""
    delta = np.asarray(delta, float); month = np.asarray(month)
    vals = []
    for m in np.unique(month):
        k = month != m
        if k.sum():
            vals.append(delta[k].mean())
    return float(np.min(vals)) if vals else np.nan


def max_share(delta, key):
    """largest single key's share of the total positive sum delta (nan when the total <= 0)."""
    s = pd.Series(np.asarray(delta, float)).groupby(np.asarray(key)).sum()
    tot = s.sum()
    if tot <= 0:
        return np.nan
    return float(s.max() / tot)


def breakeven_wr(y):
    y = np.asarray(y, float)
    w, l = y[y > 0], y[y <= 0]
    if len(w) == 0 or len(l) == 0:
        return np.nan
    return abs(l.mean()) / (w.mean() + abs(l.mean()))


def fmt(x, p=3):
    try:
        return "—" if x is None or not np.isfinite(x) else f"{x:+.{p}f}"
    except TypeError:
        return str(x)


def year_of(block):
    return {"2025 BULL": "2025", "2026-H1 BEAR": "2026", "2026-H2 BULL": "2026"}.get(block, block)


def regime_days():
    """day-level regime variables (known at the day's 00:00 UTC, no lookahead) from BTC 1m closes (fetched taker file,
    falling back to the caches): r24 sign, r7d sign, close vs 30-day mean sign, 24 h realized vol above the full median."""
    fp = os.path.join(REP, "ml_btcfwd_fetch/k1m_taker/BTCUSDT.csv")
    b = pd.read_csv(fp, usecols=["open_time", "c"]).drop_duplicates("open_time").sort_values("open_time")
    b["t"] = pd.to_datetime(b.open_time, unit="ms")
    s = b.set_index("t").c.astype(float)
    h = s.resample("1h").last()
    d0 = h.index.floor("D").unique()
    out = []
    lr = np.log(s).diff()
    rv = lr.resample("1D").std()
    for d in d0:
        prev = h[h.index < d]
        if len(prev) < 24 * 31:
            continue
        c = prev.iloc[-1]
        out.append(dict(day=d, r24=c / prev.iloc[-25] - 1, r7d=c / prev.iloc[-24 * 7 - 1] - 1,
                        gap30=c / prev.iloc[-24 * 30:].mean() - 1,
                        rv24=rv.get(d - pd.Timedelta(days=1), np.nan)))
    R = pd.DataFrame(out)
    R["rv_hi"] = (R.rv24 > R.rv24.median()).astype(float)
    for c in ("r24", "r7d", "gap30"):
        R[c + "_pos"] = (R[c] > 0).astype(float)
    return R.set_index("day")


def regime_split(delta, day, R, n=500, seed=5):
    """DAY units: per-day mean delta, split by each sign variable; diff of means vs a day-shuffled null."""
    df = pd.DataFrame(dict(d=np.asarray(delta, float), day=pd.to_datetime(np.asarray(day)).floor("D")))
    g = df.groupby("day").d.mean()
    g = g[g.index.isin(R.index)]
    rng = np.random.default_rng(seed)
    rows = []
    for v in ("r24_pos", "r7d_pos", "gap30_pos", "rv_hi"):
        x = R.loc[g.index, v].values
        a, b = g.values[x == 1], g.values[x == 0]
        if len(a) < 5 or len(b) < 5:
            continue
        obs = a.mean() - b.mean()
        null = []
        for _ in range(n):
            xs = rng.permutation(x)
            null.append(g.values[xs == 1].mean() - g.values[xs == 0].mean())
        null = np.abs(np.array(null))
        rows.append(dict(var=v, days_1=len(a), days_0=len(b), mean_1=a.mean(), mean_0=b.mean(), diff=obs,
                         p_null=float((null >= abs(obs)).mean())))
    return pd.DataFrame(rows)
