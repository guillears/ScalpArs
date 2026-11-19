"""ROUTE A FOLLOW-UP — shared helpers (weighted stats, day-block bootstrap, Wilson / Kelly). Research only."""
import numpy as np, pandas as pd

DAY = 86_400_000
RNG = np.random.default_rng(20261119)


def wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float); ok = np.isfinite(x)
    return np.average(x[ok], weights=w[ok]) if ok.any() and w[ok].sum() > 0 else np.nan


def day_boot(x, w, day, n=1000, rng=RNG):
    """day-block bootstrap of the weighted mean → (lo, hi)"""
    x = np.asarray(x, float); w = np.asarray(w, float); ok = np.isfinite(x)
    x, w, day = x[ok], w[ok], np.asarray(day)[ok]
    if len(x) < 3:
        return np.nan, np.nan
    u, inv = np.unique(day, return_inverse=True)
    sx = np.bincount(inv, x * w); sw = np.bincount(inv, w)
    pk = rng.integers(0, len(u), (n, len(u)))
    bs = sx[pk].sum(1) / sw[pk].sum(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def day_boot_diff(xa, wa, da, xb, wb, db, n=1000, rng=RNG):
    """bootstrap over the union of days of mean(a) − mean(b) (both groups resampled with the same day draw)"""
    xa, wa, xb, wb = (np.asarray(v, float) for v in (xa, wa, xb, wb))
    da, db = np.asarray(da), np.asarray(db)
    u = np.unique(np.r_[da, db]); ia = np.searchsorted(u, da); ib = np.searchsorted(u, db)
    sxa = np.bincount(ia, xa * wa, len(u)); swa = np.bincount(ia, wa, len(u))
    sxb = np.bincount(ib, xb * wb, len(u)); swb = np.bincount(ib, wb, len(u))
    pk = rng.integers(0, len(u), (n, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sxa[pk].sum(1) / swa[pk].sum(1) - sxb[pk].sum(1) / swb[pk].sum(1)
    d = d[np.isfinite(d)]
    return np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()


def wauc(s, y, w=None):
    """weighted AUC (probability a positive outranks a negative, ties half)"""
    s = np.asarray(s, float); y = np.asarray(y, int); w = np.ones(len(s)) if w is None else np.asarray(w, float)
    ok = np.isfinite(s); s, y, w = s[ok], y[ok], w[ok]
    if y.min() == y.max():
        return np.nan
    o = np.argsort(s, kind="mergesort"); s, y, w = s[o], y[o], w[o]
    _, start = np.unique(s, return_index=True); grp = np.repeat(np.arange(len(start)), np.diff(np.r_[start, len(s)]))
    wp = np.bincount(grp, w * (y == 1)); wn = np.bincount(grp, w * (y == 0))
    cn = np.cumsum(wn) - wn
    return float(((wp * (cn + 0.5 * wn)).sum()) / (wp.sum() * wn.sum()))


def wquantile(x, w, q):
    x = np.asarray(x, float); w = np.asarray(w, float); ok = np.isfinite(x); x, w = x[ok], w[ok]
    o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
    return float(x[o][np.searchsorted(cw, q)])


def wilson_lo(p, n, z=1.645):
    if n <= 0:
        return np.nan
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def kelly_dcr(x, w, tpd):
    """Kelly notional fraction at the one-sided 95 % Wilson WR floor (Kish effective N) and the implied daily compound"""
    x = np.asarray(x, float); w = np.asarray(w, float); ok = np.isfinite(x); x, w = x[ok], w[ok]
    if len(x) < 5:
        return dict(wr=np.nan, wr_floor=np.nan, avg_win=np.nan, avg_loss=np.nan, be_wr=np.nan, kelly=0.0, dcr=0.0)
    win = x > 0
    p = np.average(win, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    W = np.average(x[win], weights=w[win]) if win.any() else 0.0
    L = -np.average(x[~win], weights=w[~win]) if (~win).any() else 1e-9
    pf = wilson_lo(p, neff)
    f = (pf / (L / 100) - (1 - pf) / (W / 100)) if W > 0 and L > 0 else 0.0
    f = max(0.0, f)
    avg = np.average(x, weights=w)
    dcr = ((1 + f * avg / 100) ** tpd - 1) * 100 if f > 0 else 0.0
    return dict(wr=p * 100, wr_floor=pf * 100, avg_win=W, avg_loss=-L, be_wr=L / (W + L) * 100 if W + L > 0 else np.nan,
                kelly=f, dcr=dcr)


def loo(x, w, key):
    x = np.asarray(x, float); w = np.asarray(w, float); key = np.asarray(key)
    u = np.unique(key)
    if len(u) < 2:
        return np.nan
    return min(wmean(x[key != k], w[key != k]) for k in u)


def best5_out(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float); ok = np.isfinite(x); x, w = x[ok], w[ok]
    o = np.argsort(x)[::-1]; cut = int(np.ceil(len(x) * 0.05)); k = o[cut:]
    return wmean(x[k], w[k]) if len(k) else np.nan


def half_of(ms):
    dt = pd.to_datetime(pd.Series(ms), unit="ms")
    return (dt.dt.year.astype(str).str[2:] + np.where(dt.dt.month <= 6, "-H1", "-H2")).values
