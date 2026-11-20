"""TP_GRID_2026-11-20_FB — shared grid / statistics helpers (real-fill and backtest scripts). Research only."""
import numpy as np
import pandas as pd

TPS = [3, 4, 5, 6, 7, 8, 10]
SLS = [4, 5, 6, 7, 8, 10, 12]
TODAY = (5, 10, 72)
X_SIZE = 0.61          # today's FLATBASE ticket ≈ 0.61 × equity per trade (one size, DL 503)
CELLS = [(tp, sl) for tp in TPS for sl in SLS]


def first_hits(h, l, S, T):
    """first bar index with high ≥ S and with low ≤ T (len(h) if never)."""
    a = h >= S
    b = l <= T
    ks = int(np.argmax(a)) if a.any() else len(h)
    kt = int(np.argmax(b)) if b.any() else len(h)
    return ks, kt


def day_stats(pct, close_day, d0, d1, x=X_SIZE):
    """one-size daily compound return over EVERY calendar day d0..d1 (0 on no-fill days) + max drawdown + worst day (all in %)."""
    s = pd.Series(np.asarray(pct, float) * x / 100).groupby(pd.to_datetime(pd.Series(close_day)).values).sum()
    days = pd.date_range(d0, d1, freq="D")
    r = s.reindex(days, fill_value=0.0).values
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    return (eq[-1] ** (1 / len(r)) - 1) * 100, min(dd, 0) * 100, r.min() * 100


def breakeven_wr(p):
    p = np.asarray(p, float)
    w, lo = p[p > 0], -p[p <= 0]
    if not len(w) or not len(lo):
        return np.nan
    return lo.mean() / (w.mean() + lo.mean()) * 100


def cell_row(pct, close_day, d0, d1):
    pct = np.asarray(pct, float)
    dcr, mdd, worst = day_stats(pct, close_day, d0, d1)
    return dict(N=len(pct), WR=(pct > 0).mean() * 100 if len(pct) else np.nan, avg=pct.mean() if len(pct) else np.nan, sum=pct.sum(),
                DCR=dcr, maxDD=mdd, worst_day=worst, BE_WR=breakeven_wr(pct))


def cluster_boot(delta, cluster, B=4000, seed=7):
    """cluster (day / window) bootstrap of the mean paired Δ → (mean, lo95, hi95, P(mean > 0))."""
    d = np.asarray(delta, float)
    cl = np.asarray(cluster)
    u, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, d, len(u)); ns = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(u), (B, len(u)))
    m = sums[ix].sum(1) / ns[ix].sum(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


def fw_null(D, cluster, B=4000, seed=11):
    """family-wise (pick-the-best) null over the grid: random sign per cluster applied to every cell's paired Δ at once (keeps the
    cross-cell correlation) → the distribution of the BEST cell's mean Δ under 'no cell differs from today'.
    D = n × cells. → (null max distribution, p_fw per cell = P(null max ≥ observed mean Δ))."""
    D = np.asarray(D, float)
    cl = np.asarray(cluster)
    u, inv = np.unique(cl, return_inverse=True)
    S = np.zeros((len(u), D.shape[1]))
    np.add.at(S, inv, D)
    rng = np.random.default_rng(seed)
    mx = np.empty(B)
    for b in range(B):
        sg = rng.choice([-1.0, 1.0], len(u))
        mx[b] = (sg @ S / len(D)).max()
    obs = D.mean(0)
    return mx, np.array([(mx >= o).mean() for o in obs])


def loo_range(delta):
    """leave-one-out: (min, max) of the mean Δ with each fill removed, and the fill whose removal moves it most."""
    d = np.asarray(delta, float)
    n = len(d)
    if n < 2:
        return np.nan, np.nan, -1
    m = (d.sum() - d) / (n - 1)
    return m.min(), m.max(), int(np.argmin(m))


def drop_best_share(delta, share=0.05):
    d = np.sort(np.asarray(delta, float))
    k = max(1, int(round(len(d) * share)))
    return d[:-k].mean()


def top_share(delta, key):
    """share of the positive Σ Δ carried by the top key (pair / window); NaN if Σ Δ ≤ 0."""
    s = pd.Series(np.asarray(delta, float)).groupby(np.asarray(key)).sum()
    tot = s.sum()
    if tot <= 0:
        return np.nan
    return s.max() / tot * 100


def md(df, fl=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.{fl}f}" if abs(v) < 1000 else f"{v:,.0f}")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def heat(vals, fl=2):
    """TP rows × SL columns markdown heatmap from {(tp, sl): value}; today's cell in bold."""
    out = ["| TP \\ SL | " + " | ".join(f"+{s}" for s in SLS) + " |", "|---|" + "---|" * len(SLS)]
    for tp in TPS:
        cells = []
        for sl in SLS:
            v = vals.get((tp, sl), np.nan)
            t = "" if pd.isna(v) else f"{v:+.{fl}f}" if fl else f"{v:.0f}"
            cells.append(f"**{t}**" if (tp, sl) == TODAY[:2] else t)
        out.append(f"| −{tp} | " + " | ".join(cells) + " |")
    return "\n".join(out)
