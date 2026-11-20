"""TP_GRID FB (2026-11-20) — risk-matched read (same method as TP_GRID_2026-11-20_DIST_risk.py): each TP / SL cell is sized so its
backtest max drawdown is −30 % (and −20 %), then its daily compound return at that size is compared with today's −5 / +10 cell;
10-day-block bootstrap P(better), paired by day. Cohort = today's stack (NEAR_BASE + DL 601 funding filter; unread funding is kept, like the live fail-open), 72 h hold.
Needs reports/TP_GRID_2026-11-20_FB_data/bt_grid.pkl (git-ignored; rebuilt by scripts/TP_GRID_2026-11-20_FB_backtest.py)."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = pd.read_pickle(os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_FB_data", "bt_grid.pkl"))
G = G[~(G.funding_rate_pct > 0.005)].copy()
CELLS = [(5, 10), (6, 10), (7, 10), (8, 10), (5, 12), (6, 12), (7, 12), (8, 12), (5, 8), (3, 10)]

def daily(c, sub):
    x = pd.to_datetime(sub[f"primary_{c[0]}_{c[1]}_72_x"], unit="ms").dt.floor("D")
    s = pd.Series(sub[f"primary_{c[0]}_{c[1]}_72"].values).groupby(x.values).sum()
    d0 = pd.to_datetime(sub.cf_entry_t, unit="ms").min().floor("D")
    d1 = max(pd.to_datetime(sub[f"primary_{k[0]}_{k[1]}_72_x"], unit="ms").max() for k in CELLS).floor("D")
    return s.reindex(pd.date_range(d0, d1, freq="D"), fill_value=0.0).values

def stats(r, s):
    g = np.maximum(1 + r * s / 100, 1e-9); eq = np.cumprod(g)
    return (eq[-1] ** (1 / len(g)) - 1) * 100, (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() * 100, (g.min() - 1) * 100

def size_for_dd(r, tgt):
    lo, hi = 0.0, 50.0   # wide bracket: a resample without the cluster days may need a large size to reach the target
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if stats(r, m)[1] > tgt else (lo, m)
    return lo

rng = np.random.default_rng(13); rows = []
for blk in ("ALL", "HOLDOUT"):
    sub = G if blk == "ALL" else G[pd.to_datetime(G.cf_entry_t, unit="ms") >= pd.Timestamp("2025-07-01")]
    R = {c: daily(c, sub) for c in CELLS}
    for c, r in R.items():
        d061, dd061, w061 = stats(r, 0.61)
        row = dict(block=blk, TP=c[0], SL=c[1], N=len(sub), dcr_061=d061, dd_061=dd061, worst_061=w061)
        for tgt in (-20, -30):
            s = size_for_dd(r, tgt); d, _, w = stats(r, s)
            row[f"size_dd{-tgt}"] = s; row[f"dcr_dd{-tgt}"] = d; row[f"worst_dd{-tgt}"] = w
        rows.append(row)
    n = len(R[(5, 10)]); nb = n // 10; res = {c: [] for c in CELLS}
    for _ in range(500):
        st = rng.integers(0, n - 9, nb); idx = (st[:, None] + np.arange(10)).ravel()
        for c, r in R.items():
            rr = r[idx]; res[c].append(stats(rr, size_for_dd(rr, -30))[0])
    for row in rows:
        if row["block"] == blk:
            dl = np.array(res[(row["TP"], row["SL"])]) - np.array(res[(5, 10)])
            row["p_better_eqdd30"] = (dl > 0).mean(); row["gain_med"] = np.median(dl)
            row["gain_lo90"] = np.quantile(dl, .05); row["gain_hi90"] = np.quantile(dl, .95)
O = pd.DataFrame(rows).round(3)
O.to_csv(os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_FB_data", "risk_matched.csv"), index=False)
pd.set_option("display.width", 250); print(O.to_string(index=False))
