"""TP_GRID DIST (2026-11-20) — risk-matched read: which TP earns more at the SAME drawdown?
For each TP, scale the per-trade size s (fraction of equity) until the block's max drawdown equals a target, then read the daily
compound return at that size (r_d = sum net% x s / 100 over fills closed that day, every calendar day, 0 on no-fill days).
Also the worst day at each size."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = pd.read_csv(os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_DIST_data", "bt_rows.csv.gz"))
TPS = [4, 5, 6, 7, 8, 9, 10]

def daily(net, xday, d0, d1):
    s = pd.Series(net).groupby(np.asarray(xday)).sum()
    return s.reindex(range(d0, d1 + 1), fill_value=0.0).values

def stats(r, s):
    g = np.maximum(1 + r * s / 100, 1e-9)
    eq = np.cumprod(g); dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    return dict(dcr=(eq[-1] ** (1 / len(g)) - 1) * 100, dd=dd * 100, worst_day=(g.min() - 1) * 100,
                logg=np.log(g).mean())

def size_for_dd(r, target):
    lo, hi = 0.0, 50.0   # wide bracket: a resample without the cluster days may need a large size to reach the target
    for _ in range(60):
        m = (lo + hi) / 2
        (lo, hi) = (m, hi) if stats(r, m)["dd"] > target else (lo, m)
    return lo

rows = []
for blk in ("ALL", "HOLDOUT"):
    S = B if blk == "ALL" else B[B.block == "HOLDOUT"]
    d0 = int(S.day.min()); d1 = int(max(S[[f"xday_TP{t}_1tick" for t in TPS]].max()))
    for t in TPS:
        r = daily(S[f"TP{t}_1tick"].values, S[f"xday_TP{t}_1tick"].values, d0, d1)
        a = stats(r, 0.61)
        row = dict(block=blk, TP=t, dcr_at_061=a["dcr"], dd_at_061=a["dd"], worstday_at_061=a["worst_day"])
        for tgt in (-20, -30):
            s = size_for_dd(r, tgt); b = stats(r, s)
            row[f"size_dd{-tgt}"] = s; row[f"dcr_dd{-tgt}"] = b["dcr"]; row[f"worstday_dd{-tgt}"] = b["worst_day"]
        rows.append(row)
R = pd.DataFrame(rows).round(3)
R.to_csv(os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_DIST_data", "risk_matched.csv"), index=False)
pd.set_option("display.width", 220); print(R.to_string(index=False))

# Day-block bootstrap (10-day blocks, 500 resamples): DCR at the size that gives a -30 % max drawdown, per TP, paired vs TP 10.
rng = np.random.default_rng(11)
out = []
for blk in ("ALL", "HOLDOUT"):
    S = B if blk == "ALL" else B[B.block == "HOLDOUT"]
    d0 = int(S.day.min()); d1 = int(max(S[[f"xday_TP{t}_1tick" for t in TPS]].max()))
    R_ = {t: daily(S[f"TP{t}_1tick"].values, S[f"xday_TP{t}_1tick"].values, d0, d1) for t in (5, 6, 8, 10)}
    n = len(R_[10]); nb = n // 10
    res = {t: [] for t in R_}
    for _ in range(500):
        st = rng.integers(0, n - 9, nb); idx = (st[:, None] + np.arange(10)).ravel()
        for t, r in R_.items():
            rr = r[idx]; s = size_for_dd(rr, -30); res[t].append(stats(rr, s)["dcr"])
    for t in (5, 6, 8):
        dlt = np.array(res[t]) - np.array(res[10])
        out.append(dict(block=blk, TP=t, median_gain_vs_TP10=np.median(dlt), ci_lo=np.quantile(dlt, .05), ci_hi=np.quantile(dlt, .95),
                        p_better=(dlt > 0).mean()))
O = pd.DataFrame(out).round(3)
O.to_csv(os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_DIST_data", "risk_matched_boot.csv"), index=False)
print("\nDCR at equal drawdown (-30 %), paired vs TP 10, 10-day block bootstrap (90 % range):"); print(O.to_string(index=False))
