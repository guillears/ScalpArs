#!/usr/bin/env python3
"""DIST red-candle 2D — theory-first family (reports/DIST_REDCANDLE_2D_2026-11-24_PREREG.md, frozen before computing). Research only."""
import os
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = pd.read_csv(os.path.join(ROOT, "reports", "DIST_WL_2026-11-18_trades.csv.gz"))
K = T[(T.in_LIVE.astype(str) == "True") & (T.rsi1h >= 74.07)].copy()
K["y"] = K.net_LIVE; K["hold"] = pd.to_datetime(K.entry_ms, unit="ms") >= "2025-07-01"; K["big"] = K.bar_ret_pct < -3.0
FAM = [("T1", "close_pos15", "lo"), ("T2", "lowwick15", "lo"), ("T3", "oi_chg30m", "lo"), ("T4", "fund_last", "hi"),
       ("T5", "top_pos_chg4h", "lo"), ("T6", "dist_base_pct", "hi"), ("T7", "btc_ret4h", "lo"), ("T8", "vmult15", "hi")]
rng = np.random.default_rng(1124)


def tst(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def good_mask(k, col, side, cut):
    return (k[col] <= cut) if side == "lo" else (k[col] >= cut)


cuts = {c: K[K.big & ~K.hold][c].median() for _, c, _ in FAM}
rows = []
for tk, c, side in FAM:
    k = K[K[c].notna()]
    g = good_mask(k, c, side, cuts[c])
    out = dict(theory=tk, var=c, good=side, cut=round(cuts[c], 4), cover=f"{len(k)}/{len(K)}")
    for blk, m in (("DES", ~k.hold), ("HOL", k.hold)):
        for sz, s in (("BIG", k.big), ("SMALL", ~k.big)):
            out[f"{blk}_{sz}_good"] = k[m & s & g].y.mean(); out[f"{blk}_{sz}_bad"] = k[m & s & ~g].y.mean()
            out[f"n_{blk}_{sz}_good"] = int((m & s & g).sum()); out[f"n_{blk}_{sz}_bad"] = int((m & s & ~g).sum())
    out["HOL_BIG_t"] = tst(k[k.hold & k.big & g].y, k[k.hold & k.big & ~g].y)
    rows.append(out)
R = pd.DataFrame(rows)
# family-wise null: shuffle y within the HOLDOUT big-red trades, max one-sided t over the 8 theories
H = K[K.hold & K.big].copy()
null = []
for _ in range(2000):
    yp = rng.permutation(H.y.values); best = -9
    for tk, c, side in FAM:
        ok = H[c].notna().values; g = good_mask(H, c, side, cuts[c]).values
        best = max(best, tst(yp[ok & g], yp[ok & ~g]))
    null.append(best)
thr = np.quantile(null, 0.95)
R["fw_thr95"] = round(thr, 4)
R["fw_p"] = [(np.array(null) >= t).mean() if np.isfinite(t) else np.nan for t in R.HOL_BIG_t]
pd.set_option("display.width", 250)
print(f"cohort {len(K)} · BIG {int(K.big.sum())} · HOLDOUT BIG {len(H)} · family-wise 95 % one-sided t {thr:.2f}")
show = ["theory", "var", "good", "cut", "DES_BIG_good", "DES_BIG_bad", "HOL_BIG_good", "HOL_BIG_bad", "n_HOL_BIG_good", "n_HOL_BIG_bad", "HOL_BIG_t", "fw_p",
        "HOL_SMALL_good", "HOL_SMALL_bad", "n_HOL_SMALL_bad"]
print(R[show].round(2).to_string(index=False))
R.to_csv(os.path.join(ROOT, "reports", "DIST_REDCANDLE_2D_2026-11-24.csv"), index=False)
