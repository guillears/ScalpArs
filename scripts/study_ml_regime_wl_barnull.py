#!/usr/bin/env python3
"""ML REGIME W/L 2026-10-26 — same-statistic null for the regime-conditional filter search: how many (bad-regime side ∧ entry zone)
cohorts pass the expectancy bar (N ≥ 15, WR < 61.5 %, cluster P(avg<0) ≥ 0.95 by a day-cluster normal approximation, ≥ 8 days, no day
≥ 50 % of the loss) on the real outcome vs on circularly shifted outcomes (day clustering kept). Run after _deep.py.
    venv/bin/python -W ignore scripts/study_ml_regime_wl_barnull.py
"""
import numpy as np, pandas as pd
from math import erf, sqrt
norm_cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
from study_ml_regime_wl_deep import K, zmask, entry_columns, PFX
from study_ml_regime_wl_analyse import zones, BAD

SPL = {"A_HICORR": zmask(K, "br_corr24h_top20", ">", 0.3164).values, "D_LOWRV7": zmask(K, "btc_rv_7d", "<=", 0.1122).values,
       "C_LATEERA": K.era_group.isin(BAD).values}
cols = entry_columns()
day = pd.factorize(K.day)[0]; nd = day.max() + 1
M = []
for nm, R in SPL.items():
    for c in cols:
        x = pd.to_numeric(K[c], errors="coerce").values
        for lab, z in zones(x, c):
            m = z & R
            if m.sum() >= 15:
                M.append((nm, c, lab, m))
Z = np.stack([m[3] for m in M])


def npass(y):
    out = []
    for m in Z:
        yy = y[m]; d = day[m]
        if (yy > 0).mean() * 100 >= 61.5 or yy.mean() >= 0:
            out.append(False); continue
        s = np.bincount(d, yy, nd); c = np.bincount(d, None, nd); k = (c > 0).sum()
        if k < 8:
            out.append(False); continue
        mu = s.sum() / c.sum(); v = ((s - mu * c) ** 2).sum() / c.sum() ** 2 * k / (k - 1)
        P = norm_cdf(-mu / np.sqrt(v)) if v > 0 else 0
        neg = np.minimum(np.bincount(d, np.where(yy < 0, yy, 0), nd), 0)
        out.append(P >= 0.95 and (neg.min() / neg.sum() < 0.5))
    return np.array(out)


y = K.y.values
obs = npass(y)
rng = np.random.default_rng(3)
nul = [npass(np.roll(y, k)).sum() for k in rng.choice(np.arange(5, len(y) - 5), 60, replace=False)]
# within-side null: keep every fill's regime side (so the bad side keeps its own negative mean) and rotate outcomes INSIDE the side
split_of = np.array([m[0] for m in M])
def within(k):
    tot = 0
    for nm, R in SPL.items():
        ys = y.copy(); idx = np.where(R)[0]; ys[idx] = np.roll(y[idx], k % len(idx))
        sel = split_of == nm
        global Z
        Zs = Z; Z = Zs[sel]; tot += npass(ys).sum(); Z = Zs
    return tot
nul_w = [within(k) for k in rng.choice(np.arange(5, 60), 50, replace=False)]
res = pd.DataFrame([{"split": m[0], "variable": m[1], "zone": m[2], "N": int(m[3].sum()), "pass": bool(o)} for m, o in zip(M, obs)])
res.to_csv(PFX + "_barnull_cohorts.csv", index=False)
print(f"regime-conditional cohorts tested {len(M)} · passing the bar: observed {obs.sum()} · shifted-outcome null median {np.median(nul):.0f}, "
      f"95th pct {np.percentile(nul, 95):.0f}, max {max(nul)} · P(null ≥ observed) {(np.array(nul) >= obs.sum()).mean():.2f}")
print(f"WITHIN-SIDE null (bad-side mean kept): median {np.median(nul_w):.0f}, 95th pct {np.percentile(nul_w, 95):.0f}, max {max(nul_w)} · P(null ≥ observed) {(np.array(nul_w) >= obs.sum()).mean():.2f}")
print(res[res["pass"]].groupby("split").size().to_dict())
print(res[res["pass"]].head(40).to_string(index=False))
