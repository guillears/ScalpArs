#!/usr/bin/env python3
"""DIST_AFTERCRASH (2026-11-21) — FEATURES, outcome-blind. Research only.

For every fill of the DL 586 kept DIST cohort (reports/ENTRY_WL_2026-11-18_data/dist.pkl, 1,547 fills) compute, from 5m bars that
CLOSED at or before the signal (signal 15m bar closes at entry_ms; last closed 5m bar j has open = entry_ms - 5 min):
  r15 / r30 / r60 / r240 : close[j] / close[j-3 / -6 / -12 / -48] - 1   (%)
  dd24                   : close[j] / max(high[j-287 .. j]) - 1           (% below the 24 h high, incl. the signal bar)
  drop60                 : close[j] / max(high[j-11 .. j]) - 1            (% below the 60-min high)
  volx15                 : quote vol of the signal 15m (bars j-2..j) / median 15m quote vol over the prior 24 h (non-overlapping 15m
                           blocks ending before j-2)  -- liquidation-cascade proxy (no Route B before 2026-11-19)
Also a 5m event-study path: close[j-24 .. j+24] / entry_open - 1 (entry_open = open of bar j+1 = model fill).
Lag check: r60 must equal dist.pkl ret1h_pct (same formula), r15 ~ bar_ret_pct (close/open of the signal 15m).
Output: reports/DIST_AFTERCRASH_2026-11-21_data/features.csv.gz + paths.npy (no outcome columns written).
Usage: venv/bin/python -I scripts/DIST_AFTERCRASH_2026-11-21_features.py
"""
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import dist_exit_sim as DS  # noqa: E402
TB = DS.NS.TB
OUT = os.path.join(R, "reports", "DIST_AFTERCRASH_2026-11-21_data"); os.makedirs(OUT, exist_ok=True)
M5 = 300_000
W = 24  # 2 h each side, 5m bars


def main():
    d = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl")).reset_index(drop=True)
    rows, paths = [], np.full((len(d), 2 * W + 1), np.nan)
    for p, g in d.groupby("pair"):
        x = TB.load(p)
        if x is None:
            for i in g.index:
                rows.append(dict(i=i))
            continue
        t, o, h, l, c, v, qv = x
        q15 = pd.Series(qv).rolling(3).sum().values
        for i, r in g.iterrows():
            j = int(np.searchsorted(t, int(r.entry_ms) - M5))
            if j >= len(t) or t[j] != int(r.entry_ms) - M5 or j < 300:
                rows.append(dict(i=i)); continue
            cj = c[j]
            f = dict(i=i)
            for k, n in (("r15", 3), ("r30", 6), ("r60", 12), ("r240", 48)):
                f[k] = (cj / c[j - n] - 1) * 100
            f["dd24"] = (cj / h[j - 287:j + 1].max() - 1) * 100
            f["drop60"] = (cj / h[j - 11:j + 1].max() - 1) * 100
            prior = q15[j - 3 - 285:j - 2:3]  # 15m blocks ending at j-3, j-6, ... over the prior 24 h
            prior = prior[np.isfinite(prior)]
            f["volx15"] = q15[j] / np.median(prior) if len(prior) > 20 and np.median(prior) > 0 else np.nan
            rows.append(f)
            if j + 1 < len(t):
                e0 = o[j + 1]
                a, b = j - W, min(j + W, len(t) - 1)
                seg = c[a:b + 1] / e0 - 1
                paths[i, :len(seg)] = seg * 100
    F = pd.DataFrame(rows).set_index("i").sort_index()
    F = d[["pair", "entry_ms", "block", "half", "day", "btc_d1_regime", "bar_ret_pct", "ret1h_pct", "ret4h_pct", "dd_peak_pct",
           "vmult15"]].join(F)
    F["entry"] = pd.to_datetime(F.entry_ms, unit="ms")
    # lag checks
    ok = F.r60.notna()
    print("coverage", ok.mean(), "| r60 vs ret1h_pct max|diff|", float((F.r60 - F.ret1h_pct)[ok].abs().max()),
          "| r240 vs ret4h_pct max|diff|", float((F.r240 - F.ret4h_pct)[ok].abs().max()),
          "| corr r15~bar_ret", float(np.corrcoef(F.r15[ok], F.bar_ret_pct[ok])[0, 1]))
    F.to_csv(os.path.join(OUT, "features.csv.gz"), index=False)
    np.save(os.path.join(OUT, "paths.npy"), paths)
    desc = F[["r15", "r30", "r60", "r240", "dd24", "drop60", "volx15"]]
    print(desc.describe(percentiles=[.01, .05, .1, .25, .5]).round(2).to_string())
    for k in ("r15", "r30", "r60", "drop60"):
        for th in (-5, -10, -15, -20, -25):
            print(k, th, "DES", int(((F[k] <= th) & (F.block == "DESIGN")).sum()), "HOL", int(((F[k] <= th) & (F.block == "HOLDOUT")).sum()))


if __name__ == "__main__":
    main()
