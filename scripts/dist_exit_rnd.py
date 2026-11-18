#!/usr/bin/env python3
"""DIST_EXIT (PREREG reports/DIST_EXIT_2026-11-18_PREREG.md) — the random-entry CONTROL rows.
For every PRIMARY candidate row (rows.pkl, rsi1h keep_mask), 3 bars drawn uniformly (seed 20261118) from the 15m-close rows of
UNIFIED_SHORT pool.pkl of the same pair and UTC day, excluding the DIST bar itself. Output reports/DIST_EXIT_data/rnd.pkl.
Usage: venv/bin/python -I scripts/dist_exit_rnd.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_short_lib as UL   # noqa: E402
import dist_wl_survivors as SV   # noqa: E402
R = UL.R; OUT = os.path.join(R, "reports", "DIST_EXIT_data"); os.makedirs(OUT, exist_ok=True)
DAY = 86_400_000
REP = dict(kind="1D", var="rsi1h", lo=74.07103, hi=np.inf, mode="keep_only", var2="", lo2=np.nan, hi2=np.nan)

S = pd.read_pickle(os.path.join(R, "reports", "DIST_WL_2026-11-18_data", "rows.pkl"))
P = S[SV.keep_mask(S, REP)]
X = UL.load()
X = X.loc[X.is15.values, ["pair", "entry_ms", "atr15_pct", "q5", "btc_d1_regime", "btc_regime", "half", "day"]].copy()
X["pair"] = X.pair.astype(str)
print("pool 15m rows", len(X), "pair-days", X.groupby(["pair", "day"]).ngroups, flush=True)
G = {k: g for k, g in X.groupby(["pair", "day"])}
rng = np.random.default_rng(20261118)
out = []
for p, t in zip(P.pair.astype(str).values, P.entry_ms.values.astype(np.int64)):
    g = G.get((p, int(t) // DAY))
    if g is None:
        continue
    g = g[g.entry_ms.values != t]
    if not len(g):
        continue
    k = rng.choice(len(g), size=min(3, len(g)), replace=False)
    h = g.iloc[k].copy(); h["src_entry_ms"] = t; out.append(h)
Rn = pd.concat(out, ignore_index=True)
Rn.to_pickle(os.path.join(OUT, "rnd.pkl"))
print("RND rows", len(Rn), "from", len(P), "DIST candidates; per-candidate mean", round(len(Rn) / len(P), 2), flush=True)
