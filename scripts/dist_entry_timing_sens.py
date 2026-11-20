#!/usr/bin/env python3
"""DIST_ENTRY_TIMING — SENSITIVITY (descriptive, not a ship variant): why do matched early entries earn less despite a better price?
Re-sim the T−1 / T−2 entries with the CLOSED candle's ATR (atr15_pct, known only at T — look-ahead, used ONLY to isolate the price effect
from the partial-candle ATR effect), and split the paired Δ by exit kind. Usage: venv/bin/python -I scripts/dist_entry_timing_sens.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_entry_timing_eval as X   # noqa: E402
C = pd.read_pickle(os.path.join(X.OUT, "cand.pkl"))
E = pd.read_pickle(os.path.join(X.OUT, "sim.pkl"))
f = os.path.join(X.OUT, "sim_atr0.pkl")
if not os.path.exists(f):
    S = E[E["at"].isin(["Tm1", "Tm2"])][["pair", "cid", "T", "slip", "keep", "trig0", "ep", "at", "entry_ms"]].copy().reset_index(drop=True)
    S["atrp"] = C.loc[S.cid, "atr15_pct"].values
    S = X.simulate(S, 6); S.to_pickle(f)
S = pd.read_pickle(f)
z = E[E["at"] == "T0"].set_index("cid")
for a in ("Tm1", "Tm2"):
    e = E[E["at"] == a].set_index("cid"); s = S[S["at"] == a].set_index("cid")
    ii = e.index.intersection(z.index)
    d = pd.DataFrame(dict(T=z.loc[ii, "T"], n0=z.loc[ii, "net"], k0=z.loc[ii, "k"], nete=e.loc[ii, "net"], ke=e.loc[ii, "k"], ns=s.loc[ii, "net"],
                          a0=C.loc[ii, "atr15_pct"], ae=e.loc[ii, "atrp"])).dropna(subset=["n0", "nete", "ns"])
    for bn, sp in (("DESIGN", X.DES), ("HOLDOUT", X.HOL), ("FULL", X.FULL)):
        y = d[(d["T"] >= sp[0]) & (d["T"] < sp[1])]
        print(a, bn, "N", len(y), "| early ATR / T0 ATR median", round((y.ae / y.a0).median(), 3),
              "| paired Δ early(partial ATR)", round((y.nete - y.n0).mean(), 3), "| paired Δ early(closed ATR)", round((y.ns - y.n0).mean(), 3),
              "| TP share T0 / early", round((y.k0 == 0).mean() * 100, 1), round((y.ke == 0).mean() * 100, 1))
