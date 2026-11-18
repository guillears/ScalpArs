#!/usr/bin/env python3
"""DIST_WL — Kelly sizes with the 30-50 % in-sample haircut and the concurrency (clustering) caveat, + DCR at candidate sizes.
Usage: venv/bin/python -I scripts/dist_wl_kelly.py"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import dist_wl_analyze as A  # noqa: E402
import dist_wl_survivors as SV  # noqa: E402
OUT = A.OUT
S = pd.read_pickle(os.path.join(OUT, "rows.pkl")); T = pd.read_pickle(os.path.join(OUT, "screen.pkl"))
P = pd.read_pickle(os.path.join(OUT, "design_passers.pkl"))
print("top 1D DESIGN |t_net| among null passers:", P[P.kind == "1D"].assign(a=lambda d: d.t_net.abs()).sort_values("a", ascending=False)[["var", "bin", "t_net"]].head(5).to_string())
r = T[(T.kind == "1D") & (T["var"] == "rsi1h") & (T.bin == "T3/3")].iloc[0].to_dict(); r["mode"] = "keep_only"
k = SV.keep_mask(S, r)
span = (int(S.entry_ms.min() // A.DAY), int(S.LIVE_x.max() // A.DAY))
out = []
for lab, msk in (("before", np.ones(len(S), bool)), ("after", k)):
    m = A.one_pos(S, msk); v0 = S.net_LIVE.values[m]; day = S.day.values[m]; xd = S.LIVE_x.values[m] // A.DAY
    base_avg = S.net_LIVE.values[A.one_pos(S, np.ones(len(S), bool))].mean()
    for hc in ((0.0, 0.3, 0.5) if lab == "after" else (0.0,)):
        v = v0 - hc * (v0.mean() - base_avg)              # haircut the Δ vs baseline, applied as a shift on every kept trade
        fD, pD, nD = A.kelly(v, day); fT, pT, nT = A.kelly(v)
        row = dict(set=lab, haircut=hc, avg=v.mean(), kelly_day=fD, notional_day=nD, kelly_trade=fT, notional_trade=nT)
        for sz in (0.05, 0.10, 0.20, 0.30, 0.61):
            d, dd, wd = A.dcr(v, xd, sz, span); row[f"dcr@{sz}"] = d; row[f"dd@{sz}"] = dd; row[f"worst@{sz}"] = wd
        out.append(row)
K = pd.DataFrame(out); K.to_csv(os.path.join(OUT, "kelly_sizes.csv"), index=False)
pd.set_option("display.width", 250); print(K.round(3).to_string())
