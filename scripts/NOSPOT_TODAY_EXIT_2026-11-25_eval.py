#!/usr/bin/env python3
"""NOSPOT_TODAY_EXIT (2026-11-25, DECISION_LOG 642) — EVAL of NOSPOT_TODAY_EXIT_2026-11-25_sim.py. Nets = gross + real funding − entry
(0.05 % taker + modelled slip) − exit (maker 0.02 % on a TP; taker 0.05 % + slip otherwise) — the NOSPOT_EXIT / TP_GRID cost model.
One position per pair per cell (that cell's own exit time — the with-spot TP_GRID rows fixed it once under REF: small difference). Per block (DESIGN < 2025-07 · HOLDOUT ≥ 2025-07 · ALL): N · WR · breakeven WR ·
avg · day-block bootstrap 95 % CI (10-day blocks) · Kelly at the one-sided 95 % Wilson WR floor · top pair + leave-top-pair-out · TP / stop
share; per half-year. Run: venv/bin/python -I scripts/NOSPOT_TODAY_EXIT_2026-11-25_eval.py → reports/NOSPOT_TODAY_EXIT_2026-11-25{,_halves}.csv"""
import os, sys
import numpy as np
import pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(R, "scripts"))
import nospot_sleeve_lib as L
H = os.path.join(R, "reports", "NOSPOT_TODAY_EXIT_data"); DAY = 86_400_000
Z = dict(np.load(os.path.join(H, "sim_ns0.npz"), allow_pickle=True)); names = list(Z.pop("names"))
Z = {k: v[Z["ok"]] for k, v in Z.items()}                              # rows the sim could not price are out
X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
xi = Z["xi"]; Y = X.iloc[xi][["pair", "entry_ms", "blk", "half", "slip"]].reset_index(drop=True)
s = Y.slip.values[:, None].astype(float); tk = Z["k"] != 0
NET = Z["r"].astype(float) + Z["f"].astype(float) - s - 0.05 - np.where(tk, s + 0.05, 0.02)
rng = np.random.default_rng(7)
def onepos(j):
    o = np.lexsort((Y.entry_ms.values, Y.pair.values)); keep = []; lp = None; busy = -1
    for i in o:
        if Y.pair.values[i] != lp: lp = Y.pair.values[i]; busy = -1
        if Y.entry_ms.values[i] < busy: continue
        keep.append(i); busy = Z["xm"][i, j]
    return np.array(keep)
def wil(k, n, z=1.645):
    p = k / n; d = 1 + z*z/n
    return (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
def kel(v):
    w, lo = v[v > 0], v[v <= 0]; pf = wil(len(w), len(v)); b = w.mean() / -lo.mean()
    return pf - (1-pf)/b, len(w)/len(v)*100, -lo.mean()/(w.mean()-lo.mean())*100
def ci(v, d, nb=2000, blk=10):
    g = pd.DataFrame(dict(v=v, d=d)).groupby("d").v.agg(["sum", "count"]); dd = np.arange(g.index.min(), g.index.max()+1)
    g = g.reindex(dd, fill_value=0); n = len(dd); st = rng.integers(0, max(n-blk, 1), (nb, int(np.ceil(n/blk))))
    ix = (st[:, :, None] + np.arange(blk)).reshape(nb, -1)[:, :n]
    a = g["sum"].values[ix].sum(1) / np.maximum(g["count"].values[ix].sum(1), 1)
    return np.percentile(a, 2.5), np.percentile(a, 97.5)
rows = []; halves = {}
for j, nm in enumerate(names):
    rng = np.random.default_rng(7)                                        # own seed per cell → identical cells get identical CIs
    k = onepos(j)
    for blk in ("DESIGN", "HOLDOUT", "ALL"):
        kk = k if blk == "ALL" else k[Y.blk.values[k] == blk]
        v = NET[kk, j]; d = Y.entry_ms.values[kk] // DAY
        kf, wr, be = kel(v); lo, hi = ci(v, d)
        pr = pd.Series(v).groupby(Y.pair.values[kk]).sum()
        rows.append(dict(cell=nm, blk=blk, N=len(v), WR=wr, BE=be, avg=v.mean(), ci_lo=lo, ci_hi=hi, kelly_floor=kf,
                         top_pair=pr.idxmax(), loo_top=(v.sum()-pr.max())/(len(v)-1), tp_share=(Z["k"][kk, j] == 0).mean()*100,
                         stop_share=(Z["k"][kk, j] == 1).mean()*100))
        if blk == "ALL": halves[nm] = pd.Series(v).groupby(Y.half.values[kk]).agg(["count", "mean"]).round(2)
T = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(T.round(3).to_string())
for nm in ("REF", "TODAY"): print(nm); print(halves[nm].T.to_string())
T.to_csv(os.path.join(R, "reports", "NOSPOT_TODAY_EXIT_2026-11-25.csv"), index=False)
pd.concat({k: v for k, v in halves.items()}, names=["cell", "half"]).to_csv(os.path.join(R, "reports", "NOSPOT_TODAY_EXIT_2026-11-25_halves.csv"))
