#!/usr/bin/env python3
"""DIST_EXIT — descriptive extras AFTER the verdict (no selection): saved / deeper for representative non-picked cells, the Δ = 0 check
table for every mechanism cell, per-half paired Δ, 2 × slip on dynamic exits, and the GAVE_BACK rescue arithmetic.
Usage: venv/bin/python -I scripts/dist_exit_extras.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dist_exit_eval as E   # noqa: E402

S, Z, V, slip, prim, pair, T = E.build()
Vdy = E.net_mat(Z, slip, dyn2=True); V20 = E.net_mat(Z, slip, flat=0.20)
full = (T >= E.DES[0]) & (T < E.HOL[1]); hol = T >= E.HOL[0]
MF = E.cohort_onepos(V, Z, pair, T, prim & full); ref = MF[:, 0]
hv = E.half_of(T)
REPS = ["A_TP3_SLATR", "A_TP5_SLATR", "A_TP10_SL6", "B_BE2", "B_BE3", "B_BE4", "C_L31", "C_L53", "D_T3_G1.5", "D_T3_A2", "T_4h", "T_24h"]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
sd = []
for nm in REPS:
    j = E.NAMES.index(nm)
    for blk, msk in (("FULL", full), ("HOLDOUT", hol)):
        pm = ref & msk & np.isfinite(V[:, j]); d = V[pm, j] - V[pm, 0]; kr = Z["k"][pm, 0]; wk = Z["worked"][pm, 0] == 1
        for lab, g in (("REF stopped · worked first (GAVE_BACK)", (kr == 1) & wk), ("REF stopped · never worked", (kr == 1) & ~wk),
                       ("REF TP", kr == 0), ("REF time", kr == 2)):
            dg = d[g]
            sd.append(dict(cell=nm, blk=blk, ref_outcome=lab, N=int(g.sum()), saved=int((dg > 0.05).sum()), deeper=int((dg < -0.05).sum()),
                           same=int((np.abs(dg) <= 0.05).sum()), sum_saved=dg[dg > 0.05].sum(), sum_deeper=dg[dg < -0.05].sum(), net=dg.sum()))
SD = pd.DataFrame(sd); print("SAVED / DEEPER (descriptive)\n", SD.round(1).to_string())
d0 = []
for j, c in enumerate(E.CL):
    if c["fam"] in ("B", "C", "D", "D2"):
        pm = ref & full & np.isfinite(V[:, j]); na = Z["armhit"][pm, j] == 0; d = V[pm, j] - V[pm, 0]
        d0.append(dict(cell=c["name"], N=int(pm.sum()), never_armed=int(na.sum()), max_abs_delta_never_armed=float(np.abs(d[na]).max()),
                       armed=int((~na).sum()), delta_armed_avg=d[~na].mean(), hol_dyn2x_avg=Vdy[MF[:, j] & hol, j].mean(),
                       hol_avg=V[MF[:, j] & hol, j].mean()))
D0 = pd.DataFrame(d0); print("\nDELTA=0 CHECK + 2x dyn slip\n", D0.round(4).to_string())
hh = []
for nm in ["REF_B"] + REPS:
    j = E.NAMES.index(nm); r = dict(cell=nm)
    for h, _, _ in E.HALVES:
        pm = ref & (hv == h) & np.isfinite(V[:, j]); r[h] = (V[pm, j] - V[pm, 0]).mean() if nm != "REF_B" else V[ref & (hv == h), 0].mean()
    pm = ref & full & np.isfinite(V[:, j]); r["full_dpair"] = (V[pm, j] - V[pm, 0]).mean(); r["full_dpair20"] = (V20[pm, j] - V20[pm, 0]).mean()
    hh.append(r)
HH = pd.DataFrame(hh); print("\nPER-HALF paired Δ (REF row = REF avg)\n", HH.round(3).to_string())
pd.to_pickle(dict(SD=SD, D0=D0, HH=HH), os.path.join(E.D, "extras.pkl"))
