#!/usr/bin/env python3
"""FAST_TP — backtest BEFORE → AFTER rows (DESIGN / HOLDOUT) for the frozen pick (or DESIGN best) and the full-span best (hindsight)
per sleeve, same columns as the real-fill table. Writes reports/FAST_TP_2026-11-20_bt_tables.csv. Research only."""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_tp_eval as EV   # noqa: E402
S = json.load(open(EV.OUT + "_summary.json"))
rows = []
for sl in ("EXH", "DIST", "FLATBASE", "WILLY", "HT"):
    Z = EV.load(sl, "MAIN"); V, Fi, Dl = EV.mats(Z); r = S[sl]
    for lab, c in (("frozen pre-reg pick" if r["pick"] else "DESIGN best (no pick)", r["pick"] or r["design_best_cell"]),
                   ("full-span best (hindsight)", r["full_best_cell_hindsight"])):
        j = EV.CELLS.index(c)
        for b in ("DESIGN", "HOLDOUT"):
            m = (Z.block == b).values
            d0 = EV.ms(EV.START[sl] if b == "DESIGN" else EV.SPLIT[sl]) // EV.DAY; d1 = EV.ms(EV.SPLIT[sl] if b == "DESIGN" else EV.END) // EV.DAY
            xc = np.where(Fi[:, j] >= 0, Z.ts + (Fi[:, j] + 1) * EV.MIN, Z.ref_xms)
            ref = Z.ref_net.values[m]; v = V[m, j]; f = Fi[m, j] >= 0
            lo, hi = EV.boot_ci(Dl[m, j], Z.day.values[m])
            rows.append(dict(sleeve=EV.NAME[sl], block=b, which=lab, cell=c, N=int(m.sum()), WR_b=(ref > 0).mean() * 100, WR_a=(v > 0).mean() * 100,
                             avg_b=ref.mean(), avg_a=v.mean(), d=(v - ref).mean(), ci_lo=lo, ci_hi=hi, usd_b=ref.sum() * 61, usd_a=v.sum() * 61,
                             dcr_b=EV.dcr(ref, Z.ref_xms.values[m], d0, d1), dcr_a=EV.dcr(v, xc[m], d0, d1),
                             changed=int(f.sum()), saved=int((f & (ref <= 0)).sum()), cut=int((f & (ref > 0)).sum())))
T = pd.DataFrame(rows); T.to_csv(EV.OUT + "_bt_tables.csv", index=False)
pd.set_option("display.width", 250); print(T.round(3).to_string())
