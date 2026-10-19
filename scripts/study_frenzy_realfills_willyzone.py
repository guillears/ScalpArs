#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — Part B follow-up: the only real-fill runner separator that beat its LODO null (FRENZY_WILLY, a BTC
market-state model) priced as a ZONE-SPECIFIC EXIT, LODO on real fills (rule + exit cell chosen on the other days), then the frozen 1D cut
(entry_btc_above72_pct > 50.8, the top real 1D cell) checked on the yr6 backtest WILLY fills. Day units (market-wide variable).
Usage: venv/bin/python scripts/study_frenzy_realfills_willyzone.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_exitgrid as EG  # noqa: E402
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
CUT = 50.8
EXITS = ["FIX tp5/sl4/240m", "FIX tp3/sl4/240m", "FIX tp8/sl5/720m", "TRAIL a2/g1/sl4/120m", "SPLIT ½@1.5+½@5 BE sl4", "FIX tp6/sl2.5/60m"]
if __name__ == "__main__":
    names = [c[1] for c in EG.cells()]
    D = pd.read_pickle(f"{RC.RF}/real.pkl"); V = np.load(f"{RC.RF}/grid_V.npy"); ref = pd.read_pickle(f"{RC.RF}/grid_meta.pkl")["ref"]
    X = pd.read_pickle(f"{RC.RF}/runner_X.pkl").set_index("key")
    m = (D.sleeve == "FRENZY_WILLY").values
    G = D[m].reset_index(drop=True); Vg = V[m]; rg = ref[m]
    z = pd.to_numeric(G.entry_btc_above72_pct, errors="coerce").values
    rows = []
    # (a) frozen cut, each exit: in-sample and LODO (exit chosen on other days)
    for e in EXITS:
        j = names.index(e); alt = np.where(z > CUT, Vg[:, j], rg)
        rows.append(dict(test=f"zone(btc_above72>{CUT}) → {e}", N=len(G), zone_n=int((z > CUT).sum()), zone_days=G.day[z > CUT].nunique(),
                         avg=alt.mean(), today=rg.mean(), delta=alt.mean() - rg.mean()))
    oos = np.zeros(len(G)); pk = []
    for d in G.day.unique():
        tr = (G.day != d).values; te = ~tr
        best, bv = None, -9
        for e in EXITS:
            j = names.index(e)
            for cut in np.nanpercentile(z[tr], [33.3, 50, 66.7]):
                v = np.where(z[tr] > cut, Vg[tr, j], rg[tr]).mean()
                if v > bv:
                    bv, best = v, (e, cut)
        pk.append(best); j = names.index(best[0])
        oos[te] = np.where(z[te] > best[1], Vg[te, j], rg[te])
    dd = G.day.values
    mu, lo, hi, p = RC.day_boot([(oos - rg)[dd == d] for d in np.unique(dd)])
    rows.append(dict(test="LODO (cut tercile/median + exit chosen on other days)", N=len(G), avg=oos.mean(), today=rg.mean(), delta=mu,
                     d_lo=lo, d_hi=hi, P=p, picks=str(pd.Series([f"{a} >{b:.1f}" for a, b in pk]).value_counts().to_dict())))
    # (b) backtest at the frozen cut
    M = np.load(f"{RC.RF}/bt_grid.npy"); coh = pd.read_pickle(f"{RC.SCR}/rid/cohort.pkl"); coh = coh[coh.blk == "BT"].reset_index(drop=True)
    rb = M[:, len(names)]; run = M[:, len(names) + 1] >= 5
    w = (coh.sleeve.values == "FRENZY_WILLY") & np.isfinite(rb)
    zb = pd.to_numeric(coh.entry_btc_above72_pct, errors="coerce").values
    ts = pd.to_datetime(coh.te, unit="ms"); per = np.where(ts < "2026-01-01", "2025", np.where(ts < "2026-07-01", "2026-H1", "2026-H2"))
    for pp in ("ALL", "2025", "2026-H1", "2026-H2"):
        mm = w & np.isfinite(zb) & ((per == pp) if pp != "ALL" else True)
        zz = mm & (zb > CUT)
        r = dict(test=f"BACKTEST {pp}: btc_above72>{CUT}", N=int(mm.sum()), zone_n=int(zz.sum()), coverage=float(np.isfinite(zb[w]).mean()),
                 run5_zone=float(run[zz].mean()), run5_rest=float(run[mm & ~(zb > CUT)].mean()))
        for e in ("FIX tp5/sl4/240m", "FIX tp6/sl2.5/60m"):
            j = names.index(e); alt = np.where(zb[mm] > CUT, M[mm, j], rb[mm])
            r[f"delta_{e}"] = float(alt.mean() - rb[mm].mean())
        rows.append(r)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}_B_willy_zone.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80); print(R.round(3).to_string())
