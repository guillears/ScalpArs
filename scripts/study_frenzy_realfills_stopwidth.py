#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — Part A follow-up: FRENZY_LONG stop width at today's +1.5 TP (the one dimension every top LONG
cell shares). Read on the LIVE-STOPPED cohort (CLAUDE.md exit/stop CF rule): saved vs deeper; fills live did not stop must show Δ = 0.
LODO over the stop-only family (TP 1.5, SL ∈ {1.5, 2, 2.5, 3, 4, 5}, 12 h); backtest saved/deeper split per period at the frozen −4.
Usage: venv/bin/python scripts/study_frenzy_realfills_stopwidth.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_exitgrid as EG  # noqa: E402
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
SLS = (1.5, 2, 2.5, 3, 4, 5)
if __name__ == "__main__":
    names = [c[1] for c in EG.cells()]
    D = pd.read_pickle(f"{RC.RF}/real.pkl"); V = np.load(f"{RC.RF}/grid_V.npy"); ref = pd.read_pickle(f"{RC.RF}/grid_meta.pkl")["ref"]
    rows = []
    for g in ("FRENZY_LONG", "FRENZY_LITE", "POOLED"):
        m = (D.sleeve == g).values if g != "POOLED" else D.sleeve.isin(["FRENZY_LONG", "FRENZY_LITE"]).values
        G = D[m].reset_index(drop=True); rg = ref[m]; dd = G.day.values
        cols = [names.index(f"FIX tp1.5/sl{s}/720m") for s in SLS] if g == "FRENZY_LONG" else None
        if g != "FRENZY_LONG":   # LITE / pooled: the sleeve's own TP with SL 4 = today's walker with sl 4 (LITE C TP per fill)
            PP = pd.read_pickle(f"{RC.RF}/real_paths.pkl")
            Vs = np.array([[RC.walk(PP[r.key], tp=(1.5 if r.sleeve == "FRENZY_LONG" else RC.lite_tp(r)), sl=s, cap=720)[0] for s in SLS]
                           for _, r in G.iterrows()])
        else:
            Vs = V[m][:, cols]
        st = rg < -2
        for k, s in enumerate(SLS):
            d = Vs[:, k] - rg
            rows.append(dict(group=g, sl=s, N=len(G), avg=Vs[:, k].mean(), today=rg.mean(), delta=d.mean(), stopped_n=int(st.sum()),
                             stopped_days=G.day[st].nunique(), saved=int(((Vs[:, k] > -2.5) & st).sum()), deeper=int(((Vs[:, k] < -3.5) & st).sum()),
                             nonstopped_delta_sum=float(d[~st].sum()), stopped_delta_sum=float(d[st].sum())))
        oos = np.zeros(len(G)); pk = []
        for d_ in np.unique(dd):
            tr = dd != d_; k = int(np.argmax(Vs[tr].mean(0))); pk.append(SLS[k]); oos[~tr] = Vs[~tr, k]
        mu, lo, hi, p = RC.day_boot([(oos - rg)[dd == x] for x in np.unique(dd)])
        rows.append(dict(group=g, sl="LODO", N=len(G), avg=oos.mean(), today=rg.mean(), delta=mu, d_lo=lo, d_hi=hi, P=p,
                         picks=str(pd.Series(pk).value_counts().to_dict())))
    # backtest, frozen −4, saved / deeper on the −3-stopped fills
    M = np.load(f"{RC.RF}/bt_grid.npy"); B = pd.read_pickle(f"{RC.RF}/bt_grid_rows.pkl")
    rb = M[:, len(names)]; ts = pd.to_datetime(B.te, unit="ms")
    per = np.where(ts < "2026-01-01", "2025", np.where(ts < "2026-07-01", "2026-H1", "2026-H2"))
    j4 = names.index("FIX tp1.5/sl4/720m")
    for pp in ("ALL", "2025", "2026-H1", "2026-H2"):
        mm = (B.sleeve.values == "FRENZY_LONG") & np.isfinite(rb) & ((per == pp) if pp != "ALL" else True)
        st = mm & (rb < -2); v4 = M[:, j4]
        d = v4[mm] - rb[mm]; days = pd.Series(ts[mm]).dt.strftime("%Y-%m-%d").values
        mu, lo, hi, p = RC.day_boot([d[days == x] for x in np.unique(days)])
        rows.append(dict(group="BACKTEST LONG", sl=f"4 ({pp})", N=int(mm.sum()), avg=float(v4[mm].mean()), today=float(rb[mm].mean()), delta=mu,
                         d_lo=lo, d_hi=hi, P=p, stopped_n=int(st.sum()), saved=int((st & (v4 > -2.5)).sum()), deeper=int((st & (v4 < -3.5)).sum()),
                         nonstopped_delta_sum=float((v4 - rb)[mm & ~st].sum())))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}_A_stopwidth.csv", index=False)
    pd.set_option("display.width", 250); print(R.round(3).to_string())
