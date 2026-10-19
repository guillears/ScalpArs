#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — Part C follow-up: the earlier 70× (and 50× / 85×) entry vs the 100× entry of the SAME episode
(replica LATER class, real era), walked on the same paths through the Part-A cells; MAE of the earlier entry before the 100× bar.
Usage: venv/bin/python scripts/study_frenzy_realfills_partc2.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_engine as EN  # noqa: E402
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
HR, MIN = 3_600_000, 60_000
CELLS = ["FIX tp1.5/sl3/720m", "FIX tp1.5/sl4/720m", "FIX tp3/sl3/720m", "FIX tp5/sl5/720m", "FIX tp8/sl5/720m", "TRAIL a3/g3/sl5/720m",
         "TRAIL a2/g2/sl4/720m", "SPLIT ½@1.5+½trail g3 BE sl4", "BETR be1.5/a2/g1/sl4/720m"]
if __name__ == "__main__":
    gn = [c[1] for c in EN.GRID]
    PR = pd.read_pickle(f"{RC.RF}/cands_priced.pkl"); W = pd.read_pickle(f"{RC.RF}/walk_all.pkl")
    CC = pd.read_csv(f"{OUT}_C_replica_first_bars.csv")
    W100 = W[(W["T"] == 100) & W.fresh_on & W.long_ready]
    rows = []
    for r in CC[CC.cls == "LATER"].itertuples():
        l = W100[(W100.pair == r.pair) & (W100.t_close >= r.t_close)].t_close.min()
        a, b = PR.get(f"{r.pair}|{r.t_close}", {}), PR.get(f"{r.pair}|{int(l)}", {})
        if "grid" not in a or "grid" not in b:
            continue
        o = dict(T=r.T, pair=r.pair, day=r.day, hours=r.hours, gap_h=(l - r.t_close) / HR, price_gap=(a["E"] / b["E"] - 1) * 100)
        # MAE of the early entry before the 100× bar, and its peak before it
        P = RC.gross_path(r.pair, int(r.t_close) + 8000, float(a["E"]), int((l - r.t_close) / MIN) + 5)
        k = int(np.searchsorted(P["t"], int(l) + 8000)) if P is not None else 0
        o["mae_before_100"] = float(P["lo"][:k].min()) if k else np.nan
        o["mfe_before_100"] = float(P["hi"][:k].max()) if k else np.nan
        o["early_exit_min"] = a["long_min"]
        for c in CELLS:
            j = gn.index(c); o["early|" + c] = a["grid"][j]; o["t100|" + c] = b["grid"][j]
        rows.append(o)
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}_C_later_pairs_grid.csv", index=False)
    S = []
    for T_ in (50, 70, 85):
        z = R[R["T"] == T_]
        if not len(z):
            continue
        s = dict(T=T_, N=len(z), days=z.day.nunique(), gap_h_med=z.gap_h.median(), price_gap_med=z.price_gap.median(),
                 mae_before_100_med=z.mae_before_100.median(), share_hit_m3_before_100=float((z.mae_before_100 <= -3).mean()),
                 share_hit_m5_before_100=float((z.mae_before_100 <= -5).mean()), early_exit_min_med=z.early_exit_min.median())
        for c in CELLS:
            s[c] = f"{z['early|' + c].mean():+.2f} vs {z['t100|' + c].mean():+.2f}"
        S.append(s)
    S = pd.DataFrame(S); S.to_csv(f"{OUT}_C_later_summary.csv", index=False)
    pd.set_option("display.width", 250); print(S.T.to_string())
