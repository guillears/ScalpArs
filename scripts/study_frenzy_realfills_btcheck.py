#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — SECONDARY backtest check of the Part-A cells at FROZEN cuts (cells named from the real-fill
results; the backtest never chooses). yr6 seed 1, < 2026-10-03, periods 2025 / 2026-H1 / 2026-H2. Δ vs today's exit per fill, day-bootstrap.
Usage: venv/bin/python scripts/study_frenzy_realfills_btcheck.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_exitgrid as EG  # noqa: E402
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
CELLS = {"FRENZY_LONG": ["FIX tp1.5/sl4/720m", "FIX tp1.5/sl5/720m", "FIX tp1.5/sl2.5/720m", "BETR be1.5/a2/g1/sl4/240m", "SPLIT ½@1.5+½@5 BE sl4",
                         "FIX tp5/sl4/240m", "TRAIL a2/g1/sl4/120m", "FIX tp2/sl4/720m", "FIX tp1.5/sl4/240m"],
         "FRENZY_LITE": ["FIX tp5/sl2.5/240m", "ATR tp4x/sl3x/240m", "TRAIL a1/g3/sl1.5/60m", "BETR be1.5/a2/g1.5/sl3/240m", "FIX tp3/sl4/720m",
                         "FIX tp2/sl4/720m"],
         "FRENZY_WILLY": ["FIX tp6/sl2.5/60m", "ATR tp2x/sl1x/240m", "SPLIT ½@1.5+½@5 BE sl3", "FIX tp1.5/sl4/240m"],
         "POOLED": ["BETR be1.5/a2/g1/sl4/240m", "SPLIT ½@1.5+½@5 BE sl4", "FIX tp6/sl4/60m", "FIX tp1.5/sl4/720m"]}
if __name__ == "__main__":
    names = [c[1] for c in EG.cells()]
    M = np.load(f"{RC.RF}/bt_grid.npy"); B = pd.read_pickle(f"{RC.RF}/bt_grid_rows.pkl")
    ts = pd.to_datetime(B.te, unit="ms"); B["day"] = ts.dt.strftime("%Y-%m-%d")
    B["period"] = np.where(ts < "2026-01-01", "2025", np.where(ts < "2026-07-01", "2026-H1", "2026-H2"))
    ref = M[:, len(names)]
    rows = []
    for g, cl in CELLS.items():
        m0 = np.isfinite(ref) & ((B.sleeve == g).values if g != "POOLED" else True)
        for per in ("ALL", "2025", "2026-H1", "2026-H2"):
            m = m0 & ((B.period == per).values if per != "ALL" else True)
            dd = B.day.values[m]
            rows.append(dict(group=g, period=per, cell="TODAY", N=int(m.sum()), avg=float(ref[m].mean()), WR=float((ref[m] > 0).mean()),
                             d=0.0, d_lo=np.nan, d_hi=np.nan, P=np.nan))
            for c in cl:
                v = M[m, names.index(c)]; d = v - ref[m]
                mu, lo, hi, p = RC.day_boot([d[dd == x] for x in np.unique(dd)])
                rows.append(dict(group=g, period=per, cell=c, N=int(m.sum()), avg=float(np.nanmean(v)), WR=float(np.nanmean(v > 0)), d=mu,
                                 d_lo=lo, d_hi=hi, P=p))
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}_A_backtest_check.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print(R.round(3).to_string())
