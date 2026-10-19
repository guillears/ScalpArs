#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — SECONDARY CHECK ONLY: the Part-A exit grid on the yr6 2-year replay fills (seed 1, < 2026-10-03,
from the runner-ID cohort pickle). Same paths / walker / fees / cells as the real-fill grid (study_frenzy_realfills_exitgrid.cells).
Workers compute each fill's whole grid row (paths are not stored). Output: scratch npy + meta.
Usage: venv/bin/python scripts/study_frenzy_realfills_btgrid.py [workers]"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_exitgrid as EG  # noqa: E402
C = EG.cells()


def one(a):
    i, r = a
    try:
        P = RC.gross_path(r["pair"], int(r["te"]), float(r["E"]), 720)
    except Exception as e:  # noqa: BLE001
        return i, None, repr(e)
    if P is None:
        return i, None, "nopath"
    row = np.array([EG.run_cell(P, r, c[2])[0] for c in C] + [RC.today(P, r)[0]] + list(RC.mfe_before(P)[:2]) +
                   [RC.mfe_before(P, 720, 4.0)[0], P["n_bar"]], dtype=float)
    return i, row, None


if __name__ == "__main__":
    D = pd.read_pickle(f"{RC.SCR}/rid/cohort.pkl")
    B = D[D.blk == "BT"].reset_index(drop=True)
    cols = ["pair", "te", "E", "sleeve", "entry_frenzy_vol_mult", "entry_btc_1h_slope", "entry_atr_pct"]
    jobs = [(i, {c: r[c] for c in cols if c in B.columns}) for i, r in B.iterrows()]
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with Pool(nw) as p:
        res = p.map(one, jobs, chunksize=8)
    M = np.full((len(B), len(C) + 5), np.nan)
    for i, row, err in res:
        if row is not None:
            M[i] = row
    np.save(f"{RC.RF}/bt_grid.npy", M)
    B[["opened_at", "pair", "sleeve", "te", "seed"] + [c for c in ("period",) if c in B]].to_pickle(f"{RC.RF}/bt_grid_rows.pkl")
    print("done", len(B), "unpriced", int(np.isnan(M[:, -5]).sum()))
