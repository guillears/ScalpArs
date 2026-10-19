#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — the per-fill deliverable CSV (reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19.csv): every real
fill with its booked %, walked today's exit, path outcomes (MFE before −3, runners), key Part-A cells, and its Part C / D status.
Usage: venv/bin/python scripts/study_frenzy_realfills_csv.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_exitgrid as EG  # noqa: E402
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
KEEP = ["FIX tp1.5/sl4/720m", "FIX tp1.5/sl5/720m", "BETR be1.5/a2/g1/sl4/240m", "SPLIT ½@1.5+½@5 BE sl4", "FIX tp5/sl4/240m",
        "TRAIL a2/g1/sl4/120m", "FIX tp6/sl2.5/60m", "FIX tp3/sl3/720m"]
if __name__ == "__main__":
    names = [c[1] for c in EG.cells()]
    D = pd.read_pickle(f"{RC.RF}/real.pkl"); V = np.load(f"{RC.RF}/grid_V.npy"); meta = pd.read_pickle(f"{RC.RF}/grid_meta.pkl")
    O = D[["key", "sleeve", "pair", "era", "src", "day", "opened_at", "entry_price", "entry_frenzy_vol_mult", "entry_frenzy_hours", "y_book"]].copy()
    O["today_walk_net"] = meta["ref"]; O["today_usd"] = O.today_walk_net / 100 * O.sleeve.map(RC.NOTIONAL)
    B = pd.read_csv(f"{OUT}_B_outcomes.csv")[["key", "mfe", "run3", "run5", "run8"]]
    O = O.merge(B, on="key", how="left")
    for c in KEEP:
        O["cell|" + c] = V[:, names.index(c)]
    PP = pd.read_pickle(f"{RC.RF}/real_paths.pkl")
    O["path_provisional_1m"] = [PP[k]["n_bar"] > 0 for k in O.key]
    O["path_hours"] = [(PP[k]["end_ms"] - PP[k]["te"]) / 3.6e6 for k in O.key]
    C = pd.read_csv(f"{OUT}_C_real_fill_pairs.csv")
    C70 = C[C["T"] == 70][["key", "has_earlier", "gap_h", "price_gap_pct", "early_today"]].rename(
        columns=lambda c: c if c == "key" else "c70_" + c)
    O = O.merge(C70, on="key", how="left")
    H = pd.read_csv(f"{OUT}_D_hybrid_rows.csv")
    for v in ("T85", "T70", "T50"):
        O = O.merge(H[H.variant == v][["key", "status"]].rename(columns={"status": f"d_{v}_status"}), on="key", how="left")
    O.to_csv(f"{OUT}.csv", index=False)
    print(len(O), O.columns.tolist()[:12])
