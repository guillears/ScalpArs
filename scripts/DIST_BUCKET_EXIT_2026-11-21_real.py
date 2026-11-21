#!/usr/bin/env python3
"""DIST_BUCKET_EXIT — real DISTRIBUTION_SHORT fills re-priced on the TP × SL grid (PREREG reports/DIST_BUCKET_EXIT_2026-11-21_PREREG.md).
Reuses scripts/DIST_THEORY_EXIT_2026-11-21_real.py (fills from every exports_archive export, dedup, real fill price, 1m path from the minute
after the fill, real funding, its kline cache) with the grid engine of scripts/DIST_BUCKET_EXIT_2026-11-21_sim.py; r15 from DIST_AFTERCRASH
real_fills.csv; bucket exit = the DESIGN picks in reports/DIST_BUCKET_EXIT_2026-11-21_data/picks.csv. Research only.
Usage: venv/bin/python -I scripts/DIST_BUCKET_EXIT_2026-11-21_real.py"""
import os, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


BS = load("DBE_SIM", "DIST_BUCKET_EXIT_2026-11-21_sim.py")
RL = load("DBE_REAL", "DIST_THEORY_EXIT_2026-11-21_real.py")
RL.SM = BS.SM
D = os.path.join(R, "reports", "DIST_BUCKET_EXIT_2026-11-21_data")
RL.D = D
EDGES = [-15, -10, -5]
BL = ["≤ −15", "−15 … −10", "−10 … −5", "> −5"]


def main():
    try:
        RL.main()   # writes real_fills.csv first; its REF-named printout does not apply to grid cell names
    except KeyError:
        pass
    T = pd.read_csv(os.path.join(D, "real_fills.csv"))
    A = pd.read_csv(os.path.join(R, "reports", "DIST_AFTERCRASH_2026-11-21_data", "real_fills.csv"))
    A["opened"] = A.opened.str[5:16]
    T = T.merge(A[["pair", "opened", "r15"]], on=["pair", "opened"], how="left")
    P = pd.read_csv(os.path.join(D, "picks.csv")).set_index("bucket").pick
    T["bucket"] = [BL[int(np.digitize(x, EDGES, right=True))] for x in T.r15]
    T["bucket_cell"] = [P[b] for b in T.bucket]
    T["bucket_net"] = [r[c] for r, c in zip(T.to_dict("records"), T.bucket_cell)]
    T["bucket_kind"] = [r[c + "_k"] for r, c in zip(T.to_dict("records"), T.bucket_cell)]
    T.to_csv(os.path.join(D, "real_fills.csv"), index=False)
    print(T[["pair", "opened", "booked", "booked_reason", "r15", "bucket", "TPT_SLT", "TPT_SLT_k", "bucket_cell", "bucket_net", "bucket_kind",
             "TP15_SL10", "TP15_SL10_k", "TP12_SL8", "TP15_SLT", "TPT_SL10"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
