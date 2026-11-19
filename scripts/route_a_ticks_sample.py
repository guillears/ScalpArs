#!/usr/bin/env python3
"""ROUTE A (PREREG reports/ROUTE_A_TICKS_2026-11-19_PREREG.md) — SAMPLE.
Trigger W15_R3_V10 rows from the earlier-detector grid (reports/FRENZY_LONG_SEARCH_data/early_rows.pkl), 2024-01 → 2026-11.
Stratified per month × class (RUNNER = Frenzy flag within 2 h; FIZZLE = none) in a frozen random order (seed 20261119).
Writes reports/ROUTE_A_TICKS_data/order.csv (every trigger with its class rank) and month class counts (weights).
Research only; reads local files, 0 API weight."""
import os, pickle
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "ROUTE_A_TICKS_data"); os.makedirs(D, exist_ok=True)
CELL = "W15_R3_V10"
HOLD0 = "2026-07"
N_DES, N_HOLD = 20, 35


def main():
    X = pickle.load(open(os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data", "early_rows.pkl"), "rb"))
    # every trigger of any cell (for the baseline-moment exclusion)
    allsig = X[["pair", "sig_ms"]].drop_duplicates()
    allsig.to_csv(os.path.join(D, "all_triggers.csv"), index=False)
    g = X[X.cell == CELL].copy()
    g["month"] = pd.to_datetime(g.sig_ms, unit="ms").dt.strftime("%Y-%m")
    g = g[g.month >= "2024-01"].reset_index(drop=True)
    g["runner"] = (g.spike_after > 0).astype(int)
    g["block"] = np.where(g.month >= HOLD0, "HOLDOUT", "DESIGN")
    rng = np.random.default_rng(20261119)
    g["u"] = rng.random(len(g))
    g = g.sort_values(["month", "runner", "u"])
    g["rank"] = g.groupby(["month", "runner"]).cumcount()
    g["target"] = np.where(g.block == "DESIGN", N_DES, N_HOLD)
    cnt = g.groupby(["month", "runner"]).size().rename("pop_n").reset_index()
    g = g.merge(cnt, on=["month", "runner"])
    keep = ["pair", "sig_ms", "entry_ms", "spike_after", "month", "block", "runner", "rank", "target", "pop_n", "px_ratio_flag",
            "TR_g", "Q3_g", "T1_g", "fp", "peak24"]
    g[keep].to_csv(os.path.join(D, "order.csv"), index=False)
    print(cnt.pivot(index="month", columns="runner", values="pop_n").to_string())
    print("triggers", len(g), "calendar days", (g.sig_ms.max() - g.sig_ms.min()) / 86_400_000)


if __name__ == "__main__":
    main()
