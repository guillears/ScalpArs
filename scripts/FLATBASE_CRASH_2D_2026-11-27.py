#!/usr/bin/env python3
"""FLATBASE_CRASH_2D_2026-11-27 — Flatbase crash zone (BTC ≥ 20 % below its 30 d high) × EVERY rebuild feature (operator: "it filters too
many winners in backtest, and you try a 2D?"). Research only. Prior: today's FLATBASE_DEEP_DIVE crowding 2D (5 vars, family-wise p 0.84).
NEW: every numeric feature of reports/OOS_REAL_FLATBASE_data/wl_features.csv (pre-entry only — exit / label / outcome columns dropped), split
at its median over ALL 545 fills, both sides, inside the zone → the block cell = zone ∧ side. Score = cell mean (N ≥ 8); family-wise luck:
labels shuffled 1,000× INSIDE the zone, best (most negative) cell each time. Output (regenerable, not committed): reports/BATCH28_REVIEW_2026-11-27_data/
fb_crash_2d_cells.csv.
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
sp = importlib.util.spec_from_file_location("dd", os.path.join(HERE, "DIST_DEEP_DIVE_2026-11-27.py"))
DD = importlib.util.module_from_spec(sp); sp.loader.exec_module(DD)
POST = re.compile(r"^(cf_|s4_today|s4_netf|X\d|L\d|re_|pnl|stack_|acct_|close_reason|worked|reach|tp_first|complete72|label|kind|batch|mode|era|"
                  r"status|block|route|key|pair|spike|s4_t|conf_t|on_t|day|month|period|hh$|slip_x|slip_e)")   # (review) slip_e = the drawn entry slip → outcome


def main():
    b = pd.read_csv(os.path.join(R, "reports", "FLATBASE_NOPROGRESS_STOP_2026-11-27_data", "base.csv"))
    b = b[b.src == "BT"]
    w = pd.read_csv(os.path.join(R, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    feats = [c for c in w.columns if not POST.match(c) and pd.api.types.is_numeric_dtype(w[c]) and w[c].nunique() > 4]
    K = b.merge(w[["key"] + feats], left_on="ep", right_on="key", how="left")
    dd = DD.btc_dd()
    K["dd30"] = pd.to_datetime(K.te, unit="ms").dt.floor("D").map(dd)
    Z = (K.dd30 <= -20).fillna(False).values
    cells = {}
    for c in feats:
        v = pd.to_numeric(K[c], errors="coerce")
        if v[Z].notna().mean() < 0.8:
            continue
        med = v.median()
        for side, m in (("≤ median", v <= med), ("> median", v > med)):
            mm = Z & m.fillna(False).values
            if mm.sum() >= 8:
                cells[f"{c} {side} ({med:.4g})"] = mm
    obs = {k: K.net.values[m].mean() for k, m in cells.items()}
    best = min(obs, key=obs.get)
    rng = np.random.default_rng(27); zi = np.flatnonzero(Z); nb = []
    for _ in range(1000):
        y = K.net.values.copy(); y[zi] = rng.permutation(y[zi])
        nb.append(min(y[m].mean() for m in cells.values()))
    p = np.mean(np.array(nb) <= obs[best])
    print(f"features {len(feats)} · cells {len(cells)} (zone {int(Z.sum())} fills · {K.net[Z].mean():+.2f}) · worst cell: {best} → N {cells[best].sum()} · "
          f"avg {obs[best]:+.2f} · family-wise p {p:.3f} (shuffled best 5th pct {np.percentile(nb, 5):+.2f})")
    T = pd.DataFrame([dict(cell=k, N=int(m.sum()), WR=(K.net.values[m] > 0).mean() * 100, avg=obs[k]) for k, m in cells.items()]).sort_values("avg")
    print(T.head(12).round(2).to_string(index=False))
    T.to_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "fb_crash_2d_cells.csv"), index=False)
    print(f"\ncells with a NEGATIVE avg: {int((T.avg < 0).sum())} of {len(T)} · below breakeven-level (avg < 0) and N ≥ 15: {int(((T.avg < 0) & (T.N >= 15)).sum())}")


if __name__ == "__main__":
    main()
