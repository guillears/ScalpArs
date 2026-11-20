#!/usr/bin/env python3
"""DIST_FB_LOSERS_2026-11-20 — shuffled null for scripts/sweep_separators.py on a sleeve's REAL fills.

Runs the sweep's own pure functions (prepare / candidates / sweep_1d / sweep_2d stats_only) on the real cohort, then on N_SHUF copies
with the target column shuffled across fills (eras and every entry_* value kept). Reports observed vs null for: the number of
cross-era-consistent 1D splits, the best 1D strength, the number of consistent 2D cells and the best 2D strength.
A sweep 'survivor' is only worth a look if the observed count / strength beats the null's 95th percentile.
Usage: venv/bin/python -I scripts/DIST_FB_LOSERS_2026-11-20_sweepnull.py FLATBASE_SHORT pnl|reach|path [N_SHUF]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import sweep_separators as SS  # noqa: E402

warnings.filterwarnings("ignore")


def stats(df, cands, levels):
    r = SS.sweep_1d(df, cands, levels)
    c = r[r.consistent.astype(bool)]
    n2, b2 = SS.sweep_2d(df, r, cands, stats_only=True)
    return len(c), (float(c.strength.max()) if len(c) else 0.0), int(n2), float(b2 or 0.0)


def main():
    sleeve = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "pnl"
    nshuf = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    df, split, note = SS.prepare(sleeve, label=label)
    if df is None:
        print(note)
        return
    cands, levels, *_ = SS.candidates(df, detail=True)
    obs = stats(df, cands, levels)
    rng = np.random.default_rng(20261120)
    null = []
    for _ in range(nshuf):
        d = df.copy()
        d["pnl_percentage"] = rng.permutation(d.pnl_percentage.values)
        null.append(stats(d, cands, levels))
    null = np.array(null)
    names = ["1D consistent (count)", "1D best strength", "2D consistent (count)", "2D best strength"]
    print(note)
    print(f"shuffled null, {nshuf} label permutations · observed vs null median / 95th pct / P(null ≥ observed)")
    for i, nm in enumerate(names):
        p = float((null[:, i] >= obs[i]).mean())
        print(f"  {nm:<24} observed {obs[i]:>8.2f} · null median {np.median(null[:, i]):>8.2f} · 95th {np.percentile(null[:, i], 95):>8.2f} · p {p:.2f}")


if __name__ == "__main__":
    main()
