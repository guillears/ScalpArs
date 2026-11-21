#!/usr/bin/env python3
"""STATE_EXIT_2026-11-21 — compact per-sleeve pivot of layer-1 rules + fixed alternatives (master → batch → backtest) for the report."""
import os, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("SEC", os.path.join(HERE, "STATE_EXIT_2026-11-21_common.py"))
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)
out = []
for s in ("FB", "HT", "EXH"):
    R = pd.read_pickle(os.path.join(C.D, f"eval_{s}.pkl")); T = R["thesis"]
    T = T[~T.rule.isin(["FIXED_TP5", "FIXED_TP10"]) | (s == "EXH")]
    rows = []
    for r, g in T.groupby("rule"):
        g = g.set_index("period"); d = {"rule": r}
        for per in ("MASTER", "BATCH", "DESIGN", "HOLDOUT"):
            if per not in g.index:
                continue
            x = g.loc[per]
            if per == "MASTER":
                d.update({"M fired": int(x.fired), "M Δ": x.d_mean, "M CI": f"{x.ci_lo:+.2f}…{x.ci_hi:+.2f}", "M P(Δ>0)": x.p_pos,
                          "M saved $": x.saved_usd, "M chopped $": x.chopped_usd, "M DCR b→a": f"{x.dcr_before:+.2f}→{x.dcr_after:+.2f}"})
            else:
                d[f"{per[0]} Δ (P, n)"] = f"{x.d_mean:+.2f} ({x.p_pos:.2f}, {int(x.fired)})"
        rows.append(d)
    P = pd.DataFrame(rows)
    key = "M Δ" if "M Δ" in P else "D Δ (P, n)"
    if key == "M Δ":
        P = P.sort_values("M Δ", ascending=False)
    out += [f"\n### {s} — layer-1 pivot (M = master, B = batch, D = DESIGN, H = HOLDOUT; P = day-block bootstrap P(Δ > 0); n = fills where it fired)\n", C.md(P, 2)]
open(os.path.join(C.ROOT, "reports", "STATE_EXIT_2026-11-21_pivot.md"), "w").write("\n".join(out))
print("\n".join(out))
