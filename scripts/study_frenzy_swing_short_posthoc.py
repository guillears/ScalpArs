#!/usr/bin/env python3
"""FRENZY swing short — POST-HOC diagnostics (NOT pre-registered; descriptive only, cannot support a ship):
  A  where in the episode is a short better? every flagged bar ≥ 2 h after the spike, sampled hourly (bar index ≡ 0 mod 12), same frozen
     exit (24 h, 10 % stop, no TP, 0.20 % costs, no funding), bucketed by the close vs the anchored VWAP at entry. Year block only.
  B  Oct-1 replication: the E1 primary cell restricted to big pumps (run from the pre-pump price to the episode peak ≥ 30 % / ≥ 100 %),
     i.e. closer to the Oct-1 'frenzy' (+30 % / 24 h) cohort.
Prints markdown; writes reports/FRENZY_SWING_SHORT_2026-10-20_posthoc.csv (A rows)."""
import os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
_a, sys.argv = sys.argv, [sys.argv[0]]
import study_frenzy_swing_short as S  # noqa: E402
sys.argv = _a
eps = pickle.load(open(os.environ.get("EPS", os.path.join(S.D, "episodes.pkl")), "rb"))
rows = []
for k, e in enumerate(eps):
    on, off = e["on_i"], e["off_i"]
    ii = np.arange(on + 24, min(off, len(e["t"]) - 2), 12)
    ii = ii[(e["vol24"][ii] >= 20e6) & e["seen"][ii]]
    if not len(ii):
        continue
    for i in ii:
        rows.append((k, int(i), e["pair"], int(e["t"][i]) + S.BAR, e["c"][i], e["vwap"][i], e["base"]))
A = pd.DataFrame(rows, columns=["ep", "i", "pair", "t_ms", "c", "vwap", "base"])
A["dt"] = pd.to_datetime(A.t_ms, unit="ms"); A = A[(A.dt >= S.B_YEAR0) & (A.dt < S.B_MAST0)].reset_index(drop=True)
H, Lo, O, C, T = S.paths(eps, A.ep.values, A.i.values)
E = A.c.values * (1 - S.SLIP)
ret, kx, comp, reason = S.simulate(E, A.base.values, H, Lo, O, C, 24, 10.0, "none")
A["net"] = ret - S.FEES; A["day"] = A.dt.dt.strftime("%Y-%m-%d"); A["reason"] = reason
A["vs"] = (A.c / A.vwap - 1) * 100
A["bucket"] = pd.cut(A.vs, [-100, -10, -5, -2, 0, 2, 5, 10, 1e9], labels=["≤−10", "−10…−5", "−5…−2", "−2…0", "0…+2", "+2…+5", "+5…+10", ">+10"])
A.to_csv(S.OUT + "_posthoc.csv", index=False)
print("### A · short at any hourly flagged bar, by close vs the spike VWAP (24 h · 10 % stop · no TP · 0.20 % costs · no funding · year block)\n")
print("| close vs VWAP | bars | episodes | days | WR | avg % | 95 % day-boot | stopped |"); print("|---|---|---|---|---|---|---|---|")
for b, g in A.groupby("bucket", observed=True):
    lo, hi = S.boot_ci(g)
    print(f"| {b} | {len(g)} | {g.ep.nunique()} | {g.day.nunique()} | {(g.net > 0).mean() * 100:.0f}% | {g.net.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(g.reason == 'stop').mean() * 100:.0f}% |")
for nm, m in (("below VWAP (all)", A.vs < 0), ("at / above VWAP (all)", A.vs >= 0)):
    g = A[m]; lo, hi = S.boot_ci(g)
    print(f"| **{nm}** | {len(g)} | {g.ep.nunique()} | {g.day.nunique()} | {(g.net > 0).mean() * 100:.0f}% | {g.net.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {(g.reason == 'stop').mean() * 100:.0f}% |")

X = pd.read_pickle(os.path.join(S.D, "cells.pkl"))
X = X[(X.cell == S.cell_name(*S.PRIMARY)) & (X.block == "3_year")].copy()
run = {k: (np.nanmax(eps[k]["h"][eps[k]["on_i"]:eps[k]["off_i"]]) / eps[k]["base"] - 1) * 100 for k in X.ep.unique()}
X["run"] = X.ep.map(run)
print("\n### B · Oct-1 replication — E1 · 24 h · 10 % stop · no TP, by the episode's run (pre-pump → peak), after costs + funding\n")
print("| run | N | days | WR | avg % | 95 % day-boot | avg before funding | 2025 | 2026H1 | 2026H2 |"); print("|---|---|---|---|---|---|---|---|---|---|")
for nm, m in (("< 30 %", X.run < 30), ("≥ 30 %", X.run >= 30), ("≥ 100 %", X.run >= 100)):
    g = X[m]; lo, hi = S.boot_ci(g)
    per = " | ".join(f"{g[g.period == p].net.mean():+.2f}" for p in ("2025", "2026H1", "2026H2"))
    print(f"| {nm} | {len(g)} | {g.day.nunique()} | {(g.net > 0).mean() * 100:.0f}% | {g.net.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | {g.net_nofund.mean():+.2f} | {per} |")
