#!/usr/bin/env python3
"""HT_WILLY 2026-11-20 — event-study chart. Research only.
Top row: mean 1m close path (% from the entry price), −60 … +360 min around the entry — REAL entries vs RANDOM draws placed AFTER the real signal
(clean control) vs RANDOM draws placed BEFORE it (the biased draws that sit in the hindsight-green stretch). Bottom row: real winners vs losers on
today's exit. Paths are hindsight (they continue after the trade's exit). Output reports/HT_WILLY_2026-11-20_chart.png
Usage: venv/bin/python -I scripts/ht_willy_chart.py"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ht_willy_sim as W  # noqa: E402
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e1"
x = np.arange(-W.PATH_PRE, W.PATH_POST)
C = pd.read_pickle(os.path.join(W.D, "cohorts.pkl"))
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
fig.patch.set_facecolor("#fcfcfb")
for k, sl in enumerate(("WILLY", "HT")):
    Z = C[sl]
    real = np.vstack(Z.cpath.values)
    pre, post = [], []
    for _, r in Z.iterrows():
        for i, p in enumerate(r.rnd_paths):
            if r.C_EP_f[i, 0] != r.ts:   # the real entry minute itself is neither (as ht_willy_postsignal: post = ts > entry)
                (pre if r.C_EP_f[i, 0] < r.ts else post).append(p)
    pre, post = np.array(pre), np.array(post)
    a = ax[0, k]
    for arr, col, lab in ((real, BLUE, f"real entry (n {len(real)})"), (post, AQUA, f"random AFTER the signal (n {len(post)})"),
                          (pre, ORANGE, f"random BEFORE the signal (n {len(pre)})")):
        m = np.nanmean(arr, 0); a.plot(x, m, color=col, lw=2, label=lab + f"  · at +360 min {m[-1]:+.2f}")
    a.set_title(f"{sl}: entry vs random entry in the same episode (2025-01 → 2026-11)", fontsize=11, color=INK, loc="left")
    w = Z.ref_net.values > 0
    b = ax[1, k]
    for msk, col, lab in ((w, BLUE, f"winners on today's exit (n {w.sum()})"), (~w, ORANGE, f"losers (n {(~w).sum()})")):
        m = np.nanmean(real[msk], 0); b.plot(x, m, color=col, lw=2, label=lab)
        b.annotate(f"{m[-1]:+.2f}", (x[-1], m[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=9, va="center")
    b.set_title(f"{sl}: real winners vs losers", fontsize=11, color=INK, loc="left")
    for a_ in (a, b):
        a_.axvline(0, color=INK2, lw=1, ls=":"); a_.axhline(0, color=INK2, lw=0.8)
        a_.grid(color=GRID, lw=0.8); a_.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.tick_params(colors=INK2, labelsize=9); a_.set_ylabel("close vs entry price, %", color=INK2, fontsize=9)
        a_.legend(frameon=False, fontsize=9, loc=("lower right" if a_ is a else "upper left"))
    b.set_xlabel("minutes from entry (hindsight path; WILLY exits by 90 min, HT by 24 h)", color=INK2, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(W.RP, "HT_WILLY_2026-11-20_chart.png"), dpi=130)
print("ok")
