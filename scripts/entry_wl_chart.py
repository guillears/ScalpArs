#!/usr/bin/env python3
"""ENTRY_WL — charts: anchored 5m price paths (short-favourable %, −2 h … +24 h, from the entry open) of RUNNER vs GAVE_BACK,
split by each sleeve's best entry separator (kept side vs predicted give-back side), + RUNNER share per half-year by side.
Price paths, not trade paths (they run past each exit). Output reports/ENTRY_WL_2026-11-18_{exh,dist}.png"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import entry_wl_lib as W  # noqa: E402
C_R, C_G = "#2a78d6", "#eb6834"
SEP = {"exh": ("q5", 1.157e6, "q5 (5m quote vol) ≤ $1.16M — thin coin"), "dist": ("vol24", 4.198e7, "vol24 ≤ $42M — thin coin")}
for s, title in (("exh", "EXHAUSTION_SHORT"), ("dist", "DISTRIBUTION_SHORT")):
    D = W.load(s); P = np.load(os.path.join(W.D_OUT, f"{s}_path.npy")); var, edge, lab = SEP[s]
    bad = pd.to_numeric(D[var], errors="coerce").values <= edge
    t = (np.arange(P.shape[1]) - 24) * 5 / 60
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
    for k, (nm, m) in enumerate((("kept side (not thin)", ~bad), (f"predicted give-back side ({lab})", bad))):
        a = ax[k]
        for L_, c in (("RUNNER", C_R), ("GAVE_BACK", C_G)):
            mm = m & (D.label.values == L_)
            p = P[mm]
            med = np.nanmedian(p, 0); q1, q3 = np.nanpercentile(p, [25, 75], 0)
            a.plot(t, med, color=c, lw=2, label=f"{L_.replace('_', ' ').lower()} (N {mm.sum()})")
            a.fill_between(t, q1, q3, color=c, alpha=0.15, lw=0)
        a.axvline(0, color="#888", lw=1); a.axhline(0, color="#888", lw=0.8)
        a.axhline(10, color="#999", lw=1, ls="--"); a.text(23.8, 10.3, "TP 10 %", ha="right", fontsize=8, color="#555")
        a.axhline(-np.nanmedian(D.sl_pct.values), color="#999", lw=1, ls=":")
        a.text(23.8, -np.nanmedian(D.sl_pct.values) - 0.9, f"median stop −{np.nanmedian(D.sl_pct.values):.1f} %", ha="right", fontsize=8, color="#555")
        rs = np.nanmean(D.y_rg.values[m])
        a.set_title(f"{nm}\nRUNNER share of R+GB {rs:.0%}", fontsize=10)
        a.set_xlabel("hours from entry"); a.set_ylabel("price move in our favour, % (median, IQR band)")
        a.set_ylim(-9, 16); a.set_xlim(-2, 24); a.grid(alpha=0.25, lw=0.5); a.legend(loc="upper left", fontsize=8, frameon=False)
    a = ax[2]; x = np.arange(len(W.HALVES))
    for nm, m, c, mk in (("kept side", ~bad, C_R, "o"), ("thin side", bad, C_G, "s")):
        y = [np.nanmean(D.y_rg.values[m & (D.half.values == h)]) for h in W.HALVES]
        a.plot(x, y, color=c, lw=2, marker=mk, ms=8, label=nm)
    a.axvspan(3.5, 6.5, color="#eee", zorder=0); a.text(5, 0.38, "HOLDOUT", ha="center", fontsize=9, color="#555")
    a.set_xticks(x); a.set_xticklabels(W.HALVES, fontsize=8); a.set_ylim(0.35, 0.85); a.set_ylabel("RUNNER share of RUNNER + GAVE_BACK")
    a.set_title("RUNNER share per half-year, by side", fontsize=10); a.legend(frameon=False, fontsize=8); a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"{title}: runners vs give-backs, anchored at entry, split by the best entry separator (BACKTEST)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(W.RP, f"ENTRY_WL_2026-11-18_{s}.png"), dpi=110); plt.close(fig)
    print("saved", s)
