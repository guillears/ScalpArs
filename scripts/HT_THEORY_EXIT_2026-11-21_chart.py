#!/usr/bin/env python3
"""HT_THEORY_EXIT 2026-11-21 — event-study chart (HT). Panel 1: median path after entry in units of the distance to the spike high H
(0 = entry, 1 = H), by REF outcome, real fills (master + batch) and backtest. Panel 2: how far trades got before today's exit closed them
(peak in H units) — stopped vs take-profit. → reports/HT_THEORY_EXIT_2026-11-21_chart.png"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "HT_THEORY_EXIT_2026-11-21_data")
O = pd.read_pickle(os.path.join(D, "sim.pkl"))
O = O[(O.sleeve == "HT") & O.REF.notna() & (O.dip >= 1.4)]
O["grp"] = np.where(O.REF_k == "TP", "reached +5 (TP)", np.where(O.peak_ref >= 2.0, "gave back (peak ≥ +2, then stop)", "never worked (peak < +2)"))
rel = np.arange(-12, 289) * 5
COL = {"reached +5 (TP)": "#2a7f62", "gave back (peak ≥ +2, then stop)": "#d08a1e", "never worked (peak < +2)": "#b8423c"}
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
for a, (lab, msk) in zip(ax[:2], (("Real fills (master + batch, dip ≥ 1.4)", O.kind == "REAL"), ("Backtest (300, DL 594 stack)", O.kind == "BT"))):
    G = O[msk]
    for g, c in COL.items():
        s = G[G.grp == g]
        if len(s) < 3:
            continue
        M = np.vstack(s.path_Hu.values)
        m = rel <= 360
        a.plot(rel[m] / 60, np.nanmedian(M[:, m], 0), color=c, lw=2.2, label=f"{g} · N {len(s)}")
        a.fill_between(rel[m] / 60, np.nanpercentile(M[:, m], 25, 0), np.nanpercentile(M[:, m], 75, 0), color=c, alpha=0.12)
    a.axhline(1, color="k", ls="--", lw=1); a.text(5.9, 1.03, "spike high H", ha="right", fontsize=9)
    a.axhline(0, color="grey", lw=0.8); a.axvline(0, color="grey", lw=0.8)
    a.set_ylim(-2.5, 2.5); a.set_xlabel("hours after entry (hindsight path, continues after the exit)")
    a.set_ylabel("price, in units of (H − entry)"); a.set_title(lab, fontsize=11); a.legend(fontsize=8, loc="lower left")
b = ax[2]
bins = np.linspace(0, 3, 31)
for lab, msk, c in (("stopped at −4 (REF STOP)", O.REF_k == "STOP", "#b8423c"), ("take profit +5 (REF TP)", O.REF_k == "TP", "#2a7f62")):
    v = O[msk].peak_ref_Hu.clip(upper=2.99)
    b.hist(v, bins=bins, alpha=0.55, color=c, label=f"{lab} · N {len(v)}", density=True)
b.axvline(1, color="k", ls="--", lw=1)
b.set_xlabel("highest price before today's exit closed the trade, in units of (H − entry)")
b.set_ylabel("share of fills (density)"); b.set_title("Where trades topped out (real + backtest, dip ≥ 1.4)", fontsize=11); b.legend(fontsize=8)
fig.suptitle("FRENZY_WILLY_HT — does the spike high act as the ceiling of the trade?", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "reports", "HT_THEORY_EXIT_2026-11-21_chart.png"), dpi=110)
st = O.groupby(["kind", O.REF_k]).agg(N=("REF", "size"), touchH=("touchH", "mean"), peakHu_med=("peak_ref_Hu", "median"))
print(st.round(2))
print("stopped fills that touched H:", O[O.REF_k == "STOP"].groupby("kind").touchH.mean().round(2).to_dict())
