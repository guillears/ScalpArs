#!/usr/bin/env python3
"""ROUTE A FOLLOW-UP — charts: ① small-TP cost / WR frontier (observed WR vs breakeven WR per grid cell, C0, DESIGN, all three
selections) · ② per-month full-population all-trigger net % (Q3 / TR / T1, 1m-priced) with 2026-H2 shaded."""
import os, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_2026-11-19")
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)


def frontier():
    G = pd.read_csv(OUT + "_q3_grid.csv")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    cols = {0.5: S1, 1.0: S2, 1.5: S3, 2.0: S4}
    for ax, sel, ttl in zip(axs, ("all", "top20", "S_VW"), ("(i) all sampled triggers", "(ii) Route A top-20 % score", "(iii) Q2 VWAP-hold top quintile")):
        d = G[(G.select == sel) & (G.block == "DESIGN") & (G.chk == 0)]
        for tp, g in d.groupby("tp"):
            ax.scatter(g.be_wr, g.wr, s=18 + g.sl * 14, color=cols[tp], edgecolor="white", lw=1, label=f"TP {tp} %", zorder=3)
        ax.plot([20, 95], [20, 95], color=MUTED, lw=1, ls="--", zorder=2)
        ax.text(88, 84, "breakeven", color=MUTED, fontsize=8, rotation=38, ha="center")
        ax.set_title(ttl, fontsize=10, color=INK, loc="left"); style(ax)
        ax.set_xlabel("breakeven WR after costs (%)", fontsize=9, color=INK)
    axs[0].set_ylabel("observed WR (%)", fontsize=9, color=INK)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left", title="marker size = stop 0.5 … 3 %", title_fontsize=7)
    fig.suptitle("Small take-profits: observed vs breakeven WR per grid cell — above the dashed line = profitable (C0 decision, DESIGN 2024-01 → 2026-06, slip 0.10 %, ticks)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT + "_frontier.png", dpi=130); plt.close(fig)


def monthly():
    M = pd.read_csv(OUT + "_q1a_month.csv")
    m = M[(M["pop"] == "all") & (M.slip == 0.10)]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    months = sorted(m.month.unique()); xi = {mo: i for i, mo in enumerate(months)}
    h2 = [xi[mo] for mo in months if mo >= "2026-07"]
    for a in (ax, ax2):
        if h2:
            a.axvspan(min(h2) - 0.5, max(h2) + 0.5, color="#f3efe2", zorder=0)
    for ex, c in (("Q3", S1), ("TR", S2), ("T1", S3)):
        g = m[m.exit == ex].set_index("month").reindex(months)
        ax.plot(range(len(months)), g.avg.values, color=c, lw=2, marker="o", ms=4, label=ex, zorder=3)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_ylabel("avg net % per trade (all triggers)", fontsize=9, color=INK); style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    ax.text(np.mean(h2) if h2 else 0, ax.get_ylim()[1] * 0.92, "HOLDOUT 2026-H2", ha="center", fontsize=8, color=MUTED)
    n = m[m.exit == "TR"].set_index("month").reindex(months)
    ax2.bar(range(len(months)), n.N.values, color=S1, width=0.7)
    ax2.set_ylabel("TR trades", fontsize=9, color=INK); style(ax2)
    ax2.set_xticks(range(len(months))); ax2.set_xticklabels(months, rotation=60, fontsize=7)
    fig.suptitle("Full trigger population (W15_R3_V10, 1m-priced, 8 s + 0.10 % slip, fees + funding, one position per coin) — per month",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT + "_monthly.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    frontier()
    if os.path.exists(OUT + "_q1a_month.csv"):
        monthly()
