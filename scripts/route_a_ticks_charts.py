#!/usr/bin/env python3
"""ROUTE A — CHARTS. ① runner vs fizzle vs baseline distributions of the tick features at C0 (DESIGN) · ② per-feature AUC by checkpoint
against the family-wise null p95 · ③ event study: population-weighted price path after the entry by hindsight class and by predicted
class (DESIGN-OOF and HOLDOUT, top-20 % BOTH score vs rest). Reads data/eval.pkl; writes reports/ROUTE_A_TICKS_2026-11-19_*.png."""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "ROUTE_A_TICKS_data"); OUT = os.path.join(R, "reports", "ROUTE_A_TICKS_2026-11-19")
C1, C2, CG, INK, MUTED = "#2a78d6", "#eb6834", "#8a8a85", "#1f1f1e", "#6b6b66"
MINS = (1, 2, 5, 10, 15, 30, 60, 120, 240)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6})


def ecdf(ax, v, c, lab, ls="-"):
    v = np.sort(v[np.isfinite(v)])
    if len(v):
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2, ls=ls, label=f"{lab} (n={len(v)})")


def main():
    res = pickle.load(open(os.path.join(D, "eval.pkl"), "rb"))
    E, BL, A = res["E"], res["BL"], res["A"]
    g = E[(E.block == "DESIGN") & (E.chk == 0)]; b = BL[BL.block == "DESIGN"]
    feats = ["tps_accel", "tps_burst", "buy_qv", "buy_qv_last", "big_lift", "retrace", "last_conc", "spread_bps",
             "med_trade", "run_ratio", "uptick_share", "vwap_gap"]
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    for ax, f in zip(axs.flat, feats):
        lo, hi = np.nanpercentile(pd.concat([g[f], b[f]]), [1, 99])
        ecdf(ax, g.loc[g.runner == 1, f].clip(lo, hi).values, C1, "runner")
        ecdf(ax, g.loc[g.runner == 0, f].clip(lo, hi).values, C2, "fizzle")
        ecdf(ax, b[f].clip(lo, hi).values, CG, "random moment", "--")
        a = A[(A.chk == 0) & (A.feature == f)].auc_runner.iloc[0]
        ax.set_title(f"{f}   AUC {a:.3f}", fontsize=9, color=INK); ax.set_ylabel("cumulative share")
    axs.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Tick features at the trigger close (C0), DESIGN 2024-01 → 2026-06: runners vs fizzles vs random moments (ECDF)", color=INK)
    fig.tight_layout(); fig.savefig(OUT + "_features.png", dpi=130); plt.close(fig)

    # ② AUC by checkpoint
    fig, ax = plt.subplots(figsize=(12, 5))
    T = A[A.side == "tick"].pivot(index="feature", columns="chk", values="auc_runner")
    Cn = A[A.side == "candle"].pivot(index="feature", columns="chk", values="auc_runner")
    allf = pd.concat([T, Cn]); x = np.arange(len(allf))
    for k, (cs, mk) in enumerate(zip((0, 60, 120, 300), ("o", "s", "^", "D"))):
        ax.scatter(x + (k - 1.5) * 0.15, allf[cs].values, marker=mk, s=28, color=[C1] * len(T) + [C2] * len(Cn),
                   label=f"C{cs // 60}", edgecolor="white", linewidth=0.6)
    ax.axhline(0.5, color=MUTED, lw=1)
    nb = res["fw_null95"]
    ax.axhspan(0.5 - nb, 0.5 + nb, color="#e6e5e0", alpha=0.6, zorder=0, label="family-wise null 95 % band (tick)")
    ax.axvline(len(T) - 0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(allf.index, rotation=60, ha="right")
    ax.set_ylabel("AUC (runner vs fizzle)"); ax.legend(frameon=False, ncol=5, fontsize=8)
    ax.set_title("Per-feature AUC, DESIGN — tick features (blue, left) vs candle features (orange, right), by checkpoint", color=INK)
    fig.tight_layout(); fig.savefig(OUT + "_auc.png", dpi=130); plt.close(fig)

    # ③ event study
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    ts = res["trade_sets"]

    def path(ax, d, c, lab, ls="-"):
        if not len(d):
            return
        P = np.array([np.array(v, float) for v in d.ret_path.values]); w = d.w.values
        m = np.array([np.average(P[np.isfinite(P[:, j]), j], weights=w[np.isfinite(P[:, j])]) for j in range(P.shape[1])])
        md = np.nanmedian(P, axis=0)
        ax.plot(MINS, m, color=c, lw=2, ls=ls, label=f"{lab} mean (n={len(d)})")
        ax.plot(MINS, md, color=c, lw=1, ls=":", label=f"{lab} median")
    for ax, (cs, blk, title) in zip(axs, ((0, "DESIGN-OOF", "Hindsight class, C0 (DESIGN-OOF)"), (0, "DESIGN-OOF", "Predicted class, C0 (DESIGN-OOF)"),
                                          (0, "HOLDOUT", "Predicted class, C0 (HOLDOUT)"))):
        d = ts[(cs, blk, "all")]
        if "Hindsight" in title:
            path(ax, d[d.runner == 1], C1, "runner"); path(ax, d[d.runner == 0], C2, "fizzle")
        else:
            top = ts[(cs, blk, "top20")]
            path(ax, top, C1, "top-20 % score"); path(ax, d[~d.index.isin(top.index)], C2, "rest")
        ax.axhline(0, color=MUTED, lw=1); ax.set_xscale("log"); ax.set_xticks(MINS); ax.set_xticklabels(MINS)
        ax.set_xlabel("minutes after entry (decision + 8 s), gross"); ax.set_title(title, color=INK); ax.legend(frameon=False, fontsize=7)
    axs[0].set_ylabel("% from entry price (population-weighted)")
    fig.tight_layout(); fig.savefig(OUT + "_event.png", dpi=130); plt.close(fig)
    print("charts written")


if __name__ == "__main__":
    main()
