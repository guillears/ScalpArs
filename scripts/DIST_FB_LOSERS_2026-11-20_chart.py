#!/usr/bin/env python3
"""DIST_FB_LOSERS_2026-11-20 — event-study chart: price paths anchored at the SIGNAL close, short-positive (% the short is ahead).
Top: DISTRIBUTION_SHORT — backtest band (1,547 DL 586 fills, DIST_EXIT 5-min close paths: median + IQR of winners vs losers) and the
5 live fills (losers vs the winner) with the TP line (+10 %) and each fill's stop line. Bottom: FLATBASE_SHORT — the 3 live fills with
TP +5 % / stop −10 % (no backtest path band stored for the FB rebuild; stated in the report).
Output: reports/DIST_FB_LOSERS_2026-11-20_chart.png
Usage: venv/bin/python -I scripts/DIST_FB_LOSERS_2026-11-20_chart.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data")
WIN, LOS, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def main():
    P = pd.read_pickle(os.path.join(D, "paths.pkl"))
    z = np.load(os.path.join(ROOT, "reports", "DIST_EXIT_data", "paths_L1.npz"))
    S = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    R = pd.read_pickle(os.path.join(ROOT, "reports", "DIST_WL_2026-11-18_data", "rows.pkl")).reset_index(drop=True)
    key = dict(zip(R.pair.astype(str) + "|" + R.entry_ms.astype(str), range(len(R))))
    pos = dict(zip(z["idx"], range(len(z["idx"]))))
    rows = [pos.get(key.get(f"{p}|{t}"), None) for p, t in zip(S.pair.astype(str), S.entry_ms.astype(str))]
    ok = np.array([r is not None for r in rows])
    PP = z["p"][[r for r in rows if r is not None]]
    win = (S.net.values > 0)[ok]
    th = (np.arange(PP.shape[1]) - 24) * 5 / 60          # hours from the signal close
    keep = th <= 24

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), facecolor=SURF)
    ax = axes[0]
    ax.set_facecolor(SURF)
    for m, col, nm in ((win, WIN, "backtest winners"), (~win, LOS, "backtest losers")):
        q = np.nanpercentile(PP[m][:, keep], [25, 50, 75], axis=0)
        ax.fill_between(th[keep], q[0], q[2], color=col, alpha=0.12, lw=0)
        ax.plot(th[keep], q[1], color=col, lw=2, ls="--", label=f"{nm} (median, IQR band; N {int(m.sum())})")
    for k, v in P.items():
        sl, pair, _ = k.split("|")
        if sl != "DISTRIBUTION_SHORT":
            continue
        h = v["t_min"] / 60
        c = WIN if v["won"] else LOS
        ax.plot(h, v["p"], color=c, lw=1.6)
        ci = int(np.searchsorted(v["t_min"], v["close_min"]))
        ci = min(ci, len(h) - 1)
        ax.plot(h[ci], v["p"][ci], "o", ms=8, color=c, mec=SURF, mew=2)
        off = {"NAORISUSDT": (-10, -26), "SENTUSDT": (8, 14), "KERNELUSDT": (10, -22), "EVAAUSDT": (6, -14)}.get(pair, (6, 6))
        ax.annotate(pair.replace("USDT", ""), (h[ci], v["p"][ci]), xytext=off, textcoords="offset points", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.hlines(v["stop"], v["fill_min"] / 60, v["close_min"] / 60, color=c, lw=1, ls=":")
    ax.axhline(10, color=MUTED, lw=1, ls="-.")
    ax.text(23.8, 10.4, "TP +10 % (≈ from the fill)", ha="right", fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-1, 24); ax.set_ylim(-26, 22)
    ax.set_title("DISTRIBUTION_SHORT: live fills (solid; dot = exit, dotted = stop line) vs the backtest", loc="left", color=INK, fontsize=12)
    ax.set_ylabel("% in the short's favour, from the signal close", color=MUTED)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.set_facecolor(SURF)
    for k, v in P.items():
        sl, pair, _ = k.split("|")
        if sl != "FLATBASE_SHORT":
            continue
        h = v["t_min"] / 60
        c = WIN if v["won"] else LOS
        ax.plot(h, v["p"], color=c, lw=1.6, label=f"{pair.replace('USDT', '')} ({'win' if v['won'] else 'loss'})")
        ci = min(int(np.searchsorted(v["t_min"], v["close_min"])), len(h) - 1)
        ax.plot(h[ci], v["p"][ci], "o", ms=8, color=c, mec=SURF, mew=2)
        ax.annotate(pair.replace("USDT", ""), (h[ci], v["p"][ci]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=INK)
    ax.axhline(5, color=MUTED, lw=1, ls="-.")
    ax.text(71.5, 5.6, "TP +5 %", ha="right", fontsize=9, color=MUTED)
    ax.axhline(-10, color=MUTED, lw=1, ls=":")
    ax.text(71.5, -9.4, "stop −10 %", ha="right", fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-1, 72); ax.set_ylim(-40, 20)
    ax.set_title("FLATBASE_SHORT: the 3 live fills (MMT and PHAROS opened on the same 4-h close)", loc="left", color=INK, fontsize=12)
    ax.set_xlabel("hours from the signal close", color=MUTED)
    ax.set_ylabel("% in the short's favour", color=MUTED)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_chart.png"), dpi=130, facecolor=SURF)


if __name__ == "__main__":
    main()
