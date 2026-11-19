#!/usr/bin/env python3
"""FRENZY LONG SEARCH — charts: _tape.png (regime calendar + current tape + analogs) and _event.png (event studies of the tradable
entries — FLAG at first detection, SURVIVOR / CARRY checkpoints — winners vs losers by BTC daily regime; detection lag; earlier-detector
frequency vs edge frontier)."""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data"); OUT = os.path.join(R, "reports", "FRENZY_LONG_SEARCH_2026-11-19")
C1, C2, C3, G = "#2a6fdb", "#d1495b", "#3a9d5d", "#888888"


def tape():
    T = pd.read_csv(os.path.join(D, "tape_daily.csv"), index_col=0)
    T = T[T.index >= 19358]
    A = pd.read_csv(OUT + "_analogs.csv", index_col=0)
    x = pd.to_datetime(T.index * 86400000, unit="ms")
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(x, T.btc_c, color="black", lw=1); ax[0].set_yscale("log"); ax[0].set_ylabel("BTC (log)")
    on = T.R1.values == 1
    for a in ax:
        a.fill_between(x, 0, 1, where=on, color=C3, alpha=0.08, transform=a.get_xaxis_transform(), lw=0)
        an = T.index.isin(A.index[A.analog])
        a.fill_between(x, 0, 1, where=an, color=C2, alpha=0.25, transform=a.get_xaxis_transform(), lw=0)
    ax[0].set_title("Tape 2023 → 2026-11-17 · green = BTC daily EMA50 > EMA200 (R1 ON) · red = days most like the last 14 days (analogs)")
    ax[1].plot(x, T.breadth, color=C1, lw=1); ax[1].axhline(50, color=G, ls=":"); ax[1].set_ylabel("% perps > daily EMA50")
    ax[2].plot(x, T.spikes7, color=C2, lw=1); ax[2].set_ylabel("Frenzy spikes / day (7-d avg)")
    for a in ax:
        a.axvline(x[-1], color="black", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(OUT + "_tape.png", dpi=110); plt.close(fig)


def band(a, P, xs, col, lab):
    m = np.nanmean(P, 0); md = np.nanmedian(P, 0)
    a.plot(xs, m, color=col, lw=1.6, label=lab + f" mean (n {len(P)})"); a.plot(xs, md, color=col, lw=1, ls="--", label=lab + " median")


def event():
    fe = pickle.load(open(os.path.join(D, "flag_eval.pkl"), "rb"))
    P = fe["paths"]; xs = np.arange(-24, 289) * 5 / 60
    ev = pickle.load(open(os.path.join(D, "eval.pkl"), "rb"))
    ee = pickle.load(open(os.path.join(D, "early_eval.pkl"), "rb"))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for j, r1 in enumerate((1, 0)):
        a = ax[0, j]
        s = P[P.R1 == r1]
        W = np.vstack(s[s.TR_g > 0].path.values); L = np.vstack(s[s.TR_g <= 0].path.values)
        band(a, W, xs, C3, "TR winners"); band(a, L, xs, C2, "TR losers")
        band(a, np.vstack(s.path.values), xs, C1, "all")
        a.axhline(0, color=G, lw=0.7); a.axvline(0, color=G, lw=0.7)
        a.set_title(f"FLAG: long at first detection · BTC daily {'BULL (R1 ON)' if r1 else 'BEAR (R1 OFF)'}\n% vs entry, hours after entry")
        a.set_ylim(-15, 15); a.legend(fontsize=7)
    a = ax[0, 2]
    F = fe["F"]
    a.hist(F.moved.clip(-10, 80), bins=60, color=C1, alpha=0.8)
    a.axvline(10, color=G, ls=":"); a.axvline(20, color=G, ls=":")
    a.set_title(f"Detection lag: % already moved from the 6 h base at the flag\nmedian {F.moved.median():.1f} % · "
                f"median {F.lag_min.median():.0f} min after the base · {(F.late_min > 0).mean()*100:.0f} % flagged after the spike bar (shortlist)")
    xs2 = np.arange(-24, 169)
    for j, (fam, ck, ex) in enumerate((("SURVIVOR", 24, "H72_S20_Tx"), ("CARRY", 6, "H72_Sx_Tx"))):
        a = ax[1, j]; d = ev["store"][(fam, ck, ex)]
        for r1, col in ((1, C3), (0, C2)):
            s = d[d.R1 == r1]
            if len(s):
                band(a, np.vstack(s.path.values), xs2, col, f"R1={'ON' if r1 else 'OFF'}")
        a.axhline(0, color=G, lw=0.7); a.axvline(0, color=G, lw=0.7); a.set_ylim(-30, 30)
        a.set_title(f"{fam}_{ck}h entries · hourly path −24 h → +168 h (% vs entry)\nmean (solid) vs median (dashed), by BTC daily regime")
        a.legend(fontsize=7)
    a = ax[1, 2]; C = ee["C"]
    for ex, col in (("Q3", C1), ("Q5", "#7b4fb5"), ("TR", C2), ("T1", "#c98b1c"), ("T4", C3)):
        s = C[C.exit == ex]
        a.scatter(s.trades_day, s.avg, s=12, color=col, alpha=0.35)
        a.scatter(s.K10_trades_day, s.K10_avg, s=16, color=col, marker="x", label=f"{ex} (x = 10-position book)")
    a.axhline(0, color=G, lw=0.7); a.set_xscale("log")
    a.set_title("Earlier-detector grid (27 triggers × 5 exits): trades/day vs avg net %\ndots = every signal, x = what a 10-slot book can hold")
    a.set_xlabel("trades / day"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT + "_event.png", dpi=105); plt.close(fig)


if __name__ == "__main__":
    tape(); event(); print("ok")
