#!/usr/bin/env python3
"""DIST_PEAKSTOP — extras (descriptive, after the HOLDOUT read): ① the "does the market respect the pump top?" test — fills whose price trades
through the stop level (pump top +1 % vs a shuffled level of the same width), and what the SAME fill does afterwards with the stop moved to 15 %;
② per-half Δ; ③ chart. Research only. → reports/DIST_PEAKSTOP_2026-11-21_data/{toptest.csv, halves.csv} · reports/DIST_PEAKSTOP_2026-11-21.png"""
import os, importlib.util
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


EV = _load("DTE_EVAL", "DIST_THEORY_EXIT_2026-11-21_eval.py")
D = os.path.join(R, "reports", "DIST_PEAKSTOP_2026-11-21_data")
DAY = 86_400_000
USD = 3050.0


def main():
    S = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    Y = dict(np.load(os.path.join(D, "sim.npz"), allow_pickle=False))
    names = [str(x) for x in Y["names"]]; ci = {n: j for j, n in enumerate(names)}
    slip = S.slip.values.astype(float)
    Nn = EV.nets(slip, Y["r"].astype(float), Y["k"], Y["f"].astype(float))
    T = S.entry_ms.values.astype(np.int64); per = np.where(T < EV.SPAN["DESIGN"][1], "DESIGN", "HOLDOUT"); hv = EV.half_of(T)
    k = Y["k"]; st = Y["stops"]
    # ① level-break test: fills stopped at a level BELOW 15 % (so the 15 % fixed stop is strictly wider), above today's stop
    rows = []
    w15 = ci["W15"]
    for lab, cells in (("pump top +1 % (Q15)", ["Q15"]), ("shuffled level, same widths (S15_1..5)", [f"S15_{i}" for i in range(1, 6)]),
                       ("distribution-candle high +1 % (DQ15)", ["DQ15"])):
        for p in ("DESIGN", "HOLDOUT", "ALL"):
            pm = (per == p) if p != "ALL" else np.ones(len(T), bool)
            tp_after, net_after, n_, lvl = [], [], 0, []
            for c in cells:
                j = ci[c]
                m = pm & (k[:, j] == 1) & (st[:, j] < 14.99) & (st[:, j] > np.minimum(1.5 * S.atr15_pct.values, 6.0) + 1e-9)
                n_ += m.sum(); tp_after.append((k[m, w15] == 0).astype(float)); net_after.append(Nn[m, w15] - Nn[m, j]); lvl.append(st[m, j])
            tp_after = np.concatenate(tp_after); net_after = np.concatenate(net_after); lvl = np.concatenate(lvl)
            rows.append(dict(level=lab, period=p, broke_n=int(n_ / len(cells)), median_level=np.median(lvl),
                             share_TP_after_break=tp_after.mean() * 100, share_15stop_after=np.nan,
                             avg_gain_if_held_to_15=net_after.mean()))
    TT = pd.DataFrame(rows)
    # add the 15 % stop share
    TT.to_csv(os.path.join(D, "toptest.csv"), index=False)
    pd.set_option("display.width", 250)
    print(TT.round(2).to_string(index=False))
    # ② halves
    H = []
    for c in ("P12", "Q12", "Q8", "Q15", "W12", "W15", "S12_1"):
        j = ci[c]
        for h_, _, _ in EV.HALVES:
            m = hv == h_
            H.append(dict(cell=c, half=h_, N=int(m.sum()), ref=Nn[m, 0].mean(), cell_avg=Nn[m, j].mean(), d=(Nn[m, j] - Nn[m, 0]).mean()))
    H = pd.DataFrame(H); H.to_csv(os.path.join(D, "halves.csv"), index=False)
    print(H.pivot(index="half", columns="cell", values="d").round(2).to_string())
    # ③ chart
    E = pd.read_csv(os.path.join(D, "eval.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    top = (Y["PK"] / Y["E0_bt"] - 1) * 100
    rf = pd.read_csv(os.path.join(D, "real_fills.csv"))
    ax[0].hist(np.clip(top[np.isfinite(top)], 0, 40), bins=80, color="#4a6fa5", alpha=.85)
    for c_, lab in ((np.median(np.minimum(1.5 * S.atr15_pct.values, 6)), "today's stop (median)"),):
        ax[0].axvline(c_, color="#c0392b", ls="--", label=f"{lab} {c_:.1f} %")
    for cap in (8, 10, 12, 15):
        ax[0].axvline(cap, color="#777", ls=":", lw=1)
    for _, r in rf.iterrows():
        ax[0].plot(min(r.PK_above, 40), 2, "v", color="#e67e22", ms=9)
        ax[0].annotate(r.pair[:-4], (min(r.PK_above, 40), 4), fontsize=7, rotation=90, ha="center")
    ax[0].set_title("Pump top above the DIST entry (backtest 1,547 fills; ▼ real fills)\ndotted = caps 8 / 10 / 12 / 15 %", fontsize=10)
    ax[0].set_xlabel("% above entry (clipped at 40)"); ax[0].set_ylabel("fills"); ax[0].legend(fontsize=8)
    cells = ["P8", "P10", "P12", "P15", "Q8", "Q10", "Q12", "Q15", "W8", "W10", "W12", "W15", "S8_mean", "S10_mean", "S12_mean", "S15_mean", "DQ10", "DQ15"]
    h = E[E.period == "HOLDOUT"].set_index("cell").loc[cells]; dsg = E[E.period == "DESIGN"].set_index("cell").loc[cells]
    x = np.arange(len(cells))
    col = ["#2e86c1"] * 8 + ["#7f8c8d"] * 4 + ["#a569bd"] * 4 + ["#16a085"] * 2
    ax[1].bar(x - .2, dsg.d_mean, .4, color=col, alpha=.45, label="DESIGN Δ")
    ax[1].bar(x + .2, h.d_mean, .4, color=col, label="HOLDOUT Δ")
    ok_ = h.ci_lo.notna() & ~h.index.str.contains("mean")
    ax[1].errorbar((x + .2)[ok_.values], h.d_mean[ok_], yerr=[(h.d_mean - h.ci_lo)[ok_], (h.ci_hi - h.d_mean)[ok_]], fmt="none", ecolor="k", lw=.8)
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xticks(x); ax[1].set_xticklabels(cells, rotation=60, fontsize=8)
    ax[1].set_title("Δ net %/fill vs today's stop (pale = DESIGN, solid = HOLDOUT ± 95 % CI)\nblue = pump-top cells · grey = fixed · purple = shuffled top · green = candle high", fontsize=10)
    ax[1].legend(fontsize=8)
    PF = pd.read_csv(os.path.join(D, "perfill.csv.gz"))
    m = PF.period == "HOLDOUT"
    o = np.argsort(T[m.values])
    for c, cl_, lab in (("REF", "k", "today (min(1.5×ATR, 6 %))"), ("Q12", "#2e86c1", "Q12 wider-of top+1 %, cap 12"),
                        ("P12", "#5dade2", "P12 top+1 %, cap 12"), ("W12", "#7f8c8d", "W12 fixed 12 %")):
        ax[2].plot(pd.to_datetime(T[m.values][o], unit="ms"), np.cumsum(PF.loc[m, c].values[o]) * USD / 100, color=cl_, label=lab, lw=1.4)
    ax[2].set_title("HOLDOUT cumulative USD at one size (USD 3,050 per fill)", fontsize=10); ax[2].legend(fontsize=8); ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(os.path.join(R, "reports", "DIST_PEAKSTOP_2026-11-21.png"), dpi=120)
    print("chart saved")


if __name__ == "__main__":
    main()
