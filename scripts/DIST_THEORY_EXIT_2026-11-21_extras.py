#!/usr/bin/env python3
"""DIST_THEORY_EXIT — EXTRAS (descriptive, after the frozen HOLDOUT read). Research only, BACKTEST.
① stop families (C / R / W) on the REF-STOPPED cohort, two-sided: saved (better) vs deeper (worse) n / $ at $3,050, per block;
   assert W cells change nothing on fills REF did not stop. ② leave-one-pair-out / leave-one-day-out of the best HOLDOUT cells.
③ chart: event study (REF winners vs stopped, with the distribution-candle high and today's stop) + HOLDOUT Δ forest of every cell.
→ reports/DIST_THEORY_EXIT_2026-11-21_data/{stopcohort.csv, loo.csv} + reports/DIST_THEORY_EXIT_2026-11-21.png"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21_data")
USD = 3050.0


def main():
    PF = pd.read_csv(os.path.join(D, "perfill.csv.gz"))
    E = pd.read_csv(os.path.join(D, "eval.csv")); P = pd.read_csv(os.path.join(D, "picks.csv"))
    names = [c for c in E.cell.unique()]
    stopped = PF["REF_k"].values == 1
    rows = []
    for nm in [n for n in names if n[0] in "CRW"]:
        d = PF[nm].values - PF["REF"].values
        if nm.startswith("W_"):
            assert np.abs(d[~stopped]).max() < 1e-6, nm
        for blk in ("DESIGN", "HOLDOUT", "ALL"):
            m = (PF.period.values == blk) if blk != "ALL" else np.ones(len(PF), bool)
            s = m & stopped; ns = m & ~stopped
            rows.append(dict(cell=nm, block=blk, ref_stopped=int(s.sum()), saved_n=int((d[s] > 0.05).sum()), saved_usd=d[s & (d > 0.05)].sum() * USD / 100,
                             deeper_n=int((d[s] < -0.05).sum()), deeper_usd=d[s & (d < -0.05)].sum() * USD / 100,
                             nonstopped_changed=int((np.abs(d[ns]) > 1e-6).sum()), nonstopped_usd=d[ns].sum() * USD / 100,
                             net_usd=d[m].sum() * USD / 100))
    SC = pd.DataFrame(rows); SC.to_csv(os.path.join(D, "stopcohort.csv"), index=False)
    pd.set_option("display.width", 250)
    print(SC[SC.block != "DESIGN"].round(0).to_string(index=False))
    # LOO on the best HOLDOUT cells
    H = E[(E.period == "HOLDOUT") & (E.cell != "REF")].sort_values("d_mean", ascending=False).head(6)
    L = []
    for nm in H.cell:
        m = PF.period.values == "HOLDOUT"; d = (PF[nm].values - PF["REF"].values)[m]
        pr = pd.Series(d).groupby(PF.pair.values[m]).sum(); dy = pd.Series(d).groupby(pd.to_datetime(PF.entry.values[m]).normalize()).sum()
        L.append(dict(cell=nm, d_mean=d.mean(), top_pair=pr.idxmax(), top_pair_usd=pr.max() * USD / 100,
                      d_mean_wo_top_pair=(d.sum() - pr.max()) / (len(d) - (PF.pair.values[m] == pr.idxmax()).sum()),
                      top_day=str(dy.idxmax().date()), d_mean_wo_top_day=(d.sum() - dy.max()) / len(d),
                      d_mean_wo_top5pct_fills=np.sort(d)[: int(len(d) * 0.95)].mean()))
    L = pd.DataFrame(L); L.to_csv(os.path.join(D, "loo.csv"), index=False)
    print("\nLOO (HOLDOUT, best 6 cells by Δ)\n", L.round(3).to_string(index=False))
    chart(PF, E, P)


def chart(PF, E, P):
    S = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    Z = np.load(os.path.join(R, "reports", "DIST_EXIT_data", "paths_L1.npz"))
    pos = {int(i): j for j, i in enumerate(Z["idx"])}
    rows = [pos.get(int(z), -1) for z in S.zi.values]
    ok = np.array(rows) >= 0
    Pm = np.full((len(S), Z["p"].shape[1]), np.nan); Pm[ok] = Z["p"][np.array(rows)[ok]]
    hrs = (np.arange(Z["p"].shape[1]) - 24) / 12.0
    keep = (hrs >= -1) & (hrs <= 24)
    k = PF.REF_k.values
    dh = -(PF.DH.values / PF.E0.values - 1) * 100; sl = -S.sl_pct.values
    BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8984", "#2b2b29"
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    for msk, col, lab in ((k == 0, BLUE, f"REF take-profit winners (n {int((k == 0).sum())})"), (k == 1, ORANGE, f"REF stopped (n {int((k == 1).sum())})")):
        q = np.nanpercentile(Pm[msk][:, keep], [25, 50, 75], axis=0)
        a.fill_between(hrs[keep], q[0], q[2], color=col, alpha=0.15, lw=0)
        a.plot(hrs[keep], q[1], color=col, lw=2, label=lab + " — median, IQR band")
    a.axhline(np.median(sl), color=INK, lw=1, ls="--"); a.text(23.8, np.median(sl) + 0.3, f"today's stop, median {np.median(sl):.1f} %", ha="right", fontsize=9, color=INK)
    a.axhline(np.median(dh), color=GRAY, lw=1, ls=":"); a.text(23.8, np.median(dh) - 0.9, f"distribution-candle high, median {np.median(dh):.1f} %", ha="right", fontsize=9, color=GRAY)
    a.axhline(0, color="#d8d7d2", lw=0.8); a.axvline(0, color="#d8d7d2", lw=0.8)
    a.set_xlabel("hours from entry"); a.set_ylabel("price move our way, % (short: + = price fell)")
    a.set_title("DIST backtest 1,547 fills: the path of a winner vs a stopped trade", fontsize=11, loc="left")
    a.legend(frameon=False, fontsize=9, loc="upper left"); a.set_ylim(-14, 14); a.spines[["top", "right"]].set_visible(False)
    b = ax[1]
    H = E[(E.period == "HOLDOUT") & (E.cell != "REF")].sort_values("d_mean").reset_index(drop=True)
    picks = set(P.pick)
    for i, r in H.iterrows():
        col = BLUE if r.cell in picks else GRAY
        b.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=1.5); b.plot(r.d_mean, i, "o", color=col, ms=5)
    b.axvline(0, color=INK, lw=1)
    b.set_yticks(range(len(H))); b.set_yticklabels(H.cell, fontsize=7)
    b.set_xlabel("HOLDOUT Δ vs today's exit, % per trade (95 % day-block CI)")
    b.set_title("Every theory exit vs today's (HOLDOUT ≥ 2025-07, 988 fills)\nblue = family pick chosen on DESIGN · grey = other cells", fontsize=11, loc="left")
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21.png"), dpi=130)


if __name__ == "__main__":
    main()
