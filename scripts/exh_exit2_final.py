#!/usr/bin/env python3
"""EXH_EXIT2 — FINAL: chart (anchored paths with exit levels · one-size equity curves · EXH vs random-entry Δ) + trade rows + the ship-bar
table for the best DESIGN cells. Research only, BACKTEST.  Usage: venv/bin/python -I scripts/exh_exit2_final.py"""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exh_exit2_eval as EV   # noqa: E402
XS = EV.XS; DAY, H1 = EV.DAY, EV.H1
OUT = EV.OUT


def paths(X, idx):
    """5-min sampled favourable path (short: + = our way) from the 1m close, 24 h, for the given rows."""
    P = np.full((len(idx), 289), np.nan)
    for p, g in X.iloc[idx].groupby("pair"):
        days = sorted({int(t) // DAY + k for t in g.entry_ms for k in range(3)})
        one, _ = XS.load1m(p, days)
        if one is None:
            continue
        t, o, h, l, c = one
        for q, T in zip(g.index, g.entry_ms.astype("int64")):
            a = int(np.searchsorted(t, T))
            if a >= len(t) or t[a] != T:
                continue
            E0 = o[a]; ts = T + np.arange(289) * 300_000
            jj = np.clip(np.searchsorted(t, ts, "right") - 1, a, len(t) - 1)
            P[np.searchsorted(idx, q)] = -(c[jj] / E0 - 1) * 100
    return P


def main():
    X, Z = EV.load("EXH"); NET = EV.nets(X, Z); N20 = EV.nets(X, Z, flat=0.20)
    E = pickle.load(open(os.path.join(EV.D, "eval.pkl"), "rb"))
    names = EV.NAMES
    # trade rows
    tr = X[["pair", "entry_ms", "half", "atrp", "q5", "slip"]].copy()
    tr["entry_utc"] = pd.to_datetime(tr.entry_ms, unit="ms"); tr["block"] = np.where(tr.entry_ms >= EV.HO_MS, "HOLDOUT", "DESIGN")
    for j, n in enumerate(names):
        tr[f"net_{n}"] = NET[:, j].round(4); tr[f"k_{n}"] = Z["k"][:, j]; tr[f"x_{n}"] = Z["xm"][:, j]
    tr["ref_worked"] = Z["worked"][:, 0]; tr["ref_reach"] = Z["reach"][:, 0].round(3)
    tr.to_csv(f"{OUT}_trades.csv.gz", index=False, compression="gzip")
    # ship-bar table for the best DESIGN cells (descriptive: there is no survivor)
    SS = pd.read_csv(f"{OUT}_select.csv"); GR = pd.read_csv(f"{OUT}_grid.csv"); NU = pd.read_csv(f"{OUT}_null.csv").set_index("cell")
    JT = pd.read_csv(f"{OUT}_joint.csv"); CLU = pd.read_csv(f"{OUT}_cluster.csv").set_index("cell")
    rows = []
    for cell in list(SS.cell[:3]) + ["L_A8_L5", "T_16h"]:
        h = GR[(GR.cell == cell) & (GR.blk == "HOLDOUT")].iloc[0]; a = GR[(GR.cell == cell) & (GR.blk == "ALL")].iloc[0]
        hr = GR[(GR.cell == "REF") & (GR.blk == "HOLDOUT")].iloc[0]
        npos = sum(a[f"d_{x}"] > 0 for x in EV.HALVES)
        j_ = JT[(JT.cell == cell)] if cell in set(JT.cell) else None
        jr = JT[JT.cell == "REF"]
        rows.append(dict(cell=cell, g1=f"{h.delta:+.3f} ({h.d_lo:+.3f} … {h.d_hi:+.3f})", g1ok=h.d_lo > 0,
                         g2=f"{npos}/7", g2ok=npos >= 5, g3=f"{h.f20 - h.ref_f20:+.3f}", g3ok=h.f20 - h.ref_f20 > 0,
                         g4=f"{h.b5:+.3f} vs {h.ref_b5:+.3f}", g4ok=h.b5 > h.ref_b5,
                         g5=f"{h.kelly:.3f} / {h.kelly_day:.3f}", g5ok=(h.kelly > 0) and (h.kelly_day > 0),
                         g6=f"{NU.loc[cell, 'p_fw']:.2f}", g6ok=NU.loc[cell, "p_fw"] < 0.05,
                         g7=f"{h.pair_share:.0f} / {h.day_share:.0f}" if np.isfinite(h.pair_share) else "Δ ≤ 0", g7ok=bool(np.isfinite(h.pair_share) and h.pair_share < 50 and h.day_share < 50),
                         g8=(f"{j_[j_.blk == 'ALL'].dcr.iloc[0]:.3f} vs {jr[jr.blk == 'ALL'].dcr.iloc[0]:.3f} · HO {j_[j_.blk == 'HOLDOUT'].dcr.iloc[0]:.3f} vs {jr[jr.blk == 'HOLDOUT'].dcr.iloc[0]:.3f}") if j_ is not None and len(j_) else "n/a",
                         g8ok=bool(j_ is not None and len(j_) and (j_.dcr.values >= jr.dcr.values - 1e-9).all()),
                         g9=f"alone wd {h.worst_day:.1f} vs {hr.worst_day:.1f}, dd {h.maxdd:.0f} vs {hr.maxdd:.0f}",
                         g9ok=(h.worst_day >= hr.worst_day) and (h.maxdd <= hr.maxdd)))
    SB = pd.DataFrame(rows); SB.to_csv(f"{OUT}_shipbar.csv", index=False)
    print(SB.to_string())
    # chart
    kr = Z["k"][:, 0]; win = np.nonzero((kr == 0) & (Z["r"][:, 0] >= 9.99))[0]; gb = np.nonzero((kr == 1) & (Z["worked"][:, 0] == 1))[0]
    nv = np.nonzero((kr == 1) & (Z["worked"][:, 0] == 0))[0]
    Pw, Pg, Pn = paths(X, win), paths(X, gb), paths(X, nv)
    hrs = np.arange(289) / 12
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    for P, lab, col in ((Pw, f"REF +10 % TP winners (n={len(win)})", "#1b7f3b"), (Pg, f"REF stopped, went ≥ +1 % first (n={len(gb)})", "#c0392b"),
                        (Pn, f"REF stopped, never +1 % (n={len(nv)})", "#7f8c8d")):
        m = np.nanmedian(P, 0); q1, q3 = np.nanpercentile(P, 25, 0), np.nanpercentile(P, 75, 0)
        ax[0].plot(hrs, m, color=col, lw=2, label=lab); ax[0].fill_between(hrs, q1, q3, color=col, alpha=0.12)
    med_sl = np.median(np.minimum(1.5 * X.atrp.values, 6))
    for y, lab, ls in ((10, "TP 10 % (maker)", "-"), (-med_sl, f"stop (median {med_sl:.1f} %)", "-"), (3, "BE arm +3 %", ":"), (6, "BE arm +6 %", ":"),
                       (5, "lock +8 → +5", "--")):
        ax[0].axhline(y, color="k", ls=ls, lw=0.8); ax[0].text(23.9, y + 0.15, lab, ha="right", fontsize=8)
    # share of +10 % winners that came back to entry after +3 % (what a BE3 kills)
    reached3 = np.array([np.nanmax(p[:i + 1]) if i >= 0 else np.nan for p, i in zip(Pw, np.zeros(len(Pw), int))])
    ax[0].set(title="Anchored 24 h path after entry (close, 5-min; continues past the exit)\nmedian + IQR — up = our way", xlabel="hours after entry", ylabel="% our way (short)", ylim=(-8, 16))
    ax[0].legend(fontsize=8, loc="upper left")
    # equity
    for cell, col in (("REF", "k"), ("L_A8_L5", "#2471a3"), ("T_16h", "#8e44ad"), ("B_BE3", "#c0392b"), ("S_TP3_SL4", "#e67e22"), ("NP_8h", "#27ae60")):
        j = names.index(cell); _, r = EV.onesize(NET[:, j], Z["xm"][:, j] // DAY, EV.D0, EV.D1)
        eq = np.cumprod(1 + r.values)
        ax[1].plot(pd.to_datetime(r.index * DAY, unit="ms"), eq, color=col, lw=1.6 if cell == "REF" else 1.1, label=cell)
    import matplotlib.dates as mdates
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m")); ax[1].set_yscale("log"); ax[1].axvline(pd.Timestamp("2025-07-01"), color="grey", ls="--"); ax[1].text(pd.Timestamp("2025-07-10"), 1.2, "HOLDOUT →", fontsize=8)
    ax[1].set(title="EXH alone, one size 0.61 × equity, compounded (BACKTEST)", ylabel="equity (×)"); ax[1].legend(fontsize=8)
    # control
    CT = pd.read_csv(f"{OUT}_control.csv"); CT = CT[(CT.blk == "ALL") & (CT.cell.isin([n for n in names if n not in ("REF", "X_NOTP")]))]
    o = np.argsort(CT.exh_delta.values)
    yy = np.arange(len(CT))
    ax[2].barh(yy - 0.2, CT.exh_delta.values[o], 0.4, color="#2471a3", label="EXH entries")
    ax[2].barh(yy + 0.2, CT.rnd_delta.values[o], 0.4, color="#bbbbbb", label="random entries, same coins / episodes")
    ax[2].set_yticks(yy); ax[2].set_yticklabels(CT.cell.values[o], fontsize=7); ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set(title="Paired Δ vs today's exit, %/trade (all 2023-07 → 2026-11)", xlabel="Δ avg net %/trade"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}_chart.png", dpi=120)
    # facts for the write-up
    def back_to(P, lvl_hi, lvl_lo):
        n = 0
        for p in P:
            ok = np.isfinite(p)
            if not ok.any():
                continue
            i = np.argmax(p >= lvl_hi) if (p >= lvl_hi).any() else None
            if i is None:
                continue
            if (p[i:] <= lvl_lo).any() and np.nanmax(p[i:]) >= 9.9:
                hit = np.argmax(p[i:] <= lvl_lo); top = np.argmax(p[i:] >= 9.9)
                n += hit < top
        return n
    print("winners", len(win), "that touched +3 then came back to ≤ 0 before +10 (close-based):", back_to(Pw, 3, 0.0))
    print("winners median path at 1h/4h/8h", np.nanmedian(Pw[:, [12, 48, 96]], 0), "gave-back at 1h/4h", np.nanmedian(Pg[:, [12, 48]], 0))


if __name__ == "__main__":
    main()
