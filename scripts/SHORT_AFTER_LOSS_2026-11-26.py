#!/usr/bin/env python3
"""SHORT_AFTER_LOSS — does an EXHAUSTION / DISTRIBUTION short on a coin whose PREVIOUS short (either sleeve) just LOST do worse?
(operator 2026-11-26: "three consecutive losses on MANTRAUSDT (DIST → EXH → EXH) … also two consecutive on BIGTIMEUSDT (EXH → EXH)")

PRIOR: ENTRY_WL 2026-11-18 st_prior_fill_won — EXH-after-EXH in the same episode, at the OLD EXH exit (TP 10), runner-share label: nothing.
NEW: today's EXH exit (TP 7, DL 609) booked net, cross-sleeve (a DIST loss before an EXH, and vice versa), per time gap.

Backtest: EXH = TP_GRID 2026-11-20 rows, cell F7 (= today's EXH exit), from 2024-06-02 (the EXH_BREADTH window). DIST = DIST_EXIT PRIMARY
rows under Exit B (= today's DIST exit). Each sleeve one position per coin (its own exit time). Sequence per coin: all fills of both sleeves
by entry time; "previous" = the latest fill of either sleeve on the same coin whose EXIT is at or before this entry and within GAP hours.
Groups: after a LOSS (prev net ≤ 0) · after a WIN · no previous within GAP. Read: N · WR · avg net · 10-day block CI on the after-LOSS minus
after-WIN difference · DESIGN (< 2025-07-01) / HOLDOUT · window count. A block candidate needs the locked Pattern-C bar + the breakeven-bot
read; this script only describes.
Out: reports/SHORT_AFTER_LOSS_2026-11-26.csv"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_exit_sim as DS      # noqa: E402
import dist_exit_eval as EV     # noqa: E402
import dist_wl_survivors as SV  # noqa: E402
R = DS.R; RP = os.path.join(R, "reports")
DAY, H1 = 86_400_000, 3_600_000
START = int(pd.Timestamp("2024-06-02").value // 10**6); HOL = int(pd.Timestamp("2025-07-01").value // 10**6)
END = int(pd.Timestamp("2026-11-18").value // 10**6)


def exh():
    GD = os.path.join(RP, "TP_GRID_2026-11-20_EXH_data")
    X = pd.read_pickle(os.path.join(GD, "rows_FRENZY.pkl")).reset_index(drop=True)
    Z = dict(np.load(os.path.join(GD, "sim_FRENZY.npz"), allow_pickle=True)); names = [str(x) for x in Z.pop("names")]
    j = names.index("F7"); s = np.maximum(X.slip.values.astype(float), 0.10); tk = Z["k"][:, j] != 0
    net = Z["r"][:, j].astype(float) + Z["f"][:, j].astype(float) - s - 0.05 - np.where(tk, s + 0.05, 0.02)
    D = pd.DataFrame(dict(pair=X.pair.astype(str).values, entry_ms=X.entry_ms.values.astype(np.int64), xm=Z["xm"][:, j].astype(np.int64), net=net,
                          ok=Z["ok"] & (Z["k"][:, j] >= 0) & (Z["k"][:, j] != 9)))
    D = D[D.ok & np.isfinite(D.net)].reset_index(drop=True)
    keep = EV.one_pos(D.pair.values, D.entry_ms.values, D.xm.values, np.ones(len(D), bool))
    return D[keep].assign(sleeve="EXH")


def dist():
    S = pd.read_pickle(os.path.join(RP, "DIST_WL_2026-11-18_data", "rows.pkl"))
    F = DS.frames()["L1"]; prim = SV.keep_mask(S, EV.REP); Fp = F[prim].reset_index(drop=True)
    CL = [DS.cells()[0]]
    outs = []
    for p in Fp.pair.unique():
        rows = Fp[Fp.pair == p]
        o, _ = DS.run_rows(p, rows, CL)
        outs.append(pd.DataFrame(dict(idx=rows.index.values, r=o["r"][:, 0], k=o["k"][:, 0], xm=o["xm"][:, 0], f=o["f"][:, 0], ok=o["ok"])))
    O = pd.concat(outs).set_index("idx").sort_index()
    s = Fp.slip.values; tk = O.k.values != 0
    net = O.r.values - s - np.where(tk, s, 0) - 0.05 - np.where(tk, 0.05, 0.02) + O.f.values
    D = pd.DataFrame(dict(pair=Fp.pair.values, entry_ms=Fp.entry_ms.values.astype(np.int64), xm=O.xm.values.astype(np.int64), net=net,
                          ok=O.ok.values & (O.k.values >= 0) & (O.k.values != 9)))
    D = D[D.ok & np.isfinite(D.net)].reset_index(drop=True)
    keep = EV.one_pos(D.pair.values, D.entry_ms.values, D.xm.values, np.ones(len(D), bool))
    return D[keep].assign(sleeve="DIST")


def tag(A, gap_h):
    A = A.sort_values(["pair", "entry_ms"]).reset_index(drop=True)
    prev = np.full(len(A), np.nan); prev_sl = np.full(len(A), "", object); prev_gap = np.full(len(A), np.nan)
    for p, g in A.groupby("pair"):
        ii = g.index.values; xm = g.xm.values; en = g.entry_ms.values; nt = g.net.values; sl = g.sleeve.values
        for a, i in enumerate(ii):
            m = (xm <= en[a]) & (np.arange(len(ii)) != a) & (en[a] - xm <= gap_h * H1)
            if m.any():
                b = np.nonzero(m)[0][np.argmax(xm[m])]
                prev[i] = nt[b]; prev_sl[i] = sl[b]; prev_gap[i] = (en[a] - xm[b]) / H1
    A["prev_net"] = prev; A["prev_sleeve"] = prev_sl; A["prev_gap_h"] = prev_gap
    A["grp"] = np.where(np.isnan(prev), "no prev", np.where(prev <= 0, "after LOSS", "after WIN"))
    return A


def diff_ci(B, span):
    L = B[B.grp == "after LOSS"]; W = B[B.grp == "after WIN"]
    if len(L) < 5 or len(W) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(1126); lo0 = span[0] // DAY; nd = span[1] // DAY - lo0
    bl = 10; nb = int(np.ceil(nd / bl)); ds = []
    dl = (L.entry_ms.values // DAY - lo0); dw = (W.entry_ms.values // DAY - lo0)
    for _ in range(2000):
        st = rng.integers(0, nd - bl + 1, nb); days = (st[:, None] + np.arange(bl)).ravel()[:nd]
        cnt = np.bincount(days, minlength=nd)
        wl = cnt[np.clip(dl, 0, nd - 1)]; ww = cnt[np.clip(dw, 0, nd - 1)]
        if wl.sum() == 0 or ww.sum() == 0:
            continue
        ds.append((L.net.values * wl).sum() / wl.sum() - (W.net.values * ww).sum() / ww.sum())
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def main():
    A = pd.concat([exh(), dist()], ignore_index=True)
    A = A[(A.entry_ms >= START) & (A.entry_ms < END)].reset_index(drop=True)
    rows = []
    for gap in (6, 24):
        T = tag(A, gap)
        for sleeve in ("EXH", "DIST"):
            for blk, span in (("ALL", (START, END)), ("DESIGN", (START, HOL)), ("HOLDOUT", (HOL, END))):
                B = T[(T.sleeve == sleeve) & (T.entry_ms >= span[0]) & (T.entry_ms < span[1])]
                lo, hi = diff_ci(B, span)
                for grp in ("after LOSS", "after WIN", "no prev"):
                    g = B[B.grp == grp]
                    rows.append(dict(gap_h=gap, sleeve=sleeve, block=blk, group=grp, N=len(g), WR=(g.net > 0).mean() * 100 if len(g) else np.nan,
                                     avg=g.net.mean(), days=g.entry_ms.floordiv(DAY).nunique(), pairs=g.pair.nunique(),
                                     loss_minus_win_lo=lo, loss_minus_win_hi=hi))
                # cross-sleeve split of after-LOSS
                for ps in ("EXH", "DIST"):
                    g = B[(B.grp == "after LOSS") & (B.prev_sleeve == ps)]
                    rows.append(dict(gap_h=gap, sleeve=sleeve, block=blk, group=f"after LOSS on {ps}", N=len(g),
                                     WR=(g.net > 0).mean() * 100 if len(g) else np.nan, avg=g.net.mean() if len(g) else np.nan,
                                     days=g.entry_ms.floordiv(DAY).nunique(), pairs=g.pair.nunique()))
    O = pd.DataFrame(rows); O.to_csv(os.path.join(RP, "SHORT_AFTER_LOSS_2026-11-26.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(O.round(2).to_string())


if __name__ == "__main__":
    main()
