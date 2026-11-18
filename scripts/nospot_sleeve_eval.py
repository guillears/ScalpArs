#!/usr/bin/env python3
"""NOSPOT_SLEEVE — EVAL (PREREG §Selection / §Ship bar). B0 / NS0 / 10 rules × 6 exits on DESIGN, plateau, survivors, HOLDOUT (one run =
the single read), full span, family-wise shuffled-timing null (500 draws, same episode), first-per-episode robustness.
Writes reports/NOSPOT_SLEEVE_2026-11-18_cells.csv, _plateau.csv, data/sel.pkl (selected trade indices per cell)."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import nospot_sleeve_lib as L   # noqa: E402
OUT = os.path.join(L.R, "reports", "NOSPOT_SLEEVE_2026-11-18")
rng = np.random.default_rng(11)
NDRAW = 500


def main():
    X = L.load()
    X.to_pickle(os.path.join(L.ND, "X.pkl"))
    print("rows", len(X), "pairs", X.pair.nunique(), "episodes", X.epi.nunique(), X.groupby("blk").size().to_dict(), flush=True)
    RM = L.rule_masks(X)
    COH = {"B0": np.ones(len(X), bool), "NS0": L.ns0_mask(X)}
    for k, (m, _) in RM.items():
        COH[k] = m
    des = X.blk.values == "DESIGN"; hol = ~des
    rows, sel, plat = [], {}, []
    for c, m in COH.items():
        for e in L.EXITS:
            idx = L.one_pos(X, np.nonzero(m)[0], e)
            sel[(c, e)] = idx
            for blk, bm in (("DESIGN", des), ("HOLDOUT", hol), ("ALL", None)):
                ii = idx if bm is None else idx[bm[idx]]
                r = L.stats(X, ii, e, blk); r.update(cohort=c, exit=e, blk=blk, cov=np.nan)
                rows.append(r)
            fe = L.first_per_ep(X, idx)
            r = L.stats(X, fe, e, "ALL"); r.update(cohort=c, exit=e, blk="ALL_first_ep"); rows.append(r)
        if c in RM:
            for j, pm in enumerate(RM[c][1]):
                for e in L.EXITS:
                    ii = L.one_pos(X, np.nonzero(pm & des)[0], e)
                    plat.append(dict(cohort=c, j=j, exit=e, N=len(ii), avg=X[f"{e}_net"].values[ii].mean() if len(ii) else np.nan))
    Cdf = pd.DataFrame(rows)
    P = pd.DataFrame(plat)
    # coverage of each rule's inputs (share of rows where the primary rule is scorable) — inputs non-NaN
    cov_cols = {"R1_OI_UNWIND": ["oi_chg1h", "dd_hi"], "R2_OI_TRAP": ["oi_chg1h"], "R3_FUND_TURN": ["f_last8", "f_prev8"],
                "R4_TOP_DISTRIB": ["topP_chg1h", "glA_chg1h"], "R5_TAKER_FADE": ["taker1h", "taker_prev1h"], "R6_PREM_COLLAPSE": ["prem1h", "prem_pk"],
                "R7_LH_VWAP": ["lh2", "vwap_gap"], "R8_CASCADE": ["oi_chg1h"], "R9_CROWD_LONG": ["glA", "f_last", "oi_chg1h"],
                "R10_OI_FROM_MAX": ["oi_vs_max"], "NS0": ["rsi15", "vd"], "B0": ["dd_hi"]}
    for c, cols in cov_cols.items():
        cv = np.all([np.isfinite(X[k].values.astype(float)) for k in cols], 0)
        for blk, bm in (("DESIGN", des), ("HOLDOUT", hol), ("ALL", np.ones(len(X), bool))):
            Cdf.loc[(Cdf.cohort == c) & (Cdf.blk == blk), "cov"] = cv[bm].mean() * 100
    # selection on DESIGN
    D = Cdf[Cdf.blk == "DESIGN"].set_index(["cohort", "exit"])
    b0 = D.xs("B0", level=0)
    surv = []
    for (c, e), r in D.iterrows():
        if c in ("B0", "NS0"):
            continue
        hs = [h for h, _, _ in L.HALVES[:4] if r.get(f"h_{h}_N", 0) >= 10]
        hpos = sum(r[f"h_{h}"] > 0 for h in hs) / len(hs) if hs else 0
        pp = P[(P.cohort == c) & (P.exit == e) & (P.N >= 10)]
        ppos = (pp.avg > 0).mean() if len(pp) else 0
        ok = r.N >= 30 and r.avg > 0 and r.avg > b0.loc[e, "avg"] and hpos >= 2 / 3 and ppos >= 2 / 3
        Cdf.loc[(Cdf.cohort == c) & (Cdf.exit == e), ["des_hpos", "plat_pos", "survivor"]] = [hpos, ppos, ok]
        if ok:
            surv.append((c, e))
    print("DESIGN survivors:", surv, flush=True)
    # family-wise shuffled-timing null (full span): replace each trade's entry by a uniform random bar of the same episode
    ep = X.epi.values; st = np.r_[0, np.nonzero(np.diff(ep))[0] + 1]; cnt = np.diff(np.r_[st, len(X)])
    cells = [(c, e) for c in RM for e in L.EXITS] + [("NS0", e) for e in L.EXITS]
    obs = np.array([X[f"{e}_net"].values[sel[(c, e)]].mean() if len(sel[(c, e)]) else np.nan for c, e in cells])
    NUL = np.full((NDRAW, len(cells)), np.nan)
    for j, (c, e) in enumerate(cells):
        idx = sel[(c, e)]
        if not len(idx):
            continue
        vv = X[f"{e}_net"].values.astype(float)
        s0 = st[ep[idx]]; n0 = cnt[ep[idx]]
        U = rng.random((NDRAW, len(idx)))
        NUL[:, j] = vv[s0[None, :] + (U * n0[None, :]).astype(np.int64)].mean(1)
    mu = np.nanmean(NUL, 0)
    prim = np.array([c != "NS0" for c, _ in cells])
    mx = np.nanmax((NUL - mu)[:, prim], 1)
    for j, (c, e) in enumerate(cells):
        d = obs[j] - mu[j]
        m_ = (Cdf.cohort == c) & (Cdf.exit == e)
        Cdf.loc[m_, "null_mean"] = mu[j]; Cdf.loc[m_, "null_p"] = (NUL[:, j] - mu[j] >= d).mean()
        Cdf.loc[m_, "null_pfw"] = (mx >= d).mean()
    Cdf.to_csv(OUT + "_cells.csv", index=False)
    P.to_csv(OUT + "_plateau.csv", index=False)
    pickle.dump(dict(sel=sel, surv=surv, null=NUL, cells=cells), open(os.path.join(L.ND, "sel.pkl"), "wb"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    show = ["cohort", "exit", "blk", "N", "tpd", "WR", "avg", "f20", "b5", "kelly", "dcr", "maxdd", "pair_share", "day_share", "ndays",
            "worked", "cov", "null_mean", "null_pfw"]
    print(Cdf[Cdf.blk == "DESIGN"][show + ["des_hpos", "plat_pos", "survivor"]].round(3).to_string())


if __name__ == "__main__":
    main()
