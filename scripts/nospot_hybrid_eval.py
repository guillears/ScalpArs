#!/usr/bin/env python3
"""NOSPOT_HYBRID (PREREG reports/NOSPOT_HYBRID_2026-11-18_PREREG.md) — EVAL (primary + variants).
HYB = NS0 (15m RSI cross 70 → ≤ 70 ∧ vol_decay ≥ 0.6366, vol24 ≥ $20M, no-spot FRENZY window) ∧ top-trader POSITION ratio 4 h ABSOLUTE
change ≤ −0.01097 (distribution.pos_chg_4h window; scripts/nospot_hybrid_build.py). Exits on the 1m engine (nospot_exit_sim cells REF_XE
= primary, A_TP4_SL10 = secondary). One position per pair per (cohort, exit). Writes reports/NOSPOT_HYBRID_2026-11-18_data/*.csv / core.pkl.
Usage: venv/bin/python -I scripts/nospot_hybrid_eval.py"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_exit_eval as E   # noqa: E402
import nospot_sleeve_lib as L  # noqa: E402
OD = os.path.join(L.R, "reports", "NOSPOT_HYBRID_2026-11-18_data")
DAY = L.DAY
CUT = -0.01097
JX = E.NAMES.index("REF_XE"); J4 = E.NAMES.index("A_TP4_SL10")
EXITS = {"XE": JX, "TP4SL10": J4}
RNG = np.random.default_rng(20261119)
XCOLS = ["oi_chg15m", "oi_chg4h", "oi_usd", "oi_vol", "oi_vs_max", "f_last", "f_prev", "f_last8", "f_int_h", "topP", "topP_chg4h", "topA",
         "topA_chg4h", "glA", "glA_chg4h", "taker1h", "taker_prev1h", "prem1h", "prem_pk", "rsi15", "rsi1h", "di15", "lh2", "vwap_gap",
         "peak_age_h", "hrs_since_spike", "volmult", "vol24", "ret15m", "ret4h", "ret24h", "maxup1h_24h", "rise_on_to_hi", "turnover",
         "btc1h", "btc4h", "btc24h", "hour", "R1", "R4", "day"]


def load():
    X, Y, Z = E.load()
    xi = Z["xi"]
    for c in XCOLS:
        if c not in Y.columns:
            Y[c] = X[c].values[xi]
    F = pd.read_pickle(os.path.join(OD, "feat.pkl")).reindex(xi)
    for c in F.columns:
        Y[c] = F[c].values
    Y["f_neg"] = np.where(np.isfinite(Y.f_last), (Y.f_last < 0).astype(float), np.nan)
    Y["f_hourly"] = np.where(np.isfinite(Y.f_int_h), (Y.f_int_h == 1).astype(float), np.nan)
    Y["xi"] = xi
    Y["month"] = pd.to_datetime(Y.entry_ms, unit="ms").dt.strftime("%Y-%m").values
    return X, Y, Z


def masks(Y, pos4=None, pos1=None, acc4=None):
    ns0 = Y.NS0.values.astype(bool)
    p4 = Y.pos4.values if pos4 is None else pos4
    p1 = Y.pos1.values if pos1 is None else pos1
    a4 = Y.acc4.values if acc4 is None else acc4
    with np.errstate(invalid="ignore"):
        return {"HYB": ns0 & (p4 <= CUT), "V1_pos1h": ns0 & (p1 <= CUT), "V2_acc4h": ns0 & (a4 <= CUT),
                "V3_cut-0.005": ns0 & (p4 <= -0.005), "V4_cut-0.02": ns0 & (p4 <= -0.02)}


def select(Y, Z, M):
    pair = Y.pair.values; T = Y.entry_ms.values.astype(np.int64)
    out = {}
    for c, m in M.items():
        ii = np.nonzero(m)[0]
        for e, j in EXITS.items():
            out[(c, e)] = ii[E.one_pos(pair[ii], T[ii], Z["xm"][ii, j])] if len(ii) else ii
    return out


def wins(v, days):
    return len(np.unique(days))


def main():
    X, Y, Z = load()
    NET = E.nets(Y, Z); net, net20, _ = NET
    ns0 = Y.NS0.values.astype(bool)
    print("NS0 rows", ns0.sum(), "pos4 scored", np.isfinite(Y.pos4.values[ns0]).mean(), "1m share", Z["src1m"][ns0].mean(), flush=True)
    M = masks(Y)
    with np.errstate(invalid="ignore"):
        M2 = dict(M, NS0=ns0, NS0_notHYB=ns0 & ~(Y.pos4.values <= CUT))
    SEL = select(Y, Z, M2)
    blkv = Y.blk.values
    rows = []
    for (c, e), idx in SEL.items():
        j = EXITS[e]
        for blk in ("DESIGN", "HOLDOUT", "ALL"):
            ii = idx if blk == "ALL" else idx[blkv[idx] == blk]
            r = dict(cohort=c, exit=e, blk=blk, **E.stats(Y, Z, NET, ii, j, blk))
            if len(ii) > 20:
                r["ci_lo"], r["ci_hi"] = E.boot_ci(net[ii, j], Z["xm"][ii, j] // DAY)
                r["sig_N"] = int(M2[c][(blkv == blk) if blk != "ALL" else slice(None)].sum())
            rows.append(r)
    G = pd.DataFrame(rows); G.to_csv(os.path.join(OD, "grid.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60)
    print(G[["cohort", "exit", "blk", "N", "tpd", "WR", "avg", "ci_lo", "ci_hi", "f20", "b5", "kelly", "kelly_day", "pair_share", "day_share",
             "ndays", "dcr", "maxdd", "worst_day"]].round(3).to_string(), flush=True)
    print(G[G.blk == "ALL"][["cohort", "exit"] + [f"h_{h}" for h, _, _ in L.HALVES] + [f"h_{h}_N" for h, _, _ in L.HALVES]].round(2).to_string())

    # --- matched random control (within-episode), 500 draws
    ep = Y.epi.values; order = np.argsort(ep, kind="stable"); es = ep[order]
    ue, start, cnt = np.unique(es, return_index=True, return_counts=True); pos = {e_: i for i, e_ in enumerate(ue)}
    rc = []
    for (c, e), idx in SEL.items():
        j = EXITS[e]; ei = np.array([pos[x] for x in ep[idx]])
        u = RNG.random((500, len(idx)))
        rr = order[start[ei][None, :] + np.floor(u * cnt[ei][None, :]).astype(int)]
        V = net[rr, j]; isd = blkv[idx] == "DESIGN"
        rc.append(dict(cohort=c, exit=e, obs=net[idx, j].mean(), rnd=V.mean(), p_cell=(V.mean(1) >= net[idx, j].mean()).mean(),
                       obs_des=net[idx[isd], j].mean(), rnd_des=V[:, isd].mean(), obs_hold=net[idx[~isd], j].mean(), rnd_hold=V[:, ~isd].mean()))
    RC = pd.DataFrame(rc); RC.to_csv(os.path.join(OD, "random_control.csv"), index=False)
    print(RC.round(3).to_string(), flush=True)

    # --- family-wise null: shuffle the position-change values across NS0 signals within month (1,000)
    fam = [(c, e) for c in ("HYB", "V1_pos1h", "V2_acc4h", "V3_cut-0.005", "V4_cut-0.02") for e in EXITS]
    obs = {k: net[SEL[k], EXITS[k[1]]].mean() for k in fam}
    obs_h = {k: net[SEL[k][blkv[SEL[k]] == "HOLDOUT"], EXITS[k[1]]].mean() for k in fam}
    ii = np.nonzero(ns0)[0]; mo = Y.month.values[ii]
    groups = [ii[mo == m_] for m_ in np.unique(mo)]
    NSH = 1000
    D = np.zeros((NSH, len(fam))); DH = np.zeros((NSH, len(fam)))
    base4, base1, basea = Y.pos4.values.copy(), Y.pos1.values.copy(), Y.acc4.values.copy()
    for s in range(NSH):
        p4, p1, a4 = base4.copy(), base1.copy(), basea.copy()
        for g in groups:
            p4[g] = base4[RNG.permutation(g)]; p1[g] = base1[RNG.permutation(g)]; a4[g] = basea[RNG.permutation(g)]
        Ms = masks(Y, p4, p1, a4)
        Ss = select(Y, Z, Ms)
        for q, k in enumerate(fam):
            idx = Ss[k]; v = net[idx, EXITS[k[1]]]
            D[s, q] = v.mean(); DH[s, q] = v[blkv[idx] == "HOLDOUT"].mean()
        if s % 200 == 0:
            print("null", s, flush=True)
    mx = D.max(1); mxh = DH.max(1)
    cen = D - D.mean(0); mxc = cen.max(1)
    nu = []
    for q, k in enumerate(fam):
        nu.append(dict(cohort=k[0], exit=k[1], obs=obs[k], null_mean=D[:, q].mean(), null_p95=np.percentile(D[:, q], 95),
                       p_cell=(D[:, q] >= obs[k]).mean(), p_fw=(mx >= obs[k]).mean(), p_fw_centered=(mxc >= obs[k] - D[:, q].mean()).mean(),
                       obs_hold=obs_h[k], null_hold_mean=DH[:, q].mean(), p_cell_hold=(DH[:, q] >= obs_h[k]).mean(),
                       p_fw_hold=(mxh >= obs_h[k]).mean()))
    NU = pd.DataFrame(nu); NU.to_csv(os.path.join(OD, "null.csv"), index=False)
    print(NU.round(3).to_string(), flush=True)

    # --- path decomposition
    pg = []
    for (c, e), idx in SEL.items():
        if c not in ("HYB", "NS0", "NS0_notHYB"):
            continue
        j = EXITS[e]
        for blk in ("ALL", "HOLDOUT"):
            ii = idx if blk == "ALL" else idx[blkv[idx] == blk]
            g6 = E.path_group(Z, NET, ii, j); v = net[ii, j]; wk = Z["worked"][ii, j] == 1
            g3 = np.where(v > 0, "RUNNER", np.where(wk, "GAVE_BACK", "NEVER_WORKED"))
            tl = v[v <= 0].sum()
            for lab, g in (("3", g3), ("6", g6)):
                for grp in np.unique(g):
                    m = g == grp
                    pg.append(dict(cohort=c, exit=e, blk=blk, scheme=lab, group=grp, N=int(m.sum()), share=m.mean() * 100, sum_pct=v[m].sum(),
                                   avg=v[m].mean(), loss_share=v[m].sum() / tl * 100 if v[m].sum() < 0 else np.nan,
                                   reach_med=np.median(Z["reach"][ii[m], j]), best_med=np.median(Z["best"][ii[m], j])))
    PG = pd.DataFrame(pg); PG.to_csv(os.path.join(OD, "paths.csv"), index=False)
    print(PG[PG.scheme == "3"].round(2).to_string(), flush=True)

    # --- BTC daily regime (R1 sign) and R4, window units (UTC days)
    rg = []
    for (c, e), idx in SEL.items():
        if c not in ("HYB", "NS0", "NS0_notHYB"):
            continue
        j = EXITS[e]
        for blk in ("DESIGN", "HOLDOUT"):
            ii = idx[blkv[idx] == blk]
            for reg, col in (("R1", "R1reg"), ("R4", "R4reg")):
                for s_ in (-1, 0, 1):
                    m = Y[col].values[ii] == s_
                    if not m.any():
                        continue
                    v = net[ii[m], j]; dd = Y.entry_ms.values[ii[m]] // DAY
                    dsum = pd.Series(v).groupby(dd).mean()
                    rg.append(dict(cohort=c, exit=e, blk=blk, regime=reg, sign=s_, N=int(m.sum()), windows=len(dsum), avg=v.mean(),
                                   WR=(v > 0).mean() * 100, win_pos=(dsum > 0).mean() * 100, avg_per_window=dsum.mean()))
    RG = pd.DataFrame(rg); RG.to_csv(os.path.join(OD, "regime.csv"), index=False)
    print(RG.round(2).to_string(), flush=True)

    # --- overlap with DIST
    import dist_wl_survivors as SV  # noqa: E402
    S_ = pd.read_pickle(os.path.join(L.R, "reports", "DIST_WL_2026-11-18_data", "rows.pkl"))
    REP = dict(kind="1D", var="rsi1h", lo=74.07103, hi=np.inf, mode="keep_only", var2="", lo2=np.nan, hi2=np.nan)
    km = SV.keep_mask(S_, REP) & (S_.LIVE_k.values != 3)
    Dt = S_[km][["pair", "entry_ms", "LIVE_x", "has_spot", "net_LIVE"]].copy()
    Dt["LIVE_x"] = Dt.LIVE_x.astype("int64")
    ov = []
    shared = {}
    for e, j in EXITS.items():
        idx = SEL[("HYB", e)]
        sh = np.zeros(len(idx), bool); kind = np.array([""] * len(idx), dtype=object)
        byp = {p: g for p, g in Dt.groupby("pair")}
        for q, i in enumerate(idx):
            g = byp.get(Y.pair.values[i])
            if g is None:
                continue
            T = Y.entry_ms.values[i]
            a = ((g.entry_ms.values <= T) & (g.LIVE_x.values > T)); b = np.abs(g.entry_ms.values - T) <= 3_600_000
            if a.any() or b.any():
                sh[q] = True; kind[q] = "open" if a.any() else "±60m"
        shared[e] = sh
        v = net[idx, j]
        for blk in ("ALL", "DESIGN", "HOLDOUT"):
            m = np.ones(len(idx), bool) if blk == "ALL" else blkv[idx] == blk
            ov.append(dict(exit=e, blk=blk, N=int(m.sum()), shared=int((sh & m).sum()), shared_pct=(sh & m).sum() / max(m.sum(), 1) * 100,
                           avg_shared=v[sh & m].mean() if (sh & m).any() else np.nan, avg_not=v[~sh & m].mean(),
                           N_not=int((~sh & m).sum()), tpd_not=(~sh & m).sum() / L.cal_days(blk)[0]))
    nsD = Dt[Dt.has_spot == 0]
    ov.append(dict(exit="DIST no-spot trades (rsi1h≥74.07)", blk="ALL", N=len(nsD), avg_shared=nsD.net_LIVE.mean()))
    OV = pd.DataFrame(ov); OV.to_csv(os.path.join(OD, "overlap.csv"), index=False)
    print(OV.round(3).to_string(), flush=True)
    # DIST no-spot trades that fall in a HYB-universe episode window: how many HYB also takes
    pickle.dump(dict(SEL=SEL, shared=shared, M2={k: np.nonzero(v)[0] for k, v in M2.items()}), open(os.path.join(OD, "core.pkl"), "wb"))
    Y.to_pickle(os.path.join(OD, "Y.pkl"))


if __name__ == "__main__":
    main()
