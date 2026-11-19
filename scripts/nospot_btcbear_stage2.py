#!/usr/bin/env python3
"""NOSPOT_BTCBEAR stage 2 (PREREG Addendum E + the remaining main analyses):
E3 exit selection (DESIGN, ON-gated) for the E1 picks (+ the contaminated source cell R4 × a, descriptive) · HOLDOUT once · rotation null
over the full family (main + picks × 47 exits) · RND control per exit · saved / deeper vs REF · Δ = 0 on never-armed fills · W/L (E5)
under REF and the picked exit · path groups ON vs OFF · spot-coin EXHAUSTION split · joint book (joint_short_sim) · clustering vs EXH on
ON days · replays · chart.  Usage: venv/bin/python -I scripts/nospot_btcbear_stage2.py"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_btcbear_lib as B   # noqa: E402
L = B.L; S = B.S; EV = B.EV
OUT = os.path.join(B.R, "reports", "NOSPOT_BTCBEAR_2026-11-18")
CL = S.cells(); NAMES = [c["name"] for c in CL]
SELECT = EV.SELECT; REF = NAMES.index("REF_XE")
DAY = B.DAY
P = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); P.append(s)


def main():
    RS = B.regime_series(); Y, Z = B.load(); NET = B.nets(Y, Z); net, net20, netd2 = NET
    T = Y.entry_ms.values.astype(np.int64); blk = Y.blk.values; pair = Y.pair.values; half = Y.half.values; day = Y.day.values
    REG = {d: B.at(RS, d, T) for d in B.DEFS}
    st1 = pd.read_pickle(os.path.join(B.DD, "stage1.pkl")); picks = st1["picks"]
    log("E1 picks", picks)
    study = list(picks) + [("a", "R4")]   # the last one = the contaminated source, descriptive only
    # ---------- E3: exit family on DESIGN ON-gated trades
    sel = []; SETS = {}
    for d, c in study:
        base = np.nonzero(Y[c].values & (REG[d] == -1))[0]
        for j in [REF] + SELECT:
            ii = B.one_pos(Y, Z, base, j); SETS[(d, c, j)] = ii
        for bk in ("DESIGN", "HOLDOUT"):
            for j in [REF] + SELECT:
                ii = SETS[(d, c, j)]; q = ii[blk[ii] == bk]
                rec = dict(defn=d, cohort=c, cell=NAMES[j], fam=CL[j]["fam"], blk=bk)
                rec.update(B.cell_stats(net[q, j], net20[q, j], day[q], pair[q], half[q]))
                sel.append(rec)
    SL = pd.DataFrame(sel)
    surv = []
    for d, c in study:
        G = SL[(SL.defn == d) & (SL.cohort == c) & (SL.blk == "DESIGN")].set_index("cell")
        refd = G.loc["REF_XE", "avg"]
        for j in SELECT:
            g = G.loc[NAMES[j]]
            hv = [g[f"h_{h}"] for h, s_, e_ in L.HALVES if e_ <= L.DES[1] and g[f"h_{h}_N"] >= 5]
            hp = np.mean([x > 0 for x in hv]) if hv else 0.0
            nb = EV.neighbours(j); npos = np.mean([G.loc[NAMES[i], "avg"] > 0 for i in nb]) if nb else 0.0
            ok = (g.N >= 30) and (g.avg > 0) and (g.avg > refd) and (hp >= 2 / 3) and (npos >= 2 / 3)
            surv.append(dict(defn=d, cohort=c, cell=NAMES[j], N=g.N, avg=g.avg, ref=refd, halves_pos=hp, plateau_pos=npos, survivor=ok))
    SV = pd.DataFrame(surv); SV.to_csv(OUT + "_e3_select.csv", index=False); SL.to_csv(OUT + "_e3_grid.csv", index=False)
    PICK = {}
    for d, c in study:
        s = SV[(SV.defn == d) & (SV.cohort == c) & SV.survivor].sort_values("avg", ascending=False)
        PICK[(d, c)] = s.cell.iloc[0] if len(s) else None
        log("E3", d, c, "survivors", len(s), "pick", PICK[(d, c)], list(s.cell.head(5)))
    # ---------- candidate cells for every later step
    CAND = []
    for d, c in study:
        CAND.append((d, c, "REF_XE"))
        if PICK[(d, c)]:
            CAND.append((d, c, PICK[(d, c)]))
    CAND.append(("a", "R4", "A_TP4_SL10"))
    CAND = list(dict.fromkeys(CAND))
    # ---------- rotation null: main family (from stage 1 definition) + picks × all exits
    lo, hi = B.HOLD0, B.SPAN1
    fam = []
    for c in B.COHORTS:
        for e in ("REF_XE", "A_TP4_SL10"):
            j = NAMES.index(e); ii = B.one_pos(Y, Z, np.nonzero(Y[c].values)[0], j); ii = ii[blk[ii] == "HOLDOUT"]
            for d in B.DEFS:
                fam.append((d, c, e, net[ii, j], T[ii]))
    for d, c in study:
        for j in SELECT:
            ii = B.one_pos(Y, Z, np.nonzero(Y[c].values)[0], j); ii = ii[blk[ii] == "HOLDOUT"]
            fam.append((d, c, NAMES[j], net[ii, j], T[ii]))
    def fstat(sh):
        o = {}
        for d, c, e, v, tt in fam:
            r = B.at(RS, d, tt, sh, lo, hi) == -1
            o[(d, c, e)] = v[r].mean() - v.mean() if r.sum() >= 10 else np.nan
        return o
    obs = fstat(0); rng = np.random.default_rng(12); NB = 500; mx = np.zeros(NB); per = {k: np.zeros(NB) for k in obs}
    for b in range(NB):
        st = fstat(int(rng.uniform(14 * DAY, hi - lo - 14 * DAY)))
        mx[b] = np.nanmax(list(st.values()))
        for k, x in st.items():
            per[k][b] = x
    NUL = pd.DataFrame([dict(defn=k[0], cohort=k[1], exit=k[2], stat=x, p_cell=np.mean(per[k] >= x), p_fw=np.mean(mx >= x))
                        for k, x in obs.items() if np.isfinite(x)])
    NUL.to_csv(OUT + "_null_full.csv", index=False)
    # ---------- per candidate: HOLDOUT bar, control, saved/deeper, Δ=0, paths
    bar = []; ctl = []; sd = []; pg = []
    for d, c, e in CAND:
        j = NAMES.index(e)
        base_on = np.nonzero(Y[c].values & (REG[d] == -1))[0]
        ii = B.one_pos(Y, Z, base_on, j)
        for bk in ("DESIGN", "HOLDOUT", "ALL"):
            q = ii if bk == "ALL" else ii[blk[ii] == bk]
            rec = dict(defn=d, cohort=c, exit=e, blk=bk); rec.update(B.cell_stats(net[q, j], net20[q, j], day[q], pair[q], half[q]))
            m_r = (REG[d] == -1) & ((blk == bk) if bk != "ALL" else True)
            rec["rnd_on"] = net[m_r, j].mean(); rec["edge_vs_rnd"] = rec.get("avg", np.nan) - rec["rnd_on"]
            nr = NUL[(NUL.defn == d) & (NUL.cohort == c) & (NUL.exit == e)]
            rec["p_cell"] = nr.p_cell.iloc[0] if len(nr) else np.nan; rec["p_fw"] = nr.p_fw.iloc[0] if len(nr) else np.nan
            bar.append(rec)
        for bk in ("DESIGN", "HOLDOUT"):
            for state, sv in (("ON", -1), ("OFF", 1)):
                m = (REG[d] == sv) & (blk == bk)
                ctl.append(dict(defn=d, cohort=c, exit=e, blk=bk, state=state, rnd_N=int(m.sum()), rnd_avg=net[m, j].mean()))
        # saved / deeper vs REF on REF's ON-gated set
        if e != "REF_XE":
            r0 = B.one_pos(Y, Z, base_on, REF)
            for bk in ("DESIGN", "HOLDOUT"):
                q = r0[blk[r0] == bk]; dv = net[q, j] - net[q, REF]; kr = Z["k"][q, REF]
                for kn, kv in (("REF TP", 0), ("REF stop", 1), ("REF time", 2)):
                    m = kr == kv
                    sd.append(dict(defn=d, cohort=c, exit=e, blk=bk, ref_outcome=kn, N=int(m.sum()),
                                   saved_N=int((dv[m] > 0).sum()), saved_sum=dv[m][dv[m] > 0].sum(),
                                   cost_N=int((dv[m] < 0).sum()), cost_sum=dv[m][dv[m] < 0].sum(), avg_delta=dv[m].mean() if m.any() else np.nan))
            if CL[j]["mech"] != "none":
                bn = "T_24h" if CL[j]["mech"] in ("trail", "trailatr") else "X_TP10_SL10"
                jb = NAMES.index(bn); kk = Z["k"][r0, j]; unf = kk != 3
                log("Δ=0 check", d, c, e, "vs", bn, "unfired", int(unf.sum()), "non-zero Δ", int((np.abs(net[r0, j] - net[r0, jb])[unf] > 1e-6).sum()))
        for state, sv in (("ON", -1), ("OFF", 1)):
            q = B.one_pos(Y, Z, np.nonzero(Y[c].values & (REG[d] == sv))[0], j)
            g = EV.path_group(Z, NET, q, j); v = net[q, j]
            rec = dict(defn=d, cohort=c, exit=e, state=state, N=len(q), avg=v.mean())
            lossT = -v[v < 0].sum()
            for gg in ("CAPTURED", "LEFT_ON_TABLE", "GAVE_BACK", "NEVER_WORKED", "STOPPED_RECOVERED", "DRIFT_LOSS"):
                mg = g == gg; rec[gg] = mg.mean() * 100
                rec["loss_" + gg] = -v[mg & (v < 0)].sum() / lossT * 100 if lossT > 0 else np.nan
            rec["squeeze_share"] = rec["NEVER_WORKED"] + rec["STOPPED_RECOVERED"]
            rec["worked"] = Z["worked"][q, j].mean() * 100
            pg.append(rec)
    BAR = pd.DataFrame(bar); BAR.to_csv(OUT + "_bar.csv", index=False)
    pd.DataFrame(ctl).to_csv(OUT + "_control_cand.csv", index=False)
    pd.DataFrame(sd).to_csv(OUT + "_saved.csv", index=False); PG = pd.DataFrame(pg); PG.to_csv(OUT + "_paths.csv", index=False)
    # paths for the base cohorts under a / d, REF and TP4 SL10 (squeeze-share table)
    pg2 = []
    for c in ("NS0", "R1", "R4", "R10"):
        for e in ("REF_XE", "A_TP4_SL10"):
            j = NAMES.index(e)
            for d in B.DEFS:
                for state, sv in (("ON", -1), ("OFF", 1)):
                    q = B.one_pos(Y, Z, np.nonzero(Y[c].values & (REG[d] == sv))[0], j)
                    g = EV.path_group(Z, NET, q, j)
                    pg2.append(dict(cohort=c, exit=e, defn=d, state=state, N=len(q), avg=net[q, j].mean(),
                                    squeeze=np.isin(g, ["NEVER_WORKED", "STOPPED_RECOVERED"]).mean() * 100,
                                    gave_back=(g == "GAVE_BACK").mean() * 100, worked=Z["worked"][q, j].mean() * 100,
                                    reach_med=np.median(Z["reach"][q, j])))
    pd.DataFrame(pg2).to_csv(OUT + "_squeeze.csv", index=False)
    # ---------- W/L (E5) inside the bear-ON cohort of each study pair, under REF and the pick (or TP4 SL10)
    skip = {"pair", "entry_ms", "epi", "half", "blk", "day", "NS0"} | set(B.RULEKEY)
    cols = [c_ for c_ in Y.columns if c_ not in skip and Y[c_].dtype.kind in "fiub"]
    wl = []; cov = []
    rngw = np.random.default_rng(13)
    for d, c in study:
        exs = ["REF_XE", PICK[(d, c)] or "A_TP4_SL10"]
        for e in exs:
            j = NAMES.index(e); ii = B.one_pos(Y, Z, np.nonzero(Y[c].values & (REG[d] == -1))[0], j)
            for bk in ("DESIGN", "HOLDOUT"):
                q = ii[blk[ii] == bk]; v = net[q, j]
                bins = {}
                for col in cols:
                    x = Y[col].values[q].astype(float); ok = np.isfinite(x)
                    if bk == "DESIGN" and e == exs[0]:
                        cov.append(dict(defn=d, cohort=c, var=col, scored=ok.mean() * 100))
                    if ok.sum() < 40:
                        continue
                    for gname, qs in (("median", [0.5]), ("terc", [1 / 3, 2 / 3]), ("quint", [0.2, 0.4, 0.6, 0.8])):
                        cuts = np.unique(np.quantile(x[ok], qs))
                        lab = np.where(ok, np.searchsorted(cuts, x, side="right"), -1)
                        bins[(col, gname)] = (lab, len(cuts) + 1)
                def best(vv):
                    out = {}
                    for key, (lab, nb_) in bins.items():
                        bb = []
                        for k_ in range(nb_):
                            m = lab == k_
                            if m.sum() >= 20:
                                bb.append((vv[m].mean() - vv.mean(), k_, m.sum()))
                        if bb:
                            out[key] = max(bb)
                    return out
                ob = best(v)
                if bk == "DESIGN":
                    nm = np.zeros(300)
                    for b in range(300):
                        r_ = best(rngw.permutation(v)); nm[b] = max(x_[0] for x_ in r_.values())
                for key, (sp, kb, nn) in ob.items():
                    wl.append(dict(defn=d, cohort=c, exit=e, blk=bk, var=key[0], gran=key[1], best_bin=kb, n_bin=nn, lift=sp,
                                   p_fw=np.mean(nm >= sp)))
        log("W/L done", d, c)
    WL = pd.DataFrame(wl); WL.to_csv(OUT + "_wl.csv", index=False); pd.DataFrame(cov).to_csv(OUT + "_coverage.csv", index=False)
    pd.to_pickle(dict(PICK=PICK, CAND=CAND, study=study), os.path.join(B.DD, "stage2.pkl"))
    open(os.path.join(B.DD, "stage2.log"), "w").write("\n".join(P))


if __name__ == "__main__":
    main()
