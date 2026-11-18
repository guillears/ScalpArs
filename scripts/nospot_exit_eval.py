#!/usr/bin/env python3
"""NOSPOT_EXIT (PREREG reports/NOSPOT_EXIT_2026-11-18_PREREG.md) — EVAL. Loads the 1m sim (nospot_exit_sim.py), builds nets (modelled
slip / flat 0.20 / 2× slip on dynamic exits, real funding), one position per pair per (cohort, cell), DESIGN grid + plateau + selection,
HOLDOUT once for the survivors, the family-wise within-episode shuffled null (500), the random-entry control, saved / deeper, path
decomposition, per half-year / BTC regime, concentration, LOO, Kelly (trade + day), day-block bootstrap CI, one-size DCR and the joint add
to EXH + DIST. Writes reports/NOSPOT_EXIT_2026-11-18_*.csv and NOSPOT_EXIT_data/eval.pkl.  Usage: venv/bin/python -I scripts/nospot_exit_eval.py"""
import glob, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_sleeve_lib as L   # noqa: E402
import nospot_exit_sim as S     # noqa: E402
D = S.D
OUT = os.path.join(L.R, "reports", "NOSPOT_EXIT_2026-11-18")
DAY = L.DAY
CL = S.cells(); NAMES = [c["name"] for c in CL]; NC = len(CL)
SELECT = [i for i, c in enumerate(CL) if c["fam"] in ("A", "B", "C", "D", "T", "S")]
REF = NAMES.index("REF_XE")
COH = ["NS0", "R1", "R4"]
RNG = np.random.default_rng(20261118)


def neighbours(j):
    c = CL[j]; f = c["fam"]
    same = [i for i in SELECT if CL[i]["fam"] == f and i != j]
    if f == "A":
        tps, sls = [2, 3, 4, 5], [6, 10, 12]
        return [i for i in same if (abs(tps.index(CL[i]["tp"]) - tps.index(c["tp"])) + abs(sls.index(CL[i]["sl"]) - sls.index(c["sl"]))) == 1]
    if f == "S":
        tps, sls, hs = [1.0, 1.5], [2, 3, 4, 6], [4, 24]
        return [i for i in same if (abs(tps.index(CL[i]["tp"]) - tps.index(c["tp"])) + abs(sls.index(CL[i]["sl"]) - sls.index(c["sl"]))
                                    + abs(hs.index(CL[i]["hold"]) - hs.index(c["hold"]))) == 1]
    if f in ("B", "C", "T"):
        order = sorted(same + [j], key=lambda i: (CL[i]["arm"], CL[i]["hold"]))
        k = order.index(j); return [order[x] for x in (k - 1, k + 1) if 0 <= x < len(order)]
    if f == "D":
        if CL[j]["mech"] == "trailatr":
            return [i for i in same if CL[i]["mech"] == "trailatr"] + [i for i in same if CL[i]["arm"] == 3 and CL[i]["mech"] == "trail"]
        arms, gs = [2, 3, 4], [1.5, 3]
        return [i for i in same if CL[i]["mech"] == "trail" and
                (abs(arms.index(CL[i]["arm"]) - arms.index(c["arm"])) + abs(gs.index(CL[i]["par"]) - gs.index(c["par"]))) == 1]
    return []


def load():
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    parts = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(D, "sim", "*.npz")))]
    Z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ok = Z["ok"]
    Z = {k: v[ok] for k, v in Z.items()}
    xi = Z["xi"]
    cols = ["pair", "entry_ms", "epi", "half", "blk", "slip", "q5", "atrp", "mcap", "rsx15", "vd", "oi_chg1h", "dd_hi", "ret1h",
            "topP_chg1h", "glA_chg1h"]
    Y = X.iloc[xi][cols].reset_index(drop=True)
    M = L.rule_masks(X)
    Y["NS0"] = L.ns0_mask(X)[xi]; Y["R1sig"] = M["R1_OI_UNWIND"][0][xi]; Y["R4c"] = M["R4_TOP_DISTRIB"][0][xi]
    Y["R1reg"] = X.R1.values[xi]; Y["R4reg"] = X.R4.values[xi]
    Y["xd"] = Z["xm"][:, REF] // DAY
    return X, Y, Z


def nets(Y, Z):
    s = Y.slip.values.astype(float)[:, None]
    k = Z["k"]; tk = k != 0; dyn = k == 3
    base = Z["r"].astype(float) + Z["f"].astype(float)
    net = base - s - 0.05 - np.where(tk, s + 0.05, 0.02)
    net20 = base - 0.20 - 0.05 - np.where(tk, 0.20 + 0.05, 0.02)
    netd2 = net - np.where(dyn, s, 0.0)
    return net, net20, netd2


def one_pos(pair, T, XM):
    o = np.lexsort((T, pair)); keep = np.zeros(len(o), bool); lp = None; busy = -1
    for q, i in enumerate(o):
        if pair[i] != lp:
            lp = pair[i]; busy = -1
        if T[i] < busy:
            continue
        keep[q] = True; busy = XM[i]
    out = o[keep]
    return out[np.argsort(T[out], kind="stable")]


def wilson_floor(k, n, z=1.645):
    if n <= 0:
        return 0.0
    p = k / n; den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0))) / den


def kel(v):
    w = v[v > 0]; lo = v[v <= 0]
    if not len(w) or not len(lo):
        return np.nan, np.nan, np.nan
    aw, al = w.mean(), -lo.mean(); pf = wilson_floor(len(w), len(v))
    return pf - (1 - pf) / (aw / al), pf * 100, al / (aw + al) * 100


def onesize(r, d0, d1):
    r = r.reindex(np.arange(d0, d1), fill_value=0.0)
    eq = np.cumprod(1 + r.values); Dn = d1 - d0
    return dict(dcr=(eq[-1] ** (1 / Dn) - 1) * 100 if eq[-1] > 0 else np.nan,
                maxdd=(1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max() * 100, worst_day=r.min() * 100)


def boot_ci(v, days, nb=2000, block=10):
    s = pd.Series(v).groupby(days).agg(["sum", "count"])
    dd = np.arange(days.min(), days.max() + 1); s = s.reindex(dd, fill_value=0)
    n = len(dd); nbk = int(np.ceil(n / block)); st = RNG.integers(0, max(n - block, 1), (nb, nbk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :n]
    a = s["sum"].values[idx].sum(1) / np.maximum(s["count"].values[idx].sum(1), 1)
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def stats(Y, Z, NET, idx, j, blk="ALL"):
    net, net20, netd2 = NET
    Dn, d0, d1 = L.cal_days(blk)
    rec = dict(N=len(idx), tpd=len(idx) / Dn)
    if len(idx) < 2:
        return rec
    v = net[idx, j]; xd = Z["xm"][idx, j] // DAY
    k = Z["k"][idx, j]
    kt, pf, be = kel(v)
    dv = pd.Series(v).groupby(xd).sum().values
    kd, pfd, _ = kel(dv)
    sv = np.sort(v); tot = v.sum()
    pp = pd.Series(v).groupby(Y.pair.values[idx]).sum(); dd_ = pd.Series(v).groupby(xd).sum()
    r = pd.Series(v * L.SIZE / 100).groupby(xd).sum()
    os_ = onesize(r, d0, d1)
    loo = min(np.delete(v, np.nonzero(Y.pair.values[idx] == q)[0]).mean() for q in pp.index[np.argsort(-pp.values)[:5]])
    rec.update(WR=(v > 0).mean() * 100, avg=v.mean(), avg_win=v[v > 0].mean() if (v > 0).any() else np.nan,
               avg_loss=v[v <= 0].mean() if (v <= 0).any() else np.nan, f20=net20[idx, j].mean(), d2=netd2[idx, j].mean(),
               b5=sv[: int(np.floor(len(v) * 0.95))].mean() if len(v) >= 20 else np.nan, loo_pair=loo,
               kelly=kt, WRfloor=pf, BE=be, kelly_day=kd, WRfloor_day=pfd, ndays=len(np.unique(xd)), npairs=len(pp),
               pair_share=pp.max() / tot * 100 if tot > 0 else np.nan, pair_top=pp.idxmax(),
               day_share=dd_.max() / tot * 100 if tot > 0 else np.nan, **os_,
               tp=(k == 0).mean() * 100, sl=(k == 1).mean() * 100, time=(k == 2).mean() * 100, dyn=(k == 3).mean() * 100,
               fund=Z["f"][idx, j].mean(), worked=Z["worked"][idx, j].mean() * 100, src1m=Z["src1m"][idx].mean() * 100,
               hold_h=((Z["xm"][idx, j] - Y.entry_ms.values[idx]) / 3.6e6).mean())
    for h, _, _ in L.HALVES:
        m = Y.half.values[idx] == h
        rec[f"h_{h}_N"] = int(m.sum()); rec[f"h_{h}"] = v[m].mean() if m.any() else np.nan
    return rec


def path_group(Z, NET, idx, j):
    v = NET[0][idx, j]; wk = Z["worked"][idx, j] == 1; k = Z["k"][idx, j]; rec_ = Z["rec"][idx, j] == 1; best = Z["best"][idx, j]
    return np.where(v > 0, np.where((best - 0.09 - v) >= np.maximum(1.0, v), "LEFT_ON_TABLE", "CAPTURED"),
                    np.where(wk, "GAVE_BACK", np.where((k != 0) & (k != 2) & rec_, "STOPPED_RECOVERED",
                                                        np.where((k == 1) | (k == 3), "NEVER_WORKED", "DRIFT_LOSS"))))


def main():
    X, Y, Z = load()
    NET = nets(Y, Z); net = NET[0]
    print("rows", len(Y), "1m share", Z["src1m"].mean(), flush=True)
    cm = {"NS0": Y.NS0.values, "R1": Y.R1sig.values, "R4": Y.R4c.values, "RND": np.ones(len(Y), bool)}
    pair = Y.pair.values; T = Y.entry_ms.values.astype(np.int64)
    blkv = Y.blk.values
    # selected trade sets
    SEL = {}
    for c in COH:
        ii = np.nonzero(cm[c])[0]
        for j in range(NC):
            SEL[(c, j)] = ii[one_pos(pair[ii], T[ii], Z["xm"][ii, j])]
    allr = np.arange(len(Y))
    for j in range(NC):
        SEL[("RND", j)] = allr
    # 1. grid: every cohort × cell × block
    rows = []
    for (c, j), idx in SEL.items():
        for blk in ("DESIGN", "HOLDOUT", "ALL"):
            ii = idx if blk == "ALL" else idx[blkv[idx] == blk]
            rows.append(dict(cohort=c, cell=NAMES[j], fam=CL[j]["fam"], blk=blk, **stats(Y, Z, NET, ii, j, blk)))
    G = pd.DataFrame(rows)
    # 2. selection on DESIGN
    Gd = G[G.blk == "DESIGN"].set_index(["cohort", "cell"])
    sel_rows = []
    for c in COH:
        refd = Gd.loc[(c, "REF_XE"), "avg"]
        for j in SELECT:
            g = Gd.loc[(c, NAMES[j])]
            hv = [g[f"h_{h}"] for h, s_, e_ in L.HALVES if e_ <= L.DES[1] and g[f"h_{h}_N"] >= 10]
            hp = np.mean([x > 0 for x in hv]) if hv else 0.0
            nb = neighbours(j); npos = np.mean([Gd.loc[(c, NAMES[i]), "avg"] > 0 for i in nb]) if nb else 0.0
            nmean = np.mean([Gd.loc[(c, NAMES[i]), "avg"] for i in nb]) if nb else np.nan
            surv = (g.N >= 30) and (g.avg > 0) and (g.avg > refd) and (hp >= 2 / 3) and (npos >= 2 / 3)
            sel_rows.append(dict(cohort=c, cell=NAMES[j], fam=CL[j]["fam"], N=g.N, avg=g.avg, ref=refd, halves_pos=hp, plateau_pos=npos,
                                 plateau_mean=nmean, neighbours=",".join(NAMES[i] for i in nb), survivor=surv))
    SS = pd.DataFrame(sel_rows)
    sv = SS[SS.survivor].sort_values("avg", ascending=False)
    picks = []; seen = set()
    for _, r_ in sv.iterrows():
        key = (r_.cohort, r_.fam)
        if key in seen:
            continue
        seen.add(key); picks.append((r_.cohort, r_.cell))
        if len(picks) == 3:
            break
    print("survivors", len(sv), "picks", picks, flush=True)
    # 3. null: within-episode shuffle, 500 draws, 90 cells (full span) + matched random per block
    ep = Y.epi.values
    order = np.argsort(ep, kind="stable"); eps_sorted = ep[order]
    ue, start, cnt = np.unique(eps_sorted, return_index=True, return_counts=True)
    pos = {e: i for i, e in enumerate(ue)}
    NDR = 500
    null_rows = []; draws_all = {}
    for c in COH:
        for j in SELECT + [REF]:
            idx = SEL[(c, j)]
            ei = np.array([pos[e] for e in ep[idx]])
            u = RNG.random((NDR, len(idx)))
            rr = order[start[ei][None, :] + np.floor(u * cnt[ei][None, :]).astype(int)]
            V = net[rr, j]
            dr = V.mean(1)
            isd = blkv[idx] == "DESIGN"
            draws_all[(c, j)] = dr
            obs = net[idx, j].mean()
            null_rows.append(dict(cohort=c, cell=NAMES[j], obs=obs, null_mean=dr.mean(), null_sd=dr.std(), stat=obs - dr.mean(),
                                  p_cell=(dr >= obs).mean(), rnd_design=V[:, isd].mean() if isd.any() else np.nan,
                                  rnd_holdout=V[:, ~isd].mean() if (~isd).any() else np.nan))
    NU = pd.DataFrame(null_rows)
    fam = [(c, j) for c in COH for j in SELECT]
    Dmat = np.vstack([draws_all[k] - draws_all[k].mean() for k in fam])     # cells × draws
    mx = Dmat.max(0)
    NU["p_fw"] = [(mx >= s).mean() for s in NU.stat]
    # 4. random-entry control: Δ (cell − REF) on the same entries; cohort (REF's one-pos set), RND all bars, matched random
    ctl = []
    for c in COH + ["RND"]:
        idx = SEL[(c, REF)]
        for blk in ("DESIGN", "HOLDOUT", "ALL"):
            ii = idx if blk == "ALL" else idx[blkv[idx] == blk]
            for j in SELECT + [NAMES.index("X_TP10_SL10")]:
                d = net[ii, j] - net[ii, REF]
                ctl.append(dict(cohort=c, blk=blk, cell=NAMES[j], N=len(ii), ref=net[ii, REF].mean(), cell_avg=net[ii, j].mean(), delta=d.mean()))
    CT = pd.DataFrame(ctl)
    # 5. path decomposition (REF + every pick, + each cohort's best cells) on the cell's own selected set
    pg = []
    for c in COH + ["RND"]:
        for j in [REF] + [NAMES.index(p[1]) for p in picks] + [NAMES.index(n) for n in ("A_TP3_SL10", "B_BE3", "C_L31", "D_T3_G1.5", "D_T3_G3", "T_24h")]:
            idx = SEL[(c, j)] if c != "RND" else SEL[(c, j)][::7]
            g = path_group(Z, NET, idx, j); v = net[idx, j]; tl = v[v <= 0].sum()
            for grp in ["CAPTURED", "LEFT_ON_TABLE", "GAVE_BACK", "NEVER_WORKED", "STOPPED_RECOVERED", "DRIFT_LOSS"]:
                m = g == grp
                pg.append(dict(cohort=c, cell=NAMES[j], group=grp, N=int(m.sum()), share=m.mean() * 100, sum_pct=v[m].sum(),
                               loss_share=v[m].sum() / tl * 100 if grp not in ("CAPTURED", "LEFT_ON_TABLE") else np.nan,
                               reach_med=np.median(Z["reach"][idx[m], j]) if m.any() else np.nan))
    PG = pd.DataFrame(pg).drop_duplicates(["cohort", "cell", "group"])
    # 6. saved / deeper
    sd = []
    xb = NAMES.index("X_TP10_SL10"); tb = NAMES.index("T_24h")
    for c in COH + ["RND"]:
        idx = SEL[(c, REF)]
        for j in SELECT:
            cj = CL[j]
            if cj["mech"] != "none":
                b = xb if cj["fam"] in ("B", "C") else tb
                d = Z["r"][idx, j].astype(float) - Z["r"][idx, b].astype(float)
                fired = Z["k"][idx, j] == 3
                bad = np.abs(d[~fired]) > 1e-4
                dn = net[idx, j] - net[idx, b]
                saved = fired & (net[idx, b] < net[idx, j]); cost = fired & ~saved
                sd.append(dict(cohort=c, cell=NAMES[j], kind="mech_vs_base", base=NAMES[b], N=len(idx), unfired=int((~fired).sum()),
                               unfired_delta_nonzero=int(bad.sum()), unfired_delta_sum=dn[~fired].sum(),
                               saved_N=int(saved.sum()), saved_sum=dn[saved].sum(), cost_N=int(cost.sum()), cost_sum=dn[cost].sum(),
                               net_delta_avg=dn.mean()))
            kr = Z["k"][idx, REF]; d = net[idx, j] - net[idx, REF]
            for nm, m in (("REF stopped", kr == 1), ("REF TP", kr == 0), ("REF time", kr == 2)):
                sv_ = m & (d > 0); dp = m & (d <= 0)
                sd.append(dict(cohort=c, cell=NAMES[j], kind="vs_REF:" + nm, base="REF_XE", N=int(m.sum()), saved_N=int(sv_.sum()),
                               saved_sum=d[sv_].sum(), cost_N=int(dp.sum()), cost_sum=d[dp].sum(), net_delta_avg=d[m].mean() if m.any() else np.nan))
    SD = pd.DataFrame(sd)
    # 7. regime per half (R1 daily sign, R4 5m sign) for REF + picks
    rg = []
    for c, cell in [(c, "REF_XE") for c in COH] + picks:
        j = NAMES.index(cell); idx = SEL[(c, j)]; v = net[idx, j]
        for reg, col in (("R1", "R1reg"), ("R4", "R4reg")):
            for s_ in (-1, 0, 1):
                if reg == "R1" and s_ == 0:
                    continue
                m = Y[col].values[idx] == s_
                rec = dict(cohort=c, cell=cell, regime=reg, sign=s_, N=int(m.sum()), avg=v[m].mean() if m.any() else np.nan)
                for h, _, _ in L.HALVES:
                    mh = m & (Y.half.values[idx] == h)
                    rec[h] = f"{v[mh].mean():+.2f} ({mh.sum()})" if mh.sum() else "–"
                rg.append(rec)
    RG = pd.DataFrame(rg)
    # 8. bootstrap CI + joint add (HOLDOUT and ALL)
    E = pd.read_csv(os.path.join(L.R, "reports", "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    tk = E.exit_kind.astype(str).str.upper().str.startswith("TP")
    E["net"] = E.r - 0.10 - np.where(tk, 0, 0.10) - 0.05 - np.where(tk, 0.02, 0.05) + E.fund
    E["xd"] = E.exit_ms // DAY
    Dd = pd.read_csv(os.path.join(L.R, "reports", "DIST_WL_2026-11-18_trades.csv.gz"), usecols=["pair", "entry_ms", "rsi1h", "in_LIVE", "net_LIVE", "LIVE_x"])
    Dd = Dd[(Dd.in_LIVE == True) & (Dd.rsi1h >= 74.07)].copy(); Dd["xd"] = Dd.LIVE_x.astype("int64") // DAY  # noqa: E712
    jt = []
    cand = [(c, "REF_XE") for c in COH] + picks + [(c, n) for c in COH for n in ("A_TP3_SL10", "B_BE3", "C_L31", "D_T3_G1.5", "T_24h")]
    cand = list(dict.fromkeys(cand))
    for blk in ("ALL", "HOLDOUT"):
        Dn, d0, d1 = L.cal_days(blk); t0 = d0 * DAY
        e_ = E[E.entry_ms >= t0]; dd_ = Dd[Dd.entry_ms >= t0]
        base = pd.Series(e_.net.values * L.SIZE / 100).groupby(e_.xd.values).sum().add(
            pd.Series(dd_.net_LIVE.values * L.SIZE / 100).groupby(dd_.xd.values).sum(), fill_value=0)
        jt.append(dict(blk=blk, book="EXH + DIST", **onesize(base, d0, d1)))
        for c, cell in cand:
            j = NAMES.index(cell); idx = SEL[(c, j)]; idx = idx[T[idx] >= t0]
            s_ = pd.Series(net[idx, j] * L.SIZE / 100).groupby(Z["xm"][idx, j] // DAY).sum()
            a = onesize(s_, d0, d1); b = onesize(base.add(s_, fill_value=0), d0, d1)
            ci = boot_ci(net[idx, j], Z["xm"][idx, j] // DAY)
            jt.append(dict(blk=blk, book=f"{c} · {cell} alone", ci_lo=ci[0], ci_hi=ci[1], **a))
            jt.append(dict(blk=blk, book=f"EXH + DIST + {c} · {cell}", **b))
    JT = pd.DataFrame(jt)
    for nm, df in (("grid", G), ("select", SS), ("null", NU), ("control", CT), ("paths", PG), ("saved", SD), ("regime", RG), ("joint", JT)):
        df.to_csv(f"{OUT}_{nm}.csv", index=False)
    pickle.dump(dict(SEL=SEL, picks=picks, names=NAMES), open(os.path.join(D, "eval.pkl"), "wb"))
    pd.set_option("display.width", 250)
    print(SS.sort_values("avg", ascending=False).head(25).round(3).to_string())
    print(NU.sort_values("stat", ascending=False).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
