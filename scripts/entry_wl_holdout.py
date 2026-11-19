#!/usr/bin/env python3
"""ENTRY_WL (PREREG reports/ENTRY_WL_2026-11-18_PREREG.md) — HOLDOUT (read once) + action pricing. Research only, BACKTEST.
For every frozen forward candidate (reports/ENTRY_WL_2026-11-18_data/<sleeve>_frozen.json): HOLDOUT RUNNER share per side, AUC,
Δ avg net % with day-block CI, half-years, BTC daily sign, concentration, best-5 %-out, 0.20 % slip; the theory logistic's HOLDOUT AUC.
Then for each candidate: (a) entry filter (expectancy bar on the blocked side) · (b) size tilt · (c) conditional exit on the predicted
give-backs (cell picked on DESIGN bad-side fills), each with N, trades/day, WR, avg, one-size DCR 0.61, max DD, worst day, Kelly at the
Wilson floor (trade / day), joint book (joint_short_sim.sim, EXH + FB + L4 + DIST, EXH and DIST on their 1m REF nets), LOO, haircut.
Usage: venv/bin/python -I scripts/entry_wl_holdout.py exh|dist"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import entry_wl_lib as W        # noqa: E402
import entry_wl_design as WD    # noqa: E402
import review_math as RM        # noqa: E402

DAY = W.DAY
TS = lambda s: pd.Timestamp(s).value // 10**6 // DAY
SPANS = {"DESIGN": (TS("2023-07-01"), TS("2025-07-01")), "HOLDOUT": (TS("2025-07-01"), TS("2026-11-18")),
         "ALL": (TS("2023-07-01"), TS("2026-11-18"))}
HO_HALVES = ["25-H2", "26-H1", "26-H2"]


def cell_matrix(sleeve, D):
    """per-fill net % for every exit cell (paired entries), exit ms, kind; column 0 = REF."""
    if sleeve == "exh":
        import exh_exit2_eval as XE
        X, Z = XE.load("EXH")
        assert (X.pair.astype(str).values == D.pair.values).all() and (X.entry_ms.values == D.entry_ms.values).all()
        NET = XE.nets(X, Z); N20 = XE.nets(X, Z, flat=0.20)
        names = list(XE.NAMES); fam = [c["fam"] for c in XE.CL]
        return names, fam, NET, N20, Z["xm"], Z["k"]
    import dist_exit_eval as DE
    Z = DE.load_set("L1"); import dist_exit_sim as DS
    F = DS.frames()["L1"]; slip = F.slip.values
    V = DE.net_mat(Z, slip); V20 = DE.net_mat(Z, slip, flat=0.20)
    ii = D.zi.values
    assert np.allclose(V[ii, 0], D.net.values)
    return list(DE.NAMES), list(DE.FAM), V[ii], V20[ii], Z["xm"][ii], Z["k"][ii]


_JT = {}


def joint(sleeve, D, net, xms, stop, keep, ho_only=False):
    """one-account joint book EXH + FB + L4 + DIST at 0.609 × equity. `sleeve` rows replaced by D[keep] with net / exit."""
    import joint_short_sim as JS
    if "T" not in _JT:
        T0 = JS.load()[0]
        _JT["T"] = T0[T0.sleeve.isin(["EXH", "FB", "L4", "DIST"])].copy()
        _JT["other"] = {}
        for nm, s in (("exh", "EXH"), ("dist", "DIST")):
            Dx = W.load(nm)
            _JT["other"][nm] = pd.DataFrame(dict(sleeve=s, pair=Dx.pair.values, entry_ms=Dx.entry_ms.values.astype("int64"),
                                                 exit_ms=Dx.x.values.astype("int64"), net=Dx.net.values, nt=1 + (Dx.k.values != 0),
                                                 stop=Dx.k.values == 1, depth=JS.depth_of(Dx.q5.values)))
    T = _JT["T"]
    base = T[~T.sleeve.isin(["EXH", "DIST"])]
    other = _JT["other"]["dist" if sleeve == "exh" else "exh"]
    me = pd.DataFrame(dict(sleeve="EXH" if sleeve == "exh" else "DIST", pair=D.pair.values[keep], entry_ms=D.entry_ms.values[keep].astype("int64"),
                           exit_ms=np.asarray(xms)[keep].astype("int64"), net=np.asarray(net)[keep], nt=2, stop=np.asarray(stop)[keep],
                           depth=JS.depth_of(D.q5.values[keep])))
    Tc = pd.concat([base, other, me], ignore_index=True)
    Tc = Tc[(Tc.entry_ms >= JS.D0ms) & (Tc.entry_ms < JS.D1ms)].copy(); Tc["prio"] = Tc.sleeve.map(JS.PRIO)
    Tc = Tc.sort_values(["entry_ms", "prio"]).reset_index(drop=True)
    s = JS.sim(Tc); st = JS.stats(s["r_day"]); hm = JS.DAYS >= SPANS["HOLDOUT"][0]; sh = JS.stats(s["r_day"][hm])
    return dict(j_dcr=st["dcr"], j_maxdd=st["maxdd"], j_worst=st["worst_day"], j_dcr_ho=sh["dcr"], j_maxdd_ho=sh["maxdd"], j_worst_ho=sh["worst_day"])


def book(v, xms, days_entry, blk, size=W.SIZE):
    d0, d1 = SPANS[blk]
    os_, r = W.onesize(v, xms, d0, d1, size)
    kt, pf, be = W.kel(v)
    kd, pfd, _ = W.kel(pd.Series(v).groupby(np.asarray(xms) // DAY).sum().values)
    return dict(N=len(v), tpd=len(v) / (d1 - d0), WR=(v > 0).mean() * 100, avg=v.mean(), BE=be, kelly_t=kt, WRfloor=pf, kelly_d=kd,
                dcr=os_["dcr"], maxdd=os_["maxdd"], worst=os_["worst_day"])


def expectancy_bar(v, days, pairs, be_wr):
    n = len(v)
    if n < 2:
        return dict(eb_N=n, eb_pass=False)
    q = W.day_boot_mean(v, days)
    loss = v.sum()
    tw = pd.Series(v).groupby(days).sum(); tp = pd.Series(v).groupby(pairs).sum()
    tws = tw.min() / loss * 100 if loss < 0 else np.nan; tps = tp.min() / loss * 100 if loss < 0 else np.nan
    wr = (v > 0).mean() * 100
    items = dict(eb_N=n, eb_WR=wr, eb_BE=be_wr, eb_avg=v.mean(), eb_hi95=q[2], eb_windows=len(tw), eb_top_window=tws, eb_top_pair=tps)
    items["eb_pass"] = bool(wr < be_wr and q[2] < 0 and len(tw) >= 8 and (tws < 50) and (tps < 50) and n >= 15)
    return items


def main(sleeve):
    D = W.load(sleeve)
    F = json.load(open(os.path.join(W.D_OUT, f"{sleeve}_frozen.json")))
    G = pd.read_pickle(os.path.join(W.D_OUT, f"{sleeve}_design.pkl")); L = G["L"]
    log = []; P = lambda *a: (log.append(" ".join(str(x) for x in a)), print(*a, flush=True))
    ho = (D.block == "HOLDOUT").values; dm = ~ho; allm = np.ones(len(D), bool)
    yb = D.y_rg.values
    P(f"== {sleeve} HOLDOUT (frozen {F['frozen_utc']}) · N {ho.sum()} · labels {D[ho].label.value_counts().to_dict()}")
    # ---------- 1. HOLDOUT read of every frozen candidate
    HR = []
    for f in F["forward"]:
        bad, valid, nm = WD.side_of(L, f)
        c = WD.checks(D, bad, valid, ho, nm)
        x = pd.to_numeric(D[f["var"]], errors="coerce").astype(float).values
        # AUC oriented so that > 0.5 = the variable ranks runners above give-backs in the DESIGN direction
        a_d = W.auc(x[dm], yb[dm]); a_h = W.auc(x[ho], yb[ho])
        sgn = 1 if a_d >= 0.5 else -1
        c.update(src=f["src"], auc_design=a_d if sgn > 0 else 1 - a_d, auc_holdout=a_h if sgn > 0 else 1 - a_h)
        # the side split itself as a binary predictor (kept = 1)
        r = ho & valid & np.isfinite(yb)
        c["auc_split_ho"] = W.auc((~bad[r]).astype(float), yb[r])
        cd = WD.checks(D, bad, valid, dm, nm); c["delta_design"] = cd["delta"]; c["rs_bad_design"] = cd["run_share_bad"]; c["rs_kept_design"] = cd["run_share_kept"]
        c["same_dir"] = bool(np.sign(c["delta"]) == np.sign(cd["delta"]) and (c["run_share_kept"] - c["run_share_bad"]) * (cd["run_share_kept"] - cd["run_share_bad"]) > 0)
        HR.append(c)
    HR = pd.DataFrame(HR)
    HR.to_csv(os.path.join(W.D_OUT, f"{sleeve}_holdout_checks.csv"), index=False)
    P(HR[["src", "tag", "N_bad", "N_kept", "rs_bad_design", "rs_kept_design", "run_share_bad", "run_share_kept", "auc_design", "auc_holdout",
          "auc_split_ho", "delta_design", "delta", "ci_lo", "ci_hi", "halves", "reg_bd1", "pair_share", "day_share", "delta_b5", "delta20",
          "same_dir"]].round(3).to_string(index=False))
    # ---------- 2. theory logistic on HOLDOUT
    M = F["model"]
    Xm = np.column_stack([pd.to_numeric(D[v], errors="coerce").astype(float).values for v in M["vars"]])
    Xm = np.where(np.isfinite(Xm), Xm, np.array(M["med"])); Xm = np.clip(Xm, M["lo"], M["hi"])
    sc = W.logistic_score((Xm - np.array(M["mu"])) / np.array(M["sd"]), np.array(M["w"]))
    a_ho = W.auc(sc[ho], yb[ho]); a_ho_net = W.auc(sc[ho], (D.net.values[ho] > 0).astype(float))
    q1, q2 = M["q_edges_all"]
    tl = []
    for nm_, m_ in (("bottom third (predicted give-back)", sc <= q1), ("middle", (sc > q1) & (sc <= q2)), ("top third (predicted runner)", sc > q2)):
        for blk, bm in (("DESIGN", dm), ("HOLDOUT", ho)):
            mm = m_ & bm
            tl.append(dict(block=blk, tercile=nm_, N=int(mm.sum()), run_share=np.nanmean(yb[mm]), avg=D.net.values[mm].mean()))
    TL = pd.DataFrame(tl)
    P(f"\ntheory logistic: DESIGN in-sample {M['auc_in']:.3f} · DESIGN CV {M['auc_cv']:.3f} · HOLDOUT AUC (R vs GB) {a_ho:.3f} · HOLDOUT AUC (net>0) {a_ho_net:.3f}")
    P(TL.round(3).to_string(index=False))
    # ---------- 3. action pricing
    names, fam, NET, N20, XM, K = cell_matrix(sleeve, D)
    ref = NET[:, 0]; xr = XM[:, 0]; stop_ref = K[:, 0] == 1
    pick = HR.copy()
    # price: every candidate that kept its direction on HOLDOUT, plus the best DESIGN theory-led read (descriptive)
    sel = list(pick.index[pick.same_dir]) + [pick.index[pick.src == "theory-led"][0]]
    sel = list(dict.fromkeys(sel))
    jb = joint(sleeve, D, ref, xr, stop_ref, allm)
    P(f"\njoint book (EXH + FB + L4 + DIST, 1m REF nets) baseline: DCR {jb['j_dcr']:.3f} %/day (HOLDOUT window {jb['j_dcr_ho']:.3f}) · max DD {jb['j_maxdd']:.1f} · worst day {jb['j_worst']:.1f}")
    AR = []; EXR = []
    for ix in sel:
        f = F["forward"][ix]; bad, valid, nm = WD.side_of(L, f); bad = bad & valid
        for blk, bm in (("DESIGN", dm), ("HOLDOUT", ho), ("ALL", allm)):
            m = bm
            b0 = book(ref[m], xr[m], D.day.values[m], blk)
            AR.append(dict(cand=nm, action="before", block=blk, **b0))
            # (a) filter
            k = m & ~bad
            b1 = book(ref[k], xr[k], D.day.values[k], blk)
            hc = {}
            for h in (0.3, 0.5):
                up = ref[k].mean() - ref[m].mean()
                hc[f"dcr_hc{int(h * 100)}"] = book(ref[k] - h * up, xr[k], None, blk)["dcr"]
            eb = expectancy_bar(ref[m & bad], D.day.values[m & bad], D.pair.values[m & bad], b0["BE"])
            loo = RM.loo_text(list(-ref[m & bad] * W.SIZE), list(D.pair.values[m & bad]), unit="eq-pts")
            AR.append(dict(cand=nm, action="(a) filter", block=blk, **b1, **hc, **eb, loo=loo))
            # (b) size tilt: bad side at 0.5 × (cap for losers); + exposure-neutral variant
            for lab, sb, sk in (("(b) tilt bad 0.5×", 0.5, 1.0), ("(b) tilt bad 0.5× / rest ↑ (same exposure)", 0.5, None)):
                if sk is None:
                    nb, nk = (m & bad).sum(), (m & ~bad).sum(); sk = (nb + nk - 0.5 * nb) / nk
                sz = np.where(bad[m], sb, sk) * W.SIZE
                d0, d1 = SPANS[blk]; os_, _ = W.onesize(ref[m], xr[m], d0, d1, sz)
                AR.append(dict(cand=nm, action=lab, block=blk, N=int(m.sum()), avg=ref[m].mean(), dcr=os_["dcr"], maxdd=os_["maxdd"],
                               worst=os_["worst_day"], kelly_t=W.kel(ref[m & ~bad])[0], kelly_bad=W.kel(ref[m & bad])[0],
                               kelly_d=W.kel(pd.Series(ref[m & ~bad]).groupby(xr[m & ~bad] // DAY).sum().values)[0],
                               kelly_d_bad=W.kel(pd.Series(ref[m & bad]).groupby(xr[m & bad] // DAY).sum().values)[0], size_rest=sk))
        # (c) conditional exit: pick the cell on DESIGN bad-side fills (max avg Δ vs REF), then read HOLDOUT
        cand_cells = [j for j in range(1, len(names)) if fam[j] not in ("REF", "X")]
        mb = dm & bad
        dd = np.array([np.nanmean(NET[mb, j] - ref[mb]) for j in cand_cells])
        jbest = cand_cells[int(np.nanargmax(dd))]
        for blk, bm in (("DESIGN", dm), ("HOLDOUT", ho), ("ALL", allm)):
            m = bm
            v = np.where(bad, NET[:, jbest], ref); xx = np.where(bad, XM[:, jbest], xr)
            v = np.where(np.isfinite(v), v, ref)
            b2 = book(v[m], xx[m], None, blk)
            mb_ = m & bad; dlt = v - ref
            saved = mb_ & (K[:, 0] == 1) & (dlt > 0.05); deeper = mb_ & (K[:, 0] == 1) & (dlt < -0.05)
            cut = mb_ & (K[:, 0] == 0) & (dlt < -0.05)
            EXR.append(dict(cand=nm, cell=names[jbest], block=blk, d_design_pick=dd.max(), N=int(m.sum()), avg=v[m].mean(), avg_ref=ref[m].mean(),
                            delta=dlt[m].mean(), delta_bad=dlt[mb_].mean(), n_bad=int(mb_.sum()), saved=int(saved.sum()),
                            saved_pts=dlt[saved].sum(), deeper=int(deeper.sum()), deeper_pts=dlt[deeper].sum(), tp_cut=int(cut.sum()),
                            tp_cut_pts=dlt[cut].sum(), other_pts=dlt[mb_ & ~saved & ~deeper & ~cut].sum(),
                            dcr=b2["dcr"], maxdd=b2["maxdd"], worst=b2["worst"], WR=b2["WR"], kelly_t=b2["kelly_t"], kelly_d=b2["kelly_d"],
                            loo=RM.loo_text(list(dlt[mb_] * W.SIZE), list(D.pair.values[mb_]), unit="eq-pts")))
        # joint adds (ALL span) for (a) and (c)
        ja = joint(sleeve, D, ref, xr, stop_ref, ~bad)
        v = np.where(bad, NET[:, jbest], ref); xx = np.where(bad, XM[:, jbest], xr); v = np.where(np.isfinite(v), v, ref)
        jc = joint(sleeve, D, v, xx, np.where(bad, np.isin(K[:, jbest], (1, 3)), stop_ref), allm)
        AR.append(dict(cand=nm, action="(a) filter · joint", block="ALL", **ja))
        AR.append(dict(cand=nm, action=f"(c) exit {names[jbest]} on bad side · joint", block="ALL", **jc))
    AR = pd.DataFrame(AR); EXR = pd.DataFrame(EXR)
    AR.to_csv(os.path.join(W.D_OUT, f"{sleeve}_actions.csv"), index=False); EXR.to_csv(os.path.join(W.D_OUT, f"{sleeve}_condexit.csv"), index=False)
    pd.set_option("display.width", 250)
    P("\nACTIONS:"); P(AR.drop(columns=[c for c in AR.columns if c in ("loo",)]).round(3).to_string(index=False))
    P("\nLOO:"); P(AR[AR.loo.notna()][["cand", "block", "loo"]].to_string(index=False))
    P("\nCONDITIONAL EXIT:"); P(EXR.drop(columns=["loo"]).round(3).to_string(index=False)); P(EXR[["cand", "block", "loo"]].to_string(index=False))
    pd.to_pickle(dict(HR=HR, TL=TL, AR=AR, EXR=EXR, jb=jb, a_ho=a_ho, a_ho_net=a_ho_net, sc=sc), os.path.join(W.D_OUT, f"{sleeve}_holdout.pkl"))
    open(os.path.join(W.D_OUT, f"{sleeve}_holdout.log"), "w").write("\n".join(log))


if __name__ == "__main__":
    main(sys.argv[1])
