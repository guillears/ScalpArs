#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — PART A: exit math on the REAL fills (master kept non-probe + live B23).

Grid (frozen before any cell was read; every cell walked on each fill's own path by study_frenzy_realfills_common.walk):
  FIX    TP {1,1.5,2,2.5,3,4,5,6,8} × SL {1.5,2,2.5,3,4,5} × cap {60,120,240,720}
  TRAIL  arm A {1,1.5,2,3} × give-back G {0.5,1,1.5,2,3} × SL {1.5,2,2.5,3,4,5} × cap {60,120,240,720} (no TP)
  BETR   breakeven at peak ≥ b {0.5,1,1.5} (line +0.1), then trail: A {1.5,2,3} × G {1,1.5,2} × SL {2,3,4} × cap {240,720}
  SPLIT  half at TP1 {1,1.5,2} (maker) + half: TP2 {3,5,8} keep SL | TP2 {3,5,8} stop→BE at TP1 | trail G {1.5,2,3} from TP1 (stop→BE) ;
         SL {2,3,4}, cap 720
  ATR    TP = kt × entry ATR {0.5,1,1.5,2,3,4} × SL = ks × entry ATR {1,1.5,2,3,4} × cap {240,720} (fills without an ATR stamp = today)
  TODAY  the fill's own live exit (LONG +1.5/−3/720 · LITE C stored TP/−3/720 · WILLY +1.25/−4/90 late +0.25 from 30 min), maker TP
Validation: LEAVE-ONE-DAY-OUT (choose the best mean cell on the other UTC days, apply to the held-out day; concatenated = honest OOS),
paired vs TODAY with a day-block bootstrap; fold stability; neighbour smoothness; shuffled-path null for the best-of-grid gain.
Usage: venv/bin/python scripts/study_frenzy_realfills_exitgrid.py
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
RF = RC.RF
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
INF = np.inf


def cells():
    C = []
    for tp, sl, cap in itertools.product((1, 1.5, 2, 2.5, 3, 4, 5, 6, 8), (1.5, 2, 2.5, 3, 4, 5), (60, 120, 240, 720)):
        C.append(("FIX", f"FIX tp{tp}/sl{sl}/{cap}m", dict(kind="w", tp=tp, sl=sl, cap=cap)))
    for A, G, sl, cap in itertools.product((1, 1.5, 2, 3), (0.5, 1, 1.5, 2, 3), (1.5, 2, 2.5, 3, 4, 5), (60, 120, 240, 720)):
        C.append(("TRAIL", f"TRAIL a{A}/g{G}/sl{sl}/{cap}m", dict(kind="w", sl=sl, A=A, G=G, cap=cap)))
    for b, A, G, sl, cap in itertools.product((0.5, 1, 1.5), (1.5, 2, 3), (1, 1.5, 2), (2, 3, 4), (240, 720)):
        if A >= b:
            C.append(("BETR", f"BETR be{b}/a{A}/g{G}/sl{sl}/{cap}m", dict(kind="w", sl=sl, b=b, be=0.1, A=A, G=G, cap=cap)))
    for t1, sl in itertools.product((1, 1.5, 2), (2, 3, 4)):
        for t2 in (3, 5, 8):
            C.append(("SPLIT", f"SPLIT ½@{t1}+½@{t2} sl{sl}", dict(kind="split", t1=t1, sl=sl, rest=dict(tp=t2, sl=sl, cap=720))))
            C.append(("SPLIT", f"SPLIT ½@{t1}+½@{t2} BE sl{sl}", dict(kind="split", t1=t1, sl=sl, rest=dict(tp=t2, sl=sl, b=t1, be=0.1, cap=720))))
        for g in (1.5, 2, 3):
            C.append(("SPLIT", f"SPLIT ½@{t1}+½trail g{g} BE sl{sl}", dict(kind="split", t1=t1, sl=sl,
                                                                            rest=dict(sl=sl, b=t1, be=0.1, A=t1, G=g, cap=720))))
    for kt, ks, cap in itertools.product((0.5, 1, 1.5, 2, 3, 4), (1, 1.5, 2, 3, 4), (240, 720)):
        C.append(("ATR", f"ATR tp{kt}x/sl{ks}x/{cap}m", dict(kind="atr", kt=kt, ks=ks, cap=cap)))
    return C


def run_cell(P, r, c):
    k = c["kind"]
    if k == "w":
        a = {x: c[x] for x in ("tp", "sl", "b", "be", "A", "G", "cap") if x in c}
        return RC.walk(P, **a)
    if k == "split":
        v1 = RC.walk(P, tp=c["t1"], sl=c["sl"], cap=720)
        v2 = RC.walk(P, **c["rest"])
        return 0.5 * v1[0] + 0.5 * v2[0], ("UNRES" if "UNRES" in (v1[1], v2[1]) else v2[1]), max(v1[2], v2[2])
    if k == "atr":
        atr = pd.to_numeric(r.get("entry_atr_pct"), errors="coerce")
        if not (np.isfinite(atr) and atr > 0):
            return RC.today(P, r)
        return RC.walk(P, tp=c["kt"] * atr, sl=c["ks"] * atr, cap=c["cap"])
    raise ValueError(k)


def to_1m(P):
    """1-minute OHLC (gross %) from the path (for the shuffled-path null)."""
    m = (P["t"] - P["te"] - 1) // RC.MIN
    brk = np.flatnonzero(np.diff(m)) + 1; st = np.r_[0, brk]; en = np.r_[brk, len(m)]
    op = np.where(np.isfinite(P["op"][st]), P["op"][st], P["cl"][st])
    return dict(o=op, h=np.maximum.reduceat(P["hi"], st), l=np.minimum.reduceat(P["lo"], st), c=P["cl"][en - 1], m=m[st])


def from_1m(B, te, end_ms):
    t = te + (B["m"] + 1) * RC.MIN
    o = dict(t=t.astype("int64"), hi=B["h"], lo=B["l"], op=B["o"], cl=B["c"], end_ms=end_ms, te=te)
    o["pk_prev"] = np.r_[-np.inf, np.maximum.accumulate(o["hi"])[:-1]]
    return o


def shuffle_1m(B, rng):
    """permute the 1m bars (each bar's moves relative to the previous close) → a path with the same bars in random order."""
    prev = np.r_[0.0, B["c"][:-1]]
    rel = np.c_[B["o"] - prev, B["h"] - prev, B["l"] - prev, B["c"] - prev]
    k = rng.permutation(len(rel)); rel = rel[k]
    c = np.cumsum(rel[:, 3]); p0 = np.r_[0.0, c[:-1]]
    return dict(o=p0 + rel[:, 0], h=p0 + rel[:, 1], l=p0 + rel[:, 2], c=c, m=B["m"])


def lodo(V, ref, days, names):
    """V: fills × cells (net %), ref: fills; days: per-fill day. → oos vector, chosen cell per day."""
    oos = np.full(len(ref), np.nan); pick = {}
    for d in np.unique(days):
        tr = days != d
        mu = np.nanmean(V[tr], axis=0)
        j = int(np.nanargmax(mu)); pick[d] = names[j]
        oos[~tr] = V[~tr, j]
    return oos, pick


def summ(x, days):
    x = np.asarray(x, float); ok = np.isfinite(x)
    by = [x[ok & (days == d)] for d in np.unique(days[ok])]
    m, lo, hi, p = RC.day_boot(by)
    dsum = pd.Series(x[ok]).groupby(days[ok]).sum()
    return dict(N=int(ok.sum()), days=len(by), avg=m, WR=float((x[ok] > 0).mean()), sum=float(x[ok].sum()), worst=float(x[ok].min()),
                worst_day=float(dsum.min()), ci_lo=lo, ci_hi=hi, P_gt0=p)


if __name__ == "__main__":
    D = pd.read_pickle(f"{RF}/real.pkl"); PP = pd.read_pickle(f"{RF}/real_paths.pkl")
    C = cells(); names = [c[1] for c in C]; fam = np.array([c[0] for c in C])
    nF, nC = len(D), len(C)
    V = np.full((nF, nC), np.nan); W = np.empty((nF, nC), dtype=object); ref = np.zeros(nF); refw = []
    for i, r in D.iterrows():
        P = PP[r.key]
        v = RC.today(P, r); ref[i] = v[0]; refw.append(v[1])
        for j, c in enumerate(C):
            x = run_cell(P, r, c[2]); V[i, j] = x[0]; W[i, j] = x[1]
    unres = (W == "UNRES").sum(0)
    np.save(f"{RF}/grid_V.npy", V); pd.to_pickle(dict(names=names, fam=fam, ref=ref, W=W), f"{RF}/grid_meta.pkl")
    days = D.day.values; sv = D.sleeve.values
    rows, lrows, srows, picks = [], [], [], []
    rng = np.random.default_rng(11)
    for grp in ("FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY", "POOLED"):
        m = np.ones(nF, bool) if grp == "POOLED" else (sv == grp)
        Vg, rg, dg = V[m], ref[m], days[m]
        mu = np.nanmean(Vg, 0)
        base = summ(rg, dg)
        rows.append(dict(group=grp, cell="TODAY (live exit)", family="TODAY", **base, d_vs_today=0.0))
        for j in np.argsort(-mu)[:25]:
            s = summ(Vg[:, j], dg)
            rows.append(dict(group=grp, cell=names[j], family=fam[j], **s, d_vs_today=s["avg"] - base["avg"], unres=int(unres[j])))
        for f in ("FIX", "TRAIL", "BETR", "SPLIT", "ATR"):
            jj = np.flatnonzero(fam == f); j = jj[int(np.nanargmax(mu[jj]))]
            s = summ(Vg[:, j], dg)
            rows.append(dict(group=grp, cell="best " + f + ": " + names[j], family=f, **s, d_vs_today=s["avg"] - base["avg"]))
        # LODO — whole grid and per family
        for f in ("ALL", "FIX", "TRAIL", "BETR", "SPLIT", "ATR"):
            jj = np.arange(nC) if f == "ALL" else np.flatnonzero(fam == f)
            oos, pk = lodo(Vg[:, jj], rg, dg, [names[j] for j in jj])
            d = oos - rg
            so = summ(oos, dg); sd = summ(d, dg)
            vc = pd.Series(list(pk.values())).value_counts()
            lrows.append(dict(group=grp, family=f, N=so["N"], days=so["days"], oos_avg=so["avg"], oos_WR=so["WR"], oos_sum=so["sum"],
                              oos_worst_day=so["worst_day"], today_avg=base["avg"], today_sum=base["sum"], today_worst_day=base["worst_day"],
                              delta_avg=sd["avg"], delta_ci_lo=sd["ci_lo"], delta_ci_hi=sd["ci_hi"], P_delta_gt0=sd["P_gt0"],
                              insample_best=names[jj[int(np.nanargmax(mu[jj]))]], insample_best_avg=float(np.nanmax(mu[jj])),
                              n_distinct_picks=len(vc), modal_pick=vc.index[0], modal_share=float(vc.iloc[0] / len(pk))))
            if f == "ALL":
                for d_, p_ in pk.items():
                    picks.append(dict(group=grp, held_out_day=d_, chosen=p_, n_fills=int((dg == d_).sum()),
                                      oos_day_sum=float(np.nansum(oos[dg == d_])), today_day_sum=float(rg[dg == d_].sum())))
        # smoothness: the best FIX cell and its neighbours (same cap)
        jj = np.flatnonzero(fam == "FIX"); jb = jj[int(np.nanargmax(mu[jj]))]
        c = C[jb][2]; TPs = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8); SLs = (1.5, 2, 2.5, 3, 4, 5)
        ti, si = TPs.index(c["tp"]), SLs.index(c["sl"])
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                if 0 <= ti + a < len(TPs) and 0 <= si + b < len(SLs):
                    nm = f"FIX tp{TPs[ti + a]}/sl{SLs[si + b]}/{c['cap']}m"; j = names.index(nm)
                    srows.append(dict(group=grp, best=names[jb], neighbour=nm, avg=float(mu[j]), d_vs_today=float(mu[j] - base["avg"])))
    pd.DataFrame(rows).to_csv(f"{OUT}_A_grid_top.csv", index=False)
    pd.DataFrame(lrows).to_csv(f"{OUT}_A_lodo.csv", index=False)
    pd.DataFrame(picks).to_csv(f"{OUT}_A_lodo_picks.csv", index=False)
    pd.DataFrame(srows).to_csv(f"{OUT}_A_smooth.csv", index=False)
    # full surface (FIX 720 + every cell mean per group) for the record
    surf = []
    for grp in ("FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY", "POOLED"):
        m = np.ones(nF, bool) if grp == "POOLED" else (sv == grp)
        mu = np.nanmean(V[m], 0); wr = np.nanmean(V[m] > 0, 0); sm = np.nansum(V[m], 0); wo = np.nanmin(V[m], 0)
        for j in range(nC):
            surf.append(dict(group=grp, family=fam[j], cell=names[j], avg=mu[j], WR=wr[j], sum=sm[j], worst=wo[j],
                             d_vs_today=mu[j] - ref[m].mean(), unres=int(unres[j])))
    pd.DataFrame(surf).to_csv(f"{OUT}_A_surface.csv", index=False)
    # per-fill today + key cells
    pf = D[["key", "sleeve", "pair", "day", "era", "y_book"]].copy(); pf["today_walk"] = ref; pf["today_why"] = refw
    pf.to_csv(f"{OUT}_A_fills.csv", index=False)

    # ── shuffled-path null for the in-sample best-of-grid gain (1m resolution, FIX + TRAIL cells, 720 cap) ──
    jn = [j for j in range(nC) if fam[j] in ("FIX", "TRAIL") and C[j][2]["cap"] == 720]
    B1 = {r.key: to_1m(PP[r.key]) for r in D.itertuples()}
    RD = {r["key"]: r for _, r in D.iterrows()}

    def best_gain(paths, m):
        rr = [(k, RD[k]) for k in D.key[m]]
        Vn = np.array([[run_cell(paths[k], r, C[j][2])[0] for j in jn] for k, r in rr])
        rn = np.array([RC.today(paths[k], r)[0] for k, r in rr])
        return float(np.nanmax(np.nanmean(Vn, 0)) - rn.mean())
    nrows = []
    NS = int(os.environ.get("NSHUF", "100"))
    for grp in ("FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY", "POOLED"):
        m = np.ones(nF, bool) if grp == "POOLED" else (sv == grp)
        obs_paths = {r.key: from_1m(B1[r.key], r.te, PP[r.key]["end_ms"]) for r in D[m].itertuples()}
        obs = best_gain(obs_paths, m)
        null = []
        for s in range(NS):
            sp = {r.key: from_1m(shuffle_1m(B1[r.key], rng), r.te, PP[r.key]["end_ms"]) for r in D[m].itertuples()}
            null.append(best_gain(sp, m))
        null = np.array(null)
        nrows.append(dict(group=grp, cells=len(jn), obs_best_gain_1m=obs, null_mean=null.mean(), null_p95=np.percentile(null, 95),
                          p_value=float((null >= obs).mean()), n_shuffles=NS))
        print(nrows[-1], flush=True)
    pd.DataFrame(nrows).to_csv(f"{OUT}_A_null.csv", index=False)
    print("done")
