#!/usr/bin/env python3
"""DIST_WL stage 2 (PREREG reports/DIST_WL_2026-11-18_PREREG.md): theory tables (DESIGN) · survivor steps on every test that beat a
family-wise null (screen-wide for net / worked / reach, theory-family for the theory variables) + the top-40 |t_net| cells and the two
seen hints (descriptive, flagged NOT-NULL) · HOLDOUT read once for DESIGN passers · before → after (one position per pair re-run) ·
Kelly · charts · trade CSV.  Usage: venv/bin/python -I scripts/dist_wl_survivors.py"""
import json, os, sys, gzip
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import dist_wl_analyze as A  # noqa: E402
import unified_short_lib as L  # noqa: E402
OUT = A.OUT; RP = os.path.join(L.R, "reports")
SIZE = A.SIZE


def cut_mask(F, r):
    """row mask of the cell (scored rows only) on a frame F."""
    x1 = F[r["var"]].values.astype(float)
    m = A.in_bin(x1, r["lo"], r["hi"]) & np.isfinite(x1)
    if r["kind"] == "2D":
        x2 = F[r["var2"]].values.astype(float)
        m &= A.in_bin(x2, r["lo2"], r["hi2"]) & np.isfinite(x2)
        sc = np.isfinite(x1) & np.isfinite(x2)
    else:
        sc = np.isfinite(x1)
    return m, sc


def keep_mask(F, r):
    """kept rows: block the cell (net worse) or keep only the cell (net better); unscored rows are KEPT (a filter cannot read them)."""
    m, sc = cut_mask(F, r)
    return ~m if r["mode"] == "block" else (m | ~sc)


def steps(B_, r, des):
    v = B_.net_LIVE.values; v20 = B_.net20_LIVE.values; day = B_.day.values
    k = keep_mask(B_, r)
    o = dict()
    D = des
    vb, kb = v[D], k[D]
    o["d_N_keep"] = int(kb.sum()); o["d_N_block"] = int((~kb).sum())
    o["d_base"] = vb.mean(); o["d_keep"] = vb[kb].mean() if kb.any() else np.nan; o["d_block"] = vb[~kb].mean() if (~kb).any() else np.nan
    o["d_delta"] = o["d_keep"] - o["d_base"]
    hv = B_.half.values
    hp = []; kp = []
    for h in A.DES_HALVES:
        mm = D & (hv == h)
        if (mm & k).sum() >= 20:
            hp.append(v[mm & k].mean() - v[mm].mean()); kp.append(v[mm & k].mean())
    o["halves_delta_pos"] = f"{sum(np.array(hp) > 0)}/{len(hp)}"; o["halves_keep_pos"] = f"{sum(np.array(kp) > 0)}/{len(kp)}"
    o["s_halves"] = len(hp) >= 3 and sum(np.array(hp) > 0) >= 3 and sum(np.array(kp) > 0) >= 3
    rg = B_.btc_d1_regime.values
    rr = []
    for g in (-1, 1):
        mm = D & (rg == g)
        rr.append(v[mm & k].mean() - v[mm].mean() if (mm & k).sum() >= 20 else np.nan)
    o["btc_bear_delta"], o["btc_bull_delta"] = rr
    # DESIGN (2023-07 → 2025-06) is 1,656 of 1,660 trades BTC-daily-bull: the bear side is untestable there (NaN, skipped) and the
    # regime check is completed on HOLDOUT (both regimes present) — deviation from the PREREG, disclosed in the report.
    rv = np.array(rr, float); o["s_regime"] = bool(np.all(rv[np.isfinite(rv)] > 0))
    lo, hi = A.block_boot(vb, day[D], kb)
    o["ci_lo"], o["ci_hi"] = lo, hi; o["s_ci"] = bool(lo > 0)
    # concentration: effect = −Σ blocked net (pts); share of the largest pair / day
    bl = D & ~k
    E = -v[bl].sum()
    o["effect_pts"] = E
    if E > 0:
        ps = (-pd.Series(v[bl]).groupby(B_.pair.values[bl]).sum()).max() / E
        ds = (-pd.Series(v[bl]).groupby(day[bl]).sum()).max() / E
    else:
        ex = v[D & k] - o["d_base"]
        tot = ex.sum()
        ps = pd.Series(ex).groupby(B_.pair.values[D & k]).sum().max() / tot if tot > 0 else np.inf
        ds = pd.Series(ex).groupby(day[D & k]).sum().max() / tot if tot > 0 else np.inf
    o["top_pair_share"], o["top_day_share"] = ps, ds
    o["s_conc"] = bool(ps < 0.5 and ds < 0.5)
    o["b5_keep"] = A.b5(vb[kb]); o["s_b5"] = bool(o["b5_keep"] > 0)
    o["s20_keep"] = v20[D & k].mean(); o["s_slip"] = bool(o["s20_keep"] > 0)
    o["pass_design"] = all(o[s] for s in ("s_halves", "s_regime", "s_ci", "s_conc", "s_b5", "s_slip"))
    return o


def holdout(B_, r, hol):
    v = B_.net_LIVE.values; v20 = B_.net20_LIVE.values
    k = keep_mask(B_, r)
    o = dict(h_base=v[hol].mean(), h_keep=v[hol & k].mean(), h_block=v[hol & ~k].mean(), h_N_keep=int((hol & k).sum()),
             h_s20_keep=v20[hol & k].mean())
    hv = B_.half.values; kp = []
    for h, _, _ in L.HALVES:
        mm = (hv == h) & k
        if mm.sum() >= 20:
            kp.append(v[mm].mean())
    o["all_halves_keep_pos"] = f"{sum(np.array(kp) > 0)}/{len(kp)}"
    rg = B_.btc_d1_regime.values
    for g, nm in ((-1, "h_bear_delta"), (1, "h_bull_delta")):
        mm = hol & (rg == g); o[nm] = v[mm & k].mean() - v[mm].mean() if (mm & k).sum() >= 20 else np.nan
    o["h_pass"] = bool(o["h_keep"] > o["h_base"] and o["h_keep"] > 0 and sum(np.array(kp) > 0) >= np.ceil(2 / 3 * len(kp)) and o["h_s20_keep"] > 0
                     and np.nanmin([o["h_bear_delta"], o["h_bull_delta"]]) > 0)
    return o


def card(S, mask, label, span):
    """before / after card on the full row frame with one-position-per-pair re-run (LIVE exit)."""
    m = A.one_pos(S, mask)
    v = S.net_LIVE.values[m]; day = S.day.values[m]; xd = S.LIVE_x.values[m] // A.DAY
    n = len(v); days = span[1] - span[0] + 1
    d, dd, wd = A.dcr(v, xd, SIZE, span)
    fT, pT, nT = A.kelly(v)
    fD, pD, nD = A.kelly(v, day)
    al = -v[v <= 0].mean()
    return dict(label=label, N=n, tpd=n / days, WR=(v > 0).mean() * 100, avg=v.mean(), s20=S.net20_LIVE.values[m].mean(), b5=A.b5(v),
                dcr=d, maxdd=dd, worst_day=wd, wr_floor_trade=pT * 100, kelly_trade=fT, kelly_notional_trade=nT,
                wr_floor_day=pD * 100, kelly_day=fD, kelly_notional_day=nD, avg_loss=al, avg_win=v[v > 0].mean(),
                be_wr=al / (al + v[v > 0].mean()) * 100)


if __name__ == "__main__":
    B_ = pd.read_pickle(os.path.join(OUT, "base.pkl"))
    T = pd.read_pickle(os.path.join(OUT, "screen.pkl"))
    R = json.load(open(os.path.join(OUT, "stage1.json")))
    S = pd.read_pickle(os.path.join(OUT, "rows.pkl"))
    des = B_.entry_ms.values < A.HOLD_START; hol = ~des
    th_vars = sum(A.THEORY.values(), [])
    # ── theory tables (DESIGN only)
    v = B_.net_LIVE.values; lab = B_.label.values
    rows = []
    for fam, vs in A.THEORY.items():
        for c in vs:
            sub = T[(T.kind == "1D") & (T["var"] == c)]
            for _, r in sub.iterrows():
                x = B_[c].values.astype(float)
                m = A.in_bin(x, r.lo, r.hi) & np.isfinite(x) & des
                rows.append(dict(fam=fam, var=c, pred=A.PRED.get(c, ""), gran=r.gran, bin=r.bin, lo=r.lo, hi=r.hi, N=int(m.sum()),
                                 WR=(v[m] > 0).mean() * 100, avg=v[m].mean(), worked=(lab[m] == 1).mean() * 100,
                                 reach_med=np.median(B_.reach.values[m]), t_net=r.t_net, t_worked=r.t_worked, t_reach=r.t_reach))
    TH = pd.DataFrame(rows); TH.to_csv(os.path.join(OUT, "theory_bins.csv"), index=False)
    # prediction verdict per variable: top vs bottom tercile (or category ends) on DESIGN net
    ver = []
    for c in th_vars:
        t = TH[(TH["var"] == c) & (TH.gran.isin(["terc", "cat"]))]
        if t.empty:
            continue
        lo_, hi_ = t.iloc[0], t.iloc[-1]
        dirn = "hi" if hi_.avg > lo_.avg else "lo"
        maxt = np.nanmax(np.abs(TH[TH["var"] == c][["t_net"]].values))
        ver.append(dict(var=c, fam=next(k for k, vs in A.THEORY.items() if c in vs), pred=A.PRED[c], low_bin_avg=lo_.avg, high_bin_avg=hi_.avg,
                        observed=dirn, held=dirn == A.PRED[c], max_abs_t_net=maxt,
                        beats_theory_null_net=bool(maxt > R["null_net"]["q95_theory"]),
                        max_abs_t_reach=np.nanmax(np.abs(TH[TH["var"] == c].t_reach.values)),
                        max_abs_t_worked=np.nanmax(np.abs(TH[TH["var"] == c].t_worked.values))))
    VER = pd.DataFrame(ver); VER.to_csv(os.path.join(OUT, "theory_verdict.csv"), index=False)
    print(VER.round(3).to_string(), flush=True)
    # ── candidate set
    isth = (T.kind == "1D") & T["var"].isin(th_vars)
    sel = np.zeros(len(T), bool); why = np.full(len(T), "", object)
    for o in ("net", "worked", "reach"):
        a = np.abs(T[f"t_{o}"].fillna(0).values)
        mm = a > R[f"null_{o}"]["q95"]; why[mm & (why == "")] = f"screen-null {o}"; sel |= mm
        mt = isth.values & (a > R[f"null_{o}"]["q95_theory"]); why[mt & (why == "")] = f"theory-null {o}"; sel |= mt
    top = T.t_net.abs().sort_values(ascending=False).index[:40]
    exp = np.zeros(len(T), bool); exp[top] = True; why[exp & ~sel] = "top|t_net| (NOT null)"
    hint = ((T.kind == "1D") & (((T["var"] == "ret1h_pct") & (T.gran == "sign")) | ((T["var"] == "btc_d1_regime")))).values
    why[hint & ~sel & ~exp] = "seen hint (NOT null)"
    C = T[sel | exp | hint].copy(); C["why"] = why[sel | exp | hint]
    C["mode"] = np.where(C.m1_net > C.m0_net, "keep_only", "block")
    print("candidates", len(C), C.why.value_counts().to_dict(), flush=True)
    res = []
    for i, r in C.iterrows():
        o = steps(B_, r, des); o.update(r.to_dict()); o["tid"] = i; res.append(o)
    RS = pd.DataFrame(res)
    RS.to_pickle(os.path.join(OUT, "survivor_steps.pkl"))
    sc = ["s_halves", "s_regime", "s_ci", "s_conc", "s_b5", "s_slip"]
    print("step pass counts", {s: int(RS[s].sum()) for s in sc}, "all", int(RS.pass_design.sum()), flush=True)
    print(RS.groupby("why")[sc + ["pass_design"]].sum().to_string(), flush=True)
    # ── HOLDOUT once for DESIGN passers
    RS["null_pass"] = RS.why.str.contains("-null")
    print("DESIGN-step passers that did NOT pass a null (no HOLDOUT read):", int((RS.pass_design & ~RS.null_pass).sum()), flush=True)
    P = RS[RS.pass_design & RS.null_pass].copy()
    if len(P):
        H = pd.DataFrame([holdout(B_, r, hol) for _, r in P.iterrows()], index=P.index)
        P = pd.concat([P, H], axis=1)
    P.to_pickle(os.path.join(OUT, "design_passers.pkl"))
    print("design passers", len(P), flush=True)
    if len(P):
        print(P[["why", "kind", "var", "bin", "var2", "bin2", "mode", "d_N_keep", "d_base", "d_keep", "d_delta", "ci_lo", "h_base", "h_keep",
                 "h_N_keep", "all_halves_keep_pos", "h_s20_keep", "h_pass"]].round(3).to_string(), flush=True)
    json.dump(dict(n_cand=len(C), n_design=int(len(P)), n_hold=int(P.h_pass.sum()) if len(P) else 0), open(os.path.join(OUT, "stage2.json"), "w"))
