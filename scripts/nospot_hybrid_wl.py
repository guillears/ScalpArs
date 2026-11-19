#!/usr/bin/env python3
"""NOSPOT_HYBRID — WINNERS vs LOSERS on the HYB · XE trades (PREREG). Labels: booked net under XE, the same entries under TP4SL10, and
RUNNER (worked ∧ net > 0) vs GAVE_BACK (worked ∧ net ≤ 0). Coverage table; 1D (sign / terciles / quintiles, cuts from the screened set)
and every 2D pair at terciles with a family-wise label-shuffle null (500 / 200). DECISION screen = DESIGN only (prereg); a full-span screen
is DESCRIPTIVE. Theory candidates T1–T6 (frozen) + DESIGN survivors → kept / blocked per block × exit, halves, BTC R1 in windows, CI,
concentration, 0.20 slip, expectancy bar on the blocked side, the entry × exit check, and the three pricings (filter / tilt / exit).
Writes reports/NOSPOT_HYBRID_2026-11-18_data/wl_*.csv, joint_extra.pkl. Usage: venv/bin/python -I scripts/nospot_hybrid_wl.py"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_exit_eval as E   # noqa: E402
import nospot_sleeve_lib as L  # noqa: E402
OD = os.path.join(L.R, "reports", "NOSPOT_HYBRID_2026-11-18_data")
DAY = L.DAY
JX = E.NAMES.index("REF_XE"); J4 = E.NAMES.index("A_TP4_SL10")
rng = np.random.default_rng(11)
VARS = ["oi_chg15m", "oi_chg1h", "oi_chg4h", "oi_usd", "oi_vol", "oi_vs_max", "f_last", "f_prev", "f_last8", "f_neg", "f_hourly", "topP",
        "topP_chg1h", "topP_chg4h", "topA", "topA_chg4h", "glA", "glA_chg1h", "glA_chg4h", "taker1h", "taker_prev1h", "prem1h", "prem_pk",
        "rsi15", "rsi1h", "di15", "adx15", "adx1h", "dd_hi", "lh2", "vwap_gap", "peak_age_h", "hrs_since_spike", "vd", "volmult", "atrp",
        "vol24", "q5", "mcap", "turnover", "ret15m", "ret1h", "ret4h", "ret24h", "maxup1h_24h", "rise_on_to_hi", "hour", "R1", "R4",
        "btc1h", "btc4h", "btc24h", "pos4", "pos1", "acc4", "acc1", "pos_now", "acc_now"]
MARKET = {"R1", "R4", "btc1h", "btc4h", "btc24h", "hour"}
SIGN0 = {"oi_chg15m", "oi_chg1h", "oi_chg4h", "f_last", "f_prev", "f_last8", "topP_chg1h", "topP_chg4h", "topA_chg4h", "glA_chg1h",
         "glA_chg4h", "prem1h", "di15", "vwap_gap", "ret15m", "ret1h", "ret4h", "ret24h", "btc1h", "btc4h", "btc24h", "R1", "R4", "pos4",
         "pos1", "acc4", "acc1"}
SIGN1 = {"topP", "topA", "glA", "taker1h", "taker_prev1h", "pos_now", "acc_now"}
BIN = {"f_neg", "f_hourly"}


def buckets(x, xr, var):
    out = {}; ok = np.isfinite(x)
    if var in BIN or var == "R4":
        b = np.full(len(x), -1); b[ok] = (x[ok] > 0).astype(int) if var in BIN else (x[ok] + 1).astype(int); out["sign"] = b
        return out
    thr = 0.0 if var in SIGN0 else 1.0 if var in SIGN1 else np.nanmedian(xr)
    b = np.full(len(x), -1); b[ok] = (x[ok] > thr).astype(int); out["sign"] = b
    if var == "R1":
        return out
    for nm, q in (("terc", 3), ("quint", 5)):
        if np.isfinite(xr).sum() < q * 5:
            continue
        cuts = np.unique(np.nanquantile(xr, np.linspace(0, 1, q + 1)[1:-1]))
        b = np.full(len(x), -1); b[ok] = np.searchsorted(cuts, x[ok], side="right"); out[nm] = b
    return out


def spread(y, b, nb, minn):
    s = np.bincount(b[b >= 0], weights=y[b >= 0], minlength=nb); n = np.bincount(b[b >= 0], minlength=nb)
    m = np.where(n >= minn, s / np.maximum(n, 1), np.nan)
    return (np.nanmax(m) - np.nanmin(m)) if np.isfinite(m).sum() >= 2 else np.nan


def screen(W, sub, tag, minn, min2, nsh=500, nsh2=200):
    """W: trade frame; sub: bool mask of the screened rows (cut points from them)."""
    Ws = W[sub].reset_index(drop=True)
    labs = {"net_XE": Ws.y_xe.values, "net_TP4": Ws.y_4.values}
    wk = Ws.wk.values == 1
    rows, B, cov = [], {}, []
    for var in VARS:
        x = Ws[var].values.astype(float)
        cov.append(dict(screen=tag, var=var, coverage=np.isfinite(x).mean() * 100, N=len(x)))
        for g, b in buckets(x, x, var).items():
            nb = int(b.max()) + 1; B[(var, g)] = (b, nb)
            for k in range(nb):
                m = b == k
                if not m.any():
                    continue
                r = dict(screen=tag, var=var, gran=g, bucket=k, lo=np.nanmin(x[m]), hi=np.nanmax(x[m]), N=int(m.sum()),
                         WR=(Ws.y_xe.values[m] > 0).mean() * 100, avg_XE=Ws.y_xe.values[m].mean(), avg_TP4=Ws.y_4.values[m].mean(),
                         runner_rate=(Ws.y_xe.values[m & wk] > 0).mean() * 100 if (m & wk).any() else np.nan, N_worked=int((m & wk).sum()),
                         worked=wk[m].mean() * 100)
                if var in MARKET:
                    r["windows"] = len(np.unique(Ws.day.values[m]))
                rows.append(r)
    T = pd.DataFrame(rows)
    keys = list(B)
    run_y = (Ws.y_xe.values > 0).astype(float)
    sp = {}
    for ln, y in list(labs.items()) + [("runner", run_y)]:
        msk = wk if ln == "runner" else np.ones(len(y), bool)
        yy = y[msk]
        obs = np.array([spread(yy, B[k][0][msk], B[k][1], minn) for k in keys])
        mx = np.array([np.nanmax([spread(yy[p], B[k][0][msk], B[k][1], minn) for k in keys])
                       for p in (rng.permutation(len(yy)) for _ in range(nsh))])
        sp[ln] = (obs, np.array([(mx >= o).mean() if np.isfinite(o) else np.nan for o in obs]))
    S = pd.DataFrame([dict(screen=tag, var=k[0], gran=k[1], spread_XE=sp["net_XE"][0][q], p_fw_XE=sp["net_XE"][1][q],
                           spread_TP4=sp["net_TP4"][0][q], p_fw_TP4=sp["net_TP4"][1][q], spread_runner=sp["runner"][0][q] * 100,
                           p_fw_runner=sp["runner"][1][q]) for q, k in enumerate(keys)])
    # 2D terciles
    tv = [v for v in VARS if (v, "terc") in B]
    cells = []
    for i, a in enumerate(tv):
        for b_ in tv[i + 1:]:
            ba = B[(a, "terc")][0]; bb = B[(b_, "terc")][0]
            cells.append((a, b_, np.where((ba >= 0) & (bb >= 0), ba * 3 + bb, -1)))
    C = np.vstack([c for _, _, c in cells])                 # pairs × trades

    def cellmeans(y):
        out = np.full((len(cells), 9), np.nan)
        for q in range(len(cells)):
            c = C[q]; ok = c >= 0
            s = np.bincount(c[ok], weights=y[ok], minlength=9); n = np.bincount(c[ok], minlength=9)
            out[q] = np.where(n >= min2, s / np.maximum(n, 1), np.nan)
        return out
    y = Ws.y_xe.values
    M0 = cellmeans(y)
    mx2 = np.zeros(nsh2); mn2 = np.zeros(nsh2)
    for s in range(nsh2):
        Mp = cellmeans(y[rng.permutation(len(y))]); mx2[s] = np.nanmax(Mp); mn2[s] = np.nanmin(Mp)
    M4 = cellmeans(Ws.y_4.values)
    r2 = []
    for q, (a, b_, c) in enumerate(cells):
        for k in range(9):
            if np.isfinite(M0[q, k]):
                r2.append(dict(screen=tag, a=a, b=b_, cell=k, a_terc=k // 3, b_terc=k % 3, N=int((c == k).sum()), avg_XE=M0[q, k],
                               avg_TP4=M4[q, k], p_fw_hi=(mx2 >= M0[q, k]).mean(), p_fw_lo=(mn2 <= M0[q, k]).mean()))
    return T, S, pd.DataFrame(r2), pd.DataFrame(cov)


def dcr(v, xd, blk, w=None):
    Dn, d0, d1 = L.cal_days(blk)
    w = np.ones(len(v)) if w is None else w
    r = pd.Series(v * w * L.SIZE / 100).groupby(xd).sum()
    return E.onesize(r, d0, d1)["dcr"]


def clustered_upper(v, days, nb=2000):
    g = pd.DataFrame(dict(v=v, d=days)).groupby("d").v.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    k = rng.integers(0, len(s), (nb, len(s)))
    a = s[k].sum(1) / c[k].sum(1)
    return np.percentile(a, 95)


def evaluate(W, name, block, kind):
    """block: bool array over W (True = the filter blocks the trade; NaN-unscored are kept)."""
    rows = []
    for blk in ("DESIGN", "HOLDOUT", "ALL"):
        m = np.ones(len(W), bool) if blk == "ALL" else W.blk.values == blk
        k_, b_ = m & ~block, m & block
        r = dict(cand=name, kind=kind, blk=blk, N_all=int(m.sum()), N_kept=int(k_.sum()), N_blocked=int(b_.sum()))
        for e, col, col20 in (("XE", "y_xe", "y_xe20"), ("TP4", "y_4", "y_420")):
            v = W[col].values
            r[f"all_{e}"] = v[m].mean(); r[f"kept_{e}"] = v[k_].mean() if k_.any() else np.nan
            r[f"blocked_{e}"] = v[b_].mean() if b_.any() else np.nan
            r[f"kept20_{e}"] = W[col20].values[k_].mean() if k_.any() else np.nan
        v = W.y_xe.values
        if b_.sum() >= 5:
            vb = v[b_]; w_ = vb[vb > 0]; l_ = vb[vb <= 0]
            r["blocked_WR"] = (vb > 0).mean() * 100
            r["blocked_windows"] = len(np.unique(W.day.values[b_]))
            r["blocked_upper95"] = clustered_upper(vb, W.day.values[b_])
            pp = pd.Series(vb).groupby(W.pair.values[b_]).sum()
            r["blocked_top_pair_loss_share"] = (pp.min() / vb[vb < 0].sum() * 100) if (vb < 0).any() and pp.min() < 0 else np.nan
        if k_.sum() > 20:
            vk = v[k_]; xd = W.xd.values[k_]
            r["kept_ci_lo"], r["kept_ci_hi"] = E.boot_ci(vk, xd)
            pp = pd.Series(vk).groupby(W.pair.values[k_]).sum()
            r["kept_pair_share"] = pp.max() / vk.sum() * 100 if vk.sum() > 0 else np.nan
            kt, pf, be = E.kel(vk); r["kept_kelly"] = kt; r["kept_BE"] = be
        if blk != "ALL" and m.any():
            vx = W.y_xe.values; xd = W.xd.values
            r["dcr_before"] = dcr(vx[m], xd[m], blk)
            r["dcr_filter"] = dcr(vx[k_], xd[k_], blk)
            r["dcr_tilt"] = dcr(vx[m], xd[m], blk, np.where(block[m], 0.5, 1.5))
            alt = np.where(block, W.y_4.values, vx); xda = np.where(block, W.xd4.values, xd)
            r["dcr_condexit"] = dcr(alt[m], xda[m], blk)
            r["avg_condexit"] = alt[m].mean()
        for h, _, _ in L.HALVES:
            mh = k_ & (W.half.values == h)
            r[f"kept_{h}"] = v[mh].mean() if mh.sum() >= 5 else np.nan
        for s_ in (-1, 1):
            mm = k_ & (W.R1.values == s_); mb = b_ & (W.R1.values == s_)
            r[f"kept_R1{s_:+d}"] = v[mm].mean() if mm.any() else np.nan
            r[f"blocked_R1{s_:+d}"] = v[mb].mean() if mb.any() else np.nan
            r[f"blocked_R1{s_:+d}_win"] = len(np.unique(W.day.values[mb]))
        rows.append(r)
    return rows


def main():
    Y = pd.read_pickle(os.path.join(OD, "Y.pkl"))
    _, _, Z = E.load()
    C = pickle.load(open(os.path.join(OD, "core.pkl"), "rb"))
    NET = E.nets(Y, Z); net, net20, _ = NET
    idx = C["SEL"][("HYB", "XE")]
    W = Y.iloc[idx].reset_index(drop=True).copy()
    W["y_xe"] = net[idx, JX]; W["y_4"] = net[idx, J4]; W["y_xe20"] = net20[idx, JX]; W["y_420"] = net20[idx, J4]
    W["wk"] = Z["worked"][idx, JX]; W["xd"] = Z["xm"][idx, JX] // DAY; W["xd4"] = Z["xm"][idx, J4] // DAY
    W["R1"] = W.R1reg.values; W["R4"] = W.R4reg.values; W["day"] = W.entry_ms.values // DAY
    W["shared"] = C["shared"]["XE"]
    W["xi_row"] = idx
    des = W.blk.values == "DESIGN"
    print("trades", len(W), "DESIGN", des.sum(), flush=True)
    # screens
    T1, S1, D1, CV1 = screen(W, des, "DESIGN", 20, 15)
    T2, S2, D2, CV2 = screen(W, np.ones(len(W), bool), "FULL (descriptive)", 50, 40)
    pd.concat([T1, T2]).to_csv(os.path.join(OD, "wl_1d_buckets.csv"), index=False)
    pd.concat([S1, S2]).to_csv(os.path.join(OD, "wl_1d_spread.csv"), index=False)
    pd.concat([D1, D2]).to_csv(os.path.join(OD, "wl_2d.csv"), index=False)
    CV = CV1.merge(CV2[["var", "coverage"]].rename(columns={"coverage": "coverage_full"}), on="var")
    CV.to_csv(os.path.join(OD, "wl_coverage.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(CV[CV.coverage_full < 99].round(1).to_string())
    for S in (S1, S2):
        print(S.sort_values("p_fw_XE").head(12).round(3).to_string())
        print(S.sort_values("p_fw_runner").head(6).round(3).to_string())
        print(S.sort_values("p_fw_TP4").head(6).round(3).to_string())
    for D_ in (D1, D2):
        print(D_.sort_values("p_fw_lo").head(8).round(3).to_string()); print(D_.sort_values("p_fw_hi").head(8).round(3).to_string())
    # candidates: theory (frozen) + DESIGN survivors
    md_p4 = np.nanmedian(W.pos4.values[des]); md_dd = np.nanmedian(W.dd_hi.values[des])
    f0 = lambda a: np.nan_to_num(a, nan=0).astype(bool)
    with np.errstate(invalid="ignore"):
        cands = {"T1 shorts crowded (glA<1) block": f0(W.glA.values < 1.0), "T2 funding<0 block": f0(W.f_last.values < 0),
                 "T3 OI 1h rising block": f0(W.oi_chg1h.values > 0), f"T4 keep deep cut pos4≤{md_p4:+.4f}": f0(W.pos4.values > md_p4),
                 "T5 BTC 24h>0 block": f0(W.btc24h.values > 0), f"T6 keep near high dd_hi≤{md_dd:.2f}": f0(W.dd_hi.values > md_dd)}
    surv = []
    for _, s in S1.iterrows():
        if min(s.p_fw_XE, s.p_fw_runner) < 0.05:
            surv.append(s)
    T1d = T1.set_index(["var", "gran"])
    for s in surv:
        tb = T1[(T1["var"] == s["var"]) & (T1.gran == s.gran)]
        worst = tb.sort_values("avg_XE").iloc[0]
        x = W[s["var"]].values
        with np.errstate(invalid="ignore"):
            cands[f"S {s['var']} {s.gran} block [{worst.lo:.4g},{worst.hi:.4g}]"] = f0((x >= worst.lo) & (x <= worst.hi))
    sv2 = D1[(D1.p_fw_lo < 0.05)]
    for _, s in sv2.iterrows():
        tbs = {v: np.unique(np.nanquantile(W[v].values[des], [1 / 3, 2 / 3])) for v in (s.a, s.b)}
        ba = np.searchsorted(tbs[s.a], W[s.a].values, side="right"); bb = np.searchsorted(tbs[s.b], W[s.b].values, side="right")
        ok = np.isfinite(W[s.a].values) & np.isfinite(W[s.b].values)
        cands[f"S2 {s.a} T{s.a_terc} × {s.b} T{s.b_terc} block"] = ok & (ba == s.a_terc) & (bb == s.b_terc)
    rows = []
    for nm, b in cands.items():
        rows += evaluate(W, nm, b, "theory" if nm.startswith("T") else "survivor")
    FR = pd.DataFrame(rows); FR.to_csv(os.path.join(OD, "wl_candidates.csv"), index=False)
    cols = ["cand", "blk", "N_kept", "N_blocked", "all_XE", "kept_XE", "blocked_XE", "all_TP4", "kept_TP4", "blocked_TP4", "kept20_XE",
            "blocked_WR", "blocked_windows", "blocked_upper95", "kept_ci_lo", "kept_ci_hi", "kept_kelly", "dcr_before", "dcr_filter",
            "dcr_tilt", "dcr_condexit"]
    print(FR[cols].round(3).to_string())
    hcols = ["cand", "blk"] + [f"kept_{h}" for h, _, _ in L.HALVES] + ["kept_R1-1", "kept_R1+1", "blocked_R1-1", "blocked_R1+1", "blocked_R1-1_win", "blocked_R1+1_win"]
    print(FR[FR.blk == "ALL"][hcols].round(2).to_string())
    # joint extras (filter versions)
    ex = {}
    for nm, b in cands.items():
        ex[f"+ HYB XE filter: {nm}"] = (idx[~b], "XE", None)
        ex[f"+ HYB XE tilt: {nm}"] = (idx, "XE", np.where(b, 0.5, 1.5))
    pickle.dump(ex, open(os.path.join(OD, "joint_extra.pkl"), "wb"))
    W.to_pickle(os.path.join(OD, "W.pkl"))


if __name__ == "__main__":
    main()
