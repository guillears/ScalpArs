#!/usr/bin/env python3
"""NOSPOT_SLEEVE — ADDENDUM A: winners vs losers + theory filters.
Cohorts: NS0 × XA (the NOSPOT −0.09 cohort's exit) and × XE (shipped EXHAUSTION exit) + every primary rule at its best DESIGN exit.
Per cohort: coverage, 1D (sign / terciles / quintiles; cut points from the cohort's DESIGN trades) on booked net %, WORKED and REACH,
2D every pair at terciles, family-wise label-shuffle null (500), squeeze class vs winners, theory filters Q1–Q7 (DESIGN → survivors →
HOLDOUT + bar items + both BTC R1 signs + day-block bootstrap CI).
Writes reports/NOSPOT_SLEEVE_2026-11-18_wl1d.csv, _wl2d.csv, _coverage.csv, _squeeze.csv, _filters.csv"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import nospot_sleeve_lib as L   # noqa: E402
OUT = os.path.join(L.R, "reports", "NOSPOT_SLEEVE_2026-11-18")
rng = np.random.default_rng(5)
NSH = 500
VARS = ["oi_chg15m", "oi_chg1h", "oi_chg4h", "oi_usd", "oi_vol", "oi_vs_max", "f_last", "f_prev", "f_last8", "f_neg", "f_hourly",
        "topP", "topP_chg1h", "topP_chg4h", "topA", "topA_chg4h", "glA", "glA_chg1h", "glA_chg4h", "taker1h", "taker_prev1h", "prem1h",
        "prem_pk", "rsi15", "rsi1h", "di15", "dd_hi", "lh2", "vwap_gap", "peak_age_h", "hrs_since_spike", "vd", "volmult", "atrp", "vol24",
        "q5", "mcap", "turnover", "ret15m", "ret1h", "ret4h", "ret24h", "maxup1h_24h", "rise_on_to_hi", "hour", "R1", "R4", "btc1h", "btc4h",
        "btc24h"]
SIGN0 = {"oi_chg15m", "oi_chg1h", "oi_chg4h", "f_last", "f_prev", "f_last8", "topP_chg1h", "topP_chg4h", "topA_chg4h", "glA_chg1h",
         "glA_chg4h", "prem1h", "di15", "vwap_gap", "ret15m", "ret1h", "ret4h", "ret24h", "btc1h", "btc4h", "btc24h", "R1", "R4"}
SIGN1 = {"topP", "topA", "glA", "taker1h", "taker_prev1h"}
BIN = {"f_neg", "f_hourly"}
FILT = {"Q1_shorts_crowded": lambda X: X.glA.values < 1.0, "Q2_neg_funding": lambda X: X.f_last.values < 0,
        "Q3_oi_rising": lambda X: X.oi_chg1h.values > 0, "Q4_recent_squeeze": lambda X: X.maxup1h_24h.values >= 20,
        "Q5_hourly_funding": lambda X: X.f_int_h.values == 1, "Q6_small_float": lambda X: X.mcap.values < 50e6,
        "Q7_Q1orQ2": lambda X: (X.glA.values < 1.0) | (X.f_last.values < 0)}


def buckets(x, xd, var):
    """→ {gran: int bucket array (−1 = unscored)} with cut points from the DESIGN values xd."""
    out = {}
    ok = np.isfinite(x)
    if var in BIN or var == "R4":
        b = np.full(len(x), -1); b[ok] = (x[ok] > 0).astype(int) if var in BIN else (x[ok] + 1).astype(int); out["sign"] = b
        return out
    thr = 0.0 if var in SIGN0 else 1.0 if var in SIGN1 else np.nanmedian(xd)
    b = np.full(len(x), -1); b[ok] = (x[ok] > thr).astype(int); out["sign"] = b
    if var == "R1":
        return out
    for nm, q in (("terc", 3), ("quint", 5)):
        cuts = np.unique(np.nanquantile(xd, np.linspace(0, 1, q + 1)[1:-1])) if np.isfinite(xd).sum() >= q * 5 else None
        if cuts is None or not len(cuts):
            continue
        b = np.full(len(x), -1); b[ok] = np.searchsorted(cuts, x[ok], side="right"); out[nm] = b
    return out


def spread(y, b, nb, minn=30):
    s = np.bincount(b[b >= 0], weights=y[b >= 0], minlength=nb); n = np.bincount(b[b >= 0], minlength=nb)
    m = np.where(n >= minn, s / np.maximum(n, 1), np.nan)
    return (np.nanmax(m) - np.nanmin(m)) if np.isfinite(m).sum() >= 2 else np.nan


def wl(X, idx, e, name):
    y = X[f"{e}_net"].values[idx].astype(float); wk = X[f"{e}_worked"].values[idx].astype(float); rc = X[f"{e}_reach"].values[idx].astype(float)
    des = X.blk.values[idx] == "DESIGN"; half = X.half.values[idx]
    cov, rows, B = [], [], {}
    for var in VARS:
        x = X[var].values[idx].astype(float)
        cov.append(dict(cohort=name, var=var, coverage=np.isfinite(x).mean() * 100, des_cov=np.isfinite(x[des]).mean() * 100 if des.any() else np.nan))
        for g, b in buckets(x, x[des], var).items():
            nb = b.max() + 1
            B[(var, g)] = (b, nb)
            for k in range(nb):
                m = b == k
                if m.sum() == 0:
                    continue
                hs = [h for h, _, _ in L.HALVES if (m & (half == h)).sum() >= 10]
                rows.append(dict(cohort=name, var=var, gran=g, bucket=k, lo=np.nanmin(x[m]), hi=np.nanmax(x[m]), N=int(m.sum()),
                                 WR=(y[m] > 0).mean() * 100, avg=y[m].mean(), worked=wk[m].mean() * 100, reach=rc[m].mean(),
                                 des_avg=y[m & des].mean() if (m & des).any() else np.nan, hold_avg=y[m & ~des].mean() if (m & ~des).any() else np.nan,
                                 halves_pos=f"{sum(y[m & (half == h)].mean() > 0 for h in hs)}/{len(hs)}"))
            sp = spread(y, b, nb)
            rows.append(dict(cohort=name, var=var, gran=g, bucket=-9, N=int((b >= 0).sum()), spread=sp,
                             spread_worked=spread(wk, b, nb) * 100 if np.isfinite(sp) else np.nan, spread_reach=spread(rc, b, nb)))
    W = pd.DataFrame(rows)
    # family-wise null (1D): shuffle y
    keys = list(B)
    obs = np.array([spread(y, *B[k]) for k in keys]); obsw = np.array([spread(wk, *B[k]) for k in keys])
    mx = np.zeros(NSH); mxw = np.zeros(NSH)
    for s in range(NSH):
        p = rng.permutation(len(y))
        mx[s] = np.nanmax([spread(y[p], *B[k]) for k in keys]); mxw[s] = np.nanmax([spread(wk[p], *B[k]) for k in keys])
    for k, o, ow in zip(keys, obs, obsw):
        m_ = (W["var"] == k[0]) & (W["gran"] == k[1]) & (W["bucket"] == -9)
        W.loc[m_, "p_fw"] = (mx >= o).mean() if np.isfinite(o) else np.nan
        W.loc[m_, "p_fw_worked"] = (mxw >= ow).mean() if np.isfinite(ow) else np.nan
    # 2D terciles: stat = best cell avg (N ≥ 30) − cohort avg, and worst cell
    tv = [v for v in VARS if (v, "terc") in B]
    rows2 = []
    cells = []
    for i, a in enumerate(tv):
        for b_ in tv[i + 1:]:
            ba, _ = B[(a, "terc")]; bb, _ = B[(b_, "terc")]
            c = np.where((ba >= 0) & (bb >= 0), ba * 3 + bb, -1)
            cells.append((a, b_, c))
    def best2(yy):
        out = []
        for a, b_, c in cells:
            s = np.bincount(c[c >= 0], weights=yy[c >= 0], minlength=9); n = np.bincount(c[c >= 0], minlength=9)
            m = np.where(n >= 30, s / np.maximum(n, 1), np.nan)
            out.append(m)
        return np.array(out)
    M0 = best2(y)
    mx2 = np.zeros(NSH); mn2 = np.zeros(NSH)
    for s in range(200):
        Mp = best2(y[rng.permutation(len(y))])
        mx2[s] = np.nanmax(Mp); mn2[s] = np.nanmin(Mp)
    mx2, mn2 = mx2[:200], mn2[:200]
    for (a, b_, c), m in zip(cells, M0):
        for k in range(9):
            if np.isfinite(m[k]):
                mm = c == k
                rows2.append(dict(cohort=name, a=a, b=b_, cell=k, N=int(mm.sum()), avg=m[k], des_avg=y[mm & des].mean() if (mm & des).any() else np.nan,
                                  hold_avg=y[mm & ~des].mean() if (mm & ~des).any() else np.nan, worked=wk[mm].mean() * 100,
                                  p_fw_hi=(mx2 >= m[k]).mean(), p_fw_lo=(mn2 <= m[k]).mean()))
    return W, pd.DataFrame(rows2), pd.DataFrame(cov)


def squeeze(X, idx, e, name):
    y = X[f"{e}_net"].values[idx]; wk = X[f"{e}_worked"].values[idx]; k = X[f"{e}_k"].values[idx]
    sq = (y <= 0) & (wk == 0) & (k == 1); win = y > 0
    rows = []
    for var in VARS:
        x = X[var].values[idx].astype(float)
        a, b = x[sq], x[win]
        rows.append(dict(cohort=name, var=var, n_sq=int(sq.sum()), n_win=int(win.sum()), sq_med=np.nanmedian(a), win_med=np.nanmedian(b),
                         sq_mean=np.nanmean(a), win_mean=np.nanmean(b), auc=auc(a[np.isfinite(a)], b[np.isfinite(b)])))
    return pd.DataFrame(rows), sq.mean() * 100, y[sq].sum() / y[y <= 0].sum() * 100 if (y <= 0).any() else np.nan


def auc(a, b):
    if not len(a) or not len(b):
        return np.nan
    r = pd.Series(np.r_[a, b]).rank().values
    return (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def dayblock_ci(X, idx, e, n=1000):
    y = X[f"{e}_net"].values[idx]; d = X.day.values[idx]
    g = pd.DataFrame({"d": d, "y": y}).groupby("d").y.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    bs = []
    for _ in range(n):
        k = rng.integers(0, len(s), len(s)); bs.append(s[k].sum() / c[k].sum())
    return np.percentile(bs, [5, 95])


def filters(X, sel_all, cohorts):
    rows = []
    for name, (c, e) in cohorts.items():
        idx = sel_all[(c, e)]
        for fn, f in FILT.items():
            with np.errstate(invalid="ignore"):
                blk = np.nan_to_num(f(X)[idx].astype(float), nan=0).astype(bool)
            for bl in ("DESIGN", "HOLDOUT", "ALL"):
                bm = np.ones(len(idx), bool) if bl == "ALL" else X.blk.values[idx] == bl
                kept, blocked = idx[bm & ~blk], idx[bm & blk]
                rk = L.stats(X, kept, e, bl); rb = L.stats(X, blocked, e, bl); ru = L.stats(X, idx[bm], e, bl)
                rec = dict(cohort=name, filt=fn, blk=bl, N_all=ru["N"], avg_all=ru.get("avg"), N_kept=rk["N"], avg_kept=rk.get("avg"),
                           N_blocked=rb["N"], avg_blocked=rb.get("avg"), tpd_kept=rk["tpd"], WR_kept=rk.get("WR"), f20_kept=rk.get("f20"),
                           b5_kept=rk.get("b5"), kelly_kept=rk.get("kelly"), dcr_all=ru.get("dcr"), dcr_kept=rk.get("dcr"),
                           maxdd_kept=rk.get("maxdd"), pair_share=rk.get("pair_share"), day_share=rk.get("day_share"), ndays=rk.get("ndays"),
                           unscored=float(np.mean(~np.isfinite(X[{"Q1_shorts_crowded": "glA", "Q2_neg_funding": "f_last", "Q3_oi_rising": "oi_chg1h",
                                                                 "Q4_recent_squeeze": "maxup1h_24h", "Q5_hourly_funding": "f_int_h", "Q6_small_float": "mcap",
                                                                 "Q7_Q1orQ2": "glA"}[fn]].values[idx[bm]].astype(float))) * 100))
                for h, _, _ in L.HALVES:
                    rec[f"h_{h}"] = rk.get(f"h_{h}"); rec[f"h_{h}_N"] = rk.get(f"h_{h}_N")
                for s_ in (-1, 1):
                    mm = X.R1.values[kept] == s_
                    rec[f"R1{s_:+d}_avg"] = X[f"{e}_net"].values[kept][mm].mean() if mm.any() else np.nan
                    rec[f"R1{s_:+d}_N"] = int(mm.sum())
                if bl == "ALL" and len(kept) > 30:
                    rec["ci5"], rec["ci95"] = dayblock_ci(X, kept, e)
                rows.append(rec)
    F = pd.DataFrame(rows)
    # DESIGN survivor flag (Addendum A)
    for (cn, fn), g in F.groupby(["cohort", "filt"]):
        d = g[g.blk == "DESIGN"].iloc[0]
        hs = [h for h, _, _ in L.HALVES[:4] if (d.get(f"h_{h}_N") or 0) >= 10]
        hp = sum(d[f"h_{h}"] > 0 for h in hs) / len(hs) if hs else 0
        ok = (d.N_kept >= 30 and d.avg_kept > 0 and d.avg_kept > d.avg_all and (d.N_blocked == 0 or d.avg_blocked <= 0) and hp >= 2 / 3)
        F.loc[(F.cohort == cn) & (F.filt == fn), "des_survivor"] = ok
        F.loc[(F.cohort == cn) & (F.filt == fn), "des_hpos"] = hp
    # family-wise null for the filter family: shuffle the block flag within each cohort (keep its count), stat = kept avg − all avg (ALL span)
    obs, nul = {}, []
    for name, (c, e) in cohorts.items():
        idx = sel_all[(c, e)]; y = X[f"{e}_net"].values[idx]
        for fn, f in FILT.items():
            with np.errstate(invalid="ignore"):
                blk = np.nan_to_num(f(X)[idx].astype(float), nan=0).astype(bool)
            if blk.sum() == 0 or (~blk).sum() == 0:
                continue
            obs[(name, fn)] = (y[~blk].mean() - y.mean(), y, int((~blk).sum()))
    mx = np.zeros(NSH)
    for s in range(NSH):
        mx[s] = max(y[rng.permutation(len(y))[:nk]].mean() - y.mean() for _, y, nk in obs.values())
    for (name, fn), (o, _, _) in obs.items():
        F.loc[(F.cohort == name) & (F.filt == fn), "p_fw"] = (mx >= o).mean()
    return F


def main():
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    S = pickle.load(open(os.path.join(L.ND, "sel.pkl"), "rb")); sel = S["sel"]
    C = pd.read_csv(OUT + "_cells.csv")
    D = C[C.blk == "DESIGN"]
    cohorts = {"NS0|XA": ("NS0", "XA"), "NS0|XE": ("NS0", "XE")}
    for c in [x for x in D.cohort.unique() if x.startswith("R")]:
        e = D[D.cohort == c].sort_values("avg").iloc[-1].exit
        cohorts[f"{c}|{e}"] = (c, e)
    W1, W2, CV, SQ, SQs = [], [], [], [], []
    for name, (c, e) in cohorts.items():
        w1, w2, cv = wl(X, sel[(c, e)], e, name); W1.append(w1); W2.append(w2); CV.append(cv)
        sq, share, lossshare = squeeze(X, sel[(c, e)], e, name); SQ.append(sq)
        g = L.path_group(X, sel[(c, e)], e); vv = X[f"{e}_net"].values[sel[(c, e)]]
        pg = pd.Series(vv).groupby(g).agg(["count", "sum", "mean"]).reset_index().rename(columns={"index": "group"})
        pg["cohort"] = name; SQs.append(pg)
        print(name, "done", flush=True)
    pd.concat(W1).to_csv(OUT + "_wl1d.csv", index=False); pd.concat(W2).to_csv(OUT + "_wl2d.csv", index=False)
    pd.concat(CV).to_csv(OUT + "_coverage.csv", index=False); pd.concat(SQ).to_csv(OUT + "_squeeze.csv", index=False)
    pd.concat(SQs).to_csv(OUT + "_pathgroups_wl.csv", index=False)
    F = filters(X, sel, cohorts)
    F.to_csv(OUT + "_filters.csv", index=False)
    pickle.dump(cohorts, open(os.path.join(L.ND, "wl_cohorts.pkl"), "wb"))
    print(F[F.blk == "DESIGN"][["cohort", "filt", "N_all", "avg_all", "N_kept", "avg_kept", "N_blocked", "avg_blocked", "des_hpos",
                                "des_survivor", "p_fw"]].round(3).to_string())


if __name__ == "__main__":
    main()
