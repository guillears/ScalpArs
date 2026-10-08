#!/usr/bin/env python3
"""MOMENTUM LONG checklist items ① + ② (2026-10-08) — exhaustive 1D (sign → median → terciles → quintile tails) + all-pairs 2D
(median quadrants) scan over every PAIR-level and MACRO/REGIME variable, with a family-wise circular DAY-BLOCK shift null.

Cohorts (frame from scripts/study_ml_checklist_build.py):
  Y_all  yr5 replay ML, 3 seeds (replicates; N quoted per seed; days pooled)   Y_exw  same, ex washed-out (Jun-18→Jul-2 ∪ off30 ≤ −15)
  M_all  master kept non-probe ML ex-B1 (stack_pct)                             M_exw  same, ex washed-out
Statistic: Welch z of zone vs rest (trade-level). Null: the outcome vector (time-sorted) is circularly shifted against the fixed
features by a random offset (5–95 % of the sample) — keeps every day's outcomes together (day-block), breaks the feature link.
  p_mask  = share of shifts with |z| ≥ observed for THIS mask;  p_fwer = share of shifts whose MAX |z| over the whole scan ≥ observed.
OOS: the scan is re-run on yr5 H1 (< 2026-05-19, H1's own thresholds) and every H1 mask is re-read on H2 with the SAME thresholds.
LOMO: for the top masks, Δ with each yr5 month left out. Master: every yr5 mask re-read on master with yr5's frozen thresholds.
Out: reports/study_ml_checklist_scan_<cohort>.csv, reports/study_ml_checklist_scan_oos.csv, reports/study_ml_checklist_scan_summary.txt
"""
import os, sys, itertools
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
NPERM = int(os.environ.get("NPERM", 1000))
n_ = lambda s: pd.to_numeric(s, errors="coerce")

PAIR = ["entry_rsi", "entry_rsi_prev", "entry_adx", "entry_adx_prev", "entry_adx_delta", "entry_atr_pct", "entry_gap", "entry_ema_gap_5_8",
        "entry_ema_gap_8_13", "entry_gap_5_8_signed_pct", "entry_gap_5_20_prev_signed_pct", "entry_gap_expand_marginal",
        "entry_range_position", "entry_pair_volume_ratio", "entry_quality_score", "entry_pos_di", "entry_neg_di", "entry_ema20_slope",
        "entry_ema50_slope", "entry_dist_from_ema13_pct", "entry_price_vs_ema5_pct", "entry_ema5_stretch", "entry_pair_ema20_ema50_gap_pct",
        "entry_pair_1h_ema20_200_gap_pct", "entry_pair_rank", "log_vol24", "entry_pair_age_days", "k_pair_slope1h", "k_pair_gap1h_20_200",
        "k_pair_gap4h_20_50", "k_pair_ret1h", "k_pair_ret24h", "k_pair_off24h_high", "k_pair_above24h_low", "k_pair_qvol24_vs_7d",
        "k_pair_rel24_vs_btc", "d_rsi_chg", "d_di_spread", "k_funding_last"]
MACRO = ["entry_btc_rsi", "entry_btc_rsi_prev", "entry_btc_rsi_prev6", "entry_btc_rsi_closed", "entry_btc_rsi_1h", "entry_btc_rsi_1h_prev",
         "entry_btc_adx", "entry_btc_adx_prev", "entry_btc_atr_pct", "entry_btc_ema20_slope", "entry_btc_1h_slope", "entry_btc_trend_gap_pct",
         "entry_btc_dist_from_ema13_pct", "entry_btc_ema50_100_gap_pct", "entry_btc_1d_ret_pct", "entry_btc_off30d_high_pct",
         "entry_btc_off24h_pct", "entry_btc_off24lo_pct", "entry_btc_r72_pct", "entry_btc_eff72", "entry_btc_above72_pct", "entry_bull_pct",
         "entry_bear_pct", "entry_global_volume_ratio", "entry_eth_5m_ret1_pct", "k_btc_slope1h", "k_btc_1h_gap20_50", "k_btc_1h_gap20_200",
         "k_btc_4h_slope", "k_btc_4h_gap20_50", "k_btc_1d_slope", "k_btc_1d_gap9_20", "k_btc_vs_1d_ema20", "k_btc_day_ret",
         "k_btc_prevday_ret", "k_btc_off24h_high", "k_btc_above24h_low", "k_btc_off7d_high", "k_btc_above7d_low", "k_btc_off30d_high",
         "k_btc_ret1h", "k_btc_ret24h", "k_btc_ret72h", "k_btc_rv1h", "k_btc_rv24h", "k_btc_rv_ratio", "k_btc_eff72",
         "k_btc_slope1h_chg1h", "k_btc_slope1h_chg2h", "k_btc_slope1h_chg3h", "k_btc_hrs_since_slope_neg", "k_eth_slope1h",
         "k_eth_ret24h", "k_alt_med_ret24", "k_alt_up_share24", "k_btc_dom24", "k_hour_utc", "k_dow", "d_btc_rsi1h_chg",
         "d_btc_rsi5m_chg6", "d_btc_adx_chg", "regime_strong"]
CLASS = {**{c: "PAIR" for c in PAIR}, **{c: "MACRO" for c in MACRO}}


def load():
    D = pd.read_pickle("reports/study_ml_checklist_frame.pkl")
    D["log_vol24"] = np.log10(n_(D.entry_pair_volume_24h_usd).where(lambda x: x > 0))
    D["regime_strong"] = (D.entry_btc_regime.astype(str) == "STRONG_BULL").astype(float)
    for c in PAIR + MACRO:
        D[c] = n_(D[c]).astype(float)
    D["day"] = D.o.dt.strftime("%Y-%m-%d")
    return D


def cohorts(D):
    Y = D[D.src == "yr5"]; M = D[(D.src == "master") & (D.era != "B1")]
    return {"Y_all": Y, "Y_exw": Y[~Y.washed], "M_all": M, "M_exw": M[~M.washed]}


def usable(d, cols, cov=0.8):
    return [c for c in cols if d[c].notna().mean() >= cov and d[c].nunique() > 3]


def masks(X, ref, cols, two_d=True):
    """returns names, meta, Z (zone, n×k bool), V (valid, n×k bool). Thresholds from ref."""
    names, meta, Zs, Vs = [], [], [], []
    med = {}
    for c in cols:
        v, r = X[c].values, ref[c].dropna()
        val = ~np.isnan(v)
        q = r.quantile([0.2, 1 / 3, 0.5, 2 / 3, 0.8]).values; med[c] = q[2]
        defs = []
        if (r > 0).mean() >= 0.15 and (r <= 0).mean() >= 0.15:
            defs.append(("sign", f"{c}>0", v > 0, 0.0))
        defs += [("median", f"{c}>{q[2]:.4g}", v > q[2], q[2]), ("tercile", f"{c}<={q[1]:.4g}", v <= q[1], q[1]),
                 ("tercile", f"{c}>{q[3]:.4g}", v > q[3], q[3]), ("quintile", f"{c}<={q[0]:.4g}", v <= q[0], q[0]),
                 ("quintile", f"{c}>{q[4]:.4g}", v > q[4], q[4])]
        for g, nm, z, th in defs:
            names.append(nm); meta.append(dict(kind="1D", fam=CLASS[c], a=c, b="", gran=g, thr_a=th)); Zs.append(z & val); Vs.append(val)
    if two_d:
        for a, b in itertools.combinations(cols, 2):
            va, vb = X[a].values, X[b].values; val = ~np.isnan(va) & ~np.isnan(vb)
            ha, hb = va > med[a], vb > med[b]
            fam = "×".join(sorted([CLASS[a], CLASS[b]]))
            for sa, sb in ((1, 1), (1, 0), (0, 1), (0, 0)):
                z = (ha if sa else ~ha) & (hb if sb else ~hb) & val
                names.append(f"{a}{'>' if sa else '<='}{med[a]:.4g} ∧ {b}{'>' if sb else '<='}{med[b]:.4g}")
                meta.append(dict(kind="2D", fam=fam, a=a, b=b, gran=f"q{sa}{sb}", thr_a=med[a], thr_b=med[b])); Zs.append(z); Vs.append(val)
    return names, pd.DataFrame(meta), np.array(Zs).T, np.array(Vs).T


def zstats(Z, V, Y):
    """Y: n×P outcomes. returns z (k×P), dZ means etc. for column 0."""
    Zf, Vf = Z.astype(np.float32), V.astype(np.float32)
    Y = Y.astype(np.float32); Y2 = Y * Y
    nz, nv = Zf.sum(0)[:, None], Vf.sum(0)[:, None]
    sz, sv = Zf.T @ Y, Vf.T @ Y; qz, qv = Zf.T @ Y2, Vf.T @ Y2
    nr = nv - nz; sr, qr = sv - sz, qv - qz
    with np.errstate(divide="ignore", invalid="ignore"):
        mz, mr = sz / nz, sr / nr
        vz = (qz - nz * mz ** 2) / (nz - 1); vr = (qr - nr * mr ** 2) / (nr - 1)
        z = (mz - mr) / np.sqrt(vz / nz + vr / nr)
    return z, mz, mr


def scan(d, cols, label, minN, minD, ref=None, perm=NPERM, two_d=True, seed=11):
    d = d.sort_values("o").reset_index(drop=True)
    ref = d if ref is None else ref
    names, meta, Z, V = masks(d, ref, cols, two_d)
    y = d.pct.values.astype(np.float64)
    day = d.day.values
    # size filter (fills + days on both sides)
    dz = pd.get_dummies(day).values.astype(np.float32)
    days_z = ((Z.astype(np.float32).T @ dz) > 0).sum(1); days_r = (((V & ~Z).astype(np.float32).T @ dz) > 0).sum(1)
    nz, nr = Z.sum(0), (V & ~Z).sum(0)
    ok = (nz >= minN) & (nr >= minN) & (days_z >= minD) & (days_r >= minD)
    Z, V = Z[:, ok], V[:, ok]; meta = meta[ok].reset_index(drop=True); names = [x for x, o in zip(names, ok) if o]
    z0, mz, mr = zstats(Z, V, y[:, None]); z0, mz, mr = z0[:, 0], mz[:, 0], mr[:, 0]
    wr = (Z.astype(np.float32).T @ (y > 0).astype(np.float32)) / Z.sum(0)
    rng = np.random.default_rng(seed); n = len(y)
    exceed = np.zeros(len(z0)); maxnull = []; ZP = []
    for b0 in range(0, perm, 100):
        P = min(100, perm - b0)
        sh = rng.integers(int(0.05 * n), int(0.95 * n), P)
        Yp = np.stack([np.roll(y, s) for s in sh], 1)
        zp, _, _ = zstats(Z, V, Yp)
        zp = np.nan_to_num(np.abs(zp))
        exceed += (zp >= np.abs(z0)[:, None]).sum(1); maxnull += list(zp.max(0)); ZP.append(zp.astype(np.float16))
    maxnull = np.array(maxnull)
    # null distribution of the COUNT of masks at per-mask p < 0.01 (masks are correlated → 1 % × masks is only the mean)
    ZP = np.concatenate(ZP, 1).astype(np.float32); q99 = np.quantile(ZP, 0.99, axis=1)
    cnt_null = (ZP > q99[:, None]).sum(0); cnt_obs = int((np.abs(z0) > q99).sum()); del ZP
    T = meta.assign(name=names, nZ=nz[ok], nR=nr[ok], daysZ=days_z[ok], avgZ=mz, avgR=mr, dlt=mz - mr, wrZ=wr * 100, z=z0,
                    p_mask=(exceed + 1) / (perm + 1), p_fwer=[((maxnull >= abs(v)).sum() + 1) / (perm + 1) for v in z0])
    T["cohort"] = label
    info = dict(cohort=label, n=n, masks=len(T), max_abs_z=np.nanmax(np.abs(z0)), null95=np.percentile(maxnull, 95),
                p_fwer_best=T.p_fwer.min(), n_fwer05=int((T.p_fwer < 0.05).sum()), n_pmask01=int((T.p_mask < 0.01).sum()),
                exp_pmask01=0.01 * len(T), cnt_obs=cnt_obs, cnt_null_med=np.median(cnt_null), cnt_null95=np.percentile(cnt_null, 95),
                p_cnt=((cnt_null >= cnt_obs).sum() + 1) / (perm + 1))
    return T, info, (Z, V, d)


def reread(T, d_target, ref, cols):
    """re-read the masks of T on another cohort with ref's thresholds (frozen) → Δ, N, avg."""
    names, meta, Z, V = masks(d_target, ref, cols, True)
    idx = {nm: i for i, nm in enumerate(names)}
    y = d_target.pct.values
    out = []
    for nm in T.name:
        i = idx.get(nm)
        if i is None:
            out.append((np.nan, np.nan, np.nan, np.nan)); continue
        z, v = Z[:, i], V[:, i]; r = v & ~z
        out.append((z.sum(), y[z].mean() if z.sum() else np.nan, y[r].mean() if r.sum() else np.nan, (y[z] > 0).mean() * 100 if z.sum() else np.nan))
    o = pd.DataFrame(out, columns=["n", "avgZ", "avgR", "wrZ"], index=T.index)
    o["dlt"] = o.avgZ - o.avgR
    return o


def main():
    D = load(); CO = cohorts(D); S = []; lines = []
    Yall = CO["Y_all"]
    cols_y = usable(Yall, PAIR + MACRO)
    cols_m = usable(CO["M_all"], PAIR + MACRO)
    lines.append(f"variables usable on yr5: {len(cols_y)} (PAIR {sum(CLASS[c]=='PAIR' for c in cols_y)}, MACRO {sum(CLASS[c]=='MACRO' for c in cols_y)})")
    lines.append(f"variables usable on master (≥80 % coverage): {len(cols_m)}; yr5-only: {sorted(set(cols_y)-set(cols_m))}")
    lines.append(f"not usable on yr5: {sorted(set(PAIR+MACRO)-set(cols_y))}")
    res = {}
    for lab, minN, minD, cols in (("Y_all", 60, 10, cols_y), ("Y_exw", 60, 10, cols_y), ("M_all", 12, 6, cols_m), ("M_exw", 12, 6, cols_m)):
        T, info, _ = scan(CO[lab], cols, lab, minN, minD)
        if lab.startswith("Y"):
            # halves (full-cohort thresholds) + master re-read with frozen yr5 thresholds
            d = CO[lab]
            for h in ("H1", "H2"):
                r = reread(T, d[d.half == h].sort_values("o"), d, cols_y); T[f"dlt_{h}"] = r.dlt.values; T[f"n_{h}"] = r.n.values / 3
            mlab = "M_all" if lab == "Y_all" else "M_exw"
            r = reread(T, CO[mlab].sort_values("o"), d, cols_y); T["m_n"] = r.n.values; T["m_avgZ"] = r.avgZ.values; T["m_dlt"] = r.dlt.values
            T["nZ_seed"] = T.nZ / 3
        T.to_csv(f"reports/study_ml_checklist_scan_{lab}.csv", index=False); res[lab] = T
        lines.append(f"[{lab}] n={info['n']} masks={info['masks']:,} max|z|={info['max_abs_z']:.2f} null95(max|z|)={info['null95']:.2f} "
                     f"best p_fwer={info['p_fwer_best']:.3f} · survivors p_fwer<0.05: {info['n_fwer05']} · masks beyond their own null 99th pct: {info['cnt_obs']} "
                     f"(null count median {info['cnt_null_med']:.0f}, 95th {info['cnt_null95']:.0f}, p={info['p_cnt']:.3f})")
        for fam, g in T.groupby("fam"):
            lines.append(f"     family {fam:<12} masks {len(g):>6,} · max|z| {np.nanmax(np.abs(g.z)):.2f} · p_mask<0.01 {int((g.p_mask<0.01).sum()):>4} (chance {0.01*len(g):.0f}) · best p_fwer {g.p_fwer.min():.3f}")
        print("\n".join(lines[-6:]), flush=True)
    # OOS: discover on yr5 H1 (H1 thresholds), test on H2 with the same thresholds
    for lab in ("Y_all", "Y_exw"):
        d = CO[lab]; h1, h2 = d[d.half == "H1"], d[d.half == "H2"]
        T1, info1, _ = scan(h1, cols_y, lab + "_H1", 30, 8, perm=1000)
        r = reread(T1, h2.sort_values("o"), h1, cols_y)
        T1["h2_n_seed"] = r.n.values / 3; T1["h2_dlt"] = r.dlt.values; T1["h2_avgZ"] = r.avgZ.values
        top = T1[T1.p_mask < 0.01].copy()
        rep = (np.sign(top.dlt) == np.sign(top.h2_dlt)) & (top.h2_n_seed * 3 >= 30)
        allrep = (np.sign(T1.dlt) == np.sign(T1.h2_dlt))
        lines.append(f"[OOS {lab}] H1 scan: {info1['masks']:,} masks · max|z| {info1['max_abs_z']:.2f} vs null95 {info1['null95']:.2f} · best p_fwer {info1['p_fwer_best']:.3f} · "
                     f"H1 p_mask<0.01: {len(top)} → same sign on H2: {int(rep.sum())}/{len(top)} ({rep.mean()*100 if len(top) else float('nan'):.0f} %) vs all-mask baseline {allrep.mean()*100:.0f} %")
        T1.to_csv(f"reports/study_ml_checklist_scan_oos_{lab}.csv", index=False)
        print(lines[-1], flush=True)
    # LOMO for the top-30 yr5 masks by |z| (Y_all, Y_exw)
    for lab in ("Y_all", "Y_exw"):
        T = res[lab]; d = CO[lab].sort_values("o").reset_index(drop=True)
        names, meta, Z, V = masks(d, d, cols_y, True); idx = {nm: i for i, nm in enumerate(names)}
        top = T.reindex(T.z.abs().sort_values(ascending=False).index).head(30)
        rows = []
        for _, r in top.iterrows():
            i = idx[r["name"]]; z, v = Z[:, i], V[:, i]; dl = []
            for mth in sorted(d.month.unique()):
                k = (d.month != mth).values; zz, rr = z & k, v & ~z & k
                dl.append(d.pct.values[zz].mean() - d.pct.values[rr].mean())
            mo = [d.pct.values[z & (d.month == m_).values].mean() - d.pct.values[v & ~z & (d.month == m_).values].mean() for m_ in sorted(d.month.unique())]
            rows.append(dict(name=r["name"], fam=r.fam, z=r.z, dlt=r.dlt, lomo_min=np.nanmin(dl), lomo_max=np.nanmax(dl),
                             months_same_sign=int(np.nansum(np.sign(mo) == np.sign(r.dlt))), months=int(np.sum(~np.isnan(mo))),
                             p_fwer=r.p_fwer, dlt_H1=r.dlt_H1, dlt_H2=r.dlt_H2, m_n=r.m_n, m_dlt=r.m_dlt))
        L = pd.DataFrame(rows); L.to_csv(f"reports/study_ml_checklist_lomo_{lab}.csv", index=False)
        lines.append(f"[LOMO {lab}] top-30 by |z|: LOMO sign-stable {int(((np.sign(L.lomo_min)==np.sign(L.lomo_max))).sum())}/30 · "
                     f"both halves same sign {int((np.sign(L.dlt_H1)==np.sign(L.dlt_H2)).sum())}/30 · master agrees (n≥12) "
                     f"{int(((np.sign(L.m_dlt)==np.sign(L.dlt))&(L.m_n>=12)).sum())}/{int((L.m_n>=12).sum())}")
        print(lines[-1], flush=True)
    open("reports/study_ml_checklist_scan_summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
