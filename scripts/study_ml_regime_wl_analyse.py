#!/usr/bin/env python3
"""ML REGIME W/L study 2026-10-26 — steps 1–5 (coverage · uniform degradation · regime separators · within-regime entry screen ·
candidates). Pre-registration: reports/ML_REGIME_WL_2026-10-26_PREREG.md (frozen before any regime × outcome table).
Inputs (read-only): study_ml_wl_b25prep_data.load_all (158 KEEP master + B25 provisional, PROBE apart), the per-fill regime file built by
study_ml_regime_wl_build.py, reports/ml_regime/day_features.csv (Oct-16 day-level set, ≤ 10-16), yr6 replay (refute-only).
Outputs: reports/ML_REGIME_WL_2026-10-26_{coverage,uniform,regime_screen,regime_day,tape,within_1d,within_2d,candidates}.csv + _tables.md
    venv/bin/python -W ignore scripts/study_ml_regime_wl_analyse.py
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_ml_wl_b25prep_data as L                       # noqa: E402

PFX = os.path.join(ROOT, "reports", "ML_REGIME_WL_2026-10-26")
RNG = np.random.default_rng(20261026)
GROUPS = ["G0 BASE", "G1 B1", "G2 B2", "G3 B3", "G4 B4-B11", "G5 B12-B16", "G6 B17-B24"]
GOOD, BAD = GROUPS[:4], GROUPS[5:]
BE_WR = 61.5
OUT = []


def P(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s, flush=True)


def st(y):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(y) == 0:
        return "0"
    return f"{len(y)}·{(y > 0).mean() * 100:.0f}%·{y.mean():+.3f}"


# ───────────────────────────── data ─────────────────────────────
def load():
    D = L.load_all()
    D = D[~D.is_probe & (D.entry_strategy.astype(str) != "MANUAL")].copy()
    F = pd.read_csv(PFX + "_fill_regime.csv")
    F = F.drop_duplicates(["opened_at", "pair"])
    D = D.merge(F, left_on=["key_open", "pair"], right_on=["opened_at", "pair"], how="left", suffixes=("", "_rg"))
    # Oct-16 day-level set (value for day d uses days ≤ d-1) — coverage ends 2026-10-16
    df = pd.read_csv(os.path.join(ROOT, "reports/ml_regime/day_features.csv"))
    df["day"] = pd.to_datetime(df.day_ms, unit="ms")
    df = df.drop(columns=["day_ms"] + [c for c in df.columns if c.startswith("ml_")])  # own-result features: BT-only
    df.columns = ["day" if c == "day" else "d16_" + c for c in df.columns]
    D = D.merge(df, on="day", how="left")
    D["y"] = D.y_keep
    D["win"] = D.y > 0
    D["late"] = D.era_group.isin(BAD).astype(float)
    D["ng_loser"] = (pd.to_numeric(D.peak_pnl, errors="coerce") < 0.10) & (D.y_trad < 0)
    D["ng_loser_keep"] = (pd.to_numeric(D.peak_pnl, errors="coerce") < 0.10) & (D.y < 0)
    return D.sort_values("ts").reset_index(drop=True)


REGIME_OWN = None


def regime_columns(D):
    F = pd.read_csv(PFX + "_fill_regime.csv", nrows=2)
    own = [c for c in F.columns if c not in ("opened_at", "pair", "btc_bar_lag_min", "br_n_scored")]
    stamps = [c for c in D.columns if c.startswith("entry_") and L.is_macro(c) and c not in L.NON_SIGNAL and c not in L.CATEG
              and pd.to_numeric(D[c], errors="coerce").notna().sum() > 0]
    cats = [c for c in D.columns if c.startswith("cat__entry_btc_regime") or c.startswith("cat__entry_macro_trend")]
    d16 = [c for c in D.columns if c.startswith("d16_")]
    return own, stamps, cats, d16


NATURAL_ZERO = ("_r1h", "_r4h", "_r24h", "_r3d", "_r7d", "_r30d", "slp", "_c_e20", "_e20_e50", "_e50_e200", "_di_", "adxchg", "minus",
                "ethbtc", "ac5", "med_r", "funding_med", "1h_slope", "ema20_slope", "trend_gap", "dist_from_ema13", "1d_ret", "r72",
                "ema50_100", "4h_ema50_200", "d16_r", "d16_dma", "d16_alt_rs", "d16_ft", "d16_fund")


def has_zero(c):
    return any(k in c for k in NATURAL_ZERO)


def zones(x, c):
    """sign (natural zero) → median → terciles. returns [(label, mask)] on scored rows."""
    s = np.isfinite(x)
    v = x[s]
    out = []
    if len(np.unique(v)) < 2:
        return out
    if set(np.unique(v)) <= {0.0, 1.0}:
        out.append(("==1", s & (x == 1)))
        return out
    if c.endswith("rv24_rv7") or "vr15" in c or "d16_btc_vr" in c or "d16_alt_vr" in c or c.endswith("_x7d") or c.endswith("x24h"):
        thr0 = 1.0
        if 0.05 < (v > thr0).mean() < 0.95:
            out.append((">1", s & (x > thr0)))
    elif has_zero(c) and 0.05 < (v > 0).mean() < 0.95:
        out.append((">0", s & (x > 0)))
    m = np.median(v); q1, q2 = np.quantile(v, [1 / 3, 2 / 3])
    out.append((f">med({m:.4g})", s & (x > m)))
    out.append((f"<=q33({q1:.4g})", s & (x <= q1)))
    out.append((f">q67({q2:.4g})", s & (x > q2)))
    return out


def shifts(y, R):
    n = len(y)
    ks = RNG.choice(np.arange(5, n - 5), size=R, replace=True)
    return np.stack([np.roll(y, k) for k in ks])


def day_boot(y, day, R=4000, stat="mean"):
    """window(day)-clustered bootstrap distribution of the mean."""
    df = pd.DataFrame({"y": y, "d": day}).dropna()
    if len(df) == 0:
        return np.array([np.nan])
    g = df.groupby("d").y.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    k = len(g)
    idx = RNG.integers(0, k, size=(R, k))
    return s[idx].sum(1) / c[idx].sum(1)


def strat_delta(y, z, grp, groups=GROUPS, mn=2):
    num = den = 0.0; per = {}
    for g in groups:
        m = grp == g
        a, b = y[m & z], y[m & ~z]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) >= mn and len(b) >= mn:
            d = a.mean() - b.mean(); per[g] = d
            w = len(a) + len(b); num += w * d; den += w
    return (num / den if den else np.nan), per


def ols_late_absorb(y, late, z):
    ok = np.isfinite(y)
    y, late, z = y[ok], late[ok], z[ok].astype(float)
    X1 = np.c_[np.ones(len(y)), late]
    b1 = np.linalg.lstsq(X1, y, rcond=None)[0]
    X2 = np.c_[np.ones(len(y)), late, z]
    b2 = np.linalg.lstsq(X2, y, rcond=None)[0]
    return b1[1], b2[1], b2[2]


# ───────────────────────────── steps ─────────────────────────────
def coverage(D, cols):
    rows = []
    for c in cols:
        x = pd.to_numeric(D[c], errors="coerce")
        r = {"variable": c}
        for g in GROUPS + ["B25 (prov.)"]:
            m = D.era_group == g
            r[g] = f"{int(x[m].notna().sum())}/{int(m.sum())}"
        k = D.y.notna()
        r["KEEP_cov"] = round(x[k].notna().mean() * 100)
        rows.append(r)
    return pd.DataFrame(rows)


def uniform(D):
    K = D[D.y.notna() & D.era_group.isin(GROUPS)].copy()
    n = lambda c: pd.to_numeric(K[c], errors="coerce")
    q = lambda s: pd.qcut(s, 3, labels=["T1 low", "T2 mid", "T3 high"])
    subs = {
        "ALL": pd.Series("all", index=K.index),
        "confidence": K.confidence.astype(str),
        "btc_regime stamp": K.entry_btc_regime.astype(str),
        "pattern W-any": np.where(n("entry_pattern_w_any_match") == 1, "W-match", "no W"),
        "pattern C-any": np.where(n("entry_pattern_c_any_match") == 1, "C-match", "no C"),
        "pair 24h vol tercile": q(n("entry_pair_volume_24h_usd")).astype(str),
        "pair rank": np.where(n("entry_pair_rank") <= 25, "rank ≤ 25", "rank > 25"),
        "ATR% tercile": q(n("entry_atr_pct")).astype(str),
        "pair ADX dir": np.where(n("d_adx_dir") > 0, "ADX rising", "ADX flat/falling"),
        "pair ADX tercile": q(n("entry_adx")).astype(str),
        "RSI tercile": q(n("entry_rsi")).astype(str),
        "hour UTC": pd.cut(K.ts.dt.hour, [-1, 7, 15, 23], labels=["00-07", "08-15", "16-23"]).astype(str),
        "weekend": np.where(K.ts.dt.dayofweek >= 5, "weekend", "weekday"),
        "PAIR_1H_DOWNTREND": np.where(n("entry_pair_1h_ema20_200_gap_pct") <= 0, "pair 1h down", np.where(n("entry_pair_1h_ema20_200_gap_pct") > 0, "pair 1h up", "unscored")),
        "same-scan fills": np.where(K.d_fills_same_scan > 1, "multi-fill scan", "single"),
    }
    rows = []
    for nm, lab in subs.items():
        lab = pd.Series(lab, index=K.index).astype(str)
        for v in sorted(lab.unique()):
            m = lab == v
            r = {"dimension": nm, "cohort": v}
            for g in GROUPS:
                y = K.y[m & (K.era_group == g)]
                r[g] = st(y)
            yg, yb = K.y[m & K.era_group.isin(GOOD)], K.y[m & K.era_group.isin(BAD)]
            r["good G0-G3"] = st(yg); r["bad G5-G6"] = st(yb)
            r["Δ bad−good"] = yb.mean() - yg.mean() if len(yg) and len(yb) else np.nan
            ng_g = (K.ng_loser_keep[m & K.era_group.isin(GOOD)]).mean() * 100 if len(yg) else np.nan
            ng_b = (K.ng_loser_keep[m & K.era_group.isin(BAD)]).mean() * 100 if len(yb) else np.nan
            r["NG-loser share good→bad (% of fills)"] = f"{ng_g:.0f}→{ng_b:.0f}" if np.isfinite(ng_g) and np.isfinite(ng_b) else "—"
            r["n good / bad"] = f"{len(yg)}/{len(yb)}"
            rows.append(r)
    U = pd.DataFrame(rows)
    # per-pair concentration of the bad-era net loss
    B = K[K.era_group.isin(BAD)]
    neg = B.groupby("pair").y.sum().sort_values()
    loss = -B.y[B.y < 0].sum()
    return U, neg, loss, B


def tape(D):
    """era-group tape context from fill-level regime medians + calendar BTC path over the era span."""
    import study_ml_regime_wl_build as BB
    k = BB.load5("BTCUSDT"); k["ts"] = pd.to_datetime(k.t, unit="ms")
    rows = []
    for g in GROUPS + ["B25 (prov.)"]:
        m = D.era_group == g
        t0, t1 = D.ts[m].min().floor("D"), D.ts[m].max().ceil("D")
        kk = k[(k.ts >= t0) & (k.ts < t1)]
        r = {"group": g, "span": f"{t0.date()}→{(t1 - pd.Timedelta(days=1)).date()}", "days w/ fills": D.day[m].nunique(),
             "KEEP": st(D.y[m]), "TRAD": st(D.y_trad[m]),
             "BTC span return %": (kk.c.iloc[-1] / kk.c.iloc[0] - 1) * 100 if len(kk) else np.nan,
             "BTC span max DD %": ((kk.c / kk.c.cummax()) - 1).min() * 100 if len(kk) else np.nan}
        for c in ["btc_r7d", "btc_r30d", "btc_offhi_30d", "btc_eff_24h", "btc_eff_72h", "btc_rv_24h", "btc_rv24_pctl90d", "btc_adx_1h",
                  "btc_adx_4h", "btc_e20_e50_4h", "btc_e50_e200_1d", "br_above_ema50_1h", "br_adv_24h", "br_corr24h_top20",
                  "br_beta_btc_24h", "br_disp_24h", "alt_minus_btc_7d", "ethbtc_r7d", "br_qv24_x7d", "br_funding_med", "btc_vr15_7d",
                  "entry_bull_pct", "entry_global_volume_ratio"]:
            r[c] = pd.to_numeric(D[c][m], errors="coerce").median()
        r["NG-loser % of fills (KEEP)"] = D.ng_loser_keep[m & D.y.notna()].mean() * 100
        r["NG share of losers (KEEP)"] = D.ng_loser_keep[m & (D.y < 0)].mean() * 100 if (m & (D.y < 0)).any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def regime_screen(D, cols, R=1000):
    K = D[D.y.notna() & D.era_group.isin(GROUPS)].reset_index(drop=True)
    T = D[D.y_trad.notna() & D.era_group.isin(GROUPS)].reset_index(drop=True)
    y = K.y.values; Y = shifts(y, R)
    grp = K.era_group.values; late = K.late.values
    rows, masks = [], []
    for c in cols:
        x = pd.to_numeric(K[c], errors="coerce").values
        if np.isfinite(x).sum() < 30:
            continue
        for lab, z in zones(x, c):
            s = np.isfinite(x)
            nz, nr = z.sum(), (s & ~z).sum()
            if nz < 10 or nr < 10:
                continue
            masks.append((c, lab, z, s))
    # vectorised null: Δ = mean(z) − mean(rest on scored)
    Zm = np.stack([m[2] for m in masks]).astype(float)               # tests × n
    Rm = np.stack([m[3] & ~m[2] for m in masks]).astype(float)
    obs = Zm @ y / Zm.sum(1) - Rm @ y / Rm.sum(1)
    null = Y @ Zm.T / Zm.sum(1) - Y @ Rm.T / Rm.sum(1)             # R × tests
    p = ((np.abs(null) >= np.abs(obs)).sum(0) + 1) / (R + 1)
    fw = np.abs(null).max(1)
    pfw = ((fw[:, None] >= np.abs(obs)[None, :]).sum(0) + 1) / (R + 1)
    for i, (c, lab, z, s) in enumerate(masks):
        x = pd.to_numeric(K[c], errors="coerce").values
        yz, yr = y[z], y[s & ~z]
        sd, per = strat_delta(y, z, grp)
        # LOEO: refit the cut on the other groups (same zone type), sign on the held-out group
        agree = tot = 0
        sgn = np.sign(obs[i])
        for g in GROUPS:
            tr, te = (grp != g) & s, (grp == g) & s
            if te.sum() < 4:
                continue
            xt = x[tr]
            if len(xt) < 10:
                continue
            if lab.startswith(">med"):
                zz = x > np.median(xt)
            elif lab.startswith("<=q33"):
                zz = x <= np.quantile(xt, 1 / 3)
            elif lab.startswith(">q67"):
                zz = x > np.quantile(xt, 2 / 3)
            else:
                zz = z
            a, b = y[te & zz], y[te & ~zz]
            if len(a) >= 2 and len(b) >= 2:
                tot += 1; agree += int(np.sign(a.mean() - b.mean()) == sgn)
        b_late0, b_late1, b_z = ols_late_absorb(y, late, z & s)
        # days
        dz = K.day[z]; dr = K.day[s & ~z]
        dm_z = K[z].groupby("day").y.mean(); dm_r = K[s & ~z].groupby("day").y.mean()
        # ex washed
        w = K.washed.values
        exw = y[z & ~w].mean() - y[s & ~z & ~w].mean() if (z & ~w).sum() >= 5 and (s & ~z & ~w).sum() >= 5 else np.nan
        # TRAD
        xt = pd.to_numeric(T[c], errors="coerce").values
        thr = None
        rows.append({"variable": c, "zone": lab, "macro": L.is_macro(c) or c.startswith(("btc_", "br_", "alt_", "eth", "d16_")),
                     "N_zone": int(z.sum()), "zone": lab, "zone_stat": st(yz), "rest_stat": st(yr), "delta": obs[i],
                     "p_null": p[i], "p_fw": pfw[i], "era_strat_delta": sd,
                     "era_groups_same_sign": f"{sum(np.sign(v) == np.sign(obs[i]) for v in per.values())}/{len(per)}",
                     "loeo_agree": f"{agree}/{tot}", "late_coef_before": b_late0, "late_coef_after": b_late1,
                     "gap_absorbed_%": (1 - b_late1 / b_late0) * 100 if b_late0 else np.nan,
                     "zone_days": dz.nunique(), "rest_days": dr.nunique(), "zone_daymean": dm_z.mean(), "rest_daymean": dm_r.mean(),
                     "zone_neg_day_share": (dm_z < 0).mean() * 100, "delta_ex_washed": exw,
                     "zone_share_good": z[np.isin(grp, GOOD)].mean() * 100, "zone_share_bad": z[np.isin(grp, BAD)].mean() * 100,
                     **{f"Δ {g}": per.get(g, np.nan) for g in GROUPS}})
    S = pd.DataFrame(rows)
    S = S.loc[:, ~S.columns.duplicated()]
    return S, K


def day_level(D, cols, R=None):
    """day units: per-day mean outcome vs per-day mean regime value; Spearman across days + full day-rotation null."""
    K = D[D.y.notna() & D.era_group.isin(GROUPS)]
    Dd = K.groupby("day").agg(y=("y", "mean"), n=("y", "size"), grp=("era_group", "first"))
    rows = []
    for c in cols:
        v = pd.to_numeric(K[c], errors="coerce").groupby(K.day).mean()
        d = Dd.join(v.rename("x")).dropna()
        if len(d) < 20 or d.x.nunique() < 3:
            continue
        rx, ry = d.x.rank().values, d.y.rank().values
        rho = np.corrcoef(rx, ry)[0, 1]
        n = len(d)
        nul = np.array([np.corrcoef(rx, np.roll(ry, k))[0, 1] for k in range(3, n - 3)])
        p = ((np.abs(nul) >= abs(rho)).sum() + 1) / (len(nul) + 1)
        hi = d.x > d.x.median()
        # within-era rank corr (era-demeaned)
        dd = d.copy(); dd["yd"] = dd.y - dd.groupby("grp").y.transform("mean"); dd["xd"] = dd.x - dd.groupby("grp").x.transform("mean")
        rho_w = np.corrcoef(dd.xd.rank(), dd.yd.rank())[0, 1]
        rows.append({"variable": c, "days": n, "spearman_days": rho, "p_rotation": p, "spearman_within_era": rho_w,
                     "hi_days_meanofmeans": d.y[hi].mean(), "lo_days_meanofmeans": d.y[~hi].mean(),
                     "hi_days_neg_share": (d.y[hi] < 0).mean() * 100, "lo_days_neg_share": (d.y[~hi] < 0).mean() * 100})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = load()
    own, stamps, cats, d16 = regime_columns(D)
    allreg = own + stamps + cats + d16
    P(f"cohort: TRAD {D.y_trad.notna().sum()} · KEEP {D.y.notna().sum()} (B25 {int((D.era_group == 'B25 (prov.)').sum())}) · "
      f"regime vars: own {len(own)} · stamps {len(stamps)} · regime-label cats {len(cats)} · Oct-16 day set {len(d16)}")
    C = coverage(D, allreg); C.to_csv(PFX + "_coverage.csv", index=False)
    U, neg, loss, B = uniform(D); U.to_csv(PFX + "_uniform.csv", index=False)
    Tp = tape(D); Tp.to_csv(PFX + "_tape.csv", index=False)
    S, K = regime_screen(D, allreg); S.to_csv(PFX + "_regime_screen.csv", index=False)
    Dl = day_level(D, allreg); Dl.to_csv(PFX + "_regime_day.csv", index=False)
    P("bad-era loss by pair (top 8):", neg.head(8).round(2).to_dict(), "total loss", round(loss, 2))
    P(f"regime tests {len(S)} · p<0.05 {int((S.p_null < 0.05).sum())} (chance ≈ {0.05 * len(S):.0f}) · p_fw<0.05 {int((S.p_fw < 0.05).sum())}")
    D.to_pickle(PFX + "_D.pkl")
