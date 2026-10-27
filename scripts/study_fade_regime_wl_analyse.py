#!/usr/bin/env python3
"""FADE REGIME W/L study 2026-10-26 — steps 1–3 (coverage · uniform degradation · tape · regime separators in fill AND day units ·
LOEO · B25 · backtest caveat) + scoring of the PREREG predictions. Pre-registration: reports/FADE_REGIME_WL_2026-10-26_PREREG.md.
Outputs reports/FADE_REGIME_WL_2026-10-26_{coverage,uniform,tape,regime_screen,regime_day,prereg_score}.csv + _analyse.txt
    venv/bin/python -W ignore scripts/study_fade_regime_wl_analyse.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_fade_regime_wl_common as C    # noqa: E402
import study_ml_regime_wl_build as B        # noqa: E402

PFX = C.PFX
OUT = []
RNG = np.random.default_rng(20261026)


def P(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s, flush=True)


def coverage(A, cols):
    rows = []
    for c in cols:
        x = C.num(A, c); r = {"variable": c}
        for g in C.GROUPS + ["B25"]:
            m = (A.eg == g) & (A.blk != "MASTER_BLK")
            r[g] = f"{int(x[m].notna().sum())}/{int(m.sum())}"
        k = A.blk == "MASTER"
        r["master_cov_%"] = round(x[k].notna().mean() * 100)
        rows.append(r)
    return pd.DataFrame(rows)


def uniform(K):
    n = lambda c: C.num(K, c)
    q = lambda s: pd.qcut(s.rank(method="first"), 3, labels=["T1 low", "T2 mid", "T3 high"]).astype(str).where(s.notna(), "unscored")
    subs = {
        "ALL": pd.Series("all", index=K.index),
        "btc_regime stamp": K.entry_btc_regime.fillna("unscored").astype(str),
        "pair 24h vol tercile": q(n("entry_pair_volume_24h_usd")),
        "pair ADX tercile": q(n("entry_adx")),
        "ADX Δ (DL374 ① 2.359)": np.where(n("entry_adx_delta") >= 2.359, "ADXΔ ≥ 2.359", np.where(n("entry_adx_delta").notna(), "ADXΔ < 2.359", "unscored")),
        "RSI tercile": q(n("entry_rsi")),
        "RSI jump tercile": q(n("d_rsi_jump")),
        "pre-spike RSI tercile": q(n("entry_rsi_prev")),
        "ATR% tercile": q(n("entry_atr_pct")),
        "pair trend (EMA13-50 gap) sign": np.where(n("entry_pair_ema20_ema50_gap_pct") > 0, "pair above EMA50", np.where(n("entry_pair_ema20_ema50_gap_pct").notna(), "pair below", "unscored")),
        "BTC RSI tercile": q(n("entry_btc_rsi")),
        "hour UTC": pd.cut(K.t.dt.hour, [-1, 7, 15, 23], labels=["00-07", "08-15", "16-23"]).astype(str),
        "weekend": np.where(K.t.dt.dayofweek >= 5, "weekend", "weekday"),
        "liquidity capped": np.where(C.num(K, "liquidity_capped") == 1, "capped", "not capped"),
    }
    early = K.eg.isin(C.GROUPS[:5]); late = K.eg.isin(["G6 B17-B24", "B25"])
    rows = []
    for nm, lab in subs.items():
        lab = pd.Series(lab, index=K.index).astype(str)
        for v in sorted(lab.unique()):
            m = lab == v
            r = {"dimension": nm, "cohort": v}
            for g in C.GROUPS + ["B25"]:
                r[g] = C.st(K.pct[m & (K.eg == g)])
            ye, yl = K.pct[m & early], K.pct[m & late]
            r["early G1-G5"] = C.st(ye); r["late G6+B25"] = C.st(yl)
            r["Δ late−early"] = yl.mean() - ye.mean() if len(ye) and len(yl) else np.nan
            r["stop% early→late"] = (f"{K.is_stop[m & early].mean() * 100:.0f}→{K.is_stop[m & late].mean() * 100:.0f}"
                                     if len(ye) and len(yl) else "—")
            rows.append(r)
    return pd.DataFrame(rows)


def tape(A, K):
    k = B.load5("BTCUSDT"); k["ts"] = pd.to_datetime(k.t, unit="ms")
    rows = []
    for g in C.GROUPS + ["B25"]:
        m = K.eg == g
        t0, t1 = K.t[m].min().floor("D"), K.t[m].max().ceil("D")
        kk = k[(k.ts >= t0) & (k.ts < t1)]
        r = {"group": g, "span": f"{t0.date()}→{(t1 - pd.Timedelta(days=1)).date()}", "fills": int(m.sum()), "days w/ fills": K.day[m].nunique(),
             "kept": C.st(K.pct[m]), "stops": int(K.is_stop[m].sum()),
             "BTC span ret %": (kk.c.iloc[-1] / kk.c.iloc[0] - 1) * 100 if len(kk) else np.nan,
             "BTC span maxDD %": ((kk.c / kk.c.cummax()) - 1).min() * 100 if len(kk) else np.nan}
        for c in ["btc_r24h", "btc_r7d", "btc_r30d", "btc_offhi_30d", "btc_rv_24h", "btc_rv24_pctl90d", "btc_e50_e200_1d",
                  "br_corr24h_top20", "br_beta_btc_24h", "br_disp_24h", "br_above_ema50_1h", "alt_minus_btc_7d", "br_qv24_x7d",
                  "p_corr24_btc", "p_beta24_btc", "p_qv24_x7d", "f_pair", "entry_pair_volume_24h_usd", "entry_btc_rsi"]:
            r[c] = C.num(K, c)[m].median()
        r["HB share %"] = (K.entry_btc_regime[m] == "HEALTHY_BEAR").mean() * 100
        rows.append(r)
    return pd.DataFrame(rows)


def shifts(n, R):
    ks = RNG.choice(np.arange(3, n - 3), size=R, replace=True)
    return ks


def screen(K, cols, Bt, R=2000, tag="regime"):
    """zone vs rest on master kept: fill Δ + circular-shift null + family-wise · days · LOEO · B25 · BT per seed."""
    K = K.sort_values("t").reset_index(drop=True)
    y = K.pct.values; n = len(y)
    ks = shifts(n, R); Y = np.stack([np.roll(y, k) for k in ks])
    masks = []
    for c in cols:
        x = C.num(K, c).values.astype(float)
        if np.isfinite(x).sum() < 30:
            continue
        for lab, kind, thr, z in C.zones(x, c):
            s = np.isfinite(x)
            if z.sum() < 8 or (s & ~z).sum() < 8:
                continue
            masks.append((c, lab, kind, thr, z, s))
    Zm = np.stack([m[4] for m in masks]).astype(float); Rm = np.stack([m[5] & ~m[4] for m in masks]).astype(float)
    obs = Zm @ y / Zm.sum(1) - Rm @ y / Rm.sum(1)
    null = Y @ Zm.T / Zm.sum(1) - Y @ Rm.T / Rm.sum(1)
    p = ((np.abs(null) >= np.abs(obs)).sum(0) + 1) / (R + 1)
    fw = np.abs(null).max(1); pfw = ((fw[:, None] >= np.abs(obs)[None, :]).sum(0) + 1) / (R + 1)
    rows = []
    B25 = K_B25
    grp = K.eg.values
    for i, (c, lab, kind, thr, z, s) in enumerate(masks):
        x = C.num(K, c).values.astype(float)
        yz, yr = y[z], y[s & ~z]
        # day units
        dz = K[z].groupby("day").pct.mean(); dr = K[s & ~z].groupby("day").pct.mean()
        # LOEO sign agreement (fixed threshold, held-out group sign)
        agree = tot = 0
        for g in C.GROUPS:
            te = grp == g
            a, b = y[te & z], y[te & s & ~z]
            if len(a) >= 2 and len(b) >= 2:
                tot += 1; agree += int(np.sign(a.mean() - b.mean()) == np.sign(obs[i]))
        # leave-one-era-out Δ (drop each group, recompute Δ) → min/max
        loo = []
        for g in C.GROUPS:
            keep = grp != g
            a, b = y[keep & z], y[keep & s & ~z]
            if len(a) >= 5 and len(b) >= 5:
                loo.append(a.mean() - b.mean())
        # concentration of the zone's loss
        ds, dk = C.loss_share(yz, K.day.values[z]); ps, pk = C.loss_share(yz, K.pair.values[z])
        # B25 (provisional)
        xb = C.num(B25, c).values.astype(float); zb, sb = C.apply_zone(xb, kind, thr)
        # BT per seed
        bt = {}
        if c in Bt.columns:
            for sd in (1, 2, 3):
                bb = Bt[Bt.seed == sd]; xx = C.num(bb, c).values.astype(float); zz, ss = C.apply_zone(xx, kind, thr)
                yy = bb.pct.values
                bt[f"BT s{sd} zone"] = C.st(yy[zz]); bt[f"BT s{sd} Δ"] = (yy[zz].mean() - yy[ss & ~zz].mean()) if zz.sum() >= 5 and (ss & ~zz).sum() >= 5 else np.nan
        rows.append({"variable": c, "zone": lab, "kind": kind, "thr": thr, "zone_stat": C.st(yz), "rest_stat": C.st(yr), "delta": obs[i],
                     "p_shift": p[i], "p_fw": pfw[i], "zone_days": dz.size, "rest_days": dr.size,
                     "zone_daymean": dz.mean(), "rest_daymean": dr.mean(), "day_delta": dz.mean() - dr.mean(),
                     "zone_neg_day_%": (dz < 0).mean() * 100, "rest_neg_day_%": (dr < 0).mean() * 100,
                     "loeo_sign_agree": f"{agree}/{tot}", "leave1era_delta_min": min(loo) if loo else np.nan,
                     "leave1era_delta_max": max(loo) if loo else np.nan,
                     "zone_top_day_loss_share": ds, "zone_top_pair_loss_share": ps, "zone_top_pair": pk,
                     "zone_share_early": z[np.isin(grp, C.GROUPS[:5])].mean() * 100 / max(s[np.isin(grp, C.GROUPS[:5])].mean(), 1e-9),
                     "zone_share_G6": z[grp == "G6 B17-B24"].mean() * 100 / max(s[grp == "G6 B17-B24"].mean(), 1e-9),
                     "B25_zone": C.st(B25.pct.values[zb]), "B25_rest": C.st(B25.pct.values[sb & ~zb]), **bt})
    return pd.DataFrame(rows)


def day_level(K, cols):
    Dd = K.groupby("day").agg(y=("pct", "mean"), n=("pct", "size"), grp=("eg", "first"))
    rows = []
    for c in cols:
        v = C.num(K, c).groupby(K.day).mean()
        d = Dd.join(v.rename("x")).dropna()
        if len(d) < 15 or d.x.nunique() < 3:
            continue
        rx, ry = d.x.rank().values, d.y.rank().values
        rho = np.corrcoef(rx, ry)[0, 1]
        nul = np.array([np.corrcoef(rx, np.roll(ry, k))[0, 1] for k in range(2, len(d) - 2)])
        p = ((np.abs(nul) >= abs(rho)).sum() + 1) / (len(nul) + 1)
        dd = d.copy(); dd["yd"] = dd.y - dd.groupby("grp").y.transform("mean"); dd["xd"] = dd.x - dd.groupby("grp").x.transform("mean")
        hi = d.x > d.x.median()
        rows.append({"variable": c, "days": len(d), "spearman_days": rho, "p_rotation": p,
                     "spearman_within_era": np.corrcoef(dd.xd.rank(), dd.yd.rank())[0, 1],
                     "hi_days_mean": d.y[hi].mean(), "lo_days_mean": d.y[~hi].mean(),
                     "hi_days_neg_%": (d.y[hi] < 0).mean() * 100, "lo_days_neg_%": (d.y[~hi] < 0).mean() * 100})
    return pd.DataFrame(rows)


PRED = [  # (id, variable, predicted better side: 'low' / 'high' / 'false', cut kind)
    ("P1", "btc_r1h", "low"), ("P1", "btc_r4h", "low"), ("P1", "btc_r15m", "low"), ("P1", "btc_r30m", "low"),
    ("P2", "btc_r24h", "low"), ("P2", "btc_r7d", "low"), ("P2", "btc_c_e20_1h", "low"), ("P2", "btc_c_e20_4h", "low"),
    ("P3", "btc_offhi_30d", "high"), ("P3", "btc_offhi_7d", "high"), ("P3", "entry_btc_trend_gap_pct", "high"),
    ("P4", "br_corr24h_top20", "high"), ("P4", "br_beta_btc_24h", "high"), ("P4", "p_corr24_btc", "high"), ("P4", "p_beta24_btc", "high"),
    ("P5", "br_disp_1h", "low"), ("P5", "br_disp_24h", "low"),
    ("P6", "br_above_ema50_1h", "low"), ("P6", "br_above_ema20_5m", "low"), ("P6", "br_adv_1h", "low"), ("P6", "br_adv_24h", "low"),
    ("P6", "entry_bear_pct", "high"),
    ("P7", "alt_minus_btc_24h", "low"), ("P7", "alt_minus_btc_7d", "low"), ("P7", "ethbtc_r24h", "low"), ("P7", "ethbtc_r7d", "low"),
    ("P8", "btc_rv_24h", "low"), ("P8", "btc_rv24_pctl90d", "low"), ("P8", "btc_rv_7d", "low"), ("P8", "entry_btc_atr_pct", "low"),
    ("P9", "weekend", "high"), ("P9", "br_qv24_x7d", "low"), ("P9", "btc_qv24_x7d", "low"),
    ("P10", "f_pair", "high"), ("P10", "f_pair3", "high"), ("P10", "br_funding_med", "high"),
    ("P11", "p_qv24_x7d", "low"), ("P11", "p_r1h", "low"), ("P11", "p_r4h", "low"), ("P11", "p_r24h", "low"), ("P11", "p_offhi_7d", "low"),
    ("P12", "s_fb_run", "high"), ("P12", "s_hi15_dist", "low"), ("P12", "s_sec_in_bar", "low"),
    ("P13", "d_log_mcap", "low"), ("P14", "c_regime_HEALTHY_BEAR", "false"), ("P14", "hb_gap_zone", "false"),
]


def score_prereg(K, B25, Bt, pkey="P"):
    rows = []
    for pid, c, better in PRED:
        x = C.num(K, c).values.astype(float); y = K.pct.values; s = np.isfinite(x)
        if s.sum() < 30:
            rows.append({"pred": pid, "variable": c, "better": better, "N_scored": int(s.sum()), "verdict": "UNTESTABLE (coverage)"})
            continue
        if better == "false":
            hi = s & (x == 1)
        elif c == "weekend":
            hi = s & (x == 1)
        else:
            v = x[s]; cut = 0.0 if (any(k in c for k in C.NATURAL_ZERO) and 0.1 < (v > 0).mean() < 0.9) else float(np.median(v))
            hi = s & (x > cut)
        lo = s & ~hi
        d = y[hi].mean() - y[lo].mean()
        want = -1 if better in ("low", "false") else 1
        # day units
        dh = K[hi].groupby("day").pct.mean(); dl = K[lo].groupby("day").pct.mean()
        dd = dh.mean() - dl.mean()
        # circular-shift null on this split
        ks = shifts(len(y), 2000); null = np.array([np.roll(y, k)[hi].mean() - np.roll(y, k)[lo].mean() for k in ks])
        p = ((np.abs(null) >= abs(d)).sum() + 1) / 2001
        xb = C.num(B25, c).values.astype(float); sb = np.isfinite(xb)
        if better in ("false",) or c == "weekend":
            hb = sb & (xb == 1)
        else:
            hb = sb & (xb > (cut if better != "false" else 0))
        bd = B25.pct.values[hb].mean() - B25.pct.values[sb & ~hb].mean() if hb.sum() and (sb & ~hb).sum() else np.nan
        btd = []
        if c in Bt.columns:
            for sd in (1, 2, 3):
                bb = Bt[Bt.seed == sd]; xx = C.num(bb, c).values.astype(float); ss = np.isfinite(xx)
                hh = ss & ((xx == 1) if (better == "false" or c == "weekend") else (xx > cut))
                btd.append(bb.pct.values[hh].mean() - bb.pct.values[ss & ~hh].mean() if hh.sum() >= 10 and (ss & ~hh).sum() >= 10 else np.nan)
        same = np.sign(d) == want; same_day = np.sign(dd) == want
        verdict = ("CONFIRMED" if same and same_day and p < 0.10 else "RIGHT SIGN (n.s.)" if same and same_day
                   else "MIXED (fill vs day disagree)" if same != same_day else "REFUTED (opposite sign)")
        rows.append({"pred": pid, "variable": c, "better": better, "N_scored": int(s.sum()),
                     "cut": "==1" if better == "false" or c == "weekend" else round(cut, 5),
                     "high_side": C.st(y[hi]), "low_side": C.st(y[lo]), "delta_high_minus_low": d, "p_shift": p,
                     "day_delta": dd, "days_high/low": f"{dh.size}/{dl.size}", "B25_delta (prov.)": bd,
                     "BT_delta s1/s2/s3": "/".join("—" if not np.isfinite(v) else f"{v:+.3f}" for v in btd) if btd else "—",
                     "verdict": verdict})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    A = C.load(with_blocked=True)
    K = A[A.blk == "MASTER"].copy(); K_B25 = A[A.blk == "B25"].copy()
    try:
        Bt = C.load_bt()
    except FileNotFoundError:
        Bt = pd.DataFrame(columns=["seed", "pct"]); P("⚠ BT regime file missing — BT columns empty")
    market_own, mstamp, pair_own, pstamp = C.columns(A[A.blk != "MASTER_BLK"])
    be = C.breakeven(K.pct.values)
    P(f"cohort: master kept {len(K)} · {K.day.nunique()} days · B25 {len(K_B25)} · blocked {int((A.blk == 'MASTER_BLK').sum())} · BT {len(Bt)}")
    P(f"breakeven WR (master kept, today's stack) = {be:.1f} % · avg win {K.pct[K.pct > 0].mean():+.3f} / loss {K.pct[K.pct <= 0].mean():+.3f}")
    P(f"vars: market rebuilt {len(market_own)} · market stamps {len(mstamp)} · pair rebuilt {len(pair_own)} · pair stamps {len(pstamp)}")
    allc = market_own + mstamp + pair_own + pstamp
    coverage(A, allc).to_csv(PFX + "_coverage.csv", index=False)
    U = uniform(pd.concat([K, K_B25])); U.to_csv(PFX + "_uniform.csv", index=False)
    Tp = tape(A, pd.concat([K, K_B25])); Tp.to_csv(PFX + "_tape.csv", index=False)
    S = screen(K, market_own + mstamp, Bt); S.to_csv(PFX + "_regime_screen.csv", index=False)
    Dl = day_level(K, market_own + mstamp + pair_own + pstamp); Dl.to_csv(PFX + "_regime_day.csv", index=False)
    Sp = screen(K, pair_own + pstamp, Bt, tag="entry"); Sp.to_csv(PFX + "_entry_screen_all.csv", index=False)
    Ps = score_prereg(K, K_B25, Bt); Ps.to_csv(PFX + "_prereg_score.csv", index=False)
    P(f"regime tests {len(S)} · p<0.05 {int((S.p_shift < 0.05).sum())} (chance ≈ {0.05 * len(S):.0f}) · p_fw<0.05 {int((S.p_fw < 0.05).sum())}")
    P(f"entry tests (all fills) {len(Sp)} · p<0.05 {int((Sp.p_shift < 0.05).sum())} (chance ≈ {0.05 * len(Sp):.0f}) · p_fw<0.05 {int((Sp.p_fw < 0.05).sum())}")
    P(f"day-level vars {len(Dl)} · p_rotation<0.05 {int((Dl.p_rotation < 0.05).sum())} (chance ≈ {0.05 * len(Dl):.0f})")
    open(PFX + "_analyse.txt", "w").write("\n".join(OUT) + "\n")
