#!/usr/bin/env python3
"""ML REGIME W/L study 2026-10-26 — per-fill MARKET-REGIME features from CLOSED klines only (cache-only, no network).

For every MOMENTUM LONG fill (master KEEP/TRAD + B25 provisional via study_ml_wl_b25prep_data.load_all, and — refute-only —
the yr6 replay ML fills) compute, at the entry timestamp t, using only 5m bars whose close time ≤ t:
  BTC  : returns 1h/4h/24h/3d/7d/30d · distance from 24h/7d/30d high and low · efficiency 6h/24h/72h (engine eff72 formula)
         · above72 (engine formula) · realised vol 1h/24h/7d + rv24h percentile vs prior 90 d + rv24/rv7 ratio · RSI14 / ADX14 /
         EMA20 slope / close-vs-EMA20 / EMA20-vs-EMA50 / EMA50-vs-EMA200 on 5m / 1h / 4h / 1d closed bars · BTC qvol 24h / 7d mean
         · variance ratio (15m) and lag-1 autocorrelation of 5m returns over the trailing 3 d / 7 d (rolling, not day-anchored)
  BREADTH (top-50 of the list DATED D − 1 = ranked on day D − 2 quote volume, full USDT-perp universe — universe_daily_top50.csv /
         backtest_phase1_universe.py convention — ex BTC / stables; a missing date falls back to the latest list ≤ D − 1, which made the
         B25 values stale — DL 382): share above 5m EMA20 · share above 1h EMA50 · advance share
         1h / 24h · median return 1h / 24h / 7d · cross-sectional dispersion 1h / 24h · mean pairwise corr of 5m returns over 24 h
         (top-20) · median alt beta to BTC (24 h) · equal-weight alt index minus BTC 24h / 7d · ETHBTC 24h / 7d · aggregate top-50
         quote volume 24h / prior-7d mean · median top-50 funding (latest settlement ≤ t, and 3-day mean; cache ends 2026-10-02)
  TIME : hour, weekday, weekend flag.
Outputs reports/ML_REGIME_WL_2026-10-26_fill_regime.csv (master + B25) and _replay_regime.csv (yr6 ML fills).
    venv/bin/python -W ignore scripts/study_ml_regime_wl_build.py [--no-replay]
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(ROOT, "reports")
PFX = os.path.join(OUT, "ML_REGIME_WL_2026-10-26")
SRC = [os.path.join(ROOT, "reports/backtest_cache_2025/k5m_full"), os.path.join(ROOT, "reports/backtest_cache/k5m_full"),
       os.path.join(ROOT, "reports/backtest_cache/k5m"), os.path.join(ROOT, "reports/ml_regime/fetch/k5m"),
       os.path.join(ROOT, "reports/PREP_B25_ML_WL_fetch/k5m_full"), os.path.join(ROOT, "reports/ML_EXIT_DEEP_2026-10-26_fetch/k5m")]
TOP50 = [os.path.join(ROOT, "reports/backtest_cache_2025/universe_daily_top50.csv"),
         os.path.join(ROOT, "reports/backtest_cache/universe_daily_top50.csv")]
FUND = os.path.join(ROOT, "reports/backtest_cache/funding")
BAR, HOUR, DAY = 300_000, 3_600_000, 86_400_000
STABLE = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "DAIUSDT", "USDEUSDT", "XUSDUSDT", "BTCDOMUSDT"}


def load5(pair):
    parts = []
    for s in SRC:
        f = os.path.join(s, pair + ".csv")
        if os.path.exists(f):
            try:
                x = pd.read_csv(f, usecols=[0, 2, 3, 4, 6]); x.columns = ["t", "h", "l", "c", "qv"]; parts.append(x)
            except Exception:
                pass
    if not parts:
        return None
    k = pd.concat(parts).dropna(subset=["c"]).drop_duplicates("t", keep="last").sort_values("t").reset_index(drop=True)
    k["t"] = k.t.astype(np.int64)
    return k


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().values
    ad = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 - 100 / (1 + au / ad)


def adx(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    uh, dl = h - np.r_[h[0], h[:-1]], np.r_[l[0], l[:-1]] - l
    pdm = np.where((uh > dl) & (uh > 0), uh, 0.0); ndm = np.where((dl > uh) & (dl > 0), dl, 0.0)
    w = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values
    atr = w(tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, ndi = 100 * w(pdm) / atr, 100 * w(ndm) / atr
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
    return w(np.nan_to_num(dx)), pdi, ndi


def resample(k, ms):
    g = (k.t // ms) * ms
    r = k.groupby(g).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), qv=("qv", "sum"), n=("c", "size")).reset_index()
    r.columns = ["t", "h", "l", "c", "qv", "n"]
    return r


def tf_feats(k, ms, tag, T):
    """indicators on closed bars of timeframe ms, sampled at the last bar with open + ms ≤ t."""
    r = resample(k, ms) if ms > BAR else k.copy()
    c, h, l = r.c.values.astype(float), r.h.values.astype(float), r.l.values.astype(float)
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    rs = rsi(c); ax, pdi, ndi = adx(h, l, c)
    j = np.searchsorted(r.t.values, T - ms, side="right") - 1
    ok = j >= 3
    jj = np.clip(j, 3, None)
    o = {}
    o[f"btc_rsi_{tag}"] = np.where(ok, rs[jj], np.nan)
    o[f"btc_adx_{tag}"] = np.where(ok, ax[jj], np.nan)
    o[f"btc_adxchg_{tag}"] = np.where(ok, ax[jj] - ax[jj - 3], np.nan)
    o[f"btc_di_{tag}"] = np.where(ok, pdi[jj] - ndi[jj], np.nan)
    o[f"btc_e20slp_{tag}"] = np.where(ok, (e20[jj] / e20[jj - 3] - 1) * 100, np.nan)
    o[f"btc_c_e20_{tag}"] = np.where(ok, (c[jj] / e20[jj] - 1) * 100, np.nan)
    o[f"btc_e20_e50_{tag}"] = np.where(ok, (e20[jj] / e50[jj] - 1) * 100, np.nan)
    o[f"btc_e50_e200_{tag}"] = np.where(ok, (e50[jj] / e200[jj] - 1) * 100, np.nan)
    return o


def btc_feats(T):
    k = load5("BTCUSDT")
    t = k.t.values; c = k.c.values.astype(float); h = k.h.values.astype(float); l = k.l.values.astype(float)
    qv = k.qv.values.astype(float)
    lr = np.r_[0.0, np.diff(np.log(c))]
    j = np.searchsorted(t, T - BAR, side="right") - 1          # last CLOSED 5m bar
    o = {"btc_bar_lag_min": (T - (t[j] + BAR)) / 60000}
    ret = lambda n: (c[j] / c[np.clip(j - n, 0, None)] - 1) * 100
    for nm, n in [("1h", 12), ("4h", 48), ("24h", 288), ("3d", 864), ("7d", 2016), ("30d", 8640)]:
        o[f"btc_r{nm}"] = ret(n)
    hmax = pd.Series(h); lmin = pd.Series(l)
    for nm, n in [("24h", 288), ("7d", 2016), ("30d", 8640)]:
        hh = hmax.rolling(n, min_periods=n // 2).max().values; ll = lmin.rolling(n, min_periods=n // 2).min().values
        o[f"btc_offhi_{nm}"] = (c[j] / hh[j] - 1) * 100
        o[f"btc_offlo_{nm}"] = (c[j] / ll[j] - 1) * 100
    absd = pd.Series(np.abs(np.r_[0.0, np.diff(c)]))
    for nm, n in [("6h", 72), ("24h", 288), ("72h", 864)]:
        s = absd.rolling(n - 1).sum().values
        o[f"btc_eff_{nm}"] = np.abs(c[j] - c[np.clip(j - n + 1, 0, None)]) / s[j]
    e20 = ema(c, 20); ab = pd.Series((c > e20).astype(float)).rolling(864).mean().values * 100
    o["btc_above72"] = ab[j]
    lrs = pd.Series(lr)
    for nm, n in [("1h", 12), ("24h", 288), ("7d", 2016)]:
        o[f"btc_rv_{nm}"] = lrs.rolling(n).std().values[j] * 100
    rv24 = lrs.rolling(288).std().values * 100
    # percentile of rv24h vs the prior 90 days sampled hourly
    pct = np.full(len(T), np.nan)
    for i, jj in enumerate(j):
        hist = rv24[max(0, jj - 90 * 288):jj:12]
        hist = hist[np.isfinite(hist)]
        if len(hist) > 200:
            pct[i] = (hist < rv24[jj]).mean() * 100
    o["btc_rv24_pctl90d"] = pct
    o["btc_rv24_rv7"] = o["btc_rv_24h"] / o["btc_rv_7d"]
    q = pd.Series(qv)
    o["btc_qv24_x7d"] = q.rolling(288).sum().values[j] / (q.rolling(2016).sum().values[j] / 7)
    # persistence: 15m variance ratio and 5m lag-1 autocorrelation, trailing 3d / 7d
    for nm, n in [("3d", 864), ("7d", 2016)]:
        vr, ac = np.full(len(T), np.nan), np.full(len(T), np.nan)
        for i, jj in enumerate(j):
            x = lr[max(1, jj - n + 1):jj + 1]
            if len(x) < n * 0.9 or np.var(x) == 0:
                continue
            s3 = np.convolve(x, np.ones(3), "valid")
            vr[i] = np.var(s3) / (3 * np.var(x)); ac[i] = np.corrcoef(x[1:], x[:-1])[0, 1]
        o[f"btc_vr15_{nm}"] = vr; o[f"btc_ac5_{nm}"] = ac
    for ms, tag in [(BAR, "5m"), (HOUR, "1h"), (4 * HOUR, "4h"), (DAY, "1d")]:
        o.update(tf_feats(k, ms, tag, T))
    return pd.DataFrame(o)


def universe():
    u = pd.concat([pd.read_csv(f) for f in TOP50 if os.path.exists(f)]).drop_duplicates(["date_ms", "pair"])
    u = u[~u.pair.isin(STABLE) & (u.pair != "BTCUSDT") & u.pair.str.match(r"^[A-Z0-9]+USDT$")]
    return {d: g.sort_values("rank").pair.tolist() for d, g in u.groupby("date_ms")}


def breadth_feats(T):
    U = universe(); days = np.array(sorted(U))
    tday = (T // DAY) * DAY
    # the list DATED day − 1 (= ranked on day − 2 quote volume); fall back to the latest list ≤ day − 1 (stale after the cache end — DL 382)
    di = np.searchsorted(days, tday - DAY, side="right") - 1
    lists = [U[days[i]] if i >= 0 else [] for i in di]
    need = {}
    for i, L in enumerate(lists):
        for r_, p in enumerate(L):
            need.setdefault(p, []).append((i, r_))
    btc = load5("BTCUSDT"); bt = btc.t.values; blr = np.r_[0.0, np.diff(np.log(btc.c.values.astype(float)))]
    n = len(T)
    cols = ["ab20", "ab50h", "r1h", "r24h", "r7d", "qv24", "qv7"]
    M = {c: np.full((n, 50), np.nan) for c in cols}
    R24 = np.full((n, 20, 288), np.nan)
    eth = None
    for p, idx in need.items():
        k = load5(p)
        if k is None or len(k) < 300:
            continue
        t = k.t.values; c = k.c.values.astype(float); qv = pd.Series(k.qv.values.astype(float))
        e20 = ema(c, 20)
        hr = resample(k, HOUR); e50h = ema(hr.c.values.astype(float), 50)
        lr = np.r_[0.0, np.diff(np.log(c))]
        q24 = qv.rolling(288).sum().values; q7 = qv.rolling(2016).sum().values / 7
        for i, r_ in idx:
            if r_ >= 50:
                continue
            j = np.searchsorted(t, T[i] - BAR, side="right") - 1
            if j < 2016 or (T[i] - (t[j] + BAR)) > 30 * 60000:      # stale / too short history → unscored
                continue
            M["ab20"][i, r_] = float(c[j] > e20[j])
            jh = np.searchsorted(hr.t.values, T[i] - HOUR, side="right") - 1
            if jh >= 50:
                M["ab50h"][i, r_] = float(hr.c.values[jh] > e50h[jh])
            M["r1h"][i, r_] = (c[j] / c[j - 12] - 1) * 100
            M["r24h"][i, r_] = (c[j] / c[j - 288] - 1) * 100
            M["r7d"][i, r_] = (c[j] / c[j - 2016] - 1) * 100
            M["qv24"][i, r_] = q24[j]; M["qv7"][i, r_] = q7[j]
            if r_ < 20 and t[j - 287] == t[j] - 287 * BAR:
                R24[i, r_] = lr[j - 287:j + 1]
    o = {}
    o["br_n_scored"] = np.isfinite(M["r1h"]).sum(1)
    with np.errstate(invalid="ignore", all="ignore"):
        o["br_above_ema20_5m"] = np.nanmean(M["ab20"], 1) * 100
        o["br_above_ema50_1h"] = np.nanmean(M["ab50h"], 1) * 100
        o["br_adv_1h"] = np.nanmean(M["r1h"] > 0, 1, where=np.isfinite(M["r1h"])) * 100
        o["br_adv_24h"] = np.nanmean(M["r24h"] > 0, 1, where=np.isfinite(M["r24h"])) * 100
        o["br_med_r1h"] = np.nanmedian(M["r1h"], 1)
        o["br_med_r24h"] = np.nanmedian(M["r24h"], 1)
        o["br_med_r7d"] = np.nanmedian(M["r7d"], 1)
        o["br_disp_1h"] = np.nanstd(M["r1h"], 1)
        o["br_disp_24h"] = np.nanstd(M["r24h"], 1)
        o["br_qv24_x7d"] = np.nansum(M["qv24"], 1) / np.nansum(np.where(np.isfinite(M["qv24"]), M["qv7"], np.nan), 1)
    corr, beta = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        X = R24[i]; okr = np.isfinite(X).all(1)
        if okr.sum() >= 8:
            C = np.corrcoef(X[okr]); iu = np.triu_indices(okr.sum(), 1); corr[i] = np.nanmean(C[iu])
            j = np.searchsorted(bt, T[i] - BAR, side="right") - 1
            b = blr[j - 287:j + 1]
            if np.var(b) > 0:
                beta[i] = np.median([np.cov(x, b)[0, 1] / np.var(b) for x in X[okr]])
    o["br_corr24h_top20"] = corr; o["br_beta_btc_24h"] = beta
    bdf = btc_ret = None
    bc = btc.c.values.astype(float)
    jb = np.searchsorted(bt, T - BAR, side="right") - 1
    for nm, nn in [("24h", 288), ("7d", 2016)]:
        bret = (bc[jb] / bc[jb - nn] - 1) * 100
        o[f"alt_minus_btc_{nm}"] = o[f"br_med_r{nm}"] - bret
    e = load5("ETHUSDT")
    if e is not None:
        m = pd.merge(btc[["t", "c"]], e[["t", "c"]], on="t", suffixes=("_b", "_e"))
        r = (m.c_e / m.c_b).values; tt = m.t.values
        jj = np.searchsorted(tt, T - BAR, side="right") - 1
        o["ethbtc_r24h"] = (r[jj] / r[jj - 288] - 1) * 100
        o["ethbtc_r7d"] = (r[jj] / r[jj - 2016] - 1) * 100
    # funding (alts; cache ends ~2026-10-02 → unscored after)
    F = np.full((n, 50), np.nan); F3 = np.full((n, 50), np.nan)
    for p, idx in need.items():
        f = os.path.join(FUND, p + ".csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f).sort_values("t"); ft = d.t.values.astype(np.int64); fr = d.rate.values.astype(float)
        for i, r_ in idx:
            if r_ >= 50:
                continue
            j = np.searchsorted(ft, T[i], side="right") - 1
            if j < 0 or T[i] - ft[j] > 9 * HOUR:
                continue
            F[i, r_] = fr[j]
            lo = np.searchsorted(ft, T[i] - 3 * DAY, side="right")
            F3[i, r_] = fr[lo:j + 1].mean()
    with np.errstate(all="ignore"):
        nf = np.isfinite(F).sum(1)
        o["br_funding_med"] = np.where(nf >= 20, np.nanmedian(F, 1), np.nan)
        o["br_funding_med3d"] = np.where(nf >= 20, np.nanmedian(F3, 1), np.nan)
        o["br_funding_pos_share"] = np.where(nf >= 20, np.nanmean(F > 0, 1, where=np.isfinite(F)) * 100, np.nan)
    return pd.DataFrame(o)


def build(T):
    T = np.asarray(T, dtype=np.int64)
    a = btc_feats(T); b = breadth_feats(T)
    ts = pd.to_datetime(T, unit="ms")
    tm = pd.DataFrame({"hour": ts.hour, "dow": ts.dayofweek, "weekend": (ts.dayofweek >= 5).astype(float)})
    return pd.concat([a, b, tm], axis=1)


def main():
    import study_ml_wl_b25prep_data as L
    D = L.load_all(with_features=False)
    D = D[~D.is_probe & (D.entry_strategy.astype(str) != "MANUAL")].copy()
    T = (D.ts.values.astype("datetime64[ms]").astype(np.int64))
    F = build(T)
    F.insert(0, "pair", D.pair.values); F.insert(0, "opened_at", D.key_open.values)
    F.to_csv(PFX + "_fill_regime.csv", index=False)
    print("master+B25", F.shape, "→", PFX + "_fill_regime.csv", flush=True)
    if "--no-replay" not in sys.argv:
        from study_yr6wl_momentum_common import load_replay
        R = load_replay(["MOM-long"])
        u = R[["ts"]].drop_duplicates().sort_values("ts")
        Tr = u.ts.values.astype("datetime64[ms]").astype(np.int64)
        G = build(Tr); G.insert(0, "ts", u.ts.astype(str).values)
        G.to_csv(PFX + "_replay_regime.csv", index=False)
        print("replay", G.shape, flush=True)


if __name__ == "__main__":
    main()
