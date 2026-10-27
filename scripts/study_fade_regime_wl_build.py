#!/usr/bin/env python3
"""FADE REGIME W/L study 2026-10-26 — per-fill regime + fade-specific features, CLOSED data only (no network; run the _fetch step first).

Reuses scripts/study_ml_regime_wl_build.build() unchanged (80 BTC / breadth / alt / time variables; closed 5m bars; top-50 list dated D − 1)
with the caches extended by reports/FADE_REGIME_WL_2026-10-26_fetch (alts to 10-26 14:00) and the 10-23…10-25 top-50 lists of the
scout_ml_regime_lines cache (lists 10-12…10-22 do not exist anywhere → fills of 10-12…10-23 use the 10-11 list, flagged `uni_stale`).
Adds, per fill (all known at the fill time t):
  BTC   btc_r15m / btc_r30m (closed 5m), btc_r5m_fb (BTC forming-5m-bar move to the last closed 1m bar is NOT available → omitted)
  PAIR  (5m closed bars before t) p_r1h / p_r4h / p_r24h / p_r7d · p_offhi_7d / p_offlo_7d · p_rv24h · p_qv24_x7d (volume inflation)
        · p_corr24_btc / p_beta24_btc (pair 5m log-returns vs BTC over the 288 closed bars) · p_prevbar_ret (last closed 5m bar) ·
        p_ema50_gap_1h (pair close vs 1h EMA50, closed 1h)
  SPIKE (1m closed bars before t; fetched) s_fb_run = entry_price / open of the forming 5m bar − 1 · s_hi15_dist = max high of the last
        15 closed 1m bars / entry − 1 (≥ 0; how far the fill sits below the spike high) · s_run15 = entry / close 15 min before − 1 ·
        s_sec_in_bar = seconds since the forming 5m bar opened · s_vol5_x60 = volume of the last 5 closed 1m bars / 60-min mean × 5
  FUND  f_pair = last funding settlement ≤ t · f_pair3 = mean of the last 3 settlements · f_pair_neg = f_pair < 0
Outputs: reports/FADE_REGIME_WL_2026-10-26_fill_regime.csv (master kept + blocked + B25), _bt_regime.csv.gz (yr6 fades; local, gzipped)
    venv/bin/python -W ignore scripts/study_fade_regime_wl_build.py [--no-bt]
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_regime_wl_build as B      # noqa: E402
import study_fade_regime_wl_data as D     # noqa: E402

R = D.R
FETCH = os.path.join(R, "reports/FADE_REGIME_WL_2026-10-26_fetch")
PFX = os.path.join(R, "reports/FADE_REGIME_WL_2026-10-26")
B.SRC += [os.path.join(R, "reports/backtest_cache/scout_ml_regime_lines/k5m"), os.path.join(R, "reports/ML_MATCHED_SHADOW_2026-10-26_fetch/k5m"),
          os.path.join(FETCH, "k5m")]
B.TOP50 = B.TOP50 + [os.path.join(R, "reports/backtest_cache/scout_ml_regime_lines/universe_daily_top50.csv")]
BAR, HOUR, DAY = B.BAR, B.HOUR, B.DAY
_K = {}


def k5(p):
    if p not in _K:
        _K[p] = B.load5(p)
    return _K[p]


def pair_feats(A):
    btc = k5("BTCUSDT"); bt = btc.t.values; bc = btc.c.values.astype(float)
    blr = np.r_[0.0, np.diff(np.log(bc))]
    rows = []
    for _, r in A.iterrows():
        T = int(r["T"]); o = {}
        jb = np.searchsorted(bt, T - BAR, side="right") - 1
        o["btc_r15m"] = (bc[jb] / bc[jb - 3] - 1) * 100
        o["btc_r30m"] = (bc[jb] / bc[jb - 6] - 1) * 100
        k = k5(r.pair)
        if k is not None and len(k) > 300:
            t = k.t.values; c = k.c.values.astype(float); h = k.h.values.astype(float); l = k.l.values.astype(float)
            qv = k.qv.values.astype(float)
            j = np.searchsorted(t, T - BAR, side="right") - 1
            if j >= 300 and T - (t[j] + BAR) <= 10 * 60000:
                ret = lambda n: (c[j] / c[j - n] - 1) * 100 if j - n >= 0 else np.nan
                o.update(p_r1h=ret(12), p_r4h=ret(48), p_r24h=ret(288), p_r7d=ret(2016) if j >= 2016 else np.nan,
                         p_prevbar_ret=(c[j] / c[j - 1] - 1) * 100)
                lo7 = max(0, j - 2015)
                o["p_offhi_7d"] = (c[j] / h[lo7:j + 1].max() - 1) * 100
                o["p_offlo_7d"] = (c[j] / l[lo7:j + 1].min() - 1) * 100
                lr = np.diff(np.log(c[j - 288:j + 1]))
                o["p_rv24h"] = lr.std() * 100
                q24 = qv[j - 287:j + 1].sum(); q7 = qv[max(0, j - 2015):j + 1].sum() / (min(j + 1, 2016) / 288)
                o["p_qv24_x7d"] = q24 / q7 if q7 > 0 else np.nan
                # pair vs BTC on matched closed 5m bars
                m = pd.merge(pd.DataFrame({"t": t[j - 288:j + 1], "c": c[j - 288:j + 1]}),
                             pd.DataFrame({"t": bt[jb - 288:jb + 1], "b": bc[jb - 288:jb + 1]}), on="t")
                if len(m) >= 250:
                    x = np.diff(np.log(m.c.values)); y = np.diff(np.log(m.b.values))
                    if np.var(y) > 0 and np.var(x) > 0:
                        o["p_corr24_btc"] = np.corrcoef(x, y)[0, 1]; o["p_beta24_btc"] = np.cov(x, y)[0, 1] / np.var(y)
                hr = B.resample(k[k.t <= t[j]], HOUR)
                jh = np.searchsorted(hr.t.values, T - HOUR, side="right") - 1
                if jh >= 50:
                    e50 = B.ema(hr.c.values.astype(float), 50)
                    o["p_ema50_gap_1h"] = (hr.c.values[jh] / e50[jh] - 1) * 100
        # 1m spike geometry (closed 1m bars only)
        f1 = os.path.join(FETCH, "k1m", f"{r.pair}_{T}.csv")
        ep = float(r.entry_price) if pd.notna(r.get("entry_price", np.nan)) else np.nan
        if os.path.exists(f1) and np.isfinite(ep) and ep > 0:
            m1 = pd.read_csv(f1)
            m1 = m1[m1.t + 60000 <= T]
            fb_open_t = T - (T % BAR)
            if len(m1) >= 30:
                fb = m1[m1.t >= fb_open_t]
                o["s_fb_run"] = (ep / (fb.o.iloc[0] if len(fb) else m1.c.iloc[-1]) - 1) * 100
                last15 = m1[m1.t >= T - 15 * 60000 - (T % 60000)]
                o["s_hi15_dist"] = (last15.h.max() / ep - 1) * 100 if len(last15) else np.nan
                c15 = m1[m1.t <= T - 15 * 60000]
                o["s_run15"] = (ep / c15.c.iloc[-1] - 1) * 100 if len(c15) else np.nan
                v = m1.qv.astype(float).values
                o["s_vol5_x60"] = v[-5:].sum() / (v.mean() * 5) if v.mean() > 0 else np.nan
            o["s_sec_in_bar"] = (T - fb_open_t) / 1000
        ff = os.path.join(FETCH, "funding", r.pair + ".csv")
        if os.path.exists(ff) and os.path.getsize(ff) > 10:
            fd = pd.read_csv(ff).sort_values("t")
            jf = np.searchsorted(fd.t.values, T, side="right") - 1
            if jf >= 0 and T - fd.t.values[jf] <= 9 * HOUR:
                o["f_pair"] = fd.rate.values[jf] * 100
                o["f_pair3"] = fd.rate.values[max(0, jf - 2):jf + 1].mean() * 100
                o["f_pair_neg"] = float(fd.rate.values[jf] < 0)
        rows.append(o)
    return pd.DataFrame(rows)


def main():
    A = D.load(with_bt=False)
    A["T"] = A.t.values.astype("datetime64[ms]").astype(np.int64)
    F = B.build(A["T"].values)
    P = pair_feats(A)
    days = sorted(pd.concat([pd.read_csv(f) for f in B.TOP50]).date_ms.unique())
    tday = (A["T"].values // DAY) * DAY
    A["uni_stale"] = [float((tday[i] - DAY) not in set(days)) for i in range(len(A))]
    out = pd.concat([A[["key", "blk", "pair", "opened_at", "uni_stale"]].reset_index(drop=True), F, P], axis=1)
    out.to_csv(PFX + "_fill_regime.csv", index=False)
    print("fills", out.shape, flush=True)
    if "--no-bt" not in sys.argv:
        Bt = D.load(with_bt=True); Bt = Bt[Bt.blk == "BT"].copy()
        Bt["T"] = Bt.t.values.astype("datetime64[ms]").astype(np.int64)
        u = Bt[["T"]].drop_duplicates().sort_values("T")
        G = B.build(u["T"].values); G.insert(0, "T", u["T"].values)
        Bt = Bt.merge(G, on="T", how="left")
        Pb = pair_feats(Bt.reset_index(drop=True))
        ob = pd.concat([Bt[["key", "blk", "pair", "opened_at"]].reset_index(drop=True), Bt[G.columns].reset_index(drop=True), Pb], axis=1)
        ob.to_csv(PFX + "_bt_regime.csv.gz", index=False, compression="gzip")
        print("bt", ob.shape, flush=True)


if __name__ == "__main__":
    main()
