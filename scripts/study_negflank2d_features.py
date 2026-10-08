#!/usr/bin/env python3
"""NEGFLANK 2D study (2026-10-08) — feature builder. Read-only research.

Cohorts come ONLY from scripts/study_ml_b18_common.py (master() incl. B1 flagged, yr5()). Every fill (whole ML sleeve, not just
NEGFLANK — the complement check needs the rest) gets:
  * every entry_* stamp (as-is),
  * k_* features REBUILT from the local kline cache (no network):
      BTC 5m  reports/backtest_cache/k5m_full/BTCUSDT.csv   (Dec-30 → Oct-08)
      BTC 1h  reports/backtest_cache/btc_1h.csv              (Dec-02 → ; gives the 1h/4h EMA + 30d-high warm-up)
      BTC 1d  reports/backtest_cache/k1d/BTCUSDT.csv         (2025-01 → ; 1d EMA warm-up)
      ETH / pairs / alt index: k5m_full/<SYM>.csv
    Value "known at entry" = last 5m bar CLOSED at or before the fill time; higher-TF EMAs use the engine convention
    (ta EMAIndicator = ewm(adjust=False); the forming HTF bar's close = latest price; slope = (ema_now − ema[-4]) / ema[-4] × 100,
    exactly services/trading_engine.py:15361 for the BTC 1h slope).
Output: reports/NEGFLANK_2D_features.pkl  (one DataFrame, column src ∈ {master, yr5}).
"""
import glob, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import study_ml_b18_common as C

K5 = "reports/backtest_cache/k5m_full"
H, D4, DD = 3600_000, 4 * 3600_000, 86400_000
OUT = "reports/NEGFLANK_2D_features.pkl"


EXT = "reports/backtest_cache/negflank2d_ext"      # scripts/study_negflank2d_fetch.py (alt files in k5m_full end Oct-04)


def k5(sym, cols=("open_time", "h", "l", "c", "qvol")):
    d = pd.read_csv(f"{K5}/{sym}.csv", usecols=list(cols))
    if os.path.exists(f"{EXT}/5m/{sym}.csv"):
        d = pd.concat([d, pd.read_csv(f"{EXT}/5m/{sym}.csv", usecols=list(cols))])
    d = d.drop_duplicates("open_time", keep="last").sort_values("open_time")
    d["T"] = d.open_time + 300_000            # bar close time (ms)
    return d.reset_index(drop=True)


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def htf_partial(closes, bucket, span, T, c):
    """closes: completed HTF closes indexed by bucket open (ms). At grid time T with latest price c the forming bar is
    b = floor(T/bucket); returns (ema_now, ema_prev3) like calculate_indicators on [..., b-3, b-2, b-1, b(partial)]."""
    E = ema(closes, span)
    a = 2.0 / (span + 1)
    b = (T // bucket) * bucket
    e1 = E.reindex(b - bucket).values
    e3 = E.reindex(b - 3 * bucket).values
    now = a * c + (1 - a) * e1
    return now, e3


def btc_grid():
    b = k5("BTCUSDT")
    T, c = b["T"].values, b.c.values
    h1 = pd.read_csv("reports/backtest_cache/btc_1h.csv").drop_duplicates("open_time").set_index("open_time").sort_index()[["o", "h", "l", "c"]]
    # extend with complete hours rebuilt from 5m (btc_1h.csv ends Oct-04; k5m_full runs to Oct-08)
    gk = b.groupby(b.open_time // H * H)
    h5 = pd.DataFrame({"o": gk.c.first() * np.nan, "h": gk.h.max(), "l": gk.l.min(), "c": gk.c.last(), "n": gk.size()})
    h5 = h5[h5.n == 12].drop(columns="n")
    h1 = pd.concat([h1[h1.index < h5.index.min()], h5]).sort_index()
    # only bars fully closed before the grid's first use are needed; completed-ness is guaranteed because we index b-1 etc.
    g = pd.DataFrame({"T": T, "c": c})
    e20, e20p3 = htf_partial(h1.c, H, 20, T, c)
    g["k_btc_slope1h"] = (e20 - e20p3) / e20p3 * 100
    e50h, _ = htf_partial(h1.c, H, 50, T, c)
    e200h, _ = htf_partial(h1.c, H, 200, T, c)
    g["k_btc_1h_gap20_50"] = (e20 / e50h - 1) * 100
    g["k_btc_1h_gap20_200"] = (e20 / e200h - 1) * 100
    h4 = h1.c.groupby((h1.index // D4) * D4).last()
    e20_4, e20_4p3 = htf_partial(h4, D4, 20, T, c)
    e50_4, _ = htf_partial(h4, D4, 50, T, c)
    g["k_btc_4h_slope"] = (e20_4 - e20_4p3) / e20_4p3 * 100
    g["k_btc_4h_gap20_50"] = (e20_4 / e50_4 - 1) * 100
    d1 = pd.read_csv("reports/backtest_cache/k1d/BTCUSDT.csv").drop_duplicates("open_time").set_index("open_time").sort_index().c
    gd = b.groupby(b.open_time // DD * DD)
    d5 = gd.c.last()[gd.size() == 288]                               # complete UTC days from 5m (k1d ends Oct-03)
    d1 = pd.concat([d1[d1.index < d5.index.min()], d5]).sort_index()
    e9d, _ = htf_partial(d1, DD, 9, T, c)
    e20d, e20dp3 = htf_partial(d1, DD, 20, T, c)
    g["k_btc_1d_slope"] = (e20d - e20dp3) / e20dp3 * 100
    g["k_btc_1d_gap9_20"] = (e9d / e20d - 1) * 100
    g["k_btc_vs_1d_ema20"] = (c / e20d - 1) * 100
    day = (T - 1) // DD * DD                                         # UTC day the latest price belongs to
    y1, y2 = d1.reindex(day - DD).values, d1.reindex(day - 2 * DD).values
    g["k_btc_day_ret"] = (c / y1 - 1) * 100                          # today's 1d candle so far (vs yesterday's close)
    g["k_btc_prevday_ret"] = (y1 / y2 - 1) * 100                     # previous complete UTC day close/close = entry_btc_1d_ret_pct
    # highs / lows: 30d from the 1h cache (longer warm-up) + current 5m bar
    hh = b.h.values; ll = b.l.values
    s5h = pd.Series(hh); s5l = pd.Series(ll)
    for lab, n in [("24h", 288), ("7d", 2016)]:
        g[f"k_btc_off{lab}_high"] = (c / s5h.rolling(n, min_periods=n).max().values - 1) * 100
        g[f"k_btc_above{lab}_low"] = (c / s5l.rolling(n, min_periods=n).min().values - 1) * 100
    h1max = h1.h.rolling(720, min_periods=720).max()                 # 720 completed 1h bars = 30 d
    m30 = h1max.reindex((T // H) * H - H).values
    cur = s5h.rolling(12, min_periods=1).max().values                # this hour's high so far (≈)
    g["k_btc_off30d_high"] = (c / np.fmax(m30, cur) - 1) * 100
    for lab, n in [("1h", 12), ("24h", 288), ("72h", 864)]:
        g[f"k_btc_ret{lab}"] = (c / pd.Series(c).shift(n).values - 1) * 100
    lr = np.log(pd.Series(c)).diff()
    g["k_btc_rv1h"] = lr.rolling(12).std().values * 100
    g["k_btc_rv24h"] = lr.rolling(288).std().values * 100
    g["k_btc_rv_ratio"] = g.k_btc_rv1h / g.k_btc_rv24h
    # eff72: |net 72h move| / Σ|1h moves| (bull-monitor style efficiency; validated vs stamp below)
    hc = h1.c
    eff = (hc - hc.shift(72)).abs() / hc.diff().abs().rolling(72).sum()
    g["k_btc_eff72"] = eff.reindex((T // H) * H - H).values
    # slope dynamics on the grid
    s = g.k_btc_slope1h
    for k in (1, 2, 3):
        g[f"k_btc_slope1h_chg{k}h"] = s - s.shift(12 * k)
    pos = (s >= 0).values
    last_pos_T = pd.Series(np.where(pos, T, np.nan)).ffill().values
    g["k_btc_hrs_since_slope_neg"] = np.where(pos, 0.0, (T - last_pos_T) / H)
    return g


def sym_grid(sym):
    p = k5(sym)
    T, c = p["T"].values, p.c.values
    g = pd.DataFrame({"T": T})
    ph = pd.Series(c, index=T)
    h1 = ph.groupby(((T - 300_000) // H) * H).last()                    # hourly close by hour open
    e20, e20p3 = htf_partial(h1, H, 20, T, c)
    e200, _ = htf_partial(h1, H, 200, T, c)
    g["slope1h"] = (e20 - e20p3) / e20p3 * 100
    g["gap1h_20_200"] = (e20 / e200 - 1) * 100
    h4 = h1.groupby((h1.index // D4) * D4).last()
    e20_4, _ = htf_partial(h4, D4, 20, T, c)
    e50_4, _ = htf_partial(h4, D4, 50, T, c)
    g["gap4h_20_50"] = (e20_4 / e50_4 - 1) * 100
    g["ret1h"] = (c / pd.Series(c).shift(12).values - 1) * 100
    g["ret24h"] = (c / pd.Series(c).shift(288).values - 1) * 100
    g["off24h_high"] = (c / p.h.rolling(288, min_periods=288).max().values - 1) * 100
    g["above24h_low"] = (c / p.l.rolling(288, min_periods=288).min().values - 1) * 100
    qv = p.qvol.rolling(288, min_periods=288).sum()
    g["qvol24_vs_7d"] = (qv / qv.rolling(2016, min_periods=576).mean()).values
    return g


def alt_index():
    """hourly 24h return of every non-BTC symbol in k5m_full → median 24h return and share > 0 at each hour close."""
    cache = "reports/NEGFLANK_2D_altindex.pkl"
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    rets = {}
    for f in sorted(glob.glob(f"{K5}/*.csv")):
        s = os.path.basename(f)[:-4]
        if s == "BTCUSDT":
            continue
        d = pd.read_csv(f, usecols=["open_time", "c"]).drop_duplicates("open_time")
        hc = d.set_index((d.open_time // H) * H).c.groupby(level=0).last()
        hc = hc[d.groupby(d.open_time // H * H).size().reindex(hc.index).values == 12]     # complete hours only
        if os.path.exists(f"{EXT}/1h/{s}.csv"):
            x = pd.read_csv(f"{EXT}/1h/{s}.csv").set_index("open_time").c
            hc = pd.concat([hc, x[x.index > (hc.index.max() if len(hc) else 0)]])
        hc.index = hc.index + H                                       # value known at hour close
        rets[s] = (hc / hc.shift(24) - 1) * 100
    R = pd.DataFrame(rets).sort_index()
    R = R[R.index <= 1791482400000]
    out = pd.DataFrame({"alt_med_ret24": R.median(1), "alt_up_share24": (R > 0).sum(1) / R.notna().sum(1), "alt_n": R.notna().sum(1)})
    out.to_pickle(cache)
    return out


def lookup(grid, t_ms, cols):
    idx = np.searchsorted(grid["T"].values, t_ms, side="right") - 1
    ok = idx >= 0
    out = pd.DataFrame(index=range(len(t_ms)), columns=cols, dtype=float)
    out.loc[ok, cols] = grid[cols].values[idx[ok]]
    # stale guard: the bar must be ≤ 15 min old
    age = t_ms - np.where(ok, grid["T"].values[np.clip(idx, 0, None)], 0)
    out.loc[~ok | (age > 900_000), cols] = np.nan
    return out


def main():
    m = C.master(include_b1=True)
    y = C.yr5()
    m["seed"] = 0
    keep = sorted(set(m.columns) & set(y.columns))
    F = pd.concat([m[keep], y[keep]], ignore_index=True)
    F["t_ms"] = (F.o.astype("datetime64[ms]").astype("int64")).values
    bg = btc_grid()
    bcols = [c for c in bg.columns if c.startswith("k_")]
    F[bcols] = lookup(bg, F.t_ms.values, bcols).values
    eg = sym_grid("ETHUSDT")
    E = lookup(eg, F.t_ms.values, ["slope1h", "ret24h"])
    F["k_eth_slope1h"] = E.slope1h.values; F["k_eth_ret24h"] = E.ret24h.values
    pcols = ["slope1h", "gap1h_20_200", "gap4h_20_50", "ret1h", "ret24h", "off24h_high", "above24h_low", "qvol24_vs_7d"]
    for sym, idx in F.groupby("pair").groups.items():
        g = sym_grid(sym)
        P = lookup(g, F.loc[idx, "t_ms"].values, pcols)
        for c in pcols:
            F.loc[idx, f"k_pair_{c}"] = P[c].values
    F["k_pair_rel24_vs_btc"] = F.k_pair_ret24h - F.k_btc_ret24h
    A = alt_index()
    hk = (F.t_ms // H) * H                                            # last completed hour close
    F["k_alt_med_ret24"] = A.alt_med_ret24.reindex(hk).values
    F["k_alt_up_share24"] = A.alt_up_share24.reindex(hk).values
    F["k_btc_dom24"] = F.k_btc_ret24h - F.k_alt_med_ret24             # BTC vs median alt, 24 h (dominance proxy)
    F["k_hour_utc"] = F.o.dt.hour + F.o.dt.minute / 60
    F["k_dow"] = F.o.dt.dayofweek
    # derived from stamps (pure arithmetic on stamped fields, both cohorts)
    num = lambda c: pd.to_numeric(F[c], errors="coerce")
    F["d_btc_rsi1h_chg"] = num("entry_btc_rsi_1h") - num("entry_btc_rsi_1h_prev")
    F["d_btc_rsi5m_chg6"] = num("entry_btc_rsi") - num("entry_btc_rsi_prev6")
    F["d_btc_adx_chg"] = num("entry_btc_adx") - num("entry_btc_adx_prev")
    F["d_rsi_chg"] = num("entry_rsi") - num("entry_rsi_prev")
    F["d_di_spread"] = num("entry_pos_di") - num("entry_neg_di")
    F.to_pickle(OUT)
    # rebuild validation vs stamps where both exist
    VAL = [("k_btc_slope1h", "entry_btc_1h_slope"), ("k_btc_off30d_high", "entry_btc_off30d_high_pct"),
           ("k_btc_off24h_high", "entry_btc_off24h_pct"), ("k_btc_above24h_low", "entry_btc_off24lo_pct"),
           ("k_btc_prevday_ret", "entry_btc_1d_ret_pct"), ("k_btc_ret72h", "entry_btc_r72_pct"), ("k_btc_eff72", "entry_btc_eff72"),
           ("k_pair_gap1h_20_200", "entry_pair_1h_ema20_200_gap_pct"), ("k_btc_1h_gap20_50", "entry_btc_trend_gap_pct"),
           ("k_pair_slope1h", "entry_ema20_slope")]
    rows = []
    for a, b in VAL:
        for src in ("master", "yr5"):
            d = F[F.src == src]
            x, s = pd.to_numeric(d[a], errors="coerce"), pd.to_numeric(d[b], errors="coerce")
            ok = x.notna() & s.notna()
            rows.append(dict(rebuild=a, stamp=b, src=src, n=int(ok.sum()), r=x[ok].corr(s[ok]) if ok.sum() > 3 else np.nan,
                             med_abs_diff=(x[ok] - s[ok]).abs().median() if ok.any() else np.nan,
                             cov_rebuild=x.notna().mean()))
    V = pd.DataFrame(rows)
    V.to_csv("reports/NEGFLANK_2D_rebuild_validation.csv", index=False)
    print(V.to_string())
    print("features:", F.shape, "k-coverage master/yr5:")
    kc = [c for c in F.columns if c.startswith("k_")]
    print(F.groupby("src")[kc].apply(lambda d: d.notna().mean()).T.round(3).to_string())


if __name__ == "__main__":
    main()
