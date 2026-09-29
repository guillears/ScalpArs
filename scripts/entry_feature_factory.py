#!/usr/bin/env python3
"""Entry feature factory — rebuild EVERY price/volume indicator at each fill's entry instant from the 5m kline cache, so
winners-vs-losers screens are not limited to what the engine happened to stamp (e.g. there is no BTC EMA50/EMA200 stamp).

For BTC, ETH, BTCDOM (macro) and the traded PAIR, on 5 timeframes (5m 15m 1h 4h 1d), using only bars CLOSED before entry:
  price vs EMA 9/20/50/100/200 · EMA gaps 9-20, 20-50, 50-100, 50-200, 20-200, 100-200 · EMA20/50/200 slopes (3 bars) ·
  RSI14 + its 1/3-bar deltas · ADX14, +DI, −DI, DI spread, ADX 1/3-bar delta · ATR% · returns 1/3/6/12/24 bars ·
  range position 20/50 bars · distance from 20/50-bar high & low · realized vol 20 · volume ratio 20 · bars since EMA20/50 cross
Plus pair-vs-BTC relative (return spread, RSI spread) per timeframe. Output columns are prefixed  <SYM>_<tf>_<feature>.

Usage (library): features(df with opened_at, pair) -> DataFrame aligned to df.index
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K5 = os.path.join(ROOT, "reports", "backtest_cache", "k5m_full")
TFS = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}
MACROS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "DOM": "BTCDOMUSDT"}
_cache = {}


def _wilder(x, n):
    return x.ewm(alpha=1 / n, adjust=False).mean()


def _ind(k):
    """k: OHLCV frame indexed by bar CLOSE time → indicator frame on the same index."""
    c, h, l, v = k.c, k.h, k.l, k.vol
    o = {}
    ema = {n: c.ewm(span=n, adjust=False).mean() for n in (9, 20, 50, 100, 200)}
    for n, e in ema.items():
        o[f"px_vs_ema{n}"] = (c / e - 1) * 100
    for a, b in ((9, 20), (20, 50), (50, 100), (50, 200), (20, 200), (100, 200)):
        o[f"gap_ema{a}_{b}"] = (ema[a] / ema[b] - 1) * 100
    for n in (20, 50, 200):
        o[f"ema{n}_slope3"] = (ema[n] / ema[n].shift(3) - 1) * 100
    d = c.diff()
    rs = _wilder(d.clip(lower=0), 14) / _wilder((-d).clip(lower=0), 14)
    rsi = 100 - 100 / (1 + rs)
    o["rsi"], o["rsi_d1"], o["rsi_d3"] = rsi, rsi.diff(), rsi.diff(3)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=k.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=k.index)
    pdi, ndi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(ndm, 14) / atr
    adx = _wilder(100 * (pdi - ndi).abs() / (pdi + ndi), 14)
    o.update(adx=adx, adx_d1=adx.diff(), adx_d3=adx.diff(3), pdi=pdi, ndi=ndi, di_spread=pdi - ndi, atr_pct=atr / c * 100)
    for n in (1, 3, 6, 12, 24):
        o[f"ret{n}"] = (c / c.shift(n) - 1) * 100
    for n in (20, 50):
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        o[f"rangepos{n}"] = (c - ll) / (hh - ll) * 100
        o[f"off_hi{n}"] = (c / hh - 1) * 100
        o[f"off_lo{n}"] = (c / ll - 1) * 100
    o["rvol20"] = c.pct_change().rolling(20).std() * 100
    o["volratio20"] = v / v.rolling(20).mean()
    above = (ema[20] > ema[50]).astype(int)
    grp = (above != above.shift()).cumsum()
    o["bars_since_x20_50"] = above.groupby(grp).cumcount() * np.where(above == 1, 1, -1)
    return pd.DataFrame(o, index=k.index)


def _load(sym):
    if sym in _cache:
        return _cache[sym]
    p = os.path.join(K5, f"{sym}.csv")
    if not os.path.exists(p):
        _cache[sym] = None
        return None
    k = pd.read_csv(p)
    k["t"] = pd.to_datetime(k.open_time, unit="ms")
    k = k.set_index("t")[["o", "h", "l", "c", "vol"]].astype(float)
    out = {}
    for tf, rule in TFS.items():
        r = k.copy() if tf == "5m" else k.resample(rule, label="left", closed="left").agg(
            {"o": "first", "h": "max", "l": "min", "c": "last", "vol": "sum"}).dropna()
        r.index = r.index + pd.Timedelta(rule)            # index = bar CLOSE time → asof lookup never sees a forming bar
        out[tf] = _ind(r)
    _cache[sym] = out
    return out


def _asof(frames, times, prefix):
    cols = {}
    for tf, fr in frames.items():
        idx = fr.index.searchsorted(times.values, side="right") - 1
        ok = idx >= 0
        vals = fr.values[np.where(ok, idx, 0)]
        vals[~ok] = np.nan
        for j, c in enumerate(fr.columns):
            cols[f"{prefix}_{tf}_{c}"] = vals[:, j]
    return cols


def features(df):
    t = pd.to_datetime(df.opened_at.astype(str).str[:19], errors="coerce")
    out = {}
    for pre, sym in MACROS.items():
        fr = _load(sym)
        if fr is not None:
            out.update(_asof(fr, t, pre))
    pair_cols = None
    parts = []
    for pair, g in df.groupby(df.pair):
        fr = _load(pair)
        if fr is None:
            continue
        c = _asof(fr, t.loc[g.index], "PAIR")
        parts.append(pd.DataFrame(c, index=g.index))
    X = pd.DataFrame(out, index=df.index)
    if parts:
        P = pd.concat(parts).reindex(df.index)
        X = pd.concat([X, P], axis=1)
        for tf in TFS:
            for f in ("ret1", "ret3", "ret12", "ret24", "rsi", "px_vs_ema50", "px_vs_ema200"):
                a, b = f"PAIR_{tf}_{f}", f"BTC_{tf}_{f}"
                if a in X and b in X:
                    X[f"REL_{tf}_{f}"] = X[a] - X[b]
    return X
