#!/usr/bin/env python3
"""LEADS DEEP addendum R — regime series R1–R6, 2020-06 → 2026-11, lag-safe (value valid FROM the close of the bar it was computed on).
R1 BTC daily EMA50 vs EMA200 · R2 daily close vs EMA200 · R3 BTC 4 h EMA50 vs EMA200 · R4 services/regime.py label on BTC 5m (bear labels −1,
bull labels +1, else 0) · R5 share of USDT perps with daily close > daily EMA50 (< 50 % → −1; ≥ 30 pairs) · R6 BTC 7-day return sign.
→ reports/LEADS_DEEP_2026-11-18_data/regimes.pkl  {name: (valid_from_ms array, value array)}"""
import glob, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import frenzy_ta_build as TB  # noqa
import exh_expand_build as XB  # noqa  (K5M sources)
from services.regime import classify_btc_regime  # noqa
DD = os.path.join(R, "reports", "LEADS_DEEP_2026-11-18_data")
M5, H4, DAY = 300_000, 14_400_000, 86_400_000
ema = lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean().values


def btc5m():
    a = pd.read_csv(os.path.join(DD, "k5m", "BTCUSDT.csv"))[["open_time", "o", "h", "l", "c"]]
    xs = []
    for f in sorted(glob.glob(os.path.join(R, "reports", "UNIFIED_SHORT_data", "btc_raw", "*.csv"))):
        z = pd.read_csv(f); xs.append(pd.DataFrame({"open_time": z.open_time, "o": z.open, "h": z.high, "l": z.low, "c": z.close}))
    b = pd.concat(xs)
    t, o, h, l, c, v, qv = TB.load("BTCUSDT")
    cc = pd.DataFrame({"open_time": t, "o": o, "h": h, "l": l, "c": c})
    d = pd.concat([a, b, cc]); d["open_time"] = d.open_time.astype("int64")
    d.loc[d.open_time > 10**14, "open_time"] //= 1000
    d = d.drop_duplicates("open_time", keep="last").sort_values("open_time")
    grid = np.arange(d.open_time.iloc[0], d.open_time.iloc[-1] + 1, M5)
    return d.set_index("open_time").reindex(grid).ffill().reset_index().rename(columns={"index": "open_time"})


def daily_close(f):
    try:
        x = pd.read_csv(f, usecols=["open_time", "c"])
    except Exception:
        return None
    if len(x) < 1000:
        return None
    x["d"] = x.open_time // DAY
    s = x.groupby("d").c.last()
    return os.path.basename(f)[:-4], s


if __name__ == "__main__":
    out = {}
    b = pd.read_csv(os.path.join(DD, "BTCUSDT_spot_1d.csv")).sort_values("open_time")
    tv = b.open_time.values.astype("int64") + DAY; c = b.c.values
    out["R1"] = (tv, np.sign(ema(c, 50) - ema(c, 200)))
    out["R2"] = (tv, np.sign(c - ema(c, 200)))
    r7 = np.r_[np.full(7, np.nan), c[7:] / c[:-7] - 1]; out["R6"] = (tv, np.sign(r7))
    B = btc5m(); t = B.open_time.values.astype("int64")
    g = t // H4; st = np.r_[0, np.nonzero(np.diff(g))[0] + 1]; en = np.r_[st[1:], len(t)] - 1
    full = (en - st) == 47; c4 = B.c.values[en]
    out["R3"] = ((g[st] * H4 + H4)[full], np.sign(ema(c4[full], 50) - ema(c4[full], 200)))
    h, l, cc = B.h.values, B.l.values, B.c.values
    adx, _, _ = TB.adx_di(h, l, cc); r14 = TB.rsi(cc, 14); e20 = TB.ema(cc, 20)
    slope = (e20 - np.r_[np.full(3, np.nan), e20[:-3]]) / np.r_[np.full(3, np.nan), e20[:-3]] * 100
    lab = [classify_btc_regime(a if np.isfinite(a) else None, r if np.isfinite(r) else None, s if np.isfinite(s) else None) for a, r, s in zip(adx, r14, slope)]
    m = {"HEALTHY_BULL": 1, "STRONG_BULL": 1, "BULL_EXHAUSTED": 1, "HEALTHY_BEAR": -1, "STRONG_BEAR": -1, "BEAR_EXHAUSTED": -1}
    out["R4"] = (t + M5, np.array([m.get(x, 0) for x in lab], float))
    # R5 breadth from every 5m cache (old fetch + modern sources)
    fs = {}
    for d in [os.path.join(DD, "k5m")] + list(XB.TB.K5M_SRC):
        for f in glob.glob(os.path.join(d, "*USDT.csv")):
            fs.setdefault(os.path.basename(f)[:-4], []).append(f)
    fs = {p: v for p, v in fs.items() if p not in ("BTCUSDT", "ETHUSDT") and p.isascii()}
    S = {}
    with mp.get_context("fork").Pool(6) as pool:
        for r in pool.imap_unordered(daily_close, [f for v in fs.values() for f in v], chunksize=8):
            if r:
                p, s = r; S[p] = pd.concat([S[p], s]).groupby(level=0).last() if p in S else s
    days = np.arange(pd.Timestamp("2020-06-01").value // 10**6 // DAY, pd.Timestamp("2026-11-19").value // 10**6 // DAY)
    up = np.zeros(len(days)); n = np.zeros(len(days))
    for p, s in S.items():
        s = s.sort_index(); s = s[~s.index.duplicated()]
        full_ = s.reindex(np.arange(s.index.min(), s.index.max() + 1)).ffill()
        e = pd.Series(ema(full_.values, 50), index=full_.index)
        ok = np.arange(len(full_)) >= 60
        a = (full_.values > e.values)[ok]; ix = full_.index.values[ok]
        k = np.searchsorted(days, ix); good = (k < len(days)) & (days[np.minimum(k, len(days) - 1)] == ix)
        up[k[good]] += a[good]; n[k[good]] += 1
    br = np.where(n >= 30, up / np.maximum(n, 1), np.nan)
    out["R5"] = ((days + 1) * DAY, np.where(np.isfinite(br), np.where(br < 0.5, -1.0, 1.0), np.nan))
    out["R5_share"] = ((days + 1) * DAY, br); out["R5_n"] = ((days + 1) * DAY, n)
    pd.to_pickle(out, os.path.join(DD, "regimes.pkl"))
    for k, (tt, vv) in out.items():
        if k.startswith("R") and "_" not in k:
            x = pd.Series(vv, index=pd.to_datetime(tt, unit="ms"))
            x = x[x.index >= "2020-09-01"]
            print(k, "bear share", round((x == -1).mean(), 3), "bull", round((x == 1).mean(), 3), "switches", int((x.diff().fillna(0) != 0).sum()))
