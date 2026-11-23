"""Route B signal-level loader — the BTC DAILY-TREND stamp (weekly-scan PREREG §2 build item 4; CLAUDE.md live-mode lesson: entry rules
behave opposite across the BTC daily regime).

btc_d_trend at T = sign(EMA50 − EMA200) of BTCUSDT perpetual DAILY closes, on the last UTC day that CLOSED at or before T (a day still
open at T is never read). EMAs: pandas ewm(span, adjust=False) over the fetched daily closes (the last 1,500 days ≈ 4.1 years; the
MIN_DAYS guard alone bounds the EMA200 seed residual at ≈ 8 %, the real 1,500-day fetch at ~e^-15 — the house convention of scripts/oos_real_fan_common.btc_daily_regime / NEWSLEEVE btc_regime).
+1 = BULL (EMA50 above), −1 = BEAR, 0 = an exact tie (B4 treats 0 as NO sign; the house helpers call it BEAR), NaN = fewer than MIN_DAYS closed days (EMA200 not warmed up), no data, or STALE (the
last closed day in the table ends ≥ MAX_STALE_DAYS before T — a failed refresh never stamps an old sign silently).
Source: fapi /klines 1d (one request, limit 1500 = weight 10, through the build's rate limiter), cached at DATA/btc_1d.pkl.gz and
re-fetched only when the cache lacks the last closed day the run needs."""
import math
import os

import numpy as np
import pandas as pd

from common import DATA

DAY_MS = 86_400_000
MIN_DAYS = 250                 # EMA200 needs a warm-up; fewer closed days → unscored
MAX_STALE_DAYS = 1             # T ≥ this after the last closed day's close → unscored (complete data: always < 1 day, so one missed
                               # daily refresh already stamps NaN, never the day before's sign)
CACHE = "btc_1d.pkl.gz"


def trend_table(days):
    """days = DataFrame(open_ms, c) of DAILY bars (any order) → DataFrame(close_ms, ema50, ema200, trend) per day, sorted. Pure."""
    d = days.dropna(subset=["c"]).drop_duplicates("open_ms").sort_values("open_ms").reset_index(drop=True)
    e50 = d.c.ewm(span=50, adjust=False).mean()
    e200 = d.c.ewm(span=200, adjust=False).mean()
    n = np.arange(1, len(d) + 1)
    tr = np.where(n >= MIN_DAYS, np.sign(e50 - e200), np.nan)
    return pd.DataFrame(dict(close_ms=d.open_ms.astype("int64") + DAY_MS, ema50=e50, ema200=e200, trend=tr))


def stamp(table, T_ms):
    """{btc_d_trend, btc_d_ema50, btc_d_ema200, btc_d_day} for the last day whose close ≤ T. Pure."""
    if table is None or not len(table):
        return dict(btc_d_trend=math.nan, btc_d_ema50=math.nan, btc_d_ema200=math.nan, btc_d_day="")
    k = int(np.searchsorted(table.close_ms.to_numpy(np.int64), int(T_ms), "right")) - 1
    if k < 0 or int(T_ms) - int(table.close_ms.iloc[k]) >= MAX_STALE_DAYS * DAY_MS:
        return dict(btc_d_trend=math.nan, btc_d_ema50=math.nan, btc_d_ema200=math.nan, btc_d_day="")
    r = table.iloc[k]
    return dict(btc_d_trend=float(r.trend), btc_d_ema50=float(r.ema50), btc_d_ema200=float(r.ema200),
                btc_d_day=pd.Timestamp(int(r.close_ms) - DAY_MS, unit="ms", tz="UTC").strftime("%Y-%m-%d"))


def load(limiter, need_close_ms, data_dir=None, now_ms=None):
    """the daily table covering every close ≤ need_close_ms (floored to 00:00 UTC; re-fetched when the cache ends earlier). Only days
    whose close ≤ min(need, now) are kept (a day still open is never cached). None when unavailable."""
    need_close_ms = int(need_close_ms) // DAY_MS * DAY_MS
    if now_ms is not None:
        need_close_ms = min(need_close_ms, int(now_ms) // DAY_MS * DAY_MS)
    path = os.path.join(data_dir or DATA, CACHE)
    try:
        have = pd.read_pickle(path) if os.path.exists(path) else None
    except Exception:                                            # a corrupt / half-written cache = no cache (rebuilt below)
        have = None
    if have is not None and len(have) and int(have.open_ms.max()) + DAY_MS >= int(need_close_ms):
        return trend_table(have)
    js = limiter.get("/fapi/v1/klines", dict(symbol="BTCUSDT", interval="1d", limit=1500), 10)
    if isinstance(js, list) and js:
        k = pd.DataFrame([(int(r[0]), float(r[4])) for r in js], columns=["open_ms", "c"])
        k = k[k.open_ms + DAY_MS <= int(need_close_ms)]          # closed days only (the current day's bar is still moving)
        if have is not None and len(have):
            k = pd.concat([have, k]).drop_duplicates("open_ms", keep="last")
        if len(k):
            k.to_pickle(path + ".tmp", compression="gzip")
            os.replace(path + ".tmp", path)                      # atomic: a killed run never leaves a truncated cache
            return trend_table(k)
    return trend_table(have) if have is not None and len(have) else None
