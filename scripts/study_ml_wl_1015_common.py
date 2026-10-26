"""ML winners-vs-losers study 2026-10-15 — shared loaders (pre-registration: reports/ML_WL_2026-10-15_PREREG.txt).

Three blocks: ① live (post go-live exports via scripts/export_glob.py) · ② master (MASTER_POOL_stacked, as-traded and
stack_keep views; BASE era = SCREENED_BASELINE rows) · ③ replay (ENGINE_REPLAY_YR6_2Y_TODAY_fills, MOM-long, 3 seeds).
Cache-only kline access (reports/backtest_cache*, plus the optional reports/ml_wl_1015_fetch/ top-up for the post-cache
live window). Read-only on every input.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_momentum_common import (ROOT, day_boot, day_boot_delta, breakeven_wr, load_replay,  # noqa: F401
                                         load_master_mom, signal_columns, is_macro, NON_SIGNAL, CATEG)

C26 = os.path.join(ROOT, "reports/backtest_cache")
C25 = os.path.join(ROOT, "reports/backtest_cache_2025")
TOPUP = os.path.join(ROOT, "reports/ml_wl_1015_fetch")
FEAT_CSV = os.path.join(ROOT, "reports/ML_WL_1015_features.csv")
SPLIT_2026 = pd.Timestamp("2026-01-01").value // 10**6
EXCL = {"BTCUSDT", "USDCUSDT", "PAXGUSDT", "XAUTUSDT", "FDUSDUSDT"}
MIN1, MIN5 = 60_000, 300_000

# market-wide new features (read in DAY units) vs pair-level
NEW_MACRO = ["btc_r5", "btc_r15", "btc_r30", "btc_r60", "btc_vol5_x24h", "br_up15", "br_up30", "br_up60",
             "br_up30_chg1h", "br_ret15", "br_ret60", "disp15", "agg_vol15_x24h"]
NEW_PAIR = ["pair_r15", "pair_r60", "rs15", "rs60", "pair_vol15_x24h"]
NEW_ENTRY = NEW_MACRO + NEW_PAIR
DURING = ["btc_d5", "btc_d15", "br_d5", "br_d15", "pnl_m1", "pnl_m3", "pnl_m5", "pnl_m15"]
EARLY = ["e1_pnl", "e3_pnl", "e5_pnl", "e1_btc", "e3_btc", "e5_btc", "e5_br", "e1_open", "e3_open", "e5_open"]
DERIVED = ["d_fan_ratio", "d_adx_dir", "d_rsi_dir", "d_btc_adx_dir", "d_btc_rsi_dir", "d_btc_rsi_30m_dir",
           "d_abs_ema20_slope", "d_abs_btc_ema20_slope", "d_hour"]


def add_derived(d):
    g58 = pd.to_numeric(d.get("entry_ema_gap_5_8"), errors="coerce")
    g813 = pd.to_numeric(d.get("entry_ema_gap_8_13"), errors="coerce")
    d["d_fan_ratio"] = np.where(g813 > 0, g58 / g813, np.nan)
    n = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    d["d_adx_dir"] = n("entry_adx") - n("entry_adx_prev")
    d["d_rsi_dir"] = n("entry_rsi") - n("entry_rsi_prev")
    d["d_btc_adx_dir"] = n("entry_btc_adx") - n("entry_btc_adx_prev")
    d["d_btc_rsi_dir"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev")
    d["d_btc_rsi_30m_dir"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev6")
    d["d_abs_ema20_slope"] = n("entry_ema20_slope").abs()
    d["d_abs_btc_ema20_slope"] = n("entry_btc_ema20_slope").abs()
    d["d_hour"] = d.ts.dt.hour
    return d


def load_live():
    """Block ①: live MOMENTUM LONG fills since go-live (export_glob filters to is_paper false & opened ≥ go-live)."""
    from export_glob import glob_exports
    fs = [f for f in glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")) if "_live_" in f]
    if not fs:
        return pd.DataFrame()
    L = pd.concat([pd.read_csv(f, low_memory=False) for f in fs])
    L = L[(L.status == "CLOSED") & (L.direction == "LONG") & (L.entry_strategy.astype(str) == "MOMENTUM")]
    L = L.drop_duplicates(["opened_at", "pair", "direction"]).copy()
    L["ts"] = pd.to_datetime(L.opened_at.astype(str).str[:19]); L["day"] = L.ts.dt.floor("D")
    L["y"] = L.pnl_percentage.astype(float); L["win"] = L.y > 0
    L["block"] = "live"
    return L


def load_blocks():
    """returns dict block -> DataFrame with ts, day, y, win, closed_ts, entry_price, pair + entry_* columns."""
    R = load_replay(["MOM-long"])
    R["block"] = R.regime
    M = load_master_mom(); M = M[M.sleeve == "MOM-long"].copy(); M["win"] = M.y > 0; M["block"] = "master"
    L = load_live()
    out = {}
    for name, df in [("live", L), ("master", M)] + [(r, R[R.regime == r].copy()) for r in
                                                    ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]]:
        if not len(df):
            out[name] = df; continue
        df = df.copy()
        df["closed_ts"] = pd.to_datetime(df.closed_at.astype(str).str[:23], format="mixed")
        df["ts_open"] = pd.to_datetime(df.opened_at.astype(str).str[:23], format="mixed")
        df["dur_min"] = (df.closed_ts - df.ts_open).dt.total_seconds() / 60
        for c, n in [("peak_reached_at", "peak_min"), ("trough_reached_at", "trough_min")]:
            x = pd.to_datetime(df[c].astype(str).str[:23], format="mixed", errors="coerce")
            df[n] = (x - df.ts_open).dt.total_seconds() / 60
        df["fee_pct"] = (df.exit_price / df.entry_price - 1) * 100 - (df.pct if "pct" in df else df.y)
        df = add_derived(df)
        out[name] = df.sort_values("ts").reset_index(drop=True)
    return out


# ------------------------------------------------------------------ klines
class Klines:
    """Lazy per-pair kline arrays; picks the 2025 or 2026 cache by timestamp; top-up dir for post-cache live data."""

    def __init__(self):
        self.cache = {}

    def clear(self):
        self.cache = {}

    def _load(self, base, sub, pair):
        key = (base, sub, pair)
        if key in self.cache:
            return self.cache[key]
        f = os.path.join(base, sub, pair + ".csv")
        a = None
        if os.path.exists(f):
            try:
                x = pd.read_csv(f, usecols=["open_time", "c", "qvol"])
                x = x.drop_duplicates("open_time").sort_values("open_time")
                if base == C25:
                    x = x[x.open_time >= 1732924800000]          # 2024-11-30: the 2025 block needs ≥ 24 h of history
                a = (x.open_time.values.astype(np.int64), x.c.values.astype(np.float32), None,
                     np.concatenate([[0.0], np.cumsum(x.qvol.values.astype(float))]))
            except Exception:
                a = None
        self.cache[key] = a
        return a

    def arr(self, pair, tms, iv):
        subs = ["k1m"] if iv == MIN1 else ["k5m_full", "k5m"]
        bases = ([C25, C26] if tms < SPLIT_2026 else [C26, C25]) + [TOPUP]
        for b in bases:
            for s in subs:
                a = self._load(b, s, pair)
                if a is not None and len(a[0]) and a[0][0] <= tms - 2 * 3600_000 and a[0][-1] >= tms - 2 * iv:
                    return a
        for b in bases:            # fall back: any file that at least covers tms
            for s in subs:
                a = self._load(b, s, pair)
                if a is not None and len(a[0]) and a[0][0] <= tms and a[0][-1] >= tms - 2 * iv:
                    return a
        return None

    def px(self, pair, T, iv, a=None):
        """close of the last bar CLOSED at or before T (ms); NaN if stale (> 2 bars old) or missing."""
        a = a if a is not None else self.arr(pair, T, iv)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], T - iv, side="right") - 1
        if i < 0 or a[0][i] < T - 3 * iv:
            return np.nan
        return float(a[1][i])

    def vol_ratio(self, pair, T, nbars, iv=MIN5, base_bars=288):
        """quote volume of the last nbars closed bars / (mean over the base_bars before them × nbars)."""
        a = self.arr(pair, T, iv)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], T - iv, side="right") - 1
        if i < base_bars + nbars or a[0][i] < T - 3 * iv:
            return np.nan
        cs = a[3]
        recent = cs[i + 1] - cs[i + 1 - nbars]
        base = (cs[i + 1 - nbars] - cs[i + 1 - nbars - base_bars]) / base_bars * nbars
        return recent / base if base > 0 else np.nan

    def qvol_sum(self, pair, T, nbars, offset_bars=0, iv=MIN5):
        a = self.arr(pair, T, iv)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], T - iv, side="right") - 1 - offset_bars
        if i - nbars + 1 < 0 or a[0][i + offset_bars] < T - 3 * iv:
            return np.nan
        return a[3][i + 1] - a[3][i + 1 - nbars]


def load_universe():
    u = pd.concat([pd.read_csv(os.path.join(b, "universe_daily_top50.csv")) for b in (C25, C26)])
    u = u.drop_duplicates(["date_ms", "pair"])
    days = {}
    for dm, g in u.groupby("date_ms"):
        days[int(dm)] = [p for p in g.pair if p not in EXCL and re.match(r"^[A-Z0-9]+USDT$", p)]
    keys = np.array(sorted(days))
    return days, keys


def universe_for(days, keys, tms):
    """PREVIOUS UTC day's top-50 list (no lookahead); the last available list when the cache ends earlier."""
    day0 = tms - tms % 86_400_000
    i = np.searchsorted(keys, day0 - 1, side="right") - 1      # strictly before today
    if i < 0:
        return [], None
    return days[keys[i]], int(keys[i])
