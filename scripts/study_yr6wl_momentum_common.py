"""Shared loaders / helpers for the YR6 winners-vs-losers MOMENTUM study (2026-10-13).

Brief: reports/YR6_WL_STUDY_BRIEF.md (binding). Cache-only, no network, read-only on all inputs.
Outcome = pct_new (today's exit; for momentum pct_new == pct — the ×0.5 long sizing is $ only, % is
leverage-invariant, i.e. already "de-multiplied to 1×"). Seeds pooled, every CI clustered by UTC day.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILLS = os.path.join(ROOT, "reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
MASTER = os.path.join(ROOT, "reports/MASTER_POOL_stacked.csv")
SCREENED = os.path.join(ROOT, "reports/SCREENED_BASELINE.csv")
TOP50 = [os.path.join(ROOT, "reports/backtest_cache/universe_daily_top50.csv"),
         os.path.join(ROOT, "reports/backtest_cache_2025/universe_daily_top50.csv")]
SLEEVES = ["MOM-long", "MOM-short"]
REGIMES = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]

# columns that are not entry SIGNALS (prices, fees, sizing, timestamps, order plumbing)
NON_SIGNAL = {"entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional",
              "entry_liquidity_cap_notional", "entry_order_type", "entry_strategy",
              "entry_btc_regime_started_at", "entry_bracket_max_leverage", "entry_bracket_cap_notional",
              "entry_surge_trigger_at", "entry_frenzy_spike_at"}
# MARKET-WIDE variables -> read in DAY units (window-units rule)
MACRO_KEYS = ("btc", "global_volume", "entry_bull_pct", "entry_bear_pct", "eth_", "entry_br_", "entry_bear_r24",
              "entry_bear_below24", "entry_bear_eff24", "entry_bear_off24lo", "entry_bear_bypass", "surge_gvol",
              "frenzy_gvol")
CATEG = ["entry_macro_trend", "entry_btc_regime"]


def is_macro(c):
    return any(k in c for k in MACRO_KEYS)


def load_top50():
    u = pd.concat([pd.read_csv(f) for f in TOP50])
    u["date"] = pd.to_datetime(u.date_ms, unit="ms").dt.date
    s = set(zip(u.date, u.pair))
    return s


def load_replay(sleeves=SLEEVES):
    d = pd.read_csv(FILLS, low_memory=False)
    d = d[d.sleeve.isin(sleeves) & (d.status == "CLOSED")].copy()
    if "is_probe" in d:
        d = d[~d.is_probe.fillna(False).astype(bool)]
    d["ts"] = pd.to_datetime(d.t, format="mixed")
    d["day"] = d.ts.dt.floor("D")
    d["y"] = d.pct_new.astype(float)
    d["win"] = d.y > 0
    d["hour"] = d.ts.dt.hour
    # derived signed/categorical helpers (fill-level, not entry_* but stamped at entry)
    for c in CATEG:
        for v in sorted(d[c].dropna().unique()):
            d[f"cat__{c}=={v}"] = (d[c] == v).astype(float)
    top = load_top50()
    prev = d.day - pd.Timedelta(days=1)
    d["top50"] = [((a.date(), p) in top) or ((b.date(), p) in top) for a, b, p in zip(d.day, prev, d.pair)]
    return d.sort_values("ts").reset_index(drop=True)


def load_master_mom():
    m = pd.read_csv(MASTER, low_memory=False)
    m = m[(m.status == "CLOSED") & ~m.is_probe.fillna(False).astype(bool)]
    m = m[m.entry_strategy.astype(str).str.startswith("MOMENTUM")].copy()
    m["sleeve"] = np.where(m.direction == "LONG", "MOM-long", "MOM-short")
    m["ts"] = pd.to_datetime(m.opened_at.astype(str).str[:19])
    m["day"] = m.ts.dt.floor("D")
    m["y"] = m.pnl_percentage.astype(float)              # as traded
    m["y_stack"] = m.stack_pct.astype(float)             # today's stack (NaN when blocked)
    m["hour"] = m.ts.dt.hour
    for c in CATEG:
        if c in m:
            for v in sorted(m[c].dropna().unique()):
                m[f"cat__{c}=={v}"] = (m[c] == v).astype(float)
    return m


def signal_columns(d, min_cov=0.0):
    cols = []
    for c in d.columns:
        if not (c.startswith("entry_") or c.startswith("cat__")) or c in NON_SIGNAL or c in CATEG:
            continue
        s = pd.to_numeric(d[c], errors="coerce")
        if s.notna().mean() <= min_cov or s.nunique() < 2:
            continue
        cols.append(c)
    return cols


def day_boot(y, day, n=2000, seed=7, stat=np.mean):
    """Day-clustered bootstrap of the mean: returns (mean, lo95, hi95, P(mean>0))."""
    y = np.asarray(y, float); day = np.asarray(day)
    if len(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=y); c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return y.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()


def day_boot_delta(y, inz, day, n=2000, seed=11):
    """Day bootstrap of mean(in) - mean(out). Returns (delta, lo, hi, P(delta>0))."""
    y = np.asarray(y, float); inz = np.asarray(inz, bool); day = np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    si = np.bincount(inv, weights=y * inz, minlength=len(u)); ci = np.bincount(inv, weights=inz.astype(float), minlength=len(u))
    so = np.bincount(inv, weights=y * ~inz, minlength=len(u)); co = np.bincount(inv, weights=(~inz).astype(float), minlength=len(u))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = si[idx].sum(1) / ci[idx].sum(1) - so[idx].sum(1) / co[idx].sum(1)
    bs = bs[np.isfinite(bs)]
    dl = y[inz].mean() - y[~inz].mean()
    return dl, np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()


def breakeven_wr(y):
    y = np.asarray(y, float)
    w, l = y[y > 0], y[y <= 0]
    if len(w) == 0 or len(l) == 0:
        return np.nan
    return abs(l.mean()) / (w.mean() + abs(l.mean()))


def fmt(x, p=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"
