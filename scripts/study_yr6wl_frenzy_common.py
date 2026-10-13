#!/usr/bin/env python3
"""YR6 two-year winners vs losers — FRENZY family common loader + statistics (reports/YR6_WL_STUDY_BRIEF.md, binding).

Cache-only, no network, read-only on every input. Imported by scripts/study_yr6wl_frenzy_*.py.

Data: reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv (re-priced to today's config, 443a7b5), FRENZY sleeves, CLOSED, trimmed.
  · outcome y = pct_new (today's exit, % per notional, fee-net, leverage-invariant)
  · FRENZY seeds ≈ one path → SEED 1 is the analysis cohort; seeds 2 / 3 pooled are the independent second computation of headline numbers.
  · day = UTC day of opened_at; episode = (pair, entry_frenzy_spike_at).
  · top50 = pair-day in that day's top-50 OR the prior day's top-50 (reports/backtest_cache*/universe_daily_top50.csv) — the 2025 tick
    coverage rule; every 2025 finding must hold inside top50 == True.
"""
import os
import numpy as np, pandas as pd

ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
REP = os.path.join(ROOT, "reports")
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
SLEEVES = ["FRENZY_LITE", "FRENZY_WILLY", "FRENZY_LONG", "FRENZY_WIDE"]
REGIMES = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
RSHORT = {"2025 BULL": "25B", "2026-H1 BEAR": "26H1", "2026-H2 BULL": "26H2"}

# MACRO / market-wide stamps (read in DAY units; window-units rule). Everything else entry_* = pair-level.
MACRO_PREFIX = ("entry_btc_", "entry_br_", "entry_bear_", "entry_eth_")
MACRO_EXTRA = {"entry_global_volume_ratio", "entry_bull_pct", "entry_bear_pct", "entry_macro_trend", "entry_frenzy_gvol", "entry_surge_gvol",
               "entry_surge_btc_move_pct", "entry_btc_regime"}


def is_macro(c):
    return c.startswith(MACRO_PREFIX) or c in MACRO_EXTRA


def top50_pairdays():
    s = set()
    for f in ("backtest_cache/universe_daily_top50.csv", "backtest_cache_2025/universe_daily_top50.csv"):
        u = pd.read_csv(os.path.join(REP, f))
        d = pd.to_datetime(u.date_ms, unit="ms").dt.strftime("%Y-%m-%d")
        nxt = (pd.to_datetime(u.date_ms, unit="ms") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        s |= set(zip(d, u.pair)); s |= set(zip(nxt, u.pair))   # same day OR prior-day top-50
    return s


_CACHE = {}


def load(all_seeds=False):
    key = bool(all_seeds)
    if key in _CACHE:
        return _CACHE[key].copy()
    D = pd.read_csv(FILLS, low_memory=False)
    D = D[D.sleeve.isin(SLEEVES) & (D.status == "CLOSED") & (~D.is_probe.astype(bool))]
    if not all_seeds:
        D = D[D.seed == 1]
    D = D.copy()
    D["y"] = pd.to_numeric(D.pct_new, errors="coerce")
    D["ts"] = pd.to_datetime(D.opened_at)
    D["day"] = D.ts.dt.strftime("%Y-%m-%d")
    D["month"] = D.ts.dt.strftime("%Y-%m")
    D["ep"] = D.pair + "|" + D.entry_frenzy_spike_at.astype(str)
    T = top50_pairdays()
    D["top50"] = [(d, p) in T for d, p in zip(D.day, D.pair)]
    D["win"] = D.y > 0
    _CACHE[key] = D
    return D.copy()


def entry_cols(D):
    out = []
    for c in D.columns:
        if not c.startswith("entry_") or c in ("entry_price", "entry_fee", "entry_strategy", "entry_order_type"):
            continue
        x = pd.to_numeric(D[c], errors="coerce")
        if x.notna().sum() == 0:
            continue
        out.append(c)
    return out


def num(D, c):
    x = D[c]
    if x.dtype == object:
        xs = x.astype(str).str.lower()
        if xs.isin(["true", "false", "nan", "none"]).all():
            return xs.map({"true": 1.0, "false": 0.0})
    return pd.to_numeric(x, errors="coerce")


def breakeven_wr(y):
    w, l = y[y > 0], y[y <= 0]
    if len(w) == 0 or len(l) == 0:
        return np.nan
    return abs(l.mean()) / (w.mean() + abs(l.mean()))


def day_boot(y, day, draws=4000, seed=7, stat="mean"):
    """Day-clustered bootstrap of the fill mean: resample days with replacement. Returns (lo, hi, P(mean>0))."""
    y = np.asarray(y, float); day = np.asarray(day)
    if len(y) == 0:
        return np.nan, np.nan, np.nan
    codes, uniq = pd.factorize(day)
    s = np.bincount(codes, weights=y); n = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(draws, len(uniq)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def boot_diff_p(y, mask, day, sign, draws=4000, seed=9):
    """Day bootstrap P(sign × (mean_zone − mean_rest) > 0)."""
    y = np.asarray(y, float); mask = np.asarray(mask, bool)
    codes, uniq = pd.factorize(np.asarray(day))
    sz = np.bincount(codes, weights=y * mask, minlength=len(uniq)); nz = np.bincount(codes, weights=mask.astype(float), minlength=len(uniq))
    sr = np.bincount(codes, weights=y * ~mask, minlength=len(uniq)); nr = np.bincount(codes, weights=(~mask).astype(float), minlength=len(uniq))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(draws, len(uniq)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sz[idx].sum(1) / nz[idx].sum(1) - sr[idx].sum(1) / nr[idx].sum(1)
    d = d[np.isfinite(d)]
    return float((sign * d > 0).mean()) if len(d) else np.nan


def tstats(Z, Y):
    """Welch t of zone (Z==1) vs rest for every column of Z (n×k) against every column of Y (n×p) → k×p."""
    Z = Z.astype(float); n1 = Z.sum(0)[:, None]; n0 = (1 - Z).sum(0)[:, None]
    s1 = Z.T @ Y; s0 = (1 - Z).T @ Y
    q1 = Z.T @ (Y ** 2); q0 = (1 - Z).T @ (Y ** 2)
    m1 = s1 / n1; m0 = s0 / n0
    v1 = (q1 - n1 * m1 ** 2) / np.maximum(n1 - 1, 1); v0 = (q0 - n0 * m0 ** 2) / np.maximum(n0 - 1, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)


def block_draws(y, blocks, nperm, seed):
    """Day-block null: each block's outcome vector replaced by a randomly permuted donor block's (shuffled, cycled to size)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(np.asarray(blocks))
    idx = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    nb = len(idx); out = np.empty((len(y), nperm))
    for p in range(nperm):
        perm = rng.permutation(nb)
        col = np.empty(len(y))
        for k in range(nb):
            donor = y[idx[perm[k]]]
            donor = donor[rng.permutation(len(donor))]
            col[idx[k]] = np.resize(donor, len(idx[k]))
        out[:, p] = col
    return out


def fmt_row(y, day):
    lo, hi, p = day_boot(y, day)
    y = np.asarray(y, float)
    return dict(N=len(y), days=len(set(day)), WR=round(100 * (y > 0).mean(), 1) if len(y) else np.nan,
                avg=round(y.mean(), 3) if len(y) else np.nan, ci=f"[{lo:+.3f}, {hi:+.3f}]", P_pos=round(p, 3), be_wr=round(100 * breakeven_wr(y), 1))


def md_table(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append("" if pd.isna(v) else f"{v:.{floatfmt}f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)
