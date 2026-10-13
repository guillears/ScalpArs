#!/usr/bin/env python3
"""Shared loader + screen engine for the YR6 two-year winners-vs-losers study, AREA = OTHERS
(FLIP-short, Spike-Fade, SURGE_LONG, BullRun-Long, BearRun-Short). Brief: reports/YR6_WL_STUDY_BRIEF.md (binding).

Units (window-units rule):
  · MACRO variables (BTC / ETH / breadth / market volume / door stamps) → DAY units: one observation per UTC day
    (day-mean of the variable, day-mean of pct_new over every fill of every seed that day).
  · PAIR variables → PAIR-DAY units: one observation per (pair, UTC day) — the 3 seeds' copies of the same trade
    collapse into one unit, so seed duplication never inflates N.
Statistic (same for the real data and the shuffled null): z = (mean_zone − mean_rest) / (sd_all · sqrt(1/n_zone + 1/n_rest)) on
units; zones need ≥ MIN_UNITS units on each side. Null: permute the unit outcomes (≥ 200 shuffles), recompute every test
(2D: the top-40 column selection is redone inside every shuffle, so the null pays the same selection cost as the real screen).
Thresholds (median / terciles) are set on the DISCOVERY regime and frozen for confirmation.
Cache-only, no network, read-only on every input.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
SLEEVES = ["FLIP-short", "Spike-Fade", "SURGE_LONG", "BullRun-Long", "BearRun-Short"]
REGIMES = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
MIN_UNITS = 10

EXCLUDE = {"entry_price", "entry_fee", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_slippage_pct",
           "entry_bracket_max_leverage", "entry_bracket_cap_notional", "entry_strategy", "entry_order_type",
           "entry_btc_regime_started_at", "entry_surge_trigger_at", "entry_frenzy_spike_at"}
MACRO_PREFIX = ("entry_btc_", "entry_br_", "entry_bear_", "entry_eth_", "entry_surge_btc", "entry_surge_gvol")
MACRO_EXACT = {"entry_bull_pct", "entry_bear_pct", "entry_global_volume_ratio", "entry_macro_trend", "entry_long_heat_flags",
               "entry_btc_regime"}


def is_macro(c):
    return c in MACRO_EXACT or c.startswith(MACRO_PREFIX)


def load():
    d = pd.read_csv(FILLS, low_memory=False)
    d = d[d.sleeve.isin(SLEEVES)].copy()
    d["t"] = pd.to_datetime(d.t)
    d["day"] = d.t.dt.floor("D")
    d["y"] = pd.to_numeric(d.pct_new, errors="coerce")
    d["win"] = (d.y > 0).astype(float)
    d["pd_key"] = d.pair + "|" + d.day.dt.strftime("%Y-%m-%d")
    return d


def top50_flag(d):
    """pair-day in top-50 or prior-day top-50 (2025 + 2026 universe files)."""
    u = pd.concat([pd.read_csv(os.path.join(REP, "backtest_cache", "universe_daily_top50.csv")),
                   pd.read_csv(os.path.join(REP, "backtest_cache_2025", "universe_daily_top50.csv"))])
    u["day"] = pd.to_datetime(u.date_ms, unit="ms").dt.floor("D")
    s = set(zip(u.pair, u.day))
    return np.array([(p, dd) in s or (p, dd - pd.Timedelta(days=1)) in s for p, dd in zip(d.pair, d.day)])


def candidate_columns(d):
    """numeric entry_* columns (+ 0/1 flags) and categorical entry_* columns usable for this frame."""
    num, cat = [], []
    for c in [c for c in d.columns if c.startswith("entry_") and c not in EXCLUDE]:
        x = pd.to_numeric(d[c], errors="coerce")
        if x.notna().sum() >= 0.5 * len(d) and x.nunique() >= 2:
            num.append(c)
        elif x.notna().sum() == 0 and d[c].notna().sum() >= 0.5 * len(d) and 2 <= d[c].nunique() <= 12:
            cat.append(c)
    return num, cat


def units(g, col, macro):
    """collapse fills to analysis units: DAY for macro, PAIR-DAY for pair variables. Returns (x, y, key)."""
    key = "day" if macro else "pd_key"
    x = pd.to_numeric(g[col], errors="coerce")
    h = pd.DataFrame({"k": g[key].values, "x": x.values, "y": g.y.values}).dropna(subset=["x"])
    a = h.groupby("k").agg(x=("x", "mean"), y=("y", "mean"))
    return a


def bootstrap_diff(g, mask, n=2000, seed=7):
    """day-block bootstrap of (avg zone − avg rest) at fill level. Returns (diff, P(diff>0))."""
    rng = np.random.default_rng(seed)
    days = g.day.values
    ud, inv = np.unique(days, return_inverse=True)
    nd = len(ud)
    y = g.y.values
    m = np.asarray(mask, bool)
    s1 = np.bincount(inv, weights=y * m, minlength=nd); c1 = np.bincount(inv, weights=m.astype(float), minlength=nd)
    s0 = np.bincount(inv, weights=y * ~m, minlength=nd); c0 = np.bincount(inv, weights=(~m).astype(float), minlength=nd)
    diff = s1.sum() / max(c1.sum(), 1e-9) - s0.sum() / max(c0.sum(), 1e-9)
    idx = rng.integers(0, nd, size=(n, nd))
    w = np.zeros((n, nd)); np.add.at(w, (np.repeat(np.arange(n), nd), idx.ravel()), 1)
    a1 = (w @ s1) / np.maximum(w @ c1, 1e-9); a0 = (w @ s0) / np.maximum(w @ c0, 1e-9)
    ok = ((w @ c1) > 0) & ((w @ c0) > 0)
    dd = (a1 - a0)[ok]
    return diff, float((dd > 0).mean()) if len(dd) else np.nan


def day_ci(g, n=4000, seed=11):
    """avg % with a day-clustered 95 % CI (resample days)."""
    rng = np.random.default_rng(seed)
    a = g.groupby("day").y.agg(["sum", "size"])
    if len(a) < 2:
        return g.y.mean(), np.nan, np.nan
    idx = rng.integers(0, len(a), size=(n, len(a)))
    s = a["sum"].values[idx].sum(1) / a["size"].values[idx].sum(1)
    return g.y.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def zone_specs(xd, col, macro):
    """zone definitions frozen on discovery-regime unit values xd: sign / median / outer terciles (+ 0/1 flags)."""
    xs = xd.dropna()
    out = []
    if xs.nunique() <= 2:
        v = sorted(xs.unique())
        if len(v) == 2:
            out.append((col, f"=={v[1]:g}", "eq", v[1]))
        return out
    if (xs < 0).any() and (xs > 0).any():
        out += [(col, "<0", "lt", 0.0), (col, ">=0", "ge", 0.0)]
    med = float(xs.median()); t1, t2 = float(xs.quantile(1 / 3)), float(xs.quantile(2 / 3))
    out += [(col, f"<med({med:.4g})", "lt", med), (col, f">=med({med:.4g})", "ge", med),
            (col, f"<=T1({t1:.4g})", "le", t1), (col, f">=T2({t2:.4g})", "ge", t2)]
    return out


def apply_zone(x, op, v):
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        if op == "lt": m = x < v
        elif op == "ge": m = x >= v
        elif op == "le": m = x <= v
        elif op == "eq": m = np.isclose(x, v)
        else: raise ValueError(op)
    return m & ~np.isnan(x)


def screen_family(A_list, Y, nperm, rng):
    """A_list: list of (spec, x_units, scored_mask) sharing the SAME unit index; Y: unit outcomes (n,).
    Returns real z per test, per-test permutation p (two-sided), and the null max|z| distribution."""
    n = len(Y)
    Z = np.zeros((len(A_list)))
    Mz = np.zeros((len(A_list), n)); Ms = np.zeros((len(A_list), n))
    for i, (spec, m, s) in enumerate(A_list):
        Mz[i] = m & s; Ms[i] = s
    perms = np.vstack([Y] + [rng.permutation(Y) for _ in range(nperm)])  # (P+1)×n
    n1 = Mz.sum(1); nS = Ms.sum(1); n0 = nS - n1
    S1 = perms @ Mz.T; SS = perms @ Ms.T          # (P+1)×tests
    Q = (perms ** 2) @ Ms.T
    mu = SS / nS; var = Q / nS - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-12))
    m1 = S1 / n1; m0 = (SS - S1) / n0
    Zall = (m1 - m0) / (sd * np.sqrt(1 / n1 + 1 / n0))
    Zr = Zall[0]; Zn = Zall[1:]
    p = ((np.abs(Zn) >= np.abs(Zr)[None, :]).sum(0) + 1) / (nperm + 1)
    maxnull = np.abs(Zn).max(1)
    return Zr, p, maxnull, Zn
