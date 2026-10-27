#!/usr/bin/env python3
"""FADE REGIME W/L study 2026-10-26 — shared data + statistics (research only, no network, no writes)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_fade_regime_wl_data as DD     # noqa: E402

R = DD.R
PFX = os.path.join(R, "reports/FADE_REGIME_WL_2026-10-26")
GROUPS = ["G1 B1", "G2 B2", "G3 B3", "G4 B4-B11", "G5 B12-B16", "G6 B17-B24"]
RNG = np.random.default_rng(20261026)
BOOT_N = 4000

ML_DROP = {"btc_bar_lag_min", "br_n_scored"}
EXCL = {"entry_price", "entry_fee", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_bracket_max_leverage",
        "entry_bracket_cap_notional", "entry_order_type", "entry_strategy", "entry_btc_regime", "entry_macro_trend",
        "entry_btc_regime_started_at", "entry_frenzy_spike_at", "entry_surge_trigger_at", "entry_order_lag_s", "entry_book_drift_pct",
        "entry_book_spread_pct", "entry_fill_vs_book_pct"}
MARKET_STAMP_PREFIX = ("entry_btc_", "entry_eth_", "entry_global_", "entry_bull_pct", "entry_bear_pct", "entry_alt_", "entry_breadth",
                       "entry_market_", "entry_long_heat")
REGIMES = ["HEALTHY_BEAR", "STRONG_BEAR", "CHOPPY_WEAK", "CHOPPY_FLAT", "BEAR_EXHAUSTED", "HEALTHY_BULL", "STRONG_BULL", "CHOPPY_STRONG"]


def load(with_blocked=False):
    A = DD.load(with_bt=False, with_blocked=with_blocked)
    F = pd.read_csv(PFX + "_fill_regime.csv").drop(columns=["blk", "pair", "opened_at"])
    A = A.merge(F, on="key", how="left")
    A = derive(A)
    return A


def load_bt():
    A = DD.load(with_bt=True, with_blocked=False)
    A = A[A.blk == "BT"].copy()
    F = pd.read_csv(PFX + "_bt_regime.csv.gz").drop(columns=["blk", "pair", "opened_at"])
    A = A.merge(F, on="key", how="left", suffixes=("", "_rg"))
    return derive(A)


def num(A, c):
    x = A[c]
    if x.dtype == bool:
        return x.astype(float)
    if x.dtype == object:
        s = x.astype(str).str.lower()
        if s.dropna().isin(["true", "false", "nan", "none"]).all():
            return s.map({"true": 1.0, "false": 0.0})
    return pd.to_numeric(x, errors="coerce")


def derive(A):
    A = A.copy()
    n = lambda c: num(A, c) if c in A else pd.Series(np.nan, index=A.index)
    A["d_rsi_jump"] = n("entry_rsi") - n("entry_rsi_prev")
    A["d_di_spread"] = n("entry_pos_di") - n("entry_neg_di")
    A["d_log_vol24h"] = np.log10(n("entry_pair_volume_24h_usd"))
    A["d_log_mcap"] = np.log10(n("entry_mcap_usd"))
    A["d_btc_rsi_rising"] = (n("entry_btc_rsi") > n("entry_btc_rsi_prev")).astype(float).where(n("entry_btc_rsi_prev").notna())
    A["d_btc_adx_rising"] = (n("entry_btc_adx") > n("entry_btc_adx_prev")).astype(float).where(n("entry_btc_adx_prev").notna())
    A["d_btc_rsi1h_delta"] = n("entry_btc_rsi_1h") - n("entry_btc_rsi_1h_prev")
    A["d_breadth_diff"] = n("entry_bull_pct") - n("entry_bear_pct")
    A["d_fanr"] = n("entry_ema_gap_5_8") / n("entry_ema_gap_8_13").replace(0, np.nan)          # dashboard FanR
    A["d_adx_rising"] = (n("entry_adx") > n("entry_adx_prev")).astype(float).where(n("entry_adx_prev").notna())   # ADXDir
    A["d_rsi_rising"] = (n("entry_rsi") > n("entry_rsi_prev")).astype(float).where(n("entry_rsi_prev").notna())   # RSIDir
    A["hb_gap_zone"] = ((A.entry_btc_regime == "HEALTHY_BEAR") & (n("entry_btc_trend_gap_pct") >= -0.13)).astype(float).where(A.entry_btc_regime.notna())
    for lv in REGIMES:
        A["c_regime_" + lv] = (A.entry_btc_regime == lv).astype(float).where(A.entry_btc_regime.notna())
    for lv in ["BEARISH", "BULLISH", "NEUTRAL"]:
        if "entry_macro_trend" in A:
            A["c_macro_" + lv] = (A.entry_macro_trend == lv).astype(float).where(A.entry_macro_trend.notna())
    # pre-registered P4b interaction: high alt corr ∧ BTC 1 h up (cut = master median, computed once in the screen)
    return A


def columns(A):
    F = pd.read_csv(PFX + "_fill_regime.csv", nrows=2)
    own = [c for c in F.columns if c not in ("key", "blk", "pair", "opened_at", "uni_stale") and c not in ML_DROP]
    market_own = [c for c in own if not c.startswith(("p_", "s_", "f_"))]
    pair_own = [c for c in own if c.startswith(("p_", "s_", "f_"))]
    ent = []
    for c in A.columns:
        if (c.startswith("entry_") and c not in EXCL) or c.startswith("d_") or c.startswith("c_") or c == "hb_gap_zone":
            x = num(A, c)
            if x.notna().sum() >= 5 and x.nunique() >= 2:
                ent.append(c)
    mstamp = [c for c in ent if c.startswith(MARKET_STAMP_PREFIX) or c.startswith(("c_regime_", "c_macro_", "d_btc_", "d_breadth"))
              or c == "hb_gap_zone"]
    pstamp = [c for c in ent if c not in mstamp]
    return market_own, mstamp, pair_own, pstamp


NATURAL_ZERO = ("_r15m", "_r30m", "_r1h", "_r4h", "_r24h", "_r3d", "_r7d", "_r30d", "slp", "_c_e20", "_e20_e50", "_e50_e200", "_di_",
                "adxchg", "minus", "ethbtc", "ac5", "med_r", "funding", "1h_slope", "ema20_slope", "ema50_slope", "trend_gap",
                "dist_from_ema13", "1d_ret", "r72", "ema50_100", "4h_ema50_200", "p_r", "p_ema50_gap", "f_pair", "di_spread",
                "breadth_diff", "pair_ema20_ema50_gap", "1h_ema20_200", "prevbar", "rsi1h_delta", "btc_r", "s_run15", "s_fb_run")


def zones(x, c):
    """sign (natural zero) → median → terciles; [(label, kind, thr, mask)] on scored rows (mask = zone)."""
    s = np.isfinite(x); v = x[s]; out = []
    if len(v) < 6 or len(np.unique(v)) < 2:
        return out
    if set(np.unique(v)) <= {0.0, 1.0}:
        out.append(("==1", "eq1", 1.0, s & (x == 1)))
        return out
    if any(k in c for k in NATURAL_ZERO) and 0.05 < (v > 0).mean() < 0.95:
        out.append((">0", "gt", 0.0, s & (x > 0)))
    if c.endswith(("_x7d", "x60", "rv24_rv7")) or "vr15" in c:
        if 0.05 < (v > 1).mean() < 0.95:
            out.append((">1", "gt", 1.0, s & (x > 1)))
    m = float(np.median(v)); q1, q2 = (float(q) for q in np.quantile(v, [1 / 3, 2 / 3]))
    out.append((f">med({m:.4g})", "gt", m, s & (x > m)))
    out.append((f"<=q33({q1:.4g})", "le", q1, s & (x <= q1)))
    out.append((f">q67({q2:.4g})", "gt", q2, s & (x > q2)))
    return out


def apply_zone(x, kind, thr):
    s = np.isfinite(x)
    if kind == "eq1":
        return s & (x == 1), s
    if kind == "gt":
        return s & (x > thr), s
    if kind == "ge":
        return s & (x >= thr), s
    if kind == "lt":
        return s & (x < thr), s
    return s & (x <= thr), s


def st(y):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    return "0" if len(y) == 0 else f"{len(y)}·{(y > 0).mean() * 100:.0f}%·{y.mean():+.3f}"


def breakeven(y):
    y = np.asarray(y, float); w, l = y[y > 0], y[y <= 0]
    if len(w) == 0 or len(l) == 0:
        return np.nan
    return abs(l.mean()) / (w.mean() + abs(l.mean())) * 100


def day_boot_p_neg(y, day, R=BOOT_N, seed=7):
    """P(avg < 0) under a window(day)-clustered bootstrap."""
    df = pd.DataFrame({"y": np.asarray(y, float), "d": np.asarray(day)}).dropna()
    if len(df) == 0:
        return np.nan
    g = df.groupby("d").y.agg(["sum", "count"]); s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(g), size=(R, len(g)))
    return float((s[idx].sum(1) / c[idx].sum(1) < 0).mean())


def loss_share(y, key):
    y = np.asarray(y, float); key = np.asarray(key)
    neg = pd.Series(np.where(y < 0, -y, 0.0)).groupby(key).sum()
    tot = neg.sum()
    return (float(neg.max() / tot) if tot > 0 else 0.0), (neg.idxmax() if tot > 0 else "")


def top2_pair_share(y, pair):
    y = np.asarray(y, float)
    neg = pd.Series(np.where(y < 0, -y, 0.0)).groupby(np.asarray(pair)).sum().sort_values(ascending=False)
    return float(neg.head(2).sum() / neg.sum()) if neg.sum() > 0 else 0.0


def bar(y, day, pair, be):
    """CLAUDE.md expectancy bar on a would-be-BLOCKED cohort at 1×. returns dict + pass flag."""
    y = np.asarray(y, float)
    n = len(y); wr = (y > 0).mean() * 100 if n else np.nan
    p = day_boot_p_neg(y, day) if n else np.nan
    nd = len(set(day)); ds, dk = loss_share(y, day); ps, pk = loss_share(y, pair)
    ok = dict(wr=wr < be, p=p >= 0.95, days=nd >= 8, conc=(ds < 0.5 and ps < 0.5), n=n >= 15)
    return dict(N=n, WR=wr, avg=y.mean() if n else np.nan, P_neg=p, days=nd, top_day_share=ds, top_day=dk, top_pair_share=ps,
                top_pair=pk, top2_pair_share=top2_pair_share(y, pair), fails=",".join(k for k, v in ok.items() if not v) or "—",
                passes=all(ok.values()))
