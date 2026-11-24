#!/usr/bin/env python3
"""H4 ask vacuum (Route B trigger, LONG) — theory · TP / stop / hold grid · winners vs losers. Research only, INTERIM (PREREG Addenda 6–8).

Operator 2026-11-24: "run winners vs losers here" · "focus on theory of what the trade is about" · "different TP and SL levels? what's the optimal".
Cohort = the interim Q3 cohort (scripts/route_b_signals/interim_q3.py: final, book-priced, signal ≤ data end − 4 h) for H4_ask_vacuum and
its matched random-minute controls. Everything is in-sample on 4.7 days of one regime: a line found here is HOLDOUT-only (Addendum 8).

Run:  venv/bin/python -I scripts/H4_ASK_VACUUM_STUDY_2026-11-24.py   → reports/H4_ASK_VACUUM_STUDY_2026-11-24_{grid,wl,wl_theory}.csv
"""
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b_signals"))
import interim_q3 as I  # noqa: E402
import exits as X  # noqa: E402
import sources as SRC  # noqa: E402

KEY = "H4_ask_vacuum"
OUT = os.path.join(ROOT, "reports", "H4_ASK_VACUUM_STUDY_2026-11-24")
TPS = (1, 2, 3, 5, 8, 12)
SLS = (1, 2, 3, 5, 8)
HOLDS = (60, 240, 720, 1440)
SEED = 20261124
MIN_MS = 60_000


def walk_row(r, now_ms):
    """extended grid for one row (entry = its stored h_entry_ms open, its stored slips). → {(tp, sl, hold): net or NaN}."""
    e_ms = int(r.h_entry_ms)
    k = SRC.klines(r.pair, e_ms, e_ms + (max(HOLDS) + 5) * MIN_MS, now_ms)
    out = {}
    if k is None or not len(k):
        return out
    k = k[k.open_ms >= e_ms]
    if not len(k) or int(k.open_ms.iloc[0]) != e_ms:
        return out
    fund, fok = SRC.funding_history(r.pair, e_ms, e_ms + max(HOLDS) * MIN_MS, now_ms)
    o, h, l, c, t = (k[x].to_numpy(float) for x in ("o", "h", "l", "c", "open_ms"))
    for hold in HOLDS:
        for a in TPS:
            for b in SLS:
                w = X.walk(o[:hold], h[:hold], l[:hold], c[:hold], t[:hold], float(o[0]), 1, a, b, hold, funding=fund,
                           entry_slip_pct=max(float(r.h_slip_entry_pct), 0), exit_slip_pct=max(float(r.h_slip_exit_pct), 0),
                           entry_ms=e_ms)
                out[(a, b, hold)] = w["net_pct"] if (w["final"] and len(o) >= hold and fok) else math.nan
    return out


def grid_table(t, c, now_ms):
    rows_t = {r.trigger_id: walk_row(r, now_ms) for r in t.itertuples()}
    rows_c = [(r.trigger_id, r.cd, walk_row(r, now_ms)) for r in c.itertuples()]
    res = []
    days = sorted(set(t.day))
    mid = days[len(days) // 2]
    tday = dict(zip(t.trigger_id, t.day))
    for hold in HOLDS:
        for a in TPS:
            for b in SLS:
                tv = pd.Series({k: v.get((a, b, hold), math.nan) for k, v in rows_t.items()}).dropna()
                cv = pd.DataFrame([(tid, v.get((a, b, hold), math.nan)) for tid, _, v in rows_c], columns=["tid", "net"]).dropna()
                both = set(tv.index) & set(cv.tid)
                tm, cm = tv[list(both)], cv[cv.tid.isin(both)].net
                h1 = tv[[i for i in tv.index if tday[i] < mid]]
                h2 = tv[[i for i in tv.index if tday[i] >= mid]]
                res.append(dict(tp=a, sl=b, hold_h=hold / 60, N=len(tv), WR=(tv > 0).mean(), mean_net=tv.mean(),
                                ctrl_mean=cm.mean(), delta=tm.mean() - cm.mean(), half1=h1.mean(), half2=h2.mean(),
                                kelly_floor=I.kelly_floor(tv.to_numpy())))
    return pd.DataFrame(res), rows_t


# ── theory lines (written before the W/L numbers were computed; direction = the side predicted to WIN for the long) ──
THEORY = [
    ("T1 one-sided: only the ask is thin, the bid is normal", "ask_bid_ratio_05", "LOW", "HOLDOUT-only (algebraically imb_05, §0)"),
    ("T2 asks were EATEN (real buying volume), not just pulled", "rb_traded_vs_norm_15", "HIGH", ""),
    ("T2b buyers are the aggressors", "rb_taker_buy_share_15", "HIGH", "HOLDOUT-only (taker share, §0)"),
    ("T3 generic thinning: the bid is thin too (a vol regime, no direction)", "rb_bid_usd_05", "HIGH", ""),
    ("T3b wide spread = makers stepping back on both sides", "rb_spread_bps", "LOW", ""),
    ("T4 sellers exhausted: the coin fell into the vacuum (capitulation over)", "rb_ret_60m_pct", "LOW", ""),
    ("T4b … or late: it already ran up (chasing)", "rb_ret_15m_pct", "LOW", ""),
    ("T5 market tide: BTC rising", "mk_btc_ret_60m_pct", "HIGH", ""),
    ("T5b breadth: most coins rising", "mk_breadth_up_60", "HIGH", ""),
    ("T6 crowded shorts = squeeze fuel (funding negative)", "rb_funding_pct", "LOW", ""),
    ("T6b perp below spot index (premium negative)", "rb_premium_pct", "LOW", ""),
    ("T7 longs flushed: long liquidations in the last 15 min", "rb_liq_long_15", "HIGH", ""),
]


def tstat(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return math.nan
    return (a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b) + 1e-12)


def split_stats(x, y, rows_day):
    ok = x.notna() & y.notna()
    x, y, d = x[ok], y[ok], rows_day[ok]
    if ok.sum() < 20:
        return None
    cut = x.median()
    hi = x > cut
    lo = ~hi
    return dict(N=int(ok.sum()), cut=cut, hi_N=int(hi.sum()), lo_N=int(lo.sum()), hi_mean=y[hi].mean(), lo_mean=y[lo].mean(),
                hi_days=d[hi].nunique(), lo_days=d[lo].nunique(), t=tstat(y[hi], y[lo]))


def rotation_null(df, cols, ycol, n=1000):
    """rotate the outcome series (rows sorted by time) by ≥ one day of rows; family max |t| over all columns. PREREG §6 Q1 style."""
    df = df.sort_values("ms").reset_index(drop=True)
    y = df[ycol].to_numpy(float)
    day_rows = int(np.ceil(len(df) / max(1, df.day.nunique())))
    rng = np.random.default_rng(SEED)
    X_ = {c: df[c].to_numpy(float) for c in cols}
    mx = np.empty(n)
    for i in range(n):
        s = rng.integers(day_rows, max(day_rows + 1, len(df) - day_rows))
        yr = np.roll(y, s)
        best = 0.0
        for c in cols:
            x = X_[c]
            ok = ~np.isnan(x) & ~np.isnan(yr)
            if ok.sum() < 20:
                continue
            hi = x[ok] > np.median(x[ok])
            t = tstat(yr[ok][hi], yr[ok][~hi])
            if t == t:
                best = max(best, abs(t))
        mx[i] = best
    return mx


def main():
    T, C, meta = I.load()
    data_end = pd.Timestamp(meta["data_end"])
    now_ms = int(pd.Timestamp.now("UTC").value // 1_000_000)
    t, c = I.cohort(T, C, data_end)
    t, c = t[t.trigger == KEY].copy(), c[c.trigger == KEY].copy()

    G, rows_t = grid_table(t, c, now_ms)
    G.to_csv(OUT + "_grid.csv", index=False)

    # outcomes for W/L: booked net at the interim cell (TP 5 / SL 5, 4 h) · WORKED (+1 % before −2 %, 4 h) · REACH (fwd 240 m best ≈ gross)
    t["net_55"] = t["g_tp5_sl5_net"]
    t["worked"] = (t["g_tp1_sl2_net"] > 0).astype(float).where(t["g_tp1_sl2_net"].notna())
    t["fwd240"] = t["fwd_240m_pct"]
    t["ask_bid_ratio_05"] = t.rb_ask_usd_05 / t.rb_bid_usd_05
    t["depth_05_total"] = t.rb_ask_usd_05 + t.rb_bid_usd_05
    num = [col for col in t.columns if col.startswith(("rb_", "mk_", "oi_", "top_", "global_", "taker_", "btc_d", "fb_"))
           and pd.api.types.is_numeric_dtype(t[col]) and t[col].notna().mean() >= 0.3 and t[col].nunique() > 3]
    num = [col for col in num if col not in ("rb_mid", "rb_mark", "btc_d_ema50", "btc_d_ema200")] + ["ask_bid_ratio_05", "depth_05_total"]
    W = []
    for ycol in ("worked", "net_55", "fwd240"):
        mx = rotation_null(t, num, ycol)
        bar = float(np.percentile(mx, 95))
        for col in num:
            s = split_stats(t[col], t[ycol], t.day)
            if s:
                W.append(dict(outcome=ycol, var=col, coverage=t[col].notna().mean(), null95_abs_t=bar, **s))
    W = pd.DataFrame(W)
    W.to_csv(OUT + "_wl.csv", index=False)
    th = []
    for name, col, good, note in THEORY:
        for ycol in ("worked", "net_55", "fwd240"):
            r = W[(W["var"] == col) & (W.outcome == ycol)]
            if not len(r):
                continue
            r = r.iloc[0]
            pred = (r.hi_mean - r.lo_mean) * (1 if good == "HIGH" else -1)
            th.append(dict(theory=name, var=col, good_side=good, note=note, outcome=ycol, N=r.N, hi_mean=r.hi_mean, lo_mean=r.lo_mean,
                           good_minus_bad=pred, t=r.t, null95=r.null95_abs_t, holds=pred > 0))
    pd.DataFrame(th).to_csv(OUT + "_wl_theory.csv", index=False)
    print("done", len(t), "triggers", len(c), "controls")


if __name__ == "__main__":
    main()
