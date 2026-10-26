"""Build the NEW market-impulse (≤ entry), during-trade (descriptive) and early-state (decidable) features for every
MOMENTUM LONG fill of the three blocks. Cache-only. Output: reports/ML_WL_1015_features.csv (one row per fill, keyed by
block, pair, opened_at, seed). Definitions: reports/ML_WL_2026-10-15_PREREG.txt.

    venv/bin/python -W ignore scripts/study_ml_wl_1015_features.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_wl_1015_common import (FEAT_CSV, MIN1, MIN5, Klines, load_blocks, load_universe, universe_for)

K = Klines()
UDAYS, UKEYS = load_universe()


def ret(p1, p0):
    return (p1 / p0 - 1) * 100 if (np.isfinite(p1) and np.isfinite(p0) and p0 > 0) else np.nan


def breadth(univ, T, mins):
    """(share up, mean return, std of returns) of the universe over the prior `mins` minutes at time T (5m bars)."""
    rs = []
    for p in univ:
        a = K.arr(p, T, MIN5)
        if a is None:
            continue
        r = ret(K.px(p, T, MIN5, a), K.px(p, T - mins * 60_000, MIN5, a))
        if np.isfinite(r):
            rs.append(r)
    if len(rs) < 25:
        return np.nan, np.nan, np.nan, len(rs)
    rs = np.array(rs)
    return (rs > 0).mean(), rs.mean(), rs.std(), len(rs)


def agg_vol(univ, T):
    rec = base = 0.0; n = 0
    for p in univ:
        r = K.qvol_sum(p, T, 3)
        b = K.qvol_sum(p, T, 288, offset_bars=3)
        if np.isfinite(r) and np.isfinite(b) and b > 0:
            rec += r; base += b / 96; n += 1
    return rec / base if n >= 25 and base > 0 else np.nan


def feats(row):
    t = int(pd.Timestamp(row.ts_open).value // 10**6)
    pair = row.pair
    f = {}
    # ---- BTC (1m, last closed bar <= t)
    b0 = K.px("BTCUSDT", t, MIN1)
    for m in (5, 15, 30, 60):
        f[f"btc_r{m}"] = ret(b0, K.px("BTCUSDT", t - m * 60_000, MIN1))
    f["btc_vol5_x24h"] = K.vol_ratio("BTCUSDT", t, 1)
    # ---- breadth (5m, previous-day top-50)
    univ, uday = universe_for(UDAYS, UKEYS, t)
    f["univ_day_lag_d"] = (t - t % 86_400_000 - uday) / 86_400_000 if uday is not None else np.nan
    up15, m15, sd15, n15 = breadth(univ, t, 15)
    up30, _, _, _ = breadth(univ, t, 30)
    up60, m60, _, _ = breadth(univ, t, 60)
    up30_h, _, _, _ = breadth(univ, t - 3600_000, 30)
    f.update(br_up15=up15, br_up30=up30, br_up60=up60, br_up30_chg1h=up30 - up30_h if np.isfinite(up30_h) else np.nan,
             br_ret15=m15, br_ret60=m60, disp15=sd15, br_n=n15, agg_vol15_x24h=agg_vol(univ, t))
    # ---- pair (1m returns; 5m volume)
    p0 = K.px(pair, t, MIN1)
    f["pair_r15"] = ret(p0, K.px(pair, t - 15 * 60_000, MIN1))
    f["pair_r60"] = ret(p0, K.px(pair, t - 60 * 60_000, MIN1))
    f["rs15"] = f["pair_r15"] - f["btc_r15"]
    f["rs60"] = f["pair_r60"] - f["btc_r60"]
    f["pair_vol15_x24h"] = K.vol_ratio(pair, t, 3)
    # ---- during trade (DESCRIPTIVE) and early state (decidable at t+k)
    ep = float(row.entry_price)
    ct = int(pd.Timestamp(row.closed_ts).value // 10**6)
    for k in (1, 3, 5, 15):
        T = t + k * 60_000
        pk = K.px(pair, T, MIN1)
        f[f"pnl_m{k}"] = ret(pk, ep)
        if k in (1, 3, 5):
            f[f"e{k}_open"] = float(ct > T)
            f[f"e{k}_pnl"] = f[f"pnl_m{k}"] if ct > T else np.nan
            f[f"e{k}_btc"] = ret(K.px("BTCUSDT", T, MIN1), b0) if ct > T else np.nan
    for k in (5, 15):
        T = t + k * 60_000
        f[f"btc_d{k}"] = ret(K.px("BTCUSDT", T, MIN1), b0)
        # breadth mean return from the last closed 5m bar before entry to the last closed 5m bar <= t+k
        rs = []
        for p in univ:
            a = K.arr(p, t, MIN5)
            if a is None:
                continue
            r = ret(K.px(p, T, MIN5, a), K.px(p, t, MIN5, a))
            if np.isfinite(r):
                rs.append(r)
        f[f"br_d{k}"] = np.mean(rs) if len(rs) >= 25 else np.nan
    f["e5_br"] = f["br_d5"] if f.get("e5_open") == 1.0 else np.nan
    return f


def main():
    B = load_blocks()
    rows = []
    t0 = time.time()
    for name, df in B.items():
        if not len(df):
            continue
        K.clear()
        for i, r in enumerate(df.itertuples(index=False)):
            f = feats(r)
            f.update(block=name, pair=r.pair, opened_at=str(r.opened_at), seed=getattr(r, "seed", 0))
            rows.append(f)
            if (len(rows) % 500) == 0:
                print(f"  {len(rows)} fills · {time.time() - t0:.0f}s", flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(FEAT_CSV, index=False)
    print(f"wrote {FEAT_CSV}: {len(F)} rows")
    cov = F.groupby("block")[[c for c in F.columns if c not in ("block", "pair", "opened_at", "seed")]].apply(
        lambda g: g.notna().mean().round(3))
    print(cov.T.to_string())


if __name__ == "__main__":
    main()
