#!/usr/bin/env python3
"""ML_DOWNTREND_2D follow-up (2026-10-16) — new coin features per reports/ML_DOWNTREND_2D_PREREG_ADDENDUM.txt.
Research only. Reads reports/ML_DT2D_features.pkl (parent study) + cached 5m klines; writes reports/ML_DT2D_FU_features.pkl.
All features from LAST CLOSED bars at the fill time (lookahead-free):
  n_rsi{1h,4h}, n_rsi{1h,4h}_dir, n_adx{1h,4h}, n_adx{1h,4h}_dir   (ta RSI14 / ADX14, dir = value − one closed bar earlier)
  n_px_{tf}_e{span}  price (last closed 5m close) vs EMA(span) of tf at its last closed bar, %; minus the 4 already in the 106 set.
    venv/bin/python -W ignore scripts/study_ml_dt2d_fu_build.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_ml_dt2d_build import SRC   # noqa: E402  (same cache list incl. reports/ml_dt2d_fetch/5m)

M5, H, H4 = 300_000, 3_600_000, 4 * 3_600_000
SPANS = [5, 8, 13, 20, 50, 100, 200]
SKIP = {("5m", 5), ("5m", 13), ("1h", 20), ("1h", 200)}     # already in the 106 set
OUT = os.path.join(ROOT, "reports/ML_DT2D_FU_features.pkl")
PX = [f"n_px_{tf}_e{s}" for tf in ("5m", "1h", "4h") for s in SPANS if (tf, s) not in SKIP]
OSC = [f"n_{k}{tf}{d}" for tf in ("1h", "4h") for k in ("rsi", "adx") for d in ("", "_dir")]


def k5(sym):
    fr = []
    for s in SRC:
        f = os.path.join(s, f"{sym}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            real = "h" in d and "l" in d
            if not real:
                d["h"] = d["c"]; d["l"] = d["c"]
            d["hl"] = real
            fr.append(d[["open_time", "h", "l", "c", "hl"]])
    if not fr:
        return None
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time")
    return d.set_index("open_time")


def since_from_miss(miss, tol):
    """bars since the end of the last missing run longer than tol bars (warm-up counter restarts after a real gap)."""
    m = pd.Series(np.asarray(miss, bool))
    run = m.groupby((~m).cumsum()).cumsum().values
    pos = np.where(run > tol, np.arange(len(m)), -1)
    return np.arange(len(m)) - np.maximum.accumulate(pos)


def regrid(d, step):
    """complete grid on open_time; missing bars ffilled (h=l=c=prev c); returns frame + missing flag + bars-since-segment-start."""
    idx = np.arange(d.index.min(), d.index.max() + step, step, dtype=np.int64)
    g = d.reindex(idx)
    miss = g.c.isna().values
    g["c"] = g.c.ffill()
    g["h"] = g.h.fillna(g.c); g["l"] = g.l.fillna(g.c)
    g["hl"] = g.hl.fillna(True).astype(bool)
    g["miss"] = miss
    g["since"] = since_from_miss(miss, {M5: 12, H: 3, H4: 1}[step])
    return g


def agg(m5, step):
    b = (m5.index.values // step) * step
    a = pd.DataFrame({"h": m5.h.values, "l": m5.l.values, "c": m5.c.values, "hl": m5.hl.values,
                      "miss": m5.miss.values}, index=b).groupby(level=0).agg(
        h=("h", "max"), l=("l", "min"), c=("c", "last"), hl=("hl", "all"), nmiss=("miss", "sum"), n=("c", "size"))
    n_per = step // M5
    a["miss"] = (a.nmiss > n_per // 4) | (a.n < n_per)          # bar counts as missing if > 25 % of its 5m bars were missing
    return a


def tf_feats(a, step, tag):
    """a: OHLC on a complete grid of tf `step` (open_time index). Returns frame indexed by bar CLOSE time."""
    out = pd.DataFrame(index=a.index.values + step)
    c, h, l = a.c, a.h, a.l
    miss42 = pd.Series(a.miss.values.astype(int)).rolling(42, min_periods=1).sum().values
    fake42 = pd.Series((~a.hl.values).astype(int)).rolling(42, min_periods=1).sum().values
    since = a.since.values
    okm = (miss42 <= (2 if step == H else 1)) & ~a.miss.values.astype(bool)
    rsi = RSIIndicator(close=c.reset_index(drop=True), window=14).rsi().values
    adx = ADXIndicator(high=h.reset_index(drop=True), low=l.reset_index(drop=True), close=c.reset_index(drop=True), window=14).adx().values
    rsi = np.where(okm & (since >= 100), rsi, np.nan)
    adx = np.where(okm & (since >= 100) & (fake42 == 0), adx, np.nan)
    out[f"rsi{tag}"] = rsi
    out[f"rsi{tag}_dir"] = rsi - np.r_[np.nan, rsi[:-1]]
    out[f"adx{tag}"] = adx
    out[f"adx{tag}_dir"] = adx - np.r_[np.nan, adx[:-1]]
    for s in SPANS:
        e = c.ewm(span=s, adjust=False).mean().values
        out[f"e{s}"] = np.where(okm & (since >= 2 * s), e, np.nan)
    return out


def build_sym(sym):
    d = k5(sym)
    if d is None or len(d) < 1000:
        return None
    m5 = regrid(d, M5)
    a1 = agg(m5, H); a1["since"] = since_from_miss(a1.miss.values, 3)
    a4 = agg(m5, H4); a4["since"] = since_from_miss(a4.miss.values, 1)
    # 5m frame: missing flag per bar
    m5["miss"] = m5.miss.astype(bool)
    f5 = pd.DataFrame(index=m5.index.values + M5)
    f5["px"] = np.where(m5.miss.values, np.nan, m5.c.values)
    miss42 = pd.Series(m5.miss.values.astype(int)).rolling(42, min_periods=1).sum().values
    for s in SPANS:
        e = m5.c.ewm(span=s, adjust=False).mean().values
        f5[f"e{s}"] = np.where((miss42 <= 2) & (m5.since.values >= 2 * s), e, np.nan)
    return f5, tf_feats(a1, H, "1h"), tf_feats(a4, H4, "4h")


def at(frame, t):
    """row of the last bar whose CLOSE time ≤ t (frame index = close time)."""
    i = np.searchsorted(frame.index.values, t, side="right") - 1
    return i


def fill_from_fetched(D):
    """ADX/RSI where the cached 5m is close-only: recompute from fetched 1h/4h klines (reports/ml_dt2d_fetch/{1h,4h}, real H/L,
    closed bars only: close time ≤ fill). Parity vs the cache build is printed where both exist; fills only NaN values."""
    rows = []
    for tf, step in (("1h", H), ("4h", H4)):
        d = os.path.join(ROOT, "reports/ml_dt2d_fetch", tf)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            sym = fn[:-4]
            k = pd.read_csv(os.path.join(d, fn)).sort_values("open_time").reset_index(drop=True)
            rsi = RSIIndicator(close=k.c, window=14).rsi().values
            adx = ADXIndicator(high=k.h, low=k.l, close=k.c, window=14).adx().values
            ok = np.arange(len(k)) >= 100
            F = pd.DataFrame({f"rsi{tf}": np.where(ok, rsi, np.nan), f"adx{tf}": np.where(ok, adx, np.nan)},
                             index=k.open_time.values + step)
            F[f"rsi{tf}_dir"] = F[f"rsi{tf}"].diff(); F[f"adx{tf}_dir"] = F[f"adx{tf}"].diff()
            sel = D.index[(D.pair == sym) & D.block.isin(["master", "fresh"])]
            if not len(sel):
                continue
            t = D.loc[sel, "t_ms"].values.astype(np.int64)
            i = at(F, t)
            okk = (i >= 0) & (t - F.index.values[np.clip(i, 0, None)] < step)
            for c in F.columns:
                v = np.where(okk, F[c].values[np.clip(i, 0, None)], np.nan)
                cur = D.loc[sel, f"n_{c}"].values.astype(float)
                both = np.isfinite(cur) & np.isfinite(v)
                if both.any():
                    rows.append((sym, c, int(both.sum()), float(np.nanmax(np.abs(cur[both] - v[both])))))
                D.loc[sel, f"n_{c}"] = np.where(np.isfinite(cur), cur, v)
    if rows:
        R = pd.DataFrame(rows, columns=["sym", "feat", "n_both", "max_abs_diff"])
        print("parity fetched-vs-cache (max |diff| by feature):")
        print(R.groupby("feat").agg(n=("n_both", "sum"), max_abs_diff=("max_abs_diff", "max")).round(3).to_string())


def main():
    D = pd.read_pickle(os.path.join(ROOT, "reports/ML_DT2D_features.pkl"))
    for c in PX + OSC:
        D[c] = np.nan
    t0 = time.time()
    for k, (sym, idx) in enumerate(D.groupby("pair").groups.items()):
        r = build_sym(sym)
        if r is None:
            continue
        f5, f1, f4 = r
        t = D.loc[idx, "t_ms"].values.astype(np.int64)
        i5 = at(f5, t)
        ok5 = (i5 >= 0) & (t - f5.index.values[np.clip(i5, 0, None)] <= 900_000)
        px = np.where(ok5, f5.px.values[np.clip(i5, 0, None)], np.nan)
        res = {}
        for tf, F, ii in (("5m", f5, i5), ("1h", f1, at(f1, t)), ("4h", f4, at(f4, t))):
            okk = ii >= 0
            j = np.clip(ii, 0, None)
            for s in SPANS:
                if (tf, s) in SKIP:
                    continue
                e = np.where(okk, F[f"e{s}"].values[j], np.nan)
                res[f"n_px_{tf}_e{s}"] = (px / e - 1) * 100
            if tf != "5m":
                for kk in ("rsi", "adx"):
                    for dd in ("", "_dir"):
                        res[f"n_{kk}{tf}{dd}"] = np.where(okk, F[f"{kk}{tf}{dd}"].values[j], np.nan)
        for c, v in res.items():
            D.loc[idx, c] = v
        if k % 50 == 0:
            print(f"  {k} pairs · {time.time() - t0:.0f}s", flush=True)
    fill_from_fetched(D)
    D.to_pickle(OUT)
    print("wrote", OUT, D.shape)
    cov = D.groupby("block")[PX + OSC].apply(lambda x: x.notna().mean()).T.round(3)
    print(cov.to_string())
    cov.to_csv(os.path.join(ROOT, "reports/ML_DT2D_FU_coverage.csv"))


if __name__ == "__main__":
    main()
