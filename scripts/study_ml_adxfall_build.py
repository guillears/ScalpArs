#!/usr/bin/env python3
"""ML_ADXFALL_2D (2026-10-16) — feature builder. Research only (pre-registration reports/ML_ADXFALL_2D_PREREG.txt).

Input reports/ML_DT2D_FU_features.pkl (the ML downtrend-2D follow-up frame: all momentum longs — replay 2617/1171/521, master 191,
fresh 29; 106 parent + follow-up variables). Adds 5m CLOSED-BAR rebuilds (last 5m bar whose close ≤ fill time; cache only, the same
SRC list as study_ml_dt2d_build.py; only bars with real high/low; ≥ 60 contiguous bars before the bar):
  c_adx, c_adx_delta (vs 1 bar earlier; the DL 273 closed-bar ADX line), c_adx_delta3, c_pdi, c_ndi, c_dis (pdi − ndi), c_dis_chg,
  c_rsi (Wilder 14), c_rsi_dir, c_vol_r20 (last bar vol / mean of the 20 before), c_vol_tr3 (Σ vol last 3 / Σ prior 3),
  c_run_e20 (consecutive closed bars with close > EMA20, cap 288 = time since the move started), c_gap5_20, c_gap5_20_chg,
  c_ret6 / c_ret12 (30 / 60-min return to the last close), c_px_e5 (last close vs EMA5).
Stamp-derived: s_gap520_chg = entry_gap_5_20_signed_pct − entry_gap_5_20_prev_signed_pct, s_dis = entry_pos_di − entry_neg_di.
Zone flags: AF_stamp = entry_adx < entry_adx_prev (d_adx_dir < 0); AF_closed = c_adx_delta ≤ 0.
Output reports/ML_ADXFALL_features.pkl
    venv/bin/python -W ignore scripts/study_ml_adxfall_build.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_ml_dt2d_build import SRC   # noqa: E402

M5 = 300_000
OUT = os.path.join(ROOT, "reports/ML_ADXFALL_features.pkl")
FETCH = os.path.join(ROOT, "reports/ml_adxfall_fetch/5m")   # 9-symbol 5m top-up (fapi klines, closed bars) for the 10-12 → 10-16 fills
C_COLS = ["c_adx", "c_adx_delta", "c_adx_delta3", "c_pdi", "c_ndi", "c_dis", "c_dis_chg", "c_rsi", "c_rsi_dir", "c_vol_r20",
          "c_vol_tr3", "c_run_e20", "c_gap5_20", "c_gap5_20_chg", "c_ret6", "c_ret12", "c_px_e5"]


def load5(sym):
    fr = []
    for s in SRC + [FETCH]:
        f = os.path.join(s, f"{sym}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            if "h" not in d or "l" not in d:
                continue                                   # close-only cache: no ADX possible
            if "vol" not in d:
                d["vol"] = np.nan
            fr.append(d[["open_time", "h", "l", "c", "vol"]])
    if not fr:
        return None
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    return d


def feats(d):
    c, h, l, v = (d[k].astype(float) for k in ("c", "h", "l", "vol"))
    a = 1 / 14
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0)); ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
    ndi = 100 * ndm.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    adx = dx.ewm(alpha=a, adjust=False).mean()
    dc = c.diff()
    g = dc.clip(lower=0).ewm(alpha=a, adjust=False).mean(); ls = (-dc.clip(upper=0)).ewm(alpha=a, adjust=False).mean()
    rsi = 100 - 100 / (1 + g / ls)
    e5, e20 = c.ewm(span=5, adjust=False).mean(), c.ewm(span=20, adjust=False).mean()
    above = (c > e20).astype(int).values
    run = np.zeros(len(c)); r = 0
    for i, x in enumerate(above):
        r = min(r + 1, 288) if x else 0
        run[i] = r
    gap = (e5 / e20 - 1) * 100
    F = pd.DataFrame({
        "c_adx": adx, "c_adx_delta": adx.diff(), "c_adx_delta3": adx.diff(3), "c_pdi": pdi, "c_ndi": ndi, "c_dis": pdi - ndi,
        "c_dis_chg": (pdi - ndi).diff(), "c_rsi": rsi, "c_rsi_dir": rsi.diff(),
        "c_vol_r20": v / v.shift(1).rolling(20).mean(), "c_vol_tr3": v.rolling(3).sum() / v.shift(3).rolling(3).sum(),
        "c_run_e20": run, "c_gap5_20": gap, "c_gap5_20_chg": gap.diff(), "c_ret6": (c / c.shift(6) - 1) * 100,
        "c_ret12": (c / c.shift(12) - 1) * 100, "c_px_e5": (c / e5 - 1) * 100})
    F["T"] = d.open_time.values + M5                        # bar close time
    ok = (d.open_time.diff() == M5).astype(int)
    F["contig"] = ok.groupby((ok == 0).cumsum()).cumsum().values   # contiguous bars so far
    return F


def main():
    D = pd.read_pickle(os.path.join(ROOT, "reports/ML_DT2D_FU_features.pkl"))
    for k in C_COLS:
        D[k] = np.nan
    t0 = time.time()
    for k, (sym, idx) in enumerate(D.groupby("pair").groups.items()):
        d = load5(sym)
        if d is None or len(d) < 200:
            continue
        F = feats(d)
        t = D.loc[idx, "t_ms"].values.astype(np.int64)
        j = np.searchsorted(F["T"].values, t, side="right") - 1
        jj = np.clip(j, 0, None)
        ok = (j >= 0) & (t - F["T"].values[jj] < 2 * M5) & (F.contig.values[jj] >= 60)
        for c in C_COLS:
            D.loc[idx, c] = np.where(ok, F[c].values[jj], np.nan)
        if k % 100 == 0:
            print(f"  {k} pairs · {time.time() - t0:.0f}s", flush=True)
    n = lambda c: pd.to_numeric(D[c], errors="coerce")
    D["s_gap520_chg"] = n("entry_gap_5_20_signed_pct") - n("entry_gap_5_20_prev_signed_pct")
    D["s_dis"] = n("entry_pos_di") - n("entry_neg_di")
    D["AF_stamp"] = np.where(D.d_adx_dir.notna(), (D.d_adx_dir < 0).astype(float), np.nan)
    D["AF_closed"] = np.where(D.c_adx_delta.notna(), (D.c_adx_delta <= 0).astype(float), np.nan)
    D.to_pickle(OUT)
    print("wrote", OUT, D.shape)
    cov = D.groupby("block")[C_COLS + ["s_gap520_chg", "s_dis", "AF_stamp", "AF_closed"]].apply(lambda x: x.notna().mean()).T.round(3)
    print(cov.to_string())
    for b, g in D.groupby("block"):
        m = g.AF_stamp.notna() & g.AF_closed.notna()
        print(f"{b}: AF_stamp share {g.AF_stamp.mean():.3f} · AF_closed share {g.AF_closed.mean():.3f} · agree {(g.AF_stamp[m] == g.AF_closed[m]).mean():.3f} "
              f"· corr(stamp adx, closed adx) {n('entry_adx')[g.index].corr(g.c_adx):.3f}")


if __name__ == "__main__":
    main()
