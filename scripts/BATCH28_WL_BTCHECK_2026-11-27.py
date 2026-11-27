#!/usr/bin/env python3
"""BATCH28 W/L — backtest cross-check of the real-fill candidates (2026-11-27, research only).
The real cohorts are thin (EXH 22 · DIST 10 · FB 34 · HT 28 · WILLY 26); each sleeve's top real-fill separators are re-read on its 2-year
backtest pool at TODAY's exit: high vs low half (median) and the outer terciles, ALL / DESIGN (< 2025-07) / HOLDOUT, + per-half sign count.
A candidate counts only if the backtest agrees in sign in BOTH blocks and in ≥ 5 / 7 halves (the backtest refutes, never proves alone).
EXH: TP_GRID F7 net (EXH_SPIKE_AGE loader) ⨝ ENTRY_WL features · FB: FLATBASE_NOPROGRESS base walk (today's exit) ⨝ the OOS_REAL rebuild.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)


def table(D, col, sign, label):
    """sign +1: the real fills say HIGH is better; −1: LOW is better. → one row: better-half minus worse-half avg net per scope."""
    v = pd.to_numeric(D[col], errors="coerce")
    D = D[v.notna()].assign(v=v[v.notna()])
    med = D.v.median()
    good = (D.v > med) if sign > 0 else (D.v <= med)
    out = dict(sleeve=label, var=col, real_says=("high better" if sign > 0 else "low better"), N=len(D), med=round(float(med), 4))
    for sc, m in (("ALL", np.ones(len(D), bool)), ("DESIGN", D.entry_ms < HOL), ("HOLDOUT", D.entry_ms >= HOL)):
        m = np.asarray(m)
        out[sc] = D.net[m & good].mean() - D.net[m & ~good].mean()
    hv = [D.net[(D.half == h) & good].mean() - D.net[(D.half == h) & ~good].mean() for h in sorted(D.half.unique())]
    out["halves_agree"] = f"{sum(x > 0 for x in hv)}/{len(hv)}"
    lo, hi = D.v.quantile([1 / 3, 2 / 3])
    gt, bt = (D.v > hi, D.v <= lo) if sign > 0 else (D.v <= lo, D.v > hi)
    out["tercile_gap"] = D.net[gt].mean() - D.net[bt].mean()
    out["agrees"] = bool(out["DESIGN"] > 0 and out["HOLDOUT"] > 0 and sum(x > 0 for x in hv) >= 5)
    return out


def exh():
    sp = importlib.util.spec_from_file_location("sa", os.path.join(HERE, "EXH_SPIKE_AGE_2026-11-26.py"))
    SA = importlib.util.module_from_spec(sp); sp.loader.exec_module(SA)
    D, _ = SA.load_bt()
    K = SA.kept(D)
    E = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "exh.pkl"))
    B = pd.read_csv(os.path.join(R, "reports", "EXH_BREADTH_2026-11-25_trades.csv"), usecols=["pair", "entry_ms", "bear"])
    K = K.merge(E.drop(columns=["net", "half"], errors="ignore"), on=["pair", "entry_ms"], how="left").merge(B, on=["pair", "entry_ms"], how="left")
    # real-fill top separators (sweep, booked P&L) → their backtest analog
    cands = [("vmult15", -1, "entry_frenzy_vol_trend (proxy: 15m vol mult)"), ("bear", -1, "entry_bear_pct (breadth)"),
             ("btc_ret1h", +1, "entry_btc_dist_from_ema13 (proxy: BTC 1h ret)"), ("mcap", +1, "entry_mcap_usd"),
             ("hrs_since_spike", +1, "entry_frenzy_hours"), ("rsi15", -1, "entry_exh_rsi_prev (proxy)"), ("btc_ret1d", +1, "entry_btc_1d_ret_pct"),
             ("frz_24h", -1, "market heat: Frenzy flags 24 h"), ("st_stops24h", -1, "own stops in the prior 24 h")]
    return [dict(table(K, c, s, "EXH"), real_var=lbl) for c, s, lbl in cands if c in K]


def fb():
    b = pd.read_csv(os.path.join(R, "reports", "FLATBASE_NOPROGRESS_STOP_2026-11-27_data", "base.csv"))
    b = b[b.src == "BT"]
    w = pd.read_csv(os.path.join(R, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    D = b.merge(w, left_on="ep", right_on="key", how="left", suffixes=("", "_w"))
    D["entry_ms"] = D.te
    D["half"] = pd.to_datetime(D.te, unit="ms").dt.year.astype(str) + "-H" + np.where(pd.to_datetime(D.te, unit="ms").dt.month <= 6, "1", "2")
    cands = [("off_peak_pct", +1, "entry_flatbase_off_peak_pct"), ("flat14", +1, "entry_flatbase_flat14_pct"),
             ("btc_ret_24h", +1, "entry_btc_off24h / trend (proxy: BTC 24 h ret)"), ("peak_age_h", +1, "entry_flatbase_peak_age_h"),
             ("breadth_pos24", -1, "market up-breadth 24 h"), ("btc_bull_1d", +1, "BTC daily bull"), ("funding_rate_pct", +1, "entry funding")]
    return [dict(table(D, c, s, "FB"), real_var=lbl) for c, s, lbl in cands if c in D]


if __name__ == "__main__":
    T = pd.DataFrame(exh() + fb())
    T.to_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "wl_backtest_check.csv"), index=False)
    print(T[["sleeve", "real_var", "var", "real_says", "N", "ALL", "DESIGN", "HOLDOUT", "halves_agree", "tercile_gap", "agrees"]].round(2).to_string(index=False))
