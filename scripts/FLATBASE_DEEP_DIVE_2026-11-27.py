#!/usr/bin/env python3
"""FLATBASE_DEEP_DIVE_2026-11-27 — FLATBASE_SHORT after 1 / 6 live winners (+ open shorts never working) (operator: "we are only missing a deep
dive on Flatbase! results are terrible so far"). Research only, no bot change.
ALREADY DONE today (not re-run): no-progress / hold-time stop (21 cells, all lose) · real-fill W/L vs shuffled null (nothing) · backtest check
of the batch losers' stamps (flip) · BTC 30 d drawdown / 24 h low (FB's BEST zones) · rebuild through 11-27 (the backtest takes and loses the
same setups). Prior: OOS_REAL WL 11-17 · DL 601 funding filter · 603 TP grid · 616 funding rules · 622 caps · 649 base trend / entry timing.
NEW here — the DIST_ZONE_2D question asked of FLATBASE: does coin-level CROWDING (funding, top-trader L/S, taker L/S, OI growth) separate
FB losers, overall and in the crash zone? PRE-REGISTERED before results: the 545 backtest fills on today's stack walked at today's exit
(reports/FLATBASE_NOPROGRESS_STOP_2026-11-27_data/base.csv, BT rows) ⨝ the rebuild's features (wl_features.csv). Z = BTC prior-day close ≤ −20 %
below its 30 d high. For each variable: terciles (all markets) and Z ∧ low / high half (median of all fills), BOTH sides; family-wise luck by a
pick-the-best null (labels shuffled 1,000×, the most negative cell mean among cells with N ≥ 15). Breakeven WR from the cohort's own win / loss.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
sys.argv = sys.argv[:1]
sp = importlib.util.spec_from_file_location("dd", os.path.join(HERE, "DIST_DEEP_DIVE_2026-11-27.py"))
DD = importlib.util.module_from_spec(sp); sp.loader.exec_module(DD)
HOL = DD.HOL
VARS = [("funding_rate_pct", "funding"), ("toptrader_ls", "top-trader L/S"), ("taker_ls", "taker buy/sell"), ("oi_chg_4h", "OI 4 h change"),
        ("oi_chg_24h", "OI 24 h change")]


def main():
    b = pd.read_csv(os.path.join(R, "reports", "FLATBASE_NOPROGRESS_STOP_2026-11-27_data", "base.csv"))
    b = b[b.src == "BT"]
    w = pd.read_csv(os.path.join(R, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    K = b.merge(w[["key"] + [v for v, _ in VARS]], left_on="ep", right_on="key", how="left")
    K["entry_ms"] = K.te; K["day"] = K.te // 86_400_000; K["win"] = K.net > 0
    dd = DD.btc_dd()
    K["dd30"] = pd.to_datetime(K.te, unit="ms").dt.floor("D").map(dd)
    Z = (K.dd30 <= -20).fillna(False).values
    wn, ls = K.net[K.win], K.net[~K.win]
    be = -ls.mean() / (wn.mean() - ls.mean()) * 100
    print(f"FB backtest {len(K)} · WR {K.win.mean()*100:.1f} % · avg {K.net.mean():+.2f} · breakeven WR {be:.1f} % · zone fills {Z.sum()} "
          f"({K.net[Z].mean():+.2f}) vs rest {K.net[~Z].mean():+.2f}")
    cells = {}
    for v, lab in VARS:
        x = pd.to_numeric(K[v], errors="coerce"); ok = x.notna().values
        print(f"  coverage {v}: {ok.mean()*100:.0f} %")
        lo, hi = x.quantile([1 / 3, 2 / 3]); med = x.median()
        cells[f"ALL · {lab} low third"] = ok & (x <= lo).values
        cells[f"ALL · {lab} high third"] = ok & (x > hi).values
        cells[f"Z ∧ {lab} below median"] = Z & ok & (x < med).values
        cells[f"Z ∧ {lab} at/above median"] = Z & ok & (x >= med).values
    rows = []
    for name, m in cells.items():
        C = K[m]
        if not len(C):
            continue
        rows.append(dict(cell=name, N=len(C), days=C.day.nunique(), WR=C.win.mean() * 100, avg=C.net.mean(),
                         DESIGN=C.net[C.te < HOL].mean(), HOLDOUT=C.net[C.te >= HOL].mean(), rest_avg=K.net[~m].mean(),
                         DCR_after=DD.path(K.net[~m], K.day[~m])[0]))
    T = pd.DataFrame(rows)
    print(f"sleeve today DCR {DD.path(K.net, K.day)[0]:.2f} %/day")
    print(T.round(2).to_string(index=False))
    rng = np.random.default_rng(651)
    masks = [m for m in cells.values() if m.sum() >= 15]
    obs = min(K.net.values[m].mean() for m in masks)
    nb = [min(rng.permutation(K.net.values)[m].mean() for m in masks) for _ in range(1000)]
    print(f"luck: most negative cell {obs:+.2f} · shuffled 5th pct {np.percentile(nb, 5):+.2f} → family-wise p {np.mean(np.array(nb) <= obs):.3f} "
          f"({len(masks)} cells)")
    # funding dose-response all markets (live losers had funding −0.13 … −0.20 %)
    f = pd.to_numeric(K.funding_rate_pct, errors="coerce")
    K["fb"] = pd.cut(f, [-100, -0.1, -0.03, -0.005, 0.0049, 0.0051, 100], labels=["≤ −0.10 %", "−0.10…−0.03", "−0.03…−0.005", "−0.005…0", "= base", "> base"])
    print("\nfunding buckets (% per settlement; live losers −0.19 / −0.13 / −0.19 / +0.003 / +0.011)\n" +
          K.groupby("fb", observed=True).agg(N=("net", "size"), WR=("win", "mean"), avg=("net", "mean")).round(3).to_string())
    T.to_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "fb_crowding_2d.csv"), index=False)


if __name__ == "__main__":
    main()
