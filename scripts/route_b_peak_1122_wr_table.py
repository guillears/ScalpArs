#!/usr/bin/env python3
"""ROUTE_B_PEAK vs REAL EXIT — before → after per sleeve: WR · avg % · net $ · avg hold (2026-11-22, operator).
Reads reports/ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_trades.csv (scripts/route_b_peak_1122_vs_real_exit.py, run first).
$: REAL fills at their own notional (export pnl ÷ pnl %); SHADOW at the sleeve's median REAL notional (one size).
Output: reports/ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_wr.csv. Research only, UNVERIFIED.
Usage: venv/bin/python -I scripts/route_b_peak_1122_wr_table.py"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "route_b_signals"))
import sources as S  # noqa: E402

RULES = None  # every variable in the trades file (8 order-flow · 3 candle · 4 price-only controls · plain ½-TP)


def main():
    O, _ = S.load_orders(since=pd.Timestamp("2026-11-19", tz="UTC"))
    O = O[O.status == "CLOSED"].copy()
    O["base"] = O.pnl.astype(float) / (O.pnl_percentage.astype(float) / 100)
    O["k"] = O.entry_strategy + "|" + O.pair + "|" + O.pnl_percentage.astype(float).round(2).astype(str)
    T = pd.read_csv(os.path.join(ROOT, "reports", "ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_trades.csv"))
    T["k"] = T.sleeve + "|" + T.pair + "|" + T.today.round(2).astype(str)
    T["base"] = T.k.map(dict(zip(O.k, O.base)))
    miss = T[(T.cohort == "REAL") & T.base.isna()].k.unique()
    if len(miss):
        print("REAL fills without a $ match:", list(miss))
    size = T[T.cohort == "REAL"].groupby("sleeve").base.median()
    rows = []
    for (sl, co), G in T.groupby(["sleeve", "cohort"]):
        for v in (RULES or sorted(T["var"].unique())):
            g = G[G["var"] == v]
            b = g.base if co == "REAL" else pd.Series(size.get(sl, np.nan), index=g.index)
            ch = g.d != 0
            rows.append(dict(sleeve=sl, cohort=co, rule=v, N=len(g), changed=int(ch.sum()),
                             wr_before=(g.today > 0).mean() * 100, wr_after=(g.new > 0).mean() * 100,
                             avg_before=g.today.mean(), avg_after=g.new.mean(),
                             usd_before=(g.today * b / 100).sum(), usd_after=(g.new * b / 100).sum(),
                             hold_min_before=g.dur_b.mean(), hold_min_after=g.dur_a.mean(),
                             hold_min_changed_before=g.dur_b[ch].mean(), hold_min_changed_after=g.dur_a[ch].mean()))
    X = pd.DataFrame(rows)
    X.to_csv(os.path.join(ROOT, "reports", "ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_wr.csv"), index=False)
    pd.set_option("display.width", 260)
    print(X.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
