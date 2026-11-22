#!/usr/bin/env python3
"""ROUTE_B_PEAK vs REAL EXIT — REAL fills re-priced under TODAY's TP / stop (2026-11-22 deep-review fix; CLAUDE.md: re-simulate under
the CURRENT exit stack before any counterfactual). The booked REAL EXH fills ran TP 10 and the HT TP fills ran TP 5; today EXH TP = 7 %
(price, cap) and HT TP = 6.063 % gross (booked here as 6.063 − 0.09 = 5.97 net). Walk each REAL fill's 1m path (booked life + 30 bars after the booked exit):
  today'  = TP if the bar's favourable extreme reaches today's TP first (net = TP − 0.09 fees, maker-side TP, no slippage),
            else the booked stop (same minute as booked); an HT TP fill booked at +5 keeps walking after its booked exit to TP 6.063 or
            the −4 % stop (net −4 − 0.09 − 0.10 slip); unresolved at +30 bars → marked at the last close (flagged).
  signal' = the add-on exit (route_b_peak_1121_eval.cf arming: ≥ ½ TP and the signal firing) but only if it fires BEFORE today's TP /
            stop minute; else today'.  Signals are only evaluated inside the booked life (blind spot for the extended HT walks).
EXH stops: the ATR stop (cap 6 %) is taken as booked. EXH TP is the 7 % cap (an ATR TP below 7 % is possible — blind spot).
Output: reports/ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_today.csv. Research only, UNVERIFIED.
Usage: venv/bin/python -I scripts/route_b_peak_1122_reprice_today.py (after route_b_peak_1122_vs_real_exit.py)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_b_peak_1121_eval as EV  # noqa: E402
import route_b_peak_1122_vs_real_exit as VR  # noqa: E402

TODAY_TP = {"EXHAUSTION_SHORT": 7.0, "FRENZY_WILLY_HT": 6.063}
HT_STOP = 4.0
FEE = 0.09
SLIP = 0.10
VARS = EV.L1 + EV.CANDLE + ["C_NEWEXT15", "C_FASTMOVE", "C_OI_FALL", "C_CAPLIQ_NOT_FAST", "C_HALF_TP"]


def walk_today(tr, P):
    m, s = tr["m"], tr["s"]
    e = float(m["entry_px"])
    tp = TODAY_TP[m["sleeve"]]
    t = P.t.values
    hi = P.h.values.astype(float); lo = P.l.values.astype(float); cl = P.c.values.astype(float)
    fav = np.where(s > 0, (hi / e - 1) * 100, (1 - lo / e) * 100)
    adv = np.where(s > 0, (1 - lo / e) * 100, (hi / e - 1) * 100)
    e0 = m["entry_ms"] - m["entry_ms"] % EV.MIN
    x0 = m["exit_ms"] - m["exit_ms"] % EV.MIN
    booked_tp = "TP" in str(m["exit_reason"]).upper()
    for i in np.where(t > e0)[0]:
        if fav[i] >= tp:
            return tp - FEE, i, "TP"
        if t[i] >= x0 and not (booked_tp and m["sleeve"] == "FRENZY_WILLY_HT"):
            return float(m["net_pct"]), i, "BOOKED_" + str(m["exit_reason"])
        if t[i] > x0 and adv[i] >= HT_STOP:
            return -HT_STOP - FEE - SLIP, i, "STOP"
    i = len(t) - 1
    return s * (cl[i] / e - 1) * 100 - FEE - SLIP, i, "UNRESOLVED"


def main():
    D = pd.read_pickle(os.path.join(EV.DATA, "panels.pkl"))
    rows = []
    pans = [pn for pn in D["panels"] if pn.get("ok")]
    T = [(EV.build_trade(pn), pn["P"]) for pn in pans]
    T = [(tr, P) for tr, P in T if tr is not None and tr["m"]["cohort"] == "REAL" and tr["m"]["sleeve"] in TODAY_TP]
    for sl in TODAY_TP:
        G = [x for x in T if x[0]["m"]["sleeve"] == sl]
        VR.controls([g[0] for g in G])
        for tr, P in G:
            m = tr["m"]
            tnet, ti, tkind = walk_today(tr, P)
            t_ms = P.t.values[ti]
            row = dict(sleeve=sl, pair=m["pair"], booked=float(m["net_pct"]), booked_exit=m["exit_reason"], today=tnet, today_kind=tkind,
                       hold_today=(t_ms + EV.MIN - m["entry_ms"]) / EV.MIN)
            X = TODAY_TP[sl] / 2
            for v in VARS:
                di = np.where(tr["dec"])[0]
                f = tr["V"][v] & tr["avail"][v]
                k = [i for i in di if tr["cpl"][i] >= X and f[i] and tr["t"][i] < t_ms]
                if k:
                    i = k[0]
                    slip = tr["exitslip"][i] if tr["exitslip"][i] == tr["exitslip"][i] else SLIP
                    row[v] = tr["cpl"][i] - 2 * EV.TAKER - slip
                    row[v + "_hold"] = (tr["t"][i] + EV.MIN - m["entry_ms"]) / EV.MIN
                else:
                    row[v] = tnet
                    row[v + "_hold"] = row["hold_today"]
            rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(EV.REP, "ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_today.csv"), index=False)
    pd.set_option("display.width", 260)
    print(R[["sleeve", "pair", "booked", "booked_exit", "today", "today_kind", "hold_today", "C_HALF_TP", "CAPITULATION_LIQ", "LH2", "AGAINST_BURST", "TAKER_FADE", "BOOK_FLIP"]].round(2).to_string(index=False))
    out = []
    for sl, g in R.groupby("sleeve"):
        for v in ["booked", "today"] + VARS:
            x = g[v]
            h = g["hold_today"] if v == "today" else (g[v + "_hold"] if v + "_hold" in g else np.nan)
            out.append(dict(sleeve=sl, exit=v, N=len(g), changed=int((g[v] != g["today"]).sum()) if v != "booked" else np.nan,
                            WR=(x > 0).mean() * 100, avg=x.mean(), d_vs_today=x.mean() - g["today"].mean(),
                            hold_h=(h.mean() / 60) if isinstance(h, pd.Series) else np.nan))
    O = pd.DataFrame(out).sort_values(["sleeve", "avg"], ascending=[True, False])
    O.to_csv(os.path.join(EV.REP, "ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_today_summary.csv"), index=False)
    print(O.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
