#!/usr/bin/env python3
"""ROUTE_B_PEAK vs REAL EXIT (2026-11-22, operator: "test it vs our current real exit for each trade for each sleeve").
Every pre-registered ROUTE_B_PEAK variable (8 order-flow + 3 candle) and the 4 price-only controls, used as an EXIT: close at the
first minute the signal fires once the trade is ≥ half its TP in profit (the frozen arming rule of route_b_peak_1121_eval.cf),
market close = 2 × taker + measured book slippage (0.10 % when no book). Compared with each fill's REAL booked net % (today's exit).
Every fill counts (fills that never arm show Δ = 0). Per sleeve × cohort (REAL fills / SHADOW priced blocked signals).
Research only, UNVERIFIED (2 days of Route B data). In-trade funding is not charged to the new exit (blind spot, ≤ 0.02 % on these holds).
Usage: venv/bin/python -I scripts/route_b_peak_1122_vs_real_exit.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_b_peak_1121_eval as EV  # noqa: E402

OUT = os.path.join(EV.REP, "ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22.csv")
OUT_T = os.path.join(EV.REP, "ROUTE_B_PEAK_VS_REAL_EXIT_2026-11-22_trades.csv")
RNG = np.random.default_rng(1122)


def controls(G):
    """same construction as route_b_peak_1121_posthoc ① (q90 per group)."""
    if not G:
        return
    r3 = np.concatenate([(tr["fav"] - EV.lagged(tr["fav"], 3))[tr["dec"]] for tr in G])
    q90 = np.nanquantile(r3, 0.9) if np.isfinite(r3).any() else np.inf
    for tr in G:
        n, i0 = tr["n"], tr["i0"]
        rmx = EV.run_max(np.where(tr["ins"], tr["fav"], np.nan), i0)
        newext = np.zeros(n, bool)
        for i in range(i0, n):
            w = tr["fav"][max(i0, i - 14): i + 1]
            newext[i] = bool(len(w)) and np.nanmax(w) >= rmx[i] - 1e-12
        fast = (tr["fav"] - EV.lagged(tr["fav"], 3)) >= q90
        oi = tr["base"]["oi"]
        tr["V"]["C_NEWEXT15"] = newext
        tr["V"]["C_FASTMOVE"] = fast
        tr["V"]["C_OI_FALL"] = (oi - EV.lagged(oi, 15)) <= 0
        tr["V"]["C_CAPLIQ_NOT_FAST"] = tr["V"]["CAPITULATION_LIQ"] & ~fast
        tr["V"]["C_HALF_TP"] = np.ones(n, bool)          # benchmark: plain market exit the first minute the trade is ≥ half its TP
        for k in ("C_NEWEXT15", "C_FASTMOVE", "C_HALF_TP"):
            tr["avail"][k] = np.ones(n, bool)
        tr["avail"]["C_CAPLIQ_NOT_FAST"] = tr["avail"]["CAPITULATION_LIQ"]
        tr["avail"]["C_OI_FALL"] = ~np.isnan(oi - EV.lagged(oi, 15))


def day_ci(C, n=2000):
    days = C.day.unique()
    if len(days) < 2:
        return np.nan, np.nan
    by = {d: C.d[C.day == d].values for d in days}
    s = []
    for _ in range(n):
        pick = RNG.choice(days, len(days))
        s.append(np.concatenate([by[d] for d in pick]).mean())
    return np.quantile(s, 0.025), np.quantile(s, 0.975)


def main():
    D = pd.read_pickle(os.path.join(EV.DATA, "panels.pkl"))
    T = [x for x in (EV.build_trade(pn) for pn in D["panels"] if pn.get("ok")) if x is not None]
    VARS = EV.L1 + EV.CANDLE + ["C_NEWEXT15", "C_FASTMOVE", "C_OI_FALL", "C_CAPLIQ_NOT_FAST", "C_HALF_TP"]
    rows, trows = [], []
    sleeves = sorted({tr["m"]["sleeve"] for tr in T})
    for sl in sleeves:
        for co in ("REAL", "SHADOW"):
            G = [tr for tr in T if tr["m"]["sleeve"] == sl and tr["m"]["cohort"] == co]
            if not G:
                continue
            controls(G)
            for v in VARS:
                C = EV.cf(G, v)
                C["pair_"] = [tr["m"]["pair"] for tr in G]
                # holding time (min): booked exit vs the rule's exit minute (same arming as EV.cf)
                dur_b, dur_a = [], []
                for tr, dd in zip(G, C.d):
                    m = tr["m"]
                    db = (m["exit_ms"] - m["entry_ms"]) / EV.MIN
                    da = db
                    if dd != 0:
                        tp = m.get("tp_pct")
                        X = tp / 2 if tp == tp else 2.5
                        di = np.where(tr["dec"])[0]
                        f = tr["V"][v] & tr["avail"][v]
                        k = [i for i in di if tr["cpl"][i] >= X and f[i]]
                        da = (tr["t"][k[0]] + EV.MIN - m["entry_ms"]) / EV.MIN
                    dur_b.append(db); dur_a.append(da)
                C["dur_b"], C["dur_a"] = dur_b, dur_a
                fired = C.d != 0
                lo, hi = day_ci(C)
                loo = [C.d.drop(i).mean() for i in C.index] if len(C) > 1 else [np.nan]
                worst_pair = C[fired].groupby("pair_").d.sum().abs().max() if fired.any() else 0.0
                tot = C.d[fired].abs().sum()
                rows.append(dict(sleeve=sl, cohort=co, var=v, kind=("order-flow" if v in EV.L1 else "candle" if v in EV.CANDLE else "price-only"),
                                 n=len(C), fired=int(fired.sum()), today=C.today.mean(), new=C.new.mean(), d=C.d.mean(),
                                 ci_lo=lo, ci_hi=hi, better=int((C.d > 0).sum()), worse=int((C.d < 0).sum()),
                                 saved=C.d[C.d > 0].sum(), chopped=-C.d[C.d < 0].sum(),
                                 loo_min=np.nanmin(loo), loo_max=np.nanmax(loo), top_pair_share=(worst_pair / tot if tot else np.nan),
                                 days=C.day.nunique(), wr_b=(C.today > 0).mean() * 100, wr_a=(C.new > 0).mean() * 100,
                                 dur_b=C.dur_b.mean(), dur_a=C.dur_a.mean()))
                for r in C.itertuples():
                    trows.append(dict(sleeve=sl, cohort=co, var=v, pair=r.pair_, exit=r.kind, today=r.today, new=r.new, d=r.d, dur_b=r.dur_b, dur_a=r.dur_a))
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    pd.DataFrame(trows).to_csv(OUT_T, index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 30)
    print(R.round(3).to_string(index=False))
    # per-fill REAL table, today's exit vs every variable that fired
    TT = pd.DataFrame(trows)
    RT = TT[(TT.cohort == "REAL")]
    print("\nREAL fills — today vs new per fired variable")
    print(RT[RT.d != 0].round(2).to_string(index=False))
    print("\nREAL fills (all)")
    print(RT.drop_duplicates(["sleeve", "pair", "today"])[["sleeve", "pair", "exit", "today"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
