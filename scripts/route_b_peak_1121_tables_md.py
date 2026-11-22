#!/usr/bin/env python3
"""ROUTE_B_PEAK_STUDY 2026-11-21 — markdown tables for the report (per sleeve, layer 1 + candle comparators + price controls).
Usage: venv/bin/python -I scripts/route_b_peak_1121_tables_md.py > <scratch file>"""
import math
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = pd.read_pickle(os.path.join(ROOT, "reports", "ROUTE_B_SIGNALS_data", "peak_1121", "results.pkl"))
CT = pd.read_csv(os.path.join(ROOT, "reports", "ROUTE_B_PEAK_STUDY_2026-11-21_controls.csv"))
T = R["TAB"]


def wilson_hi(h, n, z=1.645):
    if n == 0:
        return float("nan")
    p = h / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + a) / d


def f(x, k=2):
    return "—" if x != x else f"{x:.{k}f}"


def verdict(r):
    ep = r.episodes
    if ep < 10:
        return "untestable-yet"
    if r.precision < 0.5:
        return "REFUTED at this N"
    if r.precision >= 0.6 and ep >= 15 and r.windows >= 8 and r.precision > r.null_q95:
        return "CANDIDATE"
    return "below bar"


for sl in ("EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FLATBASE_SHORT", "FRENZY_WILLY_HT", "POOLED_SHORTS", "POOLED_LONGS"):
    for co in ("REAL", "SHADOW", "REAL+SHADOW"):
        x = T[(T.sleeve == sl) & (T.cohort == co)]
        if not len(x) or x.trades.iloc[0] == 0:
            continue
        print(f"\n**{sl} · {co}** — eligible trades {x.trades.iloc[0]} · with a PEAK-zone decision minute {x.peak_trades.iloc[0]}\n")
        print("| variable | hits | false alarms | precision | 95 % upper | shuffled-null q95 | recall | windows | top pair share | "
              "fires at peak min / cont min | WENT_TP fired | GAVE_BACK fired | verdict |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in x.itertuples():
            ep = r.hits + r.false_alarms
            print(f"| {r.var}{' (candle)' if r.layer == 'candle' else ''} | {r.hits} | {r.false_alarms} | {f(r.precision)} | "
                  f"{f(wilson_hi(r.hits, ep))} | {f(r.null_q95)} | {f(r.recall)} | {r.windows} | {f(r.top_pair_share)} | "
                  f"{f(r.minute_rate_peak)} / {f(r.minute_rate_cont)} | {r.tp_fired}/{r.tp_n} | {r.gb_fired}/{r.gb_n} | {verdict(r)} |")
print("\n\n### controls\n")
print("| sleeve | cohort | variable | hits | false alarms | precision | null q95 | p(null ≥ obs) | peak min / cont min |")
print("|---|---|---|---|---|---|---|---|---|")
for r in CT.itertuples():
    print(f"| {r.sleeve} | {r.cohort} | {r.var} | {r.hits} | {r.false_alarms} | {f(r.precision)} | {f(r.null_q95)} | {f(r.p_null)} | "
          f"{f(r.minute_rate_peak)} / {f(r.minute_rate_cont)} |")
