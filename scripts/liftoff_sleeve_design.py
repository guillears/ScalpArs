#!/usr/bin/env python3
"""🛫 LIFT-OFF sleeve design (operator, 2026-09-30 — MOVR: volume exploded for ~8 h while price sat flat on its EMA200, then
lifted away from the EMA200 on rising volume and ran +90 %; the bot only ranked it top-50 at 04:30 UTC, after the move began).

PRE-DECLARED (written before any result; MOVR 09-30 is after the data end 09-28 → not in the evidence). Same machinery,
exits, units, controls, split and PASS bar as scripts/runaway_sleeve_design.py v2 (imported, not copied). A SEPARATE
pre-registration (its own 64-cell Bonferroni), designed after the runaway grid was read in the same session — state that when reading it:
  triggers (closed 5m bar, one event per pair per 8 h)
    VOL_LEAD     last-1 h quote volume ≥ 10× (prior-288 median 5m volume × 12)  ∧  |pair 1 h return| ≤ 3 %
                 LONG if close > EMA200(5m), SHORT if close < EMA200        (volume arrives BEFORE the price move)
    EMA200_LIFT  the prior 24 h hugged the EMA200 (|close/EMA200 − 1| ≤ 3 % on ≥ 80 % of bars)
                 ∧ the gap now ≥ +2 % (LONG) / ≤ −2 % (SHORT) and widened ≥ 1 point over the last hour
                 ∧ bar quote volume ≥ 3× prior-288 median                    (the lift-off away from the EMA200)
  universe     TOP50 (the bot's list) · NEXT50 (ranks 51–100 by 24 h volume — where MOVR sat before its pump)
  entry        NEXT (the next 5m bar open) only
  exits        MOMENTUM / SURGE / BR2ATR / WIDE · hold 240 / 480 min
  cells        2 triggers × 2 sides × 2 universes × 4 exits × 2 holds = 64 → Bonferroni 1 − 0.05/64 (same as runaway)
Usage: venv/bin/python scripts/liftoff_sleeve_design.py   → reports/LIFTOFF_SLEEVE_GRID_v3_2026-10-01.csv (+ the per-trade FILLS csv, not in git)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runaway_sleeve_design as R  # noqa: E402

V2, BAR, H = R.V2, R.BAR, R.H
OUT = os.path.join(V2.ROOT, "reports", "LIFTOFF_SLEEVE_GRID_v3_2026-10-01.csv")


def detect():
    out = []
    for p, d in V2.D.items():
        if len(d) < 30 * 288:
            continue
        e200 = d.c.ewm(span=200, adjust=False).mean()
        gap = (d.c / e200 - 1) * 100
        med = d.qvol.shift(1).rolling(288, min_periods=280).median()
        v1h = d.qvol.rolling(12).sum()
        r1h = (d.c / d.c.shift(12) - 1) * 100
        hug = (gap.abs() <= 3).astype(float).shift(1).rolling(288, min_periods=280).mean()
        e13 = d.c.ewm(span=13, adjust=False).mean()
        trig = {
            ("VOL_LEAD", "LONG"): (v1h >= 10 * med * 12) & (r1h.abs() <= 3) & (d.c > e200),
            ("VOL_LEAD", "SHORT"): (v1h >= 10 * med * 12) & (r1h.abs() <= 3) & (d.c < e200),
            ("EMA200_LIFT", "LONG"): (hug >= 0.8) & (gap >= 2) & (gap - gap.shift(12) >= 1) & (d.qvol >= 3 * med),
            ("EMA200_LIFT", "SHORT"): (hug >= 0.8) & (gap <= -2) & (gap - gap.shift(12) <= -1) & (d.qvol >= 3 * med),
        }
        born = d.index[0]
        for (name, side), m in trig.items():
            last = -10**18
            m = m.fillna(False)
            m = m & ~m.shift(1, fill_value=False)          # v3: a False→True EDGE (VOL_LEAD is a persistent state — it re-fired every 8 h)
            for t in d.index[m.values]:
                if t - born < 30 * 86_400_000:
                    continue
                ranked = V2.universe(int(t), 100)               # universe BEFORE the cooldown (v2), cooldown per universe
                uni = "TOP50" if p in ranked[:50] else "NEXT50" if p in ranked[50:100] else None
                if uni is None or t - last < R.COOL:           # v3: one cooldown per (pair, trigger, side), whatever the universe
                    continue
                last = t
                out.append((int(t), p, side, name, float(d.atrp.loc[t]), float(e13.loc[t]), uni))
    ev = pd.DataFrame(out, columns=["t", "pair", "side", "trig", "atr", "ema13", "uni"])
    ev["th"] = ev.trig + "·" + ev.uni                     # the grid key R.table groups on
    return ev.reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 280); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)
    ev = detect()
    print(f"events: {len(ev)}\n{ev.groupby(['trig', 'uni', 'side']).size().to_string()}")
    F = R.simulate(ev, entries_wanted=("NEXT",)); C = R.simulate(ev, control=True, entries_wanted=("NEXT",))
    C.to_csv(OUT.replace("GRID", "CONTROL"), index=False)
    T = R.table(F, C).rename(columns={"th": "cell"})
    print("drops:", R.DROPS)
    T.to_csv(OUT, index=False); F.to_csv(OUT.replace("GRID", "FILLS"), index=False)
    print(T.to_string(index=False))
    print(f"\nPASS: {(T.PASS == '✅').sum()} of {len(T)} cells · grid → {OUT}")
