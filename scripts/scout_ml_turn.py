"""👁 ML_MARKET_TURN — observe-only watch line for MOMENTUM LONG (DECISION_LOG 299, operator 2026-10-15). Rebuilt at batch review; no engine change.

Hypothesis (in-sample, from reports/ML_IMPULSE_2026-10-15.md + its deep review): the market-impulse SCORE (0–9 legs, frozen in
reports/ML_IMPULSE_frozen_params.json; scripts/study_ml_impulse_features.py) does not separate momentum longs by its LEVEL, but its
5-minute TURN did on the Jun–Oct master fills: TURN_UP (now ≥ 7, 5 min earlier < 7) 42 fills WR 93 % +0.475 · TURN_DN (now < 7,
earlier ≥ 7) 38 fills WR 47 % −0.288; replay same months weak same sign; 2025 / 2026-H1 absent.
Frozen definitions: "now" = 1m legs at the last whole minute before entry + 5m legs at the last 5-minute boundary before entry;
"earlier" = every leg at that 5-minute boundary minus 5 min. Cohort: MOMENTUM LONG, CLOSED, non-probe fills opened ≥ 2026-10-16 00:00 UTC
ONLY (the Jun–Oct master fills produced the hypothesis and never count). Evidence in DAY units.
Gate to OPEN a filter study (not a ship) — all of: TURN_DN N ≥ 15 on ≥ 8 distinct days · TURN_DN WR < the sleeve breakeven WR and avg < 0 at
day-bootstrap P ≥ 0.95 · no day / pair ≥ 50 % of the TURN_DN loss · TURN_UP − rest > 0 at P ≥ 0.90 · the 2026-H2 replay not opposite in
sign. Retire: TURN_DN reaches 15 fills with avg ≥ 0 or WR ≥ breakeven. 50 % haircut on any master effect size.

Usage (batch review): first extend the impulse grids to the batch end (scripts/study_ml_impulse_fetch.py + study_ml_impulse_features.py —
see their headers; the grids end 2026-10-15), then:  venv/bin/python scripts/scout_ml_turn.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "scripts"))
COHORT_START = "2026-10-16T00:00:00"
GATE_N, GATE_DAYS = 15, 8


def fresh_fills():
    import export_glob as EG
    fr = []
    for f in EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "pair", "direction", "status", "entry_strategy",
                                                                           "pnl_percentage", "cell_multiplier_source", "is_paper",
                                                                           "close_reason"))
        except Exception:
            continue
        fr.append(d)
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a = a[(a.entry_strategy.astype(str) == "MOMENTUM") & (a.direction == "LONG") & (a.status == "CLOSED")
          & ~a.cell_multiplier_source.astype(str).str.contains("PROBE") & (a.opened_at.astype(str) >= COHORT_START)]
    return a


def main():
    import study_ml_impulse_features as FX
    import study_ml_delay_common as DC
    a = fresh_fills()
    import scout_gatekit as _GK                                 # 🧰 (DECISION_LOG 331) operator-closed fills: not the bot's exit → out, listed
    a, opx = _GK.split_operator_closed(a) if len(a) else (a, a)
    a = a.reset_index(drop=True)
    print(f"ML_MARKET_TURN · cohort: momentum LONG closed non-probe fills opened ≥ {COHORT_START} UTC → {len(a)} · {_GK.operator_closed_line(opx)}")
    if not len(a):
        return
    a["o_ms"] = DC._ms(a.opened_at).astype(np.int64)
    t5 = (a.o_ms // FX.MIN5 * FX.MIN5).values
    t1 = (a.o_ms // FX.MIN1 * FX.MIN1).values
    need = set(t1.tolist()) | set(t5.tolist()) | set((t5 - FX.MIN5).tolist())
    g1 = pd.concat([c[c.index.isin(need)] for c in pd.read_csv(FX.OUT1, index_col=0, chunksize=500000)])
    g5 = pd.read_csv(FX.OUT5, index_col=0)
    g5 = g5[g5.index.isin(need)]
    frz = FX.load_frozen()
    five = FX.EXTRA5 + ["corr15"]

    def score(tt1, tt5):
        A = g1.reindex(tt1); A.index = range(len(tt1))
        B = g5[five].reindex(tt5); B.index = range(len(tt5))
        return FX.add_legs(pd.concat([A, B], axis=1), frz).SCORE.values

    now, early = score(t1, t5), score(t5 - FX.MIN5, t5 - FX.MIN5)
    miss = np.isnan(now) | np.isnan(early)
    if miss.any():
        print(f"  ⚠ {int(miss.sum())} fills outside the impulse grids — extend them (see the header) and re-run")
    a["turn"] = np.where(miss, "UNSCORED", np.where((now >= 7) & (early < 7), "TURN_UP", np.where((now < 7) & (early >= 7), "TURN_DN", "OTHER")))
    a["day"] = a.opened_at.astype(str).str[:10]
    p = pd.to_numeric(a.pnl_percentage, errors="coerce")
    for t, g in a.groupby("turn"):
        pp = p[g.index]
        print(f"  {t:9s} N {len(g):3d} · days {g.day.nunique():3d} · WR {(pp > 0).mean():.0%} · avg {pp.mean():+.3f} % · Σ {pp.sum():+.2f}")
    dn = a[a.turn == "TURN_DN"]
    print(f"  gate: TURN_DN N {len(dn)}/{GATE_N} on {dn.day.nunique()}/{GATE_DAYS} days → "
          + ("READY for the gate read (bootstrap + breakeven + concentration)" if len(dn) >= GATE_N and dn.day.nunique() >= GATE_DAYS else "collecting"))


if __name__ == "__main__":
    main()
