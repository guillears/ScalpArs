"""👁 LITE_WEAKBTC_BUYBLIP (Z3) — observe-only watch line for FRENZY_LITE (DECISION_LOG 300, operator 2026-10-15). Rebuilt at batch review;
no engine change.

Zone (frozen, never re-fitted): entry_btc_rsi_1h_prev ≤ 55 (stamped) ∧ d_top10_tbr > −0.003909, where d_top10_tbr = the top-10 aggregate
taker-buy share at the last 5-min boundary before entry minus the same 5 min earlier (from the market-impulse grid,
scripts/study_ml_impulse_features.py; reports/LITE_EPIC_ENTRY_2026-10-15.md, Z3). Evidence (in-sample, the frozen 2025 / 2026 backtest cohort):
refusing the zone adds +0.140 (2025) / +0.098 (2026) %/fill; refused WR 40 % / 44 % vs 50 % breakeven; P(avg<0) 1.00 / 0.96; but master −0.45
(4 of 5 refused were +3 winners) and the value sits inside LITE_BTC1H_FALLING (DECISION_LOG 288) — read the two together.
Cohort: FRENZY_LITE CLOSED fills opened ≥ 2026-10-16 00:00 UTC only, DAY units. Gate to OPEN a filter study (not a ship): zone N ≥ 15 on ≥ 8
UTC days ∧ the EXPECTANCY bar on the zone (WR < LITE breakeven, avg < 0 at day-bootstrap P ≥ 0.95, no day/pair ≥ 50 % of the loss) ∧ zone avg
< rest avg. Retire: zone avg ≥ rest avg at N 15.

Usage (batch review): extend the impulse grids to the batch end first (study_ml_impulse_fetch.py + study_ml_impulse_features.py), then
venv/bin/python scripts/scout_lite_z3.py
"""
import os
import sys

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "scripts"))
COHORT_START = "2026-10-16T00:00:00"
RSI_CUT, TBR_CUT = 55.0, -0.003909


def fresh_fills():
    import export_glob as EG
    fr = []
    for f in EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "pair", "direction", "status", "entry_strategy",
                                                                                 "pnl_percentage", "entry_btc_rsi_1h_prev", "is_paper")))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    return a[(a.entry_strategy.astype(str) == "FRENZY_LITE") & (a.status == "CLOSED") & (a.opened_at.astype(str) >= COHORT_START)]


def main():
    import study_ml_impulse_features as FX
    import study_ml_delay_common as DC
    a = fresh_fills()
    print(f"LITE_WEAKBTC_BUYBLIP (Z3) · cohort: FRENZY_LITE closed fills opened ≥ {COHORT_START} UTC → {len(a)}")
    if not len(a):
        return
    a["o_ms"] = DC._ms(a.opened_at).astype(np.int64)
    t5 = (a.o_ms // FX.MIN5 * FX.MIN5).values
    g5 = pd.read_csv(FX.OUT5, index_col=0)
    if "top10_tbr" not in g5.columns:
        print("  ⚠ the impulse 5m grid has no top10_tbr column — rebuild it (see the header)")
        return
    now, prev = g5.top10_tbr.reindex(t5).values, g5.top10_tbr.reindex(t5 - FX.MIN5).values
    d = now - prev
    rsi = pd.to_numeric(a.entry_btc_rsi_1h_prev, errors="coerce").values
    unscored = np.isnan(d) | np.isnan(rsi)
    a["zone"] = np.where(unscored, "UNSCORED", np.where((rsi <= RSI_CUT) & (d > TBR_CUT), "ZONE", "REST"))
    a["day"] = a.opened_at.astype(str).str[:10]
    p = pd.to_numeric(a.pnl_percentage, errors="coerce")
    for z, g in a.groupby("zone"):
        pp = p[g.index]
        print(f"  {z:9s} N {len(g):3d} · days {g.day.nunique():3d} · WR {(pp > 0).mean():.0%} · avg {pp.mean():+.3f} % · Σ {pp.sum():+.2f}")
    zn = a[a.zone == "ZONE"]
    print(f"  gate: zone N {len(zn)}/15 on {zn.day.nunique()}/8 days → "
          + ("READY for the gate read" if len(zn) >= 15 and zn.day.nunique() >= 8 else "collecting")
          + " · read with LITE_BTC1H_FALLING (scripts/scout_frenzy_wl_lines.py)")


if __name__ == "__main__":
    main()
