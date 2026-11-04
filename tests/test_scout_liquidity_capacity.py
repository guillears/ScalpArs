"""💧📏 DECISION_LOG 454: scout_liquidity_capacity — per-sleeve cap tally, the $ the cut left out, the account multiple where the cap binds,
full-size bot fills only (probes / MANUAL / open out), live-only slippage."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_liquidity_capacity as LQ  # noqa: E402


def _frame():
    return pd.DataFrame(dict(
        status=["CLOSED"] * 8 + ["OPEN"],
        entry_strategy=["SPIKE_FADE", "SPIKE_FADE", "MOMENTUM", "MOMENTUM", "MANUAL", "MOMENTUM", "FRENZY_WILLY", "FRENZY_WILLY", "SPIKE_FADE"],
        direction=["SHORT", "SHORT", "LONG", "LONG", "LONG", "LONG", "LONG", "LONG", "SHORT"],
        liquidity_capped=[True, "True", False, False, False, False, False, True, True],
        bracket_capped=[False, False, False, False, False, False, True, False, False],
        pnl_percentage=[1.0, -2.0, 0.5, -0.5, 9.0, 3.0, 2.0, 1.0, 1.0],
        notional_value=[1000.0, 1000.0, 500.0, 500.0, 100.0, 500.0, 1000.0, 900.0, 1000.0],
        entry_desired_notional=[4000.0, 2000.0, 500.0, 500.0, 100.0, 500.0, 2000.0, 1000.0, 4000.0],
        entry_liquidity_cap_notional=[1000.0, 1000.0, 2500.0, 10000.0, 1e6, 1e6, 5000.0, 5000.0, 1000.0],
        entry_bracket_cap_notional=[np.nan] * 6 + [1000.0, np.nan, np.nan],
        entry_slippage_pct=[0.2, np.nan, 0.01, 0.03, 0.0, 0.0, 0.0, 0.0, 0.5],
        cell_multiplier_source=["", "", "UNMATCHED", "UNMATCHED", "", "GAP_PROBE", "", "", ""]))


def test_cohort_full_size_closed_bot_only():
    d = LQ.prep(_frame())
    assert sorted(d.sleeve) == ["FRENZY_WILLY", "FRENZY_WILLY", "MOMENTUM_LONG", "MOMENTUM_LONG", "SPIKE_FADE", "SPIKE_FADE"]   # MANUAL, probe, OPEN out


def test_volume_cap_cut_missed_binds():
    r = {x["sleeve"]: x for x in LQ.sleeve_rows(LQ.prep(_frame()))}
    f, m = r["SPIKE_FADE"], r["MOMENTUM_LONG"]
    assert (f["n"], f["vcap"], f["bcap"], f["gross"]) == (2, 2, 0, 0)
    assert abs(f["v_cutmed"] - np.median([0.75, 0.5])) < 1e-9
    assert abs(f["v_missed"] - (1.0 * 3000 / 100 + -2.0 * 1000 / 100)) < 1e-9 and f["b_missed"] == 0   # a capped loser's cut SAVED money
    assert abs(f["binds"] - np.median([0.25, 0.5])) < 1e-9 and f["at"][2] == 1.0
    assert m["capped"] == 0 and m["v_missed"] == 0 and abs(m["binds"] - 12.5) < 1e-9   # limit / desired: 5 and 20 → median 12.5
    assert m["at"][5] == 0.0 and m["at"][10] == 0.5


def test_bracket_split_and_gross_only():
    w = {x["sleeve"]: x for x in LQ.sleeve_rows(LQ.prep(_frame()))}["FRENZY_WILLY"]
    assert (w["vcap"], w["bcap"], w["gross"], w["capped"]) == (0, 1, 1, 2)   # flagged liquidity_capped with cap ≥ desired = gross-only
    assert w["v_missed"] == 0 and abs(w["b_missed"] - (2.0 * 1000 / 100 + 1.0 * 100 / 100)) < 1e-9
    assert abs(w["binds_v"] - 3.75) < 1e-9                                 # volume cap alone: 5000 / 2000 = 2.5 and 5000 / 1000 = 5 → median 3.75
    assert abs(w["binds"] - np.median([0.5, 5.0])) < 1e-9                  # any cap: bracket 1000 / 2000 = 0.5 · volume 5000 / 1000 = 5


def test_slip_rows_split_capped():
    rows = {s: (nc, mc, nu, mu) for s, nc, mc, nu, mu in LQ.slip_rows(LQ.prep(_frame()))}
    assert rows["SPIKE_FADE"][:2] == (1, 0.2)                                       # NaN slip out
    assert rows["MOMENTUM_LONG"][2] == 2 and abs(rows["MOMENTUM_LONG"][3] - 0.02) < 1e-9


def test_run_renders_without_exports(tmp_path):
    L = LQ.run(paths=[], master=None)
    assert any("LIQUIDITY CAPS" in x for x in L) and any("No orders export" in x for x in L)
