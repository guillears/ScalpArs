"""👁 DECISION_LOG 335: SURGE_BRACKET (observe-only) — live SURGE_LONG fills split capped (bracket_capped or traded leverage < the 20 ×
surge_long_lev_mult target) vs uncapped, the "doubled to target notional" counterfactual (held at entry_bracket_cap_notional when > 0), the
thin-book cost sensitivity (+0.2 % entry / +0.3 % on stop exits) and the sleeve-wide Pattern W size-up gate."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_revert_gates as RG  # noqa: E402


def _o(rows):
    d = pd.DataFrame(rows)
    d["o_ms"] = [RG._ms(t) for t in d.opened_at]
    return d


# the real B22 / B23 rows (reports/BASELINE22_*_LIVE_orders.csv, BASELINE23_*_LIVE_orders.csv)
STRK = dict(opened_at="2026-10-15T15:05:27", pair="STRKUSDT", entry_strategy="SURGE_LONG", status="CLOSED", leverage=20.0, bracket_capped=False,
            entry_bracket_cap_notional=100000.0, notional_value=4887.818956, pnl=74.644276, pnl_percentage=1.527149, close_reason="BR_TRAILING_STOP")
GTC = dict(opened_at="2026-10-19T02:50:45", pair="GTCUSDT", entry_strategy="SURGE_LONG", status="CLOSED", leverage=10.0, bracket_capped=True,
           entry_bracket_cap_notional=0.0, notional_value=2304.810584, pnl=95.722098, pnl_percentage=4.153144, close_reason="BR_TRAILING_STOP")
AGT = dict(opened_at="2026-10-18T01:45:32", pair="AGTUSDT", entry_strategy="SURGE_LONG", status="CLOSED", leverage=10.0, bracket_capped=True,
           entry_bracket_cap_notional=0.0, notional_value=2123.899439, pnl=159.976629, pnl_percentage=7.532213, close_reason="BR_TRAILING_STOP")


def test_b22_b23_split_and_doubled_counterfactual():
    r = RG.surge_bracket_rows(_o([STRK, GTC, AGT]), 20.0).set_index("pair")
    assert not r.capped["STRKUSDT"] and r.capped["GTCUSDT"] and r.capped["AGTUSDT"]
    assert r.factor["STRKUSDT"] == 1.0 and r.cf_usd["STRKUSDT"] == STRK["pnl"] and r.slip_usd["STRKUSDT"] == 0.0
    assert r.factor["GTCUSDT"] == 2.0 and abs(r.cf_usd["GTCUSDT"] - 2 * GTC["pnl"]) < 1e-9        # 10× vs target 20×; cap notional 0 = unknown
    assert abs(r.cf_usd["AGTUSDT"] - 319.953258) < 1e-6
    assert abs(r.slip_usd["AGTUSDT"] + 2 * AGT["notional_value"] * 0.5 / 100) < 1e-9            # trailing stop → +0.2 entry +0.3 stop
    C = RG.surge_bracket_stats(r[r.capped.astype(bool)].reset_index())
    U = RG.surge_bracket_stats(r[~r.capped.astype(bool)].reset_index())
    assert C["n"] == 2 and C["wr"] == 100.0 and abs(C["avg"] - (4.153144 + 7.532213) / 2) < 1e-9 and abs(C["usd"] - 255.698727) < 1e-6
    assert abs(C["cf"] - 511.397454) < 1e-6 and U["n"] == 1 and abs(U["usd"] - 74.644276) < 1e-9


def test_capped_by_leverage_alone_and_bracket_notional_cap():
    x = dict(GTC, bracket_capped=False, entry_bracket_cap_notional=3000.0)          # traded 10× < 20× → capped even without the flag
    r = RG.surge_bracket_rows(_o([x]), 20.0)
    assert r.capped.iloc[0] and abs(r.factor.iloc[0] - 3000.0 / GTC["notional_value"]) < 1e-12    # held at the bracket's notional
    y = dict(STRK, bracket_capped="True")                                          # flagged at the target leverage (smaller position)
    ry = RG.surge_bracket_rows(_o([y]), 20.0)
    assert ry.capped.iloc[0] and ry.factor.iloc[0] == 1.0
    assert RG.surge_bracket_target(1.0) == 20.0 and RG.surge_bracket_target(0.5) == 10.0 and RG.surge_bracket_target(0) == 20.0
    t = RG.surge_bracket_rows(_o([x]), RG.surge_bracket_target(0.5))                # target 10× → a 10× fill is uncapped
    assert not t.capped.iloc[0]


def test_cohort_floor_is_the_later_of_go_live_and_the_surge_floor():
    early = dict(STRK, opened_at="2026-10-14T18:00:00")                            # paper era, before the live switch
    other = dict(STRK, opened_at="2026-10-15T00:00:00", entry_strategy="FRENZY_WILLY")
    r = RG.surge_bracket_rows(_o([early, other, GTC]), 20.0)
    assert list(r.pair) == ["GTCUSDT"]
    assert RG.SB_GOLIVE == "2026-10-14T18:36:30" and RG._ms(RG.SB_GOLIVE) > RG._ms(RG.SB_COHORT_FLOOR)


def test_w_gate_and_lines():
    o = _o([STRK, GTC, AGT, dict(AGT, opened_at="2026-10-20T01:00:00", status="OPEN", pnl=np.nan, pnl_percentage=np.nan)])
    met, txt = RG.surge_w_gate(RG.surge_bracket_stats(RG.surge_bracket_rows(o, 20.0)))
    assert not met and "✗ N 3/30" in txt and "✓ WR 100 %" in txt and "✓ avg +4.40 %" in txt and "✓ Σ$ +330 > 0" in txt
    big = _o([dict(STRK, opened_at=f"2026-10-{15 + i // 10:02d}T{i % 10:02d}:00:00") for i in range(30)])
    assert RG.surge_w_gate(RG.surge_bracket_stats(RG.surge_bracket_rows(big, 20.0)))[0]
    losing = _o([dict(STRK, opened_at=f"2026-10-{15 + i // 10:02d}T{i % 10:02d}:00:00", pnl=-1.0 if i < 10 else 1.0,
                      pnl_percentage=-0.5 if i < 10 else 0.5) for i in range(30)])
    assert not RG.surge_w_gate(RG.surge_bracket_stats(RG.surge_bracket_rows(losing, 20.0)))[0]      # WR 67 % < 70 %
    opx = _o([dict(STRK, opened_at="2026-10-16T00:00:00", close_reason="EXTERNAL_CLOSE")])
    L = "\n".join(RG.surge_bracket_lines(o, opx, target_lev=20.0))
    assert "| capped | 2 | 100 % | +5.843 | +255.70 | +511.40 | +467.11 |" in L
    assert "| uncapped | 1 | 100 % | +1.527 | +74.64 | +74.64 | +74.64 |" in L
    assert "1 open fill(s) not counted" in L and "operator-closed: 1 — excluded" in L and "⏳ not met" in L
