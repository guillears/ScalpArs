"""📈 Nov-11 B26 LEVERAGE STEP + MOMENTUM LONG RISK CAP (DECISION_LOG 533).

Invariants: the worst stop = the widest pre-BE stop the momentum stack applies (conf / signal-active / 1.5 × ATR, clamped −1.2, quiet SL on top) ·
an open trade's risk is read at its CURRENT exit line (second chance floor · RH hard stop · 0 once armed · else worst stop) · the fit shrinks to
the room and returns 0 with none · the live leverage values and the cap are wired (D11) · every new level keeps its exchange backstop inside
liquidation · the revert-gate scout decides once on the first 15 fills at the new leverage.
"""
import json
import os
import sys

import pandas as pd
import pytest

from services.ml_risk_cap import worst_stop_pct, open_trade_risk_pct, open_risk_pct_of_account, fit_notional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_worst_stop():
    assert worst_stop_pct(-0.70, -1.00, 0.45, 1.5, -1.2) == pytest.approx(1.0)      # signal-active stop wins
    assert worst_stop_pct(-0.70, -1.00, 0.75, 1.5, -1.2) == pytest.approx(1.125)    # ATR widening
    assert worst_stop_pct(-0.70, -1.00, 2.0, 1.5, -1.2) == pytest.approx(1.2)       # clamped at the floor
    assert worst_stop_pct(-0.70, None, None, 1.5, -1.2) == pytest.approx(0.7)
    assert worst_stop_pct(-0.70, -1.00, 0.3, 1.5, -1.2, quiet_sl=-2.0) == pytest.approx(2.0)
    assert worst_stop_pct(None, None, None, None, None) == pytest.approx(0.7)


def test_open_trade_risk():
    assert open_trade_risk_pct(sc_line=0.25) == 0.0
    assert open_trade_risk_pct(sc_line=-0.2) == pytest.approx(0.2)
    assert open_trade_risk_pct(rh_hard_stop=-1.2) == pytest.approx(1.2)
    assert open_trade_risk_pct(peak=0.42, arm=0.40, worst_stop=1.0) == 0.0
    assert open_trade_risk_pct(peak=0.30, arm=0.40, worst_stop=1.0) == pytest.approx(1.0)


def test_account_pct_and_fit():
    acct = 10_000
    # ML at 15×: one trade ≈ $36,450 notional (3.645 × account); two unarmed trades at a 1.0 stop ≈ 7.3 %
    rows = [(36_450, 1.0), (36_450, 1.0)]
    o = open_risk_pct_of_account(rows, acct)
    assert o == pytest.approx(7.29)
    room = fit_notional(36_450, 1.0, o, acct, 10.0)
    assert room == pytest.approx((10 - 7.29) / 100 * acct / 0.01)          # ≈ $27,100 → shrunk
    assert fit_notional(36_450, 1.0, 0.0, acct, 10.0) == 36_450                # one trade fits (3.6 %)
    assert fit_notional(36_450, 1.0, 10.5, acct, 10.0) == 0.0                  # no room
    assert fit_notional(36_450, 1.0, 5.0, acct, 0) is None                     # off
    assert open_risk_pct_of_account(rows, 0) == 0.0


def test_live_values_and_wiring():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert (th["momentum_long_lev_mult"], th["frenzy_lite_retry_lev_mult"], th["frenzy_willy_lev_mult"], th["flatbase_lev_mult"],
            th["frenzy_willy_ht_size_frac"], th["frenzy_lite_lev_mult"], th["ml_risk_cap_pct"]) == (0.75, 0.5, 0.4, 0.4, 1.0, 0.4, 10.0)
    from config import SignalThresholds
    assert SignalThresholds().ml_risk_cap_pct == 0.0
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert html.count('id="config-ml-risk-cap"') == 1 and html.count("'config-ml-risk-cap'") == 2
    from models import Order
    assert "entry_ml_risk_open_pct" in {c.name for c in Order.__table__.columns}
    from services.trading_engine import TradingEngine
    assert TradingEngine.stamp_class("entry_ml_risk_open_pct") == "sleeve"


def test_backstops_inside_liquidation_at_new_levels():
    import config
    from services.trading_engine import order_backstop_pct, MANUAL_BACKSTOP_MIN_GAP_PCT
    th = config.trading_config.thresholds
    for es, lev, stop in (("FLATBASE_SHORT", 8, 10.0), ("FRENZY_WILLY", 8, 4.0), ("FRENZY_WILLY_HT", 8, 4.0), ("FRENZY_LITE_RETRY", 10, 3.0),
                          ("MOMENTUM", 15, 1.2)):
        bk = order_backstop_pct(th, es, leverage=lev)
        assert bk - stop >= MANUAL_BACKSTOP_MIN_GAP_PCT - 1e-9, (es, lev, bk)
        assert bk < 100.0 / lev, (es, lev, bk)                                   # inside the liquidation distance


def test_lev_step_gate_and_scout(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_lev_step as M
    assert M.gate([1.0] * 14, 0.61)[0] == "OBSERVING"
    assert M.gate([1.0] * 10 + [-1.0] * 5, 0.61)[0] == "KEEP"                     # 67 % ≥ 61.1 %, avg +0.33
    assert M.gate([1.0] * 9 + [-1.0] * 6, 0.611)[0] == "REVERT"                   # 60 % < 61.1 %
    assert M.gate([0.3] * 12 + [-3.0] * 3, 0.5)[0] == "REVERT"                    # WR 80 % but avg < 0
    rows = []
    for i in range(16):
        rows.append(dict(opened_at=f"2026-11-13T{i:02d}:00:00", pair=f"P{i}USDT", direction="LONG", entry_strategy="FRENZY_LITE_RETRY",
                         status="CLOSED", pnl_percentage=(3.0 if i % 4 else -3.0), close_reason="FRENZY_TP", cell_multiplier_source="",
                         leverage=10, entry_ml_risk_open_pct=None, notional_value=1000, entry_desired_notional=1000))
    rows.append(dict(opened_at="2026-11-13T20:00:00", pair="XUSDT", direction="LONG", entry_strategy="FRENZY_LITE_RETRY", status="CLOSED",
                     pnl_percentage=-3.0, close_reason="FRENZY_SL", cell_multiplier_source="", leverage=7, entry_ml_risk_open_pct=None,
                     notional_value=700, entry_desired_notional=700))   # bracket-lowered → not counted
    f = tmp_path / "scalpars_orders_paper_2026-11-14_00-00-00.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    stp = tmp_path / "state.json"
    out = "\n".join(M.run(paths=[str(f)], floor_ms=int(pd.Timestamp("2026-11-12").value // 10**6), state_path=str(stp)))
    assert "| LITE RETRY | 10× | 16 | **KEEP** | first 15: WR 73% (breakeven 50.0%) · avg +1.40 % |" in out
    # frozen: a later export where the same sleeve loses never flips the decided verdict
    rows2 = [dict(r, pnl_percentage=-3.0) for r in rows]
    f2 = tmp_path / "scalpars_orders_paper_2026-11-15_00-00-00.csv"
    pd.DataFrame(rows2).to_csv(f2, index=False)
    out2 = "\n".join(M.run(paths=[str(f2)], floor_ms=int(pd.Timestamp("2026-11-12").value // 10**6), state_path=str(stp)))
    assert "| LITE RETRY | 10× | 16 | **KEEP** |" in out2 and "(frozen " in out2


def test_registry_and_hook():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import observe_registry as OR
    assert "scout_lev_step.py" in OR.classified()
    row = [r for r in OR.status() if r[0] == "scout_lev_step.py"][0]
    assert row[4] == "OK", row[5]
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_lev_step as _lvs")
    assert "_lvs.run(now_ms)" in src[j:j + 200] and "except Exception as _lvs_e" in src[j:j + 400]



def test_revert_instructions_are_legal_values():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import re
    import scout_lev_step as M
    for name, (_, _, _, _, how) in M.SLEEVES.items():
        for v in re.findall(r"frenzy_willy_ht_size_frac [0-9.]+ → ([0-9.]+)", how):
            assert 0 < float(v) <= 1.0, (name, how)          # services/frenzy.frenzy_willy_ht_frac accepts (0, 1] only
    assert "1.6" not in M.SLEEVES["WILLY"][4]


def test_pricer_knows_the_cap_line():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_blocked_filters as B
    assert B.LINES[B.ML_RISKCAP] == ("ML_RISK_CAP",) and "ML_RISK_CAP" not in B.LINES["MOMENTUM_LONG"]   # own label: never shadows the ML gates
    assert B.KEPT_AS[B.ML_RISKCAP] == "MOMENTUM_LONG" and B.exit_of(B.ML_RISKCAP, {})["cap"] == B.exit_of("MOMENTUM_LONG", {})["cap"]


def test_engine_cap_block_scope():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    j = src.index("_mlrc_probe = bool(")
    blk = src[j:j + 6000]
    assert "and not _mlrc_probe" in blk                                   # probes are never capped
    assert '_record_filter_block("ML_RISK_CAP", direction)' in blk and "ML_RISK_CAP_SHRINK" not in blk   # a shrink is stamped, never counted as a block
    assert "entry_ml_risk_shrunk_from=_mlrc_from" in src
