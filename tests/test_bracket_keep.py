"""🪜📈 DECISION_LOG 535 — keep the planned position on capped coins (pure rule, config wiring, scout gate)."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services.trading_engine import bracket_keep_position, bracket_keep_sleeves, leverage_bracket_limit   # noqa: E402

ONE = [(1900, 50), (4750, 20), (11400, 10), (47500, 5), (190000, 4)]      # today's ONEUSDT table (× 0.95 headroom)


def test_one_keeps_the_planned_position_at_5x_with_more_margin():
    lev, n, m = bracket_keep_position(ONE, 8, 19_600, 2_450 * 1.6)
    assert (lev, round(n), round(m)) == (5.0, 19_600, 3_920)
    le, n0, *_ = leverage_bracket_limit(ONE, 8, 2_450)          # today's resize: same margin → $12,250 at 5×
    assert le == 5 and round(n0) == 12_250 and n > n0


def test_margin_ceiling_binds_and_ties_keep_the_higher_leverage():
    lev, n, m = bracket_keep_position(ONE, 8, 19_600, 2_450 * 1.2)   # $2,940 max → $14,700 at 5×
    assert (lev, round(n), round(m)) == (5.0, 14_700, 2_940)
    lev, n, m = bracket_keep_position(ONE, 8, 4_000, 5_000)          # fits at 8× on the 10× tier → never a lower leverage than needed
    assert lev == 8.0 and round(n) == 4_000 and round(m) == 500


def test_fail_closed_inputs():
    assert bracket_keep_position(None, 8, 19_600, 3_920) is None
    assert bracket_keep_position(ONE, 8, 0, 3_920) is None
    assert bracket_keep_position(ONE, 8, 19_600, float("nan")) is None
    assert bracket_keep_position([("x", 5)], 8, 19_600, 3_920) is None


def test_sleeves_parse_and_ml_is_not_listed():
    class I: bracket_keep_position_sleeves = " frenzy_lite , FRENZY_WILLY_HT,,FLATBASE_SHORT "
    assert bracket_keep_sleeves(I) == {"FRENZY_LITE", "FRENZY_WILLY_HT", "FLATBASE_SHORT"}
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["investment"]
    assert cfg["bracket_keep_position_enabled"] is True and cfg["bracket_keep_position_max_margin_mult"] == 1.6
    assert "MOMENTUM" not in bracket_keep_sleeves(type("C", (), cfg))


def test_d11_wiring():
    import config
    c = config.InvestmentConfig()
    assert c.bracket_keep_position_enabled is False and c.bracket_keep_position_max_margin_mult == 1.6   # code default OFF, the JSON arms it
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    for i in ("config-bracket-keep-enabled", "config-bracket-keep-max-mult", "config-bracket-keep-sleeves"):
        assert html.count(i) >= 3, i                                        # UI + load + save
    for k in ("bracket_keep_position_enabled:", "bracket_keep_position_max_margin_mult:", "bracket_keep_position_sleeves:"):
        assert k in html, k
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "bracket_keep_margin_from=_bk_from_m, bracket_keep_notional_from=_bk_from_n" in eng
    from models import Order
    assert {"bracket_keep_margin_from", "bracket_keep_notional_from"} <= {c.name for c in Order.__table__.columns}
    db = open(os.path.join(ROOT, "database.py")).read()
    assert "'bracket_keep_margin_from', 'bracket_keep_notional_from'" in db


def _export(tmp_path, rows):
    p = tmp_path / "scalpars_orders_paper_2027-01-01_00-00-00.csv"
    base = dict(direction="LONG", status="CLOSED", close_reason="FRENZY_TP", cell_multiplier_source="FRENZY_LITE_RETRY", leverage=5.0)
    pd.DataFrame([dict(base, **r) for r in rows]).to_csv(p, index=False)
    return str(p)


def test_scout_gate_counts_kept_fills_and_freezes(tmp_path):
    import scout_bracket_keep as S
    rows = [dict(opened_at=f"2027-01-0{1 + i // 9}T{i % 9 + 10}:00:00", pair=f"P{i}USDT", entry_strategy="FRENZY_LITE_RETRY",
                 pnl_percentage=(3.0 if i % 3 else -3.0), notional_value=23_333, investment=4_667,
                 bracket_keep_margin_from=2_333, bracket_keep_notional_from=11_667) for i in range(16)]
    rows.append(dict(opened_at="2027-01-01T09:00:00", pair="ZZZUSDT", entry_strategy="FRENZY_LITE", pnl_percentage=-3.0,
                     notional_value=10_000, investment=1_250))                # not kept → never counted
    st = tmp_path / "s.json"
    out = "\n".join(S.run(paths=[_export(tmp_path, rows)], floor_ms=0, state_path=str(st)))
    assert "Kept fills since" in out and "**KEEP**" in out and st.exists()
    assert S.gate([-10.0] * 15)[0] == "REVERT" and S.gate([5.0] * 14)[0] == "OBSERVING"
    again = "\n".join(S.run(paths=[_export(tmp_path, [dict(rows[0], pnl_percentage=-30.0)])], floor_ms=0, state_path=str(st)))
    assert "frozen" in again and "**KEEP**" in again


def test_hooked_registered_and_lev_step_cohort_unchanged():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_bracket_keep as _bkp")
    assert "_bkp.run(now_ms)" in src[j:j + 120] and "except Exception as _bkp_e" in src[j:j + 200]
    import observe_registry as OR
    assert "scout_bracket_keep.py" in OR.classified()
    import scout_lev_step as LS
    assert "bracket_keep_notional_from" not in LS.COLS          # review 535: the 11-11 pre-committed lev-step cohort does not move
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    i = eng.index("KEEP THE PLANNED POSITION on a capped coin")
    assert i < eng.index("judged at today's size", i)           # the keep step runs BEFORE the min-margin skip
    assert '_wh_es != "MOMENTUM"' in eng[i:i + 2500] and "_bk_trd = _bk_tradeable" in eng[i:i + 2500]
