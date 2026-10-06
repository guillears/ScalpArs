"""DECISION_LOG 231 (2026-10-06): FRENZY_WIDE takes only hold-green setups (green signal candle, ATR within the cap, price already > N
closes above the spike VWAP). Pure rule + wiring (config default off, live JSON 12, engine call + stamp, model + migration, UI field)."""
import json, os
from types import SimpleNamespace
from services.frenzy import frenzy_wide_hold_green_block as hg
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ON, OFF = SimpleNamespace(frenzy_wide_hold_green_streak=12.0), SimpleNamespace(frenzy_wide_hold_green_streak=0.0)


def test_rule():
    g = dict(bar_ret_pct=0.52)
    assert hg(dict(g, above_streak=17), "FRENZY_GREEN_BAR", ON) is None                  # RLC 10-05: hold-green → WIDE may open
    assert hg(dict(g, above_streak=13), "FRENZY_GREEN_BAR", ON) is None                  # the first passing streak
    assert hg(dict(g, above_streak=12), "FRENZY_GREEN_BAR", ON) == "FRENZY_WIDE_RECLAIM"  # the 12th close on the signal bar = reclaim
    assert hg(dict(g, above_streak=40), "FRENZY_ATR_HIGH", ON) == "FRENZY_WIDE_ATR_HIGH"  # high ATR never passes
    assert hg(g, "FRENZY_GREEN_BAR", ON) == "FRENZY_WIDE_RECLAIM"                         # fail-closed on a missing streak
    assert hg(dict(g, above_streak='x'), "FRENZY_GREEN_BAR", ON) == "FRENZY_WIDE_RECLAIM"
    assert hg({'above_streak': 40, 'bar_ret_pct': None}, "FRENZY_GREEN_BAR", ON) == "FRENZY_WIDE_RECLAIM"   # unreadable candle
    assert hg(dict(g, above_streak=13), "FRENZY_GREEN_BAR", SimpleNamespace(frenzy_wide_hold_green_streak=12.5)) is None
    for code in ("FRENZY_GREEN_BAR", "FRENZY_ATR_HIGH"):
        assert hg({'above_streak': 1}, code, OFF) is None                                 # 0 = off: WIDE as before
    assert hg({'above_streak': 1}, "FRENZY_GREEN_BAR", SimpleNamespace()) is None        # missing field = off


def test_wiring():
    import config as C
    assert C.SignalThresholds.model_fields["frenzy_wide_hold_green_streak"].default == 0.0
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    assert (th.get("thresholds", th))["frenzy_wide_hold_green_streak"] == 12.0
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "_hg = frenzy_wide_hold_green_block(flag, flag.get('code'), th) if wide else None" in eng
    assert "entry_frenzy_above_streak=(entry_frenzy_above_streak if _frenzy else None)" in eng
    assert "entry_frenzy_above_streak = Column(Integer" in open(os.path.join(ROOT, "models.py")).read()
    assert "('entry_frenzy_above_streak', 'INTEGER')" in open(os.path.join(ROOT, "database.py")).read()
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-fz-wide-hold-streak") == 2 and "['config-fz-wide-hold-streak', 'frenzy_wide_hold_green_streak', 12]" in ui
    assert "_key === 'frenzy_wide_hold_green_streak') ? Math.round(x) : x" in ui
