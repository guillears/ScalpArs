"""🔓 Sep-29 ATR×gap LONG gate re-opened (DECISION_LOG 133) — pins the live state and the zone definition the watch item tallies on."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)


def test_gate_is_open_and_zone_definition_is_kept():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["atr_gap_block_long_enabled"] is False                      # re-opened; flips back only by the tripwire / verdict
    assert th["atr_gap_block_atr_min_long"] == 1.0 and th["atr_gap_block_gap_min_long"] == 0.5   # the zone the tally uses
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "getattr(config.trading_config.thresholds, 'atr_gap_block_long_enabled', False)" in eng   # engine still honours the toggle
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-atr-gap-block-long-enabled") >= 3                                        # toggle + load + save
    state = open(os.path.join(ROOT, "CLAUDE_CURRENT_STATE.md")).read()
    assert "ATR×GAP LONG RE-OPENED" in state and "TRIPWIRE" in state


def test_zone_tally_helper_matches_the_gate():
    """The batch-review tally (stamps) must select exactly what the gate used to refuse: same formulas, same >= comparisons."""
    import pandas as pd
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "_ag_atr_pct = (_ag_atr / _ag_price) * 100" in eng and "_ag_gap_pct = (_ag_e13 - _ag_e50) / _ag_e50 * 100" in eng
    assert "if _ag_atr_pct >= _ag_atr_min and _ag_gap_pct >= _ag_gap_min:" in eng
    d = pd.DataFrame({"entry_strategy": ["MOMENTUM"] * 5 + ["SPIKE_FADE"], "direction": ["LONG", "LONG", "LONG", "SHORT", "LONG", "LONG"],
                      "entry_atr_pct": [1.0, 0.99, 1.5, 1.5, 1.2, 1.5], "entry_pair_ema20_ema50_gap_pct": [0.5, 0.9, 0.49, 0.9, 0.9, 0.9]})
    zone = (d.entry_strategy == "MOMENTUM") & (d.direction == "LONG") & (d.entry_atr_pct >= 1.0) & (d.entry_pair_ema20_ema50_gap_pct >= 0.5)
    assert zone.tolist() == [True, False, False, False, True, False]
