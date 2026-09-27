"""🌫 Sep-27 CALM3D BTC-ATR FLOOR — pure-rule invariants + parity (engine helper, builder freeze, live JSON).

The door buys a coiled pair on a calm BTC; this leg refuses a DEAD tape (BTC 5m ATR% < nonexp_calm3d_btc_atr_min).
A block must never fire on data it lacks (fail-open), 0 switches the leg off, and the boundary is strict.
"""
import json
import os
import re
from types import SimpleNamespace

from services.trading_engine import calm3d_btc_atr_floor_block

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TH = SimpleNamespace(nonexp_calm3d_btc_atr_min=0.08)


def test_blocks_dead_tape():
    assert calm3d_btc_atr_floor_block(TH, 0.063) is True
    assert calm3d_btc_atr_floor_block(TH, 0.0799) is True


def test_admits_at_and_above_floor():
    assert calm3d_btc_atr_floor_block(TH, 0.08) is False          # strict <
    assert calm3d_btc_atr_floor_block(TH, 0.12) is False


def test_fail_open_on_missing_or_bad_reading():
    for v in (None, float("nan"), float("inf"), "x", ""):
        assert calm3d_btc_atr_floor_block(TH, v) is False


def test_leg_off_when_zero_missing_or_negative():
    for floor in (0, 0.0, None, -1.0, float("nan")):
        assert calm3d_btc_atr_floor_block(SimpleNamespace(nonexp_calm3d_btc_atr_min=floor), 0.01) is False
    assert calm3d_btc_atr_floor_block(SimpleNamespace(), 0.01) is False


def test_string_numbers_accepted():
    assert calm3d_btc_atr_floor_block(SimpleNamespace(nonexp_calm3d_btc_atr_min="0.08"), "0.05") is True


def test_parity_json_builder_default():
    live = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["nonexp_calm3d_btc_atr_min"]
    src = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    frozen = float(re.search(r"nonexp_calm3d_btc_atr_min=([0-9.]+)", src).group(1))
    assert live == frozen == 0.08
    import config
    assert config.SignalThresholds.model_fields["nonexp_calm3d_btc_atr_min"].default == 0.0   # shipped via JSON; default off


def test_engine_door_calls_the_rule_and_counts_it():
    src = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    door = src[src.find("_nonexp_calm3d_hit = False"):src.find("nonexp_calm3d=bool(_nonexp_calm3d_hit)")]
    assert "calm3d_btc_atr_floor_block(_th_pp, _pp_batr)" in door
    assert '_record_filter_block("CALM3D_BTC_ATR_MIN", "LONG"' in door
    assert "and not _pp_floor_block):" in door                     # the admit path requires the floor to pass
