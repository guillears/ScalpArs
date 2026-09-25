"""🪃 Sep-25 FAN-FLIP WEAK-BOUNCE BLOCK — the pure rule `_flip_filters`, the master-pool builder and the ledger call.

(The ledger and builder call the same helper; the builder freeze is pinned below.)

Rule: refuse a FAN_RATIO_GATE flip-SHORT when the pair is BELOW its own trend (EMA13−EMA50 gap% < gap_max, ship 0.0)
AND its EMA20 rises only gently (3-bar slope% < slope_max, ship 0.15). Evidence (34 kept fan flips, today's stack):
blocked 8·25%·−$906, kept 34·76%·+$683 → 26·92%·+$1,588; below-trend flips 6/6 W when steep vs 2W/6L when gentle.
Operator-directed ARMED override at N=8 (DECISION_LOG 113).

Invariants pinned here:
  · strict inequalities on both legs (gap == 0 or slope == 0.15 passes).
  · disabled / missing / non-finite threshold or input FAILS OPEN.
  · FAN flip-SHORT only: flip-LONGs and non-FAN sources are never touched.
  · all 8 historic blocked flips block; the steep below-trend winners pass.
  · the master pool stamps all 8 as FLIP_FAN_WEAK_BOUNCE (BASE rows must NOT vanish via the screen — review I1).
  · builder freeze == live JSON == pydantic defaults for the two thresholds.
"""
import json
import os
import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.trading_engine import flip_fan_weak_bounce, _flip_filters

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _th(on=True, gap=0.0, slope=0.15):
    return SimpleNamespace(flip_fan_weak_bounce_enabled=on, flip_fan_weak_bounce_gap_max=gap,
                           flip_fan_weak_bounce_slope_max=slope)


def test_blocks_gentle_rise_below_trend():
    assert flip_fan_weak_bounce(_th(), -0.2104, 0.1042) is True     # AVAX B12
    assert flip_fan_weak_bounce(_th(), -0.1735, 0.1473) is True     # FET B12
    assert flip_fan_weak_bounce(_th(), -0.8077, 0.1439) is True     # POL B3
    assert flip_fan_weak_bounce(_th(), -0.0991, 0.0747) is True     # PEPE B8
    assert flip_fan_weak_bounce(_th(), -0.3697, 0.1297) is True     # LAYER BASE (the one blocked big winner)
    assert flip_fan_weak_bounce(_th(), -0.3990, 0.1168) is True     # ONDO BASE
    assert flip_fan_weak_bounce(_th(), -0.5250, 0.1115) is True     # SKYAI BASE
    assert flip_fan_weak_bounce(_th(), -0.4375, 0.1415) is True     # KAITO BASE
    assert flip_fan_weak_bounce(_th(), -0.3, -0.05) is True         # falling EMA20 below trend = even weaker


def test_passes_steep_or_above_trend():
    assert flip_fan_weak_bounce(_th(), -0.8217, 0.3093) is False    # H +1.28% — steep squeeze below trend
    assert flip_fan_weak_bounce(_th(), -0.6911, 0.1893) is False    # HOME winner
    assert flip_fan_weak_bounce(_th(), -0.1975, 0.2743) is False    # ETHFI winner
    assert flip_fan_weak_bounce(_th(), 0.2474, 0.0982) is False     # XMR — gentle but above trend


def test_strict_boundaries():
    assert flip_fan_weak_bounce(_th(), 0.0, 0.05) is False          # gap == max passes
    assert flip_fan_weak_bounce(_th(), -0.5, 0.15) is False         # slope == max passes
    assert flip_fan_weak_bounce(_th(), -1e-9, 0.1499) is True


def test_disabled_or_missing_threshold_fails_open():
    assert flip_fan_weak_bounce(_th(on=False), -0.5, 0.05) is False
    assert flip_fan_weak_bounce(SimpleNamespace(), -0.5, 0.05) is False
    assert flip_fan_weak_bounce(_th(gap=None), -0.5, 0.05) is False
    assert flip_fan_weak_bounce(_th(slope=float('nan')), -0.5, 0.05) is False


def test_missing_or_garbage_inputs_fail_open():
    for g, s in ((None, 0.05), (-0.5, None), (float('nan'), 0.05), (-0.5, float('inf')), ("x", 0.05)):
        assert flip_fan_weak_bounce(_th(), g, s) is False


def _ind(**kw):
    base = {'flip_dir': 'SHORT', 'btc_regime': 'STRONG_BEAR', 'pair_gap': -0.21, 'ema20_slope': 0.10}
    base.update(kw)
    return base


def _live_th(on=True):
    import config
    th = config.trading_config.thresholds.model_copy()
    th.flip_fan_weak_bounce_enabled = on
    th.flip_fan_weak_bounce_gap_max = 0.0
    th.flip_fan_weak_bounce_slope_max = 0.15
    return th


def _fails(source, ind, on=True):
    import config
    with patch.object(config.trading_config, 'thresholds', _live_th(on)):
        return _flip_filters(source, ind)[5]


def test_flip_filters_wires_the_gate_for_fan_shorts_only():
    assert "FLIP_FAN_WEAK_BOUNCE" in _fails('FAN_RATIO_GATE', _ind())
    assert "FLIP_FAN_WEAK_BOUNCE" not in _fails('FAN_RATIO_GATE', _ind(), on=False)
    assert "FLIP_FAN_WEAK_BOUNCE" not in _fails('FAN_RATIO_GATE', _ind(ema20_slope=0.31))
    assert "FLIP_FAN_WEAK_BOUNCE" not in _fails('FAN_RATIO_GATE', _ind(flip_dir='LONG'))
    assert "FLIP_FAN_WEAK_BOUNCE" not in _fails('PAIR_RSI_OB', _ind())
    assert "FLIP_FAN_WEAK_BOUNCE" not in _fails('FAN_RATIO_GATE', _ind(ema20_slope=None))


def _json():
    with open(os.path.join(ROOT, "trading_config.json")) as fh:
        return json.load(fh)["thresholds"]


def test_builder_freeze_matches_live_json():
    live = _json()
    with open(os.path.join(ROOT, "scripts", "build_master_pool.py")) as fh:
        src = fh.read()
    m = re.search(r"_FLIP_TH = SimpleNamespace\(flip_fan_weak_bounce_enabled=(\w+), flip_fan_weak_bounce_gap_max=([-0-9.]+),\s*"
                  r"flip_fan_weak_bounce_slope_max=([-0-9.]+)\)", src)
    assert m, "builder weak-bounce freeze not found"
    assert (m.group(1) == 'True') == bool(live["flip_fan_weak_bounce_enabled"])
    assert float(m.group(2)) == float(live["flip_fan_weak_bounce_gap_max"])
    assert float(m.group(3)) == float(live["flip_fan_weak_bounce_slope_max"])
    assert "'FLIP_FAN_WEAK_BOUNCE'" in src


def test_pydantic_threshold_defaults_match_json():
    import config
    f = config.SignalThresholds.model_fields
    live = _json()
    assert float(f["flip_fan_weak_bounce_gap_max"].default) == float(live["flip_fan_weak_bounce_gap_max"])
    assert float(f["flip_fan_weak_bounce_slope_max"].default) == float(live["flip_fan_weak_bounce_slope_max"])


def test_engine_gate_sits_in_the_fan_short_block_and_flip_inputs_carry_slope():
    with open(os.path.join(ROOT, "services", "trading_engine.py")) as fh:
        src = fh.read()
    fan_block = src[src.index('if source == "FAN_RATIO_GATE":\n            stretch'):]
    fan_block = fan_block[:fan_block.index('# 2) regime block')]
    assert 'flip_fan_weak_bounce(th, ind.get(\'pair_gap\'), ind.get(\'ema20_slope\'))' in fan_block
    assert "'ema20_slope': (_ef.get('entry_ema20_slope')" in src


def test_master_pool_stamps_all_eight_blocked_flips():
    import pandas as pd
    p = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
    if not os.path.exists(p):
        pytest.skip("pool not built")
    d = pd.read_csv(p, low_memory=False, usecols=["stack_block_reason", "era"])
    blk = d[d.stack_block_reason == "FLIP_FAN_WEAK_BOUNCE"]
    assert len(blk) >= 8, f"only {len(blk)} weak-bounce rows — BASE cohort deleted by the screen again?"
    assert (blk.era == "BASE").sum() >= 4


def test_screen_does_not_carry_the_gate():
    """Review I1: SCREENED_BASELINE feeds the pool's BASE era; screening the gate there deletes the cohort."""
    with open(os.path.join(ROOT, "scripts", "screen_pool.py")) as fh:
        src = fh.read()
    body = src[src.index("def flip_ind"):src.index("_OFF30 = {}")]
    assert "'ema20_slope':" not in body
