"""Exit-ladder invariants — _bullrun_exit_for + _bullrun_ladder_floor.

These pin the replay-validated sleeve exit stack: SL widening, BE arm/lock,
2xATR trail, ladder floors, fail-safe. Config values are the LIVE defaults
(base -0.7, widen 1.5xATR floored -1.2, arm 1.0, lock 0.2, trail 2.0).
"""
import config
from services.trading_engine import _bullrun_exit_for, _bullrun_ladder_floor

LADDER = "4.0:3.5, 5.0:4.5, 6.0:5.5, 8.0:7.0, 10.0:9.0, 12.0:11.0, 15.0:13.5, 20.0:18.0, 25.0:22.5, 30.0:27.0"


import pytest


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    # caveman-review: monkeypatch (auto-restored) — never mutate the real config singleton
    th = config.trading_config.thresholds
    monkeypatch.setattr(th, "bullrun_base_sl_pct", -0.7)
    monkeypatch.setattr(th, "bullrun_be_arm_pct", 1.0)
    monkeypatch.setattr(th, "bullrun_be_lock_pct", 0.2)
    monkeypatch.setattr(th, "bullrun_trail_atr_mult", 2.0)
    monkeypatch.setattr(th, "bullrun_ladder", LADDER)
    monkeypatch.setattr(th, "sl_atr_multiplier", 1.5)
    monkeypatch.setattr(th, "sl_atr_widen_floor_pct", -1.2)


def test_base_stop_no_atr():
    close, reason, stop = _bullrun_exit_for(-0.71, 0.0, 0.0)
    assert close and reason == "STOP_LOSS" and stop == -0.7
    close, _, _ = _bullrun_exit_for(-0.69, 0.0, 0.0)
    assert not close


def test_atr_widened_stop_and_floor():
    # atr 0.6 -> widened -0.9 (within floor -1.2): stop = min(-0.7, -0.9) = -0.9
    close, reason, stop = _bullrun_exit_for(-0.85, 0.0, 0.6)
    assert not close and abs(stop - (-0.9)) < 1e-9
    close, reason, stop = _bullrun_exit_for(-0.91, 0.0, 0.6)
    assert close and reason == "STOP_LOSS"
    # atr 1.0 -> raw widen -1.5 clamped by the -1.2 floor
    _, _, stop = _bullrun_exit_for(-0.5, 0.0, 1.0)
    assert abs(stop - (-1.2)) < 1e-9


def test_be_arm_and_lock():
    # peak below arm: still on SL
    close, reason, _ = _bullrun_exit_for(0.1, 0.9, 0.5)
    assert not close
    # armed (peak 1.0), trail line 1.0 - 2*0.5 = 0.0 < lock 0.2 -> BE floor 0.2
    close, reason, stop = _bullrun_exit_for(0.19, 1.0, 0.5)
    assert close and reason == "BREAKEVEN_EXIT" and abs(stop - 0.2) < 1e-9
    close, _, _ = _bullrun_exit_for(0.21, 1.0, 0.5)
    assert not close


def test_trailing_stop():
    # peak 2.0, atr 0.5 -> trail 1.0 (> lock): trailing regime
    close, reason, stop = _bullrun_exit_for(0.99, 2.0, 0.5)
    assert close and reason == "TRAILING_STOP" and abs(stop - 1.0) < 1e-9
    close, _, _ = _bullrun_exit_for(1.01, 2.0, 0.5)
    assert not close


def test_ladder_floor_beats_trail():
    # peak 10, atr 2.0 -> trail 6.0; ladder rung 10.0:9.0 -> floor 9.0 wins
    close, reason, stop = _bullrun_exit_for(8.9, 10.0, 2.0)
    assert close and reason == "LADDER_FLOOR" and abs(stop - 9.0) < 1e-9
    close, _, _ = _bullrun_exit_for(9.1, 10.0, 2.0)
    assert not close


def test_ladder_floor_parsing():
    assert _bullrun_ladder_floor(3.9, LADDER) is None
    assert _bullrun_ladder_floor(4.0, LADDER) == 3.5
    assert _bullrun_ladder_floor(29.9, LADDER) == 22.5
    assert _bullrun_ladder_floor(35.0, LADDER) == 27.0
    # malformed rung (floor >= peak = take-profit, not lock) is ignored
    assert _bullrun_ladder_floor(5.0, "4.0:4.5") is None
    assert _bullrun_ladder_floor(5.0, "") is None


def test_fail_safe_never_stopless():
    # garbage inputs must fall back to the plain -0.7 check, never raise
    close, reason, stop = _bullrun_exit_for(-0.8, None, None)
    assert close and stop <= -0.7
