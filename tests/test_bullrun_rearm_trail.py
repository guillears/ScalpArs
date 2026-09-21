"""🌊 Sep-21 (57i) door-conditional BR trail — REARM banks, GREEN gives room.

Invariants: the REARM multiplier applies ONLY to door='REARM'; GREEN/None/unknown keep the
standard multiplier; 0/absent REARM field falls back to the GREEN multiplier (the documented
off-switch); the BE lock still floors the trail line; the ladder still overrides both; and the
stop-loss branch (peak below arm) is untouched by the door.
"""
from types import SimpleNamespace

from services.trading_engine import _bullrun_exit_for


def _th(rearm_trail=1.0, trail=2.0):
    return SimpleNamespace(
        bullrun_base_sl_pct=-0.7, sl_atr_multiplier=1.5, sl_atr_widen_floor_pct=-1.2,
        bullrun_be_arm_pct=1.0, bullrun_be_lock_pct=0.2,
        bullrun_trail_atr_mult=trail, bullrun_rearm_trail_atr_mult=rearm_trail,
        bullrun_ladder="",
    )


def _stop(th, peak, atr, door, monkey):
    monkey.setattr("services.trading_engine.config",
                   SimpleNamespace(trading_config=SimpleNamespace(thresholds=th)), raising=False)
    # pnl far above any stop so we only read the returned stop line
    _, _, stop = _bullrun_exit_for(peak, peak, atr, door)
    return stop


def test_rearm_trails_tighter_than_green(monkeypatch):
    th = _th()
    atr, peak = 0.8, 3.0
    green = _stop(th, peak, atr, "GREEN", monkeypatch)
    rearm = _stop(th, peak, atr, "REARM", monkeypatch)
    assert green == 3.0 - 2.0 * 0.8          # peak − 2×ATR
    assert rearm == 3.0 - 1.0 * 0.8          # peak − 1×ATR  → higher stop, banks earlier
    assert rearm > green


def test_green_and_unknown_doors_unchanged(monkeypatch):
    th = _th()
    expected = 3.0 - 2.0 * 0.8
    for door in ("GREEN", None, "", "green", "AMBER"):
        assert _stop(th, 3.0, 0.8, door, monkeypatch) == expected


def test_case_insensitive_rearm(monkeypatch):
    th = _th()
    assert _stop(th, 3.0, 0.8, "rearm", monkeypatch) == 3.0 - 1.0 * 0.8


def test_zero_or_missing_rearm_mult_falls_back_to_green(monkeypatch):
    expected = 3.0 - 2.0 * 0.8
    assert _stop(_th(rearm_trail=0.0), 3.0, 0.8, "REARM", monkeypatch) == expected
    assert _stop(SimpleNamespace(
        bullrun_base_sl_pct=-0.7, sl_atr_multiplier=1.5, sl_atr_widen_floor_pct=-1.2,
        bullrun_be_arm_pct=1.0, bullrun_be_lock_pct=0.2,
        bullrun_trail_atr_mult=2.0, bullrun_ladder=""), 3.0, 0.8, "REARM", monkeypatch) == expected


def test_be_lock_still_floors_the_rearm_trail(monkeypatch):
    # peak 1.1 with ATR 2.0: REARM trail line = 1.1 − 2.0 = −0.9 → the +0.2 lock must win
    assert _stop(_th(), 1.1, 2.0, "REARM", monkeypatch) == 0.2


def test_stop_loss_branch_is_door_independent(monkeypatch):
    th = _th()
    monkeypatch.setattr("services.trading_engine.config",
                        SimpleNamespace(trading_config=SimpleNamespace(thresholds=th)), raising=False)
    for door in ("REARM", "GREEN", None):
        close, reason, stop = _bullrun_exit_for(-1.5, 0.3, 0.8, door)   # never armed
        assert close is True and reason == "STOP_LOSS"
        assert stop == -1.2                                              # ATR-widened, floored
