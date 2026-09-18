"""🔥 Sep-18 long heat block — the pure rule shared by the engine and the pool builders.
Invariants: ALL enabled legs must hold; washed-out BTC is exempt; any missing ENABLED reading
fails OPEN (a block never fires on missing data); a 0 threshold drops that leg; disabled = never blocks.
"""
from types import SimpleNamespace

from services.trading_engine import long_heat_eval


def _th(**kw):
    d = dict(long_heat_block_enabled=True, long_heat_btc_slope_min=0.07, long_heat_btc_rsi_prev_min=64.0,
             long_heat_bull_pct_min=80.0, long_heat_exempt_off30d_max=-10.0)
    d.update(kw)
    return SimpleNamespace(**d)


def test_blocks_when_all_three_hot_and_not_washed_out():
    # OP Sep-17: slope 0.10, RSI prev 69.1, bull 88.4, BTC ~-7% vs 30d high
    assert long_heat_eval(_th(), 0.10, 69.1, 88.4, -7.0) == (3, True)


def test_thresholds_are_inclusive():
    assert long_heat_eval(_th(), 0.07, 64.0, 80.0, -9.99) == (3, True)


def test_two_flags_never_block():
    # DOT Sep-17: RSI prev 54.7 -> 2 flags
    assert long_heat_eval(_th(), 0.12, 54.7, 93.0, -7.0) == (2, False)


def test_washed_out_is_exempt_and_boundary_is_exempt():
    assert long_heat_eval(_th(), 0.11, 67.4, 87.8, -19.4) == (3, False)
    assert long_heat_eval(_th(), 0.11, 67.4, 87.8, -10.0) == (3, False)


def test_missing_readings_fail_open():
    assert long_heat_eval(_th(), 0.10, 69.1, 88.4, None) == (3, False)      # 30d reading stale/unknown
    assert long_heat_eval(_th(), None, 69.1, 88.4, -5.0) == (2, False)      # a heat leg unreadable
    assert long_heat_eval(_th(), None, None, None, -5.0) == (None, False)


def test_disabled_never_blocks_but_still_counts_flags():
    assert long_heat_eval(_th(long_heat_block_enabled=False), 0.10, 69.1, 88.4, -5.0) == (3, False)


def test_zero_threshold_drops_that_leg():
    th = _th(long_heat_bull_pct_min=0.0)
    assert long_heat_eval(th, 0.10, 69.1, 10.0, -5.0) == (2, True)          # bull leg off -> 2 of 2 enabled
    assert long_heat_eval(_th(long_heat_exempt_off30d_max=0.0), 0.10, 69.1, 88.4, -25.0) == (3, True)   # no exemption
    assert long_heat_eval(_th(long_heat_exempt_off30d_max=0.0), 0.10, 69.1, 88.4, None) == (3, True)    # exemption off needs no reading


def test_all_legs_off_never_blocks():
    th = _th(long_heat_btc_slope_min=0, long_heat_btc_rsi_prev_min=0, long_heat_bull_pct_min=0)
    assert long_heat_eval(th, 0.5, 90, 99, -1.0) == (None, False)


def test_garbage_inputs_fail_open():
    assert long_heat_eval(_th(), "x", 69.1, 88.4, -5.0) == (2, False)
