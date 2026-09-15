"""🐻 gate 60 — Bear-Run monitor pure math: Schmitt band, squeeze latch, sign normalisation, bypass-list parsing.
Deleting the sleeve's state machine or flipping a comparison must fail here (no network, no DB)."""
from services.trading_engine import _bearrun_next_state as nxt, _bearrun_bypass_set as bset

BASE = dict(r24_on=4.0, r24_off=3.0, below_on=50.0, below_off=47.0, eff_on=0.15, eff_off=0.15, latch_r6h=3.0, latch_ema50=True)


def test_turn_on_needs_all_three_legs():
    assert nxt(False, -4.0, 50.0, 0.15, 0.0, 100.0, 105.0, **BASE) == (True, False)     # at the bar on every leg
    assert nxt(False, -3.9, 50.0, 0.15, 0.0, 100.0, 105.0, **BASE) == (False, False)    # r24 short of the bar
    assert nxt(False, -4.0, 49.9, 0.15, 0.0, 100.0, 105.0, **BASE) == (False, False)    # share short
    assert nxt(False, -4.0, 50.0, 0.149, 0.0, 100.0, 105.0, **BASE) == (False, False)   # efficiency short


def test_schmitt_stay_band_holds_between_off_and_on_bars():
    assert nxt(True, -3.0, 47.0, 0.15, 0.0, 100.0, 105.0, **BASE) == (True, False)      # stays ON inside the band
    assert nxt(False, -3.0, 47.0, 0.15, 0.0, 100.0, 105.0, **BASE) == (False, False)    # same readings do NOT turn it ON
    assert nxt(True, -2.9, 47.0, 0.15, 0.0, 100.0, 105.0, **BASE) == (False, False)     # drops out below the stay band


def test_squeeze_latch_forces_off_and_blocks_turn_on():
    assert nxt(True, -6.0, 80.0, 0.30, 3.0, 100.0, 105.0, **BASE) == (False, True)      # r6h at +3 → OFF, latched
    assert nxt(False, -6.0, 80.0, 0.30, 3.5, 100.0, 105.0, **BASE) == (False, True)     # cannot turn ON while squeezed
    assert nxt(True, -6.0, 80.0, 0.30, 0.0, 106.0, 105.0, **BASE) == (False, True)      # price above the 1h EMA50 → OFF
    assert nxt(True, -6.0, 80.0, 0.30, 0.0, 106.0, 105.0, **{**BASE, 'latch_ema50': False}) == (True, False)  # leg 2 off
    assert nxt(True, -6.0, 80.0, 0.30, 9.0, 100.0, 105.0, **{**BASE, 'latch_r6h': 0}) == (True, False)        # leg 1 off (0)
    assert nxt(True, -6.0, 80.0, 0.30, 0.0, 100.0, None, **BASE) == (True, False)       # unknown EMA50 → leg 2 cannot fire


def test_sign_slip_and_missing_readings_are_safe():
    assert nxt(False, -4.0, 50.0, 0.15, 0.0, 100.0, 105.0, **{**BASE, 'r24_on': -4.0, 'r24_off': -3.0}) == (True, False)  # negative thresholds normalised
    assert nxt(True, None, 50.0, 0.15, 0.0, 100.0, 105.0, **BASE) == (False, False)      # missing reading → OFF, not a crash


def test_bypass_set_parsing():
    assert bset("BTC_ADX_BLOCK_SHORT, btc_slope_gate ,,") == {"BTC_ADX_BLOCK_SHORT", "BTC_SLOPE_GATE"}
    assert bset("") == set() and bset(None) == set()
