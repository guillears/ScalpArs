"""🐻 Sep-19 bear-run bypass breadth floor (bull-run 57c lesson mirrored) — the pure rule the
bypass eligibility calls. Invariants: 0/None/blank floor = off (always admits); a REAL reading is
judged bear_pct >= floor (inclusive); warm-up (bear AND bull both 0/None = breadth not computed
since restart) fails CLOSED — the bypass is the aggressive mode and waits one scan cycle (the
bull sleeve's B4 REARM lesson: 5 of 7 blind entries, 4 losers); bear=0 with bull>0 is a REAL
reading (bulls everywhere) and blocks; garbage never raises and admits.
"""
from types import SimpleNamespace

from services.trading_engine import bearrun_breadth_ok


def _th(v=40.0):
    return SimpleNamespace(bearrun_breadth_min=v)


def test_blocks_btc_falling_alone():
    # BTC -4%/24h but bears are only a third of the market = squeeze setup
    assert bearrun_breadth_ok(_th(), 33.3, 55.6) is False


def test_admits_confirmed_downmove_inclusive():
    assert bearrun_breadth_ok(_th(), 40.0, 30.0) is True
    assert bearrun_breadth_ok(_th(), 62.8, 27.9) is True


def test_warmup_fails_closed():
    assert bearrun_breadth_ok(_th(), 0.0, 0.0) is False
    assert bearrun_breadth_ok(_th(), None, None) is False


def test_real_zero_bear_with_bull_reading_blocks():
    # bull readings prove the scan ran — bear 0 is a real (bullish) market, not warm-up
    assert bearrun_breadth_ok(_th(), 0.0, 84.4) is False


def test_off_when_zero_none_blank():
    assert bearrun_breadth_ok(_th(0), 0.0, 0.0) is True
    assert bearrun_breadth_ok(_th(None), 10.0, 80.0) is True
    assert bearrun_breadth_ok(_th(''), 10.0, 80.0) is True
    assert bearrun_breadth_ok(SimpleNamespace(), 10.0, 80.0) is True


def test_never_raises_on_garbage():
    assert bearrun_breadth_ok(_th('forty'), 50.0, 30.0) is True
    assert bearrun_breadth_ok(_th(), 'x', 'y') is True
