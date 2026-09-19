"""🛡 Sep-19 (57f) entry-dislocation guard — the pure predicate both maker paths call.
Invariants: 0/None/blank max = guard OFF; symmetric (catches limit BELOW mid = gapped book AND
fill ABOVE signal = ran-away chase); strictly-greater comparison (a fill exactly at the bound is
allowed); fail-open on garbage/zero prices — a guard must never block on missing data.
Born from ONE −$225 in 0.996s (limit rested −3.2% below market, filled mid-flash-crash).
"""
from services.trading_engine import bullrun_disloc_exceeded


def test_one_case_blocks():
    # ONE 2026-09-19 20:01: signal/current 0.00403, limit rested at 0.0039 = 3.23% away
    assert bullrun_disloc_exceeded(0.3, 0.00403, 0.0039) is True


def test_ran_away_chase_blocks():
    # ONG class: fallback would fill 0.75% ABOVE the signal price
    assert bullrun_disloc_exceeded(0.3, 100.0, 100.75) is True


def test_normal_book_passes():
    # healthy top-10 book: 2 real ticks ≈ 0.004%
    assert bullrun_disloc_exceeded(0.3, 100.0, 99.996) is False
    assert bullrun_disloc_exceeded(0.3, 100.0, 100.29) is False


def test_bound_is_exclusive():
    assert bullrun_disloc_exceeded(0.3, 100.0, 100.30) is False   # exactly 0.3% = allowed
    assert bullrun_disloc_exceeded(0.3, 100.0, 100.31) is True


def test_off_when_zero_none_blank():
    assert bullrun_disloc_exceeded(0, 100.0, 90.0) is False
    assert bullrun_disloc_exceeded(None, 100.0, 90.0) is False
    assert bullrun_disloc_exceeded('', 100.0, 90.0) is False


def test_fail_open_on_garbage():
    assert bullrun_disloc_exceeded(0.3, 0.0, 100.0) is False      # zero ref price
    assert bullrun_disloc_exceeded(0.3, 100.0, 0.0) is False      # zero px
    assert bullrun_disloc_exceeded(0.3, None, 100.0) is False
    assert bullrun_disloc_exceeded('x', 100.0, 90.0) is False
