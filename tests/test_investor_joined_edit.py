"""🗓 (11-05) editing an investor's JOINED date must never change which ledger rows belong to them (main.investor_joined_change)."""
from datetime import date, datetime

import pytest

from main import investor_joined_change as jc

TODAY = date(2026, 11, 5)
CREATED = datetime(2026, 10, 23, 15, 0, 0)                      # naive UTC, Oct 23 12:00 UTC-3
LEDGER = [datetime(2026, 10, 23, 15, 0, 1), datetime(2026, 10, 26, 18, 0)]


def test_earlier_day_is_stored_as_midnight_utc3():
    assert jc(CREATED, date(2026, 10, 1), LEDGER, TODAY) == datetime(2026, 10, 1, 3, 0)


def test_same_day_is_a_no_op():
    assert jc(CREATED, date(2026, 10, 23), LEDGER, TODAY) is None


def test_after_first_deposit_is_refused():
    with pytest.raises(ValueError, match="first ledger entry"):
        jc(CREATED, date(2026, 10, 24), LEDGER, TODAY)


def test_pulling_in_a_previous_holders_rows_is_refused():
    old_holder = [datetime(2026, 6, 1, 12, 0)] + LEDGER
    with pytest.raises(ValueError, match="previous investor"):
        jc(CREATED, date(2026, 5, 30), old_holder, TODAY)
    assert jc(CREATED, date(2026, 6, 2), old_holder, TODAY) == datetime(2026, 6, 2, 3, 0)


def test_future_is_refused():
    with pytest.raises(ValueError, match="future"):
        jc(CREATED, date(2026, 11, 6), LEDGER, TODAY)


def test_utc3_day_boundary_matches_the_statement():
    created = datetime(2026, 10, 23, 2, 0)                      # Oct 22 23:00 UTC-3 → shown as Oct 22
    assert jc(created, date(2026, 10, 22), [created], TODAY) is None
