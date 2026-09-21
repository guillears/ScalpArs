"""📏 Sep-21 (105, operator-raised) — breadth scoped to the universe the sleeve can actually trade.

The live breadth floor reads the SCAN-WIDE number (`trading_pairs_limit`, 50 today) while the
bull-run sleeve only trades `br_rank <= bullrun_universe_size` (10). The gate therefore measures
~40 pairs it will never buy. These columns exist to MEASURE that mismatch before anyone moves a
threshold off it — they gate nothing.

Invariants pinned here: only ranks 1..N count · blacklisted pairs (br_rank None) are EXCLUDED,
because a pair that cannot be bought must not vote on participation · neutral pairs dilute both
sides rather than being dropped · an empty or unusable universe yields None, not 0.0 · and the
DENOMINATOR is returned, because this slice can legitimately be 2-3 rows and still read 100.0.

⚠ CORRECTED after deep review: None is NOT a safe sentinel for the existing floor. `bullrun_breadth_ok`
does `float(bull_pct or 0.0)` and coerces None→0.0 straight into its REARM fail-closed arm, so wiring
these columns into a gate needs a real no-reading sentinel first. Worse at N=10 than it was at N=50:
an all-NEUTRAL top-10 is a plausible real reading that returns a genuine (0.0, 0.0) and is therefore
indistinguishable from warm-up. Both are why these columns gate NOTHING today.
"""
import pytest

from services.trading_engine import tradeable_breadth


def _p(br_rank, regime):
    return {"br_rank": br_rank, "breadth_regime": regime}


def test_only_the_tradeable_top_n_counts():
    """Ranks 11+ are scanned but never traded — they must not move the number."""
    rows = [_p(i, "BULLISH") for i in range(1, 11)] + [_p(i, "BEARISH") for i in range(11, 51)]
    assert tradeable_breadth(rows, 10)[:2] == (100.0, 0.0)
    # the scan-wide reading over the same rows would be 20% bullish — the whole point of 105
    assert tradeable_breadth(rows, 50)[:2] == (20.0, 80.0)


def test_blacklisted_pairs_are_excluded_not_counted_as_neutral():
    """br_rank is None for blacklisted pairs; they cannot be bought so they cannot vote."""
    rows = [_p(1, "BULLISH"), _p(2, "BULLISH"), _p(None, "BEARISH"), _p(None, "BEARISH")]
    assert tradeable_breadth(rows, 10)[:2] == (100.0, 0.0)


def test_neutral_pairs_dilute_both_sides():
    rows = [_p(1, "BULLISH"), _p(2, "BEARISH"), _p(3, "NEUTRAL"), _p(4, "NEUTRAL")]
    bull, bear, _n = tradeable_breadth(rows, 10)
    assert bull == 25.0 and bear == 25.0


def test_mixed_reading_matches_a_hand_count():
    rows = [_p(1, "BULLISH"), _p(2, "BULLISH"), _p(3, "BULLISH"),
            _p(4, "BEARISH"), _p(5, "NEUTRAL")]
    assert tradeable_breadth(rows, 10)[:2] == (60.0, 20.0)


def test_empty_universe_is_none_never_zero():
    """A missing reading must not read as '0% bullish' — that is a floor breach, not a warm-up."""
    assert tradeable_breadth([], 10)[:2] == (None, None)
    assert tradeable_breadth(None, 10)[:2] == (None, None)
    assert tradeable_breadth([_p(None, "BULLISH")], 10)[:2] == (None, None)   # all blacklisted
    assert tradeable_breadth([_p(40, "BULLISH")], 10)[:2] == (None, None)     # none inside the top-10


def test_bad_universe_size_yields_no_reading():
    rows = [_p(1, "BULLISH")]
    for n in (0, -5, None, "nonsense"):
        assert tradeable_breadth(rows, n)[:2] == (None, None)


def test_garbage_rows_are_skipped_not_fatal():
    rows = [_p(1, "BULLISH"), _p("nonsense", "BEARISH"), {"breadth_regime": "BEARISH"}, _p(2, "BULLISH")]
    assert tradeable_breadth(rows, 10)[:2] == (100.0, 0.0)


def test_rank_zero_is_not_tradeable():
    """br_rank is 1-based; a 0 would be a stamping bug and must not silently count."""
    assert tradeable_breadth([_p(0, "BULLISH"), _p(1, "BEARISH")], 10)[:2] == (0.0, 100.0)


def test_universe_size_boundary_is_inclusive():
    rows = [_p(10, "BULLISH"), _p(11, "BEARISH")]
    assert tradeable_breadth(rows, 10)[:2] == (100.0, 0.0)


def test_percentages_are_rounded_like_the_scan_wide_stat():
    rows = [_p(i, "BULLISH") for i in range(1, 4)] + [_p(i, "BEARISH") for i in range(4, 8)]
    bull, bear, _n = tradeable_breadth(rows, 10)
    assert bull == pytest.approx(42.9) and bear == pytest.approx(57.1)


def test_denominator_is_returned_so_a_thin_reading_is_auditable():
    """A 100.0 over 2 pairs and over 10 must be distinguishable AFTER the fact — that asymmetry
    is invisible at the scan-wide N=50 and is exactly how a false finding gets born."""
    thin = [_p(1, "BULLISH"), _p(2, "BULLISH")]
    full = [_p(i, "BULLISH") for i in range(1, 11)]
    assert tradeable_breadth(thin, 10) == (100.0, 0.0, 2)
    assert tradeable_breadth(full, 10) == (100.0, 0.0, 10)


def test_no_reading_returns_a_zero_denominator():
    assert tradeable_breadth([], 10) == (None, None, 0)
    assert tradeable_breadth([_p(1, "BULLISH")], 0) == (None, None, 0)
