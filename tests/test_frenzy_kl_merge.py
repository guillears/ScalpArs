"""⚡ Oct-5 (DECISION_LOG 213): FRENZY's incremental 5m window must equal what a full 1500-bar read returns at the same moment —
same rows, same forming bar, same length — or merge_klines returns None and the engine falls back to the full read."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.frenzy import merge_klines, frenzy_walk, BAR_MS  # noqa: E402

T0 = 1_790_000_000_000 // BAR_MS * BAR_MS


def _exchange(n_closed, forming_close=None):
    """the exchange's view after n_closed bars: rows 0..n_closed-1 closed, plus the forming bar n_closed (its close still moving)."""
    rows = [[T0 + k * BAR_MS, 1.0 + k * 1e-3, 1.01 + k * 1e-3, 0.99 + k * 1e-3, 1.0 + k * 1e-3 + 1e-4, 100.0 + k] for k in range(n_closed)]
    fc = 9.9 if forming_close is None else forming_close
    rows.append([T0 + n_closed * BAR_MS, 1.0, 1.0, 1.0, fc, 1.0])
    return rows


def _full(n_closed, keep=1500, forming_close=None):
    return _exchange(n_closed, forming_close)[-keep:]


def test_incremental_equals_full_read_over_many_bars():
    cache = _full(2000, forming_close=5.0)            # a full read taken while bar 2000 was forming
    for n in range(2001, 2040):                       # one pass per bar
        cache = merge_klines(cache, _full(n)[-5:], 1500)
        assert cache == _full(n), n                   # rows, forming bar and length all identical
    assert len(cache) == 1500


def test_previous_forming_bar_is_overwritten_by_its_final_values():
    cache = _full(2000, forming_close=5.0)
    m = merge_klines(cache, _full(2001)[-5:], 1500)
    assert m[-2] == _exchange(2001)[2000]             # bar 2000 now carries its CLOSED values, not the forming 5.0


def test_up_to_four_missed_bars_still_join_with_a_five_bar_tail():
    cache = _full(2000)
    assert merge_klines(cache, _full(2004)[-5:], 1500) == _full(2004)


def test_a_gap_returns_none_so_the_engine_reads_in_full():
    cache = _full(2000)
    assert merge_klines(cache, _full(2010)[-5:], 1500) is None


def test_a_tail_that_starts_right_after_the_cached_forming_bar_is_refused():
    cache = _full(2000, forming_close=5.0)            # bar 2000 cached while forming
    assert merge_klines(cache, _full(2005)[-5:], 1500) is None   # (review) would otherwise keep the half-built bar 2000


def test_a_young_pair_window_grows_like_a_full_read():
    cache = _full(800, keep=1500)                     # < 1500 bars of history
    for n in range(801, 806):
        cache = merge_klines(cache, _full(n)[-5:], 1500)
        assert cache == _full(n, keep=1500)


def test_bad_inputs_return_none():
    good = _full(2000)
    assert merge_klines([], good[-5:], 1500) is None
    assert merge_klines(good, [], 1500) is None
    holey = good[-5:]; holey = holey[:2] + holey[3:]   # fresh rows not contiguous
    assert merge_klines(good, holey, 1500) is None
    assert merge_klines(good, _full(1990)[-5:], 1500) is None   # a fresh read older than the cache
    assert merge_klines(good, [["x"]], 1500) is None


def test_frenzy_decision_is_identical_on_the_merged_window():
    class Th:
        frenzy_spike_ret_pct = 5.0; frenzy_spike_vol_mult = 10.0; frenzy_spike_min_hour_usd = 0.0
        frenzy_min_hours = 2.0; frenzy_state_vol_mult = 5.0; frenzy_max_hours = 96.0
    cache = _full(2000)
    for n in range(2001, 2010):
        cache = merge_klines(cache, _full(n)[-5:], 1500)
        closed_m = [r for r in cache if r[0] + BAR_MS <= T0 + n * BAR_MS]
        closed_f = [r for r in _full(n) if r[0] + BAR_MS <= T0 + n * BAR_MS]
        assert frenzy_walk(closed_m, 1000.0, Th()) == frenzy_walk(closed_f, 1000.0, Th())
