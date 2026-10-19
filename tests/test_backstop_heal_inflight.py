"""🛡 DECISION_LOG 308: the backstop heal sweep must not re-place a stop while the bot's OWN close is in flight (close_position nulls the
backstop id just before its market close — KMNOUSDT 2026-10-17 00:32: the sweep raced a stop into the gap, Binance -4509 on the flat
symbol), but must still heal a close that is stuck past the grace window (never leave a position naked)."""
import inspect
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.golive_guards import backstop_heal_skip_inflight, BACKSTOP_CLOSE_GRACE_S  # noqa: E402

NOW = datetime(2026, 10, 17, 0, 32, 20)


def test_inflight_close_is_skipped():
    assert backstop_heal_skip_inflight(True, NOW - timedelta(seconds=1), NOW) is True
    assert backstop_heal_skip_inflight(True, NOW - timedelta(seconds=BACKSTOP_CLOSE_GRACE_S - 1), NOW) is True


def test_stuck_or_no_close_is_healed():
    assert backstop_heal_skip_inflight(True, NOW - timedelta(seconds=BACKSTOP_CLOSE_GRACE_S + 1), NOW) is False   # stuck close → heal
    assert backstop_heal_skip_inflight(False, NOW, NOW) is False
    assert backstop_heal_skip_inflight(None, None, NOW) is False
    assert backstop_heal_skip_inflight(True, "garbage", NOW) is False               # unreadable → heal (fail toward protection)
    assert backstop_heal_skip_inflight(True, None, NOW) is False                    # flag without a timestamp → heal
    assert BACKSTOP_CLOSE_GRACE_S <= 30                                             # short: the real gap is ~1–3 s


def test_clock_skew_future_timestamp_is_in_flight():
    assert backstop_heal_skip_inflight(True, NOW + timedelta(seconds=5), NOW) is True
    assert backstop_heal_skip_inflight(True, NOW + timedelta(hours=1), NOW) is False   # far-future stamp never blocks healing


def test_heal_sweep_uses_the_guard():
    import services.trading_engine as T
    src = inspect.getsource(T)
    i = src.index("BACKSTOP SWEEP (live only)")
    block = src[i:i + 3000]
    assert "backstop_heal_skip_inflight(" in block and "_bk_heal_next" in block


def test_close_marks_in_progress_before_cancelling_the_backstop():
    """the guard relies on _close_position_locked publishing closing_in_progress BEFORE it cancels + nulls the backstop."""
    import services.trading_engine as T
    src = inspect.getsource(T.TradingEngine._close_position_locked)
    assert src.index("_mark_close_in_progress(") < src.index("cancel_backstop_stop(order.pair, order.backstop_algo_id)")
