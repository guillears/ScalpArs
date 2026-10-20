"""🪜 DECISION_LOG 319: order paths size 5 % under each exchange tier's max notional (SOLVUSDT fade 2026-10-20 07:22 was sized to
exactly the $10,000 tier at 10× and Binance rejected it −2027)."""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import services.trading_engine as T  # noqa: E402

SOLV = [(5000.0, 20.0), (10000.0, 10.0), (25000.0, 5.0)]


def test_solv_case_stays_under_the_tier():
    lev, notional, _, _ = T.leverage_bracket_limit(T.bracket_tiers_with_headroom(SOLV), 20, 1286.07)
    assert notional <= 10000 * T.BRACKET_CAP_SAFETY + 1e-6 and notional < 10000          # never exactly at the exchange's line
    assert T.BRACKET_CAP_SAFETY == 0.95


def test_headroom_helper_is_total():
    assert T.bracket_tiers_with_headroom(None) is None and T.bracket_tiers_with_headroom([]) == []
    assert T.bracket_tiers_with_headroom([(100.0, 20.0)]) == [(95.0, 20.0)]
    assert T.bracket_tiers_with_headroom([("x", 1)]) == [("x", 1)]                         # bad rows pass through, never raise


def test_order_paths_use_the_headroom():
    src = inspect.getsource(T)
    assert "leverage_bracket_limit(bracket_tiers_with_headroom(" in src                   # bot open path
    assert "_ml_tiers = bracket_tiers_with_headroom(" in src                              # momentum keep
    assert "_tiers = bracket_tiers_with_headroom(" in src                                 # manual open path


def test_manual_panel_limits_use_the_same_headroom():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")).read()
    i = src.index("async def manual_limits("); blk = src[i:i + 1400]
    assert "bracket_tiers_with_headroom" in blk                                           # panel max == what the server accepts
