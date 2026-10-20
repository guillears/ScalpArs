"""🧍 DECISION_LOG 322: after a resting-TP row's close the residual sweep sells only the row's OWN shortfall — BTW #48 2026-10-20: the
bot's MANUAL_SL closed its 2,714 BTW, then swept the operator's separate 2,779 app-held BTW as "residual"."""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.golive_guards import residual_own_qty  # noqa: E402


def test_btw_case_operator_position_untouched():
    assert residual_own_qty(2779.0, 2714.0, 2714.0, 2714.0) == 0.0          # our close filled everything → none of it is ours


def test_known_shortfall_is_ours_only():
    assert residual_own_qty(2779.0 + 14.0, 2714.0, 2700.0, 2714.0) == 14.0   # our 14 unfilled + their 2,779
    assert abs(residual_own_qty(10.0, 2714.4, 2714.0, 2714.4) - 0.4) < 1e-9            # a close snapped below the remainder → the difference is ours
    assert residual_own_qty(5.0, 100.0, 50.0, 100.0) == 5.0                   # never more than is actually there


def test_unknown_fill_sweeps_everything_as_before():
    """both reviews: an unconfirmed fill must never leave our own coins without a stop — the whole residual is ours, as before 322."""
    assert residual_own_qty(9.0, 100.0, 0, 100.0) == 9.0
    assert residual_own_qty(2779.0, 2714.0, None, 2714.0) == 2779.0
    assert residual_own_qty(0, 100.0, None, 100.0) == 0.0 and residual_own_qty(-3, 100.0, 1, 100.0) == 0.0
    assert residual_own_qty("x", 100.0, 1, 100.0) == 0.0 and residual_own_qty(5, 100.0, float("nan"), 100.0) == 5.0
    assert residual_own_qty(5, 100.0, 1, 0) == 0.0


def test_engine_wiring():
    import services.trading_engine as T
    src = inspect.getsource(T.TradingEngine._close_position_locked)
    i = src.index("residual_own_qty(")
    assert "[RESIDUAL_NOT_OURS]" in src and i < src.index("[RESTING_TP_RESIDUAL] {order.pair} #{order.id}: {_rres} of ours still open")
    assert "amount=_rres), f\"{order.pair} residual close\")" in src and "_rres = _own_snap" in src
    assert "'filled': float(result.get('filled') or 0)" in src
