"""🔎 DECISION_LOG 321: positions closed in the Binance app (BR / A, 2026-10-20) were booked at the fallback price ("no trade history
match") ~1 min after the close. Pins ① the close-fill VWAP over the order's lifetime (resting-TP fills left out), ② the bounded
trade-history window, ③ the 15 s reconcile cadence, ④ the wiring in main.py + the engine."""
import asyncio
import inspect
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.golive_guards import (close_fill_vwap, fill_trades_since_ms, utc_ms, FILL_LOOKBACK_PAD_MS, FILL_MAX_WINDOW_MS,  # noqa: E402
                                    RECONCILE_INTERVAL_TICKS, RECONCILE_FALLBACK_AFTER_SKIPS, RECONCILE_FALLBACK_EVERY,
                                    UNTRACKED_CONFIRM_PASSES)

T0 = 1_792_000_000_000


def _t(side, px, amt, ts, order="9"):
    return {"side": side, "price": px, "amount": amt, "timestamp": ts, "order": order}


def test_vwap_of_all_close_trades_in_the_lifetime():
    tr = [_t("sell", 0.70, 100, T0 - 1), _t("buy", 0.67, 300, T0 + 1), _t("sell", 0.68, 100, T0 + 5), _t("sell", 0.69, 300, T0 + 6)]
    px, q, n = close_fill_vwap(tr, "LONG", T0)
    assert n == 2 and q == 400 and abs(px - (0.68 * 100 + 0.69 * 300) / 400) < 1e-12     # the pre-lifetime sell and the entry buy never count
    assert close_fill_vwap(tr, "SHORT", T0) == (0.67, 300, 1)


def test_resting_tp_fills_are_left_out():
    tr = [_t("sell", 0.7086, 50, T0 + 1, order="2746703035"), _t("sell", 0.685, 250, T0 + 2, order="555")]
    assert close_fill_vwap(tr, "LONG", T0, ["2746703035", None])[0:2] == (0.685, 250)
    assert close_fill_vwap(tr, "LONG", T0, [2746703035]) [1] == 250                       # an int id matches the string orderId
    assert close_fill_vwap([tr[0]], "LONG", T0, ["2746703035"]) is None                  # only the TP filled → nothing to read


def test_bad_rows_are_skipped_and_empty_is_none():
    tr = [{"side": "sell"}, _t("sell", "x", 1, T0 + 1), _t("sell", 0, 5, T0 + 1), _t("sell", float("nan"), 5, T0 + 1),
          _t("sell", 1.0, None, T0 + 1), None, _t("sell", 2.0, 3, T0 + 1)]
    assert close_fill_vwap(tr, "LONG", T0) == (2.0, 3.0, 1)
    assert close_fill_vwap([], "LONG", T0) is None and close_fill_vwap(None, "LONG", T0) is None


def test_since_window():
    now = T0 + 3_600_000
    assert fill_trades_since_ms(T0, now) == T0 - FILL_LOOKBACK_PAD_MS
    assert fill_trades_since_ms(T0 - 30 * 86_400_000, now) == now - FILL_MAX_WINDOW_MS     # never beyond one 7-day Binance window
    assert fill_trades_since_ms(None, now) == now - FILL_MAX_WINDOW_MS
    assert FILL_MAX_WINDOW_MS < 7 * 86_400_000


def test_cadence_constants():
    assert RECONCILE_INTERVAL_TICKS == 15
    assert RECONCILE_FALLBACK_AFTER_SKIPS * RECONCILE_INTERVAL_TICKS >= 300                # the per-symbol fallback stays ≈5 min of failures
    assert UNTRACKED_CONFIRM_PASSES == 2


def test_main_wiring():
    import main
    src = inspect.getsource(main.monitor_loop)
    assert "_RECONCILE_INTERVAL = RECONCILE_INTERVAL_TICKS" in src
    assert "_reconcile_skip_counter >= RECONCILE_FALLBACK_AFTER_SKIPS" in inspect.getsource(main._reconcile_open_orders)


def _recent():
    """an open time inside the 7-day Binance window: fill_trades_since_ms clamps to now − 7 d, so a fixed date ages out (DL 387)."""
    return (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)).replace(microsecond=0)


def test_main_fill_price_reads_the_lifetime_and_skips_the_tp():
    import main
    order = SimpleNamespace(pair="BRUSDT", direction="LONG", quantity=300.0, opened_at=_recent(),
                            protective_tp_order_id="777", resting_tp_filled_qty=100.0, current_price=0.68483, entry_price=0.671248)
    op_ms = int((order.opened_at - datetime(1970, 1, 1)).total_seconds() * 1000)
    trades = [_t("buy", 0.671248, 300, op_ms + 100, "1"), _t("sell", 0.7086, 100, op_ms + 9000, "777"),
              _t("sell", 0.690, 200, op_ms + 10_000, "2")]
    m = AsyncMock(return_value=trades)
    with patch.object(main.binance_service, "fetch_my_trades", m):
        px = asyncio.run(main._get_actual_fill_price(order))
    assert px == 0.690
    kw = m.call_args.kwargs
    assert kw["since"] == op_ms - FILL_LOOKBACK_PAD_MS and kw["limit"] >= 100
    with patch.object(main.binance_service, "fetch_my_trades", AsyncMock(return_value=None)):
        assert asyncio.run(main._get_actual_fill_price(order)) == 0.68483                     # unreadable → the old fallback


def test_engine_fill_price_excludes_settled_tp_ids():
    from services.trading_engine import trading_engine, binance_service
    order = SimpleNamespace(pair="AUSDT", direction="LONG", quantity=10.0, opened_at=_recent(),
                            protective_tp_order_id=None)
    op_ms = int((order.opened_at - datetime(1970, 1, 1)).total_seconds() * 1000)
    trades = [_t("sell", 0.098, 4, op_ms + 5000, "997521405"), _t("sell", 0.0969, 6, op_ms + 6000, "3")]
    with patch.object(binance_service, "fetch_my_trades", AsyncMock(return_value=trades)):
        assert asyncio.run(trading_engine._fetch_actual_fill_price(order, 0.5, ["997521405"])) == 0.0969
        assert asyncio.run(trading_engine._fetch_actual_fill_price(order, 0.5, ["997521405", "3"])) == 0.5
    src = inspect.getsource(type(trading_engine)._close_position_locked)
    assert src.count("_fetch_actual_fill_price(order, current_price, [*((_rtp_set or {}).get('ids') or ()), *((_rtp_set or {}).get('pending') or ())])") == 3
    assert "'ids': list(ids)" in inspect.getsource(type(trading_engine)._rtp_settle)


def test_previous_same_pair_close_inside_the_pad_never_blends_in():
    """review: a re-entry < 60 s after the previous same-pair close (LITE_RETRY / manual) — the old close's sell is before THIS entry buy."""
    op = T0
    tr = [_t("buy", 1.00, 10, op - 50_000, "a"), _t("sell", 0.90, 10, op - 20_000, "b"),     # previous lifetime, inside the pad
          _t("buy", 0.91, 7, op - 300, "c"), _t("sell", 0.95, 7, op + 90_000, "d")]           # this row: entry fill, app close
    since = op - FILL_LOOKBACK_PAD_MS
    assert close_fill_vwap(tr, "LONG", since)[0:2] != (0.95, 7)                                # without the anchor it blends
    assert close_fill_vwap(tr, "LONG", since, (), op) == (0.95, 7.0, 1)
    assert close_fill_vwap(tr[3:], "LONG", since, (), op) == (0.95, 7.0, 1)                     # no entry trade in reach → since


def test_utc_ms_naive_and_aware():
    from datetime import timezone, timedelta
    naive = datetime(2026, 10, 20, 12, 0, 0)
    assert utc_ms(naive) == utc_ms(naive.replace(tzinfo=timezone.utc)) == utc_ms(datetime(2026, 10, 20, 14, 0, 0, tzinfo=timezone(timedelta(hours=2))))
    assert utc_ms(None) is None and utc_ms("x") is None


def test_per_symbol_fallback_keeps_the_old_pace():
    import main
    calls = []

    async def _none():
        return None

    async def _per(db):
        calls.append(main._reconcile_skip_counter)
        return []
    main._reconcile_skip_counter = 0
    with patch.object(main.binance_service, "get_open_positions", _none), patch.object(main, "_reconcile_per_symbol", _per):
        for _ in range(RECONCILE_FALLBACK_AFTER_SKIPS + 3 * RECONCILE_FALLBACK_EVERY):
            asyncio.run(main._reconcile_open_orders(None))
    main._reconcile_skip_counter = 0
    assert calls == [RECONCILE_FALLBACK_AFTER_SKIPS + k * RECONCILE_FALLBACK_EVERY for k in range(4)]   # 20, 24, 28, 32 — never every pass


def test_newest_trades_up_to_the_open_qty():
    """deep review C1: a TP partial whose id was already cleared (re-price / GONE) is older than the app close — the qty cap keeps it out."""
    tr = [_t("buy", 1.0, 10, T0, "e"), _t("sell", 1.05, 4, T0 + 5_000, "old-tp"), _t("sell", 0.98, 6, T0 + 90_000, "app")]
    assert close_fill_vwap(tr, "LONG", T0, (), T0, max_qty=10 - 4) == (0.98, 6.0, 1)            # recorded TP fill 4 → remainder 6
    px, q, n = close_fill_vwap(tr, "LONG", T0, (), T0, max_qty=8)
    assert q == 8 and n == 2 and abs(px - (0.98 * 6 + 1.05 * 2) / 8) < 1e-12                    # partial last trade
    assert close_fill_vwap(tr, "LONG", T0, (), T0, max_qty=0)[1] == 10                           # 0 / None = uncapped
    assert close_fill_vwap(tr, "LONG", T0, (), T0, max_qty=float("inf"))[1] == 10


def test_main_caps_at_the_open_qty_and_retries_once_when_short(monkeypatch):
    import main
    order = SimpleNamespace(id=7, pair="XUSDT", direction="LONG", quantity=10.0, opened_at=datetime(2026, 10, 20, 12, 0, 0),
                            protective_tp_order_id=None, resting_tp_filled_qty=4.0, current_price=1.0, entry_price=1.0)
    op_ms = utc_ms(order.opened_at)
    monkeypatch.setattr(main.time, "time", lambda: (op_ms + 3_600_000) / 1000)
    slept = []

    async def _sleep(s):
        slept.append(s)
    monkeypatch.setattr(main.asyncio, "sleep", _sleep)
    full = [_t("buy", 1.0, 10, op_ms, "1"), _t("sell", 1.2, 4, op_ms + 5000, "x"), _t("sell", 1.1, 6, op_ms + 9000, "2")]
    m = AsyncMock(side_effect=[full[:2], full])                                                  # first read: the close not yet indexed
    with patch.object(main.binance_service, "fetch_my_trades", m):
        assert asyncio.run(main._get_actual_fill_price(order)) == 1.1
    assert slept == [2.0] and m.call_count == 2 and m.call_args.kwargs["since"] == op_ms - FILL_LOOKBACK_PAD_MS
    m2 = AsyncMock(return_value=full)
    with patch.object(main.binance_service, "fetch_my_trades", m2):
        assert asyncio.run(main._get_actual_fill_price(order, rest_qty=6.0, retry=False)) == 1.1
    assert m2.call_count == 1


def test_reconcile_rereads_without_unrecorded_tp_ids():
    import main
    src = inspect.getsource(main._close_orphan_orders)
    assert "_rtp_pre_id = getattr(order, 'protective_tp_order_id', None)" in src
    assert src.index("_rtp_pre_id = getattr") < src.index("_rtp_x = await trading_engine._rtp_settle(order)")
    assert src.index("_get_actual_fill_price(order, [_rtp_pre_id, *_rtp_extra], rest_qty=_rtp_rest, retry=False)") < src.index("order.exit_price = exit_price")
