"""
Tests for the Apr 17 broker-side protective stops (system-down insurance).

Covers:
  1. place_protective_stops computes correct SL/TP prices for LONG and SHORT
  2. Graceful failure: one leg failing doesn't break the other
  3. cancel_protective_stops treats "order does not exist" as success
     (Binance auto-cancel of closePosition=true orders is expected)
  4. Reconciler labels BROKER_SL / BROKER_TP when the fill matches the
     expected trigger within the tolerance window

Run with:
    python3 tests/test_protective_stops.py
"""

import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from database import Base
from models import Order
import main
from services.binance_service import BinanceService


async def _setup_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ────────────────────────────────────────────────────────────────────────────
# PART 1 — place_protective_stops price math + Binance call shape
# ────────────────────────────────────────────────────────────────────────────

async def test_place_protective_stops_long():
    """LONG: SL must be below entry, TP must be above. Binance called with
    correct side (sell), reduceOnly=True, NO closePosition (Binance -4120
    if closePosition=true is sent via standard endpoint), explicit quantity,
    MARK_PRICE."""
    svc = BinanceService()
    svc._check_ban = AsyncMock()
    svc.load_markets = AsyncMock()
    svc.get_tick_size = AsyncMock(return_value=0.01)
    svc.exchange = MagicMock()

    # Capture the create_order calls
    _calls = []
    async def _fake_create_order(**kwargs):
        _calls.append(kwargs)
        return {"id": f"order-{len(_calls)}"}
    svc.exchange.create_order = _fake_create_order

    result = await svc.place_protective_stops(
        symbol="BTC/USDT:USDT",
        direction="LONG",
        quantity=0.5,
        entry_price=100.0,
        sl_pct=1.5,
        tp_pct=5.0,
    )

    assert len(_calls) == 2, f"expected 2 orders, got {len(_calls)}"
    assert result["sl_order_id"] == "order-1"
    assert result["tp_order_id"] == "order-2"

    sl_call, tp_call = _calls
    assert sl_call["type"] == "STOP_MARKET"
    assert sl_call["side"] == "sell", "LONG close side = sell"
    # reduceOnly=True ensures close-only (cannot flip). closePosition must NOT
    # be present — Binance rejects closePosition=true with -1106.
    # portfolioMargin=True routes to /papi/v1/um/conditional/order (the
    # "Algo Order API" Binance requires for conditional orders on PM
    # accounts — without it, CCXT falls back to /fapi/v1/order which
    # rejects conditional types with -4120).
    assert sl_call["params"]["reduceOnly"] is True, "reduceOnly must be True"
    assert sl_call["params"]["portfolioMargin"] is True, (
        "portfolioMargin must be True — routes to /papi/v1/um/conditional/order "
        "(the Algo Order endpoint). Without it, Binance returns -4120."
    )
    assert "closePosition" not in sl_call["params"], (
        "closePosition must NOT be set — Binance rejects it with -1106"
    )
    assert sl_call["amount"] == 0.5, f"SL must use explicit quantity, got {sl_call['amount']}"
    assert sl_call["params"]["workingType"] == "MARK_PRICE"
    # SL price ≈ 100 * (1 - 0.015) = 98.5, rounded DOWN to 0.01 tick = 98.50
    assert abs(sl_call["params"]["stopPrice"] - 98.50) < 0.001, f"SL stopPrice={sl_call['params']['stopPrice']}"

    assert tp_call["type"] == "TAKE_PROFIT_MARKET"
    assert tp_call["side"] == "sell"
    assert tp_call["params"]["reduceOnly"] is True
    assert tp_call["params"]["portfolioMargin"] is True, "portfolioMargin must be True for TP too"
    assert "closePosition" not in tp_call["params"]
    assert tp_call["amount"] == 0.5
    # TP price ≈ 100 * 1.05 = 105.00
    assert abs(tp_call["params"]["stopPrice"] - 105.00) < 0.001, f"TP stopPrice={tp_call['params']['stopPrice']}"
    print("  [OK] LONG: SL @ 98.50, TP @ 105.00, reduceOnly+portfolioMargin+qty+MARK_PRICE, NO closePosition")


async def test_place_protective_stops_short():
    """SHORT: SL must be above entry, TP must be below. Close side = buy."""
    svc = BinanceService()
    svc._check_ban = AsyncMock()
    svc.load_markets = AsyncMock()
    svc.get_tick_size = AsyncMock(return_value=0.01)
    svc.exchange = MagicMock()
    _calls = []
    async def _fake_create_order(**kwargs):
        _calls.append(kwargs)
        return {"id": f"order-{len(_calls)}"}
    svc.exchange.create_order = _fake_create_order

    result = await svc.place_protective_stops(
        symbol="BTC/USDT:USDT", direction="SHORT",
        quantity=0.5, entry_price=100.0, sl_pct=1.5, tp_pct=5.0,
    )
    sl_call, tp_call = _calls
    assert sl_call["side"] == "buy", "SHORT close side = buy"
    assert sl_call["params"]["reduceOnly"] is True
    assert sl_call["params"]["portfolioMargin"] is True
    assert "closePosition" not in sl_call["params"]
    assert sl_call["amount"] == 0.5
    # SL above entry: 100 * 1.015 = 101.50
    assert abs(sl_call["params"]["stopPrice"] - 101.50) < 0.001
    # TP below entry: 100 * 0.95 = 95.00
    assert abs(tp_call["params"]["stopPrice"] - 95.00) < 0.001
    assert result["sl_order_id"] and result["tp_order_id"]
    print("  [OK] SHORT: SL @ 101.50, TP @ 95.00, close side = buy, reduceOnly+portfolioMargin+qty")


async def test_place_protective_stops_graceful_partial_failure():
    """If SL succeeds but TP fails (Binance error), we get sl_order_id
    and tp_order_id=None. Does NOT raise — fails open."""
    svc = BinanceService()
    svc._check_ban = AsyncMock()
    svc.load_markets = AsyncMock()
    svc.get_tick_size = AsyncMock(return_value=0.01)
    svc.exchange = MagicMock()
    call_count = {"n": 0}
    async def _fake_create_order(**kwargs):
        call_count["n"] += 1
        if kwargs["type"] == "TAKE_PROFIT_MARKET":
            raise Exception("Binance API: invalid stopPrice for TP")
        return {"id": "sl-123"}
    svc.exchange.create_order = _fake_create_order

    result = await svc.place_protective_stops(
        symbol="BTC/USDT:USDT", direction="LONG",
        quantity=0.5, entry_price=100.0, sl_pct=1.5, tp_pct=5.0,
    )
    assert result["sl_order_id"] == "sl-123"
    assert result["tp_order_id"] is None, "TP failure should surface as None"
    print("  [OK] partial failure: SL placed, TP failed gracefully, no exception raised")


async def test_place_protective_stops_invalid_direction():
    """Invalid direction returns empty dict, does not raise."""
    svc = BinanceService()
    svc.exchange = MagicMock()
    svc.exchange.create_order = AsyncMock()

    result = await svc.place_protective_stops(
        symbol="BTC/USDT:USDT", direction="BANANA",
        quantity=0.5, entry_price=100.0, sl_pct=1.5, tp_pct=5.0,
    )
    assert result["sl_order_id"] is None
    assert result["tp_order_id"] is None
    # Should not call create_order at all
    svc.exchange.create_order.assert_not_called()
    print("  [OK] invalid direction: returns empty result, does not call Binance")


# ────────────────────────────────────────────────────────────────────────────
# PART 2 — cancel_protective_stops
# ────────────────────────────────────────────────────────────────────────────

async def test_cancel_protective_stops_happy():
    """Cancel succeeds → both flagged as cancelled.

    CRITICAL: cancel_order must be called with portfolioMargin=True + trigger=True
    so CCXT routes the cancel to /papi/v1/um/conditional/order DELETE (the
    Portfolio Margin Conditional Order endpoint where the orders were placed).
    Without these flags CCXT would hit /fapi/v1/order DELETE and return
    "unknown order", which would be misinterpreted as "already gone" by our
    benign-error handler, leaving real orders live.
    """
    svc = BinanceService()
    svc._check_ban = AsyncMock()
    svc.load_markets = AsyncMock()
    svc.exchange = MagicMock()

    # Capture cancel_order call args so we can verify PM flags are passed
    _cancel_calls = []
    async def _fake_cancel(oid, symbol, params=None):
        _cancel_calls.append({"oid": oid, "symbol": symbol, "params": params})
        return {"status": "canceled"}
    svc.exchange.cancel_order = _fake_cancel

    out = await svc.cancel_protective_stops(
        symbol="BTC/USDT:USDT", sl_order_id="sl-1", tp_order_id="tp-1",
    )
    assert out["sl_cancelled"] is True
    assert out["tp_cancelled"] is True
    assert len(_cancel_calls) == 2, f"expected 2 cancel calls, got {len(_cancel_calls)}"
    # Both cancels must pass portfolioMargin=True + trigger=True
    for call in _cancel_calls:
        assert call["params"] is not None, "cancel_order must receive params"
        assert call["params"].get("portfolioMargin") is True, (
            "cancel_order must pass portfolioMargin=True — orders were placed "
            "on /papi endpoint, must be cancelled there"
        )
        assert call["params"].get("trigger") is True, (
            "cancel_order must pass trigger=True — these are conditional orders"
        )
    print("  [OK] happy path: both cancelled with portfolioMargin+trigger flags")


async def test_cancel_protective_stops_already_gone_is_success():
    """Binance returns 'unknown order id' when the protective SL/TP fired
    (i.e. position already closed via broker-side stop, then bot also tries
    to cancel as part of its close path).  This must be treated as SUCCESS —
    the order IS gone, which is what we wanted."""
    svc = BinanceService()
    svc._check_ban = AsyncMock()
    svc.load_markets = AsyncMock()
    svc.exchange = MagicMock()

    async def _fake_cancel(oid, symbol, params=None):
        raise Exception("-2011 Unknown order sent.")
    svc.exchange.cancel_order = _fake_cancel

    out = await svc.cancel_protective_stops(
        symbol="BTC/USDT:USDT", sl_order_id="gone-1", tp_order_id="gone-2",
    )
    # Both should be marked as cancelled (they're gone, which is what we want)
    assert out["sl_cancelled"] is True, "Gone-order should count as cancelled"
    assert out["tp_cancelled"] is True
    print("  [OK] 'order does not exist' from Binance treated as success")


# ────────────────────────────────────────────────────────────────────────────
# PART 3 — Reconciler BROKER_SL / BROKER_TP labelling
# ────────────────────────────────────────────────────────────────────────────

async def _insert_open_order(s: AsyncSession, direction: str, entry_price: float) -> int:
    o = Order(
        pair="BTCUSDT", direction=direction, status="OPEN",
        entry_price=entry_price, current_price=entry_price,
        investment=100.0, leverage=1.0, notional_value=100.0,
        quantity=100.0 / entry_price, confidence="STRONG_BUY",
        entry_fee=0.04, is_paper=False,
        closing_in_progress=False, close_initiated_at=None,
    )
    s.add(o)
    await s.commit()
    await s.refresh(o)
    return o.id


async def _run_reconcile_with_fill(session: AsyncSession, fill_price: float):
    """Simulate Binance has no open positions and the fill price was `fill_price`."""
    with patch.object(
        main.binance_service, "get_open_positions",
        new=AsyncMock(return_value=[]),
    ), patch(
        "main._get_actual_fill_price",
        new=AsyncMock(return_value=fill_price),
    ):
        return await main._reconcile_open_orders(session)


async def test_reconciler_labels_broker_sl_long():
    """LONG position closed at exactly -1.5% of entry → label BROKER_SL."""
    engine, Sess = await _setup_engine()
    async with Sess() as s:
        oid = await _insert_open_order(s, "LONG", entry_price=100.0)

    # Fill price = 98.5 (exactly -1.5% from entry 100.0)
    async with Sess() as s:
        closed = await _run_reconcile_with_fill(s, fill_price=98.5)
        assert len(closed) == 1

    async with Sess() as s:
        row = (await s.execute(select(Order).where(Order.id == oid))).scalar_one()
        assert row.close_reason == "BROKER_SL", f"expected BROKER_SL, got {row.close_reason}"
    await engine.dispose()
    print("  [OK] LONG filled @ -1.5% → labelled BROKER_SL (system-down insurance fired)")


async def test_reconciler_labels_broker_tp_short():
    """SHORT filled at -5% (i.e. entry * 0.95) → label BROKER_TP."""
    engine, Sess = await _setup_engine()
    async with Sess() as s:
        oid = await _insert_open_order(s, "SHORT", entry_price=100.0)

    async with Sess() as s:
        closed = await _run_reconcile_with_fill(s, fill_price=95.0)
        assert len(closed) == 1

    async with Sess() as s:
        row = (await s.execute(select(Order).where(Order.id == oid))).scalar_one()
        assert row.close_reason == "BROKER_TP", f"expected BROKER_TP, got {row.close_reason}"
    await engine.dispose()
    print("  [OK] SHORT filled @ -5% entry → labelled BROKER_TP")


async def test_reconciler_labels_external_close_when_price_doesnt_match():
    """Fill price far from both SL and TP triggers → normal EXTERNAL_CLOSE.
    E.g., user manually closed at mid-trade at -0.5%."""
    engine, Sess = await _setup_engine()
    async with Sess() as s:
        oid = await _insert_open_order(s, "LONG", entry_price=100.0)

    # Fill at 99.5 = -0.5%. Far from SL (-1.5%) or TP (+5%). Tolerance 0.15%.
    async with Sess() as s:
        closed = await _run_reconcile_with_fill(s, fill_price=99.5)
        assert len(closed) == 1

    async with Sess() as s:
        row = (await s.execute(select(Order).where(Order.id == oid))).scalar_one()
        assert row.close_reason == "EXTERNAL_CLOSE", f"expected EXTERNAL_CLOSE, got {row.close_reason}"
    await engine.dispose()
    print("  [OK] mid-trade user close → labelled EXTERNAL_CLOSE (broker stop did NOT fire)")


async def test_reconciler_tolerance_edge():
    """Within tolerance (0.15% of entry) → broker label applies. Just outside → external."""
    engine, Sess = await _setup_engine()
    async with Sess() as s:
        oid = await _insert_open_order(s, "LONG", entry_price=100.0)

    # Expected SL = 98.5. Tolerance = 0.15% * 100 = 0.15. So 98.5 ± 0.15 = [98.35, 98.65]
    # Test fill at 98.4 (within tolerance → BROKER_SL)
    async with Sess() as s:
        await _run_reconcile_with_fill(s, fill_price=98.4)

    async with Sess() as s:
        row = (await s.execute(select(Order).where(Order.id == oid))).scalar_one()
        assert row.close_reason == "BROKER_SL", f"98.4 should be within tolerance of SL 98.5, got {row.close_reason}"
    await engine.dispose()
    print("  [OK] 0.1% off from SL trigger still labelled BROKER_SL (within MARK_PRICE tolerance)")


# ────────────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────────────

async def main_test():
    print("Running protective-stops tests...\n")
    print("Part 1 — place_protective_stops:")
    await test_place_protective_stops_long()
    await test_place_protective_stops_short()
    await test_place_protective_stops_graceful_partial_failure()
    await test_place_protective_stops_invalid_direction()

    print("\nPart 2 — cancel_protective_stops:")
    await test_cancel_protective_stops_happy()
    await test_cancel_protective_stops_already_gone_is_success()

    print("\nPart 3 — Reconciler BROKER_SL/BROKER_TP labels:")
    await test_reconciler_labels_broker_sl_long()
    await test_reconciler_labels_broker_tp_short()
    await test_reconciler_labels_external_close_when_price_doesnt_match()
    await test_reconciler_tolerance_edge()

    print("\nAll 10 protective-stops tests passed.")


if __name__ == "__main__":
    asyncio.run(main_test())
