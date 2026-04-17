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
    # CCXT unified syntax: type='market' + stopLossPrice/takeProfitPrice in params
    # (NOT type='STOP_MARKET').  Binance rejects type='STOP_MARKET' with -4120
    # when combined with GTE_GTC TIF and reduceOnly.  The CCXT helper
    # create_stop_market_order uses exactly this pattern internally.
    assert sl_call["type"] == "market", f"SL must use CCXT unified type='market', got {sl_call['type']}"
    assert sl_call["side"] == "sell", "LONG close side = sell"
    assert sl_call["params"]["reduceOnly"] is True, "reduceOnly must be True"
    # portfolioMargin must NOT be set — account is standard Futures, /papi returns -2015
    assert "portfolioMargin" not in sl_call["params"], (
        "portfolioMargin must NOT be set — user's account is standard USDⓈ-M "
        "Futures, not Portfolio Margin.  /papi endpoints return -2015."
    )
    # closePosition must NOT be set — causes -1106 with reduceOnly
    assert "closePosition" not in sl_call["params"]
    # GTE_GTC timeInForce must NOT be set — only valid with closePosition=true;
    # causes -4120 on reduceOnly STOP_MARKET orders
    assert "timeInForce" not in sl_call["params"], (
        "timeInForce must NOT be set — let CCXT/Binance default to GTC. "
        "GTE_GTC is for closePosition orders and causes -4120 on reduceOnly."
    )
    assert sl_call["amount"] == 0.5, f"SL must use explicit quantity, got {sl_call['amount']}"
    assert sl_call["params"]["workingType"] == "MARK_PRICE"
    # stopLossPrice (not stopPrice) is the CCXT unified param for SL.
    # CCXT converts internally to Binance STOP_MARKET with stopPrice.
    assert "stopLossPrice" in sl_call["params"], (
        "SL must use stopLossPrice (CCXT unified), not stopPrice"
    )
    # SL price ≈ 100 * (1 - 0.015) = 98.5, rounded DOWN to 0.01 tick = 98.50
    assert abs(sl_call["params"]["stopLossPrice"] - 98.50) < 0.001, f"SL stopLossPrice={sl_call['params']['stopLossPrice']}"

    assert tp_call["type"] == "market", f"TP must use CCXT unified type='market', got {tp_call['type']}"
    assert tp_call["side"] == "sell"
    assert tp_call["params"]["reduceOnly"] is True
    assert "portfolioMargin" not in tp_call["params"]
    assert "closePosition" not in tp_call["params"]
    assert "timeInForce" not in tp_call["params"]
    assert tp_call["amount"] == 0.5
    # takeProfitPrice (not stopPrice) is the CCXT unified param for TP.
    assert "takeProfitPrice" in tp_call["params"]
    # TP price ≈ 100 * 1.05 = 105.00
    assert abs(tp_call["params"]["takeProfitPrice"] - 105.00) < 0.001, f"TP takeProfitPrice={tp_call['params']['takeProfitPrice']}"
    print("  [OK] LONG: SL @ 98.50, TP @ 105.00, CCXT unified (type=market+stopLossPrice/takeProfitPrice)")


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
    assert sl_call["type"] == "market"
    assert sl_call["side"] == "buy", "SHORT close side = buy"
    assert sl_call["params"]["reduceOnly"] is True
    assert "portfolioMargin" not in sl_call["params"]
    assert "closePosition" not in sl_call["params"]
    assert "timeInForce" not in sl_call["params"]
    assert sl_call["amount"] == 0.5
    # SL above entry: 100 * 1.015 = 101.50
    assert abs(sl_call["params"]["stopLossPrice"] - 101.50) < 0.001
    # TP below entry: 100 * 0.95 = 95.00
    assert abs(tp_call["params"]["takeProfitPrice"] - 95.00) < 0.001
    assert result["sl_order_id"] and result["tp_order_id"]
    print("  [OK] SHORT: SL @ 101.50, TP @ 95.00, close side = buy, CCXT unified (no PM, no GTE_GTC)")


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
        # TP call carries takeProfitPrice (CCXT unified) — distinguish from SL
        # which carries stopLossPrice
        if 'takeProfitPrice' in kwargs.get('params', {}):
            raise Exception("Binance API: invalid takeProfitPrice for TP")
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

    Hotfix #4: orders now live on standard /fapi/v1/order (not /papi), so the
    cancel path uses CCXT's standard cancel_order with no special params.
    """
    svc = BinanceService()
    svc._check_ban = AsyncMock()
    svc.load_markets = AsyncMock()
    svc.exchange = MagicMock()

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
    print("  [OK] happy path: both cancelled successfully on /fapi endpoint")


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
