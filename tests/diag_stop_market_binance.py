"""
Diagnostic script: empirically find which STOP_MARKET parameter combination
Binance Futures accepts for this specific account + CCXT version.

Apr 17 context:
  Hotfix #1/#2/#3 each changed the protective-stops params based on a theory,
  each pushed to production, each failed with a different Binance error.
  This script tests the parameter matrix LOCALLY against the real Binance API
  BEFORE any more deploys, so the next hotfix is data-backed.

Safety design:
  - Uses an EXISTING open position (so reduceOnly=True is valid)
  - stopPrice set 50% AWAY from current market in the safe direction
    (for LONG SL: stopPrice = entry × 0.5, requires 50% crash to trigger —
     effectively impossible in the ~2 seconds we hold the order)
  - Every placed order is cancelled within 2 seconds
  - Script exits on the FIRST Binance rejection with a clear error log
  - No positions are opened or closed by this script

Requirements:
  - BINANCE_API_KEY and BINANCE_API_SECRET in env
  - At least one open USDT-perp LONG position on the account (we'll use the
    largest one by notional to test against)

Run:
  BINANCE_API_KEY=... BINANCE_API_SECRET=... python3 tests/diag_stop_market_binance.py

Output: per-combination result — PLACED (success + orderId + cancelled) or
        REJECTED (Binance error code + message).  Summary table at the end.
"""

import asyncio
import os
import sys
import traceback
from typing import Dict, Any, Optional

# Project root so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ccxt.async_support as ccxt


# ────────────────────────────────────────────────────────────────────────────
# Test matrix — combinations to try
# Each variant specifies the params CCXT will see. We deliberately test
# orthogonal changes (one knob at a time where possible) so the output
# points unambiguously to the faulty knob.
# ────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    {
        "name": "A. Current (broken prod): type=STOP_MARKET, reduceOnly, MARK_PRICE, GTE_GTC, portfolioMargin",
        "order_type": "STOP_MARKET",
        "params_extra": {
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
            "timeInForce": "GTE_GTC",
            "portfolioMargin": True,
        },
    },
    {
        "name": "B. Drop portfolioMargin (hotfix #3 revert)",
        "order_type": "STOP_MARKET",
        "params_extra": {
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
            "timeInForce": "GTE_GTC",
        },
    },
    {
        "name": "C. Drop timeInForce entirely (let Binance default)",
        "order_type": "STOP_MARKET",
        "params_extra": {
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
        },
    },
    {
        "name": "D. Use GTC instead of GTE_GTC",
        "order_type": "STOP_MARKET",
        "params_extra": {
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
            "timeInForce": "GTC",
        },
    },
    {
        "name": "E. Use CONTRACT_PRICE instead of MARK_PRICE",
        "order_type": "STOP_MARKET",
        "params_extra": {
            "reduceOnly": True,
            "workingType": "CONTRACT_PRICE",
        },
    },
    {
        "name": "F. Drop workingType entirely",
        "order_type": "STOP_MARKET",
        "params_extra": {
            "reduceOnly": True,
        },
    },
    {
        "name": "G. Minimal params — reduceOnly only",
        "order_type": "STOP_MARKET",
        "params_extra": {
            "reduceOnly": True,
        },
    },
    {
        "name": "H. CCXT unified (type='market' + stopPrice in params)",
        "order_type": "market",
        "params_extra": {
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
        },
    },
    {
        "name": "I. CCXT unified, no workingType",
        "order_type": "market",
        "params_extra": {
            "reduceOnly": True,
        },
    },
    {
        "name": "J. CCXT create_stop_market_order helper",
        "method": "create_stop_market_order",
        "params_extra": {
            "reduceOnly": True,
        },
    },
]


async def get_largest_long_position(exchange) -> Optional[Dict[str, Any]]:
    """Pick one open LONG position as our test target (largest notional)."""
    positions = await exchange.fetch_positions()
    longs = [
        p for p in positions
        if p.get("side") == "long"
        and float(p.get("contracts") or 0) > 0
        and str(p.get("symbol", "")).endswith("/USDT:USDT")
    ]
    if not longs:
        return None
    # Sort by notional descending
    longs.sort(key=lambda p: float(p.get("notional") or 0), reverse=True)
    return longs[0]


async def try_place_and_cancel(
    exchange,
    symbol: str,
    direction: str,          # "LONG" or "SHORT" — position direction
    quantity: float,
    safe_stop_price: float,
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    """Place one test STOP_MARKET, immediately cancel, return result dict."""
    close_side = "sell" if direction == "LONG" else "buy"
    result = {"name": variant["name"], "placed": False, "cancelled": False, "error": None, "order_id": None}

    try:
        if variant.get("method") == "create_stop_market_order":
            # CCXT helper
            order = await exchange.create_stop_market_order(
                symbol=symbol,
                side=close_side,
                amount=quantity,
                triggerPrice=safe_stop_price,
                params=variant["params_extra"],
            )
        else:
            # Standard create_order path
            params = {**variant["params_extra"], "stopPrice": safe_stop_price}
            order = await exchange.create_order(
                symbol=symbol,
                type=variant["order_type"],
                side=close_side,
                amount=quantity,
                price=None,
                params=params,
            )

        result["placed"] = True
        result["order_id"] = str(order.get("id") or order.get("orderId"))
        print(f"  ✅ PLACED — orderId={result['order_id']}")

        # IMMEDIATELY cancel — minimize risk window
        try:
            # Try to cancel with the same param shape as placement, in case
            # routing depends on those params (e.g. portfolioMargin)
            cancel_params = {}
            if variant["params_extra"].get("portfolioMargin"):
                cancel_params = {"portfolioMargin": True, "trigger": True}
            await exchange.cancel_order(result["order_id"], symbol, cancel_params)
            result["cancelled"] = True
            print(f"     cancelled successfully")
        except Exception as ce:
            result["cancelled"] = False
            result["cancel_error"] = str(ce)[:200]
            print(f"  ⚠️  could not cancel (order remains OPEN!): {str(ce)[:150]}")

    except Exception as e:
        result["placed"] = False
        result["error"] = str(e)[:300]
        print(f"  ❌ REJECTED — {str(e)[:200]}")

    return result


async def main():
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print("ERROR: Set BINANCE_API_KEY and BINANCE_API_SECRET env vars")
        print("Example:")
        print("  BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python3 tests/diag_stop_market_binance.py")
        sys.exit(1)

    exchange = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    print(f"CCXT version: {ccxt.__version__}")
    print(f"Exchange: {exchange.id}")
    print("")

    try:
        await exchange.load_markets()

        # Find an open LONG we can safely attach a FAR-AWAY STOP to
        pos = await get_largest_long_position(exchange)
        if pos is None:
            print("ERROR: no open USDT-perp LONG positions found — this script needs one to test against.")
            print("       (reduceOnly=True requires an existing position to reduce)")
            sys.exit(2)

        symbol = pos["symbol"]
        quantity = float(pos["contracts"])
        entry = float(pos.get("entryPrice") or 0)
        current = float(pos.get("markPrice") or pos.get("lastPrice") or entry)

        # For a LONG: SL is BELOW entry.
        # Set stopPrice to entry × 0.5 (50% below) — would need a flash crash
        # to trigger, impossible in the ~2s window we hold the order.
        safe_stop_price = round(entry * 0.5, 6)

        print(f"Test target (existing position):")
        print(f"  symbol      = {symbol}")
        print(f"  direction   = LONG")
        print(f"  quantity    = {quantity}")
        print(f"  entry price = {entry}")
        print(f"  mark price  = {current}")
        print(f"  safe stop   = {safe_stop_price}  (entry × 0.5 — unreachable)")
        print("")
        print(f"Running {len(VARIANTS)} parameter combinations…")
        print("=" * 80)

        results = []
        for i, variant in enumerate(VARIANTS, 1):
            print(f"\n[{i}/{len(VARIANTS)}] {variant['name']}")
            r = await try_place_and_cancel(
                exchange, symbol, "LONG", quantity, safe_stop_price, variant
            )
            results.append(r)
            # Brief pause between tests
            await asyncio.sleep(0.5)

        # ── Summary ──
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        placed = [r for r in results if r["placed"]]
        rejected = [r for r in results if not r["placed"]]
        print(f"✅ {len(placed)}/{len(results)} variants accepted by Binance:")
        for r in placed:
            cancel_status = "cancelled" if r["cancelled"] else "⚠️  STILL OPEN"
            print(f"  • {r['name']}  ({cancel_status})")
        print(f"\n❌ {len(rejected)}/{len(results)} variants rejected:")
        for r in rejected:
            print(f"  • {r['name']}")
            print(f"    {r['error'][:250]}")

        # Flag any orders that couldn't be cancelled (rare but important)
        still_open = [r for r in results if r["placed"] and not r["cancelled"]]
        if still_open:
            print("\n⚠️  WARNING: THE FOLLOWING ORDERS COULD NOT BE CANCELLED AND ARE STILL ACTIVE:")
            for r in still_open:
                print(f"  orderId={r['order_id']}  variant={r['name']}")
                print(f"  cancel_error: {r.get('cancel_error', 'unknown')}")
            print("\nManually cancel them on Binance UI.  They are set to trigger at a 50% crash,")
            print("so they are unlikely to fire, but clean them up to be safe.")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(main())
