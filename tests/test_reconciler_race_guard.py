"""
Test for the Apr 16 reconciler race guard (SUIUSDT incident).

Reproduces the 55 ms race between the trailing-stop close path and the
monitor reconciler, and proves that with the closing_in_progress flag the
reconciler no longer overwrites bot-initiated closes with EXTERNAL_CLOSE.

Scenarios covered:
  1. Baseline (flag NOT set) — reconciler marks the orphan order EXTERNAL_CLOSE.
     This is the pre-fix behaviour, proven here so the test itself documents
     what the guard is preventing.
  2. Fresh intent (flag set NOW) — reconciler skips the row.
  3. Stale intent (flag set > CLOSE_INTENT_STALE_SECONDS ago) — reconciler
     proceeds with EXTERNAL_CLOSE (crashed-close recovery).

Run with:
    python3 tests/test_reconciler_race_guard.py

Uses an in-memory SQLite DB so no real database is touched.  Does NOT talk
to Binance — Binance is stubbed to return an empty open-positions set,
simulating "position already closed on Binance, DB row still OPEN".
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

# Make the project root importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force in-memory SQLite before anything imports config/database.
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from database import Base
from models import Order
import main


async def _setup_db():
    """Create a fresh in-memory SQLite DB with the full schema."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, SessionLocal


async def _insert_open_order(session: AsyncSession, **overrides) -> Order:
    order = Order(
        pair="SUIUSDT",
        direction="LONG",
        status="OPEN",
        entry_price=0.975,
        current_price=0.9789,
        investment=100.0,
        leverage=1.0,
        notional_value=100.0,
        quantity=102.56,
        confidence="STRONG_BUY",
        entry_fee=0.04,
        is_paper=False,
        closing_in_progress=False,
        close_initiated_at=None,
    )
    for k, v in overrides.items():
        setattr(order, k, v)
    session.add(order)
    await session.commit()
    await session.refresh(order)
    return order


async def _fetch_status(session: AsyncSession, order_id: int):
    res = await session.execute(select(Order).where(Order.id == order_id))
    return res.scalar_one()


async def _run_reconcile(session: AsyncSession):
    """Invoke the real reconciler with Binance stubbed to report no positions."""
    # Stub Binance: SUIUSDT is already gone from Binance (the incident scenario).
    empty_positions = []
    with patch.object(
        main.binance_service,
        "get_open_positions",
        new=AsyncMock(return_value=empty_positions),
    ), patch(
        "main._get_actual_fill_price",
        new=AsyncMock(return_value=0.9789),
    ):
        return await main._reconcile_open_orders(session)


async def test_baseline_no_guard_marks_external_close():
    """Without the flag, the reconciler overwrites the row as EXTERNAL_CLOSE.
    This is the bug — the test pins current (pre-fix) behaviour so we see the
    guard is what's doing the work in later cases."""
    engine, SessionLocal = await _setup_db()
    async with SessionLocal() as s:
        order = await _insert_open_order(s)
        oid = order.id

    async with SessionLocal() as s:
        closed = await _run_reconcile(s)
        assert len(closed) == 1, f"expected 1 reconciled, got {len(closed)}"

    async with SessionLocal() as s:
        after = await _fetch_status(s, oid)
        assert after.status == "CLOSED", f"status={after.status}"
        assert after.close_reason == "EXTERNAL_CLOSE", f"close_reason={after.close_reason}"
    await engine.dispose()
    print("  [OK] baseline (flag unset) — reconciler marks EXTERNAL_CLOSE as before")


async def test_fresh_intent_is_skipped():
    """With closing_in_progress=True and fresh close_initiated_at, the
    reconciler MUST leave the row alone so the bot's own close path can
    commit the real exit reason."""
    engine, SessionLocal = await _setup_db()
    async with SessionLocal() as s:
        order = await _insert_open_order(
            s,
            closing_in_progress=True,
            close_initiated_at=datetime.utcnow(),  # just now
        )
        oid = order.id

    async with SessionLocal() as s:
        closed = await _run_reconcile(s)
        assert len(closed) == 0, (
            f"expected 0 reconciled (fresh intent should be skipped), got {len(closed)}"
        )

    async with SessionLocal() as s:
        after = await _fetch_status(s, oid)
        assert after.status == "OPEN", f"status should remain OPEN, got {after.status}"
        assert after.close_reason is None, f"close_reason={after.close_reason}"
    await engine.dispose()
    print("  [OK] fresh intent — reconciler SKIPPED the row (55 ms race closed)")


async def test_stale_intent_is_reconciled():
    """If the intent is older than CLOSE_INTENT_STALE_SECONDS, the close path
    is assumed to have crashed.  Reconciler must proceed with EXTERNAL_CLOSE
    to avoid stuck-OPEN rows."""
    engine, SessionLocal = await _setup_db()
    stale_ts = datetime.utcnow() - timedelta(
        seconds=main.CLOSE_INTENT_STALE_SECONDS + 10
    )
    async with SessionLocal() as s:
        order = await _insert_open_order(
            s,
            closing_in_progress=True,
            close_initiated_at=stale_ts,
        )
        oid = order.id

    async with SessionLocal() as s:
        closed = await _run_reconcile(s)
        assert len(closed) == 1, (
            f"expected 1 reconciled (stale intent must not block forever), "
            f"got {len(closed)}"
        )

    async with SessionLocal() as s:
        after = await _fetch_status(s, oid)
        assert after.status == "CLOSED", f"status={after.status}"
        assert after.close_reason == "EXTERNAL_CLOSE", f"close_reason={after.close_reason}"
    await engine.dispose()
    print("  [OK] stale intent — reconciler proceeded with EXTERNAL_CLOSE as expected")


async def test_fresh_intent_without_timestamp_is_reconciled():
    """Defensive: if closing_in_progress=True but close_initiated_at is NULL
    (shouldn't happen in practice, but a DB-level inconsistency from a bad
    partial write could create it), the guard falls through to normal
    reconciliation.  This prevents a row from being stuck OPEN forever."""
    engine, SessionLocal = await _setup_db()
    async with SessionLocal() as s:
        order = await _insert_open_order(
            s,
            closing_in_progress=True,
            close_initiated_at=None,
        )
        oid = order.id

    async with SessionLocal() as s:
        closed = await _run_reconcile(s)
        assert len(closed) == 1, (
            f"intent without timestamp should NOT block reconciliation, got {len(closed)}"
        )

    async with SessionLocal() as s:
        after = await _fetch_status(s, oid)
        assert after.status == "CLOSED"
        assert after.close_reason == "EXTERNAL_CLOSE"
    await engine.dispose()
    print("  [OK] flag set but timestamp NULL — falls through to reconcile (safety)")


async def main_test():
    print("Running reconciler race guard tests...\n")
    await test_baseline_no_guard_marks_external_close()
    await test_fresh_intent_is_skipped()
    await test_stale_intent_is_reconciled()
    await test_fresh_intent_without_timestamp_is_reconciled()
    print("\nAll 4 reconciler race guard tests passed.")


if __name__ == "__main__":
    asyncio.run(main_test())
