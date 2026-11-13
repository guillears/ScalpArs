"""💸 (DECISION_LOG 539) open-trade funding accrual — paper AND live, own task + own sessions, never the trading loops.

Every OPEN row whose last counted settlement (funding_accrued_ms, else its open time) is ≥ ACCRUE_MIN_GAP_MS old gets the settlements since
read from the PUBLIC funding history and added to funding_fee_usd (negative = paid), funding_accrued_ms moved to the newest one counted.
The write is a targeted UPDATE guarded on status OPEN and the previous funding_accrued_ms (a concurrent close or a second pass can never
double-count; the close funnel folds what is here + its own top-up into pnl — services.trading_engine._funding_at_close).
Display / bookkeeping only: no exit rule reads funding (operator 2026-11-13: TP / stop unchanged until a study says otherwise).
"""
import asyncio
import logging
import time
from datetime import timezone

from sqlalchemy import and_, func, select, update

from database import AsyncSessionLocal, locked_commit
from models import Order
from services.binance_service import binance_service
from services.paper_costs import accrue_funding

logger = logging.getLogger(__name__)

WAKE_S = 600                        # every 10 min
ACCRUE_MIN_GAP_MS = 55 * 60 * 1000  # the shortest funding interval is 1 h: never read a contract more than ~once an hour
READ_TIMEOUT_S = 5.0
_LAST_READ: dict = {}               # 💸 (539 review) order id → ms of its last read: the throttle lives here, the marker only ever moves to a
                                    # REAL published settlement (a late-published one is never skipped)


def _open_ms(o) -> int:
    return int(o.opened_at.replace(tzinfo=timezone.utc).timestamp() * 1000)


async def accrue_once(now_ms=None) -> int:
    """one pass → rows updated. Never raises."""
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    n = 0
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(select(Order.id, Order.pair, Order.direction, Order.quantity, Order.opened_at, Order.funding_accrued_ms,
                                            Order.entry_price).where(Order.status == "OPEN"))).all()
    except Exception as e:
        logger.warning(f"[FUNDING_ACCRUE] open-row read failed: {str(e)[:120]}")
        return 0
    for r in rows:
        try:
            if r.opened_at is None:
                continue
            since = int(r.funding_accrued_ms) if r.funding_accrued_ms is not None else _open_ms(r)
            if now - max(since, _LAST_READ.get(r.id, 0)) < ACCRUE_MIN_GAP_MS:
                continue
            _LAST_READ[r.id] = now
            hist = await asyncio.wait_for(binance_service.fetch_funding_history(r.pair, since, now), READ_TIMEOUT_S)
            if hist is None:
                continue
            add, last, _skip = accrue_funding(r.direction, r.quantity, hist, since, now, fallback_mark=r.entry_price)
            if last is None:   # no settlement published since the marker: nothing to add, the marker stays (throttled by _LAST_READ)
                continue
            async with AsyncSessionLocal() as db:
                guard = (Order.funding_accrued_ms.is_(None) if r.funding_accrued_ms is None else Order.funding_accrued_ms == r.funding_accrued_ms)
                res = await db.execute(update(Order).where(and_(Order.id == r.id, Order.status == "OPEN", guard))
                                       .values(funding_fee_usd=func.coalesce(Order.funding_fee_usd, 0.0) + add, funding_accrued_ms=int(last)))
                await locked_commit(db)
                if res.rowcount:
                    n += 1
                    logger.info(f"[FUNDING_ACCRUE] {r.pair} #{r.id} {r.direction}: {add:+.4f} USDT over the settlements to {last} (open trade)")
        except Exception as e:
            logger.warning(f"[FUNDING_ACCRUE] {getattr(r, 'pair', '?')} #{getattr(r, 'id', '?')}: {str(e)[:120]}")
    for k in [k for k in _LAST_READ if k not in {r.id for r in rows}]:   # closed rows leave the throttle map
        _LAST_READ.pop(k, None)
    return n


async def funding_accrual_loop(should_stop) -> None:
    await asyncio.sleep(90)
    while not should_stop():
        try:
            await accrue_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[FUNDING_ACCRUE] loop error (bookkeeping only, trading unaffected): {str(e)[:120]}")
        await asyncio.sleep(WAKE_S)
