"""💸 (DECISION_LOG 539) open-trade funding accrual — paper AND live, own task + own sessions, never the trading loops.

Every OPEN row whose last counted settlement (funding_accrued_ms, else its open time) is ≥ ACCRUE_MIN_GAP_MS old gets the settlements since
read from the PUBLIC funding history and added to funding_fee_usd (negative = paid), funding_accrued_ms moved to the newest one counted.
The write is a targeted UPDATE guarded on status OPEN and the previous funding_accrued_ms (a concurrent close or a second pass can never
double-count; the close funnel folds what is here + its own top-up into pnl — services.trading_engine._funding_at_close).
Bookkeeping + (DECISION_LOG 607) the FLATBASE stop: OPEN_FUNDING_USD is the in-process view the realtime FLATBASE stop reads (flatbase_stop_counts_funding);
every other exit rule ignores funding (operator 2026-11-13; EXH / DIST studied and kept price-only, DL 607).
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
OPEN_FUNDING_USD: dict = {}         # 💸 (DECISION_LOG 607) order id → Σ funding_fee_usd of an OPEN row as of the last pass / write (− = paid):
                                    # the realtime FLATBASE stop reads it (the tick cache carries no funding). Refreshed from the DB every
                                    # pass (≤ 10 min) and on each accrual write; closed rows leave it.


def clear_open_funding() -> None:
    """💸 (607 review) a paper reset / row delete (order ids restart) never inherits funding state."""
    OPEN_FUNDING_USD.clear()
    _LAST_READ.clear()


def open_funding_usd(order_id, fallback=None):
    """💸 (607) the accrued funding of an open row (− = paid) — the newest of this process's view, else `fallback` (the row's own
    funding_fee_usd). Never raises."""
    try:
        v = OPEN_FUNDING_USD.get(order_id)
        return v if v is not None else fallback
    except Exception:
        return fallback


def _open_ms(o) -> int:
    return int(o.opened_at.replace(tzinfo=timezone.utc).timestamp() * 1000)


async def accrue_once(now_ms=None) -> int:
    """one pass → rows updated. Never raises."""
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    n = 0
    try:
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(select(Order.id, Order.pair, Order.direction, Order.quantity, Order.opened_at, Order.funding_accrued_ms,
                                            Order.entry_price, Order.funding_fee_usd).where(Order.status == "OPEN"))).all()
    except Exception as e:
        logger.warning(f"[FUNDING_ACCRUE] open-row read failed: {str(e)[:120]}")
        return 0
    for r in rows:   # 💸 (607) refresh the open-row funding view first (a restart / a skipped read still leaves the DB value)
        try:
            if r.funding_fee_usd is not None:
                OPEN_FUNDING_USD[r.id] = float(r.funding_fee_usd)
            else:
                OPEN_FUNDING_USD.pop(r.id, None)   # (607 review) a reused id (paper reset) never inherits an old row's funding
        except Exception:
            pass
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
            if last is None:   # no settlement published since the marker: nothing to add, the marker stays
                _LAST_READ.pop(r.id, None)   # 💸 (607 review) re-read next pass (≤ 10 min) — a late-published hourly settlement never waits ~55 min
                continue
            async with AsyncSessionLocal() as db:
                guard = (Order.funding_accrued_ms.is_(None) if r.funding_accrued_ms is None else Order.funding_accrued_ms == r.funding_accrued_ms)
                res = await db.execute(update(Order).where(and_(Order.id == r.id, Order.status == "OPEN", guard))
                                       .values(funding_fee_usd=func.coalesce(Order.funding_fee_usd, 0.0) + add, funding_accrued_ms=int(last)))
                await locked_commit(db)
                if res.rowcount:
                    n += 1
                    OPEN_FUNDING_USD[r.id] = float(r.funding_fee_usd or 0.0) + float(add)   # 💸 (607) the guarded write landed
                    logger.info(f"[FUNDING_ACCRUE] {r.pair} #{r.id} {r.direction}: {add:+.4f} USDT over the settlements to {last} (open trade)")
        except Exception as e:
            logger.warning(f"[FUNDING_ACCRUE] {getattr(r, 'pair', '?')} #{getattr(r, 'id', '?')}: {str(e)[:120]}")
    for k in [k for k in _LAST_READ if k not in {r.id for r in rows}]:   # closed rows leave the throttle map
        _LAST_READ.pop(k, None)
    for k in [k for k in OPEN_FUNDING_USD if k not in {r.id for r in rows}]:   # 💸 (607) closed rows leave the funding view
        OPEN_FUNDING_USD.pop(k, None)
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
