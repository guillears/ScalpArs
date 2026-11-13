"""💸 Paper = live trading costs (operator 2026-11-13, DECISION_LOG 539) — pure helpers, no I/O, never raise.

① EXIT SLIPPAGE — a paper market (taker) exit fills where a live market order would: at the live book's best bid (closing a LONG = selling)
   / best ask (closing a SHORT = buying), the same way paper Frenzy / Flatbase entries already fill at the live book. The slippage is the fill
   vs the decision price (the price the exit rule saw), signed like the entry's: + = filled worse (a cost, inside the P&L), − = better.
   Resting / maker take-profit limits are untouched (a limit fills at its own price).
② FUNDING — every open position (paper AND live) accrues each funding settlement while it is open: payment = −side × rate × quantity ×
   mark price at the settlement (side +1 LONG / −1 SHORT; negative = paid). Since DECISION_LOG 539 funding is INSIDE pnl / pnl_percentage
   (Order.funding_in_pnl = True on those rows) and stays visible on its own in Order.funding_fee_usd. Rows closed before keep the old
   convention (pnl excludes funding) — every reader goes through pnl_with_funding / funding_outside_pnl, never adds the column blindly.
"""
import math
from typing import Iterable, Optional, Sequence, Tuple


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def paper_exit_book_price(direction, best_bid, best_ask) -> Optional[float]:
    """the price a market exit meets: LONG closes into the bid, SHORT into the ask. None when that side is missing / not positive."""
    px = _f(best_bid) if str(direction).upper() == "LONG" else _f(best_ask)
    return px if px is not None and px > 0 else None


def exit_slip_pct(direction, decision_price, fill_price) -> Optional[float]:
    """signed exit slippage, % of the decision price: + = worse for us (LONG sold lower / SHORT bought higher). None on bad input."""
    d, f = _f(decision_price), _f(fill_price)
    if d is None or f is None or d <= 0 or f <= 0:
        return None
    return round(((d - f) if str(direction).upper() == "LONG" else (f - d)) / d * 100, 4)


def funding_payment_usd(direction, quantity, rate, mark_price) -> Optional[float]:
    """one settlement's cash flow for the position, USDT (negative = we paid). A LONG pays a positive rate, a SHORT receives it."""
    q, r, m = _f(quantity), _f(rate), _f(mark_price)
    if q is None or r is None or m is None or q <= 0 or m <= 0:
        return None
    side = 1.0 if str(direction).upper() == "LONG" else -1.0
    return -side * r * q * m


def accrue_funding(direction, quantity, rows: Iterable[Sequence], since_ms, until_ms, fallback_mark=None) -> Tuple[float, Optional[int], int]:
    """Σ payments over the settlements with since_ms < time ≤ until_ms. rows = [(funding_time_ms, rate, mark_price | None), …] in any order;
    a row without a mark price uses fallback_mark (the position's current price), else it is skipped (counted in the 3rd value).
    → (Σ usd, the newest settlement time counted or None, rows skipped)."""
    tot, last, skipped = 0.0, None, 0
    try:
        a, b = int(since_ms), int(until_ms)
    except (TypeError, ValueError):
        return 0.0, None, 0
    for row in sorted((r for r in (rows or []) if r and _f(r[0]) is not None), key=lambda r: int(r[0])):
        t = int(row[0])
        if t <= a or t > b:
            continue
        m = _f(row[2]) if len(row) > 2 else None
        p = funding_payment_usd(direction, quantity, row[1], m if m else fallback_mark)
        if p is None:
            skipped += 1
            continue
        tot += p
        last = t
    return tot, last, skipped


def funding_outside_pnl(funding_fee_usd, funding_in_pnl) -> float:
    """the funding a reader must STILL add to pnl to get the true net: 0 when the row already carries it (DL 539), else the column (0 if NULL)."""
    if funding_in_pnl:
        return 0.0
    v = _f(funding_fee_usd)
    return v if v is not None else 0.0


def pnl_with_funding(pnl, funding_fee_usd, funding_in_pnl) -> Optional[float]:
    """the true net P&L in USDT (fees AND funding) under either convention."""
    p = _f(pnl)
    return None if p is None else p + funding_outside_pnl(funding_fee_usd, funding_in_pnl)


def fold_funding(pnl, pnl_pct, notional, funding_usd) -> Tuple[Optional[float], Optional[float]]:
    """(pnl, pnl %) with funding_usd added; the % on the same notional the engine uses (entry price × quantity). Bad input → unchanged."""
    p, pc, n, fu = _f(pnl), _f(pnl_pct), _f(notional), _f(funding_usd)
    if p is None or fu is None:
        return pnl, pnl_pct
    newp = round(p + fu, 4)
    newpc = round(pc + fu / n * 100, 4) if (pc is not None and n is not None and n > 0) else pnl_pct
    return newp, newpc


HOUR_MS = 3_600_000


def funding_topup_needed(since_ms, now_ms) -> bool:
    """True when a whole hour boundary lies in (since_ms, now_ms] — Binance settles funding only on the hour (1 / 4 / 8 h grids), so with no
    boundary there is nothing to read (the close funnel skips the HTTP top-up for every short scalp). Bad input → True (read)."""
    try:
        a, b = int(since_ms), int(now_ms)
    except (TypeError, ValueError):
        return True
    return b // HOUR_MS > a // HOUR_MS


def live_income_foreign(income_usd, estimate_usd, mult=1.25, slack=0.05) -> bool:
    """True when the account's FUNDING_FEE income on the symbol is far beyond this position's public-rate estimate — the income then holds
    another position's funding (the operator's own app trade on the same coin, one-way mode) and the row books the estimate instead."""
    i, e = _f(income_usd), _f(estimate_usd)
    if i is None or e is None:
        return False
    return abs(i) > mult * abs(e) + slack
