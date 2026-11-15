"""📖 (DECISION_LOG 551) ENTRY DEPTH GUARD — size a taker entry to the order book it will meet.

Operator 2026-11-15 (first live trade, SHELL FRENZY_LITE $16,575): the market order filled 0.32 % above the best ask on a thin book
(11 earlier small live LITE fills: never above 0.11 %). The guard reads the book in the same call the dislocation guard already makes
and keeps the position to what fills within `max_sweep_pct` of the best price: full size → unchanged · part fits (≥ min_frac) → shrink
to what fits · less than min_frac fits → skip. It only ever shrinks or skips; a missing / unreadable book is fail-open (unchanged).
Pure math here; the engine wiring lives in TradingEngine.open_position (the FRENZY / SURGE / FLATBASE taker path)."""

FULL, SHRINK, SKIP = "FULL", "SHRINK", "SKIP"


def depth_fillable_usd(levels, direction, max_sweep_pct):
    """$ that fills within max_sweep_pct of the best price on the side a taker entry hits: LONG buys the asks (best first, rising),
    SHORT sells into the bids (best first, falling). levels = [[price, qty], …] best first. None when the book is unusable or the read
    never reached a level beyond the limit (a truncated read cannot tell a dense book from a thin one)."""
    try:
        if not levels or max_sweep_pct is None or float(max_sweep_pct) < 0:
            return None
        best = float(levels[0][0])
        if best <= 0:
            return None
        lim = best * (1 + float(max_sweep_pct) / 100.0) if direction == "LONG" else best * (1 - float(max_sweep_pct) / 100.0)
        tot = 0.0
        for lv in levels:
            p, q = float(lv[0]), float(lv[1])
            if (direction == "LONG" and p > lim + 1e-15) or (direction != "LONG" and p < lim - 1e-15):
                return tot
            if p > 0 and q > 0:
                tot += p * q
        return None   # every level read sits inside the push limit: the read was truncated, the depth is unmeasured (caveman review) → fail-open
    except (TypeError, ValueError, IndexError):
        return None


def depth_guard_size(planned_notional, fillable_usd, min_frac):
    """(action, notional): FULL (planned fits, or no usable book → fail-open) · SHRINK (min_frac ≤ fits < planned → what fits) ·
    SKIP (fits < min_frac × planned). Never returns more than planned."""
    n = float(planned_notional or 0.0)
    if n <= 0 or fillable_usd is None:
        return FULL, n
    f = max(0.0, float(fillable_usd))
    if f >= n:
        return FULL, n
    if f < max(0.0, float(min_frac or 0.0)) * n:
        return SKIP, f
    return SHRINK, f
