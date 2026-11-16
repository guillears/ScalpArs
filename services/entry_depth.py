"""📖 (DECISION_LOG 551) ENTRY DEPTH GUARD — size a taker entry to the order book it will meet.

Operator 2026-11-15 (first live trade, SHELL FRENZY_LITE $16,575): the market order filled 0.32 % above the best ask on a thin book
(11 earlier small live LITE fills: never above 0.11 %). The guard reads the book in the same call the dislocation guard already makes
and keeps the position to what fills within `max_sweep_pct` of the best price: full size → unchanged · part fits (≥ min_frac) → shrink
to what fits · less than min_frac fits → skip. It only ever shrinks or skips; a missing / unreadable book is fail-open (unchanged).
Pure math here; the engine wiring lives in TradingEngine.open_position (the FRENZY / SURGE / FLATBASE taker path).
📖↔ (DECISION_LOG 565) every bot entry (momentum, flips and every other open_position sleeve too, entry_depth_guard_all_enabled) and the
EXIT side of the book (entry_depth_guard_exit_side_enabled): the notional is kept to the THINNER side — a thin exit side is where the
INIT 11-16 flip lost 0.37 % on its market close."""

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


def exit_side_fillable_usd(book, direction, max_sweep_pct):
    """📖↔ (DECISION_LOG 565) $ that the EXIT of this trade would fill within max_sweep_pct of the best price on the book read at entry: a LONG
    exits by selling into the bids, a SHORT by buying the asks. book = {'bids': [...], 'asks': [...]} best first. None when unusable."""
    try:
        if not book:
            return None
        return depth_fillable_usd(book.get('bids') if direction == "LONG" else book.get('asks'), "SHORT" if direction == "LONG" else "LONG", max_sweep_pct)
    except (AttributeError, TypeError):
        return None


def thinner_side_usd(entry_fill, exit_fill):
    """the depth the guard sizes to: the thinner of the two readable sides; None when neither side is measured (fail-open)."""
    v = [float(x) for x in (entry_fill, exit_fill) if x is not None]
    return min(v) if v else None


# ── 📖🧢 (DECISION_LOG 553) CAPPED ENTRY — the guarded order can never fill above the price the guard measured ─────────────────────────────
# Operator 2026-11-15 (INIT LITE: fill 0.32 % above the book the guard read, ~1 s later): the entry is a LIMIT order at the guard's ceiling
# (best price ± max_sweep_pct, rounded to the tick on the safe side); what fills at once fills, the rest rests up to wait_s seconds (fills
# only if the price comes back to the ceiling), then is cancelled — the trade keeps what filled. No fill at all = no trade.

import math as _math


def cap_price(best, direction, max_sweep_pct, tick):
    """the guard's ceiling as an order price: LONG best × (1 + p) rounded DOWN to the tick (never above the ceiling) · SHORT best × (1 − p)
    rounded UP. None when the inputs are unusable."""
    try:
        b, p, t = float(best), float(max_sweep_pct), float(tick or 0)
        if b <= 0 or p < 0:
            return None
        raw = b * (1 + p / 100.0) if direction == "LONG" else b * (1 - p / 100.0)
        if t <= 0:
            return raw
        steps = raw / t
        n = _math.floor(steps + 1e-9) if direction == "LONG" else _math.ceil(steps - 1e-9)
        px = n * t
        if direction == "LONG" and px < b:          # a ceiling below the best ask could never fill: at least the best price
            px = b
        if direction != "LONG" and px > b:
            px = b
        return px
    except (TypeError, ValueError):
        return None


def capped_fill_fee(immediate_qty, total_qty, avg_price, taker_rate, maker_rate):
    """entry fee of a capped entry: the part that filled at once crossed the book (taker), the part that filled while resting was maker."""
    q, q0, px = max(0.0, float(total_qty or 0)), max(0.0, float(immediate_qty or 0)), float(avg_price or 0)
    q0 = min(q0, q)
    return q0 * px * float(taker_rate or 0) + (q - q0) * px * float(maker_rate or 0)
