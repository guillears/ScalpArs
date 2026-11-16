"""💸 FUNDING GUARD (DECISION_LOG 566, operator "ship it" after the live INIT 11-16 FAN flip: opened 96 s before an HOURLY −0.183 % funding
payment → $33.46 on an $18.3k position). Pure rule, no I/O — the engine reads Binance premiumIndex (lastFundingRate + nextFundingTime) and
asks here whether to skip the entry.

Skip a bot entry when the next payment is AGAINST the trade (a SHORT pays when the rate is negative, a LONG when it is positive), its size
|rate| ≥ rate_min_pct (percent of notional) and it settles within `minutes` from now. Anything unreadable → no block (fail-open).
Evidence (reports/FUNDING_GUARD_EXIT_DEPTH_STUDY_2026-11-16.md): 0.05 % · 15 min blocks 1 of 564 master + live fills (INIT); extreme-funding
fills are otherwise mostly winners (58 · 86 % · +2.35 %), so the cut is a narrow cost guard, never a funding filter."""
import math
from typing import Optional, Tuple


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def funding_against(direction, rate) -> bool:
    r = _f(rate)
    if r is None:
        return False
    return (str(direction).upper() == "SHORT" and r < 0) or (str(direction).upper() == "LONG" and r > 0)


def funding_guard_block(direction, rate, next_ms, now_ms, rate_min_pct, minutes) -> Tuple[bool, str]:
    """(block, why). rate = decimal (−0.00183 = −0.183 %); next_ms = the next settlement (ms); rate_min_pct in percent; minutes ≥ 0."""
    r, nx, now, mn, mins = _f(rate), _f(next_ms), _f(now_ms), _f(rate_min_pct), _f(minutes)
    if r is None or nx is None or now is None or mn is None or mins is None or mn <= 0 or mins <= 0:
        return False, ""
    left_min = (nx - now) / 60000.0
    if left_min < 0 or left_min > mins:
        return False, ""
    if not funding_against(direction, r) or abs(r) * 100.0 < mn:
        return False, ""
    return True, (f"funding {r * 100:+.3f} % against the {str(direction).upper()} settles in {left_min:.1f} min "
                  f"(guard: ≥ {mn:g} % within {mins:g} min)")
