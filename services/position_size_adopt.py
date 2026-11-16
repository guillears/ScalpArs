"""📐 POSITION SIZE ADOPTION (DECISION_LOG 572, operator 2026-11-16: "if I added to the position in Binance, does our trade update the
position size?" → "yes build it"). Pure rules, no I/O — the engine (TradingEngine.adopt_position_sizes) feeds them the 15 s reconciler's
own positions read.

Why: the bot only knew positions it had NO row for (UNTRACKED NO_ROW). A size change made in the Binance app on a TRACKED position (HUMA
SHORT #25, 11-16: the operator added size) left the row's quantity, its resting reduce-only take profit, P&L and funding on the old size,
and a bot close would have left the added part open.

Rule: a LIVE OPEN row whose pair AND side match a Binance position whose size differs from the row's quantity by more than
max(one amount step, SIZE_TOL_FRAC of the row's quantity), seen with the SAME Binance size on SIZE_CONFIRM_PASSES consecutive passes
(never a fill still being written / an add in flight) → the row adopts Binance's size and average entry (the exchange's own numbers —
nothing invented). Margin follows at the row's leverage. The entry fee is kept: the added leg's fee is unknown (not invented); a reduction
is recorded as a size change — its realised part is not booked by the bot (Binance's balance carries it; the final close books the rest).
Every helper is total (never raises on bad input)."""
import math
from typing import Dict, Optional

SIZE_TOL_FRAC = 0.005          # 0.5 % of the row's quantity …
SIZE_CONFIRM_PASSES = 2        # … on the same Binance size two reconcile passes in a row …
SIZE_CONFIRM_MIN_GAP_S = 10.0  # … at least this far apart (monotonic) — two reconcile runs racing (monitor loop + POST /api/reconcile-positions)
#                                never confirm each other within the same moment (572 review 3)

# 📐 (572 review 4) where each row keeps the NET % its software TP reads — re-derived at the pinned price so the software TP and the
# resting limit sit on the SAME price. A row without such a column is not pinned (its TP re-places at its own rule on the new average).
PIN_TARGET_COLUMN = {"MANUAL": "pattern_fixed_tp_pct", "FRENZY_LITE": "frenzy_lite_tp_used", "FRENZY_LITE_RETRY": "frenzy_lite_tp_used",
                     "FLIP:FAN_RATIO_GATE": "flip_fan_tp_pct"}


def pin_target_column(entry_strategy, exit_override_at=None) -> Optional[str]:
    """The row column holding the net % TP target its software TP reads (an operator exit override → pattern_fixed_tp_pct), or None."""
    if exit_override_at is not None:
        return "pattern_fixed_tp_pct"
    return PIN_TARGET_COLUMN.get(str(entry_strategy or ""))


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def size_tolerance(row_qty, step) -> float:
    q, s = _f(row_qty) or 0.0, _f(step) or 0.0
    return max(abs(s), SIZE_TOL_FRAC * abs(q))


def size_differs(row_qty, binance_qty, step) -> bool:
    """True when Binance's size is materially different from the row's (either way)."""
    q, b = _f(row_qty), _f(binance_qty)
    if q is None or b is None or q <= 0 or b <= 0:
        return False
    return abs(b - q) > size_tolerance(q, step)


def confirm_pass(seen: Dict, key, binance_qty, step, row_qty, now_s, min_gap_s=SIZE_CONFIRM_MIN_GAP_S) -> bool:
    """Two-pass confirmation on the SAME Binance size, ≥ min_gap_s apart (monotonic seconds): records this sighting in `seen`
    (key → (Binance size, count, time of the last counted sighting)) and returns True on the SIZE_CONFIRM_PASSES-th consecutive matching
    sighting (the entry is then cleared). A different size restarts the count; a matching sighting too soon is ignored (never counted)."""
    b, now = _f(binance_qty), _f(now_s)
    if b is None or now is None:
        seen.pop(key, None)
        return False
    prev = seen.get(key)
    tol = size_tolerance(row_qty, step)
    if prev is not None and abs(float(prev[0]) - b) <= tol:
        if now - float(prev[2]) < min_gap_s:
            return False   # a second reconcile run in the same moment: not an independent sighting
        n = int(prev[1]) + 1
    else:
        n = 1
    if n >= SIZE_CONFIRM_PASSES:
        seen.pop(key, None)
        return True
    seen[key] = (b, n, now)
    return False


def adopt_values(row_qty, row_entry, row_leverage, binance_qty, binance_entry) -> Optional[dict]:
    """The row's new numbers from Binance's position: quantity, entry_price (Binance's average), notional_value, investment (margin at the
    row's leverage), plus the change (delta_qty > 0 = added, < 0 = reduced). None when anything needed is unreadable."""
    q, e, lev, b = _f(row_qty), _f(row_entry), _f(row_leverage), _f(binance_qty)
    be = _f(binance_entry)
    if not (q and q > 0 and b and b > 0 and lev and lev > 0):
        return None
    entry = be if (be and be > 0) else e
    if not (entry and entry > 0):
        return None
    notional = b * entry
    return dict(quantity=b, entry_price=entry, notional_value=notional, investment=notional / lev,
                delta_qty=b - q, from_qty=q, from_entry=e)
