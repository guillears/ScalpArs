"""📥 Import an app-opened Binance position (DECISION_LOG 328, operator 2026-10-20).

PURE helpers (no I/O) for TradingEngine.import_binance_position / import_candidates. Why the feature exists: the operator opens positions
directly in the Binance app; the reconciler then sees a position with no row (check_untracked_positions → CRITICAL [UNTRACKED_POSITION]
NO_ROW, entries on the pair blocked) — no stop / target management, not in the tables. The old POST /api/recover-positions bulk-adopts
such positions as bare RECOVERED rows (no exit, no exchange safety stop, no UI). The 📥 import takes ONE position, confirmed by the
operator, and books it as a 🖐 MANUAL row with a real exit mode and a resting exchange safety stop.

Conventions (the same every exit uses):
  * P&L % is NET of fees on the entry notional (services.position_add.pnl_pct_net) — the import's entry fee is an ESTIMATE (taker rate
    × entry notional): Binance's real fill fee of an app order is not read here (DECISION_LOG 329 auto-record will provide it).
  * One-way position mode: one position per symbol (the bot's assumption, checked at go-live).
"""
import math
from typing import Optional

from services.position_add import qty_matches

IMPORT_SOURCE_BINANCE_APP = "BINANCE_APP"   # Order.import_source of a 📥 imported row
IMPORT_LEV_INT_TOL = 1e-3                   # Binance's leverage is an integer; a notional / margin ratio within 0.1 % of one counts as it
IMPORT_ENTRY_TOL_SHARE = 0.005              # the re-read before the commit: the average entry must match the first read within 0.5 %


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def import_pick_position(positions, symbol, pair, direction, min_notional) -> dict:
    """The ONE Binance position the import may take, from a get_open_positions() list (never None — the caller refuses a failed read).
    Refused (ValueError, readable): no position on the symbol · more than one (hedge mode) · the other side · zero size · a notional below
    min_notional (dust a close left — not a position)."""
    ex = [x for x in (positions or ()) if (x or {}).get('symbol') == symbol]
    if not ex:
        raise ValueError(f"Binance shows no {pair} position right now — nothing to import (closed in the app?)")
    if len(ex) > 1:
        raise ValueError(f"Binance holds more than one {pair} position (hedge mode?) — nothing imported")
    p = ex[0]
    side = str(p.get('side') or '').upper()
    if side != direction:
        raise ValueError(f"Binance holds a {side or '?'} {pair} position, not {direction} — nothing imported")
    qty, entry = _f(p.get('contracts')), _f(p.get('entry_price'))
    if not (qty and qty > 0):
        raise ValueError(f"the {pair} position on Binance has no size — nothing imported")
    if not (entry and entry > 0):
        raise ValueError(f"Binance reports no entry price for the {pair} position — nothing imported; try again in a moment")
    if qty * entry < float(min_notional):
        raise ValueError(f"the {pair} position is only ${qty * entry:,.2f} (under ${float(min_notional):g}) — dust, not a position; nothing imported")
    return p


def import_whole_leverage(lev) -> Optional[int]:
    """Binance's position leverage as a whole number ≥ 1, or None when unknown / not whole (the import then refuses: every live path
    — backstop distance, liquidation floor, a later ➕ add — needs the exact integer the exchange holds)."""
    x = _f(lev)
    if x is None or x < 1.0 - IMPORT_LEV_INT_TOL:
        return None
    r = round(x)
    if r < 1 or abs(x - r) > IMPORT_LEV_INT_TOL * r:
        return None
    return int(r)


def import_position_unchanged(first, again, step=0.0) -> bool:
    """The re-read right before the commit still shows the position the operator confirmed: same side, size within max(1 step, 0.5 %),
    average entry within 0.5 %. Anything else (a close, a partial close, an add in the app) → the import is refused, nothing written."""
    if not first or not again:
        return False
    if str(first.get('side') or '').upper() != str(again.get('side') or '').upper():
        return False
    if not qty_matches(again.get('contracts'), first.get('contracts'), step):
        return False
    e1, e2 = _f(first.get('entry_price')), _f(again.get('entry_price'))
    if not (e1 and e1 > 0 and e2 and e2 > 0):
        return False
    return abs(e2 - e1) <= IMPORT_ENTRY_TOL_SHARE * e1 + 1e-12


def import_level_refusal(pnl_now, stop_pct, tp_pct) -> Optional[str]:
    """A stop / target the price has ALREADY passed (P&L % net of fees on the imported entry) would close the trade at the first tick —
    refused like set_exit_override refuses it. stop_pct is negative (None = no fixed stop to judge), tp_pct positive or None."""
    p = _f(pnl_now)
    if p is None:
        return "the current price of this position is not available — try again in a moment"
    s, t = _f(stop_pct), _f(tp_pct)
    if s is not None and s >= p:
        return f"the position is at {p:+.2f}% now (net of fees) — a stop at {s:+.2f}% would close it at once. Use a wider stop, or close it in the app"
    if t is not None and t <= p:
        return f"the position is at {p:+.2f}% now (net of fees) — a target at {t:+.2f}% is already reached. Use a higher target, or close it in the app"
    return None
