"""🤖 Origin view (DECISION_LOG 323, operator 2026-10-20): the dashboard's All · Bot · Manual switch.

PURE helpers (no I/O). Who opened the trade decides its side:
  MANUAL = the operator's own entries — entry_strategy 'MANUAL' (the 🖐 pop-up) or a position the bot ADOPTED from Binance
           (confidence 'RECOVERED': a row recover_positions created for a position it found without one — normally the operator's
           app trades; a bot position whose row was LOST is also adopted this way and then counts as MANUAL — rare, accepted).
           A 📥 IMPORTED app position (DECISION_LOG 328: import_source 'BINANCE_APP') is booked with entry_strategy 'MANUAL' → MANUAL by
           the first rule; no extra rule is needed (and RECOVERED stays MANUAL as before)
  BOT    = everything else. A bot trade whose exit the operator edited (exit_override_at set), closed in the Binance app
           (close_reason EXTERNAL_CLOSE) or ADDED size to (➕ manual_added_at set, DECISION_LOG 324) stays BOT — the bot chose the
           entry — and is counted as "touched by you".

A filtered view is the account as if only that side had traded: the same starting capital and transfers, only that side's P&L
(the callers keep the real-account starting point and remove the other side's P&L from it).
"""
from typing import Iterable, Optional, Tuple

ORIGINS = ("ALL", "BOT", "MANUAL")


def norm_origin(v) -> str:
    s = str(v or "").strip().upper()
    return s if s in ORIGINS else "ALL"


def order_origin(entry_strategy, confidence=None) -> str:
    if str(entry_strategy or "").strip().upper() == "MANUAL" or str(confidence or "").strip().upper() == "RECOVERED":
        return "MANUAL"
    return "BOT"


def origin_keep(entry_strategy, confidence, origin) -> bool:
    o = norm_origin(origin)
    return o == "ALL" or order_origin(entry_strategy, confidence) == o


def bot_touched(entry_strategy, confidence, exit_override_at, close_reason, manual_added_at=None) -> bool:
    """a BOT trade the operator touched: exit edited, closed by hand in the Binance app, or size added with ➕ (324)."""
    if order_origin(entry_strategy, confidence) != "BOT":
        return False
    return (exit_override_at is not None or manual_added_at is not None
            or str(close_reason or "").upper() == "EXTERNAL_CLOSE")


def touched_summary(orders: Iterable) -> dict:
    """{n, pnl} over the BOT trades the operator touched (exit edited / closed in the app / ➕ size added) — never raises on a bad row."""
    n, pnl = 0, 0.0
    for o in orders or ():
        try:
            if bot_touched(getattr(o, 'entry_strategy', None), getattr(o, 'confidence', None),
                           getattr(o, 'exit_override_at', None), getattr(o, 'close_reason', None),
                           getattr(o, 'manual_added_at', None)):
                n += 1
                pnl += float(getattr(o, 'pnl', None) or 0)
        except (TypeError, ValueError):
            continue
    return {"n": n, "pnl": round(pnl, 2)}
