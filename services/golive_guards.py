"""🛡 Go-live guards (2026-10-14, DECISION_LOG 292 — operator-approved must-fix items 1–5 of reports/GO_LIVE_AUDIT_2026-10-14.md).

PURE helpers only (no I/O, no config import): the engine / main.py wire them to the exchange and the DB. Kept in one module so the
tests pin the money-safety rules without a running engine.

  ① per-order backstop record + re-place on change   → backstop_trigger_price · backstop_needs_replace · algo_status_resting
  ② untracked Binance positions                      → find_untracked_positions · untracked_action
  ④ safe paper → live switch                         → preflight_failures
  ⑤ stop-0 guard                                     → bot_stop_errors · sanitize_bot_stops
"""
import math
from typing import Dict, Iterable, List, Optional, Tuple

# ① a configured backstop that moved less than this (price-% points) is left resting — order_backstop_pct rounds to 4 decimals, so
# anything above float noise is a real config change; 0.02 keeps a sub-tick wiggle from cancelling + re-placing a live stop
BACKSTOP_REPLACE_TOL_PCT = 0.02
# … and a trigger that moved less than this fraction (0.02 %) is the same stop (an M2 entry-price repair moves it more than that)
BACKSTOP_REPLACE_TRIGGER_TOL = 0.0002
BACKSTOP_REPLACE_RETRY_S = 60.0          # a failed re-place is retried at most once a minute per order (never hammer the exchange)
BACKSTOP_REPRICE_INTERVAL_S = 30.0       # the re-price check runs at most every 30 s (the heal sweep itself runs every monitor pass)
ALGO_RESTING_STATUSES = ("NEW",)         # Binance algo order statuses that are still resting on the book
BACKSTOP_REPLACE_MAX_PER_PASS = 2        # live re-places per sweep pass (~4 exchange calls each) — a config change never stalls the 1 Hz monitor
BACKSTOP_HEAL_RETRY_S = 10.0             # the heal sweep retries a FAILED placement at most every 10 s per order (was every pass)
# the heal sweep leaves an order alone while the bot's OWN close is in flight (close_position nulls the backstop id right before its market
# close — KMNOUSDT 2026-10-17 00:32: the sweep re-placed into the gap and Binance answered -4509 on the now-flat symbol). The real gap is
# ~1–3 s (cancel → market close → CLOSED commit), so the grace is kept SHORT: a close that raised or failed is healed again after 20 s
# (review: a long grace would stretch any no-stop window on a failed close). DECISION_LOG 308.
BACKSTOP_CLOSE_GRACE_S = 20.0

# ② the action on a confirmed untracked position
UNTRACKED_ACTIONS = ("alert", "flatten")
UNTRACKED_CONFIRM_PASSES = 2             # seen on 2 consecutive reconcile passes (≈ 60 s apart) before it is called untracked
UNTRACKED_MIN_NOTIONAL_USD = 5.0         # a leftover below Binance's 5 USDT min notional is dust, not a position — never blocks a symbol

# ⑤ the stop a FRENZY-family sleeve falls back to when a loaded config carries ≤ 0 (never run without a bot stop)
SAFE_FRENZY_STOP_PCT = 3.0
SAFE_WILLY_STOP_PCT = 4.0
BOT_STOP_FIELDS = (("frenzy_stop_pct", SAFE_FRENZY_STOP_PCT), ("frenzy_willy_stop_pct", SAFE_WILLY_STOP_PCT))


def backstop_trigger_price(entry_price, direction, pct) -> Optional[float]:
    """The resting stop's trigger: pct % beyond the entry, against the position (LONG below, SHORT above). None on bad input."""
    try:
        px, p = float(entry_price), float(pct)
    except (TypeError, ValueError):
        return None
    if not (px > 0) or not (p > 0):
        return None
    return px * (1 - p / 100.0) if str(direction or "").upper() == "LONG" else px * (1 + p / 100.0)


def trigger_already_crossed(direction, trigger, mark) -> bool:
    """The new trigger is already through the current price (LONG: mark ≤ trigger · SHORT: mark ≥ trigger) — Binance would refuse it
    (-2021 'would immediately trigger') AFTER the -4130 clear had removed the old stop. Unreadable mark (≤ 0 / None) → False."""
    try:
        t, m = float(trigger), float(mark)
    except (TypeError, ValueError):
        return False
    if not (t > 0) or not (m > 0):
        return False
    return m <= t if str(direction or "").upper() == "LONG" else m >= t


def backstop_needs_replace(stored_pct, stored_trigger, want_pct, want_trigger,
                           tol_pct=BACKSTOP_REPLACE_TOL_PCT, tol_trigger=BACKSTOP_REPLACE_TRIGGER_TOL) -> bool:
    """True when the resting stop no longer matches the configured one: nothing recorded (a row opened before the record existed —
    its distance is unknown, so it is re-placed once), the % moved more than tol_pct, or the trigger moved more than tol_trigger
    (relative). A want that cannot be computed → False (never cancel a live stop for a value we cannot place)."""
    try:
        wp, wt = float(want_pct), float(want_trigger)
    except (TypeError, ValueError):
        return False
    if not (wp > 0 and wt > 0):
        return False
    try:
        sp, stg = float(stored_pct), float(stored_trigger)
    except (TypeError, ValueError):
        return True
    if not (sp > 0 and stg > 0):
        return True
    return abs(wp - sp) > tol_pct or abs(wt / stg - 1.0) > tol_trigger


def algo_status_resting(query_result) -> bool:
    """A query_backstop_stop() answer says the algo order is still resting. None / unreadable → False (unknown is not 'resting')."""
    if not isinstance(query_result, dict):
        return False
    return str(query_result.get("algoStatus", "")).upper() in ALGO_RESTING_STATUSES


def _pair_of(symbol) -> str:
    return str(symbol or "").replace("/USDT:USDT", "USDT").replace("/", "").replace(":USDT", "").upper()


def find_untracked_positions(binance_positions: Iterable[Dict], open_rows: Iterable[Tuple[str, str]]) -> List[Dict]:
    """Binance positions with no matching OPEN live DB row. open_rows = (pair, direction) of every OPEN live order.
    One-way position mode (the bot's assumption) → one position per symbol: no row on the pair = NO_ROW; rows on the pair but none
    on that side = SIDE_MISMATCH. → [{pair, side, qty, entry_price, reason}] (zero-size entries ignored)."""
    by_pair: Dict[str, set] = {}
    for pair, direction in open_rows or ():
        by_pair.setdefault(str(pair or "").upper(), set()).add(str(direction or "").upper())
    out = []
    for pos in binance_positions or ():
        try:
            qty = abs(float(pos.get("contracts") or 0))
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            continue
        sym = str(pos.get("symbol") or "").upper()
        if not (sym.endswith(":USDT") or (sym.endswith("USDT") and "/" not in sym and ":" not in sym)):
            continue   # USDT-M only (Multi-Assets is checked OFF); any other settle would normalise to a malformed key
        try:
            _px = float(pos.get("entry_price") or 0)
        except (TypeError, ValueError):
            _px = 0.0
        if _px > 0 and qty * _px < UNTRACKED_MIN_NOTIONAL_USD:
            continue   # dust left by a close — not a position (and cannot block the symbol)
        pair = _pair_of(pos.get("symbol"))
        side = str(pos.get("side") or "").upper()
        sides = by_pair.get(pair)
        if sides and side in sides:
            continue
        try:
            entry = float(pos.get("entry_price") or 0) or None
        except (TypeError, ValueError):
            entry = None
        out.append({"pair": pair, "side": side, "qty": qty, "entry_price": entry, "reason": "SIDE_MISMATCH" if sides else "NO_ROW"})
    return out


def untracked_action(th) -> str:
    """The configured action for a confirmed untracked position: 'flatten' only when set to exactly that, otherwise 'alert'
    (the default — a typo never auto-closes a real position)."""
    v = str(getattr(th, "untracked_position_action", "alert") or "alert").strip().lower()
    return v if v in UNTRACKED_ACTIONS else "alert"


def preflight_failures(*, api_key_present: bool, account_ok: bool, dual_side, multi_assets, untracked, positions_ok: bool,
                       isolated=()) -> List[str]:
    """④ the read-only pre-flight before PAPER → LIVE. Each argument is what the exchange answered (None = could not read).
    → readable reasons, empty = pass."""
    out = []
    if not api_key_present:
        out.append("no Binance API key / secret configured")
    if not account_ok:
        out.append("the futures account could not be read with the configured key")
    if dual_side is None:
        out.append("position mode could not be read")
    elif dual_side:
        out.append("account is in HEDGE (dual-side) position mode — the bot needs ONE-WAY mode")
    if multi_assets is None:
        out.append("Multi-Assets mode could not be read")
    elif multi_assets:
        out.append("Multi-Assets mode is ON — switch it OFF (single-asset USDT margin)")
    if isolated is None:
        out.append("per-symbol margin mode could not be read")
    elif isolated:
        _iso = sorted(isolated)
        out.append(f"{len(_iso)} symbol(s) set to ISOLATED margin ({', '.join(_iso[:8])}{' …' if len(_iso) > 8 else ''}) — the bot runs CROSS; "
                   f"switch them to CROSS")
    if not positions_ok:
        out.append("open Binance positions could not be read")
    for u in untracked or ():
        out.append(f"untracked Binance position {u.get('pair')} {u.get('side')} qty {u.get('qty')} ({u.get('reason')}) — close it or import it first")
    return out


def bot_stop_errors(th) -> List[str]:
    """⑤ config validation: frenzy_stop_pct and frenzy_willy_stop_pct must be > 0 (a zero / negative / blank stop = a sleeve with no bot
    stop; live would then run on the exchange backstop alone, paper on nothing). → readable errors, empty = valid."""
    out = []
    for field, _safe in BOT_STOP_FIELDS:
        v = getattr(th, field, None)
        try:
            ok = v is not None and math.isfinite(float(v)) and float(v) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(f"{field} must be > 0 (got {v!r}) — every FRENZY sleeve needs its own stop")
    return out


def sanitize_bot_stops(th) -> List[Tuple[str, object, float]]:
    """⑤ engine side: a loaded config whose FRENZY / WILLY stop is ≤ 0 / blank / unreadable gets the safe default (3.0 / 4.0) IN PLACE.
    → [(field, bad value, value used)] for the caller to log CRITICAL; empty = nothing changed."""
    fixed = []
    for field, safe in BOT_STOP_FIELDS:
        v = getattr(th, field, None)
        try:
            ok = v is not None and math.isfinite(float(v)) and float(v) > 0
        except (TypeError, ValueError):
            ok = False
        if not ok:
            try:
                setattr(th, field, safe)
            except Exception:
                continue
            fixed.append((field, v, safe))
    return fixed


def backstop_heal_skip_inflight(closing_in_progress, close_initiated_at, now_utc, grace_s=BACKSTOP_CLOSE_GRACE_S) -> bool:
    """🛡 (DECISION_LOG 308) True = the heal sweep must NOT re-place this order's backstop now: the bot's own close is in flight
    (closing_in_progress set within grace_s seconds). Missing / unreadable timestamp or a flag older than grace_s → heal (never leave a
    position without its exchange stop on uncertain state). Never raises."""
    try:
        if not bool(closing_in_progress) or close_initiated_at is None:
            return False
        age = (now_utc - close_initiated_at).total_seconds()
        return -float(grace_s) < age < float(grace_s)          # a small clock skew counts as in flight; a far-future stamp → heal
    except Exception:
        return False
