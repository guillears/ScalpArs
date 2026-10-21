"""🖐 / 📥 "No stop" exit with an optional −15 % EMERGENCY stop, and the ⛔ "block bot trades while this is open" flag (DECISION_LOG 343,
operator 2026-10-21).

PURE helpers (no I/O) shared by TradingEngine.open_manual_position, import_binance_position, the exit paths and the backstop sweeps.

Why: the operator holds some positions with no stop at all (the IOST import: position ≈ 3.6× the account) and closes them only at his
take profit or by hand. The bot's own stop / floor / safety stop would close them on a loss he chose to ride. Two shapes:
  * NOSTOP + emergency (default, 15 %): ONE closePosition STOP_MARKET rests on Binance at entry × (1 − 15 %) for a LONG (× (1 + 15 %) for a
    SHORT) — placed, healed, re-priced and cancelled by the SAME backstop machinery every other row uses (backstop_algo_id / backstop_pct
    / backstop_trigger_price). It is the only stop. Paper (no exchange stop) books it in software at the same trigger price (paper = live);
    live runs a software mirror MANUAL_NOSTOP_SOFT_MARGIN_PCT beyond the trigger, so it only acts when the exchange stop is missing.
  * NOSTOP, emergency unticked: NOTHING on the loss side — no exchange stop, no floor, no max hold. Loss unlimited until liquidation.
The emergency % is a PRICE move (the backstop's unit: the trigger is entry × (1 ∓ pct/100)), not the net-of-fees P&L % of the SL fields:
at 15 % the net loss is ≈ 15 % + the round-trip fee of the position.
"""
import math
from typing import Optional

MANUAL_NOSTOP_MODE = "NOSTOP"
EMERGENCY_STOP_DEFAULT_PCT = 15.0      # the pop-ups' default (config: thresholds.manual_emergency_stop_pct)
EMERGENCY_STOP_MAX_PCT = 95.0          # a LONG trigger at entry × (1 − ≥100 %) would be ≤ 0 — refused well before that
MANUAL_NOSTOP_SOFT_MARGIN_PCT = 0.3    # live: the software mirror sits this far (price %) beyond the exchange trigger → acts only without it
MANUAL_BLOCK_FILTER = "MANUAL_BLOCK"   # the _record_filter_block counter of a bot open refused by ⛔ (D11)
MANUAL_BLOCK_READ_FAILED_FILTER = "MANUAL_BLOCK_READ_FAILED"   # … refused because the ⛔ flag could not be read (fail-closed, review #4)


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def is_nostop(entry_strategy, exit_mode) -> bool:
    """A 🖐 / 📥 MANUAL row on the No-stop exit."""
    return str(entry_strategy or "") == "MANUAL" and str(exit_mode or "").upper() == MANUAL_NOSTOP_MODE


def manual_emergency_pct(th, exit_mode, emergency_on, emergency_pct) -> Optional[float]:
    """The emergency stop distance (price %, positive) a new row stores in manual_emergency_stop_pct, or None (no emergency stop). Only the
    NOSTOP mode carries one; emergency_on False → None (a true no stop); a blank % → th.manual_emergency_stop_pct (default 15). Raises
    ValueError (readable) on a % that is not a number, ≤ 0 or ≥ EMERGENCY_STOP_MAX_PCT."""
    if str(exit_mode or "").upper() != MANUAL_NOSTOP_MODE or not emergency_on:
        return None
    raw = emergency_pct
    if raw in (None, ""):
        raw = getattr(th, 'manual_emergency_stop_pct', EMERGENCY_STOP_DEFAULT_PCT)
    v = _f(raw)
    if v is None:
        raise ValueError("the emergency stop % must be a number")
    v = abs(v)
    if not (0 < v < EMERGENCY_STOP_MAX_PCT):
        raise ValueError(f"the emergency stop must be between 0 and {EMERGENCY_STOP_MAX_PCT:g} % of price (untick it for no stop at all)")
    return round(v, 4)


def emergency_trigger_price(entry, direction, pct) -> Optional[float]:
    """entry × (1 − pct %) for a LONG, × (1 + pct %) for a SHORT (the backstop formula). None on bad input."""
    e, p = _f(entry), _f(pct)
    if not (e and e > 0 and p and p > 0):
        return None
    return e * (1 - p / 100.0) if str(direction or "").upper() == "LONG" else e * (1 + p / 100.0)


def emergency_liq_refusal(pair, direction, trigger, liq_price, mirror_trigger=None) -> Optional[str]:
    """Why an emergency stop at `trigger` must be refused against the position's liquidation price, or None. A stop at or beyond liquidation
    never fires (the position is liquidated first). liq_price None = Binance did not report one → refused (never guessed). 0 on a LONG =
    no liquidation price (the account covers a fall to zero) → fine; 0 / None on a SHORT → refused. mirror_trigger (live): the software
    mirror's line (MANUAL_NOSTOP_SOFT_MARGIN_PCT further) must sit inside liquidation too (review #8)."""
    if mirror_trigger is not None:
        _r = emergency_liq_refusal(pair, direction, mirror_trigger, liq_price)
        if _r is not None and _f(liq_price) is not None and emergency_liq_refusal(pair, direction, trigger, liq_price) is None:
            return (f"the emergency stop's software back-up line ({float(mirror_trigger):.6g}, {MANUAL_NOSTOP_SOFT_MARGIN_PCT:g} % beyond it) "
                    f"would sit at or beyond the liquidation price {float(liq_price):.6g} — use a smaller emergency % (nothing changed)")
    t, liq = _f(trigger), _f(liq_price)
    d = str(direction or "").upper()
    if t is None or t <= 0:
        return f"the emergency stop price for {pair} cannot be computed"
    if liq is None or (liq <= 0 and d != "LONG"):
        return (f"Binance did not report the liquidation price of the {pair} position — the emergency stop cannot be checked against it; "
                f"nothing changed. Try again in a moment, or untick the emergency stop")
    if d == "LONG" and liq > 0 and t <= liq:
        return (f"the emergency stop at {t:.6g} is at or below the liquidation price {liq:.6g} — it would never fire; use a smaller emergency % "
                f"(nothing changed)")
    if d == "SHORT" and t >= liq:
        return (f"the emergency stop at {t:.6g} is at or above the liquidation price {liq:.6g} — it would never fire; use a smaller emergency % "
                f"(nothing changed)")
    return None


def emergency_clamp_to_liq(entry, direction, pct, liq_price, liq_share) -> Optional[float]:
    """🖐 manual ENTRY (the order already filled — it can no longer be refused): the emergency % moved inside liquidation when the requested
    one — or its live software mirror, MANUAL_NOSTOP_SOFT_MARGIN_PCT further (review #8) — is at / beyond it → liq_share × the
    entry→liquidation distance minus the mirror margin (so BOTH lines sit inside; the same share order_backstop_pct keeps). None = keep
    `pct` (fine, or the liquidation price is unknown / 0 on a LONG)."""
    e, p, liq = _f(entry), _f(pct), _f(liq_price)
    if not (e and e > 0 and p and p > 0) or liq is None or liq <= 0:
        return None
    if emergency_liq_refusal("x", direction, emergency_trigger_price(e, direction, p + MANUAL_NOSTOP_SOFT_MARGIN_PCT), liq) is None:
        return None
    dist = abs(e - liq) / e * 100.0 * float(liq_share) - MANUAL_NOSTOP_SOFT_MARGIN_PCT
    return round(dist, 4) if dist > 0 else None


def emergency_software_hit(direction, entry, price, pct, is_paper) -> bool:
    """The row's emergency stop has been crossed by `price`: paper at the exchange trigger itself (paper books what live's exchange stop
    would), live MANUAL_NOSTOP_SOFT_MARGIN_PCT beyond it (the software mirror acts only when the exchange stop is missing). False when the
    row has no emergency stop or any input is bad."""
    p = _f(pct)
    if not (p and p > 0):
        return False
    line = emergency_trigger_price(entry, direction, p if is_paper else p + MANUAL_NOSTOP_SOFT_MARGIN_PCT)
    px = _f(price)
    if line is None or not (px and px > 0):
        return False
    return px <= line if str(direction or "").upper() == "LONG" else px >= line
