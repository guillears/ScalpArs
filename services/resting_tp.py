"""🎯 RESTING TAKE-PROFIT (DECISION_LOG 296) — pure rules, no I/O. The engine wiring lives in services/trading_engine.py, the exchange calls
in services/binance_service.py, the price math in services/exec_shadow.py (resting_tp_price / net_pct_at / cached_tick — shared with the
observe-only 295 shadow, so the shadow and the real order price the SAME limit).

Why: live FRENZY_WILLY take profits (+1.25 net) filled +1.08…+1.12 (3 of 3) and LITE (+3) +2.80 — the software TP fires on the touch and the
market order meets a book that already gave way. A reduce-only, post-only (GTX) LIMIT resting on Binance at the price that books the SAME net
target with the MAKER exit fee books the target instead.

Mechanism (reports/MAKER_TP_ENTRY_DESIGN_2026-10-15.md §3):
  * placed after the fill + backstop (live), recorded on the row: protective_tp_order_id · resting_tp_price · resting_tp_filled_qty
  * the software TP is the SAFETY NET while a resting TP is alive: it fires only once the price has been ≥ resting_tp_through_ticks ticks
    THROUGH the limit for ≥ resting_tp_grace_s → cancel the limit, market-close ONLY the rest (reduce-only), book a blended exit
  * every other exit cancels the limit FIRST, re-reads its fill, closes only the remainder and books blended (one Order row)
  * paper = live bookkeeping: the paper TP books at the limit with the maker fee only when the tracked price trades ≥ 1 tick THROUGH it
Switches per sleeve (resting_tp_willy_enabled / resting_tp_lite_enabled / resting_tp_frenzy_long_enabled / resting_tp_lite_retry_enabled); OFF → paper and live behave
exactly as before this ship.
🎯 DECISION_LOG 310 — every FIXED take-profit line: the operator's MANUAL trades in FIXED or FLOOR mode with a TP > 0 (resting_tp_manual_enabled,
reason MANUAL_TP — incl. a mid-trade edit to FIXED), any bot trade whose exit the operator overrode with a TP > 0 (resting_tp_override_enabled,
reason OVERRIDE_TP), and FRENZY_LONG (resting_tp_frenzy_long_enabled, FRENZY_TP; none while frenzy_lock_arm_pct > 0). The target is derived
from the ROW alone (rtp_row_target: entry_strategy · manual_exit_mode · pattern_fixed_tp_pct · exit_override_at) — restart-proof.
Evidence: AGT MANUAL FIXED +2 (2026-10-19) software TP on the touch, market sell 0.61 % lower → +1.38 booked.
Every helper is total (never raises on bad input)."""
import math
from typing import Dict, Optional, Tuple

RTP_SLEEVES = {"FRENZY_WILLY": "resting_tp_willy_enabled", "FRENZY_LITE": "resting_tp_lite_enabled",
               "FRENZY_LONG": "resting_tp_frenzy_long_enabled",   # 🎯 310: + FRENZY_LONG (fixed frenzy_tp_pct) · never FRENZY_WIDE (off)
               "FRENZY_LITE_RETRY": "resting_tp_lite_retry_enabled",   # 🔁 316: LITE's resting TP, own switch (attribution)
               "FRENZY_WILLY_HT": "resting_tp_willy_enabled"}   # 🎲¼ 341: the quarter-size WILLY runs WILLY's exit — WILLY's switch
RTP_MANUAL_KEY, RTP_OVERRIDE_KEY = "resting_tp_manual_enabled", "resting_tp_override_enabled"   # 🎯 (310) the operator's own fixed TPs
RTP_SWITCHES = tuple(dict.fromkeys(RTP_SLEEVES.values())) + (RTP_MANUAL_KEY, RTP_OVERRIDE_KEY)   # unique (341: HT shares WILLY's switch)
RTP_ROW_STRATEGIES = tuple(RTP_SLEEVES) + ("MANUAL",)   # the sweep's row filter (+ every row with an exit override)
RTP_QUERY_MIN_S = 1.0          # at most one order query per order per second (fill detection, only while the price is at / through)
RTP_HEAL_RETRY_S = 30.0        # a failed placement retries per order no sooner than this
RTP_SWEEP_EVERY_S = 5.0        # heal / switch-off / re-price sweep cadence
RTP_SWEEP_MAX_ACTIONS = 3      # live exchange actions per sweep pass (never stalls the 1 Hz monitor)
RTP_ORPHAN_MIN_AGE_S = 60.0    # a reduce-only order younger than this is never judged an orphan (its row may still be committing)
RTP_GRACE_DEF, RTP_THROUGH_DEF = 3.0, 2
RTP_IO_TIMEOUT_S = 3.0         # every exchange call under the close lock / in the gate is bounded; a timeout = UNKNOWN (never filled / canceled)
RTP_STRAY_MAX_PER_PASS = 5      # stray reduce-only limits cancelled per arm / heal pass (bounded close-lock hold)
RTP_BUSY_WATCHDOG_S = 6.0      # an in-flight flag older than 2 × RTP_IO_TIMEOUT_S is cleared (a lost background task)
RTP_STOP_CLASS = ("STOP_LOSS", "MAX_HOLD_TIME", "BACKSTOP", "MANUAL_SL", "OVERRIDE_SL", "FL_EMERGENCY_SL", "FL_DEEP_STOP")   # + the operator's "MANUAL" close   # never wait on a blocked cancel
RTP_ORDER_TYPE_MAKER, RTP_ORDER_TYPE_BLEND = "MAKER", "MAKER+TAKER"
_QTY_EPS = 1e-9


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def rtp_enabled(th, entry_strategy) -> bool:
    key = RTP_SLEEVES.get(str(entry_strategy or ""))
    return bool(key and getattr(th, key, False) is True)


def _on(th, key) -> bool:
    return getattr(th, key, False) is True


def rtp_fixed_reason(row) -> Optional[str]:
    """🎯 (310) The close reason of an operator-owned fixed TP on this row, from the row alone: MANUAL → MANUAL_TP · any other row with an
    exit override → OVERRIDE_TP (the label both exit paths book) · None = a sleeve TP (FRENZY_TP / WILLY_LATE_TP). Never raises."""
    try:
        if str(getattr(row, 'entry_strategy', None) or "") == "MANUAL":
            return "MANUAL_TP"
        return "OVERRIDE_TP" if getattr(row, 'exit_override_at', None) is not None else None
    except Exception:
        return None


def rtp_row_enabled(th, row) -> bool:
    """The switch that governs THIS row's resting TP is ON (MANUAL → resting_tp_manual_enabled · an overridden bot row →
    resting_tp_override_enabled · else the sleeve's own switch)."""
    why = rtp_fixed_reason(row)
    if why == "MANUAL_TP":
        return _on(th, RTP_MANUAL_KEY)
    if why == "OVERRIDE_TP":
        return _on(th, RTP_OVERRIDE_KEY)
    return rtp_enabled(th, getattr(row, 'entry_strategy', None))


def rtp_row_target(th, row, held_minutes=None) -> Optional[Tuple[float, str]]:
    """🎯 (310) The resting TP this ROW should carry now → (NET tp %, close reason) or None (= withdraw / none). Restart-proof: read from the
    row only. MANUAL: manual_exit_mode FIXED (None = FIXED, the exits' own default) or FLOOR (its optional TP) with pattern_fixed_tp_pct > 0
    → MANUAL_TP; MOMENTUM / FRENZY / BULLRUN(_SL) or TP ≤ 0 / None → None. An exit override on a bot row: its pattern_fixed_tp_pct > 0 → OVERRIDE_TP,
    no TP → None (the sleeve's own TP is replaced, never kept). Otherwise the sleeve rule (rtp_target, incl. the LITE C stored TP). Never raises."""
    try:
        why = rtp_fixed_reason(row)
        if why is None:
            return rtp_target(th, getattr(row, 'entry_strategy', None), held_minutes, getattr(row, 'frenzy_lite_tp_used', None))
        if not rtp_row_enabled(th, row):
            return None
        if why == "MANUAL_TP" and str(getattr(row, 'manual_exit_mode', None) or "FIXED").upper() not in ("FIXED", "FLOOR"):   # FLOOR's optional TP is a fixed line too (Sep-30; AGT #22 10-19)
            return None
        tp = _f(getattr(row, 'pattern_fixed_tp_pct', None))
        return (tp, why) if tp and tp > 0 else None
    except Exception:
        return None


def rtp_row_reason(th, row, limit_px, maker_fee_rate, entry_fee_pct=None) -> str:
    """Which take profit a RECORDED limit on this row books: the operator's (MANUAL_TP / OVERRIDE_TP) or the sleeve's (rtp_reason_for_limit)."""
    why = rtp_fixed_reason(row)
    if why:
        return why
    return rtp_reason_for_limit(th, getattr(row, 'entry_strategy', None), getattr(row, 'direction', None), getattr(row, 'entry_price', None),
                                entry_fee_pct, limit_px, maker_fee_rate)


def rtp_target(th, entry_strategy, held_minutes=None, lite_tp=None) -> Optional[Tuple[float, str]]:
    """(NET take profit a resting TP books at this order age, the close reason it books) or None = no resting TP for this order (switch off,
    not a resting-TP sleeve, TP off, or FRENZY_LITE while frenzy_lock_arm_pct > 0 — lock-then-trail replaces the fixed TP). FRENZY_WILLY
    follows its age: the WILLY TP (FRENZY_TP) before frenzy_willy_late_tp_minutes, the late TP (WILLY_LATE_TP, DECISION_LOG 297) after.
    lite_tp (🎯 DECISION_LOG 301 LITE C): the order's stored LITE TP (Order.frenzy_lite_tp_used) — FRENZY_LITE / FRENZY_LITE_RETRY only; None → the config TP."""
    es = str(entry_strategy or "")
    if not rtp_enabled(th, es):
        return None
    try:
        from services.frenzy import frenzy_willy_tp_at, frenzy_tp_for
        if es in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):   # 🎲¼ (341) HT = WILLY's exit
            tp, why = frenzy_willy_tp_at(th, held_minutes)
        else:
            if (_f(getattr(th, 'frenzy_lock_arm_pct', 0)) or 0) > 0:
                return None
            tp, why = frenzy_tp_for(th, es, lite_tp), "FRENZY_TP"
        tp = _f(tp)
        return (tp, why) if tp and tp > 0 else None
    except Exception:
        return None


def rtp_tp_pct(th, entry_strategy, held_minutes=None, lite_tp=None) -> Optional[float]:
    t = rtp_target(th, entry_strategy, held_minutes, lite_tp)
    return t[0] if t else None


def rtp_reason_for_limit(th, entry_strategy, direction, entry_px, entry_fee_pct, limit_px, maker_fee_rate) -> str:
    """Which take profit a RECORDED limit books (restart-proof, from the price alone): WILLY_LATE_TP when the WILLY late TP is on and the
    limit's net % sits nearer the late target than the WILLY TP, else FRENZY_TP."""
    try:
        if str(entry_strategy or "") not in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):   # 🎲¼ (341)
            return "FRENZY_TP"
        from services.frenzy import frenzy_willy_late_tp, frenzy_willy_levels
        from services.exec_shadow import net_pct_at
        late = frenzy_willy_late_tp(th)
        if late is None:
            return "FRENZY_TP"
        net = net_pct_at(direction, entry_px, limit_px, entry_fee_pct, maker_fee_rate)
        if net is None:
            return "FRENZY_TP"
        return "WILLY_LATE_TP" if net < (late[1] + frenzy_willy_levels(th)[0]) / 2.0 else "FRENZY_TP"
    except Exception:
        return "FRENZY_TP"


def rtp_grace_s(th) -> float:
    g = _f(getattr(th, 'resting_tp_grace_s', RTP_GRACE_DEF))
    return g if g is not None and g >= 0 else RTP_GRACE_DEF


def rtp_through_ticks(th) -> int:
    n = _f(getattr(th, 'resting_tp_through_ticks', RTP_THROUGH_DEF))
    return int(round(n)) if n is not None and n >= 1 else RTP_THROUGH_DEF


def config_errors(th) -> list:
    """PUT /api/config refusals for the resting-TP fields."""
    errs = []
    for k in RTP_SWITCHES:
        if not isinstance(getattr(th, k, False), bool):
            errs.append(f"{k} must be true or false")
    g = _f(getattr(th, 'resting_tp_grace_s', None))
    if g is None or g < 0 or g > 60:
        errs.append("resting_tp_grace_s must be a number from 0 to 60 seconds")
    n = getattr(th, 'resting_tp_through_ticks', None)
    nf = _f(n)
    if nf is None or nf != int(nf) or not (1 <= nf <= 50):
        errs.append("resting_tp_through_ticks must be a whole number from 1 to 50")
    # ⏰ (DECISION_LOG 297) WILLY late take profit
    lm = _f(getattr(th, 'frenzy_willy_late_tp_minutes', 0))
    lp = _f(getattr(th, 'frenzy_willy_late_tp_pct', 0))
    if lm is None or lm < 0:
        errs.append("frenzy_willy_late_tp_minutes must be a number ≥ 0 (0 = off)")
    if lp is None or lp < 0:
        errs.append("frenzy_willy_late_tp_pct must be a number ≥ 0 (0 = off)")
    elif lp > 0:
        try:
            from services.frenzy import frenzy_willy_levels
            wtp = frenzy_willy_levels(th)[0]
        except Exception:
            wtp = None
        if wtp is not None and lp >= wtp:
            errs.append(f"frenzy_willy_late_tp_pct must be below the WILLY take profit (+{wtp}%)")
    return errs


def ticks_through(direction, price, limit, tick) -> Optional[float]:
    """How many ticks the price is BEYOND the limit in the take-profit direction (LONG: above, SHORT: below); ≤ 0 = not through. None on bad input."""
    p, l, t = _f(price), _f(limit), _f(tick)
    if not (p and p > 0 and l and l > 0 and t and t > 0):
        return None
    d = (p - l) if str(direction or "").upper() != "SHORT" else (l - p)
    return d / t


def at_or_through(direction, price, limit) -> bool:
    p, l = _f(price), _f(limit)
    if not (p and p > 0 and l and l > 0):
        return False
    return p >= l - 1e-12 if str(direction or "").upper() != "SHORT" else p <= l + 1e-12


def paper_filled(direction, price, limit, tick) -> bool:
    """Paper fill rule: the tracked price traded ≥ 1 tick THROUGH the limit (a touch is not a fill — queue position)."""
    n = ticks_through(direction, price, limit, tick)
    return n is not None and n >= 1 - 1e-6


def safety_net(first_through, now, direction, price, limit, tick, n_ticks, grace_s) -> Tuple[Optional[float], bool]:
    """The software TP as the safety net → (first-through time to keep, fire). Fires once the price has been ≥ n_ticks THROUGH the limit
    for ≥ grace_s; dropping back below n_ticks resets the clock."""
    n = ticks_through(direction, price, limit, tick)
    if n is None or n < float(n_ticks) - 1e-6:
        return None, False
    ft = first_through if first_through is not None else now
    return ft, (now - ft) >= float(grace_s) - 1e-9


def remaining_qty(qty, filled_qty) -> float:
    """quantity − filled, rounded to 1e-10 (raw float subtraction gives 0.3 − 0.1 = 0.19999… which the exchange's TRUNCATING precision
    turns into 0.1 — half the position left open). The call site still snaps it to the market step."""
    q, f = _f(qty) or 0.0, _f(filled_qty) or 0.0
    return max(0.0, round(q - max(0.0, f), 10))


def is_full(qty, filled_qty) -> bool:
    q, f = _f(qty) or 0.0, _f(filled_qty) or 0.0
    return q > 0 and f >= q * (1 - 1e-6) - _QTY_EPS


def blend_exit(qty, filled_qty, filled_px, rem_px, maker_rate, taker_rate) -> Optional[Dict]:
    """ONE Order row for a resting-TP exit: the filled part at its own (maker) price + fee, the remainder at its market fill + taker fee.
    → {price (quantity-weighted), fee ($), order_type, filled_qty, rem_qty} — calculate_pnl(direction, entry, price, qty, …, exit_fee=fee)
    then books exactly Σ part P&L (raw P&L is linear in price). None on bad input."""
    q, f = _f(qty), _f(filled_qty) or 0.0
    mk, tk = _f(maker_rate), _f(taker_rate)
    if not (q and q > 0) or mk is None or tk is None:
        return None
    f = min(max(0.0, f), q)
    r = round(q - f, 10)
    if f <= _QTY_EPS:
        rp = _f(rem_px)
        return None if not (rp and rp > 0) else {"price": rp, "fee": q * rp * tk, "order_type": "TAKER", "filled_qty": 0.0, "rem_qty": q}
    fp = _f(filled_px)
    if not (fp and fp > 0):
        return None
    if r <= q * 1e-6 + _QTY_EPS:
        return {"price": fp, "fee": q * fp * mk, "order_type": RTP_ORDER_TYPE_MAKER, "filled_qty": q, "rem_qty": 0.0}
    rp = _f(rem_px)
    if not (rp and rp > 0):
        return None
    return {"price": (f * fp + r * rp) / q, "fee": f * fp * mk + r * rp * tk, "order_type": RTP_ORDER_TYPE_BLEND, "filled_qty": f, "rem_qty": r}


def order_fill(info) -> Tuple[Optional[str], float, Optional[float]]:
    """Normalise an order read (binance_service.fetch_resting_tp) → (state, filled qty, average price). state: 'FILLED' · 'PARTIAL' (resting
    with a fill) · 'OPEN' · 'GONE' (canceled / expired / rejected — with whatever filled before) · None = unreadable."""
    if not isinstance(info, dict):
        return None, 0.0, None
    st = str(info.get('status') or '').upper()
    filled = _f(info.get('filled')) or 0.0
    amt = _f(info.get('amount')) or 0.0
    avg = _f(info.get('average')) or _f(info.get('price'))
    avg = avg if avg and avg > 0 else None
    if st in ('FILLED', 'CLOSED') or (amt > 0 and filled >= amt * (1 - 1e-6)):
        return 'FILLED', filled if filled > 0 else amt, avg
    if st in ('NEW', 'OPEN', 'PARTIALLY_FILLED'):
        return ('PARTIAL' if filled > _QTY_EPS else 'OPEN'), filled, avg
    if st in ('CANCELED', 'CANCELLED', 'EXPIRED', 'REJECTED', 'EXPIRED_IN_MATCH'):
        return 'GONE', filled, avg
    return None, filled, avg


def cancel_view(c) -> Tuple[str, Optional[Dict]]:
    """binance_service.cancel_resting_tp's result → (result, order view or None). result: 'canceled' (with Binance's own cancel response
    when it came back as a dict — no follow-up read needed) · 'gone' · 'error' (None / a timeout reads as 'error')."""
    if isinstance(c, dict):
        return str(c.get('result') or 'canceled'), c
    if c in ('canceled', 'gone', 'error'):
        return c, None
    return 'error', None


def stop_class(reason) -> bool:
    """A close that must never wait on a blocked resting-TP cancel (stop, time cap, backstop, manual close): the reduce-only market order for
    the unfilled rest goes out anyway; the limit's fill is reconciled afterwards (reduce-only cannot over-close)."""
    r = str(reason or "")
    for p in ("RH_", "FL_", "FLIP_", "BR_"):
        if r.startswith(p) and not r.startswith(RTP_STOP_CLASS):
            r = r[len(p):]
    return r == "MANUAL" or r.startswith(RTP_STOP_CLASS)


def ts_key(x) -> Optional[str]:
    """A comparable open-time key (naive UTC to the second) for a datetime or its string form."""
    if x is None:
        return None
    try:
        if hasattr(x, 'isoformat'):
            x = x.replace(tzinfo=None).isoformat()
        return str(x).replace('T', ' ')[:19]
    except Exception:
        return None
