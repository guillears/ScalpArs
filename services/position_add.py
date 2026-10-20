"""➕ Add to an open position (DECISION_LOG 324, operator 2026-10-20).

PURE helpers (no I/O) for TradingEngine.add_to_position / manual_add_preview. Why the feature exists: the 🖐 pop-up refused a pair that
already had an open row, so the operator added size in the Binance app; one-way mode netted that quantity into the bot's position and the
bot's exits sold it (BTW 2026-10-20, DECISION_LOG 322). The ➕ keeps the WHOLE position inside one tracked row.

Conventions (the same every exit uses):
  * P&L % is NET of fees and measured on the entry notional: ((raw move × qty) − entry fee − taker exit fee at that price) / (entry × qty).
  * A row's fixed stop / target (pattern_fixed_sl_pct / pattern_fixed_tp_pct) is such a P&L % — after an add it applies to the NEW
    average entry and the NEW total entry fee, so the stop / target PRICES move with the add.
Only FIXED-LEVEL exits can take an add: a trailing / peak-based exit (momentum stack, FRENZY, Bull-Run, an un-overridden bot exit) keeps
its trail / peak state in % of the OLD entry, so re-basing the entry would silently move every armed level.
"""
import math
from typing import Optional

ADD_STOP_BUFFER_PCT = 0.15          # refuse an add whose projected P&L sits within this of the fixed stop (it would stop out at once)
ADD_QTY_TOL_SHARE = 0.005           # live pre-check: Binance's size must match the row within max(1 step, 0.5 %)
ADD_MIN_NOTIONAL_USD = 5.0          # Binance USDT-M minimum order notional when the market table is not loaded
ADD_FIXED_MODES = ("FIXED", "FLOOR")
ADD_LEV_TOL = 0.15                  # review I5: Binance's position leverage vs the row's — a wider gap refuses (the order would re-lever it)
ADD_RECHECK_QTY_TOL = 0.10          # review I1: a late growth within 10 % of the add's requested qty is the add (else: not ours → alert only)
ADD_REFUSE_TRAILING = "switch this trade to a fixed stop/target first (✏️ Close → Set exit), then add"


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def add_exit_refusal(entry_strategy, manual_exit_mode, exit_override_at) -> Optional[str]:
    """None = the row carries a FIXED-LEVEL exit (an add may re-base it) · else the readable refusal. MANUAL rows in FIXED / FLOOR
    (a NULL mode = FIXED, the exits' own default) and ANY row with an operator exit override qualify; MOMENTUM / FRENZY / BULLRUN /
    BULLRUN_SL manual rows and un-overridden bot rows are refused (their peak / trail state is in % of the old entry)."""
    if exit_override_at is not None:
        return None
    if str(entry_strategy or "").upper() == "MANUAL" and str(manual_exit_mode or "FIXED").upper() in ADD_FIXED_MODES:
        return None
    return ADD_REFUSE_TRAILING


def pnl_pct_net(direction, entry, qty, entry_fee, price, taker_fee) -> Optional[float]:
    """The P&L % every exit compares against its levels (net of the entry fee and the taker exit fee at `price`). None on bad input."""
    e, q, p, ef, tf = _f(entry), _f(qty), _f(price), _f(entry_fee) or 0.0, _f(taker_fee) or 0.0
    if not (e and e > 0 and q and q > 0 and p and p > 0):
        return None
    raw = (p - e) * q if str(direction).upper() == "LONG" else (e - p) * q
    return (raw - ef - p * q * tf) / (e * q) * 100.0


def price_at_pnl(direction, entry, qty, entry_fee, pnl_pct, taker_fee) -> Optional[float]:
    """The price at which pnl_pct_net == pnl_pct (the stop / target PRICE of a fixed level). Inverse of pnl_pct_net. None on bad input."""
    e, q, s, ef, tf = _f(entry), _f(qty), _f(pnl_pct), _f(entry_fee) or 0.0, _f(taker_fee) or 0.0
    if not (e and e > 0 and q and q > 0) or s is None:
        return None
    s /= 100.0
    if str(direction).upper() == "LONG":   # (P − E)Q − F − PQf = sEQ  →  P = (E(1 + s) + F/Q) / (1 − f)
        p = (e * (1 + s) + ef / q) / (1 - tf)
    else:                                   # (E − P)Q − F − PQf = sEQ  →  P = (E(1 − s) − F/Q) / (1 + f)
        p = (e * (1 - s) - ef / q) / (1 + tf)
    return p if p > 0 else None


def add_fill_math(qty, entry, investment, entry_fee, leverage, fill_qty, fill_price, taker_fee) -> dict:
    """The row after an add of fill_qty @ fill_price → dict(quantity, entry_price (qty-weighted average), investment (+ fill notional /
    leverage), notional_value (new qty × new entry), entry_fee (+ fill notional × taker), add_qty, add_price, add_notional, add_margin,
    add_fee). Raises ValueError on a non-positive / non-finite input (the caller never books a zero entry)."""
    q, e, inv, ef, lev = _f(qty), _f(entry), _f(investment), _f(entry_fee) or 0.0, _f(leverage)
    fq, fp, tf = _f(fill_qty), _f(fill_price), _f(taker_fee) or 0.0
    if not (q and q > 0 and e and e > 0 and lev and lev > 0 and fq and fq > 0 and fp and fp > 0) or inv is None:
        raise ValueError("add math: quantities, prices and leverage must be positive numbers")
    nq = q + fq
    ne = (q * e + fq * fp) / nq
    add_notional = fq * fp
    add_fee = add_notional * tf
    return dict(quantity=nq, entry_price=ne, investment=inv + add_notional / lev, notional_value=nq * ne, entry_fee=ef + add_fee,
                add_qty=fq, add_price=fp, add_notional=add_notional, add_margin=add_notional / lev, add_fee=add_fee)


def rederive_peak_trough(direction, entry, qty, entry_fee, high_since, low_since, pnl_now, taker_fee):
    """(peak_pnl, trough_pnl) against the NEW entry: the best / worst price seen since the first entry (high / low_price_since_entry —
    LONG peak at the high, SHORT at the low) re-read as P&L % of the new average, clamped so peak ≥ now ≥ trough. A missing high / low →
    both = the P&L now. Fixed-level exits never read these for the exit itself (display + research only)."""
    now = _f(pnl_now)
    if now is None:
        now = 0.0
    hi, lo = _f(high_since), _f(low_since)
    long_ = str(direction).upper() == "LONG"
    best_px, worst_px = (hi, lo) if long_ else (lo, hi)
    if not (best_px and best_px > 0 and worst_px and worst_px > 0):
        return now, now
    pk = pnl_pct_net(direction, entry, qty, entry_fee, best_px, taker_fee)
    tr = pnl_pct_net(direction, entry, qty, entry_fee, worst_px, taker_fee)
    if pk is None or tr is None:
        return now, now
    return max(pk, now), min(tr, now)


def bracket_max_add_margin(cap_notional, leverage, cur_qty, price) -> Optional[float]:
    """The largest add MARGIN the exchange's tier cap (already WITH the 5 % headroom) allows at this leverage: (cap − current qty ×
    price) / leverage, floored at 0. None = no cap known (table missing / switch off)."""
    cap, lev, q, p = _f(cap_notional), _f(leverage), _f(cur_qty), _f(price)
    if cap is None or not (lev and lev > 0) or q is None or not (p and p > 0):
        return None
    return max(0.0, (cap - q * p) / lev)


def qty_matches(exchange_qty, row_qty, step=0.0) -> bool:
    """Live pre-check: does Binance's position size match the row within max(1 step, 0.5 % of the row)?"""
    x, r, s = _f(exchange_qty), _f(row_qty), _f(step) or 0.0
    if x is None or r is None or not r > 0:
        return False
    return abs(x - r) <= max(s, ADD_QTY_TOL_SHARE * r) + 1e-12


def add_preview_math(direction, qty, entry, investment, entry_fee, leverage, add_investment, price, taker_fee, sl_pct=None, tp_pct=None) -> dict:
    """What an add of add_investment margin at `price` makes of the row (a market fill AT the price, taker fee): new qty / avg entry /
    invested / notional / entry fee, the P&L % at the price on the new average, the stop / target PRICES on the new average and the loss
    at the stop in USD (positive = a loss; ≤ 0 = the stop locks a profit). Raises ValueError on bad input."""
    add, lev, p = _f(add_investment), _f(leverage), _f(price)
    if not (add and add > 0 and lev and lev > 0 and p and p > 0):
        raise ValueError("add math: size, leverage and price must be positive numbers")
    m = add_fill_math(qty, entry, investment, entry_fee, leverage, add * lev / p, p, taker_fee)
    sl, tp = _f(sl_pct), _f(tp_pct)
    out = dict(m)
    out['pnl_now'] = pnl_pct_net(direction, m['entry_price'], m['quantity'], m['entry_fee'], p, taker_fee)
    out['stop_price'] = price_at_pnl(direction, m['entry_price'], m['quantity'], m['entry_fee'], sl, taker_fee) if sl is not None else None
    out['tp_price'] = price_at_pnl(direction, m['entry_price'], m['quantity'], m['entry_fee'], tp, taker_fee) if (tp is not None and tp > 0) else None
    out['risk_usd_at_stop'] = (-sl / 100.0 * m['notional_value']) if sl is not None else None
    return out


def add_level_refusal(pnl_now, sl_pct, tp_pct, buffer=ADD_STOP_BUFFER_PCT) -> Optional[str]:
    """None = the projected P&L on the new average sits between the fixed stop (+ buffer) and the target · else the refusal."""
    now, sl, tp = _f(pnl_now), _f(sl_pct), _f(tp_pct)
    if now is None:
        return "the projected P&L after the add could not be computed — try again in a moment"
    if sl is not None and now <= sl + buffer:
        return (f"after the add the trade would sit at {now:+.2f}% on the new average entry — at or within {buffer:.2f} of its stop "
                f"({sl:+.2f}%), it would stop out at once. Add less, or move the stop first")
    if tp is not None and tp > 0 and now >= tp:
        return (f"after the add the trade would sit at {now:+.2f}% on the new average entry — already at its target ({tp:+.2f}%), "
                f"it would take profit at once. Raise the target first")
    return None


def add_fill_price_from_positions(qty_before, entry_before, qty_after, entry_after) -> Optional[float]:
    """The add's own fill price from Binance's position averages before / after it: (E_after·Q_after − E_before·Q_before) / (Q_after − Q_before).
    None when either side is unreadable or the result is not a positive price (the caller then falls back to the click price)."""
    qb, eb, qa, ea = _f(qty_before), _f(entry_before), _f(qty_after), _f(entry_after)
    if not (qb is not None and qb >= 0 and eb and eb > 0 and qa and ea and ea > 0) or not qa > qb:
        return None
    p = (ea * qa - eb * qb) / (qa - qb)
    return p if (math.isfinite(p) and p > 0) else None


def binance_position_leverage(pos) -> Optional[float]:
    """Review I5: the leverage Binance holds the position at, from a get_open_positions row. The reported 'leverage' is trusted unless it is
    the service's 1 default (the newer positionRisk read can omit it); then notional / initial margin; neither → None (cannot judge)."""
    try:
        rep = _f((pos or {}).get('leverage'))
        n, m = _f((pos or {}).get('notional')), _f((pos or {}).get('margin'))
    except AttributeError:
        return None
    if rep is not None and rep > 1:
        return rep
    if n and m and m > 0 and abs(n) > 0:
        return abs(n) / m
    return None
