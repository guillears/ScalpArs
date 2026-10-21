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
IMPORT_SNAPSHOT_STALE_S = 90.0             # (343 banner fix) the banner warns only when the reconciler's positions snapshot is older than this
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


# ── 📥 (DECISION_LOG 343) the app's own TP / SL orders on the symbol: classified, adopted (TP) or cancelled ──────────────────────────
APP_TP_TYPES = ("TAKE_PROFIT_MARKET", "TAKE_PROFIT")
APP_SL_TYPES = ("STOP_MARKET", "STOP")
APP_TP_QTY_TOL_SHARE = 0.005        # an app TP whose quantity is below the position by more than this share covers it only partly
APP_TP_PRICE_TOL_SHARE = 1e-6       # the TP the pop-up showed vs the one read at the click: a different price = the app TP moved → refused


def _order_label(o) -> str:
    lvl = _f(o.get('trigger_price')) or _f(o.get('price')) or 0.0
    q = "close-all" if o.get('close_position') else f"qty {(_f(o.get('qty')) or 0):g}"
    return f"{o.get('type') or '?'} {o.get('side') or '?'} @ {lvl:g} ({q}, {o.get('kind')} #{o.get('id')})"


def classify_app_orders(orders, direction, entry, qty, bot_leftover_ids=(), leftover_suspect=False) -> dict:
    """Sort the open orders of the symbol (binance_service.list_symbol_open_orders) for an import of a `direction` position: TP = a
    reduce-only / close-all order on the closing side whose level sits on the PROFIT side of the entry — TAKE_PROFIT(_MARKET) (its trigger)
    or a LIMIT (its price) · SL = a reduce-only / close-all STOP(_MARKET) on the closing side (a stop above the entry of a LONG — a profit
    lock — is a stop too: the bot cancels it either way) · OTHER = anything else (an entry / add order, the opening side, not reduce-only, a
    TP on the loss side, a trailing stop, an unknown type or an unreadable level) — an OTHER refuses the import (nothing changed) · LEFTOVER
    (review #7) = a regular reduce-only LIMIT on the closing side whose id matches a resting TP recorded on a closed row of the pair
    (bot_leftover_ids): cancelled at the import like the app's orders, never offered / kept as "your TP" (the bot tags no clientOrderId).
    leftover_suspect (the pair is flagged dirty by an unverified close, id unknown): a regular reduce-only order keeps its TP / SL class
    with maybe_leftover=True — shown "could be a bot leftover", still keepable (round 2 #1).
    → {'tp': [...], 'sl': [...], 'other': [...]} with each order carrying 'cls', 'level' (its price) and 'label', plus 'tp_price' / 'tp_qty'
    / 'tp_partial' of the ONE TP (None when there is not exactly one) and 'refusal' (str or None). Pure, never raises."""
    d = str(direction or "").upper()
    close_side = "SELL" if d == "LONG" else "BUY"
    e, q = _f(entry), _f(qty)
    out = {'tp': [], 'sl': [], 'other': [], 'leftover': [], 'tp_price': None, 'tp_qty': None, 'tp_partial': False, 'refusal': None}
    _left_ids = {str(i) for i in (bot_leftover_ids or ()) if i}
    for raw in (orders or ()):
        o = dict(raw or {})
        typ = str(o.get('type') or '').upper()
        reduce = bool(o.get('reduce_only')) or bool(o.get('close_position'))
        level = _f(o.get('price')) if typ == "LIMIT" else _f(o.get('trigger_price'))
        o['level'] = level
        o['label'] = _order_label(o)
        cls, why = "other", None
        if str(o.get('side') or '').upper() != close_side:
            why = "on the opening side (an entry / add order)"
        elif not reduce:
            why = "not reduce-only (it could open a position)"
        elif not (level and level > 0):
            why = "its price could not be read"
        elif typ == "LIMIT" and o.get('kind') == 'regular' and str(o.get('id')) in _left_ids:
            cls = "leftover"
        elif typ in APP_TP_TYPES or typ == "LIMIT":
            if e and e > 0 and ((d == "LONG" and level > e) or (d == "SHORT" and level < e)):
                cls = "tp"
            else:
                why = "a take-profit / limit on the loss side of the entry"
        elif typ in APP_SL_TYPES:
            cls = "sl"
        else:
            why = f"type {typ or '?'} is not a plain TP / SL"
        o['cls'] = cls
        o['maybe_leftover'] = bool(leftover_suspect and cls in ("tp", "sl") and o.get('kind') == 'regular')
        if why:
            o['why'] = why
        out[cls].append(o)
    if len(out['tp']) == 1:
        t = out['tp'][0]
        out['tp_price'] = t['level']
        tq = q if t.get('close_position') else _f(t.get('qty'))
        out['tp_qty'] = tq
        out['tp_partial'] = bool(q and tq is not None and tq < q * (1 - APP_TP_QTY_TOL_SHARE))
    if out['other']:
        out['refusal'] = (f"{len(out['other'])} open order(s) on the symbol the bot cannot take over: "
                          + "; ".join(f"{o['label']} — {o.get('why')}" for o in out['other'])
                          + " — cancel them in the Binance app first (nothing changed)")
    return out


def import_tp_pct_from_price(direction, entry, tp_price, entry_fee_pct, maker_fee_rate) -> Optional[float]:
    """The NET take profit % (the manual TP's unit: P&L net of the entry fee and a MAKER exit fee, services.exec_shadow.net_pct_at) that
    books at `tp_price` — the inverse of the resting TP's limit (resting_tp_price). Rounded UP to 4 decimals (review #5), so the bot's limit
    is never WORSE than the app's price: a LONG's limit never below it, a SHORT's never above it (a 4-dp cut on a high-priced pair with a
    tiny tick would land several ticks short). Beyond it by at most 0.0001 % of price + 1 tick. None when the price books no profit after
    fees or input is bad."""
    e, x, ef, mk = _f(entry), _f(tp_price), _f(entry_fee_pct), _f(maker_fee_rate)
    if not (e and e > 0 and x and x > 0) or ef is None or mk is None:
        return None
    # services.exec_shadow.net_pct_at's arithmetic, UNROUNDED (its 4-dp round can round UP and push the re-created limit a tick past)
    move = (x / e - 1) * 100.0 if str(direction or "").upper() != "SHORT" else (1 - x / e) * 100.0
    n = move - ef - mk * 100.0 * (x / e)
    t = math.ceil(float(n) * 1e4 - 1e-6) / 1e4
    return t if t > 0 else None


def describe_open_orders(orders) -> str:
    """A one-line account of what rests on the symbol — the exact-state text of every post-cancel failure message."""
    if orders is None:
        return "the open orders could not be read"
    if not orders:
        return "no open orders"
    return "; ".join(_order_label(o) for o in orders)
