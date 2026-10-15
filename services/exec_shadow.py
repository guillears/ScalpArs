"""📏 Execution-quality shadow (DECISION_LOG 295) — OBSERVE ONLY, never changes an order.

Why: the 2026-10-15 tick study (reports/TICK_STUDY_MAKER_LOCK_2026-10-15.md, reviewed) found WILLY's 2026 edge breaks even at
≈ 0.11 % entry cost, while the first live entries paid a median 0.22 %; and a resting take-profit only pays if the live order book
gives back what prints cannot show. These pure rules turn every FRENZY-family fill into the two measurements:

ENTRY (at the open, FRENZY / SURGE book path): where the entry cost comes from —
  drift  = the book's far side at the order vs the decision price   (the price moved between the signal and the order)
  spread = the book's ask − bid at the order, % of the mid             (the cost of crossing)
  impact = our fill vs the book's far side                             (our own size walking the book / latency to the fill)
  lag    = seconds from the decision clock to the book read (normal FRENZY: the 5m boundary = the signal bar's close; catch-up: the
           decision time — split by entry_frenzy_catchup when reading; SURGE: None)
  NOTE: drift already contains half the spread — total vs decision ≈ drift + impact; never sum all three.
  All signed so POSITIVE = cost to us (LONG pays up, SHORT sells down).

EXIT (at the close, fixed-TP FRENZY sleeves): what a resting reduce-only post-only limit at the take profit would have done —
  limit price = the price that books the SAME net take profit with a MAKER exit fee, rounded to the tick AWAY from the entry
  filled      = the trade's own extreme since entry went ≥ 1 tick THROUGH the limit (a touch is not a fill — queue position)
  pnl         = the net % at the limit with the maker fee when filled, else the trade's actual net % (it would have run unchanged)
Every helper returns None on unusable input — a shadow must never raise into the open / close paths.
"""
import math
from typing import Dict, Optional, Tuple


def _f(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def entry_book_metrics(direction, decision_px, bid, ask, fill_px, lag_s) -> Dict[str, Optional[float]]:
    """→ {entry_book_drift_pct, entry_book_spread_pct, entry_fill_vs_book_pct, entry_order_lag_s} (each None when not computable)."""
    long_ = str(direction or "").upper() != "SHORT"
    d, b, a, f, lag = _f(decision_px), _f(bid), _f(ask), _f(fill_px), _f(lag_s)
    out = {"entry_book_drift_pct": None, "entry_book_spread_pct": None, "entry_fill_vs_book_pct": None,
           "entry_order_lag_s": (round(lag, 3) if lag is not None and lag >= 0 else None)}
    far = a if long_ else b
    if d and d > 0 and far and far > 0:
        out["entry_book_drift_pct"] = round(((far / d - 1) if long_ else (1 - far / d)) * 100, 4)
    if a and b and a > 0 and b > 0 and a >= b:
        out["entry_book_spread_pct"] = round((a - b) / ((a + b) / 2) * 100, 4)
    if far and far > 0 and f and f > 0:
        out["entry_fill_vs_book_pct"] = round(((f / far - 1) if long_ else (1 - f / far)) * 100, 4)
    return out


def resting_tp_price(direction, entry_px, entry_fee_pct, tp_pct, maker_fee_rate, tick) -> Optional[float]:
    """The limit that books net +tp_pct after the entry fee (% of notional) and a MAKER exit fee, rounded to the tick away from the entry
    (never books less than the target). None on bad input."""
    e, ef, tp, mk, t = _f(entry_px), _f(entry_fee_pct), _f(tp_pct), _f(maker_fee_rate), _f(tick)
    if not (e and e > 0 and tp and tp > 0 and t and t > 0) or ef is None or mk is None or mk < 0 or ef < 0:
        return None
    # net % = 100·m − entry fee % − exit fee % of the ENTRY notional, the exit fee charged on the exit notional (e·(1 ± m)) — solved exactly
    if str(direction or "").upper() == "SHORT":
        move = (tp + ef + mk * 100.0) / (100.0 * (1 + mk))
        raw = e * (1 - move)
        return round(math.floor(raw / t + 1e-9) * t, 12) if raw > 0 else None
    move = (tp + ef + mk * 100.0) / (100.0 * (1 - mk))
    return round(math.ceil(e * (1 + move) / t - 1e-9) * t, 12)


def net_pct_at(direction, entry_px, exit_px, entry_fee_pct, exit_fee_rate) -> Optional[float]:
    e, x, ef, xf = _f(entry_px), _f(exit_px), _f(entry_fee_pct), _f(exit_fee_rate)
    if not (e and e > 0 and x and x > 0) or ef is None or xf is None:
        return None
    move = (x / e - 1) * 100.0 if str(direction or "").upper() != "SHORT" else (1 - x / e) * 100.0
    return round(move - ef - xf * 100.0 * (x / e), 4)     # the exit fee is charged on the exit notional


def resting_tp_shadow(direction, entry_px, entry_fee_pct, tp_pct, maker_fee_rate, tick, high_since, low_since,
                      actual_pnl_pct) -> Tuple[Optional[float], Optional[bool], Optional[float]]:
    """→ (limit price, filled, shadow net %). (None, None, None) when the limit cannot be computed; filled None (unknown) when the
    trade's extreme since entry is missing."""
    lim = resting_tp_price(direction, entry_px, entry_fee_pct, tp_pct, maker_fee_rate, tick)
    if lim is None:
        return None, None, None
    t = _f(tick)
    short = str(direction or "").upper() == "SHORT"
    ext = _f(low_since if short else high_since)
    if ext is None or ext <= 0:
        return lim, None, None
    filled = (ext <= lim - t + 1e-12) if short else (ext >= lim + t - 1e-12)
    if filled:
        return lim, True, net_pct_at(direction, entry_px, lim, entry_fee_pct, maker_fee_rate)
    a = _f(actual_pnl_pct)
    return lim, False, (round(a, 4) if a is not None else None)


def cached_tick(markets, symbol, price) -> Optional[float]:
    """The exchange tick from an ALREADY-LOADED ccxt markets dict (no network, never awaits — the close path holds the global close
    lock). Accepted only when plausible (> 0 and ≤ 0.2 % of the price — the _paper_tick_size guard: a missing / default precision
    must never price a shadow). → None = unknown → the shadow is left NULL."""
    try:
        m = (markets or {}).get(symbol)
        t = ((m or {}).get("precision") or {}).get("price")
        if t is None:
            return None
        t = 10.0 ** (-t) if isinstance(t, int) else float(t)
        p = _f(price)
        return t if (t > 0 and p and p > 0 and t / p <= 0.002) else None
    except Exception:
        return None
