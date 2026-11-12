"""📈 MOMENTUM LONG RISK CAP — pure rules (no I/O), DECISION_LOG 533 (B26 leverage step, operator 2026-11-11).

At a new MOMENTUM LONG entry: Σ the open momentum longs' risk (what each would lose at its CURRENT exit line, % of the account) + the new
trade's risk at its own worst stop > cap % → the new trade is shrunk to fit; below the minimum margin → skipped (the caller decides).

  open trade's exit line   in its second chance  → L − floor (its SC_FLOOR)            risk = max(0, −line)
                           in a recovery hold     → its hard stop (rh_hard_stop_pct)
                           runner armed (peak ≥ arm) → the BE lock (+0.10)            risk 0 (locked in profit)
                           otherwise              → its worst stop (worst_stop_pct)
  (GROSS price stops — fees + slip add ≈ 0.2 % of the position per trade, ≈ +0.7 % of the account per 15× momentum long on top.
   NOT counted for an unheld trade: the recovery hold's extra room (0.5) — it fires only under its BTC-RSI condition, has its own kill bar,
   and counting it on every open trade would leave room for ~1.5 unarmed momentum longs at 10 %; a trade IN a hold counts at its hard stop.)
  worst stop               the widest the momentum stack can apply before break-even: max(|confidence stop|, |signal-active stop|,
                           sl_atr_multiplier × entry ATR) clamped at |sl_atr_widen_floor_pct|, then the quiet-pair stop if one applies.
"""
from typing import Iterable, Optional


def _f(x, d=None):
    try:
        v = float(x)
        return v if v == v else d
    except (TypeError, ValueError):
        return d


def worst_stop_pct(stop_loss, signal_active_sl, atr_pct, atr_mult, floor_pct, quiet_sl=None) -> float:
    """→ the widest pre-break-even stop, as a positive %. Missing inputs fall back to 0.70 (the base stop)."""
    base = abs(_f(stop_loss, -0.70) or 0.70)
    w = max(base, abs(_f(signal_active_sl, 0.0) or 0.0))
    a, m = _f(atr_pct), _f(atr_mult, 0.0) or 0.0
    if a is not None and a > 0 and m > 0:
        w = max(w, a * m)
    fl = _f(floor_pct, 0.0) or 0.0
    if fl < 0:                       # indicators.check_exit_conditions: the widened stop is clamped at the floor (−1.2)
        w = min(w, abs(fl))
    q = _f(quiet_sl)
    if q is not None and q < 0:
        w = max(w, abs(q))
    return float(w)


def open_trade_risk_pct(sc_line=None, rh_hard_stop=None, peak=None, arm=0.40, worst_stop=0.70) -> float:
    """→ the % of the POSITION an open momentum long would lose at its current exit line (≥ 0)."""
    if sc_line is not None:
        return max(0.0, -float(sc_line))
    if rh_hard_stop is not None:
        return max(0.0, -float(rh_hard_stop))
    if peak is not None and float(peak) >= float(arm) - 0.005:
        return 0.0
    return max(0.0, float(worst_stop))


def open_risk_pct_of_account(rows: Iterable[tuple], account: float) -> float:
    """rows = (notional $, risk % of position) → Σ $ at risk as % of the account."""
    if not account or account <= 0:
        return 0.0
    return sum(max(0.0, float(n)) * max(0.0, float(r)) / 100.0 for n, r in rows) / float(account) * 100.0


def fit_notional(new_notional: float, new_stop_pct: float, open_risk_pct: float, account: float, cap_pct: float) -> Optional[float]:
    """→ the largest notional ≤ new_notional that keeps Σ risk ≤ cap_pct (None = no cap / no account). 0 = no room left."""
    if cap_pct is None or cap_pct <= 0 or not account or account <= 0 or not new_stop_pct or new_stop_pct <= 0:
        return None
    room = (cap_pct - open_risk_pct) / 100.0 * account
    if room <= 0:
        return 0.0
    return min(float(new_notional), room / (new_stop_pct / 100.0))
