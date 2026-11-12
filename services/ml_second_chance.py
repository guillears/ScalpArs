"""🔂 MOMENTUM LONG SECOND CHANCE — pure rules (no I/O), shared by the engine's close funnel, its realtime path, the tests and the scout.

Operator-directed ARMED ship (paper), 2026-11-11 (DECISION_LOG 532). Studies: reports/ML_SECOND_CHANCE_2026-11-11*, ML_RUNNER_REGIME_2026-11-11*,
ML_SC_REFINE_2026-11-11* (master 123 kept ML fills: +33.1 → +44.9 pts, one-size DCR 0.57 → 0.77 %/day, gain on winners +11.0 / losers +0.8;
B25 +10.7 → +17.8; 2-year backtest +12.1 pts, +0.022/fill CI [−0.003, +0.049]).

  zone      a MOMENTUM LONG (not flip / any other sleeve; probes run it too — paper = live — the gate counts full size) whose pair ADX was
            FALLING at entry: entry_adx_delta ≤ adx_delta_max (0) — a fresh move, not yet an established trend. Never a recovery-hold trade
            (the RH kill bar judges RH alone), never a trigger below 0 % (a gapped L would put the floor near the stop).
  trigger   today's RUNNER_TRAIL or HARD_TP_LADDER would close it at net level L → it is NOT closed; phase 2 starts (once per trade)
  phase 2   exit at the first of
              SC_FLOOR  P&L ≤ L − floor (0.30)                                 (stop-class: urgent close)
              SC_TRAIL  P&L ≤ H − trail_mult (1.0) × entry ATR %, H = the highest P&L since the trigger (starts at L)
              SC_CLOCK  the trade is ≥ clock_min (180) minutes old             (the study's clock: today's no-expansion window from entry)
            no ATR stamp → floor only. The runner trail / ladder never close a phase-2 trade; every other exit keeps its own rules.
  shadow    sc_base_pnl / sc_base_reason / sc_armed_at = what today's exit would have booked and when — the revert gate reads A vs today
            straight from these columns (sc_base_pnl vs pnl_percentage).
"""
from typing import Optional

SC_REASONS = ("SC_FLOOR", "SC_TRAIL", "SC_CLOCK")
SC_URGENT = ("SC_FLOOR",)
_INTERCEPT = ("RUNNER_TRAIL", "HARD_TP_LADDER")


def _f(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def sc_params(th) -> dict:
    """The live knobs (thresholds) with their study defaults."""
    return dict(
        enabled=bool(getattr(th, 'ml_second_chance_enabled', False)),
        adx_delta_max=_f(getattr(th, 'ml_second_chance_adx_delta_max', 0.0), 0.0),
        floor=_f(getattr(th, 'ml_second_chance_floor_pct', 0.30), 0.30),
        trail_mult=_f(getattr(th, 'ml_second_chance_trail_atr_mult', 1.0), 1.0),
        clock_min=_f(getattr(th, 'ml_second_chance_clock_min', 180.0), 180.0),
    )


def sc_in_zone(prm: dict, entry_strategy, direction, entry_adx_delta) -> bool:
    """MOMENTUM LONG with ADX falling at entry. A missing ADX stamp is never in the zone (fail closed: today's exit)."""
    if not prm.get('enabled'):
        return False
    if (entry_strategy or "") != "MOMENTUM" or (direction or "") != "LONG":
        return False
    d = _f(entry_adx_delta)
    return d is not None and d <= prm['adx_delta_max']


def sc_base_reason(reason) -> Optional[str]:
    """The runner / ladder reason this rule intercepts (RH_-, L-suffixed variants included) → its base name, else None.
    RUNNER_TRAIL_LATE (the fade late-arm) is never intercepted."""
    r = (reason or "").strip()
    if r.startswith("RH_"):
        r = r[3:]
    if r.startswith("RUNNER_TRAIL_LATE"):
        return None
    for b in _INTERCEPT:
        if r.startswith(b):
            return b
    return None


def sc_phase2_exit(pnl, level, high, entry_atr_pct, age_min, prm: dict) -> Optional[str]:
    """Phase-2 exit for one tick. `high` must already include this tick. → SC_FLOOR / SC_TRAIL / SC_CLOCK or None."""
    p, L = _f(pnl), _f(level)
    if p is None or L is None:
        return None
    if p <= L - prm['floor'] + 1e-12:
        return "SC_FLOOR"
    atr = _f(entry_atr_pct)
    h = max(_f(high, L), L)
    if atr is not None and atr > 0 and prm['trail_mult'] > 0 and p <= h - prm['trail_mult'] * atr + 1e-12:
        return "SC_TRAIL"
    a = _f(age_min)
    if a is not None and prm['clock_min'] > 0 and a >= prm['clock_min']:
        return "SC_CLOCK"
    return None


def net_pnl_pct(direction, entry_price, price, quantity, entry_fee, taker_fee) -> Optional[float]:
    """The realtime path's net P&L % (entry fee + an estimated taker exit fee, on entry notional)."""
    e, c, q = _f(entry_price), _f(price), _f(quantity)
    if not e or not c or not q or e <= 0 or q <= 0:
        return None
    raw = (c - e) * q if direction == "LONG" else (e - c) * q
    pnl = raw - (_f(entry_fee, 0.0) or 0.0) - c * q * (_f(taker_fee, 0.0) or 0.0)
    return pnl / (e * q) * 100.0
