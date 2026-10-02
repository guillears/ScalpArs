"""🩹 RECOVERY HOLD — pure rules (no I/O), shared by the monitor path, the realtime path, the tests and the report.

Operator-directed ARMED override, 2026-10-02 (DECISION_LOG 172). A momentum LONG that reaches its stop while BTC has NOT
weakened is flagged and held instead of closed:

  trigger   close reason STOP_LOSS / STOP_LOSS_WIDE on a MOMENTUM LONG, flagged at most once, and BTC RSI(14) on CLOSED 5m bars
            ≥ its value at entry ∧ rsi_min ≤ RSI ≤ rsi_max (60–66). Missing / stale reading → no hold (fail closed).
  hold      only four things close the trade while it is held (peak P&L < release):
              RH_HARD_STOP     P&L ≤ the stop that was hit − room (0.5)
              RH_PREMISE_EXIT  the closed-bar RSI is < rsi_min or < its entry value (or unreadable)
              RH_TIME_EXIT     held ≥ time_min (30) and still below entry · or held ≥ max_min (240)
              release          peak P&L ≥ release (+0.40) → the normal exit stack takes over; every later close is RH_-prefixed
Evidence (in-sample, live master stops): 13 triggers, hold vs stop +1.75 pts total (+0.31 without the trade that prompted
it), 7 better / 5 worse; the year replay does not confirm. Below every promotion gate → automatic kill bar (rh_tripwire).
"""
from typing import Optional, Sequence, Tuple

RH_PREFIX = "RH_"
RH_STOP_CLASS = ("RH_HARD_STOP", "RH_PREMISE_EXIT", "RH_TIME_EXIT")   # the hold's own closes — all stop-class (urgent, taker)
RH_NO_PREFIX = ("RH_", "MANUAL", "BACKSTOP")                          # reasons the close funnel never RH_-prefixes
BAR_MS = 300_000


def rh_strip(reason) -> str:
    """Base reason of a held trade's close: RH_RUNNER_TRAIL → RUNNER_TRAIL (also behind an FL_ / FLIP_ prefix added later in the
    close path: FL_RH_TRAILING_STOP → FL_TRAILING_STOP). The hold's own closes (RH_HARD_STOP / _PREMISE_EXIT / _TIME_EXIT) keep their name."""
    r = reason or ""
    for pre in ("", "FL_", "FLIP_"):
        if r.startswith(pre + RH_PREFIX) and not r[len(pre):].startswith(RH_STOP_CLASS):
            return pre + r[len(pre) + len(RH_PREFIX):]
    return r


def rh_prefixed(reason, was_held: bool) -> str:
    """The close funnel's rule: any close of a trade that was held carries RH_ (operator closes / exchange backstop excepted)."""
    r = reason or ""
    if was_held and r and not r.startswith(RH_NO_PREFIX):
        return RH_PREFIX + r
    return r


def closed_rsi(ohlcv, now_ms: float, n: int = 14) -> Tuple[Optional[float], Optional[int]]:
    """RSI(n) on CLOSED 5m bars only (Wilder smoothing = ewm alpha 1/n, adjust=False — the research ruler), and the open time
    of the last closed bar. The forming bar (open + 5 min > now) is dropped. (None, None) when unreadable."""
    try:
        bars = [(int(b[0]), float(b[4])) for b in ohlcv if b is not None and b[4] is not None and int(b[0]) + BAR_MS <= now_ms]
        if len(bars) < n * 4:
            return None, None
        up = dn = 0.0; a = 1.0 / n; first = True
        for (_, p0), (_, p1) in zip(bars, bars[1:]):
            d = p1 - p0; u, v = (d if d > 0 else 0.0), (-d if d < 0 else 0.0)
            if first:
                up, dn, first = u, v, False
            else:
                up, dn = up + a * (u - up), dn + a * (v - dn)
        if dn <= 0:
            return (100.0 if up > 0 else None), bars[-1][0]
        return 100.0 - 100.0 / (1.0 + up / dn), bars[-1][0]
    except Exception:
        return None, None


def rsi_fresh(bar_ts_ms, now_ms: float, max_age_bars: float = 2.5) -> bool:
    """The reading is usable when its bar closed recently (a healthy scan gives ≤ 2 bars: the bar's own 5 min + up to 5 min
    until the next close)."""
    try:
        return bar_ts_ms is not None and 0 <= (now_ms - int(bar_ts_ms)) <= max_age_bars * BAR_MS
    except Exception:
        return False


def _f(th, name, default):
    try:
        v = getattr(th, name, default)
        return default if v is None else float(v)
    except Exception:
        return default


def rh_trigger_ok(th, reason, strategy, direction, already_triggered, fl_flagged, pnl, stop_level, peak_pnl,
                  rsi_entry, rsi_now, rsi_is_fresh, floor_pct=None) -> bool:
    """True → flag the trade instead of closing it. Fail closed on anything missing. floor_pct = the deepest hard stop allowed
    (live: just inside the resting exchange backstop, so the hold's own stop always fires first); None = no limit."""
    try:
        if not bool(getattr(th, 'recovery_hold_enabled', False)):
            return False
        if (strategy or "") != "MOMENTUM" or direction != "LONG" or already_triggered or fl_flagged:
            return False
        if not isinstance(reason, str) or not reason.startswith("STOP_LOSS"):
            return False
        if rsi_entry is None or rsi_now is None or not rsi_is_fresh or pnl is None or stop_level is None:
            return False
        lo, hi = _f(th, 'recovery_hold_rsi_min', 60.0), _f(th, 'recovery_hold_rsi_max', 66.0)
        room = _f(th, 'recovery_hold_room_pct', 0.5)
        hard = float(stop_level) - room
        if room <= 0 or float(pnl) <= hard + 0.01:                       # already at / through the hard stop (gap) → plain stop
            return False
        if floor_pct is not None and hard <= float(floor_pct):           # the hold would run into the exchange backstop
            return False
        if peak_pnl is not None and float(peak_pnl) >= _f(th, 'recovery_hold_release_pct', 0.40):
            return False
        return float(rsi_now) >= float(rsi_entry) and lo <= float(rsi_now) <= hi
    except Exception:
        return False


def rh_in_hold(th, triggered_at, peak_pnl) -> bool:
    """Held = triggered and not yet released (peak P&L below the release level)."""
    if triggered_at is None:
        return False
    try:
        return (peak_pnl is None) or float(peak_pnl) < _f(th, 'recovery_hold_release_pct', 0.40)
    except Exception:
        return True


def rh_exit(th, pnl, minutes_held, hard_stop, rsi_entry, rsi_now, rsi_is_fresh, rsi_ever_read=True) -> Optional[str]:
    """While held: the reason to close now, or None to keep holding. Order = hard stop, premise, time.
    rsi_ever_read=False (process just started, no BTC scan yet) skips the premise check — a restart / deploy must not dump an
    open hold; the hard stop and the time exit still bound it, and the first scan (seconds) restores the check."""
    try:
        pnl = float(pnl)
    except Exception:
        return "RH_HARD_STOP"
    if hard_stop is None or pnl <= float(hard_stop) + 0.01:
        return "RH_HARD_STOP"
    if rsi_ever_read:
        if rsi_entry is None or rsi_now is None or not rsi_is_fresh:
            return "RH_PREMISE_EXIT"
        if float(rsi_now) < _f(th, 'recovery_hold_rsi_min', 60.0) or float(rsi_now) < float(rsi_entry):
            return "RH_PREMISE_EXIT"
    m = float(minutes_held or 0.0)
    t_min, t_max = _f(th, 'recovery_hold_time_min', 30.0), _f(th, 'recovery_hold_max_min', 240.0)
    t_min, t_max = (t_min if t_min > 0 else 30.0), (t_max if t_max > 0 else 240.0)      # 0 / blank in the UI never means "exit at once"
    if (m >= t_min and pnl < 0) or m >= t_max:
        return "RH_TIME_EXIT"
    return None


def rh_tripwire(rows: Sequence[Tuple[Optional[float], Optional[float], str]]) -> Tuple[Optional[str], bool]:
    """Automatic KILL BAR. rows = closed held trades in CLOSE order: (final P&L %, P&L % at the trigger, close reason).
    Returns (kill reason or None, judged). KILL: the last 3 holds all ended RH_HARD_STOP (checked for the life of the feature) ·
    or the first 10 holds together did worse than their plain stops (Σ(final − at trigger) < 0). judged=True once 10 holds exist."""
    rs = [(float(f or 0.0), float(t or 0.0), str(r or "")) for f, t, r in rows]
    if len(rs) >= 3 and all("RH_HARD_STOP" in r for _, _, r in rs[-3:]):
        return "3 hard stops in a row", True
    if len(rs) >= 10:
        d = sum(f - t for f, t, _ in rs[:10])
        return (f"first 10 holds {d:+.2f} pts vs their stops" if d < 0 else None), True
    return None, False
