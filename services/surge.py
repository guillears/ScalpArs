"""⚡ SURGE sleeves (Sep-30, DECISION_LOG 146–148) — pure rules shared by the engine, the tests and the research scripts.

A BTC spike (LONG side) or dump (SHORT side) on the last CLOSED 5m bar triggers a short entry window in which the top tradeable
alts that pass the selection are opened as their own sleeve (SURGE_LONG / SURGE_SHORT). Every function here is pure: the trigger
is recomputed from a fresh BTC 5m fetch on every scan, so a deploy never needs a warm-up (bull-run lesson: in-memory history
wiped by 9 deploys in one afternoon)."""
from statistics import median
from typing import Optional

BAR_MS = 300_000


def _f(th, name, default):
    try:
        v = getattr(th, name, default)
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return float(default)


def surge_trigger(btc_bars, th, side: str) -> Optional[dict]:
    """Evaluate the LAST CLOSED BTC 5m bar. `btc_bars` = exchange OHLCV rows [open_ms, o, h, l, c, v] oldest→newest, the LAST row
    being the forming bar (dropped). Needs ≥ 295 closed bars (30-min return + 288-bar 24 h window). Returns
    {bar_ts, close_ts, btc_move_pct} when the side's trigger holds, else None. Fail-closed on short / malformed data.
      LONG : 30-min return ≥ +move ∧ (close ≥ prior 24 h high if required) ∧ (quote volume ≥ mult × prior-288 median if mult > 0)
      SHORT: 30-min return ≤ −move ∧ (close ≤ prior 24 h low  if required) ∧ same volume rule
    `surge_btc_move_pct` is read as a magnitude (a sign slip in the JSON must not disable the trigger — bull-run lesson)."""
    try:
        bars = [r for r in (btc_bars or [])[:-1] if r and len(r) >= 6]
        if len(bars) < 295:
            return None
        c = [float(r[4]) for r in bars]
        last, prev6 = c[-1], c[-7]
        if not (last > 0 and prev6 > 0):
            return None
        move = (last / prev6 - 1.0) * 100.0
        need = abs(_f(th, 'surge_btc_move_pct', 1.0))
        if need <= 0:
            return None
        window = bars[-289:-1]
        if side == "LONG":
            if move < need:
                return None
            if bool(getattr(th, 'surge_long_require_24h_high', True)) and last < max(float(r[2]) for r in window):
                return None
        elif side == "SHORT":
            if move > -need:
                return None
            if bool(getattr(th, 'surge_short_require_24h_low', False)) and last > min(float(r[3]) for r in window):
                return None
        else:
            return None
        mult = _f(th, 'surge_btc_vol_mult', 3.0)
        qv = float(bars[-1][5]) * last
        med = median(float(r[5]) * float(r[4]) for r in window)
        vol_mult = (qv / med) if med > 0 else None
        if mult > 0 and not (vol_mult is not None and vol_mult >= mult):
            return None
        t = int(bars[-1][0])
        return dict(bar_ts=t, close_ts=t + BAR_MS, btc_move_pct=round(move, 4), btc_vol_mult=(round(vol_mult, 2) if vol_mult else None))
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def surge_live_readings(btc_bars, th=None) -> Optional[dict]:
    """The trigger's legs on the LAST CLOSED BTC 5m bar, for the header chip (display only — the trigger itself is surge_trigger):
    RAW 30-min move %, bar quote volume ÷ prior-288 median, close vs the prior 24 h high / low (%), plus the thresholds read with
    surge_trigger's own defaults and the pass flags judged on the UNROUNDED values (review: rounding showed ✓ on a bar that did not
    fire). None on short / bad data."""
    try:
        bars = [r for r in (btc_bars or [])[:-1] if r and len(r) >= 6]
        if len(bars) < 295:
            return None
        c = [float(r[4]) for r in bars]
        window = bars[-289:-1]
        hi = max(float(r[2]) for r in window); lo = min(float(r[3]) for r in window)
        med = median(float(r[5]) * float(r[4]) for r in window)
        move = (c[-1] / c[-7] - 1) * 100
        vm = (float(bars[-1][5]) * c[-1] / med) if med > 0 else None
        need = abs(_f(th, 'surge_btc_move_pct', 1.0)); vneed = _f(th, 'surge_btc_vol_mult', 3.0)
        vol_ok = (vneed <= 0) or (vm is not None and vm >= vneed)
        long_high = bool(getattr(th, 'surge_long_require_24h_high', True)); short_low = bool(getattr(th, 'surge_short_require_24h_low', False))
        return dict(bar_close_ts=int(bars[-1][0]) + BAR_MS, move_pct=move, vol_mult=vm,
                    off_hi_pct=(c[-1] / hi - 1) * 100, off_lo_pct=(c[-1] / lo - 1) * 100,
                    need_move=need, need_vol=vneed, long_need_high=long_high, short_need_low=short_low,
                    ok_long_move=(need > 0 and move >= need), ok_short_move=(need > 0 and move <= -need), ok_vol=vol_ok,
                    ok_high=(c[-1] >= hi), ok_low=(c[-1] <= lo))
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def surge_entry_open(now_ms: float, close_ts: int, th, side: str) -> bool:
    """Is `now` inside the side's entry window: [trigger close + delay, + delay + window)."""
    delay = max(0.0, _f(th, 'surge_long_entry_delay_min' if side == "LONG" else 'surge_short_entry_delay_min', 0.0 if side == "LONG" else 20.0))
    win = max(0.5, _f(th, 'surge_entry_window_min', 5.0))
    start = close_ts + delay * 60_000
    return start <= now_ms < start + win * 60_000


def wilder_atr_pct(bars) -> Optional[float]:
    """ATR(14) % of the last bar's close on closed OHLCV rows (Wilder), None when < 20 bars."""
    try:
        if not bars or len(bars) < 20:
            return None
        h = [float(r[2]) for r in bars]; l = [float(r[3]) for r in bars]; c = [float(r[4]) for r in bars]
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])) for i in range(1, len(bars))]
        atr = sum(trs[:14]) / 14.0
        for t in trs[14:]:
            atr = (atr * 13 + t) / 14.0
        return atr / c[-1] * 100.0 if c[-1] else None
    except (TypeError, ValueError, IndexError):
        return None


def surge_pair_pick(pair_bars, bar_ts: int, btc_move_pct: float, th, side: str):
    """Selection for one candidate pair, judged on the pair's own CLOSED 5m bars up to and including the trigger bar (no look-ahead).
    Returns (ok, refusal_code_or_None, atr_pct, pair_move_pct). Refusal codes: SURGE_NO_DATA · SURGE_ATR_LOW · SURGE_NOT_LEADER."""
    try:
        closed = [r for r in (pair_bars or []) if r and int(r[0]) <= int(bar_ts)]
        if len(closed) < 21 or int(closed[-1][0]) != int(bar_ts):
            return False, "SURGE_NO_DATA", None, None
        atr = wilder_atr_pct(closed)
        c = [float(r[4]) for r in closed]
        pmove = (c[-1] / c[-7] - 1.0) * 100.0 if c[-7] > 0 else None
        if atr is None or pmove is None:
            return False, "SURGE_NO_DATA", atr, pmove
        if atr < _f(th, 'surge_atr_min_pct', 1.5):
            return False, "SURGE_ATR_LOW", atr, pmove
        lead = bool(getattr(th, 'surge_long_require_leader' if side == "LONG" else 'surge_short_require_leader', side == "LONG"))
        if lead and btc_move_pct is not None:
            if (side == "LONG" and not pmove > btc_move_pct) or (side == "SHORT" and not pmove < btc_move_pct):
                return False, "SURGE_NOT_LEADER", atr, pmove
        return True, None, atr, pmove
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return False, "SURGE_NO_DATA", None, None


def surge_tripwire(pnl_pcts, side: str) -> Optional[str]:
    """Automatic KILL BAR on the first 10 closed fills of a side (since it was enabled). Returns the reason to switch the side OFF,
    or None. LONG: ≤ 3 winners ∨ mean ≤ −0.30 %. SHORT: ≤ 4 winners ∨ mean ≤ −0.20 % (LONG failed the grid bar → tighter)."""
    x = [float(p) for p in (pnl_pcts or []) if p is not None][:10]
    if len(x) < 10:
        return None
    wins = sum(1 for p in x if p > 0); mean = sum(x) / len(x)
    max_w, min_mean = (3, -0.30) if side == "LONG" else (4, -0.20)
    if wins <= max_w:
        return f"{wins}/10 winners ≤ {max_w}"
    if mean <= min_mean:
        return f"mean {mean:+.3f}% ≤ {min_mean:+.2f}%"
    return None
