"""🔥 FRENZY sleeve (Oct-2, DECISION_LOG 176) — pure rules shared by the engine and the tests. No I/O.

A pair is FLAGGED after a volume spike and followed for up to 96 h. While flagged, a LONG opens when the "staircase" state turns
on (the research rule of scripts/staircase_long_tight_test.py / frenzy_long_followup.py) and the pair's 5m ATR is small enough
for the fixed stop; every close below the 5m EMA50 / EMA200 of a flagged pair is recorded as a SHORT observation (no trade).

  spike     a 5m close with 30-min return ≥ frenzy_spike_ret_pct ∧ last-hour quote volume ≥ frenzy_spike_vol_mult × the pair's
            normal hour ∧ ≥ frenzy_spike_min_hour_usd. The spike-anchored VWAP starts at that bar.
  episode   starts at a spike bar; ends when 24 h pass without a state bar — the next spike bar after that starts a new one.
  state     ≥ frenzy_min_hours after the spike ∧ every 5m close of the last hour ≥ the anchored VWAP ∧ last-hour volume ≥
            frenzy_state_vol_mult × normal.
  entry     the state turns ON after ≥ 1 h off (the first state bar of a stretch) ∧ 24 h volume ≥ frenzy_min_volume_usd ∧
            5m ATR(14) ≤ frenzy_max_atr_pct, judged on that bar only.
  exit      stop −frenzy_stop_pct · once the peak reaches +frenzy_trail_arm_pct, close frenzy_trail_giveback_pct below the best
            price · frenzy_max_hold_minutes. P&L is the caller's net-of-fees % of the position.
Everything is recomputed from a fresh kline window on every evaluation, so a deploy needs no warm-up; the engine persists only
the NAMES of the flagged pairs (table frenzy_flags) so a cooled pair is re-checked after a restart.
Quote volume is taken as volume × typical price (the exchange OHLCV rows carry base volume)."""
from statistics import median
from typing import List, Optional, Tuple

BAR_MS = 300_000
HOUR_MS = 3_600_000
GAP_MS = 24 * HOUR_MS          # an episode ends after this long without a state bar
VERIFY_MS = GAP_MS + 12 * BAR_MS   # history needed before a spike to know it is the first one of its episode


def _f(th, name, default):
    try:
        v = getattr(th, name, default)
        return float(default) if v is None else float(v)
    except (TypeError, ValueError):
        return float(default)


def normal_hour_usd(hour_bars, last_closed_ms: int) -> Optional[float]:
    """The pair's normal hourly quote volume: the median of full 1h bars that ENDED at least 24 h before the last closed 5m bar
    (so the frenzy itself never inflates it). None when the pair has < 240 such hours (listed < ~10 days) or the data is bad."""
    try:
        q = [float(r[5]) * (float(r[2]) + float(r[3]) + float(r[4])) / 3.0 for r in (hour_bars or [])
             if r and len(r) >= 6 and int(r[0]) + HOUR_MS <= int(last_closed_ms) - GAP_MS]
        if len(q) < 240:
            return None
        m = median(q[-720:])
        return m if m > 0 else None
    except (TypeError, ValueError, IndexError):
        return None


def frenzy_walk(bars, norm_hour, th) -> Optional[dict]:
    """bars = CLOSED 5m OHLCV rows [open_ms, o, h, l, c, v], oldest first. Returns the LIVE episode (the one still running at the
    last bar) or None. Keys: spike_ts (close time of the spike bar), base (price 30 min before it), vwap, price, peak, hours,
    vol_mult, above_hour, in_state, fresh_on (the last bar is the first state bar after ≥ 1 h off), verified, last_bar_ts,
    vs_vwap_pct, run_pct, gain_pct. Fail-closed.
    verified = the spike is known to be its episode's FIRST: either the previous episode is fully inside the window and was itself
    verified, or the 25 h before the spike are visible and hold no bar at the state volume (a state bar needs that volume, so an
    older episode whose own spike already left the window cannot still be alive — review: a 5-day frenzy re-anchored on a
    mid-episode spike once its first spike scrolled out)."""
    try:
        if not bars or len(bars) < 40 or not norm_hour or not norm_hour > 0:
            return None
        t = [int(r[0]) for r in bars]; h = [float(r[2]) for r in bars]; l = [float(r[3]) for r in bars]; c = [float(r[4]) for r in bars]
        q = [float(r[5]) * (h[i] + l[i] + c[i]) / 3.0 for i, r in enumerate(bars)]
        n = len(c)
        ret_min, vol_min, hour_min = _f(th, 'frenzy_spike_ret_pct', 5.0), _f(th, 'frenzy_spike_vol_mult', 20.0), _f(th, 'frenzy_spike_min_hour_usd', 2e6)
        state_vol, min_bars = _f(th, 'frenzy_state_vol_mult', 100.0), max(12, int(round(_f(th, 'frenzy_min_hours', 2.0) * 12)))
        if ret_min <= 0 or vol_min <= 0 or state_vol <= 0:
            return None
        q1h = [None] * n; run = 0.0
        for i in range(n):
            run += q[i]
            if i >= 12:
                run -= q[i - 12]
            if i >= 11:
                q1h[i] = run
        volx = [(x / norm_hour) if x is not None else None for x in q1h]
        lead = [i for i in range(11, n) if c[i - 6] > 0 and (c[i] / c[i - 6] - 1) * 100 >= ret_min and volx[i] >= vol_min and q1h[i] >= hour_min]
        i = 0; chained = False                              # chained = the previous in-window episode was verified and has ended
        for on in lead:
            if on < i:
                continue                                    # inside the previous episode
            verified = chained or bool(t[on] - t[0] >= VERIFY_MS and not any(
                volx[k] is not None and volx[k] >= state_vol for k in range(max(0, on - VERIFY_MS // BAR_MS), on)))
            pv = qq = 0.0; vw = []; above = []; state = []; streak = 0
            for j in range(on, n):
                pv += (h[j] + l[j] + c[j]) / 3.0 * q[j]; qq += q[j]
                v = pv / qq if qq > 0 else c[j]; vw.append(v)
                streak = streak + 1 if c[j] >= v else 0
                ab = streak >= 12; above.append(ab)
                state.append(bool(ab and volx[j] is not None and volx[j] >= state_vol and (j - on) >= min_bars))
            last = on; end = None
            for j in range(on + 1, n):
                if t[j] - t[last] > GAP_MS:
                    end = j; break
                if state[j - on]:
                    last = j
            if end is not None:
                i = end; chained = verified; continue
            k = n - 1 - on
            fresh = bool(state[k] and not any(state[max(0, k - 12):k]))
            base = c[on - 6]; peak = max(h[on:]); price = c[-1]
            return dict(spike_ts=t[on] + BAR_MS, base=base, vwap=vw[-1], price=price, peak=peak,
                        hours=(t[-1] - t[on]) / HOUR_MS, vol_mult=volx[-1], above_hour=bool(above[-1]), in_state=bool(state[-1]),
                        fresh_on=fresh, verified=verified, last_bar_ts=t[-1],
                        vs_vwap_pct=(price / vw[-1] - 1) * 100 if vw[-1] > 0 else None,
                        run_pct=(peak / base - 1) * 100, gain_pct=(price / base - 1) * 100,
                        off_peak_pct=(price / peak - 1) * 100 if peak > 0 else None)
        return None
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


def frenzy_flagged(ep, th) -> bool:
    """A live episode counts as a FRENZY flag while its spike is verifiable and ≤ frenzy_max_hours old."""
    return bool(ep and ep.get('verified') and ep.get('hours') is not None and ep['hours'] <= _f(th, 'frenzy_max_hours', 96.0))


def frenzy_long_status(ep, atr_pct, volume_24h, th) -> Tuple[bool, str, str]:
    """(ready, code, text) for a flagged pair. ready = the LONG may open on this bar. Codes double as filter-block counters."""
    vneed = _f(th, 'frenzy_state_vol_mult', 100.0)
    if not ep.get('in_state'):
        if not ep.get('above_hour'):
            return False, "FRENZY_BELOW_AVG", "below its average price"
        if ep.get('vol_mult') is None or ep['vol_mult'] < vneed:
            return False, "FRENZY_VOL_FADED", f"volume {ep.get('vol_mult') or 0:.0f}× < {vneed:.0f}×"
        return False, "FRENZY_TOO_EARLY", f"{ep.get('hours') or 0:.1f} h after the spike < {_f(th, 'frenzy_min_hours', 2.0):g} h"
    if not ep.get('fresh_on'):
        return False, "FRENZY_ON", "ON (entry bar passed)"
    vmin = _f(th, 'frenzy_min_volume_usd', 20e6)
    if volume_24h is None or float(volume_24h) < vmin:
        return False, "FRENZY_VOL24_LOW", f"24 h volume ${(volume_24h or 0) / 1e6:.0f}M < ${vmin / 1e6:.0f}M"
    amax = _f(th, 'frenzy_max_atr_pct', 2.5)
    if amax > 0 and (atr_pct is None or float(atr_pct) > amax):
        return False, "FRENZY_ATR_HIGH", (f"ATR {atr_pct:.2f}% > {amax:g}%" if atr_pct is not None else "ATR unreadable")
    return True, "FRENZY_READY", "READY"


def frenzy_exit_for(pnl, peak_pnl, th, stop_floor=None, short=False) -> Tuple[bool, str, float]:
    """FRENZY_LONG's own exit → (close, reason, line). Stop at −frenzy_stop_pct; once the peak reaches +frenzy_trail_arm_pct the
    line trails frenzy_trail_giveback_pct of PRICE below the best point (peak − giveback × (1 + peak/100)). Reasons are the
    momentum stack's own (STOP_LOSS / RUNNER_TRAIL) so every matcher knows them; entry_strategy tells the sleeve apart.
    stop_floor (live only): never wider than this — just inside the resting exchange backstop.
    short=True (a MANUAL short on this exit): the give-back is measured from the LOWEST price → peak − giveback × (1 − peak/100)."""
    try:
        pnl = float(pnl); pk = float(peak_pnl or 0.0)
        stop = -abs(_f(th, 'frenzy_stop_pct', 3.0)) or -3.0
        if stop_floor is not None:
            stop = max(stop, float(stop_floor))
        arm, give = abs(_f(th, 'frenzy_trail_arm_pct', 5.0)), abs(_f(th, 'frenzy_trail_giveback_pct', 1.5))
        if arm > 0 and give > 0 and pk >= arm:
            line = max(stop, pk - give * ((1 - pk / 100.0) if short else (1 + pk / 100.0)))
            return (pnl <= line), "RUNNER_TRAIL", line
        return (pnl <= stop), "STOP_LOSS", stop
    except (TypeError, ValueError):
        try:
            return (float(pnl) <= -3.0), "STOP_LOSS", -3.0
        except (TypeError, ValueError):
            return False, "STOP_LOSS", -3.0


def frenzy_breaks(bars) -> List[int]:
    """SHORT observation: which lines (50 / 200) did the LAST closed 5m bar break — a close below the EMA of 5m closes with the
    previous 12 closes all at/above it (the research trigger of scripts/break_short_review.py). [] when none / too few bars."""
    out = []
    try:
        c = [float(r[4]) for r in (bars or [])]
        if len(c) < 260:
            return out
        for span in (50, 200):
            a = 2.0 / (span + 1); e = c[0]; ema = []
            for x in c:
                e = e + a * (x - e); ema.append(e)
            if c[-1] < ema[-1] and all(c[k] >= ema[k] for k in range(len(c) - 13, len(c) - 1)):
                out.append(span)
    except (TypeError, ValueError, IndexError):
        return []
    return out
