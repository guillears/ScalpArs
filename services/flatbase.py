"""🩳 FLATBASE_SHORT (2026-10-22, DECISION_LOG 348 — operator DISCIPLINE OVERRIDE: the study failed its ship bars) — pure rules shared by the
engine (services/trading_engine.py), the scout (scripts/scout_flatbase_short.py) and the tests. No I/O.

The research it implements, EXACTLY where a live bot can (reports/FRENZY_FLATBASE_SHORT_2026-10-22*.md; the rules are the study scripts
scripts/study_flatbase_short_walk.py · _sim.py · _exits.py · _winloss.py):
  episode   a LIVE FRENZY episode from the engine's own detection (services.frenzy.frenzy_walk / frenzy_flagged — the same flag the FRENZY
            longs use). Its id is spike_ts = the CLOSE time of the spike bar; the spike bar (open = spike_ts − 5 min) anchors the VWAP.
  FLAT      max(close) / min(close) − 1 ≤ flatbase_flat_max (0.444) over the 14 days ENDING at the base bar (the close 30 min before the
            spike bar's close), ≥ 90 % of the bars present — computed ONCE per episode on the SAME 4,032 5m closes as the study (three public
            klines reads, weight 30, once per episode; 15m closes were measured and rejected — flat14_from_bars).
  ENTRY S4  the FIRST 5m bar whose close lands on a UTC 4-hour boundary, ≥ 4 h (48 bars) after the spike bar, close < the spike-anchored
            VWAP (typical price weighted by volume × typical price — the walk's / the sim's convention), 24 h quote volume ≥ $20M, ≤ 96 h
            after the spike bar and not before the engine first saw the episode (the study's first-seen gate). One per episode.
  TURNOVER  24 h volume / market cap > 1 → trade; ≤ 1 → OBSERVE-ONLY shadow (no order); unknown market cap → trade (the study's unscored
            cohort was profitable). Part of the sleeve definition (operator-approved), NOT a filter that passed the expectancy bar.
  EXIT      price −8 % → take profit (a resting maker limit) · price +15 % → stop (bot + exchange backstop) · 72 h → time exit. All three
            are PRICE moves from the fill, as in the sim (gross price levels; fees come on top).
Every helper is total: bad input → a fail-closed answer, never an exception."""
import math
from typing import Dict, List, Optional, Sequence, Tuple

FLATBASE_SHORT = "FLATBASE_SHORT"
FLATBASE_CLASS = (FLATBASE_SHORT,)       # every exit / protection site: run the FLATBASE exit

BAR_MS = 300_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
BOUNDARY_MS = 4 * HOUR_MS               # the S4 candle grid: a 5m bar whose CLOSE is 00/04/08/12/16/20:00 UTC
BASE_LAG_MS = 6 * BAR_MS                # the base bar's close = the spike bar's close − 30 min (study: b = on − 6)
VOL24_BARS = 288
FLAT_TF = "5m"                          # the study's own bars (exact parity). 15m was measured on the study's 6,903 year episodes: its range
FLAT_TF_MS = BAR_MS                     # is never wider but labels 216 more episodes FLAT (+8.9 % on the 2,438 FLAT, 0 fewer) — rejected
FLAT_DAYS = 14
FLAT_EXPECTED = FLAT_DAYS * DAY_MS // FLAT_TF_MS    # 4,032 5m closes in 14 days
FLAT_COVER_MIN = 0.90                   # the study's real-bar rule (≥ 90 % of the window present)
FLAT_PAGE_LIMIT = 1500                  # public /fapi/v1/klines rows per read (weight 10 at limit > 1000)
FLAT_PAGES = 3                          # 3 × 1,500 ≥ 4,032 → request weight 30 per episode, read ONCE (background task, never the pass)
FLAT_MAX_ATTEMPTS = 3                   # an unreadable history is retried on ≤ 2 later bars, then the episode is closed (fail-closed)
ENTRY_MAX_LATE_S = 300                  # the order must reach open_position inside the SAME 5m bar (a missed boundary is never caught up)

# 🔁 (DECISION_LOG 408) tp / stop = the LIVE exit +5 / −10 (DL 348 shipped +8 / −15): a bad config value falls back to the exit that runs
DEFAULTS = dict(flat_max=0.444, min_vol=20e6, turnover_min=1.0, tp=5.0, stop=10.0, max_hold_h=72.0, min_h=4.0, max_h=96.0,
                invest=0.2, lev=0.25, max_open=0.0)
FIELDS = dict(flat_max='flatbase_flat_max', min_vol='flatbase_min_vol_usd', turnover_min='flatbase_turnover_min', tp='flatbase_tp_pct',
              stop='flatbase_stop_pct', max_hold_h='flatbase_max_hold_hours', min_h='flatbase_min_hours_after_spike',
              max_h='flatbase_max_hours_after_spike', invest='flatbase_invest_mult', lev='flatbase_lev_mult', max_open='flatbase_max_open')
MAX_H_CAP = 100.0                       # the 1,499-bar window must hold the spike bar AND 24 h of volume before every candidate bar

# decisions (the episode ledger's status) — one per episode
ST_TRACKING, ST_OPENED, ST_SHADOW, ST_MISSED, ST_EXPIRED = "TRACKING", "OPENED", "SHADOW", "MISSED", "EXPIRED"
ST_NOT_FLAT, ST_UNREADABLE, ST_REFUSED = "NOT_FLAT", "UNREADABLE", "REFUSED"
ST_FINAL = (ST_OPENED, ST_SHADOW, ST_MISSED, ST_EXPIRED, ST_NOT_FLAT, ST_UNREADABLE, ST_REFUSED)
ACT_WAIT, ACT_ENTER, ACT_MISSED, ACT_EXPIRED, ACT_NO_DATA = "WAIT", "ENTER", "MISSED", "EXPIRED", "NO_DATA"
ROUTE_TRADE, ROUTE_SHADOW = "TRADE", "SHADOW"


def _fin(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def flatbase_enabled(th) -> bool:
    """the master switch (flatbase_short_enabled); anything but True = off."""
    return getattr(th, 'flatbase_short_enabled', False) is True


def flatbase_params(th) -> Dict[str, float]:
    """the values the engine USES: each config field when usable, else the study's value (DEFAULTS). Usable = finite and inside the range
    flatbase_config_errors accepts — a hand-edited bad value never widens the rule. Never raises."""
    out = {}
    for k, d in DEFAULTS.items():
        v = _fin(getattr(th, FIELDS[k], d))
        ok = v is not None and {
            'flat_max': lambda x: 0 < x <= 5.0, 'min_vol': lambda x: x >= 0, 'turnover_min': lambda x: x >= 0,
            'tp': lambda x: 0 < x < 100, 'stop': lambda x: 0 < x < 100, 'max_hold_h': lambda x: 0 < x <= 240,
            'min_h': lambda x: 0 <= x < MAX_H_CAP, 'max_h': lambda x: 0 < x <= MAX_H_CAP,
            'invest': lambda x: 0 < x <= 1.0, 'lev': lambda x: 0 < x <= 1.0,
            'max_open': lambda x: x >= 0 and x == int(x)}[k](v)
        out[k] = v if ok else d
    if out['max_h'] <= out['min_h']:
        out['min_h'], out['max_h'] = DEFAULTS['min_h'], DEFAULTS['max_h']
    return out


def flatbase_config_errors(th, base_leverage=20.0) -> list:
    """🩳 PUT /api/config refusals for the FLATBASE_SHORT fields (DECISION_LOG 348). The stop's exchange backstop at the sleeve's leverage
    (base_leverage × flatbase_lev_mult, the STRONG_BUY level the sleeves open at) must sit ≥ MANUAL_BACKSTOP_MIN_GAP_PCT beyond the stop."""
    errs = []
    for k in ('flatbase_short_enabled', 'resting_tp_flatbase_enabled'):
        if not isinstance(getattr(th, k, False), bool):
            errs.append(f"{k} must be true or false")
    rng = dict(flat_max=("a fraction > 0 and ≤ 5 (0.444 = a 44.4 % 14-day range)", lambda x: 0 < x <= 5.0),
               min_vol=("a number ≥ 0 (USD)", lambda x: x >= 0), turnover_min=("a number ≥ 0", lambda x: x >= 0),
               tp=("a number > 0 and < 100 (price %)", lambda x: 0 < x < 100), stop=("a number > 0 and < 100 (price %)", lambda x: 0 < x < 100),
               max_hold_h=("a number > 0 and ≤ 240 (hours)", lambda x: 0 < x <= 240),
               min_h=(f"a number ≥ 0 and < {MAX_H_CAP:g} (hours)", lambda x: 0 <= x < MAX_H_CAP),
               max_h=(f"a number > 0 and ≤ {MAX_H_CAP:g} (hours — the 5m window must hold the spike bar)", lambda x: 0 < x <= MAX_H_CAP),
               invest=("a number > 0 and ≤ 1", lambda x: 0 < x <= 1.0), lev=("a number > 0 and ≤ 1", lambda x: 0 < x <= 1.0),
               max_open=("a whole number ≥ 0 (0 = no cap)", lambda x: x >= 0 and x == int(x)))
    vals = {}
    for k, (txt, ok) in rng.items():
        v = _fin(getattr(th, FIELDS[k], DEFAULTS[k]))
        if v is None or not ok(v):
            errs.append(f"{FIELDS[k]} must be {txt}")
        vals[k] = v
    if vals.get('min_h') is not None and vals.get('max_h') is not None and vals['max_h'] <= vals['min_h']:
        errs.append("flatbase_max_hours_after_spike must be above flatbase_min_hours_after_spike")
    if not errs:
        try:
            from services.trading_engine import order_backstop_pct, MANUAL_BACKSTOP_MIN_GAP_PCT
            lev = max(1, int(round(float(base_leverage or 20.0) * vals['lev'])))
            bk = order_backstop_pct(th, FLATBASE_SHORT, leverage=lev)
            if bk - vals['stop'] < MANUAL_BACKSTOP_MIN_GAP_PCT - 1e-9:
                errs.append(f"flatbase_stop_pct {vals['stop']:g} % is too wide for FLATBASE_SHORT at {lev}×: its exchange backstop ({bk:g} %, "
                            f"capped by liquidation) must sit ≥ {MANUAL_BACKSTOP_MIN_GAP_PCT:g} % beyond it — lower flatbase_lev_mult")
        except Exception:
            errs.append("flatbase_stop_pct: the backstop distance could not be checked")
    return errs


# ── FLAT ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def base_close_ms(spike_ts) -> Optional[int]:
    """the base bar's CLOSE (ms): 30 min before the spike bar's close. None when unreadable."""
    try:
        return int(spike_ts) - BASE_LAG_MS
    except (TypeError, ValueError):
        return None


def flat14_from_bars(rows, base_close, tf_ms=FLAT_TF_MS) -> Tuple[Optional[float], Optional[float], int]:
    """(range, coverage, n) of the 14-day window ENDING at the base bar's close, from kline rows [open_ms, o, h, l, c, v] of tf_ms bars
    (live: 5m — the study's exact 4,032 closes, study_flatbase_short_walk.feats): the bars whose CLOSE lies in (base_close − 14 d, base_close].
    range = max(close) / min(close) − 1 (a fraction: 0.444 = 44.4 %) — None when fewer than 90 % of the window's bars are present (a young
    listing / a data hole: the study's NaN → not FLAT), when the history does not reach back to the window's first bar (a listing younger
    than 14 days — the study's grid starts at the listing → NaN; the caller reads > 14 days so an older pair always has earlier rows) or
    the data is bad; coverage = present / expected.
    tf_ms = 15m (the documented alternative, NOT used): the 5m extremes between two 15m closes are invisible → a range never wider than the
    5m one; on the study's 6,903 year episodes it labelled 216 more FLAT (2,655 vs 2,438) and none fewer → rejected. Never raises."""
    try:
        bc = int(base_close)
        lo = bc - FLAT_DAYS * DAY_MS
        tf = int(tf_ms)
        expected = FLAT_DAYS * DAY_MS // tf
        seen = {}
        first_open = None
        for r in rows or []:
            if not r or len(r) < 5:
                continue
            first_open = int(r[0]) if first_open is None else min(first_open, int(r[0]))
            t_close = int(r[0]) + tf
            if lo < t_close <= bc:
                c = float(r[4])
                if not (math.isfinite(c) and c > 0):
                    return None, None, 0
                seen[int(r[0])] = c
        n = len(seen)
        cov = n / float(expected)
        if n == 0 or cov < FLAT_COVER_MIN - 1e-12 or first_open is None or first_open > lo:
            return None, round(cov, 4), n   # first_open > lo: the history starts inside the window (a < 14-day listing — the study's NaN)
        cs = list(seen.values())
        return max(cs) / min(cs) - 1.0, round(min(cov, 1.0), 4), n
    except (TypeError, ValueError, IndexError):
        return None, None, 0


def flat_ok(range_frac, th) -> bool:
    """FLAT = range ≤ flatbase_flat_max (unknown range → not flat, fail-closed)."""
    r = _fin(range_frac)
    return r is not None and r >= 0 and r <= flatbase_params(th)['flat_max'] + 1e-12


# ── ENTRY (the study's S4) ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def _cols(bars):
    """the rows on a continuous 5m grid, like the study's loader (study_flatbase_short_common.load_full): a missing bar is filled flat at
    the previous close with zero volume. Rows must be ascending on the 5m grid; a row off the grid / out of order → ValueError."""
    t0 = int(bars[0][0]); t1 = int(bars[-1][0])
    if (t1 - t0) % BAR_MS:
        raise ValueError("off grid")
    n = (t1 - t0) // BAR_MS + 1
    t = [t0 + k * BAR_MS for k in range(n)]
    h = [None] * n; lo = [None] * n; c = [None] * n; v = [0.0] * n
    prev = -1
    for r in bars:
        k, rem = divmod(int(r[0]) - t0, BAR_MS)
        if rem or k <= prev:
            raise ValueError("rows not ascending on the 5m grid")
        prev = k
        h[k], lo[k], c[k], v[k] = float(r[2]), float(r[3]), float(r[4]), float(r[5])
    for k in range(n):
        if c[k] is None:
            c[k] = h[k] = lo[k] = c[k - 1]   # k ≥ 1: row 0 always exists
    return t, h, lo, c, v


def s4_scan(bars, spike_ts, th, first_seen_open_ms=None) -> Dict:
    """Walk the episode on CLOSED 5m rows [open_ms, o, h, l, c, v] (oldest first, contiguous — the engine's FRENZY window) and find the
    study's S4 bar. → dict(status, idx, ...) with status:
      NO_DATA   the spike bar is not in the window / bad rows (fail-closed, nothing decided; a missing bar is filled flat like the study)
      ENTER     the FIRST qualifying bar IS the last row (the bar that just closed) → enter now
      MISSED    the first qualifying bar is an EARLIER row (an outage / restart skipped it) → the episode is done, never a later entry
      EXPIRED   no qualifying bar and the last row is already past the 96 h window → the episode is done
      WAIT      no qualifying bar yet, still inside the window
    plus idx (the S4 row, or None), vwap / close / peak / off_peak_pct / peak_age_h (hours since the peak bar, DECISION_LOG 401) / vs_vwap_pct / vol24 / hours / base / run_pct at idx (ENTER /
    MISSED); vol_low_last = the 24 h volume when the bar that just closed met every condition but the volume (WAIT, counter only).
    Qualifying (the sim's cond4, bar index w ≥ the spike bar on): (t[w] + 5 min) % 4 h == 0 ∧ w ≥ on + 48 (min_h × 12) ∧ c[w] < VWAP_w ∧
    w ≥ first seen ∧ 24 h volume at w ≥ min_vol ∧ w ≤ on + 1152 (max_h × 12). VWAP_w = Σ tp·q / Σ q from the spike bar, q = v × tp,
    tp = (h + l + c) / 3 (the walk's quote volume). 24 h volume = Σ q over the 288 bars ending at w (the study: the exchange quote volume
    column — the engine rows carry base volume, so q = v × tp is the live proxy, the same one frenzy_vol24_at uses)."""
    out = dict(status=ACT_NO_DATA, idx=None)
    try:
        if not bars:
            return out
        p = flatbase_params(th)
        t, h, lo, c, v = _cols(bars)   # a missing bar is filled flat with zero volume (the study's grid)
        n = len(t)
        anchor_open = int(spike_ts) - BAR_MS
        if anchor_open < t[0] or anchor_open > t[-1]:
            return out
        on = (anchor_open - t[0]) // BAR_MS
        min_b, max_b = int(round(p['min_h'] * 12)), int(round(p['max_h'] * 12))
        fs = int(first_seen_open_ms) if first_seen_open_ms is not None else None
        q = [v[k] * (h[k] + lo[k] + c[k]) / 3.0 for k in range(n)]
        pv = qq = 0.0
        hi = min(n - 1, on + max_b)
        peak = None
        pk = None   # 🩳⏱ (DECISION_LOG 401) the row that first printed the episode peak
        for w in range(on, hi + 1):
            tp_ = (h[w] + lo[w] + c[w]) / 3.0
            pv += tp_ * q[w]; qq += q[w]
            vw = pv / qq if qq > 0 else c[w]
            if peak is None or h[w] > peak:
                peak, pk = h[w], w
            if w == n - 1:   # the bar that just closed: its readings for the per-boundary audit log (MISSED auditability)
                out.update(last_close=c[w], last_vwap=vw, last_vol24=(sum(q[w - VOL24_BARS + 1: w + 1]) if w >= VOL24_BARS - 1 else None))
            if (t[w] + BAR_MS) % BOUNDARY_MS != 0 or w < on + min_b or not c[w] < vw:
                continue
            if fs is not None and t[w] < fs:
                continue
            if w < VOL24_BARS - 1:
                return out   # the window cannot judge the volume gate (never for a spike ≤ max_h old — MAX_H_CAP)
            vol24 = sum(q[w - VOL24_BARS + 1: w + 1])
            if vol24 < p['min_vol']:
                if w == n - 1:
                    out['vol_low_last'] = vol24   # the bar that just closed met every S4 condition but the 24 h volume (counter only)
                continue
            base = c[on - 6] if on >= 6 else None   # the pre-pump price (the study's base: the close 30 min before the spike bar's close)
            out.update(status=(ACT_ENTER if w == n - 1 else ACT_MISSED), idx=w, close=c[w], vwap=vw, peak=peak, vol24=vol24, base=base,
                       run_pct=((peak / base - 1) * 100 if base and base > 0 else None),
                       vs_vwap_pct=(c[w] / vw - 1) * 100 if vw > 0 else None,
                       off_peak_pct=(c[w] / peak - 1) * 100 if peak and peak > 0 else None,
                       hours=(t[w] - anchor_open) / HOUR_MS, bar_open=t[w],
                       peak_age_h=(t[w] - t[pk]) / HOUR_MS)   # 🩳⏱ (401) hours from the peak bar's close to the S4 bar's close
            return out
        out['status'] = ACT_EXPIRED if (n - 1) >= on + max_b else ACT_WAIT
        return out
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return dict(status=ACT_NO_DATA, idx=None)


def is_boundary_close(bar_open_ms) -> bool:
    """True when a 5m bar opened at bar_open_ms CLOSES on a UTC 4-hour boundary."""
    try:
        return (int(bar_open_ms) + BAR_MS) % BOUNDARY_MS == 0
    except (TypeError, ValueError):
        return False


def turnover_route(vol24, mcap, th) -> Tuple[str, Optional[float]]:
    """(TRADE | SHADOW, turnover) — turnover = 24 h volume / market cap. > flatbase_turnover_min → TRADE · ≤ → SHADOW (observe only) ·
    market cap unknown / ≤ 0 → TRADE with turnover None (the study's unscored cohort, mostly delisted coins, was profitable: 2026 117 · 72 % ·
    +1.62). An unreadable volume with a KNOWN cap → SHADOW (never a real order on an unmeasured ratio)."""
    m = _fin(mcap)
    if m is None or m <= 0:
        return ROUTE_TRADE, None
    vv = _fin(vol24)
    if vv is None or vv < 0:
        return ROUTE_SHADOW, None
    r = vv / m
    return (ROUTE_TRADE if r > flatbase_params(th)['turnover_min'] else ROUTE_SHADOW), round(r, 6)


# ── EXIT (paper = live) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def short_move_pct(entry_px, price) -> Optional[float]:
    """the SHORT's favourable price move from the fill, % (positive = the price fell)."""
    e, x = _fin(entry_px), _fin(price)
    if not (e and e > 0 and x and x > 0):
        return None
    return (1.0 - x / e) * 100.0


def flatbase_exit_for(entry_px, price, held_minutes, th, resting_tp=False) -> Tuple[bool, str, Optional[float]]:
    """FLATBASE_SHORT's own exit → (close, reason, line) on PRICE levels from the fill (the sim's gross levels):
    price ≤ fill × (1 − tp %) → FRENZY_TP (the resting maker limit books it; resting_tp=True → the software TP stays off and the engine's
    resting-TP gate is the safety net) · price ≥ fill × (1 + stop %) → STOP_LOSS · held ≥ max hold → MAX_HOLD_TIME. Unknown move → no close
    (the backstop and the time cap still run). line = the price move % that fired (TP / stop) or the current move (time)."""
    p = flatbase_params(th)
    try:
        mv = short_move_pct(entry_px, price)
        if mv is not None and mv <= -p['stop'] + 1e-9:   # (1e-9: a price exactly ON the level fires — float noise never holds it open)
            return True, "STOP_LOSS", -p['stop']
        if mv is not None and mv >= p['tp'] - 1e-9 and not resting_tp:
            return True, "FRENZY_TP", p['tp']
        if held_minutes is not None and float(held_minutes) >= p['max_hold_h'] * 60.0:
            return True, "MAX_HOLD_TIME", (mv if mv is not None else -p['stop'])   # always a number: the caller's log format never raises
        return False, "MAX_HOLD_TIME", -p['stop']
    except (TypeError, ValueError):
        return False, "MAX_HOLD_TIME", None


def max_hold_minutes(th) -> int:
    return int(round(flatbase_params(th)['max_hold_h'] * 60))


def rtp_net_tp(entry_px, entry_fee_pct, maker_fee_rate, th) -> Optional[float]:
    """the NET take profit (the resting-TP machinery speaks net %) whose maker limit rests at price × (1 − tp %): 100·m − entry fee % −
    the maker fee on the exit notional (services.exec_shadow.net_pct_at, solved for the limit at −tp). None on bad input."""
    e, ef, mk = _fin(entry_px), _fin(entry_fee_pct), _fin(maker_fee_rate)
    if not (e and e > 0) or ef is None or mk is None or ef < 0 or mk < 0:
        return None
    m = flatbase_params(th)['tp'] / 100.0
    net = 100.0 * m - ef - mk * 100.0 * (1.0 - m)
    return round(net, 6) if net > 0 else None


def price_path_exit(entry_px, rows_after, th, entry_fee_rate=0.00045, maker_fee_rate=0.00018, taker_fee_rate=0.00045, slip_pct=0.0
                    ) -> Optional[Dict]:
    """the SCOUT's pricing of a FLATBASE short (the shadow cohort and any re-price) on 5m rows AFTER the entry bar [open, o, h, l, c, v]:
    the sim's order — stop first on a bar that touches both (gap-aware: max(stop level, open)), then the TP (maker at min(level, open)),
    else the close of the last bar inside the max hold. → dict(net_pct, gross_pct, reason, exit_open_ms, bars) or None (no bars).
    Fees: entry taker + exit maker (TP) / taker (stop, time), % of the entry notional like the engine's pnl_percentage. slip_pct: the
    taker slippage of a real fill (% of price, positive = worse), charged on the entry and on a stop / time exit (a resting TP has none).
    Funding is the caller's (the scout adds the settlements over the hold, as the gates add funding_fee_usd to real fills)."""
    try:
        e = float(entry_px) * (1.0 - float(slip_pct or 0) / 100.0)   # a short sells lower than the decision price
        if not (e > 0) or not rows_after:
            return None
        sl = 1.0 + float(slip_pct or 0) / 100.0
        p = flatbase_params(th)
        stop_px, tp_px = e * (1 + p['stop'] / 100.0), e * (1 - p['tp'] / 100.0)
        hb = int(round(p['max_hold_h'] * 12))
        k_last = min(len(rows_after), hb) - 1
        for k in range(k_last + 1):
            o_, h_, l_ = float(rows_after[k][1]), float(rows_after[k][2]), float(rows_after[k][3])
            if h_ >= stop_px:
                x, why, fee = max(stop_px, o_) * sl, "STOP_LOSS", taker_fee_rate
            elif l_ <= tp_px:
                x, why, fee = min(tp_px, o_), "FRENZY_TP", maker_fee_rate
            else:
                continue
            g = (1 - x / e) * 100.0
            return dict(gross_pct=g, net_pct=g - entry_fee_rate * 100.0 - fee * 100.0 * (x / e), reason=why,
                        exit_open_ms=int(rows_after[k][0]), bars=k + 1)
        x = float(rows_after[k_last][4]) * sl
        g = (1 - x / e) * 100.0
        return dict(gross_pct=g, net_pct=g - entry_fee_rate * 100.0 - taker_fee_rate * 100.0 * (x / e),
                    reason=("MAX_HOLD_TIME" if k_last + 1 >= hb else "OPEN"), exit_open_ms=int(rows_after[k_last][0]), bars=k_last + 1)
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


# ── GATES (alert-only — the operator switches; nothing automatic) ───────────────────────────────────────────────────────────────────
GATE_STOP_RUN, GATE_KILL_SUM, GATE_FIRST_N, GATE_MAIN_N, GATE_MAIN_DAYS, GATE_REREAD_N, GATE_STEP_UP_AVG = 3, -30.0, 10, 30, 10, 60, 0.5
GATE_OBS_REAL_N = 30
STOP_REASONS = ("STOP_LOSS", "BACKSTOP_STOP", "BACKSTOP")


def net_of_funding_pct(pnl_pct, funding_fee_usd, entry_price, quantity) -> Tuple[Optional[float], bool]:
    """(pnl % + funding / entry notional × 100, funding known) — the gates read a short NET of funding (shorts pay it in squeezes;
    Order.funding_fee_usd: negative = paid, NOT folded into pnl). Funding unknown (paper / not yet read) → (pnl %, False)."""
    p = _fin(pnl_pct)
    if p is None:
        return None, False
    f, e, q = _fin(funding_fee_usd), _fin(entry_price), _fin(quantity)
    if f is None or not (e and e > 0 and q and q > 0):
        return p, False
    return p + f / (e * q) * 100.0, True


def flatbase_gate(fills) -> Tuple[str, str]:
    """🔒 (DECISION_LOG 348) the FLATBASE_SHORT gates on CLOSED bot fills (operator-closed already excluded by the caller): fills =
    iterable of (opened_at datetime, pnl_percentage, close_reason). Ordered by OPEN time. Verdicts (ALL alert-only):
      REVIEW   early warning — 3 consecutive stop-outs (any time) OR Σ ≤ −30 pts · or the first read: first 10 avg < 0
      STEP_UP  main read on the FROZEN sample = the shortest prefix with ≥ 30 fills on ≥ 10 UTC days: avg ≥ +0.5 % → step the size up
      OFF      main read avg < 0 → switch off
      REREAD   0 ≤ avg < +0.5 → re-read on the first 60 (same rule; still between → INCONCLUSIVE, operator decides)
      COLLECTING otherwise. → (verdict, text). Never raises."""
    try:
        f = sorted([(a, float(p), str(r or '')) for a, p, r in (fills or []) if a is not None and _fin(p) is not None], key=lambda x: x[0])
        n = len(f)
        tail = " (alert-only · operator switches · DECISION_LOG 348)"
        rules = (f"gates: {GATE_STOP_RUN} stop-outs in a row or Σ ≤ {GATE_KILL_SUM:g} pts → review · first {GATE_FIRST_N} avg < 0 → review · "
                 f"main read {GATE_MAIN_N} on ≥ {GATE_MAIN_DAYS} days: avg ≥ +{GATE_STEP_UP_AVG:g} % step up · < 0 off · else re-read at {GATE_REREAD_N}")
        if n == 0:
            return "COLLECTING", f"⏳ 0/{GATE_MAIN_N} closed — {rules}" + tail
        ssum = sum(p for _, p, _ in f)
        run = best = 0
        for _, _, r in f:
            run = run + 1 if r.upper().split(' ')[0] in STOP_REASONS else 0
            best = max(best, run)

        def _read(sample, label):
            avg = sum(p for _, p, _ in sample) / len(sample)
            head = f"📋 {label} — first {len(sample)} avg {avg:+.3f}%"
            if avg >= GATE_STEP_UP_AVG:
                return "STEP_UP", head + f" ≥ +{GATE_STEP_UP_AVG:g} → step the size up (operator)"
            if avg < 0:
                return "OFF", head + " < 0 → switch FLATBASE_SHORT off (operator)"
            return "REREAD", head

        main, ds = None, set()
        for k, (a, _, _) in enumerate(f):
            ds.add(a.date())
            if k + 1 >= GATE_MAIN_N and len(ds) >= GATE_MAIN_DAYS:
                main = f[:k + 1]
                break
        if main is None:
            mv, mt = "COLLECTING", f"⏳ {n}/{GATE_MAIN_N} closed on {len(ds)}/{GATE_MAIN_DAYS} days · Σ {ssum:+.2f} pts · avg {ssum / n:+.3f}%"
        else:
            mv, mt = _read(main, "MAIN READ")
            if mv == "REREAD":
                if n < GATE_REREAD_N:
                    mt += f" (0 … +{GATE_STEP_UP_AVG:g}) → re-read at {GATE_REREAD_N} ({n}/{GATE_REREAD_N})"
                else:
                    v2, t2 = _read(f[:GATE_REREAD_N], f"RE-READ at {GATE_REREAD_N}")
                    mv, mt = (v2 if v2 != "REREAD" else "INCONCLUSIVE"), t2 + (" → still between: operator decides" if v2 == "REREAD" else "")
        warn = []
        if best >= GATE_STOP_RUN:
            warn.append(f"{best} stop-outs in a row")
        if ssum <= GATE_KILL_SUM:
            warn.append(f"Σ {ssum:+.2f} pts ≤ {GATE_KILL_SUM:g}")
        first = f[:GATE_FIRST_N]
        if n >= GATE_FIRST_N and sum(p for _, p, _ in first) / len(first) < 0:
            warn.append(f"first {GATE_FIRST_N} avg {sum(p for _, p, _ in first) / len(first):+.3f}% < 0")
        if warn and mv in ("COLLECTING", "REREAD"):
            return "REVIEW", "⚠ EARLY WARNING — " + " · ".join(warn) + " → ALERT: review FLATBASE_SHORT · main read: " + mt + tail
        if warn:
            return mv, mt + " · ⚠ also: " + " · ".join(warn) + tail
        return mv, mt + tail
    except Exception:
        return "COLLECTING", "gate unreadable"


def flatbase_observe_gate(real_pcts: Sequence[float], shadow_pcts: Sequence[float]) -> Tuple[str, str]:
    """🔒 (DL 348) the observe cohort (turnover ≤ 1, no order): after ≥ 30 REAL closed fills, the shadow outcomes (priced by the scout from
    klines with the same exit) vs the real cohort over the same period: shadow avg > real avg → REVIEW adding them · else KEEP_OUT."""
    try:
        r = [float(x) for x in real_pcts if _fin(x) is not None]
        s = [float(x) for x in shadow_pcts if _fin(x) is not None]
        if len(r) < GATE_OBS_REAL_N:
            return "COLLECTING", f"⏳ {len(r)}/{GATE_OBS_REAL_N} real fills · {len(s)} shadow setups priced"
        if not s:
            return "KEEP_OUT", f"{len(r)} real fills · no shadow setup priced"
        ra, sa = sum(r) / len(r), sum(s) / len(s)
        txt = f"real {len(r)} avg {ra:+.3f}% vs shadow (turnover ≤ 1) {len(s)} avg {sa:+.3f}%"
        return ("REVIEW", txt + " → shadow beats real: review adding them (operator)") if sa > ra else ("KEEP_OUT", txt + " → keep them out")
    except Exception:
        return "COLLECTING", "gate unreadable"
