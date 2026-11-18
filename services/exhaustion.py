"""🪫 EXHAUSTION_SHORT (2026-11-18, DECISION_LOG 578 — operator-approved PAPER run, DECLARED OVERRIDE: the study's verdict was OBSERVE) — pure
rules shared by the engine (services/trading_engine.py), the scout (scripts/scout_exhaustion_short.py) and the tests. No I/O.

The research it implements (reports/SPOT_SHARE_DEEP_2026-11-17.md — R4 "R0 × vol_decay", ADDENDUM 2 capped exits, the SIZING paper spec).
Every definition below is COPIED from the study code and checked against study rows by tests/test_exhaustion_short.py (fixture
tests/fixtures/exh_parity.npz, built by scripts/exh_parity_fixture.py from the study's own data):
  episode   a FRENZY episode from the engine's own detection (services.frenzy.frenzy_walk / frenzy_flagged — the flag the FRENZY longs and
            FLATBASE use). spike_ts = the CLOSE of the spike bar; on_ms = its OPEN (= the study's episodes_meta on_ms; spike_ms = on_ms + 5 min).
            A signal bar is eligible while its close T lies in [spike_ts, spike_ts + 72 h] (frenzy_ta_build.work: ep_of over [on, spike + 72 h]).
  5m grid   the perp 5m klines WITH the exchange quote volume (kline field 7), on a continuous grid; a missing bar is filled flat at the
            previous close with zero volume (frenzy_ta_build.load).
  15m bars  5m bars grouped by open // 15 min: open = first, high = max, low = min, close = last, quote volume = sum (ta_wl_features.agg).
  RSI(14)   Wilder: d = diff(close, prepend close[0]); up / down = EWM(alpha 1/14, adjust=False) of max(d, 0) / max(−d, 0);
            RSI = 100 − 100 / (1 + up / down); down = 0 → 100; the first 14 values undefined (frenzy_ta_build.rsi).
  SIGNAL    RSI14X SHORT on the CLOSED 15m bar that closes at T: RSI(prev bar) > 70 and RSI(this bar) ≤ 70 (frenzy_ta_build.family_signals
            'RSI14X' short = (sh(r14) > 70) & (r14 <= 70) — the operator's "≥ 70 → < 70" in the study's own boundary convention).
  ATR(14)   Wilder ATR on the 15m bars at the signal bar: TR = max(h − l, |h − prev close|, |l − prev close|) (first TR = h − l), EWM alpha
            1/14 adjust=False, first 14 undefined (frenzy_ta_build.atr). atr_pct = ATR ÷ price × 100 (study: ÷ the entry bar's open; live: ÷
            the signal close, the decision price — the same price up to the first tick of the entry bar).
  vol24     Σ quote volume of the 288 5m bars ending at the signal bar's last 5m bar ≥ $20M (frenzy_ta_build.work VOL_GATE).
  vol_decay roll1h(j) ÷ max(roll1h over the episode's bars from the spike bar's open to j), roll1h = Σ quote volume of the 12 5m bars ending
            at that bar (rolling 12, min 12; undefined → 0 in the running max) — ta_wl_features.work. KEEP when ≥ 0.6366 (R4: ssd_capped
            VD_THR, "the DESIGN-better half"): the last hour still carries ≥ 63.66 % of the episode's peak hourly volume. (The operator's
            sentence "volume cooled ≥ ~64 %" describes the opposite side; the study's code — the source of the evidence — keeps ≥ 0.6366.)
  spot_share_chg   s1 / (s1 + p1) − s24 / (s24 + p24): s1 = spot 15m quote volume of the bars opened in [T − 1 h, T − 15 min] (≥ 3 bars),
            s24 = those opened in [T − 25 h, T − 1 h) (≥ 80 bars); p1 = perp roll1h at the signal bar; p24 = perp 5m quote volume of the
            bars opened in [T − 25 h, T − 1 h) (ta_wl_features.work B2). KEEP when finite and < −0.0066 (ssd_eval CUT). No Binance SPOT
            USDT market (same symbol) → unscored → no trade (the study's has_spot = 0 rows are never kept).
  lags      every input closes ≤ T (perp 5m bars open + 5 min ≤ T, spot 15m bars open + 15 min ≤ T). The study's metrics rule
            (create_time ≤ T − 5 min) applies to no input of this rule (no metrics feature is used).
  ENTRY     SHORT at market right after T (the study: the next 5m bar's open).
  EXIT      TP = min(3 × ATR%, 5 %) below the fill (resting maker limit) · SL = min(1.5 × ATR%, 5 %) above the fill · 24 h time exit
            (ssd_capped "TPcap 5% + SLcap 5%", 24 h; the SIZING spec). PRICE levels from the fill, as in the sim.
  Q5 / slip Q5 = median 5m quote volume of the 12 bars ending at the signal bar (ssd_bars q5); modelled slippage % = max(0, −0.042 + 0.276 ·
            √(notional ÷ Q5)) (reports/SPOT_SHARE_DEEP_data/slip_model.json) — the revert gate prices paper fills with it.
Every helper is total: bad input → a fail-closed answer, never an exception."""
import math
from typing import Dict, List, Optional, Sequence, Tuple

EXHAUSTION_SHORT = "EXHAUSTION_SHORT"
EXHAUSTION_CLASS = (EXHAUSTION_SHORT,)   # every exit / protection site: run the EXHAUSTION exit

BAR_MS = 300_000
M15_MS = 900_000
HOUR_MS = 3_600_000
VOL24_BARS = 288
ROLL1H_BARS = 12
SPOT_MIN_1H, SPOT_MIN_24H = 3, 80          # ta_wl_features: (b1 − a1) ≥ 3 and (a1 − a24) ≥ 80
KLINE_LIMIT = 1500                         # the perp 5m read on a signal (weight 10): 125 h ≥ the 72 h window + 24 h volume + RSI warm-up
SPOT_LIMIT = 104                           # spot 15m bars ending before T (100 needed: 25 h ÷ 15 min; weight 2)
ENTRY_MAX_LATE_S = 300                     # the order must reach open_position within the entry 5m bar (the study filled at its open)
SLIP_A, SLIP_B = -0.041640290342889685, 0.27612727721342023   # reports/SPOT_SHARE_DEEP_data/slip_model.json (118 live fills)

DEFAULTS = dict(rsi_period=14, rsi_level=70.0, vol_decay_min=0.6366, ssc_max=-0.0066, tp_atr=3.0, tp_cap=5.0, sl_atr=1.5, sl_cap=5.0,
                max_hold_h=24.0, max_h=72.0, min_vol=20e6, invest=0.5, lev=0.25,
                oivol_ssc_min=-0.0366, oivol_min=0.128)   # 🧯 (DECISION_LOG 582) EXH_SSC_OIVOL block cuts
FIELDS = dict(rsi_period='exhaustion_rsi_period', rsi_level='exhaustion_rsi_level', vol_decay_min='exhaustion_vol_decay_min',
              ssc_max='exhaustion_spot_share_chg_max', tp_atr='exhaustion_tp_atr_mult', tp_cap='exhaustion_tp_cap_pct',
              sl_atr='exhaustion_sl_atr_mult', sl_cap='exhaustion_sl_cap_pct', max_hold_h='exhaustion_max_hold_hours',
              max_h='exhaustion_max_hours_after_spike', min_vol='exhaustion_min_vol_usd', invest='exhaustion_invest_mult',
              lev='exhaustion_lev_mult', oivol_ssc_min='exhaustion_ssc_oivol_ssc_min', oivol_min='exhaustion_ssc_oivol_oivol_min')
MAX_H_CAP = 100.0                          # the 1,500-bar window must hold the spike bar, 25 h of volume and the RSI warm-up
_RANGES = dict(rsi_period=("a whole number ≥ 2 and ≤ 50", lambda x: 2 <= x <= 50 and x == int(x)),
               rsi_level=("a number > 50 and < 100", lambda x: 50 < x < 100),
               vol_decay_min=("a number ≥ 0 and ≤ 1 (share of the episode's peak hourly volume)", lambda x: 0 <= x <= 1.0),
               ssc_max=("a number ≥ −1 and ≤ 1 (spot-share change, a fraction)", lambda x: -1.0 <= x <= 1.0),
               tp_atr=("a number > 0 and ≤ 20", lambda x: 0 < x <= 20), tp_cap=("a number > 0 and < 50 (price %)", lambda x: 0 < x < 50),
               sl_atr=("a number > 0 and ≤ 20", lambda x: 0 < x <= 20), sl_cap=("a number > 0 and < 50 (price %)", lambda x: 0 < x < 50),
               max_hold_h=("a number > 0 and ≤ 240 (hours)", lambda x: 0 < x <= 240),
               max_h=(f"a number > 0 and ≤ {MAX_H_CAP:g} (hours — the 5m window must hold the spike bar)", lambda x: 0 < x <= MAX_H_CAP),
               min_vol=("a number ≥ 0 (USD)", lambda x: x >= 0), invest=("a number > 0 and ≤ 1", lambda x: 0 < x <= 1.0),
               lev=("a number > 0 and ≤ 1", lambda x: 0 < x <= 1.0),
               oivol_ssc_min=("a number ≥ −1 and ≤ 1 (spot-share change, a fraction)", lambda x: -1.0 <= x <= 1.0),
               oivol_min=("a number ≥ 0 and ≤ 100 (open interest ÷ 24 h quote volume)", lambda x: 0 <= x <= 100.0))

# the journal's status (models.ExhaustionSignal) — one row per evaluated signal
ST_OPENED, ST_BLOCKED, ST_REFUSED = "OPENED", "BLOCKED", "REFUSED"
# block codes (filter-block counters, D11) — EXH_ prefix = the sleeve's own (never a momentum Top-Pairs reason)
B_NO_DATA, B_NO_SPOT, B_VOL24, B_VOL_DECAY, B_SPOT_SHARE, B_NO_SIGNAL = ("EXH_NO_DATA", "EXH_NO_SPOT", "EXH_VOL24_LOW", "EXH_VOL_DECAY",
                                                                         "EXH_SPOT_SHARE", "EXH_NO_SIGNAL")
B_PAIR_HELD, B_OPEN_REFUSED, B_FAILED, B_LATE = "EXH_PAIR_HELD", "EXH_OPEN_REFUSED", "EXH_FAILED", "EXH_LATE"
B_SSC_OIVOL, B_SSC_OIVOL_UNREAD = "EXH_SSC_OIVOL", "EXH_SSC_OIVOL_UNREAD"   # 🧯 (DECISION_LOG 582) the block · the fail-open counter
TP_REASON = "FRENZY_TP"                    # the resting-TP / software TP close reason (the FRENZY / FLATBASE label the close funnel knows)


def _fin(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def exhaustion_enabled(th) -> bool:
    """the master switch (exhaustion_short_enabled); anything but True = off."""
    return getattr(th, 'exhaustion_short_enabled', False) is True


def exhaustion_params(th) -> Dict[str, float]:
    """the values the engine USES: each config field when usable (inside the range exhaustion_config_errors accepts), else the study's
    value — a hand-edited bad value never widens the rule. Never raises."""
    out = {}
    for k, d in DEFAULTS.items():
        v = _fin(getattr(th, FIELDS[k], d))
        out[k] = v if (v is not None and _RANGES[k][1](v)) else d
    out['rsi_period'] = int(out['rsi_period'])
    return out


def exhaustion_config_errors(th, base_leverage=20.0) -> list:
    """🪫 PUT /api/config refusals for the EXHAUSTION_SHORT fields (DECISION_LOG 578). The stop cap's exchange backstop at the sleeve's leverage
    (base_leverage × exhaustion_lev_mult) must sit ≥ MANUAL_BACKSTOP_MIN_GAP_PCT beyond the widest stop (exhaustion_sl_cap_pct)."""
    errs = []
    for k in ('exhaustion_short_enabled', 'resting_tp_exhaustion_enabled', 'momentum_long_entries_enabled', 'momentum_short_entries_enabled',
              'exhaustion_ssc_oivol_block_enabled'):
        if not isinstance(getattr(th, k, True), bool):
            errs.append(f"{k} must be true or false")
    vals = {}
    for k, (txt, ok) in _RANGES.items():
        v = _fin(getattr(th, FIELDS[k], DEFAULTS[k]))
        if v is None or not ok(v):
            errs.append(f"{FIELDS[k]} must be {txt}")
        vals[k] = v
    if not errs and vals.get('oivol_ssc_min') is not None and vals.get('ssc_max') is not None and vals['oivol_ssc_min'] >= vals['ssc_max']:
        errs.append(f"exhaustion_ssc_oivol_ssc_min {vals['oivol_ssc_min']:g} must be below exhaustion_spot_share_chg_max {vals['ssc_max']:g} "
                    "(else the EXH_SSC_OIVOL filter can never block — DECISION_LOG 582)")
    if not errs:
        try:
            from services.trading_engine import order_backstop_pct, MANUAL_BACKSTOP_MIN_GAP_PCT
            lev = max(1, int(round(float(base_leverage or 20.0) * vals['lev'])))
            bk = order_backstop_pct(th, EXHAUSTION_SHORT, leverage=lev)
            if bk - vals['sl_cap'] < MANUAL_BACKSTOP_MIN_GAP_PCT - 1e-9:
                errs.append(f"exhaustion_sl_cap_pct {vals['sl_cap']:g} % is too wide for EXHAUSTION_SHORT at {lev}×: its exchange backstop ({bk:g} %, "
                            f"capped by liquidation) must sit ≥ {MANUAL_BACKSTOP_MIN_GAP_PCT:g} % beyond it — lower exhaustion_lev_mult")
        except Exception:
            errs.append("exhaustion_sl_cap_pct: the backstop distance could not be checked")
    return errs


# ── series helpers (pure python; equal to the study's pandas / numpy code) ──────────────────────────────────────────────────────────────

def wilder(x: Sequence[float], n: int) -> List[float]:
    """pandas Series(x).ewm(alpha=1/n, adjust=False).mean(): y0 = x0, y_t = (1 − α) y_{t−1} + α x_t."""
    a = 1.0 / n
    out, y = [], None
    for v in x:
        y = v if y is None else (1.0 - a) * y + a * v
        out.append(y)
    return out


def rsi(c: Sequence[float], n: int = 14) -> List[Optional[float]]:
    """frenzy_ta_build.rsi: Wilder RSI; None for the first n values."""
    if not c:
        return []
    d = [0.0] + [c[i] - c[i - 1] for i in range(1, len(c))]
    up = wilder([max(x, 0.0) for x in d], n)
    dn = wilder([max(-x, 0.0) for x in d], n)
    out = []
    for i, (u, w) in enumerate(zip(up, dn)):
        if i < n:
            out.append(None)
        elif w == 0:
            out.append(100.0)
        else:
            out.append(100.0 - 100.0 / (1.0 + u / w))
    return out


def atr(h: Sequence[float], l: Sequence[float], c: Sequence[float], n: int = 14) -> List[Optional[float]]:
    """frenzy_ta_build.atr: Wilder ATR, first TR = h − l; None for the first n values."""
    if not c:
        return []
    tr = []
    for i in range(len(c)):
        pc = c[i - 1] if i else c[0]
        tr.append(max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc)))
    a = wilder(tr, n)
    return [None if i < n else v for i, v in enumerate(a)]


def grid5(rows, upto_ms) -> Optional[Tuple[list, list, list, list, list, list]]:
    """CLOSED 5m rows → (t, o, h, l, c, qv) on a continuous 5m grid (frenzy_ta_build.load: a missing bar = flat at the previous close, zero
    volume). rows = [open_ms, o, h, l, c, v(, …, quote volume at index 7)] — raw Binance klines carry the quote volume at index 7; a row
    without it (the engine's [t, o, h, l, c, v] rows) gets qv = None (callers that need volume fail closed). Only bars with open + 5 min ≤
    upto_ms are kept. None on bad / empty / off-grid rows."""
    try:
        rr = []
        for r in rows or []:
            t0 = int(r[0])
            if t0 % BAR_MS or t0 + BAR_MS > int(upto_ms):
                continue
            q = _fin(r[7]) if len(r) > 7 else None
            rr.append((t0, float(r[1]), float(r[2]), float(r[3]), float(r[4]), q))
        if not rr:
            return None
        rr.sort(key=lambda x: x[0])
        by = {}
        for x in rr:
            by[x[0]] = x
        t_first, t_last = rr[0][0], rr[-1][0]
        T, O, H, L, C, Q = [], [], [], [], [], []
        prev_c = None
        for t0 in range(t_first, t_last + 1, BAR_MS):
            x = by.get(t0)
            if x is None:
                T.append(t0); O.append(prev_c); H.append(prev_c); L.append(prev_c); C.append(prev_c); Q.append(0.0)
                continue
            if not all(v > 0 and math.isfinite(v) for v in x[1:5]):
                return None
            T.append(t0); O.append(x[1]); H.append(x[2]); L.append(x[3]); C.append(x[4]); Q.append(x[5])
            prev_c = x[4]
        return T, O, H, L, C, Q
    except (TypeError, ValueError, IndexError):
        return None


def agg15(t, o, h, l, c, q=None):
    """ta_wl_features.agg on the 5m grid: buckets by open // 15 min → (t15, o15, h15, l15, c15, q15) (q15 None when q is None)."""
    T, O, H, L, C, Q = [], [], [], [], [], []
    for i in range(len(t)):
        g = t[i] // M15_MS * M15_MS
        if not T or T[-1] != g:
            T.append(g); O.append(o[i]); H.append(h[i]); L.append(l[i]); C.append(c[i]); Q.append(q[i] if q is not None else None)
        else:
            H[-1] = max(H[-1], h[i]); L[-1] = min(L[-1], l[i]); C[-1] = c[i]
            if q is not None:
                Q[-1] = (Q[-1] or 0.0) + (q[i] or 0.0)
    return T, O, H, L, C, (Q if q is not None else None)


def _k15_at(t15, T) -> Optional[int]:
    """index of the 15m bar that CLOSES at T (open T − 15 min) — None when absent (the window does not reach T)."""
    want = int(T) - M15_MS
    if not t15 or t15[-1] != want:
        return None
    return len(t15) - 1


def rsi_cross(rows, T, th) -> Dict:
    """the SIGNAL on CLOSED 5m rows (the engine's FRENZY window, no volume needed): the 15m bar closing at T crossed DOWN through the level —
    RSI(prev) > level ≥ RSI(now). → dict(cross, rsi_prev, rsi, close). cross False (+ None values) on unreadable rows. Never raises."""
    out = dict(cross=False, rsi_prev=None, rsi=None, close=None)
    try:
        if int(T) % M15_MS:
            return out
        g = grid5(rows, T)
        if g is None:
            return out
        t, o, h, l, c, _ = g
        t15, _o, _h, _l, c15, _ = agg15(t, o, h, l, c)
        k = _k15_at(t15, T)
        p = exhaustion_params(th)
        if k is None or k < 1:
            return out
        r = rsi(c15, p['rsi_period'])
        a, b = r[k - 1], r[k]
        out.update(rsi_prev=a, rsi=b, close=c15[k])
        out['cross'] = bool(a is not None and b is not None and a > p['rsi_level'] and b <= p['rsi_level'])
        return out
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return dict(cross=False, rsi_prev=None, rsi=None, close=None)


def spot_share_chg(spot_rows, T, p1, p24) -> Tuple[Optional[float], int, int]:
    """(spot_share_chg, n bars in the last hour, n bars in the 24 h window) — spot_rows = [open_ms, …, quote volume at index 7] raw spot
    klines, or [open_ms, qv] pairs (the parity fixture). None when either window is short (< 3 / < 80 bars) or volumes are unreadable."""
    try:
        s1 = s24 = 0.0
        n1 = n24 = 0
        T = int(T)
        for r in spot_rows or []:
            t0 = int(r[0])
            q = _fin(r[7] if len(r) > 7 else r[1])
            if q is None:
                return None, 0, 0
            if T - HOUR_MS <= t0 <= T - M15_MS:
                s1 += q; n1 += 1
            elif T - 25 * HOUR_MS <= t0 < T - HOUR_MS:
                s24 += q; n24 += 1
        if n1 < SPOT_MIN_1H or n24 < SPOT_MIN_24H:
            return None, n1, n24
        if not (s1 + p1 > 0 and s24 + p24 > 0):
            return None, n1, n24
        return s1 / (s1 + p1) - s24 / (s24 + p24), n1, n24
    except (TypeError, ValueError, IndexError):
        return None, 0, 0


def features(rows_qv, spot_rows, T, on_ms, has_spot, th) -> Dict:
    """every input of the rule at the signal close T from RAW perp 5m klines (quote volume at index 7) and RAW spot 15m klines.
    → dict(ok, why, rsi_prev, rsi, cross, close, atr, atr_pct, vol24, vol_decay, spot_share_chg, spot_n1, spot_n24, q5, has_spot).
    ok False + why = the window cannot score the rule (fail-closed). spot_share_chg None when has_spot is False / unscorable. Never raises."""
    out = dict(ok=False, why="no data", rsi_prev=None, rsi=None, cross=False, close=None, atr=None, atr_pct=None, vol24=None, vol_decay=None,
               spot_share_chg=None, spot_n1=0, spot_n24=0, q5=None, has_spot=bool(has_spot))
    try:
        T = int(T)
        p = exhaustion_params(th)
        g = grid5(rows_qv, T)
        if g is None:
            out['why'] = "perp 5m klines unreadable"
            return out
        t, o, h, l, c, q = g
        if any(x is None for x in q):
            out['why'] = "perp 5m quote volume missing"
            return out
        j = len(t) - 1
        if t[j] + BAR_MS != T:
            out['why'] = "the 5m window does not reach the signal close"
            return out
        t15, o15, h15, l15, c15, _ = agg15(t, o, h, l, c)
        k = _k15_at(t15, T)
        if k is None or k < 1:
            out['why'] = "the 15m signal bar is not in the window"
            return out
        r = rsi(c15, p['rsi_period'])
        a = atr(h15, l15, c15, 14)
        out.update(rsi_prev=r[k - 1], rsi=r[k], close=c[j], atr=a[k])
        out['cross'] = bool(r[k - 1] is not None and r[k] is not None and r[k - 1] > p['rsi_level'] and r[k] <= p['rsi_level'])
        if a[k] is None or not a[k] > 0:
            out['why'] = "ATR undefined"
            return out
        out['atr_pct'] = a[k] / c[j] * 100.0
        if j < VOL24_BARS - 1:
            out['why'] = "fewer than 288 5m bars (24 h volume)"
            return out
        out['vol24'] = sum(q[j - VOL24_BARS + 1: j + 1])
        q12 = sorted(q[j - ROLL1H_BARS + 1: j + 1])
        out['q5'] = (q12[5] + q12[6]) / 2.0                      # rolling(12).median at j (12 values)
        # vol_decay — roll1h(j) ÷ the running max of roll1h from the spike bar's open (ta_wl_features: ion = searchsorted(t, on))
        on = int(on_ms)
        if on < t[0] or on > t[j]:
            out['why'] = "the spike bar is not in the 5m window"
            return out
        ion = (on - t[0] + BAR_MS - 1) // BAR_MS                  # searchsorted(t, on, 'left') on the grid
        roll = [None] * len(t)
        run = 0.0
        for i in range(len(t)):
            run += q[i]
            if i >= ROLL1H_BARS:
                run -= q[i - ROLL1H_BARS]
            if i >= ROLL1H_BARS - 1:
                roll[i] = run
        pk = max((roll[i] or 0.0) for i in range(ion, j + 1))
        out['vol_decay'] = (roll[j] / pk) if (roll[j] is not None and pk > 0) else None
        # spot share change — p1 = roll1h(j); p24 = perp quote volume of the bars opened in [T − 25 h, T − 1 h)
        if T - 25 * HOUR_MS < t[0]:
            out['why'] = "the 5m window does not hold 25 h"
            return out
        jp = (T - 25 * HOUR_MS - t[0] + BAR_MS - 1) // BAR_MS
        p24 = sum(q[jp: j - ROLL1H_BARS + 1])
        if has_spot and roll[j] is not None:
            ssc, n1, n24 = spot_share_chg(spot_rows, T, roll[j], p24)
            out.update(spot_share_chg=ssc, spot_n1=n1, spot_n24=n24)
        out.update(ok=True, why="")
        return out
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        out.update(ok=False, why="feature computation failed")
        return out


def decide(f: Dict, th) -> Tuple[bool, Optional[str], str]:
    """(trade, block code, reason) for a features() dict. Order: data → the signal (re-read on the quote-volume klines) → 24 h volume → spot
    market → spot-share scorable → vol_decay → spot-share change. Fail-closed everywhere."""
    p = exhaustion_params(th)
    if not f or not f.get('ok'):
        return False, B_NO_DATA, f"inputs unreadable: {(f or {}).get('why') or 'no data'} (fail-closed)"
    if not f.get('cross'):
        return False, B_NO_SIGNAL, (f"RSI re-read on the exchange klines: {_r(f.get('rsi_prev'))} → {_r(f.get('rsi'))} — no cross below "
                                    f"{p['rsi_level']:g}")
    if f.get('vol24') is None or f['vol24'] < p['min_vol']:
        return False, B_VOL24, f"24 h volume ${(f.get('vol24') or 0) / 1e6:.1f}M < ${p['min_vol'] / 1e6:.0f}M"
    if not f.get('has_spot'):
        return False, B_NO_SPOT, "no Binance SPOT USDT market — unscored (the study never kept these)"
    if f.get('spot_share_chg') is None:
        return False, B_NO_DATA, (f"spot share unscorable ({f.get('spot_n1', 0)} spot bars in the last hour, {f.get('spot_n24', 0)} in the "
                                  f"24 h window; need {SPOT_MIN_1H} / {SPOT_MIN_24H}) (fail-closed)")
    vd = f.get('vol_decay')
    if vd is None or vd < p['vol_decay_min']:
        return False, B_VOL_DECAY, f"vol_decay {_r(vd, 3)} < {p['vol_decay_min']:g}"
    if not f['spot_share_chg'] < p['ssc_max']:
        return False, B_SPOT_SHARE, f"spot_share_chg {f['spot_share_chg']:+.4f} ≥ {p['ssc_max']:+g}"
    return True, None, (f"RSI {_r(f.get('rsi_prev'))} → {_r(f.get('rsi'))} · vol_decay {_r(vd, 3)} · spot_share_chg {f['spot_share_chg']:+.4f} · "
                        f"ATR {_r(f.get('atr_pct'), 2)} %")


def _r(v, d=1) -> str:
    x = _fin(v)
    return "n/a" if x is None else f"{x:.{d}f}"


def in_window(spike_ms, T, th) -> bool:
    """the signal close T lies in [spike close, spike close + exhaustion_max_hours_after_spike]."""
    try:
        return int(spike_ms) <= int(T) <= int(spike_ms) + int(round(exhaustion_params(th)['max_h'] * HOUR_MS))
    except (TypeError, ValueError):
        return False


# ── 🧯 EXH_SSC_OIVOL entry filter (DECISION_LOG 582 — operator, DECLARED OVERRIDE) ──────────────────────────────────────────────────
# BLOCK a signal the rule passed when BOTH (a) spot_share_chg > exhaustion_ssc_oivol_ssc_min (−0.0366; the rule itself needs < −0.0066, so
# the block band is −0.0366 < ssc < −0.0066 — the spot share falls only a little) AND (b) oi_vol = OI_usd ÷ vol24 > exhaustion_ssc_oivol_oivol_min
# (0.128 — open interest is large against the day's turnover). Study reports/EXH_BADDAY_2026-11-18.md (builder sq/build.py L95-110, prep.py L21):
#   OI_usd  Binance metrics sum_open_interest_value of the latest 5m row whose create_time (the period START) is ≤ T − 5 min and ≥ T − 30 min,
#           T = the decision time (the signal 15m bar close). Live: GET /futures/data/openInterestHist (period 5m) → sumOpenInterestValue; the
#           latest row with T − 30 min ≤ timestamp ≤ T − 5 min — the API `timestamp` EQUALS the metrics create_time (BTCUSDT 11-17 12:00 … 12:20: identical
#           sumOpenInterestValue at identical stamps, checked live 2026-11-18), so this is the study's exact window (create_time ≤ T − 5 min, ≤ 30 min old).
#   vol24   the rule's own 24 h quote volume (features()['vol24'] — the 288 closed 5m bars ending at the last closed bar = the study's vol24_k).
# Unreadable OI (error, empty, stale, non-positive) → FAIL-OPEN (trade), counted EXH_SSC_OIVOL_UNREAD. Strict cuts (> / >), as in the study.
OI_LAG_MS = 5 * 60_000                    # (582 review, checked live 11-18: openInterestHist `timestamp` == metrics create_time, the period START)
OI_MAX_AGE_MS = 30 * 60_000
OI_LIMIT = 12                              # rows asked for (1 h of 5m periods ending at T) — a weight-0 data endpoint


def ssc_oivol_enabled(th) -> bool:
    """the EXH_SSC_OIVOL switch (exhaustion_ssc_oivol_block_enabled); anything but True = off."""
    return getattr(th, 'exhaustion_ssc_oivol_block_enabled', False) is True


def oi_pick(rows, T) -> Tuple[Optional[float], Optional[int]]:
    """(OI_usd, its timestamp ms) of the latest openInterestHist row with T − 30 min ≤ timestamp ≤ T − 5 min (the API timestamp = the metrics
    create_time — the study's window). rows = [{'timestamp': ms, 'sumOpenInterestValue': '…'}, …] (any order). (None, None) when no usable row
    (empty / stale / unreadable / non-positive). Never raises."""
    best = None
    try:
        T = int(T)
        for r in rows or []:
            try:
                ts = int(r.get('timestamp'))
                v = _fin(r.get('sumOpenInterestValue'))
            except (TypeError, ValueError, AttributeError):
                continue
            if v is None or v <= 0 or ts > T - OI_LAG_MS or ts < T - OI_MAX_AGE_MS:
                continue
            if best is None or ts > best[1]:
                best = (v, ts)
    except (TypeError, ValueError):
        return None, None
    return best if best is not None else (None, None)


def oi_vol(oi_usd, vol24) -> Optional[float]:
    """OI_usd ÷ the 24 h quote volume; None on a bad / non-positive input."""
    o, v = _fin(oi_usd), _fin(vol24)
    if o is None or v is None or o <= 0 or v <= 0:
        return None
    return o / v


def ssc_oivol_block(ssc, oi_usd, vol24, th) -> Tuple[bool, bool, Optional[float]]:
    """(block, unread, oi_vol) — block = spot_share_chg > oivol_ssc_min AND oi_vol > oivol_min (strict). unread = the OI ratio (or the
    spot-share change) is unreadable → never a block (FAIL-OPEN; the caller counts EXH_SSC_OIVOL_UNREAD when the filter is on). The switch
    is the caller's (the ratio is stamped whether or not the filter is on). Never raises."""
    p = exhaustion_params(th)
    ov = oi_vol(oi_usd, vol24)
    s = _fin(ssc)
    if ov is None or s is None:
        return False, True, ov
    return bool(s > p['oivol_ssc_min'] and ov > p['oivol_min']), False, ov


# ── EXIT (paper = live) ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def exit_levels(atr_pct, th) -> Tuple[Optional[float], Optional[float]]:
    """(TP %, SL %) price distances from the fill: min(tp_atr × ATR%, tp_cap) and min(sl_atr × ATR%, sl_cap). (None, None) on bad ATR."""
    a = _fin(atr_pct)
    if a is None or a <= 0:
        return None, None
    p = exhaustion_params(th)
    return round(min(p['tp_atr'] * a, p['tp_cap']), 6), round(min(p['sl_atr'] * a, p['sl_cap']), 6)


def row_levels(row, th) -> Tuple[float, float]:
    """the (TP %, SL %) THIS row runs — its stamps (entry_exh_tp_pct / entry_exh_sl_pct), restart-proof; a missing / bad stamp → the cap
    (the widest level the rule allows: the study's cap, never wider). Works on an Order or a cache dict."""
    g = (lambda k: row.get(k)) if isinstance(row, dict) else (lambda k: getattr(row, k, None))
    p = exhaustion_params(th)
    tp, sl = _fin(g('entry_exh_tp_pct')), _fin(g('entry_exh_sl_pct'))
    tp = tp if (tp is not None and 0 < tp <= p['tp_cap'] + 1e-9) else p['tp_cap']
    sl = sl if (sl is not None and 0 < sl <= p['sl_cap'] + 1e-9) else p['sl_cap']
    return tp, sl


def short_move_pct(entry_px, price) -> Optional[float]:
    e, x = _fin(entry_px), _fin(price)
    if not (e and e > 0 and x and x > 0):
        return None
    return (1.0 - x / e) * 100.0


def exhaustion_exit_for(entry_px, price, held_minutes, th, row=None, resting_tp=False) -> Tuple[bool, str, Optional[float]]:
    """EXHAUSTION_SHORT's own exit → (close, reason, line) on PRICE levels from the fill (ssd_capped: stop first): price ≥ fill × (1 + SL %)
    → STOP_LOSS · price ≤ fill × (1 − TP %) → FRENZY_TP (resting_tp=True → the resting maker limit books it; the engine's resting-TP gate is
    the safety net) · held ≥ max hold → MAX_HOLD_TIME. The levels come from the row (row_levels)."""
    p = exhaustion_params(th)
    tp, sl = row_levels(row or {}, th)
    try:
        mv = short_move_pct(entry_px, price)
        if mv is not None and mv <= -sl + 1e-9:
            return True, "STOP_LOSS", -sl
        if mv is not None and mv >= tp - 1e-9 and not resting_tp:
            return True, TP_REASON, tp
        if held_minutes is not None and float(held_minutes) >= p['max_hold_h'] * 60.0:
            return True, "MAX_HOLD_TIME", (mv if mv is not None else -sl)
        return False, "MAX_HOLD_TIME", -sl
    except (TypeError, ValueError):
        return False, "MAX_HOLD_TIME", None


def max_hold_minutes(th) -> int:
    return int(round(exhaustion_params(th)['max_hold_h'] * 60))


def rtp_net_tp(entry_px, entry_fee_pct, maker_fee_rate, tp_pct) -> Optional[float]:
    """the NET take profit whose maker limit rests at fill × (1 − tp %): 100·m − entry fee % − the maker fee on the exit notional (the
    services.flatbase.rtp_net_tp formula with the row's own TP). None on bad input."""
    e, ef, mk, t = _fin(entry_px), _fin(entry_fee_pct), _fin(maker_fee_rate), _fin(tp_pct)
    if not (e and e > 0 and t and t > 0) or ef is None or mk is None or ef < 0 or mk < 0:
        return None
    m = t / 100.0
    net = 100.0 * m - ef - mk * 100.0 * (1.0 - m)
    return round(net, 6) if net > 0 else None


# ── slippage model + the revert gate ────────────────────────────────────────────────────────────────────────────────────────────────

def model_slip_pct(notional, q5) -> Optional[float]:
    """the study's fitted taker slippage, % of price, floored at 0: max(0, −0.042 + 0.276 · √(notional ÷ Q5)). None on bad input."""
    n, q = _fin(notional), _fin(q5)
    if n is None or n < 0 or q is None:
        return None
    return max(0.0, SLIP_A + SLIP_B * math.sqrt(n / max(q, 1.0)))


GATE_N, GATE_SLIP_MAX = 20, 0.25


def exit_era_579(tp_stamp, atr_stamp=None) -> bool:
    """🎯 (DECISION_LOG 579) True when the fill ran the exit override (TP min(20 × ATR, 10 %) / SL min(1.5 × ATR, 6 %)). The shipped exit
    stamps TP = min(3 × ATR, 5 %): never above 5 % and never above 3 × ATR; the override stamps either > 5 % (ATR > 0.25 %) or 20 × ATR
    (> 3 × ATR) — an exact split whenever both stamps are readable. Both stamps are rounded to 6 decimals, so the 3 × ATR test carries a
    relative + absolute margin (shipped: tp ≤ 3a + 3e-6 after rounding; override: tp = 20a ≫ 3a). A missing TP stamp → False (not counted).
    ⚠ Hard-codes the shipped 3× / 5 % exit: a FUTURE exit change must give this rule a new era (the scouts' registry pins go ⚠ STALE; the
    dashboard gate in main.py reads this same rule). Pure."""
    tp, a = _fin(tp_stamp), _fin(atr_stamp)
    if tp is None or tp <= 0:
        return False
    if tp > 5.0 + 1e-4:
        return True
    return a is not None and a > 0 and tp > 3.0 * a * (1 + 1e-4) + 1e-4
TAKER_EXIT_FREE = (TP_REASON,)              # a resting maker TP pays no slippage (the study's TP: maker strictly through by 1 tick)


def entry_leg_slip(model_slip, depth_mode=None, walk_pct=None) -> Optional[float]:
    """📖📐 (DECISION_LOG 584, deep review I2) the ENTRY leg of the re-pricing: an impact-mode paper fill already booked its modelled book walk
    (walk_pct) in its entry price → only max(0, model_slip − walk) is still owed; any other row (pre-584, sweep) → the full model_slip."""
    s, w = _fin(model_slip), _fin(walk_pct)
    if s is None:
        return None
    if str(depth_mode or '').strip().lower() == 'impact' and w is not None:
        return max(0.0, s - w)
    return s


def gate_net_pct(pnl_pct, close_reason, slip_pct, entry_slip=None) -> Optional[float]:
    """a paper fill re-priced with the modelled slippage: pnl % − slip (entry) − slip (exit, unless the maker TP booked it). entry_slip
    (584): the entry leg when it differs (entry_leg_slip); None → slip_pct."""
    p, s = _fin(pnl_pct), _fin(slip_pct)
    if p is None:
        return None
    s = s or 0.0
    e = _fin(entry_slip)
    e = s if e is None else e
    taker_exit = str(close_reason or '').upper().split(' ')[0] not in TAKER_EXIT_FREE
    return p - e - (s if taker_exit else 0.0)


def exhaustion_gate(fills) -> Tuple[str, str]:
    """🔒 (DECISION_LOG 578) the EXHAUSTION_SHORT revert gate on CLOSED bot fills (operator-closed excluded by the caller): fills = iterable of
    (opened_at, pnl_percentage, close_reason, model_slip_pct[, entry_leg_slip — DL 584: impact-mode fills owe only slip − walk at entry]). FROZEN: the first 20 fills by OPEN time, each re-priced with the modelled
    slippage (gate_net_pct) → avg ≤ 0 → OFF · mean modelled slippage > 0.25 % → OFF · else KEEP (paper; a live step is a new decision).
    COLLECTING before 20. Alert-only (the operator switches). Never raises."""
    try:
        f = sorted([(t[0], float(t[1]), str(t[2] or ''), _fin(t[3]), (_fin(t[4]) if len(t) > 4 else None)) for t in (fills or [])
                    if t[0] is not None and _fin(t[1]) is not None], key=lambda x: x[0])
        n = len(f)
        tail = " (alert-only · operator switches · DECISION_LOG 578)"
        rule = f"gate: first {GATE_N} fills re-priced with the modelled slippage avg ≤ 0 → OFF · mean modelled slippage > {GATE_SLIP_MAX:g} % → OFF"
        nets = [gate_net_pct(p, r, s, e) for _, p, r, s, e in f]
        sl = [s for _, _, _, s, _ in f if s is not None]
        if n < GATE_N:
            run = (f" · so far avg {sum(nets) / n:+.3f}% re-priced · mean slip {(sum(sl) / len(sl)) if sl else 0:.3f}%" if n else "")
            return "COLLECTING", f"⏳ {n}/{GATE_N} closed{run} — {rule}{tail}"
        first = nets[:GATE_N]
        fs = [s for _, _, _, s, _ in f[:GATE_N] if s is not None]
        avg = sum(first) / GATE_N
        ms = (sum(fs) / len(fs)) if fs else 0.0
        head = f"📋 first {GATE_N}: avg {avg:+.3f}% re-priced (raw {sum(t[1] for t in f[:GATE_N]) / GATE_N:+.3f}%) · mean modelled slip {ms:.3f}%"
        if avg <= 0:
            return "OFF", head + " → avg ≤ 0: switch EXHAUSTION_SHORT off" + tail
        if ms > GATE_SLIP_MAX:
            return "OFF", head + f" → slippage > {GATE_SLIP_MAX:g} %: switch EXHAUSTION_SHORT off" + tail
        return "KEEP", head + " → keep (paper)" + tail
    except Exception:
        return "COLLECTING", "gate unreadable"
