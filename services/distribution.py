"""📉 DISTRIBUTION_SHORT (2026-11-18, DECISION_LOG 586 — operator-approved PAPER run, DECLARED OVERRIDE: the study's verdict was OBSERVE) — pure
rules shared by the engine (services/trading_engine.py), the scout (scripts/scout_distribution_586.py) and the tests. No I/O.

The research it implements: reports/LEADS_DEEP_2026-11-18.md L1 (frozen rule `trig=VSW15 & top_pos_chg4h <= −0.01097`) on the UNIFIED_SHORT
pool (reports/UNIFIED_SHORT_2026-11-18.md; builder scripts/unified_short_build.py, signals scripts/frenzy_ta_build.family_signals). Every
definition is COPIED from that code and checked against study rows by tests/test_distribution_short.py (fixture
tests/fixtures/dist_parity.npz, built by scripts/dist_parity_fixture.py from the study's own data):
  universe  a FRENZY-universe episode tracked by EXHAUSTION (services/exhaustion.py — the same FRENZY flag, spike close → +72 h; the study's
            pool window is the spike bar … +864 5m bars, the latest spike owning the bar), 24 h quote volume ≥ $20M at the signal (base_ok).
  15m bars  the exchange's CLOSED 15m klines with their quote volume (kline field 7) — identical to the study's 5m grid grouped by open //
            15 min (open = first, high = max, low = min, close = last, quote volume = sum) on every bar the exchange printed.
  VSW15     on the CLOSED 15m candle that closes at T: vavg = the mean quote volume of the 20 candles BEFORE it (sh(rolling(20).mean())),
            vavg > 0, quote volume ≥ 5 × vavg, close < open (family_signals 'VSW' short = spk & (c < o)).
  POS 4 h   metrics sum_toptrader_long_short_ratio = GET /futures/data/topLongShortPositionRatio (period 5m) `longShortRatio` — the API
            `timestamp` is the SNAPSHOT time (never shifted). The vision metrics files label each row 5 min EARLY (create_time = snapshot − 5 min)
            from 2024-05 to 2026-11-15, and correctly before and after (DECISION_LOG 614, reports/OI_TS_AUDIT_2026-11-21.md). The study's
            "create_time ≤ T − 5 min" therefore read the snapshot at ≤ T — one 5m period fresher than live can see, NO look-ahead (aligned
            re-run: avg +0.883 → +0.824 %/trade, 7/7 half-years positive). Live reads the snapshot at ≤ T − 5 min (the 11-17 parity check
            fell after vision's fix, so it matched). km = the latest row with timestamp ≤ T − 5 min and ≥ T − 30 min (fresh); k4 = the latest row
            with timestamp ≤ ts(km) − 4 h and ts(km) − ts(k4) ≤ 4 h 15 min; chg = ratio(km) − ratio(k4) (unified_short_build metrics block).
            KEEP when chg ≤ −0.01097 (strict NaN → no trade, as the study's comparison).
  ATR(14)   Wilder ATR on the 15m candles at the signal candle (frenzy_ta_build.atr); atr_pct = ATR ÷ the signal close × 100 (study: ÷ the entry
            bar's open — the same price up to the first tick).
  vol24     Σ quote volume of the 96 15m candles ending at the signal candle = the study's 288 5m bars.
  ENTRY     SHORT at market right after T (the study: the next 5m open).
  EXIT B    (operator) = EXHAUSTION's live exit (DL 579): TP = min(20 × ATR%, 10 %) below the fill (resting maker) · SL = min(1.5 × ATR%, 6 %)
            above the fill · 24 h. PRICE levels from the fill, stop first (unified_short_build.exits 'LIVE').
Every helper is total: bad input → a fail-closed answer, never an exception."""
import math
from typing import Dict, List, Optional, Sequence, Tuple

DISTRIBUTION_SHORT = "DISTRIBUTION_SHORT"
DISTRIBUTION_CLASS = (DISTRIBUTION_SHORT,)

M5_MS = 300_000
M15_MS = 900_000
HOUR_MS = 3_600_000
VSW_LOOKBACK = 20
VOL24_BARS = 96                          # 15m candles = 24 h
K15_LIMIT = 499                          # 15m candles read on a pre-screened candle (weight 2): ATR warm-up + 24 h volume + the 20-candle mean
POS_LIMIT = 60                           # topLongShortPositionRatio 5m rows read (5 h; a weight-0 data endpoint, its own 1,000 / 5 min IP cap)
POS_LAG_MS = 5 * 60_000
POS_MAX_AGE_MS = 30 * 60_000
POS_WIN_MS = 4 * HOUR_MS
POS_WIN_TOL_MS = 15 * 60_000
PRESCREEN_FRAC = 0.6                     # the engine reads the exchange klines only for a red candle whose FRENZY-window volume proxy ≥ 0.6 × the multiple
ENTRY_MAX_LATE_S = 300

DEFAULTS = dict(pos_chg_max=-0.01097, vsw_mult=5.0, min_vol=20e6, max_h=72.0, tp_atr=20.0, tp_cap=10.0, sl_atr=1.5, sl_cap=6.0,
                max_hold_h=24.0, invest=0.5, lev=0.25, rsi1h_min=74.07,
                crash_dd=-20.0, crash_fund_min=0.00005)   # 📉🌡 (DECISION_LOG 651) DIST_CRASH_FUNDING cuts   # 🔒 (586 review I1) a missing / bad value → the armed cut, never off
FIELDS = dict(pos_chg_max='distribution_pos_chg_4h_max', vsw_mult='distribution_vsw_vol_mult', min_vol='distribution_min_vol_usd',
              max_h='distribution_max_hours_after_spike', tp_atr='distribution_tp_atr_mult', tp_cap='distribution_tp_cap_pct',
              sl_atr='distribution_sl_atr_mult', sl_cap='distribution_sl_cap_pct', max_hold_h='distribution_max_hold_hours',
              invest='distribution_invest_mult', lev='distribution_lev_mult', rsi1h_min='distribution_rsi1h_min',
              crash_dd='distribution_crash_btc_dd_pct', crash_fund_min='distribution_crash_funding_min')
_RANGES = dict(pos_chg_max=("a number ≥ −1 and < 0 (a fall of the top-trader position ratio)", lambda x: -1.0 <= x < 0),
               vsw_mult=("a number ≥ 1 and ≤ 50 (× the 20-candle mean)", lambda x: 1.0 <= x <= 50.0),
               min_vol=("a number ≥ 0 (USD)", lambda x: x >= 0),
               max_h=("a number > 0 and ≤ 100 (hours — EXHAUSTION's tracking window bounds it)", lambda x: 0 < x <= 100.0),
               tp_atr=("a number > 0 and ≤ 50", lambda x: 0 < x <= 50), tp_cap=("a number > 0 and < 50 (price %)", lambda x: 0 < x < 50),
               sl_atr=("a number > 0 and ≤ 20", lambda x: 0 < x <= 20), sl_cap=("a number > 0 and < 50 (price %)", lambda x: 0 < x < 50),
               max_hold_h=("a number > 0 and ≤ 240 (hours)", lambda x: 0 < x <= 240),
               invest=("a number > 0 and ≤ 1", lambda x: 0 < x <= 1.0), lev=("a number > 0 and ≤ 1", lambda x: 0 < x <= 1.0),
               rsi1h_min=("a number ≥ 0 and < 100 (0 = off)", lambda x: 0 <= x < 100),
               crash_dd=("a number ≥ −90 and ≤ −1 (BTC % below its 30-day high)", lambda x: -90.0 <= x <= -1.0),
               crash_fund_min=("a number ≥ −0.01 and ≤ 0.01 (funding rate per settlement, a fraction: 0.00005 = 0.005 %)", lambda x: -0.01 <= x <= 0.01))

ST_OPENED, ST_BLOCKED, ST_REFUSED = "OPENED", "BLOCKED", "REFUSED"
B_NO_DATA, B_NOT_VSW, B_VOL24, B_POS_CHG, B_POS_UNREAD = "DIST_NO_DATA", "DIST_NOT_VSW", "DIST_VOL24_LOW", "DIST_POS_CHG", "DIST_POS_UNREAD"
B_PAIR_HELD, B_OPEN_REFUSED, B_FAILED, B_LATE, B_BOOK_FULL = "DIST_PAIR_HELD", "DIST_OPEN_REFUSED", "DIST_FAILED", "DIST_LATE", "DIST_BOOK_FULL"
B_RSI1H, B_RSI1H_UNREAD = "DIST_RSI1H", "DIST_RSI1H_UNREAD"   # 🔒 (DECISION_LOG 586, declared override) the RSI1h ≥ 74.07 keep filter · its FAIL-CLOSED counter
B_CRASH_FUND, B_CRASH_FUND_UNREAD = "DIST_CRASH_FUNDING", "DIST_CRASH_FUNDING_UNREAD"   # 📉🌡 (DECISION_LOG 651) the block · its fail-OPEN counter
TP_REASON = "FRENZY_TP"


def _fin(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def distribution_enabled(th) -> bool:
    return getattr(th, 'distribution_short_enabled', False) is True


def distribution_params(th) -> Dict[str, float]:
    """the values the engine USES: each config field when inside its range, else the study's value (a bad value never widens the rule)."""
    out = {}
    for k, d in DEFAULTS.items():
        v = _fin(getattr(th, FIELDS[k], d))
        out[k] = v if (v is not None and _RANGES[k][1](v)) else d
    return out


def distribution_config_errors(th, base_leverage=20.0) -> list:
    """📉 PUT /api/config refusals for the DISTRIBUTION_SHORT fields (DECISION_LOG 586) — the stop cap's backstop at the sleeve's leverage
    must sit ≥ MANUAL_BACKSTOP_MIN_GAP_PCT beyond the widest stop."""
    errs = []
    for k in ('distribution_short_enabled', 'resting_tp_distribution_enabled', 'distribution_crash_funding_block_enabled'):
        if not isinstance(getattr(th, k, True), bool):
            errs.append(f"{k} must be true or false")
    vals = {}
    for k, (txt, ok) in _RANGES.items():
        v = _fin(getattr(th, FIELDS[k], DEFAULTS[k]))
        if v is None or not ok(v):
            errs.append(f"{FIELDS[k]} must be {txt}")
        vals[k] = v
    if not errs:
        try:
            from services.trading_engine import order_backstop_pct, MANUAL_BACKSTOP_MIN_GAP_PCT
            lev = max(1, int(round(float(base_leverage or 20.0) * vals['lev'])))
            bk = order_backstop_pct(th, DISTRIBUTION_SHORT, leverage=lev)
            if bk - vals['sl_cap'] < MANUAL_BACKSTOP_MIN_GAP_PCT - 1e-9:
                errs.append(f"distribution_sl_cap_pct {vals['sl_cap']:g} % is too wide for DISTRIBUTION_SHORT at {lev}×: its exchange backstop "
                            f"({bk:g} %, capped by liquidation) must sit ≥ {MANUAL_BACKSTOP_MIN_GAP_PCT:g} % beyond it — lower distribution_lev_mult")
        except Exception:
            errs.append("distribution_sl_cap_pct: the backstop distance could not be checked")
    if not errs:   # 🔒 (586 review I5) DISTRIBUTION reads EXHAUSTION's tracked episodes — its window can never outlive EXHAUSTION's sweep
        try:
            from services.exhaustion import exhaustion_params
            xm = exhaustion_params(th)['max_h']
            if vals['max_h'] > xm + 1e-9:
                errs.append(f"distribution_max_hours_after_spike {vals['max_h']:g} must be ≤ exhaustion_max_hours_after_spike {xm:g} "
                            "(DISTRIBUTION reads EXHAUSTION's tracked episodes, ended after that window)")
        except Exception:
            errs.append("distribution_max_hours_after_spike: EXHAUSTION's window could not be checked")
    return errs


def in_window(spike_ms, T, th) -> bool:
    try:
        return int(spike_ms) <= int(T) <= int(spike_ms) + int(round(distribution_params(th)['max_h'] * HOUR_MS))
    except (TypeError, ValueError):
        return False


# ── the pre-screen on the FRENZY window (no read) ────────────────────────────────────────────────────────────────────────────────────

def prescreen(closed_rows, T, th) -> Optional[Dict]:
    """the engine's FRENZY window (CLOSED 5m rows [open, o, h, l, c, v], base volume) → the 15m candle closing at T: red (exact OHLC) AND a
    volume PROXY (Σ base volume × typical price per 15m) ≥ PRESCREEN_FRAC × the multiple × its 20-candle mean → dict(red, proxy_mult) else
    None. Only a pre-screen: the decision re-reads the exchange's quote volume (decide on features()). Never raises."""
    try:
        T = int(T)
        if T % M15_MS:
            return None
        by = {}
        for r in closed_rows or []:
            t0 = int(r[0])
            if t0 + M5_MS > T:
                continue
            g = t0 // M15_MS * M15_MS
            h, l, c, v = float(r[2]), float(r[3]), float(r[4]), float(r[5])
            q = v * (h + l + c) / 3.0
            b = by.get(g)
            if b is None:
                by[g] = [t0, float(r[1]), c, q]
            else:
                if t0 < b[0]:
                    b[0], b[1] = t0, float(r[1])
                else:
                    b[2] = c
                b[3] += q
        k = T - M15_MS
        if k not in by:
            return None
        prev = [by[k - i * M15_MS][3] for i in range(1, VSW_LOOKBACK + 1) if (k - i * M15_MS) in by]
        if len(prev) < VSW_LOOKBACK:
            return None
        avg = sum(prev) / len(prev)
        o, c, q = by[k][1], by[k][2], by[k][3]
        if not (avg > 0 and c < o):
            return None
        pm = q / avg
        if pm < PRESCREEN_FRAC * distribution_params(th)['vsw_mult']:
            return None
        return dict(red=True, proxy_mult=pm)
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return None


# ── the rule on the exchange's 15m klines ────────────────────────────────────────────────────────────────────────────────────────────

def _wilder(x, n):
    a, y, out = 1.0 / n, None, []
    for v in x:
        y = v if y is None else (1.0 - a) * y + a * v
        out.append(y)
    return out


def atr15(h, l, c, n=14) -> List[Optional[float]]:
    """frenzy_ta_build.atr: Wilder ATR, first TR = h − l; None for the first n values."""
    tr = []
    for i in range(len(c)):
        pc = c[i - 1] if i else c[0]
        tr.append(max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc)))
    a = _wilder(tr, n)
    return [None if i < n else v for i, v in enumerate(a)]


def _ema(x, n):
    """pandas ewm(span=n, adjust=False): y0 = x0."""
    a, y, out = 2.0 / (n + 1), None, []
    for v in x:
        y = v if y is None else (1.0 - a) * y + a * v
        out.append(y)
    return out


def di_diff(h, l, c, n=14) -> List[Optional[float]]:
    """frenzy_ta_build.adx_di → +DI − −DI (Wilder, the first bar's DM = 0, first TR = h − l)."""
    pdm, mdm, tr = [0.0], [0.0], [h[0] - l[0]] if h else []
    for i in range(1, len(c)):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    a, p_, m_ = _wilder(tr, n), _wilder(pdm, n), _wilder(mdm, n)
    return [(100.0 * (p_[i] - m_[i]) / a[i]) if a[i] > 0 else None for i in range(len(c))]


def observe_stamps(t, h, l, c) -> Dict:
    """📉 (DECISION_LOG 586, OBSERVE-ONLY — no filter reads them) the study's rsi1h · di_diff1h · gap15 at the signal candle (the last of
    the CLOSED 15m candles t/h/l/c): 1h bars = the 15m candles grouped by open // 1 h, COMPLETE hours only (4 candles), the last one closed
    ≤ T; RSI(14) Wilder and +DI − −DI (Wilder 14) on them (unified_short_build: R1 / adx_di on the hourly bars, the last closed hour);
    gap15 = (EMA13 − EMA50) ÷ close × 100 on the 15m closes (ta_wl_features.ema_gap; None before 50 candles). Never raises."""
    out = dict(rsi1h=None, di_diff1h=None, gap15=None)
    try:
        if len(c) > 50:
            e1, e2 = _ema(c, 13), _ema(c, 50)
            out['gap15'] = (e1[-1] - e2[-1]) / c[-1] * 100.0
        hb = {}
        for i in range(len(t)):
            g = int(t[i]) // HOUR_MS * HOUR_MS
            b = hb.setdefault(g, [0, None, -1e300, 1e300, None])
            b[0] += 1; b[2] = max(b[2], h[i]); b[3] = min(b[3], l[i]); b[4] = c[i]
        ks = sorted(k for k, b in hb.items() if b[0] == 4)
        if len(ks) > 15:
            H = [hb[k][2] for k in ks]; L = [hb[k][3] for k in ks]; C = [hb[k][4] for k in ks]
            from services.exhaustion import rsi as _rsi
            out['rsi1h'] = _rsi(C, 14)[-1]
            out['di_diff1h'] = di_diff(H, L, C, 14)[-1]
        return out
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        return dict(rsi1h=None, di_diff1h=None, gap15=None)


def features(rows15, T, th) -> Dict:
    """RAW 15m klines (open ms, o, h, l, c, v, close ms, QUOTE VOLUME, …) of CLOSED candles ≤ T → dict(ok, why, red, bar_ret_pct, vsw_mult,
    vsw, close, atr_pct, vol24, q5). The candle closing at T must be the last one. Never raises."""
    out = dict(ok=False, why="no data", red=False, bar_ret_pct=None, vsw_mult=None, vsw=False, close=None, atr_pct=None, vol24=None, q5=None,
               rsi1h=None, di_diff1h=None, gap15=None)
    try:
        T = int(T)
        rr = []
        for r in rows15 or []:
            t0 = int(r[0])
            if t0 % M15_MS or t0 + M15_MS > T:
                continue
            rr.append((t0, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[7])))
        rr.sort()
        if not rr or rr[-1][0] + M15_MS != T:
            out['why'] = "the 15m klines do not reach the signal close"
            return out
        if any(not (x[1] > 0 and x[2] > 0 and x[3] > 0 and x[4] > 0) or x[5] < 0 for x in rr):
            out['why'] = "15m klines unreadable"
            return out
        t = [x[0] for x in rr]; o = [x[1] for x in rr]; h = [x[2] for x in rr]; l = [x[3] for x in rr]; c = [x[4] for x in rr]; q = [x[5] for x in rr]
        j = len(rr) - 1
        if j < VSW_LOOKBACK:
            out['why'] = "fewer than 21 15m candles"
            return out
        avg = sum(q[j - VSW_LOOKBACK:j]) / VSW_LOOKBACK
        out['red'] = c[j] < o[j]
        out['bar_ret_pct'] = (c[j] / o[j] - 1.0) * 100.0
        out['close'] = c[j]
        out['vsw_mult'] = (q[j] / avg) if avg > 0 else None
        out['vsw'] = bool(avg > 0 and q[j] >= distribution_params(th)['vsw_mult'] * avg and c[j] < o[j])
        out.update(observe_stamps(t, h, l, c))   # 📉 (586) observe-only stamps (the DIST_WL RSI1h ≥ 74.07 line)
        a = atr15(h, l, c, 14)
        if a[j] is None or not a[j] > 0:
            out['why'] = "ATR undefined"
            return out
        out['atr_pct'] = a[j] / c[j] * 100.0
        if j < VOL24_BARS - 1 or t[j] - t[j - VOL24_BARS + 1] != (VOL24_BARS - 1) * M15_MS:
            out['why'] = "fewer than 96 continuous 15m candles (24 h volume)"
            return out
        out['vol24'] = sum(q[j - VOL24_BARS + 1: j + 1])
        out['q5'] = sorted(q[j - 3: j + 1])[1:3]
        out['q5'] = (out['q5'][0] + out['q5'][1]) / 2.0 / 3.0       # ≈ the median 5m quote volume of the last hour (the slip model's Q5)
        out.update(ok=True, why="")
        return out
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        out.update(ok=False, why="feature computation failed")
        return out


def pos_chg_4h(rows, T) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """topLongShortPositionRatio rows [{'timestamp': ms, 'longShortRatio': '…'}, …] (any order) → (chg, ratio_now, ts_now, ts_4h) with the
    study's window (module doc). (None, …) when unreadable / stale. Never raises."""
    try:
        T = int(T)
        pts = []
        for r in rows or []:
            try:
                ts = int(r.get('timestamp'))
                v = _fin(r.get('longShortRatio'))
            except (TypeError, ValueError, AttributeError):
                continue
            if v is not None and v > 0:
                pts.append((ts, v))
        pts.sort()
        now = [p for p in pts if p[0] <= T - POS_LAG_MS]
        if not now or T - now[-1][0] > POS_MAX_AGE_MS:
            return None, None, None, None
        tc, vc = now[-1]
        old = [p for p in pts if p[0] <= tc - POS_WIN_MS]
        if not old or tc - old[-1][0] > POS_WIN_MS + POS_WIN_TOL_MS:
            return None, vc, tc, None
        return vc - old[-1][1], vc, tc, old[-1][0]
    except (TypeError, ValueError):
        return None, None, None, None


def decide(f: Dict, chg, th) -> Tuple[bool, Optional[str], str]:
    """(trade, block code, reason): data → VSW15 on the exchange klines → 24 h volume → the top-trader position change. Fail-closed."""
    p = distribution_params(th)
    if not f or not f.get('ok'):
        return False, B_NO_DATA, f"inputs unreadable: {(f or {}).get('why') or 'no data'} (fail-closed)"
    if not f.get('vsw'):
        return False, B_NOT_VSW, (f"exchange quote volume {_r(f.get('vsw_mult'), 2)}× the 20-candle mean (need ≥ {p['vsw_mult']:g}×, red) — "
                                  f"the pre-screen's proxy passed, the exchange klines do not")
    if f.get('vol24') is None or f['vol24'] < p['min_vol']:
        return False, B_VOL24, f"24 h volume ${(f.get('vol24') or 0) / 1e6:.1f}M < ${p['min_vol'] / 1e6:.0f}M"
    c = _fin(chg)
    if c is None:
        return False, B_POS_UNREAD, "top-trader position ratio unreadable / stale (fail-closed)"
    if not c <= p['pos_chg_max']:
        return False, B_POS_CHG, f"top-trader position L/S 4 h change {c:+.4f} > {p['pos_chg_max']:+g}"
    return True, None, (f"red VSW {_r(f.get('vsw_mult'), 1)}× ({_r(f.get('bar_ret_pct'), 2)} %) · top-trader position L/S 4 h {c:+.4f} · "
                        f"ATR {_r(f.get('atr_pct'), 2)} %")


def rsi1h_block(rsi1h, th) -> Tuple[bool, Optional[str], str]:
    """🔒 (DECISION_LOG 586, operator 11-18, DECLARED OVERRIDE — reports/DIST_WL_2026-11-18.md) the entry filter, judged AFTER the L1 rule
    passed: KEEP only when the 1h RSI(14) at entry (features()['rsi1h'] = observe_stamps — the same value stamped entry_dist_rsi1h) ≥
    distribution_rsi1h_min (74.07). → (blocked, code, reason). distribution_rsi1h_min 0 → off (never blocks). An unreadable RSI (< 16
    complete 1h bars in the window) → BLOCKED DIST_RSI1H_UNREAD: FAIL-CLOSED — the cut defines the sleeve (the opposite of EXH's OI). Pure."""
    m = distribution_params(th)['rsi1h_min']   # (review I1) missing / invalid / out of range → 74.07; OFF only by an explicit 0
    if not m > 0:
        return False, None, ""
    r = _fin(rsi1h)
    if r is None:
        return True, B_RSI1H_UNREAD, f"1h RSI unreadable (too few closed 1h bars) — the RSI1h ≥ {m:g} filter fails closed"
    if r < m:
        return True, B_RSI1H, f"1h RSI {r:.2f} < {m:g} — the coin is no longer strong at the red candle (DL 586 RSI1h filter)"
    return False, None, f"1h RSI {r:.2f} ≥ {m:g}"


# ── 📉🌡 DIST_CRASH_FUNDING (DECISION_LOG 651) ──────────────────────────────────────────────────────────────────────────────────────
# Judged AFTER the RSI1h filter, on a signal every rule passed: BLOCK when BTC is in a crash — its last CLOSED daily close is ≤ crash_dd
# (−20 %) below the highest daily high of the 30 closed days ending with it (one read per UTC day; = reports/DIST_DEEP_DIVE_2026-11-27's
# "prior-day close vs the 30-day high", known at the start of the day) — AND the coin's funding at the decision (premiumIndex, the same read
# stamped entry_funding_rate) is ≥ crash_fund_min (+0.005 % = the 4 h-coin base; 8 h coins at the 0.01 % base are ≥ it too: longs still pay). Evidence: reports/DIST_DEEP_DIVE_2026-11-27.md
# (backtest 43 · 23 % · −2.01 %/fill, both blocks, 8 / 9 episodes negative, family-wise p 0.026; live batch 4 / 4 blocked were losers).
# Either input unreadable → FAIL-OPEN (trade, counted DIST_CRASH_FUNDING_UNREAD): the filter removes a loser cohort, it does not define DIST.

def crash_funding_enabled(th) -> bool:
    """the DIST_CRASH_FUNDING switch (distribution_crash_funding_block_enabled); anything but True = off."""
    return getattr(th, 'distribution_crash_funding_block_enabled', False) is True


def btc_dd30(daily_rows, now_ms) -> Optional[float]:
    """BTC % below its 30-day high from 1d klines [[open_ms, o, h, l, c, …], …]: the last CLOSED day's close ÷ the max high of the 30 closed
    days ending with it − 1, in %. None when < 30 closed days or unreadable. Pure."""
    try:
        day = 86_400_000
        rows = sorted({int(r[0]): r for r in daily_rows or [] if int(r[0]) + day <= int(now_ms)}.values(), key=lambda r: int(r[0]))
        if len(rows) < 30:
            return None
        last = rows[-30:]
        if int(last[-1][0]) - int(last[0][0]) != 29 * day:   # (651 review) 30 CONTIGUOUS days, never a gapped window
            return None
        hi = max(float(r[2]) for r in last)
        c = float(last[-1][4])
        if not (hi > 0 and c > 0):
            return None
        return (c / hi - 1.0) * 100.0
    except (TypeError, ValueError, IndexError):
        return None


def crash_funding_block(dd_pct, funding, th) -> Tuple[bool, bool, str]:
    """(blocked, unread, reason). The switch is checked here (off → never blocks, never unread). Never raises."""
    try:
        if not crash_funding_enabled(th):
            return False, False, ""
        p = distribution_params(th)
        d, fr = _fin(dd_pct), _fin(funding)
        if d is None:
            return False, True, "BTC 30-day drawdown unreadable — DIST_CRASH_FUNDING skipped (fail-open)"
        if d > p['crash_dd']:
            return False, False, ""
        if fr is None:
            return False, True, f"BTC {d:.1f} % below its 30 d high but funding unreadable — DIST_CRASH_FUNDING skipped (fail-open)"
        if fr >= p['crash_fund_min'] - 1e-12:
            return True, False, (f"BTC {d:.1f} % below its 30 d high (≤ {p['crash_dd']:g} %) and funding {fr * 100:+.4f} % ≥ "
                                 f"{p['crash_fund_min'] * 100:+.4f} % (longs still pay) — DL 651 crash-funding filter")
        return False, False, f"crash market but funding {fr * 100:+.4f} % < {p['crash_fund_min'] * 100:+.4f} % (a squeeze — kept)"
    except Exception:   # (651 review) an internal error trades but is COUNTED (unread), never invisible
        return False, True, "crash-funding judge error (fail-open)"


def _r(v, d=1) -> str:
    x = _fin(v)
    return "n/a" if x is None else f"{x:.{d}f}"


# ── EXIT B (paper = live) ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def exit_levels(atr_pct, th) -> Tuple[Optional[float], Optional[float]]:
    a = _fin(atr_pct)
    if a is None or a <= 0:
        return None, None
    p = distribution_params(th)
    return round(min(p['tp_atr'] * a, p['tp_cap']), 6), round(min(p['sl_atr'] * a, p['sl_cap']), 6)


def row_levels(row, th) -> Tuple[float, float]:
    """the (TP %, SL %) THIS row runs — its stamps (entry_dist_tp_pct / entry_dist_sl_pct), restart-proof; missing / bad → the cap."""
    g = (lambda k: row.get(k)) if isinstance(row, dict) else (lambda k: getattr(row, k, None))
    p = distribution_params(th)
    tp, sl = _fin(g('entry_dist_tp_pct')), _fin(g('entry_dist_sl_pct'))
    tp = tp if (tp is not None and 0 < tp <= p['tp_cap'] + 1e-9) else p['tp_cap']
    sl = sl if (sl is not None and 0 < sl <= p['sl_cap'] + 1e-9) else p['sl_cap']
    return tp, sl


def distribution_exit_for(entry_px, price, held_minutes, th, row=None, resting_tp=False) -> Tuple[bool, str, Optional[float]]:
    """(close, reason, line) on PRICE levels from the fill, stop first: price ≥ fill × (1 + SL) → STOP_LOSS · price ≤ fill × (1 − TP) →
    FRENZY_TP (resting_tp=True → the maker limit books it) · held ≥ max hold → MAX_HOLD_TIME."""
    from services.exhaustion import short_move_pct
    p = distribution_params(th)
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
    return int(round(distribution_params(th)['max_hold_h'] * 60))


# ── the revert gate (shared shape with DL 585): one-sided 80 % UPPER bound of the mean net, day-block bootstrap ───────────────────────

GATE_N, GATE_DAYS, GATE_Q, GATE_B = 30, 10, 0.80, 4000


def day_block_upper(nets_days, q=GATE_Q, B=GATE_B, seed=586) -> Optional[float]:
    """nets_days = [(net %, UTC day key)] → the q-quantile of the bootstrap mean when whole DAYS are resampled with replacement (the
    one-sided q UPPER bound of the mean). None on < 2 days. Pure (seeded)."""
    import random
    by = {}
    for v, d in nets_days or ():
        x = _fin(v)
        if x is not None:
            by.setdefault(d, []).append(x)
    days = list(by.values())
    if len(days) < 2:
        return None
    rnd = random.Random(seed)
    means = []
    for _ in range(int(B)):
        s = n = 0
        for _k in range(len(days)):
            g = days[rnd.randrange(len(days))]
            s += sum(g); n += len(g)
        means.append(s / n)
    means.sort()
    return means[min(len(means) - 1, int(math.floor(q * len(means))))]


def upper_bound_gate(fills, label, switch_key, dl) -> Tuple[str, str]:
    """🔒 frozen alert-only gate: fills = [(opened_at datetime, net %)] (closed, operator-closed out) by OPEN time → COLLECTING until the
    first ≥ GATE_N fills span ≥ GATE_DAYS days (the cohort = the first GATE_N, extended by open time until it spans GATE_DAYS); then the one-sided 80 % UPPER bound of their mean net (day-block bootstrap) < 0 → OFF · else
    KEEP. Pure."""
    try:
        f = sorted([(t, float(v)) for t, v in (fills or []) if t is not None and _fin(v) is not None], key=lambda x: x[0])
        tail = f" (alert-only · operator switches · DECISION_LOG {dl})"
        if len(f) < GATE_N:
            run = f" · so far avg {sum(v for _, v in f) / len(f):+.3f}%" if f else ""
            return "COLLECTING", f"⏳ {len(f)}/{GATE_N} closed{run} — gate: first {GATE_N} spanning ≥ {GATE_DAYS} days, 80 % upper bound < 0 → OFF{tail}"
        n = next((k for k in range(GATE_N, len(f) + 1) if (f[k - 1][0] - f[0][0]).total_seconds() / 86400.0 >= GATE_DAYS), None)
        if n is None:
            span = (f[-1][0] - f[0][0]).total_seconds() / 86400.0
            return "COLLECTING", f"⏳ {len(f)} closed span {span:.1f} days < {GATE_DAYS} — the cohort grows until it spans {GATE_DAYS} days{tail}"
        cut = f[:n]
        span = (cut[-1][0] - cut[0][0]).total_seconds() / 86400.0
        avg = sum(v for _, v in cut) / n
        ub = day_block_upper([(v, t.date().isoformat()) for t, v in cut])
        head = f"📋 first {n} ({span:.1f} days): avg {avg:+.3f}% · 80 % upper bound {ub:+.3f}%" if ub is not None else f"📋 first {n}: avg {avg:+.3f}%"
        if ub is not None and ub < 0:
            return "OFF", head + f" → < 0: switch {switch_key} off ({label})" + tail
        return "KEEP", head + " → keep (paper)" + tail
    except Exception:
        return "COLLECTING", "gate unreadable"
