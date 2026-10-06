#!/usr/bin/env python3
"""📊 MOMENTUM-SHORT pair-volume ceiling shadow (operator, 2026-10-06) — the pure pieces behind the scout's MS_PVR_BLOCKED gate
(scripts/scout_revert_gates.py). READ-ONLY: public Binance klines + the operator's exports + reports/MASTER_POOL_stacked.csv;
never talks to the bot, never changes config.

WHY: the 2026-09-18 tightening (momentum_short_pair_vol_max 1.0 → 0.86) carried a pre-committed revert (kept side PVR < 0.86 below
70 % WR on N ≥ 15 → back to 1.0). That revert is OVERRIDDEN (operator, DECISION_LOG 226): the removed 0.86–1.0 band's own fills were
worse than the kept side, so 0.86 stays, judged by two frozen gates — ① revert to 1.0 if the refused-signal shadow's Cohort A
(0.86 ≤ PVR < 1.0, from 2026-10-06 18:00 UTC) reaches 15 signals with mean ≥ the kept side's mean over the same period · ② sleeve review
(full sleeve-kill checklist, no auto-kill) if the first 20 kept fills since 2026-09-18 12:00 are below breakeven WR 59 %.

PIECES
  · PVR (the engine's pair-volume ratio, services/indicators.calculate_indicators on the last 100 5m bars INCLUDING the forming bar):
        PVR = EWM(span 5, adjust=False)(volume)[-1] / SMA(pair_volume_lookback_bars = 20)(volume)[-1]
    rebuilt from public klines: 99 closed 5m bars + the forming bar's partial volume (closed 1m bars of that 5m bar + the running
    minute pro rata). Parity on live fills: the formula reproduces 90/90 master momentum-short entry_pair_volume_ratio stamps within
    0.01 (mean |Δ| 0.0012) at the scan second that fits — that second sits a median 35 s (IQR 25–50 s) before opened_at = the live
    signal→fill delay (ENTRY_DELAY_S). At a FIXED offset the scan second is unknown → ±0.03 typical (that is what the refusal-side
    classification has to live with; see classify_pvr).
  · The journal records a MOMENTUM_SHORT_PAIRVOL refusal as a BLOCK line (open_position gate; 5-min bucket, no price, no PVR). The
    refusal PVR is therefore rebuilt over the bucket on a 5-s grid; seconds whose PVR ≥ the live ceiling are the seconds consistent
    with the refusal. Cohort A = all consistent seconds < 1.0 (exactly what a revert to 1.0 re-admits) · B = all ≥ 1.0 (context) ·
    'A~' / 'B~' = straddles 1.0 (classified by the median, flagged) · none consistent = 'unk' (reconstruction miss, shown, not counted).
  · The live momentum-SHORT exit replica: see simulate_ms (spec + validation in its docstring).

Usage:  venv/bin/python scripts/ms_pvr_shadow.py --selftest
        venv/bin/python scripts/ms_pvr_shadow.py --validate 2026-07-29   # replica vs the live momentum-short fills since the Jul-29
                                                                         # realtime short runner (older eras ran other exit stacks)
"""
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
REPORTS = os.path.join(ROOT, "reports")
DL = os.path.expanduser("~/Downloads")
MIN, H, DAY = 60_000, 3_600_000, 86_400_000
B5 = 5 * MIN
PVR_BARS, PVR_SPAN, PVR_LOOKBACK = 100, 5, 20       # get_ohlcv(symbol, '5m', 100) · EWM span 5 · pair_volume_lookback_bars
PVR_CEIL, PVR_OLD = 0.86, 1.0                        # live ceiling · the revert target
PARITY = ("exit replica parity 2026-10-06 (--validate 2026-07-29): 33 live fills · mean |Δ| 0.046 pp · 88 % ≤ 0.10 pp · "
          "PVR rebuild 90/90 stamps ≤ 0.01")                    # re-run --validate and update on any exit-config change
ENTRY_DELAY_S = 35                                   # median live scan → fill delay (90 master momentum-short fills, see above)
GRID_S = 5                                           # refusal-bucket PVR grid
KEPT_FROM = "2026-09-18 12:00"                       # the kept-side fresh tally counts fills opened from here (operator)
SHADOW_FROM = "2026-10-06 18:00"                     # MS_PVR_BLOCKED registration: refusals (and gate ①'s kept side) count from here
# The Sep-18 kept-side revert (< 70 % WR on N ≥ 15 → 1.0) is OVERRIDDEN (operator, DECISION_LOG 226): 0.86 stays. FROZEN GATES:
REVERT_N = 15                                        # ① revert to 1.0 when Cohort A reaches 15 signals with mean ≥ the kept side's mean
KEPT_MIN_FRESH = 5                                   #   kept side = fills since SHADOW_FROM; fewer than 5 → the kept side since KEPT_FROM
REVIEW_N, BREAKEVEN_WR = 20, 59.0                    # ② sleeve review (full sleeve-kill checklist, no auto-kill) if the first 20 kept fills
                                                     #   since KEPT_FROM are below the breakeven WR (≈ 59 %)


# ═══════════════════════════════ PVR (pure) ═══════════════════════════════
def partial_volume(v1, b0, t):
    """volume of the forming 5m bar [b0, b0+5m) as the engine saw it at t: closed 1m bars fully, the running minute pro rata.
    v1 = {1m open_time ms: volume}."""
    part = 0.0
    for m in range(int(b0), int(b0) + B5, MIN):
        if m + MIN <= t:
            part += float(v1.get(m, 0.0))
        elif m <= t:
            part += float(v1.get(m, 0.0)) * (t - m) / MIN
    return part


def pvr_from_volumes(closed, part, span=PVR_SPAN, n=PVR_LOOKBACK):
    """the engine's ratio from the closed 5m volumes (oldest → newest, the bars before the forming one) + the forming bar's volume.
    → float, or NaN when any volume is missing / the SMA is 0 (the engine returns 1.0 for avg 0 — never seen on a traded pair)."""
    v = pd.Series(list(closed) + [float(part)], dtype=float)
    if v.isna().any() or len(v) < n:
        return float("nan")
    avg = float(v.rolling(n).mean().iloc[-1])
    if not avg > 0:
        return float("nan")
    return float(v.ewm(span=span, adjust=False).mean().iloc[-1] / avg)


def pvr_at(v5, v1, t, bars=PVR_BARS):
    """PVR at second t from {5m open: vol} + {1m open: vol} (the engine's 100-bar window incl. the forming bar)."""
    b0 = (int(t) // B5) * B5
    closed = [v5.get(b0 - k * B5, np.nan) for k in range(bars - 1, 0, -1)]
    return pvr_from_volumes(closed, partial_volume(v1, b0, t))


def classify_pvr(vals, ceil=PVR_CEIL, old=PVR_OLD):
    """refusal-bucket PVR grid → (cls, n_consistent, lo, med, hi). Consistent = PVR ≥ the live ceiling (the refusal happened at one of
    those seconds). 'A' all < old · 'B' all ≥ old · 'A~'/'B~' straddle (by the median) · 'unk' no consistent second."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    c = v[v >= ceil]
    if not len(c):
        return "unk", 0, float("nan"), float("nan"), float("nan")
    lo, med, hi = float(c.min()), float(np.median(c)), float(c.max())
    if hi < old:
        cls = "A"
    elif lo >= old:
        cls = "B"
    else:
        cls = "A~" if med < old else "B~"
    return cls, int(len(c)), lo, med, hi


# ═══════════════════════════════ PVR (market data) ═══════════════════════════════
def _get_json(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1 + i)


_VOL_MEM = {}


def fetch_volumes(pair, tf, t0, t1):
    """{open_time: base volume} of Binance USDⓈ-M klines with open_time in [t0, t1) (closed bars only). In-memory cache per run."""
    key = (pair, tf, int(t0), int(t1))
    if key in _VOL_MEM:
        return _VOL_MEM[key]
    step = {"1m": MIN, "5m": B5}[tf]
    now = int(time.time() * 1000)
    out, s = {}, int(t0)
    end = min(int(t1), now)
    while s < end:
        q = urllib.parse.urlencode(dict(symbol=pair, interval=tf, startTime=s, endTime=end - 1, limit=1500))
        r = _get_json(f"https://fapi.binance.com/fapi/v1/klines?{q}")
        if not r:
            break
        out.update({int(x[0]): float(x[5]) for x in r if int(x[0]) + step <= now})
        nxt = int(r[-1][0]) + step
        if nxt <= s or len(r) < 1500:
            break
        s = nxt
        time.sleep(0.05)
    _VOL_MEM[key] = out
    return out


def refusal_profile(pair, bucket_ms, ceil=PVR_CEIL):
    """the refusal bucket's PVR on a 5-s grid → dict(cls, n_ok, lo, med, hi, first_ok_ms) or dict(pending=…). first_ok_ms = the first
    grid second with PVR ≥ the ceiling (the earliest moment the refusal can have happened), last_ok_ms the latest."""
    b = int(bucket_ms)
    if b + B5 > int(time.time() * 1000):
        return dict(pending="bucket still forming")
    try:
        v5 = fetch_volumes(pair, "5m", b - PVR_BARS * B5, b)
        v1 = fetch_volumes(pair, "1m", b, b + B5)
    except Exception as e:
        return dict(pending=f"klines: {str(e)[:60]}")
    if len(v5) < PVR_BARS - 1 or len(v1) < 5:
        return dict(pending="volume history incomplete")
    grid = list(range(b, b + B5, GRID_S * 1000))
    vals = [pvr_at(v5, v1, t) for t in grid]
    cls, n_ok, lo, med, hi = classify_pvr(vals, ceil)
    ok = [t for t, v in zip(grid, vals) if np.isfinite(v) and v >= ceil]
    return dict(cls=cls, n_ok=n_ok, lo=_r(lo), med=_r(med), hi=_r(hi), first_ok_ms=(ok[0] if ok else None), last_ok_ms=(ok[-1] if ok else None))


def _r(x, k=3):
    return round(float(x), k) if x is not None and np.isfinite(x) else None


# ═══════════════════════════════ kept side (PVR < ceiling) fresh tally ═══════════════════════════════
KEPT_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "exit_price", "pnl_percentage",
             "pnl", "entry_pair_volume_ratio", "close_reason", "cell_multiplier_source", "stack_keep", "stack_block_reason", "era",
             "entry_atr_pct", "leverage", "entry_fee", "quantity", "entry_order_type")


def _ms_series(s):
    t = pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace("T", " ", regex=False), errors="coerce", format="mixed")
    return ((t - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("float")


def _export_ms(path):
    """export time from the file name (…_paper_YYYY-MM-DD_HH-MM-SS.csv), else the file's mtime."""
    try:
        from datetime import datetime, timezone
        b = os.path.basename(path).rsplit("_paper_", 1)[1][:19]
        return int(datetime.strptime(b, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    except Exception:
        return int(os.path.getmtime(path) * 1000)


def load_momentum_shorts(pool=None, exports=None):
    """every live MOMENTUM SHORT fill: master pool (stack_keep / era stamped) ∪ the orders exports, de-duplicated by
    (opened_at to the second, pair, direction) — the newest export wins, the master pool supplies stack_keep where it has the row.
    MANUAL / *_PROBE excluded. → DataFrame sorted by open time with o_ms, src ('master' | export file), in_master."""
    pool = pool or os.path.join(REPORTS, "MASTER_POOL_stacked.csv")
    fs = sorted(exports if exports is not None else glob.glob(os.path.join(DL, "scalpars_orders_paper_*.csv")), key=_export_ms)
    fr = []
    if os.path.exists(pool):
        fr.append(pd.read_csv(pool, low_memory=False, usecols=lambda c: c in KEPT_COLS).assign(_rank=-1, src="master"))
    for i, f in enumerate(fs):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in KEPT_COLS).assign(_rank=i, src=os.path.basename(f)))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(KEPT_COLS) + ["o_ms", "src", "in_master"])
    A = pd.concat(fr, ignore_index=True)
    for c in KEPT_COLS:
        if c not in A:
            A[c] = np.nan
    A = A[(A.direction.astype(str) == "SHORT") & (A.entry_strategy.fillna("MOMENTUM").astype(str) == "MOMENTUM")]
    A = A[~A.cell_multiplier_source.fillna("").astype(str).str.endswith("_PROBE")].copy()
    A["_k"] = A.opened_at.astype(str).str.replace(" ", "T", regex=False).str[:19]
    keys = set(zip(A.loc[A.src == "master", "_k"], A.loc[A.src == "master", "pair"]))
    keep_map = {(k, p): v for k, p, v in zip(A.loc[A.src == "master", "_k"], A.loc[A.src == "master", "pair"],
                                              A.loc[A.src == "master", "stack_keep"])}
    A = A.sort_values("_rank").drop_duplicates(["_k", "pair", "direction"], keep="last").copy()
    A["in_master"] = [(k, p) in keys for k, p in zip(A._k, A.pair)]
    A["stack_keep"] = [keep_map.get((k, p), True) for k, p in zip(A._k, A.pair)]   # export-only fills opened under today's stack
    A["o_ms"] = _ms_series(A.opened_at).values
    A = A[np.isfinite(A.o_ms)].sort_values("o_ms").reset_index(drop=True)
    A["o_ms"] = A.o_ms.astype("int64")
    A["pnl_percentage"] = pd.to_numeric(A.pnl_percentage, errors="coerce")
    A["pvr"] = pd.to_numeric(A.entry_pair_volume_ratio, errors="coerce")
    return A


def kept_tally(A, since=KEPT_FROM, ceil=PVR_CEIL, stack_only=True):
    """the kept side's fresh tally: CLOSED momentum-short fills opened ≥ since with PVR < ceil (stack_keep only by default = the batch
    review's count; stack_only=False = every live fill). → (rows DataFrame, dict(n, wins, wr, mean, sum_usd))."""
    t0 = int(pd.Timestamp(since).value // 1_000_000)
    f = A[(A.o_ms >= t0) & (A.pvr < ceil) & (A.status.astype(str) == "CLOSED") & A.pnl_percentage.notna()]
    if stack_only:
        f = f[f.stack_keep.astype(bool)]
    v = f.pnl_percentage.astype(float).values
    st = dict(n=int(len(v)), wins=int((v > 0).sum()), wr=(100.0 * (v > 0).mean() if len(v) else float("nan")),
              mean=(float(v.mean()) if len(v) else float("nan")), sum_usd=float(pd.to_numeric(f.pnl, errors="coerce").sum()))
    return f, st


def kept_ref(A, now_ms=None):
    """gate ①'s kept-side reference: fresh kept fills since SHADOW_FROM; fewer than 5 → the kept side since KEPT_FROM.
    → (stats dict, label)."""
    fresh = kept_tally(A, SHADOW_FROM)[1]
    if fresh["n"] >= KEPT_MIN_FRESH:
        return fresh, f"kept since {SHADOW_FROM[5:]}"
    return kept_tally(A, KEPT_FROM)[1], f"kept since {KEPT_FROM[5:]} (fresh {fresh['n']} < {KEPT_MIN_FRESH})"


def revert_gate(a_vals, kept_mean, n=REVERT_N):
    """① (DECISION_LOG 226, frozen): the first n Cohort-A signals (final prices, in signal order) with mean ≥ the kept side's mean →
    'fired' (revert to 1.0); below → 'holds'; fewer than n → 'insufficient'. Never 'consistent' by default."""
    a = [x for x in a_vals if x is not None and np.isfinite(x)][:n]
    if len(a) < n or kept_mean is None or not np.isfinite(kept_mean):
        return "insufficient"
    return "fired" if float(np.mean(a)) >= kept_mean else "holds"


def review_gate(first_vals, n=REVIEW_N, be_wr=BREAKEVEN_WR):
    """② (DECISION_LOG 226, frozen): the FIRST n kept fills since KEPT_FROM (keys frozen at the first time N reaches n) below the
    breakeven WR → 'review' (full sleeve-kill checklist, no auto-kill); ≥ → 'holds'; fewer than n → 'collecting'."""
    v = [x for x in first_vals if x is not None and np.isfinite(x)][:n]
    if len(v) < n:
        return "collecting"
    return "review" if 100.0 * sum(1 for x in v if x > 0) / n < be_wr else "holds"


# ═══════════════════════════════ the live momentum-SHORT exit replica (pure) ═══════════════════════════════
TAKER, MAKER = 0.00045, 0.00018                      # fee RATES
SCAN_S = 111.6                                       # scan cadence (as scripts/ml_exit_optimize.py) — cached EMAs / signal refresh
STALE_S = 600                                        # PairData older than this → None (EMA13 exit + signal-active off)
MS = dict(sl_base=0.70, sl_sig=1.00, sl_mult=1.5, sl_cap=1.20, c1_sl=-0.70,
          ladder=[(1.0, 0.25), (1.5, 0.30), (2.0, 0.40), (3.0, 0.60), (4.0, 0.80)],
          run_arm=0.40, run_n=0.5, run_frac=0.35, k=0.5, ema13=True, noexp_min=180, maxhold_min=1200)


def live_ms_settings(path=None):
    """the CURRENT momentum-short exit values from trading_config.json (top level or nested) over the MS defaults → (cfg, diffs)."""
    try:
        cfg = json.load(open(path or os.path.join(ROOT, "trading_config.json")))
    except Exception:
        cfg = {}

    def g(key):
        st = [cfg]
        while st:
            d = st.pop()
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                st.extend(v for v in d.values() if isinstance(v, dict))
        return None
    out, diffs = dict(MS), []
    m = {"run_arm": "runner_trail_short_arm_peak", "run_n": "runner_trail_short_atr_mult", "run_frac": "runner_trail_short_giveback_frac",
         "k": "runner_trail_short_k", "sl_mult": "sl_atr_multiplier", "noexp_min": "no_expansion_minutes", "maxhold_min": "max_holding_time_minutes"}
    for k, key in m.items():
        v = g(key)
        if v is not None:
            out[k] = float(v)
    v = g("sl_atr_widen_floor_pct")
    if v is not None:
        out["sl_cap"] = abs(float(v))
    lad = g("hard_tp_ladder_short")
    if lad:
        out["ladder"] = sorted((float(a), float(b)) for a, b in (x.split(":") for x in str(lad).split(",") if x.strip()))
    if g("hard_tp_enabled") is False:
        out["ladder"] = []
    for key, want in (("ema13_cross_exit_enabled", True), ("ema13_cross_exit_short_enabled", True), ("ema13_cross_requires_stack_flip", True),
                      ("runner_trail_short_enabled", True), ("runner_trail_short_use_atr", True), ("runner_trail_short_be_ratchet_enabled", False),
                      ("fl1_for_wide_sl_enabled", False), ("fl2_enabled", False), ("signal_lost_exit_enabled", False)):
        v = g(key)
        if v is not None and bool(v) != want:
            diffs.append(f"{key}={v} (replica assumes {want} — re-validate)")
    for k in MS:
        if k != "ladder" and out[k] != MS[k]:
            diffs.append(f"{k} live {out[k]} ≠ validated {MS[k]}")
    if out["ladder"] != MS["ladder"]:
        diffs.append(f"short ladder live {out['ladder']} ≠ validated {MS['ladder']}")
    return out, diffs


def ema_closed(closes, span):
    """EMA (adjust=False, as ta's EMAIndicator) over closed-bar closes → the value after each bar."""
    return pd.Series(closes, dtype=float).ewm(span=span, adjust=False).mean().values


def scan_table(k5_ot, k5_c, scan_ts, t, p):
    """cached scan state at each scan second: the forming 5m bar's close = the last print ≤ the scan (the engine's iloc[-1]).
    k5_ot / k5_c = CLOSED 5m bars (open time, close), ≥ ~100 bars before the first scan. → dict of arrays (ts, px, e5, e8, e13, e20)."""
    scan_ts = np.asarray(scan_ts, np.int64)
    j = np.clip(np.searchsorted(t, scan_ts, side="right") - 1, 0, len(t) - 1)
    px = np.asarray(p, float)[j]
    kb = np.searchsorted(k5_ot, scan_ts - B5, side="right") - 1          # last bar CLOSED at the scan (open + 5m ≤ scan)
    ok = kb >= 0
    kb = np.clip(kb, 0, len(k5_ot) - 1)
    out = dict(ts=scan_ts, px=px)
    for n in (5, 8, 13, 20):
        e = ema_closed(k5_c, n)
        a = 2.0 / (n + 1)
        out[f"e{n}"] = np.where(ok, a * px + (1 - a) * e[kb], np.nan)
    return out


def simulate_ms(t, p, entry, atr, sc, fee_in=TAKER, c1=False, cfg=None, taker=TAKER):
    """one momentum SHORT from the first print of the path (t ms, p prices from the fill on). sc = scan_table(...) (scans from the
    signal scan on). Per print, the live realtime order (services/trading_engine.check_realtime_stop_loss):
      ① C1 PATTERN_FIXED_SL  pnl ≤ −0.70 (C1 cell only)
      ② HARD_TP short ladder floor from the PREVIOUS print's peak (0.75/1.20/1.60/2.40/3.20 after 1.0/1.5/2/3/4)
      ③ EMA13_CROSS_EXIT     print > cached EMA13 ∧ cached EMA5 > EMA8 (strict) — suppressed once the previous peak ≥ 0.40
      ④ stop                 pnl ≤ stop + 0.01; stop = −max(1.00 if the cached signal is on (EMA5<EMA8 ∧ scan px<EMA20) else 0.70,
                             min(1.5·ATR, 1.20)) → STOP_LOSS_WIDE / STOP_LOSS
      ⑤ runner (realtime)    armed at peak ≥ 0.395 (this print included); floor = peak − min(0.5·ATR, 0.35·peak)
    and per second (the 1 Hz monitor, last print of each second):
      ⑥ K-trail              armed ∧ peak stretch > 0 ∧ stretch ≤ 0.5 × peak stretch (stretch = (cached EMA5 − px)/px·100, running max
                             since entry) → RUNNER_TRAIL L1
      ⑦ NO_EXPANSION at 180 min (timer resets while the cached signal is on) · MAX_HOLD 1200 min.
    Fees: P&L % = ((E − p)/E − fee_in − taker·p/E)·100. → dict(pnl, reason, i, t, peak) · reason OPEN_END when the path ends first."""
    c = dict(MS, **(cfg or {}))
    t = np.asarray(t, np.int64)
    p = np.asarray(p, float)
    if not len(t) or not np.isfinite(entry) or entry <= 0:
        return None
    try:
        a = float(atr)
    except (TypeError, ValueError):
        a = float("nan")
    has_atr = np.isfinite(a) and a > 0
    wide = min(c["sl_mult"] * a, c["sl_cap"]) if has_atr else 0.0
    sl_on, sl_off = -max(c["sl_sig"], wide), -max(c["sl_base"], wide)
    pnl = ((entry - p) / entry - fee_in - taker * p / entry) * 100
    sts = sc["ts"]
    si = np.searchsorted(sts, t, side="right") - 1
    sig = (sc["e5"] < sc["e8"]) & (sc["px"] < sc["e20"])
    flip = sc["e5"] > sc["e8"]
    last_sec = np.r_[(t[1:] // 1000) != (t[:-1] // 1000), True]
    lad = c["ladder"] or []
    t0 = int(t[0])
    peak, pk_st = 0.0, -np.inf
    noexp_at = t0 + c["noexp_min"] * MIN
    maxhold_at = t0 + c["maxhold_min"] * MIN

    def done(i, reason):
        return dict(pnl=float(pnl[i]), reason=reason, i=int(i), t=int(t[i]), peak=float(peak))
    for i in range(len(p)):
        x, s = float(pnl[i]), int(si[i])
        fresh = s >= 0 and (t[i] - sts[s]) <= STALE_S * 1000
        prev = peak
        if c1 and x <= c["c1_sl"]:
            return done(i, "PATTERN_FIXED_SL")
        if lad and prev >= lad[0][0]:
            if x <= max(tr - o for tr, o in lad if prev >= tr):
                return done(i, "HARD_TP_LADDER")
        if c["ema13"] and fresh and prev < c["run_arm"] and np.isfinite(sc["e13"][s]) and p[i] > sc["e13"][s] and flip[s]:
            return done(i, "EMA13_CROSS_EXIT")
        on = bool(fresh and sig[s])
        if x <= (sl_on if on else sl_off) + 0.01:
            return done(i, "STOP_LOSS_WIDE" if on else "STOP_LOSS")
        if x > peak:
            peak = x
        armed = peak >= c["run_arm"] - 0.005
        if armed and has_atr:
            if x <= peak - min(c["run_n"] * a, c["run_frac"] * peak):
                return done(i, "RUNNER_TRAIL")
        if last_sec[i]:
            if fresh and np.isfinite(sc["e5"][s]):
                st = (sc["e5"][s] - p[i]) / p[i] * 100
                pk_st = max(pk_st, st)
                if armed and pk_st > 0 and st <= c["k"] * pk_st:
                    return done(i, "RUNNER_TRAIL_K")
            if t[i] >= maxhold_at:
                return done(i, "MAX_HOLD")
            if t[i] >= noexp_at:
                if on:
                    noexp_at = int(t[i]) + c["noexp_min"] * MIN
                else:
                    return done(i, "NO_EXPANSION")
    return dict(pnl=float(pnl[-1]), reason="OPEN_END", i=len(p) - 1, t=int(t[-1]), peak=float(peak))


REASON_MAP = {"PATTERN_FIXED_SL": "PATTERN_FIXED_SL", "HARD_TP_LADDER": "HARD_TP_LADDER", "EMA13_CROSS_EXIT": "EMA13_CROSS_EXIT",
              "STOP_LOSS_WIDE": "STOP_LOSS_WIDE", "STOP_LOSS": "STOP_LOSS", "RUNNER_TRAIL": "RUNNER_TRAIL", "RUNNER_TRAIL_K": "RUNNER_TRAIL L1",
              "NO_EXPANSION": "NO_EXPANSION", "MAX_HOLD": "MAX_HOLD_TIME"}


def is_c1(range_pos, gap13_50, adx_delta):
    """the C1 SHORT cell (services/trading_engine._compute_pattern_c_match): range position ≤ 15 ∧ EMA13−EMA50 gap ≤ −0.50 ∧ ADX Δ ≥ 1.0."""
    try:
        return float(range_pos) <= 15 and float(gap13_50) <= -0.50 and float(adx_delta) >= 1.0
    except (TypeError, ValueError):
        return False


# ═══════════════════════════════ market-data drivers (path + scan EMAs + entry stamps) ═══════════════════════════════
def _R():
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)
    import scout_revert_gates as R
    return R


def last_print(pair, ts, k1=None):
    """the last trade price ≤ ts: aggTrades ticks where the day archive exists, else the close of the last 1m bar CLOSED by ts
    (never a print after ts). None when nothing is known."""
    R = _R()
    M = R._replica(None)
    d = M._day(pair, (int(ts) // DAY) * DAY)
    if d is not None:
        j = int(np.searchsorted(d[0], int(ts), side="right")) - 1
        if j >= 0 and int(ts) - int(d[0][j]) <= 10 * MIN:
            return float(d[1][j])
    if k1 is None:
        k1 = R.klines(pair, "1m", int(ts) - H, int(ts))
    c = k1[k1.open_time + MIN <= int(ts)] if k1 is not None else None
    return float(c.c.iloc[-1]) if c is not None and len(c) else None


def entry_stamps(pair, scan_ms):
    """the stamps a momentum short gets at the signal scan, rebuilt from public klines (closed 5m bars + the forming bar from 1m):
    ATR % (ta ATR(14) incl. the forming bar ÷ scan price — services/trading_engine.pair_entry_stamps), and the C1 cell legs
    (range position over 20 bars, EMA13−EMA50 gap %, ADX(14) Δ). → dict(atr, c1, px, range_pos, gap13_50, adx_delta) or None."""
    from ta.trend import ADXIndicator
    from ta.volatility import AverageTrueRange
    R = _R()
    scan_ms = int(scan_ms)
    b0 = (scan_ms // B5) * B5
    k5 = R.klines(pair, "5m", b0 - 99 * B5, b0)
    k1a = R.klines(pair, "1m", b0 - H, b0 + B5)
    if k5 is None or len(k5) < 60 or k1a is None:
        return None
    k1 = k1a[(k1a.open_time >= b0) & (k1a.open_time + MIN <= scan_ms)]   # NO LOOK-AHEAD: only 1m bars CLOSED by the scan
    lp = last_print(pair, scan_ms, k1a)                                   # + the last print ≤ the scan (the forming bar's close)
    if lp is None:
        return None
    hs, ls_ = [lp] + list(k1.h), [lp] + list(k1.l)
    fb = dict(open_time=b0, o=(float(k1.o.iloc[0]) if len(k1) else lp), h=float(max(hs)), l=float(min(ls_)), c=float(lp))
    d = pd.concat([k5, pd.DataFrame([fb])], ignore_index=True)
    px = float(d.c.iloc[-1])
    atr = float(AverageTrueRange(high=d.h, low=d.l, close=d.c, window=14).average_true_range().iloc[-1]) / px * 100
    adx = ADXIndicator(high=d.h, low=d.l, close=d.c, window=14).adx()
    e13, e50 = ema_closed(d.c.values, 13)[-1], ema_closed(d.c.values, 50)[-1]
    h20, l20 = float(d.h.iloc[-20:].max()), float(d.l.iloc[-20:].min())
    rp = (px - l20) / (h20 - l20) * 100 if h20 > l20 else float("nan")
    gap = (e13 - e50) / e50 * 100
    ad = float(adx.iloc[-1] - adx.iloc[-2])
    return dict(atr=_r(atr, 4), px=px, range_pos=_r(rp, 1), gap13_50=_r(gap, 4), adx_delta=_r(ad, 4), c1=is_c1(rp, gap, ad))


def price_ms(pair, entry_ms, scan_ms, atr, c1=False, fee_in=TAKER, entry_price=None, horizon_min=None, cfg=None):
    """the live momentum-SHORT exit from the first print ≥ entry_ms → dict(pct, how, src, reason) or dict(pending=…).
    Path = aggTrades ticks where the day archive exists (scripts/ml_exit_optimize.build_path), 1m o→l/h→c otherwise. entry_price
    None → the first print (taker); the live fill price for the replica validation."""
    R = _R()
    c = dict(MS, **(cfg or {}))
    hz = int((horizon_min or c["maxhold_min"]) * MIN)
    M = R._replica(None)
    try:
        k1 = R.klines(pair, "1m", entry_ms - H, entry_ms + hz + 5 * MIN)
        k5 = R.klines(pair, "5m", scan_ms - 120 * B5, entry_ms + hz + B5)
    except Exception as e:
        return dict(pending=f"klines: {str(e)[:60]}")
    if k1 is None or not len(k1) or k1.open_time.max() < entry_ms or k5 is None or len(k5) < 60:
        return dict(pending="no klines yet")
    try:
        R._inject_k1(M, pair, k1)
        bp = M.build_path(pair, int(entry_ms), hz)
        if bp is None or not len(bp[0]):
            return dict(pending="no path")
        t, p, src = bp
        scans = int(scan_ms) + (np.arange(0, int((t[-1] - scan_ms) / (SCAN_S * 1000)) + 2) * SCAN_S * 1000).astype(np.int64)
        # the scan's forming-bar close = the last print ≤ the scan: scans before the fill read the prints between the scan and the
        # fill window (pre-path), the first print only when nothing ≤ the scan is known
        lp = last_print(pair, int(scan_ms), k1) if int(scan_ms) < int(t[0]) else None
        tt, pp = (np.r_[np.int64(scan_ms), t], np.r_[lp, p]) if lp is not None else (t, p)
        sc = scan_table(k5.open_time.values.astype(np.int64), k5.c.values.astype(float), scans, tt, pp)
        E = float(entry_price) if entry_price is not None and np.isfinite(entry_price) else float(p[0])
        r = simulate_ms(t, p, E, atr, sc, fee_in, c1, cfg)
    except Exception as e:
        return dict(pending=f"path: {str(e)[:60]}")
    if r is None:
        return dict(pending="no path")
    if r["reason"] == "OPEN_END":
        if t[-1] < entry_ms + hz - 2 * MIN:
            return dict(pending="still running")
        r["reason"] = "horizon"
    return dict(pct=round(float(r["pnl"]), 3), how=f"{r['reason']} {int((r['t'] - entry_ms) / MIN)}m", src=src, reason=r["reason"])


def _c1_of(src):
    return "C1" in [x.strip().split("[")[0] for x in str(src or "").split("+")]


def validate(since=None, delay_s=ENTRY_DELAY_S, out_csv=None):
    """replica vs every CLOSED live momentum-short fill (master ∪ exports): live entry price + live entry fee rate, the stamped ATR,
    the live cell (C1 → the −0.70 fixed stop), signal scan = opened_at − delay_s. → DataFrame + printed parity (mean |Δ| pp)."""
    A = load_momentum_shorts()
    A = A[(A.status.astype(str) == "CLOSED") & A.pnl_percentage.notna()]
    if since:
        A = A[A.o_ms >= int(pd.Timestamp(since).value // 1_000_000)]
    rows = []
    for r in A.itertuples():
        E = pd.to_numeric(r.entry_price, errors="coerce")
        q = pd.to_numeric(r.quantity, errors="coerce")
        fe = pd.to_numeric(r.entry_fee, errors="coerce")
        fee_in = float(fe / (E * q)) if all(np.isfinite([E, q, fe])) and E * q > 0 else TAKER
        x = price_ms(r.pair, int(r.o_ms), int(r.o_ms) - delay_s * 1000, pd.to_numeric(r.entry_atr_pct, errors="coerce"),
                     _c1_of(r.cell_multiplier_source), fee_in, E)
        rows.append(dict(opened_at=r.opened_at, pair=r.pair, era=r.era, live=round(float(r.pnl_percentage), 3), live_reason=r.close_reason,
                         sim=x.get("pct"), sim_how=x.get("how", x.get("pending")), src=x.get("src"), fee_in=round(fee_in * 100, 4)))
    X = pd.DataFrame(rows)
    X["d"] = (X.sim - X.live).abs()
    ok = X[X.sim.notna()]
    print(f"momentum-short replica vs live: {len(ok)}/{len(X)} priced · mean |Δ| {ok.d.mean():.3f} pp · median {ok.d.median():.3f} · "
          f"≤0.10 pp {100 * (ok.d <= 0.10).mean():.0f} % · ≤0.25 {100 * (ok.d <= 0.25).mean():.0f} % · tick paths {int((ok.src == 'tick').sum())}")
    if out_csv:
        X.to_csv(out_csv, index=False)
    return X


def selftest():
    """synthetic checks of every pure piece (no network)."""
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    # PVR: flat volume → 1.0; a forming bar at 3× the norm lifts the EWM; the running minute counts pro rata
    chk(abs(pvr_from_volumes([100.0] * 99, 100.0) - 1.0) < 1e-12, "pvr: flat = 1.0")
    chk(pvr_from_volumes([100.0] * 99, 300.0) > 1.2, "pvr: a heavy forming bar lifts the ratio")
    chk(np.isnan(pvr_from_volumes([100.0] * 98 + [np.nan], 100.0)), "pvr: a missing bar → NaN, never a guess")
    v1 = {0: 60.0, MIN: 60.0, 2 * MIN: 60.0}
    chk(abs(partial_volume(v1, 0, 2 * MIN + 30_000) - 150.0) < 1e-9, "partial: 2 closed minutes + half the running one")
    v5 = {-k * B5: 100.0 for k in range(1, 100)}
    chk(pvr_at(v5, {m: 20.0 for m in range(0, B5, MIN)}, B5 - 1) < pvr_at(v5, {m: 200.0 for m in range(0, B5, MIN)}, B5 - 1), "pvr_at: grows with volume")
    # classification of a refusal bucket
    chk(classify_pvr([0.80, 0.87, 0.95])[0] == "A", "class A: consistent seconds all < 1.0")
    chk(classify_pvr([0.90, 1.05, 1.10])[0] == "B~", "class B~: straddles, median ≥ 1.0")
    chk(classify_pvr([0.88, 0.90, 1.02])[0] == "A~", "class A~: straddles, median < 1.0")
    chk(classify_pvr([1.0, 1.3])[0] == "B", "class B: 1.0 inclusive (the old ceiling blocked at ≥ 1.0)")
    chk(classify_pvr([0.5, 0.85])[0] == "unk", "class unk: no second consistent with the refusal")
    # kept-side revert + reading rule
    chk(revert_gate([0.5] * 14, -0.3) == "insufficient", "① < 15 A signals → insufficient (never 'consistent')")
    chk(revert_gate([-0.3] * 15, -0.3) == "fired", "① A mean = kept mean → FIRED (≥)")
    chk(revert_gate([-0.31] * 15 + [9.0] * 5, -0.3) == "holds", "① only the first 15 A signals count")
    chk(revert_gate([0.1] * 15, float("nan")) == "insufficient", "① no kept reference → insufficient")
    chk(review_gate([1.0] * 19) == "collecting", "② < 20 kept fills → collecting")
    chk(review_gate([1.0] * 11 + [-1.0] * 9) == "review", "② 55 % < 59 % → review")
    chk(review_gate([1.0] * 12 + [-1.0] * 8) == "holds", "② 60 % ≥ 59 % → holds")
    chk(is_c1(10, -0.6, 1.5) and not is_c1(16, -0.6, 1.5) and not is_c1(10, -0.4, 1.5) and not is_c1(10, -0.6, 0.9), "C1 legs")
    chk(_c1_of("C1+C4") and not _c1_of("W1+W2") and not _c1_of(None), "C1 from the cell label")
    # exit replica on synthetic paths (E = 100, short; scans every 111.6 s; EMAs from a flat 100 history)
    k5_ot = np.arange(-120, 0, dtype=np.int64) * B5
    k5_c = np.full(120, 100.0)

    def run(px, atr=0.5, c1=False, fee=TAKER, dt=1000):
        t = np.arange(len(px), dtype=np.int64) * dt
        sc = scan_table(k5_ot, k5_c, np.arange(0, 3) * int(SCAN_S * 1000), t, np.asarray(px, float))
        return simulate_ms(t, px, 100.0, atr, sc, fee, c1)
    r = run([100.0, 100.5, 100.8])
    chk(r["reason"] in ("STOP_LOSS", "EMA13_CROSS_EXIT") and r["i"] >= 1, "adverse move closes (EMA13 cross or stop)")
    r = run([100.0, 100.71], atr=0.3, c1=True)
    chk(r["reason"] == "PATTERN_FIXED_SL" and abs(r["pnl"] + 0.80) < 0.01, "C1: −0.70 fixed stop first in the chain")
    r = run([100.0, 99.5, 99.4, 99.7], atr=1.0)
    chk(r["reason"] == "RUNNER_TRAIL" and r["i"] == 3, "runner: peak +0.51 → floor peak − min(0.5, 0.35·peak) ≈ +0.33")
    r = run([100.0, 98.8, 99.2], atr=10.0)
    chk(r["reason"] == "HARD_TP_LADDER" and r["i"] == 2, "ladder: peak +1.11 ≥ 1.0 → floor 0.75")
    chk(run([100.0, 99.9])["reason"] == "OPEN_END", "open end")
    cfg, diffs = live_ms_settings()
    chk(set(cfg) == set(MS), "live settings keys")
    print(f"ms_pvr_shadow selftest OK — {ok} checks · live vs validated exit: {diffs or 'identical'}")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        i = sys.argv.index("--validate")
        since = sys.argv[i + 1] if len(sys.argv) > i + 1 and not sys.argv[i + 1].startswith("--") else None
        X = validate(since)
        with pd.option_context("display.width", 250, "display.max_rows", 500):
            print(X.to_string())
    elif "--selftest" in sys.argv:
        selftest()
