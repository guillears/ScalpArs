#!/usr/bin/env python3
"""🧯 Scout: STOP_SLIP measures what a STOP exit really costs (2026-10-07). The operator asked for it after
reports/FRENZY_ENTRY_LATENCY_STUDY_2026-10-07.md §A5 found the 9 live paper FRENZY / WIDE stops filled ≈ 0.18 % below the last print at
closed_at, while the backtests assume an exit slip of 0.02 – 0.10. OBSERVE ONLY: it never changes config, never trades and never talks to
the bot. scripts/opportunity_scout.py calls it every run inside its own try/except, so it can never take the scout down. It uses public
Binance data only: fapi klines + aggTrades and the data.binance.vision daily aggTrades archive. The year studies' tick cache
(reports/backtest_cache/ticks{,_q}/) is READ, never written.

HOW A PAPER STOP EXIT IS EXECUTED (services/, by function name — line numbers drift; read-only here)
  feed     websocket_tracker.WebSocketTracker subscribes <pair>@trade, i.e. every raw public trade. _handle_message takes each trade's
           price "p" and AWAITS the price callback inline, so the next WS message is not read until the callback returns.
  trigger  trading_engine.realtime_stop_loss_callback → TradingEngine.check_realtime_stop_loss(pair, price) runs on EVERY trade. It works
           out the net P&L % at that print (entry fee + a taker exit fee on the print).
           Momentum stack (MOMENTUM / FLIP / SPIKE_*): fires at pnl ≤ effective_sl + 0.01 (STOP_LOSS / STOP_LOSS_WIDE / BREAKEVEN_EXIT /
           SPIKE_LOCK). effective_sl = the confidence level's stop_loss (signal_active_sl → STOP_LOSS_WIDE). It widens to
           −sl_atr_multiplier × entry ATR when that is wider, the WHOLE result is capped at sl_atr_widen_floor_pct, then the quiet-pair SL
           (_quiet_sl_for) applies if it is wider still. SPIKE_* stops are fixed (update_orders_cache 'stop_loss').
           FL2 deep stop / FL1 emergency backstop: + 0.01 on the realtime path, no epsilon on the polling path (update_open_positions).
           Bull-Run / SURGE / FRENZY: _bullrun_exit_for / surge_short_exit_for / services.frenzy.frenzy_exit_for fire at pnl ≤ line (no
           epsilon). RH hard stop: services.recovery_hold.rh_exit at pnl ≤ hard + 0.01.
           A slower polling path (update_open_positions, the tracker's last price, REST fallback after 90 s of WS silence) does the same.
  fill     close_position(db, order, current_price = THAT TRADE'S PRICE, reason) waits on the global _close_lock (one close at a time,
           all pairs) → _close_position_locked: a DB re-read, then actual_exit_price = current_price. In the paper branch, a reason in the
           'urgent' list (FRENZY strategies, the RH stop class, or a reason starting STOP_LOSS / BREAKEVEN_EXIT / FL_EMERGENCY_SL /
           FL_DEEP_STOP / BR_ / MANUAL_ / OVERRIDE_ …) closes as taker at that same price. SPIKE_SL, PATTERN_FIXED_SL, FLIP_STOP_LOSS and
           FL_STOP_LOSS are NOT in the urgent list: they would go through _simulate_maker_exit_paper if maker_exit_enabled were on. It is
           false today, so they also fill at the price. There is NO slippage model (exit_slippage_pct stays NULL in paper). closed_at =
           datetime.utcnow() is taken AFTER the lock + DB work.
  ⇒ In paper, the exit price IS the print that triggered the stop, and closed_at is the wall clock a little later.
    (a) paper fill vs the stop line = how far that print gapped through the line.
    (b) "live proxy" = the last public print at closed_at vs the stop line ≈ what a live market order sent then would have got.
    (c) paper fill vs that last print: NEGATIVE means paper charged MORE than a live order would have paid (a wick that bounced).
    A late trigger (WS backlog / reconnect gap / cache refresh) moves the paper fill itself → flagged 'late'.

STOP REASONS — parse_reason(): strip the trailing " Ln" / "_Ln" level, then the RH_ prefix of a released hold (rh_strip semantics: the
hold's own RH_HARD_STOP / RH_PREMISE_EXIT / RH_TIME_EXIT keep their name), then every FLIP_ / BR_ / FL_ funnel prefix in any order
(FL_FLIP_…, FLIP_FL_…, FL_RH_…). A FL_ prefix = a flagged trade.
  counted as a STOP (a fixed loss line computable from the entry): STOP_LOSS · STOP_LOSS_WIDE · FL_STOP_LOSS / FL_STOP_LOSS_WIDE (a
    flagged trade's ordinary stop: the close funnel adds FL_ to any reason of a flagged trade, so the string exists) · FL_EMERGENCY_SL ·
    FL_DEEP_STOP · PATTERN_FIXED_SL · SPIKE_SL · RH_HARD_STOP
  NOT a stop here (a moving / profit-side line, a signal exit or discretionary): TRAILING_STOP · RUNNER_TRAIL · BREAKEVEN_EXIT ·
    LADDER_FLOOR · SPIKE_LOCK · HARD_TP* · FRENZY_TP / FRENZY_TP_LATE · ATR_FIXED_TP · PATTERN_FIXED_TP · FAST_EXIT · REGIME_CHANGE · SIGNAL_LOST ·
    EMA13_CROSS_EXIT · EMA_STACK_CROSS_EXIT · RSI_* · MOMENTUM_EXIT · SLOPE_EXIT · NO_EXPANSION · RECOVERED · RH_PREMISE_EXIT ·
    RH_TIME_EXIT · SPIKE_RSI_COOL · MAX_HOLD_TIME · MANUAL* · OVERRIDE_* · anything else.
  entry_strategy MANUAL is excluded (as in every scout tracker). FL_* stops come from both the polling and realtime paths
  (path = 'fl'): they are kept out of the clean delay stats and shown on their own line.

PER STOP. No per-order stop column exists except pattern_fixed_sl_pct / rh_hard_stop_pct, so the line is DERIVED from today's
trading_config.json with the engine's formulas above; line_src says which formula was used.
  stop_px    the price at which net P&L = the line (entry fee rate from the row, taker exit fee). This is the backtests' "theoretical stop".
  trig_px    the engine's trigger price (line + its epsilon).
  line_flag  'ok' when trig − LINE_TOL_LOW ≤ pnl ≤ trig + LINE_TOL_HIGH ·
             'high' = the fill sits above the trigger: today's config does not explain this stop ·
             'low' = the fill sits more than LINE_TOL_LOW below the trigger: a real gap-through OR an older, wider line.
  cross_at   the print that TRIGGERED the stop = the start of the last continuous through-the-line run of prints that starts at or
             before closed_at (else the first one within closed_at + CROSS_GRACE_MS, clock skew). delay_s = closed_at − cross_at.
  late       fill_vs_cross < −LATE_FILL or delay_s > LATE_DELAY_S (the paper fill was not the trigger print / the close came late).
  slip_stop  paper fill vs stop_px · slip_live  last print at closed_at vs stop_px · slip_last  paper fill vs that last print ·
  move_delay crossing print → last print · fill_vs_cross  paper fill vs the crossing print.
  All are % of price, sign-normalised: NEGATIVE = worse for us (LONG (x/ref − 1)·100, SHORT (1 − x/ref)·100).
Ticks: 1m klines find the last MAX_CANDIDATE_MIN crossing minutes before closed_at. Only those minutes and the LAST_WIN_MS before
closed_at are read: by REST aggTrades (paced ≥ 0.5 s) while the window is < 46 h old; otherwise from the tick cache, otherwise from the
daily archive of just those days (≤ DL_MAX downloads per run, temp files always deleted). The network budget is RUN_BUDGET_S per run
(every urlopen timeout = min(20 s, time left)). Any of the following stops every network call for the rest of the run and costs the row
NO try: HTTP 418 / 429, the used weight (X-MBX-USED-WEIGHT-1m) reaching WEIGHT_STOP, a network error (URLError, timeout, connection
error, HTTP 5xx).
STORE reports/SCOUT_STOP_SLIP.csv, keyed (opened_at to the second, pair, direction) — never id.
  · A computed row is FROZEN: write-once, never recomputed.
  · A definite data failure costs one try: an HTTP 4xx, a bad / missing archive, no prints, or a REST read cut by the page cap (partial
    data is never frozen). After MAX_TRIES failures the row is frozen as 'failed'.
  · An archive that is not published yet costs nothing. A budget-caused wait counts one try every BUDGET_WAITS_PER_TRY waits.
  · An unreadable store is moved to a timestamped .bad and the tracker starts empty.
CLI: python scripts/scout_stop_slip.py [--store PATH] [--dry-run]. It takes reports/.scout.lock unless --dry-run, and refuses to run
while the scout holds the lock.
"""
import glob
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
CSV = os.path.join(REPORTS, "SCOUT_STOP_SLIP.csv")
LOCK = os.path.join(REPORTS, ".scout.lock")
DL = os.path.expanduser("~/Downloads")
TICK_CACHE = os.path.join(REPORTS, "backtest_cache")
VER = 2
ASSUMED_SLIP = 0.10                # the backtests' exit-slip assumption (FRENZY studies: 0.10; some replays 0.02)
MAX_TRIES = 3
BUDGET_WAITS_PER_TRY = 3           # K budget-caused waits of the same row = one try
RUN_BUDGET_S = 120
URL_TIMEOUT_S = 20
DL_MAX = 4                         # archive downloads per run
WEIGHT_STOP = 1200                 # X-MBX-USED-WEIGHT-1m at which the run stops calling fapi
AGG_PACE_S = 0.5                   # ≥ 0.5 s between aggTrades calls (weight 20 each)
KL_PACE_S = 0.1
AGG_PAGES = 20                     # REST page cap per window — a read cut by it is a data failure (never frozen partial)
REST_MAX_AGE_MS = 46 * 3600_000    # fapi aggTrades with a time window serves only ~2 days back
ARCHIVE_FIRST_TRY_H = 6            # the daily archive is tried from day end + 6 h
ARCHIVE_GIVEUP_D = 5               # a 404 this long after the day = missing (a data failure)
LAST_WIN_MS = 120_000              # last-print window before closed_at
CROSS_GRACE_MS = 2_000             # closed_at is the bot's wall clock; Binance T the exchange's
LINE_TOL_HIGH = 0.03               # pnl above trigger + this → 'high' (today's config does not explain the stop)
LINE_TOL_LOW = 0.05                # pnl below trigger − this → 'low' (gap-through or an older, wider line)
LATE_FILL = 0.02                   # paper fill worse than the crossing print by more than this → late
LATE_DELAY_S = 5.0                 # closed_at − crossing print above this → late
MAX_CANDIDATE_MIN = 3              # crossing minutes read (the last ones before closed_at)
FINAL = ("ok", "no_cross", "no_line", "failed")
COLS = ["k", "pair", "direction", "strategy", "sleeve", "close_reason", "kind", "path", "opened_at", "closed_at", "entry", "exit",
        "pnl_pct", "line_pct", "line_src", "eps", "stop_px", "trig_px", "line_flag", "late", "slip_stop", "slip_live", "cross_at",
        "cross_px", "delay_s", "last_px", "last_at", "slip_last", "move_delay", "fill_vs_cross", "src", "status", "tries", "waits", "err",
        "ver", "computed_at"]

STOP_KINDS = ("STOP_LOSS", "STOP_LOSS_WIDE", "FL_STOP_LOSS", "FL_STOP_LOSS_WIDE", "FL_EMERGENCY_SL", "FL_DEEP_STOP", "PATTERN_FIXED_SL",
              "SPIKE_SL", "RH_HARD_STOP")
RH_OWN = ("RH_HARD_STOP", "RH_PREMISE_EXIT", "RH_TIME_EXIT")
FRENZY_SET = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
BR_SET = ("BULLRUN_LONG", "SURGE_LONG")


class RateLimited(Exception):
    """HTTP 418 / 429 or the used weight at WEIGHT_STOP: no more network this run, no try counted."""


class NetworkStop(Exception):
    """a transport failure (URLError, timeout, connection error, HTTP 5xx): no more network this run, no try counted."""


class BudgetExceeded(Exception):
    """the run's time budget is spent."""


class DataFail(Exception):
    """a definite data failure for this row: counts one try."""


class Wait(Exception):
    """the data is not available yet. kind 'unpublished' costs nothing; kind 'budget' counts toward a try."""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


def log(msg):
    print(f"[stop_slip] {msg}", file=sys.stderr)


# ─────────────────────────── pure helpers (tested) ───────────────────────────
def parse_reason(reason):
    """→ (base, flagged). Strips the level, a released hold's RH_, and every FLIP_ / BR_ / FL_ prefix in any order."""
    r = "" if reason is None or (isinstance(reason, float) and np.isnan(reason)) else str(reason).strip()
    r = re.sub(r"[\s_]L\d+$", "", r)
    flagged = False
    while True:
        if r.startswith("RH_") and not r.startswith(RH_OWN):
            r = r[3:]
        elif r.startswith("FL_"):
            r, flagged = r[3:], True
        elif r.startswith("FLIP_"):
            r = r[5:]
        elif r.startswith("BR_"):
            r = r[3:]
        else:
            return r, flagged


def classify(reason):
    """→ the stop kind (one of STOP_KINDS) or None when the exit is not a fixed-line stop."""
    b, fl = parse_reason(reason)
    if b in ("EMERGENCY_SL", "DEEP_STOP"):
        return "FL_" + b
    if b in ("STOP_LOSS", "STOP_LOSS_WIDE"):
        return ("FL_" + b) if fl else b
    return b if b in ("PATTERN_FIXED_SL", "SPIKE_SL", "RH_HARD_STOP") else None


def sleeve_of(strategy):
    s = str(strategy or "MOMENTUM")
    if s in ("", "nan", "None"):
        return "MOMENTUM"
    return "FLIP" if s.startswith("FLIP:") else s


def signed_pct(px, ref, direction):
    """px vs ref in % of ref, NEGATIVE = worse for the position being closed (LONG sells: lower is worse; SHORT buys: higher is worse)."""
    try:
        px, ref = float(px), float(ref)
    except (TypeError, ValueError):
        return np.nan
    if not (np.isfinite(px) and np.isfinite(ref)) or ref <= 0:
        return np.nan
    return (px / ref - 1.0) * 100.0 if str(direction).upper() == "LONG" else (1.0 - px / ref) * 100.0


def line_to_price(entry, line_pct, direction, fe_rate, fx_rate):
    """the price at which the engine's net P&L % (entry fee + a taker exit fee on the exit notional, ÷ entry notional) equals line_pct.
    LONG: P(1 − fx) = E(1 + fe + L/100) · SHORT: P(1 + fx) = E(1 − fe − L/100)."""
    E, L = float(entry), float(line_pct) / 100.0
    if str(direction).upper() == "LONG":
        return E * (1.0 + fe_rate + L) / (1.0 - fx_rate)
    return E * (1.0 - fe_rate - L) / (1.0 + fx_rate)


def _f(d, k, default):
    try:
        v = d.get(k, default)
        v = float(default if v is None else v)
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _momentum_sl(base, atr, th, direction, es):
    """check_realtime_stop_loss order: widen to −mult×ATR when wider → cap the WHOLE line at sl_atr_widen_floor_pct → quiet-pair SL."""
    sl = base
    mult = _f(th, "sl_atr_multiplier", 0.0)
    if mult > 0 and atr is not None and np.isfinite(atr) and atr > 0 and -(atr * mult) < sl:
        sl = -(atr * mult)
    cap = _f(th, "sl_atr_widen_floor_pct", 0.0)
    if cap < 0 and sl < cap:
        sl = cap
    thr, q = _f(th, "momentum_long_sl_atr_threshold", 0.0), _f(th, "momentum_long_sl_quiet_pct", 0.0)
    if (thr > 0 and q < 0 and str(direction).upper() == "LONG" and es in ("MOMENTUM", "")
            and atr is not None and np.isfinite(atr) and 0 < atr < thr and q < sl):
        sl = q
    return sl


def _br_sl(base, atr, th):
    """_bullrun_exit_for / surge_short_exit_for: min(base, max(−mult×ATR, cap))."""
    mult, cap = _f(th, "sl_atr_multiplier", 0.0), _f(th, "sl_atr_widen_floor_pct", 0.0)
    if atr is not None and np.isfinite(atr) and atr > 0 and mult > 0:
        w = -(atr * mult)
        if cap < 0:
            w = max(w, cap)
        return min(base, w)
    return base


def stop_line(row, cfg):
    """(line_pct, eps, src) — the engine's stop line for this stop with TODAY's config (see the module doc), or (None, 0, why)."""
    th = {**cfg, **(cfg.get("thresholds") or {})}
    conf = cfg.get("confidence_levels") or {}
    es = str(row.get("strategy") or "MOMENTUM")
    kind = row.get("kind")
    d = str(row.get("direction") or "").upper()
    try:
        atr = float(row.get("entry_atr_pct"))
    except (TypeError, ValueError):
        atr = np.nan
    if kind == "PATTERN_FIXED_SL":
        try:
            v = float(row.get("pattern_fixed_sl_pct"))
            return (v, 0.0, "pattern_fixed_sl_pct") if np.isfinite(v) and v < 0 else (None, 0.0, "pattern_fixed_sl_pct missing")
        except (TypeError, ValueError):
            return None, 0.0, "pattern_fixed_sl_pct missing"
    if kind == "RH_HARD_STOP":
        try:
            v = float(row.get("rh_hard_stop_pct"))
            return (v, 0.01, "rh_hard_stop_pct") if np.isfinite(v) else (None, 0.0, "rh_hard_stop_pct missing")
        except (TypeError, ValueError):
            return None, 0.0, "rh_hard_stop_pct missing"
    if kind == "FL_EMERGENCY_SL":
        return _f(th, "fl1_wide_sl_backstop", -1.2), 0.01, "fl1_wide_sl_backstop"   # realtime + 0.01 · polling 0
    if kind == "FL_DEEP_STOP":
        return _f(th, "fl2_deep_stop", -1.0), 0.01, "fl2_deep_stop"                 # realtime + 0.01 · polling 0
    if es == "FRENZY_WILLY":   # 🎲 Oct-8 (251): NO stop by default (frenzy_willy_stop_pct 0 = none; > 0 re-arms it — services.frenzy.frenzy_willy_levels)
        _ws = abs(_f(th, "frenzy_willy_stop_pct", 0.0))
        return ((-_ws, 0.0, "frenzy_willy_stop_pct") if _ws > 0 else (None, 0.0, "FRENZY_WILLY: no stop (a STOP_LOSS = the live backstop line)"))
    if es in FRENZY_SET:
        return -abs(_f(th, "frenzy_stop_pct", 3.0)) or -3.0, 0.0, "frenzy_stop_pct"
    if es in BR_SET:
        return _br_sl(_f(th, "bullrun_base_sl_pct", -0.7), atr, th), 0.0, "bullrun_base_sl_pct+atr"
    if es == "SURGE_SHORT" and kind == "STOP_LOSS":
        a = atr if np.isfinite(atr) and atr > 0 else _f(th, "surge_atr_min_pct", 1.5)
        return _br_sl(_f(conf.get("STRONG_BUY") or {}, "stop_loss", -0.70), a, th), 0.0, "STRONG_BUY.stop_loss+atr"
    if es == "SPIKE_FADE":
        return _f(th, "spike_fade_sl_pct", -1.5), 0.01, "spike_fade_sl_pct"
    if es == "SPIKE_BOUNCE":
        return _f(th, "spike_bounce_sl_pct", -0.7), 0.01, "spike_bounce_sl_pct"
    if es == "SPIKE_CHASE":
        return _f(th, "spike_sl_pct", -1.2), 0.01, "spike_sl_pct"
    if es == "MANUAL":
        return None, 0.0, "MANUAL"
    c = conf.get(str(row.get("confidence") or "")) or {}
    if kind in ("STOP_LOSS_WIDE", "FL_STOP_LOSS_WIDE"):
        if "signal_active_sl" not in c:
            return None, 0.0, "signal_active_sl unknown"
        return _momentum_sl(_f(c, "signal_active_sl", -1.0), atr, th, d, es), 0.01, "signal_active_sl+atr"
    if "stop_loss" not in c:
        return None, 0.0, "confidence unknown"
    return _momentum_sl(_f(c, "stop_loss", -0.70), atr, th, d, es), 0.01, "stop_loss+atr"


def line_flag(pnl, trig_line):
    """'ok' / 'high' (fill above the trigger + LINE_TOL_HIGH) / 'low' (more than LINE_TOL_LOW below it) / 'na'."""
    try:
        pnl, trig_line = float(pnl), float(trig_line)
    except (TypeError, ValueError):
        return "na"
    if not (np.isfinite(pnl) and np.isfinite(trig_line)):
        return "na"
    if pnl > trig_line + LINE_TOL_HIGH:
        return "high"
    if pnl < trig_line - LINE_TOL_LOW:
        return "low"
    return "ok"


def trigger_cross(t, p, trig_px, direction, t_from, t_close, grace=CROSS_GRACE_MS):
    """the print that TRIGGERED the stop = the start of the last continuous run of through-the-line prints that STARTS at or before
    t_close (prints before t_from ignored; an earlier wick that recovered is skipped). If no run starts by t_close (clock skew), the first
    run starting within t_close + grace. → (t, p) or (None, None)."""
    t = np.asarray(t, dtype=np.int64); p = np.asarray(p, dtype=float)
    m = (t >= int(t_from)) & (t <= int(t_close) + int(grace))
    t, p = t[m], p[m]
    thr = (p <= trig_px) if str(direction).upper() == "LONG" else (p >= trig_px)
    starts = np.flatnonzero(thr & ~np.r_[False, thr[:-1]])
    if not len(starts):
        return None, None
    by_close = starts[t[starts] <= int(t_close)]
    s = by_close[-1] if len(by_close) else starts[0]
    return int(t[s]), float(p[s])


def last_print(t, p, t_at):
    """the last print at or before t_at → (t, p) or (None, None)."""
    t = np.asarray(t, dtype=np.int64)
    i = int(np.searchsorted(t, int(t_at), side="right")) - 1
    return (int(t[i]), float(np.asarray(p)[i])) if i >= 0 and len(t) else (None, None)


def measure(rec, t, p):
    """fill the tick-derived columns of rec (direction, exit, opened_ms, closed_ms, trig_px, stop_px — NaN when no line) from prints
    (t, p sorted by time). → status 'ok' / 'no_cross' / 'no_line' (a line-less stop: last-print columns only)."""
    d = rec["direction"]
    lt, lp = last_print(t, p, rec["closed_ms"])
    rec["last_px"] = lp if lp is not None else np.nan
    rec["last_at"] = _iso(lt) if lt is not None else ""
    rec["slip_last"] = signed_pct(rec["exit"], lp, d) if lp is not None else np.nan
    rec["slip_live"] = signed_pct(lp, rec.get("stop_px"), d) if lp is not None else np.nan
    trig = rec.get("trig_px")
    if trig is None or not np.isfinite(trig):
        return "no_line"
    ct, cp = trigger_cross(t, p, trig, d, rec["opened_ms"], rec["closed_ms"])
    if ct is None:
        return "no_cross"
    rec["cross_at"] = _iso(ct); rec["cross_px"] = cp
    rec["delay_s"] = (rec["closed_ms"] - ct) / 1000.0
    rec["move_delay"] = signed_pct(lp, cp, d) if lp is not None else np.nan
    rec["fill_vs_cross"] = signed_pct(rec["exit"], cp, d)
    rec["late"] = bool(rec["fill_vs_cross"] < -LATE_FILL or rec["delay_s"] > LATE_DELAY_S)
    return "ok"


def _int0(v):
    try:
        v = float(v)
        return int(v) if np.isfinite(v) else 0
    except (TypeError, ValueError):
        return 0


def after_failure(prev_tries):
    """→ (status, tries) after a definite data failure: retried until MAX_TRIES failures, then frozen as 'failed'."""
    n = _int0(prev_tries) + 1
    return ("failed" if n >= MAX_TRIES else "pending"), n


def after_budget_wait(prev_tries, prev_waits):
    """→ (status, tries, waits): every BUDGET_WAITS_PER_TRY budget-caused waits count one try."""
    w = _int0(prev_waits) + 1
    if w >= BUDGET_WAITS_PER_TRY:
        st, n = after_failure(prev_tries)
        return st, n, 0
    return "pending", _int0(prev_tries), w


def merge_store(old, new_rows):
    """write-once merge: a FINAL stored row is never replaced; a pending stored row is replaced by its newer attempt."""
    old = old.reindex(columns=COLS) if len(old) else pd.DataFrame(columns=COLS)
    if not new_rows:
        return old
    new = pd.DataFrame(new_rows).reindex(columns=COLS)
    frozen = {(a, b, c) for a, b, c, s in zip(old.k, old.pair, old.direction, old.status) if str(s) in FINAL}
    new = new[[(a, b, c) not in frozen for a, b, c in zip(new.k, new.pair, new.direction)]]
    newkeys = set(zip(new.k, new.pair, new.direction))
    old = old[[(a, b, c) not in newkeys for a, b, c in zip(old.k, old.pair, old.direction)]]
    frames = [x for x in (old, new) if len(x)]
    return (pd.concat(frames, ignore_index=True) if frames else old).sort_values("k").reset_index(drop=True)


# ─────────────────────────── data: orders ───────────────────────────
ORDER_COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "exit_price", "pnl_percentage",
              "close_reason", "confidence", "entry_atr_pct", "pattern_fixed_sl_pct", "rh_hard_stop_pct", "entry_fee", "quantity")


def _export_ms(path):
    try:
        b = os.path.basename(path).rsplit("_paper_", 1)[1][:19]
        return int(datetime.strptime(b, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    except Exception:
        return int(os.path.getmtime(path) * 1000)


def load_stops(dl=None, reports=None):
    """closed bot stops: every orders export (newest export wins) + reports/MASTER_POOL_stacked.csv (lowest priority), deduped on
    (opened_at to the second, pair, direction) — never id; CLOSED only; MANUAL excluded; only the STOP_KINDS reasons."""
    dl = DL if dl is None else dl
    reports = REPORTS if reports is None else reports
    fr = []
    pool = os.path.join(reports, "MASTER_POOL_stacked.csv")
    if os.path.exists(pool):
        try:
            fr.append(pd.read_csv(pool, low_memory=False, usecols=lambda c: c in ORDER_COLS).assign(_rank=-1))
        except Exception as e:
            log(f"master pool unreadable: {e}")
    for i, f in enumerate(sorted(glob.glob(os.path.join(dl, "scalpars_orders_paper_*.csv")), key=_export_ms)):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ORDER_COLS).assign(_rank=i))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(ORDER_COLS) + ["k", "kind"])
    A = pd.concat(fr, ignore_index=True)
    for c in ORDER_COLS:
        if c not in A:
            A[c] = np.nan
    A = A.dropna(subset=["opened_at", "pair", "direction"])
    A["k"] = A.opened_at.astype(str).str.replace(" ", "T", regex=False).str[:19]
    A = A.sort_values("_rank", kind="stable").drop_duplicates(["k", "pair", "direction"], keep="last")
    A = A[(A.status.astype(str) == "CLOSED") & (A.entry_strategy.astype(str) != "MANUAL") & A.closed_at.notna()].copy()
    A["kind"] = A.close_reason.map(classify)
    return A[A.kind.notna()].reset_index(drop=True)


# ─────────────────────────── data: prints ───────────────────────────
def _iso(ms):
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%dT%H:%M:%S.%f")[:23]


def _ms(s):
    return int(pd.Timestamp(str(s)[:26].replace(" ", "T")).tz_localize(None).value // 1_000_000)


def _remaining(budget):
    rem = budget["deadline"] - time.monotonic()
    if rem <= 0:
        raise BudgetExceeded("run budget")
    return rem


def _open(url, budget, cap=URL_TIMEOUT_S):
    """urlopen with timeout = min(cap, time left). 418/429 → RateLimited · 5xx / transport → NetworkStop (both stop the run's network)
    · other HTTP errors are re-raised for the caller (a 404 on an archive means something there). Tracks X-MBX-USED-WEIGHT-1m."""
    if budget.get("stop"):
        raise (RateLimited if budget.get("stop_kind") == "rate" else NetworkStop)(budget["stop"])
    rem = _remaining(budget)
    try:
        resp = urllib.request.urlopen(url, timeout=min(cap, rem))
    except urllib.error.HTTPError as ex:
        if ex.code in (418, 429):
            budget.update(stop=f"HTTP {ex.code}", stop_kind="rate")
            raise RateLimited(f"HTTP {ex.code}")
        if ex.code >= 500:
            budget.update(stop=f"HTTP {ex.code}", stop_kind="net")
            raise NetworkStop(f"HTTP {ex.code}")
        raise
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as ex:
        budget.update(stop=f"{type(ex).__name__}", stop_kind="net")
        raise NetworkStop(str(ex)[:80])
    try:
        w = int((resp.headers or {}).get("X-MBX-USED-WEIGHT-1m") or 0)
    except (TypeError, ValueError, AttributeError):
        w = 0
    budget["weight"] = max(budget.get("weight", 0), w)
    if w >= WEIGHT_STOP:
        budget.update(stop=f"used weight {w}", stop_kind="rate")   # this response is used; the next call refuses
    return resp


def _get_json(url, budget, pace=KL_PACE_S):
    resp = _open(url, budget)
    try:
        with resp:
            out = json.loads(resp.read())
    except (socket.timeout, TimeoutError, ConnectionError, urllib.error.URLError) as ex:
        budget.update(stop=type(ex).__name__, stop_kind="net")
        raise NetworkStop(str(ex)[:80])
    budget["calls"] = budget.get("calls", 0) + 1
    time.sleep(pace)
    return out


def _http(fn, *a, **kw):
    """a 4xx that is not a rate limit = a definite data failure for this row."""
    try:
        return fn(*a, **kw)
    except urllib.error.HTTPError as ex:
        raise DataFail(f"HTTP {ex.code}")


def klines_1m(sym, t0, t1, budget):
    out, s = [], int(t0) // 60_000 * 60_000
    while s <= t1:
        q = urllib.parse.urlencode(dict(symbol=sym, interval="1m", startTime=s, endTime=int(t1), limit=1500))
        r = _http(_get_json, f"https://fapi.binance.com/fapi/v1/klines?{q}", budget)
        if not r:
            break
        out += [(int(x[0]), float(x[2]), float(x[3])) for x in r]
        nxt = int(r[-1][0]) + 60_000
        if nxt <= s:
            break
        s = nxt
    return out


def aggtrades_rest(sym, t0, t1, budget, pages=AGG_PAGES):
    """public aggTrades prints in [t0, t1] (window ≤ 1 h) → (t, p, truncated). truncated = the page cap ended the read."""
    rows, q = [], dict(symbol=sym, startTime=int(t0), endTime=int(t1), limit=1000)
    trunc = False
    for pg in range(pages):
        r = _http(_get_json, "https://fapi.binance.com/fapi/v1/aggTrades?" + urllib.parse.urlencode(q), budget, AGG_PACE_S)
        rows += [(int(x["T"]), float(x["p"])) for x in r if int(x["T"]) <= int(t1)]
        if len(r) < 1000 or int(r[-1]["T"]) > int(t1):
            break
        q = dict(symbol=sym, fromId=int(r[-1]["a"]) + 1, limit=1000)
        trunc = pg == pages - 1
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=float), trunc
    a = np.array(rows)
    o = np.argsort(a[:, 0], kind="stable")
    return a[o, 0].astype(np.int64), a[o, 1].astype(float), trunc


_ARCH = {}


def _arch_clear():
    for v in list(_ARCH.values()):
        if isinstance(v, str) and v.endswith(".zip"):
            try:
                os.remove(v)
            except OSError:
                pass
    _ARCH.clear()


def _cache_day(pair, date):
    for base in ("ticks_q", "ticks"):
        f = os.path.join(TICK_CACHE, base, pair, f"{date}.npz")
        if os.path.exists(f):
            try:
                with np.load(f) as z:
                    return z["t"].astype(np.int64), z["p"].astype(np.float64)
            except Exception:
                continue
    return None


def _download_archive(pair, date, now_ms, budget):
    """the pair-day archive into a temp file (deleted on any failure). Raises Wait / DataFail / RateLimited / NetworkStop /
    BudgetExceeded."""
    day_end = int(pd.Timestamp(date).value // 1_000_000) + 86_400_000
    if now_ms < day_end + ARCHIVE_FIRST_TRY_H * 3600_000:
        raise Wait("unpublished")
    if budget.get("dl", 0) <= 0:
        raise Wait("budget")
    q = urllib.parse.quote(pair)
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{q}/{q}-aggTrades-{date}.zip"
    try:
        resp = _open(url, budget, cap=30)
    except urllib.error.HTTPError as ex:
        if ex.code == 404:
            if now_ms > day_end + ARCHIVE_GIVEUP_D * 86_400_000:
                raise DataFail("archive missing")
            raise Wait("unpublished")
        raise DataFail(f"archive HTTP {ex.code}")
    budget["dl"] -= 1
    fd, tmp = tempfile.mkstemp(suffix=".zip", prefix="stopslip_")
    try:
        with resp, os.fdopen(fd, "wb") as out:
            while True:
                _remaining(budget)
                try:
                    ch = resp.read(1 << 20)
                except (socket.timeout, TimeoutError, ConnectionError, urllib.error.URLError) as ex:
                    budget.update(stop=type(ex).__name__, stop_kind="net")
                    raise NetworkStop(str(ex)[:80])
                if not ch:
                    break
                out.write(ch)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    _ARCH[(pair, date)] = tmp


def _arch_slice(src, t0, t1, budget=None):
    """stream the [t0, t1] prints out of an aggTrades archive zip (time-ordered; stops once past t1; a header line is sniffed). The
    deadline is checked between chunks. Raises on a bad archive."""
    out = []
    with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as fh:
        first = fh.peek(200)[:200].split(b"\n", 1)[0]
        hdr = 0 if first[:1] and not first[:1].isdigit() else None
        rd = pd.read_csv(fh, header=hdr, names=["a", "p", "q", "f", "l", "t", "m"], usecols=["p", "t"],
                         dtype={"p": np.float64, "t": np.int64}, chunksize=500_000)
        for ch in rd:
            if budget is not None:
                _remaining(budget)
            w = ch[(ch.t >= int(t0)) & (ch.t <= int(t1))]
            if len(w):
                out.append(w)
            if len(ch) and int(ch.t.iloc[-1]) > int(t1):
                break
    if not out:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    d = pd.concat(out).sort_values("t", kind="stable")
    return d.t.values.astype(np.int64), d.p.values.astype(float)


def _windows(rec, budget):
    """the tick windows to read: the last MAX_CANDIDATE_MIN crossing minutes before closed_at (1m klines, searched backwards) + the
    LAST_WIN_MS before closed_at; merged."""
    sym, d, o, c = rec["pair"], rec["direction"], rec["opened_ms"], rec["closed_ms"]
    wins = [(max(o, c - LAST_WIN_MS), c + CROSS_GRACE_MS)]
    trig = rec.get("trig_px")
    if trig is not None and np.isfinite(trig):
        kl = klines_1m(sym, o, c + CROSS_GRACE_MS, budget)
        cand = [m for m, h, l in kl if m <= c + CROSS_GRACE_MS and (l <= trig if d == "LONG" else h >= trig)][-MAX_CANDIDATE_MIN:]
        wins += [(max(o, m), min(m + 60_000, c + CROSS_GRACE_MS)) for m in cand]
    wins.sort()
    merged = []
    for a, b in wins:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def fetch_prints(rec, now_ms, budget):
    """→ (t, p, src) over the windows. Raises Wait / DataFail / RateLimited / NetworkStop / BudgetExceeded."""
    wins = _windows(rec, budget)
    ts, ps = [], []
    if now_ms - wins[0][0] < REST_MAX_AGE_MS:
        for a, b in wins:
            for s in range(int(a), int(b), 3_600_000):                 # REST windows ≤ 1 h
                t, p, trunc = aggtrades_rest(rec["pair"], s, min(int(b), s + 3_600_000 - 1), budget)
                if trunc:
                    raise DataFail("aggTrades page cap (partial window)")
                ts.append(t); ps.append(p)
        src = "rest"
    else:
        src = "cache"
        days = {}
        for a, b in wins:                                              # only the days the windows touch
            for dd in pd.date_range(pd.Timestamp(int(a), unit="ms").normalize(), pd.Timestamp(int(b), unit="ms").normalize()):
                days.setdefault(f"{dd:%Y-%m-%d}", []).append((a, b))
        for date, ws in sorted(days.items()):
            got = _cache_day(rec["pair"], date)
            if got is None:
                if (rec["pair"], date) not in _ARCH:
                    _download_archive(rec["pair"], date, now_ms, budget)
                try:
                    got = _arch_slice(_ARCH[(rec["pair"], date)], min(a for a, _ in ws), max(b for _, b in ws), budget)
                except BudgetExceeded:
                    raise
                except Exception as ex:
                    v = _ARCH.pop((rec["pair"], date), None)
                    if v:
                        try:
                            os.remove(v)
                        except OSError:
                            pass
                    raise DataFail(f"bad archive ({type(ex).__name__})")
                src = "archive"
            t, p = got
            m = np.zeros(len(t), dtype=bool)
            for a, b in ws:
                m |= (t >= int(a)) & (t <= int(b))
            ts.append(t[m]); ps.append(p[m])
    t = np.concatenate(ts) if ts else np.array([], dtype=np.int64)
    p = np.concatenate(ps) if ps else np.array([], dtype=float)
    o = np.argsort(t, kind="stable")
    return t[o], p[o], src


# ─────────────────────────── per stop ───────────────────────────
def base_record(r, cfg, taker):
    d = str(r["direction"]).upper()
    rec = dict(k=r["k"], pair=str(r["pair"]), direction=d, strategy=str(r.get("entry_strategy") or "MOMENTUM"),
               sleeve=sleeve_of(r.get("entry_strategy")), close_reason=str(r.get("close_reason")), kind=r["kind"],
               opened_at=str(r["opened_at"]), closed_at=str(r["closed_at"]), entry=float(r["entry_price"]), exit=float(r["exit_price"]),
               pnl_pct=float(r["pnl_percentage"]) if pd.notna(r.get("pnl_percentage")) else np.nan, ver=VER, late=False)
    rec["path"] = "fl" if str(r["kind"]).startswith("FL_") else "realtime"
    rec["opened_ms"], rec["closed_ms"] = _ms(r["opened_at"]), _ms(r["closed_at"])
    line, eps, src = stop_line({**r, "strategy": rec["strategy"], "kind": rec["kind"], "direction": d}, cfg)
    rec["line_src"] = src
    if line is None:
        rec.update(line_pct=np.nan, eps=np.nan, stop_px=np.nan, trig_px=np.nan, line_flag="na", slip_stop=np.nan)
        return rec
    try:
        fe = float(r.get("entry_fee")) / (float(r["entry_price"]) * float(r.get("quantity")))
        fe = fe if np.isfinite(fe) and 0 <= fe < 0.01 else taker
    except (TypeError, ValueError, ZeroDivisionError):
        fe = taker
    rec.update(line_pct=round(line, 4), eps=eps, stop_px=line_to_price(rec["entry"], line, d, fe, taker),
               trig_px=line_to_price(rec["entry"], line + eps, d, fe, taker))
    rec["line_flag"] = line_flag(rec["pnl_pct"], line + eps)
    rec["slip_stop"] = signed_pct(rec["exit"], rec["stop_px"], d)
    return rec


def compute(rec, now_ms, budget):
    """→ the status after measuring ('ok' / 'no_cross' / 'no_line'). Raises Wait / DataFail / RateLimited / NetworkStop / BudgetExceeded."""
    t, p, src = fetch_prints(rec, now_ms, budget)
    if not len(t):
        raise DataFail("no prints")
    rec["src"] = src
    return measure(rec, t, p)


def load_store(path=None):
    path = CSV if path is None else path
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    try:
        d = pd.read_csv(path, dtype={"k": str, "pair": str, "direction": str})
        if len(d) and not {"k", "pair", "direction", "status"} <= set(d.columns):
            raise ValueError("columns missing")
        return d
    except Exception:
        bad = f"{path}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.{os.getpid()}.bad"
        try:
            os.replace(path, bad)
        except OSError:
            pass
        return pd.DataFrame(columns=COLS)


def save_store(df, path=None):
    path = CSV if path is None else path
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        df.reindex(columns=COLS).to_csv(tmp, index=False, float_format="%.10g")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_cfg():
    with open(os.path.join(ROOT, "trading_config.json")) as fh:
        return json.load(fh)


def update(now_ms, budget, cfg=None, S=None, old=None):
    """one pass: every stop not yet frozen is attempted (newest first) inside the budget → (merged store, notes, changed rows)."""
    cfg = cfg if cfg is not None else _load_cfg()
    taker = float(cfg.get("taker_fee", cfg.get("trading_fee", 0.00045)) or 0.00045)
    S = load_stops() if S is None else S
    old = load_store() if old is None else old
    prev_of = {(a, b, c): (s, t, w) for a, b, c, s, t, w in zip(old.get("k", []), old.get("pair", []), old.get("direction", []),
                                                               old.get("status", []), old.get("tries", []),
                                                               old.get("waits", [np.nan] * len(old)))}
    rows, notes = [], {"done": 0, "wait": 0, "fail": 0, "budget_wait": 0, "skipped_budget": 0}
    todo = [r for r in S.sort_values("closed_at", ascending=False).to_dict("records")
            if str(prev_of.get((r["k"], str(r["pair"]), str(r["direction"]).upper()), ("", 0, 0))[0]) not in FINAL]
    for r in todo:
        if budget.get("stop") or time.monotonic() > budget["deadline"]:
            notes["skipped_budget"] += 1
            continue
        try:
            rec = base_record(r, cfg, taker)
        except Exception as ex:
            log(f"bad row {r.get('pair')} {r.get('k')}: {ex}")
            continue
        _, ptries, pwaits = prev_of.get((rec["k"], rec["pair"], rec["direction"]), ("", 0, 0))
        try:
            status = compute(rec, now_ms, budget)
            rec.update(status=status, tries=_int0(ptries), waits=_int0(pwaits), err="")
            notes["done"] += 1
        except (RateLimited, NetworkStop) as ex:
            notes["stopped"] = f"{type(ex).__name__}: {ex}"
            notes["skipped_budget"] += 1
            continue
        except (BudgetExceeded, Wait) as ex:
            if isinstance(ex, Wait) and ex.kind == "unpublished":
                notes["wait"] += 1
                continue
            st, n, w = after_budget_wait(ptries, pwaits)
            rec.update(status=st, tries=n, waits=w, err="budget")
            notes["budget_wait"] += 1
        except DataFail as ex:
            st, n = after_failure(ptries)
            rec.update(status=st, tries=n, waits=_int0(pwaits), err=str(ex)[:120])
            notes["fail"] += 1
        rec["computed_at"] = _iso(now_ms)
        rows.append({c: rec.get(c, np.nan) for c in COLS})
    return merge_store(old, rows), notes, len(rows)


# ─────────────────────────── report ───────────────────────────
def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _true(s):
    return s.astype(str).isin(["True", "true", "1", "1.0"])


def _fmt(v, nd=2, plus=True):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(v):
        return "–"
    return f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"


def clean_mask(g):
    """rows the vs-stop / delay stats read: measured with a crossing, the line explained by today's config, not late, realtime path."""
    return ((g.status.astype(str) == "ok") & (g.line_flag.astype(str) == "ok") & ~_true(g.late) & (g.path.astype(str) != "fl"))


def group_stats(g):
    """N · paper fill vs stop (mean / median / p10 = the worst-decile edge) · live proxy vs stop (same) · paper fill vs last print (mean)
    · delay (mean / median s) · move during the delay — vs-stop / delay / move on the CLEAN rows only; the live proxy on every measured row
    with a line that is not 'high'. Extra cost = −mean − ASSUMED_SLIP (%/stopped trade; positive = the assumption is too kind)."""
    c = g[clean_mask(g)]
    lv = g[(g.status.astype(str) == "ok") & (g.line_flag.astype(str).isin(["ok", "low"]))]
    ss, live, sl = _num(c.slip_stop).dropna(), _num(lv.slip_live).dropna(), _num(g.slip_last).dropna()
    dl, mv = _num(c.delay_s).dropna(), _num(c.move_delay).dropna()
    q = lambda x, f: float(f(x)) if len(x) else np.nan
    p10 = lambda x: np.percentile(x, 10)
    return dict(n=len(g), n_clean=len(c), n_live=len(live), ss_mean=q(ss, np.mean), ss_med=q(ss, np.median), ss_p10=q(ss, p10),
                lv_mean=q(live, np.mean), lv_med=q(live, np.median), lv_p10=q(live, p10), sl_mean=q(sl, np.mean),
                d_mean=q(dl, np.mean), d_med=q(dl, np.median), mv_mean=q(mv, np.mean),
                extra_paper=(-q(ss, np.mean) - ASSUMED_SLIP) if len(ss) else np.nan,
                extra_live=(-q(live, np.mean) - ASSUMED_SLIP) if len(live) else np.nan)


def _row(name, st):
    return (f"| {name} | {st['n']} · {st['n_clean']} · {st['n_live']} | {_fmt(st['ss_mean'])} / {_fmt(st['ss_med'])} / {_fmt(st['ss_p10'])} | "
            f"{_fmt(st['lv_mean'])} / {_fmt(st['lv_med'])} / {_fmt(st['lv_p10'])} | {_fmt(st['sl_mean'])} | "
            f"{_fmt(st['d_mean'], 1, False)} / {_fmt(st['d_med'], 1, False)} | {_fmt(st['mv_mean'])} | {_fmt(st['extra_paper'])} | "
            f"{_fmt(st['extra_live'])} |")


def _side_line(name, g):
    if not len(g):
        return f"- {name}: 0"
    return (f"- {name}: {len(g)} · paper fill vs stop mean {_fmt(_num(g.slip_stop).mean())} · live proxy vs stop mean "
            f"{_fmt(_num(g.slip_live).mean())} · delay mean {_fmt(_num(g.delay_s).mean(), 1, False)} s")


def lines(store, now_ms, notes=None):
    L = ["## 🧯 Stop-exit slippage (STOP_SLIP, observe only — never changes trading)", "",
         "Every bot STOP exit (a fixed loss line: STOP_LOSS · STOP_LOSS_WIDE · FL_STOP_LOSS(_WIDE) · FL_EMERGENCY_SL · FL_DEEP_STOP · "
         "PATTERN_FIXED_SL · SPIKE_SL · RH_HARD_STOP; trailing / break-even / profit lines and MANUAL excluded) measured on public aggTrades. "
         "Paper fills a stop AT the print that triggered it (no slippage model). **Paper fill vs stop** = how far that print gapped "
         "through the line. **Live proxy vs stop** = the last print at closed_at vs the line ≈ what a live market order would get. "
         "**Paper fill vs last print** negative = paper charged MORE than the market at closed_at (a wick that bounced). "
         "All in % of price, NEGATIVE = worse for us; p10 = the worst-decile edge. Stop lines = today's config. "
         f"N = all · clean (crossing found, line explained, not late, realtime path) · live-proxy rows. "
         f"Extra = −mean − {ASSUMED_SLIP:.2f} (the backtests' exit-slip assumption), %/stopped trade; positive = the assumption is too kind.", ""]
    ok = store[store.status.astype(str).isin(["ok", "no_cross", "no_line"])].copy() if len(store) else store
    if not len(ok):
        L.append("No stop measured yet.")
    else:
        ok["cms"] = [_ms(x) for x in ok.closed_at]
        wk = now_ms - 7 * 86_400_000
        L += ["| Cohort | N | paper fill vs stop mean / med / p10 | live proxy vs stop mean / med / p10 | paper fill vs last print | "
              f"delay s mean / med | move in delay | extra vs {ASSUMED_SLIP:.2f} (paper) | extra vs {ASSUMED_SLIP:.2f} (live proxy) |",
              "|---|---|---|---|---|---|---|---|---|"]
        L.append(_row("**ALL**", group_stats(ok)))
        o7 = ok[ok.cms >= wk]
        L.append(_row("ALL · last 7 d", group_stats(o7)) if len(o7) else "| ALL · last 7 d | 0 | – | – | – | – | – | – | – |")
        for d in ("LONG", "SHORT"):
            g = ok[ok.direction.astype(str) == d]
            if len(g):
                L.append(_row(d, group_stats(g)))
        for sl_, g in sorted(ok.groupby(ok.sleeve.astype(str)), key=lambda x: -len(x[1])):
            L.append(_row(sl_, group_stats(g)))
            g7 = g[g.cms >= wk]
            if len(g7):
                L.append(_row(f"{sl_} · 7 d", group_stats(g7)))
        meas = ok[ok.status.astype(str) == "ok"]
        rt = meas[meas.path.astype(str) != "fl"]
        low = rt[rt.line_flag.astype(str) == "low"]
        L += ["", "Kept out of the paper-vs-stop / delay columns:",
              _side_line(f"late (fill worse than the crossing print by > {LATE_FILL} or delay > {LATE_DELAY_S:g} s)", rt[_true(rt.late)]),
              _side_line(f"line 'low' (fill > {LINE_TOL_LOW} below the trigger), not late = a real gap-through",
                         low[~_true(low.late)]),
              _side_line("line 'low' and late = probably an older, wider line", low[_true(low.late)]),
              _side_line(f"line 'high' (fill > {LINE_TOL_HIGH} above the trigger: today's config does not explain it)",
                         rt[rt.line_flag.astype(str) == "high"]),
              _side_line("FL_* stops (polling + realtime paths)", meas[meas.path.astype(str) == "fl"]),
              f"- no crossing found: {int((ok.status.astype(str) == 'no_cross').sum())} · no derivable line: "
              f"{int((ok.status.astype(str) == 'no_line').sum())}"]
        w = ok[clean_mask(ok) & _num(ok.slip_live).notna()].copy()
        if len(w):
            w["ws"] = _num(w.slip_live)
            L += ["", "Worst 5 by the live proxy (clean rows):", "",
                  "| Closed UTC | Pair | Sleeve | Dir | Reason | Line | paper fill vs stop | live proxy vs stop | delay s | move in delay |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in w.sort_values("ws").head(5).to_dict("records"):
                L.append(f"| {str(r['closed_at'])[5:16].replace('T', ' ')} | {str(r['pair']).replace('USDT', '')} | {r['sleeve']} | "
                         f"{r['direction']} | {r['close_reason']} | {_fmt(r.get('line_pct'))} | {_fmt(r.get('slip_stop'))} | {_fmt(r['ws'])} | "
                         f"{_fmt(r.get('delay_s'), 1, False)} | {_fmt(r.get('move_delay'))} |")
    if len(store):
        vc = store.status.astype(str).value_counts().to_dict()
        L += ["", "Store: " + " · ".join(f"{k} {v}" for k, v in sorted(vc.items())) + " (reports/SCOUT_STOP_SLIP.csv, write-once)."]
    if notes:
        L.append(f"This run: {notes.get('done', 0)} measured · {notes.get('fail', 0)} data failures · {notes.get('budget_wait', 0)} "
                 f"budget waits · {notes.get('wait', 0)} waiting for the archive · {notes.get('skipped_budget', 0)} left for the next run"
                 + (f" · ⛔ {notes['stopped']} — network stopped" if notes.get("stopped") else "") + ".")
    return L + [""]


def run(now_ms=None, store_path=None, dry_run=False):
    now_ms = int(now_ms or time.time() * 1000)
    path = CSV if store_path is None else store_path
    budget = {"deadline": time.monotonic() + RUN_BUDGET_S, "dl": DL_MAX, "calls": 0, "weight": 0}
    try:
        store, notes, changed = update(now_ms, budget, old=load_store(path))
        if changed and not dry_run:
            save_store(store, path)
    finally:
        _arch_clear()
    return lines(store, now_ms, notes)


def _take_lock():
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) >= 900:
        try:
            os.remove(LOCK)                                   # stale (a crashed run) — the scout's own rule
        except OSError:
            pass
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    return True


def _release_lock():
    try:
        with open(LOCK) as fh:
            mine = fh.read().strip() == str(os.getpid())
        if mine:
            os.remove(LOCK)
    except OSError:
        pass


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="STOP_SLIP tracker (observe only)")
    ap.add_argument("--store", default=None, help="store CSV (default reports/SCOUT_STOP_SLIP.csv)")
    ap.add_argument("--dry-run", action="store_true", help="measure and print, never write the store")
    a = ap.parse_args(argv)
    if not a.dry_run:
        if not _take_lock():
            log("the scout holds reports/.scout.lock — refusing"); return 1
    try:
        print("\n".join(run(store_path=a.store, dry_run=a.dry_run)))
    finally:
        if not a.dry_run:
            _release_lock()
    return 0


if __name__ == "__main__":
    sys.exit(main())
