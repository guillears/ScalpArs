#!/usr/bin/env python3
"""🎯 Scout — FRENZY / WIDE per-fill EXIT SHADOWS (pre-registered 2026-10-05, operator after RLC 10-05; DECISION_LOG 214). OBSERVE only —
never changes config, never a trade. Called by scripts/opportunity_scout.py every run (never breaks it). Public Binance 1m / 5m / 1h klines.

For every live FRENZY_LONG / FRENZY_WIDE fill in the orders exports (~/Downloads, MANUAL excluded), re-priced from the ACTUAL entry price on
1m bars (low before high inside a minute, a line set by a minute never closes in that minute, fill at the line or at the open when a minute
gaps through it), net of 0.09 % fees, 12 h cap — the same accounting as scripts/frenzy_fill_table (Oct-5 table):
  LOCK2   the live exit: −3 until +3, then max(+2, peak − 2)              LOCK3   the same with a 3-pt trail
  EMA20 / EMA50   −3 until +3, then a +2 floor and out at the first 5m close below the 5m EMA20 / EMA50
  FIX3    the old fixed +3 / −3                                             ACTUAL  the bot's own result (the exit live at the time)
⛔ SHADOWS 1 and 2 RETIRED 2026-10-06 (operator; DECISION_LOG 218): on the engine cohort (reports/FRENZY_REDO_EXITS_2026-10-05.md) ATRE is
  −0.333 vs the lock (CI −0.54…−0.13, both halves negative) and NOTSTRETCHED's first-entry half is negative (−0.21 / −0.14); re-entry of
  every kind is refuted (FRENZY_REDO_REENTRY_2026-10-05.md). No longer computed; their old columns stay in the CSV for the record.
SHADOW 1 (retired) — FRENZY_ATRE_SHADOW (reports/FRENZY_ATR_AND_PURE_EMA_EXIT_2026-10-05.md, frozen): FRENZY_LONG first entries only; −3 until +3, then a
  +2 floor plus an exit at the first 5m close whose ATR% (Wilder 14 on the last 300 closed bars ÷ close, the engine's own ruler) is below the
  signal bar's (the stamped entry_atr_pct); 12 h cap. BAR (read at N ≥ 60 entries on ≥ 30 days): Δ(shadow − LOCK2) mean > 0 ∧ day-block 95 %
  CI low > 0 ∧ mean > 0 without the top 5 and the top 10 ∧ no pair > 35 % of the gain ∧ both halves > 0. Otherwise retired; never re-fit.
SHADOW 2 (retired) — TYPE_III_NOTSTRETCHED (reports/FRENZY_EXIT_SELECTOR_2026-10-05.md, frozen): a FRENZY or WIDE fill whose stamped
  entry_frenzy_vs_vwap_pct ≤ +5.0 → its EMA50 shadow exit, then RE-ENTRY #1: the first 5m bar of the same UTC day, closing ≥ 15 min after
  the shadow exit, on which the FRENZY state is ON (services.frenzy.frenzy_walk on the 1500-bar window) and the bar is red / flat; entered at
  the next minute's open, EMA50 shadow exit. BAR (read at N ≥ 60 re-entry #1 fills on ≥ 20 days): re-entry mean > 0 ∧ day-block CI low > 0 ∧
  mean > 0 without the top 5 and the top 10 ∧ no pair > 35 % of the gain ∧ both halves > 0. Otherwise retired; never re-fit the +5.0 %.
TRACKER 5 — ATR_FAST_LOCK3 (reports/FRENZY_REDO_EXITS_2026-10-05.md §5a, frozen, observe-only): every FRENZY / WIDE first entry from
  SOFT_FROM tagged by the 30-min ATR% change at the signal bar — (ATR% now ÷ ATR% 6 bars earlier − 1) × 100, ATR = EWM(α 1/14) of the true
  range ÷ close on 5m bars (the study's atr_series) — "fast" when > +14.3 %. Year (engine cohort, pooled): the 3-pt trail beat the lock by
  +0.139 %/trade in that top tercile (CI +0.00…+0.29, both halves +, 2D null p 0.01) but fails drop-top-10 (−0.065) and walk-forward picked
  nothing. BAR (read on LOCK3 − LOCK2 for fast fills): ≥ 30 fills on ≥ 15 days → review candidate only if mean > 0 ∧ day CI low > 0 ∧ > 0
  without the top 5 and top 10 ∧ top pair < 50 % of the net; RETIRE if the mean ≤ 0 at ≥ 30 or no verdict by 60.
TRACKER 4 — WIDE_CHOPPY_OBS (DECISION_LOG 215, shipped OBSERVE-ONLY): every WIDE first entry from SOFT_FROM tagged by the engine's own stamp
  entry_frenzy_above_share (% of the episode's 5m closes at / above the spike VWAP at the signal) ≤ 67.8 = "would have been blocked". BAR:
  at ≥ 15 would-block fills → ARM REVIEW if their mean < 0 ∧ day CI high < 0 ∧ WR < 52 %; RETIRE if their mean ≥ 0; at ≥ 30 without an
  arm verdict → RETIRE. Frozen at 67.8, never re-fit (re-validation of the study on the engine's fresh bars pending).
TRACKER 3 — WIDE_BTC_SOFT (reports/FRENZY_REGIME_2026-10-05.md, frozen): every WIDE first entry tagged by BTC's RSI(12) on CLOSED 5m bars at
  the signal bar (ta.RSIIndicator(close, 12) — the study's ruler, NOT the bot's entry_btc_rsi stamp) ≤ 45 = "soft"; BTC 5m EMA20 slope shown
  alongside. Year: soft +1.00 / +0.64 %/day (Jan–Apr / May–Sep) vs not soft +0.09 / −0.34 — at the luck level (p 0.11), not a filter.
  BAR (review candidate only): ≥ 40 fills on ≥ 15 days in EACH state ∧ soft mean of day means > 0 ∧ day-block 95 % CI of the gap
  (soft − not soft) excludes 0 ∧ the not-soft cohort meets the expectancy bar (WR < 51.7 % breakeven ∧ day CI high < 0). RETIRE if the gap CI
  still includes 0 at ≥ 60 fills per state. Never armed from this tracker.
TRACKER 6 — WIDE_BY_CODE (2026-10-06, after RLC 10-05; reports/FRENZY_VS_WIDE_RLC_CAPTURE_2026-10-06.md §4c, observe line only): every WIDE
  fill opened from GC_FROM tagged by WHY FRENZY_LONG refused it — GREEN_BAR (ATR ≤ cap, green signal candle) / ATR_HIGH (ATR > cap, red or
  flat candle) / BOTH (ATR > cap AND green; the engine's journal says ATR_HIGH because frenzy_long_status judges the ATR first). Source, in
  order: the fill's own stamps (entry_atr_pct = the gate's wilder_atr_pct(closed[-300:]), entry_frenzy_bar_ret_pct > 0 = green) → the
  journal's FRENZY_GREEN_BAR / FRENZY_ATR_HIGH line at that bucket (+ the 5m kline's colour for ATR_HIGH) → a recompute from 5m klines with
  the engine's own wilder_atr_pct; code_src says which, code_kl is the kline recompute kept as a parity check. Year: GREEN half ≈ +0.01
  %/fill, ATR_HIGH half ≈ −0.53. Shown per code: N · days · WR · avg actual % · avg LOCK2 %. REVIEW DUE at ≥ 30 fills on ≥ 15 days per code.
TRACKER 7 — FRENZY_GREEN_CLOCK (V2, OBSERVE ONLY; post-hoc pocket, selection-adjusted p 0.87 — never armed from this tracker): every journal
  FRENZY_GREEN_BAR refusal from GC_FROM, replayed with the engine's functions (normal_hour_usd → frenzy_walk on the last 1,499 closed 5m
  bars → wilder_atr_pct → frenzy_long_status; parity = the replay also says FRENZY_GREEN_BAR). V2 = the setup's above_streak at the signal
  > 12 (FROZEN): the hour-above-average was already met, so the setup turned ON by clock / volume and the candle colour was luck (RLC's
  case). Each refusal is priced as if FRENZY_LONG had opened it, with the pre-registered shadow pricing (reports/FRENZY_GREEN_AND_WIDE_ATR_
  FORMAL_2026-10-06.md): entry = the first trade print ≥ the 5m close + 12 s (ticks, Binance aggTrades archive; until it is published the
  open of the signal-close minute on 1m bars = PROVISIONAL), the live lock exit (−3 until +3, then max(+2, peak − 2)), fees 0.09 + slippage
  0.10, 12 h cap. Tick stops can fire on wicks the live poller rides through (CLAUDE.md live-stopped rule) — stated on the table. LONG's
  remaining gates: market volume from WIDE's own lines on the same bar (GVOL_HIGH / UNREAD = blocked; a WIDE open or a WIDE refusal judged
  after the gvol gate — CHOPPY / LATE / DISLOC / OPEN_REFUSED = pass; else unknown → NOT counted, own line), LONG slots and the pair-day cap
  from the orders exports (hypothetical fills are not chained). A WIDE fill on the same bar is noted (overlap) — the hypothetical LONG is
  still priced separately at LONG sizing (0.32, no strong multiplier, as the pre-registration says). COUNTED = from GC_FROM (the floor is
  applied before the dedupe), eligible, gvol pass, the first refusal of its (pair, spike) WITHIN V2 and WITHIN V1 separately.
  BARS (FROZEN — FORMAL doc, end of Study B): ① V2 → FRENZY_LONG: N ≥ 30 on ≥ 15 days ∧ mean ≥ +0.30 ∧ day-block CI low > 0 ∧ WR ≥ 55 % ∧
  no pair > 25 % of the net ∧ mean after a 50 % haircut ≥ +0.15 → propose promotion at 0.32 (no strong), revert if the first 20 average < 0.
  ② V2 ∧ ATR ≤ 1.5 (Pattern-W): N ≥ 30 ∧ WR ≥ 70 % ∧ mean ≥ +0.50 ∧ CI low > 0 ∧ no pair > 25 % → propose; revert if the first 15 average < 0.
  ADDITIONS beyond the pre-registration (bar ① only, labelled): mean > 0 without the top 5; retire if mean ≤ 0 at ≥ 30 or no verdict by 60.
  V1 remainder = the other green refusals, same pricing, contrast only. Rows: reports/SCOUT_FRENZY_GREEN_CLOCK.csv (unreadable → a timestamped
  .bad, start empty); the journal's FRENZY lines are kept in reports/SCOUT_FRENZY_JOURNAL.csv (unreadable → copied to a timestamped .bad, never
  overwritten). One per pair-episode = spikes of a pair ≤ 30 min apart merged (episode_keys); a refusal whose replay at t says FRENZY_ON (a
  catch-up line, t ≠ the ON bar) is listed on its own line, never counted.
TRACKER 8 — GVOL_BLOCKED (2026-10-06, observe-only; DECISION_LOG 194 gate frenzy_gvol_max 1.0, whose revert gate reads only the fills it let
  through): every journal FRENZY_GVOL_HIGH / FRENZY_WIDE_GVOL_HIGH refusal (the gate sits after slots + the pair-day cap, before choppy /
  hold-green / LATE / DISLOC) replayed with the engine's functions — LONG = replay READY (lev 0.32 / 0.5 strong); WIDE = a fresh ATR_HIGH /
  GREEN_BAR refusal today's hold-green rule (frenzy_wide_hold_green_block, streak > 12 frozen) would still take (lev 0.2); a FRENZY_ON replay in
  state = a catch-up line (own line). BOTH sides — the blocked activations AND the live fills the gate let through — priced with the SAME
  ruler (_lock_shadow at the signal bar: 12 s, lock, 0.10 slip, ticks else 1m provisional); the let-through live actual is display-only.
  One per pair-episode (spikes ≤ 30 min apart merged); a blocked episode that also had a live fill is excluded and listed. From GC_FROM, DAY
  units; _GVOL_UNREAD on its own line. FROZEN bar: ≥ 20 signals on ≥ 10 days ∧ no day ≥ 50 % of the blocked net → blocked mean ≥ 0 ∧ ≥ the
  let-through mean → "review: the gate removes winners" (flag only); blocked mean ≤ −0.20 → "gate confirmed"; else inconclusive. Rows:
  reports/SCOUT_FRENZY_GVOL_BLOCKED.csv (non-final rows re-priced from their stored fields).
TRACKER 9 — VWAP_STOP (2026-10-06, observe-only; reports/FRENZY_STAIRCASE_STUDY_2026-10-06.md §3b / §5 "BP k 0.5", NOT established on the
  year: saved 72 (+400) vs deeper 103 (−395)): every FRENZY / WIDE fill the live −3 stop closed (close_reason STOP_LOSS) re-priced as: identical
  to live until the stop, then held and out at the first print after a 5m close that is ≤ −3 % net AND below VWAP × (1 − 0.5 × entry ATR %),
  hard floor −12, the lock unchanged (rules off once armed), 12 h cap, 0.09 fees + 0.10 slip; ticks else 1m pseudo prints O → H → L → C.
  Δ = shadow − actual. Fills live did NOT stop: Δ 0 by construction, and the shadow's pre-stop replica is run on them — a replica −3 before
  their live exit is counted on the parity line (wick evidence). Rows are validated before save (bad rows → a timestamped .bad); non-final
  rows re-price from their stored fields; a stopped fill without a VWAP stamp / live P&L is stored once as excluded. FROZEN gate on the first
  20 stopped fills from GC_FROM: Δ sum > 0 ∧ saved > deeper ∧ Δ sum > 0 on every sleeve with ≥ 5 of the 20 (≥ 1 such sleeve, else collecting)
  ∧ no fill > 50 % of the gain → "candidate for a pre-registered study", else "close the idea". Rows: reports/SCOUT_FRENZY_VWAP_STOP.csv.
Rows are stored in reports/SCOUT_FRENZY_EXITS.csv (keyed opened_at + pair) so fills survive their export leaving ~/Downloads; a row is FINAL
once its 12 h (and the re-entry's) have passed. 1m bars are coarser than the year studies' ticks (stated on the table).
"""
import glob
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from services.frenzy import (frenzy_walk, normal_hour_usd, frenzy_long_status, frenzy_flagged, frenzy_di_spread, frenzy_adx_delta,  # noqa: E402
                             FRENZY_WIDE_CODES, frenzy_wide_choppy, frenzy_wide_hold_green_block)
from services.surge import wilder_atr_pct  # noqa: E402

CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_EXITS.csv")
MIN, BAR, H = 60_000, 300_000, 3_600_000
FEE, CAP_MIN, NOTSTRETCHED_MAX, REENTRY_WAIT_MIN = 0.09, 720, 5.0, 15
ATRE_N, ATRE_DAYS, NS_N, NS_DAYS, PAIR_MAX = 60, 30, 60, 20, 35.0
EXITS = ["LOCK2", "LOCK3", "EMA20", "EMA50", "FIX3"]
VER = 6   # row schema / rules version: rows priced by an older version are recomputed (review) · 3 = + BTC RSI(12) · 4 = + above_share · 5 = + atr_chg30 · 6 = + WIDE refusal code
SOFT_RSI, SOFT_N, SOFT_DAYS, SOFT_RETIRE_N = 45.0, 40, 15, 60
SOFT_FROM = "2026-10-06T00:00:00"   # the trackers' cohort floor: WIDE fills opened from their registration (215 / 216)
CHOPPY_MAX, CHOPPY_N, CHOPPY_RETIRE_N = 67.8, 15, 30   # 🌀 215 observe-only: the frozen choppy-pump candidate on live WIDE fills
ATRF_MIN, ATRF_N, ATRF_DAYS, ATRF_RETIRE_N = 14.3, 30, 15, 60   # ⚡ ATR_FAST_LOCK3 watch line (FRENZY_REDO_EXITS_2026-10-05.md, frozen)
_BTC5 = {}   # per-run cache: BTC 5m klines by window end
# 🟢 2026-10-06 trackers 6 + 7 (registered; cohort floor = entries / signals from GC_FROM; everything below is FROZEN)
GC_FROM = "2026-10-07T00:00:00"
GC_SCAN_FROM = "2026-10-03T00:00:00"   # journal lines kept / priced from here; rows before GC_FROM are shown for reference, never counted
# Pre-registered bars: reports/FRENZY_GREEN_AND_WIDE_ATR_FORMAL_2026-10-06.md, end of Study B ("Pre-registered promotion bars", frozen,
# never re-fit). Shadow pricing there: live lock exit, 12 s entry, 0.10 slippage.
GC_STREAK = 12                                         # V2 = GREEN_BAR ∧ above_streak > 12 (∧ ATR ≤ 2.5, implied by the GREEN_BAR code)
V2_N, V2_DAYS, V2_MEAN, V2_WR, V2_PAIR = 30, 15, 0.30, 55.0, 25.0   # FORMAL bar 1: N ≥ 30 on ≥ 15 days · mean ≥ +0.30 · WR ≥ 55 % · no pair > 25 % of the net
V2_HAIRCUT, V2_HAIRCUT_MIN, V2_REVERT_N = 0.50, 0.15, 20            # … mean after a 50 % haircut ≥ +0.15 · promote at 0.32, no strong 0.5 · revert if the first 20 average < 0
W_ATR, W_N, W_WR, W_MEAN, W_PAIR, W_REVERT_N = 1.5, 30, 70.0, 0.50, 25.0, 15   # FORMAL bar 2 (Pattern-W): V2 ∧ ATR ≤ 1.5 · N ≥ 30 · WR ≥ 70 % · mean ≥ +0.50 · CI low > 0 · no pair > 25 % · revert if the first 15 average < 0
GC_RETIRE_N = 60                                       # ADDITION beyond the pre-registration: retire if the mean ≤ 0 at N ≥ 30, or no verdict by 60
ENTRY_LAG_MS, SLIP = 12_000, 0.10                      # FORMAL pricing: first print ≥ signal close + 12 s; 0.10 % slippage on top of the 0.09 % fees
BYCODE_N, BYCODE_DAYS = 30, 15
GC_VER = 2                                             # 2 = FORMAL pricing (12 s + slippage) and gvol from every post-gvol WIDE line
GC_TIME_BUDGET_S, TICK_DL_DEADLINE_S, TICK_FIRST_TRY_H = 180, 90, 8
GC_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_GREEN_CLOCK.csv")
JR_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_JOURNAL.csv")
TICK_CACHE = os.path.join(ROOT, "reports", "backtest_cache")   # the year studies' aggTrades cache (ticks/ or ticks_q/<PAIR>/<date>.npz)
TICK_DL_MAX = 4          # aggTrades day archives downloaded per run at most (the rest wait for the next run)
TICK_GIVEUP_D = 3        # an archive still missing this many days after its day ended → the row finalises on 1m bars
JR_MAX_AGE_D = 4         # journal exports read per run: modified within this many days (each export holds ~2.5 days; lines are kept)


# ─────────────────────────── pure exit walkers (selftest) ───────────────────────────
def walk(m1, e, kind, b5=None, atr0=None):
    """net % (fees in) and exit ms of one exit on 1m rows [open_ms, o, h, l, c] from the entry minute. b5 = DataFrame(t, c, e20, e50, atr)
    of 5m bars for the structure / ATR exits. Returns (pnl, exit_ms, how) or (None, None, 'no data')."""
    if not m1:
        return None, None, "no data"
    net = lambda p: (p / e - 1) * 100 - FEE
    trail = 3.0 if kind == "LOCK3" else 2.0
    pk = -1e9
    b5c = {int(t): (c, e20, e50, a) for t, c, e20, e50, a in b5.itertuples(index=False)} if b5 is not None else {}
    t_end = int(m1[0][0]) + CAP_MIN * MIN           # 12 h of CLOCK time, not of rows (a kline gap never stretches it — review)
    prev = None
    for b in m1:
        t, o, h, l, c = b[0], b[1], b[2], b[3], b[4]
        if t >= t_end:
            return net(prev[4]), t_end, "12 h cap"
        prev = b
        if kind == "FIX3":
            if net(l) <= -3:
                return -3.0, t + MIN, "stop"
            if net(h) >= 3:
                return 3.0, t + MIN, "take profit"
            continue
        armed = pk >= 3
        line = (max(2.0, pk - trail) if kind.startswith("LOCK") else 2.0) if armed else -3.0
        lpx = e * (1 + (line + FEE) / 100)
        if l <= lpx:
            return net(min(o, lpx)), t + MIN, ("floor / trail" if armed else "stop")
        pk = max(pk, net(h))
        if pk >= 3 and kind in ("EMA20", "EMA50", "ATRE") and (t + MIN) % BAR == 0:
            r = b5c.get(t + MIN - BAR)
            if r is not None:
                cc, e20, e50, a = r
                if (kind == "EMA20" and cc < e20) or (kind == "EMA50" and cc < e50) or (kind == "ATRE" and atr0 is not None and a is not None and a < atr0):
                    return net(cc), t + MIN, ("5m close < EMA" if kind != "ATRE" else "ATR < entry")
    return net(prev[4]), prev[0] + MIN, ("12 h cap" if prev[0] + MIN >= t_end else "open")


def wide_code(atr, bar_ret, cap):
    """🟢 WIDE_BY_CODE: why FRENZY_LONG refused a fresh setup that WIDE took. atr = the gate's ATR % (None = unreadable → WIDE never opens),
    bar_ret = the signal candle's close vs open % (> 0 = green; ≤ 0 = red / flat, the engine's bar_red). → 'GREEN_BAR' / 'ATR_HIGH' / 'BOTH',
    'NONE' (neither refusal: LONG would have opened — a parity alarm), or None when a needed input is missing."""
    try:
        if atr is None or (isinstance(atr, float) and np.isnan(atr)):
            return None
        hi = float(cap) > 0 and float(atr) > float(cap)
        if bar_ret is None or (isinstance(bar_ret, float) and np.isnan(bar_ret)):
            return None
        green = float(bar_ret) > 0
    except (TypeError, ValueError):
        return None
    return "BOTH" if (hi and green) else "ATR_HIGH" if hi else "GREEN_BAR" if green else "NONE"


def gc_entry(tt, pp, sig):
    """🟢 the pre-registered shadow entry (FORMAL doc, Study B): the first trade print at or after the signal close + ENTRY_LAG_MS (12 s).
    → (entry price, entry ms) or (None, None)."""
    tt = np.asarray(tt, dtype=np.int64)
    i = int(np.searchsorted(tt, int(sig) + ENTRY_LAG_MS, side="left"))
    return (float(np.asarray(pp)[i]), int(tt[i])) if i < len(tt) else (None, None)


def walk_ticks(tt, pp, e, t0, slip=0.0):
    """🟢 the live lock exit (−3 until the peak reaches +3, then max(+2, peak − 2)) on trade prints from t0, net of FEE; a print at or through
    the line exits AT that print (the line is set by the prints BEFORE it); 12 h of clock time. slip (%) is charged once on the result —
    the lines trigger on the bot's own net-of-fees P&L, as live. → (pnl, exit_ms, how)."""
    r = _walk_ticks(tt, pp, e, t0)
    return (r[0] - slip if r[0] is not None else None), r[1], r[2]


def _walk_ticks(tt, pp, e, t0):
    tt = np.asarray(tt, dtype=np.int64); pp = np.asarray(pp, dtype=float)
    m = tt >= int(t0)
    tt, pp = tt[m], pp[m]
    if not len(pp) or not e:
        return None, None, "no data"
    net = (pp / float(e) - 1) * 100 - FEE
    pk = np.maximum.accumulate(np.r_[-1e9, net[:-1]])
    armed = pk >= 3
    line = np.where(armed, np.maximum(2.0, pk - 2.0), -3.0)
    t_end = int(t0) + CAP_MIN * MIN
    inc = tt < t_end
    hit = np.flatnonzero((net <= line) & inc)
    if len(hit):
        i = int(hit[0])
        return float(net[i]), int(tt[i]), ("floor / trail" if armed[i] else "stop")
    if not inc.all():
        j = int(np.flatnonzero(inc)[-1]) if inc.any() else 0
        return float(net[j]), t_end, "12 h cap"
    return float(net[-1]), int(tt[-1]), "open"


def day_ci(x, days, n=3000, seed=7):
    g = pd.DataFrame(dict(x=x, d=days)).groupby("d").x.agg(["sum", "count"])
    if len(g) < 3:
        return None
    s, c = g["sum"].values, g["count"].values
    r = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    return tuple(np.percentile(s[r].sum(1) / c[r].sum(1), [2.5, 97.5]))


def bar_check(df, col, n_min, d_min):
    """the frozen review bar on a column of per-fill values (Δ or re-entry %). → (state, text)."""
    x = df[col].dropna()
    g = df.loc[x.index]
    n, nd = len(x), g.day.nunique()
    if n < n_min or nd < d_min:
        return "collecting", f"{n}/{n_min} fills · {nd}/{d_min} days"
    ci = day_ci(x.values, g.day.values)
    top = x.sort_values(ascending=False)
    gain = g.assign(v=x).groupby("pair").v.sum()
    pshare = (gain.max() / x.sum() * 100) if x.sum() > 0 else float("inf")   # the studies' ruler: top pair ÷ the NET total (review)
    h1 = g.day < g.day.sort_values().iloc[len(g) // 2]
    ok = (x.mean() > 0 and ci and ci[0] > 0 and top.iloc[5:].mean() > 0 and top.iloc[10:].mean() > 0 and pshare <= PAIR_MAX
          and x[h1].mean() > 0 and x[~h1].mean() > 0)
    return ("review" if ok else "retire"), (f"mean {x.mean():+.3f} · CI [{ci[0]:+.2f}, {ci[1]:+.2f}] · w/o top5 {top.iloc[5:].mean():+.3f} · "
                                            f"w/o top10 {top.iloc[10:].mean():+.3f} · top pair {pshare:.0f} % · halves {x[h1].mean():+.2f} / {x[~h1].mean():+.2f}")


# ─────────────────────────── data ───────────────────────────
def _kl(sym, tf, start, end):
    step = {"1m": MIN, "5m": BAR, "1h": H}[tf]
    out, s = {}, int(start)
    while s < end:
        q = urllib.parse.urlencode(dict(symbol=sym, interval=tf, startTime=s, endTime=int(end), limit=1500))
        r = json.loads(urllib.request.urlopen(f"https://fapi.binance.com/fapi/v1/klines?{q}", timeout=20).read())
        if not r:
            break
        for x in r:
            out[int(x[0])] = [int(x[0])] + [float(v) for v in x[1:6]]
        nxt = int(r[-1][0]) + step
        if nxt <= s:
            break
        s = nxt
        time.sleep(0.05)
    return [out[k] for k in sorted(out)]


def _fills():
    fr = []
    cols = ("opened_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "entry_atr_pct", "entry_frenzy_vs_vwap_pct",
            "entry_frenzy_above_share", "entry_frenzy_bar_ret_pct", "closed_at", "close_reason", "entry_frenzy_vwap", "entry_frenzy_spike_at")
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols)
        except Exception:
            continue
        if {"opened_at", "entry_strategy", "entry_price"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(cols) + ["k"])
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable").reindex(columns=list(cols) + ["_m"])   # an export missing a column never breaks the section
    o["k"] = o.opened_at.astype(str).str[:19]
    o = o.drop_duplicates(["k", "pair", "direction"], keep="last")
    o = o[o.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (o.direction.astype(str) == "LONG")]
    return o


def btc_soft_reading(t_in):
    """(RSI(12), EMA20 slope %) of BTC on the CLOSED 5m bars up to the signal bar (the last bar closed at or before the entry). Wilder RSI as
    ta.RSIIndicator (ewm alpha 1/12, adjust=False). None, None when unreadable."""
    end = (int(t_in) // BAR) * BAR                     # bars with open < end are closed by the entry
    if end not in _BTC5:
        _BTC5[end] = [b for b in _kl("BTCUSDT", "5m", end - 400 * BAR, end) if b[0] + BAR <= end]
    rows = _BTC5[end]
    if len(rows) < 100 or rows[-1][0] != end - BAR:   # the SIGNAL bar itself must be there — never read the bar before it (review)
        return None, None
    c = pd.Series([r[4] for r in rows], dtype=float)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 12, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 12, adjust=False).mean()
    rsi = float((100 - 100 / (1 + up / dn)).iloc[-1]) if float(dn.iloc[-1]) > 0 else 100.0
    e20 = c.ewm(span=20, adjust=False).mean()
    return rsi, float((e20.iloc[-1] / e20.iloc[-2] - 1) * 100)


def _journal(now_ms):
    """🟢 the decision journals' FRENZY lines (BLOCK lines whose gate starts FRENZY, OPEN lines of a FRENZY strategy) → DataFrame(t, e, pair,
    gate, strategy), t = the journal's second (a BLOCK line's t = the signal bar's CLOSE). Read from the exports modified in the last
    JR_MAX_AGE_D days and merged into JR_CSV, so lines survive their export leaving ~/Downloads. An unreadable JR_CSV is copied to .bad and
    NEVER overwritten this run (its history would be lost). Never raises (an empty frame instead)."""
    cols = ["t", "e", "pair", "gate", "strategy"]
    can_write = True
    old = pd.DataFrame(columns=cols)
    if os.path.exists(JR_CSV):
        try:
            old = pd.read_csv(JR_CSV, dtype=str, keep_default_na=False)
            if not set(cols) <= set(old.columns):
                raise ValueError("columns missing")
        except Exception:
            can_write = False; old = pd.DataFrame(columns=cols)
            try:
                import shutil
                shutil.copyfile(JR_CSV, _bad_path(JR_CSV))
            except Exception:
                pass
    rows = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")):
        try:
            if os.path.getmtime(f) * 1000 < now_ms - JR_MAX_AGE_D * 86_400_000:
                continue
            with open(f, "rb") as fh:
                for ln in fh:
                    if b"FRENZY" not in ln:
                        continue
                    p = ln.decode("utf-8", "replace").rstrip("\r\n").split(",")
                    if len(p) < 8:
                        continue
                    if (p[1] == "BLOCK" and p[4].startswith("FRENZY")) or (p[1] == "OPEN" and p[7].startswith("FRENZY")):
                        rows.append((p[0][:19], p[1], p[2], p[4], p[7]))
        except Exception:
            continue
    j = pd.concat([old.reindex(columns=cols), pd.DataFrame(rows, columns=cols)], ignore_index=True).fillna("")
    j = j[j.t.astype(str) >= GC_SCAN_FROM].drop_duplicates(cols).sort_values("t")
    try:
        if can_write and len(j) and len(j) != len(old):
            tmp = f"{JR_CSV}.{os.getpid()}.tmp"; j.to_csv(tmp, index=False); os.replace(tmp, JR_CSV)
    except Exception:
        pass
    return j.reset_index(drop=True)


def _parse_aggtrades(raw):
    """aggTrades CSV bytes → (t int64 ms, p float64), parsed numerically; a header line (newer archives) is sniffed and skipped."""
    first = raw[:200].split(b"\n", 1)[0]
    hdr = 0 if first[:1] and not first[:1].isdigit() else None
    d = pd.read_csv(io.BytesIO(raw), header=hdr, usecols=[1, 5], dtype={1: np.float64, 5: np.int64} if hdr is None else None)
    d.columns = ["p", "t"]
    t = pd.to_numeric(d.t, errors="raise").values.astype(np.int64); p = pd.to_numeric(d.p, errors="raise").values.astype(np.float64)
    if not len(t):
        raise ValueError("empty archive")
    return t, p


def _tick_day(pair, date, now_ms, budget):
    """one pair-day of aggTrades: (t int64 ms, p float) from the year studies' cache, else downloaded once the day's archive should exist
    (first attempt ≥ TICK_FIRST_TRY_H after the day ended; ≤ budget['dl'] real downloads per run — a 404 costs none; a monotonic deadline per
    download and for the whole run). Prices are float32-rounded on every path (the cache's precision). → (state, t, p): 'ok' / 'pending'
    (not yet / budget or time spent / network / bad archive) / 'missing' (absent or unreadable TICK_GIVEUP_D days after the day)."""
    for base in ("ticks_q", "ticks"):
        f = os.path.join(TICK_CACHE, base, pair, f"{date}.npz")
        if os.path.exists(f):
            try:
                with np.load(f) as z:
                    return "ok", z["t"].astype(np.int64), z["p"].astype(np.float32).astype(np.float64)
            except Exception:
                if base == "ticks":   # our own cache format: a corrupt file is removed and re-fetched later
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                    return "pending", None, None
                continue              # ticks_q belongs to the replay tooling — never touched
    day_end = int(pd.Timestamp(date, tz="UTC").value // 1_000_000) + 86_400_000
    giveup = now_ms > day_end + TICK_GIVEUP_D * 86_400_000
    fail = "missing" if giveup else "pending"
    if now_ms < day_end + TICK_FIRST_TRY_H * H or budget.get("dl", 0) <= 0 or time.monotonic() > budget.get("deadline", float("inf")):
        return "pending", None, None
    q = urllib.parse.quote(pair)
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{q}/{q}-aggTrades-{date}.zip"
    part = os.path.join(TICK_CACHE, "ticks", pair, f".{date}.{os.getpid()}.zip.part")
    try:
        resp = urllib.request.urlopen(url, timeout=30)
    except urllib.error.HTTPError as ex:
        return (fail if ex.code == 404 else "pending"), None, None
    except Exception:
        return "pending", None, None
    budget["dl"] -= 1                                   # only a real download spends the per-run budget
    stop_at = min(time.monotonic() + TICK_DL_DEADLINE_S, budget.get("deadline", float("inf")))
    try:
        os.makedirs(os.path.dirname(part), exist_ok=True)
        with resp, open(part, "wb") as out:
            while True:
                if time.monotonic() > stop_at:
                    raise TimeoutError("download deadline")
                ch = resp.read(1 << 20)
                if not ch:
                    break
                out.write(ch)
        with zipfile.ZipFile(part) as z:
            t, p = _parse_aggtrades(z.read(z.namelist()[0]))
    except TimeoutError:
        return "pending", None, None
    except Exception:                                   # BadZipFile, parse errors, disk: retried, then given up like a 404
        return fail, None, None
    finally:
        try:
            os.remove(part)
        except OSError:
            pass
    p32 = p.astype(np.float32)
    fp = os.path.join(TICK_CACHE, "ticks", pair, f"{date}.npz")
    tmp = fp[:-4] + f".{os.getpid()}.part.npz"          # the cache's own format + atomic write (scripts/backtest_fetch_ticks.py)
    try:
        np.savez_compressed(tmp, t=t, p=p32); os.replace(tmp, fp)
    except Exception:
        pass
    return "ok", t, p32.astype(np.float64)


def _ticks(pair, t0, t1, now_ms, budget):
    """aggTrades prints in [t0, t1] → ('ok', t, p) when every UTC day is available; else ('pending' | 'missing', None, None)."""
    ts, ps = [], []
    for d in pd.date_range(pd.Timestamp(t0, unit="ms").normalize(), pd.Timestamp(t1, unit="ms").normalize()):
        st, t, p = _tick_day(pair, f"{d:%Y-%m-%d}", now_ms, budget)
        if st != "ok":
            return st, None, None
        ts.append(t); ps.append(p)
    t = np.concatenate(ts); p = np.concatenate(ps); o = np.argsort(t, kind="stable"); t, p = t[o], p[o]
    m = (t >= t0) & (t <= t1)
    return "ok", t[m], p[m]


def _cfg():
    c = json.load(open(os.path.join(ROOT, "trading_config.json")))
    return SimpleNamespace(**{**c, **(c.get("thresholds") or {})})


def _price(r, th, now_ms, first, J=None):
    """one fill → dict of every exit + the shadows (first = the earliest fill of its sleeve / pair / UTC day: the frozen shadows read first
    entries only — review). Raises on data trouble (the caller keeps the old row)."""
    sym = str(r.pair); e = float(r.entry_price)
    t_in = int(pd.Timestamp(r.k, tz="UTC").value // 1_000_000)
    m0 = t_in // MIN * MIN
    day_end = (t_in // 86_400_000 + 1) * 86_400_000
    horizon = min(now_ms, day_end + CAP_MIN * MIN)
    m1 = [b for b in _kl(sym, "1m", m0, horizon) if b[0] + MIN <= now_ms]
    b5raw = [b for b in _kl(sym, "5m", m0 - 1800 * BAR, horizon) if b[0] + BAR <= now_ms]
    if len(m1) < 2 or len(b5raw) < 400:
        raise ValueError("klines unavailable")
    b5 = pd.DataFrame(b5raw, columns=["t", "o", "h", "l", "c", "v"])
    b5["e20"] = b5.c.ewm(span=20, adjust=False).mean(); b5["e50"] = b5.c.ewm(span=50, adjust=False).mean()
    atr = [None] * len(b5raw)
    for i in range(len(b5raw)):
        if b5raw[i][0] >= m0 - BAR:               # only bars from the signal bar on are ever read
            atr[i] = wilder_atr_pct(b5raw[max(0, i - 299):i + 1])
    b5["atr"] = atr
    # ⚡ ATR_FAST_LOCK3: the study's ATR ruler (EWM α 1/14 of the true range ÷ close, over the whole fetched history), signal bar = the last
    # bar closed by the entry; change vs 6 bars earlier
    _h = b5.h.values; _l = b5.l.values; _c = b5.c.values
    _pc = np.r_[_c[0], _c[:-1]]; _tr = np.maximum(_h - _l, np.maximum(abs(_h - _pc), abs(_l - _pc)))
    _atrp = pd.Series(_tr).ewm(alpha=1 / 14, adjust=False).mean().values / _c * 100
    _js = int(np.searchsorted(b5.t.values, (t_in // BAR) * BAR - BAR))
    atr_chg30 = (float(_atrp[_js] / _atrp[_js - 6] - 1) * 100
                 if _js < len(b5) and int(b5.t.values[_js]) == (t_in // BAR) * BAR - BAR and _js >= 30 else None)
    b5k = b5[["t", "c", "e20", "e50", "atr"]]
    m1in = [list(b) for b in m1 if b[0] >= m0]
    if m1in and m1in[0][0] == m0:   # the entry minute: only its prints AFTER the fill count → flatten it to entry → its close (review)
        m1in[0] = [m0, e, max(e, m1in[0][4]), min(e, m1in[0][4]), m1in[0][4], m1in[0][5]]
    out = dict(k=r.k, pair=sym, sleeve=str(r.entry_strategy).replace("FRENZY_", ""), day=r.k[:10], entry=e, first=bool(first), ver=VER,
               closed=str(r.status).upper() == "CLOSED",
               actual=float(r.pnl_percentage) if pd.notna(r.pnl_percentage) and str(r.status).upper() == "CLOSED" else None,
               atr_entry=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) else None,
               vs_vwap=float(r.entry_frenzy_vs_vwap_pct) if pd.notna(r.entry_frenzy_vs_vwap_pct) else None,
               above_share=float(r.entry_frenzy_above_share) if pd.notna(r.entry_frenzy_above_share) else None)
    fin = [out["closed"]]
    ema50_exit = None
    for kind in EXITS:
        p, x, how = walk(m1in, e, kind, b5k)
        out[kind] = p; fin.append(how != "open")
        if kind == "EMA50" and how != "open":
            ema50_exit = x                          # the re-entry search starts only from a REAL shadow exit (review)
    # ⛔ ATRE shadow retired (218) — no longer priced
    out["atr_chg30"] = atr_chg30
    # 🟢 WIDE_BY_CODE: why LONG refused this WIDE fill (stamps → journal → klines; code_kl = the kline recompute, kept as a parity check)
    out["code"] = out["code_src"] = out["code_kl"] = out["code_jr"] = None
    out["bar_ret"] = float(r.entry_frenzy_bar_ret_pct) if pd.notna(getattr(r, "entry_frenzy_bar_ret_pct", None)) else None
    if out["sleeve"] == "WIDE":
        cap = float(getattr(th, "frenzy_max_atr_pct", 2.5) or 0)
        sig = (t_in // BAR) * BAR - BAR                    # the signal bar = the last bar closed by the entry
        ks = [i for i, b in enumerate(b5raw) if b[0] == sig]
        kl_ret = kl_atr = None
        if ks:
            sb = b5raw[ks[0]]
            kl_ret = (sb[4] / sb[1] - 1) * 100 if sb[1] else None
            kl_atr = wilder_atr_pct(b5raw[max(0, ks[0] - 299):ks[0] + 1])
        out["code_kl"] = wide_code(kl_atr, kl_ret, cap)
        jl = None
        if J is not None and len(J):
            ts = pd.Timestamp(sig + BAR, unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
            g = J[(J.t == ts) & (J.pair == sym) & J.gate.isin(["FRENZY_GREEN_BAR", "FRENZY_ATR_HIGH"])]
            jl = g.gate.iloc[0] if len(g) else None
        out["code_jr"] = ("GREEN_BAR" if jl == "FRENZY_GREEN_BAR" else wide_code(float("inf"), kl_ret, cap) if jl == "FRENZY_ATR_HIGH" else None)
        if out["atr_entry"] is not None and out["bar_ret"] is not None:
            out["code"], out["code_src"] = wide_code(out["atr_entry"], out["bar_ret"], cap), "stamp"
        elif jl == "FRENZY_GREEN_BAR":
            out["code"], out["code_src"] = "GREEN_BAR", "journal"
        elif jl == "FRENZY_ATR_HIGH":
            out["code"], out["code_src"] = wide_code(float("inf"), kl_ret, cap), "journal+kline"
        else:
            out["code"], out["code_src"] = out["code_kl"], "kline"
    out["btc_rsi12"], out["btc_e20_slope"] = btc_soft_reading(t_in) if out["sleeve"] == "WIDE" else (None, None)
    if out["sleeve"] == "WIDE" and out["btc_rsi12"] is None:
        fin.append(False)                                 # a WIDE row without its BTC reading stays provisional → retried next run (review)
    out["re1"] = None; out["re1_at"] = None
    atr_max = float(getattr(th, "frenzy_max_atr_pct", 2.5) or 2.5)
    if False:   # ⛔ NOTSTRETCHED re-entry shadow retired (218) — the search below is kept only for the record
        h1 = _kl(sym, "1h", m0 - 800 * H, m0)
        cand = [i for i, b in enumerate(b5raw) if b[0] + BAR >= ema50_exit + REENTRY_WAIT_MIN * MIN and b[0] + BAR < day_end]
        for i in cand:
            bar = b5raw[i]; close_ms = bar[0] + BAR
            nh = normal_hour_usd(h1, bar[0])
            ep = frenzy_walk(b5raw[max(0, i - 1499):i + 1], nh, th) if nh else None
            if not (ep and ep.get("in_state")):
                continue
            a_i = wilder_atr_pct(b5raw[max(0, i - 299):i + 1])
            red = bar[4] <= bar[1]
            # the study's chain rules (frenzy_exit_combo_test PART 2): FRENZY = ON ∧ red / flat ∧ ATR ≤ the cap; WIDE = its fresh bars, or
            # ON ∧ red / flat ∧ ATR above the cap. (The live market-volume gate is NOT applied here — stated on the table.)
            ok = (red and a_i is not None and a_i <= atr_max) if out["sleeve"] == "LONG" else \
                 (bool(ep.get("fresh_on")) or (red and a_i is not None and a_i > atr_max))
            if not ok:
                continue
            m1r = [b for b in m1 if b[0] >= close_ms]
            if not m1r:
                break
            p, x, how = walk(m1r, m1r[0][1], "EMA50", b5k)
            out["re1"] = p; out["re1_at"] = pd.Timestamp(close_ms, unit="ms").strftime("%m-%d %H:%M"); fin.append(how != "open")
            break
        fin.append(now_ms >= day_end + CAP_MIN * MIN or out["re1"] is not None)
    out["final"] = all(fin)
    return out


def run(now_ms=None):
    now_ms = int(now_ms or time.time() * 1000)
    old = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
    if len(old) and "ver" not in old:
        old["ver"] = 1
    done = {(a, b) for a, b, f, v in zip(old.get("k", []), old.get("pair", []), old.get("final", []), old.get("ver", []))
            if str(f) in ("True", "1", "1.0") and str(v) in (str(VER), f"{VER}.0")}
    th = _cfg(); rows = []; err = 0
    F = _fills()
    try:
        J = _journal(now_ms)
    except Exception:
        J = None
    # first entry = the earliest fill of its sleeve / pair / UTC day across the stored rows AND the exports (an old export can surface late)
    keys = pd.concat([F[["k", "pair", "entry_strategy"]].assign(sl=F.entry_strategy.astype(str).str.replace("FRENZY_", "")),
                      old[["k", "pair", "sleeve"]].rename(columns={"sleeve": "sl"}) if len(old) else pd.DataFrame(columns=["k", "pair", "sl"])],
                     ignore_index=True)
    keys["day"] = keys.k.astype(str).str[:10]
    firsts = set(keys.sort_values("k").drop_duplicates(["sl", "pair", "day"]).apply(lambda x: (x.k, x.pair), axis=1)) if len(keys) else set()
    for r in F.itertuples():
        if (r.k, r.pair) in done:
            continue
        try:
            rows.append(_price(r, th, now_ms, (r.k, r.pair) in firsts, J))
        except Exception:
            err += 1
    new = pd.DataFrame(rows)
    if len(new) and len(old):   # the retired shadows' recorded values survive a reprice (review): copy them forward
        keep_cols = [c for c in ("ATRE", "re1", "re1_at") if c in old]
        if keep_cols:
            prev = old.drop_duplicates(["k", "pair"], keep="last").set_index(["k", "pair"])[keep_cols]
            idx = pd.MultiIndex.from_arrays([new.k, new.pair])
            for c in keep_cols:
                new[c] = prev[c].reindex(idx).values
    allr = pd.concat([old, new], ignore_index=True) if rows else old
    if len(allr):
        allr = allr.drop_duplicates(["k", "pair"], keep="last").sort_values("k")
        allr["first"] = [(k, p) in firsts for k, p in zip(allr.k, allr.pair)]   # every stored row's key is in `keys`, so this is complete
        tmp = CSV + ".tmp"; allr.to_csv(tmp, index=False); os.replace(tmp, CSV)
    L = ["## 🎯 FRENZY / WIDE exit shadows per fill (pre-registered, OBSERVE only — the live exit stays the lock)", "",
         "Every live FRENZY / WIDE fill re-priced from its ACTUAL entry on 1m bars (fees in, 12 h cap; coarser than the year studies' ticks). "
         "LOCK2 = live exit · LOCK3 = 3-pt trail · EMA20/EMA50 = +2 floor then the first 5m close below the EMA · FIX3 = old fixed +3/−3 · "
         "ATR Δ30m = the 30-min ATR change at the signal (ATR_FAST_LOCK3 watch) · "
         "Trackers read FIRST entries only (² = a later fill of its pair-day). Actual = the exit live at the time (the fixed +3 TP shipped 10-04 19:23, the lock 10-05 ~16:00); "
         "FIX3 fills at exactly ±3 (reference column). ᵖ = not final yet.", ""]
    if not len(allr):
        return L + ["No FRENZY / WIDE fill in the exports yet.", ""] + _extras(now_ms, th, F, J, allr)
    show = allr.tail(15)
    L += ["| Opened UTC | Pair | Sleeve | ATR | ATR Δ30m | vs avg | Actual | LOCK2 | LOCK3 | EMA20 | EMA50 | FIX3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.2f}"
    for r in show.itertuples():
        fl = ("" if str(r.final) in ("True", "1", "1.0") else "ᵖ") + ("" if r.first else "²")
        L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{fl} | {str(r.pair).replace('USDT', '')} | {r.sleeve} | {f(r.atr_entry)} | {f(getattr(r, 'atr_chg30', None))} | "
                 f"{f(r.vs_vwap)} | {f(r.actual)} | {f(r.LOCK2)} | {f(r.LOCK3)} | {f(r.EMA20)} | {f(r.EMA50)} | {f(r.FIX3)} |")
    fin = allr[allr.final.astype(str).isin(["True", "1", "1.0"])].copy()
    L += ["", "| Final fills | N | " + " | ".join(EXITS) + " |", "|---|---|" + "---|" * len(EXITS)]
    for nm, g in (("all", fin), ("FRENZY_LONG", fin[fin.sleeve == "LONG"]), ("WIDE", fin[fin.sleeve == "WIDE"])):
        if len(g):
            L.append(f"| {nm} | {len(g)} | " + " | ".join(f"{g[c].mean():+.2f}" for c in EXITS) + " |")
    L += ["", "_ATRE and NOTSTRETCHED shadows retired 2026-10-06 (DECISION_LOG 218: evidence reversed on the engine cohort)._"]
    if "atr_chg30" in fin:
        st, tx = atrfast_check(fin[fin["first"] & fin.atr_chg30.notna() & (fin.k.astype(str) >= SOFT_FROM)])
        L += [f"**ATR_FAST_LOCK3 watch (30-min ATR change > +{ATRF_MIN:g} % at the signal → 3-pt trail vs the lock):** "
              + {"review": "📋 REVIEW CANDIDATE", "retire": "❌ retire", "collecting": "⏳ collecting"}.get(st, st) + f" ({tx})"]
    if "above_share" in fin:
        st, tx = choppy_check(fin[(fin.sleeve == "WIDE") & fin["first"] & fin.above_share.notna() & (fin.k.astype(str) >= SOFT_FROM)])
        L += [f"**WIDE_CHOPPY_OBS (215, observe-only — the bot still takes these):** "
              + {"arm_review": "📋 ARM REVIEW (would-block fills clearly losing)", "retire": "❌ retire the candidate", "collecting": "⏳ collecting"}.get(st, st) + f" ({tx})"]
    if "btc_rsi12" in fin:
        st, tx = soft_check(fin[(fin.sleeve == "WIDE") & fin["first"] & fin.btc_rsi12.notna() & (fin.k.astype(str) >= SOFT_FROM)])
        L += [f"**WIDE_BTC_SOFT tracker (BTC 5m RSI(12) ≤ {SOFT_RSI:g} at the signal; WIDE first entries from {SOFT_FROM[:10]}):** "
              + {"review": "📋 REVIEW CANDIDATE", "retire": "❌ gap unclear at ≥ 60 per state → retire", "collecting": "⏳ collecting"}.get(st, st) + f" ({tx})"]
    if "code" in fin:
        try:
            L += bycode_lines(fin[(fin.sleeve == "WIDE") & (fin.k.astype(str) >= GC_FROM)])
        except Exception as _bx:
            L += ["", f"_WIDE_BY_CODE unavailable this run ({str(_bx)[:120]})._"]
    if err:
        L.append(f"_{err} fill(s) not priced this run (klines unavailable) — retried next run._")
    return L + [""] + _extras(now_ms, th, F, J, allr)


def atrfast_check(w):
    """frozen ATR_FAST_LOCK3 bar on first entries with an ATR-change reading: Δ = LOCK3 − LOCK2 on the 'fast' fills."""
    f = w[(w.atr_chg30 > ATRF_MIN) & w.LOCK2.notna() & w.LOCK3.notna()].assign(d=lambda x: x.LOCK3 - x.LOCK2)
    n, nd = len(f), f.day.nunique()
    txt = f"fast fills {n} / {nd} d" + (f" · Δ(3-pt − lock) {f.d.mean():+.2f} %" if n else "") + f" · other fills {len(w) - n} (bar ≥ {ATRF_N} fast fills on ≥ {ATRF_DAYS} days)"
    if n >= ATRF_RETIRE_N or (n >= ATRF_N and f.d.mean() <= 0):   # retirement is checked BEFORE the days gate (review)
        ok_days = nd >= ATRF_DAYS
    elif n < ATRF_N or nd < ATRF_DAYS:
        return "collecting", txt
    else:
        ok_days = True
    ci = day_ci(f.d.values, f.day.values) if nd >= 3 else None
    top = f.d.sort_values(ascending=False)
    pshare = (f.groupby("pair").d.sum().max() / f.d.sum() * 100) if f.d.sum() > 0 else float("inf")
    txt += (f" · CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "") + f" · w/o top5 {top.iloc[5:].mean():+.2f} · w/o top10 {top.iloc[10:].mean():+.2f} · top pair {pshare:.0f} %"
    if ok_days and f.d.mean() > 0 and ci and ci[0] > 0 and top.iloc[5:].mean() > 0 and top.iloc[10:].mean() > 0 and pshare < 50:
        return "review", txt
    if f.d.mean() <= 0 or n >= ATRF_RETIRE_N:
        return "retire", txt
    return "collecting", txt


def choppy_check(w):
    """215 observe-only bar on live WIDE first entries: would-block = stamped above_share ≤ 67.8 (LOCK2 = the live exit's result)."""
    w = w[w.LOCK2.notna()]
    wb, kp = w[w.above_share <= CHOPPY_MAX], w[w.above_share > CHOPPY_MAX]
    mf = lambda g: f"{g.LOCK2.mean():+.2f} %" if len(g) else "–"
    txt = f"would-block {len(wb)} fills / {wb.day.nunique()} d · {mf(wb)}  vs  kept {len(kp)} · {mf(kp)} (bar: {CHOPPY_N} would-block fills)"
    if len(wb) < CHOPPY_N:
        return "collecting", txt
    ci = day_ci(wb.LOCK2.values, wb.day.values)
    if wb.LOCK2.mean() < 0 and ci and ci[1] < 0 and (wb.LOCK2 > 0).mean() * 100 < 52:
        return "arm_review", txt + f" · CI [{ci[0]:+.2f}, {ci[1]:+.2f}]"
    if wb.LOCK2.mean() >= 0 or len(wb) >= CHOPPY_RETIRE_N:
        return "retire", txt
    return "collecting", txt


def soft_check(w):
    """the frozen WIDE_BTC_SOFT bar on WIDE first entries with a BTC RSI(12) reading (LOCK2 = the live exit's result)."""
    soft = w.btc_rsi12 <= SOFT_RSI
    a, b = w[soft], w[~soft]
    na, nb, da, db_ = len(a), len(b), a.day.nunique(), b.day.nunique()
    mf = lambda g: f"{g.LOCK2.mean():+.2f} %" if len(g) else "–"
    txt = f"soft {na} fills / {da} d · {mf(a)}  vs  not soft {nb} / {db_} d · {mf(b)}"
    if min(na, nb) < SOFT_N or min(da, db_) < SOFT_DAYS:
        return "collecting", txt + f" (bar ≥ {SOFT_N} fills on ≥ {SOFT_DAYS} days per state)"
    am, bm = a.groupby("day").LOCK2.mean(), b.groupby("day").LOCK2.mean()
    days = np.array(sorted(set(am.index) | set(bm.index)))
    rng = np.random.default_rng(7); gaps = []
    for _ in range(3000):                       # JOINT day-block bootstrap: one draw of days serves both states (they share days — review)
        dr = rng.choice(days, len(days))
        ga, gb = am.reindex(dr).dropna(), bm.reindex(dr).dropna()
        if len(ga) and len(gb):
            gaps.append(ga.mean() - gb.mean())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    nci = day_ci(b.LOCK2.values, b.day.values)
    ok = am.mean() > 0 and lo > 0 and (b.LOCK2 > 0).mean() * 100 < 51.7 and nci and nci[1] < 0
    txt += f" · gap CI [{lo:+.2f}, {hi:+.2f}]"
    if ok:
        return "review", txt
    if min(na, nb) >= SOFT_RETIRE_N and lo <= 0:   # gap unclear OR the wrong way (review) — at 60 per state either retires it
        return "retire", txt
    if min(na, nb) >= SOFT_RETIRE_N:
        return "retire", txt + " · gap clear but the soft / not-soft expectancy legs failed"
    return "collecting", txt


# ─────────────────────────── 🟢 trackers 6 (WIDE_BY_CODE) + 7 (FRENZY_GREEN_CLOCK) ───────────────────────────
CODES = ("GREEN_BAR", "ATR_HIGH", "BOTH")
_TRUE = ("True", "1", "1.0")


def bycode_stats(w):
    """WIDE_BY_CODE per refusal code on final WIDE fills → list of dicts (code, n, days, wr, act, lock, src, status). WR / avg on the actual
    fill P&L (the exit live at the time — the lock from 10-05 ~16:00); LOCK2 = the live lock re-priced on 1m bars."""
    out = []
    w = w.assign(code=w["code"].where(w["code"].isin(CODES), "UNKNOWN"))
    for c in CODES + ("UNKNOWN",):
        g = w[w.code == c]
        if c == "UNKNOWN" and not len(g):
            continue
        a = pd.to_numeric(g.actual, errors="coerce").dropna()
        n, nd = len(g), g.day.nunique()
        out.append(dict(code=c, n=n, days=nd, wr=((a > 0).mean() * 100 if len(a) else None), act=(a.mean() if len(a) else None),
                        lock=(pd.to_numeric(g.LOCK2, errors="coerce").mean() if n else None),
                        src=" ".join(f"{k} {v}" for k, v in g.code_src.fillna("?").value_counts().items()),
                        status=("review" if (c != "UNKNOWN" and n >= BYCODE_N and nd >= BYCODE_DAYS) else "collecting")))
    return out


def bycode_lines(w):
    f = lambda v, d=2: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.{d}f}"
    L = ["", f"**WIDE_BY_CODE (observe line — why FRENZY_LONG refused each WIDE fill; every WIDE fill from {GC_FROM[:10]}; "
             f"review due at ≥ {BYCODE_N} fills on ≥ {BYCODE_DAYS} days per code; year: GREEN ≈ +0.01 %/fill, ATR_HIGH ≈ −0.53):**", "",
         "| Code | N | Days | WR | Avg actual % | Avg LOCK2 % | Source | Status |", "|---|---|---|---|---|---|---|---|"]
    for s in bycode_stats(w):
        wr = "–" if s["wr"] is None else f"{s['wr']:.0f} %"
        L.append(f"| {s['code']} | {s['n']} | {s['days']} | {wr} | {f(s['act'])} | "
                 f"{f(s['lock'])} | {s['src'] or '–'} | {'📋 REVIEW DUE' if s['status'] == 'review' else '⏳ collecting'} |")
    lab = lambda m: ", ".join(f"{r.pair} {str(r.k)[5:16]}" for r in m.itertuples())
    if "code_kl" in w:
        both = w[w.code.notna() & w.code_kl.notna()]
        mis = both[both.code != both.code_kl]
        L.append(f"_Parity: the kline recompute agrees with the recorded code on {len(both) - len(mis)} of {len(both)} fills"
                 + (f" (differs: {lab(mis)})" if len(mis) else "") + "._")
    if "code_jr" in w:
        sj = w[(w.code_src == "stamp") & w.code.notna() & w.code_jr.notna()]
        mj = sj[sj.code != sj.code_jr]
        L.append(f"_Stamp vs journal: agree on {len(sj) - len(mj)} of {len(sj)} stamped fills with a journal line"
                 + (f" (differs: {lab(mj)})" if len(mj) else "") + "._")
    return L


def gc_counted(df):
    """the bar's cohort: from GC_FROM (the floor is applied BEFORE the episode dedupe — reference rows never hide cohort rows), eligible and
    market volume known to pass; then the first refusal (by signal time, stable tie-break on pair) per pair-episode (episode_keys: spikes ≤ 30 min
    apart merged) SEPARATELY inside V2 and inside V1 — a V1 first
    refusal never hides a later V2 refusal of the same spike. → bool Series."""
    s = lambda c: df[c].astype(str).isin(_TRUE)
    m = s("cohort") & s("eligible") & (df.gvol.astype(str) == "pass") & df.spike_at.notna()
    out = pd.Series(False, index=df.index)
    if m.any():
        g = df[m].assign(_v2=s("v2")[m], _ep=episode_keys(df)[m]).sort_values(["k", "pair"], kind="stable")
        out.loc[g.drop_duplicates(["_v2", "_ep"]).index] = True
    return out


def _bar_stats(w):
    x = pd.to_numeric(w.LOCK, errors="coerce")
    w = w[x.notna()]; x = x.dropna()
    n, nd = len(x), w.day.nunique()
    st = dict(n=n, nd=nd, x=x, w=w)
    if n:
        st["wr"] = (x > 0).mean() * 100; st["mean"] = x.mean()
        st["ci"] = day_ci(x.values, w.day.values)
        st["wo5"] = x.sort_values(ascending=False).iloc[5:].mean() if n > 5 else float("nan")
        st["pshare"] = w.assign(v=x).groupby("pair").v.sum().max() / x.sum() * 100 if x.sum() > 0 else float("inf")
    return st


def _bar_txt(st, n_need, d_need=None):
    t = f"{st['n']}/{n_need} signals" + (f" · {st['nd']}/{d_need} days" if d_need else f" · {st['nd']} days")
    if st["n"]:
        ci = st["ci"]
        t += (f" · WR {st['wr']:.0f} % · mean {st['mean']:+.2f} %" + (f" · day CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
              + f" · top pair {st['pshare']:.0f} % of the net")
    return t


def gc_check(w):
    """FORMAL bar 1 (V2 → FRENZY_LONG) on counted, final V2 signals priced as a LONG (column LOCK). → (state, text). Pre-registered legs:
    N ≥ 30 on ≥ 15 days ∧ mean ≥ +0.30 ∧ day-block CI low > 0 ∧ WR ≥ 55 % ∧ no pair > 25 % of the net ∧ mean × (1 − 0.50) ≥ +0.15.
    ADDITIONS beyond the pre-registration (labelled in the output): mean > 0 without the top 5; retire if mean ≤ 0 at ≥ 30 or no verdict by 60."""
    st = _bar_stats(w); n = st["n"]
    txt = _bar_txt(st, V2_N, V2_DAYS)
    if n < V2_N:
        return "collecting", txt
    txt += f" · after 50 % haircut {st['mean'] * (1 - V2_HAIRCUT):+.2f} · w/o top 5 {st['wo5']:+.2f} (addition)"
    if st["mean"] <= 0:
        return "retire", txt + " · mean ≤ 0 at ≥ 30 (addition)"
    ci = st["ci"]
    if (st["nd"] >= V2_DAYS and st["mean"] >= V2_MEAN and ci and ci[0] > 0 and st["wr"] >= V2_WR and st["pshare"] <= V2_PAIR
            and st["mean"] * (1 - V2_HAIRCUT) >= V2_HAIRCUT_MIN and st["wo5"] > 0):
        return "review", txt
    if n >= GC_RETIRE_N:
        return "retire", txt + " · no verdict by 60 (addition)"
    return "collecting", txt


def w_check(w):
    """FORMAL bar 2 (Pattern-W: V2 ∧ ATR ≤ 1.5 → FRENZY_LONG) on counted, final V2 signals with the replay ATR ≤ 1.5: N ≥ 30 ∧ WR ≥ 70 % ∧
    mean ≥ +0.50 ∧ day-block CI low > 0 ∧ no pair > 25 % → propose; revert if the first 15 promoted fills average < 0. No additions."""
    st = _bar_stats(w)
    txt = _bar_txt(st, W_N)
    if st["n"] < W_N:
        return "collecting", txt
    ci = st["ci"]
    ok = st["wr"] >= W_WR and st["mean"] >= W_MEAN and ci and ci[0] > 0 and st["pshare"] <= W_PAIR
    return ("review" if ok else "collecting"), txt


def _iso(ms):
    return pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%dT%H:%M:%S")


GVOL_PASS_GATES = ("FRENZY_WIDE_CHOPPY", "FRENZY_WIDE_LATE", "FRENZY_WIDE_DISLOC", "FRENZY_WIDE_OPEN_REFUSED")   # WIDE refusals judged AFTER the gvol gate


def gvol_state(js):
    """LONG's market-volume gate on a green refusal, read from WIDE's own journal lines on that bar (same gate, same bar): GVOL_HIGH / UNREAD →
    'high' / 'unread' (LONG would be blocked too); a WIDE open or a WIDE refusal judged after the gate → 'pass'; else 'unknown'."""
    gates = set(js.gate)
    if "FRENZY_WIDE_GVOL_HIGH" in gates:
        return "high"
    if "FRENZY_WIDE_GVOL_UNREAD" in gates:
        return "unread"
    if bool(((js.e == "OPEN") & (js.strategy == "FRENZY_WIDE")).any()) or gates & set(GVOL_PASS_GATES):
        return "pass"
    return "unknown"


def _lock_shadow(sym, sig, now_ms, budget):
    """the FORMAL shadow pricing of a hypothetical FRENZY entry on a signal bar closing at sig ms: the first print ≥ the close + 12 s, the live
    lock exit, 0.09 fees + 0.10 slippage, 12 h cap — on ticks once the archive is out, else the open of the signal-close minute on 1m bars
    (PROVISIONAL; final on 1m only when the ticks never come). → (e, t_e, pnl, x_ms, how, px_src, tick_state, final). Raises on no 1m data."""
    t_in0 = sig + ENTRY_LAG_MS
    horizon = t_in0 + CAP_MIN * MIN
    last_day_end = (horizon // 86_400_000 + 1) * 86_400_000
    giveup = now_ms > last_day_end + TICK_GIVEUP_D * 86_400_000
    px = None; st = "pending"; final = False; e = pnl = x_ms = how = None; t_e = None
    if now_ms >= horizon:
        st, tt, pp = _ticks(sym, sig, horizon + 2 * MIN, now_ms, budget)
        if st == "ok":
            e, t_e = gc_entry(tt, pp, sig)
            if e is not None:
                pnl, x_ms, how = walk_ticks(tt, pp, e, t_e, slip=SLIP); px = "tick"; final = True
            else:
                st = "empty"
    if px is None:   # 1m fallback: the open of the signal-close minute (the 12 s print is inside it), same slippage — PROVISIONAL
        m1 = [b for b in _kl(sym, "1m", sig, min(now_ms, horizon + MIN)) if b[0] + MIN <= now_ms]
        if not m1 or m1[0][0] != sig:
            raise ValueError("1m klines unavailable")
        e = float(m1[0][1]); t_e = sig; pnl, x_ms, how = walk(m1, e, "LOCK2"); px = "1m"
        pnl = pnl - SLIP if pnl is not None else None
        final = bool(now_ms >= horizon and (st == "missing" or (st == "empty" and giveup)))   # ticks never coming → finalise on 1m
    return e, t_e, pnl, x_ms, how, px, st, final


def _gc_price(sig, sym, th, now_ms, budget, J, F):
    """one FRENZY_GREEN_BAR refusal (signal bar closing at sig ms) → the engine replay + the hypothetical FRENZY_LONG priced on the live lock
    with the FORMAL shadow pricing (first print ≥ close + 12 s, 0.10 slippage). Raises on data trouble (the caller keeps the old row)."""
    k, k2 = _iso(sig), _iso(sig + BAR)
    closed = [b for b in _kl(sym, "5m", sig - 1499 * BAR, sig - 1) if b[0] + BAR <= sig]
    if len(closed) < 300 or closed[-1][0] != sig - BAR:
        raise ValueError("5m window missing")
    nh = normal_hour_usd(_kl(sym, "1h", sig - 744 * H, sig), closed[-1][0])
    ep = frenzy_walk(closed, nh, th) if nh else None
    atr = wilder_atr_pct(closed[-300:])
    code = (frenzy_long_status(ep, atr, 1e30, th)[1] if frenzy_flagged(ep, th) else "NOT_FLAGGED") if ep else "NO_EPISODE"   # 24 h volume: the journal line proves it passed
    di, ad = frenzy_di_spread(closed[-300:]), frenzy_adx_delta(closed[-300:])
    strong = bool(di is not None and ad is not None and ad > 0 and di > 0)
    js = J[(J.pair == sym) & (J.t >= k) & (J.t < k2)] if J is not None and len(J) else pd.DataFrame(columns=["t", "e", "pair", "gate", "strategy"])
    gvol = gvol_state(js)
    lg = F[F.opened_at.notna() & (F.entry_strategy.astype(str) == "FRENZY_LONG")] if len(F) else F
    ca = lg.closed_at.astype(str).str[:19] if len(lg) else pd.Series(dtype=str)
    long_open = int(((lg.k < k) & (lg.closed_at.isna() | (ca > k))).sum()) if len(lg) else 0
    pday = int(((lg.pair == sym) & (lg.k.str[:10] == k[:10]) & (lg.k < k)).sum()) if len(lg) else 0
    wf = F[(F.entry_strategy.astype(str) == "FRENZY_WIDE") & (F.pair == sym) & (F.k >= k) & (F.k < k2)] if len(F) else F
    slots = max(1, int(float(getattr(th, "frenzy_max_slots", 2) or 2)))
    dcap = max(0, int(float(getattr(th, "frenzy_max_entries_per_pair_day", 3) or 0)))
    parity = code == "FRENZY_GREEN_BAR"
    eligible = bool(parity and gvol not in ("high", "unread") and long_open < slots and (dcap == 0 or pday < dcap))
    e, t_e, pnl, x_ms, how, px, st, final = _lock_shadow(sym, sig, now_ms, budget)
    return dict(k=k, pair=sym, day=k[:10], cohort=k >= GC_FROM, ver=GC_VER,
                spike_at=(_iso(ep["spike_ts"]) if ep else None), hours=(round(ep["hours"], 2) if ep else None),
                above_streak=(int(ep["above_streak"]) if ep else None), v2=bool(ep and int(ep["above_streak"]) > GC_STREAK),
                above_share=(round(ep["above_share"], 1) if ep and ep.get("above_share") is not None else None),
                vs_vwap=(round(ep["vs_vwap_pct"], 3) if ep and ep.get("vs_vwap_pct") is not None else None),
                vol_mult=(round(ep["vol_mult"], 1) if ep and ep.get("vol_mult") is not None else None),
                bar_ret=(round(ep["bar_ret_pct"], 4) if ep and ep.get("bar_ret_pct") is not None else None), atr=atr,
                replay_code=code, parity=parity, strong=strong, long_lev=float(getattr(th, "frenzy_long_lev_mult", 0.32)),   # FORMAL: promote at 0.32, no strong multiplier
                gvol=gvol, long_open=long_open, pair_day_n=pday, eligible=eligible, entry=e, entry_at=(_iso(t_e) if t_e else None),
                px_src=px, tick_state=st, LOCK=pnl, exit_how=how, exit_at=(_iso(x_ms) if x_ms else None),
                wide_k=(wf.k.iloc[0] if len(wf) else None),
                wide_actual=(float(wf.pnl_percentage.iloc[0]) if len(wf) and pd.notna(wf.pnl_percentage.iloc[0]) and str(wf.status.iloc[0]).upper() == "CLOSED" else None),
                final=final)


def _gc_load():
    """the stored rows; an unreadable file is renamed to .bad and the tracker starts empty (never silently overwritten)."""
    if not os.path.exists(GC_CSV):
        return pd.DataFrame()
    try:
        d = pd.read_csv(GC_CSV)
        if len(d) and not {"k", "pair", "final", "ver"} <= set(d.columns):
            raise ValueError("columns missing")
        return d
    except Exception:
        try:
            os.replace(GC_CSV, _bad_path(GC_CSV))
        except OSError:
            pass
        return pd.DataFrame()


def gc_run(now_ms, th, F, J, allr):
    """price new / provisional green refusals, store, and render the FRENZY_GREEN_CLOCK section."""
    old = _gc_load()
    done = set()
    if len(old):
        done = {(a, b) for a, b, f_, v in zip(old.k, old.pair, old.final, old.ver) if str(f_) in _TRUE and str(v) in (str(GC_VER), f"{GC_VER}.0")}
    G = J[(J.e == "BLOCK") & (J.gate == "FRENZY_GREEN_BAR")].drop_duplicates(["t", "pair"]) if J is not None and len(J) else pd.DataFrame(columns=["t", "pair"])
    G = G.assign(c=G.t >= GC_FROM).sort_values(["c", "t"], ascending=False)   # the cohort first, newest first (tick downloads are rationed)
    budget = {"dl": TICK_DL_MAX, "deadline": time.monotonic() + GC_TIME_BUDGET_S}; rows = []; err = 0; late = 0
    for r in G.itertuples():
        if (r.t, r.pair) in done:
            continue
        if time.monotonic() > budget["deadline"]:
            late += 1
            continue
        try:
            rows.append(_gc_price(int(pd.Timestamp(r.t, tz="UTC").value // 1_000_000), r.pair, th, now_ms, budget, J, F))
        except Exception:
            err += 1
    new = pd.DataFrame(rows)
    if len(new) and len(old):
        prev = old.drop_duplicates(["k", "pair"], keep="last").set_index(["k", "pair"])
        idx = pd.MultiIndex.from_arrays([new.k, new.pair])
        for c in ("long_open", "pair_day_n"):          # the first reading (the exports were freshest then) wins
            if c in prev:
                o_ = prev[c].reindex(idx).values
                new[c] = np.where(pd.notna(o_), o_, new[c])
        for c in ("wide_k", "wide_actual"):            # a newer reading wins; an export that rolled off never erases one
            if c in prev:
                o_ = prev[c].reindex(idx).values
                new[c] = np.where(pd.notna(new[c].values), new[c].values, o_)
        slots = max(1, int(float(getattr(th, "frenzy_max_slots", 2) or 2)))
        dcap = max(0, int(float(getattr(th, "frenzy_max_entries_per_pair_day", 3) or 0)))
        new["eligible"] = (new.parity.astype(str).isin(_TRUE) & ~new.gvol.isin(["high", "unread"])
                           & (pd.to_numeric(new.long_open) < slots) & ((dcap == 0) | (pd.to_numeric(new.pair_day_n) < dcap)))
    allg = pd.concat([old, new], ignore_index=True) if len(new) else old
    if len(allg):
        allg = allg.drop_duplicates(["k", "pair"], keep="last").sort_values("k").reset_index(drop=True)
        if len(allr) and "wide_k" in allg:
            lk = allr.drop_duplicates(["k", "pair"], keep="last").set_index(["k", "pair"]).LOCK2
            allg["wide_lock2"] = lk.reindex(pd.MultiIndex.from_arrays([allg.wide_k.astype(str), allg.pair])).values
        allg["counted"] = gc_counted(allg)
        tmp = f"{GC_CSV}.{os.getpid()}.tmp"; allg.to_csv(tmp, index=False); os.replace(tmp, GC_CSV)
    L = ["## 🟢 FRENZY_GREEN_CLOCK — green-candle refusals priced as FRENZY_LONG (V2 observe, pre-registered 2026-10-06; never armed from here)", "",
         f"Every journal FRENZY_GREEN_BAR refusal, replayed with the engine's functions (parity = the replay also refuses it for the green candle). "
         f"V2 = above_streak > {GC_STREAK} at the signal (the hour above average was already met → the setup turned ON by clock / volume; RLC 10-05). "
         "Hypothetical LONG priced as pre-registered (reports/FRENZY_GREEN_AND_WIDE_ATR_FORMAL_2026-10-06.md): first trade print ≥ the 5m close + 12 s, "
         "the live lock exit, 0.09 % fees + 0.10 % slippage, 12 h cap — on ticks once the day's archive is out, else 1m bars (ᵖ provisional). "
         "⚠ Tick pricing can stop on wicks the live poller rides through (CLAUDE.md: exit counterfactuals read on the live-stopped cohort) — a "
         "tick stop here is not proof live would have stopped. ¹ = counted (from the floor, eligible, market volume known to pass, the first refusal "
         "of its spike within V2 and within V1 separately). Not eligible = LONG would still be refused (WIDE's market-volume block on that bar, "
         f"LONG slots full, pair-day cap) or the replay disagrees. Counted from {GC_FROM[:10]}; earlier rows are reference only.", ""]
    if not len(allg):
        return L + ["No green-candle refusal in the journals yet.", ""] + ([f"_{err} refusal(s) not priced this run — retried next run._"] if err else [])
    f = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.2f}"
    L += ["| Signal UTC | Pair | Streak | V2 | h after spike | ATR | Candle | Gvol | LONG slots | Eligible | LONG (lock) | Exit | Px | WIDE same bar (actual / lock) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    T = lambda v: str(v) in _TRUE
    for r in allg.tail(15).itertuples():
        mk = ("" if T(r.final) else "ᵖ") + ("¹" if T(r.counted) else "") + ("" if T(r.cohort) else " (ref)")
        wide = f"{f(r.wide_actual)} / {f(getattr(r, 'wide_lock2', None))}" if isinstance(r.wide_k, str) else "none"
        L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{mk} | {str(r.pair).replace('USDT', '')} | {'' if pd.isna(r.above_streak) else int(r.above_streak)} | "
                 f"{'✔' if T(r.v2) else '–'} | {'' if pd.isna(r.hours) else f'{r.hours:.1f}'} | "
                 f"{'' if pd.isna(r.atr) else f'{r.atr:.2f}'} | {f(r.bar_ret)} | {r.gvol} | {'' if pd.isna(r.long_open) else int(r.long_open)} | "
                 f"{'✔' if T(r.eligible) else '✗ ' + ('replay ' + str(r.replay_code) if not T(r.parity) else 'gate')} | "
                 f"{f(r.LOCK)} | {r.exit_how} | {r.px_src} | {wide} |")
    S_ = lambda c: allg[c].astype(str).isin(_TRUE)
    cnt = allg[S_("counted") & S_("final")]
    v2 = cnt[cnt.v2.astype(str).isin(_TRUE)]; v1 = cnt[~cnt.v2.astype(str).isin(_TRUE)]
    lab = {"review": "📋 PROPOSE PROMOTION", "retire": "❌ retire", "collecting": "⏳ collecting"}
    st, tx = gc_check(v2)
    L += ["", f"**V2 bar (FORMAL bar 1, frozen: N ≥ {V2_N} on ≥ {V2_DAYS} days ∧ mean ≥ +{V2_MEAN:.2f} ∧ day CI low > 0 ∧ WR ≥ {V2_WR:g} % ∧ no pair > "
              f"{V2_PAIR:g} % ∧ mean after a 50 % haircut ≥ +{V2_HAIRCUT_MIN:.2f} → promote at lev 0.32, no strong multiplier, revert if the first "
              f"{V2_REVERT_N} promoted fills average < 0 · ADDITIONS beyond the pre-registration: > 0 without the top 5; retire if mean ≤ 0 at ≥ 30 or "
              f"no verdict by {GC_RETIRE_N}):** " + lab.get(st, st) + f" ({tx})"]
    wv = v2[pd.to_numeric(v2.atr, errors="coerce") <= W_ATR]
    st, tx = w_check(wv)
    L.append(f"**V2 ∧ ATR ≤ {W_ATR:g} (FORMAL bar 2, Pattern-W, frozen: N ≥ {W_N} ∧ WR ≥ {W_WR:g} % ∧ mean ≥ +{W_MEAN:.2f} ∧ day CI low > 0 ∧ no pair > "
             f"{W_PAIR:g} % → propose; revert if the first {W_REVERT_N} average < 0; ~1 year at 0.09 signals / day):** "
             + {"review": "📋 PROPOSE PROMOTION", "collecting": "⏳ collecting"}.get(st, st) + f" ({tx})")
    x1 = pd.to_numeric(v1.LOCK, errors="coerce").dropna()
    ci1 = day_ci(x1.values, v1.loc[x1.index].day.values) if len(x1) else None
    L.append(f"V1 remainder (green refusals with streak ≤ {GC_STREAK}, same pricing, contrast only): {len(x1)} signals / {v1.day.nunique()} d"
             + (f" · WR {(x1 > 0).mean() * 100:.0f} % · mean {x1.mean():+.2f} %" if len(x1) else "")
             + (f" · day CI [{ci1[0]:+.2f}, {ci1[1]:+.2f}]" if ci1 else "") + ".")
    coh = allg[S_("cohort")]
    unk = coh[S_("eligible")[coh.index] & (coh.gvol.astype(str) == "unknown")]
    xu = pd.to_numeric(unk[unk.final.astype(str).isin(_TRUE)].LOCK, errors="coerce").dropna()
    L.append(f"Market volume unknown (WIDE never reached its gvol check on that bar — NOT counted): {len(unk)} signals"
             + (f" ({int(unk.v2.astype(str).isin(_TRUE).sum())} V2) · final {len(xu)} · mean {xu.mean():+.2f} %" if len(xu) else "") + ".")
    ov = cnt[cnt.wide_k.notna()] if "wide_k" in cnt else cnt.iloc[0:0]
    if len(ov):
        L.append(f"Overlap: {len(ov)} of {len(cnt)} counted signals also had a WIDE fill on the same bar (WIDE actual mean "
                 f"{pd.to_numeric(ov.wide_actual, errors='coerce').mean():+.2f} % at lev {getattr(th, 'frenzy_wide_lev_mult', 0.2)} vs the hypothetical "
                 f"LONG {pd.to_numeric(ov.LOCK, errors='coerce').mean():+.2f} % at lev 0.32).")
    cu = allg[allg.replay_code.astype(str) == "FRENZY_ON"] if "replay_code" in allg else allg.iloc[0:0]   # the replay at t says ON in state, not fresh
    L.append(f"Catch-up / not replayable at t (the journal line's t is not the ON bar — e.g. a pause catch-up; never counted): {len(cu)}"
             + (" (" + ", ".join(f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16]}" for r in cu.itertuples()) + ")" if len(cu) else "") + ".")
    L.append(f"_Cohort rows {len(coh)}: counted {int(S_('counted')[coh.index].sum())} · not eligible "
             f"{int((~S_('eligible')[coh.index] & ~coh.index.isin(cu.index)).sum())} · catch-up {int(coh.index.isin(cu.index).sum())} · "
             f"provisional {int((~S_('final')[coh.index]).sum())}._")
    if err:
        L.append(f"_{err} refusal(s) not priced this run (klines unavailable) — retried next run._")
    if late:
        L.append(f"_{late} refusal(s) left for the next run (the {GC_TIME_BUDGET_S} s time budget was spent)._")
    return L + [""]


def _gc_safe(now_ms, th, F, J, allr):
    try:
        return gc_run(now_ms, th, F, J if J is not None else pd.DataFrame(columns=["t", "e", "pair", "gate", "strategy"]), allr)
    except Exception as ex:
        return ["## 🟢 FRENZY_GREEN_CLOCK", "", f"Unavailable this run ({str(ex)[:120]}).", ""]


# ─────────────────────────── 🌊 tracker 8 (GVOL_BLOCKED) + 🪜 tracker 9 (VWAP_STOP) — 2026-10-06, observe-only ───────────────────────────
GVB_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_GVOL_BLOCKED.csv")
VWS_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_VWAP_STOP.csv")
GVB_VER = VWS_VER = 1
# 🌊 GVOL_BLOCKED (DECISION_LOG 194 gate, frenzy_gvol_max = 1.0): the gate's own revert gate reads only the fills it let through; this prices
# the side it blocks AND the side it lets through with the same ruler. Gate codes = the engine's _record_filter_block names in _frenzy_open.
GVB_HIGH = {"FRENZY_GVOL_HIGH": "LONG", "FRENZY_WIDE_GVOL_HIGH": "WIDE"}
GVB_UNREAD = {"FRENZY_GVOL_UNREAD": "LONG", "FRENZY_WIDE_GVOL_UNREAD": "WIDE"}
GVB_PASSED = "PASSED"                          # rows of the live fills the gate let through (same CSV, same _lock_shadow pricing)
GVB_N, GVB_DAYS, GVB_CONFIRM, GVB_DAY_MAX = 20, 10, -0.20, 50.0   # FROZEN bar (registered 2026-10-06): ≥ 20 counted signals on ≥ 10 days ∧ no
#   single day ≥ 50 % of the blocked net (window units: market volume is market-wide) → blocked mean ≥ 0 ∧ ≥ the let-through mean (same ruler)
#   → "review: the gate removes winners" (flag only) · blocked mean ≤ −0.20 → "gate confirmed" · else inconclusive
GVB_LONG_LEV, GVB_LONG_LEV_STRONG, GVB_WIDE_LEV = 0.32, 0.5, 0.2   # sizing at registration (trading_config 2026-10-06; 197 strong = ADX Δ > 0 ∧ DI > 0)
HG_STREAK = 12.0                                # today's WIDE hold-green rule (DECISION_LOG 231, frenzy_wide_hold_green_streak 12): frozen here
EPISODE_MERGE_MS = 30 * MIN                     # spikes of one pair ≤ 30 min apart = one episode (replay drift: AIN 14:25 vs 14:30)
GVB_YEAR = ("year (DECISION_LOG 194, 1,455 FRENZY first candles, live exit, real costs): market volume ≥ 1.0 → 605 · −0.185 %/trade vs < 1.0 → "
            "850 · +0.225 (FRENZY −0.411 / +0.369 · WIDE −0.121 / +0.182); the blocked side failed only on confidence (day CI [−0.49, +0.15])")
# 🪜 VWAP_STOP — the study's "BP k 0.5" (reports/FRENZY_STAIRCASE_STUDY_2026-10-06.md §3b / §5, pre-registered there; scratch px.py): after the
# live −3 stop the position is held and exits at the first print after a 5m close that is BOTH ≤ −3 % net AND below VWAP × (1 − 0.5 × ATR % / 100)
# (ATR = the fill's stamped entry_atr_pct; unreadable → the study's 2.0), hard floor −12 % net on prints; the lock (+3 → max(+2, peak − 2))
# unchanged and the close rule off once armed; 12 h cap from the entry; 0.09 fees + 0.10 slippage on the shadow exit.
VWS_K, VWS_LINE, VWS_FLOOR, VWS_ATR_DEFAULT = 0.5, -3.0, -12.0, 2.0
VWS_N, VWS_TOP, VWS_SLEEVE_MIN = 20, 50.0, 5   # FROZEN gate on the first 20 stopped fills (by open time) from GC_FROM, all final: Δ sum > 0 ∧ saved >
#   deeper ∧ Δ sum > 0 on EVERY sleeve with ≥ 5 of those 20 fills (at least one sleeve must have ≥ 5, else collecting) ∧ no single fill > 50 % of
#   the gain → "candidate for a pre-registered study" (never an arm); otherwise "close the idea"
VWS_YEAR = ("year (study §3b, BP k 0.5, the 179 live-stopped of 403 today's-rules fills, ticks): saved 72 (Δ +400) vs deeper 103 (Δ −395) · "
            "Δ mean on stopped +0.03 · day CI [−0.79, +0.83] · halves +0.38 / −0.33 · FRENZY +0.26 / WIDE −0.41 — NOT established")
LOCK_FROM = "2026-10-05T16:00:00"               # the lock exit went live ~10-05 16:00 UTC (205): earlier fills ran another exit regime
STALE_D = 7                                     # a row whose ticks are still not in hand this many days after its horizon finalises on 1m (age)
X_DL, X_TIME_S = 2, 90                          # per-tracker tick downloads / seconds per run (the GREEN_CLOCK budget is separate)
# WORST-CASE WALL TIME per scout run (trackers 7 – 9, after the exit table): each tracker checks its deadline before every item, a tick download
# is cut at the deadline, and an item already in flight can still spend its kline calls (≤ 4 × the 20 s urlopen timeout) → GREEN_CLOCK ≤ 180 + 80 s,
# GVOL_BLOCKED ≤ 90 + 80 s, VWAP_STOP ≤ 90 + 80 s ≈ 10 min absolute worst; the 10-06 dry-run took ~77 s for the whole module with 8 downloads.


def _bad_path(path):
    """a timestamped, never-overwritten .bad name next to path."""
    base = f"{path}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.{os.getpid()}"
    p, i = base + ".bad", 1
    while os.path.exists(p):
        p, i = f"{base}.{i}.bad", i + 1
    return p


def _load_csv(path, need):
    """stored rows; an unreadable file is moved to a timestamped .bad and the tracker starts empty (never silently overwritten)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        d = pd.read_csv(path)
        if len(d) and not set(need) <= set(d.columns):
            raise ValueError("columns missing")
        return d
    except Exception:
        try:
            os.replace(path, _bad_path(path))
        except OSError:
            pass
        return pd.DataFrame()


def _save_csv(df, path):
    tmp = f"{path}.{os.getpid()}.tmp"; df.to_csv(tmp, index=False); os.replace(tmp, path)


def _ms(s):
    return int(pd.Timestamp(str(s)[:26], tz="UTC").value // 1_000_000)


def _num(v):
    try:
        v = float(v)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def episode_keys(df, col="spike_at"):
    """one key per (pair, spike episode): spikes of the same pair ≤ EPISODE_MERGE_MS apart (chained, by time) share the key of the first.
    An unparseable spike value keys on itself; a missing one → None."""
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    if not len(df) or col not in df:
        return out
    sp = df[col]
    ts = pd.Series([pd.to_datetime(v, errors="coerce") if isinstance(v, str) else pd.NaT for v in sp], index=df.index)
    has = sp.notna()
    for pair in pd.unique(df.pair[has]):
        idx = list(df.index[(df.pair == pair) & has])
        for i in idx:
            if pd.isna(ts[i]):
                out[i] = f"{pair}|{sp[i]}"
        anchor = prev = None
        for i in sorted([i for i in idx if pd.notna(ts[i])], key=lambda i: (ts[i], str(i))):
            if prev is None or (ts[i] - prev) > pd.Timedelta(milliseconds=EPISODE_MERGE_MS):
                anchor = ts[i]
            out[i] = f"{pair}|{anchor:%Y-%m-%dT%H:%M:%S}"; prev = ts[i]
    return out


def gvb_take(sleeve, ep, code, atr, th):
    """would TODAY's sleeve have opened this activation had the market-volume gate passed? → (take, why). LONG: the replay says FRENZY_READY.
    WIDE: the replay says ATR_HIGH / GREEN_BAR on a fresh bar with a readable ATR, then (in the engine's order after the gvol gate) the
    choppy check (live config, observe-only 0 = off) and the hold-green rule frozen at streak > 12 (FRENZY_WIDE_ATR_HIGH / _RECLAIM block).
    The LATE / DISLOC checks after them cannot be replayed (stated on the table)."""
    if sleeve == "LONG":
        return (code == "FRENZY_READY"), ("" if code == "FRENZY_READY" else f"replay {code}")
    if not (ep and ep.get("fresh_on") and code in FRENZY_WIDE_CODES and atr is not None):
        return False, f"replay {code}"
    if frenzy_wide_choppy(ep, th):
        return False, "FRENZY_WIDE_CHOPPY"
    hg = frenzy_wide_hold_green_block(ep, code, SimpleNamespace(**{**vars(th), "frenzy_wide_hold_green_streak": HG_STREAK}))
    return (hg is None), (hg or "")


def gvb_masks(df):
    """→ dict of bool Series over the stored rows: counted (blocked, from GC_FROM — floor BEFORE the dedupe —, *_GVOL_HIGH, today's sleeve would
    take it, not a catch-up, its episode has NO live fill, first of its pair-episode), double (blocked would-take rows whose episode also has a
    live FRENZY / WIDE fill — excluded, listed), passed (let-through fills from GC_FROM, first per pair-episode), plus the episode keys."""
    T = lambda c: df[c].astype(str).isin(_TRUE) if c in df else pd.Series(False, index=df.index)
    ep = episode_keys(df)
    passed = df.gate.astype(str) == GVB_PASSED
    fills = set(ep[passed & ep.notna()])
    blk = df.gate.isin(list(GVB_HIGH)) & T("would_take") & ~T("catchup") & ep.notna()
    double = blk & ep.isin(fills)
    first = lambda m: df[m].assign(_ep=ep[m]).sort_values(["k", "pair", "gate"], kind="stable").drop_duplicates("_ep").index
    cnt = pd.Series(False, index=df.index); pc = pd.Series(False, index=df.index)
    m = blk & ~double & T("cohort")
    if m.any():
        cnt.loc[first(m)] = True
    m = passed & T("cohort") & ep.notna()
    if m.any():
        pc.loc[first(m)] = True
    return dict(episode=ep, counted=cnt, double=double, passed=pc)


def gvb_counted(df):
    return gvb_masks(df)["counted"]


def gvb_check(x, days, passed_mean):
    """the FROZEN GVOL_BLOCKED bar on the counted, final blocked signals' lock % (x) and their UTC days. → (state, text)."""
    x = pd.Series(np.asarray(x, dtype=float)); days = pd.Series(list(days), index=x.index)
    ok_ = x.notna(); x, days = x[ok_], days[ok_]
    n, nd = len(x), days.nunique()
    pm = None if passed_mean is None or np.isnan(passed_mean) else float(passed_mean)
    t = f"{n}/{GVB_N} signals · {nd}/{GVB_DAYS} days" + (f" · WR {(x > 0).mean() * 100:.0f} % · mean {x.mean():+.2f} %" if n else "")
    t += f" · let-through mean {pm:+.2f} % (same ruler)" if pm is not None else " · let-through mean –"
    if n < GVB_N or nd < GVB_DAYS:
        return "collecting", t
    ci = day_ci(x.values, days.values)
    ds = x.groupby(days.values).sum(); net = x.sum()
    share = ds.max() / net * 100 if net > 0 else ds.min() / net * 100 if net < 0 else float("inf")
    t += (f" · day CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "") + f" · top day {share:.0f} % of the net"
    if share >= GVB_DAY_MAX:
        return "inconclusive", t + f" · one day carries ≥ {GVB_DAY_MAX:.0f} % (window leg fails)"
    if x.mean() <= GVB_CONFIRM:
        return "confirmed", t
    if x.mean() >= 0 and pm is not None and x.mean() >= pm:
        return "review", t
    return "inconclusive", t


def _gvb_price(sig, sym, gate, th, now_ms, budget):
    """one journal *_GVOL_* BLOCK line (signal bar closing at sig ms) → the engine replay (would today's sleeve take it? a FRENZY_ON replay in
    state = a catch-up line whose t is not the ON bar) + the activation priced as that sleeve with _lock_shadow. Raises on data trouble."""
    sleeve = {**GVB_HIGH, **GVB_UNREAD}[gate]
    k = _iso(sig)
    closed = [b for b in _kl(sym, "5m", sig - 1499 * BAR, sig - 1) if b[0] + BAR <= sig]
    if len(closed) < 300 or closed[-1][0] != sig - BAR:
        raise ValueError("5m window missing")
    nh = normal_hour_usd(_kl(sym, "1h", sig - 744 * H, sig), closed[-1][0])
    ep = frenzy_walk(closed, nh, th) if nh else None
    atr = wilder_atr_pct(closed[-300:])
    code = (frenzy_long_status(ep, atr, 1e30, th)[1] if frenzy_flagged(ep, th) else "NOT_FLAGGED") if ep else "NO_EPISODE"   # the journal line proves the 24 h volume passed
    catchup = bool(ep and ep.get("in_state") and code == "FRENZY_ON")
    di, ad = frenzy_di_spread(closed[-300:]), frenzy_adx_delta(closed[-300:])
    strong = bool(sleeve == "LONG" and di is not None and ad is not None and ad > 0 and di > 0)
    take, why = (False, "catch-up / not replayable at t") if catchup else gvb_take(sleeve, ep, code, atr, th)
    e, t_e, pnl, x_ms, how, px, st, final = _lock_shadow(sym, sig, now_ms, budget)
    return dict(k=k, pair=sym, day=k[:10], sleeve=sleeve, gate=gate, cohort=k >= GC_FROM, ver=GVB_VER,
                spike_at=(_iso(ep["spike_ts"]) if ep else None), hours=(round(ep["hours"], 2) if ep else None),
                above_streak=(int(ep["above_streak"]) if ep and ep.get("above_streak") is not None else None),
                vs_vwap=(round(ep["vs_vwap_pct"], 3) if ep and ep.get("vs_vwap_pct") is not None else None),
                bar_ret=(round(ep["bar_ret_pct"], 4) if ep and ep.get("bar_ret_pct") is not None else None), atr=atr,
                replay_code=code, catchup=catchup, would_take=bool(take), why=why, strong=strong,
                lev=(GVB_WIDE_LEV if sleeve == "WIDE" else GVB_LONG_LEV_STRONG if strong else GVB_LONG_LEV),
                fill_k=None, actual=None, entry=e, entry_at=(_iso(t_e) if t_e else None), px_src=px, tick_state=st, LOCK=pnl, exit_how=how,
                exit_at=(_iso(x_ms) if x_ms else None), final=final)


def _gvb_passed_price(fk, sym, sleeve, spike, actual, now_ms, budget):
    """one live FRENZY / WIDE fill the gate let through (opened at fk) → priced with the SAME ruler as the blocked side: _lock_shadow at its
    signal bar (the 5m close the fill was opened on). The live actual is kept for display only."""
    sig = _ms(fk) // BAR * BAR
    k = _iso(sig)
    e, t_e, pnl, x_ms, how, px, st, final = _lock_shadow(sym, sig, now_ms, budget)
    return dict(k=k, pair=sym, day=k[:10], sleeve=sleeve, gate=GVB_PASSED, cohort=k >= GC_FROM, ver=GVB_VER, spike_at=spike,
                replay_code="LIVE_FILL", catchup=False, would_take=True, why="", strong=None,
                lev=(GVB_WIDE_LEV if sleeve == "WIDE" else GVB_LONG_LEV), fill_k=str(fk)[:19], actual=actual,
                entry=e, entry_at=(_iso(t_e) if t_e else None), px_src=px, tick_state=st, LOCK=pnl, exit_how=how,
                exit_at=(_iso(x_ms) if x_ms else None), final=final)


def gvb_run(now_ms, th, J, allr, F=None):
    """price new / provisional market-volume refusals AND the let-through fills (same ruler), store, render the GVOL_BLOCKED section.
    Non-final stored rows are re-priced from their own stored fields (k / pair / gate, or fill_k for a let-through), so a row never waits on an
    export or a journal that left ~/Downloads."""
    old = _load_csv(GVB_CSV, ("k", "pair", "gate", "final", "ver"))
    done = set()
    if len(old):
        done = {(a, b, g) for a, b, g, f_, v in zip(old.k, old.pair, old.gate, old.final, old.ver)
                if str(f_) in _TRUE and str(v) in (str(GVB_VER), f"{GVB_VER}.0")}
    work = {}                                                  # key → (cohort, t, kind, args)
    B = (J[(J.e == "BLOCK") & J.gate.isin(list(GVB_HIGH) + list(GVB_UNREAD))].drop_duplicates(["t", "pair", "gate"])
         if J is not None and len(J) else pd.DataFrame(columns=["t", "pair", "gate"]))
    for r in B.itertuples():
        work[(r.t, r.pair, r.gate)] = ("B", (r.t, r.pair, r.gate))
    acts = {}
    if F is not None and len(F):
        P = F[F.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (F.k.astype(str) >= GC_SCAN_FROM)]
        for r in P.itertuples():
            try:
                kk = _iso(_ms(r.k) // BAR * BAR)
                act = float(r.pnl_percentage) if str(r.status).upper() == "CLOSED" and pd.notna(r.pnl_percentage) else None
                sp = str(getattr(r, "entry_frenzy_spike_at", ""))[:19] if pd.notna(getattr(r, "entry_frenzy_spike_at", None)) else None
                acts[(kk, r.pair)] = act
                work[(kk, r.pair, GVB_PASSED)] = ("P", (r.k, r.pair, str(r.entry_strategy).replace("FRENZY_", ""), sp, act))
            except Exception:
                continue
    if len(old):                                               # stored non-final rows: re-priced from their own fields
        for r in old.itertuples():
            key = (r.k, r.pair, r.gate)
            if key in done or key in work:
                continue
            if r.gate == GVB_PASSED:
                work[key] = ("P", (r.fill_k, r.pair, r.sleeve, (r.spike_at if isinstance(r.spike_at, str) else None), _num(r.actual)))
            else:
                work[key] = ("B", (r.k, r.pair, r.gate))
    order = sorted(work.items(), key=lambda kv: (kv[0][0] >= GC_FROM, kv[0][0], kv[0][1], kv[0][2]), reverse=True)   # cohort first, newest first
    budget = {"dl": X_DL, "deadline": time.monotonic() + X_TIME_S}; rows = []; err = late = 0
    for key, (kind, a) in order:
        if key in done:
            continue
        if time.monotonic() > budget["deadline"]:
            late += 1
            continue
        try:
            rows.append(_gvb_price(_ms(a[0]), a[1], a[2], th, now_ms, budget) if kind == "B" else _gvb_passed_price(*a, now_ms, budget))
        except Exception:
            err += 1
    new = pd.DataFrame(rows)
    allg = pd.concat([old, new], ignore_index=True) if len(new) else old.copy()
    if len(allg):
        allg = allg.drop_duplicates(["k", "pair", "gate"], keep="last").sort_values(["k", "pair", "gate"], kind="stable").reset_index(drop=True)
        if acts and "actual" in allg:                          # a let-through fill that closed since it was priced: its live actual (display only)
            pa = allg.gate == GVB_PASSED
            na = pd.Series([acts.get((k, p)) for k, p in zip(allg.k, allg.pair)], index=allg.index, dtype=object)
            allg.loc[pa & na.notna(), "actual"] = na[pa & na.notna()].astype(float)
        M = gvb_masks(allg)
        allg["episode"], allg["counted"], allg["double"], allg["passed_counted"] = M["episode"], M["counted"], M["double"], M["passed"]
        _save_csv(allg, GVB_CSV)
    L = ["## 🌊 GVOL_BLOCKED — FRENZY / WIDE activations the market-volume gate refused vs the fills it let through, same ruler (observe-only, registered 2026-10-06)", "",
         "Every journal FRENZY_GVOL_HIGH / FRENZY_WIDE_GVOL_HIGH refusal (DECISION_LOG 194 gate, frenzy_gvol_max 1.0; the gate's own revert gate reads only "
         "the fills it let through). Replayed with the engine's functions: LONG = the replay says READY (lev 0.32, 0.5 strong); WIDE = a fresh ATR_HIGH / "
         f"GREEN_BAR refusal that TODAY's hold-green rule (streak > {HG_STREAK:g}, DECISION_LOG 231) would still take (lev 0.2) — the LATE / DISLOC checks "
         "after the gate are not replayable. BOTH sides priced with the FORMAL shadow pricing at their signal bar (the let-through fills too — their live "
         "actual is display-only): first print ≥ the 5m close + 12 s, the live lock exit, 0.09 % fees + 0.10 % slippage, 12 h cap; ticks once the archive "
         "is out, else 1m bars (ᵖ provisional). ⚠ Tick stops can fire on wicks the live poller rides through. One per pair-episode (spikes ≤ 30 min apart "
         f"merged); a blocked episode that ALSO had a live fill is excluded (listed). ¹ = counted (from {GC_FROM[:10]}). DAY units (market-wide variable: "
         "no single day ≥ 50 % of the net). _UNREAD refusals and catch-up lines on their own lines, never in the bar. Earlier rows are reference only.", ""]
    if not len(allg):
        return L + ["No market-volume refusal or let-through fill yet.", ""] + ([f"_{err} row(s) not priced this run — retried next run._"] if err else [])
    f = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.2f}"
    T = lambda v: str(v) in _TRUE
    S_ = lambda c: allg[c].astype(str).isin(_TRUE)
    blk = allg[allg.gate != GVB_PASSED]
    L += ["| Signal UTC | Pair | Sleeve | Gate | Replay | Streak | ATR | Would trade today | Lev | Lock % | Exit | Px |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in blk.tail(15).itertuples():
        mk = ("" if T(r.final) else "ᵖ") + ("¹" if T(r.counted) else "") + ("" if T(r.cohort) else " (ref)")
        wt = ("✗ live fill same episode" if T(r.double) else "✔") if T(r.would_take) else "✗ " + str(r.why if isinstance(r.why, str) else "")
        L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{mk} | {str(r.pair).replace('USDT', '')} | {r.sleeve} | {str(r.gate).replace('FRENZY_', '')} | "
                 f"{str(r.replay_code).replace('FRENZY_', '')} | {'' if pd.isna(r.above_streak) else int(r.above_streak)} | "
                 f"{'' if pd.isna(r.atr) else f'{r.atr:.2f}'} | {wt} | {r.lev:g} | {f(r.LOCK)} | {r.exit_how} | {r.px_src} |")
    cnt = allg[S_("counted") & S_("final")]
    pcs = allg[S_("passed_counted") & S_("final")]
    xp = pd.to_numeric(pcs.LOCK, errors="coerce").dropna()
    lab = {"review": "📋 REVIEW: the gate removes winners (flag only — no auto-change)", "confirmed": "✅ gate confirmed",
           "inconclusive": "➖ inconclusive", "collecting": "⏳ collecting"}
    st, tx = gvb_check(pd.to_numeric(cnt.LOCK, errors="coerce").values, cnt.day.values, xp.mean() if len(xp) else None)
    L += ["", f"**GVOL_BLOCKED bar (FROZEN: at ≥ {GVB_N} counted signals on ≥ {GVB_DAYS} days ∧ no day ≥ {GVB_DAY_MAX:.0f} % of the blocked net — "
              f"blocked mean ≥ 0 ∧ ≥ the let-through fills' mean on the same ruler (first per pair-episode, since {GC_FROM[:10]}) → review flag; "
              f"blocked mean ≤ {GVB_CONFIRM:+.2f} → gate confirmed):** " + lab.get(st, st) + f" ({tx})"]
    for sl in ("LONG", "WIDE"):
        g = pd.to_numeric(cnt[cnt.sleeve == sl].LOCK, errors="coerce").dropna(); p = pd.to_numeric(pcs[pcs.sleeve == sl].LOCK, errors="coerce").dropna()
        L.append(f"- {sl}: blocked {len(g)}" + (f" · mean {g.mean():+.2f} %" if len(g) else "") + f" · let through {len(p)}" + (f" · mean {p.mean():+.2f} %" if len(p) else ""))
    pa = pd.to_numeric(pcs.actual, errors="coerce").dropna()
    L.append(f"- Let-through live actual (display only, not the bar's ruler): {len(pa)} closed" + (f" · mean {pa.mean():+.2f} %" if len(pa) else "") + ".")
    dbl = blk[S_("double")[blk.index]]
    L.append(f"- Excluded — blocked episode that also had a live fill (no double count): {len(dbl)}"
             + (" (" + ", ".join(f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16]}" for r in dbl.itertuples()) + ")" if len(dbl) else "") + ".")
    cu = blk[S_("catchup")[blk.index]] if "catchup" in blk else blk.iloc[0:0]
    L.append(f"- Catch-up / not replayable at t (the replay at the journal's t says FRENZY_ON in state — the line's t is not the ON bar; never counted): {len(cu)}"
             + (" (" + ", ".join(f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16]}" for r in cu.itertuples()) + ")" if len(cu) else "") + ".")
    ref = blk[~S_("cohort")[blk.index] & S_("would_take")[blk.index] & ~S_("double")[blk.index] & blk.gate.isin(list(GVB_HIGH))]
    ref = ref.assign(_ep=allg.episode[ref.index]).dropna(subset=["_ep"]).sort_values(["k", "pair"], kind="stable").drop_duplicates("_ep")
    xr = pd.to_numeric(ref.LOCK, errors="coerce").dropna()
    pr = allg[(allg.gate == GVB_PASSED) & ~S_("cohort")].assign(_ep=allg.episode).dropna(subset=["_ep"]).sort_values(["k", "pair"], kind="stable").drop_duplicates("_ep")
    xpr, apr = pd.to_numeric(pr.LOCK, errors="coerce").dropna(), pd.to_numeric(pr.actual, errors="coerce").dropna()
    L.append(f"- Reference (before {GC_FROM[:10]}, not counted): blocked {len(xr)} pair-episodes would trade today" + (f" · mean {xr.mean():+.2f} %" if len(xr) else "")
             + (f" · {int((~S_('final')[ref.index]).sum())} provisional" if len(ref) else "")
             + f" · let through {len(xpr)} pair-episodes" + (f" · same ruler {xpr.mean():+.2f} %" if len(xpr) else "")
             + (f" · live actual {apr.mean():+.2f} %" if len(apr) else "") + ".")
    un = blk[blk.gate.isin(list(GVB_UNREAD))]
    xu = pd.to_numeric(un[S_("final")[un.index]].LOCK, errors="coerce").dropna() if len(un) else pd.Series(dtype=float)
    L.append(f"- _GVOL_UNREAD (market volume unreadable — fail-closed; never in the bar): {len(un)} refusals"
             + (f" ({int(S_('cohort')[un.index].sum())} from {GC_FROM[:10]}) · final {len(xu)} · mean {xu.mean():+.2f} %" if len(un) else "") + ".")
    L.append(f"- Year reference: {GVB_YEAR}.")
    coh = blk[S_("cohort")[blk.index] & blk.gate.isin(list(GVB_HIGH))]
    L.append(f"_Cohort rows {len(coh)}: counted {int(S_('counted')[coh.index].sum())} · today's sleeve would not trade {int((~S_('would_take')[coh.index]).sum())} · "
             f"provisional {int((~S_('final')[coh.index]).sum())} · let-through rows {int(((allg.gate == GVB_PASSED) & S_('cohort')).sum())}. "
             "% are leverage-invariant (lev shown per row)._")
    if err:
        L.append(f"_{err} row(s) not priced this run (klines unavailable) — retried next run._")
    if late:
        L.append(f"_{late} row(s) left for the next run (the {X_TIME_S} s time budget was spent)._")
    return L + [""]


def m1_prints(m1):
    """1m rows [open_ms, o, h, l, c, …] → pseudo prints o (:00) → h (:15) → l (:30) → c (:59.999) for the tick walkers' 1m fallback. High
    before low = the conservative order AFTER a stop: the low of the minute that arms the lock cannot escape the trail. → (t int64, p float)."""
    if not m1:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    a = np.asarray([r[:5] for r in m1], dtype=float)
    t = (a[:, :1].astype(np.int64) + np.array([0, 15_000, 30_000, 59_999], dtype=np.int64)).ravel()
    p = a[:, [1, 2, 3, 4]].ravel()
    return t, p


def vwap_shadow(tt, pp, e, t0, t_stop, vwap, atr, c5t, c5p, gap=False):
    """🪜 the VWAP_STOP alternative (study BP k 0.5) for a fill the live −3 stop closed at t_stop: identical to live until t_stop; from there
    held, out at the first print after a 5m close (close time > t_stop, ≤ the 12 h cap) that is BOTH ≤ −3 % net AND below
    VWAP × (1 − 0.5 × ATR % / 100), or on a print ≤ −12 % net (hard floor) — unless the lock arms first (peak ≥ +3 over the prints; pre-stop
    prints capped below +3 because live closed it as STOP_LOSS = never armed), after which max(+2, peak − 2) applies on prints and the close /
    floor rules are off. gap=True (1m pseudo prints): a line crossed between two prints fills at the line. Net of FEE and SLIP.
    → (pnl, exit_ms, how, raw pre-stop replica peak)."""
    tt = np.asarray(tt, dtype=np.int64); pp = np.asarray(pp, dtype=float)
    m = tt >= int(t0)
    tt, pp = tt[m], pp[m]
    if not len(pp) or not e or _num(vwap) is None:
        return None, None, "no data", None
    n = len(pp)
    net = (pp / float(e) - 1) * 100 - FEE
    t_end = int(t0) + CAP_MIN * MIN
    pre = tt < int(t_stop)
    pk_raw = float(net[pre].max()) if pre.any() else -1e9
    pk0 = min(pk_raw, 3.0 - 1e-9)
    capnet = np.where(pre, np.minimum(net, pk0), net)
    pkb = np.maximum.accumulate(np.r_[-1e9, capnet[:-1]])     # the peak BEFORE each print
    armed = pkb >= 3
    lockline = np.maximum(2.0, pkb - 2.0)
    live = ~pre & (tt < t_end)
    first = lambda mm: int(np.flatnonzero(mm)[0]) if mm.any() else n
    i_lock = first(live & armed & (net <= lockline))
    i_floor = first(live & ~armed & (net <= VWS_FLOOR))
    a = _num(atr)
    a = a if a is not None and a > 0 else VWS_ATR_DEFAULT
    lim = float(vwap) * (1 - VWS_K * a / 100)
    i_close = n
    for ct, cp in zip(np.asarray(c5t, dtype=np.int64), np.asarray(c5p, dtype=float)):
        if ct <= int(t_stop) or ct > t_end:
            continue
        if (cp / float(e) - 1) * 100 - FEE <= VWS_LINE and cp < lim:
            ip = int(np.searchsorted(tt, ct, side="left"))      # the first print after the close
            if ip < n and tt[ip] < t_end and not armed[ip]:
                i_close = ip
            break                                               # the first qualifying close decides (armed / no print yet → no close exit)
    prv = np.r_[net[0], net[:-1]]
    fill = lambda i, line: float(min(line, prv[i])) if (gap and prv[i] > line) else float(net[i])
    i = min(i_lock, i_floor, i_close)
    if i < n:
        if i == i_lock:
            return fill(i, lockline[i]) - SLIP, int(tt[i]), "floor / trail", pk_raw
        if i == i_floor:
            return fill(i, VWS_FLOOR) - SLIP, int(tt[i]), "−12 floor", pk_raw
        return float(net[i]) - SLIP, int(tt[i]), "5m close ≤ −3 ∧ < VWAP − 0.5 ATR", pk_raw
    if tt[-1] >= t_end:
        lv = np.flatnonzero(tt < t_end)
        return float(net[lv[-1]]) - SLIP, t_end, "12 h cap", pk_raw
    return float(net[-1]) - SLIP, int(tt[-1]), "open", pk_raw


def replica_stop(tt, pp, e, t0, t_exit):
    """the shadow's own pre-stop replica (the live lock on the same prints) on a fill live did NOT stop: does it hit −3 before the live exit?
    → (stopped, at_ms). A replica stop there = the replica sees a wick live rode through (CLAUDE.md live-stopped rule)."""
    tt = np.asarray(tt, dtype=np.int64); pp = np.asarray(pp, dtype=float)
    m = (tt >= int(t0)) & (tt < int(t_exit))
    r = _walk_ticks(tt[m], pp[m], e, t0)
    return (r[2] == "stop"), (r[1] if r[2] == "stop" else None)


def vws_delta(stopped, shadow, actual):
    """Δ (shadow − live actual) — 0 by construction for a fill the live −3 stop did not close (the shadow IS the live exit there)."""
    if not stopped:
        return 0.0
    shadow, actual = _num(shadow), _num(actual)
    return None if shadow is None or actual is None else shadow - actual


def vws_validate(r):
    """raises ValueError on a row that must not be saved: a non-stopped fill with Δ ≠ 0, or a final, priced stopped fill whose Δ is missing
    or ≠ shadow − actual. Excluded rows only need their reason."""
    T = lambda v: str(v) in _TRUE
    if not T(r.get("stopped")):
        if _num(r.get("delta")) != 0.0:
            raise ValueError(f"non-stopped fill with Δ {r.get('delta')}")
        return
    if T(r.get("excluded")):
        if not isinstance(r.get("excl_reason"), str) or not r.get("excl_reason"):
            raise ValueError("excluded row without a reason")
        return
    if T(r.get("final")):
        d, s, a = _num(r.get("delta")), _num(r.get("shadow")), _num(r.get("actual"))
        if d is None or s is None or a is None:
            raise ValueError("final stopped fill without shadow / actual / Δ")
        if abs(d - (s - a)) > 1e-6:
            raise ValueError(f"Δ {d} ≠ shadow − actual {s - a}")


def vws_check(w):
    """the FROZEN VWAP_STOP gate on the first VWS_N stopped fills (by open time) of the cohort, all final. → (state, text)."""
    w = w.sort_values(["k", "pair"], kind="stable").head(VWS_N)
    d = pd.to_numeric(w.delta, errors="coerce")
    sv, dp = int((d > 0).sum()), int((d < 0).sum())
    t = f"{len(w)}/{VWS_N} stopped fills · saved {sv} (Δ {d[d > 0].sum():+.2f}) · deeper {dp} (Δ {d[d < 0].sum():+.2f}) · Δ sum {d.sum():+.2f}"
    if len(w) < VWS_N or not w.final.astype(str).isin(_TRUE).all() or d.isna().any():
        return "collecting", t
    nsl = {sl: int((w.sleeve.values == sl).sum()) for sl in ("LONG", "WIDE")}
    per = {sl: float(d[w.sleeve.values == sl].sum()) for sl in ("LONG", "WIDE")}
    q = [sl for sl in ("LONG", "WIDE") if nsl[sl] >= VWS_SLEEVE_MIN]
    t += " · " + " · ".join(f"{sl} {nsl[sl]} Δ {per[sl]:+.2f}" + ("" if sl in q else f" (< {VWS_SLEEVE_MIN}, leg not applied)") for sl in ("LONG", "WIDE"))
    if not q:
        return "collecting", t + f" · no sleeve with ≥ {VWS_SLEEVE_MIN} fills"
    top = (d.max() / d.sum() * 100) if d.sum() > 0 else float("inf")
    t += f" · top fill {top:.0f} % of the gain"
    ok = d.sum() > 0 and sv > dp and all(per[sl] > 0 for sl in q) and top <= VWS_TOP
    return ("candidate" if ok else "close"), t


VWS_SRC = ("k", "opened_at", "pair", "sleeve", "entry", "vwap", "atr", "vs_vwap", "actual", "close_reason", "stopped", "exit_live_at")


def _vws_src_fill(r):
    """an export row → the stored source fields (raises on a malformed row; the caller counts it)."""
    stopped = str(r.close_reason) == "STOP_LOSS"
    return dict(k=str(r.k), opened_at=str(r.opened_at)[:26], pair=str(r.pair), sleeve=str(r.entry_strategy).replace("FRENZY_", ""),
                entry=float(r.entry_price), vwap=_num(r.entry_frenzy_vwap), atr=_num(getattr(r, "entry_atr_pct", None)),
                vs_vwap=_num(r.entry_frenzy_vs_vwap_pct), actual=_num(r.pnl_percentage), close_reason=str(r.close_reason),
                stopped=stopped, exit_live_at=(str(r.closed_at)[:26] if pd.notna(r.closed_at) else None))


def _vws_price(src, now_ms, budget):
    """one closed fill (source fields) → its stored row. Stopped: the BP k 0.5 shadow; not stopped: the pre-stop replica parity check (Δ 0 by
    construction). Ticks once the horizon passed and the archive is out, else 1m pseudo prints (provisional; final on 1m when the archive is
    missing / empty past TICK_GIVEUP_D, or by age after STALE_D days). A stopped fill with no VWAP stamp or no live P&L → final, excluded,
    with its reason (stored once). Raises on data trouble (the caller keeps the old row)."""
    row = dict(src, day=src["k"][:10], cohort=src["k"] >= GC_FROM, ver=VWS_VER, prelock=src["k"] < LOCK_FROM, excluded=False, excl_reason=None,
               shadow=None, shadow_how=None, shadow_exit_at=None, delta=None, pre_peak=None, replica_stop=None, replica_stop_at=None,
               px_src=None, tick_state=None, final=False)
    stopped = bool(src["stopped"])
    if stopped and (src["vwap"] is None or src["actual"] is None):
        row.update(excluded=True, excl_reason=("no entry_frenzy_vwap stamp" if src["vwap"] is None else "no live P&L"), final=True)
        return row
    if not src.get("exit_live_at"):
        raise ValueError("no live exit time")
    sym, e = src["pair"], float(src["entry"])
    t0, t_x = _ms(src["opened_at"]), _ms(src["exit_live_at"])
    t_end = t0 + CAP_MIN * MIN
    horizon = t_end if stopped else t_x
    giveup = now_ms > (horizon // 86_400_000 + 1) * 86_400_000 + TICK_GIVEUP_D * 86_400_000
    stale = now_ms > horizon + STALE_D * 86_400_000
    st, px = "pending", None
    if now_ms >= horizon:
        st, tt, pp = _ticks(sym, t0, horizon + 2 * MIN, now_ms, budget)
        if st == "ok" and len(tt):
            px = "tick"
        elif st == "ok":
            st = "empty"
    if px is None:
        m1 = [b for b in _kl(sym, "1m", t0 // MIN * MIN, min(now_ms, horizon + MIN)) if b[0] + MIN <= now_ms]
        if not m1:
            raise ValueError("1m klines unavailable")
        tt, pp = m1_prints(m1); px = "1m"
    fin = px == "tick" or bool(now_ms >= horizon and (st == "missing" or (st == "empty" and giveup) or stale))
    if px == "1m" and fin and stale and st not in ("missing", "empty"):
        px = "1m (age)"
    if stopped:
        b5 = [b for b in _kl(sym, "5m", t_x // BAR * BAR, min(now_ms, t_end)) if b[0] + BAR <= now_ms]
        p, x_ms, how, pk = vwap_shadow(tt, pp, e, t0, t_x, src["vwap"], src["atr"], [b[0] + BAR for b in b5], [b[4] for b in b5], gap=px != "tick")
        row.update(shadow=p, shadow_how=how, shadow_exit_at=(_iso(x_ms) if x_ms else None), pre_peak=pk,
                   final=fin and how not in ("open", "no data"), delta=vws_delta(True, p, src["actual"]))
    else:
        rs, rat = replica_stop(tt, pp, e, t0, t_x)
        row.update(shadow=src["actual"], shadow_how="live exit", replica_stop=bool(rs), replica_stop_at=(_iso(rat) if rat else None),
                   final=fin, delta=vws_delta(False, None, None))
    row.update(px_src=px, tick_state=st)
    return row


def vws_run(now_ms, F):
    """price every closed FRENZY / WIDE fill (stopped: the shadow; not stopped: the replica parity), re-price non-final stored rows from their
    own fields, VALIDATE (bad rows → a timestamped .bad file), save, render the VWAP_STOP section."""
    old = _load_csv(VWS_CSV, ("k", "pair", "final", "ver"))
    done = set()
    if len(old):
        done = {(a, b) for a, b, f_, v in zip(old.k, old.pair, old.final, old.ver) if str(f_) in _TRUE and str(v) in (str(VWS_VER), f"{VWS_VER}.0")}
    work, err, late = {}, 0, 0
    W = F[(F.status.astype(str).str.upper() == "CLOSED") & (F.k.astype(str) >= GC_SCAN_FROM)] if len(F) and "status" in F else F.iloc[0:0]
    for r in W.itertuples():
        if (r.k, r.pair) in done or pd.isna(getattr(r, "close_reason", None)):   # an export without close_reason never stores a fill as "not stopped"
            continue
        try:
            work[(r.k, r.pair)] = _vws_src_fill(r)
        except Exception:
            err += 1
    if len(old):
        for r in old.to_dict("records"):
            key = (r["k"], r["pair"])
            if key in done or key in work or str(r.get("excluded")) in _TRUE:
                continue
            try:
                src = {c: r.get(c) for c in VWS_SRC}
                src.update(stopped=str(r.get("stopped")) in _TRUE, vwap=_num(src["vwap"]), atr=_num(src["atr"]), actual=_num(src["actual"]),
                           vs_vwap=_num(src["vs_vwap"]), entry=float(src["entry"]), exit_live_at=(src["exit_live_at"] if isinstance(src["exit_live_at"], str) else None))
                work[key] = src
            except Exception:
                err += 1
    order = sorted(work.values(), key=lambda s: (s["k"] >= GC_FROM, bool(s["stopped"]), s["k"], s["pair"]), reverse=True)
    budget = {"dl": X_DL, "deadline": time.monotonic() + X_TIME_S}; rows = []
    for src in order:
        if time.monotonic() > budget["deadline"]:
            late += 1
            continue
        try:
            rows.append(_vws_price(src, now_ms, budget))
        except Exception:
            err += 1
    new = pd.DataFrame(rows)
    allv = pd.concat([old, new], ignore_index=True) if len(new) else old.copy()
    L = ["## 🪜 VWAP_STOP shadow — after the live −3 stop, hold to the study's BP k 0.5 exit (observe-only, registered 2026-10-06)", "",
         "Every FRENZY_LONG / FRENZY_WIDE fill the live −3 stop closed (close_reason STOP_LOSS) re-priced from its ACTUAL entry "
         "(reports/FRENZY_STAIRCASE_STUDY_2026-10-06.md §3b / §5, BP k 0.5): identical to live until the stop; then held, out at the first print after a "
         "5m close that is BOTH ≤ −3 % net AND below VWAP × (1 − 0.5 × entry ATR % / 100) (entry_frenzy_vwap / entry_atr_pct), hard floor −12 % net, "
         "the lock unchanged (+3 → max(+2, peak − 2); close / floor rules off once armed), 12 h cap from the entry, 0.09 % fees + 0.10 % slippage. "
         "Ticks once the archive is out, else 1m pseudo prints open → high → low → close (ᵖ provisional). ⚠ Tick prints can stop / trail on wicks the "
         "live poller rides through — the parity line below counts them on the fills live did NOT stop. Δ = shadow − the live actual · saved = Δ > 0 · "
         "deeper = Δ < 0; fills the −3 did NOT close are Δ 0 by construction (CLAUDE.md: exit counterfactuals on the live-stopped cohort only). "
         f"Counted from {GC_FROM[:10]}; earlier rows are reference only († = opened before the lock went live {LOCK_FROM[5:16].replace('T', ' ')} UTC: "
         "pre-lock exit regime).", ""]
    if not len(allv):
        return L + ["No closed FRENZY / WIDE fill in the exports yet.", ""] + ([f"_{err} fill(s) not priced this run — retried next run._"] if err else [])
    allv = allv.drop_duplicates(["k", "pair"], keep="last").sort_values(["k", "pair"], kind="stable").reset_index(drop=True)
    bad = []
    for i, r in zip(allv.index, allv.to_dict("records")):
        try:
            vws_validate(r)
        except ValueError as ex:
            bad.append((i, str(ex)))
    qpath = None
    if bad:
        qpath = _bad_path(VWS_CSV)
        try:
            allv.loc[[i for i, _ in bad]].assign(bad_reason=[m for _, m in bad]).to_csv(qpath, index=False)
        except Exception:
            qpath = "(write failed)"
        allv = allv.drop(index=[i for i, _ in bad]).reset_index(drop=True)
    _save_csv(allv, VWS_CSV)
    S_ = lambda c: allv[c].astype(str).isin(_TRUE) if c in allv else pd.Series(False, index=allv.index)
    f = lambda v: "–" if _num(v) is None else f"{float(v):+.2f}"
    sd = allv[S_("stopped")]
    L += ["| Opened UTC | Pair | Sleeve | vs VWAP | Live stop at | Actual | Shadow | Δ | | Shadow exit | Px |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sd.tail(15).itertuples():
        mk = ("" if str(r.final) in _TRUE else "ᵖ") + ("" if str(r.cohort) in _TRUE else " (ref" + (" · † pre-lock exit regime)" if str(r.prelock) in _TRUE else ")"))
        if str(r.excluded) in _TRUE:
            L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{mk} | {str(r.pair).replace('USDT', '')} | {r.sleeve} | {f(r.vs_vwap)} | – | {f(r.actual)} | – | – | "
                     f"excluded | {r.excl_reason} | – |")
            continue
        d = _num(r.delta)
        tag = "–" if d is None else "saved" if d > 0 else "deeper" if d < 0 else "same"
        L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{mk} | {str(r.pair).replace('USDT', '')} | {r.sleeve} | {f(r.vs_vwap)} | {str(r.exit_live_at)[5:16].replace('T', ' ')} | "
                 f"{f(r.actual)} | {f(r.shadow)} | {f(d)} | {tag} | {r.shadow_how} {str(r.shadow_exit_at)[5:16].replace('T', ' ') if isinstance(r.shadow_exit_at, str) else ''} | {r.px_src} |")
    use = sd[~S_("excluded")[sd.index]]
    st, tx = vws_check(use[S_("cohort")[use.index]])
    lab = {"candidate": "📋 CANDIDATE for a pre-registered study (no arm from here)", "close": "❌ close the idea", "collecting": "⏳ collecting"}
    L += ["", f"**VWAP_STOP gate (FROZEN, first {VWS_N} stopped fills from {GC_FROM[:10]}, all final: Δ sum > 0 ∧ saved > deeper ∧ Δ sum > 0 on every sleeve "
              f"with ≥ {VWS_SLEEVE_MIN} of those fills (≥ 1 such sleeve, else collecting) ∧ no single fill > {VWS_TOP:.0f} % of the gain):** "
              + lab.get(st, st) + f" ({tx})", f"- Year reference: {VWS_YEAR}."]
    rf = use[~S_("cohort")[use.index]]; dr = pd.to_numeric(rf.delta, errors="coerce").dropna()
    L.append(f"- Reference (before {GC_FROM[:10]}): {len(rf)} stopped fills · saved {int((dr > 0).sum())} (Δ {dr[dr > 0].sum():+.2f}) · deeper "
             f"{int((dr < 0).sum())} (Δ {dr[dr < 0].sum():+.2f}) · {int((~S_('final')[rf.index]).sum())} provisional · "
             f"{int(S_('prelock')[rf.index].sum())} from the pre-lock exit regime.")
    ns = allv[~S_("stopped") & S_("final")]
    rs = ns[S_("replica_stop")[ns.index]]
    L.append(f"- Parity (fills live did NOT stop; Δ = 0 by construction, validated before save): {len(ns)} final fills, {len(rs)} replica stops "
             "before their live exit " + ("✓" if not len(rs) else "⚠ the replica stops fills live rode through — tick / 1m wicks: "
                                         + ", ".join(f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16]} @ {str(r.replica_stop_at)[11:16]} ({r.px_src})" for r in rs.itertuples()))
             + f" · {int((~S_('stopped') & ~S_('final')).sum())} still provisional.")
    ex = sd[S_("excluded")[sd.index]]
    stall = allv[~S_("final") & ~S_("excluded")]
    stall = stall[[now_ms > _ms(k) + CAP_MIN * MIN + TICK_GIVEUP_D * 86_400_000 for k in stall.k]]
    L.append(f"- Excluded (stopped fill without a VWAP stamp / live P&L, stored once): {len(ex)}"
             + (" (" + "; ".join(f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16]}: {r.excl_reason}" for r in ex.itertuples()) + ")" if len(ex) else "")
             + f" · stalled (past 12 h + {TICK_GIVEUP_D} d, ticks still not in hand; finalise on 1m by age at {STALE_D} d): {len(stall)} · "
             f"replica peak ≥ +3 before a live stop (a wick the poller missed; the shadow ignores it): {int((pd.to_numeric(sd['pre_peak'], errors='coerce') >= 3).sum()) if 'pre_peak' in sd else 0}.")
    if bad:
        L.append(f"_⚠ {len(bad)} row(s) failed validation and were quarantined to {os.path.basename(qpath)}: " + "; ".join(m for _, m in bad[:3]) + "._")
    if err:
        L.append(f"_{err} fill(s) not priced this run (klines / fields unavailable) — retried next run._")
    if late:
        L.append(f"_{late} fill(s) left for the next run (the {X_TIME_S} s time budget was spent)._")
    return L + [""]


def _extras(now_ms, th, F, J, allr):
    """trackers 7 – 9 after the exit table, each in its own try/except (one never breaks another or the scout)."""
    out = _gc_safe(now_ms, th, F, J, allr)
    Je = J if J is not None else pd.DataFrame(columns=["t", "e", "pair", "gate", "strategy"])
    try:
        out += gvb_run(now_ms, th, Je, allr, F)
    except Exception as ex:
        out += ["## 🌊 GVOL_BLOCKED", "", f"Unavailable this run ({str(ex)[:120]}).", ""]
    try:
        out += vws_run(now_ms, F)
    except Exception as ex:
        out += ["## 🪜 VWAP_STOP shadow", "", f"Unavailable this run ({str(ex)[:120]}).", ""]
    return out


def selftest():
    ok = 0

    def chk(c, m):
        nonlocal ok
        assert c, m
        ok += 1
    t0 = 1_790_000_000_000 // BAR * BAR
    m = lambda ps: [[t0 + i * MIN, p, p, p, p] for i, p in enumerate(ps)]
    chk(abs(walk([[t0, 100, 100, 96, 96]], 100, "LOCK2")[0] + 3.0) < 1e-9, "−3 stop fills at the line inside a minute")
    chk(abs(walk(m([100, 96]), 100, "LOCK2")[0] + 4.09) < 1e-9, "a minute that opens through the stop fills at its open (gap)")
    up = m([100, 104, 110]) + [[t0 + 3 * MIN, 110, 110, 105, 105]]   # trades down through the lines inside one minute
    chk(abs(walk(up, 100, "LOCK2")[0] - 7.91) < 0.02, "lock trails 2 pts below the prior peak (fill at the line)")
    chk(abs(walk(up, 100, "LOCK3")[0] - 6.91) < 0.02, "3-pt trail")
    chk(walk(up, 100, "FIX3")[0] == 3.0, "fixed +3")
    chk(abs(walk(m([100, 104, 110, 100]), 100, "LOCK2")[0] + 0.09) < 1e-9, "a minute opening through the trail line fills at its open")
    chk(walk(m([100, 101]), 100, "LOCK2")[2] == "open", "still open")
    b5 = pd.DataFrame(dict(t=[t0], c=[105.0], e20=[106.0], e50=[104.0], atr=[1.0]))
    p = [100, 104] + [105] * 4
    r20 = walk(m(p), 100, "EMA20", b5); r50 = walk(m(p), 100, "EMA50", b5)
    chk(r20[2] == "5m close < EMA" and abs(r20[0] - 4.91) < 0.02, "EMA20 exit at the 5m close below the EMA")
    chk(r50[2] == "open", "EMA50 holds while the close is above it")
    chk(walk(m(p), 100, "ATRE", b5, atr0=2.0)[2] == "ATR < entry", "ATRE exits when the 5m ATR falls below the entry ATR")
    chk(walk(m(p), 100, "ATRE", b5, atr0=0.5)[2] == "open", "ATRE holds while ATR ≥ entry")
    df = pd.DataFrame(dict(day=[f"d{i}" for i in range(70)], pair=[f"P{i}" for i in range(70)], v=[0.3] * 70))
    chk(bar_check(df, "v", 60, 30)[0] == "review", "all bars met → review")
    chk(bar_check(df.head(10), "v", 60, 30)[0] == "collecting", "small N → collecting")
    df2 = df.copy(); df2.loc[0, "v"] = 500.0; df2.loc[1:, "v"] = -0.1
    chk(bar_check(df2, "v", 60, 30)[0] == "retire", "one-trade lottery → retire")
    gap = [[t0, 100, 100, 100, 100], [t0 + 800 * MIN, 101, 101, 101, 101]]
    chk(walk(gap, 100, "LOCK2")[2] == "12 h cap", "the 12 h cap is clock time — a kline gap cannot stretch it")
    conc = pd.DataFrame(dict(day=[f"d{i}" for i in range(70)], pair=["A"] * 35 + [f"P{i}" for i in range(35)], v=[0.3] * 70))
    chk(bar_check(conc, "v", 60, 30)[0] == "retire", "one pair carrying half the net gain fails the 35 % pair bar")
    aw = pd.DataFrame(dict(day=[f"d{i}" for i in range(40)], pair=[f"P{i}" for i in range(40)], atr_chg30=[20.0] * 35 + [5.0] * 5,
                           LOCK2=[2.0] * 40, LOCK3=[2.6, 2.4, 2.8, 2.5] * 10))
    chk(atrfast_check(aw)[0] == "review", "fast fills where the 3-pt trail beats the lock everywhere → review")
    chk(atrfast_check(aw.assign(LOCK3=1.5))[0] == "retire", "3-pt trail worse → retire")
    chk(atrfast_check(aw.head(20))[0] == "collecting", "< 30 fast fills → collecting")
    big = pd.concat([aw.assign(day=aw.day + s_) for s_ in "ab"], ignore_index=True).assign(LOCK3=[2.4, 2.3] * 40, pair="A")
    chk(atrfast_check(big)[0] == "retire", "≥ 60 fast fills, one pair carrying everything → retire")
    cw = pd.DataFrame(dict(day=[f"d{i}" for i in range(40)], above_share=[50.0] * 20 + [90.0] * 20, LOCK2=[-1.0, -1.2, 0.4, -1.5] * 5 + [0.5] * 20))
    chk(choppy_check(cw)[0] == "arm_review", "would-block fills clearly losing → arm review")
    chk(choppy_check(cw.assign(LOCK2=0.3))[0] == "retire", "would-block fills not losing → retire")
    chk(choppy_check(cw.tail(25))[0] == "collecting", "< 15 would-block fills → collecting")
    w = pd.DataFrame(dict(day=[f"d{i % 30}" for i in range(100)], btc_rsi12=[40.0] * 50 + [60.0] * 50, LOCK2=[0.8] * 50 + [-0.6] * 50))
    chk(soft_check(w)[0] == "review", "soft clearly better, not-soft clearly losing → review candidate")
    chk(soft_check(w.head(60))[0] == "collecting", "< 40 fills in one state → collecting")
    w2 = w.assign(LOCK2=[0.1, 0.1, -0.1, -0.1] * 25, btc_rsi12=[40.0, 60.0] * 50)   # both states the same mix → no gap
    chk(soft_check(pd.concat([w2, w2.assign(day=w2.day + "b")]))[0] == "retire", "no gap at ≥ 60 per state → retire")
    # 🟢 WIDE_BY_CODE
    chk(wide_code(2.38, 0.519, 2.5) == "GREEN_BAR", "RLC 10-05: ATR under the cap + green candle → GREEN_BAR")
    chk(wide_code(4.34, -0.32, 2.5) == "ATR_HIGH", "ATR over the cap + red candle → ATR_HIGH")
    chk(wide_code(6.6, 0.4, 2.5) == "BOTH", "ATR over the cap + green candle → BOTH (the journal says ATR_HIGH)")
    chk(wide_code(2.5, 0.0, 2.5) == "NONE", "ATR exactly at the cap + flat candle → neither refusal (parity alarm)")
    chk(wide_code(None, 0.4, 2.5) is None and wide_code(3.0, float("nan"), 2.5) is None, "missing input → unknown")
    chk(wide_code(float("inf"), 0.2, 2.5) == "BOTH" and wide_code(float("inf"), -0.2, 2.5) == "ATR_HIGH", "journal ATR_HIGH + kline colour")
    bw = pd.DataFrame(dict(day=[f"d{i % 16}" for i in range(40)], pair="X", code=["GREEN_BAR"] * 30 + ["ATR_HIGH"] * 10, code_src="stamp",
                           actual=[1.0, -1.0, 2.0] * 10 + [-3.0] * 10, LOCK2=0.5))
    bs = {s["code"]: s for s in bycode_stats(bw)}
    chk(bs["GREEN_BAR"]["status"] == "review" and bs["ATR_HIGH"]["status"] == "collecting" and bs["BOTH"]["n"] == 0, "review due per code at ≥ 30 on ≥ 15 days")
    chk(abs(bs["GREEN_BAR"]["wr"] - 66.67) < 0.1 and abs(bs["ATR_HIGH"]["act"] + 3.0) < 1e-9, "WR / avg on the actual fill P&L")
    # 🟢 walk_ticks (the live lock on prints)
    tk = lambda ps: (np.array([t0 + i * 1000 for i in range(len(ps))]), np.array(ps, float))
    r_ = walk_ticks(*tk([100, 99, 96.8, 95]), 100, t0)
    chk(r_[2] == "stop" and abs(r_[0] - (-3.29)) < 1e-9, "a print through −3 exits AT that print")
    r_ = walk_ticks(*tk([100, 103.5, 106, 104.5, 103.8]), 100, t0)
    chk(r_[2] == "floor / trail" and abs(r_[0] - 3.71) < 1e-9, "armed at +3 → trails 2 pts below the peak (peak +5.91 → line +3.91)")
    r_ = walk_ticks(*tk([100, 103.2, 102.0]), 100, t0)
    chk(r_[2] == "floor / trail" and abs(r_[0] - 1.91) < 1e-9, "the +2 floor once armed (exits at the crossing print)")
    chk(walk_ticks(*tk([100, 101, 100.5]), 100, t0)[2] == "open", "still open")
    gt = (np.array([t0, t0 + 1000, t0 + 800 * MIN]), np.array([100.0, 101.0, 90.0]))
    chk(walk_ticks(*gt, 100, t0)[2] == "12 h cap" and abs(walk_ticks(*gt, 100, t0)[0] - 0.91) < 1e-9, "12 h clock cap on prints")
    chk(walk_ticks(*tk([99, 100, 101]), 100, t0 + 1000)[0] is not None and walk_ticks([], [], 100, t0)[2] == "no data", "prints before t0 ignored")
    # 🟢 FRENZY_GREEN_CLOCK: cohort / episode dedupe, FORMAL bars, entry + slippage, gvol
    ep = pd.DataFrame(dict(k=["2026-10-08T01:00:00", "2026-10-08T00:00:00", "2026-10-08T00:30:00", "2026-10-08T02:00:00", "2026-10-06T23:00:00"],
                           pair=["X", "X", "X", "Y", "Y"], spike_at=["s1"] * 3 + ["s2", "s2"], v2=[True, False, True, True, True],
                           eligible=[True, True, True, True, True], gvol="pass", cohort=[True, True, True, True, False]))
    chk(list(gc_counted(ep)) == [False, True, True, True, False], "first per spike WITHIN V2 and WITHIN V1; a pre-floor row never hides a cohort row")
    chk(list(gc_counted(ep.assign(gvol=["pass", "pass", "unknown", "pass", "pass"]))) == [True, True, False, True, False], "gvol unknown is not counted")
    gw = pd.DataFrame(dict(day=[f"d{i % 20}" for i in range(36)], pair=[f"P{i}" for i in range(36)], LOCK=[2.0, -1.0, 1.5] * 12))
    chk(gc_check(gw)[0] == "review", "V2 meeting every pre-registered leg → propose promotion")
    chk(gc_check(gw.head(20))[0] == "collecting", "< 30 signals → collecting")
    chk(gc_check(gw.assign(LOCK=[1.0, -1.5] * 18))[0] == "retire", "mean ≤ 0 at 30 → retire (addition)")
    chk(gc_check(gw.assign(day="d0"))[0] == "collecting", "< 15 days → no verdict yet")
    chk(gc_check(gw.assign(LOCK=[0.6, -0.4, 0.3] * 12))[0] == "collecting", "mean +0.17 < +0.30 → no promotion")
    chk(gc_check(gw.assign(LOCK=[2.0, -1.0, -0.5] * 12))[0] == "collecting", "WR 33 % < 55 % → no promotion")
    conc = gw.assign(pair=["A"] * 12 + [f"P{i}" for i in range(24)])
    chk(gc_check(conc)[0] == "collecting", "one pair > 25 % of the net → no promotion")
    chk(gc_check(pd.concat([conc, conc.assign(day=conc.day + "b")]))[0] == "retire", "no verdict by 60 → retire (addition)")
    lot = gw.assign(LOCK=[40.0] * 3 + [-0.3] * 33)
    chk(gc_check(lot)[0] == "collecting" and "w/o top 5" in gc_check(lot)[1], "a 3-trade lottery fails the drop-top-5 leg (addition)")
    ww = pd.DataFrame(dict(day=[f"d{i % 20}" for i in range(30)], pair=[f"P{i}" for i in range(30)], LOCK=[1.5, 1.0, -0.5] * 10))
    chk(w_check(ww.assign(LOCK=[1.5, 1.0, 1.2, -0.5] * 7 + [1.0, 1.0]))[0] == "review", "Pattern-W: WR ≥ 70 ∧ mean ≥ +0.50 → propose")
    chk(w_check(ww)[0] == "collecting", "Pattern-W: WR 67 % < 70 % → collecting")
    tt = np.array([t0 + 1000, t0 + 11_999, t0 + 12_000, t0 + 60_000]); pp = np.array([99.0, 99.5, 100.0, 100.5])
    chk(gc_entry(tt, pp, t0) == (100.0, t0 + 12_000), "entry = the first print at or after the signal close + 12 s")
    chk(abs(walk_ticks(tt, pp, 100.0, t0 + 12_000, slip=SLIP)[0] - (0.5 - FEE - SLIP)) < 1e-9, "0.10 slippage on top of the 0.09 fees")
    jg = lambda *g: pd.DataFrame(dict(e=["BLOCK"] * len(g), gate=list(g), strategy=""))
    chk(gvol_state(jg("FRENZY_WIDE_CHOPPY")) == "pass" and gvol_state(jg("FRENZY_WIDE_LATE")) == "pass", "WIDE refusals after the gvol gate → pass")
    chk(gvol_state(jg("FRENZY_WIDE_GVOL_HIGH")) == "high" and gvol_state(jg("FRENZY_WIDE_MAX_SLOTS")) == "unknown", "gvol high / unknown")
    # 🌊 GVOL_BLOCKED: who would trade today, the pair-episode dedupe, the frozen bar
    thx = SimpleNamespace(frenzy_wide_hold_green_streak=0.0, frenzy_wide_above_share_min=0.0)
    epx = dict(fresh_on=True, above_streak=17, bar_ret_pct=0.5, above_share=90.0)
    chk(gvb_take("LONG", epx, "FRENZY_READY", 1.8, thx) == (True, "") and not gvb_take("LONG", epx, "FRENZY_ON", 1.8, thx)[0], "LONG = the replay says READY")
    chk(gvb_take("WIDE", epx, "FRENZY_GREEN_BAR", 2.0, thx)[0], "WIDE hold-green (streak 17 > 12) taken — the frozen 12 even with the live switch off")
    chk(gvb_take("WIDE", {**epx, "above_streak": 12}, "FRENZY_GREEN_BAR", 2.0, thx) == (False, "FRENZY_WIDE_RECLAIM"), "streak 12 = reclaim → not taken")
    chk(gvb_take("WIDE", epx, "FRENZY_ATR_HIGH", 3.1, thx) == (False, "FRENZY_WIDE_ATR_HIGH"), "WIDE ATR_HIGH → hold-green blocks it")
    chk(not gvb_take("WIDE", epx, "FRENZY_READY", 1.0, thx)[0] and not gvb_take("WIDE", epx, "FRENZY_GREEN_BAR", None, thx)[0], "WIDE parity: a WIDE code + a readable ATR")
    chk(not gvb_take("WIDE", {**epx, "above_share": 50.0}, "FRENZY_GREEN_BAR", 2.0, SimpleNamespace(**{**vars(thx), "frenzy_wide_above_share_min": 67.8}))[0],
        "choppy (when armed live) is judged before hold-green")
    S1, S2, S3 = "2026-10-07T20:00:00", "2026-10-07T21:00:00", "2026-10-07T22:00:00"
    gb = pd.DataFrame(dict(k=["2026-10-08T03:00:00", "2026-10-08T01:00:00", "2026-10-08T02:00:00", "2026-10-06T23:00:00", "2026-10-08T05:00:00",
                              "2026-10-08T06:00:00", "2026-10-08T07:00:00"],
                           pair=["X", "X", "Y", "Y", "Z", "W", "W"], spike_at=[S1, "2026-10-07T20:25:00", S2, S2, S3, S3, "2026-10-07T22:20:00"],
                           cohort=[True, True, True, False, True, True, True], would_take=[True, True, True, True, False, True, True],
                           gate=["FRENZY_GVOL_HIGH", "FRENZY_WIDE_GVOL_HIGH", "FRENZY_GVOL_HIGH", "FRENZY_GVOL_HIGH", "FRENZY_GVOL_HIGH",
                                 "FRENZY_GVOL_HIGH", GVB_PASSED]))
    gm = gvb_masks(gb)
    chk(list(gm["counted"]) == [False, True, True, False, False, False, False],
        "one per pair-episode (spikes 25 min apart merged, across sleeves); floor before the dedupe; not taken → out")
    chk(list(gm["double"]) == [False] * 5 + [True, False] and list(gm["passed"]) == [False] * 6 + [True],
        "a blocked episode that also had a live fill (spike drift 20 min) is excluded, the fill is a let-through row")
    chk(not gvb_counted(gb.assign(gate="FRENZY_GVOL_UNREAD")).any() and not gvb_counted(gb.assign(catchup=True)).any(), "UNREAD / catch-up never counted")
    ek = episode_keys(pd.DataFrame(dict(pair=["A", "A", "A", "B"], spike_at=["2026-10-04T14:25:00", "2026-10-04T14:30:00", "2026-10-04T15:30:00", "s9"])))
    chk(ek[0] == ek[1] != ek[2] and ek[3] == "B|s9", "30-min episode merge (AIN 14:25 vs 14:30); unparseable keys on itself")
    dd = [f"d{i // 2}" for i in range(24)]
    chk(gvb_check([0.5, -0.2] * 12, dd, 0.1)[0] == "review", "blocked +0.15 ≥ 0 and ≥ let-through +0.10 → review flag")
    chk(gvb_check([0.5, -0.2] * 12, dd, 0.4)[0] == "inconclusive" and gvb_check([0.5, -0.2] * 12, dd, None)[0] == "inconclusive",
        "blocked ≥ 0 but below the let-through mean (or none) → inconclusive")
    chk(gvb_check([-0.5, 0.0] * 12, dd, 0.1)[0] == "confirmed", "blocked −0.25 ≤ −0.20 → gate confirmed")
    chk(gvb_check([5.0] + [-0.1] * 23, [f"d{i % 12}" for i in range(24)], 0.0)[0] == "inconclusive", "one day carrying ≥ 50 % of the net → window leg fails")
    chk(gvb_check([0.5] * 24, ["d0"] * 24, 0.1)[0] == "collecting" and gvb_check([0.5] * 19, dd[:19], 0.1)[0] == "collecting", "< 10 days / < 20 signals → collecting")
    # 🪜 VWAP_STOP shadow (BP k 0.5: close ≤ −3 net ∧ < VWAP × (1 − 0.5 × ATR / 100); floor −12; lock unchanged; slip on the exit)
    tv = lambda ps, dt=60_000: (np.array([t0 + i * dt for i in range(len(ps))]), np.array(ps, float))
    tt_, pp_ = tv([100, 98, 96.9, 97.5, 101, 104, 106, 103.9])       # stop at i=2; arms at 104; trails
    r_ = vwap_shadow(tt_, pp_, 100, t0, t0 + 2 * MIN, 97.0, 2.0, [t0 + BAR], [97.0])
    chk(r_[2] == "floor / trail" and abs(r_[0] - (3.9 - FEE - SLIP)) < 1e-9, "close −3.09 but above VWAP − 0.5 ATR (96.03) → held; the lock arms and trails (saved), slip charged")
    tt_, pp_ = tv([100, 98, 96.9, 96, 95.5, 95])
    r_ = vwap_shadow(tt_, pp_, 100, t0, t0 + 2 * MIN, 99.0, 2.0, [t0 + 3 * MIN], [96.0])
    chk(r_[2].startswith("5m close") and r_[1] == t0 + 3 * MIN and abs(r_[0] - (-4.0 - FEE - SLIP)) < 1e-9, "close ≤ −3 ∧ < 98.01 → out at the first print after it")
    r_ = vwap_shadow(tt_, pp_, 100, t0, t0 + 2 * MIN, 99.0, 2.0, [t0 + 3 * MIN], [97.5])
    chk(r_[2] == "open", "a close above −3 net (−2.59) is not an exit even below the VWAP line (pure widening)")
    r_ = vwap_shadow(tt_, pp_, 100, t0, t0 + 2 * MIN, 96.5, 2.0, [t0 + 3 * MIN], [96.0])
    chk(r_[2] == "open", "a close ≤ −3 but above VWAP − 0.5 ATR (95.54) is not an exit")
    chk(vwap_shadow(tt_, pp_, 100, t0, t0 + 2 * MIN, 97.0, None, [t0 + 3 * MIN], [96.0])[2].startswith("5m close")
        and vwap_shadow(tt_, pp_, 100, t0, t0 + 2 * MIN, 97.0, 3.0, [t0 + 3 * MIN], [96.0])[2] == "open", "unreadable ATR → the study's 2.0 (line 96.03); ATR 3 → 95.55")
    r_ = vwap_shadow(*tv([100, 97, 92, 87.5, 86]), 100, t0, t0 + MIN, 99.0, 2.0, [], [])
    chk(r_[2] == "−12 floor" and abs(r_[0] - (-12.5 - FEE - SLIP)) < 1e-9 and r_[1] == t0 + 3 * MIN, "−12 hard floor on prints (ticks fill at the print)")
    r_ = vwap_shadow(*tv([100, 97, 92, 87.5, 86]), 100, t0, t0 + MIN, 99.0, 2.0, [], [], gap=True)
    chk(abs(r_[0] - (-12.0 - SLIP)) < 1e-9, "1m pseudo prints: a floor crossed between two prints fills at the line")
    r_ = vwap_shadow(*tv([100, 104, 96.5, 90]), 100, t0, t0 + 2 * MIN, 99.0, 2.0, [t0 + 3 * MIN], [90.0])
    chk(r_[2].startswith("5m close") and r_[3] >= 3, "a replica peak ≥ +3 before the live stop never arms the shadow (live never armed)")
    r_ = vwap_shadow(*tv([100, 97, 103.5, 101.3]), 100, t0, t0 + MIN, 99.0, 2.0, [t0 + 4 * MIN], [90.0])
    chk(r_[2] == "floor / trail" and r_[1] == t0 + 3 * MIN, "once armed the close rule is off; the lock decides")
    r_ = vwap_shadow(np.array([t0, t0 + MIN, t0 + 800 * MIN]), np.array([100.0, 96.0, 99.0]), 100, t0, t0 + MIN, 90.0, 2.0, [t0 + 5 * MIN], [97.0])
    chk(r_[2] == "12 h cap" and abs(r_[0] - (-4.0 - FEE - SLIP)) < 1e-9, "12 h clock cap from the entry")
    r_ = vwap_shadow(*tv([100, 97, 96]), 100, t0, t0 + MIN, 99.0, 2.0, [t0 + 5 * MIN], [90.0])
    chk(r_[2] == "open", "a qualifying close the prints have not reached yet is not an exit")
    mt, mp = m1_prints([[t0, 100, 102, 99, 101, 5]])
    chk(list(mp) == [100, 102, 99, 101] and list(mt - t0) == [0, 15_000, 30_000, 59_999], "1m pseudo prints: open → high → low → close")
    rp = replica_stop(*tv([100, 98, 96.5, 99, 104]), 100, t0, t0 + 4 * MIN)
    chk(rp == (True, t0 + 2 * MIN) and replica_stop(*tv([100, 98, 99, 104]), 100, t0, t0 + 3 * MIN) == (False, None),
        "parity replica: a −3 print before the live exit = a replica stop")
    chk(vws_delta(False, 9.0, -3.0) == 0.0 and abs(vws_delta(True, 2.9, -3.0) - 5.9) < 1e-9, "Δ 0 by construction when not stopped; shadow − actual when stopped")
    for bad_row, why in ((dict(stopped=False, delta=0.4), "non-stopped Δ ≠ 0"), (dict(stopped=True, final=True, shadow=1.0, actual=-3.0, delta=1.0), "Δ ≠ shadow − actual"),
                         (dict(stopped=True, final=True, shadow=None, actual=-3.0, delta=None), "final without a shadow"), (dict(stopped=True, excluded=True), "excluded without a reason")):
        try:
            vws_validate(bad_row)
            chk(False, f"validate must raise: {why}")
        except ValueError:
            chk(True, why)
    vws_validate(dict(stopped=True, final=True, shadow=1.0, actual=-3.0, delta=4.0)); vws_validate(dict(stopped=False, delta=0.0))
    vw = pd.DataFrame(dict(k=[f"2026-10-{8 + i // 10:02d}T{i % 10:02d}:00:00" for i in range(20)], pair="P", sleeve=["LONG", "WIDE"] * 10,
                           delta=[3.0, 1.0, -1.0, -0.5] * 5, final=True))
    vw2 = vw.assign(delta=[3.0, 1.0, -1.0, 0.5] * 5)
    chk(vws_check(vw2)[0] == "candidate", "Δ sum > 0 ∧ saved 15 > deeper 5 ∧ both sleeves > 0 ∧ top fill 17 % → candidate")
    chk(vws_check(vw)[0] == "close", "saved 10 = deeper 10 → close the idea")
    chk(vws_check(vw2.assign(sleeve=["LONG"] * 17 + ["WIDE"] * 3, delta=[1.0] * 17 + [-1.0] * 3))[0] == "candidate",
        "a sleeve with < 5 of the 20 fills does not carry the per-sleeve leg")
    chk(vws_check(vw2.assign(sleeve=["LONG"] * 15 + ["WIDE"] * 5, delta=[1.0] * 15 + [-1.0] * 5))[0] == "close", "a sleeve with ≥ 5 fills and Δ < 0 → close")
    chk(vws_check(vw2.assign(delta=[-0.1] * 19 + [10.0]))[0] == "close", "one fill carrying the gain (and deeper > saved) → close")
    chk(vws_check(vw2.head(19))[0] == "collecting" and vws_check(vw2.assign(final=[True] * 19 + [False]))[0] == "collecting", "< 20 or not final → collecting")
    chk(vws_check(pd.concat([vw2, vw.assign(k="2026-10-30T00:00:00")]))[0] == "candidate", "only the FIRST 20 by open time are read")
    print(f"selftest OK ({ok} checks)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
