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
  GVOL_BAND split (2026-10-07, observe-only; reports/FRENZY_GVOL_THRESHOLD_TEST_2026-10-07.md §6): each blocked row's market-volume reading
  (scout_gvol live registry first, else the frozen v2 cache; a v2 value freezes when the row is final) → a FROZEN band [1.0,1.1) [1.1,1.2)
  [1.2,1.5) [1.5,2.0) ≥ 2.0 (columns gvol / gvol_src / gvol_band / gvol_frozen); table per sleeve × band on the counted final rows + the
  frozen hypothesis "WIDE in [1.0, 1.2) is not a losing cohort" (≥ 20 signals on ≥ 10 days → propose WIDE-only 1.2 iff mean > 0 at day-block
  P ≥ 0.90 ∧ no day ≥ 50 % of the gain ∧ mean ≥ WIDE let-through − 0.30, else stays at 1.0). Never changes config.
TRACKER 9 — VWAP_STOP (2026-10-06, observe-only; reports/FRENZY_STAIRCASE_STUDY_2026-10-06.md §3b / §5 "BP k 0.5", NOT established on the
  year: saved 72 (+400) vs deeper 103 (−395)): every FRENZY / WIDE fill the live −3 stop closed (close_reason STOP_LOSS) re-priced as: identical
  to live until the stop, then held and out at the first print after a 5m close that is ≤ −3 % net AND below VWAP × (1 − 0.5 × entry ATR %),
  hard floor −12, the lock unchanged (rules off once armed), 12 h cap, 0.09 fees + 0.10 slip; ticks else 1m pseudo prints O → H → L → C.
  Δ = shadow − actual. Fills live did NOT stop: Δ 0 by construction, and the shadow's pre-stop replica is run on them — a replica −3 before
  their live exit is counted on the parity line (wick evidence). Rows are validated before save (bad rows → a timestamped .bad); non-final
  rows re-price from their stored fields; a stopped fill without a VWAP stamp / live P&L is stored once as excluded. FROZEN gate on the first
  20 stopped fills from GC_FROM: Δ sum > 0 ∧ saved > deeper ∧ Δ sum > 0 on every sleeve with ≥ 5 of the 20 (≥ 1 such sleeve, else collecting)
  ∧ no fill > 50 % of the gain → "candidate for a pre-registered study", else "close the idea". Rows: reports/SCOUT_FRENZY_VWAP_STOP.csv.
TRACKER 10 — ON_SCALP (2026-10-06, observe-only; the ONE line reports/FRENZY_ON_SCALP_STUDY_2026-10-06.md allows, PREREG
  reports/FRENZY_ON_SCALP_PREREG_2026-10-06.txt; the study verdict was "not a strategy, expected ≈ 0" — this only watches it forward).
  COUNTING UNIT = one per pair-EPISODE — a deliberate deviation from the study's per-BAR line (893 bars, +0.136 %/bar): same-episode bars are
  correlated draws of one pump; ONS_YEAR is re-derived per episode from the study's signal file on the same pricing (502 episodes, +0.105):
  every FRESH FRENZY ON bar, whatever the live refusal code (READY fills, GREEN_BAR, ATR_HIGH, gvol-blocked, VOL24_LOW, …), found from the
  journal's FRENZY lines AND the orders exports' FRENZY fills (a READY fill leaves no refusal line), each replayed with the engine's own
  functions (frenzy_walk on the last 1,499 closed 5m bars → frenzy_flagged → fresh_on on that bar; a catch-up line / fill is mapped to its
  ON bar through on_bar_ts). STRONG = frenzy_adx_delta(closed[-300:]) > 0 ∧ frenzy_di_spread(closed[-300:]) > 0 at the ON bar, exactly as
  _frenzy_open sizes strong (the study's S3). Pricing (frozen, the study's S3 TP3/T120/noSL): entry = the first trade print ≥ the ON close
  + 12 s, 0.09 % fees + 0.10 % slippage, out at the first print ≥ +3 % net (fill at that print), else at the first print ≥ entry + 2 h; NO
  stop. Ticks once the archive is out, else 1m pseudo prints open → low → high → close (the low before the high = conservative; the TP
  fills at exactly +3; ᵖ provisional). One per pair-episode (spikes ≤ 30 min apart merged), counted from GC_FROM; earlier = reference.
  FROZEN bar (the study's, every leg): N ≥ 30 fires on ≥ 15 days ∧ mean > 0 with day-block 95 % CI low > 0 ∧ P(+3 within 2 h) ≥ 70 % ∧
  forward max drawdown < 50 % of a $3k book at 0.2 sizing (sequential fixed fraction, notional 0.94 × equity, liquidation at −23.75 % price
  — the PREREG's book factors) → "candidate for a pre-registered probe study (no arming)", shown after the 30–50 % haircut. ADDITIONS
  (labelled): retire at mean ≤ 0 by 30 / no verdict by 60. Bars outside the study's universe (FRENZY_VOL24_LOW, blacklisted / non-ASCII
  pairs) are tagged and kept out of the bar. P&L finality alone decides a fire; the flow below fills in later.
  NEW INFORMATION (description only, no gate), stored on every ON row — strong rows in reports/SCOUT_FRENZY_ON_SCALP.csv, the non-strong
  ON bars as a control in reports/SCOUT_FRENZY_ON_SCALP_CONTROL.csv (same ruler): the PRE-ENTRY [close, +12 s) taker-buy share / $ volume /
  move, and the POST-ENTRY (descriptive) first 60 s after the ON close from public aggTrades
  (price, qty, isBuyerMaker; the tick cache keeps prices only, so: REST fapi/v1/aggTrades while the window is < 2 days old — the
  endpoint refuses older windows — else the daily aggTrades archive, only the 60-s slice kept): taker-buy share of $ volume, agg / raw trade count, $ volume vs the pair's median 1m $ volume over the prior 24 h and vs
  normal_hour_usd / 60, max drawdown / run-up vs the ON close, the +12 s print vs the close; and the journal BOOK snapshot (minute-level
  ob_* columns: imbalance at 0.25 / 0.5 / 1 / 2 %, walls, spread) closest to the close within 60 s after it, flagged minute-level.
  LIQUIDATIONS: NOT AVAILABLE — Binance publishes no public liquidation history (REST forceOrders returns only the caller's own account);
  capturing them needs a live recorder (the engine subscribing to the <symbol>@forceOrder stream for FRENZY-ON pairs). Not built.
TRACKER 11 — HYBRID_EXIT (2026-10-06, observe-only; V3 of reports/FRENZY_EXIT_LOCK_VS_BULLRUN_2026-10-06.md, PREREG
  reports/FRENZY_EXIT_LOCK_VS_BULLRUN_PREREG_2026-10-06.txt; the report's verdict: keep the lock, Δ −0.18 on the year): HYB = −3 stop; once
  the prior-print peak (net) reaches +1 the floor is +0.2; from a +3 peak the live lock max(+2, peak − 2) (the highest line wins); 12 h cap.
  Priced from the ACTUAL entry of every exit-table fill: the HYB column = the table's 1m ruler (next to LOCK2); the bar reads HYB − LOCK2 on
  the SAME prints (ticks once the archive is out, else 1m provisional), fees 0.09, no slippage on either side (the report charges 0.10 on
  both exits, so the Δ is unchanged). Own store reports/SCOUT_FRENZY_HYBRID.csv (validated before save; the exit rows' VER untouched).
  FROZEN re-open bar: ≥ 40 FRENZY_LONG fills opened from GC_FROM on ≥ 20 days ∧ Δ > +0.30 %/fill ∧ day-block 95 % CI low > 0 ∧ > 0 without
  the top 5 → re-open (a study, never an arm); otherwise "lock holds". Strong / normal split (stamped ADX Δ > 0 ∧ DI > 0) and SAVED (HYB ≥ 0,
  lock lost) / CUT (lock ≥ +2, HYB out at the +0.2 floor) counts shown.
TRACKER 12 — FRENZY_LITE watch + LITE_ATR (2026-10-07, DECISION_LOG 243; FRENZY_LITE shipped ARMED as a DECLARED EXCEPTION, no ATR filter,
  NO automatic off): every FRENZY_LITE fill in the orders exports (own store reports/SCOUT_FRENZY_LITE.csv, keyed opened_at + pair, so a fill
  survives its export leaving ~/Downloads). P&L = the bot's own closed pnl_percentage (no re-pricing). WATCH LINE (frozen): N closed · days ·
  WR · avg % · sum vs the review bar — REVIEW DUE at ≥ 40 closed fills on ≥ 15 days; avg < 0 at ≥ 20 closed fills → "⚠ FLAG FOR OPERATOR
  REVIEW" (a flag, never an auto-off). LITE_ATR (observe-only): the same fills split by the stamped entry ATR (entry_atr_pct = the engine's
  wilder_atr_pct on the signal window) ≤ 2.5 % vs > 2.5 % (frenzy_max_atr_pct at registration, FROZEN) — N · WR · avg · sum · days per side;
  the study's ATR > 2.5 cohort was −0.20 %/trade (87 % P(mean < 0), not 95 %), so this is the line that says whether LITE should have FRENZY's
  ATR cap. No ATR verdict is automatic — the split is read at the 40-fill review. Fills not in the export (or no ATR stamp) → "ATR ?" line.
TRACKER 13 — LITE_GVOL24_LOW (2026-10-07, operator-approved observe line; reports/FRENZY_LITE_N4_REGIME_2026-10-07.md §6): every CLOSED
  FRENZY_LITE fill gets the study's dashboard-style market volume (Σ EMA5 ÷ Σ SMA48 of base volume over the per-bar top-50 eligible pairs by
  24 h quote volume, closed 5m bars) averaged over the 24 h before the START of its 4 h UTC window; split HIGH > 0.996 / LOW ≤ 0.996 (FROZEN).
  Own write-once store reports/SCOUT_FRENZY_LITE_GVOL24.csv; public klines ≤ 60 s per run, stop on 418 / 429, unreadable → ≤ 3 runs.
  FROZEN bars: REVIEW (propose arming as a sleeve on/off switch — operator decision) when LOW ≥ 30 fills on ≥ 15 days ∧ LOW avg ≤ HIGH avg
  − 0.4 ∧ the LOW − HIGH day-block 95 % CI upper < 0; RETIRE when ≥ 30 LOW fills ∧ LOW − HIGH ≥ 0. Never changes trading.
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
                             FRENZY_WIDE_CODES, frenzy_wide_choppy, frenzy_wide_hold_green_block, frenzy_vol24_at)
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
        if kind == "HYB":                           # 🛟 V3: −3 · +0.2 floor from a +1 peak · the lock from +3 (hyb_line)
            line = hyb_line(pk)
        else:
            line = (max(2.0, pk - trail) if kind.startswith("LOCK") else 2.0) if armed else -3.0
        lpx = e * (1 + (line + FEE) / 100)
        if l <= lpx:
            return net(min(o, lpx)), t + MIN, (_hyb_how(line) if kind == "HYB" else "floor / trail" if armed else "stop")
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
            "entry_frenzy_above_share", "entry_frenzy_bar_ret_pct", "closed_at", "close_reason", "entry_frenzy_vwap", "entry_frenzy_spike_at",
            "entry_frenzy_adx_delta", "entry_frenzy_di_spread")
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
         "FIX3 fills at exactly ±3 (reference column). HYB = the hybrid V3 (−3 · +0.2 floor from a +1 peak · the lock from +3; HYBRID_EXIT line below). "
         "ᵖ = not final yet.", ""]
    if not len(allr):
        return L + ["No FRENZY / WIDE fill in the exports yet.", ""] + _extras(now_ms, th, F, J, allr)
    hmap, hlines = _hyb_safe(now_ms, F, allr)   # 🛟 HYBRID_EXIT: its own store; the exit rows / VER are untouched
    show = allr.tail(15)
    L += ["| Opened UTC | Pair | Sleeve | ATR | ATR Δ30m | vs avg | Actual | LOCK2 | HYB | LOCK3 | EMA20 | EMA50 | FIX3 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.2f}"
    for r in show.itertuples():
        fl = ("" if str(r.final) in ("True", "1", "1.0") else "ᵖ") + ("" if r.first else "²")
        L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{fl} | {str(r.pair).replace('USDT', '')} | {r.sleeve} | {f(r.atr_entry)} | {f(getattr(r, 'atr_chg30', None))} | "
                 f"{f(r.vs_vwap)} | {f(r.actual)} | {f(r.LOCK2)} | {f(hmap.get((str(r.k), str(r.pair)), (None,))[0])} | {f(r.LOCK3)} | {f(r.EMA20)} | {f(r.EMA50)} | {f(r.FIX3)} |")
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
    L += hlines
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
# 🌊 GVOL_BAND split (2026-10-07, observe-only; reports/FRENZY_GVOL_THRESHOLD_TEST_2026-10-07.md §6): every COUNTED blocked signal gets a FROZEN
# market-volume band from its signal bar's reading — the bot's own value (scout_gvol live registry: fill stamp / gate log line, keyed by the
# signal close) preferred, else the scout's frozen v2 value (scout_gvol cache, keyed by the bar open). A live value freezes at once; a scout-v2
# value freezes when the row turns final (a late server log can still supply the bot's reading meanwhile). A frozen value is never changed.
GVB_BANDS = ((1.0, 1.1), (1.1, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, float("inf")))   # FROZEN
GVB_HYP_LO, GVB_HYP_HI = 1.0, 1.2              # pre-registered hypothesis band (WIDE only): "WIDE in [1.0, 1.2) is not a losing cohort"
GVB_HYP_N, GVB_HYP_DAYS, GVB_HYP_P, GVB_HYP_DAY_MAX, GVB_HYP_MARGIN = 20, 10, 0.90, 50.0, 0.30   # FROZEN read / propose legs
GVB_HYP_BOOT, GVB_HYP_SEED = 4000, 7           # day-block bootstrap (fixed seed, ≥ 2,000 resamples)
GVB_HYP_UNREAD_MAX = 0.25                      # review: the read is HELD while > 25 % of the counted final WIDE rows have no band (coverage)
GVB_LITE = {"FRENZY_LITE_GVOL_HIGH": "LITE", "FRENZY_LITE_GVOL_UNREAD": "LITE"}   # LITE uses the same gate (engine f"{_bk}_GVOL_HIGH")
GVB_BAND_YEAR = ("year (study 2026-10-07, 588 signals, live lock on ticks): WIDE 1.0–1.1 +1.50 %/trade N 21 · FRENZY 1.0–1.1 −0.86 % N 24 · "
                 "overall the 1.0 line is the best of ten (book +349 %)")
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
# GVOL_BLOCKED ≤ 90 + 80 s, VWAP_STOP ≤ 90 + 80 s, ON_SCALP ≤ 90 + 160 s (a redirected replay + 1m + 24 h base + aggTrades pages, each ≤ 20 s;
# the aggTrades pager also stops at the deadline), HYBRID_EXIT ≤ 90 + 80 s (before the exit table) ≈ 16 min absolute worst; the 10-06 dry-run took ~77 s for the module (trackers 7 – 9) with 8 downloads.


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


def gvb_band(v):
    """market-volume value → its frozen band label ('[1.0, 1.1)' … '≥ 2.0'); < 1.0 (the reading disagrees with the live block) → '< 1.0'; None → None."""
    v = _num(v)
    if v is None:
        return None
    for lo, hi in GVB_BANDS:
        if lo <= v < hi:
            return f"≥ {lo:.1f}" if hi == float("inf") else f"[{lo:.1f}, {hi:.1f})"
    return "< 1.0"


def gvb_freeze_gvol(df, live, scout):
    """fill / freeze gvol · gvol_src · gvol_band · gvol_frozen on the BLOCKED rows (not the let-through ones). live = {close_ms: (value, src, pair)}
    (scout_gvol.live_map), scout = {open_ms: value} (scout_gvol.cached). A frozen row is never touched; a live value freezes at once; a scout
    value is provisional until the row is final, then frozen. Returns a copy."""
    d = df.copy()
    for c in ("gvol", "gvol_src", "gvol_band", "gvol_frozen"):
        if c not in d:
            d[c] = None
    d["gvol"] = d["gvol"].astype(object); d["gvol_src"] = d["gvol_src"].astype(object)
    d["gvol_band"] = d["gvol_band"].astype(object); d["gvol_frozen"] = d["gvol_frozen"].astype(object)
    for i in d.index[d.gate.astype(str) != GVB_PASSED]:
        if str(d.at[i, "gvol_frozen"]) in _TRUE:
            continue
        try:
            close = _ms(d.at[i, "k"])
        except Exception:
            continue
        lv = (live or {}).get(close)
        if lv:
            v, src, fz = float(lv[0]), str(lv[1]), True
        else:
            sv = _num((scout or {}).get(close - BAR))
            if sv is None:                             # a miss never wipes a provisional value (it freezes once the row is final)
                pv = _num(d.at[i, "gvol"])
                if pv is None:
                    d.at[i, "gvol_frozen"] = False
                    continue
                sv = pv
            v, src, fz = float(sv), "scout v2", str(d.at[i, "final"]) in _TRUE
        d.at[i, "gvol"], d.at[i, "gvol_src"], d.at[i, "gvol_band"], d.at[i, "gvol_frozen"] = round(v, 4), src, gvb_band(v), fz
    return d


def gvb_lite_bands(J, live, scout, since=None):
    """FRENZY_LITE market-volume refusals (journal BLOCK lines FRENZY_LITE_GVOL_HIGH / _UNREAD from `since`, one per (t, pair, gate)) → DataFrame
    (t, pair, gate, gvol, band). The GVOL_BLOCKED replay cannot price LITE (no LITE replay in gvb_take) → counts per band only. Not separately
    frozen: recomputed each run from the persisted journal (JR_CSV) and scout_gvol's own frozen stores."""
    cols = ["t", "pair", "gate", "gvol", "band"]
    if J is None or not len(J):
        return pd.DataFrame(columns=cols)
    b = J[(J.e == "BLOCK") & J.gate.isin(list(GVB_LITE))].drop_duplicates(["t", "pair", "gate"])
    if since:
        b = b[b.t.astype(str) >= since]
    rows = []
    for r in b.itertuples():
        try:
            close = _ms(r.t) // BAR * BAR
        except Exception:
            continue
        lv = (live or {}).get(close)
        v = float(lv[0]) if lv else _num((scout or {}).get(close - BAR))
        rows.append(dict(t=r.t, pair=r.pair, gate=r.gate, gvol=v, band=(gvb_band(v) if v is not None else "unread")))
    return pd.DataFrame(rows, columns=cols)


def gvb_coverage(cnt):
    """(read, total) over the counted final rows, and the unread share among the counted final WIDE rows (the hypothesis' eligible pool)."""
    g = pd.to_numeric(cnt.get("gvol", pd.Series(np.nan, index=cnt.index)), errors="coerce")
    w = cnt.sleeve == "WIDE"
    return int(g.notna().sum()), int(len(cnt)), (float(g[w].isna().mean()) if w.any() else 0.0)


def gvb_band_table(cnt):
    """counted + final blocked rows → markdown rows per sleeve × band: N, days, WR, avg lock %, sum."""
    L = ["| Sleeve | Band | N | Days | WR | Avg lock % | Σ % |", "|---|---|---|---|---|---|---|"]
    labs = [gvb_band(lo) for lo, _ in GVB_BANDS] + ["< 1.0", "unread"]
    for sl, nm in (("LONG", "FRENZY_LONG"), ("WIDE", "FRENZY_WIDE")):
        g = cnt[cnt.sleeve == sl]
        b = g.gvol_band.where(g.gvol_band.notna(), "unread").astype(str) if "gvol_band" in g else pd.Series("unread", index=g.index)
        for lab in labs:
            x = pd.to_numeric(g[b == lab].LOCK, errors="coerce").dropna()
            if not len(x) and lab in ("< 1.0", "unread"):
                continue
            dd = g.loc[x.index, "day"].nunique() if len(x) else 0
            L.append(f"| {nm} | {lab} | {len(x)} | {dd} | " + (f"{(x > 0).mean() * 100:.0f} % | {x.mean():+.2f} | {x.sum():+.2f} |" if len(x) else "– | – | – |"))
    return L


def gvb_wide_hyp(x, days, lt_mean):
    """the FROZEN WIDE [1.0, 1.2) hypothesis on the counted, final WIDE band signals' lock % (x) / UTC days; lt_mean = the counted WIDE let-through
    mean on the same ruler. → (state, text): collecting / propose / stays."""
    x = pd.Series(np.asarray(x, dtype=float)); days = pd.Series(list(days), index=x.index)
    ok_ = x.notna(); x, days = x[ok_], days[ok_]
    n, nd = len(x), days.nunique()
    t = f"{n}/{GVB_HYP_N} WIDE signals · {nd}/{GVB_HYP_DAYS} days" + (f" · WR {(x > 0).mean() * 100:.0f} % · mean {x.mean():+.2f} %" if n else "")
    lt = None if lt_mean is None or (isinstance(lt_mean, float) and np.isnan(lt_mean)) else float(lt_mean)
    t += f" · WIDE let-through mean {lt:+.2f} %" if lt is not None else " · WIDE let-through mean –"
    if n < GVB_HYP_N or nd < GVB_HYP_DAYS:
        return "collecting", t
    g = pd.DataFrame(dict(x=x.values, d=days.values)).groupby("d").x.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    r = np.random.default_rng(GVB_HYP_SEED).integers(0, len(g), (GVB_HYP_BOOT, len(g)))
    p = float(((s[r].sum(1) / c[r].sum(1)) > 0).mean())
    net = float(x.sum())
    share = float(g["sum"].max() / net * 100) if net > 0 else float("inf")
    t += f" · day-block P(mean > 0) {p:.2f} · top day {('–' if share == float('inf') else f'{share:.0f} %')} of the gain"
    if lt is None:
        return "stays", t + " · stays at 1.0 (no let-through baseline)"
    legs = (x.mean() > 0 and p >= GVB_HYP_P, share < GVB_HYP_DAY_MAX, x.mean() >= lt - GVB_HYP_MARGIN)
    return ("propose" if all(legs) else "stays"), t


def gvb_hyp_hold(state, unread_share):
    """coverage hold (review): while > GVB_HYP_UNREAD_MAX of the counted final WIDE rows have no band, a read is not made."""
    return "collecting (coverage)" if state != "collecting" and unread_share > GVB_HYP_UNREAD_MAX else state


def _gvb_band_lines(cnt, pcs, prov, lite=None):
    """the GVOL_BAND sub-section of GVOL_BLOCKED: sleeve × frozen band table + coverage + the frozen WIDE [1.0, 1.2) hypothesis line + LITE counts."""
    L = ["", f"**🌊 GVOL_BAND split (observe-only, registered 2026-10-07; never changes config):** the counted, final blocked signals above by the "
              "FROZEN market-volume band of their signal bar (the bot's own reading — fill stamp / gate log — else the scout's frozen v2 value; "
              "frozen once, never re-read). Same ruler (lock %)." + (f" Provisional counted rows not in the table yet: {len(prov)}." if len(prov) else ""), ""]
    L += gvb_band_table(cnt)
    rd, tot, wun = gvb_coverage(cnt)
    L += ["", f"Coverage: band read for {rd} of {tot} counted final rows · WIDE rows without a band {wun * 100:.0f} % (the hypothesis read is held "
              f"while > {GVB_HYP_UNREAD_MAX * 100:.0f} %)."]
    w = cnt[(cnt.sleeve == "WIDE") & pd.to_numeric(cnt.get("gvol", pd.Series(dtype=float)), errors="coerce").between(GVB_HYP_LO, GVB_HYP_HI, inclusive="left")]
    lt = pd.to_numeric(pcs[pcs.sleeve == "WIDE"].LOCK, errors="coerce").dropna()
    st, tx = gvb_wide_hyp(pd.to_numeric(w.LOCK, errors="coerce").values, w.day.values, lt.mean() if len(lt) else None)
    st = gvb_hyp_hold(st, wun)
    lab = {"collecting": "⏳ collecting", "collecting (coverage)": f"⏳ collecting (coverage — > {GVB_HYP_UNREAD_MAX * 100:.0f} % of WIDE rows unbanded)", "propose": "📋 PROPOSE raising the gate for WIDE alone to 1.2 (operator decision — no auto-change)",
           "stays": "➖ WIDE stays at 1.0"}
    L += ["", f"**Pre-registered hypothesis (FROZEN 2026-10-07): \"WIDE in [{GVB_HYP_LO:.1f}, {GVB_HYP_HI:.1f}) is not a losing cohort\"** — read only at ≥ "
              f"{GVB_HYP_N} counted WIDE signals in the band spanning ≥ {GVB_HYP_DAYS} days; PROPOSE (operator decision) raising WIDE's gate to "
              f"{GVB_HYP_HI:.1f} iff band mean > 0 with day-block bootstrap P ≥ {GVB_HYP_P:.2f} (seed {GVB_HYP_SEED}, {GVB_HYP_BOOT:,} resamples) ∧ "
              f"no single day ≥ {GVB_HYP_DAY_MAX:.0f} % of the band's gain ∧ band mean ≥ the WIDE let-through mean (same ruler) − {GVB_HYP_MARGIN:.2f}; "
              f"otherwise WIDE stays at 1.0. → " + lab.get(st, st) + f" ({tx})",
          f"- Band year reference: {GVB_BAND_YEAR}."]
    if lite is not None:
        hi = lite[lite.gate == "FRENZY_LITE_GVOL_HIGH"]
        bc = hi.band.value_counts()
        order = [gvb_band(lo) for lo, _ in GVB_BANDS] + ["< 1.0", "unread"]
        L.append(f"- FRENZY_LITE (same gate; the GVOL_BLOCKED replay cannot price LITE refusals → counts per band only, from {GC_FROM[:10]}): "
                 f"{len(hi)} GVOL_HIGH refusals" + (" — " + " · ".join(f"{b} {int(bc[b])}" for b in order if b in bc) if len(hi) else "")
                 + f" · {int((lite.gate == 'FRENZY_LITE_GVOL_UNREAD').sum())} UNREAD.")
    return L


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
    try:                                                       # 🌊 GVOL_BAND sources (read-only, no network): live registry + frozen v2 cache
        import scout_gvol as _SG
        _lv, _sc = _SG.live_map(_SG.load_live()), _SG.cached()
    except Exception:
        _lv, _sc = {}, {}
    allg = pd.concat([old, new], ignore_index=True) if len(new) else old.copy()
    if len(allg):
        allg = allg.drop_duplicates(["k", "pair", "gate"], keep="last").sort_values(["k", "pair", "gate"], kind="stable").reset_index(drop=True)
        if acts and "actual" in allg:                          # a let-through fill that closed since it was priced: its live actual (display only)
            pa = allg.gate == GVB_PASSED
            na = pd.Series([acts.get((k, p)) for k, p in zip(allg.k, allg.pair)], index=allg.index, dtype=object)
            allg.loc[pa & na.notna(), "actual"] = na[pa & na.notna()].astype(float)
        M = gvb_masks(allg)
        allg["episode"], allg["counted"], allg["double"], allg["passed_counted"] = M["episode"], M["counted"], M["double"], M["passed"]
        try:                                                   # 🌊 GVOL_BAND: the bar's reading frozen per row; a failure keeps the priced rows
            allg = gvb_freeze_gvol(allg, _lv, _sc)
        except Exception as _fe:
            print(f"[scout] GVOL_BAND freeze failed ({str(_fe)[:120]}) — rows kept unbanded this run", file=sys.stderr)
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
    L += _gvb_band_lines(cnt, pcs, allg[S_("counted") & ~S_("final")], gvb_lite_bands(J, _lv, _sc, GC_FROM))
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


# ─────────────────────────── ⚡ tracker 10 (ON_SCALP) — 2026-10-06, observe-only ───────────────────────────
ONS_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_ON_SCALP.csv")            # strong ON bars (the cohort + reference rows)
ONS_CTRL_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_ON_SCALP_CONTROL.csv")  # non-strong ON bars (control, same ruler) + replay-parity notes
ONS_VER = 1
# FROZEN — reports/FRENZY_ON_SCALP_STUDY_2026-10-06.md, "Recommendation" (the only observe line consistent with the PREREG): "ON-scalp S3:
# fresh ON bar ∧ adx_delta > 0 ∧ di_spread > 0, entry first print ≥ close + 12 s, TP +3 net / 2 h time stop / no stop". Net = the PREREG's
# scale: (price / entry − 1) × 100 − 0.19 (0.09 fees + 0.10 slippage); TP and time fills = that print (PREREG Part 2 (a/b); 1m: the TP at X).
ONS_TP, ONS_T_MIN, ONS_COST = 3.0, 120, FEE + SLIP
# FROZEN bar = the study's own (Recommendation): "N ≥ 30 fresh fires on ≥ 15 days, mean > 0 with day-CI low > 0, P(+3 within 2 h) ≥ 70 %, and a
# forward book DD < 50 % at 0.2. Then apply the 30–50 % haircut." → candidate for a pre-registered probe study (no arming). ADDITIONS beyond
# the study (labelled in the output): retire when the mean ≤ 0 at ≥ 30, or with no verdict by 60.
ONS_N, ONS_DAYS, ONS_RETIRE_N, ONS_DD_MAX, ONS_P3 = 30, 15, 60, 50.0, 70.0
ONS_HAIRCUT = (0.30, 0.50)
ONS_BOOK0, ONS_NOTIONAL, ONS_LIQ = 3000.0, 0.94, 23.75   # PREREG "Book": lev 0.2 → notional 0.94 × equity, liquidation cap −23.75 % price
ONS_FLOW_MS, ONS_PRE_MS, ONS_FLOW_PAGES, ONS_FLOW_GIVEUP_D, ONS_BOOK_LAG_S = 60_000, ENTRY_LAG_MS, 200, 30, 60
#   the "new information" windows: [close, close + 60 s) post-close and [close, close + 12 s) pre-entry; the REST pager is bounded by the
#   tracker's deadline (200 pages = 200k agg trades); a truncated read is never stored as ok. Only an archive 404 more than 30 days after
#   the close is "missing"; every other read / parse failure stays "pending" and is retried.
ONS_REST_MAX_AGE_MS = 46 * H   # fapi/v1/aggTrades with a time window answers only the recent 2 days (error −4166) → older: the daily archive
# COUNTING UNIT = one per pair-episode (deliberate deviation from the study's per-BAR S3 line, 893 bars · +0.136: same-episode bars are
# correlated draws of one pump). Re-derived from reports/FRENZY_ON_SCALP_SIGNALS_2026-10-06.csv on the same pricing (S3, TP3/T120/noSL =
# +3 when tX3 ≤ 120 min else at120; first strong bar per pair-episode, spike = signal − hours, 30-min merge); the book is approximate (MAE at
# 2 h on the misses only).
ONS_YEAR = ("year per pair-EPISODE (re-derived from the study's signal file, S3 strong, TP3/T120/noSL): 502 episodes on 229 days · WR 70 % · "
            "mean +0.105 %/episode · day CI [−0.33, +0.51] · P(+3 within 2 h) 67.5 % (bar leg ≥ 70 %) · book at 0.2 ≈ $3k → $2.7k, max DD ≈ 58 % "
            "(approx.) — per bar the study read 893 bars · +0.136 / +0.15; it expects ≈ 0 forward")
ONS_LIQ_NOTE = ("not available — Binance has no public liquidation history (REST forceOrders is the caller's own account); a live recorder "
                "(the engine subscribing to <symbol>@forceOrder for FRENZY-ON pairs) would be needed — not built")
ONS_FLOW_COLS = ("flow_state", "px12_vs_close", "buy_share_60", "n_agg_60", "n_trades_60", "usd_60", "base_1m_usd", "vol_x_24h", "vol_x_norm",
                 "mdd_60", "mru_60", "buy_share_pre", "usd_pre", "move_pre", "n_agg_pre", "flow_trunc", "flow_src")
ONS_BOOK_COLS = ("ob_mid", "ob_spread_pct", "ob_imb_025", "ob_imb_05", "ob_imb_1", "ob_imb_2", "ob_bid_usd_1", "ob_ask_usd_1",
                 "ob_wall_bid_dist_pct", "ob_wall_bid_usd", "ob_wall_bid_share", "ob_wall_ask_dist_pct", "ob_wall_ask_usd", "ob_wall_ask_share")
ONS_REPLAY_COLS = ("spike_at", "hours", "above_streak", "atr", "vs_vwap", "bar_ret", "vol_mult", "replay_code", "adx_delta", "di_spread",
                   "strong", "on_close", "nh", "parity", "parity_why")
_ARCH = {}   # per-run archive cache: (pair, date) → a temp zip path or a terminal state; emptied (files deleted) at the end of ons_run


def m1_prints_lh(m1):
    """1m rows [open_ms, o, h, l, c, …] → pseudo prints o (:00) → l (:15) → h (:30) → c (:59.999): the low BEFORE the high — the conservative
    order for a no-stop TP-or-time exit (a minute that dips and pops is never credited the pop first). → (t int64, p float)."""
    if not m1:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    a = np.asarray([r[:5] for r in m1], dtype=float)
    t = (a[:, :1].astype(np.int64) + np.array([0, 15_000, 30_000, 59_999], dtype=np.int64)).ravel()
    return t, a[:, [1, 3, 2, 4]].ravel()


def onscalp_walk(tt, pp, e, t_e, gap=False):
    """⚡ the frozen ON-scalp exit on prints from t_e: net = (p / e − 1) × 100 − 0.19; out at the first print with net ≥ +3 (fill = that print;
    gap=True → 1m pseudo prints, the TP fills at exactly +3), else at the first print ≥ t_e + 2 h (fill = that print); no stop. MFE / MAE on
    the prints from the entry to the exit (inclusive). → dict(pnl, exit_ms, how ('TP +3' / '2 h' / 'open' / 'no data'), mfe, mae, hit)."""
    tt = np.asarray(tt, dtype=np.int64); pp = np.asarray(pp, dtype=float)
    m = tt >= int(t_e)
    tt, pp = tt[m], pp[m]
    if not len(pp) or not e:
        return dict(pnl=None, exit_ms=None, how="no data", mfe=None, mae=None, hit=False)
    net = (pp / float(e) - 1) * 100 - ONS_COST
    t_end = int(t_e) + ONS_T_MIN * MIN
    n = len(net)
    i_tp = next(iter(np.flatnonzero((net >= ONS_TP) & (tt < t_end))), n)
    i_t = next(iter(np.flatnonzero(tt >= t_end)), n)
    i = min(i_tp, i_t)
    if i >= n:
        return dict(pnl=float(net[-1]), exit_ms=int(tt[-1]), how="open", mfe=float(net.max()), mae=float(net.min()), hit=False)
    hit = i == i_tp
    pnl = (ONS_TP if gap else float(net[i])) if hit else float(net[i])
    seg = net[:i + 1]
    return dict(pnl=pnl, exit_ms=int(tt[i]), how=("TP +3" if hit else "2 h"), mfe=float(max(seg.max(), pnl)), mae=float(min(seg.min(), pnl)), hit=bool(hit))


def onscalp_flow(T, p, q, m, nraw, t_close, close_px, base_1m_usd, nh):
    """⚡ aggTrades around the ON close (T ms, price, qty, isBuyerMaker, raw trades per agg row) — description only. POST-CLOSE [close, +60 s):
    taker-buy share of $ (isBuyerMaker False = the buyer took), agg / raw trade count, $ volume vs the pair's median 1m $ volume (prior 24 h)
    and vs normal_hour_usd / 60, max drawdown / run-up vs the ON close (price %), the first print ≥ close + 12 s vs the close. PRE-ENTRY
    [close, +12 s): taker-buy share of $, $ volume, agg count and the move (the last print before +12 s vs the close). Pure."""
    T = np.asarray(T, dtype=np.int64); p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    m = np.asarray(m, dtype=bool); nraw = np.asarray(nraw, dtype=float)
    w = (T >= int(t_close)) & (T < int(t_close) + ONS_FLOW_MS)
    pre = (T >= int(t_close)) & (T < int(t_close) + ONS_PRE_MS)
    usd = p * q
    tot, tpre = float(usd[w].sum()), float(usd[pre].sum())
    c = float(close_px) if close_px else None
    i12 = int(np.searchsorted(T, int(t_close) + ENTRY_LAG_MS, side="left"))
    px12 = float(p[i12]) if i12 < len(T) and T[i12] < int(t_close) + ONS_FLOW_MS else None
    lastpre = float(p[np.flatnonzero(pre)[-1]]) if pre.any() else None
    return dict(n_agg_60=int(w.sum()), n_trades_60=int(nraw[w].sum()), usd_60=tot,
                buy_share_60=(float(usd[w & ~m].sum()) / tot if tot > 0 else None),
                base_1m_usd=(float(base_1m_usd) if base_1m_usd else None),
                vol_x_24h=(tot / float(base_1m_usd) if base_1m_usd else None),
                vol_x_norm=(tot / (float(nh) / 60) if nh else None),
                mdd_60=((float(p[w].min()) / c - 1) * 100 if c and w.any() else None),
                mru_60=((float(p[w].max()) / c - 1) * 100 if c and w.any() else None),
                px12_vs_close=((px12 / c - 1) * 100 if c and px12 is not None else None),
                n_agg_pre=int(pre.sum()), usd_pre=tpre, buy_share_pre=(float(usd[pre & ~m].sum()) / tpre if tpre > 0 else None),
                move_pre=((lastpre / c - 1) * 100 if c and lastpre is not None else None))


def onscalp_book(w):
    """the bar's ruin leg: a $3k book, one fill at a time in entry order, equity × (1 + 0.94 × r / 100) — r = the fill's net %, or the
    liquidation (−23.75 % price − 0.19 costs) when its MAE crossed it before the exit. → (end equity, max drawdown % of the running peak)."""
    eq = peak = ONS_BOOK0; dd = 0.0
    liq = -(ONS_LIQ + ONS_COST)
    for r, mae in zip(pd.to_numeric(w.pnl, errors="coerce"), pd.to_numeric(w.mae, errors="coerce")):
        if pd.isna(r):
            continue
        rr = liq if (pd.notna(mae) and mae <= liq) else float(r)
        eq *= max(0.0, 1 + ONS_NOTIONAL * rr / 100)
        peak = max(peak, eq)
        dd = max(dd, (1 - eq / peak) * 100 if peak > 0 else 100.0)
    return eq, dd


def onscalp_check(w):
    """the FROZEN ON_SCALP bar (the study's) on counted, final strong fires (pnl, mae, hit, day, entry order). → (state, text)."""
    w = w.assign(_x=pd.to_numeric(w.pnl, errors="coerce"))
    w = w[w._x.notna()].sort_values(["entry_at", "pair"], kind="stable") if "entry_at" in w else w[w._x.notna()]
    x = w._x
    n, nd = len(x), w.day.nunique()
    t = f"{n}/{ONS_N} fires · {nd}/{ONS_DAYS} days"
    if not n:
        return "collecting", t
    ci = day_ci(x.values, w.day.values)
    eq, dd = onscalp_book(w)
    p3 = float(w.hit.astype(str).isin(_TRUE).mean() * 100) if "hit" in w else 0.0
    t += (f" · WR {(x > 0).mean() * 100:.0f} % · mean {x.mean():+.2f} %" + (f" · day CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
          + f" · P(+3 within 2 h) {p3:.0f} % (≥ {ONS_P3:g}) · book at 0.2 ${ONS_BOOK0:,.0f} → ${eq:,.0f}, max DD {dd:.0f} % (< {ONS_DD_MAX:.0f})")
    if n >= ONS_N and x.mean() <= 0:
        return "retire", t + f" · mean ≤ 0 at ≥ {ONS_N} (addition)"
    if n >= ONS_N and nd >= ONS_DAYS and x.mean() > 0 and ci and ci[0] > 0 and p3 >= ONS_P3 and dd < ONS_DD_MAX:
        return "candidate", t + f" · after the {ONS_HAIRCUT[0] * 100:.0f}–{ONS_HAIRCUT[1] * 100:.0f} % haircut {x.mean() * (1 - ONS_HAIRCUT[1]):+.2f} … {x.mean() * (1 - ONS_HAIRCUT[0]):+.2f}"
    if n >= ONS_RETIRE_N:
        return "retire", t + f" · no verdict by {ONS_RETIRE_N} (addition)"
    return "collecting", t


def onscalp_universe(pair, code, gates, th):
    """the study's universe (PREREG: FRENZY_VOL24_LOW bars and blacklisted pairs were NOT in it; Alpha / < 90-day / new listings are screened by
    the live engine before it judges a pair, so every journal / fill candidate already passed them) → 'ok' or the reason it is out of the bar."""
    if "FRENZY_VOL24_LOW" in (str(code), *str(gates or "").split(";")):
        return "VOL24_LOW"
    if not str(pair).isascii():
        return "NON_ASCII"
    bl = set()
    for nm in ("pair_blacklist", "no_trade_pairs", "frenzy_pair_blacklist"):
        bl |= {x.strip().upper() for x in str(getattr(th, nm, "") or "").split(",") if x.strip()}
    return "BLACKLIST" if str(pair).upper() in bl else "ok"


def onscalp_first(df, mask):
    """the first row (by ON close, stable on pair) per pair-episode among mask — the floor / filters are applied BEFORE the dedupe; a bar
    with no spike (a live-only NO_EPISODE bar) is its own episode. → bool Series."""
    out = pd.Series(False, index=df.index)
    if not mask.any():
        return out
    ep = episode_keys(df)
    ep = pd.Series([e if isinstance(e, str) else f"{p}|bar:{k}" for e, p, k in zip(ep, df.pair, df.k)], index=df.index)
    g = df[mask].assign(_ep=ep[mask]).sort_values(["k", "pair"], kind="stable")
    out.loc[g.drop_duplicates("_ep").index] = True
    return out


def _ons_masks(df):
    s = lambda c: df[c].astype(str).isin(_TRUE) if c in df else pd.Series(False, index=df.index)
    uni = (df["univ"].astype(str) == "ok") if "univ" in df else pd.Series(True, index=df.index)
    base = s("is_on") & uni
    return dict(counted=onscalp_first(df, base & s("cohort") & s("strong")), ref=onscalp_first(df, base & ~s("cohort") & s("strong")),
                control=onscalp_first(df, base & s("cohort") & ~s("strong")))


def onscalp_counted(df):
    """the bar's cohort: strong ON bars from GC_FROM inside the study's universe → the first per pair-episode. → bool Series."""
    return _ons_masks(df)["counted"]


def onscalp_validate(r, strong_file):
    """raises ValueError on a row that must not be saved: the wrong file for its class, a strong flag that disagrees with its ADX / DI,
    a final row without a clean exit (a stale row without any exit is final with exit 'no exit (stale)' and no P&L), a TP below +3 or at /
    after 2 h, a time exit before 2 h, MAE / MFE out of order, a truncated or out-of-range flow stored as ok."""
    T = lambda v: str(v) in _TRUE
    if not T(r.get("is_on")):
        if strong_file:
            raise ValueError("a non-ON row in the strong file")
        if not isinstance(r.get("not_on_reason"), str) or not r.get("not_on_reason"):
            raise ValueError("a non-ON row without its reason")
        return
    ad, di = _num(r.get("adx_delta")), _num(r.get("di_spread"))
    want = bool(ad is not None and di is not None and ad > 0 and di > 0)
    if T(r.get("strong")) != want:
        raise ValueError(f"strong {r.get('strong')} ≠ ADXΔ {ad} > 0 ∧ DI {di} > 0")
    if want != bool(strong_file):
        raise ValueError("a row in the wrong file for its strong flag")
    if T(r.get("final")) and r.get("exit_how") != "no exit (stale)":
        pnl, how = _num(r.get("pnl")), r.get("exit_how")
        if pnl is None or how not in ("TP +3", "2 h") or not isinstance(r.get("exit_at"), str) or not isinstance(r.get("entry_at"), str):
            raise ValueError("final row without a clean exit")
        if how == "TP +3" and pnl < ONS_TP - 1e-9:
            raise ValueError(f"TP exit at {pnl} < +3")
        held = _ms(r["exit_at"]) - _ms(r["entry_at"])
        if (how == "TP +3" and held >= ONS_T_MIN * MIN) or (how == "2 h" and held < ONS_T_MIN * MIN):
            raise ValueError(f"{how} exit after {held / MIN:.1f} min (the TP must come before 2 h, the time exit at / after it)")
        mae, mfe = _num(r.get("mae")), _num(r.get("mfe"))
        if mae is None or mfe is None or not (mae <= pnl + 1e-9 <= mfe + 2e-9):
            raise ValueError(f"MAE {mae} ≤ pnl {pnl} ≤ MFE {mfe} violated")
    if str(r.get("flow_state")) == "ok":
        if T(r.get("flow_trunc")):
            raise ValueError("a truncated flow stored as ok")
        b, u = _num(r.get("buy_share_60")), _num(r.get("usd_60"))
        lo, hi = _num(r.get("mdd_60")), _num(r.get("mru_60"))
        bp = _num(r.get("buy_share_pre"))
        if (u is None or u < 0 or (b is not None and not 0 <= b <= 1) or (bp is not None and not 0 <= bp <= 1)
                or (lo is not None and hi is not None and lo > hi + 1e-12)):
            raise ValueError("flow reading out of range")


def _quarantine(df, bad, path):
    """bad = [(index, reason)] → write the rows to a timestamped .bad ONCE per row key (k|pair; keys remembered in <path>.bad_keys.json), so a
    row failing on every run never spawns a new file each run. → (name of the .bad written or None, n new keys)."""
    side = path + ".bad_keys.json"
    try:
        seen = set(json.load(open(side)))
    except Exception:
        seen = set()
    newi = [(i, m_) for i, m_ in bad if f"{df.at[i, 'k']}|{df.at[i, 'pair']}" not in seen]
    if not newi:
        return None, 0
    qp = _bad_path(path)
    try:
        df.loc[[i for i, _ in newi]].assign(bad_reason=[m_ for _, m_ in newi]).to_csv(qp, index=False)
        seen |= {f"{df.at[i, 'k']}|{df.at[i, 'pair']}" for i, _ in newi}
        tmp = f"{side}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(sorted(seen), fh)
        os.replace(tmp, side)
    except Exception:
        qp = "(write failed)"
    return os.path.basename(qp), len(newi)


def _aggtrades_rest(sym, t0, t1, deadline):
    """public aggTrades in [t0, t1] from fapi/v1/aggTrades (the archive's stream: a, p, q, f, l, T, m), paged by fromId (≤ ONS_FLOW_PAGES
    pages of 1,000; stops at the deadline → 'pending'). → ('ok', T, p, q, m, nraw, truncated) — truncated = the page cap ended the read."""
    rows, url = [], "https://fapi.binance.com/fapi/v1/aggTrades?"
    q = dict(symbol=sym, startTime=int(t0), endTime=int(t1), limit=1000)
    trunc = False
    for pg in range(ONS_FLOW_PAGES):
        if time.monotonic() > deadline:
            return ("pending",) + (None,) * 6
        r = json.loads(urllib.request.urlopen(url + urllib.parse.urlencode(q), timeout=20).read())
        rows += [x for x in r if int(x["T"]) <= int(t1)]
        if len(r) < 1000 or int(r[-1]["T"]) > int(t1):
            break
        q = dict(symbol=sym, fromId=int(r[-1]["a"]) + 1, limit=1000)
        trunc = pg == ONS_FLOW_PAGES - 1
        time.sleep(0.05)
    if not rows:
        return "ok", *(np.array([], dtype=t) for t in (np.int64, float, float, bool, float)), trunc
    return ("ok", np.array([int(x["T"]) for x in rows], dtype=np.int64), np.array([float(x["p"]) for x in rows]),
            np.array([float(x["q"]) for x in rows]), np.array([bool(x["m"]) for x in rows]),
            np.array([int(x["l"]) - int(x["f"]) + 1 for x in rows], dtype=float), trunc)


def _read_aggtrades_zip(src, t0, t1, chunk=500_000):
    """an aggTrades archive zip (path or file object) → (T, p, q, m, nraw) of the rows in [t0, t1], streamed: z.open + read_csv chunks with
    numeric dtypes (a header line is sniffed), stops once past t1 (the archive is time-ordered). Raises on a bad archive."""
    out = []
    with zipfile.ZipFile(src) as z, z.open(z.namelist()[0]) as fh:
        first = fh.peek(200)[:200].split(b"\n", 1)[0]
        hdr = 0 if first[:1] and not first[:1].isdigit() else None
        rd = pd.read_csv(fh, header=hdr, names=["a", "p", "q", "f", "l", "t", "m"], usecols=["p", "q", "f", "l", "t", "m"],
                         dtype={"p": np.float64, "q": np.float64, "f": np.int64, "l": np.int64, "t": np.int64, "m": str}, chunksize=chunk)
        for ch in rd:
            w = ch[(ch.t >= int(t0)) & (ch.t <= int(t1))]
            if len(w):
                out.append(w)
            if len(ch) and int(ch.t.iloc[-1]) > int(t1):
                break
    if not out:
        z_ = np.array([], dtype=np.int64)
        return z_, np.array([]), np.array([]), np.array([], dtype=bool), np.array([])
    d = pd.concat(out).sort_values("t", kind="stable")
    return (d.t.values.astype(np.int64), d.p.values, d.q.values, d.m.astype(str).str.strip().str.lower().isin(["true", "1"]).values,
            (d.l.values - d.f.values + 1).astype(float))


def _aggtrades_archive(sym, t0, t1, now_ms, budget):
    """the same aggTrades from the public daily archive (data.binance.vision) for a window the REST endpoint no longer serves: each pair-day zip
    is downloaded ONCE per run into a temp file (reused for every ON bar of that pair-day; ≤ budget['dl'] downloads, deadline-bounded) and
    streamed. 404 more than ONS_FLOW_GIVEUP_D days after t0 → 'missing'; anything else that fails → 'pending' (retried next run).
    → ('ok', T, p, q, m, nraw, False) / ('pending' | 'missing', …None)."""
    date = f"{pd.Timestamp(int(t0), unit='ms'):%Y-%m-%d}"
    key = (sym, date)
    if key not in _ARCH:
        day_end = int(pd.Timestamp(date, tz="UTC").value // 1_000_000) + 86_400_000
        if now_ms < day_end + TICK_FIRST_TRY_H * H or budget.get("dl", 0) <= 0 or time.monotonic() > budget.get("deadline", float("inf")):
            return ("pending",) + (None,) * 6
        q_ = urllib.parse.quote(sym)
        url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{q_}/{q_}-aggTrades-{date}.zip"
        try:
            resp = urllib.request.urlopen(url, timeout=30)
        except urllib.error.HTTPError as ex:
            if ex.code == 404 and now_ms > int(t0) + ONS_FLOW_GIVEUP_D * 86_400_000:
                _ARCH[key] = "missing"
            return (_ARCH.get(key, "pending"),) + (None,) * 6
        except Exception:
            return ("pending",) + (None,) * 6
        budget["dl"] -= 1
        stop_at = min(time.monotonic() + TICK_DL_DEADLINE_S, budget.get("deadline", float("inf")))
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".zip", prefix="ons_aggtrades_")
        try:
            with resp, os.fdopen(fd, "wb") as out:
                while True:
                    if time.monotonic() > stop_at:
                        raise TimeoutError("download deadline")
                    ch = resp.read(1 << 20)
                    if not ch:
                        break
                    out.write(ch)
            _ARCH[key] = tmp
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            return ("pending",) + (None,) * 6
    v = _ARCH[key]
    if v == "missing":
        return ("missing",) + (None,) * 6
    try:
        T, p, q, m, nraw = _read_aggtrades_zip(v, t0, t1)
    except Exception:
        return ("pending",) + (None,) * 6
    return "ok", T, p, q, m, nraw, False


def _arch_clear():
    for v in list(_ARCH.values()):
        if v not in ("missing",):
            try:
                os.remove(v)
            except OSError:
                pass
    _ARCH.clear()


def _ons_flow(sym, sig, close_px, nh, now_ms, budget):
    """the flow features of one ON bar (sig = its close ms) → dict incl. flow_state 'ok' / 'pending' / 'missing' and flow_src. REST
    aggTrades while the window is < 2 days old ("Search window is restricted to recent 2 days only" → 400 → the archive); a REST read the
    page cap truncated goes to the archive, else stays pending (never stored as ok). Failures → 'pending'; 'missing' only from an archive
    404 past ONS_FLOW_GIVEUP_D. A rate limit (HTTP 418 / 429) is re-raised so the tracker stops for this run."""
    if now_ms < sig + ONS_FLOW_MS + 5_000:
        return dict(flow_state="pending")
    try:
        src, st, trunc, rest_trunc = ("rest" if now_ms - sig < ONS_REST_MAX_AGE_MS else "archive"), "pending", False, False
        if src == "rest":
            try:
                st, T, p, q, m, nraw, trunc = _aggtrades_rest(sym, sig, sig + ONS_FLOW_MS - 1, budget["deadline"])
                if st == "ok" and trunc:
                    src, rest_trunc = "archive", True            # the page cap cut the minute short → the archive, else wait
            except urllib.error.HTTPError as ex:
                if ex.code != 400:
                    raise
                src = "archive"
        if src == "archive":
            st, T, p, q, m, nraw, trunc = _aggtrades_archive(sym, sig, sig + ONS_FLOW_MS - 1, now_ms, budget)
        if st != "ok" or trunc:
            return dict(flow_state=("missing" if st == "missing" else "pending"), flow_trunc=(True if (trunc or rest_trunc) else None))
        b1 = [b for b in _kl(sym, "1m", sig - 1440 * MIN, sig) if b[0] + MIN <= sig]
        base = float(np.median([b[5] * (b[2] + b[3] + b[4]) / 3 for b in b1])) if len(b1) >= 60 else None
    except urllib.error.HTTPError as ex:
        if ex.code in (418, 429):
            raise
        return dict(flow_state="pending")
    except Exception:
        return dict(flow_state="pending")
    return dict(onscalp_flow(T, p, q, m, nraw, sig, close_px, base, nh), flow_state="ok", flow_trunc=False, flow_src=src)


def _ons_replay(sym, sig, th, live=False):
    """the engine replay at the bar closing at sig: → dict(kind='on', …replay fields, parity=True) when frenzy_walk on the 1,499 closed bars says
    flagged ∧ fresh_on on that bar; kind='redirect' (sig = its ON close) when in state on a later bar (a catch-up line / fill); else
    kind='not_on' — unless live=True (the bot itself opened a FRENZY fill / logged a non-catch-up FRENZY line on this bar = the live engine
    judged it a fresh ON bar): then the bar is kept as an ON bar with parity=False and the replay's reason, NO_EPISODE included
    (normal_hour_usd is cached 6–8 h live, so a borderline volume × can differ — ORCA 10-06 09:40: replay × 99.8 < 100, live opened it).
    A short 5m / 1h fetch raises (retried next run); a pair listed < ~10 days (the 1h window starts after the requested start) has no
    normal hour live either → not ON."""
    closed = [b for b in _kl(sym, "5m", sig - 1499 * BAR, sig - 1) if b[0] + BAR <= sig]
    if len(closed) < 300 or closed[-1][0] != sig - BAR:
        raise ValueError("5m window missing")
    h1 = _kl(sym, "1h", sig - 744 * H, sig)
    nh = normal_hour_usd(h1, closed[-1][0])
    if nh is None:
        if not h1 or int(h1[0][0]) <= sig - 744 * H + H:
            raise ValueError("1h window short / unreadable")
        if not live:
            return dict(kind="not_on", why="NO_NORMAL_HOUR (listed < ~10 days)")
    ep = frenzy_walk(closed, nh, th) if nh else None
    why = None
    if not ep:
        why = "NO_EPISODE"
    elif not frenzy_flagged(ep, th):
        why = "NOT_FLAGGED"
    elif not ep.get("fresh_on"):
        on = ep.get("on_bar_ts")
        if ep.get("in_state") and on is not None and int(on) + BAR < sig:
            return dict(kind="redirect", sig=int(on) + BAR)
        why = "FRENZY_ON (in state, no ON bar)" if ep.get("in_state") else "not in state"
    if why and not live:
        return dict(kind="not_on", why=why)
    ep = ep or {}
    atr = wilder_atr_pct(closed[-300:])
    code = frenzy_long_status(ep, atr, frenzy_vol24_at(closed), th)[1] if not why else "LIVE_ONLY"
    di, ad = frenzy_di_spread(closed[-300:]), frenzy_adx_delta(closed[-300:])
    rn = lambda k, d: round(ep[k], d) if ep.get(k) is not None else None
    return dict(kind="on", spike_at=(_iso(ep["spike_ts"]) if ep.get("spike_ts") else None), hours=rn("hours", 2),
                above_streak=(int(ep["above_streak"]) if ep.get("above_streak") is not None else None), atr=atr,
                vs_vwap=rn("vs_vwap_pct", 3), bar_ret=rn("bar_ret_pct", 4), vol_mult=rn("vol_mult", 1), replay_code=code,
                adx_delta=ad, di_spread=di, strong=bool(di is not None and ad is not None and ad > 0 and di > 0),
                on_close=float(closed[-1][4]), nh=nh, parity=not why,
                parity_why=(f"replay: {why} (volume × {ep.get('vol_mult') or 0:.1f}, streak {ep.get('above_streak')})" if why else None))


def _ons_shadow(sym, sig, now_ms, budget):
    """the frozen ON-scalp pricing of the bar closing at sig: ticks once the archive is out — entry = the first print ≥ close + 12 s, the
    window then re-anchored at the ACTUAL entry (t_e + 2 h + 2 min) — else (or when the tick walk is still 'open') 1m pseudo prints o → l →
    h → c from the signal-close minute (entry = its open; PROVISIONAL; final on 1m when the archive is missing / empty / stuck past
    TICK_GIVEUP_D, or by age after STALE_D days — a stale row that still has no exit is final as 'no exit (stale)', no P&L).
    → (entry, t_e, walk dict, px_src, tick_state, final). Raises on no 1m data."""
    slack = 2 * MIN
    h0 = sig + ENTRY_LAG_MS + ONS_T_MIN * MIN + slack
    giveup = now_ms > (h0 // 86_400_000 + 1) * 86_400_000 + TICK_GIVEUP_D * 86_400_000
    stale = now_ms > h0 + STALE_D * 86_400_000
    st = "pending"
    if now_ms >= h0:
        st, tt, pp = _ticks(sym, sig, h0, now_ms, budget)
        if st == "ok":
            e, t_e = gc_entry(tt, pp, sig)
            if e is None:
                st = "empty"
            else:
                h1 = int(t_e) + ONS_T_MIN * MIN + slack
                if h1 > h0 and now_ms >= h1:
                    st, tt, pp = _ticks(sym, sig, h1, now_ms, budget)
                if st == "ok":
                    w = onscalp_walk(tt, pp, e, t_e)
                    if w["how"] in ("TP +3", "2 h"):
                        return e, t_e, w, "tick", st, True
                    st = "open"                                  # ticks end before the time exit → the 1m / stale path decides
    m1 = [b for b in _kl(sym, "1m", sig, min(now_ms, h0 + 10 * MIN)) if b[0] + MIN <= now_ms]
    if not m1 or m1[0][0] != sig:
        raise ValueError("1m klines unavailable")
    tt, pp = m1_prints_lh(m1)
    e = float(m1[0][1])
    w = onscalp_walk(tt, pp, e, sig, gap=True)
    fin = bool(now_ms >= h0 and (st == "missing" or (st in ("empty", "open") and giveup) or stale))
    if fin and w["how"] not in ("TP +3", "2 h"):
        if not stale:
            fin = False
        else:
            w = dict(w, how="no exit (stale)", pnl=None, hit=False)
    return e, sig, w, ("1m (age)" if fin and stale and st not in ("missing", "empty") else "1m"), st, fin


def _ons_live(J, F, sym, cks):
    """what the bot did on the candidate bars of this ON row: the journal's FRENZY lines (gate, or OPEN:<strategy>) and any FRENZY fill."""
    sigs = {c.split("|")[0] for c in cks}
    g, f = set(), []
    if J is not None and len(J):
        js = J[J.pair == sym]
        for r in js.itertuples():
            try:
                if _iso(_ms(r.t) // BAR * BAR) in sigs:
                    g.add(f"OPEN:{r.strategy}" if r.e == "OPEN" else str(r.gate))
            except Exception:
                continue
    if F is not None and len(F):
        for r in F[F.pair == sym].itertuples():
            try:
                if _iso(_ms(r.k) // BAR * BAR) in sigs:
                    a = _num(r.pnl_percentage) if str(r.status).upper() == "CLOSED" else None
                    f.append(f"{str(r.entry_strategy).replace('FRENZY_', '')} {r.k[11:19]}" + (f" {a:+.2f}" if a is not None else ""))
            except Exception:
                continue
    return ";".join(sorted(g)) or None, ";".join(f) or None


def _ons_row(rep, sym, sig, now_ms, budget, J, F, prev, cks, src, th):
    """one ON bar → its stored row (replay fields from rep, priced; the flow once readable — independent of the P&L's finality; a stored
    flow / book reading is kept)."""
    k = _iso(sig)
    prev = prev or {}
    cks = set(cks) | {c for c in str(prev.get("cand_keys") or "").split(";") if c}
    gates, fill = _ons_live(J, F, sym, cks)
    gates = gates or (prev.get("live_gates") if isinstance(prev.get("live_gates"), str) else None)
    row = dict(k=k, pair=sym, day=k[:10], cohort=k >= GC_FROM, ver=ONS_VER, is_on=True, not_on_reason=None, cand_keys=";".join(sorted(cks)),
               src=(src or prev.get("src")), **{c: rep.get(c) for c in ONS_REPLAY_COLS},
               live_gates=gates, live_fill=(fill or prev.get("live_fill")), liq_flow=ONS_LIQ_NOTE)
    row["univ"] = prev.get("univ") if isinstance(prev.get("univ"), str) else onscalp_universe(sym, rep.get("replay_code"), gates, th)
    e, t_e, w, px, st, fin = _ons_shadow(sym, sig, now_ms, budget)
    row.update(entry=e, entry_at=_iso(t_e), pnl=w["pnl"], exit_how=w["how"], exit_at=(_iso(w["exit_ms"]) if w["exit_ms"] else None),
               mfe=w["mfe"], mae=w["mae"], hit=w["hit"], liq_02=bool(w["mae"] is not None and w["mae"] <= -(ONS_LIQ + ONS_COST)),
               px_src=px, tick_state=st, final=bool(fin))
    if str(prev.get("flow_state")) in ("ok", "missing"):
        row.update({c: prev.get(c) for c in ONS_FLOW_COLS})
    else:
        row.update(_ons_flow(sym, sig, rep.get("on_close"), rep.get("nh"), now_ms, budget))
    for c in ONS_BOOK_COLS + ("book_t", "book_lag_s", "book_level"):
        row[c] = prev.get(c)
    return row


def _ons_book_scan(need, now_ms):
    """journal BOOK lines (minute-level ob_* snapshots) for the wanted (minute t, pair) keys, from the decisions exports modified in the last
    JR_MAX_AGE_D days whose name time is after the earliest wanted minute. → {(t, pair): {col: value}}. Never raises."""
    out = {}
    if not need:
        return out
    pairs = {p for _, p in need}
    t_min = min(t for t, _ in need)
    for f in sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv"))):
        try:
            if os.path.getmtime(f) * 1000 < now_ms - JR_MAX_AGE_D * 86_400_000:
                continue
            nm = os.path.basename(f)[len("scalpars_decisions_paper_"):-4]
            if f"{nm[:10]}T{nm[11:].replace('-', ':')}" < t_min:
                continue
            with open(f, "rb") as fh:
                hdr = fh.readline().decode("utf-8", "replace").rstrip("\r\n").split(",")
                ix = {c: hdr.index(c) for c in ONS_BOOK_COLS if c in hdr}
                for ln in fh:
                    if b",BOOK," not in ln:
                        continue
                    p = ln.decode("utf-8", "replace").rstrip("\r\n").split(",")
                    key = (p[0][:19], p[2])
                    if p[2] in pairs and key in need and key not in out:
                        out[key] = {c: _num(p[i]) if i < len(p) else None for c, i in ix.items()}
        except Exception:
            continue
    return out


def _ons_attach_book(df, now_ms):
    """the BOOK snapshot closest to each ON close within 60 s after it (minute stamps: the close's own minute, then the next), kept once found."""
    if not len(df):
        return df
    T = lambda v: str(v) in _TRUE
    want = {}
    for i, r in df.iterrows():
        if not T(r.get("is_on")) or isinstance(r.get("book_t"), str) or now_ms > _ms(r.k) + JR_MAX_AGE_D * 86_400_000:
            continue
        for lag in range(0, ONS_BOOK_LAG_S + 1, 60):
            want.setdefault((_iso(_ms(r.k) + lag * 1000), r.pair), []).append((i, lag))
    got = _ons_book_scan(set(want), now_ms)
    if not got:
        return df
    df = df.copy()
    for c in ONS_BOOK_COLS + ("book_t", "book_lag_s", "book_level"):
        if c not in df:
            df[c] = None
        df[c] = df[c].astype(object)
    best = {}
    for key, lst in want.items():
        if key in got:
            for i, lag in lst:
                if i not in best or lag < best[i][1]:
                    best[i] = (key, lag)
    for i, (key, lag) in best.items():
        for c, v in got[key].items():
            df.at[i, c] = v
        df.at[i, "book_t"] = key[0]; df.at[i, "book_lag_s"] = lag; df.at[i, "book_level"] = "minute"
    return df


def _rate_limited(ex):
    return isinstance(ex, urllib.error.HTTPError) and ex.code in (418, 429)


def ons_run(now_ms, th, J, F):
    try:
        return _ons_run(now_ms, th, J, F)
    finally:
        _arch_clear()


def _ons_run(now_ms, th, J, F):
    """find every fresh ON bar (journal FRENZY lines + FRENZY fills; catch-ups mapped to their ON bar), price new / provisional ones (cohort
    first, newest first), fill in the flow on final rows still without it, attach the BOOK snapshot, VALIDATE (a failing row → one .bad per
    row key), save the two files, render the ON_SCALP section."""
    need = ("k", "pair", "final", "ver", "is_on")
    old = pd.concat([_load_csv(ONS_CSV, need), _load_csv(ONS_CTRL_CSV, need)], ignore_index=True)
    prev, resolved = {}, set()
    for r in old.to_dict("records"):
        prev[(r["k"], r["pair"])] = r
        resolved |= {c for c in str(r.get("cand_keys") or "").split(";") if c}
    cands = {}
    if J is not None and len(J):
        for r in J[J.t.astype(str) >= GC_SCAN_FROM].itertuples():
            try:
                c_ = cands.setdefault(f"{_iso(_ms(r.t) // BAR * BAR)}|{r.pair}", dict(src=set(), live=False))
                c_["src"].add("journal"); c_["live"] |= not str(r.gate).startswith("FRENZY_CATCHUP")
            except Exception:
                continue
    if F is not None and len(F):
        for r in F[F.entry_strategy.astype(str).isin(["FRENZY_LONG", "FRENZY_WIDE"]) & (F.k.astype(str) >= GC_SCAN_FROM)].itertuples():
            try:
                c_ = cands.setdefault(f"{_iso(_ms(r.k) // BAR * BAR)}|{r.pair}", dict(src=set(), live=False))
                c_["src"].add("fill"); c_["live"] = True
            except Exception:
                continue
    budget = {"dl": X_DL, "deadline": time.monotonic() + X_TIME_S}
    rows, add_ck, err, late, rl = {}, {}, 0, 0, False
    T = lambda v: str(v) in _TRUE
    for ck in sorted([c for c in cands if c not in resolved], key=lambda c: (c >= GC_FROM, c), reverse=True):   # cohort first, newest first
        if time.monotonic() > budget["deadline"]:
            late += 1
            continue
        k0, sym = ck.split("|")
        src, live = "+".join(sorted(cands[ck]["src"])), cands[ck]["live"]
        try:
            sig = _ms(k0)
            rep = _ons_replay(sym, sig, th, live=live)
            if rep["kind"] == "redirect":
                sig = rep["sig"]; src += "→catch-up"
                key = (_iso(sig), sym)
                if key in rows or key in prev:
                    add_ck.setdefault(key, set()).add(ck)
                    continue
                rep = _ons_replay(sym, sig, th, live=live)
                if rep["kind"] != "on":
                    rep = dict(kind="not_on", why=f"catch-up target not fresh ({rep.get('why', rep['kind'])})"); sig = _ms(k0)
            key = (_iso(sig), sym)
            if rep["kind"] == "not_on":
                rows[(k0, sym)] = dict(k=k0, pair=sym, day=k0[:10], cohort=k0 >= GC_FROM, ver=ONS_VER, is_on=False, not_on_reason=rep["why"],
                                       cand_keys=ck, src=src, strong=False, final=True)
                continue
            if key in rows or (key in prev and T(prev[key].get("final"))):
                add_ck.setdefault(key, set()).add(ck)
                continue
            rows[key] = _ons_row(rep, sym, sig, now_ms, budget, J, F, prev.get(key), {ck}, src, th)
        except Exception as ex:
            err += 1
            if _rate_limited(ex):
                budget["deadline"] = 0; rl = True               # rate-limited: stop, the rest waits for the next run
    redo = sorted([kr for kr in prev.items() if kr[0] not in rows and T(kr[1].get("is_on"))],
                  key=lambda kr: (kr[0][0] >= GC_FROM, kr[0][0], kr[0][1]), reverse=True)   # cohort first, newest first
    for key, r in redo:
        fin_ok = T(r.get("final")) and str(r.get("ver")) in (str(ONS_VER), f"{ONS_VER}.0")
        if fin_ok and str(r.get("flow_state")) in ("ok", "missing"):
            continue
        if time.monotonic() > budget["deadline"]:
            late += 1
            continue
        try:
            if fin_ok:                                          # P&L final, the description-only flow still to come: flow only
                rows[key] = dict(r, **_ons_flow(key[1], _ms(key[0]), _num(r.get("on_close")), _num(r.get("nh")), now_ms, budget))
                continue
            rep = {c: r.get(c) for c in ONS_REPLAY_COLS}
            rep.update(strong=T(r.get("strong")), on_close=_num(r.get("on_close")), nh=_num(r.get("nh")), parity=T(r.get("parity")))
            rows[key] = _ons_row(rep, key[1], _ms(key[0]), now_ms, budget, J, F, r, set(), None, th)
        except Exception as ex:
            err += 1
            if _rate_limited(ex):
                budget["deadline"] = 0; rl = True
    new = pd.DataFrame(list(rows.values()))
    allr = pd.concat([old, new], ignore_index=True) if len(new) else old.copy()
    L = ["## ⚡ ON_SCALP — every fresh FRENZY ON bar with the strong flag, bought at +12 s, TP +3 net / 2 h / no stop (observe-only, registered 2026-10-06)", "",
         "The one observe line reports/FRENZY_ON_SCALP_STUDY_2026-10-06.md allows (verdict there: not a strategy; expected ≈ 0). Every fresh FRENZY ON bar "
         "whatever the live refusal code — journal FRENZY lines AND the FRENZY fills (a READY fill leaves no refusal line), replayed with the engine's "
         "frenzy_walk (catch-ups mapped to their ON bar via on_bar_ts; ‡ = live-only: the bot judged it ON, the replay disagrees). Strong = ADX Δ > 0 ∧ "
         "DI spread > 0 on closed[-300:] at the ON bar, exactly as _frenzy_open sizes strong. Priced: first trade print ≥ the ON close + 12 s, 0.09 % "
         "fees + 0.10 % slippage, out at the first print ≥ +3 % net, else at 2 h, NO stop (the tail is the risk — MAE shown). Ticks once the archive "
         f"is out, else 1m prints open → low → high → close (ᵖ provisional). One per pair-EPISODE (spikes ≤ 30 min apart merged). ¹ = counted (from "
         f"{GC_FROM[:10]}, inside the study's universe — FRENZY_VOL24_LOW / blacklisted bars are tagged and kept out); earlier rows are reference only. "
         "Flow columns are DESCRIPTIVE (no gate): pre-entry = [close, +12 s) taker-buy share / move; post-entry = the first 60 s (taker-buy share "
         "of $, $ volume × the pair's median minute of the prior 24 h, low / high vs the close); book = the journal's minute-level BOOK snapshot.", ""]
    tail = (([f"_{err} candidate(s) / row(s) not priced this run (klines / trades unavailable" + (", Binance rate limit — stopped" if rl else "")
              + ") — retried next run._"] if err else [])
            + ([f"_{late} candidate(s) / row(s) left for the next run (the {X_TIME_S} s time budget was spent or a rate limit stopped it)._"] if late else []))
    if not len(allr):
        return L + ["No FRENZY ON bar found yet."] + tail + [""]
    allr = allr.drop_duplicates(["k", "pair"], keep="last").sort_values(["k", "pair"], kind="stable").reset_index(drop=True)
    for key, cks in add_ck.items():                               # a later line / fill of an already stored ON bar: remember it was seen
        m = (allr.k == key[0]) & (allr.pair == key[1])
        if m.any():
            i = allr.index[m][0]
            allr.at[i, "cand_keys"] = ";".join(sorted({c for c in str(allr.at[i, "cand_keys"] or "").split(";") if c} | cks))
    allr = _ons_attach_book(allr, now_ms)
    bad = []
    for i, r in zip(allr.index, allr.to_dict("records")):
        try:
            onscalp_validate(r, T(r.get("strong")) and T(r.get("is_on")))
        except ValueError as ex:
            bad.append((i, str(ex)))
    qname = None
    if bad:
        qname, _ = _quarantine(allr, bad, ONS_CSV)
        allr = allr.drop(index=[i for i, _ in bad]).reset_index(drop=True)
    allr["episode"] = episode_keys(allr)
    M = _ons_masks(allr)
    allr["counted"] = M["counted"]
    S_ = lambda c: allr[c].astype(str).isin(_TRUE) if c in allr else pd.Series(False, index=allr.index)
    strong = S_("is_on") & S_("strong")
    _save_csv(allr[strong], ONS_CSV); _save_csv(allr[~strong], ONS_CTRL_CSV)
    f = lambda v, d=2: "–" if _num(v) is None else f"{float(v):+.{d}f}"
    pc = lambda v: "–" if _num(v) is None else f"{float(v) * 100:.0f} %"
    sd = allr[strong]
    L += ["| ON close UTC | Pair | Code | Live | ADX Δ | DI | Pre-entry 12 s: buy / move | +12 s vs close | Post 60 s: buy | $ × median min | Low / high 60 s | Book imb ±0.5 % | Result | Exit | MAE | Px |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sd.tail(15).itertuples():
        mk = ("" if T(r.final) else "ᵖ") + ("¹" if T(r.counted) else "") + ("" if T(getattr(r, "parity", True)) else "‡") + ("" if T(r.cohort) else " (ref)")
        un = str(getattr(r, "univ", "ok"))
        live = (str(r.live_fill) if isinstance(r.live_fill, str) else str(r.live_gates).replace("FRENZY_", "") if isinstance(r.live_gates, str) else "–")
        vx = _num(getattr(r, "vol_x_24h", None))
        flow = str(getattr(r, "flow_state", ""))
        L.append(f"| {str(r.k)[5:16].replace('T', ' ')}{mk} | {str(r.pair).replace('USDT', '')} | {str(r.replay_code).replace('FRENZY_', '')}"
                 f"{'' if un in ('ok', 'nan') else ' ⊘' + un} | {live} | {f(r.adx_delta)} | {f(r.di_spread, 1)} | "
                 f"{pc(getattr(r, 'buy_share_pre', None))} / {f(getattr(r, 'move_pre', None))} | {f(getattr(r, 'px12_vs_close', None))} | "
                 f"{pc(getattr(r, 'buy_share_60', None))}{'' if flow == 'ok' else ' (' + flow + ')'} | {'–' if vx is None else f'{vx:.1f}×'} | "
                 f"{f(getattr(r, 'mdd_60', None))} / {f(getattr(r, 'mru_60', None))} | {f(getattr(r, 'ob_imb_05', None))} | "
                 f"{f(r.pnl)} | {r.exit_how} | {f(r.mae)} | {r.px_src} |")
    cnt = allr[M["counted"] & S_("final")]
    st, tx = onscalp_check(cnt)
    nlo = int((~S_("parity")[cnt.index]).sum()) if len(cnt) else 0
    lab = {"candidate": "📋 CANDIDATE for a pre-registered probe study (no arming)", "retire": "❌ retire", "collecting": "⏳ collecting"}
    L += ["", f"**ON_SCALP bar (FROZEN, the study's: N ≥ {ONS_N} fires on ≥ {ONS_DAYS} days ∧ mean > 0 with day-block 95 % CI low > 0 ∧ P(+3 within 2 h) ≥ "
              f"{ONS_P3:g} % ∧ forward max DD < {ONS_DD_MAX:.0f} % of a ${ONS_BOOK0:,.0f} book at 0.2 sizing (sequential, notional {ONS_NOTIONAL} × equity, "
              f"liquidation −{ONS_LIQ} % price) → candidate, then the {ONS_HAIRCUT[0] * 100:.0f}–{ONS_HAIRCUT[1] * 100:.0f} % haircut · ADDITIONS: retire at mean ≤ 0 "
              f"by {ONS_N} / no verdict by {ONS_RETIRE_N}):** " + lab.get(st, st) + f" ({tx} · ‡ live-only {nlo} of {len(cnt)})"]
    x = pd.to_numeric(cnt.pnl, errors="coerce").dropna()
    if len(x):
        wi = x.idxmin(); wr_ = cnt.loc[wi]
        miss = cnt.loc[x.index][~cnt.loc[x.index].hit.astype(str).isin(_TRUE)]
        xm = pd.to_numeric(miss.pnl, errors="coerce")
        L.append(f"- Worst fill: {str(wr_.pair).replace('USDT', '')} {str(wr_.k)[5:16].replace('T', ' ')} {x[wi]:+.2f} % (MAE {f(wr_.mae)}, {wr_.exit_how}) · never touched +3 "
                 f"within 2 h: {len(miss)} of {len(x)} ({len(miss) / len(x) * 100:.0f} %)" + (f", mean exit {xm.mean():+.2f} %" if len(xm) else "")
                 + f" · liquidation-deep MAE (≤ −{ONS_LIQ} % price): {int(S_('liq_02')[x.index].sum())}.")
    else:
        L.append("- Worst fill / never-touched-+3 share: no counted final fire yet.")
    rf = allr[M["ref"]]
    xr = pd.to_numeric(rf[S_("final")[rf.index]].pnl, errors="coerce").dropna()
    L.append(f"- Reference (before {GC_FROM[:10]}, not counted): {int(M['ref'].sum())} strong pair-episodes, {len(xr)} final" + (f" · WR {(xr > 0).mean() * 100:.0f} % · mean "
             f"{xr.mean():+.2f} %" if len(xr) else "") + f" · {int((~S_('final')[sd.index]).sum())} strong rows provisional.")
    cc = allr[M["control"]]
    xc = pd.to_numeric(cc[S_("final")[cc.index]].pnl, errors="coerce").dropna()
    L.append(f"- Control (NOT strong, same ruler, first per pair-episode from {GC_FROM[:10]}): {int(M['control'].sum())} episodes, {len(xc)} final"
             + (f" · WR {(xc > 0).mean() * 100:.0f} % · mean {xc.mean():+.2f} %" if len(xc) else "")
             + f" · all control rows {int((S_('is_on') & ~S_('strong')).sum())} (reports/SCOUT_FRENZY_ON_SCALP_CONTROL.csv).")
    un = allr["univ"].astype(str) if "univ" in allr else pd.Series("ok", index=allr.index)
    outu = allr[S_("is_on") & ~un.isin(["ok", "nan"])]
    L.append(f"- Outside the study's universe (tagged ⊘, never in the bar): {len(outu)}" + (" (" + ", ".join(f"{k_} {v}" for k_, v in outu.univ.value_counts().items())
             + f"; {int((S_('strong')[outu.index] & S_('cohort')[outu.index]).sum())} strong in the cohort window)" if len(outu) else "") + ".")
    fs = allr["flow_state"].astype(str) if "flow_state" in allr else pd.Series("", index=allr.index)
    tr = S_("flow_trunc")
    on = allr[S_("is_on") & S_("final") & (fs == "ok") & ~tr & allr.get("pnl", pd.Series(dtype=float)).notna()]
    if len(on):
        hit = on.hit.astype(str).isin(_TRUE)
        a = lambda g, c: f"{pd.to_numeric(g[c], errors='coerce').mean() * 100:.0f} %" if len(g) else "–"
        b = lambda g, c: f"{pd.to_numeric(g[c], errors='coerce').mean():+.2f} %" if len(g) else "–"
        L.append(f"- Flow (descriptive, every final ON row with a full flow reading, strong + control; {int(tr.sum())} truncated read(s) excluded): "
                 f"TP hit {int(hit.sum())} · pre-entry buy {a(on[hit], 'buy_share_pre')} / move {b(on[hit], 'move_pre')} · post 60 s buy "
                 f"{a(on[hit], 'buy_share_60')} · low {b(on[hit], 'mdd_60')}  vs  missed {int((~hit).sum())} · pre {a(on[~hit], 'buy_share_pre')} / "
                 f"{b(on[~hit], 'move_pre')} · post {a(on[~hit], 'buy_share_60')} · {b(on[~hit], 'mdd_60')}.")
    L.append(f"- Liquidations in the first 60 s: {ONS_LIQ_NOTE}.")
    L.append(f"- Year reference: {ONS_YEAR}.")
    lo = allr[S_("is_on") & ~S_("parity")]
    L.append(f"- ‡ Live-only ON bars (the bot opened / judged a fresh ON there, the replay disagrees — normal_hour_usd is cached 6–8 h live; kept, "
             f"counted when strong): {len(lo)}" + (" (" + "; ".join(f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16].replace('T', ' ')}: {r.parity_why}"
                                                                  for r in lo.tail(6).itertuples()) + ")" if len(lo) else "") + ".")
    no = allr[~S_("is_on")]
    L.append(f"_Candidates the replay says are NOT a fresh ON bar (journal lines / fills; parity notes, never priced): {len(no)}"
             + (" (" + "; ".join(f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16].replace('T', ' ')}: {r.not_on_reason}" for r in no.tail(6).itertuples()) + ")" if len(no) else "")
             + f" · strong ON rows {int(strong.sum())} · control ON rows {int((S_('is_on') & ~S_('strong')).sum())} · BOOK snapshot attached on "
             f"{int(allr.get('book_t', pd.Series(dtype=object)).notna().sum())}._")
    if bad:
        L.append(f"_⚠ {len(bad)} row(s) failed validation and were dropped" + (f" (quarantined to {qname})" if qname else " (already quarantined earlier)")
                 + ": " + "; ".join(m_ for _, m_ in bad[:3]) + "._")
    return L + tail + [""]


# ─────────────────────────── 🛟 HYBRID_EXIT (HYB) — 2026-10-06, observe-only per-fill shadow ───────────────────────────
HYB_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_HYBRID.csv")   # own store: the exit rows' schema / VER is untouched (no re-pricing)
HYB_VER = 1
# FROZEN — V3 of reports/FRENZY_EXIT_LOCK_VS_BULLRUN_2026-10-06.md (PREREG reports/FRENZY_EXIT_LOCK_VS_BULLRUN_PREREG_2026-10-06.txt):
# "−3; pk ≥ +1 → floor +0.2; pk ≥ +3 → max(+2, pk − 2)", pk = the PRIOR-print peak net %, levels on net (fees 0.09), 12 h cap.
HYB_ARM, HYB_FLOOR = 1.0, 0.2
# FROZEN re-open bar (report "Recommendation"): after ≥ 40 live FRENZY_LONG fills on ≥ 20 days (opened from GC_FROM), re-open only if
# HYB − LOCK2 > +0.30 %/fill ∧ day-block 95 % CI low > 0 ∧ still > 0 without the top 5 fills; otherwise "lock holds".
HYB_N, HYB_DAYS, HYB_MIN_D = 40, 20, 0.30
HYB_YEAR = ("year (report §1, 205 FRENZY_LONG engine fills, ticks, 0.10 slip on both): lock +0.387 vs V3 +0.204 → Δ −0.18 [−0.47, +0.11]; "
            "STRONG Δ −0.19 [−0.61, +0.26], halves −0.49 / +0.07; SAVED 45 (+144) vs CUT 38 (−144) + 21 lock-floor wins cut (−38)")


def hyb_line(pk):
    """the V3 exit line for a prior-print peak pk (net %): −3 below +1, the +0.2 floor from +1, the live lock max(+2, pk − 2) from +3
    (the highest applicable line wins)."""
    return max(2.0, pk - 2.0) if pk >= 3 else HYB_FLOOR if pk >= HYB_ARM else -3.0


def _hyb_how(line):
    return "stop" if line <= -3 else "+0.2 floor" if line == HYB_FLOOR else "floor / trail"


def walk_ticks_hyb(tt, pp, e, t0):
    """🛟 V3 on trade prints from t0, net of FEE: a print at / through the line (set by the prints BEFORE it) exits AT that print; 12 h of
    clock time. → (pnl, exit_ms, how)."""
    tt = np.asarray(tt, dtype=np.int64); pp = np.asarray(pp, dtype=float)
    m = tt >= int(t0)
    tt, pp = tt[m], pp[m]
    if not len(pp) or not e:
        return None, None, "no data"
    net = (pp / float(e) - 1) * 100 - FEE
    pk = np.maximum.accumulate(np.r_[-1e9, net[:-1]])
    line = np.where(pk >= 3, np.maximum(2.0, pk - 2.0), np.where(pk >= HYB_ARM, HYB_FLOOR, -3.0))
    t_end = int(t0) + CAP_MIN * MIN
    inc = tt < t_end
    hit = np.flatnonzero((net <= line) & inc)
    if len(hit):
        i = int(hit[0])
        return float(net[i]), int(tt[i]), _hyb_how(float(line[i]))
    if not inc.all():
        j = int(np.flatnonzero(inc)[-1]) if inc.any() else 0
        return float(net[j]), t_end, "12 h cap"
    return float(net[-1]), int(tt[-1]), "open"


def hyb_check(w):
    """the FROZEN HYBRID_EXIT re-open bar on final FRENZY_LONG fills from GC_FROM (column d = HYB − LOCK2, same ruler). → (state, text)."""
    d = pd.to_numeric(w.d, errors="coerce")
    w = w[d.notna()]; d = d.dropna()
    n, nd = len(d), w.day.nunique()
    t = f"{n}/{HYB_N} fills · {nd}/{HYB_DAYS} days" + (f" · Δ(HYB − lock) {d.mean():+.2f} %/fill" if n else "")
    if n < HYB_N or nd < HYB_DAYS:
        return "collecting", t
    ci = day_ci(d.values, w.day.values)
    wo5 = d.sort_values(ascending=False).iloc[5:].mean()
    t += (f" · day CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "") + f" · w/o top 5 {wo5:+.2f}"
    return ("reopen" if (d.mean() > HYB_MIN_D and ci and ci[0] > 0 and wo5 > 0) else "holds"), t


def hyb_anatomy(w):
    """SAVED = HYB ≥ 0 while the lock lost · CUT = the lock ended ≥ +2 while HYB went out at the +0.2 floor. → (n_saved, Σ, n_cut, Σ)."""
    h, l = pd.to_numeric(w.HYB_R, errors="coerce"), pd.to_numeric(w.LOCK2_R, errors="coerce")
    sv = (h >= 0) & (l < 0)
    ct = (l >= 2) & (w.HYB_how.astype(str) == "+0.2 floor")
    return int(sv.sum()), float((h - l)[sv].sum()), int(ct.sum()), float((h - l)[ct].sum())


def hyb_validate(r):
    """raises ValueError on a row that must not be saved: a final row without both results, or a Δ that is not HYB − LOCK2 on its ruler."""
    if str(r.get("final")) not in _TRUE:
        return
    h, l, d = _num(r.get("HYB_R")), _num(r.get("LOCK2_R")), _num(r.get("d"))
    if h is None or l is None or d is None:
        raise ValueError("final row without HYB / LOCK2 / Δ")
    if abs(d - (h - l)) > 1e-9:
        raise ValueError(f"Δ {d} ≠ HYB − LOCK2 {h - l}")
    if r.get("HYB_how") not in ("stop", "+0.2 floor", "floor / trail", "12 h cap"):
        raise ValueError(f"final row with exit '{r.get('HYB_how')}'")


def _hyb_price(src, now_ms, budget):
    """one fill (k = opened_at, entry = the actual entry price) → HYB and the lock on the SAME prints: ticks once the archive is out (both on
    every print), else 1m (the exit table's ruler: low before high inside a minute, the entry minute flattened) — PROVISIONAL; final on 1m
    when the archive is missing / empty past TICK_GIVEUP_D, or by age after STALE_D days. Fees 0.09, no slippage on either side (the report
    charges 0.10 on BOTH exits, so the Δ is the same). Raises on no 1m data."""
    sym, e = src["pair"], float(src["entry"])
    t0 = _ms(src["k"]); m0 = t0 // MIN * MIN
    horizon = t0 + CAP_MIN * MIN
    giveup = now_ms > (horizon // 86_400_000 + 1) * 86_400_000 + TICK_GIVEUP_D * 86_400_000
    stale = now_ms > horizon + STALE_D * 86_400_000
    m1 = [b for b in _kl(sym, "1m", m0, min(now_ms, horizon + MIN)) if b[0] + MIN <= now_ms]
    if not m1 or m1[0][0] != m0:
        raise ValueError("1m klines unavailable")
    m1in = [list(b) for b in m1]
    m1in[0] = [m0, e, max(e, m1in[0][4]), min(e, m1in[0][4]), m1in[0][4], m1in[0][5]]
    h1, _, hh1 = walk(m1in, e, "HYB"); l1, _, lh1 = walk(m1in, e, "LOCK2")
    row = dict(src, ver=HYB_VER, HYB_1m=h1, HYB_1m_how=hh1, LOCK2_1m=l1, px_src="1m", tick_state="pending")
    st = "pending"
    if now_ms >= horizon + 2 * MIN:
        st, tt, pp = _ticks(sym, t0, horizon + 2 * MIN, now_ms, budget)
        if st == "ok" and len(tt):
            hr, _, hh = walk_ticks_hyb(tt, pp, e, t0); lr, _, lh = _walk_ticks(tt, pp, e, t0)
            if hr is not None and lr is not None and hh != "open" and lh != "open":
                row.update(HYB_R=hr, LOCK2_R=lr, HYB_how=hh, d=hr - lr, px_src="tick", tick_state=st, final=True)
                return row
        if st == "ok":
            st = "empty" if not len(tt) else "open"         # ticks end before an exit → the 1m / stale path decides
    fin = bool(now_ms >= horizon and (st == "missing" or (st in ("empty", "open") and giveup) or stale) and hh1 != "open" and lh1 != "open")
    row.update(HYB_R=h1, LOCK2_R=l1, HYB_how=hh1, d=(h1 - l1 if h1 is not None and l1 is not None else None),
               px_src=("1m (age)" if fin and stale and st not in ("missing", "empty") else "1m"), tick_state=st, final=fin)
    return row


def hyb_run(now_ms, F, allr):
    """price new / provisional fills of the exit table with HYB (own store), validate, save → ({(k, pair): HYB on the table's 1m ruler}, lines)."""
    old = _load_csv(HYB_CSV, ("k", "pair", "final", "ver"))
    T = lambda v: str(v) in _TRUE
    prev = {(r["k"], r["pair"]): r for r in old.to_dict("records")} if len(old) else {}
    stamps = {}
    if F is not None and len(F):
        for r in F.itertuples():
            ad, di = _num(getattr(r, "entry_frenzy_adx_delta", None)), _num(getattr(r, "entry_frenzy_di_spread", None))
            stamps[(r.k, r.pair)] = (ad, di)
    work = []
    for r in (allr.to_dict("records") if len(allr) else []):
        key = (str(r["k"]), str(r["pair"]))
        p = prev.get(key)
        if p is not None and T(p.get("final")) and str(p.get("ver")) in (str(HYB_VER), f"{HYB_VER}.0"):
            continue
        ad, di = stamps.get(key, (None, None))
        if ad is None and p is not None:
            ad, di = _num(p.get("adx_delta")), _num(p.get("di_spread"))
        sl = str(r.get("sleeve"))
        work.append(dict(k=key[0], pair=key[1], day=key[0][:10], sleeve=sl, entry=float(r["entry"]), cohort=key[0] >= GC_FROM,
                         adx_delta=ad, di_spread=di,
                         strong=(None if ad is None or di is None or sl != "LONG" else bool(ad > 0 and di > 0))))
    work.sort(key=lambda s: (s["cohort"] and s["sleeve"] == "LONG", s["cohort"], s["k"]), reverse=True)   # the FRENZY_LONG cohort first, newest first
    budget = {"dl": X_DL, "deadline": time.monotonic() + X_TIME_S}; rows, err, late, rl = [], 0, 0, False
    for s in work:
        if time.monotonic() > budget["deadline"]:
            late += 1
            continue
        try:
            rows.append(_hyb_price(s, now_ms, budget))
        except Exception as ex:
            err += 1
            if _rate_limited(ex):
                budget["deadline"] = 0; rl = True               # rate-limited: stop, the rest waits for the next run
    new = pd.DataFrame(rows)
    allh = pd.concat([old, new], ignore_index=True) if len(new) else old.copy()
    L = []
    if len(allh):
        allh = allh.drop_duplicates(["k", "pair"], keep="last").sort_values(["k", "pair"], kind="stable").reset_index(drop=True)
        bad = []
        for i, r in zip(allh.index, allh.to_dict("records")):
            try:
                hyb_validate(r)
            except ValueError as ex:
                bad.append((i, str(ex)))
        if bad:
            qn, _ = _quarantine(allh, bad, HYB_CSV)
            allh = allh.drop(index=[i for i, _ in bad]).reset_index(drop=True)
            L.append(f"_⚠ HYBRID_EXIT: {len(bad)} row(s) failed validation and were dropped" + (f" (quarantined to {qn})" if qn else " (already quarantined earlier)")
                     + ": " + "; ".join(m_ for _, m_ in bad[:3]) + "._")
        _save_csv(allh, HYB_CSV)
    hmap = {(k, p): (h, T(f_)) for k, p, h, f_ in zip(allh.k, allh.pair, allh.HYB_1m, allh.final)} if len(allh) and "HYB_1m" in allh else {}
    S_ = lambda c: allh[c].astype(str).isin(_TRUE) if c in allh else pd.Series(False, index=allh.index)
    lab = {"reopen": "📋 RE-OPEN the hybrid question (observe → a pre-registered study; no arm)", "holds": "🔒 lock holds", "collecting": "⏳ collecting (lock holds)"}
    if len(allh):
        lg = allh[(allh.sleeve.astype(str) == "LONG") & S_("final")]
        cnt = lg[S_("cohort")[lg.index]]
        st, tx = hyb_check(cnt)
    else:
        lg = cnt = pd.DataFrame(columns=["d", "day", "HYB_R", "LOCK2_R", "HYB_how", "strong"]); st, tx = hyb_check(cnt)
    L = ["", f"**HYBRID_EXIT (HYB = V3 of reports/FRENZY_EXIT_LOCK_VS_BULLRUN_2026-10-06.md, frozen: −3 stop; peak ≥ +{HYB_ARM:g} → +{HYB_FLOOR:g} floor; "
              f"peak ≥ +3 → the lock max(+2, peak − 2); 12 h; HYB column = the table's 1m ruler; the bar reads HYB − LOCK2 on the SAME prints — ticks once "
              f"out, else 1m; re-open only if ≥ {HYB_N} FRENZY_LONG fills from {GC_FROM[:10]} on ≥ {HYB_DAYS} days show Δ > +{HYB_MIN_D:.2f} %/fill ∧ "
              f"day CI low > 0 ∧ > 0 without the top 5):** " + lab.get(st, st) + f" ({tx})"] + L
    for nm, g in (("cohort", cnt), ("reference (before " + GC_FROM[:10] + ")", lg[~S_("cohort")[lg.index]] if len(lg) else lg)):
        if not len(g):
            L.append(f"- FRENZY_LONG {nm}: no final fill yet.")
            continue
        sg = g.strong.astype(str).isin(_TRUE); ng = g.strong.astype(str).isin(["False", "0", "0.0"])
        parts = []
        for lb, gg in (("strong", g[sg]), ("normal", g[ng]), ("unstamped", g[~sg & ~ng])):
            if len(gg):
                parts.append(f"{lb} {len(gg)} · lock {pd.to_numeric(gg.LOCK2_R).mean():+.2f} · HYB {pd.to_numeric(gg.HYB_R).mean():+.2f} · Δ {pd.to_numeric(gg.d).mean():+.2f}")
        nsv, ssv, nct, sct = hyb_anatomy(g)
        L.append(f"- FRENZY_LONG {nm}: {len(g)} fills · " + " | ".join(parts) + f" · SAVED (HYB ≥ 0, lock lost) {nsv} (Δ {ssv:+.2f}) · CUT (lock ≥ +2, HYB "
                 f"out at the +0.2 floor) {nct} (Δ {sct:+.2f}).")
    if len(allh):
        lab_ = lambda r: f"{str(r.pair).replace('USDT', '')} {str(r.k)[5:16].replace('T', ' ')} {str(r.sleeve)}"
        L.append("- Per fill (HYB / lock on the bar's ruler): " + "; ".join(f"{lab_(r)} {_num(r.HYB_R) or 0:+.2f} / {_num(r.LOCK2_R) or 0:+.2f}"
                                                                          + ("" if T(r.final) else "ᵖ") + f" ({r.px_src})" for r in allh.tail(8).itertuples()) + ".")
    L.append(f"- Year reference: {HYB_YEAR}.")
    if err:
        L.append(f"_HYBRID_EXIT: {err} fill(s) not priced this run (klines unavailable" + (", Binance rate limit — stopped" if rl else "") + ") — retried next run._")
    if late:
        L.append(f"_HYBRID_EXIT: {late} fill(s) left for the next run (the {X_TIME_S} s time budget was spent)._")
    return hmap, L


def _hyb_safe(now_ms, F, allr):
    try:
        return hyb_run(now_ms, F, allr)
    except Exception as ex:
        return {}, ["", f"_HYBRID_EXIT unavailable this run ({str(ex)[:120]})._"]


# ─────────────────────────── 🪶 tracker 12 (FRENZY_LITE watch + LITE_ATR, DECISION_LOG 243) ───────────────────────────
LITE_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_LITE.csv")
LITE_ATR_CAP = 2.5          # FROZEN: frenzy_max_atr_pct at LITE's registration (2026-10-07) — the split never follows a later config change
LITE_REVIEW_N, LITE_REVIEW_DAYS = 40, 15   # review due at ≥ 40 closed fills on ≥ 15 days
LITE_FLAG_N = 20            # avg < 0 at ≥ 20 closed fills → flag for operator review (NOT an auto-off)
LITE_COLS = ("k", "pair", "day", "closed", "actual", "atr", "hours", "above_streak", "vol_mult", "vs_vwap", "bar_ret", "gvol", "adx_delta", "di_spread",
             "lev")
# 🪶 Oct-7 (DECISION_LOG 247, operator override at 3 fills): FRENZY_LITE lev 0.2 → 0.32. Results split by size era (the fill's own
# cell_lev_multiplier); the first LITE_LEV_FLAG_N closed fills at ≥ 0.3 averaging < 0 → "⚠ review" flag — NEVER an automatic revert.
LITE_LEV_NEW = 0.3          # a fill sized at lev mult ≥ this is the 0.32 era
LITE_LEV_FLAG_N = 10


def _lite_fills():
    """FRENZY_LITE LONG fills in the orders exports (newest export wins per opened_at + pair) → DataFrame in LITE_COLS (never raises on a bad file)."""
    cols = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "entry_atr_pct", "entry_frenzy_hours",
            "entry_frenzy_above_streak", "entry_frenzy_vol_mult", "entry_frenzy_vs_vwap_pct", "entry_frenzy_bar_ret_pct", "entry_frenzy_gvol",
            "entry_frenzy_adx_delta", "entry_frenzy_di_spread", "cell_lev_multiplier")
    fr = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols)
        except Exception:
            continue
        if {"opened_at", "entry_strategy", "pair"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(LITE_COLS))
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable").reindex(columns=list(cols) + ["_m"])
    o = o[(o.entry_strategy.astype(str) == "FRENZY_LITE") & (o.direction.astype(str) == "LONG")]
    return lite_rows(o)


def lite_rows(o):
    """export rows (orders CSV columns) → tracker rows (pure; the newest row per opened_at + pair wins)."""
    if not len(o):
        return pd.DataFrame(columns=list(LITE_COLS))
    k = o.opened_at.astype(str).str[:19]
    num = lambda c: pd.to_numeric(o[c], errors="coerce") if c in o else np.nan
    closed = o.status.astype(str).str.upper().eq("CLOSED") if "status" in o else False
    out = pd.DataFrame(dict(k=k, pair=o.pair.astype(str), day=k.str[:10], closed=closed,
                            actual=num("pnl_percentage").where(closed), atr=num("entry_atr_pct"), hours=num("entry_frenzy_hours"),
                            above_streak=num("entry_frenzy_above_streak"), vol_mult=num("entry_frenzy_vol_mult"), vs_vwap=num("entry_frenzy_vs_vwap_pct"),
                            bar_ret=num("entry_frenzy_bar_ret_pct"), gvol=num("entry_frenzy_gvol"), adx_delta=num("entry_frenzy_adx_delta"),
                            di_spread=num("entry_frenzy_di_spread"), lev=num("cell_lev_multiplier")))
    return out.drop_duplicates(["k", "pair"], keep="last").reset_index(drop=True)


def lite_merge(old, new):
    """stored rows + this run's export rows → one row per opened_at + pair (the export wins: a fill that closed since updates its row)."""
    parts = [x for x in (old, new) if x is not None and len(x)]
    if not parts:
        return pd.DataFrame(columns=list(LITE_COLS))
    m = pd.concat(parts, ignore_index=True).reindex(columns=list(LITE_COLS))
    m["closed"] = m.closed.astype(str).isin(_TRUE + ("true",))
    return m.drop_duplicates(["k", "pair"], keep="last").sort_values("k").reset_index(drop=True)


def lite_stats(w):
    """closed fills with a P&L → dict(n, days, wr, avg, sum) (avg / wr None when empty)."""
    a = pd.to_numeric(w.actual, errors="coerce") if len(w) else pd.Series(dtype=float)
    g = w[a.notna()] if len(w) else w
    a = a.dropna()
    n = len(a)
    return dict(n=n, days=(g.day.nunique() if n else 0), wr=((a > 0).mean() * 100 if n else None), avg=(a.mean() if n else None), sum=(a.sum() if n else 0.0))


def lite_watch(w):
    """the frozen FRENZY_LITE watch line on CLOSED fills → (state, text). state: flag (avg < 0 at ≥ 20 — operator review, never an
    auto-off) · review (≥ 40 fills on ≥ 15 days) · collecting. A flag outranks a due review (both are said)."""
    st = lite_stats(w)
    n, nd, av = st["n"], st["days"], st["avg"]
    txt = f"{n}/{LITE_REVIEW_N} closed · {nd}/{LITE_REVIEW_DAYS} days" + (f" · WR {st['wr']:.0f} % · avg {av:+.3f} % · sum {st['sum']:+.2f} %" if n else "")
    due = n >= LITE_REVIEW_N and nd >= LITE_REVIEW_DAYS
    if n >= LITE_FLAG_N and av is not None and av < 0:
        return "flag", txt + (" · review due" if due else "")
    return ("review" if due else "collecting"), txt


def lite_atr_split(w):
    """LITE_ATR: closed fills split by the stamped entry ATR vs the frozen cap → [(label, stats)] (≤ cap · > cap · ATR ? when unstamped)."""
    atr = pd.to_numeric(w.atr, errors="coerce") if len(w) else pd.Series(dtype=float)
    out = [(f"ATR ≤ {LITE_ATR_CAP:g} %", lite_stats(w[atr <= LITE_ATR_CAP])), (f"ATR > {LITE_ATR_CAP:g} %", lite_stats(w[atr > LITE_ATR_CAP]))]
    unk = w[atr.isna()] if len(w) else w
    if len(unk):
        out.append(("ATR ? (no stamp)", lite_stats(unk)))
    return out


def lite_lines(w):
    f = lambda v, d=3: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.{d}f}"
    st, tx = lite_watch(w)
    L = ["## 🪶 FRENZY_LITE (DECISION_LOG 243 — declared exception, ARMED, no ATR cap, NO automatic off)", "",
         f"**FRENZY_LITE watch:** " + {"flag": "⚠ FLAG FOR OPERATOR REVIEW (avg < 0 at ≥ 20 closed fills — not an auto-off)",
                                       "review": "📋 REVIEW DUE", "collecting": "⏳ collecting"}[st] + f" ({tx}). "
         f"Bar: review at ≥ {LITE_REVIEW_N} closed fills on ≥ {LITE_REVIEW_DAYS} days; study (HOLD_LOWVOL_EARLY, STACK − F1): 724 · +0.163 %/trade, "
         "day CI [−0.107, +0.421]; REACHABLE by the engine ≈ 696 fills at +0.134 %/trade (the FRENZY shortlist misses 21 fills avg +1.49 %, "
         "the pair-level stretch id refuses 8 re-anchors avg −1.23 %) BEFORE the 30–50 % haircut (DECISION_LOG 243).", "",
         f"**LITE_ATR (observe-only — stamped entry ATR vs {LITE_ATR_CAP:g} %, frozen; study: ATR > 2.5 −0.20 %/trade, not proven):**", "",
         "| ATR at entry | N | Days | WR | Avg % | Sum % |", "|---|---|---|---|---|---|"]
    for lab, s_ in lite_atr_split(w):
        wr = "–" if s_["wr"] is None else f"{s_['wr']:.0f} %"
        L.append(f"| {lab} | {s_['n']} | {s_['days']} | {wr} | {f(s_['avg'])} | {f(s_['sum'], 2)} |")
    L += lite_lev_lines(w)
    op = int((~w.closed.astype(bool)).sum()) if len(w) else 0
    if op:
        L.append(f"_{op} open FRENZY_LITE fill(s) not counted yet._")
    return L + [""]


def lite_lev_era(w):
    """size eras on CLOSED fills → (old_stats, new_stats, flag, first_n_avg): new = fills sized at lev mult ≥ LITE_LEV_NEW (the 0.32 era,
    DECISION_LOG 247); flag = the FIRST LITE_LEV_FLAG_N closed new-era fills (by open time) average < 0 — an operator-review flag only."""
    if not len(w):
        e = lite_stats(w)
        return e, e, False, None
    lev = pd.to_numeric(w.lev, errors="coerce") if "lev" in w else pd.Series(np.nan, index=w.index)
    new = w[lev >= LITE_LEV_NEW]
    old = w[~(lev >= LITE_LEV_NEW)]
    # the cohort is FIXED: the first LITE_LEV_FLAG_N new-era fills by OPEN time (open ones included); judged only once all of them closed (review)
    first = new.assign(_a=pd.to_numeric(new.actual, errors="coerce")).sort_values("k", kind="stable").head(LITE_LEV_FLAG_N)
    done = len(first) >= LITE_LEV_FLAG_N and first._a.notna().all()
    fav = first._a.mean() if done else None
    flag = bool(done and fav < 0)
    return lite_stats(old), lite_stats(new), bool(flag), fav


def lite_lev_lines(w):
    f = lambda v, d=3: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{float(v):+.{d}f}"
    o, n, flag, fav = lite_lev_era(w)
    st = (f"⚠ REVIEW FLAG — the first {LITE_LEV_FLAG_N} fills at 0.32 average {f(fav)} % < 0 (operator decides; NOT an automatic revert)" if flag
          else (f"✅ the first {LITE_LEV_FLAG_N} fills at 0.32 average {f(fav)} % ≥ 0 — no flag" if fav is not None
                else f"⏳ {min(n['n'], LITE_LEV_FLAG_N)}/{LITE_LEV_FLAG_N} fills at 0.32 closed"))
    L = ["", f"**LITE size eras (DECISION_LOG 247 — lev 0.2 → 0.32, operator override at 3 fills; review flag if the first {LITE_LEV_FLAG_N} at "
         f"0.32 average < 0):** {st}", "", "| Size | N | Days | WR | Avg % | Sum % |", "|---|---|---|---|---|---|"]
    for lab, s_ in (("lev 0.2 (to 10-07)", o), ("lev ≥ 0.3 (0.32 from 10-07)", n)):
        wr = "–" if s_["wr"] is None else f"{s_['wr']:.0f} %"
        L.append(f"| {lab} | {s_['n']} | {s_['days']} | {wr} | {f(s_['avg'])} | {f(s_['sum'], 2)} |")
    return L


def lite_run(now_ms=None):
    """tracker 12: merge the exports into the store, save (validated: unique keys), → markdown lines."""
    old = _load_csv(LITE_CSV, ["k", "pair"]) if os.path.exists(LITE_CSV) else pd.DataFrame(columns=list(LITE_COLS))
    allr = lite_merge(old, _lite_fills())
    if len(allr):
        if allr.duplicated(["k", "pair"]).any():
            raise ValueError("duplicate LITE rows")
        _save_csv(allr, LITE_CSV)
    if not len(allr):
        return ["## 🪶 FRENZY_LITE (DECISION_LOG 243)", "", "No FRENZY_LITE fill in the exports yet (watch: review at ≥ 40 fills on ≥ 15 days; "
                "avg < 0 at ≥ 20 → operator review; LITE_ATR split ≤ / > 2.5 %).", ""]
    return lite_lines(allr)


# ─────────────────────────── 🪶 tracker 13 (LITE_GVOL24_LOW, operator-approved observe line) ───────────────────────────
LG_CSV = os.path.join(ROOT, "reports", "SCOUT_FRENZY_LITE_GVOL24.csv")   # write-once per fill (final rows never recomputed)
LG_CUT = 0.996              # FROZEN (reports/FRENZY_LITE_N4_REGIME_2026-10-07.md §6: the median of the windows with fills) — never re-fit
LG_WIN_MS = 4 * H           # the fill's 4 h UTC window; the reading is taken at its START
LG_TIME_S = 60              # polite: ≤ 60 s of public-kline reads per run
LG_TRIES = 3                # unreadable → retried next run, at most this many runs, then stored as 'unreadable'
LG_PRESEL = 80              # pre-selection: the 80 eligible pairs with the largest 48 h quote volume before W (see lite_gv24_run)
LG_REVIEW_N, LG_REVIEW_DAYS, LG_GAP = 30, 15, 0.4
LG_YEAR = ("study (§6, 724 N=12 fills): HIGH > 0.996 373 · +0.532 %/fill · WR 59.5 % vs LOW ≤ 0.996 351 · −0.230 · WR 51.6 % "
           "(both halves: Jan–Apr +0.82, May–Oct +0.72)")


class _RateLimited(Exception):
    pass


class _Budget(Exception):
    pass


def gvdash_at(bars_by_pair, onboard, W):
    """the study's dashboard-style market volume averaged over the 24 h before W (scratchpad/manualall/gvolyear.py → lite_regime/macro.py):
    per CLOSED 5m bar t: universe = pairs with that bar, a full 288-bar quote-volume history (vol24) and onboard < t + 5 min − 90 d; top 50
    by vol24; of those, the ones with a 48-bar base-volume mean > 0 (≥ 30 needed) → Σ EMA5(base vol) ÷ Σ SMA48(base vol) (EMA span 5,
    adjust False, over the fetched history). The value at W = the mean of the per-bar ratios of the 288 bars opening W − 24 h … W − 5 min
    (≥ 200 readable, else None). bars_by_pair = {pair: [[open_ms, base_vol, quote_vol], …]} ascending; onboard = {pair: ms or 0}. Pure."""
    try:
        opens = list(range(int(W) - 288 * BAR, int(W), BAR))
        cols = {}
        for p, rows in (bars_by_pair or {}).items():
            d = pd.DataFrame(rows, columns=["t", "v", "q"]).drop_duplicates("t").set_index("t").sort_index()
            if not len(d):
                continue
            d = d.reindex(range(int(d.index[0]), int(W), BAR))   # missing bars stay NaN (as the study's matrix)
            cols[p] = pd.DataFrame(dict(v=d.v.astype(float), q=d.q.astype(float),
                                        a48=d.v.astype(float).rolling(48, min_periods=48).mean(),
                                        e5=d.v.astype(float).ewm(span=5, adjust=False).mean(),
                                        v24=d.q.astype(float).rolling(288, min_periods=288).sum()))
        out = []
        for t in opens:
            cand = []
            for p, c in cols.items():
                if t not in c.index:
                    continue
                r = c.loc[t]
                ob = int((onboard or {}).get(p) or 0)
                if np.isfinite(r.v24) and np.isfinite(r.v) and (ob == 0 or ob < t + BAR - 90 * 86_400_000):
                    cand.append((float(r.v24), p))
            top = [p for _, p in sorted(cand, reverse=True)[:50]]
            ok = [p for p in top if np.isfinite(cols[p].loc[t, "a48"]) and cols[p].loc[t, "a48"] > 0]
            if len(ok) < 30:
                continue
            out.append(sum(float(cols[p].loc[t, "e5"]) for p in ok) / sum(float(cols[p].loc[t, "a48"]) for p in ok))
        return (float(np.mean(out)), len(out)) if len(out) >= 200 else (None, len(out))
    except Exception:
        return None, 0


def lite_gv24_stats(w):
    """→ dict per side (n, days, wr, avg) + the day-block 95 % CI of LOW − HIGH (resample DAYS; None when < 3 days or a side is empty)."""
    w = w[pd.to_numeric(w.gv24, errors="coerce").notna() & pd.to_numeric(w.actual, errors="coerce").notna()].copy()
    w["gv24"] = pd.to_numeric(w.gv24); w["actual"] = pd.to_numeric(w.actual); w["low"] = w.gv24 <= LG_CUT
    side = lambda g: dict(n=len(g), days=g.day.nunique(), wr=((g.actual > 0).mean() * 100 if len(g) else None), avg=(g.actual.mean() if len(g) else None))
    lo, hi = side(w[w.low]), side(w[~w.low])
    ci = None
    days = w.day.unique()
    if lo["n"] and hi["n"] and len(days) >= 3:
        rng = np.random.default_rng(7); by = {d: g for d, g in w.groupby("day")}; diffs = []
        for _ in range(3000):
            g = pd.concat([by[d] for d in rng.choice(days, len(days))])
            if g.low.any() and (~g.low).any():
                diffs.append(g[g.low].actual.mean() - g[~g.low].actual.mean())
        if len(diffs) >= 100:
            ci = tuple(np.percentile(diffs, [2.5, 97.5]))
    return lo, hi, ci


def lite_gv24_check(w):
    """the frozen bars → (state, text). review: LOW ≥ 30 fills on ≥ 15 days ∧ LOW avg ≤ HIGH avg − 0.4 ∧ LOW − HIGH day CI upper < 0 →
    propose arming as a sleeve on/off switch (operator decision) · retire: ≥ 30 LOW fills ∧ LOW − HIGH ≥ 0 · else collecting."""
    lo, hi, ci = lite_gv24_stats(w)
    f = lambda v: "–" if v is None else f"{v:+.3f}"
    gap = (lo["avg"] - hi["avg"]) if (lo["avg"] is not None and hi["avg"] is not None) else None
    txt = (f"LOW {lo['n']}/{LG_REVIEW_N} fills · {lo['days']}/{LG_REVIEW_DAYS} days · avg {f(lo['avg'])} vs HIGH {hi['n']} · avg {f(hi['avg'])}"
           + (f" · LOW − HIGH {gap:+.3f}" if gap is not None else "") + (f" · day CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ""))
    if (lo["n"] >= LG_REVIEW_N and lo["days"] >= LG_REVIEW_DAYS and gap is not None and gap <= -LG_GAP and ci and ci[1] < 0):
        return "review", txt
    if lo["n"] >= LG_REVIEW_N and gap is not None and gap >= 0:
        return "retire", txt
    return "collecting", txt


def _lg_get(url, deadline):
    if time.time() > deadline:
        raise _Budget("budget")
    try:
        return json.loads(urllib.request.urlopen(url, timeout=20).read())
    except urllib.error.HTTPError as ex:
        if ex.code in (418, 429):
            raise _RateLimited(str(ex.code))
        raise


def _lg_window(W, deadline):
    """public klines → (gv24, n bars) for the window start W (raises _RateLimited / _Budget; other trouble — a socket timeout included —
    raises too and the caller stores (None, 0) = one try used)."""
    from concurrent.futures import ThreadPoolExecutor
    info = _lg_get("https://fapi.binance.com/fapi/v1/exchangeInfo", deadline)
    elig = {x["symbol"]: int(x.get("onboardDate") or 0) for x in info.get("symbols", [])
            if x.get("contractType") == "PERPETUAL" and x.get("quoteAsset") == "USDT" and x.get("underlyingType") == "COIN"
            and not any(t == "Alpha" for t in (x.get("underlyingSubType") or []))
            and (int(x.get("onboardDate") or 0) == 0 or int(x.get("onboardDate") or 0) < W - 90 * 86_400_000)}

    def h48(sym):   # one symbol's trouble (a 400 on a settling contract, a timeout) = that symbol unread; budget / rate limit propagate
        q = urllib.parse.urlencode(dict(symbol=sym, interval="1h", endTime=int(W) - 1, limit=48))
        try:
            r = _lg_get(f"https://fapi.binance.com/fapi/v1/klines?{q}", deadline)
        except (_RateLimited, _Budget):
            raise
        except Exception:
            return sym, 0.0
        return sym, sum(float(x[7]) for x in r) if r else 0.0

    def k5(sym):
        q = urllib.parse.urlencode(dict(symbol=sym, interval="5m", endTime=int(W) - 1, limit=1500))
        try:
            r = _lg_get(f"https://fapi.binance.com/fapi/v1/klines?{q}", deadline)
        except (_RateLimited, _Budget):
            raise
        except Exception:
            return sym, []
        return sym, [[int(x[0]), float(x[5]), float(x[7])] for x in r if int(x[0]) + BAR <= W]
    with ThreadPoolExecutor(8) as ex:
        vol = dict(ex.map(h48, sorted(elig)))
        pre = [p for p, _ in sorted(vol.items(), key=lambda kv: -kv[1])[:LG_PRESEL]]
        bars = dict(ex.map(k5, pre))
    return gvdash_at(bars, {p: elig.get(p, 0) for p in bars}, W)


def lite_gv24_run(now_ms=None):
    """tracker 13: every CLOSED FRENZY_LITE fill (tracker 12's store) gets its 24 h dashboard market volume at the start of its 4 h UTC
    window, once (write-once rows; unreadable → retried ≤ 3 runs). Public klines, ≤ 60 s per run, stops on HTTP 418 / 429. APPROXIMATION
    (stated): the per-bar top-50 is chosen inside the 80 eligible pairs with the largest 48 h quote volume before W (the study ranked the
    whole cached universe) — a pair outside the top 80 by 48 h volume reaching the per-bar top 50 is rare. Observe-only: never changes trading."""
    now_ms = int(now_ms or time.time() * 1000)
    lite = _load_csv(LITE_CSV, ["k", "pair"]) if os.path.exists(LITE_CSV) else pd.DataFrame(columns=list(LITE_COLS))
    old = _load_csv(LG_CSV, ["k", "pair", "state"]) if os.path.exists(LG_CSV) else pd.DataFrame()
    rows = {(str(r.k), str(r.pair)): r._asdict() for r in old.itertuples(index=False)} if len(old) else {}
    deadline = time.time() + LG_TIME_S; cache = {}; stopped = None
    if len(lite):
        lite = lite[lite.closed.astype(str).isin(_TRUE + ("true",)) & pd.to_numeric(lite.actual, errors="coerce").notna()]
    for r in lite.itertuples(index=False):
        key = (str(r.k), str(r.pair)); cur = rows.get(key)
        if cur and str(cur.get("state")) in ("ok", "unreadable"):
            continue   # write-once
        W = int(pd.Timestamp(str(r.k), tz="UTC").value // 1_000_000) // LG_WIN_MS * LG_WIN_MS
        tries = int(float((cur or {}).get("tries") or 0))
        if stopped is None and W not in cache:
            try:
                cache[W] = _lg_window(W, deadline)
            except _RateLimited as ex:
                stopped = f"rate limited (HTTP {ex})"
            except _Budget:
                stopped = "60 s budget used"
            except Exception:
                cache[W] = (None, 0)
        if W not in cache:
            continue   # not tried this run (budget / rate limit) — no try counted
        gv, nb = cache[W]; tries += 1
        st = "ok" if gv is not None else ("unreadable" if tries >= LG_TRIES else "pending")
        rows[key] = dict(k=key[0], pair=key[1], day=str(r.k)[:10], window_start=pd.Timestamp(W, unit="ms").strftime("%Y-%m-%dT%H:%M"),
                         gv24=gv, bars=nb, tries=tries, state=st, actual=float(r.actual))
    allr = pd.DataFrame(list(rows.values()))
    if len(allr):
        if allr.duplicated(["k", "pair"]).any():
            raise ValueError("duplicate LITE_GVOL24 rows")
        _save_csv(allr.sort_values("k"), LG_CSV)
    L = [f"**LITE_GVOL24_LOW (observe-only — the 24 h dashboard market volume at the start of the fill's 4 h UTC window, split at {LG_CUT:g} frozen; "
         f"{LG_YEAR}):**", ""]
    ok = allr[allr.state.astype(str) == "ok"] if len(allr) else allr
    if not len(ok):
        L.append("No closed FRENZY_LITE fill with a reading yet.")
    else:
        lo, hi, _ = lite_gv24_stats(ok)
        fm = lambda v: "–" if v is None else f"{v:+.3f}"
        fw = lambda v: "–" if v is None else f"{v:.0f} %"
        L += ["| 24 h market volume | N | Days | WR | Avg % |", "|---|---|---|---|---|",
              f"| HIGH > {LG_CUT:g} | {hi['n']} | {hi['days']} | {fw(hi['wr'])} | {fm(hi['avg'])} |",
              f"| LOW ≤ {LG_CUT:g} | {lo['n']} | {lo['days']} | {fw(lo['wr'])} | {fm(lo['avg'])} |"]
        st, tx = lite_gv24_check(ok)
        L += ["", "**Status:** " + {"review": "📋 REVIEW — propose arming as a sleeve on/off switch (operator decision)",
                                    "retire": "❌ RETIRE (LOW not worse than HIGH at ≥ 30 LOW fills)", "collecting": "⏳ collecting"}[st] + f" ({tx})"]
    pend = int((allr.state.astype(str) == "pending").sum()) if len(allr) else 0
    unr = int((allr.state.astype(str) == "unreadable").sum()) if len(allr) else 0
    if pend or unr or stopped:
        L.append(f"_{pend} pending · {unr} unreadable after {LG_TRIES} tries" + (f" · stopped this run: {stopped}" if stopped else "") + "._")
    return L + [""]


def _extras(now_ms, th, F, J, allr):
    """trackers 7 – 10 after the exit table, each in its own try/except (one never breaks another or the scout)."""
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
    try:
        out += ons_run(now_ms, th, Je, F)
    except Exception as ex:
        out += ["## ⚡ ON_SCALP", "", f"Unavailable this run ({str(ex)[:120]}).", ""]
    try:   # 🪶 Oct-7 (243) FRENZY_LITE watch + LITE_ATR
        out += lite_run(now_ms)
    except Exception as ex:
        out += ["## 🪶 FRENZY_LITE", "", f"Unavailable this run ({str(ex)[:120]}).", ""]
    try:   # 🪶 Oct-7 tracker 13 LITE_GVOL24_LOW (operator-approved observe line)
        out += lite_gv24_run(now_ms)
    except Exception as ex:
        out += [f"_LITE_GVOL24_LOW unavailable this run ({str(ex)[:120]})._", ""]
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
    # 🛟 HYBRID_EXIT (V3): −3 · +0.2 floor from a +1 prior peak · the lock from +3; ticks and 1m
    chk(hyb_line(0.99) == -3.0 and hyb_line(1.0) == 0.2 and hyb_line(2.9) == 0.2 and hyb_line(3.0) == 2.0 and hyb_line(6.5) == 4.5, "V3 lines")
    h_ = walk_ticks_hyb(*tk([100, 101.5, 100.2]), 100, t0)
    chk(h_[2] == "+0.2 floor" and abs(h_[0] - (0.2 - FEE)) < 1e-9, "ORCA shape: peak +1.41 then back → out at the +0.2 floor print")
    h_ = walk_ticks_hyb(*tk([100, 100.9, 96.9]), 100, t0)
    chk(h_[2] == "stop" and abs(h_[0] - (-3.1 - FEE)) < 1e-9, "a +0.81 peak never arms → the −3 stop")
    h_ = walk_ticks_hyb(*tk([100, 101.5, 104, 106, 103.8]), 100, t0)
    chk(h_[2] == "floor / trail" and abs(h_[0] - (3.8 - FEE)) < 1e-9, "from +3 the lock trails (peak 5.91 → line 3.91)")
    chk(walk_ticks_hyb(*tk([100, 101.5, 100.5]), 100, t0)[2] == "open", "above the +0.2 floor → open")
    chk(walk_ticks_hyb(np.array([t0, t0 + 1000, t0 + 800 * MIN]), np.array([100.0, 100.5, 90.0]), 100, t0)[2] == "12 h cap", "12 h clock cap")
    hm = [[t0, 100, 101.5, 100, 101.2], [t0 + MIN, 101.2, 101.3, 99.0, 99.5]]
    chk(walk(hm, 100, "HYB")[2] == "+0.2 floor" and abs(walk(hm, 100, "HYB")[0] - 0.2) < 1e-9 and walk(hm, 100, "LOCK2")[2] == "open",
        "1m: the +0.2 floor set by a prior minute fills at the line; the lock still holds")
    chk(walk([[t0, 100, 101.5, 96.0, 97]], 100, "HYB")[2] == "stop", "1m: low before high inside a minute → the peak of that minute cannot arm it")
    hw = pd.DataFrame(dict(day=[f"d{i % 20}" for i in range(40)], d=[0.5, 0.4, 0.6, 0.3] * 10))
    chk(hyb_check(hw)[0] == "reopen" and hyb_check(hw.head(39))[0] == "collecting" and hyb_check(hw.assign(day="d0"))[0] == "collecting",
        "re-open at ≥ 40 on ≥ 20 days with Δ > +0.30, CI low > 0, > 0 w/o top 5; else collecting")
    chk(hyb_check(hw.assign(d=0.25))[0] == "holds" and hyb_check(hw.assign(d=[30.0] * 5 + [-0.5] * 35))[0] == "holds", "Δ ≤ +0.30 / a top-5 lottery → lock holds")
    an = pd.DataFrame(dict(HYB_R=[0.11, 0.11, 3.5, -3.1], LOCK2_R=[-3.09, 3.9, 3.5, -3.09], HYB_how=["+0.2 floor", "+0.2 floor", "floor / trail", "stop"]))
    a_ = hyb_anatomy(an)
    chk(a_[0] == 1 and abs(a_[1] - 3.2) < 1e-9 and a_[2] == 1 and abs(a_[3] + 3.79) < 1e-9, "SAVED / CUT anatomy")
    # ⚡ ON_SCALP: TP +3 net / 2 h / no stop, costs 0.19, 1m o → l → h → c, the 60-s flow, the frozen bar + the 0.2 book
    o_ = onscalp_walk(*tv([100, 95, 99, 103.3, 104]), 100, t0)
    chk(o_["how"] == "TP +3" and abs(o_["pnl"] - (3.3 - 0.19)) < 1e-9 and abs(o_["mae"] - (-5.19)) < 1e-9, "TP at the first print ≥ +3 net (fill = that print), no stop through −5")
    o_ = onscalp_walk(*tv([100, 101, 102]), 100, t0)
    chk(o_["how"] == "open" and not o_["hit"], "under 2 h and under +3 → open")
    o_ = onscalp_walk(np.array([t0, t0 + 60 * MIN, t0 + 120 * MIN, t0 + 121 * MIN]), np.array([100.0, 90.0, 97.0, 110.0]), 100, t0)
    chk(o_["how"] == "2 h" and abs(o_["pnl"] - (-3.19)) < 1e-9 and o_["exit_ms"] == t0 + 120 * MIN, "time exit at the first print ≥ entry + 2 h (a later pop is not counted)")
    o_ = onscalp_walk(*tv([100, 104]), 100, t0, gap=True)
    chk(o_["pnl"] == ONS_TP and o_["mfe"] >= 3.8, "1m pseudo prints: the TP fills at exactly +3")
    mt, mp = m1_prints_lh([[t0, 100, 104, 95, 101, 5]])
    chk(list(mp) == [100, 95, 104, 101] and list(mt - t0) == [0, 15_000, 30_000, 59_999], "1m ON-scalp prints: open → low → high → close")
    chk(onscalp_walk(*m1_prints_lh([[t0, 100, 104, 95, 101, 5]]), 100, t0, gap=True)["mae"] < -5, "the low is printed before the high (conservative MAE)")
    fl = onscalp_flow([t0 + 1_000, t0 + 13_000, t0 + 30_000, t0 + 61_000], [100, 99, 102, 90], [1, 1, 2, 9], [False, True, False, False], [1, 2, 3, 1],
                      t0, 100.0, 50.0, 6000.0)
    chk(fl["n_agg_60"] == 3 and fl["n_trades_60"] == 6 and abs(fl["usd_60"] - 403) < 1e-9 and abs(fl["buy_share_60"] - 304 / 403) < 1e-9,
        "60-s flow: taker-buy share of $ (isBuyerMaker False = buyer took); a print after 60 s is out")
    chk(abs(fl["mdd_60"] + 1) < 1e-9 and abs(fl["mru_60"] - 2) < 1e-9 and abs(fl["px12_vs_close"] + 1) < 1e-9 and abs(fl["vol_x_24h"] - 8.06) < 1e-9
        and abs(fl["vol_x_norm"] - 4.03) < 1e-9, "60-s low / high vs the close, the +12 s print, $ vs the median minute and vs normal hour / 60")
    ow = pd.DataFrame(dict(day=[f"d{i % 16}" for i in range(32)], pair=[f"P{i}" for i in range(32)], pnl=[3.0, 2.9, -1.0, 3.0] * 8,
                           mae=[-1.0] * 32, hit=[True, True, False, True] * 8, entry_at=[f"2026-10-{8 + i // 4:02d}T{i % 4:02d}:00:00" for i in range(32)]))
    chk(onscalp_check(ow)[0] == "candidate", "N ≥ 30 on ≥ 15 days, CI low > 0, small DD → candidate (probe study only)")
    chk(onscalp_check(ow.assign(hit=[True, False, False, True] * 8))[0] == "collecting", "P(+3 within 2 h) 50 % < 70 % → no candidate (study leg)")
    chk("haircut" in onscalp_check(ow)[1] and "addition" in onscalp_check(ow.assign(pnl=[3.0, -3.5] * 16))[1], "haircut shown; retire labelled addition")
    chk(onscalp_check(ow.head(29))[0] == "collecting" and onscalp_check(ow.assign(day="d0"))[0] == "collecting", "< 30 fires / < 15 days → collecting")
    chk(onscalp_check(ow.assign(pnl=[3.0, -3.5] * 16))[0] == "retire", "mean ≤ 0 at 30 → retire")
    chk(onscalp_check(ow.assign(mae=[-1.0] * 31 + [-30.0]))[0] == "candidate" and onscalp_book(ow.assign(mae=[-30.0] * 32))[1] >= 50,
        "liquidation-deep MAE books −23.94 at 0.94 × equity; repeated → DD ≥ 50 %")
    lq = ow.assign(pnl=[3.0, 2.9, -1.0, 3.0] * 8, mae=[-1.0, -1.0, -25.0, -1.0] * 8)
    chk(onscalp_check(lq)[0] == "collecting" and "max DD" in onscalp_check(lq)[1], "a positive-mean cohort whose book draws down ≥ 50 % is no candidate")
    chk(onscalp_check(pd.concat([lq, lq.assign(day=lq.day + "b")]))[0] == "retire", "no verdict by 60 → retire")
    chk(onscalp_universe("XUSDT", "FRENZY_VOL24_LOW", None, SimpleNamespace()) == "VOL24_LOW"
        and onscalp_universe("XUSDT", "LIVE_ONLY", "FRENZY_GVOL_HIGH;FRENZY_VOL24_LOW", SimpleNamespace()) == "VOL24_LOW"
        and onscalp_universe("XUSDT", "FRENZY_READY", None, SimpleNamespace(frenzy_pair_blacklist="ABC, xusdt")) == "BLACKLIST"
        and onscalp_universe("币安USDT", "FRENZY_READY", None, SimpleNamespace()) == "NON_ASCII"
        and onscalp_universe("XUSDT", "FRENZY_ATR_HIGH", None, SimpleNamespace()) == "ok", "the study's universe tags")
    pf = onscalp_flow([t0 + 1_000, t0 + 5_000, t0 + 11_999, t0 + 30_000], [100, 101, 102, 99], [1, 1, 1, 1], [False, True, False, False], [1] * 4,
                      t0, 100.0, None, None)
    chk(pf["n_agg_pre"] == 3 and abs(pf["usd_pre"] - 303) < 1e-9 and abs(pf["buy_share_pre"] - 202 / 303) < 1e-9 and abs(pf["move_pre"] - 2.0) < 1e-9,
        "pre-entry [close, +12 s): buy share of $, $ volume, the last print's move")
    oc = pd.DataFrame(dict(k=["2026-10-08T01:00:00", "2026-10-08T00:30:00", "2026-10-08T02:00:00", "2026-10-06T23:00:00", "2026-10-08T03:00:00"],
                           pair=["X", "X", "Y", "Y", "Z"], spike_at=["2026-10-07T20:00:00", "2026-10-07T20:20:00", "2026-10-07T21:00:00",
                                                                    "2026-10-07T21:00:00", "2026-10-07T22:00:00"],
                           cohort=[True, True, True, False, True], is_on=True, strong=[True, True, True, True, False]))
    chk(list(onscalp_counted(oc)) == [False, True, True, False, False], "one per pair-episode (20-min drift merged), floor before the dedupe, strong only")
    chk(list(onscalp_counted(oc.assign(univ=["ok", "VOL24_LOW", "ok", "ok", "ok"]))) == [True, False, True, False, False],
        "a bar outside the universe is filtered BEFORE the dedupe (the next bar of the episode counts)")
    chk(list(onscalp_counted(oc.assign(spike_at=[None] * 5, strong=True))) == [True, True, True, False, True], "no spike (live-only NO_EPISODE) = its own episode")
    good = dict(is_on=True, strong=True, adx_delta=1.0, di_spread=2.0, final=True, pnl=3.1, exit_how="TP +3", entry_at=_iso(t0), exit_at=_iso(t0 + 5 * MIN),
                mae=-2.0, mfe=3.1, flow_state="ok", buy_share_60=0.6, usd_60=10.0, mdd_60=-1.0, mru_60=1.0)
    onscalp_validate(good, True); onscalp_validate(dict(is_on=False, not_on_reason="NOT_FLAGGED"), False)
    for bad_row, sf, why in ((dict(good, strong=False), True, "strong flag vs ADX / DI"), (dict(good, adx_delta=-1.0, strong=False), True, "wrong file"),
                             (dict(good, pnl=2.5), True, "TP below +3"), (dict(good, exit_at=_iso(t0 + 200 * MIN)), True, "TP after 2 h"),
                             (dict(good, exit_how="2 h", pnl=-1.0, exit_at=_iso(t0 + 119 * MIN)), True, "time exit before 2 h"),
                             (dict(good, mae=4.0), True, "MAE above the result"), (dict(good, buy_share_60=1.4), True, "buy share > 1"),
                             (dict(good, exit_how="open"), True, "final without a clean exit"), (dict(is_on=False), False, "non-ON without a reason")):
        try:
            onscalp_validate(bad_row, sf)
            chk(False, f"onscalp_validate must raise: {why}")
        except ValueError:
            chk(True, why)
    # 🪶 FRENZY_LITE watch + LITE_ATR (243)
    ex = pd.DataFrame(dict(opened_at=["2026-10-08 01:00:00.123", "2026-10-08 02:00:00", "2026-10-08 02:00:00", "2026-10-09 03:00:00"],
                           pair=["AUSDT", "BUSDT", "BUSDT", "CUSDT"], direction="LONG", entry_strategy="FRENZY_LITE",
                           status=["CLOSED", "OPEN", "CLOSED", "CLOSED"], pnl_percentage=[2.0, None, -3.0, 1.0], entry_atr_pct=[1.2, 3.0, 3.0, None]))
    lr = lite_rows(ex)
    chk(len(lr) == 3 and list(lr.k) == ["2026-10-08 01:00:00", "2026-10-08 02:00:00", "2026-10-09 03:00:00"], "one row per opened_at + pair (the later export row wins)")
    chk(bool(lr.closed.iloc[1]) and lr.actual.iloc[1] == -3.0, "an OPEN fill that closed in a newer export updates its row")
    sp = dict(lite_atr_split(lr))
    chk(sp["ATR ≤ 2.5 %"]["n"] == 1 and sp["ATR > 2.5 %"]["n"] == 1 and sp["ATR ? (no stamp)"]["n"] == 1, "ATR split ≤ / > 2.5 + unstamped on its own line")
    chk(abs(lite_stats(lr)["avg"]) < 1e-9 and lite_stats(lr)["days"] == 2 and abs(lite_stats(lr)["sum"]) < 1e-9, "N / days / avg / sum on closed fills")
    chk(len(lite_merge(lr.assign(closed=lr.closed.astype(str)), lr)) == 3, "store + export merge dedupes by key (stored booleans read back as text)")
    mk = lambda n, d, v: pd.DataFrame(dict(k=[f"2026-10-{8 + i % d:02d}T{i % 24:02d}:00:00" for i in range(n)], pair=[f"P{i}" for i in range(n)],
                                           day=[f"d{i % d}" for i in range(n)], closed=True, actual=v, atr=1.0))
    chk(lite_watch(mk(19, 10, -0.5))[0] == "collecting", "avg < 0 at 19 fills → still collecting")
    chk(lite_watch(mk(20, 10, -0.5))[0] == "flag", "avg < 0 at ≥ 20 fills → flag for operator review (never an auto-off)")
    chk(lite_watch(mk(40, 14, 0.3))[0] == "collecting" and lite_watch(mk(40, 15, 0.3))[0] == "review", "review due at ≥ 40 fills on ≥ 15 days")
    chk(lite_watch(mk(40, 15, -0.1))[0] == "flag" and "review due" in lite_watch(mk(40, 15, -0.1))[1], "a flag outranks a due review (both said)")
    chk(any("LITE_ATR" in x for x in lite_lines(lr)) and any("open FRENZY_LITE" in x for x in lite_lines(lr.assign(closed=[True, True, False]))),
        "LITE lines render (split table, open fills noted)")
    # 🪶 DECISION_LOG 247 size eras (lev 0.2 → 0.32): split by the fill's own lev mult; flag on the FIRST 10 new-era closed fills only
    era = pd.concat([mk(3, 3, 2.0).assign(lev=0.2), mk(12, 6, -0.5).assign(lev=0.32, k=[f"2026-10-20T{i:02d}:00:00" for i in range(12)])],
                    ignore_index=True)
    o_, n_, fl_, fav_ = lite_lev_era(era)
    chk(o_["n"] == 3 and n_["n"] == 12 and fl_ and abs(fav_ + 0.5) < 1e-9, "size eras split by lev; first 10 at 0.32 avg < 0 → review flag")
    era2 = era.copy(); era2.loc[era2.lev == 0.32, "actual"] = [0.5] * 10 + [-9.0, -9.0]
    chk(not lite_lev_era(era2)[2], "the flag reads the FIRST 10 at 0.32 only (later losers don't flag it)")
    era3 = era.copy(); era3.loc[era3.k == "2026-10-20T00:00:00", "actual"] = np.nan   # the earliest 0.32 fill still open
    chk(not lite_lev_era(era3)[2] and lite_lev_era(era3)[3] is None, "cohort = first 10 OPENED; judged only once all 10 closed (no shifting set)")
    chk(not lite_lev_era(mk(9, 3, -1.0).assign(lev=0.32))[2], "< 10 closed at 0.32 → no flag yet")
    chk(lite_lev_era(mk(3, 3, 1.0))[1]["n"] == 0, "rows without a lev stamp stay in the old era")
    chk(any("LITE size eras" in x for x in lite_lines(era)), "size-era table renders")
    # 🪶 tracker 13 LITE_GVOL24_LOW
    W0 = 1_790_000_000_000 // LG_WIN_MS * LG_WIN_MS
    syn = {f"P{j}USDT": [[W0 - (600 - i) * BAR, 100.0 + j, (100.0 + j) * 10] for i in range(600)] for j in range(40)}
    gv, nb = gvdash_at(syn, {}, W0)
    chk(gv is not None and abs(gv - 1.0) < 1e-9 and nb == 288, "flat volume → dashboard ratio 1.0 on all 288 bars")
    syn2 = {k: [r if r[0] < W0 - 2 * H else [r[0], r[1] * 3, r[2] * 3] for r in v] for k, v in syn.items()}
    chk(gvdash_at(syn2, {}, W0)[0] > 1.05, "a volume surge in the last 2 h lifts the 24 h mean")
    chk(gvdash_at({k: v for k, v in list(syn.items())[:29]}, {}, W0)[0] is None, "< 30 readable pairs → unreadable")
    chk(gvdash_at(syn, {k: W0 - 10 * 86_400_000 for k in syn}, W0)[0] is None, "pairs listed < 90 days → not in the universe")
    mkg = lambda lo_v, hi_v, n=32, d=16: pd.DataFrame(dict(day=[f"d{i % d}" for i in range(2 * n)], gv24=[0.9] * n + [1.1] * n,
                                                           actual=[lo_v + (0.3 if i % 2 else -0.3) for i in range(n)] + [hi_v + (0.3 if i % 2 else -0.3) for i in range(n)]))
    chk(lite_gv24_check(mkg(-0.3, 0.5))[0] == "review", "LOW clearly worse (≥ 30 on ≥ 15 days, gap ≤ −0.4, CI < 0) → review")
    chk(lite_gv24_check(mkg(0.3, 0.1))[0] == "retire", "LOW not worse at ≥ 30 LOW fills → retire")
    chk(lite_gv24_check(mkg(-0.3, 0.5, n=20))[0] == "collecting", "< 30 LOW fills → collecting")
    chk(lite_gv24_check(mkg(-0.3, 0.5, d=10))[0] == "collecting", "< 15 days → collecting (no review)")
    chk(lite_gv24_check(mkg(0.0, 0.2))[0] == "collecting", "gap −0.2 (not ≤ −0.4, not ≥ 0) → collecting")
    # 🌊 GVOL_BAND split (frozen bands, live value first, write-once freeze, the WIDE [1.0, 1.2) hypothesis)
    chk([gvb_band(v) for v in (1.0, 1.0999, 1.1, 1.2, 1.4999, 1.5, 2.0, 2.92, 0.98, None)] ==
        ["[1.0, 1.1)", "[1.0, 1.1)", "[1.1, 1.2)", "[1.2, 1.5)", "[1.2, 1.5)", "[1.5, 2.0)", "≥ 2.0", "≥ 2.0", "< 1.0", None], "frozen band edges [lo, hi)")
    kb = ["2026-10-08T01:00:00", "2026-10-08T02:00:00", "2026-10-08T03:00:00", "2026-10-08T04:00:00"]
    gz = pd.DataFrame(dict(k=kb, gate=["FRENZY_WIDE_GVOL_HIGH", "FRENZY_GVOL_HIGH", "FRENZY_GVOL_HIGH", GVB_PASSED], final=[False, False, True, True]))
    cms = [_ms(k) for k in kb]
    fz = gvb_freeze_gvol(gz, {cms[0]: (1.11, "live gate log", "X"), cms[3]: (0.9, "live fill stamp", "Y")}, {cms[0] - BAR: 1.6, cms[1] - BAR: 1.34, cms[2] - BAR: 2.5})
    chk(fz.gvol[0] == 1.11 and fz.gvol_band[0] == "[1.1, 1.2)" and fz.gvol_frozen[0] is True, "the bot's live reading wins and freezes at once")
    chk(fz.gvol_src[1] == "scout v2" and fz.gvol_frozen[1] is False and fz.gvol_frozen[2] is True, "a scout v2 value is provisional until the row is final")
    chk(fz.gvol[3] is None, "let-through rows get no band")
    fz2 = gvb_freeze_gvol(fz, {cms[2]: (1.05, "live gate log", "Z")}, {cms[0] - BAR: 3.0, cms[2] - BAR: 1.0})
    chk(fz2.gvol[0] == 1.11 and fz2.gvol[2] == 2.5, "a frozen value is never changed (later live / scout readings ignored)")
    chk(fz2.gvol[1] == 1.34 and fz2.gvol_frozen[1] is False, "a miss keeps the provisional value (never wiped)")
    fz3 = gvb_freeze_gvol(fz2.assign(final=True), {}, {})
    chk(fz3.gvol[1] == 1.34 and fz3.gvol_frozen[1] is True, "the provisional value freezes once the row is final")
    chk(gvb_hyp_hold("propose", 0.30) == "collecting (coverage)" and gvb_hyp_hold("stays", 0.25) == "stays" and gvb_hyp_hold("collecting", 0.9) == "collecting",
        "coverage hold: > 25 % unbanded WIDE rows holds a read")
    cv = gvb_coverage(pd.DataFrame(dict(sleeve=["WIDE", "WIDE", "LONG", "WIDE"], gvol=[1.1, None, None, 1.3])))
    chk(cv[0] == 2 and cv[1] == 4 and abs(cv[2] - 1 / 3) < 1e-9, "coverage: read / total, WIDE unread share")
    jl = pd.DataFrame(dict(t=["2026-10-08T01:00:00", "2026-10-08T01:00:00", "2026-10-08T02:00:00", "2026-10-06T02:00:00"], e="BLOCK",
                           pair=["A", "A", "B", "C"], gate=["FRENZY_LITE_GVOL_HIGH"] * 2 + ["FRENZY_LITE_GVOL_UNREAD", "FRENZY_LITE_GVOL_HIGH"], strategy=""))
    lb = gvb_lite_bands(jl, {_ms("2026-10-08T01:00:00"): (1.15, "live gate log", "X")}, {}, GC_FROM)
    chk(len(lb) == 2 and lb.band.iloc[0] == "[1.1, 1.2)" and lb.band.iloc[1] == "unread", "LITE refusals: one per (t, pair, gate), banded, from the floor")
    hd = [f"d{i}" for i in range(10) for _ in range(2)]
    chk(gvb_wide_hyp([0.8, -0.2] * 10, hd, 0.5)[0] == "propose", "band +0.30 > 0, spread, ≥ let-through − 0.30 → propose")
    chk(gvb_wide_hyp([0.8, -0.2] * 10, hd, 0.7)[0] == "stays", "band below the let-through mean − 0.30 → stays")
    chk(gvb_wide_hyp([-0.5, 0.1] * 10, hd, -1.0)[0] == "stays", "a losing band → stays")
    chk(gvb_wide_hyp([5.0] + [0.01] * 19, hd, 0.0)[0] == "stays", "one day ≥ 50 % of the gain → stays")
    chk(gvb_wide_hyp([0.8, -0.2] * 9, hd[:18], 0.5)[0] == "collecting" and gvb_wide_hyp([0.8] * 20, ["d0"] * 20, 0.5)[0] == "collecting",
        "< 20 signals / < 10 days → collecting")
    chk(gvb_wide_hyp([0.8, -0.2] * 10, hd, None)[0] == "stays" and "no let-through baseline" in gvb_wide_hyp([0.8, -0.2] * 10, hd, None)[1],
        "read reached, no WIDE let-through mean → stays at 1.0 (no let-through baseline)")
    chk(gvb_wide_hyp([0.8, -0.2] * 10, hd, 0.5)[1] == gvb_wide_hyp([0.8, -0.2] * 10, hd, 0.5)[1], "fixed-seed bootstrap is deterministic")
    tb = gvb_band_table(pd.DataFrame(dict(sleeve=["WIDE", "WIDE", "LONG"], gvol_band=["[1.0, 1.1)", "[1.0, 1.1)", None], LOCK=[1.0, -3.0, 2.0], day=["a", "b", "a"])))
    chk(any(x.startswith("| FRENZY_WIDE | [1.0, 1.1) | 2 | 2 | 50 %") for x in tb) and any("| FRENZY_LONG | unread | 1 |" in x for x in tb), "band table per sleeve × band")
    print(f"selftest OK ({ok} checks)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run()))
