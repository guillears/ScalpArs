# WILLY high target — PRE-REGISTRATION (2026-10-23, written before any variant result)

**Question (operator 2026-10-23):** should FRENZY_WILLY's take profit aim at the spike high (or just below it) instead of a fixed +1.25 % net?
Motivating case: ARKUSDT WILLY 2026-10-23 13:25 UTC, entry 0.2125, TP 0.2154, spike high 0.2222 (+4.6 % above entry).

Research only. No tracked code / config / git change.

## Today's live exit (the comparator, read from trading_config.json + services/frenzy.py)
- TP +1.25 % net (`frenzy_willy_tp_pct`), resting maker limit → books exactly the line (`evaluate_late(book="live")`).
- Stop −4 % net (`frenzy_willy_stop_pct`).
- Late TP: from order age ≥ 30 min (`frenzy_willy_late_tp_minutes`) the TP is +0.25 % net (`frenzy_willy_late_tp_pct`).
- Time cap 90 min (`frenzy_willy_max_hold_minutes`).
- Walker: `scout_willy_timecap.evaluate_late` (the master / scout walker), net % = gross − 0.045 entry − 0.045 × exit/entry, on the
  hybrid tick → 1m path (`scout_frenzy_willy_exit.load_path`) for real fills, 1m paths for the backtest. All variants use the same
  walker, path and accounting, so every comparison is a paired Δ on the same fill.

## Spike high H (known at entry)
H = the highest traded price from the **spike bar's open** (`entry_frenzy_spike_at` − 5 min; `entry_frenzy_spike_at` = the spike
bar's CLOSE, `frenzy_walk` spike_ts) up to the **close of the entry red bar** (the last closed 5m bar before the fill = floor(fill time to
5 min)). This is exactly `frenzy_walk`'s `peak = max(h[on:])` at the entry pass. Source: tick prints when a full tick day exists, else
the 1m highs of the same span. Nothing after the red bar's close is used.
dist-to-high d = (H / E − 1) × 100, E = the fill price.

## Variants (all keep stop −4, the 90-min cap and the late-TP rule (+0.25 from minute 30) unless stated)
Target price T = H × (1 − k); its net % target = net(T, E) on the walker's accounting.
- **H1(k)**, k ∈ {0.5, 1, 2} %: target = clamp(net(T, E), floor = +1.25, cap = +8). Never lower than today.
- **H2(k)**, k ∈ {0.5, 1, 2} %: target = min(net(T, E), +8), no +1.25 floor; floored only at the late TP +0.25 (a target at or below the
  late TP would make the late rule meaningless and a TP ≤ 0 is not a take profit — stated here, before results).
- **H3**: split — half at today's +1.25, half at H1(k = 1 %)'s target. Result = mean of the two halves (each walked independently with
  the same stop / late TP / cap).
- **H4**: H1(k = 1 %) with the time cap extended to 180 min on fills whose target > +1.25 (fills at the floor keep 90).
- Diagnostic only (not ship-eligible): **H1(1)·noLate** — H1(k = 1 %) with the late TP switched off on the fills whose target > +1.25.
  Purpose: show how much of the far target the 30-min late TP takes away.

## Blocks
- ① live WILLY fills (live mode, B22–B23 + any later live export), dedup (opened_at[:19], pair, direction), CLOSED, operator-closed out
  (`scout_gatekit.split_operator_closed`), real fill prices.
- ② master per batch: `reports/MASTER_POOL_stacked.csv` FRENZY_WILLY (+ WILLY_HT if any), stack_keep, by era B19…B23.
- ③ backtest: engine replay WILLY fills, yr6a (2025) + yr6 (2026 → 10-11), seeds 1 / 2 / 3, trimmed to each chunk's window
  (`yr5_fills_trimmed.load(prefix=…)`, no warm-up duplicates), the replay's own fill price and time; 1m paths from the WILLY late-join
  1m cache (`reports/WILLY_LATE_JOIN_2026-10-22_data/k1m`), fills without a complete path counted and excluded. Read per seed and
  pooled; periods 2025 / 2026-H1 / 2026-H2; half-years 2025-H1 / 2025-H2 / 2026-H1 / 2026-H2.
- Calibration first: today's exit replayed on ① and ② fills vs booked `stack_pct` (exit class + % within 0.05); on ③ vs the replay's
  own pnl where the replay ran today's exit.

## Reads
- Per block / period: N · WR · avg % net · Σ · worst, today vs each variant, Δ.
- dist-to-high buckets: d ≤ 2 % / 2–5 % / > 5 %.
- Affected cohort = fills whose variant target ≠ +1.25. **Unaffected fills must show Δ = 0 exactly** (walker integrity). Within the
  affected: gained (variant reached its new target) vs lost (today's exit took +1.25 and the variant did not → late TP / stop / cap).
- Walk-forward: leave-one-half-out over the 4 backtest half-years — the variant (of the 8 ship-eligible) with the best mean Δ on the
  other three halves is applied to the held-out half; held-out Δ reported per half. Anchored forward (choose on all earlier halves) as a
  secondary line.
- Day-clustered bootstrap (5,000) 95 % CI of Δ vs today's exit, per block / period.
- 30–50 % in-sample haircut on any Δ projected from the data that chose the variant.

## Ship bar (frozen)
A variant is a CANDIDATE only if ALL hold:
1. its held-out Δ avg > 0 in ≥ 3 of the 4 leave-one-half-out halves (the LOHO-chosen variant), AND
2. backtest Δ > 0 in BOTH 2026-H1 and 2026-H2, AND
3. ② master Δ ≥ 0, AND
4. backtest 2026 (H1 + H2 pooled) day-bootstrap 95 % CI lower bound of Δ > 0.
Else keep +1.25. A candidate is still observe-first (OBSERVE-FIRST rule): shadow line on live fills before any arming.

## ADDENDUM (2026-10-23 ~14:00 UTC, operator via coordinator) — BTC-direction split
**Timing, stated honestly:** this addendum was written AFTER the un-split variant tables (blocks ①②③, walk-forward, ship bar) had been
computed and read, but BEFORE any result split by BTC direction was computed. The un-split ship-bar verdict is therefore already known
(no variant passes); the BTC split and the conditional rule below are judged on the same frozen bar, nothing re-fit.

- **BTC side** = exactly the WILLY_BTC1H scout feature (`scripts/scout_willy_btc1h.py`: BTCUSDT 5m closes → 1 h bars (label right,
  closed right), EMA20 adjust=False, slope = EMA20 of the last CLOSED 1 h bar at entry ÷ the previous − 1). ≤ 0 = "BTC falling",
  > 0 = "BTC not falling", unreadable = unscored (listed, in neither side). The scout computes it from BTC klines — it does NOT use the
  stamped `entry_btc_1h_slope`; the same computation is used on every block (stamp shown only as a cross-check on ①②).
- **Reads:** every variant before → after per block × BTC side; Δ per side with day-bootstrap 95 % CI; leave-one-half-out walk-forward
  per side (③); small N per side flagged (< 15 fills or < 8 days = screen only).
- **Conditional rule C(v)** = variant v when BTC not falling, today's +1.25 when BTC falling (and when unscored). Evaluated for every
  ship-eligible v on the SAME frozen ship bar (≥ 3 / 4 LOHO halves, both 2026 halves > 0, ② master Δ ≥ 0, ③ 2026 day-bootstrap CI
  lower bound > 0). Choosing among 8 conditional rules is a further multiple comparison: the LOHO choice is made among the 8 C(v) only.
  Market-wide variable → WINDOW-UNITS rule: day counts are reported beside N.
- **ARK 2026-10-23 13:25** (TP'd at 0.2154, +1.29 %, then traded to 0.2236) is added to ① live from public 1m klines (provisional, not in
  any export).

## ADDENDUM 2 (2026-10-23 ~14:10 UTC, operator via coordinator) — lower fixed targets on the BTC-falling side
**Timing:** written before ANY BTC-split result was computed (the split script had not been run yet).
- New fixed-target walks (stop −4, 90-min cap, late TP +0.25 from minute 30 unchanged): **F100** = TP +1.0 % net, **F075** = TP +0.75 % net.
- Conditional rules (BTC side = Addendum 1 feature; unscored fills → today's exit):
  - **C1**: not falling → high-anchored V*, falling → +1.25 (today)
  - **C2**: not falling → V*, falling → +1.0
  - **C3**: not falling → V*, falling → +0.75
  - **C4a**: not falling → +1.25, falling → +1.0 · **C4b**: not falling → +1.25, falling → +0.75 (no high anchor)
- **V* is fixed by walk-forward, never the in-sample best:** for each held-out ③ half, V* = the ship-eligible high-anchored variant
  (H1 / H2 / H3 / H4) with the best mean Δ on the BTC-not-falling fills of the OTHER three halves. Every ③ per-half Δ of C1–C3 is
  therefore out of sample. For ① / ② (never used to choose), V* = the variant chosen on all four ③ halves' not-falling fills.
- Same frozen ship bar per rule: held-out Δ > 0 in ≥ 3 / 4 halves · ③ 2026-H1 and 2026-H2 Δ > 0 · ② master Δ ≥ 0 · ③ 2026 day-bootstrap
  95 % CI lower bound of Δ > 0. Five rules (C1, C2, C3, C4a, C4b) = a multiple comparison; any pass is still observe-first.
- **Related lock:** the WILLY_BTC1H scout gate (DECISION_LOG 275 / 281, still collecting) is frozen and is NOT overridden by this study.
  The report states how many post-DL 281 BTC-falling WILLY fills exist (its counted cohort) and treats this study as information to it.

## ADDENDUM 3 (2026-10-23 ~14:20 UTC, operator via coordinator) — +0.5 on the BTC-falling side
**Timing, stated honestly:** written AFTER the Addendum 1 / 2 BTC-split results (C(v), C1–C4, F100 / F075) were computed and read, but
BEFORE any +0.5 result was computed. Judged on the same frozen bar; nothing else re-fit.
- **F050** = fixed TP +0.5 % net (stop −4, 90-min cap; the late TP +0.25 from minute 30 still applies since 0.25 < 0.5).
- **C4c**: not falling → +1.25, falling → +0.5 · **C5**: not falling → V* (walk-forward-fixed exactly as in Addendum 2), falling → +0.5.
- Falling-side target ladder reported: {+0.5, +0.75, +1.0, +1.25}.
- Fees: same model as every other cell (net = gross − 0.045 entry − 0.045 × exit/entry; the walker charges the taker rate on the TP even
  though the live resting TP pays maker 0.018 — a constant +0.027 understatement on every TP of every cell). Report gross vs net for the
  +0.5 cell (gross move needed ≈ +0.59 % at taker / ≈ +0.56 % at maker, so fees are ≈ 15–18 % of the gross TP move vs ≈ 7 % at +1.25).

## FEE CONVENTION NOTE (2026-10-23, operator correction — no rule change)
Every TP / stop in this study triggers on NET P&L and a TP books exactly its net target (+1.25 → +1.25), as live. Net = gross − 0.045
entry − 0.045 × exit/entry (walker). The live resting limit is solved with a MAKER exit (0.018), so the walker's TP price line sits
≈ 0.027 % above the live limit (slightly conservative trigger, same booking). A maker-line sensitivity run (trigger at the live
resting price, book the target) is reported beside the main read. No gross-vs-net framing for fixed targets.

## ADDENDUM 4 (2026-10-23 ~14:45 UTC, operator via coordinator) — stop grid on the BTC-falling side
**Timing, stated honestly:** written AFTER Addenda 1–3 results (BTC split, C1–C5, F100 / F075 / F050) were computed and read, and
BEFORE any result with a stop other than −4 was computed.
- Falling-side grid: TP {+0.5, +0.75, +1.0, +1.25} net × stop {−2, −2.5, −3, −4} net = 16 cells (90-min cap, late TP +0.25 from
  minute 30 unchanged; booking as above). Breakeven WR per cell = |stop| / (TP + |stop|) shown beside the observed WR.
- **Cell choice by walk-forward only:** for each held-out ③ half, the cell with the best mean Δ (vs today's +1.25 / −4) on the
  BTC-falling fills of the other three halves. ①② use the cell chosen on all four ③ halves' falling fills. Full grid reported per period.
- Smoothness: the chosen cell's Δ vs its grid neighbours (± one TP step, ± one stop step) — plateau or isolated peak.
- **Stop-width counterfactual (CLAUDE.md, two-sided):** at TP +1.25, each tighter stop (−3 / −2.5 / −2) vs −4 on falling fills, split
  into ① live-stopped (stopped at −4 today): loss shrunk · ② cut winners (today TP / late TP / cap ≥ 0 that the tighter stop now
  stops) · ③ untouched (never touched the tighter stop) → Δ must be 0 exactly. Day-block bootstrap CI per era (2025 / 26-H1 / 26-H2)
  and on ①②.
- **C6** = BTC not falling → the high-anchored variant ONLY if one passed on the not-falling side; none did (not-falling LOHO 2 / 4,
  read before this addendum) → +1.25 / −4 (today); BTC falling → the walk-forward (TP, stop) cell. C6 with the not-falling V* is shown
  as information only. Same frozen ship bar.
- **Family-wise null:** 16 cells × ~150 falling days. Null = the whole walk-forward procedure (LOHO cell choice, mean held-out Δ)
  re-run 300 times with the "falling" label moved to a random set of UTC days of equal fill count (day-block relabel); p = share of
  null held-out means ≥ the real one. A pass needs p ≤ 0.05 in addition to the ship bar.
