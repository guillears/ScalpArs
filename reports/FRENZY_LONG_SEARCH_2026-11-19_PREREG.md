# FRENZY LONG SEARCH — PRE-REGISTRATION (frozen 2026-11-19 14:25 UTC, before any result of this study was computed)

Research only. No bot code, config, templates, tests or CLAUDE files are touched (report-only commit).

**Timing disclosure (review, 2026-11-19):** the header stamp 14:25 UTC is inconsistent with the files: the episode build log ends 14:21 UTC and Addendum A (stamped 14:24 UTC) says the main-family (SURVIVOR / CARRY / BASE) results were already computed. Treat the main families as frozen around the build with their results possibly seen; none of them passed, so no verdict depends on it. Addenda A and B were frozen before their own results.
Scripts: `scripts/frenzy_long_search_*.py`. Data: `reports/FRENZY_LONG_SEARCH_data/`. Report: `reports/FRENZY_LONG_SEARCH_2026-11-19.md`.
WILLY_HT (dip-buy after the spike) is studied by another agent and is NOT re-tested here.

## Question (operator)
"Market is bullish and more coins are in frenzy, yet you keep saying you find no LONG opportunity. I'm 100 % sure there is something."
Audit our own null: what was never tested, does the current tape have historical analogs where longs paid, and is there a Frenzy
long family that pays after real costs.

## What prior work already covered (from the inventory; NOT re-run unchanged)
0–24 h after the spike, every 5m bar, every rebuilt trigger (LONG / WIDE / LITE / LITE_RETRY / WILLY_A / SURGE-like), 73 features
incl. OI / L/S / taker / funding / spot share / BTC / Frenzy counts, 1D/2D/greedy-3D, walk-forward model, 27 fixed TP/SL/≤24 h exits
(UNIFIED_LONG 11-18) · 58 single exits incl. trails, VWAP/spike-low invalidation (LONG_LEADS_DEEP LL3) · 8 regime switches × triggers
in window units with a calendar-shuffle null (REGIME_LONGS 11-18) · 336 TA cells (TA_CATALOG) · pre-spike / first 30 min
(EARLY_DETECTION) · 1 h / 4 h BREV signature (BREV_SIGNATURE) · breakouts / confirm-then-ride (HUNT_FOLLOWUP) · second push (Oct) ·
high-ATR L1–L4 (HIGH_ATR) · spot-led long mirror (SPOT_SHARE_DEEP) · lift-off / runaway (Oct) · hot scalp (Oct).

## The untested gaps this study targets
- **G1 Horizon.** Every prior long held ≤ 24 h (BREV ≤ 48 h) and capped the upside with a TP ≤ 8 %. The SQUEEZER study showed the
  72 h distribution after a second peak has a fat RIGHT tail (coins going 3–10×) and that squeezed coins pay the LONG side heavy
  funding (shorts paid −1.0 % / 24 h, −1.8 % / 72 h). A long that holds the tail and collects funding was never priced.
- **G2 Late entries / survivors.** Prior entries were 0–24 h after the spike. Coins that are still holding their pump 24–72 h later
  ("which pumps don't mean-revert") were never a long universe.
- **G3 The current tape** (Nov 2026) and its historical analogs were never characterised for longs.

## Theory (written before the data), testable variables and expected sign
(a) Continuation of a breakout with expanding volume / rising OI in a bid alt market — covered 0–24 h by UNIFIED; here only as the
    survivor's volume-not-decayed leg (expected +).
(b) Second leg after a shallow consolidation above VWAP → SURVIVOR family: a coin that 24 / 48 / 72 h after its spike still holds
    ≥ +20 % over the pre-spike price, closes above the spike-anchored VWAP and within 15 % of its post-spike high has absorbed the
    supply of the spike buyers (expected +, larger with BTC daily bull and high breadth).
(c) Squeeze fuel → CARRY family: deeply negative funding (shorts crowded) on a Frenzy coin; the long is paid funding every
    settlement and owns the squeeze tail (expected + over 72 h–7 d; expected − with a tight stop because the path is noisy).
(d) Spot-led demand and (e) market breadth / BTC daily bull: tested as SIGN splits (window units) on every family, not as fitted cuts.
Against: the mean-reversion base rate (pumps fade) — the null is the same coins at the same checkpoint without the condition.

## Universe, entries
- Episodes: `reports/UNIFIED_LONG_data/eps_all.pkl` (validated live-engine walk, 10,261 episodes, 643 pairs, spikes 2023-01 → 2026-11-17).
  Spike close = `spike_ms`. An entry needs the pair already shortlisted (`first_seen_ms` ≤ signal close), 24 h quote volume ≥ $20 M
  at the signal close, and real (non-filled) bars.
- Checkpoints: signal close = spike close + h, h ∈ {6, 24, 48, 72} hours; entry = the NEXT 5m open. One entry per episode per
  checkpoint; within a cell one open position per pair (later overlapping entries on the same pair are skipped).
- **BASE_h**: every checkpoint entry (the unconditional reference / null).
- **SURVIVOR_h** (h ∈ {24, 48, 72}): gain vs the pre-spike price (open of the bar 1 h before the spike) ≥ +20 % ∧ close ≥ anchored
  VWAP (from the spike bar) ∧ close ≥ 0.85 × the max high since the spike.
- **CARRY_h** (h ∈ {6, 24, 48}): last settled funding rate at the signal close, in 8 h-equivalent, ≤ −0.10 %.
- **SURVIVOR_CARRY_h**: both.

## Exits (frozen grid, 18) and the PRIMARY exit per family (frozen now, not re-picked)
hold {24 h, 72 h, 168 h} × stop {none (−95 % catastrophe), −20 %, −10 % gap-aware} × TP {none, +20 % maker}.
PRIMARY: SURVIVOR = 72 h / stop −20 % / no TP · CARRY = 72 h / no stop / no TP · SURVIVOR_CARRY = 72 h / stop −20 % / no TP ·
BASE = same as the family it is compared with.
Costs: taker 0.05 % in and out (maker 0.02 % on TP, high ≥ TP × 1.0005), slippage 0.10 % on the entry and every taker exit (stress
0.20 % and 0.50 %), stops gap-aware (fill = min(stop, bar open) − slip), real funding at every settlement held (long PAYS a positive
rate, RECEIVES a negative one; hourly coins included; a missing pair-month is charged 0.05 % / 8 h against the position).

## Blocks
DESIGN 2025-01-01 → 2026-06-30 · HOLDOUT-A 2023-01-01 → 2024-12-31 · HOLDOUT-B 2026-07-01 → 2026-11-19. Half-years reported.

## Regime splits (lag-safe: last CLOSED day before the entry), SIGN granularity, WINDOW units
R1 BTC daily EMA50 > EMA200 (LONG_LEADS_DEEP btc_daily gap) · R6 BTC prior-day return > 0 · R5 alt breadth: share of USDT perps
(REGIME_LONGS daily_closes, delisted incl.) above their daily EMA50 > 50 % · RF Frenzy concurrency: spikes in the prior 7 days ≥ its
pooled median. Evidence per regime cell is counted in distinct entry DAYS and distinct R1 windows, not trades.
Tape-analog: the current tape vector (14-day means of BTC gap, BTC 14 d return, breadth, Frenzy spikes/day, median alt 7 d return)
ending 2026-11-17; analog days = the 10 % nearest (z-scored Euclidean) days in 2023-01 → 2026-09; families read on analog vs other days.

## Mirror states (item 4), pre-registered composite on the UNIFIED_LONG 0–24 h rows (re-read, no re-fit)
MIRROR = taker1h ≥ 1.05 (buyers) ∧ top_pos change proxy: top_acc_chg24 > 0 (top traders adding longs) ∧ oi_chg24h > 0 ∧ fund_last ≤ 0
∧ vd ≥ 0.5 (volume still ≥ half its spike level) — expected + ; read at UNIFIED's 27 exits and a 24 h no-stop hold (ev288), per block.

## Entry × exit from the path
Every primary-cell entry gets REACH (max % up before the stop / horizon), WORKED (+1 % before −stop distance; −10 % for no-stop cells),
and a path label (NEVER_WORKED / GAVE_BACK / CAPTURED / STOPPED_RECOVERED) from its 5m path. Event study chart: mean / median path
−24 h → +168 h of the primary entries, winners vs losers, by R1 regime.

## Family-wise null
Statistic = the max over the primary cells (families × checkpoints) of the day-clustered t of DESIGN avg net. Null: within each
calendar month, the family condition is permuted across that month's BASE_h entries of the same checkpoint (keeps regime / month,
breaks the condition), 200 draws; p = share of draws whose max t ≥ observed.

## PASS bar (all required for a BUILD proposal)
1 DESIGN avg net > 0 with day-clustered 95 % CI lower bound > 0 · 2 beats the family-wise null (p < 0.05) · 3 HOLDOUT-A and HOLDOUT-B
avg > 0 · 4 positive in ≥ 2/3 of half-years (or a stated window-unit regime explanation) · 5 best-5 %-out avg > 0 and leave-one-pair-out
and leave-one-month-out min > 0 · 6 at 0.20 % slip > 0 · 7 Kelly at the one-sided 95 % Wilson WR floor > 0 · 8 no single day / pair
≥ 25 % of the net · 9 master agrees where it overlaps (no overlap is stated, not assumed). Then the 30–50 % haircut, one-size DCR
before → after (0.18 × equity notional, every calendar day), and capacity (book depth: not rebuildable historically — stated).
A cell passing 1–2 but failing others is at most an OBSERVE line with a frozen definition.

## ADDENDUM A — buy AT FIRST DETECTION (operator via coordinator, frozen 2026-11-19 14:24 UTC, before any FLAG / FAST result)
Note: the main-family results (SURVIVOR / CARRY / BASE) had already been computed when this addendum was written; the FLAG and FAST
families below had not been computed. Nothing in the main families is re-picked.
- **FLAG** (all 10,261 walk episodes, 5m): detection = max(spike close, first_seen_ms) — the first close where the engine has the
  verified spike AND the pair shortlisted; entry = next 5m open (≈ close + 8 s) + 0.10 % slip; 24 h vol ≥ $20 M.
  Detection lag: base = lowest close in the 6 h up to the spike bar; moved % = detection close ÷ base − 1; lag minutes = det − base bar close.
  Exits (frozen): Q3 TP 3 maker / SL 4 / 4 h · Q5 TP 5 / SL 5 / 24 h · TR arm +3 %, trail 3 % off the high, SL 5, 24 h · T1 1 h time ·
  T4 4 h time · H72 72 h no stop. PRIMARY = TR (the long-shaped exit the theory "catch it early and ride" implies).
  Splits: moved < 10 % / 10–20 % / > 20 % (theory: edge, if any, in the early-caught < 10 %), R1 / R6 / R5 / RF signs in window units.
  Hindsight bound HB: Q3 / Q5 / TR entered at the open after the base bar (unreachable; measures what the lag costs).
- **FAST** (faster detector, 1m, pairs with local 1m klines 2026-01-22 → 2026-10-02): 5-min return ≥ +3 % (D3) or ≥ +5 % (D5) ∧
  last-5-min quote volume ≥ 10× the median 5-min quote volume of the prior 24 h ∧ 24 h volume ≥ $20 M; 6 h debounce per pair; entry =
  the open of the 2nd 1m bar after the signal minute (≥ 60 s latency, realistic for a 1m-polling detector) + 0.10 % slip; same exits on
  1m bars. Read on all signals (no outcome selection) and, separately, paired with the FLAG entry of the same episode (descriptive).
- Same PASS bar as the main families; the FLAG / FAST family-wise null = the BASE-in-episode random-time comparison is NOT available at
  first detection, so the null is: day-block bootstrap of the FLAG primary cell's mean, and every split × exit cell counts in the family
  (Bonferroni over cells for item 2).

## ADDENDUM B — EARLIER-DETECTOR grid (operator via coordinator, frozen 2026-11-19 14:31 UTC, before any EARLY result)
Context stated openly: FLAG, FAST (1m) and FAST5 (5m) results had been read when this was written. The 1m cache was found to be
outcome-selected (5m events with full 1m coverage +4.98 % vs −3.91 % without), so this grid runs on **5m bars only** (full local
universe, delisted incl., 2022-11 → 2026-11), with entry at the next 5m open (= trigger close + ~8 s) + 0.10 % slip.
- Grid (27): window W ∈ {5, 15, 30} min (1 / 3 / 6 bars) × return over W ≥ {2, 3, 5} % × window quote volume ≥ {5, 10, 20}× the
  normal rate (engine normal hour: median 1 h quote volume over the 240 h ending 24 h before, scaled to W). 24 h vol ≥ $20 M; 6 h
  debounce per pair per cell.
- Exits (5): Q3 · Q5 · TR · T1 · T4 (FLAG definitions). 135 cells, all in the family-wise null.
- Null: random-timing — each signal's outcome is replaced by a random eligible bar of the same pair and calendar month (pool of up
  to 300 eligible bars per pair-month), 200 draws; statistic = max over the 135 cells of the day-clustered t of the DESIGN mean.
- Reads per cell: trades/day, coins/day, avg, DESIGN / HOLD-A / HOLD-B, best-5 %-out, Kelly floor, K = 10 book avg and DCR at 0.18 and
  at the Kelly floor; earliness vs today's flag (minutes before the walk spike close, for signals followed by a walk spike within 2 h),
  share of the 24 h move left at entry vs at the flag; false-positive rate = never +3 % above entry before −5 %.
- Regime: R1 and RF signs (window units) for the 5 best cells by DESIGN K10 DCR. PASS bar as the main families.
