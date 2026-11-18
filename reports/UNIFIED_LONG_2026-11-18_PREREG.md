# UNIFIED LONG — PRE-REGISTRATION (frozen 2026-11-18 15:44 UTC (file time), before any result of this study was computed)

Research only. No bot code, config, templates, tests or CLAUDE files are touched; nothing is committed.
Scripts: `scripts/unified_long_*.py`. Data: `reports/UNIFIED_LONG_data/`. Report: `reports/UNIFIED_LONG_2026-11-18.md`.
The parallel SHORT study (`reports/UNIFIED_SHORT_*`) is not read or touched.

## Question (operator)
"~42 pumps and dumps a day, only 1–3 long trades a day. There is something for LONGs — maybe a better COMBINATION of the
WILLY / FRENZY_LONG / LITE / WIDE filters." Goal: a LONG sleeve on Frenzy / pump coins with ≥ 5 trades/day and a positive,
consistent edge after real costs — or an evidenced "no".

## Theory (written before the data)
Why a long on a pump coin can pay, and what should separate winners from losers:
1. **Continuation while demand persists** (FRENZY_LONG / LITE / WIDE): price holding above the spike-anchored VWAP with volume
   still high means spike buyers are in profit and new buyers keep coming → expect + when vs_vwap > 0, streak long, volume decay small.
2. **Bounce after the first red bar** (WILLY): early after the spike, the first pullback is bought → expect + when the pullback is
   shallow and the spike volume is fresh; − when the spike already failed (close ≤ VWAP).
3. **Squeeze fuel**: crowded shorts (global / top L/S low, funding ≤ 0, OI rising into the move) → continuation; crowded longs
   (L/S high, funding high) → long losers.
4. **Spot-led demand** (spot share of volume rising) = real buying → +; perp-only pump → fragile.
5. **Market tide**: BTC up / BTC daily uptrend helps alt longs; many simultaneous Frenzies = risk-on.
6. **Against**: late entries (many hours after the spike, deep off the peak, volume decayed) are distribution → losers.
Every one of these is tested; none is assumed.

## Candidate pool (the UNION — no later-shortlist selection)
**Universe F (Frenzy):** episodes of the validated live-engine walk (`study_frenzy_swing_short_walk.walk`, unchanged), sources
`EXHAUSTION_EXPAND_data/episodesD.pkl` (2023-H2), `SPOT_SHARE_DEEP_data/episodesC.pkl` (2024-01→05),
`LEAD1_VWAP_SHORT_data/episodes2024.pkl` (2024-06→12), `POST_FRENZY_SHORT_V2_data/episodes.pkl` (2025-01→2026-11-17), plus a new
walk for the never-read block if built. These lists keep verified episodes that were EVER shortlisted; to stay real-time a candidate
bar must already be shortlisted (`seen` so far) and flagged (verified ∧ ≤ 96 h) at its own close — an episode never shortlisted has no
eligible bar, so the list's selection does not leak.
**Candidate row = every closed 5m signal bar** of a flagged ∧ seen ∧ real (non-filled) bar whose close lies in [spike close − 5 min,
spike close + 24 h]; entry at the NEXT 5m open. Red / green is a feature, not a filter (WIDE takes green bars).
**Trigger flags rebuilt on every row** (services/frenzy.py rules, today's thresholds): `LONG` (fresh state-ON bar ∧ 24 h vol ≥ $20 M ∧
5m ATR ≤ 3 % ∧ red/flat bar) · `WIDE` (fresh ON ∧ green ∧ ATR ≤ cap ∧ streak > 12, the hold-green rule) · `LITE` (first judged
bar of an above-VWAP stretch: not in state, streak ≥ 12, vol× < 100, 2 h ≤ age ≤ 16.75 h, vol24 ≥ $20 M, red/flat) · `LITE_RETRY`
(the stretch's judged bar was green → first red/flat still-ready bar ≤ 15 min later) · `WILLY_A` (= WILLY_HT entry: first strictly-red
bar from the spike bar to spike + 60 min, pair already shortlisted at that bar, no earlier red bar missed) · `SURGE_LIKE` (BTC 30-min ≥
+0.3 % ∧ bar quote vol ≥ 5× its 20-bar mean ∧ close ≥ prior 24 h high) · `RED` (generic dip-buy base) · `ANY`.
Bull-Run is a BTC-regime universe sleeve (off since DL 291-era, `bullrun_sleeve_enabled` false); it is NOT rebuilt here (blind spot,
stated); its master fills are quoted as reference only.
**Universe M (pump coins outside Frenzy):** every 15m close of a pair that passes the engine shortlist (|24 h change| ≥ 15 % OR 24 h
range ≥ 15 %, 24 h vol ≥ $20 M) while NOT inside any Frenzy window of that pair, red 15m bar (dip-buy). Reported as its own family.

**Parity:** every rebuilt trigger vs real fills in `reports/MASTER_POOL_stacked.csv` (+ BASELINE orders): recall (real fill matched by
a same-pair flag row within ±15 min) and precision (flag rows inside sleeve-live periods that match a real fill). Precision is bounded by
slots / blocks / downtime — stated, not fixed.

## Features at the signal close (all lag-safe; metrics rows with create_time ≤ signal close − 5 min; funding settled ≤ close)
Episode / pair: hours since spike, run % (peak/base), gain % (close/base), off-peak %, vs VWAP %, above-streak, below-streak, above
share, vol× (1h vs normal), vol_decay (1h vol ÷ episode-max 1h vol), 24 h quote vol, turnover (1h ÷ 24 h), bar return, bar range,
ret 15m / 1h / 3h, 5m ATR %, 15m ATR %, ADX14 / DI spread 5m, EMA8–13 gap, EMA13–50 gap (5m), 1h EMA20 vs EMA200, RSI14 5m / 15m / 1h,
24 h range position, distance to the spike-bar high and to the episode high, red bars since spike, in_state, fresh_on, coin age (days
since its first 5m bar), episode # of the pair in 30 d, FAILED_SPIKE (close ≤ VWAP), WILLY_GAP (ADX > 55 or |EMA8−13| ≤ 0.9 %).
Derivatives: funding (last settled), OI change 1h / 24 h, OI ÷ 24 h vol, top-trader account L/S, top position L/S, global L/S (+ 24 h
changes), taker buy/sell ratio, spot share change (TA_WL definition) and has_spot.
Market: BTC ret 30 m / 1 h / 4 h / 24 h, BTC 1h EMA20 slope, BTC RSI 1h, BTC ADX 1h, regime label (services/regime.py on BTC 1h),
BTC 1 d EMA50 vs EMA200 gap (daily trend), # Frenzy spikes in the last 1 h / 6 h / 24 h, # episodes alive.
Blind spots (not buildable lag-safe here): Alpha tier history, market cap / turnover R, premium index, order book depth.

## ONE fixed exit family (comparable across all candidates)
TP 2 / 3 / 5 % (maker limit, filled only when the high trades ≥ 1 tick through) × SL 3 / 4 / 6 % (gap-aware: fill at min(open, stop))
× time 90 min / 4 h / 24 h (taker at the last bar's close) = 27 exits. Same-bar TP and SL → SL (conservative).
Costs: taker fee 0.05 % per taker side, maker 0.02 %, slippage 0.10 % per taker side (stress 0.20 %), real funding (longs pay positive
rates at settlements inside the hold; a missing pair-month is charged 0.05 %/8 h). One position per pair per rule (busy until exit).
**Primary (screening) exit: TP 3 / SL 4 / 4 h.** Every reported rule / decile is re-read on all 27 exits; the exit choice is part of the
family-wise null.

## Blocks
- **NEVER-READ: 2023-01-01 → 2023-06-30** (if the data + walk build; read ONCE in the final run, after the code is frozen).
- HOLDOUT-1: 2023-H2 and 2024 (already read by earlier SHORT studies and partly by long rebuilds — said so in the report).
- **DESIGN: 2025-01-01 → 2026-06-30** — the only block used to choose rules / cuts / exits.
- HOLDOUT-2: 2026-07-01 → 2026-11-17 (the bot's own live / paper period; partly read by sleeve studies — said so).
- Half-years by entry date for the consistency check.

## Combination search
**(a) Interpretable rules, DESIGN only.** Features cut at DESIGN deciles (10…90 %), both directions (x ≥ q, x ≤ q); trigger flags as
binary conditions. Stage 1: all 1D rules. Stage 2: EVERY pair of features × (quintile cuts 20/40/60/80, both directions). Stage 3:
greedy — the top 20 2D rules (by the objective) each extended by the best third condition. Base pools: F-ANY, F-RED, and each trigger.
**Objective (frozen):** mean net %/trade at the primary exit and 0.10 % slip, one position per pair, subject to DESIGN trades/day ≥ 5;
ties → higher worst DESIGN half. Also reported: the frontier at ≥ 2 / 3 / 5 / 8 / 10 trades/day.
**(b) Walk-forward model.** Own numpy gradient-boosted regression trees (`scripts/frenzy_hunt_gbt.Tree`, squared loss; depth 4, lr
0.05, 300 rounds, min leaf 500, row 0.5, col 0.8; target = net % at the primary exit clipped to [−8, +8]). Expanding window, refit on
the 1st of every month on rows whose EXIT closed before that month (24 h embargo). First scored month 2024-07 (≥ 12 months of
training, more if the never-read block exists — it is NOT used for training until it has been read). Trade a row when its score ≥ the
q80 / q90 / q95 of the scores of the training rows (thresholds frozen at each refit — causal; no daily top-k with future knowledge).
Deciles of the score reported per half-year. The model is a ranking, judged only out-of-sample.

## Pass bar (all required, for the chosen rule / model threshold)
1. trades/day ≥ 5 (pooled over the evaluated blocks) · 2. every half-year mean > 0 (N < 10 → fail) · 3. best-5 %-of-trades removed:
pooled mean > 0 · 4. pooled mean > 0 at 0.20 % slip · 5. family-wise null p < 0.05 (labels shuffled in whole-day blocks, the full
search re-run, max objective) · 6. beats random dip-buys: 20 random entries per trade from the same episode (any eligible bar), same
exit / costs, one-sided p < 0.05 (2,000 resamples) · 7. Kelly at the one-sided 95 % Wilson WR floor with n = distinct entry days > 0 ·
8. every holdout block mean > 0 (NEVER-READ included if built).
Frontier: trades/day vs avg/trade vs daily compound return at 0.18 × equity notional per trade (every calendar day), per half-year,
worst day, max drawdown.

## Decision rules (frozen)
- BUILD only if a rule or model threshold passes all 8. OBSERVE if it passes 1–4 and 8 but fails 5–7 (frozen definition, paper line).
- NO otherwise, listing what was tried and what failed. In-sample haircut 30–50 % on any DESIGN number quoted as an expectation.
- No result changes these rules; any deviation is labelled as post-hoc in the report.
