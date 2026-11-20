# DIST_ENTRY_TIMING — pre-registration (frozen 2026-11-20, before any result is read)

Question: would DISTRIBUTION_SHORT (DL 586) do better entering EARLIER — (A) seconds earlier (less execution latency) or (B) 1–2 minutes
earlier, deciding on the still-forming 15m candle?
Motivation: KERNELUSDT DIST 2026-11-20, signal close 12:30:00, fill 12:30:27 @ 0.05114; price 1 min earlier ≈ 0.0514–0.0516.

## Fixed inputs
- Rule (DL 586): EXHAUSTION-tracked FRENZY coin (≤ 72 h after the spike, 24 h vol ≥ $20M), CLOSED red 15m candle, quote volume ≥ 5 × mean
  of the previous 20 candles, top-trader POSITION L/S 4 h change ≤ −0.01097 (km = last metrics row ≤ T − 5 min, ≤ 30 min old; k4 ≤ ts(km) − 4 h,
  ≤ 4 h 15 min), 1h RSI(14) (complete hours closed ≤ T) ≥ 74.07 → short at market.
- Backtest entry convention (unified_short_build / dist_exit_sim): entry_ms = T (the signal close); E = open of the first 1m bar at T
  (= the first trade after the close). Exit B: TP min(20 × ATR15%, 10 %) resting maker (1 tick through) · SL min(1.5 × ATR15%, 6 %) market,
  gap-aware · 24 h; costs = entry slip + 0.05 % taker + (stop/time: slip + 0.05 %) / (TP: 0.02 %) + real funding; slip = max(0.10, slip_model(q5)).
- Simulator: scripts/dist_exit_sim.run_rows with the REF_B cell only (unchanged code). Net formula = dist_exit_eval.net_mat. One position per
  pair (greedy, time order, busy until the simulated exit) = dist_exit_eval.one_pos.
- Population: reports/UNIFIED_SHORT_data/pool.pkl — every closed 15m candle of every FRENZY episode window (spike … +72 h) with base_ok
  (vol24 ≥ $20M, ATR defined). Candidate scan = every pool 15m row with full-candle vmult15 ≥ 5 (necessary: partial quote volume ≤ full
  quote volume over the same prior-20 mean, so any T−k volume trigger is in this set), any colour.

## Part A — latency
A1. Real fills (paper DIST OPEN rows in the decisions exports + the operator-reported KERNEL fill): delay = fill time − T; price at T (the
    decision row's candle close and the first aggTrade ≥ T); fill vs both, in bps, signed so + = cost to the short (fill below the T price).
    Sub-minute prices from data.binance.vision daily aggTrades zips (KERNEL 11-20: fapi aggTrades, ≤ a few calls).
A2. Backtest fills (REF_B kept set): entry at T+1 min and T+2 min (1m resolution limit: the bot's ~17–27 s cannot be represented by 1m
    bars; a sub-minute curve is measured on a random sample of ≤ 40 REF_B fills with aggTrades: price at T+0/5/10/20/30/60/90 s vs T+0).
    Same simulator, levels from the later fill. Reported as the per-fill bps cost of delay and Δ summed net %.

## Part B — earlier trigger (1–2 min)
At T−k (k = 1, 2), the partial candle = minutes [T−15, T−k) from 1m klines (quote volume = kline field 7). Legs at T−k:
  red_k = close(T−k) < open of the candle · vm_k = partial quote volume ÷ the prior-20 closed-candle mean ≥ 5 · pos_k = the same metrics
  rule evaluated at T' = T − k min (exact, recomputed from metrics) ≤ −0.01097 · rsi_k = RSI(14) on complete hours closed ≤ T − k (exact;
  differs from T only for candles closing at :00) ≥ 74.07 · vol24_k = 95 prior closed candles + partial quote volume ≥ $20M ·
  ATR_k = Wilder ATR(14) with the partial candle as the last bar (h/l/c so far) ÷ the entry price. Episode window = the closed row's (approx.).
Entry at T−k = open of the 1m bar at T−k (= close of the last partial minute).
Variants (all one position per pair, Exit B, same sim):
  V0  = T0 baseline (replica of REF_B; parity reported per fill: same (pair, entry_ms) set and |net Δ| < 0.01).
  VE1 = PRIMARY for "1 min": enter at T−1 if trig_1, else at T if trig_0 (an engine that checks at both moments).
  VE2 = PRIMARY for "2 min": enter at the earliest of T−2, T−1, T0 that triggers.
  Descriptive only: pure T−1 only / T−2 only (miss the late-forming candles), latency T+1 / T+2.
  FALSE triggers = early entries whose candle does NOT qualify at T0 — their own N · WR · avg net reported.
  MATCHED read = candles qualifying at T−k AND T0: paired net (early − T0).
Splits: DESIGN 2023-07-01 → 2025-07-01 · HOLDOUT 2025-07-01 → 2026-11-18 (read once, in the same run as DESIGN — no tuning step exists:
there are no free parameters; k ∈ {1, 2} is fixed here).

## Statistics (ONE size: every fill's net % × 1, summed)
Δ = Σ net(variant) − Σ net(V0), per entry day; day-block bootstrap (10-day blocks, 4,000 draws) 95 % CI of Δ per block. Also per half-year
(7 halves), per-pair concentration (top pair's share of |Δ| via per-pair Δ), leave-one-pair-out (drop the pair carrying the most Δ; and
drop each pair, report min Δ), DCR at SIZE 0.61 before → after, N / trading possibilities (one per coin per hold) / distinct days.
Haircut 30–50 % on any positive DESIGN Δ.

## PASS (each variant VE1, VE2 separately; frozen)
HOLDOUT Δ > 0 with CI lower bound > 0 ∧ DESIGN Δ > 0 ∧ ≥ 2/3 of half-years with Δ > 0 (≥ 5 of 7) ∧ top pair < 50 % of Δ ∧ the false-trigger
trades' own expectancy reported. Else NO SHIP (a positive-but-failing variant = observe line at most).
A latency fix (Part A) has no free parameter; it is a cost estimate, not a ship gate.
