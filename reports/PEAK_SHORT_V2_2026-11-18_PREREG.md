# PEAK SHORT V2 — pre-registration (frozen 2026-11-18 13:58 UTC)

Frozen before ANY result of this study was computed (descriptive or tradable). Research only; nothing in the bot changes. No commit.

## Question (operator)
"Deep dive into the double-peak theory … double peak, or even the first peak. We are still missing the sleeve that gives us many trades
per day." Goal: a short sleeve with ≥ 8–10 trades/day at a positive, consistent average after ALL costs.

## Theory (written before the test)
1. Pumps top out mostly at the FIRST peak (Step 0 of SECOND_PEAK: 58–73 %). The loss of every earlier peak short comes from SQUEEZERS,
   which are not predictable at entry (SQUEEZER study) but declare themselves in the first ~2 h after entry. → **Survival / wait-and-see**:
   a peak that has already HELD W hours is less likely to be a squeezer; the cost is the part of the drop already gone. The test is whether
   the remaining drop pays for that.
2. A **true double top** (second high within ±3 % of the first, then a close below the neckline = the low between the two highs) is the
   classical distribution signature: buyers failed twice at the same price and the support between gave way. The stop sits above a
   level that held twice.
3. **First-peak failure with bearish RSI divergence**: a lower high made on weaker momentum = the buying pressure behind the pump is
   fading. Stop above the first peak.
4. **Funding** costs −1 to −1.8 %/trade on these coins (84 % already hourly). Skipping coins already switched to < 8 h funding, or with
   a deep negative rate, or exiting before a settlement when the rate is deep negative, should remove a known cost.
5. **Frequency**: Frenzy-flagged coins give ≈ 1–3 setups/day. A broader pump universe (every USDT perp) is the only way to 8–10/day.

## Universe (real time, no later selection)
- **U-A (primary, all pumps)**: every USDT perp in the 5m cache (658 pairs; BTC/ETH and the pair blacklist out). A pump TRIGGER at 5m
  bar i = close[i] ≥ 1.15 × min(low of the last 72 bars incl. i) (≥ +15 % in ≤ 6 h) ∧ 1 h quote volume (last 12 bars) ≥ 10 × normal hour
  (`norm_per_bar` = engine `normal_hour_usd`) ∧ 24 h quote volume ≥ $20 M. EPISODE = from the first trigger bar to 24 h after the LAST
  trigger bar (re-triggers extend it), capped at 72 h from the first trigger. A new episode can only start after the previous one ended.
- Signals trade only inside the episode window and only if, at the signal bar, 24 h quote volume ≥ $20 M (liquidity).
- **U-F (subset, reported, not separate cells)**: the U-A trades where the engine Frenzy flag (all verified `frenzy_walk` episodes,
  ≤ 96 h; same code as SECOND_PEAK) is ON at the signal bar.
- Episode running high H* = max high from the first trigger bar to now.

## Families (10 entries × 4 exits = 40 cells, + 4 funding-modifier cells = **44 cells**)
Signals on CLOSED 15m bars (clock-aligned, all 3 × 5m present); entry = next 5m bar open. One trade per signal; one position per pair
per cell (a signal while that cell's position on the pair is open is skipped).

**F1 SURVIVAL — S{W}·X{X} (6 entries)**: at a 15m close, the episode high H* was set ≥ W hours ago (no higher high since; the bar that set
it closed ≥ W h before this close) ∧ close ≤ H* × (1 − X %). W ∈ {1, 2, 4} h, X ∈ {5, 10} %. Fires once per H* (a new episode high
re-arms it). Relevant peak = H*.

**F2 DOUBLE TOP — DT (1 entry)**: 15m bars in the episode. P1 = a high confirmed by an 8 % pullback (low ≤ P1 × 0.92). Neckline N = the
lowest low after P1 until price first trades back to ≥ 0.97 × P1 (the second top). If any high goes > 1.03 × P1 the pattern is void and
that leg becomes the new P1 candidate. After the second top, trigger = first 15m close < N (while no high > 1.03 × P1). Relevant peak =
max(P1, highest high of the second top). Once per P1.

**F3 DIVERGENCE — DV15 / DV1h (2 entries)**: pivot highs on the 15m (DV15) or 1h (DV1h) bars of the episode, pivot = high above the 3 bars
before and the 2 bars after (confirmed 2 bars later). P1 = a pivot high that is the episode high H* when it prints. Signal = the first
later pivot high LH < P1 (no high > P1 in between) with RSI14(TF) at LH ≤ RSI14 at P1 − 5, at the close of LH's confirmation bar.
Relevant peak = P1. Once per P1.

**F4 FUNDING-AWARE (4 cells)**: applied to the TWO cells with the best DESIGN-block average among the 40 (picked on DESIGN only):
- **F4a** skip the entry if the coin's last funding interval < 8 h (already switched to 4 h / 1 h) or the last settled rate ≤ −0.10 %.
- **F4b** F4a + force a time exit at the open of the 5m bar 5 min before any settlement whose last known rate ≤ −0.10 %.

**Exits (4, every entry)**:
- **X1 TP5·PK·24h** TP 5 % maker, stop = relevant peak + 0.5 % (skip the trade if > 15 % above the entry open), 24 h time exit.
- **X2 TP10·PK·48h** TP 10 % maker, stop = relevant peak + 0.5 % (skip if > 15 %), 48 h.
- **X3 TP5·S12·24h** TP 5 % maker, fixed stop 12 %, 24 h.
- **X4 TP3·S8·12h** TP 3 % maker, fixed stop 8 %, 12 h.

## Execution realism
Taker 0.05 % each taker leg, maker 0.02 % on TP; slippage 0.10 % per taker leg (primary), also 0.05 / 0.20. Maker TP fills only if a
low trades ≥ 1 tick THROUGH it. Stops gap-aware (fill at max(bar open, stop)); TP / stop on the same bar → stop. Real funding at every
settlement (hourly when Binance switched the coin); a missing pair-month is charged 0.05 %/8 h against.

## Blocks
DESIGN 2024-07-01 → 2026-07-01 · HOLDOUT-A 2023-07-01 → 2024-07-01 (2023-H2 read by earlier studies — not virgin) · HOLDOUT-B
2026-07-01 → data end · **H0 never-read 2023-01-01 → 2023-07-01** (klines fetched from data.binance.vision for this study; read ONCE,
after the cell table is frozen; reported, never used to pick).

## Pass bar (per cell; all must hold)
① every half-year (23-H2 … 26-H2) avg net > 0 · ② best-5 % of trades removed > 0 · ③ avg > 0 at 0.20 % slippage · ④ beats random timing
in the same state p < 0.05 (40 random entry bars per trade, same episode, bars inside the window with gate ON and close ≤ 0.95 × H*
(post-pullback state), same exit; peak stops use the real trade's stop distance) · ⑤ family-wise max-z null p < 0.05 over all 44 cells ·
⑥ Kelly at the one-sided 95 % Wilson WR floor > 0, n = distinct trade days · ⑦ HOLDOUT-A > 0 and HOLDOUT-B > 0 · ⑧ H0 > 0 (read once).
Reported, not gating: trades/day (U-A and U-F); overlap with EXHAUSTION_SHORT (REF R4 × EA on the all-spike list) and the avg of the
non-overlapping trades; DCR at ONE size (0.183 × equity notional per trade, every calendar day), worst day, max drawdown, max concurrent.

## Descriptive ("how much of the drop is left after W hours")
For every episode high that holds ≥ W h (W ∈ {0.25, 0.5, 1, 2, 3, 4, 6, 8}): median drop already done at peak + W (close vs peak), median
further drop in the next 24 h (min low vs that close), median 24 h short return gross, and the squeezer share (high ≥ +10 % above that
close within 24 h). Hindsight-free at time W (all quantities after W are outcomes).

## Verdict rule
BUILD only if a cell passes ①–⑧ AND adds value beyond EXHAUSTION_SHORT (non-overlap avg > 0). OBSERVE if it passes ≥ 6 incl. ⑦ and ⑧.
Otherwise NO. A frequency below 8/day is reported plainly, never hidden.
