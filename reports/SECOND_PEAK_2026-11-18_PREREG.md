# SECOND PEAK vs TIME-AFTER-FLAG — pre-registration (frozen 2026-11-18 12:15 UTC)

Frozen before ANY result of this study was computed (descriptive or tradable). Research only; nothing in the bot changes.

## Question (operator)
On Frenzy coins, is the tradable SHORT trigger (a) the SECOND PEAK (pump → pullback → second top → dump) or (b) a fixed time X after the
Frenzy flag turns ON?

## Universe (real-time, no later selection)
- Episodes = engine `frenzy_walk` re-implementation (`study_frenzy_swing_short_walk` logic): spike bar = 6-bar return ≥ 5 % ∧ 1h volume ≥ 20×
  normal ∧ ≥ $2 M; **verified** at the spike bar (25 h clean or chained); flag = verified ∧ ≤ 96 h since the spike. **ALL verified episodes**
  are kept (no `ever_seen` / later-shortlist selection — the EXHAUSTION_EXPAND §0 bias is removed).
- A signal is tradable only if, at the signal bar, the flag is ON **and** the engine short-list gate holds at that bar (24 h move ≥ 15 % ∧
  24 h quote volume ≥ $20 M; pair blacklist excluded; BTC/ETH excluded). Otherwise the signal is skipped (not delayed).
- 5m klines 2023-05 → 2026-11-17 (cached); trades 2023-07-01 → data end. One trade per episode per cell; one position per pair per cell.

## Families and cells (42 cells)
Signals on CLOSED 15m bars (clock-aligned, all 3 × 5m bars present), entry = next 5m bar open.

**P2 — second peak (24 cells = P × R × trigger × exit)**
- Peak 1 = running max high since the spike bar. Pullback confirmed when a low ≤ peak1 × (1 − P %), P ∈ {8, 15}.
- Second leg starts when a later high ≥ peak1 × (1 + R %), R ∈ {0 (new high / equal high), −3 (re-test within 3 %)}. H2 = running max high
  of the second leg.
- Trigger: **K** = first 15m close ≤ H2 × (1 − 3 %) · **RSI** = 15m RSI14 crosses > 70 → ≤ 70 on the second leg.
- Only the FIRST second-leg trigger per episode is traded. A leg that pulls back P % from H2 without an RSI trigger is cancelled (no trade
  for that cell in the episode).
- Exits: **EA** = EXHAUSTION exit (TP min(3×ATR15, 5 %) maker, SL min(1.5×ATR15, 5 %), 24 h) · **S5** = TP 5 % maker, stop H2 + 0.5 %, 24 h ·
  **S10** = TP 10 % maker, stop H2 + 0.5 %, 24 h. S5 / S10 skip the trade if the stop is > 15 % above the entry open.

**T — time after the flag (18 cells = X × exit)**
- Entry at spike close + X h, X ∈ {2, 4, 6, 8, 12, 24}. Exits EA / S5 / S10; for T the S-stop is the running max high since the spike + 0.5 %
  (skip if > 15 % away).

## Execution realism
Taker 0.05 % each taker leg, maker 0.02 % on TP; slippage 0.10 % per taker leg (primary), also 0.05 / 0.20. Maker TP fills only if a low
trades ≥ 1 tick THROUGH the TP. Stops gap-aware (fill at max(bar open, stop)); TP/stop same bar → stop. Real funding at each settlement
(hourly when Binance switched the coin); a missing pair-month is charged 0.05 %/8 h against. ATR15 = Wilder 14 on closed 15m bars.

## Blocks
DESIGN 2024-07-01 → 2026-07-01 · HOLDOUT-A 2023-07-01 → 2024-07-01 (2023-H2 has been read by earlier studies — not virgin) · HOLDOUT-B
2026-07-01 → data end. Every half-year reported (23-H2 … 26-H2).

## Pass bar (all must hold, per cell)
1. every half-year avg net > 0 · 2. best-5 % of trades removed > 0 · 3. avg > 0 at 0.20 % slippage · 4. beats the random-timing null
p < 0.05 (random entry bars in the same traded episodes, flag ON ∧ gate ON, drawn after the cell's signal-eligible start: P2 = pullback
confirmation bar, T = spike close; same exit; S-stops use the real trade's stop distance) · 5. family-wise max-statistic shuffled null
p < 0.05 (over all 42 cells) · 6. Kelly at the one-sided 95 % Wilson WR floor > 0 with n = distinct trade days · 7. both holdouts > 0.
Plus reported, not gating: overlap with EXHAUSTION_SHORT (REF R4 × EA, all-spike list) and the avg of non-overlapping trades.

## Step 0 (hindsight, descriptive, not tradable)
15m zigzag peaks within the flag window (spike → +96 h) at reversal X ∈ {8 %, 15 %}; peaks per episode, which peak is the episode max,
hours spike → peak 1 / peak 2 / max, max drawdown 24 h / 72 h after each peak, and hours spike → main top. Also hours from state-ON.

## W/L (best cell only)
Coverage table first; 1D then 2D (all pairs of variables, terciles), shuffled-label null. No ship from the W/L read.

## Verdict rule
BUILD only if a cell passes all 7 AND adds value beyond EXHAUSTION_SHORT. OBSERVE if it passes ≥ 5 including holdouts. Otherwise NO.
