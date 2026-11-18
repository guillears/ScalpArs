# DIST_WL — PRE-REGISTRATION (written 2026-11-18 19:32 UTC, before any winners-vs-losers result was computed)

Question (operator): "Is there something that divides winners vs losers in DISTRIBUTION shorts, to improve the +0.30 %?"

## What was already seen before this file (disclosed)
- The frozen L1 card (`reports/LEADS_DEEP_2026-11-18.md` §L1): T5A 4,811 · +0.167; LIVE (= EXHAUSTION live exit) 4,649 · +0.302.
- Full-period (DESIGN + HOLDOUT mixed) splits at T5A: ret1h ≥ 0 +0.24 vs < 0 +0.07; BTC 1d EMA50>EMA200 +0.20 vs bear +0.10;
  top_pos_ls ≥ 1 +0.19 vs < 1 +0.06; "not EXH" +0.15; VSW with top-trader positions RISING −0.36. These are in-sample hints only.
- The reproduction run (this study) printed only N / avg / WR of the two frozen sets. Nothing else.

## Trade set (fixed)
- L1 = 15m VSW red volume candle (`g_VSW15`) ∧ `top_pos_chg4h ≤ −0.01097`, Frenzy-coin pool `reports/UNIFIED_SHORT_data/pool.pkl`.
- PRIMARY exit = `LIVE` = EXHAUSTION live exit: TP min(20 × ATR15, 10 %) resting maker (fills only when traded through by 1 tick),
  SL min(1.5 × ATR15, 6 %) gap-aware taker, 24 h. Costs: taker 0.05 % + 0.10 % slip per taker side, maker 0.02 %, real funding.
  One position per pair (greedy, time order). Secondary (cross-check only): T5A.
- Screen runs on the fixed L1 LIVE trade set. Before → after re-runs one-position-per-pair on (L1 ∧ kept).

## Split
- DESIGN = 2023-07-01 → 2025-06-30 (entry time). HOLDOUT = 2025-07-01 → 2026-11-18. Bin edges (terciles / quintiles) are
  computed on DESIGN and frozen. HOLDOUT is read ONCE, only for candidates that pass every DESIGN step below.

## Theory of the trade (written first)
Distribution = large holders unload into late buyers after a pump. The short pays when the supply they dump overwhelms the remaining
demand. It fails when (a) the "selling" is liquidations / profit-taking inside a trend that still has fresh buyers (squeeze or
continuation); (b) top traders cut only a little, or flip to shorts rather than leave; (c) the red candle is a capitulation low
(sellers exhausted → bounce), not the start of the dump; (d) the coin already dumped far from its high (nothing left to distribute).

## Theory predictions (tested FIRST; sign stated = side expected to be BETTER for the short)
| # | Failure mode | Variable(s) (entry-time) | Prediction |
|---|---|---|---|
| P1 | (d) | `dd_peak_pct` (close vs episode high), `dist_vwap_pct`, `dist_base_pct` | nearer the high / above VWAP = better; far below the high = worse |
| P2 | (c) | `close_pos15` (close position in the candle range), `lowwick15` (lower-wick share), `range15_pct`, `bar_ret_pct`, `rsi15` | close at the low (small lower wick) = better; long lower wick / huge range / RSI15 deeply oversold (capitulation) = worse |
| P3 | (a) | `oi_chg15m`, `oi_chg1h` (OI change across the candle, lag-safe ≤ T − 5 min) | OI falling = longs leaving = better; OI rising = new shorts fighting = worse |
| P4 | (b) | `top_acct_chg4h`, `pos_minus_acct_rel` (relative position-ratio change − relative account-ratio change), `top_pos_ls` level | positions falling MORE than accounts (a few whales selling) = better; `top_pos_ls` < 1 after the drop (net-short flip) = worse |
| P5 | (b) | `top_pos_chg4h`, `top_pos_chg4h_rel`, `top_pos_chg1h` | bigger drop = better (dose-response) |
| P6 | (a/c) | `last5_share` (last 5m of the candle's volume share; continuation INSIDE the candle — post-candle volume is NOT entry-known), `vmult15`, `n_prior_vsw_red` (earlier red VSW candles in the episode) | selling accelerating into the close = better; first distribution candle = better than the 4th |
| P7 | (a) | `spot_share1h`, `ssc` | perp-led selling (spot share falling, ssc < 0) = better |
| P8 | (a) | `fund_last`, `fund_hourly`, `prem_last` | positive funding / premium (longs crowded) = better; negative (shorts crowded, squeeze fuel) = worse |
| P9 | (d) | `hrs_since_spike` (since Frenzy-ON), `hrs_since_peak` | early in the episode / fresh off the peak = better |
| H1 | hint | `ret1h_pct` sign | ≥ 0 better (seen, in-sample) |
| H2 | hint | `btc_d1_regime` | bull better (seen, in-sample) |

## Exhaustive screen (cross-check, after the theory tests)
- Variables: every numeric entry-time column of the pool (all features, BTC macro, breadth, rank / market-cap, funding, premium, OI,
  L/S, spot share) + the new columns above. Trigger bits that are constant on L1, outcome columns and identifiers excluded.
- Outcomes: ① booked net % (LIVE) · ② ENTRY label WORKED (−1 % in our favour before the trade's own stop distance, 5m bars;
  AMBIG when one bar touches both) · ③ REACH (max favourable move before the stop distance / 24 h).
- 1D: each variable at sign (if it takes both signs), terciles, quintiles; each bin vs the rest. 2D: every variable pair, sign × sign
  quadrants and tercile × tercile cells; each cell vs the rest. Minimum 50 DESIGN trades in the cell.
- Statistic: Welch t of (cell mean − rest mean) on DESIGN. Family-wise null: ≥ 200 label shuffles across the whole screen
  (same statistic); report the 95th percentile of the max |t| and the expected number of |t| > 3 cells by luck.

## Survivor steps (DESIGN only)
Pass the family-wise null (|t| > null 95 % max) or, for the 9 theory predictions, the theory-family null (max over the theory
variables only) → per half-year sign consistency (≥ 3 of 4 DESIGN halves the same sign) → per BTC daily regime at sign granularity
(same sign in both) → day-block bootstrap 95 % CI of the kept-cohort Δ excludes 0 → no single pair or day ≥ 50 % of the Δ $ →
best-5 %-out still positive → 0.20 % slip still positive.

## Ship bar (all required)
A kept cohort must: beat the baseline avg on HOLDOUT (read once) and be positive there; be positive in ≥ 2/3 of half-years (N ≥ 20);
pass the family-wise shuffled null; stay positive at 0.20 % slip (DESIGN and HOLDOUT). The blocked side must also meet the locked
filter EXPECTANCY bar of CLAUDE.md (WR < sleeve breakeven WR, avg < 0 at 95 % by day-block bootstrap, ≥ 8 windows, no window / pair
≥ 50 % of the loss, N ≥ 15). 30–50 % in-sample haircut on every Δ. Anything that passes DESIGN but fails HOLDOUT = OBSERVE at most.

## Reported per candidate
Before → after: N · trades/day · WR · avg % · DCR at ONE size 0.61 × equity · max DD · Kelly at the one-sided 95 % Wilson WR floor
(per trade and per day), plus the blocked side priced.
