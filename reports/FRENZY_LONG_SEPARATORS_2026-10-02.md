# 🔬 FRENZY_LONG — what separates winners from losers? (deep dive)

1,103 trades (live rule, ATR ≤ 2.5 %, live exit, strict ruler): 42% won · 58% full stops · +0.16 per trade. 46 features, all read on closed bars at entry. Coverage < 100 %: none.

## 1 · Every feature in quintiles (lowest → highest fifth): per-trade mean

| Feature | Q1 | Q2 | Q3 | Q4 | Q5 | spread Q5−Q1 | monotonic | same sign in both halves | same sign SEEN & UNSEEN |
|---|---|---|---|---|---|---|---|---|---|
| green_1h | +0.20 | +0.46 | +0.16 | -0.02 | -0.82 | -1.02 | mostly | NO (+0.09 / -0.68) | yes (-0.14 / -0.60) |
| funding_rate | -0.43 | +0.20 | +0.17 | +0.53 | +0.33 | +0.76 | no | yes (+0.44 / +0.20) | yes (+0.46 / +0.62) |
| down1h ᵐ | +0.68 | -0.01 | -0.21 | +0.33 | +0.02 | -0.65 | mostly | NO (-0.57 / +0.07) | NO (-0.60 / +0.33) |
| run_pct | -0.17 | +0.44 | +0.08 | +0.07 | +0.38 | +0.55 | no | NO (+0.23 / -0.15) | yes (+0.21 / +0.05) |
| down24h ᵐ | +0.15 | +0.34 | +0.30 | +0.37 | -0.35 | -0.50 | no | yes (-0.40 / -0.03) | NO (+0.08 / -0.49) |
| btc_r1h ᵐ | +0.09 | +0.03 | +0.11 | -0.01 | +0.59 | +0.50 | no | yes (+0.55 / +0.09) | NO (+0.73 / -0.41) |
| q24_m | +0.49 | -0.20 | +0.05 | +0.44 | +0.02 | -0.47 | no | yes (+0.26 / +0.05) | NO (+0.22 / -0.39) |
| bar_upper_wick | -0.21 | -0.09 | +0.29 | +0.57 | +0.24 | +0.45 | mostly | yes (+0.28 / +0.64) | yes (+0.59 / +0.51) |
| down4h ᵐ | -0.11 | +0.22 | +0.45 | -0.04 | +0.30 | +0.41 | mostly | NO (-0.47 / +0.68) | yes (+0.06 / +0.22) |
| r1h | +0.44 | -0.36 | +0.60 | +0.09 | +0.04 | -0.40 | mostly | NO (+0.12 / -0.17) | NO (-0.14 / +0.06) |
| vol_trend | -0.13 | -0.04 | +0.61 | +0.11 | +0.24 | +0.37 | mostly | yes (+0.49 / +0.03) | yes (+0.42 / +0.41) |
| btc_r24h ᵐ | -0.07 | +0.14 | +0.43 | +0.01 | +0.30 | +0.37 | mostly | NO (+0.57 / -0.29) | yes (+0.12 / +0.10) |
| vol_mult_1h_ago | +0.24 | +0.39 | +0.10 | +0.19 | -0.10 | -0.34 | no | yes (-0.43 / -0.11) | yes (-0.63 / -0.09) |
| hours | -0.21 | +0.29 | +0.07 | +0.52 | +0.14 | +0.34 | no | yes (+0.18 / +0.29) | yes (+0.21 / +0.34) |
| gain_pct | -0.09 | +0.22 | +0.17 | +0.27 | +0.23 | +0.32 | no | NO (+0.56 / -0.07) | yes (+0.13 / +0.11) |
| r4h | +0.10 | -0.04 | +0.48 | -0.12 | +0.39 | +0.29 | no | NO (+0.32 / -0.05) | NO (+0.28 / -0.34) |
| spike_volx | +0.40 | +0.20 | -0.26 | +0.33 | +0.14 | -0.26 | mostly | NO (+0.48 / -0.55) | NO (-0.21 / +0.15) |
| range_1h | -0.14 | +0.35 | +0.21 | +0.28 | +0.11 | +0.26 | no | yes (+0.08 / +0.15) | yes (+0.01 / +0.20) |
| r24h | +0.38 | +0.14 | +0.34 | -0.20 | +0.14 | -0.24 | no | yes (-0.37 / -0.17) | yes (-0.19 / -0.25) |
| btc_r4h ᵐ | +0.27 | +0.29 | +0.35 | -0.13 | +0.03 | -0.24 | mostly | NO (+0.27 / -0.83) | yes (-0.38 / -0.35) |
| off_24h_high | -0.03 | +0.15 | +0.15 | +0.33 | +0.21 | +0.24 | mostly | yes (+0.39 / +0.02) | yes (+0.40 / +0.20) |
| r30m | +0.13 | +0.34 | +0.03 | +0.37 | -0.06 | -0.20 | no | NO (+0.12 / -0.26) | yes (-0.08 / -0.06) |
| e50_vs_e200 | +0.29 | +0.16 | +0.30 | -0.06 | +0.11 | -0.19 | no | yes (-0.39 / -0.06) | NO (+0.10 / -0.88) |
| below200 ᵐ | -0.08 | +0.51 | -0.07 | +0.34 | +0.10 | +0.18 | no | NO (-0.65 / +0.42) | NO (+0.30 / -0.39) |
| d_below50 ᵐ | +0.60 | -0.01 | -0.12 | -0.09 | +0.43 | -0.17 | no | NO (-0.47 / +0.15) | NO (-0.52 / +0.47) |
| stop_atr | +0.14 | +0.22 | -0.07 | +0.22 | +0.31 | +0.17 | mostly | NO (+0.31 / -0.29) | NO (+0.37 / -0.35) |
| atr | +0.31 | +0.22 | -0.07 | +0.22 | +0.14 | -0.17 | mostly | NO (-0.31 / +0.29) | NO (-0.37 / +0.35) |
| vs_e200 | +0.23 | +0.11 | +0.23 | -0.16 | +0.40 | +0.17 | no | NO (+0.03 / -0.23) | NO (+0.10 / -0.31) |
| btc_vs_e50 ᵐ | +0.37 | +0.07 | -0.39 | +0.55 | +0.20 | -0.17 | mostly | NO (+0.73 / -0.36) | NO (+0.53 / -0.44) |
| vs_e50 | +0.14 | -0.14 | +0.45 | +0.04 | +0.30 | +0.16 | no | yes (+0.49 / +0.10) | yes (+0.16 / +0.19) |
| gap5_8 | +0.19 | +0.10 | +0.16 | +0.32 | +0.04 | -0.15 | no | NO (+0.34 / -0.35) | NO (+0.18 / -0.23) |
| r5m | +0.25 | +0.77 | -0.06 | -0.24 | +0.10 | -0.15 | no | yes (-0.76 / -0.58) | yes (-0.66 / -0.43) |
| vs_vwap | +0.05 | -0.01 | +0.50 | +0.37 | -0.09 | -0.14 | mostly | yes (+0.42 / +0.03) | NO (+0.44 / -0.17) |
| rsi14 | +0.12 | -0.32 | +0.63 | +0.16 | +0.21 | +0.09 | no | NO (+0.79 / -0.09) | yes (+0.33 / +0.02) |
| gap5_20 | +0.07 | -0.41 | +0.72 | +0.26 | +0.16 | +0.08 | mostly | yes (+0.50 / +0.21) | yes (+0.46 / +0.10) |
| spike_r30 | +0.14 | -0.03 | +0.28 | +0.19 | +0.22 | +0.08 | no | NO (+0.45 / -0.31) | NO (-0.02 / +0.39) |
| off_peak | +0.08 | +0.25 | +0.08 | +0.24 | +0.16 | +0.08 | no | NO (+0.31 / -0.27) | yes (+0.02 / +0.15) |
| below50 ᵐ | +0.14 | +0.38 | -0.08 | +0.15 | +0.22 | +0.08 | mostly | NO (-0.52 / +0.48) | NO (-0.11 / +0.10) |
| btc_vs_e200 ᵐ | +0.21 | +0.09 | -0.23 | +0.46 | +0.28 | +0.07 | mostly | NO (+0.78 / -0.21) | yes (+0.07 / +0.48) |
| hour_utc ᵐ | -0.03 | +0.08 | +0.49 | +0.23 | +0.04 | +0.07 | no | NO (+0.14 / -0.05) | NO (+0.42 / -0.45) |
| state_bars_before | +0.19 | -0.02 | +0.33 | +0.12 | – | -0.07 | mostly | NO (+0.28 / -0.14) | NO (+0.12 / -0.20) |
| bar_body | +0.17 | +0.88 | -0.06 | -0.40 | +0.22 | +0.05 | no | yes (-0.72 / -0.46) | yes (-0.65 / -0.55) |
| vol_mult | +0.30 | +0.08 | -0.04 | +0.22 | +0.25 | -0.05 | no | NO (+0.12 / -0.11) | NO (-0.55 / +0.74) |
| vwap_dist_atr | -0.04 | +0.13 | +0.22 | +0.48 | +0.01 | +0.05 | mostly | NO (+0.70 / -0.21) | NO (+0.46 / -0.23) |
| attempt | +0.11 | +0.20 | +0.44 | +0.10 | – | -0.01 | mostly | NO (+0.44 / -0.14) | NO (+0.19 / -0.11) |
| vs_e20 | +0.17 | -0.21 | +0.52 | +0.15 | +0.18 | +0.01 | no | NO (+0.58 / -0.19) | yes (+0.12 / +0.05) |

ᵐ = market-wide (repeats across same-day trades). Halves / sets columns: top 40 % minus bottom 40 % of the feature inside each part.

## 2 · Best single-feature keep-rule vs luck

Best rule keeping ≥ 40 % of trades: **bar_body ≤ 0.0635** → 552 trades at +0.54 per trade (all trades: +0.16). The same search on 300 shuffles of the outcomes finds a 'best rule' of +0.46 on average (95th percentile +0.56). The real best rule beats 92% of the shuffles.

## 3 · Out of sample — a rule found on one part, frozen, applied to the other

| Found on | best rule there | kept there | applied to | kept · per trade | rest · per trade | all · per trade | holds? |
|---|---|---|---|---|---|---|---|
| Jan–Apr | btc_vs_e200 > -0.0462 | 255 at +0.60 | the other part | 339 · +0.03 | 253 · +0.24 | +0.12 | no |
| Jan–Apr | vs_vwap > 6.41 | 255 at +0.58 | the other part | 332 · -0.00 | 260 · +0.28 | +0.12 | no |
| Jan–Apr | e50_vs_e200 ≤ 5.5 | 256 at +0.54 | the other part | 287 · +0.28 | 305 · -0.03 | +0.12 | YES |
| May–Sep | btc_r4h ≤ 0.0217 | 296 at +0.48 | the other part | 272 · +0.17 | 239 · +0.26 | +0.21 | no |
| May–Sep | r5m ≤ 0.243 | 296 at +0.45 | the other part | 298 · +0.43 | 213 · -0.11 | +0.21 | YES |
| May–Sep | bar_body ≤ 0.139 | 296 at +0.44 | the other part | 297 · +0.50 | 214 · -0.19 | +0.21 | YES |
| SEEN ≥ $100M | bar_body ≤ 0.0762 | 311 at +0.59 | the other part | 248 · +0.46 | 234 · -0.18 | +0.15 | YES |
| SEEN ≥ $100M | r5m ≤ 0.138 | 311 at +0.59 | the other part | 245 · +0.46 | 237 · -0.17 | +0.15 | YES |
| SEEN ≥ $100M | vol_mult ≤ 112 | 311 at +0.48 | the other part | 237 · -0.27 | 245 · +0.56 | +0.15 | no |
| UNSEEN $20–100M | vol_mult > 112 | 241 at +0.59 | the other part | 304 · -0.11 | 317 · +0.44 | +0.17 | no |
| UNSEEN $20–100M | q24_m ≤ 49.6 | 241 at +0.49 | the other part | 0 · +nan | 621 · +0.17 | +0.17 | no |
| UNSEEN $20–100M | bar_upper_wick > 0.259 | 241 at +0.46 | the other part | 332 · +0.42 | 289 · -0.11 | +0.17 | YES |

## 4 · Every pair of features in 3 × 3 cells (990 pairs, cells ≥ 50 trades)

- Best cell: r5m low × bar_upper_wick high → 67 trades at +1.59. Shuffled outcomes give a best cell of +1.66 on average (95th pct +2.04).
- Worst cell: q24_m mid × green_1h high → 87 trades at -1.11. Shuffled: -1.25 on average (5th pct -1.50).

## 5 · What the trades look like after entry (not usable at entry — for the exit discussion)

- Winners (458): median result +4.2 %, top quarter above +5.3 %, best +15 %.
- Full stops (644): median best point before the stop +1.2 %; 44% never reached +1 %.
- Total result +178 points; the best 5 % of trades alone make +484 points.

## NOT tested

- Order book / spread / open interest / liquidations at entry (not in the data).
- News, listings, sector moves.
- Delisted pairs; no fresh time period (the two halves and the two volume groups are the only out-of-sample splits).
- Interactions of three or more features; non-quantile thresholds.
