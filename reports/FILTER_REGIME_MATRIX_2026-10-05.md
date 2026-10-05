# Filter × BTC-regime matrix: momentum-LONG sleeve, yr5 replay (2026-10-05)

Operator question: *"there must be a MACRO BTC variable that says which rule makes sense in each period."*
Scope: read-only research. Nothing in services/, config or templates was touched.

## Plain-language answer

**No robust regime-conditional filter was found.** No momentum-LONG filter helps in one BTC state and hurts in the other in a way that beats chance.

- **What was tested.** 40 live momentum-LONG refusal gates × 21 pre-declared BTC macro splits = 840 cells, of which 713 had enough signals.
  - The gates covered 199,015 priced refusal signals over 273 days of the yr5 replay (3 seeds, de-duplicated).
  - The splits: daily, 4h and 1h trend; 1d/3d/7d/30d returns; distance below the 30-day high; 5m ADX level and direction; 5m ATR %; 72h chop; breadth; 5m RSI and slope; and BTC volume ratio.
- **What a finding needed.** Opposite signs in the two states, plus ≥15 signals and ≥8 days per state. The same sign pattern also had to hold in Jan–Apr and in May–Oct, and the day-block bootstrap CI of the state gap had to exclude 0.
- **Result: 1 cell passed. The shuffled null produces 2.2 on average.** The null permutes macro tags across days within the same month and was run for 500 rounds. Its 95th percentile is 5 cells, and 84 % of null rounds produce at least 1 passing cell. The STRONG tier (each state's own CI also excludes 0) has 0 observed vs a null mean of 0.14.
- **The one passing cell is not usable.** It is MACRO:BTC_RSI_ADX_CROSS split on BTC 5m RSI terciles.
  - It helps at RSI >55.4: 4,660 signals over 273 days, −0.121 % [−0.147, −0.093].
  - It hurts at RSI ≤44.8: 23 signals over 17 days, +0.228 % [−0.121, +0.506]. That CI includes 0.
  - The split largely restates the rule's own RSI bands. No real-fill cohort exists to check it against.
- **A wider screen also found nothing above chance.** This screen asks only whether a filter is worth significantly more in one state than the other, with the gap direction stable in both halves. It found 32 cells, against a null mean of 44.6 (95th percentile 63; 92 % of null rounds produce ≥32).
- **Every filter's blocked cohort is negative over the whole year** (−0.01 to −0.28 %). The admitted yr5 momentum-long fills, priced the same way (replica, entry +60 s), average **−0.054 %** (1,122 de-duplicated fills, WR 63 %).
  - So, measured against 0, almost any refused entry looks like a loser on yr5.
  - Measured against the admitted cohort, most gates still remove entries that are worse than what gets admitted.
  - This is the main reason so few cells flip sign.

**LONG_HEAT_BLOCK (the operator's focus): the first cut is refuted.**

- **Daily golden cross (EMA50>EMA200) cannot be tested on yr5.** That state covers only 7 days (Sep 12 onward) with 26 signals, and none fall in Jan–Apr.
- **The 4h EMA50>EMA200 lead from the first cut fails.**
  - When 4h is golden, the block helps: −0.103 % over 30 days, CI [−0.278, +0.051].
  - When 4h is not golden, the block hurts: +0.151 % over 17 days, CI [−0.306, +0.447].
  - But in Jan–Apr the "not golden" state is −0.016 %, so the sign pattern flips in that half.
  - The gap CI [−0.589, +0.218] includes 0.
  - The forward check contradicts it: all 10 forward re-scope blocks fell in the 4h-golden state and won (+0.238 %), which is exactly where yr5 says the block helps.
- **The closest heat near-miss is the BTC volume ratio.**
  - On low-volume days (daily quote volume ≤0.82 × its 30-day mean) the block removes winners: +0.323 %, 47 signals over 14 days, positive in both halves.
  - On high-volume days (>1.13×) it removes losers: −0.108 %, 71 signals over 21 days.
  - The gap CI is [−0.738, +0.031], so it just misses.
- **72h chop flips sign but has only 6 days**: +0.315 % in chop vs −0.126 % outside.
- Neither is shippable. The volume ratio is a reasonable observe-only scout candidate if frozen as written.

**Recommendation.**
- **Make no filter regime-conditional.** Nothing clears the bars, and the matrix produces fewer findings than its own shuffled null.
- **Observe-only candidates for the scout**, with thresholds frozen now and never re-fit:
  1. HEAT × BTC daily quote-vol ratio: block when ratio >1.13, admit when ≤0.817.
  2. HEAT × 72h chop (eff ≤0.007).
- **Side lead at the sleeve level, not a filter.** Admitted yr5 fills lose when the BTC 1h EMA20 is falling (−0.125 % [−0.206, −0.042], 144 days, negative in both halves) and are flat when it is rising (−0.001 %).
  - This is a market-wide sleeve on/off variable, so the window-units rule and the existing 1h-slope / deadband gates apply. Check their overlap before treating it as new.

## Method (short)
1. **Signals.** Taken from the yr5 journals (57 chunk × seed dirs), counting only lines inside each chunk's [start_ms, end_ms).
   - Ladder and macro gates: journal **FAILS** lines whose complete fail set is one gate (sole blockers). `MACRO:<gate>` means the ladder passed and the scan's first macro veto refused it.
   - Engine-chain gates (post-ladder): **BLOCK** lines.
   - One signal per gate × pair per 30 min (chained) per seed, de-duplicated across seeds by 5-min bucket, weighted n_seeds/3. Gates with more than 8,000 signals were sampled to 8,000 (fixed seed). Days remain the unit.
2. **Pricing.** Live momentum-LONG exit replica, `scripts/ml_exit_optimize_yr5.py` BASE via `run_fill`, on **local** data only.
   - Entry = signal +60 s, taker fee. ATR % = Wilder-14 on the closed 5m bars. BTC RSI = the replica's live ruler.
   - Coverage: 99.9995 % priced (198,595 tick paths, 235 mixed, 184 1m, 1 no path).
   - Validation: the 153 heat signals match the earlier network-priced `HEAT_YR5_SIGNALS_priced.csv` to within 0.0005 %, with 100 % same sign.
3. **Macro tags.** Read at the signal from the last closed bar or last scan at or before t (no look-ahead).
   - Sources: k1d, k4h, btc_1h, btc_5m caches plus the journal SCAN lines (btc_rsi, btc_adx, btc_slope, eff, off30d, bull/bear, above).
   - Tercile cut-points were fixed beforehand on the year's 5-min grid. The comparison is T1 vs T3.
4. **Bars and null.** As in the summary. The day-block bootstrap is a Poisson day-weight bootstrap with days resampled jointly (2,000 reps; 300 reps inside the null). Null: 500 rounds, each day's tags replaced by another same-month day's tags at the same time of day, with the whole matrix re-run each round.
5. **Real-fill check (refute-only).** Uses `MASTER_POOL_stacked.csv` rows whose stack_block_reason maps to the gate (as traded) and the scout's forward re-priced refusals (`SCOUT_REVERT_GATES.json`, sim1).

## Null summary
| screen | observed | null mean | null p95 | P(null ≥ observed) |
|---|---|---|---|---|
| full regime-conditional bar (PASS) | 1 | 2.22 | 5.05 | 0.84 |
| STRONG (each state CI excludes 0) | 0 | 0.14 | – | 1.00 |
| secondary state-gap screen | 32 | 44.6 | 63 | 0.92 |

## Gates excluded from the inventory (and why)
- **Other sleeves**: BULLRUN_* (MAX_SLOTS, BTC_EMA13, BTC_OFF24H, BREADTH, BTC_1H_SLOPE, REARM_STALE, PVR_MAX, BELOW_1H_EMA50, DISLOC), BR_PAIR_BLACKLIST (bull-run pair list), FRENZY_* (ATR_HIGH, GREEN_BAR, GVOL_HIGH, WIDE_GVOL_HIGH, WIDE_DISLOC, DISLOC, *_OPEN_REFUSED, WIDE_PAIR_DAY_CAP), SURGE_* (ATR_LOW, NOT_LEADER, DISLOC).
- **Sleeves switched off**: FLIP_LONG_DISABLED, SPIKE_CHASE_DISABLED.
- **Housekeeping**: PAIR_HELD (803), COOLDOWN (47), NO_BALANCE (425), REDEPLOY_OPEN (1).
- **Counted via FAILS-sole instead of BLOCK.** BLOCK lines record only the first failing gate, so they are not sole blockers. This applies to the ladder gates (PAIR_ADX_MAX, PAIR_EMA_GAP_MIN, PAIR_RSI_MOMENTUM_LOADX, PAIR_RANGE_POSITION_MAX, RSICEIL_DOOR_ADXMIN, PAIR_EMA20_FILTER/SLOPE, PAIR_ADX_CONFIDENCE, PAIR_EMA_GAP_MAX, PAIR_RSI_RANGE, PAIR_EMA_GAP_5_20, EMA5_STRETCH) and to the BTC macro BLOCK lines (BTC_ADX_GATE_LOW/HIGH, BTC_RSI_ADX_CROSS, BTC_SLOPE_GATE).
- **Never a sole blocker in FAILS**: PAIR_RSI_ADX_CROSS[60-65:0-25], PAIR_RSI_RANGE[<40].
- **Absent from yr5**: LONG_CHOP_BURST has 0 lines, because code 181131e predates the block.
- Most lines of the included gates had room for a new position (97–100 %).

## G. Gate inventory — every momentum-LONG refusal gate, blocked cohort re-priced (whole year, all states)

avg = blocked-cohort avg % per signal (live exit replica, entry t+60 s, n_seeds/3-weighted). Negative = the gate removes losers.
n_all = de-duplicated signals; gates over 8,000 sampled to 8,000 (fixed seed). CI = day-block bootstrap 95 %.

| gate | source | signals (all) | priced | days | WR | avg % | 95 % CI |
|---|---|---|---|---|---|---|---|
| CALM3D_BTC_ATR_MIN | BLOCK(chain) | 24 | 24 | 12 | 41% | -0.280 | [-0.474, +0.107] |
| LONG_MEGACAP_BLOCK | BLOCK(chain) | 182 | 182 | 108 | 48% | -0.226 | [-0.337, -0.106] |
| BTC_SLOPE_MAX_GATE | BLOCK(chain) | 48 | 48 | 13 | 58% | -0.193 | [-0.535, +0.095] |
| CALM3D_DMI | BLOCK(chain) | 768 | 768 | 122 | 55% | -0.169 | [-0.239, -0.096] |
| BTC_ACCEL_CHASE_LONG | BLOCK(chain) | 4,838 | 4,838 | 232 | 54% | -0.167 | [-0.204, -0.124] |
| PAIR_EMA20_SLOPE | FAILS(sole) | 3,680 | 3,680 | 270 | 63% | -0.156 | [-0.191, -0.123] |
| PAIR_EMA20_FILTER | FAILS(sole) | 5,419 | 5,419 | 273 | 57% | -0.145 | [-0.178, -0.114] |
| RSI_SPIKE_GUARD | BLOCK(chain) | 1,707 | 1,707 | 271 | 57% | -0.139 | [-0.180, -0.098] |
| ADX_DELTA_BTC_ADX_CROSS | BLOCK(chain) | 11,486 | 8,000 | 273 | 54% | -0.136 | [-0.163, -0.110] |
| PAIR_EMA_GAP_5_20[<min] | FAILS(sole) | 7,451 | 7,451 | 273 | 59% | -0.135 | [-0.165, -0.107] |
| VOL_GATE | BLOCK(chain) | 7,265 | 7,265 | 258 | 56% | -0.131 | [-0.166, -0.098] |
| LONG_UNMATCHED_ONLY | BLOCK(chain) | 4,128 | 4,128 | 263 | 58% | -0.131 | [-0.167, -0.097] |
| PAIR_EMA_GAP_NOT_EXPANDING | BLOCK(chain) | 14,357 | 8,000 | 270 | 56% | -0.131 | [-0.159, -0.103] |
| MACRO:BTC_ADX_GATE_HIGH | FAILS(sole) | 10,291 | 8,000 | 240 | 58% | -0.128 | [-0.155, -0.105] |
| MACRO:BTC_ADX_GATE_LOW | FAILS(sole) | 38,116 | 8,000 | 272 | 56% | -0.126 | [-0.148, -0.104] |
| PAIR_EMA_GAP_MAX | FAILS(sole) | 3,822 | 3,822 | 271 | 67% | -0.125 | [-0.164, -0.090] |
| ENTRY_QUALITY_SCORE | BLOCK(chain) | 2,366 | 2,366 | 259 | 55% | -0.124 | [-0.161, -0.085] |
| MACRO:BTC_RSI_ADX_CROSS | FAILS(sole) | 62,792 | 8,000 | 273 | 56% | -0.122 | [-0.144, -0.100] |
| PAIR_ATR_MIN | BLOCK(chain) | 11,243 | 8,000 | 273 | 48% | -0.121 | [-0.143, -0.098] |
| PAIR_RSI_ADX_CROSS | BLOCK(chain) | 1,287 | 1,287 | 215 | 57% | -0.120 | [-0.171, -0.067] |
| PAIR_RANGE_POSITION_MAX | FAILS(sole) | 13,277 | 8,000 | 272 | 57% | -0.120 | [-0.148, -0.092] |
| FAN_RATIO_GATE | BLOCK(chain) | 42,109 | 8,000 | 273 | 58% | -0.120 | [-0.140, -0.100] |
| MACRO:BTC_SLOPE_GATE | FAILS(sole) | 38,252 | 8,000 | 273 | 57% | -0.119 | [-0.140, -0.099] |
| LONG_BTC1H_DEADBAND | BLOCK(chain) | 3,361 | 3,361 | 187 | 57% | -0.114 | [-0.163, -0.066] |
| PAIR_EMA_GAP_MIN | FAILS(sole) | 46,466 | 8,000 | 273 | 53% | -0.114 | [-0.134, -0.094] |
| RNGPOS_ADX_DELTA_CROSS | BLOCK(chain) | 3,666 | 3,666 | 273 | 55% | -0.113 | [-0.143, -0.083] |
| PAIR_RSI_MOMENTUM_LOADX | FAILS(sole) | 32,288 | 8,000 | 273 | 57% | -0.113 | [-0.140, -0.087] |
| MACRO:BTC_SLOPE_MAX_GATE | FAILS(sole) | 137 | 137 | 16 | 67% | -0.110 | [-0.323, +0.074] |
| PAIR_ADX_CONFIDENCE | FAILS(sole) | 10,346 | 8,000 | 272 | 60% | -0.106 | [-0.130, -0.082] |
| PAIR_NO_TRADE | BLOCK(chain) | 3,908 | 3,908 | 273 | 48% | -0.102 | [-0.124, -0.081] |
| BTC_GAP_BTC_ADX_CROSS | BLOCK(chain) | 2,607 | 2,607 | 200 | 58% | -0.100 | [-0.149, -0.047] |
| PAIR_ADX_MAX | FAILS(sole) | 30,216 | 7,999 | 273 | 58% | -0.098 | [-0.123, -0.073] |
| EMA5_STRETCH[>max] | FAILS(sole) | 12,353 | 8,000 | 273 | 65% | -0.094 | [-0.126, -0.065] |
| BTC_RSI_ATR_COND | BLOCK(chain) | 6,037 | 6,037 | 143 | 56% | -0.093 | [-0.128, -0.061] |
| RSICEIL_DOOR_ADXMIN | FAILS(sole) | 28,420 | 8,000 | 272 | 58% | -0.093 | [-0.116, -0.070] |
| PAIR_EMA_GAP_5_20[>max] | FAILS(sole) | 10,308 | 8,000 | 272 | 67% | -0.093 | [-0.120, -0.065] |
| PAIR_RSI_RANGE[>70] | FAILS(sole) | 11,961 | 8,000 | 273 | 59% | -0.085 | [-0.111, -0.057] |
| PAIR_ATR_MAX | BLOCK(chain) | 98 | 98 | 68 | 75% | -0.062 | [-0.206, +0.087] |
| CALM3D_REENTRY | BLOCK(chain) | 63 | 63 | 34 | 64% | -0.039 | [-0.162, +0.100] |
| LONG_HEAT_BLOCK | BLOCK(chain) | 153 | 153 | 47 | 67% | -0.012 | [-0.208, +0.171] |

## S. Survivors of the full regime-conditional bar (opposite signs + N/days + both halves + gap CI)

| gate | split | state A | A: N · days · avg [CI] · halves | state B | B: N · days · avg [CI] · halves | gap A−B [CI] | verdict |
|---|---|---|---|---|---|---|---|
| MACRO:BTC_RSI_ADX_CROSS | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 4660 · 273 d · -0.121 [-0.147, -0.093] · H1 -0.125 (117 d) / H2 -0.118 (156 d) | T1 ≤44.8 | 23 · 17 d · +0.228 [-0.121, +0.506] · H1 +0.264 (9 d) / H2 +0.183 (8 d) | -0.349 [-0.627, -0.006] | PASS |

## N. Near misses — opposite signs in the two states but failing one bar (why in the verdict column)

| gate | split | state A | A: N · days · avg [CI] · halves | state B | B: N · days · avg [CI] · halves | gap A−B [CI] | verdict |
|---|---|---|---|---|---|---|---|
| LONG_HEAT_BLOCK | 72h efficiency ≤0.007 (chop) (EFF72_CHOP) | yes | 43 · 6 d · +0.315 [-0.534, +0.597] · H1 +0.428 (2 d) / H2 +0.246 (4 d) | no | 110 · 44 d · -0.126 [-0.285, +0.036] · H1 -0.331 (20 d) / H2 +0.018 (24 d) | +0.441 [-0.410, +0.746] | days<8 |
| LONG_HEAT_BLOCK | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 71 · 21 d · -0.108 [-0.376, +0.097] · H1 -0.284 (10 d) / H2 -0.007 (11 d) | T1 ≤0.817 | 47 · 14 d · +0.323 [-0.104, +0.506] · H1 +0.270 (6 d) / H2 +0.368 (8 d) | -0.430 [-0.738, +0.031] | gap CI ∋ 0 |
| PAIR_ATR_MAX | breadth bull% > bear% (BULL_GT_BEAR) | yes | 39 · 34 d · -0.256 [-0.570, +0.020] · H1 -0.275 (10 d) / H2 -0.246 (24 d) | no | 59 · 42 d · +0.060 [-0.117, +0.218] · H1 -0.027 (17 d) / H2 +0.123 (25 d) | -0.316 [-0.659, +0.005] | half h1 flips |
| PAIR_ATR_MAX | BTC 7d return (R7D_T) | T3 >2.33 | 40 · 26 d · +0.094 [-0.122, +0.297] · H1 -0.035 (13 d) / H2 +0.260 (13 d) | T1 ≤-2.26 | 32 · 22 d · -0.190 [-0.491, +0.099] · H1 -0.272 (6 d) / H2 -0.160 (16 d) | +0.284 [-0.059, +0.621] | half h1 flips |
| LONG_HEAT_BLOCK | BTC 4h EMA50>EMA200 (H4_GOLDEN) | yes | 95 · 30 d · -0.103 [-0.278, +0.051] · H1 -0.311 (10 d) / H2 -0.022 (20 d) | no | 58 · 17 d · +0.151 [-0.306, +0.447] · H1 -0.016 (11 d) / H2 +0.437 (6 d) | -0.253 [-0.589, +0.218] | half h1 flips |
| PAIR_RSI_RANGE[>70] | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 7973 · 273 d · -0.086 [-0.114, -0.059] · H1 -0.078 (117 d) / H2 -0.093 (156 d) | no | 27 · 12 d · +0.160 [-0.303, +0.413] · H1 +0.216 (9 d) / H2 -0.440 (3 d) | -0.246 [-0.493, +0.209] | half h2 flips |
| CALM3D_DMI | BTC 1h EMA20 rising (H1_EMA20_UP) | yes | 753 · 120 d · -0.173 [-0.242, -0.098] · H1 -0.174 (37 d) / H2 -0.173 (83 d) | no | 15 · 5 d · +0.067 [-0.162, +0.238] · H1 +0.207 (1 d) / H2 +0.055 (4 d) | -0.241 [-0.429, -0.003] | days<8 |
| PAIR_ATR_MAX | BTC 30d return (R30D_T) | T3 >5.75 | 37 · 25 d · -0.155 [-0.460, +0.119] · H1 -0.199 (9 d) / H2 -0.125 (16 d) | T1 ≤-1.63 | 21 · 17 d · +0.072 [-0.212, +0.377] · H1 +0.032 (3 d) / H2 +0.088 (14 d) | -0.227 [-0.643, +0.165] | gap CI ∋ 0 |
| PAIR_ATR_MAX | BTC 1h EMA20 rising (H1_EMA20_UP) | yes | 56 · 41 d · -0.149 [-0.372, +0.068] · H1 -0.143 (10 d) / H2 -0.152 (31 d) | no | 42 · 29 d · +0.053 [-0.173, +0.282] · H1 -0.085 (14 d) / H2 +0.191 (15 d) | -0.202 [-0.535, +0.133] | half h1 flips |
| PAIR_ATR_MAX | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 42 · 38 d · -0.178 [-0.490, +0.106] · H1 -0.248 (10 d) / H2 -0.147 (28 d) | no | 56 · 39 d · +0.013 [-0.158, +0.190] · H1 -0.050 (17 d) / H2 +0.064 (22 d) | -0.191 [-0.552, +0.163] | half h1 flips |
| PAIR_ATR_MAX | BTC 7d return (R7D>0) | >0 | 57 · 37 d · +0.017 [-0.172, +0.185] · H1 -0.072 (16 d) / H2 +0.102 (21 d) | ≤0 | 41 · 31 d · -0.170 [-0.428, +0.083] · H1 -0.213 (7 d) / H2 -0.155 (24 d) | +0.187 [-0.130, +0.474] | half h1 flips |
| MACRO:BTC_SLOPE_MAX_GATE | BTC 5m ADX rising (ADX5_RISING) | yes | 121 · 16 d · -0.129 [-0.337, +0.083] · H1 -0.167 (11 d) / H2 -0.065 (5 d) | no | 16 · 3 d · +0.053 [-0.692, +0.143] · H1 +0.046 (2 d) / H2 +0.099 (1 d) | -0.183 [-0.390, +0.609] | days<8 |
| PAIR_ATR_MAX | BTC distance below 30d high (OFF30_T) | T3 >-3.98 | 34 · 24 d · -0.152 [-0.449, +0.131] · H1 -0.177 (8 d) / H2 -0.137 (16 d) | T1 ≤-8.84 | 23 · 18 d · +0.026 [-0.214, +0.272] · H1 -0.116 (7 d) / H2 +0.162 (11 d) | -0.178 [-0.566, +0.182] | half h1 flips |
| PAIR_ATR_MAX | BTC 30d return (R30D>0) | >0 | 68 · 44 d · -0.121 [-0.333, +0.070] · H1 -0.189 (18 d) / H2 -0.068 (26 d) | ≤0 | 30 · 24 d · +0.052 [-0.191, +0.296] · H1 +0.111 (5 d) / H2 +0.028 (19 d) | -0.174 [-0.500, +0.133] | gap CI ∋ 0 |
| PAIR_EMA_GAP_NOT_EXPANDING | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 7191 · 270 d · -0.136 [-0.167, -0.106] · H1 -0.120 (116 d) / H2 -0.151 (154 d) | T1 ≤44.8 | 22 · 9 d · +0.036 [-0.383, +0.358] · H1 +0.114 (8 d) / H2 -0.694 (1 d) | -0.171 [-0.494, +0.242] | half h2 thin |
| RSICEIL_DOOR_ADXMIN | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 7982 · 272 d · -0.093 [-0.116, -0.070] · H1 -0.088 (117 d) / H2 -0.098 (155 d) | no | 18 · 13 d · +0.076 [-0.236, +0.339] · H1 +0.132 (7 d) / H2 -0.003 (6 d) | -0.169 [-0.427, +0.139] | half h2 flips |
| PAIR_NO_TRADE | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 3052 · 273 d · -0.110 [-0.135, -0.084] · H1 -0.095 (117 d) / H2 -0.122 (156 d) | T1 ≤44.8 | 39 · 32 d · +0.048 [-0.146, +0.240] · H1 -0.088 (9 d) / H2 +0.103 (23 d) | -0.158 [-0.351, +0.042] | half h1 flips |
| PAIR_RSI_MOMENTUM_LOADX | breadth bull % (BULL_T) | T3 >52.2 | 6736 · 273 d · -0.112 [-0.141, -0.083] · H1 -0.105 (117 d) / H2 -0.118 (156 d) | T1 ≤24.4 | 83 · 61 d · +0.034 [-0.177, +0.226] · H1 +0.149 (30 d) / H2 -0.117 (31 d) | -0.146 [-0.340, +0.065] | half h2 flips |
| PAIR_EMA_GAP_5_20[>max] | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 7976 · 272 d · -0.093 [-0.121, -0.067] · H1 -0.094 (117 d) / H2 -0.092 (155 d) | no | 24 · 17 d · +0.040 [-0.338, +0.401] · H1 -0.341 (9 d) / H2 +0.465 (8 d) | -0.133 [-0.495, +0.242] | half h1 flips |
| CALM3D_REENTRY | BTC 4h EMA50>EMA200 (H4_GOLDEN) | yes | 36 · 19 d · -0.097 [-0.267, +0.068] · H1 -0.122 (4 d) / H2 -0.090 (15 d) | no | 27 · 15 d · +0.032 [-0.172, +0.302] · H1 +0.578 (4 d) / H2 -0.104 (11 d) | -0.129 [-0.444, +0.145] | half h2 flips |
| BTC_GAP_BTC_ADX_CROSS | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 1708 · 172 d · -0.090 [-0.154, -0.027] · H1 -0.060 (76 d) / H2 -0.113 (96 d) | T1 ≤44.8 | 165 · 71 d · +0.037 [-0.089, +0.171] · H1 +0.108 (30 d) / H2 -0.022 (41 d) | -0.126 [-0.271, +0.006] | half h2 flips |
| PAIR_RANGE_POSITION_MAX | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 7972 · 272 d · -0.121 [-0.148, -0.093] · H1 -0.137 (117 d) / H2 -0.108 (155 d) | no | 28 · 19 d · +0.005 [-0.330, +0.218] · H1 +0.171 (8 d) / H2 -0.160 (11 d) | -0.126 [-0.340, +0.202] | half h2 flips |
| PAIR_ATR_MAX | breadth bull % (BULL_T) | T3 >52.2 | 28 · 26 d · -0.079 [-0.413, +0.238] · H1 +0.070 (7 d) / H2 -0.149 (19 d) | T1 ≤24.4 | 40 · 30 d · +0.045 [-0.189, +0.272] · H1 -0.025 (13 d) / H2 +0.099 (17 d) | -0.124 [-0.510, +0.264] | half h1 flips |
| PAIR_ATR_MAX | BTC 3d return (R3D>0) | >0 | 49 · 36 d · +0.000 [-0.230, +0.209] · H1 -0.195 (12 d) / H2 +0.114 (24 d) | ≤0 | 49 · 32 d · -0.118 [-0.344, +0.076] · H1 -0.042 (11 d) / H2 -0.171 (21 d) | +0.118 [-0.182, +0.415] | half h1 flips |
| RSICEIL_DOOR_ADXMIN | breadth bull % (BULL_T) | T3 >52.2 | 7029 · 272 d · -0.090 [-0.115, -0.064] · H1 -0.084 (117 d) / H2 -0.094 (155 d) | T1 ≤24.4 | 59 · 47 d · +0.028 [-0.114, +0.164] · H1 -0.060 (23 d) / H2 +0.116 (24 d) | -0.118 [-0.259, +0.030] | half h1 flips |
| CALM3D_REENTRY | BTC 7d return (R7D>0) | >0 | 30 · 17 d · +0.023 [-0.100, +0.197] · H1 +0.046 (4 d) / H2 +0.015 (13 d) | ≤0 | 33 · 17 d · -0.091 [-0.290, +0.132] · H1 +0.314 (4 d) / H2 -0.183 (13 d) | +0.114 [-0.144, +0.370] | half h1 flips |
| LONG_HEAT_BLOCK | BTC 5m ADX rising (ADX5_RISING) | yes | 121 · 42 d · -0.033 [-0.252, +0.180] · H1 -0.181 (19 d) / H2 +0.077 (23 d) | no | 32 · 14 d · +0.063 [-0.245, +0.324] · H1 +0.014 (6 d) / H2 +0.087 (8 d) | -0.097 [-0.404, +0.270] | half h2 flips |
| PAIR_ATR_MAX | % pairs above 72h ref (bull-run 'above') (ABOVE72_T) | T3 >52.5 | 24 · 18 d · -0.075 [-0.485, +0.282] · H1 -0.163 (8 d) / H2 -0.002 (10 d) | T1 ≤48 | 33 · 23 d · +0.018 [-0.223, +0.235] · H1 +0.018 (8 d) / H2 +0.019 (15 d) | -0.094 [-0.549, +0.352] | gap CI ∋ 0 |
| PAIR_ADX_MAX | breadth bull % (BULL_T) | T3 >52.2 | 6759 · 272 d · -0.091 [-0.119, -0.066] · H1 -0.103 (117 d) / H2 -0.081 (155 d) | T1 ≤24.4 | 91 · 64 d · +0.001 [-0.178, +0.187] · H1 -0.061 (28 d) / H2 +0.042 (36 d) | -0.093 [-0.279, +0.088] | half h1 flips |
| LONG_HEAT_BLOCK | BTC 7d return (R7D>0) | >0 | 99 · 36 d · -0.038 [-0.243, +0.132] · H1 -0.320 (16 d) / H2 +0.101 (20 d) | ≤0 | 54 · 11 d · +0.043 [-0.412, +0.422] · H1 +0.069 (5 d) / H2 +0.009 (6 d) | -0.081 [-0.508, +0.407] | half h2 flips |
| LONG_HEAT_BLOCK | BTC 1d return (R1D>0) | >0 | 66 · 26 d · +0.034 [-0.210, +0.256] · H1 -0.216 (8 d) / H2 +0.123 (18 d) | ≤0 | 87 · 21 d · -0.048 [-0.373, +0.216] · H1 -0.120 (13 d) / H2 +0.029 (8 d) | +0.081 [-0.275, +0.460] | half h1 flips |
| MACRO:BTC_SLOPE_MAX_GATE | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 96 · 8 d · -0.070 [-0.364, +0.175] · H1 -0.107 (5 d) / H2 -0.015 (3 d) | T1 ≤0.817 | 17 · 4 d · +0.008 [-0.727, +0.334] · H1 +0.008 (4 d) / H2 +nan (0 d) | -0.077 [-0.493, +0.623] | days<8 |
| LONG_HEAT_BLOCK | BTC 7d return (R7D_T) | T3 >2.33 | 83 · 28 d · -0.033 [-0.282, +0.170] · H1 -0.319 (13 d) / H2 +0.122 (15 d) | T1 ≤-2.26 | 53 · 10 d · +0.027 [-0.438, +0.409] · H1 +0.069 (5 d) / H2 -0.030 (5 d) | -0.059 [-0.516, +0.450] | half h2 flips |
| LONG_HEAT_BLOCK | BTC close > daily EMA200 (ABOVE_D200) | yes | 57 · 16 d · +0.014 [-0.253, +0.210] · H1 +nan (0 d) / H2 +0.014 (16 d) | no | 96 · 31 d · -0.031 [-0.313, +0.224] · H1 -0.147 (21 d) / H2 +0.220 (10 d) | +0.045 [-0.320, +0.392] | half h1 thin |
| LONG_HEAT_BLOCK | BTC 3d return (R3D>0) | >0 | 81 · 30 d · +0.005 [-0.234, +0.193] · H1 -0.288 (13 d) / H2 +0.123 (17 d) | ≤0 | 72 · 17 d · -0.035 [-0.370, +0.282] · H1 -0.056 (8 d) / H2 -0.009 (9 d) | +0.040 [-0.358, +0.430] | half h1 flips |
| LONG_HEAT_BLOCK | BTC 30d return (R30D>0) | >0 | 115 · 35 d · -0.021 [-0.198, +0.135] · H1 -0.231 (12 d) / H2 +0.058 (23 d) | ≤0 | 38 · 12 d · +0.014 [-0.572, +0.475] · H1 -0.058 (9 d) / H2 +0.400 (3 d) | -0.035 [-0.515, +0.579] | half h1 flips |
| LONG_HEAT_BLOCK | % pairs above 72h ref (bull-run 'above') (ABOVE72_T) | T3 >52.5 | 63 · 26 d · +0.007 [-0.243, +0.197] · H1 -0.277 (12 d) / H2 +0.163 (14 d) | T1 ≤48 | 44 · 12 d · -0.008 [-0.495, +0.455] · H1 +0.028 (7 d) / H2 -0.101 (5 d) | +0.015 [-0.498, +0.540] | half h1 flips |

## X. Secondary state-gap screen — filter worth significantly more in one state (any signs), gap direction same in both halves

Not the regime-conditional bar (both states may be negative = the filter helps everywhere, more in one). Compare its count with the null.

| gate | split | state A | A: N · days · avg [CI] · halves | state B | B: N · days · avg [CI] · halves | gap A−B [CI] | verdict |
|---|---|---|---|---|---|---|---|
| MACRO:BTC_RSI_ADX_CROSS | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 4660 · 273 d · -0.121 [-0.147, -0.093] · H1 -0.125 (117 d) / H2 -0.118 (156 d) | T1 ≤44.8 | 23 · 17 d · +0.228 [-0.121, +0.506] · H1 +0.264 (9 d) / H2 +0.183 (8 d) | -0.349 [-0.627, -0.006] | PASS |
| PAIR_ADX_MAX | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 7976 · 273 d · -0.097 [-0.122, -0.072] · H1 -0.109 (117 d) / H2 -0.085 (156 d) | no | 23 · 16 d · -0.390 [-0.626, -0.139] · H1 -0.298 (6 d) / H2 -0.425 (10 d) | +0.294 [+0.044, +0.523] | same sign |
| CALM3D_DMI | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 213 · 32 d · -0.078 [-0.224, +0.047] · H1 -0.015 (9 d) / H2 -0.092 (23 d) | T1 ≤0.817 | 290 · 44 d · -0.270 [-0.362, -0.172] · H1 -0.248 (14 d) / H2 -0.278 (30 d) | +0.191 [+0.021, +0.352] | same sign |
| BTC_GAP_BTC_ADX_CROSS | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 873 · 65 d · -0.199 [-0.291, -0.099] · H1 -0.204 (29 d) / H2 -0.194 (36 d) | T1 ≤0.817 | 903 · 67 d · -0.015 [-0.100, +0.058] · H1 +0.043 (33 d) / H2 -0.094 (34 d) | -0.183 [-0.303, -0.060] | same sign |
| RNGPOS_ADX_DELTA_CROSS | 72h efficiency ≤0.007 (chop) (EFF72_CHOP) | yes | 457 · 100 d · -0.234 [-0.303, -0.160] · H1 -0.236 (42 d) / H2 -0.233 (58 d) | no | 3209 · 273 d · -0.097 [-0.131, -0.065] · H1 -0.094 (117 d) / H2 -0.099 (156 d) | -0.137 [-0.215, -0.054] | same sign |
| RNGPOS_ADX_DELTA_CROSS | breadth bull % (BULL_T) | T3 >52.2 | 2436 · 266 d · -0.130 [-0.168, -0.095] · H1 -0.119 (116 d) / H2 -0.138 (150 d) | T1 ≤24.4 | 330 · 163 d · -0.001 [-0.091, +0.084] · H1 -0.009 (72 d) / H2 +0.006 (91 d) | -0.129 [-0.221, -0.036] | same sign |
| RSI_SPIKE_GUARD | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 690 · 221 d · -0.199 [-0.262, -0.135] · H1 -0.145 (97 d) / H2 -0.245 (124 d) | T1 ≤44.8 | 612 · 216 d · -0.078 [-0.148, -0.005] · H1 -0.130 (93 d) / H2 -0.039 (123 d) | -0.121 [-0.228, -0.023] | same sign |
| RSI_SPIKE_GUARD | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 815 · 240 d · -0.199 [-0.258, -0.139] · H1 -0.158 (104 d) / H2 -0.234 (136 d) | no | 892 · 250 d · -0.084 [-0.148, -0.019] · H1 -0.119 (109 d) / H2 -0.056 (141 d) | -0.115 [-0.214, -0.026] | same sign |
| BTC_RSI_ATR_COND | BTC 1d return (R1D>0) | >0 | 3178 · 77 d · -0.039 [-0.086, +0.008] · H1 -0.025 (20 d) / H2 -0.043 (57 d) | ≤0 | 2859 · 66 d · -0.153 [-0.199, -0.109] · H1 -0.133 (20 d) / H2 -0.159 (46 d) | +0.113 [+0.049, +0.181] | same sign |
| RNGPOS_ADX_DELTA_CROSS | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 2675 · 272 d · -0.143 [-0.178, -0.110] · H1 -0.134 (117 d) / H2 -0.149 (155 d) | no | 991 · 251 d · -0.034 [-0.085, +0.017] · H1 -0.050 (103 d) / H2 -0.022 (148 d) | -0.109 [-0.165, -0.054] | same sign |
| RNGPOS_ADX_DELTA_CROSS | breadth bull% > bear% (BULL_GT_BEAR) | yes | 2860 · 272 d · -0.137 [-0.170, -0.105] · H1 -0.135 (117 d) / H2 -0.138 (155 d) | no | 806 · 249 d · -0.029 [-0.086, +0.027] · H1 -0.031 (104 d) / H2 -0.028 (145 d) | -0.107 [-0.169, -0.046] | same sign |
| ADX_DELTA_BTC_ADX_CROSS | BTC 7d return (R7D_T) | T3 >2.33 | 2729 · 91 d · -0.083 [-0.126, -0.040] · H1 -0.069 (43 d) / H2 -0.098 (48 d) | T1 ≤-2.26 | 2741 · 91 d · -0.184 [-0.231, -0.134] · H1 -0.190 (41 d) / H2 -0.175 (50 d) | +0.100 [+0.034, +0.166] | same sign |
| PAIR_EMA20_SLOPE | BTC 7d return (R7D>0) | >0 | 1825 · 136 d · -0.106 [-0.145, -0.070] · H1 -0.076 (60 d) / H2 -0.129 (76 d) | ≤0 | 1855 · 134 d · -0.204 [-0.260, -0.148] · H1 -0.250 (56 d) / H2 -0.163 (78 d) | +0.098 [+0.030, +0.167] | same sign |
| VOL_GATE | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 2701 · 87 d · -0.188 [-0.252, -0.124] · H1 -0.193 (39 d) / H2 -0.183 (48 d) | T1 ≤0.817 | 2394 · 86 d · -0.093 [-0.150, -0.039] · H1 -0.069 (38 d) / H2 -0.120 (48 d) | -0.095 [-0.178, -0.008] | same sign |
| RNGPOS_ADX_DELTA_CROSS | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 2482 · 271 d · -0.142 [-0.179, -0.107] · H1 -0.124 (117 d) / H2 -0.155 (154 d) | T1 ≤44.8 | 519 · 201 d · -0.054 [-0.120, +0.008] · H1 -0.060 (81 d) / H2 -0.050 (120 d) | -0.087 [-0.157, -0.018] | same sign |
| PAIR_EMA20_SLOPE | BTC 30d return (R30D_T) | T3 >5.75 | 1185 · 91 d · -0.118 [-0.162, -0.078] · H1 -0.099 (27 d) / H2 -0.125 (64 d) | T1 ≤-1.63 | 1340 · 91 d · -0.203 [-0.278, -0.132] · H1 -0.211 (45 d) / H2 -0.195 (46 d) | +0.085 [+0.000, +0.167] | same sign |
| PAIR_ADX_CONFIDENCE | BTC 5m ATR % (ATR5_T) | T3 >0.185 | 3123 · 185 d · -0.073 [-0.117, -0.032] · H1 -0.085 (92 d) / H2 -0.057 (93 d) | T1 ≤0.115 | 2115 · 175 d · -0.153 [-0.192, -0.112] · H1 -0.186 (55 d) / H2 -0.140 (120 d) | +0.079 [+0.023, +0.137] | same sign |
| PAIR_EMA20_SLOPE | BTC 4h EMA50>EMA200 (H4_GOLDEN) | yes | 1739 · 133 d · -0.116 [-0.154, -0.080] · H1 -0.124 (45 d) / H2 -0.112 (88 d) | no | 1941 · 144 d · -0.192 [-0.252, -0.133] · H1 -0.192 (73 d) / H2 -0.192 (71 d) | +0.076 [+0.006, +0.148] | same sign |
| EMA5_STRETCH[>max] | BTC 5m ATR % (ATR5_T) | T3 >0.185 | 4216 · 186 d · -0.065 [-0.117, -0.012] · H1 -0.057 (93 d) / H2 -0.077 (93 d) | T1 ≤0.115 | 1553 · 168 d · -0.139 [-0.184, -0.090] · H1 -0.138 (53 d) / H2 -0.140 (115 d) | +0.074 [+0.007, +0.143] | same sign |
| PAIR_EMA_GAP_5_20[<min] | BTC 1d return (R1D>0) | >0 | 3439 · 133 d · -0.096 [-0.132, -0.059] · H1 -0.072 (55 d) / H2 -0.115 (78 d) | ≤0 | 4012 · 140 d · -0.167 [-0.211, -0.124] · H1 -0.202 (62 d) / H2 -0.134 (78 d) | +0.071 [+0.016, +0.126] | same sign |
| MACRO:BTC_ADX_GATE_HIGH | BTC 30d return (R30D_T) | T3 >5.75 | 2842 · 80 d · -0.096 [-0.141, -0.055] · H1 -0.109 (23 d) / H2 -0.091 (57 d) | T1 ≤-1.63 | 2459 · 79 d · -0.163 [-0.209, -0.118] · H1 -0.176 (39 d) / H2 -0.154 (40 d) | +0.068 [+0.001, +0.131] | same sign |
| ADX_DELTA_BTC_ADX_CROSS | BTC 3d return (R3D>0) | >0 | 4116 · 139 d · -0.105 [-0.136, -0.073] · H1 -0.092 (61 d) / H2 -0.117 (78 d) | ≤0 | 3884 · 134 d · -0.170 [-0.211, -0.130] · H1 -0.172 (56 d) / H2 -0.167 (78 d) | +0.065 [+0.014, +0.118] | same sign |
| PAIR_ATR_MIN | % pairs above 72h ref (bull-run 'above') (ABOVE72_T) | T3 >52.5 | 2676 · 136 d · -0.148 [-0.191, -0.106] · H1 -0.134 (59 d) / H2 -0.157 (77 d) | T1 ≤48 | 2408 · 141 d · -0.084 [-0.127, -0.040] · H1 -0.089 (62 d) / H2 -0.080 (79 d) | -0.064 [-0.123, -0.004] | same sign |
| RNGPOS_ADX_DELTA_CROSS | BTC 5m ADX rising (ADX5_RISING) | yes | 1984 · 267 d · -0.142 [-0.182, -0.104] · H1 -0.133 (115 d) / H2 -0.149 (152 d) | no | 1682 · 266 d · -0.080 [-0.126, -0.034] · H1 -0.083 (116 d) / H2 -0.077 (150 d) | -0.063 [-0.122, -0.005] | same sign |
| PAIR_EMA_GAP_5_20[<min] | BTC 3d return (R3D>0) | >0 | 3553 · 139 d · -0.102 [-0.141, -0.063] · H1 -0.119 (61 d) / H2 -0.087 (78 d) | ≤0 | 3898 · 134 d · -0.164 [-0.209, -0.122] · H1 -0.172 (56 d) / H2 -0.158 (78 d) | +0.063 [+0.006, +0.121] | same sign |
| MACRO:BTC_ADX_GATE_HIGH | BTC 5m ATR % (ATR5_T) | T3 >0.185 | 3815 · 164 d · -0.135 [-0.178, -0.094] · H1 -0.140 (84 d) / H2 -0.130 (80 d) | T1 ≤0.115 | 1684 · 81 d · -0.073 [-0.117, -0.028] · H1 -0.047 (20 d) / H2 -0.077 (61 d) | -0.062 [-0.125, -0.002] | same sign |
| PAIR_NO_TRADE | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 1412 · 91 d · -0.136 [-0.179, -0.093] · H1 -0.131 (41 d) / H2 -0.140 (50 d) | T1 ≤0.817 | 1196 · 91 d · -0.077 [-0.115, -0.042] · H1 -0.078 (39 d) / H2 -0.076 (52 d) | -0.059 [-0.113, -0.003] | same sign |
| ADX_DELTA_BTC_ADX_CROSS | BTC 1d return (R1D>0) | >0 | 4012 · 133 d · -0.108 [-0.143, -0.073] · H1 -0.108 (55 d) / H2 -0.108 (78 d) | ≤0 | 3988 · 140 d · -0.165 [-0.202, -0.127] · H1 -0.154 (62 d) / H2 -0.176 (78 d) | +0.057 [+0.005, +0.109] | same sign |
| MACRO:BTC_ADX_GATE_LOW | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 3154 · 91 d · -0.150 [-0.184, -0.113] · H1 -0.155 (41 d) / H2 -0.145 (50 d) | T1 ≤0.817 | 2313 · 90 d · -0.094 [-0.133, -0.054] · H1 -0.112 (39 d) / H2 -0.076 (51 d) | -0.056 [-0.113, -0.003] | same sign |
| PAIR_ADX_MAX | BTC 5m ADX rising (ADX5_RISING) | yes | 5287 · 271 d · -0.115 [-0.143, -0.087] · H1 -0.133 (117 d) / H2 -0.099 (154 d) | no | 2712 · 265 d · -0.064 [-0.107, -0.020] · H1 -0.064 (117 d) / H2 -0.063 (148 d) | -0.051 [-0.102, -0.005] | same sign |
| MACRO:BTC_SLOPE_GATE | BTC 1d return (R1D>0) | >0 | 4105 · 133 d · -0.097 [-0.125, -0.069] · H1 -0.118 (55 d) / H2 -0.084 (78 d) | ≤0 | 3895 · 140 d · -0.141 [-0.171, -0.112] · H1 -0.159 (62 d) / H2 -0.127 (78 d) | +0.044 [+0.003, +0.086] | same sign |
| PAIR_EMA_GAP_MIN | BTC 5m ADX rising (ADX5_RISING) | yes | 4701 · 272 d · -0.127 [-0.151, -0.103] · H1 -0.124 (116 d) / H2 -0.130 (156 d) | no | 3299 · 265 d · -0.095 [-0.122, -0.066] · H1 -0.098 (116 d) / H2 -0.091 (149 d) | -0.033 [-0.067, -0.001] | same sign |

## H. LONG_HEAT_BLOCK (yr5 = breadth-only re-scope, bull ≥85) — every split

| gate | split | state A | A: N · days · avg [CI] · halves | state B | B: N · days · avg [CI] · halves | gap A−B [CI] | verdict |
|---|---|---|---|---|---|---|---|
| LONG_HEAT_BLOCK | BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 71 · 21 d · -0.108 [-0.376, +0.097] · H1 -0.284 (10 d) / H2 -0.007 (11 d) | T1 ≤0.817 | 47 · 14 d · +0.323 [-0.104, +0.506] · H1 +0.270 (6 d) / H2 +0.368 (8 d) | -0.430 [-0.738, +0.031] | gap CI ∋ 0 |
| LONG_HEAT_BLOCK | BTC 4h EMA50>EMA200 (H4_GOLDEN) | yes | 95 · 30 d · -0.103 [-0.278, +0.051] · H1 -0.311 (10 d) / H2 -0.022 (20 d) | no | 58 · 17 d · +0.151 [-0.306, +0.447] · H1 -0.016 (11 d) / H2 +0.437 (6 d) | -0.253 [-0.589, +0.218] | half h1 flips |
| LONG_HEAT_BLOCK | BTC 5m ATR % (ATR5_T) | T3 >0.185 | 74 · 23 d · +0.081 [-0.331, +0.380] · H1 -0.046 (14 d) / H2 +0.313 (9 d) | T1 ≤0.115 | 13 · 7 d · +0.224 [-0.053, +0.471] · H1 +0.107 (3 d) / H2 +0.253 (4 d) | -0.143 [-0.612, +0.269] | n<15 |
| LONG_HEAT_BLOCK | BTC 5m ADX rising (ADX5_RISING) | yes | 121 · 42 d · -0.033 [-0.252, +0.180] · H1 -0.181 (19 d) / H2 +0.077 (23 d) | no | 32 · 14 d · +0.063 [-0.245, +0.324] · H1 +0.014 (6 d) / H2 +0.087 (8 d) | -0.097 [-0.404, +0.270] | half h2 flips |
| LONG_HEAT_BLOCK | BTC 7d return (R7D>0) | >0 | 99 · 36 d · -0.038 [-0.243, +0.132] · H1 -0.320 (16 d) / H2 +0.101 (20 d) | ≤0 | 54 · 11 d · +0.043 [-0.412, +0.422] · H1 +0.069 (5 d) / H2 +0.009 (6 d) | -0.081 [-0.508, +0.407] | half h2 flips |
| LONG_HEAT_BLOCK | BTC daily EMA50>EMA200 (D1_GOLDEN) | yes | 26 · 7 d · -0.070 [-0.478, +0.158] · H1 +nan (0 d) / H2 -0.070 (7 d) | no | 127 · 40 d · -0.002 [-0.212, +0.204] · H1 -0.147 (21 d) / H2 +0.133 (19 d) | -0.068 [-0.541, +0.262] | same sign |
| LONG_HEAT_BLOCK | BTC 7d return (R7D_T) | T3 >2.33 | 83 · 28 d · -0.033 [-0.282, +0.170] · H1 -0.319 (13 d) / H2 +0.122 (15 d) | T1 ≤-2.26 | 53 · 10 d · +0.027 [-0.438, +0.409] · H1 +0.069 (5 d) / H2 -0.030 (5 d) | -0.059 [-0.516, +0.450] | half h2 flips |
| LONG_HEAT_BLOCK | BTC 30d return (R30D>0) | >0 | 115 · 35 d · -0.021 [-0.198, +0.135] · H1 -0.231 (12 d) / H2 +0.058 (23 d) | ≤0 | 38 · 12 d · +0.014 [-0.572, +0.475] · H1 -0.058 (9 d) / H2 +0.400 (3 d) | -0.035 [-0.515, +0.579] | half h1 flips |
| LONG_HEAT_BLOCK | % pairs above 72h ref (bull-run 'above') (ABOVE72_T) | T3 >52.5 | 63 · 26 d · +0.007 [-0.243, +0.197] · H1 -0.277 (12 d) / H2 +0.163 (14 d) | T1 ≤48 | 44 · 12 d · -0.008 [-0.495, +0.455] · H1 +0.028 (7 d) / H2 -0.101 (5 d) | +0.015 [-0.498, +0.540] | half h1 flips |
| LONG_HEAT_BLOCK | BTC 1h EMA20 rising (H1_EMA20_UP) | yes | 76 · 28 d · -0.001 [-0.341, +0.291] · H1 -0.084 (12 d) / H2 +0.087 (16 d) | no | 77 · 27 d · -0.023 [-0.247, +0.156] · H1 -0.244 (12 d) / H2 +0.075 (15 d) | +0.022 [-0.369, +0.391] | same sign |
| LONG_HEAT_BLOCK | BTC 3d return (R3D>0) | >0 | 81 · 30 d · +0.005 [-0.234, +0.193] · H1 -0.288 (13 d) / H2 +0.123 (17 d) | ≤0 | 72 · 17 d · -0.035 [-0.370, +0.282] · H1 -0.056 (8 d) / H2 -0.009 (9 d) | +0.040 [-0.358, +0.430] | half h1 flips |
| LONG_HEAT_BLOCK | BTC close > daily EMA200 (ABOVE_D200) | yes | 57 · 16 d · +0.014 [-0.253, +0.210] · H1 +nan (0 d) / H2 +0.014 (16 d) | no | 96 · 31 d · -0.031 [-0.313, +0.224] · H1 -0.147 (21 d) / H2 +0.220 (10 d) | +0.045 [-0.320, +0.392] | half h1 thin |
| LONG_HEAT_BLOCK | BTC 1d return (R1D>0) | >0 | 66 · 26 d · +0.034 [-0.210, +0.256] · H1 -0.216 (8 d) / H2 +0.123 (18 d) | ≤0 | 87 · 21 d · -0.048 [-0.373, +0.216] · H1 -0.120 (13 d) / H2 +0.029 (8 d) | +0.081 [-0.275, +0.460] | half h1 flips |
| LONG_HEAT_BLOCK | BTC 5m ADX (ADX5_T) | T3 >28.4 | 43 · 18 d · -0.105 [-0.403, +0.114] · H1 -0.487 (6 d) / H2 +0.028 (12 d) | T1 ≤19.4 | 11 · 6 d · -0.276 [-0.591, +0.209] · H1 -0.300 (3 d) / H2 -0.235 (3 d) | +0.171 [-0.463, +0.600] | n<15 |
| LONG_HEAT_BLOCK | BTC 30d return (R30D_T) | T3 >5.75 | 71 · 24 d · -0.050 [-0.278, +0.124] · H1 -0.284 (6 d) / H2 -0.025 (18 d) | T1 ≤-1.63 | 7 · 6 d · -0.417 [-1.047, +0.152] · H1 -0.751 (5 d) / H2 +0.250 (1 d) | +0.367 [-0.224, +1.025] | n<15 |
| LONG_HEAT_BLOCK | BTC distance below 30d high (OFF30_T) | T3 >-3.98 | 63 · 24 d · -0.123 [-0.375, +0.126] · H1 -0.376 (9 d) / H2 +0.007 (15 d) | T1 ≤-8.84 | 12 · 2 d · -0.530 [-0.572, +0.099] · H1 -0.530 (2 d) / H2 +nan (0 d) | +0.406 [-0.405, +0.647] | n<15 |
| LONG_HEAT_BLOCK | 72h efficiency ≤0.007 (chop) (EFF72_CHOP) | yes | 43 · 6 d · +0.315 [-0.534, +0.597] · H1 +0.428 (2 d) / H2 +0.246 (4 d) | no | 110 · 44 d · -0.126 [-0.285, +0.036] · H1 -0.331 (20 d) / H2 +0.018 (24 d) | +0.441 [-0.410, +0.746] | days<8 |
| LONG_HEAT_BLOCK | breadth bull% > bear% (BULL_GT_BEAR) | yes | 153 · 47 d · -0.012 [-0.201, +0.171] · H1 -0.147 (21 d) / H2 +0.080 (26 d) | no | 0 · 0 d · +nan [+nan, +nan] · H1 +nan (0 d) / H2 +nan (0 d) | +nan [+nan, +nan] | n<15 |
| LONG_HEAT_BLOCK | BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 153 · 47 d · -0.012 [-0.201, +0.171] · H1 -0.147 (21 d) / H2 +0.080 (26 d) | no | 0 · 0 d · +nan [+nan, +nan] · H1 +nan (0 d) / H2 +nan (0 d) | +nan [+nan, +nan] | n<15 |
| LONG_HEAT_BLOCK | breadth bull % (BULL_T) | T3 >52.2 | 153 · 47 d · -0.012 [-0.201, +0.171] · H1 -0.147 (21 d) / H2 +0.080 (26 d) | T1 ≤24.4 | 0 · 0 d · +nan [+nan, +nan] · H1 +nan (0 d) / H2 +nan (0 d) | +nan [+nan, +nan] | n<15 |
| LONG_HEAT_BLOCK | BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 153 · 47 d · -0.012 [-0.201, +0.171] · H1 -0.147 (21 d) / H2 +0.080 (26 d) | T1 ≤44.8 | 0 · 0 d · +nan [+nan, +nan] · H1 +nan (0 d) / H2 +nan (0 d) | +nan [+nan, +nan] | n<15 |

## R. Real-fill cross-check (refute-only) — master stack blocks (as traded) and the scout's forward re-priced refusals, per state

Flag = the real cohort's state ordering points the other way (yr5 says the filter is worth more in state X; real fills say the opposite).

| gate | split | yr5 A / B avg | source | A: N · days · avg | B: N · days · avg | flag |
|---|---|---|---|---|---|---|
| MACRO:BTC_RSI_ADX_CROSS | BTC_RSI5_T | -0.121 / +0.228 | no real-fill cohort (gate never let a real fill through / no scout tracker) | – | – | n/a |
| CALM3D_DMI | VOLR_T | -0.078 / -0.270 | master (as traded) | 5 · 2 d · +0.367 | 2 · 2 d · -0.702 | AGREES |
| LONG_HEAT_BLOCK | D1_GOLDEN | -0.070 / -0.002 | master (as traded) | 3 · 2 d · -0.173 | 4 · 4 d · -0.276 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | D1_GOLDEN | -0.070 / -0.002 | forward (scout sim1) | 10 · 3 d · +0.238 | – | one state empty |
| LONG_HEAT_BLOCK | ABOVE_D200 | +0.014 / -0.031 | master (as traded) | 5 · 4 d · -0.232 | 2 · 2 d · -0.232 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | ABOVE_D200 | +0.014 / -0.031 | forward (scout sim1) | 10 · 3 d · +0.238 | – | one state empty |
| LONG_HEAT_BLOCK | H4_GOLDEN | -0.103 / +0.151 | master (as traded) | 7 · 6 d · -0.232 | – | one state empty |
| LONG_HEAT_BLOCK | H4_GOLDEN | -0.103 / +0.151 | forward (scout sim1) | 10 · 3 d · +0.238 | – | one state empty |
| LONG_HEAT_BLOCK | H1_EMA20_UP | -0.001 / -0.023 | master (as traded) | 5 · 4 d · -0.232 | 2 · 2 d · -0.232 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | H1_EMA20_UP | -0.001 / -0.023 | forward (scout sim1) | 4 · 1 d · +0.223 | 6 · 3 d · +0.248 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | ADX5_RISING | -0.033 / +0.063 | master (as traded) | 7 · 6 d · -0.232 | – | one state empty |
| LONG_HEAT_BLOCK | ADX5_RISING | -0.033 / +0.063 | forward (scout sim1) | 9 · 3 d · +0.227 | 1 · 1 d · +0.337 | AGREES |
| LONG_HEAT_BLOCK | EFF72_CHOP | +0.315 / -0.126 | master (as traded) | 3 · 2 d · -0.173 | 4 · 4 d · -0.276 | AGREES |
| LONG_HEAT_BLOCK | EFF72_CHOP | +0.315 / -0.126 | forward (scout sim1) | 4 · 1 d · +0.223 | 6 · 3 d · +0.248 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | BULL_GT_BEAR | -0.012 / +nan | master (as traded) | 7 · 6 d · -0.232 | – | one state empty |
| LONG_HEAT_BLOCK | BULL_GT_BEAR | -0.012 / +nan | forward (scout sim1) | 8 · 2 d · +0.135 | 2 · 1 d · +0.648 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | BTC_SLOPE5_UP | -0.012 / +nan | master (as traded) | 7 · 6 d · -0.232 | – | one state empty |
| LONG_HEAT_BLOCK | BTC_SLOPE5_UP | -0.012 / +nan | forward (scout sim1) | 8 · 2 d · +0.135 | 2 · 1 d · +0.648 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | R1D>0 | +0.034 / -0.048 | master (as traded) | 5 · 5 d · -0.364 | 2 · 1 d · +0.099 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | R1D>0 | +0.034 / -0.048 | forward (scout sim1) | 4 · 2 d · +0.522 | 6 · 1 d · +0.048 | AGREES |
| LONG_HEAT_BLOCK | R3D>0 | +0.005 / -0.035 | master (as traded) | 3 · 3 d · -0.459 | 4 · 3 d · -0.061 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | R3D>0 | +0.005 / -0.035 | forward (scout sim1) | 4 · 2 d · +0.522 | 6 · 1 d · +0.048 | AGREES |
| LONG_HEAT_BLOCK | R7D>0 | -0.038 / +0.043 | master (as traded) | 3 · 3 d · -0.459 | 4 · 3 d · -0.061 | AGREES |
| LONG_HEAT_BLOCK | R7D>0 | -0.038 / +0.043 | forward (scout sim1) | 4 · 2 d · +0.522 | 6 · 1 d · +0.048 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | R30D>0 | -0.021 / +0.014 | master (as traded) | 7 · 6 d · -0.232 | – | one state empty |
| LONG_HEAT_BLOCK | R30D>0 | -0.021 / +0.014 | forward (scout sim1) | 10 · 3 d · +0.238 | – | one state empty |
| LONG_HEAT_BLOCK | R7D_T | -0.033 / +0.027 | master (as traded) | 2 · 2 d · -0.321 | 4 · 3 d · -0.061 | AGREES |
| LONG_HEAT_BLOCK | R7D_T | -0.033 / +0.027 | forward (scout sim1) | 2 · 1 d · +0.396 | 6 · 1 d · +0.048 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | R30D_T | -0.050 / -0.417 | master (as traded) | 6 · 5 d · -0.148 | – | one state empty |
| LONG_HEAT_BLOCK | R30D_T | -0.050 / -0.417 | forward (scout sim1) | 8 · 2 d · +0.135 | – | one state empty |
| LONG_HEAT_BLOCK | OFF30_T | -0.123 / -0.530 | master (as traded) | 4 · 4 d · -0.322 | – | one state empty |
| LONG_HEAT_BLOCK | OFF30_T | -0.123 / -0.530 | forward (scout sim1) | 3 · 2 d · +0.545 | – | one state empty |
| LONG_HEAT_BLOCK | ADX5_T | -0.105 / -0.276 | master (as traded) | 4 · 3 d · -0.313 | 1 · 1 d · -0.921 | AGREES |
| LONG_HEAT_BLOCK | ADX5_T | -0.105 / -0.276 | forward (scout sim1) | 1 · 1 d · +0.337 | 5 · 2 d · +0.370 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | ATR5_T | +0.081 / +0.224 | master (as traded) | 2 · 2 d · -0.324 | – | one state empty |
| LONG_HEAT_BLOCK | ATR5_T | +0.081 / +0.224 | forward (scout sim1) | – | 3 · 2 d · +0.464 | one state empty |
| LONG_HEAT_BLOCK | BULL_T | -0.012 / +nan | master (as traded) | 7 · 6 d · -0.232 | – | one state empty |
| LONG_HEAT_BLOCK | BULL_T | -0.012 / +nan | forward (scout sim1) | 8 · 2 d · +0.135 | – | one state empty |
| LONG_HEAT_BLOCK | BTC_RSI5_T | -0.012 / +nan | master (as traded) | 7 · 6 d · -0.232 | – | one state empty |
| LONG_HEAT_BLOCK | BTC_RSI5_T | -0.012 / +nan | forward (scout sim1) | 7 · 2 d · +0.055 | 2 · 1 d · +0.648 | ⚠ OPPOSITE |
| LONG_HEAT_BLOCK | ABOVE72_T | +0.007 / -0.008 | master (as traded) | 5 · 4 d · -0.236 | – | one state empty |
| LONG_HEAT_BLOCK | ABOVE72_T | +0.007 / -0.008 | forward (scout sim1) | 6 · 2 d · +0.365 | – | one state empty |
| LONG_HEAT_BLOCK | VOLR_T | -0.108 / +0.323 | master (as traded) | 4 · 3 d · -0.111 | – | one state empty |
| LONG_HEAT_BLOCK | VOLR_T | -0.108 / +0.323 | forward (scout sim1) | 6 · 1 d · +0.048 | 4 · 2 d · +0.522 | AGREES |


## A. Reference: the ADMITTED yr5 momentum-long fills per state (same replica, entry +60 s)
Context only: in which BTC states does the sleeve itself make or lose money? This is a sleeve-level (window-units) question, not a filter-conditional one.

| split | state A | A: N · days · avg [CI] · H1/H2 | state B | B: N · days · avg [CI] · H1/H2 |
|---|---|---|---|---|
| BTC daily EMA50>EMA200 (D1_GOLDEN) | yes | 70 · 17 d · +0.103 [-0.044, +0.301] · +nan/+0.103 | no | 1052 · 218 d · -0.064 [-0.118, -0.012] · -0.072/-0.056 |
| BTC close > daily EMA200 (ABOVE_D200) | yes | 183 · 37 d · -0.030 [-0.151, +0.109] · +nan/-0.030 | no | 939 · 198 d · -0.058 [-0.114, -0.000] · -0.072/-0.041 |
| BTC 4h EMA50>EMA200 (H4_GOLDEN) | yes | 488 · 109 d · -0.045 [-0.119, +0.030] · -0.066/-0.036 | no | 634 · 129 d · -0.060 [-0.136, +0.016] · -0.075/-0.040 |
| BTC 1h EMA20 rising (H1_EMA20_UP) | yes | 630 · 166 d · -0.001 [-0.068, +0.068] · -0.003/+0.000 | no | 492 · 144 d · -0.125 [-0.206, -0.042] · -0.160/-0.093 |
| BTC 5m ADX rising (ADX5_RISING) | yes | 741 · 206 d · -0.027 [-0.091, +0.035] · -0.043/-0.014 | no | 381 · 149 d · -0.107 [-0.193, -0.019] · -0.128/-0.088 |
| 72h efficiency ≤0.007 (chop) (EFF72_CHOP) | yes | 138 · 51 d · -0.124 [-0.279, +0.024] · -0.140/-0.113 | no | 984 · 226 d · -0.044 [-0.102, +0.015] · -0.064/-0.026 |
| breadth bull% > bear% (BULL_GT_BEAR) | yes | 1057 · 232 d · -0.061 [-0.118, -0.007] · -0.082/-0.044 | no | 65 · 49 d · +0.067 [-0.071, +0.196] · +0.069/+0.065 |
| BTC 5m EMA20 slope > 0 (BTC_SLOPE5_UP) | yes | 1119 · 235 d · -0.054 [-0.108, -0.000] · -0.073/-0.038 | no | 3 · – |
| BTC 1d return (R1D>0) | >0 | 577 · 115 d · -0.040 [-0.115, +0.038] · -0.053/-0.029 | ≤0 | 545 · 120 d · -0.068 [-0.141, +0.012] · -0.092/-0.048 |
| BTC 3d return (R3D>0) | >0 | 566 · 121 d · -0.047 [-0.118, +0.029] · -0.053/-0.042 | ≤0 | 556 · 114 d · -0.060 [-0.142, +0.022] · -0.091/-0.034 |
| BTC 7d return (R7D>0) | >0 | 565 · 121 d · -0.036 [-0.100, +0.029] · -0.048/-0.026 | ≤0 | 557 · 114 d · -0.071 [-0.149, +0.013] · -0.094/-0.050 |
| BTC 30d return (R30D>0) | >0 | 613 · 132 d · -0.041 [-0.115, +0.032] · -0.041/-0.040 | ≤0 | 509 · 103 d · -0.069 [-0.151, +0.011] · -0.097/-0.034 |
| BTC 7d return (R7D_T) | T3 >2.33 | 386 · 82 d · +0.009 [-0.071, +0.094] · -0.013/+0.028 | T1 ≤-2.26 | 408 · 78 d · -0.050 [-0.151, +0.045] · -0.078/-0.025 |
| BTC 30d return (R30D_T) | T3 >5.75 | 364 · 75 d · -0.044 [-0.113, +0.032] · -0.002/-0.057 | T1 ≤-1.63 | 420 · 83 d · -0.061 [-0.150, +0.026] · -0.094/-0.021 |
| BTC distance below 30d high (OFF30_T) | T3 >-3.98 | 384 · 88 d · -0.000 [-0.080, +0.087] · +0.038/-0.018 | T1 ≤-8.84 | 449 · 88 d · -0.074 [-0.172, +0.016] · -0.101/-0.031 |
| BTC 5m ADX (ADX5_T) | T3 >28.4 | 302 · 125 d · -0.076 [-0.173, +0.019] · -0.082/-0.073 | T1 ≤19.4 | 88 · 45 d · -0.098 [-0.284, +0.088] · -0.128/-0.031 |
| BTC 5m ATR % (ATR5_T) | T3 >0.185 | 515 · 134 d · -0.040 [-0.126, +0.045] · -0.049/-0.025 | T1 ≤0.115 | 206 · 81 d · -0.094 [-0.197, +0.015] · -0.144/-0.078 |
| breadth bull % (BULL_T) | T3 >52.2 | 950 · 227 d · -0.048 [-0.104, +0.015] · -0.053/-0.044 | T1 ≤24.4 | 12 · 12 d · -0.017 [-0.354, +0.223] · -0.064/+0.037 |
| BTC 5m RSI (BTC_RSI5_T) | T3 >55.4 | 1064 · 232 d · -0.057 [-0.111, -0.003] · -0.074/-0.043 | T1 ≤44.8 | 1 · – |
| % pairs above 72h ref (bull-run 'above') (ABOVE72_T) | T3 >52.5 | 392 · 98 d · -0.052 [-0.135, +0.026] · -0.067/-0.043 | T1 ≤48 | 413 · 99 d · -0.062 [-0.156, +0.031] · -0.139/+0.017 |
| BTC daily quote-vol / 30d mean (VOLR_T) | T3 >1.13 | 387 · 77 d · -0.065 [-0.159, +0.028] · -0.100/-0.039 | T1 ≤0.817 | 367 · 82 d · -0.035 [-0.114, +0.052] · -0.010/-0.060 |

## Blind spots (what this could NOT test)
- **Chain gates are "first refusing gate" only.** A BLOCK at chain gate X says nothing about the chain gates after X. Removing X alone might still have been refused. Similarly, sole `MACRO:` FAILS know only the first macro veto, and the ladder's last-mile gates are not evaluated when the ladder fails. The blocked cohorts are therefore upper bounds on what each gate alone refuses.
- **Entry convention.** Entry is at t+60 s, taker, with the 2026-10-05 exit stack. Signals are not sized; this is 1× per signal. The heat block's 30-min episode rule is approximate.
- **Daily golden-cross regime is untestable on yr5.** EMA50>EMA200 held only from Sep 12 (7 days with heat signals, none in Jan–Apr). Any rule keyed on it has no half-year replication until more bull-regime days accumulate. The same applies to "close > daily EMA200" in H1.
- **Some state × gate combinations are structurally empty.** The gate's own condition fixes the state: for example, heat requires bull ≥85, so breadth / BTC-slope splits have one side empty. These cells are "n<15", not "refuted".
- **"Market volume ratio" is BTC's own daily quote volume vs its 30-day mean.** The engine's global volume ratio is not in the SCAN lines.
- **The null permutes days within month.** Month-level regimes (daily golden cross, 30d return) are mostly preserved by it. For those variables the null is close to the observed tags, so their null count is conservative in the other direction. Read daily-trend cells with that in mind.
- **The real-fill check is thin.** Master heat blocks: 7 fills. Forward heat: 10 signals over 3 days. LOADX / MEGACAP / CALM3D have small master cohorts. MACRO / chain gates have no real cohort at all, because the live stack never lets those fills through.
- **The 30–50 % in-sample haircut was not applied**, because nothing reached a ship candidate.

## Files
- `scripts/filter_regime_extract.py`, `scripts/filter_regime_price.py`, `scripts/filter_regime_matrix.py`, `scripts/filter_regime_report.py`: the pipeline. Extract and null stages need `S=<scratch dir>`.
- `reports/FILTER_REGIME_MATRIX_signals_priced.csv`: every priced signal (gate, pair, t, n_seeds, pct, exit reason, source).
- `reports/FILTER_REGIME_MATRIX_2026-10-05.csv`: the full matrix (840 cells, all stats, verdict, gap_screen).
- `reports/FILTER_REGIME_MATRIX_2026-10-05_gates.csv`, `_null.csv`, `_summary.json`, `_coverage.csv`.
