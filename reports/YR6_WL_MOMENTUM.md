# Two-year winners vs losers: MOMENTUM (MOM-long and MOM-short), 2026-10-13

Brief: `reports/YR6_WL_STUDY_BRIEF.md`. Data: `ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv` (today's re-price; for momentum `pct_new` = `pct`, and the % figures are leverage-invariant, so they are already at 1×). The 3 seeds are pooled and every CI is clustered by UTC day. Real fills come from `MASTER_POOL_stacked.csv` (stack 2026-10-13a).
Nothing was shipped, committed or edited in config or the engine. No network calls were made.

## Plain-English summary

1. **Momentum long has no entry variable that makes it profitable in both bull regimes.**
   - Every one of ~200 single-variable splits and ~3,000 two-variable quadrants was tested.
   - Not one zone holding ≥ 5 % of the 2025 fills is reliably positive in 2025. The best is +0.01 % at P 0.55.
   - None of the 2,374 zones that are positive in 2026-H2 is positive in 2025.
   - Patterns found in one bull regime confirm in the other at or below the luck rate:
     - discovered in 2025, confirmed in H2: 15 of 499 (luck would give ~50);
     - discovered in H2, confirmed in 2025: 22 of 448 (luck ~45).
2. **The 2025 loss is uniform, not concentrated.**
   - All 24 cohorts with N ≥ 60 lose in 2025. The cohorts cover pair rank, hour, cell, pattern, confidence, quality score, BTC regime, top-50 status and order type.
   - 12 of 12 months in 2025 are negative.
   - The top-2 pairs carry 9 % of the pair losses and the worst day 4 %.
   - Per CLAUDE.md, uniform degradation points to a regime-level (sleeve-level) cause, not an entry filter.
3. **The bull/bear label is not the driver.**
   - Momentum long does better in the "bear" half (−0.085) than in the 2025 bull (−0.187).
   - It improves steadily over time: 2025 −0.19 → 2026-H1 −0.09 → 2026-H2 −0.02.
   - That time gradient fits the momentum filter stack having been fitted on Jun–Oct 2026 live fills: 2026-H2 is that stack's in-sample period, and 2025 is fully out of sample.
4. **The replay and the real fills do not actually disagree over the months they share.**
   - Real as-traded momentum long (188 fills, Jun-18 → Oct-12) averages **+0.039 %**.
   - The replay inside the live-active spans averages **+0.027 %**.
   - The +0.160 % / +$1,564 figure is the *stack-kept* subset (136 fills). Today's filters were fitted on those same live fills: they remove 52 live losers (−0.276 %), but in the replay the same filters keep the similar signals, and those lose.
   - On the 76 trades both sides took, the exit replica is faithful: real +0.233 vs replay +0.192, median gap 0.000, 91 % sign agreement.
   - The −0.187 % headline comes from 2025, a period with **zero real fills**.
5. **Momentum short** follows the same pattern (−0.081 / −0.033 / −0.136, with 22/24 cohorts negative in 2025). No separator confirms above chance in either bull direction, and no zone keeps it positive in both bulls.
6. **Only one entry split holds its sign in all three regimes with P ≥ 0.90 in each: momentum-long pair ADXΔ (`entry_adx_delta`).**
   - Falling ADX at entry is worse: the bottom-tercile Δ is −0.06 / −0.21 / −0.09, and the `>0` split is +0.07 / +0.12 / +0.07.
   - It has the same sign on the real master fills.
   - This variable is **already registered observe-only** (ML_ADX_RISING_IN_REGIME, DECISION_LOG 273, step ②). This study adds two years of out-of-sample support for it.
   - The same data **refutes step ①, BTC24H_DOWN (proxy)** in 2025 and in H1: the blocked zone does better than the kept side.
   - ADXΔ does **not** turn 2025 positive: the kept side is −0.161.

**Recommendation: no change to filters or config.** Nothing passes discovery → confirmation, and the only consistent variable is already under observation. Details and pre-committed gates are in section 6.

---

## 1. Sleeve × regime (pooled seeds, day-clustered 95 % CI)

| sleeve | regime | N (N/seed) | days | WR | avg % | 95 % CI | breakeven WR | avg win / loss |
|---|---|---|---|---|---|---|---|---|
| MOM-long | 2025 BULL | 2617 (872) | 301 | 53.1 % | **−0.187** | [−0.229, −0.145] | 68.7 % | +0.37 / −0.82 |
| MOM-long | 2026-H1 BEAR | 1171 (390) | 154 | 59.4 % | −0.085 | [−0.153, −0.019] | 66.2 % | +0.42 / −0.83 |
| MOM-long | 2026-H2 BULL | 521 (174) | 79 | 63.1 % | −0.015 | [−0.119, +0.101] | 64.4 % | +0.44 / −0.79 |
| MOM-short | 2025 BULL | 705 (235) | 118 | 57.4 % | −0.081 | [−0.171, +0.002] | 65.0 % | +0.38 / −0.70 |
| MOM-short | 2026-H1 BEAR | 381 (127) | 73 | 61.4 % | −0.033 | [−0.134, +0.063] | 64.7 % | +0.36 / −0.66 |
| MOM-short | 2026-H2 BULL | 88 (29) | 23 | 53.4 % | −0.136 | [−0.295, +0.029] | 66.4 % | +0.35 / −0.70 |

- **Computed two ways:** pandas + day-bootstrap, and the raw `csv` module. All 8 cells match with 0 mismatches. N/seed, WR and avg also match `ENGINE_REPLAY_YR6_2Y_TODAY.txt`.
- **Per seed, MOM-long 2025:** −0.171 / −0.207 / −0.183, so all three seeds agree.
- **Real master momentum long:** 136 kept · 72.8 % · +0.160 · +$1,564 (breakeven WR 61.0 %), against as-traded 188 · 66.0 % · +0.039.
- **Real master momentum short:** 37 kept · +0.173, against as-traded 49 · +0.028.

**Exit mix, MOM-long:**

| regime | stop-loss share | RUNNER_TRAIL share | avg peak | losers that peaked ≥ +0.20 |
|---|---|---|---|---|
| 2025 BULL | 39 % | 47 % | +0.48 | 23 % |
| 2026-H1 BEAR | 32 % | 51 % | +0.55 | 25 % |
| 2026-H2 BULL | 30 % | 54 % | +0.59 | 26 % |

The 2025 entries simply follow through less: more trades stop out before arming. This is not an exit-shape problem.

## 2. Coverage

- 74 signal columns are stamped on 100 % of fills in every sleeve × regime cell (30 macro, 44 pair).
- Plus 8 categorical dummies (BTC regime, macro trend).
- **Partial coverage:**
  - `entry_vol_mcap_ratio` / `entry_mcap_usd` / `entry_cmc_rank`: 88–96 %, tested.
  - `entry_pair_1h_ema20_200_gap_pct`: 96–100 %, tested.
  - `entry_chop_burst_prior_fill_s`: 20–47 %. Given its own read on scored rows only: no consistent split. 2025 >6 s −0.276 vs ≤6 s −0.164; H2 −0.067 vs −0.106, the opposite order.
- **Never stamped in the replay (43 columns, cannot test):**
  - `entry_br_*` (9), `entry_bear_*` (5), `entry_frenzy_*` (20), `entry_surge_*` (4);
  - `entry_funding_rate`, `entry_pair_1d_ndi`, `entry_btc_4h_ema50_200_gap_pct`, `entry_btc_regime_started_at`, and the bracket columns.
- **Real master momentum-long coverage:**
  - The BTC 72 h / 24 h / 30 d stamps cover only 29–37 % of fills (eras ≥ B9).
  - The gap_5_20 / btc_1d_ret / eth / 1h-ema200 stamps cover 12–15 % (≥ B15).
  - So the master check of those variables rests on few fills.

Full table: `reports/YR6_WL_MOMENTUM_base.txt` §B.

## 3. Exhaustive screen, null and out-of-sample confirmation

**Method** (`study_yr6wl_momentum_screen.py`):
- **1D splits:** every column (coverage ≥ 80 %) at sign, median, top tercile and bottom tercile. Binary columns are tested as ==1.
- **2D splits:** median × median quadrants over every pair of the top-40 columns, giving 3,120 tests per run.
- **Statistic:** Δ = mean(in) − mean(out). Pair variables use fill units. Macro variables use **day units**: each day's in-mean and out-mean count as one observation each.
- **Null:** 200 circular shifts of the time-sorted outcome. This keeps the same-day clustering of outcomes and features but breaks their alignment.
- **Reported per test:** the p-value, and the family-wise max-|Δ| rank.
- **Confirmation:** thresholds are frozen in the discovery regime. A test confirms if the other bull regime has the same sign with day-bootstrap P ≥ 0.90.

| run | 1D valid · p<.05 (luck) · family-wise p<.05 | 2D valid · p<.05 (luck) · family-wise | survivors → confirmed (luck ≈ 10 %) |
|---|---|---|---|
| ML disc 2025 → H2 | 186 · 14 (9) · 0 | 2975 · 485 (149) · 0 | 499 → **15** (~50) |
| ML disc H2 → 2025 | 188 · 17 (9) · 0 | 2969 · 431 (148) · 0 | 448 → **22** (~45) |
| ML bear (H1) | 186 · 2 (9) · 0 | 3054 · 186 (153) · 0 | — |
| MS disc 2025 → H2 | 182 · 6 (9) · 0 | 3097 · 125 (155) · 0 | 131 → 15 (~13) |
| MS disc H2 → 2025 | 157 · 9 (8) · 0 | 1874 · 266 (94) · 1 | 275 → 49 (~28)* |
| MS bear (H1) | 184 · 22 (9) · 0 | 3085 · 596 (154) · 0 | — |

\* MS H2 has only 88 fills over 23 days. Its "discoveries" are 20–40-fill quadrants, and the excess confirmations collapse onto one theme, BTC RSI (below).

**Reading the table:**
- No 1D or 2D test survives the family-wise null in any momentum-long run.
- The 2D excess at p < .05 comes from quadrants built on columns that were pre-selected by their 1D p-value on the same data. That inherited signal does **not** carry over to the other bull: confirmation runs at or below the luck rate in both directions for momentum long.

**Positive-pocket search** (any zone ≥ 5 % of fills and ≥ 15 days with mean > 0):
- ML 2025: 5 pockets, best +0.013 at P 0.55. **0** are positive in H2.
- ML H2: 2,374 pockets. **0** are positive in 2025.
- MS 2025: 339 pockets, 62 also positive in H2. 0 have P ≥ 0.90 in both.
- MS H2: 55 pockets. 0 are positive in 2025.

### All-regime consistency (thresholds frozen in 2025; computed a second, independent way with pandas)

`study_yr6wl_momentum_consistency.py` covers every 1D split across the three regimes plus the master fills.

**MOM-long:** 197 splits.
- Same sign in all 3 regimes: **46**, against a coin-flip expectation of 49.
- P ≥ 0.90 in both bulls: 5. P ≥ 0.90 in all three: **1**.

| split (MOM-long) | Δ 2025 (P) | Δ H2 (P) | Δ H1 (P) | master all Δ (N in) | master kept Δ |
|---|---|---|---|---|---|
| `entry_adx_delta` ≤ q33 (−0.053) | −0.062 (.93) | −0.213 (.99) | −0.088 (.93) | −0.113 (71) | −0.062 |
| `entry_adx_delta` > 0 | +0.072 (.95) | +0.124 (.91) | +0.067 (.87) | +0.082 (111) | +0.016 |
| BTC regime == CHOPPY_WEAK | −0.103 (.89) | n<10 | −0.126 (.84) | — (7) | — |
| `entry_btc_off24lo_pct` > q67 (2.57) | +0.065 (.82) | +0.055 (.95) | +0.100 (.98) | (7, cov 32 %) | — |
| `entry_btc_ema50_100_gap_pct` > q67 | +0.069 (.94) | +0.074 (.91) | +0.044 (.79) | (cov 12 %) | — |

**MOM-short:** 193 splits.
- Same sign in all 3 regimes: 50, against an expectation of 48.
- P ≥ 0.90 in both bulls: 4. In all three: 0.
- **Best, `entry_pos_di` > q67 (13.09):** Δ +0.115 (.96) / +0.170 (.88) / +0.148 (.95); master all +0.338 (19 in).
- **BTC RSI theme:** shorts lose when BTC is *less* oversold at entry. `entry_btc_rsi_prev` > 40.4 gives Δ −0.031 / −0.389 / −0.096. `entry_btc_rsi` > q67 gives −0.146 / −0.445 / −0.088 with master −0.266.
- **Common thread:** both sleeves lose most on *late-in-the-move* entries. For momentum long that is pair ADX already rolling over; for momentum short it is +DI already exhausted and BTC already bounced. This is a plausible mechanism, but the size is small (Δ ≈ 0.06–0.15 %).

### Expectancy bar on the blocked zone

The bar is the locked one from CLAUDE.md:
- WR below the sleeve's breakeven WR;
- avg < 0 at 95 % confidence (day-clustered bootstrap);
- ≥ 8 windows, with no single window or pair carrying ≥ 50 % of the loss;
- N ≥ 15.

**ML `adx_delta` ≤ q33:**

| sample | N | days | WR | avg % | 95 % CI | bar | kept side |
|---|---|---|---|---|---|---|---|
| 2025 BULL | 873 | 227 | 51.4 % | −0.228 | [−0.30, −0.15] | PASS | −0.166 |
| 2026-H1 BEAR | 399 | 116 | 56.9 % | −0.143 | [−0.24, −0.03] | PASS | −0.055 |
| 2026-H2 BULL | 179 | 48 | 55.3 % | −0.155 | [−0.29, −0.00] | PASS | +0.058 |
| master as-traded | 71 | 46 | 62.0 % | −0.031 | [−0.21, +0.17] | **FAIL** | +0.082 |

**Caveat:** in 2025 and H1 the *whole sleeve* clears this bar, because it is below breakeven everywhere. A filter can only help when the kept side is positive, and that happens only in H2.

**MS `pos_di` ≤ 13.09:** passes only in 2025 (−0.119, kept side −0.004). It fails in H1, H2 and on the master fills.

## 4. Why the replay disagrees with the real fills (question 2)

Overlap window: 2026-06-18 → 2026-10-12. Script: `study_yr6wl_momentum_replay_vs_real.py`.

| MOM-long | N | WR | avg % | 95 % CI |
|---|---|---|---|---|
| REAL as-traded (all non-probe) | 188 | 66.0 % | **+0.039** | [−0.08, +0.17] |
| REAL stack-kept (today's filters, in-sample) | 136 | 72.8 % | +0.160 | [+0.02, +0.31] |
| REAL blocked by today's stack | 52 | 48.1 % | −0.276 | [−0.47, −0.09] |
| REPLAY, same calendar window | 197/seed | 63.6 % | −0.004 | [−0.10, +0.10] |
| REPLAY, inside live-active era spans | 151/seed | 64.5 % | **+0.027** | [−0.09, +0.14] |
| ex washed-out window: real as-traded / real kept / replay live spans | 167 / 116 / 122 per seed | | −0.031 / +0.083 / −0.010 | |

- **At the outcome level, real as-traded and the replay agree** (+0.04 vs +0.03, and −0.03 vs −0.01 ex-wash).
- The gap exists only against the **stack-kept** figure. Today's momentum filters (LOADX, CALM3D, HEAT, MEGACAP, CHOP_BURST, …) were fitted on exactly these live fills, so their +0.16 is in-sample (DECISION_LOG 204 already flags this).
- **Trade level (same pair, open within ±15 min, any seed; two independent matchers agree):**
  - 76 of 188 real fills are reproduced. Reproduction rate: 55 % of kept vs 2 % of blocked fills; 46 % of real winners vs 30 % of real losers.
  - **Same trades:** real +0.233 vs replay +0.192, mean gap +0.042, **median 0.000**, 91 % sign agreement. The exit replica is fine.
  - Real fills the replay did not take: −0.093. Replay fills live did not take: −0.090 (−0.083 inside live spans). The two sides lose by the same amount.
  - Inside the replay, the fills live also took average +0.231, while the rest average −0.090.
  - The live-taken subset looks better mainly because the as-was configs and slot timing differed: replay parity with the *as-was* config is high (`ENGINE_REPLAY_YR6_PARITY_*`). It is not a hidden entry edge (memory: replay_selection_bias).
- **Cohort (entry_* distributions):** replay window vs real kept, momentum long.
  - 4/66 columns differ at KS 5 % (about 3 expected by chance).
  - All 4 are BTC macro stamps scored on only 30–37 % of real fills. Real kept fills sat nearer BTC's highs: off30d −3.0 vs −4.2 %, off24h −0.39 vs −0.79 %, above72 54 vs 51.
  - 11 columns have |SMD| ≥ 0.3, mostly low-coverage ones.
  - This is not a cohort mismatch large enough to explain anything. The pair-level stamps (RSI, ADX, gaps, rank, volume) match closely.
- **Momentum short, same picture:**
  - Real as-traded +0.028 vs replay live spans −0.024.
  - Kept +0.173 is in-sample (12 blocked fills at −0.420).
  - Matched trades: real +0.174 vs replay +0.081, 96 % sign agreement.
- **Conclusion for (2):** the "disagreement" is two things:
  1. **Period:** the −0.187 comes from 2025, where there are no real fills.
  2. **In-sample selection** of the stack-kept real fills.

  Where the two sources overlap and are compared like for like (as-traded vs replay), they agree within about 0.01–0.05 %/fill.

## 5. Regime read: uniform or concentrated? (question 3) — sleeve-kill checklist ①–④

- **Uniform (MOM-long):**
  - Cohorts negative: 24/24 in 2025, 20/22 in H1, 17/22 in H2.
  - Months negative: 12/12 in 2025, and 19/22 over the two years (only 2026-04, 09, 10 are positive).
  - Losing pairs: 68 % in 2025. Losing days: 67 %.
  - Excluding the worst-5 pairs only moves 2025 from −0.187 to −0.165.
  - Top-50 vs non-top-50 in 2025: −0.180 vs −0.213. The 1m-exit fidelity cannot explain this (bias ≤ 0.03).
- **Uniform (MOM-short):** 22/24 cohorts negative in 2025. The top-2 pairs carry 17 % of the losses.
- **Checklist items ①② (every pair and macro dimension, sign then buckets)** were run exhaustively above, plus 2D quadrants with a null. Nothing is direction-consistent above chance except ADXΔ (ML) and the pos_di / BTC-RSI theme (MS, weak).
- **Item ③ (uniform-degradation test):** yes. The cause is regime-level or out-of-sample, not entries.
- **Item ④ (tape context, `study_yr6wl_momentum_tape.py`):**
  - Day-level BTC stamps on momentum-long fill days are almost identical across regimes (BTC RSI ≈ 62.5, eff72 ≈ 0.03).
  - The 2025 differences: breadth was higher (bull_pct 78 vs 71), BTC was further below its 30d high (−7.5 vs −4.1 %), BTC ADX was lower (23.6 vs 26.3), and BTC ATR was higher (0.153 vs 0.132).
  - **Month-unit read (22 windows):** the monthly momentum-long avg correlates with the monthly median BTC ADX (ρ +0.70, with +0.74 in 2025 and +0.75 in 2026). It also correlates with breadth (−0.66), BTC ATR (−0.45) and eff72 (−0.43), with the same sign in both halves.
  - **But:**
    - At entry level the BTC-ADX split is non-monotonic and inconsistent within regimes (2025 buckets −0.27 / −0.19 / −0.22 / −0.08 / −0.06; H1 and H2 not ordered). That is a confound by the project rule.
    - Even the highest-ADX 2025 months are negative (Dec −0.09, Sep −0.10).
    - The between-month correlation is largely the 2025 → 2026 time trend.
- **So:** no measured macro variable gives a regime switch that would make momentum long positive in 2025.
- **The leading explanation** is that the momentum filter stack is in-sample to Jun–Oct 2026: the gradient is −0.19 → −0.09 → −0.02, peaking in the fitted period.
- **What could not be tested:** the replay logs fills only, not the blocked signals, so the filter-fit hypothesis cannot be proven directly here.

**Additional observation (signal persistence, `study_yr6wl_momentum_seed_agreement.py`):** fills that all 3 seeds take (the signal held across scan phases) beat fills only 1 seed takes in 2026:

| MOM-long regime | 3/3 seeds | 1/3 seeds |
|---|---|---|
| 2026-H2 BULL | +0.117 (P 0.88) | −0.172 |
| 2026-H1 BEAR | −0.027 | −0.154 |
| 2025 BULL | **−0.210** (worst cohort) | −0.194 |

It fails in 2025, so it is not a two-bull separator. Recorded as an idea only.

## 6. Recommendation: no change

- **No filter or config change for MOM-long or MOM-short.**
  - No separator confirms out of sample above the luck rate.
  - No zone keeps either sleeve positive in both bull regimes.
  - The 2025 loss is uniform.
  - The only consistent variable (ML ADXΔ) is already registered observe-only.
- **ML ADXΔ (registered ML_ADX_RISING_IN_REGIME, DECISION_LOG 273, step ②): keep observe-only, with frozen definition and bar unchanged.**
  - This study is two-year support for it: same sign in 2025, H1 and H2 at P ≥ 0.90 each, and the same sign on master all and kept.
  - It **fails** the expectancy bar on the real master fills (zone −0.031, CI [−0.21, +0.17]).
  - It does not make 2025 positive (kept −0.161).
  - Projected Δ is about +0.07–0.12 %/kept fill on the replay. After the 30–50 % haircut that is about +0.04–0.08, and that is not enough to change the sleeve's sign out of sample.
- **Re-scope proposal for the operator (pre-registered, not armed):** read ADXΔ ≤ 0 **standalone**, not only inside the BTC24H_DOWN keep. The threshold is the natural 0, not fitted.
  - **Frozen bar (fresh fills from 2026-10-14, window units):** at ≥ 15 zone fills on ≥ 8 days, it becomes an arm candidate only if all of these hold: WR < ML breakeven (≈ 61–64 %), day-bootstrap P(mean < 0) ≥ 0.95, no day or pair ≥ 50 % of the loss, and zone mean < kept mean.
  - **RETIRE** if the zone mean ≥ the kept mean at N 15.
  - **If armed later, revert gate:** ≥ 10 refused signals on ≥ 5 days with WR ≥ 60 % or Σ > 0.
- **BTC24H_DOWN (DL 273, step ①):**
  - On the 2-year replay with the proxy `entry_btc_1d_ret_pct ≤ 0`, the blocked zone does *better* than the kept side in 2025 (−0.169 vs −0.207) and in H1 (−0.077 vs −0.095). It is worse only in H2 (−0.100 vs +0.058).
  - Old-period evidence may only refute (memory: crossperiod_refute_only), so this counts **against** arming.
  - Its exact registered definition (closed 5m vs 288 bars back) is not stamped, so the registered gate stays the decider.
- **MOM-short:** no candidate meets the gates. pos_di ≤ 13 and BTC RSI > ~40.5 are watchlist-only: one regime passes and the H2 sample is 88 fills / 23 days. No registration is proposed.
- **Sleeve-level, the quant read (not a kill proposal):** at today's stack, momentum long has no demonstrated out-of-sample edge.
  - Real as-traded: +0.039 [−0.08, +0.17].
  - Replay over two years: −0.139 [−0.17, −0.10].
  - The positive real figure is the in-sample kept subset.
  - The checklist items run are ①②③④ (above). Item ③ says regime / out-of-sample cause, so the forward fills at today's stack remain the judge (DL 204 reading rule).

## 7. Could NOT be tested

- **43 entry_* columns never stamped in the replay** (`br_*`, `bear_*`, `frenzy_*`, `surge_*`, `funding_rate`, `pair_1d_ndi`, `btc_4h_ema50_200_gap_pct`, …).
- **`entry_chop_burst_prior_fill_s`:** 20–47 % coverage; own read only.
- **Signals blocked by today's filters:** the replay records fills only. So the in-sample-filter hypothesis, and what the stack removes in 2025 vs 2026, cannot be measured directly.
- **BTC24H_DOWN exact definition** (closed-bar 288-back return) and **closed-bar ADXΔ:** proxies only (`entry_btc_1d_ret_pct`, forming-bar `entry_adx_delta`).
- **Real-fill macro stamps:** cover only 12–37 % of master momentum-long fills (later eras). The master checks of those variables rest on 7–54 fills.
- **MOM-short 2026-H2:** 88 fills / 23 days, so confirmation in that direction is under-powered.
- **Fidelity limits:**
  - Non-top-50 pairs in 2025 exit on 1m bars (bias ≤ 0.03).
  - Entries come from the 01419d4 config.
  - Second-order slot / concurrency effects of today's sizing are not modelled.
  - 2025 mcap is a proxy.
- **Washed-out window (Jun-18 → Jul-2):** shown with and without in §4. It is not in 2025.

## 8. Exact commands (cache-only, about 1 minute total)

```
venv/bin/python scripts/study_yr6wl_momentum_base.py            > reports/YR6_WL_MOMENTUM_base.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_screen.py ML "2025 BULL"    "2026-H2 BULL" > reports/YR6_WL_MOMENTUM_screen_ML_2025.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_screen.py ML "2026-H2 BULL" "2025 BULL"    > reports/YR6_WL_MOMENTUM_screen_ML_H2.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_screen.py ML "2026-H1 BEAR" ""             > reports/YR6_WL_MOMENTUM_screen_ML_BEAR.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_screen.py MS "2025 BULL"    "2026-H2 BULL" > reports/YR6_WL_MOMENTUM_screen_MS_2025.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_screen.py MS "2026-H2 BULL" "2025 BULL"    > reports/YR6_WL_MOMENTUM_screen_MS_H2.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_screen.py MS "2026-H1 BEAR" ""             > reports/YR6_WL_MOMENTUM_screen_MS_BEAR.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_consistency.py ML > reports/YR6_WL_MOMENTUM_consistency_ML.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_consistency.py MS > reports/YR6_WL_MOMENTUM_consistency_MS.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_replay_vs_real.py > reports/YR6_WL_MOMENTUM_replay_vs_real.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_seed_agreement.py 15 > reports/YR6_WL_MOMENTUM_seed_agreement.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_registered.py > reports/YR6_WL_MOMENTUM_registered.txt
venv/bin/python -W ignore scripts/study_yr6wl_momentum_tape.py > reports/YR6_WL_MOMENTUM_tape.txt
```

**Shared helpers:** `scripts/study_yr6wl_momentum_common.py`.

**CSVs:**
- `reports/YR6_WL_MOMENTUM_screen_{ML,MS}_{2025BULL,2026BULL,2026BEAR}{,_alltests,_pockets}.csv`
- `reports/YR6_WL_MOMENTUM_consistency_{ML,MS}.csv`
- `reports/YR6_WL_MOMENTUM_replay_vs_real_{MOM-long,MOM-short}.csv`
- `reports/YR6_WL_MOMENTUM_seed_agreement.csv`
