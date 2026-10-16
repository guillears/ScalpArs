# Momentum LONG: is there a 2D cell inside PAIR_1H_DOWNTREND? (2026-10-16)

- **Operator question:** "is there not a 2D with PAIR_1H_DOWNTREND?"
- **The zone:** the momentum-long watch line PAIR_1H_DOWNTREND (DECISION_LOG 258), meaning the coin's 1h EMA20 is at or below its 1h EMA200. Its current read is 18 fresh fills · 33 % WR · −0.15 %, frozen at KEEP OBSERVING.
- **Pre-registration:** `reports/ML_DOWNTREND_2D_PREREG.txt`, written before any 2D table was computed. SHA-256 `79a1f0ffa2691fd60fec815ec579f9a7f96fe9fbed52e213782729e13d861edb`.
- **Safety:** research only. Nothing was shipped or committed. No engine, config, template or test file was touched, and no bot API was called. `backtest_cache` and `engine_replay*.py` were only read.
- **Binance:** 2 public 5m klines requests in total: APTUSDT, plus BTCUSDT to fill a 30-hour hole in the cached BTC series. The shared cooldown file was honoured. Other processes had already pushed the IP's used weight to 547–870, so fetching stopped there.
- **Validation:** `scripts/validate_against_master.py` reports **ALL CHECKS PASS**.
- **Parity checks:** every block matches the earlier studies exactly:
  - replay: 2617 / 1171 / 521 fills at −0.187 / −0.085 / −0.015 %, the same as ML_WINNERS_LOSERS_2026-10-15;
  - master: 191 fills · 64.9 % · +0.027 %;
  - fresh line cohort: 18 · 33 % · −0.153 %, the same as `scout_ml_trend_lines`.

---

## Plain-English answer

1. **No.** There is no second variable that splits the downtrend zone into a losing part and a fine part in a way that holds in the backtest **and** in master **and** in the fresh fills.
   - None of the 7 theory-driven splits holds.
   - None of the 646 valid cells in the exhaustive screen holds.
   - Nothing makes the zone itself profitable either: no DT ∧ X cell is above zero in all three backtest eras.
2. **The zone is not a stable loser in the backtest to begin with:**

   | era | DT avg %/fill | rest avg %/fill |
   |---|---|---|
   | 2025 | −0.183 | −0.193 |
   | 2026-H1 | −0.091 | −0.079 |
   | 2026-H2 | −0.156 | **+0.053** |

   - It is only worse than the rest in 2026-H2.
   - A second variable cannot fix a 1D split that does not exist in 2 of the 3 years.
3. **The theory gets no support:**
   - Neither the "counter-trend bounce into overhead EMAs" story nor the "early reversal" story shows up.
   - None of these second variables moves the zone the same way in all three eras: how deep the coin is below EMA200, whether the 1h gap is narrowing, whether price has reclaimed the 1h EMA200, BTC 1h direction, the coin's 4h trend, ADX direction, or volume.
   - The closest one, "coin 4h EMA20 ≤ EMA50 is worse", holds in 2025 (p 0.044, which misses the Bonferroni bar of 0.0071) and in 2026-H1. It flips in 2026-H2.
4. **The only thing that points the same way everywhere is coin age (listed more than about 2.4 years ago):**
   - Inside the zone, old coins lose more:
     - backtest −0.25 vs −0.12 (2025) and −0.18 vs −0.10 (2026-H2);
     - master −0.34 vs +0.09;
     - fresh −0.28 vs +0.17.
   - **But it is not a downtrend effect.** Old coins lose just as much **outside** the zone. The interaction (in-zone Δ minus out-of-zone Δ) is −0.01 / +0.04 / −0.01 across the three eras.
   - So "DOWNTREND × OLD" is simply the **coin-age theme already found on 10-15**. That is the "ADX falling on an *established* coin" frozen sub-tally on the ADX line (DECISION_LOG 273). The downtrend part adds nothing to it.
   - The statistics are no better than luck:
     - family-wise p = 1.00;
     - 4 cells confirmed walk-forward against a shuffled-null median of 7;
     - in 2026-H1 the effect is −0.01.
   - **The first live fills go the other way.** The 4 live DT ∧ OLD fills went 2 W / 2 L at +0.165 %, including PEPE +1.16 and DOT +0.96.
5. **Verdict: NOTHING.** No new observe line is warranted.
   - PAIR_1H_DOWNTREND stays a 1D observe line with its frozen gate, unchanged.
   - The coin-age read belongs to the existing established-coin sub-tally, not to a new line.

---

## STEP 0 · Theory (written before the data; full text in the prereg)

A momentum long on a coin in a 1h downtrend is buying a 5m pop inside a bigger fall. Two stories compete:

- **(a) Counter-trend bounce, which fails.** The pop runs into overhead supply: the falling 1h EMAs and the holders who bought higher. Short covering lifts price for a few minutes, then the 1h trend resumes and the stop is hit.
  - This should be worse when the coin is deep below EMA200, the gap is still widening, price is still under EMA200, BTC is falling, the 4h trend is down, and ADX is falling.
- **(b) Early reversal, which works.** The coin has already turned: the gap is narrowing, price has reclaimed EMA200, the 4h trend is turning, BTC is rising, volume is expanding and ADX is rising. The 5m signal is then the first leg of a new trend.

**Prediction:** the zone should split along those seven axes.

## STEP 1–2 · Data

| block | source | N (DT zone / all) | zone scoring |
|---|---|---|---|
| ③ backtest 2025 / 2026-H1 / 2026-H2 | ENGINE_REPLAY_YR6_2Y_TODAY_fills (today's exits), MOM-long, non-probe, 3 seeds pooled, `pct_new` | 1530/2617 · 577/1171 · 170/521 | stamp, with 42 H1 fills rebuilt from klines |
| ② master | MASTER_POOL_stacked, momentum LONG, non-probe; as-traded % (leverage-invariant = 1×) and the stack_keep view (today's stack) | 66/191 (stack_keep 50/139) | stamp 25, kline rebuild 166 |
| ① fresh | the line's own counted cohort (`scout_ml_trend_lines.load_orders`, ≥ 09-30, ex-washed), all entry_* re-read from the exports via `export_glob`, incl. `scalpars_orders_live_2026-10-16_15-19-54 (1).csv`; 6 of the 29 are LIVE | 18/29 | stamp on all 29 |

**Zone rebuild check:**
- The rebuild agrees with the stamp's sign on 96.9 / 97.5 / 95.8 % of fills (backtest eras), 92 % (master) and 93 % (fresh); correlation r ≥ 0.986.
- **Unscored fills: 0 in every block.**

**Kline features rebuilt for every fill:**
- coin: 1h gap change over 1h / 4h / 24h, price vs 1h EMA200 / EMA20, 1h slope, 4h EMA20/50 gap and slope, 4h / 24h return, distance below the 24h high and above the 24h low;
- BTC: 1h slope, 4h slope, 4h gap, 1h EMA20/200 gap and its 4h change, 24h return.

**Market-impulse features** (ML_WL_1015: BTC returns, breadth, dispersion, volumes):
- coverage is 100 % in the backtest and 97 % in master;
- **66 % in fresh**, because the post-10-14 live fills are not in that file. This is a blind spot for the fresh block.

## STEP 3a · Pre-registered theory pairs (inside the zone, BAD side minus GOOD side; full table: `reports/ML_DT2D_theory.txt`)

A Δ < 0 means the theory's bad side is worse.

| hypothesis (bad side) | 2025 Δ (block-null p) | 2026-H1 Δ | 2026-H2 Δ | master Δ | stack_keep Δ | fresh Δ | supported? |
|---|---|---|---|---|---|---|---|
| T1 deep, gap ≤ −3 % | −0.035 (0.26) | −0.124 | **+0.228** | −0.090 | −0.071 | +0.308 | no |
| T2 gap still widening (4h) | **+0.030** (0.75) | +0.104 | −0.171 | +0.054 | +0.095 | +0.017 | no (2 of 3 eras opposite) |
| T3 price ≤ 1h EMA200 | **+0.110** (0.89) | −0.041 | +0.125 | +0.025 | +0.117 | +0.650 | no (opposite) |
| T4 BTC 1h slope ≤ 0 | +0.002 (0.54) | +0.066 | −0.028 | +0.039 | +0.116 | +0.658 | no |
| T5 coin 4h EMA20 ≤ EMA50 | −0.164 (0.044) | −0.061 | **+0.303** | −0.048 | +0.037 | −0.181 | no (H2 flips; misses Bonferroni 0.0071) |
| T6 ADX falling | −0.062 (0.083) | **+0.051** | −0.208 | −0.122 | −0.173 | +0.217 | no |
| T7 low pair volume | −0.021 (0.31) | +0.016 | +0.053 | +0.037 | +0.084 | −0.219 | no |

None of the seven is direction-consistent across the three eras. The two theory stories cancel out: neither "bounce" nor "reversal" signatures separate the zone.

## STEP 3b · Exhaustive screen (DT × every variable; tables: `reports/ML_DT2D_tables.txt`, per-cell CSV `reports/ML_DT2D_screen.csv`)

**Setup:**
- 106 variables: every covered entry_* stamp, the derived d_* stamps, the k_* kline rebuilds and the m_* market features.
- 697 cells: label-blind median and tercile cuts, frozen on the 2025 zone; 646 are valid in all 3 eras.
- Two statistics: the inside-zone split, and zone ∧ X vs the rest.
- Day-clustered z; 200 day-block shuffles per era for the null.

| read | observed | null (day-block shuffles) | result |
|---|---|---|---|
| cells same-sign with \|z\| ≥ 1.96 in **all 3 eras**, inside-zone split | 0 | median 0 · 95th 0 | nothing |
| same, zone ∧ X vs rest | 0 | median 0 · 95th 0 | nothing |
| max consistency score \|C\| (min \|z\| over eras), inside split | 0.94 | 95th pct 1.71 | family-wise p ≥ 0.94 for every cell |
| max \|C\|, zone ∧ X vs rest | 1.16 | 95th pct 1.75 | best fw-p 0.57 |
| walk-forward: discover 2025 (\|z\| ≥ 1.96, \|Δ\| ≥ 0.05) → same sign in H1 and H2 | 31 discovered (luck ≈ 32) → **4 confirmed** | confirmed count median 7 · 95th 26 | p 0.73: below luck |
| keep side: a DT ∧ X cell above zero in all 3 eras | **0** | — | the zone cannot be "made to work" |

**What the 4 walk-forward cells are:** two complementary pairs of one variable, coin age (`entry_pair_age_days` > 546 d and > 861.5 d). All four also point the same way in master and fresh. That is the only theme to examine.

**The top cells by consistency** are all far from significance (C ≤ 0.94; zone ∧ X vs rest C ≤ 1.16). Examples:
- BTC RSI (closed) ≥ 63;
- coin 4h EMA20 slope below the median;
- BTC 72h return between −2.8 and +0.2 %;
- long-heat flag ≥ 1;
- coin RSI ≥ 61.7.

Several of them disagree with master or fresh.

## STEP 3c · The coin-age theme (DT ∧ OLD, age > 861.5 d = 2025-zone median); full read: `reports/ML_DT2D_deep.txt`

### Three blocks side by side: is it a 2D effect?

| block | DT ∧ OLD N·d·WR·avg [95 %] | DT ∧ YOUNG | Δ in zone | REST ∧ OLD | REST ∧ YOUNG | Δ outside zone | **interaction** |
|---|---|---|---|---|---|---|---|
| ③ 2025 | 765·167·49 %·−0.247 [−0.32,−0.18] | 765·57 %·−0.118 | −0.129 | 439·45 %·−0.264 | 648·58 %·−0.144 | −0.120 | **−0.009** |
| ③ 2026-H1 | 359·86·55 %·−0.094 | 218·61 %·−0.086 | −0.009 | 256·56 %·−0.109 | 338·65 %·−0.057 | −0.052 | **+0.044** |
| ③ 2026-H2 | 119·35·53 %·−0.179 | 51·45 %·−0.102 | −0.077 | 212·67 %·+0.026 | 139·73 %·+0.094 | −0.068 | **−0.009** |
| ② master as-traded | 31·23·42 %·−0.341 [−0.52,−0.16] | 15·67 %·+0.085 | −0.425 | 70·66 %·−0.024 | 35·60 %·+0.079 | −0.102 | −0.323 |
| ② master stack_keep | 21·17·48 %·−0.290 | 10·50 %·+0.040 | −0.330 | 48·69 %·+0.039 | 23·78 %·+0.385 | −0.346 | +0.017 |
| ① fresh line cohort | 13·9·31 %·−0.277 [−0.65,+0.27] | 5·40 %·+0.170 | −0.446 | 8·50 %·−0.266 | 3·67 %·+0.480 | −0.746 | +0.300 |
| ① of which LIVE | 4·2·50 %·**+0.165** | 0 | — | 2·100 %·+0.138 | 0 | — | — |

**Notes on the table:**
- Master is missing coin age on 40 older fills, which are left out of this table.
- The backtest interaction is about zero in every era.
- Old coins are worse **everywhere**, inside and outside the zone. The 1D sleeve-wide read (old minus young, all fills):

  | block | old minus young |
  |---|---|
  | 2025 | −0.12 |
  | 2026-H1 | −0.03 |
  | 2026-H2 | −0.09 |
  | master | −0.20 |
  | fresh | −0.56 |

- Only master as-traded shows a stronger in-zone effect, and its own stack_keep view erases that again.

### Stability checks

- **Per seed (in-zone Δ):**
  - 2025: −0.14 / −0.13 / −0.12;
  - 2026-H1: +0.02 / −0.06 / −0.01;
  - 2026-H2: +0.25 / −0.11 / −0.45 (sign unstable).
- **Leave-one-month-out (pooled backtest):** Δ stays −0.06 to −0.11 in all 22 folds. Month by month, however, it is only 15 of 22 negative, with +0.50 (2026-01) and +0.71 (2026-08).
- **Expectancy bar on master as-traded:** it would pass: N 31 · 23 d · WR 42 % vs breakeven 62.8 % · −0.341 · P(mean<0) 1.00 · top day 14 % / top pair 19 % of the loss.
  - But master is the *selection* pool's confirmation block, and the fresh block overlaps it (9 of the 13 fresh DT ∧ OLD fills are master B16–B21).
  - The fresh block fails ② (P 0.85) and N (13).
- **Overlap with the existing line:** 45–48 % of DT ∧ OLD fills are already inside the DL 273 "ADX falling ∧ age > 747 d" sub-tally.

### Before / after: refuse DT ∧ OLD (two-sided: refused winners count against it)

| block | before N·avg·Σ% | refused (W/L) | after N·avg·Σ% | Δ avg/fill | Σ$ recorded before → after | Σ$ at today's ~$700 1× notional |
|---|---|---|---|---|---|---|
| ③ 2025 | 2617·−0.187·−489 | 765 (376/389) | 1852·−0.162·−300 | +0.025 | — | −3,424 → −2,100 |
| ③ 2026-H1 | 1171·−0.085·−100 | 359 (198/161) | 812·−0.081·−66 | +0.004 | — | −699 → −462 |
| ③ 2026-H2 | 521·−0.015·−8 | 119 (63/56) | 402·+0.033·+13 | +0.049 | — | −55 → +94 |
| ② master as-traded | 191·+0.027·+5.2 | 31 (13/18) | 160·+0.099·+15.8 | +0.071 | +802 → +3,026 | +36 → +110 |
| ② master stack_keep | 139·+0.141·+19.6 | 21 (10/11) | 118·+0.217·+25.7 | +0.077 | +4,567 → +5,674 | +137 → +180 |
| ① fresh line cohort | 29·−0.118·−3.4 | 13 (4/9) | 16·+0.010·+0.2 | +0.128 | −852 → −104 | −24 → +1 |
| ① **current LIVE batch** | 6·+0.156·+0.9 | 4 (2/2) | 2·+0.138·+0.3 | **−0.018** | +4 → +1 | +7 → +2 |

**How to read this table:**
- The backtest Δ is the age effect itself. The same refusal applied *outside* the zone saves about as much.
- The backtest Δ is mostly the 2025 year that loses everywhere.
- The live batch would have lost its two best trades (PEPE +1.16, DOT +0.96).
- Master per batch (as-traded Σ% before → after):

  | batch | Σ% before → after |
  |---|---|
  | B1 | +3.51 → +4.94 |
  | B2 | −2.15 → −1.91 |
  | B3 | +3.28 → +5.22 |
  | B6 | −1.70 → +0.10 |
  | B7 | −1.09 → −1.19 |
  | B8 | −0.26 → +0.45 |
  | B14 | −4.28 → −3.99 |
  | B16 | −0.92 → −1.22 |
  | B17 | +3.93 → +4.62 |
  | B18 | −2.09 → −0.69 |
  | B19 | −1.76 → −1.42 |
  | B21 | −2.14 → 0.00 |

  BASE, B4, B5, B9, B10, B12, B13, B15 and B20 are untouched. The per-trade list of all 31 master and 13 fresh refused fills is in `ML_DT2D_deep.txt`.

## Verdict

- **NOTHING.** No DOWNTREND × X cell holds in the backtest across eras, beats its shuffled null, and points the same way in master and the fresh cohort.
- The only cross-block-consistent axis is **coin age**, and it is a **1D sleeve-wide effect, not a downtrend interaction**. It already lives on the existing established-coin sub-tally (DECISION_LOG 273 / ML_WINNERS_LOSERS_2026-10-15 §6), and its family-wise evidence is at luck level.
- **No new line, nothing to arm.** PAIR_1H_DOWNTREND keeps its frozen 1D gate, FILTER CANDIDATE at N ≥ 15 / ≥ 8 days on the expectancy bar (it is at 18 · 10 d · P(mean<0) 0.74 → KEEP OBSERVING, one re-read at 30).
- **Optional follow-up:** if the operator wants the age axis pursued, it should be pre-registered as its own **1D** coin-age question on the momentum sleeve. It should not be another downtrend sub-cut, and it should be judged on fresh fills only (cross-period = refute-only).

## Blind spots (what this screen could NOT test)

- **Constant in the backtest** (one value): pattern flags c1, c3, c4, c7, c9, w1, w3, w5, w6.
- **Below 90 % coverage in a backtest era:**
  - `entry_chop_burst_prior_fill_s` (23–26 %);
  - `m_pair_r15`, `m_pair_r60`, `m_rs15`, `m_rs60` (89 % in 2026-H1).

  Read separately inside the zone, none is direction-consistent: for example rs15 hi−lo is +0.07 / −0.08 / +0.07 across the eras, and chop-burst is −0.21 / −0.12 / +0.09.
- **Real-block stamp gaps:**
  - master lacks many newer stamps on older batches (coin age is missing on 40 fills; BTC RSI-closed and heat flags exist only on recent batches);
  - fresh market features cover 66 %.

  So real-block reads of those cells are on subsets (N shown per cell in the CSV).
- **Market-wide X:** clustered by day in the z statistic. No macro cell reached significance, so no window-unit adjudication was needed.

## Files

- `reports/ML_DOWNTREND_2D_PREREG.txt` — pre-registration.
- `scripts/study_ml_dt2d_build.py` — features (k_* rebuilds, zone scoring, fresh loader) → `reports/ML_DT2D_features.pkl`.
- `scripts/study_ml_dt2d_theory.py` → `reports/ML_DT2D_theory.txt`.
- `scripts/study_ml_dt2d_analyse.py` → `reports/ML_DT2D_tables.txt`, `reports/ML_DT2D_screen.csv`.
- `scripts/study_ml_dt2d_deep.py` → `reports/ML_DT2D_deep.txt` (z-formula check: screen z −2.79 vs bootstrap −2.80).
- `reports/ml_dt2d_fetch/5m/` — the 2 fetched klines pages.
