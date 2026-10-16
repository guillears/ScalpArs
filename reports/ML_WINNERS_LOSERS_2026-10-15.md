# Momentum LONG: winners vs losers (2026-10-15)

- **Pre-registration:** `reports/ML_WL_2026-10-15_PREREG.txt`, written before any result was computed.
- **Builds on:** `reports/YR6_WL_MOMENTUM.md` (2026-10-13). This study does not repeat it. It adds four things:
  1. an exhaustive all-pairs 2D screen;
  2. 18 new market-impulse features rebuilt from klines (BTC, top-50 breadth, volume surge, relative strength);
  3. a speed / early-in-trade analysis;
  4. the master fills as a third discovery block.
- **Safety:** nothing was shipped, committed or edited in the engine, config or templates.
- **Data:** everything comes from the local cache, except one small Binance top-up for the single live fill (51 requests, used weight 34).
- **Validation:** `scripts/validate_against_master.py` reports ALL CHECKS PASS. The study's own checks (`study_ml_wl_1015_validate.py`) also ALL PASS:
  - clock shift = 0 is the tightest alignment;
  - the minute-k prices lie inside each trade's recorded peak / trough;
  - deleting every bar after entry leaves the entry features unchanged (no lookahead).

---

## Plain-English summary

1. **Winners are not faster to close. They are faster to go green.**
   - Winners and losers stay in the trade about the same time (median 17–22 min in every block).
   - The difference is direction in the first minutes:
     - At minute 3, **46–56 % of winners are already in profit**, against only 13–26 % of losers.
     - Losers peak early (median ~4 min, and about half never get above +0.05 %), then bleed to their low at ~17–19 min.
     - Winners keep climbing to a late peak (~13–17 min).
   - So a loser is mostly an entry where the move simply did not continue.
2. **Why winners go green: the market moves after the entry, not before it.**
   - The single biggest difference between winners and losers is what BTC and the top-50 coins do in the first 15 minutes after we enter. This is the same in all five blocks:

     | BTC over the trade's first 15 min | WR across blocks |
     |---|---|
     | up ≥ 0.10 % | 77–88 % |
     | flat | 54–77 % |
     | down ≥ 0.10 % | 32–48 % |

   - Breadth over the first 15 minutes shows the same thing: winners +0.05 to +0.10 %, losers −0.12 to −0.21 %.
   - **The recent losing streak is this exact story.** Master batches B13–B21 (39 fills, 33 % WR, −0.32 %/fill):
     - BTC fell after entry 31 % of the time, against 22 % before.
     - When it fell, 1 of 12 trades won.
     - At entry, the market looked the same as before.
3. **But the market state at the moment of entry does NOT predict what the market does next.**
   - We rebuilt BTC's return over the prior 5/15/30/60 min, BTC volume, breadth (share of top-50 coins rising) and its change, the cross-coin spread, the pair's strength vs BTC, and market and pair volume surges.
   - None of them separates winners from losers in both bull years:
     - **"BTC already rising into the entry" helps in 2026** (master +0.32, H2 +0.20 per trade) but **hurts in 2025** (−0.06, the opposite sign at P 0.94).
     - Across all tested patterns, those found in one year flip sign in the other year more often than they hold (2025 → 2026-H2: 116 held vs 225 flipped).
   - That is the signature of a regime effect, not a stable entry rule.
4. **The dashboard columns ("Entry Conditions by Strategy — Winners vs Losers"):** every one was tested.
   - The only consistent direction: **winners enter with slightly more pair momentum**. That means a wider EMA fan, more stretch above EMA13 and EMA5, a steeper EMA20 and a higher ATR. It has the same sign in all four blocks.
   - The effect is tiny in 2025 (0.05–0.12 standard deviations). It is not a usable cut.
5. **Can we exit early when a trade starts badly?** We tested exiting at minute 1, 3 or 5 when the trade is red, and also when BTC or breadth turned down.
   - **No.** It only helps in 2025, the year that loses everywhere.
   - In 2026-H2 and on the real master fills it costs money: −0.02 to −0.16 %/fill.
   - Reason: about half of the trades that are red at minute 3–5 are eventual winners that recover. Cutting them costs as much as it saves.
   - **No early-exit rule passes.**
6. **The exhaustive screen:**
   - **Setup:** 95 features, 17,860 two-variable zones per year, and a 200-shuffle null.
   - **Family-wise result:** zero survivors in any block.
   - **The count of "significant" zones** (1,809 in 2025) sits inside what pure luck produces (shuffled 95th percentile: 1,645 to 1,657).
   - **Zones confirmed across both bull years and on master:** 21 (23 rows). 18 rows are the already-registered **ADX-falling** theme (DECISION_LOG 273).
   - **One refinement stands out: ADX falling on an *established* coin (listed > ~2 years).**
     - The zone loses −0.27 % (2025), −0.16 % (H2) and −0.23 % (master as-traded).
     - It **fails the expectancy bar on the current-stack master view** (−0.18 %, CI just crosses 0, N 35).
     - It is a post-hoc pick from 17k tests (family-wise p ≈ 1).
7. **Bottom line, said plainly:**
   - There is **no entry filter** that separates momentum-long winners from losers out of sample.
   - The edge, or the lack of it, lives in **what the market does in the 15 minutes after entry**, and that is not knowable at entry.
   - The losses are uniform: no pair carries more than 9 % of the loss, and no day more than 7 %. That points to a **regime / sleeve on-off question** (the CURRENT_STATE 30-fill gate at 1×), not an entry filter.
   - The one weak candidate (ADX stalling on an established coin) goes on the existing ADX observe line as a frozen sub-tally. It is **not** something to arm.

---

## 1. Blocks (MOMENTUM LONG, non-probe, closed, % at 1×, day-bootstrap 95 % CI)

| block | N | days | WR | avg % | 95 % CI | breakeven WR | avg win / loss |
|---|---|---|---|---|---|---|---|
| ① live (since go-live 10-14 18:36) | 1 | 1 | 100 % | +1.157 | — | — | 1000PEPE HARD_TP_LADDER L2, 13 min |
| ② master as-traded | 191 | 77 | 64.9 % | +0.027 | [−0.087, +0.151] | 62.8 % | +0.48 / −0.80 |
| ② master stack_keep (in-sample to today's filters) | 139 | 66 | 71.2 % | +0.141 | [−0.005, +0.293] | 60.8 % | +0.53 / −0.82 |
| ③ replay 2025 BULL | 2617 | 301 | 53.1 % | −0.187 | [−0.229, −0.145] | 68.7 % | +0.37 / −0.82 |
| ③ replay 2026-H1 BEAR | 1171 | 154 | 59.4 % | −0.085 | [−0.153, −0.019] | 66.2 % | +0.42 / −0.83 |
| ③ replay 2026-H2 BULL | 521 | 79 | 63.1 % | −0.015 | [−0.119, +0.101] | 64.4 % | +0.44 / −0.79 |

- **Computed two ways** (pandas, and the raw `csv` module on the source files): N, WR and avg are identical for master and all three regimes.
- **Master per batch** (as-traded avg):
  - BASE +0.41, B1 +0.14, B2 −0.09, B3 +0.12, B12 +0.07, B17 +0.65;
  - **B13–B21: 39 fills · 33 % · −0.315** (B14 −0.54, B18 −0.70, B19 −0.44, B20 −0.70, B21 −0.71).
  - Full per-era table: `reports/ML_WL_1015_tables.txt` §S0.
- **Live fill (PEPE):** it entered into a strong tape (BTC +1.24 % over the prior 60 min; 96 % of the top-50 up over 30 min), and BTC kept rising (+0.46 % over the trade's first 15 min).
  - This is a textbook "market tailwind" winner. N = 1, so it is descriptive only.
  - Its pair volume surge was 1.04×. That is the zone §5 found slightly worse, and the trade won anyway.

## 2. Speed: are winners faster? (medians, minutes)

| block | side | N | time in trade | time to peak | time to trough | peak % | P&L at 1 / 3 / 5 / 15 min | in profit at 3 min | stop / runner / ladder |
|---|---|---|---|---|---|---|---|---|---|
| master | W | 124 | 22.1 | 17.4 | 3.1 | +0.72 | +0.00 / +0.04 / +0.04 / +0.14 | 52 % | 0 / 85 / 9 % |
| master | L | 67 | 16.7 | 3.7 | 16.8 | +0.04 | −0.04 / −0.13 / −0.16 / −0.43 | 22 % | 97 / 0 / 0 % |
| master_keep | W | 99 | 21.6 | 17.4 | 3.1 | +0.82 | +0.00 / +0.04 / +0.04 / +0.14 | 54 % | 0 / 82 / 10 % |
| master_keep | L | 40 | 15.3 | 3.7 | 15.1 | +0.03 | −0.07 / −0.16 / −0.17 / −0.44 | 13 % | 95 / 0 / 0 % |
| 2025 | W | 1389 | 18.9 | 16.5 | 3.3 | +0.66 | −0.01 / +0.04 / +0.06 / +0.18 | 56 % | 0 / 88 / 11 % |
| 2025 | L | 1228 | 19.5 | 4.6 | 18.8 | +0.06 | −0.05 / −0.12 / −0.17 / −0.39 | 25 % | 91 / 1 / 0 % |
| 2026-H1 | W | 695 | 20.4 | 17.7 | 2.9 | +0.68 | −0.01 / +0.01 / +0.03 / +0.16 | 46 % | 0 / 84 / 16 % |
| 2026-H1 | L | 476 | 21.5 | 6.0 | 19.7 | +0.07 | −0.03 / −0.11 / −0.16 / −0.37 | 23 % | 89 / 2 / 0 % |
| 2026-H2 | W | 329 | 17.2 | 13.4 | 2.6 | +0.72 | −0.01 / +0.03 / +0.05 / +0.21 | 54 % | 0 / 85 / 15 % |
| 2026-H2 | L | 192 | 20.4 | 3.7 | 18.8 | +0.07 | −0.03 / −0.11 / −0.16 / −0.32 | 26 % | 90 / 0 / 0 % |

**Losers**, across the blocks:
- 45–60 % never exceed +0.05 %;
- only 12–26 % ever reach +0.20 %;
- their peak comes before their trough in 57–67 %.

Time in trade is not a separator either:

| block | ≤ 5 min | 5–15 min | 15–30 min | 30–60 min |
|---|---|---|---|---|
| 2025 | −0.19 | −0.26 | −0.17 | −0.11 |
| 2026-H2 | +0.03 | −0.10 | −0.02 | +0.05 |

## 3. Early in-trade state (decidable at minute k = the last closed 1m bar ≤ entry + k; trades still open)

The early state separates WR strongly in every block. These are fills still open at minute 5:

| block | P&L at 5 min ≤ 0: N · WR · avg | P&L at 5 min > 0: N · WR · avg | BTC entry→5 min ≤ 0 vs > 0 (avg) | breadth first 5 min ↓ vs ↑ (WR) |
|---|---|---|---|---|
| master | 95 · 54 % · −0.10 | 70 · 86 % · +0.30 | −0.03 vs +0.17 | 56 % vs 75 % |
| master_keep | 65 · 60 % · +0.05 | 52 · 94 % · +0.41 | +0.14 vs +0.28 | 66 % vs 81 % |
| 2025 | 1356 · 37 % · −0.36 | 968 · 73 % · +0.07 | −0.30 vs −0.06 | 43 % vs 60 % |
| 2026-H1 | 565 · 47 % · −0.25 | 378 · 76 % · +0.19 | −0.13 vs +0.01 | 54 % vs 64 % |
| 2026-H2 | 243 · 51 % · −0.13 | 173 · 78 % · +0.17 | −0.08 vs +0.09 | 57 % vs 66 % |

**Early-exit counterfactual** (pre-registered grid: exit at minute k if pair P&L_k ≤ θ; CF = 1m close − the trade's own fee − 0.02 slip).

- The table is P(Δ > 0), day bootstrap.
- **Gate:** ≥ 0.90 in 2025 **and** 2026-H2 **and** master.

| k / θ | 2025 | 2026-H1 | 2026-H2 | master | master_keep |
|---|---|---|---|---|---|
| 1 / −0.30 | 1.00 | 0.57 | 0.21 | 0.00 | 0.00 |
| 1 / 0.00 | 0.98 | 0.02 | 0.00 | 0.17 | 0.03 |
| 3 / −0.20 | 0.98 | 0.27 | 0.07 | 0.39 | 0.29 |
| 3 / 0.00 | 0.96 | 0.09 | 0.03 | 0.03 | 0.00 |
| 5 / −0.30 | 0.81 | 0.22 | **0.00** | 1.00 | 0.95 |
| 5 / 0.00 | 0.92 | 0.04 | 0.01 | 0.01 | 0.00 |

(The full 12-rule grid is in `ML_WL_1015_early_exit.csv`.)

- **No rule passes.** The best master rule (5 / −0.30: +0.017 %/fill) loses in H2 (−0.033, P 0.00).
- **Two-sided read:** for example, at 3 / 0.00 in H2, the cut losers save +58.8 % in total, but the cut winners give back −87.4 %. This is the "saved vs cut" split the stop-CF rule asks for.
- **Exploratory, not pre-registered:** exiting at minute 3 or 5 when BTC (or breadth) has turned down since entry gives the same picture. Δ is +0.02 to +0.04 in 2025 and −0.04 to −0.12 in H1, H2 and master (`ML_WL_1015_early_market.txt`).
- These fills assume a 1m close plus 0.02 slip. Live slippage (≥ 0.10 on fast tapes) would make every early-exit result worse, so the conclusion is robust.

## 4. Coverage first, then winners vs losers on every stamped column

**Coverage** (`ML_WL_1015_coverage.csv`): 98 feature columns in total.
- **Covered everywhere:** 77 are scored on ≥ 80 % of master, 2025 and H2 fills.
- **Low on master:**
  - BTC 72 h / 24 h / 30 d stamps: 30–38 %;
  - gap_5_20 / btc_1d_ret / eth / pair 1h EMA200: 13–17 %;
  - cmc rank / mcap: 17 %; vol/mcap: 5 %;
  - pair age: 79 %.
- **Never in the replay:** `funding_rate`.
- **`chop_burst_prior_fill_s`:** 2–26 %.
- **How low coverage is handled:** these columns are read on scored rows only and are screened only in blocks where coverage is ≥ 80 %.

**Dashboard table, every column**, as a standardized W − L gap (master / 2025 / H1 / H2). Full medians are in `ML_WL_1015_medians.csv`.

| column | master | 2025 | H1 | H2 | consistent? |
|---|---|---|---|---|---|
| RSI | +0.25 | −0.05 | +0.05 | −0.21 | no |
| ADX | −0.09 | −0.00 | −0.08 | −0.17 | same sign, ~0 |
| ADX dir / ADXΔ | +0.03 | −0.01 | +0.11 | +0.22 | no (tercile split yes, see §5) |
| RSI dir | +0.22 | +0.03 | +0.06 | −0.07 | no |
| Gap (EMA5–20) | +0.22 | +0.06 | +0.13 | +0.27 | **yes**, small |
| Gap5-8 / Gap8-13 | +0.16 / +0.19 | +0.02 / +0.05 | +0.11 / +0.11 | +0.10 / +0.16 | **yes**, small |
| Fan ratio | +0.11 | −0.06 | −0.01 | −0.07 | no |
| Stretch (EMA5) | +0.27 | +0.12 | +0.14 | +0.10 | **yes** |
| \|EMA20 slope\| | +0.25 | +0.03 | +0.15 | +0.33 | **yes**, small in 2025 |
| \|BTC EMA20 slope\| | −0.08 | −0.01 | −0.03 | −0.09 | same sign, ~0 |
| BTC RSI | −0.14 | −0.11 | −0.09 | +0.09 | no |
| BTC ADX | −0.02 | +0.02 | −0.03 | −0.05 | no |
| BTC ADX dir / RSI dir / RSI 30m dir | −0.35 / +0.38 / −0.06 | −0.09 / +0.10 / +0.00 | −0.04 / +0.02 / +0.03 | +0.22 / −0.02 / +0.16 | no |
| Range position | +0.20 | +0.01 | +0.02 | −0.25 | no |
| Breadth (bull %) | +0.01 | +0.06 | −0.09 | −0.03 | no |
| ATR % | +0.20 | +0.11 | +0.06 | +0.35 | **yes** |
| BTC ATR % | +0.35 | +0.09 | +0.10 | −0.17 | no |
| BTC trend gap | −0.05 | +0.04 | +0.02 | +0.00 | no |
| Global vol ratio | −0.08 | −0.08 | +0.10 | +0.10 | no |
| Pair vol $ 24 h | −0.13 | +0.09 | −0.09 | +0.06 | no |
| Ext % (vs EMA13) | +0.30 | +0.11 | +0.16 | +0.11 | **yes** |
| BTC ext % | +0.13 | +0.02 | −0.03 | −0.06 | no |
| BTC 1 h slope | −0.03 | +0.02 | +0.06 | +0.20 | no |

- **Same sign in all four blocks:** 26 of 91 columns, against a coin-flip expectation of 11. The excess is one correlated family: pair momentum strength (gap, stretch, extension, EMA20 slope, ATR).
- **Standardized gaps** are 0.05–0.12 in 2025 and 0.10–0.35 in 2026. That is a "slightly stronger impulse" signature, not a cut (see §5: no single split confirms both ways).
- AvgPeak%, AvgTrough%, WorstTrough%, SLProf and Duration are outcome columns. They are covered in §2.

## 5. Exhaustive screen: 1D + ALL-PAIRS 2D (median quadrants), circular-shift null ×200, discover → confirm

| discovery | N · days | cols | 1D valid · p<.05 (luck) · family-wise | 2D valid · p<.05 · shuffled-null 95th pct of that count · family-wise | null 95th max\|Δ\| vs observed |
|---|---|---|---|---|---|
| 2025 BULL | 2617 · 301 | 95 | 282 · 26 (14) · **0** | 16,423 · 1,809 · 1,645 · **0** | 0.251 vs 0.259 |
| 2026-H2 BULL | 521 · 79 | 94 | 285 · 21 (14) · **0** | 16,451 · 1,668 · 1,657 · **0** | 0.640 vs 0.575 |
| master | 191 · 77 | 76 | 226 · 7 (11) · **0** | 10,575 · 611 · 903 · **0** | 0.783 vs 0.536 |
| 2026-H1 BEAR | 1171 · 154 | 93 | 282 · 4 (14) · **0** | 15,940 · 586 · 1,419 · **0** | 0.385 vs 0.363 |

**Out-of-sample confirmation of the discovery survivors** (p < .05 and |Δ| ≥ 0.05; day-bootstrap P ≥ 0.90).
- The honest luck baseline is the **opposite-sign** confirmation count. If discovery carried no information, same and opposite would be about equal.

| discovery → confirm | survivors | confirmed SAME sign | confirmed OPPOSITE sign | read |
|---|---|---|---|---|
| 2025 → 2026-H2 | 1835 | 116 | **225** | patterns **reverse** across years |
| 2026-H2 → 2025 | 1689 | 155 | **326** | patterns **reverse** across years |
| master → 2025 | 618 | 101 | 80 | ≈ chance |
| 2025 → master | 1835 | 256 | 137 | weak |
| 2026-H2 → master / master → 2026-H2 | 1689 / 618 | 462 / 220 | 18 / 1 | same calendar months: the replay mirrors live, which is not OOS |
| 2026-H1 → 2025 / → 2026-H2 | 590 | 119 / 52 | 37 / 71 | mixed |

**Zones confirmed in both bull directions and on master** (separator definition, PREREG): **21 distinct quadrants (23 rows).**
- 18 rows are the already-registered ADXΔ line (ADX falling, ≤ ~0.25) crossed with something else. That something is BTC volume, the HEALTHY_BULL regime, pair age, pos_DI, BTC EMA20 slope or BTC ADX. `entry_adx_delta` and the derived "ADX dir" are the same number.
- **The rest:**
  - pair volume surge (> 0.68×) × afternoon (hour > 13) or × non-STRONG_BULL regime;
  - "breadth > 0.71 ∧ quiet pair volume" (a keep zone);
  - STRONG_BULL regime ∧ breadth60 ≤ 0.79;
  - "BTC up > 0.11 % in 15 min ∧ BTC RSI not jumping" (a keep zone).
- **Expectancy bar on the blocked side** (`ML_WL_1015_bar.txt` part 2):
  - every zone passes in 2025, because the whole sleeve is below breakeven there;
  - every zone **fails on master_keep**, the current-stack view.
  - Only one passes 2025, H2 **and** master as-traded:

| ADXΔ ≤ 0.27 ∧ pair age > 747 d (2025-frozen) | N | days | WR | breakeven WR | zone avg | CI hi | kept avg | bar |
|---|---|---|---|---|---|---|---|---|
| master as-traded | 51 | 31 | 52.9 % | 64.0 % | −0.226 | −0.056 | +0.033 | PASS |
| master stack_keep | 35 | 24 | 57.1 % | 61.0 % | −0.179 | **+0.024** | +0.169 | **fail (confidence only)** |
| 2025 BULL | 636 | 185 | 47.5 % | 68.7 % | −0.269 | −0.187 | −0.160 | PASS |
| 2026-H2 BULL | 185 | 50 | 54.1 % | 64.4 % | −0.159 | −0.026 | +0.064 | PASS |
| 2026-H1 BEAR | 319 | — | — | — | Δ −0.006 | — | — | no effect |

- **Concentration:** the top pair is 8–16 % of the zone loss.
- **Caveats:**
  - Family-wise p for this exact quadrant is 1.0. It is one cell of 17,860.
  - It is also found from H2 with different cuts (ADXΔ ≤ 0.23, age > 981 d).
  - Pair age is scored on only 79 % of master fills.
  - It does **not** make 2025 positive (kept −0.160).

**New market-impulse features (1D, pooled-replay terciles, `ML_WL_1015_impulse.csv`):**
- **None** has the same sign in all four blocks with P ≥ 0.90 in both bulls.
- **Closest misses:**
  - **low market volume surge** (agg_vol15 ≤ 0.68×): Δ +0.15 master / +0.05 2025 (P 0.86) / +0.18 H2 (P 0.93);
  - **high pair volume surge** (> 0.76×, discovered H2, confirmed 2025 P 0.96): zone +0.018 on master and +0.152 on master_keep, so it **fails on real fills**;
  - **BTC down 15 min before entry:** −0.20 H2, −0.14 master, but **+0.04 better** in 2025.

## 6. Regime / uniformity (sleeve-kill checklist items ③④, read here only as context)

- **Loss concentration:**

  | block | top pair | top-2 pairs | worst day | losing days | losing pairs |
  |---|---|---|---|---|---|
  | 2025 | 4 % | 7 % | 2 % | 67 % | 68 % |
  | H2 | 9 % | 16 % | 7 % | 53 % | 44 % |
  | master | 9 % | 15 % | 5 % | 44 % | 43 % |

  The losses are **uniform**, not concentrated.
- **Market path after entry, the descriptive driver:** share of fills where BTC fell ≥ 0.10 % in the first 15 min.

  | block | BTC fell ≥ 0.10 % in first 15 min |
  |---|---|
  | 2025 | 31 % |
  | H1 | 34 % |
  | H2 | 21 % |
  | master | 24 % |
  | master B13–B21 | **31 %** |

  Even within the "BTC up" bucket, 2025 earns less (+0.13 vs +0.32 in H2). So the losing periods combine a more adverse post-entry path with a worse conditional payoff. Both are regime traits.
- **Conclusion:** per CLAUDE.md, uniform degradation means a regime cause, not an entry filter. The right lever is the sleeve on/off decision already pre-committed in CURRENT_STATE: after 30 fills at 1×, momentum longs go OFF if avg < 0 at day-bootstrap 90 % and no ML observe line has reached its bar.

## 7. Recommendation

1. **No filter, no exit change, no config change.**
   - No entry separator holds out of sample. No early-exit rule passes.
   - Cross-year patterns reverse more often than they hold.
2. **Optional observe-only sub-tally (not armed, no code)** under the existing ML_ADX_RISING_IN_REGIME line (DL 273): **ML_ADX_STALL_OLDCOIN**.
   - **Zone definition (frozen, never re-fit):** MOMENTUM LONG with `entry_adx_delta` ≤ 0.27 **and** `entry_pair_age_days` > 747.
   - **Scope:** fills with either stamp missing are excluded, never counted as "rest".
   - **Projected effect:** Δ(zone − rest) is −0.11 (2025) / −0.22 (H2) / −0.26 (master) / −0.35 (kept). After the 30–50 % in-sample haircut that is about −0.05 to −0.18 %/fill.
   - **Frozen bar** (fresh fills opened ≥ 2026-10-15 15:00 UTC, 1×, no probes, day units). At ≥ 15 zone fills on ≥ 8 distinct days it becomes an ARM candidate only if **all** of these hold:
     - zone WR < the ML breakeven WR then current;
     - day-bootstrap P(mean < 0) ≥ 0.95;
     - no day or pair ≥ 50 % of the zone loss;
     - zone mean < kept mean.
   - **RETIRE** if zone mean ≥ kept mean at N 15.
   - **If ever armed, revert** at ≥ 10 refused signals on ≥ 5 days re-priced with WR ≥ 61 % or Σ > 0.
   - **It overlaps the registered ADXΔ ≤ 0 line.** Read both at batch reviews and arm at most one.
3. **For the next batch review:** add the §2 / §3 columns (P&L at 3 / 5 min, BTC and breadth over the first 15 min) to the momentum-long read.
   - They explain each loss ("the market reversed after we entered") but cannot block it.
   - They are the right evidence for the sleeve on/off gate, not for an entry filter.

## 8. Could NOT be tested

- **Order book, open interest, liquidations, funding at entry:**
  - not in the cache;
  - `entry_funding_rate` is never stamped in the replay;
  - `manual_ob_*` exists only on manual fills.
- **Second- or tick-level early state:** the minute-k state uses 1m closes; breadth uses 5m bars, so the early breadth read is only at k = 5.
- **Pair 1m coverage:** 89–97 % of fills (2025 1m cache is per pair-month). Pair-return and early-state features are NaN elsewhere and read on scored rows only.
- **Top-50 universe:** the cached list ends 2026-10-11. Later fills (5 master, 1 live) use that list, lagged up to 4 days. Breadth is equal-weight, not market-cap weighted.
- **Master macro stamps:** 13–38 % coverage (eras ≥ B9 / ≥ B15). The rebuilt kline features replace them here with 97 % coverage.
- **Blocked signals:** signals blocked by today's filters are absent from the replay, so the zones cannot be read on signals the stack refuses.
- **Live block:** N = 1, descriptive only.
- **Early-exit execution:** latency and queue are not modelled (0.02 slip at the 1m close). The live ruler (≥ 0.10) only worsens the result.

## 9. Exact commands (cache-only, about 3 minutes; the top-up is the only network step)

```
venv/bin/python -W ignore scripts/study_ml_wl_1015_fetch_live.py          # live-window top-up → reports/ml_wl_1015_fetch/ (51 req, weight 34)
venv/bin/python -W ignore scripts/study_ml_wl_1015_features.py            # → reports/ML_WL_1015_features.csv
venv/bin/python -W ignore scripts/study_ml_wl_1015_analyse.py   > reports/ML_WL_1015_tables.txt
venv/bin/python -W ignore scripts/study_ml_wl_1015_early_market.py > reports/ML_WL_1015_early_market.txt   # exploratory
venv/bin/python -W ignore scripts/study_ml_wl_1015_bar.py       > reports/ML_WL_1015_bar.txt
venv/bin/python -W ignore scripts/study_ml_wl_1015_validate.py           # V1 clock · V2 path · V3 no-lookahead → ALL PASS
venv/bin/python scripts/validate_against_master.py                       # ALL CHECKS PASS
```

**CSVs:**
- `ML_WL_1015_features.csv`
- `ML_WL_1015_coverage.csv`
- `ML_WL_1015_medians.csv`
- `ML_WL_1015_impulse.csv`
- `ML_WL_1015_early_exit.csv`
- `ML_WL_1015_early_market_exit.csv`
- `ML_WL_1015_screen_{2025_BULL,2026-H2_BULL,master,2026-H1_BEAR}_{alltests,survivors}.csv`
- `ML_WL_1015_bar.csv`
- `ML_WL_1015_bar_2d.csv`

**Shared helpers:** `scripts/study_ml_wl_1015_common.py`.
