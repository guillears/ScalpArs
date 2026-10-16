# ML ADX-falling × second variable (2D) — 2026-10-16

Research only. No engine, config, template or test change, no commit.
Pre-registration: `reports/ML_ADXFALL_2D_PREREG.txt` (theory, 12 primaries, screen design and decision rule; sha256 hashes frozen 16:23 UTC before the analysis ran).
Scripts: `scripts/study_ml_adxfall_build.py` (features) and `scripts/study_ml_adxfall_analyse.py --zone stamp|closed`.
Full tables: `reports/ML_ADXFALL_tables_{stamp,closed}.txt`. Per-cell CSV: `reports/ML_ADXFALL_screen_{stamp,closed}.csv`.

## Plain English answer

**No.** No second variable makes ADX-falling momentum longs clearly and reliably losing in backtest AND master AND fresh.

- **Your 12 theory picks (P1–P12) all fail.** These were the exhaustion ideas: RSI high or rolling over, price stretched above EMA5/EMA13, EMA fan closing, high ADX, DI spread shrinking, volume fading, coin 1h/4h downtrend, BTC 15-minute dip, and an old move.
  - Not one is worse in all three eras.
  - The strongest is "RSI high", with a Stouffer z of −1.5 against a bar of −2.64.
  - Master points the opposite way: ADX falling with RSI high was +0.19 %/fill on 22 fills.
  - Several of the "exhaustion" signs come out backwards. For example, "stretched above EMA5" is *better* in all three eras.
- **The exhaustive screen finds nothing beyond luck.** It covered 1,077 valid cells: 140 variables, each cut on frozen 2025 quantiles.
  - 0 cells are same-sign at |z| ≥ 1.96 in all three eras.
  - The best family-wise p is 0.99 for the inside-zone split and 0.08 for zone∧X versus the rest.
  - Walk-forward confirmation counts sit inside the shuffled null in both directions.
  - The closed-bar ADX zone (DL 273's version) gives the same answer: best family-wise p is 0.93 inside the zone and 0.36 versus the rest.
- **One labelled lead, which does not pass.** It is "ADX falling ∧ the coin's volume ratio above the median (> 0.695)". The finding is: *heavy volume while ADX fades* is worse, not light volume. That is the **opposite** of my volume prediction (P8).
  - It points the same way in every block: −0.05 / −0.10 / −0.22 by era, master −0.54, stack_keep −0.49, fresh −0.78.
  - In ADX-rising fills, high volume is neutral or good, so this is a genuine interaction, not a general volume effect.
  - The closed-bar volume rebuild agrees (c_vol_r20).
  - But it was picked from 1,077 cells. Its 2025 z is only −0.7, it has no family-wise significance (fw 1.00 inside the zone, 0.20 versus the rest), and fresh has only 2 fills in the cell.
  - Under the frozen decision rule (§4 of the prereg) it is **descriptive only**. At most it could become an observe line with a frozen bar, which is the operator's choice. It is not a filter.
- **ADX falling on its own is still the cleaner statement.**
  - Backtest selectivity (kept − refused) is +0.07 / +0.07 / +0.12.
  - Master +0.06 is not significant, and stack_keep −0.02 is the opposite sign.
  - The 7 fresh fills are positive (+0.11 versus −0.19 for ADX rising); 2 of them are the only truly post-master LIVE ADX-falling fills, and both won.
  - The existing DL 273 observe line and its frozen bar remain the right home. Nothing here sharpens it.

## Step 0 · Theory (written before the tests; prereg §0)

ADX falling means trend strength is decaying at the moment the momentum long buys strength. That makes it a late entry into an exhausting move. It should hurt most when the move is also:
- extended in price (RSI high, stretched above EMA5/EMA13, an old run above EMA20);
- losing force on other gauges (RSI rolling over, EMA5–EMA20 gap narrowing, DI spread shrinking, volume fading);
- fighting the higher timeframe or BTC (coin 1h/4h trend down, BTC 15m down).

It should **not** depend on hour, coin size, pattern flags, or BTC's 30-day position. The direction of P6 (high versus low ADX) was flagged as ambiguous in advance.

**What the data said about the theory.** The exhaustion story is not what separates ADX-falling winners from losers:
- Stretch and RSI signs flip between eras.
- Where a consistent tilt shows up, it is the reverse of "fading volume". It is heavy pair volume (and a weak 4h trend, n_adx4h ≤ median) while 5m ADX falls, which fits "selling into the pop / absorption" better than "move running out of fuel".

## Cohort and coverage (parity with the downtrend-2D frame)

| block | fills | ADX falling (stamp) | ADX falling (closed bar) | stamp vs closed agree |
|---|---|---|---|---|
| backtest 2025 BULL | 2,617 | 947 (36 %) | 1,050 | 92 % |
| backtest 2026-H1 BEAR | 1,171 | 435 (37 %) | 480 | 92 % |
| backtest 2026-H2 BULL | 521 | 199 (38 %) | 216 | 91 % |
| master (as-traded 1×) | 191 | 77 (40 %) | 71 | 95 % |
| master stack_keep | 139 | 53 | 49 | — |
| fresh (≥ 09-30, ex-washed, paper + LIVE) | 29 (6 LIVE) | 7 (2 LIVE) | 7 | 93 % |

Notes on the cohort:
- The backtest y is `pct_new`, the tick verdict where re-priced.
- Master fills are verified: 191 of 191 are real MASTER_POOL rows with equal $, and the stamp flag equals `entry_adx < entry_adx_prev` on every row.
- Fresh overlaps master up to 10-14. Only the 6 LIVE fills from 10-15/16 are post-master.

**Coverage of the variables.**
- Every primary variable is scored on 97–100 % of fills in every block. The one exception is m_btc_r15, at 66 % in fresh.
- The closed-bar rebuilds are 100 % after a 9-symbol 5m top-up (90 request weight, no 418/429).
- 32 screen variables are thin in master: the BTC off24/off30d/r72/eff72 stamps 30–34 %, entry_gap_5_20 / 5_8 signed and s_gap520_chg 13 %, mcap/cmc 17 %, vol_mcap 5 %, and btc_rsi_closed 9 %. Their master Δ is read on that subset only.
- The 8 pattern-flag variables had no valid cell, so they were not tested. This is a blind spot.

## Primaries (pre-registered, stamp zone)

Δin = (ADX falling ∧ X) − (ADX falling ∧ ¬X), in %/fill, with the day-clustered z in brackets. Pass = negative in all 3 eras ∧ Stouffer z ≤ −2.64 ∧ master not opposite.

| primary (bad side X) | 2025 | 2026-H1 | 2026-H2 | Stouffer z | master Δin (n X) | stack Δin | fresh n·Δin | verdict |
|---|---|---|---|---|---|---|---|---|
| P1 RSI > 60.1 | −0.026 (−0.4) | +0.051 (+0.5) | −0.303 (−2.7) | −1.53 | **+0.281** (22) | +0.269 | 0 | fail |
| P2 RSI rolling over | −0.011 | −0.160 | +0.452 | +0.63 | −0.113 (26) | +0.149 | 2 · +1.11 | fail |
| P3 > EMA5 by 0.149 % | +0.063 | +0.026 | +0.125 | +1.42 | −0.015 (24) | −0.163 | 3 · +0.77 | fail (opposite) |
| P4 > EMA13 by 0.372 % | −0.024 | −0.060 | +0.217 | +0.35 | +0.145 (26) | +0.127 | 3 · +0.77 | fail |
| P5 EMA5–20 gap narrowing | +0.074 | −0.176 | −0.043 | −0.54 | −0.200 (19) | +0.042 | 1 | fail |
| P6 ADX > 21.5 | −0.048 | −0.062 | +0.141 | −0.15 | −0.134 (42) | −0.134 | 3 · +0.22 | fail |
| P7 DI spread shrinking | +0.044 | −0.024 | −0.169 | −0.43 | +0.081 (33) | +0.233 | 3 · +1.53 | fail |
| P8 volume fading (Σ3 ≤ prior 3) | −0.085 | −0.020 | +0.069 | −0.60 | −0.047 (49) | +0.047 | 6 · — | fail |
| P9 coin 1h slope ≤ 0 | +0.052 | +0.092 | −0.266 | −0.10 | +0.148 (30) | −0.049 | 4 · −0.22 | fail |
| P10 coin 4h gap ≤ 0 | −0.055 | +0.110 | −0.183 | −0.53 | −0.218 (23) | −0.333 | 3 · −0.33 | fail |
| P11 BTC 15m ≤ 0 | −0.018 | −0.063 | −0.119 | −0.90 | −0.379 (24) | −0.403 | 1 · −0.95 | fail (sign ok, z −0.9) |
| P12 run > 8 bars above EMA20 | +0.058 | +0.025 | −0.109 | +0.22 | −0.179 (41) | −0.130 | 3 · −0.30 | fail |

On the closed-bar zone, all 12 primaries also fail. The best is P5 (EMA gap narrowing), with Stouffer −1.41 and a positive 2025 sign. P11 (BTC 15m down) is sign-consistent in 3 eras and master on the stamp zone, but tiny: z −0.2 / −0.6 / −0.8. It is essentially the known BTC-short-term-down effect, not a new interaction.

## Exhaustive screen

Setup: 1,077 valid cells (stamp zone) and 1,068 (closed zone); 200 day-block shuffles; C = signed minimum |z| over the 3 eras.

| zone | stat | cells \|C\| ≥ 1.96 (obs / null 95th) | best \|C\| | null 95th max | family-wise p of best |
|---|---|---|---|---|---|
| stamp | inside zone (X vs ¬X) | 0 / 0 | 0.89 (n_adx4h ≤ med) | 1.92 | 0.99 |
| stamp | zone∧X vs everything else | 0 / 0 | 1.81 (n_adx4h ≤ med) | 1.85 | **0.08** |
| closed | inside zone | 0 / 0 | 1.06 (k_gap4h mid tercile) | 1.87 | 0.93 |
| closed | zone∧X vs rest | 0 / 0 | 1.41 (global volume ratio > med) | 1.82 | 0.36 |

**Walk-forward**, both directions: discover at |z| ≥ 1.96 ∧ |Δ| ≥ 0.05, then confirm the sign.

| zone · stat | 2025 → H1 and H2: confirmed / null 95th (p) | 2026 pooled → 2025: confirmed / null 95th (p) |
|---|---|---|
| stamp · inside | 2 / 37 (0.98) | 9 / 87 (0.95) |
| stamp · vs rest | 61 / 49 (**0.045**) | 101 / 129 (0.08) |
| closed · inside | 5 / 33 (0.89) | 10 / 77 (0.93) |
| closed · vs rest | 15 / 46 (0.23) | 59 / 103 (0.16) |

The only count above its null is stamp "vs rest" 2025 → 2026 (p 0.045). It is a count of many weak cells, and it reproduces the known fact that ADX-falling fills are worse than the rest, whatever X is. Of those 61 cells, 8 also have the same sign in master and fresh. It does not repeat in the other direction or on the closed zone.

**All-block agreement.** Same sign in 3 eras + master + stack_keep + fresh: 43 of 1,077 cells against about 34 expected by luck. None of them has |C| ≥ 1.0.

## Top cells: three blocks side by side, two-sided

**Rule**: refuse the bad side. **Selectivity** = kept avg − refused avg (day-bootstrap P > 0).
**Columns**: blocks are % per fill; Σ$ is recorded $ for master and fresh, and $700 1× notional for the backtest.
**Labels**: all of these are FAILING / descriptive (no family-wise survivor).

### A · ADX falling alone (reference: refuse every ADX-falling fill)

| block | ADX falling | ADX rising | selectivity [P] | refused W/L | Σ% before → after | Σ$ before → after |
|---|---|---|---|---|---|---|
| BT 2025 | 947 · 51 % · −0.233 | 1,670 · 54 % · −0.161 | +0.072 [0.95] | 481 / 466 | −489 → −269 | −3,424 → −1,881 (@ $700) |
| BT 26-H1 | 435 · 59 % · −0.127 | 736 · 60 % · −0.060 | +0.067 [0.86] | 255 / 180 | −100 → −45 | −699 → −311 |
| BT 26-H2 | 199 · 58 % · −0.092 | 322 · 66 % · +0.032 | +0.124 [0.91] | 116 / 83 | −7.9 → +10.4 | −55 → +73 |
| master as-traded | 77 · 62 % · −0.010 | 114 · 67 % · +0.052 | +0.062 [0.73] | 48 / 29 | +5.2 → +5.9 | +$802 → +$1,366 |
| master stack_keep | 53 · 72 % · +0.150 | 86 · 71 % · +0.135 | **−0.015** [0.46] | 38 / 15 | +19.6 → +11.6 | — |
| fresh | 7 · 57 % · +0.108 | 22 · 36 % · −0.190 | **−0.299** [0.13] | 4 / 3 | −3.4 → −4.2 | −$852 → −$430 |
| LIVE | 2 · 100 % · +0.681 | 4 · 50 % · −0.106 | −0.787 | 2 / 0 | +0.9 → −0.4 | +$4 → +$0 |

Master per batch, as-traded Σ% before → after: BASE +14.40 → +8.65 · B1 +3.51 → +0.69 · B2 −2.15 → −1.07 · B4 −0.85 → +0.68 · B5 +1.42 → 0 · B6 −1.70 → 0 · B7 −1.09 → 0 · B12 +1.09 → +3.51 · B14 −4.28 → −2.57 · B15 −1.99 → −0.99 · B18 −2.09 → −0.69 (all batches in the tables file).

### B · ADX falling ∧ pair volume ratio > 0.695 (best all-block lead; refuse the cell) — DESCRIPTIVE, fails the screen

| block | AF ∧ high vol | AF ∧ low vol | Δ in-zone [95 %] | ADX rising | refused: WR vs BE · P(avg<0) | selectivity [P] | refused W/L | Σ% before → after | Σ$ before → after |
|---|---|---|---|---|---|---|---|---|---|
| BT 2025 | 473 · 48 % · −0.258 | 474 · 53 % · −0.208 | −0.050 [−0.19, +0.08] | −0.161 | 48 vs 69 % · 1.00 | +0.087 [0.94] | 228 / 245 | −489 → −367 | −3,424 → −2,570 |
| BT 26-H1 | 215 · 56 % · −0.175 | 220 · 61 % · −0.080 | −0.095 [−0.27, +0.09] | −0.060 | 56 vs 66 % · 1.00 | +0.110 [0.97] | 120 / 95 | −100 → −62 | −699 → −435 |
| BT 26-H2 | 108 · 56 % · −0.191 | 91 · 60 % · +0.026 | −0.218 [−0.50, +0.07] | +0.032 | 56 vs 64 % · 0.99 | +0.222 [0.99] | 61 / 47 | −7.9 → +12.8 | −55 → +90 |
| master as-traded | 48 · 54 % · −0.211 | 29 · 76 % · +0.324 | **−0.536 [−0.93, −0.19]** | +0.052 | 54 vs 63 % · 0.99 | +0.319 [0.99] | 26 / 22 | +5.2 → +15.4 | +$802 → +$3,342 |
| master stack_keep | 29 · 66 % · −0.071 | 24 · 79 % · +0.417 | −0.488 [−0.97, −0.02] | +0.135 | 66 vs 61 % · 0.69 | +0.268 [0.97] | 19 / 10 | +19.6 → +21.6 | — |
| fresh | 2 · −0.448 | 5 · +0.331 | −0.779 | −0.190 | n 2 | +0.354 [0.69] | 1 / 1 | −3.4 → −2.5 | −$852 → −$539 |
| LIVE | 1 · +0.206 | 1 · +1.157 | — | −0.106 | n 1 | — | 1 / 0 | +0.9 → +0.7 | +$4 → +$3 |

Supporting reads for lead B:
- **No pair or day concentration.** No pair or day carries more than 27 % of the refused loss. LOMO across 22 months: Δin −0.098 … −0.061, no sign flip.
- **Without the washed-out window** (Jun-18 → Jul-2), master still separates. Inside the zone, high volume is 44 · −0.289 and low volume 26 · +0.271. On stack_keep it is 26 · −0.156 against 21 · +0.365.
- **It is an interaction, not a general volume effect.** Among ADX-rising fills, high volume is +0.054 / +0.106 / −0.093 by era, master −0.016, fresh −0.003.
- **The closed-bar version points the same way:** c_vol_r20 > 0.571 gives −0.04 / −0.20 / −0.08, master −0.31, fresh −1.28 (n 4).

Why it is not a candidate:
- It was selected from 1,077 cells, with family-wise p 1.00 inside the zone and 0.20 versus the rest.
- Its 2025 z is only −0.7.
- It contradicts the pre-registered direction (P8 predicted fading volume as the bad side).
- Fresh has 2 fills in the cell.
- Cross-period evidence can only refute, and master is the only block where it is large.

Master per batch for B, as-traded Σ% before → after: BASE +14.40 → +13.27 · B1 +3.51 → +2.90 · B2 −2.15 → −0.32 · B3 +3.28 → +4.41 · B4 −0.85 → −0.24 · B5 +1.42 → +0.05 · B6 −1.70 → +0.29 · B7 −1.09 → 0 · B8 −0.26 → +0.36 · B9 −0.23 → +0.51 · B12 +1.09 → +2.02 · B13 −1.65 → −1.17 · B14 −4.28 → −2.57 · B15 −1.99 → −0.99 · B16 −0.92 → +0.18.
- 12 batches improve and 3 get worse (BASE, B1, B5).
- Refused W/L by batch and the per-trade rows are in `ML_ADXFALL_tables_stamp.txt`.

### C · ADX falling ∧ coin 4h ADX ≤ 26.7 (best by the screen statistic; fw 0.08 vs rest) — FAILS master

| block | AF ∧ low 4h ADX | AF ∧ high 4h ADX | Δ in-zone | selectivity [P] | refused W/L | Σ$ before → after |
|---|---|---|---|---|---|---|
| BT 2025 / H1 / H2 | 475 · −0.263 / 195 · −0.215 / 100 · −0.187 | 472 · −0.203 / 240 · −0.056 / 99 · +0.005 | −0.060 / −0.159 / −0.192 | +0.09 [0.97] / +0.16 [0.98] / +0.21 [0.96] | 213/262 · 107/88 · 46/54 | −3,424 → −2,550 · −699 → −406 · −55 → +76 |
| master as-traded | 35 · −0.008 | 42 · −0.011 | **+0.003** | +0.043 [0.63] | 20 / 15 | +$802 → +$1,379 |
| master stack_keep | 25 · +0.182 | 28 · +0.121 | **+0.061** | −0.050 [0.35] | 17 / 8 | — |
| fresh | 6 · +0.076 | 1 · +0.299 | −0.223 | −0.246 [0.27] | 3 / 3 | −$852 → −$391 |

The backtest is clean but master is flat, so this cell is **refuted** for a ship.

### D · ADX falling ∧ RSI > 60.1 (best primary, P1) — FAILS (master opposite)

Backtest Δin: −0.03 / +0.05 / −0.30. Master is 22 fills at +0.191 against −0.090 (selectivity −0.185), and stack_keep shows +0.27 for the RSI-high side. Refusing it in master turns +$802 into −$340.

### Per trade: fresh ADX-falling fills (stamp or closed). The LIVE rows are the only post-master fills.

| src | opened | pair | ADX Δ stamp | ADX Δ closed | pair vol ratio | c_vol_r20 | 4h ADX | RSI | P&L % | $ |
|---|---|---|---|---|---|---|---|---|---|---|
| paper | 10-02 01:57 | ZRO | −0.34 | −0.36 | 0.86 | 1.13 | 25.0 | 57.4 | −1.102 | −313.8 |
| paper | 10-02 02:51 | ENA | −0.81 | −1.03 | 0.53 | 0.53 | 34.3 | 59.6 | +0.299 | +38.3 |
| paper | 10-02 02:53 | SUI | +0.46 | −0.05 | 0.62 | 0.46 | 22.0 | 62.0 | +0.190 | +45.3 |
| paper | 10-05 13:57 | NIL | −0.04 | −0.02 | 0.59 | 0.50 | 23.2 | 57.4 | +1.590 | +159.0 |
| paper | 10-07 16:31 | ARB | −0.35 | −0.36 | 0.68 | 0.81 | 19.5 | 58.4 | −0.694 | −155.1 |
| paper | 10-07 16:36 | UNI | −0.38 | −0.40 | 0.45 | 0.61 | 25.0 | 55.2 | −0.698 | −154.2 |
| **LIVE** | 10-15 14:36 | 1000PEPE | −0.54 | −0.54 | 0.66 | 0.65 | 19.4 | 59.6 | +1.157 | +2.8 |
| **LIVE** | 10-16 11:58 | APT | −0.00 | +0.43 | 0.86 | — | 14.4 | 60.1 | +0.206 | +0.4 |

## Recommendation (operator decides)

- **Nothing to arm and no new filter.** No ADX-falling × X cell meets the frozen decision rule:
  - no primary pass;
  - no family-wise survivor;
  - selectivity holds only in the backtest, or master refutes.
- **Keep the existing DL 273 ADX observe line and its frozen bar unchanged.** This study does not sharpen it.
- **Optional, operator's choice:** register lead B as a pre-registered OBSERVE-only signature with a frozen cut, so later batches can test it out of sample.
  - Signature: ADX falling (stamp) ∧ `entry_pair_volume_ratio` > 0.695.
  - It costs nothing, since both columns are already stamped.
  - Suggested frozen bar: fresh N ≥ 15 cell fills on ≥ 8 days. The cell's avg must sit below the ADX-falling ∧ low-volume avg, and below the sleeve breakeven WR, at day-bootstrap P ≥ 0.90. Retire it if the cell avg ≥ the rest.
  - Until then it is a hypothesis that contradicts my own prior. Treat it as such, and apply the 30–50 % in-sample haircut to any projection; master's +$2.5k "saved" is in-sample.

## Blind spots

- Pattern-flag variables were not testable (no valid cell).
- The master values of 32 BTC/gap stamps are read on 5–34 % coverage.
- The forming-bar volume stamp depends on how many seconds into the bar the fill happened. The closed-bar c_vol_r20 removes that and agrees in sign, but is weaker.
- Fresh is 29 fills (7 ADX-falling), and only the 6 LIVE fills are post-master.
- No order-book, open-interest or liquidation data.
- `validate_against_master.py`: **ALL CHECKS PASS** (run at the end). Study-specific check: 191 / 191 master rows are real, with equal $ and sign.

Post-freeze code changes (reporting only, no cut or statistic changed):
- LOMO for the zone-alone reference now compares against ADX-rising.
- The deep-read picks skip mirror cells and add the top-2 |C_rest| variables.
