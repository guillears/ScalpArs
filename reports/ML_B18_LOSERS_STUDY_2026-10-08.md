# B18 momentum-long losers — ARB / UNI / LIT (2026-10-07) · headline: ML_B1H_NEGFLANK

Read-only research, 2026-10-08. No config, engine or scout changes. `scripts/validate_against_master.py` ran first and passed every check.

**Scripts (new):** `scripts/study_ml_b18_common.py` (loaders), `scripts/study_ml_b18_losers.py` (NEGFLANK, the specific tests, the exhaustive sweep → `reports/ML_B18_STUDY_tables.md` + `reports/ML_B18_STUDY_sweep.csv`), `scripts/study_ml_b18_watchitems.py` (which watch lines catch the 3), `scripts/study_ml_b18_frozen_combos.py` (the frozen Oct-4 watch-flag combinations re-read).

**Cohorts**
- **Master:** `MASTER_POOL_stacked.csv` (STACK 2026-10-08b, B18 included by the builder), kept, non-probe, CLOSED MOMENTUM LONG. % = `stack_pct`, $ = `stack_pnl` (today's sizing). That gives 115 fills without B1 (1× probe era, shown separately). "Washed-out" uses the scout's frozen definition (BTC ≤ −15 % below its 30-day high). That is the live stamp where it exists; otherwise it is rebuilt from BTC 5m klines, which flags the same 19 fills as the Jun-18→Jul-2 date window (only 1 fill differs). BTC 1d return is the stamp where it exists, else rebuilt from k5m. On all 6 fills where both are available the rebuild matches the stamp to within 4e-5.
- **yr5 replay:** ML fills trimmed to each chunk's window (`yr5_fills_trimmed.load`): 1,784 fills over 3 seeds (595 per seed), Jan-04 → Oct-02. N is quoted **per seed**. $ is on a fixed $3k book at live sizing, per seed. Days are pooled across seeds.
- **Units:** every market-wide variable is judged in DAYS. The bootstrap is day-clustered (4,000 resamples). The "shuffled-day null" permutes whole days of outcomes against the fixed features.

## Verification of the "known" stamps
Confirmed from the stamps: BTC 1h slope −0.30 / −0.30 / −0.23 · regime HEALTHY_BULL · BTC RSI 65.2 / 64.1 / 61.7 · global volume 0.51 / 0.44 / 0.48 · bull breadth 76 / 73 / 56 · BTC 1d −0.24 % · trend gap +0.10 · off30d −4.4.

One small correction: BTC EMA20 slope is 0.073 / 0.070 / **0.046**, so LIT is below 0.05.

- **Timing:** LIT opened 29.5 min after UNI's stop (UNI closed 16:43:43).
- **Path:** peaks were +0.04 / 0.00 / 0.00, so all three went straight to the stop. After the exit, price kept falling (final −1.65 / −1.03 / −0.17). No exit or wider-stop change would have helped.
- **Same moment:** all three fills are **one day and one BTC move**. For any market-wide signature they add ONE observation, not three.

## 1. Which watch lines catch the 3 fills

| watch item | ARB | UNI | LIT | status |
|---|---|---|---|---|
| **ML_B1H_NEGFLANK (210)**: BTC 1h slope ≤ −0.05 | ✔ | ✔ | ✔ | scout 3/15 fills · 1/8 days → collecting |
| BTC 1h slope < 0 tally (ML pin) | ✔ | ✔ | ✔ | observe tally |
| ML_STOP_COOLDOWN (248) | · | · | ✔ | LIT qualifies, but counting starts 10-08 → reference only |
| CLUSTER2_120 (exploratory) | · | · | ✔ | tally only |
| **BTC_HOT_MATURE** (Jul-16 macro watch: BTC ADX ≥ 25 ∧ (ATR ≥ 0.15 ∨ ext13 ≥ 0.20)) | ✔ | ✔ | ✔ | old watch; its gate is master N ≥ 30 across ≥ 3 weeks |
| OFF24 (BTC ≥ 2 % below 24 h high; Oct-4 combo flag) | ✔ | ✔ | ✔ | — |
| ZC (BTC EMA50/100 gap ≤ 0.006 ∧ ETH 5m ≤ 0; Oct-4 combo flag) | ✔ | ✔ | · | — |
| BTC 1d < 0 (FRENZY bearish-day half-leg) | ✔ | ✔ | ✔ | not a long watch |
| BTC-chop eff72 ≤ 0.007 · burst ≤ 2 min · LONG_CHOP_BURST · heat 3-leg · late-entry 3-leg (39) · PVR band ③ · ATR×GAP · ROLL · SPRINT · megacap · rebound · FRENZY bearish day (1d < 0 ∧ gap < 0) | · | · | · | none of these |

Three near-misses:
- LIT just missed the armed LOADX filter: its RSI was falling, but ADX was 21.86 against a 21 limit.
- ARB's pair-volume ratio was 0.676, just under the 0.68 band.
- The burst flag missed UNI by 108 s: the gap to ARB was 228 s against a 120 s limit.

## 2. HEADLINE — ML_B1H_NEGFLANK

### 2.1 NEGFLANK vs the rest of the sleeve

| cohort | NEGFLANK: N · WR · avg % · Σ$ · days · day-CI · P(mean<0) | rest | Δ | sleeve BE WR |
|---|---|---|---|---|
| master ex-B1 | 37 · 73% · +0.176 · +$1,010 · 22d · [−0.13, +0.46] · 0.12 | 78 · 76% · +0.186 · +$2,006 | −0.01 | 61.3% |
| master ex-B1 ex-BASE | 22 · 55% · −0.155 · −$658 · 12d · [−0.62, +0.24] · 0.79 | 61 · 72% · +0.146 | −0.30 | 62.6% |
| **master ex-B1 ex-washed (judged)** | **24 · 58% · −0.061 · −$354 · 13d · [−0.49, +0.33] · 0.63** | 72 · 74% · +0.151 · +$1,431 | −0.21 | 62.4% |
| master ex-washed **without B18's 3** | 21 · 67% · +0.030 · 12d · P 0.46 | +0.151 | −0.12 | |
| master WITH B1 ex-washed | 34 · 62% · −0.049 · 18d · P 0.64 | +0.194 | | |
| master as-traded (all real fills, rule 204) ex-B1 ex-washed | 33 · 58% · −0.111 · 17d · P 0.76 | 103 · 63% · −0.019 | −0.09 | |
| yr5 all | 251/seed · 59% · −0.124 · −$6,419/seed · 140d · [−0.20, −0.04] · 1.00 | 344 · 63% · −0.025 | −0.10 | 66.9% |
| **yr5 ex-washed (judged)** | **163/seed · 60% · −0.127 · −$4,533/seed · 98d · [−0.23, −0.02] · 0.99** | 263 · 65% · −0.004 | −0.12 | 67.4% |
| B18's 3 | 3 · 0% · −0.695 · −$447 · 1d | | | |

The master's negative NEGFLANK reading comes entirely from B18. Before B18, NEGFLANK on master was +0.030 % with a 67 % win rate.

### 2.2 Dose-response (pre-registered cut −0.05; the −0.15 and −0.20 cuts are POST-HOC)

| BTC 1h slope | master ex-B1 | master ex-washed | yr5 all | yr5 ex-washed |
|---|---|---|---|---|
| ≤ −0.30 | 9·89%·+0.378·7d | 3·67%·+0.009·3d | 69·57%·−0.160·49d | 34·60%·−0.114·26d |
| (−0.30, −0.15] | 9·67%·−0.119·5d | 7·57%·−0.207·3d | 87·57%·−0.139·69d | 61·56%·−0.145·44d |
| (−0.15, −0.05] | 19·68%·+0.221·15d | 14·57%·−0.003·11d | 95·63%·−0.085·77d | 69·63%·−0.118·57d |
| (−0.05, +0.05] (live dead-band blocks −0.05…+0.025) | 9·67%·+0.032·7d | 9·67%·+0.032·7d | 49·65%·−0.048·43d | 34·61%·−0.114·31d |
| > +0.05 | 69·77%·+0.206·34d | 63·75%·+0.168·30d | 295·63%·−0.021·158d | 229·66%·+0.012·125d |

**Not dose-responsive.** On yr5 it is a step: every bucket at or below +0.05 sits around −0.11…−0.15, and only a rising 1h (> +0.05) is flat or positive. A deeper pullback is no worse than a mild one. On master the buckets are noise: there are 3–14 fills per bucket.

Post-hoc deeper cuts help nowhere:
- ≤ −0.15: master 10·60%·−0.142, P 0.74 · yr5 94·58%·−0.134, P 0.96.
- ≤ −0.20: master 6·50%·−0.193 · yr5 68·57%·−0.110, P 0.89.

### 2.3 Stability

**Shuffled-day null (1,000 permutations, one-sided):**

| cohort | Δ (NEGFLANK − rest) | p |
|---|---|---|
| master ex-B1 | −0.009 | 0.47 |
| master ex-washed | −0.212 | 0.14 |
| yr5 all | −0.099 | 0.017 |
| yr5 ex-washed | −0.123 | 0.009 |

**Halves (time split):**
- yr5 ex-washed: H1 Δ −0.100, H2 Δ −0.141. Both negative.
- Master ex-washed: H1 (Jul-2→Aug-22) Δ +0.006, H2 (Aug-22→Oct-7) Δ −0.336. Only the half that contains B18 is negative.

**yr5 by month:**
- Δ is negative in 7 of 9 full months: Jan −0.11 · Feb −0.07 · Mar −0.06 · Apr +0.04 · May +0.01 · Jun −0.02 · Jul −0.14 · Aug −0.19 · Sep −0.30.
- Leaving any one month out keeps Δ between −0.077 and −0.111.

**Master leave-one-batch-out:** Δ ranges −0.30 (BASE left out) to +0.07 (B18 left out). Removing B18 is enough to flip the sign.

**Loss concentration (ex-washed):**
- Master: largest single day 23 %, largest single pair 22 %.
- yr5: 6 % and 6 %.

### 2.4 Interactions — why

Cells are NEGFLANK ∧ X. The p value is a day-null on the cell versus the rest of the sleeve. Halves are given as H1 / H2.

- **5m BTC EMA20 slope > 0 ("bounce inside a 1h pullback"):** this is not an extra variable. **100 % of NEGFLANK fills in both cohorts already have the 5m slope > 0**, because the live 5m macro-slope gate requires it for any long. Every NEGFLANK fill is a 5m bounce inside a 1h pullback, so this explains nothing beyond NEGFLANK itself.
- **Global volume < 0.6:**
  - Master: 13 · 38 % · −0.272, against +0.189 for NEGFLANK on normal volume. p 0.022, halves −0.96 / −0.31.
  - yr5: 52 · 55 % · −0.136, against −0.123 on normal volume. p 0.10, halves −0.26 / **+0.05**.
  - **yr5 refutes "low volume makes it worse".**
  - By tercile on yr5: low −0.165 · mid −0.275 · high **+0.050**. On master the high tercile is 5 · 100 %. The only consistent reading is "NEGFLANK is harmless when market volume is HIGH (≥ ~0.95)". That is post-hoc and observe only.
- **BTC RSI ≥ 60:**
  - yr5: −0.168 against +0.024 when RSI < 60. p 0.000, halves −0.11 / −0.22.
  - **Master is the opposite:** +0.031 against −0.523.
  - Master refutes it.
- **Regime:** HEALTHY_BULL and STRONG_BULL are both negative on yr5. On master there are too few STRONG_BULL fills (3) to say.
- **Washed-out:** on master, washed-out NEGFLANK fills are 13 · 100 % · +0.615, so washed-out tapes rescue the flank. On yr5 they are −0.119, the same as non-washed. Any live rule would need the washed-out exemption, which is how the scout already judges it.
- **NEGFLANK × BTC_HOT_MATURE (2×2, ex-washed):**

| | HOT | not HOT |
|---|---|---|
| NEGFLANK, yr5 | 37 · 43 % · −0.228 | 126 · 65 % · −0.098 |
| not NEGFLANK, yr5 | 38 · 57 % · −0.155 | 225 · 67 % · **+0.021** |
| NEGFLANK, master | 7 · 43 % · −0.383 (3 of the 7 are B18) | 17 · 65 % · +0.072 |

On yr5 the two legs add up. B18 is the HOT ∧ NEGFLANK corner.

### 2.5 How NEGFLANK fills lost (caps for losers?)

| cohort | losers whose peak never reached 0.10 (straight to the stop) | losers that peaked ≥ 0.40 (armed, then reversed) | winners that dipped ≤ −0.50 first |
|---|---|---|---|
| master ex-washed NEGFLANK | 60 % (of 10) | 0 % | 14 % |
| master ex-washed rest | 74 % | 0 % | 26 % |
| yr5 ex-washed NEGFLANK | 51 % (of 66) | 2 % | 23 % |
| yr5 ex-washed rest | 54 % | 2 % | 25 % |

NEGFLANK fills lose the same way as the rest of the sleeve: mostly dead on arrival, almost never after arming.
- A take-profit or giveback cap has nothing to bite on.
- A tighter stop would also cut the roughly 1 in 4 winners that dip first.

The exit side has no fix. If NEGFLANK matters, it is an entry or sizing question. (This is indicative only; no path-level stop counterfactual was run. Under the stop-counterfactual rule that would have to be done on live-stopped fills.)

### 2.6 Verdict against the frozen bar

The bar: WR < sleeve breakeven ∧ day-clustered P(mean<0) ≥ 0.95 ∧ ≥ 8 days ∧ N ≥ 15 ∧ no day or pair ≥ 50 % of the loss, judged ex washed-out.

| | WR < BE | P(<0) ≥ .95 | ≥ 8 days | N ≥ 15 | concentration | **result** |
|---|---|---|---|---|---|---|
| master (historical) | 58 < 62.4 ✔ | **0.63 ✗** | 13 ✔ | 24 ✔ | 23 % / 22 % ✔ | **FAIL** (and +0.030 without B18) |
| yr5 replay | 60 < 67.4 ✔ | 0.99 ✔ | 98 ✔ | 163/seed ✔ | 6 % / 6 % ✔ | **PASS** |
| scout forward (since 10-06) | 3 · 0 % | — | 1/8 | 3/15 | — | collecting |

**Only yr5 passes, so the rule says observe. Keep collecting; do not propose a block.**

Two caveats weaken the yr5 pass:
1. **yr5 is in-sample for this line.** NEGFLANK was registered on 10-05 citing the yr5 "falling 1h EMA20" figure, so the replay is the data that generated the hypothesis.
2. **yr5's whole sleeve is negative:** −0.051 ex-washed, against master +0.10 / as-traded ≈ 0. A block there is "cut the worse half of a losing book". On master, pre-B18, the same block would have cost about +$93.

If the forward bar ever passes, here is what it would be worth and how it would be guarded:
- yr5 Δ −0.123 %/fill, after the 30–50 % haircut, is about **+0.06 to +0.09 % per blocked fill**.
- That is a small edge. It needs the washed-out exemption.
- Suggested pre-committed revert: the first 10 refused signals, re-priced with the live exit, show WR ≥ 61 % or Σ > 0 → revert.

## 3. The other requested signatures (block cohort vs rest, ex washed-out)

| signature | master: N · WR · avg · Σ$ · days · P(<0) · day-null p | yr5: N/seed · WR · avg · days · P(<0) · day-null p | read |
|---|---|---|---|
| NEGFLANK ∧ gvol < 0.6 | 13 · 38% · −0.272 · −$628 · 8d · 0.82 · 0.024 | 52 · 55% · −0.136 · 45d · 0.87 · 0.10 | yr5 shows no extra effect over NEGFLANK alone |
| NEGFLANK ∧ gvol terciles (low/mid/high) | −0.162 / −0.110 / +0.281 | −0.165 / −0.275 / +0.050 | high-volume side fine in both (post-hoc) |
| NEGFLANK ∧ 5m BTC slope > 0 | ≡ NEGFLANK | ≡ NEGFLANK | the gate makes these identical |
| BTC 1d < 0 | 41 · 56% · −0.169 · −$1,375 · 18d · 0.89 · 0.002 | 188 · 65% · −0.047 · 84d · 0.84 · 0.56 (rest −0.055) | **master only; yr5 refutes** |
| bearish day (1d < 0 ∧ trend gap < 0) | 6 · 50% · −0.547 · 6d | 52 · 60% · −0.096 · 40d · 0.80 · 0.23 | N too small / no |
| NEGFLANK ∧ 1d < 0 (B18 shape, post-hoc) | 16 · 50% · −0.340 · −$1,104 · 9d · 0.98 · 0.000 → ticks the bar | 103 · 64% · −0.080 · 57d · 0.88 · 0.30 (rest −0.042) | **without B18 it is 13 · 62 % · P 0.93 (fails); yr5 refutes** |
| cooldown < 30 min after an ML stop | 7 · 57% · −0.146 · 6d | 9 · 44% · −0.192 · 15d · 0.90 · 0.17 | under N; the scout line stays |
| cluster ≥ 2 ML in the prior 120 min | 19 · 68% · +0.020 · 11d · 0.43 | 48 · 55% · −0.127 · 46d · **0.98** · 0.09 | yr5 ticks the bar; master refutes; exploratory |
| B18 full shape (NEGFLANK ∧ gvol < 0.6 ∧ 1d < 0) | 9 · 33% · −0.496 · 5d | 28 · 63% · −0.021 · 23d · 0.56 | **yr5 refutes the B18 composite** |

### 3a. Frozen Oct-4 watch-flag combinations that hit B18

These use the `ml_watch_combo_screen.py` definitions, never re-tuned. yr5 shares yr4's calendar, so it is a re-validation of the Oct-4 screen, not independent out-of-sample data. Master here uses scored fills only.

| combination | B18 hits | yr5 ex-washed N/seed · WR · avg · days · P(<0) | Δ · day-null p | H1 / H2 | months Δ<0 | master scored | master forward ≥ Oct-4 |
|---|---|---|---|---|---|---|---|
| **HOT** (BTC_HOT_MATURE) | all 3 | 75 · 50% · −0.191 · 67d · 1.00 | −0.170 · 0.002 | −0.13 / −0.22 | 6/7 | 17 · 65% · −0.018 (rest +0.123); **without B18 14 · 79 % · +0.127** | 4 · 25% · −0.283 |
| OFF24 | all 3 | 89 · 54% · −0.157 · 56d · 0.98 | −0.134 · 0.025 | −0.09 / −0.20 | 7/8 | 4 · 0% (3 = B18) of 44 scored | 3 · 0% |
| SLOPE ∧ ZC ∧ OFF24 | ARB, UNI | 21 · 44% · −0.408 · 24d · 1.00 | −0.375 · 0.000 | −0.34 / −0.53 | 5/6 | 2 · 0% (both B18) of 15 scored | 2 · 0% |
| NEG ∧ HOT | all 3 | 37 · 43% · −0.228 · 33d · 0.97 | −0.194 · 0.013 | −0.10 / −0.38 | 5/7 | 7 · 43% · −0.383 (4 · 75 % without B18) | 3 · 0% |

More on two of these:
- **BTC_HOT_MATURE** is the most interesting line here, because it was registered on Jul-16 from MASTER data, which makes yr5 genuinely out-of-sample for it.
  - On yr5 it clears every part of the bar: WR 50 vs breakeven 67.4, P 0.996, 67 days, largest day 8 %, largest pair 9 %. All 3 seeds are negative (−0.25 / −0.13 / −0.20) and both halves are negative.
  - On master, before B18, it was a winning zone (14 · 79 % · +0.127), so master refutes it.
  - Under the "master can only refute" rule it stays watch-grade. Its own Jul-16 gate (master N ≥ 30 across ≥ 3 weeks, negative each week) is far from met.
- **SLOPE ∧ ZC ∧ OFF24** was #3 by z in the Oct-4 screen. It failed there only because master had zero scored fills. B18's ARB and UNI are its first two genuine out-of-sample master fills, and both lost. That is still one window.

## 4. Exhaustive stamped-column sweep

**`scripts/sweep_separators.py ML`** runs on SCREENED_BASELINE: only 37 ML fills, Jun-18→Sep-18. BTC 1h slope is not among its consistent dimensions. Its consistent list is led by funding rate (not stamped on yr5) and by global volume above the median being WORSE (Δ −0.45 / −0.34), which is the opposite direction to the B18 "low volume" story.

**Own sweep:**
- Inputs: 36 entry stamps with ≥ 80 % coverage on BOTH cohorts.
- Splits: sign, median, and outer terciles, in both directions, plus every 2D pair as median quadrants. That is **2,654 masks** (1D 156 · 2D 2,498).
- Discovery set: yr5 ex-washed (1,279 fills, 179 days). Statistic: Welch z of zone vs rest.
- Family-wise null: the 95th percentile of the maximum |z| over 1,000 shuffled-DAY permutations = **5.12**. Under shuffled-trade permutations it is 4.21, but that null is anti-conservative with clustered days.

| | result |
|---|---|
| observed max \|z\| | 4.75: P(day-null max ≥ observed) = **0.134** → not significant |
| survivors of the family-wise day null | **0** |
| + both halves + leave-one-month-out + master direction (N ≥ 8) | **0** |

The top of the list is one family: **BTC 1h weakness** (1h slope ≤ 0.064, BTC 1h RSI ≤ 54) crossed with low pair ADX, low pair EMA20/50 gap, or low global volume. Each is about 110–125 fills/seed at −0.18…−0.20 against ≈ 0, with both yr5 halves negative and master direction agreeing (master Δ −0.14…−0.33 on 19–30 fills). It is the same NEGFLANK theme, and it does not beat the family-wise null. Full output: `reports/ML_B18_STUDY_sweep.csv` and `reports/ML_B18_STUDY_tables.md`.

## 5. Plain answer

- **Watch lines that catch the 3:**
  - NEGFLANK, the slope < 0 tally, BTC_HOT_MATURE and OFF24 catch all three.
  - ZC catches ARB and UNI.
  - Stop-cooldown and cluster2 catch LIT only.
  - Nothing armed should have stopped them. LOADX missed LIT on ADX 21.86 vs 21.
- **Separating signature on master AND backtest:** **none.**
  - Every candidate that looks bad on yr5 (NEGFLANK, HOT, OFF24, the combos) is flat or positive on master once B18's own three fills are removed.
  - Every candidate that looks bad on master (BTC 1d < 0, NEGFLANK ∧ 1d < 0, low global volume) is refuted by yr5.
  - The exhaustive 2,654-mask sweep has no survivor of the shuffled-day null (best |z| 4.75 vs 5.12, p 0.13).
- **NEGFLANK verdict:**
  - The historical evidence passes the bar on yr5 only. That run is in-sample and comes from a losing book.
  - Master fails: P 0.63, and +0.030 without B18.
  - → **observe-only, keep collecting** at the scout: 3/15 fills, 1/8 days.
- **Recommendation: ship nothing.** Keep NEGFLANK observing.
  - Optional: add BTC_HOT_MATURE (and NEG ∧ HOT as a comparison line) to the scout as a frozen observe line with the same bar, since yr5 is genuinely out-of-sample for it.
  - Note "NEGFLANK ∧ market volume ≥ 0.95 looks harmless" as a post-hoc hypothesis only.
  - B18 counts as ONE window for every market-wide line.

## 6. Blind spots (not tested, or tested weakly)

- **Low master coverage:** 15 stamps are below 80 % on master (stamped only since Sep-20 / Oct-1). They were swept on yr5 only, never on master: BTC 1d, eff72, off24h / off24lo, EMA50/100 gap, ETH 5m, pair 1h EMA20/200 gap, r72, above72, off30d, rsi_closed, 5-8 / 5-20 signed gaps, pair age.
  - BTC 1d and off30 were rebuilt from klines and validated (6/6 exact; off30 r 0.84, used for the washed-out flag only).
  - Master rows for OFF24, ZC and CHOP use scored fills only (44 / 15 / 32 of 96).
- **Pair 1h EMA20/200 gap** (B18's pairs were −6 to −8 %): yr5 gap ≤ 0 gives 160 · 57 % · −0.124 vs −0.006 (P 0.99), but within NEGFLANK it adds nothing (−0.037 vs −0.150). Master has 15 stamped fills, so it is untestable there. Forward stamps will accrue.
- **Not stamped on yr5 at all:** funding rate (sweep_separators' top dimension), market cap / CMC rank, order book, tick microstructure, news.
- **yr5 fidelity:** it reproduces live momentum longs about 64 %. Its ML sleeve is net-negative while master is positive, so the populations differ, and its window ends Oct-02 (no B18-era tape).
- **Exit side:** no path-level stop or cap counterfactual was run, only peak/trough anatomy. B18's three were straight-down stops with price continuing lower after the exit.
- **B18 shares one tape:** the three fills are one BTC move. ARB and UNI opened 4 minutes apart, and pair co-movement was not modelled.
- **Sequencing:** cooldown and cluster on master use as-traded non-probe ML fills within the same era. Cross-sleeve slot effects (FRENZY, fades, the bear-run positions open at the time) were not modelled.
