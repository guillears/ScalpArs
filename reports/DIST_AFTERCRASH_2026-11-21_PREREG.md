# DIST_AFTERCRASH — pre-registration (2026-11-21, written before any outcome of this study was read)

Research only. No bot code or config touched.

## Question
Do DISTRIBUTION_SHORT fills taken **after the coin already crashed hard** (the dump happened before our signal closed) lose?
Motivation: BLESS 2026-11-21 09:30 UTC — −21.9 % signal 15m bar (−24 % over 60 min), $352k long liquidations in 15 min, stopped −6 % at the top of the reflex bounce 7 min later.

## Theory (stated first)
DIST shorts a large holder distributing into a pump. The edge is the supply still to come. If the dump already happened in the signal bar(s) (capitulation / liquidation cascade done), the forced sellers are spent, the short sells the low, and the next move is the mean-reverting reflex bounce — where the stop sits.
Prediction: the deeper the pre-entry fall, the worse the short (monotone dose), worst at the extreme tail.

## Known before this pre-registration (disclosed — this is not a fresh variable)
- DIST_WL (L1 pre-RSI cohort) P1 / P9 / H1: "already dumped" (far below peak / VWAP, late in episode, 1 h change < 0) pointed the right way but weakly (|t| ≤ 1.9 on net).
- DIST_LOSERS2 (today's 1,547-fill kept cohort, quintiles frozen on DESIGN): `bar_ret_pct` Q1 (≤ −4.8 %) HOLDOUT avg +1.70 % vs rest (t +2.7 — **better**, not worse); `dd_peak_pct` Q1 (≤ −8.4 % below the episode peak) HOLDOUT +1.74 (t +3.3, better); `ret1h_pct` Q1 (≤ +3.3 %) HOLDOUT +1.60. At quintile granularity the theory is already contradicted.
- **What is new here:** the extreme tail (≤ −10 / −15 / −20 % in 15–60 min, i.e. the BLESS signature, 5 % of fills or fewer), the 30-min window, the fall from the 60-min high, and a liquidation-cascade proxy (signal-bar volume vs its 24 h norm).

## Data (frozen)
- Cohort: `reports/ENTRY_WL_2026-11-18_data/dist.pkl` (DL 586 kept: RSI1h ≥ 74.07, one position per pair), outcome = Exit B `REF` net % from `reports/DIST_THEORY_EXIT_2026-11-21_data/perfill.csv.gz` (TP min(20×ATR,10) maker / SL min(1.5×ATR,6) / 24 h, slip 0.10 %, real funding). 1,547 fills, DESIGN 2023-07 → 2025-06 (559), HOLDOUT 2025-07 → 2026-11 (988).
- Features (`scripts/DIST_AFTERCRASH_2026-11-21_features.py`, 5m bars closed at or before the signal; lag-checked: r60 = ret1h_pct to 6e-6, r240 = ret4h_pct, corr(r15, bar_ret_pct) = 0.99996):
  `r15`, `r30`, `r60`, `r240`, `dd24` (below 24 h high), `drop60` (below 60-min high), `volx15` (signal-15m quote vol / median 15m vol of prior 24 h).
- Outcome-blind tail counts (r15 ≤ −15: DESIGN 5 / HOLDOUT 70; ≤ −10: 23 / 170). **DESIGN is too thin at the tail to pick a threshold**, so thresholds are fixed by theory now, not fitted.

## Frozen variables, buckets and the candidate
- Primary: **`r15`** buckets ≤ −20 · (−20, −15] · (−15, −10] · (−10, −5] · > −5.
- Secondary family (same bucket edges): `r30`, `r60`, `drop60`; `dd24` buckets ≤ −30 · (−30,−20] · (−20,−10] · > −10; `r240` sign.
- **Primary candidate block: `r15 ≤ −15 %`** (BLESS = −21.9 %). Alternates in the family: r15 ≤ −10 / ≤ −20, r30 / r60 / drop60 ≤ −15.
- Liquidation proxy composite: `r15 ≤ −10` ∧ `volx15` ≥ the cohort's top-tercile edge (outcome-blind quantile of all 1,547 fills).
- BTC daily regime: `btc_d1_regime` sign (1d EMA50 vs EMA200), read at sign granularity.

## Decision rule (frozen)
1. Because DESIGN is thin, the read is on ALL fills, with DESIGN and HOLDOUT shown separately; a direction that flips between the blocks (each N ≥ 10) = no ship.
2. **FILTER CANDIDATE** only if the primary block passes every leg of the locked Pattern-C expectancy bar: WR < sleeve breakeven WR (|avg loss| / (avg win + |avg loss|) on all 1,547 kept fills) · avg < 0 with ≥ 95 % confidence by a 4 h-window-clustered bootstrap (5,000) · ≥ 8 distinct windows · no window / pair ≥ 50 % of the loss · N ≥ 15; AND monotone dose (r15 buckets); AND its effect beats a family-wise shuffled null (shuffle the variable within half-years, 2,000×, max |t| over the 14 family cuts). Then 30–50 % haircut and DCR before → after at one size (0.61 × equity per fill).
3. **OBSERVE line** (frozen threshold r15 ≤ −15, read bar = the expectancy bar above on real fills) if the block is below breakeven WR / avg < 0 but fails confidence or the null.
4. **REFUTED** if the block's avg ≥ 0 or ≥ the rest, or the dose is non-monotone with winning flanks.
5. Real fills (6 DIST closed fills, all 11-19 → 11-21) are shown individually — tiny N, a read of direction only.
