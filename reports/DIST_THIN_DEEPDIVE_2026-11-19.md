# DIST thin coins (24 h volume ≤ $42M): deep dive (2026-11-19)

Research only. Descriptive read of the frozen ENTRY_WL observe line; no new threshold was fitted. Script `scripts/dist_thin_deepdive.py`; fills `DIST_THIN_DEEPDIVE_2026-11-19_fills.csv.gz`.

## Question
Both live DISTRIBUTION losses (NAORIS $25.4M, SENT $24.6M) were thin coins. How strong is the thin-coin effect, how does it work, and what should be done?

## Method
1,534 RSI-kept DIST fills (2023-07 → 2026-11), Exit B on 1m, fees + slippage model + real funding (DIST_EXIT REF_B net). DESIGN < 2025-07 ≤ HOLDOUT. Day-block (10-day) bootstrap CIs; Kelly at the one-sided 95 % Wilson floor. Real slippage: the master pool's live stop-type exits (49, ALL sleeves, 0 DIST fills — a proxy).

## Results
## 1. Dose-response: 24 h volume buckets (all fills)

| group | N | days | WR | avg net % (day-block 95 % CI) | avg win / loss | worked (+1 % first) · avg reach % · stopped after ≥ +3 % | slip · funding % | Kelly floor · observed |
|---|---|---|---|---|---|---|---|---|
| < $15M | 0 | | | | | | | |
| $15–25M | 226 | 190 | 38% | +0.34 (-0.59 … +1.26) | +8.32 / -4.47 | 83% · 5.6 · 24% | 0.10 · -0.05 | -0.038 · +0.041 |
| $25–42M | 447 | 337 | 40% | +0.51 (-0.18 … +1.27) | +8.34 / -4.67 | 85% · 6.0 · 29% | 0.10 · -0.05 | +0.003 · +0.061 |
| $42–80M | 427 | 325 | 44% | +1.27 (+0.53 … +1.97) | +9.05 / -4.96 | 84% · 6.4 · 23% | 0.10 · -0.13 | +0.080 · +0.141 |
| $80–200M | 299 | 230 | 43% | +0.95 (+0.00 … +1.93) | +9.05 / -5.11 | 82% · 6.4 · 25% | 0.10 · -0.09 | +0.032 · +0.105 |
| ≥ $200M | 135 | 117 | 51% | +2.10 (+0.74 … +3.53) | +9.18 / -5.31 | 81% · 7.3 · 22% | 0.10 · -0.13 | +0.117 · +0.228 |

## 2. Thin vs rest by block and half-year

| group | N | days | WR | avg net % (day-block 95 % CI) | avg win / loss | worked (+1 % first) · avg reach % · stopped after ≥ +3 % | slip · funding % | Kelly floor · observed |
|---|---|---|---|---|---|---|---|---|
| DESIGN · thin | 185 | 140 | 35% | +0.02 (-0.84 … +0.89) | +6.79 / -3.65 | 80% · 4.8 · 23% | 0.10 · -0.02 | -0.082 · +0.003 |
| DESIGN · rest | 368 | 228 | 46% | +1.58 (+0.89 … +2.35) | +8.64 / -4.41 | 86% · 6.1 · 18% | 0.10 · -0.12 | +0.120 · +0.183 |
| HOLDOUT · thin | 488 | 304 | 41% | +0.62 (-0.11 … +1.43) | +8.84 / -4.99 | 85% · 6.3 · 29% | 0.10 · -0.06 | +0.014 · +0.070 |
| HOLDOUT · rest | 493 | 275 | 44% | +1.07 (+0.38 … +1.73) | +9.40 / -5.54 | 80% · 6.9 · 28% | 0.10 · -0.12 | +0.056 · +0.114 |

| half | thin share of fills | thin avg | rest avg | Δ (rest − thin) |
|---|---|---|---|---|
| 23-H2 | 37% | +1.07 | +3.77 | +2.70 |
| 24-H1 | 30% | +0.44 | +0.68 | +0.24 |
| 24-H2 | 33% | -0.09 | +1.09 | +1.17 |
| 25-H1 | 34% | -0.80 | +1.63 | +2.44 |
| 25-H2 | 44% | +0.74 | +0.71 | -0.02 |
| 26-H1 | 53% | +0.16 | +0.83 | +0.66 |
| 26-H2 | 57% | +0.98 | +2.32 | +1.34 |

## 3. Mechanism (all fills): how thin losses happen

| side | stop width avg % | reach avg % (best move before exit) | stopped after reaching ≥ +3 % | biggest-5-pair share of this side’s loss |
|---|---|---|---|---|
| thin | 4.32 | 5.88 | 27% | 6% |
| rest | 4.81 | 6.55 | 24% | 6% |

## 4. Continuous read (Spearman of log 24 h volume vs net %)

- DESIGN: ρ = +0.020 (shuffle p 0.638, N 553)
- HOLDOUT: ρ = -0.043 (shuffle p 0.194, N 981)

## 5. Real stop-type exit slippage by 24 h volume (master, LIVE, ALL sleeves — 0 DIST fills; a proxy for thin-coin cost; positive = cost)

| 24 h volume | N | median | p90 |
|---|---|---|---|
| < $25M | 4 | 0.22 % | 0.52 % |
| $25–42M | 9 | 0.01 % | 0.17 % |
| $42–80M | 7 | 0.04 % | 0.07 % |
| $80–200M | 11 | 0.02 % | 0.11 % |
| ≥ $200M | 18 | 0.00 % | 0.07 % |

The two live DIST fills drifted 0.61 % (NAORIS) and 0.36 % (SENT) between decision and fill (paper export 2026-11-19 12:48, `entry_book_drift_pct`).

## What it means
- **The effect is real in direction, weak in size.** By bucket, the average rises with volume: +0.34 ($15–25M) → +0.51 ($25–42M) → +1.27 → +0.95 → +2.10 (≥ $200M). Thin is worse in 6 of 7 half-years. But the rank correlation is ≈ 0 on both blocks (and slightly negative on HOLDOUT): it is a difference in how big the wins and losses are, not a steady per-trade ordering.
- **Mechanism: thin coins start the same, then follow through less.** They go +1 % in our favour just as often (83–85 % vs 81–84 %). But their best move is smaller (5.6–6.0 % vs 6.4–7.3 %), and more of them are stopped after reaching +3 % (24–29 % vs 22–25 %). A thin coin's sellers run out sooner, and it is easier to squeeze back.
- **HOLDOUT shrank the gap:** thin +0.62 (CI −0.11 … +1.43) vs rest +1.07. Thin coins still make money on average, so a filter fails the expectancy bar (ENTRY_WL §2.9: it lowers the daily compound).
- **Thin coins are now the majority:** 30–37 % of fills through 25-H1, 44 % in 25-H2, 53–57 % in 2026. A thin-coin rule now touches most of the sleeve.
- **Hidden cost:** the backtest charges every DIST fill the 0.10 % slippage floor, thin or not. Real stop-type exits (all sleeves, a proxy) on < $25M coins cost a median 0.22 % (only 4 fills), and the 2 live DIST entries drifted 0.36–0.61 %. If thin coins pay about 0.2–0.4 % more per round trip, the thin HOLDOUT average falls from +0.62 to about +0.2–0.4.

## Verdict
- **No filter.** Thin coins are positive on HOLDOUT and are now most of the sleeve.
- **Keep the observe line** and carry two items to the pre-live sizing review:
  1. a volume-scaled size (smaller on thin coins) instead of a binary cut, priced near Kelly;
  2. the real thin-coin entry / exit cost from the DIST fills' capacity and slippage stamps.
- **New batch-review read:** for each DIST fill, record entry drift, fill-vs-book and exit slippage, split thin vs rest, and compare with the 0.10 % the backtest assumes.
