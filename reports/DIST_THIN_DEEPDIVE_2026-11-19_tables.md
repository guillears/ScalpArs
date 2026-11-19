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
