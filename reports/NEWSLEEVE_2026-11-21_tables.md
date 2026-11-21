# NEWSLEEVE 2026-11-21 — tables (generated)

| idea | vi | ci | pick | design_avg | design_N | null95 | ho_N | ho_days | ho_WR | ho_avg | ho_ci_lo | ho_ctrl | ho_diff_lo | kelly_floor | halves_pos | stress | passes | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H5B | 1 | 3 | ratio≥10 share≥0.65 · TP1/SL4/6h | -0.27 | 3653 | -0.32 | 5526 | 504 | +0.74 | -0.32 | -0.37 | -0.43 | +0.04 | -0.49 | 0/3 | -0.52 | 2/7 | REFUTED |
| H5S | 1 | 10 | ratio≥10 share≤0.35 · TP4/SL4/1h | -0.19 | 8252 | -0.30 | 11406 | 504 | +0.41 | -0.34 | -0.39 | -0.38 | -0.02 | -0.26 | 0/3 | -0.54 | 1/7 | REFUTED |
| H1 | 1 | 11 | |r15|≥5% dOI≤−3% · TP4/SL4/6h | +0.16 | 7567 | -0.13 | 10442 | 468 | +0.48 | -0.36 | -0.60 | -0.09 | -0.49 | -0.11 | 0/3 | -0.56 | 2/8 | REFUTED |
| H2 | 2 | 2 | |r15|≥8% dOI≤−1% · TP1/SL4/1h | -0.04 | 53 | +0.67 | 175 | 93 | +0.56 | -1.45 | -1.74 | -0.62 | -1.15 | -2.14 | 0/3 | -1.65 | 1/8 | REFUTED |
| H6L | 0 | 2 | frozen · TP3/SL6/8h | -0.06 | 648 | +1.11 | 1158 | 402 | +0.60 | -0.36 | -0.58 | -0.15 | -0.49 | -0.20 | 1/3 | -0.56 | 1/8 | REFUTED |
| H6S | 0 | 7 | frozen · TP6/SL6/24h | +1.58 | 869 | +0.64 | 598 | 291 | +0.51 | -0.23 | -0.64 | -0.26 | -0.44 | -0.11 | 1/3 | -0.43 | 2/8 | REFUTED |

## H5B

### Trigger counts (events = raw trigger bars; possibilities = one per coin per holding period)

| variant | period | possibilities | 15-min windows | days | coins |
|---|---|---|---|---|---|
| ratio≥5 share≥0.60 | DESIGN | 41604 | 27231 | 731 | 472 |
| ratio≥5 share≥0.60 | HOLDOUT | 39366 | 24229 | 504 | 615 |
| ratio≥10 share≥0.65 | DESIGN | 3653 | 3455 | 676 | 405 |
| ratio≥10 share≥0.65 | HOLDOUT | 5526 | 5073 | 504 | 589 |

Raw trigger bars (events) for variant 1: 249446.

### DESIGN grid (avg net % per possibility; control = random entry same coin / day / direction / exit)

| variant | exit | N | days | WR | avg | control avg |
|---|---|---|---|---|---|---|
| ratio≥5 share≥0.60 | TP1/SL2/1h | 41604 | 731 | 50% | -0.32 | -0.36 |
| ratio≥5 share≥0.60 | TP1/SL2/6h | 41604 | 731 | 63% | -0.31 | -0.43 |
| ratio≥5 share≥0.60 | TP1/SL4/1h | 41604 | 731 | 53% | -0.31 | -0.36 |
| ratio≥5 share≥0.60 | TP1/SL4/6h | 41604 | 731 | 72% | -0.29 | -0.48 |
| ratio≥5 share≥0.60 | TP2/SL2/1h | 41604 | 731 | 41% | -0.34 | -0.38 |
| ratio≥5 share≥0.60 | TP2/SL2/6h | 41604 | 731 | 48% | -0.32 | -0.49 |
| ratio≥5 share≥0.60 | TP2/SL4/1h | 41604 | 731 | 44% | -0.33 | -0.37 |
| ratio≥5 share≥0.60 | TP2/SL4/6h | 41604 | 731 | 56% | -0.30 | -0.56 |
| ratio≥5 share≥0.60 | TP4/SL2/1h | 41604 | 731 | 40% | -0.35 | -0.38 |
| ratio≥5 share≥0.60 | TP4/SL2/6h | 41604 | 731 | 39% | -0.35 | -0.54 |
| ratio≥5 share≥0.60 | TP4/SL4/1h | 41604 | 731 | 42% | -0.34 | -0.38 |
| ratio≥5 share≥0.60 | TP4/SL4/6h | 41604 | 731 | 47% | -0.34 | -0.62 |
| ratio≥10 share≥0.65 | TP1/SL2/1h | 3653 | 676 | 53% | -0.32 | -0.38 |
| ratio≥10 share≥0.65 | TP1/SL2/6h | 3653 | 676 | 64% | -0.30 | -0.44 |
| ratio≥10 share≥0.65 | TP1/SL4/1h | 3653 | 676 | 55% | -0.32 | -0.37 |
| ratio≥10 share≥0.65 | TP1/SL4/6h | 3653 | 676 | 73% | -0.27 | -0.45 |
| ratio≥10 share≥0.65 | TP2/SL2/1h | 3653 | 676 | 43% | -0.34 | -0.39 |
| ratio≥10 share≥0.65 | TP2/SL2/6h | 3653 | 676 | 48% | -0.31 | -0.48 |
| ratio≥10 share≥0.65 | TP2/SL4/1h | 3653 | 676 | 45% | -0.35 | -0.38 |
| ratio≥10 share≥0.65 | TP2/SL4/6h | 3653 | 676 | 57% | -0.27 | -0.50 |
| ratio≥10 share≥0.65 | TP4/SL2/1h | 3653 | 676 | 41% | -0.36 | -0.41 |
| ratio≥10 share≥0.65 | TP4/SL2/6h | 3653 | 676 | 39% | -0.36 | -0.54 |
| ratio≥10 share≥0.65 | TP4/SL4/1h | 3653 | 676 | 43% | -0.38 | -0.40 |
| ratio≥10 share≥0.65 | TP4/SL4/6h | 3653 | 676 | 48% | -0.33 | -0.56 |

**DESIGN pick:** ratio≥10 share≥0.65 · TP1/SL4/6h → N 3653 · avg -0.27 %. Shuffled-null 95th pct of the max-cell avg: -0.32 % → PASS.

### HOLDOUT (read once at the pick)

| N | days | WR | avg win | avg loss | breakeven WR | Wilson floor | Kelly obs | Kelly floor | avg % | day-boot 90 % CI | +0.20 slip | control avg | paired Δ lower 95 % | top pair (share) | top day share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5526 | 504 | 74% | +0.74 | -3.35 | 82% | 73% | -0.44 | -0.49 | **-0.32** | -0.37 … -0.28 | -0.52 | -0.43 | +0.04 | NAORISUSDT (—) | — |

| half-year | N | days | WR | avg | control | day-boot lower 95 % |
|---|---|---|---|---|---|---|
| 25-H2 | 1812 | 184 | 73% | -0.34 | -0.43 | -0.43 |
| 26-H1 | 2155 | 181 | 75% | -0.29 | -0.44 | -0.36 |
| 26-H2 | 1559 | 139 | 73% | -0.35 | -0.41 | -0.44 |

| BTC daily regime | period | N | days | WR | avg | lower 95 % |
|---|---|---|---|---|---|---|
| BTC bull (EMA50>EMA200) | DESIGN | 3574 | 667 | +73 % | -0.28 | -0.33 |
| BTC bull (EMA50>EMA200) | HOLDOUT | 1906 | 205 | +74 % | -0.33 | -0.42 |
| BTC bear (EMA50<EMA200) | DESIGN | 79 | 9 | +73 % | +0.01 | -0.28 |
| BTC bear (EMA50<EMA200) | HOLDOUT | 3620 | 299 | +74 % | -0.32 | -0.38 |

| 24 h volume at trigger (HOLDOUT) | N | WR | avg | control |
|---|---|---|---|---|
| $10–50M | 4106 | +74 % | -0.36 | -0.46 |
| $50–200M | 1146 | +74 % | -0.21 | -0.32 |
| ≥ $200M | 274 | +72 % | -0.22 | -0.34 |

| gate | pass |
|---|---|
| 1 Kelly>0 at Wilson floor | ✗ |
| 2 day-bootstrap lower 95% > 0 | ✗ |
| 3 ≥2/3 half-years positive | ✗ |
| 4 survives +0.20% slip | ✗ |
| 5 no pair/day ≥50% of profit | ✗ |
| 6 beats random control (lower 95% > 0) | ✓ |
| 7 DESIGN pick beat shuffled null | ✓ |

**Verdict (mechanical): REFUTED**

## H5S

### Trigger counts (events = raw trigger bars; possibilities = one per coin per holding period)

| variant | period | possibilities | 15-min windows | days | coins |
|---|---|---|---|---|---|
| ratio≥5 share≤0.40 | DESIGN | 70778 | 29614 | 731 | 472 |
| ratio≥5 share≤0.40 | HOLDOUT | 61058 | 27562 | 504 | 615 |
| ratio≥10 share≤0.35 | DESIGN | 8252 | 6040 | 712 | 437 |
| ratio≥10 share≤0.35 | HOLDOUT | 11406 | 8674 | 504 | 611 |

Raw trigger bars (events) for variant 1: 508969.

### DESIGN grid (avg net % per possibility; control = random entry same coin / day / direction / exit)

| variant | exit | N | days | WR | avg | control avg |
|---|---|---|---|---|---|---|
| ratio≥5 share≤0.40 | TP1/SL2/1h | 70778 | 731 | 51% | -0.31 | -0.32 |
| ratio≥5 share≤0.40 | TP1/SL2/6h | 70778 | 731 | 62% | -0.33 | -0.38 |
| ratio≥5 share≤0.40 | TP1/SL4/1h | 70778 | 731 | 54% | -0.28 | -0.32 |
| ratio≥5 share≤0.40 | TP1/SL4/6h | 70778 | 731 | 71% | -0.29 | -0.42 |
| ratio≥5 share≤0.40 | TP2/SL2/1h | 70778 | 731 | 42% | -0.32 | -0.33 |
| ratio≥5 share≤0.40 | TP2/SL2/6h | 70778 | 731 | 46% | -0.37 | -0.42 |
| ratio≥5 share≤0.40 | TP2/SL4/1h | 70778 | 731 | 44% | -0.28 | -0.32 |
| ratio≥5 share≤0.40 | TP2/SL4/6h | 70778 | 731 | 55% | -0.31 | -0.47 |
| ratio≥5 share≤0.40 | TP4/SL2/1h | 70778 | 731 | 40% | -0.31 | -0.33 |
| ratio≥5 share≤0.40 | TP4/SL2/6h | 70778 | 731 | 37% | -0.38 | -0.41 |
| ratio≥5 share≤0.40 | TP4/SL4/1h | 70778 | 731 | 42% | -0.27 | -0.32 |
| ratio≥5 share≤0.40 | TP4/SL4/6h | 70778 | 731 | 45% | -0.32 | -0.46 |
| ratio≥10 share≤0.35 | TP1/SL2/1h | 8252 | 712 | 53% | -0.29 | -0.36 |
| ratio≥10 share≤0.35 | TP1/SL2/6h | 8252 | 712 | 65% | -0.29 | -0.43 |
| ratio≥10 share≤0.35 | TP1/SL4/1h | 8252 | 712 | 56% | -0.25 | -0.34 |
| ratio≥10 share≤0.35 | TP1/SL4/6h | 8252 | 712 | 73% | -0.26 | -0.49 |
| ratio≥10 share≤0.35 | TP2/SL2/1h | 8252 | 712 | 44% | -0.26 | -0.37 |
| ratio≥10 share≤0.35 | TP2/SL2/6h | 8252 | 712 | 49% | -0.28 | -0.47 |
| ratio≥10 share≤0.35 | TP2/SL4/1h | 8252 | 712 | 47% | -0.22 | -0.35 |
| ratio≥10 share≤0.35 | TP2/SL4/6h | 8252 | 712 | 57% | -0.23 | -0.55 |
| ratio≥10 share≤0.35 | TP4/SL2/1h | 8252 | 712 | 41% | -0.24 | -0.36 |
| ratio≥10 share≤0.35 | TP4/SL2/6h | 8252 | 712 | 39% | -0.27 | -0.47 |
| ratio≥10 share≤0.35 | TP4/SL4/1h | 8252 | 712 | 44% | -0.19 | -0.34 |
| ratio≥10 share≤0.35 | TP4/SL4/6h | 8252 | 712 | 47% | -0.24 | -0.55 |

**DESIGN pick:** ratio≥10 share≤0.35 · TP4/SL4/1h → N 8252 · avg -0.19 %. Shuffled-null 95th pct of the max-cell avg: -0.30 % → PASS.

### HOLDOUT (read once at the pick)

| N | days | WR | avg win | avg loss | breakeven WR | Wilson floor | Kelly obs | Kelly floor | avg % | day-boot 90 % CI | +0.20 slip | control avg | paired Δ lower 95 % | top pair (share) | top day share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11406 | 504 | 41% | +1.36 | -1.52 | 53% | 40% | -0.25 | -0.26 | **-0.34** | -0.39 … -0.29 | -0.54 | -0.38 | -0.02 | BROCCOLI714USDT (—) | — |

| half-year | N | days | WR | avg | control | day-boot lower 95 % |
|---|---|---|---|---|---|---|
| 25-H2 | 4024 | 184 | 42% | -0.31 | -0.40 | -0.42 |
| 26-H1 | 4099 | 181 | 40% | -0.38 | -0.35 | -0.46 |
| 26-H2 | 3283 | 139 | 42% | -0.32 | -0.39 | -0.42 |

| BTC daily regime | period | N | days | WR | avg | lower 95 % |
|---|---|---|---|---|---|---|
| BTC bull (EMA50>EMA200) | DESIGN | 8127 | 703 | +44 % | -0.20 | -0.29 |
| BTC bull (EMA50>EMA200) | HOLDOUT | 4603 | 205 | +42 % | -0.31 | -0.41 |
| BTC bear (EMA50<EMA200) | DESIGN | 125 | 9 | +59 % | +0.34 | +0.02 |
| BTC bear (EMA50<EMA200) | HOLDOUT | 6803 | 299 | +40 % | -0.36 | -0.42 |

| 24 h volume at trigger (HOLDOUT) | N | WR | avg | control |
|---|---|---|---|---|
| $10–50M | 8477 | +40 % | -0.38 | -0.41 |
| $50–200M | 2396 | +44 % | -0.24 | -0.30 |
| ≥ $200M | 533 | +48 % | -0.11 | -0.22 |

| gate | pass |
|---|---|
| 1 Kelly>0 at Wilson floor | ✗ |
| 2 day-bootstrap lower 95% > 0 | ✗ |
| 3 ≥2/3 half-years positive | ✗ |
| 4 survives +0.20% slip | ✗ |
| 5 no pair/day ≥50% of profit | ✗ |
| 6 beats random control (lower 95% > 0) | ✗ |
| 7 DESIGN pick beat shuffled null | ✓ |

**Verdict (mechanical): REFUTED**

## H1

### Trigger counts (events = raw trigger bars; possibilities = one per coin per holding period)

| variant | period | possibilities | 15-min windows | days | coins |
|---|---|---|---|---|---|
| |r15|≥5% dOI≤−1% | DESIGN | 7741 | 3260 | 594 | 460 |
| |r15|≥5% dOI≤−1% | HOLDOUT | 10665 | 7730 | 477 | 598 |
| |r15|≥5% dOI≤−3% | DESIGN | 7567 | 3141 | 587 | 460 |
| |r15|≥5% dOI≤−3% | HOLDOUT | 10442 | 7568 | 468 | 597 |
| |r15|≥8% dOI≤−1% | DESIGN | 2869 | 939 | 358 | 431 |
| |r15|≥8% dOI≤−1% | HOLDOUT | 4502 | 3512 | 439 | 562 |
| |r15|≥8% dOI≤−3% | DESIGN | 2846 | 923 | 355 | 430 |
| |r15|≥8% dOI≤−3% | HOLDOUT | 4457 | 3468 | 433 | 561 |

### DESIGN grid (avg net % per possibility; control = random entry same coin / day / direction / exit)

| variant | exit | N | days | WR | avg | control avg |
|---|---|---|---|---|---|---|
| |r15|≥5% dOI≤−1% | TP1/SL2/1h | 7741 | 594 | 61% | -0.43 | -0.29 |
| |r15|≥5% dOI≤−1% | TP1/SL2/6h | 7741 | 594 | 61% | -0.43 | -0.36 |
| |r15|≥5% dOI≤−1% | TP1/SL4/1h | 7741 | 594 | 76% | -0.33 | -0.25 |
| |r15|≥5% dOI≤−1% | TP1/SL4/6h | 7741 | 594 | 78% | -0.33 | -0.38 |
| |r15|≥5% dOI≤−1% | TP2/SL2/1h | 7741 | 594 | 53% | -0.15 | -0.27 |
| |r15|≥5% dOI≤−1% | TP2/SL2/6h | 7741 | 594 | 53% | -0.14 | -0.32 |
| |r15|≥5% dOI≤−1% | TP2/SL4/1h | 7741 | 594 | 66% | -0.02 | -0.23 |
| |r15|≥5% dOI≤−1% | TP2/SL4/6h | 7741 | 594 | 70% | -0.01 | -0.34 |
| |r15|≥5% dOI≤−1% | TP4/SL2/1h | 7741 | 594 | 42% | -0.10 | -0.24 |
| |r15|≥5% dOI≤−1% | TP4/SL2/6h | 7741 | 594 | 39% | -0.02 | -0.25 |
| |r15|≥5% dOI≤−1% | TP4/SL4/1h | 7741 | 594 | 54% | +0.02 | -0.18 |
| |r15|≥5% dOI≤−1% | TP4/SL4/6h | 7741 | 594 | 55% | +0.16 | -0.25 |
| |r15|≥5% dOI≤−3% | TP1/SL2/1h | 7567 | 587 | 61% | -0.43 | -0.29 |
| |r15|≥5% dOI≤−3% | TP1/SL2/6h | 7567 | 587 | 61% | -0.43 | -0.36 |
| |r15|≥5% dOI≤−3% | TP1/SL4/1h | 7567 | 587 | 76% | -0.33 | -0.25 |
| |r15|≥5% dOI≤−3% | TP1/SL4/6h | 7567 | 587 | 78% | -0.33 | -0.38 |
| |r15|≥5% dOI≤−3% | TP2/SL2/1h | 7567 | 587 | 53% | -0.15 | -0.28 |
| |r15|≥5% dOI≤−3% | TP2/SL2/6h | 7567 | 587 | 53% | -0.14 | -0.33 |
| |r15|≥5% dOI≤−3% | TP2/SL4/1h | 7567 | 587 | 66% | -0.03 | -0.23 |
| |r15|≥5% dOI≤−3% | TP2/SL4/6h | 7567 | 587 | 70% | -0.01 | -0.35 |
| |r15|≥5% dOI≤−3% | TP4/SL2/1h | 7567 | 587 | 42% | -0.10 | -0.24 |
| |r15|≥5% dOI≤−3% | TP4/SL2/6h | 7567 | 587 | 39% | -0.02 | -0.26 |
| |r15|≥5% dOI≤−3% | TP4/SL4/1h | 7567 | 587 | 55% | +0.02 | -0.18 |
| |r15|≥5% dOI≤−3% | TP4/SL4/6h | 7567 | 587 | 55% | +0.16 | -0.26 |
| |r15|≥8% dOI≤−1% | TP1/SL2/1h | 2869 | 358 | 46% | -0.89 | -0.31 |
| |r15|≥8% dOI≤−1% | TP1/SL2/6h | 2869 | 358 | 47% | -0.89 | -0.45 |
| |r15|≥8% dOI≤−1% | TP1/SL4/1h | 2869 | 358 | 66% | -0.96 | -0.27 |
| |r15|≥8% dOI≤−1% | TP1/SL4/6h | 2869 | 358 | 66% | -0.96 | -0.51 |
| |r15|≥8% dOI≤−1% | TP2/SL2/1h | 2869 | 358 | 43% | -0.56 | -0.28 |
| |r15|≥8% dOI≤−1% | TP2/SL2/6h | 2869 | 358 | 43% | -0.56 | -0.42 |
| |r15|≥8% dOI≤−1% | TP2/SL4/1h | 2869 | 358 | 62% | -0.53 | -0.23 |
| |r15|≥8% dOI≤−1% | TP2/SL4/6h | 2869 | 358 | 62% | -0.53 | -0.47 |
| |r15|≥8% dOI≤−1% | TP4/SL2/1h | 2869 | 358 | 38% | -0.20 | -0.25 |
| |r15|≥8% dOI≤−1% | TP4/SL2/6h | 2869 | 358 | 36% | -0.18 | -0.31 |
| |r15|≥8% dOI≤−1% | TP4/SL4/1h | 2869 | 358 | 55% | -0.01 | -0.20 |
| |r15|≥8% dOI≤−1% | TP4/SL4/6h | 2869 | 358 | 55% | +0.06 | -0.38 |
| |r15|≥8% dOI≤−3% | TP1/SL2/1h | 2846 | 355 | 46% | -0.89 | -0.31 |
| |r15|≥8% dOI≤−3% | TP1/SL2/6h | 2846 | 355 | 47% | -0.89 | -0.45 |
| |r15|≥8% dOI≤−3% | TP1/SL4/1h | 2846 | 355 | 66% | -0.96 | -0.26 |
| |r15|≥8% dOI≤−3% | TP1/SL4/6h | 2846 | 355 | 66% | -0.96 | -0.51 |
| |r15|≥8% dOI≤−3% | TP2/SL2/1h | 2846 | 355 | 43% | -0.56 | -0.28 |
| |r15|≥8% dOI≤−3% | TP2/SL2/6h | 2846 | 355 | 43% | -0.56 | -0.42 |
| |r15|≥8% dOI≤−3% | TP2/SL4/1h | 2846 | 355 | 62% | -0.53 | -0.22 |
| |r15|≥8% dOI≤−3% | TP2/SL4/6h | 2846 | 355 | 62% | -0.53 | -0.46 |
| |r15|≥8% dOI≤−3% | TP4/SL2/1h | 2846 | 355 | 38% | -0.20 | -0.24 |
| |r15|≥8% dOI≤−3% | TP4/SL2/6h | 2846 | 355 | 36% | -0.19 | -0.31 |
| |r15|≥8% dOI≤−3% | TP4/SL4/1h | 2846 | 355 | 55% | -0.02 | -0.20 |
| |r15|≥8% dOI≤−3% | TP4/SL4/6h | 2846 | 355 | 54% | +0.05 | -0.38 |

**DESIGN pick:** |r15|≥5% dOI≤−3% · TP4/SL4/6h → N 7567 · avg +0.16 %. Shuffled-null 95th pct of the max-cell avg: -0.13 % → PASS.

### HOLDOUT (read once at the pick)

| N | days | WR | avg win | avg loss | breakeven WR | Wilson floor | Kelly obs | Kelly floor | avg % | day-boot 90 % CI | +0.20 slip | control avg | paired Δ lower 95 % | top pair (share) | top day share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10442 | 468 | 48% | +3.65 | -4.09 | 53% | 47% | -0.10 | -0.11 | **-0.36** | -0.60 … -0.13 | -0.56 | -0.09 | -0.49 | NAORISUSDT (—) | — |

| half-year | N | days | WR | avg | control | day-boot lower 95 % |
|---|---|---|---|---|---|---|
| 25-H2 | 5495 | 184 | 48% | -0.35 | -0.18 | -0.75 |
| 26-H1 | 3524 | 173 | 49% | -0.30 | -0.02 | -0.62 |
| 26-H2 | 1423 | 111 | 46% | -0.54 | +0.08 | -0.72 |

| BTC daily regime | period | N | days | WR | avg | lower 95 % |
|---|---|---|---|---|---|---|
| BTC bull (EMA50>EMA200) | DESIGN | 7542 | 580 | +55 % | +0.16 | -0.35 |
| BTC bull (EMA50>EMA200) | HOLDOUT | 4508 | 187 | +47 % | -0.44 | -0.89 |
| BTC bear (EMA50<EMA200) | DESIGN | 25 | 7 | +60 % | +0.77 | -2.33 |
| BTC bear (EMA50<EMA200) | HOLDOUT | 5934 | 281 | +49 % | -0.30 | -0.51 |

| 24 h volume at trigger (HOLDOUT) | N | WR | avg | control |
|---|---|---|---|---|
| $10–50M | 4880 | +48 % | -0.40 | -0.32 |
| $50–200M | 3036 | +48 % | -0.36 | +0.08 |
| ≥ $200M | 2526 | +48 % | -0.28 | +0.14 |

| gate | pass |
|---|---|
| 1 Kelly>0 at Wilson floor | ✗ |
| 2 day-bootstrap lower 95% > 0 | ✗ |
| 3 ≥2/3 half-years positive | ✗ |
| 4 survives +0.20% slip | ✗ |
| 5 no pair/day ≥50% of profit | ✗ |
| 6 beats random control (lower 95% > 0) | ✗ |
| 7 DESIGN pick beat shuffled null | ✓ |
| 8 market-wide: ≥30 HOLDOUT days | ✓ |

**Verdict (mechanical): REFUTED**

## H2

### Trigger counts (events = raw trigger bars; possibilities = one per coin per holding period)

| variant | period | possibilities | 15-min windows | days | coins |
|---|---|---|---|---|---|
| |r15|≥5% dOI≤−1% | DESIGN | 224 | 133 | 108 | 158 |
| |r15|≥5% dOI≤−1% | HOLDOUT | 360 | 255 | 184 | 235 |
| |r15|≥5% dOI≤−3% | DESIGN | 118 | 67 | 59 | 91 |
| |r15|≥5% dOI≤−3% | HOLDOUT | 217 | 143 | 112 | 167 |
| |r15|≥8% dOI≤−1% | DESIGN | 53 | 29 | 26 | 51 |
| |r15|≥8% dOI≤−1% | HOLDOUT | 175 | 109 | 93 | 140 |
| |r15|≥8% dOI≤−3% | DESIGN | 28 | 17 | 16 | 26 |
| |r15|≥8% dOI≤−3% | HOLDOUT | 122 | 73 | 64 | 105 |

### DESIGN grid (avg net % per possibility; control = random entry same coin / day / direction / exit)

| variant | exit | N | days | WR | avg | control avg |
|---|---|---|---|---|---|---|
| |r15|≥5% dOI≤−1% | TP1/SL2/1h | 224 | 108 | 58% | -0.47 | -0.35 |
| |r15|≥5% dOI≤−1% | TP1/SL2/6h | 224 | 108 | 59% | -0.47 | -0.34 |
| |r15|≥5% dOI≤−1% | TP1/SL4/1h | 224 | 108 | 72% | -0.43 | -0.41 |
| |r15|≥5% dOI≤−1% | TP1/SL4/6h | 224 | 108 | 74% | -0.51 | -0.40 |
| |r15|≥5% dOI≤−1% | TP2/SL2/1h | 224 | 108 | 38% | -0.69 | -0.32 |
| |r15|≥5% dOI≤−1% | TP2/SL2/6h | 224 | 108 | 40% | -0.69 | -0.37 |
| |r15|≥5% dOI≤−1% | TP2/SL4/1h | 224 | 108 | 51% | -0.65 | -0.39 |
| |r15|≥5% dOI≤−1% | TP2/SL4/6h | 224 | 108 | 57% | -0.69 | -0.43 |
| |r15|≥5% dOI≤−1% | TP4/SL2/1h | 224 | 108 | 23% | -0.99 | -0.27 |
| |r15|≥5% dOI≤−1% | TP4/SL2/6h | 224 | 108 | 21% | -1.09 | -0.29 |
| |r15|≥5% dOI≤−1% | TP4/SL4/1h | 224 | 108 | 34% | -0.99 | -0.30 |
| |r15|≥5% dOI≤−1% | TP4/SL4/6h | 224 | 108 | 36% | -1.22 | -0.31 |
| |r15|≥5% dOI≤−3% | TP1/SL2/1h | 118 | 59 | 56% | -0.57 | -0.39 |
| |r15|≥5% dOI≤−3% | TP1/SL2/6h | 118 | 59 | 56% | -0.57 | -0.34 |
| |r15|≥5% dOI≤−3% | TP1/SL4/1h | 118 | 59 | 68% | -0.64 | -0.47 |
| |r15|≥5% dOI≤−3% | TP1/SL4/6h | 118 | 59 | 69% | -0.76 | -0.55 |
| |r15|≥5% dOI≤−3% | TP2/SL2/1h | 118 | 59 | 36% | -0.83 | -0.39 |
| |r15|≥5% dOI≤−3% | TP2/SL2/6h | 118 | 59 | 36% | -0.86 | -0.33 |
| |r15|≥5% dOI≤−3% | TP2/SL4/1h | 118 | 59 | 47% | -0.88 | -0.47 |
| |r15|≥5% dOI≤−3% | TP2/SL4/6h | 118 | 59 | 51% | -1.04 | -0.62 |
| |r15|≥5% dOI≤−3% | TP4/SL2/1h | 118 | 59 | 20% | -1.07 | -0.27 |
| |r15|≥5% dOI≤−3% | TP4/SL2/6h | 118 | 59 | 19% | -1.16 | -0.05 |
| |r15|≥5% dOI≤−3% | TP4/SL4/1h | 118 | 59 | 31% | -1.12 | -0.34 |
| |r15|≥5% dOI≤−3% | TP4/SL4/6h | 118 | 59 | 33% | -1.41 | -0.27 |
| |r15|≥8% dOI≤−1% | TP1/SL2/1h | 53 | 26 | 66% | -0.24 | -0.58 |
| |r15|≥8% dOI≤−1% | TP1/SL2/6h | 53 | 26 | 66% | -0.24 | -0.78 |
| |r15|≥8% dOI≤−1% | TP1/SL4/1h | 53 | 26 | 83% | -0.04 | -0.55 |
| |r15|≥8% dOI≤−1% | TP1/SL4/6h | 53 | 26 | 81% | -0.12 | -0.87 |
| |r15|≥8% dOI≤−1% | TP2/SL2/1h | 53 | 26 | 49% | -0.29 | -0.61 |
| |r15|≥8% dOI≤−1% | TP2/SL2/6h | 53 | 26 | 49% | -0.29 | -0.76 |
| |r15|≥8% dOI≤−1% | TP2/SL4/1h | 53 | 26 | 64% | -0.10 | -0.40 |
| |r15|≥8% dOI≤−1% | TP2/SL4/6h | 53 | 26 | 66% | -0.13 | -0.51 |
| |r15|≥8% dOI≤−1% | TP4/SL2/1h | 53 | 26 | 25% | -0.88 | -0.41 |
| |r15|≥8% dOI≤−1% | TP4/SL2/6h | 53 | 26 | 23% | -0.98 | -0.23 |
| |r15|≥8% dOI≤−1% | TP4/SL4/1h | 53 | 26 | 40% | -0.66 | -0.41 |
| |r15|≥8% dOI≤−1% | TP4/SL4/6h | 53 | 26 | 42% | -0.86 | -0.31 |
| |r15|≥8% dOI≤−3% | TP1/SL2/1h | 28 | 16 | 64% | -0.30 | -0.43 |
| |r15|≥8% dOI≤−3% | TP1/SL2/6h | 28 | 16 | 64% | -0.30 | -0.45 |
| |r15|≥8% dOI≤−3% | TP1/SL4/1h | 28 | 16 | 89% | +0.24 | -0.30 |
| |r15|≥8% dOI≤−3% | TP1/SL4/6h | 28 | 16 | 86% | +0.08 | -0.69 |
| |r15|≥8% dOI≤−3% | TP2/SL2/1h | 28 | 16 | 50% | -0.25 | -0.52 |
| |r15|≥8% dOI≤−3% | TP2/SL2/6h | 28 | 16 | 50% | -0.25 | -0.39 |
| |r15|≥8% dOI≤−3% | TP2/SL4/1h | 28 | 16 | 71% | +0.22 | -0.22 |
| |r15|≥8% dOI≤−3% | TP2/SL4/6h | 28 | 16 | 71% | +0.06 | -0.22 |
| |r15|≥8% dOI≤−3% | TP4/SL2/1h | 28 | 16 | 25% | -0.80 | -0.50 |
| |r15|≥8% dOI≤−3% | TP4/SL2/6h | 28 | 16 | 25% | -0.80 | -0.04 |
| |r15|≥8% dOI≤−3% | TP4/SL4/1h | 28 | 16 | 46% | -0.19 | -0.15 |
| |r15|≥8% dOI≤−3% | TP4/SL4/6h | 28 | 16 | 54% | -0.01 | +0.15 |

**DESIGN pick:** |r15|≥8% dOI≤−1% · TP1/SL4/1h → N 53 · avg -0.04 %. Shuffled-null 95th pct of the max-cell avg: +0.67 % → FAIL.

### HOLDOUT (read once at the pick)

| N | days | WR | avg win | avg loss | breakeven WR | Wilson floor | Kelly obs | Kelly floor | avg % | day-boot 90 % CI | +0.20 slip | control avg | paired Δ lower 95 % | top pair (share) | top day share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 175 | 93 | 56% | +0.83 | -4.36 | 84% | 50% | -1.75 | -2.14 | **-1.45** | -1.74 … -0.91 | -1.65 | -0.62 | -1.15 | BANKUSDT (—) | — |

| half-year | N | days | WR | avg | control | day-boot lower 95 % |
|---|---|---|---|---|---|---|
| 25-H2 | 120 | 45 | 56% | -1.48 | -0.66 | -1.79 |
| 26-H1 | 34 | 29 | 56% | -1.42 | -0.87 | -2.16 |
| 26-H2 | 21 | 19 | 57% | -1.34 | +0.05 | -2.21 |

| BTC daily regime | period | N | days | WR | avg | lower 95 % |
|---|---|---|---|---|---|---|
| BTC bull (EMA50>EMA200) | DESIGN | 53 | 26 | +83 % | -0.04 | -0.81 |
| BTC bull (EMA50>EMA200) | HOLDOUT | 105 | 35 | +52 % | -1.66 | -1.91 |
| BTC bear (EMA50<EMA200) | DESIGN | 0 | 0 | — % | — | — |
| BTC bear (EMA50<EMA200) | HOLDOUT | 70 | 58 | +61 % | -1.13 | -1.65 |

| 24 h volume at trigger (HOLDOUT) | N | WR | avg | control |
|---|---|---|---|---|
| $10–50M | 64 | +44 % | -2.21 | -0.91 |
| $50–200M | 40 | +55 % | -1.48 | -0.30 |
| ≥ $200M | 71 | +68 % | -0.75 | -0.53 |

| gate | pass |
|---|---|
| 1 Kelly>0 at Wilson floor | ✗ |
| 2 day-bootstrap lower 95% > 0 | ✗ |
| 3 ≥2/3 half-years positive | ✗ |
| 4 survives +0.20% slip | ✗ |
| 5 no pair/day ≥50% of profit | ✗ |
| 6 beats random control (lower 95% > 0) | ✗ |
| 7 DESIGN pick beat shuffled null | ✗ |
| 8 market-wide: ≥30 HOLDOUT days | ✓ |

**Verdict (mechanical): REFUTED**

## H6L

### Trigger counts (events = raw trigger bars; possibilities = one per coin per holding period)

| variant | period | possibilities | 15-min windows | days | coins |
|---|---|---|---|---|---|
| frozen | DESIGN | 648 | 529 | 352 | 177 |
| frozen | HOLDOUT | 1158 | 785 | 402 | 310 |

Raw trigger bars (events) for variant 1: 3726.

### DESIGN grid (avg net % per possibility; control = random entry same coin / day / direction / exit)

| variant | exit | N | days | WR | avg | control avg |
|---|---|---|---|---|---|---|
| frozen | TP3/SL3/8h | 648 | 352 | 51% | -0.09 | +0.03 |
| frozen | TP3/SL3/24h | 648 | 352 | 50% | -0.13 | +0.05 |
| frozen | TP3/SL6/8h | 648 | 352 | 62% | -0.06 | +0.08 |
| frozen | TP3/SL6/24h | 648 | 352 | 65% | -0.15 | +0.06 |
| frozen | TP6/SL3/8h | 648 | 352 | 35% | -0.23 | +0.33 |
| frozen | TP6/SL3/24h | 648 | 352 | 33% | -0.27 | +0.32 |
| frozen | TP6/SL6/8h | 648 | 352 | 46% | -0.14 | +0.49 |
| frozen | TP6/SL6/24h | 648 | 352 | 48% | -0.20 | +0.33 |

**DESIGN pick:** frozen · TP3/SL6/8h → N 648 · avg -0.06 %. Shuffled-null 95th pct of the max-cell avg: +1.11 % → FAIL.

### HOLDOUT (read once at the pick)

| N | days | WR | avg win | avg loss | breakeven WR | Wilson floor | Kelly obs | Kelly floor | avg % | day-boot 90 % CI | +0.20 slip | control avg | paired Δ lower 95 % | top pair (share) | top day share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1158 | 402 | 60% | +2.81 | -5.12 | 65% | 58% | -0.13 | -0.20 | **-0.36** | -0.58 … -0.15 | -0.56 | -0.15 | -0.49 | TLMUSDT (—) | — |

| half-year | N | days | WR | avg | control | day-boot lower 95 % |
|---|---|---|---|---|---|---|
| 25-H2 | 369 | 144 | 59% | -0.51 | +0.11 | -0.90 |
| 26-H1 | 513 | 150 | 57% | -0.64 | -0.37 | -0.96 |
| 26-H2 | 276 | 108 | 67% | +0.35 | -0.10 | -0.00 |

| BTC daily regime | period | N | days | WR | avg | lower 95 % |
|---|---|---|---|---|---|---|
| BTC bull (EMA50>EMA200) | DESIGN | 641 | 347 | +61 % | -0.08 | -0.34 |
| BTC bull (EMA50>EMA200) | HOLDOUT | 359 | 150 | +60 % | -0.35 | -0.74 |
| BTC bear (EMA50<EMA200) | DESIGN | 7 | 5 | +86 % | +1.69 | +1.16 |
| BTC bear (EMA50<EMA200) | HOLDOUT | 799 | 252 | +60 % | -0.37 | -0.62 |

| 24 h volume at trigger (HOLDOUT) | N | WR | avg | control |
|---|---|---|---|---|
| $10–50M | 356 | +56 % | -0.50 | -0.24 |
| $50–200M | 463 | +62 % | -0.16 | -0.16 |
| ≥ $200M | 339 | +60 % | -0.50 | -0.06 |

| gate | pass |
|---|---|
| 1 Kelly>0 at Wilson floor | ✗ |
| 2 day-bootstrap lower 95% > 0 | ✗ |
| 3 ≥2/3 half-years positive | ✗ |
| 4 survives +0.20% slip | ✗ |
| 5 no pair/day ≥50% of profit | ✗ |
| 6 beats random control (lower 95% > 0) | ✗ |
| 7 DESIGN pick beat shuffled null | ✗ |
| 8 market-wide: ≥30 HOLDOUT days | ✓ |

**Verdict (mechanical): REFUTED**

## H6S

### Trigger counts (events = raw trigger bars; possibilities = one per coin per holding period)

| variant | period | possibilities | 15-min windows | days | coins |
|---|---|---|---|---|---|
| frozen | DESIGN | 869 | 260 | 168 | 262 |
| frozen | HOLDOUT | 598 | 439 | 291 | 136 |

Raw trigger bars (events) for variant 1: 2374.

### DESIGN grid (avg net % per possibility; control = random entry same coin / day / direction / exit)

| variant | exit | N | days | WR | avg | control avg |
|---|---|---|---|---|---|---|
| frozen | TP3/SL3/8h | 869 | 168 | 49% | -0.29 | -0.20 |
| frozen | TP3/SL3/24h | 869 | 168 | 51% | -0.14 | -0.15 |
| frozen | TP3/SL6/8h | 869 | 168 | 63% | +0.12 | -0.17 |
| frozen | TP3/SL6/24h | 869 | 168 | 73% | +0.56 | +0.12 |
| frozen | TP6/SL3/8h | 869 | 168 | 44% | +0.03 | -0.04 |
| frozen | TP6/SL3/24h | 869 | 168 | 43% | +0.49 | +0.12 |
| frozen | TP6/SL6/8h | 869 | 168 | 58% | +0.53 | +0.06 |
| frozen | TP6/SL6/24h | 869 | 168 | 65% | +1.58 | +0.51 |

**DESIGN pick:** frozen · TP6/SL6/24h → N 869 · avg +1.58 %. Shuffled-null 95th pct of the max-cell avg: +0.64 % → PASS.

### HOLDOUT (read once at the pick)

| N | days | WR | avg win | avg loss | breakeven WR | Wilson floor | Kelly obs | Kelly floor | avg % | day-boot 90 % CI | +0.20 slip | control avg | paired Δ lower 95 % | top pair (share) | top day share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 598 | 291 | 51% | +5.54 | -6.19 | 53% | 47% | -0.04 | -0.11 | **-0.23** | -0.64 … +0.20 | -0.43 | -0.26 | -0.44 | BULLAUSDT (—) | — |

| half-year | N | days | WR | avg | control | day-boot lower 95 % |
|---|---|---|---|---|---|---|
| 25-H2 | 263 | 116 | 52% | -0.04 | -0.28 | -0.72 |
| 26-H1 | 246 | 115 | 48% | -0.53 | -0.40 | -1.10 |
| 26-H2 | 89 | 60 | 54% | +0.06 | +0.17 | -1.00 |

| BTC daily regime | period | N | days | WR | avg | lower 95 % |
|---|---|---|---|---|---|---|
| BTC bull (EMA50>EMA200) | DESIGN | 869 | 168 | +65 % | +1.58 | +0.70 |
| BTC bull (EMA50>EMA200) | HOLDOUT | 268 | 120 | +54 % | +0.11 | -0.53 |
| BTC bear (EMA50<EMA200) | DESIGN | 0 | 0 | — % | — | — |
| BTC bear (EMA50<EMA200) | HOLDOUT | 330 | 171 | +48 % | -0.50 | -1.01 |

| 24 h volume at trigger (HOLDOUT) | N | WR | avg | control |
|---|---|---|---|---|
| $10–50M | 301 | +52 % | -0.23 | -0.05 |
| $50–200M | 161 | +50 % | -0.20 | -0.17 |
| ≥ $200M | 136 | +49 % | -0.25 | -0.84 |

| gate | pass |
|---|---|
| 1 Kelly>0 at Wilson floor | ✗ |
| 2 day-bootstrap lower 95% > 0 | ✗ |
| 3 ≥2/3 half-years positive | ✗ |
| 4 survives +0.20% slip | ✗ |
| 5 no pair/day ≥50% of profit | ✗ |
| 6 beats random control (lower 95% > 0) | ✗ |
| 7 DESIGN pick beat shuffled null | ✓ |
| 8 market-wide: ≥30 HOLDOUT days | ✓ |

**Verdict (mechanical): REFUTED**

## H1 — the same price + volume trigger without / with the OI leg (descriptive, DESIGN | HOLDOUT, best exit of the H grid not re-picked: cell TP2/SL4/6h)

| variant | DESIGN N | DESIGN avg | HOLDOUT N | HOLDOUT avg | HOLDOUT control |
|---|---|---|---|---|---|
| |r15|≥5% (no OI leg) | 9299 | -0.14 | 14841 | -0.40 | -0.28 |
| |r15|≥5% OI known (any) | 8021 | -0.02 | 10942 | -0.37 | -0.26 |
| |r15|≥8% (no OI leg) | 3618 | -0.28 | 6306 | -0.72 | -0.20 |
| |r15|≥8% OI known (any) | 2948 | -0.52 | 4597 | -0.72 | -0.19 |
| |r15|≥5% dOI≤−1% | 7741 | -0.01 | 10665 | -0.35 | -0.25 |
| |r15|≥5% dOI≤−3% | 7567 | -0.01 | 10442 | -0.34 | -0.24 |
| |r15|≥8% dOI≤−1% | 2869 | -0.53 | 4502 | -0.71 | -0.17 |
| |r15|≥8% dOI≤−3% | 2846 | -0.53 | 4457 | -0.71 | -0.16 |

OI coverage on the raw candidates: 73% of 76917 trigger bars.

## H2 — the same price + volume trigger without / with the OI leg (descriptive, DESIGN | HOLDOUT, best exit of the H grid not re-picked: cell TP2/SL4/6h)

| variant | DESIGN N | DESIGN avg | HOLDOUT N | HOLDOUT avg | HOLDOUT control |
|---|---|---|---|---|---|
| |r15|≥5% (no OI leg) | 10299 | -0.51 | 14600 | -0.49 | -0.53 |
| |r15|≥5% OI known (any) | 9150 | -0.54 | 11024 | -0.52 | -0.57 |
| |r15|≥8% (no OI leg) | 3309 | -0.37 | 6790 | -0.78 | -0.44 |
| |r15|≥8% OI known (any) | 3014 | -0.38 | 5048 | -0.82 | -0.50 |
| |r15|≥5% dOI≤−1% | 224 | -0.69 | 360 | -0.78 | -0.28 |
| |r15|≥5% dOI≤−3% | 118 | -1.04 | 217 | -0.78 | -0.18 |
| |r15|≥8% dOI≤−1% | 53 | -0.13 | 175 | -1.21 | -0.39 |
| |r15|≥8% dOI≤−3% | 28 | +0.06 | 122 | -1.12 | -0.36 |

OI coverage on the raw candidates: 77% of 84934 trigger bars.
