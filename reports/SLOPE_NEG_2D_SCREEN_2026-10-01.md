# BTC 1h slope < 0 × second variable — 2D screen for momentum-LONG losers (2026-10-01)

Pool: current-stack momentum longs from the master ledger (validated) + fresh batch → **108 fills**, **39 with BTC 1h slope < 0**. Legs tested: 106 (43 variables × median/zero × both sides).

## The cohort itself

| | N | WR | avg % |
|---|---|---|---|
| slope < 0 | 39 | 77% | +0.194 |
| slope ≥ 0 | 69 | 81% | +0.275 |

## Per batch (era): the slope<0 cohort

| Era | N (slope<0) | WR | avg % | N (slope≥0) | WR | avg % |
|---|---|---|---|---|---|---|
| BASE | 15 | 100% | +0.662 | 18 | 83% | +0.255 |
| B1 | 11 | 64% | -0.084 | 5 | 100% | +0.807 |
| B2 | 1 | 0% | -0.847 | 9 | 89% | +0.307 |
| B3 | 7 | 86% | +0.298 | 9 | 78% | +0.295 |
| B5 | 1 | 100% | +0.047 | 2 | 100% | +0.655 |
| B7 | 0 | – | – | 1 | 100% | +0.100 |
| B8 | 1 | 100% | +0.098 | 2 | 100% | +0.544 |
| B9 | 0 | – | – | 2 | 50% | +0.254 |
| B10 | 0 | – | – | 1 | 100% | +0.100 |
| B12 | 1 | 0% | -0.826 | 11 | 82% | +0.346 |
| B13 | 0 | – | – | 3 | 67% | +0.145 |
| B14 | 0 | – | – | 6 | 50% | -0.403 |
| FRESH | 2 | 0% | -0.996 | 0 | nan% | +nan |

## Survivors (slope<0 ∧ leg: N≥8, WR≤50 %, avg<0, Δ≤−0.25) — **0 on the real data vs null median 0 (95th pct 0, P(null ≥ real) = 1.00)**


## Part 2 — discovery on the year backtest, POST-GATE (current stack), seeds collapsed to one row per trade; H1 Jan–Apr discover, H2 May–Sep confirm + H2 interaction

Backtest momentum longs with slope < 0: **561** unique trades. Legs tested: 120. **Confirmed candidates: 0 on the real data vs null median 0 (95th pct 0, P(null ≥ real) = 1.00)**


## Part 2 — discovery on the year backtest, PRE-GATE (raw replay — for the record only), seeds collapsed to one row per trade; H1 Jan–Apr discover, H2 May–Sep confirm + H2 interaction

Backtest momentum longs with slope < 0: **660** unique trades. Legs tested: 120. **Confirmed candidates: 0 on the real data vs null median 0 (95th pct 1, P(null ≥ real) = 1.00)**

