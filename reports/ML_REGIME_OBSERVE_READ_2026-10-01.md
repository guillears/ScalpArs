# Momentum-LONG observe reads — burst crowding & BTC chop

Master current stack + fresh: 108 momentum longs.

## Burst crowding (master)

| Cohort | N | windows | WR | avg % | 95 % window CI |
|---|---|---|---|---|---|
| burst | 22 | 11 | 68% | +0.199 | [-0.17, +0.55] |
| alone | 86 | 86 | 83% | +0.258 | [+0.12, +0.40] |
| slope<0 ∧ burst | 8 | 4 | 75% | +0.252 | [-0.57, +0.72] |
| slope<0 ∧ alone | 31 | 31 | 77% | +0.179 | [-0.06, +0.42] |

## BTC chop eff72 ≤ 0.007 (master; eff72 = live stamp on 24 fills, validated rebuild on 84, missing on 0; unit = DAY)

| Cohort | N | days | WR | avg % | 95 % day CI |
|---|---|---|---|---|---|
| eff72 ≤ 0.007 | 13 | 6 | 46% | -0.165 | [-0.51, +1.06] |
| eff72 > 0.007 | 95 | 52 | 84% | +0.302 | [+0.19, +0.50] |

## yr3 backtest, post-gate, seeds collapsed (H1 Jan–Apr · H2 May–Sep)

| Cohort | N | windows | WR | avg % | 95 % window CI |
|---|---|---|---|---|---|
| H1 slope<0 ∧ burst | 92 | 43 | 55% | -0.035 | [-0.26, +0.10] |
| H1 slope<0 ∧ alone | 195 | 178 | 54% | -0.165 | [-0.25, -0.06] |
| H2 slope<0 ∧ burst | 73 | 37 | 52% | -0.121 | [-0.31, +0.17] |
| H2 slope<0 ∧ alone | 201 | 187 | 57% | -0.159 | [-0.26, -0.08] |

| Cohort | N | days | WR | avg % | 95 % day CI |
|---|---|---|---|---|---|
| H1 eff72 ≤ 0.007 | 60 | 21 | 50% | -0.177 | [-0.39, +0.13] |
| H1 eff72 > 0.007 | 536 | 105 | 57% | -0.087 | [-0.16, +0.02] |
| H2 eff72 ≤ 0.007 | 84 | 32 | 42% | -0.235 | [-0.34, +0.15] |
| H2 eff72 > 0.007 | 603 | 127 | 60% | -0.074 | [-0.13, +0.01] |

## FRESH-only observe bars (fills after 2026-10-01 02:00 UTC; full-size, non-MANUAL; eff72 = live stamp, else the validated rebuild)

- eff72 source on fresh fills: 0 live · 0 rebuilt · 0 none (excluded) · rebuild vs live stamp on 24 fills: corr 0.999 · same tier 96%
- burst crowding: 0 fresh fills
- BTC chop eff72 ≤ 0.007: 0 fresh fills
- sub-line A (pre-declared: the 2nd+ fill of a burst in chop is the WORST cell): chop ∧ 2nd+-burst → 0 fills · 0 windows · chop ∧ not-2nd → 0 fills · its share of the chop tier's loss (1× pct): n/a — narrow-rule bar: chop bar passes ∧ cell N ≥ 6 on ≥ 3 windows ∧ share ≥ 50 % ∧ no window ≥ 50 % of the cell loss
- sub-line B (comparison only: middle ∧ slope<0 expected ≤ break-even while middle ∧ slope≥0 stays strong): middle ∧ slope<0 → 0 fills · middle ∧ slope≥0 → 0 fills
- comparison (no bar): 0.007 < eff72 ≤ 0.026 → 0 fills · eff72 > 0.026 → 0 fills
