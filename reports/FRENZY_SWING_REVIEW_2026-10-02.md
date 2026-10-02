# 🔪 Hostile review — FRENZY swing short (short when the frenzy flag switches off, 24 h, 10 % stop)

352 trades on 183 pairs, 184 days, Jan–Sep 2026. Entry at the bar open, stop filled at the worse of the stop and the breaching bar's open. % of position at 1×.

## 1 · Costs

| Line | per trade |
|---|---|
| before funding (after 0.11 %) | +2.56 (by day +2.34 [+0.38, +4.31], 184 days) |
| funding while open (short receives +) | mean -0.731 · median -0.018 · worst -13.35 · best +3.23 · missing 0 |
| **after funding** | **+1.83 (by day +1.66 [-0.28, +3.60], 184 days)** · week-clustered bootstrap [-0.09, +3.72] |
| + 0.5 % extra slippage on every stop | +1.56 |
| + 1.0 % extra slippage on every stop | +1.29 |
| + 2.0 % extra slippage on every stop | +0.74 |

## 2 · Is it the switch-off, or just shorting?

| Comparison (same exit, same costs, no funding) | N | per trade |
|---|---|---|
| the swing short | 352 | +2.56 (pair-weighted +3.52, day-weighted +2.34) |
| every eligible pair at the same timestamps (market drift) | 88 pairs / time | -0.01 |
| same pairs, random non-frenzy times (≥ 3 days away) | 6,883 | +1.11 (pair-weighted +1.08, day-weighted +1.09) |
| every bar INSIDE a frenzy | 42,863 | +1.29 (pair-weighted +5.51, day-weighted +2.77) |
| fixed delay: 0 h after the frenzy starts | 382 | +1.00 · won 37% |
| fixed delay: 6 h after the frenzy starts | 380 | +1.25 · won 45% |
| fixed delay: 12 h after the frenzy starts | 380 | +1.55 · won 46% |
| fixed delay: 24 h after the frenzy starts | 379 | +1.71 · won 49% |
| fixed delay: 48 h after the frenzy starts | 378 | +0.26 · won 43% |
| **excess over the market, after funding** | 352 | **+1.84 (by day +1.62 [-0.35, +3.58], 184 days)** · week bootstrap [-0.06, +3.62] |

## 3 · Robustness

| Cut | N | won | per trade after funding |
|---|---|---|---|
| 2026-02 | 18 | 44% | +4.35 · without this month: +1.69 |
| 2026-03 | 41 | 46% | -0.68 · without this month: +2.16 |
| 2026-04 | 64 | 48% | +3.84 · without this month: +1.38 |
| 2026-05 | 45 | 42% | -0.51 · without this month: +2.17 |
| 2026-06 | 32 | 38% | +4.74 · without this month: +1.54 |
| 2026-07 | 48 | 33% | -2.45 · without this month: +2.50 |
| 2026-08 | 59 | 46% | +3.17 · without this month: +1.56 |
| 2026-09 | 45 | 44% | +3.32 · without this month: +1.61 |
| best 5 % of trades removed | 335 | | -0.54 |
| best 10 % removed | 318 | | -2.00 |
| switched off because 24h<30% | 207 | 43% | +2.57 |
| switched off because ATR<2 | 141 | 44% | +0.65 |
| switched off because vol<100x | 4 | 50% | +4.79 |
| frenzy came BACK within 6 h (flicker) | 259 | 37% | -0.24 |
| frenzy did not come back within 6 h | 93 | 61% | +7.59 |
| frenzy had lasted < 1 h | 96 | 46% | +1.17 |
| 1–6 h | 124 | 47% | +1.62 |
| ≥ 6 h | 132 | 38% | +2.51 |
| 24 h volume $20M–$100M | 18 | 78% | +5.52 |
| 24 h volume $100M–$400M | 190 | 47% | +2.39 |
| 24 h volume $400M–∞ | 144 | 33% | +0.63 |

## 4 · Portfolio

- stopped out: 54 % of trades (avg stop fill -10.00 %); the rest average +16.4 %.
- max positions open at once: 8; longest losing streak: 8 trades; deepest drawdown of the running sum: -158 points (total +644).
- pairs positive 55 % of 183; best 3 pairs +276 of +644; worst day -30, best day +102; days positive 46 %.
