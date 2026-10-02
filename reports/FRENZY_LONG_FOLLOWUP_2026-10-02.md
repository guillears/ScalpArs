# 🔥 FRENZY_LONG follow-up — ATR at entry, and the ride-it exit (frozen before reading)

2,209 entries: 1,220 SEEN (24 h volume ≥ $100M) · 989 UNSEEN ($20–100M). Gap-aware fills, 0.10 % slippage, 0.11 % costs, funding. Median 5m ATR at entry: 2.5 % · ATR ≤ 2 % on 27% of entries.

## TEST 1 · ATR — SEEN (1,220 entries, 255 pairs) · 1-minute bars, 12 h cap

| Rule | trades | won | per trade | Jan–Apr / May–Sep | 95 % by day | best 5 % removed | per trade in stops | account at 2 % per stop (deepest drawdown) | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| All entries · stop 3 · trail 5/1.5 (the design) | 1209 | 41% | +0.05 | +0.01 / +0.07 | [-0.17, +0.28] | -0.43 | +0.015 R | ×0.97 (−81%) | 11 | — |
| GATE ATR ≤ 2 % · stop 3 · trail 5/1.5 | 357 | 44% | +0.25 | +0.39 / +0.10 | [-0.14, +0.65] | -0.16 | +0.078 R | ×1.57 (−46%) | 6 | — |
| (the rest: ATR > 2 %) | 853 | 39% | -0.04 | -0.21 / +0.06 | [-0.31, +0.24] | -0.53 | -0.013 R | ×0.60 (−75%) | 11 | — |
| SCALED stop 1.5×ATR · trail arms 2.5×ATR, gives back 1×ATR | 1207 | 40% | +0.04 | -0.03 / +0.09 | [-0.30, +0.37] | -0.74 | +0.016 R | ×0.97 (−72%) | 12 | — |
| SCALED stop 2×ATR · same trail | 1195 | 45% | -0.03 | -0.09 / +0.01 | [-0.42, +0.33] | -0.83 | +0.002 R | ×0.77 (−76%) | 10 | — |

## TEST 2 · RIDE IT — SEEN · 5-minute bars, sell after a close below the spike's average price

| Rule | trades | won | per trade | Jan–Apr / May–Sep | 95 % by day | best 5 % removed | per trade in stops | account at 2 % per stop (deepest drawdown) | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| stop 8 % (the earlier year test) | 970 | 19% | -0.95 | -1.03 / -0.89 | [-2.51, +0.83] | -4.91 | -0.116 R | ×0.03 (−99%) | 25 | — |
| stop 3 % | 1088 | 13% | -1.04 | -0.91 / -1.14 | [-2.26, +0.21] | -3.69 | -0.325 R | ×0.00 (−100%) | 38 | — |
| stop 1.5×ATR (2–8 %) | 1070 | 15% | -0.94 | -1.25 / -0.71 | [-2.33, +0.60] | -4.23 | -0.180 R | ×0.00 (−100%) | 36 | — |
| stop 2×ATR (2–8 %) | 1032 | 17% | -0.89 | -1.18 / -0.67 | [-2.30, +0.71] | -4.45 | -0.151 R | ×0.00 (−100%) | 25 | — |
| stop 3×ATR (3–10 %) | 975 | 19% | -0.91 | -1.11 / -0.76 | [-2.64, +0.85] | -4.87 | -0.132 R | ×0.01 (−100%) | 25 | — |

Ride-it with the 2×ATR stop: stopped 55% · median hold 1.1 h · average win +24.5 % / loss -6.1 % · biggest win +405 % · trades per day (median on trading days) 4.

## TEST 1 · ATR — UNSEEN (989 entries, 307 pairs) · 1-minute bars, 12 h cap

| Rule | trades | won | per trade | Jan–Apr / May–Sep | 95 % by day | best 5 % removed | per trade in stops | account at 2 % per stop (deepest drawdown) | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| All entries · stop 3 · trail 5/1.5 (the design) | 984 | 38% | -0.08 | -0.06 / -0.10 | [-0.34, +0.17] | -0.58 | -0.026 R | ×0.43 (−86%) | 14 | — |
| GATE ATR ≤ 2 % · stop 3 · trail 5/1.5 | 235 | 39% | +0.10 | +0.27 / -0.09 | [-0.42, +0.61] | -0.34 | +0.030 R | ×1.06 (−37%) | 11 | — |
| (the rest: ATR > 2 %) | 750 | 37% | -0.14 | -0.19 / -0.11 | [-0.45, +0.16] | -0.65 | -0.045 R | ×0.40 (−82%) | 11 | — |
| SCALED stop 1.5×ATR · trail arms 2.5×ATR, gives back 1×ATR | 977 | 39% | -0.01 | +0.11 / -0.10 | [-0.34, +0.34] | -0.74 | -0.001 R | ×0.70 (−68%) | 9 | — |
| SCALED stop 2×ATR · same trail | 963 | 45% | +0.09 | +0.26 / -0.04 | [-0.31, +0.50] | -0.70 | +0.006 R | ×0.87 (−54%) | 9 | — |

## TEST 2 · RIDE IT — UNSEEN · 5-minute bars, sell after a close below the spike's average price

| Rule | trades | won | per trade | Jan–Apr / May–Sep | 95 % by day | best 5 % removed | per trade in stops | account at 2 % per stop (deepest drawdown) | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| stop 8 % (the earlier year test) | 885 | 23% | +1.19 | +3.72 / -0.65 | [-1.09, +4.95] | -2.77 | +0.144 R | ×1.19 (−80%) | 27 | — |
| stop 3 % | 926 | 15% | +1.10 | +3.01 / -0.28 | [-0.83, +4.60] | -2.18 | +0.344 R | ×0.18 (−96%) | 30 | — |
| stop 1.5×ATR (2–8 %) | 919 | 18% | +0.97 | +3.13 / -0.61 | [-1.10, +4.41] | -2.52 | +0.284 R | ×0.41 (−94%) | 30 | — |
| stop 2×ATR (2–8 %) | 904 | 20% | +0.87 | +3.15 / -0.79 | [-1.20, +4.84] | -2.79 | +0.195 R | ×0.46 (−91%) | 27 | — |
| stop 3×ATR (3–10 %) | 888 | 23% | +1.20 | +3.72 / -0.61 | [-1.06, +5.08] | -2.71 | +0.181 R | ×1.90 (−79%) | 27 | — |

Ride-it with the 2×ATR stop: stopped 48% · median hold 1.2 h · average win +22.1 % / loss -4.4 % · biggest win +1409 % · trades per day (median on trading days) 4.

## NOT tested

- Delisted pairs; slippage beyond 0.10 % (thin UNSEEN pairs will be worse); exchange position limits; the live scan delay.
- TEST 2 runs on 5-minute bars (stop first inside a bar), TEST 1 on 1-minute bars — the two tests are not on one ruler.
- No fresh time period: UNSEEN means thinner pairs, not later dates.
