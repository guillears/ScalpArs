# ✂️ BTC-RSI early cut on momentum longs — close when BTC's RSI has dropped X points since entry

Δ = % of position versus what the trade actually did (positive = the cut helped). Two-sided: winners cut too early count against the rule.

## Year engine replay — 3591 fills, 247 days · win rate 60 % · median hold 4 closed 5m bars

| Rule | cuts fired | on eventual losers: N · Δ each | on eventual winners: N · Δ each | Δ per fill Jan–Apr / May–Sep | Δ by day [95 %] | PASS |
|---|---|---|---|---|---|---|
| RSI −5, always | 1719 (48%) | 908 · +0.444 | 811 · -0.507 | -0.0039 / -0.0008 | +0.0188 [-0.0066, +0.0441] | — |
| RSI −10, always | 958 (27%) | 558 · +0.368 | 400 · -0.545 | -0.0157 / +0.0068 | +0.0013 [-0.0183, +0.0210] | — |
| RSI −15, always | 460 (13%) | 265 · +0.305 | 195 · -0.556 | -0.0153 / -0.0012 | -0.0008 [-0.0132, +0.0116] | — |
| RSI −20, always | 181 (5%) | 111 · +0.240 | 70 · -0.619 | -0.0063 / -0.0033 | -0.0040 [-0.0132, +0.0051] | — |
| RSI −5, losing | 1480 (41%) | 896 · +0.415 | 584 · -0.677 | -0.0087 / -0.0045 | +0.0111 [-0.0120, +0.0342] | — |
| RSI −10, losing | 864 (24%) | 555 · +0.348 | 309 · -0.676 | -0.0144 / +0.0041 | +0.0006 [-0.0176, +0.0189] | — |
| RSI −15, losing | 413 (12%) | 264 · +0.297 | 149 · -0.681 | -0.0146 / +0.0006 | -0.0001 [-0.0121, +0.0119] | — |
| RSI −20, losing | 167 (5%) | 111 · +0.225 | 56 · -0.692 | -0.0054 / -0.0025 | -0.0037 [-0.0125, +0.0051] | — |

## Live master momentum longs (current stack) — 108 fills, 54 days · win rate 80 % · median hold 4 closed 5m bars

| Rule | cuts fired | on eventual losers: N · Δ each | on eventual winners: N · Δ each | Δ per fill Jan–Apr / May–Sep | Δ by day [95 %] | PASS |
|---|---|---|---|---|---|---|
| RSI −5, always | 36 (33%) | 11 · +0.541 | 25 · -0.644 | +nan / -0.0939 | -0.0868 [-0.1688, -0.0049] | — |
| RSI −10, always | 16 (15%) | 4 · +0.437 | 12 · -0.906 | +nan / -0.0845 | -0.0806 [-0.1461, -0.0151] | — |
| RSI −15, always | 15 (14%) | 4 · +0.343 | 11 · -0.842 | +nan / -0.0731 | -0.0679 [-0.1240, -0.0117] | — |
| RSI −20, always | 6 (6%) | 2 · +0.422 | 4 · -0.869 | +nan / -0.0244 | -0.0274 [-0.0699, +0.0152] | — |
| RSI −5, losing | 29 (27%) | 11 · +0.506 | 18 · -0.701 | +nan / -0.0653 | -0.0582 [-0.1237, +0.0072] | — |
| RSI −10, losing | 14 (13%) | 4 · +0.437 | 10 · -0.816 | +nan / -0.0593 | -0.0621 [-0.1215, -0.0028] | — |
| RSI −15, losing | 13 (12%) | 4 · +0.343 | 9 · -0.805 | +nan / -0.0544 | -0.0554 [-0.1067, -0.0041] | — |
| RSI −20, losing | 5 (5%) | 2 · +0.422 | 3 · -1.041 | +nan / -0.0211 | -0.0241 [-0.0630, +0.0149] | — |

