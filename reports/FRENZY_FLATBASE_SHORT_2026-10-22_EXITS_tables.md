## WALK-FORWARD (select on the other 3 half-years pooled by the day-bootstrap lower bound, N ≥ 300; score on the held-out half)
| held-out fold | selected cell | selection: N · avg [lo] | held-out N | held-out WR | held-out avg [95 %] | held-out null · p | best 5 % removed |
|---|---|---|---|---|---|---|---|
| 2025H1 (2025: flagged) | S4|tp8|72h|stop15 | 1895 · +0.67 [+0.19] | 357 | 66% | +1.00 [-0.21, +2.17] | +0.32 · 0.07 | +0.64 |
| 2025H2 (2025: flagged) | S4|tp8|72h|stop10 | 1477 · +0.87 [+0.42] | 783 | 57% | +0.19 [-0.45, +0.80] | +0.48 · 0.80 | -0.23 |
| 2026H1 | D2|tp20|48h|stop25 | 1551 · +0.94 [+0.16] | 736 | 64% | +0.85 [-0.47, +2.11] | +0.33 · 0.17 | -0.16 |
| 2026H2 | S4|tp8|72h|stop25 | 1853 · +0.82 [+0.22] | 381 | 72% | +0.46 [-0.77, +1.64] | -0.06 · 0.21 | +0.04 |

selection counts: S4|tp8|72h|stop15 ×1 · S4|tp8|72h|stop10 ×1 · D2|tp20|48h|stop25 ×1 · S4|tp8|72h|stop25 ×1
top 5 per selection set (stability view): 
- without 2025H1: S4|tp8|72h|stop15 [+0.19] · D2|tp15|24h|stop25 [+0.18] · D2|tp15|48h|stop25 [+0.16] · D2|tp20|48h|stop25 [+0.15] · S4|tp8|72h|stop10 [+0.12]
- without 2025H2: S4|tp8|72h|stop10 [+0.42] · S4|tp8|48h|stop25 [+0.41] · S4|tp8|72h|stop15 [+0.39] · S4|tp8|48h|stop15 [+0.36] · S4|tp8|48h|stop10 [+0.32]
- without 2026H1: D2|tp20|48h|stop25 [+0.16] · D2|tp20|72h|stop25 [+0.13] · D2|tp15|72h|stop25 [+0.09] · S4|tp8|72h|stop10 [+0.07] · D2|tp15|48h|stop25 [+0.06]
- without 2026H2: S4|tp8|72h|stop25 [+0.22] · S4|tp8|72h|stop10 [+0.21] · S4|tp8|72h|stop15 [+0.21] · S4|tp8|48h|stop25 [+0.16] · S4|tp8|48h|stop10 [+0.15]

CANDIDATES (selected ≥ 3 of 4 folds AND 2026-H1 > 0 AND 2026-H2 > 0): NONE

## Every walk-forward-selected cell, all blocks (2025 rows flagged: lower-reliability data)

### S4|tp8|72h|stop15
| period | N | days | WR | avg % | 95 % day-boot | pre-funding | best 5 % removed | worst | random-bar null (same exit, costs, funding, one-per-pair) | p | top pair / day share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ⚠ 2025-H1 | 357 | 132 | 66% | +1.00 | [-0.21, +2.17] | +1.29 | +0.64 | -20.7 | +0.32 | 0.070 | 11 % / 22 % |
| ⚠ 2025-H2 | 780 | 163 | 64% | +0.30 | [-0.46, +1.07] | +0.56 | -0.11 | -21.2 | +0.61 | 0.785 | 19 % / 41 % |
| 2026-H1 OOS | 730 | 165 | 68% | +1.08 | [+0.37, +1.82] | +1.54 | +0.71 | -21.0 | +0.57 | 0.100 | 5 % / 11 % |
| 2026-H2 OOS | 385 | 90 | 68% | +0.66 | [-0.39, +1.66] | +0.92 | +0.26 | -17.1 | +0.07 | 0.180 | 10 % / 18 % |
| 2026 OOS total | 1115 | 255 | 68% | +0.93 | [+0.28, +1.53] | +1.33 | +0.56 | -21.0 | +0.39 | 0.050 | 4 % / 8 % |
| ② master (real eps) | 38 | 11 | 63% | +0.32 | [-2.80, +3.93] | +0.70 | -0.10 | -16.5 | +nan | nan | 114 % / 193 % |
| ① live (real eps, open = MTM) | 13 | 6 | 77% | +1.64 | [-5.13, +7.39] | +1.89 | +nan | -16.1 | +nan | nan | 37 % / 109 % |
| ③ full year | 2252 | 550 | 66% | +0.72 | [+0.28, +1.18] | +1.05 | +0.34 | -21.2 | +nan | nan | 6 % / 6 % |

windows ①②: 17 · mean of window means +0.90 · positive 9 of 17
sizing at $3,200 (full year): 1st-pct loss -16.5 % → $972/short · taken 1459 · sum $+5,906 · worst trade $-206 · max drawdown (daily MTM) $-3,204 (-100 %)
sizing at $3,200 (2026 OOS): 1st-pct loss -16.6 % → $965/short · taken 727 · sum $+5,011 · worst trade $-199 · max drawdown (daily MTM) $-1,294 (-40 %)
② master windows: 11 · mean of window means +0.80
SHIP BARS (OOS): N PASS · lo PASS · null FAIL · h1 PASS · h2 PASS · win PASS · haircut PASS · top5 PASS · conc PASS · master_not_negative PASS · candidate FAIL → FAIL

### S4|tp8|72h|stop10
| period | N | days | WR | avg % | 95 % day-boot | pre-funding | best 5 % removed | worst | random-bar null (same exit, costs, funding, one-per-pair) | p | top pair / day share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ⚠ 2025-H1 | 359 | 132 | 62% | +1.17 | [+0.17, +2.13] | +1.40 | +0.82 | -13.8 | +0.27 | 0.015 | 9 % / 14 % |
| ⚠ 2025-H2 | 783 | 163 | 57% | +0.19 | [-0.45, +0.80] | +0.43 | -0.23 | -16.1 | +0.48 | 0.800 | 21 % / 47 % |
| 2026-H1 OOS | 733 | 165 | 62% | +0.77 | [+0.14, +1.38] | +1.19 | +0.39 | -15.6 | +0.35 | 0.135 | 7 % / 12 % |
| 2026-H2 OOS | 385 | 90 | 62% | +0.76 | [-0.26, +1.71] | +0.99 | +0.37 | -12.2 | +0.04 | 0.070 | 9 % / 16 % |
| 2026 OOS total | 1118 | 255 | 62% | +0.77 | [+0.23, +1.27] | +1.12 | +0.39 | -15.6 | +0.24 | 0.040 | 6 % / 8 % |
| ② master (real eps) | 38 | 11 | 50% | -1.18 | [-4.60, +2.00] | -0.91 | -1.68 | -11.3 | +nan | nan | nan % / nan % |
| ① live (real eps, open = MTM) | 13 | 6 | 77% | +2.83 | [-2.49, +7.35] | +3.04 | +nan | -10.7 | +nan | nan | 21 % / 63 % |
| ③ full year | 2260 | 550 | 60% | +0.63 | [+0.25, +1.01] | +0.92 | +0.25 | -16.1 | +nan | nan | 5 % / 5 % |

windows ①②: 17 · mean of window means +0.14 · positive 9 of 17
sizing at $3,200 (full year): 1st-pct loss -11.6 % → $1377/short · taken 1590 · sum $+11,100 · worst trade $-222 · max drawdown (daily MTM) $-3,165 (-99 %)
sizing at $3,200 (2026 OOS): 1st-pct loss -11.6 % → $1376/short · taken 785 · sum $+8,251 · worst trade $-215 · max drawdown (daily MTM) $-2,211 (-69 %)
② master windows: 11 · mean of window means -1.05
SHIP BARS (OOS): N PASS · lo PASS · null PASS · h1 PASS · h2 PASS · win PASS · haircut FAIL · top5 PASS · conc PASS · master_not_negative FAIL · candidate FAIL → FAIL

### D2|tp20|48h|stop25
| period | N | days | WR | avg % | 95 % day-boot | pre-funding | best 5 % removed | worst | random-bar null (same exit, costs, funding, one-per-pair) | p | top pair / day share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ⚠ 2025-H1 | 373 | 132 | 61% | +0.89 | [-0.87, +2.49] | +1.84 | -0.13 | -35.0 | -0.45 | 0.000 | 24 % / 36 % |
| ⚠ 2025-H2 | 787 | 167 | 60% | +0.70 | [-0.39, +1.74] | +1.32 | -0.32 | -31.6 | +0.30 | 0.210 | 11 % / 19 % |
| 2026-H1 OOS | 736 | 166 | 64% | +0.85 | [-0.47, +2.11] | +1.79 | -0.16 | -40.4 | +0.33 | 0.175 | 9 % / 35 % |
| 2026-H2 OOS | 391 | 88 | 64% | +1.48 | [-0.02, +3.00] | +2.07 | +0.48 | -33.5 | -0.61 | 0.000 | 8 % / 16 % |
| 2026 OOS total | 1127 | 254 | 64% | +1.07 | [+0.10, +2.05] | +1.89 | +0.06 | -40.4 | -0.00 | 0.020 | 5 % / 18 % |
| ② master (real eps) | 37 | 12 | 57% | -2.13 | [-6.79, +2.66] | -1.05 | -3.40 | -27.0 | +nan | nan | nan % / nan % |
| ① live (real eps, open = MTM) | 13 | 6 | 62% | -0.56 | [-10.47, +14.06] | -0.42 | +nan | -26.4 | +nan | nan | nan % / nan % |
| ③ full year | 2287 | 553 | 62% | +0.91 | [+0.28, +1.55] | +1.68 | -0.09 | -40.4 | +nan | nan | 6 % / 10 % |

windows ①②: 18 · mean of window means +1.26 · positive 10 of 18
sizing at $3,200 (full year): 1st-pct loss -28.2 % → $568/short · taken 1328 · sum $+8,058 · worst trade $-229 · max drawdown (daily MTM) $-1,616 (-51 %)
sizing at $3,200 (2026 OOS): 1st-pct loss -29.3 % → $545/short · taken 646 · sum $+4,135 · worst trade $-220 · max drawdown (daily MTM) $-1,553 (-49 %)
② master windows: 12 · mean of window means -0.18
SHIP BARS (OOS): N PASS · lo PASS · null PASS · h1 PASS · h2 PASS · win PASS · haircut PASS · top5 PASS · conc PASS · master_not_negative FAIL · candidate FAIL → FAIL

### S4|tp8|72h|stop25
| period | N | days | WR | avg % | 95 % day-boot | pre-funding | best 5 % removed | worst | random-bar null (same exit, costs, funding, one-per-pair) | p | top pair / day share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ⚠ 2025-H1 | 354 | 132 | 69% | +1.13 | [-0.23, +2.36] | +1.56 | +0.76 | -27.3 | +0.23 | 0.050 | 11 % / 20 % |
| ⚠ 2025-H2 | 774 | 163 | 68% | +0.47 | [-0.46, +1.36] | +0.81 | +0.07 | -28.6 | +0.69 | 0.705 | 12 % / 33 % |
| 2026-H1 OOS | 725 | 165 | 72% | +1.05 | [+0.09, +1.97] | +1.58 | +0.68 | -32.8 | +0.55 | 0.155 | 5 % / 15 % |
| 2026-H2 OOS | 381 | 90 | 72% | +0.46 | [-0.77, +1.64] | +0.80 | +0.04 | -33.5 | -0.06 | 0.215 | 15 % / 27 % |
| 2026 OOS total | 1106 | 255 | 72% | +0.85 | [+0.11, +1.56] | +1.31 | +0.47 | -33.5 | +0.33 | 0.100 | 5 % / 12 % |
| ② master (real eps) | 37 | 10 | 73% | +2.10 | [-1.87, +5.69] | +2.69 | +1.76 | -25.9 | +nan | nan | 18 % / 30 % |
| ① live (real eps, open = MTM) | 13 | 6 | 77% | -0.66 | [-10.06, +7.39] | -0.42 | +nan | -26.1 | +nan | nan | nan % / nan % |
| ③ full year | 2234 | 550 | 70% | +0.76 | [+0.22, +1.29] | +1.18 | +0.38 | -33.5 | +nan | nan | 6 % / 7 % |

windows ①②: 16 · mean of window means +1.11 · positive 12 of 16
sizing at $3,200 (full year): 1st-pct loss -26.5 % → $604/short · taken 1364 · sum $+4,306 · worst trade $-202 · max drawdown (daily MTM) $-1,401 (-44 %)
sizing at $3,200 (2026 OOS): 1st-pct loss -26.8 % → $597/short · taken 676 · sum $+2,179 · worst trade $-196 · max drawdown (daily MTM) $-1,442 (-45 %)
② master windows: 10 · mean of window means +2.64
SHIP BARS (OOS): N PASS · lo PASS · null FAIL · h1 PASS · h2 PASS · win PASS · haircut PASS · top5 PASS · conc PASS · master_not_negative PASS · candidate FAIL → FAIL

## All 504 cells (7 entries × 72 exits) — 2025 vs 2026 (2026 is OOS only for a pick selected without it), sorted by 2025 lower bound (top 25)
| cell | 2025_N | 2025_avg | 2025_lo | 2026H1_avg | 2026H2_avg | OOS26_N | OOS26_avg | OOS26_lo | OOS26_null | OOS26_p_null | OOS26_ex_top5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S4|tp5|72h|stop10 | 1143 | +0.39 | -0.04 | +0.49 | +0.15 | 1118 | +0.37 | -0.07 | +0.12 | +0.17 | +0.13 |
| S4|tp8|72h|stop10 | 1142 | +0.50 | -0.07 | +0.77 | +0.76 | 1118 | +0.77 | +0.23 | +0.24 | +0.04 | +0.39 |
| D6|tp15|72h|stop15 | 1165 | +0.76 | -0.08 | +0.55 | +0.57 | 1136 | +0.56 | -0.36 | +0.49 | +0.40 | -0.20 |
| D2|tp15|72h|stop25 | 1159 | +0.89 | -0.08 | +0.67 | +0.93 | 1123 | +0.76 | -0.24 | +0.38 | +0.21 | +0.00 |
| D2|tp20|72h|stop25 | 1155 | +0.97 | -0.08 | +0.66 | +1.00 | 1116 | +0.78 | -0.35 | +0.25 | +0.15 | -0.24 |
| S4|tp8|24h|stop10 | 1144 | +0.40 | -0.09 | +0.62 | +0.41 | 1118 | +0.55 | +0.10 | +0.07 | +0.04 | +0.16 |
| D12|tp5|12h|stop15 | 1174 | +0.23 | -0.11 | -0.24 | +0.57 | 1147 | +0.04 | -0.38 | +0.01 | +0.46 | -0.22 |
| S4|tp5|48h|stop10 | 1143 | +0.31 | -0.12 | +0.52 | +0.07 | 1118 | +0.37 | -0.03 | +0.10 | +0.17 | +0.13 |
| D2|tp15|72h|stop10 | 1177 | +0.54 | -0.12 | +0.63 | +0.25 | 1147 | +0.50 | -0.20 | +0.27 | +0.21 | -0.27 |
| S4|tp5|24h|stop10 | 1144 | +0.28 | -0.13 | +0.37 | +0.02 | 1118 | +0.25 | -0.15 | +0.07 | +0.24 | +0.00 |
| D2|tp20|48h|stop25 | 1160 | +0.76 | -0.14 | +0.85 | +1.48 | 1127 | +1.07 | +0.10 | -0.00 | +0.02 | +0.06 |
| S4|tp8|48h|stop10 | 1142 | +0.39 | -0.14 | +0.81 | +0.58 | 1118 | +0.73 | +0.20 | +0.18 | +0.03 | +0.35 |
| D2|tp20|12h|stop25 | 1183 | +0.45 | -0.15 | +0.45 | +0.74 | 1149 | +0.55 | -0.12 | -0.17 | +0.01 | -0.48 |
| S4|tp8|72h|stop25 | 1128 | +0.68 | -0.15 | +1.05 | +0.46 | 1106 | +0.85 | +0.11 | +0.33 | +0.10 | +0.47 |
| D2|tp20|72h|stop10 | 1175 | +0.62 | -0.16 | +0.73 | -0.02 | 1139 | +0.47 | -0.27 | +0.21 | +0.21 | -0.55 |
| S4|tp8|72h|stop15 | 1137 | +0.52 | -0.17 | +1.08 | +0.66 | 1115 | +0.93 | +0.28 | +0.39 | +0.05 | +0.56 |
| D2|tp15|12h|stop25 | 1183 | +0.40 | -0.17 | +0.56 | +0.85 | 1149 | +0.66 | -0.01 | -0.14 | +0.01 | -0.10 |
| D2|tp15|72h|stop15 | 1168 | +0.65 | -0.17 | +0.87 | +0.23 | 1139 | +0.65 | -0.25 | +0.46 | +0.29 | -0.11 |
| D2|tp15|24h|stop25 | 1179 | +0.54 | -0.18 | +0.69 | +1.03 | 1145 | +0.81 | -0.07 | +0.05 | +0.04 | +0.06 |
| D12|tp5|12h|stop10 | 1174 | +0.15 | -0.19 | -0.22 | +0.56 | 1147 | +0.05 | -0.31 | -0.01 | +0.39 | -0.21 |
| D2|trail8-4|12h|stop25 | 1183 | +0.31 | -0.19 | +0.42 | +0.20 | 1149 | +0.34 | -0.23 | -0.20 | +0.02 | -0.44 |
| S4|tp5|72h|stop15 | 1142 | +0.34 | -0.20 | +0.78 | +0.04 | 1118 | +0.53 | +0.01 | +0.28 | +0.20 | +0.30 |
| D2|tp10|72h|stop25 | 1165 | +0.65 | -0.20 | +0.57 | +0.67 | 1130 | +0.61 | -0.27 | +0.34 | +0.28 | +0.11 |
| D2|tp10|12h|stop25 | 1183 | +0.36 | -0.20 | +0.61 | +0.56 | 1149 | +0.59 | -0.01 | -0.12 | +0.01 | +0.09 |
| D6|tp10|72h|stop10 | 1174 | +0.41 | -0.20 | +0.50 | +0.71 | 1151 | +0.57 | -0.10 | +0.24 | +0.14 | +0.07 |

OOS (2026) cells with lo > 0: 35 of 504 · OOS mean > 0: 428 · H1 > 0 and H2 > 0: 248 · OOS beats its null at p < 0.05: 25
Best 5 cells by the OOS lower bound (hindsight — NOT a valid pick): S4|tp8|48h|stop15 OOS +0.89 [+0.30] H2 +0.47 null +0.24 p 0.03 (2025 +0.30) · S4|tp8|72h|stop15 OOS +0.93 [+0.28] H2 +0.66 null +0.39 p 0.05 (2025 +0.52) · S4|tp8|48h|stop25 OOS +0.96 [+0.26] H2 +0.58 null +0.13 p 0.02 (2025 +0.46) · S4|tp8|24h|stop25 OOS +0.85 [+0.24] H2 +0.38 null +0.07 p 0.00 (2025 +0.28) · S4|tp8|72h|stop10 OOS +0.77 [+0.23] H2 +0.76 null +0.24 p 0.04 (2025 +0.50)

## Shape: mean of the OOS (2026) avg by TP / hold (averaged over entries and stops)
hold       12h   24h   48h   72h
tp                              
tp10      0.10  0.43  0.51  0.49
tp15      0.07  0.42  0.43  0.36
tp20      0.04  0.39  0.39  0.28
tp5       0.06  0.22  0.25  0.25
tp8       0.10  0.39  0.49  0.49
trail8-4  0.01  0.24  0.28  0.23

2025 (in-sample) same shape:
hold       12h   24h   48h   72h
tp                              
tp10      0.00  0.04  0.19  0.34
tp15      0.01  0.06  0.23  0.38
tp20     -0.00  0.03  0.16  0.29
tp5      -0.05 -0.06  0.04  0.11
tp8      -0.00  0.01  0.16  0.29
trail8-4 -0.04 -0.06  0.10  0.25

## Data reliability by half-year (why 2025 is flagged)
| half-year | episodes (all) | pairs | flat share | episodes on pairs now delisted | flat signals | funding history missing (signals, 24 h exit) | funding paid avg (24 h, flat) | BTC change | BTC 30-d realised vol (median) |
|---|---|---|---|---|---|---|---|---|---|
| 2025H1 | 1139 | 324 | 33 % | 28 % | 2440 | 0.0 % | -0.363 | +13 % | 52 % |
| 2025H2 | 2368 | 504 | 34 % | 19 % | 5217 | 0.0 % | -0.242 | -17 % | 35 % |
| 2026H1 | 2158 | 457 | 35 % | 8 % | 4844 | 0.0 % | -0.400 | -32 % | 45 % |
| 2026H2 | 1029 | 356 | 39 % | 3 % | 2583 | 0.0 % | -0.259 | +45 % | 36 % |
