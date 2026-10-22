PROXY CALIBRATION (cohort pairs with a real stamped fill, Sep-28 →): 221 fills · 90 pairs · proxy ÷ stamped median 1.003 · IQR 0.996–1.018 · within ±10 % 92 % · same side of turnover = 1 when vol is fixed: 98 % within ±20 %

# Cohort tp8|72h|stop15: scored for turnover 84 %
| block | unscored trades: N · WR · avg % |
|---|---|
| ① live (real eps, open = MTM) | 0 · nan % · +nan |
| ② master (real eps) | 0 · nan % · +nan |
| ③ 2025 ⚠ | 250 · 68 % · +1.18 |
| ③ 2026-H1 | 73 · 66 % · +0.87 |
| ③ 2026-H2 | 44 · 82 % · +2.87 |
| ③ 2026 (H1+H2) | 117 · 72 % · +1.62 |

# CANDIDATE: KEEP turnover > 1 (block vol24 ≤ mcap)

## exit tp8|72h|stop5 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 54 % · +1.45 [-1.21, +4.16] · $+36 · -5.4 · $-30 | 7 · 57 % · +1.90 [-3.43, +6.86] · $+25 · -5.4 · $-20 | 6 · 50 % · +0.93 | +3.3 pp | – |
| ② master (real eps) | 38 · 32 % · -1.24 [-3.27, +0.72] · $-89 · -5.6 · $-133 | 23 · 39 % · -0.35 [-3.21, +2.08] · $-15 · -5.6 · $-67 | 15 · 20 % · -2.59 | +7.6 pp | – |
| ③ 2025 ⚠ | 1144 · 44 % · +0.31 [-0.10, +0.72] · $+676 · -7.2 · $-402 | 236 · 43 % · +0.29 [-0.52, +1.10] · $+129 · -6.8 · $-187 | 657 · 45 % · +0.32 | -0.9 pp | -0.10 · 0.15 · same filter on the random bar: -0.15 · p 0.21 |
| ③ 2026-H1 | 734 · 43 % · +0.06 [-0.45, +0.58] · $+84 · -8.2 · $-430 | 296 · 48 % · +0.70 [-0.09, +1.54] · $+395 · -6.7 · $-115 | 365 · 43 % · -0.05 | +4.8 pp | +0.16 · 0.12 · same filter on the random bar: +0.26 · p 0.15 |
| ③ 2026-H2 | 385 · 43 % · +0.13 [-0.60, +0.90] · $+97 · -8.9 · $-277 | 187 · 45 % · +0.43 [-0.58, +1.42] · $+154 · -8.9 · $-238 | 154 · 40 % · -0.36 | +1.8 pp | -0.21 · 0.15 · same filter on the random bar: -0.15 · p 0.17 |
| ③ 2026 (H1+H2) | 1119 · 43 % · +0.09 [-0.34, +0.51] · $+181 · -8.9 · $-576 | 483 · 47 % · +0.60 [-0.05, +1.23] · $+549 · -8.9 · $-241 | 519 · 42 % · -0.14 | +3.6 pp | +0.02 · 0.06 · same filter on the random bar: +0.10 · p 0.08 |

## exit tp8|72h|stop8 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 69 % · +2.51 [-3.21, +7.40] · $+62 · -8.7 · $-48 | 7 · 71 % · +2.85 [-6.14, +7.68] · $+38 · -8.7 · $-32 | 6 · 67 % · +2.10 | +2.2 pp | – |
| ② master (real eps) | 38 · 45 % · -1.05 [-4.07, +1.56] · $-76 · -9.3 · $-103 | 23 · 52 % · +0.15 [-3.94, +3.20] · $+6 · -9.3 · $-56 | 15 · 33 % · -2.89 | +7.4 pp | – |
| ③ 2025 ⚠ | 1143 · 55 % · +0.51 [+0.03, +1.00] · $+1,107 · -14.1 · $-552 | 236 · 57 % · +0.74 [-0.32, +1.74] · $+333 · -11.1 · $-200 | 657 · 55 % · +0.42 | +1.4 pp | -0.10 · 0.07 · same filter on the random bar: -0.09 · p 0.10 |
| ③ 2026-H1 | 734 · 55 % · +0.27 [-0.34, +0.84] · $+370 · -12.2 · $-369 | 296 · 60 % · +1.04 [+0.05, +2.01] · $+586 · -11.9 · $-134 | 365 · 54 % · +0.12 | +5.0 pp | +0.37 · 0.12 · same filter on the random bar: +0.59 · p 0.23 |
| ③ 2026-H2 | 385 · 57 % · +0.71 [-0.17, +1.64] · $+520 · -10.6 · $-278 | 187 · 63 % · +1.64 [+0.51, +2.69] · $+581 · -10.6 · $-164 | 154 · 50 % · -0.53 | +5.2 pp | -0.10 · 0.01 · same filter on the random bar: -0.07 · p 0.01 |
| ③ 2026 (H1+H2) | 1119 · 56 % · +0.42 [-0.07, +0.91] · $+890 · -12.2 · $-369 | 483 · 61 % · +1.27 [+0.54, +2.00] · $+1,167 · -11.9 · $-164 | 519 · 53 % · -0.07 | +5.2 pp | +0.19 · 0.01 · same filter on the random bar: +0.32 · p 0.03 |

## exit tp8|72h|stop10 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · +2.83 [-2.16, +7.40] · $+70 · -10.7 · $-59 | 7 · 71 % · +2.28 [-7.86, +7.68] · $+30 · -10.7 · $-40 | 6 · 83 % · +3.47 | -5.5 pp | – |
| ② master (real eps) | 38 · 50 % · -1.18 [-4.68, +2.05] · $-85 · -11.3 · $-137 | 23 · 57 % · +0.06 [-4.00, +3.43] · $+3 · -11.3 · $-87 | 15 · 40 % · -3.08 | +6.5 pp | – |
| ③ 2025 ⚠ | 1142 · 59 % · +0.50 [-0.05, +1.06] · $+1,082 · -16.1 · $-515 | 235 · 63 % · +0.94 [-0.22, +2.08] · $+420 · -13.1 · $-193 | 657 · 57 % · +0.24 | +4.0 pp | -0.15 · 0.02 · same filter on the random bar: -0.06 · p 0.09 |
| ③ 2026-H1 | 733 · 62 % · +0.77 [+0.12, +1.38] · $+1,073 · -15.6 · $-402 | 296 · 68 % · +1.71 [+0.63, +2.73] · $+959 · -13.9 · $-139 | 364 · 59 % · +0.36 | +6.0 pp | +0.47 · 0.01 · same filter on the random bar: +0.69 · p 0.07 |
| ③ 2026-H2 | 385 · 62 % · +0.76 [-0.26, +1.74] · $+555 · -12.2 · $-322 | 187 · 65 % · +1.46 [+0.23, +2.59] · $+517 · -12.2 · $-174 | 154 · 55 % · -0.32 | +3.7 pp | -0.17 · 0.01 · same filter on the random bar: -0.10 · p 0.01 |
| ③ 2026 (H1+H2) | 1118 · 62 % · +0.77 [+0.22, +1.28] · $+1,628 · -15.6 · $-402 | 483 · 67 % · +1.61 [+0.80, +2.38] · $+1,476 · -13.9 · $-174 | 518 · 58 % · +0.16 | +5.1 pp | +0.23 · 0.00 · same filter on the random bar: +0.37 · p 0.01 |

## exit tp8|72h|stop15 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · +1.64 [-4.64, +7.43] · $+41 · -16.1 · $-88 | 7 · 71 % · +0.79 [-12.32, +7.68] · $+11 · -16.1 · $-59 | 6 · 83 % · +2.63 | -5.5 pp | – |
| ② master (real eps) | 38 · 63 % · +0.32 [-2.91, +3.89] · $+23 · -16.5 · $-135 | 23 · 65 % · +1.04 [-4.66, +5.51] · $+45 · -16.5 · $-88 | 15 · 60 % · -0.77 | +2.1 pp | – |
| ③ 2025 ⚠ | 1137 · 64 % · +0.52 [-0.17, +1.15] · $+1,120 · -21.2 · $-602 | 235 · 69 % · +1.12 [-0.28, +2.43] · $+498 · -20.7 · $-217 | 652 · 61 % · +0.05 | +5.1 pp | -0.16 · 0.02 · same filter on the random bar: -0.06 · p 0.08 |
| ③ 2026-H1 | 730 · 68 % · +1.08 [+0.29, +1.83] · $+1,493 · -21.0 · $-354 | 295 · 74 % · +1.91 [+0.78, +3.05] · $+1,071 · -21.0 · $-141 | 362 · 64 % · +0.44 | +6.0 pp | +0.54 · 0.03 · same filter on the random bar: +0.72 · p 0.06 |
| ③ 2026-H2 | 385 · 68 % · +0.66 [-0.43, +1.69] · $+481 · -17.1 · $-314 | 187 · 71 % · +1.25 [-0.11, +2.44] · $+443 · -17.1 · $-225 | 154 · 60 % · -0.69 | +3.1 pp | -0.30 · 0.03 · same filter on the random bar: -0.17 · p 0.06 |
| ③ 2026 (H1+H2) | 1115 · 68 % · +0.93 [+0.34, +1.53] · $+1,974 · -21.0 · $-354 | 482 · 73 % · +1.65 [+0.77, +2.49] · $+1,513 · -21.0 · $-225 | 516 · 63 % · +0.10 | +4.8 pp | +0.22 · 0.01 · same filter on the random bar: +0.36 · p 0.01 |

## exit tp8|72h|stop25 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · -0.66 [-9.64, +7.45] · $-16 · -26.1 · $-145 | 7 · 71 % · -2.06 [-20.89, +7.68] · $-27 · -26.1 · $-97 | 6 · 83 % · +0.97 | -5.5 pp | – |
| ② master (real eps) | 37 · 73 % · +2.10 [-1.87, +5.61] · $+147 · -25.9 · $-70 | 22 · 77 % · +2.83 [-3.24, +6.81] · $+118 · -25.9 · $-49 | 15 · 67 % · +1.02 | +4.3 pp | – |
| ③ 2025 ⚠ | 1128 · 69 % · +0.68 [-0.12, +1.44] · $+1,447 · -28.6 · $-614 | 234 · 74 % · +1.27 [-0.42, +2.76] · $+567 · -27.4 · $-223 | 647 · 64 % · -0.09 | +5.4 pp | -0.06 · 0.04 · same filter on the random bar: +0.05 · p 0.11 |
| ③ 2026-H1 | 725 · 72 % · +1.05 [+0.11, +1.99] · $+1,452 · -32.8 · $-395 | 295 · 81 % · +2.27 [+0.91, +3.50] · $+1,270 · -31.2 · $-210 | 357 · 66 % · +0.01 | +8.3 pp | +0.30 · 0.01 · same filter on the random bar: +0.50 · p 0.03 |
| ③ 2026-H2 | 381 · 72 % · +0.46 [-0.80, +1.69] · $+334 · -33.5 · $-333 | 185 · 77 % · +1.12 [-0.59, +2.68] · $+392 · -28.5 · $-208 | 153 · 63 % · -1.12 | +4.9 pp | -0.64 · 0.08 · same filter on the random bar: -0.63 · p 0.09 |
| ③ 2026 (H1+H2) | 1106 · 72 % · +0.85 [+0.09, +1.58] · $+1,785 · -33.5 · $-395 | 480 · 79 % · +1.82 [+0.75, +2.86] · $+1,662 · -31.2 · $-210 | 510 · 65 % · -0.33 | +7.0 pp | -0.05 · 0.01 · same filter on the random bar: +0.04 · p 0.00 |

## exit tp10|72h|stop15 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 69 % · +0.88 [-8.70, +9.10] · $+22 · -16.1 · $-117 | 7 · 71 % · +2.08 [-12.04, +9.65] · $+28 · -16.1 · $-59 | 6 · 67 % · -0.53 | +2.2 pp | – |
| ② master (real eps) | 38 · 61 % · +0.78 [-2.62, +5.37] · $+56 · -16.5 · $-168 | 23 · 61 % · +1.02 [-4.26, +5.56] · $+45 · -16.5 · $-84 | 15 · 60 % · +0.41 | +0.3 pp | – |
| ③ 2025 ⚠ | 1136 · 60 % · +0.46 [-0.28, +1.18] · $+1,003 · -25.5 · $-668 | 235 · 63 % · +0.51 [-0.99, +1.94] · $+230 · -25.5 · $-249 | 651 · 58 % · +0.23 | +2.6 pp | -0.25 · 0.12 · same filter on the random bar: -0.05 · p 0.27 |
| ③ 2026-H1 | 728 · 64 % · +0.93 [+0.06, +1.79] · $+1,287 · -21.9 · $-342 | 295 · 69 % · +1.63 [+0.25, +2.98] · $+916 · -21.0 · $-229 | 360 · 61 % · +0.45 | +5.1 pp | +0.63 · 0.07 · same filter on the random bar: +0.83 · p 0.14 |
| ③ 2026-H2 | 385 · 62 % · +0.51 [-0.75, +1.70] · $+372 · -17.1 · $-440 | 187 · 65 % · +1.06 [-0.51, +2.51] · $+377 · -17.1 · $-193 | 154 · 56 % · -0.95 | +2.4 pp | -0.44 · 0.06 · same filter on the random bar: -0.31 · p 0.07 |
| ③ 2026 (H1+H2) | 1113 · 63 % · +0.78 [+0.07, +1.46] · $+1,659 · -21.9 · $-440 | 482 · 67 % · +1.41 [+0.42, +2.35] · $+1,293 · -21.0 · $-229 | 514 · 59 % · +0.03 | +4.0 pp | +0.22 · 0.01 · same filter on the random bar: +0.36 · p 0.04 |

EXPECTANCY BAR (discovery A (2025 + 26-H1)) on the BLOCKED cohort: N 1014 · windows 404 · WR 62.0 % vs breakeven 64.1 % (kept avg win +7.03 / avg loss -12.58) · avg +0.19 · P(avg < 0) 0.298 · top pair share of loss 2 % (LPTUSDT) · top window share 2 % → FAIL · kept-vs-all Δ avg +0.82 → after 40 % haircut +0.49

EXPECTANCY BAR (confirmation (26-H2 + ② + ①)) on the BLOCKED cohort: N 175 · windows 75 · WR 60.6 % vs breakeven 64.0 % (kept avg win +7.18 / avg loss -12.76) · avg -0.58 · P(avg < 0) 0.741 · top pair share of loss 4 % (PIEVERSEUSDT) · top window share 9 % → FAIL · kept-vs-all Δ avg +0.55 → after 40 % haircut +0.33

EXPECTANCY BAR (all periods) on the BLOCKED cohort: N 1189 · windows 479 · WR 61.8 % vs breakeven 64.1 % (kept avg win +7.08 / avg loss -12.63) · avg +0.07 · P(avg < 0) 0.413 · top pair share of loss 1 % (ARKUSDT) · top window share 2 % → FAIL · kept-vs-all Δ avg +0.74 → after 40 % haircut +0.44
kept cohort (③): 717 trades · 271 pairs · top pair share of profit 8 % (RAREUSDT) · top day share 6 % · best 5 % removed avg +1.13
per half-year (kept WR / avg vs blocked WR / avg): 2025: 69 % +1.12 vs 61 % +0.05 · 2026H1: 74 % +1.91 vs 64 % +0.44 · 2026H2: 71 % +1.25 vs 60 % -0.69

# CANDIDATE: KEEP turnover > 0.366 (block the low tercile)

## exit tp8|72h|stop5 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 54 % · +1.45 [-1.31, +4.25] · $+36 · -5.4 · $-30 | 9 · 56 % · +1.78 [-1.05, +5.36] · $+31 · -5.4 · $-30 | 4 · 50 % · +0.71 | +1.7 pp | – |
| ② master (real eps) | 38 · 32 % · -1.24 [-3.30, +0.77] · $-89 · -5.6 · $-133 | 31 · 32 % · -1.18 [-3.41, +1.02] · $-70 · -5.6 · $-94 | 7 · 29 % · -1.46 | +0.7 pp | – |
| ③ 2025 ⚠ | 1144 · 44 % · +0.31 [-0.11, +0.73] · $+676 · -7.2 · $-402 | 535 · 45 % · +0.37 [-0.19, +0.89] · $+380 · -7.0 · $-333 | 358 · 44 % · +0.21 | +0.5 pp | +0.04 · 0.14 |
| ③ 2026-H1 | 734 · 43 % · +0.06 [-0.45, +0.55] · $+84 · -8.2 · $-430 | 503 · 46 % · +0.39 [-0.18, +0.98] · $+373 · -7.6 · $-209 | 158 · 43 % · -0.05 | +2.5 pp | +0.20 · 0.30 |
| ③ 2026-H2 | 385 · 43 % · +0.13 [-0.63, +0.90] · $+97 · -8.9 · $-277 | 261 · 43 % · +0.14 [-0.68, +0.99] · $+70 · -8.9 · $-267 | 80 · 42 % · -0.14 | -0.2 pp | -0.15 · 0.28 |
| ③ 2026 (H1+H2) | 1119 · 43 % · +0.09 [-0.33, +0.50] · $+181 · -8.9 · $-576 | 764 · 45 % · +0.31 [-0.17, +0.80] · $+444 · -8.9 · $-292 | 238 · 43 % · -0.08 | +1.6 pp | +0.08 · 0.21 |

## exit tp8|72h|stop8 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 69 % · +2.51 [-3.26, +7.35] · $+62 · -8.7 · $-48 | 9 · 67 % · +2.19 [-3.68, +7.51] · $+38 · -8.7 · $-47 | 4 · 75 % · +3.21 | -2.6 pp | – |
| ② master (real eps) | 38 · 45 % · -1.05 [-4.14, +1.60] · $-76 · -9.3 · $-103 | 31 · 45 % · -0.99 [-4.42, +1.78] · $-58 · -9.3 · $-88 | 7 · 43 % · -1.31 | +0.4 pp | – |
| ③ 2025 ⚠ | 1143 · 55 % · +0.51 [-0.00, +1.01] · $+1,107 · -14.1 · $-552 | 535 · 56 % · +0.58 [-0.10, +1.22] · $+595 · -11.5 · $-444 | 358 · 55 % · +0.39 | +0.5 pp | +0.20 · 0.15 |
| ③ 2026-H1 | 734 · 55 % · +0.27 [-0.31, +0.83] · $+370 · -12.2 · $-369 | 503 · 59 % · +0.88 [+0.15, +1.61] · $+840 · -11.9 · $-247 | 158 · 49 % · -0.56 | +4.1 pp | +0.45 · 0.17 |
| ③ 2026-H2 | 385 · 57 % · +0.71 [-0.21, +1.60] · $+520 · -10.6 · $-278 | 261 · 59 % · +1.10 [+0.09, +2.05] · $+543 · -10.6 · $-198 | 80 · 49 % · -0.76 | +2.0 pp | -0.08 · 0.01 |
| ③ 2026 (H1+H2) | 1119 · 56 % · +0.42 [-0.07, +0.92] · $+890 · -12.2 · $-369 | 764 · 59 % · +0.95 [+0.36, +1.54] · $+1,383 · -11.9 · $-247 | 238 · 49 % · -0.63 | +3.4 pp | +0.27 · 0.02 |

## exit tp8|72h|stop10 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · +2.83 [-2.16, +7.42] · $+70 · -10.7 · $-59 | 9 · 67 % · +1.53 [-4.77, +7.51] · $+26 · -10.7 · $-59 | 4 · 100 % · +5.75 | -10.3 pp | – |
| ② master (real eps) | 38 · 50 % · -1.18 [-4.55, +1.96] · $-85 · -11.3 · $-137 | 31 · 52 % · -0.89 [-4.25, +2.22] · $-52 · -11.3 · $-118 | 7 · 43 % · -2.46 | +1.6 pp | – |
| ③ 2025 ⚠ | 1142 · 59 % · +0.50 [-0.05, +1.04] · $+1,082 · -16.1 · $-515 | 534 · 60 % · +0.54 [-0.22, +1.30] · $+549 · -13.8 · $-443 | 358 · 57 % · +0.25 | +0.7 pp | +0.17 · 0.18 |
| ③ 2026-H1 | 733 · 62 % · +0.77 [+0.14, +1.38] · $+1,073 · -15.6 · $-402 | 502 · 66 % · +1.42 [+0.64, +2.19] · $+1,350 · -13.9 · $-250 | 158 · 53 % · -0.48 | +4.2 pp | +0.55 · 0.04 |
| ③ 2026-H2 | 385 · 62 % · +0.76 [-0.24, +1.73] · $+555 · -12.2 · $-322 | 261 · 63 % · +1.01 [-0.05, +2.06] · $+499 · -12.2 · $-204 | 80 · 54 % · -0.49 | +1.3 pp | -0.16 · 0.02 |
| ③ 2026 (H1+H2) | 1118 · 62 % · +0.77 [+0.26, +1.29] · $+1,628 · -15.6 · $-402 | 763 · 65 % · +1.28 [+0.66, +1.88] · $+1,849 · -13.9 · $-250 | 238 · 53 % · -0.48 | +3.2 pp | +0.31 · 0.01 |

## exit tp8|72h|stop15 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · +1.64 [-5.13, +7.39] · $+41 · -16.1 · $-88 | 9 · 67 % · -0.18 [-7.87, +7.51] · $-3 · -16.1 · $-88 | 4 · 100 % · +5.75 | -10.3 pp | – |
| ② master (real eps) | 38 · 63 % · +0.32 [-2.80, +3.84] · $+23 · -16.5 · $-135 | 31 · 65 % · +0.46 [-2.98, +4.03] · $+27 · -16.5 · $-116 | 7 · 57 % · -0.30 | +1.4 pp | – |
| ③ 2025 ⚠ | 1137 · 64 % · +0.52 [-0.16, +1.16] · $+1,120 · -21.2 · $-602 | 529 · 65 % · +0.55 [-0.38, +1.44] · $+550 · -20.7 · $-490 | 358 · 60 % · +0.01 | +1.1 pp | +0.17 · 0.23 |
| ③ 2026-H1 | 730 · 68 % · +1.08 [+0.31, +1.81] · $+1,493 · -21.0 · $-354 | 500 · 72 % · +1.71 [+0.87, +2.54] · $+1,626 · -21.0 · $-150 | 157 · 57 % · -0.85 | +3.8 pp | +0.72 · 0.01 |
| ③ 2026-H2 | 385 · 68 % · +0.66 [-0.43, +1.68] · $+481 · -17.1 · $-314 | 261 · 69 % · +0.93 [-0.20, +2.08] · $+462 · -17.1 · $-281 | 80 · 55 % · -1.45 | +1.4 pp | -0.19 · 0.04 |
| ③ 2026 (H1+H2) | 1115 · 68 % · +0.93 [+0.33, +1.55] · $+1,974 · -21.0 · $-354 | 761 · 71 % · +1.44 [+0.71, +2.13] · $+2,088 · -21.0 · $-281 | 237 · 57 % · -1.05 | +3.0 pp | +0.41 · 0.01 |

## exit tp8|72h|stop25 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · -0.66 [-9.62, +7.42] · $-16 · -26.1 · $-145 | 9 · 67 % · -3.52 [-15.87, +7.51] · $-60 · -26.1 · $-145 | 4 · 100 % · +5.75 | -10.3 pp | – |
| ② master (real eps) | 37 · 73 % · +2.10 [-1.94, +5.55] · $+147 · -25.9 · $-70 | 30 · 77 % · +1.85 [-3.47, +5.85] · $+105 · -25.9 · $-85 | 7 · 57 % · +3.17 | +3.7 pp | – |
| ③ 2025 ⚠ | 1128 · 69 % · +0.68 [-0.12, +1.47] · $+1,447 · -28.6 · $-614 | 524 · 70 % · +0.75 [-0.32, +1.76] · $+742 · -28.5 · $-437 | 357 · 62 % · -0.42 | +1.7 pp | +0.34 · 0.24 |
| ③ 2026-H1 | 725 · 72 % · +1.05 [+0.12, +1.97] · $+1,452 · -32.8 · $-395 | 497 · 77 % · +1.84 [+0.78, +2.83] · $+1,738 · -31.2 · $-207 | 155 · 59 % · -1.56 | +4.4 pp | +0.62 · 0.02 |
| ③ 2026-H2 | 381 · 72 % · +0.46 [-0.83, +1.68] · $+334 · -33.5 · $-333 | 258 · 75 % · +0.77 [-0.74, +2.14] · $+376 · -33.5 · $-296 | 80 · 57 % · -2.03 | +2.4 pp | -0.45 · 0.10 |
| ③ 2026 (H1+H2) | 1106 · 72 % · +0.85 [+0.06, +1.61] · $+1,785 · -33.5 · $-395 | 755 · 76 % · +1.47 [+0.58, +2.31] · $+2,113 · -33.5 · $-296 | 235 · 58 % · -1.72 | +3.7 pp | +0.26 · 0.01 |

## exit tp10|72h|stop15 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 69 % · +0.88 [-8.70, +9.21] · $+22 · -16.1 · $-117 | 9 · 56 % · -1.74 [-12.86, +9.38] · $-30 · -16.1 · $-117 | 4 · 100 % · +6.76 | -13.7 pp | – |
| ② master (real eps) | 38 · 61 % · +0.78 [-2.68, +5.29] · $+56 · -16.5 · $-168 | 31 · 61 % · +0.76 [-2.93, +5.34] · $+45 · -16.5 · $-139 | 7 · 57 % · +0.84 | +0.8 pp | – |
| ③ 2025 ⚠ | 1136 · 60 % · +0.46 [-0.29, +1.22] · $+1,003 · -25.5 · $-668 | 528 · 61 % · +0.33 [-0.65, +1.30] · $+329 · -25.5 · $-508 | 358 · 58 % · +0.27 | +0.4 pp | +0.12 · 0.35 |
| ③ 2026-H1 | 728 · 64 % · +0.93 [+0.04, +1.76] · $+1,287 · -21.9 · $-342 | 499 · 67 % · +1.56 [+0.59, +2.53] · $+1,483 · -21.9 · $-222 | 156 · 56 % · -0.88 | +3.4 pp | +0.86 · 0.12 |
| ③ 2026-H2 | 385 · 62 % · +0.51 [-0.71, +1.65] · $+372 · -17.1 · $-440 | 261 · 63 % · +0.71 [-0.68, +2.07] · $+351 · -17.1 · $-261 | 80 · 52 % · -1.65 | +0.9 pp | -0.30 · 0.10 |
| ③ 2026 (H1+H2) | 1113 · 63 % · +0.78 [+0.10, +1.46] · $+1,659 · -21.9 · $-440 | 760 · 66 % · +1.27 [+0.45, +2.07] · $+1,833 · -21.9 · $-261 | 236 · 55 % · -1.14 | +2.5 pp | +0.46 · 0.04 |

EXPECTANCY BAR (discovery A (2025 + 26-H1)) on the BLOCKED cohort: N 515 · windows 287 · WR 59.2 % vs breakeven 62.5 % (kept avg win +6.84 / avg loss -11.41) · avg -0.25 · P(avg < 0) 0.712 · top pair share of loss 3 % (AKTUSDT) · top window share 2 % → FAIL · kept-vs-all Δ avg +0.38 → after 40 % haircut +0.23

EXPECTANCY BAR (confirmation (26-H2 + ② + ①)) on the BLOCKED cohort: N 91 · windows 48 · WR 57.1 % vs breakeven 64.1 % (kept avg win +7.07 / avg loss -12.65) · avg -1.05 · P(avg < 0) 0.793 · top pair share of loss 7 % (PIEVERSEUSDT) · top window share 10 % → FAIL · kept-vs-all Δ avg +0.19 → after 40 % haircut +0.11

EXPECTANCY BAR (all periods) on the BLOCKED cohort: N 606 · windows 335 · WR 58.9 % vs breakeven 62.9 % (kept avg win +6.90 / avg loss -11.69) · avg -0.37 · P(avg < 0) 0.815 · top pair share of loss 3 % (AKTUSDT) · top window share 2 % → FAIL · kept-vs-all Δ avg +0.33 → after 40 % haircut +0.20
kept cohort (③): 1290 trades · 346 pairs · top pair share of profit 6 % (RAREUSDT) · top day share 5 % · best 5 % removed avg +0.71
per half-year (kept WR / avg vs blocked WR / avg): 2025: 65 % +0.55 vs 60 % +0.01 · 2026H1: 72 % +1.71 vs 57 % -0.85 · 2026H2: 69 % +0.93 vs 55 % -1.45

# CANDIDATE: KEEP peak turnover > 1

## exit tp8|72h|stop5 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 54 % · +1.45 [-1.21, +4.22] · $+36 · -5.4 · $-30 | 7 · 57 % · +1.90 [-3.43, +6.86] · $+25 · -5.4 · $-20 | 6 · 50 % · +0.93 | +3.3 pp | – |
| ② master (real eps) | 38 · 32 % · -1.24 [-3.29, +0.71] · $-89 · -5.6 · $-133 | 23 · 39 % · -0.35 [-3.31, +2.08] · $-15 · -5.6 · $-67 | 15 · 20 % · -2.59 | +7.6 pp | – |
| ③ 2025 ⚠ | 1144 · 44 % · +0.31 [-0.12, +0.73] · $+676 · -7.2 · $-402 | 241 · 43 % · +0.20 [-0.61, +1.01] · $+94 · -6.8 · $-196 | 652 · 45 % · +0.35 | -1.4 pp | -0.09 · 0.20 |
| ③ 2026-H1 | 734 · 43 % · +0.06 [-0.44, +0.56] · $+84 · -8.2 · $-430 | 299 · 48 % · +0.69 [-0.12, +1.50] · $+391 · -6.7 · $-115 | 362 · 43 % · -0.05 | +4.6 pp | +0.16 · 0.14 |
| ③ 2026-H2 | 385 · 43 % · +0.13 [-0.65, +0.91] · $+97 · -8.9 · $-277 | 188 · 45 % · +0.47 [-0.54, +1.41] · $+169 · -8.9 · $-223 | 153 · 40 % · -0.41 | +2.1 pp | -0.20 · 0.15 |
| ③ 2026 (H1+H2) | 1119 · 43 % · +0.09 [-0.34, +0.53] · $+181 · -8.9 · $-576 | 487 · 47 % · +0.60 [-0.02, +1.25] · $+560 · -8.9 · $-226 | 515 · 42 % · -0.16 | +3.7 pp | +0.02 · 0.06 |

## exit tp8|72h|stop8 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 69 % · +2.51 [-3.21, +7.42] · $+62 · -8.7 · $-48 | 7 · 71 % · +2.85 [-6.14, +7.68] · $+38 · -8.7 · $-32 | 6 · 67 % · +2.10 | +2.2 pp | – |
| ② master (real eps) | 38 · 45 % · -1.05 [-3.96, +1.62] · $-76 · -9.3 · $-103 | 23 · 52 % · +0.15 [-3.84, +3.19] · $+6 · -9.3 · $-56 | 15 · 33 % · -2.89 | +7.4 pp | – |
| ③ 2025 ⚠ | 1143 · 55 % · +0.51 [+0.00, +1.00] · $+1,107 · -14.1 · $-552 | 241 · 56 % · +0.63 [-0.43, +1.63] · $+289 · -11.1 · $-200 | 652 · 55 % · +0.46 | +0.6 pp | -0.08 · 0.09 |
| ③ 2026-H1 | 734 · 55 % · +0.27 [-0.36, +0.85] · $+370 · -12.2 · $-369 | 299 · 60 % · +1.00 [+0.04, +1.96] · $+570 · -11.9 · $-149 | 362 · 54 % · +0.15 | +4.8 pp | +0.38 · 0.12 |
| ③ 2026-H2 | 385 · 57 % · +0.71 [-0.20, +1.63] · $+520 · -10.6 · $-278 | 188 · 63 % · +1.67 [+0.58, +2.74] · $+596 · -10.6 · $-164 | 153 · 50 % · -0.58 | +5.4 pp | -0.08 · 0.01 |
| ③ 2026 (H1+H2) | 1119 · 56 % · +0.42 [-0.10, +0.91] · $+890 · -12.2 · $-369 | 487 · 61 % · +1.26 [+0.50, +1.99] · $+1,166 · -11.9 · $-164 | 515 · 53 % · -0.07 | +5.1 pp | +0.21 · 0.01 |

## exit tp8|72h|stop10 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · +2.83 [-2.44, +7.39] · $+70 · -10.7 · $-59 | 7 · 71 % · +2.28 [-7.86, +7.68] · $+30 · -10.7 · $-40 | 6 · 83 % · +3.47 | -5.5 pp | – |
| ② master (real eps) | 38 · 50 % · -1.18 [-4.57, +2.02] · $-85 · -11.3 · $-137 | 23 · 57 % · +0.06 [-4.15, +3.47] · $+3 · -11.3 · $-87 | 15 · 40 % · -3.08 | +6.5 pp | – |
| ③ 2025 ⚠ | 1142 · 59 % · +0.50 [-0.05, +1.06] · $+1,082 · -16.1 · $-515 | 240 · 62 % · +0.88 [-0.25, +1.96] · $+400 · -13.1 · $-193 | 652 · 57 % · +0.26 | +3.5 pp | -0.12 · 0.04 |
| ③ 2026-H1 | 733 · 62 % · +0.77 [+0.13, +1.37] · $+1,073 · -15.6 · $-402 | 299 · 67 % · +1.65 [+0.59, +2.72] · $+936 · -13.9 · $-158 | 361 · 59 % · +0.40 | +5.7 pp | +0.49 · 0.01 |
| ③ 2026-H2 | 385 · 62 % · +0.76 [-0.24, +1.74] · $+555 · -12.2 · $-322 | 188 · 65 % · +1.49 [+0.28, +2.64] · $+532 · -12.2 · $-174 | 153 · 55 % · -0.37 | +3.9 pp | -0.15 · 0.01 |
| ③ 2026 (H1+H2) | 1118 · 62 % · +0.77 [+0.22, +1.29] · $+1,628 · -15.6 · $-402 | 487 · 67 % · +1.59 [+0.81, +2.34] · $+1,468 · -13.9 · $-174 | 514 · 58 % · +0.17 | +5.0 pp | +0.25 · 0.00 |

## exit tp8|72h|stop15 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · +1.64 [-5.05, +7.43] · $+41 · -16.1 · $-88 | 7 · 71 % · +0.79 [-12.32, +7.68] · $+11 · -16.1 · $-59 | 6 · 83 % · +2.63 | -5.5 pp | – |
| ② master (real eps) | 38 · 63 % · +0.32 [-2.87, +3.96] · $+23 · -16.5 · $-135 | 23 · 65 % · +1.04 [-4.61, +5.47] · $+45 · -16.5 · $-88 | 15 · 60 % · -0.77 | +2.1 pp | – |
| ③ 2025 ⚠ | 1137 · 64 % · +0.52 [-0.14, +1.15] · $+1,120 · -21.2 · $-602 | 240 · 69 % · +1.04 [-0.33, +2.36] · $+473 · -20.7 · $-217 | 647 · 61 % · +0.07 | +4.5 pp | -0.12 · 0.03 |
| ③ 2026-H1 | 730 · 68 % · +1.08 [+0.31, +1.81] · $+1,493 · -21.0 · $-354 | 298 · 74 % · +1.82 [+0.64, +2.97] · $+1,028 · -21.0 · $-149 | 359 · 64 % · +0.50 | +5.6 pp | +0.56 · 0.04 |
| ③ 2026-H2 | 385 · 68 % · +0.66 [-0.40, +1.66] · $+481 · -17.1 · $-314 | 188 · 71 % · +1.28 [-0.08, +2.52] · $+458 · -17.1 · $-225 | 153 · 59 % · -0.75 | +3.2 pp | -0.28 · 0.03 |
| ③ 2026 (H1+H2) | 1115 · 68 % · +0.93 [+0.31, +1.53] · $+1,974 · -21.0 · $-354 | 486 · 73 % · +1.61 [+0.72, +2.47] · $+1,486 · -21.0 · $-225 | 512 · 63 % · +0.13 | +4.7 pp | +0.24 · 0.01 |

## exit tp8|72h|stop25 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 77 % · -0.66 [-10.06, +7.43] · $-16 · -26.1 · $-145 | 7 · 71 % · -2.06 [-20.89, +7.68] · $-27 · -26.1 · $-97 | 6 · 83 % · +0.97 | -5.5 pp | – |
| ② master (real eps) | 37 · 73 % · +2.10 [-2.09, +5.57] · $+147 · -25.9 · $-70 | 22 · 77 % · +2.83 [-3.71, +6.67] · $+118 · -25.9 · $-49 | 15 · 67 % · +1.02 | +4.3 pp | – |
| ③ 2025 ⚠ | 1128 · 69 % · +0.68 [-0.11, +1.44] · $+1,447 · -28.6 · $-614 | 239 · 73 % · +1.18 [-0.43, +2.64] · $+535 · -27.4 · $-223 | 642 · 64 % · -0.07 | +4.7 pp | -0.02 · 0.06 |
| ③ 2026-H1 | 725 · 72 % · +1.05 [+0.08, +1.97] · $+1,452 · -32.8 · $-395 | 298 · 81 % · +2.21 [+0.82, +3.52] · $+1,252 · -31.2 · $-210 | 354 · 66 % · +0.04 | +8.1 pp | +0.31 · 0.01 |
| ③ 2026-H2 | 381 · 72 % · +0.46 [-0.82, +1.69] · $+334 · -33.5 · $-333 | 186 · 77 % · +1.15 [-0.54, +2.76] · $+407 · -28.5 · $-208 | 152 · 62 % · -1.18 | +5.0 pp | -0.62 · 0.07 |
| ③ 2026 (H1+H2) | 1106 · 72 % · +0.85 [+0.05, +1.59] · $+1,785 · -33.5 · $-395 | 484 · 79 % · +1.80 [+0.78, +2.84] · $+1,659 · -31.2 · $-210 | 506 · 65 % · -0.33 | +6.9 pp | -0.04 · 0.01 |

## exit tp10|72h|stop15 — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)
| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | WR lift | after vs fair random (kept episodes) · p |
|---|---|---|---|---|---|
| ① live (real eps, open = MTM) | 13 · 69 % · +0.88 [-8.58, +9.10] · $+22 · -16.1 · $-117 | 7 · 71 % · +2.08 [-12.04, +9.65] · $+28 · -16.1 · $-59 | 6 · 67 % · -0.53 | +2.2 pp | – |
| ② master (real eps) | 38 · 61 % · +0.78 [-2.69, +5.28] · $+56 · -16.5 · $-168 | 23 · 61 % · +1.02 [-4.28, +5.60] · $+45 · -16.5 · $-84 | 15 · 60 % · +0.41 | +0.3 pp | – |
| ③ 2025 ⚠ | 1136 · 60 % · +0.46 [-0.27, +1.18] · $+1,003 · -25.5 · $-668 | 240 · 62 % · +0.46 [-0.99, +1.89] · $+208 · -25.5 · $-249 | 646 · 58 % · +0.25 | +2.1 pp | -0.22 · 0.15 |
| ③ 2026-H1 | 728 · 64 % · +0.93 [+0.05, +1.79] · $+1,287 · -21.9 · $-342 | 298 · 68 % · +1.54 [+0.23, +2.84] · $+871 · -21.0 · $-258 | 357 · 61 % · +0.52 | +4.7 pp | +0.65 · 0.09 |
| ③ 2026-H2 | 385 · 62 % · +0.51 [-0.69, +1.72] · $+372 · -17.1 · $-440 | 188 · 65 % · +1.11 [-0.44, +2.57] · $+396 · -17.1 · $-192 | 153 · 56 % · -1.02 | +2.6 pp | -0.42 · 0.06 |
| ③ 2026 (H1+H2) | 1113 · 63 % · +0.78 [+0.09, +1.49] · $+1,659 · -21.9 · $-440 | 486 · 67 % · +1.37 [+0.39, +2.36] · $+1,267 · -21.0 · $-258 | 510 · 59 % · +0.06 | +3.8 pp | +0.25 · 0.02 |

EXPECTANCY BAR (discovery A (2025 + 26-H1)) on the BLOCKED cohort: N 1006 · windows 401 · WR 62.2 % vs breakeven 64.0 % (kept avg win +7.02 / avg loss -12.52) · avg +0.22 · P(avg < 0) 0.257 · top pair share of loss 2 % (LPTUSDT) · top window share 2 % → FAIL · kept-vs-all Δ avg +0.73 → after 40 % haircut +0.44

EXPECTANCY BAR (confirmation (26-H2 + ② + ①)) on the BLOCKED cohort: N 174 · windows 75 · WR 60.3 % vs breakeven 64.0 % (kept avg win +7.19 / avg loss -12.76) · avg -0.63 · P(avg < 0) 0.758 · top pair share of loss 4 % (PIEVERSEUSDT) · top window share 9 % → FAIL · kept-vs-all Δ avg +0.58 → after 40 % haircut +0.35

EXPECTANCY BAR (all periods) on the BLOCKED cohort: N 1180 · windows 476 · WR 61.9 % vs breakeven 64.0 % (kept avg win +7.07 / avg loss -12.59) · avg +0.10 · P(avg < 0) 0.385 · top pair share of loss 2 % (ARKUSDT) · top window share 2 % → FAIL · kept-vs-all Δ avg +0.68 → after 40 % haircut +0.41
kept cohort (③): 726 trades · 273 pairs · top pair share of profit 8 % (RAREUSDT) · top day share 6 % · best 5 % removed avg +1.07
per half-year (kept WR / avg vs blocked WR / avg): 2025: 69 % +1.04 vs 61 % +0.07 · 2026H1: 74 % +1.82 vs 64 % +0.50 · 2026H2: 71 % +1.28 vs 59 % -0.75

# Confound check (all periods ③ + ② + ①, main cell): WR / avg for turnover > 1 vs ≤ 1 inside terciles of a competing variable
| variable | tercile | N >1 · WR · avg | N ≤1 · WR · avg |
|---|---|---|---|
| mcap_musd | 0 (3.78–49.6) | 496 · 72 % · +1.68 | 164 · 64 % · +0.75 |
| mcap_musd | 1 (49.7–137) | 200 · 72 % · +1.54 | 460 · 63 % · +0.48 |
| mcap_musd | 2 (137–1.76e+05) | 69 · 67 % · +0.02 | 591 · 60 % · -0.45 |
| pump_pct | 0 (5.46–14.4) | 94 · 69 % · +1.21 | 566 · 59 % · -0.09 |
| pump_pct | 1 (14.4–25) | 196 · 66 % · +0.62 | 464 · 62 % · +0.08 |
| pump_pct | 2 (25–563) | 475 · 74 % · +1.91 | 185 · 66 % · +0.50 |
| vol_vs_prepump | 0 (0.247–3.05) | 113 · 65 % · +0.62 | 547 · 60 % · -0.24 |
| vol_vs_prepump | 1 (3.06–7.78) | 212 · 70 % · +1.37 | 448 · 60 % · -0.13 |
| vol_vs_prepump | 2 (7.79–772) | 440 · 74 % · +1.78 | 220 · 70 % · +1.22 |
| vol24_musd | 0 (20–34.9) | 154 · 71 % · +1.80 | 506 · 61 % · +0.18 |
| vol24_musd | 1 (34.9–82.1) | 231 · 73 % · +2.04 | 429 · 62 % · +0.32 |
| vol24_musd | 2 (82.4–7.99e+03) | 380 · 71 % · +1.04 | 280 · 62 % · -0.54 |
| coin_r24h | 0 (-46.7–7.34) | 155 · 65 % · +0.67 | 505 · 58 % · +0.15 |
| coin_r24h | 1 (7.35–15.7) | 204 · 70 % · +1.53 | 456 · 62 % · -0.33 |
| coin_r24h | 2 (15.8–139) | 406 · 75 % · +1.79 | 254 · 68 % · +0.60 |
| FLAT14 | 0 (0.762–25.4) | 233 · 70 % · +1.11 | 427 · 60 % · +0.15 |
| FLAT14 | 1 (25.4–34.3) | 274 · 75 % · +2.14 | 386 · 64 % · +0.37 |
| FLAT14 | 2 (34.3–44.4) | 258 · 69 % · +1.15 | 402 · 60 % · -0.33 |

# Mechanism (descriptive, look-ahead): exit reasons and hold-period BTC / funding by turnover side (main cell, all periods)
turnover > 1: N 765 · TP 67 % · stop 22 % · time 12 % · median BTC over hold -0.04 % · funding paid -0.406 % · median pump 31.5 % · median mcap $35M · median 24 h vol $81M
turnover ≤ 1: N 1215 · TP 48 % · stop 21 % · time 30 % · median BTC over hold -0.12 % · funding paid -0.315 % · median pump 15.1 % · median mcap $132M · median 24 h vol $40M
