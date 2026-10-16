# WILLY CONTINUATION — ADDENDUM A follow-up (post-hoc, declared)

## F1 — the Q1 survivor: market-wide red entry bar (≤ 30 % of the top-50 up on the last closed 5m bar) × coin EMA8–13 gap high

| era | n_zone | days_zone | pairs_zone | C2_30_zone | C2_30_rest | C3_90_zone | hold3_minus_bank_zone | y1_zone | y1_rest |
|---|---|---|---|---|---|---|---|---|---|
| 2025 | 11 | 11 | 10 | 1.000 | 0.766 | 0.909 | 0.594 | 0.750 | 0.063 |
| 2026-H1 | 25 | 25 | 23 | 0.960 | 0.783 | 0.760 | 0.861 | 0.723 | -0.010 |
| 2026-H2 | 10 | 10 | 10 | 1.000 | 0.696 | 0.900 | 1.030 | 0.750 | -0.337 |
| ALL | 46 | 46 | 42 | 0.978 | 0.758 | 0.826 | 0.834 | 0.735 | -0.069 |

Zone concentration: top pair 2/46 (STRKUSDT), top day 1/46; months 18; one-sided 95 % lower bound of C2_30 for 45/46 = 0.901
Leave-one-month-out: zone y1 (hold2 − bank) > 0 in 22/22 months left out (min +0.733)

Era-frozen cuts (lower tercile of e_br_up5, upper tercile of the gap, computed on the discovery era's Q1 fills only):

| discovery | cut_br | cut_gap | test | n_zone | C2_30_zone | C2_30_rest | y1_zone | delta_per_signal |
|---|---|---|---|---|---|---|---|---|
| 2025 | 0.229 | 1.367 | 2026-H1 | 15 | 0.933 | 0.791 | 0.705 | 0.027 |
| 2025 | 0.229 | 1.367 | 2026-H2 | 6 | 1.000 | 0.706 | 0.750 | 0.022 |
| 2026 | 0.319 | 1.418 | 2025 | 13 | 1.000 | 0.762 | 0.750 | 0.043 |
| 2026-H1 | 0.333 | 1.425 | 2025 | 14 | 1.000 | 0.760 | 0.750 | 0.046 |
| 2026-H1 | 0.333 | 1.425 | 2026-H2 | 9 | 1.000 | 0.698 | 0.750 | 0.033 |

Dose-response: C2_30 (share of +1.25 touches that reach +2 before minute 30) by terciles, seed-1 tick, all eras (N in brackets):

| breadth tercile | gap T1 | gap T2 | gap T3 (highest) |
|---|---|---|---|
| breadth T1 (lowest) | 0.781 (64) | 0.727 (66) | 0.978 (46) |
| breadth T2 | 0.696 (46) | 0.754 (65) | 0.769 (65) |
| breadth T3 | 0.727 (66) | 0.711 (45) | 0.864 (66) |

Same grid, y1 = HOLD2 − bank (%/touched fill):

| b3 | 0 | 1 | 2 |
|---|---|---|---|
| 0.000 | 0.278 | -0.377 | 0.735 |
| 1.000 | -0.115 | -0.179 | 0.027 |
| 2.000 | -0.068 | -0.203 | 0.038 |
breadth alone (≤ 0.2979): C2_30 0.818 (N 181) vs rest 0.756; y1 +0.169 vs -0.086
gap alone (> 1.393): C2_30 0.858 (N 176) vs rest 0.737; y1 +0.212 vs -0.105

1m ruler (seed 1 non-tick touched fills, refute-only, mostly 2025): zone C2_30 0.825 (N 80) vs rest 0.775; y1 zone +0.173 vs rest -0.039
Master + live touched fills: zone 3 → reached +2 by min 30: 3; rest C2_30 0.706 (N 17)

## F2 — Q2 COIN logistic: the largest standardised coefficients per training era (label r125_after30, N 185)

| var | 2025 | 2026 | same_sign | min_abs |
|---|---|---|---|---|
| cl30 | 1.600 | 0.952 | True | 0.952 |
| entry_pattern_w1_match | -0.824 | -0.871 | True | 0.824 |
| entry_cmc_rank | 0.527 | 0.598 | True | 0.527 |
| entry_pair_1h_ema20_200_gap_pct | 0.495 | 0.532 | True | 0.495 |
| entry_mcap_usd | 0.461 | 1.138 | True | 0.461 |
| entry_pattern_c1_match | 0.558 | 0.364 | True | 0.364 |
| entry_pattern_c6_match | -0.326 | -0.843 | True | 0.326 |
| entry_frenzy_bar_ret_pct | 0.509 | 0.323 | True | 0.323 |
| entry_rsi | 0.321 | 0.584 | True | 0.321 |
| entry_ema20_slope | -0.308 | -0.333 | True | 0.308 |

Top same-sign variable: cl30 (sign +). Single-variable rule, cut = discovery-era median:
| discovery | cut | test_N | lab_good | lab_bad | AUC_single | rule_minus_today | P_gt0 | kept | keep_good | keep_bad |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | -1.034 | 116 | 0.446 | 0.314 | 0.646 | -0.182 | 0.040 | 65 | -0.325 | 0.109 |
| 2026 | -0.787 | 69 | 0.600 | 0.205 | 0.763 | 0.148 | 0.919 | 30 | 0.341 | -0.092 |

Model-score rule (post-hoc): keep +1.25 at minute 30 iff the COIN-logit predicted P(R125) is in the training top third; else today's late rule:

| discovery | test_N | kept | R125_kept | R125_rest | keep_minus_today_kept | rule_minus_today_per_open30 | P_gt0 | per_signal |
|---|---|---|---|---|---|---|---|---|
| 2025 | 116 | 50 | 0.500 | 0.303 | -0.133 | -0.057 | 0.278 | -0.011 |
| 2026 | 69 | 15 | 0.667 | 0.296 | 0.477 | 0.104 | 0.962 | 0.031 |

Q2 economics by the state at minute 30 — what recovers is mostly decided by how far below the entry the coin sits:

| cl30b | N | R125 | today | keep | close30 |
|---|---|---|---|---|---|
| (-5.0, -2.0] | 40 | 0.200 | -2.235 | -2.333 | -2.712 |
| (-2.0, -1.0] | 47 | 0.340 | -1.366 | -1.240 | -1.449 |
| (-1.0, -0.25] | 44 | 0.341 | -0.627 | -1.087 | -0.605 |
| (-0.25, 0.25] | 29 | 0.483 | -0.194 | -0.004 | 0.020 |
| (0.25, 5.0] | 25 | 0.720 | 0.468 | 0.621 | 0.468 |

## F2 — Q3 COIN logistic: the largest standardised coefficients per training era (label C2_90, N 823)

| var | 2025 | 2026 | same_sign | min_abs |
|---|---|---|---|---|
| entry_frenzy_hours | -0.549 | -0.335 | True | 0.335 |
| entry_pos_di | -0.325 | -0.343 | True | 0.325 |
| entry_range_position | -0.253 | -0.247 | True | 0.247 |
| entry_pattern_c6_match | -0.190 | -0.252 | True | 0.190 |
| entry_frenzy_above_share | 0.187 | 0.173 | True | 0.173 |
| entry_frenzy_willy_wait_bars | -0.308 | -0.162 | True | 0.162 |
| entry_pattern_c5_match | -0.665 | -0.155 | True | 0.155 |
| entry_pattern_w1_match | -0.300 | -0.150 | True | 0.150 |
| entry_pattern_c_any_match | 0.369 | 0.146 | True | 0.146 |
| entry_frenzy_stop_atr | -0.292 | -0.130 | True | 0.130 |

Top same-sign variable: entry_frenzy_hours (sign −). Single-variable rule, cut = discovery-era median:
| discovery | cut | test_N | lab_good | lab_bad | AUC_single |
|---|---|---|---|---|---|
| 2025 | 0.170 | 595 | 0.630 | 0.541 | 0.552 |
| 2026 | 0.170 | 228 | 0.649 | 0.543 | 0.579 |
