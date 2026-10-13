
## FRENZY_WILLY

### ① Exit mix × regime × top-50 (seed 1)

| regime | top50 | N | WR | avg | avg_clipped | TP_share | stop_share | cap_share | stop_avg | TP_avg | cap_avg | avg_win | avg_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 BULL | False | 669 | 67.414 | -0.323 | -0.324 | 66.667 | 26.906 | 6.428 | -4.325 | 1.393 | -1.375 | 1.381 | -3.849 |
| 2025 BULL | True | 230 | 72.174 | 0.011 | 0.007 | 70.435 | 18.696 | 10.870 | -4.009 | 1.265 | -1.195 | 1.254 | -3.211 |
| 2026-H1 BEAR | False | 253 | 66.008 | 0.176 | 0.140 | 66.008 | 16.206 | 17.787 | -4.011 | 1.316 | -0.241 | 1.316 | -2.039 |
| 2026-H1 BEAR | True | 158 | 65.823 | -0.017 | -0.018 | 65.823 | 17.722 | 16.456 | -4.017 | 1.262 | -0.826 | 1.262 | -2.481 |
| 2026-H2 BULL | False | 125 | 65.600 | 0.102 | 0.086 | 65.600 | 16.000 | 18.400 | -4.016 | 1.284 | -0.534 | 1.284 | -2.153 |
| 2026-H2 BULL | True | 83 | 73.494 | 0.007 | 0.005 | 71.084 | 20.482 | 8.434 | -4.009 | 1.262 | -0.816 | 1.231 | -3.387 |

### ② Composition: 2025 avg re-weighted to 2026's cell mix (cells = top-50 × BTC-1h-slope sign)

| cells | avg_2025 | avg_2025_reweighted_to_2026_mix | avg_2026 | share_gap_explained_by_mix | cell_means_2025 | cell_means_2026 |
|---|---|---|---|---|---|---|
| top50 | -0.238 | -0.193 | 0.089 | 0.137 | False:-0.32(74%) · True:+0.01(26%) | False:+0.15(61%) · True:-0.01(39%) |
| top50 × slope_pos | -0.238 | -0.211 | 0.089 | 0.082 | (False, False):-0.56(32%) · (False, True):-0.15(43%) · (True, False):-0.03(9%) · (True, True):+0.03(17%) | (False, False):+0.01(30%) · (False, True):+0.29(31%) · (True, False):-0.04(18%) · (True, True):+0.02(21%) |
| slope_pos | -0.238 | -0.261 | 0.089 | -0.071 | False:-0.44(41%) · True:-0.10(59%) | False:-0.01(47%) · True:+0.18(53%) |

### ③ 2025 loss concentration

2025 fills 899 on 246 pairs, 295 days. Sum y -213.6 %-pts. Top-2 pairs carry 6.5 % of the gross loss; top-10 pairs 19.4 %. Worst 8 days sum -121.7 %-pts (57 % of the net). Without the worst 8 days: avg -0.107; without the worst 10 pairs (by net): -0.062.

Worst pairs (net %-pts): SAHARAUSDT -42.6 (2) · CATIUSDT -19.5 (7) · REZUSDT -16.8 (14) · WUSDT -16.0 (8) · MAVUSDT -14.5 (7) · VANAUSDT -12.0 (3) · API3USDT -10.4 (11) · WALUSDT -10.3 (3)

Worst days (net %-pts): 2025-11-29 -39.0 · 2025-08-27 -16.1 · 2025-10-10 -12.4 · 2025-01-13 -12.1 · 2025-11-04 -10.9 · 2025-07-27 -10.9 · 2025-11-14 -10.3 · 2025-05-10 -10.0


### ④ Stamp distribution shift — median per regime; KS 2025 vs 2026 (top 12 by KS)

| col | kind | med_25 | med_26H1 | med_26H2 | KS |
|---|---|---|---|---|---|
| entry_slippage_pct | pair | 0.000 | 0.000 | 0.013 | 0.352 |
| entry_mcap_usd | pair | 214735558.445 | 115377670.928 | 144777914.171 | 0.222 |
| entry_pos_di | pair | 41.250 | 36.504 | 37.433 | 0.203 |
| entry_bear_pct | MACRO | 31.900 | 40.400 | 34.400 | 0.190 |
| entry_cmc_rank | pair | 367.000 | 334.000 | 146.000 | 0.180 |
| entry_gap | pair | 3.545 | 2.901 | 3.046 | 0.171 |
| entry_ema_gap_5_8 | pair | 0.965 | 0.776 | 0.837 | 0.170 |
| entry_neg_di | pair | 7.129 | 9.805 | 8.924 | 0.169 |
| entry_dist_from_ema13_pct | pair | 2.864 | 2.182 | 2.498 | 0.167 |
| entry_ema_gap_8_13 | pair | 1.339 | 1.110 | 1.141 | 0.164 |
| entry_ema20_slope | pair | 1.663 | 1.365 | 1.385 | 0.160 |
| entry_bull_pct | MACRO | 50.000 | 38.600 | 44.400 | 0.154 |

### ⑤ Month × BTC-1h-slope sign at entry (avg % (N)) — does the bad-month set track the share of falling-BTC entries?

| month | N | avg | share_slope_le0 | slope_pos | slope_le0 |
|---|---|---|---|---|---|
| 2025-01 | 37 | -0.84 | 37.84 | -0.53 (23) | -1.35 (14) |
| 2025-02 | 43 | 0.07 | 55.81 | -0.30 (19) | +0.36 (24) |
| 2025-03 | 49 | -0.15 | 30.61 | +0.05 (34) | -0.62 (15) |
| 2025-04 | 75 | 0.05 | 37.33 | -0.06 (47) | +0.22 (28) |
| 2025-05 | 80 | -0.12 | 37.50 | +0.11 (50) | -0.51 (30) |
| 2025-06 | 49 | 0.23 | 24.49 | +0.18 (37) | +0.39 (12) |
| 2025-07 | 108 | -0.11 | 27.78 | -0.13 (78) | -0.06 (30) |
| 2025-08 | 100 | -0.32 | 48.00 | +0.07 (52) | -0.75 (48) |
| 2025-09 | 100 | -0.41 | 48.00 | -0.55 (52) | -0.26 (48) |
| 2025-10 | 106 | 0.16 | 39.62 | +0.52 (64) | -0.38 (42) |
| 2025-11 | 125 | -0.87 | 50.40 | -0.56 (62) | -1.18 (63) |
| 2025-12 | 24 | 0.05 | 45.83 | -0.20 (13) | +0.35 (11) |
| 2026-01 | 66 | -0.06 | 62.12 | -0.07 (25) | -0.05 (41) |
| 2026-02 | 61 | -0.26 | 60.66 | -0.66 (24) | -0.01 (37) |
| 2026-03 | 55 | -0.13 | 50.91 | -0.31 (27) | +0.04 (28) |
| 2026-04 | 77 | 0.26 | 44.16 | +0.22 (43) | +0.30 (34) |
| 2026-05 | 102 | 0.25 | 43.14 | +0.52 (58) | -0.12 (44) |
| 2026-06 | 53 | 0.32 | 64.15 | +0.36 (19) | +0.29 (34) |
| 2026-07 | 36 | 0.19 | 47.22 | +0.71 (19) | -0.40 (17) |
| 2026-08 | 51 | 0.28 | 31.37 | +0.28 (35) | +0.27 (16) |
| 2026-09 | 81 | -0.04 | 37.04 | -0.02 (51) | -0.07 (30) |
| 2026-10 | 40 | -0.12 | 30.00 | +0.28 (28) | -1.05 (12) |

Spearman(month avg, share of falling-BTC entries) = -0.10 over 22 months. Months with slope>0 cohort avg > 0: 11/22; slope≤0 cohort avg > 0: 8/22.


## FRENZY_LITE

### ① Exit mix × regime × top-50 (seed 1)

| regime | top50 | N | WR | avg | avg_clipped | TP_share | stop_share | cap_share | stop_avg | TP_avg | cap_avg | avg_win | avg_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025 BULL | False | 564 | 46.631 | -0.201 | -0.213 | 45.745 | 52.305 | 1.950 | -3.039 | 3.043 | -0.191 | 3.000 | -2.998 |
| 2025 BULL | True | 264 | 49.621 | -0.017 | -0.018 | 49.621 | 50.000 | 0.379 | -3.013 | 3.010 | -0.983 | 3.010 | -2.997 |
| 2026-H1 BEAR | False | 222 | 46.847 | 0.130 | 0.157 | 46.396 | 40.090 | 13.514 | -3.151 | 3.036 | -0.111 | 3.023 | -2.420 |
| 2026-H1 BEAR | True | 148 | 56.081 | 0.369 | 0.365 | 55.405 | 43.243 | 1.351 | -3.019 | 3.016 | 0.254 | 2.986 | -2.974 |
| 2026-H2 BULL | False | 140 | 45.000 | 0.037 | 0.033 | 45.000 | 42.857 | 12.143 | -3.111 | 3.078 | -0.118 | 3.078 | -2.450 |
| 2026-H2 BULL | True | 116 | 54.310 | 0.258 | 0.257 | 53.448 | 44.828 | 1.724 | -3.007 | 3.011 | -0.179 | 2.967 | -2.962 |

### ② Composition: 2025 avg re-weighted to 2026's cell mix (cells = top-50 × BTC-1h-slope sign)

| cells | avg_2025 | avg_2025_reweighted_to_2026_mix | avg_2026 | share_gap_explained_by_mix | cell_means_2025 | cell_means_2026 |
|---|---|---|---|---|---|---|
| top50 | -0.142 | -0.123 | 0.190 | 0.057 | False:-0.20(68%) · True:-0.02(32%) | False:+0.09(58%) · True:+0.32(42%) |
| top50 × slope_pos | -0.142 | -0.173 | 0.190 | -0.094 | (False, False):-0.84(20%) · (False, True):+0.07(48%) · (True, False):-0.34(10%) · (True, True):+0.12(22%) | (False, False):-0.02(22%) · (False, True):+0.17(36%) · (True, False):+0.07(15%) · (True, True):+0.45(28%) |
| slope_pos | -0.142 | -0.191 | 0.190 | -0.147 | False:-0.67(30%) · True:+0.09(70%) | False:+0.01(37%) · True:+0.29(63%) |

### ③ 2025 loss concentration

2025 fills 828 on 298 pairs, 279 days. Sum y -117.7 %-pts. Top-2 pairs carry 3.0 % of the gross loss; top-10 pairs 12.6 %. Worst 8 days sum -103.0 %-pts (87 % of the net). Without the worst 8 days: avg -0.019; without the worst 10 pairs (by net): +0.008.

Worst pairs (net %-pts): ARPAUSDT -15.2 (5) · IOSTUSDT -15.2 (5) · FIDAUSDT -15.1 (5) · NILUSDT -12.2 (4) · CYBERUSDT -12.2 (4) · ICNTUSDT -12.1 (4) · BIOUSDT -12.0 (10) · CAKEUSDT -11.9 (8)

Worst days (net %-pts): 2025-10-07 -18.1 · 2025-10-06 -12.3 · 2025-10-18 -12.2 · 2025-10-14 -12.1 · 2025-09-17 -12.1 · 2025-02-13 -12.1 · 2025-04-26 -12.1 · 2025-11-06 -12.0


### ④ Stamp distribution shift — median per regime; KS 2025 vs 2026 (top 12 by KS)

| col | kind | med_25 | med_26H1 | med_26H2 | KS |
|---|---|---|---|---|---|
| entry_slippage_pct | pair | 0.000 | 0.000 | 0.000 | 0.321 |
| entry_mcap_usd | pair | 179126178.769 | 73845295.974 | 71798654.972 | 0.287 |
| entry_bear_pct | MACRO | 26.100 | 30.750 | 31.850 | 0.176 |
| entry_pair_volume_24h_usd | pair | 92140857.340 | 63106044.850 | 69176121.320 | 0.147 |
| entry_bull_pct | MACRO | 57.100 | 46.800 | 46.700 | 0.144 |
| entry_btc_1h_slope | MACRO | 0.083 | 0.052 | 0.053 | 0.125 |
| entry_btc_off30d_high_pct | MACRO | -6.050 | -8.405 | -3.730 | 0.125 |
| entry_adx | pair | 29.511 | 27.922 | 25.006 | 0.114 |
| entry_vol_mcap_ratio | pair | 0.605 | 0.780 | 0.846 | 0.113 |
| entry_adx_prev | pair | 29.670 | 28.106 | 24.909 | 0.111 |
| entry_btc_ema50_100_gap_pct | MACRO | 0.072 | 0.041 | 0.034 | 0.108 |
| entry_cmc_rank | pair | 451.000 | 370.000 | 297.000 | 0.107 |

### ⑤ Month × BTC-1h-slope sign at entry (avg % (N)) — does the bad-month set track the share of falling-BTC entries?

| month | N | avg | share_slope_le0 | slope_pos | slope_le0 |
|---|---|---|---|---|---|
| 2025-01 | 28 | 1.00 | 21.43 | +1.01 (22) | +0.99 (6) |
| 2025-02 | 55 | -0.60 | 20.00 | -0.27 (44) | -1.92 (11) |
| 2025-03 | 51 | -0.17 | 27.45 | -0.08 (37) | -0.43 (14) |
| 2025-04 | 91 | -0.22 | 26.37 | -0.09 (67) | -0.56 (24) |
| 2025-05 | 79 | 0.19 | 18.99 | +0.29 (64) | -0.21 (15) |
| 2025-06 | 53 | 0.92 | 24.53 | +0.68 (40) | +1.63 (13) |
| 2025-07 | 114 | -0.27 | 34.21 | -0.07 (75) | -0.64 (39) |
| 2025-08 | 74 | -0.24 | 36.49 | +0.06 (47) | -0.78 (27) |
| 2025-09 | 76 | -0.40 | 35.53 | -0.31 (49) | -0.57 (27) |
| 2025-10 | 65 | -0.51 | 26.15 | +0.38 (48) | -3.03 (17) |
| 2025-11 | 98 | -0.15 | 39.80 | +0.11 (59) | -0.54 (39) |
| 2025-12 | 42 | -0.50 | 42.86 | -0.12 (24) | -1.01 (18) |
| 2026-01 | 44 | 0.41 | 36.36 | +0.42 (28) | +0.38 (16) |
| 2026-02 | 63 | -0.08 | 41.27 | +0.27 (37) | -0.59 (26) |
| 2026-03 | 60 | 0.66 | 36.67 | +0.64 (38) | +0.69 (22) |
| 2026-04 | 75 | 0.10 | 37.33 | -0.09 (47) | +0.43 (28) |
| 2026-05 | 75 | 0.40 | 40.00 | +0.31 (45) | +0.54 (30) |
| 2026-06 | 55 | -0.11 | 38.18 | -0.54 (34) | +0.57 (21) |
| 2026-07 | 68 | -0.31 | 23.53 | +0.06 (52) | -1.50 (16) |
| 2026-08 | 81 | 0.84 | 35.80 | +1.06 (52) | +0.45 (29) |
| 2026-09 | 74 | -0.28 | 47.30 | +0.43 (39) | -1.07 (35) |
| 2026-10 | 33 | 0.25 | 18.18 | +0.08 (27) | +1.00 (6) |

Spearman(month avg, share of falling-BTC entries) = -0.06 over 22 months. Months with slope>0 cohort avg > 0: 14/22; slope≤0 cohort avg > 0: 9/22.


## ⑥ Master real fills (MASTER_POOL_stacked.csv, stack_keep, stack_pct) — LITE / WILLY / LONG / WIDE BTC-1h-slope split

| sleeve | zone | N | days | WR | avg |
|---|---|---|---|---|---|
| FRENZY_LITE | btc_1h_slope > 0 | 13 | 4 | 53.846 | 0.225 |
| FRENZY_LITE | btc_1h_slope ≤ 0 | 3 | 2 | 66.667 | 0.998 |
| FRENZY_LITE | btc_off24h > −1.166 | 13 | 4 | 53.846 | 0.225 |
| FRENZY_LITE | btc_off24h ≤ −1.166 | 3 | 2 | 66.667 | 0.998 |
| FRENZY_LONG | btc_1h_slope > 0 | 14 | 5 | 71.429 | 0.213 |
| FRENZY_LONG | btc_1h_slope ≤ 0 | 6 | 4 | 66.667 | -0.004 |
| FRENZY_LONG | btc_off24h > −1.166 | 14 | 5 | 71.429 | 0.213 |
| FRENZY_LONG | btc_off24h ≤ −1.166 | 6 | 4 | 66.667 | -0.004 |
| FRENZY_WIDE | btc_1h_slope > 0 | 8 | 3 | 75.000 | 0.373 |
| FRENZY_WIDE | btc_1h_slope ≤ 0 | 0 | 0 |  |  |
| FRENZY_WIDE | btc_off24h > −1.166 | 8 | 3 | 75.000 | 0.373 |
| FRENZY_WIDE | btc_off24h ≤ −1.166 | 0 | 0 |  |  |
| FRENZY_WILLY | btc_1h_slope > 0 | 15 | 4 | 80.000 | 0.506 |
| FRENZY_WILLY | btc_1h_slope ≤ 0 | 2 | 1 | 50.000 | -1.387 |
| FRENZY_WILLY | btc_off24h > −1.166 | 16 | 4 | 75.000 | 0.223 |
| FRENZY_WILLY | btc_off24h ≤ −1.166 | 1 | 1 | 100.000 | 1.255 |
