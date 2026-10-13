### Screen size and null per sleeve × discovery regime

| sleeve | disc | family | tests | max_abs_t | null_q95 | survivors_p10 | survivors_p05 |
|---|---|---|---|---|---|---|---|
| FRENZY_LITE | 2025 BULL | MACRO(day) | 86 | 4.10 | 3.35 | 5 | 3 |
| FRENZY_LITE | 2025 BULL | PAIR | 2868 | 3.42 | 4.37 | 0 | 0 |
| FRENZY_LITE | 2026-H2 BULL | MACRO(day) | 86 | 3.50 | 3.60 | 1 | 0 |
| FRENZY_LITE | 2026-H2 BULL | PAIR | 2878 | 3.80 | 5.26 | 0 | 0 |
| FRENZY_LONG | 2025 BULL | MACRO(day) | 86 | 2.92 | 3.56 | 0 | 0 |
| FRENZY_LONG | 2025 BULL | PAIR | 2795 | 4.92 | 6.06 | 0 | 0 |
| FRENZY_LONG | 2026-H2 BULL | MACRO(day) | 86 | 2.65 | 3.57 | 0 | 0 |
| FRENZY_LONG | 2026-H2 BULL | PAIR | 2801 | 5.08 | 5.94 | 0 | 0 |
| FRENZY_WIDE | 2025 BULL | MACRO(day) | 86 | 1.70 | 3.49 | 0 | 0 |
| FRENZY_WIDE | 2025 BULL | PAIR | 2506 | 3.95 | 5.79 | 0 | 0 |
| FRENZY_WIDE | 2026-H2 BULL | MACRO(day) | 86 | 3.42 | 3.95 | 0 | 0 |
| FRENZY_WIDE | 2026-H2 BULL | PAIR | 1717 | 3.17 | 5.55 | 0 | 0 |
| FRENZY_WILLY | 2025 BULL | MACRO(day) | 86 | 2.60 | 3.21 | 0 | 0 |
| FRENZY_WILLY | 2025 BULL | PAIR | 2703 | 4.81 | 4.60 | 4 | 3 |
| FRENZY_WILLY | 2026-H2 BULL | MACRO(day) | 86 | 2.92 | 3.67 | 0 | 0 |
| FRENZY_WILLY | 2026-H2 BULL | PAIR | 2942 | 8.61 | 8.29 | 1 | 1 |


### Top 8 per sleeve × discovery regime × family (by |t|), with null-adjusted p


**FRENZY_LITE · discover 2025 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_btc_1h_slope > 0 | sign | 198 | 51.010 | 0.220 | -0.938 | 4.096 | 0.004 |
| entry_btc_1h_slope <= 0.01603 | lo3 | 93 | 32.258 | -0.825 | 0.238 | -3.947 | 0.008 |
| entry_btc_off24h_pct <= -1.166 | lo3 | 93 | 35.484 | -0.765 | 0.208 | -3.662 | 0.018 |
| entry_global_volume_ratio > 0.9669 | hi3 | 93 | 58.065 | 0.489 | -0.419 | 3.349 | 0.050 |
| entry_btc_trend_gap_pct > 0.0892 | med | 139 | 56.115 | 0.274 | -0.504 | 3.130 | 0.098 |
| entry_btc_rsi_1h <= 52.17 | lo3 | 93 | 34.409 | -0.685 | 0.168 | -3.113 | 0.110 |
| entry_btc_rsi_1h > 58.84 | hi3 | 93 | 55.914 | 0.428 | -0.388 | 3.083 | 0.120 |
| entry_btc_rsi_1h_prev > 54.87 | med | 139 | 52.518 | 0.259 | -0.488 | 3.002 | 0.150 |

**FRENZY_LITE · discover 2025 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_ema5_stretch <= 0.2324 & entry_pattern_c_any_match <= 0 | quad | 257 | 56.031 | 0.387 | -0.380 | 3.418 | 0.584 |
| entry_frenzy_vol_mult > 23.4 & entry_frenzy_adx_delta <= 0.147 | quad | 216 | 37.963 | -0.730 | 0.065 | -3.393 | 0.606 |
| entry_neg_di > 14.87 & entry_ema_gap_8_13 > 0.3009 | quad | 117 | 34.188 | -0.961 | -0.007 | -3.286 | 0.708 |
| entry_neg_di > 14.87 & entry_gap > 0.8736 | quad | 117 | 34.188 | -0.960 | -0.008 | -3.282 | 0.710 |
| entry_neg_di > 14.87 & entry_frenzy_vol_mult > 23.4 | quad | 194 | 37.629 | -0.750 | 0.044 | -3.272 | 0.722 |
| entry_adx_delta <= -0.249 & entry_frenzy_vol_mult > 23.4 | quad | 222 | 38.739 | -0.683 | 0.056 | -3.170 | 0.822 |
| entry_neg_di > 14.87 & entry_frenzy_above_share <= 55.09 | quad | 197 | 38.579 | -0.717 | 0.037 | -3.122 | 0.844 |
| entry_neg_di > 14.87 & entry_ema5_stretch > 0.2324 | quad | 236 | 39.407 | -0.646 | 0.059 | -3.086 | 0.864 |

**FRENZY_LITE · discover 2026-H2 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_btc_off24h_pct <= -1.217 | lo3 | 32 | 31.250 | -1.043 | 0.586 | -3.497 | 0.062 |
| entry_btc_1h_slope > 0 | sign | 64 | 51.562 | 0.360 | -0.627 | 2.008 | 0.876 |
| entry_btc_1d_ret_pct > 0 | sign | 55 | 54.545 | 0.422 | -0.492 | 1.942 | 0.896 |
| entry_btc_1h_slope <= 0.004107 | lo3 | 32 | 40.625 | -0.551 | 0.337 | -1.817 | 0.946 |
| entry_btc_ema20_slope > 0.03308 | hi3 | 32 | 31.250 | -0.526 | 0.324 | -1.799 | 0.950 |
| entry_btc_rsi_1h > 55.15 | med | 47 | 51.064 | 0.442 | -0.359 | 1.765 | 0.958 |
| entry_btc_1d_ret_pct <= -0.2246 | lo3 | 32 | 40.625 | -0.479 | 0.300 | -1.510 | 0.996 |
| entry_btc_atr_pct > 0.1473 | hi3 | 32 | 31.250 | -0.445 | 0.282 | -1.491 | 0.996 |

**FRENZY_LITE · discover 2026-H2 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| fz_age_min <= 425.2 & entry_pattern_w2_match > 0 | quad | 15 | 20.000 | -2.003 | 0.270 | -3.798 | 0.526 |
| entry_pair_volume_ratio > 0.6664 & entry_pattern_w1_match <= 0 | quad | 105 | 35.238 | -0.620 | 0.664 | -3.501 | 0.744 |
| entry_pair_volume_ratio > 0.6664 & entry_frenzy_hours <= 7.08 | quad | 70 | 32.857 | -0.864 | 0.514 | -3.443 | 0.772 |
| entry_frenzy_run_pct > 21.3 | hi3 | 85 | 64.706 | 0.997 | -0.290 | 3.396 | 0.790 |
| entry_pair_volume_ratio > 0.6664 & entry_atr_pct <= 1.637 | quad | 68 | 29.412 | -0.833 | 0.488 | -3.337 | 0.832 |
| entry_frenzy_run_pct <= 16.7 & entry_adx_delta <= -0.2513 | quad | 63 | 30.159 | -0.896 | 0.475 | -3.324 | 0.846 |
| entry_pair_volume_ratio > 0.6664 & fz_age_min <= 425.2 | quad | 69 | 33.333 | -0.833 | 0.495 | -3.294 | 0.852 |
| entry_frenzy_run_pct <= 16.7 & entry_frenzy_hours <= 7.08 | quad | 74 | 35.135 | -0.796 | 0.517 | -3.288 | 0.854 |

**FRENZY_LONG · discover 2025 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_bear_pct > 48.28 | hi3 | 50 | 68.000 | 0.509 | -0.324 | 2.923 | 0.228 |
| entry_bull_pct > 48.9 | med | 74 | 45.946 | -0.396 | 0.302 | -2.407 | 0.594 |
| entry_btc_above72_pct > 55 | hi3 | 49 | 63.265 | 0.414 | -0.270 | 2.339 | 0.648 |
| entry_bull_pct <= 38.38 | lo3 | 50 | 68.000 | 0.397 | -0.268 | 2.204 | 0.748 |
| entry_btc_above72_pct > 52.3 | med | 73 | 58.904 | 0.259 | -0.336 | 2.054 | 0.840 |
| entry_btc_dist_from_ema13_pct > 0.02965 | med | 74 | 47.297 | -0.343 | 0.250 | -2.037 | 0.856 |
| entry_btc_ema20_slope > 0.01325 | med | 74 | 48.649 | -0.315 | 0.223 | -1.843 | 0.938 |
| entry_btc_ema20_slope <= -0.00074 | lo3 | 50 | 60.000 | 0.306 | -0.222 | 1.804 | 0.948 |

**FRENZY_LONG · discover 2025 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_pattern_w1_match > 0 & entry_neg_di > 12.8 | quad | 18 | 94.444 | 1.250 | -0.155 | 4.919 | 0.170 |
| entry_pattern_w1_match > 0 & entry_frenzy_di_spread <= 17.37 | quad | 16 | 93.750 | 1.219 | -0.142 | 4.347 | 0.314 |
| entry_pattern_c_any_match > 0 & entry_pos_di > 30.45 | quad | 41 | 87.805 | 0.948 | -0.242 | 4.312 | 0.326 |
| entry_adx_delta > 0.2513 & entry_pattern_c_any_match > 0 | quad | 49 | 85.714 | 0.854 | -0.264 | 4.101 | 0.412 |
| entry_pattern_c_any_match > 0 & entry_frenzy_vs_vwap_pct > 5.011 | quad | 42 | 85.714 | 0.853 | -0.229 | 3.766 | 0.598 |
| entry_adx_delta > 0.2513 & entry_neg_di > 12.8 | quad | 37 | 86.486 | 0.891 | -0.212 | 3.749 | 0.612 |
| entry_pattern_c_any_match > 0 & entry_frenzy_di_spread > 17.37 | quad | 40 | 85.000 | 0.821 | -0.214 | 3.485 | 0.774 |
| entry_pattern_c_any_match > 0 & entry_rsi_prev > 68.18 | quad | 40 | 85.000 | 0.820 | -0.214 | 3.482 | 0.774 |

**FRENZY_LONG · discover 2026-H2 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_global_volume_ratio > 0.9373 | med | 33 | 36.364 | -0.864 | 0.352 | -2.649 | 0.406 |
| entry_global_volume_ratio <= 0.8173 | lo3 | 23 | 56.522 | 0.473 | -0.623 | 2.602 | 0.438 |
| entry_frenzy_gvol <= 0.845 | lo3 | 23 | 56.522 | 0.376 | -0.572 | 2.149 | 0.808 |
| entry_btc_trend_gap_pct > 0.07535 | med | 33 | 63.636 | 0.246 | -0.724 | 2.090 | 0.848 |
| entry_frenzy_gvol > 0.9435 | med | 33 | 45.455 | -0.715 | 0.208 | -1.966 | 0.912 |
| entry_eth_5m_ret1_pct > 0.01523 | med | 33 | 57.576 | 0.215 | -0.695 | 1.950 | 0.920 |
| entry_btc_1h_slope > 0.0443 | med | 33 | 51.515 | 0.201 | -0.681 | 1.895 | 0.944 |
| entry_btc_rsi_1h_prev > 54.5 | med | 33 | 51.515 | 0.201 | -0.681 | 1.894 | 0.944 |

**FRENZY_LONG · discover 2026-H2 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_frenzy_vol_mult <= 116.7 & entry_ema5_stretch <= 0.4738 | quad | 26 | 19.231 | -2.030 | 0.193 | -5.079 | 0.132 |
| entry_pattern_w1_match > 0 & entry_ema5_stretch > 0.4738 | quad | 20 | 90.000 | 1.049 | -0.528 | 3.942 | 0.608 |
| entry_frenzy_above_streak <= 19.5 & entry_rsi > 59.86 | quad | 28 | 85.714 | 0.855 | -0.602 | 3.607 | 0.776 |
| entry_frenzy_vol_mult > 116.7 & entry_ema5_stretch <= 0.4738 | quad | 36 | 83.333 | 0.748 | -0.691 | 3.598 | 0.778 |
| entry_frenzy_above_streak <= 19.5 & entry_ema_gap_5_8 > 0.3709 | quad | 27 | 85.185 | 0.832 | -0.581 | 3.442 | 0.836 |
| entry_frenzy_above_streak <= 19.5 & entry_rsi_prev > 67.5 | quad | 27 | 85.185 | 0.831 | -0.581 | 3.437 | 0.838 |
| entry_pair_volume_24h_usd > 8.305e+07 & entry_frenzy_hours <= 12.17 | quad | 17 | 88.235 | 0.967 | -0.470 | 3.261 | 0.906 |
| entry_frenzy_above_streak > 19.5 & entry_ema5_stretch <= 0.4738 | quad | 26 | 30.769 | -1.509 | 0.055 | -3.217 | 0.920 |

**FRENZY_WIDE · discover 2025 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_bear_pct > 44.73 | hi3 | 47 | 44.681 | -0.551 | 0.057 | -1.703 | 0.982 |
| entry_btc_rsi > 55.3 | med | 70 | 62.857 | 0.095 | -0.384 | 1.440 | 1.000 |
| entry_btc_dist_from_ema13_pct > 0.0377 | med | 70 | 62.857 | 0.089 | -0.377 | 1.400 | 1.000 |
| entry_btc_rsi_1h_prev <= 50.27 | lo3 | 47 | 48.936 | -0.485 | 0.024 | -1.394 | 1.000 |
| entry_btc_adx_prev <= 19.68 | lo3 | 47 | 65.957 | 0.175 | -0.306 | 1.385 | 1.000 |
| entry_btc_rsi_prev6 > 53.5 | med | 68 | 61.765 | 0.088 | -0.364 | 1.360 | 1.000 |
| entry_bull_pct <= 37.49 | lo3 | 47 | 48.936 | -0.470 | 0.016 | -1.344 | 1.000 |
| entry_btc_r72_pct > 1.603 | hi3 | 47 | 59.574 | 0.142 | -0.290 | 1.279 | 1.000 |

**FRENZY_WIDE · discover 2025 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_rsi > 73.69 & entry_frenzy_above_streak > 48 | quad | 41 | 85.366 | 0.821 | -0.415 | 3.947 | 0.550 |
| entry_ema_gap_5_8 > 0.913 & entry_frenzy_above_streak > 48 | quad | 42 | 83.333 | 0.730 | -0.399 | 3.514 | 0.760 |
| entry_ema_gap_8_13 > 1.088 & entry_frenzy_above_streak > 48 | quad | 40 | 82.500 | 0.690 | -0.375 | 3.214 | 0.898 |
| entry_ema20_slope > 1.384 & entry_frenzy_above_streak > 48 | quad | 39 | 82.051 | 0.670 | -0.364 | 3.074 | 0.942 |
| entry_gap > 2.99 & entry_frenzy_above_streak > 48 | quad | 39 | 82.051 | 0.669 | -0.364 | 3.069 | 0.944 |
| entry_rsi > 73.69 & entry_frenzy_above_share > 77.94 | quad | 42 | 80.952 | 0.619 | -0.370 | 2.966 | 0.954 |
| entry_rsi > 73.69 & entry_frenzy_run_pct > 36.85 | quad | 42 | 80.952 | 0.619 | -0.370 | 2.963 | 0.954 |
| entry_ema_gap_5_8 <= 0.913 & entry_pair_volume_ratio > 1.197 | quad | 26 | 38.462 | -1.344 | 0.003 | -2.935 | 0.962 |

**FRENZY_WIDE · discover 2026-H2 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_btc_off24h_pct > -0.5833 | hi3 | 17 | 76.471 | 0.771 | -0.867 | 3.422 | 0.132 |
| entry_btc_rsi_prev6 <= 46.37 | lo3 | 17 | 23.529 | -1.552 | 0.294 | -3.300 | 0.178 |
| entry_btc_rsi_closed > 57.26 | hi3 | 17 | 76.471 | 0.636 | -0.800 | 2.767 | 0.428 |
| entry_btc_rsi_prev > 57.84 | hi3 | 17 | 76.471 | 0.636 | -0.800 | 2.767 | 0.428 |
| entry_btc_1d_ret_pct > 0 | sign | 32 | 37.500 | -0.839 | 0.552 | -2.616 | 0.500 |
| entry_btc_ema20_slope > 0.02407 | hi3 | 17 | 70.588 | 0.571 | -0.767 | 2.546 | 0.556 |
| entry_btc_rsi_closed > 52.73 | med | 25 | 64.000 | 0.280 | -0.899 | 2.184 | 0.804 |
| entry_btc_1d_ret_pct <= -0.1438 | lo3 | 17 | 70.588 | 0.440 | -0.702 | 2.049 | 0.884 |

**FRENZY_WIDE · discover 2026-H2 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_ema20_slope <= 1.553 & entry_frenzy_above_streak <= 67 | quad | 16 | 87.500 | 0.934 | -0.581 | 3.169 | 0.890 |
| entry_frenzy_above_streak <= 67 & entry_neg_di > 10.06 | quad | 15 | 86.667 | 0.896 | -0.550 | 2.909 | 0.956 |
| entry_gap <= 3.376 & entry_frenzy_above_share > 79.81 | quad | 19 | 84.211 | 0.788 | -0.610 | 2.894 | 0.960 |
| entry_frenzy_above_share > 79.81 & entry_ema20_slope <= 1.553 | quad | 19 | 84.211 | 0.788 | -0.610 | 2.894 | 0.960 |
| entry_frenzy_stop_atr <= 1.409 & entry_pair_1h_ema20_200_gap_pct > 27.33 | quad | 18 | 33.333 | -1.503 | 0.067 | -2.707 | 0.986 |
| entry_frenzy_above_share > 79.81 & entry_rsi_prev <= 72.22 | quad | 18 | 83.333 | 0.748 | -0.576 | 2.661 | 0.986 |
| entry_ema_gap_5_8 <= 1.016 & entry_frenzy_above_streak <= 67 | quad | 18 | 83.333 | 0.746 | -0.576 | 2.652 | 0.986 |
| entry_long_heat_flags <= 0 | lo3 | 63 | 53.968 | -0.576 | 0.746 | -2.651 | 0.986 |

**FRENZY_WILLY · discover 2025 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_btc_rsi_1h_prev > 56.5 | hi3 | 98 | 62.245 | 0.070 | -0.507 | 2.601 | 0.318 |
| entry_btc_1h_slope > 0.1064 | hi3 | 98 | 61.224 | 0.053 | -0.499 | 2.484 | 0.394 |
| entry_btc_off30d_high_pct > -4.24 | hi3 | 97 | 59.794 | 0.019 | -0.480 | 2.286 | 0.578 |
| entry_btc_off30d_high_pct > -6.905 | med | 147 | 57.143 | -0.057 | -0.573 | 2.221 | 0.648 |
| entry_btc_off30d_high_pct <= -9.693 | lo3 | 99 | 43.434 | -0.723 | -0.110 | -2.156 | 0.690 |
| entry_btc_1d_ret_pct > 0.8107 | hi3 | 98 | 62.245 | -0.003 | -0.471 | 2.102 | 0.732 |
| entry_btc_1d_ret_pct > 0 | sign | 155 | 55.484 | -0.082 | -0.574 | 2.067 | 0.768 |
| entry_btc_1d_ret_pct > 0.0907 | med | 147 | 56.463 | -0.081 | -0.548 | 2.006 | 0.824 |

**FRENZY_WILLY · discover 2025 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_adx > 46.56 & entry_ema_gap_5_8 <= 0.9652 | quad | 178 | 80.337 | 0.446 | -0.406 | 4.813 | 0.030 |
| entry_frenzy_run_pct <= 10.86 & entry_ema50_slope > 2.587 | quad | 177 | 81.921 | 0.441 | -0.404 | 4.680 | 0.038 |
| entry_ema_gap_5_8 <= 0.9652 & entry_ema50_slope > 2.587 | quad | 140 | 83.571 | 0.516 | -0.377 | 4.673 | 0.040 |
| entry_ema_gap_5_8 <= 0.9652 & entry_adx_prev > 44.91 | quad | 187 | 79.144 | 0.391 | -0.403 | 4.459 | 0.094 |
| entry_ema_gap_5_8 <= 0.9652 & entry_pair_ema20_ema50_gap_pct > 4.186 | quad | 116 | 84.483 | 0.544 | -0.353 | 4.417 | 0.102 |
| entry_frenzy_run_pct <= 10.86 & entry_adx > 46.56 | quad | 214 | 78.037 | 0.338 | -0.417 | 4.363 | 0.112 |
| entry_frenzy_run_pct <= 10.86 & entry_adx_prev > 44.91 | quad | 218 | 77.523 | 0.311 | -0.413 | 4.162 | 0.176 |
| entry_ema50_slope > 2.587 & entry_dist_from_ema13_pct <= 2.864 | quad | 138 | 82.609 | 0.438 | -0.360 | 3.987 | 0.258 |

**FRENZY_WILLY · discover 2026-H2 BULL · MACRO(day)**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_btc_rsi_prev6 > 55.22 | hi3 | 28 | 75.000 | 0.647 | -0.208 | 2.918 | 0.284 |
| entry_btc_rsi_prev6 > 51.73 | med | 42 | 71.429 | 0.520 | -0.366 | 2.744 | 0.372 |
| entry_btc_trend_gap_pct <= -0.006424 | lo3 | 28 | 50.000 | -0.569 | 0.400 | -2.427 | 0.554 |
| entry_btc_atr_pct > 0.1566 | hi3 | 28 | 42.857 | -0.537 | 0.384 | -2.391 | 0.584 |
| entry_btc_trend_gap_pct > 0 | sign | 53 | 66.038 | 0.387 | -0.454 | 2.223 | 0.726 |
| entry_btc_atr_pct > 0.1287 | med | 42 | 47.619 | -0.245 | 0.399 | -1.953 | 0.900 |
| entry_eth_5m_ret1_pct > 5e-05 | med | 42 | 45.238 | -0.225 | 0.379 | -1.827 | 0.944 |
| entry_eth_5m_ret1_pct > 0 | sign | 42 | 45.238 | -0.225 | 0.379 | -1.827 | 0.944 |

**FRENZY_WILLY · discover 2026-H2 BULL · PAIR**

| zone | gran | n_zone | wr_zone | avg_zone | avg_rest | t | p_adj |
|---|---|---|---|---|---|---|---|
| entry_pattern_w2_match > 0 & entry_pair_ema20_ema50_gap_pct > 3.799 | quad | 17 | 100.000 | 1.261 | -0.043 | 8.612 | 0.020 |
| entry_pattern_w2_match > 0 & entry_adx > 42.86 | quad | 15 | 100.000 | 1.191 | -0.024 | 7.312 | 0.200 |
| entry_frenzy_vol_mult <= 39.25 & entry_pattern_w4_match > 0 | quad | 18 | 77.778 | 1.061 | -0.031 | 4.661 | 0.746 |
| entry_ema50_slope <= 2.331 & hour <= 11 | quad | 49 | 81.633 | 0.839 | -0.175 | 4.457 | 0.788 |
| entry_pattern_w2_match > 0 & entry_frenzy_vol_mult <= 39.25 | quad | 20 | 90.000 | 0.954 | -0.031 | 4.383 | 0.804 |
| fz_age_min <= 10.22 & entry_frenzy_above_streak > 0 | quad | 34 | 91.176 | 0.915 | -0.103 | 3.761 | 0.916 |
| entry_frenzy_above_streak > 0 & entry_vol_mcap_ratio > 0.2986 | quad | 46 | 86.957 | 0.816 | -0.150 | 3.701 | 0.924 |
| entry_pair_ema20_ema50_gap_pct <= 3.799 & hour <= 11 | quad | 50 | 80.000 | 0.741 | -0.151 | 3.635 | 0.940 |


### Survivors (p_adj < 0.10) → OOS confirmation in the other bull regime; bear H1 shown

| sleeve | disc | family | zone | gran | n_disc | avg_zone_disc | avg_rest_disc | t | p_adj | conf_n | conf_diff | conf_P | bear_n | bear_diff | bear_P | CONFIRMED | top50_n | top50_diff | top50_P |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FRENZY_LITE | 2025 BULL | MACRO(day) | entry_btc_1h_slope > 0 | sign | 198 | 0.220 | -0.938 | 4.096 | 0.004 | 64 | 0.987 | 0.975 | 91 | 0.287 | 0.750 | True |  |  |  |
| FRENZY_LITE | 2025 BULL | MACRO(day) | entry_btc_1h_slope <= 0.01603 | lo3 | 93 | -0.825 | 0.238 | -3.947 | 0.008 | 38 | -0.704 | 0.944 | 57 | -0.032 | 0.523 | True |  |  |  |
| FRENZY_LITE | 2025 BULL | MACRO(day) | entry_btc_off24h_pct <= -1.166 | lo3 | 93 | -0.765 | 0.208 | -3.662 | 0.018 | 33 | -1.608 | 1.000 | 70 | -0.012 | 0.519 | True |  |  |  |
| FRENZY_LITE | 2025 BULL | MACRO(day) | entry_global_volume_ratio > 0.9669 | hi3 | 93 | 0.489 | -0.419 | 3.349 | 0.050 | 34 | 0.445 | 0.815 | 53 | -0.188 | 0.303 | False |  |  |  |
| FRENZY_LITE | 2025 BULL | MACRO(day) | entry_btc_trend_gap_pct > 0.0892 | med | 139 | 0.274 | -0.504 | 3.130 | 0.098 | 43 | 0.036 | 0.536 | 65 | -0.321 | 0.211 | False |  |  |  |
| FRENZY_LITE | 2026-H2 BULL | MACRO(day) | entry_btc_off24h_pct <= -1.217 | lo3 | 32 | -1.043 | 0.586 | -3.497 | 0.062 | 86 | -1.001 | 1.000 | 68 | -0.000 | 0.505 | True |  |  |  |
| FRENZY_WILLY | 2025 BULL | PAIR | entry_adx > 46.56 & entry_ema_gap_5_8 <= 0.9652 | quad | 178 | 0.446 | -0.406 | 4.813 | 0.030 | 50 | 0.476 | 0.950 | 74 | -0.575 | 0.028 | True | 45.000 | 0.446 | 0.920 |
| FRENZY_WILLY | 2025 BULL | PAIR | entry_frenzy_run_pct <= 10.86 & entry_ema50_slope > 2.587 | quad | 177 | 0.441 | -0.404 | 4.680 | 0.038 | 48 | 0.073 | 0.609 | 67 | -0.012 | 0.481 | False | 49.000 | 0.935 | 0.999 |
| FRENZY_WILLY | 2025 BULL | PAIR | entry_ema_gap_5_8 <= 0.9652 & entry_ema50_slope > 2.587 | quad | 140 | 0.516 | -0.377 | 4.673 | 0.040 | 50 | 0.098 | 0.649 | 62 | -0.120 | 0.341 | False | 41.000 | 0.888 | 0.998 |
| FRENZY_WILLY | 2025 BULL | PAIR | entry_ema_gap_5_8 <= 0.9652 & entry_adx_prev > 44.91 | quad | 187 | 0.391 | -0.403 | 4.459 | 0.094 | 50 | 0.338 | 0.884 | 79 | -0.497 | 0.037 | False | 49.000 | 0.475 | 0.937 |
| FRENZY_WILLY | 2026-H2 BULL | PAIR | entry_pattern_w2_match > 0 & entry_pair_ema20_ema50_gap_pct > 3.799 | quad | 17 | 1.261 | -0.043 | 8.612 | 0.020 | 78 | 0.230 | 0.807 | 22 | -0.353 | 0.246 | False |  |  |  |
