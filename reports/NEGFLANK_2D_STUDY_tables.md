# NEGFLANK 2D — raw tables (2026-10-08 15:34)

yr5 NEGFLANK: 753 fills = 251/seed · 140 days · halves split at 2026-05-12 · master ex-B1 NEGFLANK 37 fills · 22 days
variables screened: 89 (stamped 47 · rebuilt 42); yr5-only stamps (master scored only): ['entry_btc_above72_pct', 'entry_btc_ema50_100_gap_pct', 'entry_eth_5m_ret1_pct', 'entry_btc_rsi_closed', 'entry_gap_5_8_signed_pct', 'entry_gap_5_20_signed_pct', 'entry_gap_5_20_prev_signed_pct', 'entry_pair_age_days']

## scan yr5_all: 17748 masks (1D 468 · 2D 17280) on 753 fills (251/seed, 140 days) · observed max |z| 3.66 · null 95th pct of max |z| 5.55 · P(null max ≥ observed max) 0.944 · reference trade-shuffle null (250×): 95th 5.71, P 0.968
- day-clustered z: obs max 3.66 vs null95 5.55 (p 0.944); masks |z|≥3: obs 23 vs null median 58 / 95th 227 (p 0.873)
- Welch z (trade SE) under the same day-block null: obs max 5.24 vs null95 6.41 (p 0.488); masks |z|≥3: obs 517 vs null median 340 / 95th 1012 (p 0.262)
survivors (either statistic): scan p<0.05 cl 0 / welch 0 · +halves 0 · +LOMO 0 · +master 0
masks passing halves ∧ LOMO ∧ master WITHOUT the scan correction: 3008 of 17748

(top 25 by day-clustered |z|; scan p = cl / welch)


| # | kind | mask | yr5 zone N/seed · avg | rest avg | Δ · z | scan p | H1Δ / H2Δ | LOMO | master zone N · avg · Δ · z | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2D | entry_gap_expand_marginal>0 ∧ k_btc_day_ret≤-0.6872 | 42 · -0.320 | -0.085 | -0.235 · -3.66 | 0.944 | -0.340 / -0.127 | 10/10 | 9 · +0.375 · +0.263 · +0.86 | HL |
| 2 | 2D | entry_btc_rsi_prev6≤56.2 ∧ entry_pair_rank≤29 | 65 · +0.067 | -0.191 | +0.259 · +3.58 | 0.966 | +0.197 / +0.330 | 10/10 | 13 · +0.295 · +0.183 · +0.61 | HLM |
| 3 | 2D | entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_btc_above7d_low>3.175 | 60 · -0.337 | -0.057 | -0.279 · -3.39 | 0.993 | -0.202 / -0.373 | 10/10 | 8 · +0.006 · -0.217 · -0.99 | HLM |
| 4 | 2D | entry_btc_rsi_closed>59.56 ∧ k_btc_ret24h≤-1.219 | 61 · -0.333 | -0.057 | -0.276 · -3.37 | 0.994 | -0.405 / -0.110 | 10/10 | 2 · -0.696 · -1.344 · -2.12 | HL |
| 5 | 2D | entry_eth_5m_ret1_pct>0.0714 ∧ k_btc_prevday_ret≤-0.538 | 64 · +0.070 | -0.191 | +0.261 · +3.31 | 0.996 | +0.257 / +0.264 | 10/10 | 0 · +nan · +nan · +nan | HL |
| 6 | 2Ds | entry_adx_delta≤0 ∧ k_btc_slope1h_chg2h≤0 | 36 · -0.347 | -0.088 | -0.259 · -3.25 | 0.999 | -0.239 / -0.279 | 10/10 | 5 · +0.086 · -0.105 · -0.17 | HLM |
| 7 | 2D | entry_btc_rsi_prev6≤56.2 ∧ d_btc_rsi5m_chg6≤6.1 | 17 · +0.191 | -0.148 | +0.339 · +3.24 | 0.999 | +0.131 / +0.599 | 10/10 | 1 · +0.632 · +nan · +nan | HL |
| 8 | 2Ds | d_rsi_chg≤0 ∧ k_pair_slope1h≤0 | 21 · -0.452 | -0.094 | -0.357 · -3.22 | 0.999 | -0.366 / -0.350 | 10/10 | 5 · +0.005 · -0.198 · -0.51 | HLM |
| 9 | 2Ds | k_btc_day_ret>0 ∧ k_pair_ret24h≤0 | 35 · +0.143 | -0.168 | +0.310 · +3.20 | 0.999 | +0.176 / +0.509 | 10/10 | 4 · +1.042 · +0.971 · +3.18 | HL |
| 10 | 2D | entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_pair_gap4h_20_50>0.1659 | 55 · -0.331 | -0.067 | -0.264 · -3.18 | 0.999 | -0.297 / -0.238 | 10/10 | 9 · +0.095 · -0.108 · -0.47 | HLM |
| 11 | 2Ds | d_btc_rsi5m_chg6>0 ∧ entry_eth_5m_ret1_pct≤0 | 49 · -0.313 | -0.079 | -0.234 · -3.17 | 1.000 | -0.312 / -0.123 | 10/10 | 0 · +nan · +nan · +nan | HL |
| 12 | 2D | k_eth_slope1h>-0.2463 ∧ k_btc_dom24>0.6118 | 51 · +0.086 | -0.178 | +0.264 · +3.15 | 1.000 | +0.459 / +0.175 | 10/10 | 13 · +0.276 · +0.154 · +0.60 | HLM |
| 13 | 2D | entry_adx_delta≤0.2369 ∧ entry_gap_expand_marginal>0 | 45 · -0.293 | -0.088 | -0.205 · -3.10 | 1.000 | -0.225 / -0.198 | 10/10 | 7 · -0.041 · -0.268 · -1.21 | HLM |
| 14 | 2D | entry_btc_rsi_prev6≤56.2 ∧ entry_btc_above72_pct≤47.8 | 58 · +0.076 | -0.185 | +0.261 · +3.08 | 1.000 | +0.205 / +0.327 | 10/10 | 1 · -0.993 · +nan · +nan | HL |
| 15 | 2D | entry_btc_ema50_100_gap_pct>-0.1938 ∧ k_btc_above7d_low≤3.175 | 53 · +0.054 | -0.172 | +0.226 · +3.06 | 1.000 | +0.272 / +0.216 | 10/10 | 2 · -0.844 · -1.155 · -1.50 | HL |
| 16 | 2D | k_pair_slope1h>-0.2701 ∧ k_btc_dom24>0.6118 | 50 · +0.050 | -0.168 | +0.219 · +3.06 | 1.000 | +0.337 / +0.118 | 10/10 | 8 · +0.473 · +0.378 · +1.22 | HLM |
| 17 | 2D | d_di_spread≤9.359 ∧ k_btc_prevday_ret>-0.538 | 64 · -0.285 | -0.069 | -0.216 · -3.06 | 1.000 | -0.128 / -0.292 | 10/10 | 13 · +0.020 · -0.242 · -0.76 | HLM |
| 18 | 2Ds | d_rsi_chg≤0 ∧ k_eth_ret24h≤0 | 23 · -0.436 | -0.093 | -0.343 · -3.06 | 1.000 | -0.423 / -0.126 | 10/10 | 4 · +0.228 · +0.058 · +0.15 | HL |
| 19 | 2D | entry_gap_expand_marginal>0 ∧ k_btc_slope1h≤-0.1898 | 38 · -0.327 | -0.088 | -0.238 · -3.04 | 1.000 | -0.308 / -0.134 | 10/10 | 5 · +0.083 · -0.108 · -0.50 | HLM |
| 20 | 2D | entry_gap_expand_marginal>0 ∧ k_eth_slope1h≤-0.2463 | 37 · -0.327 | -0.089 | -0.238 · -3.02 | 1.000 | -0.375 / +0.018 | 10/10 | 4 · +0.267 · +0.101 · +0.31 | L |
| 21 | 2D | entry_gap_expand_marginal>0 ∧ entry_btc_ema50_100_gap_pct≤-0.1938 | 42 · -0.307 | -0.088 | -0.219 · -3.01 | 1.000 | -0.265 / -0.161 | 10/10 | 1 · +2.543 · +nan · +nan | HL |
| 22 | 2D | k_eth_slope1h>-0.2463 ∧ k_pair_ret24h≤-1.653 | 47 · +0.073 | -0.170 | +0.243 · +3.01 | 1.000 | +0.168 / +0.289 | 10/10 | 8 · +0.466 · +0.370 · +0.95 | HLM |
| 23 | 2D | entry_btc_1h_slope≤-0.184 ∧ entry_gap_expand_marginal>0 | 38 · -0.323 | -0.089 | -0.235 · -3.00 | 1.000 | -0.308 / -0.126 | 10/10 | 5 · +0.083 · -0.108 · -0.50 | HLM |
| 24 | 2D | k_btc_1h_gap20_200>-1.277 ∧ k_btc_off7d_high≤-5.014 | 15 · -0.383 | -0.108 | -0.275 · -3.00 | 1.000 | -0.279 / -0.268 | 10/10 | 0 · +nan · +nan · +nan | HL |
| 25 | 2D | entry_ema50_slope≤0.1669 ∧ k_btc_above7d_low>3.175 | 60 · -0.309 | -0.067 | -0.242 · -3.00 | 1.000 | -0.170 / -0.347 | 10/10 | 7 · -0.007 · -0.226 · -1.00 | HLM |

Top 15 by Welch |z| (day-block null on Welch):

| mask | yr5 N/seed · avg vs rest · z_w · z_cl | scan p welch | H1/H2 | LOMO | master N · avg vs rest · z | flags |
|---|---|---|---|---|---|---|
| entry_btc_rsi_closed>59.56 ∧ k_btc_ret24h≤-1.219 | 61 · -0.333 vs -0.057 · -5.24 · -3.37 | 0.488 | -0.405/-0.110 | 10/10 | 2 · -0.696 vs +0.648 · -2.12 | HL |
| entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_btc_above7d_low>3.175 | 60 · -0.337 vs -0.057 · -5.07 · -3.39 | 0.611 | -0.202/-0.373 | 10/10 | 8 · +0.006 vs +0.223 · -0.99 | HLM |
| entry_eth_5m_ret1_pct>0.0714 ∧ k_btc_prevday_ret≤-0.538 | 64 · +0.070 vs -0.191 · +4.70 · +3.31 | 0.842 | +0.257/+0.264 | 10/10 | 0 · +nan vs -0.074 · +nan | HL |
| k_eth_slope1h>-0.2463 ∧ k_btc_dom24>0.6118 | 51 · +0.086 vs -0.178 · +4.64 · +3.15 | 0.859 | +0.459/+0.175 | 10/10 | 13 · +0.276 vs +0.122 · +0.60 | HLM |
| entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_btc_1d_slope>-0.4782 | 62 · -0.306 vs -0.065 · -4.54 · -2.86 | 0.895 | -0.224/-0.266 | 10/10 | 13 · -0.297 vs +0.433 · -2.72 | HLM |
| entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_pair_gap4h_20_50>0.1659 | 55 · -0.331 vs -0.067 · -4.52 · -3.18 | 0.900 | -0.297/-0.238 | 10/10 | 9 · +0.095 vs +0.203 · -0.47 | HLM |
| entry_btc_rsi_prev6≤56.2 ∧ entry_pair_rank≤29 | 65 · +0.067 vs -0.191 · +4.50 · +3.58 | 0.908 | +0.197/+0.330 | 10/10 | 13 · +0.295 vs +0.112 · +0.61 | HLM |
| k_btc_1d_slope≤-0.4782 ∧ k_hour_utc≤12.43 | 62 · +0.079 vs -0.191 · +4.45 · +2.74 | 0.931 | +0.249/+0.373 | 10/10 | 4 · +0.736 vs +0.109 · +2.18 | HL |
| k_eth_ret24h>-1.375 ∧ k_btc_dom24>0.6118 | 54 · +0.081 vs -0.180 · +4.45 · +2.84 | 0.933 | +0.422/+0.151 | 10/10 | 12 · +0.291 vs +0.121 · +0.66 | HLM |
| k_btc_off30d_high≤-9.063 ∧ k_btc_ret24h>-1.219 | 44 · +0.145 vs -0.181 · +4.43 · +2.74 | 0.937 | +0.420/+0.202 | 10/10 | 6 · +0.623 vs +0.090 · +1.58 | HLM |
| entry_ema50_slope≤0.1669 ∧ k_btc_above7d_low>3.175 | 60 · -0.309 vs -0.067 · -4.38 · -3.00 | 0.952 | -0.170/-0.347 | 10/10 | 7 · -0.007 vs +0.219 · -1.00 | HLM |
| entry_btc_rsi_closed>59.56 ∧ entry_gap_5_20_signed_pct≤0.3595 | 56 · -0.307 vs -0.071 · -4.33 · -2.65 | 0.966 | -0.158/-0.315 | 10/10 | 2 · -0.696 vs +0.648 · -2.12 | HL |
| entry_gap≤0.3595 ∧ entry_btc_rsi_closed>59.56 | 56 · -0.307 vs -0.071 · -4.33 · -2.65 | 0.966 | -0.158/-0.315 | 10/10 | 2 · -0.696 vs +0.648 · -2.12 | HL |
| entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_btc_off30d_high>-9.063 | 63 · -0.294 vs -0.067 · -4.29 · -2.66 | 0.969 | -0.221/-0.242 | 10/10 | 13 · -0.297 vs +0.433 · -2.72 | HLM |
| entry_btc_rsi_prev6≤56.2 ∧ entry_btc_above72_pct≤47.8 | 58 · +0.076 vs -0.185 · +4.28 · +3.08 | 0.969 | +0.205/+0.327 | 10/10 | 1 · -0.993 vs -0.046 · +nan | HL |

Top 15 that pass halves ∧ LOMO ∧ master direction (ignoring scan p):

| mask | yr5 N/seed · avg vs rest · z | scan p | H1/H2 | master N · avg vs rest · z |
|---|---|---|---|---|
| entry_btc_rsi_prev6≤56.2 ∧ entry_pair_rank≤29 | 65 · +0.067 vs -0.191 · +3.58 | 0.966 | +0.197/+0.330 | 13 · +0.295 vs +0.112 · +0.61 |
| entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_btc_above7d_low>3.175 | 60 · -0.337 vs -0.057 · -3.39 | 0.993 | -0.202/-0.373 | 8 · +0.006 vs +0.223 · -0.99 |
| entry_adx_delta≤0 ∧ k_btc_slope1h_chg2h≤0 | 36 · -0.347 vs -0.088 · -3.25 | 0.999 | -0.239/-0.279 | 5 · +0.086 vs +0.191 · -0.17 |
| d_rsi_chg≤0 ∧ k_pair_slope1h≤0 | 21 · -0.452 vs -0.094 · -3.22 | 0.999 | -0.366/-0.350 | 5 · +0.005 vs +0.203 · -0.51 |
| entry_pair_ema20_ema50_gap_pct≤0.2376 ∧ k_pair_gap4h_20_50>0.1659 | 55 · -0.331 vs -0.067 · -3.18 | 0.999 | -0.297/-0.238 | 9 · +0.095 vs +0.203 · -0.47 |
| k_eth_slope1h>-0.2463 ∧ k_btc_dom24>0.6118 | 51 · +0.086 vs -0.178 · +3.15 | 1.000 | +0.459/+0.175 | 13 · +0.276 vs +0.122 · +0.60 |
| entry_adx_delta≤0.2369 ∧ entry_gap_expand_marginal>0 | 45 · -0.293 vs -0.088 · -3.10 | 1.000 | -0.225/-0.198 | 7 · -0.041 vs +0.227 · -1.21 |
| k_pair_slope1h>-0.2701 ∧ k_btc_dom24>0.6118 | 50 · +0.050 vs -0.168 · +3.06 | 1.000 | +0.337/+0.118 | 8 · +0.473 vs +0.095 · +1.22 |
| d_di_spread≤9.359 ∧ k_btc_prevday_ret>-0.538 | 64 · -0.285 vs -0.069 · -3.06 | 1.000 | -0.128/-0.292 | 13 · +0.020 vs +0.261 · -0.76 |
| entry_gap_expand_marginal>0 ∧ k_btc_slope1h≤-0.1898 | 38 · -0.327 vs -0.088 · -3.04 | 1.000 | -0.308/-0.134 | 5 · +0.083 vs +0.191 · -0.50 |
| k_eth_slope1h>-0.2463 ∧ k_pair_ret24h≤-1.653 | 47 · +0.073 vs -0.170 · +3.01 | 1.000 | +0.168/+0.289 | 8 · +0.466 vs +0.096 · +0.95 |
| entry_btc_1h_slope≤-0.184 ∧ entry_gap_expand_marginal>0 | 38 · -0.323 vs -0.089 · -3.00 | 1.000 | -0.308/-0.126 | 5 · +0.083 vs +0.191 · -0.50 |
| entry_ema50_slope≤0.1669 ∧ k_btc_above7d_low>3.175 | 60 · -0.309 vs -0.067 · -3.00 | 1.000 | -0.170/-0.347 | 7 · -0.007 vs +0.219 · -1.00 |
| entry_adx_delta>0.2369 ∧ k_eth_slope1h>-0.2463 | 66 · +0.040 vs -0.183 · +2.99 | 1.000 | +0.207/+0.237 | 15 · +0.377 vs +0.040 · +1.21 |
| entry_btc_rsi_prev6>56.2 ∧ entry_bull_pct≤75 | 64 · -0.272 vs -0.074 · -2.98 | 1.000 | -0.143/-0.248 | 8 · -0.318 vs +0.313 · -2.08 |

## scan yr5_exwash: 17510 masks (1D 468 · 2D 17042) on 489 fills (163/seed, 98 days) · observed max |z| 3.88 · null 95th pct of max |z| 5.58 · P(null max ≥ observed max) 0.900 · reference trade-shuffle null (250×): 95th 5.62, P 0.952
- day-clustered z: obs max 3.88 vs null95 5.58 (p 0.900); masks |z|≥3: obs 42 vs null median 72 / 95th 267 (p 0.763)
- Welch z (trade SE) under the same day-block null: obs max 5.86 vs null95 6.57 (p 0.221); masks |z|≥3: obs 774 vs null median 380 / 95th 1091 (p 0.122)
survivors (either statistic): scan p<0.05 cl 0 / welch 0 · +halves 0 · +LOMO 0 · +master 0
masks passing halves ∧ LOMO ∧ master WITHOUT the scan correction: 2280 of 17510

(top 25 by day-clustered |z|; scan p = cl / welch)


| # | kind | mask | yr5 zone N/seed · avg | rest avg | Δ · z | scan p | H1Δ / H2Δ | LOMO | master zone N · avg · Δ · z | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2Ds | k_btc_day_ret>0 ∧ k_pair_ret24h≤0 | 20 · +0.186 | -0.170 | +0.356 · +3.88 | 0.900 | +0.337 / +0.392 | 9/9 | 0 · +nan · +nan · +nan | HL |
| 2 | 2D | entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_4h_gap20_50>0.27 | 42 · -0.394 | -0.036 | -0.359 · -3.82 | 0.923 | -0.185 / -0.479 | 9/9 | 7 · -0.052 · +0.012 · +0.04 | HL |
| 3 | 2D | entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_1d_slope>0.2811 | 46 · -0.385 | -0.025 | -0.360 · -3.68 | 0.967 | -0.267 / -0.424 | 9/9 | 10 · -0.245 · -0.316 · -0.97 | HLM |
| 4 | 2D | entry_pair_rank>29 ∧ entry_gap_5_20_signed_pct≤0.3597 | 37 · -0.391 | -0.050 | -0.341 · -3.63 | 0.976 | -0.321 / -0.362 | 9/9 | 2 · -0.694 · -0.931 · -1.40 | HL |
| 5 | 2D | entry_gap≤0.3597 ∧ entry_pair_rank>29 | 37 · -0.391 | -0.050 | -0.341 · -3.63 | 0.976 | -0.321 / -0.362 | 9/9 | 4 · +0.059 · +0.144 · +0.30 | HL |
| 6 | 2D | entry_gap_5_20_signed_pct≤0.3597 ∧ k_btc_hrs_since_slope_neg>9 | 40 · -0.381 | -0.046 | -0.335 · -3.63 | 0.976 | -0.334 / -0.373 | 9/9 | 3 · -0.695 · -1.244 · -1.71 | HL |
| 7 | 2D | entry_gap≤0.3597 ∧ k_btc_hrs_since_slope_neg>9 | 40 · -0.381 | -0.046 | -0.335 · -3.63 | 0.976 | -0.334 / -0.373 | 9/9 | 3 · -0.695 · -0.725 · -3.61 | HL |
| 8 | 2D | entry_ema50_slope>0.1685 ∧ k_btc_slope1h_chg3h≤-0.01405 | 32 · +0.131 | -0.191 | +0.322 · +3.54 | 0.988 | +0.344 / +0.307 | 9/9 | 7 · +0.427 · +0.689 · +1.56 | HLM |
| 9 | 2D | entry_ema50_slope≤0.1685 ∧ k_btc_4h_gap20_50>0.27 | 41 · -0.380 | -0.041 | -0.339 · -3.53 | 0.988 | -0.147 / -0.487 | 9/9 | 6 · -0.077 · -0.021 · -0.07 | HLM |
| 10 | 2D | entry_gap≤0.3597 ∧ entry_btc_rsi_closed>58.99 | 35 · -0.405 | -0.051 | -0.354 · -3.43 | 0.994 | -0.516 / -0.260 | 9/9 | 2 · -0.696 · -1.344 · -2.12 | HL |
| 11 | 2D | entry_btc_rsi_closed>58.99 ∧ entry_gap_5_20_signed_pct≤0.3597 | 35 · -0.405 | -0.051 | -0.354 · -3.43 | 0.994 | -0.516 / -0.260 | 9/9 | 2 · -0.696 · -1.344 · -2.12 | HL |
| 12 | 2D | entry_ema50_slope≤0.1685 ∧ k_btc_1d_slope>0.2811 | 45 · -0.372 | -0.033 | -0.339 · -3.42 | 0.994 | -0.212 / -0.434 | 9/9 | 9 · -0.283 · -0.356 · -1.06 | HLM |
| 13 | 2Ds | k_btc_prevday_ret>0 ∧ k_pair_gap1h_20_200≤0 | 18 · -0.435 | -0.088 | -0.346 · -3.38 | 0.996 | -0.393 / -0.313 | 9/9 | 1 · +2.543 · +nan · +nan | HL |
| 14 | 2D | entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_vs_1d_ema20>-0.0114 | 47 · -0.354 | -0.035 | -0.319 · -3.36 | 0.997 | -0.169 / -0.420 | 9/9 | 11 · -0.404 · -0.633 · -1.96 | HLM |
| 15 | 2D | entry_ema20_slope≤0.1459 ∧ k_btc_hrs_since_slope_neg>9 | 39 · -0.382 | -0.046 | -0.336 · -3.35 | 0.997 | -0.390 / -0.302 | 9/9 | 4 · -0.497 · -0.523 · -1.76 | HL |
| 16 | 2Ds | entry_eth_5m_ret1_pct≤0 ∧ k_pair_slope1h≤0 | 27 · -0.384 | -0.076 | -0.308 · -3.32 | 0.999 | -0.414 / -0.204 | 9/9 | 2 · -0.696 · -0.934 · -1.41 | HL |
| 17 | 2D | k_btc_1h_gap20_200≤-0.3321 ∧ k_btc_off7d_high>-4 | 14 · +0.232 | -0.161 | +0.393 · +3.24 | 0.999 | +0.088 / +0.504 | 9/9 | 5 · -0.402 · -0.430 · -0.79 | HL |
| 18 | 2D | entry_range_position≤83.8 ∧ k_btc_hrs_since_slope_neg>9 | 40 · -0.335 | -0.059 | -0.277 · -3.23 | 0.999 | -0.213 / -0.354 | 9/9 | 4 · -0.547 · -0.583 · -2.19 | HL |
| 19 | 2D | k_btc_prevday_ret>-0.4511 ∧ k_btc_hrs_since_slope_neg>9 | 31 · -0.390 | -0.066 | -0.324 · -3.23 | 0.999 | -0.132 / -0.505 | 9/9 | 3 · -0.695 · -0.725 · -3.61 | HL |
| 20 | 2D | entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_1d_gap9_20>0.329 | 46 · -0.356 | -0.038 | -0.318 · -3.20 | 0.999 | -0.296 / -0.331 | 9/9 | 11 · -0.404 · -0.633 · -1.96 | HLM |
| 21 | 2D | entry_gap>0.3597 ∧ k_pair_off24h_high>-4.43 | 34 · +0.103 | -0.189 | +0.292 · +3.17 | 1.000 | +0.413 / +0.205 | 9/9 | 5 · +0.669 · +0.922 · +1.76 | HLM |
| 22 | 2D | entry_gap_5_20_signed_pct>0.3597 ∧ k_pair_off24h_high>-4.43 | 34 · +0.103 | -0.189 | +0.292 · +3.17 | 1.000 | +0.413 / +0.205 | 9/9 | 1 · +2.543 · +nan · +nan | HL |
| 23 | 2D | entry_btc_ema50_100_gap_pct>-0.1912 ∧ k_btc_dom24>0.5864 | 38 · +0.095 | -0.195 | +0.290 · +3.17 | 1.000 | +0.323 / +0.273 | 9/9 | 1 · -0.695 · +nan · +nan | HL |
| 24 | 2D | entry_gap_5_20_signed_pct>0.3597 ∧ k_pair_ret1h≤0.8651 | 32 · +0.103 | -0.184 | +0.287 · +3.16 | 1.000 | +0.353 / +0.206 | 9/9 | 2 · +1.319 · +2.089 · +26.47 | HL |
| 25 | 2D | entry_gap>0.3597 ∧ k_pair_ret1h≤0.8651 | 32 · +0.103 | -0.184 | +0.287 · +3.16 | 1.000 | +0.353 / +0.206 | 9/9 | 6 · +0.505 · +0.754 · +1.75 | HLM |

Top 15 by Welch |z| (day-block null on Welch):

| mask | yr5 N/seed · avg vs rest · z_w · z_cl | scan p welch | H1/H2 | LOMO | master N · avg vs rest · z | flags |
|---|---|---|---|---|---|---|
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_1d_slope>0.2811 | 46 · -0.385 vs -0.025 · -5.86 · -3.68 | 0.221 | -0.267/-0.424 | 9/9 | 10 · -0.245 vs +0.071 · -0.97 | HLM |
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_4h_gap20_50>0.27 | 42 · -0.394 vs -0.036 · -5.73 · -3.82 | 0.287 | -0.185/-0.479 | 9/9 | 7 · -0.052 vs -0.064 · +0.04 | HL |
| entry_gap≤0.3597 ∧ entry_btc_rsi_closed>58.99 | 35 · -0.405 vs -0.051 · -5.49 · -3.43 | 0.407 | -0.516/-0.260 | 9/9 | 2 · -0.696 vs +0.648 · -2.12 | HL |
| entry_btc_rsi_closed>58.99 ∧ entry_gap_5_20_signed_pct≤0.3597 | 35 · -0.405 vs -0.051 · -5.49 · -3.43 | 0.407 | -0.516/-0.260 | 9/9 | 2 · -0.696 vs +0.648 · -2.12 | HL |
| entry_ema50_slope≤0.1685 ∧ k_btc_1d_slope>0.2811 | 45 · -0.372 vs -0.033 · -5.49 · -3.42 | 0.408 | -0.212/-0.434 | 9/9 | 9 · -0.283 vs +0.072 · -1.06 | HLM |
| entry_ema50_slope≤0.1685 ∧ k_btc_4h_gap20_50>0.27 | 41 · -0.380 vs -0.041 · -5.42 · -3.53 | 0.446 | -0.147/-0.487 | 9/9 | 6 · -0.077 vs -0.056 · -0.07 | HLM |
| entry_pair_rank>29 ∧ entry_gap_5_20_signed_pct≤0.3597 | 37 · -0.391 vs -0.050 · -5.25 · -3.63 | 0.562 | -0.321/-0.362 | 9/9 | 2 · -0.694 vs +0.237 · -1.40 | HL |
| entry_gap≤0.3597 ∧ entry_pair_rank>29 | 37 · -0.391 vs -0.050 · -5.25 · -3.63 | 0.562 | -0.321/-0.362 | 9/9 | 4 · +0.059 vs -0.085 · +0.30 | HL |
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_vs_1d_ema20>-0.0114 | 47 · -0.354 vs -0.035 · -5.23 · -3.36 | 0.576 | -0.169/-0.420 | 9/9 | 11 · -0.404 vs +0.229 · -1.96 | HLM |
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_1d_gap9_20>0.329 | 46 · -0.356 vs -0.038 · -5.13 · -3.20 | 0.634 | -0.296/-0.331 | 9/9 | 11 · -0.404 vs +0.229 · -1.96 | HLM |
| entry_gap_5_20_signed_pct≤0.3597 ∧ k_btc_hrs_since_slope_neg>9 | 40 · -0.381 vs -0.046 · -5.04 · -3.63 | 0.693 | -0.334/-0.373 | 9/9 | 3 · -0.695 vs +0.548 · -1.71 | HL |
| entry_gap≤0.3597 ∧ k_btc_hrs_since_slope_neg>9 | 40 · -0.381 vs -0.046 · -5.04 · -3.63 | 0.693 | -0.334/-0.373 | 9/9 | 3 · -0.695 vs +0.030 · -3.61 | HL |
| entry_ema20_slope≤0.1459 ∧ k_btc_hrs_since_slope_neg>9 | 39 · -0.382 vs -0.046 · -4.96 · -3.35 | 0.742 | -0.390/-0.302 | 9/9 | 4 · -0.497 vs +0.026 · -1.76 | HL |
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_above7d_low>2.694 | 38 · -0.372 vs -0.053 · -4.88 · -3.13 | 0.801 | -0.205/-0.422 | 9/9 | 7 · -0.052 vs -0.064 · +0.04 | HL |
| k_btc_4h_gap20_50≤0.27 ∧ k_pair_off24h_high>-4.43 | 42 · +0.122 vs -0.214 · +4.88 · +2.95 | 0.802 | +0.349/+0.329 | 9/9 | 5 · -0.423 vs +0.034 · -0.96 | HL |

Top 15 that pass halves ∧ LOMO ∧ master direction (ignoring scan p):

| mask | yr5 N/seed · avg vs rest · z | scan p | H1/H2 | master N · avg vs rest · z |
|---|---|---|---|---|
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_1d_slope>0.2811 | 46 · -0.385 vs -0.025 · -3.68 | 0.967 | -0.267/-0.424 | 10 · -0.245 vs +0.071 · -0.97 |
| entry_ema50_slope>0.1685 ∧ k_btc_slope1h_chg3h≤-0.01405 | 32 · +0.131 vs -0.191 · +3.54 | 0.988 | +0.344/+0.307 | 7 · +0.427 vs -0.262 · +1.56 |
| entry_ema50_slope≤0.1685 ∧ k_btc_4h_gap20_50>0.27 | 41 · -0.380 vs -0.041 · -3.53 | 0.988 | -0.147/-0.487 | 6 · -0.077 vs -0.056 · -0.07 |
| entry_ema50_slope≤0.1685 ∧ k_btc_1d_slope>0.2811 | 45 · -0.372 vs -0.033 · -3.42 | 0.994 | -0.212/-0.434 | 9 · -0.283 vs +0.072 · -1.06 |
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_vs_1d_ema20>-0.0114 | 47 · -0.354 vs -0.035 · -3.36 | 0.997 | -0.169/-0.420 | 11 · -0.404 vs +0.229 · -1.96 |
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_1d_gap9_20>0.329 | 46 · -0.356 vs -0.038 · -3.20 | 0.999 | -0.296/-0.331 | 11 · -0.404 vs +0.229 · -1.96 |
| entry_gap>0.3597 ∧ k_pair_off24h_high>-4.43 | 34 · +0.103 vs -0.189 · +3.17 | 1.000 | +0.413/+0.205 | 5 · +0.669 vs -0.253 · +1.76 |
| entry_gap>0.3597 ∧ k_pair_ret1h≤0.8651 | 32 · +0.103 vs -0.184 · +3.16 | 1.000 | +0.353/+0.206 | 6 · +0.505 vs -0.249 · +1.75 |
| entry_pair_ema20_ema50_gap_pct>0.238 ∧ k_btc_slope1h_chg3h≤-0.01405 | 32 · +0.098 vs -0.181 · +3.07 | 1.000 | +0.277/+0.280 | 6 · +0.482 vs -0.242 · +1.40 |
| entry_ema50_slope≤0.1685 ∧ k_btc_vs_1d_ema20>-0.0114 | 45 · -0.342 vs -0.044 · -3.06 | 1.000 | -0.117/-0.434 | 10 · -0.454 vs +0.220 · -1.96 |
| k_btc_rv_ratio>0.9098 ∧ k_btc_dom24>0.5864 | 43 · +0.065 vs -0.196 · +3.04 | 1.000 | +0.094/+0.396 | 5 · +0.015 vs -0.081 · +0.31 |
| entry_ema50_slope≤0.1685 ∧ k_btc_above7d_low>2.694 | 37 · -0.369 vs -0.057 · -3.02 | 1.000 | -0.168/-0.469 | 6 · -0.077 vs -0.056 · -0.07 |
| entry_btc_rsi_prev6≤56.4 ∧ k_btc_hrs_since_slope_neg≤9 | 47 · +0.064 vs -0.205 · +3.02 | 1.000 | +0.178/+0.337 | 8 · +0.276 vs -0.229 · +1.05 |
| entry_ema50_slope≤0.1685 ∧ k_btc_1d_gap9_20>0.329 | 45 · -0.347 vs -0.044 · -2.98 | 1.000 | -0.211/-0.366 | 10 · -0.454 vs +0.220 · -1.96 |
| entry_pair_ema20_ema50_gap_pct≤0.238 ∧ k_btc_rv_ratio≤0.9098 | 42 · -0.333 vs -0.056 · -2.96 | 1.000 | -0.234/-0.324 | 9 · -0.412 vs +0.150 · -1.70 |

## pure OOS — scan on yr5 H1 only (125/seed, 70 days): max |z| 4.21 · null 95th 5.70 · P 0.767
H1 top-25 → H2 same sign 13/25 (|z2| ≥ 1.96 same sign: 1) · master same sign 14/21

| H1 mask | H1 N/seed · Δ · z · scan p | H2 N/seed · Δ · z | master N · Δ · z |
|---|---|---|---|
| entry_gap_expand_marginal>0 ∧ entry_global_volume_ratio≤0.762 | 22 · -0.412 · -4.21 · 0.77 | 20 · +0.163 · +1.36 | 9 · +0.278 · +0.76 |
| d_btc_adx_chg>-0.0646 ∧ k_btc_1h_gap20_50≤-0.4764 | 28 · -0.473 · -4.19 · 0.77 | 37 · -0.013 · -0.13 | 9 · -0.437 · -1.49 |
| entry_global_volume_ratio≤0.762 ∧ entry_btc_rsi_closed>59.64 | 27 · -0.403 · -3.98 · 0.90 | 21 · +0.037 · +0.27 | 3 · +nan · +nan |
| entry_gap_expand_marginal>0 ∧ k_btc_4h_slope≤-0.3915 | 19 · -0.397 · -3.89 · 0.92 | 14 · +0.029 · +0.30 | 4 · +0.083 · +0.27 |
| entry_gap_expand_marginal>0 ∧ k_btc_off24h_high≤-2.649 | 17 · -0.376 · -3.85 · 0.95 | 10 · -0.123 · -1.21 | 3 · -0.148 · -0.46 |
| entry_ema50_slope>0.149 ∧ k_pair_off24h_high>-5.338 | 36 · +0.353 · +3.79 · 0.96 | 45 · +0.035 · +0.34 | 11 · +0.202 · +0.60 |
| entry_pair_ema20_ema50_gap_pct>0.2435 ∧ k_pair_off24h_high>-5.338 | 36 · +0.346 · +3.74 · 0.97 | 40 · +0.070 · +0.65 | 11 · +0.202 · +0.60 |
| k_btc_ret24h>-1.653 ∧ k_pair_ret1h≤0.812 | 33 · +0.395 · +3.69 · 0.98 | 32 · +0.042 · +0.34 | 12 · +0.545 · +1.86 |
| entry_gap_expand_marginal>0 ∧ k_btc_day_ret≤-0.8001 | 21 · -0.357 · -3.47 · 1.00 | 16 · -0.191 · -2.30 | 8 · +0.281 · +0.84 |
| entry_gap_expand_marginal>0 ∧ k_eth_slope1h≤-0.3392 | 21 · -0.378 · -3.45 · 1.00 | 9 · -0.058 · -0.46 | 3 · -0.148 · -0.46 |
| entry_global_volume_ratio>0.762 ∧ k_alt_up_share24>0.2114 | 31 · +0.397 · +3.36 · 1.00 | 43 · -0.144 · -1.51 | 11 · +0.118 · +0.39 |
| k_pair_ret1h≤0.812 ∧ k_alt_up_share24>0.2114 | 33 · +0.361 · +3.35 · 1.00 | 27 · -0.008 · -0.06 | 10 · +0.490 · +1.60 |
| entry_global_volume_ratio≤0.762 ∧ d_btc_adx_chg>-0.0646 | 26 · -0.352 · -3.35 · 1.00 | 28 · +0.166 · +1.55 | 15 · -0.230 · -0.82 |
| k_pair_ret1h≤0.812 ∧ k_pair_ret24h>-2.373 | 32 · +0.345 · +3.35 · 1.00 | 31 · +0.010 · +0.08 | 11 · +0.435 · +1.50 |
| entry_btc_rsi_closed>59.64 ∧ k_btc_ret24h≤-1.653 | 31 · -0.399 · -3.34 · 1.00 | 18 · -0.126 · -0.89 | 2 · +nan · +nan |
| entry_gap_expand_marginal>0 ∧ k_btc_1h_gap20_50≤-0.4764 | 19 · -0.370 · -3.29 · 1.00 | 19 · +0.090 · +1.09 | 5 · +0.075 · +0.29 |
| k_pair_slope1h>-0.3332 ∧ k_pair_ret1h≤0.812 | 32 · +0.329 · +3.27 · 1.00 | 33 · +0.040 · +0.37 | 8 · +0.593 · +1.51 |
| entry_gap_5_20_prev_signed_pct>0.3329 ∧ k_btc_day_ret≤-0.8001 | 28 · -0.389 · -3.26 · 1.00 | 27 · +0.108 · +1.05 | 2 · +nan · +nan |
| entry_ema_gap_8_13>0.1294 ∧ k_btc_day_ret≤-0.8001 | 29 · -0.391 · -3.26 · 1.00 | 31 · -0.064 · -0.55 | 9 · +0.652 · +2.37 |
| k_pair_ret1h≤0.812 ∧ k_alt_med_ret24>-1.961 | 31 · +0.351 · +3.21 · 1.00 | 28 · -0.013 · -0.10 | 10 · +0.490 · +1.60 |
| k_btc_ret1h≤0.4257 ∧ k_pair_off24h_high>-5.338 | 31 · +0.265 · +3.19 · 1.00 | 37 · -0.051 · -0.59 | 9 · +0.000 · +0.00 |
| entry_gap_expand_marginal>0 ∧ k_btc_ret24h≤-1.653 | 23 · -0.348 · -3.19 · 1.00 | 13 · -0.013 · -0.12 | 4 · +0.083 · +0.27 |
| k_btc_off24h_high>-2.649 ∧ k_pair_ret1h≤0.812 | 31 · +0.327 · +3.19 · 1.00 | 35 · +0.010 · +0.09 | 12 · +0.381 · +1.29 |
| entry_gap_5_20_signed_pct>0.3514 ∧ k_btc_day_ret≤-0.8001 | 29 · -0.380 · -3.18 · 1.00 | 27 · +0.093 · +0.88 | 2 · +nan · +nan |
| entry_gap>0.3514 ∧ k_btc_day_ret≤-0.8001 | 29 · -0.380 · -3.18 · 1.00 | 27 · +0.093 · +0.88 | 9 · +0.652 · +2.37 |


# Follow-ups

## A. BTC distance from its 30-day high (continuous) inside NEGFLANK

rebuild k_btc_off30d_high vs live stamp: see NEGFLANK_2D_rebuild_validation.csv (r 0.997 master / 0.9999 yr5). washed (k ≤ −15) on master = date window exactly: 0 (1 = identical)

| BTC off 30d high | yr5 NEG N/seed · WR · avg · $/seed · days | yr5 non-NEG | master NEG (ex-B1) | master non-NEG |
|---|---|---|---|---|
| (-99, -15] | 88 · 59% · -0.122 · $-1,925 · 44d · P(<0) 0.97 | 80 · 55% · -0.094 · $-2,357 · 37d · P(<0) 0.89 | 13 · 100% · +0.615 · $+1,364 · 9d · P(<0) 0.00 | 6 · 100% · +0.600 · $+575 · 4d · P(<0) 0.00 |
| (-15, -10] | 30 · 66% · -0.060 · $-610 · 16d · P(<0) 0.65 | 22 · 64% · -0.040 · $-381 · 17d · P(<0) 0.59 | 0 | 1 · 100% · +0.430 · $+51 · 1d · P(<0) 0.00 |
| (-10, -6] | 42 · 57% · -0.122 · $-877 · 25d · P(<0) 0.85 | 41 · 69% · +0.006 · $+120 · 26d · P(<0) 0.47 | 4 · 75% · +0.298 · $+178 · 3d · P(<0) 0.26 | 9 · 78% · +0.140 · $+294 · 4d · P(<0) 0.19 |
| (-6, -3] | 57 · 66% · -0.079 · $-910 · 44d · P(<0) 0.82 | 103 · 61% · -0.048 · $-834 · 51d · P(<0) 0.79 | 15 · 47% · -0.368 · $-1,128 · 8d · P(<0) 0.98 | 30 · 77% · +0.188 · $+808 · 14d · P(<0) 0.10 |
| (-3, 0.01] | 34 · 47% · -0.266 · $-2,098 · 22d · P(<0) 0.99 | 97 · 68% · +0.047 · $+790 · 53d · P(<0) 0.16 | 5 · 80% · +0.573 · $+596 · 2d · P(<0) 0.00 | 32 · 69% · +0.111 · $+279 · 15d · P(<0) 0.23 |

yr5: NEG ∧ washed fills 88 on 44 days in **4 episodes** (gap > 3 d splits): 01-30→03-03 (25d, -0.171); 06-02→06-11 (9d, -0.139); 06-16→06-18 (3d, -0.009); 06-22→07-01 (7d, +0.114)

master: NEG ∧ washed fills 13 on 9 days in **2 episodes** (gap > 3 d splits): 06-18→06-18 (1d, +0.695); 06-22→07-01 (8d, +0.600)

### A2. washed vs not, inside NEG (and the same split in non-NEG longs = interaction)

| cohort | NEG ∧ washed | NEG ∧ not washed | Δ in NEG | Δ in non-NEG | interaction (NEG − non-NEG) [95% day-CI] |
|---|---|---|---|---|---|
| yr5 | 88 · 59% · -0.122 · $-1,925 · 44d · P(<0) 0.97 | 163 · 60% · -0.125 · $-4,494 · 98d · P(<0) 0.99 | +0.003 | -0.090 | +0.093 [-0.141, +0.329] |
| master ex-B1 | 13 · 100% · +0.615 · $+1,364 · 9d · P(<0) 0.00 | 24 · 58% · -0.061 · $-354 · 13d · P(<0) 0.63 | +0.676 | +0.449 | +0.227 [-0.501, +0.932] |
| master ex-B1 ex-B18 | 13 · 100% · +0.615 · $+1,364 · 9d · P(<0) 0.00 | 21 · 67% · +0.030 · $+92 · 12d · P(<0) 0.46 | +0.585 | +0.449 | +0.136 [-0.586, +0.808] |
- Spearman(off30, pct) yr5: fills +0.002 · day-means +0.024 (140 days)
- Spearman(off30, pct) yr5 ex-washed: fills -0.020 · day-means -0.010 (98 days)
- Spearman(off30, pct) master: fills -0.281 · day-means -0.412 (22 days)
- Spearman(off30, pct) master ex-washed: fills +0.065 · day-means +0.137 (13 days)

where the 1D off30 masks sit in the scans:

| scan | mask | yr5 N/seed · zone avg vs rest · z_cl · z_w | scan p (cl / welch) | H1/H2 | master N · Δ |
|---|---|---|---|---|---|
| all | k_btc_off30d_high med>-9.063 | 125 · -0.174 vs -0.074 · -1.21 · -2.01 | 1.00 / 1.00 | -0.101/-0.126 | 24 · -0.676 |
| all | k_btc_off30d_high Q5>-4.062 | 50 · -0.178 vs -0.111 · -0.74 · -1.19 | 1.00 / 1.00 | +0.025/-0.138 | 10 · +0.079 |
| all | k_btc_off30d_high Q1≤-24.81 | 50 · -0.174 vs -0.112 · -0.68 · -1.02 | 1.00 / 1.00 | -0.077/-0.010 | 0 · +nan |
| all | k_btc_off30d_high T1≤-15.88 | 84 · -0.157 vs -0.108 · -0.59 · -0.91 | 1.00 / 1.00 | -0.148/+0.077 | 13 · +0.676 |
| all | k_btc_off30d_high T3>-5.581 | 83 · -0.122 vs -0.126 · +0.04 · +0.07 | 1.00 / 1.00 | +0.033/-0.024 | 19 · -0.434 |
| exwash | k_btc_off30d_high Q5>-2.995 | 32 · -0.243 vs -0.099 · -1.31 · -2.23 | 1.00 / 1.00 | -0.080/-0.214 | 5 · +0.800 |
| exwash | k_btc_off30d_high Q1≤-9.498 | 33 · +0.035 vs -0.168 · +1.22 · +2.38 | 1.00 / 1.00 | +0.231/+0.162 | 0 · +nan |
| exwash | k_btc_off30d_high T1≤-7.546 | 55 · -0.026 vs -0.179 · +1.19 · +2.26 | 1.00 / 1.00 | +0.090/+0.263 | 2 · +0.465 |
| exwash | k_btc_off30d_high T3>-4.284 | 54 · -0.146 vs -0.118 · -0.28 · -0.47 | 1.00 / 1.00 | +0.035/-0.069 | 11 · +0.521 |
| exwash | k_btc_off30d_high med>-5.199 | 81 · -0.115 vs -0.139 · +0.22 · +0.40 | 1.00 / 1.00 | +0.042/+0.026 | 19 · +0.125 |

### A3. candidate rule "block NEGFLANK long unless washed-out (BTC ≤ −15 % below 30d high)"

- master ex-B1: sleeve before 115 fills · $+3,016 · +0.183%/fill → after 91 · $+3,370 · +0.247%/fill · blocked 24 fills worth $-354 → Δ $+354 (30–50 % haircut: $+177 … $+248)
- master ex-B1 ex-B18: sleeve before 112 fills · $+3,463 · +0.206%/fill → after 91 · $+3,370 · +0.247%/fill · blocked 21 fills worth $+92 → Δ $-92 (30–50 % haircut: $-46 … $-65)
- B18: sleeve before 3 fills · $-447 · -0.695%/fill → after 0 · $+0 · +nan%/fill · blocked 3 fills worth $-447 → Δ $+447 (30–50 % haircut: $+223 … $+313)
- yr5: sleeve before 595 fills · $-9,081 · -0.067%/fill → after 431 · $-4,587 · -0.045%/fill · blocked 163 fills worth $-4,494 → Δ $+4,494 (30–50 % haircut: $+2,247 … $+3,146)

locked expectancy bar on the BLOCKED side (NEG ∧ not washed), BE WR from the whole sleeve's kept fills:

| cohort | N | WR vs BE | avg · day-clustered P(mean<0) | days ≥ 8 | N ≥ 15 | concentration < 50 % | verdict |
|---|---|---|---|---|---|---|---|
| yr5 | 163 | 60% vs BE 66.9% ✔ | -0.125 · P(<0) 0.99 ✔ | 98 ✔ | ✔ | day 6% / pair 6% ✔ | **PASS** |
| master ex-B1 | 24 | 58% vs BE 61.3% ✔ | -0.061 · P(<0) 0.63 ✗ | 13 ✔ | ✔ | day 23% / pair 22% ✔ | **FAIL** |
| master ex-B1 ex-B18 | 21 | 67% vs BE 61.8% ✗ | +0.030 · P(<0) 0.46 ✗ | 12 ✔ | ✔ | day 28% / pair 28% ✔ | **FAIL** |

## B. closest-to-surviving families (none passes the scan null)


### B1 pair EMA20/50 gap ≤ 0.238 ∧ BTC 1d EMA20 slope > 0.281 (ex-wash scan #1 by Welch)

| cohort | zone (NEG ∧ X) | NEG ∧ ¬X | Δ in NEG | Δ in non-NEG | interaction [95% day-CI] |
|---|---|---|---|---|---|
| yr5 all | 46 · 36% · -0.385 · $-4,029 · 36d · P(<0) 1.00 | 205 · 65% · -0.065 · $-2,391 · 126d · P(<0) 0.92 | -0.320 | +0.084 | -0.403 [-0.659, -0.155] |
| yr5 ex-washed | 46 · 36% · -0.385 · $-4,029 · 36d · P(<0) 1.00 | 117 · 69% · -0.023 · $-466 · 84d · P(<0) 0.64 | -0.362 | +0.063 | -0.425 [-0.701, -0.144] |
| master ex-B1 | 10 · 50% · -0.245 · $-574 · 6d · P(<0) 0.90 | 27 · 81% · +0.333 · $+1,584 · 18d · P(<0) 0.03 | -0.578 | +0.289 | -0.867 [-1.485, -0.152] |
| master ex-B18 | 7 · 71% · -0.052 · $-127 · 5d · P(<0) 0.68 | 27 · 81% · +0.333 · $+1,584 · 18d · P(<0) 0.03 | -0.385 | +0.289 | -0.674 [-1.315, -0.041] |
| master ex-washed | 10 · 50% · -0.245 · $-574 · 6d · P(<0) 0.90 | 14 · 64% · +0.071 · $+220 · 9d · P(<0) 0.43 | -0.316 | +0.340 | -0.656 [-1.556, +0.321] |
| master ex-washed ex-B18 | 7 · 71% · -0.052 · $-127 · 5d · P(<0) 0.68 | 14 · 64% · +0.071 · $+220 · 9d · P(<0) 0.43 | -0.123 | +0.340 | -0.463 [-1.271, +0.374] |
- yr5 NEG zone concentration: largest day 12% · largest pair 8% of zone loss; overlap with washed: 0% of zone vs 43% of rest
- B18 fills in zone: LIT=True, UNI=True, ARB=True

### B2 pair EMA20/50 gap ≤ 0.238 ∧ BTC above its 1d EMA20

| cohort | zone (NEG ∧ X) | NEG ∧ ¬X | Δ in NEG | Δ in non-NEG | interaction [95% day-CI] |
|---|---|---|---|---|---|
| yr5 all | 47 · 38% · -0.354 · $-3,750 · 39d · P(<0) 1.00 | 204 · 64% · -0.071 · $-2,669 · 125d · P(<0) 0.94 | -0.282 | +0.121 | -0.404 [-0.632, -0.170] |
| yr5 ex-washed | 47 · 38% · -0.354 · $-3,750 · 39d · P(<0) 1.00 | 116 · 69% · -0.032 · $-744 · 83d · P(<0) 0.69 | -0.321 | +0.102 | -0.424 [-0.671, -0.167] |
| master ex-B1 | 11 · 45% · -0.404 · $-949 · 7d · P(<0) 0.98 | 26 · 85% · +0.422 · $+1,959 · 17d · P(<0) 0.00 | -0.826 | +0.315 | -1.141 [-1.813, -0.489] |
| master ex-B18 | 8 · 62% · -0.295 · $-502 · 6d · P(<0) 0.91 | 26 · 85% · +0.422 · $+1,959 · 17d · P(<0) 0.00 | -0.717 | +0.315 | -1.032 [-1.832, -0.373] |
| master ex-washed | 11 · 45% · -0.404 · $-949 · 7d · P(<0) 0.98 | 13 · 69% · +0.229 · $+595 · 8d · P(<0) 0.19 | -0.633 | +0.374 | -1.007 [-1.851, -0.154] |
| master ex-washed ex-B18 | 8 · 62% · -0.295 · $-502 · 6d · P(<0) 0.91 | 13 · 69% · +0.229 · $+595 · 8d · P(<0) 0.19 | -0.524 | +0.374 | -0.898 [-1.884, -0.004] |
- yr5 NEG zone concentration: largest day 12% · largest pair 9% of zone loss; overlap with washed: 0% of zone vs 43% of rest
- B18 fills in zone: LIT=True, UNI=True, ARB=True

### B3 BTC 1h slope falling further over 3 h (chg3h ≤ −0.014) ∧ pair EMA50 slope > 0.1685 [winner side]

| cohort | zone (NEG ∧ X) | NEG ∧ ¬X | Δ in NEG | Δ in non-NEG | interaction [95% day-CI] |
|---|---|---|---|---|---|
| yr5 all | 41 · 76% · +0.060 · $+795 · 46d · P(<0) 0.22 | 210 · 56% · -0.160 · $-7,215 · 126d · P(<0) 1.00 | +0.220 | +0.050 | +0.170 [-0.081, +0.415] |
| yr5 ex-washed | 32 · 81% · +0.131 · $+1,067 · 37d · P(<0) 0.05 | 131 · 54% · -0.189 · $-5,561 · 85d · P(<0) 1.00 | +0.320 | +0.076 | +0.244 [-0.049, +0.507] |
| master ex-B1 | 12 · 83% · +0.439 · $+1,120 · 9d · P(<0) 0.06 | 25 · 68% · +0.050 · $-110 · 14d · P(<0) 0.38 | +0.389 | -0.194 | +0.583 [-0.231, +1.296] |
| master ex-B18 | 12 · 83% · +0.439 · $+1,120 · 9d · P(<0) 0.06 | 22 · 77% · +0.152 · $+337 · 13d · P(<0) 0.18 | +0.287 | -0.194 | +0.481 [-0.308, +1.274] |
| master ex-washed | 7 · 71% · +0.427 · $+651 · 5d · P(<0) 0.19 | 17 · 53% · -0.262 · $-1,005 · 8d · P(<0) 0.95 | +0.689 | -0.190 | +0.879 [-0.473, +1.886] |
| master ex-washed ex-B18 | 7 · 71% · +0.427 · $+651 · 5d · P(<0) 0.19 | 14 · 64% · -0.169 · $-558 · 7d · P(<0) 0.85 | +0.596 | -0.190 | +0.786 [-0.491, +1.885] |
- yr5 NEG zone concentration: largest day 13% · largest pair 13% of zone loss; overlap with washed: 20% of zone vs 38% of rest
- B18 fills in zone: LIT=False, UNI=False, ARB=False

### B4 BTC 1h slope negative > 9 h (hrs_since_slope_neg > 9)

| cohort | zone (NEG ∧ X) | NEG ∧ ¬X | Δ in NEG | Δ in non-NEG | interaction [95% day-CI] |
|---|---|---|---|---|---|
| yr5 all | 139 · 58% · -0.189 · $-5,591 · 80d · P(<0) 1.00 | 112 · 62% · -0.044 · $-828 · 82d · P(<0) 0.76 | -0.145 | +nan | +nan [+nan, +nan] |
| yr5 ex-washed | 81 · 56% · -0.213 · $-4,144 · 52d · P(<0) 1.00 | 82 · 63% · -0.039 · $-350 · 58d · P(<0) 0.69 | -0.173 | +nan | +nan [+nan, +nan] |
| master ex-B1 | 13 · 69% · +0.018 · $-105 · 9d · P(<0) 0.45 | 24 · 75% · +0.262 · $+1,115 · 14d · P(<0) 0.09 | -0.245 | +nan | +nan [+nan, +nan] |
| master ex-B18 | 10 · 90% · +0.232 · $+342 · 8d · P(<0) 0.10 | 24 · 75% · +0.262 · $+1,115 · 14d · P(<0) 0.09 | -0.031 | +nan | +nan [+nan, +nan] |
| master ex-washed | 6 · 33% · -0.465 · $-562 · 4d · P(<0) 0.93 | 18 · 67% · +0.074 · $+208 · 9d · P(<0) 0.39 | -0.538 | +nan | +nan [+nan, +nan] |
| master ex-washed ex-B18 | 3 · 67% · -0.234 · $-116 · 3d · P(<0) 0.71 | 18 · 67% · +0.074 · $+208 · 9d · P(<0) 0.39 | -0.308 | +nan | +nan [+nan, +nan] |
- yr5 NEG zone concentration: largest day 5% · largest pair 7% of zone loss; overlap with washed: 42% of zone vs 26% of rest
- B18 fills in zone: LIT=True, UNI=True, ARB=True

### B5 BTC dominance proxy > 0.61 (BTC 24h − median alt 24h) ∧ ETH 1h slope > −0.246 [winner side]

| cohort | zone (NEG ∧ X) | NEG ∧ ¬X | Δ in NEG | Δ in non-NEG | interaction [95% day-CI] |
|---|---|---|---|---|---|
| yr5 all | 51 · 76% · +0.077 · $+912 · 46d · P(<0) 0.17 | 200 · 55% · -0.176 · $-7,331 · 116d · P(<0) 1.00 | +0.254 | +0.068 | +0.186 [-0.022, +0.385] |
| yr5 ex-washed | 38 · 76% · +0.027 · $+364 · 30d · P(<0) 0.38 | 125 · 55% · -0.171 · $-4,859 · 81d · P(<0) 0.99 | +0.198 | +0.047 | +0.151 [-0.116, +0.368] |
| master ex-B1 | 13 · 85% · +0.276 · $+657 · 7d · P(<0) 0.03 | 24 · 67% · +0.122 · $+353 · 16d · P(<0) 0.28 | +0.154 | +0.148 | +0.006 [-0.698, +0.773] |
| master ex-B18 | 13 · 85% · +0.276 · $+657 · 7d · P(<0) 0.03 | 21 · 76% · +0.239 · $+799 · 15d · P(<0) 0.14 | +0.037 | +0.148 | -0.111 [-0.740, +0.669] |
| master ex-washed | 10 · 80% · +0.174 · $+262 · 5d · P(<0) 0.12 | 14 · 43% · -0.229 · $-616 · 9d · P(<0) 0.78 | +0.402 | +0.112 | +0.290 [-0.597, +1.159] |
| master ex-washed ex-B18 | 10 · 80% · +0.174 · $+262 · 5d · P(<0) 0.12 | 11 · 55% · -0.101 · $-170 · 8d · P(<0) 0.63 | +0.275 | +0.112 | +0.163 [-0.784, +1.166] |
- yr5 NEG zone concentration: largest day 13% · largest pair 10% of zone loss; overlap with washed: 26% of zone vs 37% of rest
- B18 fills in zone: LIT=False, UNI=False, ARB=False

### B6 global volume ratio high (> 0.95, post-hoc from the B18 study)

| cohort | zone (NEG ∧ X) | NEG ∧ ¬X | Δ in NEG | Δ in non-NEG | interaction [95% day-CI] |
|---|---|---|---|---|---|
| yr5 all | 70 · 70% · -0.003 · $+150 · 68d · P(<0) 0.52 | 181 · 55% · -0.171 · $-6,570 · 121d · P(<0) 1.00 | +0.169 | +0.037 | +0.132 [-0.067, +0.331] |
| yr5 ex-washed | 52 · 75% · +0.055 · $+765 · 49d · P(<0) 0.23 | 111 · 53% · -0.210 · $-5,260 · 82d · P(<0) 1.00 | +0.265 | +0.020 | +0.245 [+0.024, +0.444] |
| master ex-B1 | 7 · 100% · +0.245 · $+263 · 5d · P(<0) 0.00 | 30 · 67% · +0.160 · $+747 · 19d · P(<0) 0.19 | +0.085 | -0.130 | +0.215 [-0.272, +0.743] |
| master ex-B18 | 7 · 100% · +0.245 · $+263 · 5d · P(<0) 0.00 | 27 · 74% · +0.255 · $+1,194 · 18d · P(<0) 0.08 | -0.010 | -0.130 | +0.120 [-0.351, +0.610] |
| master ex-washed | 5 · 100% · +0.281 · $+227 · 3d · P(<0) 0.00 | 19 · 47% · -0.151 · $-581 · 11d · P(<0) 0.74 | +0.432 | -0.137 | +0.569 [-0.043, +1.237] |
| master ex-washed ex-B18 | 5 · 100% · +0.281 · $+227 · 3d · P(<0) 0.00 | 16 · 56% · -0.049 · $-134 · 10d · P(<0) 0.60 | +0.330 | -0.137 | +0.467 [-0.126, +1.154] |
- yr5 NEG zone concentration: largest day 11% · largest pair 10% of zone loss; overlap with washed: 25% of zone vs 39% of rest
- B18 fills in zone: LIT=False, UNI=False, ARB=False

### B1 dose-response — yr5 NEG ex-washed, avg pct by tercile grid (N/seed)

| | BTC 1d slope low | mid | high |
|---|---|---|---|
| pair gap low | +0.145 (19) | -0.286 (17) | -0.351 (18) |
| mid | -0.295 (17) | -0.225 (16) | -0.210 (21) |
| high | -0.043 (19) | +0.062 (21) | +0.042 (15) |

master NEG ex-washed, same cuts frozen from yr5 (pair gap ≤ 0.238 / BTC 1d slope > 0.281):
- pair gap ≤ 0.238 ∧ 1d slope > 0.281: 10 · 50% · -0.245 · $-574 · 6d · P(<0) 0.90
- pair gap ≤ 0.238 ∧ 1d slope ≤ 0.281: 3 · 67% · -0.468 · $-284 · 2d · P(<0) 0.76
- pair gap > 0.238 ∧ 1d slope > 0.281: 7 · 71% · +0.535 · $+698 · 4d · P(<0) 0.09
- pair gap > 0.238 ∧ 1d slope ≤ 0.281: 4 · 50% · -0.338 · $-194 · 4d · P(<0) 0.94

## C. B18's three fills on the key variables

| var | LIT | UNI | ARB | yr5 NEG median | master NEG median |
|---|---|---|---|---|---|
| slope | -0.229 | -0.300 | -0.298 | -0.184 | -0.144 |
| off30 | -4.455 | -4.407 | -4.425 | -9.063 | -5.179 |
| k_btc_1d_slope | +0.567 | +0.572 | +0.570 | -0.478 | -0.037 |
| k_btc_vs_1d_ema20 | +0.006 | +0.051 | +0.034 | -1.997 | -0.822 |
| entry_pair_ema20_ema50_gap_pct | -0.207 | -0.326 | +0.024 | +0.238 | +0.215 |
| entry_ema50_slope | -0.114 | -0.188 | +0.005 | +0.167 | +0.120 |
| k_btc_slope1h_chg3h | +0.152 | +0.050 | +0.049 | +0.006 | -0.015 |
| k_btc_hrs_since_slope_neg | +24.250 | +23.667 | +23.583 | +10.250 | +6.250 |
| k_btc_dom24 | +2.973 | +3.480 | +3.353 | +0.612 | +0.775 |
| k_eth_slope1h | -0.484 | -0.554 | -0.556 | -0.246 | -0.171 |
| entry_global_volume_ratio | +0.476 | +0.442 | +0.515 | +0.777 | +0.599 |
| k_btc_4h_gap20_50 | +0.112 | +0.115 | +0.114 | -0.510 | -0.061 |
| k_btc_day_ret | -2.357 | -2.308 | -2.327 | -0.687 | -0.776 |
| k_pair_gap1h_20_200 | -6.656 | -7.959 | -6.169 | -0.009 | +0.688 |
| k_alt_up_share24 | +0.062 | +0.053 | +0.053 | +0.222 | +0.230 |
| pct | -0.695 | -0.698 | -0.694 | +0.088 | +0.098 |

## D. B1/B2 family — legs, seeds, rule arithmetic, bar

washed flag rebuilt (off30 ≤ −15) vs Jun-18→Jul-2 date window on master: 1 fill(s) differ

1D masks of entry_pair_ema20_ema50_gap_pct:

| scan | mask | yr5 N/seed · zone vs rest · z_w | scan p welch | H1/H2 | master N · Δ |
|---|---|---|---|---|---|
| all | entry_pair_ema20_ema50_gap_pct T3>0.3944 | 84 · -0.029 vs -0.172 · +2.77 | 1.00 | +0.143/+0.144 | 13 · +0.374 |
| all | entry_pair_ema20_ema50_gap_pct med>0.2376 | 125 · -0.061 vs -0.188 · +2.56 | 1.00 | +0.138/+0.117 | 17 · +0.278 |
| all | entry_pair_ema20_ema50_gap_pct Q5>0.5615 | 50 · -0.074 vs -0.137 · +1.03 | 1.00 | -0.054/+0.188 | 7 · +0.216 |
| all | entry_pair_ema20_ema50_gap_pct Q1≤-0.0709 | 51 · -0.085 vs -0.134 · +0.71 | 1.00 | -0.017/+0.135 | 12 · -0.432 |
| all | entry_pair_ema20_ema50_gap_pct T1≤0.06683 | 84 · -0.107 vs -0.133 · +0.49 | 1.00 | -0.013/+0.069 | 17 · -0.283 |
| all | entry_pair_ema20_ema50_gap_pct sign>00 | 184 · -0.129 vs -0.111 · -0.30 | 1.00 | +0.039/-0.093 | 24 · +0.371 |
| exwash | entry_pair_ema20_ema50_gap_pct Q5>0.5385 | 33 · +0.072 vs -0.177 · +3.50 | 1.00 | +0.228/+0.265 | 4 · +0.224 |
| exwash | entry_pair_ema20_ema50_gap_pct T3>0.3839 | 54 · +0.020 vs -0.201 · +3.61 | 1.00 | +0.318/+0.152 | 9 · +0.561 |
| exwash | entry_pair_ema20_ema50_gap_pct med>0.238 | 81 · -0.047 vs -0.208 · +2.68 | 1.00 | +0.224/+0.114 | 11 · +0.514 |
| exwash | entry_pair_ema20_ema50_gap_pct T1≤0.07033 | 54 · -0.161 vs -0.110 · -0.75 | 1.00 | -0.036/-0.069 | 12 · -0.537 |
| exwash | entry_pair_ema20_ema50_gap_pct sign>00 | 121 · -0.121 vs -0.146 · +0.34 | 1.00 | +0.030/+0.029 | 15 · +0.566 |
| exwash | entry_pair_ema20_ema50_gap_pct Q1≤-0.06176 | 33 · -0.107 vs -0.132 · +0.30 | 1.00 | +0.015/+0.030 | 9 · -0.566 |

1D masks of k_btc_1d_slope:

| scan | mask | yr5 N/seed · zone vs rest · z_w | scan p welch | H1/H2 | master N · Δ |
|---|---|---|---|---|---|
| all | k_btc_1d_slope T3>0.2588 | 84 · -0.193 vs -0.090 · -2.00 | 1.00 | +0.006/-0.198 | 17 · -0.186 |
| all | k_btc_1d_slope sign>00 | 98 · -0.179 vs -0.090 · -1.80 | 1.00 | +0.002/-0.176 | 18 · -0.419 |
| all | k_btc_1d_slope med>-0.4782 | 125 · -0.167 vs -0.082 · -1.71 | 1.00 | -0.033/-0.161 | 24 · -0.676 |
| all | k_btc_1d_slope Q5>0.9391 | 50 · -0.197 vs -0.106 · -1.57 | 1.00 | +0.050/-0.208 | 11 · +0.064 |
| all | k_btc_1d_slope Q1≤-1.989 | 52 · -0.182 vs -0.109 · -1.23 | 1.00 | -0.115/+0.005 | 0 · +nan |
| all | k_btc_1d_slope T1≤-1.268 | 84 · -0.079 vs -0.147 · +1.26 | 1.00 | +0.048/+0.112 | 9 · +0.514 |
| exwash | k_btc_1d_slope Q5>1.314 | 33 · -0.253 vs -0.096 · -2.19 | 1.00 | +0.095/-0.285 | 9 · +0.061 |
| exwash | k_btc_1d_slope med>0.2811 | 81 · -0.192 vs -0.064 · -2.12 | 1.00 | -0.056/-0.180 | 17 · +0.470 |
| exwash | k_btc_1d_slope sign>00 | 98 · -0.179 vs -0.050 · -2.02 | 1.00 | -0.051/-0.189 | 18 · +0.089 |
| exwash | k_btc_1d_slope T3>0.9111 | 54 · -0.188 vs -0.097 · -1.49 | 1.00 | +0.018/-0.172 | 11 · +0.521 |
| exwash | k_btc_1d_slope T1≤-0.2147 | 55 · -0.060 vs -0.161 · +1.49 | 1.00 | +0.113/+0.083 | 2 · -0.774 |
| exwash | k_btc_1d_slope Q1≤-0.6209 | 33 · -0.049 vs -0.147 · +1.14 | 1.00 | +0.033/+0.175 | 0 · +nan |

1D masks of k_btc_vs_1d_ema20:

| scan | mask | yr5 N/seed · zone vs rest · z_w | scan p welch | H1/H2 | master N · Δ |
|---|---|---|---|---|---|
| all | k_btc_vs_1d_ema20 Q5>2.137 | 49 · -0.224 vs -0.100 · -2.12 | 1.00 | +0.023/-0.252 | 11 · +0.064 |
| all | k_btc_vs_1d_ema20 sign>00 | 81 · -0.187 vs -0.094 · -1.83 | 1.00 | +0.048/-0.216 | 17 · -0.471 |
| all | k_btc_vs_1d_ema20 T3>-0.09988 | 84 · -0.183 vs -0.095 · -1.73 | 1.00 | +0.048/-0.205 | 17 · -0.471 |
| all | k_btc_vs_1d_ema20 med>-1.997 | 125 · -0.142 vs -0.107 · -0.70 | 1.00 | +0.061/-0.147 | 24 · -0.676 |
| all | k_btc_vs_1d_ema20 T1≤-4.677 | 84 · -0.144 vs -0.114 · -0.56 | 1.00 | -0.130/+0.099 | 10 · +0.597 |
| all | k_btc_vs_1d_ema20 Q1≤-7.188 | 51 · -0.143 vs -0.120 · -0.36 | 1.00 | -0.050/+0.019 | 2 · +0.612 |
| exwash | k_btc_vs_1d_ema20 sign>00 | 81 · -0.187 vs -0.068 · -1.98 | 1.00 | +0.020/-0.225 | 17 · -0.059 |
| exwash | k_btc_vs_1d_ema20 med>-0.0114 | 81 · -0.187 vs -0.068 · -1.98 | 1.00 | +0.020/-0.225 | 17 · -0.059 |
| exwash | k_btc_vs_1d_ema20 T1≤-1.175 | 54 · -0.049 vs -0.166 · +1.77 | 1.00 | +0.023/+0.215 | 2 · -0.342 |
| exwash | k_btc_vs_1d_ema20 T3>2.083 | 54 · -0.189 vs -0.097 · -1.49 | 1.00 | +0.023/-0.185 | 11 · +0.521 |
| exwash | k_btc_vs_1d_ema20 Q1≤-2.594 | 33 · -0.016 vs -0.155 · +1.63 | 1.00 | +0.137/+0.134 | 0 · +nan |
| exwash | k_btc_vs_1d_ema20 Q5>3.115 | 32 · -0.190 vs -0.112 · -1.13 | 1.00 | +0.023/-0.151 | 9 · +0.061 |

BTC 1d EMA20 slope quintiles inside NEG (yr5 cuts):

| bucket | yr5 NEG | master NEG ex-B1 |
|---|---|---|
| (-99.00, -1.99] | 52 · 57% · -0.182 · $-2,133 · 24d · P(<0) 0.98 | 0 |
| (-1.99, -0.98] | 49 · 64% · -0.039 · $-144 · 24d · P(<0) 0.67 | 11 · 100% · +0.589 · $+1,114 · 7d · P(<0) 0.00 |
| (-0.98, -0.03] | 50 · 62% · -0.035 · $-82 · 33d · P(<0) 0.61 | 8 · 75% · +0.093 · $+147 · 6d · P(<0) 0.36 |
| (-0.03, +0.94] | 51 · 59% · -0.165 · $-1,856 · 39d · P(<0) 0.99 | 7 · 29% · -0.447 · $-728 · 4d · P(<0) 0.83 |
| (+0.94, +99.00] | 50 · 54% · -0.197 · $-2,205 · 31d · P(<0) 0.99 | 11 · 73% · +0.221 · $+478 · 5d · P(<0) 0.18 |

B1 per seed (zone avg vs rest): s1: -0.396 vs -0.069 · s2: -0.385 vs -0.048 · s3: -0.370 vs -0.081
B1 per month Δ (zone − rest, yr5 NEG): 01: -0.23 (n6) · 02: +nan (n0) · 03: -0.40 (n4) · 04: -0.45 (n6) · 05: +0.09 (n2) · 06: +nan (n0) · 07: -0.55 (n6) · 08: -0.46 (n8) · 09: -0.62 (n11) · 10: +nan (n0)

rule "block NEGFLANK long when B1":
- master ex-B1: before 115 · $+3,016 · +0.183%/fill → after 105 · $+3,590 · +0.223%/fill · blocked 10 worth $-574 → Δ $+574 (haircut $+287 … $+402)
- master ex-B1 ex-B18: before 112 · $+3,463 · +0.206%/fill → after 105 · $+3,590 · +0.223%/fill · blocked 7 worth $-127 → Δ $+127 (haircut $+64 … $+89)
- B18: before 3 · $-447 · -0.695%/fill → after 0 · $+0 · +nan%/fill · blocked 3 worth $-447 → Δ $+447 (haircut $+223 … $+313)
- yr5: before 595 · $-9,081 · -0.067%/fill → after 548 · $-5,053 · -0.040%/fill · blocked 46 worth $-4,029 → Δ $+4,029 (haircut $+2,014 … $+2,820)

| cohort | N | WR vs BE | avg · P(mean<0) | days ≥ 8 | N ≥ 15 | concentration | verdict |
|---|---|---|---|---|---|---|---|
| B1 yr5 | 46 | 36% vs BE 66.9% ✔ | -0.385 · P(<0) 1.00 ✔ | 36 ✔ | ✔ | day 12% / pair 8% ✔ | **PASS** |
| B1 master ex-B1 | 10 | 50% vs BE 61.3% ✔ | -0.245 · P(<0) 0.90 ✗ | 6 ✗ | ✗ | day 55% / pair 23% ✗ | **FAIL** |
| B1 master ex-B18 | 7 | 71% vs BE 61.8% ✗ | -0.052 · P(<0) 0.68 ✗ | 5 ✗ | ✗ | day 52% / pair 52% ✗ | **FAIL** |

B2 per seed (zone avg vs rest): s1: -0.352 vs -0.076 · s2: -0.352 vs -0.054 · s3: -0.358 vs -0.083
B2 per month Δ (zone − rest, yr5 NEG): 01: -0.25 (n7) · 02: +nan (n0) · 03: -0.09 (n3) · 04: -0.37 (n7) · 05: +0.09 (n2) · 06: +nan (n0) · 07: -0.52 (n7) · 08: -0.46 (n8) · 09: -0.62 (n11) · 10: +nan (n0)

rule "block NEGFLANK long when B2":
- master ex-B1: before 115 · $+3,016 · +0.183%/fill → after 104 · $+3,965 · +0.245%/fill · blocked 11 worth $-949 → Δ $+949 (haircut $+474 … $+664)
- master ex-B1 ex-B18: before 112 · $+3,463 · +0.206%/fill → after 104 · $+3,965 · +0.245%/fill · blocked 8 worth $-502 → Δ $+502 (haircut $+251 … $+352)
- B18: before 3 · $-447 · -0.695%/fill → after 0 · $+0 · +nan%/fill · blocked 3 worth $-447 → Δ $+447 (haircut $+223 … $+313)
- yr5: before 595 · $-9,081 · -0.067%/fill → after 547 · $-5,331 · -0.042%/fill · blocked 47 worth $-3,750 → Δ $+3,750 (haircut $+1,875 … $+2,625)

| cohort | N | WR vs BE | avg · P(mean<0) | days ≥ 8 | N ≥ 15 | concentration | verdict |
|---|---|---|---|---|---|---|---|
| B2 yr5 | 47 | 38% vs BE 66.9% ✔ | -0.354 · P(<0) 1.00 ✔ | 39 ✔ | ✔ | day 12% / pair 9% ✔ | **PASS** |
| B2 master ex-B1 | 11 | 45% vs BE 61.3% ✔ | -0.404 · P(<0) 0.98 ✔ | 7 ✗ | ✗ | day 36% / pair 34% ✔ | **FAIL** |
| B2 master ex-B18 | 8 | 62% vs BE 61.8% ✗ | -0.295 · P(<0) 0.91 ✗ | 6 ✗ | ✗ | day 54% / pair 54% ✗ | **FAIL** |
