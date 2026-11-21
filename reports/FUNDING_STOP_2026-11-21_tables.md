
## FLATBASE_SHORT

**FLATBASE_SHORT — funding paid (% of notional; + = paid by the short) under today's exit (arm A)**

| cohort | N | hourly_at_entry_pct | paying_pct | paid_median | paid_p90 | paid_worst | received_best | TP N | TP paid avg | STOP N | STOP paid avg | TIME N | TIME paid avg | corr(entry rate, received) | spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 28 | +10.71 | +42.86 | +0.00 | +0.49 | +1.82 | +0.03 | 25 | +0.16 | 3 | +0.53 | 0 |  | +0.15 | +0.68 |
| real (stack_keep, before DL 601) | 32 | +9.38 | +37.50 | +0.00 | +0.38 | +1.82 | +0.03 | 27 | +0.15 | 5 | +0.31 | 0 |  | +0.18 | +0.73 |
| backtest | 545 | +8.99 | +40.73 | +0.00 | +0.58 | +5.53 | +0.12 | 402 | +0.21 | 118 | +0.18 | 25 | +0.12 | +0.26 | +0.63 |

**FLATBASE_SHORT — arm A vs B (net stop) vs Bp (paid-only) · one size 0.61 × equity · Δ = arm − A per fill, day-block bootstrap 95 % CI**

| block | arm | N | WR | avg | sum | DCR | maxDD | d_avg | d_lo | d_hi | P_gain | loo_min | loo_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | A (today) | 28 | +89.29 | +3.09 | +86.45 | +1.69 | -10.38 |  |  |  |  |  |  |
| REAL | B net | 28 | +89.29 | +3.14 | +88.05 | +1.73 | -9.44 | +0.06 | -0.00 | +0.19 | +0.65 | -0.00 | +0.06 |
| REAL | Bp paid-only | 28 | +89.29 | +3.14 | +88.05 | +1.73 | -9.44 | +0.06 | +0.00 | +0.19 | +0.65 | +0.00 | +0.06 |
| REAL w/o KERNEL #25 | A (today) | 27 | +92.59 | +3.65 | +98.43 | +1.95 | -6.33 |  |  |  |  |  |  |
| REAL w/o KERNEL #25 | B net | 27 | +92.59 | +3.65 | +98.42 | +1.95 | -6.33 | -0.00 | -0.00 | +0.00 | +0.00 | -0.00 | -0.00 |
| REAL w/o KERNEL #25 | Bp paid-only | 27 | +92.59 | +3.65 | +98.43 | +1.95 | -6.33 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| REAL stack_keep (pre-DL 601) | A (today) | 32 | +84.38 | +2.36 | +75.60 | +1.45 | -16.05 |  |  |  |  |  |  |
| REAL stack_keep (pre-DL 601) | B net | 32 | +84.38 | +2.41 | +77.18 | +1.48 | -15.18 | +0.05 | -0.00 | +0.17 | +0.65 | -0.00 | +0.05 |
| REAL stack_keep (pre-DL 601) | Bp paid-only | 32 | +84.38 | +2.41 | +77.20 | +1.48 | -15.16 | +0.05 | +0.00 | +0.17 | +0.65 | +0.00 | +0.05 |
| BT DESIGN | A (today) | 94 | +76.60 | +1.30 | +122.15 | +0.18 | -22.00 |  |  |  |  |  |  |
| BT DESIGN | B net | 94 | +76.60 | +1.35 | +126.71 | +0.19 | -21.05 | +0.05 | +0.01 | +0.10 | +1.00 | +0.03 | +0.05 |
| BT DESIGN | Bp paid-only | 94 | +76.60 | +1.35 | +126.82 | +0.19 | -21.03 | +0.05 | +0.01 | +0.10 | +1.00 | +0.03 | +0.05 |
| BT HOLDOUT | A (today) | 451 | +75.61 | +1.10 | +497.61 | +0.52 | -41.26 |  |  |  |  |  |  |
| BT HOLDOUT | B net | 451 | +75.39 | +1.11 | +502.19 | +0.53 | -37.50 | +0.01 | -0.06 | +0.05 | +0.69 | +0.00 | +0.04 |
| BT HOLDOUT | Bp paid-only | 451 | +75.39 | +1.11 | +502.73 | +0.53 | -37.47 | +0.01 | -0.06 | +0.06 | +0.70 | +0.00 | +0.04 |
| BT ALL | A (today) | 545 | +75.78 | +1.14 | +619.77 | +0.38 | -41.26 |  |  |  |  |  |  |
| BT ALL | B net | 545 | +75.60 | +1.15 | +628.91 | +0.39 | -37.50 | +0.02 | -0.04 | +0.06 | +0.76 | +0.01 | +0.04 |
| BT ALL | Bp paid-only | 545 | +75.60 | +1.16 | +629.55 | +0.39 | -37.47 | +0.02 | -0.04 | +0.06 | +0.77 | +0.01 | +0.04 |

**FLATBASE_SHORT — what changed (SAVED = A stopped anyway, B out earlier · CUT = A reached TP / time, B stopped · WIDENED = received funding moved B's line out past A's stop bar) · $ = Σ Δ % × 61 (a $10,000 account at one size)**

| block | arm | unchanged_N | unchanged_max_abs_d | exactness_violations | SAVED N | SAVED Σ% | SAVED $ | CUT N | CUT Σ% | CUT $ | WIDENED N | WIDENED Σ% | WIDENED $ | OTHER N | OTHER Σ% | OTHER $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | B | 25 | +0.00 | 0 | 3 | +1.60 | +97.35 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL | Bp | 27 | +0.00 | 0 | 1 | +1.60 | +97.58 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL w/o KERNEL #25 | B | 25 | +0.00 | 0 | 2 | -0.00 | -0.23 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL w/o KERNEL #25 | Bp | 27 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL stack_keep (pre-DL 601) | B | 28 | +0.00 | 0 | 4 | +1.58 | +96.13 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL stack_keep (pre-DL 601) | Bp | 31 | +0.00 | 0 | 1 | +1.60 | +97.58 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | B | 78 | +0.00 | 0 | 16 | +4.56 | +278.38 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | Bp | 85 | +0.00 | 0 | 9 | +4.67 | +284.92 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | B | 391 | +0.00 | 0 | 58 | +17.37 | +1059.39 | 1 | -12.71 | -775.29 | 1 | -0.08 | -4.76 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | Bp | 412 | +0.00 | 0 | 38 | +17.83 | +1087.53 | 1 | -12.71 | -775.29 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT ALL | B | 469 | +0.00 | 0 | 74 | +21.93 | +1337.78 | 1 | -12.71 | -775.29 | 1 | -0.08 | -4.76 | 0 | +0.00 | +0.00 |
| BT ALL | Bp | 497 | +0.00 | 0 | 47 | +22.50 | +1372.45 | 1 | -12.71 | -775.29 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |

**FLATBASE_SHORT real — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| KERNELUSDT 2026-11-20 | -11.975 | STOP | -10.375 | STOP | +1.600 | +1.595 | True | -0.192 |
| MMTUSDT 2026-11-19 | -10.378 | STOP | -10.380 | STOP | -0.003 | -0.003 | False | +0.003 |
| BANKUSDT 2026-10-27 | -10.379 | STOP | -10.380 | STOP | -0.001 | -0.001 | False | +0.005 |

**FLATBASE_SHORT backtest — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| ORCAUSDT 2025-11-27 | +2.240 | TP | -10.470 | STOP | -12.710 | +2.698 | True | -0.089 |
| LRCUSDT 2025-12-12 | -13.891 | STOP | -10.878 | STOP | +3.013 | +2.990 | True | -0.064 |
| STORJUSDT 2025-12-28 | -12.551 | STOP | -10.622 | STOP | +1.929 | +1.919 | True | -0.099 |
| ENJUSDT 2026-03-18 | -11.792 | STOP | -10.230 | STOP | +1.562 | +1.560 | True | -0.056 |
| LPTUSDT 2025-05-31 | -11.909 | STOP | -10.418 | STOP | +1.491 | +1.487 | True | -0.146 |
| MEMEUSDT 2026-01-20 | -12.074 | STOP | -10.872 | STOP | +1.202 | +1.193 | False | -0.224 |
| HYPERUSDT 2025-12-10 | -11.483 | STOP | -10.387 | STOP | +1.096 | +1.093 | False | -0.122 |
| AGLDUSDT 2024-11-04 | -11.968 | STOP | -10.895 | STOP | +1.073 | +1.065 | False | -0.341 |
| STEEMUSDT 2026-02-24 | -11.183 | STOP | -10.116 | STOP | +1.067 | +1.066 | True | -0.180 |
| LAYERUSDT 2026-06-22 | -11.161 | STOP | -10.215 | STOP | +0.946 | +0.944 | False | -0.258 |
| MAGICUSDT 2025-12-09 | -11.003 | STOP | -10.240 | STOP | +0.763 | +0.762 | False | -0.351 |
| XAIUSDT 2026-03-11 | -10.916 | STOP | -10.242 | STOP | +0.674 | +0.673 | False | -0.120 |

**FLATBASE_SHORT real — per-pair Δ (arm B − A, Σ % pts)**: 3 pairs move · Σ+ +1.60 · Σ− -0.00 · top gainer share of Σ+ 100 % · top loser share of Σ− 67 %

| pair | d_sum |
|---|---|
| MMTUSDT | -0.00 |
| BANKUSDT | -0.00 |
| KERNELUSDT | +1.60 |

**FLATBASE_SHORT backtest — per-pair Δ (arm B − A, Σ % pts)**: 66 pairs move · Σ+ +22.46 · Σ− -13.32 · top gainer share of Σ+ 15 % · top loser share of Σ− 95 %

| pair | d_sum |
|---|---|
| ORCAUSDT | -12.71 |
| MUBARAKUSDT | -0.10 |
| DEGENUSDT | -0.09 |
| B3USDT | -0.09 |
| FLOCKUSDT | -0.06 |
| LPTUSDT | +1.49 |
| MEMEUSDT | +1.71 |
| ENJUSDT | +1.73 |
| STORJUSDT | +1.93 |
| LRCUSDT | +3.36 |

**FLATBASE_SHORT real — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 3 | +66.67 | -0.70 | -2.11 | +0.53 | 1 |
| rest | 25 | +92.00 | +3.54 | +88.57 | +0.16 | 2 |

**FLATBASE_SHORT backtest — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 47 | +76.60 | +0.65 | +30.46 | +0.90 | 10 |
| rest | 498 | +75.70 | +1.18 | +589.30 | +0.14 | 108 |

## EXHAUSTION_SHORT

**EXHAUSTION_SHORT — funding paid (% of notional; + = paid by the short) under today's exit (arm A)**

| cohort | N | hourly_at_entry_pct | paying_pct | paid_median | paid_p90 | paid_worst | received_best | TP N | TP paid avg | STOP N | STOP paid avg | TIME N | TIME paid avg | corr(entry rate, received) | spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | +20.00 | +20.00 | +0.00 | +1.87 | +3.11 | +0.00 | 4 | +0.78 | 1 | +0.00 | 0 |  | -0.19 | +0.00 |
| backtest | 3368 | +2.49 | +33.28 | +0.00 | +0.69 | +4.83 | +0.28 | 1127 | +0.30 | 1933 | +0.12 | 308 | +0.28 | +0.32 | +0.49 |

**EXHAUSTION_SHORT — arm A vs B (net stop) vs Bp (paid-only) · one size 0.61 × equity · Δ = arm − A per fill, day-block bootstrap 95 % CI**

| block | arm | N | WR | avg | sum | DCR | maxDD | d_avg | d_lo | d_hi | P_gain | loo_min | loo_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | A (today) | 5 | +80.00 | +5.08 | +25.41 | +3.84 | +0.00 |  |  |  |  |  |  |
| REAL | B net | 5 | +80.00 | +5.08 | +25.41 | +3.84 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| REAL | Bp paid-only | 5 | +80.00 | +5.08 | +25.41 | +3.84 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| BT DESIGN | A (today) | 1349 | +39.81 | +0.57 | +768.17 | +0.48 | -52.98 |  |  |  |  |  |  |
| BT DESIGN | B net | 1349 | +39.36 | +0.57 | +773.82 | +0.48 | -51.07 | +0.00 | -0.04 | +0.04 | +0.59 | -0.00 | +0.01 |
| BT DESIGN | Bp paid-only | 1349 | +39.21 | +0.56 | +761.47 | +0.47 | -52.45 | -0.00 | -0.05 | +0.03 | +0.42 | -0.01 | +0.00 |
| BT HOLDOUT | A (today) | 2019 | +43.14 | +0.74 | +1501.80 | +1.37 | -67.64 |  |  |  |  |  |  |
| BT HOLDOUT | B net | 2019 | +42.05 | +0.69 | +1399.81 | +1.26 | -67.32 | -0.05 | -0.12 | +0.01 | +0.05 | -0.05 | -0.04 |
| BT HOLDOUT | Bp paid-only | 2019 | +42.05 | +0.69 | +1400.94 | +1.26 | -67.30 | -0.05 | -0.11 | +0.01 | +0.05 | -0.05 | -0.04 |
| BT ALL | A (today) | 3368 | +41.81 | +0.67 | +2269.97 | +0.84 | -67.64 |  |  |  |  |  |  |
| BT ALL | B net | 3368 | +40.97 | +0.65 | +2173.63 | +0.80 | -67.32 | -0.03 | -0.07 | +0.01 | +0.08 | -0.03 | -0.02 |
| BT ALL | Bp paid-only | 3368 | +40.91 | +0.64 | +2162.41 | +0.79 | -67.30 | -0.03 | -0.07 | +0.00 | +0.05 | -0.03 | -0.03 |

**EXHAUSTION_SHORT — what changed (SAVED = A stopped anyway, B out earlier · CUT = A reached TP / time, B stopped · WIDENED = received funding moved B's line out past A's stop bar) · $ = Σ Δ % × 61 (a $10,000 account at one size)**

| block | arm | unchanged_N | unchanged_max_abs_d | exactness_violations | SAVED N | SAVED Σ% | SAVED $ | CUT N | CUT Σ% | CUT $ | WIDENED N | WIDENED Σ% | WIDENED $ | OTHER N | OTHER Σ% | OTHER $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | B | 5 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL | Bp | 5 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | B | 1017 | +0.00 | 0 | 311 | +59.87 | +3651.96 | 9 | -70.60 | -4306.62 | 12 | +16.39 | +999.49 | 0 | +0.00 | +0.00 |
| BT DESIGN | Bp | 1181 | +0.00 | 0 | 159 | +63.90 | +3898.15 | 9 | -70.60 | -4306.62 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | B | 1602 | +0.00 | 0 | 393 | +174.85 | 10,666 | 22 | -276.81 | -16,885 | 2 | -0.03 | -2.10 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | Bp | 1704 | +0.00 | 0 | 293 | +175.94 | 10,733 | 22 | -276.81 | -16,885 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT ALL | B | 2619 | +0.00 | 0 | 704 | +234.72 | 14,318 | 31 | -347.41 | -21,192 | 14 | +16.35 | +997.39 | 0 | +0.00 | +0.00 |
| BT ALL | Bp | 2885 | +0.00 | 0 | 452 | +239.85 | 14,631 | 31 | -347.41 | -21,192 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |

EXHAUSTION_SHORT real: no fill changes under arm B.

**EXHAUSTION_SHORT backtest — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| SHELLUSDT 2026-11-15 | +9.733 | TP | -6.309 | STOP | -16.042 | +0.099 | False | +0.005 |
| SENTUSDT 2026-06-09 | +8.827 | TP | -6.308 | STOP | -15.134 | +1.005 | False | -0.018 |
| COSUSDT 2026-05-13 | +8.676 | TP | -6.307 | STOP | -14.983 | +1.156 | False | -0.734 |
| ARPAUSDT 2026-01-19 | +8.653 | TP | -6.307 | STOP | -14.960 | +1.179 | False | -0.044 |
| NILUSDT 2026-07-27 | +9.337 | TP | -5.406 | STOP | -14.743 | +0.495 | False | -0.040 |
| ONEUSDT 2026-09-28 | +8.894 | TP | -5.785 | STOP | -14.679 | +0.938 | True | -0.203 |
| MINAUSDT 2025-11-04 | +9.027 | TP | -5.097 | STOP | -14.124 | +0.805 | False | -0.130 |
| SOPHUSDT 2026-10-30 | +7.713 | TP | -6.306 | STOP | -14.019 | +2.119 | False | -0.195 |
| KATUSDT 2026-03-26 | +7.428 | TP | -6.426 | STOP | -13.854 | +2.404 | False | +0.003 |
| RONINUSDT 2026-05-19 | +7.312 | TP | -6.308 | STOP | -13.619 | +2.520 | False | -1.396 |
| LAYERUSDT 2026-05-10 | +7.007 | TP | -6.305 | STOP | -13.313 | +2.825 | False | -0.013 |
| ZILUSDT 2026-07-28 | +8.098 | TP | -5.048 | STOP | -13.146 | +1.734 | False | -0.305 |

EXHAUSTION_SHORT real: Δ is zero for every pair.

**EXHAUSTION_SHORT backtest — per-pair Δ (arm B − A, Σ % pts)**: 305 pairs move · Σ+ +199.16 · Σ− -295.49 · top gainer share of Σ+ 4 % · top loser share of Σ− 8 %

| pair | d_sum |
|---|---|
| SXPUSDT | -24.40 |
| SPELLUSDT | -17.74 |
| SHELLUSDT | -16.04 |
| ONEUSDT | -14.69 |
| NILUSDT | -14.59 |
| HYPERUSDT | +4.65 |
| LSKUSDT | +4.65 |
| REDUSDT | +5.13 |
| GMTUSDT | +6.56 |
| TUSDT | +8.53 |

**EXHAUSTION_SHORT real — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 1 | +100.00 | +6.82 | +6.82 | +3.11 | 0 |
| rest | 4 | +75.00 | +4.65 | +18.59 | +0.00 | 1 |

**EXHAUSTION_SHORT backtest — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 83 | +51.81 | +1.41 | +117.43 | +0.99 | 40 |
| rest | 3285 | +41.55 | +0.66 | +2152.54 | +0.17 | 1893 |

## DISTRIBUTION_SHORT

**DISTRIBUTION_SHORT — funding paid (% of notional; + = paid by the short) under today's exit (arm A)**

| cohort | N | hourly_at_entry_pct | paying_pct | paid_median | paid_p90 | paid_worst | received_best | TP N | TP paid avg | STOP N | STOP paid avg | TIME N | TIME paid avg | corr(entry rate, received) | spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | +0.00 | +0.00 | +0.00 | -0.00 | -0.00 | +0.07 | 1 | +0.00 | 4 | -0.02 | 0 |  | +0.45 | +0.71 |
| backtest | 1547 | +1.36 | +19.13 | +0.00 | +0.23 | +4.73 | +0.29 | 537 | +0.14 | 881 | +0.04 | 129 | +0.16 | +0.19 | +0.37 |

**DISTRIBUTION_SHORT — arm A vs B (net stop) vs Bp (paid-only) · one size 0.61 × equity · Δ = arm − A per fill, day-block bootstrap 95 % CI**

| block | arm | N | WR | avg | sum | DCR | maxDD | d_avg | d_lo | d_hi | P_gain | loo_min | loo_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | A (today) | 5 | +20.00 | -2.32 | -11.58 | -1.87 | -9.33 |  |  |  |  |  |  |
| REAL | B net | 5 | +20.00 | -2.33 | -11.65 | -1.88 | -9.38 | -0.01 | -0.02 | +0.00 | +0.00 | -0.02 | +0.00 |
| REAL | Bp paid-only | 5 | +20.00 | -2.32 | -11.58 | -1.87 | -9.33 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| BT DESIGN | A (today) | 559 | +41.68 | +0.98 | +546.67 | +0.39 | -34.62 |  |  |  |  |  |  |
| BT DESIGN | B net | 559 | +41.32 | +0.96 | +536.84 | +0.38 | -34.76 | -0.02 | -0.09 | +0.03 | +0.33 | -0.02 | +0.01 |
| BT DESIGN | Bp paid-only | 559 | +41.32 | +0.96 | +538.34 | +0.38 | -34.46 | -0.01 | -0.08 | +0.03 | +0.36 | -0.02 | +0.01 |
| BT HOLDOUT | A (today) | 988 | +42.71 | +0.87 | +861.00 | +0.85 | -55.07 |  |  |  |  |  |  |
| BT HOLDOUT | B net | 988 | +42.21 | +0.84 | +832.41 | +0.81 | -55.09 | -0.03 | -0.09 | +0.02 | +0.14 | -0.03 | -0.01 |
| BT HOLDOUT | Bp paid-only | 988 | +42.21 | +0.84 | +834.07 | +0.81 | -54.98 | -0.03 | -0.08 | +0.02 | +0.15 | -0.03 | -0.01 |
| BT ALL | A (today) | 1547 | +42.34 | +0.91 | +1407.66 | +0.58 | -55.07 |  |  |  |  |  |  |
| BT ALL | B net | 1547 | +41.89 | +0.89 | +1369.25 | +0.56 | -55.09 | -0.02 | -0.07 | +0.01 | +0.11 | -0.03 | -0.01 |
| BT ALL | Bp paid-only | 1547 | +41.89 | +0.89 | +1372.41 | +0.56 | -54.98 | -0.02 | -0.07 | +0.01 | +0.13 | -0.02 | -0.01 |

**DISTRIBUTION_SHORT — what changed (SAVED = A stopped anyway, B out earlier · CUT = A reached TP / time, B stopped · WIDENED = received funding moved B's line out past A's stop bar) · $ = Σ Δ % × 61 (a $10,000 account at one size)**

| block | arm | unchanged_N | unchanged_max_abs_d | exactness_violations | SAVED N | SAVED Σ% | SAVED $ | CUT N | CUT Σ% | CUT $ | WIDENED N | WIDENED Σ% | WIDENED $ | OTHER N | OTHER Σ% | OTHER $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | B | 4 | +0.00 | 0 | 1 | -0.07 | -4.29 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL | Bp | 5 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | B | 471 | +0.00 | 0 | 81 | +10.84 | +661.22 | 2 | -20.26 | -1235.80 | 5 | -0.40 | -24.57 | 0 | +0.00 | +0.00 |
| BT DESIGN | Bp | 525 | +0.00 | 0 | 32 | +11.93 | +727.78 | 2 | -20.26 | -1235.80 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | B | 862 | +0.00 | 0 | 119 | +28.30 | +1726.40 | 5 | -56.80 | -3465.09 | 2 | -0.08 | -5.01 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | Bp | 918 | +0.00 | 0 | 65 | +29.88 | +1822.62 | 5 | -56.80 | -3465.09 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT ALL | B | 1333 | +0.00 | 0 | 200 | +39.14 | +2387.62 | 7 | -77.06 | -4700.88 | 7 | -0.48 | -29.58 | 0 | +0.00 | +0.00 |
| BT ALL | Bp | 1443 | +0.00 | 0 | 97 | +41.81 | +2550.40 | 7 | -77.06 | -4700.88 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |

**DISTRIBUTION_SHORT real — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| EVAAUSDT 2026-11-19 | -4.786 | STOP | -4.857 | STOP | -0.070 | -0.070 | False | +0.053 |

**DISTRIBUTION_SHORT backtest — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| LEVERUSDT 2025-06-30 | +9.358 | TP | -6.308 | STOP | -15.666 | +0.474 | False | -0.008 |
| STEEMUSDT 2026-09-13 | +8.590 | TP | -6.307 | STOP | -14.897 | +1.242 | False | +0.002 |
| MIRAUSDT 2026-02-26 | +7.417 | TP | -5.607 | STOP | -13.024 | +2.415 | False | -0.004 |
| CVXUSDT 2026-01-03 | +5.099 | TP | -5.986 | STOP | -11.086 | +4.733 | False | +0.005 |
| METUSDT 2026-04-22 | +4.424 | TIME | -6.309 | STOP | -10.733 | +0.427 | False | -0.000 |
| ZORAUSDT 2026-09-04 | +1.629 | TIME | -5.436 | STOP | -7.065 | +0.497 | False | -0.100 |
| ONTUSDT 2025-01-13 | +1.590 | TIME | -3.003 | STOP | -4.593 | +4.615 | False | +0.010 |
| RVNUSDT 2025-06-05 | -9.087 | STOP | -6.305 | STOP | +2.782 | +2.778 | False | -1.066 |
| ZILUSDT 2026-02-02 | -7.866 | STOP | -5.301 | STOP | +2.566 | +2.563 | False | -0.020 |
| ONEUSDT 2026-10-03 | -7.127 | STOP | -4.886 | STOP | +2.241 | +2.238 | False | -0.017 |
| BMTUSDT 2026-11-03 | -7.047 | STOP | -4.841 | STOP | +2.206 | +2.203 | False | -0.933 |
| STORJUSDT 2026-05-15 | -7.745 | STOP | -5.944 | STOP | +1.801 | +1.799 | False | -0.008 |

**DISTRIBUTION_SHORT real — per-pair Δ (arm B − A, Σ % pts)**: 1 pairs move · Σ+ +0.00 · Σ− -0.07 · top gainer share of Σ+ nan % · top loser share of Σ− 100 %

| pair | d_sum |
|---|---|
| EVAAUSDT | -0.07 |

**DISTRIBUTION_SHORT backtest — per-pair Δ (arm B − A, Σ % pts)**: 171 pairs move · Σ+ +41.32 · Σ− -79.73 · top gainer share of Σ+ 7 % · top loser share of Σ− 20 %

| pair | d_sum |
|---|---|
| LEVERUSDT | -15.67 |
| STEEMUSDT | -14.90 |
| MIRAUSDT | -13.02 |
| CVXUSDT | -11.09 |
| METUSDT | -10.73 |
| BLURUSDT | +1.80 |
| BMTUSDT | +2.21 |
| ONEUSDT | +2.24 |
| ZILUSDT | +2.53 |
| RVNUSDT | +2.78 |

**DISTRIBUTION_SHORT real — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 0 |  |  | +0.00 |  | 0 |
| rest | 5 | +20.00 | -2.32 | -11.58 | -0.01 | 4 |

**DISTRIBUTION_SHORT backtest — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 19 | +63.16 | +4.24 | +80.63 | +0.32 | 7 |
| rest | 1528 | +42.08 | +0.87 | +1327.03 | +0.08 | 874 |
