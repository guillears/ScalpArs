# DIST red-candle 2D — result (2026-11-24, research only; PREREG reports/DIST_REDCANDLE_2D_2026-11-24_PREREG.md frozen before computing)

Cohort: today's DIST rule in the backtest (RSI1h ≥ 74.07, Exit B net), 1,536 trades; BIG red = bar_ret < −3 % (771; 554 on HOLDOUT).
Cuts frozen on DESIGN big-red trades; read once on HOLDOUT; family-wise one-sided 95 % t = 2.47 (2,000 shuffles). Script
scripts/DIST_REDCANDLE_2D_2026-11-24.py · table reports/DIST_REDCANDLE_2D_2026-11-24.csv.

| theory (BIG red + …) | predicted good side | DESIGN good vs bad | HOLDOUT good vs bad | t | fw p |
|---|---|---|---|---|---|
| T1/T2 closes at the low, no lower wick (identical columns) | low | +0.93 vs +1.19 (wrong way) | +1.47 vs +1.20 | 0.41 | 0.96 |
| T3 open interest falling through the drop | falling | +1.63 vs +0.48 | +1.46 vs +1.01 | 0.68 | 0.88 |
| T4 crowd not already short (funding) | high | +1.29 vs +0.80 | +0.80 vs +1.53 (flips) | −1.04 | 1.00 |
| T5 top traders still cutting longs | falling | +1.05 vs +1.06 | +0.83 vs +1.74 (wrong way) | −1.41 | 1.00 |
| T6 room to fall above the base | high | +1.22 vs +0.89 | +1.13 vs +1.89 (flips) | −1.04 | 1.00 |
| T7 BTC weak | weak | +0.20 vs +1.92 (wrong way) | +1.52 vs +0.99 | 0.82 | 0.82 |
| T8 real volume | high | +1.41 vs +0.70 | +1.07 vs +1.52 (flips) | −0.71 | 1.00 |

Verdict: no theory pair survives (best HOLDOUT t 0.82 vs 2.47); 5 of 7 change sign DESIGN → HOLDOUT (3 predicted-then-reversed). Loser cells
on the small-red side (filter candidates; cuts = the frozen DESIGN big-red medians): SMALL + BTC not weak 362 · +0.33 (CI −0.34 … +1.01, HOLDOUT
−0.12) and SMALL + OI rising 516 · +0.51 (CI −0.06 … +1.10, HOLDOUT −0.02) — both breakeven, not losers → fail the EXPECTANCY bar. Batch: 2
big-red fills — cannot decide. No 2D line; the 1D DIST_BIG_RED watch is registered (DL 639).

## Is −3 % the right division? (operator question) — no
−3.0 was the median (−3.007), chosen so the cut could not be fitted. Quintiles of the red candle (avg % · DESIGN · HOLDOUT): ≤ −7.1 +1.75 · +1.80 ·
+1.74 | −7.1 … −3.9 +0.66 | −3.9 … −2.3 +0.97 | −2.3 … −1.1 +0.38 | > −1.1 +0.71. The effect lives in the VERY BIG fifth only (+ in 7/7 half-years,
ahead of the rest in 6/7); at −3.0 the DESIGN gap is +0.01. Every bucket is profitable → no filter; the watch reads 3 buckets (≤ −7.1 · −7.1 … −3.0 ·
> −3.0).
