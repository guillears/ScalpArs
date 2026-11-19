# DIST_THIN_TP: a lower take profit for thin DISTRIBUTION coins? (2026-11-19)

**Verdict: NO. 0 of 5 cells pass. On HOLDOUT every lower take profit is flat to negative (−0.07 … +0.09 per thin fill, CIs around 0); 2026-H2 is negative for all five. Keep Exit B for every DIST fill.**

## Question
Thin coins (24 h volume ≤ $42M) reach less far and give back more (DIST_THIN_DEEPDIVE). Would a lower take profit on thin coins only, with today's stop, pay?

## Method (pre-registered: DIST_THIN_TP_2026-11-19_PREREG.md)
- 1,534 RSI-kept DIST fills (673 thin), the DIST_EXIT 1m engine unchanged (parity with the stored Exit B nets: 1 row of 1,534 differs by > 0.05 — by 1.80 — mean Δ −0.001).
- Thin fills only: TP = min(X, today's TP), X ∈ {4, 5, 6, 7, 8} %; stop min(1.5 × ATR15, 6 %); 24 h. Liquid fills unchanged (also priced as a control).
- Paired Δ net % per thin fill vs Exit B; DESIGN < 2025-07 ≤ HOLDOUT; day-block bootstrap; family-wise sign-flip null; per-half-year; concentration; thin-vs-liquid contrast. Script: `scripts/dist_thin_tp_study.py`.

## Results
Cohort 1534 fills (thin 673); family-wise null 95th pct of the max DESIGN mean Δ = +0.576

| cell (thin only) | DESIGN mean Δ (95 % CI) | HOLDOUT mean Δ (CI) | Σ Δ thin (D / H) | halves Δ > 0 | top pair share | liquid control Δ (D / H) | thin − liquid (D / H) | pass |
|---|---|---|---|---|---|---|---|---|
| TP 4 % | +0.271 (-0.43 … +0.92) | -0.065 (-0.66 … +0.51) | +50 / -32 | 4/7 | 167% | -1.233 / -0.635 | +1.503 / +0.569 | ✗ |
| TP 5 % | +0.338 (-0.27 … +0.90) | -0.030 (-0.56 … +0.45) | +63 / -15 | 5/7 | 72% | -1.011 / -0.365 | +1.349 / +0.335 | ✗ |
| TP 6 % | +0.110 (-0.37 … +0.55) | -0.006 (-0.40 … +0.41) | +20 / -3 | 4/7 | 138% | -0.869 / -0.275 | +0.979 / +0.269 | ✗ |
| TP 7 % | +0.195 (-0.21 … +0.63) | -0.016 (-0.34 … +0.31) | +36 / -8 | 5/7 | 93% | -0.777 / -0.260 | +0.972 / +0.245 | ✗ |
| TP 8 % | +0.140 (-0.14 … +0.44) | +0.094 (-0.19 … +0.39) | +26 / +46 | 5/7 | 39% | -0.547 / -0.094 | +0.686 / +0.188 | ✗ |

### Decomposition on thin fills (all periods): who gains, who pays

| cell | Exit-B-stopped fills: saved (N · Σ) | stopped deeper (N · Σ) | Exit-B take-profit / time fills cut (N · Σ) | unchanged (N, Δ = 0 check) |
|---|---|---|---|---|
| TP 4 % | 131 · +1181 | 0 · +0 | 267 · -1163 | 275 (max |Δ| 0.000) |
| TP 5 % | 95 · +960 | 0 · +0 | 259 · -913 | 319 (max |Δ| 0.029) |
| TP 6 % | 62 · +702 | 0 · +0 | 253 · -685 | 358 (max |Δ| 0.015) |
| TP 7 % | 38 · +475 | 0 · +0 | 248 · -446 | 387 (max |Δ| 0.000) |
| TP 8 % | 25 · +341 | 0 · +0 | 229 · -269 | 419 (max |Δ| 0.035) |

### Thin fills per half-year: mean Δ per cell

| half | N thin | TP 4 | TP 5 | TP 6 | TP 7 | TP 8 |
|---|---|---|---|---|---|---|
| 23-H2 | 35 | -0.16 | +0.12 | -0.08 | +0.02 | -0.15 |
| 24-H1 | 43 | +0.50 | +0.68 | +0.56 | +0.56 | +0.38 |
| 24-H2 | 47 | +0.07 | +0.17 | -0.12 | -0.08 | +0.08 |
| 25-H1 | 60 | +0.51 | +0.35 | +0.09 | +0.25 | +0.19 |
| 25-H2 | 198 | +0.04 | +0.22 | +0.09 | +0.01 | +0.16 |
| 26-H1 | 157 | -0.12 | -0.01 | +0.12 | +0.23 | +0.18 |
| 26-H2 | 133 | -0.16 | -0.43 | -0.30 | -0.35 | -0.12 |

Note: "top pair share" = |the largest single pair's Σ Δ| ÷ |Σ Δ| over all periods; it exceeds 100 % when pairs offset each other.

## What it means
- **The mechanism is real but it nets out.** A lower TP saves thin trades that Exit B stops after a move our way (TP 5: 95 trades, +960 points). But it cuts thin trades that would have run to the 10 % TP or the 24 h close (259 trades, −913 points). Saved ≈ cut at 4–6 %; at 7–8 % the save is a little larger (TP 8: +341 vs −269), but small and not out of sample.
- **Thin coins do benefit more than liquid ones** (the thin − liquid column is positive everywhere). On liquid coins a lower TP is clearly harmful (−0.1 … −1.2 per fill). But "less harmful than on liquid coins" is not "better than today": on thin coins the gain is zero out of sample.
- **The DESIGN gains are small and none clears its CI or the shuffled null (+0.58).** HOLDOUT is −0.07 … +0.09. The most recent half-year (26-H2, 133 thin fills) is negative for every level.
- **TP 8 %** is the closest (+0.14 / +0.09), but it fails the CI, the null and concentration (top pair 39 %). It is not a candidate; at most a line to re-read if thin coins keep giving back.

## Verdict
Keep Exit B for thin and liquid DIST fills. The thin-coin lever stays where DIST_THIN_DEEPDIVE put it: size at the pre-live review, plus the real thin-coin slippage read.
