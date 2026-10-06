# FRENZY ON-scalp — "buy the moment FRENZY turns ON" (2026-10-06)

**Status:** research only, unreviewed (no caveman / deep review has been run yet). No code or config was changed, and nothing was committed. All rules, the grid and the gates were frozen in `reports/FRENZY_ON_SCALP_PREREG_2026-10-06.txt` at 18:10 UTC, before any outcome was computed. The ADX/DI subsets were added by the coordinator before any outcome was computed, and they are in the same file.

## Bottom line (plain words)

1. **"+X % is guaranteed" is only true for a tiny X, and only if you never stop out.** After 0.19 % of costs, +0.3 % is reached within 24 h on 98 % of ON signals (day-CI 97–98 %). +1 % is reached on 92 %, and within 60 minutes on 81 %. So the operator's eye is right: the pop usually comes.
2. **The signals that don't pop sink, and sink deep.** With "+1 % within 60 min, else exit at 60 min, no stop", 81 % win +1.0, and the other 19 % lose **−4.8 % on average, which is 4.8 times the win**. Net result: −0.08 % per trade. Holding the misers to 24 h is worse: they average −16.8 %. **54 % of all ON signals are 10 % or more under the entry at some point within 12 h**, and the median signal sits at **−7.5 % after 24 h**. FRENZY pumps mean-revert.
3. **No exit design passes.** That covers fixed TP, TP + stop, trail, the live lock, and every X / T / Y / Z in the frozen grid: 0 of 1,205 cells pass. None has a day-CI above 0, and the best max-t adjusted p is 0.63. The best cell on the operator's own "strong" condition (S3: ADX rising ∧ +DI > −DI) is TP +3 / 2 h / no stop: **+0.15 %/trade, CI −0.18…+0.46, adjusted p 0.97**. When the best cell is chosen on one half of the year, it reads **+0.03 / +0.06** on the other half, with both CIs spanning 0.
4. **The ADX/DI condition does not create the edge.** S3 is about +0.1–0.2 % better than "all ON" on the no-stop exits, which is within noise. The control flank S4 (ADX falling ∧ −DI > +DI, N = 82) is the best-looking subset, which is the opposite of the thesis. That is small-N noise (max-t p 0.63–0.91). The 3×3 tercile screen is non-monotone, so it is a confound, not a filter. S3's real value lives in the **live lock**: on live-taken FRENZY_LONG fills, strong earns +0.67 vs +0.02 under LOCK2. Under a quick-scalp exit, the same strong fills earn **−0.53**. So the strong flag picks runners. It does not pick "guaranteed pops", and ON-scalp is not the same thing as strong.
5. **"No stop" means ruin.** Liquidation (−15.8 % at 6×, −23.8 % at 4×) is the only stop, and it fires up to 153 times in the year depending on the rule. In books from $3k with 2+2 slots, every S0/S3 design ends below $3k, with P(book DD ≥ 50 %) between **80 % and 100 %**. The operator's literal rule ("TP +1, no stop, hold until it comes") turns $3k into **$0–$78** in the year. The worst single signals fell 50–60 % within 24 h (ACE 08-14 −59 %, ZKJ 04-28 −56 %).
6. **Verdict: not a new strategy.** In the letter of the pre-registration, 70 cells qualify as "observe candidates" (mean > 0 in both halves but failing the CI and p): S3 8, S1 11, S4 51. All of them are no-stop exits, and all fail the ruin leg. If anything is logged, make it one free observe-only scout line (frozen below). Do not build a probe.

## Engine parity (checked first)

| check | result |
|---|---|
| Signal bars = engine bars | Cohort `FRENZY_ENGINE_COHORT_2026-10-05.csv` was built by walking every pair with the real `services.frenzy.frenzy_walk` / `frenzy_long_status` / `frenzy_wide_ready`. Live check (revalidation report): **8/8** live FRENZY/WIDE fills Oct 3–5 are fresh-ON in the rebuild on exactly the fill's bar, same code. |
| Universe | ∩ `FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv` live_elig (no Alpha / < 90-day / non-ASCII pairs) → **1,443 ON signals, 255 days, 272 pairs, Jan 10 → Sep 27 2026** (5.7 per day, max 16). Every refusal code kept: READY 361, GREEN_BAR 393, ATR_HIGH 689; gvol-blocked 614; disloc > 1 % 101. |
| Pricer | Entry = first print ≥ ON close + 12 s (no dislocation guard), ticks on 1,389 + 21 (12 h ticks + 1m tail), 1m klines on 33. Re-pricing the live lock (house `all_exits` LOCK2) at this entry reproduces the cohort's LOCK2 on **1,340 / 1,340** guard-passing rows (max diff 2e-15). |
| ADX / DI stamps | `frenzy_adx_delta(closed[-300:])` / `frenzy_di_spread(closed[-300:])` recomputed from public 5m klines for all **10** stamped live FRENZY/WIDE fills of B17 (Oct 3–6): **10/10 exact to 3 decimals** (same function, window, bar). |
| Strong sizing (S3 ↔ lev) | Strong ships with commit 181131e (pushed 2026-10-04 14:06 UTC); it applies to FRENZY_LONG only (`not wide`). Post-deploy FRENZY_LONG fills: UMA 10-06 10:10 (ADXΔ −0.003 → normal) and ORCA 10-06 09:40 (−0.503 → normal) both opened at **6×** ✓. SAND 10-04 05:05 / AIN 11:15 / SAND 14:05 qualified as strong but opened **before** the deploy at 6× (expected). WIDE fills at 4× ✓. **No post-deploy strong fill exists yet → the 10× branch is unobserved live.** In the year cohort, S3 = 893 of 1,443 ON bars; S3 ∩ READY (= live strong population) = 207. |


## HEADLINE — subset × exit

Mean net % per trade at 1× (price %), after fees 0.09 + slippage 0.10. Day-CI = day-block bootstrap, one UTC day = one block. "best:" rows were chosen on the same data, and their p-values carry that selection. S1/S2 are in the appendix.

| subset | exit | N | days | WR | mean % | median | day-CI 95 % | Jan–Apr | May–Sep | LOMO min | drop top-10 | worst | CVaR 5 % | max-t adj p | timing-null adj p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S3 | LOCK2(live) | 893 | 245 | 50 % | **-0.06** | +1.67 | -0.27 … +0.15 | -0.05 | -0.07 | -0.13 | -0.18 | -3.6 | -3.2 | 1.00 | — |
| S3 | TP0.5/T15/noSL | 893 | 245 | 82 % | **-0.10** | +0.51 | -0.21 … +0.00 | -0.02 | -0.17 | -0.14 | -0.11 | -19.8 | -5.8 | 1.00 | 1.00 |
| S3 | TP1/T60/noSL | 893 | 245 | 83 % | **+0.00** | +1.01 | -0.18 … +0.17 | +0.03 | -0.02 | -0.05 | -0.01 | -29.9 | -9.3 | 1.00 | 0.97 |
| S3 | TP1/T60/SL3 | 893 | 245 | 70 % | **-0.17** | +1.00 | -0.29 … -0.05 | -0.13 | -0.20 | -0.19 | -0.18 | -3.4 | -3.1 | 1.00 | 1.00 |
| S3 | TP2/T120/SL5 | 893 | 245 | 68 % | **-0.14** | +2.00 | -0.35 … +0.05 | -0.05 | -0.23 | -0.23 | -0.17 | -5.5 | -5.1 | 1.00 | 1.00 |
| S3 | TR1z0.5/T60/noSL | 893 | 245 | 83 % | **-0.04** | +0.75 | -0.22 … +0.13 | +0.01 | -0.09 | -0.10 | -0.08 | -29.9 | -9.3 | 1.00 | 0.91 |
| S3 | TR2z1/T120/SL5 | 893 | 245 | 68 % | **-0.20** | +1.26 | -0.40 … +0.00 | -0.10 | -0.29 | -0.29 | -0.27 | -5.5 | -5.1 | 1.00 | 1.00 |
| S3 | best: TP3/T120/noSL | 893 | 245 | 70 % | **+0.15** | +3.00 | -0.18 … +0.46 | +0.06 | +0.22 | +0.06 | +0.11 | -38.7 | -15.6 | 0.97 | 0.85 |
| S3 | best: TP1.5/T30/SL5 | 893 | 245 | 67 % | **-0.09** | +1.50 | -0.25 … +0.07 | -0.00 | -0.17 | -0.15 | -0.11 | -5.5 | -5.0 | 1.00 | 1.00 |
| S3 | best: TR3z1/T120/noSL | 893 | 245 | 70 % | **+0.07** | +2.30 | -0.27 … +0.39 | -0.00 | +0.13 | -0.03 | +0.00 | -38.7 | -15.6 | 1.00 | 0.72 |
| S0 | LOCK2(live) | 1443 | 255 | 49 % | **-0.19** | -3.10 | -0.35 … -0.03 | -0.12 | -0.25 | -0.24 | -0.27 | -3.6 | -3.2 | 1.00 | — |
| S0 | TP0.5/T15/noSL | 1443 | 255 | 79 % | **-0.14** | +0.51 | -0.22 … -0.06 | -0.09 | -0.18 | -0.17 | -0.15 | -19.8 | -5.8 | 1.00 | 1.00 |
| S0 | TP1/T60/noSL | 1443 | 255 | 82 % | **-0.08** | +1.01 | -0.22 … +0.06 | -0.08 | -0.08 | -0.11 | -0.09 | -30.5 | -9.6 | 1.00 | 1.00 |
| S0 | TP1/T60/SL3 | 1443 | 255 | 69 % | **-0.19** | +1.00 | -0.28 … -0.09 | -0.17 | -0.19 | -0.21 | -0.20 | -3.4 | -3.1 | 1.00 | 1.00 |
| S0 | TP2/T120/SL5 | 1443 | 255 | 66 % | **-0.20** | +2.00 | -0.37 … -0.05 | -0.12 | -0.27 | -0.24 | -0.22 | -5.5 | -5.1 | 1.00 | 1.00 |
| S0 | TR1z0.5/T60/noSL | 1443 | 255 | 81 % | **-0.12** | +0.71 | -0.26 … +0.03 | -0.12 | -0.13 | -0.15 | -0.15 | -30.5 | -9.6 | 1.00 | 0.97 |
| S0 | TR2z1/T120/SL5 | 1443 | 255 | 66 % | **-0.26** | +1.24 | -0.44 … -0.10 | -0.18 | -0.33 | -0.31 | -0.31 | -5.5 | -5.1 | 1.00 | 1.00 |
| S0 | best: TP1.5/T60/noSL | 1443 | 255 | 75 % | **-0.05** | +1.50 | -0.23 … +0.11 | -0.04 | -0.06 | -0.07 | -0.06 | -30.5 | -10.5 | 1.00 | 1.00 |
| S0 | best: TP1.5/T30/SL5 | 1443 | 255 | 65 % | **-0.14** | +1.50 | -0.29 … -0.01 | -0.10 | -0.18 | -0.18 | -0.16 | -5.5 | -5.0 | 1.00 | 1.00 |
| S0 | best: TR1.5z1/T60/noSL | 1443 | 255 | 74 % | **-0.11** | +0.86 | -0.29 … +0.06 | -0.09 | -0.13 | -0.14 | -0.15 | -30.5 | -10.5 | 1.00 | 0.99 |
| S4 | LOCK2(live) | 82 | 69 | 49 % | **-0.14** | -3.10 | -0.84 … +0.58 | -0.39 | +0.04 | -0.28 | -0.95 | -3.4 | -3.2 | 1.00 | — |
| S4 | TP0.5/T15/noSL | 82 | 69 | 74 % | **-0.00** | +0.50 | -0.27 … +0.23 | -0.11 | +0.07 | -0.09 | -0.08 | -4.3 | -3.3 | 1.00 | 0.99 |
| S4 | TP1/T60/noSL | 82 | 69 | 85 % | **+0.35** | +1.01 | -0.07 … +0.72 | +0.23 | +0.44 | +0.26 | +0.25 | -7.9 | -5.8 | 0.66 | 0.48 |
| S4 | TP1/T60/SL3 | 82 | 69 | 74 % | **+0.03** | +1.01 | -0.36 … +0.41 | -0.14 | +0.15 | -0.08 | -0.11 | -3.4 | -3.1 | 1.00 | 0.97 |
| S4 | TP2/T120/SL5 | 82 | 69 | 68 % | **+0.08** | +2.00 | -0.61 … +0.69 | -0.03 | +0.16 | -0.02 | -0.19 | -5.1 | -5.0 | 1.00 | 0.91 |
| S4 | TR1z0.5/T60/noSL | 82 | 69 | 84 % | **+0.39** | +0.84 | -0.05 … +0.78 | +0.25 | +0.48 | +0.30 | +0.13 | -7.9 | -5.8 | 0.63 | 0.23 |
| S4 | TR2z1/T120/SL5 | 82 | 69 | 68 % | **-0.00** | +1.40 | -0.65 … +0.61 | -0.10 | +0.06 | -0.15 | -0.46 | -5.1 | -5.0 | 1.00 | 0.93 |
| S4 | best: TP3/T120/noSL | 82 | 69 | 67 % | **+0.58** | +3.00 | -0.26 … +1.39 | +0.23 | +0.82 | +0.39 | +0.23 | -14.2 | -10.2 | 0.86 | 0.15 |
| S4 | best: TP2/T30/SL5 | 82 | 69 | 63 % | **+0.15** | +2.00 | -0.39 … +0.62 | +0.00 | +0.25 | +0.03 | -0.12 | -5.0 | -5.0 | 0.99 | 0.82 |
| S4 | best: TR3z0.5/T120/noSL | 82 | 69 | 67 % | **+0.49** | +2.57 | -0.37 … +1.24 | +0.17 | +0.72 | +0.34 | +0.07 | -14.2 | -10.2 | 0.91 | 0.02 |

### P2-A. Gate legs across the whole frozen family (5 subsets × 241 exits = 1,205 cells)

| subset | cells | mean > 0 | CI low > 0 | both halves > 0 | drop-top-10 > 0 | LOMO min > 0 | max-t p ≤ 0.05 | timing-null adj p ≤ 0.10 | ALL legs |
|---|---|---|---|---|---|---|---|---|---|
| S3 strong (ADX up & +DI>-DI) | 241 | 19 | 0 | 8 | 8 | 5 | 0 | 0 | **0** |
| S0 all ON | 241 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| S4 neither | 241 | 95 | 0 | 51 | 20 | 38 | 0 | 3 | **0** |
| S1 ADX up | 241 | 25 | 0 | 11 | 9 | 6 | 0 | 0 | **0** |
| S2 +DI>-DI | 241 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **0** |
| **family** | 1205 | 139 | 0 | 70 | 37 | 49 | 0 | 3 | **0** |


**Reading the gate table:** no cell has a day-CI above 0. The max-t null (2,000 joint day-block draws, the same statistic on both sides) never goes below 0.05. Only 3 cells (all S4, N = 82) beat the timing null.

The timing null moves each entry to a random moment 30 min–12 h after the ON bar on the same pair, priced on the same 1-second pricer. Its means sit at −0.12…−0.19. So the fresh ON moment is better than a random later moment on the same pump, but "better" is still about zero, not positive.

## PART 1 — the truth before any rule

All numbers are net of 0.19 % costs. "Within T" is measured from the entry, which is the first print ≥ ON close + 12 s.

### P1-A. MFE / MAE after the ON-bar entry (net of 0.19 % costs), all ON signals (S0)

| window | median MFE | median MAE | MFE ≥ +1 | MAE ≤ −3 | MAE ≤ −5 | MAE ≤ −10 | median net at end of window | mean net at end |
|---|---|---|---|---|---|---|---|---|
| 5 min | +0.90 | -1.32 | 46 % | 15 % | 5 % | 1 % | -0.16 | -0.14 |
| 15 min | +1.71 | -2.17 | 65 % | 36 % | 16 % | 3 % | -0.30 | -0.08 |
| 30 min | +2.42 | -3.00 | 74 % | 50 % | 26 % | 6 % | -0.44 | -0.12 |
| 60 min | +3.54 | -4.01 | 81 % | 61 % | 40 % | 12 % | -0.61 | -0.14 |
| 2 h | +4.96 | -5.29 | 85 % | 72 % | 53 % | 21 % | -1.06 | -0.40 |
| 4 h | +6.49 | -6.97 | 88 % | 80 % | 66 % | 31 % | -2.10 | -0.62 |
| 12 h | +9.06 | -10.74 | 91 % | 90 % | 81 % | 54 % | -4.62 | -0.83 |
| 24 h | +12.43 | -13.98 | 92 % | 92 % | 88 % | 67 % | -7.50 | -1.65 |

### P1-B. P(MFE ≥ +X at all within T) — S0 all ON (N = 1443, 255 days)

| X \ T | 5 min | 15 min | 30 min | 60 min | 2 h | 4 h | 12 h | 24 h |
|---|---|---|---|---|---|---|---|---|
| +0.3 | 75 % | 85 % | 90 % | 93 % | 94 % | 95 % | 97 % | 98 % |
| +0.5 | 66 % | 79 % | 85 % | 90 % | 91 % | 93 % | 95 % | 96 % |
| +1 | 46 % | 65 % | 74 % | 81 % | 85 % | 88 % | 91 % | 92 % |
| +1.5 | 34 % | 54 % | 65 % | 74 % | 79 % | 84 % | 88 % | 90 % |
| +2 | 24 % | 44 % | 57 % | 67 % | 74 % | 79 % | 85 % | 87 % |
| +3 | 13 % | 29 % | 42 % | 54 % | 64 % | 71 % | 79 % | 82 % |
| +5 | 5 % | 15 % | 26 % | 39 % | 50 % | 58 % | 68 % | 73 % |

### P1-B. P(MFE ≥ +X at all within T) — S3 strong (ADX up & +DI>-DI) (N = 893, 245 days)

| X \ T | 5 min | 15 min | 30 min | 60 min | 2 h | 4 h | 12 h | 24 h |
|---|---|---|---|---|---|---|---|---|
| +0.3 | 76 % | 87 % | 90 % | 93 % | 94 % | 96 % | 97 % | 98 % |
| +0.5 | 68 % | 81 % | 87 % | 91 % | 93 % | 94 % | 96 % | 97 % |
| +1 | 50 % | 68 % | 76 % | 83 % | 87 % | 89 % | 92 % | 94 % |
| +1.5 | 38 % | 58 % | 67 % | 75 % | 81 % | 85 % | 89 % | 91 % |
| +2 | 28 % | 48 % | 60 % | 69 % | 77 % | 81 % | 86 % | 89 % |
| +3 | 16 % | 34 % | 46 % | 57 % | 68 % | 74 % | 81 % | 84 % |
| +5 | 6 % | 19 % | 31 % | 43 % | 54 % | 62 % | 71 % | 76 % |

### P1-B. P(MFE ≥ +X at all within T) — S4 neither (N = 82, 69 days)

| X \ T | 5 min | 15 min | 30 min | 60 min | 2 h | 4 h | 12 h | 24 h |
|---|---|---|---|---|---|---|---|---|
| +0.3 | 77 % | 84 % | 89 % | 94 % | 94 % | 96 % | 98 % | 98 % |
| +0.5 | 61 % | 74 % | 85 % | 91 % | 91 % | 95 % | 96 % | 96 % |
| +1 | 40 % | 60 % | 76 % | 85 % | 88 % | 93 % | 95 % | 95 % |
| +1.5 | 24 % | 45 % | 62 % | 76 % | 79 % | 88 % | 91 % | 91 % |
| +2 | 16 % | 38 % | 57 % | 68 % | 73 % | 80 % | 88 % | 89 % |
| +3 | 5 % | 17 % | 34 % | 49 % | 62 % | 71 % | 78 % | 80 % |
| +5 | 0 % | 7 % | 21 % | 32 % | 46 % | 57 % | 63 % | 70 % |

### P1-C. 'Guaranteed?' — hit rate of +X with a day-block 95 % CI (one day's many signals counted as one block)

| subset | X | within 60 min | within 12 h | within 24 h (CI) | 'guaranteed' (CI low ≥ 95 %)? |
|---|---|---|---|---|---|
| S0 | +0.3 | 93 % (91 %–94 %) | 97 % (96 %–98 %) | 98 % (97 %–98 %) | YES |
| S0 | +0.5 | 90 % (88 %–91 %) | 95 % (93 %–96 %) | 96 % (95 %–97 %) | no |
| S0 | +1 | 81 % (79 %–83 %) | 91 % (89 %–92 %) | 92 % (91 %–94 %) | no |
| S0 | +1.5 | 74 % (71 %–76 %) | 88 % (86 %–89 %) | 90 % (88 %–91 %) | no |
| S0 | +2 | 67 % (64 %–69 %) | 85 % (83 %–87 %) | 87 % (85 %–89 %) | no |
| S0 | +3 | 54 % (52 %–57 %) | 79 % (77 %–81 %) | 82 % (80 %–84 %) | no |
| S0 | +5 | 39 % (36 %–41 %) | 68 % (66 %–71 %) | 73 % (70 %–76 %) | no |
| S3 | +0.3 | 93 % (92 %–95 %) | 97 % (96 %–98 %) | 98 % (97 %–99 %) | YES |
| S3 | +0.5 | 91 % (89 %–93 %) | 96 % (94 %–97 %) | 97 % (96 %–98 %) | YES |
| S3 | +1 | 83 % (81 %–85 %) | 92 % (90 %–94 %) | 94 % (92 %–95 %) | no |
| S3 | +1.5 | 75 % (73 %–78 %) | 89 % (87 %–91 %) | 91 % (89 %–93 %) | no |
| S3 | +2 | 69 % (66 %–72 %) | 86 % (84 %–89 %) | 89 % (87 %–91 %) | no |
| S3 | +3 | 57 % (54 %–61 %) | 81 % (78 %–84 %) | 84 % (82 %–87 %) | no |
| S3 | +5 | 43 % (40 %–46 %) | 71 % (68 %–75 %) | 76 % (73 %–79 %) | no |
| S4 | +0.3 | 94 % (89 %–99 %) | 98 % (94 %–100 %) | 98 % (94 %–100 %) | no |
| S4 | +0.5 | 91 % (85 %–97 %) | 96 % (92 %–100 %) | 96 % (92 %–100 %) | no |
| S4 | +1 | 85 % (78 %–92 %) | 95 % (90 %–99 %) | 95 % (90 %–99 %) | no |
| S4 | +1.5 | 76 % (66 %–85 %) | 91 % (84 %–97 %) | 91 % (85 %–97 %) | no |
| S4 | +2 | 68 % (57 %–79 %) | 88 % (80 %–94 %) | 89 % (81 %–95 %) | no |
| S4 | +3 | 49 % (37 %–60 %) | 78 % (69 %–87 %) | 80 % (71 %–89 %) | no |
| S4 | +5 | 32 % (22 %–42 %) | 63 % (51 %–75 %) | 70 % (59 %–80 %) | no |

### P1-D. P(+X before −Y), 24 h horizon ('none' = +X reached at all within 24 h)


S0 all ON:

| X | Y none | −3 | −5 | −8 |
|---|---|---|---|---|
| +0.3 | 98 % | 85 % | 91 % | 94 % |
| +0.5 | 96 % | 80 % | 87 % | 92 % |
| +1 | 92 % | 70 % | 80 % | 86 % |
| +1.5 | 90 % | 61 % | 73 % | 82 % |
| +2 | 87 % | 55 % | 68 % | 78 % |
| +3 | 82 % | 47 % | 60 % | 71 % |
| +5 | 73 % | 36 % | 47 % | 59 % |

S3 strong (ADX up & +DI>-DI):

| X | Y none | −3 | −5 | −8 |
|---|---|---|---|---|
| +0.3 | 98 % | 86 % | 92 % | 95 % |
| +0.5 | 97 % | 82 % | 88 % | 93 % |
| +1 | 94 % | 71 % | 81 % | 87 % |
| +1.5 | 91 % | 62 % | 74 % | 84 % |
| +2 | 89 % | 57 % | 69 % | 79 % |
| +3 | 84 % | 48 % | 60 % | 72 % |
| +5 | 76 % | 38 % | 49 % | 61 % |

S4 neither:

| X | Y none | −3 | −5 | −8 |
|---|---|---|---|---|
| +0.3 | 98 % | 85 % | 91 % | 94 % |
| +0.5 | 96 % | 82 % | 89 % | 93 % |
| +1 | 95 % | 74 % | 83 % | 89 % |
| +1.5 | 91 % | 63 % | 77 % | 84 % |
| +2 | 89 % | 59 % | 73 % | 79 % |
| +3 | 80 % | 49 % | 62 % | 68 % |
| +5 | 70 % | 37 % | 49 % | 52 % |

### P1-E. The tail that kills 'no stop': signals that NEVER reach +X within T — where they stand at T, 12 h, 24 h (S0)

| X | T | share never reaching | at T: mean / median / 5 % worst / 1 % worst / max loss | at 12 h: mean / median / 5 % / worst | at 24 h: mean / median / 5 % / worst | naked 'hold until +X else exit at T' expectancy |
|---|---|---|---|---|---|---|
| +0.5 | 15 min | 21 % | -2.59 / -2.10 / -6.67 / -11.86 / -19.79 | -2.64 / -5.95 / -19.79 / -45.80 | -1.64 / -7.99 / -28.31 / -55.60 | -0.15 % |
| +0.5 | 60 min | 10 % | -5.24 / -4.64 / -10.83 / -15.64 / -30.53 | -5.64 / -7.74 / -18.93 / -45.80 | -6.12 / -9.81 / -30.49 / -55.60 | -0.10 % |
| +0.5 | 4 h | 7 % | -8.70 / -7.61 / -18.15 / -30.32 / -36.50 | -9.75 / -10.00 / -19.59 / -45.80 | -10.35 / -12.09 / -30.63 / -55.60 | -0.16 % |
| +0.5 | 12 h | 5 % | -12.19 / -11.24 / -24.54 / -44.61 / -45.80 | -12.19 / -11.24 / -24.54 / -45.80 | -12.47 / -12.86 / -27.01 / -55.60 | -0.18 % |
| +0.5 | 24 h | 4 % | -15.87 / -14.87 / -31.42 / -49.41 / -55.60 | -13.34 / -12.10 / -30.95 / -45.80 | -15.87 / -14.87 / -31.42 / -55.60 | -0.20 % |
| +1 | 15 min | 35 % | -2.36 / -1.77 / -6.60 / -11.98 / -20.47 | -3.16 / -6.12 / -22.62 / -46.36 | -2.58 / -8.10 / -28.51 / -59.49 | -0.17 % |
| +1 | 60 min | 19 % | -4.81 / -4.01 / -11.27 / -17.19 / -30.53 | -6.45 / -7.68 / -22.69 / -46.36 | -7.05 / -10.11 / -31.19 / -59.49 | -0.09 % |
| +1 | 4 h | 12 % | -8.56 / -7.53 / -18.06 / -33.48 / -50.19 | -10.29 / -10.14 / -22.65 / -46.36 | -11.46 / -12.51 / -32.43 / -59.49 | -0.15 % |
| +1 | 12 h | 9 % | -12.97 / -12.06 / -26.84 / -45.31 / -46.36 | -12.97 / -12.06 / -26.84 / -46.36 | -13.44 / -14.21 / -32.64 / -59.49 | -0.28 % |
| +1 | 24 h | 8 % | -16.76 / -14.98 / -35.45 / -54.96 / -59.49 | -13.95 / -12.60 / -31.16 / -46.36 | -16.76 / -14.98 / -35.45 / -59.49 | -0.35 % |
| +2 | 15 min | 56 % | -1.87 / -1.34 / -6.35 / -9.90 / -20.47 | -2.31 / -5.69 / -24.50 / -91.25 | -1.88 / -7.93 / -30.18 / -94.04 | -0.15 % |
| +2 | 60 min | 33 % | -4.25 / -3.36 / -11.58 / -19.09 / -30.53 | -5.62 / -6.99 / -23.62 / -46.36 | -5.24 / -8.91 / -30.70 / -59.49 | -0.06 % |
| +2 | 4 h | 21 % | -8.27 / -6.98 / -18.25 / -34.92 / -50.19 | -9.05 / -9.36 / -27.56 / -46.36 | -9.55 / -11.32 / -33.14 / -59.49 | -0.14 % |
| +2 | 12 h | 15 % | -12.74 / -11.50 / -29.11 / -45.21 / -46.36 | -12.74 / -11.50 / -29.11 / -46.36 | -13.10 / -13.63 / -33.60 / -59.49 | -0.23 % |
| +2 | 24 h | 13 % | -16.02 / -14.51 / -35.08 / -49.61 / -59.49 | -13.68 / -12.26 / -30.86 / -46.36 | -16.02 / -14.51 / -35.08 / -59.49 | -0.32 % |
| +3 | 15 min | 71 % | -1.49 / -1.04 / -6.19 / -9.89 / -20.47 | -1.72 / -5.27 / -24.61 / -91.25 | -2.01 / -7.58 / -31.72 / -94.04 | -0.17 % |
| +3 | 60 min | 46 % | -3.86 / -2.91 / -11.62 / -20.63 / -30.53 | -4.61 / -6.73 / -23.62 / -70.29 | -4.85 / -8.60 / -31.72 / -83.77 | -0.14 % |
| +3 | 4 h | 29 % | -7.94 / -6.74 / -18.84 / -34.55 / -50.19 | -8.28 / -8.96 / -27.57 / -70.29 | -9.03 / -11.00 / -33.74 / -83.77 | -0.15 % |
| +3 | 12 h | 21 % | -12.97 / -11.74 / -28.78 / -45.40 / -70.29 | -12.97 / -11.74 / -28.78 / -70.29 | -13.51 / -13.46 / -35.45 / -83.77 | -0.34 % |
| +3 | 24 h | 18 % | -16.27 / -14.44 / -36.69 / -57.15 / -83.77 | -13.77 / -12.34 / -30.73 / -70.29 | -16.27 / -14.44 / -36.69 / -83.77 | -0.48 % |

### P1-F. By live refusal code and by hours since the spike (S0)

| group | N | days | +1 within 60 min | +1 within 12 h | +2 before −5 | MAE 12 h ≤ −10 | median net 24 h | live lock (LOCK2) mean |
|---|---|---|---|---|---|---|---|---|
| code READY | 361 | 179 | 75 % | 86 % | 67 % | 40 % | -6.42 | +0.09 |
| code GREEN_BAR | 393 | 181 | 77 % | 90 % | 68 % | 47 % | -6.05 | -0.13 |
| code ATR_HIGH | 689 | 224 | 87 % | 94 % | 69 % | 65 % | -9.12 | -0.37 |
| gvol-blocked (U2 ≥ 1) | 614 | 229 | 81 % | 90 % | 67 % | 53 % | -8.07 | -0.31 |
| gvol open (U2 < 1) | 829 | 246 | 81 % | 92 % | 69 % | 55 % | -7.17 | -0.11 |
| action: refused: ATR_HIGH | 363 | 174 | 87 % | 94 % | 67 % | 67 % | -9.12 | -0.52 |
| action: refused: disloc >1% | 54 | 51 | 89 % | 94 % | 78 % | 65 % | -6.82 | +0.20 |
| action: refused: gvol | 614 | 229 | 81 % | 90 % | 67 % | 53 % | -8.07 | -0.31 |
| action: refused: hold-green (streak<=12) | 82 | 63 | 70 % | 89 % | 62 % | 44 % | -7.40 | -0.59 |
| action: taken FRENZY_LONG | 205 | 133 | 76 % | 88 % | 70 % | 39 % | -5.94 | +0.39 |
| action: taken WIDE | 125 | 93 | 80 % | 91 % | 73 % | 47 % | -4.28 | +0.46 |
| hours <3h | 236 | 158 | 83 % | 91 % | 67 % | 58 % | -9.58 | -0.15 |
| hours 3-6h | 208 | 138 | 74 % | 88 % | 64 % | 55 % | -8.77 | -0.47 |
| hours 6-12h | 277 | 163 | 81 % | 90 % | 68 % | 60 % | -8.05 | -0.32 |
| hours 12-24h | 400 | 188 | 83 % | 92 % | 69 % | 52 % | -7.58 | -0.21 |
| hours 24-48h | 216 | 126 | 83 % | 91 % | 75 % | 50 % | -4.19 | +0.26 |
| hours >=48h | 106 | 57 | 80 % | 92 % | 62 % | 43 % | +0.86 | -0.22 |
| subset S3 strong (ADX up & +DI>-DI) | 893 | 245 | 83 % | 92 % | 69 % | 53 % | -6.80 | -0.06 |
| subset S0 all ON | 1443 | 255 | 81 % | 91 % | 68 % | 54 % | -7.50 | -0.19 |
| subset S4 neither | 82 | 69 | 85 % | 95 % | 73 % | 46 % | -5.86 | -0.14 |
| subset S1 ADX up | 948 | 245 | 83 % | 92 % | 69 % | 53 % | -6.84 | -0.06 |
| subset S2 +DI>-DI | 1306 | 255 | 81 % | 90 % | 67 % | 55 % | -7.55 | -0.20 |

**What the tables say:**
- Hit rates are high and fast. Day-block CIs are tight because there are 255 days. For S0 and S3 alike, "guaranteed" (CI low ≥ 95 %) holds only for +0.3 within 24 h (S0 and S3) and +0.5 within 24 h (S3).
- But P(+X before −Y) drops fast once a stop exists. +1 before −3 happens only 70 % of the time, and +2 before −5 only 68 %. A stop converts the misers into realised losses, and those losses outweigh the wins at every X tried.
- Without a stop, the losses are not cut, they are deferred. The "naked hold until +X, else exit at T" expectancy is negative for every X × T (−0.06 to −0.48 %/trade in S0).
- Refusal codes: the codes FRENZY refuses (ATR_HIGH, hold-green, gvol) pop as often as the taken ones, or more often. ATR_HIGH hits +1 within 60 min 87 % of the time vs 76 % for READY. But they also sink further: 65 % of ATR_HIGH signals reach −10 within 12 h, vs 39 % of taken FRENZY_LONG. The filters are doing their job on the tail, not on the pop.
- Hours since spike show no monotone pattern for the pop. Older episodes (≥ 24 h) sink less at 24 h.

## PART 2 — exit designs (frozen grid)

### P2-B. Out-of-sample: choose the best cell on one half, read it on the other

| choose on | pool | chosen cell | mean on choice half | N | mean on the OTHER half | N | day-CI other half |
|---|---|---|---|---|---|---|---|
| Jan–Apr | family | S4 TR1z0.5/T120/noSL | +0.25 | 34 | **+0.57** | 48 | +0.07 … +0.97 |
| May–Sep | family | S4 TP3/T120/noSL | +0.82 | 48 | **+0.23** | 34 | -1.05 … +1.43 |
| Jan–Apr | S3 | S3 TP1.5/T120/noSL | +0.13 | 419 | **+0.03** | 474 | -0.35 … +0.39 |
| May–Sep | S3 | S3 TP3/T120/noSL | +0.22 | 474 | **+0.06** | 419 | -0.42 … +0.51 |

**Reading the OOS table:**
- When the whole family is the pool, the best-in-half cell is always an S4 cell. S4 is the control flank: ADX falling ∧ −DI > +DI, with only 34–48 signals per half.
- One direction holds out of sample: +0.57, CI +0.07…+0.97 on 48 trades. The other direction's CI spans 0.
- This is the only hint in the study, and it points against the operator's thesis. Over the full family it is not significant (max-t p 0.63–0.91). It also sits in the non-monotone 3×3, which marks a confound.
- So it is a watch item at most. It is not a candidate.
- S3, the operator's condition, reads about 0 out of sample.


### P2-C. Per month (mean net % per trade / N) — key cells

| cell | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|
| S3 TP3/T120/noSL | +0.60 /97 | +0.37 /68 | -0.10 /105 | -0.32 /149 | +0.03 /95 | +1.33 /62 | +0.10 /99 | +0.11 /113 | -0.01 /105 |
| S3 TP1/T60/noSL | +0.10 /97 | -0.26 /68 | +0.06 /105 | +0.09 /149 | +0.10 /95 | +0.70 /62 | -0.37 /99 | +0.01 /113 | -0.26 /105 |
| S3 LOCK2(live) | +0.47 /97 | -0.04 /68 | +0.02 /105 | -0.45 /149 | -0.12 /95 | -0.04 /62 | -0.05 /99 | +0.14 /113 | -0.27 /105 |
| S0 TP1.5/T60/noSL | +0.09 /149 | -0.03 /109 | +0.11 /191 | -0.26 /223 | +0.07 /159 | +0.11 /108 | -0.30 /166 | -0.05 /180 | -0.06 /158 |
| S0 TP1/T60/noSL | +0.04 /149 | -0.16 /109 | +0.11 /191 | -0.29 /223 | +0.00 /159 | +0.14 /108 | -0.46 /166 | -0.06 /180 | +0.07 /158 |
| S0 LOCK2(live) | +0.24 /149 | +0.14 /109 | -0.15 /191 | -0.47 /223 | -0.48 /159 | -0.42 /108 | -0.11 /166 | -0.18 /180 | -0.15 /158 |
| S4 TP3/T120/noSL | +1.45 /6 | +0.92 /9 | -0.95 /12 | +0.32 /7 | -0.18 /8 | +1.25 /9 | +2.10 /9 | +1.07 /10 | -0.00 /12 |


### P2-D. Book from $3,000 (slots 2 FRENZY lane + 2 WIDE lane, pair flat, 3 per pair-day; global cap = max_open_positions 4 (stress) or 10 (hard cap, redeploy on)); liquidation modelled; ruin = 5,000 day-block bootstrap paths

| subset · exit | lev | global cap | trades taken | liquidations | of which live FRENZY / WIDE fills | end $ | max DD | P(DD ≥ 50 %) | 5 % worst DD | median end $ (boot) | P(end < start) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S3 TP3/T120/noSL | 0.2 | 10 | 886 | 18 | 113 / 94 | $1,067 | -88 % | 99 % | -99 % | $1,050 | 75 % |
| S3 TP3/T120/noSL | 0.32 | 4 | 886 | 43 | 113 / 94 | $791 | -96 % | 100 % | -99 % | $781 | 77 % |
| S3 TP3/T120/noSL | 0.32 | 10 | 886 | 43 | 113 / 94 | $791 | -96 % | 100 % | -99 % | $799 | 77 % |
| S3 TP1/T60/noSL | 0.2 | 10 | 891 | 3 | 114 / 95 | $1,859 | -71 % | 80 % | -89 % | $1,887 | 73 % |
| S3 TP1/T60/noSL | 0.32 | 4 | 891 | 10 | 114 / 95 | $1,284 | -80 % | 93 % | -94 % | $1,326 | 79 % |
| S3 TP1/T60/noSL | 0.32 | 10 | 891 | 10 | 114 / 95 | $1,284 | -80 % | 94 % | -95 % | $1,298 | 80 % |
| S3 LOCK2(live) | 0.2 | 10 | 888 | 0 | 114 / 94 | $1,094 | -83 % | 95 % | -94 % | $1,100 | 86 % |
| S3 LOCK2(live) | 0.32 | 4 | 888 | 0 | 114 / 94 | $704 | -91 % | 99 % | -98 % | $722 | 89 % |
| S3 LOCK2(live) | 0.32 | 10 | 888 | 0 | 114 / 94 | $704 | -91 % | 99 % | -98 % | $715 | 89 % |
| S0 TP1.5/T60/noSL | 0.2 | 10 | 1422 | 10 | 204 / 124 | $246 | -95 % | 100 % | -99 % | $251 | 98 % |
| S0 TP1.5/T60/noSL | 0.32 | 4 | 1422 | 24 | 204 / 124 | $101 | -99 % | 100 % | -100 % | $106 | 98 % |
| S0 TP1.5/T60/noSL | 0.32 | 10 | 1422 | 24 | 204 / 124 | $101 | -99 % | 100 % | -100 % | $106 | 98 % |
| S0 TP1/T60/noSL | 0.2 | 10 | 1423 | 7 | 204 / 124 | $293 | -94 % | 100 % | -99 % | $293 | 98 % |
| S0 TP1/T60/noSL | 0.32 | 4 | 1423 | 18 | 204 / 124 | $139 | -98 % | 100 % | -100 % | $139 | 99 % |
| S0 TP1/T60/noSL | 0.32 | 10 | 1423 | 18 | 204 / 124 | $139 | -98 % | 100 % | -100 % | $145 | 99 % |
| S0 LOCK2(live) | 0.2 | 10 | 1417 | 0 | 203 / 121 | $101 | -99 % | 100 % | -100 % | $101 | 100 % |
| S0 LOCK2(live) | 0.32 | 4 | 1417 | 0 | 203 / 121 | $30 | -100 % | 100 % | -100 % | $30 | 100 % |
| S0 LOCK2(live) | 0.32 | 10 | 1417 | 0 | 203 / 121 | $30 | -100 % | 100 % | -100 % | $30 | 100 % |
| S4 TP3/T120/noSL | 0.2 | 10 | 82 | 0 | 13 / 1 | $4,424 | -20 % | 1 % | -40 % | $4,460 | 12 % |
| S4 TP3/T120/noSL | 0.32 | 4 | 82 | 1 | 13 / 1 | $4,729 | -26 % | 6 % | -51 % | $4,769 | 15 % |
| S4 TP3/T120/noSL | 0.32 | 10 | 82 | 1 | 13 / 1 | $4,729 | -26 % | 6 % | -51 % | $4,786 | 15 % |
| S0 TP0.5/T15/noSL | 0.2 | 10 | 1425 | 2 | 204 / 124 | $325 | -92 % | 100 % | -97 % | $325 | 100 % |
| S0 TP0.5/T15/noSL | 0.32 | 4 | 1425 | 3 | 204 / 124 | $180 | -96 % | 100 % | -98 % | $184 | 100 % |
| S0 TP0.5/T15/noSL | 0.32 | 10 | 1425 | 3 | 204 / 124 | $180 | -96 % | 100 % | -99 % | $182 | 100 % |
| S3 TR1z0.5/T60/noSL | 0.2 | 10 | 891 | 3 | 114 / 95 | $1,293 | -77 % | 89 % | -92 % | $1,299 | 85 % |
| S3 TR1z0.5/T60/noSL | 0.32 | 4 | 891 | 10 | 114 / 95 | $802 | -86 % | 97 % | -97 % | $805 | 90 % |
| S3 TR1z0.5/T60/noSL | 0.32 | 10 | 891 | 10 | 114 / 95 | $802 | -86 % | 97 % | -96 % | $839 | 90 % |
| S0 LITERAL 'TP +0.5, no stop, hold ≤ 24 h' | 0.2 | 10 | 1400 | 22 | 197 / 119 | $21 | -99 % | 100 % | -100 % | $22 | 100 % |
| S0 LITERAL 'TP +0.5, no stop, hold ≤ 24 h' | 0.32 | 10 | 1412 | 62 | 198 / 121 | $3 | -100 % | 100 % | -100 % | $3 | 100 % |
| S3 LITERAL 'TP +0.5, no stop, hold ≤ 24 h' | 0.2 | 10 | 886 | 10 | 113 / 93 | $790 | -85 % | 95 % | -97 % | $841 | 89 % |
| S3 LITERAL 'TP +0.5, no stop, hold ≤ 24 h' | 0.32 | 10 | 888 | 32 | 113 / 93 | $244 | -93 % | 100 % | -99 % | $253 | 98 % |
| S0 LITERAL 'TP +1, no stop, hold ≤ 24 h' | 0.2 | 10 | 1360 | 49 | 195 / 115 | $0 | -100 % | 100 % | -100 % | $0 | 100 % |
| S0 LITERAL 'TP +1, no stop, hold ≤ 24 h' | 0.32 | 10 | 1388 | 105 | 197 / 120 | $0 | -100 % | 100 % | -100 % | $0 | 100 % |
| S3 LITERAL 'TP +1, no stop, hold ≤ 24 h' | 0.2 | 10 | 875 | 27 | 113 / 93 | $78 | -98 % | 100 % | -100 % | $81 | 98 % |
| S3 LITERAL 'TP +1, no stop, hold ≤ 24 h' | 0.32 | 10 | 879 | 60 | 113 / 93 | $43 | -99 % | 100 % | -100 % | $46 | 99 % |
| S0 LITERAL 'TP +2, no stop, hold ≤ 24 h' | 0.2 | 10 | 1294 | 82 | 183 / 107 | $0 | -100 % | 100 % | -100 % | $0 | 100 % |
| S0 LITERAL 'TP +2, no stop, hold ≤ 24 h' | 0.32 | 10 | 1342 | 153 | 185 / 113 | $0 | -100 % | 100 % | -100 % | $0 | 100 % |
| S3 LITERAL 'TP +2, no stop, hold ≤ 24 h' | 0.2 | 10 | 854 | 51 | 110 / 87 | $12 | -100 % | 100 % | -100 % | $12 | 99 % |
| S3 LITERAL 'TP +2, no stop, hold ≤ 24 h' | 0.32 | 10 | 865 | 95 | 110 / 90 | $12 | -100 % | 100 % | -100 % | $13 | 99 % |


### P2-E. Worst real trades / days (S0, TP1/T60/noSL and the literal TP +1 no-stop rule; 1× price %)

| signal (UTC) | pair | code | TP1/T60/noSL | literal TP+1 no stop (24 h) | MAE 12 h | net at 24 h |
|---|---|---|---|---|---|---|
| 2026-08-14 22:30:00 | ACEUSDT | ATR_HIGH | -11.82 | -59.49 | -50.97 | -59.49 |
| 2026-04-28 15:45:00 | ZKJUSDT | ATR_HIGH | -10.43 | -55.60 | -48.41 | -55.60 |
| 2026-06-12 15:00:00 | STGUSDT | GREEN_BAR | -5.08 | -48.55 | -25.29 | -48.55 |
| 2026-06-28 16:05:00 | MANTAUSDT | ATR_HIGH | -30.54 | -45.46 | -45.74 | -45.46 |
| 2026-04-17 17:05:00 | MOVRUSDT | ATR_HIGH | -10.48 | -37.59 | -37.62 | -37.59 |
| 2026-07-01 09:20:00 | TACUSDT | READY | -1.35 | -36.84 | -35.06 | -36.84 |
| 2026-01-12 06:35:00 | 1000WHYUSDT | READY | -29.91 | -33.76 | -33.76 | -33.76 |
| 2026-01-25 21:30:00 | NOMUSDT | ATR_HIGH | -0.70 | -31.72 | -33.99 | -31.72 |

Worst days for TP1/T60/noSL (sum of 1× % over that day's signals): 2026-06-28 -28.5 %, 2026-01-12 -28.4 %, 2026-04-17 -26.0 %, 2026-07-10 -22.6 %, 2026-08-14 -22.2 %


### P2-F. Screen only: ADX-delta tercile × DI-spread tercile (mean net %/trade of TP1/T60/noSL · LOCK2 · P(+1 within 60 min))

Tercile edges: ADXΔ -0.09 / 3.43; DI spread 11.90 / 23.44

| | DI low | DI mid | DI high |
|---|---|---|---|
| ADXΔ low | -0.19 · -0.54 · 77 % (N 322, 173 d) | -0.70 · -0.42 · 74 % (N 139, 106 d) | +0.52 · +0.93 · 95 % (N 20, 20 d) |
| ADXΔ mid | -0.06 · -0.36 · 78 % (N 133, 102 d) | -0.01 · -0.39 · 81 % (N 237, 148 d) | -0.51 · +0.21 · 82 % (N 111, 91 d) |
| ADXΔ high | +0.38 · +0.82 · 92 % (N 26, 24 d) | +0.03 · +0.27 · 79 % (N 105, 88 d) | +0.25 · +0.01 · 88 % (N 350, 188 d) |


### P2-G. Is S3 the same thing as the live FRENZY_LONG 'strong' flag?

| population | N | days | LOCK2 (live exit) mean | TP1/T60/noSL mean | P(+1 within 60 min) | median net 24 h |
|---|---|---|---|---|---|---|
| live FRENZY_LONG taken, strong (S3R, lev 0.5) | 115 | 94 | +0.67 | -0.53 | 76 % | -5.85 |
| live FRENZY_LONG taken, normal (lev 0.32) | 90 | 71 | +0.02 | +0.08 | 76 % | -6.03 |
| all READY bars, S3 | 207 | 138 | +0.40 | -0.35 | 76 % | -5.85 |
| all READY bars, not S3 | 154 | 106 | -0.33 | -0.15 | 72 % | -7.01 |
| WIDE-code bars (GREEN/ATR_HIGH), S3 | 686 | 230 | -0.20 | +0.11 | 85 % | -7.53 |
| WIDE-code bars, not S3 | 396 | 196 | -0.43 | -0.24 | 81 % | -8.98 |

## GRIFFAINUSDT, 2026-10-06 (out-of-sample anecdote, not in any statistic)

- **Engine bar.** The real `frenzy_walk` on public 5m/1h klines puts the fresh ON bar at 17:35–17:40 UTC (close 17:40, which is 14:40 Argentina). Code FRENZY_ATR_HIGH (ATR 3.63 %), so WIDE is refused. Above-streak 12, vol × 155.
- **ADX/DI.** adx_delta +8.93 and di_spread +45.3, so it is **S3**. It is not READY, so it is not the live strong flag.
- **Entry.** Public aggTrades: the first print ≥ 17:40:12 is 0.025247. That is **−2.04 % under the ON close** (0.025774), so live would also refuse it on the 1 % dislocation guard.
- **Path.** Net −4.0 % at 6 s, then +2 % at 25 s, then +4.46 % at about 7 min. At 18:20 UTC (40 min) it sits at −1.16 %.
- **Exits on this one signal:**

| exit | result |
|---|---|
| TP +0.5 / +1 / +2 (no stop) | won at 25 s |
| TP +3 / 2 h | +3 at 6 min |
| TP +1 with SL −3 | **−3.0 at 6 s** |
| live LOCK2 | −3.1 at 6 s |

- **Reading.** This is the operator's picture exactly: the pop came, but only for whoever survived the first −4 % without a stop. It is one draw from the distribution in P1-E. In the year, 19 % of signals never see +1 within the hour, and they average −4.8 % at the hour.

## PART 3 — verdict

- **Is it a new strategy? No.**
  - "X % happens 80–90 % of the time" is true.
  - But the 10–20 % that don't make it lose 5–50× the target (−4.8 % at 60 min, −16.8 % at 24 h, for X = 1).
  - Every frozen exit lands at or below zero once costs are paid. A stop cuts the tail but also kills most of the winners that dip first. No stop keeps the winners but lets the tail run to liquidation.
- **Selection.** Over the 1,205-cell family, the best max-t adjusted p is 0.63 (an S4 cell, N = 82), and 0.97 on S3.
- **OOS.** Choosing on one half and reading on the other gives roughly 0 for S3 (+0.03 / +0.06, both CIs spanning 0).
- **The ADX/DI "strong" condition** (S3, the primary subset) adds about +0.1 %/trade on no-stop exits. That is inside noise. It is not monotone (3×3), and the opposite flank looks better on small N. The strong flag's genuine value is for the live lock's runners, not for this scalp.
- **"No stop" ruin risk:**
  - At FRENZY-like sizing (lev 0.32, notional 1.21× equity), any S0/S3 no-stop design has P(book DD ≥ 50 %) = 93–100 % over a year of these signals.
  - At WIDE-like sizing (0.2) it is 80–100 %.
  - The literal "wait until it comes back" rule ends at $0–$78 from $3k, with 27–105 liquidations.
  - The worst real days: 2026-06-28 (MANTA −30 % in 60 min), 01-12 (1000WHY −30 %), 04-17, 07-10, 08-14. Each day sums −22…−29 % at 1× on TP1/T60/noSL.
- **Recommendation.** Do not build it, and do not probe it.
  - If the operator wants it watched, the only pre-registered line that is consistent with the PREREG is a free observe-only journal line, frozen as: *"ON-scalp S3: fresh ON bar ∧ adx_delta > 0 ∧ di_spread > 0, entry first print ≥ close + 12 s, TP +3 net / 2 h time stop / no stop (liquidation at the sleeve's lev)."*
  - Bar: N ≥ 30 fresh fires on ≥ 15 days, mean > 0 with day-CI low > 0, P(+3 within 2 h) ≥ 70 %, and a forward book DD < 50 % at 0.2. Then apply the 30–50 % haircut.
  - My expectation is that it reads ≈ 0, like the year. (`scripts/scout_*.py` was not touched; wiring this is the coordinator's call.)
- **What could change the answer (new data, not new screens):**
  - The operator's own clicks on ON bars, stamped. His exit cuts at −0.5…−1.2 %, which no mechanical stop here reproduces.
  - Order-book / liquidation data in the first 60 s after the ON bar, when the GRIFFAIN-style −4 % flush happens.

## Pre-registration (verbatim)

```
FRENZY ON-SCALP — PRE-REGISTRATION (frozen 2026-10-06 18:10 UTC, before ANY outcome of this study was computed)
Copied verbatim into reports/FRENZY_ON_SCALP_STUDY_2026-10-06.md.

HYPOTHESIS (operator): at the fresh FRENZY ON bar, buy immediately regardless of FRENZY's other filters; price "goes X % above in a short
period, guaranteed"; take X with a fixed target or a trail, maybe with no stop.

SIGNALS
- Universe U = reports/FRENZY_ENGINE_COHORT_2026-10-05.csv (engine frenzy_walk fresh-ON bars, blacklists removed, Jan 10 -> Sep 27 2026)
  inner-joined with reports/FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv on (pair, t_signal_close), live_elig == True (tradeable universe).
  Every refusal code kept: FRENZY_READY, FRENZY_GREEN_BAR, FRENZY_ATR_HIGH; gvol-blocked (gvol_live_U2 >= 1 or NaN); dislocation (> 1 %).
  Not in U (blind spot): FRENZY_VOL24_LOW fresh bars (vol24 < $20M, never in the live sleeve universe), blacklisted pairs.
- Live-action label: FRENZY_TAKEN = READY & gvol<1 & |disloc|<=1 % ; WIDE_TAKEN = GREEN_BAR & above_streak>12 & gvol<1 & |disloc|<=1 % ;
  everything else = refused live (reason precedence: gvol, disloc, ATR_HIGH, GREEN_BAR hold-green (streak<=12)).
- Hours-since-spike buckets: <3, 3-6, 6-12, 12-24, 24-48, >=48 h.

SUBSETS (coordinator/operator addition, declared before outcomes)
- adx_delta = services.frenzy.frenzy_adx_delta(closed[-300:]), di_spread = services.frenzy.frenzy_di_spread(closed[-300:]), closed = the
  5m bars [open_ms,o,h,l,c,v] of reports/backtest_cache/k5m_full up to and INCLUDING the ON bar (the engine's closed list at that scan).
- S0 all ON · S1 adx_delta>0 · S2 di_spread>0 · S3 = S1 & S2 (the live strong comparison, services/trading_engine.py 7737; PRIMARY) ·
  S4 = not S1 & not S2 (control flank). S3R = S3 & FRENZY_READY = exactly the live 'frenzy_strong' population (reported, not in the family).
- Screen only: adx_delta tercile x di_spread tercile 3x3 (terciles over U). Non-monotone pattern = confound, not a filter.
- Parity: recompute both values for the live FRENZY/WIDE fills that carry stamps (B17, Oct 3-6) from public 5m klines; S3 vs live lev 10x/6x.

ENTRY / COSTS / DATA
- Entry e = first print >= t_signal_close + 12 s (NO dislocation guard; dislocation reported). Path = every later print to entry + 24 h.
- Ticks (ticks_q, else ticks) when every UTC day of [close-2 min, close+12 s+24 h] is on disk; else public 1m klines (fapi REST), prints
  L->H->C per bar (pessimistic), entry = the signal-bar close, levels filled at min(level, previous print) on gaps. src column; flagged.
- net % = (price/e - 1) x 100 - 0.19 (fees 0.09 round trip + slippage 0.10). All levels X / Y / Z below are on this NET scale.
- Engine parity: the house LOCK2 (frenzy_exit_selector_reprice.all_exits at the same entry) must equal the cohort's LOCK2 column on the
  non-dislocated tick rows.

PART 1 (descriptive, S0..S4, by refusal code, by hours bucket)
- MFE / MAE (net) within 5/15/30/60/120/240 min, 12 h. P(MFE>=X within T) for X in {0.3,0.5,1,1.5,2,3,5}, T in {5,15,30,60,120,240 min,12 h,24 h}.
- P(MFE>=X before MAE<=-Y) for Y in {none,3,5,8} (horizon 24 h; 'none' = reached within 24 h).
- Non-hitters of X within T: net at T, at 12 h, at 24 h: mean, median, 5 % / 1 % worst, max loss.
- Hit rates with day-block 95 % CI (UTC signal day = the block, 4000 resamples). 'Guaranteed' = day-CI low >= 95 %.

PART 2 EXIT GRID (frozen)
- (a/b) TP: X in {0.5,1,1.5,2,3} x T in {15,30,60,120 min} x Y in {none,3,5}: exit at the first of net>=X (fill = that print; 1m: X),
  net<=-Y (fill = that print; 1m: min(-Y, previous print)), first print >= entry+T (fill = that print). 60 designs.
- (c) TRAIL: X x Z in {0.3,0.5,1} x T x Y: before arming SL -Y (or none) and time stop T if not armed by T; armed once the prior-print peak
  >= X; after arming exit at the first print with net <= prior peak - Z (no floor); 12 h cap from entry. 180 designs.
- (d) LOCK2 = live FRENZY lock (house all_exits, -3 until +3, then max(+2, peak-2), 12 h cap), reference.
- Family for the selection adjustment = 5 subsets (S0..S4) x 241 exits = 1205 cells. Statistic = mean net % per trade.
- Per cell: N, days, WR, mean, median, day-block 95 % CI, worst, CVaR 5 %, per month, LOMO (min/max mean leaving one month out), halves
  Jan-Apr / May-Sep, drop top-5 / top-10, top-pair share of gain.
- Selection-adjusted p: (i) max-t over the family, joint day-block bootstrap of centred cell means (same statistic both sides, 2000);
  (ii) timing null: each signal's entry moved to a uniform random moment in [close+30 min, close+12 h] on the same pair, all cells priced on
  a 1-second L/H/C compression of the same ticks for BOTH observed and null (tick rows only), 100 draws; adjusted p = share of draws whose
  best family mean >= observed best family mean (1-s pricer).
- OOS: pick the best cell (N>=20) on Jan-Apr, read it on May-Sep, and vice versa; also within S3 alone.
- Book: $3k, fixed fraction of equity at (lev 0.2: notional 0.94 x equity, liquidation cap -23.75 % price) and (lev 0.32: 1.21 x equity, cap
  -15.83 % price) - the house core.book factors; slots: READY signals -> FRENZY lane 2, other codes -> WIDE lane 2 (2+2), pair flat,
  pair-day cap 3, global cap max_open_positions 4 (stress) and 10 (= max_open_positions_hard, redeploy on). End $, max DD, months.
  Overlap with today's FRENZY_LONG / WIDE taken fills counted. Ruin: day-block bootstrap of the book's daily return sequence (5000 paths of
  the same length): P(max DD >= 50 %), worst-5 % DD, and the worst real sequences.

VERDICT GATE (a cell is a candidate only if ALL): N>=30, >=15 days, mean>0 with day-CI low>0, both halves>0, drop-top-10>0, LOMO min>0,
top pair <50 % of gain, max-t adjusted p<=0.05 AND timing-null adjusted p<=0.10, OOS read>0 both directions, P(book DD>=50 %) <= 5 % at
lev 0.32. Pass -> OBSERVE-ONLY scout line (frozen rule) then probe size; mean>0 both halves but fails CI/p -> pre-registered observe
candidate; else refuted. Any projected delta carries the 30-50 % haircut.
ANECDOTE (out of sample, not in any statistic): GRIFFAINUSDT ON 2026-10-06 17:40 UTC, public klines, real frenzy_walk.
```

## Appendix — S1 / S2 rows of the headline table

| subset | exit | N | days | WR | mean % | median | day-CI 95 % | Jan–Apr | May–Sep | LOMO min | drop top-10 | worst | CVaR 5 % | max-t adj p | timing-null adj p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | LOCK2(live) | 948 | 245 | 50 % | **-0.06** | +1.71 | -0.26 … +0.14 | -0.02 | -0.09 | -0.12 | -0.17 | -3.6 | -3.2 | 1.00 | — |
| S1 | TP0.5/T15/noSL | 948 | 245 | 82 % | **-0.10** | +0.51 | -0.20 … +0.00 | -0.02 | -0.16 | -0.13 | -0.11 | -19.8 | -5.7 | 1.00 | 1.00 |
| S1 | TP1/T60/noSL | 948 | 245 | 84 % | **+0.03** | +1.01 | -0.15 … +0.18 | +0.05 | +0.00 | -0.03 | +0.01 | -29.9 | -9.1 | 1.00 | 0.97 |
| S1 | TP1/T60/SL3 | 948 | 245 | 70 % | **-0.15** | +1.00 | -0.26 … -0.04 | -0.12 | -0.19 | -0.17 | -0.17 | -3.4 | -3.1 | 1.00 | 1.00 |
| S1 | TP2/T120/SL5 | 948 | 245 | 68 % | **-0.12** | +2.00 | -0.32 … +0.05 | +0.01 | -0.24 | -0.21 | -0.15 | -5.5 | -5.1 | 1.00 | 1.00 |
| S1 | TR1z0.5/T60/noSL | 948 | 245 | 83 % | **-0.02** | +0.75 | -0.20 … +0.14 | +0.03 | -0.07 | -0.08 | -0.06 | -29.9 | -9.1 | 1.00 | 0.89 |
| S1 | TR2z1/T120/SL5 | 948 | 245 | 68 % | **-0.17** | +1.27 | -0.38 … +0.04 | -0.03 | -0.30 | -0.26 | -0.23 | -5.5 | -5.1 | 1.00 | 1.00 |
| S1 | best: TP3/T120/noSL | 948 | 245 | 70 % | **+0.14** | +3.00 | -0.19 … +0.47 | +0.06 | +0.21 | +0.03 | +0.10 | -38.7 | -15.8 | 0.98 | 0.85 |
| S1 | best: TP1/T30/SL5 | 948 | 245 | 76 % | **-0.08** | +1.00 | -0.22 … +0.04 | -0.00 | -0.15 | -0.12 | -0.10 | -5.5 | -5.0 | 1.00 | 1.00 |
| S1 | best: TR3z1/T120/noSL | 948 | 245 | 70 % | **+0.07** | +2.30 | -0.27 … +0.40 | +0.02 | +0.11 | -0.04 | +0.01 | -38.7 | -15.8 | 1.00 | 0.72 |
| S2 | LOCK2(live) | 1306 | 255 | 48 % | **-0.20** | -3.10 | -0.37 … -0.03 | -0.13 | -0.26 | -0.25 | -0.29 | -3.6 | -3.2 | 1.00 | — |
| S2 | TP0.5/T15/noSL | 1306 | 255 | 79 % | **-0.15** | +0.51 | -0.24 … -0.07 | -0.09 | -0.21 | -0.19 | -0.16 | -19.8 | -6.0 | 1.00 | 1.00 |
| S2 | TP1/T60/noSL | 1306 | 255 | 81 % | **-0.13** | +1.01 | -0.29 … +0.02 | -0.12 | -0.14 | -0.15 | -0.14 | -30.5 | -9.9 | 1.00 | 1.00 |
| S2 | TP1/T60/SL3 | 1306 | 255 | 69 % | **-0.21** | +1.00 | -0.31 … -0.11 | -0.19 | -0.23 | -0.23 | -0.22 | -3.4 | -3.1 | 1.00 | 1.00 |
| S2 | TP2/T120/SL5 | 1306 | 255 | 66 % | **-0.23** | +2.00 | -0.41 … -0.07 | -0.17 | -0.29 | -0.28 | -0.25 | -5.5 | -5.1 | 1.00 | 1.00 |
| S2 | TR1z0.5/T60/noSL | 1306 | 255 | 80 % | **-0.17** | +0.71 | -0.34 … -0.01 | -0.16 | -0.19 | -0.20 | -0.21 | -30.5 | -9.9 | 1.00 | 1.00 |
| S2 | TR2z1/T120/SL5 | 1306 | 255 | 66 % | **-0.30** | +1.23 | -0.48 … -0.12 | -0.25 | -0.35 | -0.35 | -0.35 | -5.5 | -5.1 | 1.00 | 1.00 |
| S2 | best: TP1.5/T60/noSL | 1306 | 255 | 74 % | **-0.09** | +1.50 | -0.26 … +0.08 | -0.09 | -0.08 | -0.11 | -0.10 | -30.5 | -10.7 | 1.00 | 1.00 |
| S2 | best: TP1.5/T30/SL5 | 1306 | 255 | 65 % | **-0.16** | +1.50 | -0.29 … -0.02 | -0.11 | -0.20 | -0.20 | -0.17 | -5.5 | -5.0 | 1.00 | 1.00 |
| S2 | best: TR0.5z0.5/T60/noSL | 1306 | 255 | 85 % | **-0.14** | +0.28 | -0.26 … -0.03 | -0.13 | -0.15 | -0.19 | -0.17 | -30.5 | -8.0 | 1.00 | 1.00 |

## Blind spots (what this study could NOT test)

1. **FRENZY_VOL24_LOW fresh bars** (98 bars, 24 h volume < $20M) and blacklisted pairs are not in the universe. Live never shortlists them. Signals after Sep 27 are also missing (the cohort build ends there).
2. **1m fallback.** 33 signals have no ticks and use public 1m klines (L→H→C order, pessimistic). Another 21 have 12 h of ticks plus 1m for 12–24 h. Parts of the Part-1 24 h numbers on those rows are 1m-based.
3. **Liquidation is a proxy.** It is triggered at the first net print ≤ −15 % (lev 0.32) or ≤ −20 % (lev 0.2), slightly earlier than the true −15.8 / −23.8. Funding and the maintenance-margin tier are ignored. Equity is marked at exits only, so intratrade drawdown is not in max DD, which flatters the no-stop books.
4. **The book sizes all ON signals at one fixed fraction**, with 2+2 slots and the global cap of 4 or 10. It does not model momentum/flip positions sharing the global cap. The 330 signals already taken by live FRENZY_LONG/WIDE would conflict with ON-scalp on the same pair (pair-flat); the books count them, not choose between them.
5. **The timing null** uses a 1-second L/H/C compression with bar-style fills on tick rows only. Its absolute means run about 0.06 higher than the tick pricer. The comparison is valid only because observed and null use the same pricer.
6. **Entry at +12 s, slippage 0.10.** The 0.02 % slippage note (hot-scalp study) would add about +0.08 %/trade to every cell. That moves no cell's CI above 0: the best S3 CI low would go from −0.18 to about −0.10.
7. **The ADX/DI values** use k5m_full 5m klines (Binance quote volume, not the live ccxt volume). Parity is exact on the 10 stamped live fills. Not tested: other ADX windows, a DI threshold other than 0, or adx_delta over a different lag (the live flag is fixed at 3 bars).
8. **Untested exits:** exits that use order-book or liquidation flow, scale-out (partial TP), and re-entry after a stop.

## Files

- `reports/FRENZY_ON_SCALP_PREREG_2026-10-06.txt`: the frozen plan.
- `reports/FRENZY_ON_SCALP_SIGNALS_2026-10-06.csv`: one row per ON signal (1,443) with action label, ADX/DI, dislocation, MFE/MAE per window, minutes to each +X / −Y, and LOCK2.
- `reports/FRENZY_ON_SCALP_GRID_2026-10-06.csv`: all 1,205 cells with every stat, the nulls and each gate leg.
- Scripts in the session scratchpad `ons/`: `price.py` (tick pricer, 8 shards), `analyze.py` (Part 1, grid, max-t, timing null), `part2.py` (gates, OOS, months, books, ruin, 3×3, strong comparison), `gr.py` + `live/` (GRIFFAIN aggTrades and walk; stamp parity).
