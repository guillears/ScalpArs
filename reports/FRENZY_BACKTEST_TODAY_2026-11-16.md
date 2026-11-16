# FRENZY family + Flatbase — 2-year backtest on TODAY's rules (2026-11-16)
Operator: "yes, rebuild the backtest for the Frenzy sleeves first" (after reports/OOS_AUDIT_2026-11-16.md). Read-only; script
`scripts/audit_frenzy_bt_today_2026_11_16.py` (no network). Nothing committed.

## Method (cheapest correct run — no new engine replay needed)
- **Entries** = the yr6 / yr6a REAL-ENGINE replay (Oct-13 stack, seeds 1-3, 2025-01-01 → 2026-10-11, warm-up trimmed, probes out).
  A fresh engine replay was not needed for LITE / LONG / WILLY: their ENTRY rules are unchanged since the snapshot except slot caps
  (2 → 10, WILLY 1 → 10: today takes MORE fills — second-order fills NOT modelled) and the armed filters (applied below as a second read).
- **Exits re-priced to today:** LITE = replay as booked (+3 / −3 / 720 min = today's) · LONG = +1.5 / −3 (first touch on the replayed
  peak — exact for a lower TP with the same stop / cap) · WILLY = the validated V2b walker (reports/PREP_B25_BT_WILLY_rows.csv arm A: each
  fill re-walked on its own path at target max(1.25, min(8, net at spike high × 0.99)), −4, late +0.25 @ 30, cap 90, resting maker TP;
  409 replay rows with no priced path dropped) — **the DL 541 runner is NOT modelled**.
- **WILLY_HT**: not in the replay (turnover ≥ 1 was untraded then) → PREP arm B study replica (2026-01-10 → 10-07, one realisation, V2b
  exit; today's HT +2 TP floor and HT FAILED_SPIKE filter not modelled). **RETRY**: in no replay → **untestable**.
- **Flatbase**: the S4 simulator (reports/FLATBASE_SL_GRID_1113_bt.pkl, TP 5 / SL 10 = today's, costs 0.20 %, one realisation; turnover
  > 1 already applied, higher-high = today's DL 540 filter).
- **Validation (replay vs real, 06-17 → 10-11):** master Frenzy fills found in replay seed 1 within 15 min: LITE 11 / 13 · WILLY 9 / 10 ·
  LONG 9 / 15. (Prior validations: yr6 fills vs master per sleeve in the ENGINE_REPLAY reports; WILLY walker ≤ 0.03 pts vs the engine.)
- **Costs:** replay LITE / LONG fills carry the replay's real-tick latency costs; V2b walker = fees, TP at the maker line. Live Frenzy stop
  slippage is small (median ≈ 0.01–0.05 %); a 0.10 % stress on every stop moves WILLY by ≈ −0.03 / fill (28 % of exits are stops) — no verdict changes.
- Stats per cell: N · WR · avg % [day-clustered 95 % CI] · P(avg > 0) · days · one-size DCR at 1× notional. **pre-June-2026 = clean
  out-of-sample for every fitted filter** (the master the filters were fitted on starts 2026-06-17).
WILLY replay rows without a V2b price (dropped): 409
Filter blocks on the replay (today's toggles): {'FRENZY_LITE:LITE_BTC_EMA20_DOWN': 965, 'FRENZY_LITE:LITE_BTC_EMA20_DOWN+LITE_WEAK_HOLD': 667, 'FRENZY_LITE:LITE_BTC_RSI_HOT': 198, 'FRENZY_LITE:LITE_BTC_RSI_HOT+LITE_WEAK_HOLD': 153, 'FRENZY_LITE:LITE_WEAK_HOLD': 1031, 'FRENZY_LONG:LONG_WEAK_FRENZY': 1024, 'FRENZY_WILLY:WILLY_FAILED_SPIKE': 1344, 'FRENZY_WILLY:WILLY_GAP': 635, 'FRENZY_WILLY:WILLY_GAP+WILLY_FAILED_SPIKE': 1178}

### FRENZY_WILLY
| read | seed | ALL | 2025 | 26-H1 | 26-H2 | pre-June-2026 (clean OOS) |
|---|---|---|---|---|---|---|
| no fitted filters | 1 | 1381 · 67% · -0.15 [-0.31, +0.01] · P 0.03 · 519d · DCR -0.423 | 818 · 63% · -0.40 [-0.61, -0.18] · P 0.00 · 283d · DCR -1.033 | 367 · 74% · +0.32 [+0.05, +0.57] · P 0.99 · 153d · DCR +0.586 | 196 · 74% · +0.02 [-0.32, +0.38] · P 0.53 · 83d · DCR -0.029 | 1169 · 66% · -0.20 [-0.38, -0.03] · P 0.01 · 426d · DCR -0.551 |
| no fitted filters | 2 | 1376 · 68% · -0.12 [-0.27, +0.03] · P 0.07 · 517d · DCR -0.339 | 811 · 63% · -0.35 [-0.54, -0.17] · P 0.00 · 281d · DCR -0.891 | 367 · 74% · +0.32 [+0.05, +0.57] · P 0.99 · 153d · DCR +0.587 | 198 · 75% · +0.03 [-0.33, +0.38] · P 0.55 · 83d · DCR -0.005 | 1162 · 66% · -0.16 [-0.32, -0.01] · P 0.02 · 424d · DCR -0.453 |
| no fitted filters | 3 | 1372 · 68% · -0.10 [-0.24, +0.04] · P 0.07 · 515d · DCR -0.311 | 806 · 63% · -0.34 [-0.55, -0.16] · P 0.00 · 279d · DCR -0.868 | 369 · 74% · +0.34 [+0.08, +0.61] · P 0.99 · 153d · DCR +0.619 | 197 · 75% · +0.05 [-0.30, +0.40] · P 0.62 · 83d · DCR +0.034 | 1159 · 66% · -0.15 [-0.32, +0.01] · P 0.03 · 422d · DCR -0.426 |
| today's filters | 1 | 325 · 69% · -0.08 [-0.37, +0.23] · P 0.31 · 248d · DCR -0.059 | 188 · 63% · -0.37 [-0.79, +0.04] · P 0.04 · 138d · DCR -0.214 | 83 · 80% · +0.41 [-0.12, +0.94] · P 0.94 · 66d · DCR +0.187 | 54 · 76% · +0.17 [-0.48, +0.84] · P 0.69 · 44d · DCR +0.072 | 268 · 68% · -0.13 [-0.44, +0.22] · P 0.24 · 201d · DCR -0.084 |
| today's filters | 2 | 324 · 69% · -0.08 [-0.40, +0.22] · P 0.30 · 249d · DCR -0.061 | 187 · 63% · -0.37 [-0.80, +0.02] · P 0.03 · 139d · DCR -0.218 | 83 · 80% · +0.41 [-0.15, +0.91] · P 0.93 · 66d · DCR +0.187 | 54 · 76% · +0.17 [-0.51, +0.85] · P 0.68 · 44d · DCR +0.072 | 267 · 68% · -0.13 [-0.46, +0.18] · P 0.22 · 202d · DCR -0.087 |
| today's filters | 3 | 323 · 70% · -0.05 [-0.36, +0.25] · P 0.36 · 248d · DCR -0.046 | 186 · 63% · -0.33 [-0.75, +0.07] · P 0.05 · 138d · DCR -0.191 | 83 · 80% · +0.41 [-0.15, +0.91] · P 0.93 · 66d · DCR +0.187 | 54 · 76% · +0.17 [-0.53, +0.85] · P 0.68 · 44d · DCR +0.072 | 266 · 68% · -0.10 [-0.43, +0.23] · P 0.31 · 201d · DCR -0.068 |

### FRENZY_LITE
| read | seed | ALL | 2025 | 26-H1 | 26-H2 | pre-June-2026 (clean OOS) |
|---|---|---|---|---|---|---|
| no fitted filters | 1 | 1454 · 49% · +0.00 [-0.16, +0.15] · P 0.50 · 518d · DCR -0.104 | 826 · 48% · -0.14 [-0.34, +0.07] · P 0.10 · 277d · DCR -0.428 | 372 · 51% · +0.23 [-0.04, +0.51] · P 0.94 · 146d · DCR +0.385 | 256 · 49% · +0.14 [-0.30, +0.57] · P 0.75 · 95d · DCR +0.195 | 1177 · 48% · -0.03 [-0.20, +0.14] · P 0.36 · 413d · DCR -0.167 |
| no fitted filters | 2 | 1445 · 49% · -0.01 [-0.15, +0.15] · P 0.47 · 515d · DCR -0.117 | 815 · 47% · -0.15 [-0.35, +0.05] · P 0.07 · 274d · DCR -0.452 | 372 · 51% · +0.23 [-0.05, +0.52] · P 0.94 · 146d · DCR +0.385 | 258 · 49% · +0.14 [-0.28, +0.55] · P 0.74 · 95d · DCR +0.192 | 1166 · 48% · -0.04 [-0.20, +0.13] · P 0.30 · 410d · DCR -0.183 |
| no fitted filters | 3 | 1444 · 49% · +0.01 [-0.15, +0.16] · P 0.52 · 512d · DCR -0.091 | 813 · 48% · -0.14 [-0.35, +0.07] · P 0.09 · 271d · DCR -0.421 | 373 · 51% · +0.23 [-0.05, +0.52] · P 0.94 · 146d · DCR +0.401 | 258 · 49% · +0.15 [-0.29, +0.58] · P 0.75 · 95d · DCR +0.221 | 1165 · 48% · -0.03 [-0.19, +0.14] · P 0.39 · 407d · DCR -0.157 |
| today's filters | 1 | 444 · 53% · +0.21 [-0.07, +0.48] · P 0.94 · 290d · DCR +0.115 | 262 · 53% · +0.17 [-0.20, +0.52] · P 0.83 · 158d · DCR +0.088 | 111 · 53% · +0.30 [-0.26, +0.88] · P 0.87 · 78d · DCR +0.169 | 71 · 54% · +0.24 [-0.49, +0.88] · P 0.75 · 54d · DCR +0.137 | 365 · 53% · +0.24 [-0.07, +0.55] · P 0.93 · 231d · DCR +0.137 |
| today's filters | 2 | 444 · 53% · +0.23 [-0.06, +0.50] · P 0.94 · 286d · DCR +0.123 | 259 · 53% · +0.20 [-0.18, +0.54] · P 0.85 · 155d · DCR +0.112 | 113 · 52% · +0.24 [-0.34, +0.81] · P 0.80 · 79d · DCR +0.131 | 72 · 54% · +0.28 [-0.43, +0.94] · P 0.79 · 52d · DCR +0.168 | 364 · 54% · +0.25 [-0.07, +0.56] · P 0.94 · 229d · DCR +0.142 |
| today's filters | 3 | 441 · 53% · +0.21 [-0.07, +0.49] · P 0.93 · 286d · DCR +0.114 | 253 · 53% · +0.18 [-0.20, +0.54] · P 0.82 · 152d · DCR +0.096 | 113 · 52% · +0.27 [-0.30, +0.83] · P 0.83 · 80d · DCR +0.150 | 75 · 53% · +0.23 [-0.43, +0.88] · P 0.75 · 54d · DCR +0.135 | 358 · 53% · +0.25 [-0.06, +0.56] · P 0.94 · 227d · DCR +0.137 |

### FRENZY_LONG
| read | seed | ALL | 2025 | 26-H1 | 26-H2 | pre-June-2026 (clean OOS) |
|---|---|---|---|---|---|---|
| no fitted filters | 1 | 623 · 64% · -0.15 [-0.33, +0.02] · P 0.04 · 322d · DCR -0.165 | 269 · 65% · -0.07 [-0.31, +0.17] · P 0.29 · 148d · DCR -0.066 | 230 · 63% · -0.17 [-0.45, +0.10] · P 0.10 · 107d · DCR -0.251 | 124 · 62% · -0.27 [-0.73, +0.16] · P 0.10 · 67d · DCR -0.367 | 485 · 64% · -0.12 [-0.33, +0.06] · P 0.10 · 248d · DCR -0.135 |
| no fitted filters | 2 | 614 · 64% · -0.16 [-0.33, +0.02] · P 0.05 · 318d · DCR -0.171 | 261 · 65% · -0.10 [-0.35, +0.17] · P 0.23 · 144d · DCR -0.086 | 230 · 63% · -0.17 [-0.46, +0.10] · P 0.11 · 107d · DCR -0.251 | 123 · 63% · -0.25 [-0.67, +0.15] · P 0.12 · 67d · DCR -0.336 | 477 · 64% · -0.14 [-0.33, +0.04] · P 0.06 · 244d · DCR -0.149 |
| no fitted filters | 3 | 613 · 64% · -0.16 [-0.34, +0.01] · P 0.03 · 318d · DCR -0.173 | 259 · 65% · -0.09 [-0.34, +0.17] · P 0.24 · 144d · DCR -0.082 | 229 · 63% · -0.18 [-0.47, +0.09] · P 0.11 · 107d · DCR -0.260 | 125 · 62% · -0.26 [-0.71, +0.17] · P 0.12 · 67d · DCR -0.351 | 474 · 64% · -0.14 [-0.35, +0.04] · P 0.07 · 244d · DCR -0.149 |
| today's filters | 1 | 277 · 65% · -0.15 [-0.44, +0.12] · P 0.16 · 197d · DCR -0.076 | 101 · 67% · +0.02 [-0.41, +0.45] · P 0.55 · 80d · DCR -0.002 | 120 · 59% · -0.36 [-0.81, +0.04] · P 0.04 · 78d · DCR -0.263 | 56 · 71% · +0.02 [-0.77, +0.63] · P 0.53 · 39d · DCR -0.010 | 214 · 63% · -0.20 [-0.51, +0.08] · P 0.09 · 154d · DCR -0.095 |
| today's filters | 2 | 274 · 65% · -0.15 [-0.43, +0.10] · P 0.13 · 194d · DCR -0.076 | 99 · 67% · -0.01 [-0.46, +0.42] · P 0.49 · 78d · DCR -0.010 | 120 · 59% · -0.36 [-0.80, +0.03] · P 0.03 · 78d · DCR -0.263 | 55 · 73% · +0.07 [-0.74, +0.68] · P 0.60 · 38d · DCR +0.021 | 212 · 62% · -0.22 [-0.52, +0.07] · P 0.07 · 152d · DCR -0.100 |
| today's filters | 3 | 275 · 64% · -0.16 [-0.43, +0.11] · P 0.13 · 195d · DCR -0.081 | 99 · 67% · -0.01 [-0.43, +0.43] · P 0.52 · 78d · DCR -0.010 | 120 · 59% · -0.36 [-0.80, +0.01] · P 0.03 · 78d · DCR -0.263 | 56 · 71% · +0.02 [-0.77, +0.61] · P 0.54 · 39d · DCR -0.010 | 212 · 62% · -0.22 [-0.52, +0.09] · P 0.08 · 152d · DCR -0.100 |

### FRENZY_WILLY_HT (study replica, 2026-01 → 10, one realisation; HT +2 TP floor not modelled)
| read | seed | ALL | 2025 | 26-H1 | 26-H2 | pre-June-2026 (clean OOS) |
|---|---|---|---|---|---|---|
| no fitted filters | 1 | 477 · 67% · -0.22 [-0.45, +0.02] · P 0.04 · 217d · DCR -0.450 | — | 259 · 64% · -0.37 [-0.70, -0.07] · P 0.01 · 127d · DCR -0.620 | 218 · 69% · -0.05 [-0.37, +0.26] · P 0.40 · 90d · DCR -0.167 | 226 · 65% · -0.39 [-0.71, -0.05] · P 0.01 · 115d · DCR -0.604 |

### FLATBASE_SHORT (S4 simulator, TP 5 / SL 10, costs 0.20 %, one realisation) — all signals
| read | seed | ALL | 2025 | 26-H1 | 26-H2 | pre-June-2026 (clean OOS) |
|---|---|---|---|---|---|---|
| no fitted filters | 1 | 1002 · 71% · +0.79 [+0.35, +1.19] · P 1.00 · 463d · DCR +0.785 | 421 · 66% · +0.21 [-0.44, +0.83] · P 0.71 · 215d · DCR -0.009 | 334 · 72% · +0.70 [-0.09, +1.48] · P 0.96 · 142d · DCR +0.797 | 247 · 79% · +1.89 [+1.12, +2.59] · P 1.00 · 106d · DCR +0.581 | 747 · 68% · +0.38 [-0.12, +0.88] · P 0.93 · 356d · DCR +0.172 |

### FLATBASE_SHORT — today's higher-high filter applied (fitted on the master: DL 540)
| read | seed | ALL | 2025 | 26-H1 | 26-H2 | pre-June-2026 (clean OOS) |
|---|---|---|---|---|---|---|
| no fitted filters | 1 | 912 · 72% · +0.92 [+0.49, +1.34] · P 1.00 · 447d · DCR +0.882 | 383 · 68% · +0.46 [-0.17, +1.11] · P 0.92 · 204d · DCR +0.254 | 308 · 72% · +0.75 [-0.12, +1.54] · P 0.95 · 138d · DCR +0.816 | 221 · 79% · +1.95 [+1.11, +2.71] · P 1.00 · 105d · DCR +0.535 | 683 · 69% · +0.53 [-0.02, +1.02] · P 0.97 · 341d · DCR +0.351 |
## Verdicts (same bar as the OOS audit)
| Sleeve | Raw (no fitted filters), 3 seeds | Today's filters, pre-June (clean OOS) | Live (OOS audit) | Verdict |
|---|---|---|---|---|
| **Flatbase short** | 1,002 · 71 % · **+0.79 [+0.35, +1.19]**, P 1.00, 463 days; every regime ≥ +0.21 | hh-filter: pre-June 683 · 69 % · +0.53 (P 0.97) | 1 · +$40 | **PROVEN (backtest)** — the only sleeve with a raw CI above 0; caveat: simulator, one realisation |
| **FRENZY_LITE** | ≈ 1,450 / seed · 49 % · ±0.01 (P 0.47–0.52) | **358–365 · 53 % · +0.24 / +0.25 / +0.25 (P 0.93–0.94)**, positive in 2025, 26-H1 and 26-H2 | 14 · 50 % · −$218 | **UNPROVEN, leaning yes** — no raw edge, but today's filters add ≈ +0.25 / fill on data they never saw, 3 / 3 seeds |
| **FRENZY_WILLY** | ≈ 1,375 / seed · 67 % · **−0.10 to −0.15** (P 0.03–0.07); 2025 −0.34…−0.40 (CI < 0) | 266–268 · 68 % · −0.10 to −0.13 (P 0.22–0.31) | 21 · 76 % · +$178 | **REFUTED on backtest** (runner and the 10-slot change not modelled; 26-H1 +0.32 CI > 0 — regime-dependent) |
| **WILLY_HT** | 477 · 67 % · **−0.22** (P 0.04); 26-H1 −0.37 (CI < 0) | filters not modelled | 1 · −$439 | **REFUTED (weak evidence)**: study replica, HT +2 floor not modelled |
| **FRENZY_LONG** | ≈ 615 / seed · 64 % · **−0.15 / −0.16** (P 0.03–0.05) | 212–214 · 62–63 % · −0.20 / −0.22 | 7 · −$13 | **REFUTED** |
| **LITE RETRY** | — | — | 3 · +$56 | **UNTESTABLE** (no replay has it) |

**Platform:** out of sample, only **Flatbase** (strong) and **filtered LITE** (≈ +0.25 / fill, P ≈ 0.94) are positive. WILLY, HT and
FRENZY_LONG are negative across 2 years and 3 seeds; WILLY is positive only in the 26-H1 bear stretch.

**Caveats:** entries are the Oct-13 snapshot (today's 10 slots would take extra fills, unmodelled); WILLY runner unmodelled (its own
study claimed a large backtest gain on 115 falling / mild fills — not re-verified here); HT / Flatbase are study replicas, one realisation;
RETRY untestable without a new replay (≈ 1–2 days of engine time).
