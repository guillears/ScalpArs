# DIST_SQUEEZE_FUEL — 2026-11-22: do "crowded short" coins squeeze DIST shorts? → NOT SUPPORTED (net leans the other way)

**Question (operator):** test theory (e) now with the data we have and add it to the watchlist if it holds. Pre-registration: `reports/DIST_SQUEEZE_FUEL_2026-11-22_PREREG.md` (written before any outcome). Script: `scripts/dist_squeeze_fuel_1122.py` → `_tables.csv`, `_signs.csv`, `_before_after.csv`, `_cuts.csv`.

**Theory (e):** the coin is already crowded with shorts → the first up-tick forces short covering → fast stop. Motivated by the batch losers (stopped in 2–37 min), so the batch is not evidence.

## Result
Backtest = DIST_WL trades on today's armed rule (RSI1h ≥ 74.07, Exit B): 1,536 trades, base fast-stop (stop ≤ 60 min) 38.8 %, +0.89 %/trade (the DIST_WL build; DL 614's live-aligned re-run of the same rule reads +0.82 — same pool, live metric timing). Crowded side = the frozen DESIGN tercile; unscored rows are out of both sides.

| variable (crowded side) | DESIGN fast-stop crowded vs rest | DESIGN net crowded vs rest | HOLDOUT fast-stop | HOLDOUT net |
|---|---|---|---|---|
| funding low | 33.7 vs 43.5 % | +0.93 vs +0.45 | 32.1 vs 37.3 % | +2.30 vs +1.17 |
| premium low (HOLDOUT 67 % scored) | 37.0 vs 41.6 % | +1.08 vs +0.40 | 38.3 vs 37.4 % | +1.70 vs +1.04 |
| top-trader position L/S low | 37.5 vs 41.6 % | +1.14 vs +0.34 | 36.2 vs 37.0 % | +1.85 vs +0.90 |
| top-trader account L/S low | 40.5 vs 40.1 % | +0.75 vs +0.54 | 34.6 vs 37.2 % | +2.58 vs +0.97 |
| global L/S low | 40.1 vs 40.3 % | +0.85 vs +0.49 | 38.2 vs 36.2 % | +2.04 vs +1.08 |
| taker ratio low | 38.2 vs 41.2 % | +0.94 vs +0.45 | 34.0 vs 37.6 % | +1.54 vs +1.24 |
| OI 1h rising (high) | **42.2 vs 39.2 %** | +1.02 vs +0.41 | **44.8 vs 32.9 %** | **+0.85 vs +1.53** |
| crowd score ≥ 3 of 5 | 40.0 vs 40.1 % | +0.66 vs +0.61 | 38.7 vs 37.3 % | +2.29 vs +0.95 |

- Theory-family null |t| 2.83; the largest DESIGN |t| is 1.60 → **nothing is significant in either direction**; no variable passes the pre-registered rule.
- **Net %:** the six positioning variables (funding, premium, the three L/S ratios, taker) lean the OTHER way in both blocks — crowded shorts earn more, not significant.
- **Fast-stop rate:** mixed / flat (premium and global L/S HOLDOUT, account L/S DESIGN, score DESIGN lean slightly the theory's way).
- **OI rising into the signal** is the one row that partly fits the theory: more fast stops in both blocks and worse net on HOLDOUT (+0.85 vs +1.53), but better net on DESIGN, a non-monotone dose (+0.59 / +0.23 / +1.02) and not significant — it does not confirm DIST_WL P3 ("rising OI better") on the fast-stop outcome either.
- Sign granularity (funding < 0, premium < 0, top-trader / global L/S < 1): same picture — net higher on the crowded side in 7 of 8 cells, none significant (|t| ≤ 1.83).
- Crowd-score dose is not monotone (DESIGN net by score 0–5: −0.34 / +1.53 / +0.94 / −0.24 / +1.34 / +2.60).

## Before → after (block the crowded side; unscored rows kept, as live fails open)
| rule | backtest: N · WR · avg · DCR (0.61×) · max DD | blocked | current batch (6 fills, one size $6,194) |
|---|---|---|---|
| today | 1,536 · 42 % · +0.89 · +0.56 %/day · −65 % | — | 6 · 17 % · −2.83 · −$1,050 |
| block funding ≤ −0.0039 % | 1,149 · 41 % · +0.79 · +0.36 · −60 % | 387 · 45 % · +1.21 | 3 · 0 % · −5.46 · −$1,015 (blocks CYBER W, KERNEL L, SENT L) |
| block top-trader L/S ≤ 1.19 | 959 · 40 % · +0.54 · +0.18 · −51 % | 577 · 46 % · +1.47 | 5 · 20 % · −2.17 · −$671 (blocks KERNEL L) |
| block crowd score ≥ 3 | 1,226 · 42 % · +0.85 · +0.42 · −55 % | 310 · 43 % · +1.05 | not computable (premium / account / global L/S not stamped live) |

Every block lowers the daily compound; each also cuts max drawdown (fewer trades), but at a lower return. Batch: the live `entry_funding_rate` is a fraction (backtest `fund_last` is in %), so the cut is −0.000039 live.

## Verdict
**No watchlist line.** Theory (e) is not supported: nothing passes, and on net the crowded-short side leans the other way in both blocks (not significant). The batch's fast squeezes are not explained by crowding at entry; the order-flow read stays in the Route B weekly scan (≈ 12-10). Not a multiplier candidate either (WR 41–46 % ≪ the W-bar 70 %).

## Deviations / blind spots
- The pre-registered descriptive Route B read (pre-entry short liquidations on 11-19+ DIST signals) was not run — it could not pass or fail anything; it stays in the 12-10 weekly scan.
- The first version of the script counted unscored `prem_last` rows (33 % of HOLDOUT) as "rest" and the score as not crowded; fixed before commit (unscored ≠ rest). The PREREG's monotone-dose rule and sign-granularity reads were added in the same fix; the top-pair share = the worst losing pair ÷ all losing pairs' loss, crowded side, DESIGN.
- Premium / account / global L/S are not stamped on live DIST fills (only funding and the top-trader position ratio).
- Backtest funding = settled rate; live = the premiumIndex rate. The trades file is one position per pair on the unfiltered set, so a block cannot let a later signal on the same coin back in (after-N slightly understated; same as the earlier DIST studies).
