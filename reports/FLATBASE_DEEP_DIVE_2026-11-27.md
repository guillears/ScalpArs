# FLATBASE deep dive — 2026-11-27 (research only, no bot change)

**Question (operator):** Flatbase results are terrible this batch (1 win in 6 closed, open shorts not working). Why, and what do we change?

**Prior / same-day work (re-used, not re-run):** OOS_REAL_FLATBASE W/L 11-17 (3,848 tests, top survivors failed holdout) · DL 601 funding-above-base
filter · DL 603 TP grid · DL 616 funding rules · DL 622 time caps · DL 649 base trend + entry timing · today: the no-progress / hold stop
(reports/FLATBASE_NOPROGRESS_STOP_2026-11-27*, 21 cells, all lose) · real-fill W/L vs a shuffled null (nothing beyond luck; the top real
separators were "this batch" markers — size, paper cost model, BTC drawdown) · BTC 30 d drawdown and BTC distance from its 24 h low (both FB's
BEST zones) · the 2-year rebuild re-run through 11-27 (reports/FLATBASE_REBUILD_2026-11-27_data/window_episodes.csv).
**New here:** (1) the DIST question — coin-level crowding × crash zone, pre-registered, with a luck null · (2) Flatbase entered on a coin
where another short sleeve was just stopped (cross-sleeve squeeze) · scripts/FLATBASE_DEEP_DIVE_2026-11-27.py.

## The batch (closed, today's stack)
| | N | WR | avg %/fill | daily compound |
|---|---|---|---|---|
| As traded | 6 | 17 % | −7.81 | −2.95 %/day |
| Today's stack (funding filter drops PHAROS) | 5 | 20 % | −7.38 | −2.34 %/day |

A 6-fill stretch this bad appears in **0.13 %** of the backtest's windows (worst ever), yet the rebuilt backtest took the same setups and lost
the same way (4 / 4 shared setups same outcome) → the strategy's own bad stretch, not a live ≠ backtest gap.

## Results
**1 · Crowding (backtest 545 fills, today's exit; family-wise luck p 0.84 over 20 cells — nothing):**

| Cell | N | WR | avg |
|---|---|---|---|
| all fills | 545 | 75.8 % | +1.13 (breakeven WR 67.9 %) |
| funding low third (shorts paying) | 182 | 80 % | +1.58 |
| funding high third | 182 | 77 % | +1.43 |
| crash zone (BTC ≤ −20 %) | 46 | 85 % | +2.23 |
| crash ∧ taker buy/sell at/above median (worst cell) | 19 | 68 % | −0.09 |
| crash ∧ top-trader L/S at/above median | 19 | 74 % | +0.57 |

Funding buckets: ≤ −0.10 % **211 · 79 % · +1.33 (the BEST bucket)** · −0.10…−0.03 77 · +0.49 · −0.03…−0.005 63 · +1.21 · −0.005…0 44 · +0.54 ·
at base 150 · +1.34. The live losers SLX / BIGTIME / KERNEL had funding −0.13 … −0.19 % — Flatbase's best historical bucket.

**2 · Cross-sleeve squeeze (an EXH / DIST short stopped on the same coin in the 48 h before):** 222 · 77 % · +1.15 vs 323 · 75 % · +1.13 — no effect.

## Meaning
Unlike DIST, Flatbase has **no** market or coin variable that marks its losers: every reading that describes this batch (deep BTC drawdown,
BTC bouncing, negative funding, crowded shorts, a squeezed coin) is neutral or its best zone over two years. The rebuild confirms the strategy
itself would have taken and lost these trades. What is left is an extreme bad stretch (or a regime the 2-year sample never contained).

## Verdict
- **No filter, no exit change** — anything fitted to the batch losers (5 of the 7 real losers on 34 merged fills, one crash window) would be built against the 2-year evidence.
- **Not go-live ready:** real fills 34 · 79 % with a one-sided 95 % win-rate floor ≈ 66 % < breakeven 67.9 % → Kelly at the floor ≤ 0. It stays
  in paper until the real floor clears breakeven (size-by-edge rule).
- **Watch (no new line needed — existing gates cover it):** the FB live-vs-rebuild parity per batch (does the rebuild keep losing the same
  setups?) and the three open shorts on their own exits (the hold study says closing early loses on average).
- Re-open trigger: the next 10 kept fills below 60 % WR, or the rebuild diverging from live (live loses what the rebuild wins).

---

## Follow-up (same day) — master + batch merged, FB_CRASH, crash 2D
- **Correction:** the real-fill read uses 34 fills (master 28 on today's stack + batch 6 as traded), not "5–8 trades"; on one basis (batch on today's stack, PHAROS out) 33 · 27 W / 6 L → win-rate floor 68.5 % vs breakeven 67.9 % (≈ at breakeven; 63.8 % if the open MOVE / LAYER shorts stop).
- **Real 34:** BTC % below its 30 d high splits them (all 5 batch losers at −25.7 … −27.0 %), but that is ONE crash episode; the backtest's 11
  crash episodes are 9 positive (zone 46 · 85 % · +2.23).
- **FB_CRASH block (rejected by the operator — filters too many winners):** master DCR +2.49 → +2.26 %/day · batch −2.34 → 0 (all 5 fills) ·
  backtest DCR 0.38 → 0.31 %/day, max DD −41.3 → −47.8 %.
- **Crash × every pre-entry feature (40 features, 80 cells — slip_e excluded as outcome, scripts/FLATBASE_CRASH_2D_2026-11-27.py):** 1 negative cell (taker L/S > median,
  19 · 68 % · −0.09), family-wise p 0.43 → no 2D filter either. Verdict unchanged: no filter, no exit change, PAPER.

## Follow-up 2 — review corrections + funding re-run (DECISION_LOG 652)
- **Today's-filter basis:** master 17 (KEPT_LIST, all TP) + batch kept 5 = 22 · 18 W / 4 L → win-rate floor **65.1 % < breakeven 67.9 %**
  (the "34" / "33" pools used the master stack_keep 28 without DL 601). Verdict unchanged: PAPER.
- **Parity:** 4 / 4 shared kept setups match; first miss PHAROS (rebuild TP, live STOP; now dropped by DL 601).
- **Backtest crash zone extended through 11-27:** ≈ 52 · 79 % · +1.38 vs rest +1.03; ≈ 6 of 8 zone runs positive — no FB_CRASH filter.
- **Funding re-run (scripts/FLATBASE_FUNDING_RERUN_2026-11-27.py):** real budgets / the new hourly rule gain only on this batch (KERNEL,
  BIGTIME, MOVE) and cut the master's POWR winner; backtest: every cell loses (budgets −0.004 … −0.18, hourly −0.06 / −0.08). No funding rule.
