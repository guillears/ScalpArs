# Momentum LONG · PAIR_1H_DOWNTREND 2D follow-up: coin RSI/ADX on 1h and 4h, the full price-vs-EMA grid, and 3-way cells (2026-10-16)

- **Operator questions:**
  - "was every variable tested?"
  - then "why not EMA20 and EMA13 too?"
- **Parent study:** `reports/ML_DOWNTREND_2D_2026-10-16.md`. Its verdict was NOTHING.
- **Addendum:** `reports/ML_DOWNTREND_2D_PREREG_ADDENDUM.txt`, SHA-256 `ba21014b4f75eee38da60a7a75259f272277ffa97009700a5b6fab83b141eed0`.
  - It was frozen before any outcome table on the new features.
  - The operator's EMA-grid extension was folded in before any outcome table too.
- **Safety:** research only. No engine, config, template or test edits, no commit and no bot API calls. `backtest_cache` and `engine_replay*.py` were only read.
- **Binance:** 26 public klines requests (1h and 4h, limit 499, weight about 2 each). Used weight peaked at 164, the shared cooldown was honoured, and the files are stored under `reports/ml_dt2d_fetch/{1h,4h}/`.
- **Validation:** `scripts/validate_against_master.py` reports **ALL CHECKS PASS**.
- **Parity:**
  - Same blocks as the parent: replay 2617 / 1171 / 521 fills, master 191 · +0.027 %, fresh 29 · −0.118 %. The live batch is 6 fills from `scalpars_orders_live_2026-10-16_15-19-54 (1).csv`.
  - All 697 parent cells reproduce exactly (max |Δz| 4e-15), and the shuffles are identical (same seed).

---

## Plain-English answer

1. **No.** None of the new variables holds in the backtest **and** master **and** the fresh fills, either alone inside the downtrend zone or in a 3-way cell. This covers:
   - the coin's RSI14 and ADX14 on 1h and 4h, both level and direction;
   - the coin's price distance from EMA5/8/13/20/50/100/200 on 5m, 1h and 4h, with above/below flags.
2. **Not one new cell clears even the weakest consistency bar:**
   - 0 of 213 new 2D cells and 0 of 535 3-way cells are same-sign with |z| ≥ 1.96 in all three backtest eras.
   - The best new 2D cell has a consistency score of 0.42, against a shuffled-null 95th percentile of 1.41.
   - The best 3-way cell scores 0.78 (null 1.76) inside the zone and 0.99 (null 1.74) as zone ∧ cell vs the rest.
   - **Family-wise p over everything tested (parent 697 + new 225 + 3-way 700 = 1,622 cells, 1,394 valid) is ≥ 0.78 for the best cell.**
3. **Walk-forward (discover on 2025, confirm on both 2026 halves):**
   - new 2D: 0 confirmed (null median 2);
   - 3-way: 0 confirmed inside the zone, and 2 confirmed in the zone ∧ cell vs rest read against a null median of 4. That is below luck.
4. **The only recurring 3-way theme is "ADX falling (5m) together with something".** Examples:
   - 4h ADX ≤ 25;
   - price above the 4h EMA5;
   - price above the 5m EMA200.

   These cells are worse in all three eras and in master (about −0.2 to −0.4 %). But they:
   - have family-wise p 0.95–1.00;
   - have 0–3 fresh fills;
   - would have refused the live PEPE +1.16 winner.

   This is the same "ADX falling" read already frozen on the existing ADX line (DECISION_LOG 273). It adds nothing new.
5. **The theory's "early reversal" story fails again.** "4h RSI rising ∧ 1h gap narrowing" is *worse*, not better: −0.08 / −0.06 / −0.19 across the eras, master −0.14, fresh −0.38. That is the opposite of the reversal prediction, and its family-wise p is 1.00.
6. **Verdict: NOTHING.** No new line and nothing to arm. PAIR_1H_DOWNTREND stays a 1D observe line with its frozen gate.

---

## What was added (addendum, lookahead-free)

- **Bars:** 1h and 4h OHLC built from cached 5m klines (UTC buckets). Only **closed** bars are used, meaning close time ≤ the fill time. "Price" is the close of the last closed 5m bar (at most 15 min old).
- **Indicators:** the `ta` library, the same one the engine uses: RSI14, ADX14 and EMA(span). Direction = value minus the value one closed bar earlier.
- **Warm-up:** at least 100 closed bars for RSI/ADX, and at least 2 × span bars for an EMA. The warm-up counter restarts after a real data gap.
- **ADX inputs:** ADX needs real high/low. Some recent fresh/master fills only had close-only 5m cache, so their 1h/4h klines were fetched (12 coins).
  - Parity of fetched vs cache values where both exist: max |Δ| 0.009 (ADX 1h), 0.07 (ADX 4h), 0.002 (RSI).

## Coverage list (updated): what has now been tested on the momentum-long downtrend zone

| group | variables | where | coverage (replay / master / fresh) |
|---|---|---|---|
| coin RSI14 | 5m (`entry_rsi`, prev, dir) · **1h level + dir · 4h level + dir** | parent · **new** | 100 / 100 / 100 % |
| coin ADX14 | 5m (`entry_adx`, prev, Δ, dir) · **1h level + dir · 4h level + dir** | parent · **new** | ≥ 99.4 / 100 / 100 % |
| price vs EMA, 5m | EMA5 (`entry_price_vs_ema5_pct`) · EMA13 (`entry_dist_from_ema13_pct`) · **EMA8, EMA20, EMA50, EMA100, EMA200** | parent (2) · **new (5)** | 100 % |
| price vs EMA, 1h | EMA20 (`k_px_e20`) · EMA200 (`k_px_e200`) · **EMA5, EMA8, EMA13, EMA50, EMA100** | parent (2) · **new (5)** | 100 % |
| price vs EMA, 4h | **EMA5, EMA8, EMA13, EMA20, EMA50, EMA100, EMA200** | **new (7)** | 100 % |
| above/below flags | **x > 0 / x ≤ 0 for all 21 distances** (incl. the 4 parent ones, which the parent only cut at quantiles) | **new** | 100 % |
| 3-way | **DT × each of the 25 new variables (median or sign halves) × each of the 7 theory variables (bad/good side)** | **new** | 700 cells, 535 valid |
| everything else | 106 parent variables: every entry_* stamp, d_* directions, k_* kline rebuilds, m_* market features | parent | see parent blind-spot list |

**Notes:**
- The engine's 5m stamps use the live price against an EMA that includes the forming bar. The new 5m distances use the last *closed* bar, as the operator specified. So they are a different quantity, not a duplicate: rank correlation 0.2–0.6 with the stamps.
- **Blind spots left:**
  - the parent's list (pattern flags constant in the backtest, `entry_chop_burst_prior_fill_s`, m_rs15/rs60 at 89 % in H1, fresh market features at 66 %);
  - DI+/DI− on 1h/4h, which were not asked for and not built;
  - 3-way cells other than "new variable × theory variable". The full 3-way over all 131 variables (about 300k cells) was not run by design (limited per the request).
- No new variable had < 90 % coverage, so no new blind spot.

## Results (full tables: `reports/ML_DT2D_FU_tables.txt`, per-cell CSV `reports/ML_DT2D_FU_screen.csv`)

### Family-wise reads (same 200 day-block shuffles for every family)

C = the signed minimum |z| over the 3 eras, counted only when all three have the same sign.

| family | stat | valid cells | cells \|C\| ≥ 1.96 (obs / null med) | max \|C\| observed | null 95th | fw-p best cell |
|---|---|---|---|---|---|---|
| new 2D | in-zone split | 213 | 0 / 0 | 0.42 | 1.41 | 0.995 |
| new 2D | zone∧X vs rest | 213 | 0 / 0 | 0.72 | 1.54 | 0.855 |
| 3-way | in-zone | 535 | 0 / 0 | 0.78 | 1.76 | 0.970 |
| 3-way | vs rest | 535 | 0 / 0 | 0.99 | 1.74 | 0.795 |
| parent + new 2D | in / rest | 859 | 0 / 0 | 0.94 / 1.16 | 1.73 / 1.78 | 0.955 / 0.660 |
| **UNION (all 1,622)** | in / rest | 1,394 | 0 / 0 | 0.94 / 1.16 | 1.88 / 1.87 | **0.985 / 0.775** |

### Direction-consistency, the operator's question taken literally

Cells whose inside-zone Δ has the same sign in all 3 eras, master as-traded, stack_keep and fresh:

| family | consistent cells | luck expectation (6 coin-flip signs) | of those with C ≥ 1.0 |
|---|---|---|---|
| new 2D | 19 of 213 (complementary pairs, so about 10 distinct) | ≈ 7 | **0** |
| 3-way | 26 of 535 | ≈ 17 | **0** |

All of them are at noise strength (|C| ≤ 0.62).

### Top cells: three blocks side by side (inside-zone Δ; n · avg · Δ)

| cell | 2025 | 2026-H1 | 2026-H2 | C / fw-p (union) | master as-traded | fresh | LIVE |
|---|---|---|---|---|---|---|---|
| 1h ADX 20–30 (new 2D best) | 509·−0.17·+0.02 | 179·+0.04·+0.19 | 57·−0.11·+0.05 | +0.42 / 1.00 | 26·+0.04·+0.14 | 10·−0.15·+0.01 | 3·+0.45 |
| 4h ADX rising | 729·−0.21·−0.05 | 268·−0.11·−0.03 | 82·−0.23·−0.15 | −0.39 / 1.00 | 34·−0.07·−0.06 | 9·−0.19·−0.08 | 2·**+1.06** |
| price above 5m EMA100 | 1162·−0.19·−0.05 | 432·−0.10·−0.04 | 136·−0.17·−0.08 | −0.36 / 1.00 | 53·−0.10·−0.28 | 14·−0.07·**+0.38** | 4·+0.17 |
| 4h ADX ≤ 25.4 ∧ 5m ADX falling (3-way best) | 284·−0.26·−0.10 | 87·−0.16·−0.08 | 47·−0.30·−0.21 | −0.78 / 0.99 | 13·−0.20·−0.20 | 3·−0.08·+0.09 | 1·**+1.16** |
| price ≤ 5m EMA100 ∧ low pair volume | 226·−0.12·+0.08 | 84·−0.01·+0.10 | 17·+0.04·+0.22 | +0.72 / 1.00 | 10·+0.18·+0.26 | 4·−0.45·**−0.38** | 0 |
| price > 5m EMA200 ∧ 5m ADX falling | 244·−0.28·−0.12 | 80·−0.18·−0.11 | 32·−0.23·−0.09 | −0.61 / 1.00 (rest: −0.99 / 0.95) | 11·−0.29·−0.30 | 0 | 0 |
| 4h RSI rising ∧ 1h gap narrowing | 510·−0.24·−0.08 | 176·−0.13·−0.06 | 64·−0.28·−0.19 | −0.62 / 1.00 | 22·−0.13·−0.14 | 8·−0.36·−0.38 | 3·−0.17 |

### Before / after for the top cells (two-sided: refused winners count against the rule)

The full tables in `ML_DT2D_FU_tables.txt` §5 have per-trade rows for every refused master and fresh fill.

**How each rule was applied:**
- A cell with a negative Δ was refused directly.
- A cell with a positive Δ was turned into "refuse the rest of the zone".

| rule | master as-traded Σ% (Σ$ recorded) | master stack_keep avg/fill | fresh Σ% (Σ$) | **current LIVE batch** Σ% (Σ$) |
|---|---|---|---|---|
| keep only 1h ADX 20–30 inside DT | +5.2 → +9.0 (+$802 → +$1,679) | +0.141 → +0.202 | −3.4 → −2.2 (−$852 → −$806) | +0.9 → **+1.6** (+$4 → +$5) |
| refuse DT ∧ 4h ADX rising | +5.2 → +7.6 (+$802 → +$1,280) | +0.141 → +0.195 | −3.4 → −1.7 (−$852 → −$154) | +0.9 → **−1.2** (+$4 → −$2): refuses PEPE +1.16 and DOT +0.96 |
| refuse DT ∧ 4h ADX ≤ 25 ∧ ADX falling | +5.2 → +7.8 (+$802 → +$1,236) | +0.141 → +0.156 | −3.4 → −3.2 | +0.9 → **−0.2** (refuses PEPE +1.16) |
| keep only (≤ 5m EMA100 ∧ low vol) inside DT | +5.2 → +9.7 (+$802 → +$1,582) | +0.141 → +0.212 | −3.4 → −2.5 | +0.9 → +0.3 (refuses PEPE + DOT) |

**How to read this table:**
- The master gains are mostly the parent's finding again: refusing *most* of the zone helps master because the zone loses there.
- Every rule is unstable at the batch level. For example, "keep only 1h ADX 20–30" costs BASE −4.5 Σ% while helping B1/B2/B6/B18 (see §5 per batch).
- Two of the four rules would have **lost money on the current live batch**.

## Verdict

- **NOTHING.** No new 2D or 3-way cell holds in all three backtest eras, beats its shuffled null on the union family, and points the same way in master and the fresh fills.
- The coverage gaps the operator raised are now closed:
  - 1h/4h RSI and ADX, level and direction;
  - the full 5m/1h/4h × 7-EMA distance grid with above/below flags;
  - a limited 3-way.
- The recurring "ADX falling" 3-way theme belongs to the existing ADX-falling line (DECISION_LOG 273). It is not a new downtrend sub-cut.
- PAIR_1H_DOWNTREND stays at KEEP OBSERVING on its frozen 1D gate, with one re-read at 30.

## Files

- `reports/ML_DOWNTREND_2D_PREREG_ADDENDUM.txt` — addendum (sha above).
- `scripts/study_ml_dt2d_fu_build.py` → `reports/ML_DT2D_FU_features.pkl`, `reports/ML_DT2D_FU_coverage.csv`.
- `scripts/study_ml_dt2d_fu_fetch.py` → `reports/ml_dt2d_fetch/1h/`, `reports/ml_dt2d_fetch/4h/` (13 coins).
- `scripts/study_ml_dt2d_fu_analyse.py` → `reports/ML_DT2D_FU_tables.txt`, `reports/ML_DT2D_FU_screen.csv`.
