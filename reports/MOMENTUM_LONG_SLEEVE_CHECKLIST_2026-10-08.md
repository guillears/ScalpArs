# MOMENTUM LONG — full sleeve-kill checklist (2026-10-08)

**Why.** `MOMENTUM_LONG_RECALL_TRACE_2026-10-08.md` put momentum longs (ML) under today's rules at about **−0.06 %/trade**
(range −0.12 … +0.03). The operator asked for the full 🔒 SLEEVE-KILL CHECKLIST (CLAUDE.md) before any size cut or pause proposal.
This report runs all four items, puts every survivor through the locked expectancy bar, and adds the sizing angle.

**Validation first.** `scripts/validate_against_master.py` → **ALL CHECKS PASS**. The feature frame was checked against today's
`MASTER_POOL_stacked.csv` (STACK 10-08c) before use: the 130 kept ML fills and their `stack_pct` match exactly (max |Δ| 0).
Today's size rule (`build_master_pool.today_size_scale`) reproduces the master's `stack_pnl` multiplier on every row (max |Δ| 0.002).

**Scope.** Read-only on code and config. No bot API. **No Binance calls** (every input came from the existing caches). No commit.

**New files (all `study_ml_checklist_` prefixed):**
- `scripts/study_ml_checklist_build.py` → `reports/study_ml_checklist_frame.pkl` (cohorts, tags, today's size), `reports/study_ml_checklist_astraded.csv` (live as-traded fills per batch)
- `scripts/study_ml_checklist_scan.py` → items ①②: `reports/study_ml_checklist_scan_{Y_all,Y_exw,M_all,M_exw}.csv`, `_scan_oos_*.csv`, `_lomo_*.csv`, `_scan_summary.txt`
- `scripts/study_ml_checklist_regime.py` → item ②: `reports/study_ml_checklist_regime_{buckets,verdict,splits_*,dayrho}.csv`, `_regime_summary.txt`
- `scripts/study_ml_checklist_cohorts.py` → items ③④, the bar, sizing: **`reports/study_ml_checklist_tables.md`** (every table), `_cohort_month.csv`, `_tape_month.csv`, `_tape_batch.csv`, `_candidates.csv`, `_sizing.csv`

**Cohorts.**

| name | what | N |
|---|---|---|
| **yr5** | engine replay ML fills, code 181131e, today's frozen config, trimmed to each chunk window, 3 seeds (replicates; N per seed; days pooled), Jan-04 → Oct-02 | 595/seed |
| yr5 ex-washed | minus the washed-out window (Jun-18 → Jul-2) and any fill with BTC ≤ −15 % below its 30-day high | 425/seed |
| **master** | `MASTER_POOL_stacked` kept, non-probe, CLOSED MOMENTUM LONG, ex-B1 (B1 = 1× probe era), `stack_pct` | 115 |
| master ex-washed | same, minus the washed-out window | 95 |
| live as-traded | every full-size live ML fill as traded, BASE → B18 (trace signals + master B17/B18) | 213 |

Units: every market-wide variable is judged in **DAYS** (day-clustered bootstrap; nulls shift whole days). % is per trade at 1×
(leverage-invariant). $ is a fixed $3,000 book (yr5 convention, `BASE_NOTIONAL_FRAC` 4.875), never compounding.

---

## Plain-English answer

1. **All four checklist items were run.** Pair-level dimensions, market/regime dimensions, the cohort split and the tape comparison.
2. **No single filter is strong enough to ship.** About 20,000 one- and two-variable splits were tested per cohort. None beats
   what luck produces in shuffled data. The same holds inside the pair-only and macro-only families.
3. **But the market regime clearly matters, as a group.** On the backtest, 25–30 of 61 market variables "stay alive". Chance
   explains about 10. Almost all of them say one thing:
   - Momentum longs lose when BTC and the alts are **falling on the 1-hour to 3-day scale**.
   - They are roughly flat when the tape is rising.
4. **The loss is spread across the whole sleeve.** Every cohort is negative in both halves of the year: UNMATCHED, CALM3D, quiet,
   mid and crowded books, pattern matches, heat, chop, maker/taker. No cohort carries the loss, so there is no targeted fix.
   Per the checklist, that points to a market-regime cause, which item ② then found.
5. **One condition looks like it pays: "trend-aligned" longs.** BTC 1h trend up, BTC 4h trend up, and the coin's own 4h EMA20 above
   its EMA50.
   - Backtest: +0.079 %/trade (all 3 seeds +0.07…+0.09, both halves positive, 8 of 10 months positive). Everything else: −0.116.
   - Live master agrees on direction: +0.235 vs +0.147. Ex-washed it is +0.235 vs −0.043.
   - **It is post-hoc and does not beat the shuffled-data bar.** After the in-sample haircut it is worth about **0 to +0.035 %/trade**.
     That is below the +0.05 the portfolio study says ML needs. → **observe-only.**
6. **A second line can be tracked for free: "coin in a 1-hour downtrend".** The coin's 1h EMA20 is at or below its EMA200.
   - Live stamps this since Sep-30.
   - Backtest: these fills lose −0.12 % and pass the filter bar.
   - Master agrees on direction (ex-washed −0.136 vs +0.168) but has too few fills to be sure. → **observe-only.**
7. **Sizing.**
   - Today's 1.5× UNMATCHED cell makes the backtest year worse, not better: 1.5× fills average −0.073 %, 1× fills −0.062 %.
   - Running ML at 1× everywhere cuts the yr5 annual loss from −$7.4k to −$5.8k and the drawdown from −$8.2k to −$6.6k (per $3k book).
   - On master it would have cost about $0.9k, but master's profit is in-sample.
8. **Recommendation, in the requested order:**
   - (a) No regime condition reaches "ship". Register two observe lines: **TREND_ALIGNED** and **PAIR_1H_DOWNTREND**.
   - (b) **Resize ML to 1× everywhere** (UNMATCHED 1.5× → 1×) as an explicitly declared risk-control override. Keep ML running so
     the observe lines get forward fills.
   - (c) **Pause only on a pre-committed trigger:** next 30 full-size ML fills across ≥ 10 days average < 0 %. The expected cost
     of collecting that evidence at 1× is about $260 on a $3k book.

---

## The checklist — what was run and what it found

| item | run? | how | result |
|---|---|---|---|
| ① every pair-level entry dimension | ✅ | `sweep_separators.py ML` (cited below) + own exhaustive scan: 38 pair variables (27 stamps + 11 kline rebuilds incl. a funding rebuild) × sign/median/terciles/quintile tails + all-pairs 2D median quadrants, on yr5 / yr5 ex-washed / master / master ex-washed; circular day-block shift null (2,000) over the full ~20k-mask scan and within the pair-1D family; OOS H1→H2; leave-one-month-out | **no survivor.** Pair-1D family best p_fwer 0.41 (yr5) / 0.25 (ex-washed). Closest: the coin's higher-TF trend (1h EMA20/200, 4h EMA20/50 below → −0.12…−0.15) → observe line |
| ② every MACRO/REGIME dimension, sign first | ✅ | 61 market variables (BTC 5m/1h/4h/1d slope, RSI, ADX, ATR / realised vol, trend gaps, 1d candle, off-24h / 7d / 30d high, eff72, breadth, global volume, regime label, ETH, alt index, dominance proxy, hour / day) at sign → median → terciles → quintiles, DAY units, with/without washed-out; null-calibrated count of "alive" variables; day-level ρ | **not refuted as a family:** 25/61 alive (null median 10, 95th 18, p 0.003); ex-washed 30/61 (null 11 / 21, p 0.007). Alive at SIGN: BTC 1h slope, hours since the 1h slope turned negative, alt median 24h return. No single variable survives the full-scan null |
| ③ uniform-degradation test | ✅ | yr5 cohort × month and half for cell, PVR band, crowd-sprint, pattern C/W, heat, chop, NEGFLANK, HOT_MATURE, regime, order type; two-way variance split with within-day label permutation; live batches BASE…B18 by cohort | **uniform:** 17 of 20 cohorts negative in BOTH halves, 0 positive in both. No cohort carries the loss. Chop and HOT are worse (p 0.04) but small; without them the rest is still −0.05 |
| ④ tape context, winning vs losing periods | ✅ | 9 yr5 months and 18 live batches vs BTC / ETH return, realised vol, % above the 1d EMA20, eff72, alt median return and up-share, dominance proxy, global volume, off-30d; prior-day tape at day level | ML tracks the tape: month ρ BTC return +0.65, off-30d +0.73, realised vol −0.72. **Even the best months are only ≈ 0** (Apr +0.02 with BTC +12 %; Aug −0.04 with BTC +25 %). Prior-day tape does not predict the day (all p > 0.1). Live batches: as-traded tracks BTC eff72 (ρ +0.69, n 11) |

---

## ① Pair-level dimensions

### 1a · `scripts/sweep_separators.py ML` (the CLAUDE.md-mandated sweep)

**Output:** `SWEEP ML: 41 dimensions x 87 tests | era A (<2026-06-30) vs B | consistent: 24`. 2D: ~2,520 quadrant cells, 131 consistent.

**Its pool is small.** It reads `SCREENED_BASELINE.csv`, which has only 37 ML fills (Jun-18 → Sep-18; 17 in era A). Its
"consistent" list is a screen with no null.

**Its leaders do not hold up:**
- **Funding rate** (sign Δ +0.42 / +0.49, the top survivor) rests on a stamp that is **not the coin's funding**:
  - Against the coin's own settled funding (cache) the correlation is r 0.04, with 65 % sign agreement.
  - ADA and PEPE carry the identical value −0.000925 at the same minute.
  - The engine reads ccxt `fundingRate`, cached for 8 h per symbol.
  - A rebuild from settled funding (`k_funding_last`) shows nothing on yr5 (every split |z| ≤ 1.05).
- **Re-read on yr5 with the sweep's own era-A thresholds,** only 5 of the 11 top 1D survivors keep their direction. That is a coin
  flip. The two strongest reverse:
  - BTC 1h RSI-prev high tercile: sweep −0.34 / −0.60, yr5 **+0.10**
  - global volume > median: sweep −0.45 / −0.34, yr5 **+0.06**

### 1b · Own exhaustive scan (pair + macro, 1D + all-pairs 2D) — `study_ml_checklist_scan.py`

Statistic: Welch z of zone vs rest. Null: the time-sorted outcome vector circularly shifted against fixed features by 5–95 % of
the sample, 2,000 shifts. This keeps every day's outcomes together. Minimum zone: 20 fills/seed and 10 days (yr5), 12 fills and
6 days (master).

| cohort | masks | max \|z\| | null 95th of max \|z\| | best p_fwer | survivors | masks beyond their own null 99th: observed vs null median / 95th (p) |
|---|---|---|---|---|---|---|
| yr5 all | 19,898 | 4.98 | 5.94 | 0.48 | **0** | 513 vs 156 / 492 (**p 0.043**) |
| yr5 ex-washed | 19,871 | 5.38 | 5.82 | 0.19 | **0** | 660 vs 146 / 460 (**p 0.018**) |
| master ex-B1 | 14,427 | 3.89 | 5.27 | 0.85 | **0** | 66 vs 42 / 320 (p 0.40) |
| master ex-washed | 14,301 | 4.69 | 5.21 | 0.31 | **0** | 135 vs 0 / 157 (p 0.08) |

**Family-only nulls (1D only):**

| family | yr5 all | yr5 ex-washed |
|---|---|---|
| pair-1D | best p_fwer 0.41 | 0.25 |
| macro-1D | best p_fwer 0.73 | 0.18 |

- **Read:** no single split is real at the scan level.
- **The count of moderately strong splits exceeds chance on yr5** (p 0.02–0.04). That is a broad, diffuse effect. The splits that
  carry it are almost all **"BTC 1h up × coin's higher-TF trend up"** (MACRO×PAIR family):
  - 1h EMA20/200 gap > 0.84 ∧ BTC 1h slope > 0.05: 159/seed · 69 % · **+0.072** vs −0.120, z 4.98, halves +0.17 / +0.22
  - 4h EMA20/50 gap > 0.61 ∧ BTC 1h slope > 0.05: +0.071 vs −0.116. Master: 45 fills, Δ **+0.06**
  - Mirror image, BTC 1h RSI-prev ≤ 52.7 ∧ BTC 1d EMA9/20 gap > 0.08: −0.206 vs −0.031

**OOS and stability:**
- **H1 → H2.** The H1-only scan's p<0.01 masks keep their sign on H2 at 41 % (all) and 57 % (ex-washed). The all-mask baselines are
  52 % and 57 %. So **H1 discovery does not replicate beyond chance.** The "trend-aligned" quadrant itself ranked #649 / #1,025 in H1
  (p_mask 0.04 / 0.02) and then got stronger in H2 (Δ +0.25 / +0.26 on 54–74 fills/seed).
- **Leave-one-month-out (top-30 full-year masks):** sign-stable 30/30, both halves 30/30. That is partly by construction, because
  they were selected on the full year.
- **Master direction agreement** (≥ 12 fills): 12/22 (all), 20/27 (ex-washed).

**Pair-level splits with the most consistent reads (none survives; listed for the observe line):**

| split | yr5 all: N/seed · avg vs rest | yr5 ex-washed | halves Δ | master ex-B1 / ex-washed |
|---|---|---|---|---|
| coin 1h EMA20 ≤ EMA200 (rebuild r 0.993 vs live stamp) | 262 · 56 % · −0.123 vs −0.023 | 152 · −0.138 vs −0.006 | −0.11 / −0.13 | 38 · +0.122 vs +0.213 / 23 · **−0.136** vs +0.168 |
| coin 4h EMA20 ≤ EMA50 (rebuild, no stamp) | 266 · 56 % · −0.124 vs −0.021 | 147 · −0.126 vs −0.015 | −0.14 / −0.11 | 38 · +0.153 vs +0.198 / 23 · −0.052 vs +0.142 |
| 24h volume vs 7d ≤ 0.69 (quintile) | 119 · −0.172 vs −0.041 | — | −0.21 / −0.06 | 17 · Δ +0.02 (master refutes) |

---

## ② Macro / regime dimensions (sign first, then buckets, DAY units, with and without washed-out)

**"ALIVE" criterion.** A variable is alive at a granularity if some bucket-vs-rest split meets all three:
1. |z| is above that split's own 95th percentile under the day-shift null.
2. Δ has the same sign in both yr5 halves.
3. Master does not contradict it (same sign, or fewer than 10 master fills = untestable).

**Refuted** = dead at every granularity in both yr5 cohorts. **Chance** = the same criterion on 300 shifted outcome vectors.

| | alive / 61 | null median / 95th | P(null ≥ obs) | alive at SIGN granularity |
|---|---|---|---|---|
| yr5 all vs master ex-B1 | **25** | 10 / 18 | **0.003** | BTC 1h slope (stamp + rebuild), hours since the 1h slope turned negative |
| yr5 ex-washed vs master ex-washed | **30** | 11 / 21 | **0.007** | the same + alt median 24h return |

- **Refuted at every granularity (24/61):** e.g. BTC 5m RSI, 1d slope, off-24h-low, hour, day of week, ETH 5m, global-volume
  extremes on yr5 all.
- **Full lists:** `study_ml_checklist_regime_verdict.csv`.
- **Untestable:** `entry_macro_trend` (constant BULLISH on every fill) and the regime label's CHOPPY_WEAK bucket (20 fills).

**Sign-granularity reads of the leading regime axes** (N/seed · avg · days · day-bootstrap P(mean<0)):

| variable | side | yr5 all | yr5 ex-washed | master ex-B1 | master ex-washed |
|---|---|---|---|---|---|
| BTC 1h EMA20 slope | ≤ 0 | 251 · −0.124 · 140d · 1.00 | 163 · −0.127 · 0.99 | 37 · +0.176 · 0.13 | 24 · −0.061 · 0.63 |
| | > 0 | 344 · −0.025 · 0.79 | 262 · −0.008 · 0.58 | 78 · +0.186 | 71 · +0.147 |
| BTC 24h return | ≤ 0 | 251 · −0.112 · 1.00 | 160 · −0.112 · 0.98 | 30 · +0.117 | 18 · −0.172 · 0.88 |
| | > 0 | 344 · −0.034 | 264 · −0.018 | 85 · +0.206 | 77 · +0.157 |
| BTC 4h slope | ≤ 0 | 300 · −0.098 · 0.99 | 181 · −0.083 · 0.94 | 42 · +0.092 | 23 · **−0.335 · 0.99** |
| | > 0 | 295 · −0.035 | 244 · −0.031 | 73 · +0.235 | 72 · +0.232 |
| ETH 24h return | ≤ 0 | 246 · −0.110 · 1.00 | 159 · −0.094 · 0.97 | 29 · +0.037 | 20 · −0.245 · 0.94 |
| | > 0 | 349 · −0.037 | 265 · −0.029 | 86 · +0.232 | 75 · +0.185 |
| alt median 24h return | ≤ 0 | 332 · −0.096 · 1.00 | 236 · −0.105 · 0.99 | 57 · +0.224 | 43 · +0.087 |
| | > 0 | 263 · −0.030 | 189 · **+0.011** | 58 · +0.142 | 52 · +0.101 |
| BTC − alt 24h (dominance proxy) | ≤ 0 (alts lead) | 176 · −0.128 · 0.99 | 114 · −0.085 · 0.89 | 31 · −0.091 · 0.72 | 26 · −0.167 · 0.83 |
| | > 0 | 419 · −0.041 | 311 · −0.042 | 84 · +0.284 | 69 · +0.193 |
| BTC above 1d EMA20 | no | 306 · −0.093 · 0.99 | 136 · −0.083 | 39 · **+0.351** (washed-out window) | 19 · +0.087 |
| | yes | 289 · −0.040 | 289 · −0.040 | 76 · +0.096 | 76 · +0.096 |

**Buckets (examples; all in `study_ml_checklist_regime_buckets.csv`):**
- **BTC 1h slope quintiles, yr5:** −0.13 / −0.13 / −0.01 / −0.08 / **+0.01**. Not monotonic in the middle, but a clear step at 0.
- **BTC eff72 quintiles, yr5 ex-washed:** −0.11 / −0.05 / −0.12 / −0.07 / **+0.06**.
- **Hours since the BTC 1h slope turned negative > 4.4:** 198/seed · **−0.146** · P 1.00 (master ex-washed 16 · −0.188).

**Day-level (one point per day, yr5, seeds pooled; fill-time variables):**
- yr5 all: 15 of 61 at p < 0.05 (3 by chance). Top: BTC 72h return ρ +0.16, BTC 4h slope +0.14, today's BTC candle +0.17.
- yr5 ex-washed: BTC eff72 ρ **+0.19** (Bonferroni-significant).

**Washed-out split.**
- On **master**, the window flips most regime reads. With the window, the "bad" side of BTC 1h / 4h / above-EMA20 is positive
  (washed-out bounces were bought in falling tapes and won 94 %). Without it, master agrees with yr5 on direction for 1h slope,
  24h return, 4h slope, ETH 24h and dominance.
- On **yr5**, removing it changes little.

**What item ② says.** The regime variable the checklist asks for exists and is measured: **the short-horizon market trend (1h → 3d)
at entry.** It explains why ML is worse in falling tapes. It does **not** turn ML into a winner. The good side is ≈ −0.03 … +0.06,
never confidently positive on its own.

---

## ③ Uniform-degradation test

**yr5 by month.** The sleeve is negative in **8 of 9 months**: Jan −0.10 · Feb −0.16 · Mar −0.09 · Apr +0.02 · May −0.02 ·
Jun −0.07 · Jul −0.06 · Aug −0.04 · Sep −0.03.

| cohort (yr5) | N/seed | avg | H1 / H2 | loss share in negative months (fill share) | master ex-B1 |
|---|---|---|---|---|---|
| UNMATCHED | 504 | −0.071 | −0.07 / −0.07 | 90 % (85 %) | 95 · +0.157 |
| NONEXP_CALM3D door | 84 | −0.024 | −0.08 / +0.01 | 6 % (14 %) | 20 · +0.306 |
| PVR quiet < 0.68 | 214 | −0.095 | −0.13 / −0.06 | 50 % (36 %) | 39 · +0.230 |
| PVR mid | 209 | −0.067 | −0.09 / −0.04 | 29 % (35 %) | 40 · +0.243 |
| PVR crowded ≥ 0.90 | 171 | −0.032 | +0.01 / −0.07 | 21 % (29 %) | 36 · +0.064 |
| crowd-sprint | 167 | −0.086 | +0.01 / −0.21 | 35 % (28 %) | 28 · +0.079 |
| pattern none / W / C | 513 / 51 / 30 | −0.073 / −0.064 / +0.025 | | | |
| heat 0 / heat 1 | 538 / 57 | −0.071 / −0.026 | | | |
| chop (eff72 ≤ 0.007) / trend | 78 / 517 | **−0.171** / −0.051 | −0.19 / −0.16 | 31 % (14 %) | 6 · −0.338 |
| NEGFLANK / not | 251 / 344 | **−0.124** / −0.025 | −0.12 / −0.13 | 79 % (43 %) | 37 · +0.176 / 78 · +0.186 |
| BTC_HOT_MATURE / not | 109 / 485 | **−0.157** / −0.047 | −0.15 / −0.16 | 39 % (18 %) | 23 · +0.147 |
| regime HEALTHY / STRONG | 419 / 169 | −0.076 / −0.035 | | | 75 · +0.095 / 36 · +0.429 |
| maker / taker fallback | 358 / 237 | −0.054 / −0.087 | | | |

- **17 of 20 cohorts are negative in BOTH halves; none is positive in both.**
- Month-to-month co-movement with the rest of the sleeve is weak (median r −0.23). The loss is a persistent level, not a shock
  in a few months.

**Two-way variance split** (fills; within-day label permutation, 300 runs):
- The cohort label is beyond chance only for **chop** (p 0.04) and **HOT_MATURE** (p 0.04).
- Month × cohort interactions are significant for PVR (0.02), crowd-sprint (0.02) and HOT (0.01). For example, crowd-sprint was
  fine in H1 and −0.21 in H2.
- Removing chop or HOT still leaves the rest negative (−0.051 / −0.047).
- **No cohort carries the loss → uniform degradation** → per the checklist, an unmeasured regime variable. Item ② measured it (the
  short-horizon trend). It explains part of the loss, not all of it.

**Live batches.**
- As-traded ML is negative in 11 of 17 batches with fills: BASE +0.15 · B1 −0.13 · B2 −0.09 · B3 +0.12 · B12 +0.07 · B13 −0.28 ·
  B14 −0.54 · B16 −0.18 · B17 +0.65 · B18 −0.70.
- The kept-set positives sit in BASE → B3, the in-sample era of today's filters.
- No live cohort is positive across batches since B6.
- Full grid: `study_ml_checklist_tables.md` §3d.

---

## ④ Tape context — winning vs losing periods

**yr5 months** (ML avg; BTC month return; daily realised vol; % of days above the 1d EMA20; alt up-share):

| month | ML | BTC | vol | above EMA20 | alt up-share | read |
|---|---|---|---|---|---|---|
| Feb | **−0.16** | −15 % | 3.4 | 0 % | 41 % | capitulation |
| Jan | −0.10 | −13 % | 2.1 | 55 % | 42 % | downtrend |
| Mar | −0.09 | +2 % | 2.6 | 47 % | 45 % | chop |
| Jun | −0.07 | −21 % | 2.8 | 0 % | 38 % | crash (incl. the washed-out bounce) |
| Jul | −0.06 | +7 % | 1.8 | 69 % | 42 % | recovery |
| Aug | −0.04 | **+25 %** | 1.9 | 69 % | 45 % | strong rally, ML still ≈ 0 |
| Sep | −0.03 | +7 % | 1.8 | 85 % | 48 % | steady up |
| May | −0.02 | −3 % | 1.6 | 44 % | 44 % | quiet |
| Apr | **+0.02** | +12 % | 1.9 | 87 % | 48 % | steady up, calm |

- **Across months:** ρ with BTC return +0.65, off-30d +0.73, realised vol −0.72, % above EMA20 +0.62, alt up-share +0.60.
- **ML's best tape is a calm, steady BTC uptrend with broad alt participation. Even there it only breaks even.** Its worst tape is a
  volatile BTC downtrend.
- **Live batches** (as traded, n = 11 with ≥ 5 fills): ρ with BTC eff72 +0.69, alt up-share +0.42, ETH return +0.45. The best live
  batches are B3 (BTC +22 %, eff72 0.24) and B17. The worst are B14 (alt median −2.1 %, eff72 0.04) and B18 (BTC −3.9 %, alt
  up-share 17 %).
- **Prior-day tape (known before the day) does not predict the day's ML result:** all 7 day-level ρ have p > 0.1 (232 days).
- **Any regime condition must use the state at entry, not a daily switch.**

---

## Candidates through the locked expectancy bar (BLOCK cohort judged)

**The bar:** WR < the sleeve's breakeven WR ∧ day-clustered P(mean<0) ≥ 0.95 ∧ ≥ 8 days ∧ N ≥ 15 ∧ no day or pair ≥ 50 % of the loss.

**Breakeven WR:** yr5 66.9 % (ex-washed 67.5 %) · master 61.3 % (ex-washed 62.4 %).

| block cohort | yr5 all | yr5 ex-washed | master ex-B1 | master ex-washed | origin / evidence |
|---|---|---|---|---|---|
| NEGFLANK (BTC 1h ≤ −0.05) | 251 · 59 % · −0.124 · P 1.00 · **PASS** | 163 · −0.127 · P 0.99 · **PASS** | 37 · 73 % · +0.176 · fail | 24 · 58 % · −0.061 · P 0.63 · fail | pre-registered 10-05 (yr5 = its discovery set) |
| BTC_HOT_MATURE | 109 · 51 % · −0.157 · **PASS** | 75 · −0.191 · **PASS** | 23 · +0.147 · fail | 17 · −0.018 · fail | pre-registered Jul-16 (yr5 genuinely OOS) |
| BTC 1h slope ≤ 0 (sign) | ≡ NEGFLANK on fills (the live dead-band removes −0.05…+0.025) | | | | |
| BTC 24h return ≤ 0 | 251 · −0.112 · **PASS** | 160 · −0.112 · **PASS** | 30 · +0.117 · fail | 18 · −0.172 · P 0.88 · fail | post-hoc |
| BTC 4h slope ≤ 0 | 300 · −0.098 · **PASS** | 181 · −0.083 · P 0.94 · fail | 42 · +0.092 · fail | 23 · 48 % · −0.335 · P 0.985 · **PASS** | post-hoc |
| **coin 1h EMA20 ≤ EMA200** | 262 · 56 % · −0.123 · **PASS** | 152 · −0.138 · **PASS** | 38 · +0.122 · fail | 23 · 52 % · −0.136 · P 0.78 · fail | blind spot named 10-08 (yr5 = discovery); **stamped live since Sep-30** |
| coin 4h EMA20 ≤ EMA50 | 266 · −0.124 · **PASS** | 147 · −0.126 · **PASS** | 38 · +0.153 · fail | 23 · −0.052 · fail | post-hoc, unvalidated rebuild |
| **NOT trend-aligned** = ¬(BTC 1h > 0 ∧ BTC 4h > 0 ∧ coin 4h EMA20 > EMA50) | 444 · 58 % · −0.116 · **PASS** (all seeds, both halves, 10/10 months below keep) | 288 · −0.114 · **PASS** (8/8) | 68 · 74 % · +0.147 · P 0.08 · fail | 48 · 63 % · −0.043 · P 0.64 · fail | post-hoc coarsening of the scan's top family |
| BTC eff72 ≤ 0.21 (not top quintile) | 476 · −0.083 · **PASS** | 330 · −0.087 · **PASS** | 84 · +0.151 · fail | 69 · +0.058 · fail | post-hoc |
| alts lead BTC 24h (dominance proxy ≤ 0) | 176 · −0.128 · **PASS** | 114 · −0.085 · P 0.90 · fail | 31 · −0.091 · P 0.72 · fail | 26 · −0.167 · P 0.83 · fail | post-hoc, unvalidated rebuild |

**Read.** The bar can only be met on yr5, which is the discovery set for every post-hoc line. Master (the only independent
sample) agrees on direction ex-washed in 7 of 8 lines, but its 17–48 fills never reach P ≥ 0.95. The one master pass (BTC 4h slope,
ex-washed) fails on yr5 ex-washed (P 0.94). Under the locked rules (master can only refute, cross-period = refute-only, no survivor
of the scan null) **nothing ships. Everything here is observe-grade.**

**The best candidate, the TREND_ALIGNED keep zone, in detail:**

| | keep zone | block zone | $ per $3k book (1×, keep-only vs all) |
|---|---|---|---|
| yr5 all | **150/seed · 71 % · +0.079** [−0.00, +0.16], P(>0) 0.97, 102 days; seeds +0.074 / +0.091 / +0.073; halves +0.05 / +0.11; months: 8 of 10 > 0 | 444 · −0.116 | **+$1,735** vs −$5,822 / seed (keep-zone DD −$545) |
| yr5 ex-washed | 137 · +0.073, P(>0) 0.95 | 288 · −0.114 | |
| master ex-B1 | 47 · 77 % · +0.235 [+0.04, +0.49], 24 days | 68 · +0.147 | +$1,614 vs +$3,073 (it would have blocked the washed-out winners) |
| master ex-washed | 47 · +0.235 | 48 · −0.043 | +$1,614 vs +$1,315 |

- **Volume:** the zone keeps only 25 % of ML fills (150 of 595/seed).
- **Pair concentration of the keep zone's $:** ENA 23 %, SUI 16 %, AAVE 16 %.
- **Haircut:** the uplift over the sleeve is +0.146 (−0.067 → +0.079). After a 30–50 % haircut, the gated sleeve projects to
  **≈ +0.006 … +0.035 %/trade**. That is below the portfolio's +0.05 hurdle.
- **Washed-out:** master's washed-out window sits entirely in the block zone (BTC 4h falling), so a live rule would need the scout's
  washed-out exemption. That is another leg, and it is post-hoc.

---

## Sizing angle (pure risk control — not a verdict)

ML-only $, fixed $3,000 book, mean of 3 yr5 seeds:

| sizing | yr5 Σ$ / yr | max drawdown | worst month | H1 / H2 | master ex-B1 Σ$ (DD) | master ex-washed |
|---|---|---|---|---|---|---|
| as-run 2× (UNMATCHED / CALM3D 2×; master = as traded) | −$9,081 | −$10,317 | −$3,643 | −$7,074 / −$2,007 | +$6,043 (−$1,255) | +$2,526 |
| **TODAY** (UNMATCHED 1.5×, sprint / PVR de-mux 1×, CALM3D 1×) | **−$7,353** | **−$8,217** | −$2,765 | −$5,167 / −$2,186 | +$4,010 (−$1,089) | +$1,541 |
| **1× everywhere** | **−$5,822** | **−$6,597** | −$2,015 | −$3,536 / −$2,286 | +$3,073 (−$893) | +$1,315 |
| today × 0.5 | −$3,677 | −$4,108 | −$1,383 | | | |
| pause (0) | $0 | $0 | | | $0 | |

**yr5 by today's size:**
- 1.5× fills (289/seed): **−0.073 %**
- 1× fills (306/seed): −0.062 %

So the 1.5× cell picks no better trades in the backtest. On master, 1.5× fills are +0.242 (53) and 1× fills +0.132 (62), but that
is in-sample.

**UNMATCHED cell gate (DECISION_LOG 206, "next 15 UNMATCHED fills: WR ≥ 70 % ∧ $ > 0 → 2× · $ < 0 → 1× · else stay").** Fills on
or after 10-05: FET, LIT, NIL, ORCA (B17) and ARB, UNI, LIT (B18) = **7 fills · 57 % · +$456 → "stay" so far (7/15).** The gate has
not fired. Moving to 1× now would be a declared override, not the gate.

**Going below 1×** (e.g. 0.5×) halves the bleed again. But it makes ML fills probe-class, which full-size tables exclude (memory:
full-size-only tables). That would starve the observe lines. **Not recommended.**

---

## Recommendation (operator's order of preference)

**(a) A regime / cohort condition where ML pays — exists, but only at OBSERVE-ONLY evidence.**
- Register two frozen scout observe lines. Judge them on fresh full-size fills only, with the locked expectancy bar (≥ 15 fills,
  ≥ 8 days, window-clustered P ≥ 0.95, no day/pair ≥ 50 %).
  - **PAIR_1H_DOWNTREND** (block side): `entry_pair_1h_ema20_200_gap_pct ≤ 0`, washed-out exempt.
    - Free: the stamp is live since Sep-30.
    - Forward so far (stamped fills B15 → B18): 8 fills · 38 % · ≈ −0.03 % over ~4 days. B18's 3 = 1 window.
  - **TREND_ALIGNED** (keep side): BTC 1h EMA20 slope > 0 ∧ BTC 4h EMA20 slope > 0 ∧ coin 4h EMA20 > EMA50.
    - Needs two new stamps (BTC 4h slope, coin 4h EMA20/50 gap), or a review-time rebuild like this one.
    - Pass = the complement fails the bar AND the keep zone ≥ +0.05 on ≥ 15 fills / ≥ 8 days.
- Keep the existing NEGFLANK and BTC_HOT_MATURE observe lines. This study re-confirms them: yr5 PASS, master FAIL.
- If either line later passes and is armed, the pre-committed revert is: the first 10 refused signals, re-priced with the live
  exit, show WR ≥ 61 % or Σ > 0 → revert.

**(b) Resize — the actionable step now (operator-declared risk override, NOT a gate verdict).**
- **ML to 1× everywhere: `pattern_cell_rules` UNMATCHED LONG `inv_mult` 1.5 → 1.0, and `long_unmatched_quiet_mult` 1.5 → 1.0 with it** (`trading_config.json`; a config change for the operator to approve, not made here).
- Evidence:
  - yr5: the 1.5× cell's fills are no better than 1× fills (−0.073 vs −0.062). 1× cuts the yr5 annual loss by $1.5k and the
    drawdown by $1.6k per $3k book.
  - Master: the cost is ≈ $0.9k, on in-sample profit.
- Transparency: the cell's own 15-fill gate is at 7/15 and has not fired. This is a discipline override and should be logged as one.
- Its revert: if the next 15 UNMATCHED fills at 1× show WR ≥ 70 % ∧ $ > 0, go back to 1.5×.
- Do **not** go below 1×.

**(c) Pause — last resort, on a pre-committed trigger, not now.**
- All four checklist items have been run (table above), so a pause proposal is now permitted.
- Proposed trigger: **pause ML if the next 30 full-size ML fills (≥ 10 distinct days, 1×) average < 0 %.**
- Expected cost of collecting that evidence: 30 × −0.06 % × 4.875 × $3k ≈ **−$260**. That is cheap next to the −$5.8k/yr the yr5
  backtest projects if ML runs unchanged and the backtest is right.
- At the trigger, the choice is pause vs "ML only in the TREND_ALIGNED zone", whichever observe line has the better forward read
  by then.
- If the operator prefers to stop the bleed immediately, a pause today is defensible on the yr5 evidence (CI excludes 0). It is the
  operator's call.

---

## Blind spots (what this study could NOT test, or tested weakly)

**Master power and coverage:**
- Master power is tiny: 115 fills ex-B1, 95 ex-washed. Master can refute, not confirm. A true Δ of −0.2 % is usually undetectable
  there.
- **Master features exist only for KEPT fills.** The 83 live fills today's filters remove, and BASE/B1 as-traded fills, have no
  rebuilt features. Item ③'s as-traded batch view uses % only.

**yr5 fidelity and the in-sample problem:**
- The replay reproduces ~42 % of live kept fills per seed (scan phase). Its ML population differs from live's.
- It ends Oct-02, so B17/B18 tape is not in it. Seeds are replicates, not independent years.
- **yr5 is the discovery data** for every post-hoc line here, and for NEGFLANK. Only BTC_HOT_MATURE is genuinely OOS on yr5.
- H1 → H2 discovery did not replicate beyond chance overall.

**Rebuilds and stamps:**
- **Unvalidated rebuilds (no stamp exists):** coin 4h EMA20/50 gap, BTC 4h / 1d slopes and gaps, BTC vs 1d EMA20, ETH 1h slope,
  alt index (equal-weight median over 583 cached symbols: survivorship and listing drift), dominance proxy (its mean is positive
  every month, so the proxy is biased), realised vol, hours-since-slope-negative.
- **Funding:** the live `entry_funding_rate` stamp does not match the coin's settled funding (r 0.04). `sweep_separators`' top ML
  survivor rests on it. The settled-funding rebuild shows nothing, but the predicted rate at entry (what the stamp intends) is not
  in any cache. Worth an engine-side check of the stamp.

**Granularity not tested:**
- 2D splits are median quadrants only. No 2D terciles, no exhaustive 3-variable cells (only the 3 post-hoc composites above), no
  trees, no continuous threshold fitting. All left out on purpose (over-fitting).
- Hour of day is a median split only.

**Small or missing cohorts:**
- `entry_macro_trend` is constant.
- CHOPPY_WEAK (20 fills) and the ADX_SURGE_OPEN cell (7/seed) are too small for item ③.
- Pattern C cells are 30/seed.

**Not stamped on yr5:** market cap / CMC rank, order book, news, sector, the live predicted funding rate.

**Washed-out definition** is Jun-18 → Jul-2 ∪ off30 ≤ −15. On yr5 that also removes Feb and June capitulation days (510 fills).
Results were shown both ways.

**Nulls:** the circular day-shift null assumes stationarity. Welch z is heavy-tailed under clustering, which the null calibrates for
the maximum and the count but not perfectly per mask. The "count beyond null 99th" on master ex-washed is degenerate (null median
0, small N, float16 ties), so it should be ignored.

**Item ④ granularity:** month (n = 9) and batch (n = 11) correlations are descriptive only. Day-level tape used prior-day variables
only. The at-entry state is covered by item ②.

**Not modelled:** exit side (no stop / cap counterfactuals; the B18 study found ML losers die straight to the stop), real-money
slippage (B4 only), funding paid over the hold, cross-sleeve slot interactions (FRENZY / fades / bear-run open at the same time).
