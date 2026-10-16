# Momentum LONG: can we tell a real market-wide move from a BTC spike? (2026-10-15)

- **Question (operator):** "measure the STRENGTH of BTC rising at entry to estimate if it will continue for 15 minutes or it's just a spike — with breadth increasing, or with ETH, SOL, ZEC, XRP, or top-10 analysis — to see if the whole market turned bullish."
- **New angle vs the earlier studies:** they tested market variables one at a time (`ML_BTCFWD_FOLLOWUP_2026-10-15.md`). This study tests *conjunctions*: the quality of BTC's rise (steady climb vs one-bar spike) and a 0–9 **market impulse score** that counts how many parts of the market agree.
- **Builds on:** `ML_WINNERS_LOSERS_2026-10-15.md` (BTC's next 15 min decides the trade), `ML_DELAY_BTC_CONFIRM_2026-10-15.md` (waiting is worse), `ML_BTCFWD_FOLLOWUP_2026-10-15.md` (single features: 0.3–1.5 bps, useless at fills).
- **Pre-registration:** `reports/ML_IMPULSE_PREREG.txt`, written 18:39 UTC before anything was scored; the script refuses to overwrite it. 46 Step-1 tests + 26 Step-2 cells = 72, all counted.
- **Safety:** nothing shipped. No engine / config / `trading_config.json` / template / test edits, no git, no bot API, nothing written to `reports/backtest_cache/`.
- **Validation:** `study_ml_impulse_validate.py` ALL STUDY CHECKS PASS · `validate_against_master.py` ALL CHECKS PASS · no-lookahead 300/300 + 300/300 identical.

---

## Plain-English summary

1. **Can we tell a real market-wide move from a BTC spike? We can *describe* it — but it does not tell us what happens next.**
   - We scored every 5-minute moment of 2025 + 2026 (188,000 moments) on: how steadily BTC rose (efficiency, share of up-minutes, whether one minute carried the move, volume burst vs rising volume, above its 1h VWAP), whether ETH / SOL / XRP / ZEC were up, whether ≥ 70 % of the top-10 coins were up, whether breadth was rising, top-10 taker buying, and a market volume surge — the 0–9 score — plus how tightly the top-10 moved together.
   - **When BTC is already up ≥ 0.10 % over 15 min, the chance it keeps rising over the next 15 min is 47–49 % whatever the market looks like** — "real move", "spike", score 0–2 or score 7–9. All the differences are within ±3 bps and their confidence intervals straddle zero.
   - On all moments, the score has a hair of predictive power, but in the **opposite** direction to the idea: a "whole market bullish" moment is followed by a *slightly weaker* 15 minutes (−0.3 to −0.7 bps for BTC and alts, same sign in both years). That is the same small give-back after a rise the earlier study found. It is far below the 2 bps usefulness bar. **0 of 46 Step-1 tests pass.**
2. **Does it help momentum longs? No.**
   - The same score, conjunction ("BTC up AND top-10 ≥ 70 % up AND breadth rising"), spike flag and the 12 spike-vs-trend features were tried as filters on the three blocks with today's exit stack: **0 of 26 cells pass** (15 "one-year-only", 11 fail; none was even eligible, since nothing passed Step 1).
   - **The direction flips between periods.** In the 2025 and 2026-H1 backtests, high-score entries are a bit *worse* (refused − kept −0.04 to −0.11 %/fill). On the real master fills (Jun–Oct 2026) they are clearly *better*: score ≥ 7 → 83 fills, WR 76 %, +0.23 %/fill vs −0.10 for the other 108. The 2026-H2 backtest (same months as master) is flat. A filter that is right in one period and wrong in the next is not a filter — it is the same "regime question" the earlier studies already pointed to.
   - **Closest miss (not a candidate, not registered):** "BTC's last-15-min volume was evenly spread (no burst)" — refusing those entries gains in every block (+0.015 to +0.038 %/signal), but the refused fills are *not* worse than the kept ones in 2025 (+0.005), the 30-minute version of the same feature fails, the master loss sits in Sep–Oct only (2 monthly windows; Jun–Jul were winners), and it would have refused the live PEPE winner.
3. **The live PEPE winner (+1.16 %):** score **5 of 9** — ETH, SOL, XRP, ZEC up and 70 % of the top-10 up, but BTC's rise was weak/choppy (efficiency 0.06), breadth was flat-to-falling, taker buying below 0.5, no volume surge. Not the conjunction, not a spike. BTC then rose +0.46 % in the next 15 min. Several "exploratory" filters (score ≤ 5, enter-only-on-conjunction, the volume-burst zone) would have refused it.
4. **Bottom line:** the market's "quality of move" at the moment we enter does not forecast BTC's or the alt market's next 15 minutes, and does not separate momentum-long winners from losers consistently across periods. **No filter, no Scout line, no config change.** The open lever stays the CURRENT_STATE momentum-long on/off gate (30 fills at 1×). Four studies now agree: the post-entry BTC path decides the trade, and nothing measurable at entry — single features or conjunctions — forecasts it.

---

## 1. Data, definitions, fidelity

- **Grid:** every 5m boundary 2025-01-01 → 2026-10-15 15:55 UTC (187,968 moments; 187,954 with every item + both targets). Coverage of every feature ≈ 100 %.
- **Targets:** `Y_btc` = BTC close 15 min ahead / close at t (identical to the prior study). `Y_alt` = equal-weight mean 15-min forward return of the top-50 (previous-day list, ex-BTC/stables), ≥ 25 members.
- **Data:** BTC/ETH/SOL 1m with taker-buy and top-50 5m with taker-buy reused from `reports/ml_btcfwd_fetch/`. **Fetched now** (budget rule, ≤ 950 used-weight/min read from `x-mbx-used-weight-1m`, no 418/429, cooldown file never written): XRPUSDT + ZECUSDT 1m with taker-buy 2024-12-29 → now (2 × 1,893 requests); live-window top-up (top-10 1m and top-50 5m from 2026-10-11 22:00, 264 requests). Stored in `reports/ml_impulse_fetch/`. Top-10 1m closes for co-movement: the cache `k1m` (6,518 of 6,520 member-days present).
- **Frozen from features only, before the first test month (2025-01..03):** `q_er` = 0.218 (median 15-min efficiency of rising BTC moments), spike 80th pct = 0.393 (`reports/ML_IMPULSE_frozen_params.json`).
- **Score legs (1 each):** L1 BTC up over 15 min with efficiency ≥ 0.218 · L2–L5 ETH / SOL / XRP / ZEC up over 15 min · L6 ≥ 70 % of top-10 up over 15 min · L7 top-50 trend breadth (share above their 60-min-ago price) higher than 15 min ago · L8 top-10 taker-buy share > 0.5 over 15 min · L9 top-50 15-min volume ≥ 1.5 × its 24 h average.
- **Fills:** the 4,501 momentum-long rows of `ML_BTCFWD_exit_rows.csv` (replay MOM-long 2025 / 2026-H1 / 2026-H2 from `ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv`, master MOMENTUM LONG closed non-probe, live via `export_glob.py`). P&L at 1× two ways: replica `base` (today's stack, `ml_exit_optimize_yr5.py` BASE) and engine `pct_new` / as-traded.

| block | N | days | replica avg % | engine / as-traded % | replica within ±0.05 | sign agree |
|---|---|---|---|---|---|---|
| ① live | 1 | 1 | +1.098 | +1.157 | — | 100 % |
| ② master | 191 | 77 | +0.045 | +0.027 | **82.7 %** (below 85 % gate — caveat) | 97.9 % |
| ② master stack_keep | 139 | 66 | +0.153 | +0.141 (stack_pct) | — | — |
| ③ 2025 | 2,617 | 301 | −0.199 | −0.187 | 85.1 % | 97.7 % |
| ③ 2026-H1 | 1,171 | 154 | −0.085 | −0.085 | 94.7 % | 99.6 % |
| ③ 2026-H2 | 521 | 79 | −0.013 | −0.015 | 97.3 % | 99.8 % |

Pooled replay parity 89.2 % (gate ≥ 85 %: PASS).

## 2. Step 1 — do the items predict the next 15 minutes? (all moments, walk-forward 2025-04 → 2026-10, 200 day-circular-shift null)

Spread = OOS mean in the training-oriented top minus bottom (quintile for continuous, TRUE − FALSE for legs/CONJ, score ≥ 7 vs ≤ 2), in bps. Orientation −1 = "higher value → weaker next 15 min". Bar: (a) beyond null 99th pct, (b) ≥ 2 bps in both years, same orientation, (c) score monotone.

| item | target | orient 25/26 | OOS AUC (null 99th) | spread 2025 / 2026 bps | pass |
|---|---|---|---|---|---|
| er_15 (BTC efficiency) | BTC / alt | − / − | 0.525 (0.504) / 0.517 (0.504) | 0.95 / 0.46 · 1.72 / 0.81 | fail (b) |
| er_30 | BTC / alt | − / − | 0.528 / 0.517 | 0.55 / 1.04 · 1.29 / 1.69 | fail (b) |
| upsh_15 / upsh_30 | BTC / alt | − / − | 0.516–0.517 / 0.513–0.515 | ≤ 0.9 · 0.3–2.1 | fail (b) |
| spike_15 / spike_30 | BTC / alt | + / mixed | 0.504–0.509 | −0.1 … 1.7 | fail (b) |
| volconc_15 / 30, volslope_15 / 30 | both | mixed | 0.498–0.503 (≤ null) | −0.96 … 0.82 | fail (a, b) |
| vwap1h | BTC / alt | − / − | 0.527 / 0.518 | 1.03 / 0.85 · **2.69 / 1.45** | fail (b, 2026) |
| corr15 (top-10 co-movement) | BTC / alt | + / − | 0.504 / 0.501 | 1.34 / 1.84 · −1.19 / −1.11 | fail (b) |
| L1–L9 legs | both | mostly − | 0.499–0.516 | −0.6 … 1.25 | fail |
| **SCORE (0–9)** | BTC | − / − | 0.519 (0.504) | 0.29 / 0.44 | fail (b, c) |
| **SCORE (0–9)** | alt | − / − | 0.515 (0.503) | 0.43 / 0.70 | fail (b, c) |
| **CONJ** | BTC / alt | flips | 0.504 / 0.500 | 0.15 / −0.01 · −0.26 / 0.00 | fail |

**0 / 46 pass.** Full table: `ML_IMPULSE_step1.txt` / `.csv`.

**Score dose-response (OOS, mean next-15-min return in bps; P = P(BTC up), P(BTC ≥ +0.10 %), P(BTC ≤ −0.10 %)):**

| score | BTC 2025 | BTC 2026 | alt 2025 | alt 2026 | P up / ≥+0.10 / ≤−0.10 (BTC 2026) |
|---|---|---|---|---|---|
| ≤ 2 | +0.28 | +0.23 | +0.65 | +0.06 | 0.515 / 0.255 / 0.235 |
| 3 | +0.25 | −0.28 | +0.18 | −0.29 | 0.494 / 0.229 / 0.232 |
| 4 | −0.24 | +0.05 | −0.51 | −0.48 | 0.503 / 0.224 / 0.230 |
| 5 | −0.31 | −0.54 | −1.43 | −0.49 | 0.471 / 0.219 / 0.241 |
| 6 | −0.61 | +0.18 | −1.83 | −0.32 | 0.492 / 0.226 / 0.232 |
| ≥ 7 | +0.01 | −0.23 | +0.25 | −0.67 | 0.473 / 0.234 / 0.261 |

Not monotone (Spearman of bucket means −0.26 to −0.83), and the whole range is ~1–2 bps against a 24 bps standard deviation. High scores raise *both* tails (more ≥ +0.10 and more ≤ −0.10 moves) — volatility, not direction.

## 3. The operator's literal question (exploratory, flagged): BTC already rising — continuation or spike?

`ML_IMPULSE_conditional.txt`. Moments with BTC r15 ≥ +0.10 % (46,483). Day-bootstrap 95 % CI.

| year | group | N | P(BTC up next 15m) | P(≥ +0.10) | BTC next 15m bps | alt next 15m bps |
|---|---|---|---|---|---|---|
| 2025 | SPIKE (one bar carried it) | 497 | 0.487 | 0.336 | +0.76 | +2.16 |
| 2025 | TREND (steady climb) | 20,153 | 0.476 | 0.261 | −0.28 | −0.85 |
| 2025 | CONJ (whole market) | 16,510 | 0.475 | 0.271 | −0.09 | −0.53 |
| 2025 | score 7–9 | 18,145 | 0.476 | 0.269 | −0.02 | −0.01 |
| 2026 | SPIKE | 528 | 0.483 | 0.312 | −0.17 | −1.34 |
| 2026 | TREND | 14,559 | 0.470 | 0.251 | −0.30 | −0.94 |
| 2026 | CONJ | 11,814 | 0.475 | 0.264 | −0.08 | −0.42 |
| 2026 | score 7–9 | 14,014 | 0.467 | 0.259 | −0.31 | −0.86 |

- TREND − SPIKE: BTC −1.04 bps [−4.3, +2.2] (2025), −0.14 [−3.0, +2.8] (2026); alt −3.0 [−8.4, +3.0], +0.4 [−3.4, +3.9].
- CONJ − not: BTC +0.24 [−0.5, +1.0] / +0.58 [−0.3, +1.5]; alt −0.48 [−2.1, +1.0] / +1.40 [+0.2, +2.7] (the only interval excluding 0 — 2026 only, 1.4 bps, sign flips vs 2025).
- At ≥ +0.20 % the picture is the same (all CIs straddle 0).
- **On momentum-long fills entered while BTC was up ≥ 0.10 %:** 2025 TREND −0.22 vs all −0.23; 2026 CONJ −0.04 vs all −0.04; master CONJ +0.26 vs not +0.19 (59 vs 31 fills). No separation in the backtest; a small one on master only.

## 4. Step 2 — as a momentum-long filter (26 cells, all exploratory)

Δ %/signal (refused = 0), replica [engine]; P(Δ > 0) day bootstrap. Separation = refused − kept avg % [P(refused worse)]. Full tables (expectancy bar, two-sided, LOMO, month/day shares): `ML_IMPULSE_step2.txt`.

| cell | 2025 Δ | 2026 Δ | master Δ | master_keep Δ | sep 2025 | sep 2026-H1 | sep 2026-H2 | sep master | verdict |
|---|---|---|---|---|---|---|---|---|---|
| SCORE refuse ≤ 3 | +0.045 [+0.042] P1.00 | +0.009 P0.81 | +0.034 P0.90 | +0.016 | +0.042 | +0.094 | −0.091 | **−0.228 [0.98]** | one-year-only |
| SCORE refuse ≤ 6 | +0.116 [+0.109] | +0.041 P0.99 | +0.054 | −0.004 | +0.050 | −0.016 | −0.026 | **−0.324 [1.00]** | fail |
| SCORE refuse ≥ 5 | +0.141 [+0.132] | +0.056 P0.99 | **−0.084** | **−0.161** | −0.073 [0.96] | −0.112 [0.96] | +0.034 | +0.239 | fail |
| SCORE refuse ≥ 7 | +0.083 | +0.023 | **−0.099** | **−0.149** | −0.050 | +0.016 | +0.026 | +0.324 | one-year-only |
| CONJ enter-only | +0.119 [+0.113] | +0.041 P0.97 | +0.034 | **−0.044** | +0.057 | +0.041 | −0.118 | −0.268 [0.99] | fail |
| CONJ refuse | +0.080 | +0.022 | **−0.079** | **−0.109** | −0.057 | −0.041 | +0.118 | +0.268 | fail |
| SPIKE refuse | +0.012 | −0.001 | −0.004 | −0.006 | −0.092 | +0.118 | −0.004 | +0.080 | one-year-only |
| Z1 er_15 (steadiest BTC climbs) | +0.082 | +0.023 | −0.073 | −0.123 | −0.029 | −0.048 | +0.156 | +0.228 | one-year-only |
| Z1 vwap1h | +0.131 | +0.040 | −0.045 | −0.106 | −0.047 | −0.040 | +0.080 | +0.089 | fail |
| **Z1 volconc_15** (no volume burst) | +0.038 [+0.039] P1.00 | +0.018 P1.00 | +0.035 P0.95 | +0.006 | **+0.005** | −0.094 | −0.115 | −0.321 [0.99] | fail (G3) |
| Z1 volconc_30 (same idea, 30 min) | +0.040 | −0.001 | −0.012 | −0.033 | −0.064 | +0.025 | +0.246 | +0.058 | one-year-only |
| Z1 corr15 | +0.058 | −0.007 | +0.002 | +0.005 | −0.062 | +0.364 | +0.353 | −0.106 | one-year-only |

- **Why the 2025 Δ is positive for almost everything:** the sleeve loses −0.20 %/fill in 2025, so refusing *any* group "helps". The separation columns are the honest test: they flip sign between 2025/H1 and H2/master for the score, CONJ and the BTC-quality zones.
- **Expectancy bar (refused cohort, 1×):** passes in 2025 for every cell (sleeve breakeven WR 68 %, refused WR 46–53 %); in 2026 only for SCORE ≤ 6, SCORE ≥ 4/5, CONJ enter-only, upsh_30, spike_30, volconc_15; on master **only volconc_15** (N 29, WR 48 % vs BE 62 %, P(avg < 0) 0.95 day / 0.89 month, 21 days). No cell passes the bar in all three.
- **Master high-score effect, by month** (score ≥ 7 vs rest, replica %): Jun +0.80 / +0.43 · Jul +0.46 / −0.02 · Aug +0.09 / −0.00 · Sep +0.14 / −0.40 · Oct −0.22 / −0.18. In the 2025 backtest the high-score group is worse in 10 of 13 months. Same months in the 2026-H2 replay: flat. Regime- (or fill-mix-) dependent, not a rule.
- **volconc_15 closest miss, by month (master zone sum %):** Jun +0.38 · Jul +1.81 · Aug −0.99 · Sep −3.77 · Oct −4.02 → the master loss is 2 monthly windows (WINDOW-UNITS rule); 2025 zone not worse than the rest; the 30-min twin fails; Step 1 failed (AUC below the null). Not registered.
- **Scan-lag sensitivity** (`ML_IMPULSE_step2_lag5.txt`): the engine's per-scan stamps see the *previous* 5m bar for 59 % of master fills (V2). Recomputing every 5m feature one bar earlier: 0 candidates (16 one-year-only, 10 fail); the master score separation weakens (refuse ≤ 3: −0.28; refuse ≤ 6: −0.05) and CONJ's master separation vanishes (+0.04).

**Live PEPE (2026-10-15 14:36 UTC, +1.16 %):** SCORE 5 (L2–L6 true; L1 false — BTC r15 +0.07 %, efficiency 0.06; L7 false — breadth −0.02; L8 false — taker 0.47; L9 false — volume 1.15×), CONJ 0, SPIKE 0, corr15 0.66, BTC next 15 min +0.46 %. Universe list stale (last list 2026-10-11, flagged). Exploratory cells that would have refused it: SCORE ≤ 5, ≤ 6, ≥ 4, ≥ 5, CONJ enter-only, Z1 volconc_15, Z1 er_30, Z1 vwap1h, Z1 corr15.

## 5. Recommendation

- **No filter, no exit change, no config change, no Scout line, no new watchlist item.** 0 / 46 Step-1 tests and 0 / 26 Step-2 cells pass. Nothing was eligible.
- What the engine would have to stamp at entry, *if* this were ever revisited: BTC 1m closes + volumes of the last 30 min; ETH / SOL / XRP / ZEC 15-min returns; top-10 15-min share up and taker-buy share; top-50 60-min trend breadth now and 15 min ago; top-50 15-min vs 24 h volume; top-10 1m returns for co-movement. None is worth stamping on this evidence.
- The open lever stays the CURRENT_STATE momentum-long on/off gate (30 fills at 1×).

## 6. Multiplicity

| arm | tests / cells | expected lucky passes | seen |
|---|---|---|---|
| Step 1 (99th-pct null, both-years 2 bps bar) | 46 | ≈ 0.5 on (a) alone | 0 full passes (19 pass (a) — the known small give-back) |
| Step 2 | 26 exploratory cells | ~1 at the 0.95 level per gate | 0 pass all gates |

## 7. Could NOT test

- **Funding / open interest / liquidations / order book:** not cached; no free historical source (as in the prior study).
- **Taker side below 1 minute:** aggTrades cache has no buyer-maker flag; taker shares come from 1m (BTC/ETH/SOL/XRP/ZEC) and 5m (top-50) klines.
- **Universe after 2026-10-11:** the last list is reused for 10-12 → 10-15 (stale, flagged); only the live fill is affected.
- **Engine scan timing:** live stamps lag a just-closed bar by seconds (V2); handled by the lag sensitivity, not by modelling per-scan timing.
- **Blocked signals:** signals today's stack refuses are not in the replay, so the score is read on admitted signals only.
- **Master parity 82.7 % (< 85 %):** master columns carry that caveat; as-traded / engine numbers shown alongside.
- **Live block:** N = 1.
- **Slot / concurrency effects of refusals:** not modelled (refused = 0).

## 8. Exact commands

```
venv/bin/python scripts/study_ml_impulse_prereg.py                              # write-once pre-registration (refuses overwrite)
venv/bin/python -W ignore scripts/study_ml_impulse_fetch.py phase1 XRPUSDT      # (and ZECUSDT; ~10 min each at the budget)
venv/bin/python -W ignore scripts/study_ml_impulse_fetch.py live                # live-window top-up (264 requests)
venv/bin/python -W ignore scripts/study_ml_impulse_features.py                  # → ML_IMPULSE_grid5m.csv.gz, ML_IMPULSE_grid1m.csv.gz, frozen params
venv/bin/python -W ignore scripts/study_ml_impulse_features.py --assert         # no-lookahead 300/300 (1m) + 300/300 (5m)
venv/bin/python -W ignore scripts/study_ml_impulse_step1.py --shifts 200 > reports/ML_IMPULSE_step1.txt
venv/bin/python -W ignore scripts/study_ml_impulse_step2.py > reports/ML_IMPULSE_step2.txt
venv/bin/python -W ignore scripts/study_ml_impulse_step2.py --lag5 > reports/ML_IMPULSE_step2_lag5.txt
venv/bin/python -W ignore scripts/study_ml_impulse_conditional.py > reports/ML_IMPULSE_conditional.txt
venv/bin/python -W ignore scripts/study_ml_impulse_validate.py                  # V1–V6 → ALL STUDY CHECKS PASS
venv/bin/python scripts/validate_against_master.py                              # ALL CHECKS PASS
```

**Files:** `reports/ML_IMPULSE_PREREG.txt` · `ML_IMPULSE_frozen_params.json` · `ML_IMPULSE_grid5m.csv.gz` · `ML_IMPULSE_grid1m.csv.gz` · `ML_IMPULSE_step1.{txt,csv}` · `ML_IMPULSE_dose.csv` · `ML_IMPULSE_step2.txt` · `ML_IMPULSE_step2_cells.csv` · `ML_IMPULSE_step2_lag5.txt` · `ML_IMPULSE_step2_cells_lag5.csv` · `ML_IMPULSE_fill_features{,_lag5}.csv` · `ML_IMPULSE_conditional.txt` · `reports/ml_impulse_fetch/` (XRP/ZEC 1m, live top-ups, logs).
**Scripts:** `scripts/study_ml_impulse_{prereg,fetch,features,step1,step2,conditional,validate}.py`.
