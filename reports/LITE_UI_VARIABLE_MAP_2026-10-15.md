# LITE: every dashboard table / variable mapped to where it was tested (2026-10-15)

Research only. No engine, config, template or test file was changed. Nothing committed, no bot API call, no network (cache only).
This is a review addition to `reports/LITE_WL_FULLSWEEP_2026-10-15.md` (the "full sweep"). Scratch scripts and outputs are outside the repo.
Not reviewed yet, so it supports no arm and no ship.

## Summary

- **Every dashboard table that can show a LITE fill was mapped to the Order columns behind it** (about 100 tables in `templates/index.html`, fed by `/api/performance` in `main.py`).
  - 125 Order columns are referenced by the performance code.
  - Every `entry_*` column the UI reads was either in the full sweep's 186 testable variables, excluded by a pre-declared rule (price, fee, notional, strategy, order type), or is empty on every LITE fill.
- **What the full sweep did NOT test, now tested here** (LITE BT cohort 828 + 624, outcome = today's exit `rm_base_pct`, 1D sign / median / terciles, 2025 vs 2026 direction-consistent S, week-block shuffle null, 500 shuffles):
  - the UI's **derived "direction" columns**: pair RSI direction, BTC RSI direction (5m and 30m), BTC 1h RSI direction, BTC ADX direction;
  - the **EMA fan ratio** (5-8 gap / 8-13 gap), including the UI dead-zone band [0.85, 1.70);
  - the **Range position × RSI direction** cross-tabs;
  - **pair identity**, read as the pair's own prior LITE record (closed before entry);
  - **hour and weekday in UTC-3** (the UI's clock);
  - **entry slippage**, notional size;
  - **funding rate** (2026 only — the full sweep never actually tested it, see below).
- **Result: nothing separates winners from losers.**
  - Derived-direction family (32 tests): best is BTC 5m RSI rising (better in both years: 2025 +0.04 vs −0.34, 2026 +0.44 vs +0.10). S = 1.45, family-wise p 0.19 inside this small family alone. That is the same "weak BTC" family already demoted, not a new signal.
  - Pair / time / size family (31 tests): best S 0.64, p ≥ 0.96.
  - Funding (2026): |t| ≤ 0.5 at every cut.
- **Post-entry variables are descriptive only** (they cannot be entry filters):
  - Losers go wrong early, and winners do not start faster. The median time to +1 % is the same for both: 10.8 vs 9.9 min in 2025, 8.6 vs 8.2 min in 2026.
  - Two in-trade exits were priced on every fill's own path. **Neither pays.**
    - "Exit if no +0.5 % by 30 min": 2025 −0.087 %/fill (P(Δ > 0) 0.02), 2026 −0.010.
    - "Exit on the first 5m close below EMA13": 2025 −0.006, 2026 −0.322 (P 0.00).
  - **The replay's `shadow_tight_pnl` column is not a usable exit for LITE.**
    - It reads +0.002 / +0.423 against today's −0.142 / +0.190, which looks like tight wins.
    - The same leash priced on the real paths (arm +0.40, 0.25 trail, −0.7 stop, live slip) gives −0.307 / −0.322 against today's −0.272 / +0.165, so it loses.
    - The two correlate at only 0.16. Rows where the shadow "never fired" carry −3.83, and "trailing" rows carry +3.30.
    - Do not read that UI column for LITE.
- **Bottom line:** no UI table hides an untested entry variable that separates LITE winners from losers. The "change nothing on LITE" recommendation stands.

## 1. UI table → variable → tested where → result

Legend:
- **SWEEP** = the full sweep (66,256 tests, day-shuffle null, no survivor, best family-wise p 0.32).
- **EPIC** = the earlier LITE EPIC entry study.
- **HERE** = this review.
- **n/a** = constant on LITE, empty on LITE, or a different sleeve.

| UI table (index.html) | variable(s) | tested where | result on LITE |
|---|---|---|---|
| Performance by Sleeve / Strategy / Entry Type | `sleeve`, `entry_strategy` | n/a (constant = FRENZY_LITE) | — |
| Performance by Macro Trend (EMA20) | `entry_macro_trend` | SWEEP (per level) | nothing |
| Performance by Confidence Level | `confidence` | n/a (always STRONG_BUY) | — |
| Trade Outcome Distribution / Closing Reason Summary / Exit Type / Signal Expired | `close_reason`, `exit_order_type` | post-exit (outcome itself) | §3: TP 389 / 310, stop 427 / 265, 12 h cap 12 / 51 (2025 / 2026) |
| Entry Conditions by Close Reason / by Outcome / by Strategy / by Strategy — Winners vs Losers | `entry_rsi(_prev)`, `entry_adx(_prev, _delta)`, `entry_gap`, `entry_ema_gap_5_8/8_13`, `entry_range_position`, `entry_ema20_slope`, `entry_ema5_stretch`, `entry_price_vs_ema5_pct`, `entry_atr_pct`, `entry_dist_from_ema13_pct`, `entry_pair_ema20_ema50_gap_pct`, `entry_pair_volume_24h_usd`, `entry_btc_rsi(_prev, _prev6)`, `entry_btc_adx(_prev)`, `entry_btc_1h_slope`, `entry_btc_atr_pct`, `entry_btc_trend_gap_pct`, `entry_btc_dist_from_ema13_pct`, `entry_btc_ema20_slope`, `entry_global_volume_ratio`, `entry_bull_pct`, `entry_bear_pct`, `peak_pnl`, `trough_pnl`, hold time | stamps: SWEEP · peak / trough / hold: HERE §3 (descriptive) | nothing (stamps); see §3 |
| Multiplier Cell Performance | `cell_multiplier*` | n/a (always 1×) | — |
| Pattern Cell Ship Performance | `pattern_cell_source`, `pattern_fixed_*` | n/a (empty on LITE) | — |
| Flip decision scorecard, Spike Program / Spike Fires, Bull-Run, Bear-Run, SURGE sleeves and ledgers | `spike_*`, `entry_surge_*`, `entry_br_*`, `entry_bear_*` | n/a (other sleeves; these stamps are empty on every LITE fill) | — |
| Monitor Periods (bull-run episode ledger) | market regime windows | SWEEP via grid breadth / regime columns | nothing |
| FRENZY Sleeve / Flagged Pairs (Hours, Gain since spike, Off peak, vs avg price, Volume ×, ATR 5m, 24 h vol) | `entry_frenzy_hours`, `entry_frenzy_run_pct`, `off_spk` / `bars_since_hi` (rebuild), `entry_frenzy_vs_vwap_pct`, `entry_frenzy_vol_mult`, `entry_atr_pct`, `entry_pair_volume_24h_usd` + all other `entry_frenzy_*` | SWEEP (and EPIC) | nothing |
| FRENZY Fills (N, WR, avg, Total$, avgPk%, Episodes) | outcome, `peak_pnl`, episode count (`k_now` / `k_all`) | SWEEP (episode family) · HERE §3 (peak) | nothing |
| FRENZY Short Observations | no trade | n/a | — |
| Entry Extension (EMA13 distance), Extension × Pair Vol Ratio / × ADX Delta / × Pair ADX | `entry_dist_from_ema13_pct`, `entry_pair_volume_ratio`, `entry_adx_delta`, `entry_adx` | SWEEP 1D + every 2D median quadrant | nothing |
| BTC Market Extension, × Global Volume, DOUBLE-STRETCH | `entry_btc_dist_from_ema13_pct`, `entry_global_volume_ratio`, pair extension | SWEEP 1D + 2D | nothing |
| BTC 24h-Range Position (below 24h high / above 24h low) | `entry_btc_off24h_pct`, `entry_btc_off24lo_pct` | SWEEP | near-miss family only (off24h ∧ buy-turn, p 0.44) |
| BTC 1h Slope, 5m × 1h alignment, 1h Slope × BTC ADX | `entry_btc_1h_slope`, `entry_btc_ema20_slope`, `entry_btc_adx` | SWEEP | LITE_BTC1H_FALLING: 2025 −0.69, 2026 +0.03 (observe line, unchanged) |
| BTC ATR × BTC ADX, BTC Volatility Regime | `entry_btc_atr_pct`, `entry_btc_adx` | SWEEP | nothing |
| 4-Cohort Pattern Coverage, Pattern C / W / C+W Combination Trackers, Pattern C / W Trackers, Unmatched Losers / Winners | `entry_pattern_c1..c9`, `w1..w6`, `c_any`, `w_any` | SWEEP (10 flags; c3 / c7 / c9 / w3 / w5 / w6 have ≤ 18 fires per year, so they are untestable, and c8 is constant) | nothing |
| Runner Trail Performance | runner exits | n/a (LITE has no runner) | — |
| Liquidity Sizing | `liquidity_capped`, `entry_liquidity_cap_notional` | flag n/a (always 0) · cap notional SWEEP | nothing |
| Entry Funnel | block counters (no outcome) | n/a | — |
| Stop Loss Deep Dive / Winning Trades Drawdown / Never Positive / Excursion (MAE vs MFE) | `peak_pnl`, `trough_pnl`, MFE / MAE, time to peak / trough | HERE §3 (descriptive) | losers rarely reach +1 % (49 / 46 %); winners' median drawdown is −1.1 % |
| Recovery Hold | `rh_*` | n/a (empty on LITE) | — |
| Hold-Time Expectancy, ATR × Hold-Time, Hold-Time × ATR | hold minutes | HERE §3 | median hold W / L: 2025 51 / 52 min, 2026 39 / 50 min |
| Performance by Entry Gap 5-20 / EMA5-8 / EMA8-13 | `entry_gap`, `entry_ema_gap_5_8`, `entry_ema_gap_8_13` | SWEEP | nothing |
| Performance by EMA Fan Acceleration | fan ratio = gap 5-8 / gap 8-13 (derived) | **HERE §2** | nothing (dead-zone band: 2025 −0.72 N 72, 2026 +0.36 N 67, signs flip) |
| Entry RSI, Range Position, ADX Delta, Pair −DI / +DI, Entry ADX, EMA5 Stretch, Quality Score, BTC Regime | `entry_rsi`, `entry_range_position`, `entry_adx_delta`, `entry_neg_di`, `entry_pos_di`, `entry_adx`, `entry_ema5_stretch`, `entry_quality_score`, `entry_btc_regime` | SWEEP | nothing |
| ADX Direction at Entry | `entry_adx` vs `entry_adx_prev` (= sign of `entry_adx_delta`) | SWEEP (sign split) | nothing |
| **RSI Direction at Entry**, **BTC RSI Direction (5m / 30m)**, **BTC 1h RSI Direction**, **BTC ADX Direction** | derived: rsi − rsi_prev, btc_rsi − prev, btc_rsi − prev6, rsi_1h − rsi_1h_prev, btc_adx − adx_prev | **HERE §2** (only the raw levels were in SWEEP) | nothing (best BTC 5m RSI rising S 1.45, p 0.19 in family) |
| Range Position × BTC RSI Direction / × Pair RSI Direction, Pair RSI Dir × BTC RSI Dir, Pair ADX Dir × BTC ADX Dir | derived 2D | **HERE §2** (range-position cross-tabs); dir × dir via the raw stamps in the SWEEP 2D | nothing (best range_pos > med ∧ BTC RSI rising: 2025 +0.24 vs −0.29, 2026 +0.55 vs +0.17, S 1.36, p 0.25) |
| Regime Transition Impact | `entry_btc_regime` → `exit_btc_regime` | entry side SWEEP · exit side post-exit | — |
| Pair EMA20 Slope, BTC EMA20 Slope, Pair / BTC EMA13-EMA50 Gap, BTC ADX, BTC Entry RSI and the cross-tabs (RSI × ADX, BTC RSI × BTC ADX, Slope × ADX, Gap × ADX, ADX Delta × BTC ADX, BTC RSI 30m × 5m, 1h × 5m, 1h × 30m) | `entry_ema20_slope`, `entry_btc_ema20_slope`, `entry_pair_ema20_ema50_gap_pct` (holds EMA13-EMA50), `entry_btc_trend_gap_pct`, `entry_btc_adx`, `entry_btc_rsi`, `entry_btc_rsi_1h`, `entry_btc_rsi_prev6` | SWEEP 1D + 2D median quadrants | nothing |
| Volume Cross-Tab, Volume Intersection (Global Vol × Pair Vol USD) | `entry_global_volume_ratio`, `entry_pair_volume_ratio`, `entry_pair_volume_24h_usd` | SWEEP 1D + 2D | nothing |
| Market Breadth at Entry | `entry_bull_pct`, `entry_bear_pct` (+ grid breadth) | SWEEP | nothing |
| Performance by Pair ATR(14)% | `entry_atr_pct` | SWEEP | nothing |
| **Performance by Pair** | `pair` (identity) | **HERE §2** (the pair's prior LITE record: any prior / prior N / prior avg) | nothing (best S 0.54) |
| Performance by Pair 24h Volume | `entry_pair_volume_24h_usd` | SWEEP | nothing |
| Performance by Pair Rank | `entry_pair_rank` (empty on LITE) | SWEEP via rebuild `pair_vol_rank_k1d` | nothing |
| Performance by Pair Listing Age | `entry_pair_age_days` (empty on LITE) | SWEEP via rebuild `pair_age_days` (2025 coverage 55 %) | nothing |
| **Performance by Hour / Day of Week (UTC-3)**, Day × Time Heatmap | open time | SWEEP had UTC `hour` (numeric) + `dow` (UTC); **HERE §2**: UTC-3 hour cuts, 6-hour sessions, UTC-3 weekday | nothing (heatmap 2D not run: about 4 fills per cell per year) |
| Period Performance / Over Time / Equity / P&L Distribution | time aggregates | n/a | — |
| Closed-orders columns: slippage, fees, notional | `entry_slippage_pct` (pre-declared exclusion in SWEEP), `notional_value` | **HERE §2** | nothing (replay slip is 0 on 2/3 of fills, so only the sign cut is usable) |
| Funding (rates) | `entry_funding_rate` empty on LITE; rebuild `pair_funding_pct` | **HERE §2**. In the SWEEP it was listed but sat in **no kept test**: 2025 coverage was 0.8 %, below the ≥ 15-per-year rule | 2026 only, N 595: sign t +0.48, median −0.34, terciles −0.50 / −0.04 — nothing |
| Live-only book stamps | `entry_book_spread_pct`, `entry_book_drift_pct`, `entry_fill_vs_book_pct`, `entry_order_lag_s` | not testable: 3 live LITE fills (ZK, DUSK, EPIC) | — |
| Post-exit tracking columns | `post_exit_*` | HERE §3 (descriptive) | see §3 |
| Shadow / phantom exits | `shadow_*`, `phantom_*` | HERE §3 | the replay `shadow_tight` is an artefact on LITE |
| `peak_reached_at`, `trough_reached_at`, `first_cross_ema13/20_*`, `first_rsi2_*`, `post_arm_min_pnl_pct` | timing stamps | **empty on every LITE fill** (replay, master, live) → rebuilt from paths and 5m klines, HERE §3 | see §3 |

## 2. Entry-known variables the full sweep did not test (1D, week-block null)

Cohort: LITE BT 2025 828 / 2026 624. S = direction-consistent min(t2025, t2026). The family-wise p is taken within each family below (500 week-block shuffles). Any family-wise p under 0.05 would still face the full-sweep multiplicity and the locked gates.

**Derived direction / fan / cross-tab family (32 tests; null max-S median 1.13, p95 1.81).** Top rows:

| test | 2025 N · in / out · WR in | 2026 N · in / out · WR in | t25 / t26 | S | family-wise p |
|---|---|---|---|---|---|
| BTC 5m RSI rising (btc_rsi > prev) | 386 · +0.04 / −0.34 · 50.0 % | 290 · +0.44 / +0.10 · 57.6 % | +1.82 / +1.45 | 1.45 | 0.19 |
| range_pos > med ∧ BTC 5m RSI rising | 190 · +0.24 / −0.29 · 53.7 % | 150 · +0.55 / +0.17 · 59.3 % | +2.15 / +1.36 | 1.36 | 0.25 |
| fan ratio ≤ q33 (0.35) | 282 · −0.34 / −0.08 | 202 · +0.06 / +0.36 | −1.22 / −1.17 | 1.17 | 0.46 |
| BTC 1h RSI rising, cut > median | 418 · −0.06 / −0.28 | 300 · +0.38 / +0.15 | +1.06 / +0.98 | 0.98 | 0.67 |
| pair RSI rising | 26 · +0.70 / −0.19 | 30 · +0.41 / +0.25 | +1.49 / +0.27 | 0.27 | 1.00 |
| fan ratio in UI dead-zone [0.85, 1.70) | 72 · −0.72 / −0.11 | 67 · +0.36 / +0.25 | −1.69 / +0.30 | −0.30 | 1.00 |

**Pair-history / time / size family (31 tests; null max-S median 1.20, p95 1.89).**
- Best is pair prior LITE count ≤ 1, S 0.54, p 0.98.
- The pair's prior LITE record points opposite ways by year (prior avg > 0: 2025 +0.06 vs −0.27, 2026 +0.08 vs +0.49).
- UTC-3 sessions and weekdays: S ≤ 0.64.

**Funding (2026 only, 595 of 624 scored):** nothing at any cut (|t| ≤ 0.50; zone P(avg < 0) ≥ 0.06 nowhere below 0.05).

## 3. Post-entry descriptive reads (cannot be entry filters)

These come from each fill's own tick / 1m path up to today's exit (+3 TP / −3 / 12 h). Today's exit reason was re-derived and matches `rm_base_why` on 100 % of fills. EMA crosses are the first 5m close below EMA13 / EMA20 on the cached 5m klines (EMA13 scored on 63–90 % of fills).

| median (2025 · 2026) | winners | losers |
|---|---|---|
| MFE % | 3.15 · 3.04 | 0.99 · 0.86 |
| MAE % | −1.09 · −1.13 | −3.13 · −3.04 |
| minutes to peak | 51 · 39 | 10 · 8 |
| minutes to trough | 10 · 7 | 52 · 50 |
| minutes to +1 % (when reached) | 10.8 · 8.6 | 9.9 · 8.2 (reached by 49 / 46 % of losers) |
| minutes to −1 % (when reached) | 10.8 · 7.5 (reached by 53 / 55 % of winners) | 8.8 · 6.8 |
| hold minutes | 51 · 39 | 52 · 50 |
| first 5m close below EMA13: minutes / P&L at it | 9.8 / −0.28 · 7.3 / −0.31 (63 / 61 % of winners) | 4.8 / −0.58 · 9.8 / −0.53 (90 / 83 % of losers) |
| first 5m close below EMA20: minutes / P&L at it | 14.8 / −0.38 · 9.8 / −0.39 | 14.8 / −0.80 · 9.8 / −0.76 |

**In-trade conditional reads** (outcome given the early path):

| condition | 2025 share · WR · avg | 2026 share · WR · avg |
|---|---|---|
| −1 hit before +1 | 52 % · 32.6 % · −1.03 | 52 % · 38.3 % · −0.74 |
| +1 hit before −1 | 48 % · 62.7 % · +0.77 | 48 % · 72.0 % · +1.33 |
| 5m close below EMA13 within 15 min | 55 % · 37.1 % · −0.76 | 53 % · 45.9 % · −0.27 |
| no 5m close below EMA13 before exit | 22 % · 76.8 % · +1.62 | 29 % · 73.7 % · +1.43 |
| no +0.5 within 30 min | 26 % · 26.2 % · −1.42 | 23 % · 22.8 % · −1.71 |

**Are these actionable as in-trade exits?**
- Each was priced at the fill's own path price at the trigger, with −0.20 live slip, and compared with today's exit on the LS ruler.
- The test is two-sided, per year, with a day-bootstrap. The results are **post-hoc**, so none is a candidate without pre-registration.

| rule | 2025 Δ %/fill (P > 0) | 2026 Δ (P > 0) | tick-only Δ | master Δ (N 20) | losers saved / winners cut (sum %) 2025 · 2026 |
|---|---|---|---|---|---|
| exit at 30 min if never +0.5 | −0.087 (0.02) | −0.010 (0.40) | −0.049 | +0.500 (4 switched) | +142 / −215 · +111 / −117 |
| exit at 60 min if never +1.0 | −0.072 (0.04) | +0.028 (0.74) | −0.011 | −0.009 | +150 / −210 · +111 / −94 |
| exit on first 5m close below EMA13 | −0.006 (0.47) | −0.322 (0.00) | −0.256 | −0.287 | +871 / −876 · +542 / −743 |

- The early-path signals do sort losers from winners. Acting on them cuts as many winners as it saves losers.
- This is the same "tight beats today ≈ today loses" collapse as the +1/−1 exit menu.

**Post-exit tracking** (replay `post_exit_*`, LITE seed 1, medians):
- After a +3 TP:
  - post-exit peak is +4.5 % (2025) and +4.7 % (2026) from entry;
  - the price is still at +2.7 / +2.8 after 15 min;
  - 41 % / 39 % of TP exits later print ≥ +5 %.
- After a −3 stop:
  - post-exit peak is −1.7 / −1.6;
  - only 19 % / 24 % get back to entry, and 5 % / 4 % to +3.
- Stops are mostly not wicks: the price stays down.
- TP exits leave some run on the table. The earlier LITE_EXITOPT walk-forward is the study that prices TP width.

**Shadow columns (UI "shadow" leash):**
- Replay `shadow_tight_pnl` reads +0.002 (2025) and +0.423 (2026), against the replay's own exit at −0.142 and +0.190.
- Priced on the real paths, the same leash (arm +0.40, 0.25 trail, −0.7 stop, LS ruler) gives −0.307 and −0.322, against today's −0.272 and +0.165. Tick-only: −0.324 against +0.075. Master: −0.251 against +0.207.
- The shadow and the path-priced leash correlate at r 0.16:
  - shadow rows with reason "window" (never fired) carry −3.83;
  - rows with reason "trailing" carry +3.30.
- The shadow is finalized at the post-exit window end on the engine's polling prices, so it does not price a tight exit on LITE.
- **Do not use the UI shadow_tight column to judge a LITE exit.**
- `phantom_*` is filled on 11 of 22 master LITE rows and empty in the replay. Not read: N too small.

## Scope and limits

- 2D was run only for the UI cross-tabs listed above. Dir × dir cross-tabs are covered by the raw stamps in the full sweep's 2D quadrants.
- Six pattern flags (c3 / c7 / c9 / w3 / w5 / w6) are too rare on LITE to test (≤ 18 fires per year).
- The Day × Time heatmap has about 4 fills per cell per year, so it was not tested.
- EMA cross timing uses 5m closes from the cached klines (the engine stamps it per scan). Coverage is 63–90 %.
- The in-trade exit pricing is a post-hoc screen. None of these rules was pre-registered.
