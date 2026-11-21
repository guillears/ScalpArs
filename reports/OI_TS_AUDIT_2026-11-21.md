# OI_TS_AUDIT 2026-11-21: do any shipped decisions rest on look-ahead in the Binance `metrics` files?

Research only. No engine, config, AWS or git change.
Scripts: `scripts/OI_TS_AUDIT_2026-11-21_aligned.py` (shared aligned reader), `_test.py` (5 synthetic tests, all pass), `_rerun.py` (re-runs DIST and EXH_SSC_OIVOL).
Data: `reports/OI_TS_AUDIT_2026-11-21_data/`.

## 0 · Verdict

- **No live decision changes.** The two armed or running decisions that read vision metrics are DISTRIBUTION_SHORT (DL 586) and EXH_SSC_OIVOL (DL 582). Both studies used `create_time ≤ T − 5 min`. That rule already absorbs the 5-minute label shift, so **neither study had look-ahead**.
- **One mismatch remains, and it is not look-ahead.** From 2024-05 to 2026-11-15, those studies read the snapshot taken **at T**. Live reads the snapshot **at T − 5 min**, one period older, because the API stamps rows with the snapshot time (checked below). Re-run on live-parity reads:
  - **DIST: WEAKER, verdict UNCHANGED.** The armed cohort goes from +0.88 to +0.82 %/trade and DCR from +0.567 to +0.524 %/day. HOLDOUT goes from +0.84 to +0.81, still 7/7 half-years positive. The RSI1h HOLDOUT Δ goes from +0.85 to +0.81.
  - **EXH_SSC_OIVOL: UNCHANGED.** The blocked cohort goes from −0.63 to −0.56 %/trade. Its upper 95 % bound is still < 0 (−0.21). Filtered-sleeve DCR goes from +0.810 to +0.793 %/day.
- **4 research scripts use a naive reader with look-ahead.** None of them fed a shipped or armed rule (see §2).
- **New fact: the shift has ended.** Vision files are correctly labelled again from **2026-11-16**. The timeline is: correct through 2024-04, +5 min from 2024-05 to 2026-11-15, correct from 2026-11-16. So the rule "create_time = period START" is wrong for new files. A reader that hard-codes −5 min will now read one period stale. It is conservative and has no look-ahead.

## 1 · The finding, re-measured (today, 2026-11-21 ~22:20 UTC)

Test: the implied price `sum_open_interest_value ÷ sum_open_interest` compared with the 5m close at the label ± offset (median |error|).

| file / source | at label | label + 5 min | reading |
|---|---|---|---|
| vision DOGE 2024-03-05 | **0.041 %** | 0.757 % | label = snapshot |
| vision SOL 2025-10-10 | 0.174 % | **0.006 %** | label = snapshot − 5 |
| vision SOL 2026-10-15 · 11-10 | 0.124 · 0.098 % | **0.010 · 0.010 %** | label = snapshot − 5 |
| vision DOGE 2026-11-11 … 11-15 | 0.10–0.16 % | **0.007–0.016 %** | label = snapshot − 5 |
| vision SOL 2026-11-16 · 17 · 18 · 20, DOGE 11-19 | **0.011–0.014 %** | 0.10–0.16 % | **label = snapshot again** |
| **live API** openInterestHist SOL / DOGE (last 30 rows) | **0.011 / 0.016 %** | 0.086 / 0.054 % | **API `timestamp` = snapshot time** |

API vs vision, same rows:
- **SOL 2026-11-10** (shift era): the API value at `timestamp` equals the vision value at `create_time = timestamp − 5 min`. This holds exactly for OI and for the top-position ratio, and within 2e-4 for the account / global ratios. So the **ratio columns carry the same shift as OI**, and the API was never shifted.
- **SOL / BTC 2026-11-20:** API timestamp = vision create_time.
- The taker ratio is stamped differently (API ts = vision ct + 5 even today). It is not used by any live rule.

Pair-wide alignment over every metrics row of the 633 pairs in the DIST / EXH pool: 61.4 M rows, 98.6 % measured. Of the measured rows, 97.4 % are +5 min and 2.6 % are +0 (`align_stats_by_pair.csv`). This agrees with `reports/NEWSLEEVE_2026-11-21_data/oi_alignment_by_month.csv`.

What each reader actually used in the shift era:

| reader rule | snapshot used in 2024-05 … 2026-11-15 | before 2024-05 / from 2026-11-16 |
|---|---|---|
| `create_time ≤ T` (naive) | **T + 5 min → look-ahead** | T (OK) |
| `create_time ≤ T − 5 min` (most studies) | T (no look-ahead; live cannot see it yet) | T − 5 (= live) |
| live engine: API `timestamp ≤ T − 5 min` | T − 5 | T − 5 |

Live stamps confirm this. All 6 real DIST fills (11-19 → 11-21) have `entry_dist_pos_lag_min` = 5.0, so live reads the snapshot at T − 5.

## 2 · Inventory (scripts/ readers of metrics / the /futures/data endpoints)

Status key:
- (a) naive, affected by look-ahead
- (b) lagged `≤ T − 5`: no look-ahead, one period fresher than live in 2024-05 … 2026-11-15
- (c) live API
- (f) fetch only

| script | status | fed | live today? |
|---|---|---|---|
| `oos_real_flatbase_wl_features.py` (`metrics_feats`, `ts ≤ T`) | **(a)** | OOS_REAL_FLATBASE Stage C rescue screen. Its 5 survivors all failed holdout; already flagged in `compile_sleeve_review_text.py`. The lag-fixed rerun is `oos_real_flatbase_wldeep.py` (b) | no rule shipped |
| `study_post_frenzy_wl_features.py` (`ts ≤ T`) | **(a)** | POST_FRENZY_SHORT_V2 W/L features. No DECISION_LOG entry | no |
| `STATE_EXIT_2026-11-21_build.py` (`create_time ≤ t_k`, the 1m bar OPEN, decision at the bar close) | **(a)** up to 4 min | STATE_EXIT study: null verdict, nothing shipped | no |
| `exh_exit2_eval.py` (`regime_lookup`, `m.t ≤ t`) | **(a)** | EXH_EXIT2 regime split only. Verdict "KEEP today's exit" (DL 579 rests on EXH_EXIT_GRID, not on this) | inert |
| `NEWSLEEVE_2026-11-21_eval.py` (`oi_at`) | fixed (implied-price aligned) | NEWSLEEVE: H2 refuted after the fix | no |
| `unified_short_build.py`, `leads_deep_l1.py`, `dist_wl_build.py`, `dist_parity_fixture.py` | (b) | **DL 586 DISTRIBUTION_SHORT trigger + RSI1h study** | **yes → re-run §3** |
| `leads_deep_l4.py` (+ the EXH_BADDAY builder `sq/build.py`, not in repo; DL 582 states the `≤ T − 5` window) | (b) | **DL 582 EXH_SSC_OIVOL** | **yes → re-run §4** |
| `ta_wl_features.py`, `unified_long_build.py`, `nospot_hybrid_build.py`, `nospot_sleeve_build.py`, `nospot_mcap_features.py`, `high_atr_eval.py`, `frenzy_early_build.py`, `frenzy_early_part3.py`, `frenzy_hunt_build.py`, `new_sleeves_run.py`, `wl_deep_bullbear.py`, `wl_deep_fan_fetch.py`, `wl_deep_frenzy_long_features.py`, `wl_deep_surge_data.py`, `oos_real_lite_wl_features.py`, `oos_real_flatbase_wldeep.py`, `oos_real_flatbase_stack3d.py`, `oos_real_willy_features.py` (`≤ sig − BAR`), `verify_1117_surge_sim.py`, `verify_lite_report.py`, `verify_lite_stack.py`, `brev_sig_fetch.py` (`dct − 5`) | (b) | W/L studies, refuted leads, observe lines | no armed rule depends on a metrics feature from these (DL 580 CROWDED_LONGS is observe-only) |
| `scout_willy_ht_lines.py` (API, `timestamp ≤ fill − 1 min`) | (c) OK | DL 580 CROWDED_LONGS, OBSERVE | observe. API ts = snapshot, so no look-ahead |
| `nospot_sleeve_live_fetch.py` | (c) | writes API timestamps as `create_time`. **This mixes conventions** with vision rows in the shift era (fine under `≤ T − 5`) | no |
| `study_bearrun_v2_sep_fetch.py`, `dist_entry_timing_realtrig.py` | (c) | Bear-Run v2 study; DIST timing study | no |
| `NEWSLEEVE_2026-11-21_fetch.py`, `STATE_EXIT_2026-11-21_common.py`, `exh_exit2_data.py`, `high_atr_data.py`, `leads_deep_fetch.py`, `leads_deep_old.py`, `long_leads_deep_build.py`, `frenzy_early_data.py`, `frenzy_hunt_data.py`, `nospot_mcap_fetch.py`, `nospot_sleeve_fetch.py`, `oos_real_willy_data.py`, `verify_1117_lib.py`, `verify_lite_aux_fetch.py`, `compile_sleeve_review_text.py` (text) | (f) | caches only; their readers are classified above | — |

Counts:
- **4 scripts affected by look-ahead**; none is upstream of a live rule.
- 2 live decisions use metrics data: DL 586 and DL 582.
- DL 580 CROWDED_LONGS is observe-only and reads the live API with no look-ahead.

## 3 · DL 586 DISTRIBUTION_SHORT: original vs aligned

Setup:
- Same pool (`reports/UNIFIED_SHORT_data/pool.pkl`), same VSW15 trigger, threshold −0.01097 and RSI1h ≥ 74.07.
- Same EXHAUSTION live exit the study used (`LIVE`), same one-position rule, 0.10 % slip, taker / maker fees, real funding.
- DCR at ONE size, 0.61 × equity.
- DESIGN < 2025-07 ≤ HOLDOUT (the DIST_WL split).

Parity: the ORIG recompute matches the pool's `top_pos_chg4h` on 99.8 % of 12,405 rows. ORIG also scores about 500 more rows, because more metrics files are cached today.

| cohort · read | N | WR | avg net % | t_cl | DESIGN avg | HOLDOUT N · avg | Kelly/day | DCR %/day | max DD | half-years > 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| armed (L1 ∧ RSI1h) · POOL (as studied) | 1,549 | 42.3 % | +0.902 | 5.04 | +1.016 | 989 · +0.837 | 0.056 | +0.572 | −64.7 % | 7/7 |
| armed · ORIG recompute | 1,573 | 42.0 % | +0.883 | 4.99 | +0.961 | 989 · +0.837 | 0.055 | +0.567 | −64.7 % | 7/7 |
| **armed · LIVE-parity aligned** | **1,584** | **41.5 %** | **+0.824** | **4.64** | **+0.855** | **997 · +0.806** | **0.048** | **+0.524** | −60.6 % | **7/7** |
| armed · NAIVE (scale only) | 1,564 | 41.8 % | +0.849 | 4.74 | +0.857 | 976 · +0.844 | 0.051 | +0.535 | −64.9 % | 7/7 |
| L1 no RSI cut · ORIG → LIVE | 4,847 → 4,828 | 35.4 → 35.5 % | +0.291 → +0.290 | 3.23 → 3.20 | | 2,989 · +0.260 → 2,975 · +0.252 | 0.001 → 0.001 | +0.405 → +0.406 | | 7 → 6 |
| RSI-blocked side · ORIG → LIVE | 3,309 → 3,278 | | +0.012 → +0.035 | | | −0.008 → −0.008 | | | | |
| **RSI1h HOLDOUT Δ (kept − blocked)** | | | **+0.845 → +0.814** | | | | | | | |

What changes and why:
- **Fills that change (ORIG → LIVE):** 1,537 fills are in both (+0.88). 36 drop out (44 % WR, +0.83). **47 are new (28 % WR, −1.14).** Live fires slightly more often, and those extra fires are losers. That is the whole drop.
- **Feature shift:** the 4 h change moves by a median 0.009 (mean 0.021). 606 of 9,161 VSW candles cross the threshold one way or the other.
- **By era (armed):**
  - pre-2024-05: 228 · +1.37 → 227 · +1.33. Near-identical, as expected: there is no shift in this era.
  - 2024-05+: 1,342 · +0.80 → 1,354 · +0.74.

**Verdict: WEAKER (−0.06 %/trade, −7 %; DCR −0.04 %/day), verdict UNCHANGED.**
- The sleeve stays positive in every half-year, in DESIGN and in HOLDOUT, with Kelly > 0.
- The RSI1h keep cut still separates kept from blocked by +0.81 on HOLDOUT.
- DL 586 is a declared override, so no locked ship gate exists to flip.
- The frozen forward gates (30 fills / 10 days) are unaffected: they read real fills.
- The aligned numbers are the honest backtest reference for live: **+0.82 %/trade · +0.52 %/day at 0.61×**, not +0.88 / +0.57.

## 4 · DL 582 EXH_SSC_OIVOL: original vs aligned

Setup:
- EXH cohort = `EXH` flag in the same pool, LIVE exit, one position per coin.
- Block = ssc > −0.0366 ∧ OI_usd ÷ vol24 > 0.128.
- NaN OI → kept (fail-open, as live).
- Bounds are a one-sided 95 % day-block bootstrap.

Parity: `oi_to_vol24` matches on 99.8 % of 12,464 rows. The base N is 4,115 here vs 3,998 in DL 582: a different pool build. The comparison stays like-for-like inside this pool.

| cohort · read | N | WR | avg net % | t_cl | HOLDOUT avg | 95 % range of the mean | days | DCR %/day (0.61×) |
|---|---|---|---|---|---|---|---|---|
| EXH base (no filter) | 4,115 | 39.5 % | +0.430 | 3.86 | +0.466 | | | +0.553 |
| blocked · ORIG | 682 | 29.8 % | −0.625 | −3.06 | −0.575 | −0.94 … −0.28 | 440 | |
| **blocked · LIVE-parity aligned** | **691** | **30.2 %** | **−0.556** | **−2.73** | **−0.454** | **−0.88 … −0.21** | 443 | |
| blocked · NAIVE | 679 | 29.3 % | −0.666 | −3.24 | −0.608 | −0.99 … −0.31 | 438 | |
| filtered sleeve · ORIG → LIVE | 3,447 → 3,439 | 41.4 → 41.4 % | +0.643 → +0.636 | 5.13 → 5.03 | +0.666 → +0.652 | | | **+0.810 → +0.793** |

- Feature shift: OI ÷ vol24 moves by a median 0.9 %.
- Only **16 of 4,257** EXH signals change their block flag.

**Verdict: UNCHANGED.**
- The blocked cohort is still clearly negative in both halves.
- The filter still adds about +0.24 %/day of DCR over the unfiltered base.
- The 5-minute read was never the driver: OI moves little in 5 minutes against 24 h volume.

## 5 · Live-code check

- `services/exhaustion.oi_pick` and `services/distribution.pos_chg_4h` read the fapi data endpoints. They pick the latest row with `timestamp ≤ T − 5 min` and `≥ T − 30 min`; DIST also reads a row 4 h back, with 15 min tolerance. **No look-ahead.**
  - The API `timestamp` is the snapshot time, so live reads the snapshot at **T − 5**.
  - The DIST real-fill stamps confirm a lag of 5.0 min.
- The **code comments are wrong**, but the behaviour is right. `services/exhaustion.py` L470–486, `services/distribution.py` L14, the DL 582 / 586 text and `scout_exh_ssc_oivol.py` say "API timestamp == metrics create_time, the period START". That was only true because the 11-17 check fell after vision's 2026-11-16 fix.
  - Consequence: **paper / live run one period behind the backtest** in 2024-05 … 2026-11-15 history.
  - §3 / §4 price that gap: DIST −0.06 %/trade, EXH filter ≈ 0.
  - No engine change is needed to remove look-ahead. A change would only make live match the older, fresher-looking backtest, and live cannot have that value yet: the T snapshot publishes ≈ T + 2 min.
- `scout_willy_ht_lines.pick_ratio` (`≤ fill − 1 min`, assumes "timestamp = period END") is correct against today's API.

## 6 · Recommendation

1. **No decision is flagged for the operator.** No locked gate flips. DL 586 and DL 582 hold. DIST's backtest reference should be quoted from the aligned row: +0.82 %/trade, +0.52 %/day at 0.61×.
2. **Retire the hard-coded "create_time ≤ T − 5" convention** for new research and use the shared reader `scripts/OI_TS_AUDIT_2026-11-21_aligned.py`:
   - It measures each (pair, day) file's offset from the implied price, with a calendar fallback (+5 for 2024-05-01 ≤ day < 2026-11-16).
   - Then `pick(snap, T, lag=5 min)` reproduces live exactly.
   - Its 5 tests prove there is no look-ahead on a synthetic shifted file, and that the naive reader does look ahead.
   - When promoted, it should move under `tests/` (the regression suite gate). That is outside this audit's file scope.
3. **Comment-only fix proposal** (engine untouched here; needs the operator, the review gate and a commit): reword the "period START" comments in `services/exhaustion.py`, `services/distribution.py` and the scout docstrings to "API timestamp = snapshot time; vision create_time = snapshot − 5 min for 2024-05 … 2026-11-15".
4. **If any of the 4 naive scripts is ever reused** (OOS_REAL_FLATBASE Stage C, POST_FRENZY v2 features, STATE_EXIT, EXH_EXIT2 regime), switch it to the aligned reader first. Their past verdicts were null or keep, so nothing needs re-deciding.
5. **Caveats:**
   - Backtest pool only. There is no master read: the real DIST / EXH fills all come after 2026-11-16, where vision and live agree.
   - The pool rebuild differs slightly from the shipped study counts (EXH 4,115 vs 3,998), so only the paired ORIG vs LIVE deltas are the evidence here.
   - No in-sample haircut is applied, because no new rule is proposed.
