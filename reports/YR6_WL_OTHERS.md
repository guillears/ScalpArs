# YR6 two-year winners vs losers — AREA = OTHERS (FLIP-short · Spike-Fade · SURGE_LONG · BullRun-Long · BearRun-Short)

2026-10-13 · brief `reports/YR6_WL_STUDY_BRIEF.md` · data `reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv` (`pct_new`, 3 seeds pooled, units
collapsed so seed copies never add N) · real fills `reports/MASTER_POOL_stacked.csv` (today's stack, `stack_keep`, no probes).
Cache-only. No config, engine, git or bot changes.

## Plain-English summary

| sleeve | 2-yr replay | verdict | what separates winners from losers |
|---|---|---|---|
| **SURGE_LONG** | +0.013 %/fill, CI −0.20…+0.24 | **ENTRY SEPARATOR, confirmed out of sample** | The **pair's own trend at entry.** When the pair's EMA13 sits **below** its EMA50 (`entry_pair_ema20_ema50_gap_pct < 0`, a sign split with no fitted threshold), the fill loses in **all three regimes**: 2025 −0.43 · H1 −0.16 · H2 −0.50, against −0.07 / −0.01 / **+0.91** for the rest. The split also holds **inside the same trigger window** (−0.84, CI −1.45…−0.15), so it is a per-pair effect and not a market-day effect. It passes the locked expectancy bar on the pooled data and in each bull half. No real fills exist yet. |
| **FLIP-short** | −0.131, CI −0.21…−0.06 | **Neither a separator nor a regime switch. The replay's flip exit is too pessimistic** | No entry or macro variable separates winners from losers in either bull regime: confirmations sit at chance and the shuffled null ranks them P = 1.00 / 0.28. The loss is uniform: 94–100 % of cohorts are negative in 2025 and H1, which is what a sleeve-level offset looks like. Why real fills are positive: (1) **half the real kept flips (18 of 35) come from the one washed-out window** (Jun-17→Jul-2: +0.47). After it, real fills are −0.02 against the replay's −0.13. (2) **On the same entries, the live exit beat the replay by +0.28 %/fill.** 4 of 25 matched kept fills were stopped in the replay but not live (+1.24 each), with 0 the other way. Trail-vs-trail fills also show +0.13. This is the Sep-23 rule that a replica stops trades live rode through. The −0.131 cannot serve as a kill signal. |
| **Spike-Fade** | +0.008, ≈ breakeven every regime | **Bull: noise. Bear: possible regime switch (observe only)** | Bull halves: nothing passes the null; 0 or 1 confirmations, at chance. In the bear half there is real structure (2D: 114 cells with \|z\|>3 against a null of 14, fw-p 0.006). The strongest cell is market-wide: **weak breadth (`bull_pct` < 21.7) ∧ high BTC ATR (≥ 0.1445)**, at −0.28 against +0.09 (44 days against 86). It points the same way in 2025 (P 0.82, below the 0.90 bar), points the opposite way in H2, and real fills contradict it: 8 master fades in the cell, 8 winners, +0.55. Real fills beat the replay because the **replay only finds 35 of the 73 live kept fades**, and the 38 it misses average +0.49. On matched entries the exits agree (Δ −0.09, stop flips 4 against 4). |
| **BullRun-Long** | +0.061, 14 days in 2 years | **Noise: no power** | 8 + 6 trading days, so a split needs a 1.2–1.8 % effect to show at day level. 1 of 34 and 1 of 29 survivors confirm, below chance. The sleeve is already a regime switch (its door); its P&L is one Aug-2026 episode (best 5 days = 76 % of winning-day Σ). The replay matches only 5 of 102 live kept BullRun fills and misses the whole Sep-18→21 live window, so its door fidelity is poor. |
| **BearRun-Short** | −0.033, 11 days in 2 years | **Noise: untestable** | 4 / 5 / 2 days per regime. No macro screen is possible (fewer than 20 day units) and 0 of 14 survivors are confirmable (each lives on fewer than 5 days). Real fills: 2, both winners. |

**Recommendation (one change):** register **SURGE_LONG gap < 0 block** as the only candidate, armable after your review. Gate and revert are below. FLIP, Spike-Fade, BullRun and BearRun: **no change**. Spike-Fade's bear cell and FLIP high-ATR go to observe only. This is not a kill proposal for any sleeve, and the sleeve-kill checklist was not completed for FLIP; see the FLIP section.

---

## 1 · Per-regime sleeve table (`pct_new`, N/seed, 95 % day-clustered CI, BE-WR = |avg loss| / (avg win + |avg loss|))

Computed two ways: from the fills (this study) and from `ENGINE_REPLAY_YR6_2Y_TODAY_regimes.csv`. All 14 rows **MATCH** to 1e-6
(`YR6_WL_OTHERS_checks.txt` §A).

| sleeve | regime | N/seed | days | WR | avg % [95 % CI] | BE-WR | $/seed |
|---|---|---|---|---|---|---|---|
| FLIP-short | 2025 BULL | 106.3 | 89 | 63.9 % | −0.105 [−0.223, +0.006] | 72.0 % | −791 |
| | 2026-H1 BEAR | 92.0 | 70 | 58.7 % | **−0.196 [−0.326, −0.074]** | 73.5 % | −1,276 |
| | 2026-H2 BULL | 46.7 | 37 | 66.4 % | −0.062 [−0.217, +0.101] | 71.0 % | −204 |
| Spike-Fade | 2025 BULL | 221.7 | 155 | 65.7 % | +0.018 [−0.166, +0.196] | 65.0 % | +1,784 |
| | 2026-H1 BEAR | 279.0 | 130 | 76.7 % | −0.010 [−0.091, +0.064] | 77.2 % | −265 |
| | 2026-H2 BULL | 110.0 | 69 | 77.0 % | +0.036 [−0.120, +0.172] | 75.1 % | +462 |
| SURGE_LONG | 2025 BULL | 127.3 | 85 | 30.1 % | −0.167 [−0.459, +0.171] | 35.8 % | −3,004 |
| | 2026-H1 BEAR | 92.3 | 53 | 32.1 % | −0.058 [−0.368, +0.235] | 33.9 % | −753 |
| | 2026-H2 BULL | 84.3 | 37 | 34.4 % | +0.361 [−0.140, +0.910] | 25.9 % | +4,304 |
| BullRun-Long | 2025 BULL | 133.0 | **8** | 48.6 % | −0.033 [−0.178, +0.296] | 50.4 % | −627 |
| | 2026-H2 BULL | 130.0 | **6** | 48.7 % | +0.158 [−0.379, +0.513] | 41.5 % | +2,903 |
| BearRun-Short | 2025 BULL | 27.7 | **4** | 57.8 % | −0.099 [−0.830, −0.005] | 66.7 % | −97 |
| | 2026-H1 BEAR | 9.3 | **5** | 67.9 % | +0.037 [−0.325, +0.257] | 64.5 % | +12 |
| | 2026-H2 BULL | 6.7 | **2** | 75.0 % | +0.141 (n/a) | 60.5 % | +33 |

Whole-sleeve breakeven WR: FLIP 72.4 % · Spike-Fade 72.4 % (win +0.58 / loss −1.52) · SURGE 31.5 % (win +2.33 / loss −1.07) · BullRun 45.6 % · BearRun 65.6 %.

**2025 fidelity split (top-50 ∪ prior-day top-50 = tick exits; others = 1m bars):** FLIP 2025 top-50 −0.113 / non-top-50 −0.070 (82 % top-50).
SURGE top-50 −0.115 / non −0.341 (77 %). BullRun 100 % top-50. **Spike-Fade 2025 is 100 % non-top-50**, so every 2025 fade exits on
1m bars, and in 2026 only 3 % are top-50. Fade conclusions are therefore the lowest-fidelity in this area.

## 2 · Coverage (`reports/YR6_WL_OTHERS_coverage_<sleeve>.csv`)

- **100 % in every sleeve × regime:** every core pair stamp (rsi/adx/DI/ATR/EMA gaps/slopes/stretch/range/volume/rank/pattern flags) and
  every BTC/breadth stamp (rsi, rsi_1h, adx, atr, trend_gap, ema50_100, off24h/lo, off30d_high, r72, eff72, above72, 1h_slope, 1d_ret,
  eth_5m, bull/bear_pct), plus the categorical `entry_macro_trend` and `entry_btc_regime` (screened as level dummies).
- **< 80 % (scored-only family, never "rest"):** `entry_mcap_usd` / `entry_vol_mcap_ratio` / `entry_cmc_rank`. Spike-Fade 64/87/91 %,
  SURGE 78/87/92 %, BullRun 72/–/68 %, FLIP 84/93/99 %. The scored-only screens found nothing (best |z| 1.95 / 1.42 / 1.26, every fw-p ≥ 0.25).
- **Sleeve-specific:** `entry_br_*` only on BullRun, `entry_surge_*` only on SURGE, `entry_bear_*` + `gap_5_*_signed` only on BearRun,
  `entry_pair_1d_ndi` / `btc_4h_ema50_200_gap` / `pair_age_days` only on Spike-Fade (pair_age also on BearRun).
- **Not stamped on Spike-Fade:** `btc_ema20_slope`, `global_volume_ratio`, `btc_1h_slope`. **Not stamped anywhere:** `funding_rate`,
  `chop_burst_prior_fill_s`, `btc_regime_started_at`.

## 3 · Exhaustive screen: real vs shuffled null (500 shuffles; `YR6_WL_OTHERS_screen.txt`, every test in `_screen_<sleeve>.csv`)

Units: macro = DAY, pair = PAIR-DAY. Zones = sign / median / outer terciles (complements dropped). 2D = all quadrants of the top-40
columns (median splits), with the top-40 **re-selected inside every shuffle**. "excess" = real count of |z|>2 (1D) or |z|>3 (2D) against
the null mean (null 95th percentile).

| sleeve | regime | 1D macro (day) | 1D pair (pair-day) | 2D quadrants | read |
|---|---|---|---|---|---|
| FLIP | 2025 | 2 vs 4.2 (11) · fw 0.60 | 4 vs 4.6 (12) · fw 0.51 | 3 vs 14 (41) · fw 0.88 | **nothing** |
| | H1 bear | 5 vs 4.2 (10) · fw 0.44 | 8 vs 4.2 (11) · **fw 0.016** | 42 vs 14 (47) · fw 0.036 | weak bear-only structure |
| | H2 | 1 vs 3.7 (11) | 3 vs 4.6 (12) | 10 vs 12 (37) | **nothing** |
| Spike-Fade | 2025 | 0 vs 4.1 (12) · fw 0.99 | 2 vs 4.1 (11) | 0 vs 11 (38) | **nothing** |
| | H1 bear | 9 vs 4.5 (12) | 13 vs 4.6 (14) · fw 0.15 | **114 vs 14 (44) · fw 0.006** | **real bear structure** |
| | H2 | 0 vs 4.1 (11) | 2 vs 4.6 (13) | 3 vs 12 (38) | **nothing** |
| SURGE | 2025 | 8 vs 3.4 (10) · fw 0.37 | 12 vs 4.0 (13) · fw 0.06 | 38 vs 16 (55) | borderline |
| | H1 bear | 0 vs 4.4 (12) | 1 vs 3.6 (14) | 1 vs 14 (42) | nothing |
| | H2 | 3 vs 4.2 (13) | **22 vs 4.7 (15)** · fw 0.13 | 23 vs 18 (53) | **pair family beyond the null** |
| BullRun | 2025 / H2 | skipped (8 / 6 days) | 6 vs 3.7 / 2 vs 3.8 | 18 vs 12 / 84 vs 14 (56) | 59 / 56 units on ≤ 8 days: 2D excess is one or two windows |
| BearRun | all | skipped (4/5/2 days) | 2025 only: 19 vs 4.7 (14) · fw 0.018 | – | 32 pair-days on **4 days**: one-window artefact |

**Discover → confirm (frozen thresholds; confirm = same sign, day-bootstrap P ≥ 0.90, zone on ≥ 5 days in both halves).**
Confirmation-count null (§C: shuffle the confirmation half's pair-day outcomes, 300×, same criterion):

| sleeve | 2025 → H2 | H2 → 2025 |
|---|---|---|
| FLIP | 0 of 6 1D survivors (null 0.6) · P 1.00 | 1 of 3 (null 0.3) · P 0.28 |
| Spike-Fade | 0 of 4 (null 0.4) | 0 of 3 (null 0.3) |
| **SURGE** | **11 of 27 (null 2.8, 95th 9) · P 0.017** | 4 of 27 (null 2.3) · P 0.21 |
| BullRun | 0 of 9 | 0 of 4 |
| BearRun | 0 of 17 (every zone on < 5 days) | – |

The FLIP and Spike-Fade 2D "confirmations" in the screen text (2 + 2 and 1) are at the expected chance count (1.4–1.7 and 0.9) and carry
fw-p 0.95–0.998. They are noise.

## 4 · Per-sleeve read

### SURGE_LONG — entry separator (pair trend at entry)

The confirmed survivors are **one factor seen through several correlated stamps**: the pair is already trending up on its own
(EMA13−EMA50 gap above median or ≥ 0, EMA50 slope above median, RSI high tercile, −DI low tercile, +DI not low, W-pattern match, ADX rising)
against a pair that is only bouncing inside a downtrend. The cleanest, threshold-free expression is the **sign** of
`entry_pair_ema20_ema50_gap_pct` (which holds **EMA13 − EMA50**: the column is misnamed and must not be renamed). It was discovered in H2
(p 0.022) and confirmed in 2025 (P 0.93). The median version was discovered in 2025 and confirmed in H2 (P 0.93).

| SURGE_LONG | gap < 0 (block candidate) | gap ≥ 0 | Δ, P(Δ<0) day-bootstrap |
|---|---|---|---|
| 2025 BULL | 33.7/seed · 32 d · WR 28 % · **−0.431** [−0.709, −0.142] | 93.7 · 70 d · 31 % · −0.072 | −0.36 · 0.93 |
| 2025 top-50 only | 24.7 · 25 d · 23 % · −0.437 | 73.3 · 31 % · −0.007 | holds inside top-50 ✓ |
| 2026-H1 BEAR | 32.0 · 27 d · 31 % · −0.158 [−0.673, +0.325] | 60.3 · 45 d · 33 % · −0.005 | −0.15 · 0.70 |
| 2026-H2 BULL | 32.7 · 26 d · 16 % · **−0.503** [−0.963, +0.088] | 51.7 · 29 d · 46 % · **+0.907** | −1.41 · 1.00 |
| ALL | 98.3 · 85 d · 25 % · **−0.366** [−0.624, −0.095] | 205.7 · 144 d · 35 % · +0.194 | −0.56 · 1.00 |

- **Within the same trigger window** (110 seed-windows on 44 days holding both zones), gap<0 − gap≥0 = **−0.84** (day CI −1.45…−0.15).
  By regime: 2025 −0.21 · H1 −0.38 · H2 −1.81. This is a pair effect and is not explained by trigger or day quality. Windows with no gap<0 pick
  average +0.19, mixed windows −0.01, all-gap<0 windows −0.33.
- **Dose-response** (gap bins): ≤ −1 −0.44 · (−1,−0.5] −0.28 · (−0.5,0] −0.08 · (0,0.5] +0.08 · (0.5,1] +0.18 · (1,2] −0.20 · (2,4] +0.29 · > 4 +0.36.
  It is monotone below 0 with one dip at 1–2. The sign split does not depend on that dip.
- **Per seed:** zone −0.36 / −0.28 / −0.45 against rest +0.33 / +0.15 / +0.10. **Leave-one-month-out:** the zone avg stays negative with any of the 22 months removed
  (−0.43…−0.30). Δ kept-vs-sleeve stays at +0.14…+0.21 %/fill. The zone is negative in 17 of 21 months.
- **Concentration:** top pair 6 % of the zone loss (max 18 % PIPPIN in H1), top day 5 %. It is not a blacklist effect.
- **Locked EXPECTANCY bar** (computed two ways, mask vs raw loop, identical): WR 25.1 % < BE 31.5 % ✓ · avg −0.366, one-sided 95 % day-bootstrap
  upper −0.137 ✓ · 85 windows ✓ · pair 6 % / day 5 % ✓ · N 98/seed ✓. Each bull half passes on its own (2025 upper −0.180; H2 upper −0.035).
  **H1 bear fails on confidence** (upper +0.245), so it carries no bear-specific claim. Each seed alone passes.
- **Effect of blocking:** the sleeve moves from +0.013 to +0.194 %/fill, ΔΣ +36.0 %-pts/seed over 2 years. **After the 30–50 % haircut: +18 … +25.** Per half:
  2025 +14.5 (haircut +7.3…+10.2) · H1 +5.1 (+2.5…+3.5) · H2 +16.4 (+8.2…+11.5). After the block the sleeve is still regime-dependent
  (2025 −0.07, H1 −0.01, H2 +0.91). The filter removes the loser cohort. It does not make SURGE a bull-and-bear earner.
- **Real-fill check: none possible.** There is no SURGE_LONG row in the master pool or in the Oct-10…12 Downloads exports. Replay entry fidelity for
  SURGE is unverified against live.

### FLIP-short — no separator, not a regime switch; the replay's exit is pessimistic

- **No entry or macro separator.** See the screens in §3. The bear half has a weak pair family (best: `entry_ema_gap_5_8 ≥ 0.156` at −0.48 vs −0.07,
  fw-p 0.016). It **reverses in both bulls** (2025 −0.05 vs −0.13; H2 −0.04 vs −0.08), so it is bear-only and in-sample.
- **Uniform degradation:** 94 % (2025) and 100 % (H1) of 18 frozen tercile cohorts share the negative sign. Months are positive 5 of 22. The
  worst-5 days carry only 30–37 % of the losing-day Σ. A loss spread evenly over every cohort, every regime and every month has the signature
  of a **sleeve-level offset**, from an unmeasured regime variable **or a systematic exit-model bias**. The master comparison points to the second.
- **Why the real fills are positive (master: 35 kept · 80 % · +0.233; replay on the same live-active days, flip-dead Jul-8→Aug-11 removed: 35.3/seed ·
  67 % · −0.097):**
  1. **Washed-out window.** 18 of 35 kept real flips (9 days, +0.473) are Jun-17→Jul-2. After Jul-2, real kept flips are 17 · 65 % · **−0.020**
     against the replay's −0.131. The replay does **not** reproduce the window's result even inside it (17/seed, −0.061). In replay terms the washed-out state
     (BTC ≤ −15 % off its 30-d high) is not a flip regime (−0.095, 55/seed), so the window was one episode.
  2. **Exit side, same entries** (25 kept live fills matched within 20 min on pair; entry-price Δ median +0.06 % worse in the replay):
     live +0.245 against replay −0.037, **Δ +0.282/fill** (median +0.04). Per distinct fill: both trailed **+0.13** (17 fills, the replay trails out earlier,
     e.g. SENT 1.2 min against 10 min live); **replay stopped / live did not: 4 fills, Δ +1.24 each** (STG, DEXE 06-24 12:55, TAC, ENA; live
     trough −0.21…−1.01 never reached the replay stop); live stopped / replay did not: **0** among kept fills (1 among blocked: 1000PEPE). That is the
     asymmetric stop pattern the Sep-23 rule warns about. Fades, run through the same matching, are symmetric (4 against 4), so the asymmetry is flip-specific.
     Matched Δ by era: ≤ Jul-2 +0.44 (12 fills) · after +0.15 (13 fills, median +0.03).
  3. Selection: replay-only flips (11.7/seed, −0.145) lose more than the matched ones (−0.073). 10 live kept flips have no replay twin (+0.205).
- **Power:** a median split needs Δ ≥ 0.28 (2025) / 0.32 (H1) / 0.44 (H2) %/fill to show at 80 % power. The sleeve loses 0.06–0.20, so a separator smaller
  than the whole loss would be invisible in H2.
- **High-ATR flips (watch):** the top ATR tercile (≥ 0.698) loses in all three halves (−0.39 / −0.33 / −0.19, against low tercile +0.03 / −0.02 / +0.06) with
  a higher replay stop share in 2025 / H1 (48 % / 45 % against 32 %). 2025 discovery p 0.024, **H2 confirmation P 0.83 (< 0.90, not confirmed)**.
  Real kept hi-ATR flips: in the washed-out window 14 · 93 % · +0.54; **after it 7 · 57 % · −0.30** (4 days) against low/mid 10 · 70 % · +0.18. It shares
  the replay's direction outside the window, but it is tiny and sits exactly where stop-fidelity bias is largest. Observe only.
- **Sleeve-kill checklist: not run to completion here and not needed.** No kill is proposed. Items ①② (exhaustive entry and macro sweep at 3 granularities)
  are done above. ③ uniform degradation is present, so keep hunting. ④ the tape comparison is the washed-out window.

### Spike-Fade — bull halves noise; one bear-regime cell (observe only)

- Breakeven in every half (WR ≈ BE-WR). Bull screens sit below the null.
- **Bear cell (market-wide, read in DAY units): `entry_bull_pct` < 21.72 ∧ `entry_btc_atr_pct` ≥ 0.1445** (thresholds from the bear 2D screen, z −4.81, fw-p 0.006):
  H1 fills 78/seed · 68 d · WR 62 % · **−0.279** [−0.432, −0.136] against rest +0.094. Days: 44 cell-days −0.240 against 86 days +0.034. Loss concentration: top pair
  6 %, top day 6 %. **2025:** −0.154 against +0.067 (P 0.82; days +0.03 against +0.12), same direction, below the bar. **H2:** +0.119 against +0.024
  (9 days), opposite. **Real fills:** 8 master kept fades in the cell, all winners, +0.547, against 65 · 88 % · +0.333. Live refutes it in H2.
  In the bear it would pass the expectancy bar in-sample (WR 62 % < 72 %, CI < 0, 68 windows), but it is unconfirmed out of sample and refuted
  by live, so it stays observe only.
- **Why real fills beat the replay** (master kept 73 · 89 % · +0.357 against replay +0.014 on the same days): **this gap is selection, not exits.**
  Matched exits: live +0.187 against replay +0.273 on 35 fills (stop flips 4 against 4). The replay finds only 35 of 73 kept live fades; the 38 it misses average **+0.491**,
  and its 20.7/seed replay-only fades average −0.324. Fade entries are spike detections on non-top-50 pairs (1m bars in the replay), so the replay
  picks a different fade set from live. Any yr6 fade separator inherits that.

### BullRun-Long / BearRun-Short — no power

- BullRun: 14 trading days in 2 years, macro screen impossible, MDE 1.2–1.8 % at day level. 1 confirmation per direction (below chance) on zones that live on
  5–8 days. 2026-H2 is one August episode (best 5 days = 76 % of winning-day Σ). Master: 102 kept · 48 % · −0.033 on 8 days. The replay matches 5 of them
  and **misses the live Sep-18→21 window entirely**, so the replay's door does not reproduce live. The sleeve is a regime switch by construction and
  its two-year read is one or two episodes per half.
- BearRun: 11 days (4 / 5 / 2). Every zone lives on fewer than 5 days, so nothing is confirmable. Real: 2 fills (Oct-7), both +, matched to replay (Δ +0.08).

## 5 · Regime-switch read

| sleeve | regimes | uniform degradation? | classification |
|---|---|---|---|
| FLIP-short | − / − / − | yes (94–100 % of cohorts share the sign) | sleeve-level offset; most likely replay exit bias plus one live window. **Not a regime switch, not an entry filter** |
| Spike-Fade | 0 / 0 / 0 | no (61–67 %) | breakeven; bear-only macro cell (observe) |
| SURGE_LONG | − / − / + | no (56–72 %) | **entry separator**; the sleeve still leans on bull tape after the block |
| BullRun-Long | − / – / + | n/a (14 days) | regime switch by design; no power |
| BearRun-Short | − / + / + | n/a (11 days) | no power |

## 6 · Recommendation and pre-committed gates

1. **SURGE_LONG `entry_pair_ema20_ema50_gap_pct < 0` → BLOCK (candidate, armable after your review).** It meets the locked expectancy bar on 2-year replay,
   holds in both bull halves out of sample, within the trigger, per seed and leave-one-month-out. Expected ΔΣ after the haircut is **+18…+25 %-pts per seed
   over 2 years (≈ +0.8…+1.1 %-pts/month at 1×)**. Caveats: 0 live SURGE fills exist, and the replay's SURGE entries are unchecked against live.
   If it is armed, this is the **pre-committed revert**: re-price every blocked SURGE_LONG candidate from ticks. If, after **≥ 10 blocked candidates over ≥ 5 distinct trigger windows**,
   the blocked cohort shows **WR ≥ 31.5 % (sleeve BE) ∧ avg > 0**, revert. Separately, once 15 kept SURGE_LONG fills exist, the kept cohort mean ≤ 0 falls under
   the existing SURGE revert (`surge_long_lev_mult 0.05`). The threshold is frozen at 0, never re-fit. Per the OBSERVE-FIRST rule you may instead
   start it observe-only (the stamp is already on every fill). The study's lean is ARM, because it clears the gate rather than falling below it.
2. **FLIP-short: no change, no kill.** The decision-grade next step is an **exit-fidelity check of the replay's flip exit** (poll-cadence replica vs tick
   replica on the 35 kept + 20 blocked live flips). Until that check exists, yr6 flip P&L is refute-only. Watch (frozen, observe only): **FLIP hi-ATR `entry_atr_pct ≥ 0.698`**.
   Bar = the locked expectancy bar on FRESH live flips (opened after 2026-10-13), N ≥ 15 on ≥ 8 days.
3. **Spike-Fade: no change.** Watch (frozen, observe only, a regime switch so read in DAY units): fades with `entry_bull_pct < 21.72 ∧ entry_btc_atr_pct ≥ 0.1445`.
   The bar is ≥ 8 fresh cell-days with day-mean < 0 at 95 % (day bootstrap) ∧ cell WR < 72.4 %. Live currently refutes it (8 · 100 %).
4. **BullRun / BearRun: no change**. There is no power to conclude anything.

## 7 · Could not test

- **SURGE_LONG on real fills** (none exist). **BearRun / BullRun macro screens** (< 20 day units per half). BearRun confirmation (every zone on < 5 days).
- `funding_rate`, `chop_burst_prior_fill_s` are not stamped on any sleeve. Spike-Fade lacks `btc_ema20_slope`, `global_volume_ratio`, `btc_1h_slope`.
  `mcap / vol_mcap / cmc_rank` are 64–92 % scored, so they were read scored-only (2025 caps are a proxy: today's supply × 2025 price; delisted coins have none).
- **Exit fidelity:** flip replay exits are pessimistic against live (above). Spike-Fade 2025 = 100 % 1m-bar exits. The replay's fade and BullRun entry sets differ from live
  (35/73 and 5/102 matched). The replay's ENTRIES are 01419d4's; second-order slot effects are not modelled.
- Intra-trigger ordering / which of the 4 SURGE slots a pick used, because `entry_surge_trigger_at` is unstamped (the 30-min open-time bucket was used as the trigger proxy).
- Each screen counts ~3,000 2D tests per half. Survivors are a SCREEN; the counts against the null are what support any family-level claim.

## 8 · Exact commands

```
venv/bin/python scripts/study_yr6wl_others_screen.py --perms 500   # §3 screens, confirmations → reports/YR6_WL_OTHERS_screen.txt, _confirm.csv, _screen_<sleeve>.csv, _coverage_<sleeve>.csv (~15 s)
venv/bin/python scripts/study_yr6wl_others_master.py               # §4 replay vs master matching → reports/YR6_WL_OTHERS_master.txt, _master_matches.csv
venv/bin/python scripts/study_yr6wl_others_checks.py               # §1 two-way table, §5 uniform degradation, confirmation null, SURGE bar ×2 + robustness, fade cell, power → reports/YR6_WL_OTHERS_checks.txt
# shared helpers: scripts/study_yr6wl_others_common.py
```
