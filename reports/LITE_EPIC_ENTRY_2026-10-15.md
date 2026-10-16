# LITE: can the entry tell which trades become EPICs? (2026-10-15)

Research only. No engine, config, `trading_config.json`, template or test file was changed. Nothing was committed and the bot API was not called. Cache-only: no Binance call was made.
- The cohort, classes, features, screen, null and filter bar were pre-registered in `reports/LITE_EPIC_ENTRY_PREREG.txt` (frozen 19:18:43 UTC) before any feature × outcome table was computed.
- This study has not been reviewed, so it supports no arm and no ship.
- Today's LITE exit: +3 net as a resting maker take-profit / −3 stop / 12 h cap, 10×.

## Plain-English summary

**Can we tell at entry which LITE trades will become EPICs? No, not reliably.**
- An "EPIC" is a LITE trade that climbs to +2 or more and then falls all the way to the −3 stop. About 1 LITE trade in 10 is one (2025: 94 of 828; 2026: 64 of 624).
- 105 things known at the moment of entry were checked:
  - all 58 values the engine already records;
  - 14 rebuilt from the coin's 5-minute candles (signal-candle volume and buyer share, 1 h and 4 h move, distance from the post-spike high, and so on);
  - 4 about the episode (how many FRENZY entries the coin already had);
  - 29 about the whole market (BTC, breadth, top-10 buying, the market-impulse score and its 5-minute **turn**).
- Every one of them was tested alone and in every pair: 22,317 tests. Each was scored against 500 reshuffled copies of the data.
- Only the weakest kind of pattern showed up, and it does not separate EPICs from the "dippers". A dipper also reaches +2, dips, and then recovers to +3.
  - A model trained on one year and tested on the other ranks EPICs only slightly better than a coin flip. AUC is 0.52 and 0.55, where 0.50 is a coin flip.
  - Testing the best one-year pattern on the other year never confirmed it: 0 of 16 attempts.
- EPIC and dipper look the same at entry, just as LITE_EXITOPT found they look the same at the dip.

**What would avoiding them be worth? A lot, if it were possible.**
- A perfect EPIC filter (refuse every EPIC and nothing else) would add about **+0.34 %/trade in 2025 and +0.31 in 2026**. That is more than LITE's whole 2026 average of +0.26.
  - On tick-priced trades it would add +0.29. On the 20 real master trades it would add +0.15 (only 1 EPIC: BTW).
- The bigger pot is the losers that never reach +2 (DUSK type). A perfect filter for those would add about +1.1 %/trade. Nothing at entry gets close to either.

**One thing did pass, and it is a loser filter, not an EPIC detector.**
- The rule: refuse LITE when **BTC's 1-hour RSI is ≤ 55 AND the top-10 coins' taker-buy share ticked up over the last 5 minutes.** In plain words: a weak hour for BTC, plus a little burst of market buying, is a bad moment to buy a frenzy coin.
- It passed the pre-registered bar in both years.
  - It improves 2025 by **+0.14 %/trade** and 2026 by **+0.10** (H1 +0.11, H2 +0.08). It loses money with confidence in both years (P 1.00 and 0.96, win rate 40 % and 44 % vs 50 % breakeven).
  - It stays positive when any single month is dropped: the minimum is +0.11.
  - Every neighbouring threshold tested is also positive in both years: 20 of 20.
- But be careful:
  1. **It is not an EPIC detector.** It removes 35 % of EPICs, but also 26 % of the never-armed losers and about 20 % of the winners. Most of its gain comes from the never-armed losers.
  2. **It barely made the screen.** Its family-wise p is 0.048, against a 0.05 cut-off.
     - Across all 22,317 tests, the number of "good-looking" patterns (37) is above the reshuffled median (12) but not above the reshuffled 95th percentile (50). As a whole, the screen is not clearly beyond chance.
  3. **The 20 real master trades disagree.** It would have refused 5, and 4 of them were +3 winners: −0.45 %/trade on master.
     - Live: it would have refused DUSK (saving +2.54). It just misses EPIC: EPIC's buying turn was −0.0042 against the −0.0039 cut. It would have kept DEEP.
  4. **It sits inside a line we already watch.** All of its value is inside the existing **LITE_BTC1H_FALLING** observe line (BTC 1h slope ≤ 0).
     - Within that line, it picks out the half that loses in every period.
     - Outside that line it adds nothing (+0.03 / −0.01).
  5. On a 30–50 % in-sample haircut, it is worth about **+0.06 to +0.09 %/trade**. The engine does not record the top-10 taker-buy turn today.

**Recommendation (operator decides).**
- **Do not add an entry filter to LITE.**
- If you want to keep watching, the defensible step is an **observe-only** Scout line on the rule above, with the gate frozen now (see Recommendation below).
- No exit change follows from this. LITE stays at +3 resting / −3 / 12 h.

## 1. Three blocks — classes under today's exit

| block | N | EPIC | DIPPER | other WIN | never-armed loss (NAL) | CAP | EPIC % | avg %/fill (resting-TP ruler) | avg, live-slip ruler |
|---|---|---|---|---|---|---|---|---|---|
| ③ BT 2025 | 828 | 94 | 158 | 227 | 338 | 11 | 11.4 | −0.165 | −0.272 |
| ③ BT 2026-H1 | 369 | 36 | 88 | 113 | 129 | 3 | 9.8 | +0.297 | +0.206 |
| ③ BT 2026-H2 | 255 | 28 | 49 | 86 | 90 | 2 | 11.0 | +0.201 | +0.107 |
| ③ BT tick-only | 888 | 84 | 196 | 270 | 332 | 6 | 9.5 | +0.170 | +0.075 |
| ② MASTER B18–B21 | 20 | 1 (BTW) | 5 | 6 | 8 | 0 | 5 | +0.297 | +0.207 |
| ① LIVE | 3 | 1 (EPIC) | 0 | 1 (DEEP) | 1 (DUSK) | 0 | – | −1.06 (booked) | |

- **EPIC** = peak ≥ +2.0 net ∧ (stopped ∨ ≤ −2.5).
  - 2025 has 94: LITE_EXITOPT's 93 stopped EPICs, plus ACH (12 h cap at −2.68, caught by the "≤ −2.5" clause).
  - 2026 has 64, matching LITE_EXITOPT.
- **DIPPER** = reached +2, then a 1m close ≥ 1.0 below the peak, then +3. Counts: 158 / 137, matching.

## 2. The ceiling (perfect filter, Δ %/fill vs today, skips = 0)

| block | perfect EPIC filter (resting-TP / live-slip) | perfect never-armed-loss filter | perfect all-losers filter |
|---|---|---|---|
| ③ 2025 | +0.341 / +0.363 | +1.226 / +1.308 | +1.567 / +1.671 |
| ③ 2026 | +0.309 / +0.329 | +1.059 / +1.129 | +1.368 / +1.459 |
| ③ tick-only | +0.285 / +0.304 | +1.127 / +1.202 | +1.412 / +1.506 |
| ② master | +0.150 / +0.160 | +1.205 / +1.285 | +1.355 / +1.445 |

The real filter below captures about 0.10–0.14 of the "all-losers" pot in-sample, roughly 0.06–0.09 after the haircut. Most of that is never-armed losers, not EPICs.

## 3. The screen vs the shuffled null (exhaustive; full tables in `reports/LITE_EPIC_ENTRY_tables.md` §3–4)

- **Tests.** 105 features at sign / median / two tercile cuts, giving 477 1D tests, plus every pair at the medians as 4 quadrants (21,840 2D tests).
  - Cut points are label-blind quantiles of all BT fills.
  - The zone needs at least 15 contrast fills per year.
- **Statistic.** S = min(z₂₀₂₅, z₂₀₂₆), the two-proportion z of the EPIC rate in the zone vs the rest. A test only scores high if the pattern holds in both years.
- **Null.** 500 within-year circular label rotations. These keep the day-clustering of EPICs, so the market variables are effectively read in day units. Same statistic, same search.

| contrast | best S | null median / p95 | family-wise p | best test |
|---|---|---|---|---|
| K1 EPIC vs DIPPER (the question) | 2.96 | 2.40 / 2.88 | 0.030 | BTC 1h RSI prev ≤ 55 ∧ market volume concentration ≤ 0.179 |
| K2 EPIC vs all winners | 3.12 | 2.44 / 2.97 | 0.022 | BTC 1h RSI prev ≤ 55 ∧ pair 1h EMA20/200 gap ≤ 11.3 |
| K3 EPIC vs every other fill | 3.09 | 2.46 / 2.98 | 0.030 | same |
| K3 tick-only | 3.06 | 2.47 / 3.02 | 0.038 | pair EMA20 slope > 0.29 ∧ not the first FRENZY fill of the episode (k_all > 1) |
| count of tests with S ≥ 1.96 (K1 / K2 / K3) | 34 / 37 / 35 | 9 / 12 / 12 median; 44 / 50 / 47 p95 | 0.10 / 0.09 / 0.11 | – |

- **Discover on one year → confirm on the other** (all, and tick-only): **0 of 16 confirmed**. The best 2025 pattern (z 5.2) shows z −0.1 to +0.8 in 2026.
- **Model view** (L2 logistic, all 105 features, EPIC vs all): out-of-sample AUC 0.521 (2025→2026, p 0.29) and 0.545 (2026→2025, p 0.045). The pre-registered bar needs both directions beyond the null p95, so this fails.
- **Cited variables are flat for EPIC (K3, median / sign).**
  - Signal-candle volume multiple: 11.5 vs 11.1 % in 2025, 12.5 vs 8.4 % in 2026.
  - RSI drop, k_now, the EMA13−50 stretch, vs-VWAP stretch, hours since spike and distance from the post-spike high: all |S| < 1.
  - The market-impulse score and its 5-minute turn: S −0.13 to −0.39.
  - **The "turn" that separated momentum fills does not separate LITE EPICs.**
  - BTC 1h slope ≤ 0: EPIC 15.6 vs 9.5 % in 2025 but 10.9 vs 9.9 % in 2026. This is the existing BTC1H_FALLING line; it is weak in 2026.

**Survivors** (family-wise p ≤ 0.05 in K1 / K2 / K3): 4 zones. All of them share the "BTC 1h weak" leg:
- Z1 BTC 1h RSI prev ≤ 55 ∧ pair 1h gap ≤ 11.3
- Z2 … ∧ volume concentration ≤ 0.179
- Z3 … ∧ top-10 taker-buy share 5-minute turn > −0.0039
- Z4 BTC 1h slope ≤ 0.065 ∧ pair 1h gap ≤ 11.3

## 4. Entry-filter test — "refuse when zone", three blocks (Δ %/fill vs today; resting-TP ruler, live-slip ruler in brackets)

| zone | ③ 2025 | ③ 2026-H1 | ③ 2026-H2 | ③ 2026 bar | ③ tick-only | ② master | ① live | LOMO min | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Z1 | +0.125 (+0.154) PASS | −0.055 | +0.093 | +0.006, P 0.51 **fail** | +0.004 | −0.001 (4 refused) | DEEP refused | +0.049 | fail (2026) |
| Z2 (the EPIC-vs-DIPPER survivor) | +0.157 (+0.189) PASS | −0.038 | +0.093 | +0.016, P 0.61 **fail** | +0.035 | +0.000 (2) | DEEP refused | +0.073 | fail (2026) |
| **Z3** | **+0.140 (+0.168) PASS** | **+0.108** | **+0.083** | **+0.098 (+0.126), WR 44 % < 50.2, P 0.96, 104 days, N 156 — PASS** | +0.096 (P 0.98) | **−0.451 (5 refused, 4 winners)** | DUSK refused (+2.54 saved); EPIC just outside; DEEP kept | **+0.109** | **passes the pre-registered bar; master disagrees** |
| Z4 | +0.125 PASS | −0.055 | +0.117 | +0.015, P 0.60 **fail** | +0.001 | +0.152 (5) | DEEP refused | +0.055 | fail (2026) |
| *not a survivor:* top tick-only (EMA20 slope ∧ k_all > 1) | +0.031 | +0.009 | −0.012 | +0.001 | −0.010 | unscored | unscored | +0.001 | fail |

**Z3 two-sided (BT, both years).** It refuses 348 of 1,452 fills (24 %):
- 56 EPICs (35 % of all EPICs) and 147 never-armed losers (26 %);
- against 143 winners (55 dippers and 88 others, 19–21 %).

In 2025 it avoids 32 EPICs (+95.7 points) and gives up 76 winners (−228.9 points). The never-armed losers make up the difference.

**Z3 checks.**
- Per quarter it is positive in 6 of 8 (2025Q1 −0.02; 2026Q4 −0.19 on 8 zone fills). 15 of 22 months are positive.
- Day and pair concentration stays ≤ 0.08.
- The neighbour grid of RSI 50–60 × turn −0.010 to +0.005 is positive in both years in all 20 cells (post-hoc, reported only).
- **Overlap with LITE_BTC1H_FALLING** (post-hoc). Z3 ∧ slope ≤ 0 gives +0.108 / +0.106 / +0.106 / +0.106 (2025 / 2026 / H1 / H2). Z3 ∧ slope > 0 gives +0.032 / −0.008. So Z3 is a sharper version of the existing line, which alone fails in H1 (−0.116).
- **Haircut.** Pooled +0.122 becomes **+0.061 to +0.085 %/fill** after the 30–50 % haircut.
- **Slots are not modelled.** A refused fill frees one of LITE's two slots.

## Recommendation

1. **Make no change to LITE**: no entry filter, and no change to the exit.
2. **Optional, operator decides: an OBSERVE-ONLY Scout line `LITE_WEAKBTC_BUYBLIP`.** The gate below is frozen now.
   - **Zone.** A FRENZY_LITE fill with `entry_btc_rsi_1h_prev` ≤ 55 (already stamped) ∧ top-10 taker-buy share at ⌊entry, 5m⌋ minus its value 5 minutes earlier > −0.003909.
     - The second leg is **not stamped by the engine**. It is rebuilt at batch review from the day's top-10 5m taker-buy klines, using the ML_IMPULSE `top10_tbr` definition in `scripts/study_ml_impulse_features.py`.
     - Stamping it live would be an engine change for you to decide.
     - A fill missing either value is UNSCORED, never "rest".
   - **Count.** LITE fills from the day the line is registered, priced under the live exit. Read it alongside LITE_BTC1H_FALLING, because Z3 sits inside it.
   - **🔒 Gate.** At N ≥ 15 zone fills on ≥ 8 UTC days, it becomes a FILTER CANDIDATE only on the expectancy bar:
     - WR below LITE's breakeven (≈ 50 %);
     - P(avg < 0) ≥ 0.95 by day-bootstrap;
     - no day or pair ≥ 50 % of the zone loss.
   - **Retire** the line if the zone average is ≥ the rest.
   - **Frozen: no re-fitting of 55 / −0.003909.** The 20 master fills currently argue against it (4 of 5 zone fills were winners).
3. **Stop hunting an "EPIC detector" at entry.** Exit (LITE_EXITOPT), market-at-dip (LITE_EXITOPT §M) and entry (this study) all say the same thing: EPIC and dipper are the same trade until after the dip.
   - The structural cost of EPICs (≈ 0.3 %/trade) is the price of a +3 / −3 sleeve that reaches +2 two-thirds of the time.
   - A new data source (order book, liquidations, open interest) would be the only new angle. None is cached.

## Could not test / blind spots

- **ZK** (live LITE) is in no export.
- **EPIC and DUSK** are not in any export, so none of their engine stamps exist (family A unscored). Their episode hours are also unknown, so off_spk / bars_since_hi and the C family are unscored.
  - BTC 1h RSI prev was rebuilt from cached BTC 1m. Parity vs stamps on 1,473 fills: r 0.984, median |Δ| 1.0, same side of 55 on 95.7 %.
  - The pair 1h EMA20/200 gap needs 200 1h bars, which are not cached, so Z1 / Z4 are unscored live.
- **Master** has no k_all / prior_win (paper exports do not carry the replay's sibling fills). Its gvol coverage is 65 %, and the breadth turns cover 80 %.
- **2025 non-top-50 fills (564)** have outcomes from 1m bars. Tick-only results are shown separately.
- **Fills only.** Signals the engine refused, and the slot that a refused fill frees, are not modelled.
- **Cuts.** Only medians and terciles in 1D, and medians only in 2D. No 3-way interactions.
- **Not in any cache:** order book, funding, OI, liquidations.
- **Not stamped by the engine:** the top-10 taker-buy turn, signal-candle volume multiple, and taker-buy share.

## Every headline two ways

- **Class counts.** Features pipeline (peak_all) vs the raw rows plus path-based A2_reach in the verifier: 94 / 64 EPIC, 158 / 137 DIPPER, 100 % agreement.
- **Ceiling.** +0.3407 / +0.3087 both ways.
- **Z3.** +0.1403 / +0.0979 both ways. The verifier rebuilds the top-10 turn straight from the grid.
- **Z1.** +0.1248 / +0.0057 both ways.
- **Screen counts.** Matrix algebra vs an independent pandas loop on 313 tests × 8 contrast-years: 0 mismatches.
- **Kline rebuild.** sb_ret / sb_vmult / sb_tbshare / rsi_sb / rsi_drop asserted equal to LITE_FADE_entry_rows on 1,467 fills: 0 mismatches.
- **`scripts/validate_against_master.py`: ALL CHECKS PASS.**

## Exact commands

```
venv/bin/python scripts/study_lite_epic_entry_prereg.py                  # pre-registration (refuses overwrite)
venv/bin/python -W ignore scripts/study_lite_epic_entry_features.py      # cohort, classes, 105 features, coverage (~5 s)
venv/bin/python -W ignore scripts/study_lite_epic_entry_screen.py        # 22,317 tests × 500 rotations + model view + way-2 loop (~15 s)
venv/bin/python -W ignore scripts/study_lite_epic_entry_filter.py        # classes, ceiling, 1D tables, filter test → LITE_EPIC_ENTRY_tables.md
venv/bin/python -W ignore scripts/study_lite_epic_entry_sensitivity.py   # post-hoc Z3 robustness (appends to the tables)
venv/bin/python -W ignore scripts/study_lite_epic_entry_verify.py        # way-2 headlines from raw inputs
venv/bin/python -W ignore scripts/validate_against_master.py
```

Outputs: `reports/lite_epic_entry/` (features.csv, coverage.csv, columns.json, tests.csv, null.csv, screen_summary.json, filter.json, verify.txt) · `reports/LITE_EPIC_ENTRY_tables.md`.


## Review corrections (deep review, 2026-10-15)
Conclusion holds; Z3 is NOT a survivor at adequate precision (fw p 0.057 ± 0.010 with 2,000 rotations; 0.062–0.066 under a day-block
shuffle; K3 0.53, K4 0.60; any-survivor null rate 10–12 %), its legs contradict across years (RSI ≤ 55 alone +0.207 / −0.121; turn split
2026-H1-driven), it has no value outside LITE_BTC1H_FALLING, and "taker-buy ticked up" should read "turn above its median (> −0.39 pp,
includes small declines)". Z2 is a genuine K1 survivor (0.030 / 0.040) that fails the 2026 filter test. Z3 demoted to a descriptive split.
