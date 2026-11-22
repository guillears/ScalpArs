# ROUTE_B_PEAK_STUDY 2026-11-21 — does any ORDER-FLOW variable change at the in-trade peak? (per sleeve)

**Question (operator):** "I still find it strange there was no finding in any sleeve of a variable that changed at the peak." Builds on STATE_EXIT 2026-11-21 (candle / 5m metrics: no state exit — precision, not timing); this study takes its stated next angle, the Route B order book + trade tape.

**Answer:** the order flow DOES change at the peak, more than chance — but the same changes happen 3–15× as often at moments that were not the top, so when a variable fires it is the real top only 2–33 % of the time (bar 60 %). The best order-flow signs are the extreme candle itself: a price-only rule ("new favourable extreme in the last 15 min") picks the top as well as any of them (0.31–0.33 on EXH / DIST). The book (depth, imbalance, spread) shows nothing at the top. At 1-minute resolution order flow says "this is a sharp move", not "this is the last sharp move". **No observe line registered.** All numbers UNVERIFIED / exploratory.

Pre-registration `reports/ROUTE_B_PEAK_STUDY_2026-11-21_PREREG.md` (written 2026-11-22 01:50 UTC, before any computation) · tables `_tables.csv`, `_screen_top.csv`, `_controls.csv`, `_cf.csv` · chart `_chart.png` · scripts `scripts/route_b_peak_1121_{data,eval,lagcheck,posthoc,tables_md}.py` (run in that order) · cache `reports/ROUTE_B_SIGNALS_data/peak_1121/` (gitignored).

## Summary (one row per sleeve)
| sleeve | REAL fills (eligible) | SHADOW priced blocked signals (eligible) | best variable: precision (hits / episodes) | its null q95 | screen survivors | verdict |
|---|---|---|---|---|---|---|
| EXHAUSTION_SHORT | 4 (3) | 62 (46) | CAPITULATION_LIQ SHADOW 0.33 (8/24); REAL 0.14 (1/7) | 0.30 | 0 of 8,033 (best 0.64 < null 0.80) | **REFUTED at this N** (SHADOW); REAL untestable yet |
| DISTRIBUTION_SHORT | 3 (2) | 77 (59) | PREMIUM_FADE SHADOW 0.26 (46/180) | 0.25 | 0 of 8,036 (best 0.31 < 0.57) | **REFUTED at this N** (SHADOW); REAL untestable yet |
| FLATBASE_SHORT | 1 (0) | 1 (1) | — | — | not run | **UNTESTABLE YET** (≈ 0.5 fills / day) |
| FRENZY_WILLY_HT | 8 (8; 4 peaked inside) | 7 (7) | PREMIUM_FADE REAL+SHADOW 0.38 (6/16) | 0.42 | 0 | **UNTESTABLE YET** (≤ 16 episodes) |
| FRENZY_WILLY | 0 | — | — | — | — | UNTESTABLE (0 fills) |
| momentum / flips | 0 | — | — | — | — | UNTESTABLE (entries off in config) |

## Method
- **Eight pre-registered order-flow variables** (frozen thresholds): TAKER_FADE, AGAINST_BURST, BOOK_FLIP, SPREAD_WIDE, CAPITULATION_LIQ, OI_NO_NEW_MONEY, BIG_AGAINST, PREMIUM_FADE; three candle comparators from STATE_EXIT (RSI_ROLL, LH2, CLIMAX).
- **Labels from each trade's 1m path:** PEAK zone = the max-favourable minute + next two closes; CONTINUATION = earlier closes (≤ peak − 3) where the trade was up ≥ max(1 %, half its final MFE) and still had ≥ 0.5 % to go. Precision = peak hits ÷ (hits + false-alarm episodes).
- **Bar:** precision ≥ 60 % on ≥ 15 episodes, ≥ 8 windows, no pair ≥ 50 % of hits, above its circular-shift null. REFUTED at this N = < 50 % on ≥ 10 episodes; UNTESTABLE YET = < 10 episodes.
- **Exhaustive screen:** 472 1D + 8,064 2D median quadrants with a family-wise null. Post-hoc price-only controls scored the same way.
- **Data:** Route B per-minute summaries (top-20 book, taker $, sampled liquidations, mark / index / predicted funding) 2026-11-19 23:00 → 11-22 01:00 UTC; OI from Binance openInterestHist 5m, used only 10 min after its label (DL 612/614). The full book / recorder OI / ratios (DL 611) start 11-21 22:00 — after the last closed fill. Downloaded ≈ 0.004 GB new from S3, 57 Binance weight. Look-ahead check: 5,357 values recomputed on panels cut at the decision minute, 0 mismatches.
- **Deviations (compute only):** 300 / 200 null shuffles instead of 1,000 (no verdict near the cut); REAL fills use their booked life (some under older TP levels).

## EXHAUSTION_SHORT — SHADOW (46 eligible, 28 peak inside)
| variable | hits | false alarms | precision | null q95 | fires at peak / cont min | verdict |
|---|---|---|---|---|---|---|
| TAKER_FADE | 6 | 204 | 0.03 | 0.06 | 0.12 / 0.22 | REFUTED |
| AGAINST_BURST | 6 | 80 | 0.07 | 0.05 | 0.10 / 0.04 | REFUTED |
| BOOK_FLIP | 1 | 12 | 0.08 | 0.27 | 0.01 / 0.01 | REFUTED |
| SPREAD_WIDE | 0 | 1 | — | — | 0 / 0 | untestable |
| CAPITULATION_LIQ | 8 | 16 | 0.33 | 0.30 | 0.23 / 0.02 | REFUTED |
| OI_NO_NEW_MONEY | 17 | 61 | 0.22 | 0.20 | 0.55 / 0.27 | REFUTED |
| BIG_AGAINST | 2 | 26 | 0.07 | 0.07 | 0.03 / 0.01 | REFUTED |
| PREMIUM_FADE | 16 | 65 | 0.20 | 0.17 | 0.53 / 0.63 | REFUTED |
| RSI_ROLL / LH2 / CLIMAX (candles) | 1 / 0 / 2 | 14 / 57 / 12 | 0.07 / 0.00 / 0.14 | | | REFUTED |
REAL (3 eligible): OI_NO_NEW_MONEY 3/11, PREMIUM_FADE 3/19, CAPITULATION_LIQ 1/7. Screen best SHADOW 7/11 = 0.64 vs null 0.80 (its counterfactual −0.05 %/fill). Price-only C_NEWEXT15 0.31 (28/91). No variable separates "went to TP" from "gave back".

## DISTRIBUTION_SHORT — SHADOW (59 eligible, 49 peak inside)
TAKER_FADE 0.02 · AGAINST_BURST 0.06 · BOOK_FLIP 0.10 · SPREAD_WIDE 0.22 (9 episodes, untestable) · CAPITULATION_LIQ 0.21 (11/52) · OI_NO_NEW_MONEY 0.23 (27/120) · BIG_AGAINST 0.05 · PREMIUM_FADE 0.26 (46/180, fires at 84 % of peak minutes and 89 % of continuation minutes) · candles 0.10 / 0.02 / 0.19 → all REFUTED. Screen best 4/13 = 0.31 vs null 0.57. Price-only C_NEWEXT15 0.33 (49/148), the best single rule. Side note (not a peak finding): a market exit at half the TP gives SHADOW +0.63 %/fill — a DIST TP-level question for the existing DIST TP shadow line, not this study.

## FLATBASE_SHORT
UNTESTABLE YET: 1 REAL fill (KERNEL, MFE +0.37 %, stopped −11.7 %), 1 SHADOW. Needs 15–30 eligible fills (6–12 weeks at ≈ 0.5 / day).

## FRENZY_WILLY_HT — REAL (8 fills; 4 peaked inside the trade)
PREMIUM_FADE 3/10 · OI_NO_NEW_MONEY 2/8 · CAPITULATION_LIQ 1/7 · TAKER_FADE 1/4 · AGAINST_BURST 1/1 · LH2 1/2; REAL+SHADOW PREMIUM_FADE 6/16 = 0.38 (null 0.42). Screen: 0 survivors (best = null). In-sample counterfactual cells (+1.4 … +2.0 %/fill on 2–4 fills) add nothing beyond the price-only "market exit at +3.03 %" (REAL +2.36) — which STATE_EXIT already refuted on the 49-fill master (TP +3: −0.73, P 0.00); this 2-day batch happens to favour it. UNTESTABLE YET.

## Pooled by direction (descriptive)
Shorts (82 eligible): CAPITULATION_LIQ 0.24, OI_NO_NEW_MONEY 0.22, PREMIUM_FADE 0.23; price-only C_NEWEXT15 0.32. Event study (81 peaks / 76 continuations, medians): at the low, opposite taker $ is 2.76× the in-trade median vs 1.07× at continuation; long liquidations $0.22k vs $0; spread 1.10× vs 1.03×; our-side taker share collapses only 3 minutes AFTER the low; book imbalance and premium show nothing. Everything is coincident with the extreme minute or lags it — nothing leads it. Longs: 9 peaks vs 9 continuations, too few.

## Verdict
- **No observe-line candidate** in any sleeve; no screen cell beats its family-wise null.
- **Refuted at this N (SHADOW, EXH + DIST):** all 8 order-flow and 3 candle variables (most beat chance, all at 2–33 % precision vs the 60 % bar). Caveat: SHADOW = blocked signals over 2–3 days, clustered in a few Frenzy episodes — a REAL-fill read is still owed.
- **Untestable yet:** every REAL cohort, all WILLY_HT, FLATBASE, plain WILLY, momentum, flips.
- **Why both the operator and the null are right:** the top minute is the most violent minute of the move; a trade that goes on to TP has several such minutes first, and a 1-minute summary can't tell the last from the earlier ones. What could still work: finer than a minute — the full book (DL 611), the side of large trades, per-second queue / imbalance.

## Data needed / the freeze week
Fill rates on the recorder: EXH ≈ 2/day, DIST 1.5, WILLY_HT 4, FB 0.5. The binding problem on shorts is precision, not N (upper bounds ≤ 0.50) — more fills won't rescue the frozen definitions, and the bar is not lowered after the fact. WILLY_HT gets a genuine first read from the freeze week (≈ 28 fills, ≈ 14 peaking inside). The full book (from 11-21 22:00) is testable for the first time during the freeze — pre-register those variables at the weekly scan. **Recommendation:** no new observe line; at the end of the freeze re-run `route_b_peak_1121_*` on REAL fills only, same frozen thresholds, as the confirmation read for WILLY_HT and EXH.

## Blind spots
Full book / recorder OI / ratios not recorded during any closed fill · nothing below 1 minute tested (no absorption / iceberg; BIG_AGAINST is a side proxy) · the liquidation stream is a sample (lower bounds) · momentum / flips / plain WILLY have 0 fills, FB 0 eligible · some REAL fills ran older TP levels; SHADOW = blocked signals, not traded fills · market-wide variables left for the weekly scan · 2 days, one regime, heavy window concentration · the counterfactual ignores in-trade funding.
