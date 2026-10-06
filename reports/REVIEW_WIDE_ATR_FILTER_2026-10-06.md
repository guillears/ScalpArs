# Review: "FRENZY_WIDE refuses ATR_HIGH" (Study A of FRENZY_GREEN_AND_WIDE_ATR_FORMAL_2026-10-06) (2026-10-06)

This is a read-only adversarial review. No code, config or test was touched, and nothing was committed. Every number below was re-derived with my own scripts: sequencing, bootstraps, nulls, books and parity (`…/scratchpad/watr/core.py`, `s1.py`–`s7.py`). Only the two cohort CSVs are shared with the study.

## Verdict

**The filter is valid as a way to limit damage. It is not a reason to keep WIDE trading.**

- **If WIDE trades at all, it should not take ATR_HIGH setups.** The direction is consistent:
  - 8 of 9 months show blocked worse than kept.
  - There is a mechanism: the stop is a fixed −3 %. The stop-out rate rises from 39 % at ATR ≤ 1.5 to 66 % at ATR > 4.5.
  - The one-sided evidence is p ≈ 0.02–0.04.
- **The headline "passes every item of the locked filter bar" is true but tells us almost nothing.**
  - WIDE as a whole loses −0.33 %/fill.
  - **48 % of random 361-fill subsets of WIDE pass legs ① and ② as well.** This is the Oct-1 "levels on a negative sleeve" trap (feedback_no_arm_before_review ③).
  - The statistic that matters is blocked minus kept. Its day 95 % CI spans zero, and most of the gap is *between* days and pumps, not within them.
- **Study A's "WIDE off" baseline is wrong.** On the corrected baseline, **WIDE off beats every WIDE-on variant over the year.**
- **Recommendation:**
  - **WIDE OFF** remains the evidence-preferred option, as in the overnight review.
  - If the operator wants real WIDE fills anyway: **arm the ATR_HIGH block and run green-only at the 0.05 probe size only, with the tighter gates in §6.**
  - **Not 0.2.**

---

## 1. Engine parity: PASS (exact)

| check | result |
|---|---|
| ATR in the cohort vs the engine | The cohort build (`scripts/frenzy_engine_cohort_build.py`) calls `services.surge.wilder_atr_pct(sub[-300:])` on the last 1,499 closed bars. Live does the same (`trading_engine.py:7170`, `wilder_atr_pct(closed[-300:])`), and the value is stamped as `entry_atr_pct` (`:7432`). |
| Cohort ATR recomputed from `k5m_full` | 80 random WIDE fills: max \|Δ\| 4e-16, 0 code flips, `bar_ret` identical. |
| Cohort ATR recomputed from **Binance klines** | 10 random fills: identical to 4 dp. |
| Live `entry_atr_pct` recomputed from Binance klines | **8 of 8** FRENZY / WIDE fills from Oct 3–6 are identical to 4 dp. |
| Code order | `frenzy_long_status` judges ATR_HIGH before GREEN_BAR. In the cohort: 0 ATR_HIGH fills with ATR ≤ 2.5, 0 GREEN_BAR fills with ATR > 2.5, 0 GREEN_BAR fills on a red bar. ATR_HIGH covers both colours (428 red / 499 green signals). |
| Exit | LOCK2 = live (`frenzy_lock_arm_pct` 3, floor 2, trail 2; DECISION_LOG 205). |
| Edge density | 30 WIDE fills sit at ATR 2.4–2.6, so the cut is sensitive to ATR rounding. Parity is exact, so this does not matter. |

## 2. Re-derived cohort statistics (TRADE universe, sequenced WIDE book, LOCK2, 1×)

Headline numbers reproduce exactly:

| cohort | N | days | episodes | WR | own breakeven WR | mean | day CI | episode CI | pair CI |
|---|---|---|---|---|---|---|---|---|---|
| WIDE as live | 565 | 227 | 390 | 47.6 % | 53.3 % | −0.331 | [−0.58, −0.07] | [−0.59, −0.09] | [−0.59, −0.07] |
| **BLOCKED (ATR_HIGH)** | 361 | 174 | 283 | 45.2 % | 54.3 % | **−0.525** | [−0.84, −0.21] | [−0.82, −0.22] | [−0.82, −0.23] |
| · red | 183 | 116 | 160 | 47.5 % | | −0.447 | [−0.88, −0.03] | | |
| · green | 178 | 114 | 151 | 42.7 % | | −0.604 | [−1.05, −0.13] | | |
| KEPT (GREEN_BAR) | 204 | 131 | 156 | 52.0 % | 51.7 % | +0.013 | [−0.45, +0.48] | | |

**Blocked cohort, by month:**
- Jan −0.07 · Feb +0.46 · Mar −0.23 · **Apr −0.93 (83 fills)** · May −1.42 · Jun −0.94 · Jul −0.37 · Aug +0.12 · Sep −0.38. That is 7 of 9 months negative.
- April carries 41 % of the net loss, a monthly concentration the report did not show. Even so, leaving any one month out keeps the mean between −0.62 and −0.40.

**Blocked cohort, by period:**
- Jan–Apr / May–Sep: −0.50 / −0.54
- Jan–Jun / Jul–Sep: −0.71 / −0.21. The second half is weaker.

**Blocked cohort, concentration:**
- worst pair: EVAA, 9.8 %
- top 3 pairs: 22 %
- worst day: 8.9 %
- top 5 days: 36 %
- worst episode: 4.9 %

**Kept half, by month:** 5 of 9 months negative. Jan–Apr +0.18, May–Sep **−0.17**. The kept half is decaying.

**Gap, blocked minus kept, by month:** negative in **8 of 9 months** (only August is positive). This is the strongest pro-filter fact, and the report did not show it.

## 3. Same-statistic nulls: is −0.525 unusual for a 361-fill slice of WIDE?

| test (observed statistic = null statistic) | result |
|---|---|
| Mean of 361 random fills out of 565 (20,000 draws) | null median −0.331, 5th percentile −0.491, 1st percentile −0.558 → **p = 0.024** |
| **Random 361-subsets that pass bar legs ① and ② (WR < breakeven and day CI high < 0)** | **48 %**, so the level bar is not informative here |
| Blocked minus kept, circular shift of labels in time | p = 0.018 |
| Blocked minus kept, day-block bootstrap | −0.537, 95 % [−1.11, +0.05], P(Δ ≥ 0) = 0.038 |
| Blocked minus kept, episode / pair cluster bootstrap | [−1.10, −0.01] / [−1.07, +0.01] |
| Blocked minus kept, **permutation within day** (keeps day effects) | **p = 0.30** |
| **Day fixed effects** (78 mixed days, 282 fills) | Δ −0.21, CI [−1.07, +0.59]. Only 53 % of mixed days have Δ < 0. |
| **Episode fixed effects** (49 pumps with both codes) | Δ −0.10 |
| Multivariate OLS (ATR_HIGH, above_share, gvol, listing age, hours, vs_vwap, BTC ATR, reclaim, green), day-cluster bootstrap | ATR_HIGH coefficient −0.56, CI [−1.24, +0.12] |
| Same Δ under other exits (day bootstrap) | LOCK3 −0.55 [−1.23, +0.08] · FIX3 −0.40 [−0.94, +0.14] · EMA20 −0.35 [−1.00, +0.29] |

How to read these:
- **The relative effect is real-looking at about one-sided 2–4 %.** It is not 95 % two-sided.
- **It lives mostly between days and pumps.**
  - Days with only ATR_HIGH fills average −0.47 (202 fills). Days with only green fills average +0.34 (81 fills).
  - Inside the same day or the same pump, the two codes barely differ.
- So ATR_HIGH works as a *pump-level* label. That is fine for a filter, but the evidence counts in episode units: 283 blocked episodes. The trade count (361) overstates it.

**Is 2.5 pre-specified?** Only partly.
- The cut is FRENZY_LONG's cap. But DECISION_LOG (179) chose 2.5 from ATR bands on the **same Jan–Sep year**. Its own note says it was "chosen on both sets (no unseen data left)".
- The WIDE-by-code split was also looked at after the overnight review's 2,178-cell family had already listed ATR_HIGH at −0.52.
- Mitigating point: this is one of only two natural halves of WIDE's definition, so the multiplicity cost is small. Treat it as a weakly pre-specified natural split, not an independent test.

## 4. Dose-response (WIDE book, mean, N, day CI)

| ATR % | ≤1.5 | 1.5–2 | 2–2.5 | 2.5–3 | 3–3.5 | 3.5–4.5 | >4.5 |
|---|---|---|---|---|---|---|---|
| all | +0.81 (36) | +0.08 (70) | **−0.33 (98)** | −0.55 (135) [−1.02, −0.07] | **+0.13 (88)** | −0.63 (73) | −1.23 (65) [−1.85, −0.60] |
| hold fills | **+2.19 (22)** | −0.08 | +0.20 | −0.42 | +0.07 | −0.17 | −1.19 |
| reclaim fills | −1.36 (14) | +0.33 | −1.09 | −0.67 | +0.18 | −0.91 | −1.25 |
| stop-out rate | 39 % | 46 % | 53 % | – | 45 % | 56 % | 66 % |

- The slope is real: trade-level Spearman −0.13, and the stop-out rate climbs with ATR. The fixed −3 % stop against a 5m ATR above 2.5 % is a sound mechanism.
- The response is **not monotone**: the 3–3.5 band is flat to positive.
- **2.5 is not a break.** Half of the kept side, the 2–2.5 band (98 fills), averages −0.33. The kept side's positive mean comes from ATR ≤ 1.5 hold fills.
- The cleanest region on its own is ATR > 4.5: −1.23 with a CI excluding 0.

## 5. Confounds: ATR_HIGH is its own dimension, with one important interaction

Medians, blocked vs kept:

| feature | blocked | kept |
|---|---|---|
| above_share | 77.1 | 75.0 |
| gvol | 0.71 | 0.72 |
| listing age (days) | 510 | 450 |
| BTC ATR | 0.132 | 0.134 |
| hours since spike | **7.9** | **14.8** |
| run_pct | **50 %** | **35 %** |
| off-peak | −6.3 | −2.4 |

- **It does not overlap the choppy rule.** 39 % of blocked fills and 37 % of kept fills are choppy (above_share ≤ 67.8).
  - Non-choppy: kept +0.19 (128) vs blocked −0.44 (219).
  - Choppy: kept −0.28 (76) vs blocked −0.66 (142).
  - The two filters are close to independent and both point the same way.
- **It does not proxy** gvol, listing age, BTC volatility or BTC RSI. Blocked minus kept stays negative in most terciles of each, but it is non-monotone in several (vs_vwap, off_peak, vol24, age).
- **The interaction that matters (Important):**

  | | blocked − kept |
  |---|---|
  | hold fills (above_streak > 12) | −0.36 (154) vs **+0.46 (123)** → Δ **−0.82** |
  | reclaim fills (streak = 12) | −0.65 (207) vs −0.66 (81) → Δ **+0.02** |

  The whole "kept is better" gap is the **HOLD_GREEN pocket**. The overnight review found that pocket indistinguishable from search luck (selection-adjusted p 0.87 on tradeable pairs). The two findings are not independent support for each other.

## 6. Book impact (TRADE, $3,000, live sizing, my own sequencer and shared equity)

| variant | year end / max DD | Jan–Apr | May–Sep | Jan–Jun | Jul–Sep |
|---|---|---|---|---|---|
| LONG + WIDE as live (0.2) | $1,188 / −94 % | $3,723 | $957 | $1,689 | $2,110 |
| LONG + WIDE all at 0.05 | $5,511 / −66 % | $6,160 | $2,684 | $6,107 | $2,707 |
| LONG + WIDE green-only 0.2 | $8,082 / −66 % | **$8,064** | $3,007 | $8,389 | $2,890 |
| LONG + WIDE green-only 0.1 | $8,568 / −56 % | $7,651 | $3,359 | $8,838 | $2,908 |
| LONG + WIDE green-only 0.05 | $8,667 / −50 % | $7,384 | $3,522 | $8,960 | $2,902 |
| **WIDE off (LONG re-sequenced alone)** | **$9,741 / −43 %** | $7,081 | **$4,127** | **$10,128** | $2,885 |
| paper pricing (+0.10/fill): green 0.2 / green 0.05 / off | $13,717 / −56 % · $12,740 / −40 % · $13,672 / −40 % | | | | |

- **Critical: Study A's "WIDE off $8,665 / −44 %" is not a WIDE-off book.**
  - It is the LONG fills taken from the as-live sequencing (204 fills at +0.359).
  - Re-sequenced without WIDE, LONG has 205 fills at +0.387, giving **$9,741 / −43 %**. That matches Study B's own V0 row.
  - So the report's line "green-only 0.05 barely changes the book ($8,665 → $8,667)" is wrong. The true gap is −$1,074 and 7 points more drawdown.
  - The gap comes from **one crowd-out**: a WIDE green fill on STG at 06-11 20:55 (−3.11) held the pair, so FRENZY_LONG's STG 21:00 entry (+6.16) was refused, since a pair holds one position across sleeves.
  - One event is fragile evidence, but the channel is structural. WIDE fires on the same pumps LONG trades, so any WIDE size carries this cost.
- **Green-only beats WIDE off only in Jan–Apr**, where the kept mean was +0.18. Off wins May–Sep and Jan–Jun, and Jul–Sep is a tie.
- **Haircut.** The haircut should apply to the *gap*, not the level: −0.54 becomes −0.27 to −0.38. As-live WIDE with the blocked cohort haircut 30 % / 50 % still ends at $2,032 / $2,906 with −91 % / −89 % drawdown. Blocking ATR_HIGH is better than as-live under any haircut. It is just not better than off.

## 7. Live fills (exports up to 2026-10-06 12:17, deduped on opened_at + pair)

| WIDE fill | ATR | candle | code | P&L |
|---|---|---|---|---|
| AIN 10-03 23:25 | 6.63 | red | ATR_HIGH | −3.01 SL |
| MOVR 10-05 09:15 | 2.45 | +1.31 green | GREEN_BAR | +3.00 TP |
| RLC 10-05 12:00 | 2.38 | +0.52 green | GREEN_BAR | +3.01 TP |
| AIN 10-05 16:15 | 4.34 | red | ATR_HIGH | −3.02 SL |
| FLUID 10-06 02:30 | 1.90 | +0.05 green | GREEN_BAR | −3.04 SL (above_share 44, so the choppy rule would have blocked it) |

- ATR_HIGH went 2 of 2 losers, both on one pair (AIN), so that is one observation. Green went 2 wins and 1 loss.
- The direction agrees with the study. The evidential weight is about zero.

## 8. Findings

### Critical

1. **The bar "pass" is uninformative.**
   - On a sleeve whose whole mean is −0.33, 48 % of random same-size subsets pass legs ① and ② as well.
   - The decision statistic is blocked minus kept: −0.54, with p 0.018–0.04 one-sided under same-statistic nulls. The day CI is [−1.11, +0.05].
   - Within-day permutation gives p 0.30. Day fixed effects give −0.21 [−1.07, +0.59]. The multivariate coefficient's CI spans 0.
   - The report must not say "passes every item" without this.
2. **The WIDE-off baseline is mis-built.** Study A's $8,665 should be $9,741 / −43 %. Corrected, **off ≥ every WIDE-on variant** over the year and in 3 of 4 half-periods. The "0.05 probe is free" claim is false: it costs one crowd-out, here −$1,074.

### Important

3. **The kept advantage is the HOLD_GREEN pocket.** On reclaim fills, blocked minus kept is +0.02. That pocket failed the selection null in the overnight review.
4. **Weak dose-response at the cut.** The kept 2–2.5 band is −0.33 on 98 fills, and 3–3.5 is +0.13. The cut is inherited, but (179) chose it on the same year's data, after the 2,178-cell search had already shown ATR_HIGH.
5. **The kept remainder has no edge and is decaying:** +0.013 overall, +0.18 → −0.17 by half, 5 of 9 months negative.
6. **The proposed gates are too weak.** These probabilities come from resampling kept fills at 40 fills, SD 3.25:
   - "Kept mean ≤ −0.30 after 40 → off" fires only **53 %** of the time if the true mean is WIDE's −0.33, and 29 % at a true 0.
   - "Re-allow ATR_HIGH if shadow mean ≥ 0 at 40" re-admits a true −0.26 cohort **29 %** of the time.

### Minor

7. Month-clustered CIs (9 clusters) are anti-conservative. Do not cite them.
8. April holds 41 % of the blocked loss. Leave-one-month-out still holds.
9. Episode units (283) are the honest N, not 361 fills.
10. Confounds are cleared for choppy, gvol, age and BTC volatility. ATR_HIGH fills come earlier and on bigger runs, but stratifying by hours keeps Δ < 0 in every tercile.

## 9. If the operator arms anything (pre-commit before the switch)

**Preferred: `frenzy_wide_enabled = false`.**
- Keep the scout's replay rows as a zero-cost shadow, tallying the ATR_HIGH and GREEN_BAR halves separately.

**Alternative: WIDE green-only at lev 0.05.**
- New D11 field, e.g. `frenzy_wide_block_atr_high`, plus a `_record_filter_block("FRENZY_WIDE_ATR_HIGH")` counter, UI input, and load/save handlers. The blocked signals keep their scout shadow.
- Frozen gates (count only fills after the deploy; live LOCK2; scout pricing at 0.10 slippage):
  - **Off gate:** switch WIDE off if the green-only mean is **≤ 0 after 40 fills**. This fires about 75 % of the time if the true mean is −0.33. A true 0 is not worth the crowd-out cost anyway.
  - **Size-up gate (0.05 → 0.2):** only if the mean is **≥ +0.30 after ≥ 80 fills on ≥ 8 windows**, the day CI lower bound is > 0, and no pair holds > 25 % of the net.
  - **Re-allow ATR_HIGH:** only if the shadow mean is **≥ +0.30 on ≥ 80 signals over ≥ 30 days**. Never at "≥ 0 on 40".
- Expected time: green-only fills arrive at about 0.8 per day, so the first gate takes about 50 days.

**Not supported: green-only at 0.2.** Over the year it costs $1,659 against off (−17 %) and adds 23 points of drawdown.
