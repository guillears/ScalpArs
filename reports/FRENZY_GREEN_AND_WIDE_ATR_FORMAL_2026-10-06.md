# FRENZY: should WIDE drop its ATR_HIGH half, and should FRENZY_LONG take green candles? (2026-10-06)

Research only. No code, config, test or template was touched. Nothing was committed. This analysis has not been through the caveman or deep review yet, so treat every recommendation here as pending review (feedback_no_arm_before_review).
Scripts are in the scratchpad (`…/scratchpad/gw/`: `base.py`, `lib.py`, `strong.py`, `studyA.py`, `studyA2.py`, `studyB.py`, `extraB.py`, `digB.py`, `robB.py`, `strk.py`).

## 0. Engine parity (read first)

| check | result |
|---|---|
| Cohort | `reports/FRENZY_ENGINE_COHORT_2026-10-05.csv`, merged with `FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv`. Signals come from the real `frenzy_walk → frenzy_flagged → frenzy_long_status → frenzy_wide_ready` on engine bars (parity shown in the overnight review). |
| Published book reproduced | PUB live book is **981 fills, exactly** (`live_book` column). |
| Tradeable universe (TRADE) | live-tradeable pairs (no Alpha, no pairs under 90 days old, ASCII names only), live-like gvol < 1, priced. Sequenced fills: **FRENZY 204 · +0.359 %, WIDE 565 · −0.331 %**. Same as the RLC report and the overnight review. |
| Refusal codes | ATR_HIGH is judged before GREEN_BAR (`frenzy_long_status`). So an ATR_HIGH signal can also be green. The green flag is rebuilt as `bar_ret > 0`, which is the engine's `bar_red = close ≤ open`. All 462 GREEN_BAR signals are green and all 430 READY signals are red. |
| "Strong" sizing (0.5) | `strong` = `frenzy_di_spread > 0 ∧ frenzy_adx_delta > 0`, using the real functions on `closed[-300:]` from `backtest_cache/k5m_full`. **Parity check: RLC 10-05 12:00 gives +16.191 / −4.022, and the live stamp was +16.2 / −4.0.** 61 % of signals qualify as strong. |
| Exit / pricing | Live lock exit (LOCK2: −3 until +3, then max(+2, peak − 2)), 12 s entry, 0.10 slippage, 12 h cap. |
| Sizing in books | FRENZY_LONG 0.5 when strong, else 0.32. WIDE 0.2. Books start at $3,000. Live sequencing: 2 slots per sleeve, one open position per pair across sleeves, 3 entries per pair per day per sleeve. |

The main universe is **TRADE**. PUB is shown as a sensitivity check.

---

# STUDY A: should WIDE refuse the ATR_HIGH half?

Under this rule WIDE would only take FRENZY_LONG's GREEN_BAR refusals. The blocked cohort is every WIDE fill whose LONG refusal was ATR_HIGH.

## A1. The blocked cohort (TRADE, sequenced WIDE book, 1×)

| cohort | N | days | pairs | WR | mean %/fill | day 95 % CI | P(mean ≥ 0) | sum % | without top 5 / top 10 |
|---|---|---|---|---|---|---|---|---|---|
| WIDE as live | 565 | 227 | 207 | 47.6 % | −0.331 | [−0.58, −0.08] | 0.006 | −187 | −0.420 / −0.488 |
| **BLOCKED = ATR_HIGH** | **361** | **174** | 167 | **45.2 %** | **−0.525** | **[−0.83, −0.21]** | **0.001** | −189 | −0.640 / −0.723 |
| · ATR-only (red candle) | 183 | 116 | 123 | 47.5 % | −0.447 | [−0.87, −0.03] | 0.019 | −82 | −0.616 / −0.754 |
| · ATR + green (both) | 178 | 114 | 110 | 42.7 % | −0.604 | [−1.04, −0.13] | 0.004 | −107 | −0.830 / −0.988 |
| KEPT = GREEN_BAR | 204 | 131 | 117 | 52.0 % | +0.013 | [−0.44, +0.49] | 0.53 | +3 | −0.209 / −0.368 |

- **WIDE's breakeven WR** = |avg loss| / (avg win + |avg loss|):
  - 53.3 % on the WIDE book
  - 51.7 % on the kept fills
- **Other clusterings of the blocked cohort's CI** (all exclude 0):
  - episode: [−0.83, −0.22]
  - ISO week: [−0.81, −0.18] (37 weeks)
  - pair: [−0.81, −0.23]
- **PUB universe:**
  - blocked: 492 fills · −0.384 · [−0.67, −0.11]
  - ATR-only: −0.446
  - both: −0.329
  - kept: +0.132

## A2. Consistency

| blocked cohort | Jan–Apr / May–Sep | Jan–Jun / Jul–Sep | leave-one-month-out | months negative |
|---|---|---|---|---|
| all ATR_HIGH | −0.50 / −0.54 | −0.71 / −0.21 | −0.62 … −0.40 | **7 of 9** (only Feb +0.46 and Aug +0.12 positive) |
| ATR-only | −0.72 / −0.26 | −0.59 / −0.23 | −0.57 … −0.28 | 7 of 9 |
| both | −0.28 / −0.84 | −0.82 / −0.19 | −0.70 … −0.43 | 6 of 9 |

**Concentration of the blocked cohort's net loss:**
- worst pair: EVAAUSDT, **9.8 %** of the loss
- top 3 pairs: 22 %
- worst day: 2026-04-06, **8.9 %**

## A3. ATR dose-response (TRADE WIDE book)

| ATR % | ≤1.5 | 1.5–2 | 2–2.5 | 2.5–3 | 3–3.5 | 3.5–4.5 | >4.5 |
|---|---|---|---|---|---|---|---|
| all WIDE fills | +0.81 (36) | +0.08 (70) | −0.33 (98) | −0.55 (135) | **+0.13 (88)** | −0.63 (73) | −1.23 (65) |
| red | – | – | – | −0.29 (63) | −0.17 (44) | −0.61 (39) | −0.88 (37) |
| green | +0.81 | +0.08 | −0.33 | −0.78 (72) | +0.43 (44) | −0.67 (34) | −1.70 (28) |

**Moving the cut** (WIDE book, blocking ATR > x):

| x | blocked: N · mean · CI | kept: N · mean · CI |
|---|---|---|
| 2.0 | 459 · −0.48 · [−0.75, −0.21] | 106 · +0.33 · [−0.30, +0.98] |
| **2.5** | 361 · −0.53 · [−0.83, −0.20] | 204 · +0.01 · [−0.45, +0.48] |
| 3.0 | 226 · −0.51 · [−0.90, −0.12] | 339 · −0.21 |
| 3.5 | 138 · −0.92 · [−1.38, −0.43] | 427 · −0.14 |
| 4.5 | 65 · −1.23 · [−1.87, −0.60] | 500 · −0.21 |

How to read the dose-response:
- **It is not monotone above 2.5 %.** The 3.0–3.5 bucket is about flat in both colours.
- The overall slope is down: higher ATR, worse fills (trade-level Spearman −0.13). The slope runs through the whole range, including the kept half (≤1.5 +0.81 → 2–2.5 −0.33).
- **2.5 is not a special edge.** It is FRENZY_LONG's existing cap, inherited, not fitted here. Every cut from 2.0 to 4.5 gives a blocked cohort with a CI below zero.
- I did not re-fit the threshold, and 2.5 should stay frozen.

## A4. The counterfactual WIDE book (TRADE, $3,000 start, live sizing)

| book | WIDE N | WIDE mean · CI | WIDE-only book end / max DD | LONG + WIDE, shared equity, end / max DD |
|---|---|---|---|---|
| WIDE as live | 565 | −0.331 · [−0.58, −0.07] | **$410 (−86 %) / −92 %** | $1,188 / −94 % |
| WIDE green-only (lev 0.2) | 204 | +0.013 · [−0.44, +0.49] | **$2,797 (−7 %) / −44 %** | $8,082 / −66 % |
| WIDE green-only (lev 0.1) | 204 | same | – | $8,568 / −56 % |
| WIDE green-only (lev 0.05) | 204 | same | – | $8,667 / −50 % |
| WIDE off (LONG alone) | 0 | – | – | **$8,665 / −44 %** |

PUB check:
- WIDE-only book: as live $500 / −92 %; green-only $3,601 / −41 %.
- LONG + WIDE green-only: $10,655 / −68 %, against LONG alone at $8,853 / −53 %.

**In-sample haircut.** The ATR split was first seen on this same cohort (the checklist and the RLC report), so the haircut applies.
- With a 30 % haircut the blocked mean is −0.37. With 50 % it is −0.26. Both are still clearly negative.
- The WIDE book as live would still end at $700–$1,000, against $2,797 green-only.

## A5. Verdict against the locked filter bar

| bar item | required | blocked cohort | pass? |
|---|---|---|---|
| ① WR below the sleeve's breakeven WR | < 53.3 % (book) / < 51.7 % (kept) | 45.2 % | **PASS** |
| ② avg < 0 at ≥ 95 % (window-clustered bootstrap) | CI upper < 0 | −0.525, day CI [−0.83, −0.21]; episode, week and pair clusterings also below 0 | **PASS** |
| ③ ≥ 8 distinct windows; no window or pair ≥ 50 % of the loss | | 174 days; worst pair 9.8 %, worst day 8.9 % | **PASS** |
| ④ N ≥ 15 | | 361 | **PASS** |
| haircut 30–50 % | still negative | −0.37 / −0.26 | **PASS** |
| robustness | | 7 of 9 months negative; leave-one-month-out all negative; both halves negative on both splits; PUB agrees | holds |

**Verdict: "WIDE refuses ATR_HIGH" passes every item of the locked filter bar.** Both overlap parts fail on their own as well: ATR-only (red) −0.45 and ATR+green −0.60.

Honest caveats:
- **The gap to the kept half is not 95 %.** Blocked minus kept is −0.54, with a day CI of [−1.14, +0.05]. The bar judges the blocked cohort's own level, and that passes clearly. The kept half, though, is not proven better than flat.
- **The filter turns WIDE from a large loser into a flat sleeve. It does not make WIDE profitable.**
  - Kept: +0.013 · CI ±0.45.
  - At lev 0.2, green-only WIDE adds drawdown (LONG alone −44 % → −66 %) and no return ($8,665 → $8,082).
- **So the filter answers "remove the ATR_HIGH half", and the answer is yes.** Whether the green-only remainder deserves real money is a separate question. The evidence says no at 0.2 (see the recommendation).

**Proposed pre-committed revert gate** (if the operator ships it):
- The blocked ATR_HIGH WIDE signals stay priced as a zero-cost scout shadow, using the live lock exit at 12 s.
- **Re-allow ATR_HIGH only if**, after ≥ 40 shadow-priced signals on ≥ 15 distinct days, their mean is ≥ 0 %/fill.
- **Kept-half gate:**
  - If green-only WIDE fills show a mean ≤ −0.30 % after 40 fills, switch WIDE off.
  - The overnight review warns this 40-fill test is weak. It is a floor, not a promotion.
- Under D11 the switch needs its own config field, a UI toggle, load/save handlers, and a `_record_filter_block` counter (for example `FRENZY_WIDE_ATR_HIGH`). Nothing was built.

---

# STUDY B: should FRENZY_LONG take green candles?

## B1. What DECISION_LOG (180) was built on, and whether it holds on the engine cohort

- **(180)'s evidence:**
  - `scripts/frenzy_long_separators.py`: 1,103 trades
  - red +0.47 vs green −0.11
  - shuffle p 0.008, 92nd percentile of the full-search luck bar
- **Its cohort:** a hand-rolled trigger (`frenzy_long_followup.triggers`), not the engine's `frenzy_walk`:
  - EWM ATR on the full history
  - its own episode-end rule
  - no `verified` flag
  - entry at the next bar open
  - the old exit (stop 3 / trail 5 / 1.5)
- **It is not the bugged P2 "moments" cohort.** Its above-VWAP streak uses `rolling(12).min`, so it resets properly. But it was never parity-checked against the engine. Under feedback_cohort_engine_parity, its premise has to be re-tested here.

**Re-test (TRADE, unsequenced priced signals, LOCK2, 1×):**

| stratum | red: N · mean · CI | green: N · mean · CI | red − green |
|---|---|---|---|
| ATR ≤ 2.5 (READY vs GREEN_BAR) | 205 · **+0.387** · [−0.10, +0.85] | 207 · **+0.042** · [−0.40, +0.52] | **+0.345**, day CI [−0.26, +0.96], P(Δ ≤ 0) = 0.13 |
| · Jan–Apr / May–Sep | +0.52 / +0.25 | +0.27 / −0.20 | same sign in both halves |
| ATR > 2.5 (independent stratum) | 185 · −0.440 | 178 · −0.604 | +0.16, same sign |

**Candle body, % of the signal bar (mean, N):**

| stratum | ≤ −1 | −1…−0.3 | −0.3…0 | 0…0.3 | 0.3…1 | > 1 |
|---|---|---|---|---|---|---|
| ATR ≤ 2.5 | +0.39 (73) | +0.64 (71) | +0.09 (61) | +0.69 (36) | **−0.59 (77)** | +0.31 (94) |
| ATR > 2.5 | −0.45 (107) | −0.24 (57) | −0.91 (21) | −0.60 (24) | **−1.46 (35)** | −0.35 (119) |

**Reading:**
- (180)'s direction survives on the engine cohort with the lock exit. Red beats green in both halves and in both ATR strata.
- The size is about 60 % of the original (Δ +0.35 against +0.58), and it is not 95 % significant on its own.
- The body effect is **not monotone**. Small green bodies (0–0.3 %) are fine and large ones (> 1 %) are positive. The hole is the 0.3–1 % bucket, in both strata. That is a mid-range hole with winning flanks, which the anti-overfit rules treat as a confound. So no body-size filter.

## B2. Variants: LONG-only book (WIDE off), TRADE, live sizing, $3,000 start

| variant | N | days | WR | mean 1× | day CI | Jan–Apr / May–Sep | months + | leave-one-month-out | without top 5 / top 10 | top pair share | book end / max DD | added fills: N · mean · CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0 live (green skipped)** | 205 | 133 | 53.7 % | **+0.387** | [−0.10, +0.83] | +0.52 / +0.25 | 6 of 9 | +0.18 … +0.53 | +0.151 / −0.018 | 23 % | **$9,741 / −43 %** | – |
| V1 every green allowed | 405 | 190 | 52.6 % | +0.170 | [−0.18, +0.51] | +0.32 / +0.01 | 5 of 9 | +0.05 … +0.23 | +0.041 / −0.060 | 29 % | $6,524 / **−76 %** | 203 · +0.003 · [−0.45, +0.46] |
| **V2 green only when above_streak > 12** (RLC's case) | 326 | 172 | 55.5 % | +0.392 | [+0.02, +0.75] | +0.56 / +0.21 | 6 of 9 | +0.24 … +0.49 | +0.235 / +0.117 | 16 % | $17,989 / −62 % | 123 · +0.418 · [−0.18, +0.99] |
| green on a reclaim bar (streak = 12) | 285 | 157 | 49.8 % | +0.061 | [−0.33, +0.47] | +0.19 / −0.07 | 3 of 9 | −0.08 … +0.16 | −0.117 / −0.247 | – | $3,389 / −71 % | 81 · −0.664 · [−1.30, +0.05] |
| green, body 0–0.3 % | 240 | 142 | 54.2 % | +0.411 | [−0.01, +0.85] | +0.47 / +0.35 | 7 of 9 | +0.24 … +0.56 | +0.207 / +0.053 | 20 % | $13,306 / −47 % | 35 · +0.551 · [−0.56, +1.71] |
| green, body 0.3–1 % | 280 | 160 | 50.7 % | +0.104 | [−0.30, +0.51] | +0.26 / −0.07 | 6 of 9 | −0.04 … +0.19 | −0.069 / −0.195 | 63 % | $3,636 / −74 % | 77 · −0.586 · [−1.22, +0.10] |
| green, body > 1 % | 296 | 165 | 54.4 % | +0.327 | [−0.06, +0.72] | +0.52 / +0.13 | 6 of 9 | +0.17 … +0.45 | +0.152 / +0.020 | 19 % | $12,399 / −66 % | 92 · +0.255 · [−0.39, +0.92] |
| green, ATR ≤ 1.5 | 241 | 143 | 54.8 % | +0.451 | [+0.03, +0.89] | +0.60 / +0.27 | 6 of 9 | +0.27 … +0.60 | +0.241 / +0.086 | 19 % | $14,954 / −46 % | 36 · +0.810 · [−0.34, +2.14] |
| green, ATR 1.5–2 | 275 | 149 | 53.8 % | +0.309 | [−0.09, +0.71] | +0.42 / +0.19 | 5 of 9 | +0.13 … +0.38 | +0.130 / −0.007 | 21 % | $9,668 / −56 % | 70 · +0.079 |
| green, ATR 2–2.5 | 301 | 170 | 51.5 % | +0.147 | [−0.25, +0.53] | +0.31 / −0.01 | 5 of 9 | +0.03 … +0.25 | −0.019 / −0.143 | 41 % | $5,237 / −65 % | 98 · −0.328 |
| green, < 3 % above VWAP | 244 | 144 | 52.0 % | +0.203 | | +0.40 / −0.01 | 6 of 9 | | −0.009 / −0.153 | 41 % | $5,664 / −64 % | 40 · −0.539 |
| green, 3–8 % above VWAP | 275 | 153 | 52.4 % | +0.267 | | +0.33 / +0.20 | 4 of 9 | | +0.090 / −0.045 | 25 % | $8,305 / −54 % | 70 · −0.084 |
| green, > 8 % above VWAP | 297 | 167 | 54.5 % | +0.342 | | +0.51 / +0.16 | 5 of 9 | | +0.174 / +0.046 | 18 % | $12,707 / −62 % | 94 · +0.267 |
| green, 2–4 h after spike | 232 | 145 | 52.6 % | +0.331 | | +0.51 / +0.13 | 7 of 9 | | +0.110 / −0.049 | 24 % | $8,466 / −56 % | 27 · −0.096 |
| green, 4–12 h after spike | 270 | 150 | 53.3 % | +0.347 | | +0.44 / +0.25 | 6 of 9 | | +0.162 / +0.015 | 19 % | $13,487 / −42 % | 65 · +0.220 |
| green, > 12 h after spike | 314 | 167 | 53.5 % | +0.208 | | +0.38 / +0.04 | 5 of 9 | | +0.052 / −0.059 | 31 % | $6,364 / −69 % | 111 · −0.099 |
| green, gvol < 0.6 | 265 | 152 | 53.2 % | +0.326 | | +0.71 / −0.07 | 6 of 9 | | +0.130 / −0.015 | 23 % | $9,892 / −61 % | 62 · +0.160 |
| green, gvol 0.6–0.8 | 274 | 154 | 51.5 % | +0.186 | | +0.36 / +0.01 | 5 of 9 | | +0.007 / −0.130 | 36 % | $5,236 / −65 % | 69 · −0.411 |
| green, gvol 0.8–1.0 | 278 | 158 | 54.7 % | +0.343 | | +0.26 / +0.43 | 5 of 9 | | +0.167 / +0.037 | 19 % | $13,503 / −63 % | 73 · +0.220 |
| **V2 ∧ ATR ≤ 1.5** (best grid cell) | 228 | – | 56.1 % | **+0.546** | [+0.12, +0.99] | | 7 of 9 | +0.36 … +0.67 | +0.326 / +0.164 | – | **$20,592 / −39 %** | **23 · +1.959 · [+0.64, +3.42]** |

PUB gives the same ordering:
- V0: +0.304 · $9,953 / −48 %
- V1: +0.194 · $9,319 / −79 %
- V2: +0.381 [+0.06, +0.70] · $27,042 / −70 %; added fills +0.523 [−0.04, +1.07]
- V2 ∧ ATR ≤ 1.5: +0.413 · $18,486 / −44 %; added fills 25 · +1.50 [+0.22, +2.94]

**System view** (LONG variant + WIDE as live on the remaining refusals, shared equity):

| | LONG | WIDE | system end / max DD |
|---|---|---|---|
| V0 | +0.359 | −0.331 | $1,188 / −94 % |
| V1 | +0.170 | −0.525 | $961 / −96 % |
| V2 | +0.403 | −0.550 | $1,603 / −95 % |

As long as WIDE keeps its ATR_HIGH half, no LONG variant rescues the system.

## B3. Selection control

**The search:**
- 35 cells on the 207 unsequenced green signals:
  - 19 single-dimension cells: body, ATR, VWAP distance, hours since spike, gvol, strong flag, streak
  - 16 cells crossing V2 with each of them
- Statistic: cell mean, minimum N = 15.

**The two nulls use the same statistic:**
- trade-level permutation of outcomes
- circular time shift of outcomes, which keeps same-day clustering

| | observed | trade-permutation null (median / 95th) | adjusted p | circular-shift null (median / 95th) | adjusted p |
|---|---|---|---|---|---|
| best cell = V2 ∧ ATR ≤ 1.5 | +2.037 (N 24) | +0.92 / +1.67 | **0.014** | +0.92 / +1.43 | **0.006** |
| V2 (streak > 12) | +0.457 (N 125) | | **0.98** | | **0.98** |

**Out-of-sample** (pick the best cell on one half, test it on the other):
- **Jan–Apr → May–Sep:**
  - The pick was V2 ∧ gvol < 0.6. It scored +2.56 in sample and **−0.93 out of sample**.
  - V2 ∧ ATR ≤ 1.5 ranked second in sample. Out of sample it scored +2.21, but on only **5 fills**.
- **May–Sep → Jan–Apr:**
  - The pick was V2 ∧ VWAP 3–8 %: +1.26 → +0.14.
  - V2 itself: +0.25 in sample → +0.64 out of sample.

**V2 does not replicate in the independent ATR > 2.5 stratum.** Its whole logic is "green on a hold bar is fine, green on a reclaim bar is a chase". Above 2.5 % ATR:
- green hold: −0.63 (76)
- green reclaim: −0.59 (102)

That is no difference. Below 2.5 %: hold +0.46 vs reclaim −0.59. Red candles show no hold/reclaim split either: +0.35 vs +0.43.

**V2 ∧ ATR ≤ 1.5 in detail:**
- 24 signals, 21 days, 19 pairs, WR 79 %.
- Top pair: STEEMUSDT, +10.8 of +48.9.
- 7 of 8 months positive.
- Without the top 5 / top 10: +0.81 / +0.10.
- Most wins are the lock floor (+1.9).
- **It holds under every exit:** LOCK3 +2.33, FIX3 +1.66, EMA20 +2.18.
- **Neighbouring ATR cuts shrink smoothly:**
  - ≤ 1.25: +0.99 (9)
  - ≤ 1.75: +1.38 (41)
  - ≤ 2.0: +0.68 (67)
  - ≤ 2.5: +0.46 (125)
- **But red candles at ATR ≤ 1.5 show only +0.21** (28). So "low ATR rescues green" is not mirrored on red.
- Only **5 of the 24 fall after April.**
- Forward rate: **0.09 signals per day**, so 30 fills take about 11 months.

## B4. Verdict for Study B

| question | answer |
|---|---|
| Does (180)'s premise hold on the engine cohort? | **Direction yes, strength weaker.** Red beats green by +0.35 (CI spans 0), in both halves and both ATR strata. Keeping the skip is supported. Its stated evidence was weaker than reported. |
| V1 (allow every green candle) | **No.** Mean halves (+0.39 → +0.17), DD −43 → −76 %, book lower. The added greens average +0.00. |
| V2 (green only on hold bars, RLC's case) | **Not shippable.** Adjusted p 0.98 here (0.87 in the overnight review). It does not replicate in the ATR > 2.5 stratum. It raises DD (−43 → −62 %). Its added fills' CI spans 0. |
| V2 ∧ ATR ≤ 1.5 | **The only cell that survives a selection null in this search** (adj p 0.006–0.014), it holds under every exit, and its dose-response is smooth. **But:** N = 23 added fills (< 30), only 5 after April, it was found today on a cohort many studies have already searched (the true family is much bigger than 35, so the real p is higher), and it is not mirrored on red. **Pre-registered observe candidate. Not a ship.** |
| body size, VWAP distance, hours, gvol | No principled cell beats the null. The body 0.3–1 % hole is non-monotone, so no filter. |

**Recommendation: keep the green-candle skip (V0). Track V2 and V2 ∧ ATR ≤ 1.5 observe-only.**

**How to observe:**
- **No real money: the zero-cost scout shadow.** Every fresh GREEN_BAR signal is already a replay row.
- **If the operator wants real fills:** WIDE restricted to GREEN_BAR (Study A) at a **lev 0.05 probe**. Over the year this barely changes the book ($8,665 → $8,667) and moves max DD from −44 % to −50 %. Its fills are exactly the V1 / V2 signals.
- At WIDE's 0.2, green-only adds DD (−66 %) with no return. Not recommended.

**Pre-registered promotion bars** (frozen now; count only signals after 2026-10-06; live lock exit, 12 s, 0.10 slippage on shadow pricing; never re-fit):

1. **V2 → FRENZY_LONG** (GREEN_BAR ∧ above_streak > 12 ∧ ATR ≤ 2.5). Promote at normal 0.32, no strong 0.5 multiplier at first, only if ALL hold:
   - N ≥ 30 fills on ≥ 15 distinct days
   - mean ≥ +0.30 %/fill (does not dilute LONG)
   - day-bootstrap 95 % lower bound > 0
   - WR ≥ 55 %
   - no pair > 25 % of the net
   - Then apply the 30–50 % haircut, which must still be ≥ +0.15.
   - **Revert** if the first 20 promoted fills average < 0.
2. **V2 ∧ ATR ≤ 1.5 → FRENZY_LONG.** This is the Pattern-W bar:
   - N ≥ 30 forward fills
   - WR ≥ 70 %
   - mean ≥ +0.50 %
   - CI lower bound > 0
   - no pair > 25 %
   - **Revert** if the first 15 fills average < 0.
   - Expect this to take about a year at 0.09 signals per day.

---

## Combined plain-language answer

- **Study A, WIDE without its ATR_HIGH half: passes the full filter bar.**
  - Blocked cohort: N 361 · WR 45 % (breakeven 53 %) · −0.53 %/fill · day CI [−0.83, −0.21] · 174 days · worst pair 10 %. Haircut still −0.26 to −0.37. 7 of 9 months negative.
  - What is left of WIDE is flat (+0.01), not profitable.
  - So the evidence-consistent shape is WIDE = green-only at a probe size (0.05) or off. Operator decision, after review.
- **Study B, green candles for FRENZY_LONG: keep the skip.**
  - Allowing all green candles dilutes LONG and nearly doubles DD.
  - The RLC-shaped V2 is indistinguishable from luck (adj p 0.98) and does not replicate above 2.5 % ATR.
  - One cell, V2 ∧ ATR ≤ 1.5 (23 fills, +1.96), beats the null but is too small and too fresh. It goes on the observe list with the frozen bar above.

## Blind spots (what this study could NOT test)

1. **The cohort ends Sep 27.** RLC (Oct 5) and the live Oct fills are not in any statistic. RLC is one window.
2. **The strong sizing flag** is computed from `k5m_full` bars. It is parity-checked on one live fill only (RLC). Books depend on the 61 % strong share.
3. **Not modelled:** other sleeves holding the pair, the global max-open limit, `FRENZY_LATE`, outages, and the live two-stage dislocation guard (the pricer does one check at 12 s). These can only remove fills.
4. **Alpha membership history** comes from the Sep-16 and Oct-5 snapshots only.
5. **The selection family here is 35 cells.** The same cohort was searched by about 2,000 cells in the checklist and by several studies today, so the cumulative selection is not adjusted. Real p-values are higher than shown.
6. **Features not tested in Study B:**
   - upper / lower wick
   - previous-bar colour
   - BTC state (1h / 4h slope, RSI, eff72)
   - breadth
   - funding
   - pair EMA gaps
   - hour of day
   - ADX and DI as separate dimensions (only the combined "strong" flag was used)
   - Regime variables were read only through gvol, not in day units.
7. **WIDE's choppy filter** (`frenzy_wide_above_share_min`) is 0 live, so it was not applied.
8. **ATR uses only the engine's Wilder ATR on `closed[-300:]`.** The 2.5 cut is inherited from LONG. Other cuts were read but not chosen.
9. **Book paths** use the pricer's 0.10 slippage. Paper fills run about +0.10/fill better.
10. **Study A** compares absolute levels. The blocked-vs-kept difference is not 95 % (CI [−1.14, +0.05]).
