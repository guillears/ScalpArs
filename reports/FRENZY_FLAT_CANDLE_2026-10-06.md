# FRENZY: near-flat signal candles. Red, green, or their own group? (2026-10-06)

Research only. No code, config or test was touched. Nothing was committed. **Unreviewed:** no caveman or deep review has been run yet, so nothing here is a ship or arm recommendation (feedback_no_arm_before_review).
Scripts: `…/scratchpad/gw/flat.py`, `flat2.py`, which reuse `base.py` / `lib.py` from the formal study. Raw output: `…/scratchpad/gw/flat.txt`.

## 0. Engine parity (read first)

| check | result |
|---|---|
| Cohort | `FRENZY_ENGINE_COHORT_2026-10-05.csv` ⋈ `FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv` (real `frenzy_walk → frenzy_long_status → frenzy_wide_ready`). This is the same TRADE universe as the formal study. |
| Published book | PUB live book is 981 fills, exactly. |
| TRADE sequenced book | FRENZY 204 fills · +0.359 %/fill, WIDE 565 · −0.331 %. This matches the formal study. |
| Candle return in the cohort vs raw klines | `bar_ret` = (close/open − 1)·100 on the signal bar. It matches `backtest_cache/k5m_full` on **775 of 775** signals. |
| Live stamp vs Binance klines | `entry_frenzy_bar_ret_pct` matches the Binance 5 m kline (close/open of the bar before entry) on **10 of 10** batch FRENZY fills, to 4 decimals. |
| Colour rule | Engine `bar_red = close ≤ open`. An exact 0.000 % bar counts as **red**, so LONG takes it. There are 11 such bars in TRADE: 8 at ATR ≤ 2.5 and 3 above. |
| Exit / pricing | Live lock exit (LOCK2: −3 until +3, then max(+2, peak − 2)), 12 s entry, 0.10 slippage, 12 h cap, 1× for the cohort stats. |
| Books | FRENZY_LONG-only book (WIDE off). Strong 0.5 / normal 0.32 sizing. Starts at $3,000. Live sequencing. |

Cut-offs were declared before any outcome was read: ±0.05, ±0.10, ±0.20 and ±0.30 %.

---

## 1. How many signals are near-flat? (ATR ≤ 2.5, 412 priced signals)

| band | clear red | near-flat | of which flat-red (incl. exact 0) | clear green |
|---|---|---|---|---|
| ±0.05 | 192 | **17** | 13 (8) | 203 |
| ±0.10 | 183 | **32** | 22 (8) | 197 |
| ±0.20 | 162 | 59 | 43 (8) | 191 |
| ±0.30 | 144 | 97 | 61 (8) | 171 |

- **Signal-candle return quantiles** (5 / 25 / 50 / 75 / 95 %): −2.31 / −0.66 / +0.02 / +0.89 / +3.25.
- **Near-flat is rare.** About 8 % of signals fall within ±0.10.
- **Near-flat is literally a few ticks.** The median tick is 0.017 % of price.
  - FLUID's +0.046 % was **1 tick**.
  - UMA's −0.044 % was **2 ticks**.

## 2. Three groups per band (ATR ≤ 2.5, unsequenced priced signals, 1×)

| band | group | N | days | WR | mean %/fill | day 95 % CI | biggest pair (of group net) |
|---|---|---|---|---|---|---|---|
| ±0.05 | clear red | 192 | 131 | 55 % | **+0.520** | [+0.04, +0.99] | PHA +18.7 of +99.9 |
| | near-flat | 17 | 16 | 41 % | −0.844 | [−2.03, +0.54] | BTR −6.2 of −14.4 |
| | · flat-red | 13 | 13 | 31 % | **−1.575** | [−2.73, −0.04] | BTR −6.2 of −20.5 |
| | · flat-green | 4 | 4 | 75 % | +1.530 | [−1.85, +4.56] | (4 fills) |
| | clear green | 203 | 129 | 52 % | +0.013 | [−0.43, +0.50] | SAHARA +15.6 of +2.6 |
| ±0.10 | clear red | 183 | 125 | 56 % | **+0.577** | [+0.09, +1.08] | PHA +21.8 of +105.7 |
| | near-flat | 32 | 29 | 47 % | −0.456 | [−1.35, +0.46] | PHA −6.2 of −14.6 |
| | · flat-red | 22 | 21 | 36 % | **−1.193** | [−2.16, −0.12] | PHA −6.2 of −26.2 (24 %) |
| | · flat-green | 10 | 10 | 70 % | +1.165 | [−0.76, +3.05] | BTR +5.5 of +11.6 |
| | clear green | 197 | 126 | 51 % | −0.015 | [−0.48, +0.46] | SAHARA +15.6 of −2.9 |
| ±0.20 | clear red | 162 | 117 | 55 % | +0.449 | [−0.07, +0.95] | BERA +18.2 of +72.7 |
| | near-flat | 59 | 46 | 51 % | +0.172 | [−0.59, +0.99] | ENJ +10.2 of +10.2 |
| | · flat-red | 43 | 40 | 49 % | +0.157 | [−0.83, +1.21] | ENJ +8.3 of +6.7 |
| | · flat-green | 16 | 15 | 56 % | +0.214 | [−1.21, +1.69] | BTR +5.5 of +3.4 |
| | clear green | 191 | 123 | 52 % | +0.028 | [−0.45, +0.52] | |
| ±0.30 | clear red | 144 | 104 | 55 % | +0.515 | [−0.09, +1.13] | BERA +18.2 of +74.1 |
| | near-flat | 97 | 69 | 54 % | +0.310 | [−0.30, +0.96] | ENJ +10.2 of +30.1 |
| | · flat-red | 61 | 52 | 51 % | +0.088 | [−0.73, +0.97] | |
| | · flat-green | 36 | 33 | 58 % | +0.687 | [−0.43, +1.86] | GMT +9.0 of +24.7 |
| | clear green | 171 | 117 | 51 % | −0.093 | [−0.59, +0.41] | |

**Differences** (day-bootstrap 95 % CI):

| band | flat − clear red | flat − clear green | flat-red − flat-green | clear red − clear green |
|---|---|---|---|---|
| ±0.05 | −1.37 [−2.68, +0.09] | −0.86 [−2.24, +0.53] | −3.11 [−6.57, +1.01] | +0.51 [−0.11, +1.16] |
| ±0.10 | −1.03 [−2.08, +0.00] | −0.44 [−1.51, +0.64] | **−2.36 [−4.51, −0.08]** | +0.59 [−0.09, +1.24] |
| ±0.20 | −0.28 [−1.23, +0.75] | +0.14 [−0.74, +1.10] | −0.06 [−2.14, +1.85] | +0.42 [−0.23, +1.09] |
| ±0.30 | −0.21 [−1.10, +0.67] | +0.40 [−0.36, +1.17] | −0.60 [−2.17, +0.91] | +0.61 [−0.13, +1.35] |
| strict (0) | – | – | – | +0.35 [−0.26, +0.97] |

**What the groups show:**
- **The near-flat band is not one group.** Inside ±0.10, the red side is bad (−1.19, 22 fills, 21 days). The green side is good (+1.17, 10 fills).
- **So the sign inside the band seems to matter, but the opposite way to the colour rule's logic.** A barely-red candle does *worse* than a barely-green one.
- **The effect is gone at ±0.20 and ±0.30.** The 0.10–0.20 slices on each side reverse it.
- **Once the near-flat bars are removed, clear red beats clear green by about +0.5 to +0.6.** That is a little stronger than the strict-colour gap (+0.35), but still not 95 % on its own.

## 3. Fine buckets of candle return

| bucket (%) | ATR ≤ 2.5: N · mean · day CI | halves Jan–Apr / May–Sep | ATR > 2.5: N · mean |
|---|---|---|---|
| ≤ −1 | 73 · +0.39 · [−0.57, +1.31] | +0.32 / +0.45 | 107 · −0.45 |
| −1 … −0.3 | 71 · +0.64 · [−0.05, +1.36] | +1.35 / −0.32 | 57 · −0.24 |
| −0.3 … −0.1 | 39 · +0.81 · [−0.27, +1.94] | +0.46 / +1.18 | 14 · −0.52 |
| **−0.1 … 0 (incl. exact 0)** | **22 · −1.19 · [−2.16, −0.14]** | **−1.70 / −0.58** | **7 · −1.69** |
| 0 … +0.1 | 10 · +1.17 · [−0.77, +2.96] | +1.93 / +0.02 | 9 · −0.39 |
| +0.1 … +0.3 | 26 · +0.50 · [−0.84, +1.91] | −0.19 / +1.10 | 15 · −0.73 |
| **+0.3 … +1** | **77 · −0.59 · [−1.23, +0.08]** | −0.24 / −1.02 | **35 · −1.46** |
| > +1 | 94 · +0.31 · [−0.32, +0.94] | +0.65 / −0.01 | 119 · −0.35 |

**Finer slices near zero** (ATR ≤ 2.5; post-hoc, read as description only):

| slice | N · mean |
|---|---|
| exact 0.000 (a 0-tick doji) | 8 · **−1.87** (2 wins) |
| −0.05 … 0 | 13 · −1.58 |
| −0.1 … −0.05 | 9 · −0.64 |
| −0.2 … −0.1 | 21 · **+1.57** |
| +0.1 … +0.2 | 6 · −1.37 |

**By body size in ticks** (both ATR zones):

| body | N · mean |
|---|---|
| 0 ticks | 11 · −2.21 |
| −1 to −2 ticks | 13 · +0.34 |
| +1 to +2 ticks | 19 · +0.64 |

Rank correlation of candle return with outcome is −0.02 at ATR ≤ 2.5 and 0.00 above 2.5.

**What the buckets show:**
- **It is not monotone. The curve zig-zags.** Small holes sit at −0.1…0 and +0.3…+1, and +0.1…+0.2 is negative too. Each hole has winning neighbours on both sides.
- This is the pattern the anti-overfit rules call a confound or noise, not a filter.
- **The formal study's +0.3…+1 hole is confirmed in both ATR zones and both halves.** It matters to WIDE (which takes green bars), not to the LONG colour rule.
- **The −0.1…0 hole is mostly the exact-0 bars.** Without them, −0.1…0 is 14 · −0.81 [−2.04, +0.62], not significant. The −1 / −2-tick red bars are fine (+0.34).
- **The 0-tick-doji story is not a stable mechanism either.** On the green side, 1–2-tick bars win (+0.64). And the 0.1–0.2 % red bars are the best slice of all (+1.57).

## 4. Rule variants: LONG-only book (live sizing, $3k)

| variant | N | days | WR | mean 1× | day CI | book end / max DD | Jan–Apr / May–Sep | months + | leave-one-month-out | changed fills vs V0: N · mean |
|---|---|---|---|---|---|---|---|---|---|---|
| **V0 live (strict colour)** | 205 | 133 | 53.7 % | **+0.387** | [−0.08, +0.83] | **$9,741 / −43 %** | +0.516 / +0.253 | 6 of 9 | +0.18 … +0.53 | – |
| D0.05 flat counts as red | 208 | 133 | 53.8 % | +0.385 | [−0.06, +0.82] | $9,953 / −43 % | +0.541 / +0.220 | 6 of 9 | +0.17 … +0.52 | +3 · +0.22 |
| D0.10 flat counts as red | 214 | 134 | 54.2 % | +0.400 | [−0.05, +0.83] | $10,720 / −44 % | +0.548 / +0.244 | 6 of 9 | +0.17 … +0.53 | +9 · +0.69 |
| D0.20 flat counts as red | 220 | 134 | 53.6 % | +0.352 | [−0.08, +0.77] | $9,174 / −47 % | +0.511 / +0.180 | 6 of 9 | +0.13 … +0.48 | +15 · −0.14 |
| D0.30 flat counts as red | 240 | 142 | 54.2 % | +0.411 | [−0.01, +0.83] | $13,306 / −47 % | +0.475 / +0.345 | 7 of 9 | +0.24 … +0.56 | +35 · +0.55 |
| D′0.05 flat counts as green | 192 | 131 | 55.2 % | +0.520 | [+0.04, +1.00] | $13,205 / −42 % | +0.764 / +0.272 | 6 of 9 | +0.30 … +0.69 | −13 · −1.58 |
| **D′0.10 flat counts as green** | 183 | 125 | 55.7 % | **+0.577** | [+0.09, +1.07] | **$15,329 / −42 %** | +0.802 / +0.345 | 6 of 9 | +0.33 … +0.76 | −22 · −1.19 |
| D′0.20 flat counts as green | 162 | 117 | 54.9 % | +0.449 | [−0.07, +0.97] | $8,759 / −40 % | +0.640 / +0.252 | 6 of 9 | +0.15 … +0.59 | −43 · +0.16 |
| D′0.30 flat counts as green | 144 | 104 | 54.9 % | +0.515 | [−0.06, +1.11] | $8,679 / −38 % | +0.896 / +0.122 | 6 of 9 | +0.15 … +0.66 | −61 · +0.09 |

The D variants make LONG take green bars up to +w. The D′ variants make LONG skip red bars down to −w.

**Per month, mean (N):**

| variant | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|
| V0 | +1.87 (25) | −0.38 (13) | +0.27 (45) | +0.01 (22) | +0.42 (30) | +1.01 (12) | −0.97 (19) | −0.15 (21) | +1.23 (18) |
| D0.10 | +1.93 (28) | −0.58 (14) | +0.27 (45) | +0.09 (23) | +0.44 (32) | +1.01 (12) | −0.97 (19) | −0.19 (23) | +1.23 (18) |
| D0.30 | +1.49 (33) | −0.33 (15) | +0.14 (50) | +0.28 (24) | +0.61 (37) | +1.01 (12) | −1.17 (21) | +0.03 (24) | +1.25 (24) |
| D′0.10 | +2.30 (23) | +0.22 (9) | +0.53 (40) | −0.08 (21) | +0.76 (26) | +1.38 (11) | −1.12 (18) | −0.10 (19) | +1.14 (16) |
| D′0.30 | +3.67 (15) | +1.17 (7) | +0.22 (33) | −0.29 (18) | +0.47 (22) | +1.20 (9) | −0.72 (15) | −0.65 (13) | +0.57 (12) |

**Out of sample** (choose w on one half by the book's mean, test on the other half):

| direction | pick | in-sample → out-of-sample | V0 in the out-of-sample half | pick's out-of-sample rank |
|---|---|---|---|---|
| Jan–Apr → May–Sep | D′0.30 | +0.90 → **+0.12** | +0.25 | **9 of 9 (worst)** |
| May–Sep → Jan–Apr | D0.30 | +0.35 → **+0.48** | +0.52 | **9 of 9 (worst)** |

- **Choosing the band width on one half picks the worst variant in the other half, both ways.** There is no transferable width.
- D′0.10 does beat V0 in both halves (+0.80 vs +0.52, then +0.35 vs +0.25). But it is never the half-sample pick, so that is hindsight.

## 5. Selection control (8 variants)

- **Statistic:** the variant's LONG-set mean minus V0's, on the 412 unsequenced ATR ≤ 2.5 signals. The null uses the same statistic.
- **Observed Δ vs V0:**

| variant | Δ |
|---|---|
| D0.05 | +0.02 |
| D0.10 | +0.04 |
| D0.20 | −0.01 |
| D0.30 | +0.05 |
| D′0.05 | +0.13 |
| **D′0.10** | **+0.19** (best) |
| D′0.20 | +0.06 |
| D′0.30 | +0.13 |

| null | null best-of-8: median / 95th | adjusted p, best \|Δ\| | adjusted p, best improvement (one-sided) |
|---|---|---|---|
| trade permutation | 0.148 / 0.313 | **0.30** | **0.16** |
| circular time shift (keeps day clustering) | 0.153 / 0.319 | **0.33** | **0.16** |

**The best variant is what the luck of picking among 8 produces about one time in six.** This family is only today's 8. The same cohort was already searched by the formal study's 35 cells and the ~2,600-test checklist, so the true p is higher.

## 6. Locked filter bar, applied to the best candidate ("LONG skips red bars > −0.10 %")

Blocked cohort inside the V0 book: N 22 · 21 days · WR 36 % · −1.19 %/fill.

| item | required | result | pass? |
|---|---|---|---|
| ① WR below LONG's breakeven WR | < 47.7 % (avg win +3.42 / avg loss −3.12) | 36 % | pass |
| ② mean < 0 at 95 %, window-clustered | CI upper < 0 | [−2.16, −0.16] | pass (unadjusted) |
| ③ ≥ 8 windows, no window or pair ≥ 50 % of the loss | | 21 days; worst pair PHA 24 %, worst day 12 %; but the top 3 pairs carry 59 % | pass |
| ④ N ≥ 15 | | 22 | pass |
| haircut 30–50 % | still < 0 | −0.84 / −0.60 | pass |
| **selection-adjusted** | | p 0.16 | **fail** |
| **monotone dose / no mid-range hole** | | both flanks win: −0.2…−0.1 at +1.57, 0…+0.1 at +1.17 | **fail (the confound rule)** |
| **out-of-sample width choice** | | the pick is worst out of sample, both ways | **fail** |
| mechanism | | mostly 8 exact-0 bars; 1–2-tick red bars are fine | weak |

## 7. Live batch (anecdote only, not evidence)

| opened (UTC) | pair | sleeve | candle % | ticks | ATR % | result |
|---|---|---|---|---|---|---|
| 10-03 23:25 | AIN | WIDE | −1.76 | | 6.63 | −3.0 stop |
| 10-04 05:05 | SAND | LONG | −1.35 | | 1.32 | −3.0 stop |
| 10-04 11:15 | AIN | LONG | −1.21 | | 2.14 | +3.47 trail |
| 10-04 14:05 | SAND | LONG | −0.25 | | 1.24 | −3.0 stop |
| 10-05 09:15 | MOVR | WIDE | +1.31 | | 2.45 | +3.0 TP |
| 10-05 12:00 | RLC | WIDE | +0.52 | | 2.38 | +3.0 TP |
| 10-05 16:15 | AIN | WIDE | −0.32 | | 4.34 | −3.0 stop |
| **10-06 02:30** | **FLUID** | **WIDE** | **+0.046** | **1** | 1.90 | **−3.0 stop** |
| 10-06 09:40 | ORCA | LONG | −0.34 | | 1.74 | −3.0 stop |
| **10-06 10:10** | **UMA** | **LONG** | **−0.044** | **2** | 1.66 | **−3.0 stop** |

- **UMA** is in the "flat-red" slice. The backtest says that slice is weak (−1.19), so UMA fits that story.
- **FLUID** is in the "flat-green" slice. The backtest says that slice is good (+1.17, N 10), so FLUID goes against it.
- **One of each, opposite stories.** Two fills can't settle anything.

---

## Verdict (plain language)

**Keep the strict colour rule. Do not adopt a dead-band now.**

1. **Near-flat candles are not their own group, and they do not behave like "red" or like "green" as a block.** Within ±0.10 %, the barely-red ones (a 0- to 2-tick drop) did badly: −1.19 %/fill, 22 fills. The barely-green ones did well: +1.17, 10 fills. Widen the band to ±0.2 or ±0.3 and all three groups converge (+0.2 to +0.3), so the effect vanishes.
2. **The only candidate that looks good is "LONG also skips barely-red candles (> −0.10 %)".** In the replay it lifts the LONG book from $9,741 to $15,329 at the same drawdown. It also passes the four items of the locked expectancy bar on its own numbers. That is the *opposite* direction to "treat UMA-like candles as red".
3. **But it fails the controls that matter:**
   - p = 0.16 after picking the best of 8 widths
   - a mid-range hole with winning neighbours on both sides (the confound rule)
   - width chosen on one half is the worst variant on the other half, both ways
   - mostly driven by 8 exact-zero bars
4. **Colour itself is not noise at ATR ≤ 2.5, but it is weak.** Clear red beats clear green by about +0.6 %/fill (day CI about [−0.1, +1.2], P(Δ ≤ 0) ≈ 0.05). Above ATR 2.5 every bucket is negative whatever the colour, so colour is irrelevant there.
5. **Separate finding:** the +0.3 … +1 % green candle hole (Study B) re-confirms in both ATR zones and both halves. It concerns WIDE's green intake, not the LONG colour rule, and it is non-monotone (> +1 % wins again). Nothing to ship.

**Pre-registered observe candidate.** Frozen now, never re-fit. It costs nothing, because LONG fills already stamp `entry_frenzy_bar_ret_pct`.
- **Cohort:** FRENZY_LONG fills opened after 2026-10-06 12:00 UTC with −0.10 < `entry_frenzy_bar_ret_pct` ≤ 0. UMA 10-06 is the reference case and is excluded from the count.
- **Promote to "LONG skips barely-red candles" only if all of these hold:**
  - N ≥ 15 fills on ≥ 10 distinct days
  - WR below LONG's breakeven WR at that time
  - mean ≤ −0.50 %/fill with a day-bootstrap 95 % upper bound < 0
  - no pair or day ≥ 50 % of the loss
  - the same cohort's mean in the rest of LONG stays ≥ 0
- **Retire it** if the first 15 fills average ≥ 0.
- **Expected pace:** about 2.5 such fills a month in the backtest, so roughly 6 months. Live has produced 1 in 3 days.
- **If it ever ships:** under D11 it needs a config field, UI, load/save handlers and a `_record_filter_block` counter.

## Blind spots (what this could NOT test)

1. **Small cells.** Every near-flat slice has N ≤ 43, and the decisive slices are 8–22 fills. Day-bootstrap CIs on N 10–22 are fragile.
2. **Tick size is estimated**, as the smallest price step seen in 300 bars, not read from exchangeInfo. The "ticks" column is an approximation.
3. **The colour rule is judged only on the signal bar's open/close.** Not tested here:
   - wick shape (upper/lower wick)
   - signal-bar volume
   - previous-bar colour
   - whether the 0-tick doji is a low-volume / illiquid bar (possible confound with coarse-tick pairs)
4. **Regime variables** (BTC state, breadth, eff72) were not crossed with candle size. Near-flat fills sit on 21–29 distinct days, so they are not one window.
5. **The cohort ends Sep 27.** October live fills are anecdote only.
6. **The selection family counts only today's 8 variants.** Earlier searches on this same cohort are not adjusted, so the real p is higher than 0.16.
7. **WIDE's side of the swap was not booked.** A barely-red bar skipped by LONG would go to WIDE at 0.2. Books are LONG-only, as specified.
8. **Same caveats as the formal study:**
   - strong-flag parity on one live fill only
   - sequencing ignores other sleeves, the max-open limit and `FRENZY_LATE`
   - 0.10 slippage, where paper is about 0.10 better
