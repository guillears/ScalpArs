# FRENZY_LONG: earlier/later clock (frenzy_min_hours) and "wait for the first red candle" (2026-10-06)

Research only. No code, config, test or template was touched, and nothing was committed. This analysis has not been through the caveman or deep review yet. Treat every verdict here as pending review (feedback_no_arm_before_review).

Scripts are in the scratchpad (`…/scratchpad/mh/`):
- `build.py` rebuilds the engine bars at every min-hours value.
- `stage1.py` checks parity and builds the wait candidates.
- `price_mh.py` prices the new signals.
- `stage2.py` computes gvol and the strong flag.
- `stage3.py` holds every table.
- `extra.py` and `extra2.py` hold the activation and missed-runner reads.
- `rlc/` holds the case study.

## Plain-language verdict

**Both ideas are refuted. Keep `frenzy_min_hours = 2.0` and keep the green-candle refusal as it is.**

**Study 1 (move the 2 h clock):**
- No value beats 2.0 h for FRENZY_LONG. Live 2.0 h is the best of the five on LONG mean, on the LONG-only book and on DD.
- The response is not monotone. It goes up and down across the five values, so the differences between them are noise.
- One statistic, the combined LONG + WIDE book, looks a little better at 2.5 h. That edge is a +0.11 equity-unit gain against a null whose 95th percentile is +1.12 (selection-adjusted p 0.76). Out of sample it loses to live.
- Moving the clock barely changes which trades happen. Only 10–18 of ~205 LONG fills change.

**Study 2 (wait up to N bars for a red candle instead of refusing the green one):**
- All 8 variants make FRENZY_LONG's mean worse: +0.36 → +0.13…+0.31.
- They raise LONG's own max DD from −44 % to −72…−80 %.
- None improves the combined book beyond noise. The best is +0.03 equity units at ≤1 bar / all (selection-adjusted p 0.75).
- **Why it fails (two-sided read):**
  - Waiting does buy cheaper: the median entry is 0.4–0.75 % below the green close.
  - But it systematically **filters out the runners**. The green signals that never print a qualifying red bar are the ones that keep going up. At the green bar they were worth **+0.6 to +1.2 %/trade**, and waiting never buys them.
  - What waiting does buy is the dips, at about 0 %/trade.
  - RLC is exactly this case: three more green bars, then a red bar with ATR already above the cap. Waiting would have **missed RLC entirely**.

**One finding worth keeping (not a ship):**
- The brief's premise that a clock-activated bar's colour is arbitrary is right in direction.
- The green-candle penalty (DECISION_LOG 180) lives almost entirely in **price-activated** signals, where the streak reaches 12 on the signal bar: green −0.59 vs red +0.43.
- On signals where the streak was already past 12, green ≈ red (+0.34 vs +0.31).
- The right response to that is not "wait". It is "buy the green bar on streak > 12". That is V2 from `FRENZY_GREEN_AND_WIDE_ATR_FORMAL_2026-10-06.md`, already an observe line (FRENZY_GREEN_CLOCK, commit 660576f). This study adds one more reason to keep watching it and no reason to arm it.

**Correction to the anecdote:** at 1.0 or 1.5 h, FRENZY_LONG would **not** have bought RLC at 11:35 / 0.4573.
- The setup turns on one bar earlier: the 11:30 bar, where the streak reaches 12.
- That bar's 5m ATR was 2.54 %, above the 2.5 cap, so FRENZY refuses with ATR_HIGH and WIDE takes it.
- The 11:35 bar is then no longer a fresh bar.
- No min-hours value gives FRENZY_LONG the RLC trade (§6).

## 0. Engine parity (read first)

| check | result |
|---|---|
| Signal build | Every pair of `backtest_cache/k5m_full` was walked with the real `frenzy_walk → frenzy_flagged → frenzy_long_status → frenzy_wide_ready`. On each bar the walk runs five times, once per thresholds copy, with only `frenzy_min_hours` changed. That value enters the engine only through `min_bars = max(12, round(h × 12))`. |
| 2.0 h vs published build | In-state rows **68,571 = 68,571** (same keys). Fresh signals **1,819 = 1,819**, all keys equal to `FRENZY_ENGINE_COHORT_2026-10-05.csv`. Code and sleeve equal on every row. ATR and bar return max diff **0.0**. |
| Pricer | Unchanged `frenzy_engine_cohort_price.price` (DELAY asserted = 12,000 ms, live lock LOCK2, 0.10 slip, 12 h cap). Re-pricing 40 random published keys gives max \|ΔLOCK2\| **0.0**, same outcome codes. New keys: 1,093 tick, 87 1m, 78 dislocation-refused. The sequenced books are 96–98 % tick-priced at every value. |
| Market-volume gate (gvol) | The overnight review's live-like U2 ruler is recomputed for every new bar with the same code. On 25 existing bars, max diff **2e-16**. |
| Universe | Live filters: onboarded ≥ 90 days, not Alpha, ASCII names. Blacklists applied. Cohort ends Sep 27. |
| Sequenced TRADE book at 2.0 h | **FRENZY 204 · +0.359 %, WIDE 565 · −0.331 %.** Identical to the overnight review and the green/ATR study. |
| Sizing | LONG 0.32, or 0.5 when strong (di_spread > 0 ∧ adx_delta > 0 on the entry bar, real functions). WIDE 0.2. Books start at $3,000 with shared equity. Live sequencing: 2 slots per sleeve, one open position per pair across sleeves, 3 entries per pair per day per sleeve. |

---

# STUDY 1: `frenzy_min_hours` ∈ {1.0, 1.5, 2.0 (live), 2.5, 3.0}

## 1a. Activation mix (fresh signals on tradeable pairs, priced, before gvol)

| min hours | signals | LONG (READY) | WIDE | GREEN_BAR | ATR_HIGH | price-activated (streak = 12) | clock bar (streak > 12 on the min-hours bar) | volume / re-on (streak > 12, later bar) | LONG signals with streak > 12 | GREEN_BAR with streak > 12 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | 1406 | 358 | 1048 | 397 | 651 | 54 % | 5 % | 41 % | 53 % | 60 % |
| 1.5 h | 1376 | 356 | 1020 | 385 | 635 | 50 % | 9 % | 41 % | 55 % | 62 % |
| **2 h (live)** | 1340 | 352 | 988 | 372 | 616 | 48 % | **10 %** | 42 % | 55 % | 63 % |
| 2.5 h | 1285 | 347 | 938 | 366 | 572 | 47 % | 10 % | 43 % | 56 % | 64 % |
| 3 h | 1255 | 343 | 912 | 359 | 553 | 46 % | 11 % | 43 % | 57 % | 65 % |

**"Clock-activated" in the brief's sense (streak > 12) is mostly not the clock.**
- Only ~10 % of signals turn on exactly on the min-hours bar.
- About 42 % turn on later with the streak long past 12. There the switch is the last-hour volume crossing 100× normal again, or a re-start after ≥ 1 h off.
- Both kinds have a signal candle whose colour is not set by the price trigger.

## 1b. FRENZY_LONG, system view (WIDE as live), LOCK2, 12 s, slip 0.10

| min hours | N | days | WR | mean %/trade (1×) | day 95 % CI | book from $3k | max DD | Jan–Apr / May–Sep | Jan–Jun / Jul–Sep | leave-one-month-out | months > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | 206 | 128 | 52.9% | +0.289 | [-0.14, +0.73] | $7,103 | -46% | +0.35 / +0.23 | +0.40 / +0.01 | +0.11 … +0.42 | 5 of 9 |
| 1.5 h | 207 | 129 | 52.2% | +0.244 | [-0.19, +0.69] | $6,461 | -50% | +0.34 / +0.15 | +0.38 / -0.08 | +0.08 … +0.37 | 5 of 9 |
| **2 h (live)** | 204 | 132 | 53.4% | **+0.359** | [-0.10, +0.83] | **$8,665** | **-44%** | +0.52 / +0.19 | +0.50 / +0.01 | +0.15 … +0.50 | 6 of 9 |
| 2.5 h | 204 | 128 | 52.9% | +0.302 | [-0.14, +0.75] | $7,846 | -52% | +0.49 / +0.10 | +0.50 / -0.18 | +0.09 … +0.43 | 5 of 9 |
| 3 h | 196 | 125 | 50.0% | +0.128 | [-0.33, +0.58] | $3,991 | -66% | +0.31 / -0.06 | +0.28 / -0.23 | -0.10 … +0.26 | 5 of 9 |

## 1c. FRENZY_LONG with WIDE off

| min hours | N | WR | mean | day CI | book | max DD | Jan–Apr / May–Sep | LOMO |
|---|---|---|---|---|---|---|---|---|
| 1 h | 207 | 53.1% | +0.317 | [-0.11, +0.75] | $7,985 | -46% | +0.35 / +0.29 | +0.14 … +0.45 |
| 1.5 h | 208 | 52.4% | +0.273 | [-0.17, +0.72] | $7,264 | -49% | +0.34 / +0.21 | +0.11 … +0.40 |
| **2 h (live)** | 205 | 53.7% | **+0.387** | [-0.07, +0.85] | **$9,741** | **-43%** | +0.52 / +0.25 | +0.18 … +0.53 |
| 2.5 h | 205 | 53.2% | +0.330 | [-0.10, +0.79] | $8,820 | -52% | +0.49 / +0.16 | +0.12 … +0.46 |
| 3 h | 197 | 50.3% | +0.159 | [-0.32, +0.61] | $4,487 | -62% | +0.31 / +0.00 | -0.07 … +0.30 |

## 1d. WIDE (as live, 0.2) and the combined book

| min hours | WIDE N | WIDE mean | WIDE day CI | WIDE book | combined N | combined mean | combined book | combined max DD | combined months > 0 |
|---|---|---|---|---|---|---|---|---|---|
| 1 h | 590 | -0.401 | [-0.63, -0.16] | $253 | 796 | -0.223 | $601 | -95% | 3 of 9 |
| 1.5 h | 589 | -0.294 | [-0.55, -0.04] | $464 | 796 | -0.154 | $1,001 | -95% | 3 of 9 |
| **2 h (live)** | 565 | -0.331 | [-0.58, -0.08] | $410 | 769 | -0.148 | $1,188 | -94% | 5 of 9 |
| 2.5 h | 542 | -0.302 | [-0.56, -0.04] | $517 | 746 | -0.137 | $1,356 | -92% | 4 of 9 |
| 3 h | 507 | -0.357 | [-0.62, -0.10] | $445 | 703 | -0.222 | $592 | -95% | 3 of 9 |

- WIDE is a loser at every value; the overnight review's verdict does not depend on the clock.
- The combined book is under water at every value because WIDE drags it down.

## 1e. FRENZY_LONG per month (system), mean % (N)

| min hours | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep |
|---|---|---|---|---|---|---|---|---|---|
| 1 h | +1.59 (25) | -0.38 (13) | +0.12 (45) | -0.18 (21) | +0.54 (31) | +0.54 (11) | -0.97 (19) | -0.15 (21) | +1.11 (20) |
| 1.5 h | +1.41 (26) | -0.22 (14) | +0.13 (44) | -0.18 (21) | +0.45 (31) | +0.54 (11) | -0.97 (19) | -0.15 (21) | +0.84 (20) |
| **2 h** | +1.87 (25) | -0.38 (13) | +0.27 (45) | +0.01 (22) | +0.42 (30) | +0.54 (11) | -0.97 (19) | -0.15 (21) | +1.23 (18) |
| 2.5 h | +1.82 (25) | -0.57 (14) | +0.40 (46) | -0.18 (21) | +0.58 (28) | +0.40 (10) | -0.97 (19) | -0.52 (24) | +1.19 (17) |
| 3 h | +1.79 (24) | -0.79 (11) | +0.02 (44) | -0.18 (21) | +0.24 (27) | +0.08 (11) | -1.07 (20) | -0.29 (22) | +0.90 (16) |

## 1f. What actually changes vs live (LONG fills, system)

| min hours | LONG fills added | LONG fills dropped |
|---|---|---|
| 1 h | 13 · WR 46 % · −0.50 [−1.93, +1.11] | 11 · WR 55 % · +0.66 [−1.34, +2.73] |
| 1.5 h | 14 · WR 36 % · −1.10 [−2.40, +0.45] | 11 · WR 55 % · +0.66 [−1.39, +2.70] |
| 2.5 h | 13 · WR 54 % · −0.05 [−1.55, +1.54] | 13 · WR 62 % · +0.85 [−0.92, +2.57] |
| 3 h | 10 · WR 10 % · −2.61 [−3.12, −1.61] | 18 · WR 67 % · +1.23 [−0.32, +2.65] |

**Each move touches 10–18 trades, and every move swaps live winners for weaker fills.** The 3 h row is the only directional signal: waiting an extra hour mostly buys tops (9 of the 10 added fills lose). That argues against a later clock, not for an earlier one.

## 1g. Judgement: monotone? selection-adjusted? out of sample?

**Selection-adjusted p.** The statistic is (best value − live). The null circularly shifts the outcomes along the time-ordered union of all fills, keeping each value's membership fixed, with 2,000 shifts and the same statistic on both sides.

| statistic | 1.0 | 1.5 | 2.0 (live) | 2.5 | 3.0 | monotone? | best | best − live | null 95th pct | adj. p |
|---|---|---|---|---|---|---|---|---|---|---|
| LONG mean %/trade (system) | +0.289 | +0.244 | **+0.359** | +0.302 | +0.128 | no (− + − −) | 2 h (live) | 0 | +0.189 | **1.00** |
| combined book, additive Σ size × P&L (equity units) | -0.960 | -0.463 | -0.295 | -0.182 | -1.029 | no (+ + + −) | 2.5 h | +0.114 | +1.115 | **0.76** |
| LONG-only book (WIDE off), additive | +1.389 | +1.286 | **+1.584** | +1.480 | +0.797 | no (− + − −) | 2 h (live) | 0 | +0.602 | **1.00** |

**Out of sample.** Pick the best value on one half (Jan–Apr | May–Sep) and score it on the other:

| statistic | direction | picked (in-sample) | picked OOS | live OOS | Δ OOS |
|---|---|---|---|---|---|
| LONG mean | Jan–Apr → May–Sep | 2 h (+0.516) | +0.193 | +0.193 | 0 |
| LONG mean | May–Sep → Jan–Apr | 1 h (+0.228) | +0.349 | +0.516 | −0.167 |
| combined book | Jan–Apr → May–Sep | 2 h | −0.839 | −0.839 | 0 |
| combined book | May–Sep → Jan–Apr | 2.5 h | +0.391 | +0.543 | −0.152 |
| LONG-only book | Jan–Apr → May–Sep | 2.5 h | +0.399 | +0.513 | −0.114 |
| LONG-only book | May–Sep → Jan–Apr | 1 h | +0.755 | +1.071 | −0.317 |

**Verdict:**
- Non-monotone on every statistic. Live is the in-sample best on two of three statistics.
- The third statistic's winner (2.5 h) is at chance (p 0.76).
- No out-of-sample pick beats live.
- **Refuted. Keep 2.0 h.**

---

# STUDY 2: wait up to N bars for the first non-green candle (min hours 2.0)

**Rule as tested:**
- A fresh FRENZY setup whose only refusal is the green candle (code GREEN_BAR, so ATR ≤ 2.5 on the signal bar) becomes **pending** instead of refused.
- The bot then looks at the next N bars. On the **first** bar with close ≤ open, it enters 12 s after that bar's close if all of these hold on that bar:
  - the setup is still ON (and stayed ON continuously since the signal)
  - ATR ≤ 2.5
  - 24 h volume ≥ $20M
  - gvol < 1
  - the price is not dislocated
- If that first red bar fails a condition, or the setup goes OFF first, the episode is dropped. There is no second chance, as live.
- **While pending, WIDE does not take the green setup**, because FRENZY did not refuse it.
- WIDE still takes the ATR_HIGH refusals.
- Scope (a) is every GREEN_BAR. Scope (b) is GREEN_BAR with streak > 12, the brief's "clock-activated".
- N = 1, 3, 6, 12 was frozen before running.

## 2a. Funnel (green signals on tradeable pairs)

| scope | green signals | first red at +1 / +2–3 / +4–6 / +7–12 | setup OFF before any red | first red fails ATR cap | no red in 12 |
|---|---|---|---|---|---|
| all | 393 | 194 / 137 / 31 / 1 | 30 | 49 | 0 |
| streak > 12 | 250 | 126 / 84 / 25 / 0 | 15 | 38 | 0 |

The first red bar almost always comes within 6 bars, so N = 6 and N = 12 are the same rule in practice.

## 2b. FRENZY_LONG book per variant (system: LONG + WIDE as live on what is left)

| variant | N | days | WR | mean %/trade (1×) | day 95 % CI | LONG book from $3k | max DD | Jan–Apr / May–Sep | Jan–Jun / Jul–Sep | LOMO | months > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **live (green refused → WIDE)** | 204 | 132 | 53.4% | **+0.359** | [-0.08, +0.81] | **$8,665** | **-44%** | +0.52 / +0.19 | +0.50 / +0.01 | +0.15 … +0.50 | 6 of 9 |
| wait ≤1 · all | 304 | 161 | 53.0% | +0.248 | [-0.16, +0.64] | $7,660 | -77% | +0.42 / +0.08 | +0.47 / -0.28 | +0.03 … +0.36 | 5 of 9 |
| wait ≤3 · all | 369 | 180 | 52.0% | +0.139 | [-0.22, +0.48] | $4,950 | -77% | +0.19 / +0.09 | +0.31 / -0.27 | -0.04 … +0.25 | 5 of 9 |
| wait ≤6 · all | 382 | 181 | 51.8% | +0.128 | [-0.21, +0.47] | $4,515 | -80% | +0.22 / +0.03 | +0.32 / -0.33 | -0.04 … +0.23 | 5 of 9 |
| wait ≤12 · all | 382 | 181 | 51.8% | +0.128 | [-0.20, +0.47] | $4,515 | -80% | +0.22 / +0.03 | +0.32 / -0.33 | -0.04 … +0.23 | 5 of 9 |
| wait ≤1 · streak > 12 | 262 | 149 | 53.1% | +0.306 | [-0.11, +0.73] | $8,446 | -72% | +0.48 / +0.13 | +0.50 / -0.15 | +0.06 … +0.41 | 5 of 9 |
| wait ≤3 · streak > 12 | 300 | 163 | 52.7% | +0.222 | [-0.17, +0.62] | $6,332 | -72% | +0.36 / +0.07 | +0.41 / -0.23 | -0.01 … +0.32 | 5 of 9 |
| wait ≤6 · streak > 12 | 310 | 164 | 52.6% | +0.221 | [-0.15, +0.60] | $6,318 | -76% | +0.40 / +0.02 | +0.41 / -0.26 | +0.01 … +0.32 | 5 of 9 |
| wait ≤12 · streak > 12 | 310 | 164 | 52.6% | +0.221 | [-0.15, +0.60] | $6,318 | -76% | +0.40 / +0.02 | +0.41 / -0.26 | +0.01 … +0.32 | 5 of 9 |

## 2c. Added fills, what WIDE gives up, combined book

| variant | added LONG fills (after sequencing) | WR | mean | day CI | live WIDE fills given up (mean) | WIDE N left | combined book | combined DD | combined Δ vs live (additive, equity units) |
|---|---|---|---|---|---|---|---|---|---|
| live | – | | | | – | 565 | $1,188 | -94% | 0 |
| ≤1 · all | 100 | 51% | -0.072 | [-0.72, +0.59] | 204 (+0.013) | 361 | $1,124 | -96% | +0.030 |
| ≤3 · all | 165 | 50% | -0.189 | [-0.71, +0.33] | 204 (+0.013) | 361 | $724 | -96% | -0.298 |
| ≤6 · all | 179 | 50% | -0.154 | [-0.61, +0.34] | 204 (+0.013) | 361 | $660 | -96% | -0.367 |
| ≤12 · all | 179 | 50% | -0.154 | [-0.61, +0.33] | 204 (+0.013) | 361 | $660 | -96% | -0.367 |
| ≤1 · streak > 12 | 58 | 50% | -0.040 | [-0.93, +0.86] | 123 (+0.458) | 442 | $721 | -97% | -0.438 |
| ≤3 · streak > 12 | 96 | 50% | -0.167 | [-0.82, +0.46] | 123 (+0.458) | 442 | $540 | -97% | -0.669 |
| ≤6 · streak > 12 | 107 | 50% | -0.075 | [-0.71, +0.54] | 123 (+0.458) | 442 | $539 | -97% | -0.651 |
| ≤12 · streak > 12 | 107 | 50% | -0.075 | [-0.70, +0.55] | 123 (+0.458) | 442 | $539 | -97% | -0.651 |

- The added fills average ≈ −0.04 to −0.19 %/trade, with every CI spanning 0.
- For streak > 12, the green setups WIDE takes today are WIDE's best pocket: +0.46 %/trade, the HOLD_GREEN cell of the overnight review. Waiting hands those to a rule that earns about −0.08.

## 2d. Two-sided: what waiting costs and what it saves

**Every waited entry, unsequenced, compared with buying the same signal on the green bar** (12 s after its close, same exit):

| variant | entries | entry vs green close (median / mean) | waited mean | same signals at the green bar | Δ | green winner → wait loser (N, Σ pts) | green loser → wait winner (N, Σ pts) |
|---|---|---|---|---|---|---|---|
| ≤1 · all | 97 | -0.75 % / -1.08 % | +0.002 | -1.022 | +1.02 | 0, +0 | 17, +97 |
| ≤3 · all | 163 | -0.52 % / -0.48 % | -0.135 | -0.363 | +0.23 | 15, -85 | 21, +118 |
| ≤6 · all | 174 | -0.45 % / -0.22 % | -0.135 | -0.109 | -0.03 | 20, -122 | 21, +118 |
| ≤12 · all | 174 | -0.45 % / -0.22 % | -0.135 | -0.109 | -0.03 | 20, -122 | 21, +118 |
| ≤1 · streak > 12 | 58 | -0.67 % / -1.11 % | +0.120 | -0.726 | +0.85 | 0, +0 | 8, +45 |
| ≤3 · streak > 12 | 98 | -0.42 % / -0.43 % | -0.082 | -0.020 | -0.06 | 12, -70 | 11, +60 |
| ≤6 · streak > 12 | 107 | -0.37 % / -0.07 % | -0.078 | +0.321 | -0.40 | 16, -102 | 11, +60 |
| ≤12 · streak > 12 | 107 | -0.37 % / -0.07 % | -0.078 | +0.321 | -0.40 | 16, -102 | 11, +60 |

**The green signals that waiting never buys** (tradeable, gvol < 1 at the green bar; their value if bought at the green bar):

| variant | green signals | never entered | their mean at the green bar | day CI | of which ≥ +3 % |
|---|---|---|---|---|---|
| ≤1 · all | 207 | 121 | +0.597 | [+0.05, +1.14] | 20 |
| ≤3 · all | 207 | 44 | +1.110 | [+0.10, +2.16] | 10 |
| ≤6 · all | 207 | 37 | +0.672 | [-0.39, +1.81] | 7 |
| ≤1 · streak > 12 | 125 | 74 | +1.181 | [+0.51, +1.81] | 14 |
| ≤3 · streak > 12 | 125 | 29 | +1.222 | [-0.01, +2.45] | 7 |
| ≤6 · streak > 12 | 125 | 24 | +0.634 | [-0.55, +1.99] | 4 |

**Reading:**
- Waiting does what it promises on price: entries are 0.4–0.75 % cheaper (median).
- **The ≤1-bar case looks good per entry** (+1.0 vs the green bar). That comparison is selected: the green bars immediately followed by a red bar were bad green-bar buys (−1.0).
- **The bill comes from the other side.** The green setups that keep printing green, or go straight into an ATR blow-off, are the runners: +0.6 to +1.2 %/trade, with up to 20 trades of ≥ +3 %. A wait-for-red rule cannot buy them by construction.
- Under the live design, WIDE (0.2) buys them. Under the variant, nobody does.

## 2e. Selection-adjusted p over the 8 variants

The test is paired per green signal: (variant contribution − live contribution) in equity units. The null flips the sign by day block, 4,000 draws, using the same max-statistic.

| variant | Σ Δ (unsequenced) |
|---|---|
| ≤1 · all | **−0.032** (best) |
| ≤3 · all | −0.386 |
| ≤6 / ≤12 · all | −0.455 |
| ≤1 · streak > 12 | −0.538 |
| ≤3 · streak > 12 | −0.795 |
| ≤6 / ≤12 · streak > 12 | −0.777 |

- The best variant is still below zero.
- The null's 95th percentile of the max is +1.44.
- **Selection-adjusted p = 0.75.**
- Per-month added fills are mixed in every variant: Aug is −1.6 to −2.2 everywhere, Jan is positive everywhere. That is regime, not rule.
- **Refuted.**

## 2f. Sensitivity, outside the frozen 8: true clock bars only (streak > 12 on exactly the 2.0 h bar)

- Tradeable green signals of this kind: **8** in the whole year (at the green bar: +2.19).
- Waiting entered 2–3 of them (+5.2), but N = 3 is anecdote-sized.
- Recorded only so that nobody re-runs it as a new idea.

## 2g. Where the green-candle penalty actually lives (ATR ≤ 2.5, tradeable, 2.0 h, unsequenced)

| activation | signals | green share | red-bar mean (CI) | green-bar mean (CI) |
|---|---|---|---|---|
| price (streak = 12 on the signal bar) | 186 | 44 % | +0.425 [-0.23, +1.11] | **-0.590 [-1.26, +0.14]** |
| clock bar (streak > 12 at exactly 2.0 h) | 19 | 42 % | +0.660 [-1.34, +2.63] | +2.187 [-0.60, +5.25] |
| volume / re-on (streak > 12, later bar) | 207 | 57 % | +0.311 [-0.45, +1.01] | +0.339 [-0.29, +0.93] |

- The DECISION_LOG 180 green skip is right where it was designed to bite: a green candle that **is** the reclaim bar is a chase.
- Where the streak is already long, colour carries no information. This is the existing V2 observe line (FRENZY_GREEN_CLOCK), and it is unchanged by this study.
- That line's earlier verdict, adjusted p 0.98, is still the binding fact.

---

# Combined decision table

| row | FRENZY_LONG (N · mean · day CI) | WIDE (N · mean) | LONG book / DD | combined book / DD | evidence vs the locked gates | RLC 10-05 (anecdote, §6) |
|---|---|---|---|---|---|---|
| **live: 2.0 h, green refused → WIDE** | 204 · +0.359 · [−0.09, +0.82] | 565 · −0.331 | $8,665 / −44 % | $1,188 / −94 % | reference | WIDE 12:00 at 0.4648 → lock +5.2 % |
| best of Study 1, LONG statistics = live (2.0 h) | same | same | same | same | live is the best of 5; adj p 1.00 | same |
| best of Study 1, combined-book statistic = 2.5 h | 204 · +0.302 · [−0.15, +0.74] | 542 · −0.302 | $7,846 / −52 % | $1,356 / −92 % | non-monotone; adj p 0.76; OOS −0.15 vs live; WIDE still −0.30 | WIDE 12:30 at 0.5388 → lock −3.3 % |
| best of Study 2 = wait ≤1 · all | 304 · +0.248 · [−0.15, +0.64] | 361 · −0.525 | $7,660 / −77 % | $1,124 / −96 % | added fills −0.07 [−0.72, +0.59]; adj p 0.75; DD +33 pts | **no trade** (12:00 still green) |
| both: 2.5 h + wait ≤1 · all | 302 · +0.144 · [−0.24, +0.52] | 342 · −0.427 | – | $1,099 / −95 % | worse than either alone | WIDE 12:30 → −3.3 % (ATR_HIGH, not a green refusal) |

**Recommendation: refuted, both. No ship, no new scout line.**
- Study 1 has no candidate: live is the best value, and the non-monotone curve says the rest is noise.
- Study 2 fails every leg:
  - LONG mean down
  - DD nearly doubled
  - added fills ≈ 0 or below
  - adj p 0.75
- Study 2 also loses the very kind of trade (RLC) that motivated it. Do not add a scout line for it; it would track a rule whose mechanism is already understood.
- The existing FRENZY_GREEN_CLOCK observe line already covers the useful part of the idea: buying the green bar when the streak is long. Keep it as frozen.

No gate needs pre-registering because nothing is proposed. If the operator still wants to revisit the clock, the only defensible re-test is forward data at the next batch review on the observe line that already exists. Do not re-fit the clock on this year.

# Blind spots (not tested, or tested only approximately)

- **Live market-volume history.** gvol is the overnight review's live-like U2 ruler, rebuilt from cached klines. Live's own cache refresh timing is not modelled.
- **Shortlist cap.** The 25-pair cap was checked only at 2.0 h (569 of 570 reachable in the overnight review). Earlier clocks shift signals ≤ 1 h earlier on the same episodes, so the effect should be similar, but this is not re-checked.
- **Normal-hour cache TTL, the two-stage dislocation guard, FRENZY_LATE / outages, other sleeves holding the pair, and global max-open:** not modelled (same as the overnight review). They can only remove fills.
- **Wait-rule design choices.** These are my choices, frozen before the run:
  - It enters on the first red bar only.
  - The setup must stay ON continuously.
  - WIDE stands aside while pending.
- **Two alternatives are not run:**
  - **"Keep waiting for a red bar that passes."** Its upper bound is small: only 49 of the 393 green signals have a first red that fails the ATR cap, and on RLC the ATR stays above the cap for the whole window anyway.
  - **"WIDE still buys the green bar and LONG waits."** Under the one-position-per-pair rule, LONG could then enter only after WIDE exits. That is a re-entry design, which earlier re-entry studies refuted.
- **Sign-flip null (Study 2)** assumes the paired differences are symmetric under "no effect". **Circular-shift null (Study 1)** keeps the overlap structure but not the exact sequencing interactions.
- **Day-block CIs.** No window clustering is used beyond day blocks.
- **Exit and data scope.** Only the LOCK2 exit is tested. The year cohort ends Sep 27; Oct live fills are not in any statistic.
- **Alpha membership is today's**, as in the overnight review.

---

# 6. Case study: RLCUSDT 2026-10-05 (illustration only, out of cohort, excluded from every statistic and recommendation)

**Data:**
- Public Binance 5m and 1h klines, run through the real `frenzy_walk / frenzy_long_status / frenzy_wide_ready` with the thresholds copy per min-hours value.
- **ATR parity:** the live WIDE fill stamped ATR 2.3775 on the 11:55 bar; this walk gives 2.3775.

**Prices:**
- Entries come from public aggTrades (ticks) at signal close + 12 s.
- The exit path uses ticks until 13:30 UTC and 1m bars after that.
- Every exit below happened inside the tick window, so none is provisional.
- "Run-up still ahead" is the highest 1m high from the entry to 2026-10-06 12:45 UTC, relative to the entry price. The peak was at 10-06 11:05.

**Caveats:**
- gvol was not recomputed for RLC bars other than 12:00 (live 0.684 there). It is assumed below 1 at the other bars.
- Sizes: WIDE 0.2. FRENZY_LONG would be 0.32; RLC was not "strong" at 11:55.

## 6a. Bar by bar, 11:20–13:10 UTC (bar open times)

The "status" columns show what the engine says on that bar at each min-hours value.

| bar open | closes | close | bar % | colour | streak | hours since spike | ATR % | status at 1.0 / 1.5 h | status at 2.0 h (live) | at 2.5 h | at 3.0 h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 11:20 | 11:25 | 0.4653 | -0.79 | red | 10 | 1.42 | 2.39 | BELOW_AVG / BELOW_AVG | BELOW_AVG | BELOW_AVG | BELOW_AVG |
| 11:25 | 11:30 | 0.4697 | +0.92 | green | 11 | 1.50 | 2.35 | BELOW_AVG / BELOW_AVG | BELOW_AVG | BELOW_AVG | BELOW_AVG |
| **11:30** | 11:35 | 0.4586 | -2.36 | red | **12** | 1.58 | **2.54** | **ATR_HIGH / ATR_HIGH** (fresh → WIDE) | TOO_EARLY | TOO_EARLY | TOO_EARLY |
| 11:35 | 11:40 | 0.4573 | -0.20 | red | 13 | 1.67 | 2.49 | ON / ON (not fresh) | TOO_EARLY | TOO_EARLY | TOO_EARLY |
| 11:40 | 11:45 | 0.4570 | -0.04 | red | 14 | 1.75 | 2.47 | ON / ON | TOO_EARLY | TOO_EARLY | TOO_EARLY |
| 11:45 | 11:50 | 0.4597 | +0.57 | green | 15 | 1.83 | 2.46 | ON / ON | TOO_EARLY | TOO_EARLY | TOO_EARLY |
| 11:50 | 11:55 | 0.4622 | +0.59 | green | 16 | 1.92 | 2.45 | ON / ON | TOO_EARLY | TOO_EARLY | TOO_EARLY |
| **11:55** | 12:00 | 0.4648 | +0.52 | green | 17 | **2.00** | 2.38 | ON / ON | **GREEN_BAR** (fresh → WIDE) | TOO_EARLY | TOO_EARLY |
| 12:00 | 12:05 | 0.4810 | +3.46 | green | 18 | 2.08 | 2.44 | ON / ON | ON | TOO_EARLY | TOO_EARLY |
| 12:05 | 12:10 | 0.4963 | +3.18 | green | 19 | 2.17 | 2.49 | ON / ON | ON | TOO_EARLY | TOO_EARLY |
| 12:10 | 12:15 | 0.5162 | +3.97 | green | 20 | 2.25 | 2.55 | ON / ON | ON | TOO_EARLY | TOO_EARLY |
| 12:15 | 12:20 | 0.5387 | +4.32 | green | 21 | 2.33 | 2.68 | ON / ON | ON | TOO_EARLY | TOO_EARLY |
| 12:20 | 12:25 | 0.5306 | -1.45 | red | 22 | 2.42 | 2.83 | ON / ON | ON (first red after 11:55; ATR > cap) | TOO_EARLY | TOO_EARLY |
| **12:25** | 12:30 | 0.5371 | +1.19 | green | 23 | **2.50** | 2.89 | ON / ON | ON | **ATR_HIGH** (fresh → WIDE) | TOO_EARLY |
| 12:30 | 12:35 | 0.5381 | +0.26 | green | 24 | 2.58 | 2.88 | ON / ON | ON | ON | TOO_EARLY |
| 12:35 | 12:40 | 0.5345 | -0.65 | red | 25 | 2.67 | 2.85 | ON / ON | ON | ON | TOO_EARLY |
| 12:40 | 12:45 | 0.5382 | +0.73 | green | 26 | 2.75 | 2.77 | ON / ON | ON | ON | TOO_EARLY |
| 12:45 | 12:50 | 0.5333 | -0.91 | red | 27 | 2.83 | 2.73 | ON / ON | ON | ON | TOO_EARLY |
| 12:50 | 12:55 | 0.5400 | +1.26 | green | 28 | 2.92 | 2.82 | ON / ON | ON | ON | TOO_EARLY |
| **12:55** | 13:00 | 0.5416 | +0.35 | green | 29 | **3.00** | 3.04 | ON / ON | ON | ON | **ATR_HIGH** (fresh → WIDE) |
| 13:00 | 13:05 | 0.5424 | +0.13 | green | 30 | 3.08 | 2.96 | ON / ON | ON | ON | ON |
| 13:05 | 13:10 | 0.5475 | +0.94 | green | 31 | 3.17 | 3.18 | ON / ON | ON | ON | ON |
| 13:10 | 13:15 | 0.5447 | -0.51 | red | 32 | 3.25 | 3.23 | ON / ON | ON | ON | ON |

## 6b. What each variant would have done

| variant | activation bar (open) · colour · close | FRENZY_LONG | WIDE on that bar | entry (12 s) | lock exit (LOCK2, ticks) | run-up still ahead (peak to date) |
|---|---|---|---|---|---|---|
| min hours 1.0 | 11:30 · red · 0.4586 (price-activated, streak 12) | **no**: ATR 2.54 > 2.5 | takes it | 11:35:12 at 0.4577 | **+6.9 %** (12:08:58) | +136 % |
| min hours 1.5 | same as 1.0 (the clock is not binding at 1.58 h) | no: ATR_HIGH | takes it | 11:35:12 at 0.4577 | +6.9 % | +136 % |
| **min hours 2.0 (live)** | 11:55 · green · 0.4648 (clock, streak 17) | no: GREEN_BAR | takes it (live fill 12:00:11 at 0.4649) | 12:00:12 at 0.4648 | **+5.2 %** (12:08:58) | +132 % |
| min hours 2.5 | 12:25 · green · 0.5371 (clock, streak 23) | no: ATR 2.89 → ATR_HIGH | takes it | 12:30:12 at 0.5388 | **−3.3 %** (12:57:04) | +100 % |
| min hours 3.0 | 12:55 · green · 0.5416 (clock, streak 29) | no: ATR 3.04 → ATR_HIGH | takes it | 13:00:12 at 0.5406 | **−3.4 %** (13:05:38) | +99 % |
| wait ≤1 (all and streak > 12; 2.0 h) | 11:55 green → 12:00 green | no entry | stands aside (pending) | – | **no trade** | – |
| wait ≤3 | 12:00, 12:05, 12:10 all green | no entry | stands aside | – | no trade | – |
| wait ≤6 | first red 12:20, ATR 2.83 > cap → dropped | no entry | stands aside | – | no trade | – |
| wait ≤12 | same first red 12:20 → dropped | no entry | stands aside | – | no trade | – |

**What the case shows:**
- No variant gives FRENZY_LONG the RLC trade. The brief's "11:35 entry at 0.4573" does not happen in the engine: the setup turns on at 11:30, where the ATR is 0.04 pt over the cap.
- An earlier clock would have given **WIDE** a better entry: +6.9 % vs +5.2 %. A later clock turns RLC into two −3 % stops.
- Every wait-for-red variant loses the trade outright, which is the year's §2d mechanism in a single episode.
- The +100–136 % run-up was not captured by any entry variant. Each one exits on the 12:08 dip (or a stop) under the lock. That is an exit question, studied in `FRENZY_RLC_REENTRY_SIGNATURE_2026-10-06.md` and `FRENZY_RIDE_CAPTURE_SYNTHESIS_2026-10-06.md`, not an entry-timing one.
