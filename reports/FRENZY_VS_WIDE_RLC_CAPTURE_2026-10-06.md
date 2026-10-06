# Why FRENZY_LONG missed RLC on 10-05 and FRENZY_WIDE took it (2026-10-06)

Research only. No bot code, config, test or template was touched. Nothing was committed.
Scripts and outputs are in the scratchpad (`…/scratchpad/rlc/`: `fetch.py`, `walk.py`, `q3.py`, `q3b.py`).

## 0. Engine parity (read this first)

- **Bars:** RLCUSDT 5m and 1h klines from the public Binance futures REST API (Sep 28 → Oct 6 04:15 UTC).
- **Code:** the real `services/frenzy.py` functions, in the engine's order: `normal_hour_usd` → `frenzy_walk` → `frenzy_flagged` → `frenzy_long_status` → `frenzy_wide_ready`. ATR is the engine's `wilder_atr_pct(closed[-300:])`. The window is the last 1,499 closed bars, as live. Thresholds come from `trading_config.json`.
- **Only hand-built input:** 24 h volume. It is the sum of the last 288 bars' quote volume, not the live ticker. It was $43M against a $20M floor, so it cannot change any decision.

The live fill against the replay, on the same signal bar (11:55–12:00 UTC):

| field | live fill (orders CSV) | replay (real functions) |
|---|---|---|
| spike | 10-05 10:00 | 10-05 10:00 |
| hours after spike | 2.0 | 2.00 |
| VWAP | 0.454769 | 0.4548 |
| vs VWAP | +2.206 % | +2.206 % |
| volume × normal | 474.7 | 474.7 |
| run since spike | +23.6 % | +23.60 % |
| ATR(14) | 2.3775 % | 2.3775 % |
| signal candle | +0.519 % (green) | +0.519 % (green) |
| market volume (gvol) | 0.684 (< 1.0, passes) | not rebuilt (live stamp used) |
| decision | journal `FRENZY_GREEN_BAR` BLOCK, then `FRENZY_WIDE` OPEN at 12:00:11, price 0.4649 | `FRENZY_GREEN_BAR` → `frenzy_wide_ready` = True |

**Parity is exact.** Every number below comes from the engine's own functions on the engine's own bar.

## 1. Direct answer

**FRENZY_LONG refused RLC for one reason only: the signal candle was green.**
- The 11:55–12:00 candle closed at +0.519 % (open 0.4624, close 0.4648).
- Rule: `frenzy_long_skip_green_bar = true` (DECISION_LOG 180). FRENZY_LONG only buys after a red or flat signal candle.
- **Every other LONG condition passed:**
  - the setup was ON
  - it was the first ON bar
  - 24 h volume was $43M against the $20M floor
  - ATR was 2.38 % against the 2.5 % cap
  - market volume was 0.68 against the 1.0 cap

**WIDE took it because WIDE is defined as "the trades FRENZY_LONG refuses ONLY for a green candle or a high ATR".**
- In code: `frenzy_wide_ready` = the bar is fresh ∧ the refusal code is `FRENZY_GREEN_BAR` or `FRENZY_ATR_HIGH` ∧ the ATR is readable.
- So WIDE is not a different detector. **It is the same setup on the same bar at the same price.** It runs at a smaller size (lev 0.2 against 0.32) and uses the same exit.

**Why that one candle decided everything:**
- The setup turned ON on the clock, not on price.
  - Price had held above its average for a full hour from 11:35 (12 closes in a row).
  - The only thing still missing was the 2-hour minimum after the spike (`frenzy_min_hours = 2`; spike at 10:00 → 12:00).
  - So the entry bar was simply whichever candle ended at 12:00. Its colour was luck.
- The setup then **stayed ON without a break for at least 16 hours**. It was still ON at the last bar read (Oct 6 04:10), with 211 closes in a row above the average.
  - A new entry needs the setup to be OFF for ≥ 1 h and then turn ON again.
  - So there was **exactly one entry chance all day**, and a coin-flip candle sent it to WIDE.

**What WIDE actually caught:**
- WIDE closed RLC at **+3.01 % after 3 minutes** (12:03, `FRENZY_TP`). The fixed +3 take-profit was still live at the time; the lock exit was deployed at about 15:55.
- It did not ride the move. RLC went from 0.4648 to a high of 0.7888 (+70 %) at Oct 6 01:35.
- Under today's lock exit the same trade would bank about +5.3 % (`FRENZY_RIDE_CAPTURE_SYNTHESIS`).
- **Had the candle been red, FRENZY_LONG would have opened the identical trade at 0.32 leverage and closed it at the identical +3.0 %.** WIDE did not catch anything LONG could not. It caught it at 62 % of LONG's size.

## 2. Bar by bar (signal bar closing at …, UTC, 10-05)

| bar close | close price | status code | what failed | WIDE? |
|---|---|---|---|---|
| ≤ 09:55 | 0.384–0.423 | no episode | no spike yet | – |
| 10:00 | 0.4264 | BELOW_AVG | spike found (30-min return ≥ 5 %, hour volume ≥ 20×). Needs 12 closes above the spike VWAP: 1 of 12 | no |
| 10:05 – 10:30 | 0.429–0.435 | BELOW_AVG | 2 … 7 of 12 closes above VWAP | no |
| 10:35 | 0.4270 | BELOW_AVG | closed **below** VWAP (−0.42 %): the streak resets to 0 | no |
| 10:40 – 11:30 | 0.431–0.474 | BELOW_AVG | 1 … 11 of 12 closes above VWAP; volume 238× → 517× normal | no |
| 11:35 – 11:55 | 0.457–0.462 | TOO_EARLY | hour above VWAP ✔, volume ≥ 100× ✔; only 1.6 … 1.9 h after the spike < 2 h | no |
| **12:00** | **0.4648** | **GREEN_BAR** | fresh ✔ · 24 h volume ✔ · ATR 2.38 ≤ 2.5 ✔ · **candle +0.519 % green ✗** | **YES → opened 12:00:11** |
| 12:05 → Oct 6 04:10 | 0.481 → 0.79 high | ON | "entry bar passed": the setup never switched off, so there was never a new fresh bar | no |

- ADX/DI was not an entry condition. The DI spread and ADX change (+16.2 / −4.0) are stamps.
  - They only matter for FRENZY_LONG's "strong" size (ADX rising ∧ +DI above −DI). ADX was falling, so even a LONG would have opened at the normal 0.32, not the strong 0.5.
- The market-volume gate passed (0.684).

## 3. Did FRENZY_LONG qualify later? No.

- The replay shows **183 scan bars in a row with the setup ON** (12:05 → Oct 6 04:10). There was no OFF hour and so no new fresh bar. Neither sleeve could fire again.
- The decision journals (three exports, Oct 3 → Oct 6 03:55) have exactly **one** FRENZY line for RLC: the 12:00 `FRENZY_GREEN_BAR` block, followed by the WIDE open.
  - There is no `FRENZY_MAX_SLOTS`, `FRENZY_PAIR_DAY_CAP`, `FRENZY_GVOL_HIGH`, `FRENZY_LATE` or `FRENZY_OPEN_REFUSED` for RLC.
  - The later RLC journal lines are momentum-sleeve MACRO / RSI / ADX blocks, not FRENZY.
- **Not a slot cap, not a cooldown, not a same-pair lock.** LONG simply never had a second qualifying bar.

## 4. Structural or incidental?

**At the trade level it was incidental:** the colour of one clock-timed candle.

**At the design level it is structural:** WIDE is, by definition, LONG's green-candle and high-ATR refusals.

The real question is whether those refusals hold a ride class that pays. Data: engine-bar year cohort, Jan 10 → Sep 27, live lock exit (LOCK2), 12 s entry, 0.10 slippage, live sequencing (`frenzy_engine_cohort_report.sequence`). Two universes:
- **PUB:** as published, the `live_book` = 981 fills, reproduced exactly.
- **TRADE:** the overnight review's live-tradeable universe (no Alpha / < 90-day / non-ASCII pairs, live-like gvol < 1), re-sequenced: 769 fills.

A **ride** = the 12 h peak reaches **+10 % before any −3 % stop**. CIs are 95 %, from a 2,000× day-block bootstrap.

### 4a. Rides are not a WIDE specialty

| sleeve (TRADE) | fills | ride rate | rides (days · pairs) | lock banks per ride | non-rides mean | all fills mean, day CI |
|---|---|---|---|---|---|---|
| FRENZY_LONG | 204 | 30 % | 62 (50 days · 45 pairs) | **+3.95 %** | −1.21 % | **+0.359** [−0.07, +0.81] |
| FRENZY_WIDE | 565 | 30 % | 167 (121 days · 103 pairs) | **+2.96 %** | −1.71 % | **−0.331** [−0.58, −0.06] |

On PUB: LONG 30 % rides, +0.280; WIDE 31 % rides, −0.214 [−0.45, +0.03].

- **Both sleeves hit a ride on 3 in 10 fills.** WIDE does not detect a different kind of ride. It sees more of them only because it has about 3× more signals.
- **Episodes** (pair × spike): 173 episodes produced a ride fill (TRADE).
  - 119 rode only through a WIDE fill.
  - 37 rode only through a LONG fill.
  - 17 rode through both.
  - This is the "WIDE catches rides LONG misses" fact. It is real, but it comes with WIDE's losers (§4b).
- **The exit, not the entry, decides what a ride is worth.** The lock banks about +3 to +4 % per ride in both sleeves. RLC-size captures need a runner, and every runner tested so far fails (`FRENZY_RIDE_CAPTURE_SYNTHESIS`).

### 4b. Do the rides pay for WIDE's losers? No.

| WIDE (sum of % per fill, 1×) | PUB | TRADE |
|---|---|---|
| rides | +688 (228 fills) | +494 (167) |
| non-rides | −845 (506) | −681 (398) |
| **net** | **−157** (mean −0.214) | **−187** (mean −0.331, CI below 0) |
| without its best 5 / 10 fills | −0.289 / −0.346 | −0.420 / −0.488 |
| book at lev 0.2 (final / max DD) | −83 % / −92 % | −86 % / −92 % |

- Per fill, the 167 rides earn +494, but the 398 non-rides cost −681.
- **On today's exit, each RLC-type ride is worth about +3 %, and WIDE carries about 2.4 non-ride fills (averaging −1.7 % each) for every ride.**

### 4c. RLC belongs to WIDE's better half

| WIDE by refusal code | PUB N · mean · CI | TRADE N · mean · CI |
|---|---|---|
| **GREEN_BAR** (RLC's code: ATR ≤ 2.5, green candle) | 242 · **+0.132** · [−0.31, +0.57] | 204 · **+0.013** · [−0.44, +0.48] |
| ATR_HIGH (ATR > 2.5) | 492 · −0.384 · [−0.67, −0.10] | 361 · **−0.525** · [−0.82, −0.22] |

- WIDE's loss is the ATR_HIGH half. It is negative in both halves of the year (the sleeve checklist found the same).
- The GREEN_BAR half, where RLC sits, is break-even.
- **So RLC is an argument for the green-candle half at most, never for WIDE as a whole.**

## 5. Could FRENZY_LONG be widened to catch RLC? (SCREEN, not a ship)

Each variant takes LONG's own trades plus some of the green-candle refusals, sized and slotted as LONG, with WIDE off. TRADE universe, lock exit, book at lev 0.32.

| variant | N | days | WR | mean | day CI | Jan–Apr / May–Sep | drop top 5 | months + | book / DD |
|---|---|---|---|---|---|---|---|---|---|
| V0 LONG as live | 205 | 133 | 54 % | +0.387 | [−0.07, +0.84] | +0.52 / +0.25 | +0.151 | 6 of 9 | +116 % / −34 % |
| V1 LONG + every green candle (green-candle rule off) | 405 | 190 | 53 % | +0.170 | [−0.18, +0.50] | +0.32 / +0.01 | +0.041 | 5 of 9 | +64 % / −65 % |
| V2 LONG + green **only when the setup turned ON by time / volume** (`above_streak > 12`, RLC's exact case) | 326 | 172 | 56 % | +0.392 | [+0.01, +0.74] | +0.56 / +0.22 | +0.235 | 6 of 9 | +254 % / −48 % |
| … V2's added fills alone | 123 | 93 | 59 % | +0.418 | [−0.16, +0.98] | +0.57 / +0.25 | +0.101 | 7 of 9 | – |

PUB gives the same picture:
- V1: +0.194 [−0.12, +0.50], DD −67 %.
- V2: +0.381 [+0.06, +0.70]; its added fills +0.523 [−0.00, +1.09].
- The green-candle split by trigger:
  - green on a **price-reclaim** bar (`above_streak = 12`): −0.505 [−1.15, +0.19]
  - green on a **time / volume** bar (`> 12`): +0.553 [+0.01, +1.09]

How to read this:
- **V1 fails.** Dropping the green-candle rule halves LONG's edge and nearly doubles its drawdown. Rule 180 is doing its job on average.
- **V2 is the only RLC-shaped extension that does not dilute LONG.**
  - The logic is simple. When the setup turns ON because the 2-hour clock ran out (as on RLC), the candle colour is luck. When it turns ON because price just reclaimed its average, a green candle is a chase.
  - **But V2 is the `WIDE_HOLD_GREEN` pocket the Oct-5 checklist pre-registered after the fact.** The overnight review found it is about what the best of ~2,000 searched cells looks like on shuffled data: selection-adjusted p 0.87 on the tradeable universe. Its added fills' CI touches 0.
  - It also raises drawdown (−34 → −48 %).
- **Verdict: a screen, not a ship.** It meets none of the locked promotion bars on fresh data. The honest path is the forward observe line the checklist already proposed: tally `above_streak > 12` ∧ GREEN_BAR fresh signals at batch reviews, with a frozen bar.

## 6. Should WIDE stay on? What the RLC evidence does and does not show

- **RLC is one window** (one pair, one day). It counts as one observation.
- It shows that WIDE takes LONG's coin-flip refusals. It does not show a ride WIDE can bank: WIDE closed RLC at +3.01 %, the same as any small winner.
- The year shows the rides are already inside WIDE's numbers: 167 rides on 121 days. WIDE is still −0.33 %/fill on the tradeable universe (CI below 0, book −86 %).
- The loss sits in the ATR_HIGH half. The green-candle half where RLC lives is about flat.
- **If the operator wants to keep RLC-type entries live**, the evidence-consistent shape is one of these, with a pre-committed revert gate (operator decision; nothing changed here):
  - WIDE restricted to the GREEN_BAR code, or
  - only its time / volume-trigger part (V2), at a probe size.
- Keeping the ATR_HIGH half has no support.
- Turning WIDE off entirely costs RLC-type +3 % fills. On the year, all of WIDE's fills together summed to −187 % of 1× fill P&L (TRADE), which you would stop paying.

## 7. Blind spots

1. **24 h volume** in the RLC replay is a kline sum, not the live ticker. It was $43M against a $20M floor, so it is irrelevant here.
2. **gvol** was not rebuilt for 10-05. The live stamp (0.684) was used. It only matters for an entry, and there was a single fresh bar.
3. **The RLC walk ends at Oct 6 04:10**, the last closed bar fetched. The setup was still ON then.
4. **The cohort ends Sep 27.** RLC (Oct 5) and the 8 live FRENZY / WIDE fills are not in the year numbers.
5. **Rides are measured on the lock exit.** A runner exit changes the value per ride but not which sleeve gets the ride. Runners were already refuted in `FRENZY_RIDE_CAPTURE_SYNTHESIS`.
6. **V1 and V2 re-sequence LONG's 2 slots.** They are pessimistic if live could run more slots, and they ignore other sleeves holding the pair (which can only remove fills).
7. **FRENZY_LONG's "strong" 0.5 sizing (197) is not modelled.** RLC would not have qualified for it anyway: ADX was falling.
