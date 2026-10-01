# Overnight report — 2026-10-01

Everything below was run while you slept (the ride-the-monster and chop-gate tests were added during the day). Nothing here changes the bot. Two things need your word (section 6).

## 1. The headline

1. **The "runaway pair" idea (MOVR) does not make a sleeve.** Three rounds of backtests, each corrected after independent review:
   0 of 64 pre-declared cells pass, and no cell has even a plain 95 % range above zero. The same holds for your two lift-off ideas
   (volume leads price, EMA200 lift-off): 0 of 64 pass in every round, and in the final round not one cell is even positive in
   both halves of the year.
2. **Why the first run looked like a pass:** the runaway-SHORT "edge" (+0.42 %/day) was real on paper but lived in the first
   60 seconds after the trigger candle — liquidation-cascade minutes (10 events carry most of it; in some the price fell 20 % inside
   one minute). The first run filled at the exact closing price. Your real entries land a median 2.4 minutes after the candle.
   One minute later the edge is +0.08 % before slippage, about zero after.
3. **Exit-free check — these setups mean-revert.** After a +8–12 % run in 4 h on heavy volume with BTC flat, the next 24 h average
   is −1.9 % to −2.7 % (significant by day). "Volume leads price" in the top-50: −2.4 % over 24 h. MOVR was the fat-tail exception.
   Fading them is not simple either: the median move against a fade is +13 % within 24 h.
4. **The real find of the day is yours: choppy BTC.** It is registered as an observe item, rebuilt for all 108 master trades (the rebuilt value checked against the live stamps), and scoped to
   momentum longs (section 3).

## 2. Runaway and lift-off backtests

Data: 5-minute and 1-minute candles, 381 pairs with at least one runaway event, 2026-01-29 → 2026-09-27. Rules written in the script headers before each run.

| Round | What changed | Runaway | Lift-off |
|---|---|---|---|
| v1 | first run | 28 "passes", then 4 after the pullback-fill fix | not finished (stopped) |
| v2 | control's own ATR; paired event − control by day; stable strict bound; gap-through-stop fills; entry 1 min after the candle; slippage; universe before cooldown | **0 of 64** | **0 of 64** |
| v3 | clean controls (no same-pair trigger nearby); entry slippage on the entry price; honest win-rate column; fixed concentration measure; lift-off triggers fire once per episode | **0 of 64** — 0 cells with 95 % range > 0, 0 paired > 0 | **0 of 64** — 0 cells with 95 % range > 0, 0 paired > 0, 0 positive in both halves (about 21,000 events) |

Runaway v3, best cells (240-min hold), per trade after fees and slippage:

| Side | Trigger | Entry | Exit | Trades | Days | Avg per trade | 95 % day range | vs own control |
|---|---|---|---|---|---|---|---|---|
| LONG | +12 % | next candle | Bull-Run 2×ATR | 1,557 | 238 | +0.03 % | [−0.12, +0.12] | −0.05 [−0.19, +0.09] |
| LONG | +12 % | next candle | SURGE 1×ATR | 1,557 | 238 | +0.03 % | [−0.13, +0.11] | −0.02 [−0.16, +0.11] |
| SHORT | −12 % | pullback | wide stop | 196 | 135 | +0.30 % | [−0.50, +1.16] | +0.79 [−0.42, +1.99] |
| SHORT | −12 % | next candle | SURGE (the v1 "passer") | 528 | 210 | −0.04 % | [−0.19, +0.24] | +0.15 [−0.13, +0.43] |

The range is around the mean of the daily averages, which differs slightly from the per-trade average (e.g. +0.02 vs −0.04 in the last row).

Lift-off v3, best cells per trade: SHORT after a volume surge below EMA200 in the top-50 −0.04 % (557 trades, range [−0.29, +0.29]);
every LONG cell is negative (−0.12 % to −0.56 %), several significantly. Against their own controls the triggers add nothing
(best +0.24 [−0.08, +0.56]); "volume leads price" LONG with the wide stop is significantly WORSE than a random entry (−0.58 [−0.90, −0.25]).

Exit-free forward odds (final events; gross; by day; returns are in the TRADE direction — a negative SHORT row means the price bounced):

| After the trigger | Next 4 h | Next 24 h | Target-first share (+5 vs −3) | Baseline |
|---|---|---|---|---|
| Runaway LONG +8 % | −0.55 % | **−1.86 % [−3.23, −0.50]** | 38 % | 35 % |
| Runaway LONG +12 % | **−0.89 % [−1.75, −0.03]** | **−2.70 % [−4.43, −0.97]** | 38 % | 35 % |
| Runaway SHORT −12 % | −0.68 % | −2.24 % [−6.47, +1.99] | 38 % | 35 % |
| Volume leads price, top-50, LONG | −0.49 % | **−2.39 % [−3.85, −0.92]** | 34 % | 35 % |
| EMA200 lift-off, LONG (top-50 / 51–100) | −0.40 % / −0.23 % | −0.47 % / −0.51 % | 39 % / 36 % | 35 % |
| Volume ∧ lift-off together (≤ 2 h), top-50 LONG | −1.07 % | −1.36 % [−4.27, +1.55] | 36 % | 35 % |

Once volatility is accounted for, none of the triggers raises the chance of hitting a target before a stop (33–41 % vs 35 %).
The runaway and volume-surge longs lose money on average over the next day: these moves mean-revert.

Review findings that matter for trust:
- Even at **zero slippage**, 0 of 64 runaway cells clear the strict bound; lift-off has no cell positive in both halves.
- My 0.05 %/side slippage is 2–5× harsher than your measured fills on normal trades (maker entries average −0.035 %, taker
  fallbacks +0.06 %). It is not what decides the result.
- Lift-off: the control trades lose about as much as the trigger trades — the triggers add nothing over a random entry on the same pair.

What these tests cannot say:
- **Survivorship:** only pairs still listed today are in the data (delisted pairs missing — this flatters longs and penalises shorts).
- **Exits:** these grids use scalper-style exits (stops −0.7 % to 2×ATR capped at −6 %, hold ≤ 8 h). Multi-day wide-stop exits were tested separately — section 2b.
- **Execution:** sub-minute entry (a websocket trigger inside the candle) is untested — that is where the SHORT edge lives, and
  1-minute candles cannot resolve those minutes in either direction.
- **Period:** one 8-month span.

Files: `scripts/runaway_sleeve_design.py`, `scripts/liftoff_sleeve_design.py`, `scripts/trigger_forward_odds.py`;
`reports/RUNAWAY_SLEEVE_GRID_v3_2026-10-01.csv`, `reports/LIFTOFF_SLEEVE_GRID_v3_2026-10-01.csv`,
`reports/TRIGGER_FORWARD_ODDS_2026-10-01.md`. The large per-trade FILLS files are not in git: re-run the two design scripts to rebuild them.

## 2b. Ride the monster (multi-day, wide stop) — added during the day

Your question after MOVR kept running: would a small long with a wide stop, held for days, on every runaway / volume-surge trigger pay
because the rare monsters cover the many that fade? `reports/MONSTER_RIDE_TEST_2026-10-01.md` (four exits: stops −10 / −20 / −30 %,
trails, +100 % target, holds 3–7 days; ranges by entry week).

- All 16 trigger × exit cells are negative per trade (−0.23 % to −4.92 %); 4 are significantly negative, none is positive.
- Only 3 of 16 beat a no-trigger baseline on the same pairs, and the baselines lose too — part of the loss is the period (alts
  bled for most of 2026), not the trigger. So this is "no edge found", not "proven loser".
- The trigger does raise monster odds: with the −20 % stop and a 7-day hold, 20 % of +12 % runaways reach +50 % and 10 % reach +100 %
  (baseline 11 % / 4 %). But 65 % hit the −20 % stop first.
- Without a trail the monsters give a lot back: of the trades that touched +100 %, half finished below +50 % and about a quarter
  finished below zero. With a trail about half the peak is kept, and the stopped trades still outweigh it.

Verdict: no sleeve (DECISION_LOG 164).

## 3. Choppy BTC and momentum longs (your finding)

BTC 72-hour efficiency = net move ÷ total path. Near 0 = BTC zig-zagged for three days and went nowhere.
Rebuilt for every master trade from BTC candles with the bot's own formula (matches the 24 live stamps: correlation 0.999,
same tier 96 %).

| Momentum longs (master, current stack + Oct-1) | Trades | Days | Win rate | Avg % |
|---|---|---|---|---|
| Choppy (≤ 0.007) | 13 | 6 | 46 % | −0.16 |
| Middle (0.007–0.026) | 33 | 25 | 91 % | +0.37 |
| Trending (> 0.026) | 62 | 34 | 81 % | +0.26 |

Other sleeves in choppy tape (few trades): momentum short 3 · 100 %, spike fade 4 · 75 %, flip short 1 · 0 %. No sign of harm —
the effect is specific to momentum longs, which fits the theory (breakout longs need a trending BTC).

Year backtest (current gates, one row per trade): choppy tier −0.20 % (Jan–Apr) / −0.23 % (May–Sep) against about −0.07 % elsewhere.
Honest limits: the gap vs the rest is not yet statistically significant by day; the finer slices disagree between halves; 6 master
days are about 3 chop episodes.

**Formal backtest of the chop rule (added during the day, `reports/CHOP_GATE_BACKTEST_2026-10-01.md`): OBSERVE, not arm.**
Chop vs the rest in the year replay: Δ −0.15 %; borderline (p ≈ 0.01–0.04 for the year by trade, not significant in Jan–Apr, gone
without July–August); no dose-response; a single bot run would gain about +10 points of summed % over 9 months (≈ +5–7 after the
haircut). I first recommended arming it and withdrew that after both reviews found my significance test and gain were overstated.

Sub-findings (tracked, no rules):
- **Choppy ∧ burst** (several longs in the same 2 minutes): the worst cell. Master: 4 trades, 0 won, in 2 windows
  (Jul-10 ADA+LIT, Oct-1 WLD+ENA). A live rule can only block the 2nd+ fill of a burst → LIT and WLD → +$395 (not +$750).
- **Middle tier ∧ BTC 1h slope falling:** backtest negative in both halves (−0.27 / −0.18) but master break-even
  (8 trades, 75 %, −0.01 %); blocking it would have cost $176. Comparison line only.
- **Blocking all choppy trades** on the master: +$294 (it costs three BASE winners, saves Sep-29 and Oct-1).

Registered in `CLAUDE_CURRENT_STATE.md` (committed): fresh trades after 2026-10-01 02:00 UTC, at least 15 trades on at least 8 days
spanning at least 4 chop episodes, win rate below 61 % and average below zero at 95 % by day. Read with
`venv/bin/python scripts/ml_regime_observe_read.py --fresh <orders.csv>`.

## 4. Other checks from yesterday

- **BTC 1h slope < 0 as a 2D filter:** refuted. Master 0 survivors of 106 splits (underpowered: 9 losers); backtest 0 confirmed of
  120 with seeds collapsed, before and after current gates. The first "2 confirmed" was five replays of one year counted as independent.
- **Crowding alone:** burst 22 trades · 68 % · +0.20 % vs solo 86 · 83 % · +0.26 %. Not a filter; registered as a frozen watch.
- **Columns the screen had silently skipped** (my miss, now a permanent rule): all 8 regime columns tested on the fully stamped
  backtest. One soft lead: momentum longs while BTC sits within 1.5 % of its 24h low are worst in both halves (−0.12 / −0.16, day
  ranges below zero); best when BTC is more than 2.6 % off its low (−0.05 / +0.04). Not registered — your call (section 6).
- **C1 momentum shorts:** no change recommended (12 current-stack trades · 67 % · +0.06 %).
- **WLD / ENA (Oct-1, −1.0 % each):** one burst window in choppy tape, 2× cell. They sit in three observe items, no tripwire.

## 5. What was committed and pushed yesterday (all with both reviews)

| Commit | What |
|---|---|
| `9d2e74b` | Full gate set per refusal (FAILS journal) + scout "loosen a filter or new sleeve?" classes |
| `e564bde` | Scout feature stamps: the bot's own entry columns + pre-move features on every missed move |
| `afd8608` | Top Pairs table: narrower Pair column, two-line Gap headers, opaque header |
| `216e6aa` | Auto daily Decisions CSV + save before reset |
| `27bb568` | Scout 4-hour exit-shape fields |
| `c5a23a3` | Scout writes its own notes; one fixed command (no approval prompts); every 4 h |
| `519620c` | Manual "Floor stop" takes an optional TP |
| `f08d83e` | Journal no longer loses fills on a deploy; manual trades out of every cell report |
| `a71c979` | Slope < 0 2D screen (refuted) + burst-crowding watch |
| `e1711c7`, `b020a20` | BTC-chop watch (≤ 0.007) + rebuilt eff72 + sub-lines |

All 430 tests pass. `trading_config.json` was not touched by any of these commits.

## 6. Decisions waiting for you

1. **Register the "BTC near its 24h low" lead** as a fourth momentum-long watch (≤ 1.5 % above the 24h low, threshold frozen, fresh trades, counted by day)?
2. **Sub-minute cascade shorts:** the only place a hint of an edge showed up is the first minute of a −12 % collapse (about 10 events, in-sample, filled at the candle close — not evidence yet). Catching it would need
   a websocket trigger inside the candle and tick data to test. Worth a design discussion, or drop it?

Scheduled today: scout review at 12:00 your time (`reports/SCOUT_REVIEW_2026-10-01.md`). The scout runs every 4 hours with no prompts.
