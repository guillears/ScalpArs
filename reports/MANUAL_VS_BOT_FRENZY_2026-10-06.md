# Manual entries vs the bot: FRENZY / WIDE, 2026-10-06

**Question (operator):** "Check my manual entries and check why the bot didn't trade. There is a lot of money we are missing in FRENZY trades."

**Short answer:** the bot did not miss a FRENZY trade on any of your three entries. On GRIFFAIN the bot had a fresh FRENZY bar 2 h 51 min before your click. It refused that bar for ATR. Had it entered, it would have been stopped at −3 % within about 10 seconds. On EDU there was no FRENZY setup at all, in either of its two episodes. What you traded is a different setup. You bought right after the price **came back above the spike average (VWAP) after 1 to 1.5 hours below it**, at a volume far under FRENZY's 100× bar, on a small dip. That setup has been tested before ("RECLAIM", Oct-2: 4,155 entries, negative). Today's three trades are not enough to overturn that. Most of your result came from your **exits**, not from the entry.

Sources and checks:
- Exports: `scalpars_orders_paper_2026-10-06_21-11-40.csv` has 4 fills: 3 MANUAL and 1 FRENZY_LONG (ORCA). `scalpars_decisions_paper_2026-10-06_21-11-45.csv` is the journal from 10-03 21:12 to 10-06 21:11.
- Server log: `web.stdout.log`, 10-06 10:01 → 21:09. **Logs checked:** the FRENZY pass ran every 5 minutes from 13:53 to 21:05. It was skipped from 13:20 to 13:53, when the bot was paused. Neither pair had a "not read" or "failed" line. All FRENZY_LONG, FRENZY_WIDE and FRENZY_CATCHUP lines are quoted below.
- **Engine parity:** the replay calls the real `services.frenzy` functions on public 5m and 1h klines: `frenzy_walk`, `frenzy_flagged`, `frenzy_long_status`, `frenzy_wide_hold_green_block` and `wilder_atr_pct`. Thresholds come from the live `trading_config.json`. It reproduces every observable engine event today:
  - GRIFFAIN fresh ON on the 17:35 bar, ATR **3.63 %**. The log says "ATR 3.63% > 2.5%".
  - GRIFFAIN first flagged at the 17:35 pass. That is when its 24 h volume crossed $20M (shortlist).
  - EDU's flag dropped at 13:10. The Oct-5 13:05 episode ended without one state bar.
  - EDU re-flagged at 17:55 on a new spike. The log line at 19:25 says "1.5 h after the spike".
- **Market volume (gvol):** engine-parity U2 definition from `FRENZY_GVOL_GATE_REVALIDATION_2026-10-06.md`. Check: the ORCA 18:05 fill recomputes to 0.7802, against a live stamp of 0.7802.

---

## Entry 1: GRIFFAINUSDT LONG, 20:31:02 UTC @ 0.024695 → MANUAL_TP +3.11 % (+$310.76)

| | |
|---|---|
| On the FRENZY list? | Yes. Flagged from the 17:35 pass on, after a spike at 14:50 (30-min +5 %, ≥ 20× volume). Before 17:35 its 24 h volume was under $20M, so it was not on the shortlist. |
| Fresh FRENZY bar | **Yes, the 17:35 bar (closed 17:40), 2 h 51 min before your click.** Setup ON 2.8 h after the spike, 12 closes above VWAP, volume 155×. |
| Bot's decision on that bar | `[FRENZY_LONG] setup turned ON but refused — ATR 3.63% > 2.5%` and `[FRENZY_WIDE] setup ON but refused — ATR above the FRENZY cap` (WIDE takes only hold-green refusals since DECISION_LOG 231). Market volume on that bar was 0.877, so the gvol gate would have let it through. |
| What that entry would have done | A flash dump began 4 s after the close: −2.2 % at +6 s, −3.0 % at +8 s, **−5.8 % at +20 s** (aggTrades). The bot judges at about 17:40:07. Either the 1 % dislocation guard refuses it (price already −2.4 % at +8 s), or it fills near −0.3…−2 % and is **stopped at −3 % (or worse on the gap) within seconds**. The same holds for every exit: lock, +3/−3 and +1/−3. **The ATR refusal saved money.** |
| State at your click | Setup **OFF**. It was ON from 17:40 to 18:50, then fell below VWAP. One short reclaim (19:10–19:30) failed into a −7.5 % dump at 19:45. Then 11 closes below VWAP, from 19:35 to 20:25. **Your click came 1 minute after the first close back above VWAP (20:30).** Code `FRENZY_BELOW_AVG` "back above its average (+1.2%) · 1 of 12 closes". Volume 72× (FRENZY needs 100×). |
| Catch-up | Nothing to catch up. The 17:40 ON bar was judged normally, and the setup never turned ON again before 21:10. |
| Your entry vs the bot's | 0.024695 is **4.2 % below** the bot's ON close (0.025774). |

## Entry 2: EDUUSDT LONG, 20:50:44 UTC @ 0.06776 → MANUAL_TP +1.00 % (+$100.16)

| | |
|---|---|
| On the FRENZY list? | Yes. Flagged from 17:55, on a new spike at 17:55. It was also flagged 10:05–13:05 on the Oct-5 episode. Order-book (BOOK) lines are logged for it all day. |
| Fresh FRENZY bar | **None, in either episode.** Oct-5 episode: up to 40 closes above VWAP, but volume stayed at 15–55× while it held above (code `FRENZY_VOL_FADED`, never in state). Oct-6 episode: never 12 closes above VWAP, and volume peaked at 59×. The bot logged no FRENZY_LONG, FRENZY_WIDE or CATCHUP line for EDU all day. Its only lines were a short observation at 19:25 and three harmless KL_MISMATCH lines in the morning. |
| State at your click | Setup OFF (`FRENZY_BELOW_AVG`, 1 of 12 closes), 2.9 h after the spike, volume 28×. There were **18 closes below VWAP** (19:20–20:45, down to −5.6 %). The first close back above came at 20:50 (+0.08 %). **Your click came 44 s after that close.** The price had already slipped back: your fill was **−0.77 % under VWAP** and −0.8 % under the reclaim close. |
| What the bot's rules give | Nothing. No FRENZY, WIDE or catch-up rule produces a trade on this episode. |

## Entry 3: EDUUSDT LONG, 21:09:27 UTC @ 0.06912 → MANUAL (closed by hand) +0.24 % (+$24.26)

| | |
|---|---|
| State at your click | Setup OFF. 4 closes above VWAP (+0.6 %), volume 24×, 3.2 h after the spike. **Market volume on the last closed bar was 1.62.** A FRENZY entry would have been refused here by the gvol gate (≥ 1.0). |
| What happened | You set the exit to FLOOR (stop −3 / target +2) and closed by hand at 21:11:28. **At 21:15 to 21:18 EDU fell −4.1 % from your entry.** The same click on the bot's lock exit or on +1/−3 is **−3.00 %**. Your hand close was worth about +3.2 points. |
| What the bot's rules give | Nothing (no setup). |

---

## Same trades on the bot's exits (1-minute bars, entry +0.02 % slippage, 0.09 % fees, stop checked before target inside a minute)

| Entry | Your result | Live lock (−3 / +2 at +3 / trail 2) | Fixed +3 / −3 | Fixed +1 / −3 (your usual FLOOR) |
|---|---|---|---|---|
| GRIFFAIN 20:31 | **+3.11** (you moved the target 1 → 3 → 3.1) | +2.00 (peak +3.52, gave back to the lock) | +3.00 | +1.00 |
| EDU 20:50 | +1.00 | +2.00 (peak +3.11 at 21:08) | +3.00 | +1.00 |
| EDU 21:09 | **+0.24** (hand close before the dump) | −3.00 | −3.00 | −3.00 |
| **Sum, % of position** | **+4.35** | +1.00 | +3.00 | **−1.00** |
| The bot's own fresh bar (GRIFFAIN 17:40) | – | −3.00 (or refused by the dislocation guard) | −3.00 | −3.00 |

Your clicks put through any fixed bracket sum to between −1.0 and +3.0 points. Your discretion added the rest: raising the GRIFFAIN target as it ran, and cutting EDU#3 6 minutes before a −4 % drop. That matches the Oct-1 to Oct-3 finding: the exit and the timing are the skill, not the tape state.

## What you did differently (features at the click)

| Feature | GRIFFAIN 20:31 | EDU 20:50 | EDU 21:09 |
|---|---|---|---|
| Minutes since the bot's fresh ON bar | 171 (ON at 17:40) | no ON bar | no ON bar |
| Hours since the spike | 5.7 | 2.9 | 3.2 |
| Closes below VWAP just before the reclaim | 11 | 18 | 18 |
| Closes back above VWAP (streak) | 1 | 1 | 4 |
| Price vs spike VWAP at the click | +0.79 % | −0.77 % | +1.21 % |
| Volume vs normal (FRENZY needs 100×) | 72× | 28× | 24× |
| 5m ATR | 3.02 % | 1.65 % | 1.62 % |
| Off the episode peak | −15.1 % | −5.2 % | −3.3 % |
| Bounce off the low of the below-VWAP stretch | +10.8 % | +5.1 % | +7.2 % |
| Off the last hour's high | −2.7 % | −0.9 % | −0.1 % |
| Price vs EMA5 / 8 / 13 / 20 (5m) | +1.0 / +1.7 / +1.9 / +1.9 % | +0.2 / +0.6 / +0.9 / +0.9 % | +0.8 / +1.3 / +1.9 / +2.1 % |
| EMA order | 5 > 8, 13 ≈ 20 | 5 > 8 | 5 > 8 > 13 > 20 |
| Gap 5-20 (signed) now → previous bar | +0.84 → +0.52 (rising) | +0.66 → +0.57 (rising) | +1.32 → +1.14 (rising) |
| RSI(14) 5m | 56 | 54 | 61 |
| Price move in the 60 s before the click | **−0.30 %** | **−0.50 %** | **−0.72 %** |
| Taker-buy share, 60 s before | 37 % (sellers) | 43 % | 25 % (sellers) |
| Taker-buy share, 30 s after | 81 % | 46 % | 36 % |
| Market volume (gvol) on the last closed bar | 0.55 | 0.46 | **1.62** |
| Order book ±0.25 % imbalance at the click | +0.42 (bids) | +0.19 | +0.10 |

Common to all three:
- a flagged pair, 3 to 6 h after its spike;
- 1 to 1.5 h spent below the spike VWAP, then a fresh reclaim (1 to 4 closes above);
- volume well under FRENZY's 100×;
- EMA5 above EMA8, and gap 5-20 positive and rising;
- clicked into a small sell flush: price down 0.3 to 0.7 % in the prior minute, sellers dominant.

That is a **buy-the-first-dip-after-the-VWAP-reclaim** pattern, not a FRENZY-ON pattern. FRENZY needs a full hour above VWAP with 100× volume. By then GRIFFAIN had already run to its peak, and EDU never got there.

## Is it a repeatable pattern in your earlier FRENZY-era manual entries?

Same replay on your 92 earlier manual longs (09-29 → 10-03, four manual exports in reports/, deduplicated):

| What you were buying | Fills | Result |
|---|---|---|
| **Fresh reclaim**: streak 1–4 after ≥ 6 closes below VWAP | **1** (ONE 10-03 11:12) | −3.00 |
| Below VWAP (dip buys inside the below stretch): NMR, MOVR 09-30 and 10-02, SCR, SAND 20:02, CAP, MANA, STRK | 15 | 60 % won · −0.33 %/fill |
| Inside the ON state, chasing (mostly MOVR 10-01 scalps, plus SAND, PUMPBTC) | 38 | 68 % won · +0.28 %/fill (+1/−3-style scalps, near the 75 % chance rate) |
| Above VWAP but not ON: AIN 10-03 in the first 1–1.4 h after the spike (13 fills, 13–22 % above VWAP: 7 wins, 6 stops at −3 %), SAND's late streak, GTC, PUMPBTC, ATH, STRK | 20 | 65 % won · +0.34 %/fill |
| Not flagged (QNT, NEAR, SOL, ZEC…) | 18 | 50 % won · −0.13 %/fill (Momentum-exit era) |

A note on the AIN rows: six of them (15:40–15:52) came 3–5 closes after a 3-bar dip below VWAP. That is a "mini-reclaim", but 13–18 % above VWAP and about 1.2 h after the spike: 4 wins at +2 % and 2 stops at −3 %. It is not the same setup as today's.

So **today's reclaim-dip is new in your record**. Before today it appears once, and that one lost. Your record is a mix of styles, not one repeated pattern. Counting today, the reclaim signature has N = 4 manual fills (3 wins, 1 at −3 %), on 3 pairs and 2 days. That is anecdote. A +1 / −3 bracket wins 75 % of the time by chance.

## Has the rule already been tested?

Yes, and closely. In `FRENZY_BELOW_AVERAGE_TEST_2026-10-02.md`, the RECLAIM entry is "a close back at or above the spike VWAP after a full hour of closes below it". It was flagged-episode only, ≥ 2 h after the spike, 24 h volume ≥ $20M, bought at the next 5m open. The year result:
- 4,155 trades: **−0.31 %/trade** on the old live trail and −0.17 % on a +2 % target, with a 59 % win rate;
- negative in both halves, and 0 of 15 cells passed;
- cut by volume: 20–50× −0.25 %, 50–100× −0.55 %. Only the < 5× bucket was positive (+0.13 %).

Caveats on that test:
- It was a hand-rolled walk, not the engine's. Engine parity was not shown.
- Its costs were harsh: 0.11 % costs plus 0.10 % slippage, against 0.09 % plus 0.02 % today. That alone is worth about +0.1 to +0.15 %/trade, which still leaves it at about −0.05…−0.2.
- It required ≥ 12 closes below. GRIFFAIN had 11.
- It had no micro-dip trigger and no lock exit.

`FRENZY_REENTRY_NEW_ANGLES_2026-10-06.md` also refuted "pullback-then-reclaim" (EMA20 / exit-price reclaim), all four variants negative. And the Oct-1 hot-scalp work found that a pullback from the 5-minute high was the only separator that replicated, but not by enough to pay.

**Verdict:** the entry is causal and can be stated without hindsight. But its parent family has been tested and is negative. The parts that are new (the micro-dip trigger, streak 1–4, a below-run of ≥ 6, the lock exit, today's costs) are what a fresh test would have to prove. Prior expectation: fail. Today's trades are not evidence for it. Do not arm or observe-ship anything from this.

## Pre-registered year test (specified here, not run)

**RECLAIM_DIP v1**, frozen before any data is read:

1. **Cohort:** the engine's own FRENZY flags. Use `services.frenzy.frenzy_walk` + `frenzy_flagged` on 1500-bar 5m windows with the live normal-hour, plus the live shortlist: |24 h change| or 24 h range ≥ 15 % and 24 h volume ≥ $20M. Built the same way as `reports/FRENZY_ENGINE_COHORT_2026-10-05.csv`. Report parity against today's GRIFFAIN 17:35 ON bar and EDU 13:10 / 17:55 flag events before reading any result.
2. **Signal bar** (closed 5m bar *t*), all of:
   - the walk is not `in_state`;
   - `hours` ≥ 2;
   - `above_streak` between 1 and 4;
   - the closes just before that streak include **≥ 6 in a row below** the spike VWAP;
   - 24 h volume ≥ $20M;
   - at most one signal per (pair, below-stretch).
   - No ATR cap, no candle-colour rule, no volume-multiple rule.
3. **Entry:**
   - **Primary:** a resting buy at close(*t*) × (1 − 0.30 %), valid until the end of bar *t+1* (10 minutes from *t*'s close). It fills only if the 1-second/tick tape trades at or below it.
   - **Secondary, reported alongside:** market buy at *t*'s close, at LAG 0 and LAG 1 minute.
4. **Exits:**
   - **Primary:** the live FRENZY lock (stop −3 %; at +3 % lock +2 %, trail 2 points; 12 h cap), via `frenzy_exit_for(use_tp=True)` on ticks.
   - **Secondary:** fixed +1 / −3 (your FLOOR default) and fixed +3 / −3.
5. **Costs:** 0.09 % fees and 0.02 % slippage on stops and market fills, plus funding. No slippage on a resting limit.
6. **Pre-committed pass bar** (CLAUDE.md expectancy bar), on the primary cell only:
   - per-trade mean > 0 in **both** halves (Jan–Apr / May–Sep);
   - 95 % **day-clustered** bootstrap CI > 0;
   - ≥ 30 fills on ≥ 8 distinct days;
   - no single pair or day ≥ 50 % of the gain;
   - LAG 1 still > 0.
   - Then the 30–50 % in-sample haircut.
   - Also report trade-weighted and first-entry-per-episode results.
7. **Null:** the same statistic on random bars from the same episodes with `above_streak` ≥ 1 and the same fill model. The rule's timing must beat ≥ 95 % of 1,000 random draws.
8. **Controls, reported but not used to choose the cell:**
   - splits by gvol on *t* (< 1 / ≥ 1), volume multiple (< 50 / ≥ 50) and below-run length (6–11 / ≥ 12);
   - the Oct-2 RECLAIM definition re-run on the engine cohort, to tell "new costs" apart from "new trigger".

   No threshold may be re-fit after reading. A fail closes the family; there is no v2 on the same data.

## What would actually help

- **Stamp the FRENZY walk on every manual fill.** Today's MANUAL rows carry no `entry_frenzy_*` values, because the manual path does not stamp them. Every reading above had to be rebuilt. Stamping `vs_vwap`, `above_streak`, the below-run length, `vol_mult`, `hours` and `on_bar_ts` at the click would make each of your clicks a labelled example. The standing route to codifying your timing is ≥ 200 stamped clicks (project_frenzy_scalp_research). This needs an operator decision; no code was changed here.
- **The bot is not leaving FRENZY money on the table on these pairs today.** Its one GRIFFAIN signal was a stop within seconds, and EDU never qualified. The ORCA FRENZY_LONG at 18:05 (−3.0 %) is the one live FRENZY loss today, and it is the sleeve's own result, not a miss.

Scratch (replay, gvol, feature and exit scripts): `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/mvb/` (`replay.py`, `gvol.py`, `feat.py`, `earlier.py`, `manual_features.csv`, `earlier_states.csv`).
