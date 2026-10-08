# "Every new FRENZY pair goes up at least X %": study (2026-10-07)

## Plain-English summary

**The observation is true, but the strategy built on it loses money at every X tested.**

- Most newly flagged pairs do tick up a little after they appear. About **91 %** reach **+1 %** within 96 h (median wait about 15 min), and about **80 %** reach +3 %.
- The other 9–20 % are the problem. A pair that never reaches +1 % has usually collapsed: the median fall over 96 h is **−22 %**, the worst 10 % fall **−35 % or more**, and the worst single case fell −72 % at the close.
- Small wins and rare large losses do not balance. At X = 1 % the winners make +1 % each and the losers lose about −18 % each, so on average **every trade loses −0.77 %** (90.6 % win rate). The arithmetic: 0.906 × 1 − 0.094 × 17.9 ≈ −0.77.
- No X from 0.5 % to 20 % is profitable, with or without a stop. No exit is profitable either: no stop held 96 h, no stop held 24 h, or a −10/−15/−20 % stop. In all **100 cohort × exit × X cells, the day-block 95 % CI is below zero or straddles it** (best upper bound +0.04 %). In none is it above zero.
- Choosing the best X on Jan–May and testing it on Jun–Oct (and the reverse) also fails: −0.63 % and −1.41 % per trade out of sample.
- **The flag moment is a worse entry than a random time later in the same episode.** On the same pairs, random entries during the 72 h after the flag lose −0.27 % per trade (X = 1). Entering at the flag loses −0.92 %. The flag tends to mark the top of the spike.
- **On a $3,000 cross-margin book** at FRENZY_LITE size ($630 at 4×, at most 3 positions), every version and every start month loses. Most runs fall to about $400–600 in 1–3 months, at which point there is no free margin left to open a trade. The worst single trade costs **−$1,247 at 4×** (−$1,996 at 6.4×), about 40–65 % of the account in one trade.
- **Verdict: no X works.** I recommend no arm and no scout line: this cohort is already measured here on 2,281 events and is clearly negative. "100 % win rate" is really 90 % with a −18 % average loss.

## 1. Cohort and engine parity

- **Signal.** The real `services.frenzy.frenzy_walk` + `frenzy_flagged` with the live `trading_config.json` (spike 5 % in 30 min, ≥ 20× normal hour, ≥ $2 M hour; verified; ≤ 96 h). Bars:
  - Jan 10 → Oct 1: the existing exact year walk (`manualall/yw_all.pkl`).
  - Oct 1 → Oct 8 01:55 UTC: a new walk on a fresh 5m read (`flag_x/octwalk.py`).
- **Parity between the two walks.** On the overlapping day (Oct 3), all 5,747 flagged bars are identical, with 0 missing and 0 extra.
- **Event.** One event per episode: its first flagged bar. I dropped 504 "re-anchor" episodes where the pair was already flagged on the previous bar (e.g. ORCA 10-06 05:35, MET 16:50, BR, ARK, AIN). They are not *new* on the tab.
- **(b) All eligible:** 3,191 events. Eligible means:
  - coin underlying, not Alpha, listed ≥ 90 days
  - not in pair_blacklist, no_trade_pairs or BTC/ETH
  - no volume or shortlist condition
- **(a) Engine-reachable:** 2,320 events. The live FRENZY shortlist rule (`frenzy_shortlist`) is rebuilt on every 5m bar for the whole year (`flag_x/shortlist.py`):
  - |24 h change| or 24 h range ≥ 15 %
  - 24 h volume ≥ $20 M
  - universe filters
  - cap 25, ranked as in the engine
  - The event is the first flagged bar at which the pair is on the shortlist. Median 0.4 h after the spike close; 25 % arrive ≥ 2.8 h late.
- **Live check (spike close times, UTC).** Each walk episode matches the time the operator gave:

  | Pair | Date | Spike close (UTC) |
  |---|---|---|
  | SAND | 10-07 | 00:55 |
  | HEMI | 10-07 | 07:30 |
  | TA | 10-07 | 14:45 |
  | AIN | 10-07 | 15:45 |
  | MET | 10-07 | 16:45 / 16:50 |
  | PARTI | 10-07 | 03:50 |
  | NMR | 10-06 | 11:40 |
  | GRIFFAIN | 10-06 | 14:50 |
  | ORCA | 10-06 | 05:35 (the walk shows it from 13:55, after the previous ORCA episode ended) |
  | EDU | 10-06 | 17:55 (the bar closing 17:50 is the flag bar) |

  - **One miss: GTC 10-07 03:05.** The walk finds that exact spike but marks it **unverified**: 94 bars at ≥ 100× volume in the 25 h before (the Oct-6 GTC pump), so `frenzy_flagged` = False. Live may have seen a slightly different 1500-bar window. GTC is not in the cohort.
- **Limitation (survivorship).** k5m_full holds the pairs that still exist. Pairs delisted during the year are missing. Delistings are mostly crashes, so the true results are, if anything, worse.

## 2. Pricing (local cache only)

- **Entry.** The flag bar's close + 8 s, plus 0.10 % slippage.
  - Price: the first aggTrade ≥ that time where the tick archive was cached (1,886 / 2,320 in (a)); otherwise the next 5m open.
- **Costs.**
  - Fees: 0.09 % round trip.
  - Market exits (stop / expiry): another 0.10 % slippage.
  - TP: a limit order at +X % **net**, no slippage.
- **Price path.** **5m bars for every event** (k5m_full + a fresh 5m read for Oct 4–8).
  - I checked 1m first. Cached 1m covers only 41 % of events, and it is **selection-biased**: k1m / tick days were fetched for pair-days the bot traded or studied. Those events show a +8.3 % mean 96 h return, against −10 % for the rest.
  - So the 1m-mixed run was discarded; it gave the same negative signs anyway. The uniform 5m source has no such selection.
- **Conservative fill rule.** The entry bar's high is not counted (part of it is before the fill), and a stop and a TP inside the same bar count as the stop.
- **Optimistic re-run.** Counting the entry bar's high lifts small X by about +0.6 % but leaves every cell ≤ 0 (best upper CI bound +0.06 %). Zero costs would add about +0.2–0.3 %, which is still negative.

## 3. Hit rates by X (cohort a, N = 2,311 at 24 h / 2,281 at 96 h; conservative rule)

| X (net) | hit ≤ 24 h | hit ≤ 96 h | median time to hit | 90th pct time | worst dip before the hit: median / 10th pct / 1st pct / min | never hit in 96 h: N · median fall · 10th pct · worst |
|---|---|---|---|---|---|---|
| 0.5 % | 90.6 % | 93.1 % | 0.16 h | 2.7 h | −1.7 / −6.7 / −20 / −58 | 157 · −23 % · −37 % · −72 % |
| 1 % | 87.4 % | 90.6 % | 0.25 h | 5.2 h | −1.8 / −7.6 / −22 / −79 | 214 · −22 % · −35 % · −72 % |
| 1.5 % | 83.9 % | 87.6 % | 0.33 h | 7.7 h | −1.9 / −8.4 / −22 / −79 | 282 · −21 % · −34 % · −72 % |
| 2 % | 81.0 % | 85.2 % | 0.41 h | 10.4 h | −2.1 / −8.8 / −22 / −79 | 338 · −20 % · −34 % · −73 % |
| 3 % | 74.7 % | 80.4 % | 0.66 h | 17.2 h | −2.6 / −10.3 / −22 / −79 | 447 · −20 % · −34 % · −73 % |
| 5 % | 63.5 % | 71.5 % | 1.6 h | 27.9 h | −3.2 / −11.9 / −24 / −79 | 649 · −19 % · −33 % · −73 % |
| 7 % | 53.9 % | 64.1 % | 3.3 h | 39 h | −3.8 / −13.1 / −26 / −79 | 820 · −19 % · −33 % · −73 % |
| 10 % | 43.9 % | 55.1 % | 5.1 h | 50 h | −4.3 / −14.3 / −27 / −79 | 1,025 · −18 % · −32 % · −73 % |
| 15 % | 32.3 % | 42.9 % | 8.3 h | 57 h | −4.8 / −15.5 / −29 / −59 | 1,303 · −18 % · −32 % · −97 % |
| 20 % | 24.7 % | 34.7 % | 10.8 h | 58 h | −5.1 / −15.7 / −28 / −57 | 1,489 · −17 % · −32 % · −97 % |

- **Cohort (b), N = 3,139.** Within about 1–4 points of (a): +1 % hits 87.6 % within 96 h, +3 % hits 78.3 %.
- **Median MFE / MAE after the flag:**

  | Window | MFE (max gain) | MAE (max loss) |
  |---|---|---|
  | 1 h | +2.7 % | −3.2 % |
  | 4 h | +4.5 % | −5.4 % |
  | 24 h | +8.2 % | −9.6 % |
  | 96 h | +11.7 % | −15.0 % |

  The MAE in 96 h is −31 % or worse in 10 % of events and −60 % in 1 %. DEXE in July fell 36 → 1.3 (−96 %), TAC −94 %.

## 4. Strategy P&L (cohort a; % of position per trade, after costs)

| Exit | X | N | WR | avg | day 95 % CI | worst | 1st / 2nd half | max losing streak | avg win / avg loss |
|---|---|---|---|---|---|---|---|---|---|
| no stop, 96 h | 0.5 | 2,281 | 93.1 % | −0.86 | [−1.11, −0.61] | −56.8 | −1.06 / −0.66 | 2 | +0.50 / −19.3 |
| no stop, 96 h | 1 | 2,281 | 90.6 % | −0.77 | [−1.06, −0.49] | −56.8 | −1.03 / −0.51 | 3 | +1.00 / −17.9 |
| no stop, 96 h | 3 | 2,281 | 80.5 % | −0.81 | [−1.23, −0.41] | −72.1 | −0.99 / −0.63 | 5 | +3.00 / −16.6 |
| no stop, 96 h | 10 | 2,281 | 57.6 % | −0.87 | [−1.57, −0.21] | −72.1 | −1.06 / −0.67 | 9 | +9.7 / −15.2 |
| no stop, 24 h | 1 | 2,311 | 87.4 % | −0.78 | [−1.02, −0.54] | −67.1 | −0.98 / −0.59 | 4 | +1.00 / −13.1 |
| no stop, 24 h | 3 | 2,311 | 75.1 % | −0.66 | [−1.01, −0.30] | −67.1 | −0.77 / −0.55 | 6 | +2.99 / −11.6 |
| stop −10 % | 2 | 2,281 | 78.0 % | −0.61 | [−0.84, −0.38] | −10.2 | −0.61 / −0.61 | 5 | +2.00 / −9.9 |
| stop −15 % | 3 | 2,281 | 77.3 % | −0.70 | [−1.02, −0.39] | −15.2 | −0.82 / −0.57 | 6 | +3.00 / −13.3 |
| stop −20 % | 1 | 2,281 | 89.5 % | −0.80 | [−1.06, −0.57] | −20.2 | −0.97 / −0.64 | 3 | +1.00 / −16.2 |

- **All 50 cells per cohort** (5 exits × 10 X) are in `flag_x/strat_5c.out`. Every average is negative. The best upper CI bound is **+0.04 %** (cohort b, no stop 96 h, X = 10, avg −0.60).
- **Per month (no stop 96 h, X = 1):**

  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct (partial) |
  |---|---|---|---|---|---|---|---|---|---|
  | −2.52 | −1.05 | −0.45 | −0.97 | −0.53 | −0.63 | −0.76 | −0.22 | −0.20 | −0.71 |

  Not one positive month. September was the best month for X = 10 (+1.37), but its neighbours are negative.
- **The 30–50 % in-sample haircut does not apply:** there is no positive in-sample Δ to cut.

## 5. Selection and null tests

- **X chosen on Jan–May → tested Jun–Oct:**
  - (a) the best cell was a −10 % stop with X = 2 (−0.59 in sample). Out of sample it made **−0.63**, CI [−0.93, −0.34].
  - (b) the best cell was a −10 % stop with X = 5. Out of sample it made **−0.45**, CI [−0.89, +0.01].
- **The reverse (chosen on Jun–Oct → tested Jan–May):**
  - (a) no stop 96 h, X = 10, made **−1.41**, CI [−2.31, −0.52].
  - (b) the same cell made **+0.28 in sample but −1.27 out of sample**.
- **Random-entry null.** Same pairs, 5 random times each within 72 h after the flag, same entry convention (bar open + slippage):

  | Rule | Flag moment: WR / avg | Random times: WR / avg |
  |---|---|---|
  | X = 1, 24 h | 86.7 % / −0.88 | 84.9 % / −0.28 |
  | X = 1, to expiry | 90.0 % / −0.92 | 88.7 % / −0.27 |
  | X = 3, to expiry | 80.5 % / −0.81 | 74.1 % / −0.30 |
  | X = 10, to expiry | 57.4 % / −0.94 | 50.1 % / −0.61 |

  The flag moment buys a few % more small wins but loses more per trade. **The flag moment adds negative value.**

## 6. Book simulation: $3,000 CROSS account

- **Margin mode.** No code in services/ sets `marginType` / margin mode. Binance's default is CROSS, and binance_service reads the cross-wallet margin-ratio fields, so CROSS is modelled. The paper engine does not simulate liquidation.
- **Setup:**
  - $630 margin per trade at 4× ($2,520 notional) or 6.4× (FRENZY_LITE lev 0.32 × 20 = $4,032)
  - ≤ 2 or 3 concurrent positions
  - a new trade only if equity minus used margin ≥ $630
  - equity marked at bar lows
  - account liquidation = equity ≤ 2 % of open notional
- **Results:**
  - Every variant (no stop 96 h with X 0.5 / 1 / 3 / 10; no stop 24 h with X 1 / 3; −15 % stop with X 3; −20 % stop with X 1; −10 % stop with X 2), at both leverages, both slot caps and **every start month (Jan, Mar, May, Jul, Aug, Sep)**, ends below $3,000.
  - Most end at about **$170–620**, with max drawdown −80 … −110 %; at that point the account can no longer post margin.
  - Best case: no stop 96 h, X = 3, starting Sep 1 → **$3,579**, but with a −52 % drawdown along the way.
  - Account-level liquidation is reached in some 6.4× / no-stop runs and in 1 of the 4×/3-slot runs (no stop 24 h, X 3, start May).
- **Worst single trade (no stop):**
  - 4×: **−$1,247** (BARD 03-18, −56.8 %)
  - 6.4×: **−$1,996**
  - X = 3 at 4× (start May): −$1,818 (−72 %)
- **One pair cannot liquidate a $3,000 cross account by itself** at these sizes; it would need roughly −100 %+ at 4×, or about −74 % at 6.4×. But a string of −20…−50 % trades drains the account long before that.

## 7. The cases you named (cohort a)

- **Pairs that fell and never reached the target:**
  - GRIFFAIN (10-06 17:35 entry): never reached +1 %, −17.6 % in 4 h, −47 % in 24 h.
  - DIA 10-06: never > 0, −17 %.
  - MAGIC 10-07: −11.8 %.
  - MOVR 10-07: −13 %.
- **Pairs that hit +1 % / +3 % only after a deep dip:**
  - TA 10-07: −6.9 % dip, then −25 % in 24 h.
  - RESOLV 10-03: +1 %, then −28 %.
  - 1000000BOB 10-05: +1 % first, then −56 %.
- **RLC 10-05 (11:00):** reached +1 % at 1.1 h after a −5.3 % dip, then ran +128 %.
- **Typical clean winners:** SAND 10-02 (+66 %), AIN 10-03 (+150 %), MET 10-07 (+33 %).
- **Wins are capped at X; losses are not.** That asymmetry is the whole result.

## 8. Relation to earlier refutations

- **Runaway / lift-off sleeves (0 of 128 cells, Oct-1):** they found that pumps mean-revert over 24 h, and the same holds here. The 96 h buy-and-hold median from the flag is −2…−11 % depending on the subset.
- **FRENZY pre-ON lines (Oct-6) and HOLD_LOWVOL / FRENZY_LITE:** these enter *later*, on a staircase / VWAP hold.
- **This idea is different:** an unconditional buy at the flag with a TP-only exit. It turns out to be the worst of these timings, worse than random later entries.
- **The "high WR, net-losing" pattern** here is not a sizing artefact; it holds at 1×. Under the expectancy bar, a 90 % WR is below this cohort's breakeven WR (≈ 95 % at X = 1, since |avg loss| / (win + |loss|) = 17.9 / 18.9).

## 9. Verdict

- **No X qualifies.** None is positive out of sample with a day-CI above 0 after costs, under any exit.
- **The real WR at X = 1 is about 90 %, not 100 %.** About 1 in 11 flags never reaches +1 % and falls about 22 % (worst 10 %: −35 % or more).
- **Recommendation: do not arm, and do not add a scout line.** The question is fully answered on 2,281 engine-reachable events. Re-examine only if a *new* data source (order book / OI / funding at the flag) separates the 9 % collapsers before entry.

## 10. Data and network log

- **Scripts and pickles** (all scratch): `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/flag_x/`
  - `octwalk.py`, `shortlist.py`, `events.py`, `paths5.py`, `core.py`, `hits.py`, `hittab.py`, `strat.py`, `extra.py`, `null.py`, `book.py`, `book2.py`
  - outputs `*_5c.pkl` (conservative) and `*_5o.pkl` (optimistic)
- **Binance requests in this study:** 589 `fapi/v1/klines` calls (5m, limit 1500, ≈ weight 10 each). They ran 22:56:05–22:58:23 local, with a peak of about 240 calls/min ≈ **2,400 weight/min, right at the IP limit**. This plausibly contributed to the frequency warning.
  - Errors were retried silently, so a 418/429 cannot be ruled out. All 589 reads came back complete (1,464 bars each). X-MBX-USED-WEIGHT was not read.
  - No data.binance.vision downloads. All work after the coordinator's pause used local caches only.

## 11. Addendum: stops sized by account risk (operator, cross margin, equity ≈ $2,650, margin $630/trade)

A −7.0 % stop is about 10 % of the account at 6×, and a −10.5 % stop about 10 % at 4×. Cohort a, N = 2,281, stops checked inside the 96 h flag window, conservative rule (`flag_x/addstops.py`).

| Stop | Best X | WR | avg | day 95 % CI | Jan–May / Jun–Oct | worst trade |
|---|---|---|---|---|---|---|
| −7.0 % | 3 | 65.4 % | −0.50 | [−0.70, −0.31] | −0.44 / −0.58 | −7.2 % |
| −7.0 % | 1 | 80.1 % | −0.62 | [−0.74, −0.49] | −0.61 / −0.62 | −7.2 % |
| −10.5 % | 2 | 78.7 % | −0.62 | [−0.86, −0.40] | −0.59 / −0.66 | −10.7 % |
| −10.5 % | 1 | 85.4 % | −0.67 | [−0.83, −0.50] | −0.67 / −0.66 | −10.7 % |

- Every X from 0.5 to 20 under both new stops is negative in both halves of the year. The best upper CI bound across all new-stop cells is −0.21 %.
- Worst single-trade loss as % of a $2,650 account:

| Exit | at 4× ($2,520 notional) | at 6.4× ($4,032) |
|---|---|---|
| stop −7.0 % | −6.9 % | −11.0 % |
| stop −10 % | −9.7 % | −15.5 % |
| stop −10.5 % | −10.2 % | −16.2 % |
| stop −15 % | −14.4 % | −23.1 % |
| stop −20 % | −19.2 % | −30.7 % |
| no stop (X = 1, 96 h) | −54 % | −86 % |
| no stop (worst over all X) | −79 % | −127 % |

- Tighter stops cap the tail and shrink the loss per trade a little: −0.50 at a −7 % stop against −0.77 with no stop. But they also stop out many trades that would later have hit the TP, so no stop width turns the cohort positive.
- Gaps through the stop are priced at the bar open: the worst outcome is −7.2 % against a −7.0 % stop.
