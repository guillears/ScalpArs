# Last 12 hours: deep assessment (2026-10-06 00:00 → 12:17 UTC)

Read-only. I changed no code, config, scout state or master file, and committed nothing.

**Sources**
- Fills: `~/Downloads/scalpars_orders_paper_2026-10-06_12-17-50.csv`, ids 15–24.
- Price paths: public Binance 1m klines (each fill ±60 min), plus BTC 5m and 1h since Jun-10.
- Server logs: EB bundle pulled 12:35 UTC (`web.stdout.log`). It covers 00:00–12:35 with no hourly gap.
- Decision journal: `scalpars_decisions_paper_2026-10-06_03-55-33.csv`. It ends at **04:00 UTC**, so it covers only the first 4 fills.
- Master: `reports/MASTER_POOL_stacked.csv` (B1–B16, 526 full-size fills), plus B17 from the export.

**Bottom line**
- On equity this is the worst 12 h on record. The account went from about $3,940 to $2,650, which is **−33 %**.
- No sleeve had an unusual 12 h on its own terms. Every sleeve that traded lost in the same window.
- Three things drove the loss: the size of each stop, three 2× fills, and the FRENZY −3 % stops.
- BTC's 12 h was ordinary, at the 40th–50th percentile on every measure.
- I found no mechanical bug that cost money.
- One gate in CURRENT_STATE has fired by its own wording: CALM3D 2× → 1×. One more gate is certain to fire: mom-short PVR.

---

## 1. Per-trade anatomy

### 1a. The fills

Path figures are direction-signed versus the entry price and come from 1m klines. "pre30" is the pair's move in the 30 min before entry.

| # | pair · sleeve · size | open UTC | P&L % · $ | 1m peak (when) | MAE (when) | hold | after exit (60 min) | main cause |
|---|---|---|---|---|---|---|---|---|
| 15 | PUMP · FLIP short · **2×** [TG_SHALLOW+NEGDI15] | 02:06 | +0.12 · +47 | +0.67 (4 m) | −0.20 (1 m) | 5 m | went on to +1.67 | winner; the trail left money |
| 16 | QNT · FLIP short · **2×** [NEGDI15] | 02:11 | **−1.15 · −438** | +0.11 (at entry) | −1.51 (6 m) | 6 m | +1.29 then −1.62 | **bad entry**: shorted into a pump still running (pre30 +1.89 %). Never positive. 2× doubled it |
| 17 | FLUID · FRENZY_WIDE · lev 0.2 | 02:30 | −3.04 · −101 | +0.32 (4 m) | −2.99 (11 m) | 11 m | **−11.1 %** | bad entry; the stop saved about 8 pts |
| 18 | FET · MOM short · 1× | 02:47 | −0.67 · −107 | +0.24 (0 m) | −0.64 (15 m) | 16 m | −0.84 trough | **bad entry**: sold at range position 4.9 (the floor), pair RSI 33.8 |
| 19 | BTW · FLIP short · 1× (lev 10, bracket cap) | 06:22 | −1.19 · −93 | −0.12 | −1.19 (3 m) | 4 m | **+2.13 peak** | **exit/noise**: direction right, stopped on the wick first |
| 20 | NIL · FLIP short · 1× ($500, bracket cap) | 06:29 | +0.26 · +26 | +0.67 (10 m) | −0.46 (1 m) | 11 m | +0.61 | winner |
| 21 | ORCA · FRENZY_LONG · lev 0.32 | 09:40 | −3.00 · −137 | +0.65 (5 m) | −3.22 (24 m) | 24 m | −3.95 | bad entry: +6.8 % above VWAP, bear breadth 74 % |
| 22 | UMA · FRENZY_LONG · lev 0.32 | 10:10 | −3.00 · −132 | +1.96 (59 m) | −2.93 (110 m) | 110 m | −3.75 | no follow-through: lock needs +3, peak was +1.96 |
| 23 | TAO · MOM long · **2× CALM3D** | 10:44 | −0.69 · −201 | +0.08 (at entry) | −0.70 (26 m) | 26 m | −0.81 | **bad entry**: never positive. No recovery hold (BTC RSI 55 < 60), which is correct |
| 24 | SAND · MOM long · **2× CALM3D** | 11:37 | −0.56 · −153 | +0.15 (3 m) | −0.76 (13 m) | 13 m | +0.57, then −0.50 | bad entry; the hold saved +0.14 pt vs the stop |

**Counts**
- 8 losers. 3 were never positive (QNT, BTW, TAO).
- Bad entry (no follow-through): 7.
- Exit / noise: 1 (BTW).
- Market move against a correct idea: 0. Two shorts lost in a falling BTC and two longs lost in a rising BTC. The individual pairs moved against the bot, not the market.
- Every stop that fired saved money versus holding 60 more minutes, except BTW and SAND.

### 1b. Entry stamps

| # | BTC regime | BTC gap | BTC RSI (closed) | BTC ATR | bull / bear % | gvol | pair PVR | pair RSI · ADX · +DI/−DI | pair ATR | other |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 PUMP | HEALTHY_BEAR | −0.09 | 36.0 | 0.12 | 18 / 78 | 1.12 | 0.91 | 58.6 · 23.4 · 31.3/17.5 | 0.49 | rng 73 |
| 16 QNT | HEALTHY_BEAR | −0.13 | 32.4 | 0.12 | 18 / 80 | 1.27 | 0.78 | 60.5 · 25.7 · 28.0/16.0 | 0.77 | rng 82 |
| 17 FLUID | HEALTHY_BEAR | −0.13 | 49.1 | 0.14 | 20 / 73 | 0.99 | – | 58.4 · 43.6 · 26.7/12.1 | 1.90 | green bar +0.05, above_share 44 %, ADXΔ −1.76 |
| 18 FET | HEALTHY_BEAR | −0.15 | 43.4 | 0.14 | 16 / 75 | 0.77 | 0.61 | 33.8 · 28.7 · 12.8/27.5 | 0.39 | rng **4.9** |
| 19 BTW | STRONG_BEAR | −0.16 | 33.1 | 0.10 | 11 / 89 | 0.95 | 0.63 | 56.0 · **19.4** · 23.7/**14.8** | 1.48 | rng 54 |
| 20 NIL | STRONG_BEAR | −0.18 | 32.0 | 0.10 | 16 / 80 | 0.89 | 1.22 | 56.5 · 28.7 · 27.1/**13.1** | 0.90 | rng 73 |
| 21 ORCA | CHOPPY_FLAT | +0.22 | 54.3 | 0.15 | **20 / 74** | 0.95 | 0.75 | 59.4 · 32.9 · 25.6/16.4 | 1.74 | red −0.34, vs VWAP **+6.8 %**, above_share 70 %, ADXΔ −0.50 |
| 22 UMA | STRONG_BULL | +0.22 | 56.5 | 0.15 | 48 / 35 | 0.82 | 0.62 | 53.8 · 23.5 · 26.6/17.6 | 1.66 | flat −0.04, above_share **16 %**, 21.7 h after the spike |
| 23 TAO | STRONG_BULL | +0.23 | 56.2 | **0.139** | 72 / 20 | 0.73 | 0.65 | 60.8 · 23.6 · 29.9/17.6 | 0.30 | rng 86, rank 27 |
| 24 SAND | STRONG_BULL | +0.25 | 62.9 | 0.127 | 46 / 41 | 0.75 | 0.68 | 58.8 · 27.0 · 30.8/13.9 | 0.42 | funding −0.076 % |

### 1c. For each loser: the closest gate and the observe lines that flagged it

| # | closest live gate | observe / watch lines that flagged it |
|---|---|---|
| 16 QNT | NEGDI15 2× cell: its revert gate fired on this fill and the cell is already 1× (deploy 04:09 UTC) | −DI < 15 observe: **no** (16.0) |
| 17 FLUID | market-volume gate 194: gvol 0.99 vs the 1.0 bar, **missed by 0.01** | **WIDE_CHOPPY_OBS yes** (44.4 ≤ 67.8). ADXΔ > 0 watch: fails. WIDE ATR_HIGH candidate: no (1.90) |
| 18 FET | mom-short PVR 0.86: kept side (0.61). `range_position_min_short` 2.0, close at 4.9 | none |
| 19 BTW | flip stop | **−DI < 15 observe yes** (14.8). **STRONG_BEAR pADX < 21 exemption yes** (19.4). Both observe lines caught it |
| 21 ORCA | gate 194: gvol 0.95, near the bar | ADXΔ > 0 watch: fails. Earlier FRENZY_SHORT_OBS breakdowns in this episode (00:30, 02:55). Breadth bear 74 % |
| 22 UMA | gate 194: gvol 0.82 (passes) | above_share 16 % would fail the WIDE choppy line, but that line covers WIDE only. Earlier breakdowns at 00:10, 02:00, 08:20 |
| 23 TAO | CALM3D BTC-ATR ceiling 0.147: 0.139, close | heat block 0 flags (bull 72 < 80). ML_B1H_NEGFLANK: no (slope +0.12). LONG_CHOP_BURST: no (eff72 0.037) |
| 24 SAND | recovery hold (worked as designed) | none |

**Not flagged by any gate or observe line:** QNT, FET and TAO, together −$746.

---

## 2. Market context, in hour and day units

### 2a. BTC over the 12 h

| hours UTC | BTC | what happened | bot regime label (from fills) |
|---|---|---|---|
| 00–06 | 85,717 → 85,267 (low 85,073) | slow drift down, −0.5 %. Hourly volume $0.13–0.56 B (thin) | HEALTHY_BEAR (02:06–02:46) → STRONG_BEAR (06:22–06:29) |
| 07–09 | → 85,949 | +0.8 % rise | CHOPPY_FLAT (09:40) |
| 10–12 | → 86,211 (high 86,349) | slow grind up, +0.3 % | STRONG_BULL (10:10–11:37) |

**Other conditions**
- BTC 5m ATR stamps were 0.10–0.15 %, which is normal.
- Funding was near zero: −0.0016 % to −0.0025 % per 8 h.
- BTC sat 1.3–2.6 % below its 30-day high. Not washed out, so REBOUND was off.
- Alt breadth (fill stamps only): **bear 73–89 % all morning**, while BTC fell only 0.5 %. Alts were weaker than BTC. In the rally, bull breadth went 20 % → 48 % → 72 % → 46 %. That is a weak, uneven rally.

**This 12 h compared with every 12 h block since Jun-17 (223 blocks)**

| measure | today | percentile |
|---|---|---|
| BTC range | 1.50 % | 40th |
| BTC efficiency (net move / path) | 0.065 | 50th |
| BTC quote volume | $4.0 B | 44th |

**Prior B17 days (BTC per UTC day)**

| day | mean 5m true range | day return | day range |
|---|---|---|---|
| 10-03 (Sat) | 0.041 % | +0.27 % | 0.69 % |
| 10-04 | 0.067 % | +2.09 % | 2.47 % |
| 10-05 | 0.142 % | −0.89 % | 2.43 % |
| 10-06 (to 12:20) | 0.116 % | +0.48 % | 1.53 % |

Today was a middle-of-the-road day.

### 2b. Was this a regime shift?

I tested the obvious "whipsaw" idea in window units on the master (145 twelve-hour blocks with fills, 550 fills).

| window type | windows | avg %/fill | Δ vs rest (95 % window bootstrap) |
|---|---|---|---|
| fills stamped both BEAR and BULL in the same 12 h (today = yes) | 47 | +0.07 | **+0.09 [−0.09, +0.29]**: not worse |
| BTC EMA20/50 gap swung both ways by >0.1 % (today = yes) | 80 | −0.02 | **−0.003 [−0.19, +0.19]**: no effect |
| BTC efficiency, low third | 25 | −0.03 | weak lean only. Today sat in the middle third |

**Verdict:** no measurable regime shift. Intraday label flips do not predict bad 12 h blocks in the master.

### 2c. Uniform-degradation test

| sleeve | B17 before today | the 12 h | master (all, as traded) |
|---|---|---|---|
| MOM long | 4 · 100 % · +1.30 % | 2 · 0 % · −0.63 % | 173 · 69 % · +0.05 % |
| MOM short | 1 · 100 % · +0.25 % | 1 · 0 % · −0.67 % | 47 · 62 % · +0.04 % |
| FLIP | – | 4 · 50 % · −0.49 % | 49 · 67 % · +0.08 % |
| FRENZY_LONG | 3 · 33 % · −0.85 % | 2 · 0 % · −3.00 % | (6 fills ever) |
| FRENZY_WIDE | 4 · 50 % · 0.00 % | 1 · 0 % · −3.04 % | (5 fills ever) |
| SPIKE_FADE | 2 · 100 % · +0.33 % | 0 | 111 · 67 % · +0.08 % |

**Reading**
- Every sleeve that traded did worse than it had earlier in the batch. By the checklist, that pattern means "suspect a common regime variable".
- Against that:
  - N is 1–4 per sleeve.
  - The losses were on **both** sides: shorts in a falling BTC, longs in a rising BTC.
  - The BTC 12 h was ordinary, and the whipsaw tests in 2b find nothing.
- The common factor I can see is **alt-specific**. Alts were weak against BTC all morning (bear 73–89 %), and the late-morning rally had thin breadth. That fits "no follow-through on either side".
- I can't test it in window units: breadth is only stamped on fills, and the journal stops at 04:00.
- **So:** I found no regime variable. This is not a reason to touch any sleeve. The hunt stays open (blind spots in section 7).

---

## 3. Sizing

| scenario | 12 h $ | Δ vs as traded |
|---|---|---|
| **As traded** | **−1,289** | – |
| Flip cells at 1× (today's change: QNT −438 → −219, PUMP +47 → +23) | −1,093 | +196 |
| Flip 1× **and** CALM3D 1× (TAO −201 → −100, SAND −153 → −76) | **−917** | +373 |
| Also refuse FLUID (WIDE choppy line, observe-only today) | −816 | +473 |

**Notes**
- **The 2× cells cost $372**, 29 % of the loss. QNT alone was $219 of that.
- **Deploy timing:** the flip change went live at **04:08–04:09 UTC** (EB events), not 04:18. PUMP (02:06) and QNT (02:11) opened before it. BTW and NIL would have been 1× under either config.
- **The % loss is the same at any size:** −1.29 %/fill. Size changes only the $.

**Loss by sleeve (as traded):**

| sleeve | $ | share of the loss |
|---|---|---|
| FLIP | −$458 | 36 % |
| MOM long (2×) | −$354 | 27 % |
| FRENZY + WIDE | −$370 | 29 % |
| MOM short | −$107 | 8 % |

**What one stop costs today** (equity ≈ $3,940 at 00:00, from `PORTFOLIO_OPEN` log lines):

| stop | notional | cost per stop |
|---|---|---|
| 1× base ticket (≈ 24 % of equity as margin, 20× lev) | ≈ 4.8× equity | – |
| 2× flip, stopped at −1.15 % | ≈ 9.6× equity | **≈ −11 % of equity** (QNT) |
| 2× momentum, stopped at −0.7 % | ≈ 9.3× equity | ≈ −6 % |
| FRENZY_LONG, stopped at −3 % (6×) | ≈ 1.2× equity | ≈ −3.5 to −4 % |

So eight ordinary stops in one morning take a third of the account. That is the sizing, not a broken engine.

---

## 4. Was anything mechanical wrong?

| check | result |
|---|---|
| RH_PREMISE_EXIT on SAND | **As designed.** Trigger 11:48:51 at −0.70 (BTC closed RSI 63.57 ≥ entry 62.87, inside 60–66). Exit 11:50:03 when the new closed bar gave 62.27 < 62.87. My independent RSI rebuild gives 63.55 / 62.25. It exited at −0.56, which is 0.14 better than the stop |
| TAO not held | **Correct.** At the stop (11:10) the closed RSI was 55.2: below 60 and below its entry value of 56.2 |
| EMA13_CROSS_EXIT on FET | **As designed.** First cross 02:59:34 at −0.42 was held, because strict mode needs the EMA5/8 stack to flip. Close at 03:02:50 at −0.67, after the stack flipped. Post-exit low −0.84 |
| Stops | FRENZY −3.00/−3.04 · momentum −0.69 · flip −1.15/−1.19, all at their levels. No gap-through |
| Entry bar | FRENZY fills 7–8 s after the 5m close (limit 120 s). The signal-bar returns rebuilt from 1m klines match the stamps exactly (ORCA −0.343, UMA −0.044, FLUID +0.046). FLUID went to WIDE because its bar was green, as designed |
| ORCA / UMA sizing | **Correct.** Both had ADX Δ < 0 (−0.50 / −0.003), so normal lev 0.32. The logs confirm `lev=0.32x via FRENZY_LONG` |
| ORCA #6 / NIL #9 at 1.0× while 2× was live | **By design.** This is the Aug-10 **crowd-sprint de-mux** (gate 42): global volume > 0.74 (0.7425 / 0.99) ∧ BTC EMA20 slope > 0.07 (0.092 / 0.091) → 1.0×. **Stamp gap:** `cell_multiplier_source` still reads "UNMATCHED", so the de-mux is invisible in the CSV. That is why the batch review couldn't explain it |
| FRENZY_KL_MISMATCH | 293 warnings in 12 h, about 1 per pair-check. Since the diagnostic deploy (01:31 UTC), **every one says "0 bar(s) differ"** and differs only in length: cache 1001–1012 bars vs a full read of 1000. The full read asks for 1500 and gets 1000. **No price data differed, no P&L effect.** But the alarm now fires on every full read, so it would hide a real mismatch |
| Errors | One `[MONITOR] stamp commit failed (database is locked)` at 11:10:09, right after TAO closed. It says the next cycle re-stamps, and TAO's row is complete. One PAIR_DATA cache-write lock at 08:15. **No error on any entry or exit path** |
| BALANCE_SYNC "drift" (28 lines) | The in-memory free USDT lags the DB by exactly one position's margin around each open or close, and is corrected each time. It already existed (59 lines on 10-03). Base tickets track equity, so I see no sizing effect |
| Deploys | 01:31 (219 diagnostic) · 04:08 (220/221 flip 1×) · 12:14 and 12:33 (after the window) |

**Verdict:** nothing mechanical cost money. There are two hygiene fixes, both logging only and both for the operator to approve:
- Stamp the crowd-sprint de-mux in `cell_multiplier_source`.
- Make the KL-mismatch check compare only the overlapping bars.

---

## 5. Is this normal variance?

**Fixed 12 h blocks with ≥ 1 full-size fill, master + B17, Jun-17 → today (145 blocks)**

| measure | today | rank |
|---|---|---|
| $ as traded | −$1,289 | **worst of 145** (next: 08-22 PM −$1,167) |
| $ at 1× | −$917 | 2nd worst (08-22 PM −$1,026) |
| sum of P&L per base ticket (leverage-aware) | −159 | worst (08-25 −149, 08-22 −147). 5th-percentile block = −147 |
| avg %/fill | −1.29 % | 4th worst of 145. Worst of the 41 blocks with ≥ 5 fills |

**Per sleeve: how often a 12 h block is this bad or worse**

| sleeve | today | share of its blocks this bad or worse |
|---|---|---|
| FLIP | −0.49 % | 28 % (29 blocks) |
| MOM long | −0.63 % | 20 % (93 blocks) |
| MOM short | −0.67 % | 27 % (30 blocks) |
| FRENZY / WIDE | – | too young (4 blocks each) |

**Reading:** each piece is a 1-in-4 or 1-in-5 bad block. The record comes from all of them landing together, plus FRENZY's −3 % stops (−9 of the −12.9 summed %), plus 2× sizing.

**FRENZY + WIDE as a whole:** 11 fills ever, 3 wins.
- At the win rate the backtests imply (about 53–56 % for a ±3 % bracket with a small positive mean), 3 or fewer wins in 11 happens about **8 %** of the time.
- FRENZY_LONG alone: 1 win in 6, about **6 %**.
- That is unlucky but not yet proof. The pre-registered 40-fill review is the decision point.

### Gates now close to firing

| gate | bar | tally | status |
|---|---|---|---|
| **CALM3D cell verdict** (CURRENT_STATE line 148) | "N ≥ 5 fresh 2× fires since Sep-23: net-negative ⇒ 1.0× (4 fires so far ≈ −$565 — one more negative fire trips it)" | TAO and SAND are two more negative fires. As traded since 09-23: **14 · 57 % · −$416**. Today's-stack kept only: 9 · 67 % · +$153 | 🚨 **Fired by its own wording.** Conflict: line 14 and the B17 review say the watch was removed on 09-23. Operator to rule |
| **Mom-short PVR 0.86 kept side** | WR < 70 % on N ≥ 15 → `momentum_short_pair_vol_max` 1.0 | **11 · 36 % · −0.30 % · −$393** (verified on master stack_keep + B17), about 6 windows | 🚨 Certain to fire. 4/4 wins gives at most 53 % |
| FRENZY market-volume gate 194 | first 20 FRENZY + WIDE fills under it, avg < 0 → `frenzy_gvol_max` 0 | 10/20 · 3 W · −1.16 % | ⚠ trending. Needs > +1.16 % average on the next 10 |
| WIDE_CHOPPY_OBS | ≥ 15 would-block fills | 1 (FLUID, loser) | watch |
| Flip −DI < 15 observe | fresh N ≥ 15 / ≥ 8 windows | 3 · 1 W · −0.57 % (AAVE, BTW, NIL) | watch |
| STRONG_BEAR pADX < 21 exemption | net-negative at N ≥ 10 → clear | 5 · 80 % · +$161 (BTW the loser) | watch |
| FRENZY lock (205) / lev watch (207) | 20 fills | 4/20, Δ 0 · 2/20 | watch |

---

## 6. Recommendations, ranked

### Act now (a gate has fired or is certain to)

1. **CALM3D 2× → 1.0×, if the operator confirms the line-148 gate is live.**
   - Evidence: the locked multiplier-cell verdict (✗ HARMFUL = Total $ negative on N ≥ 5 fresh fires). It is met as traded: 14 fires, −$416. The gate's own text says one more negative fire trips it, and there have been two.
   - Against: on today's stack the kept fires are +$153 (9 fires; the B12 09-25 winners carry it). Since the 09-28 forward test, kept fires are 4 · 25 % · −$437.
   - Today's effect: +$177.
   - I lean to honouring the gate. It is the newer and more specific text, and pre-committed gates don't move.
2. **Mom-short PVR 0.86: prepare the revert to 1.0. It formally fires at kept fill 15.**
   - Evidence: a pre-committed gate, mathematically locked.
   - Caution: a losing kept side does not prove the blocked side (PVR 0.86–1.0) wins.
   - The bigger fact is that the **whole kept mom-short sleeve is −0.30 %/fill on 11 fills / about 6 windows**. That is a sleeve question.
   - Before anything beyond the gate's own action, run `scripts/sweep_separators.py MOM_SHORT` and the full sleeve-kill checklist. **No kill is proposed.**
3. **Flip NEGDI15 / TG_SHALLOW → 1×: already done (04:09 UTC).** Nothing more to do.

### Watch (no change; evidence below the locked gates)

4. **FRENZY / WIDE: no change.**
   - 11 fills is about 8 % bad luck under the backtest win rate. It is not evidence the edge is gone.
   - Keep the 40-fill review and gate 194 as pre-registered.
   - Note for the operator: if gate 194 fires, its pre-committed action (gvol max → 0) **removes** the market-volume filter. The year data says that adds worse (gvol ≥ 1.0) setups. Read the fill table carefully when it trips.
5. **New observe idea (1 batch, 6 fills, anecdote only): FRENZY entries after the episode has already broken down.**
   - All 4 FRENZY/WIDE losers since the lock exit (AIN #11, FLUID, ORCA, UMA) had a logged `FRENZY_SHORT_OBS` close below EMA50/200 earlier in the same episode.
   - The two WIDE winners (MOVR #7, RLC #8) had none before their entry.
   - Next step if wanted: pre-register it and test it on the year cohort with engine-parity bars. Not a ship candidate.
6. **WIDE_CHOPPY_OBS, flip −DI < 15, STRONG_BEAR pADX < 21:** keep counting. BTW was flagged by two observe lines and FLUID by one.
7. **Flip stop width: do nothing.**
   - BTW (+2.1 % after its stop) looks like a stop hunt, but QNT went −1.6 % after its stop.
   - 2 fills, and it needs the two-sided, live-stopped-only counterfactual.
8. **Sizing / ruin risk: a policy question for the operator, not an evidence-gated change.**
   - One 2× flip stop is about 11 % of equity.
   - A normal bad morning took 33 %.
   - The scaling-roadmap item ("ruin risk is the binding constraint") applies. Reviewing the ticket-to-equity ratio or the 2× flip/momentum cells' leverage is a risk choice, not a filter.
9. **Logging hygiene (operator approval needed):** stamp the crowd-sprint de-mux in `cell_multiplier_source`, and make the FRENZY KL check compare overlapping bars only.

---

## 7. Validation and blind spots

**Validation**
- `venv/bin/python scripts/validate_against_master.py` → **ALL CHECKS PASS** (M1 0 sign mismatches; F1/F2/C1/A1/Y1/X1/PS1/CB1 pass).
- The 12 h block tables read master `pnl` / `pnl_percentage` directly, non-probe only.
- The mom-short tally was re-derived from master `stack_keep` and matches the B17 review (11 · 36 %).
- The CALM3D list was re-derived from `cell_multiplier_source`.

**What I could not test**
1. Refused signals after 04:00 UTC. The journal ends at 04:00, so I can't see what the bot declined in the rally.
2. A continuous alt-breadth series. Breadth is only on the 10 fills, so the "alts weak vs BTC" idea can't be read in window units.
3. Funding per pair (only 3 fills stamped).
4. The yr5 replay was not used (memory: backtest trust). FRENZY's block-level distribution is too young (4 blocks).
5. 12 h blocks are fixed 00–12 / 12–24 UTC, not rolling.
6. B17 is not in the master. B17 fills were added from the export.
7. `above_streak` and the 30-min ATR change are not stamped.
