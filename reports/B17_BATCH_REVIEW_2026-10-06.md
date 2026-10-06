# B17 batch review: what today's decisions and candidates change (2026-10-06)

This is read-only analysis. No code, config, scout state or master file was changed, and nothing was committed.

**Source:** `~/Downloads/scalpars_orders_paper_2026-10-06_12-17-50.csv`, 24 rows.

**B17 window:** ids 1–24. The first fill opened 2026-10-03 23:25 UTC. The last fill closed 2026-10-06 12:00 UTC. B16 (`reports/BASELINE16_batch1002-1003_orders.csv`) ends 10-03 15:37 UTC, so the reset falls in that gap. The previous export (02:34) held ids 1–16. This export adds ids 17–24.

**What the batch contains:**
- 24 closed bot fills.
- 0 MANUAL fills and 0 `*_PROBE` fills, so there is no separate probe line.
- **0 open positions** (every row is CLOSED).

**Units:**
- P&L % is the price move. It does not depend on leverage. Rule: compare % across batches, never $.
- Cell multipliers scale the **investment**, so going from 2× to 1× halves the $ and leaves the % unchanged.

## 1. Batch trades

| id | pair | sleeve | dir | opened UTC | closed UTC | P&L % | P&L $ | close reason | size / cell |
|---|---|---|---|---|---|---|---|---|---|
| 1 | AINUSDT | FRENZY_WIDE | LONG | 10-03 23:25 | 10-03 23:28 | −3.01 | −84.99 | STOP_LOSS | 1× inv, lev 4× (0.2) |
| 2 | SANDUSDT | FRENZY_LONG | LONG | 10-04 05:05 | 10-04 07:20 | −3.00 | −123.82 | STOP_LOSS | 1× inv, lev 6× (0.32) |
| 3 | AINUSDT | FRENZY_LONG | LONG | 10-04 11:15 | 10-04 12:00 | +3.47 | +136.72 | RUNNER_TRAIL | 1×, lev 6× |
| 4 | SANDUSDT | FRENZY_LONG | LONG | 10-04 14:05 | 10-04 14:48 | −3.01 | −124.73 | STOP_LOSS | 1×, lev 6× |
| 5 | ZRXUSDT | SPIKE_FADE | SHORT | 10-05 00:14 | 10-05 00:25 | +0.30 | +30.05 | RUNNER_TRAIL | 2× SPIKE_FADE |
| 6 | ORCAUSDT | MOM-long | LONG | 10-05 07:22 | 10-05 07:40 | +0.95 | +127.99 | HARD_TP_LADDER L1 | 1.0× UNMATCHED (as stamped) |
| 7 | MOVRUSDT | FRENZY_WIDE | LONG | 10-05 09:15 | 10-05 09:19 | +3.00 | +84.80 | FRENZY_TP | 1×, lev 4× |
| 8 | RLCUSDT | FRENZY_WIDE | LONG | 10-05 12:00 | 10-05 12:03 | +3.01 | +87.43 | FRENZY_TP | 1×, lev 4× |
| 9 | NILUSDT | MOM-long | LONG | 10-05 13:57 | 10-05 14:04 | +1.59 | +159.03 | HARD_TP_LADDER L3 | 1.0× UNMATCHED (as stamped) |
| 10 | 1000PEPEUSDT | MOM-short | SHORT | 10-05 15:32 | 10-05 15:33 | +0.25 | +39.79 | RUNNER_TRAIL | 1× |
| 11 | AINUSDT | FRENZY_WIDE | LONG | 10-05 16:15 | 10-05 16:46 | −3.02 | −96.85 | STOP_LOSS | 1×, lev 4× |
| 12 | LITUSDT | MOM-long | LONG | 10-05 18:22 | 10-05 19:03 | +2.54 | +592.57 | RUNNER_TRAIL L1 | 1.5× UNMATCHED |
| 13 | FETUSDT | MOM-long | LONG | 10-05 18:45 | 10-05 19:29 | +0.10 | +22.19 | RUNNER_TRAIL | 1.5× UNMATCHED |
| 14 | VTHOUSDT | SPIKE_FADE | SHORT | 10-05 23:51 | 10-05 23:52 | +0.36 | +89.41 | RUNNER_TRAIL | 2× SPIKE_FADE |
| 15 | PUMPUSDT | FLIP | SHORT | 10-06 02:06 | 10-06 02:10 | +0.12 | +46.93 | FLIP_RUNNER_TRAIL | **2×** [TG_SHALLOW]+[NEGDI15] |
| 16 | QNTUSDT | FLIP | SHORT | 10-06 02:11 | 10-06 02:17 | −1.15 | −438.44 | FLIP_STOP_LOSS L1 | **2×** [NEGDI15] |
| 17 | FLUIDUSDT | FRENZY_WIDE | LONG | 10-06 02:30 | 10-06 02:41 | −3.04 | −100.88 | STOP_LOSS | 1×, lev 4× |
| 18 | FETUSDT | MOM-short | SHORT | 10-06 02:46 | 10-06 03:02 | −0.67 | −107.31 | EMA13_CROSS_EXIT L1 | 1× |
| 19 | BTWUSDT | FLIP | SHORT | 10-06 06:22 | 10-06 06:25 | −1.19 | −92.72 | FLIP_STOP_LOSS L1 | 1× (lev 10×) |
| 20 | NILUSDT | FLIP | SHORT | 10-06 06:29 | 10-06 06:40 | +0.26 | +25.83 | FLIP_RUNNER_TRAIL | 1× |
| 21 | ORCAUSDT | FRENZY_LONG | LONG | 10-06 09:40 | 10-06 10:04 | −3.00 | −137.48 | STOP_LOSS | 1×, lev 6× |
| 22 | UMAUSDT | FRENZY_LONG | LONG | 10-06 10:10 | 10-06 12:00 | −3.00 | −131.60 | STOP_LOSS | 1×, lev 6× |
| 23 | TAOUSDT | MOM-long | LONG | 10-06 10:44 | 10-06 11:10 | −0.69 | −200.66 | STOP_LOSS L1 | 2× NONEXP_CALM3D |
| 24 | SANDUSDT | MOM-long | LONG | 10-06 11:37 | 10-06 11:50 | −0.56 | −152.90 | RH_PREMISE_EXIT (recovery hold) | 2× NONEXP_CALM3D |

### Totals per sleeve (as traded)

| sleeve | N | WR | avg % | $ |
|---|---|---|---|---|
| MOM-long | 6 | 67 % | +0.65 | +548.22 |
| MOM-short | 2 | 50 % | −0.21 | −67.52 |
| FLIP (FAN short) | 4 | 50 % | −0.49 | −458.40 |
| SPIKE_FADE | 2 | 100 % | +0.33 | +119.46 |
| FRENZY_LONG | 5 | 20 % | −1.71 | −380.91 |
| FRENZY_WIDE | 5 | 40 % | −0.61 | −110.48 |
| **TOTAL** | **24** | **50 %** | **−0.39** | **−349.64** |

FRENZY and WIDE together: 10 fills · 30 % WR · −1.16 % · −$491. Eight of the ten fills hit the −3 % stop.

## 2. Impact of today's decisions and candidates (before → after)

| # | change | status | fills touched | $ before → after | sleeve avg % before → after | batch total |
|---|---|---|---|---|---|---|
| a | FLIP NEGDI15 / TG_SHALLOW 2× → 1× | **LIVE** (a51d874, pushed 10-06 04:08 UTC, deploy ≈ 04:18) | QNT, PUMP (both opened before the deploy) | FLIP −458.40 → **−262.64** (QNT −438.44 → −219.22; PUMP +46.93 → +23.46) | −0.49 → −0.49 (size only) | −349.64 → **−153.88** (+195.76) |
| b1 | WIDE refuses ATR_HIGH (5m ATR > 2.5 %) | candidate | AIN #1 (ATR 6.63, red bar), AIN #11 (ATR 4.34, red bar) blocked. Both are "ATR-only" refusals (red candle) | WIDE −110.48 → **+71.36** (N 5 → 3) | −0.61 → **+0.99** | −349.64 → **−167.80** (+181.84) |
| b2 | b1 plus WIDE lev 0.05 instead of 0.2 on the kept fills | candidate | MOVR, RLC, FLUID at ¼ size | WIDE −110.48 → **+17.84** | % unchanged (+0.99) | −349.64 → −221.32 (+128.32) |
| b3 | for reference: WIDE at 0.05 with no ATR filter | – | all 5 WIDE fills | −110.48 → −27.62 | −0.61 | +82.86 |
| a+b1 | | | | | | **−349.64 → +27.96** |
| a+b2 | | | | | | −349.64 → −25.56 |

### (a) Flip cells

- Only QNT (NEGDI15, −DI 16.0) and PUMP (TG_SHALLOW: BTC trend gap −0.092, plus NEGDI15: −DI 17.5) ran at 2×. These are the two fills that fired the NEGDI15 revert gate.
- The two flips after the deploy would have been 1× under either config:
  - BTW: −DI 14.8, gap −0.165
  - NIL: −DI 13.1, gap −0.181
- So the change touches this batch only retroactively. The FLIP sleeve is still −0.49 %/fill at any size. Two of four fills were stopped at −1.15 / −1.19.

### (b) WIDE ATR_HIGH candidate (reports/FRENZY_GREEN_AND_WIDE_ATR_FORMAL_2026-10-06.md)

- **Batch result:** the two blocked fills are both −3 % stops, and the three kept fills are the two +3 % winners plus FLUID's −3 % stop. The batch agrees with the study's direction (year: blocked −0.53 %/fill, kept +0.01).
- **Read with care:**
  - This is 5 fills, and both blocked fills are the same pair (AIN) two days apart.
  - The kept winners MOVR (ATR 2.45) and RLC (ATR 2.38) sit just under the 2.5 cut.
  - This is an anecdote, not evidence. The candidate rests on the 361-fill year cohort.
- **Lev 0.05:** this shrinks the kept book to +$17.84. The study's year book says green-only WIDE at 0.05 is ≈ flat with lower drawdown. At 0.2 it adds drawdown and no return.
- **Second candidate in the same batch:** the WIDE choppy candidate (frozen 67.8, observe-only) would also have refused FLUID (above_share 44.4 %, −$100.88). Its first observed would-block fill is a loser. (Only FLUID carries the stamp. It went live 10-05 23:58 UTC.)

### (c) FRENZY_LONG green-candle V2 (observe-only, no live change)

- All 5 FRENZY_LONG fills opened on red or flat bars (bar_ret −0.04 to −1.35 %), so the skip worked as built.
- The batch's visible GREEN_BAR refusals are the 3 WIDE fills that took them:
  - MOVR: +1.31 % body
  - RLC: +0.52 %
  - FLUID: +0.05 %
- **None of them has ATR ≤ 1.5 %** (1.90 / 2.38 / 2.45), so the pre-registered **V2 ∧ ATR ≤ 1.5 cell had 0 fires**.
- Whether they qualify for V2 (above_streak > 12) can't be read: `above_streak` is not stamped. RLC is the "hold bar" case the study cites.
- **Illustration only:** as FRENZY_LONG at lev 0.32 (6× vs WIDE's 4×), these three would have been ≈ +127 / +131 / −151 = **+$107**, against the +$71 they made in WIDE.
- The scout logged one more FRENZY_READY refusal, but for market volume, not green: API3, 10-06 06:35, gvol 1.12×. Replay: +3.88 % lock trail.

### Context: yesterday's (10-05) sizing changes on this batch

- **UNMATCHED ML 2× → 1.5× (206):** LIT and FET were at 1.5×. At 2× they would have made +$205 more (LIT +$592.57 → +$790.09).
- **CALM3D stays 2×:** TAO and SAND lost −$353.56 together (at 1×: −$176.78). CALM3D was confirmed keep at 2× on 09-23 and its watch was removed, so there is no live gate. This is a note only.
- **Lock exit (205):** the 4 FRENZY/WIDE fills after the lock (AIN #11, FLUID, ORCA, UMA) all stopped at −3 without ever reaching +3 (peaks +1.88 / +0.23 / +0.56 / +1.87). The lock and the fixed +3/−3 give the same result on all four, so Δ = 0.

## 3. Watchlist check (open gates in CURRENT_STATE + scout trackers)

**Scout staleness:** `SCOUT_REPORT_latest.md` / `SCOUT_REVERT_GATES.json` ran at 11:18 UTC, but they read orders only up to the **10-06 02:34 export** and journals only up to 10-06 04:00. Fills **17–24 are missing from every scout tally**. The tallies below add those fills by hand from this export. Journal-based gates (refused-signal re-pricing) cannot be updated without re-running the scout, which I did not do because `scout_frenzy_exits.py` is being edited.

### Gates that FIRED or are trending to fire

| gate | bar | tally now (master + B17) | flag |
|---|---|---|---|
| **Mom-short PVR < 0.86 kept side** (09-18 ship, revert `momentum_short_pair_vol_max` → 1.0) | WR < 70 % on N ≥ 15 fresh mom-short fills | Stack-kept since 09-18: B12–B15 9 fills (3 W, −$325) + B17 PEPE (+$39.79) and FET (−$107.31) = **11 · 36 % WR**. As traded (adding the 2 C1-regime refusals) = 13 · 31 %. | 🚨 **Locked to fire.** Even 4/4 wins on the next fills gives at most 8/15 = 53 % < 70 %. The gate becomes due at fill 15. Check that it wasn't already adjudicated off-file. The gate re-admits higher-PVR shorts. A losing kept side doesn't prove the blocked side wins, but the gate is pre-committed. |
| **FRENZY market-volume gate (194)** | first 20 FRENZY + WIDE fills under the gate, avg < 0 → `frenzy_gvol_max` 0 | scout 7/20 (−0.37 %) → **10/20 · 3 W · avg −1.16 %** (adds FLUID, ORCA, UMA) | ⚠ Trending to fire. The last 10 would need to average > +1.16 % to avoid it. |
| NEGDI15 (DL 220) | its revert gate | fired on QNT/PUMP and is **already acted on** (→ 1×) | done |

### Gates that gained fills (no fire)

| gate | bar | B17 contribution → tally |
|---|---|---|
| FRENZY lock exit (205) | first 20 FRENZY + WIDE fills: fixed beats lock → revert | 1 → **4/20** (AIN #11, FLUID, ORCA, UMA). Lock = fixed = −3.00 on all four (Δ 0). |
| FRENZY leverage watch (207/216/217) | first 20 FRENZY_LONG lock fills: mean ≥ +0.30 ∧ stop rate ≤ 45 % → review 0.5 | **2/20** (ORCA, UMA): mean −3.00, stop rate 100 % |
| FRENZY strong lev 0.5 (197, deploy ≈ 10-04 14:16) | first 10 sized-up fills | **0/10**. ORCA and UMA had ADX Δ < 0, so they were correctly sized 0.32. SAND #4 (strong signature) opened 14:05, before the deploy. |
| FRENZY sleeve review (40 fills) / WIDE 40-fill gate | keep: mean > 0 ∧ ≥ 12 W / WIDE mean > 0 | FRENZY_LONG all-time 6 (incl. B16 ENJ) · 1 W · −1.76 %. WIDE 5 · 2 W · −0.61 %. Both well short of 40. |
| WIDE_CHOPPY_OBS (215) | ≥ 15 would-block fills | **1** (FLUID 44.4 % ≤ 67.8, −3.04 %), 0 kept |
| WIDE_BTC_SOFT | ≥ 40 fills/state | FLUID is borderline: live BTC RSI 44.6, but closed-bar RSI(14) 49.1. The tracker uses RSI(12) on closed bars, so it needs the scout. Probably not soft. |
| ATR_FAST_LOCK3 | ≥ 30 fast fills | FLUID, ORCA and UMA never reached +3, so LOCK3 − LOCK2 = 0 whatever their ATR Δ30m |
| UNMATCHED 1.5× (206) | next 15: WR ≥ 70 ∧ $ > 0 → 2× · $ < 0 → 1× | **2/15** (LIT, FET) · 100 % · +$614.76 |
| Recovery hold kill bar (172) | 3 hard stops in a row, or first 10 holds Σ(final − trigger) < 0 | **1st hold ever:** SAND #24, trigger −0.70 → exit −0.56 (**+0.14 pts better**), no hard stop. 1/10. |
| FAN −DI < 15 observe (120) | fresh N ≥ 15 / ≥ 8 windows, expectancy bar | fresh since 09-27: AAVE + **BTW (−1.19)** + **NIL (+0.26)** = 3 · 1 W · avg −0.57 %. Companion ∧ bear ≥ 76: BTW 88.6, NIL 79.5, both in. |
| STRONG_BEAR pADX < 21 exemption | net-negative at N ≥ 10 → clear | **BTW** (STRONG_BEAR, pair ADX 19.4) loser −$92.72 → about 5 · 80 % · +$161 (5/10) |
| NEGDI15 re-arm (Pattern-W on a fresh 1× cohort) | N ≥ 30 … | 0 fresh fires (both post-ship flips have −DI < 15) |
| 41h fade HEALTHY_BEAR zone (observe) | | **VTHO** in the zone (HEALTHY_BEAR, BTC gap +0.04), a **winner** +0.36 % |
| Fade BTC-RSI [45,50) re-revert | N ≥ 10: WR < 55 ∨ Σ < 0 → 45 | ZRX is borderline (live 44.7 / closed 50.95), but a winner, so it can't push toward revert |
| Fade cap 0.5 % ($10–20M band) | | 0 (VTHO $5.2M, ZRX $2.7M) |

### Gates with no B17 fills

These had no B17 fills: heat original rule / heat revert check (no ML long with bull ≥ 80), ML_B1H_NEGFLANK, LONG_CHOP_BURST, SURGE_LONG, BEARRUN, and REBOUND.
- **Heat:** no ML long had bull ≥ 80.
- **ML_B1H_NEGFLANK:** the judged cohort is 0. TAO and SAND (BTC 1h slope +0.12 / +0.14) go to "rest", 2 losers. LIT and FET were on the negative flank but opened before the 10-06 floor.
- **LONG_CHOP_BURST:** NIL #9 had eff72 0.006 ≤ 0.007, but no prior fill within 120 s, so correctly admitted (+$159).
- **SURGE_LONG:** 0 triggers filled.
- **BEARRUN:** 0 windows.
- **REBOUND:** off (BTC −1.3 to −2.6 % below its 30-day high).

### Journal-only gates (scout values as of 10-06 04:00 journal, stale)

| gate | scout value |
|---|---|
| Chop∧burst | 0/6 |
| WIDE_CHOPPY revert | dormant |
| LOADX | resolved, keep |
| FLIP_EMA13_BLOCKED | 0/10 |
| FLIP_PADX_BLOCKED | 1/10 provisional, +0.161 |
| Heat re-scope | frozen |
| Mega-cap | 3/8 |
| Crash-short | 5/30, −1.48 % |

## 4. Validation

`venv/bin/python scripts/validate_against_master.py` → **ALL CHECKS PASS** (M1 0/270 sign mismatches; M2 270 ledger fills in the pool, 384 kept; F1/F2/C1/A1/Y1/X1/PS1/CB1 pass).

**B17 is NOT in the master.** `MASTER_POOL_stacked.csv` ends at B16 (last open 10-03 15:36), and no `reports/BASELINE17_*.csv` exists. It was not archived here. Every tally above that cites "master" covers B1–B16 plus this export added by hand.

## Blind spots

1. Journal-based gates (refusal re-pricing) were not refreshed for fills 17–24.
2. `above_streak` (V2) and the 30-min ATR change (ATR_FAST) are not stamped in the CSV.
3. The WIDE_BTC_SOFT RSI(12) is not stamped.
4. ORCA #6 and NIL #9 are stamped UNMATCHED at 1.0×, not the then-live 2×. I didn't check why (possibly a size cap). They count as UNMATCHED only before the 206 floor.
5. The mom-short PVR tally uses master stack_keep since 09-18 12:00 UTC. If the gate's "fresh" counter starts later, the N changes. The direction (WR far below 70 %) does not.
