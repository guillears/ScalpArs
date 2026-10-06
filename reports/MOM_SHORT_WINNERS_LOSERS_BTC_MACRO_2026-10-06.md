# Momentum shorts: winners vs losers, the BTC "bull week" idea, and the pair-volume 0.86 gate (2026-10-06)

Read-only research. No code or config was changed.
Validation: `scripts/validate_against_master.py` reported **ALL CHECKS PASS**. My own checks:
- The rebuilt BTC 4h EMA50/200 gap matches the live stamp exactly on all 5 stamped master fills.
- The rebuilt BTC distance from its 30-day high matches the stamp with r = 0.96 (151 master fills) and r = 0.985 (13 momentum-short fills).
- For kept momentum shorts, `stack_pct` equals `pnl_percentage`, because no exit counterfactual re-prices this sleeve.
- The B17 tally reproduces the operator's "11 kept fills · 36 %".

**Units:** P&L is in % per trade at 1× size (leverage-invariant). "d" means distinct UTC days. Cell format is `N·days·WR·avg%`. Confidence intervals (CIs) are 95 % **day-block bootstraps**: whole days are resampled, never single trades.

## 0. Cohort

| Line | N · days · WR · avg % | Note |
|---|---|---|
| All full-size momentum shorts (master + B17) | 49·27d·61%·+0.028 | 14 BASE, 7 B1, 2 B3, 7 B4, 4 B5, 2 B8, 2 B12, 2 B13, 6 B14, 1 B15, 2 B17 |
| Kept under today's stack | 37·21d·73%·+0.173 | 10 blocked by PVR 0.86, 2 by the C1 STRONG_BEAR block |
| Kept since 09-18 (the revert-gate cohort) | 11·5d·36%·−0.303 | CI [−0.50, −0.03]. **56 % of the loss falls on one day, 09-29 (B14)** |
| Probes (separate line, not in any headline) | 43·9d·35%·−0.204 | all in the "not bull week" state |
| BASE fills the screen removed (re-added only for section 5) | 15 | 5 at PVR 0.86–1.0 and 10 at PVR ≥ 1.0; all from Jun-18 to Jul-8 |

- Breakeven WR for this sleeve is **58.7 %** on all full-size fills and 57.6 % on kept fills.
- Flips, fades, BEARRUN and SURGE are excluded.
- No momentum-short fill in BASE through B17 has PVR ≥ 1.0 apart from BASE rows before Jun-30, because the Jun-30 rule blocked them live after that.

## 1. Pre-registered H1: "momentum shorts lose in a BTC bull week"

H1 was written down at 12:53 UTC, before any outcome was joined to a BTC variable. Its definition:
- **Data:** closed Binance USD-M BTCUSDT bars as of each fill.
- **(a)** the 7-day return (from the last closed 1h close) is above 0.
- **(b)** the last closed daily close is above the daily EMA20.
- **(c)** the daily EMA20 is above the daily EMA50.
- **(d)** the 4h EMA50/200 gap is above 0.
- **(e)** the daily RSI14 is above 50.
- **(f)** BTC is within 5 % of its 30-day high.
- **BULL WEEK** = a ∧ b ∧ c.
- **Prediction:** the gap (bull avg − not-bull avg) is below 0.
- **Decision rule:** the locked expectancy bar, counted in day units.

### 1a. Split by sign (bull vs not)

| Variable | ALL bull | ALL not | gap | day-CI | P(gap<0) | KEPT bull | KEPT not | gap | P(gap<0) |
|---|---|---|---|---|---|---|---|---|---|
| (a) 7d return > 0 | 21·11d·67%·+0.073 | 28·16d·57%·−0.005 | **+0.078** | [−0.43,+0.40] | 0.40 | 14·8d·79%·+0.238 | 23·13d·70%·+0.134 | +0.105 | 0.38 |
| (b) close > daily EMA20 | 28·13d·57%·−0.059 | 21·14d·67%·+0.145 | −0.204 | [−0.65,+0.17] | 0.85 | 21·10d·62%·+0.007 | 16·11d·88%·+0.391 | −0.384 | 0.97 |
| (c) daily EMA20 > EMA50 | 27·11d·52%·−0.106 | 22·16d·73%·+0.193 | −0.299 | [−0.76,+0.06] | 0.95 | 18·7d·61%·+0.010 | 19·14d·84%·+0.328 | −0.318 | 0.90 |
| (d) 4h EMA50 > EMA200 | 33·14d·55%·−0.083 | 16·13d·75%·+0.257 | −0.340 | [−0.76,+0.05] | 0.96 | 22·9d·68%·+0.073 | 15·12d·80%·+0.321 | −0.248 | 0.85 |
| (e) daily RSI > 50 | 28·13d·57%·−0.059 | 21·14d·67%·+0.145 | −0.204 | [−0.65,+0.17] | 0.85 | 21·10d·62%·+0.007 | 16·11d·88%·+0.391 | −0.384 | 0.96 |
| (f) within 5 % of 30d high | 26·13d·54%·−0.088 | 23·15d·70%·+0.160 | −0.248 | [−0.72,+0.09] | 0.93 | 18·9d·61%·+0.011 | 19·13d·84%·+0.327 | −0.316 | 0.96 |
| **BULL WEEK (a∧b∧c)** | **18·8d·67%·+0.065** | **31·19d·58%·+0.007** | **+0.058** | [−0.60,+0.37] | 0.45 | **11·5d·82%·+0.270** | 26·16d·69%·+0.132 | **+0.138** | 0.33 |

**Day units.** When each day is collapsed to its mean:

| | Days | Mean of day-means | Days positive |
|---|---|---|---|
| BULL WEEK | 8 | −0.236 | 3 of 8 |
| Not bull week | 19 | +0.036 | 12 of 19 |

**The washed-out-window check is what decides this.** From Jun-18 to Jul-2, BTC was 15–24 % below its 30-day high, and that window holds 12 of the "not bull" fills at 83 % WR. Excluding it:

| Variable (ALL, ex-washed-out) | bull | not | gap | P(gap<0) | Day-means bull vs not |
|---|---|---|---|---|---|
| (a) | 20·10d·70%·+0.096 | 17·8d·35%·−0.317 | +0.414 | 0.05 | −0.11 vs −0.42 |
| (b)/(e) | 28·13d·57%·−0.059 | 9·5d·44%·−0.200 | +0.141 | 0.29 | −0.23 vs −0.31 |
| (c) | 27·11d·52%·−0.106 | 10·7d·60%·−0.059 | −0.047 | 0.59 | −0.32 vs −0.14 |
| (d) | 33·14d·55%·−0.083 | 4·4d·50%·−0.181 | +0.098 | 0.40 | −0.27 vs −0.18 |
| (f) | 26·13d·54%·−0.088 | 11·6d·55%·−0.106 | +0.018 | 0.50 | −0.32 vs −0.11 |
| **BULL WEEK** | 18·8d·67%·+0.065 | 19·10d·42%·−0.243 | **+0.308** | 0.14 | **−0.24 vs −0.26** |

Every trend-level leg (b–f) that looked negative on the full pool goes flat or flips sign once the washed-out window is removed. Those legs were measuring one thing: **shorts won in June's deep washed-out tape.** They were not measuring "bull weeks hurt shorts".

**Buckets** (ALL; detail in the scratch output):
- 7d return: ≤−3 % is +0.21; (−3, 0] is −0.30; (0, 3] has N=1; >3 % is +0.10. This is **non-monotone**, which points to a confound.
- Close vs EMA20: ≤−3 is +0.48; (−3, 0] is −0.22; (0, 3] is −0.32; >3 is +0.11. Also non-monotone.
- Daily RSI: ≤40 is +0.40; 40–50 is −0.20; 50–60 has N=3; >60 is −0.07.

**Do both states appear in several eras?** Only partly.
- BULL WEEK days fall into **3 episodes**: 08-24 to 08-26 (B3–B5, BTC +21–24 % over 7 days, 11 of 12 shorts won); 09-23 to 09-27 (B12–B13); 10-05 to 10-06 (B17).
- The strongest bull week in the data (late August, 7d +22 %) was the **best** stretch for momentum shorts. That contradicts H1 directly.
- Per era, bull vs not: B4 bull 86 % +0.33 · B5 bull 100 % +0.39 · B12 bull 0/2 · B13 bull 1/2 · B17 bull 1/2 · B14 not-bull 2/6 −0.35 · B15 not-bull 0/1 · B8 not-bull 0/2.

**yr5 replay** (refute-only; warm-up duplicates trimmed with `scripts/yr5_fills_trimmed.py`; 485 momentum-short fills, Jan–Sep, 3 seeds):
- BULL WEEK gap by seed is −0.158, −0.093 and −0.018. Pooled day-CI is [−0.32, +0.12], P(gap<0) = 0.80.
- Leg (c) is −0.13, −0.21 and −0.08, P = 0.92.
- So yr5 **does not refute the direction**. It cannot support a ship. Separately, yr5 has the whole momentum-short sleeve at −0.064 %/trade.

### 1b. H1 against the locked expectancy bar

| Leg | BULL WEEK, all full-size | BULL WEEK, kept |
|---|---|---|
| N ≥ 15 | 18 ✓ | 11 ✗ |
| ≥ 8 distinct days | 8 ✓ (3 episodes) | 5 ✗ |
| WR < breakeven 58.7 % | 67 % ✗ | 82 % ✗ |
| avg < 0 at 95 % (day bootstrap) | +0.065, P(<0) = 0.37 ✗ | +0.270, P(<0) = 0.11 ✗ |
| No day/pair ≥ 50 % of the loss | top day 19 % ✓ | DOT 09-27 = 52 % ✗ |

**Verdict on H1: REFUTED as pre-registered.** In fill units the effect has the wrong sign. In day units it disappears once the washed-out window is excluded, and the bull cohort fails every outcome leg of the bar. No observe candidate comes out of H1.

## 2. Winner-vs-loser sweep (all stamped entry_* columns plus the BTC macro variables)

**Coverage first.**
- 86 numeric entry_* columns are stamped on at least one of the 49 full-size fills.
- **50 were tested**: coverage ≥ 80 % and more than 3 distinct values. 8 rebuilt BTC macro columns are included: r7d, close/EMA20, EMA20/50, daily RSI, 4h gap, off-30d-high, r24, r72.
- Unscored fills are excluded, never counted as "rest".

**`scripts/sweep_separators.py MS`.** It runs only on SCREENED_BASELINE, i.e. the 14 BASE fills. Result: 16 tests, 7 one-dimensional consistent across its two eras, **0 consistent 2D cells**. This is useless at N=14 and is cited only because the checklist requires it.

**Own exhaustive scan on the master cohort.**
- One-dimensional: sign, median and both outer terciles of every tested column.
- Two-dimensional: every pair of median/sign legs × 4 quadrants, which is 6,844 tests.
- Survivor = N ≥ 8, ≥ 4 days, avg < 0, WR < breakeven, Δ vs rest ≤ −0.25.
- Out-of-sample (OOS) = the same direction in both time halves with N ≥ 3 each.
- The null is a shuffled-label null.

| Cohort | 1D survivors / null median (p95) | 1D OOS-confirmed / null | 2D survivors / null | 2D OOS-confirmed / null (P null ≥ real) |
|---|---|---|---|---|
| All full-size (49·27d) | 16 / 17 (31) | 6 / 10 (22) | 656 / 639 (920) | **217 / 344 (P = 0.95)** |
| Kept (37·21d) | 10 / 9 (21) | 1 / 2 (6) | 590 / 413 (746) | **49 / 92 (P = 0.87)** |

**Every count is at or below what luck produces.** There is no separator above chance in 1D or 2D, and that includes the BTC macro variables (137 of the 217 OOS-confirmed all-fill quadrants involve a BTC variable, which is also at null level).

The strongest single macro survivor is **daily EMA20/EMA50 > +4.5 %**: 15·8d·27%·−0.40 on all fills, 11·5d·36%·−0.30 kept. Every one of those fills sits in the second half, Sep-16 to Oct-6. It is **one regime episode**, and it is the recent slump itself. That makes it unproven, not a finding.

**Pattern flags (1D, binary):**
- C1: 8·50 % −0.11 all fills, but 5·80 % +0.25 kept.
- W4: 5·100 % +0.46.
- No flag is below breakeven at N ≥ 15.

**Per-pair concentration of losses:**
- All fills: HYPE 17 %, BCH 16 %, ARB 6 %, spread over 15 loser pairs.
- Kept: ARB 12 %, HYPE 12 %, DOGE 11 %, spread over 10 pairs.
- **No pair problem.** The kept-side loss is concentrated by **day** (09-29 = 41 %), not by pair.

## 3. Interaction with the existing short gates (current stack)

- The universe is already screened by the live momentum-short gates: BTC-ATR ≥ 0.12, RSI 25–50, pair ADX rising, deep-gap −1.0, weak-cap, the C1 STRONG_BEAR block, and PVR.
- No exit re-simulation was needed, because no momentum-short exit change is re-priced in the stack (`stack_pct == pct`).
- **C1 STRONG_BEAR block:** its 2 refusals (DASH 09-23, BCH 09-25) both fall in BULL WEEK. H1 would have caught them too. Both are already blocked.
- **PVR 0.86:** high PVR is **not** a bull-week proxy. 33 % of the 0.86–1.0 band is in bull week, against 30 % of the < 0.86 band. Correlation of PVR with the BTC trend variables is −0.2 to −0.4 (high-PVR shorts lean, if anything, toward weaker BTC).
- **BTC-ATR floor:** independent. No macro leg survives the sweep conditional on it.

## 4. Recent kept cohort and B17: counterfactuals

| Kept since 09-18 (11 fills) | Taken | Outcome |
|---|---|---|
| As traded / today's rule | 11 | 4W · avg −0.30 · Σ −3.33 pp |
| BULL WEEK block | blocks PHA +0.61, DOT −0.73, PEPE +0.25, FET −0.67 (Σ −0.54 pp; as traded −$83.5) | keeps 7 · 2W · **avg −0.40** (worse per trade) |

- The worst cluster, **09-29 (B14: 5 of 6 fills lost, 56 % of the loss), was NOT a bull week**: BTC's 7-day return was −3 %. ICP on 09-30 was not a bull week either.
- H1 removes 4 of the 11 fills, and the 4 it removes average better than the 7 it keeps.
- **B17:** BULL WEEK blocks both momentum shorts. PEPE was +0.25 % (+$39.79) and FET −0.67 % (−$107.31), so blocking both nets **+$67.5 as traded (+0.41 pp)**.
- FET alone is a correct call for H1. It is one fill on one day inside the third bull episode.

## 5. Pair-volume ceiling 0.86: removed cohort vs kept cohort, and head-to-head

**Cohort note.** The master holds **10** full-size MOM_SHORT_PAIRVOL rows, not the 13 quoted in the brief. Per the cohort-completeness rule I added back the **5 BASE fills at 0.86–1.0** that the v15 screen removed (`COMBINED_momentum_flip_…DEDUP.csv`, re-screened by `screen_pool.sleeve` with only the PVR check off). That gives **15**.

The universe for this section (U) is 62 fills: 47 master full-size fills (the 2 C1 refusals dropped, since every rule keeps that block) plus the 15 BASE add-backs. All 10 PVR ≥ 1.0 fills are BASE, Jun-19 to Jun-30, inside the washed-out window.

**MS_PVR_BLOCKED shadow:** its 4 priced reference refusals are not in any written report yet. `scripts/scout_revert_gates.py` is being edited by another agent. I did not run anything that writes state, so they are not included.

### 5a. Removed cohort vs kept cohort

| PVR band | N·days·WR·avg | day-CI of avg | P(avg<0) | Bull week | Not bull | Washed-out | Ex-washed-out |
|---|---|---|---|---|---|---|---|
| < 0.86 (kept) | 37·21d·73%·+0.173 | [−0.06,+0.40] | 0.07 | 11·5d·82%·+0.270 | 26·16d·69%·+0.132 | 12·9d·83%·+0.403 | 25·12d·68%·+0.063 |
| **0.86–1.0 (removed 09-18)** | **15·11d·40%·−0.219** | [−0.47,−0.00] | **0.976** | 5·3d·60%·−0.078 | 10·8d·30%·−0.289 | 3·2d·67%·+0.226 | **12·9d·33%·−0.330** (CI [−0.59,−0.10]) |
| ≥ 1.0 (Jun-30 rule) | 10·6d·40%·−0.222 | [−0.47,+0.10] | 0.92 | — | 10·6d·40%·−0.222 | all | — |

- Gap (0.86–1.0 minus < 0.86) = **−0.39, CI [−0.72, −0.08], P = 0.99**. Excluding the washed-out window it is −0.39, CI [−0.69, −0.11].
- By era, the 0.86–1.0 band reads: B1 0/3 · B3 0/1 · B8 0/2 · B4 2/3 (−0.02) · B5 1/1 · BASE 3/5.
- Concentration: top day 09-16 = 23 % of the loss; top pair HYPE = 26 %.
- **The 0.86–1.0 cohort passes every leg of the expectancy bar**: N 15, 11 days, WR 40 % below breakeven, P(avg<0) 0.976, no day or pair at 50 % or more.
- **Caveat:** all 15 fills predate the Sep-18 ship, so this is **in-sample**. After the 30–50 % haircut the gap is about −0.20 to −0.27 %/fill.

### 5b. 2×2 tables: PVR band × macro state

| | Not bull week | Bull week |
|---|---|---|
| PVR < 0.86 | 26·16d·69%·+0.132 | 11·5d·82%·+0.270 |
| PVR 0.86–1.0 | 10·8d·30%·−0.289 | 5·3d·60%·−0.078 |
| PVR ≥ 1.0 | 10·6d·40%·−0.222 | — |

| | Daily EMA20 ≤ EMA50 | Daily EMA20 > EMA50 |
|---|---|---|
| PVR < 0.86 | 19·14d·84%·+0.328 | 18·7d·61%·+0.010 |
| PVR 0.86–1.0 | 8·7d·38%·−0.206 | 7·4d·43%·−0.233 |

**PVR is not the macro effect in disguise.** High PVR loses in both macro states, and in the bull-week column it actually loses less.

### 5c. Head-to-head on U (1×, $ at the recent median 1× notional of $12,600; DD = max drawdown of the cumulative 1× $ path)

| Rule | Kept N·d·WR·avg | day-CI | $ at 1× | Max DD $ | 1st half | 2nd half | Ex-washed-out | Recent 11 taken |
|---|---|---|---|---|---|---|---|---|
| **(a) PVR < 0.86 (today)** | **37·21d·73%·+0.173** | [−0.05,+0.41] | **+808** | 496 | 22·68%·+0.07 | 15·80%·+0.32 | 25·68%·+0.06 | 11 (4W, −0.30) |
| (b) PVR < 1.0 (old) | 52·25d·63%·+0.060 | [−0.13,+0.23] | +395 | 576 | 31·58%·−0.04 | 21·71%·+0.21 | 37·57%·−0.06 | 11* (4W, −0.30) |
| (c) = (e) PVR < 1.0 ∧ not bull week | 36·19d·58%·+0.015 | [−0.18,+0.24] | +70 | 695 | 15·40%·−0.26 | 21·71%·+0.21 | 21·43%·−0.24 | 7 (2W, −0.40) |
| (d) PVR < 0.86 ∧ not bull week | 26·16d·69%·+0.132 | [−0.12,+0.44] | +434 | 421 | 11·55%·−0.12 | 15·80%·+0.32 | 14·57%·−0.10 | 7 (2W, −0.40) |
| context: no PVR rule | 62·26d·60%·+0.015 | [−0.15,+0.17] | +115 | 576 | — | — | 37·57%·−0.06 | 11 |
| context: bull-week block only | 46·20d·54%·−0.036 | [−0.21,+0.14] | −210 | 695 | — | — | 21·43%·−0.24 | 7 |

\* No 0.86–1.0 fill exists after 09-18, because the gate refused them live. Under rule (b) the recent count is therefore a lower bound; those refusals are what the scout shadow prices.

Notes on the head-to-head:
- Rules (c) and (e) are identical, because no bull-week fill has PVR ≥ 1.0.
- The 1st/2nd half split is at the median fill time.
- (d) sounds attractive, but its +0.13 average comes entirely from the washed-out BASE days: its first half is −0.12 and ex-washed-out it is −0.10.

## 6. Verdict

1. **H1 (pre-registered BULL WEEK = 7d > 0 ∧ close > daily EMA20 ∧ EMA20 > EMA50): refuted at this N.**
   - It has the wrong sign in fill units (+0.06 all fills, +0.14 kept).
   - In day units there is no difference once June's washed-out window is excluded (−0.24 vs −0.26).
   - The kept bull cohort is 82 % WR.
   - The best momentum-short stretch on record was the strongest BTC bull week (Aug 24–26).
   - yr5 leans the hypothesis's way (P 0.80) but cannot support a ship. There is no observe-first candidate.
   - **What would settle it:** ≥ 8 *new* bull-week days with kept fills, outside the three existing episodes, scored with the frozen a∧b∧c definition against the expectancy bar. Today there are 5 kept days. This can be tallied from klines at batch reviews with no code.
2. **No separator found.**
   - 1D and 2D survivor counts and OOS-confirmed counts are all at or below the shuffled null, including every BTC macro column.
   - The one strong-looking macro line (daily EMA20/50 > 4.5 %) is the Sep-16 to Oct-6 episode itself: one observation, unproven.
   - Losses are concentrated by day (09-29), not by pair.
3. **Do not replace PVR 0.86 with the macro filter, and do not add it on top.**
   - Replacing it, (c)/(e), cuts expectancy from +0.173 to +0.015 %/fill and $ from +808 to +70.
   - Adding it, (d), removes an 82 %-WR cohort and does worse on the recent 11 (−0.40 vs −0.30 per fill).
   - The 0.86 gate's own blocked cohort (15·11d·40 %·−0.22, P 0.976) passes the expectancy bar: in-sample, haircut to roughly −0.2 %/fill.
4. **About the 0.86 revert gate.**
   - Its pre-committed revert (kept-side WR < 70 % on N ≥ 15 → back to 1.0) will fire. The kept side is at 11·36 %, and even 4 straight wins would leave it at 8/15 = 53 %. Locked gates do not move, so by discipline the revert executes.
   - The evidence says reverting re-admits a cohort that is **worse** than what it keeps (−0.22 vs +0.17 historically). The kept side's slump is a day-clustered sleeve problem (09-29) that PVR did not cause.
   - Recommendation for the operator: when the gate fires, execute it as written, *or* explicitly declare a discipline override, citing this table and the MS_PVR_BLOCKED shadow's forward refusals as the arbiter. Do not quietly keep 0.86.
   - The real open question is the **sleeve** itself. Outside the June washed-out window it is about break-even (all fills 37·18d·54 %·−0.09; kept 25·12d·68 %·+0.06; yr5 −0.06). That is a watch item, not a kill proposal. Checklist items ① and ② were run here; items ③ (uniform degradation) and ④ (tape comparison) were not done formally.

## 7. Blind spots (what this could NOT test)

- **Low-coverage stamps were not in the sweep**:
  - entry_btc_eff72, off24lo, off24h, r72_pct, above72, off30d stamp: 27–31 %, stamped since about Sep-20.
  - entry_pair_age_days: 71 %.
  - pair 1h EMA20/200, BTC 1d return, ETH 5m return, BTC EMA50/100: 12 %.
  - signed 5/8/20 gaps: 6 %.
  - r24, r72 and off30 were rebuilt from klines and tested. **eff72 and the 24h-low distance were not rebuilt** (memory note: the 24h-low zone was already rebuilt on 09-29 with no candidate).
- **Binary pattern flags** got a 1D read only, not 2D.
- **BULL WEEK uses daily closes**: a 02:46 fill sees yesterday's daily bar. Intraday BTC strength enters only through the 5m and 1h stamps that were swept.
- **The bull state has 3 episodes (8 days)**, and the washed-out June window dominates the "not bull" state. Neither state is a broad multi-regime sample.
- **BASE fills use BASE-era exits**: no momentum-short exit re-pricing exists in the stack. The 15 add-back BASE fills come from the raw COMBINED pool at their recorded %.
- **The MS_PVR_BLOCKED forward refusals** (4 since 09-28) are not included.
- **$ figures assume a fixed $12,600 1× notional.** Paper fills, no market impact.
- **The 0.86–1.0 evidence is in-sample** for the Sep-18 ship. Haircut applied in words only.

Scratch artefacts (pre-registration text, scripts, sweep CSVs): `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/`.
