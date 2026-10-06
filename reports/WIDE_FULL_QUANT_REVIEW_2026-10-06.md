# FRENZY_WIDE: full quant review (sleeve-kill checklist, ride value, decision table), 2026-10-06

Research only. No bot code, config, test or template was touched, and nothing was committed. The results are **unreviewed**: no caveman or deep review has run on them. Scripts and outputs are in the scratchpad `…/scratchpad/wfq/`:
- `feat_all.py`, `post.py`: features
- `checklist_trade.py`, `p1x.py`: PART 1
- `p2a.py`–`p2d.py`: PART 2
- `p3.py`: PART 3
- `*.txt`, `ck_out.md`: outputs

---

## 0. Engine parity and cohort (read first)

| item | status |
|---|---|
| Signal bars | `reports/FRENZY_ENGINE_COHORT_2026-10-05.csv` comes from the real `services.frenzy` path (`frenzy_walk` → `frenzy_long_status` → `frenzy_wide_ready`). All 8 live FRENZY/WIDE fills from Oct 3–5 sit on their exact signal bar (overnight review). |
| Not the bugged "moments" cohort | Confirmed. The Oct-5 sleeve checklist (`FRENZY_WIDE_SLEEVE_CHECKLIST_2026-10-05.md`) already ran on the engine cohort. **But it ran on cohort A (published, 740 fills), which includes 159 fills on pairs the bot cannot trade** (Alpha / < 90 days / non-ASCII). This review re-runs the whole checklist on the corrected cohort. |
| Cohort used here | **TRADE**: live-tradeable pairs, live-like market-volume gate (`gvol_live_U2 < 1`), and the live sequencing (FRENZY + WIDE together, 2 slots each, pair flat across sleeves, ≤ 3 per pair per day). That gives **565 WIDE fills on 227 days and 204 LONG fills**, Jan 10 → Sep 27. |
| Exit | LOCK2 = the live lock (−3 stop, arms at +3, floor +2, trail 2). 12 s tick entry, 0.10 slippage, 0.09 fees. |
| Baseline books | Reproduced exactly. **WIDE off** (LONG re-sequenced alone): 205 fills, **$9,741, max DD −43 %**. **As-is** (WIDE at 0.2): **$1,188, max DD −94 %**. |
| Signal-bar features | Re-checked on the 5 live WIDE fills from Binance klines. The signal-candle returns match the fills (MOVR +1.31, RLC +0.52, FLUID +0.05, both AINs red). |
| Runner legs | Re-used from `split/run_*of8.pkl`, the reviewed 12 s pricer. Two-sided sanity check: on the 295 fills the lock stopped at −3, the runner Δ is 0 on 98 % of them, as it should be. |

---

## 1. Plain-language answer

1. **The checklist was run in full (①–④) on the tradeable cohort.** It does **not** give a clean "kill".
   - ① finds a real entry separator.
   - ③ finds the "everything degrades together" fingerprint that the checklist says means *keep hunting*.
   - ② finds no measured market variable that explains it.
2. **The entry finding is new: signal-bar volume (`qv_rel`).** `qv_rel` is the signal bar's quote volume divided by the average of the previous 12 bars.
   - WIDE fills on a bar with **volume above the recent average** averaged **+0.21 %/fill**. Fills on **quiet bars** averaged **−0.87 %/fill**.
   - The gap is +1.08, with a day-block CI of [+0.59, +1.57], and it is **positive in 9 of 9 months**.
   - It is the strongest of 31 dimensions, with a family-wise p of 0.001.
   - **But it does not replicate where it should:**
     - it is absent on FRENZY_LONG;
     - it reverses on WIDE signals from the untradeable pairs;
     - it is weak on the WIDE signals the gvol gate refused.
   - **And it would have blocked RLC** (RLC's signal bar was quiet: `qv_rel` 0.45).
3. **The operator's ride point is right about the size of the prize. The bot cannot yet catch it.**
   - **21 % of WIDE fills reach +50 % within 72 h** (≈ 113 episodes a year).
   - **84 % of those first trade down through the −3 % stop.**
   - Only **22 rides (≈ 31 a year) reach +50 % without first touching −3 %**. RLC was one of them.
   - **On those 22:**
     - the perfect exit averages +124 %;
     - the live lock banks +2.9 %;
     - the best realistic runner (EMA200, +2 floor) banks +14.6 % on average, but its median is +1.9.
4. **Post-entry information separates rides from non-rides.** Up-move by +60 min has an AUC of 0.78, holding in both halves and far beyond the null.
   - **No hold-vs-cut rule built on it beats chance in P&L** (all selection-adjusted p ≥ 0.21; out-of-sample ≈ 0).
   - The limiting factor is the exit: every runner rule tested gets shaken out on the dip that most rides take first.
5. **On the year books, only post-hoc subsets beat WIDE-off. All of them carry more drawdown.**

   | subset (lev 0.2) | year book | max DD | catches RLC? |
   |---|---|---|---|
   | WIDE off (baseline) | $9,741 | −43 % | no |
   | GREEN ∧ `qv_rel` > 1 | $16.2k | −53 % | no |
   | HOLD_GREEN | $13.5k | −54 % | yes |
   | `qv_rel` > 1 | $12.7k | −68 % | no |

   - None has passed a forward test.
   - The as-is sleeve ($1,188, −94 %) and green-only ($8,082, −66 %) do not beat off.

**What the data supports:** WIDE as-is has no edge at any size. There is one in-sample entry separator with a mechanism (`qv_rel`) and one frozen post-hoc pocket (HOLD_GREEN); neither replicates yet. There is a large, real ride tail that no exit tested so far captures. The cheapest test that answers the open question is a **zero-capital shadow** (§6).

---

## 2. PART 1: the sleeve-kill checklist on the tradeable cohort

| item | run? | method | result | passes the "kill" bar? |
|---|---|---|---|---|
| ① entry dimensions | yes | 36 entry dims, each at sign / median / terciles / quintiles (346 buckets); exhaustive 2D over every pair, median 2×2 + tercile 3×3 (7,620 cells); null = outcome permuted within day, 300 rounds; walk-forward on the halves | **2D family beats the null:** 219 cells with \|t\| ≥ 3 vs a null median of 46 (95th 114), p < 0.003. Positive pockets: 68 vs null 95th 47. Max \|t\| family p = 0.050. **Top survivor: `qv_rel`** (§2a). Walk-forward: Jan–Apr picks fail on May–Sep (0 of 5 top cells positive); May–Sep picks hold on Jan–Apr (4 of 5). | **Not passed.** A separator exists, so "no rescue" is not true on this cohort. |
| ② macro / regime (day units) | yes | 26 BTC / breadth / froth variables: sign or binary first, then terciles / quintiles; each day = 1 observation; null = day tags permuted within month | Sign level: worst states are 4h golden cross (−0.49), breadth bull > bear (−0.57), BTC 7d > 0 (−0.45), BTC ADX not rising (−0.46). **None survives the family null** (best: EFF72 chop gap −0.91, p_fw 0.23). Keep-states beating the null: 0 (P = 1.00). | Passed (no regime switch found). |
| ③ uniform degradation | yes | monthly means of 11 WIDE sub-cohorts + FRENZY | **Degradation is uniform:** 91 % of sub-cohorts negative in Apr, May, Jun and Sep. Median Spearman across sub-cohorts +0.70. WIDE vs FRENZY −0.38, so the factor is **WIDE-specific, not market-wide**. Structural layer flat across halves: reclaim, ATR_HIGH, engine-only. The rest (hold, GREEN, high `qv_rel`) falls by 0.35–0.45 after April. | **Not passed.** This is the checklist's "unmeasured common variable — keep hunting" fingerprint, and ② did not find it. |
| ④ tape context | yes | per month: BTC return, off-30d-high, breadth, BTC ATR, froth, pair ATR, gvol | Good months (Jan–Mar, Aug) vs bad: BTC deeper below its 30d high (−11.4 vs −8.1 %), higher BTC 5m ATR (0.186 vs 0.144), lower pair ATR (2.5–3.0 vs 2.9–3.6). Over 9 months, WIDE monthly mean vs breadth ρ +0.53, BTC ATR +0.52, gvol +0.48. 9 points is descriptive only. | Descriptive only. |

**Checklist verdict:**
- All four items were run.
- **Two of them (① and ③) argue against amputation as the next step.** ① has a survivor. ③ has an unexplained WIDE-specific factor.
- **But neither turns WIDE-as-is positive.** As-is it is −0.33 %/fill, CI [−0.58, −0.08], and negative in Jan–Apr too (−0.22).

### 2a. The survivor: `qv_rel` (signal-bar volume vs the prior hour)

**WIDE quintiles** (Jan–Apr / May–Sep in brackets):

| quintile | qv_rel range | mean | Jan–Apr / May–Sep |
|---|---|---|---|
| Q1 | 0.19–0.62 | −0.90 | −0.91 / −0.90 |
| Q2 | 0.62–0.89 | −0.99 | −0.49 / −1.27 |
| Q3 | 0.90–1.31 | −0.24 | −0.05 / −0.44 |
| Q4 | 1.32–2.06 | **+0.38** | +0.20 / +0.57 |
| Q5 | > 2.06 | +0.10 | −0.04 / +0.22 |

**Median split (1.11): kept +0.210 [−0.15, +0.59] / blocked −0.870 [−1.20, −0.52].**

| test | result |
|---|---|
| gap, kept − blocked | **+1.08**, day-block CI [+0.59, +1.57] |
| within-day / within-month permutation | p < 0.001 / < 0.001 |
| within-day demeaned gap (111 days with both sides) | +0.91 |
| family-wise among 31 median splits (max-gap null) | **p 0.001** (null 95th 0.95) |
| gap by month | +0.68 · +0.58 · +1.42 · +0.39 · +3.01 · +1.31 · +2.13 · +0.57 · +0.76 → **9 of 9 positive** |
| blocked cohort vs the expectancy bar | WR 39 % vs breakeven 53 % ✓ · day CI high −0.52 ✓ · 173 days ✓ · worst pair 6 %, worst day 4 % of the loss ✓ · N 283 ✓. *(Levels are weak evidence on a negative sleeve; the gap statistic above is what counts.)* |
| kept cohort | 6 of 9 months positive (Apr −0.71, Jun −0.26). −top5 +0.07, −top10 −0.04. Top pair SAHARA = 28 % of the net. |
| fixed natural cut **`qv_rel` > 1.0** | kept 307 · +0.179 [−0.18, +0.56] / blocked 258 · −0.945. Halves: kept +0.08 / +0.30, blocked −0.66 / −1.14. |
| **replication: FRENZY_LONG** (same detector, red candles) | **none.** > 1.11: +0.29 vs ≤: +0.42 |
| **replication: untradeable WIDE signals** (266, Alpha / young pairs) | **reversed.** > 1.0: −0.20 vs ≤: −0.07 |
| **replication: gvol-refused tradeable WIDE signals** (418) | weak. > 1.0: −0.15 vs ≤: −0.46 (gap +0.31; +0.07 at 1.11) |
| replication: published cohort A | gap +0.70 (A contains this cohort, so it is not independent) |
| live WIDE fills (Oct 3–6) | `qv_rel` > 1: MOVR +3.00. ≤ 1: AIN −3.01, **RLC +3.01 (qv 0.45)**, AIN −3.02, FLUID −3.04. N = 5, no weight. |
| ride rate | pre-stop peak ≥ +20: 15 % if > 1.11 vs 8 % if ≤ 1.11 |

**Reading:**
- The mechanism is plausible. A WIDE fill is a chase on a green or high-ATR bar. If the push bar comes on rising volume, the move has fuel. If it comes on a quiet bar, the state turned on by the clock and nobody is buying.
- But **the same variable does nothing on LONG and reverses on the untradeable WIDE signals.** That is what a well-chosen in-sample artefact also looks like.
- Status: **screen survivor, observe-only candidate.** It is not a ship. If it is pre-registered, use the natural cut 1.0 (not the fitted median) and the gate in §6.

**Other ① notes:**
- **HOLD_GREEN** (green candle ∧ `above_streak` > 12) stays positive on this cohort: N 123, +0.46 [−0.06, +1.12], 7 of 9 months. Its selection-adjusted p was 0.87 (overnight review). It is still the frozen observe item.
- **The loss side is robust and structural:**
  - reclaim bars (`above_streak` = 12): −0.87, CI [−1.25, −0.49];
  - ATR quintile 5: −0.92;
  - low `qv_rel`.
- **Blind spots in ①:**
  - pair rank by 24 h volume *within the live universe* (only the vol24 level was tested);
  - order-book stamps (none on the year);
  - 3D interactions;
  - quintile-level 2D;
  - `sweep_separators.py` was not used: its pool has no WIDE rows. I adapted `scripts/frenzy_wide_checklist.py` (same logic) into `wfq/checklist_trade.py`, plus listing age.
- **Blind spots in ②:** BTC eff72 is tested only as the binary EFF72_CHOP; 4h trend only as the golden-cross tag.

---

## 3. PART 2: WIDE's ride value

### 3a. How far WIDE fills run (565 fills, gross % from entry)

| horizon | ≥ +10 % | ≥ +20 % | ≥ +50 % | ≥ +100 % | median |
|---|---|---|---|---|---|
| 24 h, any path | 58 % | 38 % | 12 % | 3 % | 13.3 |
| 48 h, any path | 63 % | 45 % | 18 % | 7 % | 17.2 |
| **72 h, any path** | 65 % | 48 % | **21 % (117 fills)** | **10 % (54)** | 18.5 |
| **72 h, before first touching −3 %** | 21 % | 12 % | **4 % (22)** | 2 % (10) | 2.8 |
| FRENZY_LONG, 72 h any path | 56 % | 41 % | 19 % | 9 % | 14.5 |
| FRENZY_LONG, before −3 % | 26 % | 17 % | 8 % | 3 % | 4.1 |

- **Rides are common. Clean rides are rare.**
  - **Rides at ≥ +50 %:** 81 episodes in 261 days, ≈ **113 episodes a year**.
  - **Clean rides** (no −3 % touch first): 22 episodes, ≈ **31 a year**. RLC is in this class.
- **WIDE is not a ride specialist.** LONG has twice the clean-ride rate (8 % vs 4 %).

### 3b. What each exit captures

| ride class | N (per year) | perfect exit | lock (live) | runner E200·P2 30 % … 100 % leg | E200·BE | E100·BE | E200 no stop |
|---|---|---|---|---|---|---|---|
| any path ≥ +50 % | 117 (164) | +137 % | **−0.2 %** | +1.9 % | +1.8 % | +0.5 % | +19.3 % |
| clean ≥ +50 % | 22 (31) | +124 % | **+2.9 %** | +14.6 % (median +1.9) | +17.1 % | +9.9 % | +31.6 % |
| clean ≥ +20 % | 65 (91) | +64 % | +2.8 % | +6.4 % | +7.2 % | +5.1 % | +11.8 % |

- The no-stop runner looks best on rides, but it costs −1,635 pts on the non-rides (vs +809 on rides): a net −1.46 %/fill.
- **The non-rides pay for everything.** 447 of 565 fills never see +10 % before −3 %. They average −1.20 % on the lock (sum −538). The 118 rides bank only +352 between them.
- **Speed:** 43 % of fills touch −3 % within 15 min and 64 % within 1 h. The lock has closed 60 % of fills by 15 min and 92 % by 60 min.

### 3c. Can rides be told apart at entry or in the first minutes?

**At entry: weakly.**
- `qv_rel` doubles the clean-ride rate (15 % vs 8 %).
- GREEN vs ATR_HIGH: 12.7 % vs 10.8 %.
- The 22 clean ≥ +50 % rides split 12 ATR_HIGH / 10 GREEN.

**Post-entry: yes, for "is it a ride".** AUC among fills not yet stopped at T. The null 95th percentile of the max \|AUC − 0.5\| is 0.126.

| T | feature | alive fills | AUC | Jan–Apr / May–Sep |
|---|---|---|---|---|
| 60 min | return since entry | 209 | **0.78** | 0.80 / 0.76 |
| 60 min | close vs EMA20 | 209 | 0.76 | 0.81 / 0.72 |
| 15 min | return since entry | 357 | 0.70 | 0.73 / 0.68 |
| 30 min | above VWAP / green-bar share | 285 | 0.67 | |

Part of this is mechanical: a fill already up at 60 min is more likely to reach +20.

**Post-entry: no, for P&L.** I tested a hold-vs-cut policy:
- The runner leg follows runner rule RR from entry.
- At T it is held if condition C is true and cut at the T close if not.
- C is any of 10 post-entry features × T ∈ {5, 15, 30, 60} min × 4 cuts = 124 conditions, for each of 8 runner rules.
- Null: condition labels permuted within month; statistic = the max over the family.

| runner | hold-all Δ (100 % leg) | best condition | Δ [day CI] | selection p | −top5 | OOS (both directions) |
|---|---|---|---|---|---|---|
| E200·P2 | +0.15 | entry-bar VWAP distance at 5 min ≥ top third | +0.49 [−0.03, +1.22] | 0.21 | +0.02 | +0.08 / −0.15 |
| H50·P2 | −0.16 | same | +0.23 [−0.06, +0.56] | 0.21 | +0.01 | +0.09 / −0.24 |
| E100·BE | −0.65 | qv at 15 min ≤ bottom third | +0.11 | 0.10 | −0.23 | −0.09 / −0.06 |
| E200 / E100 / H50, no stop | −1.13 / −0.25 / −2.70 | | | ≥ 0.28 | < 0 | |

- **No post-entry rule beats its null.** Every winner collapses without its top 5 fills.
- Rides that are recognisable at 60 min have, by then, usually already dipped and shaken out the runner's floor.

### 3d. Runner exits on WIDE subsets (30 % runner, 72 h, two-sided)

| subset | runner | N | lock mean | Δ @ 30 % [day CI] | −top5 / −top10 | fills hurt (pts) / helped (pts) | months Δ > 0 | Δ incl. RLC (censored mark) |
|---|---|---|---|---|---|---|---|---|
| ALL WIDE | E200·P2 | 562 | −0.34 | +0.05 [−0.10, +0.27] | −0.11 / −0.11 | 182 (−62) / 40 (+88) | 2 / 9 | +0.11 |
| ALL WIDE | E100·BE | 558 | −0.34 | **−0.19 [−0.29, −0.07]** | −0.29 / −0.32 | 244 (−193) / 46 (+87) | 1 / 9 | −0.17 |
| ALL WIDE | E200 no stop | 414 | −0.19 | −0.49 [−1.08, +0.21] | −0.99 / −1.22 | 306 (−816) / 108 (+614) | 2 / 9 | −0.40 |
| GREEN | E200·P2 | 204 | +0.01 | −0.02 [−0.18, +0.24] | −0.16 / −0.16 | 77 (−31) / 14 (+27) | 1 / 9 | +0.15 |
| GREEN | E100·BE | 201 | +0.01 | **−0.30 [−0.41, −0.16]** | | 96 / 17 | 0 / 9 | −0.23 |
| HOLD_GREEN | E200·P2 | 123 | +0.46 | +0.05 [−0.21, +0.49] | −0.18 / −0.19 | 54 (−21) / 9 (+27) | 1 / 9 | +0.32 |
| HOLD_GREEN | E200 no stop | 90 | +0.63 | +0.53 [−0.84, +2.53] | −0.97 / −1.36 | 66 (−134) / 24 (+182) | 3 / 9 | +0.90 |
| ATR_HIGH | E200·P2 | 358 | −0.54 | +0.08 [−0.08, +0.40] | −0.08 / −0.09 | 105 (−31) / 26 (+61) | 3 / 9 | – |

- **RLC alone:** lock +5.3 (today's rule; live banked +3.01). Runner E200 +118.9 %, still open, marked 10-06 12:05 and censored. At a 30 % runner that is +34 pts on a single fill: as large as the whole year's E200·P2 Δ on ALL WIDE.
- **Every runner fails the bar on every subset.**
  - The CI spans 0 or sits below it.
  - Δ is negative without the top 5.
  - Only 1–3 of 9 months are positive.
  - The runner hurts 3–5× more fills than it helps.
  - Breakeven-stop runners are significantly *worse* than the lock.

---

## 4. PART 3: decision table

Books run from $3k, Jan 10 → Sep 27, on the TRADE universe. LONG is at live sizing (0.32 / strong 0.5) and sequenced together with the WIDE variant. **Off = $9,741 / −43 %**; each row's "vs off" is its book minus that. The halves are separate $3k books; off scores Jan–Apr $7,081 / May–Sep $4,127.

| # | option | WIDE lev | WIDE fills | WIDE %/fill [day CI] | year book | max DD | vs off | Jan–Apr / May–Sep books | evidence vs locked gates | RLC 10-05 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Off** | – | 0 | – | **$9,741** | **−43 %** | – | $7,081 / $4,127 | baseline | missed (LONG refused the green candle) |
| 2 | Keep as-is | 0.2 | 565 | −0.33 [−0.58, −0.08] | $1,188 | −94 % | −$8,553 | $3,723 / $957 | **fails**: CI below 0 in every cut | +3.01 live (≈ +2.8 % equity); today's lock ≈ +5.3 (≈ +5 % equity) |
| 3 | Probe as-is | 0.05 | 565 | same | $5,511 | −66 % | −$4,230 | $6,160 / $2,684 | fails (as row 2) | ≈ +1.2 % equity |
| 4 | Keep as-is + runner (30 % E200·P2) | 0.2 | 562 | −0.34 lock; Δ +0.05 | $1,308 | −94 % | −$8,433 | $3,552 / $1,104 | fails | **≈ +37 % equity** (blended +39.4 %; censored mark) |
| 5 | Filter: GREEN only (block ATR_HIGH) | 0.2 | 204 | +0.01 [−0.43, +0.50] | $8,082 | −66 % | −$1,659 | $8,064 / $3,007 | blocked half passes the bar levels; kept half ≈ 0 and decaying | taken (+5.3 lock) |
| 6 | Same, probe | 0.05 | 204 | same | $8,667 | −50 % | −$1,074 | $7,384 / $3,522 | same | taken (≈ +1.2 % equity) |
| 7 | Filter: HOLD_GREEN (post-hoc, frozen 10-05) | 0.2 | 124 | +0.43 [−0.13, +1.03] | $13,474 | −54 % | +$3,733 | $9,819 / $4,117 | **post-hoc**, selection-adjusted p 0.87; CI spans 0; observe-only | taken |
| 8 | Same, probe | 0.05 | 124 | same | $9,784 | −45 % | +$43 | $7,732 / $3,796 | same | taken |
| 9 | Filter: `qv_rel` > 1.0 (this review) | 0.2 | 307 | +0.18 [−0.18, +0.56] | $12,668 | **−68 %** | +$2,927 | $7,392 (DD −68 %) / $5,141 | in-sample family p 0.001, 9 of 9 months; **no replication on 3 independent sets**; threshold not frozen before data | **blocked** (qv 0.45) |
| 10 | Same, probe | 0.05 | 307 | same | $9,770 | −47 % | +$29 | $7,253 / $4,041 | same | blocked |
| 11 | Filter: GREEN ∧ `qv_rel` > 1.0 (post-hoc 2D pick) | 0.2 | 137 | +0.54 [−0.05, +1.12] | $16,167 | −53 % | +$6,426 | $9,772 / $4,963 | two post-hoc conditions; CI touches 0; 6 of 9 months | blocked |
| 12 | Same, probe | 0.05 | 137 | same | $10,257 | −44 % | +$516 | $7,727 / $3,982 | same | blocked |
| 13 | Row 9 + 30 % E200·P2 runner | 0.2 | 307 | +0.18 lock (runner Δ not tested on this subset) | $16,913 | −70 % | +$7,172 | $7,728 / $6,566 | runner fails §3d everywhere; this book gain is tail-carried | blocked |
| 14 | Row 7 + 30 % E200·P2 runner | 0.2 | 124 | Δ +0.05 | $13,593 | −56 % | +$3,852 | $10,764 / $3,788 | same | taken; ≈ +37 % equity (censored) |

**How to read this:**
- **Rows 2–6 are the evidence-backed rows. None beats off.** WIDE as-is loses money at any size. Green-only costs $1–1.7k against off and adds 7–23 points of drawdown.
- **Rows 7–14 beat off only on post-hoc subsets.**
  - Every one also has a higher max DD than off.
  - Applying the 30–50 % in-sample haircut to the gain vs off: row 11 drops from +$6.4k to about +$3.2–4.5k, and row 9 from +$2.9k to about +$1.5–2.0k.
  - The probes at 0.05 (rows 8, 10, 12) end within about $0.5k of off. They buy information, not money.
- **The RLC column is the trade-off in one line.** The only evidence-backed entry filter (`qv_rel`) would have *refused* RLC.
  - The variants that catch RLC are: as-is (losing), green-only (≈ flat), and HOLD_GREEN (post-hoc).
  - Only a runner turns RLC into +37 % of equity, and runners fail on the year.

---

## 5. What is and is not established

**Established on this cohort:**
- WIDE as-is loses money (−0.33 %/fill, CI below 0) at both sizes.
- Its loss sits in the reclaim / ATR_HIGH / quiet-bar fills.
- Its rides are real and frequent (≈ 113 ≥ +50 % episodes a year), but 84 % of them dip through −3 % first.
- No tested exit, at-entry selection, or post-entry hold/cut rule captures the rides net of the non-rides.

**Not established:**
- That any WIDE subset has a positive edge out of sample.
- What the WIDE-specific common factor behind the post-April decline is (checklist ③; still unexplained).

**Operator's point vs the data:**
- The prize is large: 31 clean rides a year averaging +124 % at the perfect exit.
- The year says the bot's current exits bank about 2 % of it.
- The binding problem is exit design on rides that dip first, more than WIDE's entry. Every runner variant tested so far was refuted.

---

## 6. Cheapest forward test (operator decision; nothing changed)

**Zero-capital shadow.** The scout already re-walks every WIDE fresh signal. Add three frozen observe lines, judged together at each batch review:

1. **`WIDE_QV1`** = WIDE fresh signal with signal-bar `qv_rel` > 1.0. `qv_rel` = the signal bar's quote volume divided by the mean of the prior 12 closed 5m bars, computable from public klines at zero bot risk.
   - **Promotion bar:** ≥ 40 shadow signals on ≥ 20 distinct days; mean > 0; **gap vs the `qv_rel` ≤ 1.0 shadow ≥ +0.5** with day-block CI low > 0; no pair > 50 % of the net.
   - **Retire** if the gap is ≤ 0 at 40 / 40.
   - Expected wait ≈ 35 days (≈ 1.2 signals a day).
2. **`WIDE_HOLD_GREEN`**: already frozen on 10-05 (gate: ≥ 40 over ≥ 8 windows, mean > 0, top pair < 50 %).
3. **`WIDE_RIDE`**: per shadow signal, record:
   - the 72 h any-path and clean peak;
   - the lock and E200·P2 / no-stop runner outcomes;
   - post-entry return at 15 / 60 min.

   This lets the ride-capture question be answered on forward data without re-fitting the year.

**If the operator wants real WIDE fills meanwhile**, the least-damaging real-money rows are 8 / 10 / 12 (probe 0.05, about −$0 to +$0.5k vs off on the year). Each needs a D11 change (new filter field, UI, `_record_filter_block`) and a pre-committed revert at 40 fills (mean ≤ 0 → off). Row 2 at 0.2 has no support.

A `qv_rel` stamp on fills would be a D11 / D12 change. The shadow does not need it.

---

## 7. Blind spots

1. **Post-hoc.** HOLD_GREEN, the `qv_rel` cut, and every 2D pocket were read on the same year. Only forward data confirms them. `qv_rel` already fails three replication sets.
2. **Day = window.** Episodes (pair × spike) are a finer unit. For the gates, the day bootstrap is the conservative choice.
3. **Post-entry features and cut-at-T prices use 5m bars, not ticks.** The runner legs use the reviewed tick pricer (5m after 12 h). The MFE uses 5m highs from the entry bar, which can include prints a few seconds before entry.
4. **The "clean ride" rule is pessimistic.** A bar whose low touches −3 % is counted as stopped even if its high came first.
5. **RLC is out of cohort** (Oct 5) and its runner value is a censored mark.
6. **Not tested:**
   - order-book / order-flow stamps (none on the year);
   - stop-width × runner grids (wider stops were refuted 10-05; not re-run);
   - 3D cells;
   - continuous BTC eff72 / 4h slope.
7. **Other sleeves holding a pair, and outages, are not modelled.** They can only remove fills.
8. **Books** compound a shared equity with LONG and assume the live slot rules. Half-books restart at $3k.
