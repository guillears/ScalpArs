# FUNDING STUDY — every sleeve (2026-11-21)

Research only. No config, engine, strategy or git change. Operator question: *"shouldn't we do something when funding is this huge due to the 19 hs?"* (ENSOUSDT EXHAUSTION_SHORT, open ≈ 19 h, funding −4.36 % of notional, price +3.84 % our way, showing −0.60 %). Scope widened twice by the operator: every sleeve, and NET take-profit / NET stop rules.

Scripts: `scripts/FUNDING_STUDY_ALL_2026-11-21_{cohort,bt,real,report}.py` + `scripts/EXH_FUNDING_STUDY_2026-11-21_fetch.py`. Data: `reports/FUNDING_STUDY_ALL_2026-11-21_data/`, rate / kline cache `reports/.funding_cache/` (gitignored). Every grid cell and every block in full: `reports/FUNDING_STUDY_ALL_2026-11-21_tables.md`. Chart: `reports/FUNDING_STUDY_ALL_2026-11-21_chart.png`.

## 0. Summary

**Bottom line.** Funding costs real money on only two sleeves: EXHAUSTION and FLATBASE. Every rule tested loses money or is a statistical tie. **Recommendation: change nothing.** The fix for the ENSO case already shipped as DL 609 (the TP cut from 10 to 7). Under TP 7, ENSO hits its target 2.5 h after entry, books +6.18 % and pays 0.75 % funding. It did not because it opened under TP 10. When the deploy clamped the TP to 7 (2026-11-21 20:59 UTC), price had already pulled back to about +3.8 %.

The units are % of notional at ONE size. "Share" is funding as a % of the sleeve's price-only P&L, and a negative share means the sleeve paid. Δ is the rule minus today, per fill, in % points, with a day-block 95 % CI. "BT" is the backtest. "—" means the rule cannot act, because no fill reaches the smallest budget (0.5 %) or holds longer than 6 h.

| sleeve | funding: real fills (master · batch) | funding: backtest | ① entry skip / ½ | ② funding budget | ③ time stop 6 / 8 / 12 h | ④ NET TP | ⑤ NET SL | ⑥ NET TP + SL | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **EXHAUSTION_SHORT** | master 0 fills · batch 8 closed: −6.19 pts (−$379) = −26 % of price P&L as booked; **−1.51 pts under TP 7** | 4,229 fills: −538 pts = **−20 %** of price P&L, −0.13 %/fill · DCR 1.08 → 0.82 %/day | **refuted**: the blocked fills are BETTER (≥ 0.05 %: +0.59 vs kept +0.46; HOLDOUT +0.84) | **refuted**: every cell ≤ 0 (F 0.5: −0.08, CI −0.12…−0.04) | **refuted**: T6 −0.09 (CI −0.15…−0.03) · T12 −0.04 | tie: +0.014 (CI −0.012…+0.037), family p 0.78 | **refuted again under TP 7**: −0.018 (CI −0.043…+0.004, P(better) 0.06) | tie: −0.007 | **leave as is** (TP 7 already does the job) |
| **FLATBASE_SHORT** (stop already counts paid funding, DL 607) | master 17: −2.49 pts (−3.0 %) · batch 3: −1.59 (KERNEL) | 541 fills: −108 pts = **−15 %** · −0.20 %/fill | **refuted**: blocked fills better (+1.25 vs +1.00) | **refuted**: every cell ≤ 0 (F 0.5: −0.17, CI −0.34…−0.00) | not tested (operator) | +0.07 (CI −0.03…+0.15; DESIGN +0.17, HOLDOUT +0.05), family p 0.46; master +0.15 on 6 fills | **DL 607 confirmed**: price-only stop instead = −0.018 (CI −0.06…+0.04) | +0.05 vs today | **leave as is.** NET TP is a TP-level effect: track it in the existing FB_TP6_SHADOW line (DL 604), no new line |
| **DISTRIBUTION_SHORT** | batch 6: +0.07 pts (received) · master 0 | 1,545 fills: −135 pts = −9 % · −0.09 %/fill | **refuted** (blocked +1.55 vs kept +0.81) | **refuted** (all ≤ 0) | **refuted**: T6 −0.15 · T12 −0.08 | −0.006 | **refuted again**: −0.023 | −0.020 | **leave as is** |
| **FRENZY_WILLY_HT** | master 12: −0.01 · batch 12: −0.09 pts (−0.8 %) | 417 fills (old exit): −9.8 pts (hand-computed, no committed generator) | cuts 1–2 winners | — (worst fill −0.04) | — (no fill > 6 h) | not walked (≤ 0.04 % paid per fill) | not walked | not walked | **immaterial, leave as is** |
| **FRENZY_WILLY** | master 24: −0.00 | 545 fills (old exit): −3.7 pts | — | — | — (90-min cap) | immaterial | immaterial | immaterial | **immaterial** |
| **MOMENTUM** (long + short, OFF) | master 136 full size: +0.01 pts (+0.02 %) | no priced backtest | cuts 1–2 winners | — | — (max hold 2.65 h) | immaterial | immaterial | immaterial | **immaterial** |
| **FLIP (FAN, OFF)** | master 57: −0.17 pts (−3.2 %), all from INIT 11-16 (DL 566 FUNDING_GUARD already blocks it) | no priced backtest | refuted (blocked 4 / 5 winners) | — | — | immaterial | immaterial | immaterial | **immaterial; DL 566 covers it** |
| FRENZY_LITE · LITE_RETRY · FRENZY_LONG · SPIKE_FADE · BULLRUN · BEARRUN · SURGE (all OFF) | +0.11 · +0.72 · **+5.96 (received)** · 0.00 · 0 · 0 · 0 pts | none | cuts winners or nothing | — | — (one LITE_RETRY fill > 6 h, manual close) | immaterial | immaterial | immaterial | **immaterial** |

**Funding pricer parity (validation).** On the 49 KEPT fills booked with funding since DL 539 (11-13) (reports/FUNDING_STUDY_ALL_2026-11-21_data/real_cohort_final.csv, funding_in_pnl True), the modelled funding matches the booked funding to the cent (max |gap| $4 × 10⁻⁷; 9 of the 49 paid or received something). Unkept fills (e.g. momentum AVAX / XRP 11-16, FB TAC 11-14) were not priced (deep review correction: an earlier draft said "all 55"). The EXH backtest walker reproduces the DL 603 / 609 TP-7 cell: exit type matches on 99.8 % of fills, median |gap| 0.005 pts, funding |gap| 0.005–0.010. The EXH real re-walk under each fill's own era exit is within ±0.03 pts of booked on 5 fills and 0.15–0.52 on the 3 stops (booked stop slippage). The open ENSO mark is reproduced exactly: −0.64 % with −4.36 % funding, against the operator's −0.60 / −4.36. The FB real re-walk matches the DL 607 study exactly (20 / 20).

**One correction to the brief.** The master has no EXH fills (0 in `MASTER_POOL_stacked.csv`). But the EXH backtest behind DL 603 / 609 already included **real funding** (TP_GRID sim column `f`), so funding was never a backtest blind spot. The new information here is how large it is (§1.3) and that no rule beats paying it.

## 1. EXHAUSTION_SHORT (first and deepest)

### 1.1 Theory: why EXH shorts pay funding, and what that predicts

EXH enters after a FRENZY pump, when 15m RSI crosses back below 70 while volume is still high and the spot share is falling (the move is perp-led). At that moment many traders are already short the top. Perp prices trade below the index, so the funding rate is negative, and a short pays. It is the same crowding that makes the fade work: the shorts are right on average. Real entries confirm it. 7 of the 8 batch fills entered with funding at or below −0.66 % per settlement (§1.4), and the long sleeves on the same coins **received** funding (FRENZY_LONG +5.96 pts on 26 fills).

Predictions:
- (a) A large negative funding rate marks a crowded short, not a bad short. Skipping on it should remove good fills.
- (b) Funding piles up with time held, so it hurts only long holds. A shorter TP (DL 609) is the direct fix.
- (c) Paid funding is sunk at the moment of any decision. Moving the TP further asks the market for more than the original target. It does not recover the cost. Tightening the stop by the amount paid turns near-miss winners into stops; DL 607 found this for EXH under TP 10.

Every prediction holds below.

### 1.2 Cohort reconciliation

| block | definition | N |
|---|---|---|
| ① current batch | export `scalpars_orders_paper_2026-11-21_22-18-17.csv`, EXHAUSTION_SHORT, CLOSED, no probes | **8** (+ ENSO open, not in the export; rebuilt from the decisions journal `OPEN 2026-11-21 03:30:33 @ 0.480889`, ATR 3.47 %) |
| ② master on today's stack | `MASTER_POOL_stacked.csv` `stack_keep` | **0**: EXH started 11-18, after the last master batch (B27). Same as DL 603 ("5 real, 0 master") |
| ③ backtest | the DL 603 / 609 set: TP_GRID rows_FRENZY 3,368 (DL 582 kept, one position per pair) + rows_WIDE 939 (DL 585 L4 ∧ DL 582) = **4,307** = the "FRENZY ∪ WIDE" set of DL 609 | 4,229 with funding data (78 pair-months missing, excluded) |

Dedup key (opened_at, pair, direction). Re-priced under today's exit: TP min(20 × ATR15, 7) (DL 609) / SL min(1.5 × ATR15, 6) / 24 h, price stop. Fills are walked on 1m from the bar after the fill: maker TP only ≥ 0.05 % through, stop checked first on the bar high, a new market exit pays taker 0.05 % + slip ≥ 0.10 %. Each funding settlement is charged at rate × (price at the settlement ÷ fill), on the real settlement timestamps (hourly coins get hourly rows). One size X = 0.61 × equity (as in TP_GRID 11-20 / DL 607); "$" means dollars on a $10,000 account at one size. Eras: DESIGN < 2025-07-01 ≤ HOLDOUT.

### 1.3 How much funding EXH pays

**Backtest, today's exit, by hold time** (pts = Σ % of notional; share = funding ÷ price-only P&L):

| block | hold | N | % of fills paying | funding pts | avg | $ at one size | price-only pts | share |
|---|---|---|---|---|---|---|---|---|
| ALL | ≤ 1 h | 1,471 | 7 % | −51 | −0.03 | −$3,103 | −1,779 | — (losers) |
| ALL | 1–4 h | 1,492 | 29 % | −197 | −0.13 | −$12,047 | +1,477 | −13 % |
| ALL | 4–8 h | 604 | 48 % | −143 | −0.24 | −$8,748 | +1,268 | −11 % |
| ALL | 8–24 h | 662 | 50 % | −146 | −0.22 | −$8,903 | +1,694 | −9 % |
| **ALL** | all | **4,229** | 27 % | **−538** | **−0.13** | **−$32,802** | **+2,659** | **−20 %** |
| DESIGN | all | 1,645 | 22 % | −132 | −0.08 | | +660 | −20 % |
| HOLDOUT | all | 2,584 | 30 % | −405 | −0.16 | | +1,999 | −20 % |
| FRENZY | all | 3,368 | 27 % | −469 | −0.14 | | +2,468 | −19 % |
| WIDE | all | 861 | 27 % | −69 | −0.08 | | +191 | −36 % |

Before → after, with funding versus without (the "blind" read). ALL: 46.9 % WR, +0.629 → **+0.502 %/fill**, DCR **1.08 → 0.82 %/day**, max DD −55 → −65 %. HOLDOUT: +0.773 → +0.617, DCR 2.07 → 1.58. DESIGN: +0.401 → +0.321.

Who pays: winners pay twice as much as losers (−358 vs −179 pts in the decomposition below), because winners stay open longer (chart, right panel). The 93 fills that entered on an hourly-funding coin paid 0.68 %/fill against 0.12 % for the rest, yet **netted +1.32 %/fill against +0.48**.

**Batch, as traded** (TP 10 era): 8 closed fills, **+$1,085 P&L and −$379 funding**, all from 3 fills held 5–15 h (ZKC −$195, RARE #26 −$97, RARE #32 −$87). **Re-priced under TP 7**, the funding falls from −6.19 to −1.51 pts and the avg goes from +2.26 to **+3.55 %/fill**. The one-size DCR goes 2.66 → **4.33 %/day** (4.56 without funding). The account-path DCR as traded is 2.71 %/day, UNVERIFIED (assumes a $10,000 start equity; the export carries none). ⚠ That before → after is mostly RARE #26, which reached +7.95 and then stopped at −7.42: the knife-edge fill DL 609 already flagged, so it is in-sample.

**Path decomposition** (DL 406; WORKED = +1 % our way before −SL; backtest ALL):

| entry label | N | winners | losers | losers' pts | funding paid by losers | funding paid by winners | median reach |
|---|---|---|---|---|---|---|---|
| WORKED | 3,581 | 1,982 | 1,599 | −7,653 | −166 | −358 | +7.35 |
| NEVER | 645 | 0 | 645 | −2,847 | −14 | — | +0.45 |

The larger realised loss is on the exit side: GAVE_BACK fills (WORKED, still lost) carry −7,653 pts against −2,847 for NEVER_WORKED. Funding is a small part of either.

**Chart** (`reports/FUNDING_STUDY_ALL_2026-11-21_chart.png`): among fills still open, winners and losers pay funding at the **same** rate (about 0.15–0.25 % by hour 10–24), so funding does not separate them. The hourly-coin subset pays about 1 % by hour 10 and about 1.9 % by hour 24. Losers still open after about 15 h drift down to −0.6 % price. That is the population a time stop would target; T12 still loses (§1.5) because the same exit cuts more winners.

**Concentration** (backtest funding paid): 429 pairs, the top pair (ARK) 2.8 %, top 5 pairs 10.9 %. The top day is 1.1 %, the top 10 days 9.7 %. Hourly coins at entry are 2.2 % of fills and 11.4 % of the funding paid. The cost is spread out, not a pair or window problem. The entry paid rate vs funding paid rank correlation is +0.30.

### 1.4 The real fills

| fill | entry rate paid (stamp / last settled) | interval | as booked (era exit) | TODAY's exit: result | funding | hold | label · reach |
|---|---|---|---|---|---|---|---|
| RARE #32 | 1.04 / 1.07 % | 4 h | TP +5.31 (−1.65 funding, 6.8 h) | TP +6.22 | −0.71 | 1.8 h | WORKED · 9.7 |
| RARE #31 | 1.31 / 1.07 | 4 h | STOP −4.95 | STOP −5.47 | 0 | 0.45 h | NEVER · 0.8 |
| RARE #26 | 1.76 / 1.76 | 4 h | STOP −7.42 (−1.43 funding, 15.3 h) | **TP +6.93** | 0 | 0.67 h | WORKED · 8.0 |
| KERNEL #24 | 2.00 / 0.92 | 4 h | TP +9.95 | TP +6.93 | 0 | 1.3 h | WORKED · 24.0 |
| YB #15 | 0.77 / 0.66 | 4 h | STOP −6.58 | STOP −6.21 | 0 | 0.1 h | NEVER · 0.1 |
| 1000LUNC #8 | 0.03 / 0.03 | 8 h | TP +9.96 | TP +6.93 | 0 | 0.7 h | WORKED · 11.5 |
| ZKC #3 | 0.78 / 0.66 | **1 h** | TP +6.83 (−3.11 funding, 5.1 h) | TP +6.13 | −0.80 | 1.45 h | WORKED · 11.7 |
| ZKP #1 | 1.63 / 1.05 | 4 h | TP +4.94 (study exit) | TP +6.93 | 0 | 1.5 h | WORKED · 20.6 |
| **ENSO (open)** | — / 0.03 (next 0.78) | 4 h | open, −0.64 with −4.36 funding at 19 h | **TP +6.18** | −0.75 | **2.5 h** | WORKED · 9.7 |

The real rules on the 8 closed fills (① block):
- Every budget cell and every time stop: Δ 0 on 7 or 8 fills. B0.5_4 changes one fill, −0.46.
- NET TP: +0.19 %/fill (RARE #32 and ZKC end at 6.93 instead of 6.2).
- NET SL: Δ 0.
- Entry skip at paid ≥ 0.05 … 0.50 %: blocks 7 of 8 fills, Δ −2.68 %/fill. At ≥ 1.00 %: blocks 4 of 5, −1.83. Interval ≤ 1 h: blocks ZKC (a winner), −0.77.

The live stamp and the last settled rate agree in sign on every stamped fill (8; ENSO has no live stamp), and in size within about 0.6 pts except KERNEL (stamp 2.00 vs settled 0.92). So the backtest's lag-safe proxy is a fair stand-in for the stamp.

### 1.5 Rules, before → after (backtest; full grid in the tables file)

| rule (frozen grid) | DESIGN Δ (CI) | HOLDOUT Δ (CI) | ALL avg · DCR · maxDD | ALL Δ (CI) · P(better) | saved (stop → better) | cut (TP fill worse) | Δ at 0.30 % slip |
|---|---|---|---|---|---|---|---|
| **today, with funding** | — | — | +0.502 · 0.82 · −65 % | — | | | |
| budget F 0.5 · G 0 | −0.04 (−0.09…+0.01) | −0.10 (−0.16…−0.04) | +0.424 · 0.66 · −78 % | −0.077 (−0.12…−0.04) · 0.00 | 121 · +474 | 100 · −766 | −0.088 |
| budget F 1.0 · G 0 | −0.02 | −0.06 (−0.11…−0.02) | +0.453 · 0.72 | −0.048 (−0.08…−0.02) · 0.00 | 60 · +232 | 53 · −419 | −0.054 |
| budget F 2.0 · G 0 | −0.01 | +0.00 | +0.500 · 0.81 | −0.001 (−0.02…+0.01) · 0.44 | 14 · +60 | 8 · −65 | −0.002 |
| budget F 3.0 · G 4 | −0.00 | −0.00 | +0.498 · 0.81 | −0.004 · 0.19 | 2 · +13 | 5 · −29 | −0.004 |
| time stop 6 h | −0.20 (−0.32…−0.08) | −0.02 (−0.09…+0.06) | +0.413 · 0.68 | −0.088 (−0.15…−0.03) · 0.00 | 287 · +1,380 | 439 · −1,711 | −0.130 |
| time stop 8 h | −0.14 | −0.03 | +0.430 · 0.70 | −0.071 (−0.12…−0.02) · 0.00 | 194 · +885 | 304 · −1,147 | −0.103 |
| time stop 12 h | −0.09 (−0.16…−0.02) | −0.01 (−0.05…+0.03) | +0.461 · 0.75 | −0.041 (−0.08…−0.01) · 0.01 | 98 · +454 | 171 · −589 | −0.061 |
| **NET TP** (resting limit re-priced at each settlement) | +0.009 (−0.02…+0.03) | +0.017 (−0.02…+0.05) | +0.516 · 0.84 · −61 % | +0.014 (−0.012…+0.037) · 0.86 | — | 629 changed · +59 | +0.014 |
| **NET SL** (stop moves in by funding paid) | −0.018 | −0.019 (−0.05…+0.01) | +0.483 · 0.78 · −70 % | −0.018 (−0.043…+0.004) · 0.06 | 435 · +178 | **23 · −232** | −0.018 |
| **NET TP + SL** | −0.004 | −0.008 | +0.495 · 0.80 | −0.007 (−0.04…+0.02) · 0.35 | 435 · +178 | 629 · −183 | −0.007 |
| skip paid ≥ 0.05 % | blocked 352 · −0.11 | blocked 970 · **+0.84** | kept 2,907 · +0.46 · DCR 0.51 | blocked 1,322 · **+0.59** · P(blocked < 0) 0.00 | | | |
| skip paid ≥ 0.50 % | blocked 122 · +0.12 | blocked 354 · +0.82 | kept 3,753 · +0.48 · 0.69 | blocked 476 · +0.64 | | | |
| skip interval ≤ 1 h | 6 · +0.01 | 87 · +1.41 | kept 4,136 · +0.48 · 0.76 | blocked 93 · **+1.32** | | | |
| ½-size versions | | | DCR 0.68 … 0.80 (all < 0.82) | | | | |

Notes on the table:
- **Live-stopped cohort rule (CLAUDE.md Sep-23 / DL 110).** Every changed fill is split by today's outcome: saved / deeper (today STOP), cut (today TP), vs time (today TIME). Fills a rule does not touch show Δ = 0 exactly (`unchanged_max` 0.0 in every block), so the replica is not inventing changes. On real fills, the only live-stopped fills (RARE #31, YB) stop under every rule. Budget and time rules cut more TP winners than they save stops, in both eras.
- **LOO.** The best exit cell (NET TP) is +0.012 … +0.019 leaving out any one pair, +0.013 leaving out any one day. It is stable but tiny.
- **Family-wise null.** Exit family (21 cells): best NET TP +0.014, p = 0.78 (day-cluster sign flips). Entry family (12 cells): best cell −0.014, p = 0.94 (joint permutation). Nothing beats luck.
- **Haircut.** The best cell after a 30–50 % in-sample haircut is +0.007 … +0.010 %/fill, which is noise.
- **Per half-year.** NET TP Δ is −0.03 … +0.05 by half-year: positive in 4 of 7, negative in 3. Funding per fill grows over time: −0.02 … −0.06 in 2023-H2 to 2024-H2, then −0.11 … −0.21 from 2025-H1 on.

### 1.6 Verdict: EXHAUSTION_SHORT

- **① Entry skip / ½ on adverse funding: REFUTED.** The blocked cohort is above the sleeve's breakeven WR (42 %): 50–60 % WR, avg +0.46 … +1.32, P(blocked mean < 0) 0.00–0.09, and better than the kept fills in HOLDOUT and ALL. Prediction (a) holds: crowded shorts are the good shorts. It also fails the locked expectancy bar on WR and sign.
- **② In-trade funding budget: REFUTED.** All 15 cells are ≤ 0. F ≤ 1.5 is negative beyond its CI in HOLDOUT.
- **③ Time stop 6 / 8 / 12 h: REFUTED.** Negative in ALL with the CI below 0, and worse at 0.30 % slip.
- **④ NET TP: no ship** (a tie, +0.014, p 0.78). It is not worth an observe line: the TP level itself (7 vs 8 vs 10) is already watched by the DL 609 gate.
- **⑤ NET SL: REFUTED again**, now under TP 7 (−0.018, P 0.06). The DL 607 finding stands: the stop is about 5–6 %, winners graze it, and one cut costs about 10 pts while one save gains about 0.4.
- **⑥ Both: REFUTED / tie** (−0.007).
- **Today's rule, with funding:** +0.50 %/fill and DCR 0.82 %/day on the backtest. Funding takes about 20 % of the price edge. That is the cost of holding the crowded-short fills, and those are the best fills. **The lever already pulled is TP 7** (holds about 40 % shorter, DL 609), which cut the batch's funding from 6.2 to 1.5 pts. The remaining lever is **size** (EXH is about 16× ¼-Kelly; already on the pre-live list).
- **Watching:** the DL 613 TP-7 gate already prints funding % and hold per fill. No new gate is needed.

## 2. FLATBASE_SHORT

Cohort: master 17 + batch 3 = **20** (`KEPT_LIST_FLATBASE_2026-11-21.csv`: today's filters incl. DL 601 / 577), with today's-stack entries from the DL 607 study (12 MODEL_CONFIRM, 8 REAL). Backtest: the OOS_REAL_FLATBASE NEAR_BASE rebuild with the DL 601 cut, 545 fills (541 with funding), 5m bars. Today's exit: TP 5 (price) / stop 10 that **counts funding paid** (DL 607) / 72 h. Per the operator, the time stop is not tested here.

Funding:
- **Real:** master −2.49 pts (−3.0 % of price P&L; POWR −1.82 is 73 % of it). Batch −1.59 (KERNEL, hourly). 7 of 17 master fills held > 6 h.
- **Backtest:** −108 pts = −15 % of price P&L, −0.20 %/fill. Avg +1.33 blind → +1.13 with funding; DCR 0.45 → 0.37; max DD −31 → −37 %.

| rule | ① batch (3) | ② master (17) | ③ BT DESIGN Δ (CI) | ③ BT HOLDOUT Δ (CI) | ③ BT ALL Δ (CI) · P | verdict |
|---|---|---|---|---|---|---|
| today (net-of-paid stop) | −5.27 %/fill | +4.78 %/fill · 100 % WR | +1.31 | +1.09 | +1.13 · DCR 0.37 | |
| budget F 0.5 · G 0 | +1.79 (KERNEL out at −5.0 instead of −10.4) | −0.30 (POWR cut −5.06) | +0.15 (−0.16…+0.52) | −0.23 (−0.42…−0.05) | −0.17 (−0.34…−0.00) · 0.02 | **refuted** |
| budget F 1.0 · G 0 | +1.82 | −0.41 (POWR −7.03) | −0.02 | −0.11 | −0.09 (−0.22…+0.02) | **refuted** |
| budget F 2 / 3 | 0 | 0 / 0 | −0.09 / −0.03 | −0.04 / 0.00 | −0.05 / −0.01 | **refuted** |
| **NET SL** = today (DL 607) vs price-only stop | price-only: −0.53 (KERNEL −1.60) | 0 | price-only −0.05 (−0.10…−0.01) | price-only −0.01 | price-only −0.018 (−0.06…+0.04) | **keep DL 607** (it holds on today's stack) |
| **NET TP** (+ today's net SL) | 0 | **+0.15** (6 fills, every one ≥ 0: POWR +1.82, CYBER +0.12, TST +0.27, G +0.23 …) | **+0.17 (+0.06…+0.33)** | +0.05 (−0.06…+0.14) | +0.07 (−0.03…+0.15) · 0.93 | no ship (family p 0.46) |
| NET TP + price SL | −0.53 | +0.15 | +0.12 | +0.04 | +0.05 | no |
| skip paid ≥ 0.05 % | blocks KERNEL (+3.46) | blocks 7 winners (−1.89) | blocked +1.30 | blocked +1.24 | blocked 268 · 78 % WR · +1.25 vs kept +1.00 (BE WR 68 %) | **refuted** |
| skip interval ≤ 1 h | +3.46 (KERNEL) | −0.58 (2 winners) | blocked 3 · −2.98 | blocked 46 · +1.01 | blocked 49 · 76 % · +0.76 · P(< 0) 0.20 | **refuted** (fails the expectancy bar; matches DL 607 arm C) |

LOO (NET TP, backtest): +0.059 … +0.096 by pair (top HOLO 12 %), day-out min +0.059. Haircut 30–50 % → +0.035 … +0.049 %/fill, **UNVERIFIED** (5m bars, maker TP ≥ 0.05 % through).

Why NET TP looks positive on FB: it is a **higher TP on the fills that paid funding**, not funding being recovered (theory (c)). FB fills that pay are long, trending shorts that often run past +5. That is the same question as FB_TP6_SHADOW (DL 604: TP 6 vs 5, +0.24 Δ, pick-the-best p 0.32). **Verdict: leave as is.** Read NET TP inside the FB_TP6_SHADOW line at the batch review; do not add a separate line. The funding budget would have saved KERNEL (+5.4 pts) but cuts POWR-type winners. In the backtest it loses in HOLDOUT beyond its CI, and KERNEL is the motivating fill (DL 561: never refit around the trade that broke it).

## 3. DISTRIBUTION_SHORT

Cohort: batch 6 (holds ≤ 3 h, funding +0.07 received), master 0. Backtest: the DL 586 kept set, 1,545 with funding, TP min(20 × ATR, 10) / SL min(1.5 × ATR, 6) / 24 h (DIST was not changed by DL 609).

Funding (backtest): −135 pts = −9 % of price P&L. Avg +0.997 blind → +0.909; DCR 0.64 → 0.58 %/day.

| rule | ① batch | ③ DESIGN Δ (CI) | ③ HOLDOUT Δ (CI) | ③ ALL Δ (CI) · P | verdict |
|---|---|---|---|---|---|
| budget F 0.5 · G 0 | 0 (no fill paid) | −0.07 (−0.16…+0.01) | −0.07 (−0.16…+0.02) | −0.068 (−0.13…−0.01) · 0.01 | **refuted** |
| budget F 2 · G 0 | 0 | +0.00 | −0.02 | −0.013 | **refuted** |
| time stop 6 / 8 / 12 h | 0 (no hold > 3 h) | −0.36 / −0.26 / −0.17 | −0.02 / +0.01 / −0.03 | −0.145 / −0.090 / −0.082 | **refuted** |
| NET TP | 0 | −0.02 | +0.00 | −0.006 (−0.05…+0.03) | **refuted** (tie) |
| NET SL | 0 | −0.02 | −0.03 | −0.023 (−0.07…+0.01) · 0.13 | **refuted** (as DL 607) |
| NET TP + SL | 0 | −0.04 | −0.01 | −0.020 | **refuted** |
| skip paid ≥ 0.05 % | blocks CYBER W + KERNEL L (−0.64) | blocked +1.05 | blocked +1.69 | blocked 208 · 48 % · **+1.55** vs kept +0.81 | **refuted** |
| skip interval ≤ 1 h | — | 2 fills | 19 · +3.57 | 21 · 57 % · +3.32 | **refuted** |

Family-wise null: exit best NET TP −0.006, p 0.996; entry p 0.99. **Verdict: leave as is.**

## 4. FRENZY_WILLY_HT

Funding is immaterial. Real: master 12 fills −0.01 pts, batch 12 fills −0.09 (−0.8 % of price P&L); worst fill −0.04 %. No fill held > 6 h (max 4.6 h) and none paid ≥ 0.5 %, so the budget and time-stop cells give Δ = 0 by construction. NET TP / NET SL were not walked: paid funding is ≤ 0.04 % per fill, which moves a TP / stop of 6 / 4 % by less than 1 %. Entry cells block 1 master winner (JCT +8.0, or INIT +1.25). Backtest (BASELINE_TRADES_WILLY_HT, **old HT exit**, holds ≤ 1.5 h): −9.8 pts on 417 fills (about 10 % of |price P&L| on a sleeve that is negative in that file); 8 fills paid ≥ 0.5 %. ⚠ Today's HT exit holds up to 24 h; this backtest cannot show that tail. Read HT funding at each batch review via the Closed Orders Funding % column (DL 613). **Verdict: immaterial, leave as is.**

## 5. FRENZY_WILLY (plain)

Master 24 fills −0.00 pts; the 90-min cap means no hold > 0.9 h. Backtest 545 fills −3.7 pts (−9 % of |price P&L|, old exit); 2 fills paid ≥ 0.5 %. **Immaterial.**

## 6. MOMENTUM (long + short; entries OFF)

Master 136 full-size kept fills (209 probes excluded): +0.01 pts; worst −0.03; max hold 2.65 h; 2 hourly coins (both winners). No priced backtest exists in scope (blind spot). **Immaterial** while the sleeve runs sub-3 h holds.

## 7. FLIP (FAN_RATIO_GATE; OFF)

Master 57: −0.17 pts (−3.2 % of a thin +5.2-pt sleeve). All of it is INIT 11-16, which paid 0.18 % hourly funding 96 s after opening; the DL 566 FUNDING_GUARD (settlement < 15 min against the position, |rate| ≥ 0.05 %) already blocks that case. The entry skip at paid ≥ 0.05 % blocks 5 fills, 4 of them winners (−0.03 %/fill). **Immaterial; DL 566 covers the one real case.**

## 8. Other sleeves (all OFF): FRENZY_LITE · LITE_RETRY · FRENZY_LONG · SPIKE_FADE · BULLRUN · BEARRUN · SURGE

| sleeve | N | funding pts | share of price P&L | note |
|---|---|---|---|---|
| FRENZY_LITE | 37 | +0.11 | +0.3 % | |
| FRENZY_LITE_RETRY | 20 | +0.72 | +2.2 % | one fill > 6 h (AUSDT 7.1 h, a manual close at +0.03; time stop not priced) |
| **FRENZY_LONG** | 26 | **+5.96 (received)** | **+75 %** | longs on post-pump coins RECEIVE: GTC +2.04, POWR +1.67, ERA +1.37 — the mirror image of EXH paying |
| SPIKE_FADE | 85 | +0.00 | 0 | |
| BULLRUN / BEARRUN / SURGE | 10 / 5 / 8 | 0 | 0 | |

The entry cells block winners or nothing. **Immaterial.**

## 9. Method notes

- **Three blocks.** ① current batch · ② master on today's stack (EXH / DIST: 0 fills, shown as empty) · ③ backtest DESIGN / HOLDOUT / ALL (EXH also split FRENZY / WIDE in the tables file). Master per batch is not split further: the funding-material sleeves have 0 (EXH / DIST) or 17 (FB, B24–B27) master fills.
- **Grid frozen before results** (bt.py docstring). Entry: paid ≥ X ∈ {0.05, 0.10, 0.25, 0.50, 1.00} % per settlement, or interval ≤ 1 h, each as skip or ½ size. Budget: F ∈ {0.5, 1, 1.5, 2, 3} × G ∈ {0, 2, 4}. Time: 6 / 8 / 12 h. NET TP / NET SL / both. That is 33 cells per sleeve (30 for FB).
- **NET TP mechanics.** The resting limit is re-priced at each settlement bar to E × (1 − (TP + paid)/100). It fills only ≥ 0.05 % through the line, at the line, as a maker. The extra distance shows up as fewer TP fills (EXH: 629 changed fills, of which some become time exits or stops). Received funding never moves either line (paid-only, as DL 607).
- **Exit counterfactuals on the live-stopped cohort.** Changes are split by today's outcome. Untouched fills show Δ = 0 in every block. On real fills, every changed fill is listed by name in the tables file.
- **Costs.** Every new market exit pays taker 0.05 % + slip (backtest the row's slip ≥ 0.10 %; real EXH 0.10, real FB 0.25). Sensitivity at 0.30 % slip is in the tables (it worsens every budget / time cell). Maker TP pays 0.02 %.
- **Statistics.** Day-block bootstrap (entry day, 4,000 resamples); family-wise nulls per sleeve per family; LOO by pair and by day; one-size DCR (X 0.61) with the account path shown beside it for the EXH batch.

## 10. Blind spots

1. **EXH has no master and 8 real fills.** The batch re-price is dominated by RARE #26, a knife-edge fill already used to motivate DL 609, so it is in-sample.
2. **The backtest entry rate is the last SETTLED rate** (lag-safe). Live stamps the premiumIndex estimate. They agree in sign on all 8 stamped real EXH fills (ENSO has no live stamp), and in size except KERNEL.
3. **Funding is mark-scaled by the 1m / 5m bar open** at each settlement, not the true mark (parity ≤ 0.01 pts on the backtest; exact on real fills).
4. **WILLY / HT backtest uses the old exits** (≤ 1.5 h holds); today's HT 24 h tail is not covered. Momentum, flips and the other OFF sleeves have no funding-priced backtest.
5. **NET TP / NET SL were not walked for the short-hold sleeves.** It is immaterial by exposure (≤ 0.18 % paid on any fill), but it is not measured.
6. **FB backtest is on 5m bars**; NET TP's positive read there is UNVERIFIED and confounded with "a higher TP is better for FB" (FB_TP6_SHADOW).
7. **Entry rules use one funding snapshot.** Coins that switch to hourly mid-hold (KERNEL 11-20) are counted by their interval at entry.
8. **Trade-level permutation for the entry null** is lenient for window-clustered variables; it cannot turn a null verdict into a pass.
9. **One size 0.61 × equity, with no shared-book slot cap**: the DCR / drawdown are one-size reads, not account forecasts.
10. **The live-mode slippage tail** (0.3–0.6 % on thin coins) is only covered by the 0.30 % sensitivity.

## 11. Verdict in plain words (for the operator)

- **"Shouldn't we do something about the 19 h funding?"** It was already done this afternoon: TP 7 (DL 609). ENSO reached +7 % our way 2.5 h after entry (its best was +9.7 %). Under TP 7 it would have closed then at about +6.2 % net, after paying 0.75 % funding instead of 4.36 %. It stayed open because it was opened under TP 10, and by the time the deploy moved its target to 7 the price had pulled back to +3.8 %.
- **Funding is a real cost** on EXHAUSTION (about 20 % of its price edge in the backtest) and FLATBASE (about 15 %). On every other sleeve it is noise, because they hold minutes, not hours.
- **Every rule that reacts to funding loses or ties**: skip or half-size on bad funding, exit when funding passes a budget, exit after 6 / 8 / 12 h, or make the TP / stop count funding. The reason is the same everywhere. The coins where shorts pay the most funding are the coins where too many people are already short a pumped coin, and those are exactly the coins that fall. Paying funding is the price of being in the good trades.
- **No change to any sleeve.** FLATBASE's "stop counts funding" (DL 607) re-reads as correct on today's stack. FLATBASE's "TP counts funding" looks mildly positive, but it is really "a higher TP for FLATBASE", which the existing TP-6 shadow line already tests. The one lever left is size (EXH / DIST / FB at 0.61 × equity, above Kelly), already on the pre-live list.
