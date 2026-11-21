# DIST_THEORY_EXIT — does an exit built from what DISTRIBUTION_SHORT trades beat today's exit? (2026-11-21, BACKTEST + 6 real fills)

**Verdict: keep today's exit (Exit B: TP min(20 × ATR15, 10 %) resting maker · stop min(1.5 × ATR15, 6 %) · 24 h).**
41 theory-derived exits in 9 families. **0 of 9 family picks pass** the pre-registered HOLDOUT gates. No pick even has a HOLDOUT CI that
excludes zero on the positive side. The structure stop (at the distribution candle's high or the pump top) behaves like a plain
wider stop, and a plain wider stop saves about as much as it adds in deeper losses. The 6 real fills cannot decide anything (N = 6).

Pre-registration: `reports/DIST_THEORY_EXIT_2026-11-21_PREREG.md` (frozen before any cell was simulated).
Scripts: `scripts/DIST_THEORY_EXIT_2026-11-21_{sim,eval,real,extras}.py`. Data: `reports/DIST_THEORY_EXIT_2026-11-21_data/`
(eval.csv · picks.csv · halves.csv · stopcohort.csv · loo.csv · perfill.csv.gz · real_fills.csv · real_summary.csv · eval.log).
Chart: `reports/DIST_THEORY_EXIT_2026-11-21.png`. Every Δ is BACKTEST; any batch-derived number takes a 30–50 % in-sample haircut
(moot: nothing passes).

---

## 1. Question
The operator asked for a solid exit per sleeve. DIST's real fills so far are 1 W / 5 L. Can an exit derived from **why** DIST enters beat
today's exit, after costs, on data it was not tuned on?

## 2. Theory of the trade → the exits it predicts
DIST shorts a Frenzy coin (≤ 72 h after its spike) on a **closed red 15m candle with ≥ 5 × normal volume**, while top traders cut their
longs and the 1h RSI is still ≥ 74. Reading: a big holder is **selling into the pump**. If that is right:

| theory | exit family tested | what it does |
|---|---|---|
| The red candle is the supply level; price back above its high = supply absorbed | **C** stop at the candle high (DH) or pump top (PK), ×1.00 / ×1.01 / tighter-of · **R** same but on a CLOSED 15m bar (wicks ignored) | structure stop |
| Buyers absorbing the supply end the trade | **G** exit on a green 15m candle with ≥ 0.5 / 1 × the distribution candle's volume | absorption exit |
| The drop unwinds the pump to its landmarks | **A** / **N** TP at the episode VWAP, 38 / 50 / 62 % retrace, pre-pump base (replace / nearer) · **M** TP at 1 / 1.5 / 2 × the candle's range | target |
| Distribution is one leg | **B** after +3/5/7 %, no new low for 60/120/240 min, in profit → exit | stall exit |
| Crowded shorts pay funding | **F** exit once 1 % / 2 % of notional funding is paid | funding budget |
| Control | **W** fixed stop 8 / 10 / 12 / 15 %, today's TP | "is it just a wider stop?" |

Landmarks at entry (no outcomes): the candle high sits a median **6.1 %** above the signal close (P10 2.6 %, P90 16 %), the pump top
7.0 %; today's stop median is **4.7 %**. So most structure stops are **wider** than today's — hence the W control.

## 3. Method
- **Cohort:** the 1,547 DIST_EXIT REF_B backtest fills (DL 586 entry incl. RSI1h ≥ 74.07, one position per pair; entry set held fixed).
- **Engine = DIST_EXIT's:** 1m path; stop checked first on the bar high (gap-aware fill); maker TP only when the low trades ≥ 1 tick
  through; both in one bar → stop; time exit at 24 h; rule exits = market at the next minute's open. Costs: entry slip + 0.05 taker, exit
  0.05 taker + slip (model, floor 0.10) or 0.02 maker; real funding.
- **Parity:** my REF reproduces DIST_EXIT REF_B (exit kind and gross within 0.01 %) on **1,546 / 1,547 = 99.94 %** (gate ≥ 99.9 % ✓).
  The one miss (MIRA 2026-03-01) is a TP-vs-time difference from a 1m-data source; mean net +0.931 vs +0.932.
- **Split:** DESIGN < 2025-07 (559 fills) picks one cell per family; HOLDOUT ≥ 2025-07 (988) is read once against 7 gates:
  ① Δ > 0 with 95 % day-block CI > 0 · ② Δ > 0 in ≥ 2/3 half-years · ③ > 0 at 0.20 % slip · ④ no pair / day ≥ 50 % of Δ · ⑤ saved $ >
  cut $ · ⑥ one-size DCR after ≥ before · ⑦ (C / R only) beats the matching fixed-width W control.
- $ at one size = $3,050 notional per fill; DCR at 0.61 × equity per fill over every calendar day.

## 4. Today's exit (REF) — the bar to beat

| block | N | WR | avg % / trade | avg loss | $ at one size | DCR %/day | max DD | worst day |
|---|---|---|---|---|---|---|---|---|
| DESIGN | 559 | 42.0 % | **+1.03** | −4.13 | +$17,497 | 0.41 | −34.5 % | −19.2 % |
| HOLDOUT | 988 | 42.7 % | **+0.88** | −5.26 | +$26,441 | 0.85 | −54.9 % | −21.8 % |
| ALL | 1,547 | 42.5 % | **+0.93** | −4.85 | +$43,938 | 0.59 | −54.9 % | −21.8 % |

Positive in 7 / 7 half-years (+2.71 · +0.62 · +0.63 · +0.76 · +0.71 · +0.54 · +1.63).

## 5. The picks on HOLDOUT (read once)

| family | DESIGN pick (Δ) | HOLDOUT Δ % / trade [95 % CI] | halves + | Δ @0.20 slip | saved $ / cut $ | DCR before → after | max DD after | worst day after | gates ①–⑦ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A target replace | A_BASE (+0.09) | −0.18 [−0.55, +0.26] | 1/3 | −0.19 | 5,892 / 49,889 | 0.85 → 0.53 | −69.7 | −23.3 | ✗✗✗✗✗✗ | FAIL |
| N target nearer | N_BASE (−0.08) | −0.10 [−0.23, +0.03] | 0/3 | −0.09 | 5,892 / 9,762 | 0.85 → 0.76 | −53.2 | −21.8 | ✗✗✗✗✗✗ | FAIL |
| M measured move | M_2 (+0.03) | −0.40 [−0.77, +0.02] | 0/3 | −0.42 | 485 / 55,750 | 0.85 → 0.25 | −86.3 | −23.3 | ✗✗✗✗✗✗ | FAIL |
| B stall | B_X7_N240 (−0.15) | +0.02 [−0.11, +0.14] | 2/3 | +0.01 | 6,077 / 6,311 | 0.85 → 0.89 | −47.3 | −18.4 | ✗✓✓✗✗✓ | FAIL (pair 89 %, day 85 % of Δ) |
| C structure stop | C_DHT (+0.03) | −0.03 [−0.12, +0.05] | 1/3 | −0.03 | 3,188 / 4,016 | 0.85 → 0.82 | −55.0 | −20.7 | ✗✗✗✗✗✗✓ | FAIL |
| R closed-bar structure | R_DH (−0.07) | −0.04 [−0.42, +0.34] | 2/3 | −0.03 | 60,968 / 744 * | 0.85 → 0.62 | −84.1 | −48.6 | ✗✓✗✗✓✗✓ | FAIL |
| G absorption | G_ABS100L (+0.00) | −0.00 [−0.05, +0.03] | 2/3 | −0.00 | 542 / 680 | 0.85 → 0.85 | −54.9 | −21.8 | ✗✓✗✗✗✗ | FAIL (fires on 9 fills) |
| F funding budget | F_P1N (−0.00) | −0.02 [−0.07, +0.02] | 0/3 | −0.02 | 711 / 1,391 | 0.85 → 0.83 | −55.8 | −21.8 | ✗✗✗✗✗✗ | FAIL |
| W control (fixed stop) | W_SL12 (+0.16) | +0.07 [−0.35, +0.52] | 3/3 | +0.08 | 76,127 / 0 * | 0.85 → 0.74 | −83.4 | −31.8 | ✗✓✓✗✓✗ | FAIL (Δ −0.00 without its best day) |

\* For stop families "cut" counts only REF winners made worse; the honest split is on the REF-stopped cohort (§6), where the deeper
losses cancel the saves.

Every cell on HOLDOUT (descriptive, after the decision) is in the right-hand panel of the chart. The best HOLDOUT cells (W_SL15 +0.21,
W_SL12 +0.07, A_R382 +0.06) were **not** the DESIGN picks except W_SL12, all have CIs straddling 0, and A_R382 was −0.40 on DESIGN
(lost in 23-H2 … 25-H2, won only in 26-H1 / 26-H2). Without their best 5 % of fills every one of them is −0.49 … −0.83 (loo.csv).

## 6. The structure stop is "just a wider stop" (stop-cohort read, two-sided)
On the 561 HOLDOUT fills today's stop actually stopped (fills it did not stop are untouched by any wider stop — asserted 0 change for W):

| cell | median stop | saved (better) n · $ | deeper (worse) n · $ | non-stopped fills changed · $ | net $ HOLDOUT | net $ ALL |
|---|---|---|---|---|---|---|
| C_DH0 (candle high) | 6.7 % | 227 · +42,923 | 323 · −37,762 | 15 · −5,288 | **−130** | −1,227 |
| C_DH1 | 7.9 % | 177 · +46,647 | 377 · −45,672 | 5 · −1,971 | −997 | −555 |
| C_PK0 (pump top) | 7.4 % | 207 · +44,384 | 347 · −42,288 | 12 · −4,204 | −2,111 | −3,435 |
| R_DH (closed 15m bar > DH) | (15 % hard) | 183 · +60,966 | 370 · −61,438 | 2 · −744 | −1,215 | −2,461 |
| W_SL8 | 8 % | 105 · +38,901 | 456 · −41,763 | 0 | −2,861 | −3,777 |
| W_SL10 | 10 % | 163 · +60,625 | 398 · −59,873 | 0 | +751 | +179 |
| W_SL12 | 12 % | 206 · +76,127 | 355 · −74,071 | 0 | +2,056 | +4,749 |
| W_SL15 | 15 % | 255 · +94,929 | 306 · −88,691 | 0 | +6,238 | +7,255 |

- The structure stops land between W_SL8 and W_SL10 and score like them. There is **no extra edge from placing the stop at the candle high
  or the pump top**: the market does not respect DH as a line (EXH's pump-top stop was +0.20 HOLDOUT; DIST's is −0.07).
- The wider stops roughly break even on $ but **raise the average loss from −5.3 % to −7 … −13.5 %** and the max drawdown from −55 % to
  −79 … −85 %; DCR falls in every one (0.85 → 0.62 … 0.83). This repeats DIST_FB_LOSERS §6.1 (ATR widths) with fixed widths.
- The chart's left panel shows why: the median stopped trade drifts back to about +1 % by 24 h, but with a wide band — about as many
  keep going against us as come back.

## 7. Real fills (6 closed DISTRIBUTION_SHORT, all exports, dedup, newest kept)
REF re-walk vs booked: same exit kind on 6 / 6; nets −4.40 / −6.30 / −4.88 / +9.83 / −6.30 / −6.30 vs booked −4.41 / −5.97 / −4.38 /
+9.95 / −6.11 / −6.03 (model slip 0.10 vs real stop slip and booked funding).

| fill | booked % | REF | C_DH0 | C_PK0 | R_DH | W_SL10 | W_SL15 | A_R382 | DH above entry |
|---|---|---|---|---|---|---|---|---|---|
| SENT 11-19 04:15 | −4.41 | −4.40 | +7.62 | +7.62 | +7.62 | −10.30 | +7.62 | −4.40 | 12.5 % |
| NAORIS 11-19 09:15 | −5.97 | −6.30 | −7.31 | −7.31 | −15.30 | −10.30 | −15.30 | −6.30 | 7.0 % |
| EVAA 11-19 14:31 | −4.38 | −4.88 | −4.90 | −4.90 | −4.96 | −10.09 | −15.03 | −4.88 | 4.7 % |
| CYBER 11-20 09:00 | +9.95 | +9.83 | −2.45 | −2.98 | +9.83 | +9.83 | +9.83 | +7.97 | 2.2 % |
| KERNEL 11-20 12:30 | −6.11 | −6.30 | −7.34 | −7.34 | −9.86 | −10.30 | −15.30 | +1.42 | 7.0 % |
| BLESS 11-21 09:30 | −6.03 | −6.30 | +9.83 | +9.83 | +9.83 | +9.83 | +9.83 | −6.30 | 36.6 % (stop capped 15 %) |
| **avg / $ at one size** | | **−3.06 / −$560** | −0.76 / −$139 | −0.85 / −$155 | −0.47 / −$87 | −3.56 / −$651 | −3.06 / −$560 | −2.08 / −$381 | |

- N = 6: the candle-high stop would have done better here (+$421), driven by two fills (SENT, BLESS) whose candle high was far above the
  stop; it would also have turned CYBER, the only winner, into a loss. C_DH0's mean Δ is +2.30 per fill; leaving out a single fill it can fall to −0.47.
  Six fills in three days are about three market windows: they **cannot** outweigh 988 HOLDOUT fills, and they do not refute the verdict.
- Stall, funding and absorption exits never fire on these six (all stopped or TP'd before any trigger).

## 8. What it means
- DIST's edge is the large follow-through after a distribution candle (Exit B's TP winners average about +9 %). Every theory target
  (VWAP, retraces, base, measured move) sells part of that tail earlier and loses more on the winners than it saves on the losers — the
  same wall DIST_EXIT hit with fixed TPs, trails and break-evens.
- The theory's best idea — "the thesis is only wrong above the distribution candle" — does not hold as a price line: a stop there is
  statistically the same as a wider fixed stop, and wider stops trade smaller frequent losses for bigger rarer ones without adding money.
- Absorption and funding-budget exits almost never fire before the stop or TP (≤ 2 % of fills), so they cannot matter.
- Multiple testing: 9 picks were read; with nothing passing, there is no luck question to answer.

## 9. Recommendation (plain English)
**Keep DISTRIBUTION_SHORT's exit exactly as it is**: take profit at about 10 % with a resting order, stop at 1.5 × the 15-minute ATR
(max 6 %), close after 24 hours. We tried 41 exits built from what this trade is betting on — profit targets at the pump's own price
levels, a stop above the selling candle or the pump top, exiting when buyers soak up the selling, exiting when the move stalls, exiting
when funding gets expensive, and plain wider stops. None of them made more money on the 16 months of data they were not tuned on, and the
wider-stop versions make each loss much bigger. The recent losing streak (5 of 6) is not an exit problem we can fix with any of these
rules; the levers left are **size** (already on the batch-review list with the TP × size question, DL 603) and the Route B squeeze data
(DL 602, ≈ 12-10).

No watch line is proposed: no cell came close enough to justify one (the existing DIST_TP_SHADOW TP 5 / 8 line stays as it is).

## 10. Blind spots and caveats
- Entry set fixed at today's one-position cohort: an earlier exit could free the coin for a later DIST signal (not modelled).
- Same-bar stop-first is pessimistic for TP-near-stop bars; 1 of 1,547 fills on 5m fallback.
- Backtest slip is the model (floor 0.10); live thin-coin stop slip ≈ 0.22 % (master proxy) would make every wider / later market exit
  slightly worse, not better.
- Drawdowns sum every fill with no shared-book slot cap (overstates clustering for all cells equally).
- Real-fill landmarks are from 1m bars, backtest landmarks from 5m bars (same definitions).
- The event-study panel plots price after the exit too (what the market did, not what we booked).
