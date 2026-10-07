# Momentum-long COOLDOWN study — 2026-10-07

## 0. Pre-registration (written 2026-10-07 21:14 UTC, BEFORE any outcome split was read)

Trigger: 10-07 ARB 16:31:56 / UNI 16:36:04 / LIT 17:13:13 momentum longs, all stopped DOA (−0.69/−0.70/−0.69 %).

Family (frozen):
- **Rule A (operator, PRIMARY T = 30 min):** after ANY momentum-LONG bot fill (non-MANUAL, non-probe) OPENS, refuse new momentum-LONG fills for T min. T ∈ {15, 30, 45, 60}.
- **Rule B (secondary):** refuse only while the previous momentum-long fill is still OPEN and opened ≤ T min ago ("one at a time").
- **Rule C (secondary):** refuse for T min after a momentum-long STOP-LOSS close.
- Interaction: overlap with the live LONG_CHOP_BURST block (eff72 ≤ 0.007 ∧ prior bot fill ≤ 120 s).
- Units: a refused fill is a 2nd+ fill in a cluster; evidence counted in WINDOW units (cluster / day), with per-window and per-pair concentration checks.
- Bar: CLAUDE.md EXPECTANCY filter bar on the blocked cohort at 1× — WR < sleeve breakeven WR; P(avg<0) ≥ 95 % by window-clustered bootstrap; ≥ 8 windows; no window/pair ≥ 50 % of the loss; N ≥ 15 — required in BOTH sources (master live fills AND trimmed yr5 replay). Then 30–50 % haircut.
- Approximation (pre-declared): removing a blocked fill does not free a slot or enable another fill.

(Exploratory, NOT pre-registered, added after reading results because the brief named it as a possible lever: a cluster-size cap — refuse when ≥ K accepted momentum longs opened in the last W min; 4 variants tried. Treated as hypothesis-generating only.)

---

## 1. Plain-English summary (read this first)

**The operator's cooldown (Rule A, "no new momentum long for 30 min after one opens") does NOT pass — in either data source.**

- **Live fills (master, 130 trades):** the trades a 30-min cooldown would have refused are still *winners on average*: 28 trades, 71 % win rate, +0.04 % each, +$232 in total. The sleeve needs ≈ 61.5 % wins to break even, so these refused trades are above break-even. Blocking them would have **cost about $232** (≈ −$2.5/day). On 10-07 it would have saved only UNI (−$154); LIT came 41 min after ARB.
- **Year replay (yr5, 1,784 fills, 3 seeds):** the refused trades lose (−0.084 %), but the trades it keeps lose almost the same (−0.062 %). The difference (−0.02 %) is noise (66 % of random resamples). In the replay the *whole* sleeve loses money, so blocking *any* random 24 % of trades "saves" money — that is not evidence for a cooldown.
- **Fills close in time are correlated, not worse.** When one trade loses, a second opened within 30 min loses more often (replay 62 % vs 27 % after a winner; live 50 % vs 22 %). That is the same market move being bought twice. But averaged over winners and losers, second fills earn about the same as isolated ones — so a blanket cooldown throws away as much good as bad.
- **The one idea that points the same way in both sources: "after a STOP-OUT, wait 30 min" (Rule C, pre-registered secondary).** Replay: the 52 refused fills lose −0.27 % vs −0.06 % for the rest (difference −0.21 %, 98 % confidence). Live: only 7 such trades (57 % wins, −0.08 % vs +0.21 % for the rest). That would have refused LIT on 10-07 (opened 29.5 min after UNI's stop), not UNI. But live N = 7 is far below the bar (N ≥ 15, ≥ 8 windows), and in the replay the effect is almost all in Jan–Mar (H2 May–Oct: −0.08 vs −0.04, a near tie). **Recommendation: an OBSERVE-ONLY tracker for Rule C at 30 min, frozen as written below. Arm nothing now.**
- **What 10-07 actually was:** one market window. BTC made a small push (5m RSI 61–65, 1h slope +0.09) on thin market volume (0.44–0.51, the volume-gate rescue path), and three UNMATCHED 1.5× longs (ARB, UNI, LIT) all bought the top of the same push and stopped out. It was one bad window, not three independent losses. Per the window-units rule it counts as **one** observation.
- **Is sizing the lever instead?** No. Cutting clustered fills to 1× instead of 1.5× would have cost money on live fills (Σ −2.9 %·size units for A30), because those fills win on average. In the replay the change is only "positive" because every trade there is negative.

---

## 2. Data and gate

- `scripts/validate_against_master.py` → **ALL CHECKS PASS** (2026-10-07 21:15 UTC, incl. CB1 LONG_CHOP_BURST parity, Y1, X1).
- **MASTER:** `reports/MASTER_POOL_stacked.csv` (STACK 2026-10-06d), MOMENTUM (or empty) · LONG · CLOSED · non-probe · non-MANUAL · `stack_keep` → 127 fills; dedup key (opened_at, pair, direction) has 0 duplicates. 1× = `stack_pct`, as-sized = `stack_pnl` (today's sizing). **+3 B18 fills added from `~/Downloads/scalpars_orders_paper_2026-10-07_21-11-41.csv`** (ARB/UNI/LIT; not in master, which ends 2026-10-06 12:26; treated as kept, priced as traded at the 1.5× UNMATCHED cell = today's size). Total **130**. Sleeve: 75.4 % WR, +0.191 %; avg win +0.530 / avg loss −0.845 → **breakeven WR 61.5 %**.
- **BACKTEST:** yr5 engine replay (config 181131e, Oct-4) via `scripts/yr5_fills_trimmed.py` (warm-up duplicates trimmed), MOM-long, 3 seeds → **1,784 fills** (616/586/582). Sleeve 61.4 % WR, −0.067 %; avg win +0.403 / avg loss −0.816 → **breakeven 66.9 %**. Max-open-positions already shapes the clusters. $ = `fixed_book_usd` ($3k fixed book, replay cell sizing).
- Sequencing: per source (per seed in yr5), bot-wide momentum-long sequence by open time. Rules simulated **sequentially** (a refused fill does not start a new cooldown). Window unit: master = 60-min chains of momentum-long fills; yr5 = UTC day (the same tape across seeds = one window).

## 3. Descriptive — P&L by gap since the previous momentum-long OPEN (1×)

| gap | MASTER N · WR · avg % · $ | YR5 N · WR · avg % |
|---|---|---|
| 0–5 min | 17 · 65 % · +0.012 · +$169 | 284 · 56 % · −0.092 |
| 5–15 | 7 · 86 % · +0.037 · −$8 | 92 · 67 % · −0.013 |
| 15–30 | 5 · 80 % · +0.163 · +$78 | 58 · 60 % · −0.114 |
| 30–60 | 8 · 75 % · +0.115 · +$163 | 128 · 48 % · **−0.272** |
| 60+ same day | 32 · 69 % · +0.189 · +$933 | 581 · 61 % · −0.067 |
| first of day | 61 · 80 % · +0.272 · +$2,292 | 641 · 66 % · −0.018 |
| **gap < 30** | **29 · 72 % · +0.044 · +$238** | **434 · 59 % · −0.078** |
| gap ≥ 30 / first | 101 · 76 % · +0.233 · +$3,389 | 1,350 · 62 % · −0.063 |

- Master: fills within 30 min are worse (+0.04 vs +0.23) but still above break-even (72 % > 61.5 %) and net positive.
- yr5: **not monotonic** — the worst bucket is 30–60 min, which a 30-min open-cooldown does not touch. That points at "after the previous trade *stopped out*" (stops close ~10–30 min after open), not "after it opened".
- Cluster position (60-min chains): master 1st 93·76 %·+0.244 / 2nd 26·77 %·+0.156 / 3rd+ 11·64 %·−0.169; yr5 1st −0.041 / 2nd −0.115 / 3rd+ −0.143.

## 4. Cluster outcome correlation (fill vs the fill just before it, gap < 30 min)

| | MASTER P(this loses) · avg | YR5 P(this loses) · avg |
|---|---|---|
| previous LOST | 50 % · −0.103 (N 6) | 62 % · −0.336 (N 172) |
| previous WON | 22 % · +0.082 (N 23) | 27 % · +0.091 (N 262) |
| control (gap ≥ 30): prev lost / won | 36 % / 20 % | 37 % / 38 % |

Outcomes within 30 min are strongly **correlated** (same move). The average 2nd fill is only slightly worse. A cooldown removes the correlated winners with the correlated losers.

## 5. Counterfactuals — blocked cohort vs the EXPECTANCY bar (1×)

Bar: N ≥ 15 · WR < breakeven · P(avg<0) ≥ 0.95 (window bootstrap) · ≥ 8 windows · no window/pair ≥ 50 % of the loss. "diff" = blocked avg − kept avg with a window-clustered 95 % CI.

| rule | MASTER blocked N · WR · avg · $ | diff [CI] | bar | YR5 blocked N · WR · avg | diff [CI] · P(diff<0) | bar* |
|---|---|---|---|---|---|---|
| A15 | 23 · 70 % · −0.005 · +$41 | −0.24 [−0.53, +0.04] | fail | 371 · 59 % · −0.075 | −0.01 [−0.11, +0.10] · 0.57 | fail |
| **A30 (primary)** | **28 · 71 % · +0.043 · +$232** | −0.19 [−0.46, +0.08] | **fail** (WR > 61.5 %, P 0.35) | **426 · 58 % · −0.084** | **−0.02 [−0.12, +0.07] · 0.69** | "pass"* |
| A45 | 33 · 73 % · +0.058 · +$348 | −0.18 [−0.44, +0.08] | fail | 510 · 58 % · −0.093 | −0.04 [−0.13, +0.06] · 0.79 | "pass"* |
| A60 | 37 · 73 % · +0.059 · +$401 | −0.18 [−0.43, +0.05] | fail | 543 · 57 % · −0.108 | −0.06 [−0.15, +0.03] · 0.90 | "pass"* |
| B30 (while open) | 24 · 71 % · −0.009 · +$17 | — | fail | 370 · 59 % · −0.065 | ≈ 0 | fail |
| C15 (after stop) | 3 · 67 % · −0.063 · −$5 | — | fail | 26 · 50 % · −0.147 | −0.08 [−0.33, +0.24] | fail |
| **C30 (after stop)** | **7 · 57 % · −0.077 · −$69** (5 windows) | −0.28 [−0.86, +0.14] | **fail (N, windows, P 0.65)** | **52 · 42 % · −0.272** (26 days) | **−0.21 [−0.39, −0.01] · 0.98** | pass |
| C45 | 8 · 62 % · +0.056 · +$29 | — | fail | 71 · 49 % · −0.189 | −0.13 [−0.34, +0.09] · 0.88 | pass* |
| C60 | 11 · 73 % · +0.096 · +$32 | — | fail | 89 · 52 % · −0.193 | −0.13 [−0.32, +0.07] · 0.91 | pass* |

\* **The yr5 bar is degenerate.** The whole yr5 sleeve is below its own breakeven (61 % vs 67 %, avg −0.067), so any large random subset "passes". Only the relative diff is informative. Rule A's diff is noise at every T. Rule C30 is the only rule whose blocked cohort is clearly worse than the rest.

Δ if blocked fills are removed (pre-declared approximation: a refused fill frees no slot and enables no other fill):
- MASTER: A30 **−$232 (−$2.5/day over 93 live days)**; A60 −$401; C30 +$69 (+$0.7/day).
- YR5 ($3k book, per seed, 273 days): A30 +$3,092 (+$11/day), but it is the "negative sleeve" effect (diff −0.02). C30 +$1,284/seed (+$4.7/day) → **after the 30–50 % haircut +$640…$900/seed/yr**.
- Sizing lever (de-size refused fills to 1× instead of blocking): master A30 −2.9 %·size units (costs money); yr5 +18 (only because every fill there loses).

## 6. Regime and time splits

**Rule A30, blocked vs kept avg %:**
- MASTER: BTC 1h slope > 0: 18 · 72 % · +0.000 vs +0.285 · slope ≤ 0: 10 · +0.121 vs +0.139 · BTC 5m RSI ≥ 60: 24 · +0.034 vs +0.247 · gvol < 0.7 (rescue): 7 · 57 % · −0.083 vs +0.394 · eff72 ≤ 0.007: 2 fills. Halves: H1 (< 08-19) 10 · 90 % · +0.215; H2 18 · 61 % · −0.052 vs kept +0.067. Months: Aug −0.117 (8), Sep +0.049 (8), Oct −0.006 (4).
- YR5: slope > 0: −0.070 vs −0.011 · slope ≤ 0: −0.102 vs −0.132 (blocked is *better*) · RSI ≥ 60: −0.068 vs −0.082 (better) · gvol < 0.7: −0.008 vs −0.080 (better; opposite to master) · chop: −0.289 vs −0.127 (64 fills, 18 days, already the LONG_CHOP_BURST area). Months alternate sign (Mar/May/Jul blocked better, Feb/Aug/Sep worse). Seeds: −0.090/−0.065/−0.098 vs −0.065/−0.052/−0.067.
- No regime slice is consistent across both sources. The live "BTC push + thin volume" signature (gvol < 0.7) reverses in the replay.

**Rule C30:**
- YR5: holds at both slope signs (−0.270 / −0.273), at RSI ≥ 60 (−0.322, 41 fills), at gvol < 0.7 (−0.495, 23 fills, 11 days); all 3 seeds −0.21…−0.31. **Time: H1 Jan–Apr 34 · 29 % · −0.375 vs H2 May–Oct 18 · 67 % · −0.078 (kept −0.042).** The effect is concentrated in Jan–Mar; zero C30 fills in May, Aug or Oct.
- MASTER: 7 fills: AGLD/JTO 07-13 (winners), XLM 09-23 (stop), ADA 09-23 (W), INJ 09-29 (W), SAND 10-06 (−0.56), LIT 10-07 (stop).

**LONG_CHOP_BURST overlap:** master A30 refused set has 0 fills with eff72 ≤ 0.007 ∧ gap ≤ 2 min (the 2 chop-burst refusals, LIT 07-10 / WLD 10-01, are already out of the kept pool). yr5 (chop-burst not in that config): 39 of 426 A30 fills and 4 of 52 C30 fills sit in chop-burst territory. The rules hardly overlap.

## 7. In-sample honesty and what could NOT be tested

- Haircut 30–50 % applied to the only positive estimate (yr5 C30). Master has no positive estimate for any Rule A.
- **Multiple testing:** 12 pre-registered cells (3 rules × 4 T) plus 4 exploratory cap variants. One survivor at 98 % one-sided out of ~16 tests is close to what luck alone would produce.
- **No re-run of the replay with a cooldown.** The engine has no such setting, and adding one means editing `services/` (out of scope). So slot effects are not modelled. In live, a refused fill frees one of the 4 slots for another signal, maybe a sleeve fill.
- The yr5 config predates 10-05/10-06 sizing (UNMATCHED 2× then; $ uses replay sizing) and LONG_CHOP_BURST.
- yr5 seeds share one tape, so 3 seeds ≈ 1 path in window units (C30 = 26 distinct days, ~17 fills/seed).
- Master: fills blocked by today's stack are invisible as cooldown triggers (they would not exist today). Cross-sleeve triggers (FRENZY/BullRun longs) were out of scope by the rule definition.
- Live stop-out timing: `closed_at` is the actual close; the replay stop fills are polling-based. Not checked against the Sep-23 live-stopped-cohort rule, because this is an entry rule, not a stop CF.

## 8. Verdict

- **Rule A (operator's cooldown), all T: does NOT meet the bar in either source.** Live: the refused trades are net winners above break-even (A30 costs ≈ $232). Replay: the refused trades are no worse than the kept ones. **Do not ship.**
- **Rule B ("one at a time"): no.** Neutral in both sources.
- **Rule C (30 min after a momentum-long stop-out): NEAR.** Replay passes clearly (−0.21 % diff, 98 %) but mostly from Jan–Mar. Live is direction-consistent but N = 7 / 5 windows. → **OBSERVE-ONLY scout tracker, frozen rule:**
  - *Signature (frozen):* a MOMENTUM LONG bot fill (non-probe, non-MANUAL) opened < 30.0 min after any momentum-long fill closed with a STOP_LOSS* reason.
  - *Tally at each batch review* in window units (60-min chains), at 1× and as-sized.
  - *Arm only when* the master cohort meets the EXPECTANCY bar: N ≥ 15 · ≥ 8 windows · WR < sleeve breakeven · window-bootstrap P(avg<0) ≥ 95 % · no window/pair ≥ 50 % of the loss. Then a 30–50 % haircut and a revert gate, pre-committed now: **revert if the first 10 refused signals (re-priced with the live exit replica) show WR ≥ 61 % OR Σ > 0.**
  - No threshold re-fit (no 25 or 35 min) on this data.
- Exploratory (not pre-registered): "≥ 2 momentum longs already opened in the last 120 min" → master 15 · 60 % · −0.059 (diff −0.28, P 0.91), yr5 194 · 54 % · −0.157 (diff −0.10, P 0.96, both halves). Watchlist only: the window was picked after seeing results. Re-test fresh if Rule C's tracker stalls. K=2 W=60 was strong live (diff −0.39) and null in yr5.
- **10-07:** one correlated window (BTC mini-push on thin volume, three UNMATCHED 1.5× longs on the same move). A30 would have saved UNI (−$154), C30 would have saved LIT (−$137), and A45/A60 would have saved both. None of these rules earns its keep across the history. Neither blocking nor de-sizing clustered fills is supported, because live clustered fills are net winners.

Scripts (scratch): `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/ml_cooldown/{prep,analyze,analyze2,analyze3}.py`; raw outputs `out.txt`, `out2.txt`, `out3.txt`; tagged fills `master_tagged.csv`, `yr5_tagged.csv`.
