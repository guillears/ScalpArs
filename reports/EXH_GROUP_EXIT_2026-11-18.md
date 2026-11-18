# EXHAUSTION_SHORT — cut the shorts when the market lifts them together? (2026-11-18)

Pre-registration: `reports/EXH_GROUP_EXIT_2026-11-18_PREREG.md` (frozen 14:18 UTC, before any rule result).
Data: `reports/EXH_GROUP_EXIT_2026-11-18_cells.csv` (every cell × book × cut slippage). Chart: `reports/EXH_GROUP_EXIT_2026-11-18.png`.
Code: `reports/EXH_GROUP_EXIT_2026-11-18_code/` (build → sim → run → null → ana; E = holdout E).
Research only. Nothing in the bot was changed.

## The question

EXHAUSTION shorts lose in clusters (2024-11-16: 13 of 14 stopped; 2026-01-24: 13 of 13). The entry cannot tell those days apart.
So: can an exit that watches the open book — several shorts going against us at the same time — cut them early and save money?

## The plain answer

**Mostly no, because the cluster losses do not happen at the same time.** On the bad days the stops come one after another,
spread over about 20 hours, about 1–3 hours apart. At any one minute there are rarely more than 2 shorts underwater together.
So a "lifted together" signal almost never fires (16 days in 3.4 years for the best rule). When it fires, it helps a little in the
design years. It hurt in the holdout. It touched only 2 of the 15 worst days.

**Verdict: OBSERVE (low value), no engine build.** Best rule G2 M2·K3: +0.010 %/day on a 0.207 %/day base (design, book 4). It passes
the shuffled null (p = 0.02), but the day CI includes 0, the holdout is negative (2 events, both lost), and one neighbour is negative.

## Theory (written before the numbers)

- A short fails when new buying keeps arriving after entry. On cluster days the buying is market-wide (alt rotation), so it should lift
  every open short at once. One short against us is noise. Several shorts against us at the same minute should be a common cause.
- If that cause persists, the lifted shorts go on to the full stop. Cutting them early saves (full stop − cut level − costs).
- How it can fail: (1) Frenzy coins mean-revert hard; a cut short that would have reached the 10 % TP costs about 10 + the cut level.
  (2) The cut is a taker order into a squeeze (slippage 0.1–0.6 %). (3) The signal comes late — most of the stop distance is spent.

## Method

- **Trades:** cell B|tp10|S2|24 (TP 10 % maker, SL min(1.5×ATR, 6 %), 24 h), 4,140 candidates, 1m path for each.
- **Parity:** one position per pair, no cap → **N 3,998 · avg +0.647 % · WR 40.2 %**, and every half-year matches `eval_grid.pkl`
  exactly (23-H2 +0.826 · 24-H1 +0.664 · 24-H2 +0.749 · 25-H1 +0.168 · 25-H2 +0.756 · 26-H1 +0.538 · 26-H2 +0.791). ✅
- **Book:** entries in time order, one per pair, cap 4 and cap 10 (EXH-only). Group checks at every 1m close on the open book only.
  Natural stop / TP inside a bar comes first. A cut fills at that minute's close, taker; the slot frees from the next minute.
- **Costs:** entry and taker exits: fee 0.05 % + slippage 0.10 %. Maker TP: 0.02 %, no slippage. Funding to the exit (to the cut minute
  for cuts). Cut slippage 0.10 / 0.20 / 0.40 %. Note: all slippage here is flat, so the baseline avg is +0.48 %, not the +0.65 % of the
  depth-model slip used above for parity.
- **Rules:** 19 rules × 2 books = 38 cells (2 void: K = 5 at cap 4) — G1-ALL (≥ K shorts each ≥ L % against → close all), G1-HIT (same
  trigger, close only those ≥ L against), G2 (≥ K open and the book's mean ≤ −M % → close all), G3 (alt-perp median 1h return ≥ X % at an
  hour close, ≥ 2 shorts underwater → close the underwater ones), G4 (no new entry while G1 / G2 is active; reads the OPEN book, not
  closed losses). Control G0: cut any single short at −L (not in the family).
- **Size:** one size 0.183 × equity per trade (0.61 for reference). P&L on the exit day, every calendar day compounded.
- **Periods:** DESIGN 2023-07 → 2026-06 · HOLDOUT 2026-07 → 2026-11-18 (read once) · E 2023-H1 (237 trades, no G3 data).
- **Null:** 200 shuffles that give each entry slot another candidate's path within the same half-year (same schedule, same paths,
  co-movement destroyed). Family max over all 36 live cells. A day-block shuffle would keep the same-day co-movement the rule reads.

## Why it rarely fires — the anatomy of the 15 worst days (no cap, all trades)

| Day | trades | stops | Σ net % (1×) | max open at once | max ≥ 1.5 % against at once | median minutes between stops | hours first → last stop |
|---|---|---|---|---|---|---|---|
| 2026-01-24 | 14 | 13 | −56.6 | 5 | 2 | 78 | 22.7 |
| 2025-11-22 | 9 | 9 | −56.2 | 2 | 2 | 114 | 20.1 |
| 2024-11-16 | 16 | 14 | −51.7 | 3 | 3 | 67 | 22.9 |
| 2025-11-17 | 10 | 9 | −42.4 | 3 | 1 | 62 | 11.5 |
| 2026-03-31 | 9 | 9 | −42.1 | 3 | 2 | 84 | 13.3 |
| 2024-11-23 | 11 | 10 | −40.9 | 6 | 3 | 23 | 22.5 |
| 2026-09-25 | 13 | 11 | −40.1 | 4 | 2 | 117 | 21.8 |
| 2024-11-27 | 10 | 9 | −39.5 | 5 | 2 | 88 | 14.3 |
| 2026-04-20 | 9 | 8 | −36.5 | 2 | 2 | 70 | 18.5 |
| 2026-10-10 | 9 | 8 | −35.6 | 3 | 1 | 195 | 19.6 |
| 2026-05-07 | 14 | 10 | −33.9 | 4 | 2 | 67 | 17.6 |
| 2024-02-26 | 11 | 9 | −33.2 | 7 | 2 | 115 | 21.4 |
| 2026-07-02 | 8 | 7 | −33.0 | 3 | 2 | 156 | 21.5 |
| 2025-05-09 | 6 | 6 | −32.9 | 3 | 2 | 182 | 21.1 |
| 2025-07-24 | 8 | 7 | −32.5 | 4 | 2 | 118 | 18.1 |

Over all 3.4 years (no cap): ≥ 2 shorts each ≥ 1.5 % against at the same minute — 2,612 minutes; ≥ 3 — 124 minutes; ≥ 4 — 2 minutes.
A short lives a median 2.7 h and is often stopped before the next one opens. The "cluster" is a day of separate stops in a row, not a
book lifted at once. The correlation from the bad-day study is real, but it plays out in sequence.

## Per-rule results (DESIGN, one size 0.183; Δ vs the same book without the rule)

Baseline book 4: N 3,175 · WR 39.6 % · avg +0.435 % · **DCR 0.207 %/day** · max DD −25.9 % · worst day −10.3 % · worst 7 d −19.7 % · holdout 0.542 %/day.
Baseline book 10: N 3,325 · WR 39.6 % · avg +0.449 % · **DCR 0.223 %/day** · max DD −26.6 % · worst day −10.4 % · worst 7 d −19.7 % · holdout 0.507 %/day.

| Rule | book | cuts | DCR after | ΔDCR @0.10 | ΔDCR @0.20 | ΔDCR @0.40 | Δ daily 95 % CI (day bootstrap) | Δ max DD | Δ HOLDOUT DCR |
|---|---|---|---|---|---|---|---|---|---|
| G1-ALL K3 L1.5 | 4 | 34 | 0.212 | +0.006 | +0.005 | +0.004 | [−0.003, +0.016] | 0.0 | −0.004 |
| G1-ALL K3 L2.5 | 4 | 7 | 0.207 | +0.000 | +0.000 | −0.000 | [−0.001, +0.002] | 0.0 | 0.000 |
| G1-ALL K4 · K5 (all L) | 4 / 10 | 0 | = | 0 | 0 | 0 | — | 0 | 0 |
| G1-HIT K3 L1.5 | 4 | 30 | 0.213 | +0.006 | +0.006 | +0.005 | [−0.003, +0.017] | 0.0 | −0.004 |
| G1-HIT K3 L2.5 | 4 | 6 | 0.207 | +0.001 | +0.001 | +0.000 | [−0.001, +0.003] | 0.0 | 0.000 |
| G1-HIT K4 (all L) | 4 / 10 | 0 | = | 0 | 0 | 0 | — | 0 | 0 |
| **G2 M2 K3** | **4** | **49** | **0.216** | **+0.010** | **+0.009** | **+0.008** | **[−0.001, +0.021]** | −0.4 | **−0.029** |
| G2 M2 K4 | 4 | 4 | 0.206 | −0.001 | −0.001 | −0.001 | [−0.003, +0.000] | 0.0 | 0.000 |
| G2 M3 K3 | 4 | 3 | 0.207 | +0.000 | +0.000 | +0.000 | [+0.000, +0.001] | 0.0 | 0.000 |
| G2 M3 K4 | 4 / 10 | 0 | = | 0 | 0 | 0 | — | 0 | 0 |
| G3 X1 | 4 | 41 | 0.197 | −0.009 | −0.010 | −0.011 | [−0.025, +0.007] | 0.0 | +0.012 |
| G3 X2 | 4 | 5 | 0.206 | −0.001 | −0.001 | −0.001 | [−0.005, +0.003] | 0.0 | 0.000 |
| G4 pause (all 3) | 4 | 0 (0 skips) | = | 0 | 0 | 0 | — | 0 | 0 |
| G1-ALL K3 L1.5 | 10 | 88 | 0.229 | +0.005 | +0.004 | +0.002 | [−0.011, +0.021] | 0.0 | −0.052 |
| G1-ALL K3 L2.5 | 10 | 25 | 0.215 | −0.008 | −0.009 | −0.009 | [−0.027, +0.001] | 0.0 | +0.014 |
| G1-HIT K3 L1.5 | 10 | 57 | 0.230 | +0.007 | +0.006 | +0.005 | [−0.002, +0.017] | 0.0 | −0.033 |
| G1-HIT K3 L2.5 | 10 | 15 | 0.220 | −0.003 | −0.003 | −0.004 | [−0.009, +0.002] | 0.0 | +0.008 |
| G2 M2 K3 | 10 | 49 | 0.226 | +0.003 | +0.002 | +0.001 | [−0.008, +0.014] | −0.4 | −0.017 |
| G2 M2 K4 | 10 | 4 | 0.222 | −0.001 | −0.001 | −0.001 | [−0.003, +0.000] | 0.0 | 0.000 |
| G2 M3 K3 | 10 | 3 | 0.223 | +0.000 | +0.000 | +0.000 | [+0.000, +0.001] | 0.0 | 0.000 |
| G3 X1 | 10 | 56 | 0.216 | −0.007 | −0.008 | −0.010 | [−0.020, +0.005] | 0.0 | +0.002 |
| G3 X2 | 10 | 5 | 0.222 | −0.001 | −0.001 | −0.001 | [−0.005, +0.003] | 0.0 | 0.000 |
| G4 pause G1 K3 L1.5 | 10 | 0 (1 skip) | 0.223 | 0 | 0 | 0 | — | 0 | −0.012 |
| G4 pause (other 2) | 10 | 0 | = | 0 | 0 | 0 | — | 0 | 0 |
| *control G0 L1.5* | 4 | 3,118 | −0.007 | **−0.214** | −0.257 | −0.344 | [−0.320, −0.137] | −30.6 | −0.491 |
| *control G0 L2.5* | 4 | 2,358 | 0.006 | **−0.200** | −0.233 | −0.298 | [−0.280, −0.137] | −34.5 | −0.341 |

ΔDCR in %/day points. "0" rows: the trigger never happened (≥ 4 shorts each ≥ 1.5 % against at once: 2 minutes in 3.4 years).
The control shows what a plain tighter stop does: it destroys the sleeve (−0.20 %/day). So the group condition does add information —
it picks the few moments when a cut is roughly break-even-to-positive instead of strongly negative.

## Saved vs cost (cut trades vs their own natural outcome; % per trade at 1×, summed)

| Rule | book | cut slip | cuts | natural was stop / TP / time | SAVED n · Σ | COST n · Σ | cut net | trades added by freed slots | trades lost | total |
|---|---|---|---|---|---|---|---|---|---|---|
| G2 M2 K3 | 4 | 0.10 | 49 | 39 / 3 / 7 | 40 · +99.3 | 9 · −64.9 | +34.4 | 5 · −7.6 | 2 · +7.8 | +34.6 |
| G2 M2 K3 | 4 | 0.40 | 49 | 39 / 3 / 7 | 40 · +87.3 | 9 · −67.6 | +19.7 | 5 · −7.6 | 2 · +7.8 | +19.9 |
| G2 M2 K3 | 10 | 0.10 | 49 | 39 / 4 / 6 | 39 · +95.8 | 10 · −81.1 | +14.7 | 2 · −11.6 | 0 | +3.1 |
| G2 M2 K3 | 10 | 0.40 | 49 | 39 / 4 / 6 | 38 · +84.4 | 11 · −84.4 | 0.0 | 2 · −11.6 | 0 | −11.6 |
| G1-HIT K3 L1.5 | 4 | 0.10 | 30 | 25 / 2 / 3 | 25 · +60.9 | 5 · −39.0 | +21.9 | 3 · +1.4 | 2 · +7.8 | +31.0 |
| G1-HIT K3 L1.5 | 10 | 0.10 | 57 | 45 / 7 / 5 | 46 · +115.7 | 11 · −105.7 | +9.9 | 2 · +4.0 | 0 | +13.9 |
| G1-HIT K3 L1.5 | 10 | 0.40 | 57 | 45 / 7 / 5 | 46 · +101.9 | 11 · −109.0 | −7.2 | 2 · +4.0 | 0 | −3.2 |
| G1-ALL K3 L1.5 | 10 | 0.10 | 88 | 54 / 19 / 15 | 59 · +175.5 | 29 · −186.5 | −11.0 | 3 · +2.0 | 0 | −9.0 |
| G3 X1 | 4 | 0.10 | 41 | 25 / 6 / 10 | 26 · +71.2 | 15 · −104.3 | −33.2 | 15 · −17.4 | 5 · +7.7 | −42.8 |
| G0 L1.5 (control) | 4 | 0.10 | 3,118 | 2,330 / 587 / 201 | 2,336 · +6,199 | 782 · −8,007 | −1,808 | 249 · −31.0 | 2 · +7.1 | −1,832 |

("trades lost": baseline trades the rule's book did not take, shown as money back — these were losers, so + is good.) Every trade the rule did not
touch has Δ = 0 (asserted in code). The pattern: about 4 of 5 cuts are SAVED, but each COST is large (a short that would have
reached the 10 % TP, closed at −2 to −3 %), so 1 COST ≈ 3 SAVED. With 0.40 % slippage the edge is gone in book 10.

In window units (G2 M2 K3, book 4): it fired on **16 days** — 14 in design (10 helped, Σ +48 trade-%) and 2 in the holdout
(2026-08-21 and 2026-08-23, both hurt: Σ −13.6; on 08-21 it cut PEOPLE at −5.2 % that went on to +9.9 %). E (2023-H1): 1 day, 3 cuts,
all saved (+4.4), ΔDCR +0.005.

## Before → after per half-year (best cell G2 M2 K3, book 4, cut slip 0.10)

| Half | N | avg % | DCR %/day | ΔDCR | Δ daily 95 % CI | max DD % | worst day % | worst 7 d % |
|---|---|---|---|---|---|---|---|---|
| 23-H2 | 281 → 281 | +0.657 → +0.657 | 0.173 → 0.173 | 0.000 | [0, 0] | −11.2 → −11.2 | −4.83 → −4.83 | −7.9 → −7.9 |
| 24-H1 | 414 → 414 | +0.551 → +0.582 | 0.213 → 0.225 | +0.012 | [−0.022, +0.065] | −20.0 → −17.2 | −6.07 → −6.07 | −12.7 → −12.9 |
| 24-H2 | 382 → 382 | +0.344 → +0.355 | 0.113 → 0.118 | +0.005 | [−0.015, +0.027] | −14.6 → −14.4 | −9.46 → −7.85 | −7.5 → −7.3 |
| 25-H1 | 427 → 427 | −0.056 → −0.025 | −0.037 → −0.023 | +0.014 | [−0.012, +0.046] | −21.3 → −21.4 | −6.02 → −6.02 | −12.0 → −13.0 |
| 25-H2 | 851 → 852 | +0.645 → +0.670 | 0.498 → 0.521 | +0.023 | [−0.001, +0.054] | −19.7 → −19.7 | −10.29 → −10.29 | −19.7 → −19.7 |
| 26-H1 | 820 → 820 | +0.382 → +0.388 | 0.277 → 0.282 | +0.005 | [−0.016, +0.027] | −25.9 → −26.3 | −9.14 → −9.14 | −16.9 → −16.9 |
| **26-H2 (holdout)** | 659 → 661 | +0.678 → +0.642 | 0.542 → 0.513 | **−0.029** | [−0.079, +0.001] | −22.7 → −22.7 | −7.34 → −7.34 | −16.8 → −16.8 |

Runner-up G1-HIT K3 L1.5 (book 10): design halves +0.000 / +0.016 / +0.015 / +0.010 / +0.009 / −0.007, holdout −0.033.
At 0.61 × equity (reference): book 4 design DCR 0.499 → 0.535 %/day, max DD −76.0 → −76.3 %; holdout 1.48 → 1.39 %/day.
The worst day and worst 7 days do not change in any half: the rule does not reach the days that set the drawdown.

## Worst and best days (book 4, the 15 worst / best baseline days, G2 M2 K3)

Worst 15 days (Σ −107 % of equity at 0.183): the rule fired on **2** of them — 2024-11-16 (3 stops avoided, −9.46 → −7.85 %) and
2025-05-30 (3 stops avoided, −5.41 → −3.23 %). Total saved +3.8 % of equity. The other 13 (incl. 2026-01-24 and 2025-11-22, the two
worst) are untouched — never ≥ 3 open with a mean ≤ −2 %.
Best 15 days: no winner cut (Δ 0.00). The COST cuts happen on ordinary days.

## Family null and plateau

- Shuffled null (200 path shuffles, max over 36 cells): observed best +0.0099 %/day; null median +0.0032, 95th pct +0.0081 →
  **family p = 0.02**. The best cell alone beats every shuffle (cell p < 0.005). So the timing information is real, not luck.
- Plateau around G2 M2 K3 (book 4): G2 M2 K4 −0.001 (4 cuts), G2 M3 K3 +0.000 (3 cuts), same rule book 10 +0.003. The neighbours
  barely fire, so they neither confirm nor kill it; formally the plateau fails (one negative neighbour).

## Pre-registered bar for G2 M2 K3

| Test | Result | Pass |
|---|---|---|
| DESIGN ΔDCR > 0 in both books at cut slip 0.20 | +0.009 (book 4) · +0.002 (book 10) | ✅ |
| ≥ 5 of 6 design halves Δ ≥ 0 | 6 / 6 (23-H2 = 0) | ✅ |
| Day-block CI of design Δ excludes 0 | [−0.001, +0.021] | ❌ |
| Family null p < 0.05 | 0.02 | ✅ |
| Plateau | one neighbour −0.001, others ≈ 0 | ❌ |
| HOLDOUT Δ ≥ 0 | −0.029 (2 events, both lost) | ❌ |
| Beats control G0 at the same level | +0.010 vs −0.20 | ✅ |
| SAVED $ > COST $ | +99.3 vs −64.9 | ✅ |

→ **OBSERVE** (design positive and SAVED > COST, but CI, plateau and holdout fail).

## What it means

1. The exit route is closed for the cluster days, for a simple reason: the losses on those days are sequential, not simultaneous.
   Each short is stopped within ~2–3 hours; the next one opens and is stopped too. An open-book signal cannot see a run of
   losses that never overlap.
2. Where shorts do overlap underwater, the group signal is informative (null p 0.02, and far better than cutting lone shorts). But it
   is rare (≈ 5 days a year) and worth ≈ +0.01 %/day at 0.183 — about +4 % of the sleeve's rate — and the 2 holdout events both lost.
3. The asymmetric payoff is the reason it stays small: each cut that would have recovered gives up ~12 % (TP 10 % + the cut level),
   so one COST cancels three SAVED.
4. A plain tighter stop (G0) is strongly negative (−0.20 %/day), which confirms the exit-grid result: the 1.5×ATR / 6 % stop is right.
5. What the bad-day data really points to is a **sequence** signal ("the last N EXH shorts in the past few hours were all stopped").
   That reads CLOSED trades, which is the family the operator already refuted ("pause after X losses"). Not tested here and not proposed;
   it would need its own pre-registration and a reason to expect a different answer.

## Verdict

**OBSERVE — low priority. NO engine build.**
Rule to observe: **G2 M2·K3** — at a 1m close, when ≥ 3 EXHAUSTION shorts are open and their mean unrealised P&L is ≤ −2 %, close all
of them (taker). It needs no engine work to observe: the scout can rebuild the open book from the closed EXH fills plus 1m klines and
price "cut vs hold" at each event offline.
Frozen gate (window units): after **10 new event days** (≈ 2 years at the historic rate), BUILD only if Σ Δ at 0.20 % cut slippage > 0
AND ≥ 6 of 10 events positive AND the pooled design + new events day-CI excludes 0. Otherwise retire.
If it were ever built, the engine work: a per-poll open-book check across EXH positions (mean unrealised, count), a group-close path
that market-closes every open EXH short with a new exit reason (`EXH_GROUP_LIFT`), its counter / journal stamp, config fields
(K, M, on/off) with UI + both exports + tests (D11 / D12 / D13).

## Assumptions and limits

- The cut fills at the 1m close + flat slippage (0.10 / 0.20 / 0.40 %). The live bot polls more often; within-minute timing is not modelled.
- EXH-only book. Live, the book is shared with other sleeves (max_open_positions 4, hard 10), so the EXH count at a given minute can be lower.
- One size 0.183 × equity; P&L booked on the exit day. Flat slippage everywhere (the depth-model slip is used only for the parity check).
- 1m gaps are forward-filled (rare). Funding to the cut uses the recorded settlements up to the cut minute.
- G3 uses the alt-perp median 1h return from the hourly 5m-cache panel (≈ 180–530 perps); no data for E.
- Holdout = 2 events and E = 1 event: both are anecdotes in window units, not evidence either way.
- Next step: none required. If the operator wants the observe line, add it to the EXHAUSTION scout (`scripts/scout_exhaustion_short.py`)
  with the frozen gate above.
