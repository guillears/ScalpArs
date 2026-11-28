# WILLY_HT blockers — theory-first audit (2026-11-28)

*Operator ask: "re-audit all here, to make if they really have fundamentals behind to be a filter. Theory first!"*
Research only — nothing in the bot changed. Script `scripts/HT_FILTER_THEORY_AUDIT_2026-11-28.py` · data `reports/HT_FILTER_THEORY_AUDIT_2026-11-28.csv`
(one row per setup with its path features) · chart `reports/HT_FILTER_THEORY_AUDIT_2026-11-28_chart.png`.

## Summary

| Filter | Theory (why the blocked setup should lose) | Its own predictions on the price path | P&L: blocked vs kept | Verdict | Cost if wrong |
|---|---|---|---|---|---|
| **NEAR_HIGH** (dip < 1.4 % under the spike high) | no real reset: the trade needs a NEW high, and the top buyers sell into it | dose-response ✅ (0–0.7 % worst) · not a TP-vs-high artefact ✅ · "stalls just under the high" ❌ — it pokes ~1–2 % ABOVE the high, then fails (smaller best move: median +2.5 vs +4.1) | backtest 113 · 32 % · −0.93 vs −0.02 (CI −2.02 … +0.17, p 0.10) · real 17 · 35 % · −0.53 vs 41 · 56 % · +1.61 (every leave-one-out < 0) | 🟢 **fundamentals mostly confirmed**, with a revised mechanism: a failed marginal new high (double top), not a stall under it | the blocked trades are worth ≈ −0.9 % each in the backtest and ≈ −0.5 % in real data: blocking them saves money |
| **BURST** (1m volume burst that closed in its lower half) | a climax that was sold into (absorbed demand) → the price drifts down at once | early drift down ✅ (+30 min: −1.6 vs −0.6 backtest, −1.0 vs +0.6 real) · "never worked" more often ❌ (backtest 23 % vs 23 %; real 8 % vs 22 %) · effect only with a burst ✅ (real) / weak (backtest) · monotone in cloc and pre_vol ❌ (backtest: the loss sits only at cloc < 0.25 ∧ pre_vol > 3) | backtest 39 · −0.71 vs −0.02 (CI −2.34 … +0.98, p 0.42) · real fills 14 · 21 % · −1.90 vs 37 · +1.96 | 🟡 **P&L real, mechanism half-confirmed**: these trades go up first (+1 % usually prints), then reverse inside the first hour. "Sold-into climax" fits the early drift but not "never worked" | the blocked trades are worth ≈ −0.7 % (backtest) / −1.7 % (real) each: blocking saves money, but the backtest CI spans 0 |
| **FUNDING guard** (next payment ≥ 0.05 % against a long within 15 min) | ① cost: avoid paying the funding · ② signal: very positive funding = crowded longs who unwind | cost ❌: one payment ≈ 0.05–0.11 % vs WILLY_HT's real expected value ≈ +2 %/trade (backtest ≈ 0) · crowding ❌, the opposite holds: setups with next funding ≥ 0.05 % did BETTER (83 · 51 % · +1.00 vs ≤ 0.03 % ≈ −0.4) and drift UP after entry (chart) | it fired on 6 setups in 2 years (5 backtest stops, 1 real TP: IDOL 11-21) — too few to read | 🔴 **no fundamentals for WILLY_HT**: the cost theory loses to the trade's edge and the signal points the wrong way | tiny: ≈ 3 triggers per year |
| **Price-moved check** | settled today, DECISION_LOG 660 | — | — | one-sided since DL 660 | — |

**What this means:** NEAR_HIGH has a real story behind it. BURST's money is real on the live fills, but its stated reason is only half right. The FUNDING guard has no reason to exist for WILLY_HT: it was built for a FAN-flip case (DL 566, a short paying hourly funding), and for WILLY_HT both of its theories fail. None of this is a ship decision. Each line keeps its frozen revert gate, and removing an armed filter needs that gate to fire or a declared operator call (CLAUDE.md).

## Prior studies (re-used, not re-run)

| Study | DL / date | What it found | What is new here |
|---|---|---|---|
| NEAR_HIGH arming | DL 594 · 11-19 | backtest zone 117 · 34 % · −0.96 (P(mean < 0) 0.98); found by a screen, family-wise null not cleared | the theory's own PATH predictions (stall under H, dose-response, TP-vs-H mechanics) |
| BURST watch + arming | DL 618 / 619 · 11-21 | pre-registered V4; 3 real blocked all −4.10; fails the locked bar (P 0.78, p ≈ 0.20) | never-worked / early-drift / monotone / no-burst control tests |
| Funding guard | DL 566 · 11-16 | 1 of 564 fills blocked (INIT, a FAN flip paying hourly −0.183 %); extreme-funding fills otherwise winners | the economics for WILLY_HT specifically + the crowded-longs signal test |
| Funding rules, all sleeves | DL 616 · 11-21 | no funding skip / budget / time-stop / net-TP rule beats today on any sleeve | — (consistent with this audit) |
| HT blockers on 2 years | DL 655 · 11-27 | all three "point the right way", none passes the full bar; loss notes: BURST at cloc < 0.25, FAILED_SPIKE in deep failures | the same 1,142-signal set, now read on its price path |
| Price check direction | DL 660 · 11-28 | CHEAPER aborts 6 · 83 % · +4.33; made one-sided (operator rule) | — |

## Theory first (written before any new number)

- **NEAR_HIGH.** WILLY_HT buys the first red 5m candle after a frenzy spike, betting on a second leg. If the price sits under 1.4 % below the spike high H, there was no real reset. The buyers who bought the top are still near breakeven, and they sell into every approach to H (overhead supply). The +6 % TP also sits above H, so the trade wins only if the coin makes a new high.
  - *Counter-theory:* a coin that holds near its high is strong (no sellers), and momentum continues.
  - *Predictions:*
    - P1: blocked trades stall just under H more often than kept ones.
    - P2: losses grow as the dip gets shallower (dose-response).
    - P3: if the effect is only that the TP needs a new high, setups whose TP sits above H lose whatever their dip (a mechanical cause, not supply).
- **BURST.** A 1m volume burst (the last 15 bars traded > 1.6× the prior 60) whose range closed in its lower half is a climax that was sold into. The aggressive buying was absorbed by a bigger seller, and the chasers are trapped.
  - *Counter-theory:* a dip after a burst is a shake-out that gives a cheaper entry before the next leg.
  - *Predictions:*
    - B1: these entries "never work" (never reach +1 % before the stop) more often, and drift down early.
    - B2: within bursts, the lower the close, the worse the result.
    - B3: a low close does not hurt without a burst.
    - B4: the bigger the burst, the worse the result.
- **FUNDING.** ① A cost guard: a long opened minutes before a positive settlement pays rate × notional at once. ② A signal: very positive funding means crowded longs who later unwind.
  - *Counter-theory:* in frenzy coins, high funding means strong demand, which is exactly what WILLY_HT trades.
  - *Predictions:*
    - F1: a skip pays only if E[trade] < the payment.
    - F2: results fall as funding rises.

## Method

- **Backtest:** the DL 655 set, 1,142 unfiltered WILLY_HT signals from 2025-01 to 2026-11-17, priced at today's exit: TP +6 net, stop −4, 24 h, 1m path, real funding. Each filter is judged on the rows the other two armed blockers keep (the DL 655 convention).
- **Real data:** 62 WILLY_HT fills (10-23 → 11-21) plus 40 priced live refusals. All are re-walked with the same function, so blocked and kept sit on one ruler.
- **Path measures** per setup:
  - best move in the trade and in 24 h;
  - whether it came within 1 % of H, and whether it cleared H;
  - WORKED: +1 % before the −4 % stop;
  - the move at +30 and +60 min;
  - the next funding rate and the minutes to it.
- **Statistics:**
  - day-clustered bootstrap 95 % CI;
  - label-shuffle null on each prediction statistic (2,000 draws);
  - leave-one-out on the real cohorts;
  - per half-year and by BTC daily trend (1d EMA50 vs EMA200, last closed day).

## NEAR_HIGH

**Dose-response by dip below the spike high** (backtest, rows FAILED_SPIKE and BURST keep):

| dip % | N · WR · avg | WORKED | best move in trade (median) | came ≤ 1 % of H | cleared H in 24 h | cleared H, given it came near | stalled 0–2 % under H |
|---|---|---|---|---|---|---|---|
| 0–0.7 | 31 · 19 % · −2.26 | 71 % | +1.77 | 100 % | 100 % | 81 % | 19 % |
| 0.7–1.4 | 82 · 37 % · −0.43 | 73 % | +2.94 | 95 % | 85 % | 78 % | 26 % |
| 1.4–3 | 132 · 40 % · +0.01 | 80 % | +4.31 | 77 % | 86 % | 87 % | 28 % |
| 3–6 | 66 · 39 % · −0.11 | 79 % | +4.28 | 53 % | 79 % | 91 % | 21 % |
| > 6 | 62 · 40 % · +0.02 | 69 % | +3.76 | 16 % | 60 % | 60 % | 11 % |

- **P2 dose-response ✅.** The loss grows as the dip gets shallower: 0–0.7 % is the worst bucket, and from 1.4 % on the result is flat. The 1.4 % cut sits where the curve flattens.
- **P1 "stalls under H" ❌.** Stalls happen at about the same rate: 24 % blocked vs 22 % kept, shuffle p 0.80. Blocked setups actually clear H *more* often (≈ 90 % vs 78 %), because H is so close. What differs is how far they run: median best move +2.5 % vs +4.1 % (real: +3.3 vs +6.3). They poke 1–2 % above the old high and reverse, a failed new high (double top). The supply story survives, but the level is just above H, not under it.
- **P3 mechanics ✅ (rejected as the explanation).** Inside the TP-above-H setups, dip < 1.4 % still trails dip 1.4–5.7 % by −0.94 points (CI −2.01 … +0.22, p 0.10). Deeper dips whose TP sits *below* H did no better (−0.11). So "the TP needs a new high" does not explain the effect; the shallow reset does.

**Per period and BTC regime (backtest):**

| period | blocked | kept |
|---|---|---|
| 2025-H1 | 19 · 47 % · +0.52 | 32 · 31 % · −0.81 |
| 2025-H2 | 22 · 41 % · −0.23 | 58 · 48 % · +0.78 |
| 2026-H1 | 35 · 26 % · −1.42 | 67 · 30 % · −1.08 |
| 2026-H2 | 37 · 24 % · −1.62 | 103 · 45 % · +0.47 |
| BTC daily up | 48 · 38 % · −0.49 | 119 · 44 % · +0.37 |
| BTC daily down | 65 · 28 % · −1.26 | 141 · 37 % · −0.34 |

The blocked side trails in 3 of 4 half-years and in both BTC regimes; 2025-H1 is reversed.

**Real data (the primary evidence).** Fills plus refusals with dip < 1.4 %: 17 · 35 % · −0.53 (leave-one-out −0.94 … −0.31, never positive) vs kept by all filters 41 · 56 % · +1.61. Fills only: 12 · 33 % · −0.73 on 9 days vs 37 · +1.96.

**Verdict: 🟢 fundamentals mostly confirmed.**
- Holds: 2 of the 3 path predictions, the fills' smaller best move, and the real-data sign.
- Fails: the "stall under H" prediction. The mechanism is restated as a *failed marginal new high*.
- Statistically it is still thin: the backtest CI spans 0 (p 0.10).
- **Recommendation: keep.** Its frozen gate (first 15 priced refusals on ≥ 8 days, avg ≥ 0 → revert) decides; it is 4 of 15 so far.

## BURST

**Close location × burst** (backtest, rows FAILED_SPIKE and NEAR_HIGH keep):

| volume | cloc15 | N · WR · avg | WORKED | move at +30 min (median) |
|---|---|---|---|---|
| no burst | < 0.25 | 22 · 32 % · −0.82 | 68 % | −0.50 |
| no burst | 0.25–0.5 | 37 · 38 % · −0.20 | 84 % | −0.10 |
| no burst | 0.5–0.75 | 58 · 34 % · −0.61 | 74 % | −0.21 |
| no burst | ≥ 0.75 | 6 · 67 % · +2.63 | 100 % | +1.56 |
| burst | < 0.25 | 13 · 15 % · −2.54 | 69 % | −3.56 |
| burst | 0.25–0.5 | 26 · 42 % · +0.20 | 81 % | −0.83 |
| burst | 0.5–0.75 | 112 · 41 % · +0.10 | 75 % | −0.98 |
| burst | ≥ 0.75 | 25 · 52 % · +1.16 | 84 % | −1.31 |

**Low close (cloc < 0.5) by burst size:**

| pre_vol | result |
|---|---|
| ≤ 1.0 | 41 · −0.90 |
| 1.0–1.6 | 18 · +0.63 |
| 1.6–3 | 26 · +0.20 |
| > 3 | 13 · −2.54 |

The shape is a U, not monotone.

- **B1 ❌ / half ✅.** "Never worked" is the same in the backtest (23 % vs 23 %, p 1.0), and in real data the zone works *more* often (8 % never worked vs 22 %). But the early drift is down: at +30 min −1.6 vs −0.6 (backtest) and −1.0 vs +0.6 (real). These trades print +1 % first, then roll over inside the first hour. That fits a climax that keeps getting sold, not a trade that never had demand.
- **B2 partly.** Within bursts the loss sits only at cloc < 0.25 (13 trades, −2.54); 0.25–0.5 is flat (+0.20). This is the DL 655 watch note. Moving the cut there would be fitting the data that failed the original cut, so it is not done (anti-overfit rule).
- **B3 ✅ in real data, weak in the backtest.**
  - Real: low close without a burst +0.88 vs with a burst −1.70.
  - Backtest: without a burst, a low close costs little (−0.43 vs −0.31, p 0.93).
- **B4 ❌.** Not monotone: the worst cell is the biggest bursts (pre_vol > 3); mid bursts are flat.

**Per period:**

| period | zone | rest |
|---|---|---|
| 2025-H1 | 1 · −4.10 | 32 · −0.81 |
| 2025-H2 | 9 · +1.50 | 58 · +0.78 |
| 2026-H1 | 10 · −1.05 | 67 · −1.08 |
| 2026-H2 | 19 · −1.40 | 103 · +0.47 |

Only 2026-H2 separates clearly; 2025-H2 is reversed. Both BTC regimes lean the right way (up −0.50 vs +0.37 · down −0.83 vs −0.34).

**Real data (the primary evidence).**
- Fills in the zone: 14 · 21 % · −1.90 on 12 days vs 37 · +1.96.
- Fills plus refusals: 26 · 23 % · −1.70.
- The real fills say the whole zone loses; the backtest says only its extreme corner does.

**Verdict: 🟡 P&L real, mechanism half-confirmed.**
- The early-drift part of "sold-into climax" holds; "never worked" and dose-response fail.
- A candidate the theory did not name: *a short-term downtrend into the entry* (the +30-min drift and cloc < 0.25 carry the loss with or without the burst label in the backtest).
- **Recommendation: keep armed.** Its tight gate (first 10 priced refusals on ≥ 5 days, avg ≥ 0 → revert) decides; it is 1 of 10 so far. Add a watch note at the next WILLY_HT review: zone result split by cloc < 0.25 vs 0.25–0.5 on forward refusals only (observe, never re-fit).

## FUNDING guard (for WILLY_HT)

- **F1 economics ❌.** WILLY_HT's expected value per trade at today's exit:
  - real kept fills re-walked: 37 · 59 % · +1.96 %;
  - backtest kept: 260 · 40 % · −0.02 % (CI −0.62 … +0.60).
  - One payment at the guard's floor costs ≈ 0.05 %; the largest seen in 2 years was ≈ 0.11 %.
  - On real data a skip throws away ≈ +2 % to save ≈ 0.05–0.1 %. The break-even rate for WILLY_HT is roughly its expected value per trade (≈ 2 % per payment on real fills), which no setup came near.
  - On the backtest expected value (≈ 0) the guard is a coin flip.
- **F2 crowding signal ❌, the opposite holds.** By the next funding rate at entry (backtest + real):

| next rate | result |
|---|---|
| ≤ 0 | 475 · −0.18 |
| 0–0.01 % | 441 · −0.57 |
| 0.01–0.03 % | 99 · −0.53 |
| 0.03–0.05 % | 62 · −0.53 |
| ≥ 0.05 % | 83 · 51 % · +1.00 |

  - High-funding setups drift *up* after entry (chart, right panel: about +4 % at 4 h vs flat). Shuffle p for ≥ 0.03 % vs the rest is 0.09.
  - In frenzy coins high funding marks demand, not a trap.
- **Where it fires:** 6 setups in 2 years (5 backtest, all −4 stops; 1 real, IDOL 11-21, a +5.93 TP). The 15-minute timing makes this a near-random slice of the 83 high-funding setups, so the 5 stops carry no weight (N = 6). Hourly-funding coins (103 setups) almost always had negative funding at entry (median −0.08 %, longs receive), so they never trigger it.

**Verdict: 🔴 no fundamentals for WILLY_HT.**
- The cost theory loses to the trade's edge, and the signal theory points the wrong way.
- The guard was justified by a FAN-flip short paying hourly funding (DL 566) — a different sleeve and side.
- **Recommendation:** exempt WILLY_HT (and probably every FRENZY long) from the funding guard, or leave it, since it fires ≈ 3 times a year. Its DL 566 revert gate (first 5 priced blocks, avg > 0 → off) has 1 real priced block so far (IDOL +5.93). Changing it is a declared operator call, never automatic.

## Price-moved check

Settled today in DECISION_LOG 660. The check now refuses only a price that moved more than 1 % *against* the trade. Its theory, "don't chase a price that ran away", holds only on that side; a better price was never a reason to skip. 12 past aborts priced: CHEAPER 6 · 83 % · +4.33 vs DEARER 6 · 50 % · +1.00.

## Blind spots

- **FAILED_SPIKE is not audited here:** it was not in the operator's list. It is the biggest WILLY_HT blocker (≈ 15 of ~26 since 11-21) and deserves the same theory-first read.
- **Real fills before 11-12 carry no FAILED_SPIKE input,** so they are treated as kept.
- **Real burst inputs were rebuilt from 1m klines at the fill minute, not the decision minute** (seconds apart).
- **The spike high for real fills** is the stamp where present (8 of 62), else the 6 h max 1m high. DL 655 measured that proxy at about 54 % agreement near the cut.
- **The backtest and the real cohorts disagree on WILLY_HT's base expectancy** (≈ 0 vs ≈ +2 %). That changes the funding economics, not the filters' sign.
- **Paper fills; 1m path** (stop-first inside a bar); **no order-book data** (Route B starts 11-19, too short).
- **BTC trend** is the daily EMA50 / EMA200 sign only.
- **Real-data CIs are not shown** (12–17 setups per cell); leave-one-out stands in for them.
