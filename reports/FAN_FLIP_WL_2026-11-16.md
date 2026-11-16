# FAN flip: winners vs losers at entry (2026-11-16)
Operator: "yes run the fan flip winners vs losers study" (after APT 11-16 −$141, a short that never went green).

## Cohort
- **Fills:** 57 FLIP:FAN_RATIO_GATE shorts (`sweep_separators.load_real`: master stack_keep full-size 54, plus live 3: PEPE +$54, INIT −$18, APT −$141).
- **Live trades:** INIT and APT are one market moment (13:56 / 13:58 on 11-16).
- **Prices:** P&L = stack_pct (today's stack). Each fill is also priced under the exit that will run, a fixed TP +0.5 % (180 min, today's stop), from the exit study walker (scratchpad/fanx/walk.csv). APT is not in the walker; it never went green, so it is priced as booked.
- **Entry label:** scripts/path_labels.py — WORKED = +1 % our way (gross) before the sleeve's stop distance (0.7 %, its 24 losing stops) within 24 h. Result: 27 WORKED · 24 NEVER · 6 pending. REACH = the best move our way before the stop.
- **Coverage:**
  - Tested: 39 numeric + 16 level entry_* columns.
  - Blind spots (< 80 % scored): funding_rate 30 %, mkt_up4h 25 %, BTC 1 d / 4 h / off-30d / r72 / eff72 ~50 %, cmc_rank / mcap ~50 %, signed gaps 30 %, pair 1d NDI 39 %.
  - Funding is read separately: 17 fills, no signal.

## Sweep (`sweep_separators.py FLIP:FAN_RATIO_GATE`, labels path / reach / pnl)
The same two market variables lead on all three labels:
- **entry_btc_dist_from_ema13_pct:** the deeper BTC sits below its EMA13, the worse the flip.
  - path: WORKED-rate median Δ +0.30 / +0.21 (both eras).
  - reach: +2.78 / +0.61.
- **entry_btc_atr_pct:** a high BTC ATR is worse (reach tercile −1.99 / −1.50).
- **Best 2D:** BTC ATR low ∧ BTC near its EMA13 → WORKED +0.50 / +0.58 (16 fills · 94 % · +0.34).
- **3D:** 0 consistent cells.
- **Max-stat shuffled null** (booked %, every column, tercile vs rest, 300 permutations): best Δ 0.50 vs null median 0.41, **p = 0.15**. The screen does not beat chance.

## Operator hypotheses
**Gates passed "by a hair" → REFUTED (the reverse holds).** Small margins are the better half on most gates. Cells are N · WR · avg % (booked).

| Gate margin | Small half | Large half |
|---|---|---|
| BTC below EMA13 vs the −0.08 cut | 29 · 86 % · +0.26 | 28 · 61 % · −0.02 |
| BTC ADX vs the 30 block | 29 · 86 % · +0.27 | 28 · 61 % · −0.02 |
| pair RSI − 55 | 29 · 76 % · +0.15 | 28 · 71 % · +0.11 |
| pair ADX − 21 | 29 · 83 % · +0.21 | 28 · 64 % · +0.04 |

The losers come from BTC already being far below its EMA13. That is a late short into a market that has just dumped.

**Maker entry fell back to market → REFUTED.** TAKER_FALLBACK 24 · 75 % · +0.25 vs MAKER 32 · 72 % · +0.03.

**Funding at entry → not readable.** 17 fills scored, no difference.

## Candidate: BTC far below its EMA13 (entry_btc_dist_from_ema13_pct ≤ −0.16, the bottom tercile, frozen)
Terciles (booked): deep 19 · 58 % · −0.14 · mid 19 · 68 % · +0.16 · shallow 19 · 95 % · +0.36. Monotonic.

| | Before | After | Blocked |
|---|---|---|---|
| Master (booked) | 54 · 76 % · +0.15 · DCR +0.10 | 36 · 81 % · +0.27 · DCR +0.125 | 18 · 67 % · −0.10 · 13 d · 16 pairs |
| Master + live, under TP +0.5 | 57 · 77 % · +0.14 · DCR +0.10 | 37 · 84 % · +0.27 · DCR +0.13 | 20 · 65 % · −0.11 · 14 d · 17 pairs |
| Live | PEPE +$54 · INIT −$18 · APT −$141 | PEPE only | INIT, APT (b13 −0.166 / −0.168) |

- **Day-bootstrap P(mean < 0):** 0.79 booked / 0.71 under TP +0.5.
- **LOO blocked mean:** −0.20 … −0.07.
- **Concentration:** no pair ≥ 50 % of the loss.
- **Blocked WR vs breakeven:** 60 % vs breakeven 63 %.
- **Eras:**
  - BASE: blocked 86 % · +0.20 vs kept +0.48.
  - B25: blocked −0.28 vs kept −0.13.
  - B17: blocked −0.74.
  - B26: blocked +0.48.
- **Backtest yr6 (2 years, per seed; secondary):**
  - seed 1: blocked 111 · −0.23 vs kept −0.07, P(worse) 0.97.
  - seed 2: −0.21 vs −0.12, P(worse) 0.79.
  - seed 3: −0.10 vs −0.11, P(worse) 0.50.
  - 2 of 3 seeds agree; seed 3 is flat.
- **Theory:** the FAN gate already requires BTC below its EMA13 (weak tape). When BTC is far below it, BTC has just dumped and is stretched, so the next move is a bounce that squeezes fresh shorts. This is the same "late" logic as the momentum-short score ≥ 5 rule (DL 561).

## Verdict (locked expectancy bar)
| Test | Result |
|---|---|
| ① WR below breakeven | ✓ (60 % vs 63 %, marginal) |
| ② 95 % confidence | **✗** (0.79) |
| ③ ≥ 8 windows, no pair ≥ 50 % | ✓ (14 days) |
| ④ N ≥ 15 | ✓ |
| Shuffled null | **✗** (p 0.15) |
| Backtest | 2 / 3 seeds |

**Not a ship.** Register it as a pre-registered **OBSERVE** line with the cut frozen at entry_btc_dist_from_ema13_pct ≤ −0.16 (FAN flips only).
- **Read:** at every batch review, on fills under the TP +0.5 exit.
- **Arm only if** the forward blocked cohort reaches N ≥ 15 on ≥ 8 days with WR below breakeven and window-bootstrap P ≥ 0.95. Never re-fit the cut.
- **Declared-override option:** arm now with a tight revert (first 8 priced blocked signals average > 0 → off). This is weaker than DL 561, which had 3 / 3 backtest seeds plus a same-minute proof; here it is 2 / 3 seeds and the null is not beaten.
- **Live today:** INIT and APT were one market moment, which counts as a single observation.
