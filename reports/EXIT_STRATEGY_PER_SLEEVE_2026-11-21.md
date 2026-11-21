# Exit strategy per sleeve — theory-first exit studies (2026-11-21)

Four pre-registered studies started from what each sleeve is trading and tested only exits that had not been tested before. Each one
reproduces today's exit exactly (parity), reads real fills and the backtest, and has frozen gates. **No exit change passes on any sleeve.**

| Sleeve | What the trade is | Exits tested | Today's exit (real fills, today's filters) | Best candidate | Why it fails | Recommended exit |
|---|---|---|---|---|---|---|
| EXHAUSTION_SHORT | short a pump that ran out of buyers | 23 + 7 controls | 7 closed · +2.40 %/fill (re-priced) | fixed 7 % stop (HOLDOUT +0.23, CI > 0) | loses the frozen 2023-H1 test (−0.20); deeper drawdown; RARE #26 worse | keep TP min(20×ATR,10) / SL min(1.5×ATR,6) / 24 h |
| DISTRIBUTION_SHORT | short a big holder selling into the pump | 41 in 9 families | 6 closed · −3.06 %/fill | stall exit +0.02 / 12 % stop +0.07 | one pair 89 % of the stall gain; wider stops lower the daily return | keep Exit B (same as EXH) |
| FLATBASE_SHORT | a failed pump on a 14-day-flat coin unwinds to its base | 26 + 7 controls | 23 · 74 % · +0.92 %/fill · +0.42 %/day | stop at pump peak +1 % (HOLDOUT +0.44, 6/6 gates) | master +0.08, P 0.52 (< 0.80); mostly a wider stop; avg loss doubles | keep TP −5 maker / stop +10 incl. funding paid / 72 h |
| WILLY_HT | buy the dip after a spike, expect a retest of the spike high | 23 | 43 · 56 % · +1.03 %/fill · +1.13 %/day | stop under the pre-spike base (+0.29, P 0.72) | below the 0.80 bar; a wider stop in disguise | keep TP +5 maker / stop −4 / 24 h |
| WILLY (plain) | same entry, V2b exit | 30 | 23 · DCR +0.40 %/day | half at V2b + half at spike high +1 % (+0.12, P 0.88) | HOLDOUT −0.03; ≈ chance among 30 rules; 0 fills since DL 595 | keep V2b |

## The pattern across all four
- **Every "structure" stop (pump top, candle high, dip low, base) is a wider stop in disguise.** A fixed stop at the same distance does as well or better.
- Wider stops win in some periods and lose in others, and they always deepen the worst day and the drawdown.
- **Every earlier target (pump levels, the spike high, stall, funding budget) cuts more from the winners than it saves on the give-backs.** The current batch is the exception, and it is the batch that prompted these studies (in-sample).
- **The give-back trades (RARE, ENSO, BLESS, EVAA) are the cost of each sleeve's edge, not a fixable exit bug.** At the moment a trade peaks, the ones that give back look the same as the ones that go on to hit the target.

## Remaining levers (not exits)
1. **Entries.** For FLATBASE and WILLY_HT, trades that never worked lose more than give-backs do.
2. **Size and clustering.** EXH backtest max drawdown is −68 % at today's size; this is the TP × size question at the batch review (DL 603).
3. **Route B** order-book / trade-tape data (~12-10 weekly scan): the only new information an exit could still use.

## Optional observe-only lines (operator's choice; none is a ship)
- EXH_SL7_SHADOW: 7 % stop vs today on every new EXH fill (≥ 20 fills / ≥ 8 days).
- FB_VWAP_STOP: stop on the episode VWAP. Best on real fills (+0.43 with today's filters, P 0.73), but the backtest disagrees on the same weeks.
- HT_NEWHIGH: TP at the spike high (pre-registered, judged on future fills only).
- WILLY_HALF: half at V2b, half at spike high +1 % (decide at 20 plain-WILLY fills).

## Reports
- EXH_THEORY_EXIT_2026-11-21.md
- DIST_THEORY_EXIT_2026-11-21.md
- FB_THEORY_EXIT_2026-11-21.md
- HT_THEORY_EXIT_2026-11-21.md

Each has a matching _PREREG.md, a chart and a data folder.

---
## ⚠ CORRECTION (2026-11-21, operator-caught): "today's filters" master cohorts were wrong for WILLY_HT and FLATBASE
- **WILLY_HT:** the study used dip ≥ 1.4 only (43 fills). Today's stack also blocks WILLY_HT_FAILED_SPIKE (vs_vwap ≤ 0, armed DL 527). Correct master = **12 fills** (the master pool's 15 kept, minus 3 that fail the dip). Re-read: `HT_THEORY_EXIT_2026-11-21_cells_fixed.csv`.
- **FLATBASE:** the study used 23 fills. Correct master = pool stack_keep 28 (CONFIRM + HIGHER_HIGH) → DL 577 NEAR_BASE gives 20 (matches DL 577's "28 → 20, 7 W / 1 L") → DL 601 funding gives **17 fills**. Re-read: `FB_THEORY_EXIT_2026-11-21_cells_fixed.csv`.
- EXH and DIST have no master fills (both started 11-18). Plain WILLY used the pool's stack_keep (correct).

### Before → after (today's exit → candidate), $ at one size ($3,050 / fill), DCR at 0.61 × equity / fill
| Sleeve | Block | N | Today's exit: WR · avg · $ · DCR | Candidate | After: WR · avg · $ · DCR | Δ / fill (P) |
|---|---|---|---|---|---|---|
| WILLY_HT | master | 12 | 83 % · +3.63 · +$1,328 · 1.63 | stop under pre-spike base | 100 % · +5.15 · +$1,884 · 2.32 | +1.52 (0.91), 2 fired |
| | | | | TP +6 | 83 % · +4.46 · +$1,632 · 1.99 | +0.83 (1.00), 0 cut |
| | | | | TP at spike high (A_H0) | 83 % · +2.60 · +$953 · 1.16 | −1.03 |
| | | | | TP 2.5 | 92 % · +1.97 · +$720 · 0.89 | −1.66 |
| | batch | 12 | 50 % · +0.45 · +$164 · 0.71 | stop under pre-spike base | 58 % · −1.00 · −$365 · −1.85 | −1.45 |
| | | | | TP +6 | 50 % · +0.95 · +$348 · 1.57 | +0.50 (0.94), 0 cut |
| | | | | TP at spike high | 83 % · +2.06 · +$755 · 3.70 | +1.61 (1.00) |
| FLATBASE | master | 17 | 82 % · +2.20 · +$1,143 · 0.81 | stop at pump top +1 % (C_PK1) | 100 % · +4.58 · +$2,377 · 1.75 | +2.38 (0.96), 3 saved / 0 cut |
| | | | | stop at episode VWAP | 82 % · +2.56 · +$1,328 · 0.94 | +0.36 (0.61) |
| | batch | 3 | 33 % · −5.27 · −$483 · −3.25 | stop at pump top +1 % | 67 % · −6.17 · −$564 · −3.97 | −0.89 (KERNEL −25.4) |
| | | | | stop at episode VWAP | 67 % · −1.24 · −$113 · −0.78 | +4.03 (0.76) |
| EXH | batch | 7 | 57 % · +2.40 · +$512 | fixed 7 % stop | 86 % · +5.34 · +$1,140 | +2.94 (YB) |
| | | | | stop at pump top | 71 % · +4.08 · +$871 | +1.68 |
| DIST | batch | 6 | 17 % · −3.06 · −$560 | stop at pump top +1 % | 50 % · +0.77 · +$141 | +3.83 (2 fills) |
| WILLY (plain) | master | 23 | DCR 0.40 | half V2b + half spike high +1 % | DCR 0.44 | +0.12 (0.88) |

---
## FINAL (2026-11-21 evening): re-runs on the exact kept lists
Kept lists: `KEPT_LIST_WILLY_HT_2026-11-21.csv` (12 master + 12 batch) · `KEPT_LIST_FLATBASE_2026-11-21.csv` (17 + 3). Reports: HT_EXIT_RERUN_2026-11-21.md · FB_EXIT_RERUN_2026-11-21.md · DIST_PEAKSTOP_2026-11-21.md · EXH fixed TP: `EXH_THEORY_EXIT_2026-11-21_data/tpfix_bt.csv` + `tpfix_real.csv`.
Correction to the table above: the −25 % trade under C_PK1 was MMT, not KERNEL (FB re-run); DIST C_PK1 was capped at 15 %, not 25 %.

| Sleeve | Recommendation | Basis |
|---|---|---|
| WILLY_HT | TP +5 → +6 (stop −4 / 24 h unchanged), operator-declared, revert alert on the first 10 fills reaching +5: Σ(TP6 − TP5) < 0 | P(+6 given +5): real kept 16/16, backtest 118/131 = 90 % = breakeven → long-run ≈ 0 to +0.06; master +3.63 → +4.46, batch +0.45 → +0.95; nothing in DL 587 argues against it |
| FLATBASE | keep +10 stop; optional FB_STOP12_SHADOW (observe) | capped-top pick = plain +12; fails the HOLDOUT CI and loses on DESIGN; master +2.20 → +3.62, batch −5.27 → −6.61 |
| EXH | operator call: TP 8 (or 7) instead of 10. Expectancy about the same, ~40 % shorter hold, lower drawdown; revert gate needed | HOLDOUT TP8 Δ 0.00 · TP7 −0.03 · E block +0.12 / +0.06 · DESIGN −0.08 / −0.14; batch (10 incl. 3 open) today +2.27 → TP7 +3.88 / TP8 +2.86 |
| DIST | keep today's exit; no line | the pump top behaves like a random level of the same width (shuffled control); all 8 cells lose on HOLDOUT |
