# FB_THEORY_EXIT — pre-registration (frozen 2026-11-21, before any cell is simulated)

Research only. No engine / service / config change, nothing committed. Template: EXH_THEORY_EXIT_2026-11-21 (same engine style, same
decision structure), adapted to FLATBASE_SHORT.

## Why
Operator: "build all studies, and bring a solid exit strategy per sleeve". FLATBASE exits so far were searched as fixed grids (TP / stop /
hold, trails, locks, in-trade state signs). None asked what the trade's own THEORY says the exit should be.

## Theory of the trade
FLATBASE_SHORT shorts a coin that sat in a FLAT 14-day range, got pumped by a Frenzy episode, and then failed: a 4-h close below the
spike-anchored VWAP (S4), confirmed by a further 4-h close below the S4 close (FB_CONFIRM), still > 2.09 % above the pre-pump base, no
funding above base. With no re-rating behind the pump, price should drift back to where it lived before. The trade is a **pump unwind
toward the old base**. What follows:
1. **Natural target = the pump's own landmarks**, not a fixed −5 %: the pre-pump base (the 5m close 30 min before the spike bar), the
   half-way point entry → base, the 61.8 / 78.6 % retraces of the pump (peak → base). Today's −5 % is blind to them: on a coin 30 % above
   its base it leaves the unwind on the table; on a coin 3 % above it the target sits below the base.
2. **The thesis is invalidated by structure, not by a fixed %**: a reclaim of the episode VWAP (the average pump buyer is back in profit),
   a break of the high of the 4-h candle that triggered the entry (the breakdown failed), or a new high above the pump peak (the pump
   resumes). Today's +10 % stop sits above, below or between those by accident.
3. **An unwind that stalls is over**: once the drop stops making new lows for hours, the seller wave is done and the rest of the 72 h is
   exposure to a second pump.
4. **Flat bases resolve within a day or two**: a short that has done nothing after 12–48 h is not the trade the thesis described.
5. **Crowded shorts pay funding**: once a set budget is paid the expectancy is spent (DL 607 already counts paid funding in the stop;
   the budget exit is a separate, earlier rule).

## Already tested — cited, not redone
FRENZY_FLATBASE_SHORT_2026-10-22_EXITS (fixed TP 5–20 × stop 10/15/25 × hold, a trail) · FRENZY_FLATBASE_LOCK_2026-10-23 (locks /
break-even / trails A×G) · FRENZY_FLATBASE_TP3_STOP5 + TP1 shadows (DL 368 / 370 / 390) · FLATBASE stop −6…−9 (DL 540 study) ·
TP_GRID_2026-11-20_FB (TP 3–10 × stop 4–12 × 24/48/72 h, DL 603; TP −6 is the best fixed alternative → shadow DL 604) ·
DIST_FB_LOSERS_2026-11-20 (stop width) · STATE_EXIT_2026-11-21 + AUDIT (1h close reclaims VWAP after +2 %, 4h close above entry, within
1 % of base after +2 %, lower highs, RSI, EMA, climax, volume, funding state — all null) · FUNDING_STOP_2026-11-21 (stop counts paid
funding → armed DL 607). Overlap note: the C_VWAP stop below is an intrabar VWAP stop active from the fill (STATE_EXIT tested a 1h-close
reclaim gated on a +2 % peak); A_* are far / anchored targets (STATE_EXIT's FB_BASE was a market exit near the base after +2 %).

## Cohorts and engine
- **Backtest** = `reports/FUNDING_STOP_2026-11-21_data/bt_FB.csv` minus entry_rate > 0.00005 (DL 601) = **545 fills** (today's stack:
  NEAR_BASE, higher-high, confirm entry). 5m paths from the OOS_REAL_FLATBASE loader (`oos_real_flatbase_recon.W.load`), real funding
  (`oos_real_flatbase_common.funding_table`, mark-scaled, assigned to bars). Walk = `FUNDING_STOP_2026-11-21_common.walk` conventions:
  stop first on the bar HIGH (filled at max(line, open) + slip_x), maker TP filled at the line only when the low trades 0.05 % through,
  time exit at the last close inside 72 h + slip; backtest fee convention taker 0.045 / maker 0.018 (the rebuild's own, kept for parity);
  rule exits = taker at the NEXT bar's open + slip_x. The stop line counts funding PAID: line = E × (1 + (d + min(F, 0)) / 100) where d is
  the cell's stop distance (DL 607 — applied to every stop cell, not only today's).
  **Parity gate: my REF must reproduce bt_FB `Bp_net` (|gap| < 0.01 %) and `Bp_why` on ≥ 99 % of fills, else stop and report.**
  DESIGN = entries < 2025-07-01 (94) · HOLDOUT ≥ 2025-07-01 (451, read once).
- **Real fills** = every `exports_archive/scalpars_orders_*.csv`, FLATBASE_SHORT, CLOSED, no probes, operator-closed out, dedup
  (opened_at[:19], pair, direction) keeping the newest export. Entry = the real fill price; walk from the first 1m bar opening after the
  fill; costs taker 0.05 entry, maker 0.02 TP, taker 0.05 + 0.25 slip on stop / time / rule exits (TP_GRID FB real-slip figure), real
  funding (settlement-assigned, mark-scaled). 1m klines: STATE_EXIT_2026-11-21 caches (read only) + a fresh fetch of the newest days into
  this study's own folder (vision zips, fapi only for days not yet published; used-weight header read; stop on 418 / 429).
  - **MASTER as taken** = opened < 2026-11-17T17:00 (every fill, incl. the S4-era entries before FB_CONFIRM, all walked under today's exit).
  - **MASTER today's filters** = MASTER minus entry_funding_rate > 0.00005 minus fills whose fill price is ≤ 2.09 % above the pre-pump base
    (approximation from 1m klines: the confirm close ≈ the fill price; labelled).
  - **BATCH** = opened ≥ 2026-11-17T17:00 (current batch), reported apart; open-ended paths are marked to the last available minute.
- **Anchors** (all known at the entry): on-bar = the spike bar (spike − 5 min); BASE = close 30 min before the spike-bar close; PEAK =
  max high from the on-bar to the bar before entry; VWAP = Σ tp·q ÷ Σ q (q = volume × tp) from the on-bar to the bar before entry;
  CFHI = the high of the 4-h candle that closed right before the entry (the FB_CONFIRM candle; for S4-era real fills the S4 candle).

## Cells (frozen) — REF + 31
| family | cell | rule |
|---|---|---|
| REF | REF | today: maker TP −5 % · stop +10 % counting paid funding · 72 h |
| X fixed-TP benchmark | X_TP6 | TP −6 / +10 / 72 h (DL 603's best fixed TP; the "beats the best fixed-TP alternative" bar) |
| A anchored target (replace today's TP) | A_BASE · A_HALF · A_R618 · A_R786 · A_FARB | maker TP at BASE / (E + BASE) ÷ 2 / PEAK − 0.618 (PEAK − BASE) / PEAK − 0.786 (PEAK − BASE) / the FARTHER of BASE and −5 %. Anchor not ≥ 1 % below entry → today's TP; capped at −25 % |
| C structure stop (replace today's stop) | C_PK0 · C_PK1 · C_TIGHT · C_VWAP · C_CFHI | stop at PEAK × 1.00 / PEAK × 1.01 / min(today, PEAK × 1.005) / VWAP × 1.005 / CFHI × 1.005. Level not ≥ 1 % above entry → today's stop; capped at +25 % |
| K fixed-stop CONTROL | K_S8 · K_S12 · K_S15 · K_S20 · K_MPK · K_MVWAP · K_MCFHI | fixed stops; K_M* = a fixed stop at the backtest-cohort MEDIAN distance of the matching structure level (PEAK×1.00 / VWAP×1.005 / CFHI×1.005, capped as in C; outcome-blind, computed from entry data) — a structure effect must beat its distance-matched control |
| B stall | B_X{2,3,4}_N{240,480,960} | after the best low reaches +X %, if no new low for N minutes and the close is in profit → taker exit at the next bar's open |
| T time-thesis | T_H12_Y1 · T_H24_Y2 · T_H24_UW · T_H48_UW | at entry + H h: exit (taker, that bar's open) if the best move so far < +Y % / if price is above entry (under water) |
| F funding budget | F_P1 · F_P2 · F_P1N | at a settlement, cumulative funding PAID ≥ 1 % / ≥ 2 % of notional → taker exit at the next bar's open · F_P1N = ≥ 1 % paid AND the price P&L < the funding paid |

Everything not named keeps today's value (a TP cell keeps the +10 paid-funding stop and 72 h; a stop cell keeps TP −5 and 72 h).

## Reads
Per cell and block: N · WR · avg net % · Δ = cell − REF per fill (paired) · Σ Δ · SAVED $ (fills where REF lost and the cell did better)
· CUT $ (fills where the cell did worse) · $ at one size = Δ % × $3,050 per fill (0.61 × a $5,000 equity) · DCR-cal before → after
(one size 0.61 × equity per fill, every calendar day of the block, 0 on no-fill days) · avg loss · worst day · max drawdown at one size.

## Decision rule (pre-committed)
1. **Pick per family on backtest DESIGN** (entries < 2025-07): the cell with the largest mean Δ (families A, C, B, T, F; K and X are
   controls, never picked). The pick is frozen before MASTER / HOLDOUT are read.
2. **MASTER decides** (cohort: MASTER today's filters; as-taken and BATCH shown): ① mean Δ ≥ 0 · ② SAVED $ ≥ CUT $ · ③ day-block bootstrap
   (UTC entry day, 4,000 draws) P(Δ > 0) ≥ 0.80 · ④ mean Δ > X_TP6's mean Δ on the same fills · C picks also ⑤ beat their distance-matched
   K control's Δ (else it is "just a different stop width").
3. **HOLDOUT cross-check, read once** (caveat, not a veto — master over backtest): ① mean Δ > 0 with the 10-day-block 95 % CI above 0 ·
   ② Δ > 0 in ≥ 2 of the 3 half-years (2025-H2, 2026-H1, 2026-H2) · ③ still > 0 with every taker exit at a flat 0.20 % slip (REF too) ·
   ④ no pair and no day ≥ 50 % of Σ Δ · ⑤ SAVED $ ≥ CUT $ · ⑥ one-size DCR after ≥ before.
4. **Verdict:** all master gates ∧ all HOLDOUT gates → SHIP CANDIDATE (operator decides; a priced revert gate is proposed) · master pass,
   HOLDOUT fail → OBSERVE line only (stated caveat) · master fail → REJECT (whatever HOLDOUT says). Leave-one-out on master is shown; a
   pick whose master Δ turns negative dropping one fill is labelled FRAGILE. Any master number derived from the batch the rule is
   judged on carries the 30–50 % in-sample haircut.
5. If no pick passes, the recommended exit is TODAY's exit, stated plainly with the evidence.
