# FB_EXIT_RERUN — pre-registration (frozen 2026-11-21, BEFORE any new cell is simulated)

Research only. No engine / config change, nothing committed. Re-run of FB_THEORY_EXIT_2026-11-21 (which used a wrong 23-fill master).

## Cohorts (fixed, not derived here)
- **MASTER (17)** and **BATCH (3)** = exactly `reports/KEPT_LIST_FLATBASE_2026-11-21.csv` (pair + opened_at[:19]).
  Derivation (for the report only): MASTER_POOL_stacked stack_keep 28 (FB_CONFIRM + HIGHER_HIGH) → DL 577 NEAR_BASE skips LA, BANK, NIL,
  SIGN, STO, PRL, KAT, EDGE (= DL 577's "28 → 20") → DL 601 funding > 0.00005 removes BTR 10-23 20:00, DODOX 11-13, TAC = 17.
  Batch = KERNEL, MMT, PIPPIN (PHAROS removed by DL 601).
- **Backtest** = the first study's 545 fills (bt_FB minus DL 601), same 5m loader, same funding, same walker
  (`FB_THEORY_EXIT_2026-11-21_common.run_cell`). DESIGN = entries < 2025-07-01 · HOLDOUT ≥ 2025-07-01 (read once).
- Real-fill walk = the first study's real walker (1m, real fill price, taker 0.05 entry, maker 0.02 TP, taker 0.05 + 0.25 slip on stops /
  time, real funding), same caches.

## Cells (frozen)
All cells keep today's maker TP −5 % and 72 h; the stop line counts funding PAID (DL 607). Only the stop distance d changes.
| cell | stop distance d (% above entry) | role |
|---|---|---|
| REF | 10 | today |
| PKC12 | min(PEAK × 1.01 / E − 1, **12 %**) | candidate |
| PKC15 | min(PEAK × 1.01 / E − 1, **15 %**) | candidate |
| PKC20 | min(PEAK × 1.01 / E − 1, **20 %**) | candidate |
| C_PK1 | min(PEAK × 1.01 / E − 1, 25 %) | reference (first study's pick; operator rejects the 25 % worst case) |
| K_S12 · K_S15 · K_S20 | fixed 12 / 15 / 20 | distance-matched controls (never picked) |

PEAK = max high from the spike bar to the bar before entry (first study's anchor). If PEAK × 1.01 is not ≥ 1 % above entry → d = 10 (today).
Note: when the pump peak is < 9 % above entry the candidate stop is TIGHTER than today's 10.

## Pick rule (frozen)
1. On backtest **DESIGN** (row slip): pick the PKC cell with the largest mean paired Δ (cell − REF, %/fill). If two are within 0.05 %/fill,
   take the tighter cap (smaller worst-case loss). The pick is frozen before HOLDOUT / MASTER / BATCH are read.
2. Structure check (reported, not a veto): the pick should beat its matched fixed control K_S{cap} on DESIGN and HOLDOUT; if it does not, the
   report says the peak element adds nothing over a plain fixed stop.
3. **HOLDOUT gates (all required):** day-block bootstrap 95 % CI of mean Δ > 0 · ≥ 2 of 3 half-years (2025-H2, 2026-H1, 2026-H2) mean Δ > 0 ·
   Δ > 0 at a flat 0.20 % slip · no single pair and no single day ≥ 50 % of the positive Σ Δ · saved $ > cut $ · one-size DCR after ≥ before.
4. **MASTER (17) decides (real fills lead):** day-block bootstrap P(Δ > 0) ≥ 0.80 · saved $ ≥ cut $ · min leave-one-out mean Δ > 0.
   BATCH (3) shown, not gated.
5. Verdict: HOLDOUT + MASTER pass → SHIP candidate (operator). MASTER passes but HOLDOUT fails → OBSERVE (shadow line). MASTER fails → no ship.

## Caveats stated up front
- The cap grid is motivated by a BATCH fill (the 25 % stop in the first study's C_PK1) → BATCH is in-sample for the cap choice.
- MASTER 17 fills / ~15 days: small N; the backtest is the cross-check, a refutation is a stated caveat.
- In-sample haircut 30–50 % applies to any MASTER Δ (the C family was designed after looking at master stops).

## Reads
N · WR · avg net % · total $ at one size ($3,050/fill = 0.61 × a $5,000 equity) · avg win · avg loss · worst trade · DCR one size
(0.61 × equity, every calendar day) · avg + median duration (h) · TP / STOP / TIME counts, before → after, per block. Per-fill tables for
MASTER and BATCH. Size read: per-stop loss, Kelly at the one-sided 95 % Wilson WR floor, clustering (same-window stops, worst day, max DD).
Parity: REF reproduces bt_FB Bp_net / Bp_why (≥ 99 %) and real REF matches booked exit kinds for fills ≥ 10-28.
