# FLATBASE_SHORT — TP −3 % and stop 5 % exit test — PRE-REGISTRATION (frozen 2026-10-24, before any result of this test)

Operator (2026-10-24): test a −3 % take-profit and a 5 % stop, which were never in the exit study's grid
(`FRENZY_FLATBASE_SHORT_2026-10-22_EXITS_PREREG.md`). Motivation: the 6 real fills of 10-23/10-24 (DECISION_LOG 367 first live read:
live re-walk −3.57 %/fill · TP5_SL5 +1.60 · TP3 −3.07). Six fills on two days are in-sample anecdote; this test asks the 2,000+ trade view.
Research only; no bot code / config change. Script: `scripts/study_flatbase_tp3_stop5.py` (reuses `study_flatbase_short_exits.py`).

## Cohort / entry (frozen, unchanged from the exit study)
- FLAT cohort (FLAT14 ≤ 44.39 %, the frozen label), spikes ≥ 2025-01-01, the study's cached data (`FRENZY_FLATBASE_SHORT_2026-10-22_data/`).
- Entry **S4 only** (the shipped rule, DECISION_LOG 348: first 4 h close below the spike VWAP), the study's signal bars (`signals.pkl`).
- One short per pair at a time per cell.

## Exit grid (9 cells × 2 holds)
- TP ∈ {−3, −5, −8 %} × stop ∈ {+5, +10, +15 %}. **TP −8 / stop 15 = the live baseline.** 8 non-baseline cells.
- Hold **72 h = primary** (the live max hold). 48 h reported as context only (no verdict on 48 h cells).
- Same simulator as the exit study: 0.20 % round-trip costs + 0.05 % slip on the exit price; REAL funding over the actual hold (missing
  history −0.10 %/8 h, counted); gap-aware stops (fill at max(level, bar open) + slip); TP limit at min(level, bar open); **stop first on a
  bar touching both**; time exit at the close of the last bar of the hold.
- Random-bar null: the exit study's non-anticipating null unchanged (one random delay per flat episode in [spike + 2 h, spike + 96 h], first
  eligible bar at / after it, same exit, costs, funding, one-short-per-pair; 200 shuffles).

## Periods
- ③ backtest halves: 2025-H1 · 2025-H2 · 2026-H1 · 2026-H2 (to 2026-10-01), by entry time (2025 flagged lower-reliability, as in the exit study).
- ② master: 2026-10-02 → 10-13, real FRENZY episodes. ① live: (a) the study's 10-14 → 10-22 block, real episodes, open = MTM; (b) the 6 real
  FLATBASE fills (10-23/10-24) replayed on their own path if the 5m path is available offline — otherwise the DL 367 scout read is quoted
  for the cells it covers and the rest is declared not computable.

## Δ definition
Δ = mean net % of the cell − mean net % of the baseline (TP −8 / stop 15, same hold) in the same period, each on its own one-per-pair taken
set. Day-block bootstrap (resample UTC entry days jointly for both cells, 4,000 resamples, fixed seed) for the CI.

## CANDIDATE bar (72 h cells only; ALL must hold)
1. Δ > 0 in EVERY half: 2025-H1, 2025-H2, 2026-H1, 2026-H2.
2. 2026 (H1 + H2) day-block bootstrap lower bound of Δ > 0 at the Bonferroni level across the 8 non-baseline cells:
   the 0.3125th percentile (two-sided 1 − 0.05/8).
3. The cell's 2026 mean beats its fair random-entry null (same exit / costs / funding / one-per-pair) at p < 0.05.
4. No single pair and no single UTC day ≥ 50 % of the 2026 gain (gain = Σ cell net − Σ baseline net, grouped by pair / day).
5. After a 30–50 % in-sample haircut (×0.5 worst case) the 2026 Δ is still > 0.
Fail any → NOT a candidate; the recommendation is then to keep TP −8 / stop 15. A candidate is still only a candidate (operator decides,
one change at a time, pre-committed revert gate: revert to −8 / 15 if the first 15 fills after a switch average below the live-exit re-walk
on the same fills).

## Also reported (no verdict weight)
- Per cell × half: N · WR · avg net % (incl. funding) · pre-funding avg · Δ vs baseline [95 % day-boot CI] · worst trade · max drawdown at
  the exit study's sizing ($3,200, 1st-pct loss = 5 % of the account, ≤ 5 concurrent, daily MTM); 2026 total vs null mean and p.
- Saved / cut split for each cell vs baseline on the signals taken by both: baseline-stopped trades the cell saved / made worse, baseline
  winners (TP) the cell cut (stopped) or capped (lower TP), baseline time exits.
- **5 % stop wick risk.** Count the 5 %-stop exits whose stop bar's close is back below ENTRY (a "wick stop": the 5m high crossed the stop,
  the 5m close is under the entry price) and, milder, whose close is below the stop level. If 1m bars are cached locally for the stop bar's
  day (e.g. `reports/WILLY_LATE_JOIN_2026-10-22_data/k1m/`), re-check every covered 5 %-stop: did any 1m CLOSE reach the stop level
  (sustained) or did only 1m highs touch it (pure wick)? Coverage is reported; uncovered stops are declared unchecked.
