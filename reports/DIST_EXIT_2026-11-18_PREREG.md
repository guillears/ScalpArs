# DIST_EXIT — exit study for DISTRIBUTION_SHORT — PRE-REGISTRATION (frozen 2026-11-18 22:27 UTC)

Frozen BEFORE any result of this study. No exit cell of this study has been simulated; no 1m path of a DIST trade has been read.
BACKTEST only, research only: no engine / config / test change, no commit.

## What was already known (disclosed)
- DIST_WL (`reports/DIST_WL_2026-11-18.md`, 5m paths, flat 0.10 % slip): RSI1h ≥ 74.07 kept cohort 1,548 trades · +0.90 %/trade
  under Exit B; L1 full 4,649 · +0.30. Path groups on L1: GAVE_BACK 43 % of trades (median reach ≈ +3.2 % then the stop),
  NEVER_WORKED 11 %, CAPTURED 29 %; the 10 % TP fills on ≈ 25 %. These facts motivated the family below (disclosed); the grid is
  stated now and HOLDOUT is read once.
- The parallel NOSPOT_EXIT study (`reports/NOSPOT_EXIT_2026-11-18_PREREG.md`) was frozen with the same exit-engine rules; its sim was
  running at freeze time; none of its results have been read. Its engine (`scripts/nospot_exit_sim.py`: `sim_trade`, `load1m`,
  funding) is reused unchanged in logic, extended only with an ATR-stop flag for fixed-TP cells (same code path as its REF).

## Question (operator)
DIST loses mainly on the exit side (reach ≈ +3 %, then the stop). Can a different exit (fixed TP, break-even, profit lock, trailing,
time exit) beat Exit B on DIST, with live costs?

## Cohorts (entries frozen — no entry search)
- **PRIMARY `DIST`** = rows of `reports/DIST_WL_2026-11-18_data/rows.pkl` kept by the DL 586 cut (rsi1h ≥ 74.07103, keep_mask of
  `dist_wl_survivors`, unscored rows kept), one position per pair re-applied per exit cell with that cell's own exit time.
  Selection and the ship bar are read on this cohort only.
- **SECONDARY `L1`** = all rows.pkl rows (the full L1 trigger), one position per pair per cell. Descriptive (direction check).
- **CONTROL `RND`** = random entries in the same coins / days: for every PRIMARY candidate row, 3 bars drawn uniformly (seed 20261118)
  from the 15m-close rows of `reports/UNIFIED_SHORT_data/pool.pkl` of the same pair and the same UTC day, excluding the DIST bar
  itself (fewer if the pair-day has fewer). Each draw is one trade (no one-position), entry = the 1m open at its entry minute, its own
  ATR15 / q5. Not selectable. Read: the same Δ (cell − Exit B) on RND. Δ that shows on RND too = a generic exit effect; Δ only on
  DIST = entry-specific.
- Entry = the 1m open at entry_ms (the 5m open after the 15m close).

## Exit family (54 selectable cells + the reference; short; 24 h hold unless stated)
"ATR stop" = Exit B's stop min(1.5 × ATR15 %, 6 %); "B-TP" = Exit B's TP min(20 × ATR15 %, 10 %).
- **REF  Exit B** = B-TP resting maker / ATR stop / 24 h. Not selectable.
- **A. Fixed TP × SL (30):** TP {1, 1.5, 2, 3, 5, 10} % maker × SL {2, 3, 4, 6 %, ATR stop}. Plateau = the 6 × 5 grid
  (neighbours = TP ± 1 step, SL ± 1 step in the order 2, 3, 4, 6, ATR→ ATR is neighbour of 4 and 6 — its median is 3.7 %).
- **B. Break-even (3):** REF + after the best low reaches +{2, 3, 4} % our way, stop → entry − round-trip costs
  (E0 × (1 − (0.10 % + 2 × slip)/100), ≈ 0 net). Plateau = arm {2, 3, 4}.
- **C. Profit lock (3):** REF + after +3 % stop → +1 %; after +4 % → +2 %; after +5 % → +3 % (cells L31, L42, L53). Plateau = the 3.
- **D. Trailing stop (14):** REF stop and REF TP + once the best low reaches +a % the stop = best low × (1 + g):
  a {2, 3, 4} × g {1, 1.5, 2, 3} % (12; plateau = this 3 × 4 grid) + a 3 with g = {1, 2} × ATR15 % (2; second plateau pair).
- **T. Time exit (4):** ATR stop, no TP, exit at the close after {4, 12, 24, 48} h. Plateau = the 4 holds.
- Partial exits: excluded. The cell count is 54 (> the ≈ 40 guide) because the operator's listed values cross to 54; all 54 enter the
  family-wise null, which makes the bar stricter, not looser.
- B / C / D are REF + one mechanism, so "mechanism off" = REF exactly: paths that never reach the arm must show Δ = 0 vs REF (asserted).

## Realism (fixed, = NOSPOT_EXIT)
- 1m paths (data.binance.vision daily 1m zips; caches `EXH_EXIT_GRID_data/k1m`, `NOSPOT_EXIT_data/k1m`, new days in
  `DIST_EXIT_data/k1m`), 48 h from entry. A trade whose 1m days are missing falls back to the 5m path (flagged, share reported).
- Intrabar: the stop as it stood at the bar OPEN is checked first on the HIGH; then the TP on the LOW (≥ 1 tick through); both in one
  bar → stop. Stops fill at max(stop, bar open) (gap-aware). Best low updates after the bar's checks (new BE / lock / trail level from
  the next bar). Time exit = close of the last bar inside the hold.
- Costs: taker 0.05 % per taker leg, maker 0.02 % on a TP fill. Slip per taker leg (entry and every stop / BE / lock / trail / time
  exit) = max(0.10 %, −0.042 + 0.276 · √(6,100 ÷ q5)) (SPOT_SHARE_DEEP slip model at ≈ $6k notional). Maker TP: no slip. Real
  funding at every settlement between entry and exit (hourly when Binance switched); a missing pair-month is charged 0.05 %/8 h.
- Stresses: flat 0.20 % per taker leg; 2 × slip on the dynamic (BE / lock / trail) exits only.

## Splits
DESIGN = entry 2023-07-01 → 2025-07-01 · HOLDOUT = 2025-07-01 → 2026-11-18, read ONCE, only for the survivors + REF (other HOLDOUT
numbers shown only as descriptive after the verdict). Half-years 23-H2 … 26-H2.

## Selection on DESIGN (PRIMARY cohort; survivor = all of)
N ≥ 30 · avg net > REF avg (DESIGN, each on its own one-position set) · paired Δ (cell − REF on REF's one-position entries) > 0 ·
≥ 2/3 of DESIGN half-years (N ≥ 10) with paired Δ > 0 · plateau: ≥ 2/3 of the cell's immediate grid neighbours (same family) with
paired Δ > 0 on DESIGN. Rank survivors by DESIGN paired Δ; the top 3 (at most one per family) go to HOLDOUT.

## Ship bar (all; HOLDOUT unless stated)
1. Paired Δ vs REF > 0 with a day-block bootstrap 95 % CI (10-day blocks, 2,000) excluding 0.
2. ≥ 2/3 of all half-years (N ≥ 10) with paired Δ > 0 (full span).
3. Holds at flat 0.20 % slip per taker leg: avg > 0 and paired Δ > 0.
4. Best-5 %-out avg > 0 (top 5 % of trades removed).
5. Kelly at the one-sided 95 % Wilson WR floor > 0, per trade AND per day.
6. Family-wise null p < 0.05 (full span): White reality check — day-block (10-day) bootstrap of the paired Δ matrix over all 54
   cells, re-centred at the observed means, 2,000 draws; statistic = max over cells of the studentised mean Δ.
7. No pair and no day ≥ 50 % of the summed HOLDOUT paired Δ.
8. Joint add: `scripts/joint_short_sim.py` book (EXH + FB + L4 + DIST, one account, 0.609 × equity, flat 0.10 % slip as in that
   script) with DIST on the candidate exit beats the same book with DIST on Exit B on joint DCR (full span AND the HOLDOUT window),
   with max DD no deeper by > 5 points. Plus the one-size DCR (0.61 × equity, every calendar day, P&L on the exit day) of DIST alone
   on HOLDOUT > REF's.
Verdict: **SHIP-candidate** = all 8 · **OBSERVE** = item 1 and ≥ 6 of 8 · else **keep Exit B**. 30–50 % in-sample haircut on any
projected Δ.

## Reads (descriptive, no re-fit)
- REF on 1m vs the DIST_WL 5m numbers (parity).
- Path decomposition on 1m (WORKED = +1 % before the initial stop distance; NEVER_WORKED / STOPPED_RECOVERED / GAVE_BACK / DRIFT_LOSS /
  LEFT_ON_TABLE / CAPTURED) before (REF) and after (each survivor).
- Saved / deeper (EXIT/STOP counterfactual rule), on REF's one-position entries: candidate vs REF split by REF's outcome (REF stopped →
  saved / deeper / same; REF TP; REF time). For B / C / D: assert Δ = 0 on paths that never reach the arm.
- RND and L1 Δ for every cell (entry × exit joint check).
- Per half-year; BTC daily EMA50 vs EMA200 (btc_d1_regime) and the BTC 5m label (btc_regime) at sign granularity.
- Concentration: leave-one-pair-out, top pair / day share, best-5 %-out.
- Trades/day, WR, avg win / loss, exit mix; one-size DCR at 0.61, max DD, worst day, alone and joint; stops per day (squeeze
  clustering: mean / p99 / max, and on the 10 worst days).
- Cross-check only (not decided on): REF vs the best cell on `reports/BASELINE_TRADES_EXHAUSTION_2026-11-18.csv` trades (1m).
- Robustness of the winner (descriptive, not selectable): its neighbours on HOLDOUT.
