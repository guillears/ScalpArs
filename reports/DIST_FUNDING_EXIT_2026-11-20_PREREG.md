# DIST_FUNDING_EXIT — close a DIST short before a strongly negative funding settlement? — PRE-REGISTRATION (frozen 2026-11-20, before any result)

Frozen BEFORE any rule variant of this study was simulated. Research only: no engine / config / test change, no commit.

## Known before freezing (disclosed)
- `reports/FUNDING_RULE_2026-11-19.md`: SKIP / HALF at ENTRY by funding rate — 0 / 20 rules pass. DIST ≤ −1.0 % entry bucket:
  33 fills · 52 % WR · +2.10 % net after funding (the best bucket). Pre-settlement close explicitly left untested there.
- DIST Exit B (REF_B) pays on average −0.09 % funding per fill (already inside net).
- Live motivation: KERNELUSDT DIST short 2026-11-20 12:30 UTC, funding −2.00 % / 4 h (cap).

## Cohort (frozen)
- DIST fills = `reports/DIST_WL_2026-11-18_trades.csv.gz` with `keep_strength` joined on (pair, entry_ms) to
  `reports/DIST_EXIT_2026-11-18_trades.csv.gz` cell `REF_B` (1,534 fills). atr15 / q5 from `DIST_WL_2026-11-18_data/rows.pkl`
  (slip = max(0.10, slip_model(q5)) exactly as `dist_exit_sim.frames`). The fill set is held fixed (an earlier B1 exit could free the pair
  for a fill one-position blocked; not modelled — stated).
- Paths / funding: exactly `scripts/dist_exit_sim.py` (`load1m`, 1m with 5m fallback, `TB.funding_for` settlement times + rates, a
  settlement is paid when entry < t ≤ exit-bar close; missing pair-months charged FUND_MISS).

## Replica parity (must hold before any rule is read)
- My REF_B re-sim reproduces the DIST_EXIT REF_B `net` per fill (|diff| ≤ 0.001 %) — report the match rate.
- Fills that never meet a rule show Δ = 0 exactly.

## Rules (16 variants = 4 X × 2 M × 2 B)
- At each funding settlement s while the fill is open (decision time s − M min must be ≥ entry + 1 min, else that settlement is
  ignored), if the SETTLED rate of s ≤ −X % per interval (X ∈ {0.25, 0.5, 1.0, 2.0}), close at market at the open of the bar at s − M
  (M ∈ {1, 5} min; on 5m-fallback paths the last bar with t ≤ s − M). Costs: taker 0.05 % + the fill's modelled slip, as REF stops.
- **B1**: close and stay out.
- **B2**: close, then re-enter short at the open of the bar at s + 1 min (5m paths: first bar with t ≥ s + 1 min), keeping the ORIGINAL
  TP price, SL price and 24 h clock; pays another entry slip + 0.05 % taker; the new leg follows the same stop-first / maker-TP rules and
  the rule again at later settlements. If the re-entry price is already ≥ SL or ≤ TP, or the 24 h clock is over, no re-entry. One size:
  each leg's % is added (same notional).
- **Look-ahead (stated)**: the SETTLED rate is used as if known at s − M. Binance's predicted rate is the interval's premium average and
  converges before settlement (at the cap it is exact). Sensitivity (not a pass criterion): the same rules triggered by the PREVIOUS settled
  rate of the coin (fully known; ≤ −X) instead.

## Periods / reads
- DESIGN entries 2023-07-01 → 2025-06-30 · HOLDOUT 2025-07-01 → 2026-11 (read once).
- Δ = summed net % at one size (rule − REF_B). Day-block bootstrap (10-day blocks as FUNDING_RULE, 2,000 draws) 95 % CI.
- Family-wise null: 300 shuffles of the settlement rates within half-year (pool = every settlement during a DIST fill's life in that
  half-year); the trigger uses the shuffled rate, the funding actually paid stays real; null = 95th percentile of the max DESIGN Δ over all
  16 variants.
- Trading possibilities: fills touched, distinct days, distinct (coin, day).
- Concentration: top pair share of Δ (of positive Δ if Δ > 0). LOO: Δ with the single biggest contributing fill removed, and with the top
  pair removed.
- Saved vs cut: per touched fill, Δ > 0 (saved: the held fill lost more / paid funding after) vs Δ < 0 (cut: the held fill recovered / hit
  TP after); also funding avoided vs price give-up.
- Half-year table. In-sample haircut 30–50 % on any positive DESIGN Δ.

## PASS (all required)
HOLDOUT Δ > 0 with bootstrap CI lower bound > 0 · DESIGN Δ > 0 · DESIGN Δ > null 95th · top pair < 50 % of Δ (all periods) ·
Δ > 0 in ≥ 2/3 of the half-years in which the rule touched any fill.

## Secondary
- EXHAUSTION_SHORT only if its EXH_EXIT2 REF exit can be replicated by the same machinery cheaply; else stated as not done.
- Live / paper DIST fills of the newest orders export: descriptive only (few fills).
