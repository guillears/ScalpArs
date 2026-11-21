# DIST_BUCKET_EXIT — pre-registration (2026-11-21, written before any grid outcome was simulated or read)

Research only. No bot code, config or engine touched.

## Question
Should DISTRIBUTION_SHORT use a different take-profit / stop-loss depending on how far the coin fell in the 15 min before entry (`r15`)?

## Known before this pre-registration (disclosed)
- DIST_AFTERCRASH (today): the outcome by `r15` bucket under today's exit is already read (deeper fall → BETTER short; ≤ −15 % 75 fills +2.33 %, exit mix 52 % TP / 44 % SL vs rest 34 % / 58 %). That shaped the hypothesis (bigger falls may afford a larger TP / need a different stop), so the bucketing variable is not fresh; the grid outcomes are.
- DL 602 (DIST_LOSERS2): a lower per-cohort TP (3–8 %) was worse for the whole cohort and for every tested sub-cohort on HOLDOUT; Exit B kept for every fill. DL 603: TP grid, no change. DIST_THEORY_EXIT: fixed wider stops (W_SL8…15) fail on HOLDOUT. **New here:** cohorts defined by the pre-entry fall, TPs ABOVE today's 10 % (12 / 15 %), a joint TP × SL grid per cohort, and a shuffled-bucket overfit control.

## Data (frozen)
- Cohort: the 1,547 DIST fills of `reports/ENTRY_WL_2026-11-18_data/dist.pkl` (DL 586 kept, RSI1h ≥ 74.07, one position per pair). `r15` from `reports/DIST_AFTERCRASH_2026-11-21_data/features.csv.gz` (row-aligned, verified).
- Engine: the 1m-path engine of `scripts/DIST_THEORY_EXIT_2026-11-21_sim.py` (stop-first on the HIGH filled at max(stop, open); maker TP 1 tick through on the LOW; 24 h time exit at the close; censored dropped from every cell; real funding). The (today, today) cell must reproduce that sim's `REF` (≥ 99.9 % kind + gross match) or the study stops.
- Nets: r − slip − slip(taker) − 0.05 − (0.05 taker | 0.02 maker) + funding; slip = per-fill modelled (≈ 0.10 %); stress = flat 0.20 %.
- Split: DESIGN entry < 2025-07-01 · HOLDOUT ≥ 2025-07-01 (halves 25-H2 / 26-H1 / 26-H2).

## Buckets (frozen, from the DIST_AFTERCRASH pre-registration edges)
`r15` ≤ −15 · (−15, −10] · (−10, −5] · > −5. Outcome-blind counts DESIGN / HOLDOUT: 5 / 70 · 18 / 100 · 86 / 200 · 450 / 618.

## Grid (frozen, 30 cells per bucket)
TP ∈ {today = min(20×ATR15, 10 %), 5, 8, 10, 12, 15 %} × SL ∈ {today = min(1.5×ATR15, 6 %), 4, 6, 8, 10 %}; price %, 24 h. (Today's TP is 10 % on 1,546 / 1,547 fills, so TP "today" ≈ TP 10.)

## Pick rule (DESIGN only)
- Per bucket, pick the cell with the highest DESIGN avg net %/fill. Ties / no cell above today → today's exit.
- **Thin-bucket rule:** a bucket with DESIGN N < 30 cannot be fitted → it keeps today's exit. This applies to ≤ −15 (5) and (−15, −10] (18) — so the bucket set can differ from today only in (−10, −5] and > −5. The all-period best cell of the thin buckets is shown descriptively only (in-sample, never a pick).
- Secondary (descriptive): the single DESIGN-best cell for ALL fills, read on HOLDOUT — separates "bucketing helps" from "a re-tuned single exit helps".

## HOLDOUT read (once)
Bucket-specific set vs today's one exit, per bucket and total: Δ avg net %/fill with day-block bootstrap 95 % CI (10-day blocks, 3,000) · halves > 0 · Δ at 0.20 % slip · top pair / top day share of the gain · one-size DCR (0.61 × equity per fill, every calendar day), worst day, max DD before → after · 30–50 % haircut on any positive Δ.

## Nested overfit control
Same procedure (DESIGN pick per bucket with the thin-bucket rule → HOLDOUT Δ total) with `r15` permuted within each period (DESIGN, HOLDOUT separately, so bucket sizes are preserved), 100 shuffles (seed 11). Report the distribution of HOLDOUT Δ; the real set must exceed its 95th percentile.

## Two-sided stop read
Per bucket, for every SL ≠ today (TP = today) and for the picked cell: on the fills stopped under today's exit — saved (not stopped under the alternative: N, Δ $) vs deeper (still stopped, loses more); on the fills NOT stopped under today — newly stopped / changed (must be Δ = 0 for a wider stop at today's TP). Backtest only (no master DIST fills exist; the real-fill check is separate).

## Real fills
Every closed non-probe DISTRIBUTION_SHORT fill in `exports_archive/` (dedup opened_at, pair, direction; newest export 2026-11-21 19:22), re-priced from the real fill price with the same engine (real.py machinery of DIST_THEORY_EXIT): today's vs the bucket exit per fill. Direction only (N ≈ 6).

## Decision rule (pre-committed)
SHIP CANDIDATE only if ALL hold on HOLDOUT: Δ > 0 with CI lower bound > 0 · Δ > the shuffled control's 95th percentile · ≥ 2 / 3 halves positive · Δ > 0 at 0.20 % slip · DCR after ≥ before · no pair / day ≥ 50 % of the gain. Else keep ONE exit (today's). A pass would still be backtest-only (no master), go to operator review, and need the in-sample haircut + a tight revert gate.
