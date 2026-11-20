# FAST_TP — pre-registration (frozen 2026-11-20, before any cell was simulated)

Operator idea (2026-11-20): "if the trade gets X % profit SO FAST, we should take it — valid for all sleeves".
Hypothesis: a favourable move of +X % within the first N minutes is a liquidity flush that mean-reverts; the slow follow-through
reaches the sleeve's wide TP. Already refuted (not redone): plain small TP / lock / BE / trail on EXH, DIST, no-spot shorts
(EXH_EXIT2 / DIST_EXIT / NOSPOT_EXIT 2026-11-18). This study tests only the SPEED-conditioned version.
The two motivating live trades (KERNELUSDT DIST 11-20, NAORISUSDT WILLY_HT 11-20) are reported as anecdotes only; they do not
enter the grid, the pick or the verdict.

## Sleeves (each its own row, never pooled) and baselines = the CURRENT exit (trading_config.json 2026-11-20)
| sleeve | entries (frozen) | current exit (REF) | REF source |
|---|---|---|---|
| EXHAUSTION_SHORT | 3,368 kept fills, EXH_EXIT2_2026-11-18_trades (= BASELINE_TRADES_EXHAUSTION dl582_block False) | TP min(20×ATR15, 10 %) maker / SL min(1.5×ATR15, 6 %) / 24 h | EXH_EXIT2 1m engine REF (parity 3,368/3,368 there) |
| DISTRIBUTION_SHORT | 1,547 REF_B fills, DIST_EXIT_2026-11-18_trades | same as EXH (Exit B) | DIST_EXIT 1m engine REF_B |
| FLATBASE_SHORT | 779 fills, BASELINE_TRADES_FLATBASE_2026-11-18 | TP 5 % maker / SL 10 % / 72 h | re-simulated here on 1m (baseline file is 5m) — parity vs the file reported |
| FRENZY_WILLY | 545 fills, BASELINE_TRADES_WILLY_HT (sleeve FRENZY_WILLY) | TP 1.25 (high target ≤ 8) · late TP 0.25 after 30 min · runner trail when 1h gap < 0 · SL 4 · 90 min | re-simulated here (wsim logic) — parity vs the file reported |
| FRENZY_WILLY_HT | 417 fills, same file (sleeve FRENZY_WILLY_HT) | **TP 5 % maker / SL 4 % / 24 h, no late TP, no high target** (today's config; the baseline file still holds the old +2 exit) | re-simulated here; parity checked first at the file's old exit |

Entries are frozen (paired). An earlier exit could free the coin / slot for a later signal — not added (same caveat as the earlier exit studies).

## Rule family (33 cells per sleeve, frozen)
- **F (resting early TP), 30 cells:** at entry a maker limit at entry ± X % in the trade's favour, live only for bars that START in
  the first N minutes; then cancelled and the sleeve's normal exit runs unchanged. X ∈ {1, 1.5, 2, 3, 4} % × N ∈ {1, 2, 3, 5, 10, 15} min.
  - Fill: the 1m bar trades ≥ 1 tick through the level; fills at the level, maker fee 0.02 %, no slip.
  - X is the price move from the sleeve's reference entry price (EXH / DIST: the entry bar open the engine uses; FLATBASE: the slipped
    fill), except WILLY / HT where X is the NET target priced like their own TPs (price_for_net, maker) — ≈ 0.07 % apart.
- **V (speed ratio), 3 cells:** at the close of each 1m bar m = 1 … 30 min after entry, if best favourable excursion so far ÷ m ≥ R
  (R ∈ {0.5, 1, 2} %/min), exit at that close as a taker (fee 0.05 % + the sleeve's slip).
- **Conflicts (conservative):** the cell changes a trade only when its trigger bar is STRICTLY before the REF exit bar. Same bar as the
  REF exit → REF wins (stop first), except a REF fixed TP farther than X in the same bar (the nearer limit fills first → early TP).
  Fills where the rule never triggers before REF's exit carry Δ = 0 (asserted).
- **Costs = live costs:** entry costs as in each sleeve's sim (EXH / DIST: model slip ≥ 0.10 + 0.05 taker; FLATBASE: 0.10 slip in the
  fill + 0.05; WILLY / HT: 0.10 slip in E + 0.05); real funding to the early exit time where the sleeve's sim has it.

## Splits
- EXH, DIST, FLATBASE: DESIGN 2023-07-01 → 2025-06-30 · HOLDOUT 2025-07-01 → 2026-11-20.
- WILLY, HT (history starts 2025-01): DESIGN 2025-01-01 → 2025-12-31 · HOLDOUT 2026-01-01 → 2026-11-20.
- HOLDOUT is read once, for the cell picked on DESIGN (the full HOLDOUT grid is shown afterwards for transparency only).

## Pick (DESIGN only)
Cell with the largest DESIGN mean Δ per fill (all fills, Δ = 0 where no trigger) among cells with ≥ 30 DESIGN triggers
(≥ 15 for WILLY / HT). If no cell has DESIGN Δ > 0 → verdict REJECT for that sleeve (HOLDOUT still shown).

## Pass bar (all must hold, per sleeve)
1. HOLDOUT mean Δ > 0 with day-block bootstrap (10-day blocks, 4,000 draws) 95 % CI lower bound > 0.
2. DESIGN mean Δ > 0.
3. DESIGN best t beats the family-wise null: day-level sign-flip over all 33 cells jointly (same flip for every cell), max-t 95th pct.
4. ≥ 2/3 of half-years (all half-years of the sleeve) with Δ > 0.
5. Top pair < 50 % of the full-span positive Δ; leave-one-pair-out keeps Δ > 0.
6. Real-fill check (master stack_keep + paper exports, priced on 1m klines) not contradicting (Δ not negative when ≥ 5 triggers; fewer = "insufficient").
7. Reported Δ on DESIGN takes a 30–50 % haircut.

## Reads (descriptive, every sleeve)
Path decomposition of triggered fills: REF outcome (full TP = cut winner · stop = saved · time / other) with $ (Σ net %) for REF vs cell;
random-entry control (same rule on random entries in the same coins / days, REF exit simulated on them; EXH / DIST reuse the RND sets of
EXH_EXIT2 / DIST_EXIT; FLATBASE / WILLY / HT: 3 random 5m times within ±12 h of each entry, same pair); per-pair concentration;
leave-one-pair-out; half-year table; N fills · distinct days; one-size (0.61) DCR before → after for the picked cell; event-study chart.
