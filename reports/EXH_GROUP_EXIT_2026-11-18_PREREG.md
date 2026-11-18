# EXHAUSTION_SHORT group-lift exit — PRE-REGISTRATION

Frozen: **2026-11-18 14:18 UTC**, before any rule outcome was computed. Research only; nothing in the bot changes.
Only facts seen before freezing: the B|tp10|S2|24 cell stats already in `eval_grid.pkl` (parity target) and the unconditional
distribution of the alt-perp median 1 h return (no trade outcomes): ≥ 1 % in 5.1 % of hours, ≥ 2 % in 0.9 %.

## Question
EXHAUSTION shorts lose in clusters (2024-11-16: 13 of 14 stopped; 2026-01-24: 13/13). The entry cannot tell those days apart
(`EXH_BADDAY_2026-11-18.md`). Can an exit that watches the OPEN book after entry — several shorts being lifted together — cut the
cluster losses early and raise the compound rate?

## Theory (written first)
- A short fails when new buying keeps arriving after entry. On cluster days that buying is market-wide (alt rotation): it lifts every
  open Frenzy short at once. A single short going against us is mostly noise (the stop is 1.5×ATR for a reason, and earlier stop-width
  studies found tighter stops lose). Several shorts going against us at the same minute is a different event: a common cause.
- If the common cause persists (rotation days trend), the shorts that are lifted together go on to hit the full stop more often than a
  lone underwater short does. Cutting them early saves (full stop − cut level − extra costs).
- Why it could fail: (1) Frenzy coins mean-revert hard; many lifted shorts later fall to TP (the payoff is asymmetric: TP 10 % vs stop ≤ 6 %),
  so each cut that would have recovered costs ≈ 10 + cut level. (2) A cut is a taker market order into a squeeze (slippage 0.1–0.6 %).
  (3) The signal is late: by the time K shorts are each L % against, most of the stop distance is already spent.
- Prediction if the theory is right: SAVED > COST on the touched trades in most half-years, the effect is concentrated on cluster days,
  and the group rule beats the same cut applied to lone shorts (control G0).

## Data and parity
- Candidates: `reports/EXH_EXIT_GRID_data/eval_grid.pkl` C (4,140) with the cell **B|tp10|S2|24** outcome (gross r, exit kind, exit time,
  funding) from `sim_grid.pkl`. Parity: one position per pair, no cap → N 3,998, avg net 0.647 % (model slip), halves as in `R`.
- 1m closes for every candidate from `reports/EXH_EXIT_GRID_data/k1m` (same slices as the grid sim). Holdout E (2023-H1, 237 trades) from
  `sim_gridE.pkl` / `E/` k1m if present.
- Market proxy: alt-perp median 1 h return, hourly closes from the 5m caches (`scratchpad/sq/hourly_px.pkl`), known at the hour close.

## Book simulation
Entries in time order. One position per pair. Book cap **4** and **10** (EXH-only book). A candidate is skipped when its pair is open or
the book is full (or, G4, while the pause is active). A position is open at the close of minute m if entry minute ≤ m < its natural exit
bar. Natural exits (stop / TP / 24 h) happen inside their bar before the group check. The group check runs at every 1m close on the open
book only. A cut fills at that minute's close, taker. Cuts free the slot (and the pair) from the next minute.

## Rules (family = 19 rules × 2 books = 38 cells, 2 void: K = 5 at cap 4 → 36 live cells)
Unrealised u = −(close / entry − 1) × 100 (short; negative = against us).
- **G1-ALL K·L**: at a minute close, ≥ K open shorts each have u ≤ −L → close ALL open EXH shorts. K ∈ {3, 4, 5}, L ∈ {1.5, 2.5} (6).
- **G1-HIT K·L**: same trigger → close only the shorts with u ≤ −L. K ∈ {3, 4}, L ∈ {1.5, 2.5} (4).
- **G2 M·K**: ≥ K open and the mean u of the open book ≤ −M → close all. M ∈ {2, 3}, K ∈ {3, 4} (4).
- **G3 X**: at an hour close the alt-perp median 1 h return ≥ X % and ≥ 2 open shorts have u < 0 → close the ones with u < 0.
  X ∈ {1, 2} (2). (No hourly data before 2023-05 → G3 not read on E.)
- **G4 pause on S**: no new EXH entry while signal S is active at the last closed minute; no cut. S ∈ {G1 K3·L1.5, G1 K3·L2.5, G2 M2·K3} (3).
  Distinct from the refuted "pause after X losses": that read CLOSED trades; this reads the live open book.
- **Control G0 L** (not in the family): cut ANY single short at u ≤ −L, L ∈ {1.5, 2.5}. Tells whether "together" adds anything beyond
  a tighter stop.

## Costs
Entry and every taker exit: fee 0.05 % + slippage 0.10 %; maker TP: fee 0.02 %, no slippage; funding as recorded to the exit (to the
cut minute for cut trades). Cut slippage 0.10 % (base), 0.20 % (stress), 0.40 % (squeeze stress). Baseline exits keep 0.10 %.

## Reads (binding)
1. Per cell, per book: N · WR · avg net % · DCR at ONE size 0.183 × equity per trade (0.61 reference) · max DD · worst day · worst 7 days;
   before → after; DESIGN 2023-07-01 → 2026-06-30, HOLDOUT 2026-07-01 → 2026-11-18 (read once), E.
2. Two-sided split on the cut trades vs their own natural outcome: SAVED (natural net < cut net: full stop, or a time exit lower) vs
   COST (natural net > cut net: TP, or a smaller loss / gain). Book side effects listed apart: trades added (freed slots) and trades
   removed (pair / slot / pause). Trades never touched must show Δ = 0 (asserted).
3. Δ daily return (rule − baseline) per half-year, day-block bootstrap 95 % CI (2,000 resamples of days).
4. The 15 worst cluster days of the baseline (stops avoided, $ saved) and the 15 best days (winners cut).
5. Selection: the best cell = highest DESIGN ΔDCR at cut slip 0.10. Family-wise null: 200 shuffles that re-assign every candidate's
   path to another candidate's entry slot within the same half-year (keeps the schedule, the slot count and each path; destroys the
   co-movement the rule reads). A day-block shuffle would keep same-day co-movement and so cannot be the null for a co-movement rule.
   p = share of shuffles whose family max DESIGN ΔDCR ≥ the observed best. Plateau: the best cell's neighbours (K ± 1, L / M / X next
   value, other book) must have the same sign.

## Verdict bar (frozen)
- **BUILD** only if all: DESIGN ΔDCR > 0 in BOTH books at cut slip 0.20; ≥ 5 of 6 design half-years Δ ≥ 0; day-block CI of the DESIGN
  Δ excludes 0; family null p < 0.05; plateau holds; HOLDOUT Δ ≥ 0; the group rule beats control G0 at the same L; SAVED $ > COST $.
- **OBSERVE** if DESIGN Δ > 0 and SAVED > COST but the null / CI / holdout / plateau fails: log the signal live, no action.
- **NO** otherwise.
