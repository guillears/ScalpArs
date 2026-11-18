# EXH_EXIT2 — the exits EXH_EXIT_GRID did not test — PRE-REGISTRATION (frozen 2026-11-18 22:31 UTC)

Frozen BEFORE any result of this study. No cell of this study has been simulated; no 1m path has been read for it.
BACKTEST only, research only: no engine / config / test change, no commit.

## What was already known (disclosed)
- EXH_EXIT_GRID (`reports/EXH_EXIT_GRID_2026-11-18.md`, 396 cells, pre-DL 582 entries, old shipped exit as reference) tested:
  ATR TPs, fixed TPs 4–10 %, stops 4 / 5 / 6 % and min(1.5 ATR, 5 / 6), trailing stops armed at +2…+5 % with 1 / 1.5 / 2 / 3 % or
  0.5 / 1 × ATR givebacks (with and without a 10 / 15 % maker cap, incl. today's stop S2), and 12 / 48 h holds for its top cells
  (incl. today's exit B|tp10|S2: 12 h +0.57 · 24 h +0.65 · 48 h +0.71). Trailing lost in every cell. After a 5 % TP the price fell a
  median +7.1 % further. These are REUSED, not repeated.
- NOT tested there (this study): break-even moves · profit locks · trailing armed LATER than +5 % (so it cannot cut a trade before
  the +10 % TP) and trailing with no TP above +8 % · small fixed TPs 1 / 1.5 / 2 / 3 % with tight stops 2 / 3 / 4 % · time exits
  shorter than 12 h · a no-progress exit.
- Today's exit (DL 579) and the baseline (`BASELINE_TRADES_EXHAUSTION_2026-11-18.csv`, kept rows): WR 42 %, TP 34 / SL 57 / time 9 %.
- Parallel studies frozen with the same engine rules: NOSPOT_EXIT (22:12 UTC), DIST_EXIT (22:27 UTC). Neither's results were read.

## Cohorts (entries frozen — no entry search)
- **PRIMARY `EXH`** = `BASELINE_TRADES_EXHAUSTION_2026-11-18.csv` rows with `dl582_block` False (3,368 fills, 2023-07 → 2026-11-18),
  entry = the 1m open at entry_ms, ATR15 = `atrp`, q5 = `q5`. **Paired on the same entries for every cell** (the one-position set
  of today's exit is kept; a shorter exit that would free the pair earlier does not add entries — stated caveat, conservative for
  short exits; no cell holds longer than 24 h, so no cell blocks an entry the reference took).
- **E (2023-01 → 2023-06, clean, built by EXH_EXIT_GRID)** = `EXH_EXIT_GRID_data/cand_E.pkl` + the DL 582 block (ssc > −0.0366 ∧
  oi_vol > 0.128, OI from data.binance.vision metrics at the last 5m row ≤ entry ÷ the 24 h futures quote volume from the 5m bars;
  missing OI → not blocked), one position per pair under today's exit. Read ONCE, reported, not used for selection. If the metrics
  cannot be fetched, E is reported without the block (flagged).
- **CONTROL `RND`** = random entries in the same coins / episodes: for every PRIMARY fill, 3 entries drawn uniformly (seed 20261118)
  from the 5m bar opens of the same pair inside its Frenzy episode window [shortlist_ms, off_ms] (`EXHAUSTION_EXPAND_data/ep_timing.csv`,
  the episode containing the fill) ∩ [entry − 24 h, entry + 24 h], excluding the real entry bar. Each draw = one trade, the fill's own
  ATR15 and q5. Not selectable. Read: the same paired Δ (cell − REF) on RND. Δ on RND too = a generic exit effect.

## Exit family (39 selectable cells + REF; short; 24 h hold unless stated)
"ATR stop" = min(1.5 × ATR15 %, 6 %); "XE-TP" = min(20 × ATR15 %, 10 %) resting maker. Every dynamic stop = MARKET.
- **REF** = XE-TP / ATR stop / 24 h (today's exit). Not selectable.
- **B. Break-even (5):** REF + once the best low reaches +a % our way, stop → E0 × (1 − (0.10 % + 2 × slip)/100) (≈ 0 net).
  a ∈ {2, 3, 4, 5, 6}. Plateau = adjacent arms.
- **L. Profit lock (6):** REF + once +a %, stop → +(a − g) %. a ∈ {4, 6, 8} × g ∈ {2, 3} (locks +2/+1, +4/+3, +6/+5). Plateau = 3 × 2 grid.
- **D1. Late trail, TP kept (6):** REF + once +a %, stop = best low × (1 + g/100). a ∈ {6, 8} × g ∈ {2, 3, 4}. Plateau = 2 × 3 grid.
- **D2. Late trail, no TP (4):** ATR stop, no TP; once +a %, trail g. a ∈ {8, 10} × g ∈ {3, 5}. Plateau = 2 × 2 grid.
- **S. Small TP × tight stop (12):** TP {1, 1.5, 2, 3} % maker × fixed SL {2, 3, 4} % (market), 24 h. Plateau = 4 × 3 grid.
- **T. Short hold (3):** REF with a {4, 8, 16} h time exit (market at the close of the last bar). Plateau = adjacent holds.
- **NP. No-progress exit (3):** REF + at h ∈ {2, 4, 8} h after entry, if the best low has not yet reached +2 %, exit market at
  that bar's close. Plateau = adjacent h.
- Partial exits: excluded.

## Realism (fixed; the NOSPOT_EXIT engine `scripts/nospot_exit_sim.py` reused, extended only with per-cell ATR / fixed TP & SL modes
and the NP mechanism)
- 1m futures paths (data.binance.vision daily zips; EXH_EXIT_GRID cache + new days fetched there; 0 API weight). Missing 1m → 5m
  fallback, flagged and counted.
- Per bar: the stop as it stood at the bar OPEN checked first on the HIGH; then the maker TP on the LOW, filled only if the low trades
  ≥ 1 tick THROUGH; both in one bar → stop. Stops fill at max(stop, bar open) (gap-aware). The best low updates AFTER the bar's
  checks; BE / lock / trail levels apply from the next bar.
- Costs: taker 0.05 % per taker leg (entry, every stop / BE / lock / trail / time / NP exit); maker 0.02 % on a TP fill, no slip.
  Slip per taker leg = max(0.10 %, −0.042 + 0.276 · √(6,100 ÷ q5)) (the size model at 0.61 × $10k, 0.10 % floor). Real funding at
  every settlement between entry and exit (missing pair-month charged 0.05 %/8 h against). Stress: flat 0.20 % per taker leg.

## Splits
DESIGN = entries 2023-07-01 → 2025-07-01 · HOLDOUT = 2025-07-01 → 2026-11-18 (read ONCE, for the pick and REF; everything else on
HOLDOUT shown only as descriptive after the decision) · E 2023-H1 (clean, once). Half-years 23-H2 … 26-H2.

## Selection (DESIGN only)
Statistic = paired Δ avg net %/trade (cell − REF), same entries. Survivor: Δ > 0 · ≥ 3 of the 4 DESIGN half-years Δ > 0 ·
≥ 2/3 of its immediate plateau neighbours Δ > 0. Pick = the survivor with the best plateau-smoothed Δ (mean of itself + neighbours).
If there is no survivor, the verdict is KEEP today's exit, and HOLDOUT is shown descriptively.

## Ship bar (all must hold for SHIP-candidate)
1. HOLDOUT paired Δ > 0 with a day-block (10-day blocks, 2,000) bootstrap 95 % CI excluding 0.
2. Δ > 0 in ≥ 2/3 of the 7 half-years 23-H2 … 26-H2.
3. HOLDOUT Δ > 0 at flat 0.20 % slip per taker leg.
4. HOLDOUT best-5 %-out: the cell's avg with its best 5 % of trades removed > REF's avg with its best 5 % removed.
5. Kelly at the one-sided 95 % Wilson WR floor > 0 (per trade and per day) on HOLDOUT.
6. Family-wise null p_fw < 0.05: day-level sign-flip of the paired Δ (same flips across all 39 cells, 2,000 draws), max-T of the
   studentised day-mean Δ, full span.
7. No pair and no day ≥ 50 % of the HOLDOUT positive Δ sum.
8. Joint book: EXH(cell) + FB + L4 + DIST through `scripts/joint_short_sim.py` `sim()` (uncapped, 0.609 × equity, same rules): DCR ≥
   the same book with EXH(REF), full span and HOLDOUT.
9. Worst day and max DD at 0.61 one-size (EXH alone) and in the joint book are not worse than with REF.
Verdict: **SHIP-candidate** = all 9 · **OBSERVE** = HOLDOUT Δ > 0 and ≥ 6 of 9 · else **KEEP today's exit**. A 30–50 % in-sample
haircut on every projected Δ.

## Reads (descriptive, no re-fit)
- Path decomposition before (REF) → after (pick + every family's best), NOSPOT_EXIT labels (WORKED = +1 % before the stop distance).
  GAVE_BACK share of the REF-stopped fills and its median reach.
- Saved / deeper (EXIT/STOP CF rule) on the REF-stopped fills; mechanism vs its own base: fills whose path never reached the arm show
  Δ = 0 (asserted); the cost on REF's TP winners: how many +10 % winners the cell cuts short and the $ lost.
- The RND control Δ; per half-year; BTC daily EMA50 vs EMA200 sign (`btc_d50_200`) and the 5m regime label sign
  (`exh_expand_build` / NOSPOT R4 label where available); pair concentration and leave-one-out; WR, avg win / loss, exit mix;
  one-size DCR at 0.61, max DD, worst day, alone and joint; stops per day on the 10 worst days before vs after.
