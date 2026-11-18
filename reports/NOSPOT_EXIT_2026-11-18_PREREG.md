# NO-SPOT EXIT STUDY — pre-registration (frozen 2026-11-18 22:12 UTC)

Frozen BEFORE any result of this study. No exit cell of this study has been simulated. Known inputs (from NOSPOT_SLEEVE_2026-11-18, 5m
paths): NS0 / R1 / R4 at the six old exits (XE NS0 −0.07, F1012 NS0 +0.18, R1 F1012 −0.10, R4 F1510 +0.20 %/trade full span), ~80 % of
shorts reach +1 % first, 60–85 % of the loss is GAVE_BACK (median reach +2.6 to +3.8 %), median path ≈ +4 % at 24 h vs mean ≈ +2 %.
Those known facts motivated the family below; that is disclosed, and it is why the family is small, the grid is stated now, and HOLDOUT is
read once. The only thing running at freeze time is the 1m data download (no P&L computed). Research only, BACKTEST only, no engine /
config / test change, no commit.

## Question (operator)
Can locking in profit (fixed TP, break-even, profit lock, trailing stop, time exit) turn short entries on perp-only (no-spot) Frenzy coins
into a positive sleeve?

## Entry cohorts (frozen, no new entry search — the NOSPOT_SLEEVE rows `reports/NOSPOT_SLEEVE_data/X.pkl` and masks in
`scripts/nospot_sleeve_lib.py`, unchanged)
- **NS0** = the EXHAUSTION trigger on no-spot rows (15m RSI14 cross 70 → ≤ 70 ∧ vol_decay ≥ 0.6366).
- **R1** OI_UNWIND primary (oi_chg1h ≤ −3 % ∧ dd_hi ≤ 15 % ∧ ret1h ≤ 0).
- **R4** TOP_DISTRIB primary (topP 1 h change ≤ −5 % ∧ glA 1 h change ≥ 0 ∧ dd_hi ≤ 20 %).
- **RND** (control, not selectable) = every eligible 15m-close bar in the episodes where NS0, R1 or R4 fires (each bar one trade). Plus a
  MATCHED random control per cohort cell = the shuffled-null mean (each trade's entry replaced by a uniform random eligible bar of the
  same episode).
- One position per pair, re-applied per (cohort, exit cell) with that cell's exit time. Entry = the 5m open after the 15m close (= the 1m
  open at that minute).

## Exit family (30 selectable cells + the reference; all short, hold 24 h unless stated)
Fixed TP = a resting MAKER order, fills only when a 1m low trades ≥ 1 tick THROUGH it. Every stop (initial, break-even, lock, trail) and
every time exit is a MARKET (taker) order.
- **REF  XE** = the shipped EXHAUSTION exit: TP min(20 × ATR15 %, 10 %) / SL min(1.5 × ATR15 %, 6 %) / 24 h. Reference, not selectable.
- **A. Fixed TP × SL (12):** TP {2, 3, 4, 5} % × SL {6, 10, 12} %. Plateau grid = this 4 × 3 grid.
- **B. Break-even (3):** SL 10 %, TP 10 % maker; after the best low reaches +{2, 3, 4} % our way the stop moves to entry − round-trip
  costs (E0 × (1 − (0.10 % fees + 2 × slip))), i.e. ≈ 0 net. Plateau = arm {2, 3, 4}.
- **C. Profit lock (3):** SL 10 %, TP 10 % maker; after +3 % → stop at +1 %; after +4 % → stop at +2 %; after +5 % → stop at +3 %
  (one cell each: L31, L42, L53). Plateau = the 3 cells.
- **D. Trailing stop (8):** SL 10 %, no TP; once the best low reaches +a % our way the stop = best low × (1 + g); a {2, 3, 4} × g {1.5,
  3} % (6) + a 3 with g = {1, 2} × ATR15 % (2). Plateau = the 3 × 2 grid (+ the two ATR cells as a second plateau pair).
- **T. Time exit (4):** SL 10 %, no TP, exit at the close after {4, 12, 24, 48} h. Plateau = the 4 holds.
- Partial take profit: excluded (operator).
- Post-selection robustness (descriptive, never selectable): the winner with its stop at 6 / 12 % and with a 48 h hold.

## Realism (fixed)
- **1m paths** (data.binance.vision daily 1m zips), 48 h from entry. A trade whose 1m days are missing falls back to the 5m path
  (flagged, share reported). Never 5m checkpoints.
- **Intrabar order:** the stop as it stood at the bar OPEN is checked first on the bar HIGH; then the TP on the LOW (≥ 1 tick through);
  both touched in one bar → stop. Stops fill at max(stop, bar open) (gap-aware). The best low (for BE / lock / trail) updates AFTER the
  bar's checks; the new level applies from the next bar.
- **Costs:** taker 0.05 % per taker leg, maker 0.02 % on a TP fill. Slip per taker leg (entry, every stop / lock / BE / trail / time
  exit) = max(0.10 %, −0.042 + 0.276 · √(6,100 ÷ Q5)) (the NOSPOT / SIZING model at ≈ $6k notional; = the `slip` column). Maker TP: no
  slip. Real funding at every settlement between entry and exit (hourly when Binance switched the coin); a missing pair-month is charged
  0.05 %/8 h against. Stresses: flat 0.20 % per taker leg; and 2 × slip on the dynamic (BE / lock / trail) exits only.

## Splits
DESIGN 2023-07-01 → 2025-07-01 · HOLDOUT 2025-07-01 → 2026-11-18 (read ONCE, only for the survivors + the reference; all other HOLDOUT
numbers are shown only as descriptive after the decision). Half-years 23-H2 … 26-H2.

## Selection on DESIGN (survivor = all of; per cohort NS0 / R1 / R4 × 30 cells = 90 cells)
N ≥ 30 · avg net > 0 · avg > the same cohort's REF XE avg (DESIGN) · ≥ 2/3 of DESIGN half-years with N ≥ 10 positive · plateau: ≥ 2/3
of the cell's immediate grid neighbours (same family, same cohort) positive on DESIGN. Rank survivors by DESIGN avg; the top 3 (at most
one per cohort × family) go to HOLDOUT.

## Ship bar (all, on HOLDOUT unless stated)
1. HOLDOUT avg > 0 and > the same cohort's REF XE HOLDOUT avg.
2. ≥ 2/3 of all half-years (N ≥ 10) positive (full span).
3. HOLDOUT avg > 0 at flat 0.20 % slip per taker leg.
4. HOLDOUT best-5 %-out avg > 0.
5. Kelly at the one-sided 95 % Wilson WR floor > 0, per trade (n = trades) AND per day (n = trade days, day P&L).
6. Family-wise shuffled null p < 0.05 (full span): each trade's entry → a uniform random eligible bar of the same episode (same exit,
   one-position not re-applied), 500 draws; statistic = cell avg − its null mean; max over the 90 cells.
7. No pair and no day ≥ 50 % of the HOLDOUT net profit.
8. ≥ 8 distinct windows (trade days) on HOLDOUT.
9. One-size DCR (0.61 × equity per fill, every calendar day, P&L on the exit day) > 0 on HOLDOUT, AND EXH + DIST + cell DCR > EXH + DIST
   DCR (HOLDOUT; EXH = BASELINE_TRADES_EXHAUSTION, DIST = DIST_WL RSI1h ≥ 74.07 LIVE exit, as in NOSPOT_SLEEVE `_joint`).
Real funding (by construction). Verdict: **SHIP-candidate** = all 9 · **OBSERVE** = items 1 and ≥ 7 of 9 · else **NOTHING**. 30–50 %
in-sample haircut on any projected Δ.

## Reads (descriptive, no re-fit)
- REF XE on 1m vs the 5m NOSPOT_SLEEVE numbers (parity).
- Path decomposition on 1m (WORKED = +1 % before the cell's initial stop distance; NEVER_WORKED / STOPPED_RECOVERED (+1 % within 120 min
  after a stop) / GAVE_BACK / DRIFT_LOSS / LEFT_ON_TABLE / CAPTURED) before (REF) and after (each survivor).
- Saved / deeper (EXIT/STOP counterfactual rule), on the SAME entries (REF's one-position set): ① the mechanism vs its own base (same SL
  / TP / hold, mechanism off): fills whose path never reached the arm level must show Δ = 0 (asserted); armed fills split into saved
  (base lost) and cost (base did better); ② the candidate vs REF split by REF's outcome (REF stopped → saved / deeper; REF TP; REF time).
- Entry × exit joint check: the same Δ (cell − REF) on RND and on the matched random entries; a pure exit effect shows on random too.
- Leave-one-out (pair) and best-5 %-out; per half-year; BTC R1 (daily EMA50 vs EMA200 sign) and R4 (5m label sign); day-block bootstrap
  95 % CI (block 10 days, 2,000) of the HOLDOUT avg; trades/day, WR, avg win / loss, exit mix.
- DCR at 0.61, max DD, worst day, the joint add to EXH + DIST.
- EXHAUSTION spot-coin cross-check: REF vs the best no-spot exit on BASELINE_TRADES_EXHAUSTION trades (1m, EXH_EXIT_GRID cache) —
  reported, not decided on.
- Replays: PIPPIN, BLESS, JELLYJELLY (11-10 → 11-18) and PHAROS 11-18 if any entry fires (NOSPOT_SLEEVE replay entries).

## ADDENDUM S (frozen 2026-11-18 22:24 UTC — operator request via coordinator; before ANY sim of this study had run: 0 sim files exist)
- **Family S, small TP (16 cells, selectable, same selection rule and ship bar):** TP {1.0, 1.5} % (resting maker, fills only when a 1m
  low trades ≥ 1 tick through) × SL {2, 3, 4, 6} % (market, thin-coin slip) × max hold {4, 24} h (time exit = market). Same-bar
  ambiguity → stop first. Plateau = immediate neighbours in the 2 × 4 × 2 grid (TP ±1, SL ±1, hold ±1 step).
- The family-wise null now covers 3 cohorts × 46 selectable cells = 138 cells. Survivor picks: still the top 3 by DESIGN avg, at most
  one per cohort × family.
- Extra reads for S (descriptive): breakeven WR (|avg loss| ÷ (avg win + |avg loss|)) vs observed WR, trades/day and one-size DCR at
  0.61 for every S cell (DESIGN and, after the decision, HOLDOUT); loser clustering = stops per day (mean / p99 / max), the same-window
  stop count (stops whose exits fall in the same UTC hour across pairs) and the worst day at 0.61; the +1 % hit rate (TP 1 % filled
  before the stop / within the hold) per entry cohort AND on the random-entry control.
