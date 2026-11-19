# WILLY_HT winners vs losers after the DL 594 dip cut — PRE-REGISTRATION

Frozen 2026-11-19T17:10Z, before any variable was read against outcomes (only the per-fill net / keep / off_peak list of the
input file had been seen, as given by the coordinator). Research only — no engine / config change.

## Cohorts
- **A** = the 44 real master FRENZY_WILLY_HT fills with off_peak ≤ −1.4 % (DL 594 dip cut), re-priced on today's exit
  (+5 % net TP / −4 % net stop / 24 h). Outcome `win` = net > 0 (all 44 are ±: 24 TP, 20 stops).
- **B** = the 12 of A kept by FAILED_SPIKE (DL 527). 10 W / 2 L → no split can reach ≥ 8 per side: descriptive only
  (ranks of the 2 losers on every variable), no test claimed.
- **C** = current batch fills since the DL 587 deploy (2026-11-18 23:06 UTC), FRENZY_WILLY_HT, dip-cut: reported apart, not pooled.

## Targets
1. `win` (booked, primary for the operator question) · 2. `WORKED` = +1 % our way before the −4 % net stop within 24 h
   (entry label, exit-independent, from 1m klines) · 3. `REACH` (best % before the stop / 24 h) — Spearman only.

## Variable family (every one listed; coverage table first)
- Every `entry_*` column in reports/MASTER_POOL_stacked.csv stamped on ≥ 80 % of A (numeric with > 3 distinct values →
  threshold splits; ≤ 6 levels → level-vs-rest). This includes every dashboard "Entry Conditions … Winners vs Losers" column
  (RSI, ADX, ADX/RSI dir, gaps, stretch, EMA20 slope, BTC slope/RSI/ADX/dirs, range pos, breadth, ATR %, BTC/pair trend, global vol,
  pair vol $, ext %, BTC ext %, 1 h slope, alt corr / beta, up4h) — those under 80 % are listed as blind spots, not dropped.
- Derived at entry time: off_peak (dip below spike high), spike age (min from spike to entry), hour UTC, weekday, FAILED_SPIKE flag
  (A only — in-sample by construction), pair repeat (n prior HT fills on the same pair in the cohort), BTC 1d EMA50 − EMA200 gap %
  (daily klines, last CLOSED day), and from 1m klines CLOSED before the entry minute: ret 15 m / 60 m / 240 m, 60 m drawdown from
  the 60 m high, 60 m realised vol (std of 1m returns), volume last 15 m ÷ prior 60 m per-minute average.
- Post-entry columns (peak, trough, duration, exit trend) are descriptive only, never tested.

## Tests (1D; 2D only if both quadrants reach the floor)
- Splits: sign (only where both signs exist), median, outer terciles (top vs bottom third); level vs rest.
- Floor: ≥ 8 fills AND ≥ 5 distinct entry days on EACH side; regime / market-wide variables (btc_*, breadth, bull/bear %, global
  volume, mkt up4h, alt corr/beta, macro trend, BTC 1d gap) are also counted in DAY units and the per-day concentration is shown.
- Statistic: two-sided Fisher exact p on WR; "nominal separator" = p < 0.05. Numeric: also Mann-Whitney W vs L.
- **Family-wise null:** 2,000 shuffles of the win labels over the 44 (and a day-block variant: whole days' labels permuted
  together); report observed count of nominal separators and the observed min p vs the null distribution (P(null ≥ count),
  P(null min p ≤ observed min p)). A variable "survives the master read" only if its min-p beats the family-wise null at 5 %.
- 2D: every pair of the 1D candidates × median quadrants, same floor per quadrant vs rest, counted in the same null.

## Survivor path (only if anything survives)
- Backtest cross-check on reports/WILLY_HT_THEORY_2026-11-19_trades.csv after off_peak ≤ −1.4: DESIGN = entries before 2026-07-01,
  SEEN = 2026-07-01 onward; same direction required in both; unmapped variables = "cannot cross-check" caveat.
- Pricing before → after on A: N · WR · avg · sum · stops · one-size DCR (0.61 × equity per fill, all calendar days of the span) ·
  LOO by coin and by day · for a block, the expectancy filter bar (WR < breakeven WR, window-bootstrap 95 % avg < 0, ≥ 8 windows,
  N ≥ 15), then 30–50 % haircut.

## Scenario question (answered whatever the screen finds)
- 44 (dip cut only) vs 12 (dip cut + FAILED_SPIKE) on the master: sum, stops, worst day, one-size DCR, Kelly at the one-sided 95 %
  Wilson WR floor (fills and days), with the in-sample note for FAILED_SPIKE.
- Backtest (operator addition): HT cohort with the dip cut WITH vs WITHOUT FAILED_SPIKE (rebuild episodes, today's exit,
  portfolio 10 slots / 1 per pair): trades/day, WR, avg, sum, stops, one-size DCR, per half-year and DESIGN / SEEN.
