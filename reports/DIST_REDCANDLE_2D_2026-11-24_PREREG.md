# DIST red-candle 2D — PRE-REGISTRATION (written 2026-11-24, before any 2D number was computed)

**Question (operator):** "What about a 2D variable, big red + something? Use theory first."
**Prior:** DL 602 screened every 2D cell blindly (42,519 cells, family-wise null) → nothing passed. So this is NOT another blind screen: a
small family of theory-driven pairs, direction fixed here, cuts frozen on DESIGN, read once on HOLDOUT.

**Cohort:** today's DIST rule in the backtest = reports/DIST_WL_2026-11-18_trades.csv.gz, in_LIVE ∧ rsi1h ≥ 74.07 (1,536 trades), outcome
net_LIVE (Exit B, live costs). DESIGN < 2025-07-01, HOLDOUT after. Red-candle axis: bar_ret_pct < −3.0 = BIG (the cohort median, frozen).

**Theory of the trade:** DIST shorts a FRENZY coin on a red volume candle while top traders cut longs. It pays when the candle is real
distribution (holders leaving) and nothing is left to push price back up; it fails when the drop invites a squeeze (fresh shorts,
absorbed selling, a strong market). A big red candle is the start of the unwind — what decides whether it continues:

| # | theory | variable (backtest column) | predicted GOOD side (for the short) |
|---|---|---|---|
| T1 | sellers in control: the candle closes at its low, no buyers absorbing | close_pos15 (close in the 15m range) | LOW |
| T2 | no absorbing lower wick | lowwick15 | LOW |
| T3 | holders leaving, not shorts opening: open interest falls through the drop (less squeeze fuel) | oi_chg30m | LOW (falling) |
| T4 | the crowd is not already short (no squeeze fuel) | fund_last (funding) | HIGH (≥ less negative) |
| T5 | smart money still exiting longs | top_pos_chg4h | LOW (falling faster) |
| T6 | room to fall: far above the pre-pump base | dist_base_pct | HIGH |
| T7 | no market tide lifting it: BTC weak | btc_ret4h | LOW |
| T8 | real volume, committed selling | vmult15 | HIGH |

**Test:** for each Tk, the cut = the variable's median on the DESIGN big-red trades (frozen); 4 quadrants (BIG / SMALL × good / bad side).
Primary read per Tk: inside BIG, good side − bad side on HOLDOUT (predicted > 0). Secondary: inside SMALL, the bad side's mean (a
loser cell = a filter candidate only if it loses). Family-wise null: 2,000 shuffles of net within the cohort, max of the 8 primary
HOLDOUT t-stats (one-sided, predicted direction). Day-block CI, half-years, both BTC daily regimes for anything passing.
**Ship bars unchanged:** a filter needs the blocked cell to clear the locked EXPECTANCY bar (below breakeven WR, mean < 0 at 95 % by a
window bootstrap, ≥ 8 windows, N ≥ 15); a size-up needs the Pattern W bar. Anything below → observe line at most.
**Batch:** 7 real fills (2 BIG) — shown, cannot decide.

**Addendum (2026-11-24, after the run, from the reviews — no rule changed):** the family-wise null shuffles net within the HOLDOUT big-red trades
(the primary read's own sample), not the whole cohort as worded above. close_pos15 and lowwick15 are the same column in this data → T1 ≡ T2, so the
family is 7 distinct tests (the max-t null is unaffected).
