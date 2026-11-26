# FLATBASE_SHORT entry-timing study — PRE-REGISTRATION (frozen 2026-11-26, before any variant result was computed)

Operator question (11-26): is today's entry timing right, or can ONE variable move the entry so the trade is shorter and spends more time in profit?

## Cohort (fixed — the variants change only the entry time / price, never which episodes qualify)
- Backtest: the 545 OOS_REAL_FLATBASE rebuild trades on today's stack (HIGHER_HIGH + NEAR_BASE > 2.09 + FUNDING ≤ 0.005 %), same as FLATBASE_BASE_TREND_2026-11-26. The filters are read at the confirm close (as live).
- Real: master 17 kept fills (KEPT_LIST_FLATBASE_2026-11-21, stack_pct) + current batch 3 (KERNEL / MMT / PIPPIN). Priced on 5m bars by the same machinery (LIVE-sim) so variant − LIVE is like-for-like; booked stack_pct shown beside.
- Blocks: DESIGN (entry < 2025-07-01) / HOLDOUT; half-years; master; batch.

## Reference point T0 = the confirm close (the live fill moment). C = confirm close price. 1h ATR % = Wilder-14 ATR of 1h bars closed by T0, % of close.
## Variants (10, all counted in the pick-the-best null)
| id | rule | order |
|---|---|---|
| LIVE | market at T0 (today) | taker + entry slip |
| MK0_4h | resting sell limit at C, 4 h | maker |
| ATR05_12h | resting sell limit at C × (1 + 0.5 × ATR1h %), 12 h | maker |
| ATR1_12h | limit at C × (1 + 1.0 × ATR1h %), 12 h | maker |
| ATR2_12h | limit at C × (1 + 2.0 × ATR1h %), 12 h | maker |
| ATR1_24h | limit at C × (1 + 1.0 × ATR1h %), 24 h | maker |
| VWAP_24h | limit at max(spike-anchored VWAP at T0, C), 24 h | maker |
| C1H | market at the open after the first 1h close < C within 24 h | taker + slip |
| C4H | market at the open after the next 4h close < C within 24 h | taker + slip |
| LOW1H | market at the open after the first 1h close < the confirm 4h candle's low within 24 h | taker + slip |
| DELAY4 | market at T0 + 4 h, unconditional | taker + slip |

- Limit fill = a 5m bar with high ≥ limit × 1.0005 (trade-through, queue-safe); fill at the limit; the fill bar can stop (high ≥ stop level) but cannot take profit; walking from the next bar. Unfilled / no trigger = NO TRADE (counted, contributes 0).
- Exit: today's −5 % TP (maker) / +10 % stop (taker, slip = max(row slip, 0.10 %)) / 72 h time exit (taker + row slip), price levels from each variant's own entry. Real funding over the hold (missing = −0.10 %/8 h). DL 607 funding-in-stop not re-walked (all variants alike).
- Fees: taker 0.045 %, maker 0.018 % per side.

## Metrics
N · lost (no fill / no trigger) · WR · avg net % · Σ · avg hold h · % of held 5m bars with gross MTM > 0 · one-size DCR (X 0.61; every calendar day of the block) · max DD · worst day · Σ net per position-day (slot-bound turnover) · paired Δ per episode vs LIVE (lost = −LIVE net) with day-block bootstrap 95 % CI · Δ on kept trades only.

## Decision rule (frozen)
- **Candidate:** master Δ/episode > 0 AND backtest Δ/episode > 0 with day-block CI lower bound > 0 AND HOLDOUT Δ > 0 AND ≥ 3/5 half-years positive AND pick-the-best null p < 0.05 AND one-size DCR higher on master AND backtest.
- **Observe line (frozen definition):** master Δ ≥ 0 AND backtest Δ ≥ 0 but any other candidate item fails.
- **Keep today:** otherwise. A shorter hold alone is not a win: it counts only if DCR / Σ per position-day rises (the book is slot-bound).
