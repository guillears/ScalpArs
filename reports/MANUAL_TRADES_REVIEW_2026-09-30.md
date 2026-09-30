# Manual trades review — B15 evening, 2026-09-29 21:38 → 2026-09-30 00:18 UTC

Source: `MANUAL_TRADES_B15_orders_2026-09-30.csv` (18 closed MANUAL fills, Momentum exit). Entry readings rebuilt from
Binance 5m klines (closed bars + the entry bar cut at the fill price) in `MANUAL_TRADES_B15_rebuilt_2026-09-30.csv`;
checked against the 10 fills that carry `manual_*` stamps (RSI within ~1–5 pts, gap 5-20 magnitude within ~0.03).
N=18, one evening, ~3 market windows → watchlist only. No rule from this.

| Cohort | N | W/L | Net $ |
|---|---|---|---|
| All | 18 | 9/9 | −99.7 (fees $478; gross +$378.5) |
| QNT (3 fills) | 3 | 3/0 | +677.8 |
| Everything else | 15 | 6/9 | −777.5 |
| Longs 23:07–23:18 UTC, BTC RSI 68–75, through BTC_SLOPE_GATE / BTC_RSI_ADX_CROSS (one window) | 4 | 0/4 | −472.9 |
| Longs with EMA5 below EMA20 (early turn) | 4 | 1/3 | −406.5 |
| Longs with EMA5 above EMA20 | 11 | 7/4 | +424.0 |

- Avg win +0.34 %, avg loss −0.58 % → breakeven WR 63 %; actual 50 %.
- 6 of 9 losers peaked ≤ +0.12 % (never went green); winners peaked +0.69 % on average and exited at +0.34 %.
- High pair ATR (≥1 %) longs 4/0 +$710 — but 3 of the 4 are QNT: pair concentration, not a dimension.
- Early-turn longs: on all four EMA5 was ALSO below EMA8. The Top Pairs "Gap 5-8" column is absolute (always
  positive); "Gap 5-20" is signed. None of the 18 met the operator's full signature (gap 5-20 < 0 and rising, gap 5-8 > 0,
  RSI rising, ADX rising); NEAR was closest (+$10.7).
