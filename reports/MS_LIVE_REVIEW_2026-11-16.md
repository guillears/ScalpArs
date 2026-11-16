# Momentum SHORT — live vs master, winners vs losers (2026-11-16)
Operator: "the momentum shorts disaster in current batch vs master batch, winners vs losers? has to be related to market conditions too".
Data: live export 2026-11-16 12:09 (MOMENTUM SHORT, 3 fills) · master `MASTER_POOL_stacked.csv` MOMENTUM SHORT, stack_keep, full-size only (no probes): 49 fills, 30 days.
Registered watch lines for this sleeve: none (`scout_blocked_filters.LINES` has no MOMENTUM_SHORT entry; `watch_filters` has no MS line).

## 1. Live fills (11-16, one morning, ~1 market window)
| Pair | Open UTC | Notional | Lev | P&L % | $ | Path | BTC 1h slope | BTC RSI / 1h | BTC ADX | bear % | gvol | entry gap | ADX Δ | ATR % | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ONE | 05:18 | $43.8k | 10× (C1) | +0.28 | +$32 | winner (peak +0.54) | −0.07 | 36.8 / 44.8 | 20.2 | 89 | 1.05 | 0.65 | +1.59 | 0.47 | 2 |
| QNT | 07:19 | $44.0k | 20× | −0.89 | −$391 | NEVER_WORKED (peak 0.00) | −0.17 | 42.5 / 43.5 | 18.7 | 76 | 0.96 | 0.33 | +1.40 | 0.43 | 5 |
| LTC | 09:54 | $39.0k | 20× | −0.71 | −$278 | NEVER_WORKED (peak 0.00) | −0.15 | 41.9 / 41.4 | 31.4 | 72 | 0.76 | 0.36 | +0.87 | 0.31 | 5 |
Total 3 · 33 % · −$638. Both losers are entry-side (never went green), stopped at ordinary sizes for this sleeve (master average loser −0.65 %).

## 2. Master (today's filters, full-size)
49 · 78 % · +0.20 %/fill · breakeven WR 59.5 % (avg win +0.45 / avg loss −0.65) · one-size DCR +0.27 %/day (X = 4.4, today's live ticket).
Era split: **Jun–Aug 26 · 88 % · +0.37** vs **Sep–Nov 23 · 65 % · −0.00** → the sleeve has already been break-even in master for 2.5 months.
Winners vs losers medians are almost identical on every market variable (bear % 79 vs 80, BTC 1h slope −0.10 vs −0.04, BTC ADX 23.6 vs 23.6, BTC RSI 39.9 vs 36.3, gvol 0.80 vs 0.86).

Coverage: 41 numeric + 15 flag entry_* columns swept (pnl and path labels, 1D × 3 granularities, all 2D, 3D); blind spots (< 80 % scored): alt beta/corr, btc 1d ret / 4h gap / off-30d-high / r72 / eff72, cmc rank, mcap, signed gaps, pair 1h EMA gap, pair age.

## 3. Market conditions — do the live losers sit where master shorts lose?
- The live losers' market (BTC 1h falling, 72–76 % of coins bearish, BTC RSI ~42) is the **normal** MS environment — master winners trade in the same conditions.
- Sign granularity: BTC 1h falling 36 · 72 % · +0.16 vs rising 13 · 92 % · +0.29; BTC trend gap > 0: 6 · 67 % · +0.06. Shorts do slightly better when BTC is bouncing (shorting weakness into a bounce), worse when chasing a falling market, but the falling cohort is still positive → not a filter.
- Weak trend (entry gap ≤ 0.45, where LTC / QNT sit) is the **best** master cohort (17 · 82 % · +0.34) → refuted.
- Sweep top separators (both eras consistent): BTC 1h RSI higher = better, global volume higher = worse, quality score higher = worse. None gives a negative blocked cohort.

| Candidate cut | Blocked (master) | P(mean<0) day-boot | Kept → DCR (base +0.27) | Live blocked |
|---|---|---|---|---|
| quality ≥ 5 | 8 · 50 % · −0.11 · 4 days · 8 pairs | 0.69 | 41 · 83 % · +0.26 → +0.30 | LTC, QNT (both losers) |
| pair RSI prev > 39.2 (fresh sharp drop) | 16 · 62 % · +0.01 · 13 days | 0.43 | 33 · 85 % · +0.29 → +0.27 | LTC, QNT |
| BTC 1h RSI ≤ 42.6 | 25 · 68 % · +0.07 | 0.28 | → +0.22 | LTC |
| global vol > 0.81 | 24 · 71 % · +0.14 | 0.13 | → +0.18 | QNT, ONE |

## 4. Verdict
- **No market-condition filter passes.** The best cut (quality ≥ 5) is 8 fills on 4 days, P 0.69. It fails N ≥ 15, ≥ 8 windows and 95 % confidence. Every market variable blocks positive-expectancy fills.
- Two of three losing is a 12 % event at a 78 % win rate. It was one morning, so it counts as one window.
- **Where the $ came from: size.** The live MS ticket is about $39–44k notional, which is 4.4× the account at 20×. That is 3.5× the master median notional of $12k. Each stop costs about 2.9 % of the account.
- **The recent era is break-even** (Sep–Nov 23 · 65 % · −0.00). Its 95 % WR floor (~48 %) is below the 59.5 % breakeven, so Kelly at the floor is ≤ 0. Under DL 547 (size by edge), this is the real lever. It is the operator's parked momentum shrink, not a filter.
- Recommended: register **MS_HIGH_QUALITY (entry_quality_score ≥ 5)** as an OBSERVE line with the cut frozen. It may only be armed once the blocked cohort reaches N ≥ 15 on ≥ 8 windows with avg < 0 at 95 %. A declared override (arm now, revert if the first 8 priced blocked signals average > 0) is possible, but the evidence is thinner than DL 555 / 559 had: 4 days and P 0.69.
