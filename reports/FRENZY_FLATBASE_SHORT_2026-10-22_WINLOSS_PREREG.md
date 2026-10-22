# FRENZY flat-base short: winners vs losers (PRE-REGISTRATION, written before any result)

Date 2026-10-22. Research only. Script: `scripts/study_flatbase_short_winloss.py`.

## Cohort
- Primary cell: **S4 · TP −8 % · max hold 72 h · stop +15 %**, FLAT14 ≤ 44.39 % (frozen), 0.20 % costs + real funding, gap-aware stops, one short per pair (as `study_flatbase_short_exits.py`; reproduction must match `_EXITS_trades.csv` exactly).
- Robustness cohort (3 nearest neighbours): S4 · TP −8 · 72 h · stop 10 · S4 · TP −8 · 72 h · stop 25 · S4 · TP −10 · 72 h · stop 15. A separator must point the same way in all four cells (discovery set).
- Label: W = net % > 0, L otherwise; continuous net % kept.
- Blocks: ③ 2025 (flagged) / 2026-H1 / 2026-H2 by entry time; ② master 10-02→10-13 and ① live 10-14→ on REAL engine episodes only.

## Features (all at the S4 signal bar close = entry time; bars ≤ entry only)
Coin (trade units): pump size (peak/base), hours since peak, hours since episode start, % below anchored VWAP, retrace fraction, % above pre-pump base, 4 h volume ÷ episode max 4 h volume (decay), 24 h quote volume, 24 h volume ÷ pre-pump 24 h volume, coin 1 h / 4 h / 24 h return, coin 1 h RSI14, funding (last settled, 8 h-normalised), funding summed over 24 h, funding trend (last − mean of 24–72 h before), listing age (capped by data start 2024-06), market cap (Info-panel circulating supply today × entry price — constant-supply proxy), TURNOVER = 24 h volume ÷ mcap at entry, peak turnover during the pump, pre-pump turnover, entry hour (4 h block), FLAT14, FLAT30, RV14, DEADVOL, DIST90.
Market-wide (DAY units): BTC 1 h / 4 h / 24 h / 7 d return, BTC 1 h EMA20−EMA50 gap %, BTC 1 h ADX14, BTC 1 h RSI14, BTC distance from 24 h high / low, BTC 24 h realised vol, breadth (share of USDT perps up over 24 h and 7 d), median alt 24 h return − BTC 24 h return, number of FRENZY spikes (all / flat) in the prior 24 h, weekday.
Descriptive only (look-ahead, NOT filters): BTC return over the hold, funding paid over the hold.
OI: Binance openInterestHist covers ~30 days → would score only ①② (≈ 50 trades); not tested (blind spot).

## Granularities and tests
- 1D: sign (signed variables; turnover at ratio − 1; unsigned → median split), terciles, quintiles. Edges frozen on the discovery set. Hour = 6 four-hour blocks; weekday = 7 days.
- 2D: exhaustive tercile × tercile over every feature pair (hour → 3 blocks, weekday → Mon–Wed / Thu–Fri / Sat–Sun).
- Test = "block this bucket": one-sided z of WR(rest) − WR(bucket); bucket and rest each ≥ 15 trades and bucket ≥ 8 distinct days in discovery; unscored trades excluded from BOTH sides.
- Nulls (1,000 shuffles each, family-wise max-z): (a) PRIMARY day-preserving null — outcome vectors of whole days permuted among days with the same trade count; (b) plain trade-level label shuffle. Families: 1D and 2D separately.
- Market-wide variables additionally screened in DAY units (one row per day: day-mean feature, day WR; day-label shuffle null).
- Survivor = family-wise p < 0.05 under null (a).

## OOS
- Variant A: discover on 2025 + 2026-H1; confirm on 2026-H2 + ② + ①.
- Variant B: discover on 2026-H1 only; same confirmation blocks.
- Confirmed = in the pooled confirmation set the blocked bucket's WR is lower with one-sided day-bootstrap p < 0.05, its avg % is lower, the direction holds in 2026-H2 alone, and the 4 neighbour cells agree in discovery.
- Then the expectancy bar (blocked cohort at 1×: WR < sleeve breakeven WR = |avg loss| / (avg win + |avg loss|), avg < 0 with 95 % window-clustered bootstrap, ≥ 8 windows, no window/pair ≥ 50 % of the loss, N ≥ 15), a 40 % haircut on Δ, per-pair and per-day concentration.
- Top-10 by raw z per family are reported with their OOS numbers regardless (transparency, not evidence).
- Conditions that only ADD trades (looser flatness 44–55 / 55–70) reported separately, never recommended here.

## Addition (operator, before results): turnover
Turnover = 24 h quote volume ÷ market cap. Theory: > 1 = speculative churn / pump-and-dump, possibly MORE likely to collapse (good for the short); it is the ratio WILLY already blocks on. Granularities: sign of (ratio − 1), terciles, quintiles; peak turnover during the pump and pre-pump turnover as separate features. Coverage reported per block; unscored ≠ rest. Proxy calibrated against the bot's real `entry_mcap_usd` stamps where the same pair has a stamped fill within ±3 days.

## Addition (operator, before results): tighter stops
Same S4 · TP −8 · 72 h, stops **5 %, 8 %** (and 10 / 15 / 25 for reference), same gap-aware / stop-first / funding / cost model. Per stop and block: N, WR, avg % [day-clustered 95 % CI], share of stop-outs that later touched the −8 % TP within the 72 h window ("winners stopped early"), worst trade, max drawdown at $190/short (realised equity by exit time), and the fair non-anticipating random-entry null (same exit, costs, funding, one short per pair; 200 draws). Verdict: does the tighter stop cost more winners than it saves in losses. Any surviving filter is shown under each stop.
