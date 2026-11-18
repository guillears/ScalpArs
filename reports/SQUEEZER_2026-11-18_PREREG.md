# SQUEEZER vs DUMPER after the second peak — pre-registration (frozen 2026-11-18 13:10 UTC)

Frozen before ANY separator, label-rate or tradable result of this study was computed. Research only; nothing in the bot changes.
Already seen before this freeze (by the coordinator, all periods pooled): the gross no-stop drift of the two entry cells (24 h / 72 h
median and mean, MAE quantiles, three stop variants ≈ 0). Nothing per variable has been seen.

## Question
After a Frenzy coin's second peak the median coin keeps dumping from the entry, but a minority squeezes hard first and erases the
profit. (1) Can the squeezers be told apart AT ENTRY? (2) If not, can sizing alone (tiny size, many trades, no stop) harvest the drift?

## Entries (frozen, no new selection)
- SP15 = cell `P2|P15|R0|RSI` (one row per entry; the `|EA` rows of reports/SECOND_PEAK_2026-11-18_trades.csv), SP8 = `P2|P8|R0|K`.
- EXH = EXHAUSTION_SHORT realisable entries (reports/EXH_EXIT_GRID_data cand_ref + its paths), same labels where the path allows.
- Entry price = open of the entry 5m bar. Path = 5m klines from the entry bar for 72 h (1m where the EXH grid already has them).

## Labels (path, from entry, 72 h window)
- **SQZ_X** (X ∈ {10, 20, 50}): the high reaches entry × (1 + X %) BEFORE the low reaches entry × 0.90 (a bar touching both = squeeze
  first). **Primary label = SQZ_20.** SQZ_10 = the complement of DUMPER on decided paths; SQZ_50 = the catastrophe tail.
- **DUMPER**: the low reaches entry × 0.90 before the high reaches entry × 1.10.
- Paths with neither within 72 h = UNDECIDED (kept in rates as non-squeezers for SQZ_20 / SQZ_50; excluded from the DUMPER race).
- Also reported: MAE before the dump, MFE, 24 / 72 h return.

## Variables (entry-time only, lag-safe)
From the trades file: vd, rsi15, atrp, h_since, volx, vol24, move24, vs_vwap, run_pct, off_peak, gain_pct, in_state, fund_last, ssc,
btc_r1h, btc_r4h, btc_r24h, btc_ema20_1h, btc_d50_200, pull, h2r, plus h2dist (H2 / entry − 1) and leg_h (hours from leg start).
New (built here):
- Open interest (data.binance.vision daily metrics, 5-min rows, create_time = period START → only rows with create_time ≤ entry − 5 min):
  oi_usd, oi_chg_1h, oi_chg_4h, oi_chg_24h (%), oi_vol = oi_usd / vol24.
- Long/short: top-trader account ratio (ls_top_acc), top-trader position ratio (ls_top_pos), global account ratio (ls_glob), and their
  4 h change; taker buy/sell ratio last 1 h mean (taker_1h) and 1 h minus prior 4 h (taker_chg).
- Funding: fund_last (exists), hourly-settlement flag at entry (fund_hourly: last settlement gap < 8 h), funding sum last 24 h
  (fund_24h), funding trend (last − mean of the 3 before).
- Market structure: spot market at entry (has_spot), coin age in days since futures onboard (age_d), Binance Alpha listed (alpha, today's
  list — not point-in-time), market cap at entry ≈ today's circulating supply × entry price (mcap — approximate: supply not point-in-time),
  mcap / vol24 turnover, oi_usd / mcap.
- Liquidation proxy: liq_short = oi_chg_1h while 1 h price change > 0 (falling OI into a rising price = shorts closing / liquidated);
  ret_1h before entry.
- Order-book depth: data.binance.vision bookDepth if it exists for the pair-days; if coverage < 50 % it is a listed blind spot.
Coverage table and blind-spot list are produced FIRST.

## Cuts / tests (DESIGN block only for the screen)
- 1D: sign (where the variable has a natural zero), median split, terciles, theory cuts: fund_last < 0, fund_last ≤ −0.1 %, fund_hourly,
  oi_chg_24h > 0, oi_chg_4h > 0, ls_glob < 1, ls_top_pos < 1, taker_1h > 1, has_spot = no, alpha = yes, age_d < 90, mcap < $100 M,
  liq_short < 0.
- 2D: every pair of variables, tercile × tercile (each of the 9 cells vs the rest) and median × median.
- Statistic: SQZ_20 rate of the bucket minus the rest, z (two-proportion); bucket N ≥ 30. Also the 24 h no-stop mean return of the bucket.
- Null: 500 shuffles of the label across DESIGN entries (also a day-block shuffle); family-wise p = share of shuffles whose max |z| over
  ALL tests ≥ the observed.
- Survivor = family-wise p < 0.05 AND same direction in every DESIGN half-year (24-H2, 25-H1, 25-H2, 26-H1) AND no single pair / window
  (day) > 50 % of the bucket's squeezers. Ranked by |z|; top 3 go to the tradable test.

## Tradable test ("short only the predicted DUMPERS")
Rule = skip the entries in the survivor's squeeze bucket. Exit grid: TP {5 %, 10 %, none} × stop {10 %, 15 %, H2 + 0.5 %, none with a
50 % catastrophe cap} × hold {24, 48, 72 h} = 36 exits. Costs: taker 0.05 % + slip 0.10 % per taker leg (stress 0.20), maker TP 0.02 %
filling only when the low trades 1 tick THROUGH; stops gap-aware (max(stop, bar open)); a bar touching both = stop; real funding
(8 h and hourly coins, missing months −0.05 %/8 h). One position per pair per exit cell. The exit is picked on DESIGN mean only.
Reported: N · WR · mean · median · best-5 %-removed · per half-year · 0.20 slip · Kelly at the one-sided 95 % Wilson floor (n = distinct
trade days) · DCR at one size · max drawdown · the same exit on ALL entries (the filter must add value in every half-year).

## Sizing route (C)
All SP15 / SP8 entries, no stop, hold 24 h and 72 h, net of costs and funding. Size f ∈ {0.01, 0.02, 0.05, 0.1} × equity notional per
trade, overlapping trades allowed. Daily P&L = Σ f × ret by close day; 7-day block bootstrap (2,000 paths × 365 days):
P(ruin = equity ≤ 50 % of start at any point), median and 5 % quantile of the end equity, and the full return distribution.

## Blocks
DESIGN 2024-07-01 → 2026-07-01 (screen + exit pick). HOLDOUT-A 2023-07-01 → 2024-07-01 (seen by earlier studies in aggregate — not
virgin). HOLDOUT-B 2026-07-01 → data end (seen in the coordinator's pooled drift numbers; never read per variable — the separator
holdout). 2023-01 → 06: NOT buildable here (5m cache starts 2023-05, episodes need ≥ 2 months of warm-up) — no clean pre-2023-07 holdout.

## Pass bar (all must hold, for BUILD)
1. every half-year mean net > 0 · 2. best-5 %-removed > 0 · 3. > 0 at 0.20 % slip · 4. family-wise p < 0.05 · 5. Kelly at the 95 %
Wilson floor > 0 · 6. both holdouts > 0 (HOLDOUT-B is the decisive one) · 7. the filter beats the same exit on all entries in every
half-year. OBSERVE if 1–5 hold on DESIGN and one holdout fails. Otherwise NO. Cuts are never refit after a holdout is read.

## ADDENDUM 1 — EXHAUSTION stop clusters (operator, frozen 2026-11-18 13:23 UTC, before reading the trade frame)
Entries/exit: EXHAUSTION_SHORT, exit `B|tp10|S2|24` (TP 10 % maker, SL min(1.5×ATR, 6 %), 24 h), trade frame
`exh_L2_trades_ctx.pkl` (3,998 trades, as handed over; net per trade as given there). Already ruled out by the coordinator (not
repeated): same-pair re-entry, lag-safe signal / stop / open-position counts, end-of-day trade counts.
Question: which MARKET condition at entry, known in real time, explains the stop-cluster days?
Variables (all lag-safe, last CLOSED bar before entry): BTC return 1 h / 4 h / 24 h; BTC close vs EMA20 and EMA50 on 1 h, vs EMA20
on 4 h, daily EMA50 vs EMA200 sign; alt breadth = share of USDT perps (5m cache, ≥ 100 pairs live) whose close is above their 1 h EMA20,
above their 4 h EMA20, and up over 24 h; Frenzy episodes market-wide that started in the prior 24 h (SECOND_PEAK eps list); total alt
OI change 24 h (sum of OI $ over pairs in the metrics cache with both points); BTC dominance proxy = BTC 24 h return − median alt 24 h
return; market-wide funding = median last funding rate over the pairs with funding data.
Unit = the DAY (window). Day label = cluster day (≥ 5 stops that day) vs not; trade-level read on the same variables second.
Tests: sign, median, terciles per variable (switch = sleeve OFF when the condition holds at entry). Null: 1,000 shuffles of the
day labels, family-wise max |z| over all tests. Per half-year direction. A switch passes only on the expectancy bar in WINDOW units:
blocked days' net mean < 0 with 95 % day-bootstrap confidence, ≥ 8 distinct days, no single day ≥ 50 % of the blocked loss, N ≥ 15,
WR below breakeven, same sign in every DESIGN half-year, and both holdouts (2023-07 → 2024-06; 2026-07 → now) not reversed.
DESIGN = 2024-07 → 2026-06.
