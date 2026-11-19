# ROUTE A — tick data inside the early Frenzy candles — PRE-REGISTRATION

**Frozen 2026-11-19 ≈ 15:01 UTC as written; the file's actual creation / last-edit time is 15:05:53 UTC (review note), about 4 minutes after the first aggTrades fetch (15:02 UTC) and about an hour before the first result csv (16:10 UTC). No tick feature or outcome had been computed when it was written.** Research only.
Nothing below may be changed after the first result is read; any later addition is labelled EXPLORATORY in the report.

## Question
Frenzy pumps build over several green 5m candles before the bot's flag. On candles, the early green candles of the ~13 % of
coins that go on to a Frenzy look the same as the ~87 % that fizzle (FRENZY_LONG_SEARCH 11-19 §5). Does trade-level data
(Binance USDT-M aggTrades) INSIDE those early candles separate runners from fizzles in real time, does it ADD to the candle
features, and is a long on it profitable after real costs at many trades a day?

## Theory (written first)
A real pump start is an information / crowd-arrival event: a growing number of participants arriving (trade-rate accelerating),
buyers lifting the offer (taker-buy share high, by $ more than by count when informed size leads), large prints that lift price
and are not given back (price holds between bursts), and sweeping algos (long taker-buy runs). A fizzle is one actor or a thin
book: few large prints moving a thin book (high spread proxy), a burst that already peaked (volume concentrated earlier, last
minute fading), retracement between bursts, taker-buy share falling at the decision time.
Counter-theory (written first, the null we expect from 11-17 / 11-19): the first leg is noise-driven; tick features at the
trigger mostly re-measure the same volume / return the candles already see, so tick data adds ≤ 0.02 AUC over candles.

## Sample (frozen)
- Universe and trigger: the earlier-detector rows `reports/FRENZY_LONG_SEARCH_data/early_rows.pkl`, cell **W15_R3_V10**
  (15-min return ≥ +3 % and 15-min quote volume ≥ 10× normal, 24 h quote volume ≥ $20 M, 6 h debounce per pair; local 5m caches,
  delisted coins included). Only one trigger (one cell) is studied.
- Months 2024-01 → 2026-11-17. **DESIGN = 2024-01 → 2026-06** (30 months), **HOLDOUT = 2026-07 → 2026-11** (read once).
- Classes: RUNNER = a Frenzy spike flag (`spike_after`) closes within 2 h after the trigger close; FIZZLE = none.
- Stratified random sample per month (numpy seed 20261119, shuffled order frozen): DESIGN 20 runners + 20 fizzles per month;
  HOLDOUT 35 + 35 per month (all if fewer exist). Events whose aggTrades file is missing (404) are replaced by the next in the
  frozen order. Target ≈ 600 + 600 DESIGN, ≈ 175 + 175 HOLDOUT.
- BASELINE (descriptive only): for each sampled event one random moment on the same pair-day ≥ 2 h from any trigger of any cell
  (re-uses the same file).
- Weighting: every priced / rate number is re-weighted to the trigger population (weight = month class count ÷ sampled
  class count), so the 50/50 sample does not inflate the runner share.

## Decision times (checkpoints)
C0 = trigger close (`sig_ms`); C1 = +60 s; C2 = +120 s; C5 = +300 s. Features use ONLY trades with transact_time < decision.
At checkpoint c an event whose flag already closed at or before the decision is EXCLUDED from that checkpoint's classification
(it is the known flag entry). Entry = first aggTrade at or after decision + 8 s, × (1 + slip).

## Tick features (pre-registered; expected sign for RUNNER)
Windows: BASE = [sig − 75 min, sig − 15 min) · BURST = [sig − 15 min, decision) · LAST = [decision − 60 s, decision).
| # | feature | definition | sign |
|---|---|---|---|
| t1 | tps_accel | trades/s LAST ÷ trades/s BASE (log) | + |
| t2 | tps_burst | trades/s BURST ÷ trades/s BASE (log) | + |
| t3 | buy_cnt | taker-buy share by count, BURST | + |
| t4 | buy_qv | taker-buy share by quote $, BURST | + |
| t5 | buy_qv_last | taker-buy share by quote $, LAST | + |
| t6 | big_lift | # taker-buy prints in BURST with $ ≥ BASE p99 trade $ that print above the previous trade price, log1p | + |
| t7 | med_trade | median trade $ BURST ÷ median trade $ BASE (log) | + |
| t8 | top10_share | share of BURST $ from its top-decile trades | + |
| t9 | retrace | (BURST high − last price) ÷ (BURST high − BURST low) | − |
| t10 | run_ratio | mean taker-buy run length ÷ mean taker-sell run length, BURST (log) | + |
| t11 | last_conc | $ in LAST ÷ (BURST $ per minute) (log) | + |
| t12 | spread_bps | median |Δ log price| between consecutive opposite-side trades, BURST, bps | − |
| t13 | uptick_share | upticks ÷ price-changing trades, BURST | + |
| t14 | vwap_gap | last price ÷ BURST VWAP − 1, % | + |
| t15 | since_burst_h | hours since the previous W15_R3_V10-like window on the coin (5m), log, capped 30 d | + |

## Candle features (the comparison set, all at the decision time from local 5m bars, lag-safe)
c1 ret15 (trigger 15 m return) · c2 volx15 (15 m vol multiple) · c3 ret6h_base (close ÷ 6 h low − 1) · c4 ret1h_prior (return of
the hour before the trigger window) · c5 close_pos (close in the trigger bar range) · c6 log q24 · c7 green6 (green bars of last 6)
· c8 btc_1h (BTC 1 h return) · t15 belongs to the candle side as well (it is 5m-derived) and is reported there.

## Tests (frozen)
1. **Per feature AUC** (runner vs fizzle) on DESIGN at each checkpoint, with the family-wise null: labels shuffled within month,
   1,000 draws, statistic = max |AUC − 0.5| over the 14 tick features × 4 checkpoints.
2. **Models** (logistic, standardised, L2 C = 1): TICK = {t1, t4, t5, t9, t11, t12}; CANDLE = {c1, c2, c3, c5, c6, t15};
   BOTH = union. Time-split CV on DESIGN by half-year (train on all earlier halves, test on the next; first test half 2024-H2).
   Statistic: out-of-fold AUC; ΔAUC = BOTH − CANDLE. Nulls: (a) labels shuffled within month (200 draws) for TICK AUC;
   (b) tick features permuted across events within month (200 draws) for ΔAUC.
3. Secondary target PATH = WORKED: price reaches +5 % before −5 % after the entry (from ticks / 1m, then 5m), and REACH (max
   favourable % before −5 %). Same AUC read for the PATH label.
4. **Separation PASS** (all): TICK OOF AUC ≥ 0.60 and null p < 0.05 · ΔAUC > 0 with null p < 0.05 · HOLDOUT (model refit on all
   DESIGN, scored once) TICK AUC ≥ 0.58 and BOTH ≥ CANDLE.
5. **Priced read** (done for every checkpoint regardless, but a ship verdict only if 4 passes): long the events whose BOTH score is
   in the top 20 % (threshold frozen from DESIGN OOF scores), and all triggers as reference. Exits Q3 (TP +3 % maker / SL −4 % /
   4 h) · TR (arm +3 %, trail 3 % off the high, SL −5 %, 24 h) · T1 (1 h time). Entry slip 0.10 % (stress 0.30 / 0.50), taker
   0.05 % in and out, exit slip on non-TP exits, maker 0.02 % on TP, real funding. Report weighted avg net %, WR, day-clustered CI,
   best-5 %-out, LOO coin / day, trades/day (= selected share × trigger rate), regime (BTC daily sign R1, breadth R5) in WINDOW
   units, one-size DCR estimate.
6. **Trading PASS** (all): DESIGN-OOF top-20 % avg net > 0 at 0.10 % with day-CI low > 0 · HOLDOUT avg > 0 · avg > 0 at 0.30 % ·
   best-5 %-out > 0 · ≥ 3 trades/day. Otherwise NO BUILD.
7. Capacity caveat: aggTrades carries no book; any fill at a pump start is unverified → ROUTE B (live websocket with depth)
   before anything trades.
