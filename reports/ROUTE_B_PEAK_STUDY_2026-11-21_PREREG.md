# ROUTE_B_PEAK_STUDY 2026-11-21 — pre-registration

**Written 2026-11-22 01:50 UTC, BEFORE any order-flow variable, label or outcome of this study was computed.** Known at writing time
(coverage only): the orders exports since the recorder start (2026-11-19 23:23:30 UTC) hold 16 CLOSED full-size fills (DIST 3 · EXH 4 ·
FB 1 · WILLY_HT 8 · momentum 0 · flips 0 · plain WILLY 0); the DL 612 loader holds ~60 + priced blocked / refused signals of the same
sleeves; the S3 per-minute summaries (`features/<date>/<HH>.csv.gz`) cover 2026-11-19 23:00 → 2026-11-22 00:00; the FULL book (`fb_*`),
open interest (`oi*`) and positioning ratios exist only from 2026-11-21 22:00 UTC (DL 611 deploy) — after the last closed fill.
Research only: no engine / config / git change.

**Question (operator):** "I still find it strange there was no finding in any sleeve of a variable that changed at the peak." STATE_EXIT
2026-11-21 tested candle / 5m-metric variables; this study tests the order book and the trade tape (Route B), per sleeve.

## Cohorts (per sleeve — each its own section; never pooled across Frenzy sleeves)
- **REAL** — taken fills from every orders export (newest export wins), dedup (opened_at[:19], pair, direction), CLOSED, no probes,
  opened ≥ 2026-11-19 23:23:30 UTC, coin with Route B top-20 book rows over the trade's life. Life = real fill → booked close; entry = the
  real fill price. These fills ran today's exits except where a level changed during the batch (HT TP 5 → 6 net, EXH TP cap) — the booked
  path is used as the trade's life (the peak question is about the path, not the level); the gave-back split uses today's TP.
- **SHADOW** (secondary, labelled, never mixed into REAL rows) — the same sleeve's blocked / refused signals priced by the DL 612 loader
  under TODAY's exit (`out_priced` ∧ `out_final`, one per possibility: `poss_dup` False), life = `out_entry_ms` → `out_exit_ms`, entry =
  `out_entry_px`. Rationale: the peak question is a property of the coin's path after a sleeve-signal entry; a blocked signal differs only
  in one entry filter. A SHADOW pass can only open an observe line, never a ship.
- Sleeves with 0 covered fills (momentum long / short, flips, plain FRENZY_WILLY — whose V2b exit the loader cannot price) are reported as
  UNTESTABLE with the N needed.

## Labels (each trade's own 1m path from its entry price; 1m klines, data.binance.vision / fapi)
- In-trade minutes k = the full minutes after the fill minute up to the exit minute. fav(k) = side × (favourable extreme of minute k ÷
  entry − 1) × 100; close P&L c(k) = side × (close_k ÷ entry − 1) × 100. MFE = max fav; **PEAK minute p** = first argmax.
- Decision points = closes of in-trade minutes that END before the exit (the exit minute itself is not a decision point).
- **Eligible trade:** MFE ≥ +1.0 %.
- **PEAK zone:** decision minutes k ∈ {p, p+1, p+2} (the close of the top minute and the next two — a market exit there still captures the
  top area).
- **CONTINUATION zone:** decision minutes k ≤ p − 3 with c(k) ≥ max(1.0, 0.5 × MFE) and MFE − c(k) ≥ 0.5 (up a similar amount, but at
  least 0.5 % more was still to come).
- **RANDOM:** every in-trade decision minute (the variable's base firing rate).
- **Firing episodes:** consecutive firing minutes merge into one episode. A trade scores one PEAK HIT if any firing minute is in its
  PEAK zone; every episode with ≥ 1 minute in the CONTINUATION zone is one FALSE ALARM.
  **Precision = peak hits ÷ (peak hits + false alarms). Recall = peak hits ÷ eligible trades with a PEAK-zone decision minute.**
- **Reached-X split:** X = 50 % of the trade's TODAY TP distance (FB 2.5 % · HT (6 net + fees) ÷ 2 ≈ 3.03 % · EXH / DIST the row's TP ÷ 2).
  Among trades with MFE ≥ X: WENT_TP (booked / walked exit = TP) vs GAVE_BACK (any other exit). Statistic per group: share of trades where
  the variable fired between the first minute with c ≥ X and the peak zone end.
- **Windows:** a firing's market moment = its UTC 30-min bucket; same-bucket firings across trades = ONE window. Per-pair concentration =
  the largest single pair's share of peak hits.

## Layer 1 — pre-registered order-flow variables (frozen thresholds; LONG wording, SHORT = mirrored; "fav" = our side)
All computed at the close of decision minute k from Route B rows of minutes ≤ k (a minute's row is complete at its close — lag-safe; a book
row older than 2 minutes is not used). In-trade baselines use minutes from the first in-trade minute to k only.
1. **TAKER_FADE** — taker share on OUR side (buy share for a long, sell share for a short) over minutes k−2…k < 0.45 AND ≥ 0.10 below the
   same share over minutes k−12…k−3. *Theory:* the move is carried by aggressive takers on our side; when they stop and the opposite side
   lifts / hits, the impulse is over.
2. **AGAINST_BURST** — opposite-side taker $ in minute k ≥ 3 × its in-trade per-minute median (≥ 5 minutes of baseline) AND ≥ 60 % of
   the minute's taker $. *Theory:* an aggressive opposite burst (distribution into strength / short covering ends) marks the turn.
3. **BOOK_FLIP** — book imbalance within ±0.5 % (imb_05 = (bid − ask) ÷ (bid + ask)), oriented to our side, mean over k−2…k ≤ −0.20.
   *Theory:* resting liquidity piles up against us (a wall of asks above a long / bids under a short) at a top.
4. **SPREAD_WIDE** — spread_bps at k ≥ 2 × its in-trade median so far (≥ 5 minutes of baseline). *Theory:* makers pull quotes when
   one-sided flow exhausts and volatility peaks.
5. **CAPITULATION_LIQ** — liquidations of the OTHER side (shorts force-bought for a long; longs force-sold for a short) ≥ $5,000 summed over
   k−2…k on the coin. *Theory:* the last leg of a move is a squeeze / cascade of the losing side; when it prints, the fuel is spent.
   (The stream is a ≤ 1 / s / coin sample — a lower bound.)
6. **OI_NO_NEW_MONEY** — price made a new in-trade favourable extreme within the last 15 min AND open interest's change over the last
   15 min ≤ 0. OI source: Binance REST openInterestHist 5m; a row labelled T is treated as available only at T + 10 min (DL 612's
   5-min mislabel + margin). *Theory:* a move without new positions is closing flow, which ends.
7. **BIG_AGAINST** — the minute's largest trade ≥ 5 × its in-trade median AND the opposite side ≥ 60 % of that minute's taker $ (the
   summary has no side for the largest trade — this is the proxy). *Theory:* large opposite participants step in at a top.
8. **PREMIUM_FADE** — perp premium (mark ÷ index − 1, %) oriented to our side (long: premium; short: −premium) was ≥ +0.05 at some in-trade
   minute and is now ≥ 0.10 pp below its in-trade max. *Theory:* perp-led buying (premium) collapses when the impulse ends.
**Candle comparators (re-checked on the same trades, STATE_EXIT definitions):** RSI_ROLL (5m RSI14 in-trade max ≥ 75, now ≥ 10 below; short
mirror ≤ 25 / +10) · LH2 (≥ 2 consecutive lower 5m highs; short: higher lows) · CLIMAX (against-us 5m candle on ≥ 2 × the prior-20 mean
quote volume, its extreme within 1 % of the in-trade extreme). 5m bars are complete clock bars known at their last minute's close.

## Layer 2 — exhaustive screen (per sleeve, REAL and SHADOW separately)
Every per-minute Route B column (spread, depth ±0.2 / 0.5 / 1 % each side and bid ÷ ask, imb_05, cost of $2.5k / $10k market orders each
side, trades count, taker buy / sell $, taker share over 1 / 3 / 5 / 10 min, traded $ vs in-trade median, largest trade vs median,
liquidations each side over 3 / 10 min, premium, predicted funding, mid range) × {level, change over 3 min, change over 10 min, ratio to
in-trade median}, oriented to the trade side → thresholds at the 20 / 33 / 67 / 80 % quantiles of the sleeve's in-trade decision minutes
(label-blind), both tails. Plus every 2D median-sign quadrant of the 1D features. Statistic = precision (as above). Tests with < 5 episodes
are not ranked. **Null:** circular shift of each trade's firing sequence within its own decision minutes (preserves runs), 1,000 shuffles;
family-wise max precision (over tests with ≥ 5 episodes) → q95. Survivor = observed precision > family-wise q95.

## Gates (frozen)
- **Observe-line candidate** (the operator's freeze-week bar): precision ≥ 60 % on ≥ 15 firing episodes (hits + false alarms) over ≥ 8
  windows, no single pair ≥ 50 % of hits, AND precision > its own shuffled-null q95 (layer 1) / family-wise q95 (layer 2). Found on SHADOW
  only → "SHADOW lead", needs REAL confirmation.
- **Exit counterfactual** run for any variable with precision ≥ 60 % on ≥ 5 episodes (REAL or SHADOW), labelled UNVERIFIED / illustrative
  below 15: exit at the close of the first decision minute with c(k) ≥ X (half the TP) AND the variable firing, market order: taker 0.045 %
  + slippage = the recorded cost of a $2.5k market order at that minute (bps ÷ 100, floor 0.02 %), else 0.10 %; vs the booked (REAL) /
  walked (SHADOW) result. Split SAVED (today's result worse) vs CHOPPED (today's better), by today's exit kind (TP / STOP / TIME), leave-one-
  out, DCR at ONE size (0.61 × equity notional per fill, the STATE_EXIT ruler) over the cohort's calendar days.
- **Ship:** impossible from this study (N). The verdict per sleeve is CANDIDATE (observe line) / REFUTED-at-this-N (precision < 50 % with
  ≥ 10 episodes) / UNTESTABLE-YET (fewer than 10 episodes).
- **Power statement:** for each variable, the N of eligible trades needed for 15 episodes at its observed firing rate.
