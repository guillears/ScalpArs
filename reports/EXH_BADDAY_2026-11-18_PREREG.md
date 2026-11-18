# EXHAUSTION bad-day indicators — pre-registration (frozen 2026-11-18 13:45 UTC, before any outcome is computed)

Question (operator): what is the theory behind EXHAUSTION_SHORT, why do stops cluster on some days, and which indicator says
"shorting exhaustion is bad today"?

## Theory (written first)
- The short pays when the pump's buyers are spent. RSI rolling over while volume is still high means supply is absorbing the last
  buyers. A falling spot share means the move is carried by leveraged perp longs, which unwind easily.
- It fails when NEW buying keeps arriving:
  (a) market-wide inflow / rotation (alt season) refuels every leg;
  (b) shorts are crowded on the coin, so a bounce turns into a short squeeze;
  (c) the exhaustion pattern itself is failing market-wide that day: other coins' RSI roll-overs go on to make new highs.

## Trades
`exh_L2_trades_ctx.pkl` (3,998 simulated trades, cell `B|tp10|S2|24`: TP 10 %, SL min(1.5×ATR15, 6 %), 24 h), net % `n0`.
DESIGN = entries 2023-07-01 → 2026-06-30. HOLDOUT = 2026-07-01 → end, read last, once.

## Indicators (all lag-safe: every input closes ≤ entry; metrics rows create_time ≤ entry − 5 min)
1. **SIGFAIL_6H / SIGFAIL_24H — market-wide exhaustion failure rate.** Universe = every 15m RSI14 70 → ≤ 70 cross inside a Frenzy
   window in `reports/EXHAUSTION_EXPAND_data/all.pkl` (`fr` & `RSI15`; vd ≥ 0.40, vol24 ≥ $20 M; NO vd ≥ 0.6366 / spot filter),
   de-duplicated on (pair, entry_ms). Reference = open of the 5m bar after the cross. FAIL = the max 5m high over the next 2 h
   (24 bars from the reference bar) ≥ reference × 1.03. A cross is known only once resolved (entry_ms + 2 h ≤ T).
   At trade entry T: crosses on OTHER pairs resolved in (T − W, T], W = 6 h / 24 h. Rate = fails ÷ count.
   Thin (count < 3 for 6 h, < 5 for 24 h) → not in the bad bucket (reported as its own row).
2. **ALTVOL_EXP — market volume expansion.** Hourly Σ quote volume over every USDT perp in the 5m caches except BTC/ETH.
   Ratio = Σ of the last 4 closed hours ÷ mean of the same 4 clock hours on each of the 7 prior days.
3. **SPOT_LED — market spot share.** Fixed basket of 22 alts with spot + perp throughout (SOL XRP DOGE ADA AVAX LINK DOT LTC BCH TRX
   ATOM NEAR FIL ETC APT ARB OP INJ UNI AAVE XLM HBAR). Share = Σ spot qvol ÷ (Σ spot + Σ perp qvol). Δ = share over the last 4 closed
   hours − share over the 168 h before those 4 h. Spot 1h klines from data.binance.vision.
4. **COIN_CROWD — short crowding on the coin.** CROWD2 = last settled funding ≤ 0 AND OI 1 h change > 0.
   CROWD3 = CROWD2 AND entry price within X % of the coin's 24 h high (X = DESIGN median of that distance, indicator values only).
   (Funding and OI were tested ALONE in SQUEEZER; only the conjunctions are new.)
5. Dropped: short-liquidation proxy (`liq_short` = OI 1 h change while price rose) — already tested in SQUEEZER.

## Bad bucket (theory direction) and cuts — 14 tests
- SIGFAIL_6H, SIGFAIL_24H: rate ≥ 0.50 · > DESIGN median · top DESIGN tercile.
- ALTVOL_EXP: ratio > 1 · > median · top tercile.
- SPOT_LED: Δ > 0 · > median · top tercile.
- COIN_CROWD: CROWD2 · CROWD3.
Medians / terciles from DESIGN trade-entry indicator values only. Opposite buckets are shown for information, never tested.

## Tests
- Per test, DESIGN: blocked (bad bucket) vs kept — N · WR · avg net % · z = (avg blocked − avg kept) / SE.
- Family nulls (2,000 draws each, all 14 tests jointly): ① shuffle — permute the trades' indicator rows; ② time-shift — roll the
  indicator rows circularly by ≥ 7 days of trades (keeps day clustering). Family-wise p = share of draws whose most-negative z ≤ observed z.
- Window units: days with a majority of entries in the bad bucket vs other days: day stop share, day net Σ %, number of days.
- Per half-year: blocked avg net % (direction consistency: blocked < 0 in ≥ 4 of 6 DESIGN halves).
- Expectancy bar on the blocked cohort (DESIGN): WR < breakeven WR (|avg loss| ÷ (avg win + |avg loss|), computed on DESIGN) ·
  P(avg < 0) ≥ 95 % by day-cluster bootstrap · ≥ 8 distinct days · N ≥ 15 · no single day ≥ 50 % of the blocked loss.
- Survivor = family-wise p < 0.05 under BOTH nulls AND the expectancy bar. Then switch "no entries while in bad bucket":
  before → after N · WR · avg · DCR at one size (0.183 × equity per trade, P&L booked on the exit day, every calendar day) ·
  max drawdown · worst day, per half-year; then HOLDOUT read once.
- Verdict: ARM SWITCH = survivor + holdout blocked avg < 0. OBSERVE = survivor, holdout too thin (< 8 days) or flat.
  NO = no survivor. Not re-fit after seeing results.

## Also shown (descriptive)
Top-15 stop-cluster days (most stops) with each indicator's day mean. Replay of 11-16 → 11-18 (HUMA, BLESS, ZKC, ZKP, PIPPIN, BREV)
where data exists (5m caches end 2026-11-16 16:00 UTC; later hours fetched from data.binance.vision daily files if published).

## ADDENDUM 1 — exhaustive 2D screen (operator: "there must be a 2D variable we are not considering to avoid the terrible days")
Frozen 2026-11-18 13:58 UTC, before any 1D or 2D outcome was computed (only coverage counts had been looked at).
- **Variables (every one, no shortlist):** (a) the theory indicators SIGFAIL_6H, SIGFAIL_24H, ALTVOL_EXP, SPOT_LED_Δ, dist_hi24;
  (b) the 16 SQUEEZER market variables (btc_r1h, btc_r4h, btc_r24h, btc_ema20_1h, btc_ema50_1h, btc_ema20_4h, btc_d50_200, br_1h, br_4h,
  br_up24, alt_med24, dom24, eps24, alt_oi_chg24, fund_med, fund_neg_share — `scratchpad/sq/mkt_hourly.pkl`, last closed hour);
  (c) every entry-time column of the EXH rows in `scratchpad/sq/D.pkl` (set = EXH): vd, rsi15, atrp, h_since, volx, vol24, move24,
  vs_vwap, run_pct, off_peak, gain_pct, in_state, ssc, f_last, fund_hourly, fund_24h, fund_trend, oi_usd, oi_chg_1h, oi_chg_4h,
  oi_chg_24h, oi_vol, ls_top_acc, ls_top_pos, ls_glob, ls_glob_chg4h, ls_top_pos_chg4h, taker_1h, taker_chg, ret_1h, ret_4h, ret_24h,
  age_d, mcap, liq_short, has_spot. Post-entry columns (sqz*, ret24/72, mae/mfe, dumper) are excluded.
- **Cuts (natural only):** sign (variables with a natural zero: returns, changes, gaps, Δ, funding) · DESIGN median (≤ / >) ·
  DESIGN terciles (T1 / T2 / T3); binary variables → their two values. A 2D cell = (bucket of var A) ∧ (bucket of var B), A ≠ B;
  cell = "block EXHAUSTION entries in that quadrant". Cells with DESIGN N < 30 or < 8 distinct days are not scored. A missing value
  is never in a bucket (the trade is kept).
- **Two scores per cell (DESIGN):** ① trade-level Δ = avg net % of kept − avg net % of all, and z = (avg blocked − avg kept) / SE;
  ② window-level terrible-day capture TDC = (share of the stops on TERRIBLE days that the cell removes) − (share of all winners it
  removes); TERRIBLE day = a DESIGN day with ≥ 5 trades whose day net Σ % is in the bottom decile of such days.
- **Null:** 300 day-block shuffles (days reordered at random, each day's outcome block kept intact) of the outcome vector against the
  fixed feature matrix; family-wise p for each score = share of shuffles whose family max (max Δ, max TDC) ≥ the observed value.
- **Pass bar (same as 1D):** family-wise p < 0.05 · blocked cohort on the expectancy bar in window units (≥ 8 days, day-bootstrap
  P(avg < 0) ≥ 95 %, WR < DESIGN breakeven, N ≥ 15, no day ≥ 50 % of the blocked loss) · blocked avg < 0 in ≥ 4 of 6 DESIGN halves.
  HOLDOUT (2026-07 → now) read ONCE for the top 10 cells by Δ (and top 10 by TDC). Best cell: before → after per half-year
  (N · WR · avg · DCR at 0.183 × equity · max drawdown · worst day) and its effect on the top-15 cluster days (stops removed vs winners blocked).
- Day-level P&L / stop share by bucket shown for the top cells. Nothing re-fit after reading.

_Timestamp note (appended 2026-11-18 13:45 UTC): the system clock read 13:45 UTC when ADDENDUM 1 was appended; its header says 13:58 by a typing error. Both the main prereg and ADDENDUM 1 were written before any outcome was computed. The two times above are left as written._

## ADDENDUM 2 — market regime axis (operator, frozen 2026-11-18 13:46 UTC, before any outcome was computed)
- Regimes (last CLOSED daily bar before entry): R1 = BTC 1d EMA50 > EMA200 (bull) vs ≤ (bear) · R2 = BTC 1d close > EMA200 vs ≤ ·
  R3 = alt regime: equal-weight alt index (median of alt 1d log returns over every USDT perp in the 5m caches except BTC/ETH,
  cumulated) vs its 1d EMA50 / EMA200 (EMA50 > EMA200 = alt bull).
- (1) EXHAUSTION itself per regime: N · WR · avg · DCR at 0.183 × equity (regime days only) · max DD · worst day, per half-year and
  per regime, in day units. Descriptive; any regime switch is judged on the same expectancy bar (window units, ≥ 8 days, etc.).
- (2) Every 1D theory test and the top 2D cells re-read WITHIN each regime (blocked vs kept avg net % · N · days). A signal that flips
  sign between regimes is reported as regime-dependent; regime-scoped switches are NOT re-fit, only read on DESIGN then HOLDOUT once.
- Prior hint (post-hoc, CONFIRMED_TOP_SHORT 2026-11-17): Frenzy shorts better in BTC daily bear (+0.97 vs −0.30) — a hypothesis, not evidence here.

## ADDENDUM 3 — bot regime labels + DI variables (operator, appended 2026-11-18 13:53 UTC)
Status at the time of writing (stated for transparency): the 1D tests, the 2D screen (74 variables) and the BTC 1d EMA regime split had
already been computed and read; DI and the bot regime label had NOT been built or looked at.
- **Bot regime label:** services/regime.py classify_btc_regime(ADX14, RSI14, EMA20 slope %) rebuilt exactly as the engine computes it on
  BTCUSDT 5m CLOSED bars at each entry. Parity check vs stamped entry_btc_regime on real orders; match rate reported.
  Read: EXHAUSTION by label (N · WR · avg · day P&L per half-year, day units), the cluster days' labels, the 1D tests and top 2D cells
  re-read within each label. A label switch is judged on the same expectancy bar (window units) with the family enlarged by the labels.
- **DI variables:** coin +DI, −DI, spread (+DI − −DI) and spread change over the last 1 h, on 15m and 5m at the signal bar (parity vs stamped
  entry_pos_di / entry_neg_di / entry_frenzy_di_spread where they exist); BTC +DI / −DI / spread on 5m and 1h. Added to the 1D theory set
  (pre-registered bad buckets: coin 15m spread > 0 · spread change 1 h > 0 · both) and to the exhaustive 2D screen, which is RE-RUN as one
  family (old + DI variables, same cuts / null / pass bar). Within-label re-reads as above. Nothing re-fit.
