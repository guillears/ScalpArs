# NO-SPOT HYBRID SHORT — pre-registration (frozen 2026-11-19 00:27 UTC)

Frozen BEFORE any result of this study. At freeze time no hybrid signal, trade or P&L has been computed. The feature (top-trader
position ratio 4 h change, DISTRIBUTION definition) has not been computed on these rows yet. Research only, BACKTEST only. No engine /
config / test change. (Study day label 2026-11-18, operator's day; wall clock already past midnight UTC.)

## Question (operator)
Test ONE hybrid short sleeve for FRENZY coins WITHOUT a Binance spot market: EXHAUSTION's trigger, with "top traders cutting longs" in
place of "spot buyers leaving". Then a deep winners-vs-losers study on its trades.

## Already known (stated before the test)
- **NS0** (the EXHAUSTION trigger on no-spot rows, no spot leg) loses: NOSPOT 2,795 trades −0.09 %/trade; NOSPOT_SLEEVE (with the 0.10 %
  slip floor) −0.07 (XE) to −0.25 (XA) %/trade. Random entries in the same episodes lose ≈ −0.4 %/trade.
- **NOSPOT_SLEEVE R4 TOP_DISTRIB** (top-trader position 1 h change ≤ −5 % ∧ global account 1 h ≥ 0 ∧ ≤ 20 % below the high; no RSI /
  vol_decay trigger) ≈ 0 on HOLDOUT (+0.15 at F1510, −0.09 to +0.20 across exits), swings ±2 % between halves, Kelly < 0. Not a DESIGN
  survivor.
- **NOSPOT_EXIT:** the best fixed exit on no-spot shorts was TP 4 / SL 10 (maker TP); still no ship.
- **DISTRIBUTION** (red ≥ 5× volume 15m candle + top-trader 4 h cut ≤ −0.01097 + RSI1h ≥ 74.07) is positive on no-spot coins too
  (≈ +0.8 % net on HOLDOUT, ≈ 0.35 trades/day). It is in PAPER (DL 586). The hybrid must ADD trades DISTRIBUTION does not already take.

## Theory (why futures positioning can stand in for the spot leg)
- On a spot coin, EXHAUSTION's spot-share fall says the real-money buyers (spot) left, the move is now futures-only leverage, and the
  top has no fresh demand behind it.
- A perp-only coin has no spot buyers to leave. Its fuel is leveraged longs. The best proxy for "the informed money is leaving" is the
  biggest accounts' net position: Binance "top traders" (top 20 % of accounts by margin). When their position long/short ratio falls while
  the RSI rolls over from > 70 and volume is decaying, the large holders are selling INTO the crowd — distribution. The crowd is left
  long and becomes the fuel of the dump.
- Why it could fail: (1) top-trader ratios move for hedging / basis reasons, not only distribution; (2) on perp-only coins the squeeze
  risk is high (shorts crowd, funding turns negative), so even a correct top can squeeze through a 6 % stop (NOSPOT_SLEEVE: 60–85 % of
  the loss is GAVE_BACK); (3) R4 already says the positioning signal alone is ≈ 0; the hybrid only helps if the RSI / vol_decay timing
  and the positioning cut are complementary.

## PRIMARY rule (frozen)
On the NOSPOT_SLEEVE universe (`reports/NOSPOT_SLEEVE_data/X.pkl`: no-spot FRENZY episodes, flag on → spike + 72 h, CLOSED 15m bars,
spot_status(pair, month) = 0, no 1000-prefix), SHORT at the next 5m open T after a closed 15m bar with ALL of:
1. RSI(14) on 15m crosses > 70 → ≤ 70 on that bar (`rsx15`);
2. vol_decay ≥ 0.6366 (`vd`);
3. 24 h quote volume ≥ $20M (universe);
4. **top-trader POSITION long/short ratio 4 h change ≤ −0.01097**, exactly `services/distribution.py pos_chg_4h`: km = the latest
   metrics row (`sum_toptrader_long_short_ratio`, stamp = create_time) with stamp ≤ T − 5 min and T − stamp ≤ 30 min; k4 = the latest
   row with stamp ≤ stamp(km) − 4 h and stamp(km) − stamp(k4) ≤ 4 h 15 min; chg = ratio(km) − ratio(k4) (ABSOLUTE difference, not %).
   Unreadable → no trade. Computed fresh from the raw metrics files (the X.pkl `topP_chg4h` column is a % change with a 35-min
   freshness, so it is NOT used for the rule).
- One position per pair (re-applied per exit). Name: **HYB**.

## Secondary variants (≤ 4, never selectable over the primary; reported beside it)
- **V1 1h:** the same with the 1 h position change (k1 = stamp ≤ stamp(km) − 1 h, within 1 h 15 min) ≤ −0.01097.
- **V2 ACCOUNT:** the top-trader ACCOUNT ratio (`count_toptrader_long_short_ratio`) 4 h change ≤ −0.01097 instead of the position ratio.
- **V3 cut −0.005** and **V4 cut −0.02** on the primary feature: a PLATEAU check only (the primary passes the plateau if both have the
  same sign as the primary on DESIGN).

## Exits (frozen)
- **PRIMARY XE** = the EXHAUSTION live exit: TP min(20 × ATR15 %, 10 %) maker / SL min(1.5 × ATR15 %, 6 %) / 24 h.
- **SECONDARY TP4 / SL10** (NOSPOT_EXIT best, maker TP 4 %, SL 10 %, 24 h).
- Both on the 1m engine (`scripts/nospot_exit_sim.py` cells REF_XE and A_TP4_SL10; 5m fallback flagged). Fills: taker 0.05 % per
  taker leg, maker 0.02 % on the TP, slip per taker leg = max(0.10 %, size slip model at $6.1k), TP only when traded through by 1 tick,
  stop checked first on the bar high, gap-aware stop fill, real funding (missing month 0.05 %/8 h against). Stress: 0.20 % per taker leg.

## Splits
DESIGN 2023-07-01 → 2025-07-01 · HOLDOUT 2025-07-01 → 2026-11-18, read ONCE. Half-years 23-H2 … 26-H2. Top-trader metrics are blank
for most of 2022 (and the universe starts 2023-07 anyway); rows with an unreadable change are not trades and their share is reported.
DESIGN is 100 % BTC-daily-bull (R1 = +1), so the regime split is HOLDOUT-only.

## Controls and null
- **Random entry:** each trade's entry moved to a uniform random eligible 15m-close bar of the SAME episode (same exit), 500 draws.
- **Family-wise null (the operator's):** among all NS0 signals (RSI cross ∧ vol_decay, before one-position), shuffle the
  position-change values across signals WITHIN the calendar month, re-apply the cut and one-position; statistic = full-span avg; family =
  {HYB, V1, V2, V3, V4} × {XE, TP4SL10}; p_fw = share of 1,000 shuffles whose family MAX ≥ the observed value of the cell. Also per
  cell (uncorrected).

## Ship bar (all must hold; HOLDOUT unless stated)
1. HOLDOUT avg > 0 with a day-block (10-day blocks, 2,000) 95 % lower bound > 0;
2. ≥ 2/3 of half-years (N ≥ 10) positive (full span);
3. positive at 0.20 % slip per taker leg;
4. best-5 %-out > 0 (drop the best 5 % of trades);
5. Kelly at the 95 % Wilson WR floor > 0;
6. beats the family-wise null (p_fw < 0.05);
7. no pair and no day ≥ 50 % of the total profit;
8. ≥ 8 distinct windows (UTC days) with a trade;
9. ADDS to the joint book (EXH + FB + L4 + DIST via `scripts/joint_short_sim.py` machinery) after removing the overlap with DIST:
   joint DCR (one book, coin lock, book 10, margin, depth cap) with HYB > without, and the non-overlapping HYB trades alone avg > 0.
Missing any item = OBSERVE (the item is named) or NOTHING (if HOLDOUT avg ≤ 0).

## Overlap with DISTRIBUTION
DIST trades = `reports/DIST_WL_2026-11-18_data/rows.pkl` kept by RSI1h ≥ 74.07 (as in joint_short_sim), `has_spot == 0` for the no-spot
part. A HYB trade is SHARED when a DIST trade on the same pair is open at the HYB entry or enters within ± 60 min of it. Reported: N
shared, avg shared vs not shared, and the marginal add = the joint book with HYB minus without (DIST takes priority on a shared coin).

## Winners vs losers (deep dive, after the primary verdict, on HYB trades)
- Labels: booked net (XE and TP4SL10) AND the path label: RUNNER (worked = +1 % before the stop distance, and net > 0) vs GAVE_BACK
  (worked, net ≤ 0); NEVER_WORKED shown apart.
- Variables: every entry-time column of X.pkl (OI, OI changes, OI / volume, funding level / sign / hourly, top-trader position /
  account level and changes, global account level and changes, taker flow, premium, RSI15 / RSI1h / DI15, distance from the high, VWAP
  gap, lower high, peak age, hours since the spike, vol_decay, volume multiple, ATR %, 24 h volume, Q5, mcap / turnover, returns, hour,
  BTC 1 h / 4 h / 24 h, BTC regimes R1 / R4) + the new absolute position / account changes + ADX15 / ADX1h computed from the klines.
  Coverage table and blind spots first.
- 1D at 3 granularities (sign / terciles / quintiles, cut points from DESIGN) and every 2D pair at terciles; family-wise label-shuffle
  null (500 1D / 200 2D). Screen on DESIGN; survivors (p_fw < 0.05 on DESIGN) → per half-year, both BTC regimes in window units, CI,
  concentration, 0.20 slip → HOLDOUT once.
- **Theory candidates first (frozen here):** T1 shorts crowded (global account L/S < 1) → squeeze risk, block; T2 funding < 0 → block;
  T3 OI 1 h rising (> 0) → longs still adding, block; T4 deeper top-trader cut (≤ DESIGN median of the HYB change) → better (dose);
  T5 BTC 24 h > 0 → squeeze tailwind, block; T6 close to the high (dd_hi ≤ DESIGN median) → better.
- Each survivor priced three ways: filter (block the losing side), size tilt (1.5× / 0.5×), conditional exit (the other exit on the
  losing side); DCR before → after at one size 0.61 and the joint add.
- **Entry × exit joint check:** a separator found under one exit must hold (same sign, kept side better) under the other, or it does
  not ship.

## Replays
PHAROS (11-18 20:30 / 21:15 UTC EXHAUSTION signals blocked as no-spot), PIEVERSE, PIPPIN, BLESS, JELLYJELLY: every HYB signal and its
outcome (open trades marked to the last bar), on the NOSPOT_SLEEVE live rows / X.pkl.

## Daily compound return
One size 0.61 × equity per fill, every calendar day of the block, r_d = Σ fill % × 0.61 / 100 over fills closed that day.
