# NOSPOT_MCAP — PRE-REGISTRATION (frozen 2026-11-18 12:55 UTC, before any result of this study)

Research only. No bot code, config, template, test or CLAUDE file is touched. No commit.

## What has already been seen (honesty list)
- Every block 2023-07 → 2026-11-18 has been read by EXHAUSTION_EXPAND / SPOT_SHARE_DEEP. I have read their reports, incl.
  - X1 ("no spot filter" = REF + spot-blocked + no-spot rows): 8.2/day, +0.22 %/trade, fails best-5 % and 0.20 % slip.
  - SPOT_SHARE_DEEP T2c: no-spot rows of the base R0 trigger (NO vol_decay cut): C +0.66 (177) · A −0.13 · DESIGN −0.15 · B −0.26.
  - HUMA / BLESS 11-15 → 17 replays (BLESS no-spot: two −5.1 % stops, one +4.93 % TP under X1/X3b).
- So **every 2023-07 → now result here is a ROBUSTNESS read**, not a clean test.
- **HOLDOUT-E = spikes 2023-01-01 → 2023-07-01** (trades entered < 2023-07-01) has never been built or read by anyone. If the build
  succeeds it is read ONCE, in the final run, for: REF, NS0, the union, and the sign of every pre-registered size cut. If it fails to
  build, that is stated and there is no clean read.

## Base (parity first)
- Rows: `reports/EXHAUSTION_EXPAND_data/allspk/all.pkl` (ALL bar-rule spikes, no later-flag selection, window spike → +72 h).
- Parity target: REF|EA on all-spikes ≈ +0.43 %/trade, ≈ 3.3/day, 23-H2 ≈ +0.50 (EXHAUSTION_EXPAND §0 (3)). If parity misses by
  > 0.03 %/trade or > 3 % of N, stop and fix before any cohort read.
- Trigger (all cohorts): Frenzy window row ∧ complete 15m close ∧ RSI14(15m) cross 70 → ≤ 70 ∧ vol_decay ≥ 0.6366 ∧ vol24 ≥ $20M;
  entry next 5m open. Exit EA: TP = min(3 × ATR15, 5 %) resting maker, filled only through by ≥ 1 tick; SL = min(1.5 × ATR15, 5 %)
  gap-aware; 24 h time exit. One position per pair per cell. Real funding (incl. hourly-funding coins), taker 0.05 %, maker 0.02 %.
- Slippage: primary = live model slip % = max(0, −0.042 + 0.276·√(6000 ÷ Q5)) on taker legs (shipped size ≈ $6k notional). Also
  shown: $500 (parity with EXPAND), flat 0.10 / 0.20. The engine depth guard needs the live order book → NOT rebuildable; stated.

## Cohorts (A)
- **has_spot** per (pair, entry month) from the spot 15m caches: file with data = 1 · file "missing" (data.binance.vision has no spot
  zip for the same base) = 0 · never fetched = unknown (counted, excluded, shown separately).
- **REF** = EXHAUSTION_SHORT: has_spot ∧ spot_share_chg finite ∧ < −0.0066.
- **NS0** = has_spot = 0, nothing replaces the spot filter.
- **NS-P** = NS0 ∧ perp-only proxy **OI 1 h change ≤ 0** (OI $ from Binance `metrics` 5m, create_time = period start, last row
  ≤ entry − 5 min vs the row 1 h earlier; positions closing ≈ the "buyers fading" analogue of spot share falling). Unscorable → no
  trade. If < 70 % of NS0 trades are scorable, NS-P is descriptive only.
- **U0 = REF ∪ NS0**, **UP = REF ∪ NS-P** (one position per pair across the union).

## Pass bar for adding no-spot coins (A) — all at the $6k slip model, on NS0 (or NS-P) alone AND on the union
1. every half-year (23-H2 … 26-H2, + 23-H1 if HOLDOUT-E builds) avg > 0 (half-years with N < 10 noted, not failed);
2. best-5 %-removed avg > 0;
3. avg > 0 at flat 0.20 % slippage;
4. Kelly f at the one-sided 95 % Wilson WR floor with n = distinct entry days (daily clustering, EXPAND `stats`) > 0;
5. union vs REF alone: union avg ≥ 0.70 × REF avg AND union Kelly f ≥ 0.70 × REF Kelly f;
6. per-pair and per-day share of the cohort's total $ < 25 %;
7. HOLDOUT-E (if built): NS0 avg > 0 (if N ≥ 10; else "no read").
Reported, not gated: squeeze tail (max loss, worst 1 %, hourly-funding coins and their funding paid), trades/day, overlap with REF.
Verdict ADD only if NS0 or NS-P passes 1–6 (and 7 when readable). NS-P is a second test → if only NS-P passes, the verdict is OBSERVE.

## Size (B)
- **Market cap at entry** = circulating supply × entry price ÷ multiplier, with the supply from the bot's own source
  (`services/mcap_service.py` endpoint = CMC data, fields cs / mc / fdmc / ts; cache `reports/cache_vol_mcap/bapi*` + fetch for missing
  pairs at ≥ 1.5 s spacing). **FDV** = total supply × entry price ÷ multiplier. This assumes constant supply (validated 2026-01 → 10 in
  FRENZY_VOL_MCAP: median 1.00); drift is re-checked on the CoinGecko 365-day cache. Delisted coins have no supply → unscored
  (NOT "rest"). Live stamps (`entry_mcap_usd`) used as a check where they exist.
- **Proxies (full span):** vol24 (perp 24 h quote volume), Q5, OI $ at entry (metrics), turnover vol24 ÷ mcap, listing age
  (days since the pair's first 5m kline in the caches), Alpha listing (today's Binance Alpha token list — look-ahead, descriptive only).
- **Cuts (natural only, frozen):** mcap ≤ $50 M vs > $50 M (operator's value) · median split · terciles. Every other variable:
  median split + terciles + sign where signed. No optimised thresholds.
- **Cohorts read:** REF kept trades, NS0, U0. Coverage table first; unscored shown separately.
- **Tests:** 1D per variable × cut; 2D size-variable median × every other entry variable median (vd, ATR %, ssc, RSI prev, hour,
  BTC-free variables available in the rows); shuffled max-stat null (2,000 shuffles of the P&L over the scored rows, max |Δ avg|
  over every 1D variable × cut) → family-wise p; per half-year direction; per-day concentration (market-wide variables in day units).
- **A size BLOCK rule passes only on the CLAUDE.md expectancy bar:** blocked cohort WR < the cohort's breakeven WR · avg < 0 with 95 %
  confidence by a day-clustered bootstrap · ≥ 8 distinct days, no day / pair ≥ 50 % of the loss · N ≥ 15 · same sign in ≥ 5 of 7
  half-years with N ≥ 5 · family-wise p < 0.05 · HOLDOUT-E same sign (if readable). Then the 30–50 % haircut. Otherwise OBSERVE.
- **The operator's hypothesis ("< $50 M dumps harder after the peak") is read as:** the ≤ $50 M side has a HIGHER avg than > $50 M on
  REF and on NS0. A size MULTIPLIER is NOT on the table (Pattern W bar N ≥ 30, WR ≥ 70 %, needs its own study) — at most OBSERVE.

## Also reported
- Event-study chart: post-signal gross short path (−2 h … +24 h) by mcap bucket (≤ $50 M / $50–250 M / > $250 M / unscored) and
  spot vs no-spot.
- Replays 2026-11-16 → 11-18 of PIPPIN, BLESS, AIOT, SPACE, HUMA, ZKC, BREV (from the newest caches; if klines are missing, fetched
  from data.binance.vision daily / the public klines endpoint at ≤ 150 weight/min).
- Live: every EXHAUSTION_SHORT paper fill in the newest export.
