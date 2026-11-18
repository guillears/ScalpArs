# UNIFIED SHORT — pre-registration (frozen 2026-11-18 15:45 UTC, before any result of this study)

Research only. No bot file changes, no commits. This file is never edited after freezing; deviations go in the report's
"Deviations" section.

## Question
Operator: "~42 pumps and dumps a day, only 1–3 trades a day. There is something for SHORTs you are not finding — maybe a better
COMBINATION of the filters we already have." Is there a SHORT sleeve on pump / Frenzy coins with ≥ 5 trades/day and a positive,
consistent edge after real costs incl. funding — or a clear, evidenced "no"?

## Theory (written before the test)
1. Prior studies (EXHAUSTION_EXPAND) say the exhaustion edge is a STATE edge (volume still high ∧ spot share falling inside a Frenzy
   episode); faster triggers add ≈ 0. If true, a combination search over the union of triggers will mostly re-discover the state
   variables, and the frequency ceiling is set by how many coins sit in that state per day, not by triggers.
2. The short tail risk is squeezes (SQUEEZER: not predictable at entry). A combination can only raise frequency at the bar if some
   OTHER state (funding / OI / crowding / BTC regime) carries independent edge. Prior: unlikely; the test decides.
3. A model can only beat the best simple rule if interactions matter; with ~1,000–2,000 trading days of clustered outcomes, the model's
   effective sample is days, not rows. Expect shrinkage toward the simple state rule.

## Universe and pool (frozen)
- Spikes: `reports/EXHAUSTION_EXPAND_data/allspk_meta.csv` — EVERY spike of the bar rule (6-bar return ≥ 5 % ∧ 1h vol ≥ 20× normal ∧
  ≥ $2 M), 9,959 spikes, 645 pairs, no later shortlist selection. Window = spike bar → +72 h (inside the engine flag ≤ 96 h). Bars in
  overlapping windows belong to the most recent spike. vol24 ≥ $20 M at the signal bar. BTC/ETH and the study blacklist excluded.
- 5m klines from the EXHAUSTION_EXPAND sources (2023-05 → 2026-11-17). Trades 2023-07-01 → data end.
- CANDIDATE POOL = every 15m close bar in the window (the "state" rows, also the random-entry pool) PLUS every 5m-only trigger bar,
  with these trigger bits (all lag-safe: inputs close ≤ the signal close; entry = next 5m open):
  RSI14X 70→≤70 cross on 5m / 15m / 1h; RSI2 95→≤95 15m; BBRE 15m; STOCH 15m; VSW red-candle 15m; MACD / EMAX / ST / DON 15m (catalog);
  VWAP fail (T2: two consecutive 1h closes below the spike-anchored VWAP, first such close); VWAP stretch (LEAD1: 5m close ≥ VWAP × 1.01);
  Flatbase S4 (T3: first 4h-boundary close ≥ 4 h after spike below VWAP) and S1 (T4: 1h boundary) — without flat-base / confirm;
  FB confirm on 4h / 1h (first boundary close below the S close within 24 h, HIGHER_HIGH skip); structure break T1 (15m lower-high pivot
  then close below the low between peak and LH); divergence DV15 / DV1h (PEAK_SHORT_V2 definitions); double top DT (PEAK_SHORT_V2);
  second peak P2 (P = 8 %, R = 0, trigger RSI or K); wait-and-see WS2 / WS4 (first 15m close at which the running high has held ≥ W h
  and close ≤ peak × 0.95); time-after-spike TIME2/4/8/12/24; FLAGOFF (first 15m close ≥ 24 h after spike = allspk off).
  EXHAUSTION = RSI14X-15m ∧ vol_decay ≥ 0.6366 ∧ spot_share_chg < −0.0066 (spot coins).
- Parity: EXHAUSTION rows vs the EXHAUSTION_EXPAND REF build (same bars) and vs any real paper EXHAUSTION fills; Flatbase S4 vs the
  master Flatbase fills (entry bar match ± 5 min where the flat-base gate holds).

## Features at entry (frozen list; NaN allowed, coverage reported)
Every trigger gate above as a bit; vol_decay, vd24, spot_share1h, spot_share_chg, has_spot, OI ÷ vol24, OI chg 1h / 4h, OI-price
quadrant, funding last / hourly flag / est-change, top-trader position & account L/S, global L/S, taker 1h, premium last / chg1h / vs24h,
ATR15 %, ATR1h %, run %, dd from peak %, dist VWAP %, dist base %, hours since spike, RSI 5m / 15m / 1h / 4h, ADX 15m / 1h, EMA gaps
15m / 1h, ret 1h / 4h, vmult15, bar body / wick shares, BTC ret 1h / 4h / 1d, BTC RSI 1h / 4h, BTC 1h slope sign, BTC daily EMA50/200
regime, breadth, Frenzy count, hour UTC, coin age (censored), lvol24, q5. (Alpha listing and market cap: not point-in-time → excluded
and listed as blind spots.) Metrics rows: create_time (period START) ≤ entry − 5 min.

## ONE exit family (frozen)
TP ∈ {3, 5, 10} % resting maker (fills when low ≤ TP − 1 tick), × SL ∈ {A = min(1.5 × ATR15, 6 %), F = fixed 8 %}, gap-aware taker
(bar high ≥ SL → fill at max(SL, bar open)), 24 h time exit (taker). Six exits. Plus LIVE = EXHAUSTION's live exit
(TP min(20 × ATR15, 10 %) / SL min(1.5 × ATR15, 6 %) / 24 h) for parity only.
Costs: taker 0.05 % per taker leg, maker 0.02 %, slippage 0.10 % on every taker leg (entry, stop, time exit) — stress 0.20 % and 0.40 %.
Real funding incl. hourly coins (shorts receive positive funding); missing pair-month → 0.05 %/8 h against.
One position per pair per strategy (greedy in time).

## Blocks
- DESIGN 2025-01-01 → 2026-07-01: rules are searched ONLY here.
- Holdouts: H-OLD 2023-07 → 2024-12 (2023-H2 and 2024 — READ BEFORE by EXHAUSTION_EXPAND / SPOT_SHARE_DEEP / SQUEEZER etc. for other
  rules; not virgin for this question's ingredients, said so) · H-NEW 2026-07-01 → 2026-11-17 (also read by prior studies).
- 2023-H1: not buildable here (5m data start 2023-05, window + warm-up) → excluded, said so.
- Model: walk-forward, expanding window, monthly refit, 2-day embargo, first traded month 2024-01 (train from 2023-07).

## Combination search (frozen)
(a) Rules on DESIGN: the base pool = all candidate rows (state rows + triggers). Conditions = trigger bits (each trigger, and "any
    trigger") × feature thresholds at the DESIGN quintile edges (≤ q20, ≤ q40, ≥ q60, ≥ q80) and sign splits. Exhaustive 1D and every
    2D pair; greedy 3D from the top 50 2D. Scored at exit TP5·A (primary) and each of the six exits for the winner only. A rule is a
    candidate only if DESIGN trades/day ≥ 5 after one-position-per-pair. Score = DESIGN avg net %/trade at 0.10 % slip with day-clustered
    t ≥ 2. Family-wise null: the max day-clustered t over the whole search re-run on 200 circular time-shifts of the outcome vector.
(b) Model: ridge additive model on quantile-binned features (GAM) and numpy depth-2 boosted trees, target = net % at TP5·A; each day
    trade the top-k (k = 5, 10) and the top decile by prediction (threshold from the training window); compare with EXHAUSTION alone,
    overlap, and the union.
(c) Frontier: trades/day vs avg net for the best rule at each frequency and the model thresholds.

## Pass bar (all items, at the base 0.10 % slip unless stated)
1 trades/day ≥ 5 · 2 every half-year > 0 · 3 best-5 %-out > 0 · 4 > 0 at 0.20 % slip · 5 family-wise null p < 0.05 · 6 beats random
shorts in the same state (random 15m state rows from the same episode passing the rule's non-trigger conditions, same exit; p < 0.05 by
matched resampling) · 7 Kelly at the one-sided 95 % Wilson WR floor with daily clustering (n = distinct entry days) > 0 · 8 both
holdouts > 0 · 9 union with EXHAUSTION raises DCR at 0.18 × equity notional per trade without deepening the worst drawdown by > 5 points.
Verdict: BUILD (all 9) · OBSERVE (≥ 7 incl. 2, 4, 8) · NO.
In-sample haircut 30–50 % on any DESIGN number. Window units: one market moment = one observation (day clustering everywhere).

---
## ADDENDUM A — peer-relative "drops down the list" (frozen 2026-11-18 15:52 UTC, before any result on it; the main pool build was
running, no trade result of this study had been read)
Operator theory: the dashboard FRENZY list is ordered by 24 h quote volume; a coin sliding down as NEWER pumps take the volume = attention
rotating away, nobody left to defend the price.
- Flagged set at time T = every allspk spike with spike close ≤ T < spike close + 96 h (engine flag), BTC/ETH/blacklist excluded.
  Computed on the 15m grid (T = a 15m close); a 5m row uses the last 15m grid value ≤ T. vol24 = trailing 24 h quote volume through T.
- Variables at every candidate row: (1) `rk` = rank of the coin's vol24 among flagged coins (1 = top), `rk_best` = best rank since its
  latest spike, `rk_drop` = rk − rk_best; (2) `n_newer_above` = flagged coins whose latest spike is AFTER this coin's latest spike and whose
  vol24 is higher; (3) `vshare` = vol24 ÷ Σ vol24 of flagged coins, `vshare_rel` = vshare ÷ its max since the spike; plus `n_flagged`.
- Trigger family R (7 cells, one entry each = the FIRST closed 15m bar in the episode window meeting it): R-D2 / R-D3 / R-D5 = rk_drop ≥ D
  ∧ n_newer_above ≥ 1; R-K3 / R-K5 = first fall out of the top-K after having been in it ∧ n_newer_above ≥ 1; R-D3N2 = rk_drop ≥ 3 ∧
  n_newer_above ≥ 2; R-K5N2 = top-5 exit ∧ n_newer_above ≥ 2.
- Every R cell and every variable is read split by mcap ≤ $50 M vs > $50 M (proxy = TODAY's circulating supply × entry price from the
  NOSPOT_MCAP cache — look-ahead / unlock bias stated; unscored rows reported as their own group).
- Same exits, costs, blocks, holdouts, null, pass bar. Replays: PIPPIN (11-18), BLESS, HUMA, BREV, ZKC, ZKP — data ends 2026-11-17 in the
  caches; 11-18 is fetched from data.binance.vision daily files if available, else stated as not replayable.
- The R variables also enter the 1D / 2D / 3D search and the models as ordinary features (they are added before stage 2 runs).

---
## ADDENDUM B — "point of no return below the average price" (frozen 2026-11-18 16:52 UTC, before any result on it)
Disclosure: the pool's `dist_vwap_pct` (close vs the spike-anchored VWAP) was ALREADY a feature of the main 1D / 2D search, whose results
had been read before this addendum (best DESIGN 2D rules listed one `dist_vwap_pct ≤ −4.55 ∧ rsi1h ≥ 63.4` cell at +0.14). The no-return
curve, the V trigger family, the closes-below count and the time-since-first-close-below are new and unread.
Average price = spike-anchored VWAP (engine convention, typical price × volume from the spike bar), on closed 15m bars of the episode
(window spike → +72 h for the trigger; outcomes read up to +72 h after the trigger).
(A) Descriptive no-return curve, X ∈ {0, 1, 2, 3, 4, 6, 8, 10, 15} % below VWAP, at the FIRST 15m close reaching it: P(a 15m close back ≥
    the (still-anchored) VWAP within 6 / 24 / 72 h); median AND mean further move (close at +24 h / +72 h vs the trigger close, and the
    lowest low within 24 / 72 h); P(+10 % against before −10 % further). By half-year and by mcap ≤ / > $50 M (today's-supply proxy).
    Pre-stated reading: a "point of no return" needs P(reclaim 24 h) ≤ 20 % AND mean +24 h short move ≥ +2 % at the same X, in every
    half-year.
(B) Tradable family V: short at the next 5m open after the first 15m close ≥ X % below VWAP, X ∈ {2, 4, 6, 8, 10}, × {no count gate,
    n_below12 ≥ 8, n_below12 = 12} (closes below VWAP among the last 12 15m closes) = 15 cells. Exits: TP 3 / 5 / 10 % maker × SL
    {A = min(1.5×ATR15, 6 %), F = 8 %, W = VWAP at the signal × 1.005 (fixed price; skip if > 15 % away)}, 24 h; same costs, funding,
    one position per pair, blocks, holdouts. Pass bar = the main 9 items; the random-in-state null = random 15m rows of the same episode
    that are also ≥ X % below VWAP; family-wise null = max-T over the 15 × 9 cells via 200 circular shifts. Overlap / union with
    EXHAUSTION. New variables `n_below12`, `hrs_since_first_below` and `dist_vwap_pct` enter a 2D screen against every main condition
    (pairs involving at least one new variable; the old pairs were already run).
Replays: PIPPIN, ZKC, ZKP, EVAA, BLESS, HUMA, TLM, ZIL (11-18 not in the data; data end 11-17 16:50 UTC).
Cited, not repeated: POST_FRENZY_PATHS "VWAP break", CONFIRMED_TOP T2, this study's VWAPF (−0.10).
