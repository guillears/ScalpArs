# NEWSLEEVE 2026-11-21 — PRE-REGISTRATION (frozen before any outcome is computed)

Research only. No engine / config / AWS change. Scripts: `scripts/NEWSLEEVE_2026-11-21_*.py`. Data: `reports/NEWSLEEVE_2026-11-21_data/` (gitignored).
Source of the ideas: `reports/ROUTE_B_REPORT_2026-11-21.md` §4 (H1, H2, H5, H6). H3 / H4 need the order book (Route B only) and are not tested here.

## 0 · Prior work (STEP 0) and what is new

| idea | overlapping prior work | what was found | what is NEW here | keep? |
|---|---|---|---|---|
| H5 buy-climax → short | SPIKE_FADE (live sleeve: 5m vol ≥ 5× + RSI jump, coins ≤ $20M); EXHAUSTION_SHORT / DISTRIBUTION_SHORT (Frenzy coins, hours after the spike); DL 164 runaway longs mean-revert over 24 h | fade of pumps works inside the engine's own species | full-universe, 3.4-year, **taker-flow one-sidedness** (never a trigger leg; Route A only showed ρ −0.027 ≈ 2 bps); 15-min window instead of one 5m candle; coins of any size ≥ $10M | yes (new leg) |
| H5 sell-climax → long | SPIKE_BOUNCE (DL 2026-07-31 → OFF for good 08-10: 15 live fires, the edge lived only in the 5 fit trades); HOT dip separators (DL 169) | dump-bounce longs failed forward | taker-flow leg + multi-year history (SPIKE_BOUNCE never had a backtest) | yes, low prior |
| H1 long-liq cascade bounce | SPIKE_BOUNCE (same direction); DL 164 | failed | the **open-interest-falling** leg (forced closes) — never tested in this repo; read as the OI-proxy vs the same trigger without OI | yes, low prior |
| H2 short-squeeze fade | EXH / DIST / SPIKE_FADE (fade pumps); DL 164 (sub-minute cascade shorts: ~10 events, not evidence) | fades of pumps work in the engine species | the OI-falling leg (shorts forced out) as the trigger itself | yes |
| H6 funding ≤ −0.5 % → long | FUNDING_RULE 11-19 (EXH shorts at ≤ −0.5 % still earned +0.07 … +0.94 % net: a prior **against** the squeeze-long); DIST_FUNDING_EXIT 11-20 (holding through deep negative funding beats closing) | shorts paid funding and still won | a stand-alone long sleeve on ALL coins, not a filter on the short sleeves | yes, low prior |
| H6 funding ≥ +0.1 % → short | none (no study of positive-funding fades found) | — | new | yes |

No idea is a literal re-run of a refuted design; H5-sell / H1 / H6-long carry low priors from SPIKE_BOUNCE and FUNDING_RULE.

## 1 · Theory (one line each)
- **H5:** a 15-min burst of one-sided aggressive flow at ≥ 5× normal turnover exhausts the marginal buyer (seller); with no new information the price overshoots and partly reverts within hours.
- **H1:** a sharp drop with heavy volume while open interest FALLS means positions are being closed (long liquidations / stop-outs), not new shorts opening; forced selling is price-insensitive, so the price overshoots and bounces once the forced flow ends.
- **H2:** mirror — a sharp rise with OI falling = shorts forced to buy; the squeeze ends and the price fades.
- **H6:** an extreme funding rate means one side is crowded and pays to stay; the crowded side is fragile (squeeze risk) and the trade against it collects funding while it waits.

## 2 · Data
- 5m USDT-M klines with taker-buy quote volume, every USDT symbol listed on data.binance.vision (delisted included → survivorship largely removed; symbols missing from vision entirely are the residual bias), 2023-06 → 2026-11-17. 0 Binance weight.
- 5m `metrics` (sum_open_interest_value) — daily vision zips, fetched only for the pair-days a price/volume trigger touches (+ the previous day).
- Funding settlements — vision monthly fundingRate (reused from `reports/HIGH_ATR_2026-11-18_data/funding*`).
- The exit walk is on **5m bars** (stop first when a bar touches both; see §5). If an idea passes on 5m, its HOLDOUT is re-walked on 1m (vision daily 1m) before any CANDIDATE verdict.

## 3 · Universe
Binance USDT perps excluding BTC, ETH, stable / index symbols (USDC, FDUSD, TUSD, BUSD, USDP, DAI, BTCDOM, DEFI, FOOTBALL). At the trigger bar the coin's trailing 24 h quote volume must be ≥ **$10M** (capacity read by tier: $10–50M / $50–200M / ≥ $200M). Coin must have ≥ 7 days of history before the trigger.

## 4 · Triggers (frozen; evaluated on the CLOSE of a 5m bar; window = the last three 5m bars = 15 min)
- `qv15` = quote volume of the last 15 min; `usual` = median of `qv15` over the preceding 7 days (2,016 bars), lagged so the current hour is excluded. `ratio` = qv15 / usual.
- `share` = taker-buy quote / quote volume over the 15 min. `r15` = close / close 15 min earlier − 1.
- `dOI15` = OI value at the trigger bar's close / OI value 15 min earlier − 1 (5m metrics, nearest create_time ≤ the time).

| id | direction | trigger | small grid |
|---|---|---|---|
| H5B | SHORT | ratio ≥ M ∧ share ≥ S | (M, S) ∈ {(5, 0.60), (10, 0.65)} |
| H5S | LONG | ratio ≥ M ∧ share ≤ 1 − S | same |
| H1 | LONG | r15 ≤ −X ∧ ratio ≥ 5 ∧ dOI15 ≤ −Y | X ∈ {5 %, 8 %}, Y ∈ {1 %, 3 %} |
| H2 | SHORT | r15 ≥ +X ∧ ratio ≥ 5 ∧ dOI15 ≤ −Y | same |
| H6L | LONG | settled funding ≤ −0.50 % (per settlement, any interval) | frozen |
| H6S | SHORT | settled funding ≥ +0.10 % | frozen |

H1 / H2 are also shown WITHOUT the OI leg (descriptive: does the OI proxy add anything?). H6 trigger time = the settlement; entry = open of the 5m bar starting at the settlement + 5 min (the rate is published at settlement; no look-ahead).

## 5 · Entry, exits, costs
- **Entry:** open of the next 5m bar after the trigger bar closes (= the next 1m open), taker, adverse slip.
- **Exit grid (from theory):**
  - H5 / H1 / H2: TP ∈ {1, 2, 4} % · SL ∈ {2, 4} % · max hold ∈ {1 h, 6 h} → 12 cells.
  - H6: TP ∈ {3, 6} % · SL ∈ {3, 6} % · max hold ∈ {8 h, 24 h} → 8 cells.
- **Walk (5m):** SL checked before TP on every bar (conservative); a bar opening beyond the SL fills at its open. TP is a resting maker order and needs 0.05 % through. Time exit at the close of the last bar within the hold.
- **Costs:** taker 0.05 % per side (entry, SL, time exit), maker 0.02 % (TP). Slip per taker side by trailing 24 h volume: ≥ $200M 0.02 % · $50–200M 0.05 % · $10–50M 0.10 %; **×2 on the entry and on SL exits** (climax / stop moments). Funding: every real settlement inside (entry, exit], long receives −rate, short receives +rate.
- **Stress:** +0.20 % extra round-trip slip.
- **Dedup ("possibilities, not events"):** after a coin triggers, the same coin cannot trigger the same idea-direction again for the max hold of the grid (6 h for H1/H2/H5, 24 h for H6) — one possibility per coin per holding period. Counted three ways: raw trigger bars (events) · possibilities · distinct 15-min windows and distinct UTC days.

## 6 · Design / holdout
- **DESIGN 2023-07-01 → 2025-06-30:** pick ONE cell per idea-direction (trigger-grid × exit-grid) = highest avg net % among cells with N ≥ 30 possibilities and ≥ 15 days. The pick must also beat the **shuffled null**: 100 replicates where every possibility's entry is moved to a random 5m bar of the same coin, same UTC day (≥ 1 h away), same direction, full grid re-run; pick's avg must exceed the 95th percentile of the replicate max-cell avg.
- **HOLDOUT 2025-07-01 → 2026-11-17:** read ONCE for the pick (all other cells shown for information only, never to re-pick).

## 7 · Locked promotion bar for a NEW sleeve (ALL must hold on HOLDOUT at the pick)
1. Kelly > 0 at the one-sided 95 % Wilson WR floor (Kelly = p − (1 − p) / (avg win / |avg loss|)).
2. Avg net % > 0 with a **day-clustered** bootstrap one-sided 95 % lower bound > 0 (2,000 resamples of UTC days).
3. ≥ 2 of the 3 HOLDOUT half-years (25-H2, 26-H1, 26-H2 to 11-17) positive.
4. Avg still > 0 with +0.20 % round-trip slip.
5. No single pair and no single UTC day carries ≥ 50 % of the net profit.
6. Beats the random-entry control (same coin, same day, same direction, same exit): paired mean difference with day-clustered one-sided 95 % lower bound > 0.
7. DESIGN pick beat the shuffled null (§6).
Then the 30–50 % in-sample haircut on the DESIGN number, and for market-wide ideas (H1 / H2 / H6) the evidence is read in DAY units (N days ≥ 30 on HOLDOUT).

## 8 · Splits / reads (descriptive, not used to pick)
- BTC daily regime: EMA50 vs EMA200 on BTCUSDT daily closes (sign, the day before the trigger).
- Capacity: 24 h volume tier at the trigger; avg net % by tier.
- Event study: average path in trade direction from −60 min to +6 h around the entry (winners vs losers at the pick, vs the random control).

## 9 · Verdicts
- **CANDIDATE** (paper sleeve proposal; Route B must still confirm real liquidations / real book cost forward): all of §7 pass on 5m AND on the 1m re-walk.
- **OBSERVE** (frozen Route B line): the pick is positive on HOLDOUT and beats control but fails on confidence / half-years only.
- **REFUTED:** HOLDOUT avg ≤ 0, or fails the control, or DESIGN pick fails the null with HOLDOUT not positive.
