# WILLY_HT — PRE_VOL_SURGE × theory variable (2D) — PRE-REGISTRATION

**Frozen 2026-11-21 22:55 UTC, BEFORE any 2D number was computed.** Research only. Nothing below may be re-fitted on the data it is tested on.
Inputs already known when this was written (and therefore NOT evidence for anything here): the 1D PRE_VOL_SURGE reads of
`WILLY_HT_TP1_PREVOL_2026-11-21.md` and the W/L screen of `WILLY_HT_WL_2026-11-21.md` (adx_delta ≈ twin of pre_vol).

## What the trade is (from the engine)
- **Spike (FRENZY flag):** a 5m close with a 30-min return ≥ +5 % AND last-hour quote volume ≥ 20 × the pair's normal hour AND ≥ $2 M.
- **WILLY entry A:** the first RED 5m bar (close < open) within 60 min of that spike → buy at market (long only).
- **HT = "high turnover":** the WILLY setups whose 24 h volume ÷ market cap ≥ 1.0 (WILLY's own turnover filter refuses them; HT takes them).
- **Today's armed filters:** dip ≥ 1.4 % below the spike high (DL 594) · FAILED_SPIKE = price must still be above the spike-anchored VWAP (DL 527).
- **Exit (DL 608):** TP +6 net resting maker · SL −4 net · 24 h.

## Theory (written before the test)
- **Bet:** buy the first pause of a fresh, violent, high-turnover pump, betting on a second leg (attention / momentum continuation).
- **Other side:** spike buyers taking profit, shorts fading the pump, market makers unloading inventory.
- **Why price should continue:** new attention brings new buyers over hours; the first red bar is a pause, not the top, while the pump is
  still above its own VWAP (the average spike buyer is in profit and not forced to sell).
- **What kills it:** the spike was the climax — the buyers who would carry the second leg already bought IN the spike. Then the pump
  "works" a little (late buyers + short covering push +2…+5 %), and then drifts down (Frenzy coins mean-revert: median backtest HT
  fill −6 % after 24 h, WILLY_HT_THEORY 11-19). This is exactly the real losers' path: +2…+5 within ~2 h, then through −4.
- **pre_vol15_vs60 is mechanically tied to spike age.** The spike needs ≥ 20 × normal hourly volume. If the spike bar sits inside the
  last 15 min, the 15-min window holds the spike and the 60-min base holds the quiet pre-spike hour → pre_vol is large by construction.
  If the spike is older (> ~20 min), the base window already contains the spike, and pre_vol > 1.6 needs volume STILL ACCELERATING
  after the spike. So the same "burst" means two different things:
  - **young spike + burst** = we buy the first crack of a vertical climax bar (the last buyers just bought) → BAD;
  - **older spike + burst** = a second volume wave after the spike = renewed demand → GOOD (or at least not bad).
- **A burst is GOOD** when it is fresh demand: aggressive buyers, new positions opening (OI up), price holding the top of its range,
  a supportive tape (BTC rising), not stretched far above the buyers' average cost.
- **A burst is BAD** when it is a climax: the volume is sellers hitting bids (distribution), OI falls (shorts covering — a squeeze ends
  when the covering ends), the burst range closes near its low (rejection wick), BTC is falling (no tide), price stretched far above
  the VWAP (lots of profit to take).

## Pre-registered second variables (6), all entry-time, all rebuilt the same way for real AND backtest fills
Source: 1m klines of the fill's pair (vision daily files in `STATE_EXIT_2026-11-21_data/k1m`: base volume v, quote volume qv,
taker-buy quote tbq), aligned vision metrics (`OI_TS_AUDIT_2026-11-21_aligned.py`, live lag 5 min), BTCUSDT 1m.
"Window" = the 15 1m bars OPENED before the fill minute (= pre_vol's recent window; all closed before the fill → no look-ahead).

| # | Variable | Definition | Natural cut (granularity 1) | Expected sign — burst is BAD when … | Why |
|---|---|---|---|---|---|
| V1 | `spike_age_min` | (fill minute − spike bar open) in minutes | spike bar inside the 15-min window ⇔ age ≤ 20 min | **young** (≤ cut) | burst = the climax bar itself; we buy its first crack |
| V2 | `tbuy15` | Σ tbq ÷ Σ qv over the window | 0.50 | **low** (< cut) | sellers are the aggressors → distribution, not demand |
| V3 | `oi_chg30` | % change of open-interest value, aligned snapshot ≤ T−5 vs the snapshot 30 min before it | 0 | **falling** (≤ 0) | short covering, not new longs → the squeeze ends when covering ends |
| V4 | `cloc15` | (last close − window low) ÷ (window high − window low) | 0.50 | **low** (< cut) | the burst range closed near its low = rejection on volume |
| V5 | `btc_ret60` | BTCUSDT % change, close of the bar before the fill minute vs 60 min earlier | 0 | **BTC falling** (≤ 0) | no tide; MARKET-WIDE → counted in day units |
| V6 | `vs_vwap` | % of the last close above the spike-anchored VWAP (1m Σ typical × v ÷ Σ v from the spike bar open) | pooled median (label-free) | **stretched** (> cut) | the average spike buyer sits on a big profit → supply |

Granularities: ① natural cut above · ② the label-free median over all 324 fills (real + backtest) · ③ outer terciles (label-free).
Filter cell tested for each Vi: **block = burst (pre_vol > 1.60, frozen) ∧ Vi on its BAD side**; everything else is kept.
Also shown: the 2 × 2 table (burst / calm × good / bad) and the within-burst contrast (good − bad avg).

## Tests and decision rule (frozen)
- Outcome: each fill's TP +6 / SL −4 / 24 h net % from the DL 608 re-walk engine (`grid_tp1.pkl`, G6), one size $3,050.
- Blocks: ① batch 12 · ② master 12 · ①+② real 24 · ③ backtest DESIGN 33 / **HOLDOUT 267 (the judge)** / half-years.
- **SUPPORTED** = on HOLDOUT the within-burst contrast has the predicted sign AND the blocked cell's avg < the kept avg AND the filter's
  Δ$ > 0, AND the real 24 agree in sign (pooled; master / batch shown apart). **REFUTED** = HOLDOUT contrast has the opposite sign or
  Δ$ ≤ 0. **UNTESTABLE** = < 80 % of fills scored.
- A SUPPORTED pair then faces: day-block bootstrap, per-pair concentration, leave-one-out, BTC daily trend sign split, the locked
  "Pattern C → FILTER — EXPECTANCY bar" on the blocked cohort, the 30–50 % haircut, window units for V5.
- Even a pass becomes only an **OBSERVE** line with a frozen threshold. Nothing is armed.
- **Screen:** pre_vol > 1.60 × EVERY rebuilt feature (backtest) / every numeric entry_* stamp (real 24), each at median + terciles,
  both sides blocked → filter Δ avg; shuffled-label null (labels permuted within block, 1,000×). Purpose: do the 6 theory picks beat
  what luck finds? (statistic: HOLDOUT Δ avg of each pick vs its own null; and the best-of-6 vs the null's best-of-6).
