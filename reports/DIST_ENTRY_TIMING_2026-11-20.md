# DIST_ENTRY_TIMING: should DISTRIBUTION_SHORT enter earlier? (2026-11-20, BACKTEST + 4 paper fills)

**Verdict.**
- **(B) Deciding 1–2 minutes early on the still-forming candle: NO SHIP.** Both primary variants fail the frozen gate, and both are *worse* than today's close-entry.
  - **VE1** (enter at T−1 if the forming candle already qualifies, else at T): HOLDOUT Δ = **−230 %** summed (CI −453 … +10).
  - **VE2** (earliest of T−2 / T−1 / T): HOLDOUT Δ = **−331 %** (CI −584 … −65).
  - DESIGN Δ is about 0 for both. Only 2 of 7 (VE1) and 3 of 7 (VE2) half-years are positive.
- **(A) Latency is a real cost. Entering at the close is what the backtest assumes, and every minute of delay hurts.**
  - On the same fills, a 1-minute-late entry earns **−0.32 %/trade** (FULL; CI −0.55 … −0.11) and −0.48 % on HOLDOUT. 2 minutes late earns −0.52 %/trade.
  - Today's bot fills land **17–27 s** after the close (82 s once).
  - Sub-minute price at about 20–30 s after the close: **+16 to +30 bps worse** for the short. This comes from a 40-fill aggTrades sample, so it is **UNVERIFIED**: mean, wide CI that includes 0.
  - On 3 of the 4 real paper fills the price drift during the delay was **+41 / +71 / +43 bps** against us.
  - The lever is **making the at-close entry faster**, not deciding earlier. That is an engineering item, not a rule change.

Pre-registration: `reports/DIST_ENTRY_TIMING_2026-11-20_PREREG.md`, frozen before any result. No free parameter: k ∈ {1, 2} was fixed, and DESIGN and HOLDOUT came out of one run.
Scripts:
- `scripts/dist_entry_timing_build.py` (`--stage legs` → `fetch` → `partial`)
- `scripts/dist_entry_timing_eval.py`
- `scripts/dist_entry_timing_sens.py`
- `scripts/dist_entry_timing_latency.py` (`--real`, `--sample`)
- `scripts/dist_entry_timing_realtrig.py`

Data: `reports/DIST_ENTRY_TIMING_data/`:
- `cand.pkl`: every candidate candle with its T−k legs
- `sim.pkl`: every simulated entry
- `trades.csv.gz`: kept fills per variant
- `latency_real.csv`, `latency_sample.csv`, `realtrig.csv`

Chart: `reports/DIST_ENTRY_TIMING_2026-11-20.png`.

---

## 1. Question
KERNELUSDT DIST short, 2026-11-20:
- The signal candle closed 12:30:00 and the fill came 12:30:27 at 0.05114.
- One minute earlier the price was about 0.0516, roughly 0.5–1 % better for a short.

Two questions follow:
- (A) What does the bot's execution delay cost?
- (B) Would deciding on the still-forming candle at T−1 / T−2 minutes pay, once we count the candles that look like a signal at T−1 but are not one at the close?

## 2. Method
- **Rule (DL 586), unchanged:**
  - EXHAUSTION-tracked FRENZY coin, ≤ 72 h after the spike, 24 h volume ≥ $20M.
  - Red 15m candle, quote volume ≥ 5 × the prior-20 mean.
  - Top-trader position L/S 4 h change ≤ −0.01097, using the metrics row ≤ T − 5 min.
  - 1h RSI ≥ 74.07, on complete hours.
- **Backtest entry convention** (`unified_short_build` / `dist_exit_sim`): entry_ms = T = the candle close, at the open of the first 1m bar at T, which is the first trade after the close.
- **Exit B:** TP min(20 × ATR, 10 %) resting maker / SL min(1.5 × ATR, 6 %) / 24 h. The simulator is `dist_exit_sim.run_rows` with the REF_B cell, code unchanged. Costs: slip max(0.10 %, model) + taker/maker fees + real funding. One position per coin.
- **Population, including false triggers:**
  - Every closed 15m candle in every FRENZY episode window of the UNIFIED_SHORT pool (1.83 M candles).
  - Every candle whose *full* quote volume is ≥ 5 × the mean was scanned: 25,676 candles on 637 coins. A forming candle's volume can only be ≤ the full candle's, so no early volume trigger can sit outside this set.
  - 5,169 of them pass the position + RSI legs at some k. Those were rebuilt from 1m klines *with quote volume* (8,173 pair-days from data.binance.vision, 0 API weight).
  - Check: 1m quote volume summed = the 5m data on 100 % of candles, and the T0 trigger from 1m reproduces the pool's 1,610 / 1,610 signals exactly.
- **Legs at T−k:**
  - Red so far (close at T−k < candle open).
  - Volume so far ≥ 5 × the prior-20 mean.
  - Position change recomputed at T−k with the same lag rule.
  - RSI on hours closed ≤ T−k.
  - 24 h volume ≥ $20M including the partial candle.
  - ATR with the partial candle as the last bar.
  - Entry at the 1m open at T−k.
  - Approximations: (a) the episode window comes from the closed row; (b) the metrics archive has grown since the pool was built, so k = 0 keeps the pool's value and k ≥ 1 is set to NaN wherever the pool's T0 value is NaN. No early trade is judged on data the study did not have. 40 of 23,211 values differ slightly between the two archive versions.
- **Variants:**
  - **V0** = today (T0).
  - **VE1** = T−1, else T0.
  - **VE2** = earliest of T−2 / T−1 / T0.
  - Descriptive only: P1 / P2 (T−1 only / T−2 only, which miss late-forming candles) and LAT1 / LAT2 (enter 1 / 2 min after the close).
- **Statistics:**
  - Σ net % at ONE size.
  - Δ vs V0 per entry day, with a 10-day block bootstrap.
  - DCR at 0.61 (as in DIST_EXIT).
  - Splits: DESIGN 2023-07 → 2025-06, HOLDOUT 2025-07 → 2026-11-18.

## 3. Parity (replica vs REF_B)
- V0 = 1,547 kept fills = REF_B 1,547, **same (pair, entry) set on 100 %**.
- Per-fill net within 0.01 % on **1,546 / 1,547**. The one exception is MIRAUSDT 2026-03-01: TP in REF_B vs time exit here, because one extra 1m day file is now on disk.
- Σ net 1,440.6 vs 1,442.4.

## 4. Part A — latency

### 4a. Real paper fills (aggTrades; KERNEL from fapi aggTrades)

| fill | delay | T price (first trade ≥ T) | market at fill | paper fill | drift during delay | paper walk | total vs T | T−1 price vs fill |
|---|---|---|---|---|---|---|---|---|
| SENT 11-19 04:15 | 16.7 s | 0.02220 | 0.02211 | 0.022077 | **+41 bps** | +15 | +56 | +33 |
| NAORIS 11-19 09:15 | 17.3 s | 0.03273 | 0.03250 | 0.032491 | **+71 bps** | +3 | +73 | +218 |
| EVAA 11-19 14:30 | 82.0 s | 4.9957 | 5.0469 | 5.0443 | **−101 bps** | +5 | −96 | −157 |
| KERNEL 11-20 12:30 | 27 s | 0.05140 | 0.05118 | 0.05114 | **+43 bps** | +8 | +51 | +104 |

How to read the table:
- **+ = a cost to the short** (the fill is below the T price).
- The candle close in the decision row equals the first trade after T to within 1 tick, so the backtest's entry price is the right reference.
- The 3 fast fills each lost 41–71 bps to the delay; the one slow fill gained.
- N = 4: anecdote, not a rate.

### 4b. Backtest fills, sub-minute
Random sample: 40 REF_B fills (seed 1120), aggTrades. Cost = the price at T vs the price at T + s, in bps; + = worse for a late short. **UNVERIFIED** (N = 40).

| delay | 5 s | 10 s | 15 s | 20 s | 30 s | 60 s |
|---|---|---|---|---|---|---|
| mean bps | −1 | +4 | +14 | +16 | +30 | +31 |
| median bps | −9 | 0 | +9 | +19 | +10 | +16 |
| 95 % CI of mean | −11 … +10 | −10 … +18 | −10 … +37 | −7 … +37 | −2 … +66 | −14 … +79 |

### 4c. Backtest fills, full Exit B, entry 1 / 2 minutes late
The same candles were re-simulated. TP and SL levels are set from the later fill. This is the 1m resolution limit: a 17–27 s delay cannot be simulated directly.

| | N | avg net at T | avg net late | paired Δ %/trade | 95 % CI |
|---|---|---|---|---|---|
| T+1 min, DESIGN | 590 | +1.17 | +1.11 | −0.06 | −0.31 … +0.18 |
| T+1 min, HOLDOUT | 1,019 | +0.86 | +0.38 | **−0.48** | −0.76 … −0.18 |
| T+1 min, FULL | 1,609 | +0.97 | +0.65 | **−0.32** | −0.55 … −0.11 |
| T+2 min, FULL | 1,609 | +0.97 | +0.45 | **−0.52** | — |

**What this means:**
- After a qualifying close, the price keeps falling for the next minute or so. Each minute of delay gives that move away.
- The bot's 17–27 s delay costs roughly **0.15–0.3 %/trade**. This is interpolated, UNVERIFIED, and comes on top of the backtest's +0.93 %/trade.
- That is 15–30 % of the sleeve's edge, and the backtest does not include it.
- Live expectancy should be marked down by this amount until the delay is shortened.

## 5. Part B — earlier trigger

### 5a. Variants (Σ net % at one size; Δ vs V0 with day-block 95 % CI)

| variant | block | N | days | WR | avg %/trade | Σ net | **Δ Σ vs V0** | CI | DCR @0.61 | max DD |
|---|---|---|---|---|---|---|---|---|---|---|
| **V0 today** | DESIGN | 559 | 299 | 42.0 | +1.03 | 574 | 0 | — | 0.41 | −34.5 |
| | HOLDOUT | 988 | 412 | 42.7 | +0.88 | 867 | 0 | — | 0.85 | −54.9 |
| **VE1** T−1 else T | DESIGN | 598 | 308 | 40.8 | +0.94 | 564 | **−9** | −134 … +106 | 0.40 | −31.0 |
| | HOLDOUT | 1,092 | 424 | 40.6 | +0.58 | 637 | **−230** | −453 … +10 | 0.56 | −60.7 |
| **VE2** earliest T−2/T−1/T | DESIGN | 606 | 310 | 40.6 | +0.94 | 571 | **−3** | −124 … +114 | 0.40 | −33.1 |
| | HOLDOUT | 1,118 | 429 | 39.9 | +0.48 | 536 | **−331** | −584 … −65 | 0.44 | −66.9 |
| P1 T−1 only | HOLDOUT | 825 | 380 | 38.9 | +0.29 | 239 | −628 | −923 … −337 | 0.12 | −70.3 |
| P2 T−2 only | HOLDOUT | 723 | 360 | 38.6 | +0.20 | 143 | −724 | −1,072 … −397 | 0.04 | −67.2 |
| LAT1 T+1 | HOLDOUT | 985 | 411 | 39.5 | +0.38 | 371 | −496 | −770 … −204 | 0.28 | −78.5 |
| LAT2 T+2 | HOLDOUT | 986 | 411 | 38.4 | +0.22 | 215 | −652 | −1,001 … −294 | 0.08 | −78.2 |

Trading possibilities: one per coin per hold. FULL counts (trades · days · episodes):

| variant | trades | days | episodes |
|---|---|---|---|
| V0 | 1,547 | 711 | 1,391 |
| VE1 | 1,690 | 732 | 1,495 |
| VE2 | 1,724 | 739 | 1,519 |

The earlier variants add about 10 % more trades, and those extra trades are the problem (5b).

### 5b. Where the early variants lose: false triggers, matched candles, late-forming candles

**FULL**, inside the kept VE1 / VE2 sets:

| group | VE1 N | WR | avg net | Σ | VE2 N | WR | avg net | Σ |
|---|---|---|---|---|---|---|---|---|
| early entry, candle also qualifies at the close | 1,081 | 40.3 | +0.61 | +664 | 1,095 | 39.9 | +0.56 | +614 |
| **FALSE early trigger** (does not qualify at the close) | **147** | **25.2** | **−1.34** | **−197** | **182** | **24.2** | **−1.50** | **−274** |
| T0 fallback (formed only in the last minute(s)) | 462 | 46.3 | **+1.59** | +735 | 447 | 47.2 | **+1.72** | +767 |

False triggers by block:
- VE1: DESIGN 39 · +0.12 %; HOLDOUT 108 · **−1.87 %**, WR 23 %.
- VE2: DESIGN 47 · −0.27 %; HOLDOUT 135 · **−1.93 %**, WR 23 %.

Why a false trigger fails at the close, VE1 (153 candles):
- 78: the candle closed green or flat. The last minute bounced.
- 47: the position leg failed at T.
- 31: the RSI leg failed at T.

**Matched candles** (qualify at both T−k and T, each simulated alone):

| | N | entry price gain vs T (avg / median) | paired Δ net %/trade | 95 % CI | early better |
|---|---|---|---|---|---|
| T−1, FULL | 1,127 | +0.13 % / +0.01 % | −0.05 | −0.26 … +0.15 | 30 % |
| T−2, FULL | 969 | +0.22 % / +0.07 % | −0.07 | −0.30 … +0.16 | 32 % |

What this shows:
- On candles that would have fired anyway, the earlier price is better on average but not at the median.
- Under Exit B, whose levels are set as % of the fill, that price gain does not turn into P&L (Δ ≈ 0).
- Sensitivity, `dist_entry_timing_sens.py`: using the closed candle's ATR for the early entry gives the same Δ (−0.06 / −0.09). The partial-candle ATR is not the cause; the median ATR ratio is 1.00.

**Late-forming candles:** 482 of the 1,610 T0 signals do not qualify at T−1:
- 261: the position leg (the 5-min metrics lag moves the readable row back).
- 170: volume below 5 × until the last minute.
- 89: not red yet.

As a group these are the **best** DIST trades (+1.6–1.7 %/trade). Deciding only early (P1 / P2) throws them away, which is why P1 / P2 are the worst variants.

### 5c. Per half-year (Δ Σ net % vs V0)

| half | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | positive |
|---|---|---|---|---|---|---|---|---|
| VE1 | −64 | +25 | −16 | +46 | −64 | −41 | −125 | **2 / 7** |
| VE2 | +0.5 | +18 | −35 | +13 | −109 | −11 | −211 | **3 / 7** |
| LAT1 | −78 | +2 | −0 | +71 | −306 | −80 | −110 | 2 / 7 |

### 5d. Concentration and leave-one-out
**HOLDOUT** (the Δ is negative, so "top pair" means the largest single loss):

| variant | Δ | pairs changed | top pair (share) | without the top pair | worst LOO |
|---|---|---|---|---|---|
| VE1 | −230 | 272 | METUSDT −33 (14 %) | −197 | −256 |
| VE2 | −331 | 292 | AKEUSDT −39 (12 %) | −292 | −357 |

The loss is broad: no pair carries more than 15 %, and no single pair flips the sign.

### 5e. Pass check (frozen)

| gate | VE1 | VE2 |
|---|---|---|
| HOLDOUT Δ > 0 with CI lower bound > 0 | ✗ (−230) | ✗ (−331) |
| DESIGN Δ > 0 | ✗ (−9) | ✗ (−3) |
| ≥ 5 / 7 half-years positive | ✗ (2) | ✗ (3) |
| top pair < 50 % of Δ | ✓ (loss is broad) | ✓ |
| false-trigger expectancy reported | −1.34 %/trade (N 147) | −1.50 %/trade (N 182) |

No positive DESIGN Δ, so there is nothing to haircut. **FAIL. No ship, and no observe line**: the sign is negative, not borderline.

## 6. Real-fill cross-check (4 paper signals, fapi reads, `realtrig.csv`)
Using the engine's own `pos_chg_4h`:
- SENT, EVAA and KERNEL would also have triggered at T−1 and T−2.
- NAORIS would not: at T−1 it was not yet red and was at 4.7 × volume.
- The T0 volume multiples reproduce the bot's journal (29.4 / 5.2 / 6.2 ×).

Entry at T−1 vs the T price, + = better for the short:

| | SENT | EVAA | KERNEL |
|---|---|---|---|
| T−1 vs T | −22 bps | −61 bps | **+53 bps** |
| T−2 vs T | +18 bps | +11 bps | +37 bps |

KERNEL is the case the operator saw. On the other two coins, entering early would have been worse, which matches the backtest: the early price advantage has a median of about 0. The delay cost (§4a) is the consistent part.

## 7. Chart
`reports/DIST_ENTRY_TIMING_2026-11-20.png` shows the average gross short P&L path, with no exit, anchored at T−2, T−1, T and T+1 min:
- **Left panel:** kept T0 fills (N 1,547). The T−2 / T−1 lines sit above T because these fills are *conditioned on having qualified at the close*, and many fell in the final minutes. This is hindsight: you only know at T which candles qualify. The T+1 line sits about 0.3 % below T for the whole 4 h, which is the latency cost.
- **Right panel:** the 182 false early triggers (VE2). They drift *up* against the short, to about −1.5 % within 2 h, at every anchor.

## 8. Meaning
- **The close is the information.** Of the extra signal an early decision "buys":
  - about 10 % are candles that bounce or lose a leg in the last minute. Those lose about 1.5 %/trade with a 24 % WR;
  - the rest are the same trades at a price that is better on average but not at the median.
  - Meanwhile the best trades (late-forming, +1.6 %/trade) only exist at the close.
- **Latency is the actionable part.** The backtest's edge assumes an entry at the first trade after the close. A minute of delay costs −0.32 %/trade (FULL) and −0.48 % (HOLDOUT). The real fills land 17–27 s late, at roughly 0.15–0.3 %/trade (UNVERIFIED interpolation). Shortening the pass from close to order is worth more than any rule change tested here. For example: pre-read the 15m history and the position-ratio rows before the close, and judge DIST first in the pass. EVAA's 82 s was the second-pass slot after EXHAUSTION.
- **Live expectancy:** read DIST's backtest +0.93 %/trade as about +0.6–0.8 % at today's latency. The haircut belongs in the Kelly / sizing read.

## 9. Blind spots
- **Sub-minute cost is estimated from N = 40 aggTrades fills plus 4 real fills.** The CI includes 0 below 30 s. The 1m sim can only show 0 / 60 / 120 s.
- **Early ATR / vol24 definitions** (partial candle as the last bar) are design choices. The ATR sensitivity shows no effect; vol24 was not varied.
- **Episode window and base_ok** come from the closed row. Candles excluded at close for vol24 < $20M but above it at T−k are not in the population. This is tiny and would only add more early triggers.
- **Metrics archive drift:** k ≥ 1 position values come from today's archive, masked to the pool's availability. 40 values differ slightly.
- **Paper fills** include a modelled walk (+3 to +15 bps). Real live slippage was not measured here.
- **The backtest ends 2026-11-18,** so the 4 paper fills cannot be simulated through Exit B, only priced at entry.
