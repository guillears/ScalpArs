# Top-20 backtest: Spike Fade port (idea 1) and BTC-lead catch-up (idea 2), 2026-11-09

Research only. Nothing in the bot was changed. Scripts are in the session scratchpad (`top20/`). Row data: `reports/TOP20_IDEA1_SPIKEFADE_CELLS.csv`, `reports/TOP20_IDEA2_BTCLEAD_CELLS.csv`, `reports/TOP20_IDEA1_BEST_TRADES.csv`.

## Verdicts

| Idea | Verdict | One-line reason |
|---|---|---|
| 1 · Top-20 Spike Fade | **WATCH** (fails robustness) | The only cells that pass every gate are short fades on a post-hoc "24h vol ≥ $20M" subset. All of them turn negative or flat when the exit is simulated with pessimistic intrabar ordering. The edge sits in how the 1m bars are read, not yet in the market. The live trigger, unchanged, is break-even (591 · 74 % · −0.004 %). |
| 2 · BTC-lead catch-up | **NO EDGE** | 180 cells: the average cell nets −0.12 %. Before costs the catch-up is worth about +0.02 %, and costs are 0.14 %. The 6 positive cells depend on one month (Oct-2025 crash week): top-5 events = 120–250 % of their total, OOS < 0, LOMO < 0, CI includes 0. |

## Data and span

- **Universe:** top-20 Binance USDT-M **crypto** perps by 24h quote volume on 2026-11-09 (`/fapi/v1/ticker/24hr`). TradFi perps (SOXL, XAU, NVDA …) excluded, and so is anything listed after 2026-05-08. Pairs: BTC ETH JCT SOL ZEC XRP HYPE 1000PEPE NEAR DOGE BNB ENA SUI AIOT UNI ON HOT WLD PUMP ADA.
- **Span:** 2025-01-01 → 2026-11-08 (677 days). Late listings start at their own listing date: HYPE 2025-05-30 · AIOT 2025-04-30 · PUMP 2025-04-12 · ON 2025-10-24 · JCT 2025-11-10.
- **Source:** 1m klines already on disk, merged from `reports/backtest_cache_2025/{k1m,k1m_spike}` and `reports/backtest_cache/{k1m,k1m_spike,k1m_ondemand}`. Missing pieces were fetched from fapi klines: 2026-10-12 → 11-09 for all 20, plus the 2026 gaps in JCT/AIOT/ON/HOT/UNI/PUMP. That was about 1.6 M bars, peak used weight 850/min, no 418/429. Coverage is complete except one 7.5h PUMP hole.
- **Survivorship caveat:** AIOT, JCT, ON and HOT are top-20 today but were small caps for most of the span. Spike Fade already trades them below $20M.

## Common rules (pre-declared before any result)

- **Signal:** evaluated at the close of a 1m bar. **Fill:** open of the next bar moved 8/60 of the way to its close (≈ 8 s latency). **Costs:** 0.05 % fee + 0.02 % slippage per side, 0.14 % round trip. Paper uses a 0.045 % taker fee; the larger value is used here.
- One open trade per pair per cell.
- **Gates for OPPORTUNITY:** N ≥ 30, avg > 0, OOS (last 1/3 by time) > 0, day-block bootstrap 95 % CI low > 0, avg > null 95th pct, top-2 pairs < 60 % of P&L, leave-one-month-out min > 0.
- **Null:** same pair, direction, minute-of-day and exit rule, random day. 100–300 reps.

# Idea 1: Top-20 Spike Fade

**Live mechanism ported** (`services/trading_engine.py` spike scanner, `trading_config.json`):
- **Trigger:** on the forming 5m candle, RSI12 prev in [35, 55] with a jump ≥ 25, candle ≥ +0.5 %, volume ≥ 5× the prior-20 average.
- **Router:** fade when the BTC regime is not STRONG/HEALTHY_BULL or pair ADX > 30.
- **Fade guards:** BTC RSI ≤ 50 · BTC dist-EMA13 ≤ 0 · fresh-break guard (rsi_prev < 44 & pgap > −0.4) · laggard (pair 1d −DI > 16.1 & BTC 4h EMA50/200 > 0).
- **Exit stack:** stop −1.5 net · runner trail arms at 0.40 with floor = peak − min(0.5·ATR5m, 0.35·peak) · late arm at 15 min / 0.30 · HARD_TP ladder 1.0:0.25 … 4.0:0.80 · 24h cap.
- **Not ported:** the $20M volume ceiling, which by design blocks every top-20 coin.
- **Down-spike mirror (long):** RSI prev in [45, 65], drop ≥ 25, candle ≤ −0.5 %, vol ≥ 5×, with the live SPIKE_BOUNCE guards (BTC RSI ≥ 50, no STRONG/HEALTHY_BEAR, pgap > −1.0).

**Grid:** 13 triggers (L0 = live absolute, plus V{N}k{k}: N-minute move ≥ k·σ_N with σ from a 7-day trailing 1m std, N ∈ {1, 3, 5, 15}, k ∈ {4, 6, 8}, vol ≥ 5×) × filters {F0 none, F1 live guards} × exits {EL live stack, EA ATR alt: TP = SL = 2·ATR5m, 60 min} × direction = **104 cells**.

### 1a. All cells, grouped (avg of cell avg net %)

| Exit | Filter | Dir | cells | fills | mean cell avg % | cells > 0 |
|---|---|---|---|---|---|---|
| EL live | F0 | Long (fade dump) | 13 | 62,825 | −0.058 | 2 |
| EL live | F0 | Short (fade pump) | 13 | 60,177 | −0.045 | 2 |
| EL live | F1 | Long | 13 | 14,228 | −0.112 | 0 |
| EL live | **F1** | **Short** | 13 | 7,880 | **+0.031** | **7** |
| EA ATR | any | any | 52 | 147k | −0.13 … −0.32 | 0 |

The ATR-alt exit loses everywhere. The long mirror loses everywhere once filtered. The only life is **short fade + live guards + live exit stack**, the same recipe the bot already runs.

### 1b. Live trigger unchanged (L0), live exits

| Cell | N | days | WR % | avg % | BE WR % | OOS % | 95 % CI | fires/day | median 0.1 %·vol24h |
|---|---|---|---|---|---|---|---|---|---|
| L0 F0 short | 591 | 295 | 74.1 | −0.004 | 74.3 | −0.009 | [−0.12, +0.10] | 0.87 | $118k |
| L0 F1 short | 53 | 49 | 67.9 | −0.010 | 68.3 | +0.297 | [−0.33, +0.34] | 0.08 | $3k |
| L0 F0 long | 409 | 191 | 69.7 | −0.144 | 76.8 | −0.085 | [−0.33, +0.03] | 0.60 | $182k |
| L0 F1 long | 242 | 111 | 61.2 | −0.330 | 77.7 | −0.238 | [−0.58, −0.06] | 0.36 | $264k |

### 1c. Best cells, all fills (none passes: CI includes 0, and V1k8 is below its null)

| Cell | N | days | WR | avg % | BE WR | IS / OOS % | LOMO min | 95 % CI | top-2 share | null95 | fires/d | cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V15k8 F1 EL S | 134 | 79 | 70.9 | +0.238 | 61.4 | +0.19 / +0.34 | +0.18 | [−0.02, +0.53] | 69 % | +0.031 | 0.20 | $25k |
| V15k6 F1 EL S | 255 | 140 | 72.2 | +0.108 | 67.5 | +0.10 / +0.12 | +0.06 | [−0.04, +0.26] | 32 % | +0.023 | 0.37 | $32k |
| V1k8 F1 EL S | 440 | 238 | 72.7 | +0.099 | 68.3 | +0.11 / +0.07 | +0.08 | [−0.01, +0.22] | 59 % | **+0.107** | 0.65 | $33k |
| V1k6 F1 EL S | 889 | 359 | 76.0 | +0.049 | 73.7 | +0.06 / +0.03 | +0.03 | [−0.03, +0.12] | 57 % | −0.017 | 1.30 | $37k |

The median cap is only $25–37k because many of these fires came on AIOT/JCT/ON/HOT while they were still small.

### 1d. Post-hoc read: only fills with 24h vol ≥ $20M (outside the live sleeve's ceiling)

This split was chosen after looking (102 more reads). It is a natural boundary, but the cell is not pre-registered.

| Cell | N | days | WR | avg % | BE WR | IS / OOS % | LOMO | 95 % CI | top-2 | null95 | gates | **pessimistic exit order** | pess. OOS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V15k6 F1 EL S | 136 | 78 | 77.9 | +0.226 | 68.0 | +0.23 / +0.22 | +0.16 | [+0.01, +0.45] | 36 % | +0.031 | all pass | **−0.017** | −0.08 |
| V1k8 F1 EL S | 244 | 149 | 77.5 | +0.190 | 69.0 | +0.26 / +0.05 | +0.14 | [+0.03, +0.36] | 50 % | +0.032 | all pass | **−0.150** | −0.11 |
| V1k6 F1 EL S | 496 | 250 | 80.0 | +0.119 | 74.3 | +0.16 / +0.04 | +0.09 | [+0.01, +0.23] | 39 % | +0.041 | all pass | **−0.107** | −0.12 |
| V15k8 F1 EL S | 70 | 44 | 75.7 | +0.326 | 62.6 | +0.17 / +0.62 | +0.21 | [+0.00, +0.73] | 74 % | +0.085 | fails top-2 | +0.111 | +0.45 |
| V15k4 F1 EL S | 311 | 156 | 75.6 | +0.116 | 70.3 | +0.05 / +0.25 | +0.07 | [−0.05, +0.32] | 96 % | −0.019 | fails CI | −0.039 | +0.07 |

- **Capacity of the passing cells:** median 0.1 % × 24h vol is $257k–$363k per trade. They fire 0.2 / 0.36 / 0.73 times a day.
- **Extra 0.08 % exit slip** (to reach the ≥ 0.10 % fast-setup ruler): about −0.08 on every row. V15k6 drops to +0.15 base and −0.10 pessimistic.
- **Pair mix:** the P&L leaders are still ZEC, ON, AIOT, JCT, HOT, PUMP. On the 10 largest coins only, V15k6 = 54 · +0.12 %, V1k8 = 75 · +0.12 %, V15k8 = 25 · −0.12 %.

**Why WATCH, not OPPORTUNITY.** The live fade exit is a tight trail: arm at 0.40, give back ≤ 35 % of peak. With 1m OHLC you cannot know whether the low came before the high inside a minute. The base simulation lets the trail lock the low of a bar and then exit at the floor. The pessimistic one takes the adverse extreme first and only trails off completed bars. Every passing cell flips from about +0.12…+0.23 % to about −0.02…−0.15 % under the pessimistic order, which is the same replica trap as the DL 110 stop-width rule. On top of that, the passing universe was cut post hoc, and the winners lean on ex-small-cap names. There is a real signature: idiosyncratic alt pumps while BTC is soft, faded with the live stack, high WR. It needs tick or 1s data before any number can be trusted.

# Idea 2: BTC-lead catch-up

- **Pre-declared design:** BTC M-minute impulse ≥ k·σ(24h trailing 1m std)·√M, newly true, with M ∈ {1, 3, 5} and k ∈ {3, 4, 5}.
- **Alt filter:** beta ≥ 1, from 5m returns over the prior 7 days (past only), and the alt's same-window move < x·beta·BTC move, x ∈ {0.25, 0.5}.
- **Trade:** enter the alt in BTC's direction. Exits: hold H ∈ {1, 3, 5, 15} min, or TPg (TP at the remaining beta gap, symmetric SL, 15-min cap).
- **Cells:** 3 × 3 × 2 × 5 × 2 directions = **180 cells**, 1,154 fires/day summed over cells.

| Cut | mean cell avg net % (latency) | no-latency | +0.10 exit slip |
|---|---|---|---|
| All 180 cells | −0.123 | −0.117 | −0.223 |
| Long cells | −0.087 | −0.078 | −0.187 |
| Short cells | −0.158 | −0.156 | −0.258 |
| By exit: H1 / H3 / H5 / H15 / TPg | −0.138 / −0.133 / −0.116 / −0.102 / −0.124 | −0.132 … −0.096 | — |

- **Latency** costs only about 0.006–0.017 % (8 s inside a 1m bar).
- **Gross edge before costs** is +0.01…+0.04 %. The catch-up exists but is roughly 1/5 of the round-trip cost.
- **BTC itself** after the same impulses (control): net −0.12…−0.16 %, WR 22–32 %. There is no continuation to ride.

Positive cells (6 of 180). Every one beats its null95 (≈ −0.10, because random entries just pay costs) and every one fails the time and concentration gates:

| Cell | N | BTC events | days | WR | avg % | no-lat | +0.10 slip | IS / OOS % | LOMO | 95 % CI | top-5 events share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M5k5 x0.25 H15 L | 584 | 261 | 158 | 44.9 | +0.226 | +0.243 | +0.126 | +0.38 / −0.09 | −0.05 | [−0.11, +0.81] | 121 % |
| M5k5 x0.5 H15 L | 1418 | 404 | 230 | 42.7 | +0.105 | +0.117 | +0.005 | +0.21 / −0.10 | −0.07 | [−0.12, +0.45] | 168 % |
| M5k5 x0.25 TPg L | 596 | 263 | 158 | 46.3 | +0.068 | +0.081 | −0.032 | +0.17 / −0.14 | −0.10 | [−0.16, +0.41] | 252 % |
| M3k5 x0.25 H15 L | 691 | 307 | 181 | 41.7 | +0.065 | +0.073 | −0.035 | +0.08 / +0.03 | −0.04 | [−0.09, +0.28] | 211 % |
| M5k4 x0.25 H15 S | 860 | 452 | 261 | 47.4 | +0.054 | +0.054 | −0.046 | +0.20 / −0.24 | −0.11 | [−0.23, +0.46] | 436 % |
| M5k5 x0.25 H5 L | 614 | 267 | 158 | 37.3 | +0.036 | +0.051 | −0.064 | +0.14 / −0.17 | −0.09 | [−0.13, +0.32] | 174 % |

- **Month dependence:** best cell by month: Oct-2025 contributes +160 %-points of a +132 %-point total (121 %); without it the cell is negative.
- **Window units:** market-wide signal, so the true N is BTC events (261–452), not fills.
- **Capacity:** median 0.1 %·vol24h ≈ $250k. Irrelevant without an edge.

# Multiple testing

- **Cell reads:** 104 pre-registered (idea 1) + 102 post-hoc subset reads (idea 1) + 180 (idea 2) = **386**.
- **Expected by luck:** the null gate alone (5 %) would pass about 19. The CI-low > 0 gate alone (2.5 % one-sided) about 10.
- **What passed:** 3 cells cleared every gate. They form one nested family (V1k6 ⊂ V1k8-type 1-minute spikes plus V15k6), all on the post-hoc subset, and all fail fill-order robustness. That is fully consistent with luck plus a simulation artefact.

# What's missing / next step if the operator wants to pursue idea 1

1. **Tick or 1s data** (aggTrades) for the ~450 ≥ $20M fires of V15k6 / V1k6 / V1k8. This settles the intrabar order. On-disk `ticks/` and `k1s_*` don't cover these coins or dates. About 450 pair-hours would be needed.
2. If the tick replay keeps V15k6 F1 short ≥ +0.10 % net, pre-register it as an **observe-only** line. Its existing trigger already stamps it. Keep the $20M boundary frozen. Read it on live fills.
3. **Approximations not modelled exactly:**
   - BTC RSI, regime and pair ADX are read from the last closed 5m bar, not the forming one.
   - The volume leg uses quote volume.
   - The router/regime classifier is re-implemented, not imported.
   - The "standard short stack" is modelled as the engine's own fade shadow (stop / trail / late-arm / ladder), with no other exits.
