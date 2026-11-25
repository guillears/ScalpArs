# EXHAUSTION_SHORT × market breadth (bearish %) — theory · backtest · batch (2026-11-25, research only, DECISION_LOG 643)

**Question (operator):** the two BIGTIME losses + YB came with > 65 % of coins bearish — "why would > 65 % be bad for a short? it
shouldn't" · "breadth was increasing for bearish or decreasing?" · "can we run the backtest?".
**Prior-study check:** EXH_BADDAY (11-18) tested SQUEEZER breadth (alt share up 1 h / 4 h), BTC returns / trend and the bot's BTC labels
on 3,998 trades (TP 10 era) — nothing passed. NEW here: the bot's OWN breadth statistic, its direction, the three mechanisms, today's exit.
Code: scripts/EXH_BREADTH_2026-11-25.py (pre-registration in its docstring, written before any outcome) · data: _trades.csv · .csv · _halves.csv.

## 1 · What "bearish %" is
Share of the top-50 USDT perps (trailing 24 h quote volume) whose 5m EMA20 fell > 0.02 % over 3 bars = **the market falling over the
last ~15 min**, not a bear market. Rebuilt from the local 5m store; parity on 400 real fills (MASTER_POOL_stacked `entry_bear_pct`):
correlation 0.955, median |diff| 3.9 pts, same > 65 side 93 %.

## 2 · Theory (before the numbers)
Why bearish breadth should HELP a short: no market bid under the pump; risk-off deleveraging. Three ways it could HURT:
| | mechanism | prediction | backtest |
|---|---|---|---|
| H1 | relative strength — the coin has its own buyer | losers lose while BTC keeps falling | ✗ zone losers saw BTC UP in the next 60 min 56 % vs 50 % for the rest |
| H2 | crowded shorts squeezed | losers carry more negative funding | ✗ zone losers −0.12 vs rest losers −0.10 % funding (like for like; winners pay more only because they hold longer) |
| H3 | oversold market bounces, lifts every alt | BTC up after the losing entries | ~ weak tilt only (56 vs 50 %) |

## 3 · Backtest — 2,793 EXH fills 2024-06 → 2026-11-16 (TP_GRID FRENZY rows, DL 582 applied), TODAY's exit (TP 7 / SL ≤ 6 / 24 h), live costs, funding
Breakeven WR 44.1 %. avg %/trade (N).
| bearish % | ALL | DESIGN < 2025-07 | HOLDOUT ≥ 2025-07 | WR ALL |
|---|---|---|---|---|
| 0–20 | +0.70 (471) | +0.22 | +0.99 | 49 % |
| 20–40 | +0.68 (845) | +1.10 | +0.56 | 50 % |
| 40–65 | +0.74 (690) | +0.81 | +0.71 | 50 % |
| 65–80 | **+0.97 (450)** | +0.58 | +1.09 | 53 % |
| > 80 | **−0.02 (337)** | −0.33 | +0.19 | 45 % |
| **> 65 (the zone)** | **+0.55 (787)** | +0.07 | **+0.75** | 49 % |
| rest | +0.70 (2,006) | +0.71 | +0.70 | 50 % |
Zone vs the locked Pattern-C bar: WR above breakeven, avg > 0, 95 % CI +0.14 … +0.98 (10-day moving day blocks, empty days = 0 trades — the TP_GRID boot_ci method) → **fails** (not a filter). Zone by half-year:
negative only in 2024-H2 (−0.05); positive in 5 of 6. > 80: CI −0.60 … +0.64, sign flips DESIGN → HOLDOUT. Only Z > 65 was pre-registered as the zone; the 65–80 / > 80 split is a bucket read (1 of 10 cohorts) — post-hoc for any rule. 0 fills dropped (every bar had ≥ 50 pairs).
Direction (bearish share rising over the prior 60 min) in the zone: rising +0.48 (637) vs falling +0.83 (150) — DESIGN −0.17 vs +0.91,
HOLDOUT +0.75 vs +0.78 → no consistent split.
Caveats (universe ≠ the engine's): top 50 of the pairs in the local 5m store (only pairs earlier studies fetched — selection / survivor bias); the engine also drops new listings, Alpha-subtype and non-coin underlyings, drops the blacklist after its top-50 cut and divides by the pairs it actually collected (can be < 50); the engine reads EMA20 on the FORMING 5m candle, this rebuild on the last CLOSED bar (≈ one partial bar earlier) — part of the 0.955 parity gap. Pre-2024-06 excluded (no majors in the store); batch fills not in the backtest (ends 11-16).

## 4 · Current batch (14 EXH fills 11-18 → 11-25, today's exit: a fill whose best move reached +7.05 % re-booked at TP 7, full booked funding kept)
| bearish % (stamp) | fills | won | avg % | days |
|---|---|---|---|---|
| > 80 | 3 (YB, ENSO, BIGTIME #38) | 1 | −3.04 | 3 |
| 65–80 | 2 (BIGTIME #39, PROM) | 1 | +0.40 | 2 |
| ≤ 65 | 9 | 8 | +4.21 | 6 |
Buckets use the bot's STAMP; the direction uses the REBUILT series (60 min before → at entry): YB 54 → 84 · ENSO 50 → 90 · BIGTIME 60 → 88 / 14 → 84 — rising into all four; PROM (stamp 65.2) rebuilds at 48 (20 → 48 rising), i.e. outside the zone on the rebuild.
Re-booking caveat (optimistic for today's exit): a fill is re-booked at TP 7 when its best move reached +7.05 % — it is not checked whether today's stop would have hit first on the path, and the full booked funding is kept even where TP 7 would have closed earlier.

Before → after blocking > 80 % in the batch (one size: 0.61 × account = the median EXH size on the $10k start, every calendar day 11-18 08:30 → 11-25 12:34): today's exit 14 · 71 % · +2.11 % · DCR +2.20 %/day → 11 · 82 % · +3.51 % · +2.92 %/day (Δ +0.72; LOO Δ +0.22 … +0.86; IN-SAMPLE — the trades that raised the idea, 30–50 % haircut ≈ +0.36 … +0.50 on 3 trades / 3 days); booked +0.93 → +2.56 %/day. Backtest before → after: ALL 2,793 · +0.66 → 2,456 · +0.75 %/trade; HOLDOUT 2,019 · +0.72 → 1,816 · +0.78 (the blocked trades ≈ 0 → total P&L and the daily compound barely move).
Exhaustive 2D over 101 entry stamps on the 4 losers: no cell holds all 4 with 0 winners (random labels give a median of 2) → no separator.

## 5 · Verdict
No filter, no switch, no observe line (operator: option 1 only). A bearish market is not bad for EXH — 65–80 % is its best bucket in ALL and HOLDOUT (DESIGN +0.58, below the rest +0.71); only
the extreme (> 80 %) is about breakeven and flips between eras. The batch's > 80 % read (3 fills · −3.04) points the same weak way but is 3 days.
