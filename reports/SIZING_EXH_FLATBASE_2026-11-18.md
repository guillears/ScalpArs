# SIZING: EXHAUSTION_SHORT and FLATBASE_SHORT from their edge, alone and together (2026-11-18)

Research only. No config, engine, UI or test file changed. Nothing committed.
Script: `scripts/sizing_exh_flatbase.py`. Tables: `reports/SIZING_EXH_FLATBASE_2026-11-18/` (tables.txt, kelly_*.csv, halves.csv, depth_sens.csv, candidates.csv) + `reports/SIZING_EXH_FLATBASE_2026-11-18_sizes.csv` and `_combined.csv`.

## Question
Operator: "Recompute EXHAUSTION's size from its edge — now, not later." Do the same for FLATBASE_SHORT, and for the two together. They share the book and both short Frenzy-type coins, so their losses can land on the same day.

## Plain answer
1. **Today's paper size is too big for both sleeves together.**
   - Each sleeve trades **0.61 × equity** in notional (0.5 × 5×; formula below).
   - Run together at that size over 2023-07 → 2026-11, the worst drawdown is **−64 %** and the worst single day is **−45 %** (2026-01-24, 15 stops in one day).
   - A block bootstrap gives a **−50 % drawdown in 99.5 % of paths** over the 3.4 years.
2. **EXHAUSTION at 0.61 is at full Kelly, not a fraction of it.**
   - Its growth-optimal size after a 50 % edge haircut is 0.60 × equity. Today's size is 1.0 × that.
   - With 0.10 % slippage per side it is 1.3 × full Kelly. With 0.20 % it is 2 × full Kelly.
   - Only the less strict version (the daily mean moved to its 95 % lower bound) puts today's size below full Kelly, at 0.68 ×.
3. **FLATBASE at 0.61 is about 0.55 – 0.72 × full Kelly.** That is still above ½ Kelly, and it costs 6.2 % of equity per stop.
4. **The two sleeves barely move together day to day** (daily correlation +0.10). So sizing them together changes little versus alone. EXHAUSTION's own clustering is the problem: up to **14 EXH stops in one day** (2024-11-16), and 10 on a 1-in-100 active day.
5. **Recommended size (paper now = the intended first live size):**
   - **EXHAUSTION: `exhaustion_invest_mult` 0.15, `exhaustion_lev_mult` 0.25 (5×) → 0.18 × equity.**
   - **FLATBASE: `flatbase_invest_mult` 0.25, `flatbase_lev_mult` 0.25 (5×) → 0.30 × equity.**
   - Together over the full history: **+0.53 %/day**, max drawdown **−25 %**, worst day **−16 %**, P(−50 % DD) **0.9 %**.
   - The same history at today's size: +1.27 %/day, −64 % max DD, 99.5 % chance of a −50 % drawdown.
6. **Book depth caps how far this can grow.** These coins hold about **$6 – 8k within 0.15 %** of the best price, whatever their volume.
   - Above ≈ $14k planned notional, the depth guard **skips** the trade instead of shrinking it.
   - At today's 0.61 that starts at ≈ $25k equity. At the recommended sizes it starts at ≈ $45k (FLATBASE) and ≈ $75k (EXH).

## How the engine turns the mults into size (services/trading_engine.py `calculate_position_size`)
- `investment.mode` = equal_split with `max_open_positions` 4, and the reserve schedule is linear between tiers.
  - Margin = (equity − reserve) ÷ 4 × invest_mult. The reserve is ≈ 2.5 % at ~$10k equity (schedule 10,000 → 9,750; fee reserve 2.5 %).
  - Leverage = round(20 × lev_mult). The VERY_STRONG level is 20×, and `compose_leverage` rounds to a whole number.
- **Notional ÷ equity ≈ 0.24375 × invest_mult × leverage.**
  - Today: 0.24375 × 0.5 × 5 = **0.609**.
  - ZKP check: equity $10.0k → margin $1,218.75 → $6,094 planned. The depth guard and capped entry then cut it to $4,893 (fillable read $7,086).
- **Note:** DL 577 described invest 0.5 as "≈ 0.45 – 0.5 × notional". The engine actually gives 0.61 × equity, because of the ÷ 4 and the 20× base. That is 1.22 × invest_mult at 5×.
- Floors: invest_mult ≥ 0.1, lev_mult ≥ 0.05 (1×). At 5× leverage each 0.05 step of invest_mult is 0.061 × equity.

| invest × lev | notional ÷ equity | EXH per stop (% equity) | FB per stop (% equity) |
|---|---|---|---|
| 0.5 × 5× (today) | 0.609 | 2.85 | 6.15 |
| 0.35 × 5× | 0.427 | 2.0 | 4.31 |
| 0.25 × 5× | 0.305 | 1.43 | 3.08 |
| 0.15 × 5× | 0.183 | 0.86 | 1.85 |
| 0.125 × 5× | 0.152 | 0.71 | 1.53 |

## Method
- **Trades (reused, nothing rebuilt):**
  - **EXH**, 2023-07-01 → 2026-11-17 (the exit-grid "seen" pool, 3.23 trades a day), from `EXH_EXIT_GRID_data/{sim,eval}_grid.pkl`. One position per pair. Two exits:
    - **L2** = TP 10 % fixed (maker) / SL min(1.5 × ATR15, 6 %) / 24 h, the exit going to paper today: 3,998 trades.
    - **Shipped** = TP min(3 × ATR, 5 %) / SL min(1.5 × ATR, 5 %): 4,015 trades.
    - Holdout E (2023-H1, 237 trades) is shown separately.
  - **FLATBASE**: today's rules with route + NEAR_BASE 2.09 + 4 slots, stop 10 % / TP 5 % / 72 h, from `HIGH_ATR_2026-11-18_data/fb_eval.pkl` (the 11-18 rebuild).
    - 779 trades, 0.63 a day, same window.
    - Parity: HIGH_ATR quotes 778 · 73.5 % · +0.91 at 0.10 % flat slip. Here it is 779 · 73.8 % · +1.0 at the model slip.
- **Costs:**
  - Taker 0.05 % in. Exit at 0.05 % taker, or 0.02 % maker on the TP.
  - Slippage = max(0, −0.042 + 0.276 · √(notional ÷ q5)) on entry and on every taker exit, at each trade's own notional. q5 = median 5m quote volume over the last hour.
  - Real funding to each exit. EXH pays 0.18 % per trade on average, worst −5.0 %. FB pays 0.14 % on average, worst −5.6 %.
  - Stress runs add a flat 0.10 % and 0.20 % per side.
- **Depth guard:** fillable within 0.15 % = 3,505 × q5^0.054. This is a log fit on the 10 live and paper depth-guard stamps on alt coins (11-15 → 11-18, momentum majors excluded). The book is ≈ $5 – 11k whatever the volume. Fits → full size; ≥ ½ fits → shrink; else skip. All sizing is referenced to $10k equity (one size, DL 503).
- **Kelly, three ways:**
  - (a) Per trade: bet form f = p − q/b at the one-sided 95 % Wilson floor of the WR (n = trades, and n = distinct days). Notional X = f ÷ avg loss.
  - (b) Daily: each calendar day's booked P&L is one bet at size X. We choose the X that maximises mean log(1 + r_day), after either moving the daily mean to its one-sided 95 % lower bound (bootstrap over 1-day and 10-day blocks), or cutting the mean by 50 % (in-sample haircut).
  - (c) The same on a 2-D grid for EXH + FB combined.
  - Kelly is computed **without** the depth cap. Kelly measures the edge; the cap is a separate capacity limit.
- **Paths:** every calendar day 2023-07-01 → 2026-11-17 (1,236 days), compounding at one size, P&L booked on the exit day.
  - Outputs: %/day (DCR), CAGR, max DD, worst day, worst 7 days, longest time under water.
  - P(DD ≤ −30 % / −50 %) comes from 2,000 bootstrap paths of 10-day blocks over the same horizon.

## Per-sleeve Kelly (notional × equity per trade)

**(a) Per trade, at today's size:**

| sleeve / exit | N | WR | BE WR | avg win / loss | avg net % | WR floor (trades / days) | Kelly X (floor trades) | Kelly X (floor days) |
|---|---|---|---|---|---|---|---|---|
| EXH L2 (TP 10 / SL 1.5 ATR ≤ 6) | 3,998 | 40.2 % | 35.3 % | +8.52 / −4.64 | **+0.65** | 38.9 / 37.6 % | 1.22 | 0.80 |
| EXH shipped | 4,015 | 52.5 % | 47.4 % | +4.78 / −4.31 | +0.46 | 51.2 / 49.9 % | 1.66 | 1.09 |
| FLATBASE | 779 | 73.8 % | 66.7 % | +4.72 / −9.45 | **+1.00** | 71.1 / 70.4 % | 1.41 | 1.17 |
| EXH L2, holdout E 2023-H1 | 237 | 38.4 % | 34.0 % | +6.88 / −3.54 | +0.46 | 33.4 / 30.6 % | 0 (f < 0) | 0 |
| EXH shipped, holdout E | 237 | 44.3 % | 43.0 % | +4.68 / −3.53 | +0.11 | 39.1 / 36.2 % | 0 | 0 |

**(b) Daily growth-optimal X, alone (no depth cap):**

| sleeve | observed | mean at 95 % LB (1-day / 10-day blocks) | mean × 0.5 (haircut) | +0.10 % slip: LB / haircut | +0.20 % slip: LB / haircut |
|---|---|---|---|---|---|
| EXH L2 | 1.20 | 0.90 / 0.86 | **0.62** | 0.58 / 0.46 | 0.24 / 0.30 |
| EXH shipped | 1.85 | 1.40 / 1.40 | 1.00 | — | — |
| FLATBASE | 1.70 | 1.10 / 1.05 | **0.96** | 0.86 / 0.84 | 0.64 / 0.72 |

**(c) Joint (EXH L2 + FB on the same days):**

| | EXH X | FB X |
|---|---|---|
| observed | 1.1 | 1.3 |
| mean at 95 % LB (1-day) | 0.90 | 1.10 |
| mean at 95 % LB (10-day) | 0.85 | 1.20 |
| **mean × 0.5** | **0.60** | **0.85** |
| ½ of that | 0.30 | 0.43 |
| ¼ of that | 0.15 | 0.21 |

- The daily correlation is +0.10, and both sleeves book on 445 of 1,236 days. So the joint optimum is almost the two separate ones.
- **Today's 0.61 vs full Kelly:**
  - EXH is 1.0 × the haircut Kelly and 0.68 × the LB Kelly. It is above full Kelly once slippage is 0.10 % or more.
  - FB is 0.72 × the haircut Kelly and 0.55 × the LB Kelly.
- EXH's holdout E Kelly floor is negative. E is a slow half (1.3 trades a day, 94 trade days), so it does not support a size above a small fraction.

## Size → result, each sleeve alone (2023-07 → 2026-11, depth cap on at $10k)

| sleeve | X | per stop (% eq) | %/day | CAGR | max DD | worst day | worst 7 d | longest under water | P(DD ≤ −30 %) | P(DD ≤ −50 %) |
|---|---|---|---|---|---|---|---|---|---|---|
| EXH L2 | 0.10 | 0.47 | +0.20 | +108 % | −12.8 % | −5.4 % | −10.4 % | 93 d | 0 % | 0 % |
| EXH L2 | 0.15 | 0.70 | +0.30 | +193 % | −18.7 % | −8.2 % | −15.4 % | 93 d | 10 % | 0 % |
| **EXH L2** | **0.183** | **0.86** | **+0.36** | **+265 %** | **−22.6 %** | **−10.0 %** | **−18.6 %** | 93 d | 28 % | 0.05 % |
| EXH L2 | 0.20 | 0.94 | +0.39 | +307 % | −24.5 % | −10.9 % | −20.2 % | 93 d | 41 % | 0.7 % |
| EXH L2 | 0.30 | 1.40 | +0.55 | +650 % | −35.2 % | −16.3 % | −29.4 % | 94 d | 94 % | 16 % |
| EXH L2 | 0.40 | 1.87 | +0.71 | +1,201 % | −44.8 % | −21.8 % | −38.0 % | 94 d | 100 % | 53 % |
| **EXH L2 today** | **0.609** | **2.85** | **+0.97** | — | **−62.6 %** | **−33.2 %** | **−54.1 %** | 108 d | 100 % | **98 %** |
| EXH shipped | 0.183 | 0.79 | +0.26 | +159 % | −27.2 % | −6.3 % | −16.0 % | 133 d | 2 % | 0 % |
| EXH shipped | 0.30 | 1.30 | +0.42 | +355 % | −41.6 % | −10.3 % | −25.1 % | 134 d | 44 % | 0.7 % |
| EXH shipped today | 0.609 | 2.63 | +0.78 | — | −70.9 % | −21.0 % | −45.7 % | 154 d | 100 % | 60 % |
| FB | 0.15 | 1.51 | +0.09 | +41 % | −12.8 % | −4.5 % | −6.7 % | 129 d | 0 % | 0 % |
| FB | 0.20 | 2.02 | +0.12 | +57 % | −16.9 % | −6.0 % | −8.8 % | 129 d | 2 % | 0 % |
| **FB** | **0.305** | **3.08** | **+0.18** | **+95 %** | **−25.1 %** | **−9.2 %** | **−13.3 %** | 129 d | 21 % | 0.25 % |
| FB | 0.40 | 4.04 | +0.23 | +133 % | −32.1 % | −12.1 % | −17.2 % | 129 d | 57 % | 2.9 % |
| FB | 0.50 | 5.06 | +0.28 | +178 % | −39.9 % | −15.1 % | −21.2 % | 234 d | 85 % | 12 % |
| **FB today** | **0.609** | **6.15** | **+0.33** | +232 % | **−48.2 %** | −18.4 % | −25.4 % | 238 d | 97 % | **32 %** |

- The exit-grid report quoted EXH L2 at X 0.5 as +0.84 %/day with −53 % max DD. This run reproduces it: +0.84, −53.5 %.
- The shipped EXH exit has smaller days but deeper slow drawdowns. At 0.183 it earns +0.26 %/day vs L2's +0.36 %/day.

## Combined portfolio (EXH L2 + FB on one account, same days)

| EXH X | FB X | config (invest × 5×) | %/day | max DD | worst day | worst 7 d | P(DD ≤ −30 %) | P(DD ≤ −50 %) |
|---|---|---|---|---|---|---|---|---|
| **0.609** | **0.609** | **0.5 / 0.5 (today)** | +1.27 | **−63.5 %** | **−45.5 %** | −54.0 % | 100 % | **99.5 %** |
| 0.305 | 0.427 | 0.25 / 0.35 | +0.80 | −37.8 % | −25.3 % | −32.4 % | 99 % | 25 % |
| 0.244 | 0.305 | 0.2 / 0.25 | +0.64 | −30.3 % | −19.5 % | −26.1 % | 83 % | 4.4 % |
| 0.183 | 0.427 | 0.15 / 0.35 | +0.60 | −30.1 % | −18.7 % | −25.4 % | 80 % | 4.4 % |
| **0.183** | **0.305** | **0.15 / 0.25 (recommended)** | **+0.53** | **−24.8 %** | **−16.2 %** | **−21.2 %** | 49 % | **0.9 %** |
| 0.152 | 0.305 | 0.125 / 0.25 | +0.48 | −23.3 % | −14.5 % | −19.5 % | 32 % | 0.2 % |

- With the shipped EXH exit, 0.183 / 0.305 gives +0.44 %/day, −24.7 % max DD and a −10.6 % worst day.
- Slippage stress on the recommended pair:
  - +0.10 % per side: +0.41 %/day, max DD −27 %, P(−50 %) 5 %.
  - +0.20 % per side: +0.28 %/day, max DD −35 %, P(−50 %) 21 %.
- The same stresses at today's size give max DD −69 % and −87 %.

## Clustering (stops landing together)

| | EXH L2 | EXH shipped | FB | EXH L2 + FB |
|---|---|---|---|---|
| most stops in one hour | 4 (2024-11-23 17:00, 2024-11-24 08:00) | 4 | 2 | 4 |
| most stops in one day | **14** (2024-11-16) | 11 | 3 (2024-06-03) | **15** (2026-01-24) |
| p99 stops per day | 10 | 8 | 2 | 10 (active days) |
| most open at once | 11 (p99 7) | — | 4 (p99 4) | 12 (p99 8) |

**Per-stop cost × the realistic same-day count (% of equity):**

| size | EXH per stop | × 4 (worst hour) | × 10 (p99 day) | × 14 (worst day) | FB per stop | × 3 (worst day) |
|---|---|---|---|---|---|---|
| today 0.609 | 2.85 | 11.4 | 28.5 | 39.9 | 6.15 | 18.5 |
| 0.30 | 1.40 | 5.6 | 14.0 | 19.6 | 3.03 | 9.1 |
| **0.183 (EXH) / 0.305 (FB)** | **0.86** | 3.4 | 8.6 | 12.0 | **3.08** | 9.2 |

- **Worst real joint days:**
  - At today's size: 2026-01-24 −45.5 %, 2025-11-17 −30.8 %, 2026-03-31 −30.7 %, 2024-11-16 −29.7 %.
  - At the recommended size: −16.2 %, −10.5 %, −10.4 %, −10.0 %.
- These are alt-season days. The pumps the sleeve shorts kept pumping, and many coins stopped out on the same day. One market day is one observation, so the daily-bet Kelly is the right unit, not the per-trade Kelly.
- The sim does not apply `max_open_positions_hard` 10 or the FRENZY sleeves sharing the book. The joint max open is 12 (p99 8), so the cap would bind only on the worst days.

## Per half-year at the recommended size (EXH L2 0.183 + FB 0.305): %/day / max DD

| half | EXH | FB | together |
|---|---|---|---|
| 23-H2 | +0.22 / −10 % | +0.09 / −9 % | +0.31 / −12 % |
| 24-H1 | +0.28 / −16 % | +0.10 / −13 % | +0.38 / −20 % |
| 24-H2 | +0.31 / −11 % | +0.02 / −13 % | +0.33 / −18 % |
| 25-H1 | +0.06 / −17 % | +0.19 / −11 % | +0.25 / −13 % |
| 25-H2 | +0.62 / −19 % | −0.02 / −24 % | +0.61 / −25 % |
| 26-H1 | +0.41 / −23 % | +0.30 / −19 % | +0.70 / −25 % |
| 26-H2 | +0.65 / −22 % | +0.72 / −13 % | +1.37 / −21 % |
| EXH holdout E 23-H1 (L2) | ≈ +0.10 (0.085 at 0.15, 0.16 at 0.30) / ≈ −12 % | — | — |

- The combined sleeve is positive in all 7 halves.
- Each sleeve has one weak half: EXH 25-H1 (+0.06) and FB 25-H2 (−0.02). They do not overlap, which is why running them together helps.

## Book depth (capacity)

| | EXH coins | FB coins |
|---|---|---|
| fillable within 0.15 % (p10 / median / p75) | $7.1k / $7.5k / $7.8k | $6.1k / $6.7k / $7.1k |
| full size at 0.609 and $10k equity | 99.9 % | 90 % |
| skipped at today's 0.609, $25k equity | 57 % | 96 % |
| skipped at today's 0.609, $50k equity | 100 % | 100 % |

- At $10k equity the guard barely binds at any size up to 0.61, and the modelled slippage is ≈ 0 – 0.03 %.
- Real fills show more slippage than the model (live median ≈ 0.02 %, thin-coin outliers 0.3 – 0.6 %). That is why the 0.10 / 0.20 % stresses matter.
- **The binding limit is the ~$7k book, not the edge.** Planned notional above ≈ $14k is skipped, not shrunk.
- Planned notional should therefore stay ≤ ≈ $7k per trade to fill in full. That is X × equity ≤ 7k: at X 0.183 up to ≈ $38k equity, at X 0.305 up to ≈ $23k.
- The depth fit rests on 10 stamps from one 4-day window. Treat it as an estimate.

## Master / paper check
- **FLATBASE master** (MASTER_POOL_stacked stack_keep): 28 fills on 18 days, 93 % WR, +3.87 %/trade, worst −10.1 %.
  - With NEAR_BASE (DL 577): 20 fills, 95 % WR, +4.20 %/trade.
  - The master agrees on the sign and is far stronger than the backtest (+1.0). It is too few days to size from, so the backtest (more cautious) sets the size.
- **EXHAUSTION paper**: one fill (ZKP 11-18, +4.94 %). Nothing to size from yet.

## Recommendation

**(i) Paper now** (paper = live, so paper sizes are the first live sizes):

| sleeve | config | notional ÷ equity | per stop | why |
|---|---|---|---|---|
| EXHAUSTION_SHORT | `exhaustion_invest_mult` **0.5 → 0.15**, `exhaustion_lev_mult` 0.25 (5×) | 0.183 | 0.86 % | ≈ 0.3 × the haircut Kelly (0.60) and 0.2 × the LB Kelly (0.90). The exit was picked after the fact (the grid's verdict was OBSERVE, p_fw 0.37). The holdout's Kelly floor is negative. Up to 14 stops a day. DL 578 already named ≈ 0.19 as the live start. |
| FLATBASE_SHORT | `flatbase_invest_mult` **0.5 → 0.25**, `flatbase_lev_mult` 0.25 (5×) | 0.305 | 3.1 % | ≈ 0.36 × the haircut Kelly (0.85) and 0.28 × the LB Kelly (1.1). A 10 % stop makes each loss large (stops come at most 3 a day). At 0.61 the backtest max DD is −48 %. |

- Together: +0.53 %/day over 3.4 years, max DD −25 %, worst day −16 %, P(−50 %) ≈ 1 %.
- Versus today: about 40 % of the return for a drawdown that is far less likely to wipe out the account (−64 % → −25 %).

**(ii) Future live steps** (each one is a new operator decision; gates are alert-only):
- **EXH:** no live until the frozen paper revert gate passes (first 20 fills, re-priced average > 0, model slip ≤ 0.25 %).
  - Go live at 0.15 × 5× (0.183).
  - Step to 0.25 × 5× (0.305 = ½ haircut Kelly) only after ≥ 30 forward fills on ≥ 10 distinct days with a day-clustered CI above 0 and live slippage ≤ 0.10 % per side. At 0.20 % slip, ½ Kelly is only 0.12 – 0.15.
- **FB:** 0.25 × 5× (0.305). Step to 0.35 × 5× (0.427 ≈ ½ haircut Kelly) after the DL 577 gate reads clean on 20 live fills **and** live slippage ≤ 0.10 % per side.
  - That step costs about −30 % max DD in the combined history (P(−50 %) 4 %) for +0.07 %/day.
- **Never above** EXH 0.30 / FB 0.43 at this edge. Today's 0.61 is at or above full Kelly for EXH under live-like slippage.
- **Capacity:** once equity × X exceeds ≈ $7k, raising the mults buys nothing. Above ≈ $14k planned, the depth guard skips the trade. Growth beyond that needs more trades, not bigger ones.
- **Recompute monthly** from full history + live fills (the DL 577 rule), with this script.

## Assumptions and limits
- **In-sample choice:** EXH L2's +0.65 %/trade was chosen after looking at 378 exits. That is why the haircut Kelly (mean × 0.5) is used as the primary reference, not the observed one. The clean holdout E earns +0.46 %/trade with a negative Kelly floor.
- **Backtest, not fills:** EXH has 1 paper fill. FB has 20 – 28 master fills. Neither is enough to size from directly.
- **Slippage:** the √ model says ≈ 0 at these sizes, which flatters the results. The flat 0.10 / 0.20 % stresses are the realistic bracket. Funding is real (8 h settlements), and hourly squeeze funding is captured only where it happened in history.
- **Depth model:** 10 stamps (11-15 → 11-18). At today's $10k equity it does not change the sizing answer; it matters only for capacity.
- **One size:** returns are referenced to $10k equity throughout (DL 503). The equal_split ÷ 4 base and the reserve schedule move the real ratio by < 1 % between $10k and $25k.
- **Not modelled:** the 10-position hard cap, the other paper sleeves (FRENZY_LONG / WILLY / HT) taking slots, and the paper vs study one-position-per-pair ordering difference (DL 578).
- **Paths and probabilities:** P&L is booked on the exit day. DD probabilities come from 10-day block bootstraps over 3.4 years, so they describe a 3.4-year run, not one year.
- The Flatbase trade list is the HIGH_ATR 11-18 rebuild (779 vs the quoted 778; one slot tie).
