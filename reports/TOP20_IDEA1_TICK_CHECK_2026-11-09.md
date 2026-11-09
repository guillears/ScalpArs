# Top-20 Spike Fade (idea 1): tick check, 2026-11-09

Research only. Nothing in the bot was changed. This follows `TOP20_SLEEVE_BACKTEST_2026-11-09.md` §1d. Row data: `reports/TOP20_IDEA1_TICK_ROWS.csv` (876 cell-rows, 619 unique fires). Scripts and tick cache are in the session scratchpad under `top20/ticks/`.

## Verdict: **CLOSE idea 1**

The rule was fixed before the run: PASS only if V15k6 on ticks has avg ≥ +0.10 % net, CI low > 0 and OOS > 0. On ticks V15k6 nets **−0.082 %** per trade, with CI [−0.24, +0.06] and OOS −0.09. It fails all three legs. The other two cells are worse, and their CIs sit entirely below zero. The $20M volume cut was chosen post hoc either way, and that does not matter now.

## Headline (net % per trade, 0.14 % round-trip cost)

| Cell | N | days | WR tick | **avg tick** | avg 1m optimistic | avg 1m pessimistic | parity tick = 1m-opt reason | parity tick = 1m-pess reason | total % tick |
|---|---|---|---|---|---|---|---|---|---|
| **V15k6 F1 EL S** | 136 | 78 | 76.5 | **−0.082** | +0.226 | −0.017 | 72.8 % | 70.6 % | −11.2 |
| V1k8 F1 EL S | 244 | 149 | 72.5 | **−0.211** | +0.190 | −0.150 | 66.4 % | 71.7 % | −51.5 |
| V1k6 F1 EL S | 496 | 250 | 76.8 | **−0.129** | +0.119 | −0.107 | 71.4 % | 74.4 % | −64.0 |

## Robustness on ticks

| Cell | IS / **OOS** (last third) | LOMO min | day-block 95 % CI | BE WR | worst-2 pairs' share of the net loss | months positive | **+0.10 exit slip** (avg / CI / OOS) |
|---|---|---|---|---|---|---|---|
| V15k6 | −0.078 / **−0.091** | −0.136 | [−0.244, +0.062] | 80.9 % | 102 % (ON −6.5, WLD −4.9) | 9 of 23 | −0.182 / [−0.34, −0.04] / −0.19 |
| V1k8 | −0.265 / −0.104 | −0.238 | [−0.392, −0.044] | 84.0 % | 32 % (ENA, ZEC) | 11 of 23 | −0.311 / [−0.49, −0.14] / −0.20 |
| V1k6 | −0.143 / −0.101 | −0.148 | [−0.242, −0.029] | 83.9 % | 30 % (WLD, UNI) | 10 of 23 | −0.229 / [−0.34, −0.13] / −0.20 |

- The top-2 pair share of P&L is not defined when the total is negative, so the table shows the worst-2 pairs' share of the loss.
- The +0.10 slip is applied to every STOP, TRAIL, TRAIL_LATE and LADDER exit. There were no timeouts.
- By year (V15k6): 2025 −0.18 (54 fills), 2026 −0.02 (82 fills). Neither year is positive.

## By exit reason (ticks)

| Cell | Reason | n | avg tick | avg 1m-opt (same fills) | avg 1m-pess | WR |
|---|---|---|---|---|---|---|
| V15k6 | TRAIL | 93 | +0.342 | +0.557 | +0.298 | 99 % |
| V15k6 | TRAIL_LATE | 7 | +0.174 | −0.039 | −0.041 | 100 % |
| V15k6 | LADDER | 5 | +0.762 | +1.117 | +0.881 | 100 % |
| V15k6 | STOP | 31 | −1.549 | −0.852 | −1.100 | 0 % |
| V1k8 | TRAIL / LATE / LADDER / STOP | 157 / 13 / 7 / 67 | +0.294 / +0.155 / +0.598 / −1.550 | +0.467 / +0.404 / +0.244 / −0.507 | +0.241 / +0.219 / −0.026 / −1.152 | 100 / 100 / 100 / 0 % |
| V1k6 | TRAIL / LATE / LADDER / STOP | 327 / 49 / 7 / 113 | +0.302 / +0.167 / +0.598 / −1.550 | +0.365 / +0.283 / +0.244 / −0.671 | +0.204 / +0.217 / −0.026 / −1.153 | 99 / 100 / 100 / 0 % |

**V15k6 reason crosstab (1m-opt → tick):**
- **1m STOP (30):** 21 stay STOP; 8 turn TRAIL and 1 TRAIL_LATE.
- **1m TRAIL (91):** 74 stay TRAIL; **10 become STOP**, 4 TRAIL_LATE, 3 LADDER.
- **1m LADDER (12):** 10 become TRAIL, i.e. the trail closes them before the ladder rung.

The 1m-optimistic read banked trail profits on wicks that ticks show reversing first: 10 "winners" get stopped at −1.55. Each tick trail exit also lands at the tick that crossed the floor, not on the floor itself.

## Where the 1m edge went: decomposition (avg net %)

| Cell | 1m optimistic | tick exits, 1m study's entry price | **tick exits, tick entry at +8 s (the test)** | tick entry at 0 s (no latency) |
|---|---|---|---|---|
| V15k6 | +0.226 | +0.090 (CI [−0.09, +0.25]) | **−0.082** | +0.025 (CI [−0.14, +0.17]) |
| V1k8 | +0.190 | +0.111 | **−0.211** | +0.051 |
| V1k6 | +0.119 | +0.060 | **−0.129** | +0.017 |

1. **Exit ordering costs about 0.08–0.14 %/trade.** That is the intrabar artefact the 1m study suspected.
2. **The entry convention costs a further ~0.17–0.32 %.** The 1m study filled at open + 8/60 of the bar's move. On ticks, after a σ-spike minute the price has often already turned down by +8 s.
   - Tick fill vs 1m fill: median −5 bp, 5th percentile −90 bp. For a short, lower is worse.
   - The fade's best seconds are gone before the bot can be in.
3. **Even with zero latency** no cell reaches +0.10 or a positive CI. Every variant fails the frozen bar.

## Method

- **Fires:** the three passing cells from §1d, 876 cell-fills, deduped to **619 unique (pair, signal minute) fires**. Each was priced once and mapped back to every cell it belongs to.
- **Entry:** first aggTrade at or after the signal-bar close + 8 s (same latency convention as the 1m study). Data check: the tick entry lies inside the 1m entry bar on 619 of 619 fires.
- **Exit:** a replica of the live realtime fade stack, re-read from `services/trading_engine.py`.
  - The live bot has no polling interval. The websocket `@trade` callback runs `check_realtime_stop_loss` on every trade, so the replica evaluates **every aggTrade**. aggTrades carry the same price sequence as `@trade`.
  - Per tick: `pnl = (E − px)/E·100 − 0.09` (the engine's fee view). The HARD_TP ladder `1.0:0.25 … 4.0:0.80` uses the previous peak, then the peak updates when pnl > 0.
  - Late peak (`next_fade_late_peak`): the running max from minute 15, seeded by the first reading.
  - Runner arm (`short_runner_arm`): since-entry peak ≥ 0.395, or late peak ≥ 0.295. Floor = pk − min(0.5·ATR5m, 0.35·pk); a negative floor is suppressed. Exit when pnl ≤ floor.
  - Stop at pnl ≤ −1.49. There is a 24h cap (never hit).
  - **The exit fills at the triggering tick's price**, as paper `close_position(current_price)` does.
- **Costs:** 0.05 % fee + 0.02 % slippage per side. Sensitivity adds +0.10 % on stop/trail/ladder exits.
- **1m comparators:** re-run with exit reasons from verbatim copies of `idea1.sim_live` and `robust1.sim_pess`. The optimistic re-run reproduces the stored 1m nets exactly (max |Δ| = 0).
- **Stats:** `common.summarize`, the same code as the 1m study. OOS = last third by time; LOMO = leave-one-month-out minimum; CI = day-block bootstrap with 2,000 draws.

## Data

- **Tick source:** Binance USDT-M aggTrades.
  - 408 of 467 entry pair-days were already on disk (`reports/backtest_cache*/ticks*`, read only). The earlier report's note that on-disk ticks don't cover these coins was wrong.
  - The rest came from Binance's public daily archive (`data.binance.vision`, the same source as `scripts/backtest_fetch_ticks.py`, no API weight), 56 pair-days, cached in `top20/ticks/cache/days`.
- **Why not `/fapi/v1/aggTrades`:** the endpoint now rejects historical windows with error −4166, "Search window is restricted to recent 2 days only".
  - It was used only for hours not yet archived (2026-11-08/09), 5 hours in all.
  - Peak `X-MBX-USED-WEIGHT-1M` read was 910. The fetcher sleeps out the minute at ≥ 900. No 418 or 429 responses.
- **Not modelled:**
  - Fill latency on the exit side; the +0.10 sensitivity covers it.
  - The engine's 1 s monitor backstop. It only re-checks the same floors.
  - BE-lock shadows. These are observation only and don't affect exits.

## Implications

- **Idea 1 closes.** This leaves no observe line and nothing to pre-register.
  - The signature (idiosyncratic top-20 pumps while BTC is soft, faded with the live stack) has a positive 1m read.
  - That read comes from intrabar ordering plus an entry price the bot cannot get.
- **Lesson for any future 1m fade study:** the open + 8/60·(close − open) fill rule overstates short-fade entries after spike minutes by ~0.17 %/trade on these coins. Spike-reversal studies should be priced on ticks from the start.
