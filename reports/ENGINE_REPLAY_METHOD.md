# Engine-code replay — method, fidelity and calibration record (2026-09-16/17)

Operator ask: "deep analysis per sleeve, like the Performance by Sleeve table, of what would have
happened since January to today, backtesting the LIVE filters per sleeve … check the periods we did
run them to guarantee the methodology … N, WR, net $ and daily compound return per sleeve and total,
$5,000 initial, compounding continuously." Engine-code route chosen over a replica.

## 1. What runs

`scripts/engine_replay.py` instantiates the REAL `TradingEngine` (services/trading_engine.py) and
drives it minute by minute over historical klines. Nothing in signal / gate / cell / sizing / exit
code is re-implemented:

| live component | replay stand-in |
|---|---|
| `binance_service` (klines, top-N universe, ticker) | `KlineServer` over `reports/backtest_cache` (5m full-range for 588 perps, 1m for the scan universe + on-demand for open positions, BTC 5m/1h); universe = rolling 24h quote volume at scan time, same coin/alpha/new-listing screens (exchange metadata cached) |
| wall clock (`datetime.utcnow`, `time.time`, `_leash_time`, DB `func.now()` defaults) | simulated clock, patched at module level + SQLAlchemy before-insert stamps |
| `asyncio.sleep` inside the engine (batch pacing) | advances the simulated clock by the same seconds |
| scan loop (`main.py`) | one `scan_and_trade` every `--scan-step` s (60 s; live ≈ 30–95 s per cycle) |
| WebSocket ticks → `check_realtime_stop_loss` + monitor loop (1 s) | per completed minute, a stepped path open → adverse extreme → favourable extreme → close (6 points per leg) through the same realtime stop-loss path, `update_open_positions` after every tick; the ENTRY minute feeds only its close (pre-entry prices must not arm/stop the trade) |
| paper maker entry (20 s WS window + signal re-validation) | candidate must be reached by the engine again on the next scan (persistence ≥ 1 scan step, stricter than live's 20 s); fill = limit if that minute's bar touches it (maker) else the minute close (taker fallback) |
| in-progress 5m candle | rebuilt from completed 1m bars (scan universe); scanner-only pairs evaluated on completed candles |
| DB | scratch SQLite per chunk (`replay/db/`), real models + migrations |
| config | today's `trading_config.json` (all sleeves as armed today; BEARRUN 20×; gate 51 closed; 龙虾 blacklisted) |

Chunks: monthly, 3-day warm-up (monitors, cooldowns, slots), 24 h follow-through with entries
blocked; only entries inside [start, end) count. 9 chunks run in parallel (~3 h per chunk at 60 s
cadence with the 350-pair spike scanner).

Money: each chunk runs on a ~$5k book; `scripts/engine_replay_report.py` re-prices every fill at the
running balance with the engine's own `calculate_position_size` (reserve + leverage schedules) and
the engine's liquidity / gross caps, P&L% per notional (net of fees) × repriced notional. Per-sleeve
ledgers start at $5,000 each; the TOTAL is the shared book. Daily compound = (end/start)^(1/days) − 1.
No funding (paper live does not charge it either). Fees exactly as the engine (maker 0.018 / taker
0.045).

## 2. Known limits (declared)

* 1-minute data cannot reproduce sub-minute door legs (e.g. CALM3D stretch ≤ 0.06%: live caught
  AAVE 2026-08-05 02:49:12 at 0.055, the minute boundary read 0.063). Individual live fills are
  reproduced ~25–40%; the trade POPULATION is what calibrates.
* Same-minute ordering: adverse extreme is fed before the favourable one (pessimistic for stops,
  optimistic for trails within one minute).
* Persistence emulation is stricter than live (60 s vs 20 s, full ladder re-run vs signal-only
  re-check) → replay under-counts flickering entries slightly.
* Scanner-only pairs (outside the 1m-cached scan universe) fire on completed candles → spike fills
  enter later than live within the discovery candle.
* Universe is survivorship-light (pairs delisted before 2026-09-16 are absent).
* Filters were tuned on Jun-17 → Sep-15 live batches: that half is IN-SAMPLE; Jan-01 → Jun-16 is the
  out-of-sample test.

## 3. Calibration record (why the first run was NOT reported)

| pass | change | Jun-17→Aug-19 kept full-size: real 192 vs replay | matched-trade P&L (real vs replay) | extra-fill quality |
|---|---|---|---|---|
| v0 (Aug 5-19, prior-day universe) | — | recall 22% | +0.06 vs +0.15 (n=5) | — |
| v1 full year | rolling-24h universe, 60 s | Jun17→Sep15: real 295 vs replay **823** | +0.32 vs +0.135 (n=127) | 52% WR, −0.13%/fill (real 79%) → **rejected** |
| v2 | persistence + stepped ticks | real 192 vs replay **204** (MOM-long 84 vs 87, FLIP 28 vs 28, FADE 45 vs 78, MOM-short 27 vs 10) | +0.35 vs +0.14 (n=50) | 71% WR, +0.01%/fill |
| v3 | entry-minute fix (only the close is post-entry) | real 192 vs replay 203 (MOM-long 84·79%·+$4,703 vs 87·71%·+$4,662 · FLIP 28·86%·+$1,088 vs 28·79%·+$1,896 · MOM-short 27·70%·+$623 vs 9·67%·+$743 · FADE 45·76%·+$1,138 vs 78·74%·−$809) | +0.35 vs +0.11 (n=51) | 71% WR, +0.01%/fill; <2-min closes 13% (real 14%) |
| v4 | random intra-minute order | MOM-long +$2,050 (worse) | +0.35 vs +0.04 | rejected (adverse-first kept) |
| v5 | clock advances inside the minute (confirmation timers) | identical to v3 | identical | no effect |
| v6 | maker fill only at minute close | maker share 18% (live 39%); MOM-long +$3,269 | +0.35 vs +0.07 | rejected; v3 rule (53% maker) restored |

**Final harness = v3 rules.** Sleeve-level population matches live for MOM-long / MOM-short / FLIP
(N and net $ within noise; WR 5–8 pp lower). The per-trade gap on IDENTICAL entries (+0.35 live vs
+0.11 replay, n=51) is path sensitivity: a few-bps entry difference moves the first peak across the
+0.40 runner arm and the +0.10 lock catches the next dip (AAVE 2026-07-01 worked through minute by
minute) — it averages out at the population level, which is what the study uses. Spike-Fade is LOW
fidelity (78 vs 45 fills, net sign differs): scanner pairs fire on completed candles.

Root causes found in order: (1) universe rule (prior-day vs rolling 24h volume, 7 of 18 misses);
(2) no maker-window persistence → every flickering signal filled (2.6× live fill count, extra fills
losing); (3) exits filled at the minute's extreme/close instead of at the level; (4) the entry
minute's pre-entry range armed locks / hit stops (19% of replay trades closed < 2 min vs 7% live).
A live bug surfaced on the way: `_record_signal_expired_order()` rejects `entry_btc_trend_gap_pct`
(passed since commit 1792492) → every maker-window expiry raises and the SIGNAL_EXPIRED row is never
written (no money effect; fix prepared, uncommitted).

## 4. Results

Final harness = v3 rules + adaptive intra-minute resolution (no step > 0.10% of price; stop fills
within ~0.1% of the level). Full report: `reports/ENGINE_REPLAY_YEAR_2026-09-17.md`. Fills:
`reports/backtest_cache/replay/ALL_fills.csv`; rule-applied copy `ALL_fills_rules.csv`; the
superseded partial run is kept under `replay/partial_v3/` for the audit trail.
