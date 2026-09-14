# Robinhood Chain Meme Coin Trading Bot — project brief (v2, corrected against SCALPARS learnings, 2026-09-14)

## 0. How to read this brief

Sections 1–2 define the project. Sections 3–36 are the operating, quant and engineering discipline carried over from SCALPARS; they are binding from day one. Sections 37 onward are the architecture and the phased plan; every threshold in them is a hypothesis to be measured, not a setting to be trusted.

## 1. Objective

Build an automated trading system for meme coins on Robinhood Chain.

Business target: eventually at least $1,000/day net trading profit. That is a scaling objective, not a number the bot may force through overtrading. The system optimises for maximum positive expectancy per dollar deployed, subject to risk, liquidity, slot-time and market-capacity constraints.

Initial reference hypothesis (to be tested, not assumed):
- position ≈ $200
- take profit +5%
- catastrophe stop −50%
- no intermediate fixed-percentage stop as a LIVE rule at first, but a maximum-hold clock IS live from version one (see §24) and every intermediate-stop variant runs as a shadow
- high frequency only if opportunities qualify: 100–500+ trades/day is possible, zero on a dead day is acceptable; never a minimum trade count

Validation order before meaningful capital: data collection → paper → forward paper → LIVE_SMALL → controlled scaling.

## 2. Context from SCALPARS

SCALPARS is a Binance USDT-M futures bot (shorting, leverage, funding, centralised book) operated with a non-developer/non-quant operator; the assistant is technical owner and quant analyst. This bot is spot-only, no shorting, no leverage initially, DEX liquidity, on-chain settlement, Uniswap pools. SCALPARS thresholds do not transfer; its discipline does. The two SCALPARS failure classes most relevant here: (a) fixed-percentage exits calibrated on one volatility population failed on a higher-volatility population, and (b) a size multiplier and a wider stop, each approved alone, compounded into a 20%-of-book loss nobody had priced.

## 3. Permanent working rules (OPERATING_RULES.md)

- Never commit or push unless the operator literally says `commit`, `push` or `commit and push`. "ok", "yes", "do it", "build it", "looks good" authorise development only.
- Before every commit: terse findings review → deep verification review → apply findings → regression suite green → show the final diff → wait for explicit git authorisation. No exception for trivial changes; trivial changes caused real trading failures.
- Ask before modifying strategy, filters, thresholds, multipliers, sizing, stop or take-profit methodology, scoring thresholds or capital limits. Build the analysis, recommend, wait. Configuration changes are trading decisions.
- When the operator asks a question, deliver the assessment and stop; do not implement until told.
- Keep OPERATING_RULES.md small and permanent, CURRENT_STATE.md small and live (strategy, gates, watchlists, experiments, risk parameters, sizing rules, deployment mode), and a decision log plus archive that is never auto-loaded. Every ship, revert or gate resolution updates CURRENT_STATE.md in place and appends to the decision log. Never grow the live files with narrative.

## 4. Money-math regression suite

Pure math, no API or chain connectivity, runs in seconds before every commit. Minimum coverage: position sizing, TP and SL levels, ATR-based sizing, maximum book loss, fees, slippage and price-impact assumptions, expectancy, capital allocation, cache keys, P&L, mark-to-market valuation, slot accounting. Every money-math change adds an invariant test that fails if the change is deleted. Example: if maximum risk per trade is 0.50% of equity, a test proves no allowed combination of position size, stop distance, volatility multiplier and signal multiplier can exceed it — and a second test proves the same for the BOOK: max concurrent positions × per-trade risk ≤ the book cap.

## 5. Strategy change governance

hypothesis → analysis → shadow/paper test → locked confirm and revert gates → operator approval → implementation → two reviews → diff → explicit commit → LIVE_SMALL → confirm or revert. Nothing goes from idea to production.

## 6. Quant discipline — evidence gates (provisional for this venue)

- Block a cohort only at N≥30, WR≤40%, average P&L below a venue-calibrated negative threshold, ≥60% never positive. Below the bar: observe only or reduce size. Never hard-block below the bar.
- Multiply a cohort only at N≥30, WR≥70%, average P&L above a venue-calibrated positive threshold, total positive. Below the bar: observe or small-size probe. Never auto-multiply.
- The SCALPARS averages (−0.20% / +0.10%) are in a 20× leveraged, ±1% world; here they must be re-expressed in this venue's units (percent of position or ATR units) before use.
- Pre-committed gates define, before the experiment starts: confirm condition, revert condition, minimum N, minimum duration, allowed early-stop. They are not loosened because a result is close. Reverting early toward the safer state is allowed; moving a bar to let a ship pass is not. A ship below a locked gate is labelled DISCIPLINE OVERRIDE with reason, date and a tighter reopen/revert condition.
- Compare across batches on average P&L %, expectancy %, return on capital; never raw dollars. Report both the 1×-normalised and the live-sized result.
- Cross-batch identity is (opened_at, token_address, pool_address, strategy); never a database id, ids reset.
- Cross-period evidence is refute-only. Old data under an older exit stack can argue against a change, never for one. Re-simulate old data under the CURRENT entries, exits, filters, sizing, fees and slippage model before using it as a counterfactual.
- Chain-wide variables (ETH move, chain volume spike, risk-on/off, congestion, broad dump) count in independent time windows, not fills.
- Concentration before any dimension filter: if one or two tokens carry most of a zone's loss, blacklist the token or pool, not the dimension.
- Caps for losing cohorts, multipliers for winning cohorts, never inverted. A profitable cohort with occasional catastrophic losses is first a sizing, stop-width, liquidity or exit problem, not an entry problem.
- Full-size, probe and half-size trades are always separate lines.
- Before any "no separator", "this variable does not matter" or "kill this sleeve" claim: sweep every stamped entry variable at three granularities and every 2D pair, then re-run on label-shuffled data; if real data does not beat shuffled noise, there is no separator. No 3D on small N.
- Any improvement found on the batch it was derived from gets a 30–50% haircut in planning.
- One strategy change at a time; measure; confirm or revert; then the next.
- Daily compound return: starting equity derived from data, P&L booked on close date, active days not calendar days, reconciled to the pinned ledger before quoting.

## 7. Risk is a fraction of the book, and the book has its own cap

- Per position: worst-case loss = a fixed fraction of equity (starting reference 0.50%), so position size = allowed risk dollars ÷ stop distance. At $20,000 equity, 0.50% = $100; with a −50% stop the position is ≈$200; with a −25% stop the same risk allows ≈$400, subject to liquidity.
- Per book: a maximum simultaneous risk cap (starting reference 5% of equity) that bounds concurrency. At 0.50% per trade that is ten fully-at-risk positions, not twenty. The SCALPARS lesson: 20 slots × $100 worst case = $2,000 = 10% of a $20k book in one bad hour if a chain-wide dump hits every open position at once, and correlated meme positions do exactly that.
- A daily loss limit that pauses new entries.
- Every configuration change prices the WORST combination of position multiplier, signal multiplier, stop width, concurrent positions, correlated exposure and slippage under stress. Parameters are never approved independently.
- Ruin risk, not expectancy, is the binding constraint on scaling.

## 8. ATR and volatility from day one

Meme coins are the high-ATR, large-excursion population that broke fixed-percentage stops on SCALPARS. Even while the live reference exit is +5% / −50%, compute from day one: ATR-like short-term range, realised volatility, entry-to-stop distance in ATR units, MAE and MFE in ATR units. Express every stop and drawdown in both percent and ATR units. Monitor the scanner population's median ATR, liquidity, market cap, pool age and price impact over time; a stop calibrated on one population is wrong on the next, and the shift is visible in those medians before it is visible in P&L.

## 9. Slot-time and the maximum-hold clock

Without shorting or leverage the failure mode is capital trapped in positions that reach neither TP nor SL. Measure from version one: capital slots, average / median / 95th-percentile slot duration, capital turnover per day, stuck-capital percentage. A maximum-hold exit is a LIVE rule in version one (starting reference to be chosen from the first data, e.g. a few hours); the "no time exit" variant is a shadow. Also track time-to-no-progress, time since last new high, time since meaningful volume, and support no-progress and momentum-dead exits without rewriting the position engine. SCALPARS learned this the expensive way: a wider stop was priced as a stop-width question while nine of 21 affected positions would have sat open for hours holding a slot.

## 10. Exit rules must match the entry state

Every exit mechanism gets a synthetic test: construct the sleeve's natural entry state, feed the first market tick, assert the exit does not fire; then test the intended trigger separately. An EMA-cross exit on SCALPARS closed positions 36 ms after open because the entry was placed on the other side of that EMA by construction.

## 11. Missing-data behaviour and cache symmetry

Every gate declares FAIL CLOSED, FAIL OPEN or OBSERVE ONLY on missing data, deliberately and tested. A rule must behave identically whether its inputs come from fresh calculation, Redis, database fallback or an RPC miss. A SCALPARS exit fired only when a cached indicator existed, which hit the top-ranked pairs and never the rest.

## 12. Absence is a logs question

"Why was there no trade in token X" starts with logs and counters, never with performance tables; a dead execution path creates no rows. The UI must make absence observable: per-pool block reason, candidate counters, per-filter block counters, signals generated, quotes requested and rejected, executions rejected, transaction failures.

## 13. Batch archive discipline

Every reset first archives the current batch under a fixed naming convention (e.g. `exports/2026-09-15_batch_001_orders.csv` and `_performance.json`), plus batch metadata (id, start, end, config version, git hash, mode, notes). A master-pool builder auto-discovers archives; keep its filter-stack version aligned with the live filters or state plainly that it is not. After a reset the dashboard is CURRENT BATCH only and is labelled with the batch start timestamp; cross-batch reads come from the archives, never from the live DB. Every trade stores its strategy config version and batch id.

## 14. Config and analytics completeness

Every new configuration field ships with: default, evidence comment, schema/JSON value, engine wiring, UI input, load handler, save handler, validation, blank-field behaviour (blank → default; explicit 0 = off only where 0 is the documented sentinel), and a test. Config saves carry a version stamp and the server rejects a save built on a stale version (a browser tab opened before a deploy must not write old values back). Every new analytics table ships on the UI and on BOTH text exports (clipboard and saved file); it is incomplete until all three exist.

## 15. Paper fills vs real fills

Paper execution on a DEX uses pool state and executable quotes, which is more faithful than a CEX ticker but still not a fill: latency, pool changes, MEV, quote expiry and block timing all differ. Label paper results SIMULATED EXECUTION; store theoretical, quoted-executable and after-cost P&L separately; from LIVE_SMALL onward compare every real trade to its paper twin (entry and exit slippage, gas, latency, quote age, failures) to build a real execution-penalty model. Anything decided in seconds during a sharp move must be evaluated on swap-level data, not candles.

## 16. Analysis communication rules

- Lead with the number and its N. Label every result: in-sample, out-of-sample, forward paper, live, reset-censored, probe-only.
- When two rows count the same trade, say so and show the de-duplicated total beside them.
- A sleeve corrected mid-batch is re-screened under its own current gates, never folded in raw.
- Every counterfactual states per batch what it removes, including the winners it removes, and whether the gain is concentrated in one batch or one trade.
- When a closed gate is shown on the dashboard, its row is frozen: pinned verdict and a date cap so later fills cannot drift the record.
- When an earlier claim is wrong: "My claim was wrong", then the corrected number, N and reason in the same message.

## 17. Core architecture

Separate layers: chain ingestion → market-data construction → pool discovery and token database → feature engine → signal engine → risk / filter engine → paper or live executor (Uniswap API, later direct contracts) → position manager → performance database → analytics and backtesting.

```
ROBINHOOD CHAIN → RPC + WebSocket → EVENT INDEXER → POOL SCANNER / TOKEN DB
→ FEATURE ENGINE → SIGNAL ENGINE → RISK / FILTER ENGINE → TRADE
→ PAPER EXECUTOR | LIVE EXECUTOR → POSITION MANAGER → PERFORMANCE DB → ANALYTICS / BACKTEST
```

## 18. Technology stack

Python 3.12+, web3.py, asyncio/aiohttp/websockets, EVM JSON-RPC + WebSocket RPC via a provider (Alchemy / QuickNode / equivalent), Uniswap (Trading API first, direct contracts later only after measured benefit), PostgreSQL, Redis, pandas/numpy, FastAPI, one FastAPI-served HTML template with Tailwind and no build step, pytest, Docker. All chain ids, RPC URLs, contract addresses and endpoints from config/environment; nothing hardcoded.

## 19. Repository structure

```
trading-bot/
  OPERATING_RULES.md   CURRENT_STATE.md   decision_log/   archive/
  config/      settings.py chains.py tokens.py defaults.py
  blockchain/  rpc.py websocket.py contracts.py event_decoder.py transaction_monitor.py
  scanner/     pool_discovery.py pool_monitor.py token_monitor.py liquidity.py
  market_data/ prices.py volume.py trades.py candles.py order_flow.py volatility.py
  features/    momentum.py volume_features.py liquidity_features.py wallet_features.py volatility_features.py
  strategy/    signal_engine.py scoring.py entry_rules.py exit_rules.py sleeves.py
  risk/        token_checks.py liquidity_checks.py sizing.py atr_risk.py slot_manager.py portfolio_limits.py exposure.py
  execution/   uniswap_api.py quote.py approvals.py swap.py transaction.py wallet.py
  positions/   position_manager.py pnl.py exits.py path_tracker.py
  paper/       simulator.py paper_executor.py
  backtest/    engine.py replay.py parameter_search.py shuffled_labels.py statistics.py
  analytics/   performance.py dimensions.py exit_analysis.py post_exit.py batch_compare.py
  storage/     postgres.py redis.py models.py migrations.py
  api/         server.py routes.py
  templates/dashboard.html   static/   exports/
  tests/       math/ exits/ sizing/ strategy/ integration/
```

## 20. Blockchain data layer

RPC for blocks, transactions, balances, contract reads, pool state, token metadata, receipts, broadcast. WebSocket subscriptions for new blocks, Uniswap Swap events, pool creation, liquidity add/remove, token transfers. Prefer subscriptions over polling.

## 21. Pool discovery and token eligibility

Discover the universe continuously from on-chain pools (token0, token1, pool address, DEX version, fee tier, creation block/time, initial and current liquidity), initially quoted against configurable trusted assets (WETH, approved stablecoins). Eligibility: minimum liquidity, recent volume, transaction count, unique traders; maximum price impact; pool age; sellability; contract and liquidity risk. Never trade something because it is pumping.

## 22. Token safety checks

Can it be sold; transfer restrictions; buy/sell tax; owner can change fees, blacklist, pause, mint; liquidity removable; upgradeable / proxy; holder concentration (deployer, top 5, top 10). A safety score is stamped on every candidate and every trade.

## 23. Market data, candles, features

Per eligible pool in Redis, persisted to PostgreSQL: price; returns over 5s/15s/30s/1m/3m/5m/15m/1h; volumes over 1m/5m/15m/1h split buy/sell; buy and sell counts; unique buyers and sellers; trade sizes; liquidity and its change; price impact at $50/$100/$200/$500/$1,000; realised volatility; ATR-like range; trade frequency; new-wallet activity. Internal candles from swaps at 5s/15s/30s/1m/5m/15m/1h with OHLC, volume, buy/sell volume, trade count, unique traders. Features: momentum (returns, acceleration, distance from local high and VWAP-like reference), volume (levels, acceleration, trade-count and unique-buyer acceleration), order flow (buy/sell ratios, sizes, largest, concentration), liquidity (total, change, concentration, impact ladder), volatility (realised, ATR, drawdown, range expansion, acceleration).

## 24. Signal engine and stamped entry state

A 0–100 score (starting weights: momentum 25, volume acceleration 20, buy pressure 15, liquidity 15, volatility quality 10, trade frequency 10, safety/other 5), starting threshold 75, all provisional. Initial entry hypothesis: liquidity > $100k, 5-min volume > $50k, 1-min return > +1%, volume acceleration > 2×, buy/sell volume ratio > 1.5, minimum recent trade count, price impact at $200 < 0.5%, score ≥ 75. Every trade permanently stores its full entry snapshot (score components, liquidity, pool age, price, market cap, returns, ATR, volume acceleration, buy/sell ratio, unique buyers, trade frequency, price impact, safety score, chain-wide conditions, time of day, config version). Never reconstruct it later.

## 25. Block reasons and filter counters

For every scanned pool save the FIRST rule that prevented entry (LOW_LIQUIDITY, LOW_VOLUME, MOMENTUM_TOO_LOW, BUY_PRESSURE_LOW, PRICE_IMPACT_TOO_HIGH, TOKEN_UNSAFE, SCORE_TOO_LOW, MAX_POSITIONS, BOOK_RISK_CAP, TOKEN_COOLDOWN, MISSING_DATA). Keep per-filter counters since process start, persisted and restored on restart.

## 26. Quote-based entry and sizing

Entries are decided on an executable quote (route, fee, slippage, price impact, minimum output, quote age), never on displayed price. Position size supports $10 to $1,000 and derives from equity, stop distance, ATR, liquidity, confidence, price impact, safety and existing exposure (§7). Slippage is configurable and later adaptive by liquidity and volatility; a trade needing excessive slippage is rejected, never retried at higher slippage.

## 27. Exits

Live reference: TP at executable value ≥ ≈$210, catastrophe stop at ≤ ≈$100, maximum hold from version one. Triggers use executable value, not chart value. The −50% level is a backstop, not a designed exit: the architecture supports tight stops, ATR stops, time stops, liquidity exits, momentum exits and dynamic stops without rewriting the position engine. Shadow exits run on every entry (e.g. TP 2.5/SL 10; TP 5/SL 15; TP 5/SL 25; TP 5/SL 50; TP 5/no SL; ATR exit; TP 5/max hold; dynamic), evaluated adverse-first on swap-level data so they are conservative bounds. Every exit writes a specific label (TAKE_PROFIT, EMERGENCY_SL, ATR_STOP, MAX_HOLD, NO_PROGRESS, LIQUIDITY_COLLAPSE, SAFETY_EXIT, MOMENTUM_EXIT, MANUAL_CLOSE); never just CLOSED. Liquidity is monitored after entry; an emergency exit path exists independent of TP/SL because a nominal −50% stop is irrelevant if the pool is drained.

## 28. Position manager and excursion tracking

Per position: id, token, pool, sleeve, entry time and block, amount, tokens received, actual entry price, price impact, score, full entry snapshot, current / highest / lowest executable value, TP, SL, ATR at entry, risk dollars, slot id, status, exit time, exit reason, realised P&L. Compute MAE and MFE in %, $ and ATR units; time to +2.5/+5/+10% and to −10/−25/−50%; time to new high; time since last progress. After exit, track the token for a defined horizon (1, 2, 5, 15, 30, 45 min) with post-exit peak and trough, so "did TP exit too early" and "did the stop avoid worse" are answerable.

## 29. Paper engine and real-execution comparison

Paper buys and marks on executable quotes every block or relevant update; models DEX fee, slippage, price impact, gas, quote expiry, latency; stores chart, quoted and after-cost P&L separately. From LIVE_SMALL, every real trade is compared to its paper twin (§15).

## 30. Performance metrics and economics

Track total / realised / unrealised / mark-to-market P&L (the dashboard always shows all three; +$2,500 closed with −$2,900 open is −$400, never "profitable"), expectancy, win rate, average win and loss, profit factor, max drawdown, capital utilisation and turnover, trades/day, holding-time distribution, MAE, MFE, return on deployed capital, risk per slot, stuck-capital %. Reference economics of +5%/−50% on $200: gross winner +$10, full loser −$100, break-even WR ≈ 90.9%; expectancy per trade ≈ −$1 at 90%, +$1.20 at 92%, +$4.50 at 95%, +$6.70 at 97%; ≈223 trades/day at 95% for ≈$1,000 gross before costs. One full loser erases about ten winners, so the analysis must concentrate on the frequency and predictors of −50% events, whether liquidity deteriorates first, whether ATR predicts them, whether earlier non-price exits reduce them, and whether sizing should shrink in those cohorts. This is the hurdle to test, not an expected result.

## 31. Slots, concurrency and turnover

Slots are a first-class resource (id, capital, age, locked capital, expected resolution, P&L, progress). Configurable limits: max simultaneous positions, max deployed capital, max per token and per pool, max correlated exposure, max new positions per minute, daily loss limit, max unresolved positions — all bounded by the book risk cap in §7. Capital requirement depends on concurrency and holding-time tails, not trades/day.

## 32. Execution layer, wallet and transaction safety

Flow: signal → buy quote → validate route / slippage / impact → build → verify (chain id, router, token, amounts, minimum output, calldata, approval, nonce, quote age) → sign locally → broadcast → receipt → record actual execution; mirror for exits. No unlimited approvals where avoidable. Dedicated bot wallet, limited funds, keys from environment first and a secrets manager or hardware signing later; never the operator's main wallet. Duplicate protection: signal ids, per-token locks, pending-transaction state, cooldowns, nonce manager, position-existence check. Failure handling for reverts, timeouts, dropped or stuck transactions, nonce conflicts, quote expiry, pool changes, insufficient gas or balance; a position is never marked open until the receipt and token balance are verified. Direct contract execution only after measured API latency, routing quality, failure rate and price difference justify it.

## 33. Sleeves, regimes, wallet intelligence

Every trade carries a sleeve (MOMENTUM live first; BREAKOUT, VOLUME_ACCELERATION, NEW_POOL, SMART_WALLET, REVERSAL as schema-ready). ESTABLISHED_MEME and NEW_POOL are separate sleeves with their own size, liquidity minimum, risk limits, score threshold and max hold. Track regimes (chain-wide meme volume, ETH direction, DEX volume, volatility state, risk-on/off, time of day, weekday) as window-counted variables. Wallet intelligence (smart-money, deployer, LP, large-holder behaviour) is a later phase, not MVP.

## 34. Research: replay, forward paper, parameter search, overfitting controls

Replay persisted chain data under the CURRENT stack; forward paper outranks in-sample history; datasets split into research / validation / out-of-sample / forward paper. Parameter search over TP, SL, ATR multiples, score, liquidity, volume, momentum and buy/sell thresholds is ranked by expectancy, drawdown, profit factor, capital efficiency, robustness and out-of-sample performance, never by in-sample profit, and always accompanied by the label-shuffled control of §6. Performance by token and by pool (token, pool, DEX version, fee tier) supports individual exclusions before any dimension filter.

## 35. Front end

One FastAPI app, one template, Tailwind, JSON polling (`/api/status`, `/api/balance`, `/api/pools`, `/api/signals`, `/api/positions/open`, `/api/positions/closed`, `/api/performance`, `/api/filters`, `/api/config`), git hash in the status endpoint.

- Header: Running/Paused, Paper/Live, server, runtime clock, git hash, batch start timestamp, Start, Pause, and a smaller Reset with confirmation that refuses while live positions are open, archives the batch first, wipes trades and tallies only, never config. Pause-new-entries and Emergency-close-all are separate controls and never the same button.
- Cards: available base asset, capital deployed, capital reserved, gas balance, gas burn rate, gas runway (amber when low), open positions, total portfolio value, realised P&L, unrealised P&L.
- Scan universe table (the "why is nothing trading" screen): token, pool, volume, price, liquidity, market cap, pool age, ATR, 1m and 5m return, volume acceleration, buy/sell ratio, score, block reason, open positions, price impact at $200, safety score.
- Transactions: one row per on-chain event (timestamp, trade id, token, pool, sleeve, action OPEN / CLOSE_TP / CLOSE_SL / CLOSE_MAX_HOLD / CLOSE_LIQUIDITY / MANUAL_CLOSE, type, amount, quantity, price, gas, fee, slippage, tx hash).
- Open positions: opened, token, pool, sleeve, score, size, entry, current executable value, P&L $ and %, duration, TP, SL, ATR at entry, MAE, MFE, high/low, risk dollars, slot id, flags, manual close.
- Closed positions: opened, closed, token, pool, sleeve, score, entry type, exit type, investment, entry, exit, fees, gas, slippage, P&L $ and %, MAE, MFE, peak, trough, duration, close reason (always populated), risk score.
- Performance page, labelled CURRENT BATCH: period performance (today/7d/30d/MTD/batch: N, WR, avg %, invested, turnover, P&L/invested, PF, fees, gas, slippage % and $, realised, projected); performance by sleeve (N, WR, avg %, total %, net $, $ per active day, PF, worst %, avg duration, avg slot time, MAE, MFE — the batch-review table); by token, pool, score band, liquidity band, ATR band, regime, pool age, time of day; trade outcome distribution buckets; closing-reason summary with post-exit performance; entry conditions by outcome; gate/experiment tables with name, hypothesis, locked bars, N, result, verdict (BUILDING / APPROACHING / CONFIRM / REVERT), frozen with a pinned verdict once closed, numbers matching CURRENT_STATE.md; exit diagnostics (exit-type performance, stop deep dive, winner drawdown, MAE before TP, MFE after TP, hold-time expectancy, max-hold / ATR-stop / TP-ladder shadows, post-exit regret); dimension tables and 2D cross-tabs as exploratory evidence only; entry funnel (scanned → eligible → momentum → volume → liquidity → safety → score → quote requested → accepted → generated → executed); filter blocks (count, share, last seen); daily P&L calendar by close date; daily compound return card reconciled to the ledger.
- Portfolio ledger (shares, NAV, deposits, withdrawals with preview-before-confirm) is not MVP.
- Config UI grouped by global risk, scanner, sleeves, liquidity, execution, exits, slot limits; each field shows default, current, evidence note and locked revert; save writes the full validated object with a version stamp.
- Two exports, Copy Review and Save Review, reproducing every analytics table.

## 36. Logging, reproducibility, data model

Every decision logs timestamp, block, token, pool, market and feature state, score, each gate result, block reason, quote, risk decision, trade decision, tx hash, result, and explicit reasons for non-trades. Given a trade id the system reconstructs why it entered, the state, gates passed, quote, risk calculation, transaction, why it exited, P&L. Trades link signal id → quote id → tx hash → position id → exit tx hash; no orphan trades. Tables: pools, tokens, market_snapshots, swaps, candles, signals, filter_events, quotes, positions, transactions, closed_trades, shadow_positions, post_exit_paths, experiments, strategy_config_versions, batch_metadata, decision_log.

## 37. Modes, phases, MVP

Modes: DATA_ONLY, PAPER, LIVE_SMALL ($10–$25), LIVE, all driven by the same engine.

Phases: 1 data collector (RPC/WS, discovery, swap decoder, price/volume/liquidity/ATR, Postgres, Redis; no trading) → 2 scanner (features, safety, score, block reasons, counters) → 3 paper trader ($200 simulated entries on executable quotes, TP +5, SL −50, max hold live, MAE/MFE/ATR units, slot time, post-exit tracking) → 4 strategy laboratory (TP/SL/ATR/hold/no-progress/score/liquidity/sizing variants via shadows and replay, shuffle-calibrated) → 5 LIVE_SMALL (real slippage, gas, latency, quote accuracy, failure rate, paper/live divergence) → 6 controlled scaling ($25 → $50 → $100 → $200, only with sufficient N, positive expectancy, acceptable execution, drawdown and slot utilisation, and the book cap respected) → 7 dynamic risk-normalised sizing.

MVP does not require live execution. It connects to the chain, discovers pools, tracks swaps, builds short-horizon market data, computes price/volume/liquidity/ATR/momentum/buy-sell/acceleration, runs safety checks, scores candidates, saves block reasons, simulates $200 trades on executable quotes with TP +5 / SL −50 / max hold, records P&L, MAE, MFE, ATR excursion, holding and slot time, exit reason, tracks post-exit paths, and produces daily analytics with text exports.

Questions the MVP must answer before live capital: eligible pools; qualified signals per day; share reaching +5% before −50%; N; net expectancy after simulated costs; average win and loss; median time to +5%; 95th-percentile holding time; how far winners fall first (MAE in ATR units); frequency of −50% events and what precedes them; positions unresolved for hours; stuck capital; price impact at $200 and $500; concurrent positions and true capital requirement; maximum drawdown; signal quality by token, liquidity, ATR and time; whether 100 / 200 / 500 trades per day exist; whether the market can support scaling toward $1,000/day.

## 38. Operations

Alerts on RPC/WS/Redis/DB failures, low gas, stuck transactions, nonce conflicts, unexpected zero trades, quote-failure, slippage or price-impact spikes, liquidity collapse in an open position, daily-loss and risk-limit breaches. Kill criteria (daily drawdown, execution-failure rate, abnormal slippage, RPC instability, wallet inconsistency, expectancy collapse, liquidity regime shift) are operator-approved before live. Pause-new-entries never affects open-position monitoring.

## 39. Instruction to the coding assistant

Act as technical owner and quant analyst for an operator who is neither. Explain material decisions plainly; surface risk before implementation; never silently alter strategy; never confuse statistical evidence with certainty; lead every analysis with the result and its N; distinguish in-sample from out-of-sample; do not overfit small samples; when a question is asked, answer it and stop; when uncertain, measure first rather than add complexity; when wrong, say so and correct in the same message; never commit or push without explicit permission.

Build first: Phase 1 data collector, Phase 2 scanner, minimum dashboard observability. The initial success criterion is not profitability; it is: can we reliably observe the Robinhood Chain meme market, reproduce every trading decision, and produce a clean dataset that proves or disproves the strategy?

## 40. Central thesis

Frequency is not the edge. The edge is finding situations where P(target) × reward exceeds P(failure) × loss plus execution costs, while preserving enough capital turnover and enough book-level safety to scale. Every number in this brief is provisional until Robinhood Chain data validates it.
