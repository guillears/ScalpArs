# Context for a new trading-bot project — learnings from SCALPARS (2026-09-14)

Written to seed a fresh session building a different bot (meme coins, spot venue, not Binance). SCALPARS is a Binance USDT-M futures bot built and operated with an operator who is neither a developer nor a quant; the assistant is the technical owner and quant analyst. These are the lessons in the order they mattered.

## 1. Working rules that prevented disasters

- Never commit or push without the literal words "commit", "push" or "commit and push". "Ok", "yes", "build it" authorise building, not git. Show the diff before every commit.
- Every commit passes two reviews first: a terse findings pass and a deep verification pass, findings applied before the diff is shown. No exceptions for "simple" changes; the simple ones broke things.
- A pure-math regression suite (sizing, exits, fees, cache keys) runs green before every commit. Every money-math change adds its invariant as a test that would fail if the change were deleted.
- Ask before changing strategy, filters, multipliers or config. Build the analysis, recommend, then wait for the word.
- Keep the operating rules file small and permanent; keep a separate small "current state" file for live gates and watchlists; push history and rationale into an archive that is never loaded by default. Every ship, revert or gate resolution updates the state file in place and appends to the decision log.

## 2. Quant discipline, learned by being corrected

- Never ship a filter from small samples. Block rule: N≥30, win rate ≤40%, average ≤−0.20%, ≥60% never positive. Multiplier rule: N≥30, win rate ≥70%, average ≥+0.10%, total positive. Below the bar, ship observe-only or at reduced size; never a block.
- Pre-committed revert and confirm gates do not move at decision time. Reverting early toward the safer state is allowed; loosening a bar to permit a ship is not. A ship below the gates is a labelled "discipline override" with a tighter reopen condition and a date.
- Compare batches on average P&L %, never raw dollars: sizing and leverage differ by batch. Report both the 1× anchor and the live-sized figure.
- Dedup trades across batches on (opened_at, pair, direction), never on database ids; ids reset on every reset.
- Cross-period evidence is refute-only: old data with old exit mechanisms can argue against a change, never for one. Re-simulate under the current exit and filter stack before any counterfactual.
- Market-wide variables count in windows, not fills: ten fills in the same hour are one observation.
- Check per-pair and per-window concentration before any dimension filter. If one or two pairs carry most of a zone's loss, blacklist the pair, do not filter the dimension.
- Caps for losers, multipliers for winners, never crossed. A winning cohort with a fat-tail loss has a sizing or stop problem, not an entry problem.
- Full-size only in every cross-batch table; probes and half-size experiments are a separate line.
- Before any "no separator exists" or "kill this sleeve" claim, run an exhaustive sweep: every stamped entry variable at three granularities and every 2D pair, then re-run it on label-shuffled data. On 54 trades the real data produced fewer survivors than noise. 3D on small N is meaningless.
- Apply a 30-50% haircut to any gain projected on the batch it was derived from. Ship one change at a time.
- Daily compound return: derive starting equity from the data, book P&L on close date, count active days not calendar days, and reconcile to the pinned ledger before quoting a number.

## 3. Engineering lessons that cost real trades

- Exit rules must be checked against the entry they serve. An "exit when price crosses above the EMA" rule closed short entries that were placed above the EMA by construction, 36 ms after open. Any new exit gets a test that opens a synthetic position in the sleeve's natural state and asserts the exit does not fire at tick one.
- Sizing and stop width compound. A 2× multiplier approved on one stop, plus a wider stop approved on one size, produced a 20%-of-book loss nobody had priced. Price the worst case of every combination, not each rule alone.
- Fixed percentage exits break when the population's volatility changes. Stops calibrated on 0.4% ATR pairs were one ATR wide on 1.5% ATR pairs and resolved in seconds. Watch for population shifts in the pairs the scanner admits.
- Cached data availability creates hidden asymmetries: a rule that only fired when a cached indicator existed hit top-ranked pairs and never the rest. Every gate should behave the same whether its inputs are cached or not, and fail in a stated direction.
- "Why were there no X trades" is a logs question first, never a data-analysis question. Dead code paths are invisible in trade data; a sleeve was silently dead for 34 days.
- A reset wipes the live database: archive every batch's orders export under a fixed naming convention before resetting, and keep a master pool builder that auto-discovers the archives. Dashboard tallies after a reset are lower bounds only.
- Every new config field ships with: default plus evidence comment, JSON value, engine wiring, UI input, load and save handlers, and a blank-field save that falls back to the default rather than silently zeroing the gate. Every new analytics table ships on the UI and on both text exports.
- Paper fills come from a ticker, real fills from a book. Anything decided in seconds around a spike needs tick data to evaluate, and even then the paper fill may not match the tape.

## 4. Analysis habits the operator enforced

- Lead with the number and its N. Say plainly when a table is in-sample, when a sleeve was corrected mid-batch (re-screen it under its own gates, never fold it in raw), and when a result rests on one trade or one day.
- When two rows count the same trade, say so, and show a de-duplicated total beside them.
- When wrong, say "my claim was wrong" and give the corrected number in the same message.

## 5. Two things that transfer directly to meme coins on a spot venue

- Meme coins are the high-ATR, large-excursion population that broke the fixed-percentage stops here. Size and stop distance must be set in ATR units from day one, and the worst-case loss per position must be a fixed fraction of the book, not of the entry.
- Without shorting or leverage the failure mode changes from squeezes to being stuck in a position that never resolves. Price the slot-time and the exit-on-no-progress rule before anything else, and keep a maximum-hold clock from the first version.

## 6. The front end — what the operator looks at, and how each table works

One page, one FastAPI app serving a single HTML template; every table is filled by polling JSON endpoints (`/api/status`, `/api/balance`, `/api/pairs`, `/api/orders/open`, `/api/orders/closed`, `/api/performance`) every few seconds. Tailwind styling, no build step. Two text exports (clipboard and saved file) reproduce every performance table so a review can be pasted into chat; a rule (D12) says a new table is incomplete until it exists on the UI and on both exports.

**Header.** Live status badge (Running / Paused), the server IP, a Paper/Live toggle, the runtime clock, one solid Start button (outlined amber Pause while running), and an outlined red Reset one size smaller with a dropdown (refused while positions are open; wipes orders and tallies, never config).

**Account Balance cards.** USDT Balance with its split into Tradeable and Reserve (reserve = max of a floor, a % of equity, and 12 h of fee burn net of BNB already held); BNB Balance (fee reserve) with burn rate per hour and runway in hours; USDT in Open Orders with the open-position count; Total Portfolio Value with the trade count. The BNB card is what the auto-swap logic maintains; the runway is the first thing to check when it turns amber.

**Top Pairs by Volume.** The scan universe (top-N by 24 h quote volume, N configurable): pair, volume, price, EMA5/8/13/20, EMA gaps, RSI, ADX, block reason (why the pair did not open on the last scan), confidence, open positions. This is the live "why is nothing trading" view: the block reason column names the first gate that stopped each pair.

**Orders tabs (three tables, same page).**
- *Transactions*: the ledger of every fill event — time, order id, pair, confidence, action (OPEN_LONG, CLOSE_SHORT, …), type (MAKER / TAKER / FALLBACK), quantity, price, investment, leverage, notional, fee. One row per event, so an open and its close are two rows with the same order id.
- *Open Orders*: one row per live position — time, pair, confidence, direction, strategy badges (sleeve name, multiplier cell, gate tags such as G51-②), entry type, investment, leverage, notional, live P&L, flag, duration, peak P&L %, entry and current price, fee, high/low since entry, quality score, and a manual close action.
- *Closed Orders*: one row per finished trade — opened, closed, pair, confidence, direction, strategy badges, entry type, exit type, investment, leverage, entry, exit, total fee, P&L $ (and %), peak P&L %, break-even level reached, duration, close reason (STOP_LOSS L1, RUNNER_TRAIL, HARD_TP_LADDER, EMA13_CROSS_EXIT, …), quality score. The close reason column is the single most useful field for post-mortems; every exit mechanism writes its own label.

**Closed Orders Performance (the analytics page below the tabs).** Everything here is computed from the closed orders in the live database, so it is reset-censored: after a reset it shows the current batch only, and cross-batch reads come from the archived exports and the master pool, not from here.
- *Period Performance*: per period (today, 7 d, 30 d, MTD, all) — trades, win rate, avg P&L %, total invested, total notional, P&L / invested, P&L / notional, profit factor, fees, slip % and slip $ (combined entry+exit slippage, positive = cost), projected balance.
- *Performance by Sleeve*: per sleeve (momentum long/short, flip short, spike fade, bull-run) — N, win rate, avg P&L %, total P&L %, net $, % per day, profit factor, worst %, average duration. This is the table the batch reviews are built on.
- *Performance by Strategy / Macro Trend / Confidence Level*, *Trade Outcome Distribution*, *Closing Reason Summary* (per close reason: count, share, avg and total P&L, avg peak and trough, post-exit peak and trough, duration).
- *Entry Conditions by Close Reason / by Outcome / by Strategy* (winners vs losers): the mean of every stamped entry indicator (pair RSI, ADX, gaps, stretch, EMA slopes, BTC RSI/ADX/slope/ATR/regime, breadth, range position, ATR, volume ratios) per cohort. This is where entry-side hypotheses start; the exhaustive sweep script is where they are tested.
- *Gate and program tables*: one table per shipped experiment with its locked bar printed next to its live tally — multiplier cells, pattern cells, spike program summary and per-fire log, graduation doors, gate 51 bands and their cell partition, gate 53 quiet stop (frozen), bull-run monitor periods and the gate 57 sleeve, runner-trail and BE-lock shadows, liquidity sizing throttles. Each row carries a verdict column (building / approaching revert / revert / confirm) computed from the same pre-committed numbers the state file records.
- *Exit mechanism diagnostics*: exit type performance, stop-loss deep dive, winning-trades drawdown, post-exit regret (what price did in the 45 minutes after every exit, at 1/2/5/15/30 min), flagged exits, hold-time expectancy, hard-TP ladder shadow, fast-exit counterfactual.
- *Dimension tables*: performance by every entry variable in buckets (gaps, RSI, ADX, DI, stretch, range position, BTC RSI/ADX/ATR/1 h RSI, regime) plus 2-D cross-tabs (range position × BTC RSI direction, BTC ATR × BTC ADX, pair RSI direction × BTC RSI direction). Useful for a first look; never a ship basis on their own.
- *Entry Funnel and Filter Blocks*: counts of candidates declined per named filter since the last restart (in-memory counters, restored from the database on boot). Paired with the block-reason column in the pairs table, this is the "absence is a logs question" instrument on the UI.
- *Daily P&L Calendar* and a Daily Compound Return card (per running day, not calendar day).

**Portfolio Management.** Investor ledger (shares, NAV, deposits and withdrawals with a preview-before-confirm modal that shows the NAV and share delta before the real POST), and the configuration form: every threshold in the config file has an input here with a tooltip carrying the evidence and the locked revert for that field, grouped by sleeve. A Save writes the whole thresholds object; blank fields fall back to defaults, an explicit 0 means off where 0 is the sentinel.

**What to carry into a new bot's UI from day one:** the close-reason label on every exit, stamped entry indicators on every order (they are the analysis surface later), a block-reason column on the scan table, a per-filter block counter, a post-exit path tracker, the sleeve table with N/WR/avg %/net $/per-day, and the two text exports. Everything else was added when a question came up, and could be again.

## Caution

These are learnings, not rules for the new venue. Fee structure, no shorting, no funding, spot-only settlement and the order types available will change which of them bind hardest. The new session should start by measuring that venue's fills before importing any threshold.
