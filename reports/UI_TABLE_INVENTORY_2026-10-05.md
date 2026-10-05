# UI Table Inventory — 2026-10-05 (read-only audit)

Scope: `templates/index.html` (24,590 lines), `main.py`, `static/live_terminal.js`, `trading_config.json` (flags checked at HEAD working tree, `paper_trading=true`).
Line numbers: **HTML** = the `<tbody>`/element line; **JS** = the line that writes it.

## Legend

- **PERF** = data comes from `GET /api/performance` (one call feeds all ~150 analytics tables). The page renders every one of them inside `loadPerformance()` (index.html 13985 → ~19780). Refresh triggers:
  - page load (the first `loadClosedOrders`);
  - every time the newest closed order changes (polled every 10 s, so in practice once per closed trade);
  - a 5-minute safety-net interval;
  - any filter dropdown change;
  - clicking the "Closed" orders tab;
  - reset, paper-mode toggle and manual close.

  The endpoint's server cost is **HIGH** per call and is shared by all of these tables. The "srv" column below gives each table's *marginal* share of that cost.
- The analytics section has **no collapsibles**. All ~150 tables plus 7 charts are always in the DOM and fully re-rendered with `innerHTML` on every PERF refresh. Nothing is skipped when a table is off-screen.
- **Flags:**
  - SHADOW = observe-only, counterfactual, probe or watch data.
  - RETIRED/STALE = the feature is off in config, or its cells are neutralised.
  - DUP = shows overlapping data with another table.
  - DEBUG = diagnostic or research instrument.
- **Exports:**
  - Y = in BOTH text exports: `copyAnalytics()` inline builder (11879–13836) and `_buildPerfLines()` (9807–11608, used by `copySplitReport()`).
  - N = in neither.
  - "cfg" = config dump (`_buildConfigLines`).

## Master table

| # | Section / tab | Title | id (HTML ln → JS ln) | Source | Refresh | CPU srv / client | Flags | In exports |
|---|---|---|---|---|---|---|---|---|
| 1 | Top — Account Balance | Balance cards (not a table) | `balance-cards` (257 → `loadBalance` 9674) | /api/balance | 10 s | MED / LOW | — | partial (header) |
| 2 | Top Pairs by Volume | Top Pairs by Volume | `pairs-table-body` (352 → `loadPairs` 19789) | /api/pairs?limit=N | 10 s | LOW-MED / MED (≤50 rows + FRENZY group) | — | N |
| 3 | Orders tabs › Transactions | Transactions | `transactions-table-body` (399 → `loadTransactions` 20095) | /api/transactions (limit 200) | 10 s only while tab active | LOW / LOW | — | N (orders CSV) |
| 4 | Orders tabs › Open | Open Orders | `open-orders-table-body` (435 → `loadOpenOrders` 20147) | /api/orders/open | **1 s, even when the tab is hidden** | LOW per call, ×86,400/day / MED (19-col string rebuilt every second) | DUP (Live-Terminal positions) | N |
| 5 | Orders tabs › Closed | Closed Orders (last 100) | `closed-orders-table-body` (471 → `loadClosedOrders` 20400) | /api/orders/closed (limit 100) | **10 s, even when the tab is hidden**; also triggers PERF | LOW / MED | — | N (orders CSV) |
| 6 | Orders tabs › BNB Swaps | BNB Swaps | `bnb-swaps-table-body` (520 → `loadBnbSwaps` 20605) | /api/bnb-swaps (limit 50) | on tab click / manual swap | LOW / LOW | — | Y |
| 7 | 📅 Daily P&L Calendar (collapsible) | Calendar grid + EOM panel | `pnl-calendar-grid` (555 → `renderPnlCalendar` 8593), `pnl-eom-panel` (556 → 8778) | /api/pnl-calendar | **2 min, even when collapsed** | MED (all closed rows, 3 cols, Python day-bucketing) / LOW | — | EOM Projection Y; grid N |
| 8 | Closed Orders Performance | Period Performance | `period-performance-body` (667 → 14065) | PERF (+ /api/balance) | PERF | LOW / LOW | — | Y |
| 9 | 〃 | Performance by Sleeve | `sleeve-perf-body` (835 → 14226) | PERF | PERF | LOW / LOW | DUP-partial with #10 | Y |
| 10 | 〃 | Performance by Strategy | `strategy-perf-body` (862 → 14257) | PERF | PERF | LOW / LOW | DUP-partial with #9 | Y |
| 11 | 〃 | Performance by Macro Trend (EMA20) | `macro-trend-perf-body` (888 → 14289) | PERF | PERF | LOW / LOW | (macro_trend_filter_enabled=false) | Y |
| 12 | 〃 | Performance by Confidence Level | `confidence-perf-body` (917 → 14319) | PERF | PERF | LOW / LOW | STALE-ish: only VERY_STRONG + STRONG_BUY enabled | Y |
| 13 | 〃 | Trade Outcome Distribution | `outcome-distribution-body` (939 → 14381) | PERF | PERF | LOW / LOW | DUP-partial with P&L Distribution chart (C3) | Y |
| 14 | 〃 | Closing Reason Summary | `close-reason-body` (977 → 14399) | PERF | PERF | LOW / LOW | — | Y |
| 15 | 〃 | Entry Conditions by Close Reason | `entry-conditions-reason-body` (1032 → 14497) | PERF | PERF | MED (many column means) / MED | — | Y |
| 16 | 〃 | Entry Conditions by Outcome (W vs L) | `entry-conditions-outcome-body` (1086 → 14571) | PERF | PERF | MED / MED | — | Y |
| 17 | 〃 | Entry Conditions by Strategy | `entry-conditions-strategy-body` (1099 → 14722) | PERF | PERF | MED / MED | DUP-partial with #18 | Y |
| 18 | 〃 | Entry Conditions by Strategy — W vs L | `entry-conditions-strategy-outcome-body` (1107 → 14725) | PERF | PERF | MED / MED | DUP-partial with #17 | Y |
| 19 | 〃 | 💰 Multiplier Cell Performance (LONG + SHORT) | `mult-cell-perf-long-body` (1136), `-short-body` (1162) → 14795 | PERF | PERF | LOW / LOW | partial: rsi_adx LONG empty, SHORT 1.0× (neutral); btc_rsi_adx 2× cells live | Y |
| 20 | 〃 | 🎲 Pattern Cell Ship Performance | `pattern-cell-perf-body` (1202 → 14811) | PERF | PERF | LOW / LOW | **RETIRED-mostly**: every pattern_cell rule is 1.0×/1.0× except UNMATCHED-L 1.5× and 2 blocks | Y |
| 21 | 〃 | 🎯 Extension Multiplier Performance | `extension-mult-perf-body` (1240 → 14875) | PERF | PERF | LOW / LOW | **RETIRED**: all 3 rules are 1.0×/1.0× | Y |
| 22 | 〃 | 📈 BTC 1h Slope × BTC ADX Multiplier Perf | `btc1h-mult-perf-body` (1278 → 14939) | PERF | PERF | LOW / LOW | live (2 rules at 2.0×) | Y |
| 23 | 〃 | 🔀 Flip Trade Log (scorecard by trigger) | `flip-trades-body` (1310 → 15816) | PERF | PERF | MED / LOW | live (FAN flip-short); flip_long disabled | Y |
| 24 | 〃 | Flip Trades × BTC-Regime x-tab | `flip-regime-xtab-body` (1330 → 15862) | PERF | PERF | LOW / LOW | — | Y |
| 25 | 〃 | 🛡️ EMA13 Strict-Mode Performance | `ema13-strict-perf-body` (1359 → 15886) | PERF | PERF | LOW / LOW | SHADOW-CF (live mechanism, May-8 study) | Y |
| 26 | 〃 | 🔀 EMA13 Cross — Disabled-Direction CF | `ema13-cross-cf-body` (1385 → 15926) | PERF | PERF | LOW / LOW | SHADOW (phantom LONG cross exit) | **N** (D12 gap) |
| 27 | 〃 | 🛡️ Trailing Min-Profit Gate — Suppressed-Fire CF | `trail-gate-cf-body` (1410 → 15956) | PERF | PERF | LOW / LOW | SHADOW (Jun-8 study) | **N** (D12 gap) |
| 28 | 〃 | 🪟 Gap-Expand Relaxation — MARGINAL vs STRICT | `gap-expand-cohort-body` (1434 → 15986) | PERF | PERF | LOW / LOW | SHADOW/STALE (Jun-8 A/B, decision long made) | **N** (D12 gap) |
| 29 | 〃 | 🚀 Spike Program Summary | `spike-summary-body` (1448 → 19203) | PERF | PERF | MED / LOW | partial-RETIRED: spike_chase_enabled=false, spike_bounce=false; fade live | Y |
| 30 | 〃 | 🚀 Spike Fires (per-fire, CHASE + FADE) | `spike-fires-body` (1473 → 19473) | PERF | PERF | LOW / MED (row per fire, grows unbounded) | partial-RETIRED (CHASE era) | Y |
| 31 | 〃 | 🎓 Graduation Doors (Jul 27 PM) | `graduation-doors-body` (1484 → 19223) | PERF | PERF | MED / LOW | **STALE** (Jul-27 promotion gates; every *_probe_enabled flag is false) | Y |
| 32 | 〃 | 🌊 Monitor Periods — episode ledger | `bullrun-periods-body` (1493 → 19247) | PERF (+ extra DB query) | PERF | MED / LOW | live (bullrun_sleeve_enabled) | Y |
| 33 | 〃 | 🌊 Gate 57 — Bull-Run Sleeve | `bullrun-body` (1501 → 19280) | PERF (+ lifetime query) | PERF | MED / LOW | live | Y |
| 34 | 〃 | 🐻 Bear-Run Windows — episode ledger | `bearrun-periods-body` (1509 → 19307) | PERF (+ DB) | PERF | MED / LOW | live, but sized as a probe (bearrun_lev_mult 0.05) | Y |
| 35 | 〃 | 🐻 Gate 60 — Bear-Run Sleeve | `bearrun-body` (1517 → 19338) | PERF (+ DB) | PERF | MED / LOW | probe-sized (0.05 lev) | Y |
| 36 | 〃 | ⚡ SURGE Sleeves | `surge-body` (1525 → 19422) | PERF (+ DB) | PERF | MED / LOW | LONG live; SHORT disabled (surge_short_enabled=false) | Y |
| 37 | 〃 | ⚡ SURGE Trigger Ledger | `surge-triggers-body` (1530 → 19447) | PERF | PERF | LOW / LOW | SHORT side shadow-only | Y |
| 38 | 〃 | 🔥 FRENZY Flagged Pairs (now) | `frenzy-flags-body` (1539 → 19370) | PERF (in-memory) | PERF | LOW / LOW | live; DUP-partial with the FRENZY group in Top Pairs | Y (frenzyReportLines) |
| 39 | 〃 | 🔥 FRENZY Fills | `frenzy-body` (1543 → 19387) | PERF (+ DB) | PERF | MED / LOW | live | Y |
| 40 | 〃 | 🔥 FRENZY Short Observations | `frenzy-breaks-body` (1547 → 19403) | PERF | PERF | LOW / LOW | SHADOW (frenzy_short_observe) | Y |
| 41 | 〃 | ⏱️ Trailing Confirmation Performance | `trailing-confirm-perf-body` (1584 → 16015) | PERF | PERF | LOW / LOW | **RETIRED** (trailing_pullback_confirmation_seconds=0) | Y |
| 42 | 〃 | ⏱️ Post-Exit P&L Snapshots — EMA13 & SL | `post-exit-snap-body` (1624 → 16214) | PERF | PERF | MED / LOW | SHADOW (post-exit CF); DUP-partial with #77 | Y |
| 43 | 〃 | 🚦 Fast-Exit CF — Grid / Direction / Close-reason (3 tables) | `fast-exit-grid-body` (1663 → 16100), `fast-exit-direction-body` (1683 → 16141), `fast-exit-cr-body` (1727 → 16164) | PERF | PERF | **MED-HIGH** (threshold×window grid × orders × post-exit snaps) / LOW | **SHADOW + RETIRED** (fast_exit_enabled=false) | Y |
| 44 | 〃 › 📐 Entry Extension | Perf by Entry Distance from EMA13 (L/S) | `ema13-ext-long/short-body` (1765/1786 → 15051) | PERF | PERF | LOW / LOW | DEBUG/research (filter off) | Y |
| 45 | 〃 | Extension × Pair Vol Ratio (L/S) | `ema13-ext-pvol-*` (1812/1831 → 15058) | PERF | PERF | LOW / LOW | DEBUG | Y |
| 46 | 〃 | Extension × ADX Delta (L/S) | `ema13-ext-adxd-*` (1857/1876 → 15063) | PERF | PERF | LOW / LOW | DEBUG | Y |
| 47 | 〃 | Extension × Pair ADX (L/S) | `ema13-ext-pair-adx-*` (1902/1921 → 15068) | PERF | PERF | LOW / LOW | DEBUG | Y |
| 48 | 〃 › 🌐 BTC Market Extension | Perf by BTC Market Extension (L/S) | `btc-ext-long/short-body` (1959/1980 → 15097) | PERF | PERF | LOW / LOW | DEBUG | Y |
| 49 | 〃 | BTC Ext × Global Vol Ratio (L/S) | `btc-ext-gvol-*` (2006/2025 → 15104) | PERF | PERF | LOW / LOW | DEBUG | Y |
| 50 | 〃 | ⚠ Double-Stretch: BTC Ext × Pair Ext (L/S) | `btc-ext-pair-*` (2051/2070 → 15109) | PERF | PERF | LOW / LOW | DEBUG | Y |
| 51 | 〃 › 📏 BTC 24h-Range Position | BTC below 24h HIGH (L/S) | `btc-off24h-long/short-body` (2084/2085 → loop 15189) | PERF | PERF | LOW / LOW | live gate analytics (bull-run gate) | Y |
| 52 | 〃 | BTC above 24h LOW (L/S) | `btc-off24lo-long/short-body` (2089/2090 → 15189) | PERF | PERF | LOW / LOW | live (bear-run gate) | Y |
| 53 | 〃 › 🕐 BTC 1h Slope | Perf by BTC 1h Slope (L/S) | `btc-1h-slope-long/short-body` (2119/2136 → 15183) | PERF | PERF | LOW / LOW | — | Y |
| 54 | 〃 | 5m × 1h Slope Alignment (L/S) | `btc-align-*` (2158/2174 → 15199) | PERF | PERF | LOW / LOW | DEBUG | Y |
| 55 | 〃 | BTC 1h Slope × BTC ADX x-tab (L/S) | `btc-1h-slope-adx-*` (2195/2212 → 15204) | PERF | PERF | LOW / LOW | DUP-partial with #22 | Y |
| 56 | 〃 | BTC ATR × BTC ADX x-tab (L/S) | `btc-atr-adx-*` (2233/2250 → 15233) | PERF | PERF | LOW / LOW | DUP-likely with #109 | **N** (D12 gap) |
| 57 | 〃 | 🧬 4-Cohort Pattern Coverage | `pattern-4cohort-body` (2283 → 15279) | PERF | PERF | MED (pattern signature eval per order) / LOW | SHADOW (pattern research) | Y |
| 58 | 〃 | 🧩 Pattern C Combination Tracker | `pattern-c-combo-body` (2319 → 15366) | PERF | PERF | MED / MED | SHADOW; DUP-partial with #61 | Y |
| 59 | 〃 | 🧩 Pattern W Combination Tracker | `pattern-w-combo-body` (2354 → 15367) | PERF | PERF | MED / MED | SHADOW; DUP-partial with #63 | Y |
| 60 | 〃 | 🧩 Pattern C + W Combination Tracker | `pattern-cw-combo-body` (2390 → 15368) | PERF | PERF | MED / MED | SHADOW | Y |
| 61 | 〃 | 🎯 Pattern C Tracker (observation-only) | `pattern-c-body` (2431 → 15372) | PERF | PERF | MED / LOW | SHADOW | Y |
| 62 | 〃 | 🔍 Unmatched Losers Deep Dive | `unmatched-losers-body` (2485 → 15497) | PERF | PERF | LOW / MED (per-trade rows) | SHADOW | Y |
| 63 | 〃 | 🏆 Pattern W Tracker (observation-only) | `pattern-w-body` (2527 → 15538) | PERF | PERF | MED / LOW | SHADOW | Y |
| 64 | 〃 | 🔎 Unmatched Winners Deep Dive | `unmatched-winners-body` (2577 → 15634) | PERF | PERF | LOW / MED | SHADOW | Y |
| 65 | 〃 | 🏃 Runner Trail Performance | `runner-trail-body` (2614 → 15672) | PERF | PERF | LOW / LOW | live | Y |
| 66 | 〃 | 💧 Liquidity Sizing | `liq-sizing-body` (2642 → 15714) | PERF | PERF | LOW / LOW | live | Y |
| 67 | 〃 | 🏃 Leash Shadow: Exit Capture by Direction | `leash-shadow-body` (2684 → 15753) | PERF | PERF | LOW / LOW | SHADOW (May-30; decided Jun-29) | Y |
| 68 | 〃 | Entry Type Performance | `entry-type-perf-body` (2715 → 18488) | PERF (+ SIGNAL_EXPIRED rows) | PERF | LOW / LOW | — | Y |
| 69 | 〃 | Exit Type Performance | `exit-type-perf-body` (2743 → 18564) | PERF | PERF | LOW / LOW | — | Y |
| 70 | 〃 | Signal Expired Breakdown | `signal-expired-breakdown-body` (2770 → 18618) | PERF | PERF | LOW / LOW | — | Y |
| 71 | 〃 | 🔎 Entry Funnel (cards) | `entry-funnel`, `entry-funnel-blockers` (2781 → 14148) | PERF | PERF | LOW / LOW | DUP-partial with #72 | Y |
| 72 | 〃 | Filter Blocks | `filter-blocks-body` (2807 → `renderFilterBlocks` 9126, called from `loadStatus` 9443) | **/api/status** | **10 s** | LOW (in-memory counters) / MED (re-rendered every 10 s) | — | Y |
| 73 | 〃 | Stop Loss Deep Dive | `sl-deep-dive-body` (2843 → 18658) | PERF | PERF | LOW / LOW | — | Y |
| 74 | 〃 | Winning Trades Drawdown | `winning-drawdown-body` (2878 → 18741) | PERF | PERF | LOW / LOW | DUP-partial with MAE/MFE chart | Y |
| 75 | 〃 | HARD_TP Mechanism Shadow (Jul 22) | `hard-tp-shadow-body` (2902 → 18873) | PERF | PERF | LOW / LOW | SHADOW (the ladder now ships live; study done) | Y |
| 76 | 〃 | 🩹 Recovery Hold | `recovery-hold-body` (2924 → 18896) | PERF | PERF | LOW / LOW | live (recovery_hold_enabled) | Y |
| 77 | 〃 | Post-Exit Regret Deep Dive | `post-exit-regret-body` (2985 → 18910) | PERF | PERF | MED / LOW | SHADOW; DUP-partial with #42 | Y |
| 78 | 〃 | Flagged Exits (Signal Lost Flag) | `flagged-exits-body` (3021 → 18978) | PERF | PERF | LOW / LOW | **RETIRED** (signal_lost_flag_enabled=false, fl2 off) | Y |
| 79 | 〃 | Hold-Time Expectancy | `hold-time-expectancy-body` (3049 → 19685) | PERF | PERF | LOW / LOW | DUP-partial with #127 | Y |
| 80 | 〃 | Perf by Entry Gap 5-20 | `gap-performance-body` (3074 → 16260) | PERF | PERF | LOW / LOW | — | Y |
| 81 | 〃 | Perf by Entry Gap EMA5-EMA8 | `ema58-gap-performance-body` (3100 → 16303) | PERF | PERF | LOW / LOW | — | Y |
| 82 | 〃 | Perf by Entry Gap EMA8-EMA13 | `ema813-gap-performance-body` (3126 → 16347) | PERF | PERF | LOW / LOW | — | Y |
| 83 | 〃 | Perf by EMA Fan Acceleration | `ema-fan-accel-performance-body` (3153 → 16388) | PERF | PERF | LOW / LOW | — | Y |
| 84 | 〃 | Perf by Entry RSI | `rsi-performance-body` (3178 → 16426) | PERF | PERF | LOW / LOW | — | Y |
| 85 | 〃 | Perf by Range Position | `range-position-performance-body` (3204 → 16469) | PERF | PERF | LOW / LOW | — | Y |
| 86 | 〃 | Range Position × BTC RSI Dir | `range-pos-btc-rsi-dir-crosstab-body` (3230 → 16502) | PERF | PERF | LOW / LOW | — | Y |
| 87 | 〃 | Range Position × Pair RSI Dir | `range-pos-pair-rsi-dir-crosstab-body` (3256 → 16533) | PERF | PERF | LOW / LOW | — | Y |
| 88 | 〃 | Perf by ADX Delta | `adx-delta-performance-body` (3281 → 16563) | PERF | PERF | LOW / LOW | — | Y |
| 89 | 〃 | Perf by Pair −DI | `neg-di-performance-body` (3306 → 16628) | PERF | PERF | LOW / LOW | — | Y |
| 90 | 〃 | Perf by Pair +DI | `pos-di-performance-body` (3331 → 16629) | PERF | PERF | LOW / LOW | — | Y |
| 91 | 〃 | Perf by Entry ADX | `adx-performance-body` (3356 → 16632) | PERF | PERF | LOW / LOW | — | Y |
| 92 | 〃 | Perf by ADX Direction | `adx-direction-performance-body` (3382 → 16675) | PERF | PERF | LOW / LOW | — | Y |
| 93 | 〃 | Perf by RSI Direction | `rsi-direction-performance-body` (3408 → 16717) | PERF | PERF | LOW / LOW | (rsi_momentum_filter off) | Y |
| 94 | 〃 | Perf by EMA5 Stretch | `stretch-performance-body` (3433 → 16759) | PERF | PERF | LOW / LOW | — | Y |
| 95 | 〃 | Perf by Entry Quality Score | `quality-score-performance-body` (3458 → 16802) | PERF | PERF | LOW / LOW | — | Y |
| 96 | 〃 | Perf by BTC Regime (at Entry) | `regime-performance-body` (3484 → 16840) | PERF | PERF | LOW / LOW | — | Y |
| 97 | 〃 | Regime Transition Impact | `regime-transition-body` (3507 → 16891) | PERF | PERF | LOW / LOW | DUP-partial with #128 | Y |
| 98 | 〃 | Perf by Pair EMA20 Slope | `pair-slope-performance-body` (3536 → 16915) | PERF | PERF | LOW / LOW | — | Y |
| 99 | 〃 | Perf by BTC EMA20 Slope | `btc-slope-performance-body` (3565 → 16950) | PERF | PERF | LOW / LOW | — | Y |
| 100 | 〃 | Perf by Pair EMA13-EMA50 Gap | `pair-ema20-ema50-gap-performance-body` (3595 → 16985) | PERF | PERF | LOW / LOW | observation-only label | Y |
| 101 | 〃 | Perf by BTC EMA13-EMA50 Gap | `btc-ema20-ema50-gap-performance-body` (3625 → 17022) | PERF | PERF | LOW / LOW | observation-only label | Y |
| 102 | 〃 | Perf by BTC ADX | `btc-adx-performance-body` (3653 → 17059) | PERF | PERF | LOW / LOW | — | Y |
| 103 | 〃 | Perf by BTC ADX Direction | `btc-adx-direction-performance-body` (3679 → 17093) | PERF | PERF | LOW / LOW | — | Y |
| 104 | 〃 | Perf by BTC RSI Direction (5m) | `btc-rsi-direction-performance-body` (3705 → 17130) | PERF | PERF | LOW / LOW | — | Y |
| 105 | 〃 | Perf by BTC RSI Direction (30m) | `btc-rsi-direction-30m-performance-body` (3731 → 17167) | PERF | PERF | LOW / LOW | — | Y |
| 106 | 〃 | BTC RSI 30m × 5m x-tab | `btc-rsi-30m-5m-crosstab-body` (3757 → 17200) | PERF | PERF | LOW / LOW | — | Y |
| 107 | 〃 | Perf by BTC Volatility (ATR%) | `btc-volatility-performance-body` (3783 → 17232) | PERF | PERF | LOW / LOW | — | Y |
| 108 | 〃 | Perf by BTC 1h RSI Direction | `btc-rsi-1h-direction-performance-body` (3809 → 17261) | PERF | PERF | LOW / LOW | — | Y |
| 109 | 〃 | BTC Volatility × BTC ADX x-tab | `btc-vol-adx-crosstab-body` (3835 → 17290) | PERF | PERF | LOW / LOW | **DUP-likely** of #56 BTC ATR × BTC ADX (verify buckets) | Y |
| 110 | 〃 | BTC 1h RSI × 5m RSI x-tab | `btc-rsi-1h-5m-crosstab-body` (3861 → 17319) | PERF | PERF | LOW / LOW | (BTC_1H_5M_RSI_DIR gate live) | Y |
| 111 | 〃 | 🎯 BTC 1h × 30m RSI Dir x-tab (L/S) | `btc-1h-30m-rsi-long/short-body` (3891/3908 → 15271) | PERF | PERF | LOW / LOW | — | Y |
| 112 | 〃 | Perf by BTC Entry RSI | `btc-rsi-performance-body` (3935 → 17350) | PERF | PERF | LOW / LOW | — | Y |
| 113 | 〃 | Pair ADX Dir × BTC ADX Dir | `adx-dir-crosstab-body` (3961 → 17384) | PERF | PERF | LOW / LOW | — | Y |
| 114 | 〃 | Pair RSI Dir × BTC RSI Dir | `rsi-dir-crosstab-body` (3987 → 17418) | PERF | PERF | LOW / LOW | — | Y |
| 115 | 〃 | Pair EMA20 Slope × Pair ADX | `pair-slope-adx-crosstab-body` (4014 → 17488) | PERF | PERF | LOW / LOW | — | Y |
| 116 | 〃 | BTC Slope × BTC ADX | `btc-slope-adx-crosstab-body` (4041 → 17497) | PERF | PERF | LOW / LOW | — | Y |
| 117 | 〃 | ADX Delta × BTC ADX | `adx-delta-btc-adx-crosstab-body` (4068 → 17506) | PERF | PERF | LOW / LOW | (filter live) | Y |
| 118 | 〃 | BTC EMA13-50 Gap × BTC ADX | `btc-gap-btc-adx-crosstab-body` (4095 → 17540) | PERF | PERF | LOW / LOW | (filter live) | Y |
| 119 | 〃 | Pair EMA13-50 Gap × Pair ADX | `pair-gap-pair-adx-crosstab-body` (4122 → 17578) | PERF | PERF | LOW / LOW | — | Y |
| 120 | 〃 | Perf by RSI × ADX | `rsi-adx-crosstab-body` (4148 → 17615) | PERF | PERF | LOW / LOW | DUP-partial with #19 (mult cells live in this grid) | Y |
| 121 | 〃 | Perf by BTC RSI × BTC ADX | `btc-rsi-adx-crosstab-body` (4174 → 17647) | PERF | PERF | LOW / LOW | DUP-partial with #19 | Y |
| 122 | 〃 | Volume Cross-Tab (Global × Pair) | `volume-crosstab-body` (4200 → 19022) | PERF | PERF | LOW / LOW | DUP-partial with #123 | Y |
| 123 | 〃 | 📊 Volume Intersection (GVR × Pair Vol USD) | `volume-intersection-body` (4224 → 19049) | PERF | PERF | LOW / LOW | DUP-partial with #122 / #131 | Y |
| 124 | 〃 | Perf by Market Breadth | `breadth-crosstab-body` (4249 → 19081) | PERF | PERF | LOW / LOW | (breadth filter off) | Y |
| 125 | 〃 | 📊 Perf by Pair ATR(14)% | `atr-bucket-perf-body` (4275 → 19527) | PERF | PERF | LOW / LOW | — | Y |
| 126 | 〃 | 📊 ATR × Hold-Time x-tab | `atr-holdtime-body` (4301 → 19554) | PERF | PERF | LOW / LOW | DUP-partial with #127 (transpose) | Y |
| 127 | 〃 | ⏱ Hold-Time × ATR | `holdtime-atr-body` (4323 → 19589) | PERF | PERF | LOW / LOW | DUP-partial with #126 / #79 | Y |
| 128 | 〃 | 🎲 Regime-Drift × Outcome | `regime-drift-body` (4344 → 19613) | PERF | PERF | LOW / LOW | DEBUG (Jun-21 study) | Y |
| 129 | 〃 | 🔀 Regime-Change-Exit CF | `regime-exit-cf-body` (4369 → 19644) | PERF | PERF | LOW / LOW | SHADOW (regime_change_exit_enabled=false) | Y |
| 130 | 〃 | Performance by Pair | `pair-performance-body` (4397 → 19107) | PERF | PERF | LOW / **MED-HIGH** (one row per pair ever traded, unbounded; blacklist buttons) | — | Y |
| 131 | 〃 | 📊 Perf by Pair 24h Volume | `pair-volume-bucket-body` (4424 → 19143) | PERF | PERF | LOW / LOW | DUP-partial with #132 | Y |
| 132 | 〃 | 🏅 Perf by Pair Rank | `pair-rank-perf-body` (4449 → 19177) | PERF | PERF | LOW / LOW | DUP-partial with #131 | Y |
| 133 | 〃 | 🐣 Perf by Pair Listing Age | `pair-age-perf-body` (4475 → 19501) | PERF | PERF | LOW / LOW | — | Y |
| 134 | 〃 | Never Positive Deep Dive | `never-positive-deep-dive-body` (4508 → 17679) | PERF | PERF | LOW / LOW | DUP-partial with #73 (SL never-positive) | Y |
| 135 | 〃 | Day x Time Heatmap (UTC-3) | `day-time-heatmap-body` (4597 → 18427) | PERF | PERF | LOW / LOW | its marginal totals = charts C5 + C6 | Y |
| 136 | Portfolio Management | Investors table (+ expandable ledger) | `investors-table-body` (4662 → `loadInvestors` 21334); ledger on click → /api/investors/{id}/ledger | /api/investors | **10 s** | MED (repeats the paper-balance aggregates of /api/balance) / LOW | — | NAV history Y; investor rows N |
| 137 | Trading Configuration (collapsed by default) | 🖐 Manual stop by leverage | `manual-stop-table` (5127 → `renderManualStopTable` 20816) | derived client-side from /api/config | on load / config edit | — / LOW | — | N (derived) |
| 138 | 〃 | ⚖️ Sleeve Sizing | `sleeve-sizing-body` (5138 → `renderSleeveSizing` 21018) | /api/config + /api/balance | on load / edit | LOW / LOW | — | cfg |
| 139 | 〃 | Legacy RSI Thresholds | (no id, 6330; `hidden` div) | /api/config | on load | — | **RETIRED** (hidden, "kept for config compatibility") | cfg |
| 140 | 〃 | Momentum Signals (EMA5/EMA8 Gap) | (no id, 6422) | /api/config | on load | — | — | cfg |
| 141 | 〃 | Filter-rule editors: BTC RSI×ADX, ADX-Δ×BTC ADX, RngPos×ADX-Δ, BTC 1h/5m RSI dir, BTC Gap×BTC ADX, BTC ATR×BTC ADX, RSI×ADX (each L/S) | `*-filter-long/short-body` (6952–7406 → `loadXxxFilter` 22762–23133) | /api/config | on load | — / LOW | input editors, not analytics | cfg |
| 142 | 〃 | Multiplier-rule editors: RSI×ADX mult, BTC RSI×ADX mult (L/S), Pattern Cell rules, Extension mult rules, BTC 1h mult rules | `rsi-adx-mult-*`, `btc-rsi-adx-mult-*` (7445–7472), `pattern-cell-rules-body` (7505), `extension-mult-rules-body` (7536), `btc1h-mult-rules-body` (7566) → 23186–23428 | /api/config | on load | — / LOW | Extension rules all 1.0× (RETIRED); Pattern cells mostly 1.0× | cfg |
| 143 | 〃 | Confidence Levels & Risk Management | (no id, 7792) | /api/config | on load | — / LOW | LOW/MEDIUM/HIGH/EXTREME rows disabled | cfg |
| 144 | Live Terminal view (separate view) | POSITIONS / PAIR FUNNEL panels | `lt-positions-body`, `lt-funnel-body` (8374/8379, `static/live_terminal.js` 1407) | SSE /api/terminal/stream + /api/orders/open | 3 s, only while the terminal is visible | LOW / LOW | DUP of #4 | N |

### Charts (non-table panels; all PERF, all inside "Closed Orders Performance")

| # | Title | id (HTML → JS) | Payload key | In exports |
|---|---|---|---|---|
| C1 | Performance Over Time | `performance-over-time-chart` (4520 → 17731) | performance_over_time | Y |
| C2 | Cumulative P&L (Equity Curve) | `equity-curve-chart` (4532 → 17894) | equity_curve | N |
| C3 | P&L Distribution | `pnl-distribution-chart` (4544 → 18012) | pnl_distribution (+ stats) | N (Outcome Distribution covers it partly) |
| C4 | Excursion Scatter (MAE vs MFE) | `mae-mfe-chart` (4554 → 18147) | mae_mfe — one point per trade | N |
| C5 | Performance by Hour (UTC-3) | `hourly-performance-chart` (4564 → 18233) | hourly_performance | N (DUP of heatmap marginals) |
| C6 | Performance by Day of Week | `daily-performance-chart` (4571 → 18331) | daily_performance | Y |
| C7 | NAV / Share History | `nav-history-chart` (4643 → 21207, hash-guarded) | /api/investors nav_history | Y |

Charts that are not tables:
- the Live-Terminal canvases: radar, heat compass, equity and heatmap (terminal-only, gated by `state.inTerminal`);
- the header chips (bull-run, bear-run, surge, off-24h), fed by /api/status.

## Polling loops (start on DOMContentLoaded, index.html 8436–8537)

All of these are gated only by `document.hidden`, so they **keep running while the Live-Terminal view is shown**. They pause only when the browser tab itself is hidden. Each open browser tab polls independently.

| Interval | Calls | Notes |
|---|---|---|
| **1 s** | `/api/orders/open` → Open Orders table | Rebuilt even when the Open tab is not selected. |
| **10 s** (`batch10`) | `/api/balance`, `/api/pairs?limit=N`, `/api/orders/closed`, `/api/investors`, `/api/status`; plus `/api/transactions` only if the Transactions tab is active | Closed-orders signature change → full `/api/performance`. |
| 60 s | `/api/server-ip` | trivial |
| 120 s | `/api/pnl-calendar` | runs even when the calendar is collapsed |
| 300 s | `/api/performance` + `/api/balance` (safety net) | |
| event | `/api/performance` | On each closed trade, filter change, Closed-tab click, reset, paper-mode toggle and manual close. Copy Analytics = 1 extra call; Copy Split Report = **2** extra calls (BULLISH + BEARISH). |
| 10 min (opt-in) | `/api/decisions/export.csv?days=3` | only if the "Auto 2×/day" checkbox is on |
| SSE + 3 s / 5 s / 1 s | Live Terminal: `/api/terminal/stream`, `/api/orders/open` every 3 s, plus local timers | only while the terminal view is active |

## Heaviest costs (rough)

1. **`/api/performance` — the dominant cost by far.** Per call (main.py 2711 → `_compute_performance` 4793–~9650, ~9,000 lines):
   - loads **every CLOSED order as a full ORM object** (Order = 370 columns), plus SIGNAL_EXPIRED rows, open orders and ≥4 extra lifetime queries (bull-run, bear-run, surge, frenzy);
   - makes a balance call;
   - runs ~1,200 `for o in …` passes in pure Python to build ~150 tables and 7 chart series;
   - has **no server cache**.

   The client then rebuilds ~150 tables and 7 Chart.js charts with `innerHTML`. Two tables grow without bound: #130 Performance by Pair and #30 Spike Fires, plus the MAE/MFE scatter (one point per trade). Every closed trade triggers one recompute, so cost grows as trades × history.
2. **`/api/orders/open` at 1 Hz.** Each call is small (only OPEN rows, a loop over the in-memory cache), but it runs 86,400 times a day per open tab. On top of that, a 19-column HTML string is rebuilt every second. While the terminal view is open, a second poll every 3 s runs alongside it.
3. **`/api/status` every 10 s.** It runs `_recompute_bnb_burn_rate` (SUM/COUNT aggregates over the orders table), the gross margin/notional sums, portfolio components, the leverage-bracket count and the monitor payloads (bull/bear/surge/frenzy), and it re-renders the Filter Blocks table.
4. **The paper-balance recompute, run 3× every 10 s.**
   - `/api/balance` runs `_recalculate_paper_balance` (~5 SUM aggregates) and `_recalculate_paper_bnb` (3 SUMs), then loads the open orders.
   - `/api/investors` repeats the same work through `_portfolio_components`.
   - `/api/status` may repeat it a third time through `_portfolio_components`.

   The result is about **15–20 aggregate queries every 10 s** for the same numbers.
5. **`/api/pnl-calendar` every 2 min** (all closed rows; 3 columns; Python bucketing; runs even when collapsed), and **`/api/pairs` every 10 s** (PairData select plus a GROUP BY of open positions; up to 50 rows rendered).

## Notes / uncertainties

- **Wasted CPU:**
  - Every PERF table is computed and rendered regardless of whether it is visible or scrolled to. The page has no lazy or collapsed analytics sections.
  - Open and Closed orders are rendered every 1 s / 10 s while their tabs are hidden.
  - The calendar polls while collapsed.
  - Dashboard polling continues behind the Live-Terminal view.
- Clicking the "Closed" orders tab fires a full `/api/performance` (`switchOrdersTab`, 20085), even though the analytics live in a different section.
- **D12 parity gaps found (UI-only, in neither text export):**
  - #26 EMA13 Cross Disabled-Direction CF
  - #27 Trailing Min-Profit Gate CF
  - #28 Gap-Expand MARGINAL vs STRICT
  - #56 BTC ATR × BTC ADX cross-tab
  - charts C2 (equity curve), C3 (P&L distribution), C4 (MAE/MFE) and C5 (hourly)
  - the P&L calendar grid
- **Flag judgements to verify:**
  - #31 Graduation Doors STALE: all `*_probe_enabled` flags are false. `nonexp_calm3d_enabled=true` may still feed one door row.
  - #109 vs #56 duplicate: both are BTC volatility (ATR%) × BTC ADX. Bucket edges were not compared.
  - #41 Trailing Confirmation: the engine still writes `trailing_first_pullback_pnl_pct` (trading_engine 18488) on every trail fire. With confirmation seconds = 0 the delta is ~0, so the table is informationally dead. It also costs an extra DB UPDATE per trade.
  - #25 EMA13 Strict and #75 HARD_TP Shadow are live-data studies whose decisions were already taken. I flagged them SHADOW, not RETIRED.
- `trading_config.json` holds the sleeve flags under `thresholds.*`. Keys missing from the JSON fall back to `config.py` defaults, which were not individually checked.
- I did not take server timings. The CPU ratings are static estimates from the code shape (query type, row width, loop count, call frequency). Profiling one `/api/performance` call against the production DB would put a number on #1.
- Line numbers refer to the working tree on 2026-10-05, which has uncommitted changes in `scripts/` only; index.html and main.py are unmodified.
