# Decision Log

Chronological record of every ship / demote / revert / A-B / batch decision.

**Pre-2026-06-02 decisions:** full verbatim text in `HISTORY_FULL_through_2026-06-02.md`. Index of all entries below (date + title); open HISTORY_FULL for the full evidence/rationale of any one.

**New decisions (2026-06-02 onward):** appended in full at the bottom under '## NEW ENTRIES'.

---

## Historical index (pre-2026-06-02, see HISTORY_FULL for full text)

- [NEW ENTRIES] June 3, 2026 — REVERTED `adx_dir_long` rising→both (LONG side was backwards on proper proxy; shipped on broken full-pool adx_delta). SHORT stays rising.
- [NEW ENTRIES] June 3, 2026 — SHIPPED: `no_trade_pairs` track-only mechanism + put BTCUSDT in it (visible in volume list, entries blocked)
- [NEW ENTRIES] June 3, 2026 — WHITELISTED BTCUSDT (user override of the blacklist, AGAINST evidence; revert gate locked)
- [NEW ENTRIES] June 3, 2026 — TRIMMED pair_blacklist 23→11: RELEASED 12 thin-N pairs for forward re-test (kept 5 evidenced losers + commodities + no-data)
- [NEW ENTRIES] June 3, 2026 — BLACKLISTED BTCUSDT (structural low-vol loser: edge < fee)
- [NEW ENTRIES] June 3, 2026 — SHIPPED: BTC-Accel Chase Filter (STATEFUL, LONG only) — block LONG when BTC EMA20 slope > last-LONG within 30min (chasing)
- [NEW ENTRIES] June 3, 2026 — SHIPPED: BTC 1h Slope MIN floor `btc_1h_slope_min_short = -0.60` (SHORT only; LONG plumbed-but-off)
- [NEW ENTRIES] June 2, 2026 (evening) — SHIPPED: Pair ADX Direction filter `both` → `rising` (BOTH LONG + SHORT; falling-ADX = 1W/9L cross-batch)
- [L3] June 2, 2026 — 🚨 LOCKED GO-LIVE WATCH: liquidity-aware sizing (gross 30× + redeploy + ① cap)
- [L58] June 2, 2026 — RE-ENABLED Global Volume Filter (as-is) + resolved the May 30 fan-redundancy A/B
- [L105] June 2, 2026 — Liquidity-sizing skip + redeploy counters (Filter Blocks, observation-only)
- [L133] June 2, 2026 — TUNED: Redeploy Leftover ON + max_gross_leverage 25 → 30
- [L168] June 2, 2026 — SHIPPED: Liquidity-sizing reporting surface (pure observability)
- [L224] June 2, 2026 — DEMOTED: PAIR_35-40_30-35 SHORT multiplier 2.0× → 1.0×
- [L248] June 2, 2026 — SHIPPED: Liquidity-aware position sizing (3 caps, all under `investment`)
- [L297] June 2, 2026 — RE-ENABLED ADX Δ × BTC ADX Cross-Filter (both directions) — ends May 18 A/B
- [L369] June 1, 2026 (later) — UNMATCHED LONG multiplier 1.0× → 2.0× (no fixed SL — deliberate)
- [L424] June 1, 2026 (later) — `range_position_max_long` 98 → 97.5 (boundary trim, not an edge play)
- [L474] June 1, 2026 — SHIPPED: Runner Stretch-Trail (scoped high-ATR LONG runner exit) + Leash Shadow redefine
- [L572] June 1, 2026 — SHIPPED SHORT entry: pair-ATR-min <0.25 + fan upper 1.65→1.90 (BEARISH batch)
- [L680] June 1, 2026 — SHIPPED LONG entry: BTC RSI 50-55 full block + NEW pair-ATR-min filter (<0.25)
- [L771] May 31, 2026 — WATCHLIST: high-ATR LONG = the runner cohort + asymmetric runner exit (NOT shipped, N≥30 gate)
- [L840] May 31, 2026 — C1 SHORT: fixed SL −0.70% added (cap the ATR-widened tail, NOT an entry filter)
- [L891] May 31, 2026 — 🚨 POST-MORTEM + LOCKED CHECKLIST: removing a table can swallow shared module-level constants
- [L938] May 31, 2026 — RETIRED 2 observation-only report surfaces (Phantom BE + Time-to-L1)
- [L963] May 31, 2026 — fan_ratio SHORT: floor lowered `1.02-1.65` → `1.00-1.65`
- [L1009] May 31, 2026 — fan_ratio LONG: added >5.0 flat-base cap (`0.85-1.70` → `0.85-1.70,5.0-99`)
- [L1060] May 31, 2026 — strpk K-bracket (0.4 / 0.3 looser stretch-trail variants) — observation-only
- [L1096] May 31, 2026 — Leash fire-minute capture (pre/post-close) — both tables (observation-only)
- [L1123] May 31, 2026 — Leash Shadow calibration fix + Post-Exit Regret stretch-band columns (observation-only)
- [L1158] May 30, 2026 — ADX-min tighten DECISION (5-batch deep dive): SHIPPED LONG 15→18 · HELD SHORT (data contradicted)
- [L1196] May 30, 2026 — BTC Independent Filters AUDIT (observation; NOTHING changed — locked for next-batch decision)
- [L1272] May 30, 2026 — `rngpos_adx_delta_filter_long`: REVERTED `85-95:0.0-0.3` → `90-95:0.0-0.3` (drop curve-fit half) + SHORT cell flagged for removal
- [L1339] May 30, 2026 — RETIRED 4 observation-only report surfaces (UI/report only; engine capture left inert)
- [L1362] May 30, 2026 — SHIPPED: Leash Shadow Tracker (observation-only; runner-exit validation infra)
- [L1462] May 30, 2026 — DISABLED Global Volume Filter (A/B test — filter-audit + redundancy hypothesis)
- [L1540] May 29, 2026 (RESET) — fresh batch begins on locked config (fan_ratio both directions live)
- [L1573] May 29, 2026 (later) — LONG fan_ratio filter PROMOTED observation → ACTIVE + exit-strategy deep-dive recorded
- [L1605] May 29, 2026 — EXIT-STRATEGY DEEP DIVE (recorded for later; NOTHING shipped on exits)
- [L1664] May 29, 2026 — SHIPPED: EMA Fan Acceleration (fan_ratio) dead-zone filter (SHORT active / LONG observation-only) + full batch analysis
- [L1791] May 28, 2026 — A/B RE-OPENED: Pair ADX Direction back to `both` — RUN TO COMPLETION (N≥15), no early call
- [L1843] May 28, 2026 — A/B RESULT: Pair ADX Direction REVERTED `both` → `rising` (falling-ADX bled both sides)
- [L1882] May 28, 2026 — A/B TEST: Pair ADX Direction filter relaxed `rising` → `both` (LONG + SHORT)
- [L1945] May 28, 2026 — LOCKED NEXT-BATCH ANALYSIS PLAN: filter remaining LONG + SHORT losers (4 dimensions)
- [L2067] May 27, 2026 (evening) — SHIPPED: Time-to-L1 Protection Tracker (observation-only, NO engine hook)
- [L2141] May 27, 2026 (afternoon, follow-up) — REFINEMENT: BTC RSI 65-70 LONG block replaced with A3 conditional (BTC ATR < 0.10)
- [L2227] May 27, 2026 (afternoon) — SHIPPED: 8-change defensive stack after disaster batch (-$1,001)
- [L2354] May 26, 2026 (very late evening) — SHIPPED: Pattern C / W Combination Trackers (multi-pattern combos surfaced)
- [L2443] May 26, 2026 (very late evening) — BUG FIX: Pattern Cell rule with baseline mults didn't block other dimensional multipliers
- [L2525] May 26, 2026 (late evening) — SHIPPED: BTC 1h × BTC 5m RSI Direction Cross-Filter (SHORT RR blocked)
- [L2590] May 26, 2026 (late evening) — WATCHLIST: BTC 1h × BTC 30m RSI Direction Cross-Tab
- [back-filled 2026-06-02; commit `8a8f8ba`] May 26, 2026 (morning, 09:27 -0300) — SHIPPED: disabled FAST_EXIT L1 + L2 (`fast_exit_enabled`/`fast_exit_l2_enabled` → false) and removed all `fixed_tp_pct`/`fixed_sl_pct` from pattern-cell rules. Exit thesis became runner-trail + trailing only. (Originally unlogged; reconstructed from git when CURRENT_STATE drift was caught — CURRENT_STATE had still listed FAST_EXIT L1 as ON.)
- [L2919] May 25, 2026 (late evening, post-FE-floor) — FE ATR floor caps shipped (L1: 0.60%, L2: 0.80%)
- [L3028] May 25, 2026 (late evening) — Triple ship: FE ATR floors + Market Breadth disabled + SHORT Bear%≥85 watchlist
- [L3204] May 25, 2026 (later evening) — BUG FIX v4: cumulative runtime, not per-session started_at
- [L3270] May 25, 2026 (later evening) — BUG FIX v3: BNB burn rate denominator must be BOT UPTIME, not "time since oldest trade"
- [L3334] May 25, 2026 (evening) — BUG FIX: BNB burn rate `max(1.0, ...)` floor inflated post-restart, triggered phantom EMERGENCY swap
- [L3476] May 25, 2026 (late afternoon) — ROLLED BACK same-day disable of PAIR_EXT_MIN + PAIR_EMA20_SLOPE_MIN
- [L3547] May 25, 2026 (afternoon) — DISABLED Pair Extension floor + zero'd Pair EMA20 Slope Min SHORT (redundancy audit)
- [L3624] May 25, 2026 — 🚨 CRITICAL LESSON: Filter compound-effect blind spot — ROLLBACK of 2 over-aggressive cross-filter rules
- [L3712] May 25, 2026 — SHIPPED: `global_volume_rescue_max_long: 0.60` (rescue MAX ceiling)
- [L3810] May 25, 2026 (post-reset 1-trade) — VVVUSDT LONG -$107 forensic + structural correction
- [L4049] May 24, 2026 (latest evening) — ⚠️ MANDATORY WATCHLIST: btc_1h_slope_max_long: 0.15 + btc_1h_slope_max_short: 0.10 — OVER-BLOCK RISK
- [L4148] May 24, 2026 (very late evening, post-methodology lock) — PHASE 1 STRUCTURAL SHIP: 3 filters + 4 multipliers from full-pool baseline
- [L4262] May 24, 2026 (very late evening) — LOCKED METHODOLOGY: Full-pool structural cell analysis (replaces reactive batch analysis)
- [L4403] May 24, 2026 (late evening) — `btc_rsi_adx_filter_short` tightened: `0-30:0-30` → `0-30:25-30` (ADX MIN floor)
- [L4519] May 24, 2026 (late evening) — W4 SHORT demoted 2.0× → 1.0× (BTC SHORT loss + structural watch)
- [L4607] May 24, 2026 (late evening) — WATCHLIST: W6 SHORT lev-stack candidate (re-evaluate next batch)
- [L4733] May 24, 2026 (late evening) — `btc_1h_slope_max_short: 0.10` SHIPPED + SHORT semantics fix
- [L4841] May 24, 2026 (late afternoon) — Extension Multiplier dimension SHIPPED (L1b + L2a + L2b at 2.0× investment)
- [L4976] May 24, 2026 (afternoon) — W4 LONG fixed TP+SL shipped (treatment-decoupled W cell)
- [L5090] May 24, 2026 (afternoon) — WATCHLIST: 6-pair blacklist candidates (SUI, TAO, TON, COS, PLAY, ONDO)
- [L5182] May 24, 2026 (afternoon) — WATCHLIST: SHORT-only BTC Trend Filter — cross-batch DEFERRED
- [L5311] May 24, 2026 (early morning, post-W6-demote) — WATCHLIST: W6 LONG sub-cell refinements
- [L5414] May 24, 2026 (early morning) — W6 LONG multiplier demoted 2.0× → 1.0× (✗ HARMFUL gate triggered)
- [L5532] May 24, 2026 (early morning, post-analysis) — ROLLBACK: C2 SHORT defensive rule removed pending deeper analysis
- [L5632] May 24, 2026 (early morning) — Pattern Cell Ship rule: C2 SHORT defensive (TP +0.10 / SL -0.50, baseline sizing)
- [L5797] May 23, 2026 (late evening) — Pattern Cell lookup: Option D strict-C-blocks-W (defang W mults on bare C match)
- [L5911] May 23, 2026 (same-day refinement) — `btc_rsi_adx_filter_long` 65-70 rule: `:40` → `:0-35` (surgical: preserve winner sub-zone)
- [L6002] May 23, 2026 — `btc_rsi_adx_filter_long`: tighten BTC RSI 65-70 AND 70-100 from `:30/:35` → `:40` (climax-buying block)
- [L6103] May 23, 2026 — WATCHLIST: FAST_EXIT + PATTERN_FIXED_TP ATR scaling (defer until cross-batch data)
- [L6255] May 23, 2026 — Post-Exit Regime-Flip diagnostic (RegFlipMin / RegFlip P&L columns)
- [L6360] May 23, 2026 — `sl_atr_widen_floor_pct: -1.20` shipped (ATR-SL cap on extreme-volatility pairs)
- [L6474] May 23, 2026 — Trailing Confirmation Performance: 3 new ATR-trailing diagnostic columns
- [L6512] May 23, 2026 — `trailing_atr_multiplier: 0.30 → 0.50` shipped (analog of sl_atr_multiplier)
- [L6927] May 22, 2026 — BTC ATR × BTC ADX 2D Cross-Filter shipped (SHORT-only `0.0-0.10:30-999`)
- [L6990] May 22, 2026 — BTC RSI 60-65 LONG cross-filter tightened (replaced `0-30` with `22-25` + `27-30`)
- [L7068] May 22, 2026 — Shipped `entry_dist_from_ema13_min_long: 0.20` (Pair Extension floor for LONGs)
- [L7156] May 22, 2026 — Option A: removed `fixed_tp_pct` from C4 LONG + UNMATCHED LONG (kept SL caps)
- [L7231] May 21, 2026 (very late evening) — BUG FIX: `_lookup_pattern_cell_rule` Option C fall-through
- [L7287] May 21, 2026 (very late evening) — LONG ema_gap_threshold_long: 0.06 → 0.04 (mark for review)
- [L7325] May 21, 2026 (very late evening, post-rollback) — C1 SHORT lev-stacked to 3.0× effective
- [L7356] May 21, 2026 (very late evening) — Full rollback of all 4 BTC RSI × BTC ADX loosenings
- [L7393] May 14, 2026 (evening) — BTC 1h Slope Analytics watchlist (locked validation gates, NO filters shipped)
- [L7564] May 14, 2026 (late PM) — Phantom BE 0.20/0.05 counterfactual tracker (NEW, observation-only)
- [L7632] May 14, 2026 (PM) — BTC 1h Slope dimension (NEW, observation-only — higher-TF macro context)
- [L7747] May 14, 2026 — BTC Market Extension / BTC Late Regime Risk (NEW, observation-only — macro counterpart of pair extension)
- [L7848] May 13, 2026 (PM) — Entry Extension / Late Entry Risk dimension (NEW, observation-only)
- [L7959] May 13, 2026 (LATE PM) — Multiplier re-balance based on 602-trade cross-pool analysis
- [L8036] May 13, 2026 — Observation watchlist (filters NOT shipped, pending fresh data)
- [L8114] May 12, 2026 UTC-3 (LATE PM) — Watchlist: BCHUSDT + TRUMPUSDT + BTC slope signed-bucket finding
- [L8206] May 12, 2026 UTC-3 (LATE PM) — STRATEGIC IDEA: Decouple WR from $/trade via lower TP + multiplier compensation
- [L8308] May 12, 2026 UTC-3 (LATE PM) — Watchlist: BUSDT + TAOUSDT (held below blacklist gate)
- [L8401] May 12, 2026 UTC-3 (LATE PM, last commit) — SKYAIUSDT blacklisted (override of strict gate)
- [L8460] May 12, 2026 UTC-3 (LATE PM) — ATR aggregate filter REJECTED + ADAUSDT blacklisted (per-pair concentration check)
- [L8585] May 12, 2026 UTC-3 (LATE PM) — Post-exit time-bucketed snapshots methodology
- [L8700] May 12, 2026 UTC-3 (LATE PM) — Watchlist: SL Wide tightening -0.90% → -0.85%
- [L8821] May 12, 2026 UTC-3 (LATE PM) — SHORT Range Position min filter shipped (RP<2% block)
- [L8928] May 12, 2026 UTC-3 (PM) — Pair-level multiplier cell removed + LINK/ICP/BNB blacklist + Range Position table refactor
- [L9083] May 12, 2026 UTC-3 — `ema_gap_5_20_max_long: 0.80 → 0.60` (asymmetric cap, cross-batch validated)
- [L9188] Trading Strategy Analysis Context (188 trades, March 2026)
- [L9263] April 11, 2026 — DB-Loss Incident & AWS Hardening
- [L9338] April 14, 2026 — Locked Baseline for 100-Trade Fine-Tuning Sample
- [L9601] April 16, 2026 — SUIUSDT Reconciler Race Guard (EXTERNAL_CLOSE mislabeling bug)
- [L9659] April 17, 2026 — Broker-Side Protective Stops: REMOVED after failed rollout
- [L9713] April 17, 2026 — Broker-Side Protective Stops (OLD — original design, kept for reference only)
- [L9818] April 17, 2026 — Phase 1c Amendment (81-trade sample analysis + filter tightening)
- [L10030] April 18, 2026 — Phase 1c amendment #5 (33-trade fresh data) — SHORT overhaul
- [L10260] April 28, 2026 — Phase 1c-Explore (sub-phase) — Loosen restrictions for ablation testing
- [L10366] April 28, 2026 — Exploration Analytics (Tier 1 indicators added, observation-only)
- [L10516] April 28, 2026 — LOCKED Phase 1c-Explore Plan (200-trade frozen exploration batch)
- [L10912] April 29, 2026 — Peak/Trough P&L Invariant Bug + Option A Fix (forward guard + diagnostic logs)
- [L11010] April 30, 2026 — BTC RSI Re-Validation Filter Mismatch Bug (Phase 1c-Explore data partially contaminated)
- [L11140] April 30, 2026 — Winner Exit Optimization Plan (200-trade counterfactual analysis)
- [L11314] May 1, 2026 — BE Layer Introduction Plan (sister analysis to Winner Exit)
- [L11518] May 2, 2026 — Reporting granularity expansion (no strategy changes)
- [L11621] May 2, 2026 — SIGNAL_EXPIRED enrichment (Aborted entries become first-class analytical population)
- [L11755] May 2, 2026 — Phase 1d-ExitTest plan (RSI handoff at high TP levels — code shipped INERT)
- [L11892] May 2, 2026 — Three new max-guard filters (split + new), feature ships permissive
- [L12150] May 3, 2026 — Cross-sample SHORT findings to validate at 200-trade Phase 1c-Explore checkpoint
- [L12232] May 3, 2026 — Pair blacklist candidates for 200-trade Phase 1c-Explore review
- [L12270] May 3, 2026 — Phase 3 Position Multiplier Mechanism (DESIGN, post-200-trade bonus)
- [L12397] May 3, 2026 — Decision to revert Amendments #6 and #8 (40s→20s timeout, 2→1 tick offset) at 200-trade checkpoint
- [L12478] May 4, 2026 — Phase 1c-Explore 224-trade checkpoint analysis & LONG-side config changes
- [L12586] May 4, 2026 — Phase 3 Position Multiplier (IMPLEMENTED, infrastructure + initial LONG cells)
- [L12698] May 4, 2026 — RSI Handoff activated for LONG L3+ (against this-batch counterfactual)
- [L12777] May 4, 2026 — Phase 1c-Explore SHORT-side analysis & config changes (224-trade checkpoint, SHORT subset)
- [L12868] May 4, 2026 — SHORT Premium Multiplier cells activated (4 cells at 2.0×)
- [L12939] May 4, 2026 — Exploration Analytics section REMOVED
- [L12995] May 4, 2026 — LOCKED next-batch validation plan (reference baseline + revert criteria)
- [L13088] May 4, 2026 — Toggle for signal re-validation at maker timeout (`revalidate_on_taker_fallback`)
- [L13135] May 4, 2026 — Pair blacklist additions: HYPEUSDT + ASTERUSDT
- [L13181] May 4, 2026 — Multiplier Cell Performance: Δ vs BL redesigned to dollar terms
- [L13237] May 5, 2026 — S-P2 promoted to HARD BLOCK + S-B1 activated (`btc_rsi_adx_filter_short: "30-35:30,35-40:20"`)
- [L13299] May 5, 2026 — CRITICAL BUG FIX: BTC RSI × BTC ADX Cross-Filter was dead code
- [L13364] May 5, 2026 — Cross-Filter syntax extension: range-form (block when ADX > X)
- [L13417] May 5, 2026 — Watchlist: LONG BTC RSI 65-70 × BTC ADX 35+
- [L13464] May 5, 2026 — Return Multiple bug fix (paper mode): immutable initial baseline + BNB inclusion
- [L13527] May 5, 2026 — Return Multiple paper-mode fix v2: switched to reverse-derive (corrects the v1 backfill bug)
- [L13602] May 5, 2026 — `btc_adx_max_long: 40 → 35` (HARD BLOCK on LONG BTC ADX 35+, 4-sample structural)
- [L13659] May 5, 2026 — Fresh start: pre-reset batch archived, new batch begins on locked config
- [L13798] May 5, 2026 — Regime Stability Instrumentation (REVERTED same day)
- [L13935] May 5, 2026 — BTC Trend Filter (EMA20 vs EMA50, ~4h macro context)
- [L14016] May 5, 2026 — Filter-rollback candidates locked for next-batch validation
- [L14127] May 5, 2026 — Filter Block counter instrumentation (Option B shipped)
- [L14244] May 5, 2026 — Alpha-subtype pre-filter (auto-blacklist Binance launchpad tier)
- [L14419] May 5, 2026 — Pair EMA20-EMA50 Gap at Entry (`entry_pair_ema20_ema50_gap_pct`) — observation-only
- [L14482] May 5, 2026 — RSI Handoff level changed L3 → L2, RSI Handoff Performance table added
- [L14523] May 5, 2026 (evening) — `adx_dir_long/short: rising → both` + bot reset (final pre-batch change)
- [L14586] May 6, 2026 — `btc_adx_min_long: 18 → 15` (USER-DIRECTED override of IRON RULE)
- [L14622] May 6, 2026 — `rsi_handoff_level: 2 → 3` (live data + corrected historical math)
- [L14660] May 6, 2026 (afternoon) — Major repositioning: 6 simultaneous config changes (user-directed)
- [L14704] May 6, 2026 (evening) — BTC Trend Filter + Pair Trend Gap switched EMA20 → EMA13
- [L14804] May 7, 2026 — Realtime-close cache→DB sync bug (peak/low undercount on realtime-fired exits)
- [L14933] May 7, 2026 — Disabled redundant PAIR_RSI_MOMENTUM filter
- [L14998] May 7, 2026 — Loosened ADX max caps (LONG 25→30, SHORT 33→40)
- [L15044] May 7, 2026 — Pair Trend Filter shipped (pair-level analog of BTC Trend Filter)
- [L15096] May 7, 2026 (evening) — Reset #3 of week, locked config snapshot
- [L15155] May 9, 2026 — BTC RSI × BTC ADX cross-filter additions + SHORT watchlist
- [L15193] May 9, 2026 — EMA5 Stretch < 0.16% LONG = strongest cross-sample loser zone (filter shipped)
- [L15287] May 9, 2026 (afternoon) — SHORT-side EMA5 Stretch watchlist
- [L15346] May 9, 2026 (evening) — Trailing pullback confirmation timer (15s default)
- [L15427] May 9, 2026 (late evening) — Watchlist items + Trailing Confirmation TP-level breakdown
- [L15497] May 9, 2026 — `btc_adx_max_long: 40 → 35` (revert; LONG-only; honest cross-sample framing)
- [L15568] May 9, 2026 — SHORT-only BTC Trend Filter (watchlist for next ≥30-SHORT batch)
- [L15642] May 10, 2026 — `min_adx_delta_long/short: 0.10` filter shipped (cross-sample validated)
- [L15712] May 10, 2026 — Global Volume Filter shipped LONG-only at 0.95 (3-sample cross-sample validated)
- [L15795] May 10, 2026 (evening) — Volume Filter Intersection Rescue Clause
- [L15895] May 11, 2026 — Deep review: SHORT GlobalVol cliff at 1.10 + methodological correction on BTC RSI 30-35 × BTC ADX 30-35
- [L16020] May 11, 2026 — Loss-Cleanup Filter Watchlist (full cross-batch landscape)
- [L16193] May 11, 2026 — Addendum to Loss-Cleanup Watchlist: SHORT `adx_strong` revert candidate
- [L16266] May 11, 2026 — LONG-side filter+multiplier shipped (BTC ADX 18 revert, ADX Δ × BTC ADX cross-filter, multipliers neutralized)
- [L16411] May 11, 2026 — SHORT Multi-Axis GlobalVol Filter with BTC Capitulation Override
- [L16530] May 11, 2026 — PAIR_60-65 LONG multipliers RE-ACTIVATED at 2.0× (filter-overlap evidence)
- [L16628] May 11, 2026 — ADX Δ × BTC ADX Cross-Tab — cross-batch pool findings (May 4 → tonight) + watchlist
- [L16727] May 11, 2026 UTC-3 — Phantom Regime Change Exit shadow tracking (observation-only counterfactual)
- [L16784] May 11, 2026 UTC-3 — ADX Δ × BTC ADX filter extended: 18-25 → 18-30
- [L16818] May 11, 2026 UTC-3 — Block LONG BTC RSI 60-65 × BTC ADX 25-30
- [L16874] May 11, 2026 UTC-3 — Block LONG BTC RSI 55-60 × BTC ADX 25-30 (locked watchlist gate fired)
- [L16939] May 11, 2026 UTC-3 — Cross-batch CSV dedup methodology (locked)
- [L17016] May 11, 2026 UTC-3 — Block SHORT BTC RSI <30 × BTC ADX > 30 (cross-batch loss zone)
- [L17097] May 11, 2026 UTC-3 — `btc_adx_min_short: 18 → 20` (user-directed override of locked gate)
- [L17168] May 11/12, 2026 UTC-3 — End-of-night SHORT batch review + SUIUSDT-style watchlist
- [L17242] May 12, 2026 UTC-3 — `momentum_ema20_slope_min_short: 0.04 → 0.06` (full-history validated)
- [L17329] May 15, 2026 PM — BTC Volatility Regime + BTC 1h RSI Direction (observation-only)
- [L17408] May 15, 2026 PM — Entry Quality Score ≤ 1 watchlist (DO NOT ship yet)
- [L17504] May 15, 2026 PM — Analytical baseline convention (May 14 onwards)
- [L17544] May 16, 2026 — Observation: SHORT BTC 1h Slope × BTC ADX cell structure + BTC Volatility candidate confound (analyze later)
- [L17590] May 16, 2026 — Pre-BE-activation baseline (locked at commit `1aad9e6`)
- [L17683] May 16, 2026 — Watchlist: Entry Quality Score 3 SHORT as multiplier candidate
- [L17774] May 16, 2026 — Watchlist: 3 SHORT multiplier candidates (1-sample, locked gates)
- [L17867] May 16, 2026 — Partition timestamps for next-checkpoint analysis (NO RESET decision)
- [L17953] May 16, 2026 PM — Structural framework: 3-pattern failure taxonomy + BE-compatibility rule
- [L18066] May 16, 2026 PM — Watchlist WL-D: BTC-Gap-Floor SHORT filter (locked gates)
- [L18179] May 16, 2026 (19:22 UTC-3) — `tp_min: 0.50 → 0.80` shipped (SHORT-side Post-Exit Regret driven)
- [L18266] May 17, 2026 (21:12 UTC-3) — Post-arm-min instrumentation + BE Floor Counterfactual table
- [L18397] May 17, 2026 UTC-3 — Entry Quality Score filter disabled (test under new BE 0.05 floor)
- [L18459] May 18, 2026 UTC-3 — Next-batch BE floor decision: 0.05 → 0.10
- [L18541] May 18, 2026 UTC-3 — NEXT-BATCH DECISION CHECKLIST (consolidated, locked)
- [L19025] May 18, 2026 UTC-3 — Volume + ADX Δ filters DISABLED for A/B test (locked decision pending next batch)
- [L19138] May 18, 2026 UTC-3 — Methodological lesson: proxy fallbacks corrupt gate-checks silently
- [L19238] May 18, 2026 (PM) — `btc_adx_max_long: 35 → 40` (symmetric with SHORT)
- [L19310] May 18, 2026 PM — `rngpos_adx_delta_filter_short: "5-10:1.0-2.0"` shipped (new 2D primitive)
- [L19424] May 18, 2026 PM (FINAL BATCH) — Multi-ship session: exit stack + 3 LONG filters + 2 mult demotions
- [L19581] May 18, 2026 (late PM) — `btc_rsi_adx_filter_long` rule `60-65:0-25 → 60-65:0-30` (loosen)
- [L19643] May 18, 2026 (late PM) — Entry Quality Score multiplier shipped (NEW dimension, 3 cells at 2.0×)
- [L19757] May 18, 2026 (late PM) — BTC RSI 55-60 LONG cap rollback `99-100 → 20-25`
- [L19823] May 19, 2026 — `global_volume_threshold_short: 0.0 → 0.50` (NEW MIN-side SHORT filter)
- [L19921] May 19, 2026 — `rngpos_adx_delta_filter_long: "90-95:0.0-0.3"` (NEW LONG rule, small-N override)
- [L19981] May 19, 2026 — 2 multiplier cells demoted 2.0× → 1.0× (✗ HARMFUL verdict applied)
- [L20067] May 19, 2026 — BTC Gap × BTC ADX 2D Cross-Filter shipped + cross-tab re-bucketed to 24 fine bins
- [L20170] May 19, 2026 (late) — New LONG multiplier shipped: BTC RSI 60-65 × BTC ADX 22-25 at 2.0×
- [L20273] May 19, 2026 (evening) — FAST_EXIT L2 shipped (0.40% / 5min slow-climber tier)
- [L20377] May 19, 2026 (late) — Pattern C Tracker shipped (4 signatures × 2 directions, observation-only)
- [L20545] May 19, 2026 (late PM) — `btc_adx_min_short: 20 → 18` (user-directed override)
- [L20625] May 19, 2026 (late PM) — Phantom Regime Change Exit CF analytics shipped (analytics surface for May 11 capture)
- [L20706] May 19, 2026 (late PM) — Phantom BE floor: 0.05 → 0.10 (table renamed 0.20/0.10)
- [L20763] May 19, 2026 (late PM) — Pattern C Tracker extended with C5 + C6 (LONG + SHORT, observation-only)
- [L20877] May 20, 2026 — BE 0.20/0.10 RE-ACTIVATED (cross-batch validated)
- [L21024] May 20, 2026 — SL tightened -0.80% → -0.70% (BE-active regime change)
- [L21135] May 20, 2026 — METHODOLOGY LESSON: counterfactual analysis must respect the active exit stack
- [L21251] May 20, 2026 — Pattern C Tracker extended with C7 — Pair Countertrend Bounce
- [L21387] May 20, 2026 — BUG FIX: Phantom Regime Change Exit cache preservation (same class as May 15 phantom_be_aggr bug)
- [L21544] May 20, 2026 (late PM) — Pattern C C8 shipped: Oversold/Overbought Chop (observation-only)
- [L21656] May 20, 2026 (late evening) — Pattern C Tracker: TP counterfactual columns + LOCKED next-batch decision matrix
- [L21841] May 20, 2026 (latest evening) — Pattern C tracker: 3 enhancements + C9 ship
- [L21977] May 20, 2026 (latest+1 evening) — Pattern C framework SYMMETRIC extension: MULTIPLIER CANDIDATE verdict
- [L22154] May 20, 2026 (latest+2 evening) — Pattern W shipped + Score 3 SHORT demoted
- [L22355] May 20, 2026 (latest+3 evening) — Pattern W symmetric extension: 4 enhancements
- [L22483] May 20, 2026 (latest+4 evening) — BUG FIX: verdict logic — "★ Winners cohort" required only WR, ignored P&L sign
- [L22592] May 20, 2026 (latest+5 evening) — Three refinements: R:R column, Loser % in W, MULT gate threshold
- [L22719] May 20, 2026 (latest+6) — Pattern C: SL 0.50 + SL 0.60 counterfactual columns shipped
- [L22815] May 20, 2026 (latest+7) — Pattern C: drop TP 0.05 + add combined TP 0.10 + SL 0.50 column
- [L22878] May 20, 2026 (latest+8) — Pattern C & Pattern W: per-row Batch P&L projection columns
- [L22949] May 20, 2026 (latest+9) — Pattern Calculator widget (combined C + W simulator)
- [L23060] May 20, 2026 (latest+10) — Pattern Calculator: drop 1.5× option, add 2.0× multiplier mode to Pattern C
- [L23108] May 20, 2026 (latest+11) — Pattern Calculator: C effect breakdown by cap + NEITHER baseline P&L
- [L23163] May 20, 2026 (latest+12) — Pattern Calculator: caps + multiplier are independent on Pattern C
- [L23230] May 20, 2026 (latest+13) — Pattern Calculator: mult effect computed on OG transaction (decomposition fix)
- [L23308] May 20, 2026 (latest+14) — Pattern Calculator: REVERT latest+13 decomposition (mult applies to new exit, not OG)
- [L23375] May 20, 2026 (latest+15) — Pattern Calculator: sub-split mult-extra by cap-fire type (diagnostic)
- [L23443] May 20, 2026 (latest+16) — Pattern Calculator: Pattern W gets symmetric caps + mult controls
- [L23510] May 20, 2026 (latest+17) — Pattern Calculator: remove fee adjustment from multiplier math (align with Pattern W table)
- [L23563] May 20, 2026 (latest+18) — Pattern Calculator: Unmatched Losers / Unmatched Winners pseudo-cohorts
- [L23639] May 21, 2026 — Cross-batch Pattern Calculator finding (522 trades May 4+)
- [L23751] May 21, 2026 (deep dive) — Cross-batch Unm. L is inflated; filter-overlap analysis shows minimal revert gain
- [L23826] May 21, 2026 — Filter reverts shipped (4 filters relaxed to expand Unm. L cohort)
- [L23937] May 21, 2026 (02:31 UTC) — Pre-revert baseline snapshot LOCKED
- [L24011] May 21, 2026 — Unmatched Winners (Unm. W) cohort: cross-batch finding
- [L24094] May 21, 2026 — VALIDATED corrected ship: "caps for losers, mult for winners"
- [L24201] May 21, 2026 — Improved ship: disable BE on Pattern W cohort + 2× mult
- [L24328] May 21, 2026 (revised) — CORRECTED forward ship picture (caps DO matter on non-W cohort)
- [L24460] May 21, 2026 — Pattern Calculator: locked working configuration (Δ +$650.53 on 42-trade batch)
- [L24600] May 21, 2026 (late PM) — REJECTED: SHORT W1 HighConv trend at 2.0× multiplier (cross-batch falsified)
- [L24689] May 21, 2026 (late PM) — Score-based multiplier dimension REMOVED entirely
- [L24799] May 21, 2026 (late PM) — Premium Multiplier: "Both (Invest + Lev)" mode shipped
- [L24963] May 21, 2026 (late evening) — Pattern Cell Ship Rules: Phase 1 (engine backend) SHIPPED
- [L25139] May 21, 2026 (late evening) — Pattern Cell Ship Rules Phase 2 (UI + reporting) SHIPPED
- [L25263] May 21, 2026 (late evening, post-Phase-2) — W6 PATTERN SHIPPED — Unmatched-cohort deep dive
- [L25412] May 21, 2026 (very late evening) — 4-Cohort Coverage table + treatment-type de-coupling
- [L25521] May 21, 2026 (evening) — UNMATCHED pattern ship: TP 0.10 / SL -0.50 for trades with no C/W signature
- [L25620] May 21, 2026 (evening, post-UNMATCHED ship) — WATCHLIST: W5 SHORT as ENTRY FILTER candidate (cross-batch anti-pattern)
- [L25735] May 21, 2026 (evening) — Per-Rule Contribution baseline (64-trade batch May 20-21)
- [L25853] May 21, 2026 (evening) — W2 LONG + W4 LONG multipliers demoted 2.0× → 1.0× (SHORT-only)
- [L25909] May 21, 2026 (evening) — WATCHLIST: Stack 1.5× Leverage on validated 2.0× Investment cells
- [L26017] May 21, 2026 (evening, pre-batch-reset) — LEV stacking SHIPPED: BTC_60-65_22-25 LONG to 2.0×inv + 1.5×lev (3.0× effective)
- [L26121] May 21, 2026 (evening, pre-batch-reset) — ADX Δ × BTC ADX Cross-Filter DISABLED (A/B test under new exit stack)
- [L26209] May 21, 2026 (evening) — BTC RSI × BTC ADX Cross-Filter: loosened 2 cells (new-exit-stack A/B)
- [L26304] May 21, 2026 (late evening) — BTC RSI × BTC ADX: 2 more surgical openings (30-35 ADX in 55-60 and 60-65 RSI bands)

---

## NEW ENTRIES (2026-06-02 onward — full text)

### 2026-06-03 — REVERTED `adx_dir_long` rising → both (LONG side was backwards; SHORT keeps rising)

**Change:** `adx_dir_long` "rising" → "both" in trading_config.json. `adx_dir_short` stays "rising".

**Why (error correction):** the Jun-2 ship set both sides to "rising" (block falling-pair-ADX = exhaustion). On the proper **7-batch proxy** (which has correctly-signed `adx_delta`: 59 of 222 falling), the LONG side is **backwards**:
- LONG FALLING-ADX (cut by rising): N=48, **50% WR, Avg +0.002** (breakeven) — the filter was removing these.
- LONG RISING-ADX (kept): N=72, **39% WR, Avg −0.174** — the filter was keeping the actual losers.
- SHORT FALLING (cut): N=11, 45% WR, **−0.239** (loser ✓); SHORT RISING (kept): N=91, **62% WR** ✓ — short side is correct, kept.

**Root cause:** the original LONG ship's "falling-ADX LONG = 1W/9L" evidence came from `dedupe_pool.csv` (full pool), whose `entry_adx_delta` is **broken** — only **6 negative of 558**, vs **59 of 222** in `dedupe_pool_7batches`. The two pool files compute/populate adx_delta inconsistently; the 7-batch file is the trustworthy one. So the LONG ship rested on a bad field. (Methodology flag: earlier cross-pool adx_delta comparisons were apples-to-oranges — trust the 7-batch file for adx_delta.)

**Net:** SHORT-side falling-ADX block retained (evidence-backed, −0.24 losers cut). LONG-side reverted to no-direction-filter. Caveat: 7-pool N=48/72, single pool; the "rising-LONG=loser" split may carry confounds, but the direction clearly contradicts the (broken-data) ship rationale, so revert is the defensible call.

### 2026-06-03 — SHIPPED: `no_trade_pairs` (track-only) mechanism — BTCUSDT visible but non-trading

**Change:** new config field `no_trade_pairs` (comma-separated, top-level, mirrors `pair_blacklist`) + `BTCUSDT` placed in it. **The distinction:**
- `pair_blacklist` → pair removed from the top-pair/volume universe entirely (not subscribed, scanned, or displayed). Applied at universe-fetch (`trading_engine.py:5433`).
- `no_trade_pairs` → pair STAYS in the universe (subscribed, scanned, **shown in Top-Pair-by-Volume**) but every LONG/SHORT signal is forced to NO_TRADE at the per-pair eval (`trading_engine.py:~5811`), counter `PAIR_NO_TRADE`.

**Why:** user wants BTCUSDT visible/tracked on the dashboard for reference but not trading (consistent with the edge<fee evidence — BTC shouldn't open positions, but should stay in view). The blacklist removes it from sight; this keeps it visible while blocking entries. BTC's macro reference (regime/RSI/ADX/slope) was already fetched independently (`get_ohlcv('BTC/USDT:USDT')`), so this is purely about the *tradeable/displayed* universe.

**Full D11:** config.py (`no_trade_pairs: str = ""` + comment), trading_config.json (`"BTCUSDT"`), engine (per-pair entry block + PAIR_NO_TRADE counter), main.py ConfigUpdate field, UI (text input + helper text + load/save + summary line). Generic config-merge apply (model_dump) — no special handler needed, mirrors pair_blacklist. Verified: py+json syntax OK, input IDs wired (input=1, load+save). Could not runtime-test locally (pydantic_settings is deploy-only) — verify PAIR_NO_TRADE fires on BTC post-deploy.

**State note:** BTCUSDT is now in `no_trade_pairs`, NOT `pair_blacklist`, and NOT freely tradeable — a distinct third state. Supersedes the same-day "whitelisted BTC" entry (BTC is now track-only, not tradeable).

### 2026-06-03 — WHITELISTED BTCUSDT (user override; removed from blacklist same day it was added)

**Change:** removed `BTCUSDT` from `pair_blacklist` (now 10 pairs). Reverses the BTC blacklist shipped earlier today.

**Discipline note (override, flagged):** This is **against the evidence.** BTC was the single best-evidenced structural loser: N=27, 26% WR, −0.208% avg, −5.62% total, with the deciding mechanism being edge < fee (avg favorable excursion ~0.19% vs 0.077% roundtrip fee → no extractable edge after costs). Tight-exit and lower-TP simulations both failed to make it profitable. Shipped per explicit user direction as a deliberate override. Mitigants: (1) current filters block ~83% of BTC trades historically, so it should rarely fire; (2) the per-pair revert gate applies — re-blacklist at ≤35% WR on N≥10 fresh, which BTC's 26% historical WR is near-certain to trip if it trades meaningfully. **Watch BTC per-pair WR closely; expect to re-blacklist.**

**Blacklist now (10):** BNBUSDT, ENAUSDT, FILUSDT, MUSDT, RAVEUSDT, TRUMPUSDT, VVVUSDT, XAGUSDT, XAUUSDT, ZECUSDT. **Released total (13):** the 12 thin-N pairs + BTCUSDT.

### 2026-06-03 — TRIMMED pair_blacklist 23 → 11 (released 12 thin-evidence pairs for forward re-test)

**Change:** `pair_blacklist` reduced from 23 to 11 pairs.

**Why:** Audit of the 23 blacklisted pairs found only **5 had solid evidence** (N≥15 losers). The blacklist was built on **raw pre-filter performance, never re-simulated under the current 15+ filter stack** — and blacklisted pairs have NO post-blacklist data (they stopped trading), so the thin-evidence entries can't be validated from history. Twelve were blacklisted on N<15 (violating the locked "never ship from <N=15" discipline) — e.g. **LABUSDT on N=1**, **ADAUSDT at −0.033% (≈breakeven)**. The only honest way to re-evaluate is to release them and observe forward, now that the current filters gate entries.

**KEPT BLACKLISTED (11):**
- Evidenced losers (N≥15): `BTCUSDT` (27/26%/−5.62), `FILUSDT` (31/55%/−3.36), `VVVUSDT` (17/41%/−4.40), `BNBUSDT` (16/38%/−2.20), `TRUMPUSDT` (15/33%/−3.83).
- Commodities (different asset class, intentional): `XAGUSDT`, `XAUUSDT`.
- No-data / preemptive (new-listing/illiquid, likely also covered by new_listing/alpha_subtype filters): `ENAUSDT`, `MUSDT`, `RAVEUSDT`, `ZECUSDT`.

**RELEASED / WHITELISTED (12) — now tradeable again, under current filters:**
`ADAUSDT, ASTERUSDT, BCHUSDT, DOGEUSDT, ERAUSDT, HYPEUSDT, ICPUSDT, LABUSDT, LINKUSDT, PUMPUSDT, SKYAIUSDT, WLFIUSDT`. (All had N<15 pre-filter; full table: ADA 12/42%/−0.033, HYPE 13/38%/−0.173, LINK 13/31%/−0.175, WLFI 8/38%/−0.431, DOGE 6/0%/−0.743, BCH 6/33%/−0.250, ICP 6/17%/−0.421, PUMP 6/17%/−0.307, SKYAI 6/33%/−0.285, ASTER 4/25%/−0.104, ERA 3/33%/−0.304, LAB 1/0%/−1.192.)

**LOCKED REVERT GATE (per released pair):** re-blacklist any released pair that shows **≤35% WR on N≥10 fresh** (current-stack) trades. Track per-pair WR as fresh trades accumulate. This converts 12 overfit pre-filter blacklists into a controlled forward re-test.

**Caveat acknowledged:** releasing adds pairs whose *historical* (pre-filter) numbers look poor (DOGE 0%, ICP 17%, LAB −1.19) — but on N=1–6, those are noise, and the current filter stack should gate the bad entries. The revert gate is the safety net. This is a deliberate "trust the filters + re-test" move, not an assertion these pairs are good.

### 2026-06-03 — BLACKLISTED BTCUSDT (added to `pair_blacklist`)

**Change:** added `BTCUSDT` to `trading_config.json` `pair_blacklist` (now 23 pairs).

**Rationale (structural, not just a bad streak):** Across all configs (27 unique BTC trades, May 4–Jun 3): **26% WR, −0.208% avg, −5.62% total.** BTC LONG 17% WR (N=18), BTC SHORT 44% (N=9). The mechanism is the deciding factor: BTC's average favorable excursion is only **+0.19%**, barely above the **0.077% roundtrip fee** — there's essentially no extractable edge after costs. Expectancy decomposition (whole book) showed gross edge ≈ 0 and fees the deterministic drag; for BTC specifically the move size is too small to clear fees even with tight exits (simulated tp 0.08/SL −0.35 raised WR 28→56% but Σ stayed −3.77, because net win ≈ +0.01 after fee). Lower-TP "rescue" rejected: only 4/20 BTC losers are never-positive, but the 15 that go green peak at just +0.14% avg — a 0.15%-gross TP nets ~+0.08 and catches only 6, leaving it deeply negative. **BNBUSDT (BTC's only true low-vol peer, ATR 0.12) was already blacklisted** — this completes the low-vol-major set. BTC was already ~83% filter-blocked, so live impact is small.

**Scope note:** blacklisting BTCUSDT as a *tradeable pair* is independent of BTC as the *macro reference* — the engine computes BTC regime/RSI/ADX/EMA-slope separately, so all BTC-macro filters keep working. (Worth a one-time eyeball post-deploy, but the code paths are distinct.)

**Not blacklisted (considered, held):** SOLUSDT (36 trades, 31% WR, −7.58% — worse total than BTC) is a *different* problem — it has genuine runner edge (10 trades peaked ≥0.45%, 9 won) but 44% of its losers are never-positive (bad entries, likely LONG-side). A lower TP would *harm* SOL (caps its runners). Decision deferred pending a SOL LONG-vs-SHORT split — keep-short-only / entry-filter / blacklist TBD. Do NOT reflexively blacklist on pooled old-config data.

### 2026-06-03 — SHIPPED: BTC-Acceleration Chase Filter (STATEFUL evolution filter, LONG only)

**Change:** new STATEFUL entry filter. Blocks a LONG when the live BTC EMA20 slope (`_btc_ema20_slope_pct`) is **higher** than it was at the most recent LONG that actually opened within `evo_chase_window_min` (30) minutes = BTC has accelerated since the last entry = chasing a maturing move (late). First stateful filter in the stack: engine tracks `self._last_long_open_ts` + `self._last_long_open_btc_ema20_slope`, updated in `open_position()` on every LONG that opens (blocked LONGs never reach there, so the reference stays the last REAL entry; the 30-min window auto-expires a stale reference). Config: `evo_chase_filter_long_enabled=true`, `evo_chase_filter_short_enabled=false` (untested side, plumbed-off), `evo_chase_window_min=30`. Counter `BTC_ACCEL_CHASE_LONG`. Full D11: config.py, trading_config.json, engine (state init + filter check at ~6472 + open-hook at end of open_position), UI (toggle + window input + load/save + summary).

**Evidence (7-batch proxy = current-config; full pool = directional check only — older configs):**
- "BTC EMA20 slope improving vs last LONG (30min)" block cohort: **7-batch N=26, 30.8% WR, Σ-3.1% (net-losing); full pool N=83, 48.2%** (directional). Mechanism = mean-reversion: chasing BTC as it accelerates past your last entry = late, lagging-alt entry into the tail of a BTC thrust.
- Live confirmation: caught the **06-03 4-loss cluster 0/4** (blocked all 4 losers, kept both winners). On the **06-02 original 4-loss event** it caught 2/4 (plateau misses SUI/DOT — see method notes). So ~50-100% cluster coverage, ZERO winners cut in those batches.
- The signal is the user's "evolution vs last trade" idea, validated; it INVERTS the naive intuition (BTC "better" → worse) because it measures *chasing*, not absolute conditions.

**Window = 30 is load-bearing (do NOT shorten):** at 10/15min the block cohort is NET-POSITIVE (Σ +0.3/+0.5%) — it contains fat-tail winners, so blocking it HURTS expectancy despite improving WR (the inverse of the high-WR-net-losing trap). 30min is where the cohort flips net-negative (-3.1%). 60min slightly stronger (29%) but 30 is the locked choice.

**Method notes (tested + rejected refinements):** (1) "vs cluster baseline" (block if slope > min-since-cluster) catches all 4 on 06-02 but dilutes cross-batch (7-batch 41.5% WR, cuts 17 winners vs 8) — REJECTED, overfit. (2) Blocking "flat" (Δ≥0) too — REJECTED: flat = same 5m candle = ALL BTC metrics identical (43/43 flat-slope also flat-RSI), and the flat cohort is neutral-to-winning (full 60.7%), so blocking it cuts winners. RSI can't sub-split flat cases (same candle). (3) Cluster-rank cap (block Nth rapid entry) — REJECTED on proxy: cuts more winners (rank≥3 = 15 vs 8) for a weaker cohort (39.5% vs 30.8%). Evolution dominates.

**Discipline (below-gate STATEFUL ship, acknowledged):** 7-batch block cohort N=26 (<30 gate), Avg -0.12 (>-0.20 gate) — clears WR≤40 only. Shipped as a discipline-override: best-evidenced LONG signal of the session (consistent batch + 7-batch + full directional + mechanism + live cluster catch), regime-agnostic (relative comparison survives regime shifts), zero winners cut in-batch.

**LOCKED REVERT GATE:** revert `evo_chase_filter_long_enabled`→false if would-be-blocked LONGs show **≥45% WR on N≥10 fresh**, OR if BTC_ACCEL_CHASE_LONG blocks a **net-positive cohort (Σ%>0) on N≥15 fresh**. Re-confirm at next ≥30-trade checkpoint. SCOPE NOTE: cluster filter only — does not catch isolated losers (e.g. SUI dead-tape/UNMATCHED-2× type); different levers for those.

### 2026-06-03 — SHIPPED: BTC 1h Slope MIN floor (`btc_1h_slope_min_short = -0.60`; SHORT only)

**Change:** new config field `btc_1h_slope_min_{long,short}` (a FLOOR on BTC 1h EMA20 slope). Blocks an entry when `btc_1h_slope < min` — i.e. when the higher-TF slope is too steeply NEGATIVE = entering into a steep 1h crash = exhaustion/mean-reversion bounce. **`min_short = -0.60` (ACTIVE)**, `min_long = 0.0` (plumbed but DISABLED). Disable convention: `0 = off`, any negative value activates. Full D11 stack: config.py default + comment, trading_config.json, engine block (`services/trading_engine.py` ~6450, mirrors the existing MAX gate; runs only when signal still LONG/SHORT after the max gate), UI inputs ("Min BTC 1h Slope L/S"), load+save handlers, config-summary line, counter `BTC_1H_SLOPE_MIN_GATE` (auto-surfaces in Filter Blocks).

**Evidence:** BTC 1h Slope (signed) analysis on the 7-batch pool surfaced a SHORT loser tail below ~-0.60. Sorted steep-negative-1h SHORTs (pool 6 from 06-02 + 2 fresh post-reset 06-02 23:26): SEI -1.006→-1.01, XRP -0.829→-0.71, BTC -0.829→-0.69, JTO -0.620→-1.20 = **STEEP (<-0.60): 0W/4L, Avg -0.90**; MILD (-0.60..-0.40): TON/AAVE/AVAX win, SOL loses = 3W/1L, Avg +0.06. Clean break at -0.60 (empty gap between -0.491 win and -0.620 loss → threshold anchored on the shallowest confirmed loser, JTO -0.62, with buffer from nearest winner TON -0.49; -0.50 rejected as too close to the winning mild band). Mechanism: shorting the exhausted hole (entries showed BTC RSI ~31.6 oversold, range-position 4-20 near bottom). NOT caught by existing filters — these had RISING pair ADX (Pair-ADX-Dir passes) and BTC ADX 32.8 (outside the 24-30 SHORT kill-zone).

**Discipline (below-gate ship, acknowledged):** N=4 across only 2-3 *correlated* events (XRP+BTC same minute, BTC is one of them) — below the N≥30 / ≥6-fresh bar. Shipped as a recent-evidence bet (clean threshold + real mechanism + out-of-sample confirmation across 2 sessions + uncovered by current stack), with a tighter-than-standard revert gate.

**LOCKED REVERT GATE:** revert `btc_1h_slope_min_short`→0 if would-be-blocked (slope<-0.60) SHORTs show **≥50% WR on N≥6 fresh**, OR if BTC_1H_SLOPE_MIN_GATE blocks a would-be-WINNER SHORT on **3+ separate sessions**. Re-confirm at the next ≥30-trade checkpoint.

**LONG deliberately left OFF:** the LONG BTC-1h-slope loser zone is the FLAT band (-0.10..0%, N=60, ~28% WR, structural across 6 dates / distributed pairs) — the *opposite* shape (chop, not steepness). That needs a mid-band range filter, not a floor; not built here. The fresh SUI LONG (slope -0.60) that prompted the look sits in the `<-0.40` band that historically WINS (60% WR) — an outlier loss (its real drivers were dead-tape GV + UNMATCHED 2×), so it does NOT motivate a LONG slope floor.

### 2026-06-02 (evening) — SHIPPED: Pair ADX Direction filter `both` → `rising` (BOTH LONG + SHORT)

**Change:** `trading_config.json` `adx_dir_long` and `adx_dir_short` flipped `"both"` → `"rising"`. Blocks any entry where pair ADX ≤ prior ADX (falling pair ADX = decelerating/exhausting momentum). Config-only flip — full feature stack (engine block `services/trading_engine.py:5873-5891`, PAIR_ADX_DIR `_record_filter_block` counter, UI load/save, display) already existed; nothing code-side changed.

**Trigger — today's batch (scalpars_orders_paper 2026-06-02 20:22):** the last 4 LONGs all lost (ETH −0.69%, DOT −0.97%, SUI −0.83%, BTC −0.69%, STOP_LOSS). They were one correlated BTC-top event (opened 20:06–20:10, BTC itself one of them), all at BTC RSI ~55.6 rising / BTC ADX 23.4 falling, range-position ~91 (extended top), and **pair ADX falling hard (adx_delta −1.3 to −1.7)**. Peak ~0.41% (below the 0.45% trail-arm) → never got going. The 2 LONG winners (XLM, TON) entered 4–8 min earlier, lower in range (~78), BTC RSI 49.5 falling, and trended (peak ~0.70%). Differentiator = falling pair ADX (momentum exhaustion).

**Cross-batch evidence (6-batch pool `reports/dedupe_pool.csv`, 558 trades):**
- Falling-ADX (adx_delta<0) **LONGs**: pool N=4 WR 25% + today's 4 losers = **N=8, ~13% WR** (vs rising/flat-ADX LONGs N=286 WR 52%).
- Falling-ADX **SHORTs**: N=2, **WR 0%**, Avg −1.10% (vs rising/flat SHORTs N=266 WR 60%).
- Combined both sides: **1 win in 10**. Falling ADX is a direction-agnostic loser (exhaustion entry).
- Falling-ADX is rare by construction: only 6 of 558 pool entries (4 L / 2 S), and all 6 are from 06-02 — the book historically almost never entered on falling ADX.

**Discipline note (below-gate ship, acknowledged):** N=8 LONG / N=2 SHORT is below the locked N≥30 filter gate, and all fresh samples are same-day (06-02). This is shipped as a **re-activation**, not a fresh 1-sample filter — `rising` already had cross-batch support on May 28 (`[L1843]` REVERTED both→rising "falling-ADX bled both sides", later relaxed to `both` at `[L1882]`), and was already on the active watchlist (CURRENT_STATE falling-ADX blind-spot). Per the discipline-override rule it carries a tighter-than-standard revert gate.

**LOCKED REVERT GATE:** Revert a side to `both` if would-be-blocked (falling-ADX) entries on that side show **≥50% WR on N≥6 fresh**. Also: if the PAIR_ADX_DIR counter blocks **>15% of that side's attempts** (vs ~1% historical base rate), treat as a regime shift and re-examine (possible over-block). Watch PAIR_ADX_DIR counter in Filter Blocks.

**Expected behavior:** very low fire-rate (~1% of historical entries), asymmetric payoff — blocks a 1W/9L cohort. Main downside: occasionally clips a falling-ADX mean-reversion winner (e.g. ORDI +0.60% today) — 1 such winner vs 9 losers in-sample.


---

### 2026-06-04 — DEMOTED LONG Extension Multiplier (Ext0.4-0.6_L family) 2× → 1×

**Change:** `extension_multiplier_rules` — all 3 LONG rules (`Ext0.4-0.6_L`, `Ext_QuietVol_L`, `Ext_SlowADX_L`) `inv_mult` 2.0 → **1.0** (lev_mult already 1.0). Tags KEPT (rules still fire and label `EXT_*` for tracking) — only the sizing is neutralized. config.py comment + trading_config.json + CURRENT_STATE updated. No engine/UI change (UI uses generic load/save rules editor; runtime reads inv_mult live).

**Trigger:** 2026-06-04 batch (12:10 report). RENDERUSDT LONG closed **−$171.87** — a 2×-multiplied extension cell (`EXT_Ext0.4-0.6_L+Ext_QuietVol_L+Ext_SlowADX_L`). At 1× this is ~−$86. Forensic: late entry (rngPos 83%, +0.53% above EMA13) into a fading move (ADX falling, adxΔ −0.64), BTC RSI falling 5m+30m, BTC 1h slope −0.57; peaked only +0.32%, never armed trailing, EMA13-cross exit saved it from a −3.96% post-exit crater. The multiplier doubled a no-edge late long.

**Cross-batch evidence (FULL pool, deduped, CLOSED, per-cell):**
- `Ext0.4-0.6_L` (base): N=5, WR 40%, Avg −0.216%, Tot **−$235** → ✗ HARMFUL (Total$<0, N≥5)
- `Ext0.4-0.6_L + QuietVol`: N=5, WR 40%, Avg −0.352%, Tot **−$252** → ✗ HARMFUL
- `Ext0.4-0.6_L + SlowADX`: N=3, +$21 → ⚠ low-N noise (positive but below 5)
- Whole LONG-2× class context: 7POOL N=27 −0.194% −$788 · FULL N=131 −0.174% −$3360 (long side carries no gross edge; amplifying it is structurally backwards). Most other harmful pooled LONG-2× cells (`PAIR_60-65_15-18` −$561, `STRETCH_*`) were ALREADY at 1× in current config (`rsi_adx_multiplier_long` empty) — historical, moot. The extension family was the only still-active harmful LONG 2×.

**Verdict basis:** Locked multiplier-cell rule — "✗ HARMFUL (Total$ negative on N≥5) → revert to 1.0×." Base + QuietVol both qualify. Conservative direction (2×→1× on a no-edge losing cohort cuts loss/variance, never amplifies), so the borderline N (=5, at the gate) is acceptable. "Caps for losers, multipliers for winners" — never multiply a side with no edge.

**LOCKED REVERT GATE:** RESTORE 2× only if `Ext0.4-0.6_L` reaches **N≥15 fresh (current-stack) AND Total$ > 0** in the Extension Multiplier Performance table. Until then it fires at 1.0× but stays tagged, so the cohort's true 1× edge is observable.

**Method note (per-batch vs pool):** In the 2026-06-04 batch itself the LONG 2× looked fine (+~$24 net — RENDER −172 offset by SKYAI +158 / HYPE +38, both 2×). Damage is only visible cross-batch — reaffirms core principle: judge multipliers on the pool, never a single batch.

---

### 2026-06-04 — WATCHLIST gate: C1+C6 SHORT toxic-combo (observation only, NOT shipped)

**Context:** Investigating the 2026-06-04 batch Pattern-C tables + cross-batch. Initial pass conflated "any C6 SHORT" (N=27, −$802) with a C6 problem. Re-run by **C-signature (UI Pattern-C Combination Tracker convention — group by which C's fire, W ignored)** on the 7-batch pool (BE-off proxy) corrected it:

- **C6 SHORT (C6 the only C):** N=24, 67% WR, Avg +0.011%, **−$135**, NP 8% → ≈flat, high-WR, NOT a loser cohort.
- **C1+C6 SHORT:** N=3, **0W/3L**, Avg −0.693%, **−$667**, NP 67% → the real bleed. Mechanism: C1 (capitulation chase) + C6 (macro over-extended) co-occur = shorting an over-extended capitulation that bounces. BOTH are multiplied cells (C1 = 3× eff), so the fat-tail losses are amplified (JTO −$320, SOL −$240, TON −$232 in the C6 audit were all 2× and mostly C1+C6 / C6+W stacks).
- For reference, LONG worst C-signatures same pool: C7 N=5 20% WR −$245 (countertrend bad-long), C4 N=12 42% −$314.

**Method note (definition mismatch that triggered this):** I had been computing "Cx alone" as *Cx only-C AND no W at all*, which gave C6-SHORT-alone = 0 and contradicted the UI (where TON shows under "C6" because the tracker groups by C-signature and ignores W). Corrected to the UI convention. Lesson logged: **match the UI's grouping convention when the user is reading off the UI.**

**Status:** WATCHLIST — N=3 ≪ N≥30 ship gate (and ≪ the ≥6 needed to act on a multiplier verdict). Direction-consistent (0/3 across 3 dates) so it clears the watchlist bar, not the ship bar. No config change.

**LOCKED GATE:** Cap effective multiplier to **1.0× on C1+C6 SHORT** if the combo holds **≤30% WR on N≥6 fresh** (current-stack). Track via the Pattern-C Combination Tracker row "C1+C6 SHORT". Until then, do nothing — C6 alone is fine and must not be blocked.

---

### 2026-06-04 — WATCHLIST gate: C7 LONG-alone / no-W (observation only, NOT shipped)

**Finding:** C7 = "Pair Countertrend Bounce." A LONG on a countertrend bounce with NO W (trend) confirmation = buying a dead-cat bounce / falling knife. Recent BE-off data (7-batch pool + 2026-06-04 batch, deduped):

- **C7 LONG with no W: N=3, 0W/3L, ≈−$265** — 1000LUNC (−$130, SL), ONDO (−$15, EMA13 cross), HOME (−$121, SL, this batch). 3 separate dates (May 28 / May 30 / Jun 4) → direction-consistent, not a single fat tail.
- Cross-ref (UI C-signature convention, W ignored): "C7" LONG row = N=5, 20% WR, −$245. The 2 extra are C7-with-W (trend-confirmed) and should NOT be blocked.
- Mechanism match: HOME was the batch's "textbook bad-long" (pairGap −1.44% countertrend, +0.90% over EMA13, BTC 1h −1.00); C7-alone *is* that bad-long isolated as a signature.

**Definition note:** block target = **C7-match AND `entry_pattern_w_any_match`=False** (countertrend long lacking trend confirmation). Distinct from the UI "C7" row, which groups by C-signature and ignores W (hence N=5 not N=3). Stated explicitly to avoid the alone-vs-C-signature confusion from earlier this session.

**Status:** WATCHLIST — N=3 ≪ N≥30 ship gate; clears the ≥3-sample direction-consistent watchlist bar only. C7 LONG is observation-only (no multiplier), so the only lever is an entry filter. No config change.

**LOCKED GATE:** Ship a LONG entry-block on "C7-match AND w_any=False" only if the cohort holds **≤30% WR on N≥8 fresh** (current stack). Would have blocked HOME this batch — but one batch is not enough.

---

### 2026-06-04 — OBSERVATION-tracking added: C6 LONG (NOT a loser — do not block)

**Request:** track C6 LONG in detail. **Finding (recent 7-batch pool + 2026-06-04 batch, BE-off, deduped):**
- C6 LONG (all with C6 match): **N=7, 4W/3L, 57% WR, Avg ≈−0.002% (flat), Tot −$46.6, de-multiplied +$18.9 (positive), NP 14%.**
- By C-signature: C6 (only-C) N=6 50% −$85 (1×: −$19.5); C6+C7 N=1 +$38 (PORTAL).
- By W: **C6 LONG always fires no-W (N=0 with W)** — macro-over-extended longs carry no trend confirmation, yet still win 57% and are de-mux positive.
- Trades: RENDER −172 (2×), FET −117 (2×), HOME −28 (1×), INJ +36 (1×), SWARMS +38 (1×), PORTAL +38 (1×), SKYAI +158 (2×). The −$46.6 as-sized is ENTIRELY the 2× amplification on RENDER+FET; RENDER's 2× = the extension multiplier demoted earlier today, so the main amplifier is already handled.

**Verdict:** C6 LONG is NOT a loser cohort. Flat-to-positive, 57% WR, de-mux positive. No cap, no block — blocking would mislabel a breakeven cohort (high-WR-net-losing trap is about *sizing*, and the sizing culprit is already demoted). Observation-tracking only.

**WATCH-FOR-DETERIORATION GATE:** revisit (consider cap/block) only if C6 LONG turns net-negative **DE-MULTIPLIED** with **≤35% WR on N≥10 fresh**. Track the Pattern-C Combination Tracker "C6 LONG" row.

---

### 2026-06-04 — DISABLED BTC-Accel Chase Filter (LONG) — A/B test, C-levels hypothesis

**Change:** `evo_chase_filter_long_enabled` true → **false** (trading_config.json). config.py default was already False; `evo_chase_window_min=30` retained for if re-enabled. Counter BTC_ACCEL_CHASE_LONG will read 0 while off.

**Rationale (operator-directed):** Hypothesis that the LONG-side bleed is driven by **C-pattern levels (C4 low-vol chop, C6 macro-over-extended, C7 countertrend-bounce), NOT by BTC-chasing.** The BTC-Accel Chase filter was a below-gate Jun-3 ship (N=26, 30.8% WR proxy). Turning it OFF isolates the variable: with chase-blocking removed, if the LONG losers still cluster on C-levels, the C-level filters (C7-no-W block, C1+C6 cap, C6-LONG watch) are the real lever and the chase filter was noise.

**RE-ENABLE / KEEP-OFF GATE:** Track LONGs that the chase filter WOULD have blocked (live BTC EMA20 slope > slope at last LONG within 30min) now that they can open:
- ≤35% WR on N≥10 fresh → chase signal is real → RE-ENABLE.
- ≥50% WR on N≥10 fresh → chase was a false signal → KEEP OFF, pursue C-level filters instead.
Re-evaluate at next ≥30-trade checkpoint.

**Note:** one change at a time for clean attribution — this is the only live toggle this step; the C-level items remain observation-only watchlist until their own gates trip.

---

### 2026-06-04 — REDEFINED C1+C6 gate → full C1 SHORT combination review (incl. W)

**Change:** the Jun-4 "C1+C6 SHORT cap" watchlist gate is broadened to a **full C1 SHORT combination review** carried into the next batch, tracking **C+W signatures** (not just C), with de-mux 1× alongside as-sized. Two refinements vs the original:
1. **cap → BLOCK ENTRY** for C1+C6 — de-multiplied it still loses (−$333, 0% WR, 2 NP), so the entry is bad, not merely over-sized; a cap would only lose less.
2. **W dimension added** and shown not to matter — C6 is the clean driver.

**7-pool baseline (BE-off, beW=0):**
- C1+C6: N=3, 0W/3L, −$667 (de-mux −$333), 2 NP → TOXIC (loses under W1 / W1+W2 / W1+W2+W6 alike).
- C1-only N=9 78% +$48 · C1+C3 N=2 100% +$202 · C1+C2 N=2 100% +$441 → all winners (win under every W overlay).
- Cross-tab: **C1·has-C6 = 0% WR / −$667 vs C1·no-C6 = 85% WR / +$691** (razor-clean). C1·has-W2 78% vs C1·no-W2 57% (muddy, and its negativity is C6-contaminated — 2/3 C1+C6 losers carry W2). ⇒ block target = **C1+C6, W-agnostic**.
- C1-alone (only C1, no W): N=1 BE-off (MMTUSDT −$182); historically BE-inflated (91% WR = breakeven/fast-exit locks) → unreliable, do not act.

**GATES (observation only):** (1) C1+C6 SHORT → BLOCK ENTRY if ≤30% WR on N≥6 fresh. (2) Any winner C1 combo flips net-negative on N≥5 fresh → review/demote its multiplier. Next batch: report the full C+W C1 table.

---

### 2026-06-04 — OBSERVATION added: W2 LONG + W3 LONG/SHORT (next-batch tracking)

**W2 LONG (NOT a winner — W2 value is short-side only):**
- 7-pool: SHORT N=55, 76% WR, +0.109%, **+$634** (multiplied 3× eff, winning — keep). LONG N=14, 36% WR, −0.048%, **−$47** (baseline 1×, ≈breakeven).
- 2026-06-04 batch: SHORT N=4, 75% WR, +$718. LONG N=5, **0W/5L, −0.673%, −$337, 4 NP (DOA)**.
- W2-long losers this batch (AAVE/SOL/XRP/1000PEPE) overlap the C6/Neither DOA bleed → regime, not a standalone W2-long signal. W2-LONG is 1× (not multiplied), so the only lever is an entry block.
- **GATE:** consider a LONG entry-block on W2 only if ≤35% WR on N≥10 fresh AND net-negative. Do NOT touch W2 SHORT.

**W3 LONG & SHORT (too thin — no verdict):**
- W3 = "Energetic volatility." LONG: 7-pool N=3 33% −$96, batch N=1 −$129 (XPL DOA). SHORT: 7-pool N=2 0% −$232, batch N=0.
- Mildly negative both directions but N=1–3 everywhere → no statistical weight.
- **NO gate. Accumulate to N≥8 per side before any verdict.** Track in Pattern-W Combination Tracker.

---

### 2026-06-05 — SHIP: 6-change stack (chase ON · ATR-split LONG · gvol-override removed · ETH no-trade) + pool rename 7→8

**Derived from the 6-05 batch autopsy** (60 closed Jun 3–5, 44L −$2,321 / 16S +$714). LONG side proven to have no durable edge; only entry-removal (chase) + cohort-correct exits/sizing help. Operator-directed ship, batch-derived (in-sample) — gates below carry the haircut.

**Shipped (all 6, D11-complete: config.py + trading_config.json + engine + UI + load/save):**
1. **Chase filter ON** (`evo_chase_filter_long_enabled=true`) — was OFF since Jun 4 A/B. Re-enabled: on the batch it blocks 13 longs, removing −$853 of realized loss (the only lever that removes losers). Stateful: blocks a LONG when live BTC EMA20 slope > slope at last LONG opened within 30min.
2. **ATR-LOW Fixed TP (LONG)** (`atr_low_fixed_tp_long_enabled=true`, ATR<1.1 → TP +0.25%). New engine exit `ATR_FIXED_TP L1` — a profit-LOCK (fires only on a green trade; never cuts a DOA loser). Low-ATR longs have no runners (batch: high-ATR reach trailing arm ~80% vs ~30%, all 6 RUNNER_TRAIL longs ATR≥1.0) → lock the pop. Wired into both post-exit-tracking whitelists (live reg 4238 + recovery 751) → appears as its own row in Post-Exit Regret Deep Dive.
3. **ATR-HIGH multiplier (LONG)** (`atr_high_mult_long_enabled=true`, ATR>1.1 → inv ×2.0). New `_lookup_atr_multiplier` dimensional candidate (max-wins, **pattern-blocked** so 2× stays off C-pattern/DOA high-ATR longs like INJ, hard-capped 2×). Note: near-neutral on the batch (doubles INJ/STO losers ≈ +$13 net) — operator accepted; tight revert gate.
4. *(Fast-exit 0.20%/5min — DROPPED.)* Operator chose fix-TP-only after I flagged FAST_EXIT/PATTERN_FIXED_TP are profit-LOCKS in the engine, not loss-cuts. The earlier "+$1,040 modeled" assumed fast-exit cut DOA longs — WRONG; corrected realistic batch ≈ breakeven, chase-driven.
5. **Remove gvol capitulation override (SHORT)** (`global_volume_max_short_capitulation_override_enabled=false`). New master toggle; when off, high-GV shorts always blocked regardless of BTC capitulation. **No-op unless `global_volume_max_short>0`** (cap currently disabled) — flagged in config + UI. $0 effect on this batch (no capitulation event Jun 3–5; the override's historical losers were May-27 / 7-pool).
6. **ETH → no-trade (track-only)** (`no_trade_pairs="BTCUSDT,ETHUSDT"`). ETH stays visible/scanned, entries blocked (counter PAIR_NO_TRADE). Evidence: ETH shorts −$230 this batch (3 trades, squeezed on the bounce) AND −$230 prior batch — recurring squeeze pair. Track-only (not blacklist) so the would-be record stays observable.

**Corrected expectation (stated to operator):** full stack ≈ breakeven on the batch (chase −$853 does the real work; exit caps reduce give-back on the few low-ATR winners; 2× ATR neutral; gvol-removal $0; ETH-track +$230). NOT the earlier +$1,040.

**REVERT GATES (locked):**
- **ATR-HIGH 2× LONG:** drop to 1.0× if Total$<0 on N≥5 fresh ATR>1.1 longs.
- **ATR-LOW Fixed TP:** watch the `ATR_FIXED_TP L1` row in Post-Exit Regret — if avg Post-Peak is high (cohort kept running after the cap), 0.25% is too tight → raise/disable. If Post-Peak ≈ 0, lock is correct.
- **Chase ON:** keep while would-be-blocked LONGs run ≤50% WR; the Jun-4 re-enable gate is now resolved (re-enabled).
- **gvol override removed:** re-enable only if would-be-passed capitulation shorts (high-GV + BTC RSI<30 & slope<0) show ≥55% WR on N≥6 fresh.
- **ETH no-trade:** revisit if ETH (either side) would-have-won ≥55% WR on N≥8 fresh while track-only.

**Pools:** added the 60 batch trades to the pool. **`dedupe_pool_7batches_may26-jun2.csv` (222) renamed → `dedupe_pool_8batches_may26-jun5.csv` (282, May26→Jun5, 164L/118S)** (batch aligned to the 196-col schema). `dedupe_pool_FULL.csv` rebuilt → 1,193 closed (Apr28→Jun5). Batch text report saved to `reports/batch_report_2026-06-05.txt`.

---

### 2026-06-05 — WATCHLIST (NOT shipped): BTC RSI×ADX Cross-Filter simplification

**Proposal (queued for next-batch review, operator-directed hold):**
1. **Delete dead rule 3** — `RSI 60-65 × ADX 27-30`. First-match-wins by RSI band means rule 2 (`60-65 × 22-25`) claims the 60-65 band first, so rule 3 never evaluates. Pure dead config. (Side note: the cell it *intended* to allow, 60-65×27-30, is actually a decent FULL-pool cell — N=14/79% WR/+$78 — but that's contaminated/thin, not a re-open trigger.)
2. **Replace the 5 RSI×ADX rules with two RSI-only blocks** (drop the ADX axis entirely):
   - `RSI 50-55 → block`
   - `RSI 60-100 → block`
   - Net effective LONG surface: allowed = **RSI 40-50 + 55-60** (all ADX 18-40); blocked = 50-55 & ≥60.

**Why drop ADX:** it does not separate within any RSI band — every ADX slice of a band carries the band's sign:
- RSI 50-55: ADX 18-25 −$394 / 25-32 −$551 / 32-40 −$196 (all negative).
- RSI 65-70: ADX 18-25 −$1,564 / 25-32 −$1,251 / 32-40 −$829 (all negative).
The per-ADX carve-outs (rules 2 & 4) are fitting noise.

**Theoretical critique:** the filter triangulates "good BTC regime for longs" from BTC RSI (momentum *level*) × BTC ADX (trend *strength*) — neither encodes BTC *direction*, which is what a long needs. Hence it barely separates (blocked −0.194 vs allowed −0.174). It is also internally inconsistent: it blocks ≥70 (overbought) but ALLOWS 65-70 — the same "long into BTC near-exhaustion" mechanism one notch earlier, and the biggest LONG loser.

**Band sign-consistency (FULL pool vs last-4-batch):**
- 40-50: +$293 / +$61 → **+ both** (the only consistent winner; tiny N — longs rarely fire <50 BTC RSI).
- 50-55: −$1,141 / −$450 → **− both** (keep blocked).
- 55-60: −$733 / +$524 → **FLIPS** (non-stationary — allow as least-bad firing band, do NOT bank).
- 60-65: −$1,976 / −$745 → **− both** (currently only sliver-blocked → block fully).
- 65-70: −$3,644 (N=184) / −$1,643 (N=48) → **− both, biggest loser, currently WIDE OPEN** ← the hole.
- ≥70: −$168 / — → − (already blocked).

**Last-4-batch impact (as-traded):** current surface allows 115 longs / −$1,860; new surface allows 26 / +$585. **Kills 89 longs (77%)** — all from 60-65 (41 / −$802) + 65-70 (48 / −$1,643) = **−$2,445 of loss removed**. 0 measurable adds (the 55-60-full opening has no historical trades — those cells were live-blocked; forward-only). The surviving +$585 rides the non-stationary 55-60 streak → NOT durable; the durable piece is the −$2,445 of 60-70 losers.

**Caveats:** big volume cut (77% of longs) ≈ near-shutdown of longs in the BTC-mid-RSI regime; losses partly overlap with chase + ATR-low fix-TP already live (incremental benefit < raw −$2,445); 55-60 non-stationary; FULL pool BE-on-contaminated (recent confirms the 65-70 finding, which is the load-bearing one).

**SHIP GATE (next batch):** ship the `60-100 → block` simplification IF 60-65 AND 65-70 longs are net-negative AGAIN (3rd-window confirmation of the both-window pattern). Delete rule 3 anytime (zero-risk cleanup). **Post-ship revert:** re-open 60-65 if would-be-blocked 60-65 longs show ≥50% WR on N≥10 fresh. Keep 55-60 allowed regardless (least-bad firing band) but treat its P&L as noise.

---

### 2026-06-06 — SHIP: SLWide widen `signal_active_sl` −0.70 → −1.00 (STOP_LOSS_WIDE only)

**Change:** `confidence_levels.{VERY_STRONG,STRONG_BUY}.signal_active_sl` −0.70 → **−1.00**. `stop_loss` kept −0.70. Lower confidence levels (−0.35) untouched (they barely trade).

**Level chosen from a sweep (the first counterfactual used −1.20; corrected here).** Modeling the real `signal_active_sl`×ATR-widen(×1.5)×−1.20-floor interaction (signal_active_sl only bites for ATR<0.60 — above that the ATR-widen already exceeds it): NET Δ is **monotonic to −1.20 (EV-max)** — 9-pool: −0.90 +$861 · −1.00 +$939 · −1.10 +$1,459 · −1.20 +$1,532; last-4+06-06: −0.90 +$1,044 · −1.00 +$1,149 · −1.20 +$2,020. Past −0.90 the marginal survivors troughed −1.0/−1.1% (proxy-soft recovery) and the deepening tail grows + is leveraged (2-3× cells). **Operator chose −1.00 (middle): ~60% of −1.20 EV, short of the deepest-trough trades.** −1.20 stage-up watchlisted for next batch.

**Mechanism (engine trading_engine.py:7551-7612):** `effective_sl` defaults to `stop_loss`; when the entry signal is STILL active at the stop, it's overridden to `signal_active_sl`. ATR-widen (×1.5) + floor (−1.20) are applied to whichever base, THEN the label is set: signal-active → **STOP_LOSS_WIDE**, signal-dead → **STOP_LOSS**. So `signal_active_sl` moves ONLY STOP_LOSS_WIDE; `stop_loss` governs STOP_LOSS. They were both −0.70 (identical fire level) until now. This is the dedicated lever to widen the signal-active stop independently.

**Thesis:** STOP_LOSS_WIDE = "stopped while the setup was still valid" — the reversal/regret population. Give it +0.20% more room so a wick/pullback within a still-valid signal doesn't kill the trade before it plays out. Signal-DEAD stops (STOP_LOSS) stay tight — those are correct exits (thesis gone, nothing to ride).

**Evidence (widen-to-−1.20 counterfactual, fix-TP applied to lo-ATR long survivors):**
- 8-pool (May26-Jun5): STOP_LOSS_WIDE N=56, −$6,554 → Δ **+$434** (survive 19 +$1,537 / deepen 35 −$1,103). Reverse rate 28%.
- last-4-batch (May29-Jun5): N=41, −$4,643 → Δ **+$922** (survive 18 +$1,464 / deepen 23 −$542, 2.7:1). Reverse rate 32%.
- Positive in BOTH windows (sign robust); magnitude regime-dependent (bigger in choppy-bounce, smaller in trend-crash because deepeners grow — May26-28 crash days shrink the 8-pool figure). After in-sample haircut ≈ +$250 (8-pool) to +$500 (last-4).
- Direction: last-4 SHORT Δ+$591 (9 save / 8 deepen) > LONG Δ+$331 (9 save / 15 deepen) — both positive.

**Lever choice — base SL, NOT the ATR multiplier:** the reversers are LOW-ATR (DOGE/WLFI/UNI, ATR<0.5) → ATR×1.5 < 0.70, so the ATR-widen never engages for them and they stop at the −0.70 base. Raising `sl_atr_multiplier` barely moves them; raising the base `signal_active_sl` widens them directly. ATR mult stays 1.5 (it's for high-ATR pairs, already handled).

**Caveats:** (1) widens into the LEVERAGED tail — the 35 (8-pool) deepeners are 2-3× cells too. (2) Regime-fragile in magnitude (not sign): a severe cascade could grow the deepen side. (3) Symmetric L/S — current SL config is per-confidence, not per-direction; split into `_long`/`_short` only if data diverges.

**REVERT GATE:** revert `signal_active_sl`→−0.70 if (a) STOP_LOSS_WIDE survive-vs-deepen goes net-negative over N≥30 fresh, OR (b) a single correlated-crash window adds ≥−$300 of deepened STOP_LOSS_WIDE loss vs the −0.70 baseline. Drawdown-tied, not just net-$, because the risk is the leveraged crash tail.

**Pools/report:** saved `reports/orders_2026-06-06_13L_10S.csv`; `dedupe_pool_8batches_may26-jun5.csv` (282) renamed → `dedupe_pool_9batches_may26-jun6.csv` (305, May26→Jun6, 177L/128S); `dedupe_pool_FULL.csv` rebuilt → 1,216 closed (Apr28→Jun6); batch report template `reports/batch_report_2026-06-06.txt`. Operator will reset and start a fresh batch on this config.

---

### 2026-06-06 — EXPERIMENT: BTC RSI×ADX Cross-Filter OFF (both sides, ~24h open run)

**Change:** blanked `btc_rsi_adx_filter_long` and `btc_rsi_adx_filter_short` (both → ""). Operator-directed time-boxed exploration. NOT a permanent removal.

**Archived for re-add (verbatim):**
- LONG: `70-100:40,60-65:22-25,60-65:27-30,55-60:20-25,50-55:99-100`
- SHORT: `30-35:30,35-40:20-26,45-50:25,0-30:25-30`
- (Pair-level `rsi_adx_filter_long`=`60-65:0-25` / `rsi_adx_filter_short`=`30-35:25,35-50:30` LEFT INTACT — different filter.)

**Rationale — break the measurement deadlock.** All session we've been blocked: the blocked-cell cohorts can't be validated because they're blocked (no recent data), and the historical pools are BE-on contaminated. The open BTC RSI×ADX simplification (ship `60-100→block` IF net-negative a 3rd window) is stuck for exactly this reason. A deliberate, time-boxed open run generates fresh current-stack per-band data on the LONG (50-55, 55-65, ≥70) and SHORT surfaces. Paper, so cost is bounded.

**Guardrails:** only the RSI×ADX surface opens — chase, ADX-delta, pair-ADX-dir, BTC-1h-slope, pair-level rsi_adx_filter, ADX [18,40] gate, quality score all STAY on (chase + ATR-low fix-TP still protect). Abort early if opened-band entries ≤30% WR on N≥20, or drawdown past comfort. **SHORT = the money-maker → tighter watch** (its cross-filter cells were validated winners historically; removing them is higher-risk than the no-edge LONG side — re-add SHORT immediately if the open short book degrades).

**Post-run analysis (pre-committed):** bucket fresh longs + shorts by BTC RSI band × ADX; re-add the cross-filter rules ONLY for bands confirming net-negative on fresh data; leave open any that surprise positive. Feeds the BTC RSI×ADX simplification watchlist directly. Judge on per-band breakdown, NOT the run P&L (LONG expected worse — the cost of the data).

**Attribution caveat:** stacks with the just-shipped SLWide −1.00 (entry-mix vs exit — separable but flagged).

**Addendum (same experiment):** also DISABLED the **Pair-Extension floor** via master toggle (`entry_dist_from_ema13_filter_enabled`→false; LONG min ext 0.20 value retained for one-flip re-enable). Same profile as the cross-filter — thin (N=9, same as ship), deadlocked (zero data since May 22), rare/tiny-save cohort, immaterial to LONG P&L (allowed longs lose −$5,669 vs this saving $237). The <0.20 band IS the worst-per-trade extension (−0.304% / 22% WR), so expect those longs to lose — the point is fresh data on whether bottom-of-pullback longs still die under the current stack. Post-run: re-enable the toggle if ext<0.20 confirms loser; leave off if it surprises. (Toggle disables both directions; SHORT was already 0.)

---

### 2026-06-06 — DISABLE: Runner Stretch-Trail exit (`runner_trail_enabled`→false); + UI toggle added

**Change:** `runner_trail_enabled` true → false. High-ATR LONG runners now exit via normal tight trailing (the stretch-trail handoff no longer fires). Also added a UI control row (toggle + atr_min/arm_peak/k) — none existed before; flippable from the dashboard now.

**Evidence (Runner Trail Performance table, all 6 live RUNNER_TRAIL exits since the Jun-1 ship):** net gain vs tight = **+$30** (3/6 beat tight = coin flip). The mechanism WON on the small "runners" (ORDI/SKYAI/RENDER, peaks 0.76-1.23) but LOST on the bigger ones (VIC −$15 / NEAR −$14 / PORTAL −$45) — those peaked 1.34-1.55% then FADED, and the stretch-trail gave back more than tight would have (PORTAL: peak +1.51 → exit +0.39, 26% of max; tight CF +0.84). Root cause: arms at peak≥0.70 which catches non-runners that fade; built for IDU-class monsters (peak >2-3%), none of which appeared in 6 fires. So it's a wash that quietly hurts the 1.0-1.5% peakers → disabled.

**Independence verified:** the high-ATR **2× ATR Multiplier** (`atr_multiplier_rules` "Runner", entry SIZING) is a SEPARATE code path (`_lookup_atr_multiplier` at entry) and does NOT reference `runner_trail_enabled`; the exit (`indicators.py`) does NOT reference the sizing multiplier. Disabling the exit leaves the 2× sizing fully intact (operator confirmed intent: "keep the multiplier").

**Re-enable gate:** turn back on only if a true monster runner (peak >2-3%) appears where the stretch-trail clearly out-captures tight. UI toggle exposed for one-click flip. (Stacks with the open-filter run, but separable — high-ATR LONG runners are rare and the P&L impact is ~$0, so it won't muddy the entry-band read.)

---

### 2026-06-07 — SHIP (as measurement): EMA13-cross-LONG phantom-CF + per-direction toggle

**Change:** added per-direction gates `ema13_cross_exit_long_enabled` / `ema13_cross_exit_short_enabled` under the master `ema13_cross_exit_enabled`. Set **LONG=false, SHORT=true**. When a side is off, the EMA13 cross no longer closes — it records a **phantom** (`phantom_ema13_cross_pnl` / `_at`, NEW DB cols) of the would-have-exited pnl and the trade rides to its real exit.

**Why (overrides the Jun-7 model-rejection):** the 9-pool model said disabling EMA13-cross-LONG is net-negative (−$65 at SLWide −1.00; coin-flip 22/22; cross protects reversing longs from the wider stop). Operator's call: don't argue the model — **measure it live at zero blind risk** via the phantom CF. This is the bot's shadow/phantom pattern applied to the EMA13 cross, mirroring the 🛡️ EMA13 Strict-Mode table.

**Build (full D11 + DB):**
- config.py: 2 per-direction fields (default True).
- models.py + database.py auto-migrate: `phantom_ema13_cross_pnl` (Float) + `phantom_ema13_cross_at` (DateTime).
- trading_engine.py (7368): per-direction gate in the EMA13 cross block — `_e13_stack_confirms and not _e13_dir_enabled` → record phantom (first-fire only) + fall through (no close); `... and _e13_dir_enabled` → close as before.
- main.py: `_compute_ema13_cross_disabled_cf(orders)` (phantom vs actual per direction, verdict ★ DISABLE-wins / ⚠ KEEP-cross) + payload wire.
- templates/index.html: LONG/SHORT sub-toggles (load+save, IDs verified 3×) + 🔀 EMA13 Cross Disabled-Direction CF table + render.

**READ GATE:** at N≥20 fresh LONG phantom-fires — ⚠ KEEP (cross beats held, net-$ neg) → re-enable LONG; ★ DISABLE-wins (held beats cut) → keep off (model was wrong). Self-contained (phantom vs held per trade), so it answers the EMA13 question cleanly even amid the open-surface run.

**Caveat:** stacks a 3rd live change on the open run (open entry + SLWide −1.00 + EMA13-cross-LONG-off), but the CF is per-trade-isolated so attribution holds for this specific question. Modest expected cost (model −$65) bought with real data.

---

### 2026-06-08 — SHIP: Trailing Min-Profit Gate (`trailing_min_profit_to_fire = +0.10`)

**Change:** the price-drop trailing stop now fires only when its exit level `(peak_pnl − effective_pullback) ≥ +0.10%`. Below that it's SUPPRESSED — the trailing goes dormant (does NOT realize a loss/sub-min exit), the trade rides the hard SL, and the trailing RE-ARMS once the peak climbs enough to lock ≥ +0.10, then trails the new peak normally through L2/L3/… like any runner.

**Root cause:** the ATR-widened pullback (`max(0.25, ATR×0.5)` + 0.1/level) exceeds the peak on **high-ATR L1** trades that barely arm. E.g. VELVET peak +0.45, ATR 1.33 → pullback 0.67 → trailing level 0.45−0.67 = **−0.22** → it exited RED on a pair that then ran to **+6.15%**. All 5 negative trailing exits (9-pool+batch) were this exact pattern: L1 + ATR 0.97-1.71 + pullback > peak.

**Evidence (9-pool + 06-08 batch):** the 16 trailing trades whose stop sat <+0.10 (the whipsaw zone) **all 16 recovered to ≥+0.25% after being cut**. Counterfactual (suppress + ride, with re-armed trailing / fix-TP / SLWide −1.00): **current +$190 → +$1,697 (Δ +$1,506)**. Big recoverers (VELVET −$24→+$645, HOME −$28→+$281, WLD +$90→+$307, XLM +$45→+$228, NEAR +$7→+$176) dwarf the few that ride to SL (NFP +$56→−$134, SEI, RENDER). In-sample → ~40% haircut → still ~+$900. Structural (a trailing stop shouldn't realize a loss), not curve-fit.

**Design (operator-chosen +0.10, not the floor):** it's a GATE, not a forced exit value — the trailing still exits at the natural peak−pullback when it fires; the gate only decides *whether it's armed*. Self-scopes to L1 (data: every L2+ trade already has peak−pullback ≥ +0.09; the gate is mathematically incapable of biting above L1). Default −99 = disabled (current behavior preserved). Kept ATR multiplier ×0.5 (validated, needed for runner captures — the gate keeps the wide pullback where it works, disables it only in the L1 whipsaw).

**Build (D11):** config.py field (default −99) · indicators.py gate on both LONG/SHORT price-drop fire (not the TREND_BREAK path — that's always +) · trading_config.json = 0.10 · UI input + load/save (ID 3×) · CURRENT_STATE exit-stack line.

**REVERT GATE:** revert (`→ −99`) if trailing exits net-negative vs the old behavior on N≥30 fresh L1-whipsaw trades (i.e. the suppressed trades ride to SL more than they recover). Watch: do suppressed trades recover (capture) or dump to −1.00 (cost)?

**ADDENDUM (same commit) — Phantom-CF instrument bundled with the gate.** Because the gate adds NO new exit reason (suppressed trades exit via existing TRAILING_STOP L2-5 / STOP_LOSS_WIDE), suppressed fires can't be identified post-hoc. So the gate ships WITH a measurement instrument, mirroring the EMA13-cross-disabled phantom CF (commit a182718): ① `models.py` cols `phantom_trail_suppress_pnl` + `_at` · ② `database.py` auto-migrate ALTERs · ③ `indicators.py` threads `trail_suppressed_pnl` (= would-have-cut pnl_pct) out of `check_exit_conditions` on every gate-blocked fire · ④ `trading_engine.py` (monitor loop, after the check_exit_conditions call ~5267) records `phantom_trail_suppress_pnl` the FIRST time it's set per order · ⑤ `main.py` `_compute_trail_gate_cf(orders)` → payload key `trail_gate_cf` (per-direction: avg phantom vs avg actual, Δpp, held>cut, Total$Δ, verdict ★ SUPPRESS wins / ⚠ REVERT / ✓ Marginal / ⚠ Low N) · ⑥ `templates/index.html` 🛡️ Trailing Min-Profit Gate — Suppressed-Fire CF table (id `trail-gate-cf-body`) below the EMA13 CF table + render JS. **Read gate logged in CURRENT_STATE watchlist (N≥20 fresh suppressed-fires → ⚠ REVERT to −99 / ★ keep +0.10).** Held the gate commit until the instrument was in so the decision is measurable from day one.

### 2026-06-08 — RELAX: EMA-Gap-Expanding filter → `prev2_only` mode (live A/B w/ MARGINAL tag)

**Trigger:** the live **Filter Blocks** table (16-trade 06-08 split report) showed **PAIR_EMA_GAP_NOT_EXPANDING = the #1 entry blocker by a wide margin: 8,684 blocks / 31% of all (29% of LONG, 34% of SHORT)**; next were PAIR_RSI_RANGE (17%) and PAIR_ADX_MAX (12%). Operator concern = too few trades; wanted the dominant *throughput* blocker, not more restrictions.

**Mechanic (was, strict 'both'):** block entry unless the EMA5-EMA13 gap is strictly greater than BOTH prev1 AND prev2 candles — i.e. a *fresh 3-bar expansion high at the entry tick*. Momentum stairsteps (expand→pause→expand) even in healthy trends, so this rejects a large share of valid continuation entries sitting in a 1-candle pause.

**Change:** new `ema_gap_expanding_mode` ('both' default | 'prev2_only'). `prev2_only` drops the prev1 check (block only if gap ≤ prev2), tolerating a 1-candle pause within an intact trend. Shipped live = `prev2_only`. **Theory rationale to relax (not disable):** the filter measures the one orthogonal axis nothing else does — trend-spread *acceleration* (the derivative) vs all the level-based gates (RSI/ADX/gap-level) — so the concept is sound; only the strictness was overdone, and it's belt-and-suspenders with the NO_EXPANSION *exit* (trading_engine.py:4803) that already culls non-developing trades.

**Measurement (live A/B, NOT phantom — these entries actually open):** `entry_gap_expand_marginal` boolean tagged at entry = True iff the trade passed prev2 but would have FAILED prev1 (`gap > prev2 AND gap ≤ prev1`) — exactly the cohort the relaxation newly admits. `indicators.gap_expand_marginal(indicators, dir)` helper (pure read of the same gaps get_signal uses); set in `open_position`. `main._compute_gap_expand_cohort` → payload `gap_expand_cohort`; UI 🪟 Gap-Expand Relaxation table (id `gap-expand-cohort-body`) splits MARGINAL vs STRICT × dir (WR / Avg% / Total$ demux). Build (D11): config.py mode field · indicators.py mode gating of prev1 branch (LONG+SHORT) + helper · trading_config.json='prev2_only' · models.py + database.py column/migrate · UI mode `<select>` + load/save + report line + updated filter description comment + cohort table/render. (Chose live-tag over phantom because the engine tracks post-exit trajectories for real closed trades, not forward P&L for never-opened entries — so admitting+tagging is the only accurate way to measure entry-admission.)

**READ GATE (N≥20 fresh MARGINAL):** ★ keep prev2_only if MARGINAL WR ≥ STRICT AND Total$>0 · ⚠ revert mode→'both' if MARGINAL net-negative OR WR ≫10pp below STRICT. **ABORT EARLY (real $ at risk):** revert to 'both' if MARGINAL WR≤30% OR Total$<−$150 on N≥10. Default 'both' preserved in config.py so legacy behavior is one flip away.

### 2026-06-08 — RE-ADD: BTC RSI×ADX cross-filter SHORT (abort gate tripped on the open experiment)

**Trigger:** the 06-08 evening batch (21:24 export) flipped to mostly SHORTS, all losing. Attribution of the −$734 window: trailing gate = $0 (never fired), gap relaxation = −$89 (1 MARGINAL trade), **rest = pre-existing strategy.** Drilling into the 4 fresh shorts: **4/4 losers, 0% WR, −$577, and ALL 4 fall in cells the removed BTC RSI×ADX SHORT filter would have blocked** — BANK/DASH (BTC RSI 28.7 × ADX 18.0; rule `0-30:25-30` needs ADX 25-30), AAVE/ADA (BTC RSI 30.6/34.7 × ADX 19.8/21.9; rule `30-35:30` needs ADX≥30). Mechanism is textbook: shorting an **oversold BTC (RSI 28-35) with weak trend conviction (ADX 18-22)** = shorting into exhaustion → squeeze; all 4 stopped out (STOP_LOSS / STOP_LOSS_WIDE / PATTERN_FIXED_SL). Exactly the loser zone the filter was built to catch, mirrored short-side.

**Change:** restored `btc_rsi_adx_filter_short` = `30-35:30,35-40:20-26,45-50:25,0-30:25-30` (the archived string). **LONG side left BLANK** — its experiment arm is unresolved AND the lone 06-08 LONG loser (ALLO −$238) shared its BTC cell (RSI 64.7 × ADX 18.4) with 2 winners (ADA, PUMP same minute), so the cross-filter doesn't separate the long loser from long winners; that loss was a high-ATR pair (1.79) × ATR_Runner 2× amplification event (−$119 demux × 2), not a BTC-cross issue. Re-adding LONG would block winners — declined.

**Discipline note:** honored the experiment's pre-committed clause — *"SHORT is the money-maker → tighter watch: if the open SHORT book degrades vs its filtered baseline (WR drop / net-neg on N≥15), re-add SHORT immediately."* N=4 is below the formal N≥15, but 0/4 WR with clean per-cell mechanism attribution = the "draws down past operator comfort" clause; SHORT held to the tighter standard by design. Parser semantics (trading_engine.py:6119): each `RSIlo-RSIhi:ADXlo-ADXhi` rule is an allow-list — block the short when BTC RSI is in-band AND BTC ADX is OUTSIDE the required range. Single ADX value = "requires ADX ≥ that". CURRENT_STATE experiment section updated in place. **Re-evaluation:** if the re-added SHORT filter blocks a would-be-WINNER short (BTC RSI 28-35 × ADX<25-30 that recovers ≥50% WR on N≥6 fresh), revisit the band bounds.

### 2026-06-08 — CAPABILITY: pattern-cell engine generalized (UNMATCHED + combos + block action)

**Context:** truly-unmatched shorts (no C, no W) are a net-loser cohort (9-pool N=23, 43% WR, −$515). Operator asked for a generic way to block/cap/multiply *any* C, W, or combination, both directions — initially to "block truly-unmatched shorts." Investigated: the `pattern_cell_rules` engine already did single-code × direction × {inv_mult, lev_mult, fixed_tp_pct, fixed_sl_pct} and already recognized `UNMATCHED` for *treatment*. Two gaps: combos and a block action.

**Quant decision — do NOT block truly-unmatched shorts.** Against the locked Pattern-C→FILTER gate (N≥30 AND WR≤40% AND Avg≤−0.20% AND NP≥60%) the cohort fails 3 of 4: N=23<30, WR 43%>40%, NP 26%<60% (only Avg −0.222% passes). It carries 7 runners (peak≥0.60); blocking forfeits them. Fix-TP also rejected — sim shows it chops the runners (the cohort's only value) to "save" a minority of poppers (+$115 in-sample but fragile, against "don't chop a winner"). The methodology-correct treatment is a CAP: fix-SL ~−0.50 (sim +$151 in-sample / ~+$90 haircut, kills 1 of 10 winners, targets the non-recovering loser tail). Even that waits for N≥30 on the RESIDUAL after the re-added BTC cross-filter thins the squeeze-shorts (overlap), plus a per-pair concentration check.

### 2026-06-10 — REFINE: RSI-spike guard → require 1-candle jump ≥4 (semantics fix, operator-caught)

**Operator challenge:** "50→51 is not a spike — 50→60 is." Tested both designs on the 11-pool unmatched longs (N=39): (a) the feared false-positive (prev<50 with a small jump) has NEVER occurred — entries need RSI ~54+, so prev<50 mechanically implies jump ≥4; `prev<50` and `prev<50 AND jump≥4` block the IDENTICAL 7 trades (+$554 NET, $60 winner-kill). (b) **Jump size alone does NOT separate**: winner avg jump +5.6 ≈ loser avg +5.8; pure `jump≥5` = NET **−$318** (kills 14 winners incl. NEAR +12.5) — winners accelerate too. The true spike signature is the **sub-neutral origin** (momentum born from below RSI 50 this candle), not the delta. **Change:** added `rsi_spike_min_jump_long=4.0` (0 = pure floor); gate now fires only on `rsi_prev < 50 AND (rsi − rsi_prev) ≥ 4` — historic behavior identical, theoretical 49.8→51 non-spike formally excluded. D11 complete (config/json/engine/UI second input/load/save/report).

### 2026-06-10 — SHIP: Jun-10 guard set (post-crash-batch; 4 changes, operator-directed)

**Context:** first post-reset batch was −$1,101 (−$555L/−$546S, balance →$1,550). Forensics: (a) the unmatched longs that fired were OUT-OF-DISTRIBUTION meme spikes (PIPPIN ATR 1.3-1.4, ESPORTS ATR 4.68 = p100 vs historic max-winner 2.49) in a bear tape with a flickering bull regime flag — the 85%-WR cohort was bull-tape majors, a different population; (b) the risk-stacking identity: high-ATR meme → no guard blocks it + widest ATR-SL (−1.20 floor) + 2× unmatched mult + max DOA risk = avg loss $195 vs avg win $29; (c) shorts were squeezed by a mid-session BTC bounce — all entries had BTC 1h RSI 34-40 (hourly oversold). Full 31-variable winners-vs-losers sweep + wider-SL CF run. **Wider SL definitively rejected:** SL−1.5 = −$2,083, SL−2.0 = −$2,497 vs actual −$1,807 (dumps blow through; the V-bounces require surviving −5/−6% = liquidation at 20×). **Fix-TP confirmed irrelevant** (losers' peaks 0.06-0.17 — never reached +0.25).

**Shipped (D11 complete):** ① `btc_rsi_1h_min_short=35` — NEW filter, the session's one real cross-batch lever (monotonic bands, +$690 ex-06-10, 5/7 dates; mechanism = hourly twin of the climax-oversold block). ② `fan_ratio_block_long` 0.85-1.70→0.85-3.00 — window correction of the existing fan gate; the open 1.70-3.00 window held all 06-10 long losers + 06-01/06-09 losers (N=13, −$1,043); fan SUBSUMES the stretch-cap idea (overlap matrix: stacking stretch on fan = strictly worse). ③ `btc_rsi_adx_filter_long` += `70-100:40` + NEW `pair_atr_max_long=2.5` (counters PAIR_ATR_MAX) — free guards, zero historic winners killed. ④ NEW `rsi_prev_min_long=50` RSI-SPIKE GUARD (counter RSI_SPIKE_GUARD) — operator-chosen despite ~zero marginal $ on top of fan (+$69 all-time/−$38 last-6): justified by unique candle-1 coverage (fan needs candles 2-5 to deform; VVV 44.6→65 was spike-only-caught) and capped cost ($60 lifetime winner-kills). **Stack impact: 06-10 batch −$1,385→−$167 (Δ+$1,218, hindsight-fit); last-6 +$21→+$1,311; HONEST forward expectancy ≈ +$30-50/batch ex-06-10.** Overblocking check: throughput 10.3→~8 trades/session; kept book 65% WR / ~+$28/tr in-sample. Spike+stretch redundancy documented (one detector per phenomenon — fan chosen as primary, spike as candle-1 complement, stretch DROPPED). **Operator declined the UNMATCHED-LONG 2×→1× demote** (its ✗-HARMFUL gate sits at N=4 of 5 fresh fires — one more bad fire trips it by rule). Gates in CURRENT_STATE.

### 2026-06-10 — RE-ADD: LONG cross-filter `50-55:99-100` (watchlist band gate tripped) + REVERT max positions 4→5

**① LONG cross-filter 50-55 re-add.** The 06-10 batch's two unmatched-long losers (MORPHO −$160, CHZ −$274 — both BTC RSI 53.6 × ADX 20.4, both passing keep-only-unmatched) fall exactly in the archived `50-55:99-100` block. Validated per the Jun-5 watchlist's PRE-COMMITTED band method ("re-add only bands that confirm net-negative on fresh data"): **50-55 = 3rd consecutive negative window** (full pool −, last-4 −, fresh unmatched N=12, 67% WR but −$396 — fat-tail R:R losers: CHZ/MORPHO/HEI/HIVE vs 8 small winners). Mechanism: BTC mid-range/directionless (RSI 50-55, that morning literally 58% bear breadth under a flickering bull regime) = no tailwind for a momentum long. **60-100 NOT re-added — its own ship gate FAILED:** among unmatched longs (the only longs trading now), BTC RSI 60-65 = N=7, 86% WR, +$368 (keep-only-unmatched FIXED the overbought-long bleed — it was the matched longs). 55-60 stays open. **Ext floor stays OFF** — 0 unmatched longs below ext 0.20 (cohort moot; the pre-committed "does <0.20 still die?" check has no data). Caveats: N=12, 67%-WR-net-losing shape; mitigated by 3-window sign-consistency + pre-committed gate. **Revert: re-blank 50-55 if would-be-blocked longs ≥55% WR AND net-positive on N≥8 fresh.** Config: `btc_rsi_adx_filter_long` "" → "50-55:99-100".

**② max_open_positions 4→5 (REVERT of yesterday's 5→4).** Operator call: with the UNMATCHED-LONG 2× multiplier, a single position reached ~$2,300 of ~$3k margin (base $631 ×2 — MORPHO/CHZ were $1,160 each) — two bad trades = −$434, an outsized equity hit. The 5→4 ship logged exactly this revert trigger ("cluster drawdowns at the bigger size breach operator comfort"). Concentration-vs-deployment resolved toward diversification: per-position base back to ~$505 (×2 mult = ~$1,010). The +25%-deployment idea remains valid in principle but belongs to the post-forward-test scaling phase (with de-levering), not now. Config: `investment.max_open_positions` 4→5.

### 2026-06-09 — SHIP: max_open_positions 5→4 (+25% per-position capital)

**Evidence:** time-weighted concurrency on the 10-pool + 06-09 partial — all 5 slots simultaneously in use only **5.9%** of open-time historically and **0.3%** under the post-cut config (06-08+); 1–3 concurrent positions = ~94% of open-time. The 5th slot was reserving 20% of capital for a state that almost never occurs after the throughput cuts (keep-only-unmatched longs + short filters).

**Change:** `investment.max_open_positions` 5→4 (one line, trading_config.json). equal_split sizing = (balance − reserve)/max_pos → per-position **$505 → $631 (+25%)**. Max total exposure UNCHANGED (5×20% = 4×25% = 100% of balance when full) — only granularity coarsens. `max_open_positions_hard` (redeploy ceiling 10) untouched; redeploy band now starts at ≥4.

**Trade-off accepted:** per-stopped-trade equity hit ~4%→~5%; a correlated 4-cluster = ~20% equity vs 16% before (mitigated: the Jun-8/9 filters specifically target the correlated-cluster entries). Sizing is %-metric-invariant so the keep-only-unmatched forward-test read gates (WR / Avg% / demux) stay clean; only batch $ totals scale.

**Deferred siblings (memory: project_scaling_roadmap):** de-lever-as-balance-grows + drawdown caps — revisit after the forward test passes. **REVERT: back to 5 if the bot measurably hits the 4-cap and forgoes entries on ≥3 sessions (watch had_room=False blocks), or if cluster drawdowns at the bigger size breach operator comfort.**

### 2026-06-09 — SHIP (in-sample, forward-test): KEEP ONLY UNMATCHED LONGS + disable fix-TP

**Finding (the session's biggest):** the 4-Cohort Pattern Coverage, re-simulated on the 10-pool under the **full current stack** (fix-TP +0.25, EMA13-cross-OFF→post-exit ride, wide-SL −1.00 floor, ATR/cell multipliers — both demux and as-sized), shows the cohort edge is **inverted by direction**:
- **LONG:** CROSSED −$218, C-only −$557, **W-only −$1,210**, **TRULY-UNMATCHED +$471 (N=39, 85% WR)** demux. Every individual long pattern is net-negative (W6 −$574, W2 −$480, C7 −$261, C6 −$297, W3 −$294…); the multiplier *amplifies* the matched losses (≥2× longs −$191 demux → −$398 as-sized) but *helps* unmatched (+$471→+$679). No surgical subset works — blocking the top-4 patterns still leaves −$361. The long patterns are caution/countertrend signatures (C7 bounce, W6 top); the no-pattern cohort is clean momentum.
- **SHORT:** matched cohorts win (W-only +$1,047 as-sized), truly-unmatched loses (−$531) → already optimal (unmatched-short block shipped 06-08).

**Exit refinement:** the unmatched longs RUN — 54% peak ≥0.40, 36% peak ≥0.70. The +0.25 fix-TP (built for the pop-and-fade matched cohort) strangles them. Grid: raising the fix-TP is worse (+0.35→+$294, +0.50→+$158); **disabling it and trailing models +$360→+$3,215 demux** (optimistic trailing model — haircut to ~+$1,500–2,200, still 4–6×). So fix-TP OFF for the unmatched runners.

**Change:** new `long_unmatched_only` toggle (default False) — blocks any LONG with c_any OR w_any (counter `LONG_UNMATCHED_ONLY`), keep only unmatched. Set True. `atr_low_fixed_tp_long_enabled`→False (unmatched longs trail). Combined long book −$1,515 → +$471 demux in-sample. Build (D11): config.py field · trading_engine.py filter in open_position (before pattern-cell lookup; uses _pc_any_e/_pw_any_e) · trading_config.json · UI toggle + load/save + report line. **Architecture note:** chose a dedicated toggle over 15 pattern-cell `block` rows — blanket all-or-nothing strategy fits a single robust switch (auto-covers future patterns, trivial to flip for the forward-test); the pattern-cell block feature stays the tool for *surgical* cuts. They coexist (toggle runs before the cell lookup).

**Caveats:** IN-SAMPLE (10-pool, N=39 unmatched-long winners) + drastic (blocks ~80% of longs) + the trailing/EMA13/SL parts are *modeled* from trajectory data. So **forward-test, not proven.** **READ GATE:** keep if unmatched-long cohort ≥70% WR AND net-positive demux on N≥20 fresh full-stack; revert (toggle off + fix-TP back on) if WR<60% OR net-negative on N≥20. **Coupling logged:** if the toggle goes off, the fix-TP must go back on (matched longs need the cap).

### 2026-06-09 — SHIP (DISCIPLINE-OVERRIDE): BTC cross-filter SHORT `30-35:30`→`30-35:30-32`

**Trigger:** the 06-09 bear batch's short loss was a single correlated cluster — SUI/SOL/AVAX shorts fired 08:18-08:19, all BTC RSI 32 + ADX 34, all W2/W6 3×, all reversed (peak 0.00) → −$318 as-sized (97% of the batch short loss). Investigated W2/W6 history: **W2 SHORT is a durable winner** (cross-batch N=74, 70% WR, +$1,510 / +$585 demux); **W6 SHORT (= BTC ADX≥32, single-axis "mature bear") is a net loser** (N=28, 57% WR, −$220 demux) and W6 ⊆ W2 (every W6 short is a W2 short, so W6 is an ADX≥32 overlay that drags W2 into its loser zone — same shape as C1+C6).

**The cut is an INTERACTION, not either variable.** 2×2 on W2 shorts (demux): ADX<32×RSI<35 **+$151** (80% WR) · ADX<32×RSI35-45 **+$655** (78%) · ADX≥32×RSI35-45 **+$179** (100%) · **ADX≥32×RSI<35 −$399 (N=22, 45% WR)** — only that corner loses. So neither "block RSI<35" (oversold wins at ADX<32) nor "retire W6/ADX≥32" (wins at RSI 35-45) is right; the loser is the climax(ADX≥32)+oversold(RSI<35) intersection = exhaustion-bounce. Within RSI 30-35 by ADX: ADX 30-32 **+$43 demux/69% WR** (winner) vs ADX 32-36 **−$551 demux/47%** (loser) — clean boundary at 32.

**Change:** SHORT cross-filter `30-35:30` (require ADX≥30) → `30-35:30-32` (require 30≤ADX≤32; block ADX>32). Full string `30-35:30-32,35-40:20-26,45-50:25,0-30:25-30`. (RSI 0-30 × ADX≥32 already blocked by the `0-30:25-30` rule; the leak was only RSI 30-35.)

**Impact (cross-batch):** blocks 22 trades (12 losers −$1,442 / 10 winners +$562 as-sized; net −$881 as-sized / −$399 demux). SHORT book **−$172→+$709 as-sized (+$881) / demux +$399**. ~1.7 blocks/active-short-day. Preserves the ADX 30-32 winner band + W6 RSI 35-45 100%-WR cell.

**DISCIPLINE-OVERRIDE:** below the locked block gate (N=22<30, WR 45%>40%). Chose BLOCK over de-mux because the cell is −EV even at 1× (−$399 demux, 45% WR) — de-muxing would leave a residual losing cohort trading; and the block is a one-line config change (no build). Interaction is clean (3 winning cells, 1 loser, sound climax-bounce mechanism, confirmed live by the cluster), which de-risks the override. Considered+rejected: de-mux W2 to 1× in the cell (keeps a −EV cohort trading; needs a build); retire W6 (kills the RSI 35-45 100%-WR winners). **REVERT GATE: re-open the cell (`→30-35:30`) if RSI 30-35 × ADX>32 shorts show ≥50% WR on N≥8 fresh.** Config-only change; CURRENT_STATE updated.

### 2026-06-08 — SHIP (DISCIPLINE-OVERRIDE): block UNMATCHED SHORT

**Operator override.** After I recommended a fix-SL cap (not a block) and flagged that the cohort fails the locked Pattern-C→FILTER gate, operator chose to **block truly-unmatched shorts**: set `block:true` on the existing `{UNMATCHED, SHORT}` pattern_cell_rule (was inert 1×). **This is a below-gate / fails-gate ship** (N=23<30, WR 43%>40%, NP 26%<60% — only Avg −0.222% passes), logged transparently per the override rule. **Accepted costs:** (a) forfeits the cohort's runner upside (~7 runners/23, peak≥0.60); (b) blocking renders the cohort UNMEASURABLE going forward — unlike the fix-SL, which would have kept it trading and visible. **Overlap:** the re-added BTC cross-filter already blocks the RSI 30-35 squeeze unmatched-shorts; this block additionally removes the RSI 38-42 residual (UNI/AVAX zone the BTC filter misses) + unmatched shorts in any other RSI zone. **TIGHTER-THAN-STANDARD REVERT GATE:** (1) re-open (block→false) for a 1-batch measurement window at the next ≥30-trade review — if the cohort shows ≥45% WR OR net-positive demux on N≥10 fresh → remove block (or downgrade to `fixed_sl_pct:-0.50`); (2) remove immediately if PATTERN_CELL_BLOCK >15% of SHORT attempts (over-block) OR short throughput thins past operator comfort. Watch the `PATTERN_CELL_BLOCK` counter (LONG should stay 0 — rule is SHORT-only; UNMATCHED LONG remains a 2× multiplier, untouched). One-line config change (`block:true`); no code change (capability already shipped above).

**Build (capability only, no rule populated):** `_lookup_pattern_cell_rule` (trading_engine.py:1790) — added `_rule_side_and_match` helper: `UNMATCHED` = no C & no W; combo `A+B` = AND of all parts; mixed C+W combo resolves to the C side (C-blocks-W). Added `applied_block` accumulation; return is now a 6-tuple `(inv, lev, source, tp, sl, block)`. Call site (open_position ~2851): if `block` → log + `_record_filter_block("PATTERN_CELL_BLOCK", dir)` + `return None` (before sizing/Order creation/exchange call — clean skip). config.py: documented combo syntax + `block` field. UI: pattern cell now a free-text input + datalist (accepts `UNMATCHED`/combos), new **Block** checkbox column, load/collect updated. Standalone logic test passed 6 cases (UNMATCHED fires only when truly unmatched; combo needs all parts; block propagates; direction-scoped; existing C1/C4/C8 singles unaffected). `trading_config.json` pattern_cell_rules UNCHANGED — capability is inert until a rule is added. Watchlist gate for the eventual UNMATCHED-SHORT fix-SL logged in CURRENT_STATE.

### 2026-06-10 (evening) — SHIP: multiplier-audit package (5 changes) + `!` negation in pattern-cell engine + ATR multiplier dimension REMOVED

**Trigger:** before shipping the 55-60 cross-filter rule + W2→W2+W1 cell change, operator asked to audit ALL multiplied cells for the same "value lives in the combo, not the base pattern" disease. Audit (FULL pool + 06-10 CSVs, deduped, demux):
- **C1 SHORT 3× — KEEP.** C1-alone = N=35, 77% WR, +$154 demux. The C1+C6 contaminant (N=8, 50% WR, −$174 demux) was found to be **already 100% fenced**: overlap audit shows all 8 historical C1+C6 shorts entered at BTC RSI 22-35 / 1h-RSI <35 → every one is now blocked by the SHORT cross-filter + `btc_rsi_1h_min_short=35`. C1+C6 block-gate RESOLVED MOOT (no pattern-level block shipped; would be dead code).
- **W6 SHORT 2× — DELETED.** Cell cohort (W6-no-C) N=10, 50% WR, −$37 demux; split: with-W1 +$24 (80% WR) vs without-W1 −$61 (20% WR). Same W2-finding anatomy. Winners auto-covered by the new W2+W1 rule (W6 ⊆ W2).
- **W6+!W1 SHORT → BLOCK (new rule, needed `!` negation).** Mechanism-aware re-sim under TODAY'S stack: the only winner (PHA +$169) is already blocked by the 1h guard; survivors = N=4, 0% WR, −$303 across 3 dates → clears the ≥3-sample direction-consistent bar. Theoretically coherent: macro-bear tag (BTC ADX≥32) with no pair momentum (no W1) = squeeze-prone macro-only short. REVERT GATE: remove if PATTERN_CELL_BLOCK(W6+!W1) >10% of SHORT attempts.
- **W2 SHORT cell → `W2+W1`** (keeps 2×/1.5×). W2+W1 = +$1,480/79% WR (N=14) vs W2-alone −$123 demux (N=16, incl fresh XLM −$210). READ GATE: N≥5 fresh fires, ✗ HARMFUL → demote.
- **ATR Multiplier dimension REMOVED end-to-end** (operator-directed; was `Runner` ATR 1.1-99 2×, Jun 5). Refuted: all-time guard-stack survivors in 1.1-1.5 = N=14, 36% WR, −$328 demux (worst ATR bucket); current-era survivors >0.8 = N=0 (ATR cap + fan + spike guards fence the zone); also fully redundant under keep-only-unmatched (UNMATCHED cell already gives identical 2×/1×, max-wins). Removed: engine `_lookup_atr_multiplier` + candidate, `config.py` field, `trading_config.json` rules, `main.py` `_compute_atr_multiplier_performance` + payload wiring (3 spots), UI rules table + performance table + render + 2 exports + load/save lines. Removal protocol: repo-wide grep clean; py_compile OK.
- **LONG cross-filter → `50-55:99-100,55-60:20-25,70-100:40`.** Adds the validated 55-60:20-25 rule (kills exactly the 2 fresh losers 1000PEPE −$123 + CHZ −$103 at ADX<20, zero winners). DROPS the two 60-65 rules that existed as **UI-only config drift** (net −$21, killed a +$58 winner). 3-rule set dominates the UI 5-rule set: blocks 16 vs 28, kills $146 vs $412 of winners, removes −$1,020 vs −$982. **UI must be synced post-deploy (delete the two 60-65 rows).**
- **UNMATCHED LONG 2× — gate TRIPPED, operator OVERRIDE (kept).** Fresh post-reset N=7, 14% WR, −$546 demux trips the locked ✗-HARMFUL gate (N≥5, Total$<0); robust to the new filters (excl. now-blocked: N=5, −$658). Operator explicitly declined the demote (2nd time, post-trip) — logged as a formal gate violation on operator authority. Replacement gate (tighter, locked): next N≥5 fresh fires under the Jun-10 guard set also net-negative demux → demote to 1× WITHOUT FURTHER DEBATE.
- **Deep-dive (unmatched longs, all axes):** no axis qualifies for a multiplier under the locked promote gate. ATR>1.1 hypothesis refuted (above). BTC RSI non-monotonic noise. Only consistent warm zone = **BTC ADX 20-25 (×RSI 60-70)**: current-era survivors 83% WR +$199 (N=6), all-time 55% WR +$90 (N=124, only positive band) — fails N/WR gates → WATCHLIST with promote gate (WR≥70% + Total$>0 on N≥20 fresh survivors). Equals the existing-but-shadowed BTC cell 60-65×22-25.

**Engine build:** `_rule_side_and_match` extended — `!` prefix negates a part (`W6+!W1` = W6 AND NOT W1); all-negated patterns refused; side from positive parts; C-blocks-W priority means W-side negation rules never fire on C-matched trades (matches the no-C cohort definition the evidence came from). 9-case standalone logic test ALL PASS. UI: pattern-cell help text + placeholder document the syntax (free-text input already accepts it).

### 2026-06-12 — SHIP: pair universe 50→75 + entry_pair_rank tracking + Performance-by-Pair-Rank table

**Trigger:** operator asked where profitable volume could come from without relaxing filters. Boundary audits killed the filter-relaxation candidates (PAIR_ADX_MAX: LONG monotonic decay into the cap, zero data >30, SHORT >35 = N=13 losing + zero current-stack survivors; PAIR_RSI_RANGE LONG 65-70: 18-42% WR, ZERO trades ever peaked ≥1% — flat-liners, not chopped runners). The clean volume lever = universe expansion.

**Evidence for 50→75 (Tier A, rank 51-75 ≈ $49-82M 24h vol):** pool trades at $50-80M vol behave like core book both sides; the bad cohorts live below $50M and only on the LONG side (39% WR) — and the existing global-vol rescue line ($50M, LONG-only, rescue_max 0.6) auto-fences sub-$50M longs in quiet tape. Liquidity caps (0.1%/24h-vol) never bind ≥$49M vs max desired notional $36k. Tier B ($27-49M) NOT opened: long-hostile + caps pinch; Phase-2 idea = shorts-only via $50M long floor.

**API-safety audit (the old 100-pair crash was a REST rate-limit ban, pre-batching era):** at 75 pairs ≈195 weight/scan over ~60-90s cycle ≈ 8% of Binance's 2,400/min; protections now: OHLCV batches of 10 with 5s delay, ccxt enableRateLimit, ban-detect+sleep-until-expiry+DB-persist, scan-loop exponential backoff. WS: one combined connection, 75 @trade streams ≪ 200 cap. Cost: scan sweep +15-20s.

**Build:** ① `entry_pair_rank` column (models.py + database.py auto-migration; CSV export picks it up automatically via model-column introspection) stamped in scan BEFORE blacklist removal (rank = position in eligible top-N) and threaded scan→_collected→open_position→Order. ② `/api/config/pairs-limit` validation: 100 replaced by 75 (operator-directed). ③ UI dropdown: "Top 75" added as new default. ④ `_compute_pair_rank_performance` (buckets 1-10…61-75,>75) + payload + 🏅 UI table (new-tier rows highlighted amber ✦) + both text-report exports. ⑤ trading_config.json `trading_pairs_limit` 50→75.

**READ GATE (locked): N≥20 closed trades at rank 51-75 → compare WR/Avg% vs rank≤50; materially worse → revert limit to 50.** Bonus: rank structure inside the old universe (1-20 vs 21-50) measurable for the first time.

### 2026-06-12 (later) — REVERT: pair universe 75→50 (same-day; audit blind spot, operator-caught)

**What went wrong with the 50→75 ship:** the expansion audit ranked the RAW Binance volume list (raw rank-75 ≈ $49M = "Tier A"), but `get_top_futures_pairs` applies the 180-day new-listing filter (~47 of the highest-volume symbols excluded — recent listings dominate volume) + alpha-subtype filter BEFORE the top-N cut. The bot's ELIGIBLE rank-75 therefore reached **$24M** (operator observed $21M live), squarely in the $27-49M tier the same audit had classified long-hostile (39% WR) + liquidity-cap-pinched. Even eligible rank-50 dips to ~$32M — the old Top 50 was always touching the $30-50M zone (which is also where that tier's training data came from, so status quo restored = no harm done). Net exposure window: ~2h at limit 75.

**Lesson (methodology):** when auditing universe/eligibility questions, reproduce the bot's OWN selection pipeline (eligibility filters → rank), never the exchange's raw ranking. Blocked/excluded cohorts must be measured through the system's lens.

**Reverted:** `trading_pairs_limit` 75→50 (UI immediately by operator, then git), dropdown default back to Top 50 (75 stays selectable). **Kept:** `entry_pair_rank` column + 🏅 Performance by Pair Rank table — buckets restructured to 1-10…41-50 + `>50` catch-all (the 51-60/61-75 rows could never populate at limit 50). **Parked:** ① `min_pair_volume_usd` ≥$40M eligibility floor; ② new-listing 180→90 step-down + `entry_pair_age_days` instrumentation (NOT removal — short-side meme-squeeze tail uncovered: ATR cap LONG-only, short guards BTC-level; ruin-risk asymmetry rules).

### 2026-06-12 (evening) — NO-SHIP ×4: universe/volume/age boundary exploration closed (all rejected by data)

**Context:** operator asked where profitable volume could come from; full boundary exploration ran same-day. Four candidates, four rejections, zero shipped — logged so none get re-litigated without NEW data:

1. **Pair universe 75 — REVERTED same day** (separate entry above): eligible rank-75 = $24M after the 180-day new-listing filter (~47 high-vol pairs excluded pre-cut). Audit lesson: reproduce the bot's own selection pipeline, never raw exchange rankings.
2. **New-listing filter 180→90 days — REJECTED.** Today's snapshot: the 90-180d window holds 30 pairs, exactly ONE would crack the eligible top-50 (SPACE, $85M, 25.6% day range vs 9.5% median for established ≥$30M pairs — ~3× wilder than the calibration universe). Decisive: our actual burns were OLD pairs (ESPORTS ~290d, PIPPIN ~475d, VVV/TRUMP ~450-480d at trade time) — age was NEVER the protective mechanism; the behavioral guard stack (ATR cap, fan, spike, unmatched-only) is, and it's age-blind. The filter's real job = keeping the <90d casino out; both settings do that identically. KEEP 180.
3. **`pair_atr_max_short` — REJECTED (dead code).** 525 historical shorts: ZERO above ATR 2.0, four above 1.5 (net positive); current-stack book shorts max ≈1.25. Structural reason: a short requires bearish EMA stack + RSI 25-50 + pattern match — a squeezing/pumping meme cannot generate a short signal. The LONG ATR cap was necessary (you CAN chase a pump long); the SHORT mirror has nothing to block. Post-entry squeeze tail is bounded by the ATR-widened SL floor (worst trough in 70 book shorts = −1.19% vs floor −1.20).
4. **`min_pair_volume_usd` ≥$40M floor — REJECTED (dead code).** Sub-$40M entries under the current stack: longs 2/134 (1%, −$39), shorts 1/70 (1%, +$23) — three trades, net −$16. Mechanism: bottom-of-universe pairs almost never produce stack-passing signals (gap/expanding/quality need real participation) — the boundary self-protects. The "long-hostile $30-50M" evidence is mostly the $40-50M band, which a $40M floor wouldn't touch anyway.

**Carried forward:** live-mode gap-through-SL tail at 20-30× = a leverage/sizing question → de-lever phase of the scaling roadmap (memory: project_scaling_roadmap). **Kept from today:** `entry_pair_rank` column + 🏅 Performance by Pair Rank table (buckets 1-10…41-50, >50) — first visibility into rank structure inside the Top 50.

**Meta-conclusion:** the current stack + Top-50 is self-protecting at every boundary probed (volume, rank, age, short-ATR). The volume edge is not hiding behind a universe knob; the forward test stays clean and untouched.
