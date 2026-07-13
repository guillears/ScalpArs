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

### 2026-06-12 (night) — INSTRUMENT: Leash Shadow table repurposed to ALL SHORTS / ALL LONGS + locked SHORT exit-capture gate

**Trigger:** operator flagged the Post-Exit Regret table — winners exit and the move continues (avg post-exit peak +1.3 to +3.3% on trailing exits); with few trades/day the per-winner capture matters. Full leash-shadow read (the 9 virtual exit policies that keep tracking past the actual close):
- **FULL armed pool (N=200, mixed eras): misleading** — every leash hurts LONGS (−12 to −30pp) because that cohort is dominated by pre-keep-only-unmatched matched longs (faders). Cohort discipline applied.
- **Current-stack book shorts (N=20): EVERY leash beats actual.** stren Δ+7.05pp · wide +6.41 · strpk03 +6.33 (16 better/4 worse) · tight-sanity +5.26 (16/4). Recent era (Jun 9+, N=8): tight/strpk/tierA/tierB beat actual 8/8. Mechanism: C1/W2 shorts are capitulation cascades; live exits (trailing L1 + EMA13-short cross) fire on the first micro-bounce minutes in; the cascade continues. Even the 0.25-flat sanity leash beats the live chain → the leak is the exit CHAIN, not just trail width. Uplift likely UNDERSTATED (several leashes still open at tracking-window end). In $: ≈+$300-500/3 days at current sizing (shorts carry 3× multipliers).
- **Current-stack unmatched longs (N=9 / recent N=3): all leashes ≤ actual.** Long trail already optimal — confirmed do-not-touch (consistent with Jun-6 runner-trail wash and the ADA shadow read).

**Build (operator-directed repurpose):** `_compute_leash_shadow` BUCKETS — stale runner-era slices (LONG RUNNER ATR≥1.0, LONG <0.25-stretch control) replaced with ALL SHORTS (gate slice, drill) + ALL LONGS (baseline-expectation control); LEASH set re-expanded to tight/wide/tierA/tierB/strpk/strpk03/stren/strpk_signed (strpk04 dropped, redundant). UI header/title/legend rewritten; gate printed in amber in the table description. Verified on the live batch: 6 armed shorts → strpk Δ+3.68pp/+$1,022 CF, 5 of 6 leashes ★ helps; longs ✗/marginal as expected.

**PROMOTION GATE (locked): at N≥30 armed SHORTS — ship the best leash as the live SHORT exit policy IF Δ≥+0.15pp/trade AND Clean:Trap ≥2:1.** Candidates ranked stren / strpk03 / wide. ETA ~1-2 weeks of fires; zero new instrumentation needed — every closed short adds to the read.

### 2026-06-12 (night, follow-up) — SHIP (DISCIPLINE-OVERRIDE): Runner Stretch-Trail SHORT (shadow-strpk promoted early)

**Operator override.** The locked gate said promote at N≥30 armed shorts; operator shipped at N=20 ("results too conclusive") — logged transparently per the override rule. Evidence at ship: shadow strpk vs actual on current-stack book shorts = Δ+5.06pp/+$996 (N=20, 13/7 better); recent era (Jun 9+) 8/0 better +$979; live batch +$1,022/6 armed shorts; every one of 8 leash variants positive on shorts (stren best $, strpk best recent consistency); leashes are tick-level forward sims (first-trigger-locks, no lookahead; verified _leash_update call sites: live monitor tick + post-exit continuation with EMA13/signal-lost/hard-SL backstops). strpk chosen over stren: existing engine mechanism (Jun-1 runner trail = strpk mechanic), 8/0 recent, operator named it.

**Build (D11):** per-direction generalization of the Jun-1 runner trail. config.py + trading_config.json: `runner_trail_short_enabled=true / _atr_min=0.0 / _arm_peak=0.45 / _k=0.5` (params MUST mirror the measured leash sim: ACT 0.45, no ATR gate). indicators.py: `_runner_armed` + main handoff both direction-aware (atr_min<=0 = gate off); RUNNER_TRAIL log direction-aware. trading_engine.py: realtime tight-trailing suppression direction-aware; **EMA13-short cross suppressed once runner-armed — flips to the existing phantom path** (`phantom_ema13_cross_pnl` recorded, `[EMA13_RUNNER_SUPPRESS]` log) — without this the live version would NOT reproduce the sim (its uplift comes from riding through the first cross). UI: SHORT row in the Runner Stretch-Trail box (4 inputs, load/save, grep-verified 3 refs each); Leash-table header now marks it SHIPPED + revert-monitor. LONG side untouched (OFF, all leashes ≤ actual).

**Known divergence:** live peak-stretch tracked from ENTRY vs shadow's post-arm tracking → live can exit marginally earlier (conservative bias; same semantics as the validated Jun-1 LONG build).

**REVERT GATES (tighter than standard):** ① cumulative (actual − shadow_tight) < 0 on N≥8 fresh armed shorts → OFF (shadow_tight = old-policy proxy, still recording). ② 2 armed shorts converting ≥+0.45 peaks into hard-SL losses → instant review. ③ strpk leash row must ≈ actual once live (sanity); persistent gap = implementation diverges from sim.

### 2026-06-13 — SHIP (DISCIPLINE-OVERRIDE): ATR×GAP LONG block (resolves the high-ATR contradiction)

**Trigger:** ENJ −$253 in 57s (unmatched long, peak 0.0% — never green). Operator flagged the apparent contradiction: we'd historically found ATR>1 GOOD for longs (the ATR_Runner thesis — high-ATR reach the trailing arm, higher peaks), now calling it bad.

**Resolution (the key finding):** high-ATR is high-VARIANCE, not directional. Split high-ATR unmatched longs by trend-extension (pair (EMA13-EMA50)/EMA50 gap), and they separate cleanly: ATR≥1.0 × gap<0.5 = N=14, **64% WR, ~breakeven→+** (FULL) / N=4, **75% WR +$81** (recent) = the genuine runner, fuel for a move just starting — PRESERVED. ATR≥1.0 × gap≥0.5 = N=16, **31% WR, −$611 demux** (FULL) / N=5, **20% WR −$414** (recent) = volatile pair already extended above trend = buying the exhaustion top → reverts. The gap is the cleanest 2nd axis (tested ADX-delta −333/−319, ADX, RSI, stretch, range-pos — none separate the dollars; gap = +$15 vs −$611). Second-order: the ATR-widened SL (−1.20 vs −0.70 base) makes each loss bigger — but that widening is CORRECT for the good runners (they dip before running); the fix is at ENTRY not exit.

**Orthogonality check (operator question):** removing the bad quadrant rehabilitates ZERO banned LONG pattern — re-ran all 17 C/W cohorts, every one stays net-negative after the filter (any-C −$2,599→−$2,655, any-W −$2,514→−$2,187, W6 unchanged −$1,272). The quadrant is only 0-25% of each pattern and lives almost entirely INSIDE the unmatched cohort (unmatched −$1,449→−$838). So this filter is ADDITIVE to keep-only-unmatched, not a pattern-rehab lever; keep-only-unmatched stays correct.

**Build (D11):** config.py `atr_gap_block_long_enabled/_atr_min_long(1.0)/_gap_min_long(0.5)` + evidence comment; trading_config.json (enabled=true); engine block right after PAIR_ATR_MAX (counter `ATR_GAP_LONG`, `_record_filter_block` + `_last_pair_block_reason`); gap recomputed at the filter from indicators ema13/ema50 EXACTLY matching the entry_pair_ema20_ema50_gap_pct field formula ((EMA13-EMA50)/EMA50*100); UI guard row (toggle + 2 inputs, load/save, 3 refs each grep-verified). Verified on the live batch: blocks ENJ (ATR 1.24 × gap +0.94, −$253), passes all 3 winners (PUMP/TAO/FET, ATR 0.28-0.44).

**DISCIPLINE-OVERRIDE:** N=16 full / N=5 recent in the block zone, below the N≥30 filter gate (WR 20-31% clears ≤40%; mechanism clean + winner-preserving + fresh −$253 example justify the override). Precision ~80% (clips HOME +73 / 5 recent). **REVERT GATE: disable if would-be-blocked longs ≥50% WR on N≥8 fresh.** Batch saved (orders_2026-06-13_4L_0S, −$173); pools → 13batches (362) + FULL (1,273). Operator resets after deploy.

### 2026-06-13 (later) — SHIP: Pair Trend Filter split per-direction, SHORT gap≥0 block re-enabled

**Trigger:** analyzing the ATR×gap LONG finding's mirror for shorts. The ATR axis is moot for shorts (structural — short entries exclude high-ATR pairs, N=2 book shorts ≥ATR 1.0). But the GAP axis showed the inverted mirror: book shorts by trend-extension — gap −0.8..−0.5 (breakdown confirmed) = 95% WR +$476; **gap 0..+0.3 (at/above trend) = 0% WR −$247**; gap −0.2..0 (mild) = 43% WR −$192. Longs enter early (not extended up); shorts enter late (already below trend) — same mechanism, opposite sign.

**Found the filter already existed but was OFF + bidirectional.** `pair_trend_filter_enabled` (May 7, blocks LONG when EMA13<EMA50 AND SHORT when EMA13>EMA50) was disabled in current config. The gap≥0 short losers are all May8–Jun1 (the off-era). Could NOT just flip it on: bidirectional → would also block gap<0 unmatched longs = N=67, 58% WR, −$27 (≈breakeven, fine). So split it.

**Build (D11):** config.py — retired `pair_trend_filter_enabled` into `pair_trend_filter_long_enabled`(False) + `pair_trend_filter_short_enabled`(True) + `pair_trend_short_gap_max`(0.0, the SHORT block threshold, parameterized so the watchlist −0.2 tightening is a config change). Engine: per-direction gate; SHORT now blocks when `gap >= short_gap_max` (was hardcoded EMA13>EMA50). trading_config.json updated. UI: single toggle → LONG/SHORT toggles + gap-max number input (load/save, 3 refs each grep-verified; old id fully removed). Counter unchanged (PAIR_TREND_FILTER).

**Watchlist:** tighten `pair_trend_short_gap_max` 0 → −0.2 when the −0.2..0 zone confirms ≤35% WR on N≥30. **Revert:** re-disable short side if would-be-blocked (gap≥0) shorts ≥50% WR on N≥8 fresh. Note: C1 (capitulation) shorts already require pair_gap≤−0.50 so can't land in the blocked zone; the leak this closes is W-pattern shorts near/above trend.

### 2026-06-13 (night) — SHIP: Phantom Flip Tracker (observation-only "fade the block" instrumentation)

**Trigger:** operator proposed (3×, across fan-ratio / ATR×gap / pair-trend) that since these filters cleanly mark loser entries, we should FLIP them — block-long → enter short, block-short → enter long. The trough/peak data supports a directional tilt (ATR×gap-block longs: avg trough −0.76% vs peak +0.39%; ENJ fell −1.21% after block). But the naive sign-flip P&L (+3-4pp) is a mirage (MAE not realized) and faces structural incoherence (our short system is cascade-continuation w/ stretch-trail + 3× multipliers — wrong machinery for a mean-reversion fade) + high-ATR loser tail (HOME rose +1.85% → a flip-short eats its SL). So instead of trading the mirror, INSTRUMENT it: measure realized flip P&L with a real exit.

**Build (observation-only, fail-silent, modeled on the leash system):** `models.py` PhantomFlip table (auto-creates via create_all). `trading_engine.py`: module-level `_PHANTOM_FLIP_STATE` + `_seed_phantom_flip()` (de-duped — the block filters re-fire every scan cycle a pair sits in the zone, so 1 phantom per pair|source per 30min cooldown) seeded at the 3 block sites (FAN_RATIO_GATE both dirs, ATR_GAP_LONG, PAIR_TREND_FILTER short); `async update_phantom_flips()` called each monitor tick (1s) — prices each phantom from the ws feed, applies base SL −0.70 / arm +0.45 / trail −0.25 / 45min horizon, persists realized P&L per isolated DB session. main.py: monitor-loop call + `_compute_phantom_flip_performance(db,is_paper)` aggregating by source×flip_dir (N/WR/avg%/total%/SL-rate/peak/trough/verdict) + payload key. UI: 🔄 Phantom Flip Tracker table + render. Unit-tested the exit model: ENJ-flip-short → +0.96% trailed (NOT the mirror +1.21%), HOME-flip-short → −0.70 SL — exactly the realistic picture the mirror hid.

**READ GATE: N≥30 per source×flip cell. ★ flip pays (avg≥+0.10%, WR≥50%) → design a proper mean-reversion sleeve (own quick-target exits), NOT a filter-flip. ✗ whipsaws → delete the tracker (grep PHANTOM_FLIP).** Granularity caveat logged. Zero live-trading impact.

### 2026-06-13 (night, follow-up) — Phantom Flip: + BTC_RSI_ADX_CROSS source (extremes only)

Operator extended the fade idea to the BTC RSI×ADX macro cross. Added it as a 4th phantom-flip source, EXTREMES ONLY: BTC RSI≥70 long-block→fade short, ≤35 short-block→fade long (mid-RSI cells are directionless, skipped). Flagged as a MACRO/correlated bet (BTC state is the same for all pairs → a blocked cell flips a basket, not independent trades) — read separately from the pair-level reversion fades. Analysis lean: the short-cross flip (oversold→long bounce) is the cleaner candidate; the long-cross flip (overbought→short) fights a strong BTC uptrend (ADX≥40) and is riskier — but we MEASURE both rather than decide. Engine seed at the BTC_RSI_ADX_CROSS block site gated on the extreme; compute sources list + UI text updated. **Deferred the "solely-blocked-by-X" semantics refactor** (continue-past-block to confirm no later filter also blocks) — too invasive for an observation tool; the proxy ("blocked by X, passed all prior filters") is acceptable for these late-stack filters, and the UI now labels it honestly. Read gate unchanged (N≥30/cell; ★→mean-reversion sleeve, ✗→delete).

### 2026-06-13 (night) — Phantom Flip: + LONG_UNMATCHED_ONLY source (matched-long → short, the strongest candidate)

Operator extended the fade test to the pattern cohorts. Proxy ranking across all flip candidates: MATCHED LONGS → SHORT is the standout by far — N=271 (vs 7-16 for the filter cohorts), +0.142pp/trade proxy estimate, coherent mechanism (matched longs are countertrend/exhaustion: C7=dead-cat-bounce-long, W6=BTC-top — they fail as longs because the pair reverses; the short catches the reversal). C7 sub-cell sharpest (+0.259/trade, N=41 = short the failed dead-cat bounce, textbook continuation). UNMATCHED SHORTS → LONG rejected (N=327 but origWR 60%, edge +0.008 ≈ wash). Seeded `_seed_phantom_flip(pair, current_price, "LONG", "LONG_UNMATCHED_ONLY")` at the keep-only-unmatched block in open_position (fires on matched longs → flip SHORT). Naming note: the filter is named for what it KEEPS (unmatched) but FIRES on what it drops (matched) — the block event IS a matched long. Compute sources + UI updated. Caveat unchanged: proxy overstates (ENJ −1.21 trough → +0.96 realized); a flip-short on a matched long needs its own mean-reversion exits (not the cascade machinery). READ GATE: N≥30 realized per cell; ★ → build a mean-reversion short sleeve, do NOT live-flip on proxy. The N=271 base means this cell accumulates tracker N fast.

### 2026-06-17 — FIX: Entry Funnel all-zeros (Counter NameError) + Opened-NORMAL/FLIP direction split

**Built (earlier same day):** the Entry Funnel diagnostic — proves the bot opens only flips because the filter stack rejects momentum signals (not a bug, not the 5-position cap). Split "Opened — NORMAL/FLIP" and "Blocked by filter" by direction (L/S) via a single `Counter` pass over `all_orders` in `_compute_performance` (main.py ~6795); UI cells + both text exports. **Bug (operator caught — "seems bugged", dashboard all-zeros):** `Counter` was used but only imported locally in OTHER functions (8876/9125), never in `_compute_performance` → `NameError` swallowed by the funnel's `except` → all-zeros default while the Filter Blocks table (separate `_get_filter_block_summary` call) correctly showed ~250 blocks. Exactly the D11 "all-zeros dashboard = suspect a perf-compute exception" failure mode in our own rules. Fix: `from collections import Counter` inside the try (commit 89f0ce8). Verified the funnel then yields Opened FLIP N(0L/NS), Opened NORMAL 0. Funnel split commit 61c34d3.

### 2026-06-17 — SHIP (DISCIPLINE-OVERRIDE): 2D flip-SHORT regime×ADXΔ entry filter (fulfills the standing universal-regime-gate)

**Trigger:** fresh batches opened all-flip-short and bled in bull/chop (regime had flipped from the bear batches where flip-shorts pay). Operator: ship the regime gate the standing watchlist called for.

**Finding (deduped pool 76+39+11, flip-shorts, key=opened_at|pair|direction):** the dividers are ORTHOGONAL — ADXΔ sign splits ~50/50 WITHIN each regime family, so the 2D intersection sharpens both. 2D cells: BEAR∧ADXΔ≥0 = N=25/68%WR/+$1490 (golden, future multiplier); BULL/CHOP∧ADXΔ<0 = N=38/40%WR/−0.34%/−$1070 (block). ADXΔ≥0 is the better SINGLE divider (44%→63% WR swing) but neither 1D passes the WR≤40% filter gate alone (both dilute) — only the intersection does. **NP-gate caveat resolved:** literal Never-Positive on the block cell = 13% (≪60% gate) — but 96% of its losers peak BELOW the 0.45 arm, so the give-back cap provably can't save them → it IS entry-territory, not exit. Per-pair concentration 43% (dimensional, not a pair-blacklist). Counterfactual: last(bear) batch −$63 [filter dormant — bear excluded by design], current(bull) batch −$611→−$5.

**Build (D11):** config.py `flip_short_regime_block_adxd_max(0.0)` + `flip_short_regime_block_regimes("STRONG_BULL,HEALTHY_BULL,CHOPPY_FLAT")` (empty regimes = OFF, no boolean); trading_config.json; universal block at the TOP of `_flip_filters` (reason `FLIP_SHORT_REGIME`, rides `_record_filter_block`); `_ff_in` gains flip_dir/adx_delta/btc_regime (regime classified from globals if unrecorded); UI 2 inputs in the FLIP panel (later merged to one full-width row for clarity). Verified the live filter blocks exactly the 8 bleeders / keeps the 3 on the current batch. Commit 43653da; UI one-line fix + per-source watchlist 99d48d3.

**DISCIPLINE-OVERRIDE** (NP literal 13%<60%, overridden by arm-saveability). **TIGHT REVERT: re-open the blocked cells if they flip to WR>45% on N≥15 fresh.** **PER-SOURCE WATCHLIST (operator-flagged):** the −$1070 cell is 97% FAN_RATIO_GATE (N=37); PAIR_RSI_OB has N=1 in-cell and it WON (+$21, net-positive pool-wide N=15/80%WR); LONG_UNMATCHED zero data. Universal is mechanism-justified but data-unproven for non-FAN. GATE: scope to FAN-only if PAIR_RSI_OB bull/chop∧ADXΔ<0 stays net-positive/WR≥55% on N≥10-15.

### 2026-06-17 — SHIP: passthrough-long tracker (bull-mechanism hunt, observation-only)

**Concept:** the flip-short fade is a BEAR mechanism (fade a wrong signal). The BULL mechanism is the OPPOSITE of fading — the engine WANTS to go long (311 blocked longs/batch) but the macro gates kill them; if those blocked longs would WIN as longs in bull, the fix is to RELAX the gate (re-enable normal longs), not spin up a flip sleeve. Literal short→long mirror is starved in bull (≈no short signals to fade). So instead: PASSTHROUGH — seed the blocked long SAME-direction.

**Build (zero schema change):** `_seed_phantom_flip` gains `mode='FADE'|'PASS'` (PASS → flip_dir = blocked_dir). Seeded as `PASS:<filter>` at the 3 Tier-1 macro long-block sites (BTC_ADX_GATE_LOW, BTC_RSI_ADX_CROSS, FAN_RATIO_GATE — the last is a free A/B vs its existing fade). Tier selection: macro gates only (high-N, the over-block hypothesis); excluded structural/safety + tiny-N + pair-setup filters. PASS rows are EXCLUDED from the fade aggregates (main.py split `_all_flips` vs `flips`) and route ONLY into the existing Source×BTC-Regime cross-tab as LONG rows — rides existing D12 surfaces. Commit 43653da. **Early read (small N, single window): NEGATIVE so far** — PASS:FAN_RATIO_GATE LONG 18%WR/−0.41, PASS:BTC_RSI_ADX_CROSS 50%/−0.11 → macro gates look correctly conservative; no bull edge to harvest yet. Refinement: BULL xtab column later split STRONG vs HEALTHY (commit e211f42) so a strong-bull edge isn't hidden inside a weak-bull average.

### 2026-06-17 PM — CONFIG: disable LONG_UNMATCHED_ONLY flip (live off, phantom-only)

**Evidence:** live N=0 both recent batches (structurally starved — its trigger, a matched long surviving to the keep-only-unmatched stage, is rarely reached because longs die at the upstream macro gates first); phantom N≈8/38%WR/−0.190% ✗ whipsaws (the weakest of the 3 sleeves). Operator: not enough good data to keep it active. `flip_entry_sources` → `"FAN_RATIO_GATE:1.0,PAIR_RSI_OB:1.0"` (config.py + trading_config.json). The phantom seed (line 3327) is DECOUPLED from the registry (always fires) so the tracker keeps accruing for a future re-enable. **RE-ENABLE GATE: phantom WR≥55% AND net-positive on N≥20.** Verified: registry excludes it, `_flip_active("LONG_UNMATCHED_ONLY")`=False, phantom seed still fires. Commit e211f42.

### 2026-06-17 — WATCHLIST: flip-SHORT liquidity×ATR squeeze-gap guard (the SKYAI tail)

SKYAI flip-short (FAN⊘LONG→SHORT, STRONG_BEAR, 2× cell) lost −1.21%/−$253 in 44s: shorted a small-cap ($92M vol, rank 32) high-ATR (1.49%) parabolic-fan (2.76) alt that squeezed +1.14% straight up — never armed (peak 0.0), and the −0.70 SL GAPPED to −1.21 on the fast move. Signature = small-cap + high pair ATR + parabolic fan → squeeze gaps the monitored stop. No live guard catches it (FLIP_FAN_SPIKE is ≥5 and the 2-5 band is a kept edge; meme guard is LONG-only & ATR 1.49<2.5; 2D regime filter correctly exempts STRONG_BEAR; give-back cap can't help never-armed). PROPOSAL (when N accrues): flip-SHORT entry guard on pair vol < ~$100-150M AND pair ATR ≥ ~1.2-1.5% (block or down-size-to-1×), NOT on fan alone. N=1 fat-tail, NOT shippable. Recorded in CURRENT_STATE with tracking plan (stop-overshoot by pair-vol×ATR; ship at N≥8-10 cluster, must not chop bear-regime winners). Commit e211f42 (the watchlist note).

### 2026-06-17 — KNOWN BUG (identified, parked for the closed-order CSV): FAN multiplier leaks onto flip-LONGs

Operator spotted a live BCHUSDT flip-**LONG** carrying `FLIP:FAN_RATIO_GATE×2` (2× size, $20,955 notional). Root cause: the `flip_fan_mult_rule` cell (`40-45:35-99:2.0`, validated ONLY on SHORT fades into strong-bear) is applied in `_flip_filters` (lines ~330-344) with NO direction check → any FAN flip matching the BTC RSI/ADX cell inherits the 2×, including a flip-LONG (blocked SHORT → LONG). Worse: it sizes 2× into BTC RSI 40-45 (bearish-leaning), the opposite of the cell's context. Same direction-agnostic structure likely also leaks the FAN regime-block and strpk exit onto flip-LONGs. Fix deferred to when the order closes + CSV (operator's call): gate the multiplier — and likely the whole FAN entry branch — to `flip_dir=='SHORT'`. No change made yet.

### 2026-06-17 PM — SHIP (operator, MIRROR): flip-LONG regime gate (FLIP_LONG_REGIME)

Mirror of the validated flip-SHORT 2D gate, for the long side. A flip-LONG (a blocked SHORT faded → LONG) in a bear regime = long-into-the-trend. This batch's two flip-LONGs, AAVE+TAO (STRONG_BEAR, RSI~34), went straight to SL = 2/0%WR/−$220 (real Source×Regime: `FAN_RATIO_GATE LONG · S.BEAR = 2/0%/−1.10/−$220`). KEY ASYMMETRY vs the short gate (ADXΔ<0 cut): the long losers were ADXΔ-AGNOSTIC (ADXΔ +1.5 — regime was the killer), so `flip_long_regime_block_adxd_max=99.0` = REGIME-ONLY block, `flip_long_regime_block_regimes="STRONG_BEAR,HEALTHY_BEAR,CHOPPY_FLAT"`. `BEAR_EXHAUSTED` excluded on purpose (winding-down bear → mean-reversion long defensible). Block in `_flip_filters` gated to `flip_dir=='LONG'`, reason `FLIP_LONG_REGIME` (auto-counts via `_record_filter_block`); D11 wired (config.py + trading_config.json + UI input/load/save, cyan row). Simulated: blocks AAVE+TAO, shorts untouched. DISCIPLINE: shipped on N=2 (operator-directed, below the N≥15 bar) — structural mirror of a shipped filter, mechanism-justified but thin sample. TIGHT REVERT: re-open the bear cells if would-be-blocked flip-LONGs flip to WR>45% on N≥15 fresh. Why the entry gate, not a wider stop: operator-requested ATR-SL check this batch showed a −1.5×ATR stop saves only 1/4 losers (SKYAI, razor-thin) and deepens the other 3 — no stop saves a wrong-direction entry.

### 2026-06-17 PM — SHIP (operator): give-back cap DISABLED (frac 0.35→0); N held 0.5

`runner_trail_short_giveback_frac` 0.35→0.0 (the `_sp_frac>0` guard clean-disables → floor reverts to `max(peak − N×ATR, lock)`). RATIONALE: the give-back cap was designed to protect bull/chop BOUNCE-FADE shorts (peak-then-reverse) — but those are now BLOCKED by FLIP_SHORT_REGIME, so the cap guards a population that no longer trades while CLIPPING the bear-regime TREND runners the filter leaves us. Evidence: SKYAI (STRONG_BEAR runner) capped at +0.30 close vs +3.92% post-exit continuation with only −0.02 retrace = a continuous trend the ATR-floor would've ridden to ~+3.3. The BE-ratchet (lock) IS the round-trip backstop and makes the cap redundant: re-ran VELVET (pk+0.76/ATR1.59) cap-off → floor=max(0.76−0.80,0.10)=+0.10, still flat not the −1.20 SL. Recommended DISABLE cap + KEEP ratchet (best option); disabling the ratchet instead would be worst (loses round-trip safety, keeps clipping). N=0.5 HELD: operator asked to bump→1.0 ("now that we have BE"); I recommended HOLD — the BE-lock makes N=1.0 SAFE (downside bounded) but not proven BETTER; zero distinguishing evidence (lone runner trended → atr05/10/15 identical); and it'd be a 3rd same-direction loosening (one-change-at-a-time). REVERT GATE (cap): re-enable frac→0.35 if an armed bear runner round-trips peak≥+0.45→≤0 on N≥3 fresh (lock proves insufficient).

### 2026-06-17 PM — FIX: ATR-shadow leash made LOCK-AWARE (decision surface for N)

The atr05/10/15 leash shadows tested a LOCKLESS chandelier (`pnl ≤ peak − N×ATR`), but the live runner floor is now `max(peak − N×ATR, lock)` (cap-off, BE-ratchet on). So `atr05` did NOT mirror the live config — on low-peak reversers the lockless shadow fell through to hard-SL while live holds at +0.10 — meaning the table could not validly decide N=0.5 vs 1.0 under the policy actually run. FIX: the atr shadows now read `runner_trail_short_be_lock_pct` + `be_ratchet_enabled` and apply `floor=max(peak−N×ATR, lock)` (reason tags `lock` vs `atr`). Result: `atr05` = EXACT live config, `atr10`/`atr15` = the N=1.0/1.5 candidates under the same lock = clean decision surface. Only changes low-peak reversers (caught at the lock, matching live); trending runners unaffected (high peak → ATR-floor > lock anyway). UI helper text updated to document atr*→N and cap*→give-back-cap decision rows. GATE (decide N with data): bump `runner_trail_short_atr_mult`→1.0 only if `atr10` beats `atr05` cumulatively on N≥8 fresh armed shorts. This is the empirical path the operator's "later decide the best configuration" asked for. `slconv` (counts `hard_sl` only) unaffected by the new `lock` reason.

### 2026-06-17 PM — WATCHLIST: flip-SHORT ATR≥2 × REGIME (2D inversion); both rules HELD on thin/one-window N

Investigated 3 consecutive ATR>2 flip-short losses (AGT 2.1, HUSDT 3.0, ESPORTS 4.0 — all STRONG_BEAR, all never-armed, all gapped −0.70 SL to ~−1.2). ATR>2 ALONE is non-monotonic (pooled >2.00 bucket = 100% WR/+$495/N=4 — high ATR is the *best* bucket), so hunted the 2nd axis. Found: BTC regime, running OPPOSITE to the normal flip-short edge — deduped pool, ATR≥2 flip-shorts WIN in BULL (3W/0L, +$223, 2 batches) and LOSE in STRONG_BEAR. Mechanism: a FAN flip shorts a parabolic pump; in a strong bear that's a counter-trend short-squeeze that keeps ripping (never arms, high ATR gaps the stop); in a bull it's an exhausting overextension that the fade mean-reverts. INTERACTION: FLIP_SHORT_REGIME has it backwards for ATR≥2 — blocks bull/chop∧ADXΔ<0 (kills the high-ATR bull winners HUSDT-W/EVAA-W) and exempts bear (keeps the high-ATR losers). Operator proposed shipping both (ATR≥2∧bear→block, ATR≥2∧bull→unblock) ASAP. PER-BATCH INDEPENDENCE CHECK refuted shipping: the entire bear∧ATR≥2 loss is ONE ~10-min window today (3 correlated never-armed losses, −$366); the only out-of-window bear∧ATR≥2 sample (BR, ATR2.0) WON +1.81%/peaked +3.34% → 1-big-winner vs 1-bad-window = correlated-cluster overfit, not a cross-batch loser. The bull-unblock is more consistent (3W/0L, 2 batches) and low-risk (the filter's −$691 value is entirely ATR<2, the ATR≥2 slice is 2W/0L) but rests on N=2–3 and loosens a validated filter. DECISION: HOLD BOTH, watchlist with gates — BLOCK at ≤30% WR on N≥8 across ≥3 separate windows; CARVE-OUT (unblock) at ≥60% WR on N≥6–8 for bull∧ADXΔ<0∧ATR≥2. Retired the SKYAI liquidity×ATR framing (AGT/HUSDT mid-cap $175–227M → liquidity is NOT the discriminator, regime is). CURRENT_STATE watchlist updated in place. No code change.

### 2026-06-17 PM — SHIP (operator, discipline-override N=2): flip-SHORT high-ATR bear block (FLIP_SHORT_HIATR), cut ≥3

Carve-out to FLIP_SHORT_REGIME's bear exemption: block flip-SHORT when pair ATR% ≥ 3.0 AND BTC regime ∈ {STRONG_BEAR,HEALTHY_BEAR,BEAR_EXHAUSTED}. `flip_short_atr_block_min=3.0` + `flip_short_atr_block_regimes`; block in `_flip_filters` gated to flip_dir=='SHORT', reason FLIP_SHORT_HIATR (auto-counts via _record_filter_block); added `atr_pct` to the `_ff_in` filter input; D11 wired (config.py + trading_config.json + UI rose row + load/save). MECHANISM: a high-ATR parabolic pump in a strong bear is a counter-trend short-SQUEEZE that keeps ripping → the fresh short never arms and the high ATR gaps the −0.70 SL to ~−1.2 (ESPORTS 4.0, HUSDT 3.0 = 0%WR/−$245 this window). The same high-ATR pair WINS in a bull (regime inversion, HUSDT 3.9 / ESPORTS 2.6 bull = W) so the block is bear-only. CUT REFINED ≥2 → ≥3 after the operator asked "is it >2, >3 or >4": bear flip-shorts by ATR = 1.5 STG-W/PORTAL-W (+$362), 1.9 EVAA-L, 2.0 BR-W (+$272), 2.1 AGT-L, 3.0 HUSDT-L, 4.0 ESPORTS-L → the clean 0%-WR cliff is ATR≥3; ATR<2.5 is net-POSITIVE (cutting at 2 would kill PORTAL/BR/STG winners and only nets −$94 because BR offsets). Sim verified: ESPORTS+HUSDT blocked, AGT/BR/PORTAL allowed, bull high-ATR allowed. DISCIPLINE: N=2, both ATR≥3 losers from one bear window today — operator-directed override below the N≥8/≥3-window gate. TIGHT REVERT: raise min toward 4 or drop the regime if blocked ATR≥3 bear shorts would have ≥40% WR on N≥6 fresh (track via phantom + LIVE Flip Trades × BTC-Regime). Exact cut (2.5/3/4) unresolved on N=2. Context: this closed a multi-message investigation that also REJECTED widening the SL (tested N=3 + N=11, net-negative/tail-driven, ruin-unsafe at 20×) — the lever is entry (this block), not exit.

### 2026-06-17 PM — SHIP (operator): flip-SHORT B2 (strong-bull any-ADXΔ) + B1 (anti-parabola stretch≥2)

Two new flip-SHORT entry blockers from the PAIR_RSI_OB bull-short losses this window: ESPORTS (STRONG_BULL, EMA5 stretch 10.47%, RSI 94.5, ATR 5.73 → −2.25%/−$225 gapped stop in 0s, a parabolic blow-off) and TAC (STRONG_BULL, stretch 0.40, RSI 68.8 → −1.20% ordinary wrong-way bull short). Neither was caught by the existing stack (FLIP_SHORT_REGIME needs ADXΔ<0 — both had ADXΔ>0 +3.76/+1.20; FLIP_SHORT_HIATR is bear-only; PAIR_RSI_OB has no source filters).

B2 = `flip_short_regime_block_any_adxd_regimes="STRONG_BULL"`: block flip-SHORT in STRONG_BULL regardless of ADXΔ. Cross-pool (N=111 short flips) STRONG_BULL loses in BOTH ADXΔ halves — ADXΔ<0 −$190/55%WR/N=11 (already blocked) + ADXΔ≥0 −$531/47%WR/N=15 (was leaking through the ADXΔ<0-only gate). Strong-bull-specific: healthy-bull/chop ADXΔ≥0 ≈ breakeven (−$103/N=16) so they stay at the ADXΔ<0 cut. Catches both ESPORTS+TAC. Mechanism: don't short a strong bull. Engine: added an any-ADXΔ regime set checked before the ADXΔ<0 branch (same FLIP_SHORT_REGIME reason).

B1 = `flip_short_stretch_block_max=2.0`: block flip-SHORT when entry EMA5 stretch% ≥ 2 (anti-parabola — shorting a vertical blow-off that keeps ripping). Pool stretch≥2 = N=2/0%WR (ASTER+ESPORTS), 0 winners removed (the 1–2% band is 67%WR, preserved; cut sits cleanly between). Regime-agnostic catastrophe guard (catches a blow-off B2 misses, e.g. ASTER in CHOPPY_FLAT). Reason FLIP_SHORT_HISTRETCH, checked before the regime block so a parabola tags distinctly. Uses ema5_stretch already in _ff_in.

Complementary (near-disjoint trades: B2=regime/TAC, B1=parabola/ASTER, overlap only ESPORTS) so attribution stays clean. On the prechange-39 + current-2 batches: PREVIOUS +$454 → POST +$952, all 3 pool losers removed, 0 winners touched (B2 had ZERO false positives on the prechange batch). D11 wired (config + json + UI orange/rose row + load/save). Sim verified: ESPORTS→B1, TAC→B2, ASTER→B1, healthy-bull-ADXΔ>0 + bear correctly allowed.

DISCIPLINE: B2's independent edge is MILD (~63% of the −$531 is the 2 in-sample motivators → ~−$197/13 independent) and it WIDENS a filter validated on ADXΔ<0 — so STRONG_BULL-only + a tight revert: re-add the ADXΔ<0 requirement for STRONG_BULL if its ADXΔ≥0 shorts hit ≥50% WR on N≥10 fresh. B1 is N≈1 independent but a never-false-positive catastrophe guard (justified as risk mgmt, not edge): raise toward 3 / off if a stretch∈[2,3] flip-short wins on N≥5 fresh. Note: PAIR_RSI_OB (both losers' source) still has no source filters and is 0/N live — separate watch.

### 2026-06-18 — CONFIG (operator): blacklist ESPORTSUSDT (per-pair concentration, serial parabolic-pump loser)

ESPORTS flip-shorts lost in 3 straight batches: 21:53 ATR4.0 STRONG_BEAR −1.24 (now caught by HIATR), 23:00 ATR5.7 STRONG_BULL stretch10.47 −2.25 (caught by B1), 00:17 ATR7.4 HEALTHY_BULL stretch0.08 −1.20 (slipped EVERY filter — healthy-bull so B2/regime miss, ema5-stretch 0.08<2 so B1 misses, bull so HIATR is bear-only). The dimension values MORPH each batch (stretch 0.86↔10.47↔0.08, regime bear↔strong-bull↔healthy-bull, ATR rising 4.0→5.7→7.4), so dimension filters play whack-a-mole; the only constant is the pair. Per the locked per-pair-concentration rule (≥60% of a loser zone in 1–2 pairs → blacklist, not a dimension filter), ESPORTS is ~100% of the catastrophic flip-short losses while the cohort wins (HU/TAC/VELVET all +0.10 this batch). NOT a new-listing case: Binance onboardDate 2025-07-29 = 323 days (past the live 180-day new_listing_filter; subtype Gaming, not Alpha) — verified via fapi exchangeInfo. NOT a bug: all 4 trades evaluated correctly through the filters, none mis-blocked; the −0.70 SL gapped to −1.20 normally on a 7.44% ATR pair; the new filters only add blocks (no regression). Added ESPORTSUSDT to pair_blacklist (trading_config.json). Verified the blacklist covers flips: the scan fans over top_pairs (engine ~6510) which the blacklist filters at ~6220, so a blacklisted pair is never scanned → no signal → no block → no flip (live or phantom). This also confirms the new-listing filter structurally covers flips too (ESPORTS slipped only because it's 323d, not a bypass). Operator deferred the separate flip↔new-listing fail-open audit (missing-onboardDate pairs kept conservatively) to later.

### 2026-06-18 — CONFIG (operator): disable the SHORT BE-ratchet (runner_trail_short_be_ratchet_enabled true→false)

Operator asked to remove the BE-ratchet (lock 0.10). Investigated on the prechange-39 batch first: of 36 shorts, 26 armed (peak≥0.45); the lock would have cut 0 winners (by construction it can't cap a runner — once a runner runs the ATR-floor trails it up and the lock goes dormant) and saved only 4 SMALL round-trips (EVAA −0.36, PORTAL −0.13, PORTAL −0.02, AERO −0.02 → +0.10 each ≈ +$139, possibly in-sample). My counter-argument to keep it cited VELVET (pk0.76→−1.20 round-trip) as the big-save case — OPERATOR CORRECTED: VELVET was a flip-LONG, and the ratchet is short-only, so it never covered that. With VELVET removed, no SHORT has been observed arming then round-tripping to the SL, so the dramatic-save scenario is unproven for shorts; residual value is just the modest small-saves. Clarified the winner-cutting mechanism was the GIVE-BACK CAP (SKYAI +0.30 vs +3.92), already disabled — the lock cuts no winners, so removing it recovers no upside, it only drops a small safety. Net: removing is mildly net-negative (−$139/batch of small saves) but low-stakes and not dangerous. CURRENT runner-exit stack is now ATR-floor(N=0.5) ONLY — no cap, no lock; an armed short that fully reverses rides to the −0.70 hard SL. trading_config.json only (toggle already wired). RE-ADD GATE: re-enable if a SHORT arms (peak≥0.45) then round-trips to the −0.70 SL on N≥2 fresh.

### 2026-06-18 — CONFIG (operator): RE-ENABLE the SHORT BE-ratchet (revert same-day disable — was a mistake)

Operator clarified the prior-message disable was not intended — re-enabled `runner_trail_short_be_ratchet_enabled` false→true. Decision: keep the lock ON (it cuts 0 winners by construction and saves small round-trips) and GATHER data rather than remove it. New tracking gate: watch the lock-bound exits (`runner_trail_bound='lock'`, armed shorts caught at +0.10) and read their post-exit path — if price keeps going the short's way after the +0.10 exit, the lock is cutting potential (consider lowering/disabling); if it reverses, the lock is earning its keep. Verdict on N≥4–5 lock-exits, from the CSV (no new instrumentation). Runner-exit stack back to ATR-floor(N=0.5) ∧ BE-ratchet(lock 0.10), no give-back cap. Supersedes the same-day disable entry above.

### 2026-06-18 — BUGFIX: flip runner-trail arm missing the 0.005pp float-tolerance (JTO near-arm dead zone)

Operator spotted JTO showing armed/at-a-level in the UI while open, yet the BE-ratchet didn't save it (peaked +0.4471%, rode to −1.04 SL). Root cause: the flip runner stretch-trail arm (trading_engine.py ~9073) used a STRICT `current_peak >= _sp_arm` (0.45), while the STANDARD trailing path (~8567) arms at `peak >= effective_tp_target − 0.005` (0.445) — a 0.005pp float-tolerance added May-6 precisely to avoid missing arms by a rounding hair (comment: "+0.4998 when tp_min 0.50 … strict >= would never arm"). So a peak in the 0.445–0.450 band armed the UI/standard logic but NOT the flip BE-ratchet → JTO (+0.4471) fell in that dead zone and got no lock. FIX: `if current_peak >= _sp_arm - 0.005:` — unifies the flip arm with the standard path. Chose the buffer over the operator's round-to-2dp suggestion (functionally identical here, but round() near .xx5 is float-/version-dependent via bankers rounding; the buffer is deterministic AND is the exact mechanism the standard path already uses). Verified: 0.4471→armed, 0.4449→not, 0.45/1.73→armed. Effect on JTO: +0.10 lock instead of −1.04 SL (+1.1% swing) AND the flip arm no longer disagrees with the UI. Correctness/consistency fix (not edge-mining); the buffer is already the validated codebase standard. trading_engine.py only.

### 2026-06-18 — SHIP (operator): flip-SHORT BTC-30m-RSI-rising block + DISABLE PAIR_RSI_OB source

Disaster batch (06-18, flip net −$1521, HEALTHY_BULL 30%WR/−$604). Two actions.

(1) DISABLE PAIR_RSI_OB: `flip_entry_sources` "FAN_RATIO_GATE:1.0,PAIR_RSI_OB:1.0" → "FAN_RATIO_GATE:1.0". ✗ HARMFUL cross-batch (this batch N=18/28%WR/−$523, all HEALTHY_BULL, 72% SL-rate, no source filters). Phantom (`Pair RSI >65`) keeps seeding; re-enable only at WR≥60% & net-positive on N≥20.

(2) BTC-30m-RSI-rising block (`flip_short_btc30_rise_block_min=0.0`): block flip-SHORT when (entry_btc_rsi − entry_btc_rsi_prev6) > 0 = BTC 30m RSI rising. THE cleanest cross-batch differentiator found this session. Method: among the 33 FLIP_STOP_LOSS L1 losers, 32/33 NEVER ARMED (peak ~0.15) → an ENTRY problem, not exit (N/trail irrelevant). Continuous indicators showed NO SL-loss-vs-win separation (RSI 57.6 vs 57.8, ADXΔ −0.20 vs −0.18, stretch 0.51 vs 0.45). Pair-RSI direction FLIPPED across batches (rising +$668 yest → −$1128 today = noise). But BTC-30m-RSI direction was STABLE & strong in BOTH batches (FAN-only): rising −$1031 (today −$965, yest −$66) vs falling +$811 (today −$33, yest +$844); win−loss gap +$910/+$932 both days. Corroborated by the report's per-regime BTC-30m tables (bear: Falling −$39 vs Rising −$990). Mechanism: BTC 30m rising = macro bouncing → the faded parabolic pump squeezes; falling = pump exhausts → short pays. Re-sim: TODAY −$998→−$33, YESTERDAY +$778→+$844. Engine: block in _flip_filters SHORT-gated (reason FLIP_SHORT_BTC30_RISE), added btc_rsi_prev6 to _ff_in (from entry_fields / _current_btc_rsi_prev6 global); D11 wired (config + json + UI emerald row + load/save). Sim-verified on both batches. DISCIPLINE: 2 windows (below ≥3 bar), operator-directed — but it's the rare cross-batch-STABLE signal (everything else flipped) + clean mechanism. Halves the flip-short count (only short when BTC 30m rolls over). CONFIRM/REVERT: raise threshold or disable (min=99) if BTC-30m-falling shorts go net-negative OR the rising cohort flips to ≥55% WR on N≥10 fresh. Note: 2× FAN multiplier KEPT per operator (excluded from this analysis). Trail-improvement (strpk/cap beats atr05 by +$90-210 on armed runners) noted but NOT shipped — separate follow-up.

### 2026-06-18 — CONFIG (operator-directed): DISABLE the SHORT BE-ratchet (lock)

Reverses the same-day "keep it ON and gather data" decision once the data came in. `runner_trail_short_be_ratchet_enabled` true→false. Clarified first the operator's confusion ("aren't we already on strpk?") — YES, strpk/ATR-floor is the live trail; my earlier "switch the trail to strpk" was a misstatement, there's nothing to switch. The actual point: the +0.10 lock sits ON TOP of strpk and over-tightens it; disabling it just lets the existing trail do its job. EVIDENCE (15-42-03 batch, the 6 armed lock-exits = `runner_trail_bound='lock'`, all in the post-filter KEPT set): under the operator's framing that a loser can't drop past the −1.20 SL (so the lock's max credit is +0.10 vs −1.20 = +1.30/trade), the held-with-strpk outcome (`shadow_strpk`, the stretch-trail WITHOUT the lock) banked +0.37 to +0.83 on ALL 6 — 0 saved / 6 cut runners, net −2.35pp. WHY it inverts the earlier "the lock saves disasters" read: (a) the −1.20 cap removes the catastrophe upside the lock was credited with (HU/AGT −5.5/−5.9 → capped −1.20); (b) more importantly, the strpk TRAIL exits those reversers at +0.44/+0.47 BEFORE the crash, so the lock's "saves" only beat a no-trail baseline — with strpk live they're redundant AND too tight (banks +0.10 while the runner re-runs). Full-stack re-sim of 15-42-03: raw −$1521 → entry filters −$33 → +BE-off +$85 (the 6 lock-exits re-price +$118: TAC +$37, SYN +$34 lead). REVERT GATE (replaces the prior "+0.10 round-trip tracking" gate): re-enable if over N≥8 fresh armed SHORT reversers the no-lock book bleeds past −0.70 vs the `shadow_atr05` no-ratchet control. Caveat: N=6/one-batch + `shadow_strpk` is a model not realized fills → TIGHT gate. trading_config.json only (D11 already wired). Runner-exit stack now: ATR-floor(N=0.5) ONLY.

### 2026-06-18 — INVESTIGATED, NO CHANGE: BTC-30m-RSI-rising threshold (keep `>0`)

Operator asked whether tightening `flip_short_btc30_rise_block_min` from 0 to a higher cut (e.g. >+3) could spare yesterday's fat-tail winner BRUSDT (+$544/+1.81%, the prechange-39 batch's biggest flip winner, cut by the rising block) while still saving today's losers. My own "marginal riser" suggestion — KILLED by the data: BR's Δ(entry_btc_rsi − entry_btc_rsi_prev6) = **+10.60**, one of the STEEPEST risers in the batch, not marginal. To spare BR you'd need T>+10.6, which keeps essentially the entire rising cohort and forfeits the ~+$965 today-save. Threshold sweep (block SHORT if Δ>T) on both batches: 2-batch net effect peaks at T≈0.5–1.5 (+$767) and DECLINES monotonically as you tighten (T=3 → +$274, T=5 → +$315) — today's losers leak back faster than yesterday's winners recover; yesterday's winner-cut doesn't even drop until T=4. VERDICT: `>0` is within noise of optimal (T=0.5–1.5 buys a trivial +$8). BR is an irreducible fat-tail sacrifice — the priced-in cost of blocking a cohort that is net −$1k+/batch. No config change. Honest framing for the record: this filter WILL occasionally kill a steep-rising fat-tail winner like BR; the threshold cannot separate it from the losers (same regime, same steep-rising signature).

### 2026-06-18 — SHIP (operator): Entry-Funnel REAL cap-cost counter (normal vs flip)

Operator asked to see, in the "Blocked at max-5 (no trade lost)" box, how many NORMAL vs FLIP trades the 5-cap turns away, and whether the 1,417 figure is even correct. DIAGNOSIS: 1,417 is the WRONG metric — `blocked_at_max` counts filter-rejections that happen DURING full-book scans (a scan-start `_scan_had_room_snapshot`, per-cycle inflated), NOT trades the cap actually prevented; the genuine "the 5-cap turned away a ready signal" event (open_position max-pos gate, line ~3271) was NEVER counted (just `return None`). FIX: new in-memory `_cap_skip_counts {normal, flip}` (trading_engine.py ~900), incremented at that gate keyed `"flip" if flip_source else "normal"`. Exposed in main.py `entry_funnel` (real + exception-fallback dicts) as `cap_skip_normal`/`cap_skip_flip`; rendered as a subline on the Blocked-at-max box (templates/index.html ~11382) + BOTH text-report exports (~8583 clipboard, ~10374 saved-file) per D12 (grep-verified render+2 exports). CAVEAT baked into the design: still per-scan (a ready signal re-counts each cycle while full) → read RELATIVE (normal-vs-flip ratio, trend), not absolute distinct trades — but it's the CORRECT event now (a tradeable signal the cap stopped), not filter-noise-while-full. USE: answers the "raise positions 5→10/15?" question with data — expect flip≫normal and good-cohort skips LOW. Companion finding (this batch): the filtered good cohort runs ~0.3 concurrent positions vs the 5-cap (23 trades/12h @ ~11min) → the cap is NOWHERE near binding on the good trades → do NOT raise max-positions (the 1,417 pressure was the now-entry-filtered bad cohort). 3 files (trading_engine.py + main.py + templates/index.html).

### 2026-06-18 — SHIP (operator): open the fan 3-10 flip band (trade clean high-fan exhaustion, block only the parabolic gappers)

Operator-directed, from a deep fan-ratio analysis. TWO config edits, one thesis. (1) `flip_fan_spike_max` 5.0→10.0 — the spike veto now fires only at fan≥10. (2) `fan_ratio_block_long` "0.85-3.00,5.0-99" → "0.85-99.0" — longs blocked (→ flip seeded) contiguously from fan 0.85 up, closing the 3.0-5.0 gap. NET: flips now trade fan 0.85-10.0 (added 3-5, which NEVER seeded before, + 5-10, which seeded-then-vetoed); fan 10+ stays blocked.

WHY. (a) Operator caught my error: I claimed fan 3-5 was "already allowed, just hadn't fired" — WRONG. Verified in code: a flip is born ONLY from a blocked long (`_maybe_open_flip` inside the FAN_RATIO_GATE block path; `_flip_filters` can only veto/size an existing flip, never create one). The live block band was "0.85-3.00,5.0-99", so fan 3.0-5.0 was structurally unreachable — 0 of 145 FAN flips ever landed there (confirmed across 5 batches). (b) The fan 3-5 phantom row (+0.281%/73% WR aligned/N=19) is the A/B CONTROL stream (PASS:FAN_RATIO_GATE = fades of longs that passed at 3-5) — exactly the evidence for whether to add the band, and it pays. (c) Gap-risk reconsidered by magnitude: the 3 high-fan losers were ASTER 5.7→−0.69 (clean −0.70, no gap), ALLO 13.2→−1.20 (gapped to floor), VELVET 28.3→−1.20 (gapped). The catastrophic SL-gap kicks in at fan ~13+, NOT at 5 — so 3-10 is the clean exhaustion-fade band and only 10+ truly gaps. Confound test: of the 3 losers 2 were BTC-30m rising (now filtered separately) but ALLO was FALLING and still gapped → fan carries independent gap-risk, so we keep the 10+ block rather than unblock entirely. Normal-long impact of the block extension is negligible (bot is ~flip-only: 3 normal vs 66 flips/batch) and thesis-consistent (high-fan longs = late/over-extended). trading_config.json only (existing fields). DISCIPLINE: phantom/control-derived + partly in-sample → TIGHT revert gate. LOCKED REVERT: revert BOTH (spike_max→5, block_long→"0.85-3.00,5.0-99") if fan 3-10 fresh flips show WR≤50% OR avg≤+0.05% on N≥10, OR ≥2 of the first 5 gap past −1.0%. Read surface = the Fan-Ratio Curve live rows + the flip log by fan bucket.

### 2026-06-18 — ANALYSIS (no change): max-open-positions 5→8 considered, DEFERRED to the cap-skip counter

Operator asked whether to raise max positions 5→8 since the recent filters trade "very little" at 5. ANALYSIS → not yet, and the framing is a trap. (1) MECHANIC: `equal_split` sizes each trade = balance/max_positions (engine line 1859), so 5→8 SHRINKS every bet 37.5% — it does NOT add fills. Gross leverage at full book is constant either way (no ruin-risk argument for or against). (2) BINDING? The "1,417 Blocked at max-5" that motivates the ask is the WRONG metric (filter-rejections during full-book scans, not signals the cap turned away); the REAL cap-cost is the new `_cap_skip_counts` counter, which reads 0 (no fresh data). Historically the book ran ~0.3 concurrent positions (23 trades/12h × ~11min) — the cap was never binding. (3) RIGHT LEVER: "trade more" is a SUPPLY problem (just addressed via the fan 3-10 add), not a cap problem — equal_split can't manufacture trades, it only slices thinner. DECISION: run one fresh batch with the new fan supply, read the cap-skip counter. Raise only if flip cap-skips are HIGH and the sleeve is net-positive, and then modestly (5→6/7, one step) — never blind to 8. Caveat worth watching: fan-spike fades cluster (market-wide pumps), so the new supply is the type that could start binding — hence measure, don't assume. No config change.

### 2026-06-18 — SHIP (operator): BULL_LONG entry sleeve (build-side long) + Bull-Long Curve observation table

First LIVE long-build sleeve. Genesis: PASS:FAN_RATIO_GATE LONG (the longs that PASS the fan gate, tracked same-direction in the Source×Regime bull-hunt) pays strongly in HEALTHY_BULL — cross-POOL validated: pre-reset 34/79%/+0.33 and post-reset 18/94%/+0.49 (combined H.BULL ~52/84%/+0.39), with a STABLE S.BULL exclusion (22% then 20% WR — loses both pools). Thin live corroboration: the 2 live unmatched longs this batch (1000PEPE/ADA) were HEALTHY_BULL, both won (+0.69 avg). Phantom→live haircut ~0.25% (fees 0.063-0.09 gross + slippage) → ~+0.14% live net floor; but the live sleeve uses the NORMAL long exit (per-level trailing) which BEAT the flat phantom model on those 2 examples, so realized expected ABOVE +0.33.

WHAT SHIPPED. `entry_strategy="BULL_LONG"` — a REAL momentum long (NOT a fade, NOT _is_flip → rides the normal long exit stack: per-level trailing / ATR-widened SL / EMA13-off-for-longs / no 45min cap). Fires at the fan-gate-PASS site (FAN_CONTROL else branch) via `_maybe_open_bull_long` (fail-silent, mirrors `_maybe_open_flip`); self-gates: bull_long_enabled AND fan_ratio < bull_long_fan_max(0.85) AND entry_btc_regime ∈ bull_long_regimes("HEALTHY_BULL") AND per-pair/30min cooldown. BYPASSES the downstream long filters (long_unmatched_only + pattern-cell block) to match the naked phantom population (the +0.33 was measured on the passthrough); all HARD risk controls (max-open, one-per-pair, cooldown, liquidity/gross-lev caps) still enforced inside open_position. Multipliers `bull_long_size_mult`/`bull_long_lev_mult` built from scratch, DEFAULT 1.0/1.0 = NO amplification, normal STRONG_BUY 20× leverage (operator: "same leverage, 1 as multiplier"). cell_src "BULL_LONG" → own row in Multiplier Cell Performance.

OBSERVATION TABLES (D12, UI + both exports). ① 📈 Bull-Long Trade Log (live scorecard, after the Flip Trade Log) — N/WR/avg/Total$/Size×/Lev×/Δ$-1× + ×BTC-Regime row + VERDICT BUILDING n/15. ② 📈 Bull-Long Curve — virtual longs (PASS + new BLOCK:FAN_RATIO_GATE seeds) by fan bucket × regime, flat-model phantom as a CONSISTENT comparator (not absolute) to map whether the H.BULL edge is fan-dependent or fan-agnostic. THE EXPANSION LOOP: the curve's axes ARE the config dials — if H.BULL wins at all fan buckets → raise `bull_long_fan_max`→99 (drop the fan gate); if CHOP/H.BEAR win → add to `bull_long_regimes`. Widen by config, no code. Post-Exit Regret already includes BULL_LONG (inclusive by close-reason). D11: 5 fields fully plumbed (config.py + json + UI input/load/save). Built by mirroring the flip; entry path reviewed line-by-line + AST/JSON/grep verified.

LOCKED REVERT GATE (phantom-derived + single-regime-window → TIGHT): disable (`bull_long_enabled`=false) if at live N≥15 the sleeve shows WR≤55% OR avg≤+0.05% net. INSTANT review if 3 of the first 6 hit SL, OR if live−phantom gap > 0.25%/trade (the PAIR_RSI_OB failure mode). Do NOT raise the multiplier off 1.0 until live clears WR≥70% & avg≥+0.10% on N≥30. RISK: bypassing downstream long filters is the real exposure (some may have protected against bad longs the phantom didn't see) — the N≥15 gate is the backstop; the regime gate means it can't bleed outside HEALTHY_BULL (just stops firing). 5 files (config.py + trading_config.json + services/trading_engine.py + main.py + templates/index.html).

### 2026-06-18 — BUGFIX (operator-caught): BULL_LONG fired on the WRONG population — corrected to the validated blocked-long zone + fan_max 0.85→5.0

The BULL_LONG sleeve shipped earlier today was wired to the WRONG branch. ROOT CAUSE: `PASS:FAN_RATIO_GATE LONG` (the validated edge, H.BULL 94% WR) is NOT "longs that passed the fan gate" — it's the "un-block hunt": a long the fan gate BLOCKED (fan ≥ 0.85), tracked as a virtual long ("what if we un-blocked it"). Both I (spec) and the build agent misread "PASS" as "passed the gate" and wired the live `_maybe_open_bull_long` into the fan-PASSED else-branch (fan < 0.85) — a rare, UNVALIDATED population. Operator caught it ("nothing below 0.85?" — the Bull-Long Curve's 0-0.85 bucket was empty because passed longs are rare; all 37 validated longs sit at fan 0.85+). The sleeve as shipped barely fired and, when it did, traded the wrong longs. Near-zero live damage (fan<0.85 longs rare → ~0 opens). FIX (3 parts): (1) moved the live open to the BLOCKED branch (fires on blocked longs, the validated population), right beside the flip-fade hook; (2) `bull_long_fan_max` 0.85→5.0 — now the UPPER fan bound of the traded blocked-zone (H.BULL edge runs 0.85-5; nothing above 5), so it fires on blocked longs fan 0.85..5 in HEALTHY_BULL; (3) the bull-long now PRE-EMPTS the flip-short fade (`_maybe_open_bull_long` returns the Order; if it opened, skip `_maybe_open_flip`) — same blocked-long event was triggering BOTH a flip-SHORT and a bull-LONG (opposite positions racing for the one-per-pair slot); today they're mutually exclusive only because FLIP_SHORT_REGIME vetoes H.BULL shorts, but the pre-empt makes it robust. Also cleaned the muddied seeds (removed the subagent's redundant BLOCK:FAN_RATIO_GATE double-seed + the passed-branch PASS pollution) so the Bull-Long Curve = the clean blocked-long pool. Curve config-driven LIVE marker (✓ = fan < bull_long_fan_max) + a TOTAL footer row (per-regime summatory, operator-requested) added. Revert gate UNCHANGED (live N≥15: WR≤55%/avg≤+0.05% → disable). 5 files. AST/JSON/grep verified; entry branch re-read line-by-line post-fix.

### 2026-06-20 — ANALYSIS (no change, operator: keep collecting): FAN flip-short loser-separator hunt — all 6 candidates rejected cross-batch; sleeve is ~zero-edge fat-tailed coin flip

Operator asked, over a long session, to find a clean winner/loser separator for the FAN flip-short sleeve (`entry_strategy="FLIP:FAN_RATIO_GATE"`, SHORT, 20×) after a brutal run (last-6h −$937 on batch scalpars_orders_paper_2026-06-20_13-13-05). Tested 6 candidate entry filters on FAN flip-shorts across 3 batches (Jun17 prechange +$1039 N=28 / Jun18 postfan −$998 N=48 / Jun20 −$20 N=50). De-muxed to 1×, CLOSED only, dedup (opened_at,pair,direction). EVERY candidate failed the cross-batch / anti-overfit gates:

- **U1 (BTC vol<0.15 × BTC ADX>30):** Jun17 cut 11 (6W/5L) Δ−$335 (destroys winning-batch profit), Jun18 Δ+$9, Jun20 Δ+$533. Over-blocks normal conditions; the same low-vol×hi-ADX signature was a +$335 WINNER in Jun17.
- **U2 (pair ADX falling [entry_adx_delta<0] × BTC ADX rising [entry_btc_adx−entry_btc_adx_prev>0]):** best live precision (last-6h cut 6, 1W/5L, −$566; catches RIF/BICO/HEI that U3 misses) and soundest mechanism (divergence — pair losing downside momentum while BTC trend strengthens up → squeezes the fade). BUT Jun17 cut 6 (4W/2L) Δ−$31, Jun18 cut 12 (8W/4L) Δ+$13 — cut 2:1 WINNERS for ≈$0 in both out-of-sample batches. Loser-fraction 33%/33%/54% → only loser-heavy in its derivation batch = flags regime, not outcome.
- **U3 (BTC vol<0.10 = "Performance by BTC Volatility Regime (ATR%) <0.10%" bucket = entry_btc_atr_pct<0.10):** 0 trades in BOTH prior batches (the <0.10 BTC-vol regime didn't occur → literally unvalidatable, 1-episode; prior reports' BTC-Vol-Regime tables have no <0.10 row, lowest bucket 0.10-0.15). Also LEAKY: misses the 3 biggest last-6h losers (RIF −151/BICO −145/HEI −134 sit at vol 0.103-0.107, just above the line — −$430 escapes). Adjacent 0.10-0.15 band FLIPS SIGN cross-batch (Jun17 bearish +$876 / Jun18 −$344) → low BTC vol is not a stable loser signal. Operator correctly flagged U3 as a bad recommendation (I had wrongly framed "0 prior trades / near-zero footprint" as safe insurance — it's the signature of no demonstrated edge; retracted).
- **U4 (BTC ADX 30-35):** pooled curve <22 +$387 / 22-26 +$154 / 26-30 +$389 / **30-35 −$1077** / 35-40 +$304 = non-monotonic mid-range hole flanked by winners (locked-rule confound). Expressed as honest threshold BTC ADX≥30 it FLIPS sign (Jun17 −$477 = those were winners). ~78% of the 30-35 loss is 6 singleton fat-tails.
- **U5 (pair ADX 22-25):** N=8-10/batch; U2∪U3 already kills 7 of 10 (−$595 of −$635). Only incremental loser is ORDI −$128, at the cost of cutting INJ +$34 & NEAR +$53 (1L/2W) → pure redundancy + winner-bleed.

- **U2 ∪ U3 together (the build we were about to wire):** current batch cut 24/50 = 48% of the sleeve, 11W/13L, Δ+$906 (keep N=26, 81% WR, +$886); last-6h cut 16/17 → leaves −$12. BUT prior batches Δ−$31 / +$13 cutting mostly winners (U3=0 there, so it's U2-alone). Catches 16/17 of the live disaster only because the last 6h was ~all losers — it identifies the regime, not winners-vs-losers within it (still cut 6 winners). Best candidate found, but fails the same out-of-sample test (winner-heavy precision in the two honest batches). If ever shipped → only as an explicit unvalidated mechanism bet (discipline-override) with a tight revert (revert if next ≥15 blocked trades come back net-positive).

PER-PAIR concentration (CLAUDE.md mandatory check): current-batch loss diffuse — top-2 pairs (MET −$301 / HEI −$251) = 41% < 60% blacklist threshold; rest singletons. Cross-batch NO pair is a consistent loser: VELVET/JTO/XLM/XPL/SKYAI each WON ≥1 batch; this batch's worst (MET, HEI) have 0 prior history (1-batch). → no pair-blacklist either.

SCOPE: U3-for-all-flips evaluated and REJECTED — on PAIR_RSI_OB (0.05×, S.BULL) U3 fires on 14/26 (54%) for ±$7; that sleeve has no 20×-gap mechanism and its real losses (Jun18 −$523) were at vol 0.13-0.15. If anything ships → FAN-flip-only.

CORE FINDING: the FAN flip-short sleeve is a ~ZERO-EDGE fat-tailed coin flip — pooled +$1039 −$998 −$20 = +$21 net / 126 trades, avg ≈−0.005%/trade; the bearish Winners-vs-Losers table shows no entry separator (BTC RSI 40.5 vs 35.8, ATR 0.96 vs 1.08, pADX 20.7 vs 21.0). You cannot filter a coin flip into an edge — which is why all 6 candidates + regime + pair all overfit. The ONLY factor consistent across all 3 batches is the 20× gap-through-SL mechanism (FLIP_STOP_LOSS L1 slips −0.70 → −1.0/−1.2%). DE-LEVER quantified (pure $ scaling, leverage-invariant — %/trade unchanged): last-6h −$937 → −$469 @10× → −$234 @5×; worst single loss −$167 → −$84 → −$42. De-lever is ruin-risk control, NOT an expectancy fix (sleeve stays ~0). RECOMMENDATION made to operator: pause the FAN flip-short sleeve or cut it to ~5× rather than ship any entry filter. OPERATOR DECISION: do nothing now, keep collecting data (declined the proposed 20×→1× de-lever AND all 6 filters). All 6 candidates + U2∪U3 logged to CURRENT_STATE watchlist for re-evaluation at N≥2 fresh batches or the next low-BTC-vol grind episode. No code, config, or leverage change made.

ADDENDUM (same session, 2 more variables evaluated per operator request):

ADXΔ (pair ADX delta = entry_adx_delta), FAN flip-short, 3 batches — block ADXΔ<−0.5 Δnet: Jun17 cut 12(5W/7L,−$644) +$644 (keep +$1683) / Jun18 cut 19(12W/7L,−$527) +$527 / Jun20 cut 22(11W/11L,−$687) +$687. **THE ONLY candidate net-positive on ALL 3 batches including the winning Jun-17** — no sign flip; both loser sub-buckets negative all 3 (<−1.0 −$259/−$495/−$167 · −1.0..−0.5 −$385/−$31/−$520); −0.5..0 mixed (+$213/−$84/+$366) → cut at −0.5 not 0. Mechanism: flips bypass the live `Pair ADX Dir S: rising` filter → enter on collapsing pair-ADX (no momentum). PROMOTED to TOP re-eval candidate (above U2∪U3) in CURRENT_STATE. CAVEATS: (a) trips the locked "high-WR-but-net-losing → fix sizing not block" rule on Jun-18 (cut cohort 63% WR, cuts 12 winners); (b) it is the 2026-06-17 ADXΔ<−0.5 candidate already DOWNGRADED — BE-ratchet rescues the armed reversers, residual = never-armed straight-to-SL = the same 20× gap fat-tails (PORTAL/XLM/NEAR repeats). GATE: ship `flip_adx_delta_min=-0.5` (discipline-override+tight revert) at N≥30 ONLY IF cut stays net-neg AND ≤50% WR AND residual not 1-2 repeat pairs; else it's a sizing/de-lever fix.

Pair EMA13-EMA50 gap (entry_pair_ema20_ema50_gap_pct, misnamed — engine stores (ema13−ema50)/ema50), FAN flip-short — FAN flips INVERT the normal pair-trend filter. Live PAIR_TREND_FILTER blocks NORMAL shorts at gap≥0; flips want the opposite: gap≥0 (extended-up) non-losing all 3 (+$233/−$77/+$49), gap<0 (downtrend) is the disaster side — BUT gap<0 was the +$806 profit engine in Jun-17, so blocking it kills the winning batch (overfit trap). Only cross-batch-consistent sub-cell = gap 0–1.0 (positive all 3: 0-0.5 +0.05/+0.31/+0.32, 0.5-1.0 +0.44/+0.49/+0.27); gap>1.0 over-extended = consistent small loser (−$227/−$342, overlaps fan-spike). Mechanistic insight (flips need opposite trend condition), watchlist not ship. Both logged to CURRENT_STATE 2026-06-20 block. Still no code/config/leverage change.

### 2026-06-20 — SHIP (operator N=9/N=15 DISCIPLINE-OVERRIDE): PAIR_RSI_OB ADX≥33 floor + lev 1×→20×, AND BULL_LONG fan 1.65–3.00 + lev 1×→20×

Two operator-directed ships, both N<30 + a straight 1×→20× leverage jump (skipping the locked 1.5×-first Phase-3 staging). I flagged both as hard discipline-overrides against the locked multiplier gate (N≥30, staged) + this session's own finding that 20× counter-trend fades are the gap-through-SL disaster profile; operator chose full 20× on both via AskUserQuestion ("Full 20× now (override)"). Shipped with tight reverts.

(1) PAIR_RSI_OB (overbought-fade short). ① `flip_pair_rsi_ob_adx_min=33.0` — new config field; engine gate added in the PAIR_RSI_OB branch of `_flip_filters` (AFTER the regime gate), block when entry pair ADX < 33 → counter `FLIP_PAIR_RSI_OB_ADX` (auto-recorded via the existing flip `_record_filter_block(_reason, flip_dir)` path). ② `flip_entry_sources` PAIR_RSI_OB lev_mult `0.05→1.0` = full 20× (was 1× de-risked obs). EVIDENCE (live, N=9): PAIR_RSI_OB × Pair ADX bucket 33+ = 9/89%WR/+0.58%/+$698; every bucket <33 net-negative (18-22 −$75, 22-25 −$138, 25-28 −$163, 28-30 −$76, 30-33 −$91) → the fade pays only on a blow-off in a strongly-trending pair. The ADX≥33 floor restricts firing to the one winner cell, partly offsetting the 20× risk. D11 full: config.py default+comment, trading_config.json, engine gate, templates/index.html (input `config-flip-pair-rsi-ob-adx-min` + load + save, grep-verified 1/1/1). TIGHT REVERT: set `flip_pair_rsi_ob_adx_min`→0 and/or lev→0.05 at live N≥15 new PAIR_RSI_OB shorts if 33+ WR≤70% OR avg≤+0.05%; INSTANT de-lever if any single fade gaps SL past ~−1.0% OR 3 of first 6 hit FLIP_STOP_LOSS.

(2) BULL_LONG (build-side long). `bull_long_fan_min` 1.35→1.65, `bull_long_fan_max` 10.0→3.0, `bull_long_lev_mult` 0.05→1.0 (=20×). Existing fields (no new D11 — config.py defaults + trading_config.json values only; UI already wired). EVIDENCE (live Bull-Long × Fan): only 1.65-2.0 (5/80%/+$175) and 2.0-3.0 (10/90%/+$705) are positive; below 1.65 net-neg (0.85-1.0 −$419, 1.0-1.35 −$732, 1.35-1.65 −$120) and above 3.0 small-N losers (3-5 −$206, 5-10 −$130). The fan restriction is itself a risk reducer (cuts the documented losers, keeps the two winner bands = N=15); 20× then amplifies the filtered winner cell. CAVEAT: violates "ship one change at a time" (fan-min + fan-max + lev together); sleeve was net-LOSING overall (47/49%/−$352, H.BULL −$789 de-muxed) — the bet is the 1.65-3.00 sub-band's 80-90% WR holds live at 20×. Supersedes the prior BULL_LONG 1×-observation gate (its N≥15 revert had already tripped → this is the operator's hunt-the-2nd-variable resolution at 20×). TIGHT REVERT: de-lever `bull_long_lev_mult`→0.05 at live N≥15 new bull-longs if kept band WR≤70% OR avg≤+0.05%; INSTANT de-lever if 3 of first 6 SL OR any single gaps past ~−1.0%; widen fan back only if 1.65-3.00 holds AND adjacent bands turn positive on N≥30.

Validated: trading_config.json parses (bull_long fan 1.65/3.0/lev 1.0; pair_rsi_ob adx_min 33.0, sources …PAIR_RSI_OB:1.0:1.0); config.py + services/trading_engine.py AST-clean; UI ids 1/1/1. 5 files (config.py, trading_config.json, services/trading_engine.py, templates/index.html, CLAUDE_CURRENT_STATE.md). Both sleeves now run full 20× — the next batch's FLIP TRADES × BTC-REGIME (PAIR_RSI_OB row) and Bull-Long Trade Log are the make-or-break reads; watch the first 6 fires of each per the instant-de-lever triggers.

### 2026-06-20 — SHIP (operator): PAIR_RSI_OB ADX floor RAISED 33→40 + analytics retargeted to ADX≥40

Same-day follow-up after the split-bucket analytics (shipped earlier today) populated with live data. The ADX≥33 cell — which we'd shipped at 20× on N=9/89%/+$698 — had (a) DECAYED to 13/62%/+$431 (WR through the 70% revert line), and (b) split cleanly: 33-40 = N=5/20%WR/−$237 (a loser band) vs 40+ = N=8/88%WR/+$668. The edge is entirely ADX≥40. Operator raised `flip_pair_rsi_ob_adx_min` 33→40. I flagged this as a 2nd finer re-cut on a decaying cell (N=8 < the N=13 it replaces — re-optimizing to the latest snapshot = the overfit trap; my recommendation was to NOT re-cut but watch the revert gate / de-lever). Operator chose 40. Shipped with the same tight revert (now keyed to the 40+ cell: de-lever/disable at N≥15 if WR≤70% or avg≤+0.05%). RSI→70 was the alternative cut considered and rejected (it only removes the 65-70 breakeven band +$45, not the −$237 loser; ADX→40 dominates and likely subsumes it). Also retargeted the analytics: 🎯 PAIR_RSI_OB × Pair ADX now splits 33+ into 33-36/36-40/40-45/45+, and the ADX≥33×RSI cross-bucket became ADX≥40×RSI (key `adx33_rsi`→`adx40_rsi`, threshold ≥40, all UI titles + both exports + tbody id). config.py 40.0 default + comment; trading_config.json 40.0. Validated: floor=40.0, zero adx33 remnants, 3 ADX≥40 export blocks (1 UI title + 2 exports), main.py/config.py AST-clean. The honest note for next batch: this cell has now been re-cut twice on N≤13 — if the 40+ cohort doesn't hold ≥70% WR on the next ~5 live fires, de-lever (gate), do NOT slice a third time.

### 2026-06-20 PM — NO-SHIP (analysis): U3 (BTC vol<0.10) + MET deep-dive → resolved to one out-of-sample ship gate

Extended FAN flip-short investigation on the fresh batch (export 19-43-12 / 20-41-55, FAN flip-short N=60, net −$131.78). Operator hunted hard for a shippable loser-cut; the honest finding after exhaustive testing is that none survives the anti-overfit bar, and the whole thing resolves to a single forward-confirmation gate.

**The U3 reframe.** The dashboard "Performance by BTC Volatility Regime (ATR%)" table is the FAN flips sliced by `entry_btc_atr_pct` (verified: 13/34/12/1 = 60 trades, −$681.85/+$430.08/+$107.40/+$12.59 = −$131.78 exactly; pure FAN_RATIO_GATE, no entry-mixing — confirmed the row universe contains only that one strategy). So that table IS U3. It looks clean because the entire loss concentrates in one cell: **BTC ATR <0.10 = 13 trades, 38% WR, −$682; ≥0.10 = 47 trades, 68% WR, +$550.**

**Combined U3+MET on this batch:** block (BTC ATR<0.10 OR pair=MET) → cut 15 (5W/10L, −$788), keep 45 (32W/13L, 71% WR, +$656), Δ+$788 vs the −$132 baseline. MET's marginal contribution beyond U3 is just +$106 (2 MET trades survive U3; U3 already eats MET's two biggest −167/−122).

**Why every candidate is no-ship (all fail the same way — single-batch / in-sample):**
- Cross-batch footprint: U3 fired 0 in Jun-17, 0 in Jun-18, 13 in Jun-20. MET 0/0/4. HEI 0/0/5 (2W/3L). SAND 0/0/3 (1W/2L). None has a single out-of-sample data point. The +$656/+$788 is scored on the exact trades the rules were carved from → can only look good.
- Pair blacklists: only MET is all-loser (4/4, −$396); HEI and SAND have winners (HEI +$38/+$35, SAND +$65) so blacklisting them cuts winners. And per-concentration doctrine wants cross-batch consistency, which 0/0/N can't supply.
- Theory (the decisive kill on U3): the U3-cut losses are PAIR gaps while BTC was calm — cut losers had BTC ATR ~0.095 (median) yet closed −1.2% on pair ATR 0.6-1.25, and even genuinely calm pairs (ETC pair-ATR 0.22, XMR 0.37) hit the −0.70 SL. So BTC's volatility doesn't predict an individual alt's gap; the "low BTC vol = worse for shorts" reading is backwards from intuition AND the opposite of the table's own stated "high-ATR+low-ADX = violent chop" hypothesis → finding the reverse of your hypothesis in one sample = noise. Corrected a mischaracterization mid-analysis: pair-ATR 0.89 is normal-to-low, NOT "the alt was moving"; pair-ATR governs how far PAST the SL the loss overshoots, not WHETHER it loses — the loss is a wrong-direction fade amplified by 20×.

**Exhaustive separator search on the 47-trade U3-keep cohort (15 losers vs 32 winners) — nothing separates:**
- Every tracked indicator overlaps: pair-ADX 19.9 vs 20.1, ADXΔ −0.33 vs −0.60, pair-RSI 58.5 vs 57.0, BTC-ADX 27.5 vs 29.5, range-pos 75 vs 69, quality 2.19 vs 2.07 — all small, overlapping deltas.
- Filter combos within keep: ADXΔ<−0.5 cuts 14W/9L (kills more winners than losers); U1/U4 cut net-POSITIVE pockets (Δ negative); U5 (pADX 22-25) cut 4W/4L; the only high-loser-precision cut is ADXΔ∩U5 = 1W/4L (+$526) but that's an N=5 blocklist of the exact gappers (RIF/BICO/ORDI/BIO), three filters deep on a single batch.
- Pair-ATR is non-monotonic (losers at both 0.64 AND 2.8; the 0-0.50 cell is 9/9 winners = a winner pocket, not a loser cut). Pair-volume is non-monotonic and backwards (<$50M is net-POSITIVE +$229; cutting low-vol destroys value). Range-position 50-75% is the loser zone (−$595) but at 52% WR.
- The 15 losers share exactly one thing: each gapped through the −0.70% SL at 20× (closes −0.9% to −1.2%, scattered across every pair and regime, good and bad). No entry filter can see a gap that hasn't happened yet.

**Conclusion (unchanged from the session's core finding, now triple-confirmed — indicator scan, filter combinatorics, pair ATR/volume):** the FAN flip-short bleed is a leverage/tail problem, not an entry problem. The mechanism-true levers are on the tail (de-lever the sleeve, or per-trade cap), not entry filters or one-batch pair blacklists.

**SHIP GATE (operator-set, logged):** ship U3 (and/or a discretionary MET risk-blacklist — bounded downside, one microcap) the INSTANT the next batch fires sub-0.10 BTC ATR (or another MET flip) and LOSES again — that is the confirming out-of-sample sample we don't yet have. Until then: NO config change.

**Process decision: keep the run going, NO reset.** Out-of-sample validation = elapsed time + new fires, not a paper reset; the cross-batch dedup key `(opened_at, pair, direction)` is reset-proof by design (never uses `id`). Keeping the run preserves N and enables a clean before/after-today split for the forward test. Reset is purely a capital-hygiene lever — only if the paper balance falls low enough that position sizing degrades (balance ~$2,330, runway ~19.6h with emergency swaps firing, but workable). No FAN-sleeve config changed, so future FAN flips remain poolable with today's 60 for clean attribution. The 17:42/20:41 re-exports showed no new FAN fires yet (still 60/−$131.78) — not the confirming batch.

### 2026-06-20 PM — FIX (PAIR_RSI_OB ADX floor was a no-op) + SHIP U3 (FAN sub-0.10 weekend block, FAN-only)

Two engine changes, shipped together before an operator paper reset (pre-reset snapshot saved: reports/flip_ref_2026-06-20_presreset_22-31-57.csv + flip_2026-06-20_presreset_results.txt).

**FIX — `flip_pair_rsi_ob_adx_min` (the ADX≥40 floor) had never fired live.** The gate in `services/trading_engine.py` `_flip_filters` reads `ind.get('adx')` (pair ADX), but the `_ff_in` dict assembled at the call site (~line 3164) never populated a pair-`adx` key — it had `btc_adx`, `adx_delta`, `pair_rsi`, etc. but not `adx`. So `ind.get('adx')` returned None, the guard `if _ob_adx is not None and _ob_adx < _ob_amin` short-circuited, and every PAIR_RSI_OB short passed the floor regardless of pair ADX. The sleeve has been running fully unfiltered at 20× since the floor "shipped" (commit 840e913) — explaining the live fires at pADX 25-36 and the −$596 de-muxed bleed. Fix: added `'adx'` to `_ff_in`. Verified on the 22-31-57 batch: with the floor live, 35 of 43 PAIR_RSI_OB shorts (the pADX<40 fires, −$427) get blocked; the 8 at pADX≥40 stay (+$33, the 40-45/45+ winning band). Audit confirmed no OTHER key read by `_flip_filters` is missing from `_ff_in` (rsi/adx_prev1 matches were false positives on `btc_ind` in another function). The floor's tight-revert clock effectively starts now (first time the 40+ cohort is the only thing firing).

**SHIP — U3, FAN flip-short sub-0.10 BTC-ATR block, FAN-ONLY.** New config field `flip_fan_btc_atr_min = 0.10`: block FAN flip-short when entry BTC ATR% < 0.10. Gate placed inside the `source == "FAN_RATIO_GATE"` / `flip_dir == 'SHORT'` block (counter FLIP_FAN_LOATR; added `btc_atr_pct` to `_ff_in`). D11 complete: config.py default + evidence comment, trading_config.json (thresholds.flip_fan_btc_atr_min=0.10), engine gate, UI input (cyan "Block BTC ATR% <" in the FAN filters panel) + load + save; the filter-block counter rides the existing `_record_filter_block(_reason, flip_dir)` at the call site. Grep-verified: id `config-flip-fan-btc-atr-min` x3 (input/load/save), key present in all 4 files, py+json AST clean.

EVIDENCE: FAN sub-0.10 cell = N=14 / 36% WR / −$775, every loss a 20× straight-to-SL gap. Weekday batches Jun17 (Wed, BTC ATR min 0.109) and Jun18 (Thu, min 0.122) never dipped <0.10 → the regime is **weekend-only** (operator's day-of-week insight; thin weekend liquidity is the common cause of both quiet BTC and gappy alts). Meets the locked Pattern-C→FILTER gate on WR≤40%/Avg%≤−0.20%/NP≥60% — fails ONLY on N (14<30) and is ONE weekend. Shipped as a **discipline-override** (transparent, like PAIR_RSI_OB N=8 / BULL_LONG N=15).

SCOPE = FAN-ONLY, deliberately. The PAIR_RSI_OB sub-0.10 losers (AXS/AVAX/SOL, pADX 27-34) were the ADX-floor BUG above, not clean evidence — once the fix deploys they don't fire. There is zero clean evidence that a properly-filtered PAIR_RSI_OB short (pADX≥40, STRONG_BULL) loses sub-0.10, and it's a different macro regime (FAN's sub-0.10 losses are bear/chop). Don't stack two unproven rules on PAIR_RSI_OB. LONG_UNMATCHED has no data. Widen to universal only if a future weekend shows the other sleeves lose sub-0.10 on real (non-bug) fires.

TIGHT REVERT (N=14 / one-weekend override): set `flip_fan_btc_atr_min`→0 if next weekend's blocked-cohort sub-0.10 fades would have been ≥45% WR on N≥6 (track via the phantom flip tracker, which still observes the blocked side). Blocking FAN does not fully blind the regime test — PAIR_RSI_OB + phantom still fire/observe sub-0.10 next weekend.

Operator will reset paper after deploy for a clean post-change forward window (dedup is reset-proof).

### 2026-06-20 PM — SHIP: decouple PAIR_RSI_OB overbought-fade seed from the BTC_ADX_GATE_HIGH long veto (live, de-risked)

Operator-surfaced design defect: the PAIR_RSI_OB fade (overbought-long blocked at pair RSI>65 → fade SHORT, STRONG_BULL only) was structurally choked above BTC ADX 40. Root cause: the seed lives inside `_signal_block_recorder`, which early-returns whenever a BTC-macro gate vetoes the long (`_btc_macro_blocks_long is not None`). That guard exists for bookkeeping (Filter Blocks should show the decisive gate) + to not fade longs macro already killed — but it's an ACCIDENTAL inheritance for this sleeve: the fade is a SHORT, so the long's BTC-ADX ceiling is irrelevant to it, and overbought pairs are richest exactly when BTC trends hardest (ADX>40). Net effect was a contradiction — the flip's own regime gate WANTS STRONG_BULL (BTC ADX≥28, no upper bound) but the seed died above 40, squeezing the live window to BTC ADX [28,40] and excluding the strongest trends. At BTC ADX 51 the dashboard showed 7/7 long blocks = BTC_ADX_GATE_HIGH → zero seeds → no fades, even though regime/pADX gates would have passed.

FIX (new config field `flip_pair_rsi_ob_btc_adx_high_mode`, off/phantom/live, shipped **live**): when the long's SOLE macro veto is `BTC_ADX_GATE_HIGH` and mode≠off, the recorder seeds the fade THROUGH the veto (suppressing only the redundant pair-block count + decisive-reason stamp, so Filter Blocks stays honest). Phantom seeds always; the LIVE marker (`rsi_ob_flip`) is set only when mode==live. Scoped to BTC_ADX_GATE_HIGH ONLY — every other macro veto (BTC_TREND_FILTER, BTC_ADX_GATE_LOW, slope/cross) corresponds to a non-STRONG_BULL state the flip's own regime gate already rejects, so decoupling those would be wrong.

DE-RISK: the newly-unchoked BTC ADX>40 cohort has ZERO prior data (the seed was never allowed there), so `_flip_filters` returns lev **0.05 (1× eff)** for it (boundary = `btc_adx_max_long`, the same ceiling that defines the seed-through cohort) while the validated [28,40] cohort stays at **1.0 (20×)** — a per-cohort split, not a sleeve-wide de-risk (de-multiplying the just-shipped 20× cohort would chop a winner). PROMOTE the >40 cohort to full lev on its own evidence; the regime (STRONG_BULL) + pADX≥40 gates still apply to it downstream.

D11 full: config.py default "off" + evidence comment, trading_config.json "live", engine (recorder carve-out + `_flip_filters` de-risk; counters unchanged — these are normal PAIR_RSI_OB flips from a new regime), UI cyan select (off/phantom/live) + load + save. Grep-verified id ×3, key in all 4 files, py+json AST clean. REVERT: set mode→off if the >40 fades whipsaw (WR≤45% on N≥8). Note: builds on the same-day ADX-floor fix (the floor must be live for pADX≥40 to mean anything) — this is shipped after a paper reset, so the >40 cohort accrues from a clean window.

### 2026-06-21 — SHIP: PAIR_RSI_OB floor 40→45 + >40-BTC-ADX cohort promoted 1×→20× (N=17 override)

First post-reset batch (export 13-02-28) gave the first real read on the newly-decoupled BTC-ADX>40 PAIR_RSI_OB cohort. The de-risk worked exactly as intended: the 1× (>40) cohort ran N=32/66%WR/−$9 (vs −$182 it would have lost at full 20× — the de-risk saved ~$173). Verified the report's PAIR_RSI_OB × pADX table is correct: **40-45 = 17/47%WR/−$605 (de-mux), 45+ = 17/82%WR/+$347 (+0.20% avg)**. Per-trade in the 45+ band: only 1 of 17 gapped straight to SL (EPIC pADX 46.3); the other 2 losers ARMED (peaked +0.21 / +0.83) → **BE-compatibility = 67% (2/3 losers peak ≥+0.20%), clears the ≥60% lock** for lev-stacking. Mechanism: when BTC trends this hard (ADX 45-55 all batch) only the most extreme pair blow-offs (pADX≥45) mean-revert; 40-45 squeezes through.

CHANGES (operator-directed): ① `flip_pair_rsi_ob_adx_min` 40→**45** (config.py + json) — cut the 40-45 loser band sleeve-wide. ② Removed the de-risk in `_flip_filters` so the BTC-ADX>40 cohort fires at **full 20×** (was lev 0.05); all pADX≥45 STRONG_BULL now fires at 20× regardless of BTC ADX. `flip_pair_rsi_ob_btc_adx_high_mode` stays "live" (still gates the >40 SEED). UI helper text + mode tooltip/option updated (no new field — the floor input + mode select already exist).

DISCIPLINE: N=17/one-batch, below the N≥30 multiplier gate — explicit override (analyst recommended staging 1×→5×→20×; operator chose straight to 20×). The conceding point: the table is right and BE-compat passes, so it's not a confound; the risk is purely the thin N. **TIGHT REVERT: set `flip_pair_rsi_ob_btc_adx_high_mode`→off (stops the >40 seed entirely) AND/OR floor→40 if the next ~6-8 fresh 45+ fires drop below ~65% WR, OR a gap cluster (≥2 straight-to-SL) hits at 20×, OR BE-compat falls <60%.** py+json AST clean; de-risk return removed (grep 0).

NOT shipped this commit: the FAN flip-short fix (analysis below) — operator asked to finish the analysis first.

### 2026-06-21 — ANALYSIS (no ship): FAN flip-short positive-expectancy lever = block BTC-slope<0 × BTC-ADX 30-35

Finished the cross-batch (J17/18/20/21) hunt for a FAN flip-short divider across the three slope tables. Winner: the **BTC Slope × BTC ADX cross-tab**. The cell **BTC EMA20 slope <0 × BTC ADX 30-35** is negative in ALL FOUR batches (J17 −488, J18 −174, J20 −410, J21 −169; pooled N=60/53%WR/−$1241) — the most cross-batch-consistent FAN loser found to date. Mechanism: don't fade a pop when the market is TRENDING (ADX 30-35 bear = strong downtrend, the faded pop is a violent squeeze that gaps the 20× SL); the fade pays only when the market RANGES (BTC slope<0 × ADX<30 = +$766/71%, positive 3/3). Counterfactual (block the 30-35 cell): every batch improves — J17 +1039→+1527, J18 −998→−825, J20 −225→+185, J21 −287→−118; **pooled −$471 → +$770**. It is surgical: it does NOT touch J17's winners (the broader "block ADX≥30" cut sacrifices J17's +964 35-40 fat winner → worse). Regime-scope alternative (block STRONG_BEAR/BEAR_EXH/CHOP, keep H.BEAR) also flips pooled positive (+$653) BUT cuts $882 of J17 PROFIT (J17 STRONG_BEAR was +$552) — too coarse; the 2D cell is cleaner. CAVEAT: even after the block, J18 stays −$825 (its losers are outside the cell) — so this improves the sleeve and likely flips expectancy positive (clears N≥30 + all-4-batch consistency) but is not a guaranteed-positive-every-batch fix. BONUS (separate, multiplier candidate): pair EMA20 slope 0.20-0.30 = N=27/67%/+0.354%/+$1097, positive 3/3. PENDING operator decision to ship the BTC-slope<0×ADX-30-35 block (new `_flip_filters` gate; FAN-scoped).

### 2026-06-21 — SHIP: FLIP_SHORT_PAIR_GAP (pair EMA13-EMA50 gap≥1.0 block) + ADX×RSI table 40→45 retitle + negative-gap dig (NO-SHIP)

**Ship 1 — FLIP_SHORT_PAIR_GAP `flip_short_pair_gap_max=1.0` (commit 62d2d3b, pushed).** Block flip-SHORT when pair (EMA13−EMA50)/EMA50% ≥ 1.0. Replaces the BTC-gap×ADX-30-35 plug as the FAN-short loser cut. PROVENANCE: systematic survivor-pool scan (filter-overlap audit applied FIRST — current stack incl. U3/RSI<55/btc30-rise/regime) over winner-vs-loser separators showed the losers are defined by PAIR extension, not BTC state (losing FAN survivors: pair-ATR 1.38 vs 0.99, pair EMA13-50 gap +0.44 vs +0.12). Bucketing the gap gave a clean monotonic cliff: 0–0.3=+0.36, 0.3–0.6=+0.47, 0.6–1.0=+0.51 (87%WR sweet spot), then **≥1.0 = 16/44%WR/−0.359%avg/Σ−$461**, net-negative every batch it fires (J18 5/40%/−$168, J20 10/50%/−$172, J21 1/0%/−$121). Mechanism: fading a pair already steeply above its own 4h trend = shorting a parabola that keeps ripping → never arms (0.45 peak) → 20× gaps the ATR-widened SL to ~−1.2%. ONE-SIDED by design: operator asked re symmetric |gap| — tested and REJECTED. Full spectrum: extreme NEGATIVE gap is the BEST bucket (≤−1.5 = 3/67%/+0.79%) because a negative gap = pair in a downtrend = the short is WITH-trend momentum, not a fade. Symmetric |gap|≥1.0 keeps +$1,172 vs one-sided's +$1,401 (throws away the negative-side winners incl J17 +$175). gap≥1.0 SUBSUMES the ATR≥2.0 loser cohort (all 10 ATR≥2.0 ⊂ gap≥1.0) and overlaps the old gap×ADX plug only 2/16 → bigger, cleaner, more mechanistic. Pool effect: survivor FAN 68%/+0.095% → 74%/+0.220%/+$1,401. N=16 < the N≥30 Pattern-C gate (WR 44% not ≤40, never-pos 56% not ≥60) → DISCIPLINE-OVERRIDE, justified because BE is OFF so the −1.2% tails can't be capped (block is the only lever, not de-size), mechanism is sound (anti-parabola, the flip-side of the live non-flip pair_trend_short filter tuned for flips), and it's direction-consistent across 3 batches. Engine: new shared flip-SHORT block in `_flip_filters` (after FLIP_SHORT_RSI_MIN; counter FLIP_SHORT_PAIR_GAP); **`pair_gap` added to `_ff_in`** (= entry feature `entry_pair_ema20_ema50_gap_pct`, fallback (ema13−ema50)/ema50 from `_ind`) — guarded against the silent no-op trap that bit the PAIR_RSI_OB ADX floor. PAIR_RSI_OB returns early → unaffected (correct; it's a different setup). D11 full (config.py default 1.0 + evidence comment, trading_config.json 1.0, engine input+filter, templates input id `config-flip-short-pair-gap-max` + _setVal load + save payload). TIGHT REVERT: `flip_short_pair_gap_max`→0 if blocked gap≥1.0 flip-shorts hit ≥50% WR on N≥10 fresh (phantom/passthrough still observes the blocked side).

**Ship 2 — ADX≥40×RSI cross-bucket table retitled to ADX≥45 (reporting-only).** The live PAIR_RSI_OB floor was raised 40→45 in b4b3db1, but the 🎯 ADX×RSI analytics table still computed/labeled ≥40. Changed `main.py` threshold `entry_adx >= 40` → `>= 45`, renamed the response key `adx40_rsi → adx45_rsi` (avoid the misnaming trap), and updated every surface: UI header + column label + tbody id `pair-rsi-ob-adx45-rsi-body` + render + BOTH text-report exports (`_obtab` clipboard + `_obtab2` saved-file, D12 parity) + the stale config-panel footnote (now "ADX≥45 floor; raised 40→45 — 40-45 = 17/47%/−$605 loser, edge = 45+ = 17/82%/+$347"). No behavior change.

**Dig (NO SHIP) — FAN flip-short negative-gap −1.2% bleed = variance.** Operator asked what the with-trend (negative-gap) −1.2% bleeders (BTR/VELVET/EPIC) share, after gap≥1.0 only catches the counter-trend (positive) side. Profiled the post-gap-filter kept pool (58/74%/+$1,401; 15 losers): close_reason 11/15 = FLIP_STOP_LOSS straight-to-SL (−$1,102); per-pair concentration worst-2 = 38% (< the 60% blacklist gate), losses spread across singletons (HEI −130/BTR −111/MET −106/EPIC −85/XLM −74); winner-vs-loser separators all weak (pair-ATR 0.94 vs 0.81, pair-vol $89M vs $120M, BTC-ADX 30.5 vs 27.8 — faint low-vol/high-BTC-ADX tilt, not filterable at N=15); and two of the three named pairs don't hold up (VELVET net WINNER 4/75%/+$68; BTR & EPIC N=1 each). VERDICT: irreducible 20×-short straight-to-SL tail, no separable signature, no concentration → de-leverage territory (deferred scaling roadmap), NOT another entry filter. Do not re-hunt.

### 2026-06-21 — BULL-LONG → 1× OBSERVATION ARM (config) + the BTC-momentum driver dig + BOUNCE-LONG falling-knife finding

**Config change (bull-long obs arm, config.py + trading_config.json):** `bull_long_lev_mult` 1.0→0.05 (20×→1×), `bull_long_regimes` "HEALTHY_BULL"→"STRONG_BULL,HEALTHY_BULL,CHOPPY_FLAT,CHOPPY_WEAK" (bear EXCLUDED), `bull_long_fan_min` 1.65→1.35, `bull_long_fan_max` 3.0→5.0. Existing UI fields (no new D11).

**The dig (why):** operator flagged the Bull-Long Curve keeps shifting batch-to-batch and live bull-longs underperform. Pooled live BULL_LONG cross-batch (J20+J21, N=52, 50% WR, −$391, avg −0.126%) and profiled winners vs losers across every entry variable. Result: **fan bucket & regime do NOT separate** (entry RSI sep 0.00, fan sep 0.01, regime all HEALTHY_BULL anyway) — the driver is **BTC momentum/extension at entry**: (1) BTC 30m-RSI accel (btc_rsi − btc_rsi_prev6) monotonic — <+5 ~29%WR, +5-12 33%, **≥+12 = 26/73%WR/+0.149%**; (2) BTC EMA13-EMA50 gap — <+0.06 ~67%WR, **≥+0.06 = 22/27%WR/−$430 (the entire net loss)**. 2D good cell (Δrsi≥5 & gap<0.06) = 29/69%/+$44 vs REST 23/26%/−$434. Mechanism: a build-side long is a momentum bet — needs BTC accelerating UP with headroom; fails when BTC mildly-rising-but-stretched (20/26 losers DOA, peak <+0.20%). The Bull-Long Curve (fan×regime flat phantom) is structurally blind to this → that's the instability. CAVEAT: N=52 is ~90% from J20 (single-batch; BULL_LONG only exists since Jun 20) → NOT ship-ready; converted to a 1× obs arm to confirm on fresh batches. **2 TRACKING GATES: ① BTC gap≥+0.06 ≤40% WR → ship `bull_long_btc_gap_max≈0.06`; ② BTC 30m-RSI Δ≥+12 ≥65% WR → make it the entry condition.** Both split ×regime to test whether the pattern is regime-invariant (the actual experiment behind broadening regimes). Bear excluded because build-side long-into-bear is the opposite thesis (flip-LONG bear evidence = 2/0%WR/−$220). Fan widened back to 1.35-5.0 because fan is a suspected red herring — open the aperture and let the tracked BTC cells reveal the gate. REVERT: any single bull-long gaps past ~−1.0% OR the arm <40% WR at N≥20 → cut back to HEALTHY_BULL / fan 1.65-3.0 / lev 0.05.

**BOUNCE-LONG falling-knife finding (no config change):** operator asked to apply the same widen-regimes idea to bounce-long. First checked: it had ZERO live fires across J17-J21 (rare trigger btc_rsi≤35 + HEALTHY_BEAR gate). Then the 21:00 batch produced the first 2 live BOUNCE_LONG (NEAR, JTO) — BOTH LOST (−0.71%/−0.69%, straight-to-SL, DOA peak ≤0.08%). Both fired mid-crash: BTC 30m-RSI Δ = −18.5 (NEAR) and −25.5 (JTO) — still plunging hard = FALLING KNIVES. This is the MIRROR of bull-long: bounce-long is a mean-reversion bet that needs the down-move EXHAUSTING (BTC 30m-RSI recovering/decelerating), NOT just oversold. The current oversold-cell gate catches "low RSI" but not "RSI turning." DECISION: do NOT widen bounce_long_regimes (STRONG_BEAR would catch MORE falling knives); the real lever is an exhaustion/turn gate (don't fire while BTC 30m-RSI Δ deeply negative). Kept HEALTHY_BEAR + 1×; watchlist the exhaustion variable. Unifying insight: BTC 30m-RSI momentum is the master variable for BOTH long sleeves — bull-long rides positive acceleration (Δ≥+12), bounce-long must avoid negative acceleration (don't catch the down-spike).

### 2026-06-22 — SHIP: PAIR_RSI_OB pair-gap≥1.0 block + SYN blacklist + BULL_LONG fan-min 1.35→1.0

**Context — PAIR_RSI_OB was the batch's big drainer.** Latest paper batch (CSV 03-26-00): the PAIR_RSI_OB overbought-fade SHORT sleeve = N=22, 41% WR, −0.38% avg, **−$982.88** (all in STRONG_BULL; the entire bullish-section short book IS this sleeve, so every short split = a clean cut of it). Operator asked: is there a single entry variable that divides winners from losers?

**The dig (per-trade CSV, winners included — not bucket aggregates).** ① **Per-pair concentration:** EIGEN (8, 3W/5L, −$521) + SYN (5, 0W/5L, −$424) = **96% of the loss** → trips the locked "≥60% in 1-2 pairs → pair treatment, not dimension filter" rule. ② **Cleanest dimension = pair EMA13-EMA50 gap:** gap<1.0 = 3/100%/+$97 vs gap≥1.0 = 19/32%/−$1,080; ALL 13 losers have gap≥1.148 → gap≥1.0 catches 100% of losers. ③ **EIGEN is dimension-inseparable:** its 3 winners and 5 losers fully overlap on gap/ATR/RSI/RngPos/stretch/pADX; the ONLY separator is peak_pnl (winners armed peak≥0.78, losers gap-down peak≤0.41) = an OUTCOME, not an entry input. The big non-EIGEN winners (PUMP+$77/WLD+$95/ALLO+$31) sit at the SAME high gap as losers → no extension filter can spare them. ④ **Alternatives are worse:** ATR≥0.85 misses 3 low-ATR losers (−$336); RngPos 85-95 & RSI≥85 catch only subsets. ⑤ **Exit-tightening REJECTED:** fast-exit 0.20%/2min CF = +$502 but fires on 5 trades, 4 of them SYN (Real −$436 matches exactly) → REDUNDANT with the SYN block; after SYN it saves only EIGEN −127 (~+$140), leaving −$421. The 7 non-SYN immediate-gap losers (peak<0.20, −$759) can't be exit-saved — entry block is the only lever. Lowering the arm threshold alone barely helps (the +0.10 min-profit gate still rides the SL for sub-0.30 pokes).

**SHIP 1 — `flip_pair_rsi_ob_pair_gap_max=1.0` (new field, full D11).** Block PAIR_RSI_OB flip-SHORT when pair (EMA13−EMA50)/EMA50% ≥ 1.0. PAIR_RSI_OB RETURNS EARLY in `_flip_filters` (so it doesn't inherit FAN's regime gates) → it never saw the universal `flip_short_pair_gap_max` block; this replicates that block inside the PAIR_RSI_OB branch on its OWN field (independent revert clock from FAN). Impact (in-sample): blocks 19/22 (13 losers + 6 inseparable winners), keeps 3 clean low-gap winners (DOT/AVAX/BTC) → cohort **−$983 → +$97**. Cost = 6 forfeited winners (+$288), and the sleeve shrinks to ~14% pass — near-shutting it in S.BULL, acceptable since it's net-losing there. DISCIPLINE-OVERRIDE: N=22 < N≥30, one window — justified because it MIRRORS the cross-batch-validated FAN gap≥1.0 filter (N=16, J17-21, identical anti-parabola mechanism), so it extends a proven gate to the sister source, not a fresh fit. Engine gate in PAIR_RSI_OB branch (counter `FLIP_PAIR_RSI_OB_GAP`, auto-recorded via the line-3229 flip-block path; `pair_gap` already in `_ff_in` from the 06-21 FAN ship). D11: config.py default 1.0 + evidence comment, trading_config.json 1.0, engine gate, templates UI input `config-flip-pair-rsi-ob-pair-gap-max` + _setVal load + save payload. **TIGHT REVERT: `flip_pair_rsi_ob_pair_gap_max`→0 if blocked (gap≥1.0) PAIR_RSI_OB fades show ≥50% WR on N≥10 fresh.**

**SHIP 2 — SYN → pair_blacklist.** SYNUSDT added to `pair_blacklist` (0W/5L this batch, gap ~9-11%, ATR ~3%). NOTE: redundant with Ship 1 on this data (SYN gap ~10 already caught by gap≥1.0); kept as belt-and-suspenders (catches SYN if it ever fires at gap<1.0 in any sleeve) and operator-directed. N=5 single-batch = below the evidence bar → the weaker of the two; re-whitelist if it doesn't repeat as a loser. Global block (whole universe), acceptable since the bot is ~flip-only and SYN is a high-ATR meme.

**SHIP 3 — BULL_LONG `bull_long_fan_min` 1.35→1.0 (config-only, operator-directed).** Widens the 1× observation arm's lower fan bound to 1.0 (fan 1.0-5.0; fan_max stays 5.0, lev stays 0.05/1×). NOTE the opened 1.0-1.35 band is a documented 20× LOSER (06-20: 14/36%WR/−$732) — opened ONLY because the arm runs at 1×/0.05-lev observation (data collection, not sizing). Consistent with the 06-21 "fan is a suspected red herring; track BTC-momentum instead" reframe. REVERT unchanged: any single bull-long gaps past ~−1.0% OR arm <40% WR at N≥20 → cut back.

All three are staged together this commit. py+json AST clean.

### 2026-06-22 — RETIRE-TO-OBSERVATION: PAIR_RSI_OB de-levered 20×→1× (Option B)

`flip_entry_sources` `PAIR_RSI_OB:1.0:1.0` → **`PAIR_RSI_OB:1.0:0.05`** (size 1× unchanged; lev 1.0→0.05 = 20×→~1× actual). After the gap≥1.0 ship cut the parabola tail, the operator asked to dig the post-filter survivors for a winner/loser separator; the dig was thorough and CROSS-BATCH NEGATIVE, so we retired the sleeve to observation rather than keep hunting.

**Why (the full exhaustion case):** PAIR_RSI_OB is a counter-trend overbought-fade SHORT in STRONG_BULL at 20×. It has FAILED its locked source-revert gate (WR≤55% OR avg≤+0.05% at N≥20) by a wide margin — live 33/42%WR/−0.32%avg, deduped cross-batch N=136 net-negative. Every entry/exit lever was tested cross-batch and none yields a stable positive cohort: ① gap≥1.0 — REAL and KEPT (removed the −$1,080 high-gap parabola tail, also the FAN-short filter) but the residual gap<1.0 cohort (N=76) is still net-neg (−$292 at ATR≥0.25); ② pADX≥45 floor decayed (9/89%→13/62%→neg); ③ RSI buckets all neg live (65-70 −$319, 75-80 −$260, 85-90 −$445); ④ regime already S.BULL-gated; ⑤ ATR — a FLOOR at 0.25 INVERTS cross-batch (06-18 low 50% vs high 25%, 06-21 low 100% vs high 75%, clean only 06-22), and the better-defined ATR WINDOW 0.25-0.50 (gap<1.0 cohort = the only positive band, 29/~48%/+$55) is marginal + today-heavy; ⑥ exit — fast-TP only rescues the already-blocked SYN (residual losers are DOA/never-armed). Mechanism diagnosis = ~zero-edge fat-tailed coin flip (same as FAN-flip-short); the only constant across all 6 batches is the 20× gap-through-SL + broken R:R (wins +0.2-0.5% vs losses full −0.7 to −1.2%), which makes ~45% WR net-negative by construction. No entry variable can rescue a cohort whose loser-defining trait (did it arm) is an OUTCOME unseeable at entry.

**Why Option B (de-lever to 1× obs) not A (disable):** operator chose to keep it firing live at 1× so live+phantom data keeps accruing (caps the bleed at ~1× instead of 20×) rather than full phantom-only. **RE-LEVER only on a cross-batch-STABLE (≥3 batch) winner/loser separator — explicitly NOT another in-sample finding.** KEPT: the gap≥1.0 filter, the pADX≥45/regime gates (harmless at 1×), all PAIR_RSI_OB analytics tables, and the ATR-window watchlist (next-batch re-check). Config-only (existing `flip_entry_sources` field); CURRENT_STATE PAIR_RSI_OB line updated with the de-lever + re-lever gate.

---
## 2026-06-22 — BULL-LONG: narrow live aperture (fan_max 5.0→2.0) + drop CHOP regimes + S/H/o analytics column
**Context:** 06-22 batch (N=32 bull-longs @1×/0.05-lev observation). Both 06-21 BTC-momentum tracking gates FAILED/INVERTED on fresh data (gap≥+0.06 "loser" → 4/100%/+0.745%; 30m-RSI Δ≥+12 "winner" → 11/55%/−0.135%) → N=52 dig was single-batch overfit; neither ships. The recurring (but confound-flagged) shape: regime > fan.

**Live realized (avg% lev-invariant):**
- S+H BULL = 14/64%/+0.155% (S.BULL 7/57%/+0.20, H.BULL 7/71%/+0.11) · CHOP = 18/56%/−0.247%/−$482 de-mux (>½ sleeve volume = the whole drain).
- Fan: 1.0-1.35 5/60%/−$96 · 1.35-1.65 8/75%/+$95 · 1.65-2.0 8/75%/+$305 · 2.0-3.0 9/33%/−$452 · 3.0-5.0 2/50%/−$100. Tighter: S+H BULL fan 1-3 = 13/69%/+0.262% · fan 1-2 = 10/80%/+0.422%.

**Changes (operator-directed, config-only — existing UI fields):**
1. `bull_long_fan_max` 5.0→**2.0** — narrow LIVE aperture to fan 1.0-2.0 (current best-belief band; fan>2 gives bull-regime wins back this batch). ⚠ IN-SAMPLE: fan 2-3 was the 06-20 WINNER (10/90%/+$705) → INVERTS cross-batch. Safe only because (a) obs-lev 0.05×, (b) phantom Bull-Long Curve still tracks fan 2-5 regardless of live config.
2. `bull_long_regimes` STRONG_BULL,HEALTHY_BULL,CHOPPY_FLAT,CHOPPY_WEAK→**STRONG_BULL,HEALTHY_BULL** — drop CHOP (the whole live drain). Bull regimes alone are +0.155%.
3. **Analytics:** new `S/H/o` column on the 🐂 Bull-Long × Fan Bucket table = per-bucket entry-count split S.BULL/H.BULL/other (other=CHOP+bear), reconciles to N. Server `_compute_bull_long_trades` adds sbull_n/hbull_n/other_n via existing `_rb()` (zero new Order columns). D12: UI table + BOTH text exports (grep-verified: FAN BUCKET ×2 exports, S/H/o ×3 surfaces).

**Framing:** NOT a hard rule / not a re-lever. Stays at 0.05× observation. Companion CURRENT_STATE "🔭 OBSERVATION-FOR-REVIEW" note records the future-review trigger: IF S+H BULL fan ≤2.0 reaches N≥30 cumulative across ≥2 batches holding WR≥70%/avg≥+0.15% → THEN consider re-lever. Until then keep collecting.
**Files:** config.py, trading_config.json, main.py, templates/index.html, CLAUDE_CURRENT_STATE.md.

---
## 2026-06-22 — FAN flip multiplier DROPPED: strong-bear 2× cell → 1×
**Change:** `flip_fan_mult_rule` `"40-45:35-99:2.0:1.0"` → **`"40-45:35-99:1.0:1.0"`** (BTC RSI 40-45 × BTC ADX≥35 strong-bear cell; size 2.0→1.0). config.py + trading_config.json + CLAUDE_CURRENT_STATE.md (FAN-flip line ④).
**Why:** First fresh live fire was ✗ — EIGENUSDT SHORT −1.206% / **−$209.02** (the batch's worst loss); the 2× doubled a −$104.51 1× baseline loss. The trade peaked **+0.166%**, which FAILS the locked **2× BE-compatibility gate** (≥60% of a cell's losses must peak ≥+0.20% so caps can bound the tail) → caps structurally couldn't bound it, which is exactly why 2× hurt. Cell was already a below-N≥30 discipline override. Multiplying a fade-short amplifies the gap-through-SL tail = the realized risk.
**Discipline note:** Formal multiplier verdict gate is ≥5 fresh fires (we're at N=1), so not a formal "✗ HARMFUL" call — but de-levering a loser is the asymmetric always-allowed side, AND the BE-compat gate is independently failed regardless of N. The FAN_RATIO_GATE flip itself stays 1× (validated winner, +$548 this batch) — only the amplifier is removed.
**Re-earn gate:** restore a multiplier only at N≥5 fresh fires + Total$>0 + BE-compatibility (≥60% of losses peak ≥+0.20%). Existing UI field (Invest Multi) — no new D11 wiring.

---
## 2026-06-22 — BULL-LONG RE-LEVERED 1×→20× + band restricted to fan 1.35-2.0 (operator discipline-override, fresh-reset test)
**Change:** `bull_long_lev_mult` 0.05→**1.0** (1×→20×), `bull_long_fan_min` 1.0→**1.35** (fan_max=2.0, regimes=STRONG_BULL,HEALTHY_BULL unchanged). Live cell = **S+H BULL × fan 1.35-2.0 @ 20×**. config.py + trading_config.json + CLAUDE_CURRENT_STATE.md. Operator will RESET paper + start a fresh batch on this config.
**Evidence (06-22, bull-only fan 1.35-2.0):** 8/88%WR/+0.554% avg — the in-band sweet spot (fan<1.35 = 5/60%/−$96; fan>2 = SL losers; CHOP half of the band = −0.091%, excluded by the regime gate).
**⚠️ DISCIPLINE-OVERRIDE — acknowledged transparently:** N=8 bull-only < the locked N≥30 multiplier/re-lever gate, AND 0.05→1.0 is a 20× jump that skips the mandated 1.5×-first staging. Same override profile as the PAIR_RSI_OB / earlier BULL_LONG 20× ships. Analyst recommendation was HOLD at 1× for one more session then ship 1.5× first; operator chose 20× now. Justification accepted: a FRESH RESET gives clean single-config attribution (the cleanest possible test surface), and the in-band edge is strong this batch.
**Cross-batch risk:** bull-long sub-bands have demonstrably inverted (fan 2-3 = 10/90%/+$705 on 06-20 → 9/33%/−$452 on 06-22). 88%/N=8 may regress. 20× build-side long carries gap-through-SL exposure on the ruin axis.
**TIGHT REVERT (tighter-than-standard per override rule):** de-lever `bull_long_lev_mult`→0.05 at live N≥10 NEW bull-longs if fan 1.35-2.0 S+H-bull WR≤70% OR avg≤+0.15%; INSTANT de-lever if 3 of first 6 hit SL OR any single bull-long gaps past ~−1.0%. Existing UI fields — no new D11.

---
## 2026-06-22 — Blacklist EIGENUSDT (structural SL-gapper insurance, operator-directed)
**Change:** `pair_blacklist` += `EIGENUSDT` (trading_config.json; now 13 pairs).
**Rationale (explicitly NOT the batch $):** EIGEN was 35% of this batch's loss (−$729.6/9 trades), BUT 8 of 9 were gap≥1.0 PAIR_RSI_OB trades the gap filter already blocks + the 9th's 2× is already dropped → re-sim under the current stack = ~−$104 go-forward. So the −$729 does NOT justify a block (already neutralized). The REAL basis: EIGEN is a structural **SL-gapper** — every loss closes ~−1.2%, blowing through the −0.70 stop by ~0.5%, INCLUDING a clean low-gap (0.28) FAN flip. That's the illiquid-meme-gapper profile the existing blacklist targets (ENA/RAVE/VVV/ESPORTS). Matters more now that bull-long is at 20× (a gap-through is a ruin-axis tail). **Caveat (acknowledged): the filter-RESISTANT evidence is N=1** (the one low-gap gap-through); the gap filter already catches EIGEN's high-gap losses. This is an insurance call, low opportunity cost (gappy alt, not a core pair), not a data-proven dimension filter. **REVERT if needed:** remove from `pair_blacklist` (no code). Existing field — no D11.

---
## 2026-06-23 — SHIP: FAN flip pair-ADX floor (flip_fan_pair_adx_min=20)
**Change:** new `flip_fan_pair_adx_min: float = 20.0`. Blocks FAN flip-SHORT when entry pair ADX < 20 (counter `FLIP_FAN_PAIR_ADX`, gate in the FAN-SHORT branch of `_flip_filters`, reads `ind['adx']`). config.py + trading_config.json + engine + UI input (`config-flip-fan-pair-adx-min`, same row format as the gap filter) + load/save. ConfigUpdate.thresholds is `Optional[Dict]` → generic, no whitelist edit.
**Mechanism:** FAN flips bypass the momentum short system's pair-ADX requirement (`Pair ADX Dir: rising` + ADX-Strong>20), so they fire weak-trend fades (pADX 15-19) with no follow-through that chop/gap the 20× SL. The floor restores what momentum already enforces — principled, not data-mined.
**Evidence — 3-batch deduped N=89** (J20 pre-reset, J22 20:28, J23 00:54 — all ran the identical FAN filter stack since 06-16):
- pADX≥20 KEEP = 42/71%WR/+$482 · pADX<20 BLOCK = 47/51%WR/−$850 (the whole drain).
- Per batch KEEP/BLOCK: J20 27/67%/+$94 vs 34/56%/−$319 · J22a 11/82%/+$399 vs 8/62%/+$24 · J22b 4/75%/−$11 vs 5/0%/−$555. KEEP>BLOCK + WR-up in all three.
- Kept cohort N=42 clears the N≥30 gate at 71% WR.
**Per-pair concentration check (PASSED):** block gross loss −$2,260 diffuse across ~17 pairs — top-1 (MET) 18%, top-2 28%, top-3 38% (well under the 60%/1-2-pair blacklist threshold → dimension filter is correct). MET decomposes: ADX<20 = 4/−$396 vs ADX≥20 = 1/+$153 (bad-condition, not bad-pair; a blacklist would wrongly kill the winner). SAND only fired <20 (all losers) → correctly cut.
**Cost acknowledged:** sacrifices the occasional sub-20-ADX winner (HU = 6/+$204 at low ADX). Diffuse loss + 3-batch consistency + mechanism outweigh it.
**TIGHT REVERT:** set `flip_fan_pair_adx_min`→0 if pADX≥20 FAN flips drop ≤60% WR on N≥15 fresh.
**Context:** chosen over a regime-conditional fast-exit (operator-rejected — fast-exit capped winners on the winning batch for only +$58 and was a weak/asymmetric lever). The pADX floor is surgical (cuts weak-trend entries, not winners broadly).

---
## 2026-06-23 — SHIP (analytics): Entry Conditions by Strategy + by Strategy×Outcome
**What:** two new report tables below "Entry Conditions by Outcome", mirroring its full wide column set but grouped by `entry_strategy` sleeve: (1) "Entry Conditions by Strategy" = one row per sleeve×direction; (2) "Entry Conditions by Strategy — Winners vs Losers" = W/L profile within each sleeve. Strategies: FAN_RATIO_GATE / PAIR_RSI_OB / LONG_UNMATCHED_ONLY (FLIP: prefix stripped) · BULL_LONG · BOUNCE_LONG · MOMENTUM.
**Why:** operationalises the per-sleeve winner-vs-loser separator analysis we kept doing by hand from the CSV (the FAN-flip pADX≥20 dig, bull-long separators). The W/L-by-strategy table = read it first each batch to spot a dimension where winners≠losers within a sleeve.
**Build:** nested `_ec_row(group, direction)` in `_compute_performance` returns the full ~40-col entry-condition dict (SL profile computed over the group's losers so it works for mixed groups); existing two tables left untouched (no refactor risk). Payload keys `entry_conditions_by_strategy` + `entry_conditions_by_strategy_outcome` (+ empty-default in both early-return paths). D12 complete: UI 2 tables (header cloned from the by-Outcome thead via JS, shared `_stratRow` renderer wrapped in try/catch so a render bug can't break the dashboard) + BOTH text exports (clipboard `_ecStratLine` + saved-file `_ecStratLine2`). Read-only — zero engine/config/risk.
**Verify:** main.py parses; D12 grep — export header ×2, UI bodies wired, payload keys ×3 (1 real + 2 fallback).

---
## 2026-06-23 — Blacklist VELVETUSDT (structural SL-gapper, operator-directed)
**Change:** `pair_blacklist` += `VELVETUSDT` (trading_config.json; now 14 pairs).
**Rationale (structural gapper, same basis as EIGEN — NOT chasing the $):** cross-batch (deduped, 06-07→06-22, N=18): 55% WR but net **−$690** = high-WR/net-negative from a fat tail. ALL 8 losers close **−1.19% to −1.22%** — i.e. every loss gaps through the −0.70 SL by ~0.5%; **4 of 8 are DOA** (peak +0.00, straight to SL). Illiquid-meme-gapper signature (EIGEN/SYN profile). Damage concentrated in FAN flips (−$597 of −$690). **The shipped pADX≥20 filter only catches 1 of the 4 FAN-short losers** — VELVET gap-throughs at pADX 22-24 too (−$258/−$133/−$118 survive the floor), so the entry filter isn't enough; and wider-SL was shown not to help gap-throughs (price jumps past any stop). At 20× the −1.2% gap-throughs are a ruin-axis tail. Only forgone upside = the de-levered bull-long (+$6 at ~1×), negligible. **REVERT:** remove from `pair_blacklist` (no code). Existing field — no D11.

---
## 2026-06-23 — DISABLE PAIR_RSI_OB (removed from flip_entry_sources)
**Change:** `flip_entry_sources` `"FAN_RATIO_GATE:1.0,PAIR_RSI_OB:1.0:0.05"` → **`"FAN_RATIO_GATE:1.0"`**. config.py + trading_config.json. Operator-directed.
**Why — every lever exhausted, verdict converged:** under the FULL current filter stack (gap<1.0 + pADX≥45 + STRONG_BULL) the survivors are a zero-edge breakeven **N=22 / 64% WR / +0.003% avg** — the filters already extracted whatever sub-edge existed. No remaining lever recovers a positive edge:
- **Fast-exit (0.2%/5min) fails:** of 76 cross-batch losers, 54% DOA (peak<0.05%) + 22% peak 0.05-0.20% → **76% never reach +0.2%** so fast-exit can't fire; it saves only 11/76 while capping 11/67 winners → net +0.04%/trade = noise. (The losers go straight-to-SL because the overbought pair keeps ripping — an ENTRY failure no exit fixes.)
- **dist-from-EMA13 is a real separator but redundant:** cross-batch (N=143) losers more extended (1.34% vs 0.99%), >0.7% zone = 39%WR/−0.39% drain, sweet spot 0.4-0.7% only positive bucket — BUT 74% of dist≥0.7 is already gap≥1.0-blocked, and on the full stack only 1 of 22 survivors has dist≥0.7 (a winner) → adds 0.
**Status:** disabled, not deleted. Phantom seed stays decoupled (always fires) → keeps tracking. RE-ENABLE only if the phantom shows a cross-batch-STABLE (≥3 batch) separator the live filters don't already capture. Existing field — no D11.

---
## 2026-06-23 — DISABLE BOUNCE_LONG (bounce_long_enabled=false)
**Change:** `bounce_long_enabled` true→**false**. config.py + trading_config.json. Operator-directed.
**Why — no edge, both entry theses falsified cross-batch (N=8 deduped = 2W/6L / −0.443% avg):**
- **dist-from-EMA13 does NOT separate:** WIN mean −0.597 vs LOSS −0.741, but driven entirely by one outlier (WU −1.80). Strip it and the 5 remaining losers (−0.39..−0.69) bracket both winners (−0.59/−0.60); 2 losers (XRP −0.39, WLD −0.49) were *closer* to EMA13 than both wins.
- **BTC 30m-RSI Δ (exhaustion-gate thesis) is INVERTED:** both winners fired at Δ≈−16 (deeply negative / still plunging); the two mildest deltas (−7/−8) were losers → "fire only when the impulse decelerates" would have blocked both winners. BTC 1h Δ doesn't separate either.
- Winners are indistinguishable from losers on every captured dimension = no-edge counter-trend long. Same verdict + same action as PAIR_RSI_OB (disabled same day).
**Tracking-watchlist deliberately NOT added:** the candidate fields (entry_btc_rsi_1h/_prev, entry_btc_rsi_prev6, entry_dist_from_ema13_pct) already exist on every Order and were just falsified — nothing to instrument.
**Status:** disabled, not deleted. Phantom seed stays decoupled (always fires) → keeps tracking. RE-ENABLE only on a cross-batch-STABLE (≥3 batch) separator. Existing field — no D11.

---

## 2026-06-23 — FAN flip pair-ADX floor RAISED 20→21 (commit af29d28)

**Change:** `flip_fan_pair_adx_min` 20.0 → 21.0 (config.py default + trading_config.json). Value-only; engine gate + UI input already shipped same day with the @20 floor.

**Trigger:** operator flagged BICOUSDT FAN flip-short, 2026-06-23 13:31 — lost −1.222% (−$124.59) in 20 seconds, never armed (peak 0.00%), straight to FLIP_STOP_LOSS L1. pADX at entry = 20.6 (cleared the @20 floor by a hair). ATR 1.16%, pair gap −2.07% (with-trend, negative → not caught by the gap≥1.0 block), range-pos 96%.

**Cross-batch dig (4 batches J20-23, deduped by (opened_at,pair,direction), CLOSED only, N=102 FAN flip-shorts; non-blacklisted base N=92):**
- **Reframe: 97% of all FAN-short loss is the never-armed (peak<0.45) straight-to-SL gap-through bucket** (38 trades, −40.1% of −41.2% total loss). Armed-then-lost = 4 trades / −1.1%. → entry-selection problem, not exit.
- **pADX floor sweep (non-blacklisted):** 20 = 42/71%/+0.123/+5.16 · **21 = 36/75%/+0.207/+7.47** · 22 = 27/78%/+0.203/+5.47 · 23 = 19/79%/+0.229/+4.36.
- **20-21 marginal band = N=6/50%WR/−2.31%** (BICO −1.22 + HEI −1.19 never-armed gappers + HOME −0.82, vs only WLD/XMR/JTO +0.93) → net-negative, the 20 floor leaked it.
- **21-22 band = N=9/67%WR/+1.99% (POSITIVE, incl. ORDI +2.80% runner)** → going to 22 sacrifices real upside and flips J20 to −2.8. So 21 is the stop.
- Per-date floor 21: J18 +1.3 · J19 −0.9 · J20 +0.9 · J21 +3.2 · J22 +1.2 · J23 +1.8 = positive 5/6.

**Rejected alternatives:** ATR≥1.1 cut — fails anti-overfit (non-monotonic mid-hole, ≥1.5 breakeven; single-batch J20-dominated; 66% of its loss in 2 pairs BICO/HEI → per locked rule that's a pair-blacklist signal not a dimension filter). De-lever — declined by operator (P&L% is leverage-invariant; sleeve is net-positive under current stack, so the edge wins at all leverage). Residual after pADX≥21 is pair-concentrated (BICO/HEI/RIF recurring never-armed gappers) → future per-pair watchlist, not shipped.

**Discipline notes:** in-sample on the same 4 batches; deciding band N=6 marginal (30-50% haircut → real ~+1.2 to +1.6%); 2nd finer re-cut on a dimension shipped the SAME day (re-overfit risk acknowledged, same pattern as the old PAIR_RSI_OB 33→40→45 saga). Mechanism sound (BICO/HEI gappers cluster just above 20).

**TIGHT REVERT:** `flip_fan_pair_adx_min`→20 if 20-21-band FAN shorts come back ≥55% WR on N≥8 fresh; →0 if pADX≥21 FAN flips drop ≤60% WR on N≥15 fresh.

---

## 2026-06-23 — BULL-LONG DE-LEVERED 20×→1× obs (INSTANT revert gate tripped)

**Change:** `bull_long_lev_mult` 1.0 → 0.05 (config.py + trading_config.json). Band/regime config (S+H BULL × fan 1.35-2.0) unchanged — leverage only.

**Trigger:** the 06-22 operator re-lever to 20× (shipped on N=8/88%WR/+0.55% in-sample, fresh-reset) failed its locked INSTANT revert gate on the very next batch (orders CSV 2026-06-23 14:46):
- BULL_LONG = **7 trades / 29% WR (2W/5L) / −$335.24**, ALL HEALTHY_BULL.
- INSTANT gate arm ① "≥3 of first 6 hit SL" → **4 of first 6 hit SL** (AAVE/TNSR/ALLO/XLM).
- INSTANT gate arm ② "any single gaps past −1.0%" → **TNSR −1.19%, ALLO −1.01%** both gapped.
- Per-trade: BEL −0.28(TRAIL) · AAVE −0.70(SL) · ONDO +0.19(W) · TNSR −1.19(SL) · ALLO −1.01(SL) · XLM −0.73(SL) · LINK +0.27(W).
- The fan **1.65-2.0** sub-band = 5/20%WR/−$264 — inverted exactly as the cross-batch warning predicted (06-20 fan 2-3 = 90%/+$705 → 06-22 loser; same instability one band over).

**Mechanism:** broken R:R amplified by 20× — wins +0.19/+0.27% vs losses −0.70 to −1.19%. In H.BULL chop the build-side long got stopped repeatedly; 20× turned −0.7 to −1.2% price moves into −$68 to −$116 hits.

**ATR check (operator's open watchlist):** NOT the separator this batch — the 2 winners (LINK 0.31, ONDO 0.41) were the LOWEST-ATR; losers averaged ATR 0.82. Argues against a low-ATR floor; pair-ATR<0.3 watchlist still has zero in-sample fires (lowest was 0.31).

**Discipline:** pre-committed INSTANT gate = automatic, no re-litigation ("pre-committed gates do not move"). The re-lever was an acknowledged N=8 in-sample discipline-override; it regressed on first fresh data exactly as flagged.

**RE-LEVER gate (tightened):** only at N≥30 cumulative AND WR≥70% AND avg≥+0.15% across ≥2 batches (cross-batch-STABLE). Never again on a single in-sample band. Sleeve stays at 1×/0.05 observation, collecting.

**Also in this terrible batch (logged, not yet actioned):** BICO flip-short −$125 (pADX 20.6, already blocked going forward by today's flip_fan_pair_adx_min=21) · AAVE FAN flip-short −$86 in H.BULL pADX22.5 ADXΔ+0.47 (leaked FLIP_SHORT_REGIME which needs ADXΔ<0 + S.BULL-only B2 → the standing flip-universal-regime-gate gap) · HYPE UNMATCHED-LONG 2× −$62 (Δ$ vs 1× −$28, the ✗-harmful 2× cell accruing toward its N≥5 demote gate).

---

## 2026-06-23 — FAN flip-short CHOPPY_FLAT soft spot (WATCHLIST, no ship — re-sim refuted raw signal)

**Trigger:** fresh post-reset batch — 2 ALLO FAN flip-shorts both lost (CHOP regime, pADX 22.6/24.3, ADXΔ +0.28/+0.87, both never-armed → −$96/−$97). Both passed the new pADX≥21 floor and leaked FLIP_SHORT_REGIME (which only blocks bull/chop when ADXΔ<0). (Also confirmed same batch: BULL_LONG de-lever working — BEL bull-long ran at 1× = −$5.97 vs ~−$119 at 20×.)

**Raw read (multi-era pool):** CHOPPY_FLAT FAN flip-short = N=44/43%WR/−$1,352; the ADXΔ≥0 slice that leaks the gate = N=33/45%/−$1,085.

**Re-sim under CURRENT filter stack (pADX≥21, fan<10, stretch∈[0.12,2), BTC RSI60&ADX30, BTC30-rise, pair-gap≥1, pair-RSI≥55, BTC-ATR≥0.10, ATR≥3-bear, regime∧ADXΔ<0):**
- FAN flip-short survivors = **N=50 / 74% WR / +$802** (raw −$3,103 was stale; sleeve net-positive under current stack — matches the 06-21 filter-overlap audit).
- By regime: H.BEAR 13/92%/+$881 · S.BEAR 18/78%/+$284 · H.BULL 5/60%/−$7 · **CHOPPY_FLAT 7/43%/−$317** (only negative regime).
- CHOP-block counterfactual: +$802 → +$1,119 / 79% WR (removes 7 = 3W/4L).

**Why NOT shipped:**
1. N=7 ≪ N≥30.
2. CHOP survivor loss is **83% in 2 pairs** (ALLO −$192, PLAY −$170) → locked rule: ≥60% in 1–2 pairs = pair-blacklist signal, not a dimension/regime filter.
3. Pair-blacklist also fails: ALLO's CHOP winners (J20 ×2, pADX 17.7) are now pADX≥21-blocked → only N=2 ALLO survivors (today's losers); PLAY N=1.
4. Per-date CHOP survivors mixed (J15 −, J18 +, J22 −, J23 −), tiny N each.

**Decision (locked) at N≥15 cross-batch CHOP survivors:** (a) diffuse → add CHOPPY_FLAT to `flip_short_regime_block_any_adxd_regimes`; (b) ALLO/PLAY-concentrated → 2-pair blacklist; (c) reverts positive → nothing. Until then observation-only; no config touched. ALLO itself NOT blacklisted (net-mixed, winners would be forfeited).

---

## 2026-06-23 — BULL_LONG concurrency cap (bull_long_max_concurrent=3)

**Change:** new field `bull_long_max_concurrent` = 3 (0 = uncapped). Caps the number of *concurrent OPEN* BULL_LONG positions. Operator-requested.

**Rationale:** BULL_LONG is a de-levered 1×-observation sleeve. In a bull cluster many pairs fan up at once (the 06-23 batch opened 5-7 bull-longs within ~5 min), which can fill the entire max-5 book and crowd out higher-conviction MOMENTUM / UNMATCHED longs. Capping at 3 reserves ≥2 slots for the proven sleeves. A low-conviction obs sleeve monopolizing the position book is backwards.

**Engine:** gate in `_maybe_open_bull_long` (after the per-pair cooldown, before sizing) — counts `Order.status=='OPEN' & entry_strategy=='BULL_LONG'`; if ≥ cap, `_record_filter_block('BULL_LONG_MAX','LONG')` and return. Soft cap (count-then-open not atomic, but the per-pair/30min cooldown + open_position's own max-open make clustering rare; fine for an obs sleeve).

**D11 full:** config.py default (3) + evidence comment · trading_config.json (3) · engine gate · UI input `config-bull-long-max-concurrent` · load + save handlers. Verified: field across 3 code surfaces, UI id count=3, python compiles, JSON valid.

**Caveat:** caps concurrency, not trade quality — assumes a better long is waiting when bull-longs fill up. The entry-funnel cap-cost counter (1 norm / 0 flip last batch) says the 5-cap rarely binds on the good cohort historically, so the benefit is cluster-specific, not per-batch. Cheap insurance, not a P&L lever. Watch the new BULL_LONG_MAX counter to confirm it earns its place. Reversible: set →0 to uncap.

---

## 2026-06-23 — BULL_LONG sleeve DISABLED (bull_long_enabled=false)

**Change:** `bull_long_enabled` true → false (config.py default + trading_config.json). Operator-directed.

**Rationale:** the build-side bull-long sleeve never established a stable edge — it was re-levered to 20× on an N=8 in-sample band (06-22), tripped its instant revert gate the very next batch (7t/29%WR/−$335), and was de-levered to 1× observation. Keeping a no-edge 1×-obs sleeve consuming slots in the max-5 book competes with the proven FAN-flip + MOMENTUM/UNMATCHED longs. Decision: turn it OFF entirely and dedicate the book to what works; re-introduce bull-long later as a clean, isolated EXPERIMENT rather than a permanently-on drain.

**Data retention:** the phantom Bull-Long Curve (virtual longs by fan×regime) is seeded independently of `bull_long_enabled`, so observation data keeps accruing while the live sleeve is off. The de-lever (lev_mult 0.05) and concurrency cap (max_concurrent 3) remain set but inert.

**Re-enable bar:** a cross-batch-STABLE (≥3 batch) band/regime edge in the phantom curve — not another single-in-sample band. When re-enabled, start as a 1× observation experiment, never straight to leverage.

**Config-only** (existing fields). Related: BULL_LONG de-lever + concurrency cap (same day, both now inert under the disable).

---

## 2026-06-23 — FAN flip-short CHOPPY_FLAT block (all ADXΔ) — operator discipline-override

**Change:** `flip_short_regime_block_any_adxd_regimes` "" → "CHOPPY_FLAT" (config.py + trading_config.json). Config-only — the B2 "block-any-ADXΔ" path in `_flip_filters` (line ~366) was already wired (used for STRONG_BULL Jun-17, emptied Jun-19). Now blocks ALL flip-SHORTS in CHOPPY_FLAT regardless of ADXΔ. Universal flip-short gate; FAN is the only live flip source so in practice it's the FAN gate. CHOPPY_FLAT remains in `flip_short_regime_block_regimes` too (ADXΔ<0 path) — redundant now, harmless (any-set check fires first).

**Why:** the existing regime gate only blocked bull/chop when ADXΔ<0; the CHOP∧ADXΔ≥0 slice leaked and lost (today's 2 ALLO shorts −$192, both pADX≥21, ADXΔ +0.28/+0.87, never-armed). Mechanism: a FAN fade needs downward follow-through to profit; CHOP has no trend → the fade never arms (peak ≈0) → 20× SL whipsaw.

**Evidence (current-stack re-sim, FAN flip-short survivors):** total N=50/74%WR/+$802; by regime CHOPPY_FLAT is the ONLY negative one (N=7/43%/−$317). Raw (pre-filter) CHOP = N=44/43%/−$1352; ADXΔ≥0 leak = N=33/45%/−$1085. Bear regimes carry the edge (H.BEAR +$881, S.BEAR +$284).

**Discipline-override (acknowledged, per CLAUDE.md):**
- N=7 current-stack survivors ≪ N≥30 gate.
- CHOP survivor loss is 83% in 2 pairs (ALLO −$192, PLAY −$170) → the locked rule says pair-concentration ⇒ blacklist not dimension filter. BUT neither pair clears blacklist individually: ALLO = 50% WR coin-flip with a +$404 fat-tail winner; PLAY = stale (last 06-15), 44% WR under current stack, half its raw loss already filtered. So a pair blacklist isn't available — the regime gate is the lever, and the mechanism (chop hostile to fades) is sound + consistent with the bear-pays/bull-loses gradient.
- Operator-directed after seeing this analysis (I had recommended watchlist).

**TIGHT REVERT (tighter than standard):** remove CHOPPY_FLAT from `flip_short_regime_block_any_adxd_regimes` if would-be-blocked CHOP FAN-shorts hit ≥50% WR on N≥10 fresh (phantom still seeds the blocked side; watch FLIP_SHORT_REGIME counter + the CHOP phantom row). Cost: clips the occasional CHOP winner (HU +1.71).

---

## 2026-06-23 — BULL-LONG revived as a zero-risk OBSERVATION TEST (flat -0.70 SL)

**Change:** `bull_long_enabled` false→true @ 1× obs (`bull_long_lev_mult=0.05`), `bull_long_fan_max` 2.0→3.0 (now S+H BULL × fan 1.35-3.0), cap 3 (already), + NEW field `bull_long_fixed_sl=-0.70`. D11 full (config.py + json + engine + UI input `config-bull-long-fixed-sl` + load/save).

**Why a test, and why now:** after exhausting the long-side hunt this week (bull-long extension, pullback N=0, un-throttle unmatched 51% base rate, regime-split matched patterns all negative), the one open lever left was the EXIT. Re-sim of the 109 actual bull-longs under the flat phantom exit (arm0.45/trail0.25/SL-0.70) vs their live normal exit: ALL -0.151→-0.078, H.BULL -0.188→-0.161, **S.BULL +0.077→+0.365** — the flat -0.70 SL (tighter than the live ATR-widened -1.20) caps the gap-through tail and materially helps. It doesn't flip the cohort positive overall (still -0.078 even optimistically), but S.BULL turns clearly positive and the live disaster's -1.0/-1.2 losses would've been -0.70. So: test bull-long under the BETTER exit, at 1× (zero ruin risk — the -$335 disaster was the 20×), to definitively settle it.

**Engine:** in open_position's `if bull_long:` block, override `_pcell_fixed_sl = bull_long_fixed_sl` when negative → stamped as `pattern_fixed_sl_pct` on the order + cache → enforced by the existing PATTERN_FIXED_SL realtime exit (fires at -0.70, before the normal -1.20). Arm/trail unchanged. 0 = off (normal exit).

**Scope rationale:** S+H BULL only (drop CHOP -0.155 even flat, drop bear = long-into-downtrend wrong-thesis). Fan 1.35-3.0 (exclude confirmed losers <1.35 [25%WR] & ≥3.0; cross-batch N=86 shows 2-3 ≈ 1.65-2.0 so widen from 2.0→3.0 to not pre-fit). Cap 3 keeps it off the FAN/MOMENTUM book.

**LOCKED DECISION GATE:** run to N≥30 per regime (S.BULL/H.BULL tagged separately) across ≥2 batches. A cell graduates to consider-leverage ONLY at WR≥65% AND avg≥+0.10% cross-batch under the flat SL. NEVER re-lever on test data alone (the exact 06-22 N=8 mistake). Two clean outcomes: (a) a found long-sleeve, or (b) the non-edge confirmed under the best exit → close the question and commit bull-market growth to the parked Donchian module. Cost at 1× obs ≈ $0.

---

## 2026-06-24 — ALLOUSDT BLACKLISTED (per-pair lever) + `flip_fan_btc_adx_min` REFUTED (no ship)

**Action:** added `ALLOUSDT` to `pair_blacklist` (trading_config.json). No other config change. Resolves the 2026-06-23 "FAN flip-short CHOPPY_FLAT soft spot / ALLO+PLAY" watchlist.

**Context:** operator asked whether a BTC-ADX floor on FAN flip-shorts (`flip_fan_btc_adx_min`) should ship, after the 06-24 batch showed a clear late-night loss cluster (BTC ADX ~20-23) vs a morning winning cluster (BTC ADX 25-39), corroborated by the batch's "Performance by BTC ADX" table (20-25 = 16.7%WR/−$330; 25-30 = 100%/+$216; 35-40 = 100%/+$124) and the 5m×1h slope-alignment split (aligned-down +$125 vs counter-trend −$200).

**Cross-batch test (deduped pool, CLOSED, FROM 2026-05-04, entry_strategy=FLIP:FAN_RATIO_GATE ∧ SHORT, re-sim under CURRENT stack = drop CHOPPY_FLAT + pair-ADX<21): N=69.**
- BTC ADX buckets: <22 N=7/43%/−1.91% · 22-25 N=10/70%/+1.36% · 25-30 N=14/71%/+3.63% · 30-35 N=23/65%/−0.70% · ≥35 N=15/67%/−1.28%. **Non-monotonic** — low AND high negative, mid-band (22-30) positive. Per locked rule = confound, not a single-variable edge. The ≥30 leg is high-WR-but-net-losing (fat-tail) → fix sizing/pairs, not block.
- **Overlap audit (decisive):** `<22 ALL pairs = N=7/43%WR/−1.91%` vs `<22 excl ALLO+LAYER = N=4/75%WR/+1.11%`. The entire "<22 loses" signal is ALLO+LAYER. `<25 excl ALLO+LAYER = N=12/75%WR/+2.50%`. A <22 block would chop NEAR/APT-type winners to catch losers the ALLO blacklist removes surgically. **Locked per-pair-concentration rule fires (~100% of loss in 2 pairs ≫ 60% threshold): pair blacklist, not dimension filter.**
- **Time-window evidence = same two pairs, not BTC ADX:** 06-24 late-night losers were LAYER (bADX19.7, −1.20) + ALLO (bADX22.4, −1.19); morning winners WLD/DEXE/BICO + LAYER again (bADX26.5, +0.96). LAYER lost at low bADX and won at high bADX *same day* → "time of day / BTC ADX" and "which pair fired" are the same axis; the pair axis isolates losers without collateral damage.

**ALLO evidence (the ship):** cross-batch deduped FAN flip-shorts = 9 trades / 6 dates (J18,J20,J23,J24) / 33% WR / −4.14%; the entire ALLO loss is the flip-short sleeve (normal longs net-positive +0.32%). Under CURRENT stack the residue = 0W/3L / −3.45% / 3 dates, all full FLIP_STOP_LOSS gap-throughs (BICO failure mode). Forfeits ~zero current-config upside (CHOP winners now blocked; bull-longs disabled). REVERT GATE: un-blacklist if would-be ALLO flip-shorts run ≥50% WR on N≥5 fresh.

**LAYER NOT blacklisted:** breakeven under current stack (2W/2L / +0.16%); losses were at low BTC ADX or now-disabled BULL_LONG. Within-LAYER trend-strength pattern noted (loses low-ADX, wins high-ADX) but N=4-5 ≪ act threshold. Watchlist.

**Files:** trading_config.json (pair_blacklist +ALLOUSDT). CURRENT_STATE updated in place (blacklist line 15 pairs + ALLO ship entry + BTC-ADX refutation entry, replacing the 06-23 ALLO/PLAY watchlist). config.py default unchanged (blacklist lives in json). Not committed/pushed (no operator authorization).

---

## 2026-06-24 — BULL_LONG per-regime fan window SHIPPED (`bull_long_fan_by_regime`) + RSI-ceiling candidate REFUTED

**Action:** new config field `bull_long_fan_by_regime = "STRONG_BULL:1.35-2.0,HEALTHY_BULL:2.0-3.0"`. Per-regime fan window overrides global `bull_long_fan_min/max` when the gated regime is mapped; globals (still 0.85/5.0) remain the fallback for any allowed-but-unmapped regime. Operator-directed.

**Why:** operator asked to implement the regime-specific winning fan bands found in the cross-batch dig.

**Evidence (deduped pool, CLOSED, FROM 2026-05-04, entry_strategy=BULL_LONG, re-sim under current sleeve: regime∈{S_BULL,H_BULL} ∧ fan 0.85-5.0): N=106, 50.9% WR, −0.106% avg, −11.27% total (net losing).**
- By regime: STRONG_BULL 16/62.5%/+0.156%/+2.5% (3 dates) · HEALTHY_BULL 90/48.9%/−0.153%/−13.8% (6 dates) · CHOPPY_FLAT 18/55.6%/−0.247% (already blocked).
- Regime×fan: **HEALTHY_BULL × 0.85-1.35 = 25/28%/−12.0%** (≈ entire sleeve loss; the band the 06-23 fan_min 1.35→0.85 widening imported) · HEALTHY_BULL × 1.35-2.0 = 37/54%/−3.6 · **HEALTHY_BULL × 2.0-3.0 = 24/67%/+3.9 (6 dates)** · H_BULL 3.0-5.0 = 4/25%/−2.1 · **STRONG_BULL × 1.35-2.0 = 11/73%/+5.1 (3 dates)** · S_BULL other bands N≤2 negative.
- Indicators do NOT separate within either regime: BTC RSI flat (W/L 62.0 vs 61.9; bands 55-60/60-65/65+ all ~50%), ADXΔ & PairTrend INVERTED in HEALTHY (ADXΔ>0.5 = 20%WR/−5.5%, PairTrend>0.1 = 29%/−6.0%, BTC RSI 65+ = 38%/−3.9% — the "momentum-quality" tails are the losers). STRONG_BULL indicators point the right way but N=16 with every sub-cell N≤4 and PairTrend sign-flips vs HEALTHY = confound/noise. Conclusion: fan×regime is the only axis.
- Mechanism: BULL_LONG buys a long that already passed the fan gate in a bull regime = entering an extended 5m pop; low fan (no thrust) fizzles, high-fan parabola reverts; the workable band differs by regime (strong bull carries a modest 1.35-2.0 entry, softer healthy bull needs 2.0-3.0 pair thrust).

**Counterfactual:** restricting to the two windowed cells removes the −12.0% HEALTHY low-fan leak; the prior global counterfactual (fan 1.35-3.0 = +4.7% on N=74) corroborates. In-sample-haircut caveat: positive cells N=11/24, partly in-sample; sleeve stays at 1×-obs (lev 0.05).

**RSI-ceiling candidate (06-24 single-batch RSI≥62 = 5L/0W) — REFUTED:** does not survive cross-batch (RSI flat within both regimes). Do not ship `bull_long_rsi_max`. CURRENT_STATE watchlist marked refuted.

**Implementation (D11 full):** config.py `bull_long_fan_by_regime` default + evidence comment + NOTE on fan_min that per-regime overrides globals · trading_config.json value · engine `_maybe_open_bull_long` reordered (regime resolved BEFORE fan; per-regime window parsed `REGIME:min-max`, fallback to globals) · UI text input `config-bull-long-fan-by-regime` (templates/index.html) · load (_setVal) + save (trimmed string). Verified: both files ast-parse; gate truth-table correct (H_BULL fan1.0 blocked / 2.5 fires; S_BULL fan1.5 fires / 2.5 blocked; unmapped regime → global 0.85-5.0). Known cosmetic gap: report Bull-Long×Fan "LIVE ✓" still keys off global bull_long_fan_max (not per-regime).

**REVERT GATE:** drop a regime's entry if its windowed cell runs ≤45% WR on N≥10 fresh. Re-lever still gated at N≥30 cumulative + WR≥65%/avg≥+0.10% across ≥2 batches.

**Not committed/pushed** (no operator authorization).

---

## 2026-06-24 — BULL_LONG re-levered 1×→20× (DISCIPLINE-OVERRIDE) on the cleaned per-regime fan cell + pre-reset snapshot

**Action:** `bull_long_lev_mult` 0.05→1.0 (1×→20×), operator-directed, applied together with the new per-regime fan window (S.BULL 1.35-2.0 / H.BULL 2.0-3.0) and the ALLOUSDT blacklist. Reset taken ~16:06 UTC-3 to apply all three.

**Override acknowledgment (anti-overfit rule):** below the locked re-lever gate (N≥30 cumulative + WR≥65-70%/avg≥+0.10% across ≥2 batches). The windowed cells are N=11 (S.BULL 1.35-2.0, 73% WR, +5.1%) and N=24 (H.BULL 2.0-3.0, 67% WR, +3.9%), partly in-sample. Operator-directed; carries a TIGHTER-than-standard revert (below). Rationale for shipping despite the gate: the per-regime window excludes the loser bands every prior 20× attempt traded — the 06-22 20× re-lever tripped its instant gate the next batch but it was on the OLD wide fan (incl. the inverting 1.65-2.0 and sub-1.35 bands now excluded).

**TIGHT REVERT (override-grade, automatic):** INSTANT de-lever bull_long_lev_mult→0.05 if 3 of the first 6 new bull-longs hit SL OR any single bull-long gaps past ~−1.0%; de-lever at N≥10 fresh windowed bull-longs if WR≤60% OR avg≤+0.05%.

**Pre-reset snapshot saved:** reports/orders_2026-06-24_prereset_preregimefan_20x_39trades_-178.csv (39 closed, net −$178: 2 BULLISH FAN-flip longs, 16 BEARISH shorts, 21 NEUTRAL bull-longs) + _results.txt stub (operator to paste the text report). This captures state immediately before the reset that applies ALLO blacklist + per-regime fan + 20×.

**Files:** config.py (bull_long_lev_mult default 0.05→1.0 + override note) · trading_config.json (1.0). Not committed/pushed (no authorization).

---

## 2026-06-24 — BULL_LONG global fan bounds → union (1.35/3.0) + regime-aware LIVE marker

**Action (operator-directed, cosmetic/consistency):**
- `bull_long_fan_min` 0.85→1.35, `bull_long_fan_max` 5.0→3.0 (config.py + trading_config.json). These globals are INERT while both bull regimes are mapped in `bull_long_fan_by_regime` (the per-regime window overrides them) — set to the union of the per-regime windows purely so the UI fallback isn't contradictory (0.85/5 looked like it allowed bands the sleeve actually blocks). No behavior change: STRONG_BULL still 1.35-2.0, HEALTHY_BULL still 2.0-3.0.
- Report Bull-Long×Fan "LIVE ✓" marker made regime-aware: new `_bull_long_fan_live(lo,hi,th)` in main.py computes live = bucket overlaps the union of per-regime windows (bull_long_fan_by_regime) with global fan_min/max fallback for allowed-but-unmapped regimes; mirrors the engine gate. Replaces the old global-`fan_max`-only check in BOTH the live fan-bucket table (`fan_rows`) and the phantom Bull-Long Curve. Fixes the stale ✓ that ticked buckets up to 5.0 as live. Export header strings (both text-report functions) + UI tooltip updated to match (D12).

**Verified:** main.py ast-parses; helper unit-test → live True only for 1.35-1.65/1.65-2.0/2.0-3.0 buckets, False for 0.85-1.35 and 3.0-5.0 (= union of the two regime windows). config.py/json values updated.

**Not committed at write time** (committed/pushed in the same operator-authorized push).

---

## 2026-06-24 — LONG runner exit: SHORT-parity machinery SHIPPED + ENABLED (Option A)

**Action (operator-directed):** gave the LONG runner the same exit mechanism the shorts run — ATR-floor (chandelier) + BE-ratchet lock + give-back cap — on independent `runner_trail_*` fields, and turned it ON.

**Config:** `runner_trail_enabled` false→true · `runner_trail_atr_min` 1.0→0.0 · `runner_trail_arm_peak` 0.70→0.45 · `runner_trail_k` 0.5 (kept) · NEW `runner_trail_use_atr=true / runner_trail_atr_mult=0.5 / runner_trail_be_ratchet_enabled=true / runner_trail_be_lock_pct=0.10 / runner_trail_giveback_frac=0.0` — all mirroring the live-proven `runner_trail_short_*` values.

**Engine:** added the ATR-floor decision to the LONG branch of `check_exit_conditions` (services/indicators.py). Math is direction-agnostic P&L: once armed (peak_pnl ≥ arm), giveback = N×entry_atr_pct (capped at frac×peak if frac>0), floor = peak_pnl − giveback, ratcheted to ≥ be_lock_pct; fire RUNNER_TRAIL when pnl_pct ≤ floor. `use_atr=false` → existing K×peak-stretch fallback. SHORT path here UNCHANGED (its ATR-floor fires in the realtime strpk block; `_l_use_atr` stays False for SHORT). Once armed, tight-trailing is suppressed (existing realtime + handoff logic) so the runner owns the profit side; EMA13 / hard SL still backstop.

**Scope:** NON-FLIP longs only (BULL_LONG, MOMENTUM long, normal longs) via the existing `not is_flip` gate. **Flip-LONGs DEFERRED** (the dormant N=2 FAN⊘SHORT→LONG idle-insurance sleeve — build the flip-long trail with its source per the standing twin watchlist; not worth a hot-path change to the realtime block for 2 trades now).

**Discipline:** shipped ON ahead of the N≥30 Leash-Shadow strpk-LONG validation gate — explicit operator override. Rationale: a runner trail is net-PROTECTIVE (it banks armed peaks and the ratchet caps round-trips into losses), and the bull-longs are now at 20× where round-tripping to the hard SL is the costly failure mode. Downside of a bug is bounded (exits armed longs early, never worsens a loss past the existing SL). Could not runtime-test locally (no pydantic_settings); verified by ast-parse + logic trace + UI id counts (3× each).

**REVERT GATE:** `runner_trail_enabled=false` if armed long runners net WORSE than the `actual` long baseline on N≥10 fresh closes, OR if any armed long round-trips peak≥0.45 → ≤0 (ratchet failure).

**D11 full:** config.py (defaults + 5 new fields + evidence) · trading_config.json · services/indicators.py (engine) · templates/index.html (5 UI inputs in the LONG box + load + save). CURRENT_STATE watchlist entry converted to SHIP. Committed/pushed same turn (operator authorized).

---

## 2026-06-25 — FLIP-SHORT quality-score floor SHIPPED (`flip_short_quality_min=2.0`, blocks score ≤1)

**Action (operator-directed):** block flip-SHORT entries with entry quality score ≤1, by extending the already-shipped global Entry-Quality-Score filter (which blocks ≤1 for NORMAL entries but which flips BYPASS) to the flip-short sleeve. New config `flip_short_quality_min` (block when score < min; =2 → blocks 0,1). Counter `FLIP_SHORT_QUALITY`.

**Evidence (deduped pool, CLOSED, FROM 2026-05-04, FLIP:FAN_RATIO_GATE ∧ SHORT, current stack = drop CHOPPY_FLAT + pADX<21 + blacklist; N=66, +6.18%):**
- By quality score (monotonic): 0 = N=2/50%/−0.32% (≈empty) · 1 = N=16/56%/−2.67%/8 dates · 2 = N=25/64%/+1.75% · 3 = N=17/76%/+4.46% · 4 = N=5/80%/+2.79% · 5 = N=1.
- score≤1 cut cohort = N=18/56%WR/−2.98%/8 dates. **Per-pair concentration PASSED:** 16 pairs, top SAHARA 21% of gross loss (no 1–2 pair domination) → genuine dimensional effect, not a pair issue.
- 06-25 batch confirm: FAN flip-short 9t/−$337; score≤1 = 3/3 losers (SUI score0 −88, PAXG −16, SAHARA −145), −$249, zero winners cut → sleeve −$337→−$88. Report's own phantom LEFTOVER-FILTER TEST flags `Quality score >1: kept 71%/+0.127 vs blocked 52%/−0.045 → ★ gate candidate`.

**Discipline:** N=18 < the locked Pattern-C filter gate (N≥30/WR≤40%/avg≤−0.20%) → DISCIPLINE-OVERRIDE. Mitigation: the score≤1 threshold is NOT a novel filter — it's globally validated (N=95) and live for normal entries; this only closes the flip bypass. Concentration check passed; monotonic; mechanism sound (low score = weak setup). Score 0 immaterial (N=2) — the band doing the work is score 1.

**Implementation (D11 full):** config.py `flip_short_quality_min=2.0` + evidence · trading_config.json 2.0 · services/trading_engine.py (check in `_flip_filters` FAN-SHORT branch + `quality_score` added to `_ff_in`; block auto-recorded via existing `_record_filter_block(_reason, flip_dir)`) · templates/index.html (input `config-flip-short-quality-min` + load + save). Verified: ast-parse OK, json=2.0, UI id 3×.

**REVERT GATE:** `flip_short_quality_min`→0 if would-be-blocked (score≤1) flip-shorts run ≥55% WR on N≥10 fresh.

**Not committed/pushed at write time** (awaiting explicit operator authorization — "ship it" does not authorize git per CLAUDE.md).

---

## 2026-06-25 — DISABLE BULL_LONG sleeve + DISABLE LONG runner BE-ratchet (operator-directed)

**Context:** The 06-24 operator-directed 20× re-lever of BULL_LONG (on the cleaned per-regime fan cell) tripped its instant-revert gate on the very next batch.

**1) BULL_LONG DISABLED — `bull_long_enabled: true→false`** (config.py line 613 + trading_config.json).
- **Gate trip:** 06-25 batch ran 5 bull-longs, 1W/4L, **−$299; 4 of 5 hit the −0.70 flat SL** → exceeds "3 of first 6 hit SL" instant-de-lever trigger.
- **Deep cross-batch dig (N≈135):** sleeve is structurally **arm-or-die** — windowed losers avg peak ≈+0.08%, 13/15 never reach the +0.20% arm band → straight to SL; winners avg peak ≈+1.07%. **No entry separator survives cross-batch:** BTC RSI / BTC ADX / ADXΔ / pTrend all refuted or non-monotonic. Range-position is the only mild tendency (low-range wins ~82%, top-of-range 85+ loses ~39%/−10.5%) but had **ZERO impact on 06-25** (all 5 were mid-range 54–75).
- **Verdict:** windowed cell ≈+0.16%/trade at 1× — too thin and too SL-variance-heavy (≈38% never-arm SL rate) for ANY leverage, and unfixable by entry filters. Chose full DISABLE over de-lever-to-1× (cleaner; 1× edge not worth the slot/complexity). Per-regime fan config + flat −0.70 SL + cap 3 retained but inert. Long-side scaling deferred to the parked Donchian module. **RE-ENABLE only with a genuinely NEW edge (an arm-predictor) — not another leverage attempt.**

**2) LONG runner BE-ratchet DISABLED — `runner_trail_be_ratchet_enabled: true→false`** (config.py line 357 + trading_config.json).
- Armed LONG runners now exit on the bare ATR-floor (peak − 0.5×ATR), no +be_lock_pct clamp; hard SL still backstops. **SHORT ratchet `runner_trail_short_be_ratchet_enabled` left ON (proven).** `runner_trail_enabled` itself stays ON; only the LONG ratchet clamp is removed.

**Not committed/pushed at write time** (awaiting explicit operator authorization — bundled with the still-staged flip-short quality floor). trading_config.json included in the push set per standing rule.

---

## 2026-06-25 — WATCHLIST add: FAN flip-SHORT quality-score≥3 multiplier candidate (sleeve-scoped)

**Trigger:** while reviewing the last-5-batch (06-20→06-25) current-stack survivor set (N=67/65.7%WR/+$1,725), qs=3 looked like a standout cell (+0.307%/74%WR). Operator asked whether the qs edge is in MOMENTUM, FAN flip, or both before watchlisting it as a multiplier.

**Finding — it's a FAN-flip edge, NOT momentum.** Last-5-batch deduped, by sleeve × quality score:
- **FAN flip (monotonic):** qs1 26/54%/−0.197% · qs2 37/51%/−0.073% · qs3 24/67%/+0.118% · qs4 10/80%/+0.247%. **qs≥3 = N=35/71.4%/+0.156%/+$524** · qs≤2 = N=69/53.6%/−0.101%/−$939. Flip-SHORT subset of qs≥3 = N=33/72.7%/+0.200%/+$627 (flip-LONG only N=2).
- **MOMENTUM (no qs edge):** qs2 7/71%/+0.254% (its BEST) · qs3 7/57%/+0.175% · qs4 4/50%/−0.023%. Non-monotonic, qs≥3 (+0.076%) < qs≤2 (+0.254%). Quality score is NOT a momentum separator.

**Verdict:** scope the multiplier candidate to **FAN flip-SHORT qs≥3 only**. In-sample it already clears the locked Pattern-W gate (N=35≥30, WR 71.4%≥70%, avg +0.156%≥+0.10%, Total$>0) — but it's the SAME data `flip_short_quality_min=2` was fit on; a 30–50% haircut drops +0.156% toward/under +0.10%, so NOT shipped. Complements the floor (block ≤1 / multiply ≥3 = caps-for-losers + multipliers-for-winners on one dial).

**PROMOTE GATE:** N≥30 FORWARD (out-of-sample) FAN flip-short qs≥3 holding WR≥70% AND avg≥+0.10% → ship 1.5× (staging), step 2× after +50; BE-compat check before 2×. Watchlist line added to CURRENT_STATE. No config/code change. Cross-check overlap vs the EMA20-slope 0.20-0.30 candidate.

---

## 2026-06-26 — WATCHLIST add (×2, FAN flip-SHORT): QS=2 doubt-zone + H.BULL no-data capture

Both scoped to FAN flip-gate trades (entry_strategy=FLIP:FAN_RATIO_GATE), survivors of the live qs≥1 floor.

**1) QS=2 doubt-zone (possible floor raise 1→2... actually →3).** Distribution of QS≥2 FAN flips:
- Last-5-batch: qs2 N=37/51.4%/−0.073%/−$377 · qs3 N=24/66.7%/+0.118%/+$223 · qs4 N=10/80%/+0.247%/+$280 (clean monotonic; edge only qs≥3).
- Full-pool (all snapshots): qs2 N=99/53.5%/−0.112%/−$1,424 · qs3 N=57/61.4%/−0.030%/−$229 · qs4 N=16/68.8%/+0.367%/+$617. qs=2 loss DIFFUSE (44 pairs, top-8=57%, worst 11%) → cohort weakness, not blacklist.
- **Not blocked:** qs=2 wins >50% ("high-WR net-losing" anti-pattern) and FAILS the locked Pattern-C gate (WR≤40% & avg≤−0.20%). It's ~58% of the surviving sleeve; qs=3 is only ~breakeven so floor→3 removes a near-neutral band, real edge is qs≥4 (small N). PROMOTE: raise flip_short_quality_min→3 only if qs=2 holds ≤45% WR AND avg≤−0.10% on N≥30 forward across ≥2 batches.

**2) H.BULL FAN flip-short — no data.** Last 5 batches = ZERO H.BULL/S.BULL FAN flip-shorts (recent fires all bear/chop). Full-history H.BULL only N=5 (old). Recent regime edge: HEALTHY_BEAR N=35/63%/+0.132%/+$433 (engine), STRONG_BEAR N=24/67%/+0.088%, BEAR_EXHAUSTED N=6/33%/−0.433% (small loser). Next-batch: capture the FLIP_RATIO_GATE SHORT × H.BULL/S.BULL cells (first live bull fills); ties to the S.BULL-unblock-on-phantom watchlist.

**3) 4-cohort note (not a watchlist, recorded):** last-5 QS≥2 FAN flip-SHORT — W-only N=17/88%/+0.467%/+$792 (star), C-only N=28/50%/−$376 + Both-C&W N=16/50%/−$295 (the losers), TRULY UNMATCHED N=9/67%/+0.112%/+$129 (positive → NOT a block, contra the full-pool −$124 read). If any cohort is suspect it's the C-pattern-matched fades, not unmatched.

Docs only; no config/code change.

---

## 2026-06-26 — REFUTED + removed: BTC_RSI_ADX_CROSS "fade BTC overbought" SHORT phantom (Phantom Flip Tracker)

**Trigger:** Operator flagged the lone ★ in the Phantom Flip Tracker — `BTC_RSI_ADX_CROSS` fade-to-SHORT: phantom N=28 / 71% WR / +0.406% avg / +11.37% total / SL 25% ("★ flip pays"), concentrated in **BTC RSI 70-75 × ADX 30-35** (N=17 / 65% WR / +0.607% / +10.32%). Mechanism = a LONG blocked because BTC is overbought (RSI 70-75) → fade into a SHORT (= short the BTC top in a bull).

**Cross-batch check (complete pool N=1799, 896 real shorts w/ btc_rsi) — REFUTED:**
- **BTC RSI 70-75: ZERO real short fills** (and 75+: zero). The phantom's ★ cell has no realized counterpart — the bot never shorts there (BTC-RSI>60 shorts are gated out). Pure virtual-fill artifact.
- Nearest real band **BTC RSI 65-70 = N=75 / 40% WR / −0.310% avg / −$1,466** — the worst short band in the pool. 55-60 = 19/16%/−0.703%. Shorting into elevated BTC RSI fights the trend and loses.
- This is the BTC-level twin of PAIR_RSI_OB (pair overbought-fade, killed 06-26: phantom 76-80% WR vs live 39%). Same overbought-fade family, same phantom-overstatement mechanism (virtual entry at clean price + flat exit on a trade the live bot never takes).

**Pattern (3rd this session):** phantom signals that looked great and died on cross-check — bull-long (88% phantom / 47% live), PAIR_RSI_OB (76-80% / 39%), now BTC_RSI_ADX_CROSS short (★ phantom / 0 real fills, adjacent band −$1,466). Treat any Phantom-Flip-Tracker ★ as guilty-until-proven; the tracker systematically overstates because the live entry is gated out or fills worse. Whole tracker pooled = N=900/52%WR/−0.028%/−24.88% = fade-the-block is a non-edge.

**Action:** removed `BTC_RSI_ADX_CROSS` (both dirs) from the Phantom Flip Tracker `source_specs` (main.py). Its LONG fade (✗ whipsaws) was the dead bounce-long thesis; its SHORT fade is the refuted ★. Remaining tracker sources: PAIR_ADX_MAX, BTC_ADX_BLOCK_SHORT, PAIR_RSI_ADX_CROSS, PAIR_TREND_FILTER. **DO NOT re-investigate a BTC-overbought-fade short.** Docs + 1-line source removal only; no engine/config change.

---

## 2026-06-26 — WATCHLIST: FAN flip-SHORT dimensional separators (range<50 / BEAR_EXHAUSTED blocks + qs≥3×bear≥70×range60-90 winner cell)

Dimensional dig of the FAN flip QS≥2 cohort (last-5-batch deduped, N=72) to test whether qs=2 should be blocked OR a sub-dimension is the real separator.

**Dimensional tables (all QS≥2):**
- Breadth (bear%): <60 16/62%/+0.025/−$46 · 60-70 24/54%/−0.164/−$442 · 70-80 14/64%/−0.004/+$52 · 80+ 18/67%/+0.353/+$583.
- Range position: <50 8/38%/−0.443/−$388 · 50-60 5/60%/−0.124/−$68 · 60-90 51/65%/+0.149/+$692 · 90-100 8/62%/−0.084/−$88.
- BTC regime: HEALTHY_BEAR 35/63%/+0.132/+$433 · STRONG_BEAR 24/67%/+0.088/+$169 · BEAR_EXHAUSTED 6/33%/−0.433/−$307 · HEALTHY_BULL 2/50%/−0.556/−$103.

**KEY confound test:** within the favorable bucket (bear%≥70 AND range 60-90, N=26): qs=2 = 12/42%/−0.110% (still loses) vs qs=3 = 7/71%/+0.430% vs qs=4 = 6/100%/+0.548%. → **QS and the dimensions are INDEPENDENT separators; they stack.** qs=2 is genuinely weak (not a confound) but DIFFUSE/weak as a blanket cut.

**Candidate comparison vs Pattern-C gate (WR≤40% AND avg≤−0.20%):**
- range<50: N=8/38%/−0.443/−$388 — CLEARS gate. Mechanism: short a pair already in lower-half of range = no downside left to fade.
- BEAR_EXHAUSTED: N=6/33%/−0.433/−$307 — CLEARS gate. Mechanism: fade into exhausting bear → bounce overruns the short.
- qs=2 blanket: N=37/51%/−0.073/−$377 — FAILS gate (>50% WR, thin avg, contains winners e.g. qs=2 at 80%+ breadth +$203).
- Breadth 60-70: 24/54%/−0.164 — FAILS (>40% WR); and breadth is non-monotonic across qs=2 (confound).
- WINNER cell (multiplier track): qs≥3 AND bear≥70 AND range 60-90 = N=14/~79%/+0.49%/+$671.

**qs=2-specific carve REJECTED:** within the 37 qs=2 trades, breadth is non-monotonic (U-shape: loss in 60-80 mid-band, +ve flanks = confound); range 50-70 is the loss slice (−$479) but W-vs-L entry means are near-identical (bear 66.0/65.0, rangePos 77.9/73.3, RSI 58.9/57.3 — feature-inseparable); loss diffuse across 23 pairs (top-2=27%, not blacklist). So qs=2 winners can't be reliably picked at entry.

**⚠ ALL N's ABOVE ARE IN-SAMPLE-DERIVED → FORWARD-N=0 for the gate.** range<50 (N=8) and BEAR_EXHAUSTED (N=6) are tiny + in-sample = pure hypotheses; need N≥30 fresh out-of-sample holding the C-gate before any ship. Rare sub-conditions → accumulate slowly (many batches).

**Action:** added a consolidated watchlist line (range<50 block / BEAR_EXHAUSTED block / qs≥3×bear≥70×range60-90 winner cell) to CURRENT_STATE; annotated the qs=2 line as the WEAKEST candidate (dimensions-first). No config/code change.

---

## 2026-06-26 — UPDATE: range<50 vs BEAR_EXHAUSTED overlap (FAN flip-SHORT block candidates)

Checked overlap of the two block candidates on the QS≥2 N=72 cohort. **They overlap heavily — largely the SAME loser slice (low-range fade in an exhausted bear):**
- range<50 ∩ BEAR_EXHAUSTED = 4 trades (25%WR/−0.372%/−$193).
- UNION (range<50 OR BEX) = 10 unique (8+6−4), 40%WR/−0.465%/−$502.
- range<50's regime mix: 4 BEAR_EXHAUSTED, 2 H.BULL, 1 STRONG_BEAR, 1 H.BEAR. BEX's range mix: 4 of 6 are <50.

**Block-impact on N=72 (baseline 61.1%WR/+0.038%/+$147.5):**
- block range<50 → 64t/64.1%/+0.099%/+$535.8 (Δ +$388)
- block BEAR_EXHAUSTED → 66t/63.6%/+0.081%/+$455.0 (Δ +$307)
- block UNION → 62t/64.5%/+0.120%/+$649.9 (Δ +$502, NOT +$695 — the 4 shared trades mean you can't sum them)

**Conclusion:** range<50 is the broader, mechanism-fundamental cut (captures 4 of 6 BEX losers; +$388 of the +$502 combined uplift). BEAR_EXHAUSTED adds only ~$114 unique. → reframed CURRENT_STATE: range<50 = PRIMARY filter, BEX = small non-overlapping residual; concentrate forward-N accrual on range<50; re-test whether BEX adds unique value before adding it separately. Still in-sample / FORWARD-N=0. No config/code change.

---

## 2026-06-26 — FAN flip-short: dimensional blocks REFUTED out-of-sample; winner-cell multiplier SURVIVES; qs=2 NOT a clean loser; + pool-contamination lesson

**Context:** continued the 06-26 dimensional dig. First fresh out-of-sample batch arrived (06-26 14:28 / 15:02 snapshots) + operator pushed twice on numbers that didn't match the earlier screenshot → forced a full reconciliation that exposed two pool-construction errors of mine. No config/code change. CURRENT_STATE lines 84 + 86 edited in place; this is the archival record.

**Saved canonical reference:** `reports/flip5batch_curated_jun20-25_FANflipshort_72pool.csv` — deduped union of the 5 curated reports/ batch files (`orders_2026-06-25_1839...`, `orders_2026-06-24_prereset...`, `orders_2026-06-23_prereset...`, `batch_2026-06-22_2028...`, `flip_ref_2026-06-20_presreset...`), key=(opened_at,pair,direction), CLOSED, FLIP:FAN_RATIO_GATE → N=104 flip-shorts, qs≥2 subset = the **72-pool**. This file matches the original line-86 figures and the screenshot exactly; use it, not ad-hoc globs.

**Canonical 72-pool numbers (the source of truth):** qs=2 N=37/51.4%WR/−0.073%/−$377 (mild doubt zone, FAILS C-gate) · qs≥3 N=35/71.4%/+0.156%/+$524 (positive) · winner cell (qs≥3×bear≥70×range60-90) N=14/85.7%WR/+0.463%/+$671 · range<50 N=8/37.5%/−$388 · BEAR_EXHAUSTED N=6/33.3%/−$307 · full pool ≈breakeven (+$148/72).

**① Dimensional BLOCKS refuted out-of-sample.** range<50 + BEAR_EXHAUSTED were clean C-gate losers IN-SAMPLE (−$388 / −$307) but the 06-26 batch (FAN flip-shorts N=7, 3W/4L, −$196) refuted both: the 3 real losers (IDU/LTC/TAO) were all qs=2, range 70–80, mostly HEALTHY_BULL — none range<50, none BEAR_EXHAUSTED. Blocking range<50 = removes +$61.5 net (kills AAVE +$78 winner); blocking BEX = removes INJ +$36 winner; both = removes +$97.6 of winners. Same overfit failure mode as bull-long / PAIR_RSI_OB / BTC_RSI_ADX_CROSS phantoms breaking on first contact (tiny in-sample N=8/N=6). VERDICT: ship NEITHER block; kept as a closed negative so the dead leads aren't re-derived.

**② Winner-cell multiplier SURVIVES.** Canonical 72-pool N=14/85.7%/+0.463%/+$671 — clears the W→MULT gate on WR, avg, $; fails only N=14<30. Out-of-sample 06-26 fired once = INJ +$36 (1-for-1 positive). Multiplier impact: 1.5× pool Δ +$336, 2× pool Δ +$671. ⚠ BE-compat FAILS for 2× (0/2 cell losers peaked ≥+0.20% — gate needs ≥60%) → a 2× amplifies gap-to-SL tails; would ship 1.5× first per Phase-3 staging regardless. Remains a LIVE multiplier candidate pending FORWARD-N (cadence thin, ~1-3 fires/batch).

**③ qs=2 floor→3 — NOT a clean in-sample loser.** On the canonical pool qs=2 is the original mild doubt zone (−$377, 51.4% WR), FAILS the C-gate (wins >50%). It led only on the small 06-26 batch (0/3, −$306, caught all losers). NET: in-sample favors the dimensions over qs=2; first out-of-sample favors qs=2; both low-N. Stays a floor-raise WATCH; PROMOTE only if qs=2 FAN flip-shorts hold ≤45% WR AND avg≤−0.10% on N≥30 FORWARD across ≥2 batches.

**④ POOL-CONTAMINATION LESSON (the reason for the back-and-forth).** I twice produced wrong numbers the operator caught by comparing to the screenshot:
- (a) Scored `reports/flip_ref_2026-06-16_76trades.csv` raw → PRE-filter (gap≥1.0, FAN_LOATR shipped AFTER 06-16) → over-counts losers; falsely showed "all qs buckets lost." RETRACTED. Violated the locked rule "re-simulate under the CURRENT filter stack before any counterfactual."
- (b) Globbed ALL Downloads `scalpars_orders_paper_*.csv` with opened_at≥06-20 → pulled the 06-20 PRE-RESET disaster batch the curated files excluded; falsely showed qs=2=−$734/qs≥3≈breakeven/cell=+$393. RETRACTED.
- FIX: canonical FAN-flip pool = the 5 curated reports/ files only (frozen csv above); recorded to memory (`reference_fan_flip_72pool.md`) so it can't recur. Always reconcile FAN-flip stats against the curated csv, not ad-hoc globs.

**No config/code/git change. Watchlist (CURRENT_STATE) lines 84 + 86 corrected in place with retraction notes.**

---

## 2026-06-27 — SHIP: FAN flip-LONG DISABLED (flip_long_enabled=false; discipline-override)

**Change:** new `flip_long_enabled: bool` (config.py default True / trading_config.json false). Engine: hard gate at the top of the flip-LONG path in `_flip_filters` — `if flip_dir=='LONG' and not flip_long_enabled → (True, "FLIP_LONG_DISABLED", …)`, sits ABOVE the regime block (supersedes it). Counter FLIP_LONG_DISABLED via `_record_filter_block`. D11 full (config + json + engine + UI cyan toggle `config-flip-long-enabled` + load/save). So a blocked SHORT no longer fades to LONG; only blocked LONGs fade to SHORT.

**Evidence:** FAN flip-LONG is a rare, structurally net-negative micro-sleeve — full-history deduped N=8, 5W/3L (62% WR) but **net −$297** (1:8 R:R: wins +$8..27, the 3 losers VELVET −$141 / HYPE −$120 / DYDX −$115 gap to the −0.7/−1.2 SL at 20×). Mechanism: a flip-LONG goes long into a pair that was about to be SHORTed (oversold / bottom-of-range / falling) = countertrend long → SL. The live `FLIP_LONG_REGIME` block already removes the H.BEAR/CHOP losers (VELVET/HYPE), so the residual is **H.BULL countertrend-longs** which that block can't catch. Forward H.BULL flip-LONGs: **DYDX −$115 (06-24) + XPL −$164 (06-27) = 0/2, −$279**. XPL (06-27): RSI 32.3, range position 6% (rock-bottom), ADXΔ +2.21 — long into a crash, straight to −1.10% SL; it was the single biggest loser of the 06-27 batch.

**⚠ DISCIPLINE-OVERRIDE acknowledged:** N=2 fresh H.BULL losers is below the watchlist's own N≥10-fresh gate. Shipped anyway — operator-directed, clean & consistent mechanism (same arm-or-die / countertrend-long signature, 0/2 fresh + 0/3 full-history H.BEAR/CHOP already handled), and the flip-LONG fires only ~1/batch so N≥10 would take many weeks. Carries a TIGHTER-than-standard revert.

**TIGHT REVERT:** set `flip_long_enabled`→true if blocked flip-LONGs would have been **≥55% WR AND net-positive on N≥8 fresh** (the phantom/passthrough seed still observes the blocked flip-LONG side, so the counterfactual keeps accruing). Also revert if the disable removes a cross-batch-stable winning flip-LONG cell that emerges.

**No risk to the long edge:** flip-LONGs are NOT the momentum/unmatched longs (those are the +EV engine, untouched). This only kills the FAN-flip-SHORT→LONG fade. CURRENT_STATE watchlist line flipped 🔭→✅.

---

## 2026-06-27 — REDUNDANCY CHECK: winner-cell 2× BE-compat retracted + qs floor→3 case weakened (both pre-filter-contaminated)

**Trigger:** operator asked whether the candidates' losers are already caught by other live filters. Re-scored both against the current stack (pADX≥21 / gap≥1.0 / FAN_LOATR / stretch / regime / hi-ATR). Result: both candidates' raw pool numbers are dominated by PRE-FILTER ghost trades that no longer fire. No config/code change — analysis only; CURRENT_STATE qs=2 + winner-cell lines annotated.

**① Winner cell (qs≥3×bear≥70×range60-90) — BE-compat objection RETRACTED.** The 2 cell losers (HUSDT −$150 / HMSTR −$100) that "gapped straight to SL, peak 0.00/0.018" both have pADX < 21 (17.4 / 15.4) and are dated 06-19 & 06-22 — BEFORE the pADX≥21 floor shipped (~06-23). Under the current stack they're blocked → don't fire. Re-scoped (pADX≥21): cell = **N=6, 6W/0L, 100%WR, +0.686%, +$433 — zero losers.** So the prior "BE-compat FAILS (0/2 armed) → don't 2×" is retracted: there are no un-armed gap losers left to amplify, and the tracked 1× cell only captures pADX≥21 members forward (auto-clean). REMAINING 2× blocker is purely forward-N (6 in-sample + 1 forward INJ vs N≥30) AND the cell overlaps the pADX floor (floor removed 6 of the 14 winners too). Verdict unchanged (no 2× now) but the REASON shifts from tail-risk to sample-size; when N≥30 forward accrues, stage 1.5×→2× normally.

**② qs floor→3 (block qs≤2) — case WEAKENED, prior "cleaner override" lean RETRACTED.** Of the 37 in-sample qs=2 flip-shorts, **20 are already blocked by a live filter** (pADX<21 ×16, gap ×5, LOATR ×5) and carry the entire −$631 of loss. The **17 that pass the live stack** (what floor→3 would net-new block) = **11W/6L, 65%WR, +$255 — POSITIVE.** So in-sample, raising the floor removes a net-positive cohort (forfeits +$925 winners to kill −$670 losers). Forward (06-26+06-27, N=6, all pass) = 2W/4L, −$366. Current-stack qs=2 = in-sample **+$255** vs forward **−$366** = contradictory, combined ~−$111/N=23 ≈ breakeven. The strong −$377/−$743 case was pre-filter contamination. ⇒ NOT shippable; keep track-only; N≥30-forward gate stands. (Earlier this session I leaned "ship floor→3 as a clean override" — RETRACTED; it would block a net-positive in-sample cohort.)

**Lesson (3rd time this session):** always score a candidate on CURRENT-STACK survivors, not the raw pool — the pADX-21/gap-1.0/LOATR floors already do most of the loser-removal, so marginal cohorts are far better (winner cell) or far less bad (qs=2) than the raw deduped pool shows. Same class as the 06-16-pre-filter and 06-20-glob retractions.

---

## 2026-06-27 — SHIP: winner-cell multiplier 2× SIZE (flip_fan_qs_cell 1.0→2.0; operator override)

**Change:** `flip_fan_qs_cell` "3:70:60-90:1.0:1.0" → **"3:70:60-90:2.0:1.0"** (trading_config.json; config.py code-default stays inert 1.0:1.0). So a FAN flip-SHORT in the winner cell (qs≥3 AND bear%≥70 AND range 60-90) now sizes at **2× INVESTMENT / 1× leverage**. Engine path = `_fan_qs_cell_match` → cell tag → open_position cell_src `FLIP:FAN_RATIO_GATE[QS≥3×BEAR≥70×RNG60-90]×2`. 2× is exactly at the inv hard cap (2.0), so not clamped.

**Why 2× SIZE not 2× LEV:** doubling investment doubles notional via more margin at the SAME 20× per-position leverage → liquidation distance unchanged (~−4.7%). Doubling LEVERAGE → 40× → halves the liquidation distance (~−2.35%), directly worsening the gap-to-SL tail. The lev field was deliberately left at 1.0.

**Evidence (current-stack):** the winner cell screened to live survivors (pADX≥21) = N=6, 6W/0L, 100%WR, +0.69%, +$433 — **zero losers**. The prior "BE-compat FAILS (0/2 armed) → don't 2×" was RETRACTED 2026-06-27: the 2 gap-losers (HUSDT/HMSTR, peak 0.00/0.018) are PRE-pADX-floor ghosts (dated 06-19/06-22, blocked by the live pADX≥21 floor) → no un-armed gap losers left to amplify. Raw in-sample cell = N=14/86%/+$671; out-of-sample 06-26 = INJ +$36.

**⚠ DISCIPLINE-OVERRIDE acknowledged:** forward-N=1 (one out-of-sample fire) is far below the locked W→MULT gate (N≥30 forward + WR≥70% + avg≥+0.10%), and this skips the mandated 1.5×-first Phase-3 staging (jumped straight to 2×). My analyst recommendation was to WAIT for forward-N; operator chose to ship now on the BE-clean current-stack cell. Justified by: cell is gap-loser-free under the live stack, 2× is SIZE-only (no leverage/liquidation change), it's at the hard cap, and it carries a tighter-than-standard revert.

**TIGHT REVERT (override-grade):** set `flip_fan_qs_cell` size→1.0 if the 2× cell runs **WR<70% OR avg≤+0.05% OR Total$ negative on N≥5 fresh fires**, OR **INSTANT** revert if any single 2× cell trade gaps the SL past ~−1.0%. Track its row in 💰 Multiplier Cell Performance.

**Scope:** only the qs≥3×bear≥70×range60-90 FAN flip-SHORT cell is 2×; all other flip-shorts stay 1×; flip-LONGs are disabled; momentum/unmatched longs untouched. CURRENT_STATE winner-cell line flipped to ✅ SHIPPED 2×.

---

## 2026-06-28 — SHIP ×2: C1 SHORT 2× breadth-scope de-mux + universal flip-short ADXΔ<−0.5 block (operator-directed; from the 06-28 8-11pm window analysis)

**Trigger:** the 06-28 batch lost −$779 in a ~3h Sat-evening window — 6 shorts across 3 sleeves opened into one BTC down-leg that bounced and ran them to SL together (all DOA, peak ≈0). Decomposed the window; two addressable levers + one concentration watch.

**(1) C1 SHORT 2× breadth-scope de-mux.** New fields `c1_short_demux_breadth_enabled=true`, `c1_short_demux_breadth_lo=70.0`, `c1_short_demux_breadth_hi=85.0` (config.py + trading_config.json). Engine: in `open_position`, immediately after `_lookup_pattern_cell_rule`, if direction==SHORT AND src contains 'C1' AND inv>1× AND enabled, de-mux inv/lev→1× when `entry_bear_pct` is OUTSIDE [lo,hi) (log `C1_DEMUX_BREADTH`). Sizing only — entry NOT blocked. UI panel under the winner-cell multiplier + load/save (D11). **Evidence:** cross-pool (reports/dedupe_pool_13batches_may26-jun13 + all-history glob, deduped) C1 by breadth — 70-85 = 73-76%WR/+0.05..0.10%/+$ (sweet spot, N≈26-43); both tails 50-60%WR/−avg% where the 2× merely doubles fat-tail DOA losers. The ≥85 *entry-block* was REJECTED (50%WR ⇒ fails the locked Pattern-C FILTER gate WR≤40%; would forfeit 9 winners for +$116 — a sizing artifact, not an edge). De-mux recovers nearly the same (~+$112) without killing winners — the rule-aligned tool (caps/multipliers, never block a >50%-WR cohort). 06-28 effect: AAVE(bear87)/HYPE(bear62) −$242/−$186 → −$121/−$93 = +$214. Scoped to C1 ONLY (W2+W1 and the UNMATCHED-LONG 2× are NOT touched — verified that a naive "de-mux all 2× outside 70-85" would crater the long engine −$586 because longs fire at low breadth). REVERT: disable if C1 70-85 drops <70%WR OR the ≥85 band turns net-positive on N≥10 fresh.

**(2) Universal flip-short ADXΔ<−0.5 block.** New field `flip_adx_delta_min` (config.py default −99.0=OFF; json −0.5 LIVE). Engine: `_flip_filters` SHORT branch, block when adx_delta < min (sentinel −99=off, strict <), counter `FLIP_SHORT_ADXD` (auto-recorded). UI under the regime-scoped ADXΔ row + load/save (D11). This is the standing #1 flip-short re-eval candidate (was DOWNGRADED Jun 17, never refuted), now shipped as a UNIVERSAL (all-regime) cut — distinct from the existing regime-scoped `flip_short_regime_block_adxd_max` (bull/chop @ <0, which does NOT cover bear and so missed BEL). Mechanism: flip-shorts bypass the momentum `Pair ADX Dir: rising` filter, so they can fade a pair whose ADX is collapsing (the faded trend dying → no follow-through → never arms → 20× gaps the SL). 06-28 BEL (STRONG_BEAR, ADXΔ −1.02) −$195: the cut blocks BEL only, 0 winners removed, +$195 (operator chose universal field over extending the regime gate, to avoid loosening the shipped bull/chop <0 cut). ⚠ DISCIPLINE-OVERRIDE forward-N=1. REVERT: → −99 if would-be-blocked flip-shorts hit ≥50% WR on N≥8 fresh.

**Combined 06-28 batch effect: +$409 of the −$779 window (de-mux +$214, ADXΔ +$195).** The residual −$298 (3 non-C1 momentum DOA shorts at breadth 92-95) is NOT addressable per-trade — breadth≥85 as an *entry* signal was proven cross-batch NOISE (≥85 flips sign pool-to-pool; raw momentum-short sleeve ~breakeven across all breadth). Logged a CONCURRENT-SHORT CONCENTRATION watch (cap simultaneous shorts) as the real root-cause lever — track-only. Also broadened the DOA watch: DOA (never-armed, peak≈0) is now the dominant SHORT-loss mode cross-cell, not just unmatched-LONG. Both ships pending push.

---

## 2026-06-28 — REVERT: winner-cell 2× → 1× (flip_fan_qs_cell 2.0→1.0; INSTANT revert-gate FIRED)

**Change:** `flip_fan_qs_cell` "3:70:60-90:2.0:1.0" → **"3:70:60-90:1.0:1.0"** (trading_config.json; config.py default was already inert). Cell back to INERT/track-only.

**Trigger — the pre-committed INSTANT revert fired.** The 06-27 ship locked: *"INSTANT revert if any single 2× cell trade gaps the SL past ~−1.0%."* On the 06-28 batch, **WIF flip-short (winner-cell 2×, src `FLIP:FAN_RATIO_GATE[QS≥3×BEAR≥70×RNG60-90]×2`) closed −1.159%** — gapped the −0.70 SL past −1.0% → **−$321 at 2×** (would have been −$160 at 1×). Peak 0.17% (never armed). It fired in the 07:30-11am UTC-3 **chop-whipsaw**: BTC trendless 59.9-60.4k all morning, 6 of 7 window trades DOA (peak ≈0), both longs and shorts run to SL. Per "pre-committed gates do not move," this is non-discretionary; operator confirmed.

**Forward record at 2×: N=2 = INJ +$36 (06-27) / WIF −$321 (06-28) = −$285.** The 2× was a forward-N=1 discipline-override from the start; two forward fires in, it's net-negative AND tripped the tail-gap guard. Cell reverts to 1× (track-only) and re-qualifies only via the normal W→MULT path (N≥30 forward, WR≥70%, avg≥+0.10%, then 1.5×-first staging).

**Scope:** only the winner-cell sizing; the C1 de-mux and universal ADXΔ block (shipped 06-28 same day) are untouched. Same-batch context: unmatched-LONG 2× took 2 DOA stop-losses (JTO −257 / PENGU −179, both peak 0.000) — the DOA watch ticks up (~3/5) but the long 2× stays (still the engine, +$1,256 batch). Logged the WIF trigger to CURRENT_STATE; cell line flipped SHIPPED→REVERTED.

---

## 2026-06-28 — SHIP: Momentum-short dead-tape block (`momentum_short_btc_atr_min = 0.12`) [DISCIPLINE-OVERRIDE, N=5]

**Decision:** New entry filter `momentum_short_btc_atr_min` (default 0.12, live 0.12). Blocks MOMENTUM-SHORT entries when BTC ATR% < 0.12. Counter `MOMENTUM_SHORT_LOATR`. Operator-directed ("makes theoretical sense + proven in this batch, ship it").

**Origin — the morning −$1,075 (06-28 07:30-11am UTC-3) investigation.** Operator pushed hard for an analytical explanation of a 6-hour −$1,057 window. Decomposition: −$500 flip-shorts / −$435 momentum-LONGs (JTO −257, PENGU −179, both DOA peak 0.000) / −$139 momentum-shorts. Key reframes en route:
- The momentum-LONG losses have NO entry fix — every cohort they fall in (BTC-ADX-rising +$854, HEALTHY_BULL +$1,115, low-BTC-ATR longs +$736/83%WR) is net-POSITIVE. The long engine is +EV; forfeiting their cohort costs more than it saves. ~$435 of the morning is unfilterable DOA.
- "Low BTC ATR hurts both directions" was REFUTED on the sleeve split: L MOMENTUM <0.12 = 83%WR/+$736 (best cohort); the damage is SHORT-specific.
- The broad "block all shorts when BTC quiet (ATR<0.12 OR 5m-rise≥3)" guard was REJECTED: 47% WR coin-flip, the 5m-rise arm blocks WINNING flip-shorts (+$252), the flip 0.10→0.12 extension forfeits +$340 of winners. Not shippable.

**What survived — momentum-short × BTC ATR<0.12 ONLY.** Across ALL 26 dedup momentum-shorts (5 curated reports/ batches + Downloads 06-26/27/28): the ATR<0.12 band = **N=5 / 0%WR / 100%DOA / −$638** (AAVE −242, PUMP −115, ONDO −106, NEAR −98, HYPE −77; spanning 06-27 22:12 → 06-28 10:46 dead-BTC pockets, not one batch). **ZERO winners ever killed** — every momentum-short win had ATR ≥ 0.132. On the 5 curated current-filter batches it fires 0× (no momentum-short dipped <0.132; lowest 0.132) = neutral, untested in normal vol — it only acts in genuinely dead tapes where momentum-shorts are proven 0% WR.

**Why a new field, not a tweak.** The two existing BTC-ATR floors miss this band: `flip_fan_btc_atr_min`=0.10 is FLIP-only; `btc_atr_btc_adx_filter_short` (<0.10) also requires BTC ADX≥30. Momentum-shorts at ATR 0.10-0.12 with normal ADX slip past both. The flip floor was deliberately NOT extended to 0.12 — flip-shorts WIN in the 0.10-0.12 band (14W/8L, +$340 on the curated pool; they fade a popping/quiet BTC). MOMENTUM-only.

**Discipline note.** N=5 << the N≥30 Pattern-C→FILTER gate, so this is a DISCIPLINE-OVERRIDE. The override is justified by STRUCTURE, not hope: the failure mode operator cares about (killing winners) is zero by construction — the ATR<0.12 band contains only DOA shorts (BTC too quiet to fall). Worst case forfeits a momentum-short in a dead tape where they win ~0%.

**D11 wiring (all 6, grep-verified, py compiles, json 0.12):** ① config.py field + evidence comment ② trading_config.json 0.12 ③ engine block in normal-entry SHORT chain after `BTC_ADX_BLOCK_SHORT` (fail-open on missing btc_atr_pct) ④ UI input `config-momentum-short-btc-atr-min` under the BTC-ATR×BTC-ADX short rules ⑤ load (`_setVal`) + save (`parseFloat … || 0`) ⑥ `_record_filter_block("MOMENTUM_SHORT_LOATR", …)` counter.

**This-batch proof (06-28 15:08):** momentum-short ATR<0.12 = 5/5 losers, −$638, 100% DOA, 0 winners forfeited (the whole momentum-short loss of the batch).

**TIGHT REVERT (override-grade):** set `momentum_short_btc_atr_min=0` if ANY momentum-short in the ATR<0.12 band closes ≥+0.30% (first time it would kill a winner), OR if blocked-zone WR ≥40% on N≥10 fresh.

---

## 2026-06-28 — SHIP: Momentum-short weak-capitulation block (`momentum_short_weakcap_*`) [DISCIPLINE-OVERRIDE, N=4]

**Decision:** New entry filter — block momentum SHORT when ALL three hold: `range_position < 15` AND `pair ATR% < 0.45` AND `pair ADX < 28`. Config `momentum_short_weakcap_enabled=true / range_max=15 / atr_max=0.45 / padx_max=28`. Counter `MOMENTUM_SHORT_WEAKCAP`. Operator-directed ("ship it").

**Origin — chasing the MOM-short range<15 drag.** On the COMBINED dedup pool (`reports/COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv`), MOM-short range<15 = N=18 / 39% WR / −$416. Investigation path:
- **Exits refuted:** fast-exit/lower-TP all WORSE than actual (−$2.5k..−$6.5k) — losers peak <0.25% (too low to bank), winners run far (cap forfeits them). Give-back hypothesis dead; the losers are near-DOA.
- **Broad `range<15 + pATR<0.45` blocker rejected:** forfeits the SUI +2.08% runner (+$157, same entry signature as losers) AND blocks 100% of C1 (all C1 is range<15) → kills the C1 N≥5 hold.
- **The fix = add `pADX<28`:** pADX cleanly separates by behavior — the 3 winners (incl. SUI runner) all have pADX≥28; the DOA losers are pADX<28. Cut = N=4 non-C1 / 0W/4L / all peak<0.10 / −$263.
- **Drop the non-C1 carve-out (operator insight "all C1 are negative"):** blocking by BEHAVIOR not tag — a low-pADX C1 (XLM, pADX 25.8, peak +0.00, the inside-band-2× DOA hole) IS caught at entry; trend-backed C1 (HYPE pADX 33.2) still fires for the N≥5 gate. This entry-blocks XLM (so the −$239 never opens) without needing a separate C1 revert, and resolves the "why protect an all-negative cohort" inconsistency.

**Evidence:** the block cell = N=4 (non-C1) or 5 (incl XLM) / 0% WR / every trade peak<0.10 (DOA) / −$263..−$502. ZERO winners in the band — the SUI +2.08% runner sits at pADX 31.7, above padx_max. Mechanism triple-confirmed: capitulation entry (range<15) + low vol (pATR<0.45) + weak trend (pADX<28) = short with no follow-through.

**Overall pool effect:** full screen +$2757 → **+$3258** (Δ +$501), MOM-short sleeve −$159 → positive, 0 winners forfeited, C1 gate preserved (HYPE still fires).

**Discipline note.** N=4 << gate → DISCIPLINE-OVERRIDE, parallel to `momentum_short_btc_atr_min` (N=5). Asymmetric-safe: band holds only DOA losers (all peak<0.10), so winner-forfeit risk ~0 by construction; mechanism is structural, not curve-fit. In-sample on the COMBINED pool → apply 30-50% haircut for forward expectation.

**D11 wiring (all 6, grep-verified, py+json compile):** ① config.py 4 fields + evidence comment ② trading_config.json ③ engine block in normal-entry SHORT chain after `MOMENTUM_SHORT_LOATR` (computes range_position from price/high_20/low_20; fail-open on missing inputs) ④ UI toggle+3 inputs under the momentum-short BTC-ATR field ⑤ load (checkbox + 3 `_setVal`) + save ⑥ `_record_filter_block("MOMENTUM_SHORT_WEAKCAP")`.

**TIGHT REVERT (override-grade):** set `momentum_short_weakcap_enabled=false` if ANY blocked-band trade closes ≥+0.30% (first time it would kill a winner), OR blocked-zone WR ≥40% on N≥8 fresh.

**Related (held, NOT shipped):** C1 2× cell stays as-is (breadth de-mux live) on the N≥5 verdict gate — the weak-cap block now entry-blocks the low-pADX C1 (XLM) so the C1 2× hold no longer carries that DOA-tail risk; the C1 gate accrues on trend-backed (pADX≥28) fires only. Watchlist.

---

## 2026-06-29 — Two ships (C1 de-mux + flip-short bear floor) + exit/cell watchlist + forward batch

**Context:** Post-reset forward batches 06-29 (13 trades, 7W/6L, −$442). Dominant loss = 3 high-ATR pairs (SKYAI/MANTA/AGLD, pATR 1.0-1.28%) gapping the −0.70 SL to −1.2% (−$396 ≈ 90% of the batch). De-mux/filter/exit all proven unable to reach the gap tail (de-lever roadmap remains the only lever; not an expectancy fix — variance/Kelly).

**SHIP 1 — C1 SHORT 2×→1× + breadth de-mux disabled.** `pattern_cell_rules` C1 SHORT `inv_mult 2.0→1.0`; `c1_short_demux_breadth_enabled true→false` (config.py + trading_config.json). Pooled in-sample+forward C1 SHORT = 1W/4L / −$661@2× (−$330@1×), losing at EVERY qs (qs=2 holds the only winner WLD; qs≥3 = 0W/2L). The "de-mux only C1+qs2" hypothesis was tested and REFUTED (qs does not separate C1). REVERT: re-enable a C1 multiplier only at N≥30/WR≥70%/+avg cross-batch.

**SHIP 2 — `flip_short_bear_min=20`** (config.py default 20.0; trading_config.json; engine `_flip_filters` SHORT block, counter `FLIP_SHORT_BEAR_MIN`; UI input + load/save in index.html). Block FAN flip-SHORT when entry bear% < 20. Fine bear-band split (COMBINED + 06-29 fwd) localises the loss ENTIRELY to bear<20 (1W/4L/−$314, no high-ATR confound); bear 30-40 WINS (+$146), 50-80 are the edge → floor=20 NOT 40 (a <40 floor forfeits winners). 06-29 币安人生 (bear 17.5, −$95 DOA) lead confirm. ⚠ N=5 DISCIPLINE-OVERRIDE. TIGHT REVERT: →0 if blocked bear<20 flip-shorts hit >40% WR or net-positive on N≥10 fresh.

**REFUTED this session (do not re-propose):** (a) `FLIP_SHORT_HIATR` tightening — high-ATR flip-shorts WIN by sleeve (flip ≥0.9 = +$4.3/64%); whole-pool ≥0.9-negative was a screened-out-junk confound. (b) qs=2 flip-short floor→3 — qs=2 is the biggest winner cohort (+$519 in-sample); this batch's qs=2 −$215 was the high-ATR gap confound. (c) unmatched-long pADX<18 block (fails C-gate 4/50%/−0.197, non-monotonic, chops WLD; losers span both pADX extremes LTC15/JTO25.8 = feature-inseparable) and GVol≥0.85 de-mux (HARMFUL −$56, halves XPL +$314 winner) — logged D③, re-screen each batch. (d) regime as a dynamic-exit signal (transition does not separate long outcomes: HELD 77%/+0.161 vs DRIFTED 72%/+0.166).

**WATCHLIST added:** (G) non-strict EMA13-cross exit for LONGS only — pool CF NET +2.3% (saves losers, momentum break real) vs SHORT −1.0% wash (fades whipsaw); catches LTC (crossed EMA13 −0.16% but strict held to −0.69% SL). Make strict toggle direction-aware; forward-validate via EMA13 Strict-Mode table, ship at N≥10-15 fresh. Short fast-exit ~0.25/5min = pool CF +13-14.6% (catches pop-then-gap SKYAI) BUT contradicts leash-shadow (wider trail helps shorts) + optimistic fill → forward-test only, reconcile first.

**Pool:** COMBINED appended +13 (321→334 rows). Re-screened baseline (current stack, C1 1× + bear<20) = 97 / 69% / +$3210 (was 84/73%/+$3723 in-sample-only; now mixed in+forward).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-06-30 — SHIP: MOMENTUM-SHORT W1-regime block (`momentum_short_w1_block_regimes='HEALTHY_BEAR'`) — DISCIPLINE-OVERRIDE N=20

**What:** New entry filter. Block a MOMENTUM-SHORT that matches pattern **W1** ("HighConv trend") when entry BTC regime ∈ a comma-separated list (live = `HEALTHY_BEAR`). Empty = off. Flips bypass (gated by `_flip_filters`); momentum-shorts reach the `open_position` path.

**Evidence (analysed ONLY on `reports/SCREENED_BASELINE.csv` per the screened-pool rule + the fresh 06-29/30 batch — 2 independent windows):**
- W1 mom-short **HEALTHY_BEAR = N=20 / 40%WR / −$650 / avg −0.265%** (baseline 17·47%·−$383 + batch 3·0%·−$267, direction-consistent).
- **Confound check PASSED:** non-W1 mom-short CONTROL in the SAME regime = **N=7 / +$24 / +0.014% (breakeven+)** → the drain is W1-specific, not a regime-wide "all shorts lose in HEALTHY_BEAR" effect.
- Loss **diffuse** (no pair ≥60% — not a blacklist).
- **STRONG_BEAR W1 WINS (N=4 / 75% / +$229)** → deliberately NOT in the block list (regime-asymmetry, same lesson as the FAN_PAIR_ADX STRONG_BEAR exemption).
- Batch counterfactual: blocks exactly PUMP −$91 / NEAR −$75 / AVAX −$101 (**+$267**); preserves non-W1 winners XLM +$48 / AAVE +$68 and flip winner CHZ +$68. Bearish sleeve −$359 → −$92.

**Gate status (transparent):** Locked promotion gate = N≥30 AND WR≤40% AND avg≤−0.20% AND NP≥60%. Here **N=20 < 30** and **WR=40% exactly at the bar** → **DISCIPLINE-OVERRIDE**. Justified per the methodology's "N≥30 *or* ≥3-sample direction-consistent cross-batch" clause: 2 independent windows same sign + a clean positive control + diffuse + regime-specific. avg −0.265% clears −0.20%.

**Wiring (D11):** config.py default+comment · trading_config.json · engine block in `open_position` after `PATTERN_CELL_BLOCK` (`direction=="SHORT" and not flip_source and not bull_long and not bounce_long and _pw1_e and entry_btc_regime in list`), counter `MOMENTUM_SHORT_W1_REGIME` + `_seed_phantom_flip(...,"SHORT","MOMENTUM_SHORT_W1_REGIME")` LONG-fade for revert observability · UI text input under the momentum-short weak-cap row + load/save. Grep-verified id ×3; predicate dry-run over the 11-trade batch blocks exactly the 3 W1-HEALTHY_BEAR shorts and nothing else.

**TIGHT REVERT:** clear the list (→'') if the `MOMENTUM_SHORT_W1_REGIME` phantom (LONG fade) goes **net-NEGATIVE on N≥10 fresh** (= the blocked shorts would have won). Widen to more regimes ONLY if STRONG_BEAR W1 mom-short flips negative on N≥10 (today exempt 4·75%·+$229).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

**POOL + BASELINE RE-FREEZE 2026-06-30 (same session, after the W1 ship + final batch):** COMBINED pool grown **334 → 345** (+11 = the 06-29/30 batch id 5-15, all new keys, deduped; pre-append snapshot at `reports/COMBINED_..._DEDUP.csv.prebatch_bak`). Batch CSV saved `reports/orders_2026-06-30_2208_11trades_postW1ship.csv` + text placeholder `reports/report_2026-06-30_2208_11trades.txt`. `scripts/screen_pool.py` updated: `sleeve()` MOM_SHORT now applies the W1-HEALTHY_BEAR block (current-stack parity); anchors updated **23/$2146 → 25/$2496 (MOM-long)** and **28/−$122 → 14/$310 (MOM-short)** + new assert `0 W1-block survivors`. Re-frozen `reports/SCREENED_BASELINE.csv` = **85 survivors: MOM-long 25·84%·+$2496 · MOM-short 14·57%·+$310 · FLIP-short 46·72%·+$561 · TOTAL 85·73%·+$3367.** **Archived prior baseline (pre-W1, pre-batch): MOM-long 23·83%·+$2146 · MOM-short 28·50%·−$122 · FLIP-short 43·74%·+$702.** Independent reconciliation: MOM-short 28·−$122 − 17 W1-HB-COMBINED (47%/−$383) + 3 batch survivors TAO/XLM/AAVE (−67/+48/+68) = 14·+$310 ✓. Operator resetting paper after this → next batch is fresh; compare it against THIS baseline.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-06-30 — REVERT W1-regime block + SHIP pair-volume block (same session)

**REVERT `momentum_short_w1_block_regimes` → '' (off).** The W1-HEALTHY_BEAR block (shipped earlier this session) failed the FIRST cross-period test we ran: same W1+HEALTHY_BEAR+short entries WON +$1175/65%WR (≤06-13) but LOST −$650/40%WR (06-16→30). Diagnostic (peak/trough excursion): the driver is FOLLOW-THROUGH, not exits — avg peak% halved +0.611→+0.345, %reaching +0.30 favorable fell 73%→45%, while avg trough was identical (−0.50 vs −0.54). The BTC-regime LABEL "HEALTHY_BEAR" does NOT capture follow-through (same label, opposite behavior) → the filter was overfit-to-window, not adaptive. Operator's principle (correct): filters must encode a stable entry-measurable condition that works in any period.

**Variable hunt (what separates mom-short winners from losers, robustly):** ranked all entry variables on the 34-trade mom-short universe (COMBINED 345). adx_delta (−0.71σ) and bear_pct (−0.55σ) looked best but **INVERT across periods** (adx_delta: recent <1.0 wins / old <1.0 loses; bear_pct ≥75 recent loses / old wins) = overfit, rejected. **`entry_pair_volume_ratio` is the ONLY non-inverting separator:** pair_vol<1.0 = 69%WR/+$392 (recent) AND 64%WR/+$449 (≤06-13); pair_vol≥1.0 = 28%WR/−$732 recent AND net-negative ≤06-13. Mechanism: shorting into HIGH pair volume = climactic/exhaustion → bounce (no follow-through); LOW vol = orderly continuation. This directly explains the W1 follow-through collapse. Blocking ≥1.0 is +EV in BOTH windows (+$732 recent, +$318 old avoided) — the cross-period property W1 lacked.

**SHIP `momentum_short_pair_vol_max=1.0`** (block momentum-SHORT when entry pair-vol ≥ this; 0=off). MAX semantics — distinct from the legacy `pair_volume_threshold_short=1.1` which is a MIN (require ≥, the OPPOSITE) and stays OFF; legacy pair-vol filter remains disabled. D11: config.py + json + engine block in `open_position` after the (now-inert) W1 block + UI input/load/save (id ×3 verified) + counter `MOMENTUM_SHORT_PAIRVOL`. Momentum-only (flips bypass via `_flip_filters`). ⚠ DISCIPLINE-OVERRIDE (N=34) but passes cross-period (unlike W1). TIGHT REVERT: 0 if pair_vol≥1.0 mom-shorts return ≥50%WR AND net-positive on N≥15 fresh, OR if the kept <1.0 side drops <55%WR on N≥15.

**BASELINE re-freeze v2 (`screen_pool.py`):** `sleeve()` MOM_SHORT now applies the pair_vol≥max block (W1 logic kept but config-disabled); anchors 14/$310 (W1-block v1) → **16/$392 (low-pair-vol cohort)**; assert switched to `0 pair_vol≥max survivors`. SCREENED_BASELINE = **87 survivors: MOM-long 25·84%·+$2496 · MOM-short 16·69%·+$392 · FLIP-short 46·72%·+$561 · TOTAL 87·75%·+$3449.** Today's batch (11) under the filter: mom-shorts −$218 → +$116 (blocks TAO/PUMP/NEAR/AVAX = −$334, keeps XLM/AAVE).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-01 — Phantom Flip Tracker: RETIRE all sources except LONG_UNMATCHED_ONLY (collect + display)

Operator principle: don't run phantoms we don't display — either keep the UI or remove from the DB too. Chose the latter (stop collecting the retired sources).

- **Engine (`_seed_phantom_flip`):** added an allowlist `_PHANTOM_KEEP_SOURCES = {LONG_UNMATCHED_ONLY, MOMENTUM_SHORT_W1_REGIME}` — a single gate at the top of the function; every other seed site (PAIR_ADX_MAX / BTC_ADX_BLOCK_SHORT / PAIR_RSI_ADX_CROSS / PAIR_TREND_FILTER / FAN_RATIO_GATE / FAN_CONTROL / Pair RSI>65 / PASS:*) becomes a no-op. One line, reversible.
- **DB (`database.py::init_db`):** idempotent startup cleanup `DELETE FROM phantom_flips WHERE source_filter NOT IN (allowlist)` — purges the historical retired rows on the live server after deploy.
- **Report (main.py + templates):** Phantom Flip Tracker now shows ONLY the LONG_UNMATCHED_ONLY matched-long→SHORT fade, broken out **per PATTERN (threshold lowered 3→1 so the W6 flip-short lead + all patterns surface from trade 1)** and **per REGIME** (operator ask). REMOVED the **Fan-Ratio Curve** + **Leftover-Filter Test** tables (UI + both export blocks). Source×BTC-Regime naturally focuses on LONG_UNMATCHED (only source left). The aligned/counter + pADX + BTC-cell sub-blocks removed (dead once FAN/cross sources retired).
- **Why safe:** the retired sources were the PRE-LIVE flip-short research surface; flip-shorts are now LIVE and analysed via SCREENED_BASELINE + the live Flip Trade Log (real fills, not naked fades). Their watchlist candidates (bear-70-80, qs≥3 multipliers) are already captured in CURRENT_STATE — nothing lost. Observation-only, ZERO live-trading impact (phantoms never trade).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-01 — Flip-short ENTRY-SEPARATOR search: tested EMA/RSI/DI variables + 10 DI-diff 2D combos → ALL REJECTED (overfit, invert out-of-sample)

Operator-driven, triggered by an XPL flip-short loser (id 18, `FLIP:FAN_RATIO_GATE` SHORT, −0.79%, RSI rising +11 into a reversal bottom). Question: does any entry slope/momentum variable divide FAN flip-short winners vs losers? Analysed **SCREENED_BASELINE.csv** (46 flip-shorts, 71.7% WR) and cross-checked the fresh **07-01 batch** (9 flip-shorts) + dedupe pools.

**Single-variable results (SCREENED_BASELINE flip-shorts):**
- `entry_ema20_slope` — **no separation** (win-mean +0.166 ≈ loss-mean +0.181).
- `entry_ema50_slope` / `entry_pair_ema20_ema50_gap_pct` (EMA13-50 gap, misnamed) — grade WR weakly but **NON-MONOTONIC** (terciles 56%→87%→73%) AND **no net-negative zone** (deepest-neg quartile still +0.120% avg / +1.44%). The two are highly correlated (carve identical trades) = one signal. Not a divider.
- RSI-direction (`entry_rsi − entry_rsi_prev`) — **non-monotonic confound** (terciles 50%→87%→80%); XPL's big RSI-rise lands in the 80%-WR tercile. "Short into rising RSI = loser" is FALSE in data.
- **DI-diff (`entry_pos_di − entry_neg_di`, +DI/−DI spread from Wilder DMI)** — the ONLY monotonic, net-negative single divider in-sample: terciles WR 87.5%→73.3%→**53.3%**, top tercile (>12.9) avg **−0.234% / −3.51%**. XPL sits here (13.2). Mechanism coherent (+DI≫−DI = bullish momentum = shorting into strength).

**DI-diff filter before/after (block DI-diff ≥ 13):**
| pool | before | after (kept) | blocked cohort |
|---|---|---|---|
| BASELINE ≤06-30 | 46·71.7%·+8.37% | 32·78.1%·+10.90% | 14 (8W/6L)·57.1%·**−2.53%** GOOD |
| CURRENT 07-01 | 9·66.7%·+0.53% | 6·66.7%·+0.06% | 3 (2W/1L)·66.7%·**+0.47%** HARMFUL (inverts) |

The blocked 07-01 trades: XPL id18 (DI 13.1, −0.79% ✓) but also JTO id17 (DI 15.6, **+0.79% ✗**) and XPL id12 (DI 21.6, **+0.48% ✗**). The 2 losers it MISSED (JTO id8 DI 9.1, TIA id7 DI 5.2) had LOW DI-diff — opposite of the baseline prediction.

**10 DI-diff 2D combos (DI≥12 AND second var) — same cell, baseline vs 07-01:**
- `ema13_50gap<0`: baseline 25.0%WR/−4.09% → 07-01 66.7%WR/**+0.47%** (inverted)
- `ema50_slope<0`: 33.3%/−3.93% → 66.7%/**+0.47%** (inverted)
- `pair_vol≥1.0`: 25.0%/−2.93% → 100%/**+1.26%** (inverted hard)
- `global_vol≥1.0`: 33.3%/−4.32% → n=0
- `adx_delta≥0.5`: 54.5%/−0.97% → 75%/**+0.64%** (inverted)
- `dist_ema13≥0.7`: 45.5%/−2.34% → 100%/**+1.43%** (inverted hard)
- (only cells negative on 07-01 were N=1 = XPL itself, circular)

**EVERY 2D cell that looks clean in-sample (down to 25%WR/−4%) is net-POSITIVE out-of-sample.** Textbook overfitting on a 46-trade window — more slicing → cleaner in-sample fit + harder inversion.

**Structural wall (why this can't be fixed by more analysis):** the FAN flip-short sleeve is <3 weeks old — **0 flip-shorts before ~06-13** in BOTH `dedupe_pool_FULL.csv` (525 shorts, `entry_strategy` all empty) and `dedupe_pool.csv` (503 shorts, empty). No historical window exists → cross-period validation (the W1-killer test) is IMPOSSIBLE; the ONLY test is forward, and the first forward batch (07-01) refutes every candidate.

**VERDICT — do NOT ship any flip-short entry filter from these variables; do NOT re-derive.** Confirms the standing conclusion (memory `reference_combined_momflip_pool`): the flip-short sleeve is **inseparable at entry — the edge IS the sleeve**. Losers → pair-blacklist (when ≥60% of a loss zone is 1–2 pairs) or exit/BE mechanics, never entry dimensions. Note on DI: `entry_pos_di`/`entry_neg_di` ARE stored per-order (computed `services/indicators.py:47`) but the DI spread is NOT surfaced in any UI table (`avg_di_spread` is orphaned payload; `_di_spread_bucket` at main.py:2400 is dead code) — not worth building a table for a rejected signal. Value of this negative result: it prevented shipping a filter that would have removed net-positive trades live (+0.47% on the 07-01 batch alone).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-01 — Baseline re-freeze v3: weakcap parity closed in screen_pool.py (MOM-short 16/$392 → 15/$443)

Caught in the Jul-1 operator-requested deep dive: `momentum_short_weakcap_*` (engine-live since Jun-28, blocks SHORT when range<15 ∧ pairATR<0.45 ∧ pairADX<28) was NEVER ported to `scripts/screen_pool.py` — the screen passed TAO 06-24 (rng 6 / pATR 0.29 / pADX 26.4, −$51 loser) that today's engine would block. Added the weakcap block to the MOM_SHORT branch (exact engine parity), anchors v2→v3.

**SCREENED_BASELINE v3 (86 survivors): MOM-long 25·84%·+$2496 (unchanged) · MOM-short 15·73%·+$443 · FLIP-short 46·72%·+$561 (unchanged) · TOTAL 86·76%·+$3500.** Only TAO removed (the other 15 mom-shorts all pass weakcap — checked per-trade). Prior anchors archived: 23/$2146+28/−$122, 25/$2496+14/$310 (W1 v1), 25/$2496+16/$392 (v2).

Also from the same deep dive (evidence, no ships):
- **Exit stack VERIFIED GOOD on post-exit data:** every MOM-long loser SL (JTO would have bled to −2.2%) and every MOM-short EMA13-cross cut (SOL→−1.06, AAVE→−0.81, TAO→−0.81, FARTCOIN→−0.64 at 5min) was the right call. No exit change.
- **⚠ shadow_atr05 flip-trail Δ+3.01% is CONTAMINATED — do NOT ship or re-derive:** the shadow does not respect the hard SL. Its "gains" are FLIP_STOP_LOSS trades riding through the stop and recovering (ID −0.98→+0.45, ONDO −0.70→+0.48, TIA −0.86→+0.49) and its tail is SKYAI −1.19→−6.66 (5.6× the SL on one position at 20×). Net +3.01% = +11.03 post-SL recoveries − 8.02 bleeds = fat-tail asymmetry, same class as the rejected caps-for-losers. Clean variants (tight/cap035/cap050) all Δ-negative → current FLIP trail (1.0×ATR + hard SL) is locally optimal among tested variants.
- **GVol threshold sweep (both windows):** recent (baseline+07-01, N=28): de-mux netΔ$ ≈ 0 at every cut (−$71 @0.85 / +$51 @0.90 / −$36 @0.95 — noise; losers spread 0.79→1.10, no crisp separation). OLD pool (≤06-13, N=70 UNMATCHED longs): **0.85 is the ONLY cut with separation** (≥0.85 −0.132% vs <0.85 −0.016%; at ≥0.90 separation vanishes: −0.060 vs −0.052). → If the GVol de-mux ships, the threshold is 0.85 — 0.90+ would be recent-noise fitting.
- MOM-short range-position ("don't short the hole") REFUTED: whole sleeve trades rng<25 and wins there; both buckets positive.
- FLIP-short losers: 10/13 gap-through DOA across 10 DIFFERENT pairs → no blacklist add.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

**ADDENDUM (same day, operator methodology correction):** the GVol sweep's old-pool leg above must NOT be read as ship support — old-pool P&L reflects OLD exit mechanisms (inadmissible as evidence for the current stack; cross-period may only REFUTE, never SUPPORT — new memory `feedback_crossperiod_refute_only`). Baseline-only verdict (v3 + 07-01, N=28): GVol≥0.85 = 12·66.7%·+0.013% avg·+$142@2×, de-mux netΔ −$71 → **WATCHLIST, not ship.** Gate: promote if the cohort stays ≤+0.10% avg on N≥20 fresh (now 12, 1 fresh). The 0.85-threshold finding stands (0.90+ = noise-fitting) as the cut IF it ever ships. CURRENT_STATE item rewritten accordingly (the earlier "SHIP-worthy via cross-period" framing retracted).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-01 — Winner-MAE (worst-trough) SL analysis by sleeve → MOM-short −0.60 registered as watchlist (G2); flip + mom-long SLs confirmed correct

Operator asked whether any sleeve's SL should be lowered, based on the worst trough of WINNERS (MAE). Method: winners' worst adverse dip = the room winning trades actually need; counterfactuals use optimistic no-gap fills (negative verdicts are conservative). Pool = SCREENED_BASELINE v3 (86).

- **FLIP-short (33W/13L): DO NOT tighten.** Winner MAE min −1.02 / p10 −0.79 / median −0.35; two winners used ~all of the ATR-widened ~−1.05 room (troughs −1.01/−1.02 → wins +0.68/+0.63). Every tighter SL net-negative with perfect fills: −0.30 → −11.92%, −0.50 → −2.97%, −0.70 flat → −4.69%, −1.00 → −2.31%. Mechanism: the fade structurally endures a final push before rollover. Re-confirms the Jun-28 caps-for-losers rejection on the cleaner v3 pool.
- **MOM-long (21W/4L): leave alone — base −0.70 near-perfectly calibrated.** Deepest winner dips: TAO −0.67 (ATR 0.35, living on the BASE −0.70 with 0.03 margin) and ACT −0.66 (ATR 1.48, on the widened room). −0.60 kills both → net −2.13%. Only theoretical gain = one trade's widening (JTO −1.13→−0.70 = +0.43) — not worth breaking the high-ATR-runner protection (locked Jun-13 note).
- **MOM-short (11W/4L): the one genuine candidate.** NO winner has ever dipped past −0.45 (median MAE −0.22) — the −0.70/−1.00-wide/−1.20 room is unused by winners, pure loser-budget. SL −0.60: kills 0 winners (margin 0.15), saves +0.60%. SL −0.50: kills 0, saves +0.76% but margin 0.05 = razor-thin. Mechanism sound: continuation trade — median winner dip −0.22; a >0.6% bounce against = thesis refuted. **NOT shipped: realized saving is N=1** (MANTA −1.20, the only SL-depth loser; SOL/AAVE/FARTCOIN exit via EMA13-cross at −0.40..−0.56 before SL binds → SL rarely binds for this sleeve).
- **Structural takeaway:** fade (flip) needs deep SL room; continuation (momentum) needs little; today all sleeves share the same base. The differentiation worth making eventually: mom-short tighter, others unchanged.

**Registered CURRENT_STATE watchlist G2 with pre-committed gate:** ship mom-short-specific SL −0.60 (base+wide) when N≥20 mom-short winners still show max MAE shallower than −0.60 AND ≥2 SL-depth losers accumulate; abandon if any mom-short winner dips below −0.60 and recovers. Re-run the winner-MAE table each batch.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

**MECHANISM-CORRECTION ADDENDUM (same day, operator challenge: "did you model the ATR widening and the −1.20 floor?"):** No — the first-pass counterfactual compared actual outcomes against a FLAT SL at X, implicitly deleting the widening/floor. Re-run with the exact engine formula (`effective_sl = min(base, −ATR×1.5), floor −1.20`, widen-only, trading_engine.py:9409-9427; base −0.70 signal-dead / −1.00 signal-active; NO sleeve has a fixed SL — the −0.35 json rows are untraded LOW/MEDIUM confidence tiers; only pattern cells carry fixed SLs, e.g. C1-SHORT −0.70):
- **MOM-short base −0.70→−0.60 = NO-OP in-sample** (kills 0, saves 0): widening dominates above ATR 0.40 (effective stays −0.71..−1.20 regardless of base), and below it the EMA13-cross exits fire at −0.40..−0.56 before any SL binds. The earlier "+0.60% saving" was the flat-SL artifact. G2's lever REWRITTEN.
- **The real mom-short lever = sleeve-specific floor −1.20→−0.90:** saves MANTA (the only SL-depth loser, −1.20→−0.90, +0.30%), kills 0 winners (deepest winner MAE −0.45; BEL/JTO whose SLs would tighten to −0.90 have troughs −0.26/−0.17). Still N=1 benefit + needs per-sleeve-floor plumbing → watchlist, gate updated (≥2 floor-depth losers + N≥20 winners with MAE shallower than −0.60).
- **FLIP re-verified under the real formula — even stronger do-not-touch:** DEXE won with trough −1.01 vs its effective SL −1.08 (0.08 margin); ONDO won at −1.02 on the −1.20 floor. A −0.90 floor kills both.
- **MOM-long unchanged:** TAO's 0.03 margin is on the real base −0.70 (ATR 0.35 → no widening).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-01 — Watchlist E resolved to a design: SHORT-LOSS BRAKE (quantified; concurrency CAP rejected; build pending operator go)

E's gate ("quantify how often a short-cluster maps to one BTC bounce before designing the cap") executed on the CURRENT stack (SCREENED_BASELINE v3 + 07-01 batch, 72 shorts):
- **The 06-28 −$779 cluster is mostly extinct under today's filters** — its members (C1 AAVE/HYPE, weak momentum shorts, W2+W1 PUMP) are blocked by since-shipped weakcap / bear-breadth floor / ADXD / C1→1×. Residual clustering: 11 overlap-pairs + one 3-cluster; only ONE all-loss pair (TIA+BCH −$209 = 10% of short losses).
- **Concurrency cap REJECTED (net-negative):** the only ≥3-concurrent cluster was 3 wins (+$193, XPL/LIT/XLM correlated breakdown — the moment the short book SHOULD be loaded). A cap can't distinguish good clustering (breakdown) from bad (bounce); it forfeits more than it saves.
- **SHORT-LOSS BRAKE (block new shorts 30min after any short-SL close): saves +$548** — blocks 5 (1W/4L: WIF −321 / JTO −129 / AGLD −128 / AAVE −38 vs 币安人生 +68), 4 distinct dates, 4 pairs, window-robust (45min +$519 / 60min +$483), and each of the 3 worst short 2h-windows (−$327/−$219/−$191) contained a brake-catchable second loss. Mechanism: a fresh short stop-out = the tape is actively bouncing against shorts — don't add; asymmetric (only reacts to realized damage, never blocks winning clusters); event-triggered state machine, not a mined feature.
- Discipline: N=5 = DISCIPLINE-OVERRIDE, direction-consistent on 4 dates (≥3-sample rule), no pair concentration, 30-50% haircut → ~+$275-380/2wk expected. Build spec + tight revert gate registered in CURRENT_STATE E (would-be-blocked shorts are retro-identifiable in any batch CSV, no new tracking needed).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-01 — Balance→Leverage Schedule: design rule fixed (threshold = TargetGross/L), guess tiers rejected

Operator drafted the UI schedule with the original guess tiers (0:20/10k:15/25k:10/100k:5). Analysis: mechanically valid but never converges to constant volume — within-tier gross still compounds 3-4× (10× tier: $250k→$990k; $198k/position at $99k balance vs thin-pair engine caps $30-50k and a ~$25-30k/pos slippage ceiling). The transition phase's goal is a ramp to CONSTANT gross (lev ∝ 1/balance = the fixed-volume cap via the lev knob), so the correct generator is **threshold(L) = G/L**. Concrete table at G=$150k placeholder registered in CURRENT_STATE ①: `0:20, 10000:15, 12500:12, 15000:10, 19000:8, 25000:6, 30000:5, 37500:4, 50000:3, 75000:2, 150000:1` (overshoot ≤~30%/tier; plateau from ~$7.5k balance). Locks: G calibrated from exit_slippage_pct telemetry (not chosen a priori); schedule ends at 1× → working-capital reserve mode is the continuation past balance=G; pick-one-primary still applies. Inert at $3k either way.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-02 — Flip-short batch W-vs-L cross-check: every flashy axis inverts except bear-breadth; bear≥80 registered as BLOCK candidate with pre-committed gate

Operator-requested deep analysis of the 07-01/02 batch's "Entry Conditions by Strategy — Winners vs Losers" (FAN flip-shorts 6W/4L: ZBT −1.19 / JTO −1.06 / XPL −0.79 / TIA −0.72; mom-shorts 2W/0L = nothing to separate). Method: theory-first per axis, then the inversion test vs SCREENED_BASELINE v3 (46 flips).

**Inversion results (batch direction vs baseline direction):**
- ADXΔ: batch losers LOW (+0.12 vs +0.51) / baseline losers HIGH (+0.43 vs +0.12) → ✗ INVERTED (3rd flip of this axis).
- RngPos: batch losers at range top (90 vs 70) / baseline NO separation (66.6 vs 68.3) → ✗.
- Pair EMA13-50 gap: batch losers positive-gap / baseline WINNERS positive-gap (+0.18 vs −0.21) → ✗ (the seductive "short before breakdown confirms" theory refuted both directions).
- BTC 1h slope: batch losers 1h-rising (ZBT +0.46/XPL +0.35) / baseline 1h>+0.10 flips = **88.9%WR, the BEST cohort** → ✗ (fades ≠ continuations; the momentum-short 1h filter does NOT generalize to flips).
- Pair 24h vol: batch losers thin (<$60M 2W/3L) / baseline thin pairs = best cohort (77.8%WR +7.03%) → ✗.
- BTCExt%: trivial both windows. **Breadth(bear%): CONSISTENT — the only survivor** (losers higher-bear both windows).

**bear≥80 (exhaustion zone, 3rd consistent window):** combined N=11, 5W/6L, 45.5%WR, −0.02% avg; holds the 4 worst losers (PUMP/AGLD/ZBT/WIF −1.16..−1.24 + TIA) AND the 2 biggest winners (ORDI +2.80, H +1.28). Block impact: baseline +$25 ≈ 0 (forfeits $609 winners / saves $634); fresh batch +$237 clean (0 winners forfeited) — 90% of the $ case is 2 fresh trades. Fails the C-gate (needs ≤40%WR ≤−0.20% N≥30). Mechanism sound (bear≥80 = capitulation extreme → bounce → fade squeezed; breadth-mirror of the shipped bear<20 floor).

**Registered in CURRENT_STATE (winner-cell block, 2026-07-02 update): PRE-COMMITTED GATE — ship `flip_short_bear_max=80` at N≥15 with WR≤40% AND avg≤−0.15%; abandon if the zone re-earns ≥60%WR.** No ship today. Batch also re-confirms the closed entry-separator conclusion: flip W/L are feature-inseparable at entry; per-batch inversion checks are the immune system.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-02 — Capital-scaling v3 SHIPPED: working_capital reserve mode built (OFF) + max gross 30→25 + 4-tier lev schedule ratified

Operator-driven redesign of the Liquidity & Risk Caps section (supersedes the 07-01 11-tier threshold=G/L table):
- **Operator corrections accepted:** (a) a global gross target was the wrong slippage tool — the per-pair 0.1% cap + $500k ceiling are adaptive and already govern capacity; gross only matters for the correlated tail → it can breathe $150-500k as long as cluster-loss %-of-balance falls. (b) 11 lev tiers = over-engineered → **4 coarse tiers `0:20, 25000:15, 75000:10, 250000:5`** (reserve does the fine work; avoids double de-risk by construction). (c) Reserve schedule was too conservative → v3 anchored at operator's "$150k → save $100k, trade $50k"; beyond $250k keep GROWING tradeable to a $100k ceiling hypothesis at ~$500k (validated by exit_slippage_pct), then reserve absorbs all growth. (d) Reserve today = $0 (already live: percentage @ 0%) — full-send growth phase at $3k.
- **BUILT (D11 full): `reserve_mode="working_capital"` + `working_capital_target`** — engine: at the safe-reserve calc, `reserve = max(0, available − target)` when target>0 → tradeable = min(available, target), reserve auto-grows with balance, clamping max cluster loss to a fixed $; withdrawals stage FROM the reserve. config.py default + json 0.0 + UI (3rd Reserve-Mode option, shared value input flips to USD, load/save) . **Shipped INERT** (mode=percentage, target=0). At each milestone the operator flips the target manually.
- **SHIPPED: `max_gross_leverage` 30→25.** Clarified mechanics for the record: NOT per-trade leverage (positions stay 20×) — it caps Σ(open notional) ≤ balance×25 and blocks the NEXT entry; binds only under multiplier stacks = exactly the correlated-dump moment.
- **Liquidation math recorded:** 20× isolated → liq ≈ −4.7% price move vs SL −0.70..−1.20 (~4× buffer); residual liq risk = violent gap/bot outage (covered by the ② server-side reduce-only stop item). With isolated margin the absolute worst case = working capital; the reserve is structurally untouchable.
- v3 milestone table in CURRENT_STATE ① (10k/20% · 25k/30% · 50k/45% · 100k/60% · 150k/67% · 250k/72% · 500k/80%→$100k trade).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-02 — Capital-scaling v3 made FULLY AUTOMATIC: reserve_mode="schedule" shipped ON + lev tiers set in config

Operator correction to the same-day working_capital ship: "the whole idea is to be all automatic, not me putting 40000 manually at 100k." Built `reserve_mode="schedule"` + `reserve_schedule` (balance→tradeable tiers, parsed by the SAME `_lookup_leverage_schedule` tier-lookup as the lev schedule; active target = highest tier ≤ free balance; below first tier → no reserve; fail-open). Engine branch in the safe-reserve calc; `_reserve_split` display helper mirrors it; UI = 4th Reserve-Mode option + Balance→Tradeable row-editor (8 rows, live readout "active: trade $X / reserve $Y") + save serializer; balance-card Reserve line emerald when active. D11 full.

**Shipped ON with the v3 table:** `10000:8000, 25000:17500, 50000:27500, 100000:40000, 150000:50000, 250000:70000, 500000:100000` — inert at $3k (below first tier = full balance tradeable). End-to-end verified: reproduces the operator's table at every milestone; between milestones the reserve absorbs all growth ($60k → trade $27.5k/reserve $32.5k). Withdrawal mechanics recorded earlier same day: withdrawals come out of the reserve first, trading untouched while balance ≥ tier target, reserve self-rebuilds from profits, under-target degrades gracefully to full-balance trading.

Also set `leverage_balance_schedule="0:20, 25000:15, 75000:10, 250000:5"` in config (the ratified 4 tiers — the UI had unsaved guess values; config is now the truth). Both schedules inert at $3k; the only live-behavior config remains max_gross 25×.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

---

## 2026-07-02 — Code-review fixes on the auto reserve schedule: C1 equal_split sizing + M2 readout parity; I2 lev-clamp documented

Code review (dispatched on the working tree) findings and resolutions:
- **C1 (Critical, FIXED):** the equal_split per-position sizing branch computed its own reserve knowing only percentage/fixed — in schedule mode it silently fell back to `reserve_fixed=$500`, so per-position base at the $150k tier would be (150k−500)/5 ≈ $29.9k instead of target/5 = $10k (concentration bug: 2 fat positions instead of 5×$10k; total deployment was still clamped by the tradeable cap post equity-fix, but sizing was wrong). Fixed: equal_split's reserve_from_total is now mode-aware (schedule via the same tier lookup on total equity, working_capital via target, percentage, fixed). Verified: $150k → $10k/position, $50k → $5.5k/position.
- **I1 (already fixed pre-commit):** tier keyed on total equity (free+margin) like the lev schedule, not free balance; reviewer confirmed the correct formulation matches what shipped (new tradeable = target − deployed, clamped ≥0).
- **I2 (documented, operator-blessed):** the ratified `0:20` lev tier is NOT fully inert — it clamps the one remaining lev-stacked cell BTC_60-65_22-25 LONG (2×inv×1.5×lev = 30× → capped 20×, eff 3×→2×). Kept: consistent with the capital-scaling de-lever intent and the Jun-19 lev de-mux precedent. CURRENT_STATE multiplier line updated; its verdict gate now effectively judges a 2×-inv cell.
- **M2 (FIXED):** UI schedule readout now treats a 0-valued tier as fail-open (matches engine) — `tgt > 0` guard.
- **M1/M3/M4 (noted, no change):** 0-tier = "trade everything" fail-open semantics (convention); mode-switch save rewrites non-selected mode defaults (pre-existing pattern); UI table fixed at 8 rows (7 used +1 spare — a 9-tier schedule would truncate; revisit if the table ever grows).
- **Withdrawal semantics clarified for the record (operator Q):** the schedule is STATELESS — withdrawing the $22.5k reserve at $50k leaves a $27.5k account that re-tiers immediately to the $25k row (trade $17.5k, reserve $10k) and re-climbs the curve with profits. Each withdrawal steps down the table; risk always matches CURRENT equity.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>

## 2026-07-02 — NAV/share history + true-equity NAV fix + heatmap totals (4a2fb05)
- **BUG FIX:** `_get_portfolio_value` excluded open-position unrealized P&L → NAV/share stepped on every close and deposits/withdrawals while positions were open priced shares at wrong NAV. Now: free + margin + unrealized (from live `current_price`, gross — fees hit BNB at close) + BNB, both paper and live.
- **NavSnapshot** table + hourly server-side upsert loop (one row per UTC-3 day). `/api/investors` serves 120d `nav_history` + `nav_high_water` + tradeable/reserve split. UI: NAV chart (HWM + $1.00 breakeven dashes) + panel sub-lines. D12: NAV section in both text exports via shared `_buildNavHistoryLines`.
- Portfolio panel item #3 (ledger NAV-at-transaction) verified already built (Jun 1 ledger).
- **Day×Time Heatmap:** marginal totals (per-day col, per-block row, grand corner), trade-weighted `dtHeatmapAgg`, all 3 surfaces. Descriptive only — any time-of-day gate still needs N≥30 cross-batch.
- Also this session (analysis, no ship): RngPos×PairRSIdir and 24h-volume flip separators both INVERT baseline↔fresh batch → rejected, do-not-re-derive; XPL blacklist evaluated and rejected (current-stack 3W/1L +$168).

## 2026-07-02 — NAV package code-review fixes (19a93ed)
Review (be35834..5cf0c8b) verified paper equity identity + no fee double-count + D12; found C1 + I1-I5, all fixed same day:
- **C1 (critical, live-mode only):** get_balance fail-opens to zeros → could have persisted NAV≈0 snapshots and priced deposits at NAV≈0. Fixed: 'ok' sentinel, _portfolio_components raises on failed fetch, loop skips ≤0 portfolio, deposit/withdraw 503 on NAV≤0.
- **Equity semantics unified:** balance-card total = TRUE equity (incl. unrealized) like the Portfolio panel; seed + calendar day-% stay REALIZED-only (stable baselines). New `_portfolio_components()` = single source of truth.
- I3 chart try-isolation · I4 HWM whole-table MAX (was 120d window — would decay) · I5 export column width 16 · reset clears nav_snapshots · balance card "Open Positions: N" sub-line.

## 2026-07-03 — SHIPPED: flip BTC-1h regime gate + FAN 2× · baseline v4 · bear≥80/brake closed
**The regime finding (operator-driven).** Jun30–Jul3 batch: flips 16·50%·−$354 while momentum earned. All 4 batch separators (mom-long ext>0.6, flip 1h-slope sub-buckets, RngPos≥85, vol<$50M) INVERTED vs baseline again (now 8 total) — but per-cohort degradation was UNIFORM (4-cohort table: every pattern cohort fell together), and BTC context showed the periods traded different markets under identical 5m regime labels (baseline 1h RSI 43.9/slope −0.07 = falling week; fresh 51.7/+0.10 = V-recovery off 57.8k). Operator hypothesis "baseline was an overall bearish regime" → re-tested 1h slope at SIGN level: **direction-consistent both periods** (baseline slope>0 17·65%·−$73 / ≤0 29·76%·+$774; fresh slope>0 9·33%·−$405 / ≤0 7·71%·+$51). My earlier "refuted" call was a granularity error (judged on a 9-trade sub-bucket) — corrected.
**Shipped ①** `flip_short_btc_1h_slope_max=0.0` (config+json+engine `_flip_filters`+UI+counter FLIP_SHORT_BTC1H_SLOPE; `btc_1h_slope` added to `_ff_in` AND screen_pool's flip_ind for parity). Variant B (block only 5m↓/1h↑, keep both-up) rejected: +$115 better purely on N=2, theoretically worst cell. ⚠ N=26<30 near-gate; TIGHT REVERT →99 if blocked phantoms ≥60% WR on N≥10. **Fixed pre-ship bug:** `float(x or 99)` turned ship-value 0.0 into 99 (falsy) — caught by dry-run screen (filter silently off).
**Shipped ②** `flip_entry_sources FAN_RATIO_GATE:2.0` — kept cohort passes W-gate (36≥30, 75%≥70, +0.236%≥+0.10). ⚠ OPERATOR-DIRECTED 2× skipping 1.5× staging; revert via multiplier machinery (✗ HARMFUL N≥5 → 1.0×, DRAG → 1.5×). History note: FAN 2× was ✗ HARMFUL Jun-15 — that was pre-gate, unconditioned; this 2× rides the regime-gated cohort.
**Baseline v4 re-frozen:** batch (27 rows) appended to COMBINED pool (345→372; filename kept, spans 06-16..07-03; .pre0703_bak saved); anchors ML 34·82%·+$2708 / MS 17·76%·+$661 / FLIP 36·75%·+$845 / TOTAL 87·78%·+$4214 (+ new FLIP N assert). Flip $ at historical 1× — de-mux forward. Batch archived reports/batch_2026-06-30_to_07-03.csv; text report reports/SCREENED_BASELINE_v4_report.txt.
**Closed:** bear≥80 block gate KILLED (shadow of 1h slope; incremental −$46 combined after the gate). SHORT-LOSS BRAKE REJECTED by operator (patch; clusters now blocked at entry). Watchlist tallies: GVol≥0.85 demux 5/20 @+0.01% on-track; G2 mom-short MAE consistent (2 more shallow winners); XPL blacklist rejected (3W/1L +$168 current-stack).

## 2026-07-03 (later) — code-review fixes on the regime-gate ship
Reviewer runtime-verified the full chain (gate fires at slope>0, 2.0x parses size2/lev1 + stamps cell_multiplier, screen byte-reproduces, json semantic diff = exactly 2 changes). Fixed from findings:
- **I1 + minors 4/5 — three more `or`-falsy filter bugs** (same class as the pre-ship 0.0→99 bug): `flip_adx_delta_min` (0.0 = the plausible "block ADXΔ<0" setting would silently DISABLE), `flip_short_btc30_rise_block_min` (None would fail-CLOSED), `flip_long_regime_block_adxd_max` (0.0→99 = block-regardless). All → explicit None-checks.
- **I2 — screen_pool flip de-mux:** pnl_current now de-muxes ANY multiplied FLIP row to 1x (was C1-only). The new $-anchor immediately caught that the $845 pin counted the Jun-17/18 legacy 2x flips at 2x — **true 1x flip baseline = $825, TOTAL $4194** (matches the original analysis figure). Re-pinned everywhere.
- **I7 — FLIP anchor now count+$ (36/$825).**
- **I3 (logged, no code):** Filter-Blocks counter undercounts FLIP_SHORT_BTC1H_SLOPE (SHORT blocks skipped while macro=BULLISH, which correlates with slope>0). Phantom tracker is the true revert-gate surface.

## 2026-07-04 — GATE REGISTERED: max_open_positions 5→4 (operator-proposed, roadmap item)
Capital-utilization move: equal_split at 4 slots = +25% margin/trade (+25% $ expectancy, %-invariant) at empirically zero cost (cap-cost counter has never recorded a block at max-5; the one ≥3-concurrent cluster on record was 3 wins). Accepted costs: 4-SL day −$670 vs −$535; a 2× cell = 50% of book. SHIP at N≥30 fresh post-regime-gate trades with positive net (no stacking with the Jul-3 ship); REVERT on ≥3 forfeited-signal events/batch or a ≥4-concurrent loss cluster.

## 2026-07-04 — BUILT: matched-long PASS phantom (re-enable hunt, operator-driven)
Operator concern: LONG_UNMATCHED_ONLY (Jun 9, blocks ALL C/W-matched longs) may be over-restrictive in the developing bull — the flip lesson applied to longs. Key finding: the Jun-9 evidence predates the modern exit stack entirely (old-era unmatched control −0.096%/50%WR vs +0.31%/82% under today's stack) → matched longs never tested on today's bot. Old-pool (refute-only) regime read: W1 67% WR under old exits; matched longs 57% WR at BTC-1h-rising vs 39% falling. Build: reused the dormant mode='PASS' phantom (Jun 17 bull-hunt machinery) — LONG_UNMATCHED_ONLY blocks now also seed PASS:LONG_UNMATCHED_ONLY same-direction phantom LONGs (added to _PHANTOM_KEEP_SOURCES; own tracker section w/ pattern+regime sub-rows via the generic rows pipe = UI + both exports free). Re-enable gates (per-cell) in CURRENT_STATE. ~2-4 wk to verdict at current block cadence (51 since reset).

## 2026-07-05 — BUILT: SPIKE_REV_BTC phantom (spike-reversion hunt, operator-directed)
Operator: "there were moments" (Jul-5 weekend: 63,084 pump + 62,410 dump, both round-tripped ≤90min; bot correctly sat out via chase-filter/1h-gate). Analyst initially waved it off citing the PAIR_RSI_OB tombstone — operator pushed; conceded the mechanisms differ (RSI_OB faded pair-level overbought trends; this fades BTC-level 15m velocity ≥0.5% — never measured). Built: ~5s BTC sampler in update_phantom_flips (before the empty-state early-return), seeds SPIKE_REV_BTC fade phantoms on BTCUSDT via the standard machinery (keep-sources + generic tracker rows + regime cross-tab; excluded from the fade aggregate like PASS:). Unit-tested end-to-end through the REAL seeder (up→SHORT, down→LONG, 0.4% ignored, dedup/cooldown; one caught test artifact: epoch-0 clock trips the cooldown gate — production unaffected). Bar: N≥30 & WR≥60% & avg≥+0.15%; analyst prediction on record: fails.

## 2026-07-05 (later) — INCIDENT + FIX: startup purge deleted PASS phantoms on redeploy
Operator caught it: phantom LONGs (PASS) vanished after the spike-phantom deploy while fade SHORTs survived. Root cause: database.py boot purge ran `DELETE FROM phantom_flips WHERE source_filter NOT IN (<hardcoded stale allowlist>)` on EVERY restart — the list never learned PASS:/SPIKE_REV (its own comment warned about drift; missed on both ships). 4 PASS phantoms deleted, unrecoverable (re-accrue ~2/day).
**Fix (operator invariant locked: phantoms die ONLY on /api/reset, NEVER on redeploy):** ① purge INVERTED to fail-safe `DELETE ... WHERE source_filter IN (models.PHANTOM_RETIRED_SOURCES)` — an unknown/future source ALWAYS survives a deploy even if every registry is forgotten (tested: fake unregistered source survives, legacy retired rows deleted, live sources intact); ② both lists single-sourced in models.py (keep for seeding, retired for cleaning) with disjointness asserted; ③ /api/reset full-kill untouched. Residual (documented): phantoms OPEN at the deploy instant are in-memory only and lost — sub-45-min ephemera, both directions equally.

## 2026-07-05 PM — PASS phantoms on the two decision-gated flip-SHORT blockers (+ revert-gate bug fix)
Operator asked "can we take more advantage of the blocking logs — post-tracking of price for X minutes?" Analysis: (a) the phantom tracker already IS post-block tracking and strictly better than price-at-X-min (replica exit is path-aware — a +0.8%@45min print that troughed −0.9% first was a stop-out, not a win); (b) the raw 19k block log is ~95% the same pair-episodes re-firing every 5s scan — the 30-min phantom cooldown is what makes N meaningful; (c) principle locked: a phantom source earns a slot only if an armed decision gate consumes its output — never instrument core signal-definition filters (EMA gap / RSI range etc., data nobody will act on).
While auditing coverage, found a REAL bug in the Jul-3 ship: the flip BTC-1h gate's locked revert ("→99 if blocked slope>0 flips ≥60% WR on N≥10 phantoms") had NO data surface — FLIP_SHORT_BTC1H_SLOPE was never seeded and wasn't in PHANTOM_KEEP_SOURCES. 172 blocks already went uncollected; the revert could never trigger.
Shipped (operator: "ship 1 and 2"): at the _flip_filters veto site in _maybe_open_flip, reasons FLIP_SHORT_BTC1H_SLOPE and FLIP_SHORT_REGIME now seed a SAME-direction PASS:<reason> phantom SHORT (mode='PASS', full entry_fields, flat replica exit). models.py PHANTOM_KEEP_SOURCES += both (auto-survives redeploys via the Jul-5 fail-safe purge). main.py source_specs += both (own tracker rows + pattern/regime sub-rows; regime cross-tab rides free; UI + both exports render rows generically — D12 satisfied with zero template changes).
Consumption: PASS:FLIP_SHORT_BTC1H_SLOPE feeds the existing Jul-3 revert gate verbatim. PASS:FLIP_SHORT_REGIME (bear≥80-era gate, #1 flip blocker at 290) is observation-only; watchlist bar to OPEN a relaxation discussion: N≥15 · WR≥55% · avg>+0.10%, then standard promotion gates + three-pillar. Explicitly NOT instrumented: PAIR_EMA_GAP_* / PAIR_RSI_RANGE / other core-signal blockers (no decision consumes them).

## 2026-07-05 PM (2) — SHIP: LONG_BTC1H_DEADBAND 0.05 (flat-hourly DOA block, operator-directed override)
Trigger: 07-05 batch went 1W/2L (RPL −$258 peaked +0.4434 just under the 0.45 arm; ETHFI −$190 died in 11min). Operator asked for a deep-dive of these unmatched-long losers vs the baseline's.
Method: all-dims W-vs-L comparison of fresh losers against baseline ML (34 trades, 28W/6L). The fresh report's flashy candidates ALL refuted by baseline (BTC-RSI-falling 80%WR / RSI 60-65 87% best band / pair-gap≥0.4 75% / BTC-ADX<25 86%) — the standing lesson: never trust a 3-trade table without the baseline cross-check.
Finding: BTC 1h slope DEAD-BAND. Baseline ML |1h|<0.05 = 7·43%WR·−0.19% vs pullback flank ≤−0.05 = 14·93%·+0.52 (the shipped 1hPullback 2× cell) and rising flank ≥+0.05 = 13·92%·+0.35. Both fresh losers in-zone (−0.016/+0.024), fresh winner outside (+0.097). 6 of 8 cross-era losers in-zone, 6 distinct pairs, monotone-worse toward 0 (<0.02 = 0%WR — a real mechanism shape, not a carved hole; monotonic in |slope|). Fisher p≈0.0096 baseline-only (~20 dims tested → suggestive, not conclusive). Zone-loser fingerprint: DOA (peaks 0.00/0.00/0.00/0.35/0.44/0.26). Theory: momentum longs are long CONTINUATION; flat hourly = measured absence of carry — the alt pop has no sponsor. 4th independent appearance of BTC-1h-slope-as-regime; the Jun-3 min-floor config comment had already predicted "the flat band" as the LONG loser zone.
Analyst recommended gated watchlist (N=9 < 30; Jun-26 the zone went 3/3 W — one window contradicts; ±0.05 in-sample). Operator: theory probed and confirmed ("it makes theoretical sense?"), then "ship it" — DISCIPLINE-OVERRIDE acknowledged, tighter-than-standard revert attached.
Shipped (full D11): config.py long_btc_1h_deadband=0.0 default + evidence; json 0.05; engine block after BTC_1H_SLOPE_MIN_GATE (LONG-only, None-check, fail-open on missing slope), counter LONG_BTC1H_DEADBAND; day-one revert surface — every block seeds PASS:LONG_BTC1H_DEADBAND phantom (keep-sources + main.py source_specs row; the Jul-3 no-surface bug not repeated); UI cyan row config-long-btc-1h-deadband + load/save + config report line. Allowed region with max gate: [−∞,−0.05] ∪ [+0.05,+0.15]. ~7% of long signals in-zone.
🔒 REVERTS: →0 if blocked phantoms ≥60% WR on N≥10 fresh, OR phantom cohort avg ≥+0.15% at N≥8.

## 2026-07-05 PM (3) — BASELINE RE-FROZEN v5 (deadband parity) + GVol≥0.85 gate explained
Per the screened-pool rule (re-screen on any filter change), long_btc_1h_deadband=0.05 added to screen_pool.py's MOM-long branch and the baseline re-frozen WITHOUT ingesting the live 3-trade batch (batch joins the pool later when saved, as usual). v5 anchors: ML 27·93%·+$2964 (was 34·82%·+$2708; −7 flat-zone trades 4L/3W −$256) · MS 17·76%·+$661 · FLIP 36·75%·+$825 · TOTAL 80·81%·+$4450. Asserts updated; memory + CURRENT_STATE pins updated.
SIDE FINDING (2×2 GVol × deadband): the registered GVol≥0.85 unmatched-long de-mux gate was a CONFOUND of the flat-1h zone — GVol≥0.85 raw 16·75%·+0.08% avg, but its damage sat in the 4-trade overlap cell (25%WR·−$354, PENGU/JTO/XLM all flat-zone); survivors post-deadband = 12·92%·+0.24%·+$748, well above the +0.10 de-mux trigger. Gate stays registered (pre-committed gates don't move) but is now expected to self-bury — forward GVol≥0.85 accrual happens under the deadband stack.

## 2026-07-05 PM (4) — REVERT: CHOPPY_FLAT flip block removed (evidence dissolved into mechanism gates) + v6 re-freeze
Trigger: PASS:FLIP_SHORT_REGIME phantoms (first weekend, 12·67%·+0.155; CHOP row 5·60%·+0.275) vs the Jun-23 live CHOP evidence — operator hypothesized "maybe the block filter we should use was other and not choppy flat."
Audit (mechanism-aware: relax ONLY the CHOP block, keep everything else incl. the Jul-3 1h gate, run the REAL _flip_filters over the raw pool's 30 historical CHOP flip-shorts, raw 47%·−$586 @1×): 17 blocked by FLIP_SHORT_BTC30_RISE, 11 by FLIP_SHORT_BTC1H_SLOPE (the worst slice: 36%·−0.47%·−$507), 1 by FLIP_SHORT_QUALITY, 1 ADMITTED (winner +0.19%). The CHOPPY_FLAT label was a PROXY for BTC-turning-up-under-the-fade — now gated directly, twice. Third label to dissolve into BTC-slope mechanics in 2 days (GVol≥0.85 ML de-mux, ML flat-band, now CHOP).
Burden-of-proof note (analyst initially said "don't touch yet", operator pushed, analyst conceded): N≥30 promotion gates exist for ADDING constraints; a filter whose own evidence base is refuted defaults to OFF — demanding N≥30 forward evidence to remove an unjustified constraint is backwards. Theory also flips: the Jun-23 "chop = no follow-through" story was fitted to trades that were actually slope>0 squeezes; an idiosyncratic alt-pump fade doesn't need BTC follow-through, only pump exhaustion.
Same-day sibling audits: HEALTHY_BULL (ADXd<0 slice) fully redundant (0 re-admits) BUT removal buys nothing (phantom avg +0.07 < ~0.27 friction) → KEPT as free backstop. ATR≥3 bear guard (FLIP_SHORT_HIATR) fully redundant → KEPT as free tail insurance. Bears never regime-blocked.
Shipped (config-only, fields long since D11-wired): flip_short_regime_block_any_adxd_regimes ""; flip_short_regime_block_regimes "HEALTHY_BULL". ⚠ new CHOP flips ride the 2× multiplier. Phantom note: PASS:FLIP_SHORT_REGIME keeps seeding for the remaining HB slice; the regime block fires FIRST in _flip_filters, so its phantom rows are NOT net-admissible counterfactuals (downstream gates unevaluated) — filter offline before citing.
Baseline re-frozen v6: FLIP 36·+$825 → 37·+$833 (STG 06-18 re-admitted); ML 27/$2964, MS 17/$661 unchanged; TOTAL 81·81%·+$4458. Anchors asserted.
🔒 RE-BLOCK GATE: re-add CHOPPY_FLAT if live CHOP flip-shorts net-negative (de-muxed 1×) or ≤45% WR on N≥8 fresh (live Flip Trades × BTC-Regime table).

## 2026-07-05 PM (5) — Pre-purge PASS-phantom prior recovered + registered
The operator's 07-05 10:56 report turned out to be the only surviving record of the 4 PASS:LONG_UNMATCHED_ONLY phantoms deleted by the purge bug (fixed f14d584): W2 3·67%·+0.065 · one C2+W1+W4+W5 multi-match 1·0%·−0.70 (H.BULL) · regimes S.BULL 3·67%·+0.07 / H.BULL 1·0%·−0.70. Registered in CURRENT_STATE as a MANUAL prior to add to the tracker N on every matched-long re-enable gate evaluation (DB rows unrecoverable; tracker will otherwise understate the W-cell tallies). Combined with the post-fix C6 +0.725: matched-long PASS record = N=5·60%·≈+0.04% avg — blocked matched longs ≈ breakeven under today's stack (and their fade is also ≈ flat: the signal currently chops, neither runs nor reverses). No gate near firing; W2 is the most active cell at 3·67%·+0.065 (below the +0.15 avg bar).

## 2026-07-05 PM (6) — 07-04/05 batch saved + pool appended + baseline v7
Batch saved: reports/batch_2026-07-04_to_07-05.csv (3 trades: PUMP LONG W +$208 · RPL LONG L −$258 · ETHFI LONG L −$190; the RPL/ETHFI losses are what motivated the same-day LONG_BTC1H_DEADBAND ship). Blank reports/batch_2026-07-04_to_07-05_report.txt created for the operator's pasted UI report. Pool appended with dedup (opened_at,pair,direction): 372→375 rows, now spans 06-16..07-05 (.pre0705_bak backup kept, untracked per convention). Baseline re-frozen v7 under the LATEST stack (operator-directed): PUMP survives (UNMATCHED, 1h +0.097 outside the dead-band) → ML 28·93%·+$3172; RPL (1h −0.016) and ETHFI (+0.024) are deadband-screened — the screen now agrees with what the live gate would do. MS 17/$661, FLIP 37/$833 unchanged. TOTAL 82·82%·+$4666. Anchors asserted; CURRENT_STATE pin + memory updated.

## 2026-07-06 — TRIPLE SHIP: flip de-mux 1× + SLOPEUP live admit + arm-level shadows (+ v8 re-freeze)
Context: 07-06 batch (5 trades, −$174, tape flipped BEARISH) fired the Jul-3 flip-1h-gate revert on its locked terms (PASS:FLIP_SHORT_BTC1H_SLOPE 18·78%WR·+0.268 ≥ 60%/N≥10; with pre-reset prior 21·71%). Analyst caveats on record: (a) phantoms NOT net-admissible — code order check showed 6 blockers run AFTER the 1h gate (HIATR/RSI_MIN/QUALITY/BEAR_MIN/ADXD/PAIR_GAP + FAN sub-gates), so the 78% overcounts; (b) avg +0.268 ≈ live friction; (c) 13/18 phantoms in one bear session. Operator proposed the graduated resolution; analyst endorsed.
① flip_entry_sources FAN_RATIO_GATE:2.0→1.0 (operator-directed full de-mux at 0W/2L, −$393 — below the N≥5 verdict gate, acknowledged; analyst had recommended 1.5× per the locked staging protocol, operator chose 1×; multiplier can re-earn via W-gate + 1.5×-first staging on fresh data).
② flip_short_btc_1h_slope_admit_mult=1.0 (new field, full D11): slope>max flip-shorts ADMIT live with cell mult capped at 1.0, tag B1H_SLOPEUP (own Multiplier-Cell + Flip×Regime row; _flip_filters falls through, cap+tag at the open site; screen_pool picks the admit up automatically). Phantom seeding for the source stops (no block → live log is the surface). This beats the net-admissible-tracker alternative: downstream gates run for real, friction measured with real fills. 🔒 RE-BLOCK →0 at ≤50% WR or net-negative on N≥10 live; PROMOTE to full size at ≥65%/avg≥+0.15/N≥20.
③ arm035/arm040 leash shadows (arm the 0.25-trail at peak≥0.35/0.40): tracked from tick 1 on EVERY trade (before the 0.45 leash guard — new xexits section in _leash_update/_leash_finalize), 4 Order columns + DB migration + Leash-Shadow-table rows; unit-tested (rescue path fires trailing at peak−0.25 on a 0.38-peaker; unarmed variant falls back to actual). Closes the recurring arm-lowering question (asked 3×: flips 0.45→0.40/0.35, longs post-RPL, today) — decision at N≥30 all-trade shadows, both sides measured; baseline CSV counterfactuals were path-blind (zone [0.35,0.45) N=2 in 82; armed cohort 70 @ 90-100% WR).
v8 re-freeze: FLIP 37/$833 → 63/$356 (26 historical slope>0 flips re-enter at 1×: the SLOPEUP prior = 26·50%·−$478). SPLIT PIN — judge flips on core 37·76%·+$833 vs SLOPEUP separately; blended 63-row total is NOT the sleeve's health metric. ML 28/$3172, MS 17/$661, TOTAL 108·75%·+$4189. Split asserted in screen_pool.

## 2026-07-06 (2) — 07-06 batch saved + pool 380 + baseline v9 + pre-reset prior #2
Batch saved: reports/batch_2026-07-06.csv (5 trades: AAVE L W +$263 · ADA C1 S W +$91 · GIGGLE W4 S L −$135 · KAITO flip L −$246@2× · AAVE flip L −$146@2×) + blank batch_2026-07-06_report.txt for the operator's paste. Pool 375→380 (dedup key). Re-frozen v9 under the CURRENT stack: ML 29·93%·+$3435 · MS 18·72%·+$526 · FLIP 65·65%·+$159 (SPLIT: core 39·74%·+$637 / SLOPEUP 26·50%·−$478) · TOTAL 112·73%·+$4120.
De-mux question (operator asked): NO manual action — pnl_current() already divides any multiplied FLIP row by cell_multiplier, so the two 2× flip fills entered at 1× (KAITO −$123, AAVE −$73) matching the Jul-6 de-mux ship. The rule generalizes; nothing to change.
Parity note: ADA C1 short (+$91) traded LIVE but is WEAKCAP-screened on stamped entry fields (rng 10.3 / ATR 0.33 / pADX 26.1 all under thresholds) — boundary sampling difference between signal-time and fill-time values; the screen stays the stricter consistent ruler (conservative: excluded a winner).
Pre-reset phantom prior #2 registered (operator resetting for a fresh batch): PASS:FLIP_SHORT_REGIME 13·54%·+0.041 · PASS:LONG_BTC1H_DEADBAND 2·50%·−0.247 · SPIKE_REV 2·0% (4-for-4 with prior#1) · PASS:FLIP_SHORT_BTC1H_SLOPE 18·78% RESOLVED into the SLOPEUP admit.

## 2026-07-06 (3) — WATCHLIST: mom-short deep-gap floor (GIGGLE post-mortem)
GIGGLE W4-short DOA (−1.21% in 109s) → operator asked what W4 is and whether to block it. W4-SHORT = rng 25-60 × pair gap ≤−0.10 × ADXΔ≥0 ("short the bounce in a downtrend") — theory-sound, NOT blocked; its gap condition has NO FLOOR, so GIGGLE's −1.72 capitulation hole wore the pullback label. True dimension: pair 13-50 gap ≤−1.0 mom-shorts = 3·33%·−$224 (BEL W +0.41 / MANTA / GIGGLE; 3 dates/3 pairs) vs −1.0..−0.6 = 2·100%·+$164; distribution EMPTY −1.5..−1.0 → −1.0 threshold sits in clean space (edge cliff, monotone). Ship impact: MS +$526→+$750. N=3 ≪ bar → registered watchlist: SHIP momentum_short_pair_gap_min=−1.0 at N≥10 & WR≤40%; kill at ≥55% WR on N≥8. Also noted: flip dead-pair-vol <0.35 (KAITO, N=1 first-ever; sweet spot 0.35-0.60 = 90% WR) — watch only.

## 2026-07-06 (4) — SHIP: MOMENTUM_SHORT_DEEPGAP −1.0 (watchlist→live same day, operator-directed)
Operator overrode the same-day watchlist parking: "if it makes a lot of theoretical sense... maybe we should land the block now — waiting means losing real money when it lands live." Analyst AGREED and documented why this clears our own rules better than the initial 'wait' stance: ① the locked ships rule has a ≥3-sample direction-consistent path (3 trades / 3 dates / 3 pairs / −$224 net, lone winner +0.41 small); ② mechanism precedent: btc_1h_slope_min_short (−0.60) is the SAME don't-short-into-the-crash mechanism at BTC level, shipped at N=4; ③ asymmetry: zone ≈1.6% of shorts → unprotected accrual to N≥10 costs ~$130-260/occurrence to protect a ~$48-class winner; ④ the day-one PASS phantom converts 'wait unprotected' into 'wait protected' — the same N≥10 evidence keeps accruing via PASS:MOMENTUM_SHORT_DEEPGAP.
Shipped (full D11): config momentum_short_pair_gap_min=0.0 default/−1.0 json; engine block at the PAIR_TREND_FILTER site (gap already computed there; SHORT-only, momentum-only — flips untouched; 0=off sentinel None-checked); counter MOMENTUM_SHORT_DEEPGAP; PASS phantom (keep-sources + source_specs row); UI cyan input next to the pair-trend gap field + load/save; screen_pool parity. Threshold −1.0 sits in the EMPTY −1.5..−1.0 band (all 3 hole-trades ≤−1.5; mild pullback −1.0..−0.6 = 2·100%·+$164 untouched).
Baseline v10: MS 18·72%·+$526 → 15·80%·+$750; ML 29/$3435, FLIP 65/$159 (core 39/$637 + SLOPEUP 26/−$478) unchanged; TOTAL 109·74%·+$4344.
🔒 TIGHT REVERT: →0 if PASS:MOMENTUM_SHORT_DEEPGAP phantoms ≥55% WR on N≥8 fresh.

## 2026-07-06 (5) — GATE REGISTERED: UNMATCHED-long 2.0×→2.5× step (+ lev-stack declined)
Operator proposed 2.5× (and 2.5×+1.5× lev) on the 93%-WR unmatched-long cell. Lev 1.5× = hard no: locked BE-compat gate fails at 50% (cell losses HYPE peak +0.07 / ME +0.24; need ≥60% peaking ≥+0.20 so caps bound the tail); 30× turns a widened stop into −36% of position margin — ruin-risk binding constraint. Size 2.5× declined NOW (no earned tier above 2.0 in the locked ladder; equal-split 2.5× ≈ half the book at 20× per idea; fresh 2× uplift ≈ breakeven on 4 fires) but REGISTERED as a pre-committed step gate: 2.5× after 50 fresh 2.0× fires (from the 07-06 reset) with ★ WORKING (WR≥70% + positive Δ$ vs 1×) and no pair ≥40% of cell P&L; BE-compat re-checked inside every evaluation; leverage stays 1.0× until ≥60% clears. Full-stack audit of the 2 cell losers on record: HYPE = DOA in the 93% pullback flank (no filter can cover, BTC-RSI 67.3 stamped vs ≤65 live = boundary sampling); ME = clean pass everywhere, 1h +0.10 just above the deadband edge (widening the band = threshold-chasing into the 92% flank, refused). Losers = cohort price, not leaks.

## 2026-07-06 (6) — REALTIME LONG runner ATR-floor + BE-lock gate (USELESS post-mortem)
USELESS LONG armed +0.63, closed RUNNER_TRAIL L1 at −0.47 (operator: "how can this happen?"). Diagnosis in two parts, with a corrected first draft: analyst initially computed the floor with the config DEFAULTS (N=0.5 + ratchet on → floor +0.11, "−$260 latency tax across 3 trades") — operator corrected: live LONG config is N=1.0 with ratchet OFF → true floor −0.40, so the red exit was mostly BY DESIGN; the real latency tax is ~$56 (PUMP −$40, USELESS −$15, AAVE ~0), caused by the armed-runner floor living only in the slow monitor loop (~3-4 min/50 pairs) while arming suppresses the 1s tight trail.
Shipped: realtime evaluation of the LONG runner ATR-floor in the tick path (mirror of Jun-16 flip-short Fix A; same formula/config fields incl. cap+ratchet reads; close_position "RUNNER_TRAIL"; _closing_in_progress guard; monitor path kept as backstop; fail-open). Momentum-SHORT K-trail left on monitor cadence (small observed lag; needs peak-stretch state in realtime — separate change if its lag shows up).
Gate registered: re-enable runner_trail_be_ratchet_enabled (LONG) if the lock-aware atr10 shadow (max(peak−1×ATR, +0.10) = the exact lock-ON counterfactual) beats actual cumulatively on N≥8 fresh armed longs. Tally 3/8: PUMP +0.13, AAVE 0.00, USELESS ~+0.57.
Also resolved: "empty shadow columns" false alarm — shadows write at post-exit window end (~45 min); the export was 65s after close.

## 2026-07-06 (7) — REVERT: SLOPEUP admit re-blocked <1 day after ship (operator fundamentals objection) + v11
Operator: "we block it with fundaments, now we unblock it 1 day later? doesn't make sense." Analyst agreed and owned the error: the Jul-3 block had all three pillars (two-period direction-consistent data: baseline slope>0 17·65%·−$73 / fresh 9·33%·−$405; recovery-squeeze mechanism; regime consistency). The un-block rested on one flawed pillar — 18 phantoms with 13 from a single bear Monday, NOT net-admissible (6 downstream gates unevaluated), avg ≈ friction — obeyed because the locked revert gate fired on its LETTER; the letter was under-specified (analyst's own spec, missing multi-window + net-admissibility clauses). Lesson locked: a fired gate discovered to be spec-defective gets its SPEC fixed transparently, not obeyed into contradicting three-pillar evidence (distinct from bar-moving: this corrects WHAT was measured, not the threshold).
Shipped: flip_short_btc_1h_slope_admit_mult 1.0→0.0 (hard block restored; engine machinery + UI kept, dormant; PASS phantom seeding resumes automatically on the block path). Live exposure during the ~22h probation: any B1H_SLOPEUP-tagged fills to be counted in the next batch review wherever they land. Baseline re-frozen v11: FLIP returns to core-only 39·72%·+$637; ML 29/$3435, MS 15/$750; TOTAL 83·81%·+$4822.
🔒 REWRITTEN REVERT GATE (replaces the fired Jul-3 one): re-open at 1×-probation ONLY when NET-ADMISSIBLE phantoms (offline join through the downstream flip gates) hit ≥60% WR on N≥15 spanning ≥3 distinct dates including ≥1 bull-tape window, avg ≥+0.30%.

## 2026-07-06 (8) — SHIP: W2 re-enable, 1h-rising conditioned (first matched-long cell back since Jun-9)
Trigger: 07-06 phantom day — PASS:LONG_UNMATCHED_ONLY 10·100%·+0.52 with the perfect 0/10 fade mirror; operator asked for the historical W-cell division and then directed the live ship.
Historical dive (dedupe_pool_FULL, 692 momentum longs Apr28–Jun13, OLD exit stack = refute-only): W2 unconditional 87·54%·−0.09 (block was justified) but splits hard on BTC 1h slope: rising ≥+0.05 = 29·72% / flat = 20·45% / pullback ≤−0.05 = 14·14%·−0.55 (7 DOA). Era decay 63→56→37% = regime fingerprint. W6 = the MIRROR (pullback 55·60% / rising 21·43%) — coherent from definitions: W2 needs pair gap ≥+0.10 (continuation, engine must run), W6 needs pair NOT extended (laggard catch-up, wants the dip). W4: no 1h structure (48% everywhere) — unconditioned forward gate only. C6 (climax chase): 55·47%·−0.21, worst when rising (39%) — refuted + no theory → stays blocked.
Shipped (full D11): long_w2_reenable_1h_min (default 99=off, json 0.05); engine carve-out ahead of the LONG_UNMATCHED_ONLY block (W2 AND no C co-match AND 1h≥0.05 AND fail-closed on missing 1h → admit; everything else still blocks + phantoms); UI cyan row under the unmatched-longs toggle + load/save + config report line. Sizing = W2 pattern cell 1× (not UNMATCHED 2×); tracking = Pattern Cell Ship Performance W2 LONG row. Screen anchors UNCHANGED (v11 re-verified — W2 trades are their own cell outside the ML anchor).
⚠ Operator override: conditioned phantom N≈10 < registered N≥20. 🔒 TIGHT REVERT: →99 if live W2-rising longs ≤50% WR or net-negative on N≥8. The W6/W4/C6 re-enable map registered in CURRENT_STATE.

## 2026-07-06 (9) — SHIP: W6 re-enable, 2D-conditioned (pullback ∧ thrust; second matched cell back)
Operator pushed past the single-variable read ("maybe we missed other variables") → head-to-head on identical rows (dedupe_pool_FULL, W6 longs with 1h data N=96, era split 05-26): pullback-alone (the registered map condition) FAILED the era test (eraA 14·100% / eraB 41·46% — would have been a SLOPEUP-class mistake if shipped); thrust alone 65% consistent; **pullback ∧ thrust = 23·78%WR, the only both-era W6 cell (eraA 8·100%·+0.43 / eraB 15·67%)**. Contrast locked in: within W2-rising, thrust adds NOTHING (77% without vs 69% with — W2's signature already includes pair momentum via gap≥+0.10); within W6-pullback, thrust is THE discriminator (78% with vs 47%/eraB-35% without — W6's signature excludes pair extension, so laggard initiative must be verified separately). Same sweep also re-validated the shipped W2 filter as dominant on identical rows (the sweep's thrust-first ranking was an artifact of the missing-1h rows + thrust/rising correlation).
Shipped (full D11): long_w6_reenable_1h_max=−0.05 + long_w6_reenable_stretch_min=0.31 (defaults 99/0.31 = off); engine W6 branch after the W2 admit (W6 ∧ no C co-match ∧ 1h≤−0.05 ∧ stretch≥0.31 → admit; fail-closed); UI second cyan row + load/save + config report line. Cell 1× via the W6 pattern cell; screen anchors unchanged (own cell, re-verified).
⚠ Acknowledged: weakest override of the sequence (search-derived 2D, eraB $≈breakeven under old exits, phantom N=2; analyst had recommended holding to the phantom gate — operator directed the ship; risk bounded ≈$80/loss at 1× × the N≥8 revert). 🔒 TIGHT REVERT: →99 if live W6 cohort ≤50% WR or net-negative on N≥8.

## 2026-07-07 — Phantom net-admissibility tooling (export endpoint + screen_phantoms.py); blanket B1H regime re-enable REFUSED
**Context:** Operator proposed re-enabling FLIP_SHORT_BTC1H_SLOPE (slope-up flip-shorts) for HEALTHY_BEAR / STRONG_BEAR / CHOPPY_FLAT / BEAR_EXHAUSTED off the Phantom Flip Tracker (current window 23·65%·+4.66; H.BEAR 9·56%, S.BEAR 5·80%·+0.62, CHOP 4·100%, BEAR_EXH 1·100%).
**Verdict: NO — none of the four cells clears the locked per-cell carve-out gate (net-admissible N≥10 · WR≥60% · avg≥+0.30 · Σ>0 · ≥3 dates).** Reasons: ① phantoms are FIRST-BLOCK seeded (6 gates downstream of B1H) → tracker N over-counts what a re-enable admits; ② H.BEAR fails avg (+0.15 ≈ friction) AND its genuine net-admissible history is the worst cell (16·44%·−$402); ③ CHOP has no historical would-trade twin (all 19 die downstream) — existence unproven; ④ BEAR_EXH N=1; ⑤ S.BEAR closest (WR+avg pass) but N≈7<10 and not net-admissible-counted. Re-opening on first-block phantom evidence <2 days after the SLOPEUP re-block would repeat the exact operator-caught flip-flop.
**Shipped (tooling, no strategy change):** ⓐ `/api/phantom-flips/export.csv` (main.py, mirrors orders export — full PhantomFlip schema, auto-includes future columns) + teal UI button; operational rule: download before every reset (phantoms deleted on reset; priors #1/#2 row-level fields permanently lost → net-admissible gate counts restart at the 07-06 window). ⓑ `scripts/screen_phantoms.py` — offline net-admissibility join: replays real `_flip_filters` with only the phantom's own gate disabled (supports PASS:FLIP_SHORT_BTC1H_SLOPE + PASS:FLIP_SHORT_REGIME), per-regime net-admissible table + downstream-blocker census + automated carve-out gate check; dedups (entry_at, pair, source) so stacked exports are safe. Smoke-tested (synthetic rows route to FLIP_SHORT_QUALITY / FLIP_SHORT_PAIR_GAP / admissible correctly; 1h gate confirmed off during replay, restored after).
**Consumption:** the rewritten SLOPEUP revert gate and both sub-cell carve-out gates now have their required surface — run screen_phantoms.py on each phantom export at batch review; the gate check line is the decision.

## 2026-07-07 (later) — First screen_phantoms.py run: the B1H phantom tracker's winning cells DISSOLVE under net-admissibility
Operator downloaded the first phantoms CSV (new export endpoint; saved to reports/phantoms_2026-07-06_reset_window_asof_07-07_1211.csv — the first permanent row-level phantom prior). Join on the 23 closed PASS:FLIP_SHORT_BTC1H_SLOPE phantoms, real _flip_filters with only the 1h gate off:
- **NET-ADMISSIBLE: 3/23 · 33% WR · avg −0.21 · Σ −0.64** (S.BEAR 2·50%·+0.03 / H.BULL 1·0%·−0.70).
- Tracker cells vs reality: S.BEAR 5·80%·+0.62 → 2 net-adm breakeven; CHOP 4·100% → 0 (all die downstream — exactly the historical N=0 twin's prediction); BEAR_EXH 1·100% → 0.
- Downstream blocker census: RSI_MIN 7 (86%·+4.14Σ — most of the tracker's headline profit sits behind the pair-RSI floor, which has its own cross-batch negative-zone basis; observation-only), QUALITY 5, ADXD 4, FAN_PAIR_ADX 2, PAIR_GAP 1, FAN_STRETCH 1.
- Carve-out gate check: S.BEAR ❌ N 2/10 · WR 50% · avg +0.03 · 1 date; nothing else close.
**Consequence:** the 07-07 refusal of the blanket regime re-enable is now empirically confirmed, not just methodological — un-blocking those regimes would have admitted 3 trades for a net loss. All future SLOPEUP/carve-out gate evaluations count ONLY net-admissible phantoms via this script; the raw tracker table is a seeding surface, not a decision surface.

## 2026-07-07 (evening) — Flip-short SL grid: tightening REFUTED at every level; ATR-widened −1.20 floor is load-bearing
Operator asked whether lowering the flip SL was ever evaluated — it never had been (the high-ATR-tail resolution covered longs/global). Grid on the 39 v11 FLIP survivors (trough-based counterfactual: trough ≤ SL → loss at SL, else actual):
- Actual (ATR-widened ×1.5, floor −1.20): **+6.91% Σ, 72% WR**
- Hard −0.70 (widening removed): −0.60% (7 winners chopped) · −0.60: −2.21% (10) · −0.50: −1.51% (11) · −0.40: −3.26% (14)
- Winners' trough distribution: 11/28 dip <−0.50, 7 <−0.70, 3 <−1.00 before paying — whipsaw-through-deep-drawdown is how flip WINNERS win, not just how losers lose.
- Other direction (same day, AM): no-SL/hold-to-EMA13 shadows on the pool's 11 flip losers = 6 saved / 5 worse, avg Δ −0.60pp → widening/removing also refuted (runaway-squeeze tail).
**Verdict: the current ATR-widened stop is the only defensible width for flips; RIF/DEXE (07-07) died at the −1.20 floor = irreducible left tail of a +EV sleeve (mirrors the 06-29 high-ATR resolution). Remaining loser levers: 20-trade sleeve verdict (BUILDING 3/20), pair blacklist, sizing (1× de-mux already in effect). Do not re-litigate SL width without ≥30 fresh flips contradicting.**

## 2026-07-08 — SHIP: flip BTC trend-gap depth gate (flip_short_btc_trend_gap_min=-0.22) + pre-committed −0.10..0 multiplier twin; baseline v12
**Origin:** operator demanded an exhaustive every-dimension flip winner/loser analysis ("act as a real quant... not just a few variables"). Full sweep: 31 stored entry dimensions × fresh flips (8) × baseline flips (39), sign-consistency required in BOTH windows. 26 flat, 2 invert (pair gap, BTC dist-EMA13), 3 consistent (BTC trend gap, bull-breadth, pair 24h vol). Bull-breadth and pair-vol refuted as filters (baseline cohorts breakeven, not negative) → watch-only. **BTC trend gap (EMA13-50 depth) = the ship: MONOTONE across 5 baseline buckets (≤−0.30 = 4·25%·−0.33 → −0.10..0 = 8·100%·+0.53), fresh window reproduces the shape.**
**Complementarity test (operator question — substitute for the 1h gate?):** 2×2 on the 65-flip common universe (full stack, 1h off): only-1h 24·54%·−$409 / only-gap 15·47%·−$146 / both 2 / neither 24·88%·+$783. Swap costs −$263 vs current; both on = +$146 in-sample over current. Corr(gap,1h slope)=+0.49; gate cuts WITHIN regimes (11/18 S.BEAR vs 3/17 H.BEAR) — explains the fresh-window "regime inversion" (S.BEAR wasn't the variable, depth was). **Verdict: complementary; the 1h ship was correct and stays.**
**Threshold:** sweep −0.35..−0.10. Plateau −0.25..−0.20 (fresh outcome identical across it — empty value-space −0.32..−0.20 in fresh). −0.20 REJECTED: knife-edge (fresh DEXE loser −0.1995 vs baseline JTO winner −0.1994). −0.22 = best kept-$ ($881) AND worst-quality blocked cohort (42%WR) with clean space both sides. Below −0.18 the filter degrades into a volume cut (blocked cohort → 53-62% WR, Σ→0/positive) — chasing RIF/DEXE (gaps −0.13/−0.20, historically the healthy band) is this-week overfit, refused.
**Impact:** batch: flip sleeve −$362 → −$189, total −$346 → −$173 (blocks TLM −$208/WLD −$74/VANRY +$72/LTC +$36 — halves the loss, doesn't flip it; stated to operator pre-ship). Baseline: FLIP 39·72%·+$637 → **27·85%·+$881**; TOTAL → **71·87%·+$5066 (v12, anchors asserted, re-frozen; pre-v12 backup SCREENED_BASELINE.csv.prev12_bak)**. Haircut expectation ~+$75-100/window.
**Discipline status:** N=16 < 30 locked filter gate (WR 44% > 40, avg −0.13 > −0.20) → operator-directed near-gate ship, same evidence class as the Jul-3 1h gate (N=26): two-window monotone + mechanism + diffuse pairs + multi-date. Protections: day-one PASS:FLIP_SHORT_BTC_TRENDGAP phantoms (keep-source + purge-safe + report rows + NEW PhantomFlip column entry_btc_trend_gap_pct with migration so future joins can evaluate this gate downstream) · 🔒 revert →0 at NET-ADMISSIBLE blocked ≥60% WR N≥10 (screen_phantoms.py has the gate-off mapping).
**Multiplier twin registered (operator-directed):** BTC gap −0.10..0 flip cell = 10/10 combined (+$455 base). 🔒 promotion: fresh N≥20 · WR≥70 · avg≥+0.10 · $>0 → 1.5× staging → 2.0× after +50; BE-compat when losses exist. Not shipped now: N=10 ≪ 30, flip multiplier in re-earn penalty box, sleeve mid-verdict (scorecard 8/20) — one change at a time.
**Files:** trading_engine (_flip_filters gate + _ff_in btc_trend_gap + PASS seed + phantom field stamp), config.py, trading_config.json, models.py (keep-source + column), database.py (migration), main.py (report spec), index.html (row+load+save, grep-verified 3), screen_pool.py (flip_ind + v12 anchors), screen_phantoms.py (field + gate-off mapping).

## 2026-07-08 (same session) — SHIP: TG_SHALLOW 2× multiplier cell (operator-directed double override)
Operator shipped the −0.10..0 multiplier twin same-day at 2× invest, skipping BOTH the N≥20 promotion gate and the locked 1.5×-first staging. Rationale (verbatim intent): "Flip Shorts else is still too bad, this batch moves to +16, that doesn't scale... we found a lot of filters together that confound in 10 trades with 100% win rate, I want to test it now." Double override acknowledged transparently per the discipline rule; carries a TIGHTER-than-standard revert.
**Spec:** `flip_short_tg_shallow_mult=2.0`, `flip_short_tg_shallow_min=-0.10` — flip-SHORT with BTC EMA13-50 gap in [−0.10, 0) takes 2× invest (lev 1×). Cell block in `_maybe_open_flip` between the QS cell and the B1H_SLOPEUP cap (cap ordering preserved: a future slope-up admit still caps the mult). Tag `[TG_SHALLOW]` → own row in 💰 Multiplier Cell Performance — SHORTs with automatic verdicts.
**Evidence:** the monotone gradient's top bucket = the sleeve's edge core: baseline 8·100%·+0.53·+$455, fresh 2·100% (SUI/DEXE) → 10/10 combined. N=10 ≪ 30; a 10/10 cell WILL regress — the question the live test answers is to what.
**🔒 REVERTS:** ✗ HARMFUL (cell net-negative on N≥5 fresh fires) → 1.0× · ⚠ DRAG → 1.5× · BE-compat (≥60% of losses peak ≥+0.20) checked at ≥3 cell losses · if the flip scorecard (8/20) verdicts ✗ on the sleeve, the cell reverts with it.
**Files:** trading_engine (_maybe_open_flip cell block), config.py (2 fields), trading_config.json, index.html (cyan row: mult + zone-min inputs, load/save, grep-verified 3+3). Screen parity: none needed — sizing not filtering; screen_pool's pnl_current already de-muxes FLIP mult>1 rows, so v12 anchors stand.

## 2026-07-08 PM — BATCH SAVE (07-06..08, 20 trades, −$306) + pool append + baseline v13 + pre-reset phantom prior #4
Batch archived (reports/batch_2026-07-06_to_07-08.csv + .txt). Pool 380→400 (dedup clean). v13 anchors: ML 35·91%·+$3482 / MS 19·79%·+$794 / FLIP 31·81%·+$692 / TOTAL 85·85%·+$4969. Note FLIP anchor DROPS v12 $881→$692: the batch's surviving flips (RIF/DEXE±/SUI/LIT after the trend-gap screen removes TLM/WLD/VANRY/LTC) net −$189 — the first fresh window is inside the new gate's expectation (halved loss, not green). Batch story: ML 7·71%·−$29 (UNMATCHED 2x ★ WORKING +$47; W2 cell 0/1 −$76), MS 4·75%·+$45, FLIP 9·56%·−$322 (TLM −$208 one-second gap = the tail). Phantom prior #4 = first ROW-LEVEL prior (135 rows saved; REGIME gate net-adm 0/41 → relaxation closed; deadband 7/10@86% band-edge caveat; W2 falling-flank inversion watch). Operator resetting after this save.

## 2026-07-08 PM — SHIP: range_position_max_long 97.5→95.0 (AGLD post-mortem; zero-cost cut, operator-directed at N=1)
AGLD (first trade of the fresh batch, −$257 at the ATR-widened stop on the 2× unmatched cell) was the FIRST unmatched long above range-pos 95 in the screened history. Baseline audit: 95–97.5 sliver N=0 across all 35 v13 unmatched longs (buckets 75-95 run 90-93% WR); the sliver's only-ever trade is the loser itself. Tightening the ceiling forfeits $0 of historical profit — shipped on the free-insurance property with the N=1 override acknowledged. Config-only (existing filter PAIR_RANGE_POSITION_MAX + UI). 🔒 REVERT →97.5 if blocked 95-97.5 longs run ≥60% WR on N≥8 fresh. ATR/widened-stop side of the AGLD loss explicitly NOT touched (Jun-29 resolution: accepted variance).

## 2026-07-10 — MOM-LONG QUAD SHIP (weakcap bugfix + stretch ceiling + PVR de-mux + W6 off) + loss-mechanism taxonomy + baseline v14

**Context.** Batch 07-08..10 (9 trades): mom-longs 8 = 3W/5L −$394 (TIA/PUMP/AVAX W; LDO −285, UNI −160, TAC W2 −132, LIT −112, ADA −75 manual) + NEAR mom-short −$73. Multi-day loser deep-dive across pair, 2D, BTC/market and trade-dynamics dimensions.

**Findings (the taxonomy).** ML outcomes are BIMODAL: all 35 pool winners peaked ≥+0.46% in-trade; 4/7 losers peaked ≤+0.07 (DOA). Three loss modes: (1) FALSE IMPULSE (PYTH/UNI/ADA/HYPE) — irreducible at entry (MARGINAL flag shared by 16/35 winners +$1963; stretch floor impossible, winners at 0.008; DOA clock-kill refuted: ~$1,900 of winners still <+0.10 at min 15 vs ~$300 saved — the wide-SL marination that pays winners must tolerate the fakes, ≈12% premium on gross edge); (2) STALE/CROWDED CHASE (LDO/ME/TAC + HYPE/PYTH on the crowding axis) — measurable at entry, two mechanisms shipped below; (3) MACRO BREAK (UNI/LIT/ADA opened within 22s at 17:02 07-10, BTC flipped bear mid-hold; every BTC-context entry gate refuted — the weak-BTC context is historically 70% WR net-positive; the entry-stagger idea was tested on the pool and REFUTED, −$280, operator also rejected on principle).

**Ships.**
- ⓪ WEAKCAP KEY BUGFIX (trading_engine.py): filter read indicators['atr_pct'] which never existed ('atr' absolute is the real key) → fail-open → ZERO fires since the Jun-28 ship; NEAR met all three conditions and leaked. Now computes pct from atr/price. Screen had the correct port all along (parity was engine-side). Restores a locked ship — no new gate.
- ① ema5_stretch_max_long 0→0.35 (JSON; enforcement pre-existed in indicators.py, UI pre-wired). Zero-cost: no pool winner ever above 0.34. Blocks LDO/TAC class. REVERT →0 if blocked >0.35 longs ≥60% WR on N≥8.
- ② long_unmatched_mult_pvr_max = 0.90 (NEW field, full D11: config+json+engine de-mux beside the C1 breadth de-mux+UI row in the Unmatched panel+load/save+report line; screen pnl_current parity). UNMATCHED 2×→1× when PVR≥0.90. ✗ HARMFUL sub-cell (zone 10·60%·net-neg at both sizings vs <0.90 = 29W/3L). REVERT →0 if fresh zone ≥70% WR net-positive at N≥8. The PVR≥0.90 HARD BLOCK variant was analysed and retired (kills 6 winners +$303; N=10/WR60% fails the locked filter gate).
- ③ long_w6_reenable_1h_max → 99 (W6 re-enable OFF): 0 fires ever; stretch ceiling kills its 0.31+ window. W2 KEPT until its locked N≥8 gate (0W/2L, both stretch-confounded — the ceiling blocks that class, clean measurement starts now).

**Baseline v14 (pool 409 rows 06-16..07-10; anchors asserted + frozen):** ML 41·85%·+$3550 · MS 19·79%·+$794 (NEAR weakcap-screened) · FLIP 31·81%·+$692 · TOTAL 91·82%·+$5037. Batch counterfactual under the new stack: mom-longs 8→6 · 50% · −$394→+$24.

**Refuted this session (do NOT re-derive):** ML entry filters for the false-impulse mode (all axes have winner flanks); ADXΔ floors (kill JTO +$478 etc.); 2D quadrants (190 pairs tested — PVR dominated every combo, refinements were fitted at loser extremes); DOA time-boxed kill; BTC-context gates for 07-10 (tg<0 & slope<0.05 kills 7W +$1276); weak-BTC de-mux (context net-positive at 2×); entry stagger (pool −$280); MS pair-RSI/ATR floors for NEAR (knife-edge vs winner minimums — superseded by the weakcap fix, which IS the intended protection).

**Files.** batch_2026-07-08_to_07-10.csv (+.txt shell), phantoms_2026-07-08_reset_window_asof_07-10_1840/2032.csv (prior #5), pool → 409 rows (.pre0710_bak), SCREENED_BASELINE.csv v14 (91 rows), screen_pool.py (stretch+demux parity, v14 anchors), trading_engine.py (weakcap fix + PVR de-mux), config.py (+1 field, comments), trading_config.json (3 values), index.html (demux row + handlers + report line + W6 note).

## 2026-07-10 (addendum) — $-per-day scaling line added to the batch-review routine + 2.5× gate re-scoped

Operator reaction to the 07-08..10 batch (+$24/2d under the new stack): "sounds really bad." Diagnosis stands (bad tape: BTC mid-bull bear flip killed 3 correlated positions; small size + low throughput cap $/day; pool expectancy ≈ +$210/day at current stack) — but the scaling path was invisible in gate language. ROUTINE ADDED (CURRENT_STATE watchlist header): every batch review reports ① actual $/day, ② the same trades re-priced under the two pending sizing unlocks (5→4 slots ×1.25; sub-0.90 UNMATCHED at 2.5×), ③ both gate tallies. Also RE-SCOPED the Jul-6 2.0→2.5× step gate to the demuxed reality: only sub-0.90 UNMATCHED 2× fires count toward the 50 (zone fires are 1× by the Jul-10 demux and excluded). Tallies at registration: 5→4 = 46 fresh since 07-03, 52%, −$1,254 (N met, net failing); 2.5× = 12/50, 9W, +$531.

## 2026-07-10 (addendum 2) — max_open_positions 5→4 SHIPPED (operator-directed early ship; $-per-day routine "actual" definition corrected)

Operator: post-filter throughput is very low (6 surviving trades/2d) → increase the ticket. The Jul-4 gate (N≥30 fresh positive) sat at 46·52%·−$1,254 = failing; shipped anyway as a transparent override on three grounds: paper ticket size doesn't alter the forward test's evidence (WR/% size-invariant); the cap-cost counter has NEVER recorded a forfeited trade (cost side empty, structurally so at ~4 trades/day); operator capital-utilization call. Sizing epoch note: batch $ vs v14 anchor $ now incomparable (~+25%) — Avg P&L%/WR only (principle #3). Revert conditions kept (≥3 forfeited signals in a batch OR ≥4-concurrent loss cluster → back to 5). The real-money bar is unchanged: forward test must pass before live. Also this session: the $-per-day routine's "actual" was corrected after operator challenge (batch actual = realized −$234/day, NOT the +$12/day current-stack counterfactual; historical pool "actual" −$114/day dropped from the yardstick entirely — it blends 20+ dead configurations and describes nothing). Yardstick stays ~+$215/day counterfactual (day-by-day verified: 18/21 active days positive, max day 14% of total), haircut to ~+$110–150/day expected forward.

## 2026-07-10 (addendum 3) — NEGDI15 flip multiplier cell SHIPPED (proactive all-sleeve deep sweep session)

Operator requested a proactive winners-vs-losers deep dive on all three sleeves (all variables, 2D, 3D, theory). Sweep = scripts/sweep_separators.py × 3 sleeves (~2,600 tests each, cross-era ranked) + manual bucket/threshold/overlap passes.

**Findings.** ML: top lead global_volume_ratio = mid-range hole (0.7-1.0 · 67% · +$1,140) flanked by 100%/93% zones → locked confound rule refutes; its 2D survivors all reduce to the crowding family = the already-shipped PVR demux re-found through a second lens. MS: d13/stretch leads both mid-hole shapes at N=19; zero consistent 2D quadrants → nothing. FS: **pair −DI monotone and era-consistent** — <12 = 4·50%·−$21 · 12-15 = 10·60%·−$257 · 15-18 = 12·100%·+$706 · ≥18 = 5·100%·+$265; all 6 losers below 15; ≥15 cell = 17·100%·+$971 across 13d/15 pairs; 2D shows −DI-hi wins at BOTH +DI levels (explains the Jul-1 DI-spread OOS inversion — spread mixed signal with noise); corr: btcTG +0.01 / pairATR −0.25 / adxΔ −0.59. NOT a filter (every threshold blocks a 50-67% WR cohort; sub-15 is 8W/6L interleaved). 3D declined on principle (N=41/31/19 → 2-5-trade cells = noise manufacturing).

**Ship.** flip_short_negdi_mult=2.0 / negdi_min=15.0 / lev 1.0, tag [NEGDI15], engine cell beside TG_SHALLOW, 'neg_di' added to _ff_in (entry-fields first, ind fallback), full D11 (UI multiplier-cell panel in the flip-short section + load/save; verified 3 hits/ID; smoke-tested 16.5→2×, 14.9→1×, None→1×). ⚠ DOUBLE OVERRIDE: N=17<30 AND skips 1.5× staging (analyst recommended 1.5×; operator directed 2×; TG_SHALLOW precedent). 🔒 Reverts: ✗ HARMFUL (net-neg N≥5 fresh) → 1.0× · ⚠ DRAG → 1.5× · BE-compat at ≥3 cell losses. Screen anchors unaffected (flips de-muxed to 1× offline).

**Watchlist registered.** ① −DI<15 flip-short BLOCK: gate = fresh <15 flips ≤40% WR AND net-negative on N≥10 (≥3 dates); today it fails the filter gate (57% flank). ② PVR-demux revert question re-litigated same session (operator challenged via the gvol confound): demux KEPT — it stands on the multiplier-verdict rule (zone 9·67%·net-NEGATIVE at both sizings never earned a 2×), not the retired ladder narrative; gvol hole keeps its 2× under the same criterion (net-POSITIVE +$1,140). The consistent framework: sizing follows the cell's earned record, blocking follows the (unmet) filter gates.

## 2026-07-13 — LONG runner BE-lock SHIPPED (`runner_trail_be_ratchet_enabled` false→true, lock +0.10)

**Decision.** Re-enable the LONG armed-runner BE ratchet (operator-disabled Jun 25): once a LONG runner arms (peak ≥ 0.45%), the exit floor clamps to `max(peak − 1.0×ATR, +0.10)`. Config flip only — engine (realtime tick path since 07-06), UI toggle, and load/save were already wired; `runner_trail_be_lock_pct` stays 0.10. Operator words: "Activate the BE 0.10 for LONGs, ship it, commit and push."

**Evidence.** ① Pre-committed gate (registered 07-06) fired: 8/8 fresh armed longs, cumulative +0.70pp, zero negatives (PUMP +0.13, USELESS +0.57, AAVE + 5 ties). ② Pooled, gate-grade: N=38 armed ML runners, lock +6.11pp vs actual, 14 better/9 worse, 10/3 dates. ③ Same-session full-baseline head-to-head (this ship's trigger): atr10+BE +18.35% vs lockless atr10 +17.62% (+0.73pp floor benefit) vs actual +15.17%. ④ atr05 (N 1.0→0.5) refuted a THIRD time and now strictly dominated: atr05+BE +12.41% (−5.94pp vs atr10+BE) — the floor rescues sagging winners (atr05's only benefit) without the runner tax (ACT −2.83pp, NEAR −1.45, WLD −1.31 under the tight trail). Do not re-propose tighter N; the floor is the correct instrument.

**Tracking (roles invert at ship).** Live = locked; the existing lockless `atr10` shadow column becomes the revert monitor (rides Order columns → CSV/pool for free). Per batch: Δ = actual − atr10-raw shadow, cumulative. Floor fires observable as RUNNER_TRAIL closes at ~+0.10 (`[REALTIME_RUNNER_TRAIL]` floor log); post-exit regret table gives trap color (PostPeak% after a floor fire) but the shadow prices it. 🔒 REVERT GATE (locked at ship): revert to false if cumulative Δ < 0 on N≥10 fresh armed LONGs across ≥3 dates. Method note: baseline head-to-head approximates the floor as max(shadow, +0.10) on armed trades — gap-through slippage applies equally to both sides, ranking robust.

## 2026-07-13 PM — PUMP double-loss post-mortem + watchlist adds (no ship)

**Event.** PUMP LONG 15:18 (2×, $1,543) never-positive → STOP_LOSS −1.01% / −$313 in 12.5min; second identical PUMP death of the day (07:54, PVR-demuxed to 1×, −$151 — the demux's first save). Batch 5t 3W/2L −$178; both losses = one pair.

**Analysis run (all against SCREENED baseline v14, 41 ML longs).** ① Entry legal on every gate; red-flag stack = gap 0.5993 (ceiling 0.60), rngpos 85.2, RSI falling, late-day chase. ② Gap round-up (block ≥0.595): round-up zone = 1 baseline trade, a WINNER (TIA +$163) vs today's PUMP loser → 1W/1L, watchlist gate set. Gap 0.5-0.6 band overall WINS (8·87.5%·+$957) — ceiling stays 0.60. ③ BTC-1h-slope cohort: batch's "5m-up/1h-down 50%" was noise — baseline 1h-DOWN longs are the sleeve's BEST cohort (19·89.5%·+$2,205); withdrawn as a flag (also corrected: it's BTC's 1h slope, not the pair's). ④ 2D sweep centered on the PUMP entry (325 tercile cells): 13 negative cells, ALL reduce to the refuted gvol-mid-hole confound / the 07-10 macro-day cluster / N≤6 sweep noise. ⑤ Pair ledger: PUMP under current stack was 3/3 runner wins before today; lifetime ML-long 3W/2L ≈ −$96 → NOT blacklistable; watch with standard gate (net-neg on N≥5, NP-dominated).

**Also this session (trail thread, closed):** atr05 refuted 3rd time incl. +BE-floor head-to-head (LONG N=38: atr10+BE +18.35% vs atr05+BE +12.41%, −5.94pp; atr10 adv is tail-concentrated — flips only if the 4 best runners are deleted; symmetric trim stays +); per-pair/conditional trail N refuted (runner-ness is episode-level: WLD/AAVE/JTO/PEPE each flip sides across own trades; ATR buckets non-monotonic); runner soft signature = high BTC ADX (28 vs 23, N=7) → future SIZING candidate at N≥30, never a trail rule. Watchlist E-block added to CURRENT_STATE.

## 2026-07-13 PM (addendum) — hole-cell deep dive series (operator-driven; all no-ship, 2 watches armed)

**Context.** Operator pursued the gvol-mid × btcATR-mid negative 2D cell across five follow-up angles after the PUMP post-mortem. Every angle tested against SCREENED baseline v14 (41 ML longs) + today's 5 live trades (kept separate — batch not yet folded; pool absorbs it at next full-batch re-screen).

**Findings.** ① 3×3 grid + today: PUMP am (btcATR-low) landed in the 86%-WR neighbor, PUMP pm in the hole → same-pair same-day losses in DIFFERENT cells = coordinates aren't the mechanism. Hole now 7t·29%·−$665, all 8 neighbors 75–100% positive. ② Forensic all-column sweep of the 7 cell trades: the 6 baseline members collapse to 3 entry-moments (UNI/LIT/ADA 07-10 bit-identical BTC context = the 17:02 macro trio; ME/UNI 07-03 identical context with OPPOSITE outcomes) — the common factor is temporal clustering, and the stagger lever was already refuted (−$280). ③ Regime: entry regime uniform bull (HB 34·82%·+$2,103 / SB 7·100%·+$1,403 — nothing to cut); 3/4 hole losers exited in degraded regimes (Mode-3 confirmation) but drifted-regime cohort WINS overall (19·84%·+$2,061) and drift is post-hoc → no lever, regime-change exit stays off. ④ Within-cell separator hunt: exactly ONE perfect separator (ADXΔ: W≥+0.318 vs L≤+0.279, 0.04 margin) → INVERTS on the sleeve (ADXΔ<−0.3 = 6/6·+$881 best bucket; today's 3 winners ALL ADXΔ-negative; global block forfeits 19t·74%·+$1,370) = textbook sweep-noise. ⑤ DI/RSI family flat on longs: −DI all buckets 79–100% positive (flip-short-specific mechanism confirmed); +DI and DI-spread mid-wiggle confound shapes; RSI 55-60 = 73%·+$664 (blockable only by violating the anti-pattern rule). ⑥ rngpos 90→85 tighten REFUTED: 85–90 band = 13·77%·+$744 baseline (holds PUMP +$208 and 2 AAVE runners), still +$280 net WITH today's two PUMP losses.

**Armed.** CURRENT_STATE watchlist E④ hole-cell de-mux gate (N≥10 · WR≤40% · net-neg across ≥5 independent entry-moments — clusters count once; sizing-only if it trips) + E⑤ rngpos 85–90 tally (re-open at ≤40% WR net-neg N≥10 ≥3 dates). Thin-tape theory logged as plausible-but-unconfirmed; its predicted discriminators failed.

## 2026-07-13 PM — GAPFLAT PROBE shipped (`gap_probe_enabled=true`) + max_open 4→5

**Decision (operator: "Agree, ship it, commit and push" + mid-build "set back to 5 max open positions, now we will have more volume").** Real-data A/B on the #1 entry blocker PAIR_EMA_GAP_NOT_EXPANDING (~22% of all blocks, 5,094 counted): a momentum LONG failing ONLY the gap-expanding check opens as a REAL paper order at ~1× effective leverage (invest 0.5× · lev 0.05× × 20×), tagged cell_src=GAPFLAT_PROBE. Chosen over ① live disable (rejected: the filter is half the signal definition — the ONLY acceleration gate in the stack since ADXΔ-min sits at 0/off; disabling floods the book with Mode-2 stale chases) and ② phantom collection (operator preferred real trades: full real exit stack + 190 entry columns per row for the sub-cohort filter hunt). Key facts established first: the block counter is first-fail inflated (ladder position mid-chain; REGIME precedent 0/106 net-admissible), and no offline counterfactual is possible (blocks aren't stored).

**Build.** indicators.py: gap-flat LONGs fall through the ladder when probe on (byte-identical when off; smoke-tested all 4 quadrants) + `gap_expand_flat()` helper (exact tag, LONG-only, mode-aware). Engine: open_position(gap_probe=) — caps (last 2 slots reserved via open_count ≤ max_open−2; 1 concurrent; 3/day DB-counted restart-safe; cap rejection records PAIR_EMA_GAP_NOT_EXPANDING) + sizing override after the bounce_long block (overrides UNMATCHED 2×; own cell_src row; BULL_LONG/BOUNCE_LONG observation-sleeve pattern). Config×5 + JSON + UI row in the EMA Gap Expanding section + load/save (all ids 3 hits). D12: `_compute_gap_probe_cohort` (EXPANDING vs NON-EXPANDING; $ demuxed to 1×; DOA%<0.10 peak; auto-verdict) + payload + 2 empty fallbacks + UI table under the Relaxation table + BOTH text exports + config-report line. Contamination guards: screen_pool drops the tag (v14 anchors re-verified byte-identical post-change); relaxation MARGINAL/STRICT A/B skips probes; reviews quote headline ex-probe. Funnel semantics note: probe-on means gap-flat candidates dying downstream record their TRUE downstream blocker (free join).

**max_open 4→5 (operator).** Reverses the 07-10 early ship; rationale: probe flow needs slots + the probe's own guard keeps the last 2 slots for real signals (with 5 slots probes fire only at ≤3 open). The 07-10 revert gate resolved by operator decision. ⚠ SIZING EPOCH #2: tickets ×0.8 vs the 4-slot window — WR/Avg P&L% comparisons only.

**🔒 Gates (pre-committed, auto-rendered in the table):** N≥30 probes (≥10 dates): WR≥60% AND avg≥+0.15% → open relaxation discussion; WR≤45% OR avg<0 → filter vindicated, probe off permanently. Expected cadence ~2-3/day → decision in ~2 weeks; worst-case cost ≈ −$40-60 total.
