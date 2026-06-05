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
