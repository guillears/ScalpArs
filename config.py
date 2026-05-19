"""
SCALPARS Trading Platform - Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Dict, Optional
from enum import Enum
import json
import os


class ConfidenceLevel(str, Enum):
    NO_TRADE = "NO_TRADE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class ConfidenceConfig(BaseModel):
    """Configuration for each confidence level"""
    enabled: bool = True
    trade_mode: str = "both"  # "long", "short", "both"
    leverage: float = 3.0
    investment_multiplier: float = 1.0  # Multiplier for investment size
    stop_loss: float = -0.4  # % of notional
    signal_active_sl: float = -0.40  # Wider SL used when entry signal is still active
    tp_min: float = 0.6  # % of notional
    pullback_trigger: float = 0.3  # % price pullback from peak
    gap_min: float = 0.08  # % minimum gap required (EMA5-EMA20)/price
    gap_max: float = 0.40  # % maximum gap allowed (filters overextended entries)
    gap_enabled: bool = True  # Whether to enforce gap requirement
    max_ema5_stretch: float = 0.12  # % max distance from EMA5 allowed for entry
    be_levels_enabled: bool = True  # Master toggle for break-even trailing stop levels
    # 3-Level Trailing Break-Even: progressive SL tightening as trade moves in favor
    be_level1_trigger: float = 0.08  # P&L % to activate Level 1 (micro-protection)
    be_level1_offset: float = -0.15  # SL once Level 1 active (reduce max loss)
    be_level2_trigger: float = 0.18  # P&L % to activate Level 2 (profit lock)
    be_level2_offset: float = 0.05   # SL once Level 2 active (small profit locked)
    be_level3_trigger: float = 0.25  # P&L % to activate Level 3 (full protection)
    be_level3_offset: float = 0.15   # SL once Level 3 active (meaningful profit locked)
    be_level4_trigger: float = 0.40  # P&L % to activate Level 4 (runner protection)
    be_level4_offset: float = 0.25   # SL once Level 4 active
    be_level5_trigger: float = 0.60  # P&L % to activate Level 5 (deep runner protection)
    be_level5_offset: float = 0.40   # SL once Level 5 active
    tp_trailing_enabled: bool = True  # Enable TP extension and trailing stop logic


class SignalThresholds(BaseModel):
    """Thresholds for signal generation"""
    # LONG conditions
    long_rsi_extreme: float = 30.0
    long_rsi_high: float = 30.0
    long_rsi_medium: float = 35.0
    long_rsi_low: float = 40.0
    
    # SHORT conditions
    short_rsi_extreme: float = 70.0
    short_rsi_high: float = 70.0
    short_rsi_medium: float = 65.0
    short_rsi_low: float = 60.0
    
    # ADX thresholds
    adx_extreme: float = 35.0
    adx_high: float = 25.0
    adx_medium: float = 20.0
    adx_low: float = 20.0
    
    # Volume threshold for EXTREME
    volume_multiplier: float = 1.5
    
    # Momentum signal thresholds (EMA5/EMA8 gap)
    ema_gap_threshold: float = 3.0  # % minimum EMA5-EMA8 gap for momentum signals (legacy fallback)
    ema_gap_threshold_long: float = 0.02  # Min EMA5-EMA8 gap for LONG entries
    ema_gap_threshold_short: float = 0.05  # Min EMA5-EMA8 gap for SHORT entries
    ema_gap_5_8_max: float = 0.0  # Max EMA5-EMA8 gap % for entry (0 = disabled) — DEPRECATED, kept for back-compat. Use ema_gap_5_8_max_long/_short below.
    # May 2: split EMA5-EMA8 max by direction (mirrors existing min split). When
    # both _long and _short are set, those override the legacy ema_gap_5_8_max.
    # Auto-migration in code: if legacy field is set and direction-split fields
    # are zero, populate both from legacy on first read.
    ema_gap_5_8_max_long: float = 0.0  # Max EMA5-EMA8 gap % for LONG entries (0 = disabled)
    ema_gap_5_8_max_short: float = 0.0  # Max EMA5-EMA8 gap % for SHORT entries (0 = disabled)
    adx_strong: float = 16.0  # ADX threshold for STRONG_BUY (SHORT)
    adx_very_strong: float = 30.0  # ADX threshold for VERY_STRONG (SHORT)
    adx_strong_long: float = 16.0  # ADX threshold for STRONG_BUY (LONG)
    adx_very_strong_long: float = 30.0  # ADX threshold for VERY_STRONG (LONG)
    momentum_adx_max_long: float = 100.0  # Max ADX for LONG entries (100 = disabled)
    momentum_ema20_filter_long: bool = True
    momentum_ema20_filter_short: bool = True
    momentum_ema20_slope_filter_long: bool = True
    momentum_ema20_slope_filter_short: bool = True
    momentum_ema20_slope_min_long: float = 0.0
    momentum_ema20_slope_min_short: float = 0.0
    # May 12: Range Position min filter (price position in 20-candle high-low range).
    # SHORT @ RP <2% = catastrophic pile-on zone (cross-batch: N=22, 32% WR, -$452).
    # 0 = disabled. Independent per direction.
    range_position_min_short: float = 0.0
    range_position_max_long: float = 100.0
    # May 10: minimum ADX delta (current ADX − ADX 1 candle ago).
    # Cross-sample validated 2-sample finding (May 4 224tr survivors + May 10 34tr):
    # ADXΔ <0.10 = ~17% WR / -0.42% Avg; ADXΔ ≥0.10 = ~62% WR / +0.03% Avg.
    # Independent per direction. SHORT side is essentially a no-op at 0.10
    # (only 1-2 trades affected per batch) — kept symmetric for simplicity.
    # 0 = disabled.
    min_adx_delta_long: float = 0.0
    min_adx_delta_short: float = 0.0
    # May 2: per-pair EMA20 slope MAX filter (new). Block entry when
    # abs(pair_ema20_slope) > max — guards against over-extended pair trends.
    # 0 = disabled.
    momentum_ema20_slope_max_long: float = 0.0  # Max abs EMA20 slope % for LONG entries (0 = disabled)
    momentum_ema20_slope_max_short: float = 0.0  # Max abs EMA20 slope % for SHORT entries (0 = disabled)
    # May 2: BTC EMA20 slope MAX filter (new). Block entry when
    # abs(btc_ema20_slope) > max — guards against over-extended BTC trends
    # (late-cycle entries when BTC has already run too far).
    # 0 = disabled.
    btc_ema20_slope_max_long: float = 0.0  # Max abs BTC EMA20 slope % for LONG entries (0 = disabled)
    btc_ema20_slope_max_short: float = 0.0  # Max abs BTC EMA20 slope % for SHORT entries (0 = disabled)
    macro_trend_filter_enabled: bool = True
    macro_trend_neutral_mode: str = "both"  # "both" or "none"
    macro_trend_flat_threshold: float = 0.07  # DEPRECATED — kept for backward compat
    macro_trend_flat_threshold_long: float = 0.0  # % change below which EMA20 slope is NEUTRAL for longs (0 = any slope counts)
    macro_trend_flat_threshold_short: float = 0.02  # % change below which EMA20 slope is NEUTRAL for shorts
    momentum_long_rsi_min: float = 55.0  # Min RSI for momentum LONG (0 = disabled)
    momentum_long_rsi_max: float = 100.0  # Max RSI for momentum LONG (100 = disabled)
    momentum_short_rsi_max: float = 50.0  # Max RSI for momentum SHORT (100 = disabled)
    momentum_adx_max: float = 100.0  # Max ADX for momentum entries (100 = disabled)
    momentum_short_rsi_min: float = 30.0  # Min RSI for momentum SHORT - avoid shorting oversold (0 = disabled)
    btc_global_filter_enabled: bool = True  # Use BTC regime to gate all pairs (overrides per-pair regime)
    btc_rsi_min_long: float = 0  # Min BTC RSI to allow LONGs (0 = disabled)
    btc_rsi_max_long: float = 100  # Max BTC RSI to allow LONGs (100 = disabled)
    btc_rsi_min_short: float = 0  # Min BTC RSI to allow SHORTs (0 = disabled)
    btc_rsi_max_short: float = 100  # Max BTC RSI to allow SHORTs (100 = disabled)
    btc_adx_min_long: float = 0  # Min BTC ADX to allow LONGs (0 = disabled)
    btc_adx_max_long: float = 100  # Max BTC ADX to allow LONGs (100 = disabled)
    btc_adx_min_short: float = 0  # Min BTC ADX to allow SHORTs (0 = disabled)
    btc_adx_max_short: float = 100  # Max BTC ADX to allow SHORTs (100 = disabled)
    btc_rsi_min_long: float = 0  # Min BTC RSI to allow LONGs (0 = disabled)
    btc_rsi_max_long: float = 100  # Max BTC RSI to allow LONGs (100 = disabled)
    btc_rsi_min_short: float = 0  # Min BTC RSI to allow SHORTs (0 = disabled)
    btc_rsi_max_short: float = 100  # Max BTC RSI to allow SHORTs (100 = disabled)
    btc_adx_min_long: float = 0  # Min BTC ADX to allow LONGs (0 = disabled)
    btc_adx_max_long: float = 100  # Max BTC ADX to allow LONGs (100 = disabled)
    btc_adx_min_short: float = 0  # Min BTC ADX to allow SHORTs (0 = disabled)
    btc_adx_max_short: float = 100  # Max BTC ADX to allow SHORTs (100 = disabled)
    btc_adx_dir_long: str = "both"  # BTC ADX direction filter for LONGs: "both", "rising", "falling"
    btc_adx_dir_short: str = "both"  # BTC ADX direction filter for SHORTs: "both", "rising", "falling"
    btc_trend_filter_enabled: bool = False  # BTC EMA20 vs EMA50 macro trend filter (May 5). Blocks countertrend entries: EMA20 > EMA50 blocks SHORTs, EMA20 < EMA50 blocks LONGs.
    adx_dir_long: str = "both"  # Pair ADX direction filter for LONGs: "both", "rising", "falling"
    adx_dir_short: str = "both"  # Pair ADX direction filter for SHORTs: "both", "rising", "falling"
    signal_lost_exit_enabled: bool = True  # Close when EMA5/EMA8 momentum reverses while in profit
    signal_lost_min_profit: float = 0.05  # Min P&L % (notional) to trigger signal-lost exit
    signal_lost_max_profit: float = 999.0  # Max P&L % for signal-lost exit (creates a range with min)
    signal_lost_flag_enabled: bool = True  # Flag trades at signal lost instead of exiting; let them run
    signal_lost_flag_security_min: float = -0.9  # Security gap lower bound for flagged trades
    signal_lost_flag_security_max: float = -0.7  # Security gap upper bound — flagged trades exit here if signal still lost
    # FL1 extension: flag STOP_LOSS_WIDE trades (instead of closing) and let them run to emergency backstop
    fl1_for_wide_sl_enabled: bool = True
    fl1_wide_sl_backstop: float = -1.2  # Emergency SL for FL1[WIDE_SL] trades (hit this → FL_EMERGENCY_SL)
    # FL2 double-flag: when a flagged trade hits the security gap, promote to FL2 with tighter recovery target
    fl2_enabled: bool = True
    fl2_recovery_target: float = -0.4  # Tight recovery target — close as FL_RECOVERED if P&L climbs back to this level
    fl2_deep_stop: float = -1.0  # Deep stop — close as FL_DEEP_STOP if P&L falls below this level
    ema5_slope_exit_enabled: bool = True  # Exit when EMA5 slope decelerates (momentum loss)
    ema5_slope_lookback: int = 3  # Number of candles back for EMA5 slope calculation
    ema5_slope_threshold: float = 0.01  # Min EMA5 slope % to stay in trade (0 = original behavior)
    price_ema5_exit_ratio: float = 0.3  # Exit when price-to-EMA5 distance drops to this fraction of peak (0 = disabled)
    min_peak_ema5_gap_pct: float = 0.05  # Min peak gap (% of entry price) before distance trailing activates (0 = no minimum)
    pnl_trailing_trigger: float = 0.1  # Min peak P&L % to activate P&L trailing exit (0 = disabled)
    pnl_trailing_ratio: float = 0.5  # Ratio when signal lost (MOMENTUM_EXIT) -- tighter
    pnl_trailing_ratio_signal_active: float = 0.3  # Ratio when signal active (PNL_TRAILING) -- wider
    ema_gap_expanding_filter: bool = True  # Block entry if EMA5-EMA8 gap is compressing (current <= previous candle)
    # EMA5-EMA20 Gap Filter (signal quality gate — separate for longs/shorts)
    ema_gap_5_20_enabled: bool = True  # Master toggle for EMA5-EMA20 gap requirement
    ema_gap_5_20_min_long: float = 0.15  # Min EMA5-EMA20 gap % for LONG entries
    ema_gap_5_20_min_short: float = 0.15  # Min EMA5-EMA20 gap % for SHORT entries
    ema_gap_5_20_max_long: float = 0.8  # Max EMA5-EMA20 gap % for LONG entries (overextended filter)
    ema_gap_5_20_max_short: float = 0.8  # Max EMA5-EMA20 gap % for SHORT entries (overextended filter)
    # EMA5 Stretch Filter (May 9 — moved from per-confidence-level to top-level per-direction min/max).
    # Tests: |price - ema5| / price * 100. Replaces per-confidence-level max_ema5_stretch.
    # Set min > 0 to require minimum stretch (decisive momentum). 0 = disabled. Cross-sample
    # confirmed (May 4 + May 9) that LONG stretch <0.16% is a structural loser zone.
    ema5_stretch_filter_enabled: bool = True
    ema5_stretch_min_long: float = 0.16  # Min EMA5 stretch % for LONG entries (0 = disabled)
    ema5_stretch_max_long: float = 0.0   # Max EMA5 stretch % for LONG entries (0 = disabled)
    ema5_stretch_min_short: float = 0.0  # Min EMA5 stretch % for SHORT entries (0 = disabled)
    ema5_stretch_max_short: float = 0.0  # Max EMA5 stretch % for SHORT entries (0 = disabled)
    # REMOVED May 15 PM — Stretch-based multiplier retired (UI panel + engine
    # lookup deleted). Fields kept here purely so old JSON files with these
    # keys still load without Pydantic errors. No code reads them anymore.
    # Historical trades with cell_multiplier_source = "STRETCH_*" retain their
    # attribution in the Multiplier Cell Performance table.
    ema5_stretch_multiplier_long: str = ""
    ema5_stretch_multiplier_short: str = ""
    # Trailing pullback confirmation (May 9): require N seconds of sustained
    # pullback before trailing exit fires. Catches single-tick noise wicks
    # (e.g. SAHARAUSDT 1-second wick on high-ATR pair). 0 = disabled (fire
    # immediately like before). Default 15s — short enough to add minimal
    # delay on real reversals (~0.05pp), long enough to filter <15s noise.
    trailing_pullback_confirmation_seconds: int = 15
    rsi_momentum_filter_enabled: bool = True  # Block LONG if RSI falling, block SHORT if RSI rising (vs 3 candles ago)
    rsi_momentum_exit_enabled: bool = True  # Exit LONG on 2 consecutive RSI drops, SHORT on 2 consecutive rises
    rsi_momentum_exit_min_profit: float = 0.05  # Min P&L % (notional) to trigger RSI momentum exit
    rsi_momentum_exit_max_profit: float = 999.0  # Max P&L % to trigger RSI momentum exit (caps to losers when set to 0)
    # EMA13 Cross Exit (May 6) — closes trade on first tick where price crosses EMA13
    # against trade direction (LONG: price < EMA13, SHORT: price > EMA13). Fires
    # in parallel to FL flags, RSI Handoff, trailing stop — first-to-fire wins.
    # Reuses the realtime cross detection from the Phase 1 shadow tracker.
    # Default OFF — flip to True to activate.
    ema13_cross_exit_enabled: bool = False
    # May 8: optional strict mode — when True, EMA13 cross only fires the
    # exit if EMA5/EMA8 stack has ALSO flipped against trade direction.
    # Filters single-candle wicks below EMA13 from triggering (real
    # reversals also flip the stack). Adds 1-3 candles of latency vs
    # EMA13-only. Fail-closed on missing EMA5/EMA8 data.
    ema13_cross_requires_stack_flip: bool = False
    # EMA Stack Cross Exit (May 6) — closes trade when EMA5 crosses EMA8 against
    # trade direction (LONG: ema5 < ema8, SHORT: ema5 > ema8) past the configured
    # TP level.  Mirrors RSI Handoff architecture: at current_tp_level >= level,
    # SUPPRESSES trailing pullback and becomes the exclusive natural exit until
    # the EMA stack inverts.  Faster than RSI 2-drop (~5min lag vs ~15min).
    # Default OFF; level default 2 (peak >= tp_min*2 to activate).
    ema_stack_cross_exit_enabled: bool = False
    ema_stack_cross_exit_level: int = 2
    # May 7: Tier-aware trailing pullback widening. Effective pullback at
    # current_tp_level N = pullback_trigger + pullback_widening_per_level * (N - 1).
    # Default 0.0 = flat trailing (current behavior, no change). Set to 0.10 to
    # add +0.10% room per TP level (L1=0.20, L2=0.30, L3=0.40 with base 0.20).
    # Rationale: bigger winners get more room to ride; small winners stay tight.
    pullback_widening_per_level: float = 0.0
    # May 7 (Phase 1): ATR-normalized trailing pullback floor.
    # effective_pullback = max(fixed_pullback, entry_atr_pct × trailing_atr_multiplier).
    # Default 0.50 = "give the trade half a normal candle of noise". Volatile
    # pairs (high ATR) get wider pullback; calm pairs use the fixed pullback.
    # Set to 0.0 to disable ATR floor entirely.
    trailing_atr_multiplier: float = 0.50
    # May 7 (Phase 2): early-arm trailing zone. Trailing activates with a tight
    # pullback when peak is between this threshold and tp_min (the regular L1
    # arming point). Locks in profit on moderate-momentum trades that peak in
    # the +0.30% to +0.50% range and reverse before reaching L1. Default 0.30
    # arms at peak ≥ +0.30%; set to 0.0 to disable.
    trailing_early_arm_threshold: float = 0.30
    # Pullback used in the early-arm zone (peak between early_arm_threshold
    # and tp_min). Tight by design — only fires on real reversals, not noise.
    trailing_early_arm_pullback: float = 0.10
    # May 7: Pair Trend Filter (pair-level analog of BTC Trend Filter).
    # Compares pair EMA13 vs EMA50 (5m candles, ~65min vs ~250min). Blocks
    # countertrend entries:
    #   pair_ema13 < pair_ema50 → pair in 4hr downtrend → block LONGs
    #   pair_ema13 > pair_ema50 → pair in 4hr uptrend → block SHORTs
    # Defensive ship — same primitive operating one level down from BTC,
    # 6-trade cross-sample evidence (May 5 SHORTs against pair uptrends +
    # May 7 LONGs against pair downtrends, all 6 lost). Default ON.
    pair_trend_filter_enabled: bool = True
    tick_momentum_exit_enabled: bool = False  # Real-time tick-based momentum exit via WebSocket
    tick_momentum_exit_min_profit: float = 0.05  # Min P&L % to trigger tick momentum exit
    tick_momentum_exit_min_profit_flagged: float = -0.10  # Min P&L % for flagged trades (Signal Lost Flag system)
    tick_momentum_exit_min_delta: float = 0.05  # Min % price drop across each window to confirm fade (fallback)
    tick_momentum_exit_min_deltas: str = ""  # Per-window deltas, comma-separated (overrides min_delta when set)
    tick_momentum_exit_windows: str = "15,30,60"  # Comma-separated rolling window sizes in seconds
    regime_change_exit_enabled: bool = True  # Close positions when BTC macro regime flips against trade direction
    # Phase 1d-ExitTest (May 2): RSI handoff at high TP levels. When ON, disables
    # trailing-stop pullback past `rsi_handoff_level` and enables 2-drop RSI
    # momentum exit (any P&L) at that level instead. Tests "RSI is the better
    # winner-exit signal once trade has proven itself." Default OFF — feature
    # ships inert; user enables via UI when ready to test live.
    rsi_handoff_active: bool = False  # Master toggle for RSI handoff at high TP levels
    rsi_handoff_level: int = 3  # Promote-past level at which trailing disables and RSI takes over
    rsi_adx_filter_long: str = ""  # RSI x ADX cross-filter for LONGs, e.g. "55-60:18,60-65:25" (empty = allow all)
    rsi_adx_filter_short: str = ""  # RSI x ADX cross-filter for SHORTs, e.g. "30-35:25,35-50:30" (empty = allow all)
    btc_rsi_adx_filter_long: str = ""  # BTC RSI x ADX cross-filter for LONGs (empty = allow all)
    btc_rsi_adx_filter_short: str = ""  # BTC RSI x ADX cross-filter for SHORTs (empty = allow all)
    # ADX Delta x BTC ADX cross-filter (May 11, 2026 — pooled-data finding, see CLAUDE.md).
    # Format per rule: "<deltaLo>-<deltaHi>:<btcAdxLo>-<btcAdxHi>" (block when both ranges match).
    # Example: "1.0-2.0:18-25" blocks LONG entries when pair ADX delta in [1.0,2.0) AND BTC ADX in [18,25).
    # Multi-batch evidence: catastrophic loser zone (N=49 pooled, 31% WR, -$267).
    adx_delta_btc_adx_filter_long: str = ""
    adx_delta_btc_adx_filter_short: str = ""
    # May 18: master toggle for the cross-filter. When False, rules are stored
    # but not enforced — lets the operator disable for A/B testing without
    # losing the rule definitions.
    adx_delta_btc_adx_filter_enabled: bool = True
    # Range Position × ADX Delta 2D Cross-Filter (May 18 PM).
    # Catches the "bottom/top-fishing into momentum acceleration" pattern that
    # the existing filters don't cover. Range Position alone blocks only <2%
    # (too tight), ADX Delta filters only block LOW delta (wrong direction).
    # This 2D rule blocks entries where price is at the EXTREME of recent
    # range (5-10% for SHORTs, 90-95% for LONGs) AND ADX Δ is in the
    # accelerating-but-not-extreme zone (1.0-2.0).
    # Format per rule: "<rngposLo>-<rngposHi>:<adxDeltaLo>-<adxDeltaHi>".
    # Block when range_position in [rngposLo, rngposHi] AND ADX Δ in [adxDeltaLo, adxDeltaHi).
    # Cross-batch evidence (today + May 16+): N=10, 30% WR, -$359 SHORT.
    # Captures 4 of 4 May 18 SHORT cluster losers (1000PEPE/TON/SUI/BTC).
    rngpos_adx_delta_filter_long: str = ""
    rngpos_adx_delta_filter_short: str = ""
    # Master toggle for the RngPos × ADX Δ cross-filter. Same A/B pattern.
    rngpos_adx_delta_filter_enabled: bool = True
    # BTC EMA13-EMA50 Gap × BTC ADX 2D Cross-Filter (May 19, 2026).
    # Catches the "BTC mid-extension + low/climax trend conviction" LONG loser zone
    # that single-axis filters can't express. Inside Gap [+0.10, +0.20%]:
    #   - ADX <22 = mean-revert (-$1,022 / 31t / 39% WR, 5 of 6 dates losing)
    #   - ADX 22-25 = healthy continuation (+$177 / 10t / 90% WR — RESCUE, preserved)
    #   - ADX 25-28 = climax (-$415 / 9t / 22% WR — added with N=9 override)
    # Format per rule: "<gapLo>-<gapHi>:<adxLo>-<adxHi>" (block when both ranges match).
    # Half-open ranges [lo, hi). Multi-rule comma-separated.
    btc_gap_btc_adx_filter_long: str = ""
    btc_gap_btc_adx_filter_short: str = ""
    # Master toggle. Same A/B pattern as other cross-filters.
    btc_gap_btc_adx_filter_enabled: bool = True
    # Premium Multiplier (May 4, 2026 — Phase 3 Position Multiplier Mechanism, per CLAUDE.md May 3 design).
    # Format per rule: "<RSI_min>-<RSI_max>:<ADX_min>-<ADX_max>:<multiplier>", comma-separated.
    # Example: "55-60:22-25:2.0,60-65:18-22:1.5" — boost LONG entries in those two cells by the listed factor.
    # Empty = no rules active (everything 1.0×). Cells not listed default to 1.0×.
    # Hard cap clamps any per-cell value to `rsi_adx_multiplier_hard_cap` regardless of UI input — safety guard.
    # Conflict resolution: when a single trade matches BOTH a pair-level rule AND a BTC-level rule,
    # the HIGHER multiplier applies (max, not multiply). Rationale: independent confirmation bonus
    # would compound past the hard cap; HIGHER is safer + intuitive.
    rsi_adx_multiplier_long: str = ""  # Pair-level RSI x ADX multiplier rules for LONG
    rsi_adx_multiplier_short: str = ""  # Pair-level RSI x ADX multiplier rules for SHORT
    btc_rsi_adx_multiplier_long: str = ""  # BTC-level BTC RSI x BTC ADX multiplier rules for LONG
    btc_rsi_adx_multiplier_short: str = ""  # BTC-level multiplier rules for SHORT
    rsi_adx_multiplier_target: str = "investment"  # "investment" (multiply position size $) or "leverage"
    rsi_adx_multiplier_hard_cap: float = 2.0  # UI-configurable safety cap; engine clamps any cell to this
    # Entry Quality Score multiplier (May 18, NEW dimension). Format: "<score_lo>-<score_hi>:<multiplier>", comma-separated.
    # Example: "4-5:2.0" matches score=4 only (range is half-open [lo, hi)).
    # Multi-rule: "3-4:2.0,6-7:2.0" matches score=3 OR score=6.
    # HIGHER-wins conflict resolution same as RSI×ADX multipliers; hard cap applies.
    score_multiplier_long: str = ""
    score_multiplier_short: str = ""
    # Entry Quality Score block filter (May 15 PM, opt-in). Toggle + threshold.
    # When enabled, blocks entries with entry_quality_score <= block_max.
    # Threshold semantics match the table: block_max=1 → blocks Score 0 AND
    # Score 1; block_max=2 → blocks Score 0,1,2; etc.
    # Cross-sample evidence (CLAUDE.md May 15 watchlist): Score ≤ 1 across 10
    # archived samples + today = N=95, 34.7% WR, −$684, direction-consistent.
    # Ship gated behind explicit operator opt-in; default disabled.
    entry_quality_score_filter_enabled: bool = False
    entry_quality_score_block_max: int = 1
    # Fast Exit (May 15 PM, opt-in). Quick-profit lock for trades that hit a
    # threshold within a small window after entry. Mirrors the Fast-Exit
    # Counterfactual table's mechanic but fires LIVE — the moment price ticks
    # at or above the threshold within the window, the trade closes at that
    # price. Distinction from the counterfactual: the table uses
    # peak_reached_at as proxy (conservative); live fires on first qualifying
    # tick. So live results may lock smaller profits than the counterfactual
    # implied on big peakers (closer to the threshold itself).
    # Close reason: "FAST_EXIT L1".
    fast_exit_enabled: bool = False
    fast_exit_threshold_pct: float = 0.20  # P&L % required to fire
    fast_exit_window_minutes: int = 2      # Time window since opened_at (inclusive)
    # Fast Exit L2 (May 19, 2026) — "slow climber" tier between L1 and trailing.
    # L1 catches fast bursts (peak ≥0.20% in 2min). Trailing arms at peak ≥0.50%.
    # L2 fills the gap: trades that build to 0.40% over 2-5min then would die
    # without ever hitting trailing's 0.50% arming threshold.
    # Logic: L1 wins on overlap (fires first if peak hits 0.20% within 2min).
    # L2 fires only when L1 didn't (peak crosses 0.40% in the 2-5min window).
    # Close reason: "FAST_EXIT L2". Auto-included in Post-Exit Regret table
    # (no whitelist) and post-exit running state preservation (startswith FAST_EXIT).
    fast_exit_l2_enabled: bool = True
    fast_exit_l2_threshold_pct: float = 0.40
    fast_exit_l2_window_minutes: int = 5
    # Pattern C Tracker (May 19, 2026 — observation-only, no behavior change).
    # Captures 4 candidate Pattern C precursor signatures at entry for each
    # direction. Pattern C = trade peaks <+0.10% (never positive). Multiple
    # structural causes (capitulation chase, macro counter-trend, stretch
    # exhaustion, low-vol chop) tested simultaneously. Locked promotion gates
    # at N≥30 matches per pattern. See CLAUDE.md May 19 entry.
    pattern_c_tracker_enabled: bool = True
    # SHORT C1 — Capitulation chase
    pc_short_c1_rngpos_max: float = 15.0
    pc_short_c1_pair_gap_max: float = -0.50
    pc_short_c1_adxd_min: float = 1.0
    # SHORT C2 — Macro counter-trend (BTC RSI rising + BTC ADX falling + BTC Gap > -0.05)
    pc_short_c2_btc_gap_min: float = -0.05
    # SHORT C3 — Stretch exhaustion
    pc_short_c3_stretch_min: float = 0.40
    pc_short_c3_pair_adx_min: float = 30.0
    pc_short_c3_rngpos_max: float = 15.0
    # SHORT C4 — Low-vol chop
    pc_short_c4_btc_atr_max: float = 0.15
    pc_short_c4_btc_adx_max: float = 22.0
    pc_short_c4_pair_adx_max: float = 25.0
    # LONG C1 — Climax chase (mirror)
    pc_long_c1_rngpos_min: float = 85.0
    pc_long_c1_pair_gap_min: float = 0.50
    pc_long_c1_adxd_min: float = 1.0
    # LONG C2 — Macro counter-trend (BTC RSI falling + BTC ADX falling + BTC Gap < +0.05)
    pc_long_c2_btc_gap_max: float = 0.05
    # LONG C3 — Stretch exhaustion
    pc_long_c3_stretch_min: float = 0.40
    pc_long_c3_pair_adx_min: float = 30.0
    pc_long_c3_rngpos_min: float = 85.0
    # LONG C4 — Low-vol chop (same as SHORT)
    pc_long_c4_btc_atr_max: float = 0.15
    pc_long_c4_btc_adx_max: float = 22.0
    pc_long_c4_pair_adx_max: float = 25.0
    global_volume_filter_enabled: bool = False  # Gate trades when top-N aggregate volume is below average
    global_volume_threshold_long: float = 1.05  # MIN global volume ratio to allow LONGs (block if vol < this)
    global_volume_threshold_short: float = 1.05  # MIN global volume ratio to allow SHORTs (block if vol < this)
    # SHORT-only MAX-side cap with BTC CAPITULATION OVERRIDE (May 11, 2026 — multi-axis filter).
    # Block SHORTs when GlobalVol > max UNLESS BTC is in capitulation state.
    # Multi-batch evidence (47 SHORTs at GlobalVol >1.05, 5 batches):
    #   - Capitulation cell (BTC RSI < 30 AND BTC slope < 0): N=19, 63% WR, +$157 ★ (preserve — ride cascade)
    #   - Non-capitulation cell: N=28, 29% WR, -$243 ✗ (block — whip/squeeze risk)
    # The high-vol SHORT loser pattern is conditional on BTC NOT being in capitulation.
    # When BTC is dumping (RSI low + slope falling), high vol = selling climax = SHORT-friendly.
    # When BTC is bouncing/chopping, high vol = two-sided fight = squeeze risk for SHORTs.
    # See CLAUDE.md May 11 SHORT capitulation finding for full analysis.
    global_volume_max_short: float = 0.0  # MAX GlobalVol cap for SHORTs (0 = disabled)
    global_volume_max_short_capitulation_rsi: float = 30.0  # Override threshold: skip block if BTC RSI < this (signals deep oversold)
    global_volume_max_short_capitulation_slope: float = 0.0  # Override threshold: skip block if BTC slope < this (signals falling; negative = down)
    pair_volume_filter_enabled: bool = False  # Gate trades when per-pair volume is below its own average
    pair_volume_threshold_long: float = 1.10  # Min pair volume ratio to allow LONGs
    pair_volume_threshold_short: float = 1.10  # Min pair volume ratio to allow SHORTs
    # May 10 evening: intersection-style rescue clause for global volume filter.
    # When global vol filter would block, the pair is rescued from blocking if
    # its absolute 24h USD volume is ≥ this threshold. 0 = disabled.
    # Independent per direction. Effective mechanism: filter A (Global<0.95)
    # AND (Pair Vol $ < rescue_threshold) — large-cap pairs in quiet markets pass.
    pair_volume_usd_rescue_long: float = 0.0
    pair_volume_usd_rescue_short: float = 0.0
    global_volume_lookback_bars: int = 48  # Rolling window for global volume average (5m bars)
    pair_volume_lookback_bars: int = 20  # Rolling window for per-pair volume average (5m bars)
    market_breadth_filter_enabled: bool = True  # Gate entries based on % of pairs in Bull/Bear regime
    market_breadth_bull_threshold_long: float = 50.0  # Min Bull% of scanned pairs to allow LONGs
    market_breadth_bear_threshold_short: float = 65.0  # Min Bear% of scanned pairs to allow SHORTs
    market_breadth_flat_threshold: float = 0.03  # EMA20 slope % threshold for breadth regime classification (independent of macro_trend_flat_threshold)

    # Spike Guard: block entries during abnormal candles (crashes/pumps)
    spike_guard_enabled: bool = True
    spike_guard_volume_multiplier: float = 3.0  # Block if candle volume >= X × 20-bar avg AND price moved >= spike_guard_price_move_pct
    spike_guard_price_move_pct: float = 1.5  # Min candle price move % to trigger volume spike block
    spike_guard_max_ema20_distance_pct: float = 2.0  # Block if price is >= X% away from EMA20 (overextended)


class InvestmentConfig(BaseModel):
    """Investment configuration"""
    mode: str = "percentage"  # "fixed", "percentage", or "equal_split"
    fixed_amount: float = 100.0  # USD
    percentage: float = 5.0  # % of available balance
    
    # Safe reserve
    reserve_mode: str = "percentage"  # "fixed" or "percentage"
    reserve_fixed: float = 500.0  # USD
    reserve_percentage: float = 20.0  # % of total balance
    
    # Cooldown after losing trade (prevents immediate re-entry on same pair)
    cooldown_after_loss_minutes: int = 5  # Minutes to wait before re-entering same pair after loss
    
    # Position limits
    max_open_positions: int = 100  # Max simultaneous open positions
    min_investment_size: float = 100.0  # Min investment per trade (USD)
    max_investment_size: float = 50000.0  # Max investment per trade (USD)
    max_holding_time_minutes: int = 180  # Max time to hold a trade (minutes), 0 = disabled
    no_expansion_minutes: int = 15  # Close if no expansion after N minutes (peak < BE trigger & current < BE offset), 0 = disabled


class TradingConfig(BaseModel):
    """Main trading configuration"""
    # Trading fee per side (legacy field, kept for backward compatibility)
    trading_fee: float = 0.00045  # 0.045% per side (taker default)
    
    # Independent fee rates
    maker_fee: float = 0.00018  # 0.018% per side (limit order fills)
    taker_fee: float = 0.00045  # 0.045% per side (market order fills)
    
    # Maker entry settings
    maker_entry_enabled: bool = False
    maker_timeout_seconds: int = 15
    maker_offset_ticks: int = 2
    # Signal re-validation before taker fallback (Apr 18 Phase 1c Amendment #7).
    # When ON: after maker timeout exhausts, re-evaluate the original signal's
    # filters; if signal is no longer valid, abort entry and persist as
    # SIGNAL_EXPIRED row (no taker fallback fires).  When OFF: taker fallback
    # fires immediately at timeout (pre-Apr-18 behaviour).  Toggle added May 4,
    # 2026 because (a) maker timeout reverted to 20s (Amendment #6 revert)
    # which materially reduces signal-staleness exposure, and (b) some users
    # want the option to disable re-validation entirely as it can systematically
    # block trades that would have entered cleanly under the looser pre-Apr-18 path.
    revalidate_on_taker_fallback: bool = False  # default OFF (May 4 user-directed; pre-Apr-18 behaviour as base)
    
    # Maker exit settings
    maker_exit_enabled: bool = False
    maker_exit_timeout_seconds: int = 10
    maker_exit_offset_ticks: int = 2
    
    # Paper trading
    paper_trading: bool = True
    paper_balance: float = 2000.0  # Starting balance for paper trading
    
    # BNB fee management
    bnb_swap_enabled: bool = True
    bnb_check_interval_hours: int = 12
    bnb_runway_hours: int = 24
    paper_bnb_initial_usd: float = 500.0
    
    # Trading pairs limit (how many top pairs by volume to trade)
    trading_pairs_limit: int = 20  # 5, 10, 20, or 50
    pair_blacklist: str = ""  # Comma-separated pairs to exclude, e.g. "XRPUSDT,DOGEUSDT"
    # Skip pairs whose Binance futures onboardDate is within the last N days.
    # Binance flags early-stage pairs ("Seed Tag" / "Monitoring Tag") as
    # higher-risk: low liquidity, wider spreads, manipulation-prone — poor fit
    # for the 5m EMA-based strategy.  Filtered BEFORE the top-N-by-volume cut,
    # so "top 50" always means "top 50 of eligible pairs."  0 = disabled.
    new_listing_filter_days: int = 0
    # Alpha-subtype filter (May 5, 2026): skip pairs Binance flags as
    # `underlyingSubType: ["Alpha"]` — their launchpad / Innovation Zone tier.
    # These pairs carry Binance's "high volatility" UI warning, have elevated
    # triggerProtect (0.15 vs 0.05 for liquid pairs), and historically show the
    # "never-positive + emergency-SL" failure pattern (LABUSDT, RAVEUSDT in
    # the May 5 batch).  Independent of listing age — catches launchpad pairs
    # regardless of when they listed.  6 pairs in current top-50 are Alpha.
    # Default ON.  Toggle off only for analysis (e.g., to test whether Alpha
    # pairs are systematically bad or sometimes profitable).
    alpha_subtype_filter_enabled: bool = True

    # Broker-side protective stops feature REMOVED Apr 17 after 4 failed
    # hotfix attempts.  Binance rejected STOP_MARKET/TAKE_PROFIT_MARKET on
    # /fapi/v1/order with -4120 for this account/CCXT combo, and the Portfolio
    # Margin routing path returned -2015 because the account is not PM-enrolled.
    # See CLAUDE.md "Broker-side Protective Stops removal" for forensic detail.
    # The bot relies exclusively on internal in-process exits for risk mgmt.
    
    # Investment settings
    investment: InvestmentConfig = InvestmentConfig()
    
    # Signal thresholds
    thresholds: SignalThresholds = SignalThresholds()
    
    # Post-exit regret tracking
    post_exit_tracking_enabled: bool = True
    post_exit_tracking_minutes: int = 45

    # Confidence levels configuration
    confidence_levels: Dict[str, ConfidenceConfig] = {
        "LOW": ConfidenceConfig(
            enabled=True,
            trade_mode="both",
            leverage=3.0,
            investment_multiplier=1.0,
            stop_loss=-0.4,
            tp_min=0.6,
            pullback_trigger=0.3,
            gap_min=0.08,
            gap_max=0.40,
            gap_enabled=True,
        ),
        "MEDIUM": ConfidenceConfig(
            enabled=True,
            trade_mode="both",
            leverage=5.0,
            investment_multiplier=1.5,
            stop_loss=-0.6,
            tp_min=0.9,
            pullback_trigger=0.4,
            gap_min=0.08,
            gap_max=0.40,
            gap_enabled=True,
        ),
        "HIGH": ConfidenceConfig(
            enabled=True,
            trade_mode="both",
            leverage=10.0,
            investment_multiplier=2.0,
            stop_loss=-0.8,
            tp_min=1.2,
            pullback_trigger=0.5,
            gap_min=0.08,
            gap_max=0.40,
            gap_enabled=False,
        ),
        "EXTREME": ConfidenceConfig(
            enabled=True,
            trade_mode="both",
            leverage=15.0,
            investment_multiplier=2.5,
            stop_loss=-1.0,
            tp_min=1.5,
            pullback_trigger=0.6,
            gap_min=0.08,
            gap_max=0.40,
            gap_enabled=False,
        ),
        "VERY_STRONG": ConfidenceConfig(
            enabled=True,
            trade_mode="both",
            leverage=10.0,
            investment_multiplier=1.0,
            stop_loss=-0.25,
            tp_min=0.25,
            pullback_trigger=0.05,
            gap_min=0.12,
            gap_max=0.30,
            gap_enabled=True,
        ),
        "STRONG_BUY": ConfidenceConfig(
            enabled=True,
            trade_mode="both",
            leverage=10.0,
            investment_multiplier=1.0,
            stop_loss=-0.25,
            tp_min=0.25,
            pullback_trigger=0.05,
            gap_min=0.12,
            gap_max=0.30,
            gap_enabled=True,
        )
    }


# Pick the right default DB path for the environment:
#
# On Elastic Beanstalk, /opt/scalpars-data/ is created by the predeploy hook
# and survives deploys because it lives OUTSIDE /var/app/current/ (which is
# replaced on every code push). Connecting directly to the absolute path
# means the DB survives deploys even if staging symlinks break.
#
# Locally, use a relative path next to the repo. This auto-detects based on
# whether /opt/scalpars-data/ exists at import time, so no env-var juggling
# is needed for dev vs prod.
#
# URL format reminder: sqlite+aiosqlite:///relative.db  = 3 slashes = relative
#                      sqlite+aiosqlite:////absolute.db = 4 slashes = absolute
if os.path.isdir("/opt/scalpars-data"):
    _DEFAULT_DB_URL = "sqlite+aiosqlite:////opt/scalpars-data/scalpars.db"
else:
    _DEFAULT_DB_URL = "sqlite+aiosqlite:///./scalpars.db"


class Settings(BaseSettings):
    """Application settings from environment"""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    app_env: str = "development"
    debug: bool = False
    database_url: str = _DEFAULT_DB_URL

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Config file path
CONFIG_FILE = "trading_config.json"


def load_trading_config() -> TradingConfig:
    """Load trading configuration from file or return defaults"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                return TradingConfig(**data)
        except Exception as e:
            print(f"Error loading config: {e}")
    return TradingConfig()


def save_trading_config(config: TradingConfig) -> bool:
    """Save trading configuration to file"""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


# Global trading config instance
trading_config = load_trading_config()
