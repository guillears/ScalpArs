"""
SCALPARS Trading Platform - Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import Dict, List, Optional
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
    # May 24: BTC 1h Slope MAX guard. Block LONG entries when BTC 1h EMA20 slope
    # over the prior 3 hours exceeds the threshold — catches late-stage steep-rising
    # BTC trends where LONGs are mean-reversion candidates.
    # Cross-batch + today: slope > +0.15% LONG cohort N=26 / 30.8% WR / -$837 (today),
    # active-window pool also showed cliff at 0.12-0.15%. 0 = disabled.
    # SHORT side disabled by default (no clean cliff observed yet).
    btc_1h_slope_max_long: float = 0.0
    btc_1h_slope_max_short: float = 0.0
    # Jun 3: minimum BTC 1h slope FLOOR (block entries when 1h slope is too steeply
    # NEGATIVE = shorting into a steep 1h crash = exhaustion/bounce). 0 = disabled;
    # a negative value activates. SHORT cross-batch: 1h slope < -0.60 = 0W/4L
    # (SEI, XRP, BTC, JTO). LONG left disabled (its loser zone is the FLAT band, not
    # the steep band — a different mechanism, not shipped here).
    btc_1h_slope_min_long: float = 0.0
    btc_1h_slope_min_short: float = 0.0
    # Jun 10: BTC 1h RSI FLOOR for SHORTs — block shorting when BTC's HOURLY RSI is already
    # deep-oversold (= shorting into the hourly bounce zone; the 1h twin of the 5m
    # climax-oversold block). Cross-batch matched shorts: 1hRSI<30 = -$940, 30-35 = -$382,
    # 35-40 = +$651 (monotonic; blocking <35 = NET +$1,322, helps 5 of 7 dates). 0 = disabled.
    btc_rsi_1h_min_short: float = 0.0
    # Jun 3: BTC-ACCELERATION CHASE filter (STATEFUL, evolution-vs-last-entry).
    # Blocks a LONG when the live BTC EMA20 slope is HIGHER than it was at the most
    # recent LONG that actually opened within `evo_chase_window_min` minutes — i.e.
    # BTC has accelerated since the last entry = chasing a maturing move = late.
    # Cross-batch (7-batch proxy, 30min): block cohort 30.8% WR / Σ-3.1% (net-losing,
    # the 4-loss clusters). Caught the 4 consecutive LONG losses on 06-03 (0/4) while
    # keeping both winners. LONG only; SHORT plumbed-but-disabled (untested side).
    evo_chase_filter_long_enabled: bool = False
    evo_chase_filter_short_enabled: bool = False
    evo_chase_window_min: int = 30
    # May 10: minimum ADX delta (current ADX − ADX 1 candle ago).
    # Cross-sample validated 2-sample finding (May 4 224tr survivors + May 10 34tr):
    # ADXΔ <0.10 = ~17% WR / -0.42% Avg; ADXΔ ≥0.10 = ~62% WR / +0.03% Avg.
    # Independent per direction. SHORT side is essentially a no-op at 0.10
    # (only 1-2 trades affected per batch) — kept symmetric for simplicity.
    # 0 = disabled.
    min_adx_delta_long: float = 0.0
    min_adx_delta_short: float = 0.0
    # May 22: Entry Distance from EMA13 minimum filter (Pair Extension floor).
    # Block LONG entries with (price - ema13)/ema13 × 100 < min — these are
    # bottom-of-pullback bounce-buying entries that historically die at NP /
    # EMA13_CROSS_EXIT. Cross-batch evidence (153-trade LONG pool, 7 batches):
    # pair_ext < 0.20% = 9 trades / 7L / saves $250 / cuts $13 / ratio 19.82.
    # 0 = disabled. SHORT analog not yet validated.
    entry_dist_from_ema13_min_long: float = 0.0
    entry_dist_from_ema13_min_short: float = 0.0  # SHORT analog — uses abs(pair_ext_pct) when active. 0 = disabled.
    entry_dist_from_ema13_filter_enabled: bool = True  # Master toggle (May 22 UI ship).
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
    # SHORT-only BTC ADX block range (May 27, 2026 — see CLAUDE.md). Blocks SHORT entry when
    # btc_adx_block_min_short <= btc_adx < btc_adx_block_max_short. Both 0 = disabled.
    # Default 24/30 from cross-batch evidence (965-trade pool): BTC ADX 24-30 SHORT = 49% WR / -$16/tr.
    btc_adx_block_min_short: float = 0.0
    btc_adx_block_max_short: float = 0.0
    # BTC RSI band × BTC ATR conditional block (May 27, 2026 — A3 ship per CLAUDE.md).
    # Replaces the broad BTC RSI 65-70 block with a surgical "RSI band AND BTC ATR condition" filter.
    # Format per rule: "RSI_LO-RSI_HI:OP" where OP is "<X", ">X", or "X-Y". Multi-rule comma-separated.
    # Default "65-70:<0.10" blocks LONG when BTC RSI in [65, 70) AND BTC ATR < 0.10% (dead-tape top).
    # Cross-batch evidence (965-trade pool): 35 trades / 40% WR / -$1,118 / -$32/tr in this cell.
    # Save:cut ratio 3.99:1 (vs 1.91:1 for broad block) — preserves NEAR +$197 / GMT +$86 winners.
    btc_rsi_band_atr_block_long: str = ""
    btc_rsi_band_atr_block_short: str = ""
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
    # Jun 8: strictness mode for the gap-expanding filter. 'both' (legacy) = block unless the
    # EMA5-EMA13 gap beats BOTH prev1 AND prev2 candles (a fresh 3-bar expansion high — very
    # strict, the #1 entry blocker at 31% of all blocks). 'prev2_only' = block only if the gap
    # fails prev2 (tolerates a 1-candle pause within an intact trend; admits "MARGINAL" entries
    # the strict rule rejected). Trades admitted by prev2_only that would have failed prev1 are
    # tagged entry_gap_expand_marginal=True so the cohort's WR can be isolated. Default 'both'.
    ema_gap_expanding_mode: str = 'both'  # 'both' | 'prev2_only'
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
    # Jun 7: per-direction gates for EMA13 cross exit (under the master
    # ema13_cross_exit_enabled toggle). When a side is False, the EMA13 cross
    # does NOT close that side — instead it records a PHANTOM (phantom_ema13_cross_pnl/_at)
    # of where it would have exited, and the trade rides to its real exit. Lets us
    # measure "disable EMA13 cross for LONGs" live (phantom vs held CF) at zero
    # blind risk. Both default True = fire for whichever direction the master enables.
    ema13_cross_exit_long_enabled: bool = True
    ema13_cross_exit_short_enabled: bool = True
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
    # Jun 8: trailing min-profit GATE. The price-drop trailing stop only fires when its
    # exit level (peak_pnl − effective_pullback) is ≥ this. Below it, the trailing is
    # SUPPRESSED (dormant) — it does NOT realize a loss/sub-min exit; the trade rides on
    # the hard SL until the peak climbs enough to lock ≥ this, then the trailing re-arms
    # and trails the new peak normally. Fixes high-ATR L1 whipsaws where the ATR-widened
    # pullback exceeds the peak (e.g. peak +0.45 − pullback 0.67 = −0.22 → exits red on a
    # pair that recovers). Default −99 = disabled (current behavior, fires at any level).
    # Cross-batch (9-pool+batch, N=16 whipsaw trades): suppress+ride = +$1,506 vs +$190.
    trailing_min_profit_to_fire: float = -99.0
    # May 22: ATR-adjusted SL floor (analog of trailing_atr_multiplier but for SL).
    # Widens the hard SL on high-ATR pairs to prevent wicks from stopping trades
    # whose signal is still valid. Only WIDENS (more negative); never tightens.
    # effective_sl = min(stop_loss, -(entry_atr_pct × sl_atr_multiplier))
    # 0.0 = disabled. 1.5 default = "1.5 candles of noise breathing room."
    # Cross-batch evidence (May 22): 68 STOP_LOSS_WIDE trades, 19% had post-peak
    # ≥+0.60% (heavy regret). LONG heavy-regret avg ATR 1.165% vs right-exits 0.631%.
    # SHORT heavy-regret avg ATR 0.633% vs right-exits 0.500%. Projected save:
    # ~$700-1000 across pool after in-sample bias haircut.
    sl_atr_multiplier: float = 1.5
    # May 23: ATR-SL widening floor cap. The sl_atr_multiplier formula
    # produces effective_sl = -(atr × mult). On extreme-ATR pairs (e.g.,
    # ATR 2.3%) this gives -3.47% — effectively no SL. Today's COSUSDT
    # trade ran to -1.52% before EMA13 caught it (~$75 worse than base
    # -0.70 SL would have produced). This field clamps the WIDENING:
    # if (atr × mult) > |floor|, effective_sl is capped at floor.
    # Negative value = active cap. 0.0 = disabled (no cap, current behavior).
    # Default -1.20 chosen from cross-batch: cap engages for ATR > 0.80%,
    # zero winners killed (all high-ATR winners had trough > -0.68%).
    # See CLAUDE.md May 23 entry for full rationale.
    sl_atr_widen_floor_pct: float = -1.20
    # Jun 1, 2026 — RUNNER STRETCH-TRAIL (scoped high-ATR LONG runner exit).
    # Once a high-ATR LONG proves itself a runner (peak ≥ arm_peak), HAND OFF
    # from the tight price-trailing to a loose STRETCH-trail: hold until live
    # |price−EMA5| stretch collapses to runner_trail_k × the peak stretch.
    # Lets IDU-class runners run (shadow strpk +6.80 vs tight +1.47 on IDU);
    # faders are excluded because they never reach the 0.70 arm. Backstops
    # (ATR-widened hard SL −1.20 floor + EMA13 strict cross) stay live — the
    # stretch-trail only governs the profit-taking side. See CLAUDE.md Jun 1.
    # Validated: shadow-armed LONG arm-0.70 strpk net +4.57 vs actual −1.36 (N=16).
    runner_trail_enabled: bool = True
    runner_trail_atr_min: float = 1.0    # only pairs with entry ATR ≥ this get the loose leash
    runner_trail_arm_peak: float = 0.70  # handoff point: peak P&L ≥ this swaps tight→stretch trail
    runner_trail_k: float = 0.5          # exit when live stretch ≤ k × peak stretch (unsigned, matches shadow strpk)
    # Jun 12 — SHORT-side runner stretch-trail (DISCIPLINE-OVERRIDE ship, N=20<30).
    # Evidence: shadow strpk on current-stack book shorts = Δ+5.1pp/+$996 vs actual
    # (N=20, 13/7 better); recent era (Jun 9+) 8/0 better, +$979. Shorts are
    # capitulation cascades — live exits fire on the first micro-bounce.
    # Params MUST match the measured leash sim: arm at peak>=0.45 (leash ACT),
    # NO ATR gate (atr_min=0 — book shorts enter at ATR 0.4-1.0), K=0.5.
    # Once armed: tight trailing suppressed AND the EMA13-short cross records a
    # phantom instead of closing (the sim's uplift comes from riding through it).
    runner_trail_short_enabled: bool = True
    runner_trail_short_atr_min: float = 0.0   # 0 = no ATR gate (shadow had none)
    runner_trail_short_arm_peak: float = 0.45 # matches leash ACT + live trailing arm
    runner_trail_short_k: float = 0.5         # shadow strpk K=0.5 (stretch-ratio trail — fallback when use_atr=false)
    # Jun 16 — ATR-floored give-back trail (chandelier). Root cause of strpk early exits: the
    # K×peak_stretch ratio collapses to ~0 width on a freshly-armed (tiny) peak, so a first
    # bounce trips it before the move develops. The ATR-floor gives a VOLATILITY floor: exit
    # only when P&L retraces > atr_mult × entry_atr_pct from peak — a normal bounce (<1 ATR)
    # can't trip it; only a real reversal does. Applies to ALL flip shorts running strpk.
    # N=1.0 robust default (would have held AERO/HYPE/STG); shadow tests 0.5/1.0/1.5.
    runner_trail_short_use_atr: bool = True   # true = ATR-floor trail; false = K×peak_stretch ratio trail
    runner_trail_short_atr_mult: float = 0.5  # N — give back N×ATR% from peak before exit (hard SL still backstops). Jun 17 PM: REVERTED 1.0→0.5 — live sim showed N=1.0 captured LESS than N=0.5 (good batch +10.18% vs +15.37%; the N=1.0 shadow win was post-exit-continuation inflated). N=0.5 preserves the low-ATR winners (PORTAL 2.70 vs 1.96).
    # Jun 17 — BREAKEVEN RATCHET (min floor under the ATR-floor). Root cause of "peaked +0.5% then
    # closed negative": on high-ATR/modest-peak shorts (EVAA ATR1.89 peak0.59, VELVET ATR1.59 peak0.76)
    # the give-back N×ATR EXCEEDS the peak, so the chandelier floor (peak − N×ATR) sits BELOW breakeven —
    # it permits a full round-trip into a loss (or to the −0.70 SL). FIX: once ARMED, the effective exit
    # floor = max(peak − N×ATR, be_lock_pct). Only binds when peak − N×ATR < lock (the broken set); by
    # construction it CANNOT touch a runner (its floor stays well above the lock). be_lock 0.10 ≈ net-flat
    # after the 0.09% roundtrip fee. Sim (23 armed): converts the 5 broken trades from −0.02/−1.20 to ~+0.10.
    runner_trail_short_be_ratchet_enabled: bool = True  # true = clamp the armed exit floor to >= be_lock_pct
    runner_trail_short_be_lock_pct: float = 0.10        # min P&L an armed runner may give back to (the ratchet lock)
    # Jun 17 PM — GIVE-BACK CAP. On high-ATR pairs N×ATR exceeds any realistic peak, so the floor pins at
    # the lock and the trail surrenders the WHOLE runner to breakeven (AGT ATR3.9 peak+2.42 -> closed +0.10).
    # FIX: give_back = min(N×ATR, giveback_frac × peak) — never give back more than a fraction of the peak,
    # so the floor RISES with the peak instead of sticking at the lock. Binds only when frac×peak < N×ATR
    # (the high-ATR/modest-peak case); normal-ATR runners unaffected (ATR-floor stays tighter). 0 = off.
    # frac 0.35 from the good-batch sweep + noise-stop constraint (tighter over-fits / re-introduces noise-stops).
    runner_trail_short_giveback_frac: float = 0.0       # Jun 17: DISABLED. The cap protected bull/chop bounce-fades (now blocked by FLIP_SHORT_REGIME) but CLIPPED bear-regime trend runners (SKYAI +0.30 close vs +3.92 continuation, -0.02 retrace). Lock (BE-ratchet) is the round-trip backstop; cap was redundant tightening on the now-filtered population. 0 = off, raw N×ATR. Revert if an armed bear runner round-trips peak≥0.45→≤0 on N≥3.
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
    # Pair Trend Filter — pair EMA13 vs EMA50. Jun 13: SPLIT per-direction.
    # SHORT side: block when pair gap >= pair_trend_short_gap_max (default 0 =
    # EMA13>EMA50 = pair not yet below its 4hr trend → shorting before the
    # breakdown confirms → bounces). Book shorts gap>=0: 0% WR -$247 (N=6 book /
    # 39 all-pool, all May8-Jun1 while this was OFF). Watchlist: tighten the
    # threshold toward -0.2 (the -0.2..0 mild zone = 43% WR -$192).
    # LONG side: kept OFF — gap<0 unmatched longs are N=67, 58% WR, -$27 (≈breakeven),
    # not worth blocking. Legacy `pair_trend_filter_enabled` retired into these two.
    pair_trend_filter_long_enabled: bool = False
    pair_trend_filter_short_enabled: bool = True
    pair_trend_short_gap_max: float = 0.0  # block SHORT when pair (EMA13-EMA50)/EMA50% >= this
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
    # EMA Fan Acceleration (fan_ratio) dead-zone filter (May 29, 2026).
    # fan_ratio = abs(EMA5-EMA8 gap%) / abs(EMA8-EMA13 gap%). Measures whether the
    # EMA fan front is still widening (>1 accelerating) or compressing (<1).
    # May 29 batch (N=83) discovery: the MID-fan band is a clean loser dead-zone in
    # BOTH directions (mature/fully-developed trend = entering late, no edge):
    #   SHORT fan [1.02,1.65) = 0W/5-6L this batch (CLEAN, 0 winners killed).
    #   LONG  fan [0.85,1.70) = 25L/10W this batch (effective but kills 10 winners).
    # Symmetric mechanism = strong evidence it's structural, NOT single-batch luck.
    # CAVEAT: fan_ratio is UNVALIDATED cross-batch — entry_ema_gap_8_13 column only
    # exists from May-27 onward (FULL/4-batch pools have it NULL), so no historical
    # validation is possible yet. Next post-May-27 batch is the validation gate.
    # Rule format: "lo-hi" band(s), comma-separated. Block when fan_ratio in [lo, hi).
    # Empty = that direction inactive (observation-only).
    # SHIPPED May 29: SHORT active. LONG promoted to active same day (operator call —
    # N=35 killed/batch is significant despite single batch; symmetric mechanism). LONG
    # cut [0.85,1.70) kills ~10 winners but the 25 losers dominate 3.4:1.
    fan_ratio_block_long: str = "0.85-1.70,5.0-99"   # dead-zone block (May 29) + >5.0 flat-base cap (May 31)
    fan_ratio_block_short: str = "1.00-1.90"  # SHORT dead-zone block (floor 1.02->1.00 May 31; upper 1.65->1.90 Jun 1, spares 2.076 winner)
    fan_ratio_filter_enabled: bool = True     # master toggle (same A/B pattern)
    # === Flip Entry sleeve (Jun 14) — promote a proven Phantom-Flip cell to a LIVE
    # naked mean-reversion entry. When a listed filter BLOCKS an entry, open the
    # OPPOSITE direction (block LONG -> SHORT, block SHORT -> LONG). Jun 14 (REVERTED to
    # the EXIT FALLBACK after the momentum stack lost money): flips exit via the FLAT
    # phantom-replica model `_eval_flip_exit` (hard SL -0.70 / arm +0.45 / trail 0.25 /
    # 45min) — the exact exit the +0.175% edge was measured under. Bypasses the momentum
    # exit stack entirely (its ATR-widened trailing gave back 0.25-0.36 on moderate-peak
    # reversions -> -$243 / 46% WR). Exit reasons FLIP_-prefixed (FLIP_STOP_LOSS L1 etc.).
    # Entry is NAKED = NOT re-checked vs the opposite direction's filters. Tagged
    # entry_strategy="FLIP:<SOURCE>". Registry: comma-separated "SOURCE:size_mult" — a
    # SOURCE present = active (both sides); size_mult scales per-trade investment vs base.
    # FAN_RATIO_GATE shipped on N=97/39-pair/Top6%/WR69%/+0.175% phantom (in-sample).
    flip_entry_enabled: bool = True                       # master kill-switch for the whole sleeve
    flip_entry_sources: str = "FAN_RATIO_GATE:1.0,PAIR_RSI_OB:1.0"  # SOURCE:size:lev (lev optional→1.0). Jun 16 PM: RE-ACTIVATED PAIR_RSI_OB @1x base. Jun 17 PM: LONG_UNMATCHED_ONLY DISABLED (live off, phantom-only) — live N=0 both batches + phantom N≈8/38%WR/-0.190% ✗ whipsaws = starved & losing; phantom seed is decoupled (always fires) so it keeps tracking for a future re-enable. FAN_RATIO DE-MUXED 2x→1x Jun 15 (multiplier gate ✗ HARMFUL: live N=24/50%WR/-0.24%/-$912). RE-ENABLE LONG_UNMATCHED_ONLY only if its phantom clears WR≥55% AND net-positive on N≥20. [phantom prior PAIR_RSI_OB N=11/82%WR/+0.405%]
    # ── FAN_RATIO_GATE flip filter section (Jun 16, 76-trade batch). Source-namespaced
    #    (`flip_fan_*`); future sources get parallel `flip_unmatched_*`/`flip_pairsi_*` sets,
    #    evaluated independently in _flip_filters(). All fail-open. Block reasons FLIP_FAN_*.
    flip_fan_stretch_min: float = 0.12        # block FAN flip if entry EMA5 stretch < this (thin fuel, batch N=10/10%WR/-$495). 0 = off
    flip_fan_block_btc_rsi: float = 60.0      # block FAN flip if BTC RSI >= this AND BTC ADX >= flip_fan_block_btc_adx (fade into strong un-exhausted bull: N=19/47%WR/-$416). 0 = off
    flip_fan_block_btc_adx: float = 30.0      # paired with flip_fan_block_btc_rsi
    # Jun 17 — 2D regime×ADXΔ block for flip-SHORTS (ALL sources). Block a short flip when entry
    # ADXΔ < adxd_max AND BTC regime ∈ regimes set. Cross-batch (76+39+11 pool, deduped) the cell
    # BULL/CHOP ∧ ADXΔ<0 = N=38 / 40%WR / -0.34% / -$1070; 96% of its losers peak < 0.45 arm so the
    # give-back cap can't save them → entry block, not exit. Orthogonal to regime alone (ADXΔ sign
    # is ~50/50 within each regime). Counterfactual: last(bear) batch -$63 [dormant], current(bull)
    # batch -$611→-$5. Discipline-override (literal NP gate 13%<60%; saveability analysis overrides).
    # TIGHT REVERT: re-open if these cells flip to WR>45% on N>=15 fresh. Empty regimes = OFF.
    flip_short_regime_block_adxd_max: float = 0.0   # block flip-SHORT when entry ADXΔ < this (0.0 = the ADXΔ<0 cut)
    flip_short_regime_block_regimes: str = "STRONG_BULL,HEALTHY_BULL,CHOPPY_FLAT"  # CSV of BTC regimes to block flip-SHORTS in (ADXΔ<adxd_max gate); empty = filter OFF
    # Jun 17 (B2) — regimes where flip-SHORTS lose REGARDLESS of ADXΔ → block any-ADXΔ. STRONG_BULL loses
    # in BOTH ADXΔ halves (N=26: ADXΔ<0 −$190/N=11 already-blocked + ADXΔ≥0 −$531/N=15 was LEAKING through
    # the ADXΔ<0-only gate). Strong-bull-specific (healthy-bull/chop ADXΔ≥0 ≈ breakeven, kept at ADXΔ<0).
    flip_short_regime_block_any_adxd_regimes: str = "STRONG_BULL"  # CSV; empty = OFF
    # Jun 17 (B1) — anti-parabola: block flip-SHORT when EMA5 stretch% ≥ this (shorting a vertical blow-off
    # that keeps ripping; ESPORTS 10.47% stretch = −2.25% gapped stop in 0s). Pool stretch≥2 = N=2/0%WR
    # (ASTER+ESPORTS), 0 winners removed (1–2% band 67%WR preserved). Regime-agnostic catastrophe guard.
    flip_short_stretch_block_max: float = 2.0   # block flip-SHORT when entry EMA5 stretch% ≥ this (0 = off)
    # Jun 18 — BTC 30m-RSI-rising block (the cleanest cross-batch differentiator). FAN flip-SHORTS LOSE when
    # BTC 30m RSI is rising (macro bouncing → the faded pump squeezes) and PAY when falling. 2-batch consistent:
    # BTC-30m-rising −$1031 vs falling +$811; today −$965 of the −$998 FAN loss was BTC-30m-rising. Block SHORT
    # when (entry_btc_rsi − entry_btc_rsi_prev6) > this. 0.0 = block ANY 30m-RSI rise; 99 = OFF.
    flip_short_btc30_rise_block_min: float = 0.0
    # Jun 17 — high-ATR bear block (the regime-inverted hole in the bear exemption above). Block flip-SHORT
    # when pair ATR% ≥ min AND BTC regime ∈ bear set. High-ATR parabolic pump in a bear = counter-trend
    # squeeze (ESPORTS 4.0/HUSDT 3.0 = 0%WR/−$245). CUT=3.0 not 2.0 (ATR<2.5 bear shorts net-positive).
    flip_short_atr_block_min: float = 3.0   # block flip-SHORT when pair ATR% ≥ this (0 = off)
    flip_short_atr_block_regimes: str = "STRONG_BEAR,HEALTHY_BEAR,BEAR_EXHAUSTED"  # ...in these bear regimes; empty = OFF
    # Jun 19 — pair-RSI floor for flip-SHORTS. Fade quality scales with how overbought the blocked long was.
    # Cross-batch (Jun17/18/19, deduped): RSI<55 = N=21/57%WR/−0.094%/Σ−1.98 (the only consistently-negative
    # zone); RSI≥55 = N=78/65%WR/+0.056%/Σ+4.33 (carries ~all the edge); 60-65 = N=24/71%WR/+0.187%. Block
    # SHORT when pair RSI < this. Operator-directed, N below the locked filter gate → TIGHT REVERT.
    # 0 = OFF (fail-open on missing rsi too).
    flip_short_rsi_min: float = 0.0   # block flip-SHORT when entry pair RSI < this (0 = off)
    # Jun 17 — MIRROR of the short gate for flip-LONGS. A flip-LONG fades a blocked SHORT -> goes LONG;
    # in a STRONG_BEAR that's long-into-the-trend (AAVE/TAO this batch: 2/0%WR/-$220, straight to SL).
    # The observed long losers were ADXΔ-AGNOSTIC (ADXΔ +1.5, regime was the killer) → adxd_max default
    # 99 = REGIME-ONLY block; lower it later only if a long ADXΔ cell proves out cross-batch.
    flip_long_regime_block_adxd_max: float = 99.0   # block flip-LONG when entry ADXΔ < this (99 = regime-only, no ADXΔ cut)
    flip_long_regime_block_regimes: str = "STRONG_BEAR,HEALTHY_BEAR,CHOPPY_FLAT"  # CSV of BTC regimes to block flip-LONGS in; empty = filter OFF
    # Jun 17 — fan-SPIKE block (ALL flip sources, not just FAN). Block the flip when the pair's
    # entry fan ratio (|EMA5-8 gap| / |EMA8-13 gap|) >= this — a violently-accelerating parabolic
    # fan that the fade gets run over by (never arms, straight to SL). Cross-batch N=3, 0% WR,
    # ~-1.0% (ASTER 5.7/VELVET 28.3 [12-20-07] + ALLO 13.2 [06-16 ref], 3 pairs) — clears the
    # >=3-sample direction-consistent bar; mirrors the already-live fan_ratio_block_long 5.0-99.
    # Threshold is specifically >=5 (the 2-5 band CONTRADICTS cross-batch). 0 = off.
    flip_fan_spike_max: float = 5.0           # block any flip when pair fan ratio >= this (0 = off). TIGHT REVERT: re-open if fan>=5 flips >=40% WR on N>=5 fresh
    flip_fan_runner_strpk: bool = True        # exit FAN flips via the SHORT runner stretch-trail (strpk, arm 0.45/K0.5) instead of trailing-like-a-long. Reuses runner_trail_short_* params
    flip_runner_strpk_shorts: bool = True     # Jun 16: extend the SHORT runner stretch-trail (strpk) to the NON-FAN flip short sleeves too (PAIR_RSI_OB, LONG_UNMATCHED_ONLY). A flip short runs strpk if FAN+flip_fan_runner_strpk OR non-FAN+this. = strpk for ALL flip shorts
    flip_fan_mult_rule: str = "40-45:35-99:2.0:1.0"  # btc_rsi_lo-hi : btc_adx_lo-hi : size_mult : lev_mult cells (lev optional, defaults 1.0; same 4-part format as the other multiplier cells). Strong-bear cell N=10/90%WR/+$308 @2x size/1x lev. Empty = off. BELOW N>=30 gate — operator override
    # === Bull-Long Entry Sleeve (Jun 18) — the BUILD-side twin of the flip sleeve.
    # When a LONG PASSES the fan gate (low fan ratio) in a HEALTHY_BULL regime, open the
    # SAME direction (a real momentum-style long, NOT a fade) and let it run on the NORMAL
    # long exit stack (per-level trailing, ATR-widened SL) — it is NOT tagged _is_flip.
    # Multipliers default 1.0 (no amplification); leverage stays the normal STRONG_BUY level.
    # Tagged entry_strategy="BULL_LONG"; bypasses the long_unmatched_only + pattern-cell
    # entry blocks (it is explicitly a trend-build, not a pattern-matched late long). All
    # hard risk controls (max-open, existing-position, cooldown, liquidity caps) still apply.
    # TO REMOVE: grep "BULL_LONG" / "bull_long" + the main.py bull-long perf blocks + the UI.
    bull_long_enabled: bool = True                     # master toggle for the bull-long sleeve
    bull_long_regimes: str = "HEALTHY_BULL"            # CSV of BTC regimes the sleeve fires in
    bull_long_fan_max: float = 5.0                     # upper fan bound of the traded BLOCKED-long zone (fires on blocked longs fan 0.85..this; H.BULL edge runs 0.85-5)
    bull_long_size_mult: float = 1.0                   # investment multiplier (1.0 = no amplification)
    bull_long_lev_mult: float = 1.0                    # leverage multiplier (1.0 = normal leverage)
    # Pair ATR minimum filter (June 1, 2026). Block entries when pair ATR% < min
    # — the dead-tape, no-fuel fade zone (mirror of the high-ATR runner finding).
    # LONG <0.25%: 5-batch 12% WR / -$230 (cleanest loser sub-band), 0 overlap with
    # fan>5 / BTC-RSI-50-55. SHORT side disabled (0 = off) pending evidence.
    pair_atr_min_long: float = 0.0
    pair_atr_min_short: float = 0.0   # Jun 1: SHORT <0.25% validated (5-batch 20% WR / -$257). trading_config.json = 0.25
    pair_atr_filter_enabled: bool = True
    # Jun 10: pair ATR CEILING for LONGs — distribution guard. Historic max unmatched-long
    # winner = HOME at ATR 2.49; ESPORTS (ATR 4.68, p100 outlier meme) was a -$220 DOA.
    # Blocks only pairs outside everything ever validated. 0 = disabled. Live = 2.5.
    pair_atr_max_long: float = 0.0
    # Jun 13 — ATR×GAP LONG block (DISCIPLINE-OVERRIDE, N=16 full / N=5 recent < 30).
    # The "volatile-and-already-extended" quadrant: a high-ATR pair that has ALREADY
    # run far above its 4hr trend = buying the exhaustion top, which mean-reverts
    # (ENJ -$253 in 57s). Unmatched longs ATR>=1.0 & gap>=0.5: N=16 31%WR -$611 demux
    # (recent 12-batch: N=5 20%WR -$414); the SAME high-ATR with gap<0.5 = 64-75%WR
    # POSITIVE (the genuine runner — PRESERVED, do NOT widen the gap floor toward it).
    # gap = (EMA13-EMA50)/EMA50*100, matches entry_pair_ema20_ema50_gap_pct field.
    # Orthogonal to keep-only-unmatched (lives INSIDE the unmatched cohort; removing
    # the quadrant rehabilitates NO banned C/W pattern). 0 atr_min OR disabled = off.
    # REVERT GATE: drop if would-be-blocked longs >=50% WR on N>=8 fresh.
    atr_gap_block_long_enabled: bool = False
    atr_gap_block_atr_min_long: float = 1.0
    atr_gap_block_gap_min_long: float = 0.5
    # Jun 10: RSI-SPIKE GUARD (LONG) — block when the pair's RSI one candle ago was below
    # this floor, i.e. RSI teleported from neutral into the entry zone in a single candle =
    # first-candle pump chase (VVV 44.6->65, PIPPIN 45.5->58.3). Complements the fan-window
    # block (fan sees candles 2-5 of a spike; this sees candle 1). Cross-batch: blocks the
    # ESPORTS/PIPPIN/PEPE/VVV meme spikes, kills only $60 of winners. 0 = disabled. Live = 50.
    # GATE: drop if it blocks >=3 would-be winners with no loser saves on fresh data.
    rsi_prev_min_long: float = 0.0
    # Jun 10 (refinement): the spike guard fires only when BOTH (a) rsi_prev < rsi_prev_min_long
    # AND (b) the 1-candle jump (rsi - rsi_prev) >= this. Historic behavior identical (every
    # prev<50 entry jumped >=4 mechanically — entries need RSI ~54+), but formally excludes the
    # 49.8->51 non-spike case. Jump SIZE alone does NOT separate (winner avg +5.6 vs loser +5.8;
    # jump>=5 = NET -$318) — the signal is "momentum born from below the 50 neutral line".
    # 0 = jump condition off (pure floor).
    rsi_spike_min_jump_long: float = 0.0
    # BTC 1h × BTC 5m RSI Direction Cross-Filter (May 26, 2026 PM).
    # Block entry when both BTC RSI timeframes are in specified directions.
    # Rule format: 2-char codes "RR" "RF" "FR" "FF" where first=1h dir, second=5m dir.
    # R=Rising (curr > prev), F=Falling (curr <= prev).
    # Multiple rules comma-separated. Empty = filter inactive.
    # Default SHORT="RR" — blocks SHORT when both 1h and 5m BTC RSI are rising
    # (double-countertrend setup). N=5 combined evidence, 60% WR, -$182, 20% NP rate.
    # 11th locked-discipline override per CLAUDE.md May 26 PM watchlist.
    btc_1h_5m_rsi_dir_filter_long: str = ""
    btc_1h_5m_rsi_dir_filter_short: str = "RR"
    btc_1h_5m_rsi_dir_filter_enabled: bool = True
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
    # BTC ATR × BTC ADX 2D Cross-Filter (May 22, 2026).
    # Cross-batch SHORT evidence at BTC ADX ≥ 30:
    #   - BTC ATR <0.10% × BTC ADX ≥30 = 3 trades / 33% WR / -$159 ✗ killer
    #   - BTC ATR 0.10-0.15% × BTC ADX ≥30 = 17 trades / 100% WR / +$230 ★
    #   - BTC ATR 0.20-0.30% × BTC ADX ≥30 = 8 trades / 100% / +$83 ★
    # Mechanism: SHORTs at strong BTC trend (ADX ≥30) need volatility. Dead-quiet
    # BTC at strong trend = exhausted move + accumulated squeeze ammo. LONG mirror
    # shows OPPOSITE (8t / 88% WR at same cell) — asymmetric, SHORT-only filter.
    # Format per rule: "<atrLo>-<atrHi>:<adxLo>-<adxHi>" (block when both match).
    # Half-open ranges [lo, hi). Multi-rule comma-separated.
    btc_atr_btc_adx_filter_long: str = ""
    btc_atr_btc_adx_filter_short: str = "0.0-0.10:30-999"
    btc_atr_btc_adx_filter_enabled: bool = True
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
    # Apply mode (May 21 — extended): "investment" | "leverage" | "both".
    #   "investment": only the invest_mult column is applied; lev_mult column is stored but inert.
    #   "leverage":   only the lev_mult column is applied; invest_mult column is stored but inert.
    #   "both":       BOTH columns apply (compounding) — effective notional = investment × invest_mult × leverage × lev_mult.
    rsi_adx_multiplier_target: str = "investment"
    # Two independent hard caps (May 21). Each multiplier is clamped to its own cap regardless of UI input.
    # In "both" mode, max effective notional = invest_cap × lev_cap. Default cap_inv=2.0 reproduces pre-change behavior.
    rsi_adx_multiplier_hard_cap: float = 2.0  # Investment-side hard cap (was the single cap pre-May-21)
    rsi_adx_multiplier_lev_hard_cap: float = 2.0  # Leverage-side hard cap (NEW May 21)
    # Pattern Cell Ship Rules (May 21, NEW dimension) — per-pattern multipliers + fixed exits.
    # JSON list of objects with fields:
    #   pattern: signature to match. Supports (Jun 8 generalization):
    #       single code   "C4" | "W2"                    — fires when that C/W matches
    #       UNMATCHED     "UNMATCHED"                     — fires when NO C and NO W match
    #       combo (AND)   "C1+C6" | "C7+W2"               — fires when ALL listed codes match;
    #                                                        a mixed C+W combo resolves to the C
    #                                                        side (C-blocks-W priority)
    #   direction: "LONG" | "SHORT"
    #   inv_mult: float (default 1.0)            — investment multiplier
    #   lev_mult: float (default 1.0)            — leverage multiplier
    #   fixed_tp_pct: Optional[float]            — pnl% above which trade exits via PATTERN_FIXED_TP
    #   fixed_sl_pct: Optional[float]            — pnl% below which trade exits via PATTERN_FIXED_SL (negative value)
    #   block: Optional[bool] (Jun 8)            — if true, BLOCK entry entirely (counter PATTERN_CELL_BLOCK).
    #                                              Use for confirmed loser signatures (gate: N≥30, WR≤40%,
    #                                              Avg≤−0.20%, NP≥60%). Caps (fixed_sl) preferred for cohorts
    #                                              that still carry winners/runners — block only true junk.
    # Conflict resolution at engine: a trade matching ANY C pattern blocks all W-side
    # multiplier rules ("C presence blocks W treatment" — Option C from CLAUDE.md May 21
    # design discussion). Forward Unmatched cells NOT in initial ship (will be added
    # as proper pattern signatures once cross-batch identifies their structural shape).
    pattern_cell_rules: List = []
    # Extension Multiplier Rules (May 24, 2026) — Pair Distance from EMA13 multiplier dimension.
    # Each rule: dict with keys
    #   name: short label e.g. "L1b" (appears in source label as "EXT_L1b")
    #   direction: "LONG" or "SHORT"
    #   ext_min, ext_max: required range on entry_dist_from_ema13_pct (% from EMA13)
    #   pair_vol_max: optional — pair volume ratio max (e.g. 0.95 to require quiet pair tape)
    #   adx_delta_max: optional — ADX delta max (e.g. 0.3 to require slow momentum)
    #   inv_mult: investment multiplier (default 1.0)
    #   lev_mult: leverage multiplier (default 1.0)
    # Source label format: "EXT_{name}" — joined with "+" if multiple rules match.
    # Conflict resolution at engine: HIGHER wins across matching rules (same as
    # RSI×ADX cells). Hard caps via existing rsi_adx_multiplier_hard_cap +
    # _lev_hard_cap apply to the combined effective multiplier.
    # Cross-batch evidence at ship (May 24, post 3 LONG filters, May 22+ active window):
    #   L1b Ext +0.40-0.60% LONG: N=12 / 83% WR / +$256 / 2 dates / no NP losers
    #   L2a L1b × PairVol<0.95:   N=8  / 75% WR / +$165 / 2 dates / no NP losers
    #   L2b L1b × ADXΔ<0.3:        N=3  / 100% / +$274 / 1 date / no losers
    # All cells below the locked N≥30 gate but BE-compatible under new filter regime.
    # Operator-directed ship accepting the discipline override.
    # 2026-06-04 DEMOTED 2x->1x (all 3 LONG ext rules): cross-batch FULL pool turned
    #   negative — Ext0.4-0.6_L N=5/-0.216%/-$235, +QuietVol N=5/-0.352%/-$252 (both
    #   ✗ HARMFUL per Total$<0 N>=5 verdict); +SlowADX N=3/+$21 noise. Triggered by
    #   RENDER -$172 (2x-amplified; ~-$86 at 1x). Tags KEPT for tracking, sizing killed.
    #   "Caps for losers" — long side has no gross edge, so amplifying it is backwards.
    #   REVERT GATE: restore 2x only if Ext0.4-0.6_L reaches N>=15 fresh AND Total$>0.
    extension_multiplier_rules: List = []
    # ATR Multiplier Rules dimension REMOVED Jun 10, 2026 (was atr_multiplier_rules,
    # Jun 5 ship). All-time survivors of the Jun-10 guard stack in the Runner zone
    # (ATR 1.1-1.5): N=14, 36% WR, -$328 demux; current-era survivors above ATR 0.8:
    # N=0 (the ATR cap + fan + spike guards fence the zone). The May promotion was
    # earned under the pre-guard entry regime that no longer exists.
    # ATR-LOW fixed TP (Jun 5, 2026) — LONG exit, NOT a multiplier. entry_atr_pct <
    # atr_max AND pnl_pct ≥ tp_pct → exit "ATR_FIXED_TP L1" (a profit-LOCK; does NOT cut
    # DOA losers — those ride to stop). Locks the pop on the no-runner cohort.
    atr_low_fixed_tp_long_enabled: bool = False
    atr_low_fixed_tp_atr_max: float = 1.1  # entry_atr_pct strictly less than this = "pop-and-fade" cohort
    atr_low_fixed_tp_pct: float = 0.25     # LONG exits at this pnl% (profit-lock; never cuts a losing trade)
    # Jun 9, 2026 — "keep only unmatched longs". 4-cohort analysis (10-pool, current stack):
    # the LONG pattern library uniformly selects for LOSERS (every C/W pattern net-negative:
    # W6 −$574, W2 −$480, C7 −$261 demux), while TRULY-UNMATCHED longs (no C, no W) are the
    # edge (N=39, 85% WR, +$471 demux). When True, block any LONG matching ANY C or W pattern
    # (counter LONG_UNMATCHED_ONLY) — keep only the unmatched runner cohort. Pair this with
    # atr_low_fixed_tp_long_enabled=False: the fix-TP caps the pop-and-fade (matched) cohort,
    # but unmatched longs RUN (54% peak ≥0.40) — capping them at +0.25 strangles the edge, so
    # disable it and let them trail. (Coupling: if you re-enable matched longs, re-enable fix-TP.)
    long_unmatched_only: bool = False
    # BTC 1h Slope × BTC ADX Multiplier Rules (May 24 evening, 2026) — NEW dimension.
    # Sister to btc_rsi_adx_multiplier (existing) and extension_multiplier (today).
    # JSON-list format (not the string-CSV format used by btc_rsi_adx_multiplier_*)
    # because BTC 1h slope can be negative and the string-CSV format conflates `-`
    # as both negative-sign and range-separator. Each rule:
    #   {"name": str, "direction": "LONG"/"SHORT",
    #    "slope_min": float, "slope_max": float,
    #    "adx_min": float, "adx_max": float,
    #    "inv_mult": float, "lev_mult": float}
    # Both ranges half-open [min, max). HIGHER inv wins on overlap with other dims.
    # Source label: "BTC1H_{name}" (e.g., "BTC1H_M2_SHORT").
    # Cross-batch evidence at ship (May 24 full-pool structural analysis):
    #   M2 SHORT BTC 1h Slope 0 to +0.10 × BTC ADX 25-30:
    #     N=17 / 88% WR / +0.16% Avg / +$159 / 5 dates / 20% BE-rescue / NP 6%
    #   M3 LONG  BTC 1h Slope -0.20 to -0.10 × BTC ADX 18-25:
    #     N=17 / 76% WR / +0.17% Avg / +$156 / 4 dates / 23% BE-rescue / NP 6%
    # Both pass strict BE-floor + median-win + BE-rescue gates per CLAUDE.md
    # locked May 24 methodology.
    btc_1h_slope_btc_adx_multiplier_rules: List = []
    # Entry Quality Score multiplier (May 18 → REMOVED May 21): the Score-based 1D
    # multiplier dimension was retired after cross-batch evidence showed cells
    # decaying or showing no edge over baseline. See CLAUDE.md May 21 removal entry.
    # The Score block filter below (entry_quality_score_filter_enabled) is unaffected
    # — that's the entry-blocking mechanism, separate from the position-sizing one.
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
    # May 25, 2026 — ATR-normalized FE thresholds. Mirror of trailing_atr_multiplier
    # primitive. Formula: effective_threshold = max(fixed_threshold, entry_atr_pct × multiplier).
    # Prevents FE from firing on sub-noise moves on high-ATR pairs (e.g. on a 1.5%
    # ATR pair, 0.20% is sub-candle noise — at multiplier 0.50, FE waits for 0.75%
    # before firing). Cross-batch evidence (888-trade pool): post-FE give-up scales
    # monotonically with ATR (0.225pp at <0.3% ATR → 3.92pp at >1.5% ATR).
    # Counterfactual at 0.50: +$2,345 across 31 FE-skip trades. Set multiplier to
    # 0.0 to disable ATR floor entirely (preserves fixed threshold).
    fast_exit_l1_atr_multiplier: float = 0.50
    fast_exit_l2_atr_multiplier: float = 0.50
    # May 25, 2026 — ATR-floor caps. The ATR multiplier alone can drive the
    # effective FE threshold absurdly high on extreme-ATR pairs (e.g., XAN
    # at ATR 1.6% gave eff threshold 0.84% — peak never reached, FE never
    # fired, trade rode to SL). Cap bounds the floor: effective_threshold =
    # min(cap, max(fixed_threshold, entry_atr_pct × multiplier)).
    # Differentiated by tier to preserve FE1/FE2 semantics:
    #   - L1 cap 0.60 (fast-burst tier stays eager)
    #   - L2 cap 0.80 (slow-climber tier stays patient)
    # Cross-batch evidence: May 25 PM batch had 3 XANUSDT FE-saves (ATR 1.6%)
    # that would have ridden to SL without the cap. Set cap to 0 to disable.
    fast_exit_l1_atr_floor_cap_pct: float = 0.60
    fast_exit_l2_atr_floor_cap_pct: float = 0.80
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
    # C5 — Slow Climber Death (weak-trend slow bleed; common LONG failure)
    # SHORT mirror: weak-trend slow bleed up (slope ≥ -0.05% = flat or weakly bearish)
    pc_short_c5_pair_adx_max: float = 22.0
    pc_short_c5_adxd_max: float = 0.3
    pc_short_c5_ema20_slope_min: float = -0.05  # slope ≥ this = flat/weak (SHORT slow death zone)
    pc_long_c5_pair_adx_max: float = 22.0
    pc_long_c5_adxd_max: float = 0.3
    pc_long_c5_ema20_slope_max: float = 0.05  # slope ≤ this = flat/weak (LONG slow death zone)
    # C6 — Macro over-extended same direction (BTC about to climax, drags pair)
    # LONG: BTC RSI high + ADX strong + above 4hr trend = late BTC top
    # SHORT: BTC RSI low + ADX strong + below 4hr trend = late BTC bottom
    pc_short_c6_btc_rsi_max: float = 35.0
    pc_short_c6_btc_adx_min: float = 28.0
    pc_short_c6_btc_gap_max: float = -0.15  # BTC clearly BELOW 4hr trend
    pc_long_c6_btc_rsi_min: float = 65.0
    pc_long_c6_btc_adx_min: float = 28.0
    pc_long_c6_btc_gap_min: float = 0.15  # BTC clearly ABOVE 4hr trend
    # C7 — Pair Countertrend Bounce (NEW May 20 — dead-cat / failed-breakdown pattern)
    # LONG: pair deeply BELOW 4hr trend + declining slope + mid-range = dead-cat bounce LONG
    # SHORT: pair deeply ABOVE 4hr trend + rising slope + mid-range = failed-breakdown SHORT
    pc_long_c7_pair_gap_max: float = -0.50  # pair_ema20_ema50_gap ≤ this = deep countertrend
    pc_long_c7_ema50_slope_max: float = -0.05  # ema50_slope ≤ this = 4hr trend declining
    pc_long_c7_rngpos_min: float = 40.0  # RngPos ≥ this = bot longing mid-range bounce (not capitulation low)
    pc_short_c7_pair_gap_min: float = 0.50  # pair_gap ≥ this = pair stretched above 4hr trend
    pc_short_c7_ema50_slope_min: float = 0.05  # ema50_slope ≥ this = 4hr trend rising
    pc_short_c7_rngpos_max: float = 60.0  # RngPos ≤ this = bot shorting mid-range pullback (not blow-off top)
    # C8 — Oversold/Overbought Chop (NEW May 20 — hypothesis from C4 sub-pattern analysis)
    # Mechanism: pair entered at range extreme with sharp ADX accel during low-BTC-vol regime
    # where pair itself has NO clear direction (|gap|≤0.20). Bot reads RSI extreme + EMA cross,
    # signal fires, but chop dictates the trade gets squeezed (SHORT) or fades (LONG) instead
    # of riding continuation.
    # Cross-batch backtest (May 20): N=46 / 61% WR cross-batch (winner cohort).
    # Today's batch (May 20): N=3 SHORT / 0% WR / -$174 (loser cohort).
    # Observation-only — let live data resolve whether it's a real loser pattern or regime noise.
    pc_long_c8_rngpos_min: float = 75.0  # RngPos ≥ this = LONG at top of range
    pc_long_c8_adx_delta_min: float = 1.0  # ADXΔ ≥ this = sharp ADX acceleration
    pc_long_c8_pair_gap_abs_max: float = 0.20  # |pair_gap| ≤ this = pair NOT in clear trend
    pc_long_c8_btc_atr_max: float = 0.15  # BTC ATR ≤ this = low-vol regime (chop)
    pc_short_c8_rngpos_max: float = 25.0  # RngPos ≤ this = SHORT at bottom of range
    pc_short_c8_adx_delta_min: float = 1.0  # ADXΔ ≥ this = sharp ADX acceleration
    pc_short_c8_pair_gap_abs_max: float = 0.20  # |pair_gap| ≤ this = pair NOT in clear trend
    pc_short_c8_btc_atr_max: float = 0.15  # BTC ATR ≤ this = low-vol regime (chop)
    # C9 — Low-vol Countertrend Chop (NEW May 20-late, the "tight C4-LOSS" sub-pattern)
    # C9 = C4 base + MILD countertrend pair_gap. Catches losses where the bot
    # enters into a pair that's slightly against its own 4hr trend WHILE BTC is
    # in low-vol regime. Different from C7 (deep countertrend ≤ -0.50%) — C9 is
    # the milder variant where pair is BARELY going the wrong way but chop kills
    # follow-through.
    # Origin: today's C4 LONG deep-dive — EDEN losers had PairTGap -0.88 to -1.13%
    # (clearly negative) while FIDA/DASH had PairTGap +0.66/-0.085 (neutral-to-positive).
    # C7 misses these because EDEN slopes weren't ≤ -0.05%.
    # Observation-only. Per CLAUDE.md May 19 promotion gate (N≥30, WR≤40%, Avg≤-0.20%).
    pc_long_c9_btc_atr_max: float = 0.15  # BTC ATR ≤ this = low-vol regime
    pc_long_c9_btc_adx_max: float = 22.0  # BTC ADX ≤ this = no BTC trend conviction
    pc_long_c9_pair_adx_max: float = 25.0  # Pair ADX ≤ this = no pair trend conviction
    pc_long_c9_pair_gap_max: float = -0.10  # PairTGap ≤ this = pair countertrending vs LONG (mild)
    pc_short_c9_btc_atr_max: float = 0.15  # BTC ATR ≤ this = low-vol regime
    pc_short_c9_btc_adx_max: float = 22.0  # BTC ADX ≤ this = no BTC trend conviction
    pc_short_c9_pair_adx_max: float = 25.0  # Pair ADX ≤ this = no pair trend conviction
    pc_short_c9_pair_gap_min: float = 0.10  # PairTGap ≥ this = pair countertrending vs SHORT (mild)
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
    # Capitulation override GV CAP (May 27, 2026 — see CLAUDE.md).
    # Even when BTC is in capitulation (RSI<30 AND slope<0), CAP the override at this GV value.
    # Today's TON SHORT at GV 5.24 + capitulation conditions hit -$232 — extreme GV signals
    # blow-off-the-top volume that overpowers capitulation continuation. 0 = disabled (legacy behavior).
    # Default 2.0 = override fires only when GlobalVol ≤ 2.0; SHORT blocked when GV > 2 regardless.
    global_volume_max_short_capitulation_gv_cap: float = 0.0
    # Jun 5, 2026 — master toggle to REMOVE the SHORT capitulation override entirely.
    # When False, the override never fires: a SHORT at GlobalVol > global_volume_max_short
    # is ALWAYS blocked, regardless of BTC capitulation (RSI/slope). Rationale: the override
    # let through violent-spike SHORTs (the May-27 TON/FET/UNI/AVAX cohort) that bounced.
    # NOTE: this is a no-op unless global_volume_max_short > 0 (the cap must be active for
    # there to be an override to remove). Default True = legacy behavior (override active).
    global_volume_max_short_capitulation_override_enabled: bool = True
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
    # May 25: rescue MAX ceiling. Rescue clause only fires when GlobalVol < this
    # value. Above the ceiling but below global_volume_threshold_* = block (no
    # rescue). 0 = no ceiling (rescue fires across full <threshold zone).
    # Cross-batch evidence (CLAUDE.md May 25): GVol 0.60-0.70 LONG rescue zone
    # = N=36, 47% WR, -$717 (loser). GVol <0.60 LONG rescue zone = N=46, 67%
    # WR, +$62 (winner). Default 0.60 LONG isolates the loser sub-zone while
    # preserving the winner zone. AGT (today) confirmed: 37th data point in
    # 0.60-0.70, lost -$98 as predicted.
    global_volume_rescue_max_long: float = 0.0
    global_volume_rescue_max_short: float = 0.0
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
    
    # Cooldown after trade close (prevents immediate re-entry on same pair, win or loss)
    # CLAUDE.md May 26: cross-batch evidence on 919-trade pool shows 84 same-pair re-entries
    # within 5min after a WINNING trade had 61.9% WR but -$731 net (2.71:1 R:R loss asymmetry).
    # Fast-exit/trailing/BE wins lock tiny profit on fading momentum → re-entry catches the fade.
    # Applies to ANY close (was previously loss-only).
    cooldown_after_loss_minutes: int = 5  # Minutes to wait before re-entering same pair (any close)
    
    # Position limits
    max_open_positions: int = 100  # Max simultaneous open positions
    min_investment_size: float = 100.0  # Min investment per trade (USD)
    max_investment_size: float = 50000.0  # Max investment per trade (USD)
    max_holding_time_minutes: int = 180  # Max time to hold a trade (minutes), 0 = disabled
    no_expansion_minutes: int = 15  # Close if no expansion after N minutes (peak < BE trigger & current < BE offset), 0 = disabled

    # ── Liquidity-aware sizing (Jun 2, 2026 — see CLAUDE.md) ──────────────────
    # ① Per-pair liquidity cap: cap a single order's NOTIONAL to a small slice of
    #    the pair's 24h volume so the order is absorbable (slippage protection).
    #    max_notional = min(pct_of_pair_volume × pair_24h_vol, hard_ceiling).
    #    0 = disabled. Notional, not margin — what actually hits the book.
    max_notional_pct_of_pair_volume: float = 0.0  # e.g. 0.10 = 0.10% of 24h vol; 0 = off
    max_notional_hard_ceiling: float = 0.0  # flat $ notional backstop even on BTC-tier; 0 = off
    # ② Gross-notional cap: Σ(open notional) ≤ balance × max_gross_leverage.
    #    Portfolio liquidation/correlation guard (a -X% correlated dump costs
    #    X% × gross_leverage of the account). 0 = disabled.
    max_gross_leverage: float = 0.0  # e.g. 25.0; 0 = off
    # ③ Redeploy leftover: when ① throttles a slot below its equal-split slice,
    #    allow opening MORE positions (overrule max_open_positions up to the hard
    #    ceiling) to deploy the freed capital — gated by ② + tradeable margin.
    redeploy_leftover_enabled: bool = False
    max_open_positions_hard: int = 10  # absolute ceiling when redeploying


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
    pair_blacklist: str = ""  # Comma-separated pairs to exclude ENTIRELY (removed from the top-pair/volume universe)
    # Jun 3: comma-separated pairs to TRACK but NOT TRADE — they stay in the top-pair/volume
    # list (subscribed, scanned, displayed) but entries are blocked. Use for a pair you want
    # visible (e.g. BTCUSDT for reference) without opening positions. Distinct from pair_blacklist
    # (which removes the pair from the universe completely). Counter: PAIR_NO_TRADE.
    no_trade_pairs: str = ""
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
