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
    # ⚖️ Aug-21 2026 (operator-directed; gate 56, DECISION_LOG 2026-08-21 (2)): LONG side
    # JSON 0.15 → 0 (OFF). The gate's May-24 founding evidence (N=26, mixed matched
    # population, pre-runner exits) was never re-adjudicated and it shipped with an
    # UNRESOLVED "over-block risk" watchlist tag. First-ever cohort isolation (Aug-19/21
    # +19% BTC rally, 109 sole-blocked screen-passing episodes, 14 distinct hours): 72%
    # armed (+0.40 before −0.70) — capacity-honest foregone ≈ $1,200-1,800/37h vs the
    # actual window's 1 trade. Largest single blocker of the rally (30.7% of 48h).
    # Climax protection intact via 70-100:40 cross + ADX>40 (both KEPT on their own data).
    # 🔒 REVERT (manual, no auto): fills with stamped entry_btc_1h_slope > 0.15 —
    # N≥10 across ≥4 dates, WR≤45% ∨ Σ<0 → restore 0.15. Falling-BTC tripwire shared
    # with gate 51. ⚠ Single-phase override, acknowledged (window-units: one bullish
    # phase); the SHORT cap (0.1) untouched.
    btc_1h_slope_max_long: float = 0.0
    btc_1h_slope_max_short: float = 0.0
    # Jun 3: minimum BTC 1h slope FLOOR (block entries when 1h slope is too steeply
    # NEGATIVE = shorting into a steep 1h crash = exhaustion/bounce). 0 = disabled;
    # a negative value activates. SHORT cross-batch: 1h slope < -0.60 = 0W/4L
    # (SEI, XRP, BTC, JTO). LONG left disabled (its loser zone is the FLAT band, not
    # the steep band — a different mechanism, not shipped here).
    btc_1h_slope_min_long: float = 0.0
    btc_1h_slope_min_short: float = 0.0
    # Jul 5: BTC 1h slope DEAD-BAND for LONGs — block a momentum LONG when |BTC 1h slope|
    # < this value (flat hourly = no macro carry → alt breakout has no sponsor → DOA).
    # The FLAT-band loser zone the Jun-3 comment above predicted, now measured: baseline
    # ML |1h|<0.05 = 7/43%WR/−0.19% avg vs flanks 27/93%WR (pullback flank is the shipped
    # 1hPullback 2× cell); fresh 07-05 both losers in-zone (RPL −0.016, ETHFI +0.024);
    # 6 of 8 cross-era losers, 6 pairs, monotone-worse toward 0, Fisher p≈0.01.
    # ⚠ N=9 discipline-override ship (operator-directed) — tight phantom revert:
    # PASS:LONG_BTC1H_DEADBAND ≥60% WR on N≥10 → set back to 0. 0 = disabled.
    long_btc_1h_deadband: float = 0.0
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
    # 🛡 FAKE_BULL_GUARD — ✝ REVERTED 2026-08-16 by its own locked gate 47, 2 days after the
    # Aug-14 override ship (N=10 in-sample: 3W/7L·−$818). Forward replay of the first 12 blocked
    # would-be longs (price-stamped block logs → kline replay): 6W/6L = 50% would-be winners AND
    # phantom-net +4.65pp — BOTH revert legs tripped (already tripped at the first 6: 3W/3L,
    # net +1.37). The winner-rate leg is MODEL-FREE (+0.40 reached before −0.70). The mined
    # 4D pattern did not survive out-of-sample — textbook N=10 override fate; the tight gate
    # bounded the damage to ~2 days of phantom-only cost. Machinery kept (fields/UI/counter/
    # engine block) for a future re-hearing ONLY with N≥30 cross-batch evidence per the locked
    # promotion gates — do NOT re-enable on the original evidence. DECISION_LOG 2026-08-16.
    fake_bull_guard_enabled: bool = False
    fake_bull_guard_bull_pct_max: float = 71.0   # dormant (see revert note above)
    fake_bull_guard_tg_max: float = 0.01         # dormant (see revert note above)
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
    # Jul 13: GAPFLAT PROBE — real-data A/B on the #1 entry blocker (PAIR_EMA_GAP_NOT_EXPANDING,
    # ~22% of all blocks, never counterfactual-tested; blocks aren't stored so no offline re-sim is
    # possible). When enabled, a momentum LONG failing ONLY the gap-expanding check (passes the whole
    # rest of the ladder + engine gates) opens as a REAL order at ~1x effective leverage (invest_mult
    # x lev_mult x 20x base), tagged cell_src=GAPFLAT_PROBE (own Multiplier-Cell row + CSV column;
    # EXCLUDED from screen anchors + the MARGINAL/STRICT relaxation A/B). Caps: only when the book is
    # light (last 2 slots always reserved for real signals), max_open concurrent, max_per_day budget.
    # LONG-only; shorts stay hard-blocked. 🔒 GATES (pre-committed): at N>=30 probes (>=10 dates) —
    # WR>=60% AND avg>=+0.15% => discuss relaxation (probe rows carry all entry columns for the
    # sub-cohort hunt); WR<=45% OR avg<0 => filter vindicated, probe off permanently.
    gap_probe_enabled: bool = False
    gap_probe_invest_mult: float = 0.5   # x equal-split invest (~$385 ticket)
    gap_probe_lev_mult: float = 0.05     # x 20x base = 1x effective leverage
    gap_probe_max_open: int = 3          # concurrent probes (Jul 13 PM operator: 1→3, and the per-day budget REMOVED — at ~$4/trade a daily cap only slows the N≥30 clock; slot guard + concurrency are the protections)
    # Jul 29: FLIPGATE PROBES (operator-directed after the flip-drought forensics). The flip
    # sleeve produced 0 trades since Jul-8; down-window cross-ref showed candidates DO exist
    # inside BTC down-windows (313 vetoes/7d in-window) but die on three June-fitted secondary
    # floors (QUALITY 85 / RSI_MIN 72 / TRENDGAP 54 = 67% of in-window kills; TRENDGAP's
    # registered Sole-growth reopen trigger fired). A flip-SHORT candidate sole-blocked by
    # exactly ONE gate listed below opens at gap-probe sizing (~1x eff) under its own cell tag
    # (FGP_QS / FGP_RSI / FGP_TG) — per-gate cohorts, per-gate locked N>=30 verdicts.
    flipgate_probe_enabled: bool = False  # ✝ PROBE FLEET RETIRED 2026-08-11 (operator: near-zero flow, no value; ADXMAX math-dead for its relaxation gate at 12W/13L). Gap-Expand Probe table deleted same day; engine gates stay dormant.
    flipgate_probe_gates: str = "FLIP_SHORT_QUALITY,FLIP_SHORT_RSI_MIN,FLIP_SHORT_BTC_TRENDGAP"
    flipgate_probe_max_open: int = 3     # concurrent FGP_* probes (shared across the 3 cohorts)
    # Jul 13 PM: GAPMIN PROBE — sibling of the GAPFLAT probe, on the #2 LONG blocker
    # (PAIR_EMA_GAP_MIN, 2,101 L blocks). A momentum LONG whose EMA5-8 gap sits in
    # [gapmin_probe_floor, ema_gap_threshold_long) — accelerating (passed gap-expanding)
    # but still SMALL = "young trend caught early" — opens as a 1x GAPMIN_PROBE. Cohort
    # purity: gap-flat candidates are excluded (GAPFLAT owns those; double-relaxation
    # candidates stay blocked). Shares gap_probe_invest_mult / gap_probe_lev_mult sizing
    # + the last-2-slots guard. Same 🔒 gates: N>=30 probes (>=10 dates) -> WR>=60% &
    # avg>=+0.15% = relaxation discussion; WR<=45% or avg<0 = filter vindicated, off.
    gapmin_probe_enabled: bool = False
    gapmin_probe_floor: float = 0.01     # band floor (Jul 14 operator: 0.02→0.01 — Funnel v2 [<floor] variant led the GAP_MIN sole split 57v32 ~half-day in; early open of the leading door, probe-scope risk only, [0.01-0.02) sliced as its own sub-band at verdict. Jul 13 PM: 0.04→0.02 "test it entirely"); below floor = blocked as always
    # Jul 14 — SLOPEGATE PROBE (operator-directed same-night ship). The BTC 5m flat
    # dead-band (macro_trend_flat_threshold_long/short = 0.02/0.03, Apr-14 calibration
    # on N=4) was measured killing ~133 signal-found candidates (~43 distinct
    # opportunities) per day in live logs — the true #1 marginal blocker, invisible to
    # Funnel v2 (engine gate). With the probe on, a candidate hit ONLY by this gate
    # (and passing every other engine gate) opens as a 1x-effective SLOPEGATE_PROBE
    # (shares gap_probe_invest_mult/lev_mult sizing; own max_open; last-2-slots guard).
    # 🔒 Gates per side at N>=30 (>=5 dates — flow is ~40/day so dates accrue fast):
    # WR>=60% & avg>=+0.15% -> dead-band relaxation discussion; WR<=45% or avg<0 ->
    # gate vindicated (its FIRST evidence-grade validation since April), probe off.
    slopegate_probe_enabled: bool = False  # ✝ PROBE FLEET RETIRED 2026-08-11 (operator: near-zero flow, no value; ADXMAX math-dead for its relaxation gate at 12W/13L). Gap-Expand Probe table deleted same day; engine gates stay dormant.
    slopegate_probe_max_open: int = 3    # concurrent SLOPEGATE probes (shared across both directions)
    # Jul 15 — RSIADX PROBE (probe #4, operator-directed day-2 early open of the locked
    # >=3-dates gate; transparent override, probe-scope risk). The Mar-27 short RSI×ADX
    # cross-filter ("30-35:25,35-50:30") is the #1 short-side sole blocker under the
    # uncensored Funnel v2 (840 soles = 66% of short soles in 2 days; [35-50:30] = 617)
    # with ~$6 of contested April evidence (Apr-17: "funnels entries into toxic
    # high-ADX zones — inverting good logic" vs Apr-9 "benign gate"). Candidates whose
    # ONLY ladder fail is this filter open as 1x RSIADX_PROBE (both directions; the
    # LONG rule 60-65:0-25 rides along at low flow). 🔒 Gates PER SIDE at N>=30
    # (>=5 dates): WR>=60% & avg>=+0.15% -> relaxation discussion (slice [35-50:30] vs
    # [30-35:25] vs LONG rule at verdict); WR<=45% or avg<0 -> filter vindicated
    # (first evidence-grade validation since March), probe off.
    rsiadx_probe_enabled: bool = False  # Jul 27 PM: verdict-closed/graduated (promotion package) — default matches ship state
    rsiadx_probe_max_open: int = 3       # concurrent RSIADX probes (shared across both directions)
    # Jul 20 — RSIADX·SHORT VERDICT EXECUTED (operator-directed early close at 20/30,
    # practically locked: 20·20%·−0.358% (Σ−7.16%) · 45% DOA · 4 dates — escape needs 10
    # straight wins averaging +0.72% (2x the typical short win) and even 10/10 only reaches
    # 46.7% WR. The Mar-27 cross-filter's SHORT rule is VINDICATED with real money.
    # Anatomy for the record: aligned-DOWN entries (BTC 5m & 1h both falling) 1W/8L —
    # mid-decline shorts are late; winners were bounce-fades (5m-up/1h-down 3/3) and
    # range-floor breaks. LONG side (12·50%) keeps collecting to its own N>=30.
    rsiadx_probe_short_enabled: bool = False
    # Jul 15 — DEADBAND PROBE (probe #5, operator-directed HALF-OPEN of the Jul-5
    # LONG_BTC1H_DEADBAND gate). The raw phantom revert gate technically fired
    # (24·63%·+0.145% >= "60% on N>=10") but was HELD: 20/24 single-day, ~4-5
    # independent episodes, 5 rows equity-perps (now untradeable), and the halves
    # split — flat-UP [0,+0.05) 14·79%·+0.284% vs flat-DOWN 10·40%·−0.05% — while
    # the HISTORICAL pool refutes flat-up (6·50%·−0.08%). Probe: LONGs with BTC 1h
    # slope in [0, +deadband) open as 1x DEADBAND_PROBE (every other gate still
    # applies; 0/14 phantom overlap with the 5m SLOPEGATE band); flat-down keeps
    # blocking + seeding PASS phantoms. 🔒 Gates at N>=30 (>=5 dates):
    # WR>=60% & avg>=+0.15% -> promote the half-open to FULL sizing (uncap);
    # WR<=45% or avg<0 -> band re-closes, probe off.
    deadband_probe_enabled: bool = False  # Jul 27 PM: verdict-closed/graduated (promotion package) — default matches ship state
    deadband_probe_max_open: int = 3     # concurrent DEADBAND probes (LONG-only by construction)
    # Jul 15 — RSICEIL PROBE (probe #6, operator-directed). The LONG RSI ceiling (65) was
    # last set May-4 (Phase-1c, PRE-UNMATCHED-only longs + old exit stack) and the zone
    # above it is DARK: N=2 lifetime trades with entry RSI 65-70. Current-era gradient
    # RISES toward the ceiling (55-60: 62% / 60-63: 77%·+0.26% / 63-65: 73%) — the filter
    # cuts where the book is strongest, on a dead configuration's evidence. Probe: LONGs
    # whose ONLY ladder fail is RSI in (65, ceiling] open as 1x RSICEIL_PROBE (every other
    # gate applies); RSI > ceiling stays blocked. 🔒 Gates at N>=30 (>=5 dates):
    # WR>=60% & avg>=+0.15% -> ceiling-raise discussion (65->70 full size);
    # WR<=45% or avg<0 -> ceiling vindicated, probe off. Verdict protocol applies
    # (slice 65-67 vs 67-70 x other probes' dimensions).
    rsiceil_probe_enabled: bool = False  # Jul 27 PM: verdict-closed/graduated (promotion package) — default matches ship state
    rsiceil_probe_max_open: int = 3      # concurrent RSICEIL probes (LONG-only by construction)
    rsiceil_probe_ceiling: float = 70.0  # probe band upper bound: RSI in (momentum_long_rsi_max, this]
    # Jul 20 — SLOPEGATE·LONG VERDICT EXECUTED (✗ VINDICATED, closed at 27/30 with the avg
    # arm mathematically locked: 27·44.4%·−0.376% — even 3 max-size wins leave avg<0; full
    # separation pass in DECISION_LOG 2026-07-18/20). LONG side of the probe is OFF (the
    # 5m dead-band gate resumes blocking LONGs normally); SHORT side keeps collecting (2/30).
    slopegate_probe_long_enabled: bool = False
    # Jul 20 — GMINFLAT PROBE (probe #7, operator-directed). Target = PAIR_EMA_GAP_MIN[flat]
    # sub-rule: gap in [floor, threshold) AND gap-flat — the "flat+small" cohort-purity class
    # both GAPFLAT and GAPMIN deliberately excluded. Funnel v2: 2,213 LONG soles (23% of ALL
    # long soles — the single biggest unprobed long blocker) + 511 short soles. Candidates
    # sole-blocked by [flat] open as 1x GMINFLAT_PROBE, BOTH directions. 🔒 Gates PER SIDE
    # at N>=30 (>=5 dates): WR>=60% & avg>=+0.15% -> relax discussion; WR<=45% or avg<0 ->
    # cohort-purity block vindicated, probe off.
    # ⚠ DEPENDENCY (code-review): the [flat] suppression is only reachable while BOTH
    # gap_probe_enabled (GAPFLAT) AND gapmin_probe_enabled are True — switching either off
    # (e.g. executing their verdicts) silently starves this probe to zero flow (candidates
    # stay safely blocked; nothing admits untagged). Re-check this flag at those verdicts.
    gminflat_probe_enabled: bool = False  # Jul 27 PM: verdict-closed/graduated (promotion package) — default matches ship state
    gminflat_probe_max_open: int = 3     # concurrent GMINFLAT probes (both directions combined)
    # Jul 20 — ADXMAX PROBE (probe #8, operator-directed; RSICEIL-clone). The pair-ADX
    # ceiling (L30/S35) is an old calibration with a DARK zone above it; Funnel v2 shows
    # 1,335 LONG + 1,148 SHORT soles, and the Jul-20 batch WR gradient RISES into the
    # ceiling (pADX 28-30 = 80% WR). Probe: candidates whose ADX sits in (per-side max,
    # per-side probe ceiling] open as 1x ADXMAX_PROBE; above the probe ceiling stays
    # blocked. 🔒 Gates PER SIDE at N>=30 (>=5 dates): WR>=60% & avg>=+0.15% ->
    # ceiling-raise discussion; WR<=45% or avg<0 -> ceiling vindicated, probe off.
    # Verdict slices 30-32 vs 32-35 (long) per protocol.
    adxmax_probe_enabled: bool = False  # ✝ PROBE FLEET RETIRED 2026-08-11 (operator: near-zero flow, no value; ADXMAX math-dead for its relaxation gate at 12W/13L). Gap-Expand Probe table deleted same day; engine gates stay dormant.
    adxmax_probe_max_open: int = 3       # concurrent ADXMAX probes (both directions combined)
    adxmax_probe_ceiling_long: float = 35.0   # LONG probe band: ADX in (momentum_adx_max_long, this]
    adxmax_probe_ceiling_short: float = 40.0  # SHORT probe band: ADX in (momentum_adx_max, this]
    # ADXMAX2 (Jul 21, 2026 — probe #10, LONG-only): SECOND rung of the LONG pair-ADX
    # ladder — band (adxmax_probe_ceiling_long, this] = (35, 40]. Parallel to (not instead
    # of) the 30-35 first rung: disjoint populations, separate cohort tag, each band faces
    # its own N>=30/>=5-date gates independently (no post-hoc merging). Rationale: batch
    # ADX gradient rises into the 35 ceiling (28-30: 80% / 30-33: 100% / 33-35: 100%;
    # ADXMAX·L first rung 3/3 W early) and >35 is fully dark for LONGs; theory two-sided
    # (very strong trend vs late/exhausted trend) — measure, don't assume. If the first
    # rung lands ✗ the gradient thesis dies and this band's tuition is accepted moot cost.
    adxmax2_probe_enabled: bool = False  # Jul 27 PM: verdict-closed/graduated (promotion package) — default matches ship state
    adxmax2_probe_max_open: int = 3           # concurrent ADXMAX2 probes (LONG-only)
    adxmax2_probe_ceiling_long: float = 40.0  # LONG band 2: ADX in (adxmax_probe_ceiling_long, this]
    # Jul 24 SPIKE_CHASE probe (#11, LONG-only, operator-directed): NEW ENTRY CLASS — chase a
    # single-candle 5m RSI explosion (jump >= rsi_jump pts in ONE candle with rsi_prev1 <= prev_max).
    # Calibrated on MIRA Jul-22 00:00 UTC: RSI 49->82 (+33) on the +1.84% discovery candle with
    # +17% still ahead; pre-pump chop never jumps >9. Bypasses the signal ladder BY DESIGN (these
    # candles have no fan yet; RSI>65 blocks all follow-ups). Fires only when the ladder produced
    # NO signal. Right-tail cohort — pre-registered read weights avg over WR (CURRENT_STATE #11).
    spike_chase_probe_enabled: bool = True   # ⚠ MISNOMER — since the Jul-27 graduation this is the MASTER TRIGGER for the full-size 🚀 chase/fade program (scanner pump branch + top-50 hook detection), NOT a probe slot. Aug-10 "retirement" to False silently killed all fades/chases (review-caught, reverted same day). Must stay True while the program runs; the frozen probe-era cohort row is display-only.
    # 🛑 Aug-21 2026 (operator-directed; DECISION_LOG 2026-08-21): dedicated CHASE kill-switch —
    # the species had NO own flag (its probe flag became the program master trigger at
    # graduation; disabling via the stretch-guard was rejected as semantic abuse — "more
    # professional"). Mirrors spike_fade_enabled / spike_bounce_enabled. JSON ships FALSE:
    # full-size chases 0W/3L −$632 (SYRUP/DODOX Jul-27 bear-hour · MORPHO Aug-21 in
    # HEALTHY_BULL through the 1.5×ATR guard — the "dormant at 1.5" claim was leaky), lifetime
    # honest 3W/8L, non-bull 0W/9L. Both entry sites gated (scanner + top-50 hook); every
    # would-be fire still logs [SPIKE_ROUTER_BLOCK] + counter SPIKE_CHASE_DISABLED, so revival
    # evidence accrues for free. FADE/BOUNCE untouched. Revival = fresh probe proposal (11b).
    spike_chase_enabled: bool = True   # default True = legacy; JSON carries the live value (False since Aug-21)
    spike_chase_probe_max_open: int = 3          # same slot cap as the rest of the fleet
    spike_chase_probe_rsi_jump: float = 25.0     # min single-candle RSI(12) jump (pts)
    spike_chase_probe_rsi_prev_max: float = 55.0 # from-quiet condition: prev candle RSI <= this
    spike_chase_probe_rsi_prev_min: float = 35.0 # Jul 24 PM (FHE dead-cat fire): quiet FLOOR — prev RSI in
                                                  # [prev_min, prev_max] = resting band. FHE fired from RSI ~12
                                                  # (active markdown; bounce = knife-catch, not discovery).
                                                  # MIRA prev 49.1 (14-pt headroom); 35 not 40 keeps shakeout-
                                                  # wick launches (high-30s) alive.
    spike_chase_probe_min_vol_ratio: float = 5.0  # Jul 24 PM: discovery-candle volume >= this x its prior
                                                  # 20-candle avg ("attention arrived"). MIRA discovery 59.6x
                                                  # (12x headroom); pre-pump chop max 5.7x; USDCUSDT fire was
                                                  # 2.39x. Free from klines already fetched (col 5).
    spike_chase_probe_min_candle_pct: float = 0.5 # Jul 24 PM: discovery candle must MOVE PRICE >= this %.
                                                  # RSI is scale-free: USDCUSDT fired on a +0.01% wiggle
                                                  # (RSI 35->71 on stablecoin noise). MIRA's real discovery
                                                  # candle was +1.84% (3.7x headroom); pre-pump chop was
                                                  # +-0.1-0.3%. Kills flatline/stable noise, never real pumps.
    # Jul 24 PM FULL-UNIVERSE SPIKE SCANNER (operator-directed same day; ZERO-ENGINEERING-RISK
    # revert = this toggle OFF): extends the SPIKE_CHASE trigger beyond the top-50 to ALL
    # eligible USDT perps (same protective screens: new-listing days / Alpha / coin-only /
    # blacklist / no-trade). Rationale (MIRA forensics): pre-pump daily vol $1.2-4.4M = rank
    # ~200+, invisible to the top-50; the pump promotes the pair with one-pump lag. Extended
    # pairs get ONLY the spike door (never the ladder); fires route into the SAME
    # SPIKE_CHASE_PROBE cohort/caps/gates. Piggybacks the scan loop; fail-silent.
    spike_scanner_enabled: bool = True
    # ⚡ RAISED $1M→$2M 2026-08-04 PM (operator-directed): restores the STRUCTURAL floor that
    # existed pre-LIQ2 (0.1% cap × $100 min-investment made sub-$2M pairs unbuyable — B1 has
    # ZERO trades under $2M; the Aug-3 LIQ2 raise removed it by accident, admitting the $1-2M
    # sliver = 7 fires · 1W/6L · −$71 · worst avg −0.395%, incl. all three instant gap-throughs
    # FRAX 7s / SPELL 24s / GTC + the stablecoin-wobble class). $2M = the ONLY line whose
    # blocked side is negative everywhere it exists ($3M/$5M lines fail cross-batch: blocked
    # side POSITIVE in B2 +$30/+$36 — EGLD/KAS/QNT/KSM winners live at $2-3M). Cost: one
    # forfeited winner ever (IO +$8.89). NOT a fitted boundary. Transparency: shipped mid-
    # LIQ2-window — the gate's conditions are unchanged but its remaining fires get cleaner
    # by construction (logged in DECISION_LOG 2026-08-04 (4)).
    spike_scanner_min_vol_usd: float = 2000000.0  # dead-book floor: skip pairs under $2M 24h vol
    spike_scanner_max_pairs: int = 400            # universe cut (top-N by volume incl. the top-50)
    # ══ Jul 27 — 🚀 SPIKE FULL SHIP (operator-directed one-ship: both species full size).
    # Trigger fires (legs 1-5) then pair ADX ROUTES the direction (leg 6, 10/10 lifetime:
    # riders all <=20.2, duds all >=29.6): ADX <= max_adx -> SPIKE_CHASE LONG at full size;
    # ADX > max_adx -> SPIKE_FADE SHORT (4/4 backtest, avg +0.31, max adverse +0.49).
    spike_chase_max_adx: float = 30.0             # leg 6 router cut (empty 20-29 gap; default 30)
    spike_chase_max_stretch_atr: float = 1.5  # ✗ TRIPWIRE FIRED 2026-08-10: 5-fire Σ −$209 < 0 → mechanical revert 2.5→1.5 (MIRA −$228 @2.46×ATR = the driver); ALL 6 lifetime full-size chases entered 1.77-2.47 → species DORMANT at 1.5; revival = fresh probe proposal      # Jul 30 EXTENSION GUARD (0=off): block CHASE when
    # (price-EMA5)/EMA5 % > mult x pair ATR%. Chase-only — fade winners ARE the stretched ones
    # (PROM 3.4xATR +1.16). ⚡ RAISED 1.5→2.5 Aug-3 (operator-directed): the founding 0/8 evidence
    # was demolished by three confounds — 6/8 non-bull regime (router now seals it), AKT was an
    # old-exit clock-kill (trails to ≈+0.55 today), and ALL 8 entered via the 20s maker-delay
    # (fixed Jul-30 direct-taker; fill-stretch was delay-INFLATED, so the 1.5 line was calibrated
    # on contaminated data). Chase-as-currently-built has zero record; CloudWatch ledger since
    # Jul-30: 6 bull-eligible candidates blocked at ratios 1.7-3.9 (band ≤2.5 = 3 of 6, ~0.6/day).
    # 🔒 TRIPWIRE (mechanical): at N=5 post-raise chases, Σ pnl% < 0 → revert to 1.5.
    # 🔒 VERDICT: N>=15 · >=5 dates, slice 1.5-2.0 vs 2.0-2.5 — redraw the line from clean
    # taker-era data; WR≤45% ∨ avg<0 → guard re-tightens to 1.5. The Jul-30 post-guard eval
    # (N>=8 net-neg → probe size) still stands on top.
    spike_invest_mult: float = 2.0                # CHASE sizing: Inv 2x of equal-split base
    spike_lev_mult: float = 1.0                   # CHASE leverage mult (1.0 = confidence base 20x)
    spike_fade_enabled: bool = True               # master kill toggle for the fade species
    spike_fade_invest_mult: float = 2.0           # ⚠ OPERATOR OVERRIDE OF FIRED GATE 2026-08-05:
    # the Jul-30 tight-revert gate FIRED (EVAA −3.07% ≤ −1.0% single-fire condition → mechanical
    # 2x→1x executed) and the operator explicitly overrode it back to 2x same day, informed of the
    # tail math (EVAA at 2x+0.2%cap = −$255 vs ≈−$64 at 1x+0.1%) and that tripwire auto-disable is
    # now OFF (sizing = the only fade tail protection). Logged DECISION_LOG 2026-08-05.
    # 📋 WATCHLIST (operator directive Aug-5 PM: NO automatic gate — review item only): quant
    # FLAGS for operator decision on any single 2x fade ≤ −1.0% (gap-through class; −0.75 was
    # miscalibrated — ordinary thin-pair slippage lands −0.76/−0.82) or rolling-15 Σ pnl% < 0
    # (5-fire windows false-fire 8/18 in a +$267 era; 15-fire: 0/8). Note: 2x has NEVER bound —
    # every era fade was liquidity-cap-bound (marginal $0 in 22 fires); binds only >$36M-vol pairs.
    spike_fade_lev_mult: float = 1.0              # FADE leverage mult (1.0 = base 20x)
    spike_fade_sl_pct: float = -1.50              # FADE fixed SL — NO ATR widening. Aug-10: −0.70→−1.50
                                                  # (operator ship; −0.70 sat INSIDE the spike candle's wick:
                                                  # 8/9 stopped fades reverted after the stop, B1+B2 CF Δ+$400
                                                  # WR 74→84%, SUSHI −1.46% wick +1.6% revert = same-day OOS
                                                  # confirm; only CHIP-class continuation pays more. N=7 SL
                                                  # losers = DISCIPLINE-OVERRIDE, tight revert: saved-band
                                                  # (adverse ∈(−0.70,−1.5]) fresh N≥6 WR<50%∨Σ<0 → −0.70)
    # ⚙️ Aug-5 OPERATOR DIRECTIVE after the EVAA event ("re-enable it, and disable the
    # auto-disable feature"): tripwires are now ALERT-ONLY by default — a breach logs
    # CRITICAL [.._TRIPWIRE] but no longer flips the species off. Tail protection = the
    # bd13/bRSI gates + the override-class re-revert gates on 2x/0.2% (operator overrode the
    # fired sizing gates same day — at 2x+0.2% an EVAA-class fire costs ≈ −$255, not −$64).
    # Set True to restore the Jul-27 auto-disable behavior (applies to fade AND bounce).
    spike_tripwire_autodisable: bool = False
    # 🛡 Aug-10 FRESH-BREAKOUT GUARD (operator "ship B" after the fade deep-dive; zone leg on
    # watchlist): block a FADE when the pair spiked from a LOW-RSI base (rsi_prev < min) AND is
    # NOT a crash-extreme (EMA13-50 gap > pgap_min) — a low-base spike on a non-crashed pair is
    # the START of a move (fresh breakout / squeeze ignition), not an exhaustion; shorting
    # beginnings is how EVAA/ETHFI/DEEP/SAND happened (the DOA 6-40s death class lives here).
    # Stack-screened evidence: CUR blocked 17·41%·−$353 / after 26·69%·+$775; B1 blocked 3·+$9
    # (free); combined blocked 20·45%·−$344. Thresholds: 44 ≈ pooled rsi_prev median (B1 42.0 /
    # CUR 43.9); −0.40 = crash-extreme boundary (VANRY −0.89 / PIPPIN −0.57 kept).
    # 🔒 READS: ① post-ship fades N≥10 → WR≥65 ∧ Σ>0; ② blocked-side re-sim N≥8 → would-be
    # WR≥55 ∨ Σ>0 reverts. Counter SPIKE_FADE_FRESHBREAK (logs entry px). rsi_prev_min 0 = off.
    spike_fade_fb_rsi_prev_min: float = 44.0
    spike_fade_fb_pgap_min: float = -0.40
    spike_fade_tripwire_pct: float = -2.5         # tripwire threshold: any fade closing <= this means the
                                                  # price GAPPED THROUGH the -1.50 stop (squeeze
                                                  # signature). Aug-10 review fix: -1.5 -> -2.5 after the
                                                  # SL widened to -1.5 (equal values = alarm on EVERY
                                                  # ordinary stop + a species-kill landmine if autodisable
                                                  # ever returns; ~1pp spacing restored, mirroring the old
                                                  # -0.70/-1.5 pair) -> flips species OFF only
                                                  # if spike_tripwire_autodisable (default Aug-5:
                                                  # CRITICAL alert-only, species stays on)
    # 🔒 SPIKE PROFIT LOCK (Aug-3, #24b variant-② verdict at fade-capture N=13: Saved $87 /
    # Killed $0, Saved>=2xKilled bar passed on BOTH measured species; winners' post-touch dips
    # bottom at -0.09 -> the -0.15 floor sits below the band with 6bp margin). Once a spike
    # trade's peak P&L touches the arm, its fixed SL (-0.70 fade/bounce, -1.2 chase) tightens
    # to the lock level; close reason SPIKE_LOCK L1 (SPIKE_ prefix = post-exit whitelists
    # auto-covered). ALL spike species uniformly — fade/bounce measured, chase =
    # mechanism-transfer (own tally row; 7/8 lifetime chases peaked 0.00 so it rarely arms).
    # REVERT GATE (pre-committed): at N>=10 SPIKE_LOCK fires — revert if >=60% of locked
    # trades' post-exit continuation recovers to >0 (peak-first) OR cumulative delta vs the
    # fixed-SL counterfactual < 0.
    spike_lock_enabled: bool = True
    spike_lock_arm_pct: float = 0.20              # arm when peak P&L touches this (0 = off)
    spike_lock_sl_pct: float = -0.15              # armed stop replaces the species fixed SL
    spike_lock_exempt_fade: bool = True           # Aug-10 PM (operator): FADES EXEMPT from the lock —
                                                  # under the −1.5 SL the lock's save class evaporated
                                                  # (its 2-saves record was vs dying at −0.70): belock-grid
                                                  # CF on all covered kept fades = 0 saves / 4 kills −$90
                                                  # (SENT +0.21→−0.15 · HOLO +0.79→−0.15 · ZEN +0.06→−0.15
                                                  # · PROM live −0.19 vs ~+0.5); no lock = best config
                                                  # (+$1,451 vs live +$1,390). Multi-wave pumps arm on
                                                  # wave-1 pullback and eject before the wave-2 wick the
                                                  # wide SL holds through. Chase/bounce keep the lock.
                                                  # 🔒 REVERT (→False) if ≥2 fresh fades arm ≥+0.20 then
                                                  # run to the full −1.5 (re-enable at floor −0.40 instead)
    # 🛡 Aug-11 BROKER BACKSTOP — resting exchange-side STOP_MARKET per LIVE position via the
    # Algo Order API (POST /fapi/v1/algoOrder; -4120 was Binance's MANDATORY conditional-order
    # migration, NOT an account defect — DECISION_LOG 2026-08-11 (6)). Dead-man's brake for
    # deploy/crash/WS-starve/ban windows: WIDE by design (never races the software stops; fires
    # only when the bot cannot act). Fires reconcile as close_reason BACKSTOP_STOP (own row in
    # close-reason tables + post-exit regret whitelists — ANY live fire = investigate the outage).
    # Paper mode: fully dormant. Arm ONLY at go-live, after the testnet contract test passes.
    broker_backstop_enabled: bool = False
    broker_backstop_pct: float = 2.5   # trigger distance (% of entry px) ≈ caps bot-dead tail at ~50% of one slot's margin at 20x
    # ⭐ OPTION-D 3-layer exit for SPIKE_CHASE longs (replaces the normal long exit stack):
    # L1 fixed SL (MIRA-1 wicked -0.71 pre +17.4; SWARMS breathed -0.75 pre +0.91 — two
    # documented winner-breaths through -0.70; dud premium ~0.5pp/fire accepted).
    spike_sl_pct: float = -1.2                    # L1 — fixed, NOT ATR-derived
    spike_rsi_cool_arm: float = 75.0              # L2 arm: 5m RSI(12) >= this after entry
    spike_rsi_cool_drop: float = 10.0             # L2 exit: RSI <= ride-max minus this (relative —
                                                  # pumps pin 82-98; -8..-12 plateau robust)
    # L3 insurance floors — SAME trigger:offset format as the book's Profit Floor Ladder
    # (floor = trigger − offset; parsed by services/hard_tp_ladder.py, ratchet-only, NEVER
    # a TP — collapse insurance). UNARMED (RSI never confirmed >= arm): tight normal rungs
    # (SWARMS live proof: the 1.25 rung took it at +1.0 floor).
    spike_ladder_unarmed: str = "1.25:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80"
    # Jul 28 EXIT PATCH (operator-directed after full-size fires 0W/6L exposed two holes):
    # ① mid-zone trail — unarmed chases peaking in [arm, 1.25) had ZERO protection (AKT
    #   +0.83 peak → −0.45 NO_EXP). Once unarmed peak >= this, run the standard runner
    #   trail (1×ATR giveback, BE-lock +0.10 — reuses runner_trail_atr_mult/be_lock);
    #   RSI-arm hands off to the wide envelope so true tails keep their room. 0 = off.
    spike_trail_arm_pct: float = 0.40  # Aug-11: 0.45→0.40 (operator; alignment with the system-wide 0.40 trail arm — zero lifetime chases peaked in [0.40,0.45), pure consistency)
    # ② stale-spike kill — a spike with no follow-through is dead by construction, yet
    #   QTUM/KAS sat 3h at 0.00 peak bleeding to NO_EXP (−0.73/−0.95). Unarmed AND
    #   peak < +0.2 after this many minutes → exit at market (30 = six 5m candles;
    #   dial down to 20 at the read if zombies persist). 0 = off.
    spike_stale_kill_min: float = 30.0
    # Jul 28 REGIME ROUTER (operator-directed; leg-6 becomes regime-FIRST): fine BTC regime
    # decides the species before ADX. In a chase regime -> ADX router as before (<=max CHASE,
    # >max FADE). In ANY other regime -> FADE at standard fade sizing (1x/20x, SL -0.70,
    # tripwire armed). Evidence: 20 lifetime fires — non-bull chases 12/13 never reached
    # +0.45 honest across 3 distinct dates (gate-met for no-chase); routing the trigger to
    # FADE instead = discipline-override at fade N=4, bounded by the armed fade OFF-gate
    # (N>=8 WR<=50%/Σ<0) + tripwire −1.5 + 1x sizing. OFF = restore ADX-only router.
    spike_regime_router_enabled: bool = True
    spike_chase_regimes: str = "STRONG_BULL,HEALTHY_BULL"
    # ARMED (pump confirmed): the 27-rung MIRA breathing envelope (archived Jul-24-26
    # calibration string verbatim; e.g. peak 2.5 -> floor 0.5, peak 10 -> floor 5.8).
    spike_ladder_armed: str = ("2.5:2.0,3:2.5,4:1.8,5:2.8,6:3.5,7:3.8,8:4.2,9:4.2,10:4.2,"
                               "11:4.6,12:5,13:5.3,14:5.7,15:6.1,16:6.5,17:6.9,18:7.2,"
                               "19:7.6,20:8,22.5:9,25:9.9,27.5:10.8,30:11.8,"
                               "32.5:12.7,35:13.7,37.5:14.6,40:15.6")
    spike_no_expansion_exempt_armed: bool = True  # ZEREBRO: armed +66min, 3h clock closed a +3.47
                                                  # ride at -0.09; unarmed zombies KEEP the sweep
    # ══ Jul 27 PM — PROMOTION PACKAGE (operator-directed, discipline-override at N=9/8/8/5
    # acknowledged; tight revert gates in CURRENT_STATE). Five probe verdicts executed early:
    # NONEXP conditional admission + RSIADX breadth release + DEADBAND pos-half open +
    # RSICEIL 65->70 + ADXMAX2 off.
    # ① NONEXP_CALM3D: gap-flat / flat+small LONG candidates admitted at FULL SIZE when
    # BTC regime is in the list AND BTC ATR <= max (pooled cohort 9·77.8%·+0.213 at ship;
    # pre-registered branch-(b) decider; bypasses keep-only-unmatched per the Jul-20 spec).
    nonexp_calm3d_enabled: bool = True
    nonexp_calm3d_btc_atr_max: float = 0.147      # Jul-23 sweep #1 2D separator (calm-BTC)
    nonexp_calm3d_regimes: str = "STRONG_BULL"    # comma list; the SBULL cell is the evidence
    nonexp_calm3d_max_stretch: float = 0.06       # Jul-30 COILED-PAIR leg (0=off): admit only when
    # EMA5-stretch <= this. Pooled #24b read (door 10 + calm-probes 9 = N=19): stretch<=0.06 ->
    # 11W/1L +0.328% vs stretched 2W/5L -0.350%; door fires alone perfectly monotonic (6W all
    # <=0.06, 4L all 0.07-0.17). Mechanism: calm tape has no follow-through fuel — a stretched
    # pair mean-reverts, a coiled one drifts with the trend. Threshold fitted on N=19 (full
    # haircut). Ships as the Option-A alternative to the FIRED N=10/Σ<0 revert gate; fresh
    # TIGHTER gate: next N>=10 post-refinement fires WR>=60 ∧ Σ$>0 else door OFF for good.
    # Jul-31 RISING-HOUR leg (operator-directed at pooled N=15 coiled; acknowledged override):
    # admit only when BTC 1h EMA20 slope > this. The door's third identity condition — calm
    # BTC ∧ coiled pair ∧ RUNNING hourly engine ("quiet market, loaded spring, rising tide").
    # Coiled-cohort evidence (door+calm-probes): b1h>0 = 10·90% WR·+0.372%·+$288 (EUL/TAO/UNI
    # all here) vs b1h<=0 = 5·60%·−0.202%·−$65 (2 of 3 losses = ONE same-tick cluster —
    # acknowledged). Sign boundary, NOT fitted (0 canonical). Mechanism kin: W2×1h-rising,
    # deadband family. <= -98 = leg off; missing b1h fails open.
    nonexp_calm3d_b1h_min: float = 0.0
    # 🔒 SAME-PAIR RE-ENTRY COOLDOWN (Aug-4, operator-directed at 90min; acknowledged N=2
    # free-insurance ship, AGLD-class): the door's ONLY two current-admission losers are
    # same-pair re-fires <=57min after a prior CALM3D fire on that pair (ONDO 39min -$198,
    # HYPE 57min -$215 — the HYPE instance occurred AFTER the watch was pre-registered);
    # all 12 winners were first-of-episode (nearest same-pair winner gap 8.4h). Mechanism =
    # the door's own thesis: the edge is the FIRST release of a coil — a re-entry buys the
    # spent spring. Historically-free zone for the line: (57min, 8.4h); 90 = operator pick
    # (1.6x above the loser edge). Door-scoped ONLY (other books' same-pair re-fires WIN:
    # ADA/EGLD/SXT — do not generalize). 0 = off. Counter CALM3D_REENTRY; in-memory
    # tracker (resets on redeploy — known gap, revert surface covers it).
    # 🔒 REVERT ->0 if blocked re-entries' would-be record runs >=60% WR on N>=8 fresh.
    nonexp_calm3d_reentry_cooldown_min: float = 90.0
    # 🎯 Aug-10 DMI THRUST leg (5th leg, operator "ship A" after the +DI deep-dive the operator
    # drove): +DI(14) ≥ 28 ∧ pair-ADX ≥ 21 — the coil must already be directionally driven.
    # Evidence: CUR door (only 4-leg-door pool) keep 8·100%·+$874 vs blocked 12·33%·−$1,383;
    # B1 keep 2·100%·+$120, sacrifices 3 old-2-leg-door-era winners (+$165, discount on record);
    # combined +$994. +DI buckets CUR: [25,28)=17%·−$955, ≥30=100%·+$739; pADX≥21 removes the
    # PENGU 29.3/pADX-17.7 residual free of B1 cost (ADX = sweep #2 survivor, same DMI family).
    # 🔒 READS: ① post-ship door fires N≥10 → WR≥60 ∧ Σ>0, else leg re-reviewed; ② STEP-BACK:
    # [CALM3D_DMI] blocks log entry px — re-sim blocked DI∈[26,28) at N≥6: would-be WR≥60% →
    # threshold 28→26 (the B1 EUL/UNI mid-band pattern returning under the 4-leg door).
    # Cell stays 2× (kept cohort 100%/100% both pools at current sizing — the ✗ HARMFUL verdict
    # is executed as fix-the-losers, per the locked caps/multipliers rule). 0 = leg off.
    nonexp_calm3d_min_pos_di: float = 28.0
    nonexp_calm3d_min_pair_adx: float = 21.0
    nonexp_calm3d_invest_mult: float = 2.0        # Jul-31 RE-ESCALATED 1.0→2.0 with the b1h leg (operator-
    #   directed DOUBLE staging override: skips the locked 1.5×-first ladder AND the N>=30 W-bar —
    #   evidence = the refined cohort 10·90%·+0.372% (partially in-sample; legs discovered on it).
    #   🔒 TIGHT REVERT = the standard cell verdict machinery at N>=5 fresh 2× fires: ✗ HARMFUL
    #   (net-negative) → 1.0× · ⚠ DRAG → 1.5×; PLUS the door's own fresh N>=10 gate (reset at this
    #   ship: WR>=60 ∧ Σ>0 else door OFF) rules the cell itself.
    #   History: 2.0 original → 1.0 Jul-29 (✗ HARMFUL at 7/10, NIL-class 2× tails) → 2.0 Jul-31.
    nonexp_calm3d_lev_mult: float = 1.0
    # Jul-29 RSICEIL door re-scope (operator-shipped at 6/10 fires as experimental narrowing):
    # the graduated RSI (65,70] LONG band additionally requires pair-ADX >= this floor.
    # Evidence: door fires split 2W (pADX 30.3/34.4) vs 4L (19.4-29.1) + lifetime RSI65-70
    # x ADX>=30 positive / <30 all-negative + 11c clock mechanism (overbought needs trend
    # support). Blocked candidates counted under RSICEIL_DOOR_ADXMIN. 0 = off (band open).
    rsiceil_door_adx_min: float = 30.0
    # ② RSIADX breadth release: the Mar-27 RSI x ADX cross-filter releases a sole-blocked
    # LONG when market breadth (bull%) <= this — candidate flows the NORMAL pipeline and
    # inherits the UNMATCHED patron (2x / PVR ladder). Cohort at ship: 8·87.5%·+0.326.
    # 0 = off. Cohort reconstruction = CSV slice (cross-filter bands ∧ entry_bull_pct).
    rsiadx_breadth_admit_max: float = 63.6
    # ③ DEADBAND asymmetric split (operator-ratified option b): POSITIVE side opens
    # [pos, old-band) as normal flow (upper half 8·75%·+0.248 at ship); NEGATIVE side keeps
    # the full long_btc_1h_deadband width (flat-down unproven — DBDOWN probe keeps collecting).
    long_btc_1h_deadband_pos: float = 0.025       # block LONG when 1h slope in [0, this); neg side
                                                  # uses long_btc_1h_deadband (0.05) unchanged
    # Jul 20 — DBDOWN PROBE (probe #9, operator-directed): the FLAT-DOWN half of the 1h
    # dead-band, [−deadband, 0). The Jul-5 gate's locked phantom revert FIRED (95·60.0%·
    # +0.100%, 7 dates; fresh flat-down >=Jul-17: 51·65%·+0.154% meets BOTH arms; H.BULL
    # 41·71%·+0.225 = the payer) while the traded flat-UP half runs 11·55%·−$107 norm —
    # the halves invert (pullback-long > drift-long, matching baseline's 1h-down best
    # cohort 19·89%). Graduated execution of the fired gate: flat-down opens as 1x
    # DBDOWN_PROBE (own tag/row; flat-up DEADBAND_PROBE untouched, 11/30). PASS phantom
    # seeding dries up naturally (no half left blocked). 🔒 Gates at N>=30 (>=5 dates):
    # WR>=60% & avg>=+0.15% -> execute the fired revert (full open, consider H.BULL scope);
    # WR<=45% or avg<0 -> dead-band re-locks flat-down, phantom-revert-gate logged RESOLVED.
    dbdown_probe_enabled: bool = False  # ✝ PROBE FLEET RETIRED 2026-08-11 (operator: near-zero flow, no value; ADXMAX math-dead for its relaxation gate at 12W/13L). Gap-Expand Probe table deleted same day; engine gates stay dormant.
    dbdown_probe_max_open: int = 3       # concurrent DBDOWN probes (LONG-only by construction)
    # Jul 30 DEEPGAP probe (#13, SHORT-only) — graduated from the PASS:MOMENTUM_SHORT_DEEPGAP
    # phantom (final read at retirement: N=17 · 71% WR · Σ+1.85% · avg +0.109%; H.BULL 6·83%·+2.05
    # carries it). Momentum-SHORTs killed ONLY by the Jul-6 deep-gap floor (pair ≥1% below its 4h
    # trend) open at gap-probe sizing as DEEPGAP_PROBE instead of blocking. ALL regimes admitted
    # (operator + quant: regime-slicing N=17 at promotion = pre-fit; the regime read is the
    # VERDICT'S job). 🔒 Pre-registered verdict at N>=30 (>=5 dates): bull-family cohort
    # (S.BULL+H.BULL) WR>=70% & Σ>0 at N>=15 -> promote bull-scoped full size; overall WR<=45%
    # or Σ<0 -> floor re-locks, probe off. Phantom tracker retired same day (Jul 30).
    deepgap_probe_enabled: bool = False  # ✝ PROBE FLEET RETIRED 2026-08-11 (operator: near-zero flow, no value; ADXMAX math-dead for its relaxation gate at 12W/13L). Gap-Expand Probe table deleted same day; engine gates stay dormant.
    deepgap_probe_max_open: int = 3      # concurrent DEEPGAP probes (SHORT-only by construction)
    # Jul 30 MAJORS probe (#14, BOTH directions) — strategic scaling experiment: BTC/ETH
    # (no_trade_pairs, track-only since Jun 3) run the FULL normal ladder and a candidate whose
    # ONLY blocker is the no-trade list opens at gap-probe sizing as MAJORS_PROBE. Rationale:
    # the 0.1% liquidity cap already binds on alt-sized orders (TWT -75.5%) — majors are the
    # only pairs where the $1M-roadmap capital deploys; "does the edge transfer?" is the key
    # scaling unknown, answerable at ~$4/fire. ⚠ Expectation set at ship: the threshold stack
    # is alt-vol calibrated (BTC ATR ~0.10-0.15% vs alts 0.3-1%) — thin flow IS a finding
    # ("re-scale before judging the edge"), record fires/day alongside WR. Per-PAIR verdict
    # rows (BTC vs ETH judged separately). 🔒 Read protocol in CURRENT_STATE #35.
    majors_probe_enabled: bool = False  # ✗ RETIRED 2026-08-10 (operator): 0 fires in 11 days = the experiment's ANSWER — alt-calibrated thresholds never trigger on BTC/ETH; a majors sleeve needs its own calibration (separate project)
    majors_probe_max_open: int = 2       # concurrent MAJORS probes (~1 per major)
    # ── Jul 31 🏀 SPIKE_BOUNCE (third spike species — LONG the violent dump; operator-directed
    # full-size ship at N=0, acknowledged: two dead countertrend-long predecessors (BOUNCE_LONG
    # 2W/6L Jun-23, C7 falling-knife class) BUT neither was this trade — this is the fade
    # MIRRORED (violent single-candle trigger, fixed SL never widens, direct taker, tripwire).
    # Trigger (mirror of pump legs, same polled candles, zero extra API): RSI crash >= crash_pts
    # in one 5m candle, prev RSI in [prev_min, prev_max], candle <= -min_candle_pct, vol >=
    # min_vol_ratio x avg20, no normal signal. GUARDS (each side-specific evidence, NOT symmetry
    # aesthetics — operator-caught twice): ① dump cap (candle >= -max_dump_pct; deeper = news/
    # delist/hack, the -5/-6% liquidation post-mortem) · ② bRSI FLOOR >= min_btc_rsi (the TRUE
    # mirror of the fade's shipped bRSI<=50 ceiling: buy a dump only when BTC momentum state is
    # firm = idiosyncratic panic; cold BTC = cascade; pre-registered: winners cluster >=~53) ·
    # ③ crashed-pair exclusion (pair EMA13-50 gap > min_pair_gap; DEEPGAP N=17·71% direct
    # evidence that dumps on crashed pairs CONTINUE — deliberately NOT mirrored to the fade,
    # its book shows no separation and EVAA won in-zone) · ④ regime block STRONG_BEAR/
    # HEALTHY_BEAR (where BOUNCE_LONG died — dump in a bear trend = continuation).
    # Exits = fade-mirrored verbatim (cohort comparability): fixed SL -0.7 NO widen, arm 0.45 +
    # ~0.5xATR giveback (NOT 1x — atr05 beat atr10 on the fade cohort AND entry ATR is inflated
    # by the dump itself), floor ladder = tail-catcher, tripwire auto-disable, BE-lock capture
    # from fill #1, inherits the fade N>=12 three-way exit-read winner. Sizing: FULL Inv 1x/
    # Lev 1x, normal book rules — NO sleeve max-open, NO auto kill gate (operator: manual
    # review here; tally reported every read). 🔒 READ (locked): N>=10 · >=4 dates → WR>=55% ∧
    # Σ>0; slices per-regime/bRSI/dump-magnitude/rng, never pooled; thresholds frozen.
    spike_bounce_enabled: bool = False  # ✝ OFF FOR GOOD 2026-08-10 PM (operator): PGAP-window read mathematically failed at 0W/4L post-ship (SKY/NAORIS/BANANA/TA −$87; WR≥55 unreachable at N=8). Lifetime: every $ earned came from the 5 fit-sample trades (+$77), forward gave it back (−$78). B2 impact of removal ≈ $0 (8 kept · −$1). Revival = phantom-only read (rsi_prev≥48 healthy-base leg, watchlist 37)
    # ⚡ Aug-5 PM RE-ENABLED with the pgap window (operator ship after the theory review).
    # History same day: frozen-leg read FAILED at N=11·4 dates (45.5%·−$27, XMR counter-exampled
    # the alignment-split rescue) → species OFF AM; post-mortem found the pgap separator
    # (0W/4L blocked · all 5 winners kept) → re-entry as this probe-class ship. Read gate lives
    # on the spike_bounce_max_pair_gap line below; miss → OFF for good.
    spike_bounce_rsi_crash: float = 25.0     # RSI points DOWN in one 5m candle
    spike_bounce_rsi_prev_min: float = 45.0  # resting band before the crash (mirror of [35,55])
    spike_bounce_rsi_prev_max: float = 65.0
    spike_bounce_min_candle_pct: float = 0.5   # candle must move <= -this %
    spike_bounce_max_dump_pct: float = 3.0     # candle deeper than -this % = news class, NO trade
    spike_bounce_min_vol_ratio: float = 5.0    # discovery-candle volume vs avg20
    spike_bounce_min_btc_rsi: float = 50.0     # bRSI FLOOR (0=off; fail-open on missing)
    spike_bounce_min_pair_gap: float = -1.0    # pair EMA13-50 gap must be > this (0=off; DEEPGAP guard)
    # 🏀 PGAP WINDOW upper bound (Aug-5 re-enable ship): bounce fires only when the pair was
    # ALREADY mildly weak — gap ∈ (min, max]. N=11 post-mortem: pgap ≥ −0.10 = 0W/4L·−$106
    # (MEME/XMR/IMX/B2 — healthy-pair dump = news class, incl. the XMR alignment counter-example);
    # pgap ≤ −0.15 = ALL 5 winners (+$70). Threshold −0.125 frozen in the empty [−0.15,−0.10] gap
    # (sign < 0 is the theory; the extra magnitude is empirical — if the read fails, retest the
    # sign boundary before killing the mechanism). Single-pool post-hoc evidence, acknowledged.
    # 🔒 PRE-COMMITTED READ: at N≥8 post-filter fires — WR≥55% ∧ Σ>0 → keep; miss → species OFF
    # for good. Blocked side logs [SPIKE_BOUNCE_PGAP] with entry px (re-sim rows). ≥99 = off.
    spike_bounce_max_pair_gap: float = -0.125
    spike_bounce_blocked_regimes: str = "STRONG_BEAR,HEALTHY_BEAR"  # comma list; empty = no regime block
    spike_bounce_invest_mult: float = 1.0
    spike_bounce_lev_mult: float = 1.0
    spike_bounce_sl_pct: float = -0.70         # fixed, NEVER ATR-widened (the one dump law)
    spike_bounce_trail_atr_mult: float = 0.5   # runner-trail giveback N (LONG global is 1.0; bounce
                                               # frozen at 0.5 — atr05 beat atr10 on the fade cohort
                                               # and entry ATR is inflated by the dump candle itself)
    spike_bounce_tripwire_pct: float = -1.5    # close <= this → tripwire (gap-through); flips species OFF only if spike_tripwire_autodisable, else CRITICAL alert-only (Aug-5)
    # Jul 30 PM — FADE BTC-RSI CEILING (operator discipline-override ship at sub-cohort N=3,
    # below the pre-registered N>=5 bar; acknowledged). Block a SPIKE_FADE when BTC RSI at
    # entry > this: fading an alt spike while BTC's own momentum is hot = shorting into
    # market-wide beta (squeeze), not idiosyncratic exhaustion. Evidence at ship: 7/7 fade
    # winners entered bRSI <= 47.2; bRSI > 50 = 0W/3L (SNX 64.7 · ZEREBRO 53.7 · XPL 52.8
    # −$103 at 2x, XPL = out-of-sample post-freeze). Ceiling frozen at the pre-registered 50.
    # 🔒 REVERT SURFACE (phantoms retired -> blocked cohort logged instead): every block logs
    # [SPIKE_FADE_BRSI] with pair/price/bRSI; at N>=5 blocked candidates, re-simulate their
    # outcomes from 1m klines under the fade exit stack (fixed -0.7 SL / ladder) — blocked
    # cohort >=60% WR or Σ>0 -> ceiling OFF. 0 = disabled. Fail-open on missing bRSI.
    # ⚙️ Aug-5 TIGHTEN 50→45 (operator ship, evidence acknowledged watch-grade/one-pool):
    # zone [45,50) current batch = 8·3W/5L·38%·avg −0.617%·−$472 (EVAA/ICNT/PIPPIN/FRAX all
    # live here — the 4 highest-bRSI entries of the batch are its 4 biggest losers); B1 zone
    # only 2·1W/1L·−$33 = no cross-pool support; EVAA = 54% of blocked-$ (LIQ2-amplified;
    # de-sized Δ ≈ +$344). Shipped on operator call with a TIGHT pre-committed revert:
    # 🔒 at N≥6 blocked-in-[45,50) candidates (each logs [SPIKE_FADE_BRSI] with entry px),
    # price-replay them under the fade exit stack — WR≥55% ∨ Σ>0 → ceiling back to 50.
    spike_fade_max_btc_rsi: float = 45.0
    # 🔒 FADE BTC-DIST13 GATE (Aug-4, fade N>=25-30 read lead item executed): block a FADE
    # when BTC trades ABOVE its 5m EMA13 (dist13 > max). Blocked cohort lifetime = 0W/5L
    # -$304 across 4 dates / both batches (XPL/ZEREBRO/SNX B1 + ICNT/FRAX B2 — incl. the
    # worst fade loss ever, ICNT -$131 slip-through); zero winners ever forfeited (AGLD
    # free-insurance class). SIGN boundary (not fitted). Mechanism: BTC above its short-term
    # mean = market bid intact -> the alt pump gets beta tailwind and squeezes the short;
    # mirror-confirmed by the LONG species (ALL 10 bounce fires + both fresh chases had
    # dist13>0 — the long side LIVES there; FADE-ONLY scope is structural, do not extend).
    # Kin of the shipped bRSI<=50 ceiling (same family, mean-vs-level expression).
    # Sentinel >= 99 = off. Fail-open on missing BTC price/EMA13. Counter SPIKE_FADE_BD13.
    # 🔒 REVERT ->99 (off): blocked-candidate logs re-simmed from 1m klines (same surface
    # as the bRSI ceiling) — >=60% WR or Σ>0 at N>=8 -> off. Note: does NOT address the
    # SPELL crash-extreme class (BTC below EMA13 there — separate d6-band question at the
    # fade read).
    spike_fade_max_btc_dist13: float = 0.0
    gapmin_probe_max_open: int = 3       # concurrent GAPMIN probes (both directions combined)
    # Jul 13 PM (operator: "both ways"): the GAPMIN probe covers SHORTS too — band
    # [floor, ema_gap_threshold_short=0.08); the 0.06/0.08 thresholds predate most of the
    # current filter stack, so the whole excluded band is re-measured at 1x, per side.
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
    # Jul 10 SHIP (live JSON = 0.35): LONG stretch ceiling — chase-entry block. Zero-cost cut:
    # across the ENTIRE screened history every mom-long winner entered at stretch ≤ 0.34 (max
    # winner ACT 0.332); the only-ever occupants above 0.35 are losers (LDO 0.53 −$285,
    # TAC 0.37 −$132 — buying an already-exhausted burst, pair ADX rolling over).
    # 🔒 Revert → 0 if blocked stretch>0.35 longs run ≥60% WR on N≥8 fresh evidence.
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
    # 🛡 Aug-19 2026: QUIET-PAIR CONDITIONAL SL (gate 53; operator-directed OVERRIDE ship —
    # N=18 real stops below the locked observe gate, acknowledged; tighter-than-standard
    # revert in CURRENT_STATE gate 53). Momentum LONGs from quiet pairs (entry ATR% <
    # threshold): the fixed −0.70 stop is a ≥2.3-ATR flash move that mean-reverts — real
    # eligible stops 6/8 armed (+0.40 before −2.0); 138-episode dose-response 71%→31% arm
    # by ATR bucket, eligible split positive all 7 months, shuffle P=0.09. Hot pairs keep
    # the −0.70/ATR-widened chain (their stop IS information — same mechanism that refuted
    # the flip wide SL 5/7). N×ATR scaling REJECTED on data (widens exactly the blown
    # class: all clamp variants −18 to −30pp vs +4.1 two-regime). Width −2.0 = interior
    # optimum (−1.5 sits inside the quiet-flush zone: BONK/AVAX blown at −1.5, armed at
    # −2.0; curve decays past −2.0). Wired in BOTH SL paths (realtime + monitor, ROSE-fix
    # lesson). BE floors, FL1/FL2, fast exits all unchanged — hard FLOOR, not a hold
    # guarantee. threshold 0 = OFF (instant revert). 🔴 LIVE-CUTOVER: OFF at live start.
    momentum_long_sl_atr_threshold: float = 0.45  # entry ATR% below this = quiet class
    momentum_long_sl_quiet_pct: float = -2.0      # quiet-class hard SL (negative)
    # 🌊 Aug-21 2026: BULL-RUN CONTINUATION SLEEVE (gate 57). Regime-gated dip-buy LONGs on the
    # top-N COIN pairs, active ONLY while the Bull-Run Monitor is GREEN. Derivation (Aug 19-21
    # +19% episode + 8.7-month false-positive scan, 6,169 windows): the regime signature is
    # trend EFFICIENCY (|net|/Σ|moves| ≥0.10) — every historical trap-rally (Dec-3/4, Feb-7/8,
    # Feb-26, Mar-4/5) peaked at eff 0.065-0.082 despite r72 up to +12.9%; a real accumulation
    # run travels orderly. Composite ON(r72≥5 ∧ above≥56 ∧ eff≥0.10) fired exactly 3× in 8.7
    # months (Jan-5 4h −0.29%, Jun-15 6h −1.36%, Aug-19 41h+ +11.92%). Sleeve replay on the
    # founding episode (COIN top-10, GREEN-gated, 3 slots, wide exits, fees): N=76 · 67% WR ·
    # +$1,443 (haircut expectation ≈ +$700-900/72h GREEN). Live alt exits REFUTED for this
    # class (BE 0.4→0.1 + 1×ATR: avg trade +0.046% < 0.09% fee toll → −$2,003 same entries);
    # arm sweep monotone 0.4→1.5, plateau 0.8-1.2 → arm 1.0 (plateau middle, not the edge).
    # TradFi perps (EQUITY/COMMODITY underlyingType) 40% WR −$5,043 vs COIN 66% +$7,014 —
    # universe rides coin_underlying_only + rank ≤ size. 🔒 KILL BAR (manual, no auto): first
    # 10 fills WR≤45% ∨ Σ<0 → toggle OFF. Sizing frozen 1×/1× until ≥2 profitable episodes.
    bullrun_sleeve_enabled: bool = True    # master toggle (kill switch — entries only; monitor keeps computing)
    bullrun_green_r72_on: float = 5.0      # GREEN turn-ON: BTC 72h return ≥ this %
    bullrun_green_r72_off: float = 4.0     # GREEN stay-ON floor (Schmitt band)
    bullrun_green_above_on: float = 56.0   # GREEN turn-ON: % of 5m bars above EMA20 ≥ this
    bullrun_green_above_off: float = 53.0  # GREEN stay-ON floor
    bullrun_green_eff_on: float = 0.10     # GREEN turn-ON: trend efficiency ≥ this (THE load-bearing leg)
    # Aug-23 (16): stay band raised 0.08 → 0.10 (= the ON threshold; efficiency loses its hysteresis, r72/above keep
    # theirs). Stale-GREEN detector: every fill taken with eff < 0.10 lost — live 12·17%·−$997 across 3 windows incl. all
    # six post-gate losers; the founding window never traded below 0.10 (zero out-of-sample cost by construction).
    # Counterfactual this episode: GREEN ends Aug-22 15:25 UTC (brief re-arms 17:05-18:40, 19:10-20:05), dark from 20:05.
    # Before/after: ex-ONG/ETH 31·42%·+$515 → 19·58%·+$1,512. Sleeve switch (window units) — acknowledged override.
    # 🔒 revert: next GREEN episode close → scripts/bullrun_replay.py with stay 0.08 vs 0.10; 0.10 net-worse → back to 0.08.
    bullrun_green_eff_off: float = 0.10    # GREEN stay-ON floor (was 0.08)
    bullrun_latch_r6h: float = -3.0        # crash-latch: BTC 6h return ≤ this → instant OFF (also price < 1h EMA50)
    bullrun_amber_r24: float = 6.0         # AMBER alert (24h tight variant) — display/log only, never arms
    bullrun_amber_above: float = 65.0
    bullrun_amber_eff: float = 0.12
    bullrun_universe_size: int = 10        # sleeve trades scan-rank ≤ N (COIN-only universe; rank 11-20 refuted: 50% WR −$1,933)
    bullrun_dip_atr_mult: float = 0.3      # entry: dip ≥ N×ATR(14,5m) below 5m EMA20, then close reclaims
    bullrun_pair_spacing_hours: float = 2.0  # min hours between sleeve fires on the same pair
    bullrun_max_slots: int = 4             # sleeve concurrency cap — operator-directed Aug-21: match max_open_positions (4), no separate sleeve throttle; global max_open remains the real bound
    bullrun_invest_mult: float = 1.0       # Inv Mult (same cell plumbing as other sleeves)
    bullrun_lev_mult: float = 1.0          # Lev Mult
    bullrun_base_sl_pct: float = -0.7      # sleeve base SL; widened by the existing sl_atr_multiplier/floor chain
    bullrun_be_arm_pct: float = 1.0        # BE arms at peak ≥ this (must clear ~2×ATR dip-entry noise band)
    bullrun_be_lock_pct: float = 0.2       # BE floor once armed
    bullrun_trail_atr_mult: float = 2.0    # trail giveback = N × entry ATR% from peak (plateau 2.0-2.5)
    # Aug-21 (day-1 post-mortem, DECISION_LOG 2026-08-21 (11)): PULLBACK-PHASE GATE — no sleeve entry while
    # BTC sits more than N% below its 24h high. The one variable that separated the founding replay's
    # winners from losers (top of 21, 3/3 per-day): off24h ≤ −1.6% → −$1,679·50% WR vs near the high
    # +$2,985·84%. 8.7-month regime validation inside uptrends (N=1,167, 23 distinct days): fwd-6h −0.33%
    # / P(dd≤−1%) 53-62% when >1.6% under the high vs +0.05% / 34% within 0.8%; no effect outside
    # uptrends (= specifically the pullback phase of a run). Mechanism-aware re-sim at −2.0: +$2,649 →
    # +$3,770, thrust days untouched. All 5 day-1 live losers fired at −2.5..−3.3%. Plateau −1.6..−2.0;
    # −1.0 kills thrust-day entries. Refused dips stay ALIVE (consuming them cost −$925 on the founding
    # window — DECISION_LOG 2026-08-21 (13)). 0 = disabled (the ONLY off value; a positive value is normalized to its
    # negative with a warning — a sign slip must never silently disable the gate). Stamped per fill as entry_br_off24h.
    bullrun_btc_off24h_max: float = -2.0
    # Aug-21 (15) HIGH-RUNG PROFIT LOCK (operator-directed after the PEPE +4.37 → +2.52 exit). Empirical retrace
    # study (two 30-path cohorts): after reaching +2/+3 a runner goes on to the next +1% in 93-100% of cases but
    # pulls back a MEDIAN 1.2-1.4% first (p80 1.8-3.0) → mid-level rungs stop the continuers (refuted: rungs from
    # +2 lose in both cohorts). Past +4 continuation decays (84%) and the retrace grows → a lock there is positive
    # in BOTH cohorts (+$136 / +$100; PEPE 2.53 → 3.50). Floor = max(2×ATR trail line, rung floor); below the
    # first rung nothing changes. "peak:floor" pairs. 🔒 REVERT: at N≥8 trades reaching +4, if ladder-exited
    # trades underperform their trail counterfactual (peak/exit recorded) → clear the string. "" = off.
    bullrun_ladder: str = "4.0:3.5, 5.0:4.5, 6.0:5.5, 8.0:7.0, 10.0:9.0, 12.0:11.0, 15.0:13.5, 20.0:18.0, 25.0:22.5, 30.0:27.0"
    # Aug-22: sleeve-scoped pair blacklist (comma-separated). ONG: 0/5 across v1+v2 (−$760), every loss a −1.2 stop —
    # its entry ATR (1.1-1.9%) puts the floored SL at ~1× ATR (majors sit at 2-4×). Discipline-override ship (founding
    # replay had ONG winning); 🔒 revert: if the non-ONG ATR>1.0 sleeve cohort wins (N≥5, Σ>0) re-admit + fix the floor.
    # Aug-22 (2): ETH added — replay (founding + live windows) shows ETH fills ≈ zero expectancy (+0.06%/t, 13 fills) while
    # its freed slot pays in BOTH windows (+$163 / +$702); live v2: 2 fills 0W peak 0.00. Mechanism: ATR 0.3-0.5 → runner
    # capped below the ladder, stops cost the same = slot-hog. BTC NOT blacklisted (its fills won the founding window).
    # 🔒 revert ETH: low-ATR (≤0.5%) sleeve fills avg ≥ +0.30%/t on N≥8 in the next GREEN episode.
    bullrun_pair_blacklist: str = "ONGUSDT,ETHUSDT"
    # Aug-22 (3): BTC-leader gate — no sleeve entry while BTC is BELOW its own 5m EMA13 (an alt reclaim without BTC's
    # reclaim = alt moving without its leader). Only BTC variable losing in EVERY window: live 6·33%·−$545, founding
    # replay 8·38%·−$220, replay-live 5·40%·−$65 (pooled N=14, 36% WR, −0.35%/t). Discipline-override (N<30).
    # Live 25·48%·+$989 → 19·53%·+$1,534; founding 59·63%·+$3,033 → 51·67%·+$3,252. DECISION_LOG 2026-08-22 (15).
    # 🔒 revert: re-run scripts/bullrun_replay.py at the next GREEN episode close — blocked cohort net-positive → off.
    bullrun_btc_ema13_required: bool = True
    # Aug-23 (18): BTC 1h-EMA20 slope gate — no sleeve entry while BTC's 1h slope ≤ this (0.0 = the tested cut; blank = off).
    # Live 13·23%·−$979 removed (4 windows; monotone: <−0.02 → 23% WR, 0-0.05 → 25%, >0.3 → 64%·+1.27/t); replay-live
    # removed 5·20%·−$561; founding window had zero cases. Shipped WITH the 0.10 stay band (overlap 9 fills; each adds
    # 3-4 losers the other misses): ex-ONG/ETH 31·42%·+$515 → 15·67%·+$1,918. Discipline-override (N<30).
    # 🔒 revert: next GREEN episode close — blocked cohort (replay or live entry_btc_1h_slope ≤ min) net-positive → off.
    bullrun_btc_1h_slope_min: Optional[float] = 0.0
    # Aug-23 (20): RE-ARM DOOR — second trigger while the 72h composite is OFF (it detects continuation late and
    # cannot re-arm after a pullback: eff needs +3.9% straight-line). ON = BTC ADX ≥ adx_min AND rising vs 30 min ago
    # AND alts leading (median universe 6h return > alt_r6h_min %, ≥ alt_above_pct % above their 1h EMA50) AND
    # EMA13>EMA20>EMA50 AND BTC > 1h EMA50. OFF = BTC < 1h EMA50 | ADX < adx_off | max_hours | composite GREEN.
    # Replay Aug-13→23 (sleeve entries/exits, composite-OFF only): 14·79%·+$2,338 (ex-ETH 13·77%·+$1,497), no losing
    # window, skipped the 3 fake bounces (Aug-17 15:05, Aug-18 14:25, Aug-22 night); Aug-19 14:10 = 6·6W·+$2,045 six hours
    # before the composite confirmed. Caveats: 3 windows with fills, late by construction. In REARM the off-24h gate is
    # bypassed (bounce from a ≥2% low); EMA13 + 1h-slope gates apply. Watchlist ship (operator).
    # 🔒 REVERT (manual): ≤1 winning window of the first 3 REARM windows with fills → bullrun_rearm_enabled=false.
    bullrun_rearm_enabled: bool = True
    bullrun_rearm_adx_min: float = 40.0        # entry: BTC ADX(14, 5m) ≥ this AND > its value 30 min ago
    bullrun_rearm_adx_off: float = 30.0        # stay: ADX ≥ this
    bullrun_rearm_alt_r6h_min: float = 1.0     # entry: median universe-pair 6h return > this %
    bullrun_rearm_alt_above_pct: float = 80.0  # entry: ≥ this % of universe pairs above their 1h EMA50
    bullrun_rearm_max_hours: float = 24.0      # hard off
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
    runner_trail_atr_min: float = 0.0    # Jun 24: 1.0→0.0 (no ATR gate, mirror the SHORT runner) when porting the short exit to longs
    runner_trail_arm_peak: float = 0.40  # ⚡ 0.45→0.40 Aug-5 (operator; [0.40,0.45) peak band = 1W/3L·−$287
    # across ALL THREE pools — AAVE/baseline, XLM/B1, WLD −$214 CALM3D Aug-5 — one loser per pool,
    # three different cells; CF at arm 0.40 (trail floor = max(peak−ATR, +0.10 lock)) saves all three
    # to ≈+0.10, MIRA the lone winner unaffected. Clip bound: current-era captured tr30 = 9/10 winners
    # never near +0.10 post-0.30. 🔒 REVERT →0.45 if at N≥8 band-armed trades the trail Δ vs no-arm CF < 0.
    # (Mechanic: peak P&L ≥ this swaps tight→stretch/ATR-floor trail. History: Jun-24 0.70→0.45, Aug-5 0.45→0.40.)
    runner_trail_k: float = 0.5          # exit when live stretch ≤ k × peak stretch (unsigned, matches shadow strpk) — fallback when use_atr=false
    # Jun 24 — LONG-runner parity with the SHORT runner (operator-directed): give longs the SAME
    # ATR-floor (chandelier) + BE-ratchet + give-back-cap machinery the shorts run, on independent
    # runner_trail_* fields (so longs can be tuned tighter — the VELVET shadow showed long peaks are
    # smaller, ATR give-back too wide). Direction-agnostic P&L math, applied in indicators.py's LONG
    # runner branch. Mirrors runner_trail_short_{use_atr,atr_mult,be_ratchet_enabled,be_lock_pct,giveback_frac}.
    # OBSERVATION caveat: shipped ON ahead of the N≥30 Leash-Shadow strpk-LONG gate (operator call);
    # affects non-flip LONG sleeves (BULL_LONG, MOMENTUM long, normal longs). Flip-LONGs (is_flip) keep
    # their existing exit — deferred (the dormant idle-insurance sleeve, build with its source). REVERT:
    # turn off if armed long runners net WORSE than the `actual` long baseline on N≥10 fresh closes.
    runner_trail_use_atr: bool = True            # true = ATR-floor (chandelier) trail; false = K×peak_stretch ratio trail
    runner_trail_atr_mult: float = 1.0           # N — give back N×ATR% from peak before exit (hard SL still backstops). Jun 29: 0.5→1.0 (LONG-only field; shorts use runner_trail_short_atr_mult, untouched at 0.5). Evidence: leash-shadow atr05-vs-atr10 across 10 batches, unmatched-longs N=31 → atr10 (1.0×) beat atr05 (0.5×) +0.231%/tr (+7.2 cum%), outlier-robust (drop ACTUSDT +0.144, ex-high-ATR<1.3 +0.105, drop-top-3-winners +0.053). 1.0 NOT 1.5 (1.5 over-widens, gives back even ≥2.0 ATR). FLAT not ATR-conditional (non-monotonic = confound). Right-tail edge: bounded-SL give-back vs open runner upside. REVERT if atr05≥atr10 over next N≥20 fresh armed longs.
    runner_trail_be_ratchet_enabled: bool = True # (LONG) Jul 13: RE-ENABLED (was operator-disabled Jun 25) — armed long runners' exit floor clamps to >= be_lock_pct (+0.10). Evidence: gate fired 8/8 fresh armed longs (cum +0.70pp, zero negatives); full baseline N=38 armed-long head-to-head atr10+BE +18.35% vs lockless atr10 +17.62% (+0.73pp) vs actual +15.17%. atr05 (tighter N) refuted 3x — the floor rescues sagging winners WITHOUT the runner tax. REVERT if actual-vs-lockless-atr10-shadow cum Δ < 0 on N≥10 fresh armed longs (≥3 dates).
    runner_trail_be_lock_pct: float = 0.10       # min P&L an armed long runner may give back to (the ratchet lock)
    runner_trail_giveback_frac: float = 0.0      # cap give-back at frac×peak (0 = off, raw N×ATR). Off to start, mirror current short
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
    runner_trail_short_arm_peak: float = 0.40 # matches leash ACT + live trailing arm
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
    runner_trail_short_giveback_frac: float = 0.35      # Aug-14 RE-ENABLED (operator; the Jun-17 disable's OWN re-enable condition met: 3 armed shorts round-tripped peak→≤0 — AAVE −0.40 fr +0.44 · IDOL −0.13 fr +0.83 · BICO −0.06 fr +0.58). giveback = min(0.5×ATR, 0.35×peak): the floor can never surrender >35% of the running peak NOR go negative (retires the negfloor class structurally). 4-pool assumption-free CF: BASE +$647 · B1 +$73 · B2 +$94 · batch +$114 = +$928 on 41 lifted trades, zero made worse in CF (floor-only construction). Known cost class: SKYAI-type dip-past-floor-then-re-run (the Jun-17 1-sample disable) — bounded: BASE tick-shadow incl. those cuts still +$487. LONGS refuted for this (tick shadows −$1,411; tails own long P&L) — long frac stays 0. 🔒 REVERT: exits log bound='cap' — at N≥12 fresh armed shorts, if cap-bound exits underperform their would-be atr floors cum, OR a SKYAI-class clip (cap exit then ≥+1.5pp continuation) appears twice → back to 0.
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
    # Jul 6: DEEP-GAP FLOOR (GIGGLE post-mortem) — block a momentum-SHORT when pair gap ≤ this
    # (already ≥1% below the 4h trend = selling after the crash → bounce). Pair-level twin of
    # btc_1h_slope_min_short (−0.60, shipped N=4, same mechanism). Baseline zone (all ≤−1.5):
    # 3·33%·−$224; band −1.5..−1.0 EMPTY (threshold in clean space); mild pullback −1.0..−0.6 =
    # 2·100%·+$164 untouched. Ship impact MS $526→$750. ⚠ N=3 operator-directed override.
    # 🔒 REVERT (→0 off) if PASS:MOMENTUM_SHORT_DEEPGAP phantoms hit ≥55% WR on N≥8 fresh.
    # 0 = disabled; negative activates. Counter MOMENTUM_SHORT_DEEPGAP.
    momentum_short_pair_gap_min: float = 0.0
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
    # Aug-18 2026 BAND RELAXATION (operator-directed, gate 51; DECISION_LOG 2026-08-18 (2)):
    # JSON "50-55:99-100,55-60:20-25,70-100:40" → "50-55:15-40,55-60:15-30,70-100:40" (review I1: first cut used 18-floors, silently blocking RSI-50-60×ADX-15-18 ≈5.4% of tape vs the new 15 floor).
    # ① 50-55 was a TOTAL block (requires ADX 99-100 = impossible), shipped Jun-1 on PRE-UNMATCHED
    # mixed-flow evidence under the retired exit stack; 18-40 = no-op vs the global ADX gate
    # (band neutralized, handle kept). ③ 55-60 ADX window 20-25 → 18-30 (May-18 Option B, same
    # era caveats). 70-100:40 climax rule KEPT (weakest relaxation case). Three-pillar docket:
    # retired exits (breakeven WR 50→32%) · retired population (pre-Jun-9 keep-only-unmatched)
    # · phased replays (blocked cohorts armed 58-76% by band, phases 1-2). ⚠ Gate-49's
    # falling-BTC phase-3 SKIPPED by operator decision — the live flow IS phase-3 at full size;
    # 🔒 reverts in gate 51 (CURRENT_STATE), incl. the falling-BTC tripwire.
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
    flip_entry_sources: str = "FAN_RATIO_GATE:1.0"  # SOURCE:size:lev (lev optional→1.0). Jun 23: PAIR_RSI_OB DISABLED (removed from sources) — every lever exhausted (gap/pADX/regime/ATR/fast-exit/dist-from-EMA13 all in-sample or already captured); under the full current stack the survivors are a zero-edge breakeven (N=22/64%WR/+0.003%), with no positive edge to recover (fast-exit can't help: 76% of losers go straight-to-SL/peak<0.2%; dist-from-EMA13 is 74% redundant with the gap filter + adds 0 over the full stack). Phantom seed keeps tracking for a future re-enable IF a cross-batch-stable separator ever emerges. Jun 22: PAIR_RSI_OB DE-LEVERED 1.0→0.05 (=20×→1× obs) — sleeve failed its locked revert gate decisively (live 33/42%WR/−0.32%, cross-batch N=136 net-neg) AND every entry/exit lever exhausted (gap/pADX/RSI/regime/ATR all in-sample-only); kept firing at 1× for observation (phantom + live data) instead of bleeding at 20×. Re-lever only if a cross-batch-stable separator emerges. Jun 20: PAIR_RSI_OB lev 0.05→1.0 = full 20× (operator N=9 override, paired with the ADX≥33 floor — winner cell only). Jun 19: was @1x size × 0.05 lev = 1× live observation (S.BULL-only). Jun 16 PM: RE-ACTIVATED PAIR_RSI_OB. Jun 17 PM: LONG_UNMATCHED_ONLY DISABLED (live off, phantom-only) — live N=0 both batches + phantom N≈8/38%WR/-0.190% ✗ whipsaws = starved & losing; phantom seed is decoupled (always fires) so it keeps tracking for a future re-enable. FAN_RATIO DE-MUXED 2x→1x Jun 15 (multiplier gate ✗ HARMFUL: live N=24/50%WR/-0.24%/-$912). RE-ENABLE LONG_UNMATCHED_ONLY only if its phantom clears WR≥55% AND net-positive on N≥20. [phantom prior PAIR_RSI_OB N=11/82%WR/+0.405%]
    # ── FAN_RATIO_GATE flip filter section (Jun 16, 76-trade batch). Source-namespaced
    #    (`flip_fan_*`); future sources get parallel `flip_unmatched_*`/`flip_pairsi_*` sets,
    #    evaluated independently in _flip_filters(). All fail-open. Block reasons FLIP_FAN_*.
    flip_fan_stretch_min: float = 0.12        # block FAN flip if entry EMA5 stretch < this (thin fuel, batch N=10/10%WR/-$495). 0 = off
    flip_fan_block_btc_rsi: float = 60.0      # block FAN flip if BTC RSI >= this AND BTC ADX >= flip_fan_block_btc_adx (fade into strong un-exhausted bull: N=19/47%WR/-$416). 0 = off
    flip_fan_block_btc_adx: float = 30.0      # paired with flip_fan_block_btc_rsi
    flip_fan_pair_adx_min: float = 21.0       # block FAN flip-SHORT when entry pair ADX < this (0=off). FAN flips BYPASS the momentum short system's pair-ADX floor (Pair ADX Dir rising + ADX-Strong>20) → they fire weak-trend fades (pADX 15-19) with no follow-through that chop/gap back. Restore it. Cross-batch 3 batches (J20/J22a/J22b, deduped N=89): pADX>=20 = 42/71%WR/+$482 vs <20 = 47/51%WR/-$850 (the entire drain); KEEP>BLOCK + WR up EVERY batch; loss diffuse (top-2 pairs 28% → dimension not blacklist). Counter FLIP_FAN_PAIR_ADX. RAISED 20→21 (2026-06-23): 4-batch deduped FAN-S re-sim (J20-23, N=92 non-blacklisted) — the pADX 20-21 band = N=6/50%WR/-2.31% (incl. BICO -1.22 straight-to-SL gapper), so the 20 floor leaked it; floor 21 = 36/75%WR/+0.207avg/+7.47 vs floor 20 = 42/71%/+0.123/+5.16. Do NOT go to 22: the 21-22 band is net-POSITIVE (+1.99, incl. ORDI +2.80 runner). In-sample N=6 marginal → 30-50% haircut. TIGHT REVERT: →20 if 20-21-band FAN shorts come back ≥55% WR on N≥8 fresh; →0 if pADX>=21 flips ≤60% WR on N≥15 fresh
    flip_fan_pair_adx_exempt_regimes: str = 'STRONG_BEAR'  # 2026-06-30: regimes where the pADX<min floor does NOT apply. The pADX<20 zone LOSES in HEALTHY_BEAR (N=7/29%WR/-$383, floor correct) but WINS in STRONG_BEAR (N=9/78%WR/+$310; ex-top-2-pairs still +$58/4-of-5). Mechanism: in a strong bear even a weak-pADX pump deflates because macro is down → the fade follows through. Matches SCREENED_BASELINE (STRONG_BEAR = best flip-short regime, 5/5). Recovers +$310/+9 trades (flip baseline 34/74%/+$392 → 43/74%/+$702). N=9/one-bear-window (06-15→20) DISCIPLINE-OVERRIDE → TIGHT REVERT: clear the exemption (→'') if STRONG_BEAR pADX<min flip-shorts net-negative on N≥10 fresh. Counter still FLIP_FAN_PAIR_ADX (just not fired in the exempt regimes). Empty = no exemption (filter universal).
    # U3 (Jun 20, N=14/1-weekend DISCIPLINE-OVERRIDE): block FAN flip-short when BTC ATR% < this — the
    # weekend thin-liquidity regime (20× gap-through-SL fat tails). FAN-only (PAIR_RSI_OB returns early).
    # Jun20 Sat sub-0.10 cell = N=14/36%WR/-$775, all gaps; weekday Jun17/18 never dipped <0.109 = regime
    # is weekend-only. Counter FLIP_FAN_LOATR. 0 = off. TIGHT REVERT: →0 if next weekend's sub-0.10 phantom ≥45% WR on N≥6.
    flip_fan_btc_atr_min: float = 0.10
    # Jun 17 — 2D regime×ADXΔ block for flip-SHORTS (ALL sources). Block a short flip when entry
    # ADXΔ < adxd_max AND BTC regime ∈ regimes set. Cross-batch (76+39+11 pool, deduped) the cell
    # BULL/CHOP ∧ ADXΔ<0 = N=38 / 40%WR / -0.34% / -$1070; 96% of its losers peak < 0.45 arm so the
    # give-back cap can't save them → entry block, not exit. Orthogonal to regime alone (ADXΔ sign
    # is ~50/50 within each regime). Counterfactual: last(bear) batch -$63 [dormant], current(bull)
    # batch -$611→-$5. Discipline-override (literal NP gate 13%<60%; saveability analysis overrides).
    # TIGHT REVERT: re-open if these cells flip to WR>45% on N>=15 fresh. Empty regimes = OFF.
    flip_short_regime_block_adxd_max: float = 0.0   # block flip-SHORT when entry ADXΔ < this (0.0 = the ADXΔ<0 cut)
    flip_short_regime_block_regimes: str = "HEALTHY_BULL,CHOPPY_FLAT"  # CSV of BTC regimes to block flip-SHORTS in (ADXΔ<adxd_max gate); empty = filter OFF. Jun 19: STRONG_BULL REMOVED (carve-out — see below).
    # Jun 28 — UNIVERSAL (all-regime) collapsing-pair-ADX block for flip-SHORTS. Distinct from the
    # regime-scoped gate above: flip shorts BYPASS the momentum-short `Pair ADX Dir S: rising` filter,
    # so a flip-SHORT can fire into a pair whose ADX is COLLAPSING (ADXΔ << 0 = the very trend that
    # justified the fade is dying → no downward follow-through → never arms → 20× gaps the SL). The
    # strongest flip-short loser-separator cross-batch (the standing re-eval candidate). 06-28 BEL
    # (STRONG_BEAR, ADXΔ −1.02) = −$195 is the fresh confirm; on that batch the cut blocks 1 (BEL),
    # 0 winners touched, +$195. Block flip-SHORT when ADXΔ < this. SENTINEL −99.0 = OFF (code default);
    # json ships −0.5 (LIVE). Counter FLIP_SHORT_ADXD. ⚠ DISCIPLINE-OVERRIDE: forward N=1 (BEL) — TIGHT
    # REVERT: set back to −99 if would-be-blocked flip-shorts hit ≥50% WR on N≥8 fresh.
    flip_adx_delta_min: float = -99.0
    # Jun 17 (B2) — regimes where flip-SHORTS lose REGARDLESS of ADXΔ → block any-ADXΔ. (Jun 19: EMPTIED.)
    # STRONG_BULL was blocked here on PHANTOM evidence; cross-batch S.BULL FAN-short is actually a WINNER
    # (Jun18 9/78%/+0.46, Jun19 11/82%/+0.24 — two independent windows; H.BULL stays a real live loser).
    # The dangerous S.BULL fades are already fenced by stretch≥2 + fan-spike≥10 + BTC RSI≥60/ADX≥30, so
    # this regime gate was redundant belt-and-suspenders over-blocking the milder (winning) S.BULL fades.
    flip_short_regime_block_any_adxd_regimes: str = "CHOPPY_FLAT"  # CSV; empty = OFF. Block flip-SHORT in these regimes regardless of ADXΔ. Jun 19: emptied (un-block S.BULL FAN-short). Jun 23: ="CHOPPY_FLAT" (operator) — block ALL CHOP FAN-shorts (was only ADXΔ<0 via _regs; the ADXΔ≥0 slice LEAKED and lost). Mechanism: a fade needs downward follow-through; CHOP provides none → never arms → 20× SL whipsaw. Current-stack survivors CHOP = only negative regime (N=7/43%WR/−$317; raw CHOP N=44/43%/−$1352). ⚠ DISCIPLINE-OVERRIDE: N=7 survivors < N≥30, loss 83% in 2 pairs (ALLO/PLAY) which neither clears blacklist individually. TIGHT REVERT: remove CHOPPY_FLAT if would-be-blocked CHOP FAN-shorts hit ≥50% WR on N≥10 fresh.
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
    # Jun 19 — PAIR_RSI_OB per-source regime ALLOW-LIST (overbought-fade short fires ONLY here). Cross-batch
    # S.BULL 76-80% WR vs H.BULL 29-47% — scope to STRONG_BULL only. Empty = source OFF. Decoupled from the
    # FAN flip-short gates above (PAIR_RSI_OB returns early in _flip_filters, never inherits them).
    flip_pair_rsi_ob_short_regimes: str = "STRONG_BULL"  # PAIR_RSI_OB flip-SHORT fires ONLY in these regimes; empty = OFF
    # Jun 20 — PAIR_RSI_OB pair-ADX floor (N=8 DISCIPLINE-OVERRIDE, RAISED 33→40 same day). The 33+ cell
    # decomposed when split: 33-40 = N=5/20%WR/−$237 (a LOSER band) vs 40+ = N=8/88%WR/+$668 — the edge is
    # entirely ADX≥40. Raised the floor to 40 (operator). Still N=8 in-sample + 20× → TIGHT REVERT: set →0
    # (or de-lever lev→0.05) at live N≥15 new fires if the 40+ cell WR≤70% OR avg≤+0.05%. 0 = disabled.
    # Counter FLIP_PAIR_RSI_OB_ADX. (Prior 33-floor cell had already decayed 9/89%/+$698 → 13/62%/+$431.)
    # Jun 21 — RAISED 40→45: in the newly-unchoked BTC-ADX>40 regime the 40-45 band = 17/47%WR/−$605
    # (loser) while 45+ = 17/82%WR/+$347 (+0.20% avg, BE-compat 67%). When BTC trends this hard, only the
    # most extreme pair blow-offs (pADX≥45) mean-revert; 40-45 squeezes. N=17/one-batch DISCIPLINE-OVERRIDE.
    flip_pair_rsi_ob_adx_min: float = 45.0
    # U3-followup (Jun 20): the overbought-fade seed was choked above BTC ADX 40 — an ACCIDENTAL
    # inheritance of the long pipeline's BTC_ADX_GATE_HIGH veto (the fade is a SHORT; the long's
    # ceiling is irrelevant to it, and overbought pairs are richest when BTC trends hardest). This
    # decouples the seed from that veto. off = current (seed choked >40) · phantom = seed phantom-only
    # at >40 (observe, no live trade) · live = seed live at >40. Jun 21: the >40 cohort was PROMOTED to
    # full 20x (de-risk removed in _flip_filters) after its first batch held at the 45 floor. REVERT the
    # whole >40 experiment by setting this to off (stops the >40 seed); the [28,40] cohort is unaffected.
    flip_pair_rsi_ob_btc_adx_high_mode: str = "off"
    # Jun 22 — PAIR_RSI_OB pair EMA13-EMA50 gap ceiling (the parabola guard). The overbought-fade pays only
    # on a pair that's overbought but still NEAR its own 4h trend; a pair already steeply extended above EMA50
    # (gap≥1.0%) is a parabola that keeps ripping → the 20× short never arms (peak ~0) and gaps the SL to ~-1.2%.
    # Latest batch (N=22, in-sample, single STRONG_BULL window): gap≥1.0 = 19/32%WR/-$1080 vs gap<1.0 = 3/100%/+$97
    # → the block flips the cohort -$983 → +$97. Per-trade-proven the losers gap-down (peak<0.20) so an exit can't
    # save them; only this entry block does. DISCIPLINE-OVERRIDE (N=22<30, one window) but mirrors the cross-batch
    # FAN gap≥1.0 filter (flip_short_pair_gap_max, N=16 J17-21, same mechanism) → extends a proven gate to the
    # sister source that previously RETURNED EARLY and skipped it. SEPARATE field (not flip_short_pair_gap_max) so
    # its revert clock is independent of FAN. Counter FLIP_PAIR_RSI_OB_GAP. 0 = OFF (fail-open on missing gap).
    # TIGHT REVERT: set →0 if blocked (gap≥1.0) PAIR_RSI_OB fades show ≥50% WR on N≥10 fresh.
    flip_pair_rsi_ob_pair_gap_max: float = 1.0
    # Jun 19 — pair-RSI floor for flip-SHORTS. Fade quality scales with how overbought the blocked long was.
    # Cross-batch (Jun17/18/19, deduped): RSI<55 = N=21/57%WR/−0.094%/Σ−1.98 (the only consistently-negative
    # zone); RSI≥55 = N=78/65%WR/+0.056%/Σ+4.33 (carries ~all the edge); 60-65 = N=24/71%WR/+0.187%. Block
    # SHORT when pair RSI < this. Operator-directed, N below the locked filter gate → TIGHT REVERT.
    # 0 = OFF (fail-open on missing rsi too).
    # 2026-06-29: market-breadth FLOOR for flip-SHORTS — block a fade-short when the market isn't broadly
    # bearish (no downside tailwind → DOA grind → 20× SL gap). Block flip-SHORT when entry_bear_pct < this.
    # Fine bear-band split (COMBINED in-sample + 06-29 forward): the loss is concentrated ENTIRELY in
    # bear<20 (in-sample 1W/3L/−$219, forward 0W/1L/−$95 = 1W/4L/−$314, NO high-ATR confound — genuine
    # low-breadth DOA fades); bear 30-40 actually WINS (+$146) and 50-80 are the edge, so a <40 floor would
    # forfeit winners — <20 is the only clean cut. The 06-29 币安人生 (bear 17.5, −$95 DOA) is the lead OOS
    # confirm; bear 20 & 32 flip-shorts won. Counter FLIP_SHORT_BEAR_MIN. 0 = off. ⚠ N=5 < N≥30 gate =
    # DISCIPLINE-OVERRIDE (low-stakes, rare ~1-2/wk, kills ~0 winners). TIGHT REVERT: →0 if would-be-blocked
    # (bear<20) flip-shorts hit >40% WR or net-positive on N≥10 fresh. NOTE: addresses a small leak (~−$95/
    # batch); the dominant loss is high-ATR SL gap-through (de-lever roadmap), NOT this.
    flip_short_bear_min: float = 20.0
    flip_short_rsi_min: float = 0.0   # block flip-SHORT when entry pair RSI < this (0 = off)
    # Jul 6: gate revert FIRED on its locked terms (blocked slope>0 phantoms 18·78%WR ≥ 60% on N≥10)
    # → graduated response (operator-directed): instead of un-blocking outright, ADMIT slope>0
    # flip-shorts LIVE with the cell multiplier CAPPED at this value + tag B1H_SLOPEUP (own row in
    # Multiplier Cell Perf + Flip×Regime). 0 = legacy hard block. Caveats on record: phantom avg
    # +0.268 ≈ friction; 13/18 phantoms one bear session; phantoms not net-admissible (6 gates run
    # after this one). 🔒 RE-BLOCK (→0) if live SLOPEUP cohort ≤50% WR or net-negative on N≥10;
    # PROMOTE to full flip size at ≥65% WR & avg ≥+0.15% on N≥20.
    flip_short_btc_1h_slope_admit_mult: float = 0.0
    # Jul 8 — flip-SHORT BTC trend-gap DEPTH gate: block flip-SHORT when BTC (EMA13-50)/EMA50 gap
    # <= this value (BTC deep below its own trend = oversold bounce zone; the faded pump is a market
    # relief squeeze). The ONLY ship-grade survivor of the full 31-dim winner/loser sweep (Jul 8):
    # baseline monotone 5-bucket gradient (<=-0.30 = 25% WR ... -0.10..0 = 100% WR), fresh-window
    # direction-consistent; -0.22 = plateau cut in clean space (-0.20 knife-edge rejected). Blocked
    # 16·44%·-$417 over 10+ dates / kept baseline 27·85%·+$881. COMPLEMENT of the 1h slope gate
    # (overlap 2/65 — depth vs direction). Sentinel 0 = OFF (active when < 0). Ship -0.22 in json.
    flip_short_btc_trend_gap_min: float = 0.0
    # Jul 8 — TG_SHALLOW multiplier cell (the depth gate's twin, operator-directed 2× ship): when a
    # flip-SHORT fires with BTC gap in [shallow_min, 0) — the monotone gradient's top bucket, 10/10
    # combined (baseline 8·100%·+0.53·+$455) — apply this INVEST multiplier (lev stays 1×). 0/1 = off.
    # ⚠ Double override (N=10 << 30; skips 1.5× staging). REVERT: ✗ HARMFUL (net-neg N>=5) -> 1.0;
    # ⚠ DRAG -> 1.5. Tag [TG_SHALLOW] = own row in Multiplier Cell Performance — SHORTs.
    flip_short_tg_shallow_mult: float = 0.0
    flip_short_tg_shallow_min: float = -0.10
    flip_short_tg_shallow_max: float = 0.0   # zone upper bound (gap < max); 0 = the baseline bucket edge
    flip_short_tg_shallow_lev_mult: float = 1.0  # leverage multi for the cell (ship 1.0 = invest-only; BE-compat unmeasured)
    # Jul 10 SHIP (live JSON = 2.0/15.0/1.0): NEGDI15 "sellers-present" multiplier cell. Flip-SHORT
    # with pair −DI (downward directional movement) ≥ negdi_min takes negdi_mult INVEST. Mechanism:
    # a flip fades a fresh alt pump; −DI high = sellers ALREADY active = the fade has fuel; −DI low
    # = uncontested vertical pump = you're the first seller (squeeze risk). Baseline cell 17·100%WR·
    # +$971 (~+0.4%/tr), 13 dates / 15 pairs, era-consistent (12/12 pre-06-30, 5/5 post); ALL 6
    # sleeve losers sit below −DI 15. Deliberately NOT a <15 block (that flank = 57%-WR mixed —
    # locked rule: multiply winners, never block them; <15 block is WATCHLIST-gated only).
    # ⚠ DOUBLE OVERRIDE: N=17 < 30 W-gate AND skips 1.5× staging (operator-directed; TG_SHALLOW
    # precedent). Distinct from the Jul-1 REFUTED DI-SPREAD (+DI−−DI inverted OOS; raw −DI is
    # 2D-verified independent: −DI-hi wins at BOTH +DI levels, W/L spread medians ≈ equal).
    # 🔒 TIGHT REVERT (cell verdict machinery): ✗ HARMFUL (net-neg on N≥5 fresh) → 1.0× ·
    # ⚠ DRAG (Δ$ vs BL <−$1) → 1.5×. Tag [NEGDI15] in 💰 Multiplier Cell Performance — SHORTs.
    flip_short_negdi_mult: float = 0.0      # invest multi for the cell (0/1 = off; ship 2.0)
    flip_short_negdi_min: float = 15.0      # −DI floor defining the cell (ship 15.0)
    flip_short_negdi_lev_mult: float = 1.0  # leverage multi (ship 1.0 = invest-only; BE-compat unmeasured)
    flip_short_btc_1h_slope_max: float = 99.0   # block flip-SHORT when BTC 1h EMA20 slope > this (99 = off; ship 0.0). Jul 3: THE regime gate — fading alt pumps loses when BTC's HOURLY trend is rising (pumps are real in a recovery and run over the short) and pays when falling (pumps are exhaustion). Two-period direction-CONSISTENT (the only flip separator of 8 tested that did not invert): baseline slope>0 = 17fl/65%WR/−$73 vs slope≤0 = 29/76%/+$774; fresh Jun30-Jul3 slope>0 = 9/33%/−$405 vs slope≤0 = 7/71%/+$51. Combined blocked N=26, Δ+$478. Mechanism = the momentum-short btc_1h_slope_max(+0.1) gate that flips BYPASSED (parity fix). ⚠ N=26 < N≥30 = near-gate ship. Counter FLIP_SHORT_BTC1H_SLOPE. TIGHT REVERT: →99 if would-be-blocked (slope>0) flips run ≥60% WR on N≥10 fresh phantoms.
    # Aug-23 (21): FAN_RATIO_GATE shorts need a BEARISH BTC — refuse when BTC's distance from its 5m EMA13 is ABOVE this
    # (i.e. BTC at/above its EMA13 = flat tape, no downtrend to lean on). Pooled 41 fan-gate shorts: losers sit at −0.04%
    # vs winners −0.12%; NEUTRAL-BTC fills 4/4 lost. Blocked cohort: current 6·33%·−$296 (3 dates), master 7·71%·−$39 —
    # never positive. Before/after: current 10·50%·−$131 → 4·75%·+$165; master 31·81%·+$589 → 24·83%·+$629.
    # Discipline-override (13 pooled, ~4 windows). Blank/None = off. DECISION_LOG 2026-08-23 (21).
    # 🔒 revert: kept fan-gate shorts < 60% WR on N≥10 fresh fills → off.
    flip_fan_btc_ema13_max: Optional[float] = -0.08
    flip_short_quality_min: float = 2.0   # block flip-SHORT when entry quality score < this (so =2 blocks score ≤1). 0 = off. Jun 25: extends the global Entry-Quality-Score floor (already blocks ≤1 for NORMAL entries: validated N=95/34.7%WR/−$684) to the flip-short sleeve, which BYPASSES it. Cross-batch FAN flip-short (deduped, current stack): score is monotonic (1→4 = 56/64/76/80% WR, −0.17→+0.56% avg); score≤1 = N=18/56%WR/−2.98%/8 dates (the only negative band), loss DIFFUSE (16 pairs, top 21% — not pair-concentrated). Score 0 ≈ empty (N=2). Confirmed on 06-25 batch (score≤1 = 3/3 losers, −$249, incl. SAHARA −$145 gap-through; sleeve −$337→−$88). ⚠ N=18 < N≥30 gate = DISCIPLINE-OVERRIDE, but the score≤1 threshold itself is already globally validated — we only close the flip bypass. Counter FLIP_SHORT_QUALITY. TIGHT REVERT: →0 if would-be-blocked (score≤1) flip-shorts run ≥55% WR on N≥10 fresh.
    # Jun 21 — pair EMA13-EMA50 gap ceiling for flip-SHORTS. Refuse to fade a pair already steeply
    # extended above its OWN 4h trend (gap = (EMA13-EMA50)/EMA50 %): a parabola that keeps ripping →
    # the 20× short gaps the SL to ~-1.2%. Cross-batch FAN survivors (Jun17-21 deduped): gap≥1.0 =
    # N=16/44%WR/-0.359%avg/Σ-$461, net-negative every batch; the 0-1.0 band is the fade sweet spot
    # (19/87%WR/+0.45%). ONE-SIDED (positive tail only): a big NEGATIVE gap is with-trend momentum that
    # WINS (≤-1.5 = +0.79%), so it is NOT blocked. The flip-side of the live non-flip pair_trend_short
    # filter, tuned for flips (which want MILD extension to fade). N=16 / DISCIPLINE-OVERRIDE (< the
    # N≥30 gate). Counter FLIP_SHORT_PAIR_GAP. 0 = OFF (fail-open on missing gap too).
    # TIGHT REVERT: →0 if blocked gap≥1.0 flip-shorts hit ≥50% WR on N≥10 fresh.
    flip_short_pair_gap_max: float = 1.0   # block flip-SHORT when pair EMA13-EMA50 gap% ≥ this (0 = off)
    # Jun 17 — MIRROR of the short gate for flip-LONGS. A flip-LONG fades a blocked SHORT -> goes LONG;
    # in a STRONG_BEAR that's long-into-the-trend (AAVE/TAO this batch: 2/0%WR/-$220, straight to SL).
    # The observed long losers were ADXΔ-AGNOSTIC (ADXΔ +1.5, regime was the killer) → adxd_max default
    # 99 = REGIME-ONLY block; lower it later only if a long ADXΔ cell proves out cross-batch.
    flip_long_regime_block_adxd_max: float = 99.0   # block flip-LONG when entry ADXΔ < this (99 = regime-only, no ADXΔ cut)
    flip_long_regime_block_regimes: str = "STRONG_BEAR,HEALTHY_BEAR,CHOPPY_FLAT"  # CSV of BTC regimes to block flip-LONGS in; empty = filter OFF
    # Jun 27: HARD-DISABLE the flip-LONG side entirely (fade blocked LONGs→SHORT only; never fade a
    # blocked SHORT→LONG). Flip-LONG is a net-negative micro-sleeve: full-history N=8/5W-3L/62%WR but
    # net −$297 (1:8 R:R — wins +$8..27, the 3 losers VELVET/HYPE/DYDX gap to −0.7/−1.2 SL). The regime
    # block removes the H.BEAR/CHOP losers but the residual is H.BULL countertrend-longs (long an
    # oversold/bottom-of-range pair that keeps falling): fresh DYDX −$115 (06-24) + XPL −$164 (06-27,
    # RSI 32 / range 6%) = 0/2 H.BULL, −$279. ⚠ DISCIPLINE-OVERRIDE: N=2 fresh < the N≥10 gate, shipped
    # on a clean mechanism. Default True (code-safe); json sets False (operator opted in). Counter
    # FLIP_LONG_DISABLED. TIGHT REVERT: set True if blocked flip-LONGs would be ≥55% WR AND net-positive
    # on N≥8 fresh (phantom/passthrough still observes the blocked side). → DECISION_LOG/CURRENT_STATE 2026-06-27.
    flip_long_enabled: bool = True  # False = disable all flip-LONGs (short-fades only)
    # Jun 17 — fan-SPIKE block (ALL flip sources, not just FAN). Block the flip when the pair's
    # entry fan ratio (|EMA5-8 gap| / |EMA8-13 gap|) >= this — a violently-accelerating parabolic
    # fan that the fade gets run over by (never arms, straight to SL). Cross-batch N=3, 0% WR,
    # ~-1.0% (ASTER 5.7/VELVET 28.3 [12-20-07] + ALLO 13.2 [06-16 ref], 3 pairs) — clears the
    # >=3-sample direction-consistent bar; mirrors the already-live fan_ratio_block_long 5.0-99.
    # Threshold is specifically >=5 (the 2-5 band CONTRADICTS cross-batch). 0 = off.
    flip_fan_spike_max: float = 5.0           # block any flip when pair fan ratio >= this (0 = off). TIGHT REVERT: re-open if fan>=5 flips >=40% WR on N>=5 fresh
    flip_fan_runner_strpk: bool = True        # exit FAN flips via the SHORT runner stretch-trail (strpk, arm 0.45/K0.5) instead of trailing-like-a-long. Reuses runner_trail_short_* params
    flip_runner_strpk_shorts: bool = True     # Jun 16: extend the SHORT runner stretch-trail (strpk) to the NON-FAN flip short sleeves too (PAIR_RSI_OB, LONG_UNMATCHED_ONLY). A flip short runs strpk if FAN+flip_fan_runner_strpk OR non-FAN+this. = strpk for ALL flip shorts
    flip_fan_qs_cell: str = "3:70:60-90:1.0:1.0"  # qs_min : bear_min : range_lo-range_hi : size_mult : lev_mult — the FAN flip-SHORT "winner cell" (quality_score ≥ qs_min AND entry_bear_pct ≥ bear_min AND range_lo ≤ entry_range_position ≤ range_hi). Jun 26: REPLACED the old BTC-RSI×ADX mult cell (flip_fan_mult_rule, retired — it was never used at 1×). Code default 1.0:1.0 = INERT; json SHIPPED 2.0:1.0 on 2026-06-27 = 2× SIZE / 1× LEV (double margin at the SAME 20× — liquidation distance unchanged; lev was NOT touched, since 2× lev → 40× halves the gap-to-SL distance). Cell evidence: current-stack (pADX≥21) winner cell = N=6/6W-0L/100%WR/+0.69%/+$433 (the 2 "BE-compat-fail" gap-losers HUSDT/HMSTR are PRE-pADX-floor GHOSTS, blocked live → BE-compat objection RETRACTED, 0 losers to amplify). ⚠ DISCIPLINE-OVERRIDE: forward-N=1 (INJ +$36) << N≥30 gate, operator-directed; carries a TIGHT revert. SHORT flips only (a flip-LONG ignores it, sizes 1×). Empty = off. TIGHT REVERT: set size→1.0 if the cell runs WR<70% OR avg≤+0.05% OR Total$ negative on N≥5 fresh fires, OR any single 2× cell trade gaps the SL past ~−1.0%. ✅ REVERTED 2.0→1.0 on 2026-06-28: the INSTANT gate FIRED — WIF flip-short (winner-cell 2×) gapped the −0.70 SL to −1.16% close (−$321 at 2×, would've been −$160 at 1×) in the 7:30-11am chop-whipsaw. Forward record N=2 = INJ +$36 / WIF −$321 = −$285. Pre-committed gate, non-discretionary. Cell now INERT (1×), back to track-only. → DECISION_LOG/CURRENT_STATE 2026-06-28.
    # === Bull-Long Entry Sleeve (Jun 18) — the BUILD-side twin of the flip sleeve.
    # When a LONG PASSES the fan gate (low fan ratio) in a HEALTHY_BULL regime, open the
    # SAME direction (a real momentum-style long, NOT a fade) and let it run on the NORMAL
    # long exit stack (per-level trailing, ATR-widened SL) — it is NOT tagged _is_flip.
    # Multipliers default 1.0 (no amplification); leverage stays the normal STRONG_BUY level.
    # Tagged entry_strategy="BULL_LONG"; bypasses the long_unmatched_only + pattern-cell
    # entry blocks (it is explicitly a trend-build, not a pattern-matched late long). All
    # hard risk controls (max-open, existing-position, cooldown, liquidity caps) still apply.
    # TO REMOVE: grep "BULL_LONG" / "bull_long" + the main.py bull-long perf blocks + the UI.
    bull_long_enabled: bool = False                    # master toggle. Jun 25: DISABLED (operator) — the 20× re-lever tripped its instant-revert gate (4 of 5 hit the −0.70 SL, 1W/4L/−$299) and the deep cross-batch dig confirmed the sleeve is structurally "arm-or-die" (38% never arm → straight to SL) with NO entry separator that survives (BTC RSI/ADX/ADXΔ all refuted; range-position only a weak cross-batch tendency that didn't help this batch). Windowed cell is only +0.16%/trade at 1× — too thin/variance-heavy for leverage, and unfixable by entry filters. Long-side scaling belongs to the parked Donchian module, not this scalp sleeve. Per-regime fan window + flat SL config retained but inert. RE-ENABLE only with a genuinely new edge (e.g. an arm-predictor) — not another leverage attempt.
    # Jun 21 — converted to a 1× OBSERVATION ARM. The cross-batch dig (N=52) showed fan bucket & regime
    # are NOT the driver of bull-long outcomes — BTC momentum/extension at entry is: BTC 30m-RSI Δ≥+12 =
    # 26/73%WR/+0.149% (the winning band) and BTC EMA13-EMA50 gap≥+0.06 = 22/27%WR/−$430 (the whole loss).
    # So we OPEN the aperture (broaden regimes + widen fan) at 1× and TRACK those two cells × regime to
    # confirm the real gate before re-levering. Bear regimes EXCLUDED (build-side long-into-bear = the
    # opposite thesis; flip-LONG bear evidence = 2/0%/−$220). REVERT: any single bull-long gaps past
    # ~−1.0% OR the arm <40% WR at N≥20 → cut back to bull_long_regimes=HEALTHY_BULL, fan 1.65–3.0, lev 0.05.
    bull_long_regimes: str = "STRONG_BULL,HEALTHY_BULL"  # obs-arm regimes (Jun 22: dropped CHOPPY_FLAT,CHOPPY_WEAK — CHOP was the whole live drain [18/56%/-0.247%/-$482 de-mux; >½ sleeve volume]; bull regimes alone = 14/64%/+0.155%. No bear)
    bull_long_fan_max: float = 3.0                     # upper fan bound (Jun 24: 5.0→3.0 — match the union of the per-regime windows; the ≥3.0 tail is a confirmed loser. Inert while both bull regimes are mapped in bull_long_fan_by_regime [that per-regime window overrides this]; set to the union purely so the fallback is consistent, not contradictory.)
    bull_long_fan_min: float = 1.35                    # lower fan bound (Jun 24: 0.85→1.35 — match the union of the per-regime windows; the 0.85-1.35 band was ~the entire sleeve loss. 0 = disabled). NOTE: when bull_long_fan_by_regime has an entry for the gated regime, that per-regime [min,max] OVERRIDES these globals; these remain the fallback for any allowed regime NOT in the map.
    bull_long_fan_by_regime: str = "STRONG_BULL:1.35-2.0,HEALTHY_BULL:2.0-3.0"  # PER-REGIME fan window (CSV of REGIME:min-max). Jun 24: cross-batch (N=106 deduped, current sleeve) showed the winning fan band is regime-specific — STRONG_BULL × 1.35-2.0 = 11/73%/+5.1% (3 dates), HEALTHY_BULL × 2.0-3.0 = 24/67%/+3.9% (6 dates), while HEALTHY_BULL × 0.85-1.35 = 25/28%/−12.0% was ~the entire sleeve loss (the band the 06-23 widening imported). Indicators (BTC RSI/ADXΔ/PairTrend) showed NO separator within either regime (flat in HEALTHY, inverted in tails, sign-flipping/noise in STRONG N=16) — fan×regime is the only axis. Empty = OFF (fall back to global bull_long_fan_min/max). Engine: _maybe_open_bull_long resolves regime then applies this window. In-sample-haircut caveat: positive cells N=11/24, partly in-sample → still a 1×-obs sleeve. REVERT: drop a regime's entry if its windowed cell runs ≤45% WR on N≥10 fresh.
    bull_long_size_mult: float = 1.0                   # investment multiplier (1.0 = no amplification)
    bull_long_max_concurrent: int = 3                  # max concurrent OPEN bull-longs (0 = uncapped). Jun 23: reserve book slots for higher-conviction MOMENTUM/UNMATCHED longs — BULL_LONG is a 1×-obs sleeve and a bull cluster (e.g. the 06-23 batch opened 5-7 at once) must not monopolize the max-5 book and block proven longs. Counter BULL_LONG_MAX. Engine gate in _maybe_open_bull_long.
    bull_long_fixed_sl: float = -0.70                  # flat fixed SL% for bull-longs (negative = active, 0 = off → normal ATR-widened exit). Jun 23 revival test: re-sim showed flat -0.70 beats the live -1.20 ATR-widened SL for this cohort (caps the gap-through tail). Stamped onto the order in open_position's bull_long block → enforced by the existing PATTERN_FIXED_SL exit path (fires before -1.20 engages). Arm/trail unchanged.
    bull_long_lev_mult: float = 1.0                    # leverage multiplier (1.0 = 20×, 0.05 = 1× obs). Jun 24: 0.05→1.0 RE-LEVERED to 20× (operator-directed) ON TOP of the new per-regime fan window (bull_long_fan_by_regime = S.BULL 1.35-2.0 / H.BULL 2.0-3.0), which excludes the loser bands the prior 20× attempts traded. ⚠️ DISCIPLINE-OVERRIDE (acknowledged): below the locked re-lever gate (N≥30 cumulative + WR≥65-70%/avg≥+0.10% across ≥2 batches) — the windowed cells are N=11 (S.BULL 1.35-2.0, 73%) / N=24 (H.BULL 2.0-3.0, 67%), partly in-sample. The 06-22 20× re-lever tripped its instant gate the very next batch — but that was on the OLD wide fan (incl. the inverting 1.65-2.0 / sub-1.35 bands); this ships only the cleaned regime×fan cells. TIGHT REVERT (override-grade): INSTANT de-lever →0.05 if 3 of the first 6 new bull-longs hit SL OR any single gaps past ~−1.0%; de-lever at N≥10 fresh if windowed WR≤60% OR avg≤+0.05%. ─── history: Jun 22: 0.05→1.0 (N=8/88% fresh-reset). Jun 23: DE-LEVERED 1.0→0.05 — instant gate tripped (7 bull-longs/29%WR/−$335 all H.BULL, 4 of first 6 hit SL, two gapped past −1.0% [TNSR −1.19, ALLO −1.01]; fan 1.65-2.0 inverted 5/20%/−$264).
    # Bounce-Long sleeve (Jun 19, 2026) — oversold-WASHOUT dead-cat bounce LONG. Fades the
    # BTC_RSI_ADX_CROSS oversold short-block: in a bear, a SHORT blocked because BTC is washed out
    # (the validated BTC RSI × BTC ADX cells) → open a REAL LONG to catch the bounce. NORMAL long
    # exit (not _is_flip), own sleeve (never routes through _flip_filters → the flip-long bear veto
    # does NOT apply). Validated phantom BTC_RSI_ADX_CROSS LONG: N=21, 95% WR, 0% SL, ALL H.BEAR.
    # TIGHT cells only: 25-30:20-25 (89% WR) + 30-35:15-20 (100% WR). 1× observation (lev_mult 0.05).
    # TO REMOVE: grep "BOUNCE_LONG" / "bounce_long" + the main.py bounce-long perf blocks + the UI.
    bounce_long_enabled: bool = False                  # master toggle (Jun 23: DISABLED — cross-batch N=8 = 2W/6L/−0.443%, no-edge counter-trend long; both entry theses FAILED to separate W/L: dist-from-EMA13 (win −0.60 sits in the loser cluster, only the WU −1.80 outlier supports "extended=bad") AND BTC 30m-RSI Δ (winners at −16 deeply-negative, the mildest deltas −7/−8 were losers → exhaustion-gate inverted). Phantom seed stays decoupled → keeps tracking for a future re-enable if a cross-batch-stable separator emerges)
    bounce_long_regimes: str = "HEALTHY_BEAR"          # CSV of BTC regimes the sleeve fires in
    bounce_long_btc_cells: str = "25-30:20-25,30-35:15-20"  # TIGHT (BTC RSI lo-hi : BTC ADX lo-hi) washout cells; empty = OFF
    bounce_long_size_mult: float = 1.0                 # investment multiplier (1.0 = no amplification)
    bounce_long_lev_mult: float = 0.05                 # leverage multiplier (0.05 × 20× base = 1× live observation)
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
    # Jun 28, 2026 — MOMENTUM-SHORT dead-tape block (DISCIPLINE-OVERRIDE, N=5). Block momentum
    # SHORT entries when BTC ATR% < this. The two existing ATR floors miss this band: flip_fan_btc_atr_min
    # (0.10) is FLIP-only; btc_atr_btc_adx_filter_short (<0.10) also needs BTC ADX≥30. Momentum-shorts at
    # ATR 0.10-0.12 with normal ADX slip past both and die: across ALL 26 dedup momentum-shorts, the
    # ATR<0.12 band = N=5 / 0% WR / 100% DOA / -$638 (AAVE -242, PUMP -115, ONDO -106, NEAR -98, HYPE -77),
    # spanning the 06-27 22:12→06-28 10:46 dead-BTC pockets. ZERO winners ever killed (every momentum-short
    # win had ATR≥0.132). Asymmetric safety: the band contains only DOA shorts (BTC too quiet to fall).
    # On the 5 curated current-filter batches it fires 0× (neutral — no momentum-short dipped <0.132), so it
    # can only act in genuinely dead tapes. MOMENTUM-only (flip-shorts WIN in 0.10-0.12, +$340 — do NOT extend
    # the flip floor). Counter MOMENTUM_SHORT_LOATR. 0 = off. TIGHT REVERT: →0 if ANY momentum-short in the
    # ATR<0.12 band closes ≥+0.30% (first time it would kill a winner), OR if blocked-zone WR ≥40% on N≥10 fresh.
    momentum_short_btc_atr_min: float = 0.12
    # Jun 28, 2026 — MOMENTUM-SHORT weak-capitulation block (DISCIPLINE-OVERRIDE, N=4). Block momentum
    # SHORT when ALL three hold: range_position < range_max (capitulation entry near the low) AND pair
    # ATR% < atr_max (low volatility) AND pair ADX < padx_max (weak trend). The triple-weak signature =
    # a short with no follow-through → DOA straight to SL. On the COMBINED dedup pool the cell = N=4 (or 5
    # incl. the low-pADX C1 XLM) / 0%WR / all peak<0.10 / −$263..−$502, ZERO winners (the SUI +2.08% runner
    # has pADX 31.7, safely above padx_max). Blocks BY BEHAVIOR not by tag: a C1 capitulation short with
    # pADX<28 (e.g. XLM, the inside-band-2× DOA hole) IS caught here, while trend-backed C1 (pADX≥28) still
    # fires to resolve its N≥5 verdict gate. Distinct from momentum_short_btc_atr_min (that's BTC dead-tape;
    # this is PAIR weak-trend). Counter MOMENTUM_SHORT_WEAKCAP. enabled=False → off. TIGHT REVERT: disable
    # if ANY blocked-band trade closes ≥+0.30%, OR blocked-zone WR ≥40% on N≥8 fresh.
    momentum_short_weakcap_enabled: bool = True
    momentum_short_weakcap_range_max: float = 15.0   # block when entry range position % < this
    momentum_short_weakcap_atr_max: float = 0.45     # AND pair ATR% < this
    momentum_short_weakcap_padx_max: float = 28.0    # AND pair ADX < this
    # Jun 30, 2026 — MOMENTUM-SHORT W1-regime block (DISCIPLINE-OVERRIDE, N=20). Block a momentum SHORT that
    # matches pattern W1 ("HighConv trend") when entry BTC regime is in this comma-separated list. W1 fired as
    # a SHORT drains specifically in HEALTHY_BEAR: 2 direction-consistent windows (SCREENED_BASELINE 06-16→28 +
    # 06-29/30 batch) = W1 mom-short HEALTHY_BEAR N=20 / 40%WR / -$650 / avg -0.265%, while the non-W1 mom-short
    # CONTROL in the same regime is breakeven+ (N=7 / +$24 / +0.014%) → discriminator is W1, not the regime
    # (confound passed). Loss diffuse (no pair ≥60%). STRONG_BEAR W1 WINS (N=4/75%/+$229) → NOT listed (exempt).
    # Momentum-shorts reach the entry path; flips bypass (gated by _flip_filters). Counter MOMENTUM_SHORT_W1_REGIME.
    # Empty = off (filter universal-off). TIGHT REVERT (override): clear the list (→'') if this block's phantom
    # (LONG fade) goes net-NEGATIVE on N≥10 fresh (= the blocked shorts would have won).
    momentum_short_w1_block_regimes: str = ''  # REVERTED 2026-06-30 (was 'HEALTHY_BEAR'). Failed the cross-period
    # robustness test: the SAME W1+HEALTHY_BEAR+short entries WON +$1175/65%WR ≤06-13 and LOST -$650/40%WR 06-16→30
    # — the regime LABEL did not capture what changed (follow-through: peak% halved +0.61→+0.34, 73%→45% reached
    # +0.30). A robust filter encodes a stable entry-measurable condition; "HEALTHY_BEAR" did not → overfit-to-window.
    # Replaced by momentum_short_pair_vol_max (the one separator that holds in BOTH periods). Empty = off.
    # Jun 30, 2026 — MOMENTUM-SHORT high-pair-volume block. Block a momentum SHORT when entry pair-volume ratio
    # (current vol vs its own recent avg) >= this (0 = off). Mechanism: shorting into HIGH pair volume = climactic/
    # exhaustive move → bounce → fails to follow through; LOW pair volume = orderly continuation → follows through.
    # This is the ONLY entry variable that separates mom-short winners/losers WITHOUT inverting across periods:
    # pair_vol<1.0 = 69%WR/+$392 (06-16→30) AND 64%WR/+$449 (≤06-13); pair_vol>=1.0 = 28%WR/-$732 recent AND
    # net-negative ≤06-13 → blocking >=1.0 is +EV in BOTH windows (unlike W1, which was +EV only recently).
    # Momentum-only (flips bypass — handled in _flip_filters). NOTE: this is a MAX (block at/above); distinct from
    # the legacy pair_volume_threshold_short=1.1 which is a MIN (require >=, the OPPOSITE) and stays OFF. Counter
    # MOMENTUM_SHORT_PAIRVOL. DISCIPLINE-OVERRIDE (N=34 mom-short universe). TIGHT REVERT: set 0 if pair_vol>=1.0
    # mom-shorts come back >=50% WR AND net-positive on N>=15 fresh, OR if pair_vol<1.0 (the kept side) drops <55% WR.
    momentum_short_pair_vol_max: float = 1.0
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
    rsi_adx_multiplier_hard_cap: float = 2.5  # Investment-side hard cap (was the single cap pre-May-21).
                                              # Jul 26: 2.0 -> 2.5 — the QUIET BOOST (2.5x at PVR<0.68) was
                                              # silently clamped to 2.0 by this cap (review Critical: the
                                              # ship was a no-op). 2.5 = exactly the highest earned rung;
                                              # the future 3x step must raise this AGAIN consciously.
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
    # HARD TP (Jul 20, 2026) — flat profit cap, BOTH directions, parallel to the full exit
    # stack (runner trail untouched). Peak-based CF: baseline +$519 / Jul-20 batch +$338
    # norm; plateau 0.8-1.3% positive BOTH eras (0.7% flips negative on baseline = floor;
    # 1.0% beats 0.9% in both). Mechanism: runner trail is condition-based and captures
    # only ~35-50% of peak on wick round-trips (DEXE +3.64→+0.43 anatomy); TP is the
    # resting-order harvest of that class. ATR-scaling refuted (all k worse, 0.5×ATR
    # direction-inconsistent); atr05-leash combo refuted (cannibalize, −$257 BL).
    # 🔒 Revert gate: N≥15 fires — revert if forfeited-runner tail (post-exit
    # continuation in the Regret table) exceeds cumulative saves.
    hard_tp_enabled: bool = True
    hard_tp_pct: float = 1.0               # exit the tick pnl% reaches this (0 = disabled)
    # Jul 22 (operator-directed mechanism swap): HARD_TP LADDER — per-side rising profit
    # floors replace the flat cap, NO upper cap (MIRA +19.1% anatomy: any cap forfeits the
    # tail; ladder floors only fire on the way DOWN). Rungs "trigger:offset,...": peak
    # crosses trigger -> floor locks at trigger-offset (monotone). Baseline CF (exact-ish,
    # stepped floors are near-path-independent): per-side L1 (L 1.25 / S 1.00) = +$474 vs
    # flat-cap +$519 — traded ~$45 of steady wick-capture for unbounded tail upside +
    # DEXE-class collapse insurance (3.64%->0.43% actual; ladder floor 2.40). Runner trail
    # runs in PARALLEL (upside engine); ladder is the collapse floor. Empty string = legacy
    # flat hard_tp_pct (the revert path). 🔒 Revert gate: back to flat 1.0 (empty ladders)
    # if N>=10 ladder-managed fires (peak >= first trigger, >=3 dates) underperform the
    # exact flat-1.0 CF by >=$100 norm.
    hard_tp_ladder_long: str = "1.25:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80"
    hard_tp_ladder_short: str = "1.0:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80"
    # Jun 9, 2026 — "keep only unmatched longs". 4-cohort analysis (10-pool, current stack):
    # the LONG pattern library uniformly selects for LOSERS (every C/W pattern net-negative:
    # W6 −$574, W2 −$480, C7 −$261 demux), while TRULY-UNMATCHED longs (no C, no W) are the
    # edge (N=39, 85% WR, +$471 demux). When True, block any LONG matching ANY C or W pattern
    # (counter LONG_UNMATCHED_ONLY) — keep only the unmatched runner cohort. Pair this with
    # atr_low_fixed_tp_long_enabled=False: the fix-TP caps the pop-and-fade (matched) cohort,
    # but unmatched longs RUN (54% peak ≥0.40) — capping them at +0.25 strangles the edge, so
    # disable it and let them trail. (Coupling: if you re-enable matched longs, re-enable fix-TP.)
    long_unmatched_only: bool = False
    # Jul 10 SHIP (live JSON = 0.90): UNMATCHED-LONG crowded-entry DE-MUX. When an UNMATCHED long
    # would take the 2× cell multiplier but entry pair_volume_ratio ≥ this value, size at 1×
    # instead (sizing only — entry NOT blocked). Evidence: pool ladder is near-monotone — PVR
    # <0.90 = 29W/3L while the ≥0.90 zone is 10 trades · 60% WR · net-NEGATIVE at both sizings
    # (✗ HARMFUL sub-cell per the locked multiplier-verdict gate; catches HYPE/ME/PYTH/LDO).
    # Mechanism: PVR ≥ 0.90 = the volume burst already happened = buying someone's exit.
    # 🔒 Revert → 0 (full 2×) if fresh zone trades run ≥70% WR and net-positive at N≥8. 0 = off.
    long_unmatched_mult_pvr_max: float = 0.0
    # Jul 26 (operator patron fix): the de-mux TARGET multipliers are configurable, not hardcoded
    # 1x/1x — same Invest/Lev patron as every other sizing cell. Defaults reproduce the Jul-10
    # behavior exactly (full de-mux to 1x/1x). Values <=0 are coerced to 1.0 (a zero here would
    # zero the position). Sub-1x targets are legitimate (e.g. the PVR>=0.93 escalation read may
    # earn a sub-1x cap) — that decision stays behind its own gate, this only makes it a UI edit.
    long_unmatched_demux_inv_mult: float = 1.0
    long_unmatched_demux_lev_mult: float = 1.0
    # Jul 26: QUIET BOOST (operator-directed DISCIPLINE-OVERRIDE at N=19 < 30 gate — acknowledged;
    # tighter-than-standard revert below). The OPPOSITE end of the same PVR dial: UNMATCHED longs
    # entered on a QUIET book (pair-vol ratio < pvr_max) size at quiet_mult INVEST (lev unchanged —
    # the BE-compat rule blocks any lev-stacking until the cell shows real losses to test against).
    # Evidence: C5 quiet cohort 19-0 · avg +0.529% · 14 dates · top pair 24%; full PVR curve
    # monotonic with the cliff at 0.93 (quiet 100% / 0.68-0.90 ~78-83% / >=0.93 42%·-0.392).
    # Take-the-max semantics: replaces (not stacks) the UNMATCHED 2x for qualifying trades.
    # 🔒 TIGHT REVERT (pre-committed): -> 0 (back to 2x) if N>=8 fresh fires at 2.5x run WR<=60%
    # or cumulative dollar-delta vs 2x < 0; TRIPWIRE: any TWO never-positive quiet losses ->
    # immediate revert without waiting for N=8. 3x step / leverage route / boundary move: only at
    # the C5+step-gate merged read (N>=30) with BE-compat on observed losses. 0 = off.
    long_unmatched_quiet_mult: float = 2.5      # INVEST multiplier (replaces the UNMATCHED 2x)
    long_unmatched_quiet_lev_mult: float = 1.0  # LEV multiplier — KEEP 1.0 until BE-compat passes
                                                # on observed quiet losses (locked rule; 19-0 = untestable)
    long_unmatched_quiet_pvr_max: float = 0.68
    # ⚡ Aug-10 CROWD-SPRINT DE-MUX (operator ship; the 3-era unmatched "Rest" hunt): de-size to
    # 1×/1× when global-vol ratio > gvr_min ∧ BTC EMA20 slope > slope_min — the FOMO window where
    # the fan trigger is beta noise, not pair-specific flow. 3-era window: 19·58%·−$714 pooled
    # (BASE blocked slice +$267 = ACT alone → BLOCK refused per the cross-pool rule; de-size =
    # keep the claim, withdraw the conviction). Takes precedence over the quiet boost.
    # 🔒 READ (frozen): N≥10 fresh window fires — WR≤45 ∧ Σ<0 → escalate to BLOCK proposal;
    # WR≥60 ∨ Σ>0 → revert the de-mux. Counter row = [UNMATCHED_SPRINT_DEMUX] logs. 0 = off.
    long_unmatched_sprint_demux_gvr_min: float = 0.74
    long_unmatched_sprint_demux_b20slope_min: float = 0.07
    # Jul 6: W2 RE-ENABLE, 1h-rising conditioned (operator-directed; first matched-long cell back
    # since the Jun-9 block). Admit a W2-matched long (macro tailwind; NO C co-match) when BTC 1h
    # slope ≥ this value. Evidence: historical live W2 longs split hard on 1h — rising ≥+0.05 =
    # 29·72%WR vs pullback ≤−0.05 = 14·14%·−0.55 (refute-only history REFUTES unconditional
    # re-enable, does NOT refute the conditioned cell); current-stack phantoms W2 ≈10·90%·+0.39
    # (> friction); theory: W2 = continuation-with-macro — needs the hourly engine RUNNING
    # (mirror: W6 prefers the pullback — same variable, opposite sign, mechanism-coherent).
    # ⚠ phantom N ≈10 < the registered N≥20 gate = operator override. Trades at the W2 pattern
    # cell's 1× (no UNMATCHED 2×). 🔒 TIGHT REVERT: →99 (off) if live W2-rising longs ≤50% WR
    # or net-negative on N≥8. 99 = off; ship 0.05.
    long_w2_reenable_1h_min: float = 99.0
    # Jul 6: W6 RE-ENABLE, 2D-conditioned (operator-directed; second matched cell back). Admit a
    # W6-matched long (laggard catch-up; NO C co-match) when BTC 1h slope ≤ long_w6_reenable_1h_max
    # (the dip = the discount) AND EMA5 stretch ≥ long_w6_reenable_stretch_min (the laggard is
    # actually waking up). Head-to-head on identical rows (dedupe_pool_FULL, refute-only):
    # pullback∧thrust = 23·78%WR (eraA 8·100%·+0.43 / eraB 15·67%) — the ONLY W6 cell consistent
    # in both eras; pullback alone FAILED the era test (eraB 46%); thrust alone 65%. Mirror logic
    # vs W2: W2's signature includes pair momentum (thrust redundant); W6's excludes it (thrust
    # mandatory). ⚠ WEAKEST override yet: 2D cell search-derived, eraB $≈breakeven under OLD
    # exits, phantom N=2. Trades at the W6 cell 1×. 🔒 TIGHT REVERT: →99/off if live cohort
    # ≤50% WR or net-negative on N≥8. 1h_max 99 = off; ship −0.05 / stretch_min 0.31.
    long_w6_reenable_1h_max: float = 99.0
    long_w6_reenable_stretch_min: float = 0.31
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
    # Jun 28 — C1 SHORT multiplier breadth-SCOPE (de-mux outside the window). The C1 capitulation-chase
    # 2× only EARNS its multiplier in the 70–85 bear-breadth band: cross-pool (13-batch May26-Jun13 +
    # all-history) the 70–85 band = 73–76% WR / +avg% / +$, while BOTH tails (<70 and ≥85) are 50–60%
    # WR / −avg% and the 2× merely amplifies their fat-tail DOA losers. So when a C1 SHORT cell rule
    # would size >1×, KEEP the multiplier only if lo ≤ entry_bear_pct < hi, else DE-MUX to 1× (sizing
    # change only — entry is NOT blocked; a 50%-WR cohort must not be blocked, only de-amplified). 06-28:
    # AAVE (bear 87) −$242→−$121 and HYPE (bear 62) −$186→−$93 = +$214 batch. enabled=False → no de-mux.
    # 2026-06-29: C1 SHORT 2× REVERTED to 1× (see pattern_cell_rules C1 inv_mult 2.0→1.0) — pooled
    # in-sample+forward C1 SHORT is 1W/4L / −$661@2× (−$330@1×), losing at EVERY quality score (qs=2
    # holds the only winner; qs≥3 = 0W/2L), so the 2× was amplifying a now-losing cell on contaminated
    # evidence. With C1 flat at 1× the breadth de-mux is moot → DISABLED. Re-enable only if C1 re-earns
    # a multiplier (N≥30, WR≥70%, +avg%) cross-batch.
    c1_short_demux_breadth_enabled: bool = False
    c1_short_demux_breadth_lo: float = 70.0   # keep the C1 2× only when entry_bear_pct ≥ this …
    c1_short_demux_breadth_hi: float = 85.0   # … AND entry_bear_pct < this; outside → 1×
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
    # Jul 2, 2026: added "working_capital" mode — the capital-scaling PRIMARY de-risk knob
    # (CLAUDE_CURRENT_STATE capital-scaling strategy, operator-ratified v3 schedule). In this mode
    # tradeable = min(available, working_capital_target) and the reserve auto-grows with balance,
    # clamping the max correlated-cluster loss to a fixed $ regardless of account size. Withdrawals
    # pull from the reserve pool. target=0 or mode != working_capital = inert (current behavior).
    reserve_mode: str = "percentage"  # "fixed" | "percentage" | "working_capital" | "schedule"
    reserve_fixed: float = 500.0  # USD
    reserve_percentage: float = 20.0  # % of total balance
    working_capital_target: float = 0.0  # USD tradeable cap for working_capital mode (0 = off)
    # Jul 2, 2026 (operator-directed): FULLY AUTOMATIC version of working_capital — mode="schedule"
    # walks this balance→tradeable table by itself (no manual milestone flips). Format mirrors
    # leverage_balance_schedule: "balance:tradeable_target, ..." ascending; active target = highest
    # tier ≤ free balance; below the first tier → no reserve (full balance tradeable). The v3
    # operating table (CURRENT_STATE capital-scaling) expressed as tiers. Empty = off.
    reserve_schedule: str = ""  # e.g. "10000:8000, 25000:17500, ..., 500000:100000"
    # Aug 21, 2026 (operator) — FEE RESERVE FLOOR: USDT that sizing never deploys, on top of whatever
    # the reserve mode/schedule says. Root cause on record: the reserve_schedule has no tier below
    # $10k, so a sub-$10k book goes 100% into margin and the runway-triggered BNB auto-swap starves
    # ("[BNB_SWAP] Cannot swap: insufficient USDT (available=-0.00)" at 17:36:49 Aug-21, the second
    # the 4th position opened). $75 ≈ 1.5 days of burn at $2.17/h, covers the $50-minimum swap.
    # Paper: cosmetic (sim charges fees in USDT at 0 BNB). Live: without it, fees lose the BNB
    # discount once BNB hits 0 and the bot cannot refuel while the book is full. 0 = disabled.
    fee_reserve_usd: float = 75.0
    
    # Cooldown after trade close (prevents immediate re-entry on same pair, win or loss)
    # CLAUDE.md May 26: cross-batch evidence on 919-trade pool shows 84 same-pair re-entries
    # within 5min after a WINNING trade had 61.9% WR but -$731 net (2.71:1 R:R loss asymmetry).
    # Fast-exit/trailing/BE wins lock tiny profit on fading momentum → re-entry catches the fade.
    # Applies to ANY close (was previously loss-only).
    cooldown_after_loss_minutes: int = 5  # Minutes to wait before re-entering same pair (any close)
    
    # Position limits
    max_open_positions: int = 100  # Max simultaneous open positions
    # Aug-20 2026 (operator-directed; gate 55, DECISION_LOG 2026-08-20 (2)): JSON 5 → 4.
    # Concurrency replay over all 209 kept trades / 63 days: max observed concurrency = 3;
    # ZERO trades would have been blocked at 4 (85-87% of in-market time at ONE position).
    # Equal-split slot = working/max_open → 4 gives every trade 1.25× sizing for free
    # (CF: master +$1,902 · current +$128). Per-trade concentration +25% (quiet-SL blow
    # = −10% of working vs −8%); max gross UNCHANGED (4×1.25 = 5 slots-worth). ⚠ KNOWN
    # BLIND SPOT: a max-open block is neither counted nor logged — an unexplained no-trade
    # stretch in a hot tape → check open-count FIRST. Revert to 5 if any bind observed
    # or suspected during the gate-51 flow ramp (concurrency profile recomputed each review).
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
    # 🔒 SPIKE LOW-VOL CAP RAISE (Aug-3, operator-directed acknowledged OVERRIDE of the
    # pre-committed gate's fresh-N term — evidence: scoped raise direction-consistent
    # both batches (B1 <$10M fades Δ+$46 · B2 Δ+$492, +$21 ex-Sunday), loser side now
    # lock-capped; $10M = frozen concentration-slice boundary (every large-$ spike loser
    # on record is >=$10M). ALL spike species (chase: 0/6 historic <$10M, router-dark,
    # accepted). Cap reason stamps LIQ2. 🔒 REVERT to 0 (=inherit 0.1%) if over the next
    # 15 full-size <$10M spike fires: cohort Σ pnl% < 0 OR any stop/lock fill slips
    # >15bps OR DOA (peak<+0.10) > 20%.
    # ✗ GATE FIRED 2026-08-05 (window complete 16/15, ALL THREE legs failed: Σ −1.09% <0 ·
    # DOA 5/16=31% >20% · EVAA slip-through −3.07% vs −0.70 stop ≫15bps) → mechanical revert to
    # 0.1% executed. ⚠ OPERATOR OVERRIDE same day: back to 0.2%, informed of the three-leg failure.
    # 📋 WATCHLIST (operator directive Aug-5 PM: NO automatic gate — review item only): quant
    # FLAGS for operator decision on any doubled fire ≤ −1.5% or rolling-15 doubled-fire Σ pnl% < 0;
    # DOA% reported each batch as diagnostic (not a trigger). Measured LIQ2 marginal at review:
    # −$62 over 17 fires (boost gave +$130 across winners, −$127 on EVAA alone).
    # Aug-17 2026: 0.2 → 0.3 (operator ship). Evidence: 46/48 kept fades are liq-capped at
    # median 31% of desired size; capped cohort 83% WR +$1,475 across all eras — linear paper
    # CF at 0.3 = +$340 (avg size 1.35×). ⚠ ON RECORD: the increment's P&L is UNVERIFIABLE
    # UNTIL LIVE — the cap exists for market impact, which paper fills don't model (a 0.3%
    # order on a $2-4M/day pair is minutes of the pair's volume in one taker order during a
    # spike reversal; fade edge ~+0.3%/trade vs realistic thin-book slippage 0.2-0.5%).
    # 0.4 DECLINED. 🔒 LIVE-CUTOVER REVIEW (mandatory, in the live punch list): revert to 0.2
    # at live start; walk back up only with measured per-fill slippage data (exit_slippage_pct
    # column) proving each step. Paper review: fade cohort at each batch review — sizing-up a
    # loser streak reverts to 0.2 per the standard cohort tally.
    # Aug-18 2026: threshold $10M → $1T (operator: "apply the 0.3% for all spike trades") —
    # config-only mechanism: every spike pair now falls below the threshold, so ALL spike
    # species get the raised cap (LIQ2 stamp everywhere → they all enter the Liquidity Sizing
    # table = gate-50's revert instrument). Increment measured on the $10-36M fade band the
    # old boundary excluded: master 5 fades 80% WR net extra +$144 (+$72-101 haircut) · current
    # batch 1 fade +$38 — at 0.3% all affected trades reach 100% of desired size. Bounces
    # unaffected (all <$10M already). Momentum/flips untouched (global 0.1% keeps the book rail;
    # binds ~never: 3 trims ever at ≥92% of desired). Revert = restore 10_000_000.0.
    spike_lowvol_liq_cap_pct: float = 0.3         # spikes on thin pairs: % of 24h vol (0 = off → global pct)
    spike_lowvol_threshold_usd: float = 10_000_000.0  # "thin" = 24h vol below this (JSON carries $1T = all-spikes since Aug-18)
    # ② Gross-notional cap: Σ(open notional) ≤ balance × max_gross_leverage.
    #    Portfolio liquidation/correlation guard (a -X% correlated dump costs
    #    X% × gross_leverage of the account). 0 = disabled.
    max_gross_leverage: float = 0.0  # e.g. 25.0; 0 = off
    # ③ Redeploy leftover: when ① throttles a slot below its equal-split slice,
    #    allow opening MORE positions (overrule max_open_positions up to the hard
    #    ceiling) to deploy the freed capital — gated by ② + tradeable margin.
    redeploy_leftover_enabled: bool = False
    max_open_positions_hard: int = 10  # absolute ceiling when redeploying
    # ④ Balance→leverage schedule (Jun 26): a balance-aware leverage CEILING — de-lever as the
    #    account grows so a fat-tail/correlated event can't end the (now larger) account. Format:
    #    "bal0:lev0, bal1:lev1, ..." ascending balance tiers; the cap = highest tier ≤ current equity.
    #    e.g. "0:20, 10000:15, 25000:10, 100000:5" → ≥$0 cap 20×, ≥$10k 15×, ≥$25k 10×, ≥$100k 5×.
    #    Empty = OFF (no cap, current behavior). Clamps the FINAL leverage (after the cell lev-mult),
    #    sits alongside the gross cap as the second systemic risk knob. Genuinely inert until balance
    #    crosses the first non-base tier. Tier VALUES should come from the tail-stressed blended-pool
    #    Kelly (growth-optimal ≈ fractional-Kelly < 20×), not round guesses → see GO-LIVE TODO.
    # Aug-20 2026 (operator-directed, drawdown-derived — DECISION_LOG 2026-08-20): JSON simplified
    # "0:20,25000:15,75000:10,250000:5" → "0:20,25000:15" (75k/250k tiers DELETED). Derivation from
    # 54-day measured tail (worst day −4.13 slot-units · stress w/ gate-53 widths −5.12 · design
    # −7.69 = 1.5× safety): above $25k the tradeable schedule ALONE keeps effective exposure
    # (trade% × lev) inside a 15%-worst-day bound — 10.5×@25k → 8.2×@50k → 6.0×@100k → 4.2×@250k
    # → 3.0×@500k, monotonic. The deleted tiers double-braked $75k+ to 1.4-4× effective. ON RECORD:
    # sub-$25k 20× runs ABOVE the 15% bound (design-tail day ≈ −31%; lived −16% Jul-10) — accepted
    # legacy aggression at small balances. 🔴 RE-DERIVE ON LIVE FILLS AT CUTOVER (punch list).
    leverage_balance_schedule: str = ""  # "0:20, 10000:15, 25000:10, 100000:5"; empty = off


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
    # Jul 14 (operator, $50): HARD dollar floor on the BNB fee reserve. The runway system is
    # trailing-burn-based and collapses to a few dollars during idle stretches (observed burn
    # swing 0.32→3.16 $/hr = 10x; balance sat at $11 with "36h runway"), so a trading burst
    # could drain it between 6h checks. Floors: top-up target = max(burn×runway, THIS);
    # emergency threshold = max(burn12×12, THIS×0.5); auto-sell never drains below THIS.
    bnb_min_balance_usd: float = 50.0
    paper_bnb_initial_usd: float = 100.0  # Aug 10: 200->100 (operator; USDT seed 2800->2900, total $3000 unchanged)
    # BNB AUTO-SELL (Jun 22) — symmetric rebalance. The buy path tops BNB UP to a 24h
    # runway, but never claws back: when activity slows the 24h burn rate decays, runway
    # inflates, and the over-funded reserve locks USDT out of trading (tradable balance =
    # ... − bnb_swaps). Auto-sell drains the excess back to tradable USDT when the reserve
    # is genuinely over-funded. HYSTERESIS (anti-churn, matters LIVE where each swap costs
    # ~0.15-0.2% round-trip — invisible in paper): sell ceiling 48h is 2× the 24h buy floor,
    # and we sell DOWN TO 36h (not the floor) so a small burn uptick won't immediately re-buy.
    # The trigger uses max(24h, 12h) burn so a RECENT pickup keeps more BNB (don't sell into
    # rising fees). Runs on the same 6h scheduled check (≤1 action/6h) and is mathematically
    # unable to fire in the 6h right after a buy (runway can't double that fast).
    # NOTE: code default is False (a fresh deploy without trading_config.json stays safe/off),
    # but trading_config.json sets it true — the operator opted in. The json value wins at runtime.
    bnb_auto_sell_enabled: bool = False
    bnb_sell_runway_hours: float = 48.0   # sell when BNB runway > this (ceiling). 0 = off.
    bnb_sell_target_hours: float = 36.0   # sell DOWN TO this runway (buffer above the 24h buy floor).
    bnb_min_sell_usd: float = 50.0        # min auto-sell size (avoid tiny churn swaps).
    
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
    # Crypto-only universe (Jul 14, 2026): allowlist `underlyingType == "COIN"` —
    # excludes Binance's 132+ EQUITY/TradFi perps (tokenized stocks: MU/INTC/SPY/NVDA...,
    # commodities: NATGAS/COPPER/XPT, indexes, leveraged ETFs) in ONE condition.
    # Motivation: MU short 07-14 (−$77) was our FIRST-EVER equity-perp trade (zero in the
    # 409-trade pool) — admitted by the 90-day step-down; signals are calibrated to crypto
    # microstructure and equity perps trade synthetically while the underlying is CLOSED.
    # Subsumes the hand-built XAU/XAG/ALL blacklist entries + the XAUT question... (XAUT
    # itself is COIN-type Tether Gold — stays blacklist territory if wanted). Future-proof:
    # new stock listings carry the tag on day one. Fail-open on missing metadata.
    coin_underlying_only: bool = True

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
