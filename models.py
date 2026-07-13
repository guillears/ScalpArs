"""
SCALPARS Trading Platform - Database Models
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Enum as SQLEnum, Text
from sqlalchemy.sql import func
from database import Base
from datetime import datetime
from enum import Enum


class OrderDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class CloseReason(str, Enum):
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    MANUAL = "MANUAL"
    SIGNAL_CHANGE = "SIGNAL_CHANGE"


class Order(Base):
    """Trading orders table"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    binance_order_id = Column(String(50), nullable=True)  # Null for paper trades
    
    # Trade info
    pair = Column(String(20), nullable=False)  # e.g., "BTCUSDT"
    direction = Column(String(10), nullable=False)  # LONG or SHORT
    status = Column(String(15), nullable=False, default="OPEN")
    
    # Position details
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    
    # Size and leverage
    investment = Column(Float, nullable=False)  # Margin/investment amount
    leverage = Column(Float, nullable=False)
    notional_value = Column(Float, nullable=False)  # investment * leverage
    quantity = Column(Float, nullable=False)  # Amount of asset
    
    # Confidence
    confidence = Column(String(15), nullable=False)  # LOW, MEDIUM, HIGH, EXTREME, STRONG_BUY, VERY_STRONG
    
    # Entry gap: abs((ema5 - ema20) / price * 100) at time of entry
    entry_gap = Column(Float, nullable=True)
    # Entry EMA5-EMA8 gap: abs((ema5 - ema8) / ema8 * 100) at time of entry
    entry_ema_gap_5_8 = Column(Float, nullable=True)
    # Entry EMA8-EMA13 gap: abs((ema8 - ema13) / ema13 * 100) at time of entry (May 27).
    # Complement to EMA5-EMA8 — measures the gap between two smoothed momentum lines vs
    # the bot's trend reference (EMA13 is what EMA13_CROSS_EXIT fires against). Less
    # noisy than EMA5-EMA13 because EMA8 is already smoothed. Tracked alongside
    # entry_dist_from_ema13_pct (price vs EMA13, different signal).
    entry_ema_gap_8_13 = Column(Float, nullable=True)
    # RSI(12) value at time of entry
    entry_rsi = Column(Float, nullable=True)
    # RSI(12) value 2 candles prior (~10min ago, matches RSI Momentum Filter comparison
    # which uses rsi_prev2). Used by Pair RSI Direction analytics. May 15.
    entry_rsi_prev = Column(Float, nullable=True)
    # ADX(14) value at time of entry
    entry_adx = Column(Float, nullable=True)
    # ADX(14) value one candle prior (for ADX direction analysis)
    entry_adx_prev = Column(Float, nullable=True)
    # EMA5 stretch: abs(price - ema5) / price * 100 at time of entry
    entry_ema5_stretch = Column(Float, nullable=True)
    # Signed price vs EMA5 at entry: (price - ema5) / ema5 * 100 (positive = above, negative = below)
    entry_price_vs_ema5_pct = Column(Float, nullable=True)
    # Macro trend regime at time of entry (BULLISH, BEARISH, NEUTRAL)
    entry_macro_trend = Column(String(10), nullable=True)
    # EMA20 slope % of the pair at entry: ((ema20 - ema20_prev3) / ema20_prev3) * 100
    entry_ema20_slope = Column(Float, nullable=True)
    # EMA20 slope % of BTC at entry
    entry_btc_ema20_slope = Column(Float, nullable=True)
    # BTC ADX(14) value at entry
    entry_btc_adx = Column(Float, nullable=True)
    # BTC ADX(14) value one candle prior (for BTC ADX direction analysis)
    entry_btc_adx_prev = Column(Float, nullable=True)
    # BTC RSI(14) value at entry
    entry_btc_rsi = Column(Float, nullable=True)
    # BTC RSI(14) one candle prior (for BTC RSI direction analysis)
    entry_btc_rsi_prev = Column(Float, nullable=True)
    # BTC RSI 6 candles prior (~30min) — sustained-momentum dimension. May 15.
    entry_btc_rsi_prev6 = Column(Float, nullable=True)
    # May 15 PM — BTC Volatility Regime + BTC 1h RSI Direction (observation-only).
    # See CLAUDE.md May 15 entry for hypothesis and promotion gates.
    # BTC ATR(14) / BTC price × 100 at entry. Measures swing magnitude regardless of direction.
    # Distinct from BTC ADX (which measures trend strength). High ATR + low ADX = violent chop.
    entry_btc_atr_pct = Column(Float, nullable=True)
    # BTC 1h RSI(14) at entry — multi-hour momentum slice (vs 5m and 30m already captured).
    entry_btc_rsi_1h = Column(Float, nullable=True)
    # BTC 1h RSI(14) 1 candle prior (= 1h ago). With entry_btc_rsi_1h enables 1h direction.
    entry_btc_rsi_1h_prev = Column(Float, nullable=True)
    # May 19, 2026 — Pattern C Tracker (observation-only).
    # 4 candidate Pattern C precursor signatures captured at entry. Locked
    # promotion gates at N≥30 per pattern. See CLAUDE.md May 19 entry.
    # C1=capitulation chase, C2=macro counter-trend, C3=stretch exhaustion, C4=low-vol chop.
    entry_pattern_c1_match = Column(Boolean, nullable=True)
    entry_pattern_c2_match = Column(Boolean, nullable=True)
    entry_pattern_c3_match = Column(Boolean, nullable=True)
    entry_pattern_c4_match = Column(Boolean, nullable=True)
    # May 19 — extension: C5=slow climber death (weak-trend slow bleed), C6=macro over-extended same direction.
    entry_pattern_c5_match = Column(Boolean, nullable=True)
    entry_pattern_c6_match = Column(Boolean, nullable=True)
    # May 20 — C7=Pair Countertrend Bounce (dead-cat LONG / failed-breakdown SHORT)
    entry_pattern_c7_match = Column(Boolean, nullable=True)
    # May 20 (later) — C8=Oversold/Overbought Chop (range extreme + sharp ADXΔ + no pair trend + low BTC vol)
    entry_pattern_c8_match = Column(Boolean, nullable=True)
    # May 20 (latest) — C9=Low-vol Countertrend Chop (C4 base + mild countertrend pair_gap)
    entry_pattern_c9_match = Column(Boolean, nullable=True)
    entry_pattern_c_any_match = Column(Boolean, nullable=True)  # OR of c1-c9 (post-May-20-latest; c1-c8 May-20-late; c1-c7 May-20; c1-c6 May-19-late; c1-c4 pre-May-19)
    # May 21 — Pattern W computed at entry (lifted from main.py post-hoc helper to support live multiplier ship).
    # W1=HighConv trend, W2=Macro tailwind, W3=Energetic vol breakout, W4=Pullback aligned, W5=Confluence.
    entry_pattern_w1_match = Column(Boolean, nullable=True)
    entry_pattern_w2_match = Column(Boolean, nullable=True)
    entry_pattern_w3_match = Column(Boolean, nullable=True)
    entry_pattern_w4_match = Column(Boolean, nullable=True)
    entry_pattern_w5_match = Column(Boolean, nullable=True)
    # May 21 (late) — W6 added: "Healthy BTC Tailwind" LONG / "Mature BTC Bear" SHORT
    # Cross-batch validated 100% WR on N=14 LONG / N=25 SHORT (post-May-15 pool).
    # Catches unmatched-winner cohort missed by W1-W5.
    entry_pattern_w6_match = Column(Boolean, nullable=True)
    entry_pattern_w_any_match = Column(Boolean, nullable=True)  # OR of w1-w6 (post-May-21-late; w1-w5 May-21 morning)
    # May 21 — Pattern Cell Ship rules: per-trade attribution of which pattern rule fired.
    # pattern_cell_source = comma-separated source labels (e.g., "C4", "W1+W2", "C4+C8")
    # pattern_fixed_tp_pct = override TP from C-side rule (fires via PATTERN_FIXED_TP close_reason)
    # pattern_fixed_sl_pct = override SL from C-side rule (fires via PATTERN_FIXED_SL close_reason)
    pattern_cell_source = Column(String(60), nullable=True)
    pattern_fixed_tp_pct = Column(Float, nullable=True)
    pattern_fixed_sl_pct = Column(Float, nullable=True)
    # Volume ratios at entry (for volume filter analytics)
    entry_global_volume_ratio = Column(Float, nullable=True)
    entry_pair_volume_ratio = Column(Float, nullable=True)
    # Market breadth at entry (% of scanned pairs in Bull/Bear regime)
    entry_bull_pct = Column(Float, nullable=True)
    entry_bear_pct = Column(Float, nullable=True)
    # Price range position at entry: (price - low_20) / (high_20 - low_20) * 100 (0=bottom, 100=top of 20-candle range)
    entry_range_position = Column(Float, nullable=True)
    # ADX delta at entry: adx - adx_prev (positive = rising, negative = falling, magnitude = strength)
    entry_adx_delta = Column(Float, nullable=True)
    # Signal quality score at entry: 0-6 (higher = more favorable conditions aligned)
    entry_quality_score = Column(Float, nullable=True)
    # BTC regime classification at entry (frozen) and exit (current)
    entry_btc_regime = Column(String(20), nullable=True)
    exit_btc_regime = Column(String(20), nullable=True)
    # When the current BTC regime began (frozen at entry). Used at report time
    # to compute btc_regime_age_seconds = opened_at - entry_btc_regime_started_at.
    # Diagnostic for "did this trade enter on a fresh / volatile regime that flips
    # quickly, or an aged / stable one?". See CLAUDE.md May 5 regime stability entry.
    entry_btc_regime_started_at = Column(DateTime, nullable=True)
    # BTC EMA20 vs EMA50 gap at entry (% of EMA50). Captured at entry to support
    # post-hoc analysis of "did this trade pass the BTC Trend Filter? by how much?"
    # at the next-batch rollback validation checkpoint. See CLAUDE.md May 5 entry
    # on filter-rollback candidates and BTC Trend Filter.
    entry_btc_trend_gap_pct = Column(Float, nullable=True)

    # Exploration Analytics (Phase 1c+, observation-only) — added Apr 28
    # Captured at signal time, NOT used in any entry filter logic. Purpose:
    # bucket-analysis at next 100-trade checkpoint to identify which dimensions
    # discriminate winners from losers. Promote to filter only after cross-sample
    # confirmation per anti-overfit discipline.
    entry_pos_di = Column(Float, nullable=True)              # +DI (positive directional indicator) — directional component of ADX
    entry_neg_di = Column(Float, nullable=True)              # -DI (negative directional indicator) — directional component of ADX
    entry_atr_pct = Column(Float, nullable=True)             # ATR(14) as % of entry price; volatility regime per pair
    entry_ema50_slope = Column(Float, nullable=True)         # 5m EMA50 slope vs prev12 candles (~4h higher-TF context proxy)
    entry_funding_rate = Column(Float, nullable=True)        # Binance Futures funding rate at entry (positioning context)
    # Pair EMA20 vs EMA50 gap at entry (% of EMA50). Positive = pair uptrend, negative = pair downtrend.
    # Observation-only (May 5) — same methodology as Apr 28 Exploration Analytics.
    # Captures multi-hour pair trend context (EMA50 spans ~4 hours on 5m chart).
    entry_pair_ema20_ema50_gap_pct = Column(Float, nullable=True)

    # May 13 PM: Entry Extension / Late Entry Risk dimension.
    # Signed distance from EMA13 at entry: (entry_price - ema13) / ema13 × 100.
    # LONG positive = price ABOVE EMA13 = potentially chasing/late.
    # SHORT negative = price BELOW EMA13 = potentially shorting after capitulation.
    # Tests whether bad trades are late entries (timing) vs wrong signals (quality).
    # Pre-deploy trades stay NULL — no retroactive backfill.
    entry_dist_from_ema13_pct = Column(Float, nullable=True)
    # May 14: BTC Market Extension / BTC Late Regime Risk dimension.
    # Signed distance of BTC price from BTC EMA13 at the trade's entry instant:
    # (btc_price - btc_ema13) / btc_ema13 × 100. Macro counterpart of pair extension.
    # Positive = BTC above its EMA13 (LONGs are entering a stretched market);
    # negative = BTC below its EMA13 (SHORTs are entering after market capitulation).
    # Combined with pair extension (double-stretch detection), tests whether
    # losses cluster when BOTH levels are extended.
    entry_btc_dist_from_ema13_pct = Column(Float, nullable=True)
    # May 14 — BTC 1h EMA20 slope at trade entry.
    # Captures multi-hour BTC trend direction (12× longer than 5m slope).
    # Slope = (ema20_1h - ema20_1h_prev3) / ema20_1h_prev3 × 100.
    # Tests whether the "5m bearish blip during 1h uptrend" failure mode is the
    # SHORT loss driver. Discriminator candidate after May 4 found that every
    # existing 5m-timeframe dimension showed identical winner/loser signatures.
    entry_btc_1h_slope = Column(Float, nullable=True)
    # May 14 — Phantom BE @ 0.20/0.05 (observation-only counterfactual tracker).
    # Mirrors existing phantom_be_l1/l2 mechanism but at aggressive trigger/floor.
    # Triggered: peak first crossed +0.20%. Would-exit: P&L retraced to ≤+0.05%.
    phantom_be_aggr_triggered_at = Column(DateTime, nullable=True)
    phantom_be_aggr_would_exit_pnl = Column(Float, nullable=True)

    # May 10: absolute pair 24h volume in USD at entry time. For size-bucket
    # analysis — find structural threshold below which pairs underperform,
    # rather than blacklisting one-by-one. NULL on pre-deploy trades.
    entry_pair_volume_24h_usd = Column(Float, nullable=True)

    # Jun 12: pair's volume rank (1 = highest 24h vol) in the eligible top-N list
    # at entry, captured BEFORE blacklist removal. Read gate for the 50->75
    # universe expansion: rank>50 cohort vs rank<=50 at N>=20. NULL pre-deploy.
    entry_pair_rank = Column(Integer, nullable=True)
    # Jul 13: listing age (days since Binance onboardDate) at entry — read gate for
    # the new-listing filter 180->90-day step-down (edge-by-age; NULL pre-deploy or
    # missing metadata). Rides the orders CSV via column introspection.
    entry_pair_age_days = Column(Float, nullable=True)

    # Liquidity-aware sizing observability (Jun 2, 2026 — see CLAUDE.md).
    # entry_desired_notional       = notional the order WOULD have opened at pre-cap (investment×leverage).
    # entry_liquidity_cap_notional = the ① per-pair liquidity cap value (_liq_cap); NULL if ① not configured.
    # liquidity_capped             = True when ①/② throttled this order's notional below desired.
    # Final notional is the existing notional_value column; throttle% = 1 − notional_value/entry_desired_notional.
    entry_desired_notional = Column(Float, nullable=True)
    entry_liquidity_cap_notional = Column(Float, nullable=True)
    liquidity_capped = Column(Boolean, default=False)

    # Pair / BTC EMA20 vs EMA50 gap at EXIT (May 6). Captures multi-hour trend context
    # at close time — diagnostic for REGIME_CHANGE / FL_REGIME_CHANGE: did BTC's actual
    # 4h trend flip, or just the 5m regime classifier? Historical trades have NULL.
    exit_pair_ema20_ema50_gap_pct = Column(Float, nullable=True)
    exit_btc_trend_gap_pct = Column(Float, nullable=True)

    # Phase 1 shadow tracking (May 6) — counterfactual exit at first price-vs-EMA cross
    # against trade direction. Observation-only: captures the moment + counterfactual
    # close P&L if we had exited at that point. Two confirmation modes per EMA:
    #   Naive: first tick where price > EMA13 (SHORT) or price < EMA13 (LONG)
    #   Confirmed: first such cross where the FOLLOWING candle's CLOSE also stays
    #              on the wrong side (filters single-candle wicks)
    # Counterfactual P&L is computed at the moment of cross using current price
    # and accounts for fees (taker exit). Once recorded, never overwritten.
    first_cross_ema13_at = Column(DateTime, nullable=True)
    first_cross_ema13_pnl_pct = Column(Float, nullable=True)
    confirmed_cross_ema13_at = Column(DateTime, nullable=True)
    confirmed_cross_ema13_pnl_pct = Column(Float, nullable=True)
    first_cross_ema20_at = Column(DateTime, nullable=True)
    first_cross_ema20_pnl_pct = Column(Float, nullable=True)
    confirmed_cross_ema20_at = Column(DateTime, nullable=True)
    confirmed_cross_ema20_pnl_pct = Column(Float, nullable=True)

    # Fees
    entry_fee = Column(Float, nullable=False, default=0.0)
    exit_fee = Column(Float, nullable=True, default=0.0)
    total_fee = Column(Float, nullable=True, default=0.0)
    
    # P&L tracking
    pnl = Column(Float, nullable=True, default=0.0)  # Realized P&L
    pnl_percentage = Column(Float, nullable=True, default=0.0)
    peak_pnl = Column(Float, nullable=False, default=0.0)  # For trailing stop
    trough_pnl = Column(Float, nullable=False, default=0.0)  # Lowest P&L reached during trade
    high_price_since_entry = Column(Float, nullable=True)  # For LONG
    low_price_since_entry = Column(Float, nullable=True)  # For SHORT
    peak_ema5_gap = Column(Float, nullable=True, default=0.0)  # Peak price-to-EMA5 distance for momentum exit
    peak_ema5_dist_pct = Column(Float, nullable=True)  # Price-to-EMA5 distance at moment of peak P&L
    peak_ema5_slope_pct = Column(Float, nullable=True)  # EMA5 slope at moment of peak P&L
    peak_reached_at = Column(DateTime, nullable=True)
    trough_reached_at = Column(DateTime, nullable=True)
    trough_ema5_dist_pct = Column(Float, nullable=True)
    ema5_went_negative = Column(String(20), nullable=True)  # NEVER, RECOVERED, ENDED_NEG
    
    # Dynamic TP tracking
    current_tp_level = Column(Integer, nullable=False, default=1)  # Which TP level (1, 2, 3, ...)
    dynamic_tp_target = Column(Float, nullable=True)  # Current TP target (% of notional)
    
    # NO_EXPANSION timer reset: last time signal was re-verified (None = use opened_at)
    no_expansion_last_check = Column(DateTime, nullable=True)

    # Whether the entry signal was still active when the order was closed
    signal_active_at_close = Column(Boolean, nullable=True)

    # Slippage tracking: difference between decision price (WebSocket) and actual Binance fill
    exit_slippage_pct = Column(Float, nullable=True)  # positive = filled worse than expected
    # Entry-fill slippage (Jun 2): signed % the entry filled WORSE than the decision price.
    # positive = paid more (LONG) / sold cheaper (SHORT). ~0 in paper (sim fills at signal price);
    # only meaningful live. Used to give ① the per-pair liquidity cap a real slippage verdict
    # (slice by liquidity_capped: do capped fills come in tighter?).
    entry_slippage_pct = Column(Float, nullable=True)

    # Reconciler race guard (Apr 16 — SUIUSDT incident).
    # The bot sets closing_in_progress=True and close_initiated_at=NOW() BEFORE
    # sending the close order to Binance. The monitor reconciler
    # (main._reconcile_open_orders) must skip rows with a fresh flag so it can
    # tell a bot-initiated close in flight apart from a truly external close.
    # A stale flag (older than CLOSE_INTENT_STALE_SECONDS in main.py) is ignored
    # so crashed close paths can still be reconciled.
    closing_in_progress = Column(Boolean, nullable=False, default=False)
    close_initiated_at = Column(DateTime, nullable=True)

    # Broker-side protective stops feature REMOVED Apr 17 after 4 failed
    # hotfix attempts (Binance -4120 / -2015 errors on this account).  These
    # two columns are kept (not dropped) because removing columns requires a
    # schema migration; leaving them NULL is harmless.  All existing rows
    # have NULL values for these since no protective stop order ever placed
    # successfully.  If the feature is ever re-attempted, these columns can
    # be reused without schema change.  See CLAUDE.md "Broker-side Protective
    # Stops removal" for the forensic trail.
    protective_sl_order_id = Column(String(50), nullable=True)
    protective_tp_order_id = Column(String(50), nullable=True)

    # Premium Multiplier (May 4, 2026 — Phase 3 Position Multiplier per CLAUDE.md May 3 design).
    # cell_multiplier = the INVESTMENT multiplier ACTUALLY applied to this trade after hard-cap clamping.
    # cell_lev_multiplier = the LEVERAGE multiplier ACTUALLY applied (May 21 — added when "both" mode shipped).
    #   In "investment" mode: only cell_multiplier affects sizing (cell_lev_multiplier stored as 1.0).
    #   In "leverage" mode:   only cell_lev_multiplier affects sizing (cell_multiplier stored as 1.0).
    #   In "both" mode:       BOTH apply; effective notional = investment × cell_multiplier × leverage × cell_lev_multiplier.
    # cell_multiplier_source = which rule fired, format "PAIR_<RSI>_<ADX>" or "BTC_<RSI>_<ADX>",
    #   or NULL if no rule matched (default 1.0×).  When both pair- and BTC-level rules match,
    #   the HIGHER multiplier wins (per design); source identifies which rule was the winner.
    # cell_multiplier_capped = True if available balance forced sub-target investment
    #   (multiplier wanted X but only Y available; trade still proceeds at Y).
    cell_multiplier = Column(Float, nullable=False, default=1.0)
    cell_lev_multiplier = Column(Float, nullable=False, default=1.0)
    cell_multiplier_source = Column(String(40), nullable=True)
    cell_multiplier_capped = Column(Boolean, nullable=False, default=False)

    # May 8: tracks EMA13 cross strict-mode (ema13_cross_requires_stack_flip).
    # Captures pnl_pct at the FIRST moment strict mode held an EMA13 cross
    # exit (price wick filtered because EMA5/EMA8 stack hadn't flipped).
    # NULL = strict mode never held this trade. Used by EMA13 Strict-Mode
    # Performance report to compute counterfactual delta vs final close.
    ema13_strict_held_pnl_pct = Column(Float, nullable=True)

    # May 9: Trailing pullback confirmation tracking. Captures the
    # would-have-fired-immediately P&L at the first moment the trailing
    # threshold was crossed. After confirmation period elapses, exit fires
    # and we can compare actual close vs counterfactual to see if the
    # confirmation timer helped (positive Δ = save) or hurt (negative Δ).
    trailing_first_pullback_pnl_pct = Column(Float, nullable=True)
    trailing_pullback_resets = Column(Integer, nullable=True, default=0)
    trailing_confirmed_at = Column(DateTime, nullable=True)

    # Exit quality: Price vs EMA5 at exit
    exit_price_vs_ema5_pct = Column(Float, nullable=True)
    exit_ema5_slope_pct = Column(Float, nullable=True)
    exit_ema5_crossed = Column(Boolean, nullable=True)

    # Post-exit regret tracking: hypothetical P&L if the trade had stayed open for N minutes after close
    # May 8: running state persisted continuously during tracking so a bot restart
    # mid-window doesn't reset the peak/trough captured so far. Recovery reads
    # these and resumes tracking from saved state instead of starting from current
    # price + now timestamps.
    post_exit_running_high = Column(Float, nullable=True)
    post_exit_running_low = Column(Float, nullable=True)
    post_exit_running_peak_at = Column(DateTime, nullable=True)
    post_exit_running_trough_at = Column(DateTime, nullable=True)
    post_exit_peak_pnl = Column(Float, nullable=True)
    post_exit_trough_pnl = Column(Float, nullable=True)
    post_exit_peak_minutes = Column(Float, nullable=True)
    post_exit_trough_minutes = Column(Float, nullable=True)
    post_exit_signal_lost_minutes = Column(Float, nullable=True)
    post_exit_pnl_at_signal_lost = Column(Float, nullable=True)
    post_exit_final_pnl = Column(Float, nullable=True)
    post_exit_peak_before_signal_lost = Column(Float, nullable=True)
    post_exit_rsi_exit_minutes = Column(Float, nullable=True)
    post_exit_rsi_exit_pnl = Column(Float, nullable=True)
    post_exit_rsi3_exit_minutes = Column(Float, nullable=True)
    post_exit_rsi3_exit_pnl = Column(Float, nullable=True)
    # May 17: post-arm minimum P&L tracking for BE-floor counterfactual analysis.
    # Tracks the minimum P&L observed from the moment peak_pnl first crosses the
    # BE trigger (be_level1_trigger, typically 0.20%) onward, through trade close.
    # Captures BOTH pre-global-peak dips (after BE armed) AND post-peak retraces —
    # which is the right window for "would BE at floor X have fired" analysis.
    # NULL if peak never reached BE trigger (trade never armed BE).
    post_arm_min_pnl_pct = Column(Float, nullable=True)
    post_arm_min_pnl_at = Column(DateTime, nullable=True)
    # May 16: EMA13 cross counterfactual during post-exit window.
    # Records the first moment EMA13 cross-against-direction condition would
    # have fired (strict-mode: requires EMA5/EMA8 stack flip too) after we exited.
    # Used to validate "extended hold" exit experiments (tp_min raise, wider
    # pullback) — answers "would EMA13 cross have caught this before our
    # hypothetical new exit fired?"
    post_exit_ema13_cross_minutes = Column(Float, nullable=True)
    post_exit_ema13_cross_pnl = Column(Float, nullable=True)
    # Jun 7: phantom EMA13 cross — when EMA13 cross exit is DISABLED for this
    # direction, record the pnl% at the FIRST would-have-fired cross (without
    # closing). CF: phantom (would-have-exited) vs actual (held to real exit).
    phantom_ema13_cross_pnl = Column(Float, nullable=True)
    phantom_ema13_cross_at = Column(DateTime, nullable=True)
    # Jun 8: trailing min-profit gate — when the gate suppresses a trailing fire,
    # record the would-have-cut pnl% (CF: this vs the actual exit the trade rode to).
    phantom_trail_suppress_pnl = Column(Float, nullable=True)
    phantom_trail_suppress_at = Column(DateTime, nullable=True)
    # Jun 8: gap-expanding relaxation A/B. True = this entry was admitted by
    # ema_gap_expanding_mode='prev2_only' but would have failed the strict prev1 check
    # (the MARGINAL cohort). False = clean expander. NULL = pre-feature / undefined.
    entry_gap_expand_marginal = Column(Boolean, nullable=True)
    # Jun 14: Flip Entry sleeve. Which entry strategy opened this position:
    # NULL/"MOMENTUM" = the normal momentum bot; "FLIP:<SOURCE>" = a fade-the-block
    # mean-reversion entry (e.g. "FLIP:FAN_RATIO_GATE" — opened OPPOSITE a blocked
    # entry, with its own SL/arm/trail exit emitting FLIP_SL/TRAIL/HORIZON). Used to
    # segregate flip P&L from core momentum stats everywhere in the reports.
    entry_strategy = Column(String(40), nullable=True)
    # May 23: post-exit regime-flip tracker. After trade closes, post-exit
    # watcher continues to read live BTC regime classification each tick.
    # When BTC regime first transitions to OPPOSITE-of-trade-direction
    # (or to NEUTRAL — counts as "no longer supporting our direction"),
    # capture the timestamp and current post-exit running P&L. Answers:
    # "if we had held past the actual exit until regime flipped, what
    # would the P&L have been?" Compare to actual close to decide
    # whether EMA13 cross / SL exit fired too early vs the regime signal.
    # NULL = regime never flipped in the post-exit tracking window
    # (= EMA13 cross / SL was the structural exit; regime would have
    # held into deeper drawdown).
    post_exit_regime_flip_at = Column(DateTime, nullable=True)
    post_exit_regime_flip_pnl_pct = Column(Float, nullable=True)
    post_exit_signal_regained_minutes = Column(Float, nullable=True)
    post_exit_pnl_at_signal_regained = Column(Float, nullable=True)
    post_exit_floor_before_signal_regain = Column(Float, nullable=True)
    # May 12 (LATE PM): Direct time-bucketed P&L snapshots after exit.
    # Answers "what would close % be if we held N min more?" Captured by
    # post-exit monitor when (now - closed_at) first crosses each threshold.
    # NULL = signal/tracking ended before the snapshot threshold was reached
    # (interpret as "this counterfactual is invalid — trade was no longer holdable").
    post_exit_pnl_at_1min = Column(Float, nullable=True)
    post_exit_pnl_at_2min = Column(Float, nullable=True)
    post_exit_pnl_at_5min = Column(Float, nullable=True)
    post_exit_pnl_at_15min = Column(Float, nullable=True)
    post_exit_pnl_at_30min = Column(Float, nullable=True)

    # Phantom BE shadow tracking: what would have happened if BE L1/L2 were active
    phantom_be_l1_triggered_at = Column(DateTime, nullable=True)
    phantom_be_l1_would_exit_pnl = Column(Float, nullable=True)
    phantom_be_l2_triggered_at = Column(DateTime, nullable=True)
    phantom_be_l2_would_exit_pnl = Column(Float, nullable=True)

    # Phantom Regime Change Exit shadow tracking (added May 11 UTC-3):
    # Locked at the FIRST monitor cycle where BTC regime flipped to opposite of trade direction
    # during the hold. NULL = regime never flipped (SAME_REGIME trade).
    # Allows counterfactual evaluation of regime_change_exit_enabled before flipping it on.
    phantom_regime_change_exit_triggered_at = Column(DateTime, nullable=True)
    phantom_regime_change_exit_pnl = Column(Float, nullable=True)

    # Phantom Tick Momentum shadow tracking: alternative exit configs
    # A = same windows (15,30,45), higher delta (0.15%)
    # B = wider windows (30,45,60), same delta (0.12%)
    # C = wider windows (30,45,60), higher delta (0.15%)
    # D = wide windows (30,60,90), same delta (0.12%)
    # E = wide windows (30,60,90), higher delta (0.15%)
    # F = wide windows (30,60,90), variable delta (0.08%,0.12%,0.18%)
    phantom_tick_a_triggered_at = Column(DateTime, nullable=True)
    phantom_tick_a_pnl = Column(Float, nullable=True)
    phantom_tick_b_triggered_at = Column(DateTime, nullable=True)
    phantom_tick_b_pnl = Column(Float, nullable=True)
    phantom_tick_c_triggered_at = Column(DateTime, nullable=True)
    phantom_tick_c_pnl = Column(Float, nullable=True)
    phantom_tick_d_triggered_at = Column(DateTime, nullable=True)
    phantom_tick_d_pnl = Column(Float, nullable=True)
    phantom_tick_e_triggered_at = Column(DateTime, nullable=True)
    phantom_tick_e_pnl = Column(Float, nullable=True)
    phantom_tick_f_triggered_at = Column(DateTime, nullable=True)
    phantom_tick_f_pnl = Column(Float, nullable=True)
    phantom_tick_g_triggered_at = Column(DateTime, nullable=True)
    phantom_tick_g_pnl = Column(Float, nullable=True)

    # ===== LEASH SHADOW START (May 30) — remove this fenced block to delete the feature =====
    # Leash Shadow Tracker (May 30): observation-only virtual trailing leashes run
    # alongside the real exit to measure the true net of a runner-tuned exit on the
    # high-stretch LONG profile (separates XLM-clean-capture from NEAR-trap-mirage,
    # which the coarse post-exit snapshots cannot). Each virtual leash respects the
    # SAME live exits (hard SL, EMA13 cross, signal-lost) and only swaps the trailing
    # width. NULL = leash never armed (peak < activation). Reason: trailing/hard_sl/
    # ema13/signal_lost/window. See CLAUDE.md May 30 entry. NEVER affects live trading.
    #   tight = flat 0.25% pullback (SANITY — should land ~= actual close)
    #   wide  = flat 0.6% pullback whole trade
    #   tierA = tight 0.25 -> wide 0.8, switch @ running-peak >= 1.0%
    #   tierB = tight 0.30 -> wide 1.0, switch @ running-peak >= 1.0%
    # May 31: *_min = minutes from open to the leash fire (vs trade duration = pre/post-close).
    shadow_tight_pnl = Column(Float, nullable=True)
    shadow_tight_reason = Column(String(15), nullable=True)
    shadow_tight_min = Column(Float, nullable=True)
    shadow_wide_pnl = Column(Float, nullable=True)
    shadow_wide_reason = Column(String(15), nullable=True)
    shadow_wide_min = Column(Float, nullable=True)
    shadow_tierA_pnl = Column(Float, nullable=True)
    shadow_tierA_reason = Column(String(15), nullable=True)
    shadow_tierA_min = Column(Float, nullable=True)
    shadow_tierB_pnl = Column(Float, nullable=True)
    shadow_tierB_reason = Column(String(15), nullable=True)
    shadow_tierB_min = Column(Float, nullable=True)
    # Stretch-exit variants (May 30 ext): exit on extension-fade not price pullback.
    #   strpk = exit when live stretch <= 0.5x peak stretch · stren = exit when stretch <= entry stretch
    #   peak_stretch = max favorable %-distance from EMA5 during the trade (moment-evolution signal)
    shadow_strpk_pnl = Column(Float, nullable=True)
    shadow_strpk_reason = Column(String(15), nullable=True)
    shadow_strpk_min = Column(Float, nullable=True)
    # May 31: strpk K-bracket — 0.4 (strpk04) / 0.3 (strpk03) looser stretch-trail variants
    shadow_strpk04_pnl = Column(Float, nullable=True)
    shadow_strpk04_reason = Column(String(15), nullable=True)
    shadow_strpk04_min = Column(Float, nullable=True)
    shadow_strpk03_pnl = Column(Float, nullable=True)
    shadow_strpk03_reason = Column(String(15), nullable=True)
    shadow_strpk03_min = Column(Float, nullable=True)
    shadow_stren_pnl = Column(Float, nullable=True)
    shadow_stren_reason = Column(String(15), nullable=True)
    shadow_stren_min = Column(Float, nullable=True)
    shadow_strpk_signed_pnl = Column(Float, nullable=True)     # Jun 1: hold-until-EMA5-cross variant
    shadow_strpk_signed_reason = Column(String(15), nullable=True)
    shadow_strpk_signed_min = Column(Float, nullable=True)
    shadow_peak_stretch = Column(Float, nullable=True)
    # Jun 16 — sampling-vs-post-exit diagnostic: the shadow's peak stretch SNAPSHOTTED at the
    # instant the LIVE trade closed (vs shadow_peak_stretch which keeps growing post-exit).
    # Compare against runner_peak_stretch (live peak at exit): if ≈ equal, the live strpk was
    # NOT under-sampling (the whole shadow gap is post-exit continuation → Fix B, not Fix A).
    shadow_peak_stretch_at_close = Column(Float, nullable=True)
    # Jun 16 — ATR-floored give-back trail shadows (chandelier), N = 0.5/1.0/1.5 × entry_atr_pct.
    # Tunes the live runner_trail_short_atr_mult: compare these vs actual + post_exit_peak to pick N.
    shadow_atr05_pnl = Column(Float, nullable=True)
    shadow_atr05_min = Column(Float, nullable=True)
    shadow_atr10_pnl = Column(Float, nullable=True)
    shadow_atr10_min = Column(Float, nullable=True)
    shadow_atr15_pnl = Column(Float, nullable=True)
    shadow_atr15_min = Column(Float, nullable=True)
    # Jun 17 PM — give-back-cap shadows (ATR-floor at live N + lock, give-back capped at frac×peak).
    # frac 0.25/0.35/0.50 → tune runner_trail_short_giveback_frac from data (which captures most w/o noise-stop).
    shadow_cap025_pnl = Column(Float, nullable=True)
    shadow_cap025_min = Column(Float, nullable=True)
    shadow_cap035_pnl = Column(Float, nullable=True)
    shadow_cap035_min = Column(Float, nullable=True)
    shadow_cap050_pnl = Column(Float, nullable=True)
    shadow_cap050_min = Column(Float, nullable=True)
    # Jul 6 — ARM-LEVEL shadows: arm the 0.25-trail at peak≥0.35/0.40 instead of the live 0.45.
    # Tracked on EVERY trade from tick 1 (not just the armed cohort) so both sides of the
    # arm-lowering trade-off are measured: rescues on 0.35-0.45 peakers that died AND early-chop
    # on runners. Decision offline at N≥30 from the orders CSV. Unfired → actual close pnl.
    shadow_arm035_pnl = Column(Float, nullable=True)
    shadow_arm035_min = Column(Float, nullable=True)
    shadow_arm040_pnl = Column(Float, nullable=True)
    shadow_arm040_min = Column(Float, nullable=True)
    # ===== LEASH SHADOW END =====

    # Jun 1, 2026 — RUNNER STRETCH-TRAIL: live peak |price−EMA5| stretch since
    # entry, tracked in the monitor loop while the runner trail is armed. Used to
    # fire RUNNER_TRAIL when live stretch ≤ runner_trail_k × this. Persisted so it
    # survives bot restart (re-arms cleanly otherwise). See CLAUDE.md Jun 1.
    runner_peak_stretch = Column(Float, nullable=True)
    # Jun 17 — which mechanism bound the runner-trail exit: 'lock' (BE-ratchet), 'atr'
    # (ATR-floor give-back), or 'stretch' (K×peak-stretch fallback). Lets the report tell a
    # ratchet-save apart from a normal ATR exit (both close as FLIP_RUNNER_TRAIL).
    runner_trail_bound = Column(String(15), nullable=True)

    # Signal Lost Flag: trade was in signal-lost zone but kept open
    signal_lost_flagged = Column(Boolean, nullable=True, default=False)
    signal_lost_flag_pnl = Column(Float, nullable=True)
    signal_lost_flagged_at = Column(DateTime, nullable=True)
    # FL1 origin: "SIGNAL_LOST" (classic) or "WIDE_SL" (flagged from STOP_LOSS_WIDE instead of closing)
    fl1_origin = Column(String(20), nullable=True)
    # FL2 double-flag: promoted from FL1 security gap, monitored against fl2_recovery_target / fl2_deep_stop
    fl2_flagged = Column(Boolean, nullable=True, default=False)
    fl2_flagged_at = Column(DateTime, nullable=True)
    fl2_flag_pnl = Column(Float, nullable=True)  # P&L at moment FL2 was triggered

    # Regime Neutral tracking: what happened when BTC regime went NEUTRAL during trade
    regime_neutral_hit_at = Column(DateTime, nullable=True)
    regime_neutral_pnl = Column(Float, nullable=True)
    regime_comeback_at = Column(DateTime, nullable=True)
    regime_comeback_pnl = Column(Float, nullable=True)
    regime_opposite_at = Column(DateTime, nullable=True)
    regime_opposite_pnl = Column(Float, nullable=True)

    # In-trade RSI pattern tracking (first occurrence, no P&L threshold)
    first_rsi2_pnl = Column(Float, nullable=True)
    first_rsi2_minutes = Column(Float, nullable=True)
    first_rsi3_pnl = Column(Float, nullable=True)
    first_rsi3_minutes = Column(Float, nullable=True)
    
    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=func.now())
    closed_at = Column(DateTime, nullable=True)
    
    # Close reason (FL_ prefix for signal-lost-flagged trades)
    close_reason = Column(String(40), nullable=True)
    
    # Entry order type: MAKER, TAKER, or TAKER_FALLBACK
    entry_order_type = Column(String(15), nullable=True, default="TAKER")
    # Exit order type: MAKER, TAKER, or TAKER_FALLBACK
    exit_order_type = Column(String(15), nullable=True, default="TAKER")
    
    # Paper trading flag
    is_paper = Column(Boolean, nullable=False, default=True)
    
    # Additional info
    notes = Column(Text, nullable=True)


# Canonical phantom source allowlist (Jul 5 — SINGLE SOURCE OF TRUTH). Both the engine's
# seeding keep-set (_PHANTOM_KEEP_SOURCES) and database.py's startup purge are generated
# from THIS tuple. History: the purge had its own hardcoded copy; adding PASS:/SPIKE_REV
# to the engine but not the purge caused every app restart to DELETE the new sources' rows
# (Jul-5 incident: 4 PASS phantoms purged by the spike-phantom deploy's restart).
PHANTOM_KEEP_SOURCES = (
    "LONG_UNMATCHED_ONLY",
    "MOMENTUM_SHORT_W1_REGIME",
    "PASS:LONG_UNMATCHED_ONLY",
    "SPIKE_REV_BTC",
    # Jul 5 — same-direction PASS phantoms of the two decision-gated flip-SHORT blockers:
    # BTC1H_SLOPE feeds the Jul-3 gate's locked revert (≥60% WR on N≥10 blocked → gate off;
    # the gate shipped WITHOUT this surface — the revert could never trigger, bug fix).
    # REGIME (bear≥80, #1 flip blocker at 290) measures what the bear-era filter forfeits in a bull.
    "PASS:FLIP_SHORT_BTC1H_SLOPE",
    "PASS:FLIP_SHORT_REGIME",
    # Jul 5 PM — revert surface of the LONG_BTC1H_DEADBAND ship (flat-1h DOA block,
    # N=9 discipline-override): re-open the band at >=60% WR on N>=10 blocked phantoms.
    "PASS:LONG_BTC1H_DEADBAND",
    # Jul 6 — revert surface of the MOMENTUM_SHORT_DEEPGAP ship (pair gap <= -1.0 block,
    # N=3 operator-directed): re-open at >=55% WR on N>=8 blocked phantoms.
    "PASS:MOMENTUM_SHORT_DEEPGAP",
    # Jul 8 — revert surface of the FLIP_SHORT_BTC_TRENDGAP ship (BTC EMA13-50 gap <= -0.22
    # depth block, N=16 operator-directed): re-open at net-admissible >=60% WR on N>=10.
    "PASS:FLIP_SHORT_BTC_TRENDGAP",
)

# Jul 5 (operator invariant: "phantoms are killed ONLY on reset, never on redeploy").
# The startup purge deletes ONLY the sources explicitly listed here (fail-SAFE: an
# unknown/new source always survives a deploy, even if someone forgets every registry).
# These are the pre-Jul-1 research sources whose rows were already purged once; the
# list exists only to keep legacy DBs clean. NEVER add a live source here.
PHANTOM_RETIRED_SOURCES = (
    "FAN_RATIO_GATE", "ATR_GAP_LONG", "PAIR_TREND_FILTER", "PAIR_ADX_MAX",
    "BTC_ADX_BLOCK_SHORT", "PAIR_RSI_ADX_CROSS", "BTC_RSI_ADX_CROSS", "PAIR_RSI_OB",
    "PASS:BTC_ADX_GATE_LOW", "PASS:BTC_RSI_ADX_CROSS", "PASS:FAN_RATIO_GATE",
)


class PhantomFlip(Base):
    """Phantom Flip Tracker (Jun 13, observation-only). When an entry is BLOCKED by
    fan-ratio / ATR×gap / pair-trend, a virtual OPPOSITE-direction ("fade") position
    is simulated on live websocket prices with a real entry/SL/trailing exit. Records
    realized P&L to answer: does the reversion the block implies actually pay, or just
    whipsaw? NEVER affects live trading. TO REMOVE: drop this model + the
    _PHANTOM_FLIP_STATE block in trading_engine.py + the perf block in main.py + UI."""
    __tablename__ = "phantom_flips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False)
    source_filter = Column(String(30), nullable=False)   # FAN_RATIO_GATE / ATR_GAP_LONG / PAIR_TREND_FILTER
    blocked_direction = Column(String(6), nullable=False)  # the signal that WAS blocked
    flip_direction = Column(String(6), nullable=False)     # the simulated opposite
    entry_price = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=True)        # realized raw price-move % of the flip
    peak_pct = Column(Float, nullable=True)       # max favorable excursion
    trough_pct = Column(Float, nullable=True)     # max adverse excursion
    exit_reason = Column(String(16), nullable=True)  # sl / trail / horizon
    is_paper = Column(Boolean, default=True)
    entry_at = Column(DateTime, nullable=True)
    exit_at = Column(DateTime, nullable=True)
    # Jun 14: for LONG_UNMATCHED_ONLY (matched-long fade) only — which pattern family the
    # blocked long matched, so the fade can be sub-divided. "C+W" / "C" / "W" (None for
    # other sources, where C/W is not computed at the seed site). Forward-only.
    entry_cohort = Column(String(32), nullable=True)  # Jun 29: 8->32 — now holds joined pattern codes ("C6+W6"); SQLite ignores length, PG-safe
    # Jun 15: full entry-indicator capture (parity with the flip Order / normal trade) so the
    # phantom POOL is analyzable by RSI / ATR / fan-ratio / regime cross-batch. Forward-only
    # (existing rows stay NULL). Populated from _flip_entry_fields() at seed time.
    entry_gap = Column(Float, nullable=True)
    entry_rsi = Column(Float, nullable=True)
    entry_rsi_prev = Column(Float, nullable=True)
    entry_adx = Column(Float, nullable=True)
    entry_adx_prev = Column(Float, nullable=True)
    entry_adx_delta = Column(Float, nullable=True)
    entry_pos_di = Column(Float, nullable=True)
    entry_neg_di = Column(Float, nullable=True)
    entry_ema_gap_5_8 = Column(Float, nullable=True)
    entry_ema_gap_8_13 = Column(Float, nullable=True)
    entry_ema5_stretch = Column(Float, nullable=True)
    entry_price_vs_ema5_pct = Column(Float, nullable=True)
    entry_atr_pct = Column(Float, nullable=True)
    entry_pair_ema20_ema50_gap_pct = Column(Float, nullable=True)
    entry_dist_from_ema13_pct = Column(Float, nullable=True)
    entry_range_position = Column(Float, nullable=True)
    entry_btc_adx = Column(Float, nullable=True)
    entry_btc_rsi = Column(Float, nullable=True)
    entry_btc_ema20_slope = Column(Float, nullable=True)
    entry_btc_1h_slope = Column(Float, nullable=True)
    entry_btc_dist_from_ema13_pct = Column(Float, nullable=True)
    entry_btc_trend_gap_pct = Column(Float, nullable=True)  # Jul 8 — BTC EMA13-50 gap (flip depth gate surface); forward-only
    entry_macro_trend = Column(String(20), nullable=True)
    entry_btc_regime = Column(String(20), nullable=True)
    # Jun 15 (full parity round 2): pair slopes, market context, quality score.
    entry_ema20_slope = Column(Float, nullable=True)
    entry_ema50_slope = Column(Float, nullable=True)
    entry_global_volume_ratio = Column(Float, nullable=True)
    entry_pair_volume_ratio = Column(Float, nullable=True)
    entry_bull_pct = Column(Float, nullable=True)
    entry_bear_pct = Column(Float, nullable=True)
    entry_pair_volume_24h_usd = Column(Float, nullable=True)
    entry_pair_rank = Column(Integer, nullable=True)
    entry_quality_score = Column(Float, nullable=True)  # Float to match Order.entry_quality_score
    # Jun 15 (full parity round 3): BTC prev/higher-TF companions — the "vs prev candle /
    # vs 6-ago / 1h" values the "Performance by BTC ... Direction / Volatility / 1h RSI"
    # tables compare against. Without these on the Order, flips were invisible to those
    # tables; mirrored onto the phantom row too so the phantom analyses see the same slice.
    entry_btc_adx_prev = Column(Float, nullable=True)
    entry_btc_rsi_prev = Column(Float, nullable=True)
    entry_btc_rsi_prev6 = Column(Float, nullable=True)
    entry_btc_atr_pct = Column(Float, nullable=True)
    entry_btc_rsi_1h = Column(Float, nullable=True)
    entry_btc_rsi_1h_prev = Column(Float, nullable=True)


class Transaction(Base):
    """Transaction history log"""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, nullable=False)
    binance_order_id = Column(String(50), nullable=True)
    
    # Transaction details
    pair = Column(String(20), nullable=False)
    action = Column(String(20), nullable=False)  # OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT
    
    # Price and amount
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    investment = Column(Float, nullable=False)
    leverage = Column(Float, nullable=False)
    notional_value = Column(Float, nullable=False)
    fee = Column(Float, nullable=False, default=0.0)
    
    # Order type: MAKER or TAKER
    order_type = Column(String(15), nullable=True, default="TAKER")
    
    # Timestamp
    timestamp = Column(DateTime, nullable=False, default=func.now())
    
    # Paper trading flag
    is_paper = Column(Boolean, nullable=False, default=True)


class BotState(Base):
    """Bot state tracking"""
    __tablename__ = "bot_state"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Status
    is_running = Column(Boolean, nullable=False, default=False)
    is_paper_mode = Column(Boolean, nullable=False, default=True)
    
    # Timer
    started_at = Column(DateTime, nullable=True)
    total_runtime_seconds = Column(Integer, nullable=False, default=0)
    last_pause_at = Column(DateTime, nullable=True)
    
    # Paper trading balance
    paper_balance = Column(Float, nullable=False, default=10000.0)

    # Paper BNB balance (USDT equivalent)
    paper_bnb_balance_usd = Column(Float, nullable=False, default=500.0)

    # Immutable starting capital baseline (set ONCE at cold start, never changed
    # by config edits). Used as the denominator for Return Multiple / Daily
    # Compound Return so the metric stays comparable across runs even if the
    # operator edits paper_balance or paper_bnb_initial_usd in config mid-run.
    # In paper mode: paper_balance + paper_bnb_initial_usd at first init.
    # See CLAUDE.md May 5 entry on Return Multiple bug fix.
    runtime_initial_total_usd = Column(Float, nullable=True)

    # BTC regime tracking — persisted across restarts so regime age survives
    # downtime. Updated each scan cycle when BTC regime classification changes.
    # See CLAUDE.md May 5 entry on regime stability instrumentation.
    current_btc_regime = Column(String(20), nullable=True)
    btc_regime_started_at = Column(DateTime, nullable=True)

    # Filter block counters — persisted as JSON so the panel survives redeployments.
    # Format: '{"BTC_TREND_FILTER|SHORT": 12, "BTC_ADX_GATE|LONG": 3, ...}'
    # Restored into _filter_block_counts on initialize(); flushed by save_state().
    filter_block_counts_json = Column(Text, nullable=True)

    # Last BNB scheduled-check timestamp — persisted across restarts so the
    # check interval is respected across redeployments. Without this, every
    # restart triggered a fresh check ~60s after startup, causing repeated
    # tiny rebalance swaps when the operator deployed multiple times in a
    # short window. See CLAUDE.md May 7 entry on BNB swap-on-deploy fix.
    last_bnb_check_at = Column(DateTime, nullable=True)

    # Binance IP ban expiry (epoch seconds) -- persisted so it survives restarts
    ban_until = Column(Float, nullable=True, default=0.0)
    
    # Last updated
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())


class ConfigChangeLog(Base):
    """Log of configuration changes"""
    __tablename__ = "config_change_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    field = Column(String(100), nullable=False)
    old_value = Column(String(200), nullable=True)
    new_value = Column(String(200), nullable=True)
    changed_at = Column(DateTime, nullable=False, default=func.now())


class BnbSwapLog(Base):
    """Log of automatic USDT→BNB swaps for fee coverage"""
    __tablename__ = "bnb_swap_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=func.now())
    swap_type = Column(String(20), nullable=False)  # "scheduled" or "emergency"
    amount_usdt = Column(Float, nullable=False)
    bnb_price = Column(Float, nullable=False)
    amount_bnb = Column(Float, nullable=False)
    pre_bnb_usd = Column(Float, nullable=False)
    post_bnb_usd = Column(Float, nullable=False)
    pre_usdt = Column(Float, nullable=False)
    post_usdt = Column(Float, nullable=False)
    burn_rate = Column(Float, nullable=True)
    is_paper = Column(Boolean, nullable=False, default=True)


class PairData(Base):
    """Cached pair data with indicators"""
    __tablename__ = "pair_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair = Column(String(20), nullable=False, unique=True)
    
    # Price data
    price = Column(Float, nullable=False)
    volume_24h = Column(Float, nullable=False)
    avg_volume = Column(Float, nullable=True)
    
    # EMAs
    ema5 = Column(Float, nullable=True)
    ema5_prev3 = Column(Float, nullable=True)
    ema8 = Column(Float, nullable=True)
    ema13 = Column(Float, nullable=True)
    ema20 = Column(Float, nullable=True)
    ema20_prev3 = Column(Float, nullable=True)
    ema50 = Column(Float, nullable=True)  # May 7: persisted for exit_pair_ema20_ema50_gap_pct capture in _get_exit_trend_gaps
    
    # Indicators
    rsi = Column(Float, nullable=True)
    rsi_prev1 = Column(Float, nullable=True)
    rsi_prev2 = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    
    # Signal
    signal = Column(String(10), nullable=True)  # LONG, SHORT, NOTHING
    confidence = Column(String(10), nullable=True)
    
    # Macro trend regime (EMA50-based)
    macro_regime = Column(String(10), nullable=True)  # BULLISH, BEARISH, NEUTRAL
    
    # Volume ratio (current / 20-bar avg)
    volume_ratio = Column(Float, nullable=True)
    
    # Timestamp
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())


class Investor(Base):
    """Portfolio investor with share-based ownership tracking.

    Jul 3, 2026: sqlite_autoincrement — without AUTOINCREMENT, SQLite reuses a deleted
    investor's row-id, so a re-added (or brand-new) investor silently inherited the old
    id's ledger history. IDs are now never reused (migration rebuilds the table and seeds
    sqlite_sequence past every id ever seen in investor_ledger)."""
    __tablename__ = "investors"
    __table_args__ = {'sqlite_autoincrement': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    shares = Column(Float, nullable=False, default=0.0)
    total_deposited = Column(Float, nullable=False, default=0.0)
    total_withdrawn = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=func.now())


class InvestorLedger(Base):
    """Dated cash-flow ledger per investor (Jun 1, 2026). Audit trail — one row
    per deposit / withdrawal / cash-out. The Investor.shares/NAV math stays the
    source of truth for ownership; this is the dated history alongside it."""
    __tablename__ = "investor_ledger"

    id = Column(Integer, primary_key=True, autoincrement=True)
    investor_id = Column(Integer, nullable=False, index=True)
    type = Column(String(12), nullable=False)        # DEPOSIT | WITHDRAW | CASHOUT | OPENING
    amount = Column(Float, nullable=False)            # USD moved (always positive)
    nav_at_time = Column(Float, nullable=True)        # NAV/share at the moment
    shares_delta = Column(Float, nullable=True)       # + on deposit, − on withdraw
    note = Column(String(200), nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())


class NavSnapshot(Base):
    """Daily NAV/share snapshot (Jul 2, 2026). One row per UTC-3 calendar day,
    upserted hourly so today's row tracks live equity. NAV history is the
    deposit/withdrawal-proof performance record (the equity curve distorts on
    cash flows; NAV/share does not) and the basis for any future HWM fee math."""
    __tablename__ = "nav_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True, index=True)  # YYYY-MM-DD (UTC-3, matches dashboard days)
    portfolio_value = Column(Float, nullable=False)   # total equity: free + margin + unrealized + BNB
    total_shares = Column(Float, nullable=False)
    nav_per_share = Column(Float, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
