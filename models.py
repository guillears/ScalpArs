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
    # RSI(12) value at time of entry
    entry_rsi = Column(Float, nullable=True)
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

    # Exit quality: Price vs EMA5 at exit
    exit_price_vs_ema5_pct = Column(Float, nullable=True)
    exit_ema5_slope_pct = Column(Float, nullable=True)
    exit_ema5_crossed = Column(Boolean, nullable=True)

    # Post-exit regret tracking: hypothetical P&L if the trade had stayed open for N minutes after close
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
    post_exit_signal_regained_minutes = Column(Float, nullable=True)
    post_exit_pnl_at_signal_regained = Column(Float, nullable=True)
    post_exit_floor_before_signal_regain = Column(Float, nullable=True)

    # Phantom BE shadow tracking: what would have happened if BE L1/L2 were active
    phantom_be_l1_triggered_at = Column(DateTime, nullable=True)
    phantom_be_l1_would_exit_pnl = Column(Float, nullable=True)
    phantom_be_l2_triggered_at = Column(DateTime, nullable=True)
    phantom_be_l2_would_exit_pnl = Column(Float, nullable=True)

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
    """Portfolio investor with share-based ownership tracking"""
    __tablename__ = "investors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    shares = Column(Float, nullable=False, default=0.0)
    total_deposited = Column(Float, nullable=False, default=0.0)
    total_withdrawn = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=func.now())
