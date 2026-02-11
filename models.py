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
    confidence = Column(String(10), nullable=False)  # LOW, MEDIUM, HIGH, EXTREME
    
    # Fees
    entry_fee = Column(Float, nullable=False, default=0.0)
    exit_fee = Column(Float, nullable=True, default=0.0)
    total_fee = Column(Float, nullable=True, default=0.0)
    
    # P&L tracking
    pnl = Column(Float, nullable=True, default=0.0)  # Realized P&L
    pnl_percentage = Column(Float, nullable=True, default=0.0)
    peak_pnl = Column(Float, nullable=False, default=0.0)  # For trailing stop
    high_price_since_entry = Column(Float, nullable=True)  # For LONG
    low_price_since_entry = Column(Float, nullable=True)  # For SHORT
    
    # Dynamic TP tracking
    current_tp_level = Column(Integer, nullable=False, default=1)  # Which TP level (1, 2, 3, ...)
    dynamic_tp_target = Column(Float, nullable=True)  # Current TP target (% of notional)
    
    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=func.now())
    closed_at = Column(DateTime, nullable=True)
    
    # Close reason
    close_reason = Column(String(20), nullable=True)
    
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
    
    # Last updated
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())


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
    ema8 = Column(Float, nullable=True)
    ema13 = Column(Float, nullable=True)
    ema20 = Column(Float, nullable=True)
    
    # Indicators
    rsi = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    
    # Signal
    signal = Column(String(10), nullable=True)  # LONG, SHORT, NOTHING
    confidence = Column(String(10), nullable=True)
    
    # Timestamp
    updated_at = Column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
