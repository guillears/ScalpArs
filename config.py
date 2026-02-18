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
    tp_min: float = 0.6  # % of notional
    pullback_trigger: float = 0.3  # % price pullback from peak
    gap_min: float = 0.08  # % minimum gap required (EMA5-EMA20)/price
    gap_max: float = 0.40  # % maximum gap allowed (filters overextended entries)
    gap_enabled: bool = True  # Whether to enforce gap requirement
    # Break-even stop loss: once peak P&L reaches breakeven_trigger,
    # the effective stop loss moves from stop_loss to breakeven_offset
    breakeven_trigger: float = 0.15  # P&L % that activates break-even protection
    breakeven_offset: float = -0.05  # New effective SL once break-even is active (slightly negative for fees/slippage)


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


class TradingConfig(BaseModel):
    """Main trading configuration"""
    # Trading fee per side (Binance futures taker fee)
    # Applied to both entry and exit orders (total fee = 2 * trading_fee)
    trading_fee: float = 0.0004  # 0.04% per side
    
    # Paper trading
    paper_trading: bool = True
    paper_balance: float = 2000.0  # Starting balance for paper trading
    
    # Trading pairs limit (how many top pairs by volume to trade)
    trading_pairs_limit: int = 50  # 5, 10, 20, or 50
    
    # Investment settings
    investment: InvestmentConfig = InvestmentConfig()
    
    # Signal thresholds
    thresholds: SignalThresholds = SignalThresholds()
    
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
            breakeven_trigger=0.15,
            breakeven_offset=-0.05
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
            breakeven_trigger=0.20,
            breakeven_offset=-0.05
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
            breakeven_trigger=0.25,
            breakeven_offset=-0.03
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
            breakeven_trigger=0.30,
            breakeven_offset=-0.02
        )
    }


class Settings(BaseSettings):
    """Application settings from environment"""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    app_env: str = "development"
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///./scalpars.db"
    
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
