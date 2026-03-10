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
    # Break-even stop loss: once peak P&L reaches breakeven_trigger,
    # the effective stop loss moves from stop_loss to breakeven_offset
    breakeven_trigger: float = 0.15  # P&L % that activates break-even protection
    breakeven_offset: float = -0.05  # New effective SL once break-even is active (slightly negative for fees/slippage)
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
    ema_gap_threshold: float = 3.0  # % minimum EMA5-EMA8 gap for momentum signals
    ema_gap_5_8_max: float = 0.0  # Max EMA5-EMA8 gap % for entry (0 = disabled)
    adx_strong: float = 16.0  # ADX threshold for STRONG_BUY
    adx_very_strong: float = 30.0  # ADX threshold for VERY_STRONG
    momentum_ema20_filter_long: bool = True
    momentum_ema20_filter_short: bool = True
    momentum_ema20_slope_filter_long: bool = True
    momentum_ema20_slope_filter_short: bool = True
    macro_trend_filter_enabled: bool = True
    macro_trend_neutral_mode: str = "both"  # "both" or "none"
    macro_trend_flat_threshold: float = 0.07  # % change below which EMA20 slope is considered neutral
    momentum_long_rsi_min: float = 55.0  # Min RSI for momentum LONG (0 = disabled)
    momentum_long_rsi_max: float = 100.0  # Max RSI for momentum LONG (100 = disabled)
    momentum_short_rsi_max: float = 50.0  # Max RSI for momentum SHORT (100 = disabled)
    momentum_adx_max: float = 100.0  # Max ADX for momentum entries (100 = disabled)
    momentum_short_rsi_min: float = 30.0  # Min RSI for momentum SHORT - avoid shorting oversold (0 = disabled)
    btc_global_filter_enabled: bool = True  # Use BTC regime to gate all pairs (overrides per-pair regime)
    signal_lost_exit_enabled: bool = True  # Close when EMA5/EMA8 momentum reverses while in profit
    signal_lost_min_profit: float = 0.05  # Min P&L % (notional) to trigger signal-lost exit
    ema5_slope_exit_enabled: bool = True  # Exit when EMA5 slope decelerates (momentum loss)
    ema5_slope_lookback: int = 3  # Number of candles back for EMA5 slope calculation
    ema5_slope_threshold: float = 0.01  # Min EMA5 slope % to stay in trade (0 = original behavior)
    price_ema5_exit_ratio: float = 0.3  # Exit when price-to-EMA5 distance drops to this fraction of peak (0 = disabled)
    min_peak_ema5_gap_pct: float = 0.05  # Min peak gap (% of entry price) before distance trailing activates (0 = no minimum)
    pnl_trailing_trigger: float = 0.1  # Min peak P&L % to activate P&L trailing exit (0 = disabled)
    pnl_trailing_ratio: float = 0.5  # Ratio when signal lost (MOMENTUM_EXIT) -- tighter
    pnl_trailing_ratio_signal_active: float = 0.3  # Ratio when signal active (PNL_TRAILING) -- wider
    rsi_momentum_filter_enabled: bool = True  # Block LONG if RSI falling, block SHORT if RSI rising (vs 3 candles ago)


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
    # Trading fee per side (Binance futures taker fee)
    # Applied to both entry and exit orders (total fee = 2 * trading_fee)
    trading_fee: float = 0.0004  # 0.04% per side
    
    # Paper trading
    paper_trading: bool = True
    paper_balance: float = 2000.0  # Starting balance for paper trading
    
    # Trading pairs limit (how many top pairs by volume to trade)
    trading_pairs_limit: int = 20  # 5, 10, 20, or 50
    
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
            breakeven_trigger=0.18,
            breakeven_offset=0.10
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
            breakeven_trigger=0.18,
            breakeven_offset=0.10
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
