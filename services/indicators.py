"""
SCALPARS Trading Platform - Technical Indicators
"""
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from typing import Dict, List, Optional, Tuple
import config as config_module
from config import ConfidenceLevel


def calculate_indicators(ohlcv: List) -> Dict:
    """
    Calculate all technical indicators from OHLCV data
    
    Args:
        ohlcv: List of [timestamp, open, high, low, close, volume]
    
    Returns:
        Dictionary with all indicator values
    """
    if not ohlcv or len(ohlcv) < 51:  # Need at least 51 candles for EMA50
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Calculate EMAs
    ema5 = EMAIndicator(close=df['close'], window=5).ema_indicator()
    ema8 = EMAIndicator(close=df['close'], window=8).ema_indicator()
    ema13 = EMAIndicator(close=df['close'], window=13).ema_indicator()
    ema20 = EMAIndicator(close=df['close'], window=20).ema_indicator()
    ema50 = EMAIndicator(close=df['close'], window=50).ema_indicator()
    
    # Calculate RSI (12 period as specified)
    rsi = RSIIndicator(close=df['close'], window=12).rsi()
    
    # Calculate ADX (14 period default)
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    adx = adx_indicator.adx()
    
    # Get average volume (20 period)
    avg_volume = df['volume'].rolling(window=20).mean()
    
    # Get latest values
    return {
        'price': float(df['close'].iloc[-1]),
        'ema5': float(ema5.iloc[-1]) if not pd.isna(ema5.iloc[-1]) else None,
        'ema5_prev1': float(ema5.iloc[-2]) if len(ema5) >= 2 and not pd.isna(ema5.iloc[-2]) else None,
        'ema5_prev3': float(ema5.iloc[-4]) if len(ema5) >= 4 and not pd.isna(ema5.iloc[-4]) else None,
        'ema8': float(ema8.iloc[-1]) if not pd.isna(ema8.iloc[-1]) else None,
        'ema8_prev1': float(ema8.iloc[-2]) if len(ema8) >= 2 and not pd.isna(ema8.iloc[-2]) else None,
        'ema13': float(ema13.iloc[-1]) if not pd.isna(ema13.iloc[-1]) else None,
        'ema20': float(ema20.iloc[-1]) if not pd.isna(ema20.iloc[-1]) else None,
        'ema20_prev6': float(ema20.iloc[-7]) if len(ema20) >= 7 and not pd.isna(ema20.iloc[-7]) else None,
        'ema50': float(ema50.iloc[-1]) if not pd.isna(ema50.iloc[-1]) else None,
        'ema50_prev12': float(ema50.iloc[-13]) if len(ema50) >= 13 and not pd.isna(ema50.iloc[-13]) else None,
        'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        'rsi_prev1': float(rsi.iloc[-2]) if len(rsi) >= 2 and not pd.isna(rsi.iloc[-2]) else None,
        'rsi_prev2': float(rsi.iloc[-3]) if len(rsi) >= 3 and not pd.isna(rsi.iloc[-3]) else None,
        'rsi_prev3': float(rsi.iloc[-4]) if len(rsi) >= 4 and not pd.isna(rsi.iloc[-4]) else None,
        'adx': float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None,
        'adx_prev1': float(adx.iloc[-2]) if len(adx) >= 2 and not pd.isna(adx.iloc[-2]) else None,
        'volume': float(df['volume'].iloc[-1]),
        'avg_volume': float(avg_volume.iloc[-1]) if not pd.isna(avg_volume.iloc[-1]) else None
    }


def is_signal_direction_active(direction: str, ema5: float, ema8: float, ema20: float, price: float) -> bool:
    """Check if the core directional momentum still holds (ignoring entry filters)."""
    if not all([ema5, ema8, ema20, price]):
        return False
    if direction == "LONG":
        return ema5 > ema8 and price > ema20
    elif direction == "SHORT":
        return ema5 < ema8 and price < ema20
    return False


def determine_macro_regime(ema_current: float, ema_prev: float, flat_threshold: float = 0.07) -> str:
    """
    Determine macro trend regime from EMA slope (EMA20 by default).
    Returns "BULLISH", "BEARISH", or "NEUTRAL".
    """
    if ema_current is None or ema_prev is None or ema_prev == 0:
        return "NEUTRAL"
    pct_change = ((ema_current - ema_prev) / ema_prev) * 100
    if pct_change > flat_threshold:
        return "BULLISH"
    elif pct_change < -flat_threshold:
        return "BEARISH"
    return "NEUTRAL"


def get_signal(
    ema5: float,
    ema8: float,
    ema13: float,
    ema20: float,
    rsi: float,
    adx: float,
    volume: float,
    avg_volume: float,
    price: float = None,
    config: Optional[Dict] = None,
    ema20_prev6: float = None,
    ema50: float = None,
    ema50_prev12: float = None,
    rsi_prev3: float = None,
    ema5_prev1: float = None,
    ema8_prev1: float = None
) -> Tuple[str, Optional[str]]:
    """
    Generate trading signal based on indicators
    
    Signal Logic:
    - LONG: Bullish EMA stack (5>8>13>20) + oversold RSI conditions + gap check
    - SHORT: Bearish EMA stack (5<8<13<20) + overbought RSI conditions + gap check
    
    Confidence Levels (based on RSI + ADX + Volume):
    - EXTREME: Most extreme RSI + ADX > 35 + Volume spike
    - HIGH: Extreme RSI + ADX > 25 (strong trend)
    - MEDIUM: Moderate RSI + ADX 20-25 (moderate trend)
    - LOW: Mild RSI + ADX < 20 (weak trend, marginal signal)
    
    Returns:
        Tuple of (signal, confidence) where signal is LONG/SHORT/NOTHING
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if any(v is None for v in [ema5, ema8, ema13, ema20, rsi, adx]):
        logger.debug(f"Signal check failed - None values: ema5={ema5}, ema8={ema8}, ema13={ema13}, ema20={ema20}, rsi={rsi}, adx={adx}")
        return "NOTHING", None
    
    # Load thresholds from config (always get fresh from module to pick up updates)
    if config is None:
        th = config_module.trading_config.thresholds
        conf_levels = config_module.trading_config.confidence_levels
    else:
        from config import SignalThresholds
        th = SignalThresholds(**config) if isinstance(config, dict) else config
        conf_levels = config_module.trading_config.confidence_levels
    
    # Calculate gap: (EMA5 - EMA20) / price * 100
    gap = None
    if price and price > 0:
        gap = ((ema5 - ema20) / price) * 100
    
    def check_gap_and_mode(direction: str, confidence: str) -> bool:
        """Check if gap requirement and trade mode are satisfied"""
        conf = conf_levels.get(confidence)
        if not conf or not conf.enabled:
            return False
        
        # Check trade mode
        if direction == "LONG" and conf.trade_mode not in ["long", "both"]:
            return False
        if direction == "SHORT" and conf.trade_mode not in ["short", "both"]:
            return False
        
        # Check gap requirement (range: gap_min <= |gap| <= gap_max)
        if conf.gap_enabled and gap is not None:
            gap_min = getattr(conf, 'gap_min', 0.08)
            gap_max = getattr(conf, 'gap_max', 0.40)
            if direction == "LONG":
                if gap < gap_min:
                    logger.debug(f"LONG {confidence} rejected: gap {gap:.4f}% < min {gap_min}%")
                    return False
                if gap > gap_max:
                    logger.debug(f"LONG {confidence} rejected: gap {gap:.4f}% > max {gap_max}% (overextended)")
                    return False
            if direction == "SHORT":
                abs_gap = abs(gap)
                if abs_gap < gap_min:
                    logger.debug(f"SHORT {confidence} rejected: |gap| {abs_gap:.4f}% < min {gap_min}%")
                    return False
                if abs_gap > gap_max:
                    logger.debug(f"SHORT {confidence} rejected: |gap| {abs_gap:.4f}% > max {gap_max}% (overextended)")
                    return False
        
        # EMA5 Stretch filter: reject if price too far from EMA5 (abs for defensive edge-case coverage)
        max_stretch = getattr(conf, 'max_ema5_stretch', 0.14)
        if price and price > 0 and max_stretch > 0:
            stretch_pct = abs(price - ema5) / price * 100
            allowed = stretch_pct <= max_stretch
            if not allowed:
                logger.debug(f"{direction} {confidence} rejected: EMA5 stretch {stretch_pct:.4f}% > max {max_stretch}% | price={price}, ema5={ema5}")
                return False
            logger.debug(f"{direction} {confidence} stretch OK: {stretch_pct:.4f}% <= {max_stretch}% | price={price}, ema5={ema5}")
        
        return True
    
    # --- Macro trend regime (EMA20 slope) ---
    macro_filter_enabled = getattr(th, 'macro_trend_filter_enabled', True)
    neutral_mode = getattr(th, 'macro_trend_neutral_mode', 'both')
    flat_threshold = getattr(th, 'macro_trend_flat_threshold', 0.02)
    regime = determine_macro_regime(ema20, ema20_prev6, flat_threshold)
    
    def regime_allows(direction: str) -> bool:
        if not macro_filter_enabled:
            return True
        if regime == "BULLISH":
            return direction == "LONG"
        if regime == "BEARISH":
            return direction == "SHORT"
        # NEUTRAL
        return neutral_mode == "both"
    
    # --- Momentum signals (EMA5/EMA8 gap) - evaluated FIRST ---
    ema20_filter_long = getattr(th, 'momentum_ema20_filter_long', True)
    ema20_filter_short = getattr(th, 'momentum_ema20_filter_short', True)
    ema20_slope_long = getattr(th, 'momentum_ema20_slope_filter_long', True)
    ema20_slope_short = getattr(th, 'momentum_ema20_slope_filter_short', True)
    long_rsi_min = getattr(th, 'momentum_long_rsi_min', 0)
    long_rsi_max = getattr(th, 'momentum_long_rsi_max', 100)
    short_rsi_max = getattr(th, 'momentum_short_rsi_max', 100)
    short_rsi_min = getattr(th, 'momentum_short_rsi_min', 0)
    adx_max = getattr(th, 'momentum_adx_max', 100)
    adx_s_long = getattr(th, 'adx_strong_long', th.adx_strong)
    adx_vs_long = getattr(th, 'adx_very_strong_long', th.adx_very_strong)
    adx_max_long = getattr(th, 'momentum_adx_max_long', adx_max)
    rsi_momentum_enabled = getattr(th, 'rsi_momentum_filter_enabled', True)
    gap_expanding_enabled = getattr(th, 'ema_gap_expanding_filter', True)

    if ema8 and ema8 > 0:
        if ema5 > ema8:
            if not regime_allows("LONG"):
                logger.debug(f"[MOMENTUM] LONG skipped: regime={regime}, ema20={ema20}, ema20_prev6={ema20_prev6}")
            elif ema20_filter_long and (price is None or price <= ema20):
                logger.debug(f"[MOMENTUM] LONG skipped: EMA20 filter active, price={price}, ema20={ema20}")
            elif ema20_slope_long and (ema20_prev6 is None or ema20 <= ema20_prev6):
                logger.debug(f"[MOMENTUM] LONG skipped: EMA20 slope filter active, ema20={ema20}, ema20_prev6={ema20_prev6}")
            elif rsi_momentum_enabled and rsi is not None and rsi_prev3 is not None and rsi < rsi_prev3:
                logger.debug(f"[MOMENTUM] LONG skipped: RSI falling ({rsi_prev3:.1f} -> {rsi:.1f}), momentum against LONG")
            elif long_rsi_min > 0 and rsi is not None and rsi < long_rsi_min:
                logger.debug(f"[MOMENTUM] LONG skipped: RSI {rsi:.1f} < min {long_rsi_min}")
            elif long_rsi_max < 100 and rsi is not None and rsi > long_rsi_max:
                logger.debug(f"[MOMENTUM] LONG skipped: RSI {rsi:.1f} > max {long_rsi_max}")
            elif adx_max_long < 100 and adx > adx_max_long:
                logger.debug(f"[MOMENTUM] LONG skipped: ADX {adx:.1f} > max_long {adx_max_long}")
            else:
                ema_gap_pct = ((ema5 - ema8) / ema8) * 100
                prev_gap_pct = ((ema5_prev1 - ema8_prev1) / ema8_prev1) * 100 if ema5_prev1 and ema8_prev1 and ema8_prev1 > 0 else None
                ema_gap_max = getattr(th, 'ema_gap_5_8_max', 0)
                long_gap_min = getattr(th, 'ema_gap_threshold_long', th.ema_gap_threshold)
                gap_threshold_met = ema_gap_pct >= long_gap_min
                if gap_expanding_enabled and prev_gap_pct is not None and ema_gap_pct <= prev_gap_pct:
                    logger.debug(f"[MOMENTUM] LONG skipped: EMA5-8 gap compressing ({prev_gap_pct:.4f}% -> {ema_gap_pct:.4f}%)")
                elif ema_gap_max > 0 and ema_gap_pct > ema_gap_max:
                    logger.debug(f"[MOMENTUM] LONG skipped: EMA5-8 gap {ema_gap_pct:.4f}% > max {ema_gap_max}")
                elif gap_threshold_met:
                    if adx > adx_vs_long:
                        if check_gap_and_mode("LONG", "VERY_STRONG"):
                            logger.info(f"[MOMENTUM] LONG VERY_STRONG: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime}, ema20_slope={'up' if ema20_prev6 and ema20 > ema20_prev6 else 'n/a'}")
                            return "LONG", "VERY_STRONG"
                    if adx > adx_s_long and adx <= adx_vs_long:
                        if check_gap_and_mode("LONG", "STRONG_BUY"):
                            logger.info(f"[MOMENTUM] LONG STRONG_BUY: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime}, ema20_slope={'up' if ema20_prev6 and ema20 > ema20_prev6 else 'n/a'}")
                            return "LONG", "STRONG_BUY"
        elif ema5 < ema8 and ema5 > 0:
            if not regime_allows("SHORT"):
                logger.debug(f"[MOMENTUM] SHORT skipped: regime={regime}, ema20={ema20}, ema20_prev6={ema20_prev6}")
            elif ema20_filter_short and (price is None or price >= ema20):
                logger.debug(f"[MOMENTUM] SHORT skipped: EMA20 filter active, price={price}, ema20={ema20}")
            elif ema20_slope_short and (ema20_prev6 is None or ema20 >= ema20_prev6):
                logger.debug(f"[MOMENTUM] SHORT skipped: EMA20 slope filter active, ema20={ema20}, ema20_prev6={ema20_prev6}")
            elif rsi_momentum_enabled and rsi is not None and rsi_prev3 is not None and rsi > rsi_prev3:
                logger.debug(f"[MOMENTUM] SHORT skipped: RSI rising ({rsi_prev3:.1f} -> {rsi:.1f}), momentum against SHORT")
            elif short_rsi_max < 100 and rsi is not None and rsi > short_rsi_max:
                logger.debug(f"[MOMENTUM] SHORT skipped: RSI {rsi:.1f} > max {short_rsi_max}")
            elif short_rsi_min > 0 and rsi is not None and rsi < short_rsi_min:
                logger.debug(f"[MOMENTUM] SHORT skipped: RSI {rsi:.1f} < min {short_rsi_min} (oversold)")
            elif adx_max < 100 and adx > adx_max:
                logger.debug(f"[MOMENTUM] SHORT skipped: ADX {adx:.1f} > max {adx_max}")
            else:
                ema_gap_pct = ((ema8 - ema5) / ema5) * 100
                prev_gap_pct = ((ema8_prev1 - ema5_prev1) / ema5_prev1) * 100 if ema5_prev1 and ema8_prev1 and ema5_prev1 > 0 else None
                ema_gap_max = getattr(th, 'ema_gap_5_8_max', 0)
                short_gap_min = getattr(th, 'ema_gap_threshold_short', th.ema_gap_threshold)
                gap_threshold_met = ema_gap_pct >= short_gap_min
                if gap_expanding_enabled and prev_gap_pct is not None and ema_gap_pct <= prev_gap_pct:
                    logger.debug(f"[MOMENTUM] SHORT skipped: EMA5-8 gap compressing ({prev_gap_pct:.4f}% -> {ema_gap_pct:.4f}%)")
                elif ema_gap_max > 0 and ema_gap_pct > ema_gap_max:
                    logger.debug(f"[MOMENTUM] SHORT skipped: EMA5-8 gap {ema_gap_pct:.4f}% > max {ema_gap_max}")
                elif gap_threshold_met:
                    if adx > th.adx_very_strong:
                        if check_gap_and_mode("SHORT", "VERY_STRONG"):
                            logger.info(f"[MOMENTUM] SHORT VERY_STRONG: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime}, ema20_slope={'down' if ema20_prev6 and ema20 < ema20_prev6 else 'n/a'}")
                            return "SHORT", "VERY_STRONG"
                    if adx > th.adx_strong and adx <= th.adx_very_strong:
                        if check_gap_and_mode("SHORT", "STRONG_BUY"):
                            logger.info(f"[MOMENTUM] SHORT STRONG_BUY: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime}, ema20_slope={'down' if ema20_prev6 and ema20 < ema20_prev6 else 'n/a'}")
                            return "SHORT", "STRONG_BUY"
    
    # Check for bullish EMA stack (LONG conditions - looking for oversold)
    if ema5 > ema8 > ema13 > ema20:
        # EXTREME: RSI ≤ 30 + ADX > 35 + Volume > 1.5x avg
        if (rsi <= th.long_rsi_extreme and 
            adx > th.adx_extreme and 
            volume > (avg_volume or volume) * th.volume_multiplier):
            if check_gap_and_mode("LONG", "EXTREME"):
                return "LONG", "EXTREME"
        
        # HIGH: RSI ≤ 30 + ADX > 25
        if rsi <= th.long_rsi_high and adx > th.adx_high:
            if check_gap_and_mode("LONG", "HIGH"):
                return "LONG", "HIGH"
        
        # MEDIUM: RSI ≤ 35 + ADX between 20-25
        if rsi <= th.long_rsi_medium and adx >= th.adx_medium:
            if check_gap_and_mode("LONG", "MEDIUM"):
                return "LONG", "MEDIUM"
        
        # LOW: RSI <= threshold + ADX between adx_low and adx_medium (weak trend)
        if rsi <= th.long_rsi_low and adx >= th.adx_low and adx < th.adx_medium:
            if check_gap_and_mode("LONG", "LOW"):
                return "LONG", "LOW"
        
        # EMA stacked but conditions not met - no trade
        return "NOTHING", "NO_TRADE"
    
    # Check for bearish EMA stack (SHORT conditions - looking for overbought)
    elif ema5 < ema8 < ema13 < ema20:
        logger.info(f"Bearish EMA stack detected: RSI={rsi:.1f}, ADX={adx:.1f}, gap={f'{gap:.4f}%' if gap else 'N/A'}")
        
        # EXTREME: RSI ≥ 70 + ADX > 35 + Volume > 1.5x avg
        if (rsi >= th.short_rsi_extreme and 
            adx > th.adx_extreme and 
            volume > (avg_volume or volume) * th.volume_multiplier):
            if check_gap_and_mode("SHORT", "EXTREME"):
                logger.info(f"SHORT EXTREME signal!")
                return "SHORT", "EXTREME"
        
        # HIGH: RSI ≥ 70 + ADX > 25
        if rsi >= th.short_rsi_high and adx > th.adx_high:
            if check_gap_and_mode("SHORT", "HIGH"):
                logger.info(f"SHORT HIGH signal!")
                return "SHORT", "HIGH"
        
        # MEDIUM: RSI ≥ 65 + ADX between 20-25
        if rsi >= th.short_rsi_medium and adx >= th.adx_medium:
            if check_gap_and_mode("SHORT", "MEDIUM"):
                logger.info(f"SHORT MEDIUM signal!")
                return "SHORT", "MEDIUM"
        
        # LOW: RSI >= threshold + ADX between adx_low and adx_medium (weak trend)
        if rsi >= th.short_rsi_low and adx >= th.adx_low and adx < th.adx_medium:
            if check_gap_and_mode("SHORT", "LOW"):
                logger.info(f"SHORT LOW signal!")
                return "SHORT", "LOW"
        
        # EMA stacked but conditions not met - no trade
        return "NOTHING", "NO_TRADE"
    
    # No clear EMA stack
    return "NOTHING", None


def check_exit_conditions(
    direction: str,
    entry_price: float,
    current_price: float,
    leverage: float,
    confidence: str,
    peak_pnl: float,
    trough_pnl: float = 0.0,
    quantity: float = 0,
    entry_fee: float = 0,
    investment: float = 0,
    config: Optional[Dict] = None,
    high_price: float = None,
    low_price: float = None,
    # New params for dynamic TP
    ema5: float = None,
    ema8: float = None,
    ema13: float = None,
    ema20: float = None,
    current_tp_level: int = 1,
    dynamic_tp_target: float = None,
    signal_active: bool = False,
    tp_trailing_enabled: bool = True
) -> Dict:
    """
    Check if position should be closed based on SL/TP/Trailing stop
    With dynamic TP extension when trend continues.
    
    Args:
        direction: LONG or SHORT
        entry_price: Entry price of the position
        current_price: Current market price
        leverage: Position leverage
        confidence: Confidence level (LOW, MEDIUM, HIGH, EXTREME)
        peak_pnl: Peak P&L reached since entry (used for TP min tracking)
        quantity: Position quantity (for fee calculation)
        entry_fee: Fee paid on entry
        investment: Margin/investment amount
        config: Optional config override
        high_price: Highest price since entry (for LONG trailing stop)
        low_price: Lowest price since entry (for SHORT trailing stop)
        ema5, ema8, ema13, ema20: Current EMA values for trend check
        current_tp_level: Current TP level (1, 2, 3, ...)
        dynamic_tp_target: Current TP target percentage
    
    Returns:
        Dict with:
            - should_close: bool
            - reason: Optional[str] (STOP_LOSS, TRAILING_STOP, EXTEND_TP, None)
            - peak_pnl: float
            - new_tp_level: int (if EXTEND_TP)
            - new_tp_target: float (if EXTEND_TP)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    tc = config_module.trading_config
    
    # Get confidence level config
    conf_config = tc.confidence_levels.get(confidence)
    if not conf_config:
        return {"should_close": False, "reason": None, "peak_pnl": peak_pnl, "trough_pnl": trough_pnl}
    
    # Calculate current P&L percentage INCLUDING FEES (on margin/investment)
    if direction == "LONG":
        raw_pnl = (current_price - entry_price) * quantity
    else:  # SHORT
        raw_pnl = (entry_price - current_price) * quantity
    
    # Calculate estimated exit fee
    current_notional = current_price * quantity
    estimated_exit_fee = current_notional * getattr(tc, 'taker_fee', tc.trading_fee)
    total_fees = entry_fee + estimated_exit_fee
    
    # P&L after fees
    pnl = raw_pnl - total_fees
    
    # P&L percentage as % of notional (not investment)
    entry_notional = entry_price * quantity if quantity > 0 else (investment * leverage if investment > 0 else 1)
    pnl_pct = (pnl / entry_notional) * 100
    
    # Get thresholds
    stop_loss = conf_config.stop_loss
    tp_min = conf_config.tp_min
    pullback_trigger = conf_config.pullback_trigger
    be_l1_trigger = conf_config.be_level1_trigger
    be_l1_offset = conf_config.be_level1_offset
    be_l2_trigger = conf_config.be_level2_trigger
    be_l2_offset = conf_config.be_level2_offset
    be_l3_trigger = conf_config.be_level3_trigger
    be_l3_offset = conf_config.be_level3_offset

    # Use dynamic_tp_target if set, otherwise use tp_min
    effective_tp_target = dynamic_tp_target if dynamic_tp_target is not None else tp_min
    
    # ALWAYS track maximum P&L seen BEFORE stop loss check (so break-even uses latest peak)
    if pnl_pct > 0:
        peak_pnl = max(peak_pnl, pnl_pct)
    if pnl_pct < 0:
        trough_pnl = min(trough_pnl, pnl_pct)
    
    # Trailing stop (pullback from peak price) activates once peak reaches TP target or at L2+.
    trailing_stop_active = peak_pnl >= effective_tp_target or current_tp_level >= 2
    
    # 3-Level break-even stop loss (highest level wins)
    effective_stop_loss = stop_loss
    breakeven_active = False
    be_level = 0
    be_enabled = getattr(conf_config, 'be_levels_enabled', True)
    if be_enabled and peak_pnl >= be_l3_trigger:
        breakeven_active = True
        be_level = 3
        effective_stop_loss = be_l3_offset
        logger.debug(f"[BREAKEVEN_L3] {direction} L{current_tp_level}: peak={peak_pnl:.4f}% >= L3={be_l3_trigger}%, SL={be_l3_offset}%")
    elif be_enabled and peak_pnl >= be_l2_trigger:
        breakeven_active = True
        be_level = 2
        effective_stop_loss = be_l2_offset
        logger.debug(f"[BREAKEVEN_L2] {direction} L{current_tp_level}: peak={peak_pnl:.4f}% >= L2={be_l2_trigger}%, SL={be_l2_offset}%")
    elif be_enabled and peak_pnl >= be_l1_trigger:
        breakeven_active = True
        be_level = 1
        effective_stop_loss = be_l1_offset
        logger.debug(f"[BREAKEVEN_L1] {direction} L{current_tp_level}: peak={peak_pnl:.4f}% >= L1={be_l1_trigger}%, SL={be_l1_offset}%")
    elif signal_active:
        effective_stop_loss = conf_config.signal_active_sl
        logger.debug(f"[SIGNAL_ACTIVE_SL] {direction} L{current_tp_level}: Signal still active, SL widened from {stop_loss}% to {effective_stop_loss}%")
    
    # Check Stop Loss (P&L based, with break-even adjustment in pre-TP zone only)
    if pnl_pct <= effective_stop_loss:
        if breakeven_active:
            close_reason = f"BREAKEVEN_SL_L{be_level}"
        elif signal_active:
            close_reason = f"STOP_LOSS_WIDE L{current_tp_level}"
        else:
            close_reason = f"STOP_LOSS L{current_tp_level}"
        logger.info(f"[{close_reason}] {direction} triggered: pnl_pct={pnl_pct:.4f}% <= effective_sl={effective_stop_loss}% (original_sl={stop_loss}%, peak={peak_pnl:.4f}%)")
        return {
            "should_close": True,
            "reason": close_reason,
            "peak_pnl": peak_pnl,
            "trough_pnl": trough_pnl,
            "tp_level": current_tp_level
        }
    
    # TP extension and trailing stop (skipped when tp_trailing_enabled=False)
    if not tp_trailing_enabled:
        return {"should_close": False, "reason": None, "peak_pnl": peak_pnl, "trough_pnl": trough_pnl}

    # Check if we've reached the current TP target
    if pnl_pct >= effective_tp_target:
        
        # Check if trend continues (for potential TP extension)
        trend_continues = False
        if all(v is not None for v in [ema5, ema8, ema13, ema20, current_price]):
            # Calculate gap
            gap = ((ema5 - ema20) / current_price) * 100
            
            if direction == "LONG":
                # Bullish: EMA5 > EMA8 > EMA13 > EMA20 AND gap > 0
                trend_continues = (ema5 > ema8 > ema13 > ema20) and (gap > 0)
            else:  # SHORT
                # Bearish: EMA5 < EMA8 < EMA13 < EMA20 AND gap < 0
                trend_continues = (ema5 < ema8 < ema13 < ema20) and (gap < 0)
            
            logger.info(f"[TP_CHECK] {direction} L{current_tp_level}: pnl={pnl_pct:.4f}% >= target={effective_tp_target:.4f}%, trend_continues={trend_continues}, gap={gap:.4f}%")
        
        if trend_continues:
            # Extend TP target - calculate correct level based on actual P&L
            # This allows jumping multiple levels if price moved fast
            calculated_level = int(pnl_pct / tp_min)
            new_tp_level = max(calculated_level, current_tp_level + 1)  # Never go backwards
            new_tp_target = new_tp_level * tp_min
            logger.info(f"[EXTEND_TP] {direction}: Extending from L{current_tp_level} ({effective_tp_target:.4f}%) to L{new_tp_level} ({new_tp_target:.4f}%) [pnl={pnl_pct:.4f}%]")
            return {
                "should_close": False,
                "reason": "EXTEND_TP",
                "peak_pnl": peak_pnl,
                "trough_pnl": trough_pnl,
                "new_tp_level": new_tp_level,
                "new_tp_target": new_tp_target
            }
        else:
            # Trend broken while at/above TP target -- exit immediately
            logger.info(f"[TREND_BREAK_EXIT] {direction} L{current_tp_level}: pnl={pnl_pct:.4f}% >= target={effective_tp_target:.4f}% but trend broken, closing now")
            return {
                "should_close": True,
                "reason": f"TRAILING_STOP L{current_tp_level}",
                "peak_pnl": peak_pnl,
                "trough_pnl": trough_pnl,
                "tp_level": current_tp_level
            }
    
    # peak_pnl is already tracked unconditionally above
    
    # Check trailing stop (PRICE-BASED pullback from best price)
    # trailing_stop_active was already computed above (before break-even SL check)
    
    # INFO logging for trailing stop tracking (to diagnose issues)
    if current_tp_level >= 2:
        if direction == "LONG":
            if high_price and high_price > 0:
                price_drop = ((high_price - current_price) / high_price) * 100
                logger.info(f"[TRAILING_CHECK] LONG L{current_tp_level}: entry={entry_price:.4f}, current={current_price:.4f}, HIGH={high_price:.4f}, drop={price_drop:.4f}%, trigger={pullback_trigger}%, pnl={pnl_pct:.4f}%")
            else:
                logger.warning(f"[TRAILING_CHECK] LONG L{current_tp_level}: HIGH_PRICE NOT SET! entry={entry_price:.4f}, current={current_price:.4f}, high={high_price}")
        else:
            if low_price and low_price > 0:
                price_rise = ((current_price - low_price) / low_price) * 100
                logger.info(f"[TRAILING_CHECK] SHORT L{current_tp_level}: entry={entry_price:.4f}, current={current_price:.4f}, LOW={low_price:.4f}, rise={price_rise:.4f}%, trigger={pullback_trigger}%, pnl={pnl_pct:.4f}%")
            else:
                logger.warning(f"[TRAILING_CHECK] SHORT L{current_tp_level}: LOW_PRICE NOT SET! entry={entry_price:.4f}, current={current_price:.4f}, low={low_price}")
    
    if trailing_stop_active:
        if direction == "LONG" and high_price and high_price > 0:
            # For LONG: check if price dropped X% from highest
            price_drop_pct = ((high_price - current_price) / high_price) * 100
            if price_drop_pct >= pullback_trigger:
                # SAFEGUARD: Never close at a loss via trailing stop
                # If tracking failed and we'd close at loss, wait for recovery or SL
                if pnl_pct < 0:
                    logger.info(f"[TRAILING_STOP_BLOCKED] LONG L{current_tp_level}: Would close at loss ({pnl_pct:.4f}%), waiting for recovery or SL")
                else:
                    logger.info(f"[TRAILING_STOP] LONG L{current_tp_level} triggered: high={high_price}, current={current_price}, drop={price_drop_pct:.4f}% >= {pullback_trigger}%, pnl={pnl_pct:.4f}%")
                    return {
                        "should_close": True,
                        "reason": f"TRAILING_STOP L{current_tp_level}",
                        "peak_pnl": peak_pnl,
                        "trough_pnl": trough_pnl,
                        "tp_level": current_tp_level
                    }
        elif direction == "SHORT" and low_price and low_price > 0:
            # For SHORT: check if price rose X% from lowest
            price_rise_pct = ((current_price - low_price) / low_price) * 100
            if price_rise_pct >= pullback_trigger:
                # SAFEGUARD: Never close at a loss via trailing stop
                # If tracking failed and we'd close at loss, wait for recovery or SL
                if pnl_pct < 0:
                    logger.info(f"[TRAILING_STOP_BLOCKED] SHORT L{current_tp_level}: Would close at loss ({pnl_pct:.4f}%), waiting for recovery or SL")
                else:
                    logger.info(f"[TRAILING_STOP] SHORT L{current_tp_level} triggered: low={low_price}, current={current_price}, rise={price_rise_pct:.4f}% >= {pullback_trigger}%, pnl={pnl_pct:.4f}%")
                    return {
                        "should_close": True,
                        "reason": f"TRAILING_STOP L{current_tp_level}",
                        "peak_pnl": peak_pnl,
                        "trough_pnl": trough_pnl,
                        "tp_level": current_tp_level
                    }
    
    return {"should_close": False, "reason": None, "peak_pnl": peak_pnl, "trough_pnl": trough_pnl}


def calculate_pnl(
    direction: str,
    entry_price: float,
    current_price: float,
    quantity: float,
    leverage: float,
    entry_fee: float = 0,
    exit_fee: float = 0
) -> Dict:
    """
    Calculate P&L for a position
    
    Returns:
        Dictionary with pnl, pnl_percentage, etc.
    """
    if direction == "LONG":
        price_change = current_price - entry_price
    else:  # SHORT
        price_change = entry_price - current_price
    
    # Raw P&L (without fees)
    raw_pnl = price_change * quantity
    
    # P&L after fees
    total_fees = entry_fee + exit_fee
    pnl = raw_pnl - total_fees
    
    # P&L percentage as % of notional (not investment)
    notional = quantity * entry_price
    pnl_percentage = (pnl / notional) * 100 if notional > 0 else 0
    
    # Also keep investment-based for reference
    investment = (quantity * entry_price) / leverage
    pnl_percentage_investment = (pnl / investment) * 100 if investment > 0 else 0
    
    return {
        'pnl': pnl,
        'pnl_percentage': pnl_percentage,
        'pnl_percentage_investment': pnl_percentage_investment,
        'raw_pnl': raw_pnl,
        'total_fees': total_fees
    }
