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
    if not ohlcv or len(ohlcv) < 21:  # Need at least 21 candles for EMA20
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
        'ema8': float(ema8.iloc[-1]) if not pd.isna(ema8.iloc[-1]) else None,
        'ema13': float(ema13.iloc[-1]) if not pd.isna(ema13.iloc[-1]) else None,
        'ema20': float(ema20.iloc[-1]) if not pd.isna(ema20.iloc[-1]) else None,
        'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        'adx': float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None,
        'volume': float(df['volume'].iloc[-1]),
        'avg_volume': float(avg_volume.iloc[-1]) if not pd.isna(avg_volume.iloc[-1]) else None
    }


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
    config: Optional[Dict] = None
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
        
        return True
    
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
    dynamic_tp_target: float = None
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
        return {"should_close": False, "reason": None, "peak_pnl": peak_pnl}
    
    # Calculate current P&L percentage INCLUDING FEES (on margin/investment)
    if direction == "LONG":
        raw_pnl = (current_price - entry_price) * quantity
    else:  # SHORT
        raw_pnl = (entry_price - current_price) * quantity
    
    # Calculate estimated exit fee
    current_notional = current_price * quantity
    estimated_exit_fee = current_notional * tc.trading_fee
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
    breakeven_trigger = conf_config.breakeven_trigger
    breakeven_offset = conf_config.breakeven_offset
    
    # Use dynamic_tp_target if set, otherwise use tp_min
    effective_tp_target = dynamic_tp_target if dynamic_tp_target is not None else tp_min
    
    # ALWAYS track maximum P&L seen BEFORE stop loss check (so break-even uses latest peak)
    # This ensures we don't lose the peak when price drops after TP extension
    if pnl_pct > 0:
        peak_pnl = max(peak_pnl, pnl_pct)
    
    # Trailing stop (pullback from peak price) only activates at L2+.
    # At L1, the TP check handles exits: extend if trend valid, close if trend broken.
    trailing_stop_active = current_tp_level >= 2
    
    # Break-even stop loss: applies when trailing stop is NOT active (L1 and pre-TP zone).
    # At L2+ the pullback trailing provides tighter protection so BE is not needed.
    effective_stop_loss = stop_loss
    breakeven_active = False
    if not trailing_stop_active and peak_pnl >= breakeven_trigger:
        breakeven_active = True
        effective_stop_loss = breakeven_offset
        logger.debug(f"[BREAKEVEN] {direction} L{current_tp_level}: Active! peak_pnl={peak_pnl:.4f}% >= trigger={breakeven_trigger}%, SL moved from {stop_loss}% to {breakeven_offset}%")
    
    # Check Stop Loss (P&L based, with break-even adjustment in pre-TP zone only)
    if pnl_pct <= effective_stop_loss:
        if breakeven_active and pnl_pct < 0:
            # Break-even triggered but P&L gapped into negative territory.
            # Don't close at a loss via BE - wait for recovery or original SL.
            logger.info(f"[BREAKEVEN_BLOCKED] {direction} L{current_tp_level}: pnl={pnl_pct:.4f}% gapped below 0, skipping BE close (peak={peak_pnl:.4f}%)")
            if pnl_pct <= stop_loss:
                logger.info(f"[STOP_LOSS] {direction} L{current_tp_level} triggered: pnl_pct={pnl_pct:.4f}% <= stop_loss={stop_loss}% (BE was active but P&L negative)")
                return {
                    "should_close": True,
                    "reason": f"STOP_LOSS L{current_tp_level}",
                    "peak_pnl": peak_pnl,
                    "tp_level": current_tp_level
                }
        else:
            reason_prefix = "BREAKEVEN_SL" if breakeven_active else "STOP_LOSS"
            logger.info(f"[{reason_prefix}] {direction} L{current_tp_level} triggered: pnl_pct={pnl_pct:.4f}% <= effective_sl={effective_stop_loss}% (original_sl={stop_loss}%, peak={peak_pnl:.4f}%, be_trigger={breakeven_trigger}%)")
            return {
                "should_close": True,
                "reason": f"{reason_prefix} L{current_tp_level}",
                "peak_pnl": peak_pnl,
                "tp_level": current_tp_level
            }
    
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
                        "tp_level": current_tp_level
                    }
    
    return {"should_close": False, "reason": None, "peak_pnl": peak_pnl}


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
