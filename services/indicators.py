"""
SCALPARS Trading Platform - Technical Indicators
"""
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from typing import Dict, List, Optional, Tuple
import config as config_module
from config import ConfidenceLevel


def calculate_indicators(ohlcv: List, pair_volume_bars: int = 20, global_volume_bars: int = 48) -> Dict:
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
    
    # Calculate ADX (14 period default) — also expose +DI / -DI for directional analysis
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    adx = adx_indicator.adx()
    pos_di = adx_indicator.adx_pos()
    neg_di = adx_indicator.adx_neg()

    # ATR(14) for volatility context (Exploration Analytics, Apr 28)
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    atr = atr_indicator.average_true_range()

    avg_volume = df['volume'].rolling(window=pair_volume_bars).mean()
    avg_volume_global = df['volume'].rolling(window=global_volume_bars).mean()
    
    # Get latest values
    return {
        'price': float(df['close'].iloc[-1]),
        'ema5': float(ema5.iloc[-1]) if not pd.isna(ema5.iloc[-1]) else None,
        'ema5_prev1': float(ema5.iloc[-2]) if len(ema5) >= 2 and not pd.isna(ema5.iloc[-2]) else None,
        'ema5_prev2': float(ema5.iloc[-3]) if len(ema5) >= 3 and not pd.isna(ema5.iloc[-3]) else None,
        'ema5_prev3': float(ema5.iloc[-4]) if len(ema5) >= 4 and not pd.isna(ema5.iloc[-4]) else None,
        'ema8': float(ema8.iloc[-1]) if not pd.isna(ema8.iloc[-1]) else None,
        'ema8_prev1': float(ema8.iloc[-2]) if len(ema8) >= 2 and not pd.isna(ema8.iloc[-2]) else None,
        'ema8_prev2': float(ema8.iloc[-3]) if len(ema8) >= 3 and not pd.isna(ema8.iloc[-3]) else None,
        'ema13': float(ema13.iloc[-1]) if not pd.isna(ema13.iloc[-1]) else None,
        'ema13_prev1': float(ema13.iloc[-2]) if len(ema13) >= 2 and not pd.isna(ema13.iloc[-2]) else None,
        'ema13_prev2': float(ema13.iloc[-3]) if len(ema13) >= 3 and not pd.isna(ema13.iloc[-3]) else None,
        'ema20': float(ema20.iloc[-1]) if not pd.isna(ema20.iloc[-1]) else None,
        'ema20_prev3': float(ema20.iloc[-4]) if len(ema20) >= 4 and not pd.isna(ema20.iloc[-4]) else None,
        'ema50': float(ema50.iloc[-1]) if not pd.isna(ema50.iloc[-1]) else None,
        'ema50_prev12': float(ema50.iloc[-13]) if len(ema50) >= 13 and not pd.isna(ema50.iloc[-13]) else None,
        'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
        'rsi_prev1': float(rsi.iloc[-2]) if len(rsi) >= 2 and not pd.isna(rsi.iloc[-2]) else None,
        'rsi_prev2': float(rsi.iloc[-3]) if len(rsi) >= 3 and not pd.isna(rsi.iloc[-3]) else None,
        'rsi_prev3': float(rsi.iloc[-4]) if len(rsi) >= 4 and not pd.isna(rsi.iloc[-4]) else None,
        # May 15: rsi_prev6 = RSI 6 candles ago (~30min on 5m chart) for sustained momentum analytics
        'rsi_prev6': float(rsi.iloc[-7]) if len(rsi) >= 7 and not pd.isna(rsi.iloc[-7]) else None,
        'adx': float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None,
        'adx_prev1': float(adx.iloc[-2]) if len(adx) >= 2 and not pd.isna(adx.iloc[-2]) else None,
        # Exploration Analytics (Apr 28): +DI / -DI / ATR for next-batch bucket analysis
        'pos_di': float(pos_di.iloc[-1]) if not pd.isna(pos_di.iloc[-1]) else None,
        'neg_di': float(neg_di.iloc[-1]) if not pd.isna(neg_di.iloc[-1]) else None,
        'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
        'volume': float(df['volume'].ewm(span=5, adjust=False).mean().iloc[-1]),
        'avg_volume': float(avg_volume.iloc[-1]) if not pd.isna(avg_volume.iloc[-1]) else None,
        'avg_volume_global': float(avg_volume_global.iloc[-1]) if not pd.isna(avg_volume_global.iloc[-1]) else None,
        'candle_open': float(df['open'].iloc[-1]),
        'candle_high': float(df['high'].iloc[-1]),
        'candle_low': float(df['low'].iloc[-1]),
        'candle_volume_raw': float(df['volume'].iloc[-1]),
        'candle_avg_volume_20': float(df['volume'].rolling(window=20).mean().iloc[-1]) if len(df) >= 20 else None,
        'high_20': float(df['high'].iloc[-20:].max()) if len(df) >= 20 else None,
        'low_20': float(df['low'].iloc[-20:].min()) if len(df) >= 20 else None
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


def _passes_rsi_adx_filter(direction: str, rsi: float, adx: float, th) -> bool:
    """Check RSI x ADX cross-filter rules. Returns True if the entry is allowed.

    Rule formats supported (backward compatible):
      "RSI_LO-RSI_HI:MIN_ADX"          → require ADX >= MIN_ADX (existing)
      "RSI_LO-RSI_HI:MIN_ADX-MAX_ADX"  → require MIN_ADX <= ADX <= MAX_ADX (May 5)

    The range form lets us express "block when ADX > X" by setting MIN low.
    Example: "65-70:0-34" blocks RSI 65-70 entries when ADX > 34.
    """
    key = 'rsi_adx_filter_long' if direction == 'LONG' else 'rsi_adx_filter_short'
    filter_str = getattr(th, key, '')
    if not filter_str or not filter_str.strip():
        return True
    for rule in filter_str.split(','):
        rule = rule.strip()
        if not rule or ':' not in rule:
            continue
        try:
            rsi_part, adx_part = rule.split(':')
            rsi_min, rsi_max = map(float, rsi_part.split('-'))
            if rsi_min <= rsi < rsi_max:
                bounds = adx_part.split('-')
                if len(bounds) == 1:
                    return adx >= float(bounds[0])
                elif len(bounds) == 2:
                    min_adx = float(bounds[0])
                    max_adx = float(bounds[1])
                    return min_adx <= adx <= max_adx
                else:
                    continue
        except (ValueError, TypeError):
            continue
    return True


def gap_expand_marginal(indicators: dict, direction: str):
    """Jun 8: tag for the gap-expanding relaxation A/B.

    Returns True iff this entry would have been BLOCKED by the strict prev1 check
    (current EMA5-EMA13 gap <= prev1) but PASSES the relaxed prev2 check (gap > prev2)
    — i.e. exactly the cohort admitted by `ema_gap_expanding_mode='prev2_only'` that
    the legacy 'both' rule rejected. False = clean expander (passed prev1 too).
    None = insufficient data or undefined. Pure read of the same gaps get_signal uses.
    """
    try:
        e5 = indicators.get('ema5'); e13 = indicators.get('ema13')
        e5p1 = indicators.get('ema5_prev1'); e13p1 = indicators.get('ema13_prev1')
        e5p2 = indicators.get('ema5_prev2'); e13p2 = indicators.get('ema13_prev2')
        if direction == "LONG":
            gap = (e5 - e13) / e13 * 100 if e5 and e13 and e13 > 0 else None
            p1 = (e5p1 - e13p1) / e13p1 * 100 if e5p1 and e13p1 and e13p1 > 0 else None
            p2 = (e5p2 - e13p2) / e13p2 * 100 if e5p2 and e13p2 and e13p2 > 0 else None
        elif direction == "SHORT":
            gap = (e13 - e5) / e5 * 100 if e5 and e13 and e5 > 0 else None
            p1 = (e13p1 - e5p1) / e5p1 * 100 if e5p1 and e13p1 and e5p1 > 0 else None
            p2 = (e13p2 - e5p2) / e5p2 * 100 if e5p2 and e13p2 and e5p2 > 0 else None
        else:
            return None
        if gap is None or p1 is None or p2 is None:
            return None
        return bool(gap > p2 and gap <= p1)
    except Exception:
        return None


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
    ema20_prev3: float = None,
    ema50: float = None,
    ema50_prev12: float = None,
    rsi_prev3: float = None,
    rsi_prev2: float = None,
    ema5_prev1: float = None,
    ema8_prev1: float = None,
    ema5_prev2: float = None,
    ema8_prev2: float = None,
    ema13_prev1: float = None,
    ema13_prev2: float = None,
    adx_prev1: float = None,
    high_20: float = None,
    low_20: float = None,
    block_recorder=None,
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
    
    # Phase B observability (May 6) — records SIGNAL-time silent filter blocks
    # so they appear in the Filter Blocks counter (was previously a black hole;
    # only engine-chain filters were tracked). block_recorder is a callable
    # taking (filter_name, direction). No-op if caller passed None.
    def _record(filter_name: str, direction: str):
        if block_recorder is not None:
            try:
                block_recorder(filter_name, direction)
            except Exception:
                pass  # Never let observability break signal generation

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
        
        # Check EMA5-EMA20 gap requirement (from global thresholds, separated by direction)
        gap_520_enabled = getattr(th, 'ema_gap_5_20_enabled', True)
        if gap_520_enabled and gap is not None:
            if direction == "LONG":
                gap_min = getattr(th, 'ema_gap_5_20_min_long', 0.15)
                gap_max = getattr(th, 'ema_gap_5_20_max_long', 0.8)
                if gap < gap_min:
                    logger.debug(f"LONG {confidence} rejected: gap5-20 {gap:.4f}% < min {gap_min}%")
                    return False
                if gap > gap_max:
                    logger.debug(f"LONG {confidence} rejected: gap5-20 {gap:.4f}% > max {gap_max}% (overextended)")
                    return False
            if direction == "SHORT":
                gap_min = getattr(th, 'ema_gap_5_20_min_short', 0.15)
                gap_max = getattr(th, 'ema_gap_5_20_max_short', 0.8)
                abs_gap = abs(gap)
                if abs_gap < gap_min:
                    logger.debug(f"SHORT {confidence} rejected: |gap5-20| {abs_gap:.4f}% < min {gap_min}%")
                    return False
                if abs_gap > gap_max:
                    logger.debug(f"SHORT {confidence} rejected: |gap5-20| {abs_gap:.4f}% > max {gap_max}% (overextended)")
                    return False
        
        # EMA5 Stretch filter (May 9: moved from per-confidence-level to top-level per-direction
        # min/max in thresholds). Tests |price - ema5| / price * 100.
        # Cross-sample confirmed (May 4 + May 9): LONG stretch <0.16% is a structural loser zone.
        stretch_enabled = getattr(th, 'ema5_stretch_filter_enabled', True)
        if stretch_enabled and price and price > 0:
            stretch_pct = abs(price - ema5) / price * 100
            if direction == "LONG":
                stretch_min = getattr(th, 'ema5_stretch_min_long', 0)
                stretch_max = getattr(th, 'ema5_stretch_max_long', 0)
            else:
                stretch_min = getattr(th, 'ema5_stretch_min_short', 0)
                stretch_max = getattr(th, 'ema5_stretch_max_short', 0)
            if stretch_min and stretch_min > 0 and stretch_pct < stretch_min:
                logger.debug(f"{direction} {confidence} rejected: EMA5 stretch {stretch_pct:.4f}% < min {stretch_min}%")
                return False
            if stretch_max and stretch_max > 0 and stretch_pct > stretch_max:
                logger.debug(f"{direction} {confidence} rejected: EMA5 stretch {stretch_pct:.4f}% > max {stretch_max}%")
                return False

        return True
    
    # --- Macro trend regime (EMA20 slope) ---
    macro_filter_enabled = getattr(th, 'macro_trend_filter_enabled', True)
    neutral_mode = getattr(th, 'macro_trend_neutral_mode', 'both')
    flat_threshold_long = getattr(th, 'macro_trend_flat_threshold_long',
                                  getattr(th, 'macro_trend_flat_threshold', 0.02))
    flat_threshold_short = getattr(th, 'macro_trend_flat_threshold_short',
                                   getattr(th, 'macro_trend_flat_threshold', 0.02))
    regime_for_long = determine_macro_regime(ema20, ema20_prev3, flat_threshold_long)
    regime_for_short = determine_macro_regime(ema20, ema20_prev3, flat_threshold_short)

    def regime_allows(direction: str) -> bool:
        if not macro_filter_enabled:
            return True
        regime = regime_for_long if direction == "LONG" else regime_for_short
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
    ema20_slope_min_long = getattr(th, 'momentum_ema20_slope_min_long', 0.0)
    ema20_slope_min_short = getattr(th, 'momentum_ema20_slope_min_short', 0.0)
    # May 12: Range Position filter — price position in 20-candle high-low range
    rp_min_short = getattr(th, 'range_position_min_short', 0.0)
    rp_max_long = getattr(th, 'range_position_max_long', 100.0)
    range_position = None
    if price is not None and high_20 is not None and low_20 is not None and high_20 != low_20:
        range_position = (price - low_20) / (high_20 - low_20) * 100
    # May 10: ADX delta minimum filter (current ADX − ADX 1 candle ago).
    # Independent per direction; 0 = disabled.
    min_adx_delta_long = getattr(th, 'min_adx_delta_long', 0.0)
    min_adx_delta_short = getattr(th, 'min_adx_delta_short', 0.0)
    adx_delta = (adx - adx_prev1) if (adx is not None and adx_prev1 is not None) else None
    # May 22: Entry Distance from EMA13 minimum (Pair Extension floor). LONG-side only.
    # Block LONG entries with pair_ext < min — bottom-of-pullback bounce-buying NP zone.
    _ext_enabled = getattr(th, 'entry_dist_from_ema13_filter_enabled', True)
    entry_dist_ema13_min_long = getattr(th, 'entry_dist_from_ema13_min_long', 0.0) if _ext_enabled else 0.0
    entry_dist_ema13_min_short = getattr(th, 'entry_dist_from_ema13_min_short', 0.0) if _ext_enabled else 0.0
    pair_ext_pct = None
    if price is not None and ema13 is not None and ema13 != 0:
        pair_ext_pct = (price - ema13) / ema13 * 100
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
    # Jun 8: 'both' (legacy) blocks unless gap beats prev1 AND prev2; 'prev2_only' drops the
    # prev1 check (tolerates a 1-candle pause). prev1 branch active only in 'both' mode.
    _gap_expand_mode = getattr(th, 'ema_gap_expanding_mode', 'both')
    _gap_prev1_active = gap_expanding_enabled and _gap_expand_mode != 'prev2_only'

    if ema8 and ema8 > 0:
        if ema5 > ema8:
            if not regime_allows("LONG"):
                logger.debug(f"[MOMENTUM] LONG skipped: regime={regime_for_long}, ema20={ema20}, ema20_prev3={ema20_prev3}")
                _record("PAIR_REGIME", "LONG")
            elif ema20_filter_long and (price is None or price <= ema20):
                logger.debug(f"[MOMENTUM] LONG skipped: EMA20 filter active, price={price}, ema20={ema20}")
                _record("PAIR_EMA20_FILTER", "LONG")
            elif ema20_slope_long and (ema20_prev3 is None or ema20 <= ema20_prev3):
                logger.debug(f"[MOMENTUM] LONG skipped: EMA20 slope filter active, ema20={ema20}, ema20_prev3={ema20_prev3}")
                _record("PAIR_EMA20_SLOPE", "LONG")
            elif ema20_slope_min_long > 0 and ema20_prev3 and ema20_prev3 != 0 and abs((ema20 - ema20_prev3) / ema20_prev3 * 100) < ema20_slope_min_long:
                logger.debug(f"[MOMENTUM] LONG skipped: EMA20 slope {abs((ema20 - ema20_prev3) / ema20_prev3 * 100):.4f}% < min {ema20_slope_min_long}%")
                _record("PAIR_EMA20_SLOPE_MIN", "LONG")
            elif rp_max_long < 100.0 and range_position is not None and range_position > rp_max_long:
                logger.debug(f"[MOMENTUM] LONG skipped: Range Position {range_position:.2f}% > max {rp_max_long}% (chasing top of range)")
                _record("PAIR_RANGE_POSITION_MAX", "LONG")
            elif min_adx_delta_long > 0 and adx_delta is not None and adx_delta < min_adx_delta_long:
                logger.debug(f"[MOMENTUM] LONG skipped: ADX delta {adx_delta:.4f} < min {min_adx_delta_long}")
                _record("PAIR_ADX_DELTA_MIN", "LONG")
            elif entry_dist_ema13_min_long > 0 and pair_ext_pct is not None and pair_ext_pct < entry_dist_ema13_min_long:
                logger.debug(f"[MOMENTUM] LONG skipped: pair_ext {pair_ext_pct:.4f}% < min {entry_dist_ema13_min_long}% (bottom-of-pullback NP zone)")
                _record("PAIR_EXT_MIN", "LONG")
            elif rsi_momentum_enabled and rsi is not None and rsi_prev2 is not None and rsi < rsi_prev2:
                logger.debug(f"[MOMENTUM] LONG skipped: RSI falling ({rsi_prev2:.1f} -> {rsi:.1f}), momentum against LONG (2 candles)")
                _record("PAIR_RSI_MOMENTUM", "LONG")
            elif long_rsi_min > 0 and rsi is not None and rsi < long_rsi_min:
                logger.debug(f"[MOMENTUM] LONG skipped: RSI {rsi:.1f} < min {long_rsi_min}")
                _record("PAIR_RSI_RANGE", "LONG")
            elif long_rsi_max < 100 and rsi is not None and rsi > long_rsi_max:
                logger.debug(f"[MOMENTUM] LONG skipped: RSI {rsi:.1f} > max {long_rsi_max}")
                _record("PAIR_RSI_RANGE", "LONG")
            elif adx_max_long < 100 and adx > adx_max_long:
                logger.debug(f"[MOMENTUM] LONG skipped: ADX {adx:.1f} > max_long {adx_max_long}")
                _record("PAIR_ADX_MAX", "LONG")
            else:
                ema_gap_pct = ((ema5 - ema8) / ema8) * 100
                # May 8: gap-expanding check switched from EMA5-EMA8 to EMA5-EMA13
                # for less restrictiveness in trending markets. EMA8 lags EMA5 by
                # only 3 periods, so the 5-8 gap micro-oscillates on every minor
                # pullback even within a healthy trend, blocking legitimate
                # trend-continuation entries. EMA13 lags by 8 periods — gap stays
                # monotonically expanding for the full duration of a sustained
                # trend, only compressing when momentum genuinely fades.
                exp_gap_pct = ((ema5 - ema13) / ema13) * 100 if ema5 and ema13 and ema13 > 0 else None
                exp_prev_gap_pct = ((ema5_prev1 - ema13_prev1) / ema13_prev1) * 100 if ema5_prev1 and ema13_prev1 and ema13_prev1 > 0 else None
                exp_prev2_gap_pct = ((ema5_prev2 - ema13_prev2) / ema13_prev2) * 100 if ema5_prev2 and ema13_prev2 and ema13_prev2 > 0 else None
                # May 2: per-direction EMA5-EMA8 max with auto-fallback to legacy single field.
                ema_gap_max = getattr(th, 'ema_gap_5_8_max_long', 0) or getattr(th, 'ema_gap_5_8_max', 0)
                long_gap_min = getattr(th, 'ema_gap_threshold_long', th.ema_gap_threshold)
                gap_threshold_met = ema_gap_pct >= long_gap_min
                if _gap_prev1_active and exp_gap_pct is not None and exp_prev_gap_pct is not None and exp_gap_pct <= exp_prev_gap_pct:
                    logger.debug(f"[MOMENTUM] LONG skipped: EMA5-13 gap compressing vs 1 candle ({exp_prev_gap_pct:.4f}% -> {exp_gap_pct:.4f}%)")
                    _record("PAIR_EMA_GAP_NOT_EXPANDING", "LONG")
                elif gap_expanding_enabled and exp_gap_pct is not None and exp_prev2_gap_pct is not None and exp_gap_pct <= exp_prev2_gap_pct:
                    logger.debug(f"[MOMENTUM] LONG skipped: EMA5-13 gap compressing vs 2 candles ({exp_prev2_gap_pct:.4f}% -> {exp_gap_pct:.4f}%)")
                    _record("PAIR_EMA_GAP_NOT_EXPANDING", "LONG")
                elif ema_gap_max > 0 and ema_gap_pct > ema_gap_max:
                    logger.debug(f"[MOMENTUM] LONG skipped: EMA5-8 gap {ema_gap_pct:.4f}% > max {ema_gap_max}")
                    _record("PAIR_EMA_GAP_MAX", "LONG")
                elif not gap_threshold_met:
                    _record("PAIR_EMA_GAP_MIN", "LONG")
                elif not _passes_rsi_adx_filter("LONG", rsi, adx, th):
                    logger.debug(f"[MOMENTUM] LONG skipped: RSI x ADX cross-filter (RSI={rsi:.1f}, ADX={adx:.1f})")
                    _record("PAIR_RSI_ADX_CROSS", "LONG")
                else:
                    if adx > adx_vs_long:
                        if check_gap_and_mode("LONG", "VERY_STRONG"):
                            logger.info(f"[MOMENTUM] LONG VERY_STRONG: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime_for_long}, ema20_slope={'up' if ema20_prev3 and ema20 > ema20_prev3 else 'n/a'}")
                            return "LONG", "VERY_STRONG"
                    if adx > adx_s_long and adx <= adx_vs_long:
                        if check_gap_and_mode("LONG", "STRONG_BUY"):
                            logger.info(f"[MOMENTUM] LONG STRONG_BUY: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime_for_long}, ema20_slope={'up' if ema20_prev3 and ema20 > ema20_prev3 else 'n/a'}")
                            return "LONG", "STRONG_BUY"
        elif ema5 < ema8 and ema5 > 0:
            if not regime_allows("SHORT"):
                logger.debug(f"[MOMENTUM] SHORT skipped: regime={regime_for_short}, ema20={ema20}, ema20_prev3={ema20_prev3}")
                _record("PAIR_REGIME", "SHORT")
            elif ema20_filter_short and (price is None or price >= ema20):
                logger.debug(f"[MOMENTUM] SHORT skipped: EMA20 filter active, price={price}, ema20={ema20}")
                _record("PAIR_EMA20_FILTER", "SHORT")
            elif ema20_slope_short and (ema20_prev3 is None or ema20 >= ema20_prev3):
                logger.debug(f"[MOMENTUM] SHORT skipped: EMA20 slope filter active, ema20={ema20}, ema20_prev3={ema20_prev3}")
                _record("PAIR_EMA20_SLOPE", "SHORT")
            elif ema20_slope_min_short > 0 and ema20_prev3 and ema20_prev3 != 0 and abs((ema20 - ema20_prev3) / ema20_prev3 * 100) < ema20_slope_min_short:
                logger.debug(f"[MOMENTUM] SHORT skipped: EMA20 slope {abs((ema20 - ema20_prev3) / ema20_prev3 * 100):.4f}% < min {ema20_slope_min_short}%")
                _record("PAIR_EMA20_SLOPE_MIN", "SHORT")
            elif rp_min_short > 0 and range_position is not None and range_position < rp_min_short:
                logger.debug(f"[MOMENTUM] SHORT skipped: Range Position {range_position:.2f}% < min {rp_min_short}% (pile-on at bottom of range)")
                _record("PAIR_RANGE_POSITION_MIN", "SHORT")
            elif min_adx_delta_short > 0 and adx_delta is not None and adx_delta < min_adx_delta_short:
                logger.debug(f"[MOMENTUM] SHORT skipped: ADX delta {adx_delta:.4f} < min {min_adx_delta_short}")
                _record("PAIR_ADX_DELTA_MIN", "SHORT")
            elif entry_dist_ema13_min_short > 0 and pair_ext_pct is not None and abs(pair_ext_pct) < entry_dist_ema13_min_short:
                logger.debug(f"[MOMENTUM] SHORT skipped: |pair_ext| {abs(pair_ext_pct):.4f}% < min {entry_dist_ema13_min_short}% (bottom-of-pullback NP zone, SHORT mirror)")
                _record("PAIR_EXT_MIN", "SHORT")
            elif rsi_momentum_enabled and rsi is not None and rsi_prev2 is not None and rsi > rsi_prev2:
                logger.debug(f"[MOMENTUM] SHORT skipped: RSI rising ({rsi_prev2:.1f} -> {rsi:.1f}), momentum against SHORT (2 candles)")
                _record("PAIR_RSI_MOMENTUM", "SHORT")
            elif short_rsi_max < 100 and rsi is not None and rsi > short_rsi_max:
                logger.debug(f"[MOMENTUM] SHORT skipped: RSI {rsi:.1f} > max {short_rsi_max}")
                _record("PAIR_RSI_RANGE", "SHORT")
            elif short_rsi_min > 0 and rsi is not None and rsi < short_rsi_min:
                logger.debug(f"[MOMENTUM] SHORT skipped: RSI {rsi:.1f} < min {short_rsi_min} (oversold)")
                _record("PAIR_RSI_RANGE", "SHORT")
            elif adx_max < 100 and adx > adx_max:
                logger.debug(f"[MOMENTUM] SHORT skipped: ADX {adx:.1f} > max {adx_max}")
                _record("PAIR_ADX_MAX", "SHORT")
            else:
                ema_gap_pct = ((ema8 - ema5) / ema5) * 100
                # May 8: gap-expanding check switched from EMA5-EMA8 to EMA5-EMA13
                # for less restrictiveness in trending markets (see LONG comment).
                exp_gap_pct = ((ema13 - ema5) / ema5) * 100 if ema5 and ema13 and ema5 > 0 else None
                exp_prev_gap_pct = ((ema13_prev1 - ema5_prev1) / ema5_prev1) * 100 if ema5_prev1 and ema13_prev1 and ema5_prev1 > 0 else None
                exp_prev2_gap_pct = ((ema13_prev2 - ema5_prev2) / ema5_prev2) * 100 if ema5_prev2 and ema13_prev2 and ema5_prev2 > 0 else None
                # May 2: per-direction EMA5-EMA8 max with auto-fallback to legacy single field.
                ema_gap_max = getattr(th, 'ema_gap_5_8_max_short', 0) or getattr(th, 'ema_gap_5_8_max', 0)
                short_gap_min = getattr(th, 'ema_gap_threshold_short', th.ema_gap_threshold)
                gap_threshold_met = ema_gap_pct >= short_gap_min
                if _gap_prev1_active and exp_gap_pct is not None and exp_prev_gap_pct is not None and exp_gap_pct <= exp_prev_gap_pct:
                    logger.debug(f"[MOMENTUM] SHORT skipped: EMA5-13 gap compressing vs 1 candle ({exp_prev_gap_pct:.4f}% -> {exp_gap_pct:.4f}%)")
                    _record("PAIR_EMA_GAP_NOT_EXPANDING", "SHORT")
                elif gap_expanding_enabled and exp_gap_pct is not None and exp_prev2_gap_pct is not None and exp_gap_pct <= exp_prev2_gap_pct:
                    logger.debug(f"[MOMENTUM] SHORT skipped: EMA5-13 gap compressing vs 2 candles ({exp_prev2_gap_pct:.4f}% -> {exp_gap_pct:.4f}%)")
                    _record("PAIR_EMA_GAP_NOT_EXPANDING", "SHORT")
                elif ema_gap_max > 0 and ema_gap_pct > ema_gap_max:
                    logger.debug(f"[MOMENTUM] SHORT skipped: EMA5-8 gap {ema_gap_pct:.4f}% > max {ema_gap_max}")
                    _record("PAIR_EMA_GAP_MAX", "SHORT")
                elif not gap_threshold_met:
                    _record("PAIR_EMA_GAP_MIN", "SHORT")
                elif not _passes_rsi_adx_filter("SHORT", rsi, adx, th):
                    logger.debug(f"[MOMENTUM] SHORT skipped: RSI x ADX cross-filter (RSI={rsi:.1f}, ADX={adx:.1f})")
                    _record("PAIR_RSI_ADX_CROSS", "SHORT")
                else:
                    if adx > th.adx_very_strong:
                        if check_gap_and_mode("SHORT", "VERY_STRONG"):
                            logger.info(f"[MOMENTUM] SHORT VERY_STRONG: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime_for_short}, ema20_slope={'down' if ema20_prev3 and ema20 < ema20_prev3 else 'n/a'}")
                            return "SHORT", "VERY_STRONG"
                    if adx > th.adx_strong and adx <= th.adx_very_strong:
                        if check_gap_and_mode("SHORT", "STRONG_BUY"):
                            logger.info(f"[MOMENTUM] SHORT STRONG_BUY: ema_gap={ema_gap_pct:.4f}%, ADX={adx:.1f}, RSI={rsi:.1f}, regime={regime_for_short}, ema20_slope={'down' if ema20_prev3 and ema20 < ema20_prev3 else 'n/a'}")
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
    tp_trailing_enabled: bool = True,
    entry_atr_pct: float = None,  # May 7 Phase 1: ATR-normalized trailing
    current_stretch: float = None,  # Jun 1: live |price−EMA5| stretch % (runner trail)
    peak_stretch: float = None,     # Jun 1: peak stretch since entry (runner trail)
    is_flip: bool = False,          # Jun 14: Flip Entry — disable runner-trail for flips
                                    # (it's a continuation-ride; a flip is a reversion) →
                                    # flips fall back to the normal tiered trailing.
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
    # May 7: tier-aware widening — effective pullback grows with TP level so
    # bigger winners get more room. Reads from global thresholds; default 0.0
    # = current flat behavior. Effective = base + widening × (level - 1).
    try:
        _widening_per_level = float(getattr(config_module.trading_config.thresholds, 'pullback_widening_per_level', 0.0) or 0.0)
    except Exception:
        _widening_per_level = 0.0

    # May 7 Phase 2: early-arm zone. When peak is between trailing_early_arm_threshold
    # and tp_min, use the (typically tighter) early_arm_pullback instead of the regular
    # base pullback + widening. Default arms at peak ≥ +0.30% with 0.10% pullback —
    # locks in moderate-momentum profits that would otherwise reverse to losses.
    try:
        _early_arm_threshold = float(getattr(config_module.trading_config.thresholds, 'trailing_early_arm_threshold', 0.0) or 0.0)
        _early_arm_pullback = float(getattr(config_module.trading_config.thresholds, 'trailing_early_arm_pullback', 0.10) or 0.10)
    except Exception:
        _early_arm_threshold = 0.0
        _early_arm_pullback = 0.10
    _in_early_arm = (
        _early_arm_threshold > 0
        and peak_pnl >= _early_arm_threshold
        and peak_pnl < (tp_min - 0.005)
        and (current_tp_level or 1) <= 1
    )
    if _in_early_arm:
        # Override: use tight early-arm pullback regardless of widening
        pullback_trigger = _early_arm_pullback
    else:
        # Standard tier-aware widening
        _level_minus_one = max(0, (current_tp_level or 1) - 1)
        pullback_trigger = pullback_trigger + _widening_per_level * _level_minus_one

    # May 7 Phase 1: ATR-normalized pullback floor. Volatile pairs need wider
    # pullback to avoid triggering on normal candle noise. Floor = entry_atr_pct
    # × trailing_atr_multiplier. Default 0.50 = "half a candle of noise". Set
    # multiplier to 0.0 to disable ATR floor entirely.
    try:
        _atr_multiplier = float(getattr(config_module.trading_config.thresholds, 'trailing_atr_multiplier', 0.0) or 0.0)
    except Exception:
        _atr_multiplier = 0.0
    if _atr_multiplier > 0 and entry_atr_pct is not None and entry_atr_pct > 0:
        _atr_floor = entry_atr_pct * _atr_multiplier
        if _atr_floor > pullback_trigger:
            pullback_trigger = _atr_floor
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
    # 0.005pp tolerance (May 6 — bug fix): float-rounding around the tp_min threshold
    # could leave trailing perpetually unarmed when peak is e.g. +0.4998% vs tp_min 0.50%.
    # May 7 Phase 2: ALSO activates in the early-arm zone (peak between early_arm_threshold
    # and tp_min) with tight pullback — locks in moderate-momentum gains.
    trailing_stop_active = (
        peak_pnl >= (effective_tp_target - 0.005)
        or current_tp_level >= 2
        or _in_early_arm
    )
    
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

    # May 22: ATR-adjusted SL widening for high-volatility pairs.
    # Same mechanism as services/trading_engine.py realtime path.
    # Only WIDENS (more negative); never tightens. Skipped under BE.
    if not breakeven_active:
        try:
            _sl_atr_mult = float(getattr(config_module.trading_config.thresholds, 'sl_atr_multiplier', 0.0) or 0.0)
        except Exception:
            _sl_atr_mult = 0.0
        if _sl_atr_mult > 0 and entry_atr_pct is not None and entry_atr_pct > 0:
            _atr_sl = -(entry_atr_pct * _sl_atr_mult)
            if _atr_sl < effective_stop_loss:  # more negative = wider
                logger.debug(f"[ATR_SL_WIDEN] {direction}: SL widened from {effective_stop_loss}% to {_atr_sl}% (ATR {entry_atr_pct}% × {_sl_atr_mult})")
                effective_stop_loss = _atr_sl
        # May 23: cap ATR widening at floor. Prevents extreme-ATR pairs
        # (e.g., ATR 2.3% → -3.47% SL) from effectively disabling the SL.
        try:
            _sl_floor = float(getattr(config_module.trading_config.thresholds, 'sl_atr_widen_floor_pct', 0.0) or 0.0)
        except Exception:
            _sl_floor = 0.0
        if _sl_floor < 0 and effective_stop_loss < _sl_floor:
            logger.debug(f"[ATR_SL_FLOOR] {direction}: SL clamped from {effective_stop_loss}% to {_sl_floor}% (floor cap)")
            effective_stop_loss = _sl_floor

    # Check Stop Loss (P&L based, with break-even adjustment in pre-TP zone only)
    if pnl_pct <= effective_stop_loss:
        if breakeven_active:
            close_reason = f"BREAKEVEN_EXIT_L{be_level}"
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

    # Jun 1: is this trade in the runner stretch-trail regime? If so, the
    # trend-break exit below must DEFER to the stretch-trail (fires lower down),
    # matching the validated shadow strpk (which respects EMA13 cross + hard SL,
    # NOT the ema5/8/13/20 stack break). Otherwise a minor stack flip while above
    # target would exit the runner early and defeat the whole handoff.
    _runner_armed = False
    try:
        from config import trading_config as _rtc0
        # Jun 12: per-direction params. SHORT side ships with NO ATR gate
        # (atr_min=0) + arm 0.45 — must match the measured shadow strpk policy.
        if direction == "LONG":
            _ra_en = getattr(_rtc0.thresholds, 'runner_trail_enabled', False)
            _ra_amin = float(getattr(_rtc0.thresholds, 'runner_trail_atr_min', 1.0) or 0.0)
            _ra_arm = float(getattr(_rtc0.thresholds, 'runner_trail_arm_peak', 0.70) or 0.70)
        else:
            _ra_en = getattr(_rtc0.thresholds, 'runner_trail_short_enabled', False)
            _ra_amin = float(getattr(_rtc0.thresholds, 'runner_trail_short_atr_min', 0.0) or 0.0)
            _ra_arm = float(getattr(_rtc0.thresholds, 'runner_trail_short_arm_peak', 0.45) or 0.45)
        _runner_armed = (_ra_en and not is_flip and peak_pnl >= _ra_arm
                         and (_ra_amin <= 0
                              or (entry_atr_pct is not None and entry_atr_pct >= _ra_amin)))
    except Exception:
        _runner_armed = False

    # Check if we've reached the current TP target
    if pnl_pct >= effective_tp_target and not _runner_armed:

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
    
    # Phase 1d-ExitTest (May 2): RSI handoff disables trailing stop past
    # `rsi_handoff_level`. The actual RSI 2-drop exit fires in the live monitor
    # loop where pair_data is available; this block just suppresses the trailing
    # check so it doesn't pre-empt the handoff. Default OFF — feature inert.
    # May 6: same suppression for EMA Stack Cross Exit when active past its level.
    _handoff_suppress_trailing = False
    try:
        from config import trading_config as _tc
        if getattr(_tc.thresholds, 'rsi_handoff_active', False):
            _handoff_level = getattr(_tc.thresholds, 'rsi_handoff_level', 3)
            if (current_tp_level or 1) >= _handoff_level:
                _handoff_suppress_trailing = True
                logger.info(f"[RSI_HANDOFF] {direction} L{current_tp_level}: trailing suppressed (handoff_level={_handoff_level}, RSI exit will handle)")
        if not _handoff_suppress_trailing and getattr(_tc.thresholds, 'ema_stack_cross_exit_enabled', False):
            _es_level = getattr(_tc.thresholds, 'ema_stack_cross_exit_level', 2)
            if (current_tp_level or 1) >= _es_level:
                _handoff_suppress_trailing = True
                logger.info(f"[EMA_STACK_HANDOFF] {direction} L{current_tp_level}: trailing suppressed (level={_es_level}, EMA Stack Cross will handle)")
    except Exception:
        # Fail safe: if config read fails, behave as before (handoff inactive)
        pass

    # Jun 1, 2026 — RUNNER STRETCH-TRAIL handoff (scoped high-ATR LONG runner exit).
    # When a high-ATR LONG proves a runner (peak ≥ arm_peak), SUPPRESS the tight
    # price-trailing and instead exit only when live stretch collapses to
    # runner_trail_k × peak stretch. Lets IDU-class runners run. One-way: once
    # armed it stays in stretch-trail mode. Backstops (hard SL, EMA13) still fire.
    try:
        from config import trading_config as _rtc
        # Jun 12: direction-aware. LONG keeps its (currently OFF) Jun-1 config;
        # SHORT ships ON (no ATR gate, arm 0.45, K=0.5 — the measured strpk).
        if direction == "LONG":
            _rh_en = getattr(_rtc.thresholds, 'runner_trail_enabled', False)
            _rh_amin = float(getattr(_rtc.thresholds, 'runner_trail_atr_min', 1.0) or 0.0)
            _rh_arm = float(getattr(_rtc.thresholds, 'runner_trail_arm_peak', 0.70) or 0.70)
            _rt_k = float(getattr(_rtc.thresholds, 'runner_trail_k', 0.5) or 0.5)
        else:
            _rh_en = getattr(_rtc.thresholds, 'runner_trail_short_enabled', False)
            _rh_amin = float(getattr(_rtc.thresholds, 'runner_trail_short_atr_min', 0.0) or 0.0)
            _rh_arm = float(getattr(_rtc.thresholds, 'runner_trail_short_arm_peak', 0.45) or 0.45)
            _rt_k = float(getattr(_rtc.thresholds, 'runner_trail_short_k', 0.5) or 0.5)
        if (_rh_en and not is_flip and peak_pnl >= _rh_arm
                and (_rh_amin <= 0
                     or (entry_atr_pct is not None and entry_atr_pct >= _rh_amin))):
            _handoff_suppress_trailing = True  # take over the profit-taking side
            # SIGNED stretch (matches the validated shadow strpk): fires when the
            # favorable extension retraces to ≤ k× its peak — INCLUDING when price
            # crosses back below EMA5 (signed goes negative). Do NOT use abs() here
            # or it re-introduces the unsigned fader-ride bug (CLAUDE.md May 31).
            if (current_stretch is not None and peak_stretch is not None
                    and peak_stretch > 0 and current_stretch <= _rt_k * peak_stretch):
                logger.info(f"[RUNNER_TRAIL] {direction} L{current_tp_level}: stretch {current_stretch:.3f}% <= "
                            f"{_rt_k}× peak {peak_stretch:.3f}% — runner banked, pnl={pnl_pct:.4f}%, peak={peak_pnl:.4f}%")
                return {
                    "should_close": True,
                    "reason": f"RUNNER_TRAIL L{current_tp_level}",
                    "peak_pnl": peak_pnl,
                    "trough_pnl": trough_pnl,
                    "tp_level": current_tp_level
                }
    except Exception:
        # Fail safe: never block exits on a runner-trail config/compute error
        pass

    # Jun 8: trailing min-profit GATE. Suppress the (price-drop) trailing stop when its
    # exit level (peak_pnl − pullback) is below the configured min — i.e. when it would
    # realize a loss/sub-min exit. The trade then rides on the hard SL until the peak
    # climbs enough to lock >= the min, at which point this re-engages and trails normally.
    try:
        _trail_min_profit = float(getattr(config_module.trading_config.thresholds, 'trailing_min_profit_to_fire', -99.0))
    except Exception:
        _trail_min_profit = -99.0
    _trail_level_ok = (peak_pnl - pullback_trigger) >= _trail_min_profit
    _trail_suppressed_pnl = None  # set to the would-have-cut pnl% if the gate blocks a trailing fire this tick
    if trailing_stop_active and not _handoff_suppress_trailing:
        if direction == "LONG" and high_price and high_price > 0:
            # For LONG: check if price dropped X% from highest
            price_drop_pct = ((high_price - current_price) / high_price) * 100
            if price_drop_pct >= pullback_trigger and _trail_level_ok:
                # May 6 — bug fix: removed `if pnl_pct < 0` safeguard.  The original
                # intent was to prevent corrupted high_price tracking from forcing a
                # bad close, but it conflated "current pnl negative" with "tracking
                # broken" — two different things.  In fast reversals, the realtime
                # callback can land on a tick where pnl already crossed zero even
                # though peak was legitimately reached and trailing should fire.
                # Hard SL / FL_EMERGENCY_SL still cover the truly-corrupted cases.
                logger.info(f"[TRAILING_STOP] LONG L{current_tp_level} triggered: high={high_price}, current={current_price}, drop={price_drop_pct:.4f}% >= {pullback_trigger}%, pnl={pnl_pct:.4f}%")
                return {
                    "should_close": True,
                    "reason": f"TRAILING_STOP L{current_tp_level}",
                    "peak_pnl": peak_pnl,
                    "trough_pnl": trough_pnl,
                    "tp_level": current_tp_level
                }
            elif price_drop_pct >= pullback_trigger and not _trail_level_ok:
                # Gate suppressed this fire — record the would-have-cut pnl (phantom CF).
                _trail_suppressed_pnl = pnl_pct
        elif direction == "SHORT" and low_price and low_price > 0:
            # For SHORT: check if price rose X% from lowest
            price_rise_pct = ((current_price - low_price) / low_price) * 100
            if price_rise_pct >= pullback_trigger and _trail_level_ok:
                # May 6 — bug fix: removed `if pnl_pct < 0` safeguard (see LONG branch comment).
                logger.info(f"[TRAILING_STOP] SHORT L{current_tp_level} triggered: low={low_price}, current={current_price}, rise={price_rise_pct:.4f}% >= {pullback_trigger}%, pnl={pnl_pct:.4f}%")
                return {
                    "should_close": True,
                    "reason": f"TRAILING_STOP L{current_tp_level}",
                    "peak_pnl": peak_pnl,
                    "trough_pnl": trough_pnl,
                    "tp_level": current_tp_level
                }
            elif price_rise_pct >= pullback_trigger and not _trail_level_ok:
                _trail_suppressed_pnl = pnl_pct

    return {"should_close": False, "reason": None, "peak_pnl": peak_pnl, "trough_pnl": trough_pnl, "trail_suppressed_pnl": _trail_suppressed_pnl}


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
