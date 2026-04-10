"""BTC regime classifier — Phase 1 (tracking only, no behavior change).

Classifies the current BTC market state into one of 7 regime labels based on
BTC ADX (trend strength), BTC slope (direction), and BTC RSI (position).

The regime is computed at trade entry and at trade exit so we can analyze:
1. Per-regime baseline performance (which regimes does the bot work in?)
2. Regime-shift impact (do trades that span regime changes perform worse?)

Phase 2 will use this data to gate or adapt strategy per regime.
"""
from typing import Optional


REGIME_LABELS = [
    "UNKNOWN",
    "CHOPPY",
    "HEALTHY_BULL",
    "STRONG_BULL",
    "BULL_EXHAUSTED",
    "HEALTHY_BEAR",
    "STRONG_BEAR",
    "BEAR_EXHAUSTED",
]


def classify_btc_regime(
    btc_adx: Optional[float],
    btc_rsi: Optional[float],
    btc_slope_pct: Optional[float],
) -> str:
    """Classify the current BTC state into a single regime label.

    Args:
        btc_adx: BTC ADX(14) value
        btc_rsi: BTC RSI(14) value
        btc_slope_pct: BTC EMA20 slope as percentage (signed)

    Returns:
        One of REGIME_LABELS.
    """
    if btc_adx is None or btc_rsi is None or btc_slope_pct is None:
        return "UNKNOWN"

    # CHOPPY: no trend energy or no clear direction
    if btc_adx < 18:
        return "CHOPPY"
    if abs(btc_slope_pct) < 0.02:
        return "CHOPPY"

    # Trending regimes — direction by slope sign, strength by ADX
    is_bull = btc_slope_pct > 0
    is_strong = btc_adx >= 28

    if is_bull:
        # Exhaustion: very strong trend + overbought
        if is_strong and btc_rsi >= 70:
            return "BULL_EXHAUSTED"
        return "STRONG_BULL" if is_strong else "HEALTHY_BULL"
    else:
        if is_strong and btc_rsi <= 30:
            return "BEAR_EXHAUSTED"
        return "STRONG_BEAR" if is_strong else "HEALTHY_BEAR"
