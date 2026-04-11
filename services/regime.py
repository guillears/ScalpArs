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
    # Legacy single CHOPPY label — kept for historical rows that were tagged
    # before the sub-split was introduced.  New trades never use this label.
    "CHOPPY",
    # Choppy sub-labels (new, split from the old single CHOPPY bucket):
    #   - CHOPPY_WEAK: ADX < 18.  Dead market, low trend strength, noise-dominated.
    #   - CHOPPY_FLAT: ADX >= 18 but |slope| < 0.02%.  "Coiled" market, often
    #     precedes a breakout.  ADX says something is brewing but direction is
    #     still ambiguous.
    "CHOPPY_WEAK",
    "CHOPPY_FLAT",
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

    # CHOPPY split into two sub-buckets (see REGIME_LABELS for rationale).
    # Low-ADX case is checked first because it's the stronger filter — a dead
    # market with ADX < 18 is "weak" regardless of slope.
    if btc_adx < 18:
        return "CHOPPY_WEAK"
    if abs(btc_slope_pct) < 0.02:
        return "CHOPPY_FLAT"

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
