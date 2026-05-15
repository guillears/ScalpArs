"""Tag → category lookup for Live Terminal event coloring.

Derived from a `grep -nE '\\[[A-Z_]+\\]'` sweep across services/ and main.py
at build time (May 15 2026). Edit this file freely; it does not affect
trading logic. Unknown tags fall back to record.levelname (WARNING+ → WARN,
else INFO).

Categories: SCAN, WATCH, ENTRY, EXIT, REJECT, WARN, INFO
"""

TAG_CATEGORY: dict[str, str] = {
    # SCAN — periodic loop / market scan activity
    "SCAN": "SCAN",
    "BTC_1H_SLOPE": "SCAN",
    "BTC_REGIME": "SCAN",
    "DEBUG_TREND": "SCAN",
    "EMA_GAP_EXPANDING": "SCAN",
    "STARTUP": "SCAN",
    "SHUTDOWN": "SCAN",

    # WATCH — observation, no trading action
    "EMA13_STRICT_FIRST_HOLD": "WATCH",
    "PEAK_INVARIANT_FIX": "WATCH",
    "TROUGH_INVARIANT_FIX": "WATCH",
    "POST_EXIT_TRACKING": "WATCH",
    "PROJECTED_BALANCE": "WATCH",

    # ENTRY — position opening
    "TRADE": "ENTRY",
    "OPEN_POSITION": "ENTRY",
    "MAKER_ENTRY": "ENTRY",
    "MAKER_FILL": "ENTRY",
    "CELL_MULT": "ENTRY",

    # EXIT — position closing
    "CLOSE_COMMITTED": "EXIT",
    "REALTIME_EMA13_CROSS_EXIT": "EXIT",
    "REALTIME_EMA_STACK_CROSS_EXIT": "EXIT",
    "REALTIME_FAST_EXIT": "EXIT",
    "RSI_HANDOFF_EXIT": "EXIT",
    "RSI_MOMENTUM_EXIT": "EXIT",
    "BREAKEVEN_SL": "EXIT",
    "SIGNAL_LOST": "EXIT",
    "TRAILING_STOP": "EXIT",
    "STOP_LOSS_WIDE": "EXIT",
    "FL_DEEP_STOP": "EXIT",
    "FL_REGIME_CHANGE": "EXIT",
    "EMERGENCY_SL": "EXIT",
    "REGIME_CHANGE_EXIT": "EXIT",
    "EXIT_SLIPPAGE": "EXIT",
    "PORTFOLIO_CLOSE": "EXIT",
    "TICK_MOMENTUM_EXIT": "EXIT",
    "NO_EXPANSION": "EXIT",
    "RECOVERED": "EXIT",
    "MANUAL": "EXIT",

    # REJECT — filter blocks and entry rejections
    "PAIR_ADX_DIR": "REJECT",
    "BTC_ADX_GATE": "REJECT",
    "BTC_ADX_GATE_HIGH": "REJECT",
    "BTC_ADX_GATE_LOW": "REJECT",
    "BTC_ADX_DIR": "REJECT",
    "BTC_RSI_ADX_CROSS": "REJECT",
    "BTC_TREND_FILTER": "REJECT",
    "BTC_SLOPE_GATE": "REJECT",
    "BTC_SLOPE_MAX_GATE": "REJECT",
    "PAIR_SLOPE_MAX_GATE": "REJECT",
    "VOL_GATE": "REJECT",
    "VOL_GATE_MAX_SHORT": "REJECT",
    "BREADTH_GATE": "REJECT",
    "SPIKE_GUARD": "REJECT",
    "PAIR_ADX_DELTA_MIN": "REJECT",
    "PAIR_EMA20_SLOPE_MIN": "REJECT",
    "PAIR_EMA20_SLOPE": "REJECT",
    "PAIR_EMA20_FILTER": "REJECT",
    "PAIR_EMA_GAP_MIN": "REJECT",
    "PAIR_EMA_GAP_MAX": "REJECT",
    "PAIR_EMA_GAP_NOT_EXPANDING": "REJECT",
    "PAIR_RSI_RANGE": "REJECT",
    "PAIR_RSI_ADX_CROSS": "REJECT",
    "PAIR_RANGE_POSITION_MIN": "REJECT",
    "PAIR_ADX_MAX": "REJECT",
    "ADX_DELTA_BTC_ADX_CROSS": "REJECT",
    "QUALITY_SCORE_GATE": "REJECT",
    "SIGNAL_EXPIRED": "REJECT",
    "SKIP": "REJECT",
    "COOLDOWN": "REJECT",
    "MOMENTUM": "REJECT",

    # WARN — non-fatal warnings
    "RECONCILE_SKIP": "WARN",
    "RECONCILE_STALE_INTENT": "WARN",
    "CELL_MULT_CAPPED": "WARN",
    "CELL_MULT_CAPPED_HARD": "WARN",
    "BNB_LOW": "WARN",
    "MAKER_FAILED": "WARN",
    "WS_RECONNECT": "WARN",
    "DEBUG_OPENED": "WARN",
    "FL1_WIDE_SL_FIRED": "WARN",

    # INFO — generic informational
    "BOTSTATE": "INFO",
    "BNB_SWAP": "INFO",
    "BNB_PROJECTED": "INFO",
    "BNB_CHECK": "INFO",
    "CONFIG_CHANGE": "INFO",
    "RECONCILE": "INFO",

    # Synthetic
    "HEARTBEAT": "INFO",
    "RAW": "INFO",
}


def category_for(tag: str | None, level_name: str = "INFO") -> str:
    """Return category for a parsed tag, with level-based fallback."""
    if tag and tag in TAG_CATEGORY:
        return TAG_CATEGORY[tag]
    # Fallback by level (logging.WARNING == 30, logging.ERROR == 40, CRITICAL == 50)
    if level_name in ("WARNING", "ERROR", "CRITICAL"):
        return "WARN"
    return "INFO"
