"""🧱 ARMED WATCH-LINE FILTERS (2026-11-11, DECISION_LOG 527 — operator: "ship this filters, and track the block trades"; a DECLARED
OVERRIDE: every line below was in-sample / before its clean checkpoint when armed — only WILLY_GAP had PASSED clean). Pure rules, no I/O:
the ONE definition shared by the engine (services/trading_engine.py), the master-pool builder (scripts/build_master_pool.py), the best case
(scripts/watch_zones.py) and the tests.

Each function takes the values the engine holds BEFORE open_position (the same values the fill would be stamped with) and returns the list of
lines that refuse the entry — [] = no block. A line refuses only when it is switched ON in the config AND every value it needs is readable:
an unreadable input NEVER blocks (fail-open — a block never fires on data it lacks; the scouts list such entries as unscored).
Thresholds are the FROZEN registered cuts (never re-fit — CLAUDE.md); only the on / off toggles live in the config.

Lines (sleeve → refuse when):
  FRENZY_WILLY     WILLY_GAP           coin ADX > 55 OR |EMA8 − EMA13| gap ≤ 0.90 %         (DL 414)
                   WILLY_FAILED_SPIKE  the signal bar closed at / below the spike VWAP       (DL 509)
  FRENZY_WILLY_HT  WILLY_FAILED_SPIKE  the same                                             (DL 511)
  FRENZY_LITE_RETRY RETRY_BTC1H        BTC 1 h EMA20 slope ≤ 0                               (DL 415)
  FRENZY_LITE      LITE_BTC_EMA20_DOWN BTC 5 m EMA20 slope ≤ 0 (the RAW slope — the engine's display copy reads an unreadable slope as 0)  (DL 522)
  FRENZY_LONG      LONG_WEAK_FRENZY    frenzy volume multiple ≤ 119× normal                  (DL 525)
  SPIKE_FADE       OBS-F1 late weak spike  BTC 1 d EMA50 > EMA200 (late tape) ∧ pair +DI ≤ 48.5 ∧ price ≤ 1.80 % above the pair's EMA13  (DL 384)
  MOMENTUM LONG    ML_MKT_HOT          > 50 % of the rolling top-100 up over 4 h (services/mkt_up4h share > 0.50)   (DL 416 / 489)
                   ML_BULL90           ≥ 90 % of the scanned coins bullish (entry_bull_pct)   (DL 523)
"""
import math

WILLY_GAP_ADX_MAX = 55.0
WILLY_GAP_EMA_GAP_MAX = 0.90
FAILED_SPIKE_VWAP_MAX = 0.0
RETRY_BTC1H_MAX = 0.0
LITE_EMA20_MAX = 0.0
LONG_WEAK_VOL_MAX = 119.0
FADE_LATE_GAP_MIN = 0.0            # BTC 1 d EMA50 / EMA200 − 1 (%) > this = late tape
FADE_POS_DI_MAX = 48.5
FADE_EXT_MAX = 1.80
ML_HOT_SHARE_MIN = 0.50            # share > this = market hot (frozen DL 416 cut; services/mkt_up4h.HOT_CUT)
ML_BULL_MIN = 90.0

# config toggle per line (SignalThresholds field) — one place, read by every caller
TOGGLES = {
    "WILLY_GAP": "watch_block_willy_gap",
    "WILLY_FAILED_SPIKE": "watch_block_willy_failed_spike",
    "WILLY_HT_FAILED_SPIKE": "watch_block_willy_ht_failed_spike",
    "RETRY_BTC1H": "watch_block_retry_btc1h",
    "LITE_BTC_EMA20_DOWN": "watch_block_lite_btc_ema20_down",
    "LONG_WEAK_FRENZY": "watch_block_long_weak_frenzy",
    "FADE_LATE_WEAK": "watch_block_fade_late_weak",
    "ML_MKT_HOT": "watch_block_ml_mkt_hot",
    "ML_BULL90": "watch_block_ml_bull90",
}


def _num(v):
    """a finite float or None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _on(th, line):
    return bool(getattr(th, TOGGLES[line], False))


def willy_blocks(th, adx, ema_gap_8_13, vs_vwap_pct, ht=False):
    """FRENZY_WILLY (ht=False) / FRENZY_WILLY_HT (ht=True) → the refusing lines. WILLY_GAP fires on EITHER leg it can read (a readable
    ADX > 55 blocks even when the gap is unreadable, and vice versa). HT runs WILLY_FAILED_SPIKE only (its own toggle)."""
    out = []
    if not ht and _on(th, "WILLY_GAP"):
        a, g = _num(adx), _num(ema_gap_8_13)
        if (a is not None and a > WILLY_GAP_ADX_MAX) or (g is not None and abs(g) <= WILLY_GAP_EMA_GAP_MAX):
            out.append("WILLY_GAP")
    if _on(th, "WILLY_HT_FAILED_SPIKE" if ht else "WILLY_FAILED_SPIKE"):
        v = _num(vs_vwap_pct)
        if v is not None and v <= FAILED_SPIKE_VWAP_MAX:
            out.append("WILLY_FAILED_SPIKE")
    return out


def retry_blocks(th, btc_1h_slope):
    v = _num(btc_1h_slope)
    return ["RETRY_BTC1H"] if (_on(th, "RETRY_BTC1H") and v is not None and v <= RETRY_BTC1H_MAX) else []


def lite_blocks(th, btc_ema20_slope_raw):
    v = _num(btc_ema20_slope_raw)
    return ["LITE_BTC_EMA20_DOWN"] if (_on(th, "LITE_BTC_EMA20_DOWN") and v is not None and v <= LITE_EMA20_MAX) else []


def long_blocks(th, frenzy_vol_mult):
    v = _num(frenzy_vol_mult)
    return ["LONG_WEAK_FRENZY"] if (_on(th, "LONG_WEAK_FRENZY") and v is not None and v <= LONG_WEAK_VOL_MAX) else []


def fade_blocks(th, btc_1d_ema50_200_gap_pct, pos_di, dist_from_ema13_pct):
    """SPIKE_FADE OBS-F1: every leg must be readable (the line is a conjunction — one unreadable leg = unscored, never blocked)."""
    if not _on(th, "FADE_LATE_WEAK"):
        return []
    g, p, x = _num(btc_1d_ema50_200_gap_pct), _num(pos_di), _num(dist_from_ema13_pct)
    if g is None or p is None or x is None:
        return []
    return ["FADE_LATE_WEAK"] if (g > FADE_LATE_GAP_MIN and p <= FADE_POS_DI_MAX and x <= FADE_EXT_MAX) else []


def ml_blocks(th, mkt_hot_share, bull_pct):
    out = []
    s, b = _num(mkt_hot_share), _num(bull_pct)
    if _on(th, "ML_MKT_HOT") and s is not None and s > ML_HOT_SHARE_MIN:
        out.append("ML_MKT_HOT")
    if _on(th, "ML_BULL90") and b is not None and b >= ML_BULL_MIN:
        out.append("ML_BULL90")
    return out


def frenzy_blocks(th, sleeve, adx=None, ema_gap_8_13=None, vs_vwap_pct=None, btc_1h_slope=None, btc_ema20_slope_raw=None,
                  frenzy_vol_mult=None):
    """the FRENZY open helper's single call: sleeve = the order's entry_strategy → its refusing lines (FRENZY_WIDE / unknown → [])."""
    if sleeve == "FRENZY_WILLY":
        return willy_blocks(th, adx, ema_gap_8_13, vs_vwap_pct, ht=False)
    if sleeve == "FRENZY_WILLY_HT":
        return willy_blocks(th, adx, ema_gap_8_13, vs_vwap_pct, ht=True)
    if sleeve == "FRENZY_LITE_RETRY":
        return retry_blocks(th, btc_1h_slope)
    if sleeve == "FRENZY_LITE":
        return lite_blocks(th, btc_ema20_slope_raw)
    if sleeve == "FRENZY_LONG":
        return long_blocks(th, frenzy_vol_mult)
    return []


def block_code(sleeve, lines):
    """the filter-block counter / journal gate name, e.g. FRENZY_WILLY_BLOCK_WILLY_GAP+WILLY_FAILED_SPIKE."""
    return f"{sleeve}_BLOCK_{'+'.join(lines)}"
