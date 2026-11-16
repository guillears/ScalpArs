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
                   LITE_BTC_RSI_HOT    BTC RSI (the entry stamp, 1 dp) ≥ 70 — BTC overbought                (DL 554, declared override)
                   LITE_WEAK_HOLD      the price spent ≤ 50 % of the episode's 5m closes above the spike VWAP (entry_frenzy_above_share, 2 dp) (DL 556)
  FRENZY_LONG      LONG_WEAK_FRENZY    frenzy volume multiple ≤ 119× normal                  (DL 525)
  SPIKE_FADE       OBS-F1 late weak spike  BTC 1 d EMA50 > EMA200 (late tape) ∧ pair +DI ≤ 48.5 ∧ price ≤ 1.80 % above the pair's EMA13  (DL 384)
  MOMENTUM LONG    ML_MKT_HOT          > 50 % of the rolling top-100 up over 4 h (services/mkt_up4h share > 0.50)   (DL 416 / 489)
                   ML_BULL90           ≥ 90 % of the scanned coins bullish (entry_bull_pct)   (DL 523)
                   ML_WEAK_RESCUE      weak trend (entry_gap ≤ 0.27) let in by the volume-gate rescue (global volume < 0.70)   (DL 555, declared override)
                   ML_WEAK_MARKET      market breadth weak (entry_bull_pct ≤ 64.1) ∧ BTC trend weak (entry_btc_adx ≤ 21.71)        (DL 559, declared override)
                   ML_FADING_QUIET     the coin's ADX falling at entry (entry_adx_delta < 0) ∧ a quiet coin (entry_atr_pct ≤ 0.38)  (DL 559, declared override)
  MOMENTUM SHORT   MS_REGIME_ENDS      BTC daily uptrend (1 d EMA50 > EMA200, closed bars) ∧ entry_quality_score ≤ 2 or ≥ 5            (DL 561, declared override)
"""
import math

WILLY_GAP_ADX_MAX = 55.0
WILLY_GAP_EMA_GAP_MAX = 0.90
FAILED_SPIKE_VWAP_MAX = 0.0
RETRY_BTC1H_MAX = 0.0
LITE_EMA20_MAX = 0.0
LITE_BTC_RSI_HOT_MIN = 70.0        # (DL 554) the textbook overbought line, NOT fitted (75 sat just above the one winner — cherry-picked)
LITE_WEAK_HOLD_MAX = 50.0          # (DL 556) "half the time or less above the spike average" — a natural line, not fitted
LONG_WEAK_VOL_MAX = 119.0
FADE_LATE_GAP_MIN = 0.0            # BTC 1 d EMA50 / EMA200 − 1 (%) > this = late tape
FADE_POS_DI_MAX = 48.5
FADE_EXT_MAX = 1.80
ML_HOT_SHARE_MIN = 0.50            # share > this = market hot (frozen DL 416 cut; services/mkt_up4h.HOT_CUT)
ML_BULL_MIN = 90.0
ML_WEAK_GAP_MAX = 0.27             # (DL 555) EMA gap ≤ this = weak trend (master bottom third; robust 0.25–0.36)
ML_WEAK_BULL_MAX, ML_WEAK_BTC_ADX_MAX = 64.1, 21.71   # (DL 559) master full-size terciles, frozen
ML_FADING_ADX_DELTA_MAX, ML_QUIET_ATR_MAX = 0.0, 0.38  # (DL 559) ADX falling (< 0) · ATR % bottom tercile, frozen
MS_QS_WEAK_MAX, MS_QS_LATE_MIN = 2, 5   # (DL 561) momentum short: score ≤ 2 = unconfirmed, ≥ 5 = exhausted (scores 3–4 trade in any regime)
ML_QUIET_GVOL_MAX = 0.70           # (DL 555) global volume < this = the VOL_GATE's long threshold AT SHIP (frozen — the rescue fires < 0.60, so the live band
#                                    is < 0.60 today; if the VOL_GATE long threshold is ever lowered, longs at the new gate..0.70 would start being blocked — re-read)

# config toggle per line (SignalThresholds field) — one place, read by every caller
TOGGLES = {
    "WILLY_GAP": "watch_block_willy_gap",
    "WILLY_FAILED_SPIKE": "watch_block_willy_failed_spike",
    "WILLY_HT_FAILED_SPIKE": "watch_block_willy_ht_failed_spike",
    "RETRY_BTC1H": "watch_block_retry_btc1h",
    "LITE_BTC_EMA20_DOWN": "watch_block_lite_btc_ema20_down",
    "LITE_BTC_RSI_HOT": "watch_block_lite_btc_rsi_hot",
    "LITE_WEAK_HOLD": "watch_block_lite_weak_hold",
    "LONG_WEAK_FRENZY": "watch_block_long_weak_frenzy",
    "FADE_LATE_WEAK": "watch_block_fade_late_weak",
    "ML_MKT_HOT": "watch_block_ml_mkt_hot",
    "ML_BULL90": "watch_block_ml_bull90",
    "ML_WEAK_RESCUE": "watch_block_ml_weak_rescue",
    "ML_WEAK_MARKET": "watch_block_ml_weak_market",
    "ML_FADING_QUIET": "watch_block_ml_fading_quiet",
    "MS_REGIME_ENDS": "watch_block_ms_regime_ends",
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


def lite_blocks(th, btc_ema20_slope_raw, btc_rsi=None, above_share=None):
    out = []
    v = _num(btc_ema20_slope_raw)
    if _on(th, "LITE_BTC_EMA20_DOWN") and v is not None and v <= LITE_EMA20_MAX:
        out.append("LITE_BTC_EMA20_DOWN")
    r = _num(btc_rsi)
    if _on(th, "LITE_BTC_RSI_HOT") and r is not None and round(r, 1) >= LITE_BTC_RSI_HOT_MIN:   # judged on the stamp's 1-dp value (master parity)
        out.append("LITE_BTC_RSI_HOT")
    a = _num(above_share)
    if _on(th, "LITE_WEAK_HOLD") and a is not None and round(a, 2) <= LITE_WEAK_HOLD_MAX:   # the stamp's 2-dp value (master parity)
        out.append("LITE_WEAK_HOLD")
    return out


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


def ms_blocks(th, btc_1d_ema50_200_gap_pct, quality_score):
    """(DL 561) MOMENTUM SHORT in a BTC daily uptrend (gap > 0) whose quality score is at an end: ≤ 2 (no confirmation on the coin) or ≥ 5
    (already oversold / exhausted). An unreadable gap or score never blocks. Note: the engine's quality score is always an int — a missing
    input scores 0 on its leg (the same function stamped every analysed fill, so live and master agree)."""
    if not _on(th, "MS_REGIME_ENDS"):
        return []
    g, q = _num(btc_1d_ema50_200_gap_pct), _num(quality_score)
    if g is None or q is None:
        return []
    return ["MS_REGIME_ENDS"] if (g > 0 and (q <= MS_QS_WEAK_MAX or q >= MS_QS_LATE_MIN)) else []


def ml_blocks(th, mkt_hot_share, bull_pct, entry_gap=None, global_volume_ratio=None, btc_adx=None, adx_delta=None, atr_pct=None):
    out = []
    s, b = _num(mkt_hot_share), _num(bull_pct)
    if _on(th, "ML_MKT_HOT") and s is not None and s > ML_HOT_SHARE_MIN:
        out.append("ML_MKT_HOT")
    if _on(th, "ML_BULL90") and b is not None and b >= ML_BULL_MIN:
        out.append("ML_BULL90")
    g, v = _num(entry_gap), _num(global_volume_ratio)
    if _on(th, "ML_WEAK_RESCUE") and g is not None and v is not None and g <= ML_WEAK_GAP_MAX and v < ML_QUIET_GVOL_MAX:
        out.append("ML_WEAK_RESCUE")   # both legs readable — a missing reading never blocks
    ba = _num(btc_adx)
    # (559 review) bull % 0.0 is the engine's "no pairs scanned" fallback, not a reading → fail open
    if _on(th, "ML_WEAK_MARKET") and b is not None and b > 0 and ba is not None and b <= ML_WEAK_BULL_MAX and ba <= ML_WEAK_BTC_ADX_MAX:
        out.append("ML_WEAK_MARKET")
    ad, at = _num(adx_delta), _num(atr_pct)
    if _on(th, "ML_FADING_QUIET") and ad is not None and at is not None and ad < ML_FADING_ADX_DELTA_MAX and at <= ML_QUIET_ATR_MAX:
        out.append("ML_FADING_QUIET")
    return out


def frenzy_blocks(th, sleeve, adx=None, ema_gap_8_13=None, vs_vwap_pct=None, btc_1h_slope=None, btc_ema20_slope_raw=None,
                  frenzy_vol_mult=None, btc_rsi=None, above_share=None):
    """the FRENZY open helper's single call: sleeve = the order's entry_strategy → its refusing lines (FRENZY_WIDE / unknown → [])."""
    if sleeve == "FRENZY_WILLY":
        return willy_blocks(th, adx, ema_gap_8_13, vs_vwap_pct, ht=False)
    if sleeve == "FRENZY_WILLY_HT":
        return willy_blocks(th, adx, ema_gap_8_13, vs_vwap_pct, ht=True)
    if sleeve == "FRENZY_LITE_RETRY":
        return retry_blocks(th, btc_1h_slope)
    if sleeve == "FRENZY_LITE":
        return lite_blocks(th, btc_ema20_slope_raw, btc_rsi, above_share)
    if sleeve == "FRENZY_LONG":
        return long_blocks(th, frenzy_vol_mult)
    return []


def block_code(sleeve, lines):
    """the filter-block counter / journal gate name, e.g. FRENZY_WILLY_BLOCK_WILLY_GAP+WILLY_FAILED_SPIKE."""
    return f"{sleeve}_BLOCK_{'+'.join(lines)}"
