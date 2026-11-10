"""🧭 Watch-line ZONES — the one definition of every current per-sleeve block line, as a boolean zone per fill (DECISION_LOG 471).

Shared by the best-case view (scripts/best_case/best_dcr.py, best_dcr_noflip.py) and the ALPHA_SLEEVES confound note
(scripts/scout_alpha_sleeves_watch.py: "is the loss the Alpha coin, or another watch line?" — operator 11-05). Every Alpha fill is a fill
of its sleeve: it counts in that sleeve's own lines like any other fill; this module only lets a reader see WHICH lines it sits in.

Lines that need market data the fill row does not carry take it as an argument: hot (ML_MKT_HOT market-breadth zone, per fill
'zone' / other), late (fade OBS-F1 late-spike gap, > 0 = late), cool (momentum-long stop cooldown, bool). Passing None marks those
lines NOT SCORED (value None, never a silent False). Unscored inputs inside a scored line (NaN column) read False — never in the zone,
as the best case always did. PURE (no I/O, no network).
"""
import numpy as np
import pandas as pd


def sleeve_name(entry_strategy, direction):
    """entry_strategy (+ direction) → the zone sleeve name. PURE."""
    s = "" if entry_strategy is None or (isinstance(entry_strategy, float) and entry_strategy != entry_strategy) else str(entry_strategy)
    if s in ("", "nan", "MOMENTUM"):
        return "MOMENTUM_" + str(direction)
    return s


def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)


def zones(d, hot=None, late=None, cool=None):
    """d = fills (entry_* columns) → {sleeve: {line: boolean Series over d.index, or None when its market input was not given}}."""
    idx = d.index
    hb = d.entry_btc_regime.astype(str).eq("HEALTHY_BEAR") if "entry_btc_regime" in d else pd.Series(False, index=idx)
    ad, ax, ba = _num(d, "entry_adx_delta"), _num(d, "entry_adx"), _num(d, "entry_btc_adx_prev")
    pdi, ext, rsi1 = _num(d, "entry_pos_di"), _num(d, "entry_dist_from_ema13_pct"), _num(d, "entry_btc_rsi_1h")
    gap = _num(d, "entry_ema_gap_8_13")
    w4 = d.get("entry_pattern_w4_match", pd.Series(False, index=idx)).astype(str).str.lower().isin(("true", "1", "1.0"))
    hotz = None if hot is None else pd.Series(np.asarray(hot), index=idx).eq("zone")
    latez = None if late is None else (pd.Series(np.asarray(late, dtype=float), index=idx) > 0) & (pdi <= 48.5) & (ext <= 1.80)
    coolz = None if cool is None else pd.Series(np.asarray(cool), index=idx).fillna(False).astype(bool)
    wgap = (ax > 55) | (gap <= 0.90)
    b1h, fh = _num(d, "entry_btc_1h_slope"), _num(d, "entry_frenzy_hours")
    return {
        "SPIKE_FADE": {"① ADX not rising": ad < 2.359, "② pair ADX ≤ 19.2": ax <= 19.2, "⑤ HB ∧ ADX flat": hb & (ad <= 2.4341),
                       "⑥ HB ∧ BTC ADX high": hb & (ba > 24.5874), "OBS-F1 late weak spike": latez,
                       "NOSTRETCH": (rsi1 <= 50) & (pdi <= 45) & (ext <= 2)},
        "FRENZY_WILLY": {"WILLY_GAP": wgap, "WILLY_ADX55": ax > 55, "MKT_HOT": hotz,
                         "WILLY_FAILED_SPIKE": _num(d, "entry_frenzy_vs_vwap_pct") <= 0},   # DL 509 (scout_willy_failed_spike_watch)
        "FRENZY_WILLY_HT": {"WILLY_GAP": wgap, "MKT_HOT": hotz,
                            "WILLY_FAILED_SPIKE": _num(d, "entry_frenzy_vs_vwap_pct") <= 0},   # DL 511
        "MOMENTUM_LONG": {"ML_MKT_HOT": hotz, "ML_W4": w4, "STOP_COOLDOWN": coolz,
                          "PAIR_1H_DOWNTREND": _num(d, "entry_pair_1h_ema20_200_gap_pct") <= 0},
        "FRENZY_LITE": {"LITE_LATE_BTCFALL": (fh >= 12) & (b1h <= 0), "LITE_BTC1H_RISING": b1h > 0,   # RISING: DL 437 (scout_lite_btc1h_rising_watch)
                        "LITE_BTC_EMA20_DOWN": _num(d, "entry_btc_ema20_slope") <= 0},   # DL 522 (scout_lite_btc_ema20_down_watch)
        "FRENZY_LITE_RETRY": {"RETRY_BTC1H": b1h <= 0, "LATE_BTCFALL": (fh >= 12) & (b1h <= 0)},
        "FRENZY_LONG": {"LONG_ADX35": ax > 35},
        "SURGE_LONG": {"SURGE_BTC_HOT": _num(d, "entry_btc_rsi") >= 78, "SURGE_THIN": _num(d, "entry_pair_volume_24h_usd") < 150e6},
    }


ZONE_COLS = ("entry_btc_regime", "entry_adx_delta", "entry_adx", "entry_btc_adx_prev", "entry_pos_di", "entry_dist_from_ema13_pct",
             "entry_btc_rsi_1h", "entry_ema_gap_8_13", "entry_pattern_w4_match", "entry_pair_1h_ema20_200_gap_pct", "entry_btc_1h_slope",
             "entry_frenzy_hours", "entry_btc_rsi", "entry_pair_volume_24h_usd", "entry_frenzy_vs_vwap_pct", "entry_btc_ema20_slope")
