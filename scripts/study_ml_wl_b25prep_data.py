"""B25-prep MOMENTUM-LONG winners-vs-losers study — shared data loader (2026-10-26).

One row per MOMENTUM LONG CLOSED fill:
  · master (reports/MASTER_POOL_stacked.csv, every era BASE…B24, STACK 2026-10-26a) — probes flagged (separate line, never pooled)
  · B25 provisional (latest ~/Downloads/scalpars_orders_paper_2026-10-2*.csv, rows not already in the master, opened ≥ 2026-10-23 18:00)
Outcomes (% at 1×, leverage-invariant): y_trad = pnl_percentage (as traded) · y_keep = stack_pct on today's-stack KEPT rows (B25 = as
traded, it already runs today's stack; provisional).
Features: every entry_* stamp + the dashboard-derived items (main.py _ec_row: fan ratio, ADX/RSI/BTC-ADX/BTC-RSI/BTC-RSI-30m dir,
|EMA20 slope|, |BTC EMA20 slope|, LONG Ext% = entry_dist_from_ema13_pct, BTCExt% = entry_btc_dist_from_ema13_pct, Breadth = entry_bull_pct)
+ prior-study per-fill features merged by (pair, opened_at): ML_WL_1015 impulse (btc_r*, br_*, disp, agg/pair vol, rs*),
ML_IMPULSE legs/score (Oct-15), ML_PAIR_STRENGTH (Oct-4), SCOUT_ML_COOLDOWN, and this study's top-up for the post-10-15 fills
(reports/PREP_B25_ML_WL_features_new.csv, built by study_ml_wl_b25prep_features.py).
Read-only on every input.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "reports/MASTER_POOL_stacked.csv")
NEWFEAT = os.path.join(ROOT, "reports/PREP_B25_ML_WL_features_new.csv")
B25_START = "2026-10-23T18:00:00"
WASH = ("2026-06-18", "2026-07-03")          # washed-out window Jun-18 → Jul-2 inclusive
ERA_GROUP = {"BASE": "G0 BASE", "B1": "G1 B1", "B2": "G2 B2", "B3": "G3 B3"}
for _e in range(4, 12):
    ERA_GROUP[f"B{_e}"] = "G4 B4-B11"
for _e in range(12, 17):
    ERA_GROUP[f"B{_e}"] = "G5 B12-B16"
for _e in range(17, 25):
    ERA_GROUP[f"B{_e}"] = "G6 B17-B24"
ERA_GROUP["B25"] = "B25 (prov.)"

IMPULSE_1015 = ["btc_r5", "btc_r15", "btc_r30", "btc_r60", "btc_vol5_x24h", "br_up15", "br_up30", "br_up60", "br_up30_chg1h",
                "br_ret15", "br_ret60", "disp15", "agg_vol15_x24h", "pair_r15", "pair_r60", "rs15", "rs60", "pair_vol15_x24h"]
DURING_1015 = ["btc_d5", "btc_d15", "br_d5", "br_d15", "pnl_m1", "pnl_m3", "pnl_m5", "pnl_m15"]
IMP_SCORE = ["er_15", "upsh_15", "spike_15", "volconc_15", "volslope_15", "er_30", "upsh_30", "spike_30", "volconc_30",
             "volslope_30", "vwap1h", "eth_r15", "sol_r15", "xrp_r15", "zec_r15", "top10_up", "br_trend", "br_trend_d15",
             "top10_tbr", "vsurge", "corr15", "L1_btc", "L2_eth", "L3_sol", "L4_xrp", "L5_zec", "L6_top10", "L7_brrise",
             "L8_taker", "L9_vsurge", "SCORE", "CONJ", "SPIKE"]
IMP_FWD = ["Y_btc", "Y_alt"]   # FORWARD targets of the Oct-15 impulse study (BTC / top-50 return over the NEXT 15 min) — post-entry only
PSTR = ["rel15", "rel1h", "rel4h", "rel24h", "p15", "p1h", "p4h", "p24h", "btc_d13", "btc_d50", "btc1h", "btc4h", "btc24h",
        "L_rel24d", "L_relE200", "zoneD", "CHOP", "BURST", "CB"]
COOL = ["cd_flag", "cd_min_since_stop", "c120_prior", "c120_flag"]
DERIVED = ["d_fan_ratio", "d_adx_dir", "d_rsi_dir", "d_rsi_dir2", "d_btc_adx_dir", "d_btc_rsi_dir", "d_btc_rsi_30m_dir",
           "d_btc_rsi_1h_dir", "d_abs_ema20_slope", "d_abs_btc_ema20_slope", "d_di_spread", "d_gap5_20_chg", "d_hour", "d_dow",
           "d_burst_prior_min", "d_fills_same_scan"]
POST = ["peak_pnl", "trough_pnl", "dur_min", "peak_min", "trough_min", "post_Y_btc", "post_Y_alt"]

# Dashboard tables (templates/index.html L1510-1620 / exports ~L10110 / ~L11969; computed in main.py _ec_row) → column used here
DASHBOARD_MAP = [
    ("RSI", "entry_rsi"), ("ADX", "entry_adx"), ("ADXDir", "d_adx_dir (entry_adx − entry_adx_prev; Rising = > 0)"),
    ("RSIDir", "d_rsi_dir (entry_rsi − entry_rsi_prev; Rising = > 0)"), ("Gap", "entry_gap"), ("Gap5-8", "entry_ema_gap_5_8"),
    ("Gap8-13", "entry_ema_gap_8_13"), ("Stretch", "entry_ema5_stretch"), ("EMA20Slp", "d_abs_ema20_slope (|entry_ema20_slope|) + signed entry_ema20_slope"),
    ("ADXΔ", "entry_adx_delta"), ("RngPos", "entry_range_position"), ("ATR%", "entry_atr_pct"),
    ("PairTrend", "entry_pair_ema20_ema50_gap_pct (holds EMA13−EMA50)"), ("PairVol$", "entry_pair_volume_24h_usd"),
    ("Ext% (LONG)", "entry_dist_from_ema13_pct"), ("1hSlope", "entry_btc_1h_slope"),
    ("BTCSlp", "d_abs_btc_ema20_slope (|entry_btc_ema20_slope|) + signed entry_btc_ema20_slope"), ("BTCRSI", "entry_btc_rsi"),
    ("BTCADX", "entry_btc_adx"), ("BTCADXDir", "d_btc_adx_dir (entry_btc_adx − entry_btc_adx_prev)"),
    ("BTCRSIDir", "d_btc_rsi_dir (entry_btc_rsi − entry_btc_rsi_prev) + d_btc_rsi_30m_dir (vs entry_btc_rsi_prev6)"),
    ("Breadth (LONG)", "entry_bull_pct"), ("BTCTrend", "entry_btc_trend_gap_pct"), ("GlobalVol", "entry_global_volume_ratio"),
    ("BTCExt% (LONG)", "entry_btc_dist_from_ema13_pct"), ("Conf", "confidence (cat__confidence==*)"), ("Fan ratio", "d_fan_ratio"),
    ("BTC ATR% (perf payload)", "entry_btc_atr_pct"), ("EMA5 dist (perf payload)", "entry_price_vs_ema5_pct"),
    ("BTCTrend(exit) — EXIT var", "exit_btc_trend_gap_pct (descriptive only)"),
    ("Peak — POST-ENTRY", "peak_pnl (descriptive only)"), ("Trough — POST-ENTRY", "trough_pnl (descriptive only)"),
    ("Duration — POST-ENTRY", "dur_min (descriptive only)"), ("SLProf — POST-ENTRY", "peak_pnl of losers (descriptive only)"),
]
NON_SIGNAL = {"entry_price", "entry_fee", "entry_slippage_pct", "entry_desired_notional", "entry_liquidity_cap_notional",
              "entry_order_type", "entry_strategy", "entry_btc_regime_started_at", "entry_bracket_max_leverage",
              "entry_bracket_cap_notional", "entry_surge_trigger_at", "entry_frenzy_spike_at", "entry_btc_price",
              "entry_fill_vs_book_pct", "entry_order_lag_s", "entry_frenzy_vwap"}
CATEG = ["entry_macro_trend", "entry_btc_regime", "confidence", "entry_br_door"]
MACRO_KEYS = ("btc", "global_volume", "entry_bull_pct", "entry_bear_pct", "eth_", "entry_br_", "entry_bear_", "surge_gvol",
              "frenzy_gvol", "entry_macro_trend", "br_", "disp15", "agg_vol", "sol_r", "xrp_r", "zec_r", "top10", "vsurge",
              "corr15", "Y_btc", "Y_alt", "L1_", "L2_", "L3_", "L4_", "L5_", "L6_", "L7_", "L8_", "L9_", "SCORE", "CONJ", "SPIKE",
              "d_hour", "d_dow", "CHOP", "confidence")


def is_macro(c):
    if c in ("rs15", "rs60", "rel15", "rel1h", "rel4h", "rel24h", "L_rel24d", "L_relE200"):
        return False
    return any(k in c for k in MACRO_KEYS)


def _key(s):
    return s.astype(str).str[:19].str.replace(" ", "T")


def _b25_file():
    fs = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-10-2*.csv")), key=os.path.getmtime)
    return fs[-1] if fs else None


def add_derived(d):
    n = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
    g58, g813 = n("entry_ema_gap_5_8"), n("entry_ema_gap_8_13")
    d["d_fan_ratio"] = np.where(g813 > 0, g58 / g813, np.nan)
    d["d_adx_dir"] = n("entry_adx") - n("entry_adx_prev")
    d["d_rsi_dir"] = n("entry_rsi") - n("entry_rsi_prev")
    d["d_rsi_dir2"] = np.nan   # rsi_prev2 is not stamped (dashboard RSIDir uses rsi_prev) — blind spot, kept for the coverage table
    d["d_btc_adx_dir"] = n("entry_btc_adx") - n("entry_btc_adx_prev")
    d["d_btc_rsi_dir"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev")
    d["d_btc_rsi_30m_dir"] = n("entry_btc_rsi") - n("entry_btc_rsi_prev6")
    d["d_btc_rsi_1h_dir"] = n("entry_btc_rsi_1h") - n("entry_btc_rsi_1h_prev")
    d["d_abs_ema20_slope"] = n("entry_ema20_slope").abs()
    d["d_abs_btc_ema20_slope"] = n("entry_btc_ema20_slope").abs()
    d["d_di_spread"] = n("entry_pos_di") - n("entry_neg_di")
    d["d_gap5_20_chg"] = n("entry_gap_5_20_signed_pct") - n("entry_gap_5_20_prev_signed_pct")
    d["d_hour"] = d.ts.dt.hour
    d["d_dow"] = d.ts.dt.dayofweek
    return d


def _scan_windows(d, all_bot):
    """2-min scan windows: consecutive ML fills ≤ 120 s apart share a window id; d_burst_prior_min = minutes since the previous
    non-probe non-MANUAL bot fill (any sleeve) in the same source; d_fills_same_scan = ML fills in the same window."""
    d = d.sort_values("ts").copy()
    gap = d.ts.diff().dt.total_seconds()
    d["win_id"] = (gap.isna() | (gap > 120)).cumsum()
    d["d_fills_same_scan"] = d.groupby("win_id").pair.transform("size")
    bt = np.sort(all_bot.values.astype("datetime64[ns]").astype(np.int64))
    t = d.ts.values.astype("datetime64[ns]").astype(np.int64)
    i = np.searchsorted(bt, t, side="left") - 1
    prev = np.where(i >= 0, bt[np.clip(i, 0, None)], np.nan)
    d["d_burst_prior_min"] = np.where(i >= 0, (t - prev) / 6e10, np.nan)
    return d


def load_all(with_features=True, use_new=True):
    m = pd.read_csv(MASTER, low_memory=False)
    m["src"] = "master"
    bf = _b25_file()
    parts = [m]
    if bf:
        b = pd.read_csv(bf, low_memory=False)
        b = b[(b.status == "CLOSED") & (_key(b.opened_at) >= B25_START)].copy()
        mk = set(zip(_key(m.opened_at), m.pair, m.direction))
        b = b[[k not in mk for k in zip(_key(b.opened_at), b.pair, b.direction)]].copy()
        b["era"] = "B25"; b["src"] = "B25 " + os.path.basename(bf)
        b["stack_keep"] = True; b["stack_pct"] = b.pnl_percentage
        b["is_probe"] = b.get("is_probe", pd.Series(False, index=b.index))
        if "is_probe" not in b or b.is_probe.isna().all():
            b["is_probe"] = False
        parts.append(b)
    A = pd.concat(parts, ignore_index=True, sort=False)
    A["ts"] = pd.to_datetime(_key(A.opened_at))
    A["is_probe"] = A.is_probe.astype(str).str.lower().isin(["true", "1", "1.0"])
    A["stack_keep"] = A.stack_keep.astype(str).str.lower().isin(["true", "1", "1.0"])
    bot = A[(A.entry_strategy.astype(str) != "MANUAL") & ~A.is_probe].ts
    D = A[(A.entry_strategy.astype(str) == "MOMENTUM") & (A.direction == "LONG") & (A.status == "CLOSED")].copy()
    D["key_open"] = _key(D.opened_at)
    D["day"] = D.ts.dt.floor("D")
    D["era_group"] = D.era.map(ERA_GROUP).fillna("?")
    D["washed"] = (D.ts >= WASH[0]) & (D.ts < WASH[1])
    D["y_trad"] = pd.to_numeric(D.pnl_percentage, errors="coerce")
    D["y_keep"] = np.where(D.stack_keep, pd.to_numeric(D.stack_pct, errors="coerce"), np.nan)
    # 1×-notional $ for concentration: % × the fill's notional ÷ its effective multiplier is not uniform across eras → use % only
    D["closed_ts"] = pd.to_datetime(D.closed_at.astype(str).str[:23], format="mixed", errors="coerce")
    D["dur_min"] = (D.closed_ts - D.ts).dt.total_seconds() / 60
    for c, nm in [("peak_reached_at", "peak_min"), ("trough_reached_at", "trough_min")]:
        x = pd.to_datetime(D[c].astype(str).str[:23], format="mixed", errors="coerce")
        D[nm] = (x - D.ts).dt.total_seconds() / 60
    D = add_derived(D)
    D = _scan_windows(D, bot[bot.notna()])
    for c in CATEG:
        if c in D:
            for v in sorted(D[c].dropna().astype(str).unique()):
                D[f"cat__{c}=={v}"] = (D[c].astype(str) == v).astype(float).where(D[c].notna())
    if with_features:
        D = _merge_features(D, use_new)
    return D.sort_values("ts").reset_index(drop=True)


def _merge_features(D, use_new=True):
    k = list(zip(D.pair, D.key_open))
    f = pd.read_csv(os.path.join(ROOT, "reports/ML_WL_1015_features.csv"))
    f = f[f.block.isin(["master", "live"])].copy(); f["k"] = list(zip(f.pair, _key(f.opened_at)))
    f = f.drop_duplicates("k").set_index("k")
    if use_new and os.path.exists(NEWFEAT):      # this study's rebuild fills the keys the 10-15 file lacks or left NaN
        g = pd.read_csv(NEWFEAT); g["k"] = list(zip(g.pair, _key(g.opened_at)))
        g = g.drop_duplicates("k").set_index("k")
        meta = ("block", "pair", "opened_at", "seed")
        f = f.drop(columns=[c for c in meta if c in f]).combine_first(g.drop(columns=[c for c in meta if c in g]))
    for c in IMPULSE_1015 + DURING_1015:
        D[c] = [f[c].get(x, np.nan) if c in f else np.nan for x in k]
    s = pd.read_csv(os.path.join(ROOT, "reports/ML_IMPULSE_fill_features.csv"))
    s = s[s.block.isin(["master", "live"])].copy()
    s["k"] = list(zip(s.pair, pd.to_datetime(s.o_ms, unit="ms").dt.strftime("%Y-%m-%dT%H:%M:%S")))
    s = s.drop_duplicates("k").set_index("k")
    for c in IMP_SCORE:
        D["imp_" + c] = [s[c].get(x, np.nan) for x in k]
    for c in IMP_FWD:
        D["post_" + c] = [s[c].get(x, np.nan) for x in k]
    p = pd.read_csv(os.path.join(ROOT, "reports/ML_PAIR_STRENGTH_2026-10-04_master_features.csv"))
    p["k"] = list(zip(p.pair, _key(p.opened_at))); p = p.drop_duplicates("k").set_index("k")
    for c in PSTR:
        v = [p[c].get(x, np.nan) for x in k]
        D["ps_" + c] = pd.to_numeric(pd.Series(v, index=D.index).replace({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce")
    cd = pd.read_csv(os.path.join(ROOT, "reports/SCOUT_ML_COOLDOWN.csv"))
    cd["k"] = list(zip(cd.pair, _key(cd.opened_at))); cd = cd.drop_duplicates("k").set_index("k")
    for c in COOL:
        v = [cd[c].get(x, np.nan) for x in k]
        D["cd_" + c] = pd.to_numeric(pd.Series(v, index=D.index).replace({True: 1, False: 0, "True": 1, "False": 0}), errors="coerce")
    return D


def feature_columns(D):
    """every candidate entry-time variable (numeric, ≥ 2 distinct values somewhere)."""
    cols = []
    for c in D.columns:
        ok = (c.startswith("entry_") and c not in NON_SIGNAL and c not in CATEG) or c.startswith("cat__") or c in DERIVED \
            or c in IMPULSE_1015 or c.startswith("imp_") or c.startswith("ps_") or c.startswith("cd_")
        if not ok:
            continue
        s = pd.to_numeric(D[c], errors="coerce")
        if s.notna().sum() == 0:
            cols.append(c); continue
        cols.append(c)
    return cols
