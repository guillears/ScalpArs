#!/usr/bin/env python3
"""WILLY THEORY STUDY 2026-10-16 — shared code (first-principles test of the FRENZY_WILLY mechanism, W0–W8).

Pre-registration: reports/WILLY_THEORY_PREREG.txt (frozen with input hashes BEFORE any outcome is read).
Research only: no engine / config / template / test edit, no bot API, no commit. Engine functions are imported READ-ONLY
(services.frenzy). Episode bars reuse the LITE theory rebuild (reports/lite_theory/bars.pkl, built by the ENGINE'S frenzy_walk) plus the
same builder (study_lite_theory_episodes.build_pair) for the WILLY pairs it did not cover. Walkers reused, never rewritten:
scout_frenzy_willy_exit.load_path / evaluate and scout_willy_timecap.evaluate_late (via study_tick_maker_lock_common 2025 loader patches).
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_lite_theory_common as LT  # noqa: E402

OUT = os.path.join(REP, "willy_theory")
os.makedirs(OUT, exist_ok=True)
PREREG = os.path.join(REP, "WILLY_THEORY_PREREG.txt")
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
LOWVWAP_ROWS = os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_rows.csv")
LOWVWAP_LIVE = os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_live.csv")
MASTER = os.path.join(REP, "MASTER_POOL_stacked.csv")
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-16_13-10-21 (1).csv")
BARS_LITE = os.path.join(REP, "lite_theory", "bars.pkl")
BARS_EXTRA = os.path.join(OUT, "bars_extra.pkl")
SIG = os.path.join(OUT, "signals.pkl")          # one row per WILLY episode (trigger + entry + features)
ALT = os.path.join(OUT, "alt_entries.pkl")      # every candidate entry bar in the 60-min window of every episode
BAR_MS, MIN, HOUR_MS, DAY_MS = 300_000, 60_000, 3_600_000, 86_400_000
T_LAG_MS = 12_000                               # replay entry ≈ red-bar close + 11.9 s
WAIT_BARS = 12                                  # frenzy_willy_red_max_wait_minutes 60 → bars T0 … T0 + 12
R_CUT = 1.0                                     # frenzy_willy_max_vol_mcap_ratio
TP, SL, CAP, LATE, LTP, SLIP = 1.25, 4.0, 90, 30, 0.25, 0.2   # TODAY's WILLY exit
ENTRY_SLIP = 0.036                              # % taker entry slip over the print (paper-measured T1 ruler)
SPLIT26 = LT.SPLIT26
SPLITH2 = LT.SPLITH2
era_of = LT.era_of
year_of = LT.year_of
sha = LT.sha
load_5m = LT.load_5m


def ms(ts):
    return pd.to_datetime(ts, format="mixed").values.astype("datetime64[ms]").astype("int64")


def book(ev):
    """the LOWVWAP / WILLY_DURATION booking of today's exit: TP / LATE_TP at the line (resting maker), STOP at min(print, −4) − 0.2 slip,
    CAP at the last print (market)."""
    w = ev["why"]
    if w in ("TP", "LATE_TP"):
        return float(ev["pct_line"])
    if w == "STOP":
        return min(float(ev["pct"]), float(ev["pct_line"])) - SLIP      # pct_line = −sl
    if w == "CAP":
        return float(ev["pct"])
    return np.nan


def supply_table():
    """pair → circulating-supply proxy used by the replay's mcap (entry_mcap_usd = supply × price; bapi_supply.json cs / mult)."""
    d = json.load(open(os.path.join(REP, "cache_vol_mcap", "bapi_supply.json")))
    out = {}
    for p, v in d.items():
        if isinstance(v, dict) and v.get("cs") and v.get("mc"):
            out[p] = float(v["cs"]) / float(v.get("mult") or 1)
    return out


def listing_ms():
    out = {}
    for f in (os.path.join(REP, "backtest_cache_2025", "exchange_info.json"), os.path.join(REP, "backtest_cache", "exchange_info.json")):
        try:
            for p, v in json.load(open(f)).items():
                if isinstance(v, dict) and v.get("onboardDate"):
                    out[p] = min(out.get(p, 1 << 62), int(v["onboardDate"]))
        except Exception:  # noqa: BLE001
            pass
    return out


# ───────────────────────── statistics ─────────────────────────
def boot_mean(v, clus, n=4000, seed=7):
    """window(cluster)-bootstrap of the mean of v → (P(mean > 0), lo95, hi95)."""
    return LT.day_boot_diff(v, clus, n=n, seed=seed)


def be_wr(y):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    w, l = y[y > 0], y[y <= 0]
    if not len(w) or not len(l):
        return np.nan
    return abs(l.mean()) / (w.mean() + abs(l.mean())) * 100


def summ(y):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if not len(y):
        return dict(n=0, wr=np.nan, avg=np.nan, sum=np.nan)
    return dict(n=len(y), wr=(y > 0).mean() * 100, avg=y.mean(), sum=y.sum())
