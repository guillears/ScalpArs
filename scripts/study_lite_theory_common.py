#!/usr/bin/env python3
"""LITE THEORY STUDY 2026-10-15 — shared code (operator: "go with theory — what is a LITE trade — and from there start testing everything").

Questions the LITE DEFINITION itself (A0–A9, pre-registered in reports/LITE_THEORY_PREREG.txt).
Research only: no engine / config / template / test edit, no bot API, no commit, cache-only (no Binance call).
Engine functions are imported READ-ONLY (services.frenzy: frenzy_walk, normal_hour_usd, frenzy_lite_status, frenzy_lite_stretch_id,
frenzy_bearish_block); nothing is re-implemented for the signal itself.
"""
import os, sys, json, types, hashlib
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
OUT = os.path.join(REP, "lite_theory")
os.makedirs(OUT, exist_ok=True)
PREREG = os.path.join(REP, "LITE_THEORY_PREREG.txt")
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
COHORT = os.path.join(REP, "LITE_PROTECT_rows.csv")
BARS = os.path.join(OUT, "bars.pkl")
C25 = os.path.join(REP, "backtest_cache_2025")
C26 = os.path.join(REP, "backtest_cache")
BAR_MS, HOUR_MS, MIN = 300_000, 3_600_000, 60_000
WIN = 1499                      # closed 5m bars the engine walks (FRENZY_KL_LIMIT 1500 incl. the forming bar)
T_LAG_MS = 12_000               # replay / live entry ≈ signal-bar close + 10–14 s
START = int(pd.Timestamp("2025-01-01", tz="UTC").value // 1_000_000)
END = int(pd.Timestamp("2026-10-16", tz="UTC").value // 1_000_000)
SPLIT26 = int(pd.Timestamp("2026-01-01", tz="UTC").value // 1_000_000)
SPLITH2 = int(pd.Timestamp("2026-07-01", tz="UTC").value // 1_000_000)


def th():
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
    d = cfg.get("thresholds", cfg)
    return types.SimpleNamespace(**d)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_5m(pair):
    """every cached 5m source merged (dedup on open_time, first source wins): 2025 cache k5m_full, 2026 k5m_full, k5m, k5m_ext,
    lite_fade_klines/k5m. Columns open_time o h l c vol qvol."""
    srcs = [os.path.join(C25, "k5m_full", f"{pair}.csv"), os.path.join(C26, "k5m_full", f"{pair}.csv"),
            os.path.join(C26, "k5m", f"{pair}.csv"), os.path.join(C26, "k5m_ext", f"{pair}.csv"),
            os.path.join(REP, "lite_fade_klines", "k5m", f"{pair}.csv")]
    fr = []
    for s in srcs:
        if os.path.exists(s):
            try:
                d = pd.read_csv(s)
                if len(d) and {"open_time", "o", "h", "l", "c", "vol"} <= set(d.columns):
                    if "qvol" not in d:
                        d["qvol"] = d.vol * (d.h + d.l + d.c) / 3.0
                    fr.append(d[["open_time", "o", "h", "l", "c", "vol", "qvol"]])
            except Exception:  # noqa: BLE001
                pass
    if not fr:
        return None
    d = pd.concat(fr).drop_duplicates("open_time", keep="first").sort_values("open_time").reset_index(drop=True)
    d = d[d.open_time % BAR_MS == 0]
    return d.astype({"open_time": "int64"})


def era_of(ms):
    ms = int(ms)
    if ms < SPLIT26:
        return "2025"
    return "2026-H1" if ms < SPLITH2 else "2026-H2"


def year_of(ms):
    return "2025" if int(ms) < SPLIT26 else "2026"


# ───────────────────────── statistics (same conventions as the earlier LITE studies) ─────────────────────────
def day_boot_diff(vals, days, n=4000, seed=7):
    """day-clustered bootstrap of the SUM over days of a per-row value (Δ) → (P(sum > 0), lo95, hi95) in mean-per-row units."""
    v = np.asarray(vals, float); d = np.asarray(days)
    ok = np.isfinite(v); v, d = v[ok], d[ok]
    if not len(v):
        return np.nan, np.nan, np.nan
    s = pd.Series(v).groupby(d).agg(["sum", "count"])
    S, N = s["sum"].values, s["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), size=(n, len(S)))
    m = S[idx].sum(1) / np.maximum(N[idx].sum(1), 1)
    return float((m > 0).mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def lomo(v, month):
    v = np.asarray(v, float); month = np.asarray(month)
    full = np.nanmean(v)
    vals = [np.nanmean(v[month != m]) for m in sorted(set(month)) if (month != m).sum()]
    if not vals:
        return np.nan, np.nan, False
    return float(min(vals)), float(max(vals)), bool(all(np.sign(x) == np.sign(full) for x in vals))


def conc(v, key):
    v = pd.Series(np.asarray(v, float)); k = pd.Series(np.asarray(key))
    s = v.groupby(k).sum(); tot = v.sum()
    if not len(s) or abs(tot) < 1e-12:
        return np.nan, None
    top = s.idxmax() if tot > 0 else s.idxmin()
    return float(s[top] / tot), top
