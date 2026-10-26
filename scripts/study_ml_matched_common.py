"""ML MATCHED-LONG SHADOW study (2026-10-26) — shared helpers (cache-only, no network, read-only on every input).

  · pattern labels  = the ENGINE's own services.trading_engine._compute_pattern_c_match / _compute_pattern_w_match (LONG, today's
                      thresholds from trading_config.json) — never a re-implementation.
  · pair fields     = derived from the journalled indicator snapshot (ctx) with the engine's own formulas (open_position callers,
                      services/trading_engine.py ~L12088-12103 / L22250-22265).
  · BTC fields      = services.indicators.calculate_indicators on the last 100 BTC 5m bars at the scan (closed bars + the forming bar
                      rebuilt from closed 1m bars) — the engine's BTC block (trading_engine.py ~L20120-20145). Parity is MEASURED against
                      the journalled SCAN btc_rsi / btc_adx and against stamped fills (never assumed).
  · exit pricing    = the VALIDATED ML exit replica scripts/ml_exit_optimize_yr5.py (RP.BASE = today's live ML exit; DL 380: 100 %
                      exit-reason match on the master) through scripts/study_ml_exit_b25_data.py (its cache top-ups), extended here
                      read-only with the 2025 cache (reports/backtest_cache_2025) and this study's own fetch dir.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_ml_exit_b25_data as DD  # noqa: E402  (patches RP's readers with the DL 380 top-ups)

RP = DD.RP
PFX = os.path.join(ROOT, "reports", "ML_MATCHED_SHADOW_2026-10-26")
FETCH = PFX + "_fetch"
C25 = os.path.join(ROOT, "reports", "backtest_cache_2025")
C_L = [f"C{i}" for i in range(1, 10)]
W_L = [f"W{i}" for i in range(1, 7)]
BE_WR = 61.5          # ML breakeven WR (sleeve, Sep-25 / PREREG)


# ───────────── replica readers: + 2025 cache + this study's fetch (read-only; patched on the RP module) ─────────────
_k1_dd, _k5_dd = RP._k1, RP._k5


def _day_ext(pair, day_ms):
    key = (pair, day_ms)
    if key in RP._DAYS:
        return RP._DAYS[key]
    ds = pd.Timestamp(day_ms, unit="ms").strftime("%Y-%m-%d")
    out = None
    for base in (RP.CACHE, C25):
        for sub in ("ticks_q", "ticks"):
            fp = os.path.join(base, sub, pair, f"{ds}.npz")
            if os.path.exists(fp):
                try:
                    z = np.load(fp); out = (z["t"].astype(np.int64), z["p"].astype(np.float64)); break
                except Exception:
                    out = None
        if out is not None:
            break
    if len(RP._DAYS) > 12:
        RP._DAYS.pop(next(iter(RP._DAYS)))
    RP._DAYS[key] = out
    return out


def _k1_ext(pair):
    if pair in RP._K1:
        return RP._K1[pair]
    fr = []
    for fp in ([os.path.join(RP.CACHE, s, f"{pair}.csv") for s in ("k1m", "k1m_ondemand")]
               + glob.glob(os.path.join(RP.CACHE, "k1m_spike", f"{pair}*.csv"))
               + [os.path.join(C25, s, f"{pair}.csv") for s in ("k1m", "k1m_ondemand")]
               + glob.glob(os.path.join(C25, "k1m_spike", f"{pair}*.csv"))
               + [os.path.join(DD.PREP, "k1m", f"{pair}.csv"), os.path.join(DD.FETCH, "k1m", f"{pair}.csv"),
                  os.path.join(FETCH, "k1m", f"{pair}.csv")]):
        if os.path.exists(fp):
            try:
                fr.append(pd.read_csv(fp, usecols=["open_time", "o", "h", "l", "c"]))
            except Exception:
                pass
    df = (pd.concat(fr).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True) if fr else None)
    if len(RP._K1) > 6:
        RP._K1.pop(next(iter(RP._K1)))
    RP._K1[pair] = df
    return df


def _k5_ext(pair):
    if pair in RP._K5:
        return RP._K5[pair]
    fr = []
    for fp in (os.path.join(RP.CACHE, "k5m_full", f"{pair}.csv"), os.path.join(C25, "k5m_full", f"{pair}.csv"),
               os.path.join(DD.PREP, "k5m_full", f"{pair}.csv"), os.path.join(FETCH, "k5m", f"{pair}.csv")):
        if os.path.exists(fp):
            try:
                fr.append(pd.read_csv(fp, usecols=["open_time", "c"]))
            except Exception:
                pass
    df = pd.concat(fr) if fr else None
    k1 = _k1_ext(pair)
    if k1 is not None and len(k1):
        g = k1.assign(b=(k1.open_time // 300_000) * 300_000).groupby("b").c.last().reset_index().rename(columns={"b": "open_time"})
        df = g if df is None else pd.concat([df, g[~g.open_time.isin(df.open_time)]])
    if df is None or not len(df):
        RP._K5[pair] = None
        return None
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    c = df.c.astype(float)
    out = dict(ot=df.open_time.values.astype(np.int64), c=c.values)
    for n in (5, 8, 13, 20):
        out[f"e{n}"] = c.ewm(span=n, adjust=False).mean().values
    if len(RP._K5) > 6:
        RP._K5.pop(next(iter(RP._K5)))
    RP._K5[pair] = out
    return out


def _btc5_all():
    fr = []
    for fp in (os.path.join(C25, "k5m_full", "BTCUSDT.csv"), os.path.join(RP.CACHE, "btc_5m.csv"),
               os.path.join(RP.CACHE, "k5m_full", "BTCUSDT.csv"), os.path.join(DD.PREP, "k5m_full", "BTCUSDT.csv"),
               os.path.join(DD.FETCH, "k5m", "BTCUSDT.csv"), os.path.join(FETCH, "k5m", "BTCUSDT.csv")):
        if os.path.exists(fp):
            fr.append(pd.read_csv(fp, usecols=["open_time", "o", "h", "l", "c"]))
    return pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)


def _btc_ext():
    if RP._BTC is None:
        RP._BTC = _btc5_all()[["open_time", "o", "h", "l", "c"]].values.astype(float)
    return RP._BTC


RP._day, RP._k1, RP._k5, RP._btc = _day_ext, _k1_ext, _k5_ext, _btc_ext


# ───────────── engine pattern labels ─────────────
def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def labels(rng_pos, pair_gap, adx_delta, btc_rsi, btc_rsi_prev, btc_adx, btc_adx_prev, btc_gap, stretch, pair_adx, btc_atr,
           ema20_slope, ema50_slope, rsi, pair_vol_ratio):
    from services.trading_engine import _compute_pattern_c_match, _compute_pattern_w_match
    a = {k: _f(v) for k, v in dict(rng_pos=rng_pos, pair_gap=pair_gap, adx_delta=adx_delta, btc_rsi=btc_rsi, btc_rsi_prev=btc_rsi_prev,
                                    btc_adx=btc_adx, btc_adx_prev=btc_adx_prev, btc_gap=btc_gap, stretch=stretch, pair_adx=pair_adx,
                                    btc_atr=btc_atr, ema20_slope=ema20_slope, ema50_slope=ema50_slope, rsi=rsi,
                                    pair_vol_ratio=pair_vol_ratio).items()}
    c = _compute_pattern_c_match("LONG", a["rng_pos"], a["pair_gap"], a["adx_delta"], a["btc_rsi"], a["btc_rsi_prev"], a["btc_adx"],
                                 a["btc_adx_prev"], a["btc_gap"], a["stretch"], a["pair_adx"], a["btc_atr"],
                                 ema20_slope=a["ema20_slope"], ema50_slope=a["ema50_slope"])
    w = _compute_pattern_w_match("LONG", a["rsi"], a["pair_adx"], a["adx_delta"], a["stretch"], a["rng_pos"], a["pair_gap"],
                                 a["btc_rsi"], a["btc_adx"], a["btc_atr"], a["btc_gap"], a["pair_vol_ratio"])
    on = [n for n, m in zip(C_L, c[:9]) if m] + [n for n, m in zip(W_L, w[:6]) if m]
    return "+".join(on)


def pair_fields_from_ctx(x, px=None):
    """engine formulas on the journalled indicator snapshot (price = ctx price unless px given)."""
    g = lambda k: _f(x.get("ctx_" + k))
    p = px if px is not None else g("price")
    e5, e13, e20, e50 = g("ema5"), g("ema13"), g("ema20"), g("ema50")
    e20p3, e50p12, hi, lo = g("ema20_prev3"), g("ema50_prev12"), g("high_20"), g("low_20")
    adx, adxp = g("adx"), g("adx_prev1")
    vol, avg = g("volume"), g("avg_volume")
    atr = g("atr")
    o = dict(price=p, rsi=g("rsi"), rsi_prev2=g("rsi_prev2"), pair_adx=adx)
    o["adx_delta"] = (adx - adxp) if adx is not None and adxp is not None else None
    o["stretch"] = abs(p - e5) / p * 100 if p and e5 else None
    o["pair_gap"] = (e13 - e50) / e50 * 100 if e13 and e50 else None
    o["rng_pos"] = (p - lo) / (hi - lo) * 100 if hi and lo and hi != lo and p else None
    o["ema20_slope"] = (e20 - e20p3) / e20p3 * 100 if e20 and e20p3 else None
    o["ema50_slope"] = (e50 - e50p12) / e50p12 * 100 if e50 and e50p12 else None
    o["pair_vol_ratio"] = (vol or 0) / avg if avg else None
    o["atr_pct"] = atr / p * 100 if atr is not None and p else None
    o["dist_ema13"] = (p - e13) / e13 * 100 if e13 and p else None
    return o


# ───────────── BTC state at a scan (engine BTC block on 100 5m bars, forming bar from closed 1m bars) ─────────────
class BtcState:
    def __init__(self):
        b = _btc5_all()
        self.t5 = b.open_time.values.astype(np.int64)
        self.b5 = b[["o", "h", "l", "c"]].values.astype(float)
        fr = []
        for fp in (os.path.join(C25, "k1m", "BTCUSDT.csv"), os.path.join(RP.CACHE, "k1m", "BTCUSDT.csv"),
                   os.path.join(DD.FETCH, "k1m", "BTCUSDT.csv"), os.path.join(FETCH, "k1m", "BTCUSDT.csv")):
            if os.path.exists(fp):
                fr.append(pd.read_csv(fp, usecols=["open_time", "o", "h", "l", "c"]))
        k1 = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time")
        self.t1 = k1.open_time.values.astype(np.int64)
        self.b1 = k1[["o", "h", "l", "c"]].values.astype(float)
        self.cache = {}

    def at(self, t_ms):
        key = int(t_ms // 1000)
        if key in self.cache:
            return self.cache[key]
        from services.indicators import calculate_indicators
        cur5 = (t_ms // 300_000) * 300_000
        j = int(np.searchsorted(self.t5, cur5))           # closed bars < cur5
        bars = [[int(self.t5[i]), *self.b5[i], 1.0] for i in range(max(0, j - 99), j)]
        a = int(np.searchsorted(self.t1, cur5)); b = int(np.searchsorted(self.t1, t_ms - 60_000, side="right"))
        if b > a:   # forming 5m bar from the CLOSED 1m bars inside it
            m = self.b1[a:b]
            bars.append([int(cur5), m[0, 0], m[:, 1].max(), m[:, 2].min(), m[-1, 3], 1.0])
        elif bars:
            bars.append([int(cur5), bars[-1][4], bars[-1][4], bars[-1][4], bars[-1][4], 1.0])
        out = dict(btc_rsi=None, btc_rsi_prev=None, btc_adx=None, btc_adx_prev=None, btc_atr=None, btc_gap=None, btc_slope=None)
        try:
            ind = calculate_indicators(bars[-100:])
        except Exception:
            ind = None
        if ind:
            px = ind.get("price")
            e13, e50, e20, e20p3 = ind.get("ema13"), ind.get("ema50"), ind.get("ema20"), ind.get("ema20_prev3")
            out = dict(btc_rsi=ind.get("rsi"), btc_rsi_prev=ind.get("rsi_prev1"), btc_adx=ind.get("adx"), btc_adx_prev=ind.get("adx_prev1"),
                       btc_atr=(ind.get("atr") / px * 100) if ind.get("atr") is not None and px else None,
                       btc_gap=((e13 - e50) / e50 * 100) if e13 and e50 else None,
                       btc_slope=((e20 - e20p3) / e20p3 * 100) if e20 and e20p3 else None)
        self.cache[key] = out
        return out


# ───────────── pricing ─────────────
def price(fills, jobs=8):
    """fills: list of dicts (key, pair, o_ms, E, fee_in, atr, rsi_entry) → DataFrame key, pct, reason, exit_ms, src, peak (RP.BASE)."""
    if not fills:
        return pd.DataFrame(columns=["key", "pct", "reason", "exit_ms", "src", "peak"])
    R = RP.run_many(fills, {"BASE": RP.BASE}, jobs)
    return R[R.variant == "BASE"].drop(columns="variant")


def episodes(df, t_col="o_ms", exit_col="exit_ms", group=("pair",), cooldown_ms=0):
    """one shadow position per pair at a time: a refusal while the previous shadow on that pair is still open (by the replica's exit)
    is dropped (the engine would hold the position, and the refusal would not re-open it)."""
    keep = np.zeros(len(df), bool)
    df = df.reset_index(drop=True)
    for _, g in df.sort_values(t_col).groupby(list(group), sort=False):
        busy = -1
        for i, r in g.iterrows():
            if r[t_col] >= busy:
                keep[i] = True
                e = r[exit_col]
                busy = (int(e) if pd.notna(e) else int(r[t_col]) + 3 * 3600_000) + cooldown_ms
    return df[keep].copy()


def day_boot(y, day, R=4000, seed=20261026):
    """day-block bootstrap of the mean %/fill: (lo95 one-sided 5 %, hi, P(mean ≤ 0))."""
    y = np.asarray(y, float); day = np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 2:
        return np.nan, np.nan, np.nan
    s = np.bincount(inv, y); n = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (R, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95)), float((m <= 0).mean())


def day_boot_lb(y, day, alpha, R=4000, seed=20261026):
    y = np.asarray(y, float); day = np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 2:
        return np.nan
    s = np.bincount(inv, y); n = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (R, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(m, alpha))
