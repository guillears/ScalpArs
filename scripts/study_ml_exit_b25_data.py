"""ML EXIT DEEP study (2026-10-26) — shared cohort + price-path loader.

Cohort (master-led, % at 1×):
  · KEEP   = every MOMENTUM LONG CLOSED non-probe fill of reports/MASTER_POOL_stacked.csv kept by today's stack (STACK 2026-10-26a,
             outcome stack_pct) + the B25 provisional fills of the latest export (rows not in the master, opened ≥ 2026-10-23 18:00,
             dedup key (opened_at, pair, direction); as traded = today's stack). Loader = scripts/study_ml_wl_b25prep_data.load_all
             (the PREP_B25_ML_WL cohort definition, reused unchanged).
  · PROBE  = is_probe momentum-long fills (B1/B2 probe era) — separate line, never pooled.
Price paths: the VALIDATED exit replica of scripts/ml_exit_optimize_yr5.py (RP) — real ticks where the day archive exists, else 1m
o→l/h→c bars. Its 1m / 5m / BTC readers are extended (read-only) with this study's top-up (reports/ML_EXIT_DEEP_2026-10-26_fetch/)
and the PREP_B25 fetch, so the newest fills (B19–B25, after the cache end) get a path too.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_exit_optimize_yr5 as RP  # noqa: E402  (the validated replica — reused, not modified)
import study_ml_wl_b25prep_data as LD  # noqa: E402

ROOT = RP.ROOT
CACHE = RP.CACHE
FETCH = os.path.join(ROOT, "reports", "ML_EXIT_DEEP_2026-10-26_fetch")
PREP = os.path.join(ROOT, "reports", "PREP_B25_ML_WL_fetch")
GROUP_ORDER = ["G0 BASE", "G1 B1", "G2 B2", "G3 B3", "G4 B4-B11", "G5 B12-B16", "G6 B17-B24", "B25 (prov.)"]


# ───────────── replica cache readers, extended with the study top-ups (monkey-patched on the RP module) ─────────────
def _k1_ext(pair):
    if pair in RP._K1:
        return RP._K1[pair]
    fr = []
    for fp in ([os.path.join(CACHE, s, f"{pair}.csv") for s in ("k1m", "k1m_ondemand")]
               + glob.glob(os.path.join(CACHE, "k1m_spike", f"{pair}*.csv"))
               + [os.path.join(PREP, "k1m", f"{pair}.csv"), os.path.join(FETCH, "k1m", f"{pair}.csv")]):
        if os.path.exists(fp):
            fr.append(pd.read_csv(fp))
    df = (pd.concat(fr).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True) if fr else None)
    if len(RP._K1) > 6:
        RP._K1.pop(next(iter(RP._K1)))
    RP._K1[pair] = df
    return df


def _k5_ext(pair):
    """RP._k5 + the PREP 5m top-up (k5m_full ends 2026-10-12 for the universe pairs)."""
    if pair in RP._K5:
        return RP._K5[pair]
    fr = []
    for fp in (os.path.join(CACHE, "k5m_full", f"{pair}.csv"), os.path.join(PREP, "k5m_full", f"{pair}.csv")):
        if os.path.exists(fp):
            fr.append(pd.read_csv(fp, usecols=["open_time", "c"]))
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


def _btc_ext():
    if RP._BTC is None:
        fr = [pd.read_csv(os.path.join(CACHE, "btc_5m.csv"))]
        for fp in (os.path.join(PREP, "k5m_full", "BTCUSDT.csv"), os.path.join(FETCH, "k5m", "BTCUSDT.csv")):
            if os.path.exists(fp):
                fr.append(pd.read_csv(fp))
        b = pd.concat(fr).drop_duplicates("open_time").sort_values("open_time")
        RP._BTC = b[["open_time", "o", "h", "l", "c"]].values.astype(float)
    return RP._BTC


RP._k1, RP._k5, RP._btc = _k1_ext, _k5_ext, _btc_ext


# ───────────── cohort ─────────────
def cohort():
    D = LD.load_all(with_features=False)
    D["probe"] = D.is_probe
    D["keep"] = (~D.probe) & D.stack_keep & D.y_keep.notna()
    D = D[D.keep | D.probe].copy()
    D["block"] = np.where(D.probe, "PROBE", np.where(D.era == "B25", "B25", "MASTER"))
    D["o_ms"] = (D.ts.values.astype("datetime64[ms]").astype(np.int64))
    D["c_ms"] = D.closed_ts.values.astype("datetime64[ms]").astype(np.int64)
    D["key"] = D.block + "|" + D.key_open + "|" + D.pair
    D["fee_in"] = np.where(D.entry_order_type.astype(str).str.upper().str.startswith("MAKER"), RP.MAKER, RP.TAKER)
    D["E"] = pd.to_numeric(D.entry_price, errors="coerce")
    D["X"] = pd.to_numeric(D.exit_price, errors="coerce")
    D["atr"] = pd.to_numeric(D.entry_atr_pct, errors="coerce")
    D["rsi_stamp"] = pd.to_numeric(D.entry_btc_rsi_closed, errors="coerce")
    D["y"] = np.where(D.keep, D.y_keep, D.y_trad)          # outcome % at 1× (KEEP: stack_pct; probe: as traded)
    D["reason_live"] = D.close_reason.astype(str).map(RP._base_reason)
    D["notional"] = pd.to_numeric(D.notional_value, errors="coerce")
    for c in ("peak_pnl", "trough_pnl"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D["sbull"] = D.entry_btc_regime.astype(str).eq("STRONG_BULL")
    return D.sort_values("ts").reset_index(drop=True)


def fill_dict(r):
    """replica fill input; entry BTC RSI = live stamp when present, else the live ruler rebuilt at entry (RP.btc_rsi_at)."""
    rsi = r.rsi_stamp if pd.notna(r.rsi_stamp) else RP.btc_rsi_at(int(r.o_ms))
    return dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.E), fee_in=float(r.fee_in),
                atr=float(r.atr) if pd.notna(r.atr) and r.atr > 0 else None, rsi_entry=None if rsi is None else float(rsi))


def coverage(C):
    rows = []
    for r in C.itertuples():
        ds = r.ts.strftime("%Y-%m-%d")
        tick = any(os.path.exists(os.path.join(CACHE, s, r.pair, f"{ds}.npz")) for s in ("ticks_q", "ticks"))
        k1 = _k1_ext(r.pair)
        n = 0 if k1 is None else int(((k1.open_time >= r.o_ms - 60_000) & (k1.open_time < r.o_ms + 9 * 3600_000)).sum())
        rows.append(dict(key=r.key, block=r.block, era=r.era, pair=r.pair, o_ms=r.o_ms, tick=tick, n1m=n))
    return pd.DataFrame(rows)
