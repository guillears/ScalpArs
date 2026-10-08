#!/usr/bin/env python3
"""📊 Oct-8 research (YR5 halves at today's stack) — BTC regime readings rebuilt from the local 5m cache, for cohorts that carry no stamp.

Readings at an entry time t (ms, UTC), engine definitions:
  ret1d   = the last CLOSED UTC daily candle's return vs the one before (indicators.last_closed_bar_ret_pct on 1d klines)
  gap     = BTC 5m (EMA13 − EMA50) / EMA50 × 100 (engine _current_btc_trend_gap_pct; ta EMA = ewm(span, adjust=False))
  slope   = BTC 5m (EMA20 − EMA20[3 bars back]) / EMA20[3 back] × 100 (heat leg ①)
  rsi_prev= BTC 5m RSI(12) one bar back (heat leg ②)
The engine computes the 5m indicators on a window whose LAST row is the forming bar (close = live price). Two variants:
  mode="forming": the bar containing t enters with its final close (≤ 5 min look-ahead on that one row)
  mode="closed":  only bars closed at t
Validation (python scripts/study_yr5_halves_btc.py): both variants vs the stamps on yr5 fills and on real master fills.
Local cache only (reports/backtest_cache/btc_5m.csv) — no network."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B5 = os.path.join(ROOT, "reports", "backtest_cache", "btc_5m.csv")
BAR = 300_000


def _rsi(c, n=12):
    d = c.diff(); up = d.clip(lower=0); dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(); ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)


def load_btc():
    k = pd.read_csv(B5).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    c = k.c.astype(float)
    k["e13"] = c.ewm(span=13, adjust=False).mean(); k["e50"] = c.ewm(span=50, adjust=False).mean()
    k["e20"] = c.ewm(span=20, adjust=False).mean(); k["rsi"] = _rsi(c)
    k["gap"] = (k.e13 - k.e50) / k.e50 * 100
    k["slope"] = (k.e20 - k.e20.shift(3)) / k.e20.shift(3) * 100
    k["rsi_prev"] = k.rsi.shift(1)
    # daily closes = close of the 23:55 bar of each UTC day
    k["day"] = pd.to_datetime(k.open_time, unit="ms").dt.floor("D")
    dc = k.groupby("day").c.last()
    dret = (dc / dc.shift(1) - 1) * 100            # return of day D (close D vs close D−1)
    return k, dret


_K = None


def btc_at(t_ms, mode="forming"):
    """DataFrame (ret1d, gap, slope, rsi_prev) for each entry time in t_ms."""
    global _K
    if _K is None:
        _K = load_btc()
    k, dret = _K
    t = np.asarray(t_ms, dtype=np.int64)
    ot = k.open_time.values.astype(np.int64)
    # forming: the bar containing t (open ≤ t); closed: the last bar with open + 5m ≤ t
    idx = np.searchsorted(ot, t, side="right") - 1 if mode == "forming" else np.searchsorted(ot, t - BAR, side="right") - 1
    ok = (idx >= 0) & (idx < len(k))
    out = pd.DataFrame(index=range(len(t)))
    for c in ("gap", "slope", "rsi_prev"):
        v = np.full(len(t), np.nan); v[ok] = k[c].values[idx[ok]]; out[c] = v
    day = pd.to_datetime(t, unit="ms").floor("D")
    prev = day - pd.Timedelta(days=1)                  # last CLOSED daily candle = yesterday's
    out["ret1d"] = dret.reindex(prev).values
    last = pd.to_datetime(ot[-1], unit="ms")
    out.loc[pd.to_datetime(t, unit="ms") > last + pd.Timedelta(minutes=5), ["gap", "slope", "rsi_prev", "ret1d"]] = np.nan
    return out


def bearish(ret1d, gap):
    """services.frenzy.frenzy_bearish_day semantics: True both < 0 · False a readable leg ≥ 0 · None undecidable."""
    r = None if (ret1d is None or not np.isfinite(ret1d)) else ret1d
    g = None if (gap is None or not np.isfinite(gap)) else gap
    if (r is not None and r >= 0) or (g is not None and g >= 0):
        return False
    if r is None or g is None:
        return None
    return True


def _validate(df, label):
    t = pd.Series(pd.to_datetime(df.opened_at.astype(str).str[:19].str.replace("T", " ")).values.astype("datetime64[ms]").astype("int64"))
    for mode in ("forming", "closed"):
        R = btc_at(t.values, mode)
        g, r = pd.to_numeric(df.entry_btc_trend_gap_pct, errors="coerce").values, pd.to_numeric(df.entry_btc_1d_ret_pct, errors="coerce").values
        m = np.isfinite(g) & np.isfinite(R.gap.values); mr = np.isfinite(r) & np.isfinite(R.ret1d.values)
        bs = [bearish(a, b) for a, b in zip(r, g)]; br = [bearish(a, b) for a, b in zip(R.ret1d.values, R.gap.values)]
        both = [(x is not None and y is not None) for x, y in zip(bs, br)]
        agree = np.mean([x == y for x, y, z in zip(bs, br, both) if z]) if any(both) else np.nan
        line = (f"{label:<22} {mode:<8} gap r {np.corrcoef(g[m], R.gap.values[m])[0, 1]:.3f} sign-agree {np.mean(np.sign(g[m]) == np.sign(R.gap.values[m])) * 100:.1f}% (N {m.sum()}) · "
                f"ret1d max|Δ| {np.nanmax(np.abs(r[mr] - R.ret1d.values[mr])) if mr.any() else np.nan:.4f} (N {mr.sum()}) · bearish-day agree {agree * 100:.1f}% (N {sum(both)})")
        for c, s in (("slope", "entry_btc_ema20_slope"), ("rsi_prev", "entry_btc_rsi_prev")):
            if s in df:
                x = pd.to_numeric(df[s], errors="coerce").values; mm = np.isfinite(x) & np.isfinite(R[c].values)
                if mm.sum() > 10:
                    line += f" · {c} r {np.corrcoef(x[mm], R[c].values[mm])[0, 1]:.3f}"
        print(line)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import yr5_fills_trimmed as YT
    F = YT.load()
    F = F[F.sleeve.isin(["FRENZY_LONG", "FRENZY_WIDE", "MOM-long"])]
    F = F[pd.to_datetime(F.opened_at.astype(str).str[:19]) < pd.Timestamp("2026-10-03 23:00")]
    for s in ("FRENZY_LONG", "FRENZY_WIDE", "MOM-long"):
        _validate(F[F.sleeve == s], f"yr5 {s}")
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[pd.to_datetime(M.opened_at.astype(str).str[:19], errors="coerce") < pd.Timestamp("2026-10-03 23:00")]
    M = M[M.entry_btc_1d_ret_pct.notna() | M.entry_btc_trend_gap_pct.notna()]
    _validate(M, "master (live fills)")
    _validate(M[M.entry_strategy.astype(str).str.startswith("FRENZY")], "master FRENZY fills")
