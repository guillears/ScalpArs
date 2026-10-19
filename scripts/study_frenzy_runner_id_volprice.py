#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 · Volume threshold (operator) — price the lower-threshold FRENZY_LONG entries.

Input: the volwalk shards (study_frenzy_runner_id_volwalk.py: FRESH-ON bars at T = 50 / 70 / 85 / 100, real engine status codes).
Frozen before pricing:
  entry      FRENZY_READY bars only (= a FRENZY_LONG entry at that T, today's ATR 3.0 / red bar / $20M / 2 h rules); entry at the signal
             close + 8 s (live ruler), price = first print after (ticks, else the next 1m open); refused if > 1 % from the signal close
             (frenzy_max_entry_dislocation_pct); bearish-day block approximated from BTC 5m (24 h return < 0 ∧ EMA13 < EMA50)
  episode    (pair, spike_ts) of the T walk; FIRST = the first READY bar of the episode
  reaches100 a 100x READY bar of the same pair at or after the T entry, within 96 h of the T spike (the would-be FRENZY_LONG today)
  outcomes   same walkers as the main study (study_frenzy_runner_id_build): mfe_720 before −3 (stop-first), runner >= +5,
             X0 = +1.5 / −3 / 720, R1 = half +1.5 / rest trail max(+0.2, peak − 3), R2 = −3 until +3 then max(+0.2, peak − 3)
Usage: venv/bin/python scripts/study_frenzy_runner_id_volprice.py <scratch_dir> [workers]
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"; sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
import study_frenzy_runner_id_build as B  # noqa: E402
FX = B.FX
S = sys.argv[1]
MIN = 60_000


def btc_bear():
    fr = [pd.read_csv(f, usecols=["open_time", "c"]) for f in ("reports/backtest_cache_2025/k5m_full/BTCUSDT.csv",
                                                                "reports/backtest_cache/k5m_full/BTCUSDT.csv")]
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    c = d.c.values
    e13 = pd.Series(c).ewm(span=13, adjust=False).mean().values; e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    r24 = np.r_[np.full(288, np.nan), c[288:] / c[:-288] - 1]
    return d.open_time.values.astype("int64") + 300_000, (r24 < 0) & (e13 < e50)


def price(args):
    i, pair, sc, close = args
    te = int(sc) + 8_000
    try:
        P0 = FX.load_path(pair, te, float(close), 1, {})
        if len(P0["t"]):
            # first event after te: tick price (hi == lo) or the 1m open
            # recover the raw price from the net path of the probe (net = (r − 1)·100 − 0.045 − 0.045 r)
            v = P0["op"][0] if (P0["bar"][0] and np.isfinite(P0["op"][0])) else P0["cl"][0]
            r = (v + 100 + 0.045) / (100 - 0.045)
            E = float(close) * r
        else:
            return dict(i=i, err="nopath")
        if abs(E / float(close) - 1) > 0.01:
            return dict(i=i, E=E, disloc=True)
        P = FX.load_path(pair, te, E, 720, {})
    except Exception as e:  # noqa: BLE001
        return dict(i=i, err=repr(e))
    o = dict(i=i, E=E, disloc=False, n_tick=P["n_tick"], path_end_min=(P["end_ms"] - te) / MIN)
    m, tm, st, cen, tst = B.mfe_before_stop(P, 720)
    o.update(mfe_720=m, tmfe_720=tm, stop_720=st, cens_720=cen)
    o["mfe_120"] = B.mfe_before_stop(P, 120)[0]
    o["x0"], o["x0_why"] = B.walk(P, "FIX", tp=1.5, sl=3.0, cap=720)
    o["r1"], _ = B.walk(P, "R1"); o["r2"], _ = B.walk(P, "R2")
    return o


if __name__ == "__main__":
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    V = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{S}/vw/vw_*.pkl")], ignore_index=True)
    V = V[V.ready].copy()
    bt, bear = btc_bear()
    k = np.searchsorted(bt, V.t_signal_close.values, side="right") - 1
    V["bear"] = np.where(k >= 0, bear[np.clip(k, 0, None)], False)
    V = V.sort_values(["T", "pair", "t_signal_close"]).reset_index(drop=True)
    # one pricing per (pair, bar): bars READY at several T share an entry
    U = V.drop_duplicates(["pair", "t_signal_close"])[["pair", "t_signal_close", "price"]].reset_index(drop=True)
    with Pool(nw) as p:
        res = p.map(price, [(i, r.pair, r.t_signal_close, r.price) for i, r in U.iterrows()], chunksize=10)
    Pr = U.join(pd.DataFrame(res).set_index("i"))
    V = V.merge(Pr.drop(columns=["price"]), on=["pair", "t_signal_close"], how="left")
    V.to_pickle(f"{S}/vol_priced.pkl")
    print("done", len(V), V.groupby("T").size().to_dict())
