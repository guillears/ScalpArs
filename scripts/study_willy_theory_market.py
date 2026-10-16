#!/usr/bin/env python3
"""WILLY THEORY — W6 / W6b market features at each rebuilt WILLY entry (outcome-blind) + the post-entry market path (descriptive only).

At the entry (red) 5m bar T (the entry is ≈ its close + 12 s):
  W6   btc_1h_slope  (scout_willy_btc1h.slope_at — the live WILLY_BTC1H definition, imported) · btc_15m (last 3 closed 5m bars %) ·
       btc_1h (last 12 bars %) · brd_1h (share of the day's top-50 with a positive 12-bar return)
  W6b  btc_bar / eth_bar / sol_bar (the SAME 5m bar's return %) · brd_red (share of the top-50 red on the same bar) · mkt_tbs (BTC + ETH +
       SOL 1m taker-buy share of quote volume over the 5 one-minute bars of T, reports/ml_btcfwd_fetch/k1m_taker)
  POST btc_f5 / f15 / f30 (BTC close-to-close from T's close) · brd_f5 / f15 / f30 (top-50 equal-weight mean return) — after entry,
       NEVER an entry filter.
top-50 = reports/backtest_cache(_2025)/universe_daily_top50.csv of the PREVIOUS UTC day (no look-ahead). Output market.pkl.
"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402
import scout_willy_btc1h as B1  # noqa: E402

BAR = W.BAR_MS
OFFS = (-12, -3, 0, 1, 3, 6)          # bar offsets whose CLOSE is needed (close of T+k)


def top50_map():
    U = pd.concat([pd.read_csv(os.path.join(W.REP, f)) for f in ("backtest_cache_2025/universe_daily_top50.csv", "backtest_cache/universe_daily_top50.csv")])
    U = U.drop_duplicates(["date_ms", "pair"])
    return {int(d): set(g.pair) for d, g in U.groupby("date_ms")}


def closes(pair, Ts):
    d = W.load_5m(pair)
    if d is None:
        return pair, None
    s_c = pd.Series(d.c.values.astype(float), index=d.open_time.values.astype("int64"))
    s_o = pd.Series(d.o.values.astype(float), index=d.open_time.values.astype("int64"))
    Ts = np.asarray(Ts, dtype="int64")
    out = {f"c{k}": s_c.reindex(Ts + k * BAR).values for k in OFFS}
    out["o0"] = s_o.reindex(Ts).values
    return pair, pd.DataFrame(dict(T=Ts, **out))


def _job(a):
    return closes(*a)


def main():
    S = pd.read_pickle(W.SIG)
    S = S[S.T_ent.notna()].copy()
    S["T"] = S.T_ent.astype("int64")
    Ts = np.sort(S["T"].unique())
    tm = top50_map()
    days = sorted(tm)
    need = {}
    tday = {}
    for T in Ts:
        dprev = (T // W.DAY_MS) * W.DAY_MS - W.DAY_MS
        k = np.searchsorted(days, dprev, side="right") - 1
        if k < 0:
            continue
        tday[T] = days[k]
        for p in tm[days[k]]:
            need.setdefault(p, []).append(T)
    for m in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        need[m] = sorted(set(need.get(m, [])) | set(Ts.tolist()))
    with Pool(int(os.environ.get("JOBS", "8"))) as P:
        res = dict(P.map(_job, list(need.items()), chunksize=4))
    print(f"top-50 pairs loaded {len(res)} · missing {sum(v is None for v in res.values())}", flush=True)
    rows = {T: dict(T=T) for T in Ts}
    for m, nm in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol")):
        R = res[m].set_index("T")
        for T in Ts:
            x = R.loc[T]
            r = rows[T]
            r[f"{nm}_bar"] = (x.c0 / x.o0 - 1) * 100
            if nm == "btc":
                r["btc_15m"] = (x.c0 / x["c-3"] - 1) * 100
                r["btc_1h"] = (x.c0 / x["c-12"] - 1) * 100
                for k, lab in ((1, "f5"), (3, "f15"), (6, "f30")):
                    r[f"btc_{lab}"] = (x[f"c{k}"] / x.c0 - 1) * 100
    # breadth
    acc = {T: dict(red=[], up1h=[], f5=[], f15=[], f30=[]) for T in Ts}
    for p, df in res.items():
        if df is None or p not in need:
            continue
        Tv = df["T"].values; c0 = df.c0.values; o0 = df.o0.values; cm12 = df["c-12"].values
        f = {"f5": df.c1.values, "f15": df.c3.values, "f30": df.c6.values}
        for i in range(len(df)):
            T = int(Tv[i])
            if T not in tday or p not in tm[tday[T]]:
                continue
            a = acc[T]
            if np.isfinite(c0[i]) and np.isfinite(o0[i]):
                a["red"].append(c0[i] < o0[i])
            if np.isfinite(c0[i]) and np.isfinite(cm12[i]):
                a["up1h"].append(c0[i] > cm12[i])
            for lab, arr in f.items():
                if np.isfinite(c0[i]) and np.isfinite(arr[i]):
                    a[lab].append((arr[i] / c0[i] - 1) * 100)
    for T in Ts:
        a = acc[T]; r = rows[T]
        r["brd_n"] = len(a["red"])
        r["brd_red"] = np.mean(a["red"]) * 100 if len(a["red"]) >= 20 else np.nan
        r["brd_1h"] = np.mean(a["up1h"]) * 100 if len(a["up1h"]) >= 20 else np.nan
        for lab in ("f5", "f15", "f30"):
            r[f"brd_{lab}"] = np.mean(a[lab]) if len(a[lab]) >= 20 else np.nan
    M = pd.DataFrame(list(rows.values()))
    # BTC 1h EMA20 slope (the live WILLY_BTC1H definition)
    hb = B1.hourly(W.load_5m("BTCUSDT")[["open_time", "c"]])
    te = M["T"].values + BAR + W.T_LAG_MS
    M["btc_1h_slope"], M["btc_1h_slope_st"] = B1.slope_at(hb, te)
    # market 1m taker-buy share (BTC + ETH + SOL) over T's five 1m bars
    tb = None
    for m in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        k = pd.read_csv(os.path.join(W.REP, "ml_btcfwd_fetch", "k1m_taker", f"{m}.csv"), usecols=["open_time", "qvol", "tbq"])
        k["T"] = (k.open_time // BAR) * BAR
        g = k.groupby("T")[["qvol", "tbq"]].sum()
        tb = g if tb is None else tb.add(g, fill_value=0)
    tb["mkt_tbs"] = tb.tbq / tb.qvol * 100
    M = M.merge(tb[["mkt_tbs"]].reset_index(), on="T", how="left")
    M.to_pickle(os.path.join(W.OUT, "market.pkl"))
    print(M.describe().T[["count", "mean", "50%"]].to_string())


if __name__ == "__main__":
    main()
