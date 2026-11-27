#!/usr/bin/env python3
"""SHORT_BTC_DRAWDOWN_2026-11-27 — short sleeves vs how far BTC sits below its 30-day high (research only).
Theory: after a deep BTC drawdown the market is short-crowded and capitulated; pumped alts then squeeze shorts instead of fading
(memory: the Jun-18 → Jul-2 'washed-out' window, BTC 15–24 % below its 30 d high, every LONG sleeve won). Never tested on EXH / FB / DIST.
Variable (known at entry): BTC prior-day close ÷ max daily high of the 30 days before − 1 (fapi 1d klines). Buckets 0…−5 · −5…−10 ·
−10…−15 · ≤ −15 %; DAY counts; DESIGN / HOLDOUT; half-years. EXH = TP_GRID F7 kept (3,358) · FB = the 545 rebuild walked at today's exit.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)
BINS = [-100, -20, -15, -10, -5, 0.01]


def btc_dd():
    rows, s = [], int(pd.Timestamp("2023-05-01").value // 10**6)
    while True:
        js = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol="BTCUSDT", interval="1d", startTime=s, limit=1000),
                          headers={"User-Agent": "research-client/1.0"}, timeout=30).json()
        rows += js
        if len(js) < 1000:
            break
        s = int(js[-1][0]) + 86_400_000
    d = pd.DataFrame([(pd.to_datetime(int(r[0]), unit="ms"), float(r[2]), float(r[4])) for r in rows], columns=["d", "h", "c"]).set_index("d")
    d["dd"] = (d.c / d.h.rolling(30).max() - 1) * 100            # close of day D vs the 30-day high through D
    return d.dd.shift(1)                                           # known at the start of the next day


def tab(K, dd, name):
    K = K.assign(dd=pd.to_datetime(K.entry_ms, unit="ms").dt.floor("D").map(dd))
    K["b"] = pd.cut(K.dd, BINS)
    g = K.groupby("b", observed=True).agg(N=("net", "size"), days=("day", "nunique"), WR=("net", lambda x: (x > 0).mean() * 100), avg=("net", "mean"))
    for blk, m in (("DESIGN", K.entry_ms < HOL), ("HOLDOUT", K.entry_ms >= HOL)):
        g[blk] = K[m].groupby("b", observed=True).net.mean()
    g["halves_pos"] = [f"{(K[K.b == b].groupby('half').net.mean() > 0).sum()}/{K[K.b == b].half.nunique()}" for b in g.index]
    print(f"\n{name}\n" + g.round(2).to_string())
    return g.assign(sleeve=name)


def main():
    dd = btc_dd()
    print("BTC 30 d drawdown now:", round(float(dd.iloc[-1]), 1), "% · batch days 11-18…11-27:", dd["2026-11-18":].round(1).tolist())
    sp = importlib.util.spec_from_file_location("sa", os.path.join(HERE, "EXH_SPIKE_AGE_2026-11-26.py"))
    SA = importlib.util.module_from_spec(sp); sp.loader.exec_module(SA)
    D, _ = SA.load_bt(); K = SA.kept(D)
    out = [tab(K, dd, "EXH (TP 7)")]
    b = pd.read_csv(os.path.join(R, "reports", "FLATBASE_NOPROGRESS_STOP_2026-11-27_data", "base.csv"))
    b = b[b.src == "BT"].rename(columns={"te": "entry_ms"})
    b["day"] = b.entry_ms // 86_400_000
    t = pd.to_datetime(b.entry_ms, unit="ms"); b["half"] = t.dt.year.astype(str) + "-H" + np.where(t.dt.month <= 6, "1", "2")
    out.append(tab(b, dd, "FLATBASE (today's exit)"))
    pd.concat(out).to_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "short_btc_drawdown.csv"))


if __name__ == "__main__":
    main()
