#!/usr/bin/env python3
"""Fetch per-second VOLUME and TAKER-BUY volume for every cached hot episode (reports/backtest_cache/k1s_hot/*.npz, built by
hot_scalp_backtest.py) → reports/backtest_cache/k1s_hot_flow/<same name>.npz with t, q (quote volume), tb (taker-buy quote
volume), n (trade count). Binance SPOT public 1s klines; resumable (existing files are skipped).
Usage: venv/bin/python scripts/hot_flow_fetch.py"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "reports", "backtest_cache", "k1s_hot"); FLOW = os.path.join(ROOT, "reports", "backtest_cache", "k1s_hot_flow")


def get(url):
    for k in range(6):
        try:
            return json.load(urllib.request.urlopen(url, timeout=25))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return None
            time.sleep(3 + 5 * k)
        except Exception:
            time.sleep(3 + 5 * k)
    return None


if __name__ == "__main__":
    os.makedirs(FLOW, exist_ok=True); files = sorted(glob.glob(os.path.join(RAW, "*.npz"))); t0 = time.time(); bad = 0
    for n_, f in enumerate(files):
        out = os.path.join(FLOW, os.path.basename(f))
        if os.path.exists(out):
            continue
        ts = np.load(f)["t"]; pair = os.path.basename(f)[:-4].rsplit("_", 1)[0]; s, e = int(ts[0]), int(ts[-1]); rows = {}
        while s <= e:
            r = get(f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1s&startTime={s}&endTime={e}&limit=1000")
            if not r:
                break
            for x in r:
                rows[int(x[0])] = (float(x[7]), float(x[10]), int(x[8]))
            s = int(r[-1][0]) + 1000; time.sleep(0.04)
        got = np.array([rows.get(int(t), (np.nan, np.nan, -1)) for t in ts])
        if np.isnan(got[:, 0]).mean() > 0.02:
            bad += 1
        np.savez_compressed(out, t=ts, q=got[:, 0], tb=got[:, 1], n=got[:, 2])
        if n_ % 25 == 0:
            print(f"[{n_ + 1}/{len(files)}] {pair} · incomplete so far {bad} · {time.time() - t0:.0f}s", flush=True)
    print(f"done · {len(files)} episodes · incomplete {bad} · {time.time() - t0:.0f}s", flush=True)
