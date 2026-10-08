#!/usr/bin/env python3
"""SPIKE_FADE recall trace — step 2: real trades (t, price, qty) for every signal pair-day the replay cache lacks.

Cache first: a pair-day already in reports/backtest_cache/ticks_q/ is never fetched. Missing ones come from the public Binance
USDT-M aggTrades DAILY ARCHIVES (data.binance.vision — static files, no API weight; same source and format as
scripts/backtest_fetch_ticks.py --qty) and are written to reports/study_fade_trace_ticks/<PAIR>/<date>.npz — NOT into the replay
cache, so the replay inputs are untouched. Serial (one request at a time), 0.3 s pause between files, resumable.
Usage: venv/bin/python scripts/study_fade_trace_fetch.py
"""
import csv, io, os, sys, time, urllib.request, zipfile
from urllib.parse import quote
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "study_fade_trace_ticks")


def one(pair, date):
    fp = os.path.join(OUT, pair, f"{date}.npz")
    if os.path.exists(fp) or os.path.exists(os.path.join(REP, "backtest_cache", "ticks_q", pair, f"{date}.npz")):
        return "skip"
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{quote(pair)}/{quote(pair)}-aggTrades-{date}.zip"
    data = None
    for i in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            break
        except Exception as ex:
            if "404" in str(ex):
                return f"MISSING {pair} {date}"
            if i == 2:
                return f"FAIL {pair} {date}: {ex}"
            time.sleep(3)
    z = zipfile.ZipFile(io.BytesIO(data))
    t, p, q = [], [], []
    with z.open(z.namelist()[0]) as f:
        for row in csv.reader(io.TextIOWrapper(f, encoding="utf-8")):
            if not row or not row[0].isdigit():
                continue
            t.append(int(row[5])); p.append(float(row[1])); q.append(float(row[2]))
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = fp[:-4] + ".part.npz"
    np.savez_compressed(tmp, t=np.array(t, dtype=np.int64), p=np.array(p, dtype=np.float32), q=np.array(q, dtype=np.float32))
    os.replace(tmp, fp)
    return f"{pair} {date} {len(t)} trades {len(data) / 1e6:.1f} MB"


def main():
    PD = pd.read_csv(os.path.join(REP, "study_fade_trace_pairdays.csv"))
    n = 0
    for r in PD.itertuples():
        msg = one(r.pair, r.date)
        if msg != "skip":
            n += 1
            print(n, msg, flush=True)
            time.sleep(0.3)
    print("done")


if __name__ == "__main__":
    main()
