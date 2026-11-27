#!/usr/bin/env python3
"""FLATBASE rebuild through 2026-11-27 — 5m kline top-up (research only). Universe = argv[1] (JSON list of pairs: every coin the bot's
decisions journal tracked for FRENZY / EXH / DIST / WILLY / FLATBASE since 11-12 + the 8 live Flatbase coins). Each pair's
reports/OOS_REAL_FLATBASE_data/k5m/<PAIR>.csv is extended from its last cached bar (any rebuild source; ≥ 2026-10-01 when uncached) to now.
Budget: public fapi klines, read X-MBX-USED-WEIGHT-1M on every call, pause above 600, stop on 418 / 429; generic User-Agent.
"""
import json
import os
import sys
import time

import pandas as pd
import requests

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(R, "scripts"))
import study_frenzy_swing_short_walk as W  # noqa: E402
W.SRC = W.SRC + [os.path.join(R, "reports", d, "k5m") for d in ("FRENZY_FLATBASE_SHORT_2026-10-22_data", "POST_FRENZY_PATHS_2026-11-17_data",
                                                                 "OOS_REAL_FLATBASE_data")]
OUT = os.path.join(R, "reports", "OOS_REAL_FLATBASE_data", "k5m")
BAR = 300_000


def last_cached(p):
    m = None
    for d in W.SRC:
        f = os.path.join(d, p + ".csv")
        if os.path.exists(f):
            x = pd.read_csv(f, usecols=["open_time"])
            if len(x):
                m = max(m or 0, int(x.open_time.max()))
    return m


def main():
    pairs = json.load(open(sys.argv[1]))
    now = int(time.time() * 1000) // BAR * BAR - BAR
    for k, p in enumerate(pairs):
        s = last_cached(p)
        s = (s + BAR) if s else int(pd.Timestamp("2026-10-01").value // 10**6)
        rows = []
        while s < now:
            r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=p, interval="5m", startTime=s, limit=1000),
                             headers={"User-Agent": "research-client/1.0"}, timeout=30)
            if r.status_code in (418, 429):
                sys.exit(f"STOP HTTP {r.status_code}")
            if r.status_code == 400:
                break
            r.raise_for_status()
            js = r.json()
            if not js:
                break
            rows += [dict(open_time=int(a[0]), o=float(a[1]), h=float(a[2]), l=float(a[3]), c=float(a[4]), vol=float(a[5]), qvol=float(a[7])) for a in js
                     if int(a[0]) <= now]
            s = int(js[-1][0]) + BAR
            if int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0")) > 600:
                time.sleep(61 - time.time() % 60)
            if len(js) < 1000:
                break
        if rows:
            f = os.path.join(OUT, p + ".csv")
            old = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
            new = pd.DataFrame(rows)
            if len(old):
                new = new[[c for c in old.columns if c in new.columns]]
            pd.concat([old, new]).drop_duplicates("open_time", keep="last").sort_values("open_time").to_csv(f, index=False)
        print(f"{k + 1}/{len(pairs)} {p} +{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
