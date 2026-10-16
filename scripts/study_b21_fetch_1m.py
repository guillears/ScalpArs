#!/usr/bin/env python3
"""B21 review: budgeted 1m-kline fetch for the few B21 pair-windows not in the cache (read-only research; public endpoint).
Respects reports/backtest_cache/.binance_cooldown_until, reads X-MBX-USED-WEIGHT-1M, stops on 418/429. Output reports/B21_cache/<PAIR>_1m.csv"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import os, sys, time, json
import requests, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "reports", "backtest_cache", ".binance_cooldown_until")
OUT = os.path.join(ROOT, "reports", "B21_cache")
JOBS = [a.split(":") for a in sys.argv[1:]]   # PAIR:startISO:endISO
for pair, a, b in JOBS:
    if os.path.exists(CD):
        try:
            if _BC.active(CD):
                print("cooldown active — stop"); sys.exit(1)
        except ValueError:
            pass
    s = int(pd.Timestamp(a.replace("_", ":"), tz="UTC").timestamp() * 1000); e = int(pd.Timestamp(b.replace("_", ":"), tz="UTC").timestamp() * 1000)
    rows = []
    while s < e:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=pair, interval="1m", startTime=s, endTime=e, limit=1000), timeout=20)
        w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
        if r.status_code in (418, 429):
            print("rate limited", r.status_code); sys.exit(2)
        r.raise_for_status(); k = r.json()
        if not k: break
        rows += k; s = k[-1][0] + 60000
        print(pair, len(rows), "weight", w)
        if w > 1000: time.sleep(60)
        time.sleep(0.5)
    df = pd.DataFrame([x[:6] for x in rows], columns=["t", "o", "h", "l", "c", "v"]).astype(float)
    df.to_csv(os.path.join(OUT, f"{pair}_1m.csv"), index=False)
