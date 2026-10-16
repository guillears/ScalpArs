#!/usr/bin/env python3
"""B21 review: budgeted 5m-kline fetch (public endpoint, read-only research). Same guards as study_b21_fetch_1m.py.
Usage: study_b21_fetch_5m.py START END PAIR... → reports/B21_cache/<PAIR>_5m.csv (open_time,o,h,l,c,v)"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import os, sys, time
import requests, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "reports", "backtest_cache", ".binance_cooldown_until")
OUT = os.path.join(ROOT, "reports", "B21_cache")
a, b, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
for pair in pairs:
    if os.path.exists(CD):
        try:
            if _BC.active(CD):
                print("cooldown active — stop"); sys.exit(1)
        except ValueError:
            pass
    s = int(pd.Timestamp(a, tz="UTC").timestamp() * 1000); e = int(pd.Timestamp(b, tz="UTC").timestamp() * 1000)
    rows = []
    while s < e:
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol=pair, interval="5m", startTime=s, endTime=e, limit=1000), timeout=20)
        w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
        if r.status_code in (418, 429):
            print("rate limited", r.status_code); sys.exit(2)
        r.raise_for_status(); k = r.json()
        if not k: break
        rows += k; s = k[-1][0] + 300000
        if w > 1000: time.sleep(60)
        time.sleep(0.4)
    df = pd.DataFrame([x[:6] for x in rows], columns=["open_time", "o", "h", "l", "c", "v"]).astype(float)
    df["open_time"] = df.open_time.astype("int64")
    df.to_csv(os.path.join(OUT, f"{pair}_5m.csv"), index=False); print(pair, len(df), "weight", w)
