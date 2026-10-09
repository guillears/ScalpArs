#!/usr/bin/env python3
"""🪶 Oct-8 LITE entry-signs study — fetch the live loser's (SKLUSDT 2026-10-08 21:40 UTC) 5m + 1m klines (the year cache ends Oct 4).
Two-three sequential calls (weight 2 each), reads X-MBX-USED-WEIGHT-1M (pause > 900), stops on 418/429. Writes scratch CSVs only.
Usage: venv/bin/python scripts/study_lite_signs_fetch_skl.py <out_dir>"""
import json, os, sys, time, urllib.request
import pandas as pd

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&endTime={e}&limit={n}"
SIG_CLOSE = int(pd.Timestamp("2026-10-08 21:40", tz="UTC").timestamp() * 1000)
JOBS = [("5m", SIG_CLOSE - 1, 400), ("1m", SIG_CLOSE + 90 * 60_000, 300)]
for itv, end, n in JOBS:
    fp = os.path.join(OUT, f"SKLUSDT_{itv}.csv")
    if os.path.exists(fp):
        continue
    req = urllib.request.Request(URL.format(s="SKLUSDT", i=itv, e=end, n=n), headers={"User-Agent": "scalpars-research"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0")); data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            print("STOP: HTTP", e.code); sys.exit(2)
        raise
    d = pd.DataFrame([x[:6] + [x[7]] for x in data], columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    d.to_csv(fp, index=False)
    print(itv, len(d), "used-weight", w, flush=True)
    if w > 900:
        time.sleep(60)
    time.sleep(2.0)
