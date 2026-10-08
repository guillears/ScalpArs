#!/usr/bin/env python3
"""🎯 Oct-8 study (+3 vs +4) — fetch aggTrades for a FEW live-fill windows on today's UTC day (the daily archive is not published yet).
Sequential, ≤ 1 call / 2 s, reads X-MBX-USED-WEIGHT-1M and pauses above 900, stops on 418 / 429. Writes scratch npz only (never the
shared tick cache: a partial day there would be trusted as complete).
Usage: venv/bin/python scripts/study_tp34_fetch_api.py <out_dir> PAIR:start_ms:end_ms ..."""
import json, os, sys, time, urllib.request
import numpy as np

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
URL = "https://fapi.binance.com/fapi/v1/aggTrades?symbol={s}&startTime={a}&endTime={b}&limit=1000"
used_total = 0
for spec in sys.argv[2:]:
    pair, a, b = spec.split(":"); a, b = int(a), int(b)
    fp = os.path.join(OUT, f"{pair}_{a}_{b}.npz")
    if os.path.exists(fp):
        continue
    T, P = [], []
    cur = a
    while cur < b:
        end = min(b, cur + 3600_000 - 1)
        req = urllib.request.Request(URL.format(s=pair, a=cur, b=end), headers={"User-Agent": "scalpars-research"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0")); data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                print("STOP: HTTP", e.code); sys.exit(2)
            raise
        used_total += 20
        if data:
            T += [int(x["T"]) for x in data]; P += [float(x["p"]) for x in data]
        print(f"{pair} {cur} n={len(data)} used-weight={w}", flush=True)
        if w > 900:
            print("pause 60 s (weight > 900)"); time.sleep(60)
        if len(data) == 1000:
            cur = int(data[-1]["T"]) + 1           # same ms prints split across pages are rare; aggTrades are aggregated per ms/price/side
        else:
            cur = end + 1
        time.sleep(2.0)
    np.savez_compressed(fp, t=np.array(T, dtype=np.int64), p=np.array(P, dtype=np.float64))
print("approx weight spent:", used_total)
