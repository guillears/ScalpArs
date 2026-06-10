#!/usr/bin/env python3
"""Phase-1 BTC backtest — data fetch.
Pulls ~6 months of BTCUSDT perpetual 5m klines from Binance public REST
(fapi, no key) and caches to reports/btc_klines_5m_cache.csv.
Pure stdlib. Re-runnable (refetches fully, overwrites cache)."""
import json, time, urllib.request, csv, os, sys

BASE="https://fapi.binance.com/fapi/v1/klines"
SYM="BTCUSDT"; IV="5m"; LIMIT=1000
MONTHS=6
out=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"reports","btc_klines_5m_cache.csv")

def fetch(params):
    url=BASE+"?"+"&".join(f"{k}={v}" for k,v in params.items())
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())

# latest first to find "now", then page backwards
latest=fetch({"symbol":SYM,"interval":IV,"limit":2})
now_ms=latest[-1][0]
start_ms=now_ms - MONTHS*30*24*3600*1000
rows=[]; cur=start_ms
while cur < now_ms:
    batch=fetch({"symbol":SYM,"interval":IV,"limit":LIMIT,"startTime":cur})
    if not batch: break
    rows.extend(batch)
    cur=batch[-1][0]+1
    time.sleep(0.15)
    sys.stdout.write(f"\r  fetched {len(rows)} candles..."); sys.stdout.flush()
print()
# dedup by open time, sort
seen=set(); ded=[]
for b in rows:
    if b[0] in seen: continue
    seen.add(b[0]); ded.append(b)
ded.sort(key=lambda b:b[0])
with open(out,"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["open_time","open","high","low","close","volume"])
    for b in ded: w.writerow([b[0],b[1],b[2],b[3],b[4],b[5]])
import datetime
print(f"saved {len(ded)} candles -> {out}")
print(f"range: {datetime.datetime.utcfromtimestamp(ded[0][0]/1000)} -> {datetime.datetime.utcfromtimestamp(ded[-1][0]/1000)} UTC")
