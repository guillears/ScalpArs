import sys, time, json, urllib.request
sys.path.insert(0, "/Users/guillearslanian/Downloads/NOFA AI/scripts")
import binance_cooldown as bc
from datetime import datetime, timezone
if bc.active(): print("cooldown active"); sys.exit()
def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        w = r.headers.get('X-MBX-USED-WEIGHT-1M'); print('weight', w, file=sys.stderr)
        if w and int(w) > 700: time.sleep(60)
        return json.loads(r.read())
ts = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)
t0, t1 = ts("2026-10-17T00:28:30"), ts("2026-10-17T00:33:00")
out=[]; start=t0
while start < t1:
    d = get(f"https://fapi.binance.com/fapi/v1/aggTrades?symbol=KMNOUSDT&startTime={start}&endTime={min(start+60000,t1)}&limit=1000")
    out += d
    start = (d[-1]['T']+1) if len(d)==1000 else start+60000
    time.sleep(0.3)
json.dump(out, open("kmno_agg.json","w"))
k = get(f"https://fapi.binance.com/fapi/v1/klines?symbol=KMNOUSDT&interval=1m&startTime={ts('2026-10-17T00:15:00')}&endTime={ts('2026-10-17T00:45:00')}")
json.dump(k, open("kmno_1m.json","w"))
print(len(out), len(k))
