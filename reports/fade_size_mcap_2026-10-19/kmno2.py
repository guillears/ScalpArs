import sys, time, json, urllib.request
sys.path.insert(0, "/Users/guillearslanian/Downloads/NOFA AI/scripts")
import binance_cooldown as bc
from datetime import datetime, timezone
if bc.active(): print("cooldown"); sys.exit()
def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        w = int(r.headers.get('X-MBX-USED-WEIGHT-1M') or 0)
        if w > 700: print('pause', w); time.sleep(60)
        return json.loads(r.read())
ts = lambda s: int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()*1000)
t0, t1 = ts("2026-10-17T00:28:50"), ts("2026-10-17T00:33:00")
lo, hi = 0, 27220026
while hi - lo > 500:
    mid = (lo+hi)//2
    d = get(f"https://fapi.binance.com/fapi/v1/aggTrades?symbol=KMNOUSDT&fromId={mid}&limit=1")
    if d[0]['T'] < t0: lo = mid
    else: hi = mid
    time.sleep(0.25)
out=[]; fid=lo
while True:
    d = get(f"https://fapi.binance.com/fapi/v1/aggTrades?symbol=KMNOUSDT&fromId={fid}&limit=1000")
    out += [x for x in d if t0 <= x['T'] <= t1]
    if d[-1]['T'] > t1 or len(d)<1000: break
    fid = d[-1]['a']+1; time.sleep(0.3)
json.dump(out, open("kmno_agg.json","w"))
k = get(f"https://fapi.binance.com/fapi/v1/klines?symbol=KMNOUSDT&interval=1m&startTime={ts('2026-10-17T00:15:00')}&endTime={ts('2026-10-17T00:45:00')}")
json.dump(k, open("kmno_1m.json","w"))
print(len(out), len(k))
