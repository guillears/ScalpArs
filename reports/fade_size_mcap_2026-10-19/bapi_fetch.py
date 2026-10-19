import sys, json, time, urllib.request, pandas as pd
R="/Users/guillearslanian/Downloads/NOFA AI"
sys.path.insert(0, R+"/scripts"); sys.path.insert(0, R)
import binance_cooldown as bc
from services.mcap_service import lookup_candidates, parse_detail
if bc.active(): sys.exit("cooldown")
B=json.load(open(R+'/reports/cache_vol_mcap/bapi_supply.json'))
m=pd.read_csv(R+'/reports/MASTER_POOL_stacked.csv',low_memory=False); b=pd.read_csv(R+'/reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv',low_memory=False)
need=set(m[m.entry_strategy=='SPIKE_FADE'].pair)|set(b[b.entry_strategy=='SPIKE_FADE'].pair)
need={p for p in need if not (B.get(p) or {}).get('cs')}
out={}
for p in sorted(need):
    got=None
    for c in lookup_candidates(p):
        try:
            with urllib.request.urlopen(f"https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/tardingPair/detail?symbol={c}", timeout=15) as r:
                j=json.loads(r.read())
        except Exception as e:
            if '418' in str(e) or '429' in str(e): sys.exit('ban '+str(e))
            continue
        d=(j or {}).get('data') or {}
        if d.get('mc') and d.get('cs'):
            got=dict(symbol=c, mc=d['mc'], cs=d['cs'], rk=d.get('rk')); break
        time.sleep(0.4)
    out[p]=got; time.sleep(0.5)
json.dump(out, open('bapi_extra.json','w'), indent=0)
print(len(need), sum(1 for v in out.values() if v))
