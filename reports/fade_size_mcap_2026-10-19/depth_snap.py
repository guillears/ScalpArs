import sys, json, time, urllib.request, urllib.error, pandas as pd, numpy as np
R="/Users/guillearslanian/Downloads/NOFA AI"; sys.path.insert(0, R+"/scripts"); import binance_cooldown as bc
def get(url):
    if bc.active(): sys.exit('cooldown')
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            w=int(r.headers.get('X-MBX-USED-WEIGHT-1M') or 0)
            if w>700: print('pause at',w,flush=True); time.sleep(62)
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418,429): sys.exit(f'stop {e.code}')
        raise
m=pd.read_csv(R+'/reports/MASTER_POOL_stacked.csv',low_memory=False); b=pd.read_csv(R+'/reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv',low_memory=False)
pairs=set(m[m.entry_strategy=='SPIKE_FADE'].pair)|set(b[b.entry_strategy=='SPIKE_FADE'].pair)|{'KMNOUSDT'}
tk={t['symbol']:t for t in get("https://fapi.binance.com/fapi/v1/ticker/24hr")}
rows=[]
for p in sorted(pairs):
    if p not in tk or not p.isascii(): continue
    d=get(f"https://fapi.binance.com/fapi/v1/depth?symbol={p}&limit=500"); time.sleep(1.1)
    bids=np.array([[float(a),float(q)] for a,q in d['bids']]); asks=np.array([[float(a),float(q)] for a,q in d['asks']])
    if len(bids)<5 or len(asks)<5: continue
    mid=(bids[0,0]+asks[0,0])/2
    r=dict(pair=p, vol24=float(tk[p]['quoteVolume']), last=float(tk[p]['lastPrice']), spread_pct=(asks[0,0]-bids[0,0])/mid*100)
    for side,lv in (('buy',asks),('sell',bids)):
        usd=lv[:,0]*lv[:,1]; cu=np.cumsum(usd)
        for N in (1500,3000,6000,12000,24000):
            if cu[-1]<N: r[f'{side}_{N}']=np.nan; continue
            i=np.searchsorted(cu,N); q=lv[:i,1].sum()+(N-(cu[i-1] if i else 0))/lv[i,0]; v=N/q
            r[f'{side}_{N}']=abs(v/mid-1)*100
        for band in (0.5,1.0):
            sel=np.abs(lv[:,0]/mid-1)*100<=band; r[f'{side}_depth_{band}']=usd[sel].sum()
    rows.append(r)
pd.DataFrame(rows).to_csv('depth_snap.csv',index=False); print(len(rows))
