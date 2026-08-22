import pandas as pd, numpy as np, sys
S="reports/backtest_cache/bullrun_5m"  # usage: ./venv/bin/python scripts/bullrun_replay.py [--ong]; klines auto-fetched (Aug-13 → now)
import os, requests, time
os.makedirs(S, exist_ok=True)
def _fetch(p):
    t0=int(pd.Timestamp("2026-08-13",tz="UTC").timestamp()*1000); t1=int(pd.Timestamp.utcnow().timestamp()*1000); rows=[]; s_=t0
    while s_<t1:
        r=requests.get("https://fapi.binance.com/fapi/v1/klines",params=dict(symbol=p,interval="5m",startTime=s_,endTime=t1,limit=1500)).json()
        if not isinstance(r,list) or not r: break
        rows+=r; s_=r[-1][0]+300000
        if len(r)<1500: break
        time.sleep(0.1)
    d=pd.DataFrame(rows)[[0,1,2,3,4,5,7]]; d.columns=['open_time','o','h','l','c','vol','qvol']; d.astype(float).to_csv(f"{S}/{p}_5m.csv",index=False)
PAIRS=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","1000PEPEUSDT","HYPEUSDT","SUIUSDT","ADAUSDT","LINKUSDT"]
INCLUDE_ONG='--ong' in sys.argv
if INCLUDE_ONG: PAIRS.append("ONGUSDT")
def _arg(k,default=None,cast=float):
    for a in sys.argv:
        if a.startswith(k+'='): return cast(a.split('=',1)[1])
    return default
for _p in (_arg('--extra','',str) or '').split(','):
    if _p and _p not in PAIRS: PAIRS.append(_p)
ATR_MAX=_arg('--atr-max'); CHG_MAX=_arg('--chg-max'); SL_FLOOR=_arg('--sl-floor',-1.2); EXCL=(_arg('--exclude','',str) or '').split(',')
PAIRS=[p for p in PAIRS if p not in EXCL]
QUIET='--quiet' in sys.argv
def load(p):
    if not os.path.exists(f"{S}/{p}_5m.csv"): _fetch(p)
    d=pd.read_csv(f"{S}/{p}_5m.csv"); d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True); return d.set_index('ts')
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def atr(d,n=14):
    tr=pd.concat([d.h-d.l,(d.h-d.c.shift()).abs(),(d.l-d.c.shift()).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean()
def adx(d,n=14):
    up=d.h.diff(); dn=-d.l.diff(); pdm=np.where((up>dn)&(up>0),up,0.0); ndm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([d.h-d.l,(d.h-d.c.shift()).abs(),(d.l-d.c.shift()).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/n,adjust=False).mean(); pdi=100*pd.Series(pdm,index=d.index).ewm(alpha=1/n,adjust=False).mean()/a; ndi=100*pd.Series(ndm,index=d.index).ewm(alpha=1/n,adjust=False).mean()/a
    dx=100*(pdi-ndi).abs()/(pdi+ndi); return dx.ewm(alpha=1/n,adjust=False).mean()
btc=load("BTCUSDT")
c=btc.c; btc['r72']=(c/c.shift(864)-1)*100; e20=ema(c,20); btc['above']=(c>e20).astype(float).rolling(864).mean()*100
btc['eff']=(c-c.shift(864)).abs()/c.diff().abs().rolling(864).sum()
btc['r6']=(c/c.shift(72)-1)*100; h1=btc.c.resample('1h').last().dropna(); e50h=ema(h1,50); btc['e50h']=e50h.reindex(btc.index,method='ffill').shift(1)
btc['off24']=(c/btc.h.rolling(288).max()-1)*100; btc['adx']=adx(btc); btc['adx_prev']=btc['adx'].shift(1)
# GREEN state machine on closed bars
state=[];g=False
for i,(r72,ab,ef,r6,cl,e5) in enumerate(zip(btc.r72,btc.above,btc.eff,btc.r6,btc.c,btc.e50h)):
    if np.isnan(r72) or np.isnan(ab) or np.isnan(ef): state.append(False); continue
    if not g and r72>=5 and ab>=56 and ef>=0.10: g=True
    elif g and not (r72>=4 and ab>=53 and ef>=0.08): g=False
    if g and ((not np.isnan(r6) and r6<=-3) or (not np.isnan(e5) and cl<e5)): g=False
    state.append(g)
btc['green']=state
P={}
for p in PAIRS:
    d=load(p); d['e20']=ema(d.c,20); d['e5']=ema(d.c,5); d['atr']=atr(d); d['atrp']=d.atr/d.c*100
    h1=d.c.resample('1h').last().dropna(); d['e50h']=ema(h1,50).reindex(d.index,method='ffill').shift(1); d['chg24']=(d.c/d.c.shift(288)-1)*100; P[p]=d
idx=btc.index[btc.index>=pd.Timestamp("2026-08-14",tz="UTC")]
LADDER=[(4,3.5),(5,4.5),(6,5.5),(8,7),(10,9),(12,11),(15,13.5),(20,18),(25,22.5),(30,27)]
FEE=0.09
def simulate(entry_i,p,ep,atrp):
    d=P[p]; sl=max(min(-0.7,-1.5*atrp),SL_FLOOR); peak=0.0; armed=False
    for j in range(entry_i+1,min(entry_i+240,len(d))):
        hi=(d.h.iloc[j]/ep-1)*100; lo=(d.l.iloc[j]/ep-1)*100
        stop=sl; reason='SL'
        if peak>=1.0:
            fl=max(0.2,peak-2*atrp); rung=max([f for t,f in LADDER if peak>=t],default=-99)
            stop=max(fl,rung); reason='LADDER' if rung>=fl else ('BE' if fl==0.2 else 'TRAIL')
        if lo<=stop: return stop-FEE,peak,reason,d.index[j]
        peak=max(peak,hi)
    j=min(entry_i+240,len(d)-1); return (d.c.iloc[j]/ep-1)*100-FEE,peak,'MAXHOLD',d.index[j]
trades=[]; open_until={}; last_fire={}; dip={}
for ts in idx:
    b=btc.loc[ts]
    if not b.green: dip={}; continue
    open_pos={p:u for p,u in open_until.items() if u>ts}
    for p in PAIRS:
        d=P[p]
        if ts not in d.index: continue
        i=d.index.get_loc(ts); r=d.iloc[i]
        if np.isnan(r.e50h) or np.isnan(r.atr): continue
        if r.c<=r.e50h: dip.pop(p,None); continue
        if r.c<=r.e20-0.3*r.atr: dip[p]=ts; continue
        if p in dip and (ts-dip[p])<=pd.Timedelta(hours=6) and r.c>r.e20:
            if b.off24< -2.0: continue            # gate; dip stays alive
            dip.pop(p,None)
            if ATR_MAX is not None and r.atrp>ATR_MAX: continue
            if CHG_MAX is not None and abs(r.chg24)>CHG_MAX: continue
            if p in open_pos or (p in last_fire and ts-last_fire[p]<pd.Timedelta(hours=2)) or len(open_pos)>=4: continue
            ep=d.o.iloc[i+1]; pnl,peak,reason,cts=simulate(i+1,p,ep,r.atrp)
            trades.append(dict(ts=ts,pair=p,pnl=pnl,peak=peak,reason=reason,atr=r.atrp,adx_prev=b.adx_prev,reclaim=(r.c-r.e5)/r.e5*100,off24=b.off24,r72=b.r72,chg24=r.chg24))
            open_until[p]=cts; open_pos[p]=cts; last_fire[p]=ts
t=pd.DataFrame(trades); t['usd']=t.pnl/100*12700; t['win']=t.pnl>0
t.to_csv(f"{S}/replay_trades{'_ong' if INCLUDE_ONG else ''}.csv",index=False)
if QUIET:
    LIVE=pd.Timestamp('2026-08-21 17:06',tz='UTC'); tag=' '.join(a for a in sys.argv[1:] if a!='--quiet')
    for wl,w in [('FOUND',t[t.ts<LIVE]),('LIVE',t[t.ts>=LIVE]),('ALL',t)]:
        print(f"{tag:52s} {wl:5s} N={len(w):3d} WR={100*w.win.mean() if len(w) else 0:4.0f}% net=${w.usd.sum():6.0f} avg={w.pnl.mean() if len(w) else 0:+.2f}%")
    sys.exit(0)
g=btc[btc.green]; print("GREEN bars:",len(g),"| episodes:", (btc.green & ~btc.green.shift(1,fill_value=False)).sum(), "| first/last:",g.index.min(),g.index.max())
def rep(x,lab):
    print(f"{lab:44s} N={len(x):3d} WR={100*x.win.mean() if len(x) else 0:4.0f}% net=${x.usd.sum():6.0f} avg={x.pnl.mean() if len(x) else 0:+.2f}%")
LIVE=pd.Timestamp("2026-08-21 17:06",tz="UTC")
for wl,w in [("FOUNDING (pre-live, <Aug-21 17:06)",t[t.ts<LIVE]),("LIVE-OVERLAP (≥Aug-21 17:06)",t[t.ts>=LIVE]),("ALL",t)]:
    print("\n===",wl); rep(w,"all")
    rep(w[~(w.adx_prev<=21.5)],"① drop BTC ADX(prev) ≤21.5"); rep(w[w.adx_prev<=21.5],"   blocked by ①")
    rep(w[~(w.reclaim<=0.15)],"② drop reclaim ≤0.15%"); rep(w[w.reclaim<=0.15],"   blocked by ②")
    m=(w.adx_prev<=21.5)&(w.reclaim<=0.15); rep(w[~m],"③ drop ① AND ②"); rep(w[m],"   blocked by ③")
    if len(w)>=12:
        print("   ADX(prev) terciles:"); 
        for lo_,hi_,lab in [(0,18,'<18'),(18,21.5,'18-21.5'),(21.5,25,'21.5-25'),(25,99,'>25')]:
            rep(w[(w.adx_prev>lo_)&(w.adx_prev<=hi_)],f"     ADX {lab}")
        print("   reclaim buckets:")
        for lo_,hi_,lab in [(-9,0,'<0'),(0,0.15,'0-0.15'),(0.15,0.4,'0.15-0.4'),(0.4,99,'>0.4')]:
            rep(w[(w.reclaim>lo_)&(w.reclaim<=hi_)],f"     reclaim {lab}")
