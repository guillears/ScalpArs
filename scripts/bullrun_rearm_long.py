import pandas as pd, numpy as np, requests, time, os, sys
D="reports/backtest_cache/bullrun_5m_long"; os.makedirs(D,exist_ok=True)
PAIRS=["BTCUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","1000PEPEUSDT","HYPEUSDT","SUIUSDT","ADAUSDT","LINKUSDT","ETHUSDT"]
T0=int(pd.Timestamp("2025-11-25",tz="UTC").timestamp()*1000); T1=int(pd.Timestamp.now("UTC").timestamp()*1000)
def fetch(p):
    f=f"{D}/{p}_5m.csv"
    if os.path.exists(f): return pd.read_csv(f)
    rows=[]; s=T0
    while s<T1:
        try: r=requests.get("https://fapi.binance.com/fapi/v1/klines",params=dict(symbol=p,interval="5m",startTime=s,endTime=T1,limit=1500),timeout=30).json()
        except Exception: time.sleep(1); continue
        if not isinstance(r,list) or not r: break
        rows+=r; s=r[-1][0]+300000
        if len(r)<1500: break
        time.sleep(0.08)
    d=pd.DataFrame(rows)[[0,1,2,3,4,5]]; d.columns=['open_time','o','h','l','c','vol']; d=d.astype(float); d.to_csv(f,index=False); return d
def load(p):
    d=fetch(p); d['ts']=pd.to_datetime(d.open_time,unit='ms',utc=True); return d.set_index('ts')[['o','h','l','c']]
ema=lambda s,n: s.ewm(span=n,adjust=False).mean()
def atr(d,n=14):
    tr=pd.concat([d.h-d.l,(d.h-d.c.shift()).abs(),(d.l-d.c.shift()).abs()],axis=1).max(axis=1); return tr.ewm(alpha=1/n,adjust=False).mean()
def adx(d,n=14):
    up=d.h.diff(); dn=-d.l.diff(); pdm=np.where((up>dn)&(up>0),up,0.0); ndm=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([d.h-d.l,(d.h-d.c.shift()).abs(),(d.l-d.c.shift()).abs()],axis=1).max(axis=1); a=tr.ewm(alpha=1/n,adjust=False).mean()
    pdi=100*pd.Series(pdm,index=d.index).ewm(alpha=1/n,adjust=False).mean()/a; ndi=100*pd.Series(ndm,index=d.index).ewm(alpha=1/n,adjust=False).mean()/a
    return (100*(pdi-ndi).abs()/(pdi+ndi)).ewm(alpha=1/n,adjust=False).mean()
btc=load("BTCUSDT"); c=btc.c; print("BTC bars",len(btc),btc.index.min(),btc.index.max(),flush=True)
e13,e20,e50=ema(c,13),ema(c,20),ema(c,50); h1=c.resample('1h').last().dropna(); e50h=ema(h1,50).reindex(btc.index,method='ffill').shift(1)
A=adx(btc); rise=A-A.shift(6); e20h=ema(h1,20); s20h=((e20h/e20h.shift(1)-1)*100).reindex(btc.index,method='ffill'); e200h=ema(h1,200).reindex(btc.index,method='ffill'); r7d=(c/c.shift(2016)-1)*100; fan=(e13>e20)&(e20>e50); above=c>e50h
hi24=btc.h.rolling(288).max(); lo24=btc.l.rolling(288).min(); depth=(lo24/hi24-1)*100; from_low=(c/lo24-1)*100
r72=(c/c.shift(864)-1)*100; ab=(c>e20).astype(float).rolling(864).mean()*100; eff=(c-c.shift(864)).abs()/c.diff().abs().rolling(864).sum(); r6=(c/c.shift(72)-1)*100
g=False; st=[]
for a_,ab_,ef_,r6_,cl,e5 in zip(r72.values,ab.values,eff.values,r6.values,c.values,e50h.values):
    if np.isnan(a_) or np.isnan(ab_) or np.isnan(ef_): st.append(False); continue
    if not g and a_>=5 and ab_>=56 and ef_>=0.10: g=True
    elif g and not (a_>=4 and ab_>=53 and ef_>=0.10): g=False
    if g and ((not np.isnan(r6_) and r6_<=-3) or (not np.isnan(e5) and cl<e5)): g=False
    st.append(g)
green=pd.Series(st,index=btc.index); print("GREEN bars",int(green.sum()),"episodes",int((green&~green.shift(1,fill_value=False)).sum()),flush=True)
P={}
for p in PAIRS[1:]:
    d=load(p).reindex(btc.index); d['e20']=ema(d.c,20); d['atr']=atr(d); d['atrp']=d.atr/d.c*100
    hh=d.c.resample('1h').last().dropna(); d['e50h']=ema(hh,50).reindex(d.index,method='ffill').shift(1); d['r6']=(d.c/d.c.shift(72)-1)*100; d['ab']=(d.c>d.e50h); P[p]=d
alt_r6=pd.concat({p:P[p].r6 for p in P},axis=1); alt_ab=pd.concat({p:P[p].ab for p in P},axis=1)
med_r6=alt_r6.median(axis=1); pct_ab=alt_ab.mean(axis=1)*100; nalts=alt_r6.notna().sum(axis=1)
LAD=[(4,3.5),(5,4.5),(6,5.5),(8,7),(10,9),(12,11),(15,13.5),(20,18),(25,22.5),(30,27)]
def simulate(d,i,ep,atrp):
    sl=max(min(-0.7,-1.5*atrp),-1.2); peak=0.0; H=d.h.values; L=d.l.values; C=d.c.values
    for j in range(i+1,min(i+240,len(d))):
        if np.isnan(H[j]): continue
        hi=(H[j]/ep-1)*100; lo=(L[j]/ep-1)*100; stop=sl
        if peak>=1.0:
            fl=max(0.2,peak-2*atrp); rung=max([f for t,f in LAD if peak>=t],default=-99); stop=max(fl,rung)
        if lo<=stop: return stop-0.09,peak,j
        peak=max(peak,hi)
    j=min(i+240,len(d)-1); return (C[j]/ep-1)*100-0.09,peak,j
gend=np.where((~green.values)&(green.shift(1,fill_value=False).values))[0]   # bar indices where GREEN ended
hours_since_green=np.full(len(btc),np.inf); last=None
for i in range(len(btc)):
    if i in set(gend): last=i
    if last is not None: hours_since_green[i]=(i-last)/12.0
def door(btc_leg, post_h=None):
    on=False; st=[]; t0=None
    for i in range(len(btc)):
        if green.iloc[i] or nalts.iloc[i]<5: on=False; st.append(False); continue
        alts_ok=(med_r6.iloc[i]>1.0 and pct_ab.iloc[i]>=80)
        cond=btc_leg[i] and alts_ok and fan.iloc[i] and above.iloc[i] and (post_h is None or hours_since_green[i]<=post_h)
        if not on and cond: on=True; t0=i
        elif on and ((not above.iloc[i]) or A.iloc[i]<30 or (i-t0)>288): on=False
        st.append(on)
    return np.array(st)
legs={'ADX≥35 rising':((A>=35)&(rise>0)).values,'range ≥1.5% off ≥2% low':((depth<=-2.0)&(from_low>=1.5)).values}
IDX=btc.index; E13=e13.values; BC=c.values; S20H=s20h.values; UP=(c>e200h).values; R7=r7d.values
for nm,leg,ph in [('ADX door, any time',legs['ADX≥35 rising'],None),('ADX door ≤24h after GREEN end',legs['ADX≥35 rising'],24),('ADX door ≤48h after GREEN end',legs['ADX≥35 rising'],48),('ADX door ≤96h after GREEN end',legs['ADX≥35 rising'],96)]:
    s=door(leg,ph); trades=[]; open_until={}; last_fire={}; dip={}; wins=[]; cur=None
    for i in range(1,len(s)):
        if s[i] and not s[i-1]: cur=i
        if (not s[i]) and s[i-1] and cur is not None: wins.append((cur,i)); cur=None
    if cur is not None: wins.append((cur,len(s)-1))
    on_idx=np.where(s)[0]
    for i in on_idx:
        ts=IDX[i]; open_pos={p:u for p,u in open_until.items() if u>i}
        for p,d in P.items():
            cc=d.c.values[i]; e20v=d.e20.values[i]; at=d.atr.values[i]; e5=d.e50h.values[i]
            if np.isnan(cc) or np.isnan(e20v) or np.isnan(at) or np.isnan(e5): continue
            if cc<=e5: dip.pop(p,None); continue
            if cc<=e20v-0.3*at: dip[p]=i; continue
            if p in dip and (i-dip[p])<=72 and cc>e20v:
                if BC[i]<E13[i]: continue
                if S20H[i]<=0: continue   # live 1h-slope gate
                dip.pop(p,None)
                if p in open_pos or (p in last_fire and i-last_fire[p]<24) or len(open_pos)>=4: continue
                ep=d.o.values[i+1] if i+1<len(d) and not np.isnan(d.o.values[i+1]) else cc
                pnl,peak,j=simulate(d,i+1,ep,d.atrp.values[i]); w=[a for a,b_ in wins if a<=i<=b_][0]
                trades.append(dict(i=i,ts=ts,pair=p,pnl=pnl,peak=peak,w=w,up=bool(UP[i]),r7=R7[i])); open_until[p]=j; open_pos[p]=j; last_fire[p]=i
    t=pd.DataFrame(trades)
    if len(t): t['usd']=t.pnl/100*12700
    per=t.groupby('w').usd.sum() if len(t) else pd.Series(dtype=float)
    nw=len([1 for a,b_ in wins if (b_-a)>=6]); wf=len(per); lose=int((per<0).sum()); 
    print(f"\n=== {nm}: windows(≥30m) {nw} | with fills {wf} | losing {lose} ({100*lose/max(wf,1):.0f}%) | fills {len(t)} WR {100*(t.pnl>0).mean() if len(t) else 0:.0f}% net ${t.usd.sum() if len(t) else 0:.0f} avg {t.pnl.mean() if len(t) else 0:+.2f}%/t | best window ${per.max() if wf else 0:.0f} ({100*per.max()/per.sum() if wf and per.sum()>0 else 0:.0f}% of net) | median window ${per.median() if wf else 0:.0f}",flush=True)
    if len(t):
        print("  windows:",[(str(IDX[a])[:16],round(v)) for a,v in per.items()])
        t['month']=t.ts.dt.strftime('%Y-%m'); print("  by month:",t.groupby('month').agg(n=('pnl','size'),net=('usd','sum')).round(0).to_dict('index'))
        for lab,m in [("BTC above 1h EMA200 (uptrend ctx)",t.up),("BTC below 1h EMA200",~t.up),("BTC r7d > +3%",t.r7>3),("BTC r7d ≤ +3%",t.r7<=3)]:
            x=t[m]; pw=x.groupby('w').usd.sum(); print(f"    {lab:34s} fills {len(x):3d} WR {100*(x.pnl>0).mean() if len(x) else 0:3.0f}% net ${x.usd.sum() if len(x) else 0:6.0f} avg {x.pnl.mean() if len(x) else 0:+.2f} | windows {len(pw)} losing {int((pw<0).sum())}")
        t.to_csv(f"{D}/rearm_long_{nm.split()[0]}.csv",index=False)
