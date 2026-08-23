import pandas as pd, numpy as np, sys
sys.argv=['x','--quiet']
src=open("scripts/bullrun_replay.py").read()
exec(src[:src.index("trades=[]")])      # loads btc (with green/off24/r6/adx/e50h), P (pairs), simulate(), LADDER, FEE, idx
c=btc.c; e20=ema(c,20); e50=ema(c,50); e13=ema(c,13)
hi24=btc.h.rolling(288).max(); lo24=btc.l.rolling(288).min()
dd=(lo24/hi24-1)*100                       # depth of the 24h low vs the 24h high (negative)
from_low=(c/lo24-1)*100
BOUNCE_MIN=float(sys.argv[sys.argv.index('--bmin')+1]) if '--bmin' in sys.argv else 1.5
DEPTH_MIN=2.0
# bounce state machine on closed bars
st=[];on=False;on_since=None
for i,(t,row) in enumerate(btc.iterrows()):
    cond=(dd.iloc[i]<=-DEPTH_MIN) and (from_low.iloc[i]>=BOUNCE_MIN) and (row.c>e20.iloc[i]) and (not np.isnan(row.e50h) and row.c>row.e50h)
    if not on and cond: on=True; on_since=t
    elif on:
        if row.c<e50.iloc[i] or row.c<row.e50h or (t-on_since)>pd.Timedelta(hours=24): on=False
    st.append(on)
btc['bounce']=st
b=btc.bounce; ch=btc.index[(b!=b.shift())]
wins=[]; cur=None
for t in ch:
    if btc.bounce.loc[t] and cur is None: cur=t
    elif not btc.bounce.loc[t] and cur is not None: wins.append((cur,t)); cur=None
if cur is not None: wins.append((cur,btc.index[-1]))
print(f"BOUNCE windows (depth≥{DEPTH_MIN}%, bounce≥{BOUNCE_MIN}% off 24h low, >EMA20 & >1hEMA50; off when <EMA50 or <1hEMA50 or 24h):")
for a,b_ in wins: print(f"  {str(a)[5:16]} → {str(b_)[5:16]}  ({(b_-a).total_seconds()/3600:.1f}h)  green-at-start={btc.green.loc[a]}  BTC {btc.c.loc[a]:.0f}→{btc.c.loc[b_]:.0f} ({(btc.c.loc[b_]/btc.c.loc[a]-1)*100:+.2f}%)")
# run the sleeve entry machinery inside bounce windows (composite-independent), with/without EMA13 gate
def run(gate_e13, only_when_green_off=True):
    trades=[]; open_until={}; last_fire={}; dip={}
    for ts in idx:
        b_=btc.loc[ts]
        if not b_.bounce or (only_when_green_off and b_.green): dip={}; continue
        open_pos={p:u for p,u in open_until.items() if u>ts}
        for p in PAIRS:
            d=P[p]
            if ts not in d.index: continue
            i=d.index.get_loc(ts); r=d.iloc[i]
            if np.isnan(r.e50h) or np.isnan(r.atr): continue
            if r.c<=r.e50h: dip.pop(p,None); continue
            if r.c<=r.e20-0.3*r.atr: dip[p]=ts; continue
            if p in dip and (ts-dip[p])<=pd.Timedelta(hours=6) and r.c>r.e20:
                if gate_e13 and b_.c<e13.loc[ts]: continue
                dip.pop(p,None)
                if p in open_pos or (p in last_fire and ts-last_fire[p]<pd.Timedelta(hours=2)) or len(open_pos)>=4: continue
                ep=d.o.iloc[i+1]; pnl,peak,reason,cts=simulate(i+1,p,ep,r.atrp)
                trades.append(dict(ts=ts,pair=p,pnl=pnl,peak=peak,reason=reason,atr=r.atrp,win_start=[a for a,bb in wins if a<=ts<=bb][0]))
                open_until[p]=cts; open_pos[p]=cts; last_fire[p]=ts
    t=pd.DataFrame(trades)
    if len(t): t['usd']=t.pnl/100*12700; t['win']=t.pnl>0
    return t
for lab,g in [("bounce door, no EMA13 gate",False),("bounce door + EMA13 gate",True)]:
    t=run(g)
    print(f"\n=== {lab} (only while composite GREEN is OFF): N={len(t)} WR={100*t.win.mean() if len(t) else 0:.0f}% net=${t.usd.sum() if len(t) else 0:.0f} avg={t.pnl.mean() if len(t) else 0:+.2f}%")
    if len(t):
        print(t.groupby('win_start').agg(n=('pnl','size'),w=('win','sum'),net=('usd','sum'),avg=('pnl','mean')).round(2).to_string())
        print("  by exit:",t.groupby('reason').agg(n=('pnl','size'),net=('usd','sum')).round(0).to_dict('index'))
        print("  today (Aug-23):"); tt=t[t.ts>=pd.Timestamp('2026-08-23',tz='UTC')]; print(tt[['ts','pair','pnl','peak','reason']].round(2).to_string(index=False) if len(tt) else "  none")
