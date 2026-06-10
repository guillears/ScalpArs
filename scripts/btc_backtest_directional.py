#!/usr/bin/env python3
"""Phase-1b: BTC DIRECTIONAL backtest on 4h bars (from 1h cache), 3 years.
Systems: D1 EMA20/50 cross · D2 Donchian-20 breakout · D3 EMA50-slope regime.
Both sides, % trailing exits, fee 0.063% RT (negligible at this scale).
Per-year breakdown = the cycle-robustness test. Benchmark: buy & hold."""
import csv, os
from datetime import datetime, timezone
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows=list(csv.reader(open(os.path.join(HERE,'reports','btc_klines_1h_cache.csv'))))[1:]
# resample 1h -> 4h
T=[];O=[];H=[];L=[];C=[]
for i in range(0,len(rows)-len(rows)%4,4):
    blk=rows[i:i+4]
    T.append(int(blk[0][0])); O.append(float(blk[0][1]))
    H.append(max(float(b[2]) for b in blk)); L.append(min(float(b[3]) for b in blk))
    C.append(float(blk[-1][4]))
N=len(C)
def ema(vals,p):
    k=2/(p+1); out=[None]*len(vals); s=None
    for i,v in enumerate(vals):
        s=v if s is None else v*k+s*(1-k); out[i]=s
    return out
E20=ema(C,20); E50=ema(C,50)
def hh(i,n): return max(H[max(0,i-n):i]) if i>=n else None
def ll(i,n): return min(L[max(0,i-n):i]) if i>=n else None
FEE=0.063
def run(system, side, trail, sl, i0=60, i1=N-1):
    trades=[]; i=i0
    while i<i1-1:
        sig=False
        if system=='D1':
            if side=='L': sig=E20[i-1]<=E50[i-1] and E20[i]>E50[i]
            else:         sig=E20[i-1]>=E50[i-1] and E20[i]<E50[i]
        elif system=='D2':
            h20,l20=hh(i,20),ll(i,20)
            if h20 is None: i+=1; continue
            if side=='L': sig=C[i]>h20
            else:         sig=C[i]<l20
        elif system=='D3':
            if E50[i-6] is None: i+=1; continue
            slp=(E50[i]-E50[i-6])/E50[i-6]*100; slp_p=(E50[i-1]-E50[i-7])/E50[i-7]*100 if E50[i-7] else None
            if slp_p is None: i+=1; continue
            if side=='L': sig=slp_p<=0.30 and slp>0.30
            else:         sig=slp_p>=-0.30 and slp<-0.30
        if not sig: i+=1; continue
        e=i+1; entry=O[e]; peak=0.0; out=None
        for k in range(e,i1):
            hi=(H[k]-entry)/entry*100 if side=='L' else (entry-L[k])/entry*100
            lo=(L[k]-entry)/entry*100 if side=='L' else (entry-H[k])/entry*100
            if lo<=-sl and peak<trail*0.5: out=-sl; i=k; break   # hard SL early
            peak=max(peak,hi)
            cl=(C[k]-entry)/entry*100*(1 if side=='L' else -1)
            if peak-cl>=trail: out=peak-trail; i=k; break
        if out is None: out=(C[i1-1]-entry)/entry*100*(1 if side=='L' else -1); i=i1
        trades.append((T[e],out-FEE,(i-e)*4)); i+=1
    return trades
def stats(tr,days):
    if not tr: return "N=0"
    n=len(tr); w=sum(1 for _,p,_ in tr if p>0)/n*100; tot=sum(p for _,p,_ in tr)
    eq=0;pk=0;mdd=0
    for _,p,_ in tr:
        eq+=p;pk=max(pk,eq);mdd=min(mdd,eq-pk)
    hold=sum(h for *_,h in tr)/n
    return f"N={n:3d} ({n/days*30:.1f}/mo) WR={w:3.0f}% exp{tot/n:+6.2f}%/tr tot{tot:+7.1f}% maxDD{mdd:6.1f}% hold{hold/24:4.1f}d"
DAYS=(T[-1]-T[0])/86400000
print(f"4h bars={N}  {DAYS:.0f} days  BUY&HOLD: {(C[-1]-C[60])/C[60]*100:+.0f}%")
GRID=[(3.0,3.0),(4.0,3.0),(5.0,4.0),(2.0,2.0)]
best=[]
for system in ('D1','D2','D3'):
    for side in ('L','S'):
        for trail,sl in GRID:
            tr=run(system,side,trail,sl)
            tag=f"{system}-{side} trail{trail:.0f}/SL{sl:.0f}"
            tot=sum(p for _,p,_ in tr) if tr else 0
            best.append((tot,tag,tr))
best.sort(reverse=True)
print("\n=== ALL CELLS (sorted by total; fee 0.063 incl) ===")
for tot,tag,tr in best:
    print(f"  {tag:22}: {stats(tr,DAYS)}")
# per-year breakdown for the top combined long+short choice
print("\n=== PER-YEAR for top LONG cell and top SHORT cell ===")
topL=next(x for x in best if '-L' in x[1]); topS=next(x for x in best if '-S' in x[1])
from collections import defaultdict
for tot,tag,tr in (topL,topS):
    bym=defaultdict(list)
    for t,p,_ in tr: bym[datetime.fromtimestamp(t/1000,tz=timezone.utc).year].append(p)
    print(f"  {tag}: "+"  ".join(f"{y}:{sum(v):+.1f}%(N={len(v)})" for y,v in sorted(bym.items())))
