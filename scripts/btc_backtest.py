#!/usr/bin/env python3
"""Phase-1 BTC offline backtester (pure stdlib).

Data: reports/btc_klines_5m_cache.csv (from btc_backtest_fetch.py).
Systems (theory-driven, deliberately simple):
  S1 PULLBACK : 1h-trend aligned + 5m EMA5xEMA13 re-ignition cross
  S2 MOMENTUM : bot-style 5m stack (EMA5>8>13 + RSI band + gap) gated on 1h trend
  S3 RIDE     : 1h EMA20-slope regime flip entry, trail-only exit (long holds)
Execution model: signal on closed 5m bar -> enter next bar OPEN. Intrabar
TP/SL via high/low; if both touch in one bar -> SL first (conservative).
Fees (round trip, charged per trade): MM=0.036% (maker in+out), MT=0.063%
(maker in / taker out). P&L = sum of net % moves (leverage-invariant).
IS = first 4 months, OOS = last 2 months. All grid cells reported."""
import csv, os, math
from datetime import datetime, timezone

HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows=list(csv.reader(open(os.path.join(HERE,'reports','btc_klines_5m_cache.csv'))))[1:]
T=[int(r[0]) for r in rows]; O=[float(r[1]) for r in rows]; H=[float(r[2]) for r in rows]
L=[float(r[3]) for r in rows]; C=[float(r[4]) for r in rows]; V=[float(r[5]) for r in rows]
N=len(C)

def ema(vals, period):
    k=2/(period+1); out=[None]*len(vals); s=None
    for i,v in enumerate(vals):
        s = v if s is None else v*k + s*(1-k)
        out[i]=s
    return out
def rsi(vals, period=14):
    out=[None]*len(vals); ag=al=None
    for i in range(1,len(vals)):
        ch=vals[i]-vals[i-1]; gain=max(ch,0); loss=max(-ch,0)
        if i<=period:
            ag=(0 if ag is None else ag)+gain/period; al=(0 if al is None else al)+loss/period
            if i==period: out[i]=100-100/(1+(ag/al if al>0 else 1e9))
        else:
            ag=(ag*(period-1)+gain)/period; al=(al*(period-1)+loss)/period
            out[i]=100-100/(1+(ag/al if al>0 else 1e9))
    return out
def adx_atr(period=14):
    adx=[None]*N; atrp=[None]*N
    trs=[]; pdms=[]; ndms=[]; satr=spdm=sndm=None; sadx=None
    for i in range(1,N):
        tr=max(H[i]-L[i],abs(H[i]-C[i-1]),abs(L[i]-C[i-1]))
        up=H[i]-H[i-1]; dn=L[i-1]-L[i]
        pdm=up if (up>dn and up>0) else 0.0; ndm=dn if (dn>up and dn>0) else 0.0
        if i<=period:
            trs.append(tr); pdms.append(pdm); ndms.append(ndm)
            if i==period:
                satr=sum(trs); spdm=sum(pdms); sndm=sum(ndms)
        else:
            satr=satr-satr/period+tr; spdm=spdm-spdm/period+pdm; sndm=sndm-sndm/period+ndm
            pdi=100*spdm/satr if satr>0 else 0; ndi=100*sndm/satr if satr>0 else 0
            dx=100*abs(pdi-ndi)/(pdi+ndi) if (pdi+ndi)>0 else 0
            sadx = dx if sadx is None else (sadx*(period-1)+dx)/period
            adx[i]=sadx; atrp[i]=(satr/period)/C[i]*100
    return adx,atrp

E5,E8,E13,E20,E50 = (ema(C,p) for p in (5,8,13,20,50))
R=rsi(C); ADX,ATRP=adx_atr()

# 1h series from COMPLETED hours only (no lookahead): map each 5m bar -> last closed 1h bar
h_close=[]; h_t=[]
for i in range(N):
    dt=datetime.fromtimestamp(T[i]/1000, tz=timezone.utc)
    if dt.minute==55:  # last 5m bar of the hour closes the 1h candle
        h_close.append(C[i]); h_t.append(T[i])
HE20=ema(h_close,20); HR=rsi(h_close)
# 1h slope %/bar = EMA20 vs 3 bars back (mirrors bot convention)
HSLOPE=[None]*len(h_close)
for j in range(3,len(h_close)):
    if HE20[j-3]: HSLOPE[j]=(HE20[j]-HE20[j-3])/HE20[j-3]*100
# forward map
h_idx=[-1]*N; j=0
for i in range(N):
    while j<len(h_t)-1 and h_t[j+1]<=T[i]: j+=1
    h_idx[i]= j if h_t[j]<=T[i] else -1
def h1(i, arr):
    j=h_idx[i]
    return arr[j] if j>=0 else None

WARM=600
def run(system, side, tp, sl, trail_arm, trail_pb, fee, max_hold_bars=288, i0=WARM, i1=N-1):
    """Returns list of net-pct trade results + meta. tp/sl in %, trail optional."""
    trades=[]; i=i0
    while i<i1-1:
        sig=False
        sl5,sl5p = E5[i],E5[i-1]; s13,s13p=E13[i],E13[i-1]
        hs=h1(i,HSLOPE); hr=h1(i,HR)
        if None in (sl5,sl5p,s13,s13p,R[i],ADX[i]) or hs is None or hr is None:
            i+=1; continue
        if system=='S1':  # 1h trend + 5m EMA5xEMA13 re-ignition
            if side=='L': sig = hs>0.05 and 45<=hr<=70 and sl5p<=s13p and sl5>s13
            else:         sig = hs<-0.05 and 30<=hr<=55 and sl5p>=s13p and sl5<s13
        elif system=='S2':  # bot-style 5m momentum stack gated on 1h
            gap=(E5[i]-E13[i])/E13[i]*100
            if side=='L': sig = hs>0.05 and E5[i]>E8[i]>E13[i] and 52<=R[i]<=68 and gap>=0.03 and ADX[i]>=18
            else:         sig = hs<-0.05 and E5[i]<E8[i]<E13[i] and 32<=R[i]<=48 and gap<=-0.03 and ADX[i]>=18
        elif system=='S3':  # 1h regime flip (slope crosses threshold) ride
            hsp = HSLOPE[h_idx[i]-1] if h_idx[i]>=1 else None
            if hsp is None: i+=1; continue
            if side=='L': sig = hsp<=0.08 and hs>0.08
            else:         sig = hsp>=-0.08 and hs<-0.08
        if not sig: i+=1; continue
        # enter next bar open
        e=i+1; entry=O[e]; peak=0.0; trough=0.0; out=None; armed=False; hwm=entry
        for k in range(e, min(e+max_hold_bars, i1)):
            if side=='L':
                hi=(H[k]-entry)/entry*100; lo=(L[k]-entry)/entry*100
            else:
                hi=(entry-L[k])/entry*100; lo=(entry-H[k])/entry*100
            # conservative: SL checked first
            if sl and lo<=-sl: out=-sl; i=k; break
            if trail_arm:
                if hi>=trail_arm: armed=True
                if armed:
                    peak=max(peak,hi)
                    cl=(C[k]-entry)/entry*100*(1 if side=='L' else -1)
                    if peak-cl>=trail_pb: out=max(cl,0.0) if cl>peak-trail_pb else cl; out=peak-trail_pb if out<peak-trail_pb else out; out=peak-trail_pb; i=k; break
            if tp and hi>=tp: out=tp; i=k; break
            peak=max(peak,hi); trough=min(trough,lo)
        if out is None:
            k=min(e+max_hold_bars,i1)-1
            out=(C[k]-entry)/entry*100*(1 if side=='L' else -1); i=k
        trades.append(out-fee)
        i+=1
    return trades

def stats(tr):
    if not tr: return "N=0"
    n=len(tr); wr=sum(1 for x in tr if x>0)/n*100; tot=sum(tr); avg=tot/n
    eq=0; pk=0; mdd=0
    for x in tr:
        eq+=x; pk=max(pk,eq); mdd=min(mdd,eq-pk)
    return f"N={n:4d} WR={wr:3.0f}% avg{avg:+6.3f}% tot{tot:+7.1f}% maxDD{mdd:6.1f}%"

# IS/OOS split index (first 4 of 6 months)
split=int(N*4/6)
FEES={'MM':0.036,'MT':0.063}
GRID=[  # (tp, sl, trail_arm, trail_pb, label)
    (0.40,0.35,None,None,'TP.40/SL.35'),
    (0.60,0.40,None,None,'TP.60/SL.40'),
    (0.80,0.50,None,None,'TP.80/SL.50'),
    (None,0.40,0.30,0.25,'trail.30/.25 SL.40'),
    (None,0.50,0.40,0.30,'trail.40/.30 SL.50'),
]
print(f"bars={N}  IS=[{WARM}:{split}]  OOS=[{split}:{N}]")
for system in ('S1','S2','S3'):
    g = GRID if system!='S3' else [(None,0.70,0.50,0.40,'trail.50/.40 SL.70'),(1.00,0.70,None,None,'TP1.0/SL.70')]
    for side in ('L','S'):
        print(f"\n=== {system} {side} ===")
        for tp,sl,ta,tpb,lbl in g:
            for fl,fee in FEES.items():
                isr=run(system,side,tp,sl,ta,tpb,fee,i1=split)
                oos=run(system,side,tp,sl,ta,tpb,fee,i0=split)
                print(f"  {lbl:20} fee={fl}: IS {stats(isr)} | OOS {stats(oos)}")
