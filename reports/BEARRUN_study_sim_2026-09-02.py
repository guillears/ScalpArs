import urllib.request, json, datetime, time, bisect
S="/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/95900cf1-d63a-45fc-b16c-f92d4a8356b3/scratchpad"
def get(u):
    for a in range(4):
        try: return json.load(urllib.request.urlopen(u, timeout=25))
        except Exception: time.sleep(1+a)
    raise SystemExit("fetch fail "+u)
def kl(sym, interval, limit, start=None, end=None):
    u=f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&limit={limit}"
    if start: u+=f"&startTime={start}"
    if end: u+=f"&endTime={end}"
    return get(u)
UTC=datetime.UTC
fmt=lambda t: datetime.datetime.fromtimestamp(t/1000,UTC).strftime('%m-%d %H:%M')

wins=json.load(open(f"{S}/red_windows.json"))
# merge stay-band gaps <=30min (engine reopen rule)
merged=[]
for w in wins:
    if merged and w['start']-merged[-1]['end']<=30*60000 and merged[-1]['ended_by']=='stay-band':
        merged[-1]['end']=w['end']; merged[-1]['ended_by']=w['ended_by']
    else: merged.append(dict(w))
# drop episodes shorter than 30 min (too short for a 5m dip-reclaim to complete)
episodes=[w for w in merged if w['end']-w['start']>=30*60000]
print("merged episodes:", len(merged), "| tradeable (>=30min):", len(episodes))
for w in episodes: print("  EP", fmt(w['start']),"->",fmt(w['end']), f"{(w['end']-w['start'])/3600000:.1f}h", w['ended_by'])

# universe: top-45 current USDT perps by 24h quote volume (proxy list), blacklist mirrored
BL={"ONGUSDT","ETHUSDT","BTCUSDT"}
tick=get("https://fapi.binance.com/fapi/v1/ticker/24hr")
cand=[t['symbol'] for t in sorted(tick,key=lambda t:-float(t['quoteVolume'])) if t['symbol'].endswith('USDT') and t['symbol'] not in BL and t['symbol'].isascii() and t['symbol'].isalnum()][:45]
# daily quote volumes for per-window ranking
dvol={}
for p in cand:
    d=kl(p,'1d',210)
    dvol[p]={r[0]:float(r[7]) for r in d}
def top10(day_ts):
    # rank by PRIOR day's quote volume
    day0=day_ts-(day_ts%86400000)-86400000
    rows=[(p,v[day0]) for p,v in dvol.items() if day0 in v]
    rows.sort(key=lambda x:-x[1])
    return [p for p,_ in rows[:10]]

btc=json.load(open(f"{S}/btc5m.json"))
bts, bcl = btc['ts'], btc['closes']
def btc_off_low(t):
    i=bisect.bisect_right(bts,t)-1
    if i<288: return None
    lo=min(bcl[i-288:i])
    return (bcl[i]/lo-1)*100

LADDER=[(4.0,3.5),(5.0,4.5),(6.0,5.5),(8.0,7.0),(10.0,9.0),(12.0,11.0),(15.0,13.5),(20.0,18.0),(25.0,22.5),(30.0,27.0)]
def exit_check(pnl, peak, atr):
    sl=min(-0.7, max(-(atr*1.5), -1.2)) if atr>0 else -0.7
    if peak>=1.0:
        trail=(peak-2.0*atr) if atr>0 else 0.2
        stop=max(0.2, trail); reason="TRAILING_STOP" if trail>0.2 else "BREAKEVEN_EXIT"
        lf=None
        for thr,fl in LADDER:
            if peak>=thr: lf=fl
        if lf is not None and lf>stop: stop=lf; reason="LADDER_FLOOR"
        return (pnl<=stop), reason, stop
    return (pnl<=sl), "STOP_LOSS", sl

FEE=0.08  # round-trip taker approx, net-of-fees convention
all_trades=[]
for ep in episodes:
    w0,w1=ep['start'],ep['end']
    pairs=top10(w0)
    slots=[]  # (pair, close_ts)
    ep_trades=[]
    for p in pairs:
        k5=kl(p,'5m',1000,start=w0-14*3600000, end=w1+6*3600000)
        if len(k5)<80: continue
        k1=kl(p,'1h',500,end=w1+3600000)
        h_ts=[r[0] for r in k1]; h_cl=[float(r[4]) for r in k1]
        e=h_cl[0]; kk=2/51; h_e50=[]
        for c in h_cl: e=c*kk+e*(1-kk); h_e50.append(e)
        def pe50(t):
            i=bisect.bisect_right(h_ts,t)-2  # last CLOSED 1h bar
            return h_e50[i] if i>=0 else None
        cl=[float(r[4]) for r in k5]; hi=[float(r[2]) for r in k5]; lo=[float(r[3]) for r in k5]; t5=[r[0] for r in k5]
        e=cl[0]; k2=2/21; em=[]
        for c in cl: e=c*k2+e*(1-k2); em.append(e)
        trs=[max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1])) for i in range(1,len(k5))]
        atrs=[None]
        a=sum(trs[:14])/14
        for j,tr in enumerate(trs):
            if j<14: atrs.append(None); continue
            a=(a*13+tr)/14; atrs.append(a)
        flag=False; dip_ts=0; last_entry=-1e18; open_pos=None
        for i in range(20,len(k5)):
            if t5[i]>=w1 and open_pos is None: break
            atr=atrs[i] if i<len(atrs) else None
            if atr is None: continue
            atr_pct=atr/cl[i]*100
            # manage open position on this closed bar
            if open_pos is not None:
                ent=open_pos['entry']
                worst=(ent/hi[i]-1)*100-FEE
                best=(ent/lo[i]-1)*100-FEE
                closed=False
                c1,r1,s1=exit_check(worst, open_pos['peak'], open_pos['atr'])
                if c1:
                    px=max(worst, s1)
                    open_pos.update(pnl=px, reason=r1, close_ts=t5[i]); ep_trades.append(open_pos); open_pos=None; closed=True
                else:
                    open_pos['peak']=max(open_pos['peak'], best)
                    c2,r2,s2=exit_check((ent/cl[i]-1)*100-FEE, open_pos['peak'], open_pos['atr'])
                    if c2:
                        open_pos.update(pnl=max((ent/cl[i]-1)*100-FEE,s2), reason=r2, close_ts=t5[i]); ep_trades.append(open_pos); open_pos=None; closed=True
                    elif t5[i]-open_pos['ts']>=180*60000:
                        open_pos.update(pnl=(ent/cl[i]-1)*100-FEE, reason="MAX_HOLD", close_ts=t5[i]); ep_trades.append(open_pos); open_pos=None; closed=True
                if closed: continue
            if not (w0<=t5[i]<w1): continue
            # rally flag (mirror dip)
            if hi[i]>=em[i]+0.3*atr: flag=True; dip_ts=t5[i]
            if flag and t5[i]-dip_ts>6*3600000: flag=False
            if open_pos is not None or not flag: continue
            if cl[i]>=em[i]: continue          # rejection close below EMA20 required
            ref=pe50(t5[i])
            if ref is None or cl[i]>=ref: continue   # pair must be BELOW its 1h EMA50
            off=btc_off_low(t5[i])
            if off is not None and off>=2.0: continue  # bounce-phase gate (dip kept alive)
            if t5[i]-last_entry<2*3600000: continue    # 2h spacing
            live=[x for x in slots if x[1]>t5[i]]
            if len(live)>=4: continue                  # 4 slots
            open_pos={'pair':p,'ts':t5[i],'entry':cl[i],'peak':(cl[i]/cl[i]-1),'atr':atr_pct,'win':fmt(w0)}
            open_pos['peak']=0.0
            last_entry=t5[i]; flag=False
            slots.append((p, t5[i]+180*60000))  # reserve slot till worst-case close; refined below
        if open_pos is not None:
            ent=open_pos['entry']
            open_pos.update(pnl=(ent/cl[-1]-1)*100-FEE, reason="EOD", close_ts=t5[-1]); ep_trades.append(open_pos)
    for tr in ep_trades: all_trades.append(tr)
    n=len(ep_trades); wns=sum(1 for t in ep_trades if t['pnl']>0)
    print(f"\nEP {fmt(w0)} -> {fmt(w1)} ({(w1-w0)/3600000:.1f}h) pairs={','.join(pairs[:5])}... trades={n} W={wns} sum={sum(t['pnl'] for t in ep_trades):+.2f}%")
    for t in sorted(ep_trades,key=lambda x:x['ts']):
        print(f"   {t['pair']:14s} {fmt(t['ts'])} atr={t['atr']:.2f} peak={t['peak']:+.2f} pnl={t['pnl']:+.2f}% {t['reason']}")
n=len(all_trades); wn=sum(1 for t in all_trades if t['pnl']>0)
print(f"\n===== TOTAL: {n} trades | WR {100*wn/max(n,1):.0f}% | sum {sum(t['pnl'] for t in all_trades):+.2f}% | avg {sum(t['pnl'] for t in all_trades)/max(n,1):+.3f}%/trade | windows {len(episodes)}")
json.dump(all_trades, open(f"{S}/bearrun_trades.json",'w'))
