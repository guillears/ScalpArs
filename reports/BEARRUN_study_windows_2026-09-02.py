import urllib.request, json, datetime, time, sys
def kl(sym, interval, limit, end=None):
    u=f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&limit={limit}"+(f"&endTime={end}" if end else "")
    for a in range(3):
        try: return json.load(urllib.request.urlopen(u, timeout=20))
        except Exception as e:
            time.sleep(1+a)
    raise SystemExit(f"fetch failed {sym}")
# ~6.5 months of BTC 5m
rows=[]; end=None
for _ in range(58):
    d=kl("BTCUSDT","5m",1000,end)
    rows=d+rows; end=d[0][0]-1
    if len(rows)>=57000: break
seen=set(); rows=[r for r in rows if not (r[0] in seen or seen.add(r[0]))]
rows.sort(key=lambda r:r[0])
closes=[float(r[4]) for r in rows]; ts=[r[0] for r in rows]
print("bars:", len(rows), datetime.datetime.fromtimestamp(ts[0]/1000,datetime.UTC), "->", datetime.datetime.fromtimestamp(ts[-1]/1000,datetime.UTC))
W=864
ema=closes[0]; k=2/21; above=[]
for c in closes: ema=c*k+ema*(1-k); above.append(c>ema)
h=kl("BTCUSDT","1h",1000); h2=kl("BTCUSDT","1h",1000,h[0][0]-1); h3=kl("BTCUSDT","1h",1000,h2[0][0]-1); h4=kl("BTCUSDT","1h",1000,h3[0][0]-1); h5=kl("BTCUSDT","1h",1000,h4[0][0]-1)
hh=h5+h4+h3+h2+h
sh=set(); hh=[r for r in hh if not (r[0] in sh or sh.add(r[0]))]; hh.sort(key=lambda r:r[0])
hc=[float(r[4]) for r in hh[:-1]]; hts=[r[0] for r in hh[:-1]]
e=hc[0]; kk=2/51; e50=[]
for c in hc: e=c*kk+e*(1-kk); e50.append(e)
import bisect
def e50_at(t):
    i=bisect.bisect_right(hts,t)-1
    return e50[i] if i>=0 else None
red=False; windows=[]; cur=None
for i in range(W, len(closes)):
    win=closes[i-W+1:i+1]
    r72=(win[-1]/win[0]-1)*100
    below=100*(1-sum(above[i-W+1:i+1])/W)
    diffs=sum(abs(win[j]-win[j-1]) for j in range(1,W))
    eff=abs(win[-1]-win[0])/diffs if diffs else 0
    r6=(win[-1]/win[-72]-1)*100
    p=win[-1]; ev=e50_at(ts[i])
    latch=(r6>=3.0) or (ev is not None and p>ev)
    if latch: nr=False
    elif red: nr=(r72<=-4.0 and below>=53.0 and eff>=0.10)
    else:     nr=(r72<=-5.0 and below>=56.0 and eff>=0.10)
    if nr and not red: cur={'start':ts[i],'end':None,'ended_by':None}
    if red and not nr and cur:
        cur['end']=ts[i]; cur['ended_by']='latch' if latch else 'stay-band'; windows.append(cur); cur=None
    red=nr
if cur: cur['end']=ts[-1]; cur['ended_by']='open'; windows.append(cur)
print("RED windows:", len(windows))
for w in windows:
    d=(w['end']-w['start'])/3600000
    print(" ", datetime.datetime.fromtimestamp(w['start']/1000,datetime.UTC).strftime('%Y-%m-%d %H:%M'), "->",
          datetime.datetime.fromtimestamp(w['end']/1000,datetime.UTC).strftime('%m-%d %H:%M'), f"({d:.1f}h, end={w['ended_by']})")
json.dump(windows, open(sys.argv[1],'w'))
# BTC 24h-low proximity series saved for the bounce-phase gate (mirror of off24h)
json.dump({'ts':ts,'closes':closes}, open(sys.argv[2],'w'))
