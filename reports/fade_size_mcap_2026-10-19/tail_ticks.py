import sys, os, json, time, zipfile, urllib.request, pandas as pd, numpy as np
R="/Users/guillearslanian/Downloads/NOFA AI"; DL="/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/dl_ticks"
sys.path.insert(0, R+"/scripts"); import binance_cooldown as bc
b=pd.read_csv(R+'/reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv',low_memory=False)
f=b[(b.entry_strategy=='SPIKE_FADE')&(b.close_reason=='STOP_LOSS L1')].sort_values('pct').head(int(sys.argv[1]))
rows=[]
for _,r in f.iterrows():
    t0=pd.Timestamp(r.opened_at); t1=pd.Timestamp(r.closed_at)
    days=sorted({t0.date(), t1.date()})
    fr=[]
    for d in days:
        fn=f"{DL}/{r.pair}-aggTrades-{d}.csv"
        if not os.path.exists(fn):
            if bc.active(): sys.exit('cooldown')
            url=f"https://data.binance.vision/data/futures/um/daily/aggTrades/{r.pair}/{r.pair}-aggTrades-{d}.zip"
            try:
                zp=fn+'.zip'; urllib.request.urlretrieve(url, zp); zipfile.ZipFile(zp).extractall(DL); os.remove(zp)
            except Exception as e:
                print('miss', r.pair, d, e); continue
            time.sleep(0.5)
        x=pd.read_csv(fn); 
        if 'transact_time' not in x.columns:
            x=pd.read_csv(fn, header=None, names=['agg_trade_id','price','quantity','first_trade_id','last_trade_id','transact_time','is_buyer_maker'])
        fr.append(x)
    if not fr: continue
    x=pd.concat(fr); x['t']=pd.to_datetime(x.transact_time,unit='ms')
    w=x[(x.t>=t0)&(x.t<=t1+pd.Timedelta('10min'))].sort_values('t')
    e=r.entry_price; fee=r.total_fee/r.notional_value*100
    line=e*(1+(1.5-fee)/100)     # SHORT: stop price where net pnl = -1.5
    hit=w[w.price>=line]
    if hit.empty: rows.append(dict(pair=r.pair, opened=r.opened_at, bt_pct=round(r.pct,3), tick='no stop hit in ticks')); continue
    h=hit.iloc[0]; after=w[(w.t>=h.t+pd.Timedelta('1s'))]; p1=after.iloc[0].price if len(after) else h.price
    pre=w[w.t<h.t].price.iloc[-1] if (w.t<h.t).any() else e
    pct=lambda p: round(-(p/e-1)*100-fee,3)
    rows.append(dict(pair=r.pair, opened=r.opened_at, vol24=r.entry_pair_volume_24h_usd, bt_pct=round(r.pct,3), tick_first_through=pct(h.price), tick_plus1s=pct(p1), last_before=pct(pre), hit_at=str(h.t)))
out=pd.DataFrame(rows); out.to_csv('tail_ticks.csv',index=False); print(out.to_string())
