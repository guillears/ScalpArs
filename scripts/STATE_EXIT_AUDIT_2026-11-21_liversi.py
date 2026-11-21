# STATE_EXIT_AUDIT_2026-11-21 — audit helper (liversi). Research only, read-only on study data. Run from the repo root: venv/bin/python scripts/STATE_EXIT_AUDIT_2026-11-21_liversi.py
import importlib.util, numpy as np, pandas as pd
sp=importlib.util.spec_from_file_location("EV","scripts/STATE_EXIT_2026-11-21_eval.py"); EV=importlib.util.module_from_spec(sp); sp.loader.exec_module(EV)
C=EV.C; MIN=C.MIN; A=1/14
O=pd.read_csv(C.D+'/fills_out.csv'); O=O[(O.sleeve=='HT')&O.base_net.notna()].copy()
O['day']=pd.to_datetime(O.te,unit='ms').dt.strftime('%Y-%m-%d')
dec=pd.read_pickle(C.D+'/dec_HT.pkl').sort_values(['fid','k']).reset_index(drop=True)
dec=dec[dec.peak>=2].copy()
out=[]
for r in O.itertuples():
    d=dec[dec.fid==r.fid]
    if not len(d): continue
    t0=(r.te//MIN)*MIN+(MIN if r.start=='next' else 0)
    a=r.te-26*3600_000; a=a//MIN*MIN
    raw=C.k1m(r.pair,a,t0+(int(d.k.max())+2)*MIN)
    if raw is None: continue
    g=C.grid(raw,a,t0+(int(d.k.max())+2)*MIN); T=g['t']; c=g['c']
    # 1m RSI (Wilder)
    dd=np.diff(c,prepend=c[0]); up=pd.Series(np.maximum(dd,0)).ewm(alpha=A,adjust=False).mean().values; dn=pd.Series(np.maximum(-dd,0)).ewm(alpha=A,adjust=False).mean().values
    r1=100-100/(1+up/np.where(dn==0,1e-12,dn))
    # closed 5m bars
    bid=T//(5*MIN); last_of_bar=np.r_[bid[1:]!=bid[:-1],True]
    # first full bar start
    idx_close=np.flatnonzero(last_of_bar & ((T+MIN)%(5*MIN)==0))
    cc=c[idx_close]; d5=np.diff(cc,prepend=cc[0])
    u5=pd.Series(np.maximum(d5,0)).ewm(alpha=A,adjust=False).mean().values; n5=pd.Series(np.maximum(-d5,0)).ewm(alpha=A,adjust=False).mean().values
    live=np.full(len(T),np.nan)
    j=np.searchsorted(idx_close,np.arange(len(T)),'right')-1   # last closed bar index at/before minute
    for m in range(len(T)):
        jj=j[m]
        if jj<1: continue
        if idx_close[jj]==m:
            live[m]=100-100/(1+u5[jj]/max(n5[jj],1e-12))
        else:
            dl=c[m]-cc[jj]; uu=(1-A)*u5[jj]+A*max(dl,0); nn=(1-A)*n5[jj]+A*max(-dl,0)
            live[m]=100-100/(1+uu/max(nn,1e-12))
    i0=int(np.searchsorted(T,t0)); ks=d.k.values.astype(int)+i0
    # in-trade running max from i0
    lv=live.copy(); lv[:i0]=np.nan; lmax=pd.Series(lv).cummax().values
    o1=r1.copy(); o1[:i0]=np.nan; omax=pd.Series(o1).cummax().values
    out.append(pd.DataFrame(dict(fid=r.fid,k=d.k.values,live=live[ks],lmax=lmax[ks],r1=r1[ks],r1max=omax[ks],rsi5=d.rsi_5m.values,rmax=d.rsi_max_in.values,label=d.label.values,pnl=d.pnl.values)))
X=pd.concat(out,ignore_index=True)
X.to_pickle('reports/STATE_EXIT_AUDIT_2026-11-21_ht_live.pkl')
rules={'PREREG closed-5m (max≥75, −10)':(X.rmax>=75)&(X.rsi5<=X.rmax-10),
       'live-5m (max≥75, −10)':(X.lmax>=75)&(X.live<=X.lmax-10),
       'closed max≥75, live now −10':(X.rmax>=75)&(X.live<=X.rmax-10),
       '1m RSI (max≥75, −10)':(X.r1max>=75)&(X.r1<=X.r1max-10),
       '1m RSI (max≥80, −20)':(X.r1max>=80)&(X.r1<=X.r1max-20)}
res=[]
for nm,m in rules.items():
    for per in ['MASTER','BATCH','DESIGN','HOLDOUT','CASE']:
        fids=O[O.period==per].fid.values; xp=X[X.fid.isin(fids)]
        dl,kk=EV.per_fill_delta(xp.reset_index(drop=True),m[xp.index].values,fids)
        o=O.set_index('fid').loc[dl.index]
        lo,hi,pp=EV.boot(dl.values,o.day.values) if per!='CASE' else (np.nan,np.nan,np.nan)
        tp=(o.base_why=='TP')
        res.append(dict(rule=nm,per=per,fired=len(kk),saves=int(((dl>0)&~tp).sum()),chops=int(((dl<0)&tp).sum()),d=round(dl.mean(),3),P=round(pp,2)))
R=pd.DataFrame(res); print(R.to_string(index=False))
