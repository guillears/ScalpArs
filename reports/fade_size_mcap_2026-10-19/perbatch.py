import numpy as np, pandas as pd
exec(open('analyse.py').read().split("# ---------- per-row arrays")[0])
R=['a today 2x','b 1x all','b2 0.5x all','c1 2x if mcap<T1 else 1x','c2 2x if mcap>T2 else 1x','d liq 0.05%','e acct 5xE','f 1x, <=0.05%V','f 1x, <=0.1%V']
out={}
for sc,tag in ((1.0,'imp'),(0.0,'noimp')):
    rows=[]
    for name in R:
        N=rule_N(name,E_LIVE); pe=eff_pct(N,sc); usd=pe*N/100
        df=pd.DataFrame(dict(src=A.src,batch=A.batch,usd=usd,pe=pe))
        g=df.groupby(['src','batch']).usd.sum().rename(name); rows.append(g)
    t=pd.concat(rows,axis=1)
    order=sorted(t.index, key=lambda x: (['MASTER','BT','LIVE'].index(x[0]), int(x[1][1:]) if x[0]=='MASTER' else 0, x[1]))
    t=t.loc[order]; t['N']=A.groupby(['src','batch']).size().reindex(t.index)
    tot=t.groupby(level=0).sum(); tot.index=pd.MultiIndex.from_tuples([(s,'TOTAL') for s in tot.index])
    t=pd.concat([t,tot]); out[tag]=t
    print('\n####',tag); print(t.round(0).to_string())
    # batches positive count
    mb=t.loc['MASTER'].drop('TOTAL'); print('master batches better than today:', {c:int((mb[c]>mb['a today 2x']+0.5).sum()) for c in R[1:]}, 'of', len(mb))
pd.concat(out).to_csv('perbatch.csv')
