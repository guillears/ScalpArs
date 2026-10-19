import pandas as pd, numpy as np
d=pd.read_csv('depth_snap_mcap.csv'); d=d[(d.vol24>=1.5e6)&(d.vol24<=30e6)]
rows=[]
for N in (3000,6000,12000,24000):
    for s in ('buy','sell'):
        x=d[['pair','vol24','mcap',f'{s}_{N}']].rename(columns={f'{s}_{N}':'imp'}); x['N']=N; rows.append(x)
L=pd.concat(rows).dropna(subset=['imp']); L=L[L.imp>0]
def fit(df, cols):
    X=np.column_stack([np.ones(len(df))]+[np.log(df[c]) for c in cols]); Y=np.log(df.imp)
    b=np.linalg.lstsq(X,Y,rcond=None)[0]; p=X@b; r2=1-((Y-p)**2).sum()/((Y-Y.mean())**2).sum(); return b, r2
for cols in (['N','vol24'],['N','mcap'],['N','vol24','mcap']):
    sub=L.dropna(subset=[c for c in cols])
    b,r2=fit(sub,cols); print(cols, b.round(3), round(r2,3), len(sub))
