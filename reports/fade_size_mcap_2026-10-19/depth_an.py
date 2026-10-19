import json, pandas as pd, numpy as np
R="/Users/guillearslanian/Downloads/NOFA AI"
d=pd.read_csv('depth_snap.csv')
B=json.load(open(R+'/reports/cache_vol_mcap/bapi_supply.json')); X=json.load(open('bapi_extra.json'))
def cs_mult(p):
    v=B.get(p) or {}
    if v.get('cs'): return v['cs'], v.get('mult',1)
    v=X.get(p)
    if v:
        full=p[:-4].lower(); mult=1
        if v['symbol']!=full:
            import re; mm=re.match(r'^(1000000|100000|10000|1000|100)',p); mult=int(mm.group(1)) if mm else 1
        return v['cs'], mult
    return None, None
d['mcap']=[ (lambda c,m: c*l/m if c else np.nan)(*cs_mult(p)) for p,l in zip(d.pair,d['last'])]
for N in (6000,12000): d[f'imp_{N}']=d[[f'buy_{N}',f'sell_{N}']].mean(axis=1)
d['depth1']=d[['buy_depth_1.0','sell_depth_1.0']].mean(axis=1)
u=d[(d.vol24>=1.5e6)&(d.vol24<=30e6)].dropna(subset=['imp_12000'])
print('N pairs vol 1.5-30M:',len(u),' with mcap',u.mcap.notna().sum())
from numpy.linalg import lstsq
def fit(df, cols, y='imp_12000'):
    X=np.column_stack([np.ones(len(df))]+[np.log(df[c]) for c in cols]); Y=np.log(df[y])
    b,res,_,_=lstsq(X,Y,rcond=None); pred=X@b; r2=1-((Y-pred)**2).sum()/((Y-Y.mean())**2).sum(); return b.round(3), round(r2,3)
um=u.dropna(subset=['mcap'])
print('log imp12k ~ log vol      ', fit(um,['vol24']))
print('log imp12k ~ log mcap     ', fit(um,['mcap']))
print('log imp12k ~ log vol+mcap ', fit(um,['vol24','mcap']))
print('log depth1 ~ vol', fit(um,['vol24'],'depth1'), ' ~ mcap', fit(um,['mcap'],'depth1'), ' ~ both', fit(um,['vol24','mcap'],'depth1'))
spearmanr=lambda a,b: (pd.Series(np.asarray(a)).rank().corr(pd.Series(np.asarray(b)).rank()),)
print('spearman imp12k vs vol', spearmanr(um.imp_12000, um.vol24)[0].round(3), 'vs mcap', spearmanr(um.imp_12000, um.mcap)[0].round(3), 'vs vol/mcap', spearmanr(um.imp_12000, um.vol24/um.mcap)[0].round(3))
u=u.copy(); u['vb']=pd.cut(u.vol24,[0,3e6,6e6,12e6,20e6,31e6]); u['mb']=pd.cut(u.mcap,[0,50e6,200e6,1e9,1e13])
print(u.groupby('vb',observed=True)[['imp_6000','imp_12000','spread_pct','depth1']].median().round(3).assign(n=u.groupby('vb',observed=True).size()))
print(u.groupby('mb',observed=True)[['imp_6000','imp_12000','spread_pct','depth1','vol24']].median().round(3).assign(n=u.groupby('mb',observed=True).size()))
print(pd.crosstab(u.vb,u.mb,values=u.imp_12000,aggfunc='median').round(3)); print(pd.crosstab(u.vb,u.mb))
print(d[d.pair=='KMNOUSDT'][['pair','vol24','mcap','imp_6000','imp_12000','spread_pct','depth1']])
d.to_csv('depth_snap_mcap.csv',index=False)
