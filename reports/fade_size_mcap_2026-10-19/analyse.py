import pandas as pd, numpy as np, sys, json
pd.set_option('display.width',250); pd.set_option('display.max_columns',40)
A=pd.read_csv('fades_all.csv'); A['day']=pd.to_datetime(A.day)
A.loc[A.src=='MASTER','is_stop']=A.loc[A.src=='MASTER','pct']<=-1.49
A.loc[A.src=='MASTER','overshoot']=np.nan
RNG=np.random.default_rng(20261019)
E_LIVE=1225.0; E_PAPER=2900.0
def base1(E): return E*0.975/4*20
# ---------- impact model (calm book fit on 165-pair snapshot 2026-10-19; KMNO-calibrated stress multipliers)
CONTAM=1.8; M_ENTRY=1.4; M_STOP=2.05; M_OTHER=1.0
def calm(N,V,M):
    V=np.asarray(V,float)/CONTAM; M=np.asarray(M,float); N=np.asarray(N,float)
    full=np.exp(-0.704+0.545*np.log(N)-0.189*np.log(V)-0.188*np.log(np.where(np.isfinite(M)&(M>0),M,1)))
    volo=np.exp(-1.867+0.546*np.log(N)-0.336*np.log(V))
    return np.where(np.isfinite(M)&(M>0), full, volo)
def impact(N,V,M,stop,scale=1.0):
    c=calm(N,V,M); return scale*c*(M_ENTRY+np.where(stop,M_STOP,M_OTHER))
# KMNO: measured ladder (entry+exit) at N; interpolate log-log
KN=np.array([1487,2973,5942,8908,11872.]); KI=np.array([0.039+0.049,0.076+0.109,0.133+0.200,0.193+0.274,0.239+0.345])
def kmno_imp(N): return np.exp(np.interp(np.log(np.maximum(N,1487)),np.log(KN),np.log(KI)))
islive=(A.src=='LIVE').values
A['pct0']=A.pct.copy()
A.loc[islive,'pct0']=A.loc[islive,'pct']+kmno_imp(A.loc[islive,'notional_value'].values)   # impact-free base for live
def eff_pct(N, scale=1.0):
    N=np.asarray(N,float)
    out=A.pct0.values-impact(N,A.V.values,A.mcap.values,A.is_stop.values,scale)
    if islive.any(): out[islive]=A.pct0.values[islive]-scale*kmno_imp(N[islive])
    return out
# ---------- mcap terciles (label-blind: all BT fades with a value)
bt=A[A.src=='BT']; T1,T2=np.nanquantile(bt.mcap,[1/3,2/3]); Vmed=np.nanmedian(bt.V)
print(f'BT mcap terciles T1={T1/1e6:.0f}M T2={T2/1e6:.0f}M ; vol median {Vmed/1e6:.1f}M')
brk=A.entry_bracket_cap_notional.where(A.entry_bracket_cap_notional>0).fillna(1e12).values
def rule_N(name,E):
    b=base1(E); V=A.V.values; M=A.mcap.values; liq=0.005*V
    two=np.minimum.reduce([2*b*np.ones(len(A)),liq,brk]); one=np.minimum.reduce([b*np.ones(len(A)),liq,brk])
    if name=='a today 2x': return two
    if name=='b 1x all': return one
    if name=='b2 0.5x all': return np.minimum(0.5*b*np.ones(len(A)),np.minimum(liq,brk))
    if name=='c1 2x if mcap<T1 else 1x': return np.where(np.isfinite(M)&(M<T1),two,one)
    if name=='c2 2x if mcap>T2 else 1x': return np.where(np.isfinite(M)&(M>T2),two,one)
    if name.startswith('d liq'):
        k=float(name.split()[2].rstrip('%'))/100; return np.minimum(two,k*V)
    if name.startswith('e acct'):
        mm=float(name.split()[2].rstrip('xE')); return np.minimum(two,mm*E)
    if name=='f 2x, <=5xE, <=0.1%V': return np.minimum.reduce([two,5*E*np.ones(len(A)),0.001*V])
    if name=='f 1x, <=0.1%V': return np.minimum(one,0.001*V)
    if name=='f 1x, <=0.05%V': return np.minimum(one,0.0005*V)
    if name=='f 0.5x, <=0.1%V': return np.minimum(np.minimum(0.5*b*np.ones(len(A)),np.minimum(liq,brk)),0.001*V)
    if name=='f 2x, <=3xE, <=0.1%V': return np.minimum.reduce([two,3*E*np.ones(len(A)),0.001*V])
    if name=='f 2x if mcap>T2 else 1x, <=5xE': return np.minimum(np.where(np.isfinite(M)&(M>T2),two,one),5*E)
    raise KeyError(name)
RULES=['a today 2x','b 1x all','b2 0.5x all','c1 2x if mcap<T1 else 1x','c2 2x if mcap>T2 else 1x','d liq 0.05%','d liq 0.1%','d liq 0.2%',
       'e acct 3xE','e acct 4xE','e acct 5xE','e acct 6xE','f 2x, <=5xE, <=0.1%V','f 1x, <=0.1%V','f 2x if mcap>T2 else 1x, <=5xE']
# ---------- per-row arrays for E_LIVE
def wr(x): return (x>0).mean()*100
def day_ci(df,col,nb=2000):
    g=df.groupby('day')[col].agg(['sum','size']).values
    if len(g)<3: return (np.nan,np.nan)
    idx=RNG.integers(0,len(g),(nb,len(g))); s=g[idx,0].sum(1)/g[idx,1].sum(1); return np.quantile(s,[.025,.975])
N_today=rule_N('a today 2x',E_LIVE); A['N_today']=N_today; A['nv_today']=N_today/A.V*100
A['imp_today']=impact(N_today,A.V,A.mcap,A.is_stop); A.loc[islive,'imp_today']=kmno_imp(A.loc[islive,'notional_value'].values)
A['pct_live2x']=eff_pct(N_today); A['pct_live1x']=eff_pct(rule_N('b 1x all',E_LIVE))
def bucket_table(col, bins, labels, title):
    print(f'\n### {title}')
    rows=[]
    for src in ('MASTER','BT','LIVE'):
        S=A[A.src==src]
        for lab,lo,hi in labels:
            x=S[(S[col]>=lo)&(S[col]<hi)] if lab!='unscored' else S[S[col].isna()]
            if not len(x): continue
            st=x[x.is_stop]; ci=day_ci(x,'pct') if src!='LIVE' else (np.nan,np.nan)
            rows.append(dict(src=src,bucket=lab,N=len(x),WR=round(wr(x.pct)),avg=x.pct.mean(),ci_lo=ci[0],ci_hi=ci[1],sum=x.pct.sum(),worst=x.pct.min(),
                             stops=len(st),ovs_mean=st.overshoot.mean(),ovs_p90=st.overshoot.quantile(.9) if len(st) else np.nan,tail_le2=(x.pct<=-2).sum(),
                             imp2x=x.imp_today.mean(),avg_live2x=x.pct_live2x.mean(),avg_live1x=x.pct_live1x.mean()))
        x=S; st=x[x.is_stop]
        rows.append(dict(src=src,bucket='TOTAL',N=len(x),WR=round(wr(x.pct)),avg=x.pct.mean(),ci_lo=np.nan,ci_hi=np.nan,sum=x.pct.sum(),worst=x.pct.min(),stops=len(st),
                         ovs_mean=st.overshoot.mean(),ovs_p90=st.overshoot.quantile(.9) if len(st) else np.nan,tail_le2=(x.pct<=-2).sum(),imp2x=x.imp_today.mean(),
                         avg_live2x=x.pct_live2x.mean(),avg_live1x=x.pct_live1x.mean()))
    t=pd.DataFrame(rows); print(t.round(3).to_string(index=False)); return t
inf=np.inf
tm=bucket_table('mcap',None,[('<50M',0,50e6),('50-200M',50e6,200e6),('200M-1B',200e6,1e9),('>1B',1e9,inf),('unscored',0,0)],'by MCAP (stamp or rebuilt = circulating supply today x entry price)')
tv=bucket_table('V',None,[('<3M (N/V>0.40%)',0,3e6),('3-6M (0.20-0.40%)',3e6,6e6),('6-12M (0.10-0.20%)',6e6,12e6),('>=12M (<0.10%)',12e6,inf)],'by 24h VOLUME (= notional/volume at today\'s $11.9k ticket)')
A['vm']=A.V/A.mcap
tr=bucket_table('vm',None,[('vol/mcap<0.02',0,0.02),('0.02-0.05',0.02,0.05),('0.05-0.15',0.05,0.15),('>=0.15',0.15,inf),('unscored',0,0)],'by VOL/MCAP turnover')
# ---------- predictors of overshoot and tail (BT stops; spearman)
def sp(a,b):
    m=np.isfinite(a)&np.isfinite(b); return pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()), m.sum()
st=A[(A.src=='BT')&A.is_stop]; print('\n### BT stop overshoot predictors (Spearman, N stops)')
for c in ('mcap','V','vm'): print(c, [round(v,3) if isinstance(v,float) else v for v in sp(np.log(st[c].values),st.overshoot.values)])
allbt=A[A.src=='BT']; print('tail (pct<=-2) predictors over all BT fills: ')
for c in ('mcap','V','vm'): print(c, [round(v,3) if isinstance(v,float) else v for v in sp(np.log(allbt[c].values),(allbt.pct<=-2).astype(float).values)])
print('BT avg pct predictors:')
for c in ('mcap','V','vm'): print(c, [round(v,3) if isinstance(v,float) else v for v in sp(np.log(allbt[c].values),allbt.pct.values)])
mm=A[A.src=='MASTER']; print('MASTER avg pct predictors:')
for c in ('mcap','V','vm'): print(c, [round(v,3) if isinstance(v,float) else v for v in sp(np.log(mm[c].values),mm.pct.values)])
A.to_csv('fades_scored.csv',index=False)
json.dump(dict(T1=T1,T2=T2),open('terciles.json','w'))
tm.to_csv('t_mcap.csv',index=False); tv.to_csv('t_vol.csv',index=False); tr.to_csv('t_vm.csv',index=False)

print('model impact KMNO at 11.9k', impact(np.array([11872.]),np.array([17.8e6]),np.array([332e6]),np.array([True]))[0])
