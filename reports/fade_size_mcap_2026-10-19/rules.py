import numpy as np, pandas as pd, sys
exec(open('analyse.py').read().split("# ---------- per-row arrays")[0])   # model + rule defs (prints terciles)
A['is_stop']=A.is_stop.astype(bool)
def path_dd(days_r):
    eq=np.cumprod(1+np.maximum(days_r,-1)); pk=np.maximum.accumulate(np.r_[1,eq])[1:]; return (eq/pk-1).min()
# ---------- day arrays per source for bootstrap
def day_struct(src, start, end):
    S=A[A.src==src]; days=pd.date_range(start,end,freq='D'); di={d:i for i,d in enumerate(days)}
    idx=np.array([di[d] for d in S.day]); K=max(1,np.bincount(idx).max()); D=len(days)
    slot=np.full((D,K),-1); cnt=np.zeros(D,int)
    for r,i in zip(S.index.values, idx): slot[i,cnt[i]]=r; cnt[i]+=1
    return slot
STRUCT={'BT':day_struct('BT','2025-01-01','2026-10-11'),'MASTER':day_struct('MASTER','2026-07-28','2026-10-13')}
def rule_vec(name,E):   # E: array of equities (paths) -> N for every row x path is heavy; compute N per row as function of E via pieces
    pass
def simulate(src,name,scale,E0=E_LIVE,P=4000,L=365,blk=7):
    slot=STRUCT[src]; D,K=slot.shape; nb=int(np.ceil(L/blk))
    starts=RNG.integers(0,D-blk+1,(P,nb)); seq=(starts[:,:,None]+np.arange(blk)).reshape(P,-1)[:,:L]
    E=np.full(P,E0); peak=E.copy(); mdd=np.zeros(P)
    V=A.V.values; M=A.mcap.values; brk_=A.entry_bracket_cap_notional.where(A.entry_bracket_cap_notional>0).fillna(1e12).values
    pct0=A.pct0.values; stop=A.is_stop.values; live=(A.src=='LIVE').values
    for s in range(L):
        rows=slot[seq[:,s]]       # P x K
        valid=rows>=0; r=np.where(valid,rows,0)
        b=(E*0.975/4*20)[:,None]; Vr=V[r]; Mr=M[r]; liq=0.005*Vr; br=brk_[r]
        two=np.minimum(np.minimum(2*b,liq),br); one=np.minimum(np.minimum(b,liq),br); Er=E[:,None]
        if name=='a today 2x': N=two
        elif name=='b 1x all': N=one
        elif name=='b2 0.5x all': N=np.minimum(np.minimum(0.5*b,liq),br)
        elif name=='c1 2x if mcap<T1 else 1x': N=np.where(np.isfinite(Mr)&(Mr<T1),two,one)
        elif name=='c2 2x if mcap>T2 else 1x': N=np.where(np.isfinite(Mr)&(Mr>T2),two,one)
        elif name.startswith('d liq'): N=np.minimum(two,float(name.split()[2].rstrip('%'))/100*Vr)
        elif name.startswith('e acct'): N=np.minimum(two,float(name.split()[2].rstrip('xE'))*Er)
        elif name=='f 2x, <=5xE, <=0.1%V': N=np.minimum(np.minimum(two,5*Er),0.001*Vr)
        elif name=='f 1x, <=0.1%V': N=np.minimum(one,0.001*Vr)
        elif name=='f 1x, <=0.05%V': N=np.minimum(one,0.0005*Vr)
        elif name=='f 0.5x, <=0.1%V': N=np.minimum(np.minimum(np.minimum(0.5*b,liq),br),0.001*Vr)
        elif name=='f 2x, <=3xE, <=0.1%V': N=np.minimum(np.minimum(two,3*Er),0.001*Vr)
        elif name=='f 2x if mcap>T2 else 1x, <=5xE': N=np.minimum(np.where(np.isfinite(Mr)&(Mr>T2),two,one),5*Er)
        pe=pct0[r]-impact(np.maximum(N,1),Vr,Mr,stop[r],scale)
        ret=np.where(valid,pe*N/Er/100,0).sum(1)
        E=E*np.maximum(1+ret,0.0); peak=np.maximum(peak,E); mdd=np.minimum(mdd,E/peak-1)
    return (mdd<=-0.30).mean(),(mdd<=-0.50).mean(),np.median(E/E0),(E/E0<=0.1).mean()
def static(src,name,E,scale):
    S=(A.src==src).values; N=rule_N(name,E)[S]; Nt=rule_N('a today 2x',E)[S]
    full=eff_pct(rule_N(name,E),scale)[S]
    usd=full*N/100; acct=usd/E*100
    return dict(N=S.sum(), shrunk=int((N<Nt-1).sum()), avg_pct=full.mean(), sum_acct_pct=acct.sum(), usd=usd.sum(), worst_acct=acct.min(), med_notional_xE=np.median(N)/E)
out=[]
RUN=sys.argv[1].split('|') if len(sys.argv)>1 else RULES
SCALES=[float(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [1.0,0.0]
for name in RUN:
    for src in ('MASTER','BT','LIVE'):
        r=static(src,name,E_LIVE,1.0); r0=static(src,name,E_LIVE,0.0); rh=static(src,name,E_LIVE,0.5)
        rr=dict(rule=name,src=src,**{k:r[k] for k in ('N','shrunk','med_notional_xE','avg_pct','sum_acct_pct','usd','worst_acct')},
                avg_pct_noimp=r0['avg_pct'],usd_noimp=r0['usd'],avg_pct_half=rh['avg_pct'],usd_half=rh['usd'])
        if src=='MASTER':
            p=static(src,name,E_PAPER,0.0); rr['usd_paper_noimp']=p['usd']
        if src in STRUCT:
            for sc in SCALES:
                tag='' if sc==1.0 else ('_noimp' if sc==0 else f'_s{sc}')
                p30,p50,med,ruin=simulate(src,name,sc,P=2000); rr['P30'+tag]=p30; rr['P50'+tag]=p50; rr['medx'+tag]=med
        out.append(rr); print(rr, flush=True)
pd.DataFrame(out).to_csv(sys.argv[3] if len(sys.argv)>3 else 'rules_out.csv',index=False)
