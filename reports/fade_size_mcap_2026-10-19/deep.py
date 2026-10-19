import numpy as np, pandas as pd, sys
exec(open('analyse.py').read().split("# ---------- per-row arrays")[0])
A['is_stop']=A.is_stop.astype(bool)
M=A.mcap.values; fin=np.isfinite(M)
def ladder(spec):   # spec: list of (upper_bound, mult); unscored -> unscored mult
    def f(unscored):
        out=np.full(len(A),unscored,float)
        for i in np.where(fin)[0]:
            for ub,mu in spec:
                if M[i]<ub: out[i]=mu; break
        return out
    return f
INF=np.inf
# rule = (invest mult per row, liq pct, account cap k or None)
RULES={
 '(0) today 2x, liq 0.5%':            (np.full(len(A),2.0),0.5,None),
 '(a) REC 1x, liq 0.05%':             (np.full(len(A),1.0),0.05,None),
 '(b) 1x if mcap>=200M else 2x (unsc 1x)': (ladder([(200e6,2.0),(INF,1.0)])(1.0),0.5,None),
 '(b2) 1x if mcap>=300M else 2x (unsc 1x)': (ladder([(300e6,2.0),(INF,1.0)])(1.0),0.5,None),
 '(c) 1x flat, liq 0.5%':             (np.full(len(A),1.0),0.5,None),
 '(d) ladder 2/1.5/1/0.5x (<50M/<200M/<500M/>=500M, unsc 1x)': (ladder([(50e6,2.0),(200e6,1.5),(500e6,1.0),(INF,0.5)])(1.0),0.5,None),
 '(d2) inverse ladder 0.5/1/1.5x (<50M/<200M/>=200M, unsc 0.5x)': (ladder([(50e6,0.5),(200e6,1.0),(INF,1.5)])(0.5),0.5,None),
 '(e) REC + notional <= 3x acct':      (np.full(len(A),1.0),0.05,3.0),
 '(e2) REC + notional <= 2x acct':     (np.full(len(A),1.0),0.05,2.0),
 '(f) REC + 0.5x if mcap>=200M (unsc 1x)': (ladder([(200e6,1.0),(INF,0.5)])(1.0),0.05,None),
}
brk=A.entry_bracket_cap_notional.where(A.entry_bracket_cap_notional>0).fillna(1e12).values
def N_of(rule,E):
    mu,lp,k=RULES[rule]; b=E*0.975/4*20
    N=np.minimum.reduce([mu*b, lp/100*A.V.values, brk])
    if k: N=np.minimum(N,k*E)
    return N
STRUCT={'BT':None,'MASTER':None}
def day_struct(src, start, end):
    S=A[A.src==src]; days=pd.date_range(start,end,freq='D'); di={d:i for i,d in enumerate(days)}
    idx=np.array([di[d] for d in S.day]); K=max(1,np.bincount(idx).max()); D=len(days)
    slot=np.full((D,K),-1); cnt=np.zeros(D,int)
    for r,i in zip(S.index.values, idx): slot[i,cnt[i]]=r; cnt[i]+=1
    return slot
STRUCT={'BT':day_struct('BT','2025-01-01','2026-10-11'),'MASTER':day_struct('MASTER','2026-07-28','2026-10-13')}
def simulate(src,rule,scale,E0=E_LIVE,P=2000,L=365,blk=7):
    mu,lp,k=RULES[rule]; slot=STRUCT[src]; D,K=slot.shape; nb=int(np.ceil(L/blk))
    starts=RNG.integers(0,D-blk+1,(P,nb)); seq=(starts[:,:,None]+np.arange(blk)).reshape(P,-1)[:,:L]
    E=np.full(P,E0); peak=E.copy(); mdd=np.zeros(P); V=A.V.values; pct0=A.pct0.values; st=A.is_stop.values
    for s in range(L):
        rows=slot[seq[:,s]]; valid=rows>=0; r=np.where(valid,rows,0); Er=E[:,None]
        N=np.minimum.reduce([mu[r]*Er*0.975/4*20, lp/100*V[r], brk[r]*np.ones_like(Er)])
        if k: N=np.minimum(N,k*Er)
        pe=pct0[r]-impact(np.maximum(N,1),V[r],M[r],st[r],scale)
        E=E*np.maximum(1+np.where(valid,pe*N/Er/100,0).sum(1),0); peak=np.maximum(peak,E); mdd=np.minimum(mdd,E/peak-1)
    return (mdd<=-0.3).mean(),(mdd<=-0.5).mean()
if __name__=='__main__':
    out=[]
    Nt=N_of('(0) today 2x, liq 0.5%',E_LIVE)
    for rule in RULES:
        N=N_of(rule,E_LIVE)
        for sc,tag in ((1.0,'imp'),(0.5,'half'),(0.0,'paper')):
            pe=eff_pct(N,sc); usd=pe*N/100; acct=usd/E_LIVE*100
            for src in ('MASTER','BT','LIVE'):
                S=(A.src==src).values
                rr=dict(rule=rule,impact=tag,src=src,N=S.sum(),shrunk=int((N[S]<Nt[S]-1).sum()),med_xE=np.median(N[S])/E_LIVE,max_xE=N[S].max()/E_LIVE,
                        avg=pe[S].mean(),usd=usd[S].sum(),worst_acct=acct[S].min())
                if src!='LIVE' and tag!='half':
                    rr['P30'],rr['P50']=simulate(src,rule,sc)
                out.append(rr)
        print(rule,flush=True)
    pd.DataFrame(out).to_csv('deep_rules.csv',index=False)
