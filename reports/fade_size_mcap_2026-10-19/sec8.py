import numpy as np, pandas as pd
exec(open('deep.py').read().split("if __name__")[0])
R_="/Users/guillearslanian/Downloads/NOFA AI"
def d(x): return (f"{x:+,.0f}").replace('-','−')
def p2(x): return "–" if pd.isna(x) else (f"{x:+.2f}").replace('-','−')
KEYS={'today':'(0) today 2x, liq 0.5%','rec':'(a) REC 1x, liq 0.05%','b':'(b) 1x if mcap>=200M else 2x (unsc 1x)','c':'(c) 1x flat, liq 0.5%','fin':'(e) REC + notional <= 3x acct'}
for k,r in KEYS.items():
    N=N_of(r,E_LIVE); A['N_'+k]=N; A['u_'+k]=eff_pct(N,1.0)*N/100; A['up_'+k]=eff_pct(N,0.0)*N/100
m=pd.read_csv(R_+'/reports/MASTER_POOL_stacked.csv',low_memory=False); m=m[(m.entry_strategy=='SPIKE_FADE')&(m.stack_keep==True)]
mm=A[A.src=='MASTER'].merge(m[['opened_at','pair','peak_pnl','trough_pnl']],on=['opened_at','pair'],how='left')
mm['bn']=mm.batch.str[1:].astype(int); mm=mm.sort_values(['bn','opened_at'])
L=["| batch | opened (UTC) | pair | mcap $M | 24h vol $M | peak % | trough % | exit | stack_pct | today 2× $ | final rule notional | final rule $ |","|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _,r in mm.iterrows():
    mc="unscored" if pd.isna(r.mcap) else f"{r.mcap/1e6:,.0f}"
    L.append(f"| {r.batch} | {r.opened_at[5:16].replace('T',' ')} | {r.pair} | {mc} | {r.V/1e6:.1f} | {p2(r.peak_pnl)} | {p2(r.trough_pnl)} | {r.close_reason} | {p2(r.pct)} | {d(r.u_today)} | ${r.N_fin:,.0f} | {d(r.u_fin)} |")
L.append(f"| **TOTAL** | | **74** | | | | | 8 losers (as traded) · 7 stops on today's stack | **{p2(mm.pct.sum())} Σ** | **{d(mm.u_today.sum())}** | median ${mm.N_fin.median():,.0f} | **{d(mm.u_fin.sum())}** |")
ledger="\n".join(L)
def btab(src):
    S=A[A.src==src]; out=["| mcap | N | WR | avg % | worst % | stops | gap mean | ≤ −2 % | today 2× $ | §6 rule $ | (b) 1× ≥ $200M $ | (c) 1× flat $ | **final rule $** | today paper $ |","|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def row(lab,x,bold=False):
        if not len(x): out.append(f"| {lab} | 0 | | | | | | | | | | | | |"); return
        st=x[x.is_stop]; g="–" if src=='MASTER' or not len(st) else f"{st.overshoot.mean():.2f}"
        b=lambda v: f"**{v}**" if bold else v
        out.append(f"| {b(lab)} | {b(len(x))} | {(x.pct>0).mean()*100:.0f} % | {p2(x.pct.mean())} | {p2(x.pct.min())} | {len(st)} | {g} | {(x.pct<=-2).sum()} | {d(x.u_today.sum())} | {d(x.u_rec.sum())} | {d(x.u_b.sum())} | {d(x.u_c.sum())} | **{d(x.u_fin.sum())}** | {d(x.up_today.sum())} |")
    for lab,lo,hi in (('< $50M',0,50e6),('$50–200M',50e6,200e6),('$200–300M',200e6,300e6),('$300–500M',300e6,500e6),('≥ $500M',500e6,np.inf)):
        row(lab,S[(S.mcap>=lo)&(S.mcap<hi)])
    for lab,lo in (('cum ≥ $200M',200e6),('cum ≥ $300M',300e6),('cum ≥ $500M',500e6)): row(lab,S[S.mcap>=lo])
    row('unscored',S[S.mcap.isna()]); row('TOTAL',S,True)
    out.append("| *by 24 h volume* | | | | | | | | | | | | | |")
    for lab,lo,hi in (('vol < $3M',0,3e6),('vol $3–6M',3e6,6e6),('vol $6–12M',6e6,12e6),('vol ≥ $12M',12e6,np.inf)): row(lab,S[(S.V>=lo)&(S.V<hi)])
    row('TOTAL',S,True)
    return "\n".join(out)
T={s:btab(s) for s in ('MASTER','BT','LIVE')}
pb=mm.groupby('batch').agg(bn=('bn','first'),N=('pct','size'),W=('pct',lambda s:(s>0).sum()),avg=('pct','mean'),worst=('pct','min'),
   mc_med=('mcap','median'),t=('u_today','sum'),f=('u_fin','sum'),p=('up_today','sum')).sort_values('bn')
P=["| batch | N | wins | avg % | worst % | median mcap $M | today 2× $ | final rule $ | today paper $ |","|---|---|---|---|---|---|---|---|---|"]
for b,r in pb.iterrows(): P.append(f"| {b} | {int(r.N)} | {int(r.W)} | {p2(r.avg)} | {p2(r.worst)} | {'–' if pd.isna(r.mc_med) else f'{r.mc_med/1e6:,.0f}'} | {d(r.t)} | {d(r.f)} | {d(r.p)} |")
P.append(f"| **TOTAL** | **74** | **{int(pb.W.sum())}** | **{p2(mm.pct.mean())}** | −1.50 | {mm.mcap.median()/1e6:,.0f} | **{d(pb.t.sum())}** | **{d(pb.f.sum())}** | **{d(pb.p.sum())}** |")
g=mm.groupby('pair').agg(n=('pct','size'),s=('pct','sum'),t=('u_today','sum'),p=('up_today','sum'),f=('u_fin','sum'),mc=('mcap','first'),V=('V','median')).sort_values('t')
C=["| pair | fills | Σ % | mcap $M | vol $M | today 2× $ | today paper $ | final rule $ |","|---|---|---|---|---|---|---|---|"]
for pr,r in g.head(8).iterrows(): C.append(f"| {pr} | {int(r.n)} | {p2(r.s)} | {'–' if pd.isna(r.mc) else f'{r.mc/1e6:,.0f}'} | {r.V/1e6:.1f} | {d(r.t)} | {d(r.p)} | {d(r.f)} |")
C.append(f"| **TOTAL (all 63 pairs)** | **74** | **{p2(g.s.sum())}** | | | **{d(g.t.sum())}** | **{d(g.p.sum())}** | **{d(g.f.sum())}** |")
top5=g.p.sort_values(ascending=False).head(5); 
open('sec8_tables.md','w').write("LEDGER\n"+ledger+"\n\nMASTER\n"+T['MASTER']+"\n\nBT\n"+T['BT']+"\n\nLIVE\n"+T['LIVE']+"\n\nPB\n"+"\n".join(P)+"\n\nCONC\n"+"\n".join(C)+f"\n\ntop5 paper share {top5.sum()/g.p.sum():.2f} {list(top5.index)}\n")
mm.to_csv('master_deep.csv',index=False)
