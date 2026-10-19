import numpy as np, pandas as pd
exec(open('deep.py').read().split("if __name__")[0])
pd.set_option('display.width',300); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',30)
R="/Users/guillearslanian/Downloads/NOFA AI"
m=pd.read_csv(R+'/reports/MASTER_POOL_stacked.csv',low_memory=False); m=m[(m.entry_strategy=='SPIKE_FADE')&(m.stack_keep==True)]
A2=A.copy()
for rule,tag in (('(0) today 2x, liq 0.5%','usd_today'),('(a) REC 1x, liq 0.05%','usd_rec'),('(b) 1x if mcap>=200M else 2x (unsc 1x)','usd_b'),('(c) 1x flat, liq 0.5%','usd_c')):
    N=N_of(rule,E_LIVE); A2[tag]=eff_pct(N,1.0)*N/100; A2[tag+'_paper']=eff_pct(N,0.0)*N/100
    if tag=='usd_today': A2['N_today']=N
    if tag=='usd_rec': A2['N_rec']=N
mm=A2[A2.src=='MASTER'].copy()
mm=mm.merge(m[['opened_at','pair','peak_pnl','trough_pnl','stack_block_reason','pnl_percentage']],on=['opened_at','pair'],how='left')
mm['bn']=mm.batch.str[1:].astype(int); mm=mm.sort_values(['bn','opened_at'])
cols=['batch','opened_at','pair','mcap','V','peak_pnl','trough_pnl','close_reason','pnl_percentage','pct','N_today','usd_today','N_rec','usd_rec','usd_today_paper']
x=mm[cols].copy(); x['mcap']=(x.mcap/1e6).round(0); x['V']=(x.V/1e6).round(1); x['opened_at']=x.opened_at.str[5:16]
print(x.round(2).to_string(index=False))
mm.to_csv('master_deep.csv',index=False)
# buckets
def bt(df,lab):
    st=df[df.is_stop]
    return dict(bucket=lab,N=len(df),WR=round((df.pct>0).mean()*100) if len(df) else np.nan,avg=df.pct.mean(),worst=df.pct.min(),
                stops=len(st),gap=st.overshoot.mean(),tail2=(df.pct<=-2).sum(),
                usd_today=df.usd_today.sum(),usd_rec=df.usd_rec.sum(),usd_b=df.usd_b.sum(),usd_c=df.usd_c.sum(),usd_today_paper=df.usd_today_paper.sum(),
                pairs=df.pair.nunique())
for src in ('MASTER','BT','LIVE'):
    S=A2[A2.src==src]; rows=[]
    for lab,lo,hi in (('<$50M',0,50e6),('$50-200M',50e6,200e6),('$200-300M',200e6,300e6),('$300-500M',300e6,500e6),('>=$500M',500e6,np.inf)):
        rows.append(bt(S[(S.mcap>=lo)&(S.mcap<hi)],lab))
    for lab,lo in (('>=$200M',200e6),('>=$300M',300e6),('>=$500M',500e6)): rows.append(bt(S[S.mcap>=lo],'cum '+lab))
    rows.append(bt(S[S.mcap.isna()],'unscored')); rows.append(bt(S,'TOTAL'))
    print('\n##',src); print(pd.DataFrame(rows).round(3).to_string(index=False))
    rows=[]
    for lab,lo,hi in (('<$3M',0,3e6),('$3-6M',3e6,6e6),('$6-12M',6e6,12e6),('>=$12M',12e6,np.inf)): rows.append(bt(S[(S.V>=lo)&(S.V<hi)],lab))
    rows.append(bt(S,'TOTAL')); print(pd.DataFrame(rows).round(3).to_string(index=False))
# per pair concentration master
g=mm.groupby('pair').agg(n=('pct','size'),sum_pct=('pct','sum'),usd_today=('usd_today','sum'),usd_rec=('usd_rec','sum'),paper=('usd_today_paper','sum')).sort_values('usd_today')
print('\nmaster per pair (worst 10 by today $ w/ impact)'); print(g.head(10).round(1)); print('top 5 paper $ share', (g.paper.sort_values(ascending=False).head(5).sum()/g.paper.sum()).round(2))
L=mm[mm.pct<0]; print('master losers', len(L), 'pairs', L.pair.tolist(), 'batches', L.batch.tolist(), 'mcap', (L.mcap/1e6).round(0).tolist(), 'V', (L.V/1e6).round(1).tolist())
# per batch
pb=mm.groupby('batch').agg(bn=('bn','first'),N=('pct','size'),WR=('pct',lambda s:(s>0).mean()*100),avg=('pct','mean'),worst=('pct','min'),usd_today=('usd_today','sum'),usd_rec=('usd_rec','sum'),usd_b=('usd_b','sum'),usd_c=('usd_c','sum'),paper=('usd_today_paper','sum')).sort_values('bn')
print(pb.round(2).to_string())
# BT high-mcap worst
b=A2[(A2.src=='BT')&(A2.mcap>=200e6)].sort_values('pct').head(6); print(b[['pair','opened_at','mcap','V','pct','overshoot']])
