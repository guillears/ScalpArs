"""assemble master / backtest / live fade rows with rebuilt mcap, tick-tail overrides, impact model coefficients."""
import json, re, pandas as pd, numpy as np
R="/Users/guillearslanian/Downloads/NOFA AI"
B=json.load(open(R+'/reports/cache_vol_mcap/bapi_supply.json')); X=json.load(open('bapi_extra.json'))
def cs_mult(p):
    v=B.get(p) or {}
    if v.get('cs'): return v['cs'], v.get('mult',1) or 1
    v=X.get(p)
    if v:
        mult=1
        if v['symbol']!=p[:-4].lower():
            mm=re.match(r'^(1000000|100000|10000|1000|100)',p); mult=int(mm.group(1)) if mm else 1
        return v['cs'], mult
    return None, None
def mcap_rebuild(p, px):
    cs,mu=cs_mult(p); return cs*px/mu if cs else np.nan
m=pd.read_csv(R+'/reports/MASTER_POOL_stacked.csv',low_memory=False)
m=m[(m.entry_strategy=='SPIKE_FADE')&(m.stack_keep==True)].copy()
m['src']='MASTER'; m['pct']=m.stack_pct; m['usd_paper']=m.stack_pnl; m['batch']=m.era
m['is_stop']=m.close_reason.str.startswith('STOP_LOSS')
b=pd.read_csv(R+'/reports/ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv',low_memory=False)
b=b[b.entry_strategy=='SPIKE_FADE'].copy(); b['src']='BT'; b['batch']=b.regime; b['usd_paper']=np.nan
b['is_stop']=b.close_reason.str.startswith('STOP_LOSS')
tt=pd.read_csv('tail_ticks.csv')
b['pct_1m']=b.pct
for _,r in tt.iterrows():
    if pd.notna(r.get('tick_plus1s')):
        b.loc[(b.pair==r.pair)&(b.opened_at==r.opened),'pct']=r.tick_plus1s
l=pd.read_csv('/Users/guillearslanian/Downloads/scalpars_orders_live_2026-10-19_02-59-03.csv',low_memory=False)
l2=pd.read_csv(R+'/reports/BASELINE22_batch1014-1016_LIVE_orders.csv',low_memory=False)
l=pd.concat([l,l2]); l=l[(l.entry_strategy=='SPIKE_FADE')&(l.status=='CLOSED')].drop_duplicates(['opened_at','pair','direction']).copy()
l['src']='LIVE'; l['pct']=l.pnl_percentage; l['batch']='B23-live'; l['usd_paper']=np.nan; l['is_stop']=l.close_reason.str.startswith('STOP_LOSS')
cols=['src','batch','pair','opened_at','pct','is_stop','close_reason','entry_price','entry_pair_volume_24h_usd','entry_mcap_usd','notional_value','usd_paper','exit_slippage_pct','entry_bracket_cap_notional']
A=pd.concat([x[cols] for x in (m,b,l)],ignore_index=True)
A['V']=A.entry_pair_volume_24h_usd
A['mcap_stamp']=A.entry_mcap_usd
A['mcap']=A.mcap_stamp
miss=A.mcap.isna()
A.loc[miss,'mcap']=[mcap_rebuild(p,px) for p,px in zip(A.loc[miss,'pair'],A.loc[miss,'entry_price'])]
A['mcap_src']=np.where(A.mcap_stamp.notna(),'stamp',np.where(A.mcap.notna(),'rebuilt','unscored'))
A['day']=pd.to_datetime(A.opened_at,format='mixed').dt.floor('D')
A['overshoot']=np.where(A.is_stop, -1.5-A.pct, np.nan)
A.to_csv('fades_all.csv',index=False)
print(A.groupby(['src','mcap_src']).size())
print(A.groupby('src').pct.agg(['size','mean','sum','min']))
