# STATE_EXIT_AUDIT_2026-11-21 — audit helper (cohort). Research only, read-only on study data. Run from the repo root: venv/bin/python scripts/STATE_EXIT_AUDIT_2026-11-21_cohort.py
import glob,os,sys,pandas as pd,numpy as np
sys.path.insert(0,'scripts'); import scout_gatekit as GK
fs=sorted(glob.glob('exports_archive/scalpars_orders_*.csv'))+sorted(glob.glob('reports/.live_exports/**/scalpars_orders_*.csv',recursive=True))
SLV=("FLATBASE_SHORT","FRENZY_WILLY_HT","EXHAUSTION_SHORT","FRENZY_WILLY","FRENZY_LONG")
parts=[];allrows=[]
for f in fs:
    h=pd.read_csv(f,nrows=0).columns
    if 'entry_strategy' not in h: continue
    x=pd.read_csv(f,low_memory=False)
    x['src']=f
    allrows.append(x[x.entry_strategy.astype(str).str.contains('HT|FLATBASE|EXHAUST',regex=True)])
X=pd.concat(allrows,ignore_index=True)
print('strategies',X.entry_strategy.value_counts().to_dict())
print('status',X.groupby('entry_strategy').status.value_counts().to_dict())
X=X[X.status=='CLOSED']
if 'is_probe' in X: X=X[~X.is_probe.fillna(False).astype(str).str.lower().isin(['true','1'])]
X=X[~X.entry_strategy.astype(str).str.endswith('_PROBE')]
X['k']=X.opened_at.astype(str).str[:19]
X=X.sort_values('src').drop_duplicates(['k','pair','direction'],keep='last')
X,oc=GK.split_operator_closed(X)
print(GK.operator_closed_line(oc))
R=pd.read_csv('reports/STATE_EXIT_2026-11-21_data/real_cohort.csv')
for s,g in X.groupby('entry_strategy'):
    r=R[R.entry_strategy==s]
    print(s,'mine',len(g),'study',len(r),'first',g.k.min(), 'master(<11-17T17)',(g.k<'2026-11-17T17:00').sum(),'study master',(r.k<'2026-11-17T17:00').sum())
    a=set(zip(g.k,g.pair)); b=set(zip(r.k,r.pair))
    print('  only mine',sorted(a-b)[:20]); print('  only study',sorted(b-a)[:20])
X.to_csv('reports/STATE_EXIT_AUDIT_2026-11-21_mine.csv',index=False)
