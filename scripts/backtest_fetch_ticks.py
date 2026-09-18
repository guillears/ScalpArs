#!/usr/bin/env python3
"""Tick data (Binance USDT-M aggTrades daily archives, no rate limit) for the pair-days listed in
reports/backtest_cache/tick_pairdays.csv → reports/backtest_cache/ticks/<PAIR>/<YYYY-MM-DD>.npz
(t: int64 ms, p: float32). Parallel, resumable."""
import os, sys, io, zipfile, csv, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')); CACHE=os.path.join(ROOT,'reports','backtest_cache'); OUT=os.path.join(CACHE,'ticks')
rows=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else os.path.join(CACHE,'tick_pairdays.csv'))
def one(pair,date):
    fp=os.path.join(OUT,pair,f'{date}.npz')
    if os.path.exists(fp): return 'skip'
    url=f'https://data.binance.vision/data/futures/um/daily/aggTrades/{pair}/{pair}-aggTrades-{date}.zip'
    for i in range(4):
        try:
            with urllib.request.urlopen(url,timeout=120) as r: data=r.read()
            break
        except Exception as ex:
            if i==3: return f'FAIL {pair} {date}: {ex}'
    z=zipfile.ZipFile(io.BytesIO(data)); name=z.namelist()[0]
    t=[]; p=[]
    with z.open(name) as f:
        rd=csv.reader(io.TextIOWrapper(f,encoding='utf-8'))
        for row in rd:
            if not row or not row[0].isdigit(): continue     # header line in newer files
            t.append(int(row[5])); p.append(float(row[1]))
    os.makedirs(os.path.dirname(fp),exist_ok=True)
    np.savez_compressed(fp,t=np.array(t,dtype=np.int64),p=np.array(p,dtype=np.float32))
    return f'{pair} {date} {len(t)}'
jobs=[(r.pair,r.date) for r in rows.itertuples()]
done=0
with cf.ThreadPoolExecutor(8) as ex:
    for res in ex.map(lambda j: one(*j), jobs):
        done+=1
        if res!='skip' and (done%25==0 or res.startswith('FAIL')): print(f'{done}/{len(jobs)} {res}',flush=True)
print('done',flush=True)
