#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 6b: re-walk every master momentum-long fill (non-probe; kept and removed) with the SAME independent
today's-exit walker used on yr5 (part 1) from live's own entry second/price/fee type — "live entries under today's exits". Tells how
much of the batch gap on shared trades is the exit-rule era (ladder / recovery hold / runner arm 0.40) rather than the replay.
Out: reports/study_ml_bughunt_master_rewalk.csv"""
import os, numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(__file__))
import study_ml_bughunt_exitwalk as X
m = pd.read_csv(os.path.join(X.REP, 'MASTER_POOL_stacked.csv'), low_memory=False)
ml = m[(m.entry_strategy == 'MOMENTUM') & (m.direction == 'LONG') & (~m.is_probe.astype(bool))].copy()
ml['open_ms'] = pd.to_datetime(ml.opened_at).values.astype('datetime64[ms]').astype(np.int64)
ml['key'] = ml.era + '|' + ml.pair + '|' + ml.opened_at.astype(str)
ml['entry_order_type'] = np.where(ml.entry_order_type.astype(str).str.startswith('MAKER'), 'MAKER', 'TAKER')
res = X.work(ml[['key', 'pair', 'entry_price', 'open_ms', 'entry_order_type', 'entry_atr_pct']].to_dict('records'))
o = pd.DataFrame(res)
out = ml[['key', 'era', 'pair', 'opened_at', 'close_reason', 'pnl_percentage', 'stack_keep', 'stack_pct']].merge(o, on='key')
out.to_csv(os.path.join(X.REP, 'study_ml_bughunt_master_rewalk.csv'), index=False)
k = out[out.stack_keep.astype(bool) & (out.era != 'B1') & (out.my_reason != 'NO_TICKS')]
print('kept ex-B1 with ticks', len(k), '/', int((out.stack_keep.astype(bool) & (out.era != 'B1')).sum()))
print('master stack_pct', round(k.stack_pct.mean(), 4), '| as traded', round(k.pnl_percentage.mean(), 4), '| TODAY exits (independent walk)', round(k.my_pct.mean(), 4))
print(k.groupby('era').agg(n=('my_pct', 'size'), stack=('stack_pct', 'mean'), today=('my_pct', 'mean')).round(3).T.to_string())
