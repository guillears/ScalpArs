#!/usr/bin/env python3
"""Phase-1 backtest — validation gate: replay candidates vs real Jun-Jul fills."""
import os, sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "reports", "backtest_cache")

cand_file = sys.argv[1] if len(sys.argv) > 1 else 'phase1_candidates.csv'
cand = pd.read_csv(os.path.join(CACHE, cand_file), parse_dates=['ts'])
mas = pd.read_csv(os.path.join(ROOT, 'reports', 'MASTER_POOL_stacked.csv'), low_memory=False)
mas['opened_at'] = pd.to_datetime(mas['opened_at'], format='mixed')
ml = mas[(mas['direction'] == 'LONG') & (~mas['is_probe'].fillna(False).astype(bool))
         & (mas['entry_strategy'].fillna('MOMENTUM').isin(['MOMENTUM', '']))]
lo, hi = pd.Timestamp('2026-06-18'), pd.Timestamp('2026-08-01')
real_kept = ml[(ml['opened_at'] >= lo) & (ml['opened_at'] < hi) & (ml['stack_keep'] == True)]
cwin = cand[(cand['ts'] >= lo) & (cand['ts'] < hi)]
print(f"window Jun-18→Jul-31: replay={len(cwin)} real stack-kept={len(real_kept)}")
hits, misses = 0, []
for _, r in real_kept.iterrows():
    m = cwin[(cwin['pair'] == r['pair'])
             & (abs((cwin['ts'] - r['opened_at']).dt.total_seconds()) <= 45 * 60)]
    if len(m):
        hits += 1
    else:
        misses.append(f"{str(r['opened_at'])[:16]} {r['pair']}")
print(f"RECALL: {hits}/{len(real_kept)} = {hits / max(len(real_kept), 1) * 100:.0f}%")
for m in misses:
    print(f"  miss: {m}")
