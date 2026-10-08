#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 4: tick-archive integrity for every pair that carries a yr5 ML fill, every day Jan-01 → Oct-04:
file present, timestamps in ms and inside the file's own UTC day, monotone, exact-duplicate rows, price outliers (|tick/median−1|
> 20 %), and per-month coverage. Out: reports/study_ml_bughunt_tickaudit.csv (pair-day rows)"""
import os, time, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'reports', 'backtest_cache'); DAY = 86_400_000


def one(args):
    pair, day = args
    ds = time.strftime('%Y-%m-%d', time.gmtime(day / 1000))
    for sub in ('ticks_q', 'ticks'):
        fp = os.path.join(CACHE, sub, pair, f'{ds}.npz')
        if os.path.exists(fp):
            try:
                z = np.load(fp); t = z['t'].astype(np.int64); p = z['p'].astype(np.float64)
                q = z['q'] if 'q' in z.files else None
            except Exception as ex:
                return dict(pair=pair, day=ds, src=sub, err=str(ex)[:80])
            n = len(t)
            if n == 0:
                return dict(pair=pair, day=ds, src=sub, n=0)
            med = float(np.median(p))
            dup = int(((np.diff(t) == 0) & (np.diff(p) == 0) & ((np.diff(q) == 0) if q is not None else True)).sum())
            return dict(pair=pair, day=ds, src=sub, n=n, has_q=q is not None, t0_off_s=(t[0] - day) / 1000, t1_off_s=(t[-1] - day) / 1000,
                        in_day=bool(t[0] >= day and t[-1] < day + DAY), nonmono=int((np.diff(t) < 0).sum()), dup_rows=dup,
                        outliers=int((np.abs(p / med - 1) > 0.2).sum()), max_gap_s=float(np.diff(t).max() / 1000) if n > 1 else None)
    return dict(pair=pair, day=ds, src=None, n=None)


if __name__ == '__main__':
    f = pd.read_csv(os.path.join(ROOT, 'reports', 'ENGINE_REPLAY_YR5_ML_fills.csv'), usecols=['pair', 'opened_at', 'closed_at'])
    f['d0'] = pd.to_datetime(f.opened_at).dt.floor('D'); f['d1'] = pd.to_datetime(f.closed_at).dt.floor('D')
    jobs = set()
    for r in f.itertuples():               # every pair-day an ML fill was open on, plus the day before (forming-candle reads)
        for d in pd.date_range(r.d0 - pd.Timedelta(days=1), r.d1):
            jobs.add((r.pair, int(d.value // 1_000_000)))
    jobs = sorted(jobs)
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs, chunksize=20))
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(ROOT, 'reports', 'study_ml_bughunt_tickaudit.csv'), index=False)
    o['mon'] = o.day.str[:7]
    print(len(o), 'pair-days')
    print(o.groupby('mon').agg(n=('pair', 'size'), missing=('src', lambda s: s.isna().sum()), ticks_only=('src', lambda s: (s == 'ticks').sum()),
                              not_in_day=('in_day', lambda s: (s == False).sum()), nonmono=('nonmono', 'sum'), dup=('dup_rows', 'sum'),
                              outl=('outliers', 'sum'), med_n=('n', 'median'), max_gap=('max_gap_s', 'max')))
    print(o[(o.outliers > 0) | (o.nonmono > 0) | (o.in_day == False)].head(20))
