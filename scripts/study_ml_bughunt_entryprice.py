#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 2a: entry-price sanity of every yr5 ML fill against raw aggTrades — the fill price must be a price
that traded around the open (MAKER: a bid ≤ the last trade, touched by a later trade; TAKER: ≈ the last trade). Also the price
reached in the 2 minutes BEFORE the open vs after (look-ahead / scale check). Out: reports/study_ml_bughunt_entryprice.csv"""
import os, time, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'reports', 'backtest_cache'); DAY = 86_400_000
d = pd.read_csv(os.path.join(ROOT, 'reports', 'ENGINE_REPLAY_YR5_ML_fills.csv'), low_memory=False)
d['open_ms'] = pd.to_datetime(d.opened_at).values.astype('datetime64[ms]').astype(np.int64)
rows = []
cache = {}
for r in d.itertuples():
    day = (r.open_ms // DAY) * DAY
    k = (r.pair, day)
    if k not in cache:
        cache.clear()
        ds = time.strftime('%Y-%m-%d', time.gmtime(day / 1000)); z = None
        for sub in ('ticks_q', 'ticks'):
            fp = os.path.join(CACHE, sub, r.pair, f'{ds}.npz')
            if os.path.exists(fp):
                z = np.load(fp); z = (z['t'].astype(np.int64), z['p'].astype(float)); break
        cache[k] = z
    z = cache[k]
    if z is None:
        rows.append(dict(key=r.Index, ok=False)); continue
    t, p = z
    i = np.searchsorted(t, r.open_ms)            # first tick at/after open
    last = p[i - 1] if i > 0 else np.nan
    a = np.searchsorted(t, r.open_ms - 30_000); b = np.searchsorted(t, r.open_ms + 30_000)
    pre = p[a:i]; post = p[i:b]
    rows.append(dict(key=r.Index, ok=True, last_before=last,
                     d_last=(r.entry_price / last - 1) * 100,
                     pre_min=(pre.min() / r.entry_price - 1) * 100 if len(pre) else np.nan,
                     post_min=(post.min() / r.entry_price - 1) * 100 if len(post) else np.nan,
                     post_max=(post.max() / r.entry_price - 1) * 100 if len(post) else np.nan,
                     gap_s=(t[i] - r.open_ms) / 1000 if i < len(t) else np.nan))
o = pd.DataFrame(rows).set_index('key')
m = d[['seed', 'pair', 'opened_at', 'entry_price', 'entry_order_type', 'pct', 'mon']].join(o)
m.to_csv(os.path.join(ROOT, 'reports', 'study_ml_bughunt_entryprice.csv'))
pd.set_option('display.width', 200)
print(m.groupby('entry_order_type')[['d_last', 'pre_min', 'post_min', 'post_max', 'gap_s']].describe(percentiles=[.01, .5, .99]).T.round(4))
print('rows without ticks', (~m.ok).sum())
print(m.groupby(['mon', 'entry_order_type']).d_last.mean().unstack().round(4))
