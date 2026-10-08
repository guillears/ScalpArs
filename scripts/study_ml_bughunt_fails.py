#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 2f: REFUSED momentum-long candidates (journal FAILS lines, src MOMENTUM, dir LONG, seed 1, each chunk's
own window, reservoir-sampled per month) — rebuild the pair's 5m view from raw data (k5m_full closed bars + forming bar from aggTrades)
at the line's second minus 5 s (the median fetch lag measured in part 2b) and re-test every PAIR gate the engine listed, with rules
written here from services/indicators.get_signal (frozen yr5 thresholds). Reports, per month, the share of listed pair gates the
independent rebuild also finds failing (knife-edge disagreements from the unknown exact fetch second are expected at a few %).
Out: reports/study_ml_bughunt_fails.csv"""
import os, glob, json, random, numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(__file__))
import study_ml_bughunt_inputs as S

T = json.load(open(os.path.join(S.CACHE, 'replay/frozen_config_yr5_181131e.json')))['thresholds']


def rule(g, c):
    p, e5, e8, e13, e20 = c['price'], c['ema5'], c['ema8'], c['ema13'], c['ema20']
    if g == 'PAIR_EMA20_FILTER':
        return p <= e20
    if g == 'PAIR_EMA20_SLOPE':
        return e20 <= c['ema20_prev3']
    if g == 'PAIR_RSI_MOMENTUM_LOADX':
        return c['rsi'] < c['rsi_prev2'] and c['adx'] < T['long_rsi_momentum_adx_max']
    if g == 'PAIR_ADX_MAX':
        return c['adx'] > T['momentum_adx_max_long']
    if g == 'PAIR_EMA_GAP_MAX':
        return (e5 - e8) / e8 * 100 > T['ema_gap_5_8_max_long']
    if g == 'PAIR_EMA_GAP_MIN':
        return (e5 - e8) / e8 * 100 < T['ema_gap_threshold_long']
    if g.startswith('PAIR_RSI_RANGE'):
        return c['rsi'] < T['momentum_long_rsi_min'] or c['rsi'] > T['momentum_long_rsi_max']
    return None


def main(per_month=40):
    rng = random.Random(7)
    pool, seen = {}, {}
    for jd in sorted(glob.glob(os.path.join(S.JDIR, 'yr5_*_s1'))):
        chunk = os.path.basename(jd).split('_')[1]
        meta = json.load(open(os.path.join(S.CACHE, 'replay', f'yr5_{chunk}_s1_meta.json')))
        a, z = meta['start_ms'], meta['end_ms']
        for f in sorted(glob.glob(os.path.join(jd, 'decisions-*.jsonl'))):
            for line in open(f):
                if '"e":"FAILS"' not in line or '"src":"MOMENTUM"' not in line or '"dir":"LONG"' not in line:
                    continue
                j = json.loads(line)
                gates = [x.split('[')[0] for x in j['gates'].split('+') if x.startswith('PAIR_') and not x.startswith('PAIR_NO_TRADE')]
                gates = [g for g in gates if g in ('PAIR_EMA20_FILTER', 'PAIR_EMA20_SLOPE', 'PAIR_RSI_MOMENTUM_LOADX', 'PAIR_ADX_MAX',
                                                   'PAIR_EMA_GAP_MAX', 'PAIR_EMA_GAP_MIN', 'PAIR_RSI_RANGE')]
                if not gates or j['pair'] in ('BTCUSDT',):
                    continue
                t = int(pd.Timestamp(j['t']).value // 1_000_000)
                if not (a <= t < z):
                    continue
                m = j['t'][:7]; seen[m] = seen.get(m, 0) + 1
                lst = pool.setdefault(m, [])
                item = (j['t'], j['pair'], gates)
                if len(lst) < per_month:
                    lst.append(item)
                else:
                    k = rng.randrange(seen[m])
                    if k < per_month:
                        lst[k] = item
    rows = []
    for m, lst in sorted(pool.items()):
        for tstr, pair, gates in lst:
            t = int(pd.Timestamp(tstr).value // 1_000_000) - 5000
            ohl, src = S.ohlcv_at(pair, t)
            if not ohl or len(ohl) < 60:
                continue
            c = S.inds(ohl)
            for g in gates:
                rows.append(dict(mon=m, t=tstr, pair=pair, gate=g, src=src, confirmed=bool(rule(g, c))))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(S.ROOT, 'reports/study_ml_bughunt_fails.csv'), index=False)
    pd.set_option('display.width', 200)
    print(D.groupby('mon').agg(n=('confirmed', 'size'), confirmed=('confirmed', 'mean')).round(3).to_string())
    print(D.groupby('gate').agg(n=('confirmed', 'size'), confirmed=('confirmed', 'mean')).round(3).to_string())


if __name__ == '__main__':
    main()
