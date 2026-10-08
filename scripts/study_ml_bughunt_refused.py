#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 2d: REFUSED signals — does the gate the yr5 journal names really hold on the context it recorded?
Every BLOCK line (dir LONG, MOMENTUM-path pair gates with a pure rule) in each seed-1 chunk's own window; each rule re-written here from
services/indicators.get_signal with the frozen yr5 thresholds. Also the BTC 'veto_long' on SCAN lines vs the SCAN's own BTC readings
(BTC_ADX_GATE_LOW: adx < btc_adx_min_long). Out: printed table (per month: lines checked, rule true share)."""
import os, json, glob, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'reports', 'backtest_cache')
JDIR = os.path.join(CACHE, 'replay', 'code_yr5_181131e', 'reports', 'backtest_cache', 'replay', 'year', 'journal')
T = json.load(open(os.path.join(CACHE, 'replay/frozen_config_yr5_181131e.json')))['thresholds']


def rule(gate, c):
    p, e5, e8, e13, e20 = c.get('price'), c.get('ema5'), c.get('ema8'), c.get('ema13'), c.get('ema20')
    rsi, rp2, adx = c.get('rsi'), c.get('rsi_prev2'), c.get('adx')
    if gate == 'PAIR_EMA20_FILTER':
        return p <= e20
    if gate == 'PAIR_EMA20_SLOPE':
        return c.get('ema20_prev3') is None or e20 <= c['ema20_prev3']
    if gate == 'PAIR_RSI_MOMENTUM_LOADX':
        return rsi < rp2 and adx < T['long_rsi_momentum_adx_max']
    if gate == 'PAIR_ADX_MAX':
        return adx > T['momentum_adx_max_long']
    if gate == 'PAIR_EMA_GAP_MAX':
        return (e5 - e8) / e8 * 100 > T['ema_gap_5_8_max_long']
    if gate == 'PAIR_EMA_GAP_MIN':
        return (e5 - e8) / e8 * 100 < T['ema_gap_threshold_long']
    if gate == 'PAIR_RSI_RANGE':
        return rsi < T['momentum_long_rsi_min'] or rsi > T['momentum_long_rsi_max']
    if gate == 'PAIR_RANGE_POSITION_MAX':
        h, l = c.get('high_20'), c.get('low_20')
        return h != l and (p - l) / (h - l) * 100 > T.get('range_position_max_long', 100)
    if gate == 'EMA5_STRETCH':
        return abs(p - e5) / p * 100 > T['ema5_stretch_max_long']
    if gate == 'PAIR_EMA_GAP_5_20':
        g = (e5 - e20) / p * 100
        return g < T['ema_gap_5_20_min_long'] or g > T['ema_gap_5_20_max_long']
    return None


rows = []
for jd in sorted(glob.glob(os.path.join(JDIR, 'yr5_*_s1'))):
    chunk = os.path.basename(jd).split('_')[1]
    meta = json.load(open(os.path.join(CACHE, 'replay', f'yr5_{chunk}_s1_meta.json')))
    a, z = meta['start_ms'], meta['end_ms']
    for f in sorted(glob.glob(os.path.join(jd, 'decisions-*.jsonl'))):
        for line in open(f):
            if '"e":"BLOCK"' in line and '"dir":"LONG"' in line and '"ctx"' in line:
                j = json.loads(line)
                t = int(pd.Timestamp(j['t']).value // 1_000_000)
                if not (a <= t < z):
                    continue
                g = j['gate'].split('[')[0]
                r = rule(g, j['ctx'])
                rows.append((j['t'][:7], g, r))
            elif '"e":"SCAN"' in line:
                j = json.loads(line)
                t = int(pd.Timestamp(j['t']).value // 1_000_000)
                if not (a <= t < z):
                    continue
                if j.get('veto_long') == 'BTC_ADX_GATE_LOW' and j.get('btc_adx') is not None:
                    rows.append((j['t'][:7], 'SCAN:BTC_ADX_GATE_LOW', j['btc_adx'] < T['btc_adx_min_long']))
                elif j.get('btc_adx') is not None and j['btc_adx'] < T['btc_adx_min_long']:
                    rows.append((j['t'][:7], 'SCAN:adx<min but no LOW veto', j.get('veto_long')))
R = pd.DataFrame(rows, columns=['mon', 'gate', 'true'])
R['checked'] = R['true'].apply(lambda v: isinstance(v, (bool, np.bool_)))
C = R[R.checked]
pd.set_option('display.width', 250)
print(C.groupby(['gate']).agg(n=('true', 'size'), rule_true=('true', 'mean')).sort_values('n', ascending=False).to_string())
print(C.groupby('mon').agg(n=('true', 'size'), rule_true=('true', 'mean')).to_string())
print('unchecked gate names:', R[~R.checked].gate.value_counts().head(25).to_dict())
x = R[R.gate == 'SCAN:adx<min but no LOW veto']
print('SCAN adx<min without LOW veto:', len(x), x['true'].value_counts().head().to_dict())
