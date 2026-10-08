#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 2c: do the yr5 ML fills satisfy today's ML gates ON THEIR OWN ENTRY STAMPS? Thresholds read from the
frozen yr5 config; each rule written here from services/indicators.get_signal / trading_engine (not imported). A fill that violates a
gate on its own stamp = an inverted / skipped gate or a stamp/decision mismatch. Stamps are written at the open (≈ 6–25 s after the
decision), so knife-edge violations within a small tolerance are expected; gross or month-concentrated ones are not.
Out: reports/study_ml_bughunt_gatestamps.csv"""
import os, json, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
T = json.load(open(os.path.join(ROOT, 'reports/backtest_cache/replay/frozen_config_yr5_181131e.json')))['thresholds']
d = pd.read_csv(os.path.join(ROOT, 'reports/ENGINE_REPLAY_YR5_ML_fills.csv'), low_memory=False).copy()
door = d.cell_multiplier_source.fillna('')
g = {}
g['GAP_5_20'] = ~d.entry_gap.between(T['ema_gap_5_20_min_long'], T['ema_gap_5_20_max_long'])
g['EMA5_STRETCH'] = d.entry_ema5_stretch.abs() > T['ema5_stretch_max_long']
g['GAP_5_8_MAX'] = d.entry_ema_gap_5_8 > T['ema_gap_5_8_max_long']
g['GAP_5_8_MIN(non-door)'] = (d.entry_ema_gap_5_8 < T['ema_gap_threshold_long']) & (door == 'UNMATCHED')
g['RSI_RANGE'] = ~d.entry_rsi.between(T['momentum_long_rsi_min'], T['momentum_long_rsi_max'])
g['ADX_MAX'] = d.entry_adx > T['momentum_adx_max_long']
g['ADX_MIN_STRONG'] = d.entry_adx <= T['adx_strong_long']
g['LOADX'] = (d.entry_rsi < d.entry_rsi_prev) & (d.entry_adx < T['long_rsi_momentum_adx_max'])
g['BTC_RSI_RANGE'] = ~d.entry_btc_rsi.between(T['btc_rsi_min_long'], T['btc_rsi_max_long'])
g['BTC_ADX_RANGE'] = ~d.entry_btc_adx.between(T['btc_adx_min_long'], T['btc_adx_max_long'])
g['BTC_SLOPE_MAX'] = d.entry_btc_ema20_slope > T['btc_ema20_slope_max_long']
g['PAIR_ATR'] = ~d.entry_atr_pct.between(T['pair_atr_min_long'], T['pair_atr_max_long'])
g['MEGACAP'] = d.entry_pair_rank <= T['long_megacap_rank_max']
g['HEAT(bull>=85, not washed)'] = (d.entry_bull_pct >= T['long_heat_bull_pct_min']) & ~(d.entry_btc_off30d_high_pct <= T['long_heat_exempt_off30d_max'])
g['GLOBAL_VOL<rescue'] = d.entry_global_volume_ratio < T['global_volume_rescue_max_long']
g['PAIR_RANK>50'] = d.entry_pair_rank > 50
g['PAIR_EMA20_SLOPE<=0'] = d.entry_ema20_slope <= 0
g['BTC_1H_DEADBAND'] = d.entry_btc_1h_slope.abs() < T['long_btc_1h_deadband']
G = pd.DataFrame(g)
G['mon'] = d.mon; G['pct'] = d.pct; G['door'] = door
out = G.groupby('mon')[list(g)].sum().T
out['ALL'] = out.sum(axis=1); out['rate'] = (out.ALL / len(d)).round(4)
pd.set_option('display.width', 250)
print(out.to_string())
pd.concat([d[['seed', 'pair', 'opened_at', 'mon', 'pct', 'cell_multiplier_source', 'entry_gap', 'entry_ema5_stretch', 'entry_ema_gap_5_8', 'entry_rsi',
              'entry_rsi_prev', 'entry_adx', 'entry_btc_rsi', 'entry_btc_adx', 'entry_btc_ema20_slope', 'entry_atr_pct', 'entry_pair_rank',
              'entry_bull_pct', 'entry_btc_off30d_high_pct', 'entry_global_volume_ratio', 'entry_btc_1h_slope']], G[list(g)]], axis=1).to_csv(
    os.path.join(ROOT, 'reports/study_ml_bughunt_gatestamps.csv'), index=False)
for k in g:
    x = G[G[k]]
    if len(x):
        print(k, len(x), 'avg pct', round(x.pct.mean(), 3), 'doors', x.door.value_counts().to_dict())
