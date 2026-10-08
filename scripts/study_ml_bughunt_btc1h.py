#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 2e: BTC 1h (RSI(12) 1h, EMA20-1h slope over 3 bars) and BTC 5m RSI stamps of every seed-1 yr5 ML
fill vs an independent rebuild at the fill's own scan instant (the last journal SCAN line before the open): 1h bars = CLOSED 1h
bars aggregated here from BTC 5m klines + the forming hour (its closed 5m bars + the forming 5m from raw BTC aggTrades up to T), `ta`
indicators on 100 bars. Per month: share within tolerance and mean signed error. Out: reports/study_ml_bughunt_btc1h.csv"""
import os, glob, json, numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(__file__))
import study_ml_bughunt_inputs as S
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
ROOT = S.ROOT; H = 3_600_000
d = pd.read_csv(os.path.join(ROOT, 'reports/ENGINE_REPLAY_YR5_ML_fills.csv'), low_memory=False)
d = d[d.seed == 1].copy()
d['open_ms'] = pd.to_datetime(d.opened_at).values.astype('datetime64[ms]').astype(np.int64)
scans = []
for jd in sorted(glob.glob(os.path.join(S.JDIR, 'yr5_*_s1'))):
    for f in glob.glob(os.path.join(jd, 'decisions-*.jsonl')):
        for line in open(f):
            if '"e":"SCAN"' in line:
                scans.append(int(pd.Timestamp(json.loads(line)['t']).value // 1_000_000))
scans = np.unique(np.array(scans, dtype=np.int64))
b5 = S.k5('BTCUSDT')
rows = []
for r in d.itertuples():
    i = np.searchsorted(scans, r.open_ms) - 1
    T = int(scans[i]) - 200          # the SCAN line is written just after the BTC fetch
    rows5, _ = S.ohlcv_at('BTCUSDT', T, limit=12 * 101)
    df = pd.DataFrame(rows5, columns=['t', 'o', 'h', 'l', 'c', 'v']).astype(float)
    df['hb'] = (df.t // H) * H
    hb = df.groupby('hb').agg(c=('c', 'last')).reset_index().tail(100)
    c = hb.c.reset_index(drop=True)
    e20 = EMAIndicator(close=c, window=20).ema_indicator(); rsi = RSIIndicator(close=c, window=12).rsi()
    slope = round((e20.iloc[-1] - e20.iloc[-4]) / e20.iloc[-4] * 100, 4)
    r5 = RSIIndicator(close=df.c.tail(100).reset_index(drop=True), window=12).rsi().iloc[-1]
    rows.append(dict(mon=r.mon, pair=r.pair, opened_at=r.opened_at, lag_s=(r.open_ms - T) / 1000, eng_rsi1h=r.entry_btc_rsi_1h, my_rsi1h=round(rsi.iloc[-1], 1),
                     eng_slope1h=r.entry_btc_1h_slope, my_slope1h=slope, eng_rsi5=r.entry_btc_rsi, my_rsi5=r5))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(ROOT, 'reports/study_ml_bughunt_btc1h.csv'), index=False)
R['d1'] = R.my_rsi1h - R.eng_rsi1h; R['ds'] = R.my_slope1h - R.eng_slope1h; R['d5'] = R.my_rsi5 - R.eng_rsi5
print(R.groupby('mon').agg(n=('d1', 'size'), rsi1h_within0p15=('d1', lambda s: (s.abs() <= 0.15).mean()), rsi1h_mean=('d1', 'mean'),
                           slope_within0p002=('ds', lambda s: (s.abs() <= 0.002).mean()), slope_mean=('ds', 'mean'),
                           rsi5_within0p5=('d5', lambda s: (s.abs() <= 0.5).mean()), rsi5_mean=('d5', 'mean'), lag_med=('lag_s', 'median')).round(4).to_string())
