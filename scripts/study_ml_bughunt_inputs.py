#!/usr/bin/env python3
"""ML BACKTEST BUG HUNT (2026-10-08) part 2b: are the replay's decision INPUTS right in Jan–May (no live data there)?

At the EXACT decision second the yr5 journal recorded (BLOCK lines carry the pair indicator context the engine used; SCAN lines the
BTC readings), rebuild the 5m OHLCV a live bot would have fetched (ohlcv 5m limit 100 = 99 CLOSED bars from the k5m_full exchange
klines + the FORMING bar [bar open, T) built here from the raw aggTrade archive) and recompute with the `ta` library (the library
live uses) — no harness, no engine code. Compare per month: price, EMA5/8/13/20/50, RSI(12) and its prev1/prev2, ADX(14) and prev,
+DI/−DI, ATR, 20-bar high/low (pair, BLOCK ctx) and BTC RSI / ADX / ADX prev / EMA20-slope (SCAN). Jan–May vs Jun–Sep: any
systematic difference only in the months without live data is a candidate bug.

Usage: venv/bin/python scripts/study_ml_bughunt_inputs.py [--per-month 60] [--seed 1]
Out:   reports/study_ml_bughunt_inputs_pair.csv, reports/study_ml_bughunt_inputs_btc.csv
"""
import os, json, glob, random, argparse, time
import numpy as np, pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'reports', 'backtest_cache')
JDIR = os.path.join(CACHE, 'replay', 'code_yr5_181131e', 'reports', 'backtest_cache', 'replay', 'year', 'journal')
DAY, BAR = 86_400_000, 300_000

_K = {}


def k5(pair):
    if pair not in _K:
        f = os.path.join(CACHE, 'k5m_full', f'{pair}.csv')
        if pair == 'BTCUSDT':
            fr = [pd.read_csv(os.path.join(CACHE, 'btc_5m.csv'))]
            if os.path.exists(f):
                fr.append(pd.read_csv(f))
            d = pd.concat(fr)
        else:
            d = pd.read_csv(f) if os.path.exists(f) else None
        if d is not None:
            d = d.drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        if len(_K) > 20:
            _K.clear()
        _K[pair] = d
    return _K[pair]


_T = {}


def tick_day(pair, day):
    k = (pair, day)
    if k not in _T:
        ds = time.strftime('%Y-%m-%d', time.gmtime(day / 1000)); out = None
        for sub in ('ticks_q', 'ticks'):
            fp = os.path.join(CACHE, sub, pair, f'{ds}.npz')
            if os.path.exists(fp):
                z = np.load(fp)
                out = (z['t'].astype(np.int64), z['p'].astype(float), z['q'].astype(float) if 'q' in z.files else None)
                break
        if len(_T) > 6:
            _T.pop(next(iter(_T)))
        _T[k] = out
    return _T[k]


def ohlcv_at(pair, T, limit=100):
    d = k5(pair)
    if d is None:
        return None, 'no_k5'
    cur = (T // BAR) * BAR
    closed = d[d.open_time + BAR <= T].tail(limit - 1)
    rows = closed[['open_time', 'o', 'h', 'l', 'c', 'vol']].values.tolist()
    z = tick_day(pair, (cur // DAY) * DAY)
    src = 'ticks'
    if z is not None and z[2] is not None:
        t, p, q = z
        a, b = np.searchsorted(t, cur), np.searchsorted(t, T)
        if b > a:
            rows.append([cur, p[a], p[a:b].max(), p[a:b].min(), p[b - 1], q[a:b].sum()])
        else:
            src = 'ticks_empty'
    else:
        src = 'no_ticks'
    return rows, src


def inds(rows):
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).astype(float)
    c, h, l = df.close, df.high, df.low
    e = {n: EMAIndicator(close=c, window=n).ema_indicator() for n in (5, 8, 13, 20, 50)}
    rsi = RSIIndicator(close=c, window=12).rsi()
    rsi14 = RSIIndicator(close=c, window=12).rsi()
    a = ADXIndicator(high=h, low=l, close=c, window=14)
    adx, pdi, ndi = a.adx(), a.adx_pos(), a.adx_neg()
    atr = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
    return dict(price=c.iloc[-1], ema5=e[5].iloc[-1], ema8=e[8].iloc[-1], ema13=e[13].iloc[-1], ema20=e[20].iloc[-1], ema50=e[50].iloc[-1],
                ema20_prev3=e[20].iloc[-4], rsi=rsi.iloc[-1], rsi_prev1=rsi.iloc[-2], rsi_prev2=rsi.iloc[-3], adx=adx.iloc[-1],
                adx_prev1=adx.iloc[-2], pos_di=pdi.iloc[-1], neg_di=ndi.iloc[-1], atr=atr.iloc[-1],
                high_20=h.iloc[-20:].max(), low_20=l.iloc[-20:].min())


def sample_lines(seed, per_month, rng):
    """reservoir-sample BLOCK (dir LONG, with ctx) and SCAN lines per calendar month from the seed's journals (each chunk's own
    window only — warm-up days belong to the previous chunk)."""
    blocks, scans = {}, {}
    seen = {}
    for jd in sorted(glob.glob(os.path.join(JDIR, f'yr5_*_s{seed}'))):
        chunk = os.path.basename(jd).split('_')[1]
        meta = json.load(open(os.path.join(CACHE, 'replay', f'yr5_{chunk}_s{seed}_meta.json')))
        a, z = meta['start_ms'], meta['end_ms']
        for f in sorted(glob.glob(os.path.join(jd, 'decisions-*.jsonl'))):
            day = pd.Timestamp(os.path.basename(f)[10:20]).value // 1_000_000
            if day + DAY <= a or day >= z:
                continue
            for line in open(f):
                if '"e":"BLOCK"' in line and '"dir":"LONG"' in line and '"ctx"' in line:
                    kind = 'B'
                elif '"e":"SCAN"' in line:
                    kind = 'S'
                else:
                    continue
                j = json.loads(line)
                T = int(pd.Timestamp(j['t']).value // 1_000_000)
                if not (a <= T < z):
                    continue
                m = j['t'][:7]
                key = (kind, m); seen[key] = seen.get(key, 0) + 1
                tgt = blocks if kind == 'B' else scans
                lst = tgt.setdefault(m, [])
                if len(lst) < per_month:
                    lst.append(j)
                else:
                    r = rng.randrange(seen[key])
                    if r < per_month:
                        lst[r] = j
    return blocks, scans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-month', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1)
    A = ap.parse_args()
    rng = random.Random(20261008)
    blocks, scans = sample_lines(A.seed, A.per_month, rng)
    keys = ['price', 'ema5', 'ema8', 'ema13', 'ema20', 'ema50', 'rsi', 'rsi_prev1', 'rsi_prev2', 'adx', 'adx_prev1', 'pos_di', 'neg_di',
            'atr', 'high_20', 'low_20']
    out = []
    for m, lst in sorted(blocks.items()):
        for j in lst:
            T = int(pd.Timestamp(j['t']).value // 1_000_000)
            rows, src = ohlcv_at(j['pair'], T)
            r = dict(mon=m, t=j['t'], pair=j['pair'], gate=j['gate'], src=src)
            if rows and len(rows) >= 60:
                me = inds(rows)
                for k in keys:
                    if k in j['ctx'] and j['ctx'][k] is not None and me.get(k) is not None:
                        r[f'eng_{k}'] = j['ctx'][k]; r[f'my_{k}'] = me[k]
            out.append(r)
    P = pd.DataFrame(out)
    P.to_csv(os.path.join(ROOT, 'reports', 'study_ml_bughunt_inputs_pair.csv'), index=False)
    bo = []
    for m, lst in sorted(scans.items()):
        for j in lst:
            T = int(pd.Timestamp(j['t']).value // 1_000_000)
            rows, src = ohlcv_at('BTCUSDT', T)
            me = inds(rows)
            slope = round((me['ema20'] - me['ema20_prev3']) / me['ema20_prev3'] * 100, 4)
            bo.append(dict(mon=m, t=j['t'], src=src, eng_rsi=j.get('btc_rsi'), my_rsi=me['rsi'], eng_adx=j.get('btc_adx'), my_adx=me['adx'],
                           eng_adx_prev=j.get('btc_adx_prev'), my_adx_prev=me['adx_prev1'], eng_slope=j.get('btc_slope'), my_slope=slope))
    B = pd.DataFrame(bo)
    B.to_csv(os.path.join(ROOT, 'reports', 'study_ml_bughunt_inputs_btc.csv'), index=False)
    pd.set_option('display.width', 250)
    # relative error summaries
    def rel(e, mm, k):
        if k in ('rsi', 'rsi_prev1', 'rsi_prev2', 'adx', 'adx_prev1', 'pos_di', 'neg_di'):
            return (mm - e)            # points
        return (mm / e - 1) * 100       # %
    rows = []
    for m, g in P.groupby('mon'):
        r = dict(mon=m, n=len(g), src_ticks=(g.src == 'ticks').mean())
        for k in keys:
            if f'eng_{k}' in g:
                x = rel(g[f'eng_{k}'].astype(float), g[f'my_{k}'].astype(float), k).dropna()
                r[f'{k}_medabs'] = x.abs().median(); r[f'{k}_max'] = x.abs().max()
        rows.append(r)
    S = pd.DataFrame(rows)
    print(S[['mon', 'n', 'src_ticks', 'price_max', 'ema5_max', 'ema20_max', 'ema50_max', 'rsi_max', 'rsi_prev2_max', 'adx_max', 'atr_max',
             'high_20_max', 'low_20_max']].round(4).to_string())
    B['d_rsi'] = B.my_rsi - B.eng_rsi; B['d_adx'] = B.my_adx - B.eng_adx; B['d_slope'] = B.my_slope - B.eng_slope
    print(B.groupby('mon')[['d_rsi', 'd_adx', 'd_slope']].agg(lambda s: s.abs().max()).round(5))


if __name__ == '__main__':
    main()


def lag_match(csv=os.path.join(ROOT, 'reports', 'study_ml_bughunt_inputs_pair.csv')):
    """second pass: for each sampled BLOCK line find the instant T ≤ t (back to 5 min) at which the rebuilt forming bar reproduces the
    engine's price AND 20-bar high/low exactly, then re-compare every indicator at that T. Lag = journal stamp − T (the pair's
    klines are fetched earlier in the scan than the line is written)."""
    P = pd.read_csv(csv)
    res = []
    for r in P.itertuples():
        if pd.isna(getattr(r, 'eng_price', np.nan)):
            continue
        t = int(pd.Timestamp(r.t).value // 1_000_000)
        z = tick_day(r.pair, (t // DAY) * DAY)
        if z is None:
            continue
        tt, pp = z[0], z[1]
        a = np.searchsorted(tt, t - 300_000); b = np.searchsorted(tt, t)
        cand = np.flatnonzero(np.abs(pp[a:b] / r.eng_price - 1) < 1e-6)
        best = None
        for k in cand[::-1][:40]:
            T = int(tt[a + k]) + 1
            rows, src = ohlcv_at(r.pair, T)
            me = inds(rows)
            if abs(me['high_20'] / r.eng_high_20 - 1) < 1e-6 and abs(me['low_20'] / r.eng_low_20 - 1) < 1e-6:
                best = (T, me); break
        if best is None:
            res.append(dict(mon=r.mon, t=r.t, pair=r.pair, matched=False)); continue
        T, me = best
        d = dict(mon=r.mon, t=r.t, pair=r.pair, matched=True, lag_s=(t - T) / 1000)
        for k in ('ema5', 'ema20', 'ema50', 'rsi', 'rsi_prev2', 'adx', 'adx_prev1', 'atr', 'pos_di', 'neg_di'):
            e = getattr(r, f'eng_{k}', np.nan)
            if pd.notna(e):
                d[f'd_{k}'] = (me[k] - e) if k in ('rsi', 'rsi_prev2', 'adx', 'adx_prev1', 'pos_di', 'neg_di') else (me[k] / e - 1) * 100
        res.append(d)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(ROOT, 'reports', 'study_ml_bughunt_inputs_pair_lagmatch.csv'), index=False)
    pd.set_option('display.width', 250)
    g = R.groupby('mon')
    print(pd.concat([g.matched.mean().rename('matched'), g.lag_s.median().rename('lag_med_s'), g.lag_s.max().rename('lag_max_s'),
                     g[[c for c in R.columns if c.startswith('d_')]].agg(lambda s: s.abs().max())], axis=1).round(5).to_string())
