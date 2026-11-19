"""WILLY_HT W/L after the DL 594 dip cut (2026-11-19) — build step. Research only.

Inputs : reports/WILLY_HT_WL594_data/ht_master_offpeak.csv (44 + 5 master HT fills re-priced on today's exit, + 4 current-batch rows)
         reports/MASTER_POOL_stacked.csv (entry stamps) · the current-batch orders export (argv[1], optional)
Fetches: Binance USDT-M 1m klines entry−4 h → entry+24 h per fill, BTC 1d klines (cached in reports/WILLY_HT_WL594_data/;
         ≤ 1,200 weight/min — reads X-MBX-USED-WEIGHT-1M, stops on 418/429).
Output : reports/WILLY_HT_WL594_2026-11-19_fills.csv (one row per fill: outcome, path labels, every entry_* stamp, derived features)

    venv/bin/python scripts/willy_ht_wl594_build.py [orders_export.csv]
"""
import os, sys, json, time, urllib.request
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'reports', 'WILLY_HT_WL594_data')
OUT = os.path.join(REPO, 'reports', 'WILLY_HT_WL594_2026-11-19_fills.csv')
MIN = 60_000
TAKER, MAKER = 0.05, 0.02
_used = [0]


def get(url):
    for _ in range(5):
        req = urllib.request.Request(url, headers={'User-Agent': 'research-client'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                w = int(r.headers.get('X-MBX-USED-WEIGHT-1M', '0') or 0)
                if w > 900:
                    time.sleep(61)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                sys.exit(f'Binance {e.code} — stopping')
            time.sleep(2)
        except (TimeoutError, OSError):
            time.sleep(3)
    raise RuntimeError(url)


def klines(sym, t0, t1, iv='1m'):
    fn = os.path.join(DATA, f'{sym}_{iv}_{t0}_{t1}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    out, s = [], t0
    while s < t1:
        k = get(f'https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={iv}&startTime={s}&endTime={t1 - 1}&limit=1000')
        if not k:
            break
        out += k
        s = k[-1][0] + (MIN if iv == '1m' else 86_400_000)
        time.sleep(0.25)
    json.dump(out, open(fn, 'w'))
    return out


def ms(v):
    return int(pd.Timestamp(v).tz_localize(None).value // 1_000_000) if pd.Timestamp(v).tzinfo is None else int(pd.Timestamp(v).tz_convert(None).value // 1_000_000)


def path_facts(K, t_in, E):
    """K = 1m klines from entry−4 h. Re-price today's exit (+5 net maker TP / −4 net taker stop / 24 h), WORKED, REACH,
    and the pre-entry features from bars CLOSED before the entry minute."""
    a = np.array([[float(x) for x in k[:6]] for k in K])
    t, o, h, l, c, v = a.T
    m0 = (t_in // MIN) * MIN
    pre = a[t < m0]
    post = a[(t >= m0) & (t < t_in + 1440 * MIN)]
    x_tp = E * (1 + (5 + TAKER) / 100) / (1 - MAKER / 100)
    p_sl = E * (100 - 4 + TAKER) / (100 - TAKER)
    p_w = E * 1.01
    lo, hi = post[:, 3], post[:, 2]
    i_sl = int(np.argmax(lo <= p_sl)) if (lo <= p_sl).any() else 10 ** 9
    i_tp = int(np.argmax(hi >= x_tp * 1.0001)) if (hi >= x_tp * 1.0001).any() else 10 ** 9
    if i_sl == 10 ** 9 and i_tp == 10 ** 9:
        rsn, i_out = 'TIME', len(post) - 1
    elif i_sl <= i_tp:
        rsn, i_out = 'SL', i_sl
    else:
        rsn, i_out = 'TP', i_tp
    k = i_sl if i_sl < 10 ** 9 else len(post)
    hh = hi[:k] if k > 0 else hi[:1]
    reach = (hh.max() / E - 1) * 100 if len(hh) else 0.0
    worked = bool((hi[:k] >= p_w).any()) if k > 0 else False
    f = dict(re_reason=rsn, re_minutes=int(i_out), worked=worked, reach=reach,
             trough_before_tp=(lo[:min(i_tp, len(lo) - 1) + 1].min() / E - 1) * 100)
    if len(pre) >= 240:
        cl = pre[:, 4]
        f.update(pre_ret15=(cl[-1] / cl[-16] - 1) * 100, pre_ret60=(cl[-1] / cl[-61] - 1) * 100, pre_ret240=(cl[-1] / cl[0] - 1) * 100,
                 pre_dd60=(cl[-1] / pre[-60:, 2].max() - 1) * 100,
                 pre_rvol60=float(np.std(np.diff(np.log(cl[-61:]))) * 100),
                 pre_vol15_vs60=float(pre[-15:, 5].mean() / max(pre[-75:-15, 5].mean(), 1e-12)))
    return f, a


def main():
    src = pd.read_csv(os.path.join(DATA, 'ht_master_offpeak.csv'))
    pool = pd.read_csv(os.path.join(REPO, 'reports', 'MASTER_POOL_stacked.csv'), low_memory=False)
    pool = pool[pool.entry_strategy.astype(str).str.contains('WILLY_HT')].copy()
    pool['opened'] = pool.opened_at.astype(str).str[:19]
    exp = None
    if len(sys.argv) > 1:
        exp = pd.read_csv(sys.argv[1], low_memory=False)
        exp = exp[exp.entry_strategy.astype(str) == 'FRENZY_WILLY_HT'].copy()
        exp['opened'] = exp.opened_at.astype(str).str[:19].str.replace(' ', 'T')
    pool['opened'] = pool.opened.str.replace(' ', 'T')
    # BTC daily EMA50 − EMA200 (last closed day before entry)
    b = klines('BTCUSDT', ms('2025-09-01'), ms('2026-11-19'), '1d')
    bd = pd.DataFrame([[int(k[0]), float(k[4])] for k in b], columns=['t', 'c'])
    bd['e50'] = bd.c.ewm(span=50, adjust=False).mean(); bd['e200'] = bd.c.ewm(span=200, adjust=False).mean()
    bd['gap'] = (bd.e50 / bd.e200 - 1) * 100
    rows, paths = [], {}
    for _, r in src.iterrows():
        o = r.to_dict()
        st = pool if r.set == 'master' else exp
        if st is not None:
            m = st[(st.pair == r.pair) & (st.opened == str(r.opened)[:19])]
            if len(m):
                o.update({c: m.iloc[0][c] for c in st.columns if c.startswith('entry_') or c in ('close_reason', 'pnl_percentage', 'stack_pct', 'stack_keep', 'leverage', 'investment')})
                o['stamp_match'] = True
            else:
                o['stamp_match'] = False
        t_in = ms(r.opened)
        o['entry_ms'] = t_in
        K = klines(r.pair, (t_in // MIN) * MIN - 240 * MIN, (t_in // MIN) * MIN + 1441 * MIN)
        f, a = path_facts(K, t_in, float(r.entry_price))
        o.update(f)
        paths[f"{r.pair}|{t_in}"] = a[:, [0, 2, 3, 4]].tolist()
        o['spike_age_min'] = (t_in - ms(r.entry_frenzy_spike_at)) / MIN if isinstance(r.entry_frenzy_spike_at, str) else np.nan
        ts = pd.Timestamp(t_in, unit='ms')
        o['hour_utc'] = ts.hour; o['weekday'] = ts.weekday(); o['day'] = ts.strftime('%Y-%m-%d')
        j = bd.index[bd.t + 86_400_000 <= t_in]
        o['btc_1d_ema50_200_gap'] = bd.gap.loc[j[-1]] if len(j) else np.nan
        o['failed_spike_kept'] = bool(r.keep)
        rows.append(o)
    F = pd.DataFrame(rows)
    F['win'] = F.net > 0
    F = F.sort_values('entry_ms').reset_index(drop=True)
    F['pair_repeat'] = F.groupby(['set', 'pair']).cumcount()
    F.to_csv(OUT, index=False)
    json.dump(paths, open(os.path.join(DATA, 'paths_1m.json'), 'w'))
    print('rows', len(F), 'stamp match', F.stamp_match.value_counts().to_dict())
    print(pd.crosstab(F.re_reason, F.net.round(1)))


if __name__ == '__main__':
    main()
