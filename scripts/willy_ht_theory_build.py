"""WILLY_HT THEORY (2026-11-19) — build step: HT episodes, their price paths and entry-time regime features.

Research only. Reuses the independent WILLY rebuild (the `wsignals` / `wsim` / `wdata` / `wdeep_feat` modules and the signal pickles
sig_all.pkl / sig_D.pkl). Point WHT_REBUILD_DIR at the folder that holds them (the modules live in its `code/` sub-folder).

    WHT_REBUILD_DIR=<rebuild folder> venv/bin/python scripts/willy_ht_theory_build.py main     # 2025-01 → 2026-11-17
    WHT_REBUILD_DIR=<rebuild folder> venv/bin/python scripts/willy_ht_theory_build.py D        # 2023-H2 holdout (all setups as HT)

Episode = an ARMED WILLY-A episode routed the way today's engine routes it, WITHOUT the portfolio (slots / pair-held are applied
in the eval step): walk its red bars in order; nomiss + 24 h volume < floor → ends; unreadable turnover → stays armed (next red);
DISLOC > 1 % → ends; else this red is the DECISIVE bar (R, the FAILED_SPIKE flag and every red-bar stamp are recorded).
Plain-vs-HT routing (R ≥ cut) and FAILED_SPIKE are applied in the eval step so their cuts can be varied.
"""
import os, sys, pickle
import numpy as np, pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else 'main'
if MODE == 'D':
    os.environ['WMODE'] = 'D'
RB = os.environ.get('WHT_REBUILD_DIR')
if not RB:
    sys.exit('set WHT_REBUILD_DIR to the WILLY rebuild folder')
sys.path.insert(0, os.path.join(RB, 'code'))
import wsim                                   # noqa: E402
import wdeep_feat as WF                       # noqa: E402
from wdata import load_5m, load_funding       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, 'reports', 'WILLY_HT_THEORY_2026-11-19_data'); os.makedirs(OUT, exist_ok=True)
MIN = 60_000; BAR = 300_000; H1 = 3_600_000
WIN = (240 + 1440 + 15) * MIN                 # longest entry window (4 h) + 24 h hold + buffer
VMIN = wsim.VMIN; DISLOC = wsim.DISLOC


def decisive(e, sup, force_ht):
    for r in e['reds']:
        if e.get('nomiss') and (r['vol24'] is None or r['vol24'] < VMIN):
            return None
        s = sup.get(e['pair'])
        R = (r['vol24'] / (r['close'] * s)) if (s and r['vol24']) else None
        if not force_ht and R is None:
            continue
        if r['disloc'] > DISLOC:
            return None
        return r, R
    return None


def load_path(sym, t0, t1):
    """1m when every day of [t0, t1) is cached, else 5m. → (array t, o, h, l, c), res."""
    fk = WF.k1m('raw1m', sym, t0, t1)
    if fk is not None and len(fk) >= (t1 - t0) / MIN * 0.97:
        return fk[['t', 'open', 'high', 'low', 'close']].astype(float).values, '1m'
    d = WF.p5(sym)
    if d is None:
        return None, None
    x = d[(d.t >= t0) & (d.t < t1)]
    return x[['t', 'o', 'h', 'l', 'c']].values.astype(float), '5m'


def main():
    sig = pickle.load(open(os.path.join(RB, 'sig_D.pkl' if MODE == 'D' else 'sig_all.pkl'), 'rb'))
    sup = wsim.supply_table()
    force_ht = MODE == 'D'
    btc = WF.Btc()
    br = pickle.load(open(os.path.join(RB, 'breadth_D.pkl' if MODE == 'D' else 'breadth_main.pkl'), 'rb'))
    fz = np.array(sorted(x['armed_ms'] for v, _ in sig.values() for x in v if x['status'] == 'ARMED'))
    t_from = pd.Timestamp('2023-07-01' if MODE == 'D' else '2025-01-13').value // 1_000_000
    rows, paths = [], {}
    pairs = sorted(sig)
    for pi, pair in enumerate(pairs):
        eps = [e for e in sig[pair][0] if e['status'] == 'ARMED' and e['reds']]
        if not eps:
            continue
        d5 = None
        for e in eps:
            dd = decisive(e, sup, force_ht)
            if dd is None:
                continue
            r, R = dd
            if r['pass_ms'] < t_from:
                continue
            if d5 is None:
                d5 = load_5m(pair)
            if d5 is None:
                continue
            j = int(np.searchsorted(d5.t.values, r['bar_ms']))      # by time: sig 'j' indexes the loader of the run that built it
            if j >= len(d5) or int(d5.t.iloc[j]) != r['bar_ms']:
                continue
            red_hi = float(d5.h.iloc[j])
            P, res = load_path(pair, r['pass_ms'], r['pass_ms'] + WIN)
            if P is None or len(P) < 50:
                continue
            f = load_funding(pair)
            fund = f[(f.t > r['pass_ms']) & (f.t <= r['pass_ms'] + WIN)][['t', 'rate']].values if f is not None else np.zeros((0, 2))
            key = (pair, int(r['pass_ms']))
            paths[key] = dict(P=P.astype(np.float64), res=res, fund=fund.astype(np.float64))
            vwap = r['close'] / (1 + r['vs_vwap'] / 100) if r['vs_vwap'] is not None else np.nan
            o = dict(pair=pair, spike_ms=int(e['spike_ms']), armed_ms=int(e['armed_ms']), pass_ms=int(r['pass_ms']), bar_ms=int(r['bar_ms']),
                     R=R, close=r['close'], live=r['live'], red_hi=red_hi, peak=r['peak'], vwap=vwap, vs_vwap=r['vs_vwap'],
                     off_peak=(r['live'] / r['peak'] - 1) * 100, bar_ret=r['bar_ret'], vol_mult=r['vol_mult'], atr=r['atr'],
                     adx=r['adx'], gap813=r['gap813'], vol24=r['vol24'], g1h=r['g1h'], s4h_ema20=r['s4h_ema20'],
                     late=e.get('late'), nomiss=bool(e.get('nomiss')), wait_bars=(r['pass_ms'] - e['armed_ms']) // BAR, res=res)
            o.update(btc.at(r['pass_ms']))
            hk = (r['pass_ms'] // H1) * H1
            i = np.searchsorted(br.index.values, hk, 'right') - 1
            o['breadth'] = float(br.iloc[i]) if i >= 0 else np.nan
            o['mkt_frenzy_24h'] = int(((fz > r['pass_ms'] - 86_400_000) & (fz <= r['pass_ms'])).sum())
            rows.append(o)
        if pi % 50 == 0:
            print(pi, pair, len(rows), flush=True)
    X = pd.DataFrame(rows).sort_values('pass_ms').reset_index(drop=True)
    pickle.dump(dict(X=X, paths=paths), open(os.path.join(OUT, f'episodes_{MODE}.pkl'), 'wb'))
    print('done', len(X), X.res.value_counts().to_dict())


if __name__ == '__main__':
    main()
