#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 5b: do the yr5 momentum-long ENTRIES beat RANDOM entries on the same pairs? For every yr5 ML fill, 3
random taker entries on the same pair within ±3 days (seeded), each priced on raw aggTrades with the same simple bracket exits as
part 5 (net +0.5 / −0.7, 180-min cap) and a 60-min hold. Per half-year: ML entries vs random. Out: reports/study_ml_bughunt_control.csv"""
import os, numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.dirname(__file__))
import study_ml_bughunt_exitwalk as X
rng = np.random.default_rng(20261008)
w = pd.read_csv(os.path.join(X.REP, 'study_ml_bughunt_exitwalk.csv'))
w['open_ms'] = pd.to_datetime(w.opened_at).values.astype('datetime64[ms]').astype(np.int64)
rows = []
for r in w.sort_values(['pair', 'open_ms']).itertuples():
    for k in range(3):
        t0 = int(r.open_ms + rng.integers(-3 * 86_400_000, 3 * 86_400_000))
        t, p, _ = X.ticks(r.pair, t0, t0 + 3 * 3600_000)
        if len(t) < 10:
            continue
        e = p[0]; pnl = (p / e - 1) * 100 - X.TAKER - X.TAKER * p / e
        a = X.first(pnl >= 0.5); b = X.first(pnl <= -0.7); h = X.first((t - t[0]) >= 180 * 60_000)
        c = [x for x in (a, b, h) if x is not None]; kk = min(c) if c else len(pnl) - 1
        br = 0.5 if (a is not None and kk == a) else -0.7 if (b is not None and kk == b) else pnl[kk]
        k60 = X.first((t - t[0]) >= 60 * 60_000)
        rows.append(dict(key=r.key, mon=r.mon, rnd_bracket=float(br), rnd_hold60=float(pnl[k60] if k60 is not None else pnl[-1]),
                         rnd_gross60=float((p[k60] / e - 1) * 100 if k60 is not None else (p[-1] / e - 1) * 100)))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(X.REP, 'study_ml_bughunt_control.csv'), index=False)
w['H'] = np.where(w.mon <= '2026-04', 'H1 Jan-Apr', 'H2 May-Oct'); R['H'] = np.where(R.mon <= '2026-04', 'H1 Jan-Apr', 'H2 May-Oct')
fee_ml = np.where(w.entry_order_type == 'MAKER', X.MAKER, X.TAKER) + X.TAKER
w['gross60'] = w.bl_hold60m + fee_ml
out = pd.DataFrame({'ML engine pct': w.groupby('H').pct.mean(), 'ML bracket +0.5/-0.7': w.groupby('H')['bl_tp0.5_sl0.7'].mean(),
                    'RANDOM bracket (taker)': R.groupby('H').rnd_bracket.mean(), 'ML hold60 net': w.groupby('H').bl_hold60m.mean(),
                    'RANDOM hold60 net (taker)': R.groupby('H').rnd_hold60.mean(), 'ML gross 60m': w.groupby('H').gross60.mean(),
                    'RANDOM gross 60m': R.groupby('H').rnd_gross60.mean()})
out.loc['YEAR'] = [w.pct.mean(), w['bl_tp0.5_sl0.7'].mean(), R.rnd_bracket.mean(), w.bl_hold60m.mean(), R.rnd_hold60.mean(), w.gross60.mean(), R.rnd_gross60.mean()]
print(out.round(4).T.to_string())
