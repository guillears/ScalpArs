"""Flatbase: enter at the S4 signal vs wait for FB_CONFIRM, both on TODAY's filters (NEAR_BASE 2.09 % + HIGHER_HIGH), 2026-11-19.
Rows: reports/OOS_REAL_FLATBASE_2026-11-17_trades.csv (route TRADE; statuses TRADE / SKIP_CONFIRM / SKIP_HH) with its s4_netf / cf_netf
(today's +5 / −10 / 72 h exit, 0.10 % slip, real funding). Base = the 5m close 30 min before the spike bar (services/flatbase.py c[on − 6]),
read from the shared 5m cache (frenzy_ta_build.load). NEAR_BASE on the S4 path = the S4 close ≤ 2.09 % above base (DL 577 follow-up)."""
import os, sys, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(R, 'scripts')); sys.path.insert(0, R)
import frenzy_ta_build as TB
NB = 2.09; BAR = 300_000
rng = np.random.default_rng(3)
t = pd.read_csv(f'{R}/reports/OOS_REAL_FLATBASE_2026-11-17_trades.csv')
t = t[(t.route == 'TRADE') & t.status.isin(['TRADE', 'SKIP_CONFIRM', 'SKIP_HH'])].reset_index(drop=True)
ms = lambda s: pd.to_datetime(s).values.astype('datetime64[ms]').astype('int64')
t['spk'] = ms(t.spike); t['s4o'] = ms(t.s4_t); t['cfe'] = ms(t.cf_entry_t.fillna('1970-01-01 00:00:00'))
base = np.full(len(t), np.nan); s4px = np.full(len(t), np.nan); cfpx = np.full(len(t), np.nan)
for p, g in t.groupby('pair'):
    d = TB.load(p)
    if d is None: continue
    ot = d[0] if isinstance(d, tuple) else d.index.values
    cc = d[4] if isinstance(d, tuple) else d['c'].values
    def close_at(o):
        i = np.searchsorted(ot, o); return cc[i] if i < len(ot) and ot[i] == o else np.nan
    for i, r in g.iterrows():
        base[i] = close_at(r.spk - 7 * BAR); s4px[i] = close_at(r.s4o)   # OOS `spike` = the spike bar's CLOSE; engine base = c[on - 6] = the bar opening 7 bars before that close
        if r.cfe > 0: cfpx[i] = close_at(r.cfe - BAR)
t['base'] = base; t['d_s4'] = (s4px / base - 1) * 100; t['d_cf'] = (cfpx / base - 1) * 100
print('rows', len(t), 'base found', np.isfinite(base).sum(), 'cf px found', np.isfinite(cfpx).sum(), 'of', (t.cfe > 0).sum())
ok = np.isfinite(t.base)
t = t[ok].reset_index(drop=True)
conf = (t.status == 'TRADE') & (t.d_cf > NB) & t.cf_netf.notna()
s4 = (t.d_s4 > NB) & t.s4_netf.notna()
t['half'] = pd.to_datetime(t.spike).dt.year.astype(str).str[2:] + np.where(pd.to_datetime(t.spike).dt.month <= 6, '-H1', '-H2')
t['day'] = t.spk // 86_400_000
L = ['| block | policy | trades | WR | avg / trade (day-block 95 % CI) | Σ points |', '|---|---|---|---|---|---|']
def ci(v, days):
    b = days // 10; u = np.unique(b); idx = {x: np.where(b == x)[0] for x in u}
    bs = [np.concatenate([v[idx[x]] for x in rng.choice(u, len(u))]).mean() for _ in range(2000)]; return np.percentile(bs, [2.5, 97.5])
for blk in ['H2024', 'DESIGN', 'FWD', 'ALL']:
    m = (t.block == blk) if blk != 'ALL' else pd.Series(True, index=t.index)
    for nm, sel, col in [('wait for confirm (today)', conf, 'cf_netf'), ('enter at signal', s4, 's4_netf')]:
        g = t[m & sel]; v = g[col].values
        c = ci(v, g.day.values) if len(g) >= 10 else (np.nan, np.nan)
        L.append(f'| {blk} | {nm} | {len(g)} | {(v > 0).mean():.0%} | {v.mean():+.2f} ({c[0]:+.2f} … {c[1]:+.2f}) | {v.sum():+.0f} |' if len(g) else f'| {blk} | {nm} | 0 | | | |')
L += ['', '| half (spike) | confirm: N · avg · Σ | signal: N · avg · Σ |', '|---|---|---|']
for h in sorted(t.half.unique()):
    a = t[(t.half == h) & conf].cf_netf; b = t[(t.half == h) & s4].s4_netf
    L.append(f'| {h} | {len(a)} · {a.mean():+.2f} · {a.sum():+.0f} | {len(b)} · {b.mean():+.2f} · {b.sum():+.0f} |')
L += ['', '### Where the difference comes from (signal-entry trades, all blocks, by what the confirm path did with them)', '',
      '| what confirm did | N | signal-entry avg | signal-entry Σ | confirm Σ on the same episodes |', '|---|---|---|---|---|']
for nm, m in [('confirmed and traded', s4 & conf), ('confirmed but NEAR_BASE at confirm', s4 & (t.status == 'TRADE') & ~(t.d_cf > NB)),
              ('never confirmed (SKIP_CONFIRM)', s4 & (t.status == 'SKIP_CONFIRM')), ('higher high (SKIP_HH)', s4 & (t.status == 'SKIP_HH'))]:
    g = t[m]; cs = g.cf_netf.where(conf[m], 0).fillna(0).sum()
    L.append(f'| {nm} | {len(g)} | {g.s4_netf.mean():+.2f} | {g.s4_netf.sum():+.0f} | {cs:+.0f} |')
g = t[conf & ~s4]; L.append(f'| confirm-only (NEAR_BASE at the signal, not at confirm) | {len(g)} | — | 0 | {g.cf_netf.sum():+.0f} |')
open(f'{R}/reports/FLATBASE_S4_VS_CONFIRM_NB_2026-11-19_tables.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
