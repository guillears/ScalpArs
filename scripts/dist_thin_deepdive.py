"""DIST thin-coin deep dive (2026-11-19). Descriptive read of the frozen ENTRY_WL observe line (vol24 <= $42M); no new rule is fitted.
Uses the DIST_EXIT REF_B net (Exit B, 1m, fees + slip model + real funding) on the RSI1h-kept cohort."""
import os, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(11)
w = pd.read_csv(f'{R}/reports/DIST_WL_2026-11-18_trades.csv.gz'); w = w[w.keep_strength]
d = pd.read_csv(f'{R}/reports/DIST_EXIT_2026-11-18_trades.csv.gz'); d = d[d.cell == 'REF_B']
x = w.merge(d[['pair', 'entry_ms', 'net', 'kind', 'slip', 'funding', 'sl_pct', 'reach', 'worked']], on=['pair', 'entry_ms'], suffixes=('_wl', '')).copy()
x['day'] = x.entry_ms // 86400000; x['thin'] = x.vol24 <= 42e6
x['blk'] = np.where(x.entry_ms < pd.Timestamp('2025-07-01').value // 10**6, 'DESIGN', 'HOLDOUT')
x['half'] = pd.to_datetime(x.entry_ms, unit='ms').dt.year.astype(str).str[2:] + np.where(pd.to_datetime(x.entry_ms, unit='ms').dt.month <= 6, '-H1', '-H2')
print(f'fills {len(x)} (thin {x.thin.sum()})')

def wilson_lo(k, n, z=1.645):
    p = k / n; return (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
def kelly(g):
    W = g[g.net > 0].net.mean(); L = -g[g.net <= 0].net.mean(); b = W / L; p = wilson_lo((g.net > 0).sum(), len(g))
    return p - (1 - p) / b, (g.net > 0).mean() - (1 - (g.net > 0).mean()) / b
def boot(g, n=2000):
    days = g.day.values // 10; u = np.unique(days); idx = {b: np.where(days == b)[0] for b in u}; v = g.net.values; s = []
    for _ in range(n):
        pick = rng.choice(u, len(u)); s.append(np.concatenate([v[idx[b]] for b in pick]).mean())
    return np.percentile(s, [2.5, 97.5])
def row(lbl, g):
    if len(g) < 5: return f'| {lbl} | {len(g)} | | | | | | | |'
    ci = boot(g); kf, ko = kelly(g)
    return (f'| {lbl} | {len(g)} | {g.day.nunique()} | {(g.net>0).mean():.0%} | {g.net.mean():+.2f} ({ci[0]:+.2f} … {ci[1]:+.2f}) | {g[g.net>0].net.mean():+.2f} / {g[g.net<=0].net.mean():+.2f} '
            f'| {(g.entry_label=="WORKED").mean():.0%} · {g.reach.mean():.1f} · {((g.net<0)&(g.reach>=3)).mean():.0%} | {g.slip.mean():.2f} · {g.funding.mean():+.2f} | {kf:+.3f} · {ko:+.3f} |')
H = '| group | N | days | WR | avg net % (day-block 95 % CI) | avg win / loss | worked (+1 % first) · avg reach % · stopped after ≥ +3 % | slip · funding % | Kelly floor · observed |\n|---|---|---|---|---|---|---|---|---|'
L = ['## 1. Dose-response: 24 h volume buckets (all fills)', '', H]
for lo, hi, lb in [(0, 15e6, '< $15M'), (15e6, 25e6, '$15–25M'), (25e6, 42e6, '$25–42M'), (42e6, 80e6, '$42–80M'), (80e6, 200e6, '$80–200M'), (200e6, 1e13, '≥ $200M')]:
    L.append(row(lb, x[(x.vol24 >= lo) & (x.vol24 < hi)]))
L += ['', '## 2. Thin vs rest by block and half-year', '', H]
for b in ['DESIGN', 'HOLDOUT']:
    for t in [True, False]: L.append(row(f'{b} · {"thin" if t else "rest"}', x[(x.blk == b) & (x.thin == t)]))
L += ['', '| half | thin share of fills | thin avg | rest avg | Δ (rest − thin) |', '|---|---|---|---|---|']
for h, g in x.groupby('half'):
    L.append(f'| {h} | {g.thin.mean():.0%} | {g[g.thin].net.mean():+.2f} | {g[~g.thin].net.mean():+.2f} | {g[~g.thin].net.mean()-g[g.thin].net.mean():+.2f} |')
L += ['', '## 3. Mechanism (all fills): how thin losses happen', '', '| side | stop width avg % | reach avg % (best move before exit) | stopped after reaching ≥ +3 % | biggest-5-pair share of this side’s loss |', '|---|---|---|---|---|']
for t in [True, False]:
    g = x[x.thin == t]; los = g[g.net < 0]; pl = los.groupby('pair').net.sum().sort_values()
    L.append(f'| {"thin" if t else "rest"} | {g.sl_pct.mean():.2f} | {g.reach.mean():.2f} | {((g.net<0)&(g.reach>=3)).mean():.0%} | {pl.head(5).sum()/pl.sum():.0%} |')
# 4. continuous: rank correlation of log vol24 with net, per block
L += ['', '## 4. Continuous read (Spearman of log 24 h volume vs net %)', '']
for b, g in x.groupby('blk'):
    r = pd.Series(np.log10(g.vol24.values)).rank().corr(pd.Series(g.net.values).rank())
    perm = [pd.Series(np.log10(g.vol24.values)).rank().corr(pd.Series(rng.permutation(g.net.values)).rank()) for _ in range(500)]
    L.append(f'- {b}: ρ = {r:+.3f} (shuffle p {np.mean(np.abs(perm) >= abs(r)):.3f}, N {len(g)})')
m = pd.read_csv(f'{R}/reports/MASTER_POOL_stacked.csv', low_memory=False)
m = m[(m.is_paper == False) & m.exit_slippage_pct.notna() & m.entry_pair_volume_24h_usd.notna() & m.close_reason.astype(str).str.contains('STOP|_SL', regex=True)]
m = m.assign(v=pd.cut(m.entry_pair_volume_24h_usd, [0, 25e6, 42e6, 80e6, 200e6, 1e13], labels=['< $25M', '$25–42M', '$42–80M', '$80–200M', '≥ $200M']))
L += ['', '## 5. Real stop-type exit slippage by 24 h volume (master, LIVE, ALL sleeves — 0 DIST fills; a proxy for thin-coin cost; positive = cost)', '',
      '| 24 h volume | N | median | p90 |', '|---|---|---|---|']
for b, g in m.groupby('v', observed=True):
    L.append(f'| {b} | {len(g)} | {g.exit_slippage_pct.median():.2f} % | {g.exit_slippage_pct.quantile(.9):.2f} % |')
open(f'{R}/reports/DIST_THIN_DEEPDIVE_2026-11-19_tables.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
x.to_csv(f'{R}/reports/DIST_THIN_DEEPDIVE_2026-11-19_fills.csv.gz', index=False, columns=['pair', 'entry_ms', 'half', 'blk', 'vol24', 'thin', 'net', 'kind', 'entry_label', 'reach', 'sl_pct', 'slip', 'funding'])
