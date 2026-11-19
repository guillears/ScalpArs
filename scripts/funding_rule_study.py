"""FUNDING_RULE study (2026-11-19): skip / half-size EXH and DIST shorts at deep negative funding. Pre-reg: reports/FUNDING_RULE_2026-11-19_PREREG.md"""
import numpy as np, pandas as pd, os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(7)
TS = [-0.05, -0.10, -0.20, -0.50, -1.00]
CUT = pd.Timestamp('2025-07-01').value // 10**6

def load():
    e = pd.read_csv(f'{R}/reports/BASELINE_TRADES_EXHAUSTION_2026-11-18.csv'); e = e[~e.dl582_block]
    x2 = pd.read_csv(f'{R}/reports/EXH_EXIT2_2026-11-18_trades.csv.gz', usecols=['pair', 'entry_ms', 'net_REF'])
    e = e.merge(x2, on=['pair', 'entry_ms'])  # net = EXH_EXIT2 1m REF net (today's exit, real funding); fund = funding of the baseline exit (descriptive)
    e = pd.DataFrame(dict(pair=e.pair, ms=e.entry_ms, net=e.net_REF, fund=e.fund, rate=e.f_last))
    w = pd.read_csv(f'{R}/reports/DIST_WL_2026-11-18_trades.csv.gz'); w = w[w.keep_strength]
    d = pd.read_csv(f'{R}/reports/DIST_EXIT_2026-11-18_trades.csv.gz'); d = d[d.cell == 'REF_B']
    m = w[['pair', 'entry_ms', 'fund_last']].merge(d[['pair', 'entry_ms', 'net', 'funding']], on=['pair', 'entry_ms'])
    d = pd.DataFrame(dict(pair=m.pair, ms=m.entry_ms, net=m.net, fund=m.funding, rate=m.fund_last))
    print(f'joins: EXH {len(e)} fills, DIST {len(w)} kept → {len(d)} with an Exit-B row')
    out = {}
    for k, x in (('EXHAUSTION', e), ('DISTRIBUTION', d)):
        x = x.copy(); x['day'] = x.ms // 86400000; x['blk'] = np.where(x.ms < CUT, 'DESIGN', 'HOLDOUT')
        x['half'] = pd.to_datetime(x.ms, unit='ms').dt.year.astype(str) + np.where(pd.to_datetime(x.ms, unit='ms').dt.month <= 6, 'H1', 'H2')
        out[k] = x.reset_index(drop=True)
    return out

def delta(x, T, mode, rate=None):
    r = x.rate.values if rate is None else rate
    hit = np.nan_to_num(r, nan=9) <= T
    return -(x.net.values * hit).sum() * (1.0 if mode == 'SKIP' else 0.5), hit

def boot_ci(x, hit, mode, n=2000):
    days = np.unique(x.day.values // 10); idx = {b: np.where(x.day.values // 10 == b)[0] for b in days}
    v = -(x.net.values * hit) * (1.0 if mode == 'SKIP' else 0.5); s = []
    for _ in range(n):
        pick = rng.choice(days, len(days)); s.append(sum(v[idx[b]].sum() for b in pick))
    return np.percentile(s, [2.5, 97.5])

def main():
    L = []; data = load()
    for k, x in data.items():
        win = x[x.net > 0].net.mean(); los = -x[x.net <= 0].net.mean(); be = los / (win + los)
        L += [f'\n## {k}', f'- Fills {len(x)} (DESIGN {sum(x.blk=="DESIGN")} · HOLDOUT {sum(x.blk=="HOLDOUT")}); rate coverage {x.rate.notna().mean():.0%}; '
              f'avg net {x.net.mean():+.2f} %; WR {(x.net>0).mean():.0%}; breakeven WR {be:.0%}; funding paid avg {x.fund.mean():+.2f} % / fill (net already includes it).', '',
              '### Funding paid by entry rate (descriptive)', '', '| rate at entry (%/interval) | N | WR | avg net % | avg funding % | funding as % of gross on winners | days |', '|---|---|---|---|---|---|---|']
        bins = [-99, -1.0, -0.5, -0.2, -0.1, -0.05, 0, 99]
        x['b'] = pd.cut(x.rate, bins, right=True)
        for b, g in x.groupby('b', observed=True):
            wg = g[g.net > 0]; share = (-wg.fund).sum() / (wg.net - wg.fund).sum() * 100 if len(wg) else np.nan
            L.append(f'| {b} | {len(g)} | {(g.net>0).mean():.0%} | {g.net.mean():+.2f} | {g.fund.mean():+.2f} | {share:.0f} % | {g.day.nunique()} |')
        L += ['', '### Rules (Δ = change in summed net % at one size vs today)', '',
              '| rule | DESIGN Δ (95 % CI) | blocked N · WR · avg · days | HOLDOUT Δ | blocked N · WR · avg | top pair share of loss | null 95th | pass |', '|---|---|---|---|---|---|---|---|']
        Dx = x[x.blk == 'DESIGN'].reset_index(drop=True); Hx = x[x.blk == 'HOLDOUT'].reset_index(drop=True)
        nullmax = []
        for _ in range(1000):
            r = Dx.groupby('half').rate.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values).values
            nullmax.append(max(delta(Dx, T, m, r)[0] for T in TS for m in ('SKIP', 'HALF')))
        q95 = np.percentile(nullmax, 95)
        for m in ('SKIP', 'HALF'):
            for T in TS:
                dD, hD = delta(Dx, T, m); dH, hH = delta(Hx, T, m); ci = boot_ci(Dx, hD, m)
                bD = Dx[hD]; bH = Hx[hH]; ball = x[np.nan_to_num(x.rate.values, nan=9) <= T]
                lp = ball[ball.net < 0].groupby('pair').net.sum(); top = (lp.min() / lp.sum() * 100) if len(lp) else 0
                ok = (dD > 0 and ci[0] > 0 and dH > 0 and len(ball) >= 15 and (ball.net > 0).mean() < be and ball.net.mean() < 0
                      and ball.day.nunique() >= 8 and top < 50 and dD > q95)
                f = lambda g: f'{len(g)} · {(g.net>0).mean():.0%} · {g.net.mean():+.2f} · {g.day.nunique()}d' if len(g) else '0'
                L.append(f'| {m} rate ≤ {T} | {dD:+.1f} ({ci[0]:+.1f} … {ci[1]:+.1f}) | {f(bD)} | {dH:+.1f} | {f(bH)} | {top:.0f} % | {q95:.1f} | {"✓" if ok else "✗"} |')
    open(f'{R}/reports/FUNDING_RULE_2026-11-19_tables.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    chart(data)

def chart(d):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    bins = [-99, -1.0, -0.5, -0.2, -0.1, -0.05, 0, 99]; lab = ['≤−1', '−1…−0.5', '−0.5…−0.2', '−0.2…−0.1', '−0.1…−0.05', '−0.05…0', '>0']
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, (k, x) in zip(ax, d.items()):
        x = x.copy(); x['b'] = pd.cut(x.rate, bins)
        for j, blk in enumerate(['DESIGN', 'HOLDOUT']):
            a.bar(np.arange(7) + (j - 0.5) * 0.27, x[x.blk == blk].groupby('b', observed=False).net.mean().values, 0.27, label=f'avg net % ({blk})', color=['#4C78A8', '#72B7B2'][j])
        a.bar(np.arange(7) + 0.4, x.groupby('b', observed=False).fund.mean().values, 0.18, label='avg funding paid %', color='#E45756')
        a.axhline(0, color='k', lw=.6); a.set_xticks(range(7)); a.set_xticklabels(lab, rotation=30, fontsize=8)
        a.set_title(f'{k}: net % per fill by funding rate at entry'); a.set_xlabel('funding rate at entry (% per interval; negative = shorts pay)')
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f'{R}/reports/FUNDING_RULE_2026-11-19.png', dpi=110)

if __name__ == '__main__':
    main()
