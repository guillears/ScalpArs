"""DIST_THIN_TP study (2026-11-19). Pre-reg: reports/DIST_THIN_TP_2026-11-19_PREREG.md. Engine = scripts/dist_exit_sim.py (unchanged)."""
import os, sys, multiprocessing as mp, numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(R, 'scripts')); sys.path.insert(0, R)
import dist_exit_sim as DS, nospot_sleeve_lib as NL
XS = [4, 5, 6, 7, 8]
CL = [DS.cells()[0]] + [dict(name=f'TP{x}', fam='A', sl=np.nan, tp=float(x), slatr=1, tpatr=0, mech='none', arm=np.inf, par=0.0, hold=24) for x in XS]
CUT = pd.Timestamp('2025-07-01').value // 10**6
rng = np.random.default_rng(5)

def cohort():
    S = pd.read_pickle(os.path.join(R, 'reports', 'DIST_WL_2026-11-18_data', 'rows.pkl'))
    w = pd.read_csv(f'{R}/reports/DIST_WL_2026-11-18_trades.csv.gz'); w = w[w.keep_strength]
    d = pd.read_csv(f'{R}/reports/DIST_EXIT_2026-11-18_trades.csv.gz'); d = d[d.cell == 'REF_B'][['pair', 'entry_ms', 'net']]
    k = w[['pair', 'entry_ms', 'vol24']].merge(d, on=['pair', 'entry_ms'])
    S = S.assign(pair=S.pair.astype(str), entry_ms=S.entry_ms.astype(np.int64))[['pair', 'entry_ms', 'atr15_pct', 'q5']].drop_duplicates(['pair', 'entry_ms'])
    F = k.merge(S, on=['pair', 'entry_ms'], how='left')
    F['atrp'] = F.atr15_pct.astype(float); F['slip'] = np.maximum(0.10, NL.slip_model(F.q5.values.astype(float))); F['keep'] = False
    return F.reset_index(drop=True)

def work(a):
    p, rows = a; out, _ = DS.run_rows(p, rows, CL); return rows.index.values, out

def main():
    F = cohort(); print('cohort', len(F), 'thin', int((F.vol24 <= 42e6).sum()), 'missing atr', int(F.atrp.isna().sum()))
    n, nC = len(F), len(CL); Z = dict(r=np.full((n, nC), np.nan), k=np.full((n, nC), -1), f=np.zeros((n, nC)), tpp=np.zeros((n, nC)), ok=np.zeros(n, bool), reach=np.zeros((n, nC)))
    with mp.get_context('fork').Pool(4) as pool:
        for ix, out in pool.imap_unordered(work, [(p, F[F.pair == p]) for p in F.pair.unique()], chunksize=4):
            for kk in Z: Z[kk][ix] = out[kk]
    k = Z['k']; tk = k != 0; s = F.slip.values[:, None]
    V = Z['r'] - s - np.where(tk, s, 0) - 0.05 - np.where(tk, 0.05, 0.02) + Z['f']
    V = np.where((~Z['ok'][:, None]) | (k == 9) | (k < 0), np.nan, V)
    reft = Z['tpp'][:, 0]
    for j, x in enumerate(XS, 1):   # cap binds only where today's TP > X
        V[:, j] = np.where(reft <= x, V[:, 0], V[:, j]); k[:, j] = np.where(reft <= x, k[:, 0], k[:, j])
    F['ref_engine'] = V[:, 0]; ok = np.isfinite(V).all(1)
    dd = (F.ref_engine - F.net).abs(); print('engine parity vs DIST_EXIT REF_B: max |Δ|', round(float(dd.max()), 3), '· rows |Δ| > 0.05:', int((dd > 0.05).sum()), '· mean Δ', round(float((F.ref_engine - F.net).mean()), 4), '· ok rows', int(ok.sum()))
    F = F[ok].reset_index(drop=True); V = V[ok]; k = k[ok]
    F['thin'] = F.vol24 <= 42e6; F['blk'] = np.where(F.entry_ms < CUT, 'DESIGN', 'HOLDOUT'); F['day'] = F.entry_ms // 86_400_000
    dt = pd.to_datetime(F.entry_ms, unit='ms'); F['half'] = dt.dt.year.astype(str).str[2:] + np.where(dt.dt.month <= 6, '-H1', '-H2')
    D = V[:, 1:] - V[:, [0]]
    L = []
    def ci(m, dlt):
        b = F.day.values[m] // 10; u = np.unique(b); idx = {x: np.where(b == x)[0] for x in u}; v = dlt[m]
        bs = [np.concatenate([v[idx[x]] for x in rng.choice(u, len(u))]).mean() for _ in range(2000)]; return np.percentile(bs, [2.5, 97.5])
    md = (F.blk == 'DESIGN').values & F.thin.values
    b = F.day.values[md] // 10; u = np.unique(b); nul = []
    for _ in range(2000):
        sg = dict(zip(u, rng.choice([-1, 1], len(u)))); sv = np.array([sg[x] for x in b])
        nul.append(max((D[md, j] * sv).mean() for j in range(len(XS))))
    q95 = np.percentile(nul, 95)
    L += [f'Cohort {len(F)} fills (thin {F.thin.sum()}); family-wise null 95th pct of the max DESIGN mean Δ = {q95:+.3f}', '',
          '| cell (thin only) | DESIGN mean Δ (95 % CI) | HOLDOUT mean Δ (CI) | Σ Δ thin (D / H) | halves Δ > 0 | top pair share | liquid control Δ (D / H) | thin − liquid (D / H) | pass |',
          '|---|---|---|---|---|---|---|---|---|']
    for j, x in enumerate(XS):
        r = {}
        for blk in ('DESIGN', 'HOLDOUT'):
            mt = (F.blk == blk).values & F.thin.values; ml = (F.blk == blk).values & ~F.thin.values
            r[blk] = (D[mt, j].mean(), ci(mt, D[:, j]), D[mt, j].sum(), D[ml, j].mean())
        hv = [D[(F.half == h).values & F.thin.values, j].mean() for h in sorted(F.half.unique())]
        mt = F.thin.values; ps = pd.Series(D[mt, j]).groupby(F.pair.values[mt]).sum(); top = ps.abs().max() / max(abs(ps.sum()), 1e-9)
        ok_ = (r['DESIGN'][1][0] > 0 and r['HOLDOUT'][0] > 0 and r['DESIGN'][0] > q95 and sum(h > 0 for h in hv) >= 5 and top <= 0.25
               and r['DESIGN'][0] > r['DESIGN'][3] and r['HOLDOUT'][0] > r['HOLDOUT'][3])
        L.append(f"| TP {x} % | {r['DESIGN'][0]:+.3f} ({r['DESIGN'][1][0]:+.2f} … {r['DESIGN'][1][1]:+.2f}) | {r['HOLDOUT'][0]:+.3f} ({r['HOLDOUT'][1][0]:+.2f} … {r['HOLDOUT'][1][1]:+.2f}) "
                 f"| {r['DESIGN'][2]:+.0f} / {r['HOLDOUT'][2]:+.0f} | {sum(h > 0 for h in hv)}/{len(hv)} | {top:.0%} | {r['DESIGN'][3]:+.3f} / {r['HOLDOUT'][3]:+.3f} "
                 f"| {r['DESIGN'][0]-r['DESIGN'][3]:+.3f} / {r['HOLDOUT'][0]-r['HOLDOUT'][3]:+.3f} | {'✓' if ok_ else '✗'} |")
    L += ['', '### Decomposition on thin fills (all periods): who gains, who pays', '',
          '| cell | Exit-B-stopped fills: saved (N · Σ) | stopped deeper (N · Σ) | Exit-B take-profit / time fills cut (N · Σ) | unchanged (N, Δ = 0 check) |', '|---|---|---|---|---|']
    mt = F.thin.values; ks = np.isin(k[:, 0], [1, 3])
    for j, x in enumerate(XS):
        d = D[:, j]
        sv = mt & ks & (d > 0.05); dp = mt & ks & (d < -0.05); ct = mt & ~ks & (np.abs(d) > 0.05); un = mt & (np.abs(d) <= 0.05)
        L.append(f'| TP {x} % | {sv.sum()} · {d[sv].sum():+.0f} | {dp.sum()} · {d[dp].sum():+.0f} | {ct.sum()} · {d[ct].sum():+.0f} | {un.sum()} (max |Δ| {np.abs(d[un]).max() if un.any() else 0:.3f}) |')
    L += ['', '### Thin fills per half-year: mean Δ per cell', '', '| half | N thin | ' + ' | '.join(f'TP {x}' for x in XS) + ' |', '|---|---|' + '---|' * len(XS)]
    for h in sorted(F.half.unique()):
        m = (F.half == h).values & mt
        L.append(f'| {h} | {m.sum()} | ' + ' | '.join(f'{D[m, j].mean():+.2f}' for j in range(len(XS))) + ' |')
    open(f'{R}/reports/DIST_THIN_TP_2026-11-19_tables.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

if __name__ == '__main__':
    main()
