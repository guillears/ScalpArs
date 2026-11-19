"""WILLY_HT W/L after the DL 594 dip cut (2026-11-19) — evaluation. Research only. Frozen design:
reports/WILLY_HT_WL594_2026-11-19_PREREG.md. Input: reports/WILLY_HT_WL594_2026-11-19_fills.csv (scripts/willy_ht_wl594_build.py),
reports/WILLY_HT_THEORY_2026-11-19_data/episodes_main.pkl (rebuild episodes, today's exit priced by scripts/willy_ht_theory_eval.py).

    venv/bin/python scripts/willy_ht_wl594_eval.py
"""
import os, sys, json, math, pickle, itertools
import numpy as np, pandas as pd


# ── stats helpers (no scipy in the venv) ──
def _hpmf(N, K, n, k):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def fisher_p(N, K, n, k):
    """two-sided Fisher exact p (sum of tables no more likely than the observed one)."""
    lo, hi = max(0, n - (N - K)), min(n, K)
    p0 = _hpmf(N, K, n, k)
    return min(1.0, sum(q for q in (_hpmf(N, K, n, j) for j in range(lo, hi + 1)) if q <= p0 * (1 + 1e-9)))


def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mw_p(a, b):
    """Mann-Whitney U two-sided p, normal approximation with tie correction."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    r = pd.Series(np.concatenate([a, b])).rank().values
    U = r[:n1].sum() - n1 * (n1 + 1) / 2
    _, t = np.unique(np.concatenate([a, b]), return_counts=True)
    N = n1 + n2
    sd = math.sqrt(n1 * n2 / 12 * ((N + 1) - (t ** 3 - t).sum() / (N * (N - 1))))
    return 2 * (1 - _ncdf(abs(U - n1 * n2 / 2) / sd)) if sd > 0 else np.nan


def spearman(x, y):
    d = pd.DataFrame(dict(x=x, y=y)).dropna()
    if len(d) < 5:
        return np.nan, np.nan
    rho = d.x.rank().corr(d.y.rank())
    z = rho * math.sqrt(len(d) - 1)
    return rho, 2 * (1 - _ncdf(abs(z)))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import willy_ht_theory_eval as TE                               # noqa: E402  (run_combo / today's exit pricer, research module)

OUTP = os.path.join(REPO, 'reports', 'WILLY_HT_WL594_2026-11-19')
DATA = os.path.join(REPO, 'reports', 'WILLY_HT_WL594_data')
X_SIZE = 0.61
DIP = -1.4
SEEN_FROM = pd.Timestamp('2026-07-01').value // 1_000_000
FLOOR_N, FLOOR_D = 8, 5
NSHUF = 2000
rng = np.random.default_rng(594)
MARKET = ('entry_btc', 'entry_bull_pct', 'entry_bear_pct', 'entry_global_volume_ratio', 'entry_mkt_', 'entry_alt_', 'entry_macro_trend',
          'entry_eth_', 'btc_1d_ema50_200_gap', 'entry_frenzy_gvol')   # entry_frenzy_hours = this pair's spike age (pair-level)
DERIVED = ['off_peak', 'spike_age_min', 'hour_utc', 'weekday', 'pair_repeat', 'btc_1d_ema50_200_gap', 'pre_ret15', 'pre_ret60',
           'pre_ret240', 'pre_dd60', 'pre_rvol60', 'pre_vol15_vs60']
DASH = {'AvgRSI': 'entry_rsi', 'AvgADX': 'entry_adx', 'ADXDir/ADXΔ': 'entry_adx_delta', 'RSIDir': 'entry_rsi − entry_rsi_prev (derived)',
        'AvgGap': 'entry_gap', 'Gap5-8': 'entry_ema_gap_5_8', 'Gap8-13': 'entry_ema_gap_8_13', 'Stretch': 'entry_ema5_stretch',
        'EMA20Slp': 'entry_ema20_slope', 'BTCSlp': 'entry_btc_ema20_slope', 'BTCRSI': 'entry_btc_rsi', 'BTCADX': 'entry_btc_adx',
        'BTCADXDir': 'entry_btc_adx − entry_btc_adx_prev (derived)', 'BTCRSIDir': 'entry_btc_rsi − entry_btc_rsi_prev (derived)',
        'RngPos': 'entry_range_position', 'Breadth': 'entry_bull_pct', 'ATR%': 'entry_atr_pct', 'BTCTrend': 'entry_btc_regime',
        'PairTrend': 'entry_macro_trend', 'GlobalVol': 'entry_global_volume_ratio', 'PairVol$': 'entry_pair_volume_24h_usd',
        'AvgExt%': 'entry_dist_from_ema13_pct', 'AvgBTCExt%': 'entry_btc_dist_from_ema13_pct', 'Avg1hSlope': 'entry_btc_1h_slope',
        'AltCorr': 'entry_alt_corr24', 'AltBeta': 'entry_alt_beta24', 'Up4h': 'entry_mkt_up4h_pct'}
EXCL = {'entry_price', 'entry_fee', 'entry_frenzy_spike_at', 'entry_strategy', 'entry_order_type', 'entry_btc_price',
        'entry_desired_notional', 'entry_willy_ht_size_frac', 'entry_liquidity_cap_notional', 'entry_bracket_cap_notional',
        'entry_bracket_max_leverage', 'entry_frenzy_vwap', 'entry_frenzy_willy_spike_high'}   # price levels / sizing, not conditions


def is_market(c):
    return any(c.startswith(p) for p in MARKET)


def wilson_lo(p, n, z=1.645):
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n) if n else np.nan


def one_size(T, day_col='xday'):
    """one-size DCR (% / day) over every calendar day from the first entry day to the last exit day; worst day (account %)."""
    if not len(T):
        return np.nan, np.nan
    d0, d1 = min(T.day.min(), T[day_col].min()), max(T.day.max(), T[day_col].max())
    days = pd.date_range(d0, d1, freq='D').strftime('%Y-%m-%d')
    r = T.groupby(day_col).net.sum() * X_SIZE / 100
    v = np.array([1 + r.get(d, 0.0) for d in days])
    return (np.prod(v) ** (1 / len(v)) - 1) * 100, r.min() * 100


def summ(T, label):
    if not len(T):
        return dict(set=label, N=0)
    w, l = T.net[T.net > 0], T.net[T.net <= 0]
    aw, al = (w.mean() if len(w) else 0), (l.mean() if len(l) else 0)
    wr = (T.net > 0).mean(); nd = T.day.nunique()
    b = aw / abs(al) if al < 0 else np.inf
    kf = lambda p: p - (1 - p) / b if b != np.inf else p
    dcr, worst = one_size(T)
    span = (pd.Timestamp(T.xday.max()) - pd.Timestamp(T.day.min())).days + 1
    return dict(set=label, N=len(T), days=nd, span_d=span, per_day=len(T) / span, WR=100 * wr, avg=T.net.mean(), sum=T.net.sum(),
                stops=int((T.net < -3).sum()), be_WR=100 * abs(al) / (aw + abs(al)) if aw + abs(al) else np.nan,
                wilson_fills=100 * wilson_lo(wr, len(T)), kelly_fills=kf(wilson_lo(wr, len(T))),
                wilson_days=100 * wilson_lo(wr, nd), kelly_days=kf(wilson_lo(wr, nd)), DCR=dcr, worst_day=worst,
                sum_per_day=T.net.sum() / span)


def day_boot_avg(T, B=4000):
    g = T.groupby('day').net.agg(['sum', 'count']).values
    idx = rng.integers(0, len(g), (B, len(g)))
    m = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return np.percentile(m, [2.5, 97.5]), float((m >= 0).mean())


# ───────────── split construction (label-independent) ─────────────
def build_splits(A, cands, levels):
    S = []   # (variable, split, mask)
    for c in cands:
        v = A[c].astype(float); ok = v.notna()
        if (v[ok] < 0).any() and (v[ok] > 0).any():
            S.append((c, '> 0', ok & (v > 0), ok))
        med = v[ok].median()
        S.append((c, f'> median {med:.4g}', ok & (v > med), ok))
        q1, q2 = v[ok].quantile([1 / 3, 2 / 3])
        tm = ok & ((v <= q1) | (v >= q2))
        S.append((c, f'top third ≥ {q2:.4g} vs bottom ≤ {q1:.4g}', tm & (v >= q2), tm))
    for c in levels:
        for lv in A[c].dropna().unique():
            ok = A[c].notna()
            S.append((c, f'== {lv}', ok & (A[c] == lv), ok))
    keep = []
    for var, sp, m, ok in S:
        a, b = m, ok & ~m
        if a.sum() >= FLOOR_N and b.sum() >= FLOOR_N and A.day[a].nunique() >= FLOOR_D and A.day[b].nunique() >= FLOOR_D:
            keep.append((var, sp, m.values, ok.values))
    return keep


def pvals(splits, Y):
    """Y: (n,) or (n, S) 0/1 labels. Fisher two-sided p per split (vectorised via the hypergeometric pmf on the universe = ok rows)."""
    Y = Y.reshape(len(Y), -1).astype(np.float64)
    out = np.empty((len(splits), Y.shape[1]))
    for i, (_, _, m, ok) in enumerate(splits):
        N = ok.sum(); n_in = m.sum()
        K = Y[ok].sum(0); k = Y[m].sum(0)
        for j, (Kj, kj) in enumerate(zip(K, k)):
            out[i, j] = _fp(int(N), int(Kj), int(n_in), int(kj))
    return out


_FP = {}
def _fp(N, K, n, k):
    key = (N, K, n, k)
    if key not in _FP:
        _FP[key] = fisher_p(N, K, n, k)
    return _FP[key]


def main():
    F = pd.read_csv(OUTP + '_fills.csv', low_memory=False)
    F['xday'] = pd.to_datetime(F.entry_ms + F.re_minutes * 60_000, unit='ms').dt.strftime('%Y-%m-%d')
    for a, b, n in (('entry_rsi', 'entry_rsi_prev', 'rsi_dir'), ('entry_btc_adx', 'entry_btc_adx_prev', 'btc_adx_dir'),
                    ('entry_btc_rsi', 'entry_btc_rsi_prev', 'btc_rsi_dir')):
        F[n] = pd.to_numeric(F[a], errors='coerce') - pd.to_numeric(F[b], errors='coerce')
    M = F[F.set == 'master']
    A = M[M.off_peak <= DIP].reset_index(drop=True)
    B = A[A.failed_spike_kept].reset_index(drop=True)
    C = F[(F.set == 'current') & (F.off_peak <= DIP)]
    Call = F[F.set == 'current']
    out = {}
    # ── coverage ──
    ent = [c for c in A.columns if c.startswith('entry_') and c not in EXCL]
    for c in ent:
        if not pd.api.types.is_numeric_dtype(A[c]):
            n = pd.to_numeric(A[c], errors='coerce')
            if n.notna().sum() >= 0.8 * A[c].notna().sum() and n.notna().any():
                A[c] = n; B[c] = pd.to_numeric(B[c], errors='coerce')
    cov = pd.DataFrame(dict(col=ent + DERIVED + ['rsi_dir', 'btc_adx_dir', 'btc_rsi_dir'])).drop_duplicates('col')
    cov['coverage'] = [A[c].notna().mean() * 100 for c in cov.col]
    cov['distinct'] = [A[c].nunique() for c in cov.col]
    cov['kind'] = ['numeric' if (pd.api.types.is_numeric_dtype(A[c]) and A[c].nunique() > 3) else ('level' if 2 <= A[c].nunique() <= 6 else ('constant' if A[c].nunique() <= 1 else 'text>6')) for c in cov.col]
    cov['tested'] = (cov.coverage >= 80) & cov.kind.isin(['numeric', 'level'])
    cov['market_wide'] = [is_market(c) for c in cov.col]
    cov.to_csv(OUTP + '_coverage.csv', index=False)
    cands = cov[cov.tested & (cov.kind == 'numeric')].col.tolist()
    levels = cov[cov.tested & (cov.kind == 'level')].col.tolist()
    # ── W vs L medians (every tested numeric var), A and B ──
    rows = []
    for c in cands:
        v = A[c].astype(float)
        w, l = v[A.win].dropna(), v[~A.win].dropna()
        p = mw_p(w, l) if len(w) > 2 and len(l) > 2 else np.nan
        wk, nk = v[A.worked].dropna(), v[~A.worked].dropna()
        pw = mw_p(wk, nk) if len(wk) > 2 and len(nk) > 2 else np.nan
        rr = spearman(v.values, A.reach.values)
        vb = B[c].astype(float)
        lr = (vb.rank(pct=True)[~B.win]).round(2).tolist()
        rows.append(dict(var=c, market=is_market(c), N=int(v.notna().sum()), med_W=w.median(), med_L=l.median(), MW_p_win=p,
                         med_WORKED=wk.median(), med_NEVER=nk.median(), MW_p_worked=pw, spearman_reach=rr[0], sp_p=rr[1],
                         B_med_W=vb[B.win].median(), B_losers_pct_rank=lr))
    WL = pd.DataFrame(rows).sort_values('MW_p_win'); WL.to_csv(OUTP + '_wl_table.csv', index=False)
    # ── splits + null ──
    splits = build_splits(A, cands, levels)
    res_rows = []
    for tgt in ('win', 'worked'):
        y = A[tgt].values.astype(int)
        P = pvals(splits, y)[:, 0]
        obs_n = int((P < 0.05).sum()); obs_min = float(P.min())
        # null 1: fill-level shuffle · null 2: circular shift of the time-ordered labels (keeps same-day clustering)
        Ys = np.stack([rng.permutation(y) for _ in range(NSHUF)], 1)
        Pn = pvals(splits, Ys)
        sh = [np.roll(y, k)[:] if k else None for k in range(1, len(y))]
        Yc = np.stack([np.roll(y, k) for k in range(1, len(y))] + [np.roll(y[::-1], k) for k in range(len(y))], 1)
        Pc = pvals(splits, Yc)
        out[tgt] = dict(n_splits=len(splits), n_vars=len(set(s[0] for s in splits)), obs_nominal=obs_n, obs_min_p=obs_min,
                        null_nominal_med=float(np.median((Pn < 0.05).sum(0))), null_nominal_p95=float(np.percentile((Pn < 0.05).sum(0), 95)),
                        P_null_count_ge=float(((Pn < 0.05).sum(0) >= obs_n).mean()), P_null_minp_le=float((Pn.min(0) <= obs_min).mean()),
                        circ_P_count_ge=float(((Pc < 0.05).sum(0) >= obs_n).mean()), circ_P_minp_le=float((Pc.min(0) <= obs_min).mean()))
        for i, (var, sp, m, ok) in enumerate(splits):
            a, b = A[m], A[ok & ~m]
            res_rows.append(dict(target=tgt, var=var, split=sp, market=is_market(var), N_in=len(a), days_in=a.day.nunique(), WR_in=100 * a[tgt].mean(),
                                 avg_in=a.net.mean(), N_out=len(b), days_out=b.day.nunique(), WR_out=100 * b[tgt].mean(), avg_out=b.net.mean(),
                                 p=P[i], fw_p=float((Pn.min(0) <= P[i]).mean()),
                                 top_pair_share_in=a.pair.value_counts(normalize=True).iloc[0] * 100,
                                 top_day_share_in=a.day.value_counts(normalize=True).iloc[0] * 100))
    SP = pd.DataFrame(res_rows).sort_values(['target', 'p']); SP.to_csv(OUTP + '_splits_1d.csv', index=False)
    # ── 2D: median quadrants of every pair of tested numeric vars, quadrant vs rest, same floor ──
    q2 = []
    meds = {c: A[c].astype(float).median() for c in cands}
    for c1, c2 in itertools.combinations(cands, 2):
        v1, v2 = A[c1].astype(float), A[c2].astype(float)
        ok = (v1.notna() & v2.notna()).values
        for s1, s2 in itertools.product((True, False), repeat=2):
            m = ok & (((v1 > meds[c1]) == s1) & ((v2 > meds[c2]) == s2)).values
            if m.sum() >= FLOOR_N and (ok & ~m).sum() >= FLOOR_N and A.day[m].nunique() >= FLOOR_D and A.day[ok & ~m].nunique() >= FLOOR_D:
                q2.append((f'{c1} × {c2}', f"{'hi' if s1 else 'lo'}×{'hi' if s2 else 'lo'}", m, ok))
    y = A.win.values.astype(int)
    P2 = pvals(q2, y)[:, 0] if q2 else np.array([])
    Ys = np.stack([rng.permutation(y) for _ in range(400)], 1)
    P2n = pvals(q2, Ys) if q2 else np.zeros((0, 400))
    out['2d_win'] = dict(n_cells=len(q2), obs_nominal=int((P2 < 0.05).sum()), obs_min_p=float(P2.min()) if len(P2) else np.nan,
                         null_nominal_med=float(np.median((P2n < 0.05).sum(0))), null_nominal_p95=float(np.percentile((P2n < 0.05).sum(0), 95)),
                         P_null_count_ge=float(((P2n < 0.05).sum(0) >= (P2 < 0.05).sum()).mean()),
                         P_null_minp_le=float((P2n.min(0) <= P2.min()).mean()) if len(P2) else np.nan)
    t2 = []
    for i in np.argsort(P2)[:15]:
        var, sp, m, ok = q2[i]
        a, b = A[m], A[ok & ~m]
        t2.append(dict(cell=var, quad=sp, N_in=len(a), days_in=a.day.nunique(), WR_in=100 * a.win.mean(), avg_in=a.net.mean(),
                       N_out=len(b), WR_out=100 * b.win.mean(), avg_out=b.net.mean(), p=P2[i], fw_p=float((P2n.min(0) <= P2[i]).mean())))
    pd.DataFrame(t2).to_csv(OUTP + '_splits_2d_top.csv', index=False)
    # ── backtest cohorts (rebuild episodes, today's exit) ──
    E = pickle.load(open(os.path.join(REPO, 'reports', 'WILLY_HT_THEORY_2026-11-19_data', 'episodes_main.pkl'), 'rb'))
    X, paths = E['X'], E['paths']
    dipm = (X.off_peak <= DIP).values
    bt = {}
    for nm, kw in (('FS on + dip', dict(fs_cut=0.0, keep_mask=dipm)), ('FS off + dip', dict(fs_cut=None, keep_mask=dipm)),
                   ('FS on, no dip', dict(fs_cut=0.0)), ('FS off, no dip', dict(fs_cut=None))):
        T = TE.run_combo(X, paths, 'E0_RED', TE.TODAY[1], **kw)
        T = T.merge(X.drop(columns=['pair'], errors='ignore'), left_on='idx', right_index=True, suffixes=('', '_ep'))
        T['fs_kept'] = ~(T.vs_vwap.fillna(1) <= 0)
        bt[nm] = T
    bt['FS off + dip'].to_csv(OUTP + '_backtest_fsoff_dip_trades.csv', index=False)
    sc = []
    for nm, T in bt.items():
        for per, msk in (('DESIGN <2026-07', T.entry_ms < SEEN_FROM), ('SEEN 2026-H2', T.entry_ms >= SEEN_FROM),
                         ('2025-H1', T.half == '2025-H1'), ('2025-H2', T.half == '2025-H2'), ('2026-H1', T.half == '2026-H1'),
                         ('2026-H2', T.half == '2026-H2'), ('ALL', T.entry_ms > 0)):
            s = summ(T[msk], nm); s['period'] = per; sc.append(s)
    for nm in ('FS off + dip',):   # the FS-blocked share of the FS-off portfolio (as it trades inside that portfolio)
        T = bt[nm]
        for per, msk in (('DESIGN <2026-07', T.entry_ms < SEEN_FROM), ('SEEN 2026-H2', T.entry_ms >= SEEN_FROM), ('ALL', T.entry_ms > 0)):
            for k, lab in ((True, 'FS off + dip: FS-kept part'), (False, 'FS off + dip: FS-blocked part')):
                s = summ(T[msk & (T.fs_kept == k)], lab); s['period'] = per; sc.append(s)
    SC = pd.DataFrame(sc); SC.to_csv(OUTP + '_scenario_backtest.csv', index=False)
    # master scenario
    ms_rows = [summ(A, 'master: dip cut only (44)'), summ(B, 'master: dip + FAILED_SPIKE (12)'), summ(A[~A.failed_spike_kept], 'master: FS-blocked part (32)'),
               summ(M, 'master: all HT (49, no dip)'), summ(C, 'current batch, dip-cut'), summ(Call, 'current batch, all')]
    MS = pd.DataFrame(ms_rows); MS.to_csv(OUTP + '_scenario_master.csv', index=False)
    blk = A[~A.failed_spike_kept]
    ci, pge = day_boot_avg(blk)
    out['fs_block_bar'] = dict(N=len(blk), windows=int(blk.day.nunique()), WR=100 * blk.win.mean(),
                               be_WR=100 * 4.1 / 9.1, avg=blk.net.mean(), ci=list(ci), P_boot_avg_ge0=pge,
                               top_pair_loss_share=float(blk[blk.net < 0].groupby('pair').net.sum().min() / blk[blk.net < 0].net.sum() * 100))
    # LOO on the 44 vs 12 difference in sum and avg (coins / days)
    loo = []
    for key in ('pair', 'day'):
        for g in A[key].unique():
            a2 = A[A[key] != g]; b2 = a2[a2.failed_spike_kept]
            loo.append(dict(by=key, drop=g, sum44=a2.net.sum(), sum12=b2.net.sum(), avg44=a2.net.mean(), avg12=b2.net.mean() if len(b2) else np.nan))
    L = pd.DataFrame(loo); L.to_csv(OUTP + '_loo.csv', index=False)
    out['loo'] = {k: dict(min_sum44=float(g.sum44.min()), max_sum44=float(g.sum44.max()), min_sum12=float(g.sum12.min()), max_sum12=float(g.sum12.max()),
                          n_sum12_gt_sum44=int((g.sum12 > g.sum44).sum()), n=len(g)) for k, g in L.groupby('by')}
    # ── backtest cross-check of the top master splits (mapped variables) ──
    MAP = {'off_peak': 'off_peak', 'entry_frenzy_vs_vwap_pct': 'vs_vwap', 'entry_frenzy_vol_mult': 'vol_mult', 'entry_atr_pct': 'atr',
           'entry_adx': 'adx', 'entry_ema_gap_8_13': 'gap813', 'entry_pair_volume_24h_usd': 'vol24', 'btc_1d_ema50_200_gap': 'btc_1d_ema50_200',
           'entry_btc_1d_ret_pct': 'btc_1d_ret', 'entry_btc_1h_slope': 'btc_1h_slope', 'entry_frenzy_bar_ret_pct': 'bar_ret',
           'entry_frenzy_willy_wait_bars': 'wait_bars', 'spike_age_min': 'spike_age_min', 'hour_utc': 'hour_utc', 'entry_bull_pct': 'breadth'}
    T = bt['FS off + dip'].copy()
    T['spike_age_min'] = (T.entry_ms - T.spike_ms) / 60_000; T['hour_utc'] = pd.to_datetime(T.entry_ms, unit='ms').dt.hour
    T['win'] = T.net > 0
    T['spike_age_h'] = T.spike_age_min / 60
    b5 = os.path.join(DATA, 'btc_5m.csv')
    if os.path.exists(b5):                     # BTC 5m EMA13 − EMA50 gap % at the last CLOSED 5m bar (the engine's trend-gap stamp)
        bb = pd.read_csv(b5).sort_values('t')
        bb['gap'] = (bb.c.ewm(span=13, adjust=False).mean() / bb.c.ewm(span=50, adjust=False).mean() - 1) * 100
        def at(tms):
            i = np.searchsorted(bb.t.values + 300_000, np.asarray(tms), 'right') - 1
            return np.where(i >= 0, bb.gap.values[np.clip(i, 0, None)], np.nan)
        T['btc5m_gap'] = at(T.entry_ms.values)
        chk = pd.DataFrame(dict(stamp=A.entry_btc_trend_gap_pct.astype(float), rebuilt=at(A.entry_ms.values)))
        out['btc5m_gap_rebuild_check'] = dict(spearman=float(chk.stamp.rank().corr(chk.rebuilt.rank())),
                                              sign_agree=float((np.sign(chk.stamp) == np.sign(chk.rebuilt)).mean()),
                                              med_abs_diff=float((chk.stamp - chk.rebuilt).abs().median()))
        MAP['entry_btc_trend_gap_pct'] = 'btc5m_gap'
    MAP['entry_frenzy_hours'] = 'spike_age_h'
    xc = []
    top = SP[(SP.target == 'win')].sort_values('p').drop_duplicates('var').head(12)
    for _, r in top.iterrows():
        bv = MAP.get(r['var'])
        row = dict(var=r['var'], split=r['split'], master_WR_in=r.WR_in, master_WR_out=r.WR_out, master_p=r.p, fw_p=r.fw_p, backtest_col=bv or '—')
        if bv and bv in T:
            sp = r['split']
            if sp.startswith('> 0'):
                thr = 0.0; hi = T[bv] > 0; ok = T[bv].notna()
            elif sp.startswith('> median'):
                thr = float(sp.split()[-1]); hi = T[bv] > thr; ok = T[bv].notna()
            else:
                q2v = float(sp.split('≥ ')[1].split()[0]); q1v = float(sp.split('≤ ')[1])
                ok = T[bv].notna() & ((T[bv] <= q1v) | (T[bv] >= q2v)); hi = T[bv] >= q2v
            if bv in ('btc_1d_ema50_200',) and T[bv].abs().median() < 0.5 < abs(A[r['var']].median()):
                row['unit_note'] = 'check units'
            sgn = np.sign(r.WR_in - r.WR_out)
            for per, pm in (('DESIGN', T.entry_ms < SEEN_FROM), ('SEEN', T.entry_ms >= SEEN_FROM)):
                a, b = T[pm & ok & hi], T[pm & ok & ~hi]
                row[f'{per} in'] = f"{len(a)} · {100 * a.win.mean():.0f}% · {a.net.mean():+.2f}" if len(a) else '0'
                row[f'{per} out'] = f"{len(b)} · {100 * b.win.mean():.0f}% · {b.net.mean():+.2f}" if len(b) else '0'
                row[f'{per} same_dir'] = bool(len(a) and len(b) and np.sign(a.win.mean() - b.win.mean()) == sgn)
        xc.append(row)
    XC = pd.DataFrame(xc); XC.to_csv(OUTP + '_backtest_xcheck.csv', index=False)
    json.dump(out, open(OUTP + '_summary.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(json.dumps(out, indent=1, default=float))
    print(MS.round(2).to_string()); print(SC.round(2).to_string())
    print(SP[SP.target == 'win'].head(15).round(3).to_string())
    print(SP[SP.target == 'worked'].head(8).round(3).to_string())
    print(pd.DataFrame(t2).round(3).to_string()); print(XC.to_string())
    print(cov.tested.sum(), 'tested of', len(cov)); print(cov[~cov.tested & (cov.coverage > 0)].to_string())


if __name__ == '__main__':
    main()
