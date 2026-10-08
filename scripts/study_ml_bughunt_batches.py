#!/usr/bin/env python3
"""ML BUG HUNT (2026-10-08) part 6: BATCH BY BATCH over the exact live batch windows — master kept momentum longs (STACK 2026-10-08c,
stack_keep, non-probe, ex-B1, pct = stack_pct) vs the yr5 replay's momentum-long fills on the same hours (3 seeds, warm-up trimmed),
with today's two ML entry rules that post-date the yr5 code snapshot applied post-hoc to the replay (LONG_CHOP_BURST: BTC eff72 ≤ 0.007
∧ another non-probe bot fill of the same seed opened 0…120 s earlier; LONG_HEAT 3-leg re-scope can only ADMIT more than yr5's
bull ≥ 85 leg, which the replay cannot add — counted from the master side instead).

Matching (as the recall trace): same pair, LONG, opened within ±10 min, nearest, per seed. Shared = in both; live-only = master kept
row with no yr5 fill in that seed; replay-only = yr5 fill with no master kept row (split: master has the fill but REMOVED it /
live traded nothing on the pair ±10 min).

Out: reports/study_ml_bughunt_batches.csv (one row per batch) + reports/study_ml_bughunt_batches_pairs.csv (every live/replay row with
its class) and a printed table.
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REP = os.path.join(ROOT, 'reports')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import yr5_fills_trimmed as YT                        # the canonical trimmed loader (it only reads the run CSVs)

TOL = pd.Timedelta(minutes=10)
CHOP_EFF, CHOP_WIN = 0.007, 120.0                    # trading_config.json long_chop_burst_eff72_max / _window_s (today)


def eras():
    """live-up windows (reports/study_ml_trace_liveup.csv) labelled with the master era whose rows fall inside."""
    up = pd.read_csv(os.path.join(REP, 'study_ml_trace_liveup.csv'), parse_dates=['up_from', 'up_to'])
    m = pd.read_csv(os.path.join(REP, 'MASTER_POOL_stacked.csv'), low_memory=False, usecols=['era', 'opened_at'])
    m['t'] = pd.to_datetime(m.opened_at)
    lab = []
    for r in up.itertuples():
        x = m[(m.t >= r.up_from - pd.Timedelta(minutes=1)) & (m.t <= r.up_to + pd.Timedelta(minutes=1))].era.value_counts()
        lab.append('+'.join(sorted(x.index, key=lambda e: (len(e), e))) if len(x) else '?')
    up['era'] = lab
    return up


def main():
    W = eras()
    m = pd.read_csv(os.path.join(REP, 'MASTER_POOL_stacked.csv'), low_memory=False)
    m['t'] = pd.to_datetime(m.opened_at)
    mlall = m[(m.entry_strategy == 'MOMENTUM') & (m.direction == 'LONG') & (~m.is_probe.astype(bool))].copy()
    mlall['pct_live'] = pd.to_numeric(mlall.pnl_percentage, errors='coerce')
    K = mlall[mlall.stack_keep.astype(bool) & (mlall.era != 'B1')].copy()
    K['pct'] = K.stack_pct.astype(float)

    F = YT.load()                                    # every sleeve, trimmed, closed, probes + MANUAL out
    F['t'] = pd.to_datetime(F.t)
    ML = F[F.sleeve == 'MOM-long'].copy()
    # chop-burst post-hoc (today's rule, not in the yr5 snapshot): other non-probe bot fills of the same seed opened 0…120 s before
    chop = np.zeros(len(ML), bool)
    for s, g in ML.groupby('seed'):
        allt = np.sort(F[F.seed == s].t.values.astype('datetime64[ms]').astype(np.int64))
        for i, (ix, r) in enumerate(g.iterrows()):
            tm = np.datetime64(r.t, 'ms').astype(np.int64)
            a = np.searchsorted(allt, tm - int(CHOP_WIN * 1000)); b = np.searchsorted(allt, tm)
            prior = b > a                              # strictly earlier fills inside the window (the fill itself is at tm, excluded)
            eff = pd.to_numeric(r.entry_btc_eff72, errors='coerce')
            chop[ML.index.get_loc(ix)] = bool(prior and pd.notna(eff) and eff <= CHOP_EFF)
    ML['chop_block'] = chop
    ML['in_window'] = False; ML['era'] = None
    K['era_w'] = None
    for r in W.itertuples():
        sel = (ML.t >= r.up_from) & (ML.t <= r.up_to)
        ML.loc[sel, 'in_window'] = True; ML.loc[sel, 'era'] = r.era
    MLw = ML[ML.in_window & ~ML.chop_block].copy()

    # ── matching, per seed ──
    rows = []
    lk = K.copy()
    for s in sorted(ML.seed.unique()):
        R = MLw[MLw.seed == s]
        used = set()
        for ix, r in lk.iterrows():
            c = R[(R.pair == r.pair) & ((R.t - r.t).abs() <= TOL)]
            c = c[~c.index.isin(used)]
            if len(c):
                j = (c.t - r.t).abs().idxmin(); used.add(j)
                rows.append(dict(seed=s, era=r.era, cls='shared', pair=r.pair, t_live=r.t, t_rep=R.loc[j, 't'], live=r.pct, rep=R.loc[j, 'pct'],
                                 rep_reason=R.loc[j, 'close_reason'], live_reason=r.close_reason))
            else:
                rows.append(dict(seed=s, era=r.era, cls='live_only', pair=r.pair, t_live=r.t, live=r.pct, live_reason=r.close_reason))
        for j, q in R[~R.index.isin(used)].iterrows():
            near = mlall[(mlall.pair == q.pair) & ((mlall.t - q.t).abs() <= TOL)]
            sub = ('replay_only:master_removed' if len(near) and not near.stack_keep.astype(bool).any() else
                   'replay_only:live_B1_or_kept_elsewhere' if len(near) else 'replay_only:live_none')
            rows.append(dict(seed=s, era=q.era, cls=sub, pair=q.pair, t_rep=q.t, rep=q.pct, rep_reason=q.close_reason,
                             live_removed_pct=(near.pct_live.mean() if len(near) else np.nan),
                             live_block=(near.stack_block_reason.iloc[0] if len(near) else None)))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(REP, 'study_ml_bughunt_batches_pairs.csv'), index=False)

    # ── per batch table ──
    order = list(dict.fromkeys(W.era))
    out = []
    for e in order:
        if e == 'B1' or e.startswith('B17') or e.startswith('B18') or e == '?':
            continue
        eras_in = e.split('+')
        k = K[K.era.isin(eras_in)]
        y = MLw[MLw.era == e]
        yc = ML[ML.in_window & (ML.era == e) & ML.chop_block]
        p = P[P.era.isin(eras_in) | (P.era == e)]
        sh = p[p.cls == 'shared']; lo = p[p.cls == 'live_only']; ro = p[p.cls.str.startswith('replay_only')]
        rm = p[p.cls == 'replay_only:master_removed']; rn = p[p.cls == 'replay_only:live_none']
        out.append(dict(
            batch=e,
            m_n=len(k), m_wr=(k.pct > 0).mean() if len(k) else np.nan, m_avg=k.pct.mean() if len(k) else np.nan,
            r_n_seed=len(y) / 3, r_wr=(y.pct > 0).mean() if len(y) else np.nan, r_avg=y.pct.mean() if len(y) else np.nan,
            r_chop_removed_seed=len(yc) / 3, r_chop_avg=yc.pct.mean() if len(yc) else np.nan,
            shared_seed=len(sh) / 3, shared_live=sh.live.mean() if len(sh) else np.nan, shared_rep=sh.rep.mean() if len(sh) else np.nan,
            liveonly_seed=len(lo) / 3, liveonly_live=lo.live.mean() if len(lo) else np.nan,
            reponly_seed=len(ro) / 3, reponly_avg=ro.rep.mean() if len(ro) else np.nan,
            rep_masterremoved_seed=len(rm) / 3, rep_masterremoved_avg=rm.rep.mean() if len(rm) else np.nan,
            rep_livenone_seed=len(rn) / 3, rep_livenone_avg=rn.rep.mean() if len(rn) else np.nan))
    T = pd.DataFrame(out)
    tot = dict(batch='ALL (ex-B1, ≤ Oct-3)')
    k = K[K.era.isin(sum([e.split('+') for e in T.batch], []))]
    tot.update(m_n=len(k), m_wr=(k.pct > 0).mean(), m_avg=k.pct.mean())
    y = MLw[MLw.era.isin(T.batch)]
    tot.update(r_n_seed=len(y) / 3, r_wr=(y.pct > 0).mean(), r_avg=y.pct.mean())
    pp = P[P.era.isin(sum([e.split('+') for e in T.batch], []) + list(T.batch))]
    for c, nm in (('shared', 'shared'), ('live_only', 'liveonly')):
        q = pp[pp.cls == c]; tot[f'{nm}_seed'] = len(q) / 3
        if c == 'shared':
            tot['shared_live'] = q.live.mean(); tot['shared_rep'] = q.rep.mean()
        else:
            tot['liveonly_live'] = q.live.mean()
    q = pp[pp.cls.str.startswith('replay_only')]; tot['reponly_seed'] = len(q) / 3; tot['reponly_avg'] = q.rep.mean()
    q = pp[pp.cls == 'replay_only:master_removed']; tot['rep_masterremoved_seed'] = len(q) / 3; tot['rep_masterremoved_avg'] = q.rep.mean()
    q = pp[pp.cls == 'replay_only:live_none']; tot['rep_livenone_seed'] = len(q) / 3; tot['rep_livenone_avg'] = q.rep.mean()
    yc = ML[ML.in_window & ML.era.isin(T.batch) & ML.chop_block]; tot['r_chop_removed_seed'] = len(yc) / 3; tot['r_chop_avg'] = yc.pct.mean()
    T = pd.concat([T, pd.DataFrame([tot])], ignore_index=True)
    T.to_csv(os.path.join(REP, 'study_ml_bughunt_batches.csv'), index=False)
    pd.set_option('display.width', 300); pd.set_option('display.max_columns', 40)
    print(T.round(3).to_string())
    print('chop-burst post-hoc removed (all yr5 ML, year):', ML.chop_block.sum(), 'avg', ML[ML.chop_block].pct.mean().round(3),
          '| year ML after chop', ML[~ML.chop_block].pct.mean().round(4), 'N/seed', round((~ML.chop_block).sum() / 3, 1))


if __name__ == '__main__':
    main()
