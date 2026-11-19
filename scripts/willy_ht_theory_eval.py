"""WILLY_HT THEORY (2026-11-19) — evaluation: entry × exit grid, family-wise random-entry null, FAILED_SPIKE / turnover cuts,
regime at sign granularity in day units, path labels, master re-price, event study. Frozen design:
reports/WILLY_HT_THEORY_2026-11-19_PREREG.md. Input: reports/WILLY_HT_THEORY_2026-11-19_data/episodes_{main,D}.pkl
(scripts/willy_ht_theory_build.py) + reports/MASTER_POOL_stacked.csv + reports/WILLY_BACKTEST_WHY_2026-11-18_fills.csv.

    venv/bin/python scripts/willy_ht_theory_eval.py   (only willy_ht_theory_build.py needs WHT_REBUILD_DIR)
"""
import os, sys, pickle, math, json
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'reports', 'WILLY_HT_THEORY_2026-11-19_data')
OUTP = os.path.join(REPO, 'reports', 'WILLY_HT_THEORY_2026-11-19')
MIN = 60_000; BAR = 300_000; DAY = 86_400_000
TAKER, MAKER, SLIP = 0.05, 0.02, 0.10
X_SIZE = 0.61
DESIGN_END = pd.Timestamp('2026-01-01').value // 1_000_000
ENTRIES = ['E0_RED', 'E1_CONFIRM', 'E2_DIP3', 'E3_VWAP', 'E4_BREAKOUT']
EXITS = [(tp, sl, hd) for tp in (3, 5, 8) for sl in (3, 4, 6) for hd in (180, 1440)]
TODAY = ('E0_RED', (5, 4, 1440))
rng = np.random.default_rng(20261119)


def half(ms):
    d = pd.Timestamp(int(ms), unit='ms'); return f"{d.year}-H{1 if d.month <= 6 else 2}"


# ───────── entries ─────────
def five_min_close_idx(P):
    """indices of bars that CLOSE a 5m bar (1m path: bar whose t+1m is on a 5m boundary; 5m path: every bar)."""
    step = P[1, 0] - P[0, 0] if len(P) > 1 else BAR
    if step >= BAR:
        return np.arange(len(P))
    return np.flatnonzero(((P[:, 0] + MIN) % BAR) == 0)


def entry(name, ep, P):
    """→ (start index, E, entry fee %, tp_blocked_on_first_bar) or None. P starts at the decisive pass time."""
    t0 = ep['pass_ms']
    if name == 'E0_RED':
        return 0, ep['live'] * (1 + SLIP / 100), TAKER, False
    if name == 'E1_CONFIRM':
        ci = five_min_close_idx(P)
        ci = ci[P[ci, 0] < t0 + 60 * MIN]
        hit = ci[P[ci, 4] > ep['red_hi']]
        if not len(hit) or hit[0] + 1 >= len(P):
            return None
        s = hit[0] + 1
        return s, P[s, 1] * (1 + SLIP / 100), TAKER, False
    if name in ('E2_DIP3', 'E3_VWAP'):
        bid = ep['close'] * 0.97 if name == 'E2_DIP3' else ep['vwap']
        if not (bid == bid) or bid >= ep['close']:
            return None
        w = np.flatnonzero((P[:, 0] < t0 + 120 * MIN) & (P[:, 3] <= bid * 0.9999))
        if not len(w):
            return None
        s = w[0]
        return s, min(bid, P[s, 1]), MAKER, True
    if name == 'E4_BREAKOUT':
        lvl = ep['peak'] * 1.001
        w = np.flatnonzero((P[:, 0] < t0 + 240 * MIN) & (P[:, 2] >= lvl))
        if not len(w):
            return None
        s = w[0]
        return s, max(lvl, P[s, 1]) * (1 + SLIP / 100), TAKER, True
    raise ValueError(name)


# ───────── exit ─────────
def price(P, fund, s, E, fe, tp, sl, hold, tp_block_first):
    """NET-% exit as live: resting maker TP at net +tp (traded through by 1e-4), taker stop at net −sl (gap-aware, slip), time
    exit at market. Stop before TP inside a bar. → net, exit_ms, reason, worked, reach_before_stop, reach_tp_before_stop."""
    t_in = P[s, 0]; amb = False
    e = np.searchsorted(P[:, 0], t_in + hold * MIN, 'left')
    seg = P[s:e]
    if not len(seg):
        return None
    x_tp = E * (1 + (tp + fe) / 100) / (1 - MAKER / 100)
    p_sl = E * (100 - sl + fe) / (100 - TAKER)
    lo, hi = seg[:, 3], seg[:, 2].copy()
    if tp_block_first:
        hi[0] = -np.inf
    sl_hit = lo <= p_sl
    tp_hit = hi >= x_tp * 1.0001
    i_sl = int(np.argmax(sl_hit)) if sl_hit.any() else 10 ** 9
    i_tp = int(np.argmax(tp_hit)) if tp_hit.any() else 10 ** 9
    if i_sl == 10 ** 9 and i_tp == 10 ** 9:
        if e < len(P):
            x = P[e, 1] * (1 - SLIP / 100); t_out = P[e, 0]; rsn = 'TIME'
        else:
            x = seg[-1, 4] * (1 - SLIP / 100); t_out = seg[-1, 0]; rsn = 'END'
        net = (x / E - 1) * 100 - fe - TAKER * x / E
    elif i_sl <= i_tp:
        x = min(p_sl, seg[i_sl, 1]) * (1 - SLIP / 100); t_out = seg[i_sl, 0]; rsn = 'SL'
        net = (x / E - 1) * 100 - fe - TAKER * x / E
        amb = i_sl == i_tp
    else:
        t_out = seg[i_tp, 0]; rsn = 'TP'; net = (x_tp / E - 1) * 100 - fe - MAKER * x_tp / E
    if len(fund):
        net -= fund[(fund[:, 0] > t_in) & (fund[:, 0] <= t_out + MIN), 1].sum() * 100
    # path labels (24 h, −4 net stop, regardless of the exit under test)
    e24 = np.searchsorted(P[:, 0], t_in + 1440 * MIN, 'left')
    s24 = P[s:e24]
    p4 = E * (100 - 4 + fe) / (100 - TAKER)
    k = int(np.argmax(s24[:, 3] <= p4)) if (s24[:, 3] <= p4).any() else len(s24)
    pre = s24[:k + 1] if k < len(s24) else s24
    hh = pre[:, 2].copy()
    if tp_block_first and len(hh):
        hh[0] = E
    reach = (hh.max() / E - 1) * 100 if len(hh) else 0.0
    if k < len(s24):           # the stop bar's high may precede its low: exclude it from REACH unless it is the only bar
        reach = (hh[:-1].max() / E - 1) * 100 if len(hh) > 1 else 0.0
    x5 = E * (1 + (5 + fe) / 100) / (1 - MAKER / 100)
    net_opt = ((x_tp / E - 1) * 100 - fe - MAKER * x_tp / E) if amb else net
    return dict(net=net, net_opt=net_opt, amb=amb, exit_ms=int(t_out), reason=rsn, worked=reach >= 1.0, reach=reach, reach5=bool(reach >= (x5 / E - 1) * 100))


def run_combo(X, paths, ent, ex, portfolio=True, fs_cut=0.0, r_cut=1.0, force_ht=False, keep_mask=None):
    """→ DataFrame of trades for (entry, exit) on episodes routed HT (R ≥ r_cut) and passing FAILED_SPIKE (vs_vwap > fs_cut)."""
    tp, sl, hd = ex
    rows = []
    for i, ep in X.iterrows():
        if keep_mask is not None and not keep_mask[i]:
            continue
        if not force_ht and not (ep.R is not None and ep.R == ep.R and ep.R >= r_cut):
            continue
        if fs_cut is not None and ep.vs_vwap is not None and ep.vs_vwap == ep.vs_vwap and ep.vs_vwap <= fs_cut:
            continue
        pp = paths[(ep.pair, int(ep.pass_ms))]
        P = pp['P']
        en = entry(ent, ep, P)
        if en is None:
            continue
        s, E, fe, blk = en
        r = price(P, pp['fund'], s, E, fe, tp, sl, hd, blk)
        if r is None:
            continue
        r.update(idx=i, pair=ep.pair, entry_ms=int(P[s, 0]), E=E, res=pp['res'])
        rows.append(r)
    T = pd.DataFrame(rows)
    if not len(T):
        return T
    T = T.sort_values('entry_ms').reset_index(drop=True)
    if portfolio:
        keep = []; open_ = []
        for _, r in T.iterrows():
            open_ = [o for o in open_ if o[1] > r.entry_ms]
            if len(open_) >= 10 or any(o[0] == r.pair for o in open_):
                keep.append(False); continue
            keep.append(True); open_.append((r.pair, r.exit_ms + MIN))
        T = T[np.array(keep)].reset_index(drop=True)
    T['half'] = T.entry_ms.map(half)
    T['day'] = pd.to_datetime(T.entry_ms, unit='ms').dt.strftime('%Y-%m-%d')
    T['xday'] = pd.to_datetime(T.exit_ms, unit='ms').dt.strftime('%Y-%m-%d')
    return T


# ───────── stats ─────────
def wilson_lo(p, n, z=1.645):
    if n <= 0:
        return np.nan
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def day_boot(T, B=2000):
    if len(T) < 3:
        return (np.nan, np.nan)
    g = T.groupby('day').net.agg(['sum', 'count']).values
    idx = rng.integers(0, len(g), (B, len(g)))
    m = g[idx, 0].sum(1) / g[idx, 1].sum(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def dcr(T, t0, t1):
    days = pd.date_range(pd.Timestamp(t0, unit='ms').normalize(), pd.Timestamp(t1, unit='ms').normalize(), freq='D').strftime('%Y-%m-%d')
    r = T.groupby('xday').net.sum() * X_SIZE / 100 if len(T) else pd.Series(dtype=float)
    v = np.array([1 + r.get(d, 0.0) for d in days])
    return (np.prod(v) ** (1 / len(v)) - 1) * 100


def stats(T, t0=None, t1=None):
    if T is None or not len(T):
        return dict(N=0)
    n = len(T); wr = (T.net > 0).mean(); w = T.net[T.net > 0]; l = T.net[T.net <= 0]
    aw = w.mean() if len(w) else 0; al = l.mean() if len(l) else 0
    nd = T.day.nunique()
    lo = wilson_lo(wr, nd)
    b = aw / abs(al) if al < 0 else np.inf
    kel = lo - (1 - lo) / b if b and b != np.inf else lo
    ci = day_boot(T)
    o = dict(N=n, days=nd, WR=100 * wr, avg=T.net.mean(), sum=T.net.sum(), ci_lo=ci[0], ci_hi=ci[1], be_WR=100 * abs(al) / (aw + abs(al)) if (aw + abs(al)) > 0 else np.nan,
             wilson_lo=100 * lo, kelly_floor=kel, worked=100 * T.worked.mean(), reach_med=T.reach.median(), reach5=100 * T.reach5.mean())
    if t0 is not None:
        o['DCR'] = dcr(T, t0, t1)
    return o


def fmt(s):
    if not s or s.get('N', 0) == 0:
        return '0'
    return f"{s['N']} · {s['WR']:.0f}% · {s['avg']:+.2f}"


PERIODS = {'2025-H1': ('2025-01-13', '2025-07-01'), '2025-H2': ('2025-07-01', '2026-01-01'), '2026-H1': ('2026-01-01', '2026-07-01'),
           '2026-H2': ('2026-07-01', '2026-11-17'), 'DESIGN 2025': ('2025-01-13', '2026-01-01'), 'HOLDOUT 2026': ('2026-01-01', '2026-11-17'),
           'ALL': ('2025-01-13', '2026-11-17')}
ms = lambda d: pd.Timestamp(d).value // 1_000_000


def per_period(T):
    out = {}
    for k, (a, b) in PERIODS.items():
        s = T[(T.entry_ms >= ms(a)) & (T.entry_ms < ms(b))] if len(T) else T
        out[k] = stats(s, ms(a), ms(b) - 1)
    return out


def path_groups(T):
    g = np.where(T.net > 0, 'CAPTURED', np.where(T.worked, 'GAVE_BACK', 'NEVER_WORKED'))
    return pd.Series(g).value_counts().to_dict(), {k: float(T.net[g == k].sum()) for k in ('CAPTURED', 'GAVE_BACK', 'NEVER_WORKED')}


# ───────── null ─────────
def null_dist(X, paths, design_mask, reps=200):
    """max over (5 random-entry draws × 18 exits) of the DESIGN episode-level avg."""
    eps = [(ep, paths[(ep.pair, int(ep.pass_ms))]) for i, ep in X[design_mask].iterrows()]
    out = []
    for r in range(reps):
        best = -9
        for dr in range(5):
            sel = []
            for ep, pp in eps:
                P = pp['P']; ci = five_min_close_idx(P)
                ci = ci[(P[ci, 0] < ep.pass_ms + 240 * MIN) & (ci + 1 < len(P))]
                if not len(ci):
                    continue
                s = int(rng.choice(ci)) + 1
                sel.append((P, pp['fund'], s, P[s, 1] * (1 + SLIP / 100)))
            for ex in EXITS:
                v = [price(P, f, s, E, TAKER, *ex, False) for P, f, s, E in sel]
                v = [x['net'] for x in v if x is not None]
                best = max(best, np.mean(v))
        out.append(best)
        if r % 20 == 0:
            print('null', r, round(best, 3), flush=True)
    return np.array(out)


def main():
    A = pickle.load(open(os.path.join(DATA, 'episodes_main.pkl'), 'rb'))
    X, paths = A['X'], A['paths']
    D = pickle.load(open(os.path.join(DATA, 'episodes_D.pkl'), 'rb')) if os.path.exists(os.path.join(DATA, 'episodes_D.pkl')) else None
    res = {}
    # ── 1. grid (portfolio-level, HT R ≥ 1, FAILED_SPIKE on) ──
    combo_rows = []; trades = {}
    for ent in ENTRIES:
        for ex in EXITS:
            T = run_combo(X, paths, ent, ex)
            trades[(ent, ex)] = T
            pp = per_period(T)
            row = dict(entry=ent, exit=f"TP{ex[0]}/SL{ex[1]}/{ex[2]}m")
            for k, s in pp.items():
                for m in ('N', 'days', 'WR', 'avg', 'ci_lo', 'ci_hi', 'kelly_floor', 'DCR', 'worked', 'reach5'):
                    row[f"{k}|{m}"] = s.get(m, np.nan)
            if D is not None:
                TD = run_combo(D['X'], D['paths'], ent, ex, force_ht=True)
                sD = stats(TD, ms('2023-07-01'), ms('2024-01-01') - 1)
                for m in ('N', 'WR', 'avg', 'DCR'):
                    row[f"D 2023-H2|{m}"] = sD.get(m, np.nan)
            combo_rows.append(row)
        print('grid', ent, flush=True)
    C = pd.DataFrame(combo_rows)
    C.to_csv(OUTP + '_combos.csv', index=False)
    # episode-level (no portfolio) DESIGN avg for the null comparison
    dm = (X.pass_ms < DESIGN_END).values
    ep_design = {}
    for ent in ENTRIES:
        for ex in EXITS:
            T = run_combo(X, paths, ent, ex, portfolio=False, keep_mask=dm)
            ep_design[(ent, ex)] = T.net.mean() if len(T) else np.nan
    obs_best = max(ep_design.items(), key=lambda kv: kv[1])
    # null on the HT, FAILED_SPIKE-pass DESIGN episodes
    hm = dm & (X.R.fillna(-1) >= 1.0).values & ~(X.vs_vwap.fillna(1) <= 0).values
    nd = null_dist(X, paths, hm, reps=200)
    res['null'] = dict(obs_best_combo=f"{obs_best[0][0]} TP{obs_best[0][1][0]}/SL{obs_best[0][1][1]}/{obs_best[0][1][2]}m", obs_best=obs_best[1],
                       today_design_ep=ep_design[TODAY], p=float((nd >= obs_best[1]).mean()), null_med=float(np.median(nd)),
                       null_p95=float(np.percentile(nd, 95)))
    res['ep_design'] = {f"{k[0]}|TP{k[1][0]}/SL{k[1][1]}/{k[1][2]}m": v for k, v in ep_design.items()}
    # ── 2. FAILED_SPIKE + turnover cuts on E0 × TODAY ──
    cut_rows = []
    for nm, kw in [('FS ≤ 0 (today)', dict(fs_cut=0.0)), ('FS off', dict(fs_cut=None)), ('FS ≤ −1', dict(fs_cut=-1.0)),
                   ('FS ≤ +1', dict(fs_cut=1.0)), ('FS ≤ +2', dict(fs_cut=2.0)),
                   ('R ≥ 0.7', dict(r_cut=0.7)), ('R ≥ 1.5', dict(r_cut=1.5)), ('R ≥ 2.5', dict(r_cut=2.5))]:
        T = run_combo(X, paths, 'E0_RED', TODAY[1], **kw)
        pp = per_period(T)
        cut_rows.append(dict(rule=nm, **{k: fmt(s) for k, s in pp.items()}, **{f"{k}|DCR": s.get('DCR', np.nan) for k, s in pp.items()}))
    # the FS-blocked cohort alone
    Tb = run_combo(X, paths, 'E0_RED', TODAY[1], fs_cut=None)
    Tk = trades[TODAY]
    blk = Tb[~Tb.idx.isin(run_combo(X, paths, 'E0_RED', TODAY[1], portfolio=False).idx)]
    cut_rows.append(dict(rule='FS-blocked cohort only', **{k: fmt(s) for k, s in per_period(blk).items()}))
    pd.DataFrame(cut_rows).to_csv(OUTP + '_cuts.csv', index=False)
    # ── 3. regime (E0 × TODAY and the best portfolio combo) ──
    Xd = X[X.pass_ms < DESIGN_END]
    REG = {'BTC 1d EMA50 > EMA200': ('btc_1d_ema50_200', 0.0), 'BTC prior-day return > 0': ('btc_1d_ret', 0.0),
           'BTC 4h EMA20 slope > 0': ('btc_4h_slope', 0.0), f"alt breadth > DESIGN median": ('breadth', float(Xd.breadth.median())),
           f"Frenzy breadth (armed 24 h) > DESIGN median": ('mkt_frenzy_24h', float(Xd.mkt_frenzy_24h.median()))}
    best_pf = C.sort_values('DESIGN 2025|avg', ascending=False).iloc[0]
    best_key = (best_pf.entry, tuple(int(x.strip('TPSLm')) for x in best_pf.exit.split('/')))
    reg_rows = []
    for lab, key in (('TODAY E0 × TP5/SL4/1440m', TODAY), (f"best DESIGN combo {best_key[0]} {best_pf.exit}", best_key)):
        T = trades[key].merge(X[['btc_1d_ema50_200', 'btc_1d_ret', 'btc_4h_slope', 'breadth', 'mkt_frenzy_24h']], left_on='idx', right_index=True)
        for rn, (v, cut) in REG.items():
            row = dict(set=lab, regime=rn, cut=cut, coverage=100 * T[v].notna().mean())
            pos_in = 0; diff_pos = 0; ok_days = 0
            for hf in ('2025-H1', '2025-H2', '2026-H1', '2026-H2'):
                h = T[T.half == hf]
                a, b = h[h[v] > cut], h[h[v] <= cut]
                sa, sb = stats(a), stats(b)
                row[f"{hf} in"] = f"{fmt(sa)} · {sa.get('days', 0)}d"
                row[f"{hf} out"] = f"{fmt(sb)} · {sb.get('days', 0)}d"
                if sa.get('N', 0) and sa['avg'] > 0 and sa['days'] >= 8:
                    pos_in += 1
                if sa.get('N', 0) and sb.get('N', 0) and sa['avg'] > sb['avg']:
                    diff_pos += 1
            a = T[T[v] > cut]; b = T[T[v] <= cut]
            sa, sb = stats(a), stats(b)
            row.update({'ALL in': f"{fmt(sa)} · {sa.get('days', 0)}d · CI [{sa.get('ci_lo', np.nan):+.2f}, {sa.get('ci_hi', np.nan):+.2f}]",
                        'ALL out': f"{fmt(sb)} · {sb.get('days', 0)}d", 'halves in>0 (≥8 d)': pos_in, 'halves in>out': diff_pos,
                        'PASS': bool(pos_in >= 3 and sa.get('ci_lo', -1) > 0 and diff_pos >= 3)})
            reg_rows.append(row)
    R = pd.DataFrame(reg_rows); R.to_csv(OUTP + '_regime.csv', index=False)
    # ── 4. path facts per entry under TODAY's exit ──
    pf_rows = []
    for ent in ENTRIES:
        T = trades[(ent, TODAY[1])]
        cnt, pts = path_groups(T)
        s = stats(T)
        pf_rows.append(dict(entry=ent, N=s['N'], WORKED=s['worked'], reach_med=s['reach_med'], reach5=s['reach5'],
                            **{f"{k} N · pts": f"{cnt.get(k, 0)} · {pts[k]:+.0f}" for k in pts}, booked=fmt(s)))
    pd.DataFrame(pf_rows).to_csv(OUTP + '_pathfacts.csv', index=False)
    Tk.merge(X, left_on='idx', right_index=True, suffixes=('', '_ep')).to_csv(OUTP + '_trades.csv', index=False)
    pickle.dump(dict(res=res, C=C, R=R, trades={f"{k[0]}|{k[1]}": v for k, v in trades.items()}), open(os.path.join(DATA, 'eval.pkl'), 'wb'))
    print(json.dumps(res['null'], indent=1, default=float))


if __name__ == '__main__':
    main()
