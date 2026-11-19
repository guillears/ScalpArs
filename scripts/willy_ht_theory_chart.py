"""WILLY_HT THEORY (2026-11-19) — event-study chart: price path (close vs the fill, 5-min grid, 24 h) of the E0 × TODAY trades,
winners vs losers vs a random bar of the same episodes, and the same paths split by the two regime axes of the operator's thesis
(BTC 1d EMA50 > EMA200; Frenzy breadth > the DESIGN median). Input: reports/WILLY_HT_THEORY_2026-11-19_data/{episodes_main,eval}.pkl.

    venv/bin/python scripts/willy_ht_theory_chart.py
"""
import os, sys, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import willy_ht_theory_eval as W   # noqa: E402

GRID = np.arange(0, 1441, 5)


def rel_path(P, s, E):
    t0 = P[s, 0]
    x = P[s:]
    m = (x[:, 0] - t0) / W.MIN
    keep = m <= 1440
    return np.interp(GRID, m[keep], (x[keep, 4] / E - 1) * 100, right=np.nan)


def main():
    A = pickle.load(open(os.path.join(W.DATA, 'episodes_main.pkl'), 'rb')); X, paths = A['X'], A['paths']
    E = pickle.load(open(os.path.join(W.DATA, 'eval.pkl'), 'rb'))
    T = E['trades']["E0_RED|(5, 4, 1440)"].merge(X[['btc_1d_ema50_200', 'mkt_frenzy_24h', 'pass_ms']], left_on='idx', right_index=True)
    rng = np.random.default_rng(7)
    Y, Rnd = [], []
    for _, r in T.iterrows():
        pp = paths[(r.pair, int(r.pass_ms))]; P = pp['P']
        Y.append(rel_path(P, 0, r.E))
        ci = W.five_min_close_idx(P); ci = ci[(P[ci, 0] < r.pass_ms + 240 * W.MIN) & (ci + 1 < len(P))]
        s = int(rng.choice(ci)) + 1
        Rnd.append(rel_path(P, s, P[s, 1]))
    Y = np.array(Y); Rnd = np.array(Rnd)
    fz_cut = float(X[X.pass_ms < W.DESIGN_END].mkt_frenzy_24h.median())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    h = GRID / 60
    for lab, m, col in (('winners (net > 0)', (T.net > 0).values, '#2a9d8f'), ('losers', (T.net <= 0).values, '#e76f51')):
        med = np.nanmedian(Y[m], 0); q1, q3 = np.nanpercentile(Y[m], [25, 75], 0)
        ax[0].plot(h, med, color=col, lw=2, label=f"{lab} N={m.sum()}"); ax[0].fill_between(h, q1, q3, color=col, alpha=.15)
    ax[0].plot(h, np.nanmedian(Rnd, 0), color='#6c757d', lw=1.5, ls='--', label='random bar, same episodes (median)')
    ax[0].plot(h, np.nanmedian(Y, 0), color='k', lw=1.5, label=f'all E0 trades (median) N={len(T)}')
    ax[0].axhline(5, color='#2a9d8f', lw=.6, ls=':'); ax[0].axhline(-4, color='#e76f51', lw=.6, ls=':')
    ax[0].set_title('WILLY_HT E0 (today) — close vs fill, hindsight split by outcome\n(TP +5 / SL −4 / 24 h, 2025-02 → 2026-11-17, rebuild)')
    ax[0].set_xlabel('hours after the fill'); ax[0].set_ylabel('% vs fill (gross)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for lab, m, col, ls in (('BTC 1d bull · Frenzy breadth high', ((T.btc_1d_ema50_200 > 0) & (T.mkt_frenzy_24h > fz_cut)).values, '#264653', '-'),
                            ('BTC 1d bull · breadth low', ((T.btc_1d_ema50_200 > 0) & (T.mkt_frenzy_24h <= fz_cut)).values, '#2a9d8f', '--'),
                            ('BTC 1d bear · breadth high', ((T.btc_1d_ema50_200 <= 0) & T.btc_1d_ema50_200.notna() & (T.mkt_frenzy_24h > fz_cut)).values, '#e76f51', '-'),
                            ('BTC 1d bear · breadth low', ((T.btc_1d_ema50_200 <= 0) & T.btc_1d_ema50_200.notna() & (T.mkt_frenzy_24h <= fz_cut)).values, '#f4a261', '--')):
        if m.sum() >= 5:
            ax[1].plot(h, np.nanmedian(Y[m], 0), color=col, lw=2, ls=ls, label=f"{lab} N={m.sum()} avg {T.net[m].mean():+.2f}")
    ax[1].axhline(0, color='k', lw=.6)
    ax[1].set_title(f'Same trades by the operator\'s regime (tradable at entry)\nFrenzy breadth = armed episodes in the prior 24 h > {fz_cut:.0f} · 2025-H1 unscored (no 200 d BTC history)')
    ax[1].set_xlabel('hours after the fill'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(W.OUTP + '_event_study.png', dpi=120)
    print('saved', W.OUTP + '_event_study.png')
    for k, hh in ((12, 1), (36, 3), (288, 24)):
        print(f"median at {hh} h: all {np.nanmedian(Y[:, k]):+.2f} · winners {np.nanmedian(Y[(T.net > 0).values, k]):+.2f} · losers {np.nanmedian(Y[(T.net <= 0).values, k]):+.2f} · random {np.nanmedian(Rnd[:, k]):+.2f}")


if __name__ == '__main__':
    main()
