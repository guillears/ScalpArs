"""WILLY_HT W/L 594 — anchored price-path chart of the 44 dip-cut master fills (winners vs losers; FAILED_SPIKE kept vs blocked).
Close / entry fill − 1 (%), 1m bars, −60 min → +24 h; the path keeps running after the exit (hindsight past the exit). Research only."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTP = os.path.join(REPO, 'reports', 'WILLY_HT_WL594_2026-11-19')
P = json.load(open(os.path.join(REPO, 'reports', 'WILLY_HT_WL594_data', 'paths_1m.json')))
F = pd.read_csv(OUTP + '_fills.csv', low_memory=False)
A = F[(F.set == 'master') & (F.off_peak <= -1.4)]
grid = np.arange(-60, 1441)


def curve(r):
    a = np.array(P[f"{r.pair}|{int(r.entry_ms)}"])
    m = (a[:, 0] - (int(r.entry_ms) // 60000) * 60000) / 60000
    s = pd.Series(a[:, 3] / r.entry_price * 100 - 100, index=m)
    return s.reindex(grid).ffill().values


C = {i: curve(r) for i, r in A.iterrows()}
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
for k, (title, groups, xmax) in enumerate([
        ('44 dip-cut fills: winners vs losers, first 6 h', [('winners', A.win, '#1b9e77'), ('losers', ~A.win, '#d95f02')], 360),
        ('44 dip-cut fills: winners vs losers, 24 h', [('winners', A.win, '#1b9e77'), ('losers', ~A.win, '#d95f02')], 1440),
        ('FAILED_SPIKE kept (12) vs blocked (32), first 6 h', [('FS kept', A.failed_spike_kept, '#386cb0'), ('FS blocked', ~A.failed_spike_kept, '#7f7f7f')], 360)]):
    a = ax[k]
    for lab, m, col in groups:
        M = np.array([C[i] for i in A[m].index])
        med = np.nanmedian(M, 0); q1, q3 = np.nanpercentile(M, [25, 75], 0)
        a.plot(grid, med, color=col, lw=2, label=f"{lab} (n={len(M)}) median")
        a.fill_between(grid, q1, q3, color=col, alpha=0.18)
    a.axhline(5.07, color='#1b9e77', ls=':', lw=1); a.axhline(-3.9, color='#d95f02', ls=':', lw=1); a.axhline(0, color='k', lw=0.6)
    a.axvline(0, color='k', lw=0.6, ls='--')
    a.set_xlim(-60, xmax); a.set_ylim(-15, 20); a.set_title(title, fontsize=10); a.set_xlabel('minutes from entry'); a.set_ylabel('% vs entry fill')
    a.legend(fontsize=8, loc='upper left')
fig.suptitle('WILLY_HT master fills after the DL 594 dip cut — anchored price paths (band = IQR; dotted = TP +5 net / stop −4 net; '
             'paths continue after the exit = hindsight)', fontsize=10)
fig.tight_layout(); fig.savefig(OUTP + '_paths.png', dpi=110)
print('ok')
