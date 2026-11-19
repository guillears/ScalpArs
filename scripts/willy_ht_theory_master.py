"""WILLY_HT THEORY (2026-11-19) — master check: every real FRENZY_WILLY_HT fill (reports/MASTER_POOL_stacked.csv, 49 fills
10-23 → 11-15) re-priced on every entry × exit of the frozen grid (reports/WILLY_HT_THEORY_2026-11-19_PREREG.md), from its own
1m path (5m fallback). E0 = the real fill price; the other entries are rebuilt from the same episode (red bar from 5m klines, spike
high / VWAP from the fill's stamps). FAILED_SPIKE kept / blocked from reports/WILLY_BACKTEST_WHY_2026-11-18_fills.csv.

    WHT_REBUILD_DIR=<rebuild folder> venv/bin/python scripts/willy_ht_theory_master.py
"""
import os, sys, pickle
import numpy as np, pandas as pd

RB = os.environ.get('WHT_REBUILD_DIR')
if not RB:
    sys.exit('set WHT_REBUILD_DIR to the WILLY rebuild folder')
sys.path.insert(0, os.path.join(RB, 'code'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wdeep_feat as WF                       # noqa: E402
from wdata import load_funding                # noqa: E402
import willy_ht_theory_eval as W             # noqa: E402
from willy_ht_theory_build import load_path, WIN   # noqa: E402

REPO = W.REPO; MIN = W.MIN; BAR = W.BAR


def main():
    m = pd.read_csv(os.path.join(REPO, 'reports', 'MASTER_POOL_stacked.csv'), low_memory=False)
    m = m[m.entry_strategy == 'FRENZY_WILLY_HT'].copy()
    f = pd.read_csv(os.path.join(REPO, 'reports', 'WILLY_BACKTEST_WHY_2026-11-18_fills.csv'))
    f = f[f.sleeve == 'FRENZY_WILLY_HT'][['pair', 'opened', 'today_keep', 'today_blk']]
    m['opened'] = m.opened_at.str[:19]
    m = m.merge(f, on=['pair', 'opened'], how='left')
    rows = []
    for _, r in m.iterrows():
        t_open = int(pd.Timestamp(r.opened_at).value // 1_000_000)
        pass_ms = (t_open // BAR) * BAR                  # the pass right after the red bar closed
        bar_ms = pass_ms - BAR
        d5 = WF.p5(r.pair)
        if d5 is None:
            continue
        j = int(np.searchsorted(d5.t.values, bar_ms))
        if j >= len(d5) or int(d5.t.iloc[j]) != bar_ms:
            continue
        P, res = load_path(r.pair, pass_ms, pass_ms + WIN)
        if P is None or len(P) < 50:
            continue
        fu = load_funding(r.pair)
        fund = fu[(fu.t > pass_ms) & (fu.t <= pass_ms + WIN)][['t', 'rate']].values.astype(float) if fu is not None else np.zeros((0, 2))
        close = float(d5.c.iloc[j])
        ep = dict(pass_ms=pass_ms, live=float(r.entry_price) / (1 + W.SLIP / 100), close=close, red_hi=float(d5.h.iloc[j]),
                  peak=float(r.entry_frenzy_willy_spike_high) if r.entry_frenzy_willy_spike_high == r.entry_frenzy_willy_spike_high else np.nan,
                  vwap=float(r.entry_frenzy_vwap) if r.entry_frenzy_vwap == r.entry_frenzy_vwap else np.nan)
        for ent in W.ENTRIES:
            en = W.entry(ent, ep, P)
            if en is None:
                continue
            s, E, fe, blk = en
            if ent == 'E0_RED':
                E = float(r.entry_price)                   # the real fill
            for ex in W.EXITS:
                x = W.price(P, fund, s, E, fe, *ex, blk)
                if x is None:
                    continue
                rows.append(dict(pair=r.pair, opened=r.opened, keep=bool(r.today_keep) if r.today_keep == r.today_keep else bool(r.stack_keep),
                                 entry=ent, exit=f"TP{ex[0]}/SL{ex[1]}/{ex[2]}m", res=res, entry_ms=int(P[s, 0]), **x))
    M = pd.DataFrame(rows)
    M['day'] = pd.to_datetime(M.entry_ms, unit='ms').dt.strftime('%Y-%m-%d')
    M['xday'] = pd.to_datetime(M.exit_ms, unit='ms').dt.strftime('%Y-%m-%d')
    M.to_csv(W.OUTP + '_master.csv', index=False)
    print(M.groupby(['entry']).pair.size() / len(W.EXITS), M.res.value_counts() / len(W.EXITS) / len(W.ENTRIES))


if __name__ == '__main__':
    main()
