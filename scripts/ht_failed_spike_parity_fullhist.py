"""HT FAILED_SPIKE parity — full-history sensitivity (research, 2026-11-19).

The rebuild and the engine share ONE FAILED_SPIKE definition (services.watch_filters.willy_blocks on frenzy_walk's vs_vwap_pct of the
closed red signal bar, 3 dp, ≤ 0 blocks). The only per-fill input that differed on the 49 real HT fills is the normal hour (norm), which
moves the episode anchor and therefore the spike VWAP. This script re-walks every unfiltered rebuild HT entry (h7 OFF run) with the norm
scaled by the live/rebuild ratios measured on the 49 real fills and counts how often the FAILED_SPIKE decision flips; flipped entries are
priced at the HT exit +5 / −4 / 24 h on the rebuild's 5m klines (stop checked first inside a bar, 0.10 % slip + 0.07/0.10 fees as the rebuild).
Usage: WILLY_REBUILD_DIR=<dir> venv/bin/python -I scripts/ht_failed_spike_parity_fullhist.py
"""
import os, sys, pickle, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REB = os.environ.get("WILLY_REBUILD_DIR") or sys.exit("set WILLY_REBUILD_DIR")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(REB, "code"))
from services.frenzy import frenzy_walk                       # noqa: E402
import ht_failed_spike_parity as P                           # noqa: E402

BAR = 300_000
TP, SL, HOLD_MS, SLIP = 5.0, -4.0, 24 * 3_600_000, 0.10


def price(df, entry_ms, live):
    """+5 / −4 / 24 h long from the live price at entry_ms (the forming bar's open)."""
    e = live * (1 + SLIP / 100)
    w = df[(df.t >= entry_ms) & (df.t < entry_ms + HOLD_MS)]
    for _, b in w.iterrows():
        if b.l <= e * (1 + SL / 100):
            return SL - SLIP - 0.10
        if b.h >= e * (1 + TP / 100):
            return TP - 0.07
    if not len(w):
        return None
    return (float(w.c.iloc[-1]) / e - 1) * 100 - SLIP - 0.10


def main():
    par = pd.read_csv(f"{ROOT}/reports/HT_FAILED_SPIKE_PARITY_2026-11-19.csv")
    ratios = par.norm_ratio_live_over_rebuild.dropna().values
    on, ron, off, roff = pickle.load(open(os.path.join(REB, "why", "h7_nofilter.pkl"), 'rb'))
    ht = off[off.sleeve == "FRENZY_WILLY_HT"].reset_index(drop=True)
    rows = []; t0 = time.time()
    for pair, g in ht.groupby('pair'):
        df = P.wdata.load_5m(pair); nm = P.norm_series(df, pair); tv = df.t.values
        for _, r in g.iterrows():
            bo = int(r.entry_ms) - BAR
            jj = np.where(tv == bo)[0]
            if not len(jj):
                continue
            j = int(jj[0]); a = max(0, j - 1498)
            bars = df.iloc[a:j + 1][['t', 'o', 'h', 'l', 'c', 'v']].values.tolist()
            base = P.blk(r.vs_vwap)
            flips = noep = 0
            for k in ratios:
                ep = frenzy_walk(bars, nm[j] * k, P.TH)
                if not ep:
                    noep += 1; continue
                if P.blk(round(ep['vs_vwap_pct'], 3)) != base:
                    flips += 1
            rows.append(dict(pair=pair, entry_ms=int(r.entry_ms), vs_vwap=r.vs_vwap, base_block=base, flip_share=flips / len(ratios),
                             noep_share=noep / len(ratios), net_rebuild_exit=r.net))
        print(pair, len(rows), f"{time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    fl = out[out.flip_share > 0].copy()
    cache = {}
    nets = []
    for _, r in fl.iterrows():
        if r.pair not in cache:
            cache[r.pair] = P.wdata.load_5m(r.pair)
        df = cache[r.pair]
        live = float(df.o[df.t == r.entry_ms].iloc[0]) if (df.t == r.entry_ms).any() else None
        nets.append(price(df, int(r.entry_ms), live) if live else None)
    fl['net_5_4_24h'] = nets
    out = out.merge(fl[['pair', 'entry_ms', 'net_5_4_24h']], on=['pair', 'entry_ms'], how='left')
    out.to_csv(f"{ROOT}/reports/HT_FAILED_SPIKE_PARITY_2026-11-19_fullhist.csv", index=False)
    n = len(out)
    print('entries', n, 'expected flip share', round(out.flip_share.mean() * 100, 2), '% · expected no-episode share', round(out.noep_share.mean() * 100, 2),
          '% · entries with any flip', len(fl))
    # expected net effect on the KEPT set: entries that would flip KEEP→BLOCK leave it, BLOCK→KEEP join it (weighted by flip_share)
    k2b = out[(~out.base_block) & (out.flip_share > 0)]; b2k = out[(out.base_block) & (out.flip_share > 0)]
    print('KEEP→BLOCK expected N', round(k2b.flip_share.sum(), 2), 'avg net', round(k2b.net_5_4_24h.mean(), 3) if len(k2b) else None)
    print('BLOCK→KEEP expected N', round(b2k.flip_share.sum(), 2), 'avg net', round(b2k.net_5_4_24h.mean(), 3) if len(b2k) else None)
    print('kept N', int((~out.base_block).sum()))
    print(fl.to_string())


if __name__ == '__main__':
    main()
