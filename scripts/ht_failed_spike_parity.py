"""HT FAILED_SPIKE parity — live engine vs the WILLY rebuild, on all real FRENZY_WILLY_HT fills (research, 2026-11-19).

For each real HT fill (reports/WILLY_BACKTEST_WHY_2026-11-18_fills.csv) three FAILED_SPIKE decisions:
  LIVE      services.watch_filters.willy_blocks(ht=True) on the fill's own stamp entry_frenzy_vs_vwap_pct (MASTER_POOL_stacked.csv)
  REBUILD   the rebuild's value for the same signal bar: services.frenzy.frenzy_walk on the rebuild's 5m data + the rebuild's
            normal-hour model (wsignals.py), and — when the rebuild itself generated that red row — its stored sig_all.pkl value
  TODAY     the today_keep / today_blk columns of the fills CSV
Rebuild code + data live outside the repo: set WILLY_REBUILD_DIR to the folder holding code/wdata.py and sig_all.pkl.
Usage: WILLY_REBUILD_DIR=<dir> venv/bin/python -I scripts/ht_failed_spike_parity.py
"""
import os, sys, pickle
from types import SimpleNamespace
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REB = os.environ.get("WILLY_REBUILD_DIR")
if not REB:
    sys.exit("set WILLY_REBUILD_DIR")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REB, "code"))
from services.frenzy import frenzy_walk                      # noqa: E402
from services.watch_filters import willy_blocks             # noqa: E402
import wdata                                                # noqa: E402

BAR = 300_000; H1 = 3_600_000
CFG, TH = wdata.cfg()
TH_ON = SimpleNamespace(**vars(TH)); TH_ON.watch_block_willy_ht_failed_spike = True


def norm_series(df, sym):
    """the rebuild's normal-hour model (copied from wsignals.pair_signals)."""
    t = df.t.values.astype(np.int64); h, l, c, v = df.h.values, df.l.values, df.c.values, df.v.values
    hk = t // H1
    hdf = pd.DataFrame({'k': hk, 'h': h, 'l': l, 'c': c, 'v': v, 'o': df.o.values}).groupby('k').agg(
        h=('h', 'max'), l=('l', 'min'), c=('c', 'last'), v=('v', 'sum'), nb=('o', 'size'))
    hdf = hdf[hdf.nb == 12]
    hmed = (hdf.v * (hdf.h + hdf.l + hdf.c) / 3.0).rolling(720, min_periods=240).median()
    hkeys = hdf.index.values.astype(np.int64)
    P = (360 + sum(map(ord, sym)) % 120) * 60_000
    tr = ((t + BAR) // P) * P
    pos = np.searchsorted(hkeys, (tr - BAR - 25 * H1) // H1, side='right') - 1
    return np.where(pos >= 0, hmed.values[np.clip(pos, 0, None)], np.nan)


def blk(v):
    return bool(willy_blocks(TH_ON, None, None, v, ht=True))


def main():
    f = pd.read_csv(f"{ROOT}/reports/WILLY_BACKTEST_WHY_2026-11-18_fills.csv")
    f = f[f.sleeve == "FRENZY_WILLY_HT"].copy()
    m = pd.read_csv(f"{ROOT}/reports/MASTER_POOL_stacked.csv", low_memory=False)
    m = m[m.entry_strategy.astype(str).str.contains("WILLY_HT")].copy()
    m['ok'] = pd.to_datetime(m.opened_at).dt.floor('s')
    sig = pickle.load(open(os.path.join(REB, "sig_all.pkl"), 'rb'))
    cache = {}
    rows = []
    for _, r in f.iterrows():
        op = pd.Timestamp(r.opened).floor('s')
        mm = m[(m.pair == r.pair) & ((m.ok - op).abs() <= pd.Timedelta('2s'))]
        s = mm.iloc[0] if len(mm) else None
        op_ms = int(op.value // 1_000_000)
        bar_open = (op_ms // BAR) * BAR - BAR            # the closed red signal bar before the live pass
        catch = bool(s is not None and str(s.get('entry_frenzy_catchup')) in ('True', '1', '1.0'))
        L_vs = float(s.entry_frenzy_vs_vwap_pct) if s is not None and pd.notna(s.entry_frenzy_vs_vwap_pct) else None
        L_vw = float(s.entry_frenzy_vwap) if s is not None and pd.notna(s.entry_frenzy_vwap) else None
        L_close = L_vw * (1 + L_vs / 100) if (L_vw and L_vs is not None) else None
        L_spike = pd.Timestamp(s.entry_frenzy_spike_at) if s is not None and pd.notna(s.entry_frenzy_spike_at) else None
        # rebuild walk at the live signal bar
        if r.pair not in cache:
            df = wdata.load_5m(r.pair)
            cache[r.pair] = (df, norm_series(df, r.pair) if df is not None else None)
        df, nm = cache[r.pair]
        R = dict(R_vwap=None, R_vs=None, R_close=None, R_spike=None, R_bar_red=None)
        norm_reb = norm_live = None; LN = dict(LN_vs=None, LN_spike=None)
        if df is not None:
            jj = np.where(df.t.values == bar_open)[0]
            if len(jj):
                j = int(jj[0]); a = max(0, j - 1498)
                bars = df.iloc[a:j + 1][['t', 'o', 'h', 'l', 'c', 'v']].values.tolist()
                norm_reb = float(nm[j])
                ep = frenzy_walk(bars, norm_reb, TH)
                if ep:
                    R = dict(R_vwap=ep['vwap'], R_vs=round(ep['vs_vwap_pct'], 3), R_close=ep['price'],
                             R_spike=pd.Timestamp(int(ep['spike_ts']), unit='ms'), R_bar_red=bool(ep['bar_red']))
                # the LIVE normal hour, implied by the stamp: entry_frenzy_vol_mult = last-hour quote volume / norm (1 dp)
                vm = float(s.entry_frenzy_vol_mult) if (s is not None and pd.notna(s.entry_frenzy_vol_mult)) else None
                if vm and vm > 0:
                    w = df.iloc[j - 11:j + 1]
                    q1h = float((w.v * (w.h + w.l + w.c) / 3.0).sum())
                    norm_live = q1h / vm
                    ep2 = frenzy_walk(bars, norm_live, TH)
                    if ep2:
                        LN = dict(LN_vs=round(ep2['vs_vwap_pct'], 3), LN_spike=pd.Timestamp(int(ep2['spike_ts']), unit='ms'))
        # the rebuild's own generated red row at that bar (sig_all.pkl), and its first red row for the episode
        S_vs = S_first_bar = S_first_vs = None
        for e in (sig.get(r.pair, ([], ''))[0]):
            if e.get('status') != 'ARMED':
                continue
            for k, rr in enumerate(e['reds']):
                if rr['bar_ms'] == bar_open:
                    S_vs = rr['vs_vwap']
                    S_first_bar = pd.Timestamp(e['reds'][0]['bar_ms'], unit='ms'); S_first_vs = e['reds'][0]['vs_vwap']
        rows.append(dict(pair=r.pair, opened=r.opened, signal_bar=pd.Timestamp(bar_open, unit='ms'), catchup=catch,
                         live_spike_at=L_spike, rebuild_spike_at=R['R_spike'],
                         live_vwap=L_vw, rebuild_vwap=R['R_vwap'],
                         vwap_diff_pct=((R['R_vwap'] / L_vw - 1) * 100 if (L_vw and R['R_vwap']) else None),
                         live_bar_close=L_close, rebuild_bar_close=R['R_close'],
                         live_vs_vwap=L_vs, rebuild_vs_vwap=R['R_vs'], sigall_vs_vwap=S_vs,
                         sigall_first_red_bar=S_first_bar, sigall_first_red_vs=S_first_vs,
                         norm_rebuild=norm_reb, norm_live_implied=norm_live,
                         norm_ratio_live_over_rebuild=(norm_live / norm_reb if (norm_live and norm_reb) else None),
                         rebuild_livenorm_spike_at=LN['LN_spike'], rebuild_livenorm_vs_vwap=LN['LN_vs'],
                         dec_rebuild_livenorm=('BLOCK' if (LN['LN_vs'] is not None and blk(LN['LN_vs'])) else ('KEEP' if LN['LN_vs'] is not None else 'NO_EPISODE')),
                         dec_live=('BLOCK' if (L_vs is not None and blk(L_vs)) else ('KEEP' if L_vs is not None else 'UNREAD')),
                         dec_rebuild=('BLOCK' if (R['R_vs'] is not None and blk(R['R_vs'])) else ('KEEP' if R['R_vs'] is not None else 'NO_EPISODE')),
                         dec_today=('KEEP' if bool(r.today_keep) else 'BLOCK'), today_blk=r.today_blk,
                         stack_keep=(s.stack_keep if s is not None else None)))
    out = pd.DataFrame(rows)
    out['agree_live_rebuild'] = out.dec_live == out.dec_rebuild
    out['agree_live_rebuild_livenorm'] = out.dec_live == out.dec_rebuild_livenorm
    out['agree_live_today'] = out.dec_live == out.dec_today
    out.to_csv(f"{ROOT}/reports/HT_FAILED_SPIKE_PARITY_2026-11-19.csv", index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(out[['pair', 'opened', 'live_spike_at', 'rebuild_spike_at', 'vwap_diff_pct', 'live_vs_vwap', 'rebuild_vs_vwap', 'sigall_vs_vwap',
               'dec_live', 'dec_rebuild', 'dec_today', 'catchup']].to_string())
    print(out[['pair', 'opened', 'norm_ratio_live_over_rebuild', 'rebuild_livenorm_spike_at', 'rebuild_livenorm_vs_vwap', 'dec_rebuild_livenorm']].to_string())
    print(out.norm_ratio_live_over_rebuild.describe(percentiles=[.1, .25, .5, .75, .9]))
    print('live==rebuild(livenorm)', out.agree_live_rebuild_livenorm.sum())
    print('live==rebuild', out.agree_live_rebuild.sum(), '/', len(out), ' live==today', out.agree_live_today.sum())


if __name__ == '__main__':
    main()
