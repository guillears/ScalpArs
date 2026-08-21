#!/usr/bin/env python3
"""Phase-1.5 backtest — 1m partial-candle replay of the momentum-LONG stack.

Fidelity upgrade over phase-1: the engine scans every ~30-95s and its last 5m
candle is PARTIAL. Here every minute reconstructs that exact view: a 100-bar
5m window = 99 completed candles + the in-progress candle aggregated from 1m
bars. Decision points call the REAL calculate_indicators + get_signal on that
window; cheap one-step EMA/RSI/ADX updates are used only as a loose pre-screen.

Output: reports/backtest_cache/phase2_candidates.csv
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
CACHE = os.path.join(ROOT, "reports", "backtest_cache")

import backtest_phase1_replay as P1
from backtest_phase1_replay import TH
from services.indicators import (calculate_indicators, get_signal, gap_expand_flat,
                                 gminflat_band, _rsi_adx_block_rule)
from services.regime import classify_btc_regime

ARM_WIN, ARM_LOSS, FWD_1M = 0.40, -0.70, 360
EPISODE_MIN = 45

def load_raw(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates('open_time').sort_values('open_time')
    return {c: df[c].astype(float).values for c in ('open_time', 'o', 'h', 'l', 'c', 'vol', 'qvol')}

def completed_states(k5):
    """Vector states on COMPLETED 5m bars for the one-step pre-screen."""
    c = pd.Series(k5['c'])
    st = {}
    for w in (5, 8, 20):
        st[f'e{w}'] = c.ewm(span=w, adjust=False).mean().values
    d = c.diff()
    st['ag'] = d.clip(lower=0).ewm(alpha=1 / 12, adjust=False).mean().values
    st['al'] = (-d.clip(upper=0)).ewm(alpha=1 / 12, adjust=False).mean().values
    st['close'] = k5['c']
    st['t'] = k5['open_time']
    return st

def window_ohlcv(k5, k1, b_idx, bucket_open, m_idx):
    """99 completed 5m bars (ending index b_idx) + partial from 1m bars."""
    lo = b_idx - 98
    if lo < 0:
        return None
    rows = [[int(k5['open_time'][i]), k5['o'][i], k5['h'][i], k5['l'][i], k5['c'][i], k5['vol'][i]]
            for i in range(lo, b_idx + 1)]
    sel = slice(*m_idx)
    o = k1['o'][sel.start]
    h = float(np.max(k1['h'][sel])); l = float(np.min(k1['l'][sel]))
    cl = k1['c'][sel.stop - 1]; v = float(np.sum(k1['vol'][sel]))
    rows.append([int(bucket_open), o, h, l, cl, v])
    return rows

def main():
    uni = pd.read_csv(os.path.join(CACHE, 'universe_daily_top50.csv'))
    uni['date'] = pd.to_datetime(uni['date_ms'], unit='ms').astype('datetime64[ns]')
    day_pairs = {d: g.sort_values('rank')['pair'].tolist() for d, g in uni.groupby('date')}
    blacklist = set((getattr(TH, 'pair_blacklist', '') or '').replace(' ', '').split(','))
    ctx_only = {'BTCUSDT', 'ETHUSDT'}

    k5_cache, k1_cache = {}, {}
    def k5_of(p):
        if p not in k5_cache:
            fp = os.path.join(CACHE, 'k5m', f'{p}.csv')
            k5_cache[p] = load_raw(fp) if os.path.exists(fp) else None
        return k5_cache[p]
    def k1_of(p):
        if p not in k1_cache:
            fp = os.path.join(CACHE, 'k1m', f'{p}.csv')
            k1_cache[p] = load_raw(fp) if os.path.exists(fp) else None
        return k1_cache[p]

    # BTC/ETH 5m context needs full range in k5m; ensure present
    btc5, btc1m = k5_of('BTCUSDT'), k1_of('BTCUSDT')
    b1h = pd.read_csv(os.path.join(CACHE, 'btc_1h.csv')).drop_duplicates('open_time').sort_values('open_time')
    b1h_t = b1h['open_time'].values.astype(np.int64)
    b1h_c = b1h['c'].values.astype(float)

    candidates = []
    last_entry, calm_last = {}, {}
    days = sorted(day_pairs)
    _from = os.environ.get('DAYS_FROM')
    if _from:
        days = [d for d in days if d >= pd.Timestamp(_from)]
        print(f"day filter: {len(days)} days from {_from}", flush=True)
    MIN5, MIN1, HOUR = 300_000, 60_000, 3_600_000

    def btc_minute(dec_ms):
        """BTC 5m window indicators + 1h slope/rsi at decision minute (exact)."""
        bo = (dec_ms - MIN1) // MIN5 * MIN5
        b_idx = np.searchsorted(btc5['open_time'], bo) - 1
        m0 = np.searchsorted(btc1m['open_time'], bo)
        m1 = np.searchsorted(btc1m['open_time'], dec_ms)
        if b_idx < 99 or m1 <= m0:
            return None
        w = window_ohlcv(btc5, btc1m, b_idx, bo, (m0, m1))
        ind = calculate_indicators(w)
        if not ind or ind.get('adx') is None:
            return None
        out = {'btc_rsi': ind['rsi'], 'btc_rsi_prev': ind['rsi_prev1'],
               'btc_adx': ind['adx'], 'btc_adx_prev': ind['adx_prev1'],
               'btc_atr_pct': ind['atr'] / ind['price'] * 100,
               'btc_slope': (ind['ema20'] - ind['ema20_prev3']) / ind['ema20_prev3'] * 100
                            if ind['ema20_prev3'] else None,
               'btc_gap': (ind['ema13'] - ind['ema50']) / ind['ema50'] * 100 if ind['ema50'] else None}
        # 1h: 99 completed + partial from 1m
        ho = (dec_ms - MIN1) // HOUR * HOUR
        h_idx = np.searchsorted(b1h_t, ho) - 1
        if h_idx >= 22:
            hm0 = np.searchsorted(btc1m['open_time'], ho)
            closes = np.append(b1h_c[max(0, h_idx - 98):h_idx + 1], btc1m['c'][m1 - 1])
            s = pd.Series(closes)
            e20 = s.ewm(span=20, adjust=False).mean()
            r = s.diff().clip(lower=0).ewm(alpha=1 / 12, adjust=False).mean() / \
                (-s.diff().clip(upper=0)).ewm(alpha=1 / 12, adjust=False).mean()
            rsi1h = 100 - 100 / (1 + r)
            out['btc_1h_slope'] = (e20.iloc[-1] - e20.iloc[-4]) / e20.iloc[-4] * 100
            out['btc_rsi_1h'] = rsi1h.iloc[-1]; out['btc_rsi_1h_prev'] = rsi1h.iloc[-2]
        else:
            out['btc_1h_slope'] = None
        return out

    for di, day in enumerate(days):
        day_ms = int(day.value // 1e6)
        pairs = [p for p in day_pairs[day] if p not in blacklist]
        scan_pairs = [p for p in pairs if p not in ctx_only]
        btc_cache_min = {}

        # panel per minute: gvr + breadth over ALL pairs incl. ctx
        panel_pairs = [p for p in pairs if k5_of(p) is not None and k1_of(p) is not None]
        minute_grid = np.arange(day_ms, day_ms + 86_400_000, MIN1)
        vol_part = np.zeros(len(minute_grid)); vol_avg = np.zeros(len(minute_grid))
        bull_n = np.zeros(len(minute_grid)); tot_n = np.zeros(len(minute_grid))
        pair_pre = {}
        for p in panel_pairs:
            k5, k1 = k5_of(p), k1_of(p)
            st = completed_states(k5)
            m0 = np.searchsorted(k1['open_time'], day_ms)
            m1 = np.searchsorted(k1['open_time'], day_ms + 86_400_000)
            if m1 <= m0:
                continue
            mt = k1['open_time'][m0:m1].astype(np.int64)
            dec = mt + MIN1
            bo = (dec - MIN1) // MIN5 * MIN5
            b_idx = np.searchsorted(st['t'], bo) - 1
            valid = b_idx >= 99
            # partial running aggregates per minute (vectorized per bucket)
            pc = k1['c'][m0:m1]
            pv = np.zeros(m1 - m0); ph = np.zeros(m1 - m0); pl = np.zeros(m1 - m0)
            starts = np.searchsorted(k1['open_time'], bo)
            for j in range(m1 - m0):
                s0 = starts[j] - m0
                pv[j] = k1['vol'][m0 + s0: m0 + j + 1].sum()
                ph[j] = k1['h'][m0 + s0: m0 + j + 1].max()
                pl[j] = k1['l'][m0 + s0: m0 + j + 1].min()
            prev = np.clip(b_idx, 0, len(st['e5']) - 1)
            a5, a8, a20 = 2 / 6, 2 / 9, 2 / 21
            e5 = a5 * pc + (1 - a5) * st['e5'][prev]
            e8 = a8 * pc + (1 - a8) * st['e8'][prev]
            e20 = a20 * pc + (1 - a20) * st['e20'][prev]
            e20p3 = st['e20'][np.clip(b_idx - 2, 0, None)]
            g = np.maximum(pc - st['close'][prev], 0); l = np.maximum(st['close'][prev] - pc, 0)
            b = 1 / 12
            ag = (1 - b) * st['ag'][prev] + b * g; al = (1 - b) * st['al'][prev] + b * l
            rsi1 = np.where(al > 0, 100 - 100 / (1 + ag / np.maximum(al, 1e-12)), 100.0)
            idx = ((mt - day_ms) // MIN1).astype(int)
            # panel aggregates (approximate rolling48 with completed avg)
            av48 = pd.Series(k5['vol']).rolling(48).mean().values
            va = av48[prev]
            ok = valid & ~np.isnan(va)
            np.add.at(vol_part, idx[ok], pv[ok]); np.add.at(vol_avg, idx[ok], va[ok])
            slope_pm = (e20 - e20p3) / e20p3 * 100
            oks = valid & ~np.isnan(slope_pm)
            np.add.at(bull_n, idx[oks], (slope_pm[oks] > 0.02).astype(float))
            np.add.at(tot_n, idx[oks], 1.0)
            if p in scan_pairs:
                screen = valid & (e5 > e8) & (pc > e20) & (e20 > e20p3) \
                         & (rsi1 > 35) & (rsi1 < 75) \
                         & ((e5 - e8) / e8 * 100 > 0.04)
                pair_pre[p] = (mt[screen], bo[screen], b_idx[screen],
                               starts[screen], np.searchsorted(k1['open_time'], mt[screen]) + 1)
        gvr_min = np.where(vol_avg > 0, vol_part / vol_avg, np.nan)
        bull_min = np.where(tot_n > 0, bull_n / tot_n * 100, np.nan)

        for p, (mts, bos, bidx, s0s, s1s) in pair_pre.items():
            k5, k1 = k5_of(p), k1_of(p)
            for mt_, bo_, bi_, s0_, s1_ in zip(mts, bos, bidx, s0s, s1s):
                dec = int(mt_) + MIN1
                le = last_entry.get(p)
                if le is not None and dec - le < EPISODE_MIN * 60_000:
                    continue
                w = window_ohlcv(k5, k1, int(bi_), int(bo_), (int(s0_), int(s1_)))
                if w is None:
                    continue
                ind = calculate_indicators(w)
                if not ind or ind.get('adx') is None or ind.get('ema50') is None:
                    continue
                sig, conf = get_signal(
                    ema5=ind['ema5'], ema8=ind['ema8'], ema13=ind['ema13'], ema20=ind['ema20'],
                    rsi=ind['rsi'], adx=ind['adx'], volume=ind['volume'], avg_volume=ind['avg_volume'],
                    price=ind['price'], ema20_prev3=ind['ema20_prev3'], ema50=ind['ema50'],
                    ema50_prev12=ind['ema50_prev12'], rsi_prev3=ind['rsi_prev3'], rsi_prev2=ind['rsi_prev2'],
                    ema5_prev1=ind['ema5_prev1'], ema8_prev1=ind['ema8_prev1'],
                    ema5_prev2=ind['ema5_prev2'], ema8_prev2=ind['ema8_prev2'],
                    ema13_prev1=ind['ema13_prev1'], ema13_prev2=ind['ema13_prev2'],
                    adx_prev1=ind['adx_prev1'], high_20=ind['high_20'], low_20=ind['low_20'])
                if sig != 'LONG':
                    continue
                bmin_key = dec // MIN1
                if bmin_key not in btc_cache_min:
                    btc_cache_min[bmin_key] = btc_minute(dec)
                b = btc_cache_min[bmin_key]
                if b is None or b['btc_slope'] is None:
                    continue
                atrp = ind['atr'] / ind['price'] * 100
                adx_d = ind['adx'] - ind['adx_prev1'] if ind['adx_prev1'] is not None else None
                rng = ((ind['price'] - ind['low_20']) / (ind['high_20'] - ind['low_20']) * 100
                       if ind['high_20'] and ind['high_20'] != ind['low_20'] else None)
                pair_gap = (ind['ema13'] - ind['ema50']) / ind['ema50'] * 100
                stretch = abs(ind['price'] - ind['ema5']) / ind['price'] * 100
                mi = int((dec - MIN1 - day_ms) // MIN1)
                gvr_now = gvr_min[mi] if 0 <= mi < len(gvr_min) else np.nan
                bull_now = bull_min[mi] if 0 <= mi < len(bull_min) else np.nan
                g58 = abs((ind['ema5'] - ind['ema8']) / ind['ema8'] * 100)
                g813 = abs((ind['ema8'] - ind['ema13']) / ind['ema13'] * 100)
                fan = g58 / g813 if g813 > 0 else None

                blk = None
                if b['btc_adx'] < float(TH.btc_adx_min_long): blk = 'BTC_ADX_LOW'
                elif b['btc_adx'] > float(TH.btc_adx_max_long): blk = 'BTC_ADX_HIGH'
                elif P1.rsi_atr_blocks(P1.RSIATR_L, b['btc_rsi'], b['btc_atr_pct']): blk = 'BTC_RSI_ATR'
                elif P1.cross_blocks(P1.CROSS_L, b['btc_rsi'], b['btc_adx']): blk = 'CROSS'
                elif adx_d is not None and any(a1 <= adx_d < a2 and b1 <= b['btc_adx'] < b2
                                               for a1, a2, b1, b2 in P1.ADXD_L): blk = 'ADX_DELTA'
                elif rng is not None and adx_d is not None and any(
                        a1 <= rng <= a2 and b1 <= adx_d < b2 for a1, a2, b1, b2 in P1.RNGP_L): blk = 'RNGPOS'
                elif fan is not None and any(a <= fan < bd for a, bd in P1.FAN_L): blk = 'FAN_RATIO'
                elif atrp < float(TH.pair_atr_min_long): blk = 'PAIR_ATR_MIN'
                elif atrp >= float(TH.pair_atr_max_long): blk = 'PAIR_ATR_MAX'
                elif atrp >= float(TH.atr_gap_block_atr_min_long) and pair_gap >= float(TH.atr_gap_block_gap_min_long): blk = 'ATR_GAP'
                elif ind['rsi_prev1'] is not None and ind['rsi_prev1'] < float(TH.rsi_prev_min_long) \
                        and ind['rsi'] - ind['rsi_prev1'] >= float(TH.rsi_spike_min_jump_long): blk = 'RSI_SPIKE'
                elif b['btc_gap'] is not None and any(a1 <= b['btc_gap'] < a2 and b1 <= b['btc_adx'] < b2
                                                      for a1, a2, b1, b2 in P1.BGAP_L): blk = 'BTC_GAP'
                elif b['btc_slope'] < float(TH.macro_trend_flat_threshold_long): blk = 'BTC_SLOPE'
                elif abs(b['btc_slope']) > float(TH.btc_ema20_slope_max_long): blk = 'BTC_SLOPE_MAX'
                elif b['btc_1h_slope'] is not None and float(TH.btc_1h_slope_max_long or 0) > 0 and b['btc_1h_slope'] > float(TH.btc_1h_slope_max_long): blk = 'BTC_1H_MAX'
                if blk is None and b['btc_1h_slope'] is not None:
                    db = float(TH.long_btc_1h_deadband); dbp = min(db, float(TH.long_btc_1h_deadband_pos))
                    s1 = b['btc_1h_slope']
                    if db > 0 and ((0 <= s1 < dbp) or (-db < s1 < 0)): blk = 'DEADBAND'
                if blk is None:
                    p20s = ((ind['ema20'] - ind['ema20_prev3']) / ind['ema20_prev3'] * 100
                            if ind['ema20_prev3'] else None)
                    if p20s is not None and abs(p20s) > float(TH.momentum_ema20_slope_max_long): blk = 'PAIR_SLOPE_MAX'
                    elif not np.isnan(gvr_now) and gvr_now < float(TH.global_volume_threshold_long):
                        # engine rescue leg: high-24h-vol pairs trade through deep lulls
                        _rv = float(getattr(TH, 'pair_volume_usd_rescue_long', 0) or 0)
                        _rm = float(getattr(TH, 'global_volume_rescue_max_long', 0) or 0)
                        _lo24 = max(0, int(bi_) - 287)
                        _vol24 = float(np.sum(k5['qvol'][_lo24:int(bi_) + 1]))
                        if not (_rv > 0 and _vol24 >= _rv and (_rm <= 0 or gvr_now < _rm)):
                            blk = 'VOL_GATE'
                    elif getattr(TH, 'spike_guard_enabled', False) and ind['candle_avg_volume_20'] \
                            and ind['candle_volume_raw'] / ind['candle_avg_volume_20'] >= float(TH.spike_guard_volume_multiplier) \
                            and abs((ind['price'] - ind['candle_open']) / ind['candle_open']) * 100 >= float(TH.spike_guard_price_move_pct):
                        blk = 'SPIKE_GUARD'
                calm3d = False
                if blk is None:
                    gap520 = round(abs((ind['ema5'] - ind['ema20']) / ind['price'] * 100), 4)
                    p20s = ((ind['ema20'] - ind['ema20_prev3']) / ind['ema20_prev3'] * 100
                            if ind['ema20_prev3'] else 0)
                    score = sum([55 <= ind['rsi'] < 60, 20 <= ind['adx'] < 25, 0.25 <= gap520 <= 0.50,
                                 (not np.isnan(bull_now)) and bull_now > 50,
                                 20 <= b['btc_adx'] < 25, abs(p20s) > 0.12])
                    if score <= int(TH.entry_quality_score_block_max): blk = 'EQS'
                if blk is None:
                    try:
                        routed = gminflat_band(ind, 'LONG', TH) is True or gap_expand_flat(ind, 'LONG', TH)
                    except Exception:
                        routed = False
                    if routed:
                        regime = classify_btc_regime(b['btc_adx'], b['btc_rsi'], b['btc_slope'])
                        ok = (regime in (getattr(TH, 'nonexp_calm3d_regimes', 'STRONG_BULL')).split(',')
                              and b['btc_atr_pct'] <= float(TH.nonexp_calm3d_btc_atr_max)
                              and stretch <= float(TH.nonexp_calm3d_max_stretch)
                              and (b['btc_1h_slope'] is None or b['btc_1h_slope'] > float(TH.nonexp_calm3d_b1h_min))
                              and (ind['pos_di'] is None or ind['pos_di'] >= float(TH.nonexp_calm3d_min_pos_di))
                              and ind['adx'] >= float(TH.nonexp_calm3d_min_pair_adx))
                        cl = calm_last.get(p)
                        if ok and cl is not None and dec - cl < float(TH.nonexp_calm3d_reentry_cooldown_min) * 60_000:
                            blk = 'CALM3D_REENTRY'
                        elif ok:
                            calm3d = True; calm_last[p] = dec
                        else:
                            blk = 'ROUTER'
                    if blk is None and not calm3d:
                        rule = _rsi_adx_block_rule('LONG', ind['rsi'], ind['adx'], TH)
                        if rule is not None:
                            adm = float(getattr(TH, 'rsiadx_breadth_admit_max', 0) or 0)
                            if not (adm > 0 and not np.isnan(bull_now) and bull_now <= adm):
                                blk = 'PAIR_RSIADX'
                if blk is None and not calm3d and getattr(TH, 'long_unmatched_only', False):
                    p20s = ((ind['ema20'] - ind['ema20_prev3']) / ind['ema20_prev3'] * 100
                            if ind['ema20_prev3'] else None)
                    p50s = ((ind['ema50'] - ind['ema50_prev12']) / ind['ema50_prev12'] * 100
                            if ind['ema50_prev12'] else None)
                    row = {'rng_pos': rng, 'pair_gap': pair_gap, 'adx_delta': adx_d, 'stretch': stretch,
                           'adx': ind['adx'], 'btc_rsi': b['btc_rsi'], 'btc_rsi_prev': b['btc_rsi_prev'],
                           'btc_adx': b['btc_adx'], 'btc_adx_prev': b['btc_adx_prev'],
                           'btc_gap': b['btc_gap'], 'btc_atr_pct': b['btc_atr_pct'],
                           'p20_slope': p20s, 'p50_slope': p50s,
                           'pair_vol_ratio': (ind['volume'] / ind['avg_volume']) if ind['avg_volume'] else None}
                    if P1.cw_matched(row): blk = 'UNMATCHED'
                if blk is not None:
                    continue
                # accepted: score forward on 1m bars
                entry = ind['price']
                f0 = int(s1_)
                outcome, final = 'TIMEOUT', None
                fh = k1['h'][f0:f0 + FWD_1M]; fl = k1['l'][f0:f0 + FWD_1M]; fc = k1['c'][f0:f0 + FWD_1M]
                for j in range(len(fh)):
                    if (fl[j] - entry) / entry * 100 <= ARM_LOSS: outcome = 'LOSS'; break
                    if (fh[j] - entry) / entry * 100 >= ARM_WIN: outcome = 'WIN'; break
                if len(fc):
                    final = (fc[-1] - entry) / entry * 100
                last_entry[p] = dec
                ts = pd.Timestamp(dec, unit='ms')
                candidates.append({'ts': ts, 'pair': p, 'confidence': conf, 'calm3d': calm3d,
                                   'regime': classify_btc_regime(b['btc_adx'], b['btc_rsi'], b['btc_slope']),
                                   'btc_rsi': round(b['btc_rsi'], 1), 'btc_adx': round(b['btc_adx'], 1),
                                   'bull_pct': None if np.isnan(bull_now) else round(bull_now, 1),
                                   'outcome': outcome, 'fwd_final': None if final is None else round(final, 3),
                                   'month': ts.strftime('%Y-%m')})
        # evict
        if di % 5 == 0:
            keep = set()
            for dd in days[max(0, di - 1): di + 6]:
                keep.update(day_pairs.get(dd, []))
            keep |= ctx_only
            for cache in (k5_cache, k1_cache):
                for k in list(cache):
                    if k not in keep:
                        del cache[k]
        if di % 10 == 0:
            print(f"  {di}/{len(days)} {day.date()} candidates: {len(candidates)}", flush=True)

    out = pd.DataFrame(candidates)
    out.to_csv(os.path.join(CACHE, 'phase2_candidates.csv'), index=False)
    print(f"TOTAL: {len(out)}", flush=True)
    for m, g in out.groupby('month'):
        print(f"  {m}: N={len(g)} armWR={(g['outcome']=='WIN').mean()*100:.0f}% avg6h={g['fwd_final'].mean():+.3f}%", flush=True)

if __name__ == '__main__':
    main()
