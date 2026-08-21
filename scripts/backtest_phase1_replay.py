#!/usr/bin/env python3
"""Phase-1 backtest — step C: entry-signal replay of the CURRENT momentum-LONG
stack over Jan-01→Jul-31 2026 on the reconstructed daily top-50 universe.

Fidelity design (spec: gate-extraction agent, 2026-08-19):
- Signals: the REAL services.indicators.get_signal with the REAL trading_config
  (imported, not reimplemented). Pair/BTC indicators mirror the engine's ta-lib
  calls; EMA50 uses an exact 100-tap kernel matching the engine's 100-bar window
  (all faster-decaying indicators use full-span ta — seed influence < 1e-3).
- Engine chain: every ACTIVE momentum-LONG gate from the spec, in engine order.
  Inactive-by-config gates are asserted inactive at load (fails loudly if config
  drifts). Stateful approximations: PAIR_HELD/COOLDOWN → 45-min per-pair episode
  dedup; BTC_ACCEL_CHASE skipped (needs live open events); max_open ignored
  (capacity reported separately).
- Scan cadence: one decision per completed 5m bar (live: ~30-95s incl. partial
  candle — validation vs real Jun-Jul fills measures the timing cost).
- Outcome: model-free arm test (+0.40% high before -0.70% low within 6h,
  same-bar ambiguity counted as LOSS) + 6h final mark.

Output: reports/backtest_cache/phase1_candidates.csv
"""
import os, sys, math
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "reports", "backtest_cache")

import config as config_module
from services.indicators import (get_signal, gap_expand_flat, gminflat_band,
                                 _rsi_adx_block_rule, determine_macro_regime)
from services.regime import classify_btc_regime
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

TH = config_module.trading_config.thresholds
START = pd.Timestamp("2026-01-01")
ARM_WIN, ARM_LOSS, FWD_BARS = 0.40, -0.70, 72
EPISODE_MIN = 45  # per-pair episode dedup (PAIR_HELD/COOLDOWN proxy)

# ---- config sanity: gates the spec marks inactive must still be inactive ----
def _assert_cfg():
    a = []
    if getattr(TH, 'btc_global_filter_enabled', False): a.append('btc_global_filter_enabled')
    if getattr(TH, 'fake_bull_guard_enabled', False): a.append('fake_bull_guard_enabled')
    if getattr(TH, 'btc_trend_filter_enabled', False): a.append('btc_trend_filter_enabled')
    if getattr(TH, 'market_breadth_filter_enabled', False): a.append('market_breadth_filter_enabled')
    if getattr(TH, 'pair_trend_filter_long_enabled', False): a.append('pair_trend_filter_long_enabled')
    if getattr(TH, 'adx_dir_long', 'both') != 'both': a.append('adx_dir_long')
    if getattr(TH, 'btc_adx_dir_long', 'both') != 'both': a.append('btc_adx_dir_long')
    if getattr(TH, 'btc_1h_5m_rsi_dir_filter_long', '') != '': a.append('btc_1h_5m_rsi_dir_filter_long')
    if getattr(TH, 'btc_atr_btc_adx_filter_long', '') != '': a.append('btc_atr_btc_adx_filter_long')
    if float(getattr(TH, 'btc_1h_slope_min_long', 0) or 0) != 0.0: a.append('btc_1h_slope_min_long')
    if a:
        raise SystemExit(f"CONFIG DRIFT vs gate spec — reimplement before replay: {a}")
_assert_cfg()

# ---- indicator series (engine-mirrored) ----
def ema_kernel_100(close, window):
    """EMA(window) computed exactly as the engine sees it: ewm(adjust=False)
    over a trailing 100-bar window, via a 100-tap linear kernel."""
    a = 2.0 / (window + 1.0)
    w = np.array([a * (1 - a) ** j for j in range(99)] + [(1 - a) ** 99])
    x = close.values.astype(float)
    conv = np.convolve(x, w, mode='full')   # conv[t] = sum_j w[j] * x[t-j]
    out = np.full(len(x), np.nan)
    out[99:] = conv[99:len(x)]
    return pd.Series(out, index=close.index)

def pair_series(df):
    c, h, l, v = df['c'], df['h'], df['l'], df['vol']
    s = pd.DataFrame(index=df.index)
    s['price'] = c
    for wn in (5, 8, 13, 20):
        s[f'ema{wn}'] = EMAIndicator(close=c, window=wn).ema_indicator()
    s['ema50'] = ema_kernel_100(c, 50)
    s['rsi'] = RSIIndicator(close=c, window=12).rsi()
    adxi = ADXIndicator(high=h, low=l, close=c, window=14)
    s['adx'], s['pos_di'], s['neg_di'] = adxi.adx(), adxi.adx_pos(), adxi.adx_neg()
    s['atr'] = AverageTrueRange(high=h, low=l, close=c, window=14).average_true_range()
    s['volume'] = v.ewm(span=5, adjust=False).mean()
    s['avg_volume'] = v.rolling(20).mean()
    s['avg_volume_global'] = v.rolling(48).mean()
    s['candle_open'], s['candle_high'], s['candle_low'] = df['o'], h, l
    s['candle_volume_raw'] = v
    s['candle_avg_volume_20'] = v.rolling(20).mean()
    s['high_20'] = h.rolling(20).max()
    s['low_20'] = l.rolling(20).min()
    for col, k in [('ema5_prev1', 1), ('ema5_prev2', 2), ('ema8_prev1', 1), ('ema8_prev2', 2),
                   ('ema13_prev1', 1), ('ema13_prev2', 2)]:
        s[col] = s[col.split('_')[0]].shift(k)
    s['ema20_prev3'] = s['ema20'].shift(3)
    s['ema50_prev12'] = s['ema50'].shift(12)
    for k in (1, 2, 3, 6):
        s[f'rsi_prev{k}'] = s['rsi'].shift(k)
    s['adx_prev1'] = s['adx'].shift(1)
    return s

def load_k(path):
    df = pd.read_csv(path)
    df['ts'] = pd.to_datetime(df['open_time'], unit='ms').astype('datetime64[ns]')
    df = df.drop_duplicates('open_time').set_index('ts').sort_index()
    for col in ('o', 'h', 'l', 'c', 'vol', 'qvol'):
        df[col] = df[col].astype(float)
    return df

# ---- BTC panels ----
def btc_panels():
    b5 = load_k(os.path.join(CACHE, 'btc_5m.csv'))
    p = pair_series(b5)
    btc = pd.DataFrame(index=p.index)
    btc['btc_rsi'] = p['rsi']; btc['btc_rsi_prev'] = p['rsi_prev1']; btc['btc_rsi_prev6'] = p['rsi_prev6']
    btc['btc_adx'] = p['adx']; btc['btc_adx_prev'] = p['adx_prev1']
    btc['btc_atr_pct'] = p['atr'] / p['price'] * 100
    btc['btc_slope'] = (p['ema20'] - p['ema20_prev3']) / p['ema20_prev3'] * 100
    btc['btc_gap'] = (p['ema13'] - p['ema50']) / p['ema50'] * 100
    b1 = load_k(os.path.join(CACHE, 'btc_1h.csv'))
    e20 = EMAIndicator(close=b1['c'], window=20).ema_indicator()
    r1h = RSIIndicator(close=b1['c'], window=12).rsi()
    h = pd.DataFrame(index=b1.index)
    h['btc_1h_slope'] = (e20 - e20.shift(3)) / e20.shift(3) * 100
    h['btc_rsi_1h'] = r1h; h['btc_rsi_1h_prev'] = r1h.shift(1)
    # map each 5m bar to the latest COMPLETED 1h bar (close at bar+1h)
    h_completed = h.copy()
    h_completed.index = (h_completed.index + pd.Timedelta(hours=1)).astype('datetime64[ns]')
    btc = pd.merge_asof(btc.sort_index(), h_completed.sort_index(),
                        left_index=True, right_index=True)
    return btc

# ---- C/W pattern signatures (spec §10) ----
def cw_matched(r):
    def nn(*vals):  # None/nan -> pattern leg false
        return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in vals)
    rp, pg, dA, st = r['rng_pos'], r['pair_gap'], r['adx_delta'], r['stretch']
    pa, brsi, brsip = r['adx'], r['btc_rsi'], r['btc_rsi_prev']
    badx, badxp, bgap, batr = r['btc_adx'], r['btc_adx_prev'], r['btc_gap'], r['btc_atr_pct']
    e20s, e50s, pvr = r['p20_slope'], r['p50_slope'], r['pair_vol_ratio']
    C = [nn(rp, pg, dA) and rp >= 85 and pg >= 0.50 and dA >= 1.0,
         nn(brsi, brsip, badx, badxp, bgap) and brsi < brsip and badx < badxp and bgap < 0.05,
         nn(st, pa, rp) and st >= 0.40 and pa >= 30 and rp >= 85,
         nn(batr, badx, pa) and batr < 0.15 and badx < 22 and pa < 25,
         nn(pa, dA, e20s) and pa <= 22 and dA <= 0.3 and e20s <= 0.05,
         nn(brsi, badx, bgap) and brsi >= 65 and badx >= 28 and bgap >= 0.15,
         nn(pg, e50s, rp) and pg <= -0.50 and e50s <= -0.05 and rp >= 40,
         nn(rp, dA, pg, batr) and rp >= 75 and dA >= 1.0 and abs(pg) <= 0.20 and batr <= 0.15,
         nn(batr, badx, pa, pg) and batr <= 0.15 and badx <= 22 and pa <= 25 and pg <= -0.10]
    W = [nn(pa, dA, st) and pa >= 22 and dA >= 0.5 and st >= 0.16,
         nn(brsi, badx, bgap) and 50 <= brsi <= 65 and badx >= 22 and bgap >= 0.10,
         nn(batr, pvr, st) and batr >= 0.20 and pvr >= 1.20 and st >= 0.20,
         nn(rp, pg, dA) and 40 <= rp <= 75 and pg >= 0.10 and dA >= 0,
         nn(badx, brsi, pa, st) and 22 <= badx <= 30 and 55 <= brsi <= 65 and 22 <= pa <= 30 and 0.16 <= st <= 0.25,
         nn(badx, pg) and 22 <= badx < 26 and pg < 0.20]
    return any(C) or any(W)

# ---- band-rule helpers (spec §5 semantics) ----
def parse_cross(rule_str):
    rules = []
    for part in (rule_str or '').split(','):
        part = part.strip()
        if not part:
            continue
        rsi_s, adx_s = part.split(':')
        lo, hi = [float(x) for x in rsi_s.split('-')]
        toks = adx_s.split('-')
        if len(toks) == 1:
            rules.append((lo, hi, float(toks[0]), float('inf')))
        else:
            rules.append((lo, hi, float(toks[0]), float(toks[1])))
    return rules

def cross_blocks(rules, rsi, adx):
    for lo, hi, amin, amax in rules:
        if lo <= rsi < hi:
            return adx < amin or adx > amax
    return False

def parse_two_band(rule_str):
    out = []
    for part in (rule_str or '').split(','):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(':')
        a1, a2 = [float(x) for x in a.split('-')]
        b1, b2 = [float(x) for x in b.split('-')]
        out.append((a1, a2, b1, b2))
    return out

CROSS_L = parse_cross(getattr(TH, 'btc_rsi_adx_filter_long', ''))
ADXD_L = parse_two_band(getattr(TH, 'adx_delta_btc_adx_filter_long', ''))
RNGP_L = parse_two_band(getattr(TH, 'rngpos_adx_delta_filter_long', ''))
BGAP_L = parse_two_band(getattr(TH, 'btc_gap_btc_adx_filter_long', ''))
RSIATR_L = getattr(TH, 'btc_rsi_band_atr_block_long', '')
FAN_L = [(float(a), float(b)) for a, b in
         (p.split('-') for p in (getattr(TH, 'fan_ratio_block_long', '') or '').split(',') if p.strip())]

def rsi_atr_blocks(rule_str, rsi, atrp):
    for part in (rule_str or '').split(','):
        part = part.strip()
        if not part:
            continue
        band, spec = part.split(':')
        lo, hi = [float(x) for x in band.split('-')]
        if lo <= rsi < hi:
            if spec.startswith('<'):
                return atrp < float(spec[1:])
            if spec.startswith('>'):
                return atrp > float(spec[1:])
            a, b = [float(x) for x in spec.split('-')]
            return a <= atrp < b
    return False

def main():
    uni = pd.read_csv(os.path.join(CACHE, 'universe_daily_top50.csv'))
    uni['date'] = pd.to_datetime(uni['date_ms'], unit='ms')
    blacklist = set((getattr(TH, 'pair_blacklist', '') or '').replace(' ', '').split(','))
    no_trade = set((getattr(TH, 'no_trade_pairs', '') or 'BTCUSDT,ETHUSDT').replace(' ', '').split(','))
    btc = btc_panels()

    day_pairs = {d: g.sort_values('rank')['pair'].tolist() for d, g in uni.groupby('date')}
    ctx_only = no_trade  # BTC/ETH: in the volume/breadth panel, never candidates
    pair_cache = {}
    def get_pair(sym):
        if sym not in pair_cache:
            fp = os.path.join(CACHE, 'k5m', f'{sym}.csv')
            if not os.path.exists(fp):
                pair_cache[sym] = None
            else:
                raw = load_k(fp)
                s = pair_series(raw)
                s['p20_slope'] = (s['ema20'] - s['ema20_prev3']) / s['ema20_prev3'] * 100
                s['p50_slope'] = (s['ema50'] - s['ema50_prev12']) / s['ema50_prev12'] * 100
                pair_cache[sym] = s
        return pair_cache[sym]

    candidates = []
    last_entry = {}       # pair -> ts (episode dedup)
    calm_last = {}        # pair -> ts (CALM3D 90-min cooldown)
    days = sorted(day_pairs)
    for di, day in enumerate(days):
        pairs = [p for p in day_pairs[day] if p not in blacklist]
        frames = {}
        for p in pairs:
            s = get_pair(p) if p not in ctx_only else None
            if p in ctx_only:
                # context pairs come from the dedicated full-range files
                fp = os.path.join(CACHE, 'k5m', f'{p}.csv')
                s = get_pair(p) if os.path.exists(fp) else None
            if s is None:
                continue
            frames[p] = s.loc[day: day + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)]
        if not frames:
            continue
        # cross-sectional per-bar: global volume ratio + breadth
        vol_now = pd.DataFrame({p: f['candle_volume_raw'] for p, f in frames.items()})
        vol_avg = pd.DataFrame({p: f['avg_volume_global'] for p, f in frames.items()})
        gvr = vol_now.sum(axis=1, min_count=1) / vol_avg.sum(axis=1, min_count=1)
        e20 = pd.DataFrame({p: f['ema20'] for p, f in frames.items()})
        e20p = pd.DataFrame({p: f['ema20_prev3'] for p, f in frames.items()})
        slope_all = (e20 - e20p) / e20p * 100
        bull_pct = (slope_all > 0.02).sum(axis=1) / slope_all.notna().sum(axis=1) * 100

        for p, f in frames.items():
            if p in ctx_only:
                continue  # context pair: panel only, never a candidate
            # cheap pre-screen before calling the real get_signal
            pre = f[(f['ema5'] > f['ema8']) & (f['price'] > f['ema20'])
                    & (f['ema20'] > f['ema20_prev3'])
                    & f['rsi'].between(30, 80) & (f['adx'] <= 35)]
            for ts, r in pre.iterrows():
                le = last_entry.get(p)
                if le is not None and (ts - le).total_seconds() < EPISODE_MIN * 60:
                    continue
                if any(pd.isna(r[k]) for k in ('ema5', 'ema8', 'ema13', 'ema20', 'ema50', 'rsi', 'adx')):
                    continue
                sig, conf = get_signal(
                    ema5=r['ema5'], ema8=r['ema8'], ema13=r['ema13'], ema20=r['ema20'],
                    rsi=r['rsi'], adx=r['adx'], volume=r['volume'], avg_volume=r['avg_volume'],
                    price=r['price'], ema20_prev3=r['ema20_prev3'], ema50=r['ema50'],
                    ema50_prev12=r['ema50_prev12'], rsi_prev3=r['rsi_prev3'], rsi_prev2=r['rsi_prev2'],
                    ema5_prev1=r['ema5_prev1'], ema8_prev1=r['ema8_prev1'],
                    ema5_prev2=r['ema5_prev2'], ema8_prev2=r['ema8_prev2'],
                    ema13_prev1=r['ema13_prev1'], ema13_prev2=r['ema13_prev2'],
                    adx_prev1=r['adx_prev1'], high_20=r['high_20'], low_20=r['low_20'])
                if sig != 'LONG':
                    continue
                if ts not in btc.index:
                    continue
                b = btc.loc[ts]
                if pd.isna(b['btc_adx']) or pd.isna(b['btc_rsi']) or pd.isna(b['btc_slope']):
                    continue
                ind = {**{k: (None if pd.isna(v) else v) for k, v in r.items()}}
                atr_pct = r['atr'] / r['price'] * 100
                rng = ((r['price'] - r['low_20']) / (r['high_20'] - r['low_20']) * 100
                       if r['high_20'] and r['high_20'] != r['low_20'] else None)
                adx_d = r['adx'] - r['adx_prev1'] if not pd.isna(r['adx_prev1']) else None
                pair_gap = (r['ema13'] - r['ema50']) / r['ema50'] * 100
                stretch = abs(r['price'] - r['ema5']) / r['price'] * 100
                gvr_now = gvr.get(ts, np.nan)
                bull_now = bull_pct.get(ts, np.nan)

                block = None
                bmin = float(getattr(TH, 'btc_adx_min_long', 0) or 0)
                bmax = float(getattr(TH, 'btc_adx_max_long', 100) or 100)
                if bmin > 0 and b['btc_adx'] < bmin: block = 'BTC_ADX_LOW'
                elif bmax < 100 and b['btc_adx'] > bmax: block = 'BTC_ADX_HIGH'
                elif rsi_atr_blocks(RSIATR_L, b['btc_rsi'], b['btc_atr_pct']): block = 'BTC_RSI_ATR'
                elif cross_blocks(CROSS_L, b['btc_rsi'], b['btc_adx']): block = 'BTC_RSI_ADX_CROSS'
                elif adx_d is not None and any(a1 <= adx_d < a2 and b1 <= b['btc_adx'] < b2
                                               for a1, a2, b1, b2 in ADXD_L): block = 'ADX_DELTA_CROSS'
                elif rng is not None and adx_d is not None and any(
                        a1 <= rng <= a2 and b1 <= adx_d < b2 for a1, a2, b1, b2 in RNGP_L): block = 'RNGPOS_CROSS'
                if block is None and FAN_L:
                    g58 = abs((r['ema5'] - r['ema8']) / r['ema8'] * 100)
                    g813 = abs((r['ema8'] - r['ema13']) / r['ema13'] * 100)
                    if g813 > 0:
                        fan = g58 / g813
                        if any(a <= fan < bnd for a, bnd in FAN_L): block = 'FAN_RATIO'
                if block is None:
                    if float(getattr(TH, 'pair_atr_min_long', 0) or 0) > 0 and getattr(TH, 'pair_atr_filter_enabled', True) \
                            and atr_pct < float(TH.pair_atr_min_long): block = 'PAIR_ATR_MIN'
                    elif float(getattr(TH, 'pair_atr_max_long', 0) or 0) > 0 and atr_pct >= float(TH.pair_atr_max_long):
                        block = 'PAIR_ATR_MAX'
                    elif getattr(TH, 'atr_gap_block_long_enabled', False) \
                            and atr_pct >= float(getattr(TH, 'atr_gap_block_atr_min_long', 1.0)) \
                            and pair_gap >= float(getattr(TH, 'atr_gap_block_gap_min_long', 0.5)): block = 'ATR_GAP_LONG'
                    elif float(getattr(TH, 'rsi_prev_min_long', 0) or 0) > 0 and not pd.isna(r['rsi_prev1']) \
                            and r['rsi_prev1'] < float(TH.rsi_prev_min_long) \
                            and (float(getattr(TH, 'rsi_spike_min_jump_long', 0) or 0) <= 0
                                 or r['rsi'] - r['rsi_prev1'] >= float(TH.rsi_spike_min_jump_long)): block = 'RSI_SPIKE_GUARD'
                    elif any(a1 <= b['btc_gap'] < a2 and b1 <= b['btc_adx'] < b2
                             for a1, a2, b1, b2 in BGAP_L) and getattr(TH, 'btc_gap_btc_adx_filter_enabled', False):
                        block = 'BTC_GAP_CROSS'
                    elif float(getattr(TH, 'macro_trend_flat_threshold_long', 0) or 0) > 0 \
                            and b['btc_slope'] < float(TH.macro_trend_flat_threshold_long): block = 'BTC_SLOPE_GATE'
                    elif float(getattr(TH, 'btc_ema20_slope_max_long', 0) or 0) > 0 \
                            and abs(b['btc_slope']) > float(TH.btc_ema20_slope_max_long): block = 'BTC_SLOPE_MAX'
                    elif not pd.isna(b['btc_1h_slope']) and float(getattr(TH, 'btc_1h_slope_max_long', 0) or 0) > 0 \
                            and b['btc_1h_slope'] > float(TH.btc_1h_slope_max_long): block = 'BTC_1H_SLOPE_MAX'
                if block is None and not pd.isna(b['btc_1h_slope']):
                    db = float(getattr(TH, 'long_btc_1h_deadband', 0) or 0)
                    dbp = min(db, float(getattr(TH, 'long_btc_1h_deadband_pos', db) or db))
                    s1 = b['btc_1h_slope']
                    if db > 0 and ((0 <= s1 < dbp) or (-db < s1 < 0)): block = 'BTC1H_DEADBAND'
                if block is None:
                    if float(getattr(TH, 'momentum_ema20_slope_max_long', 0) or 0) > 0 \
                            and not pd.isna(r['p20_slope']) and abs(r['p20_slope']) > float(TH.momentum_ema20_slope_max_long):
                        block = 'PAIR_SLOPE_MAX'
                    elif getattr(TH, 'global_volume_filter_enabled', False) and not pd.isna(gvr_now) \
                            and gvr_now < float(getattr(TH, 'global_volume_threshold_long', 0)):
                        resc = float(getattr(TH, 'pair_volume_usd_rescue_long', 0) or 0)
                        vol24 = None  # 24h USD vol not tracked per bar; rescue leg approximated off
                        block = 'VOL_GATE'
                    elif getattr(TH, 'spike_guard_enabled', False) and r['candle_avg_volume_20'] \
                            and not pd.isna(r['candle_avg_volume_20']) and r['candle_avg_volume_20'] > 0 \
                            and r['candle_volume_raw'] / r['candle_avg_volume_20'] >= float(getattr(TH, 'spike_guard_volume_multiplier', 99)) \
                            and abs((r['price'] - r['candle_open']) / r['candle_open']) * 100 >= float(getattr(TH, 'spike_guard_price_move_pct', 99)):
                        block = 'SPIKE_GUARD'
                # ENTRY_QUALITY_SCORE
                calm3d = False
                if block is None and getattr(TH, 'entry_quality_score_filter_enabled', False):
                    gap520 = round(abs((r['ema5'] - r['ema20']) / r['price'] * 100), 4)
                    score = sum([55 <= r['rsi'] < 60, 20 <= r['adx'] < 25, 0.25 <= gap520 <= 0.50,
                                 (not pd.isna(bull_now)) and bull_now > 50,
                                 20 <= b['btc_adx'] < 25,
                                 (not pd.isna(r['p20_slope'])) and abs(r['p20_slope']) > 0.12])
                    if score <= int(getattr(TH, 'entry_quality_score_block_max', 1)): block = 'EQS'
                # promo router (gap-flat / gminflat / pair RSIxADX sole)
                if block is None:
                    ind_l = {'ema5': r['ema5'], 'ema8': r['ema8'], 'ema13': r['ema13'],
                             'ema20': r['ema20'], 'ema50': r['ema50'], 'price': r['price'],
                             'ema5_prev1': ind['ema5_prev1'], 'ema8_prev1': ind['ema8_prev1'],
                             'ema5_prev2': ind['ema5_prev2'], 'ema8_prev2': ind['ema8_prev2'],
                             'ema13_prev1': ind['ema13_prev1'], 'ema13_prev2': ind['ema13_prev2'],
                             'rsi': r['rsi'], 'adx': r['adx']}
                    routed = False
                    try:
                        if gminflat_band(ind_l, 'LONG', TH) is True or gap_expand_flat(ind_l, 'LONG', TH):
                            routed = True
                    except Exception:
                        routed = False
                    if routed:
                        regime = classify_btc_regime(b['btc_adx'], b['btc_rsi'], b['btc_slope'])
                        ok = (regime in (getattr(TH, 'nonexp_calm3d_regimes', 'STRONG_BULL') or 'STRONG_BULL').split(',')
                              and b['btc_atr_pct'] <= float(getattr(TH, 'nonexp_calm3d_btc_atr_max', 0.147))
                              and (stretch <= float(getattr(TH, 'nonexp_calm3d_max_stretch', 0.06))
                                   if float(getattr(TH, 'nonexp_calm3d_max_stretch', 0.06)) > 0 else True)
                              and (pd.isna(b['btc_1h_slope']) or b['btc_1h_slope'] > float(getattr(TH, 'nonexp_calm3d_b1h_min', 0.0)))
                              and (r['pos_di'] >= float(getattr(TH, 'nonexp_calm3d_min_pos_di', 28.0)) if not pd.isna(r['pos_di']) else True)
                              and (r['adx'] >= float(getattr(TH, 'nonexp_calm3d_min_pair_adx', 21.0))))
                        cl = calm_last.get(p)
                        if ok and cl is not None and (ts - cl).total_seconds() < float(getattr(TH, 'nonexp_calm3d_reentry_cooldown_min', 90)) * 60:
                            block = 'CALM3D_REENTRY'
                        elif ok:
                            calm3d = True
                            calm_last[p] = ts
                        else:
                            block = 'ROUTER_BLOCK'
                    if block is None and not calm3d:
                        rule = _rsi_adx_block_rule('LONG', r['rsi'], r['adx'], TH)
                        if rule is not None:
                            adm = float(getattr(TH, 'rsiadx_breadth_admit_max', 0) or 0)
                            if not (adm > 0 and not pd.isna(bull_now) and bull_now <= adm):
                                block = 'PAIR_RSI_ADX_CROSS'
                # keep-only-unmatched (CALM3D bypasses)
                if block is None and not calm3d and getattr(TH, 'long_unmatched_only', False):
                    row = {'rng_pos': rng, 'pair_gap': pair_gap, 'adx_delta': adx_d, 'stretch': stretch,
                           'adx': r['adx'], 'btc_rsi': b['btc_rsi'], 'btc_rsi_prev': b['btc_rsi_prev'],
                           'btc_adx': b['btc_adx'], 'btc_adx_prev': b['btc_adx_prev'],
                           'btc_gap': b['btc_gap'], 'btc_atr_pct': b['btc_atr_pct'],
                           'p20_slope': r['p20_slope'], 'p50_slope': r['p50_slope'],
                           'pair_vol_ratio': (r['volume'] / r['avg_volume']) if r['avg_volume'] else None}
                    if cw_matched(row):
                        block = 'LONG_UNMATCHED_ONLY'
                if block is not None:
                    continue
                # ---- accepted candidate: score forward ----
                fseries = get_pair(p)
                fwd = fseries.loc[ts:].iloc[1:FWD_BARS + 1]
                entry = r['price']
                outcome, final = 'TIMEOUT', None
                for _, fr in fwd.iterrows():
                    hi_p = (fr['candle_high'] - entry) / entry * 100
                    lo_p = (fr['candle_low'] - entry) / entry * 100
                    if lo_p <= ARM_LOSS:
                        outcome = 'LOSS'; break
                    if hi_p >= ARM_WIN:
                        outcome = 'WIN'; break
                if len(fwd):
                    final = (fwd['price'].iloc[-1] - entry) / entry * 100
                last_entry[p] = ts
                candidates.append({
                    'ts': ts, 'pair': p, 'confidence': conf, 'calm3d': calm3d,
                    'regime': classify_btc_regime(b['btc_adx'], b['btc_rsi'], b['btc_slope']),
                    'btc_rsi': round(b['btc_rsi'], 1), 'btc_adx': round(b['btc_adx'], 1),
                    'bull_pct': None if pd.isna(bull_now) else round(bull_now, 1),
                    'outcome': outcome, 'fwd_final': None if final is None else round(final, 3),
                    'month': ts.strftime('%Y-%m')})
        # free memory for pairs leaving the universe
        if di % 7 == 0:
            keep = set()
            for dd in days[max(0, di - 1): di + 8]:
                keep.update(day_pairs.get(dd, []))
            for k in list(pair_cache):
                if k not in keep:
                    del pair_cache[k]
        if di % 14 == 0:
            print(f"  {di}/{len(days)} {day.date()} candidates so far: {len(candidates)}", flush=True)

    out = pd.DataFrame(candidates)
    out.to_csv(os.path.join(CACHE, 'phase1_candidates.csv'), index=False)
    print(f"TOTAL candidates: {len(out)}", flush=True)
    if len(out):
        for m, g in out.groupby('month'):
            wr = (g['outcome'] == 'WIN').mean() * 100
            print(f"  {m}: N={len(g)} armWR={wr:.0f}% avg6h={g['fwd_final'].mean():+.3f}%", flush=True)

if __name__ == '__main__':
    main()
