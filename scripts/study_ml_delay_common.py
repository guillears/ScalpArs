"""ML delayed-entry BTC-confirmation study (2026-10-15) — shared data access, signal recompute and replica wiring.

Pre-registration: reports/ML_DELAY_BTC_PREREG.txt (write-once, before pricing).
Read-only on every input. Imports the momentum-long exit replica scripts/ml_exit_optimize_yr5.py (variant BASE = today's live
ML exit stack) and RE-POINTS its data accessors (in this process only) at both kline caches (2025 + 2026), the ML_WL_1015
top-up and this study's own top-up (reports/ml_delay_fetch/, outside backtest_cache). No file of the replica is modified.
Signal validity uses services.indicators.calculate_indicators + get_signal (read-only import) with the frozen yr6 config held
in memory (trading_config.json is never written).
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                     # config.py resolves trading_config.json relative to cwd (read only)

import ml_exit_optimize_yr5 as RX                   # noqa: E402  the replica (read-only import)

C26 = os.path.join(ROOT, "reports/backtest_cache")
C25 = os.path.join(ROOT, "reports/backtest_cache_2025")
TOP1 = os.path.join(ROOT, "reports/ml_wl_1015_fetch")
TOP2 = os.path.join(ROOT, "reports/ml_delay_fetch")
FROZEN = os.path.join(C26, "replay/frozen_config_yr6_01419d4.json")
FILLS = os.path.join(ROOT, "reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
MASTER = os.path.join(ROOT, "reports/MASTER_POOL_stacked.csv")
ROWS = os.path.join(ROOT, "reports/ML_DELAY_rows.csv")
DAY = 86_400_000
MIN1, MIN5 = 60_000, 300_000
DS = (5, 10, 15)
XS = (0.05, 0.10, 0.15)
TAKER, MAKER = 0.045, 0.018


# ───────────────────────── data access (both caches + top-ups) ─────────────────────────
_TICK, _K1, _K5O = {}, {}, {}


def tick_day(pair, day_ms):
    key = (pair, day_ms)
    if key in _TICK:
        return _TICK[key]
    ds = time.strftime('%Y-%m-%d', time.gmtime(day_ms / 1000))
    out = None
    for base in (C26, C25):
        for sub in ('ticks_q', 'ticks'):
            fp = os.path.join(base, sub, pair, f'{ds}.npz')
            if os.path.exists(fp):
                try:
                    z = np.load(fp)
                    out = (z['t'].astype(np.int64), z['p'].astype(np.float64),
                           (z['q'].astype(np.float64) if 'q' in z.files else None))
                    break
                except Exception:
                    out = None
        if out is not None:
            break
    if len(_TICK) > 16:
        _TICK.pop(next(iter(_TICK)))
    _TICK[key] = out
    return out


def k1(pair):
    """all 1m bars (open_time, o, h, l, c, vol) from every cache dir, de-duplicated."""
    if pair in _K1:
        return _K1[pair]
    fr = []
    for base in (C25, C26, TOP1, TOP2):
        for sub in ('k1m', 'k1m_ondemand'):
            fp = os.path.join(base, sub, f'{pair}.csv')
            if os.path.exists(fp):
                try:
                    fr.append(pd.read_csv(fp, usecols=['open_time', 'o', 'h', 'l', 'c', 'vol']))
                except Exception:
                    pass
        for fp in glob.glob(os.path.join(base, 'k1m_spike', f'{pair}*.csv')):
            try:
                fr.append(pd.read_csv(fp, usecols=['open_time', 'o', 'h', 'l', 'c', 'vol']))
            except Exception:
                pass
    df = None
    if fr:
        df = pd.concat(fr)
        for c in ('open_time', 'o', 'h', 'l', 'c', 'vol'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['open_time', 'o', 'h', 'l', 'c'])
        df = df.astype({'open_time': np.int64}).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
        df['vol'] = df.vol.fillna(0.0)
    if len(_K1) > 6:
        _K1.pop(next(iter(_K1)))
    _K1[pair] = df
    return df


def k5ohlcv(pair):
    """closed 5m OHLCV (k5m_full both caches + top-ups), gaps filled with 1m-aggregated bars."""
    if pair in _K5O:
        return _K5O[pair]
    fr = []
    for base in (C25, C26, TOP1, TOP2):
        for sub in ('k5m_full', 'k5m'):
            fp = os.path.join(base, sub, f'{pair}.csv')
            if os.path.exists(fp):
                try:
                    fr.append(pd.read_csv(fp, usecols=['open_time', 'o', 'h', 'l', 'c', 'vol']))
                except Exception:
                    pass
    df = None
    if fr:
        df = pd.concat(fr)
        for c in ('open_time', 'o', 'h', 'l', 'c', 'vol'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['open_time', 'o', 'h', 'l', 'c']).astype({'open_time': np.int64}).drop_duplicates('open_time')
        df['vol'] = df.vol.fillna(0.0)
    a = k1(pair)
    if a is not None and len(a):
        g = a.assign(b=(a.open_time // MIN5) * MIN5).groupby('b').agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'),
                                                                        c=('c', 'last'), vol=('vol', 'sum'), n=('o', 'size'))
        g = g[g.n == 5].drop(columns='n').reset_index().rename(columns={'b': 'open_time'})
        df = g if df is None else pd.concat([df, g[~g.open_time.isin(df.open_time)]])
    if df is not None and len(df):
        df = df.astype({'open_time': np.int64}).sort_values('open_time').drop_duplicates('open_time').reset_index(drop=True)
        out = dict(ot=df.open_time.values.astype(np.int64), o=df.o.values.astype(float), h=df.h.values.astype(float),
                   l=df.l.values.astype(float), c=df.c.values.astype(float), v=df.vol.values.astype(float))
    else:
        out = None
    if len(_K5O) > 6:
        _K5O.pop(next(iter(_K5O)))
    _K5O[pair] = out
    return out


def _k5_for_replica(pair):
    """the replica's _k5 contract: closed 5m closes + EMA5/8/13/20 (adjust=False)."""
    x = k5ohlcv(pair)
    if x is None:
        return None
    c = pd.Series(x['c'])
    out = dict(ot=x['ot'], c=x['c'])
    for n in (5, 8, 13, 20):
        out[f'e{n}'] = c.ewm(span=n, adjust=False).mean().values
    return out


_K5R = {}


def _k5_cached(pair):
    if pair not in _K5R:
        if len(_K5R) > 6:
            _K5R.pop(next(iter(_K5R)))
        _K5R[pair] = _k5_for_replica(pair)
    return _K5R[pair]


def _k1_for_replica(pair):
    return k1(pair)


def _day_for_replica(pair, day_ms):
    d = tick_day(pair, day_ms)
    return None if d is None else (d[0], d[1])


def _btc5():
    fr = [pd.read_csv(os.path.join(C25, 'k5m_full', 'BTCUSDT.csv'), usecols=['open_time', 'o', 'h', 'l', 'c']),
          pd.read_csv(os.path.join(C26, 'btc_5m.csv'), usecols=['open_time', 'o', 'h', 'l', 'c']),
          pd.read_csv(os.path.join(C26, 'k5m_full', 'BTCUSDT.csv'), usecols=['open_time', 'o', 'h', 'l', 'c'])]
    for b in (TOP1, TOP2):
        fp = os.path.join(b, 'k5m_full', 'BTCUSDT.csv')
        if os.path.exists(fp):
            fr.append(pd.read_csv(fp, usecols=['open_time', 'o', 'h', 'l', 'c']))
    b = pd.concat(fr).drop_duplicates('open_time').sort_values('open_time')
    return b[['open_time', 'o', 'h', 'l', 'c']].values.astype(float)


def wire_replica():
    RX._day = _day_for_replica
    RX._k1 = _k1_for_replica
    RX._k5 = _k5_cached
    RX._BTC = _btc5()


# ───────────────────────── prices ─────────────────────────
def price_at(pair, T):
    """last traded price at/before T (ticks, same or previous day); else close of the last 1m bar CLOSED at/before T.
    Returns (price, src)."""
    for dms in ((T // DAY) * DAY, (T // DAY) * DAY - DAY):
        d = tick_day(pair, dms)
        if d is not None and len(d[0]):
            i = np.searchsorted(d[0], T, side='right') - 1
            if i >= 0 and d[0][i] >= T - 5 * MIN1:
                return float(d[1][i]), 'tick'
    a = k1(pair)
    if a is not None and len(a):
        ot = a.open_time.values
        i = np.searchsorted(ot, T - MIN1, side='right') - 1
        if i >= 0 and ot[i] >= T - 4 * MIN1:
            return float(a.c.values[i]), '1m'
    return np.nan, 'none'


_BTC1 = None


def btc_close_at(T):
    """BTCUSDT close of the last 1m bar closed at/before T."""
    global _BTC1
    if _BTC1 is None:
        a = k1('BTCUSDT')
        _BTC1 = (a.open_time.values.astype(np.int64), a.c.values.astype(float))
    ot, c = _BTC1
    i = np.searchsorted(ot, T - MIN1, side='right') - 1
    if i < 0 or ot[i] < T - 4 * MIN1:
        return np.nan
    return float(c[i])


# ───────────────────────── signal recompute ─────────────────────────
_TH = None


def _engine():
    global _TH
    import config as config_module
    from services import indicators as IND
    if _TH is None:
        fz = json.load(open(FROZEN))
        try:
            config_module.trading_config = config_module.TradingConfig(**fz)      # in memory only
        except Exception:
            pass
        _TH = config_module.trading_config.thresholds
    return IND, _TH


def ohlcv_at(pair, T, limit=100):
    """the engine's 5m read at T: (limit-1) closed bars + the forming bar [cur_open, T) from ticks_q (with qty) else 1m."""
    x = k5ohlcv(pair)
    if x is None:
        return None, 'none'
    cur = (T // MIN5) * MIN5
    n_done = int(np.searchsorted(x['ot'], cur))
    if n_done < 60 or x['ot'][n_done - 1] < cur - 3 * MIN5:
        return None, 'none'
    lo = max(0, n_done - (limit - 1))
    rows = [[int(x['ot'][i]), x['o'][i], x['h'][i], x['l'][i], x['c'][i], x['v'][i]] for i in range(lo, n_done)]
    d = tick_day(pair, (cur // DAY) * DAY)
    if d is not None and d[2] is not None and (T - 1) // DAY == cur // DAY:
        a, b = np.searchsorted(d[0], cur), np.searchsorted(d[0], T)
        if b > a:
            rows.append([cur, float(d[1][a]), float(d[1][a:b].max()), float(d[1][a:b].min()), float(d[1][b - 1]),
                         float(d[2][a:b].sum())])
        else:
            px = rows[-1][4]; rows.append([cur, px, px, px, px, 0.0])
        return rows, 'tick'
    a1 = k1(pair)
    if a1 is not None and len(a1):
        ot = a1.open_time.values
        m_end = (T // MIN1) * MIN1
        a, b = np.searchsorted(ot, cur), np.searchsorted(ot, m_end)
        if b > a:
            rows.append([cur, float(a1.o.values[a]), float(a1.h.values[a:b].max()), float(a1.l.values[a:b].min()),
                         float(a1.c.values[b - 1]), float(a1.vol.values[a:b].sum())])
        else:
            px = rows[-1][4]; rows.append([cur, px, px, px, px, 0.0])
        return rows, '1m'
    return rows, 'closed_only'


def signal_at(pair, T):
    """→ dict(valid=get_signal=='LONG', proxy=stack 5>8>13>20 ∧ price>EMA5, atr_pct, src); None fields when no data."""
    IND, th = _engine()
    rows, src = ohlcv_at(pair, T)
    if rows is None:
        return dict(valid=np.nan, proxy=np.nan, atr_pct=np.nan, sig_src=src)
    ind = IND.calculate_indicators(rows)
    if not ind:
        return dict(valid=np.nan, proxy=np.nan, atr_pct=np.nan, sig_src=src)
    try:
        s, _ = IND.get_signal(
            ema5=ind.get('ema5'), ema8=ind.get('ema8'), ema13=ind.get('ema13'), ema20=ind.get('ema20'),
            rsi=ind.get('rsi'), adx=ind.get('adx'), volume=ind.get('volume'), avg_volume=ind.get('avg_volume'),
            price=ind.get('price'), config=th, ema20_prev3=ind.get('ema20_prev3'), ema50=ind.get('ema50'),
            ema50_prev12=ind.get('ema50_prev12'), rsi_prev3=ind.get('rsi_prev3'), rsi_prev2=ind.get('rsi_prev2'),
            ema5_prev1=ind.get('ema5_prev1'), ema8_prev1=ind.get('ema8_prev1'), ema5_prev2=ind.get('ema5_prev2'),
            ema8_prev2=ind.get('ema8_prev2'), ema13_prev1=ind.get('ema13_prev1'), ema13_prev2=ind.get('ema13_prev2'),
            adx_prev1=ind.get('adx_prev1'), high_20=ind.get('high_20'), low_20=ind.get('low_20'))
        valid = float(s == 'LONG')
    except Exception:
        valid = np.nan
    e5, e8, e13, e20, p = (ind.get(k) for k in ('ema5', 'ema8', 'ema13', 'ema20', 'price'))
    proxy = float(all(v is not None for v in (e5, e8, e13, e20, p)) and e5 > e8 > e13 > e20 and p > e5)
    atr = ind.get('atr')
    return dict(valid=valid, proxy=proxy, atr_pct=(atr / p * 100 if atr and p else np.nan), sig_src=src)


# ───────────────────────── populations ─────────────────────────
def _ms(s):
    return RX._ms(s)


def load_replay():
    d = pd.read_csv(FILLS, low_memory=False)
    d = d[(d.sleeve == 'MOM-long') & (d.status == 'CLOSED')]
    d = d[~d.is_probe.fillna(False).astype(bool)].copy()
    d['o_ms'] = _ms(d.opened_at)
    d['key'] = 'r' + d.seed.astype(str) + '|' + d.opened_at.astype(str) + '|' + d.pair
    d['block'] = d.regime
    d['today_engine'] = d.pct_new.astype(float)
    d['batch'] = d.regime
    return d


def load_master():
    m = pd.read_csv(MASTER, low_memory=False)
    m = m[(m.status == 'CLOSED') & ~m.is_probe.fillna(False).astype(bool)]
    m = m[m.entry_strategy.astype(str).str.startswith('MOMENTUM') & (m.direction == 'LONG')].copy()
    m['o_ms'] = _ms(m.opened_at)
    m['key'] = 'm|' + m.opened_at.astype(str) + '|' + m.pair
    m['block'] = 'master'
    m['today_engine'] = m.pnl_percentage.astype(float)          # as traded
    m['stack_pct'] = pd.to_numeric(m.stack_pct, errors='coerce')
    m['stack_keep'] = m.stack_keep.astype(str).eq('True')
    m['batch'] = m.era
    return m


def load_live():
    from export_glob import glob_exports
    fs = [f for f in glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")) if "_live_" in f]
    if not fs:
        return pd.DataFrame()
    L = pd.concat([pd.read_csv(f, low_memory=False) for f in fs])
    L = L[(L.status == 'CLOSED') & (L.direction == 'LONG') & (L.entry_strategy.astype(str) == 'MOMENTUM')]
    L = L.drop_duplicates(['opened_at', 'pair', 'direction']).copy()
    L['o_ms'] = _ms(L.opened_at)
    L['key'] = 'l|' + L.opened_at.astype(str) + '|' + L.pair
    L['block'] = 'live'
    L['today_engine'] = L.pnl_percentage.astype(float)
    L['batch'] = 'live'
    return L


def fx_of(r, o_ms, E, atr, rsi):
    fee_in = MAKER if str(r.get('entry_order_type', '')).upper().startswith('MAKER') else TAKER
    return dict(key=r['key'], pair=r['pair'], o_ms=int(o_ms), E=float(E), fee_in=fee_in,
                atr=(float(atr) if atr is not None and np.isfinite(atr) and atr > 0 else None),
                rsi_entry=(float(rsi) if rsi is not None and np.isfinite(rsi) else None))


# ───────────────────────── stats ─────────────────────────
def day_boot(y, day, n=2000, seed=7):
    y = np.asarray(y, float); day = np.asarray(day)
    if len(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, weights=y); c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(y.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean())
