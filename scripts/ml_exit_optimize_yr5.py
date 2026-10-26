#!/usr/bin/env python3
"""MOMENTUM-LONG EXIT OPTIMIZATION on the CORRECTED year replay yr5 (2026-10-05). Parameterised copy of
scripts/ml_exit_optimize.py (yr4, 2026-10-04): same replica, same pre-declared grid, PLUS three new pre-declared families
(NL = no-lock runner, AA = ATR-scaled arm, GF = giveback as a fraction of peak floored at 0) written before any yr5 variant run.
yr5 = frozen_config_yr5_181131e.json (exit keys identical to yr4's cd5bc3e — only frenzy_* differ), code 181131e (pre chop∧burst
block → removed here post-hoc exactly as the yr4 study).
Usage: venv/bin/python scripts/ml_exit_optimize_yr5.py --jobs 10   then   --report
Outputs: reports/ML_EXIT_OPT_variants_yr5.csv, ML_EXIT_OPT_variants_master_YR5.csv, ML_EXIT_OPT_yr5_cohort.csv,
         ML_EXIT_OPT_tables_YR5.md, ML_EXIT_OPT_summary_YR5.csv

--- original docstring (yr4) ---
MOMENTUM-LONG EXIT OPTIMIZATION (2026-10-04) — tick-level exit REPLICA of today's live ML exit stack + one-at-a-time
exit-family counterfactuals on the yr4 year replay (today's exact config, 3 seeds) WITHOUT the CHOP∧BURST cohort, then the
master-batch re-pricing (before as traded / replica baseline / candidate).

Live momentum-LONG exit stack at HEAD (services/trading_engine.py check_realtime_stop_loss + update_open_positions; frozen config
reports/backtest_cache/replay/frozen_config_yr4_cd5bc3e.json) — every threshold is on NET P&L % (entry fee: maker 0.018 % /
taker 0.045 %; exit fee estimated taker 0.045 % of the current notional):
  ① STOP          base = STRONG_BUY stop_loss −0.70, or signal_active_sl −1.00 while the 5m core signal is still on
                  (EMA5 > EMA8 ∧ price > EMA20 at the last scan → reason STOP_LOSS_WIDE); widened to −1.5 × entry ATR %, the
                  widening capped at −1.2 (sl_atr_widen_floor_pct). Quiet-pair −2.0 stop is OFF (momentum_long_sl_atr_threshold 0).
                  BE levels OFF for STRONG_BUY/VERY_STRONG (be_levels_enabled False).
  ② RECOVERY HOLD (armed Oct-2): a STOP_LOSS* hit while BTC closed-5m RSI(14) ≥ its entry value ∧ 60 ≤ RSI ≤ 66 ∧ peak < 0.40
                  is held: RH_HARD_STOP at stop − 0.5 · RH_PREMISE_EXIT when RSI < 60 or < entry · RH_TIME_EXIT ≥ 30 min ∧ P&L < 0
                  or ≥ 240 min · release at peak ≥ +0.40 → normal stack (closes RH_-prefixed).
  ③ HARD-TP LADDER "1.25:0.25,1.5:0.30,2.0:0.40,3.0:0.60,4.0:0.80" → profit floors 1.00/1.20/1.60/2.40/3.20 once the peak crossed
                  each trigger (previous-tick peak).
  ④ RUNNER TRAIL  armed at peak ≥ 0.40 (−0.005 tolerance): floor = max(peak − 1.0 × entry ATR %, +0.10 BE-lock). giveback_frac 0.
                  The legacy tight trailing is suppressed for every runner-eligible momentum long (atr_min 0).
  ⑤ NO_EXPANSION  at 180 min (peak < BE_L1 trigger 99 → always; the signal-reset never fired in yr4: max hold 180.03 min).
  OFF: EMA13-cross exit (long), fast exit, signal-lost exit, tick-momentum exit, regime-change exit, BE levels, fixed TP.

Usage:  venv/bin/python scripts/ml_exit_optimize.py [--jobs 10] [--only replica]
Outputs: reports/ML_EXIT_OPT_replica_yr4.csv, ML_EXIT_OPT_replica_master.csv, ML_EXIT_OPT_variants_yr4.csv (long: fill × variant),
         ML_EXIT_OPT_variants_master.csv, then the report is written by --report.
"""
import os, sys, glob, json, math, argparse, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from services.recovery_hold import closed_rsi   # the live RH ruler (pure)

CACHE = os.path.join(ROOT, 'reports', 'backtest_cache')
REP = os.path.join(ROOT, 'reports')
DAY = 86_400_000
HORIZON_MS = 1200 * 60_000          # max_holding_time_minutes
SCAN_S = 111.6                       # replay scan cadence (synthetic 111.6 s) — PairData / signal_active refresh
TAKER, MAKER = 0.045, 0.018

LADDER_LIVE = [(1.25, .25), (1.5, .30), (2.0, .40), (3.0, .60), (4.0, .80)]
BASE = dict(sl_base=-0.70, sl_sig=-1.00, sl_atr=1.5, sl_cap=-1.2, sl_fixed=None,
            rh=True, run_arm=0.40, run_g_atr=1.0, run_g_fixed=None, run_g_frac=None, run_lock=0.10,
            be_arm=None, be_lock=None, ladder=LADDER_LIVE, tp=None, doa=None, ema13=False, noexp=180, maxhold=1200,
            sl_quiet=None, run_arm_atr=None, run_g_frac_only=None)
PREFIX = 'yr5'
OUT_SFX = '_YR5'


def V(**kw):
    p = dict(BASE); p.update(kw); return p


# ───────────── PRE-DECLARED GRID (written before any variant was run; see report §3) ─────────────
VARIANTS = {
    'BASE': V(),
    # A — stop width (fixed and ATR-scaled; live-stopped-only reading applies)
    'SL_fix-0.5': V(sl_fixed=-0.5), 'SL_fix-0.6': V(sl_fixed=-0.6), 'SL_fix-0.9': V(sl_fixed=-0.9),
    'SL_fix-1.2': V(sl_fixed=-1.2), 'SL_noSigWide': V(sl_sig=-0.70), 'SL_atr1.0': V(sl_atr=1.0),
    'SL_atr2.0cap-1.5': V(sl_atr=2.0, sl_cap=-1.5),
    # B — BE lock (runner lock level / an earlier BE stop raise)
    'BE_none': V(run_lock=None), 'BE_lock0.0': V(run_lock=0.0), 'BE_lock0.2': V(run_lock=0.20),
    'BE_arm0.25lock-0.1': V(be_arm=0.25, be_lock=-0.10), 'BE_arm0.20lock0.0': V(be_arm=0.20, be_lock=0.0),
    'BE_arm0.30lock0.05': V(be_arm=0.30, be_lock=0.05),
    # C — runner giveback
    'TR_atr0.5': V(run_g_atr=0.5), 'TR_atr0.75': V(run_g_atr=0.75), 'TR_atr1.5': V(run_g_atr=1.5),
    'TR_atr2.0': V(run_g_atr=2.0), 'TR_fix0.2': V(run_g_fixed=0.20), 'TR_fix0.3': V(run_g_fixed=0.30),
    'TR_frac0.35': V(run_g_frac=0.35), 'TR_frac0.5': V(run_g_frac=0.50),
    # D — arm level
    'ARM_0.25': V(run_arm=0.25), 'ARM_0.30': V(run_arm=0.30), 'ARM_0.50': V(run_arm=0.50), 'ARM_0.60': V(run_arm=0.60),
    'ARM_0.80': V(run_arm=0.80),
    # E — fixed take-profit (on top of the stack)
    'TP_0.4': V(tp=0.4), 'TP_0.6': V(tp=0.6), 'TP_0.8': V(tp=0.8), 'TP_1.0': V(tp=1.0), 'TP_1.5': V(tp=1.5),
    # F — hard-TP ladder variants
    'LAD_off': V(ladder=[]), 'LAD_tight': V(ladder=[(0.8, .20), (1.0, .25), (1.25, .25), (1.5, .30), (2.0, .40), (3.0, .60)]),
    'LAD_loose': V(ladder=[(2.0, .50), (3.0, .75), (4.0, 1.0)]),
    # G — time stop / dead-on-arrival
    'DOA_5m<0.10': V(doa=(5, 0.10)), 'DOA_10m<0.20': V(doa=(10, 0.20)), 'DOA_15m<0.20': V(doa=(15, 0.20)),
    'DOA_30m<0.30': V(doa=(30, 0.30)), 'HOLD_60m': V(noexp=60), 'HOLD_120m': V(noexp=120),
    # H — EMA13-cross exit / recovery hold toggle
    'EMA13_on': V(ema13=True), 'RH_off': V(rh=False),
    # yr4's post-hoc stop dose-response + combinations are carried over as PRE-DECLARED on yr5 (fixed before the yr5 run)
    'SL_fix-1.0': V(sl_fixed=-1.0), 'SL_fix-1.1': V(sl_fixed=-1.1), 'SL_fix-1.35': V(sl_fixed=-1.35), 'SL_fix-1.5': V(sl_fixed=-1.5),
    'C_SL12+TP10': V(sl_fixed=-1.2, tp=1.0), 'C_TP10+TRf05': V(tp=1.0, run_g_frac=0.5),
    'C_SL12+TP10+TRf05': V(sl_fixed=-1.2, tp=1.0, run_g_frac=0.5),
    # ── NEW pre-declared families (2026-10-05, written before any yr5 variant run) ──
    # NL — "no lock" runner (SURGE_LONG Oct-5 lesson): once armed at +0.40 the line trails peak − k×ATR, floored ONLY at the
    #      initial stop (the +0.10 lock removed). Ladder kept (NL_k*) and ladder also removed (NL_k*_noLAD, the SURGE shape).
    'NL_k0.5': V(run_lock=None, run_g_atr=0.5), 'NL_k1.0': V(run_lock=None, run_g_atr=1.0), 'NL_k1.5': V(run_lock=None, run_g_atr=1.5),
    'NL_k0.5_noLAD': V(run_lock=None, run_g_atr=0.5, ladder=[]), 'NL_k1.0_noLAD': V(run_lock=None, run_g_atr=1.0, ladder=[]),
    'NL_k1.5_noLAD': V(run_lock=None, run_g_atr=1.5, ladder=[]),
    # AA — ATR-scaled arm: arm = m × entry ATR % (fallback 0.40 when ATR missing); lock +0.10 kept (AA_m*) / removed (AA_m*_NL)
    'AA_m0.5': V(run_arm_atr=0.5), 'AA_m1.0': V(run_arm_atr=1.0),
    'AA_m0.5_NL': V(run_arm_atr=0.5, run_lock=None), 'AA_m1.0_NL': V(run_arm_atr=1.0, run_lock=None),
    # GF — giveback as a fraction of the peak, floor at 0 (replaces the ATR giveback): line = max(peak × (1 − f), 0.0) once armed
    'GF_0.3': V(run_g_frac_only=0.3, run_lock=0.0), 'GF_0.5': V(run_g_frac_only=0.5, run_lock=0.0),
    'GF_0.7': V(run_g_frac_only=0.7, run_lock=0.0),
}
N_PREDECLARED = len(VARIANTS) - 1
# ───────────── POST-HOC (added after the pre-declared grid was read — labelled as such in the report) ─────────────
POSTHOC = set()   # filled only if variants are added after reading the yr5 grid (labelled in the report)
FAMILY = {k: (k.split('_')[0] if k != 'BASE' else 'BASE') for k in VARIANTS}


# ───────────────────────────── data access (per worker caches) ─────────────────────────────
_DAYS, _K5, _K1 = {}, {}, {}
_BTC = None


def _day(pair, day_ms):
    key = (pair, day_ms)
    if key in _DAYS:
        return _DAYS[key]
    ds = time.strftime('%Y-%m-%d', time.gmtime(day_ms / 1000))
    out = None
    for sub in ('ticks_q', 'ticks'):
        fp = os.path.join(CACHE, sub, pair, f'{ds}.npz')
        if os.path.exists(fp):
            try:
                z = np.load(fp); out = (z['t'].astype(np.int64), z['p'].astype(np.float64)); break
            except Exception:
                out = None
    if len(_DAYS) > 12:
        _DAYS.pop(next(iter(_DAYS)))
    _DAYS[key] = out
    return out


def _k1(pair):
    if pair in _K1:
        return _K1[pair]
    fr = []
    for sub in ('k1m', 'k1m_ondemand'):
        fp = os.path.join(CACHE, sub, f'{pair}.csv')
        if os.path.exists(fp):
            fr.append(pd.read_csv(fp))
    for fp in glob.glob(os.path.join(CACHE, 'k1m_spike', f'{pair}*.csv')):
        fr.append(pd.read_csv(fp))
    df = (pd.concat(fr).drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True) if fr else None)
    if len(_K1) > 6:
        _K1.pop(next(iter(_K1)))
    _K1[pair] = df
    return df


def _k5(pair):
    """closed 5m bars (open_time, close) + EMA5/8/13/20 on closes (adjust=False, as ta's EMAIndicator)."""
    if pair in _K5:
        return _K5[pair]
    fp = os.path.join(CACHE, 'k5m_full', f'{pair}.csv')
    df = pd.read_csv(fp, usecols=['open_time', 'c']) if os.path.exists(fp) else None
    k1 = _k1(pair)
    if k1 is not None and len(k1):     # extend / fill with 1m-aggregated 5m bars where k5m_full is missing
        g = k1.assign(b=(k1.open_time // 300_000) * 300_000).groupby('b').c.last().reset_index().rename(columns={'b': 'open_time'})
        df = g if df is None else pd.concat([df, g[~g.open_time.isin(df.open_time)]]).sort_values('open_time')
    if df is None or not len(df):
        _K5[pair] = None; return None
    df = df.drop_duplicates('open_time').sort_values('open_time').reset_index(drop=True)
    c = df.c.astype(float)
    out = dict(ot=df.open_time.values.astype(np.int64), c=c.values)
    for n in (5, 8, 13, 20):
        out[f'e{n}'] = c.ewm(span=n, adjust=False).mean().values
    if len(_K5) > 6:
        _K5.pop(next(iter(_K5)))
    _K5[pair] = out
    return out


def _btc():
    global _BTC
    if _BTC is None:
        b = pd.read_csv(os.path.join(CACHE, 'btc_5m.csv'))
        _BTC = b[['open_time', 'o', 'h', 'l', 'c']].values.astype(float)
    return _BTC


def btc_rsi_at(now_ms):
    """the live reading: RSI(14) on the CLOSED bars of the last 100 5m bars fetched at now (forming bar dropped)."""
    b = _btc()
    j = np.searchsorted(b[:, 0], now_ms, side='right')
    bars = [(int(r[0]), r[1], r[2], r[3], r[4]) for r in b[max(0, j - 100):j]]
    r, ts = closed_rsi(bars, now_ms)
    return r if (ts is not None and 0 <= now_ms - ts <= 2.5 * 300_000) else None


def _compress(t, p):
    """drop consecutive repeated prices (lossless for every level crossing AND its exit price). A per-second first/max/min/last
    compression was tried first and rejected: it keeps the crossing second but not the first tick below the level, so the
    exit price came out at the second's low (replica −0.02 %/trade vs the engine)."""
    keep = np.r_[True, p[1:] != p[:-1]] if len(p) > 1 else np.ones(len(p), bool)
    return t[keep], p[keep]


def _compress_seconds(t, p):   # rejected (see _compress) — kept for the record
    if len(t) < 3:
        return t, p
    sec = t // 1000
    st = np.r_[0, np.flatnonzero(np.diff(sec)) + 1]
    en = np.r_[st[1:] - 1, len(t) - 1]
    gmax = np.maximum.reduceat(p, st); gmin = np.minimum.reduceat(p, st)
    cnt = en - st + 1
    gid = np.repeat(np.arange(len(st)), cnt)
    im = np.flatnonzero(p == gmax[gid]); im = im[np.r_[True, np.diff(gid[im]) != 0]]
    iv = np.flatnonzero(p == gmin[gid]); iv = iv[np.r_[True, np.diff(gid[iv]) != 0]]
    idx = np.unique(np.concatenate([st, en, im, iv]))
    t, p = t[idx], p[idx]
    keep = np.r_[True, p[1:] != p[:-1]]
    return t[keep], p[keep]


def _bars_path(k1, t0, t1):
    a = np.searchsorted(k1.open_time.values, (t0 // 60_000) * 60_000); b = np.searchsorted(k1.open_time.values, t1)
    rows = k1.iloc[a:b][['open_time', 'o', 'h', 'l', 'c']].values.astype(float)
    tt, pp = [], []
    for ot, o, h, l, c in rows:
        seq = ((0, o), (20e3, l), (40e3, h), (59e3, c)) if c >= o else ((0, o), (20e3, h), (40e3, l), (59e3, c))
        for dt, px in seq:
            if ot + dt >= t0:
                tt.append(int(ot + dt)); pp.append(px)
    return np.array(tt, dtype=np.int64), np.array(pp, dtype=float)


def build_path(pair, o_ms, horizon_ms=HORIZON_MS):
    """(t, p, src) from the fill to fill+horizon: real ticks where the day archive exists, 1m o→l/h→c elsewhere."""
    t1 = o_ms + horizon_ms
    T, P, n_bar = [], [], 0
    d0 = (o_ms // DAY) * DAY
    for dms in range(d0, t1, DAY):
        lo, hi = max(o_ms, dms), min(t1, dms + DAY)
        if lo >= hi:
            continue
        d = _day(pair, dms)
        if d is not None:
            a, b = np.searchsorted(d[0], lo), np.searchsorted(d[0], hi)
            T.append(d[0][a:b]); P.append(d[1][a:b])
        else:
            k1 = _k1(pair)
            if k1 is None:
                continue
            tt, pp = _bars_path(k1, lo, hi); T.append(tt); P.append(pp); n_bar += len(tt)
    if not T:
        return None
    t = np.concatenate(T); p = np.concatenate(P)
    if not len(t):
        return None
    t, p = _compress(t, p)
    src = 'tick' if n_bar == 0 else ('1m' if n_bar >= len(t) * 0.9 else 'mixed')
    return t, p, src


def scan_states(pair, o_ms, t, p):
    """signal_active (EMA5>EMA8 ∧ price>EMA20) and EMA13-exit (price<EMA13 ∧ EMA5<EMA8 — strict) cached at each scan; the
    forming 5m bar's close = last price at the scan. Returns per-tick arrays (sig, e13 level, e5<e8)."""
    k5 = _k5(pair)
    n = len(t)
    if k5 is None:
        return np.zeros(n, bool), np.full(n, np.nan), np.zeros(n, bool)
    scans = o_ms - 6000 + np.arange(0, int((t[-1] - o_ms) / (SCAN_S * 1000)) + 2) * SCAN_S * 1000
    scans = scans.astype(np.int64)
    # price at scan: last tick ≤ scan (first scan: entry scan, use the first tick)
    j = np.searchsorted(t, scans, side='right') - 1
    px = p[np.clip(j, 0, n - 1)]
    # last CLOSED bar at scan: open + 5m ≤ scan
    kb = np.searchsorted(k5['ot'], scans - 300_000, side='right') - 1
    ok = kb >= 0
    kb = np.clip(kb, 0, len(k5['ot']) - 1)
    em = {}
    for m in (5, 8, 13, 20):
        a = 2.0 / (m + 1)
        em[m] = a * px + (1 - a) * k5[f'e{m}'][kb]
    sig = ok & (em[5] > em[8]) & (px > em[20])
    flip = ok & (em[5] < em[8])
    e13 = np.where(ok, em[13], np.nan)
    # map scans → ticks (state valid from the scan time on)
    si = np.searchsorted(scans, t, side='right') - 1
    si = np.clip(si, 0, len(scans) - 1)
    return sig[si], e13[si], flip[si]


# ───────────────────────────── the exit evaluator ─────────────────────────────
def _first(mask, start=0):
    if start >= len(mask):
        return None
    k = np.flatnonzero(mask[start:])
    return int(k[0] + start) if len(k) else None


def evaluate(P, fx, pr):
    """P = path dict (t, p, net, sig, e13, flip), fx = fill dict (E, atr, rsi_entry, o_ms), pr = params → (idx, reason, pct)."""
    t, net = P['t'], P['net']
    n = len(t)
    pk = np.maximum.accumulate(np.maximum(net, 0.0))
    pk_prev = np.r_[0.0, pk[:-1]]
    rel = t - fx['o_ms']
    atr = fx['atr'] if (fx['atr'] and fx['atr'] > 0) else None
    # ① stop line
    if pr['sl_fixed'] is not None:
        S = np.full(n, float(pr['sl_fixed']))
    else:
        S = np.where(P['sig'], pr['sl_sig'], pr['sl_base']).astype(float)
        if atr is not None and pr['sl_atr'] > 0:
            S = np.minimum(S, -atr * pr['sl_atr'])
        if pr['sl_cap'] < 0:
            S = np.maximum(S, pr['sl_cap'])
        if pr['sl_quiet'] is not None and atr is not None and atr < pr['sl_quiet'][0]:   # Aug-20→Sep-14 quiet-pair −2.0
            S = np.minimum(S, pr['sl_quiet'][1])
    if pr['be_arm'] is not None:
        S = np.where(pk >= pr['be_arm'], np.maximum(S, pr['be_lock']), S)
    stop_hit = net <= S + 0.01
    # ③ ladder
    if pr['ladder']:
        fl = np.full(n, -np.inf)
        for trg, off in pr['ladder']:
            fl = np.where(pk_prev >= trg, np.maximum(fl, trg - off), fl)
        lad_hit = net <= fl
    else:
        lad_hit = np.zeros(n, bool)
    # ④ runner
    if pr['run_g_fixed'] is not None:
        g = np.full(n, float(pr['run_g_fixed']))
    elif atr is not None:
        g = np.full(n, atr * pr['run_g_atr'])
    else:
        g = np.full(n, np.inf)
    if pr['run_g_frac'] is not None:
        g = np.minimum(g, pr['run_g_frac'] * pk)
    if pr.get('run_g_frac_only') is not None:          # GF family: giveback = f × peak (replaces the ATR giveback)
        g = pr['run_g_frac_only'] * pk
    rfl = pk - g
    if pr['run_lock'] is not None:
        rfl = np.maximum(rfl, pr['run_lock'])
    arm_lvl = pr['run_arm']
    if pr.get('run_arm_atr') is not None and atr is not None:   # AA family: arm = m × entry ATR %
        arm_lvl = pr['run_arm_atr'] * atr
    armed = pk >= arm_lvl - 0.005
    run_hit = armed & (net <= rfl)
    tp_hit = (net >= pr['tp']) if pr['tp'] is not None else np.zeros(n, bool)
    e13_hit = (P['p'] < P['e13']) & P['flip'] if pr['ema13'] else np.zeros(n, bool)
    # ⑤ clocks
    tmax = min(pr['noexp'], pr['maxhold']) * 60_000
    clk = rel >= tmax
    doa_hit = np.zeros(n, bool)
    if pr['doa'] is not None:
        dmin, dx = pr['doa']
        j = _first(rel >= dmin * 60_000)
        if j is not None and pk[j] < dx:
            doa_hit[j] = True
    order = (('HARD_TP_LADDER', lad_hit), ('FIXED_TP', tp_hit), ('EMA13_CROSS', e13_hit), ('STOP', stop_hit),
             ('RUNNER_TRAIL', run_hit), ('DOA_EXIT', doa_hit), ('NO_EXPANSION', clk))

    def first_exit(start, rh_avail, prefix):
        best, why = None, None
        for name, m in order:
            k = _first(m, start)
            if k is not None and (best is None or k < best):
                best, why = k, name
        if best is None:
            return None
        if why == 'STOP':
            why_full = 'STOP_LOSS_WIDE' if P['sig'][best] and pr['sl_fixed'] is None else 'STOP_LOSS'
            if pr['be_arm'] is not None and pk[best] >= pr['be_arm'] and S[best] >= pr['be_lock'] - 1e-9:
                why_full = 'BE_LOCK'
            if rh_avail and pr['rh'] and why_full.startswith('STOP_LOSS'):
                held = _rh(best, S[best])
                if held is not None:
                    return held
            return best, prefix + why_full
        return best, prefix + why

    def _rh(i0, stop_level):
        hard = round(stop_level - 0.5, 4)
        if net[i0] <= hard + 0.01 or hard <= -2.2 or pk[i0] >= 0.40 or fx['rsi_entry'] is None:
            return None
        r0 = btc_rsi_at(int(t[i0]))
        if r0 is None or not (r0 >= fx['rsi_entry'] and 60.0 <= r0 <= 66.0):
            return None
        # hold: hard stop / premise (checked from each 5m close on) / time / release
        t0 = t[i0]
        hs = _first(net <= hard + 0.01, i0 + 1)
        rel_ = _first(pk >= 0.40, i0 + 1)
        tm = _first(((t - t0) >= 30 * 60_000) & (net < 0), i0 + 1)
        tm2 = _first((t - t0) >= 240 * 60_000, i0 + 1)
        cand = [(hs, 'RH_HARD_STOP'), (tm, 'RH_TIME_EXIT'), (tm2, 'RH_TIME_EXIT'), (rel_, 'RELEASE')]
        # premise: every 5m close after the trigger up to the earliest other event
        lim = min([k for k, _ in cand if k is not None] + [n - 1])
        bc = (int(t0 // 300_000) + 1) * 300_000
        while bc <= t[lim]:
            r = btc_rsi_at(bc + 2000)
            if r is None or r < 60.0 or r < fx['rsi_entry']:
                k = _first(t >= bc + 2000, i0 + 1)
                if k is not None:
                    cand.append((k, 'RH_PREMISE_EXIT'))
                break
            bc += 300_000
        cand = [(k, w) for k, w in cand if k is not None]
        if not cand:
            return None
        k, w = min(cand, key=lambda x: x[0])
        if w == 'RELEASE':
            r = first_exit(k, False, 'RH_')
            return r
        return k, w

    r = first_exit(0, True, '')
    if r is None:
        return n - 1, 'END_OF_DATA', float(net[-1])
    return r[0], r[1], float(net[r[0]])


def prep_path(fx, horizon_ms=HORIZON_MS):
    bp = build_path(fx['pair'], fx['o_ms'], horizon_ms)
    if bp is None:
        return None
    t, p, src = bp
    E = fx['E']
    net = (p / E - 1.0) * 100.0 - fx['fee_in'] - TAKER * p / E
    sig, e13, flip = scan_states(fx['pair'], fx['o_ms'], t, p)
    return dict(t=t, p=p, net=net, sig=sig, e13=e13, flip=flip, src=src)


def run_fill(fx, variants):
    """evaluate on a short window first (exits are causal) and extend only when a variant has not exited."""
    out = {}
    pending = dict(variants)
    for hz in (2 * 3600_000, 6 * 3600_000, HORIZON_MS):
        P = prep_path(fx, hz)
        if P is None:
            break
        for name, pr in list(pending.items()):
            k, why, pct = evaluate(P, fx, pr)
            if why != 'END_OF_DATA' or hz == HORIZON_MS:
                out[name] = (k, why, pct, int(P['t'][k]), P['src'], float(np.max(np.maximum(P['net'][:k + 1], 0))))
                pending.pop(name)
        if not pending:
            break
    for name in pending:
        out[name] = (None, 'NO_DATA', np.nan, None, 'none', np.nan)
    return out


def _work(args):
    fxs, variants = args
    res = []
    for fx in fxs:
        vv = dict(variants)
        if fx.get('aswas') is not None:
            vv['ASWAS'] = fx['aswas']
        try:
            r = run_fill(fx, vv)
        except Exception as ex:   # never silently: record the error
            r = {v: (None, f'ERR:{ex}', np.nan, None, 'err', np.nan) for v in vv}
        for v, (k, why, pct, tex, src, pk) in r.items():
            res.append(dict(key=fx['key'], variant=v, reason=why, pct=pct, exit_ms=tex, src=src, peak=pk))
    return res


def run_many(fills, variants, jobs):
    fills = sorted(fills, key=lambda f: (f['pair'], f['o_ms']))
    n = max(1, len(fills) // (jobs * 6))
    batches = [(fills[i:i + n], variants) for i in range(0, len(fills), n)]
    out = []
    with ProcessPoolExecutor(jobs) as ex:
        for r in ex.map(_work, batches):
            out.extend(r)
    return pd.DataFrame(out)


# ───────────── AS-WAS live exit params (config_history_v2 timeline of every exit key; validation of the replica on master) ─────────────
ASWAS = [  # (from UTC, overrides) — cumulative; before 06-24 20:20 the tight-trailing stack ran (not replicated → 'OLD_STACK')
    ('2026-06-24 20:20', dict(run_arm=0.45, run_g_atr=0.5, run_lock=0.10, ladder=[], rh=False)),
    ('2026-06-25 19:14', dict(run_lock=None)),
    ('2026-06-29 23:13', dict(run_g_atr=1.0)),
    ('2026-07-13 14:57', dict(run_lock=0.10)),
    ('2026-07-20 20:07', dict(tp=1.0)),
    ('2026-07-22 22:33', dict(tp=None, ladder=LADDER_LIVE)),
    ('2026-08-05 12:36', dict(run_arm=0.40)),
    ('2026-08-20 13:06', dict(sl_quiet=(0.45, -2.0))),
    ('2026-09-14 22:16', dict(sl_quiet=None)),
    ('2026-10-02 13:17', dict(rh=True)),
]


def aswas_params(o_ms):
    p = dict(BASE)
    if o_ms < pd.Timestamp(ASWAS[0][0]).value // 10**6:
        return None
    for ts, ov in ASWAS:
        if o_ms >= pd.Timestamp(ts).value // 10**6:
            p.update(ov)
    return p


# ───────────────────────────── cohorts ─────────────────────────────
def _ms(s):
    return (pd.to_datetime(pd.Series(s).astype(str).str[:23].str.replace('T', ' '), format='mixed') - pd.Timestamp(0)) \
        .dt.total_seconds().mul(1000).round().astype('int64').values


def load_yr():
    d = pd.read_csv(os.path.join(REP, f'ENGINE_REPLAY_{PREFIX.upper()}_ML_fills.csv'), low_memory=False)
    d['o_ms'] = _ms(d.opened_at); d['c_ms'] = _ms(d.closed_at)
    d['key'] = 'y' + d.seed.astype(str) + '|' + d.opened_at.astype(str) + '|' + d.pair
    # BURST: opened ≤ 120 s from ANOTHER bot fill (any sleeve) of the same seed — all-sleeve per-run orders
    bur = {}
    for s in sorted(d.seed.unique()):
        fr = []
        for f in glob.glob(os.path.join(CACHE, 'replay', f'{PREFIX}_*_s{s}_orders.csv')):
            o = pd.read_csv(f, usecols=['pair', 'direction', 'opened_at', 'entry_strategy', 'status'], low_memory=False)
            fr.append(o)
        A = pd.concat(fr).dropna(subset=['opened_at']).drop_duplicates(['opened_at', 'pair', 'direction'])
        ta = np.sort(_ms(A.opened_at))
        sub = d[d.seed == s]
        for k, tm in zip(sub.key, sub.o_ms):
            a, b = np.searchsorted(ta, tm - 120_000), np.searchsorted(ta, tm + 120_000, side='right')
            bur[k] = (b - a) >= 2          # itself + at least one other
    d['burst'] = d.key.map(bur).astype(bool)
    d['chop'] = pd.to_numeric(d.entry_btc_eff72, errors='coerce') <= 0.007
    d['cb'] = d.chop & d.burst
    return d


def load_master():
    m = pd.read_csv(os.path.join(REP, 'ML_WATCHLIST_master_features.csv'), low_memory=False)
    m['o_ms'] = _ms(m.opened_at); m['c_ms'] = _ms(m.closed_at)
    m['key'] = 'm|' + m.opened_at.astype(str) + '|' + m.pair
    m['chop'] = m.chop.astype(str).eq('True'); m['burst'] = m.burst.astype(str).eq('True')
    m['cb'] = m.chop & m.burst
    m['pct_live'] = pd.to_numeric(m.pnl_percentage, errors='coerce')
    return m


def fills_of(df):
    out = []
    for r in df.itertuples():
        fee_in = MAKER if str(getattr(r, 'entry_order_type', '')).upper().startswith('MAKER') else TAKER
        rsi = getattr(r, 'entry_btc_rsi_closed', np.nan)
        out.append(dict(key=r.key, pair=r.pair, o_ms=int(r.o_ms), E=float(r.entry_price), fee_in=fee_in,
                        atr=float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) else None,
                        rsi_entry=float(rsi) if pd.notna(rsi) else None))
    return out



# ───────────────────────────── analysis / report ─────────────────────────────
STOP_LIKE = ('STOP_LOSS', 'RH_HARD_STOP', 'RH_PREMISE_EXIT', 'RH_TIME_EXIT', 'BE_LOCK')
RNG_SEED = 20261005


def _base_reason(r):
    r = str(r)
    for a, b in (('STOP_LOSS_WIDE', 'STOP'), ('STOP_LOSS', 'STOP'), ('HARD_TP_LADDER', 'LADDER'), ('HARD_TP', 'FIXED_TP'),
                 ('RUNNER_TRAIL', 'RUNNER'), ('TRAILING_STOP', 'TRAIL_OLD')):
        r = r.replace(a, b)
    return r.split(' ')[0]


def day_boot(vals, days, reps=2000, seed=RNG_SEED):
    """day-block bootstrap of the mean of vals (resample calendar days with replacement) → (lo, hi) 95 %."""
    vals = np.asarray(vals, float); days = np.asarray(days)
    ud, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=vals, minlength=len(ud)); c = np.bincount(inv, minlength=len(ud))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ud), (reps, len(ud)))
    m = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def nwr(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return '0'
    return f"{len(x)}·{(x > 0).mean() * 100:.0f}%·{x.mean():+.3f}"


def payoff(x):
    x = pd.Series(x).dropna(); w = x[x > 0].mean(); l = x[x <= 0].mean()
    if not (w > 0) or not (l < 0):
        return np.nan, np.nan
    return w / abs(l), abs(l) / (w + abs(l)) * 100


def live_windows():
    """live hours = MASTER_POOL_stacked era spans [first open, last open + 1 min] (as scripts/yr5_ml_unmatched_why.py)."""
    W_ = pd.read_csv(os.path.join(REP, 'MASTER_POOL_stacked.csv'), low_memory=False, usecols=['era', 'opened_at'])
    W_['t'] = pd.to_datetime(W_.opened_at.astype(str).str[:19])
    return W_.groupby('era').t.agg(['min', 'max']).values


def report(jobs_note=''):
    Y = load_yr().set_index('key'); M = load_master().set_index('key')
    RY = pd.read_csv(os.path.join(REP, f'ML_EXIT_OPT_variants_{PREFIX}.csv'))
    RM = pd.read_csv(os.path.join(REP, f'ML_EXIT_OPT_variants_master{OUT_SFX}.csv'))
    PY = RY.pivot(index='key', columns='variant', values='pct'); RYr = RY.pivot(index='key', columns='variant', values='reason')
    PM = RM.pivot(index='key', columns='variant', values='pct'); RMr = RM.pivot(index='key', columns='variant', values='reason')
    EXY = RY.pivot(index='key', columns='variant', values='exit_ms')
    Y['day'] = Y.opened_at.astype(str).str[:10]; Y['mon'] = Y.opened_at.astype(str).str[:7]
    Y['H'] = np.where(Y.mon <= '2026-04', 'H1', 'H2')
    Y['tt'] = pd.to_datetime(Y.opened_at.astype(str).str[:19])
    WIN = live_windows(); first = pd.Timestamp(WIN[:, 0].min())
    Y['zone'] = 'pre-live'
    Y.loc[Y.tt >= first, 'zone'] = 'off-hours'
    for a, z in WIN:
        Y.loc[(Y.tt >= a) & (Y.tt <= z + pd.Timedelta(minutes=1)), 'zone'] = 'live-hours'
    L = []; W = L.append
    U = (PREFIX.upper())

    # ── §1 fidelity
    W(f'## 1. Replica fidelity on {U} (must match before any counterfactual is trusted)\n')
    rows = []
    X = Y.join(PY['BASE'].rename('rep')).join(RYr['BASE'].rename('rr')).join(EXY['BASE'].rename('rex'))
    X['lb'] = X.close_reason.map(_base_reason); X['rb'] = X.rr.map(_base_reason); X['d'] = X.rep - X.pct
    X['dt'] = (X.rex - X.c_ms) / 1000

    def fid(lab, x, live):
        rows.append(f"| {lab} | {len(x)} | {(x.rb == x.lb).mean() * 100:.1f} % | {np.corrcoef(x[live], x.rep)[0, 1]:.3f} | "
                    f"{(x.rep - x[live]).mean():+.4f} | {((x.rep - x[live]).abs() <= 0.05).mean() * 100:.1f} % | "
                    f"{(x.dt.abs() <= 5).mean() * 100:.1f} % | {x[live].mean():+.4f} | {x.rep.mean():+.4f} |")
    fid(f'{U} engine (today\'s config, 3 seeds) — all', X, 'pct')
    fid(f'{U} engine — ex chop∧burst', X[~X.cb], 'pct')
    for s in sorted(Y.seed.unique()):
        fid(f'{U} engine — ex chop∧burst, seed {s}', X[(~X.cb) & (X.seed == s)], 'pct')
    MX = M.join(PM['ASWAS'].rename('rep')).join(RMr['ASWAS'].rename('rr')).join(
        RM.pivot(index='key', columns='variant', values='exit_ms')['ASWAS'].rename('rex'))
    MX['lb'] = MX.close_reason.map(_base_reason); MX['rb'] = MX.rr.map(_base_reason); MX['dt'] = (MX.rex - MX.c_ms) / 1000
    mx = MX[MX.rep.notna()]
    fid('master live, AS-WAS exit params (fills ≥ Jun-24 20:20)', mx, 'pct_live')
    fid('master live, AS-WAS — bot exits only (ex MANUAL / tight-trail)', mx[~mx.lb.isin(['MANUAL', 'TRAIL_OLD'])], 'pct_live')
    W('| Cohort | N | same exit reason | corr(pct) | mean Δ (rep − real) | \\|Δ\\| ≤ 0.05 | exit time ±5 s | real avg % | replica avg % |')
    W('|---|---|---|---|---|---|---|---|---|')
    L.extend(rows)
    ct = pd.crosstab(X.lb, X.rb)
    W(f'\n{U} exit-reason confusion (rows = engine, cols = replica):\n')
    W('| engine \\ replica | ' + ' | '.join(ct.columns) + ' |'); W('|---' * (len(ct.columns) + 1) + '|')
    for i, r in ct.iterrows():
        W(f'| {i} | ' + ' | '.join(str(v) for v in r.values) + ' |')
    st = X[X.close_reason.astype(str).str.startswith('STOP')]
    lab_ok = ((st.close_reason.str.startswith('STOP_LOSS_WIDE')) == (st.rr.astype(str) == 'STOP_LOSS_WIDE')).mean()
    mis = X[X.rb != X.lb]
    W(f"\nStop label (−0.70 vs −1.00 WIDE = scan signal-active state) agrees on {lab_ok * 100:.0f} % of engine stops. "
      f"{len(mis)} reason mismatches, Σ Δ {mis.d.sum():+.2f} % (mean over all fills {X.d.mean():+.4f}): "
      + ', '.join(f"{a}→{b} {n}" for (a, b), n in mis.groupby(['lb', 'rb']).size().items()) + '.\n')

    # ── §2 cohort
    k = Y.index[~Y.cb]
    W('## 2. Cohort\n')
    W(f"{U} (frozen config 181131e, seeds 1–3, Jan-04 → Oct-03): {len(Y)} ML fills; CHOP∧BURST (entry_btc_eff72 ≤ 0.007 ∧ opened ≤ 120 s "
      f"from another bot fill of the SAME seed, any sleeve incl. FRENZY — all-sleeve per-run orders replay/{PREFIX}_*_s<seed>_orders.csv) = "
      f"{int(Y.cb.sum())} fills ({nwr(Y.pct[Y.cb])} engine) → removed. Working cohort {len(k)} fills ({nwr(Y.pct[k])} engine, "
      f"{nwr(PY.loc[k, 'BASE'])} replica). Per seed (engine / replica): "
      + ' · '.join(f"s{s} {Y.loc[k][Y.loc[k].seed == s].pct.mean():+.3f} / {PY.loc[Y.loc[k][Y.loc[k].seed == s].index, 'BASE'].mean():+.3f}" for s in sorted(Y.seed.unique())) + '\n')

    # ── master deltas per variant
    M['notional'] = pd.to_numeric(M.notional_value, errors='coerce')
    mk = M.index[~M.cb]
    recent = M.index[M.era.isin(['B12', 'B13', 'B14', 'B15', 'B16']) & ~M.cb]

    def mdelta(v, keys):
        d = PM.loc[keys, v] - PM.loc[keys, 'BASE']
        return d.mean(), float((d / 100 * M.loc[keys, 'notional']).sum())

    # ── §3 single-variant table
    W(f'## 3. Every variant vs the replica baseline ({U} ex chop∧burst)\n')
    nv = len(PY.columns) - 1
    W(f"Pre-declared grid: {N_PREDECLARED} variants (the 42 of the yr4 study + its 7 post-hoc stop/combination variants, now "
      f"pre-declared + 13 NEW in families NL/AA/GF), all written into scripts/ml_exit_optimize_yr5.py VARIANTS before the yr5 run. "
      f"Post-hoc: {len(POSTHOC)}. Total tried {nv}. Δ = paired per fill vs BASE; CI = day-block bootstrap 95 % (2,000 resamples of "
      f"calendar days, seeds pooled). live/off = live-hours (master-pool batch windows) / off-hours since {first:%b-%d}. "
      f"Master = 109 current-stack live fills ex chop∧burst re-priced (Δ %/trade, Δ $ live-sized); recent = B12–B16.\n")
    W('| Variant | fam | N·WR·avg % | Δ avg % [95 % CI] | Δ H1 | Δ H2 | s1 / s2 / s3 Δ | months better | Δ live-h / off-h | payoff | BE WR | master Δ % · Δ $ | recent Δ $ |')
    W('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    b = PY.loc[k, 'BASE']; days = Y.loc[k, 'day'].values
    res = []
    for v in PY.columns:
        x = PY.loc[k, v]; d = x - b
        lo, hi = day_boot(d.values, days)
        dh = {h: d[(Y.loc[k, 'H'] == h).values].mean() for h in ('H1', 'H2')}
        dz = {z: d[(Y.loc[k, 'zone'] == z).values].mean() for z in ('live-hours', 'off-hours', 'pre-live')}
        ds = [d[(Y.loc[k, 'seed'] == s).values].mean() for s in sorted(Y.seed.unique())]
        mb = d.groupby(Y.loc[k, 'mon'].values).mean()
        po, be = payoff(x)
        mdp, mdu = mdelta(v, mk); _, rdu = mdelta(v, recent)
        res.append(dict(variant=v, fam=FAMILY.get(v, 'POST'), N=len(x), WR=(x > 0).mean() * 100, avg=x.mean(), d=d.mean(), lo=lo, hi=hi,
                        dH1=dh['H1'], dH2=dh['H2'], d_live=dz['live-hours'], d_off=dz['off-hours'], d_pre=dz['pre-live'],
                        s1=ds[0], s2=ds[1], s3=ds[2], mb=int((mb > 0).sum()), nm=len(mb), payoff=po, be_wr=be,
                        master_d_pct=mdp, master_d_usd=mdu, recent_d_usd=rdu, x=x))
    res.sort(key=lambda r: -r['d'])
    for r in res:
        W(f"| {r['variant']}{' (post-hoc)' if r['variant'] in POSTHOC else ''} | {r['fam']} | {nwr(r['x'])} | {r['d']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
          f"{r['dH1']:+.3f} | {r['dH2']:+.3f} | {r['s1']:+.3f} / {r['s2']:+.3f} / {r['s3']:+.3f} | {r['mb']}/{r['nm']} | "
          f"{r['d_live']:+.3f} / {r['d_off']:+.3f} | {r['payoff']:.2f} | {r['be_wr']:.1f} % | {r['master_d_pct']:+.3f} · ${r['master_d_usd']:+,.0f} | "
          f"${r['recent_d_usd']:+,.0f} |")
    pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != 'x'} for r in res]).to_csv(
        os.path.join(REP, f'ML_EXIT_OPT_summary{OUT_SFX}.csv'), index=False)
    RES = {r['variant']: r for r in res}

    # ── §4 stop-change two-sided split
    W(f'\n## 4. Stop-width changes — two-sided split on the ENGINE-STOPPED cohort (CLAUDE.md EXIT/STOP rule)\n')
    W('Engine-stopped = fills whose ENGINE close was STOP_LOSS* or RH_HARD/PREMISE/TIME. SAVED = engine-stopped, variant exits otherwise · '
      'DEEPER = engine-stopped, variant stops too · NOT-STOPPED = engine did not stop (must be Δ ≈ 0 for a WIDER stop).\n')
    W('| Variant | stopped N | SAVED N · Δ Σ % | DEEPER N · Δ Σ % | stopped Δ avg [95 % CI] | H1 / H2 stopped Δ | NOT-stopped N changed · Δ Σ % |')
    W('|---|---|---|---|---|---|---|')
    eng_stop = Y.loc[k, 'close_reason'].astype(str).str.startswith(('STOP_LOSS', 'RH_HARD', 'RH_PREMISE', 'RH_TIME'))
    ks, kn = k[eng_stop.values], k[~eng_stop.values]
    for v in [c for c in PY.columns if c.startswith('SL_') or c in ('RH_off', 'C_SL12+TP10', 'C_SL12+TP10+TRf05')]:
        d = PY.loc[ks, v] - PY.loc[ks, 'BASE']
        vst = RYr.loc[ks, v].astype(str).str.startswith(STOP_LIKE)
        lo, hi = day_boot(d.values, Y.loc[ks, 'day'].values)
        dn = PY.loc[kn, v] - PY.loc[kn, 'BASE']
        hs = [d[(Y.loc[ks, 'H'] == h).values].mean() for h in ('H1', 'H2')]
        W(f"| {v} | {len(ks)} | {int((~vst).sum())} · {d[~vst].sum():+.1f} | {int(vst.sum())} · {d[vst].sum():+.1f} | "
          f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {hs[0]:+.3f} / {hs[1]:+.3f} | {int((dn.abs() > 1e-9).sum())} · {dn.sum():+.1f} |")

    # ── §5 the new families: anatomy (which fills change; runners vs chopped)
    W('\n## 5. New families — anatomy (fills changed, better / worse, on the armed cohort)\n')
    W('| Variant | fills changed | better N · Δ Σ % | worse N · Δ Σ % | Δ on peak ≥ +1.0 fills (N) | Δ on high-ATR (≥ median) / low-ATR |')
    W('|---|---|---|---|---|---|')
    pkb = RY[RY.variant == 'BASE'].set_index('key').peak.reindex(k)
    atr = pd.to_numeric(Y.loc[k, 'entry_atr_pct'], errors='coerce'); hi_atr = atr >= atr.median()
    for v in [c for c in PY.columns if c.split('_')[0] in ('NL', 'AA', 'GF', 'BE', 'TP')] :
        d = PY.loc[k, v] - b; ch = d.abs() > 1e-9
        big = (pkb >= 1.0).values
        W(f"| {v} | {int(ch.sum())} | {int((d > 1e-9).sum())} · {d[d > 1e-9].sum():+.1f} | {int((d < -1e-9).sum())} · {d[d < -1e-9].sum():+.1f} | "
          f"{d[big].mean():+.3f} ({int(big.sum())}) | {d[hi_atr.values].mean():+.3f} / {d[~hi_atr.values].mean():+.3f} |")

    # ── §6 monthly for the top candidates
    top = [r['variant'] for r in res if r['variant'] != 'BASE'][:6]
    cands = list(dict.fromkeys(top + ['TP_1.0', 'NL_k1.0', 'NL_k1.0_noLAD']))
    W(f'\n## 6. Month by month ({U} ex chop∧burst; N·WR·avg %, Δ vs BASE)\n')
    W('| Month | BASE | ' + ' | '.join(cands) + ' |'); W('|---' * (len(cands) + 2) + '|')
    for mo, g in Y.loc[k].groupby('mon'):
        cells = [nwr(PY.loc[g.index, 'BASE'])] + [f"{PY.loc[g.index, c].mean():+.3f} ({(PY.loc[g.index, c] - PY.loc[g.index, 'BASE']).mean():+.3f})" for c in cands]
        W(f"| {mo} | " + ' | '.join(cells) + ' |')

    # ── §7 sleeve level with CIs, halves, live/off hours
    W('\n## 7. Sleeve avg %/trade with day-block 95 % CI — year, halves, live-hours vs off-hours\n')
    W('| Stack | YEAR | H1 Jan–Apr | H2 May–Oct | live-hours | off-hours (since ' + f'{first:%b-%d})' + ' | pre-live |')
    W('|---|---|---|---|---|---|---|')

    def ci_cell(x, dd):
        if not len(x):
            return '—'
        lo, hi = day_boot(x.values, dd); return f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (N {len(x)})"
    for lab, keys, col in [('engine, all fills', Y.index, None), ('replica BASE, all fills', Y.index, 'BASE'),
                           ('replica BASE, chop∧burst blocked', k, 'BASE')] + [(f'{c}, chop∧burst blocked', k, c) for c in cands]:
        xs = Y.loc[keys, 'pct'] if col is None else PY.loc[keys, col]
        cells = [ci_cell(xs, Y.loc[keys, 'day'].values)]
        for colname, val in (('H', 'H1'), ('H', 'H2'), ('zone', 'live-hours'), ('zone', 'off-hours'), ('zone', 'pre-live')):
            kk = keys[(Y.loc[keys, colname] == val).values]
            cells.append(ci_cell(xs.loc[kk], Y.loc[kk, 'day'].values))
        W(f"| {lab} | " + ' | '.join(cells) + ' |')
    # Δ CI by zone for the candidates
    W('\nΔ vs BASE by zone [day-block 95 % CI]:\n')
    W('| Variant | live-hours Δ | off-hours Δ | pre-live Δ |'); W('|---|---|---|---|')
    for c in cands:
        cells = []
        for z in ('live-hours', 'off-hours', 'pre-live'):
            kk = k[(Y.loc[k, 'zone'] == z).values]; d = PY.loc[kk, c] - PY.loc[kk, 'BASE']
            lo, hi = day_boot(d.values, Y.loc[kk, 'day'].values); cells.append(f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        W(f"| {c} | " + ' | '.join(cells) + ' |')

    # ── §8 master
    W('\n## 8. MASTER re-pricing (113 current-stack live ML fills; $ = pct × notional as traded) — refute-only\n')
    cols = [('as traded', None, False), ('replica BASE', 'BASE', False)] + [(c, c, False) for c in cands] + \
           [('cb-blocked BASE', 'BASE', True)] + [(f'cb-blocked {c}', c, True) for c in cands[:3]]

    def mcell(keys, col):
        if not len(keys):
            return '—'
        x = M.loc[keys, 'pct_live'] if col is None else PM.loc[keys, col]
        usd = M.loc[keys, 'pnl'].sum() if col is None else (x / 100 * M.loc[keys, 'notional']).sum()
        return f"{len(x)}·{(x > 0).mean() * 100:.0f}%·${usd:+,.0f}·{x.mean():+.3f}"
    W('| Batch | ' + ' | '.join(c[0] for c in cols) + ' |'); W('|---' * (len(cols) + 1) + '|')
    eras = [e for e in ['BASE'] + [f'B{i}' for i in range(1, 17)] if e in set(M.era)]
    for e in eras + ['B12–B16', 'TOTAL']:
        g = M if e == 'TOTAL' else (M[M.era.isin(['B12', 'B13', 'B14', 'B15', 'B16'])] if e == 'B12–B16' else M[M.era == e])
        cells = [mcell(g.index[~g.cb] if blk else g.index, col) for _, col, blk in cols]
        W(f"| {'**' + e + '**' if e in ('TOTAL', 'B12–B16') else e} | " + ' | '.join(cells) + ' |')
    return '\n'.join(L), RES, cands



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--only', default='all')          # replica | all
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--report', action='store_true')
    A = ap.parse_args()
    if A.report:
        txt, _, _ = report()
        open(os.path.join(REP, f'ML_EXIT_OPT_tables{OUT_SFX}.md'), 'w').write(txt)
        print(txt)
        sys.exit(0)
    t0 = time.time()
    Y = load_yr(); M = load_master()
    Y[['key', 'seed', 'pair', 'opened_at', 'chop', 'burst', 'cb']].to_csv(os.path.join(REP, f'ML_EXIT_OPT_{PREFIX}_cohort.csv'), index=False)
    fy, fm = fills_of(Y), fills_of(M)
    for f in fm:
        f['aswas'] = aswas_params(f['o_ms'])
    if A.limit:
        fy = fy[:A.limit]
    vs = {'BASE': VARIANTS['BASE']} if A.only == 'replica' else VARIANTS
    print(f'[ML_EXIT_OPT] {PREFIX} {len(fy)} fills · master {len(fm)} · variants {len(vs)} · chop∧burst {PREFIX} {int(Y.cb.sum())} master {int(M.cb.sum())}', flush=True)
    rm = run_many(fm, vs, A.jobs)
    rm.to_csv(os.path.join(REP, f'ML_EXIT_OPT_variants_master{OUT_SFX}.csv'), index=False)
    print(f'[ML_EXIT_OPT] master done {time.time() - t0:.0f}s', flush=True)
    ry = run_many(fy, vs, A.jobs)
    ry.to_csv(os.path.join(REP, f'ML_EXIT_OPT_variants_{PREFIX}.csv'), index=False)
    print(f'[ML_EXIT_OPT] {PREFIX} done {time.time() - t0:.0f}s', flush=True)
