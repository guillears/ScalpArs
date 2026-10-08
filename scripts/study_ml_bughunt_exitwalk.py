#!/usr/bin/env python3
"""ML BACKTEST BUG HUNT (2026-10-08) — part 1: INDEPENDENT tick re-walk of every yr5 momentum-long fill.

Written from a fresh reading of services/trading_engine.py check_realtime_stop_loss (realtime chain order: RECOVERY HOLD →
HARD_TP_LADDER (previous-tick peak) → peak update → STOP (BE off for STRONG_BUY / VERY_STRONG; signal-active −1.00 else −0.70;
ATR widen −1.5×ATR capped at −1.20; fires at pnl ≤ sl + 0.01) or RH trigger → LONG RUNNER (peak ≥ arm − 0.005; floor =
max(peak − 1.0×ATR, +0.10))) and update_open_positions (NO_EXPANSION 180 min, MAX_HOLD 1200 min) + services/recovery_hold.py
semantics, and the frozen yr5 config (reports/backtest_cache/replay/frozen_config_yr5_181131e.json). It does NOT import the
engine, the harness, services.recovery_hold or scripts/ml_exit_optimize.py: every rule, the fee model, the BTC closed-bar RSI and
the pair EMA signal are re-implemented here from raw data (aggTrade archives ticks_q/ticks, k5m_full 5m klines, btc_5m.csv).

P&L % = net / entry notional: (p/e − 1)·100 − entry fee (MAKER 0.018 / TAKER 0.045) − 0.045 · p/e  (exit taker on the current
notional) — the engine's realtime formula.

Also prices SIMPLE BASELINE exits on the same entries (net TP/SL brackets, fixed holds) for the sanity check (task item 5).

Usage: venv/bin/python scripts/study_ml_bughunt_exitwalk.py [--jobs 8] [--limit N]
Out:   reports/study_ml_bughunt_exitwalk.csv  (one row per yr5 ML fill)
"""
import os, sys, json, math, argparse, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'reports', 'backtest_cache')
REP = os.path.join(ROOT, 'reports')
DAY = 86_400_000
BAR = 300_000

CFG = json.load(open(os.path.join(CACHE, 'replay', 'frozen_config_yr5_181131e.json')))
TH = CFG['thresholds']
TAKER = float(CFG['taker_fee']) * 100          # 0.045 %
MAKER = float(CFG['maker_fee']) * 100          # 0.018 %
SL_BASE = float(CFG['confidence_levels']['STRONG_BUY']['stop_loss'])            # −0.70 (VERY_STRONG identical, checked below)
SL_SIG = float(CFG['confidence_levels']['STRONG_BUY']['signal_active_sl'])      # −1.00
assert CFG['confidence_levels']['VERY_STRONG']['stop_loss'] == SL_BASE and CFG['confidence_levels']['VERY_STRONG']['signal_active_sl'] == SL_SIG
assert not CFG['confidence_levels']['STRONG_BUY']['be_levels_enabled'] and not CFG['confidence_levels']['VERY_STRONG']['be_levels_enabled']
SL_ATR = float(TH['sl_atr_multiplier']); SL_CAP = float(TH['sl_atr_widen_floor_pct'])
RUN_ARM = float(TH['runner_trail_arm_peak']); RUN_N = float(TH['runner_trail_atr_mult']); RUN_LOCK = float(TH['runner_trail_be_lock_pct'])
assert TH['runner_trail_enabled'] and TH['runner_trail_be_ratchet_enabled'] and float(TH['runner_trail_giveback_frac']) == 0.0
assert float(TH['runner_trail_atr_min']) == 0.0 and float(TH['momentum_long_sl_atr_threshold']) == 0.0
for k in ('fast_exit_enabled', 'fast_exit_l2_enabled', 'atr_low_fixed_tp_long_enabled', 'ema13_cross_exit_long_enabled',
          'ema_stack_cross_exit_enabled', 'rsi_handoff_active', 'rsi_momentum_exit_enabled', 'tick_momentum_exit_enabled',
          'signal_lost_exit_enabled', 'regime_change_exit_enabled', 'fl1_for_wide_sl_enabled', 'fl2_enabled'):
    assert not TH[k], k
assert float(TH['pnl_trailing_trigger']) == 0.0
LADDER = [tuple(float(x) for x in r.split(':')) for r in TH['hard_tp_ladder_long'].split(',')]   # (trigger, offset)
RH_ON = bool(TH['recovery_hold_enabled']); RH_LO, RH_HI = float(TH['recovery_hold_rsi_min']), float(TH['recovery_hold_rsi_max'])
RH_ROOM, RH_REL = float(TH['recovery_hold_room_pct']), float(TH['recovery_hold_release_pct'])
RH_TMIN, RH_TMAX = float(TH['recovery_hold_time_min']), float(TH['recovery_hold_max_min'])
NOEXP_MIN = float(CFG['investment']['no_expansion_minutes']); MAXHOLD_MIN = float(CFG['investment']['max_holding_time_minutes'])
SCAN_S = 111.6
HORIZON_MS = 8 * 3600_000          # NO_EXPANSION (180 min) + a full 240-min hold fit inside; MAX_HOLD 1200 never binds before

# ───────────── data ─────────────
_btc = None


def btc5():
    global _btc
    if _btc is None:
        b = pd.read_csv(os.path.join(CACHE, 'btc_5m.csv'), usecols=['open_time', 'c'])
        f = os.path.join(CACHE, 'k5m_full', 'BTCUSDT.csv')
        if os.path.exists(f):
            b2 = pd.read_csv(f, usecols=['open_time', 'c'])
            b = pd.concat([b, b2[~b2.open_time.isin(b.open_time)]])
        b = b.drop_duplicates('open_time').sort_values('open_time')
        _btc = (b.open_time.values.astype(np.int64), b.c.values.astype(float))
    return _btc


def wilder_rsi(closes, n=14):
    """RSI(n), Wilder smoothing seeded with the first change (ewm alpha 1/n adjust=False on gains/losses) — written here."""
    d = np.diff(closes)
    if len(d) < 1:
        return None
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    up, dn = g[0], l[0]
    a = 1.0 / n
    for i in range(1, len(d)):
        up += a * (g[i] - up); dn += a * (l[i] - dn)
    if dn <= 0:
        return 100.0 if up > 0 else None
    return 100.0 - 100.0 / (1.0 + up / dn)


def btc_closed_rsi(now_ms):
    """the reading live holds at now: a 100-bar BTC 5m fetch, forming bar dropped (≥ 56 closed bars needed), refreshed ~2 s after
    each 5m close → as of now − 2 s. Returns (rsi, last closed bar open)."""
    ot, c = btc5()
    t = now_ms - 2000
    j = np.searchsorted(ot, t - BAR, side='right')            # bars with open + 5 min ≤ t
    i0 = max(0, j - 99)
    if j - i0 < 56:
        return None, None
    return wilder_rsi(c[i0:j]), int(ot[j - 1])


_K5 = {}


def k5(pair):
    if pair in _K5:
        return _K5[pair]
    f = os.path.join(CACHE, 'k5m_full', f'{pair}.csv')
    out = None
    if os.path.exists(f):
        d = pd.read_csv(f, usecols=['open_time', 'c']).drop_duplicates('open_time').sort_values('open_time')
        c = d.c.astype(float)
        out = (d.open_time.values.astype(np.int64), c.values,
               {n: c.ewm(span=n, adjust=False).mean().values for n in (5, 8, 20)})
    _K5.clear(); _K5[pair] = out
    return out


def ema_last(x, n):
    a = 2.0 / (n + 1); e = x[0]
    for v in x[1:]:
        e = a * v + (1 - a) * e
    return e


def signal_active_at(pair, t_ms, price):
    """EMA5 > EMA8 ∧ price > EMA20 on 5m closes with the forming bar's close = the current price (scan PairData)."""
    kk = k5(pair)
    if kk is None or price is None:
        return False
    ot, c, em = kk
    j = np.searchsorted(ot, t_ms - BAR, side='right')         # closed bars
    if j < 30:
        return False
    e = {n: (2.0 / (n + 1)) * price + (1 - 2.0 / (n + 1)) * em[n][j - 1] for n in (5, 8, 20)}   # forming bar close = price
    return e[5] > e[8] and price > e[20]


_DAYS = {}


def day_ticks(pair, day_ms):
    key = (pair, day_ms)
    if key in _DAYS:
        return _DAYS[key]
    ds = time.strftime('%Y-%m-%d', time.gmtime(day_ms / 1000))
    out = None
    for sub in ('ticks_q', 'ticks'):
        fp = os.path.join(CACHE, sub, pair, f'{ds}.npz')
        if os.path.exists(fp):
            z = np.load(fp)
            out = (z['t'].astype(np.int64), z['p'].astype(np.float64))
            break
    if len(_DAYS) > 4:
        _DAYS.pop(next(iter(_DAYS)))
    _DAYS[key] = out
    return out


def ticks(pair, t0, t1):
    ts, ps, missing = [], [], []
    d = (t0 // DAY) * DAY
    while d < t1:
        z = day_ticks(pair, d)
        if z is None:
            missing.append(d)
        else:
            a = np.searchsorted(z[0], t0); b = np.searchsorted(z[0], t1)
            ts.append(z[0][a:b]); ps.append(z[1][a:b])
        d += DAY
    if not ts:
        return np.zeros(0, np.int64), np.zeros(0), missing
    return np.concatenate(ts), np.concatenate(ps), missing


# ───────────── the walk ─────────────
def first(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return (start + int(idx[0])) if len(idx) else None


def walk(r):
    pair = r['pair']; e = float(r['entry_price']); t_open = int(r['open_ms'])
    fe = MAKER if r['entry_order_type'] == 'MAKER' else TAKER
    atr = r['entry_atr_pct']; atr = float(atr) if pd.notna(atr) else None
    t, p, missing = ticks(pair, t_open, t_open + HORIZON_MS)
    out = dict(n_ticks=len(t), tick_days_missing=len(missing))
    if not len(t):
        out.update(my_reason='NO_TICKS'); return out
    pnl = (p / e - 1) * 100 - fe - TAKER * p / e
    # stop line per tick (signal_active refreshed on a 111.6 s scan grid from the open; entry follows a scan by ~6–20 s)
    sl0 = SL_BASE
    def widen(s):
        if atr is not None and atr > 0:
            s = min(s, -atr * SL_ATR)
        return max(s, SL_CAP) if SL_CAP < 0 else s
    sl_plain, sl_wide = widen(SL_BASE), widen(SL_SIG)
    grid = np.arange(t_open - 10_000, t[-1] + int(SCAN_S * 1000), int(SCAN_S * 1000))
    gi = np.clip(np.searchsorted(t, grid) - 1, 0, None)
    sig = np.array([signal_active_at(pair, int(g), float(p[i]) if t[i] <= g else e) for g, i in zip(grid, gi)])
    seg = np.searchsorted(grid, t, side='right') - 1
    sig_t = sig[np.clip(seg, 0, len(sig) - 1)]
    sl_t = np.where(sig_t, sl_wide, sl_plain)
    rsi_entry, _ = btc_closed_rsi(t_open)
    out['my_rsi_entry'] = rsi_entry

    def normal_stack(i0, peak0, held_before):
        """first exit from index i0 on (no hold trigger if held_before). Returns (reason, idx, extra)"""
        pk = np.maximum(peak0, np.maximum.accumulate(np.maximum(pnl[i0:], 0.0)))
        prev = np.r_[peak0, pk[:-1]]
        fl = np.full(len(pk), -np.inf)
        for trig, off in LADDER:
            fl = np.where(prev >= trig, trig - off, fl)
        lad = pnl[i0:] <= fl
        stp = pnl[i0:] <= sl_t[i0:] + 0.01
        if atr is not None and atr > 0:
            rfl = np.maximum(pk - RUN_N * atr, RUN_LOCK)
        else:
            rfl = np.full(len(pk), RUN_LOCK)
        run = (pk >= RUN_ARM - 0.005) & (pnl[i0:] <= rfl)
        age = (t[i0:] - t_open) / 60_000.0
        tim = age >= NOEXP_MIN
        cands = []
        for nm, m in (('HARD_TP_LADDER', lad), ('STOP', stp), ('RUNNER_TRAIL', run), ('NO_EXPANSION', tim)):
            k = first(m)
            if k is not None:
                cands.append((k, ['HARD_TP_LADDER', 'STOP', 'RUNNER_TRAIL', 'NO_EXPANSION'].index(nm), nm))
        if not cands:
            return 'OPEN_AT_END', len(pnl) - 1, pk[-1]
        k, _, nm = min(cands)
        if nm == 'STOP':
            nm = 'STOP_LOSS_WIDE' if sig_t[i0 + k] else 'STOP_LOSS'
        return nm, i0 + k, pk[k]

    reason, i, pk = normal_stack(0, 0.0, False)
    out['first_reason'] = reason
    held = False
    if reason.startswith('STOP_LOSS') and RH_ON:
        rsi_now, bt = btc_closed_rsi(int(t[i]))
        fresh = bt is not None and 0 <= t[i] - bt <= 2.5 * BAR
        hard = float(sl_t[i]) - RH_ROOM
        ok = (rsi_entry is not None and rsi_now is not None and fresh and pnl[i] > hard + 0.01 and pk < RH_REL
              and rsi_now >= rsi_entry and RH_LO <= rsi_now <= RH_HI)
        if ok:
            held = True
            t_h = int(t[i]); out['my_rh_at'] = t_h; out['my_rh_pnl'] = float(pnl[i])
            j0 = i + 1
            # premise: the first BTC closed bar (refresh ~2 s after close) whose RSI < 60 or < entry
            prem_t = None
            bc = (t_h // BAR) * BAR + BAR + 2000
            while bc <= t_h + int(RH_TMAX * 60_000):
                rr, _ = btc_closed_rsi(bc + 2000)
                if rr is None or rr < RH_LO or rr < rsi_entry:
                    prem_t = bc; break
                bc += BAR
            pk_h = np.maximum(pk, np.maximum.accumulate(np.maximum(pnl[j0:], 0.0))) if j0 < len(pnl) else np.zeros(0)
            sub = pnl[j0:]; tt = t[j0:]; m = (tt - t_h) / 60_000.0
            cands = []
            for nm, mask in (('RH_HARD_STOP', sub <= hard + 0.01),
                             ('RH_PREMISE_EXIT', (tt >= prem_t) if prem_t is not None else np.zeros(len(tt), bool)),
                             ('RH_TIME_EXIT', ((m >= RH_TMIN) & (sub < 0)) | (m >= RH_TMAX)),
                             ('RELEASE', np.maximum(pk_h, sub) >= RH_REL)):
                k = first(mask)
                if k is not None:
                    cands.append((k, nm))
            if not cands:
                reason, i = 'OPEN_AT_END', len(pnl) - 1
            else:
                # same-tick order inside the realtime path: release is tested first (rh_in_hold on max(peak, pnl)), then hard → premise → time
                k = min(c[0] for c in cands)
                names = [c[1] for c in cands if c[0] == k]
                nm = 'RELEASE' if 'RELEASE' in names else [x for x in ('RH_HARD_STOP', 'RH_PREMISE_EXIT', 'RH_TIME_EXIT') if x in names][0]
                if nm == 'RELEASE':
                    reason2, i2, _ = normal_stack(j0 + k, float(pk_h[k]) if len(pk_h) else pk, True)
                    reason, i = ('RH_' + reason2 if not reason2.startswith('OPEN') else reason2), i2
                else:
                    reason, i = nm, j0 + k
    # monitor-path timing: NO_EXPANSION fires on the 1 Hz monitor (here: the tick that crossed 180 min)
    out.update(my_reason=reason, my_exit_ms=int(t[i]), my_pct=float(pnl[i]), my_held=held,
               my_peak=float(np.max(np.maximum(pnl[:i + 1], 0))))
    # simple baselines on the same entry (net P&L, first touch, ticks)
    for tp, sl in ((0.5, -0.7), (0.4, -0.7), (1.0, -1.0), (0.3, -0.3), (0.6, -0.6)):
        a = first(pnl >= tp); b = first(pnl <= sl)
        h = first((t - t_open) >= 180 * 60_000)
        c = [x for x in (a, b, h) if x is not None]
        k = min(c) if c else len(pnl) - 1
        out[f'bl_tp{tp}_sl{-sl}'] = float(tp if (a is not None and k == a) else sl if (b is not None and k == b) else pnl[k])
    for hm in (15, 60, 180):
        k = first((t - t_open) >= hm * 60_000)
        out[f'bl_hold{hm}m'] = float(pnl[k]) if k is not None else float(pnl[-1])
    w2 = (t - t_open) <= 120 * 60_000
    out['mfe120'] = float(pnl[w2].max()); out['mae120'] = float(pnl[w2].min())
    return out


def work(rows):
    res = []
    for r in rows:
        try:
            o = walk(r)
        except Exception as ex:
            o = dict(my_reason='ERROR', err=str(ex)[:200])
        o['key'] = r['key']
        res.append(o)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    A = ap.parse_args()
    d = pd.read_csv(os.path.join(REP, 'ENGINE_REPLAY_YR5_ML_fills.csv'), low_memory=False)
    d['open_ms'] = pd.to_datetime(d.opened_at).values.astype('datetime64[ms]').astype(np.int64)
    d['key'] = d.seed.astype(str) + '|' + d.pair + '|' + d.opened_at.astype(str)
    if A.limit:
        d = d.sample(A.limit, random_state=7)
    rows = d[['key', 'pair', 'entry_price', 'open_ms', 'entry_order_type', 'entry_atr_pct']].to_dict('records')
    by = {}
    for r in rows:
        by.setdefault(r['pair'], []).append(r)
    batches = []
    for pr, lst in by.items():
        lst.sort(key=lambda r: r['open_ms'])
        for k in range(0, len(lst), 40):
            batches.append(lst[k:k + 40])
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(A.jobs) as ex:
        for k, res in enumerate(ex.map(work, batches)):
            out.extend(res)
            if k % 20 == 0:
                print(f'{k}/{len(batches)} batches {time.time() - t0:.0f}s', flush=True)
    o = pd.DataFrame(out)
    m = d.merge(o, on='key', how='left')
    keep = ['key', 'seed', 'chunk', 'pair', 'opened_at', 'closed_at', 'close_reason', 'pct', 'peak_pnl', 'entry_price', 'exit_price',
            'entry_order_type', 'entry_atr_pct', 'entry_btc_rsi_closed', 'rh_triggered_at', 'rh_trigger_pnl', 'signal_active_at_close',
            'mon', 'half'] + [c for c in o.columns if c != 'key']
    m[keep].to_csv(os.path.join(REP, 'study_ml_bughunt_exitwalk.csv'), index=False)
    print('wrote', len(m), 'rows', f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
