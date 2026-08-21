#!/usr/bin/env python3
"""WIDE-SL OBSERVE instrument (registered 2026-08-19; see CURRENT_STATE gate).

For every momentum-LONG stopped at the fixed SL (close_reason STOP, pnl<=-0.6),
replay the full 1m path from entry: does it reach +0.40 (arm) before -W?
ARMED trades are banked at the BE floor (+0.20) ONLY — runner upside is shown
(48h max) but never credited. This is the UNBIASED method for the wide-SL
question: the post_exit_* columns are 6h-censored and systematically
understate wide-SL benefit (method correction, DECISION_LOG 2026-08-19).

Usage: venv/bin/python scripts/wide_sl_armtest.py <orders_csv> [W=2.0]
Master pool is always included. 1m data: reports/backtest_cache/k1m/ then live fetch.
"""
import os, sys, json, urllib.request
from urllib.parse import quote
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
K1M = os.path.join(ROOT, "reports", "backtest_cache", "k1m")

def fetch1m(sym, start_ms, bars=2880):
    rows, cur = [], start_ms
    while len(rows) < bars:
        url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={quote(sym)}"
               f"&interval=1m&startTime={cur}&limit=1500")
        with urllib.request.urlopen(url, timeout=30) as r:
            kl = json.loads(r.read())
        if not kl:
            break
        rows.extend(kl)
        cur = kl[-1][0] + 1
        if len(kl) < 1500:
            break
    return [(float(k[2]), float(k[3]), float(k[4])) for k in rows[:bars]]

def path1m(sym, start_ms, bars=2880):
    fp = os.path.join(K1M, f"{sym}.csv")
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        df = df[df['open_time'] >= start_ms].head(bars)
        if len(df) >= 360:
            return list(zip(df['h'].astype(float), df['l'].astype(float), df['c'].astype(float)))
    return fetch1m(sym, start_ms, bars)

def armtest(trades, name, W):
    print(f"\n=== {name} @ SL -{W} (arm +0.40 first?) ===")
    tot, armed, blown = 0.0, 0, 0
    for _, r in trades.iterrows():
        ep = float(r['entry_price'])
        start = int(pd.Timestamp(r['opened_at']).value // 1e6)
        path = path1m(r['pair'], start)
        if not path:
            print(f"   {r['pair']:12s} NO DATA")
            continue
        res = None
        for i, (h, l, c) in enumerate(path):
            if (l - ep) / ep * 100 <= -W:
                res = ('BLOWN', -W, i); blown += 1; break
            if (h - ep) / ep * 100 >= 0.40:
                res = ('ARMED', 0.20, i); armed += 1; break
        if res is None:
            res = ('NEITHER', (path[-1][2] - ep) / ep * 100, len(path))
        dpp = abs(r['pnl_dollar'] / r['pnl_percentage'])
        d = (res[1] - r['pnl_percentage']) * dpp
        tot += d
        mx = max((h - ep) / ep * 100 for h, l, c in path)
        print(f"   {r['pair']:12s} act {r['pnl_percentage']:+.2f} → {res[0]:7s} @bar{res[2]:4d} "
              f"floorΔ${d:+6.0f}  48h max {mx:+.2f}%")
    print(f"   ARMED {armed} · BLOWN {blown} | floor Δ$ {tot:+.0f}")
    return armed, blown, tot

def sl_losers(df, pnl_col):
    ml = df[(df['direction'] == 'LONG')
            & (df['entry_strategy'].fillna('MOMENTUM').isin(['MOMENTUM', '']))]
    sl = ml[ml['close_reason'].str.contains('STOP', case=False, na=False)
            & (ml['pnl_percentage'] <= -0.6)].copy()
    sl['pnl_dollar'] = sl[pnl_col]
    return sl

def main():
    W = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    mas = pd.read_csv(os.path.join(ROOT, 'reports', 'MASTER_POOL_stacked.csv'), low_memory=False)
    mas = mas[(mas['stack_keep'] == True) & (~mas['is_probe'].fillna(False).astype(bool))]
    armtest(sl_losers(mas, 'stack_pnl'), 'MASTER', W)
    if len(sys.argv) > 1:
        cur = pd.read_csv(sys.argv[1], low_memory=False)
        cur = cur[cur['status'].str.upper().eq('CLOSED')]
        armtest(sl_losers(cur, 'pnl'), 'CURRENT batch', W)

if __name__ == '__main__':
    main()
