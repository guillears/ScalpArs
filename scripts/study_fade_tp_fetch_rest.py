#!/usr/bin/env python3
"""SPIKE_FADE fixed-TP study (2026-10-09) — REST aggTrades for a pair-window the daily archive does not hold yet (today's fills).

GET fapi/v1/aggTrades (weight 20/call, ≤ 1 h span, 1000 trades/call), serial. Budget rules (feedback_binance_request_budget):
honours reports/backtest_cache/.binance_cooldown_until, reads X-MBX-USED-WEIGHT-1M after every call, pauses ~65 s when it
reaches 900, stops on 418/429. Writes reports/study_fade_tp_ticks/<PAIR>/<date>.npz (t, p, q) covering only the requested
window — a partial day, so the study walker must never read it outside that window (it only reads entry → +3 h 10 min).
Usage: venv/bin/python scripts/study_fade_tp_fetch_rest.py PAIR START_UTC END_UTC   (e.g. ARXUSDT "2026-10-09 18:49" "2026-10-09 22:20")"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import json, os, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "reports", "backtest_cache", ".binance_cooldown_until")


def cooldown():
    if os.path.exists(CD):
        try:
            until = _BC.read_until_ms(CD) / 1000.0
        except ValueError:
            until = 0
        if until > time.time():
            sys.exit(f"cooldown active until {until} — not fetching")


def main(pair, s, e):
    cooldown()
    t0 = int(pd.Timestamp(s).value // 1_000_000); t1 = int(pd.Timestamp(e).value // 1_000_000)
    T, P, Q = [], [], []
    cur = t0
    while cur < t1:
        end = min(cur + 3_600_000 - 1, t1)
        url = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={pair}&startTime={cur}&endTime={end}&limit=1000"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
                rows = json.loads(r.read())
        except urllib.error.HTTPError as ex:
            if ex.code in (418, 429):
                sys.exit(f"HTTP {ex.code} — stopping")
            raise
        for x in rows:
            T.append(int(x["T"])); P.append(float(x["p"])); Q.append(float(x["q"]))
        cur = (int(rows[-1]["T"]) + 1) if len(rows) == 1000 else end + 1
        print(f"{pd.Timestamp(cur, unit='ms')} rows {len(rows)} weight {w}", flush=True)
        time.sleep(65 if w >= 900 else 0.25)
    ds = pd.Timestamp(t0, unit="ms").strftime("%Y-%m-%d")
    fp = os.path.join(ROOT, "reports", "study_fade_tp_ticks", pair, f"{ds}.npz")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    np.savez_compressed(fp, t=np.array(T, dtype=np.int64), p=np.array(P, dtype=np.float32), q=np.array(Q, dtype=np.float32))
    print(f"wrote {fp}: {len(T)} trades")


if __name__ == "__main__":
    main(*sys.argv[1:4])
