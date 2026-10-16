#!/usr/bin/env python3
"""LITE MACRO-REGIME study — gap fetch (cache ends 2026-10-12 04:00 UTC; master B21 + live days are 10-13..10-15).
Daily (1d, 2026-09-01 ->) and 5m (2026-10-11 ->) klines for every pair in reports/backtest_cache/k5m_full. Public fapi, read-only.
Budget <= 1,200 weight/min: reads X-MBX-USED-WEIGHT-1M, pauses above 700, stops on 418/429 and writes / respects
reports/backtest_cache/.binance_cooldown_until. Output reports/lite_regime/fetch/{k1d,k5m}/<PAIR>.csv (outside backtest_cache). Resumable.
Usage: venv/bin/python scripts/study_lite_regime_fetch.py
"""
import os, sys, time, glob
import pandas as pd, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD = os.path.join(ROOT, "reports", "backtest_cache", ".binance_cooldown_until")
OUT = os.path.join(ROOT, "reports", "lite_regime", "fetch")
URL = "https://fapi.binance.com/fapi/v1/klines"
STATE = {"w": 0, "calls": 0}
COLS = ["open_time", "o", "h", "l", "c", "vol", "close_time", "qvol"]


def cooldown_check():
    if os.path.exists(CD):
        try:
            if float(open(CD).read().strip() or 0) > time.time():
                raise SystemExit("cooldown active — stop")
        except ValueError:
            pass


def get(params):
    cooldown_check()
    if STATE["w"] > 700:
        print(f"  used weight {STATE['w']} > 700 -> sleep 61 s", flush=True)
        time.sleep(61); STATE["w"] = 0
    r = requests.get(URL, params=params, timeout=20)
    STATE["w"] = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
    STATE["calls"] += 1
    if r.status_code in (418, 429):
        ra = int(r.headers.get("Retry-After", "120") or 120)
        open(CD, "w").write(str(time.time() + max(ra, 120)))
        raise SystemExit(f"rate limited {r.status_code} — cooldown written, stop")
    if r.status_code == 400:
        return []
    r.raise_for_status()
    time.sleep(0.25)
    return r.json()


def fetch(sym, interval, s, e, limit):
    rows = []
    while s < e:
        k = get(dict(symbol=sym, interval=interval, startTime=int(s), endTime=int(e) - 1, limit=limit))
        if not k:
            break
        rows += k
        s = int(k[-1][0]) + 1
        if len(k) < limit:
            break
    return rows


def save(rows, f):
    df = pd.DataFrame([r[:8] for r in rows], columns=COLS)[["open_time", "o", "h", "l", "c", "vol", "qvol"]]
    df.to_csv(f, index=False)


def main():
    pairs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "reports", "backtest_cache", "k5m_full", "*.csv")))
    now = int(time.time() * 1000)
    s1d = int(pd.Timestamp("2026-09-01", tz="UTC").timestamp() * 1000)
    s5m = int(pd.Timestamp("2026-10-11", tz="UTC").timestamp() * 1000)
    for d in ("k1d", "k5m"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    t0 = time.time()
    for i, p in enumerate(pairs):
        f1, f5 = os.path.join(OUT, "k1d", p + ".csv"), os.path.join(OUT, "k5m", p + ".csv")
        if not os.path.exists(f1):
            save(fetch(p, "1d", s1d, now, 99), f1)          # weight 1
        if not os.path.exists(f5):
            save(fetch(p, "5m", s5m, now, 1000), f5)        # weight 5 per call, 2 calls
        if i % 50 == 0:
            print(f"{i + 1}/{len(pairs)} {p} calls {STATE['calls']} used-weight {STATE['w']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("done", STATE)


if __name__ == "__main__":
    main()
