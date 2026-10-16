#!/usr/bin/env python3
"""ML_DOWNTREND_2D follow-up — fetch 1h / 4h Binance futures klines (real high/low) for the fresh/master fills whose cached 5m
series is close-only (ADX impossible from cache), plus 3 parity symbols. ≤ 2 weight per request; reads x-mbx-used-weight-1m,
pauses at ≥ 700, stops on 418/429 and arms reports/backtest_cache/.binance_cooldown_until. Writes reports/ml_dt2d_fetch/{1h,4h}/.
    venv/bin/python -W ignore scripts/study_ml_dt2d_fu_fetch.py SYM [SYM ...]
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOL = os.path.join(ROOT, "reports/backtest_cache/.binance_cooldown_until")
OUT = os.path.join(ROOT, "reports/ml_dt2d_fetch")


def cooldown_live():
    try:
        return _BC.active(COOL)
    except Exception:
        return False


def main(syms):
    n = 0
    for sym in syms:
        for tf, lim in (("1h", 499), ("4h", 499)):
            if cooldown_live():
                print("cooldown live — stop"); return
            f = os.path.join(OUT, tf, f"{sym}.csv")
            if os.path.exists(f):
                continue
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={tf}&limit={lim}"
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    rows = json.loads(r.read().decode()); used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
            except urllib.error.HTTPError as e:
                print(sym, tf, "HTTP", e.code)
                if e.code in (418, 429):
                    ra = int(e.headers.get("Retry-After") or 120)
                    open(COOL, "w").write(str(int((time.time() + ra) * 1000)))
                    return
                continue
            n += 1
            os.makedirs(os.path.dirname(f), exist_ok=True)
            pd.DataFrame([(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), int(x[6])) for x in rows],
                         columns=["open_time", "o", "h", "l", "c", "close_time"]).to_csv(f, index=False)
            print(f"{sym} {tf} {len(rows)} bars · used weight {used}", flush=True)
            if used >= 700:
                print("used weight ≥ 700 — pausing 60 s"); time.sleep(60)
            time.sleep(0.5)
    print("requests:", n)


if __name__ == "__main__":
    main(sys.argv[1:])
