"""ML BTC-REACTIVE EXIT study (2026-10-26) — BTCUSDT 1m top-up after the backtest_cache end.

The BTC tick archive (reports/backtest_cache/ticks_q/BTCUSDT) ends 2026-10-11 and the 1m cache (k1m/BTCUSDT.csv) ends 2026-10-12 04:09;
the PREP_B25 fetch only holds windows around its own fills. This fetches the CONTIGUOUS BTCUSDT 1m series from 2026-10-12 00:00 UTC to
now (public fapi klines, 499 bars per call, weight 2), so every master B19–B23 + B25 fill has a BTC path.

Budget: ≤ 500 weight/min of our own calls, reads x-mbx-used-weight-1m (pause 60 s when the IP total > 900), stops on 418/429 and writes
the shared cooldown file (max of the existing deadline and now + Retry-After or 15 min), retries any other error
up to 3× then exits non-zero (no silent holes), honours scripts/binance_cooldown.py before every call. Output: reports/ML_BTC_EXIT_2026-10-26_fetch/k1m/BTCUSDT.csv

    venv/bin/python -W ignore scripts/study_ml_btcexit_fetch.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binance_cooldown as BC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "ML_BTC_EXIT_2026-10-26_fetch", "k1m")
URL = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&startTime={a}&endTime={b}&limit=499"
MY = []


def _budget(w):
    while True:
        now = time.time()
        MY[:] = [(t, x) for t, x in MY if now - t < 60]
        if sum(x for _, x in MY) + w <= 500:
            break
        time.sleep(2)
    MY.append((time.time(), w))


def _arm_cooldown(retry_after):
    """shared cooldown = max(existing deadline, now + Retry-After or 15 min), in ms (binance_cooldown reads both units)."""
    try:
        wait_ms = int(float(retry_after) * 1000) if retry_after else 15 * 60_000
    except (TypeError, ValueError):
        wait_ms = 15 * 60_000
    until = max(BC.read_until_ms(), int(time.time() * 1000) + wait_ms)
    with open(BC.SHARED, "w") as fh:
        fh.write(str(until))
    return until


def get(a, b, tries=3):
    """one klines call; 418/429 → arm the shared cooldown and exit non-zero; any other error → retry (≤ 3), then exit non-zero."""
    last = None
    for k in range(tries):
        if BC.active():
            sys.exit("shared Binance cooldown active — stopping")
        _budget(2)
        req = urllib.request.Request(URL.format(a=a, b=b), headers={"User-Agent": "scalpars-study"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                used = int(r.headers.get("x-mbx-used-weight-1m", "0"))
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                until = _arm_cooldown(e.headers.get("Retry-After") if e.headers else None)
                sys.exit(f"HTTP {e.code} — shared cooldown until {until} ms, stopping")
            last = f"HTTP {e.code}"
        except Exception as e:  # network / timeout / bad JSON
            last = f"{type(e).__name__}: {e}"
        else:
            if used > 900:
                print(f"IP used weight {used} — pausing 60 s", flush=True); time.sleep(60)
            time.sleep(0.3)
            return data, used
        print(f"call [{a}, {b}] failed ({last}) — retry {k + 1}/{tries}", flush=True)
        time.sleep(2 * (k + 1))
    sys.exit(f"call [{a}, {b}] failed {tries}× ({last}) — stopping (no silent hole)")


def main():
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, "BTCUSDT.csv")
    a = int(pd.Timestamp("2026-10-12 00:00").value // 10**6)
    if os.path.exists(f):
        a = max(a, int(pd.read_csv(f).open_time.max()) + 60_000)
    now = int(time.time() * 1000)
    rows, calls, peak = [], 0, 0
    while a < now:
        b = min(a + 499 * 60_000 - 1, now)
        d, used = get(a, b); calls += 1; peak = max(peak, used)
        rows += [[int(r[0]), r[1], r[2], r[3], r[4], r[5], r[7]] for r in d]
        a = b + 1
    df = pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    if os.path.exists(f):
        df = pd.concat([pd.read_csv(f), df])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df[df.open_time + 60_000 <= now]             # closed bars only
    df.to_csv(f, index=False)
    print(f"done: {calls} calls, own weight {calls * 2}, IP peak used {peak}, bars {len(df)}")


if __name__ == "__main__":
    main()
