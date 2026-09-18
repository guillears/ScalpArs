#!/usr/bin/env python3
"""Spike-scanner universe: FULL-RANGE 5m klines (Jan-01 → END) for every USDT perp that
had at least one day with quote volume ≥ the scanner floor ($2M). Cache:
reports/backtest_cache/k5m_full/<PAIR>.csv (append-only, resumable)."""
import json, os, sys, time, glob
import urllib.request
from urllib.parse import quote

BASE = "https://fapi.binance.com"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "reports", "backtest_cache")
OUTD = os.path.join(CACHE, "k5m_full"); os.makedirs(OUTD, exist_ok=True)
START_MS = 1767225600000 - 2 * 86400_000
END_MS = int(sys.argv[1]) if len(sys.argv) > 1 else (int(time.time() * 1000) // 300_000) * 300_000 - 1
FLOOR = 2_000_000.0
SLEEP = float(os.environ.get("FETCH_SLEEP", "0.25"))

def get(url, retries=8):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception as ex:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))

def last_ts(fp):
    try:
        with open(fp, "rb") as f:
            f.seek(max(-400, -os.path.getsize(fp)), 2)
            for line in reversed(f.read().decode().strip().split("\n")):
                try:
                    return int(line.split(",")[0])
                except ValueError:
                    pass
    except Exception:
        pass
    return None

def main():
    pairs = []
    for fp in sorted(glob.glob(os.path.join(CACHE, "daily", "*.csv"))):
        sym = os.path.basename(fp)[:-4]
        if any(float(l.strip().split(",")[2]) >= FLOOR for l in open(fp) if l.count(",") == 2):
            pairs.append(sym)
    print(f"pairs with a ≥$2M day: {len(pairs)} | END={END_MS}", flush=True)
    for i, sym in enumerate(pairs):
        fp = os.path.join(OUTD, f"{sym}.csv")
        lt = last_ts(fp) if os.path.exists(fp) else None
        cur = (lt + 1) if lt else START_MS
        if cur >= END_MS - 300_000:
            continue
        n = 0
        try:
            with open(fp, "a") as f:
                if lt is None:
                    f.write("open_time,o,h,l,c,vol,qvol\n")
                while cur < END_MS:
                    kl = get(f"{BASE}/fapi/v1/klines?symbol={quote(sym)}&interval=5m&startTime={cur}&endTime={END_MS}&limit=1500")
                    if not kl:
                        break
                    for k in kl:
                        f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]}\n")
                    n += len(kl)
                    nxt = kl[-1][0] + 1
                    if nxt <= cur:
                        break
                    cur = nxt
                    time.sleep(SLEEP)
        except Exception as ex:
            print(f"  FAIL {sym}: {ex}", flush=True)
            continue
        if i % 10 == 0 or n > 20000:
            print(f"  {i}/{len(pairs)} {sym} +{n}", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
