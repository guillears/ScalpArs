#!/usr/bin/env python3
"""Phase-1 backtest — step B (span-based): download 5m klines only for each
pair's top-50 MEMBER DAYS (plus 3-day indicator warmup before each span, spans
merged when <5 days apart). BTCUSDT gets the full range 5m + 1h (macro gates).

Cache: reports/backtest_cache/k5m/<PAIR>.csv   (open_time,o,h,l,c,vol,qvol)
       reports/backtest_cache/btc_5m.csv, btc_1h.csv
Resumable: per-pair .done marker files.
"""
import json, os, time
import urllib.request
from urllib.parse import quote

BASE = "https://fapi.binance.com"
CACHE = os.path.join(os.path.dirname(__file__), "..", "reports", "backtest_cache")
START_MS = 1767225600000            # 2026-01-01 UTC
END_MS   = 1787270399000  # 2026-08-19 23:59:59 UTC (extended for current-batch validation)
DAY = 86400_000
K5M = os.path.join(CACHE, "k5m")
os.makedirs(K5M, exist_ok=True)

def get(url, retries=6):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))

def fetch_series(sym, interval, start, end):
    rows, cur = [], start
    while cur < end:
        kl = get(f"{BASE}/fapi/v1/klines?symbol={quote(sym)}&interval={interval}"
                 f"&startTime={cur}&endTime={end}&limit=1500")
        if not kl:
            break
        rows.extend(kl)
        nxt = kl[-1][0] + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.12)
    return rows

def write_rows(path, rows, append=False):
    mode = "a" if append else "w"
    with open(path, mode) as f:
        if not append:
            f.write("open_time,o,h,l,c,vol,qvol\n")
        for k in rows:
            f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]}\n")

def main():
    # member days per pair
    member = {}
    with open(os.path.join(CACHE, "universe_daily_top50.csv")) as f:
        next(f)
        for line in f:
            ts, rank, pair, qv = line.rstrip("\n").split(",")
            member.setdefault(pair, []).append(int(ts))

    # BTC full range (needed every scan regardless of membership)
    for iv, fn in [("5m", "btc_5m.csv"), ("1h", "btc_1h.csv")]:
        fp = os.path.join(CACHE, fn)
        if not os.path.exists(fp + ".done"):
            rows = fetch_series("BTCUSDT", iv, START_MS - 30 * DAY, END_MS)
            write_rows(fp, rows)
            open(fp + ".done", "w").close()
            print(f"BTC {iv}: {len(rows)} bars", flush=True)

    # ETH context pair: needed full-range for the volume/breadth panel
    member.setdefault('ETHUSDT', [])
    member['ETHUSDT'] = sorted(set(member['ETHUSDT']) |
                               set(range(START_MS, END_MS, DAY)))

    def covers(fp, need_start, need_end):
        """existing file already spans the needed range?"""
        try:
            with open(fp) as f:
                next(f)
                first = int(f.readline().split(",")[0])
            with open(fp, "rb") as f:
                f.seek(max(-300, -os.path.getsize(fp)), 2)
                last = int(f.read().decode().strip().split("\n")[-1].split(",")[0])
            return first <= need_start + 3600_000 and last >= need_end - 900_000
        except Exception:
            return False

    pairs = sorted(member)
    total_days = sum(len(v) for v in member.values())
    print(f"pairs: {len(pairs)} | member pair-days: {total_days}", flush=True)
    for i, sym in enumerate(pairs):
        fp = os.path.join(K5M, f"{sym}.csv")
        days = sorted(member[sym])
        # merge member days into spans (warmup 3d before; merge gaps < 5d)
        spans = []
        for d in days:
            s, e = d - 3 * DAY, d + DAY
            if spans and s - spans[-1][1] < 5 * DAY:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        if os.path.exists(fp) and all(covers(fp, s, min(e, END_MS)) for s, e in [(spans[0][0], spans[-1][1])]) \
                and len(spans) == 1:
            continue
        if os.path.exists(fp + ".done") and spans and covers(fp, spans[0][0], min(spans[-1][1], END_MS)):
            continue
        try:
            first = True
            n = 0
            for s, e in spans:
                rows = fetch_series(sym, "5m", max(s, START_MS - 3 * DAY), min(e, END_MS))
                write_rows(fp, rows, append=not first)
                n += len(rows)
                first = False
            open(fp + ".done", "w").close()
        except Exception as ex:
            print(f"  FAIL {sym}: {ex}", flush=True)
            continue
        if i % 20 == 0:
            print(f"  {i}/{len(pairs)} {sym} ({len(days)}d/{len(spans)} spans, {n} bars)", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
