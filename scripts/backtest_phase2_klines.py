#!/usr/bin/env python3
"""Phase-1.5 backtest — 1m klines for member spans (same span logic as phase-1
5m downloader) + BTC/ETH full-range 1m. Cache: reports/backtest_cache/k1m/."""
import json, os, time
import urllib.request
from urllib.parse import quote

BASE = "https://fapi.binance.com"
CACHE = os.path.join(os.path.dirname(__file__), "..", "reports", "backtest_cache")
START_MS = 1767225600000
END_MS   = 1787270399000  # 2026-08-19 23:59:59 UTC (extended for current-batch validation)
DAY = 86400_000
K1M = os.path.join(CACHE, "k1m")
os.makedirs(K1M, exist_ok=True)

def get(url, retries=6):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))

def fetch(sym, start, end, out, append=False):
    rows, cur = [], start
    while cur < end:
        kl = get(f"{BASE}/fapi/v1/klines?symbol={quote(sym)}&interval=1m&startTime={cur}&endTime={end}&limit=1500")
        if not kl:
            break
        rows.extend(kl)
        nxt = kl[-1][0] + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.1)
    mode = "a" if append else "w"
    with open(out, mode) as f:
        if not append:
            f.write("open_time,o,h,l,c,vol,qvol\n")
        for k in rows:
            f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]}\n")
    return len(rows)

def main():
    member = {}
    with open(os.path.join(CACHE, "universe_daily_top50.csv")) as f:
        next(f)
        for line in f:
            ts, rank, pair, qv = line.rstrip("\n").split(",")
            member.setdefault(pair, []).append(int(ts))
    for ctx in ("BTCUSDT", "ETHUSDT"):
        member[ctx] = sorted(set(member.get(ctx, [])) | set(range(START_MS, END_MS, DAY)))

    pairs = sorted(member)
    print(f"pairs: {len(pairs)} | member pair-days: {sum(len(v) for v in member.values())}", flush=True)
    for i, sym in enumerate(pairs):
        fp = os.path.join(K1M, f"{sym}.csv")
        if os.path.exists(fp + ".done"):
            continue
        days = sorted(member[sym])
        spans = []
        for d in days:
            s, e = d - 1 * DAY, d + DAY   # 1-day warmup is enough: 5m context comes from k5m
            if spans and s - spans[-1][1] < 3 * DAY:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        try:
            first, n = True, 0
            for s, e in spans:
                n += fetch(sym, max(s, START_MS - DAY), min(e, END_MS), fp, append=not first)
                first = False
            open(fp + ".done", "w").close()
        except Exception as ex:
            print(f"  FAIL {sym}: {ex}", flush=True)
            continue
        if i % 20 == 0:
            print(f"  {i}/{len(pairs)} {sym} ({n} bars)", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
