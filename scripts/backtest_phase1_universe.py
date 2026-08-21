#!/usr/bin/env python3
"""Phase-1 backtest — step A: reconstruct the daily top-50 USDT-perp universe
Jan-01→Jul-31 2026 from Binance daily klines (quote volume), survivorship-free.

Output: reports/backtest_cache/universe_daily_top50.csv  (date, rank, pair, quote_vol)
Cache:  reports/backtest_cache/daily/<PAIR>.csv  (one file per pair, 1d klines)
"""
import json, os, sys, time
import urllib.request
from urllib.parse import quote

BASE = "https://fapi.binance.com"
CACHE = os.path.join(os.path.dirname(__file__), "..", "reports", "backtest_cache")
START_MS = 1767225600000  # 2026-01-01 UTC
END_MS   = 1787270399000  # 2026-08-19 23:59:59 UTC (extended for current-batch validation)

os.makedirs(os.path.join(CACHE, "daily"), exist_ok=True)

def get(url, retries=5):
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))

def main():
    info = get(f"{BASE}/fapi/v1/exchangeInfo")
    syms = [s["symbol"] for s in info["symbols"]
            if s["symbol"].endswith("USDT") and s.get("contractType") == "PERPETUAL"
            and s.get("status") in ("TRADING", "SETTLING")]
    print(f"USDT perps: {len(syms)}", flush=True)

    rows = []
    for i, sym in enumerate(syms):
        fp = os.path.join(CACHE, "daily", f"{sym}.csv")
        if os.path.exists(fp):
            with open(fp) as f:
                for line in f:
                    rows.append(line.strip().split(","))
            continue
        kl = get(f"{BASE}/fapi/v1/klines?symbol={quote(sym)}&interval=1d&startTime={START_MS}&endTime={END_MS}&limit=1500")
        with open(fp, "w") as f:
            for k in kl:
                # open_time, quote_volume
                f.write(f"{sym},{k[0]},{k[7]}\n")
                rows.append([sym, str(k[0]), str(k[7])])
        if i % 25 == 0:
            print(f"  {i}/{len(syms)} {sym} ({len(kl)} days)", flush=True)
        time.sleep(0.15)  # stay far under weight limits

    # daily top-50: day D ranked by day D-1 volume (NO look-ahead — the live
    # engine ranks by rolling 24h volume; prior-day volume is the honest proxy)
    by_day = {}
    for sym, ts, qv in rows:
        by_day.setdefault(int(ts), []).append((float(qv), sym))
    DAY = 86400_000
    out = os.path.join(CACHE, "universe_daily_top50.csv")
    union_set = set()
    with open(out, "w") as f:
        f.write("date_ms,rank,pair,quote_vol\n")
        for ts in sorted(by_day):
            src = ts - DAY
            if src not in by_day:
                continue
            top = sorted(by_day[src], reverse=True)[:50]
            for r, (qv, sym) in enumerate(top, 1):
                f.write(f"{ts},{r},{sym},{qv:.0f}\n")
                union_set.add(sym)
    union = sorted(union_set)
    print(f"days: {len(by_day)} | union of all top-50 members: {len(union)} pairs", flush=True)
    with open(os.path.join(CACHE, "union_pairs.txt"), "w") as f:
        f.write("\n".join(union))
    print("done", flush=True)

if __name__ == "__main__":
    main()
