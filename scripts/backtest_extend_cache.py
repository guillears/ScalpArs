#!/usr/bin/env python3
"""Extend reports/backtest_cache to END (default: now) — daily universe (top-50 by
prior-day quote volume, same rule as backtest_phase1_universe.py), member-span
5m + 1m klines (same span logic as the phase-1/phase-2 fetchers), BTC/ETH full-range.

Append-only: every cached file is extended from its last open_time; new pairs get
their full spans. Idempotent (re-run = no-op once caught up)."""
import json, os, sys, time
import urllib.request
from urllib.parse import quote

BASE = "https://fapi.binance.com"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(ROOT, "reports", "backtest_cache")
START_MS = 1767225600000                      # 2026-01-01 UTC
DAY = 86400_000
END_MS = int(sys.argv[1]) if len(sys.argv) > 1 else (int(time.time() * 1000) // 300_000) * 300_000 - 1

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

def last_ts(fp):
    try:
        with open(fp, "rb") as f:
            f.seek(max(-400, -os.path.getsize(fp)), 2)
            tail = f.read().decode().strip().split("\n")
        for line in reversed(tail):
            try:
                return int(line.split(",")[0])
            except ValueError:
                continue
    except Exception:
        pass
    return None

def write_rows(path, rows, append):
    with open(path, "a" if append else "w") as f:
        if not append:
            f.write("open_time,o,h,l,c,vol,qvol\n")
        for k in rows:
            f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]}\n")

def extend_series(sym, interval, fp, spans, warm_ms):
    """spans = [[s,e],...] in ms (already merged). Fetch only what the file lacks."""
    lt = last_ts(fp) if os.path.exists(fp) else None
    n = 0
    for s, e in spans:
        s = max(s, START_MS - warm_ms)
        e = min(e, END_MS)
        if lt is not None:
            if e <= lt:
                continue
            s = max(s, lt + 1)
        if s >= e:
            continue
        rows = fetch_series(sym, interval, s, e)
        if rows:
            write_rows(fp, rows, append=os.path.exists(fp))
            lt = rows[-1][0]
            n += len(rows)
    return n

def main():
    # 1) daily klines for all USDT perps (append tail), then rebuild universe
    info = get(f"{BASE}/fapi/v1/exchangeInfo")
    syms = [s["symbol"] for s in info["symbols"]
            if s["symbol"].endswith("USDT") and s.get("contractType") == "PERPETUAL"
            and s.get("status") in ("TRADING", "SETTLING")]
    ddir = os.path.join(CACHE, "daily")
    os.makedirs(ddir, exist_ok=True)
    rows = []
    print(f"USDT perps: {len(syms)} | END={END_MS}", flush=True)
    for i, sym in enumerate(syms):
        fp = os.path.join(ddir, f"{sym}.csv")
        have = {}
        if os.path.exists(fp):
            with open(fp) as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) == 3:
                        have[int(p[1])] = p[2]
        start = (max(have) + DAY) if have else START_MS
        if start < END_MS - DAY:      # only fully closed days
            try:
                kl = get(f"{BASE}/fapi/v1/klines?symbol={quote(sym)}&interval=1d&startTime={start}&endTime={END_MS}&limit=1500")
            except Exception as ex:
                print(f"  daily FAIL {sym}: {ex}", flush=True); kl = []
            for k in kl:
                if k[6] < END_MS:          # closed day only
                    have[int(k[0])] = str(k[7])
            with open(fp, "w") as f:
                for ts in sorted(have):
                    f.write(f"{sym},{ts},{have[ts]}\n")
            time.sleep(0.12)
        for ts, qv in have.items():
            rows.append((sym, ts, float(qv)))
        if i % 100 == 0:
            print(f"  daily {i}/{len(syms)}", flush=True)
    by_day = {}
    for sym, ts, qv in rows:
        by_day.setdefault(ts, []).append((qv, sym))
    member, union = {}, set()
    with open(os.path.join(CACHE, "universe_daily_top50.csv"), "w") as f:
        f.write("date_ms,rank,pair,quote_vol\n")
        for ts in sorted(by_day):
            src = ts - DAY
            if src not in by_day:
                continue
            for r, (qv, sym) in enumerate(sorted(by_day[src], reverse=True)[:50], 1):
                f.write(f"{ts},{r},{sym},{qv:.0f}\n")
                member.setdefault(sym, []).append(ts)
                union.add(sym)
    with open(os.path.join(CACHE, "union_pairs.txt"), "w") as f:
        f.write("\n".join(sorted(union)))
    print(f"universe days: {len(by_day)} | members: {len(union)}", flush=True)

    # 2) BTC context series (full range)
    for iv, fn in [("5m", "btc_5m.csv"), ("1h", "btc_1h.csv")]:
        n = extend_series("BTCUSDT", iv, os.path.join(CACHE, fn), [[START_MS - 30 * DAY, END_MS]], 30 * DAY)
        print(f"BTC {iv}: +{n}", flush=True)
    for ctx in ("BTCUSDT", "ETHUSDT"):
        member[ctx] = sorted(set(member.get(ctx, [])) | set(range(START_MS, END_MS, DAY)))

    # 3) member-span 5m (3d warmup, merge <5d) and 1m (1d warmup, merge <3d)
    k5, k1 = os.path.join(CACHE, "k5m"), os.path.join(CACHE, "k1m")
    os.makedirs(k5, exist_ok=True); os.makedirs(k1, exist_ok=True)
    pairs = sorted(member)
    for i, sym in enumerate(pairs):
        days = sorted(member[sym])
        def spans_for(warm, gap):
            sp = []
            for d in days:
                s, e = d - warm * DAY, d + DAY
                if sp and s - sp[-1][1] < gap * DAY:
                    sp[-1][1] = max(sp[-1][1], e)
                else:
                    sp.append([s, e])
            return sp
        try:
            n5 = extend_series(sym, "5m", os.path.join(k5, f"{sym}.csv"), spans_for(3, 5), 3 * DAY)
            n1 = extend_series(sym, "1m", os.path.join(k1, f"{sym}.csv"), spans_for(1, 3), 1 * DAY)
            open(os.path.join(k5, f"{sym}.csv.done"), "w").close()
            open(os.path.join(k1, f"{sym}.csv.done"), "w").close()
        except Exception as ex:
            print(f"  FAIL {sym}: {ex}", flush=True)
            continue
        if n5 or n1 or i % 50 == 0:
            print(f"  {i}/{len(pairs)} {sym} +5m {n5} +1m {n1}", flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
