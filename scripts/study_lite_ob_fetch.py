#!/usr/bin/env python3
"""LITE OB study (2026-10-15) — budgeted public fetch of 5m (from 2026-08-29, for the frenzy_walk episode rebuild) and 1m klines WITH
taker-buy volume (from 2026-10-03 00:00) for every pair in the BOOK snapshots + BTCUSDT. Public fapi, read-only.
Budget: limit 1000 (weight 5), pause when X-MBX-USED-WEIGHT-1M > 900, stop on 418/429 and arm reports/backtest_cache/.binance_cooldown_until.
Output reports/lite_ob/k5m/<PAIR>.csv and reports/lite_ob/k1m/<PAIR>.csv. Resumable (skips a pair/interval already covered)."""
import os, sys, time
import pandas as pd, requests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports"); OUT = os.path.join(REP, "lite_ob")
CD = os.path.join(REP, "backtest_cache", ".binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines"
COLS = ["open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tbv", "tbqv", "ig"]
ST = {"w": 0, "calls": 0}


def get(params):
    if os.path.exists(CD):
        try:
            if float(open(CD).read().strip() or 0) > time.time():
                sys.exit("cooldown active — stop")
        except ValueError:
            pass
    if ST["w"] > 900:
        print(f"  used weight {ST['w']} > 900 → sleep 61 s", flush=True); time.sleep(61); ST["w"] = 0
    r = requests.get(URL, params=params, timeout=20)
    ST["w"] = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0); ST["calls"] += 1
    if r.status_code in (418, 429):
        ra = int(r.headers.get("Retry-After", "120") or 120)
        open(CD, "w").write(str(time.time() + max(ra, 120)))
        sys.exit(f"rate limited {r.status_code} — cooldown written, stop")
    if r.status_code == 400:
        return []
    r.raise_for_status(); time.sleep(0.25)
    return r.json()


def fetch(sym, iv, step, s, e, path):
    if os.path.exists(path):
        d = pd.read_csv(path)
        if len(d) and d.open_time.min() <= s and d.open_time.max() >= e - 3 * step:
            return
    rows = []
    while s < e:
        k = get(dict(symbol=sym, interval=iv, startTime=int(s), endTime=int(e) - 1, limit=1000))
        if not k:
            break
        rows += k; s = int(k[-1][0]) + step
        if len(k) < 1000:
            break
    if rows:
        pd.DataFrame([r[:12] for r in rows], columns=COLS).apply(pd.to_numeric).drop_duplicates("open_time").to_csv(path, index=False)


def main():
    B = pd.read_pickle(os.path.join(OUT, "book.pkl"))
    pairs = sorted(set(B.pair)) + ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    now = int(time.time() // 60) * 60_000
    s5 = int(pd.Timestamp("2026-08-29", tz="UTC").value // 1e6); s1 = int(pd.Timestamp("2026-10-03", tz="UTC").value // 1e6)
    for d in ("k5m", "k1m"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    t0 = time.time()
    for i, p in enumerate(pairs):
        fetch(p, "5m", 300_000, s5, now, os.path.join(OUT, "k5m", f"{p}.csv"))
        fetch(p, "1m", 60_000, s1, now, os.path.join(OUT, "k1m", f"{p}.csv"))
        print(f"{i + 1}/{len(pairs)} {p} · calls {ST['calls']} · w {ST['w']} · {time.time() - t0:.0f}s", flush=True)
    print("done", ST)


if __name__ == "__main__":
    main()
