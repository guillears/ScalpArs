#!/usr/bin/env python3
"""NEGFLANK 2D study — tiny cache extension (k5m_full alt files end 2026-10-04; B17/B18 fills are Oct-05..07).

Fetches Binance USDT-M public klines, sequential, cache-first, budget-guarded (≤ 1,200 weight/min; reads X-MBX-USED-WEIGHT-1M,
pauses near 900, stops on 418/429):
  * 5m klines 2026-10-03 00:00 → 2026-10-08 18:00 for ETHUSDT + the pairs of master fills after 2026-10-04 (limit 1500, weight 10)
  * 1h klines 2026-10-02 00:00 → 2026-10-08 18:00 for every other k5m_full symbol (alt index; limit 170, weight 2)
Output: reports/backtest_cache/negflank2d_ext/{5m,1h}/<SYM>.csv (open_time,o,h,l,c,vol,qvol).
"""
import json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "reports/backtest_cache/negflank2d_ext"
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&limit={n}"
START5, START1, END = 1791072000000 - 86400_000, 1791072000000 - 2 * 86400_000, 1791482400000  # Oct-03 / Oct-02 → Oct-08 18:00
used = 0


def get(sym, itv, start, lim):
    global used
    if used >= 900:
        print("pause (weight", used, ")"); time.sleep(61); used = 0
    req = urllib.request.Request(URL.format(s=sym, i=itv, a=start, n=lim), headers={"User-Agent": "research"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", used))
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            print("STOP: HTTP", e.code); sys.exit(2)
        return None


def save(sym, itv, rows):
    os.makedirs(f"{OUT}/{itv}", exist_ok=True)
    with open(f"{OUT}/{itv}/{sym}.csv", "w") as f:
        f.write("open_time,o,h,l,c,vol,qvol\n")
        for k in rows:
            if k[0] < END:
                f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]}\n")


def main(pairs5):
    syms = sorted(f[:-4] for f in os.listdir("reports/backtest_cache/k5m_full"))
    for s in ["ETHUSDT"] + sorted(pairs5):
        if os.path.exists(f"{OUT}/5m/{s}.csv"):
            continue
        r = get(s, "5m", START5, 1500)
        if r:
            save(s, "5m", r)
    for s in syms:
        if s == "BTCUSDT" or not s.isascii() or os.path.exists(f"{OUT}/1h/{s}.csv") or os.path.exists(f"{OUT}/1h/{s}.none"):
            continue
        r = get(s, "1h", START1, 170)
        if r:
            save(s, "1h", r)
        else:
            os.makedirs(f"{OUT}/1h", exist_ok=True); open(f"{OUT}/1h/{s}.none", "w").close()
    print("done, last used weight", used)


if __name__ == "__main__":
    main(sys.argv[1:])
