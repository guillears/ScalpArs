"""ML EXIT DEEP study (2026-10-26) — minimal Binance top-up of 1m price paths for the momentum-long fills that have neither a tick
archive day nor ≥ 6 h of cached 1m bars (master B19–B23 + the B25 provisional export), plus BTC 5m after the cache end (for the
recovery-hold BTC-RSI ruler on the newest fills).

Fetches ONLY public fapi klines: per fill pair 1m over [entry − 10 min, entry + 9 h] (2 calls of ≤ 499 bars, weight 2 each);
BTCUSDT 5m from the PREP 5m cache end to now. Writes to reports/ML_EXIT_DEEP_2026-10-26_fetch/{k1m,k5m}/ (never into backtest_cache).
Budget: ≤ 500 weight/min of OUR OWN calls (another agent shares the IP), reads x-mbx-used-weight-1m (pause 60 s when the IP total
> 900), stops on 418/429 and writes the shared cooldown file, honours scripts/binance_cooldown.py before every call.

    venv/bin/python -W ignore scripts/study_ml_exit_b25_fetch.py
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
import study_ml_exit_b25_data as DD  # noqa: E402

URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit={n}"
OUT = DD.FETCH
MY = []          # (epoch s, weight) of our own calls in the last minute
WEIGHT = {True: 2}


def _budget(w):
    now = time.time()
    while True:
        MY[:] = [(t, x) for t, x in MY if now - t < 60]
        if sum(x for _, x in MY) + w <= 500:
            break
        time.sleep(2); now = time.time()
    MY.append((now, w))


def get(sym, iv, a, b, n):
    if BC.active():
        sys.exit("shared Binance cooldown active — stopping")
    w = 1 if n < 100 else 2 if n < 500 else 5
    _budget(w)
    req = urllib.request.Request(URL.format(s=sym, i=iv, a=a, b=b, n=n), headers={"User-Agent": "scalpars-study"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            used = int(r.headers.get("x-mbx-used-weight-1m", "0"))
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            open(BC.SHARED, "w").write(str(int(time.time() * 1000) + 15 * 60_000))
            sys.exit(f"HTTP {e.code} — cooldown written, stopping")
        print(f"  {sym} {iv} HTTP {e.code} — skipped"); return []
    if used > 900:
        print(f"  IP used weight {used} — pausing 60 s", flush=True); time.sleep(60)
    time.sleep(0.3)
    return data


def save(sub, sym, rows):
    d = os.path.join(OUT, sub); os.makedirs(d, exist_ok=True)
    f = os.path.join(d, sym + ".csv")
    df = pd.DataFrame([[int(r[0]), r[1], r[2], r[3], r[4], r[5], r[7]] for r in rows],
                      columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    if os.path.exists(f):
        df = pd.concat([pd.read_csv(f), df])
    df.drop_duplicates("open_time").sort_values("open_time").to_csv(f, index=False)


def main():
    C = DD.cohort()
    cov = DD.coverage(C)
    need = cov[(~cov.tick) & (cov.n1m < 540)]
    print(f"fills needing a 1m top-up: {len(need)} of {len(cov)}", flush=True)
    now = int(time.time() * 1000)
    calls = 0
    for r in need.itertuples():
        a = r.o_ms - 10 * 60_000
        end = min(r.o_ms + 9 * 3600_000, now)
        while a < end:
            b = min(a + 499 * 60_000 - 1, end)
            rows = get(r.pair, "1m", a, b, 499); calls += 1
            if rows:
                save("k1m", r.pair, rows)
            a = b + 1
    # BTC 5m beyond the PREP cache (RH ruler) — from 2026-10-26 00:00 to now
    a = int(pd.Timestamp("2026-10-25 12:00").value // 10**6)
    while a < now:
        b = min(a + 499 * 300_000 - 1, now)
        rows = get("BTCUSDT", "5m", a, b, 499); calls += 1
        if rows:
            save("k5m", "BTCUSDT", rows)
        a = b + 1
    print(f"done: {calls} calls, own weight ≈ {calls * 2}")


if __name__ == "__main__":
    main()
