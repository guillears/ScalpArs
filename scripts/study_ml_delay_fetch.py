#!/usr/bin/env python3
"""Minimal Binance top-up for the master fills the cache cannot price (late B19-B21, after the 1m cache ends 2026-10-12 04:09).
Reads a rows CSV from study_ml_delay_price.py, picks signals with no price / no signal recompute, and fetches ONLY:
pair 1m [t0 - 9 h, t0 + 8 h], pair 5m [t0 - 10 h, t0 + 1 h], and BTCUSDT 1m + 5m over the span of those fills (± 9 h).
Writes to reports/ml_delay_fetch/{k1m,k5m_full}/ (never into reports/backtest_cache). Budget rule: <= 1,200 weight/min,
reads x-mbx-used-weight-1m, stops on 418/429, honours and writes the shared cooldown reports/backtest_cache/.binance_cooldown_until.

    venv/bin/python -W ignore scripts/study_ml_delay_fetch.py <rows.csv>
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
OUT = os.path.join(ROOT, "reports/ml_delay_fetch")
COOL = os.path.join(ROOT, "reports/backtest_cache/.binance_cooldown_until")
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit=499"
H = 3600_000
NREQ, LASTW = 0, 0


def cooldown_ok():
    try:
        return not _BC.active(COOL)
    except Exception:
        return True


def get(sym, iv, a, b):
    global NREQ, LASTW
    if not cooldown_ok():
        sys.exit("shared Binance cooldown active — stopping")
    req = urllib.request.Request(URL.format(s=sym, i=iv, a=a, b=b), headers={"User-Agent": "scalpars-study"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            w = int(r.headers.get("x-mbx-used-weight-1m", "0"))
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            open(COOL, "w").write(str(time.time() + 600))
            sys.exit(f"HTTP {e.code} — cooldown written, stopping")
        raise
    NREQ += 1; LASTW = w
    if w > 900:
        print(f"  used weight {w} — pausing 60 s"); time.sleep(60)
    time.sleep(0.2)
    return data


def series(sym, iv, a, b):
    step = 60_000 if iv == "1m" else 300_000
    rows, cur = [], a
    while cur < b:
        d = get(sym, iv, cur, b)
        if not d:
            break
        rows += d
        nxt = int(d[-1][0]) + step
        if nxt <= cur or len(d) < 499:
            break
        cur = nxt
    return rows


def save(sub, sym, rows):
    d = os.path.join(OUT, sub); os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, sym + ".csv")
    df = pd.DataFrame([[int(r[0]), r[1], r[2], r[3], r[4], r[5], r[7]] for r in rows],
                      columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    if os.path.exists(fp):
        df = pd.concat([pd.read_csv(fp), df]).drop_duplicates("open_time")
    df.sort_values("open_time").to_csv(fp, index=False)


def main():
    R = pd.read_csv(sys.argv[1])
    g = R[(R.block == "master") & (R.rep0.isna() | R.valid0.isna() | R.btc15.isna() | R.rep15.isna())]
    if not len(g):
        print("nothing to fetch"); return
    print(len(g), "signals:", g.key.tolist())
    for r in g.itertuples():
        t = int(r.o_ms)
        save("k1m", r.pair, series(r.pair, "1m", t - 9 * H, t + 8 * H))
        save("k5m_full", r.pair, series(r.pair, "5m", t - 10 * H, t + 1 * H))
    a, b = int(g.o_ms.min()) - 9 * H, int(g.o_ms.max()) + 9 * H
    save("k1m", "BTCUSDT", series("BTCUSDT", "1m", a, b))
    save("k5m_full", "BTCUSDT", series("BTCUSDT", "5m", a - 10 * H, b))
    print(f"requests {NREQ} · last used weight {LASTW}")


if __name__ == "__main__":
    main()
