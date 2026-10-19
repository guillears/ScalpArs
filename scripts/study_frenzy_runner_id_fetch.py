#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 — public USDT-M klines for the REAL fills the caches do not cover (k5m caches end 2026-10-12).

Public endpoint only (fapi /fapi/v1/klines; never the bot's auth API). Budget rules (memory: binance request budget): <= 1,200 weight/min,
reads X-MBX-USED-WEIGHT-1M after every call and sleeps when > 900, stops on 418/429 (writes the shared cooldown file), honours
scripts/binance_cooldown.py. Per pair: 5m from (first fill - 12 d) to (last fill + 13 h); 1m from (first fill - 5 min) to (last fill + 13 h).
Usage: venv/bin/python scripts/study_frenzy_runner_id_fetch.py <outdir> <pairs_windows.csv: pair,t0_ms,t1_ms>
"""
import os, sys, time, json, urllib.request, urllib.error
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binance_cooldown as BC  # noqa: E402

URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit=1500"
STEP = {"5m": 300_000, "1m": 60_000}


def get(url):
    if BC.active():
        raise SystemExit("cooldown active — stop")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            ra = int(e.headers.get("Retry-After", "120") or 120)
            with open(BC.SHARED, "w") as f:
                f.write(str(int((time.time() + ra) * 1000)))
            raise SystemExit(f"HTTP {e.code} — cooldown {ra}s written, stop")
        raise
    if w > 900:
        time.sleep(61)
    return data


def fetch(sym, iv, a, b):
    out = []
    while a < b:
        d = get(URL.format(s=sym, i=iv, a=a, b=b))
        if not d:
            break
        out += d
        a = int(d[-1][0]) + STEP[iv]
        time.sleep(0.15)
    df = pd.DataFrame([r[:6] + [r[7]] for r in out], columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
    return df.astype(float).astype({"open_time": "int64"}).drop_duplicates("open_time")


if __name__ == "__main__":
    od = sys.argv[1]; os.makedirs(od, exist_ok=True)
    W = pd.read_csv(sys.argv[2])
    for _, r in W.iterrows():
        for iv, lead in (("5m", 12 * 86_400_000), ("1m", 5 * 60_000)):
            fn = f"{od}/{r.pair}_{iv}.csv"
            if os.path.exists(fn):
                continue
            df = fetch(r.pair, iv, int(r.t0_ms) - lead, int(r.t1_ms) + 13 * 3_600_000)
            df.to_csv(fn, index=False)
            print(r.pair, iv, len(df), flush=True)
