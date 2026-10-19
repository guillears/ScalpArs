#!/usr/bin/env python3
"""⚡ SURGE missed-opps study — LIVE-block data fetch (public Binance futures REST only; never the bot's API).

5m klines 2026-10-02 → now for the current top-70 COIN perps by 24 h quote volume ∪ every pair the SURGE journals judged, and 1m
klines on demand for walking. Budget: ≤ 1,200 weight/min — reads X-MBX-USED-WEIGHT-1M after every call, sleeps when ≥ 900, stops
(and arms reports/backtest_cache/.binance_cooldown_until) on 418 / 429; honours an active cooldown before any call.
Usage: venv/bin/python scripts/study_surge_missed_live_fetch.py 5m <pairs_file> | 1m <need_csv(pair,t0,t1)>
"""
import json, os, sys, time
import pandas as pd, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import binance_cooldown as BC

OUT = os.path.join(ROOT, "reports", "backtest_cache", "surge_missed", "live")
os.makedirs(os.path.join(OUT, "k5m"), exist_ok=True); os.makedirs(os.path.join(OUT, "k1m"), exist_ok=True)
API = "https://fapi.binance.com"
S = requests.Session()


def get(path, params, weight_hint=5):
    if BC.active():
        sys.exit(f"cooldown active until {BC.read_until_ms()} — stop")
    r = S.get(API + path, params=params, timeout=20)
    used = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
    if r.status_code in (418, 429):
        ra = int(r.headers.get("Retry-After", "120") or 120)
        with open(BC.SHARED, "w") as f:
            f.write(str(int(time.time() * 1000) + ra * 1000))
        sys.exit(f"HTTP {r.status_code} — cooldown armed {ra}s, stop")
    r.raise_for_status()
    if used >= 900:
        time.sleep(61 - (time.time() % 60))
    else:
        time.sleep(0.15)
    return r.json(), used


def klines(sym, iv, t0, t1):
    rows, t = [], t0
    step = 300_000 if iv == "5m" else 60_000
    while t < t1:
        js, used = get("/fapi/v1/klines", dict(symbol=sym, interval=iv, startTime=t, endTime=t1, limit=1000))
        if not js:
            break
        rows += [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[7])] for x in js]
        t = int(js[-1][0]) + step
        if len(js) < 1000:
            break
    return pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "vol", "qvol"]).drop_duplicates("open_time")


def main():
    mode = sys.argv[1]
    if mode == "universe":
        ex, _ = get("/fapi/v1/exchangeInfo", {})
        info = {s["symbol"]: s for s in ex["symbols"]}
        json.dump({k: dict(underlyingType=v.get("underlyingType"), underlyingSubType=v.get("underlyingSubType"), onboardDate=v.get("onboardDate"),
                           status=v.get("status"), contractType=v.get("contractType")) for k, v in info.items()},
                  open(os.path.join(OUT, "exchange_info.json"), "w"))
        tk, _ = get("/fapi/v1/ticker/24hr", {})
        tk = sorted([x for x in tk if x["symbol"].endswith("USDT") and info.get(x["symbol"], {}).get("contractType") == "PERPETUAL"
                     and info[x["symbol"]].get("status") == "TRADING"], key=lambda x: -float(x["quoteVolume"]))
        top = [x["symbol"] for x in tk[:90]]
        extra = [p.strip() for p in open(sys.argv[2]).read().split() if p.strip()]
        pairs = sorted(set(top) | set(extra) | {"BTCUSDT", "ETHUSDT"})
        open(os.path.join(OUT, "pairs.txt"), "w").write("\n".join(pairs))
        print(len(pairs), "pairs")
    elif mode == "5m":
        t0 = int(pd.Timestamp("2026-10-02").value // 10**6); t1 = int(time.time() * 1000)
        pairs = open(os.path.join(OUT, "pairs.txt")).read().split()
        for n, p in enumerate(pairs):
            f = os.path.join(OUT, "k5m", f"{p}.csv")
            if os.path.exists(f):
                continue
            klines(p, "5m", t0, t1).to_csv(f, index=False)
            if n % 10 == 0:
                print(n, p, flush=True)
    elif mode == "1m":
        need = pd.read_csv(sys.argv[2])
        for r in need.itertuples():
            f = os.path.join(OUT, "k1m", f"{r.pair}_{int(r.t0)}.csv")
            if os.path.exists(f):
                continue
            klines(r.pair, "1m", int(r.t0), int(r.t1)).to_csv(f, index=False)
        print("1m done", len(need))


if __name__ == "__main__":
    main()
