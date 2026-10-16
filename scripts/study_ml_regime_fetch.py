#!/usr/bin/env python3
"""ML REGIME study 2026-10-16 — 5m top-up 2026-10-14 00:00 -> 2026-10-16 00:00 UTC for BTC + the alt universe (top-60 by 30-day
avg quote volume on the latest cached k1d). Public fapi, read-only. Budget: reads X-MBX-USED-WEIGHT-1M, pauses above 700, stops on
418/429 and writes / honours reports/backtest_cache/.binance_cooldown_until. Output reports/ml_regime/fetch/k5m/<PAIR>.csv. Resumable.
Pre-registration: reports/ML_REGIME_PREREG.txt (section 0).
    venv/bin/python scripts/study_ml_regime_fetch.py
"""
import os, sys, time
import pandas as pd, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_lite_regime_features import load_daily, STABLE   # noqa: E402

CD = os.path.join(ROOT, "reports", "backtest_cache", ".binance_cooldown_until")
OUT = os.path.join(ROOT, "reports", "ml_regime", "fetch", "k5m")
URL = "https://fapi.binance.com/fapi/v1/klines"
STATE = {"w": 0, "calls": 0}


def cooldown_check():
    if os.path.exists(CD):
        try:
            if float(open(CD).read().strip() or 0) > time.time():
                raise SystemExit("cooldown active — stop")
        except ValueError:
            pass


def get(params):
    cooldown_check()
    if STATE["w"] > 700:
        print(f"  used weight {STATE['w']} > 700 -> sleep 61 s", flush=True)
        time.sleep(61); STATE["w"] = 0
    r = requests.get(URL, params=params, timeout=20)
    STATE["w"] = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
    STATE["calls"] += 1
    if r.status_code in (418, 429):
        ra = int(r.headers.get("Retry-After", "120") or 120)
        open(CD, "w").write(str(time.time() + max(ra, 120)))
        raise SystemExit(f"rate limited {r.status_code} — cooldown written, stop")
    if r.status_code == 400:
        return []
    r.raise_for_status()
    time.sleep(0.3)
    return r.json()


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load_daily()
    QV = D.pivot(index="t", columns="pair", values="qv").sort_index()
    qv = QV.iloc[-30:].mean().dropna().drop(labels=[p for p in ("BTCUSDT",) + STABLE if p in QV.columns], errors="ignore")
    pairs = ["BTCUSDT"] + list(qv.sort_values(ascending=False).index[:60])
    s = int(pd.Timestamp("2026-10-14", tz="UTC").timestamp() * 1000)
    e = int(pd.Timestamp("2026-10-16", tz="UTC").timestamp() * 1000)
    for i, p in enumerate(pairs):
        f = os.path.join(OUT, p + ".csv")
        if os.path.exists(f):
            continue
        k = get(dict(symbol=p, interval="5m", startTime=s, endTime=e - 1, limit=1000))
        df = pd.DataFrame([r[:8] for r in k], columns=["open_time", "o", "h", "l", "c", "vol", "close_time", "qvol"])
        df[["open_time", "o", "h", "l", "c", "vol", "qvol"]].to_csv(f, index=False)
        if i % 10 == 0:
            print(f"{i + 1}/{len(pairs)} {p} rows {len(df)} calls {STATE['calls']} used-weight {STATE['w']}", flush=True)
    print("done", STATE)


if __name__ == "__main__":
    main()
