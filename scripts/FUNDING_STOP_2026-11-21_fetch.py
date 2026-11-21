#!/usr/bin/env python3
"""FUNDING_STOP_2026-11-21 — fetch only the gaps for the real-fill re-walk (research only; no bot code / config touched).

  * KERNELUSDT 1m klines 2026-11-20 12:00 → now (FLATBASE #25 opened 11-20 20:00:35, stopped 11-21 01:07:22; the DIST / EXH KERNEL fills
    of 11-20 12:30 / 14:15 are covered by the same file).
  * /fapi/v1/fundingRate history 2026-11-16 → now for every real EXHAUSTION / DISTRIBUTION / FLATBASE-#25 pair (the DIST TP_GRID funding
    cache came back empty, so the DIST real fills were priced with 0 funding there).
Budget (CLAUDE.md / feedback_binance_request_budget): IP-wide X-MBX-USED-WEIGHT-1M read after every call, stop at ≥ 900, stop on HTTP 418 / 429,
never call during a scripts/binance_cooldown.py cooldown. ≈ 12 calls, ≈ 20 weight in total.
Output: reports/FUNDING_STOP_2026-11-21_data/{k1m_KERNELUSDT.csv, funding_<PAIR>.csv}
Usage: venv/bin/python -I scripts/FUNDING_STOP_2026-11-21_fetch.py"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import binance_cooldown as BC  # noqa: E402

OUT = os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data")
PAIRS = ["KERNELUSDT", "SENTUSDT", "NAORISUSDT", "EVAAUSDT", "CYBERUSDT", "ZKPUSDT", "ZKCUSDT", "1000LUNCUSDT", "YBUSDT"]
STOP_W = 900
_w = {"w": 0}


def get(url):
    if BC.active():
        sys.exit("binance cooldown active — stop")
    if _w["w"] >= STOP_W:
        sys.exit(f"weight budget reached ({_w['w']}) — stop")
    req = urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w is not None:
                _w["w"] = int(w)
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            sys.exit(f"HTTP {e.code} — stop, not retrying")
        raise
    print(f"  used-weight-1m {_w['w']}  {url.split('?')[0].rsplit('/', 1)[-1]}", flush=True)
    time.sleep(0.3)
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    now = int(time.time() * 1000)
    a = int(pd.Timestamp("2026-11-20 12:00").value // 10**6)
    rows, s = [], a
    while s < now:
        d = get(f"https://fapi.binance.com/fapi/v1/klines?symbol=KERNELUSDT&interval=1m&startTime={s}&endTime={now}&limit=1000")
        if not d:
            break
        rows += [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in d]
        s = int(d[-1][0]) + 60_000
        if len(d) < 1000:
            break
    k = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).drop_duplicates("t")
    k = k[k.t + 60_000 <= now]                      # complete bars only
    k.to_csv(os.path.join(OUT, "k1m_KERNELUSDT.csv"), index=False)
    print("KERNEL 1m", len(k), pd.to_datetime(k.t.min(), unit="ms"), "→", pd.to_datetime(k.t.max(), unit="ms"))
    f0 = int(pd.Timestamp("2026-11-16").value // 10**6)
    for p in PAIRS:
        d = get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={p}&startTime={f0}&endTime={now}&limit=1000")
        x = pd.DataFrame([dict(t=int(j["fundingTime"]), rate=float(j["fundingRate"]),
                               mark=float(j["markPrice"]) if j.get("markPrice") not in (None, "") else float("nan")) for j in d])
        x.to_csv(os.path.join(OUT, f"funding_{p}.csv"), index=False)
        print(p, "funding rows", len(x))


if __name__ == "__main__":
    main()
