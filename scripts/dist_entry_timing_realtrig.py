#!/usr/bin/env python3
"""DIST_ENTRY_TIMING — real-fill cross-check: would the 4 paper DIST signals (11-19 / 11-20) have triggered at T−1 / T−2? fapi reads only:
1m klines (weight 1 ×1 per signal), 15m klines (weight 1), topLongShortPositionRatio 5m (data endpoint). Used-weight header read; stops on
418 / 429. → reports/DIST_ENTRY_TIMING_data/realtrig.csv.  Usage: venv/bin/python -I scripts/dist_entry_timing_realtrig.py"""
import json, os, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, R)
from services.distribution import pos_chg_4h   # noqa: E402  (the engine's own pure rule)
UA = {"User-Agent": "Mozilla/5.0 (research client)"}
F = "https://fapi.binance.com"
M1, M15 = 60_000, 900_000


def get(path):
    try:
        with urllib.request.urlopen(urllib.request.Request(F + path, headers=UA), timeout=30) as r:
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w is not None and int(w) > 1000:
                raise SystemExit(f"used weight {w} — stopping")
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise SystemExit(f"HTTP {e.code} — stopping")
        raise


SIG = [("SENTUSDT", "2026-11-19 04:15"), ("NAORISUSDT", "2026-11-19 09:15"), ("EVAAUSDT", "2026-11-19 14:30"), ("KERNELUSDT", "2026-11-20 12:30")]
out = []
for p, ts in SIG:
    T = pd.Timestamp(ts).value // 10**6
    k15 = get(f"/fapi/v1/klines?symbol={p}&interval=15m&endTime={T - 1}&limit=21")
    k15 = [r for r in k15 if int(r[0]) + M15 <= T]
    prev = [float(r[7]) for r in k15[-21:-1]]; vavg = np.mean(prev)
    k1 = get(f"/fapi/v1/klines?symbol={p}&interval=1m&startTime={T - M15}&limit=15")
    O = float(k1[0][1]); d = dict(pair=p, T=ts, vmult_T0=float(k15[-1][7]) / vavg, red_T0=float(k15[-1][4]) < float(k15[-1][1]))
    pr = get(f"/futures/data/topLongShortPositionRatio?symbol={p}&period=5m&limit=60&endTime={T}")
    for k in (1, 2):
        m = 15 - k
        q = sum(float(r[7]) for r in k1[:m]); c = float(k1[m - 1][4])
        chg = pos_chg_4h(pr, T - k * M1)[0]
        d[f"red_T-{k}"] = c < O; d[f"vmult_T-{k}"] = q / vavg; d[f"pos_chg_T-{k}"] = chg
    d["pos_chg_T0"] = pos_chg_4h(pr, T)[0]
    out.append(d); time.sleep(0.3)
X = pd.DataFrame(out); X.to_csv(os.path.join(R, "reports", "DIST_ENTRY_TIMING_data", "realtrig.csv"), index=False)
print(X.round(4).to_string())
