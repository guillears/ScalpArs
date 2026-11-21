#!/usr/bin/env python3
"""EXH_FUNDING_STUDY_2026-11-21 — fetch the REAL-fill gaps only (research only; no bot code / config touched).

  * /fapi/v1/fundingRate 2026-11-17 → now for every real EXHAUSTION_SHORT pair of the current batch + the open ENSO fill
    (real settlement rows: an hourly coin gets hourly rows).
  * /fapi/v1/fundingInfo once (symbols with an adjusted interval / cap right now).
  * /fapi/v1/klines 1m from each fill's minute to min(fill + 26 h, now) (complete bars only).
Budget (CLAUDE.md / feedback_binance_request_budget): IP-wide X-MBX-USED-WEIGHT-1M read after every call, stop at >= 900, stop on
HTTP 418 / 429, never call during a scripts/binance_cooldown.py cooldown. ~ 30 calls, ~ 150 weight in total.
Output: reports/.funding_cache/{funding_<PAIR>.csv, fundingInfo.json, k1m_<PAIR>_<fill ms>.csv}
Usage: venv/bin/python -I scripts/EXH_FUNDING_STUDY_2026-11-21_fetch.py"""
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

OUT = os.path.join(ROOT, "reports", ".funding_cache")
EXPORT = next((p for p in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports_archive", "scalpars_orders_paper_2026-11-21_22-18-17.csv"), os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-21_22-18-17.csv")) if os.path.exists(p)), os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-21_22-18-17.csv"))
# the open ENSO fill is not in the export (open rows are not exported): from the decisions journal 2026-11-21_13-21-05 (OPEN row)
ENSO = dict(pair="ENSOUSDT", opened_at="2026-11-21T03:30:33")
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
            if w is None and "/klines" in url:
                sys.exit("no used-weight header on klines — stop")
            if w is not None:          # fundingRate / fundingInfo: own 500 / 5 min IP limit, no weight header; ~ 10 calls here
                _w["w"] = int(w)
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            sys.exit(f"HTTP {e.code} — stop, not retrying")
        raise
    print(f"  used-weight-1m {_w['w']}  {url.split('?')[0].rsplit('/', 1)[-1]}", flush=True)
    time.sleep(0.3)
    return data


def ms(s):
    return int(pd.Timestamp(s).value // 10**6)


def main():
    os.makedirs(OUT, exist_ok=True)
    now = int(time.time() * 1000)
    d = pd.read_csv(EXPORT, low_memory=False)
    e = d[d.entry_strategy == "EXHAUSTION_SHORT"][["pair", "opened_at"]].to_dict("records") + [ENSO]
    pairs = sorted({r["pair"] for r in e})
    f0 = ms("2026-11-17")
    for p in pairs:
        x = get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={p}&startTime={f0}&endTime={now}&limit=1000")
        pd.DataFrame([dict(t=int(j["fundingTime"]), rate=float(j["fundingRate"]),
                           mark=float(j["markPrice"]) if j.get("markPrice") not in (None, "") else float("nan")) for j in x]
                     ).to_csv(os.path.join(OUT, f"funding_{p}.csv"), index=False)
        print(p, "funding rows", len(x))
    fi = get("https://fapi.binance.com/fapi/v1/fundingInfo")
    json.dump([r for r in fi if r.get("symbol") in pairs] + [{"_all_symbols_listed": len(fi)}],
              open(os.path.join(OUT, "fundingInfo.json"), "w"), indent=1)
    for r in e:
        p = r["pair"]; a = ms(r["opened_at"]) // 60_000 * 60_000
        fo = os.path.join(OUT, f"k1m_{p}_{a}.csv")
        end = min(a + 26 * 3_600_000, now)
        if os.path.exists(fo) and pd.read_csv(fo).t.max() + 120_000 >= end:
            continue
        rows, s = [], a
        while s < end:
            k = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={p}&interval=1m&startTime={s}&endTime={end}&limit=1000")
            if not k:
                break
            rows += [[int(z[0]), float(z[1]), float(z[2]), float(z[3]), float(z[4])] for z in k]
            s = int(k[-1][0]) + 60_000
            if len(k) < 1000:
                break
        k = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).drop_duplicates("t")
        k = k[k.t + 60_000 <= now]
        k.to_csv(fo, index=False)
        print(p, r["opened_at"], "1m", len(k))


if __name__ == "__main__":
    main()
