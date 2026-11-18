#!/usr/bin/env python3
"""NOSPOT_SLEEVE — fresh public fapi data for the operator-coin replays (2026-11-08 → now): 5m klines, OI history + top-trader position /
account + global account ratios + taker buy/sell ratio (5m, /futures/data), 5m premiumIndexKlines, fundingRate. Generic UA, no account,
no key. ≥ 0.35 s between requests, reads X-MBX-USED-WEIGHT-1M, stops above 900 or on 418 / 429.
Output reports/NOSPOT_SLEEVE_data/live/{k5m,metrics,premium,funding}/  ·  Usage: venv/bin/python -I scripts/nospot_sleeve_live_fetch.py PAIR …"""
import json, os, sys, time, urllib.request, urllib.error
import pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OD = os.path.join(R, "reports", "NOSPOT_SLEEVE_data", "live")
F = "https://fapi.binance.com"
UA = {"User-Agent": "Mozilla/5.0"}
T0 = int(pd.Timestamp("2026-11-08").value // 10**6)


def get(path):
    time.sleep(0.35)
    try:
        with urllib.request.urlopen(urllib.request.Request(F + path, headers=UA), timeout=30) as r:
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w is not None and int(w) > 900:
                raise SystemExit(f"used weight {w} — stopping")
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429, 403):
            raise SystemExit(f"HTTP {e.code} — stopping")
        raise


def paged(path, key_t, step_ms, limit):
    out = []; s = T0; now = int(time.time() * 1000)
    while s < now:
        x = get(f"{path}&startTime={s}&endTime={min(s + step_ms * limit, now)}&limit={limit}")
        if not x:
            s += step_ms * limit; continue
        out += x
        s = int(x[-1][key_t] if isinstance(x[-1], dict) else x[-1][0]) + step_ms
        if len(x) < limit and s >= now - step_ms:
            break
    return out


def main(pairs):
    for d in ("k5m", "metrics", "premium", "funding"):
        os.makedirs(os.path.join(OD, d), exist_ok=True)
    for p in pairs:
        k = paged(f"/fapi/v1/klines?symbol={p}&interval=5m", 0, 300_000, 1000)
        kd = pd.DataFrame([[int(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]), float(a[5]), float(a[7])] for a in k],
                          columns=["open_time", "o", "h", "l", "c", "vol", "qvol"]).drop_duplicates("open_time")
        now = int(time.time() * 1000); kd = kd[kd.open_time + 300_000 <= now]
        kd.to_csv(os.path.join(OD, "k5m", p + ".csv"), index=False)
        pr = paged(f"/fapi/v1/premiumIndexKlines?symbol={p}&interval=5m", 0, 300_000, 1000)
        pd.DataFrame([[int(a[0]), float(a[4])] for a in pr], columns=["open_time", "c"]).drop_duplicates("open_time").to_csv(
            os.path.join(OD, "premium", f"{p}.2026-11.csv"), index=False)
        fr = get(f"/fapi/v1/fundingRate?symbol={p}&startTime={T0 - 7 * 86_400_000}&limit=1000")
        pd.DataFrame([[int(a["fundingTime"]), float(a["fundingRate"])] for a in fr], columns=["t", "rate"]).to_csv(
            os.path.join(OD, "funding", f"{p}.live.csv"), index=False)
        cols = {}
        for nm, path, fld in (("oi", f"/futures/data/openInterestHist?symbol={p}&period=5m", None),
                              ("tp", f"/futures/data/topLongShortPositionRatio?symbol={p}&period=5m", "longShortRatio"),
                              ("ta", f"/futures/data/topLongShortAccountRatio?symbol={p}&period=5m", "longShortRatio"),
                              ("ga", f"/futures/data/globalLongShortAccountRatio?symbol={p}&period=5m", "longShortRatio"),
                              ("tk", f"/futures/data/takerlongshortRatio?symbol={p}&period=5m", "buySellRatio")):
            x = paged(path, "timestamp", 300_000, 500)
            df = pd.DataFrame(x)
            if not len(df):
                continue
            df["timestamp"] = df.timestamp.astype("int64")
            df = df.drop_duplicates("timestamp").set_index("timestamp")
            if nm == "oi":
                cols["sum_open_interest"] = df.sumOpenInterest.astype(float); cols["sum_open_interest_value"] = df.sumOpenInterestValue.astype(float)
            else:
                cols[{"tp": "sum_toptrader_long_short_ratio", "ta": "count_toptrader_long_short_ratio", "ga": "count_long_short_ratio",
                      "tk": "sum_taker_long_short_vol_ratio"}[nm]] = df[fld].astype(float)
        M = pd.DataFrame(cols).sort_index()
        M.insert(0, "create_time", M.index); M.insert(1, "symbol", p)
        M["date"] = pd.to_datetime(M.create_time, unit="ms").dt.strftime("%Y-%m-%d")
        for d_, g in M.groupby("date"):
            g.drop(columns="date").to_csv(os.path.join(OD, "metrics", f"{p}.{d_}.csv"), index=False)
        print(p, len(kd), len(pr), len(fr), len(M), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
