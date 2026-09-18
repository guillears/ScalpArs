#!/usr/bin/env python3
"""BTC % below its 30-day HIGH, hourly — the washed-out reading of the 🔥 long heat block (Sep-18).

Fills pool rows that predate the stamped column `entry_btc_off30d_high_pct`, so
scripts/build_master_pool.py and scripts/screen_pool.py can apply the block with live parity.
Same formula as the engine monitor: last CLOSED 1h close / max(high of the prior 720 closed 1h bars) − 1.
Public Binance klines, no key. Re-run to extend; output is small and tracked.

  venv/bin/python scripts/build_btc_off30d.py          # → reports/btc_off30d_hourly.csv
"""
import json, time, urllib.request
import pandas as pd

OUT = "reports/btc_off30d_hourly.csv"
START = "2026-04-01"

def main():
    rows, cur, end = [], int(pd.Timestamp(START).timestamp() * 1000), int(time.time() * 1000)
    while cur < end:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&startTime={cur}&limit=1500"
        k = json.loads(urllib.request.urlopen(url, timeout=30).read())
        if not k:
            break
        rows += k
        cur = k[-1][0] + 3_600_000
        time.sleep(0.15)
    b = pd.DataFrame(rows).iloc[:, :5].astype(float)
    b.columns = ["t", "o", "h", "l", "c"]
    b = b.drop_duplicates("t").sort_values("t").iloc[:-1]            # drop the still-open bar
    b.index = pd.to_datetime(b.t, unit="ms")
    hi = b.h.rolling(720, min_periods=360).max()
    off = ((b.c / hi - 1) * 100).shift(1)                            # known at the START of the hour (no lookahead)
    out = off.dropna().round(2).rename("btc_off30d_high_pct").rename_axis("hour_utc").reset_index()
    out["hour_utc"] = out.hour_utc.dt.strftime("%Y-%m-%d %H:00")
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} hours → {OUT} ({out.hour_utc.iloc[0]} … {out.hour_utc.iloc[-1]})")

if __name__ == "__main__":
    main()
