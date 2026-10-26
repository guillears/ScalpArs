#!/usr/bin/env python3
"""ML MATCHED-LONG SHADOW study (2026-10-26) — step 1c: the LIVE CloudWatch refusals (Oct-12 → Oct-26) + their price paths.

① parse the read-only CloudWatch pull (aws logs filter-log-events --filter-pattern '"LONG_UNMATCHED_ONLY"' on
   /aws/elasticbeanstalk/SCALPARS-env-2/var/log/web.stdout.log, 14-day retention) → one row per refusal: t (UTC, ms), pair,
   c_any / w_any (the only pattern info the log line carries) and the signal confidence from the paired [OPEN_FILTERED] line.
   → reports/ML_MATCHED_SHADOW_2026-10-26_cloudwatch_refusals.csv
② fetch ONLY public fapi klines the caches lack: per refusal pair 5m × 110 bars before t (ATR / pair state) and 1m [t − 10 min,
   t + 9 h]; BTCUSDT 1m + 5m from 2026-10-11 to now (BTC state + recovery-hold ruler). Writes to reports/ML_MATCHED_SHADOW_2026-10-26_fetch/
   (never into backtest_cache). Budget ≤ 400 weight/min of OUR calls (another study shares the IP), reads x-mbx-used-weight-1m
   (pause 60 s when the IP total > 800), stops on 418/429 and writes the shared cooldown file, honours scripts/binance_cooldown.py.

    venv/bin/python -W ignore scripts/study_ml_matched_fetch.py <cloudwatch_json>
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binance_cooldown as BC  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PFX = os.path.join(ROOT, "reports", "ML_MATCHED_SHADOW_2026-10-26")
OUT = PFX + "_fetch"
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&endTime={b}&limit={n}"
MY = []
BUDGET = 400
RX_L = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - services\.trading_engine - INFO - \[LONG_UNMATCHED_ONLY\] (\w+): "
                  r"LONG blocked — matched a pattern \(c_any=(\w+), w_any=(\w+)\)")
RX_F = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - services\.trading_engine - INFO - \[OPEN_FILTERED\] (\w+) LONG (\w+): "
                  r"declined by a counted entry filter \(last block: LONG_UNMATCHED_ONLY\)")


def parse(path):
    ev = json.load(open(path))
    L, F = [], {}
    for _, m in ev:
        a = RX_L.search(m)
        if a:
            L.append(dict(t=pd.Timestamp(a.group(1) + "." + a.group(2)), pair=a.group(3), c_any=a.group(4) == "True",
                          w_any=a.group(5) == "True", host_line=m.split(" web[")[1].split("]")[0] if " web[" in m else None))
            continue
        b = RX_F.search(m)
        if b:
            F[(b.group(3), b.group(1))] = b.group(4)
    df = pd.DataFrame(L).drop_duplicates(["t", "pair"]).sort_values("t").reset_index(drop=True)
    df["conf"] = [F.get((r.pair, r.t.strftime("%Y-%m-%d %H:%M:%S"))) for r in df.itertuples()]
    df["o_ms"] = (df.t.values.astype("datetime64[ms]").astype("int64"))
    df["family"] = df.c_any.map({True: "C", False: ""}) + df.w_any.map({True: "W", False: ""})
    df["family"] = df.family.replace({"CW": "C+W"})
    return df


def _budget(w):
    now = time.time()
    while True:
        MY[:] = [(t, x) for t, x in MY if now - t < 60]
        if sum(x for _, x in MY) + w <= BUDGET:
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
    if used > 800:
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


def have(sub, sym, a, b, step):
    f = os.path.join(OUT, sub, sym + ".csv")
    if not os.path.exists(f):
        return False
    t = pd.read_csv(f, usecols=["open_time"]).open_time
    return ((t >= a) & (t <= b)).sum() >= (b - a) // step - 2


def span(sym, iv, step, a, end):
    calls = 0
    while a < end:
        b = min(a + 499 * step - 1, end)
        if not have("k1m" if iv == "1m" else "k5m", sym, a, b, step):
            rows = get(sym, iv, a, b, 499); calls += 1
            if rows:
                save("k1m" if iv == "1m" else "k5m", sym, rows)
        a = b + 1
    return calls


def main():
    df = parse(sys.argv[1])
    df.drop(columns=["host_line"]).to_csv(PFX + "_cloudwatch_refusals.csv", index=False)
    print(f"{len(df)} refusals {df.t.min()} → {df.t.max()}  families {df.family.value_counts().to_dict()}", flush=True)
    now = int(time.time() * 1000)
    calls = 0
    t0 = int(pd.Timestamp("2026-10-11").value // 10**6)
    calls += span("BTCUSDT", "5m", 300_000, t0, now)
    calls += span("BTCUSDT", "1m", 60_000, t0, now)
    for r in df.itertuples():
        calls += span(r.pair, "5m", 300_000, r.o_ms - 110 * 300_000, r.o_ms)
        calls += span(r.pair, "1m", 60_000, r.o_ms - 10 * 60_000, min(r.o_ms + 9 * 3600_000, now))
    print(f"done: {calls} calls")


if __name__ == "__main__":
    main()
