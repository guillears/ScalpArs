#!/usr/bin/env python3
"""ROUTE A FOLLOW-UP (PREREG reports/ROUTE_A_FOLLOWUP_2026-11-19_PREREG.md) — Q1(a) 1m kline coverage + fetch.
For every W15_R3_V10 trigger (reports/ROUTE_A_TICKS_data/order.csv) checks that the local 1m caches cover
[sig − 20 min, sig + 25 h] minute by minute; for the pair-days that do not, downloads data.binance.vision USDT-M 1m kline zips (monthly when ≥ 3 days of a month are needed, else daily)
(CHECKSUM verified, 0 Binance API weight, ≤ 8 parallel, back-off) into reports/ROUTE_A_FOLLOWUP_data/k1m/<PAIR>.csv.
Run with -I:  venv/bin/python -I scripts/route_a_followup_fetch1m.py"""
import hashlib, io, os, sys, threading, time, urllib.error, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_data"); K = os.path.join(D, "k1m"); os.makedirs(K, exist_ok=True)
LOCAL = [os.path.join(R, "reports", "backtest_cache_2025", "k1m"), os.path.join(R, "reports", "backtest_cache", "k1m")]
URL = "https://data.binance.vision/data/futures/um/daily/klines/{s}/1m/{s}-1m-{d}.zip"
URLM = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1m/{s}-1m-{d}.zip"   # month with >= 3 needed days
MIN, DAY, H1 = 60_000, 86_400_000, 3_600_000
PRE, POST = 20 * MIN, 25 * H1
UA = {"User-Agent": "research-client/1.0"}
LOCK = threading.Lock(); ST = dict(files=0, bytes=0, miss=0, err=0)


def load_local(pair, extra=True):
    parts = []
    for d in LOCAL + ([K] if extra else []):
        f = os.path.join(d, pair + ".csv")
        if os.path.exists(f) and os.path.getsize(f) > 0:
            try:
                parts.append(pd.read_csv(f, usecols=[0, 1, 2, 3, 4, 5, 6]))
            except Exception:
                pass
    if not parts:
        return None
    for p in parts:
        p.columns = ["open_time", "o", "h", "l", "c", "vol", "qvol"]
    x = pd.concat(parts).drop_duplicates("open_time", keep="last").sort_values("open_time")
    return x[x.open_time % MIN == 0].reset_index(drop=True)


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** k * 2)
        except Exception:
            time.sleep(2 ** k * 2)
    raise RuntimeError("failed " + url)


def fetch_day(pair, ds, monthly=False):
    url = (URLM if monthly else URL).format(s=pair, d=ds)
    b = get(url)
    if b is None:
        return None
    ck = get(url + ".CHECKSUM")
    if ck is not None and ck.split()[0].decode() != hashlib.sha256(b).hexdigest():
        b = get(url)
        if hashlib.sha256(b).hexdigest() != ck.split()[0].decode():
            raise RuntimeError("checksum " + url)
    with LOCK:
        ST["files"] += 1; ST["bytes"] += len(b)
    z = zipfile.ZipFile(io.BytesIO(b)); raw = z.read(z.namelist()[0])
    hdr = 0 if raw[:1].isalpha() else None
    x = pd.read_csv(io.BytesIO(raw), header=hdr, usecols=[0, 1, 2, 3, 4, 5, 7])
    x.columns = ["open_time", "o", "h", "l", "c", "vol", "qvol"]
    return x


def needed(pair, sigs):
    """pair-days (UTC day index) whose minutes are missing for any trigger window"""
    x = load_local(pair)
    have = set() if x is None else set(x.open_time.values.astype("int64").tolist())
    days = set()
    last_ok = (pd.Timestamp("2026-11-18").value // 10**6)
    for s in sigs:
        a, b = s - PRE, min(s + POST, last_ok)
        mins = range(a, b, MIN)
        miss = [m for m in mins if m not in have]
        if miss:
            days |= {m // DAY for m in miss}
    return sorted(d for d in days if d * DAY < last_ok)


def do_pair(args):
    pair, sigs = args
    days = needed(pair, sigs)
    if not days:
        return pair, 0, 0
    got = []
    miss = 0
    mon = pd.Series([pd.Timestamp(d * DAY, unit="ms").strftime("%Y-%m") for d in days])
    big = set(mon.value_counts()[lambda v: v >= 3].index) - {"2026-11"}
    jobs = [(m, True) for m in sorted(big)] + [(pd.Timestamp(d * DAY, unit="ms").strftime("%Y-%m-%d"), False)
                                               for d, m in zip(days, mon) if m not in big]
    for ds, monthly in jobs:
        try:
            x = fetch_day(pair, ds, monthly)
            if x is None and monthly:     # no monthly file: fall back to the days
                xs = [fetch_day(pair, pd.Timestamp(d * DAY, unit="ms").strftime("%Y-%m-%d"))
                      for d, m in zip(days, mon) if m == ds]
                xs = [v for v in xs if v is not None]
                x = pd.concat(xs) if xs else None
        except Exception as e:
            print("ERR", pair, ds, e, flush=True); x = None
            with LOCK:
                ST["err"] += 1
        if x is None:
            miss += 1
            with LOCK:
                ST["miss"] += 1
        else:
            got.append(x)
    if got:
        f = os.path.join(K, pair + ".csv")
        x = pd.concat(got + ([pd.read_csv(f)] if os.path.exists(f) else []))
        x.drop_duplicates("open_time").sort_values("open_time").to_csv(f, index=False)
    return pair, len(days), miss


def main():
    O = pd.read_csv(os.path.join(R, "reports", "ROUTE_A_TICKS_data", "order.csv"))
    tasks = [(p, g.sig_ms.values.astype("int64").tolist()) for p, g in O.groupby("pair")]
    t0 = time.time(); log = []
    with ThreadPoolExecutor(8) as P:
        for k, (pair, nd, miss) in enumerate(P.map(do_pair, tasks)):
            log.append(dict(pair=pair, days_needed=nd, days_missing=miss))
            if k % 25 == 0:
                print(time.strftime("%H:%M:%S"), k, "/", len(tasks), f"{ST['bytes']/1e9:.2f} GB {ST['files']} files miss {ST['miss']} err {ST['err']}",
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
    pd.DataFrame(log).to_csv(os.path.join(D, "fetch1m_log.csv"), index=False)
    print("DONE", ST, f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
