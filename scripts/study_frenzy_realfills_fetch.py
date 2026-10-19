#!/usr/bin/env python3
"""FRENZY REAL-FILLS exit / runner / volume-boundary study 2026-10-19 — public market data fetcher (research only).

Sources, in order of preference (no API weight first):
  · data.binance.vision daily zips (USD-M futures): 5m klines, 1m klines, aggTrades — CDN, no request weight.
  · fapi.binance.com public /fapi/v1/klines for the current, still-open UTC day only. Budget (memory: binance request budget):
    ≤ 1,200 weight/min — reads X-MBX-USED-WEIGHT-1M after every call, sleeps when > 900, stops on 418/429 and writes the shared cooldown
    file, honours scripts/binance_cooldown.py. Never the bot's auth API.
Downloads land in their own directory under the scratch dir (untrusted data; parsed as plain CSV only).
Usage:
  python -I scripts/study_frenzy_realfills_fetch.py k5   <outdir> <day0> <day1>  [pairs.txt]   daily 5m zips, every pair of k5m_full
  python -I scripts/study_frenzy_realfills_fetch.py k5api <outdir> <t0_ms> [pairs.txt]           5m from t0 to now via fapi (today)
  python -I scripts/study_frenzy_realfills_fetch.py k1   <outdir> <pairs_days.csv>               daily 1m zips (pair,day)
  python -I scripts/study_frenzy_realfills_fetch.py k1api <outdir> <t0_ms> <pairs.txt>            1m from t0 to now via fapi
  python -I scripts/study_frenzy_realfills_fetch.py ticks <outdir> <pairs_days.csv>              aggTrades zips → npz (t ms, p)
"""
import io, os, sys, time, json, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import binance_cooldown as BC  # noqa: E402

V = "https://data.binance.vision/data/futures/um/daily"
API = "https://fapi.binance.com/fapi/v1/klines?symbol={s}&interval={i}&startTime={a}&limit={n}"


def _get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * k)
        except Exception:  # noqa: BLE001
            time.sleep(2 + 3 * k)
    return None


def _kl(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        b = z.read(z.namelist()[0])
    d = pd.read_csv(io.BytesIO(b), header=None)
    if not str(d.iloc[0, 0]).isdigit():
        d = d.iloc[1:]
    d = d.iloc[:, [0, 1, 2, 3, 4, 5, 7]].astype(float)
    d.columns = ["open_time", "o", "h", "l", "c", "vol", "qvol"]
    return d.astype({"open_time": "int64"})


def k5(out, d0, d1, pairs):
    days = [f"{x:%Y-%m-%d}" for x in pd.date_range(d0, d1)]
    jobs = [(p, d) for p in pairs for d in days if not os.path.exists(f"{out}/{p}/{d}.csv")]

    def one(j):
        p, d = j
        raw = _get(f"{V}/klines/{p}/5m/{p}-5m-{d}.zip")
        if raw is None:
            return p, d, 0
        df = _kl(raw); os.makedirs(f"{out}/{p}", exist_ok=True); df.to_csv(f"{out}/{p}/{d}.csv", index=False)
        return p, d, len(df)
    with ThreadPoolExecutor(12) as ex:
        n = 0
        for p, d, k in ex.map(one, jobs):
            n += 1
            if n % 500 == 0:
                print(n, len(jobs), flush=True)
    print("k5 done", len(jobs))


def k1(out, pd_csv):
    W = pd.read_csv(pd_csv)
    jobs = [(r.pair, r.day) for r in W.itertuples() if not os.path.exists(f"{out}/{r.pair}/{r.day}.csv")]

    def one(j):
        p, d = j
        raw = _get(f"{V}/klines/{p}/1m/{p}-1m-{d}.zip")
        if raw is None:
            return p, d, 0
        df = _kl(raw); os.makedirs(f"{out}/{p}", exist_ok=True); df.to_csv(f"{out}/{p}/{d}.csv", index=False)
        return p, d, len(df)
    with ThreadPoolExecutor(8) as ex:
        for p, d, k in ex.map(one, jobs):
            print(p, d, k, flush=True)


def ticks(out, pd_csv):
    W = pd.read_csv(pd_csv)
    jobs = [(r.pair, r.day) for r in W.itertuples() if not os.path.exists(f"{out}/{r.pair}/{r.day}.npz")]

    def one(j):
        p, d = j
        raw = _get(f"{V}/aggTrades/{p}/{p}-aggTrades-{d}.zip")
        if raw is None:
            return p, d, 0
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            b = z.read(z.namelist()[0])
        df = pd.read_csv(io.BytesIO(b), header=None, low_memory=False)
        if not str(df.iloc[0, 0]).isdigit():
            df = df.iloc[1:]
        t = df.iloc[:, 5].astype("int64").values; pr = df.iloc[:, 1].astype(float).values
        os.makedirs(f"{out}/{p}", exist_ok=True)
        np.savez_compressed(f"{out}/{p}/{d}.npz", t=t, p=pr)
        return p, d, len(t)
    with ThreadPoolExecutor(4) as ex:
        for p, d, k in ex.map(one, jobs):
            print(p, d, k, flush=True)


def api(out, t0, pairs, iv, step):
    if BC.active():
        raise SystemExit("cooldown active — stop")
    os.makedirs(out, exist_ok=True)
    now = int(time.time() * 1000)
    for p in pairs:
        fn = f"{out}/{p}.csv"
        a, rows = int(t0), []
        while a < now:
            if BC.active():
                raise SystemExit("cooldown active — stop")
            n = 499
            try:
                with urllib.request.urlopen(API.format(s=p, i=iv, a=a, n=n), timeout=30) as r:
                    w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0") or 0)
                    d = json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (418, 429):
                    ra = int(e.headers.get("Retry-After", "120") or 120)
                    with open(BC.SHARED, "w") as f:
                        f.write(str(int((time.time() + ra) * 1000)))
                    raise SystemExit(f"HTTP {e.code} — cooldown {ra}s written, stop")
                print("skip", p, e.code, flush=True); break
            if w > 900:
                print("weight", w, "sleep 61", flush=True); time.sleep(61)
            if not d:
                break
            rows += d
            a = int(d[-1][0]) + step
            if len(d) < n:
                break
            time.sleep(0.12)
        if rows:
            df = pd.DataFrame([r[:6] + [r[7]] for r in rows], columns=["open_time", "o", "h", "l", "c", "vol", "qvol"])
            df = df.astype(float).astype({"open_time": "int64"}).drop_duplicates("open_time")
            df = df[df.open_time + step <= now]                     # closed bars only
            df.to_csv(fn, index=False)
        time.sleep(0.12)
    print("api done", len(pairs))


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)

    def plist(i):
        if len(sys.argv) > i:
            return [x.strip() for x in open(sys.argv[i]) if x.strip()]
        return sorted(f[:-4] for f in os.listdir(f"{ROOT}/reports/backtest_cache/k5m_full") if f.endswith(".csv"))
    if mode == "k5":
        k5(out, sys.argv[3], sys.argv[4], plist(5))
    elif mode == "k5api":
        api(out, int(sys.argv[3]), plist(4), "5m", 300_000)
    elif mode == "k1api":
        api(out, int(sys.argv[3]), plist(4), "1m", 60_000)
    elif mode == "k1":
        k1(out, sys.argv[3])
    elif mode == "ticks":
        ticks(out, sys.argv[3])
