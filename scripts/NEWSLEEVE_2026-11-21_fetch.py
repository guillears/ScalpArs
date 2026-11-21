#!/usr/bin/env python3
"""NEWSLEEVE 2026-11-21 — DATA (PREREG reports/NEWSLEEVE_2026-11-21_PREREG.md). data.binance.vision only (0 Binance weight).
Stages:
  k5m     5m klines WITH taker-buy quote volume, 2023-06 → 2026-11-17, for every symbol-month where the coin's daily quote volume
          reached ≥ $10M at least once (screened on the reused reports/HIGH_ATR_2026-11-18_data/k5m cache, which lacks taker-buy)
          plus the previous month (7-day baseline) → k5m/<SYM>.npz (t int64, o h l c float64, qv tb float32)
  metrics daily 5m metrics for the pair-days listed in metrics_need.csv (written by the build stage) → met/<SYM>/<day>.npy
          (t_ms, oi_value, oi_contracts). oi_value / oi_contracts = the mark price at the snapshot, used by the eval to find the TRUE
          snapshot time (later vision files label a snapshot taken 5 min after create_time — look-ahead if read naively).
Writes only reports/NEWSLEEVE_2026-11-21_data/."""
import concurrent.futures as cf, io, os, sys, time, zipfile, urllib.request, urllib.error
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "NEWSLEEVE_2026-11-21_data")
HA = os.path.join(R, "reports", "HIGH_ATR_2026-11-18_data")
for s in ("k5m", "met"):
    os.makedirs(os.path.join(D, s), exist_ok=True)
VIS = "https://data.binance.vision/data/futures/um"
UA = {"User-Agent": "Mozilla/5.0 (research client)"}
M0, M1 = "2023-06", "2026-10"
NOV = [f"2026-11-{d:02d}" for d in range(1, 18)]
EXCL = {"BTCUSDT", "ETHUSDT", "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "DAIUSDT", "BTCDOMUSDT", "DEFIUSDT",
        "FOOTBALLUSDT", "BLUEBIRDUSDT"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url):
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 418, 429):
                raise SystemExit(f"HTTP {e.code} — stopping")
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    return None


def parse_kl(z):
    zz = zipfile.ZipFile(io.BytesIO(z))
    b = zz.open(zz.namelist()[0]).read().decode().strip().split("\n")
    if b and not b[0][0].isdigit():
        b = b[1:]
    return np.array([[float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[7]), float(p[10])]
                     for p in (ln.split(",") for ln in b)], dtype=np.float64)


def months_needed(sym):
    """symbol-months with any UTC-day quote volume ≥ $10M (+ previous month), screened on the HIGH_ATR cache."""
    f = os.path.join(HA, "k5m", sym + ".npy")
    if not os.path.exists(f):
        return set(), False
    a = np.load(f)
    if len(a) == 0:
        return set(), False
    day = (a[:, 0] // 86_400_000).astype(np.int64)
    s = pd.Series(a[:, 6]).groupby(day).sum()
    good = s.index[s.values >= 10e6]
    ms = set()
    for d in good:
        m = pd.Timestamp(int(d) * 86_400_000, unit="ms")
        ms.add(m.strftime("%Y-%m"))
        ms.add((m - pd.offsets.MonthBegin(1)).strftime("%Y-%m"))
    nov = any(m >= "2026-10" for m in ms)
    return {m for m in ms if M0 <= m <= M1}, nov


def one_k5m(sym):
    fo = os.path.join(D, "k5m", sym + ".npz")
    if os.path.exists(fo):
        return sym, -1
    ms, nov = months_needed(sym)
    parts = []
    for m in sorted(ms):
        z = get(f"{VIS}/monthly/klines/{sym}/5m/{sym}-5m-{m}.zip")
        if z:
            try:
                parts.append(parse_kl(z))
            except Exception as e:
                log("bad", sym, m, e)
    if nov:
        for d in NOV:
            z = get(f"{VIS}/daily/klines/{sym}/5m/{sym}-5m-{d}.zip")
            if z:
                try:
                    parts.append(parse_kl(z))
                except Exception:
                    pass
    if not parts:
        np.savez(fo, t=np.zeros(0, np.int64))
        return sym, 0
    k = np.concatenate(parts)
    k[k[:, 0] > 1e14, 0] //= 1000
    k = k[(k[:, 0] % 300_000 == 0) & np.isfinite(k).all(1) & (k[:, 4] > 0)]
    k = k[np.argsort(k[:, 0], kind="stable")]
    _, ix = np.unique(k[:, 0], return_index=True)
    k = k[ix]
    np.savez(fo, t=k[:, 0].astype(np.int64), o=k[:, 1], h=k[:, 2], l=k[:, 3], c=k[:, 4], qv=k[:, 5].astype(np.float32),
             tb=k[:, 6].astype(np.float32))
    return sym, len(k)


def stage_k5m():
    syms = [s for s in pd.read_csv(os.path.join(HA, "symbols.csv")).sym if s.endswith("USDT") and s not in EXCL] + ["BTCUSDT"]
    log("symbols", len(syms))
    n = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for s, k in ex.map(one_k5m, syms):
            n += 1
            if n % 50 == 0:
                log(n, s, k)
    log("k5m done")


def one_met(job):
    sym, day = job
    fo = os.path.join(D, "met", sym, day + ".npy")
    if os.path.exists(fo) and (os.path.getsize(fo) < 200 or np.load(fo).shape[1] == 3):
        return 0
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    z = get(f"{VIS}/daily/metrics/{sym}/{sym}-metrics-{day}.zip")
    rows = np.zeros((0, 2))
    if z:
        try:
            zz = zipfile.ZipFile(io.BytesIO(z))
            x = pd.read_csv(zz.open(zz.namelist()[0]), usecols=["create_time", "sum_open_interest", "sum_open_interest_value"])
            t = pd.to_datetime(x.create_time).values.astype("datetime64[ms]").astype(np.int64)
            rows = np.c_[t.astype(np.float64), x.sum_open_interest_value.values.astype(float), x.sum_open_interest.values.astype(float)]
        except Exception as e:
            log("bad met", sym, day, e)
    np.save(fo, rows)
    return 1


def stage_metrics():
    need = pd.read_csv(os.path.join(D, "metrics_need.csv"))
    jobs = list(need[["sym", "day"]].itertuples(index=False, name=None))
    log("metrics jobs", len(jobs))
    n = 0
    with cf.ThreadPoolExecutor(32) as ex:
        for r in ex.map(one_met, jobs):
            n += 1
            if n % 500 == 0:
                log("met", n)
    log("metrics done")


if __name__ == "__main__":
    {"k5m": stage_k5m, "metrics": stage_metrics}[sys.argv[1]]()
