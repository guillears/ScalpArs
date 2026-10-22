#!/usr/bin/env python3
"""FRENZY swing-short study (2026-10-20) — DATA step. Read-only on the bot; writes only reports/FRENZY_SWING_SHORT_2026-10-20_data/.

  k5m    extends every pair of reports/backtest_cache/k5m_full (ends 2026-10-12) to 2026-10-19 23:55 from data.binance.vision DAILY
         5m zips (0 API weight) + every USDT perp listed since (monthly 2026-09 + daily Oct zips) → data/k5m/<PAIR>.csv (Oct-01 → )
  today  2026-10-20 00:00 → now for the pairs passed with --today (fapi /fapi/v1/klines, weight 2 per call, header-checked)
  fund   funding history for the (pair, month) list in data/fund_needed.csv: monthly fundingRate zips (data.binance.vision) for
         months ≤ 2026-09; fapi /fapi/v1/fundingRate for 2026-10 → data/funding/<PAIR>.csv (t, rate)
Binance etiquette: honours scripts/binance_cooldown.py, reads X-MBX-USED-WEIGHT-1M (stops above 900), stops on 418 / 429.
Usage: venv/bin/python scripts/study_frenzy_swing_short_data.py --stages k5m | --stages today --today ERAUSDT,ALTUSDT | --stages fund
"""
import argparse, glob, io, json, os, sys, time, zipfile, urllib.request, urllib.error, concurrent.futures as cf
from urllib.parse import quote
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import binance_cooldown as BC  # noqa: E402

OUT = os.path.join(ROOT, "reports", "FRENZY_SWING_SHORT_2026-10-20_data")
LIVE = os.path.join(ROOT, "reports", "backtest_cache")
BASE = "https://data.binance.vision/data/futures/um"
FAPI = "https://fapi.binance.com"
COLS = ["open_time", "o", "h", "l", "c", "vol", "qvol"]
DAYS = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-10-01", "2026-10-19")]
ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="k5m")
ap.add_argument("--today", default="")
ap.add_argument("--threads", type=int, default=6)
A = ap.parse_args()
os.makedirs(os.path.join(OUT, "k5m"), exist_ok=True); os.makedirs(os.path.join(OUT, "funding"), exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 418, 429):
                raise SystemExit(f"HTTP {e.code} on {url} — stopping")
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed: {url}")


def fapi(path):
    if BC.active():
        raise SystemExit("shared Binance cooldown active — stopping")
    req = urllib.request.Request(FAPI + path)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise SystemExit(f"fapi {e.code} — stopping")
        raise
    if w > 900:
        log(f"used weight {w} > 900 — sleeping 60 s"); time.sleep(60)
    time.sleep(0.25)
    return body


def kl_zip(data):
    z = zipfile.ZipFile(io.BytesIO(data))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4, 5, 7], dtype=str)
    df = df[df[0].str.isdigit()]
    df.columns = COLS
    return df.astype({"open_time": "int64", "o": float, "h": float, "l": float, "c": float, "vol": float, "qvol": float})


def symbols():
    fp = os.path.join(OUT, "exchange_info_symbols.json")
    if not os.path.exists(fp):
        x = fapi("/fapi/v1/exchangeInfo")   # weight 1
        json.dump([s["symbol"] for s in x["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"], open(fp, "w"))
    live = set(json.load(open(fp)))
    cached = {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(LIVE, "k5m_full", "*.csv"))}
    return sorted(cached | live), cached


def stage_k5m():
    syms, cached = symbols()

    def one(s):
        fp = os.path.join(OUT, "k5m", f"{s}.csv")
        if os.path.exists(fp + ".done"):
            return s, "skip"
        parts = []
        if s in cached:
            x = pd.read_csv(os.path.join(LIVE, "k5m_full", f"{s}.csv"))
            parts.append(x[x.open_time >= pd.Timestamp("2026-08-15").value // 10**6])
        else:
            for m in ("2026-08", "2026-09"):
                z = get(f"{BASE}/monthly/klines/{quote(s)}/5m/{quote(s)}-5m-{m}.zip")
                if z:
                    parts.append(kl_zip(z))
        for d in DAYS:
            if s in cached and d < "2026-10-12":
                continue
            z = get(f"{BASE}/daily/klines/{quote(s)}/5m/{quote(s)}-5m-{d}.zip")
            if z:
                parts.append(kl_zip(z))
        if parts:
            df = pd.concat(parts).drop_duplicates("open_time", keep="last").sort_values("open_time")
            df.to_csv(fp, index=False)
        open(fp + ".done", "w").close()
        return s, sum(len(p) for p in parts)
    n = 0
    with cf.ThreadPoolExecutor(A.threads) as ex:
        for s, r in ex.map(one, syms):
            n += 1
            if n % 50 == 0:
                log(f"k5m {n}/{len(syms)} {s} {r}")
    log("k5m done", len(syms))


def stage_today():
    t0 = pd.Timestamp("2026-10-20").value // 10**6
    for s in [x.strip() for x in A.today.split(",") if x.strip()]:
        rows = fapi(f"/fapi/v1/klines?symbol={s}&interval=5m&startTime={t0}&limit=499")   # weight 2
        df = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7])] for r in rows
                           if int(r[6]) < time.time() * 1000], columns=COLS)   # closed bars only
        fp = os.path.join(OUT, "k5m", f"{s}.csv")
        old = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame(columns=COLS)
        pd.concat([old, df]).drop_duplicates("open_time", keep="last").sort_values("open_time").to_csv(fp, index=False)
        log("today", s, len(df))


def stage_fund():
    need = pd.read_csv(os.path.join(OUT, "fund_needed.csv"))   # pair, month (YYYY-MM)

    def monthly(pm):
        pair, m = pm
        fp = os.path.join(OUT, "funding", f"{pair}.{m}.part")
        if os.path.exists(fp):
            return
        z = get(f"{BASE}/monthly/fundingRate/{quote(pair)}/{quote(pair)}-fundingRate-{m}.zip")
        if z:
            zz = zipfile.ZipFile(io.BytesIO(z)); x = pd.read_csv(zz.open(zz.namelist()[0]))
            tc = [c for c in x.columns if "time" in c.lower()][0]; rc = [c for c in x.columns if "rate" in c.lower()][-1]
            pd.DataFrame({"t": x[tc].astype("int64"), "rate": x[rc].astype(float)}).to_csv(fp, index=False)
        else:
            pd.DataFrame(columns=["t", "rate"]).to_csv(fp, index=False)
    jobs = [(p, m) for p, m in need.itertuples(index=False) if m < "2026-10"]
    with cf.ThreadPoolExecutor(A.threads) as ex:
        for k, _ in enumerate(ex.map(monthly, jobs)):
            if k % 200 == 0:
                log("fund monthly", k, len(jobs))
    s0 = pd.Timestamp("2026-10-01").value // 10**6
    for pair in sorted(set(need[need.month == "2026-10"].pair)):
        fp = os.path.join(OUT, "funding", f"{pair}.2026-10.part")
        if os.path.exists(fp):
            continue
        r = fapi(f"/fapi/v1/fundingRate?symbol={pair}&startTime={s0}&limit=1000")
        pd.DataFrame([(int(a["fundingTime"]), float(a["fundingRate"])) for a in r], columns=["t", "rate"]).to_csv(fp, index=False)
    for pair in sorted(set(need.pair)):
        parts = [pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "funding", f"{pair}.*.part"))]
        fp = os.path.join(OUT, "funding", f"{pair}.csv")
        if os.path.exists(fp):
            parts.append(pd.read_csv(fp))
        parts = [x for x in parts if len(x)]
        if parts:
            pd.concat(parts).drop_duplicates("t").sort_values("t").to_csv(fp, index=False)
    log("fund done", need.pair.nunique())


if __name__ == "__main__":
    for st in A.stages.split(","):
        {"k5m": stage_k5m, "today": stage_today, "fund": stage_fund}[st]()
