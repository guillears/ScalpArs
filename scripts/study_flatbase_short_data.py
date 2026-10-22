#!/usr/bin/env python3
"""FRENZY flat-base short study (2026-10-22) — DATA step. Read-only on the bot; writes only reports/FRENZY_FLATBASE_SHORT_2026-10-22_data/.
Pre-registration: reports/FRENZY_FLATBASE_SHORT_2026-10-22_PREREG.md.

  symbols   S3 listing of data.binance.vision futures/um monthly klines (every USDT perp that ever had a file) + fapi exchangeInfo (weight 1)
            → data/symbols.json {trading: [...], delisted: [...]}
  delisted  5m history (2024-06 → end) for every delisted USDT perp: monthly zips + daily zips of the final partial month (0 API weight)
            → data/k5m/<PAIR>.csv  (the survivorship fix: the caches hold only 11 delisted pairs)
  extend    every TRADING pair: daily 5m zips 2026-10-20, 10-21 (pairs absent from every cache: monthly 2026-08/09 + daily Oct) → data/k5m
  today     every TRADING pair: fapi /fapi/v1/klines 2026-10-22 00:00 → now (closed bars; weight 2 per call)
  fund      funding for data/fund_needed.csv (pair, month): monthly fundingRate zips (≤ 2026-09); fapi /fapi/v1/fundingRate for 2026-10
            → data/funding/<PAIR>.csv (t, rate)
Binance etiquette: scripts/binance_cooldown.py honoured before every fapi call; X-MBX-USED-WEIGHT-1M read (sleep 60 s above 900);
418 / 429 → write nothing more, stop. data.binance.vision = public CDN (no API weight).
Usage: venv/bin/python scripts/study_flatbase_short_data.py --stages symbols,delisted,extend,today | --stages fund
"""
import argparse, glob, io, json, os, re, sys, time, zipfile, urllib.request, urllib.error, concurrent.futures as cf
from urllib.parse import quote
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import binance_cooldown as BC  # noqa: E402

OUT = os.path.join(ROOT, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_data")
OLD = [os.path.join(ROOT, "reports", "backtest_cache_2025", "k5m_full"), os.path.join(ROOT, "reports", "backtest_cache", "k5m_full"),
       os.path.join(ROOT, "reports", "FRENZY_SWING_SHORT_2026-10-20_data", "k5m")]
BASE = "https://data.binance.vision/data/futures/um"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FAPI = "https://fapi.binance.com"
COLS = ["open_time", "o", "h", "l", "c", "vol", "qvol"]
ap = argparse.ArgumentParser()
ap.add_argument("--stages", default="symbols,delisted,extend,today")
ap.add_argument("--threads", type=int, default=6)
A = ap.parse_args()
for d in ("k5m", "funding"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)


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
    try:
        with urllib.request.urlopen(urllib.request.Request(FAPI + path), timeout=30) as r:
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


def s3_list(prefix, delim=True):
    keys, pre, marker = [], [], ""
    while True:
        u = f"{S3}?prefix={quote(prefix)}" + ("&delimiter=/" if delim else "") + (f"&marker={quote(marker)}" if marker else "")
        x = get(u).decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", x); pre += re.findall(r"<Prefix>([^<]+)</Prefix>", x)[1:]
        if "<IsTruncated>true</IsTruncated>" not in x:
            return keys, pre
        nm = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)
        marker = nm[0] if nm else (keys[-1] if keys else pre[-1])


def kl_zip(data):
    z = zipfile.ZipFile(io.BytesIO(data))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4, 5, 7], dtype=str)
    df = df[df[0].str.isdigit()]
    df.columns = COLS
    return df.astype({"open_time": "int64", "o": float, "h": float, "l": float, "c": float, "vol": float, "qvol": float})


def stage_symbols():
    _, pre = s3_list("data/futures/um/monthly/klines/")
    allu = sorted({p.rstrip("/").split("/")[-1] for p in pre if p.rstrip("/").endswith("USDT")})
    x = fapi("/fapi/v1/exchangeInfo")   # weight 1
    trading = sorted(s["symbol"] for s in x["symbols"] if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                     and s.get("status") == "TRADING")
    listed_any = sorted(s["symbol"] for s in x["symbols"] if s.get("quoteAsset") == "USDT")
    tradfi = {s["symbol"] for s in x["symbols"] if s.get("contractType") != "PERPETUAL"}   # TradFi stock / commodity perps: not crypto, out
    delisted = sorted(set(allu) - set(trading) - tradfi)
    json.dump(dict(trading=trading, delisted=delisted, tradfi=sorted(tradfi), exchange_any=listed_any, s3_all=allu), open(os.path.join(OUT, "symbols.json"), "w"))
    log("symbols: s3", len(allu), "trading", len(trading), "delisted/settling", len(delisted))


def write(fp, parts):
    parts = [p for p in parts if p is not None and len(p)]
    if parts:
        df = pd.concat(parts).drop_duplicates("open_time", keep="last").sort_values("open_time")
        df.to_csv(fp, index=False)
        return len(df)
    return 0


def stage_delisted():
    S = json.load(open(os.path.join(OUT, "symbols.json")))

    def one(s):
        fp = os.path.join(OUT, "k5m", f"{s}.csv")
        if os.path.exists(fp + ".done"):
            return s, "skip"
        keys, _ = s3_list(f"data/futures/um/monthly/klines/{s}/5m/", delim=False)
        months = sorted({k.split("-5m-")[1][:7] for k in keys if k.endswith(".zip")})
        months = [m for m in months if m >= "2024-06"]
        parts = []
        for m in months:
            z = get(f"{BASE}/monthly/klines/{quote(s)}/5m/{quote(s)}-5m-{m}.zip")
            if z:
                parts.append(kl_zip(z))
        if months:   # the final partial month (daily files after the last monthly file)
            nxt = (pd.Period(months[-1], "M") + 1).strftime("%Y-%m")
            dk, _ = s3_list(f"data/futures/um/daily/klines/{s}/5m/{s}-5m-{nxt}", delim=False)
            for k in sorted(k for k in dk if k.endswith(".zip")):
                z = get(f"https://data.binance.vision/{k}")
                if z:
                    parts.append(kl_zip(z))
        n = write(fp, parts)
        open(fp + ".done", "w").close()
        return s, n
    k = 0
    with cf.ThreadPoolExecutor(A.threads) as ex:
        for s, r in ex.map(one, S["delisted"]):
            k += 1
            if k % 25 == 0:
                log(f"delisted {k}/{len(S['delisted'])} {s} {r}")
    log("delisted done", len(S["delisted"]))


def stage_extend():
    S = json.load(open(os.path.join(OUT, "symbols.json")))
    have = {os.path.basename(f)[:-4] for d in OLD for f in glob.glob(os.path.join(d, "*.csv"))}
    days = ["2026-10-20", "2026-10-21"]

    def one(s):
        fp = os.path.join(OUT, "k5m", f"{s}.csv")
        if os.path.exists(fp + ".done"):
            return s, "skip"
        parts = []
        dd = days
        if s not in have:
            for m in ("2026-08", "2026-09"):
                z = get(f"{BASE}/monthly/klines/{quote(s)}/5m/{quote(s)}-5m-{m}.zip")
                if z:
                    parts.append(kl_zip(z))
            dd = [d.strftime("%Y-%m-%d") for d in pd.date_range("2026-10-01", "2026-10-21")]
        for d in dd:
            z = get(f"{BASE}/daily/klines/{quote(s)}/5m/{quote(s)}-5m-{d}.zip")
            if z:
                parts.append(kl_zip(z))
        n = write(fp, parts)
        open(fp + ".done", "w").close()
        return s, n
    k = 0
    with cf.ThreadPoolExecutor(A.threads) as ex:
        for s, r in ex.map(one, S["trading"]):
            k += 1
            if k % 100 == 0:
                log(f"extend {k}/{len(S['trading'])} {s} {r}")
    log("extend done")


def stage_today():
    S = json.load(open(os.path.join(OUT, "symbols.json")))
    t0 = pd.Timestamp("2026-10-22").value // 10**6
    now = time.time() * 1000
    for k, s in enumerate(S["trading"]):
        if not s.isascii():          # non-ASCII symbols never enter the shortlist (walk: pair.isascii())
            continue
        fp0 = os.path.join(OUT, "k5m", f"{s}.csv")
        if os.path.exists(fp0) and pd.read_csv(fp0, usecols=[0]).open_time.max() >= now - 30 * 60_000:
            continue                 # resumable: already fetched to now
        rows = fapi(f"/fapi/v1/klines?symbol={s}&interval=5m&startTime={t0}&limit=499")   # weight 2
        df = pd.DataFrame([[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[7])] for r in rows
                           if int(r[6]) < now], columns=COLS)
        fp = os.path.join(OUT, "k5m", f"{s}.csv")
        old = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame(columns=COLS)
        write(fp, [old, df])
        if k % 100 == 0:
            log("today", k, len(S["trading"]), s, len(df))
    log("today done")


def stage_fund():
    need = pd.read_csv(os.path.join(OUT, "fund_needed.csv"))   # pair, month

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
            if k % 500 == 0:
                log("fund monthly", k, len(jobs))
    s0 = pd.Timestamp("2026-10-01").value // 10**6
    S = json.load(open(os.path.join(OUT, "symbols.json")))
    oct_pairs = sorted(set(need[need.month == "2026-10"].pair) & set(S["exchange_any"]))
    for k, pair in enumerate(oct_pairs):
        fp = os.path.join(OUT, "funding", f"{pair}.2026-10.part")
        if os.path.exists(fp):
            continue
        r = fapi(f"/fapi/v1/fundingRate?symbol={pair}&startTime={s0}&limit=1000")   # shared 500 / 5 min / IP limit
        time.sleep(0.5)
        pd.DataFrame([(int(a["fundingTime"]), float(a["fundingRate"])) for a in r], columns=["t", "rate"]).to_csv(fp, index=False)
        if k % 100 == 0:
            log("fund oct", k, len(oct_pairs))
    for pair in sorted(set(need.pair)):
        parts = [pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "funding", f"{pair}.*.part"))]
        parts = [x for x in parts if len(x)]
        if parts:
            pd.concat(parts).drop_duplicates("t").sort_values("t").to_csv(os.path.join(OUT, "funding", f"{pair}.csv"), index=False)
    log("fund done", need.pair.nunique())


if __name__ == "__main__":
    for st in A.stages.split(","):
        {"symbols": stage_symbols, "delisted": stage_delisted, "extend": stage_extend, "today": stage_today, "fund": stage_fund}[st]()
