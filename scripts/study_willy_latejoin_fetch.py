#!/usr/bin/env python3
"""WILLY late joiners (2026-10-22) — 1m DATA step. Read-only on the bot; writes only reports/WILLY_LATE_JOIN_2026-10-22_data/k1m/.
  archive  data.binance.vision daily 1m kline zips for every (pair, day) in need_days.csv (+ --extra file) — public CDN, 0 API weight.
  fapi     days the archive does not have yet (today / yesterday): /fapi/v1/klines 1m (weight 5 per 1000-bar call... limit 1000 → weight 5),
           scripts/binance_cooldown.py honoured, X-MBX-USED-WEIGHT-1M read (sleep 60 s above 900), stop on 418 / 429.
Usage: venv/bin/python scripts/study_willy_latejoin_fetch.py [--extra reports/WILLY_LATE_JOIN_2026-10-22_data/need_days_live.csv] [--threads 8]
"""
import argparse, io, json, os, sys, time, zipfile, urllib.request, urllib.error, concurrent.futures as cf
from urllib.parse import quote
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_latejoin_common as C  # noqa: E402
import binance_cooldown as BC  # noqa: E402

BASE = "https://data.binance.vision/data/futures/um/daily/klines"
FAPI = "https://fapi.binance.com"


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 418, 429):
                raise SystemExit(f"HTTP {e.code} on {url} — stopping")
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return "ERR"


def save(pair, day, a):
    os.makedirs(os.path.join(C.K1, pair), exist_ok=True)
    np.savez_compressed(os.path.join(C.K1, pair, f"{day}.npz"), a=a)


def archive(pd_):
    pair, day = pd_
    f = os.path.join(C.K1, pair, f"{day}.npz")
    if os.path.exists(f):
        return pair, day, "have"
    z = get(f"{BASE}/{quote(pair)}/1m/{quote(pair)}-1m-{day}.zip")
    if z is None:
        return pair, day, "404"
    if z == "ERR":
        return pair, day, "err"
    zz = zipfile.ZipFile(io.BytesIO(z))
    df = pd.read_csv(zz.open(zz.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4], dtype=str)
    df = df[df[0].str.isdigit()].astype(float)
    save(pair, day, df.values)
    return pair, day, "ok"


def fapi_day(pair, day):
    if BC.active():
        raise SystemExit("shared Binance cooldown active — stopping")
    t0 = int(pd.Timestamp(day).value // 10**6); now = time.time() * 1000
    rows = []
    for s in (t0, t0 + 1000 * C.MIN):
        if s > now:
            break
        try:
            with urllib.request.urlopen(f"{FAPI}/fapi/v1/klines?symbol={pair}&interval=1m&startTime={s}&endTime={min(t0 + C.DAY - 1, s + 1000 * C.MIN - 1)}&limit=1000",
                                        timeout=30) as r:
                w = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
                rows += [x for x in json.loads(r.read()) if int(x[6]) < now]
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                raise SystemExit(f"fapi {e.code} — stopping")
            return "err"
        if w > 900:
            print("weight", w, "→ sleep 60", flush=True); time.sleep(60)
        time.sleep(0.3)
    if rows:
        save(pair, day, np.array([[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])] for x in rows]))
    return "ok" if rows else "empty"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extra", default=""); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--refresh_fapi", action="store_true"); A = ap.parse_args()
    need = pd.read_csv(os.path.join(C.D, "need_days.csv"))
    if A.extra:
        need = pd.concat([need, pd.read_csv(A.extra)]).drop_duplicates()
    jobs = list(need.itertuples(index=False, name=None))
    st = {}
    with cf.ThreadPoolExecutor(A.threads) as ex:
        for k, (p, d, s) in enumerate(ex.map(archive, jobs)):
            st.setdefault(s, []).append((p, d))
            if k % 500 == 0:
                print(k, len(jobs), {a: len(b) for a, b in st.items()}, flush=True)
    recent = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=3)
    todo = [(p, d) for p, d in st.get("404", []) if pd.Timestamp(d) >= recent]
    if A.refresh_fapi:
        todo += [(p, d) for p, d in jobs if pd.Timestamp(d) >= recent - pd.Timedelta(days=0) and (p, d) not in todo
                 and pd.Timestamp(d) >= pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)]
    for p, d in todo:
        print("fapi", p, d, fapi_day(p, d), flush=True)
    print({a: len(b) for a, b in st.items()}, "fapi", len(todo))
    old404 = [x for x in st.get("404", []) if x not in todo]
    pd.DataFrame(old404, columns=["pair", "day"]).to_csv(os.path.join(C.D, "missing_days.csv"), index=False)
