#!/usr/bin/env python3
"""TP_GRID_2026-11-20_FB — build the REAL FLATBASE_SHORT cohort and fetch its 1m paths + funding history.

Cohort (today's stack, operator question "optimal TP / SL for Flatbase"):
  master  = reports/MASTER_POOL_stacked.csv FLATBASE_SHORT rows with stack_keep == True (today's entry stack)
  batch   = exports_archive/scalpars_orders_paper_2026-11-20_16-01-39.csv FLATBASE_SHORT CLOSED fills (no probe column → none),
            dedup key (opened_at[:19], pair, direction) vs the master.
  Fills opened BEFORE the FB_CONFIRM deploy (DL 428 + 10 min) were entered one 4-h close too early: the master re-prices them from the
  CONFIRM entry (first 4-h close in (fill, fill + 24 h] below the fill price → next bar's open; scout_flatbase_confirm_watch.price_shadow).
  Here they get the same confirm entry rebuilt on 1m (+ 0.25 % entry drift so the level sits where a real fill would) — flagged MODEL.
Data: data.binance.vision daily 1m zips (0 API weight) for complete UTC days, public fapi klines for today only; fapi fundingRate for the
hold. Budget: stop at X-MBX-USED-WEIGHT-1M >= 900, stop on 418 / 429, never call during a scripts/binance_cooldown.py cooldown.
Output: reports/TP_GRID_2026-11-20_FB_data/cohort.csv · k1m/<SYM>.csv · funding/<SYM>.csv
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_FB_fetch.py
Research only — no bot code / config touched.
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import binance_cooldown as BC  # noqa: E402

OUT = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_FB_data")
EXPORT = os.path.join(ROOT, "exports_archive", "scalpars_orders_paper_2026-11-20_16-01-39.csv")
OLD_K1M = os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data", "k1m")
COLS = ["open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tbv", "tbqv", "ig"]
WEIGHT_STOP = 900
HR = 3_600_000
_used = {"w": 0}


def batch_of(k):
    return ("B24" if k < "2026-10-23T18:30" else "B25" if k < "2026-11-11T20:49" else "B26" if k < "2026-11-14T20:30" else "B27")


def cohort():
    import scout_gatekit as GK
    ship = GK.dl_deploy_ms(428, lag_min=10, label="FB_CONFIRM deploy")
    if ship is None:
        sys.exit("FB_CONFIRM deploy time not found")
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    mk = m[(m.entry_strategy == "FLATBASE_SHORT") & (m.stack_keep.astype(str).str.lower() == "true")].copy()
    mk["k"] = mk.opened_at.astype(str).str.replace(" ", "T").str[:19]
    mk["batch"] = mk.k.map(batch_of)
    mk["booked"] = mk.stack_pct
    o = pd.read_csv(EXPORT, low_memory=False)
    cb = o[(o.status == "CLOSED") & (o.entry_strategy == "FLATBASE_SHORT")].copy()
    if "is_probe" in cb:
        cb = cb[~cb.is_probe.astype(str).str.lower().isin(["true", "1"])]
    cb["k"] = cb.opened_at.astype(str).str.replace(" ", "T").str[:19]
    cb["batch"] = "B28"
    cb["booked"] = cb.pnl_percentage
    seen = set(zip(mk.k, mk.pair, mk.direction))
    cb = cb[[(a, b, c) not in seen for a, b, c in zip(cb.k, cb.pair, cb.direction)]]
    cols = ["k", "pair", "direction", "batch", "entry_price", "booked", "pnl_percentage", "close_reason", "closed_at", "entry_funding_rate",
            "funding_fee_usd", "notional_value"]
    R = pd.concat([mk[cols], cb[cols]], ignore_index=True)
    R["te"] = (pd.to_datetime(R.k) - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)
    R["kind"] = np.where(R.te >= ship, "REAL", "MODEL_CONFIRM")
    return R.sort_values("te").reset_index(drop=True), ship


def vision_day(sym, day):
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{sym}/1m/{sym}-1m-{day}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    except Exception:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        txt = z.read(z.namelist()[0]).decode()
    df = pd.read_csv(io.StringIO(txt), header=None)
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df.columns = COLS
    return df.astype({"open_time": "int64"})


def fapi_get(url):
    if BC.active():
        raise RuntimeError("binance cooldown active — stop")
    if _used["w"] >= WEIGHT_STOP:
        raise RuntimeError(f"weight budget reached ({_used['w']})")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            w = r.headers.get("X-MBX-USED-WEIGHT-1M")
            if w is not None:
                _used["w"] = int(w)
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise RuntimeError(f"HTTP {e.code} — stop")
        raise
    time.sleep(0.25)
    return data


def fapi_klines(sym, start_ms, end_ms):
    rows, s = [], start_ms
    while s < end_ms:
        data = fapi_get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1m&startTime={s}&endTime={end_ms}&limit=1000")
        if not data:
            break
        rows += data
        s = int(data[-1][0]) + 1
    return pd.DataFrame(rows, columns=COLS).astype({"open_time": "int64"})


def funding(sym, a, b):
    rows, s = [], a
    while s < b:
        data = fapi_get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&startTime={s}&endTime={b}&limit=1000")
        if not data:
            break
        rows += data
        if len(data) < 1000:
            break
        s = int(data[-1]["fundingTime"]) + 1
    if not rows:
        return pd.DataFrame(columns=["t", "rate", "mark"])
    d = pd.DataFrame(rows)
    return pd.DataFrame({"t": d.fundingTime.astype("int64"), "rate": d.fundingRate.astype(float),
                         "mark": pd.to_numeric(d.get("markPrice", np.nan), errors="coerce")})


def main():
    os.makedirs(os.path.join(OUT, "k1m"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "funding"), exist_ok=True)
    R, ship = cohort()
    R.to_csv(os.path.join(OUT, "cohort.csv"), index=False)
    print(f"cohort {len(R)} · FB_CONFIRM deploy {pd.to_datetime(ship, unit='ms')} · kinds {R.kind.value_counts().to_dict()}")
    now = int(time.time() * 1000)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    need = {}
    for r in R.itertuples():   # te − 10 min … te + 24 h (confirm) + 72 h + 1 h, capped at now
        a, b = r.te - 600_000, min(r.te + (24 + 72 + 1) * HR, now)
        x = need.get(r.pair)
        need[r.pair] = (min(a, x[0]), max(b, x[1])) if x else (a, b)
    for sym, (a, b) in sorted(need.items()):
        f = os.path.join(OUT, "k1m", f"{sym}.csv")
        have = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame(columns=["open_time"])
        frs = [have]
        old = os.path.join(OLD_K1M, f"{sym}.csv")
        if os.path.exists(old):
            frs.append(pd.read_csv(old))
        got = set(pd.concat(frs).open_time.astype("int64")) if any(len(x) for x in frs) else set()
        d0 = datetime.fromtimestamp(a / 1000, timezone.utc).date()
        d1 = datetime.fromtimestamp(b / 1000, timezone.utc).date()
        while d0 <= d1:
            day = d0.strftime("%Y-%m-%d")
            t0 = int(datetime(d0.year, d0.month, d0.day, tzinfo=timezone.utc).timestamp() * 1000)
            lo, hi = max(t0, a // 60000 * 60000), min(t0 + 86_400_000 - 60_000, b // 60000 * 60000 - 60_000)
            if lo <= hi and not all(t in got for t in range(lo, hi + 1, 60_000)):
                df = vision_day(sym, day) if day < today else None
                src = "vision"
                if df is None:
                    df = fapi_klines(sym, t0, min(t0 + 86_400_000, now) - 1)
                    src = "fapi"
                frs.append(df[["open_time", "o", "h", "l", "c", "v", "qv"]])
                print(sym, day, src, len(df), "weight", _used["w"], flush=True)
            d0 += timedelta(days=1)
        k = pd.concat(frs, ignore_index=True)
        k["open_time"] = k.open_time.astype("int64")
        k = k.drop_duplicates("open_time").sort_values("open_time")
        k[["open_time", "o", "h", "l", "c", "v", "qv"]].to_csv(f, index=False)
        fu = funding(sym, a - 8 * HR, b)
        fu.to_csv(os.path.join(OUT, "funding", f"{sym}.csv"), index=False)
        print(sym, "1m bars", len(k), "funding rows", len(fu), "weight", _used["w"], flush=True)


if __name__ == "__main__":
    main()
