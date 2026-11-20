#!/usr/bin/env python3
"""DIST_FB_LOSERS_2026-11-20 — fetch 1m klines for the batch's DISTRIBUTION / FLATBASE fills (+ BTC and the
RSI-blocked DIST signal) and 1h klines for the prior move.

Source order: data.binance.vision daily zips first (complete UTC days), then public fapi /fapi/v1/klines for the
days the archive does not hold yet (today). fapi budget: stop at X-MBX-USED-WEIGHT-1M >= 900 (well under 1,200),
stop on 418 / 429, never call while scripts/binance_cooldown.py says a cooldown is active.
Output: reports/DIST_FB_LOSERS_2026-11-20_data/k1m/<SYM>.csv and k1h/<SYM>.csv (open_time ms, o, h, l, c, v, qv).
Research only — no bot code / config touched.
"""
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import binance_cooldown as BC  # noqa: E402

OUT = os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data")
COLS = ["open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n", "tbv", "tbqv", "ig"]
WEIGHT_STOP = 900
_used = {"w": 0}

# symbol -> (first day, last day) of 1m bars needed
PAIRS_1M = {
    "KERNELUSDT": ("2026-11-19", "2026-11-20"), "CYBERUSDT": ("2026-11-19", "2026-11-20"),
    "EVAAUSDT": ("2026-11-18", "2026-11-20"), "NAORISUSDT": ("2026-11-18", "2026-11-20"),
    "SENTUSDT": ("2026-11-18", "2026-11-20"), "IRYSUSDT": ("2026-11-18", "2026-11-20"),
    "PHAROSUSDT": ("2026-11-18", "2026-11-20"), "MMTUSDT": ("2026-11-16", "2026-11-20"),
    "PIPPINUSDT": ("2026-11-18", "2026-11-20"), "BTCUSDT": ("2026-11-16", "2026-11-20"),
}
PAIRS_1H = ["KERNELUSDT", "CYBERUSDT", "EVAAUSDT", "NAORISUSDT", "SENTUSDT", "PHAROSUSDT", "MMTUSDT", "PIPPINUSDT", "IRYSUSDT"]


def _days(a, b):
    d0, d1 = datetime.fromisoformat(a), datetime.fromisoformat(b)
    while d0 <= d1:
        yield d0.strftime("%Y-%m-%d")
        d0 += timedelta(days=1)


def vision_day(sym, day, iv="1m"):
    url = f"https://data.binance.vision/data/futures/um/daily/klines/{sym}/{iv}/{sym}-{iv}-{day}.zip"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = r.read()
    except Exception:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = z.namelist()[0]
        txt = z.read(name).decode()
    df = pd.read_csv(io.StringIO(txt), header=None)
    if not str(df.iloc[0, 0]).isdigit():
        df = df.iloc[1:]
    df.columns = COLS
    return df.astype({"open_time": "int64"})


def fapi(sym, iv, start_ms, end_ms):
    if BC.active():
        raise RuntimeError("binance cooldown active")
    rows = []
    s = start_ms
    while s < end_ms:
        if _used["w"] >= WEIGHT_STOP:
            raise RuntimeError(f"weight budget reached ({_used['w']})")
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={iv}&startTime={s}&endTime={end_ms}&limit=1000"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                w = r.headers.get("X-MBX-USED-WEIGHT-1M")
                if w is None:
                    raise RuntimeError("no weight header")
                _used["w"] = int(w)
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (418, 429):
                raise RuntimeError(f"HTTP {e.code} — stop")
            raise
        if not data:
            break
        rows += data
        s = int(data[-1][0]) + 1
        time.sleep(0.25)
    df = pd.DataFrame(rows, columns=COLS)
    return df.astype({"open_time": "int64"})


def main():
    os.makedirs(os.path.join(OUT, "k1m"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "k1h"), exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for sym, (a, b) in PAIRS_1M.items():
        fr = []
        for day in _days(a, b):
            df = vision_day(sym, day) if day < today else None
            if df is None:
                t0 = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)
                df = fapi(sym, "1m", t0, min(t0 + 86_400_000, int(time.time() * 1000)) - 1)
                src = "fapi"
            else:
                src = "vision"
            print(sym, day, src, len(df), "weight", _used["w"], flush=True)
            fr.append(df)
        k = pd.concat(fr).drop_duplicates("open_time").sort_values("open_time")
        k[["open_time", "o", "h", "l", "c", "v", "qv"]].to_csv(os.path.join(OUT, "k1m", f"{sym}.csv"), index=False)
    end = int(time.time() * 1000)
    for sym in PAIRS_1H:
        df = fapi(sym, "1h", end - 30 * 86_400_000, end)
        print(sym, "1h", len(df), "weight", _used["w"], flush=True)
        df[["open_time", "o", "h", "l", "c", "v", "qv"]].to_csv(os.path.join(OUT, "k1h", f"{sym}.csv"), index=False)


if __name__ == "__main__":
    main()
