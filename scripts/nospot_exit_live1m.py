#!/usr/bin/env python3
"""NOSPOT_EXIT — 1m futures klines for the replay coins (PIPPIN, BLESS, JELLYJELLY, PHAROS) 2026-11-08 → now from the public fapi
(limit 1500, weight 10 per call; reads X-MBX-USED-WEIGHT-1M, pauses > 1,000, stops on 418 / 429; UA "Mozilla/5.0", no personal info).
Writes reports/NOSPOT_EXIT_data/live1m/<PAIR>.npz (t, o, h, l, c). Usage: venv/bin/python -I scripts/nospot_exit_live1m.py"""
import json, os, time, urllib.request, urllib.error
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(R, "reports", "NOSPOT_EXIT_data", "live1m")
PAIRS = ["PIPPINUSDT", "BLESSUSDT", "JELLYJELLYUSDT", "PHAROSUSDT"]
T0 = 1794096000000   # 2026-11-08 00:00 UTC


def get(url):
    for i in range(4):
        try:
            return _get(url)
        except (TimeoutError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError):
                raise
            time.sleep(5 * (i + 1))
    raise SystemExit("network")


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0"))
            return json.loads(r.read()), w
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise SystemExit(f"HTTP {e.code} — stopping")
        raise


def main():
    os.makedirs(OUT, exist_ok=True)
    now = int(time.time() * 1000)
    for p in PAIRS:
        rows = []; s = T0
        while s < now:
            x, w = get(f"https://fapi.binance.com/fapi/v1/klines?symbol={p}&interval=1m&startTime={s}&limit=1500")
            if not x:
                break
            rows += x; s = int(x[-1][0]) + 60_000
            print(p, len(rows), "weight", w, flush=True)
            if w > 1000:
                time.sleep(60)
            time.sleep(0.3)
        a = np.array([[float(r_[i]) for i in range(5)] for r_ in rows])
        cl = np.array([int(r_[6]) for r_ in rows]) < now          # closed bars only
        a = a[cl]
        np.savez_compressed(os.path.join(OUT, p + ".npz"), t=a[:, 0].astype(np.int64), o=a[:, 1], h=a[:, 2], l=a[:, 3], c=a[:, 4])


if __name__ == "__main__":
    main()
