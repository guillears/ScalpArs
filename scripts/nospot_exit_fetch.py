#!/usr/bin/env python3
"""NOSPOT_EXIT (PREREG reports/NOSPOT_EXIT_2026-11-18_PREREG.md) — DATA. 1m futures klines (data.binance.vision daily zips, 0 API weight,
generic UA via lead1_data.get, stops on 403/418/429) for every pair-day an exit path can touch: entry day + 2 days for every eligible
bar of every episode where NS0 / R1 / R4 fires (cohort trades + the within-episode random pool).
Reuses reports/EXH_EXIT_GRID_data/k1m/<PAIR>/<day>.csv when present; new days → reports/NOSPOT_EXIT_data/k1m/<PAIR>/<day>.npz
(t int64 ms, o/h/l/c float64; a missing day writes an empty marker <day>.none).
Usage: venv/bin/python -I scripts/nospot_exit_fetch.py"""
import io, os, sys, time, zipfile
import concurrent.futures as cf
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import lead1_data as LD          # noqa: E402
import nospot_sleeve_lib as L    # noqa: E402
OUT = os.path.join(R, "reports", "NOSPOT_EXIT_data", "k1m")
OLD = os.path.join(R, "reports", "EXH_EXIT_GRID_data", "k1m")
VIS = "https://data.binance.vision/data/futures/um/daily/klines"
DAY = 86_400_000


def need():
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    M = L.rule_masks(X)
    u = L.ns0_mask(X) | M["R1_OI_UNWIND"][0] | M["R4_TOP_DISTRIB"][0]
    inep = np.isin(X.epi.values, np.unique(X.epi.values[u]))
    s = set()
    last = pd.Timestamp("2026-11-18").value // 10**6 // DAY
    for p, t in zip(X.pair.values[inep], X.entry_ms.values[inep]):
        for k in range(3):
            d = int(t) // DAY + k
            if d < last:
                s.add((p, d))
    return sorted(s)


def one(pd_):
    p, d = pd_
    ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d")
    if os.path.exists(os.path.join(OLD, p, ds + ".csv")):
        return 0
    fo = os.path.join(OUT, p, ds + ".npz"); fn = os.path.join(OUT, p, ds + ".none")
    if os.path.exists(fo) or os.path.exists(fn):
        return 0
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    z = LD.get(f"{VIS}/{p}/1m/{p}-1m-{ds}.zip")
    if not z:
        open(fn, "w").close(); return 0
    zz = zipfile.ZipFile(io.BytesIO(z)); lines = zz.open(zz.namelist()[0]).read().decode().strip().split("\n")
    if lines and not lines[0][0].isdigit():
        lines = lines[1:]
    a = np.array([ln.split(",")[:5] for ln in lines], dtype=float)
    t = a[:, 0].astype(np.int64); t = np.where(t > 10**14, t // 1000, t)
    np.savez_compressed(fo, t=t, o=a[:, 1], h=a[:, 2], l=a[:, 3], c=a[:, 4])
    return 1


def main():
    nd = need(); n = 0; t0 = time.time()
    print("pair-days", len(nd), flush=True)
    with cf.ThreadPoolExecutor(10) as ex:
        for k, r in enumerate(ex.map(one, nd)):
            n += r
            if k % 500 == 0:
                print(time.strftime("%H:%M:%S"), k, len(nd), "fetched", n, f"{time.time() - t0:.0f}s", flush=True)
    print("done", len(nd), "fetched", n, flush=True)


if __name__ == "__main__":
    main()
