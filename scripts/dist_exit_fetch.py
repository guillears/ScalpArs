#!/usr/bin/env python3
"""DIST_EXIT (PREREG reports/DIST_EXIT_2026-11-18_PREREG.md) — DATA. 1m futures klines from data.binance.vision daily zips (0 API weight,
generic UA via lead1_data.get, stops on 403/418/429) for every pair-day a DIST / L1 trade (rows.pkl) or an EXHAUSTION baseline trade can
touch (entry day + 2). Reuses EXH_EXIT_GRID_data/k1m (csv) and NOSPOT_EXIT_data/k1m (npz); new days → reports/DIST_EXIT_data/k1m/<PAIR>/
<day>.npz (same format as nospot_exit_fetch; a missing day writes <day>.none).
Usage: venv/bin/python -I scripts/dist_exit_fetch.py"""
import io, os, sys, time, zipfile
import concurrent.futures as cf
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import lead1_data as LD          # noqa: E402
OUT = os.path.join(R, "reports", "DIST_EXIT_data", "k1m")
OLD = [os.path.join(R, "reports", "EXH_EXIT_GRID_data", "k1m"), os.path.join(R, "reports", "NOSPOT_EXIT_data", "k1m")]
VIS = "https://data.binance.vision/data/futures/um/daily/klines"
DAY = 86_400_000


def need():
    S = pd.read_pickle(os.path.join(R, "reports", "DIST_WL_2026-11-18_data", "rows.pkl"))
    E = pd.read_csv(os.path.join(R, "reports", "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"), usecols=["pair", "entry_ms"])
    last = pd.Timestamp("2026-11-18").value // 10**6 // DAY
    s = set()
    for p, t in list(zip(S.pair.astype(str).values, S.entry_ms.values)) + list(zip(E.pair.values, E.entry_ms.values)):
        for k in range(3):
            d = int(t) // DAY + k
            if d < last:
                s.add((p, d))
    return sorted(s)


def one(pd_):
    p, d = pd_
    ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d")
    if os.path.exists(os.path.join(OLD[0], p, ds + ".csv")) or os.path.exists(os.path.join(OLD[1], p, ds + ".npz")):
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
    with cf.ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(one, nd)):
            n += r
            if k % 1000 == 0:
                print(time.strftime("%H:%M:%S"), k, len(nd), "fetched", n, f"{time.time() - t0:.0f}s", flush=True)
    print("done", len(nd), "fetched", n, flush=True)


if __name__ == "__main__":
    main()
