#!/usr/bin/env python3
"""FAST_TP (PREREG reports/FAST_TP_2026-11-20_PREREG.md) — 1m DATA. Research only.
One loader over every existing 1m day cache in the repo / session scratchpad (csv / npz per pair-day, daily / monthly zips),
plus a fetch of missing pair-days from data.binance.vision (0 API weight, lead1_data.get: generic UA, stops on 403 / 418 / 429)
into reports/FAST_TP_2026-11-20_data/k1m/<PAIR>/<YYYY-MM-DD>.csv (empty file = not on the server).
  get1m(pair, t0, t1) → (t, o, h, l, c) numpy arrays for bars with t in [t0, t1) (None when a needed day is missing)
  fetch(list of (pair, day)) → downloads the missing ones (8 threads)"""
import io, os, sys, zipfile, time
import concurrent.futures as cf
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
RP = os.path.join(R, "reports")
D = os.path.join(RP, "FAST_TP_2026-11-20_data")
K1 = os.path.join(D, "k1m")
SP = os.environ.get("FAST_TP_ZIP_DIR", os.path.join(R, "reports", "FAST_TP_2026-11-20_data", "zips"))   # downloaded zip cache (local only)
CSV_DIRS = [os.path.join(RP, "EXH_EXIT_GRID_data", "k1m"), os.path.join(RP, "EXH_EXIT2_data", "k1m"),
            os.path.join(RP, "OOS_REAL_WILLY_2026-11-17_data", "k1m"), K1]
NPZ_DIRS = [os.path.join(RP, "DIST_EXIT_data", "k1m"), os.path.join(RP, "NOSPOT_EXIT_data", "k1m")]
ZIP_DAILY = [os.path.join(SP, "raw1m"), os.path.join(SP, "raw1m_x")]
ZIP_MONTH = os.path.join(SP, "zips1m")
VIS = "https://data.binance.vision/data/futures/um"
DAY, MIN = 86_400_000, 60_000
_day = {}
_mon = {}


def _zip_arr(fn):
    try:
        with zipfile.ZipFile(fn) as z:
            raw = z.read(z.namelist()[0]).decode()
    except Exception:
        return None
    lines = raw.strip().split("\n")
    if lines and not lines[0][:1].isdigit():
        lines = lines[1:]
    if not lines:
        return None
    a = np.array([[float(v) for v in ln.split(",")[:5]] for ln in lines])
    if a[:, 0].max() > 1e14:
        a[:, 0] //= 1000
    return a


def day_arr(p, ds):
    """(n, 5) float array [t, o, h, l, c] for pair p, UTC day 'YYYY-MM-DD', or None."""
    k = (p, ds)
    if k in _day:
        return _day[k]
    a = None
    for d in CSV_DIRS:
        f = os.path.join(d, p, ds + ".csv")
        if os.path.exists(f) and os.path.getsize(f) > 20:
            try:
                x = pd.read_csv(f)
                if len(x):
                    x = x.rename(columns={"open_time": "t"})
                    a = x[["t", "o", "h", "l", "c"]].values.astype(float); break
            except Exception:
                pass
    if a is None:
        for d in NPZ_DIRS:
            f = os.path.join(d, p, ds + ".npz")
            if os.path.exists(f):
                try:
                    z = np.load(f); a = np.column_stack([z[c].astype(float) for c in ("t", "o", "h", "l", "c")]); break
                except Exception:
                    pass
    if a is None:
        for d in ZIP_DAILY:
            f = os.path.join(d, f"{p}-{ds}.zip")
            if os.path.exists(f):
                a = _zip_arr(f)
                if a is not None:
                    break
    if a is None:
        mk = (p, ds[:7])
        if mk not in _mon:
            f = os.path.join(ZIP_MONTH, f"{p}-1m-{ds[:7]}.zip")
            _mon[mk] = _zip_arr(f) if os.path.exists(f) else None
            if len(_mon) > 40:
                _mon.pop(next(iter(_mon)))
        m = _mon.get(mk)
        if m is not None:
            t0 = int(pd.Timestamp(ds).value // 10**6)
            s = m[(m[:, 0] >= t0) & (m[:, 0] < t0 + DAY)]
            a = s if len(s) else None
    if a is not None:
        if a[:, 0].max() > 1e14:
            a[:, 0] //= 1000
        a = a[np.argsort(a[:, 0], kind="stable")]
    if len(_day) > 3000:
        _day.clear()
    _day[k] = a
    return a


def days_of(t0, t1):
    return [pd.Timestamp(d * DAY, unit="ms").strftime("%Y-%m-%d") for d in range(int(t0) // DAY, (int(t1) - 1) // DAY + 1)]


def have(p, ds):
    return day_arr(p, ds) is not None


def get1m(p, t0, t1):
    parts = []
    for ds in days_of(t0, t1):
        a = day_arr(p, ds)
        if a is None:
            return None
        parts.append(a)
    a = np.concatenate(parts)
    a = a[(a[:, 0] >= t0) & (a[:, 0] < t1)]
    if not len(a):
        return None
    _, ix = np.unique(a[:, 0], return_index=True); a = a[ix]
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def _one(pd_):
    import lead1_data as LD
    p, ds = pd_
    fo = os.path.join(K1, p, ds + ".csv")
    if os.path.exists(fo):
        return 0
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    z = LD.get(f"{VIS}/daily/klines/{p}/1m/{p}-1m-{ds}.zip")
    if not z:
        open(fo, "w").close(); return 0
    zz = zipfile.ZipFile(io.BytesIO(z)); lines = zz.open(zz.namelist()[0]).read().decode().strip().split("\n")
    if lines and not lines[0][:1].isdigit():
        lines = lines[1:]
    open(fo, "w").write("\n".join(["t,o,h,l,c"] + [",".join(ln.split(",")[:5]) for ln in lines])); return 1


def fetch(need):
    need = sorted({x for x in need if not have(*x) and not os.path.exists(os.path.join(K1, x[0], x[1] + ".csv"))})
    print(time.strftime("%H:%M:%S"), "fetch need", len(need), flush=True)
    n = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(_one, need)):
            n += r
            if i % 500 == 0:
                print(time.strftime("%H:%M:%S"), "fetch", i, len(need), n, flush=True)
    print(time.strftime("%H:%M:%S"), "fetched", n, "of", len(need), flush=True)
    for x in need:
        _day.pop(x, None)
    return n
