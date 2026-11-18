#!/usr/bin/env python3
"""EXH_EXIT2 (PREREG reports/EXH_EXIT2_2026-11-18_PREREG.md) — DATA. Research only, data.binance.vision only (0 API weight, generic UA
via lead1_data.get, stops on 403 / 418 / 429).
  rows     → reports/EXH_EXIT2_data/rows_{EXH,RND,E}.pkl  [xi, pair, entry_ms, atrp, q5, slip, ...]
             EXH = BASELINE_TRADES_EXHAUSTION kept rows (dl582_block False) · RND = 3 random 5m bars per EXH fill in the same pair's
             Frenzy episode window [shortlist, off] ∩ [entry ± 24 h] (seed 20261118) · E = cand_E (2023-H1) + DL 582 block (needs metrics)
             + one position per pair under today's exit (REF sim) — the E one-position is applied in exh_exit2_eval.
  metricsE → daily metrics zips for the E candidates' entry days → reports/EXH_EXIT2_data/metrics/<PAIR>.<YYYY-MM-DD>.csv
  k1m      → 1m daily zips for every row's needed day (entry − 1 … entry + 2) missing in the EXH_EXIT_GRID / NOSPOT caches
             → reports/EXH_EXIT2_data/k1m/<PAIR>/<YYYY-MM-DD>.csv (empty file = not on the server)
Usage: venv/bin/python -I scripts/exh_exit2_data.py rows | metricsE | k1m"""
import io, os, sys, time, zipfile
import concurrent.futures as cf
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import lead1_data as LD  # noqa: E402
RP = os.path.join(R, "reports")
D = os.path.join(RP, "EXH_EXIT2_data")
G = os.path.join(RP, "EXH_EXIT_GRID_data")
K1_CSV = [os.path.join(G, "k1m"), os.path.join(D, "k1m")]
K1_NPZ = os.path.join(RP, "NOSPOT_EXIT_data", "k1m")
VIS = "https://data.binance.vision/data/futures/um"
DAY, H1, M5 = 86_400_000, 3_600_000, 300_000
NOTIONAL = 6_100.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def slip_of(q5):
    q5 = np.asarray(q5, float)
    return np.maximum(0.10, -0.042 + 0.276 * np.sqrt(NOTIONAL / np.maximum(q5, 1.0)))


def rows():
    os.makedirs(D, exist_ok=True)
    B = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    B = B[~B.dl582_block.astype(bool)].reset_index(drop=True)
    E = pd.DataFrame(dict(xi=np.arange(len(B)), pair=B.pair, entry_ms=B.entry_ms.astype("int64"), atrp=B.atrp, q5=B.q5,
                          slip=slip_of(B.q5), half=B.half, btc_d=B.btc_d50_200, base_r=B.r, base_kind=B.exit_kind, base_exit=B.exit_ms))
    E.to_pickle(os.path.join(D, "rows_EXH.pkl")); log("EXH", len(E))
    # random control
    tm = pd.read_csv(os.path.join(RP, "EXHAUSTION_EXPAND_data", "ep_timing.csv"))
    rng = np.random.default_rng(20261118); out = []; nep = 0
    for i, rw in E.iterrows():
        g = tm[(tm.pair == rw.pair) & (tm.shortlist_ms <= rw.entry_ms) & (tm.off_ms >= rw.entry_ms)]
        if not len(g):
            g = tm[(tm.pair == rw.pair) & (tm.on_ms <= rw.entry_ms) & (tm.off_ms + 2 * DAY >= rw.entry_ms)]
        if not len(g):
            continue
        nep += 1
        e = g.iloc[-1]; a = max(int(e.shortlist_ms if e.shortlist_ms > 0 else e.on_ms), rw.entry_ms - 24 * H1)
        b = min(int(e.off_ms), rw.entry_ms + 24 * H1)
        bars = np.arange((a + M5 - 1) // M5 * M5, b + 1, M5); bars = bars[bars != rw.entry_ms]
        if not len(bars):
            continue
        for t in rng.choice(bars, size=min(3, len(bars)), replace=False):
            out.append(dict(src=int(rw.xi), pair=rw.pair, entry_ms=int(t), atrp=rw.atrp, q5=rw.q5, slip=rw.slip, half=rw.half, btc_d=rw.btc_d))
    Rn = pd.DataFrame(out); Rn["xi"] = np.arange(len(Rn))
    Rn.to_pickle(os.path.join(D, "rows_RND.pkl")); log("RND", len(Rn), "fills with an episode", nep, "of", len(E))
    # E
    C = pd.read_pickle(os.path.join(G, "cand_E.pkl")).reset_index(drop=True)
    Ce = pd.DataFrame(dict(xi=np.arange(len(C)), pair=C.pair, entry_ms=C.entry_ms.astype("int64"), atrp=C.atrp, q5=C.q5, slip=slip_of(C.q5),
                           ssc=C.ssc, half="23-H1"))
    Ce.to_pickle(os.path.join(D, "rows_E.pkl")); log("E", len(Ce))


def need_days():
    s = set()
    for nm in ("EXH", "RND", "E"):
        x = pd.read_pickle(os.path.join(D, f"rows_{nm}.pkl"))
        for p, t in zip(x.pair, x.entry_ms):
            for k in range(0, 3):
                s.add((p, pd.to_datetime(int(t) + k * DAY, unit="ms").strftime("%Y-%m-%d")))
    return sorted(s)


def have(p, d):
    for k in K1_CSV:
        f = os.path.join(k, p, d + ".csv")
        if os.path.exists(f):
            return True
    return os.path.exists(os.path.join(K1_NPZ, p, d + ".npz"))


def one_1m(pd_):
    p, d = pd_
    fo = os.path.join(D, "k1m", p, d + ".csv")
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    z = LD.get(f"{VIS}/daily/klines/{p}/1m/{p}-1m-{d}.zip")
    if not z:
        open(fo, "w").close(); return 0
    zz = zipfile.ZipFile(io.BytesIO(z)); lines = zz.open(zz.namelist()[0]).read().decode().strip().split("\n")
    if lines and not lines[0][0].isdigit():
        lines = lines[1:]
    open(fo, "w").write("\n".join(["t,o,h,l,c"] + [",".join(ln.split(",")[:5]) for ln in lines])); return 1


def k1m():
    nd = [x for x in need_days() if not have(*x)]
    log("need", len(nd))
    n = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(one_1m, nd)):
            n += r
            if k % 500 == 0:
                log("k1m", k, len(nd), n)
    log("k1m done", len(nd), "fetched", n)


def metricsE():
    x = pd.read_pickle(os.path.join(D, "rows_E.pkl"))
    od = os.path.join(D, "metrics"); os.makedirs(od, exist_ok=True)
    need = sorted({(p, pd.to_datetime(int(t), unit="ms").strftime("%Y-%m-%d")) for p, t in zip(x.pair, x.entry_ms)} |
                  {(p, pd.to_datetime(int(t) - DAY, unit="ms").strftime("%Y-%m-%d")) for p, t in zip(x.pair, x.entry_ms)})

    def one(pd_):
        p, d = pd_; fo = os.path.join(od, f"{p}.{d}.csv")
        if os.path.exists(fo):
            return 0
        z = LD.get(f"{VIS}/daily/metrics/{p}/{p}-metrics-{d}.zip")
        if not z:
            open(fo, "w").close(); return 0
        zz = zipfile.ZipFile(io.BytesIO(z)); open(fo, "wb").write(zz.open(zz.namelist()[0]).read()); return 1
    with cf.ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(one, need))
    log("metricsE", len(need), "fetched", n)


if __name__ == "__main__":
    {"rows": rows, "k1m": k1m, "metricsE": metricsE}[sys.argv[1]]()
