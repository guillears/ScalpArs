#!/usr/bin/env python3
"""NOSPOT_HYBRID (PREREG reports/NOSPOT_HYBRID_2026-11-18_PREREG.md) — FEATURES. For every NS0 signal row (15m RSI cross 70 → ≤ 70 ∧
vol_decay ≥ 0.6366) of the NOSPOT_SLEEVE rows (X.pkl, and the live replay rows), the top-trader POSITION / ACCOUNT ratio ABSOLUTE changes
with services/distribution.pos_chg_4h's exact window (km: stamp ≤ T − 5 min, T − stamp ≤ 30 min; k4: stamp ≤ stamp(km) − W,
stamp(km) − stamp(k4) ≤ W + 15 min), W = 4 h and 1 h; plus ADX15 / ADX1h (Wilder 14) at the signal bar from the 5m klines.
Output reports/NOSPOT_HYBRID_2026-11-18_data/feat.pkl (index = X row) and feat_live.pkl.
Usage: venv/bin/python -I scripts/nospot_hybrid_build.py [--procs 7]"""
import argparse, glob, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import nospot_sleeve_build as NB   # noqa: E402
import nospot_sleeve_lib as L   # noqa: E402
TB = NB.TB; XB = NB.XB
OD = os.path.join(R, "reports", "NOSPOT_HYBRID_2026-11-18_data"); os.makedirs(OD, exist_ok=True)
LD = os.path.join(L.ND, "live")
M5, H1 = 300_000, 3_600_000
LAG, AGE, TOL = 5 * 60_000, 30 * 60_000, 15 * 60_000


def chg_abs(ms, val, T, W):
    """distribution.pos_chg_4h window on arrays (ms sorted, val > 0 finite rows only)."""
    ok = np.isfinite(val) & (val > 0); ms, val = ms[ok], val[ok]
    out = np.full(len(T), np.nan); now = np.full(len(T), np.nan)
    if not len(ms):
        return out, now
    k = np.searchsorted(ms, T - LAG, side="right") - 1
    g = (k >= 0) & (T - ms[np.maximum(k, 0)] <= AGE)
    kc = np.maximum(k, 0); tc = ms[kc]
    k4 = np.searchsorted(ms, tc - W, side="right") - 1
    g4 = g & (k4 >= 0) & (tc - ms[np.maximum(k4, 0)] <= W + TOL)
    now[g] = val[kc[g]]
    out[g4] = val[kc[g4]] - val[np.maximum(k4, 0)[g4]]
    return out, now


def adx(h, l, c, n=14):
    up = np.r_[0, np.diff(h)]; dn = np.r_[0, -np.diff(l)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a = TB.wilder(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * TB.wilder(pdm, n) / a; mdi = 100 * TB.wilder(mdm, n) / a
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    dx = np.nan_to_num(dx, nan=0.0)
    return TB.wilder(dx, n)


def work(args):
    p, idx, T, mfiles = args
    T = np.asarray(T, "int64")
    days = set()
    for t_ in T:
        for d_ in pd.date_range(pd.Timestamp(t_ - 6 * H1, unit="ms").normalize(), pd.Timestamp(t_, unit="ms").normalize()):
            days.add(d_.strftime("%Y-%m-%d"))
    M = NB.load_metrics(mfiles, days)
    res = dict(xi=np.asarray(idx))
    for nm in ("pos4", "pos1", "acc4", "acc1", "pos_now", "acc_now", "adx15", "adx1h"):
        res[nm] = np.full(len(T), np.nan)
    if M is not None and len(M):
        ms = M.ms.values.astype("int64")
        tP = M.sum_toptrader_long_short_ratio.values.astype(float); tA = M.count_toptrader_long_short_ratio.values.astype(float)
        res["pos4"], res["pos_now"] = chg_abs(ms, tP, T, 4 * H1)
        res["pos1"], _ = chg_abs(ms, tP, T, H1)
        res["acc4"], res["acc_now"] = chg_abs(ms, tA, T, 4 * H1)
        res["acc1"], _ = chg_abs(ms, tA, T, H1)
    x = TB.load(p)
    if x is not None:
        t, o, h, l, c, v, qv = x
        st15, en15, O15, H15, L15, C15, V15 = XB.agg(t, o, h, l, c, v, 900_000)
        A15 = adx(H15, L15, C15)
        j = np.searchsorted(t, T) - 1                       # last 5m bar closed before T = the signal 15m bar's last 5m bar
        k = np.searchsorted(en15, j)
        ok = (k < len(en15)) & (en15[np.minimum(k, len(en15) - 1)] == j)
        res["adx15"][ok] = A15[k[ok]]
        s1, e1, _, H1h, L1h, C1, _ = XB.agg(t, o, h, l, c, v, H1)
        A1 = adx(H1h, L1h, C1); full1 = (e1 - s1) == 11
        k1 = np.searchsorted(t[e1], t[np.maximum(j, 0)], side="right") - 1
        ok1 = (k1 >= 0) & full1[np.maximum(k1, 0)]
        res["adx1h"][ok1] = A1[np.maximum(k1, 0)[ok1]]
    return p, res


def run(X, mdirs, procs):
    sig = np.nonzero(L.ns0_mask(X))[0]
    MI = NB.met_index(sorted(set(mdirs)))
    jobs = []
    for p, g in pd.Series(sig).groupby(X.pair.values[sig]):
        jobs.append((p, g.values, X.entry_ms.values[g.values], MI.get(p, [])))
    parts = []
    with mp.get_context("fork").Pool(procs) as pool:
        for i, (p, r) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            parts.append(pd.DataFrame(r))
            if i % 25 == 0:
                print(i, len(jobs), p, flush=True)
    return pd.concat(parts).set_index("xi").sort_index()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=7); A = ap.parse_args()
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    F = run(X, NB.MET_DIRS + [os.path.join(L.ND, "metrics")], A.procs)
    F.to_pickle(os.path.join(OD, "feat.pkl"))
    print("hist rows", len(F), "pos4 scored", np.isfinite(F.pos4).mean().round(3), flush=True)
    # live replay rows (PHAROS / PIPPIN / BLESS / JELLYJELLY 11-10 → 11-18)
    TB.K5M_SRC.append(os.path.join(LD, "k5m")); TB.FUND_SRC.append(os.path.join(LD, "funding"))
    LV = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(os.path.join(LD, "rows", "*.pkl")))], ignore_index=True)
    LV.to_pickle(os.path.join(OD, "live_rows.pkl"))
    FL = run(LV, NB.MET_DIRS + [os.path.join(L.ND, "metrics"), os.path.join(LD, "metrics")], 4)
    FL.to_pickle(os.path.join(OD, "feat_live.pkl"))
    print("live rows", len(FL), flush=True)


if __name__ == "__main__":
    main()
