#!/usr/bin/env python3
"""NOSPOT_EXIT (PREREG reports/NOSPOT_EXIT_2026-11-18_PREREG.md) — SIM. Every eligible bar of every episode where NS0 / R1 / R4 fires
(cohort trades + the within-episode random pool) × every pre-registered exit cell, on the 1m path (48 h; 5m fallback flagged).
Fill rules (prereg): entry = 1m open at the entry minute · per bar: the stop as it stood at the bar OPEN is checked first on the HIGH,
then the maker TP on the LOW (≥ 1 tick through); both → stop · stops fill at max(stop, bar open) · the best low updates AFTER the checks
(BE / lock / trail levels apply from the next bar) · time exit = close of the last bar inside the hold.
Kinds: 0 maker TP · 1 initial stop · 2 time · 3 dynamic stop (BE / lock / trail; market).
Output: reports/NOSPOT_EXIT_data/sim/<PAIR>.npz {xi (row index into X.pkl), r (gross short %), k, xm (exit ms), f (funding %), worked,
rec, best, reach, src1m} + path/<PAIR>.npz (5-min sampled fav / adv paths for cohort signal rows, 48 h).
Usage: venv/bin/python -I scripts/nospot_exit_sim.py [--pairs A,B] [--procs 7]"""
import argparse, glob, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import exh_expand_build as XB   # noqa: E402  (patched 5m / funding sources)
import nospot_sleeve_lib as L   # noqa: E402
TB = XB.TB
D = os.path.join(R, "reports", "NOSPOT_EXIT_data")
K1 = [os.path.join(R, "reports", "EXH_EXIT_GRID_data", "k1m"), os.path.join(D, "k1m")]
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
MAXH = 48


def cells():
    """prereg family: name, fam, sl %, tp % (nan = none; 'atr' handled for REF), mech (none/be/lock/trail/trailatr), arm, par, hold h."""
    C = [dict(name="REF_XE", fam="REF", sl=np.nan, tp=np.nan, mech="none", arm=np.inf, par=0.0, hold=24)]
    for sl in (6, 10, 12):
        for tp in (2, 3, 4, 5):
            C.append(dict(name=f"A_TP{tp}_SL{sl}", fam="A", sl=sl, tp=tp, mech="none", arm=np.inf, par=0.0, hold=24))
    for a in (2, 3, 4):
        C.append(dict(name=f"B_BE{a}", fam="B", sl=10, tp=10, mech="be", arm=a, par=0.0, hold=24))
    for a, k in ((3, 1), (4, 2), (5, 3)):
        C.append(dict(name=f"C_L{a}{k}", fam="C", sl=10, tp=10, mech="lock", arm=a, par=k, hold=24))
    for a in (2, 3, 4):
        for g in (1.5, 3):
            C.append(dict(name=f"D_T{a}_G{g:g}", fam="D", sl=10, tp=np.nan, mech="trail", arm=a, par=g, hold=24))
    for m in (1, 2):
        C.append(dict(name=f"D_T3_A{m}", fam="D", sl=10, tp=np.nan, mech="trailatr", arm=3, par=m, hold=24))
    C.append(dict(name="X_TP10_SL10", fam="X", sl=10, tp=10, mech="none", arm=np.inf, par=0.0, hold=24))  # base of B / C (not selectable)
    for tp in (1.0, 1.5):          # ADDENDUM S (22:24 UTC)
        for sl in (2, 3, 4, 6):
            for h in (4, 24):
                C.append(dict(name=f"S_TP{tp:g}_SL{sl}_{h}h", fam="S", sl=sl, tp=tp, mech="none", arm=np.inf, par=0.0, hold=h))
    for h in (4, 12, 24, 48):
        C.append(dict(name=f"T_{h}h", fam="T", sl=10, tp=np.nan, mech="none", arm=np.inf, par=0.0, hold=h))
    return C


def base_of(c):
    """the same cell with its mechanism switched off (for the Δ = 0 check)."""
    d = dict(c); d["mech"] = "none"; d["arm"] = np.inf; d["name"] = c["name"] + "_base"; return d


def load1m(p, days):
    xs = []; ok = set()
    for d in days:
        ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d")
        f = os.path.join(K1[0], p, ds + ".csv")
        if os.path.exists(f) and os.path.getsize(f) > 20:
            x = pd.read_csv(f)
            if len(x):
                xs.append(x[["t", "o", "h", "l", "c"]]); ok.add(d); continue
        f = os.path.join(K1[1], p, ds + ".npz")
        if os.path.exists(f):
            z = np.load(f); xs.append(pd.DataFrame({k: z[k] for k in ("t", "o", "h", "l", "c")})); ok.add(d)
    if not xs:
        return None, ok
    x = pd.concat(xs).drop_duplicates("t").sort_values("t")
    t = x.t.values.astype("int64"); t = np.where(t > 10**14, t // 1000, t)
    return (t, x.o.values.astype(float), x.h.values.astype(float), x.l.values.astype(float), x.c.values.astype(float)), ok


def tick_of(*a):
    u = np.unique(np.concatenate(a)); d = np.diff(u); d = d[d > 1e-15]
    return float(d.min()) if len(d) else 0.0


def sim_trade(T, t, o, h, l, c, atrp, slip, tick, CC, nb):
    """one trade (bars from the entry bar on, ≤ 48 h + 2 h) × all cells (vectorised). Returns per-cell r, k, xm, worked, rec, best, reach."""
    E0 = o[0]; n = len(o); nC = len(CC["sl"])
    end_t = T + CC["hold"] * H1
    H = np.searchsorted(t, end_t, "left"); H = np.maximum(np.minimum(H, n), 1)
    Hmax = int(H.max())
    tt, oo, hh, ll, cc = t[:Hmax], o[:Hmax], h[:Hmax], l[:Hmax], c[:Hmax]
    bp = np.minimum(E0, np.r_[E0, np.minimum.accumulate(ll)[:-1]])          # best low BEFORE each bar
    favbp = (1 - bp / E0) * 100
    sl = np.where(CC["ref"], np.minimum(1.5 * atrp, 6.0), CC["sl"])
    tp = np.where(CC["ref"], np.minimum(20 * atrp, 10.0), CC["tp"])
    S0 = E0 * (1 + sl / 100)
    lvl_be = E0 * (1 - (0.10 + 2 * slip) / 100)
    base = np.where(CC["mech"] == 1, lvl_be, np.where(CC["mech"] == 2, E0 * (1 - CC["par"] / 100), 0.0))
    mult = np.where(CC["mech"] == 3, 1 + CC["par"] / 100, np.where(CC["mech"] == 4, 1 + CC["par"] * atrp / 100, 0.0))
    armed = favbp[None, :] >= CC["arm"][:, None]
    dyn = np.where(armed, base[:, None] + mult[:, None] * bp[None, :], np.inf)
    stop = np.minimum(S0[:, None], dyn)
    valid = np.arange(Hmax)[None, :] < H[:, None]
    hs = (hh[None, :] >= stop) & valid
    TPp = E0 * (1 - tp / 100)
    with np.errstate(invalid="ignore"):
        ht = (ll[None, :] <= (TPp - tick)[:, None]) & valid & np.isfinite(TPp)[:, None]
    BIG = 10**9
    fs = np.where(hs.any(1), hs.argmax(1), BIG); ft = np.where(ht.any(1), ht.argmax(1), BIG)
    rows = np.arange(nC)
    isstop = (fs < BIG) & (fs <= ft); istp = (~isstop) & (ft < BIG)
    fsi = np.minimum(fs, Hmax - 1); last = H - 1
    stp_lvl = stop[rows, fsi]
    X = np.where(isstop, np.maximum(stp_lvl, oo[fsi]), np.where(istp, TPp, cc[last]))
    col = np.where(isstop, fs, np.where(istp, ft, last))
    k = np.where(isstop, np.where(stp_lvl < S0 - 1e-12, 3, 1), np.where(istp, 0, 2)).astype(np.int8)
    r = -(X / E0 - 1) * 100
    xm = tt[np.minimum(col, Hmax - 1)] + nb * MIN
    # path facts: WORKED = +1 % before the initial stop distance (same bar → not worked), within the hold
    fav = (1 - l / E0) * 100; adv = (h / E0 - 1) * 100
    f1 = np.argmax(fav >= 1.0) if (fav >= 1.0).any() else BIG
    worked = np.zeros(nC, np.int8); rec = np.zeros(nC, np.int8); best = np.zeros(nC, np.float32); reach = np.zeros(nC, np.float32)
    cmf = np.maximum.accumulate(fav)
    after = int(120 // nb); a45 = int(45 // nb)
    for j in range(nC):
        aj = np.argmax(adv[:H[j]] >= sl[j]) if (adv[:H[j]] >= sl[j]).any() else BIG
        worked[j] = (f1 < aj) and (f1 < H[j])
        cj = int(col[j])
        reach[j] = cmf[min(cj, n - 1, aj if aj < BIG else n - 1)]
        best[j] = cmf[min(cj + a45, n - 1)]
        if k[j] in (1, 3):
            seg = fav[cj + 1: cj + 1 + after]
            rec[j] = len(seg) > 0 and seg.max() >= 1.0
    return r.astype(np.float32), k, xm.astype(np.int64), worked, rec, best, reach


def cell_arrays(CL):
    mm = {"none": 0, "be": 1, "lock": 2, "trail": 3, "trailatr": 4}
    return dict(ref=np.array([c["fam"] == "REF" for c in CL]), sl=np.array([c["sl"] for c in CL], float),
                tp=np.array([c["tp"] for c in CL], float), mech=np.array([mm[c["mech"]] for c in CL]),
                arm=np.array([c["arm"] for c in CL], float), par=np.array([c["par"] for c in CL], float),
                hold=np.array([c["hold"] for c in CL], np.int64))


def run_rows(p, rows, CL, k5=None, fund=None, keep_path=None, one1m=None):
    """rows: DataFrame [xi, entry_ms, atrp, slip]; returns dict of arrays. one1m = (t, o, h, l, c) overrides the 1m cache (replays)."""
    CC = cell_arrays(CL); nC = len(CL)
    days = set()
    for T in rows.entry_ms.values:
        for kk in range(3):
            days.add(int(T) // DAY + kk)
    if one1m is not None:
        one, okd = one1m, set(range(int(one1m[0][0]) // DAY, int(one1m[0][-1]) // DAY + 3))
    else:
        one, okd = load1m(p, sorted(days))
    if one is not None:
        t1, o1, h1, l1, c1 = one; tick1 = tick_of(o1, h1, l1, c1)
    five = None
    ft, fc, months = fund if fund is not None else TB.funding_for(p)
    ft = np.asarray(ft, "int64"); fc = np.asarray(fc, float)
    n = len(rows)
    out = dict(xi=rows.xi.values.astype(np.int64), r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8),
               xm=np.zeros((n, nC), np.int64), f=np.zeros((n, nC), np.float32), worked=np.zeros((n, nC), np.int8),
               rec=np.zeros((n, nC), np.int8), best=np.zeros((n, nC), np.float32), reach=np.zeros((n, nC), np.float32),
               src1m=np.zeros(n, bool), ok=np.zeros(n, bool))
    paths = {}
    for i, (T, atrp, slip) in enumerate(zip(rows.entry_ms.values.astype(np.int64), rows.atrp.values.astype(float), rows.slip.values.astype(float))):
        need = {int(T) // DAY + kk for kk in range(3)}
        use1 = one is not None and need <= okd
        if use1:
            a = int(np.searchsorted(t1, T)); use1 = a < len(t1) and t1[a] == T
        if use1:
            b = int(np.searchsorted(t1, T + (MAXH + 2) * H1))
            t, o, h, l, c, tick, nb = t1[a:b], o1[a:b], h1[a:b], l1[a:b], c1[a:b], tick1, 1
        else:
            if five is None:
                five = k5 if k5 is not None else TB.load(p)
                if five is not None:
                    tick5 = tick_of(five[1], five[2], five[3], five[4])
            if five is None:
                continue
            t5 = five[0]; a = int(np.searchsorted(t5, T))
            if a >= len(t5) or t5[a] != T:
                continue
            b = int(np.searchsorted(t5, T + (MAXH + 2) * H1))
            t, o, h, l, c, tick, nb = t5[a:b], five[1][a:b], five[2][a:b], five[3][a:b], five[4][a:b], tick5, 5
        if len(o) < 2:
            continue
        r, k, xm, wk, rc, be, rh = sim_trade(T, t, o, h, l, c, atrp, slip, tick, CC, nb)
        if len(ft):
            fs = (fc[np.searchsorted(ft, xm, "right")] - fc[np.searchsorted(ft, T, "right")]) * 100
        else:
            fs = np.zeros(nC)
        me = pd.Timestamp(int(T), unit="ms").strftime("%Y-%m")
        mx = [pd.Timestamp(int(x_), unit="ms").strftime("%Y-%m") for x_ in xm]
        miss = np.array([(me not in months) or (m_ not in months) for m_ in mx])
        fs = np.where(miss, -(xm - T) / (8 * H1) * TB.FUND_MISS, fs)
        out["r"][i] = r; out["k"][i] = k; out["xm"][i] = xm; out["f"][i] = fs; out["worked"][i] = wk; out["rec"][i] = rc
        out["best"][i] = be; out["reach"][i] = rh; out["src1m"][i] = use1; out["ok"][i] = True
        if keep_path is not None and keep_path[i]:
            E0 = o[0]
            ts_ = np.arange(0, MAXH * 12 + 1) * 300_000 + T
            jj = np.searchsorted(t, ts_, "right") - 1
            # 5-min sampled close-based path + running low / high (for anchored charts)
            cpath = np.where(jj < len(t), -(c[np.clip(jj, 0, len(c) - 1)] / E0 - 1) * 100, np.nan)
            cpath[ts_ > t[-1] + nb * MIN] = np.nan
            paths[int(rows.xi.values[i])] = cpath.astype(np.float16)
    return out, paths


def work(args):
    p, rows, CL, outdir = args
    fo = os.path.join(outdir, "sim", p + ".npz")
    if os.path.exists(fo):
        return p, -1
    out, paths = run_rows(p, rows, CL, keep_path=rows.cohort.values)
    np.savez_compressed(fo, **out)
    if paths:
        np.savez_compressed(os.path.join(outdir, "path", p + ".npz"), xi=np.array(list(paths)), p=np.vstack(list(paths.values())))
    return p, int(out["ok"].sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=""); ap.add_argument("--procs", type=int, default=7)
    A = ap.parse_args()
    os.makedirs(os.path.join(D, "sim"), exist_ok=True); os.makedirs(os.path.join(D, "path"), exist_ok=True)
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    M = L.rule_masks(X)
    coh = L.ns0_mask(X) | M["R1_OI_UNWIND"][0] | M["R4_TOP_DISTRIB"][0]
    inep = np.isin(X.epi.values, np.unique(X.epi.values[coh]))
    S = pd.DataFrame(dict(xi=np.nonzero(inep)[0], pair=X.pair.values[inep], entry_ms=X.entry_ms.values[inep], atrp=X.atrp.values[inep],
                          slip=X.slip.values[inep], cohort=coh[inep]))
    CL = cells()
    pairs = sorted(S.pair.unique()) if not A.pairs else A.pairs.split(",")
    jobs = [(p, S[S.pair == p], CL, D) for p in pairs]
    jobs.sort(key=lambda j: -len(j[1]))
    print("pairs", len(jobs), "rows", len(S), "cells", len(CL), flush=True)
    t0 = time.time(); tot = 0
    with mp.get_context("fork").Pool(A.procs) as pool:
        for i, (p, n_) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            tot += max(n_, 0)
            if i % 10 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p, n_, tot, f"{time.time() - t0:.0f}s", flush=True)
    print("done", tot, flush=True)


if __name__ == "__main__":
    main()
