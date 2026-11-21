#!/usr/bin/env python3
"""DIST_PEAKSTOP (PREREG reports/DIST_PEAKSTOP_2026-11-21_PREREG.md) — SIM, BACKTEST, research only.
Pump-top stop variants for DISTRIBUTION_SHORT on the 1,547 DIST_EXIT REF_B fills, using the DIST_THEORY_EXIT engine (sim_one, family W with a
per-fill stop %). Landmarks PK / DH / E0 come from reports/DIST_THEORY_EXIT_2026-11-21_data/sim_BT.npz (same engine, same bars).
Output: reports/DIST_PEAKSTOP_2026-11-21_data/sim.npz
Usage: venv/bin/python -I scripts/DIST_PEAKSTOP_2026-11-21_sim.py [--procs 6]"""
import argparse, os, sys, time, importlib.util
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)


def _load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


SM = _load("DTE_SIM", "DIST_THEORY_EXIT_2026-11-21_sim.py")
DS, NS, TB = SM.DS, SM.NS, SM.TB
D0 = os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21_data")
D = os.path.join(R, "reports", "DIST_PEAKSTOP_2026-11-21_data")
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
CAPS = (8, 10, 12, 15)
NPERM = 5
SPLIT = pd.Timestamp("2025-07-01").value // 10**6


def cell_names():
    n = ["REF"] + [f"P{c}" for c in CAPS] + [f"Q{c}" for c in CAPS] + [f"W{c}" for c in CAPS]
    n += [f"S{c}_{k}" for c in CAPS for k in range(1, NPERM + 1)] + ["DQ10", "DQ15"]
    return n


def stop_matrix(PK, DH, E0, atrp, T):
    """per-fill stop % above E0 for every cell (NaN for REF = today's)."""
    names = cell_names(); n = len(E0)
    S = np.full((n, len(names)), np.nan)
    s0 = np.minimum(1.5 * atrp, 6.0)
    top = (PK * 1.01 / E0 - 1) * 100          # pump top + 1 % buffer, % above entry
    dh = (DH * 1.01 / E0 - 1) * 100
    okp = np.isfinite(top); okd = np.isfinite(dh)
    blk = T < SPLIT
    rng = np.random.default_rng(20261121)
    perms = []
    for k in range(NPERM):
        d = np.full(n, np.nan)
        for b in (True, False):
            idx = np.nonzero((blk == b) & okp)[0]
            src = idx.copy()
            while True:                              # derangement-ish: no fill keeps its own top
                pr = rng.permutation(src)
                if not (pr == idx).any() or len(idx) < 2:
                    break
            d[idx] = top[pr]
        perms.append(d)
    for j, nm in enumerate(names):
        if nm == "REF":
            continue
        if nm.startswith("DQ"):
            cap = float(nm[2:]); S[:, j] = np.where(okd, np.maximum(s0, np.minimum(dh, cap)), s0); continue
        fam = nm[0]; cap = float(nm[1:].split("_")[0])
        if fam == "P":
            S[:, j] = np.where(okp & (PK >= E0 * 1.01), np.minimum(top, cap), s0)
        elif fam == "Q":
            S[:, j] = np.where(okp, np.maximum(s0, np.minimum(top, cap)), s0)
        elif fam == "W":
            S[:, j] = cap
        elif fam == "S":
            d = perms[int(nm.split("_")[1]) - 1]
            S[:, j] = np.where(okp, np.maximum(s0, np.minimum(d, cap)), s0)
    return names, S


def work(args):
    p, rows, ST = args
    nC = ST.shape[1]; n = len(rows)
    out = dict(ix=rows.index.values.astype(np.int64), r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8),
               xm=np.zeros((n, nC), np.int64), f=np.zeros((n, nC), np.float32), slp=np.full((n, nC), np.nan, np.float32), ok=np.zeros(n, bool),
               src1m=np.zeros(n, bool), E0=np.full(n, np.nan))
    days = set()
    for T in rows.entry_ms.values:
        for kk in range(3):
            days.add(int(T) // DAY + kk)
    one, okd = DS.load1m(p, sorted(days))
    if one is not None:
        t1, o1, h1, l1, c1 = one; tick1 = NS.tick_of(o1, h1, l1, c1)
    five = TB.load(p)
    tick5 = NS.tick_of(five[1], five[2], five[3], five[4]) if five is not None else None
    ft, fc, months = TB.funding_for(p)
    ft = np.asarray(ft, "int64"); fc = np.asarray(fc, float)
    last_day = pd.Timestamp("2026-11-18").value // 10**6 // DAY
    for i, (T, atrp) in enumerate(zip(rows.entry_ms.values.astype(np.int64), rows.atrp.values.astype(float))):
        need = {int(T) // DAY + kk for kk in range(3)}
        use1 = one is not None and need <= okd
        if one is not None and not use1:
            use1 = {d for d in need if d < last_day} <= okd
        if use1:
            a = int(np.searchsorted(t1, T)); use1 = a < len(t1) and t1[a] == T
        if use1:
            b = int(np.searchsorted(t1, T + (NS.MAXH + 2) * H1))
            t, o, h, l, c, tick, nb = t1[a:b], o1[a:b], h1[a:b], l1[a:b], c1[a:b], tick1, 1
        else:
            if five is None:
                continue
            t5 = five[0]; a = int(np.searchsorted(t5, T))
            if a >= len(t5) or t5[a] != T:
                continue
            b = int(np.searchsorted(t5, T + (NS.MAXH + 2) * H1))
            t, o, h, l, c, tick, nb = t5[a:b], five[1][a:b], five[2][a:b], five[3][a:b], five[4][a:b], tick5, 5
        if len(o) < 2:
            continue
        st = ST[i]
        CL = [dict(name="REF", fam="REF")] + [dict(name=str(j), fam="W", w=float(st[j])) for j in range(1, nC)]
        res = SM.sim_one(t, o, h, l, c, int(T), atrp, tick, None, None, ft, fc, CL, nb)
        xm = np.array([x[2] for x in res], np.int64)
        fs = (fc[np.searchsorted(ft, xm, "right")] - fc[np.searchsorted(ft, T, "right")]) * 100 if len(ft) else np.zeros(nC)
        me = pd.Timestamp(int(T), unit="ms").strftime("%Y-%m")
        miss = np.array([(me not in months) or (pd.Timestamp(int(x_), unit="ms").strftime("%Y-%m") not in months) for x_ in xm])
        fs = np.where(miss, -(xm - T) / (8 * H1) * TB.FUND_MISS, fs)
        for j, (r, k, x, s) in enumerate(res):
            out["r"][i, j] = r; out["k"][i, j] = k; out["xm"][i, j] = x; out["slp"][i, j] = s
        out["f"][i] = fs; out["ok"][i] = True; out["src1m"][i] = use1; out["E0"][i] = o[0]
    return p, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); A = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    S = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    Z = np.load(os.path.join(D0, "sim_BT.npz"))
    assert (Z["ix"] == np.arange(len(S))).all()
    T = S.entry_ms.values.astype(np.int64); atrp = S.atr15_pct.values.astype(float)
    names, ST = stop_matrix(Z["PK"], Z["DH"], Z["E0"], atrp, T)
    F = pd.DataFrame(dict(pair=S.pair.astype(str).values, entry_ms=T, atrp=atrp))
    jobs = [(p, g, ST[g.index.values]) for p, g in F.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
    res = []; t0 = time.time()
    with mp.get_context("fork").Pool(A.procs) as pool:
        for i, (p, out) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            res.append(out)
            if i % 100 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p, f"{time.time() - t0:.0f}s", flush=True)
    Y = {k: np.concatenate([r_[k] for r_ in res]) for k in res[0]}
    o = np.argsort(Y["ix"]); Y = {k: v[o] for k, v in Y.items()}
    np.savez_compressed(os.path.join(D, "sim.npz"), names=np.array(names), stops=ST, PK=Z["PK"], DH=Z["DH"], E0_bt=Z["E0"], **Y)
    print("done", len(Y["ix"]), "ok", int(Y["ok"].sum()), "1m", int(Y["src1m"].sum()), flush=True)


if __name__ == "__main__":
    main()
