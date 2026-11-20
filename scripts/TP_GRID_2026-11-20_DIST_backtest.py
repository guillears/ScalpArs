#!/usr/bin/env python3
"""TP_GRID_2026-11-20_DIST — STEP 2: the 1,547-fill DL 586 DISTRIBUTION_SHORT backtest cohort (reports/ENTRY_WL_2026-11-18_data/dist.pkl,
= DIST_EXIT REF_B kept fills, one position per pair) re-walked on its 1m paths with the DIST_EXIT engine (scripts/dist_exit_sim.py run_rows:
same fill rules, real funding, slip model, cost formula as dist_exit_eval.net_mat / the stop-width study) under
    TP = min(20 x ATR15, X) % resting maker, X in {4, 5, 6, 7, 8, 9, 10 = today (Exit B)}; stop = min(1.5 x ATR15, 6 %) (today); 24 h.
Two TP-fill rules: '1tick' (engine: the bar LOW trades >= 1 tick through) and '0.05%' (operator spec: >= 0.05 % through; stricter).
Parity: X = 10 / 1tick must reproduce dist.pkl net (Exit B booked) fill by fill.
Simplification (stated): the entry set is held at Exit B's one-position cohort (a faster TP frees the pair earlier -> could admit a later signal).
Output: reports/TP_GRID_2026-11-20_DIST_data/bt_rows.csv.gz (net per fill per cell + exit day) -> read by TP_GRID_2026-11-20_DIST_report.py.
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_DIST_backtest.py [--procs 6]
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import dist_exit_sim as DE  # noqa: E402

OUT = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_DIST_data")
XS = [4, 5, 6, 7, 8, 9, 10]
RULES = [("1tick", 0.0), ("0.05%", 0.05)]
H1, MIN, BIG = 3_600_000, 60_000, DE.BIG


def sim_trade_tp(T, t, o, h, l, c, atrp, slip, tick, CC, nb):
    """= dist_exit_sim.sim_trade for mech 'none' (Exit B family), with TP = min(20 x ATR, CC['tpx']) and a per-cell trade-through."""
    E0 = o[0]; n = len(o); nC = len(CC["tpx"])
    end_t = T + 24 * H1
    H = max(min(int(np.searchsorted(t, end_t, "left")), n), 1)
    tt, oo, hh, ll, cc = t[:H], o[:H], h[:H], l[:H], c[:H]
    sl = min(1.5 * atrp, 6.0)
    tp = np.minimum(20 * atrp, CC["tpx"])
    S0 = E0 * (1 + sl / 100)
    TPp = E0 * (1 - tp / 100)
    lim = np.where(CC["thr"] > 0, TPp * (1 - CC["thr"] / 100), TPp - tick)
    hs = hh >= S0
    fs = int(hs.argmax()) if hs.any() else BIG
    ht = ll[None, :] <= lim[:, None]
    ft = np.where(ht.any(1), ht.argmax(1), BIG)
    isstop = (fs < BIG) & (fs <= ft); istp = (~isstop) & (ft < BIG)
    fsi = min(fs, H - 1); last = H - 1
    X = np.where(isstop, max(S0, oo[fsi]), np.where(istp, TPp, cc[last]))
    col = np.where(isstop, fs, np.where(istp, ft, last))
    k = np.where(isstop, 1, np.where(istp, 0, 2)).astype(np.int8)
    r = -(X / E0 - 1) * 100
    xm = tt[np.minimum(col, H - 1)] + nb * MIN
    cens = (k == 2) & (tt[last] + nb * MIN < end_t - nb * MIN)
    k = np.where(cens, 9, k).astype(np.int8)
    z = np.zeros(nC, np.int8); zf = np.zeros(nC, np.float32)
    return r.astype(np.float32), k, xm.astype(np.int64), z, z, zf, zf, z, np.full(nC, sl, np.float32), tp.astype(np.float32)


def cells():
    return [dict(name=f"TP{x}_{rn}", tpx=float(x), thr=thr) for rn, thr in RULES for x in XS]


def cell_arrays(CL):
    return dict(tpx=np.array([c["tpx"] for c in CL], float), thr=np.array([c["thr"] for c in CL], float),
                sl=np.zeros(len(CL)), hold=np.full(len(CL), 24, np.int64))


def work(a):
    p, rows, CL = a
    out, _ = DE.run_rows(p, rows, CL, keep_path=None)
    return p, rows.index.values, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6)
    A = ap.parse_args()
    DE.cell_arrays = cell_arrays
    DE.sim_trade = sim_trade_tp
    S = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl")).reset_index(drop=True)
    F = pd.DataFrame(dict(pair=S.pair.astype(str).values, entry_ms=S.entry_ms.values.astype(np.int64), atrp=S.atr15_pct.values.astype(float),
                          slip=S.slip.values.astype(float), keep=False))
    CL = cells(); nC = len(CL); n = len(F)
    res = dict(r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8), f=np.zeros((n, nC), np.float32),
               xm=np.zeros((n, nC), np.int64), ok=np.zeros(n, bool))
    jobs = [(p, F[F.pair == p], CL) for p in F.pair.unique()]
    with mp.get_context("fork").Pool(A.procs) as pool:
        for p, ix, out in pool.imap_unordered(work, jobs, chunksize=4):
            for kk in res:
                res[kk][ix] = out[kk]
    k = res["k"]; tk = k != 0
    s = F.slip.values[:, None] * np.ones((1, nC))
    net = res["r"].astype(float) - s - np.where(tk, s, 0.0) - 0.05 - np.where(tk, 0.05, 0.02) + res["f"].astype(float)
    net = np.where((~res["ok"][:, None]) | (k == 9) | (k < 0), np.nan, net)
    names = [c["name"] for c in CL]
    ref = names.index("TP10_1tick")
    dpar = np.abs(net[:, ref] - S.net.values)
    print(f"PARITY X=10/1tick vs dist.pkl Exit B net: max |d| = {np.nanmax(dpar):.4f} %, mean {np.nanmean(dpar):.5f}, "
          f"N ok {np.isfinite(net[:, ref]).sum()} of {n}; kind match {(k[:, ref] == S.k.values).mean() * 100:.2f} %")
    D = pd.DataFrame(net, columns=names)
    for j, nm in enumerate(names):
        D["k_" + nm] = k[:, j]
        D["xday_" + nm] = res["xm"][:, j] // 86_400_000
    D = D.assign(pair=S.pair.values, entry_ms=S.entry_ms.values, block=S.block.values, half=S.half.values, day=S.day.values,
                 net_exitB=S.net.values, k_exitB=S.k.values, atrp=S.atr15_pct.values)
    os.makedirs(OUT, exist_ok=True)
    D.to_csv(os.path.join(OUT, "bt_rows.csv.gz"), index=False)
    print("saved", len(D))


if __name__ == "__main__":
    main()
