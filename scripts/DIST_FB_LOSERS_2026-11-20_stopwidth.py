#!/usr/bin/env python3
"""DIST_FB_LOSERS_2026-11-20 — stop-width counterfactual, read on the STOPPED cohort, two-sided (CLAUDE.md exit/stop rule).

DISTRIBUTION_SHORT: the 1,547 DL 586 backtest fills (reports/ENTRY_WL_2026-11-18_data/dist.pkl = DIST_EXIT REF_B cohort, one position
per pair under REF) re-walked on their 1m paths with the DIST_EXIT engine (scripts/dist_exit_sim.py run_rows, same fill rules, same
real funding, same slip model and cost formula as dist_exit_eval.net_mat) under stop = min(k × ATR15, cap):
    k / cap ∈ {1.0 / 4, 1.25 / 5, 1.5 / 6 (= Exit B, REF), 2.0 / 8, 2.5 / 10, 3.0 / 12}; TP = min(20 × ATR15, 10 %) maker; 24 h.
DIST_EXIT (2026-11-18) never tested a WIDER stop than Exit B with Exit B's TP — this fills that gap. Tighter ATR stops are shown for
symmetry only (fixed 2–4 % stops were refuted there).
Fills that never touched the REF stop must show Δ = 0 (asserted). Stopped fills split: SAVED (the wider stop survives and books better)
vs DEEPER (stops anyway, bigger loss) vs OTHER. Day-block (10-day) bootstrap CI on the summed Δ, per block (DESIGN < 2025-07 ≤ HOLDOUT).
Simplification (stated): the entry set is held at REF's one-position cohort; a longer hold could block a later signal on the same pair.

FLATBASE_SHORT: the OOS_REAL_FLATBASE rebuild (reports/OOS_REAL_FLATBASE_data/wl_features.csv, 923 trades) already prices
stop +15 (X3) and +7 (X4) beside today's +10 (X0) on 5m paths with real funding; kept = today's NEAR_BASE (dist_base > 2.09). Same split.
Output: reports/DIST_FB_LOSERS_2026-11-20_data/stopwidth_*.csv + printed tables.
Usage: venv/bin/python -I scripts/DIST_FB_LOSERS_2026-11-20_stopwidth.py [--procs 4]
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

OUT = os.path.join(ROOT, "reports", "DIST_FB_LOSERS_2026-11-20_data")
VARIANTS = [(1.0, 4.0), (1.25, 5.0), (1.5, 6.0), (2.0, 8.0), (2.5, 10.0), (3.0, 12.0)]
REF_I = 2
H1 = 3_600_000
BIG = DE.BIG
MIN = 60_000


def sim_trade_k(T, t, o, h, l, c, atrp, slip, tick, CC, nb):
    """= dist_exit_sim.sim_trade, with stop = min(slk × ATR, slcap) per cell (CC['slk'] / CC['slcap'])."""
    E0 = o[0]; n = len(o); nC = len(CC["sl"])
    end_t = T + CC["hold"] * H1
    H = np.searchsorted(t, end_t, "left"); H = np.maximum(np.minimum(H, n), 1)
    Hmax = int(H.max())
    oo, hh, ll, cc = o[:Hmax], h[:Hmax], l[:Hmax], c[:Hmax]
    tt = t[:Hmax]
    sl = np.minimum(CC["slk"] * atrp, CC["slcap"])
    tp = np.where(CC["tpatr"], min(20 * atrp, 10.0), CC["tp"])
    S0 = E0 * (1 + sl / 100)
    stop = np.repeat(S0[:, None], Hmax, 1)
    valid = np.arange(Hmax)[None, :] < H[:, None]
    hs = (hh[None, :] >= stop) & valid
    TPp = E0 * (1 - tp / 100)
    ht = (ll[None, :] <= (TPp - tick)[:, None]) & valid
    fs = np.where(hs.any(1), hs.argmax(1), BIG); ft = np.where(ht.any(1), ht.argmax(1), BIG)
    rows = np.arange(nC)
    isstop = (fs < BIG) & (fs <= ft); istp = (~isstop) & (ft < BIG)
    fsi = np.minimum(fs, Hmax - 1); last = H - 1
    X = np.where(isstop, np.maximum(stop[rows, fsi], oo[fsi]), np.where(istp, TPp, cc[last]))
    col = np.where(isstop, fs, np.where(istp, ft, last))
    k = np.where(isstop, 1, np.where(istp, 0, 2)).astype(np.int8)
    r = -(X / E0 - 1) * 100
    xm = tt[np.minimum(col, Hmax - 1)] + nb * MIN
    cens = (k == 2) & (tt[np.minimum(last, Hmax - 1)] + nb * MIN < end_t - nb * MIN)
    k = np.where(cens, 9, k).astype(np.int8)
    z = np.zeros(nC, np.int8); zf = np.zeros(nC, np.float32)
    return r.astype(np.float32), k, xm.astype(np.int64), z, z, zf, zf, z, sl.astype(np.float32), tp.astype(np.float32)


def cells():
    return [dict(name=f"SL{kk:g}x_cap{cap:g}", fam="W", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="none", arm=np.inf, par=0.0, hold=24,
                 slk=kk, slcap=cap) for kk, cap in VARIANTS]


def cell_arrays(CL):
    C = DE.__dict__["_orig_cell_arrays"](CL)
    C["slk"] = np.array([c["slk"] for c in CL], float)
    C["slcap"] = np.array([c["slcap"] for c in CL], float)
    return C


def work(a):
    p, rows, CL = a
    out, _ = DE.run_rows(p, rows, CL, keep_path=None)
    return p, rows.index.values, out


def boot_sum(v, day, B=2000, block=10, seed=7):
    """day-block bootstrap CI of the SUM of v (blocks = 10-day buckets)."""
    rng = np.random.default_rng(seed)
    day = np.asarray(day)
    dn = day.astype(np.int64) if np.issubdtype(day.dtype, np.integer) else pd.to_datetime(day).values.astype("datetime64[D]").astype(np.int64)
    b = dn // block                                      # dist.pkl stores the day as an integer day number
    ub = np.unique(b)
    sums = np.array([v[b == u].sum() for u in ub])
    bs = rng.choice(len(ub), (B, len(ub)))
    s = sums[bs].sum(1)
    return np.percentile(s, 2.5), np.percentile(s, 97.5), float((s > 0).mean())


def dist_part(procs):
    DE.__dict__["_orig_cell_arrays"] = DE.cell_arrays
    DE.cell_arrays = cell_arrays
    DE.sim_trade = sim_trade_k
    S = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    F = pd.DataFrame(dict(pair=S.pair.astype(str).values, entry_ms=S.entry_ms.values.astype(np.int64), atrp=S.atr15_pct.values.astype(float),
                          slip=S.slip.values.astype(float), keep=False))
    CL = cells(); nC = len(CL); n = len(F)
    res = dict(r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8), f=np.zeros((n, nC), np.float32),
               ok=np.zeros(n, bool), slp=np.zeros((n, nC), np.float32))
    jobs = [(p, F[F.pair == p], CL) for p in F.pair.unique()]
    with mp.get_context("fork").Pool(procs) as pool:
        for p, ix, out in pool.imap_unordered(work, jobs, chunksize=4):
            for kk in res:
                res[kk][ix] = out[kk]
    k = res["k"]; tk = k != 0
    s = F.slip.values[:, None] * np.ones((1, nC))
    net = res["r"].astype(float) - s - np.where(tk, s, 0.0) - 0.05 - np.where(tk, 0.05, 0.02) + res["f"].astype(float)
    net = np.where((~res["ok"][:, None]) | (k == 9) | (k < 0), np.nan, net)
    parity = np.nanmax(np.abs(net[:, REF_I] - S.net.values))
    print(f"DIST parity: REF re-walk vs DIST_EXIT REF_B net, max |d| = {parity:.4f} % over {np.isfinite(net[:, REF_I]).sum()} fills")
    ref = net[:, REF_I]; refstop = k[:, REF_I] == 1
    rows = []
    for j, (kk, cap) in enumerate(VARIANTS):
        d = net[:, j] - ref
        ok = np.isfinite(d)
        if j > REF_I:   # wider: a fill REF never stopped must be unchanged
            nz = np.abs(d[ok & ~refstop]) > 1e-6
            assert nz.sum() == 0, f"{nz.sum()} never-stopped fills changed under a wider stop"
        for blk in ("DESIGN", "HOLDOUT", "ALL"):
            m = ok & ((S.block.values == blk) if blk != "ALL" else True)
            st = m & refstop
            saved = st & (d > 0); deeper = st & (d < 0)
            lo, hi, pp = boot_sum(d[m], S.day.values[m])
            rows.append(dict(sleeve="DIST", variant=f"min({kk:g}×ATR, {cap:g} %)", block=blk, N=int(m.sum()), avg_net=np.nanmean(net[m, j]),
                             WR=float((net[m, j] > 0).mean() * 100), ref_avg=np.nanmean(ref[m]), d_avg=np.nanmean(d[m]), d_sum=np.nansum(d[m]),
                             ci_lo=lo, ci_hi=hi, p_gain=pp, ref_stopped=int(st.sum()), saved_n=int(saved.sum()), saved_sum=float(d[saved].sum()),
                             deeper_n=int(deeper.sum()), deeper_sum=float(d[deeper].sum())))
    T = pd.DataFrame(rows)
    # per half-year for the wider variants
    hy = []
    for j, (kk, cap) in enumerate(VARIANTS):
        if j == REF_I:
            continue
        d = net[:, j] - ref
        for h_ in sorted(S.half.unique()):
            m = np.isfinite(d) & (S.half.values == h_)
            hy.append(dict(variant=f"{kk:g}x/{cap:g}", half=h_, N=int(m.sum()), d_sum=float(np.nansum(d[m]))))
    HY = pd.DataFrame(hy).pivot(index="variant", columns="half", values="d_sum")
    pd.DataFrame(net, columns=[c["name"] for c in CL]).assign(pair=S.pair.values, entry_ms=S.entry_ms.values, block=S.block.values,
                                                               day=S.day.values).to_csv(os.path.join(OUT, "stopwidth_dist_rows.csv.gz"), index=False)
    return T, HY


def fb_part():
    X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09)].copy()
    X["day"] = pd.to_datetime(X.cf_entry_t, unit="ms").dt.strftime("%Y-%m-%d")
    X["blk"] = np.where(pd.to_datetime(X.cf_entry_t, unit="ms") < pd.Timestamp("2025-07-01"), "DESIGN", "HOLDOUT")
    rows = []
    refstop = X.X0_why.values == "STOP"
    for nm, col in (("stop +7 (X4)", "X4_net"), ("stop +15 (X3)", "X3_net")):
        d = (X[col] - X.X0_net).values
        ok = np.isfinite(d)
        for blk in ("DESIGN", "HOLDOUT", "ALL"):
            m = ok & ((X.blk.values == blk) if blk != "ALL" else True)
            st = m & refstop
            lo, hi, pp = boot_sum(d[m], X.day.values[m])
            rows.append(dict(sleeve="FLATBASE", variant=nm, block=blk, N=int(m.sum()), avg_net=X[col].values[m].mean(),
                             WR=float((X[col].values[m] > 0).mean() * 100), ref_avg=X.X0_net.values[m].mean(), d_avg=d[m].mean(), d_sum=d[m].sum(),
                             ci_lo=lo, ci_hi=hi, p_gain=pp, ref_stopped=int(st.sum()), saved_n=int((st & (d > 0)).sum()),
                             saved_sum=float(d[st & (d > 0)].sum()), deeper_n=int((st & (d < 0)).sum()), deeper_sum=float(d[st & (d < 0)].sum()),
                             unstopped_changed=int((m & ~refstop & (np.abs(d) > 1e-6)).sum()),
                             unstopped_d_sum=float(d[m & ~refstop].sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4)
    A = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    T, HY = dist_part(A.procs)
    T.to_csv(os.path.join(OUT, "stopwidth_dist.csv"), index=False)
    HY.to_csv(os.path.join(OUT, "stopwidth_dist_halves.csv"))
    print(T.round(2).to_string(index=False)); print(HY.round(1).to_string())
    F = fb_part()
    F.to_csv(os.path.join(OUT, "stopwidth_fb.csv"), index=False)
    print(F.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
