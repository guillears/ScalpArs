#!/usr/bin/env python3
"""EXH_EXIT2 (PREREG reports/EXH_EXIT2_2026-11-18_PREREG.md) — SIM. Reuses the NOSPOT_EXIT engine (scripts/nospot_exit_sim.run_rows:
1m path, funding, gap-aware stops, stop-first same bar, maker TP through 1 tick, dynamic levels from the next bar), extended only with
per-cell TP / SL modes (ATR = today's XE formulas, fixed, none) and the NP (no-progress) exit. Research only, BACKTEST.
Kinds: 0 maker TP · 1 initial stop · 2 time · 3 dynamic stop (BE / lock / trail) · 4 no-progress exit.
Output: reports/EXH_EXIT2_data/sim_<tag>.npz (r, k, xm, f, worked, rec, best, reach, src1m, ok, xi) + mfe24 / mae24 of the path.
Usage: venv/bin/python -I scripts/exh_exit2_sim.py EXH|RND|E [--procs 7]"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import nospot_exit_sim as S   # noqa: E402  (engine; also patches the 5m / funding sources via exh_expand_build)
TB = S.TB
G = os.path.join(R, "reports", "EXH_EXIT_GRID_data")
TB.K5M_SRC = [os.path.join(G, "E", "k5m")] + TB.K5M_SRC
TB.FUND_SRC = TB.FUND_SRC + [os.path.join(G, "E", "funding")]
D = os.path.join(R, "reports", "EXH_EXIT2_data")
K1_CSV = [os.path.join(G, "k1m"), os.path.join(D, "k1m")]
K1_NPZ = os.path.join(R, "reports", "NOSPOT_EXIT_data", "k1m")
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000


def cells():
    """sl: 'atr' or %, tp: 'atr' / % / None, mech none/be/lock/trail/np, arm, par, hold h."""
    C = [dict(name="REF", fam="REF", sl="atr", tp="atr", mech="none", arm=np.inf, par=0.0, hold=24)]
    for a in (2, 3, 4, 5, 6):
        C.append(dict(name=f"B_BE{a}", fam="B", sl="atr", tp="atr", mech="be", arm=a, par=0.0, hold=24))
    for a in (4, 6, 8):
        for g in (2, 3):
            C.append(dict(name=f"L_A{a}_L{a - g}", fam="L", sl="atr", tp="atr", mech="lock", arm=a, par=a - g, hold=24))
    for a in (6, 8):
        for g in (2, 3, 4):
            C.append(dict(name=f"D1_A{a}_G{g}", fam="D1", sl="atr", tp="atr", mech="trail", arm=a, par=g, hold=24))
    for a in (8, 10):
        for g in (3, 5):
            C.append(dict(name=f"D2_A{a}_G{g}", fam="D2", sl="atr", tp=None, mech="trail", arm=a, par=g, hold=24))
    for tp in (1, 1.5, 2, 3):
        for sl in (2, 3, 4):
            C.append(dict(name=f"S_TP{tp:g}_SL{sl}", fam="S", sl=sl, tp=tp, mech="none", arm=np.inf, par=0.0, hold=24))
    for h in (4, 8, 16):
        C.append(dict(name=f"T_{h}h", fam="T", sl="atr", tp="atr", mech="none", arm=np.inf, par=0.0, hold=h))
    for h in (2, 4, 8):
        C.append(dict(name=f"NP_{h}h", fam="NP", sl="atr", tp="atr", mech="np", arm=np.inf, par=float(h), hold=24))
    C.append(dict(name="X_NOTP", fam="X", sl="atr", tp=None, mech="none", arm=np.inf, par=0.0, hold=24))  # base of D2 (not selectable)
    return C


def cell_arrays(CL):
    mm = {"none": 0, "be": 1, "lock": 2, "trail": 3, "np": 5}
    return dict(slatr=np.array([c["sl"] == "atr" for c in CL]), sl=np.array([np.nan if c["sl"] == "atr" else c["sl"] for c in CL], float),
                tpatr=np.array([c["tp"] == "atr" for c in CL]),
                tp=np.array([np.nan if c["tp"] in ("atr", None) else c["tp"] for c in CL], float),
                mech=np.array([mm[c["mech"]] for c in CL]), arm=np.array([c["arm"] for c in CL], float),
                par=np.array([c["par"] for c in CL], float), hold=np.array([c["hold"] for c in CL], np.int64))


def sim_trade(T, t, o, h, l, c, atrp, slip, tick, CC, nb):
    E0 = o[0]; n = len(o); nC = len(CC["sl"])
    H = np.searchsorted(t, T + CC["hold"] * H1, "left"); H = np.maximum(np.minimum(H, n), 1)
    fav_all = (1 - l / E0) * 100
    cmf_all = np.maximum.accumulate(fav_all)
    # NP: at par h, if the best low has not reached +2 % → hold ends there
    isnp = CC["mech"] == 5
    if isnp.any():
        for j in np.nonzero(isnp)[0]:
            hj = int(np.searchsorted(t, T + CC["par"][j] * H1, "left"))
            if 1 <= hj < H[j] and cmf_all[hj - 1] < 2.0:
                H[j] = hj
    Hmax = int(H.max())
    tt, oo, hh, ll, cc = t[:Hmax], o[:Hmax], h[:Hmax], l[:Hmax], c[:Hmax]
    bp = np.minimum(E0, np.r_[E0, np.minimum.accumulate(ll)[:-1]])
    favbp = (1 - bp / E0) * 100
    sl = np.where(CC["slatr"], min(1.5 * atrp, 6.0), CC["sl"])
    tp = np.where(CC["tpatr"], min(20 * atrp, 10.0), CC["tp"])
    S0 = E0 * (1 + sl / 100)
    lvl_be = E0 * (1 - (0.10 + 2 * slip) / 100)
    base = np.where(CC["mech"] == 1, lvl_be, np.where(CC["mech"] == 2, E0 * (1 - CC["par"] / 100), 0.0))
    mult = np.where(CC["mech"] == 3, 1 + CC["par"] / 100, 0.0)
    dynm = (CC["mech"] >= 1) & (CC["mech"] <= 3)
    armed = (favbp[None, :] >= CC["arm"][:, None]) & dynm[:, None]
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
    full24 = np.searchsorted(t, T + CC["hold"] * H1, "left")
    k = np.where(isstop, np.where(stp_lvl < S0 - 1e-12, 3, 1), np.where(istp, 0, np.where(isnp & (H < np.minimum(full24, n)), 4, 2))).astype(np.int8)
    r = -(X / E0 - 1) * 100
    xm = tt[np.minimum(col, Hmax - 1)] + nb * MIN
    fav = fav_all; adv = (h / E0 - 1) * 100
    f1 = np.argmax(fav >= 1.0) if (fav >= 1.0).any() else BIG
    worked = np.zeros(nC, np.int8); rec = np.zeros(nC, np.int8); best = np.zeros(nC, np.float32); reach = np.zeros(nC, np.float32)
    after = int(120 // nb); a45 = int(45 // nb)
    for j in range(nC):
        aj = np.argmax(adv[:H[j]] >= sl[j]) if (adv[:H[j]] >= sl[j]).any() else BIG
        worked[j] = (f1 < aj) and (f1 < H[j])
        cj = int(col[j])
        reach[j] = cmf_all[min(cj, n - 1, aj if aj < BIG else n - 1)]
        best[j] = cmf_all[min(cj + a45, n - 1)]
        if k[j] in (1, 3):
            seg = fav[cj + 1: cj + 1 + after]
            rec[j] = len(seg) > 0 and seg.max() >= 1.0
    return r.astype(np.float32), k, xm.astype(np.int64), worked, rec, best, reach


def load1m(p, days):
    xs = []; ok = set()
    for d in days:
        ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d"); got = False
        for kd in K1_CSV:
            f = os.path.join(kd, p, ds + ".csv")
            if os.path.exists(f) and os.path.getsize(f) > 20:
                x = pd.read_csv(f)
                if len(x):
                    xs.append(x[["t", "o", "h", "l", "c"]]); ok.add(d); got = True; break
        if not got:
            f = os.path.join(K1_NPZ, p, ds + ".npz")
            if os.path.exists(f):
                z = np.load(f); xs.append(pd.DataFrame({k: z[k] for k in ("t", "o", "h", "l", "c")})); ok.add(d)
    if not xs:
        return None, ok
    x = pd.concat(xs).drop_duplicates("t").sort_values("t")
    t = x.t.values.astype("int64"); t = np.where(t > 10**14, t // 1000, t)
    return (t, x.o.values.astype(float), x.h.values.astype(float), x.l.values.astype(float), x.c.values.astype(float)), ok


S.sim_trade = sim_trade; S.cell_arrays = cell_arrays; S.load1m = load1m


def work(args):
    p, rows, CL = args
    out, _ = S.run_rows(p, rows, CL)
    return p, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--procs", type=int, default=7)
    A = ap.parse_args()
    X = pd.read_pickle(os.path.join(D, f"rows_{A.tag}.pkl"))
    CL = cells()
    jobs = [(p, g, CL) for p, g in X.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
    res = []; t0 = time.time()
    with mp.get_context("fork").Pool(A.procs) as pool:
        for i, (p, out) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            res.append(out)
            if i % 50 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p, f"{time.time() - t0:.0f}s", flush=True)
    Z = {k: np.concatenate([r_[k] for r_ in res]) for k in res[0]}
    o = np.argsort(Z["xi"]); Z = {k: v[o] for k, v in Z.items()}
    np.savez_compressed(os.path.join(D, f"sim_{A.tag}.npz"), names=np.array([c["name"] for c in CL]), **Z)
    print("done", len(Z["xi"]), "ok", Z["ok"].sum(), "1m", Z["src1m"][Z["ok"]].mean(), flush=True)


if __name__ == "__main__":
    main()
