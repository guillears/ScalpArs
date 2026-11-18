#!/usr/bin/env python3
"""DIST_EXIT (PREREG reports/DIST_EXIT_2026-11-18_PREREG.md) — SIM. Every L1 row (rows.pkl; the PRIMARY DIST cohort is a subset), every
RND control row and every EXHAUSTION baseline trade × the 54 pre-registered exit cells + REF (Exit B), on the 1m path (48 h, 5m fallback
flagged).  The fill engine is scripts/nospot_exit_sim.py (run_rows / sim_trade logic, same rules: stop at bar open checked first on the
HIGH, then the maker TP ≥ 1 tick through on the LOW, both → stop; gap-aware stop fill; best low updates after the bar; real funding),
extended only with two flags: slatr (stop = min(1.5 × ATR15, 6 %)) and tpatr (TP = min(20 × ATR15, 10 %)) per cell.
Output reports/DIST_EXIT_data/sim_<SET>.npz  (SET = L1 / RND / EXH) + paths_L1.npz (5-min sampled close path for the anchored chart).
Usage: venv/bin/python -I scripts/dist_exit_sim.py [--procs 4]"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import nospot_exit_sim as NS   # noqa: E402  (engine; also patches 5m / funding sources via exh_expand_build)
import nospot_sleeve_lib as NL  # noqa: E402  (slip model)
D = os.path.join(R, "reports", "DIST_EXIT_data")
K1 = [os.path.join(R, "reports", "EXH_EXIT_GRID_data", "k1m"), os.path.join(R, "reports", "NOSPOT_EXIT_data", "k1m"), os.path.join(D, "k1m")]
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
BIG = 10**9


def cells():
    """name, fam, sl (% or nan with slatr), tp (% / nan), slatr, tpatr, mech, arm, par, hold."""
    C = [dict(name="REF_B", fam="REF", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="none", arm=np.inf, par=0.0, hold=24)]
    for tp in (1, 1.5, 2, 3, 5, 10):
        for sl in (2, 3, 4, 6, "ATR"):
            C.append(dict(name=f"A_TP{tp:g}_SL{sl}", fam="A", sl=np.nan if sl == "ATR" else float(sl), tp=float(tp),
                          slatr=int(sl == "ATR"), tpatr=0, mech="none", arm=np.inf, par=0.0, hold=24))
    for a in (2, 3, 4):
        C.append(dict(name=f"B_BE{a}", fam="B", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="be", arm=float(a), par=0.0, hold=24))
    for a, k in ((3, 1), (4, 2), (5, 3)):
        C.append(dict(name=f"C_L{a}{k}", fam="C", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="lock", arm=float(a), par=float(k), hold=24))
    for a in (2, 3, 4):
        for g in (1, 1.5, 2, 3):
            C.append(dict(name=f"D_T{a}_G{g:g}", fam="D", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="trail", arm=float(a), par=float(g), hold=24))
    for m in (1, 2):
        C.append(dict(name=f"D_T3_A{m}", fam="D2", sl=np.nan, tp=np.nan, slatr=1, tpatr=1, mech="trailatr", arm=3.0, par=float(m), hold=24))
    for h in (4, 12, 24, 48):
        C.append(dict(name=f"T_{h}h", fam="T", sl=np.nan, tp=np.nan, slatr=1, tpatr=0, mech="none", arm=np.inf, par=0.0, hold=h))
    return C


def cell_arrays(CL):
    mm = {"none": 0, "be": 1, "lock": 2, "trail": 3, "trailatr": 4}
    return dict(ref=np.zeros(len(CL), bool), slatr=np.array([c["slatr"] for c in CL], bool), tpatr=np.array([c["tpatr"] for c in CL], bool),
                sl=np.array([c["sl"] for c in CL], float), tp=np.array([c["tp"] for c in CL], float),
                mech=np.array([mm[c["mech"]] for c in CL]), arm=np.array([c["arm"] for c in CL], float),
                par=np.array([c["par"] for c in CL], float), hold=np.array([c["hold"] for c in CL], np.int64))


def load1m(p, days):
    xs = []; ok = set()
    for d in days:
        ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d")
        f = os.path.join(K1[0], p, ds + ".csv")
        if os.path.exists(f) and os.path.getsize(f) > 20:
            x = pd.read_csv(f)
            if len(x):
                xs.append(x[["t", "o", "h", "l", "c"]]); ok.add(d); continue
        for kd in (K1[1], K1[2]):
            f = os.path.join(kd, p, ds + ".npz")
            if os.path.exists(f):
                z = np.load(f); xs.append(pd.DataFrame({k: z[k] for k in ("t", "o", "h", "l", "c")})); ok.add(d); break
    if not xs:
        return None, ok
    x = pd.concat(xs).drop_duplicates("t").sort_values("t")
    t = x.t.values.astype("int64"); t = np.where(t > 10**14, t // 1000, t)
    return (t, x.o.values.astype(float), x.h.values.astype(float), x.l.values.astype(float), x.c.values.astype(float)), ok


def sim_trade(T, t, o, h, l, c, atrp, slip, tick, CC, nb):
    """= nospot_exit_sim.sim_trade with per-cell slatr / tpatr flags (logic otherwise identical)."""
    E0 = o[0]; n = len(o); nC = len(CC["sl"])
    end_t = T + CC["hold"] * H1
    H = np.searchsorted(t, end_t, "left"); H = np.maximum(np.minimum(H, n), 1)
    Hmax = int(H.max())
    tt, oo, hh, ll, cc = t[:Hmax], o[:Hmax], h[:Hmax], l[:Hmax], c[:Hmax]
    bp = np.minimum(E0, np.r_[E0, np.minimum.accumulate(ll)[:-1]])
    favbp = (1 - bp / E0) * 100
    sl = np.where(CC["slatr"], min(1.5 * atrp, 6.0), CC["sl"])
    tp = np.where(CC["tpatr"], min(20 * atrp, 10.0), CC["tp"])
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
    # censored: a time exit whose path ended before the hold
    cens = (k == 2) & (tt[np.minimum(last, Hmax - 1)] + nb * MIN < end_t - nb * MIN)
    k = np.where(cens, 9, k).astype(np.int8)
    fav = (1 - l / E0) * 100; adv = (h / E0 - 1) * 100
    f1 = np.argmax(fav >= 1.0) if (fav >= 1.0).any() else BIG
    worked = np.zeros(nC, np.int8); rec = np.zeros(nC, np.int8); best = np.zeros(nC, np.float32); reach = np.zeros(nC, np.float32)
    armhit = np.zeros(nC, np.int8)
    cmf = np.maximum.accumulate(fav)
    after = int(120 // nb); a45 = int(45 // nb)
    for j in range(nC):
        aj = np.argmax(adv[:H[j]] >= sl[j]) if (adv[:H[j]] >= sl[j]).any() else BIG
        worked[j] = (f1 < aj) and (f1 < H[j])
        cj = int(col[j])
        reach[j] = cmf[min(cj, n - 1, aj if aj < BIG else n - 1)]
        best[j] = cmf[min(cj + a45, n - 1)]
        armhit[j] = bool(armed[j, :cj + 1].any()) if np.isfinite(CC["arm"][j]) else 0
        if k[j] in (1, 3):
            seg = fav[cj + 1: cj + 1 + after]
            rec[j] = len(seg) > 0 and seg.max() >= 1.0
    return r.astype(np.float32), k, xm.astype(np.int64), worked, rec, best, reach, armhit, sl.astype(np.float32), tp.astype(np.float32)


def run_rows(p, rows, CL, keep_path=None):
    CC = cell_arrays(CL); nC = len(CL)
    days = set()
    for T in rows.entry_ms.values:
        for kk in range(3):
            days.add(int(T) // DAY + kk)
    one, okd = load1m(p, sorted(days))
    if one is not None:
        t1, o1, h1, l1, c1 = one; tick1 = NS.tick_of(o1, h1, l1, c1)
    five = None
    ft, fc, months = NS.TB.funding_for(p)
    ft = np.asarray(ft, "int64"); fc = np.asarray(fc, float)
    n = len(rows)
    out = dict(r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8), xm=np.zeros((n, nC), np.int64),
               f=np.zeros((n, nC), np.float32), worked=np.zeros((n, nC), np.int8), rec=np.zeros((n, nC), np.int8),
               best=np.zeros((n, nC), np.float32), reach=np.zeros((n, nC), np.float32), armhit=np.zeros((n, nC), np.int8),
               slp=np.zeros((n, nC), np.float32), tpp=np.zeros((n, nC), np.float32), src1m=np.zeros(n, bool), ok=np.zeros(n, bool))
    paths = {}
    for i, (T, atrp, slip) in enumerate(zip(rows.entry_ms.values.astype(np.int64), rows.atrp.values.astype(float), rows.slip.values.astype(float))):
        need = {int(T) // DAY + kk for kk in range(3)}
        use1 = one is not None and need <= okd
        if one is not None and not use1:      # last days of the data: accept if the entry day and the next exist up to the data end
            last_day = pd.Timestamp("2026-11-18").value // 10**6 // DAY
            use1 = {d for d in need if d < last_day} <= okd
        if use1:
            a = int(np.searchsorted(t1, T)); use1 = a < len(t1) and t1[a] == T
        if use1:
            b = int(np.searchsorted(t1, T + (NS.MAXH + 2) * H1))
            t, o, h, l, c, tick, nb = t1[a:b], o1[a:b], h1[a:b], l1[a:b], c1[a:b], tick1, 1
        else:
            if five is None:
                five = NS.TB.load(p)
                if five is not None:
                    tick5 = NS.tick_of(five[1], five[2], five[3], five[4])
            if five is None:
                continue
            t5 = five[0]; a = int(np.searchsorted(t5, T))
            if a >= len(t5) or t5[a] != T:
                continue
            b = int(np.searchsorted(t5, T + (NS.MAXH + 2) * H1))
            t, o, h, l, c, tick, nb = t5[a:b], five[1][a:b], five[2][a:b], five[3][a:b], five[4][a:b], tick5, 5
        if len(o) < 2:
            continue
        r, k, xm, wk, rc, be, rh, ah, slp, tpp = sim_trade(T, t, o, h, l, c, atrp, slip, tick, CC, nb)
        fs = (fc[np.searchsorted(ft, xm, "right")] - fc[np.searchsorted(ft, T, "right")]) * 100 if len(ft) else np.zeros(nC)
        me = pd.Timestamp(int(T), unit="ms").strftime("%Y-%m")
        mx = [pd.Timestamp(int(x_), unit="ms").strftime("%Y-%m") for x_ in xm]
        miss = np.array([(me not in months) or (m_ not in months) for m_ in mx])
        fs = np.where(miss, -(xm - T) / (8 * H1) * NS.TB.FUND_MISS, fs)
        for nm, v in (("r", r), ("k", k), ("xm", xm), ("f", fs), ("worked", wk), ("rec", rc), ("best", be), ("reach", rh), ("armhit", ah),
                      ("slp", slp), ("tpp", tpp)):
            out[nm][i] = v
        out["src1m"][i] = use1; out["ok"][i] = True
        if keep_path is not None and keep_path[i]:
            E0 = o[0]
            ts_ = np.arange(-24, NS.MAXH * 12 + 1) * 300_000 + T
            jj = np.searchsorted(t, ts_, "right") - 1
            cpath = np.where(jj >= 0, -(c[np.clip(jj, 0, len(c) - 1)] / E0 - 1) * 100, np.nan)
            cpath[ts_ > t[-1] + nb * MIN] = np.nan
            paths[i] = cpath.astype(np.float32)
    return out, paths


def work(a):
    p, rows, CL = a
    out, paths = run_rows(p, rows, CL, keep_path=rows.keep.values)
    return p, rows.index.values, out, paths


def frames():
    S = pd.read_pickle(os.path.join(R, "reports", "DIST_WL_2026-11-18_data", "rows.pkl"))
    L1 = pd.DataFrame(dict(pair=S.pair.astype(str).values, entry_ms=S.entry_ms.values.astype(np.int64), atrp=S.atr15_pct.values.astype(float),
                           slip=np.maximum(0.10, NL.slip_model(S.q5.values.astype(float))), keep=True))
    Rn = pd.read_pickle(os.path.join(D, "rnd.pkl"))
    RND = pd.DataFrame(dict(pair=Rn.pair.values, entry_ms=Rn.entry_ms.values.astype(np.int64), atrp=Rn.atr15_pct.values.astype(float),
                            slip=np.maximum(0.10, NL.slip_model(Rn.q5.values.astype(float))), keep=False))
    E = pd.read_csv(os.path.join(R, "reports", "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    EXH = pd.DataFrame(dict(pair=E.pair.values, entry_ms=E.entry_ms.values.astype(np.int64), atrp=E.atrp.values.astype(float),
                            slip=np.maximum(0.10, NL.slip_model(E.q5.values.astype(float))), keep=False))
    return dict(L1=L1, RND=RND, EXH=EXH)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--sets", default="L1,RND,EXH")
    A = ap.parse_args()
    CL = cells(); nC = len(CL)
    F = frames()
    for nm in A.sets.split(","):
        Fr = F[nm]; n = len(Fr)
        res = dict(r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8), xm=np.zeros((n, nC), np.int64),
                   f=np.zeros((n, nC), np.float32), worked=np.zeros((n, nC), np.int8), rec=np.zeros((n, nC), np.int8),
                   best=np.zeros((n, nC), np.float32), reach=np.zeros((n, nC), np.float32), armhit=np.zeros((n, nC), np.int8),
                   slp=np.zeros((n, nC), np.float32), tpp=np.zeros((n, nC), np.float32), src1m=np.zeros(n, bool), ok=np.zeros(n, bool))
        P = {}
        jobs = [(p, Fr[Fr.pair == p], CL) for p in Fr.pair.unique()]
        jobs.sort(key=lambda j: -len(j[1]))
        t0 = time.time()
        with mp.get_context("fork").Pool(A.procs) as pool:
            for i, (p, ix, out, paths) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
                for kk, v in out.items():
                    res[kk][ix] = v
                for j, v in paths.items():
                    P[int(ix[j])] = v
                if i % 100 == 0:
                    print(nm, time.strftime("%H:%M:%S"), i, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(os.path.join(D, f"sim_{nm}.npz"), names=np.array([c["name"] for c in CL]), **res)
        if P:
            ks = np.array(sorted(P)); np.savez_compressed(os.path.join(D, f"paths_{nm}.npz"), idx=ks, p=np.vstack([P[k] for k in ks]))
        print(nm, "done ok", int(res["ok"].sum()), "of", n, "1m", int(res["src1m"].sum()), flush=True)


if __name__ == "__main__":
    main()
