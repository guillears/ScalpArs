#!/usr/bin/env python3
"""EXH_THEORY_EXIT (PREREG reports/EXH_THEORY_EXIT_2026-11-21_PREREG.md) — SIM, BACKTEST. Research only.
Theory-anchored exits for EXHAUSTION_SHORT: pump-anchored maker TPs (VWAP / retraces / base, replace or nearer), stall exit, structure stop
at the pump peak, funding budget. Engine rules = EXH_EXIT2 (stop-first in a bar, stop filled at max(stop, open), maker TP 1 tick through,
taker rule exits at the next minute's open, real funding at settlements). Parity vs reports/EXH_EXIT2_data/sim_EXH.npz REF is checked.
Output: reports/EXH_THEORY_EXIT_2026-11-21_data/sim_BT.npz
Usage: venv/bin/python -I scripts/EXH_THEORY_EXIT_2026-11-21_sim.py [--procs 7]"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import exh_exit2_sim as E2   # noqa: E402  (1m loader with the EXH_EXIT2 caches; patches the engine module)
S = E2.S; TB = S.TB
import exh_expand_build as XB   # noqa: E402
D = os.path.join(R, "reports", "EXH_THEORY_EXIT_2026-11-21_data")
MIN, H1, DAY, BAR = 60_000, 3_600_000, 86_400_000, 300_000
HOLD = 24


def cells():
    C = [dict(name="REF", fam="REF")]
    for a in ("VWAP", "R382", "R50", "R618", "BASE"):
        C.append(dict(name=f"A_{a}", fam="A", anc=a, mode="replace"))
    for a in ("VWAP", "R382", "R50", "R618", "BASE"):
        C.append(dict(name=f"N_{a}", fam="N", anc=a, mode="nearer"))
    for x in (3, 5, 7):
        for n in (60, 120, 240):
            C.append(dict(name=f"B_X{x}_N{n}", fam="B", x=x, n=n))
    C += [dict(name="C_PK0", fam="C", b=0.0, mode="replace"), dict(name="C_PK1", fam="C", b=1.0, mode="replace"),
          dict(name="C_TIGHT", fam="C", b=0.5, mode="tight")]
    C += [dict(name="F_P1", fam="F", y=1.0, neg=False), dict(name="F_P2", fam="F", y=2.0, neg=False), dict(name="F_P1N", fam="F", y=1.0, neg=True)]
    if os.environ.get("EXH_CTRL"):     # post-hoc CONTROL (labelled, not pre-registered): fixed wider stops + wider-of(peak, today)
        C += [dict(name=f"K_SL{s_}", fam="K", sl=float(s_)) for s_ in (7, 8, 10, 12, 15)]
        C += [dict(name="W_PK0", fam="C", b=0.0, mode="wider"), dict(name="C_PK0_NOCAP", fam="C", b=0.0, mode="nocap")]
    if os.environ.get("EXH_TPFIX"):    # operator 11-21: fixed take profits (today's stop / 24 h), descriptive — TP_GRID DL 603 is the reference
        C += [dict(name=f"TPF{x_}", fam="TPF", tp=float(x_)) for x_ in (5, 7, 8, 9)]
    return C


def anchor_px(anc, a):
    pk, bs, vw = a["peak"], a["base"], a["vwap"]
    if anc == "VWAP":
        return vw
    if anc == "BASE":
        return bs
    f = {"R382": 0.382, "R50": 0.5, "R618": 0.618}[anc]
    return pk - f * (pk - bs)


def sim_one(t, o, h, l, c, T, atrp, tick, anc, ft, fcum, CL):
    """returns per cell: r gross %, kind (0 TP maker, 1 stop, 2 time, 5 rule taker), exit ms."""
    E0 = o[0]; n = len(o)
    Hi = int(np.searchsorted(t, T + HOLD * H1, "left")); Hi = max(min(Hi, n), 1)
    t, o, h, l, c = t[:Hi], o[:Hi], h[:Hi], l[:Hi], c[:Hi]; n = Hi
    sl0 = min(1.5 * atrp, 6.0); tp0 = min(20 * atrp, 10.0)
    S0 = E0 * (1 + sl0 / 100); TP0 = E0 * (1 - tp0 / 100)
    runmin = np.minimum.accumulate(l)
    best = (1 - runmin / E0) * 100
    newlow = np.r_[True, l[1:] < runmin[:-1]]
    lastnew = np.maximum.accumulate(np.where(newlow, np.arange(n), 0))
    # funding: settlements strictly after T and ≤ bar time; short receives the rate → paid = −Σ rate × 100
    if len(ft):
        i0 = np.searchsorted(ft, T, "right")
        st = ft[i0:np.searchsorted(ft, t[-1] + MIN, "right")]
        cum_at = (fcum[i0 + 1:i0 + 1 + len(st)] - fcum[i0]) * 100        # received % after each settlement
    else:
        st = np.zeros(0, "int64"); cum_at = np.zeros(0)
    out = []
    for cl in CL:
        S_ = S0; TP_ = TP0; trig = None
        f = cl["fam"]
        if f in ("A", "N") and anc is not None:
            ap = anchor_px(cl["anc"], anc)
            if np.isfinite(ap) and ap <= E0 * 0.99:
                ap = max(ap, E0 * 0.75)
                TP_ = ap if f == "A" else max(ap, TP0)
        elif f == "C" and anc is not None:
            pk = anc["peak"]
            if np.isfinite(pk) and pk >= E0 * 1.01:
                lvl = pk * (1 + cl["b"] / 100) if cl["mode"] == "nocap" else min(pk * (1 + cl["b"] / 100), E0 * 1.15)
                S_ = {"replace": lvl, "nocap": lvl, "tight": min(S0, lvl), "wider": max(S0, lvl)}[cl["mode"]]
        elif f == "K":
            S_ = E0 * (1 + cl["sl"] / 100)
        elif f == "TPF":
            TP_ = E0 * (1 - cl["tp"] / 100)
        elif f == "B":
            ok = (best >= cl["x"]) & ((np.arange(n) - lastnew) * 1 >= cl["n"]) & (c < E0)
            ii = np.nonzero(ok[:-1])[0]
            if len(ii):
                trig = int(ii[0]) + 1                       # exit at the open of the next bar
        elif f == "F" and len(st):
            paid = -cum_at
            for si, s_ in enumerate(st):
                if paid[si] >= cl["y"]:
                    j = int(np.searchsorted(t, s_, "left"))
                    if j >= n:
                        break
                    if cl["neg"]:
                        pr = (1 - c[max(j - 1, 0)] / E0) * 100
                        if not (pr < paid[si]):
                            continue
                    trig = j if t[j] > s_ else j + 1
                    if trig >= n:
                        trig = None
                    break
        hs = h >= S_
        ht = l <= TP_ - tick
        lim = n if trig is None else trig
        fs = int(np.argmax(hs[:lim])) if hs[:lim].any() else 10**9
        ftp = int(np.argmax(ht[:lim])) if ht[:lim].any() else 10**9
        if fs < 10**9 and fs <= ftp:
            X = max(S_, o[fs]); k = 1; j = fs
        elif ftp < 10**9:
            X = TP_; k = 0; j = ftp
        elif trig is not None:
            X = o[trig]; k = 5; j = trig
        else:
            X = c[n - 1]; k = 2; j = n - 1
        xm = int(t[j]) + MIN if k != 5 else int(t[j])
        out.append((-(X / E0 - 1) * 100, k, xm))
    return out


def episodes_by_pair():
    E = XB.episodes().dropna(subset=["on_ms", "spike_ms"])
    E = E.sort_values("on_ms").drop_duplicates(["pair", "on_ms"])
    return {p: g for p, g in E.groupby("pair")}


def anchors(p, Ts, EP):
    five = TB.load(p)
    res = {}
    if five is None or p not in EP:
        return res
    t5, o5, h5, l5, c5, v5, qv5 = five
    g = EP[p]
    for T in Ts:
        m = g[(g.on_ms <= T) & (g.spike_ms + 72 * H1 >= T)]
        if not len(m):
            continue
        on = int(m.on_ms.iloc[-1]); a = int(np.searchsorted(t5, on)); j = int(np.searchsorted(t5, T))
        if a < 6 or j <= a or j > len(t5):
            continue
        tp = (h5[a:j] + l5[a:j] + c5[a:j]) / 3; vv = v5[a:j]
        res[T] = dict(base=float(c5[a - 6]), peak=float(h5[a:j].max()), vwap=float((tp * vv).sum() / vv.sum()) if vv.sum() > 0 else np.nan,
                      on=on)
    return res


def work(args):
    p, rows, CL, EP = args
    ft, fcum, months = TB.funding_for(p)
    ft = np.asarray(ft, "int64"); fcum = np.asarray(fcum, float)
    days = sorted({int(T) // DAY + k for T in rows.entry_ms.values for k in range(3)})
    one, okd = E2.load1m(p, days)
    AN = anchors(p, rows.entry_ms.values.astype(np.int64), EP)
    nC = len(CL); n = len(rows)
    out = dict(xi=rows.xi.values.astype(np.int64), r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8),
               xm=np.zeros((n, nC), np.int64), f=np.zeros((n, nC), np.float32), ok=np.zeros(n, bool), anc=np.zeros(n, bool),
               peak=np.full(n, np.nan), base=np.full(n, np.nan), vwap=np.full(n, np.nan))
    if one is None:
        return p, out
    t1, o1, h1, l1, c1 = one; tick1 = S.tick_of(o1, h1, l1, c1)
    for i, (T, atrp) in enumerate(zip(rows.entry_ms.values.astype(np.int64), rows.atrp.values.astype(float))):
        need = {int(T) // DAY + k for k in range(3)}
        if not need <= okd:
            continue
        a = int(np.searchsorted(t1, T))
        if a >= len(t1) or t1[a] != T:
            continue
        b = int(np.searchsorted(t1, T + (HOLD + 2) * H1))
        an = AN.get(int(T))
        res = sim_one(t1[a:b], o1[a:b], h1[a:b], l1[a:b], c1[a:b], int(T), atrp, tick1, an, ft, fcum, CL)
        for j, (r, k, xm) in enumerate(res):
            out["r"][i, j] = r; out["k"][i, j] = k; out["xm"][i, j] = xm
            if len(ft):
                out["f"][i, j] = (fcum[np.searchsorted(ft, xm, "right")] - fcum[np.searchsorted(ft, T, "right")]) * 100
        out["ok"][i] = True
        if an is not None:
            out["anc"][i] = True; out["peak"][i] = an["peak"]; out["base"][i] = an["base"]; out["vwap"][i] = an["vwap"]
    return p, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=7); ap.add_argument("--pairs", default=""); ap.add_argument("--tag", default="EXH")
    A = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    X = pd.read_pickle(os.path.join(R, "reports", "EXH_EXIT2_data", f"rows_{A.tag}.pkl"))
    if A.pairs:
        X = X[X.pair.isin(A.pairs.split(","))]
    CL = cells(); EP = episodes_by_pair()
    jobs = [(p, g, CL, EP) for p, g in X.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
    res = []; t0 = time.time()
    with mp.get_context("fork").Pool(A.procs) as pool:
        for i, (p, out) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            res.append(out)
            if i % 50 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p, f"{time.time() - t0:.0f}s", flush=True)
    Z = {k: np.concatenate([r_[k] for r_ in res]) for k in res[0]}
    o = np.argsort(Z["xi"]); Z = {k: v[o] for k, v in Z.items()}
    np.savez_compressed(os.path.join(D, ("sim_BT_tpf" if os.environ.get("EXH_TPFIX") else "sim_BT_ctrl" if os.environ.get("EXH_CTRL") else "sim_BT") + ("" if A.tag == "EXH" else "_" + A.tag) + ".npz"), names=np.array([c["name"] for c in CL]), **Z)
    print("done", len(Z["xi"]), "ok", int(Z["ok"].sum()), "anchored", int(Z["anc"].sum()), flush=True)


if __name__ == "__main__":
    main()
