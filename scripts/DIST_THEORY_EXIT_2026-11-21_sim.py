#!/usr/bin/env python3
"""DIST_THEORY_EXIT (PREREG reports/DIST_THEORY_EXIT_2026-11-21_PREREG.md) — SIM, BACKTEST. Research only.
Theory-anchored exits for DISTRIBUTION_SHORT on the 1,547 DIST_EXIT REF_B fills (reports/ENTRY_WL_2026-11-18_data/dist.pkl):
pump-anchored maker TPs (A replace / N nearer), measured move of the distribution candle (M), stall (B), structure stop at the
distribution-candle high / pump peak (C intrabar, R closed 15m bar), absorption exit (G), funding budget (F), fixed wider-stop control (W).
Engine = DIST_EXIT (scripts/dist_exit_sim.py): same 1m / 5m-fallback path selection, stop-first on the HIGH (fill max(stop, open)), maker TP
1 tick through on the LOW, time exit = close of the last bar < 24 h, censored → k 9, rule exits = open of the minute after the trigger,
real funding (missing months → FUND_MISS). Parity vs reports/DIST_EXIT_data/sim_L1.npz REF_B is checked in the eval.
Output: reports/DIST_THEORY_EXIT_2026-11-21_data/sim_BT.npz
Usage: venv/bin/python -I scripts/DIST_THEORY_EXIT_2026-11-21_sim.py [--procs 6]"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import dist_exit_sim as DS   # noqa: E402  (1m loader + engine reference)
NS = DS.NS; TB = NS.TB
D = os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21_data")
MIN, H1, DAY, M15 = 60_000, 3_600_000, 86_400_000, 900_000
HOLD = 24
KIND = {0: "TP", 1: "SL", 2: "T", 5: "R", 9: "cens"}


def cells():
    C = [dict(name="REF", fam="REF")]
    for a in ("VWAP", "R382", "R50", "R618", "BASE"):
        C.append(dict(name=f"A_{a}", fam="A", anc=a))
    for a in ("VWAP", "R382", "R50", "R618", "BASE"):
        C.append(dict(name=f"N_{a}", fam="N", anc=a))
    for m, nm in ((1.0, "1"), (1.5, "15"), (2.0, "2")):
        C.append(dict(name=f"M_{nm}", fam="M", m=m))
    for x in (3, 5, 7):
        for n in (60, 120, 240):
            C.append(dict(name=f"B_X{x}_N{n}", fam="B", x=x, n=n))
    for lv in ("DH", "PK"):
        C += [dict(name=f"C_{lv}0", fam="C", lv=lv, b=0.0, mode="replace"), dict(name=f"C_{lv}1", fam="C", lv=lv, b=1.0, mode="replace"),
              dict(name=f"C_{lv}T", fam="C", lv=lv, b=0.5, mode="tight")]
    C += [dict(name="R_DO", fam="R", lv="DO"), dict(name="R_DH", fam="R", lv="DH")]
    C += [dict(name="G_ABS50", fam="G", x=0.5, loss=False), dict(name="G_ABS100", fam="G", x=1.0, loss=False),
          dict(name="G_ABS100L", fam="G", x=1.0, loss=True)]
    C += [dict(name="F_P1", fam="F", y=1.0, neg=False), dict(name="F_P2", fam="F", y=2.0, neg=False), dict(name="F_P1N", fam="F", y=1.0, neg=True)]
    C += [dict(name=f"W_SL{w}", fam="W", w=float(w)) for w in (8, 10, 12, 15)]
    return C


def anchor_px(anc, a):
    pk, bs, vw = a["PK"], a["base"], a["vwap"]
    if anc == "VWAP":
        return vw
    if anc == "BASE":
        return bs
    f = {"R382": 0.382, "R50": 0.5, "R618": 0.618}[anc]
    return pk - f * (pk - bs)


def sim_one(t, o, h, l, c, T, atrp, tick, an, b15, ft, fcum, CL, nb=1):
    """t.. = path from the entry bar. an = landmarks dict or None. b15 = (end_ms, open, close, qv) of the closed 15m bars after T (or None).
    returns per cell (r gross %, kind, exit ms, stop % used)."""
    E0 = o[0]; n = len(o)
    end_t = T + HOLD * H1
    H = int(np.searchsorted(t, end_t, "left")); H = max(min(H, n), 1)
    t, o, h, l, c = t[:H], o[:H], h[:H], l[:H], c[:H]
    last = H - 1
    cens = (t[last] + nb * MIN) < (end_t - nb * MIN)
    sl0 = min(1.5 * atrp, 6.0); tp0 = min(20 * atrp, 10.0)
    S0 = E0 * (1 + sl0 / 100); TP0 = E0 * (1 - tp0 / 100)
    runmin = np.minimum.accumulate(l)
    best = (1 - runmin / E0) * 100
    newlow = np.r_[True, l[1:] < runmin[:-1]]
    lastnew = np.maximum.accumulate(np.where(newlow, np.arange(H), 0))
    if len(ft):
        i0 = np.searchsorted(ft, T, "right")
        st = ft[i0:np.searchsorted(ft, t[-1] + MIN, "right")]
        cum_at = (fcum[i0 + 1:i0 + 1 + len(st)] - fcum[i0]) * 100
    else:
        st = np.zeros(0, "int64"); cum_at = np.zeros(0)
    out = []
    for cl in CL:
        S_ = S0; TP_ = TP0; trig = None
        f = cl["fam"]
        if f in ("A", "N") and an is not None:
            ap = anchor_px(cl["anc"], an)
            if np.isfinite(ap) and ap <= E0 * 0.99:
                ap = max(ap, E0 * 0.75)
                TP_ = ap if f == "A" else max(ap, TP0)
        elif f == "M" and an is not None:
            tpm = E0 - cl["m"] * (an["DH"] - an["DL"])
            if np.isfinite(tpm) and tpm <= E0 * 0.99:
                TP_ = max(tpm, E0 * 0.75)
        elif f == "C" and an is not None:
            lv = an[cl["lv"]]
            if np.isfinite(lv) and lv >= E0 * 1.01:
                lvl = min(lv * (1 + cl["b"] / 100), E0 * 1.15)
                S_ = lvl if cl["mode"] == "replace" else min(S0, lvl)
        elif f == "W":
            S_ = E0 * (1 + cl["w"] / 100)
        elif f == "R" and an is not None and b15 is not None:
            lv = an[cl["lv"]]
            ok_ = np.isfinite(lv) and (cl["lv"] == "DO" or lv >= E0 * 1.01)
            if ok_:
                if cl["lv"] == "DH":
                    S_ = E0 * 1.15
                be, bo, bc, bq = b15
                hit = np.nonzero(bc > lv)[0]
                if len(hit):
                    j = int(np.searchsorted(t, be[hit[0]], "left"))
                    if j < H:
                        trig = j
        elif f == "G" and an is not None and b15 is not None and np.isfinite(an["DQV"]) and an["DQV"] > 0:
            be, bo, bc, bq = b15
            m = (bc > bo) & (bq >= cl["x"] * an["DQV"])
            if cl["loss"]:
                m &= bc > E0
            hit = np.nonzero(m)[0]
            if len(hit):
                j = int(np.searchsorted(t, be[hit[0]], "left"))
                if j < H:
                    trig = j
        elif f == "B":
            ok = (best >= cl["x"]) & ((np.arange(H) - lastnew) * nb >= cl["n"]) & (c < E0)
            ii = np.nonzero(ok[:-1])[0]
            if len(ii):
                trig = int(ii[0]) + 1
        elif f == "F" and len(st):
            paid = -cum_at
            for si, s_ in enumerate(st):
                if paid[si] >= cl["y"]:
                    j = int(np.searchsorted(t, s_, "left"))
                    if j >= H:
                        break
                    if cl["neg"]:
                        pr = (1 - c[max(j - 1, 0)] / E0) * 100
                        if not (pr < paid[si]):
                            continue
                    trig = j if t[j] > s_ else j + 1
                    if trig >= H:
                        trig = None
                    break
        hs = h >= S_
        ht = l <= TP_ - tick
        lim = H if trig is None else trig
        fs = int(np.argmax(hs[:lim])) if hs[:lim].any() else 10**9
        ftp = int(np.argmax(ht[:lim])) if ht[:lim].any() else 10**9
        if fs < 10**9 and fs <= ftp:
            X = max(S_, o[fs]); k = 1; j = fs
        elif ftp < 10**9:
            X = TP_; k = 0; j = ftp
        elif trig is not None:
            X = o[trig]; k = 5; j = trig
        else:
            X = c[last]; k = 9 if cens else 2; j = last
        xm = int(t[j]) + nb * MIN if k != 5 else int(t[j])
        out.append((-(X / E0 - 1) * 100, k, xm, (S_ / E0 - 1) * 100))
    return out


def landmarks_5m(five, T, on):
    t5, o5, h5, l5, c5, v5, qv5 = five
    res = dict(DH=np.nan, DO=np.nan, DL=np.nan, DQV=np.nan, PK=np.nan, base=np.nan, vwap=np.nan)
    d0 = int(np.searchsorted(t5, T - M15))
    if d0 + 3 <= len(t5) and t5[d0] == T - M15 and t5[d0 + 2] == T - 300_000:
        res.update(DH=float(h5[d0:d0 + 3].max()), DO=float(o5[d0]), DL=float(l5[d0:d0 + 3].min()), DQV=float(qv5[d0:d0 + 3].sum()))
    if np.isfinite(on):
        a = int(np.searchsorted(t5, int(on))); j = int(np.searchsorted(t5, T))
        if a >= 6 and j > a and j <= len(t5):
            tp = (h5[a:j] + l5[a:j] + c5[a:j]) / 3; vv = v5[a:j]
            res.update(base=float(c5[a - 6]), PK=float(h5[a:j].max()), vwap=float((tp * vv).sum() / vv.sum()) if vv.sum() > 0 else np.nan)
    return res


def bars15_5m(five, T):
    t5, o5, h5, l5, c5, v5, qv5 = five
    a = int(np.searchsorted(t5, T)); b = int(np.searchsorted(t5, T + (HOLD + 1) * H1))
    tt = t5[a:b]
    if len(tt) < 3:
        return None
    g = (tt - T) // M15
    ok = np.bincount(g, minlength=g.max() + 1) == 3
    ks = np.nonzero(ok)[0]
    if not len(ks):
        return None
    be, bo, bc, bq = [], [], [], []
    for k in ks:
        m = np.nonzero(g == k)[0] + a
        be.append(T + (k + 1) * M15); bo.append(o5[m[0]]); bc.append(c5[m[-1]]); bq.append(qv5[m].sum())
    return np.array(be, np.int64), np.array(bo), np.array(bc), np.array(bq)


def work(args):
    p, rows, CL = args
    nC = len(CL); n = len(rows)
    out = dict(ix=rows.index.values.astype(np.int64), r=np.full((n, nC), np.nan, np.float32), k=np.full((n, nC), -1, np.int8),
               xm=np.zeros((n, nC), np.int64), f=np.zeros((n, nC), np.float32), slp=np.full((n, nC), np.nan, np.float32), ok=np.zeros(n, bool),
               src1m=np.zeros(n, bool), anc=np.zeros(n, bool), dcand=np.zeros(n, bool), b15ok=np.zeros(n, bool))
    for nm in ("DH", "DO", "DL", "DQV", "PK", "base", "vwap", "E0"):
        out[nm] = np.full(n, np.nan)
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
    for i, (T, atrp, on) in enumerate(zip(rows.entry_ms.values.astype(np.int64), rows.atrp.values.astype(float), rows.on.values.astype(float))):
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
        an = landmarks_5m(five, int(T), on) if five is not None else None
        b15 = bars15_5m(five, int(T)) if five is not None else None
        res = sim_one(t, o, h, l, c, int(T), atrp, tick, an, b15, ft, fc, CL, nb)
        xm = np.array([x[2] for x in res], np.int64)
        fs = (fc[np.searchsorted(ft, xm, "right")] - fc[np.searchsorted(ft, T, "right")]) * 100 if len(ft) else np.zeros(nC)
        me = pd.Timestamp(int(T), unit="ms").strftime("%Y-%m")
        miss = np.array([(me not in months) or (pd.Timestamp(int(x_), unit="ms").strftime("%Y-%m") not in months) for x_ in xm])
        fs = np.where(miss, -(xm - T) / (8 * H1) * TB.FUND_MISS, fs)
        for j, (r, k, x, s) in enumerate(res):
            out["r"][i, j] = r; out["k"][i, j] = k; out["xm"][i, j] = x; out["slp"][i, j] = s
        out["f"][i] = fs; out["ok"][i] = True; out["src1m"][i] = use1; out["E0"][i] = o[0]
        if an is not None:
            for nm in ("DH", "DO", "DL", "DQV", "PK", "base", "vwap"):
                out[nm][i] = an[nm]
            out["anc"][i] = np.isfinite(an["PK"]); out["dcand"][i] = np.isfinite(an["DH"])
        out["b15ok"][i] = b15 is not None
    return p, out


def frame():
    S = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    return S, pd.DataFrame(dict(pair=S.pair.astype(str).values, entry_ms=S.entry_ms.values.astype(np.int64),
                                atrp=S.atr15_pct.values.astype(float), slip=S.slip.values.astype(float),
                                on=S.ep_on_ms.values.astype(float)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--pairs", default="")
    A = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    S, F = frame()
    if A.pairs:
        F = F[F.pair.isin(A.pairs.split(","))]
    CL = cells()
    jobs = [(p, g, CL) for p, g in F.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
    res = []; t0 = time.time()
    with mp.get_context("fork").Pool(A.procs) as pool:
        for i, (p, out) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            res.append(out)
            if i % 100 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p, f"{time.time() - t0:.0f}s", flush=True)
    Z = {k: np.concatenate([r_[k] for r_ in res]) for k in res[0]}
    o = np.argsort(Z["ix"]); Z = {k: v[o] for k, v in Z.items()}
    np.savez_compressed(os.path.join(D, "sim_BT.npz"), names=np.array([c["name"] for c in CL]), **Z)
    print("done", len(Z["ix"]), "ok", int(Z["ok"].sum()), "1m", int(Z["src1m"].sum()), "anchored", int(Z["anc"].sum()),
          "dist-candle", int(Z["dcand"].sum()), "15m bars", int(Z["b15ok"].sum()), flush=True)


if __name__ == "__main__":
    main()
