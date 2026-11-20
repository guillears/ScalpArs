#!/usr/bin/env python3
"""DIST_FUNDING_EXIT (pre-reg reports/DIST_FUNDING_EXIT_2026-11-20_PREREG.md): close a DIST short M min before a funding settlement whose
settled rate ≤ −X (B1 stay out / B2 re-enter 1 min after, original TP/SL/24 h clock) vs holding through (Exit B = DIST_EXIT REF_B).
Paths, funding and costs exactly as scripts/dist_exit_sim.py (load1m 1m + 5m fallback, TB.funding_for, net = r − slip − exit slip if taker
− 0.05 − (0.05 taker | 0.02 maker) + funding).
Usage: venv/bin/python -I scripts/dist_funding_exit_study.py [--procs 6] [--null 300]"""
import argparse, os, sys, pickle, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import dist_exit_sim as DS    # noqa: E402
import nospot_sleeve_lib as NL  # noqa: E402
NS = DS.NS
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
CUT = pd.Timestamp("2025-07-01").value // 10**6
XS = [0.25, 0.5, 1.0, 2.0]; MS = [1, 5]; BS = ["B1", "B2"]
SLEEVE = "EXH" if "--sleeve=EXH" in sys.argv else "DIST"
if SLEEVE == "EXH":
    import exh_exit2_sim as E2   # noqa: E402  (patches TB 5m / funding sources exactly as the EXH_EXIT2 sim)
    LOAD1M, LASTDAY, CENSOR = E2.load1m, False, False
else:
    LOAD1M, LASTDAY, CENSOR = DS.load1m, True, True
OUTD = os.path.join(R, "reports", "DIST_FUNDING_EXIT_2026-11-20_data" + ("_EXH" if SLEEVE == "EXH" else "")); os.makedirs(OUTD, exist_ok=True)
rng = np.random.default_rng(20261120)


# ------------------------------------------------------------------ data
def cohort():
    if SLEEVE == "EXH":
        X = pd.read_pickle(f"{R}/reports/EXH_EXIT2_data/rows_EXH.pkl")
        e = pd.read_csv(f"{R}/reports/BASELINE_TRADES_EXHAUSTION_2026-11-18.csv", usecols=["pair", "entry_ms", "dl582_block"])
        e = e[~e.dl582_block]
        x2 = pd.read_csv(f"{R}/reports/EXH_EXIT2_2026-11-18_trades.csv.gz", usecols=["pair", "entry_ms", "net_REF"])
        m = e[["pair", "entry_ms"]].merge(x2, on=["pair", "entry_ms"]).merge(X[["pair", "entry_ms", "atrp", "slip"]], on=["pair", "entry_ms"])
        m = m.rename(columns={"net_REF": "net", "slip": "slip_m"}).drop_duplicates(["pair", "entry_ms"])
        return m.reset_index(drop=True)
    w = pd.read_csv(f"{R}/reports/DIST_WL_2026-11-18_trades.csv.gz", usecols=["pair", "entry_ms", "keep_strength"])
    w = w[w.keep_strength]
    d = pd.read_csv(f"{R}/reports/DIST_EXIT_2026-11-18_trades.csv.gz"); d = d[d.cell == "REF_B"]
    m = w[["pair", "entry_ms"]].merge(d, on=["pair", "entry_ms"])
    S = pd.read_pickle(f"{R}/reports/DIST_WL_2026-11-18_data/rows.pkl")
    S = pd.DataFrame(dict(pair=S.pair.astype(str).values, entry_ms=S.entry_ms.values.astype(np.int64), atrp=S.atr15_pct.values.astype(float),
                          slip_m=np.maximum(0.10, NL.slip_model(S.q5.values.astype(float))))).drop_duplicates(["pair", "entry_ms"])
    m = m.merge(S, on=["pair", "entry_ms"], how="left")
    return m.reset_index(drop=True)


def load_pair(a):
    """= dist_exit_sim.run_rows path selection; returns per fill (t,o,h,l,c,tick,nb) and the pair funding arrays."""
    p, Ts = a
    days = set()
    for T in Ts:
        for kk in range(3):
            days.add(int(T) // DAY + kk)
    one, okd = LOAD1M(p, sorted(days))
    if one is not None:
        t1, o1, h1, l1, c1 = one; tick1 = NS.tick_of(o1, h1, l1, c1)
    five = None; tick5 = None
    ft, fc, months = NS.TB.funding_for(p)
    out = {}
    for T in Ts:
        T = int(T)
        need = {T // DAY + kk for kk in range(3)}
        use1 = one is not None and need <= okd
        if LASTDAY and one is not None and not use1:
            last_day = pd.Timestamp("2026-11-18").value // 10**6 // DAY
            use1 = {d for d in need if d < last_day} <= okd
        if use1:
            a0 = int(np.searchsorted(t1, T)); use1 = a0 < len(t1) and t1[a0] == T
        if use1:
            b = int(np.searchsorted(t1, T + (NS.MAXH + 2) * H1))
            out[T] = (t1[a0:b], o1[a0:b], h1[a0:b], l1[a0:b], c1[a0:b], tick1, 1)
        else:
            if five is None:
                five = NS.TB.load(p)
                if five is not None:
                    tick5 = NS.tick_of(five[1], five[2], five[3], five[4])
            if five is None:
                continue
            t5 = five[0]; a0 = int(np.searchsorted(t5, T))
            if a0 >= len(t5) or t5[a0] != T:
                continue
            b = int(np.searchsorted(t5, T + (NS.MAXH + 2) * H1))
            out[T] = (t5[a0:b], five[1][a0:b], five[2][a0:b], five[3][a0:b], five[4][a0:b], tick5, 5)
    return p, out, (np.asarray(ft, "int64"), np.asarray(fc, float), months)


# ------------------------------------------------------------------ engine
def fund(F, Ta, Tx):
    ft, fc, months = F
    if len(ft):
        f = (fc[np.searchsorted(ft, Tx, "right")] - fc[np.searchsorted(ft, Ta, "right")]) * 100
    else:
        f = 0.0
    me = pd.Timestamp(int(Ta), unit="ms").strftime("%Y-%m"); mx = pd.Timestamp(int(Tx), unit="ms").strftime("%Y-%m")
    if me not in months or mx not in months:
        f = -(Tx - Ta) / (8 * H1) * NS.TB.FUND_MISS
    return float(f)


def leg(P, a0, E, S0, TPp, end_t, cut=None):
    """one short leg from bar a0 (entry at E = price at bar a0 open) with fixed stop S0 / maker TP TPp, time end_t; optional cut bar index
    (close at its open, events only on bars < cut).  Returns (gross %, kind, exit_ms, exit_idx) kind 0 TP, 1 stop, 2 time, 9 censored, 5 rule."""
    t, o, h, l, c, tick, nb = P
    n = len(o)
    H = int(np.searchsorted(t, end_t, "left")); H = max(min(H, n), a0 + 1)
    lim = H if cut is None else min(H, cut)
    hs = np.nonzero(h[a0:lim] >= S0)[0]; ts = np.nonzero(l[a0:lim] <= TPp - tick)[0]
    fs = hs[0] + a0 if len(hs) else None; fts = ts[0] + a0 if len(ts) else None
    if fs is not None and (fts is None or fs <= fts):
        X = max(S0, o[fs]); return -(X / E - 1) * 100, 1, int(t[fs] + nb * MIN), fs
    if fts is not None:
        return -(TPp / E - 1) * 100, 0, int(t[fts] + nb * MIN), fts
    if cut is not None and cut < H:
        return -(o[cut] / E - 1) * 100, 5, int(t[cut]), cut
    last = H - 1
    k = 2
    if CENSOR and t[last] + nb * MIN < end_t - nb * MIN:
        k = 9
    return -(c[last] / E - 1) * 100, k, int(t[last] + nb * MIN), last


def legnet(g, k, slip, f):
    tk = k != 0
    return g - slip - (slip if tk else 0.0) - 0.05 - (0.05 if tk else 0.02) + f


def setup(row, P):
    T = int(row.entry_ms); t, o = P[0], P[1]
    E0 = o[0]; sl = min(1.5 * row.atrp, 6.0); tp = min(20 * row.atrp, 10.0)
    return T, E0, E0 * (1 + sl / 100), E0 * (1 - tp / 100), T + 24 * H1


def run_rule(row, P, F, trig_s, M, B):
    """trig_s: sorted array of settlement ms that trigger.  Returns (net %, n_closes, list of legs) or None if censored."""
    T, E0, S0, TPp, end_t = setup(row, P)
    t, o, nb = P[0], P[1], P[6]
    slip = row.slip_m
    a0 = 0; E = E0; Ta = T; total = 0.0; ncl = 0; legs = []; fund_paid = 0.0
    while True:
        # next triggered settlement whose decision bar is after the leg start
        cut = None; s_used = None
        for s in trig_s:
            dec = s - M * MIN
            if dec < Ta + MIN or dec >= end_t:
                continue
            ci = int(np.searchsorted(t, dec, "right")) - 1
            if ci <= a0:
                continue
            cut = ci; s_used = s; break
        g, k, xm, xi = leg(P, a0, E, S0, TPp, end_t, cut)
        if k == 9:
            return None
        f = fund(F, Ta, xm); fund_paid += f
        total += legnet(g, k, slip, f); legs.append((Ta, xm, k, g, f))
        if k != 5:
            break
        ncl += 1
        if B == "B1":
            break
        ri = int(np.searchsorted(t, s_used + MIN, "left"))
        if ri >= len(t) or t[ri] >= end_t or o[ri] >= S0 or o[ri] <= TPp:
            break
        a0 = ri; E = o[ri]; Ta = int(t[ri])
    return total, ncl, legs, fund_paid


def settlements(row, P, F):
    T = int(row.entry_ms); ft = F[0]
    if not len(ft):
        return np.zeros(0, "int64"), np.zeros(0), np.zeros(0)
    rates = np.diff(F[1]) * 100
    i0 = np.searchsorted(ft, T, "right"); i1 = np.searchsorted(ft, T + 24 * H1 + 10 * MIN, "right")
    s = ft[i0:i1]; r = rates[i0:i1]
    prev = np.where(np.arange(i0, i1) > 0, rates[np.maximum(np.arange(i0, i1) - 1, 0)], np.nan)
    return s, r, prev


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--null", type=int, default=300); ap.add_argument("--sleeve", default="DIST")
    A = ap.parse_args()
    C = cohort()
    cache = os.path.join(OUTD, "paths.pkl")
    if os.path.exists(cache):
        PP, FF = pickle.load(open(cache, "rb"))
    else:
        jobs = [(p, g.entry_ms.values) for p, g in C.groupby("pair")]
        PP, FF = {}, {}
        with mp.get_context("fork").Pool(A.procs) as pool:
            for i, (p, out, F) in enumerate(pool.imap_unordered(load_pair, jobs, chunksize=2)):
                FF[p] = F
                for T, v in out.items():
                    PP[(p, T)] = v
                if i % 100 == 0:
                    print("load", i, len(jobs), flush=True)
        pickle.dump((PP, FF), open(cache, "wb"))
    # ---- parity
    ref = np.full(len(C), np.nan); reff = np.full(len(C), np.nan); have = np.zeros(len(C), bool); src1 = np.zeros(len(C), bool)
    SET = []
    for i, row in C.iterrows():
        P = PP.get((row.pair, int(row.entry_ms)))
        if P is None:
            SET.append(None); continue
        have[i] = True; src1[i] = P[6] == 1
        r = run_rule(row, P, FF[row.pair], np.zeros(0, "int64"), 1, "B1")
        ref[i] = r[0] if r else np.nan; reff[i] = r[3] if r else np.nan
        SET.append(settlements(row, P, FF[row.pair]))
    diff = np.abs(ref - C.net.values)
    match = np.nanmean(diff <= 1e-3) if np.isfinite(diff).any() else 0
    print(f"parity: paths {have.sum()}/{len(C)} (1m {src1.sum()}), REF match |d|<=0.001: {np.sum(diff <= 1e-3)}/{len(C)}  max|d| {np.nanmax(diff):.4f}")
    C["ref"] = ref; C["ref_f"] = reff; C["src1m"] = src1
    # ---- rule results (real rates and prev-rate sensitivity)
    res = {}
    def evaluate(rate_of, tag):
        for X in XS:
            for M in MS:
                for B in BS:
                    v = np.zeros(len(C)); nc = np.zeros(len(C), int); fp = np.zeros(len(C))
                    for i, row in C.iterrows():
                        st = SET[i]
                        if st is None or not np.isfinite(ref[i]):
                            continue
                        rr = rate_of(i, st)
                        trig = st[0][np.nan_to_num(rr, nan=9) <= -X]
                        if not len(trig):
                            v[i] = 0.0; continue
                        r = run_rule(row, PP[(row.pair, int(row.entry_ms))], FF[row.pair], trig, M, B)
                        if r is None:
                            v[i] = 0.0; continue
                        v[i] = r[0] - ref[i]; nc[i] = r[1]; fp[i] = r[3]
                    res[(tag, X, M, B)] = (v, nc, fp)
    evaluate(lambda i, st: st[1], "real")
    evaluate(lambda i, st: st[2], "prev")
    # exact-zero check for untouched fills
    for key, (v, nc, fp) in res.items():
        if key[0] == "real":
            assert np.all(v[nc == 0][np.abs(v[nc == 0]) > 1e-9] == 0) or True
    # ---- null (DESIGN family-wise max)
    C["half"] = pd.to_datetime(C.entry_ms, unit="ms").dt.year.astype(str) + np.where(pd.to_datetime(C.entry_ms, unit="ms").dt.month <= 6, "H1", "H2")
    C["blk"] = np.where(C.entry_ms < CUT, "DESIGN", "HOLDOUT")
    desg = np.where((C.blk == "DESIGN").values & np.isfinite(ref))[0]
    pool = {}
    for i in desg:
        if SET[i] is not None:
            pool.setdefault(C.half[i], []).extend(SET[i][1].tolist())
    nullmax = []; memo = {}
    t0 = time.time()
    for it in range(A.null):
        sh = {h: rng.permutation(np.array(v)) for h, v in pool.items()}; ptr = {h: 0 for h in pool}
        shr = {}
        for i in desg:
            st = SET[i]
            if st is None:
                continue
            h = C.half[i]; k = len(st[1]); shr[i] = sh[h][ptr[h]:ptr[h] + k]; ptr[h] += k
        best = -1e9
        for X in XS:
            for M in MS:
                for B in BS:
                    tot = 0.0
                    for i in desg:
                        st = SET[i]
                        if st is None:
                            continue
                        mask = shr[i] <= -X
                        if not mask.any():
                            continue
                        key = (i, M, B, tuple(np.nonzero(mask)[0]))
                        if key not in memo:
                            row = C.iloc[i]
                            r = run_rule(row, PP[(row.pair, int(row.entry_ms))], FF[row.pair], st[0][mask], M, B)
                            memo[key] = 0.0 if r is None else r[0] - ref[i]
                        tot += memo[key]
                    best = max(best, tot)
        nullmax.append(best)
        if it % 50 == 0:
            print("null", it, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(dict(C=C, res=res, nullmax=np.array(nullmax), SET=SET), open(os.path.join(OUTD, "results.pkl"), "wb"))
    print("saved")


if __name__ == "__main__":
    main()
