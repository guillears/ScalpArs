#!/usr/bin/env python3
"""DIST_BUCKET_EXIT (PREREG reports/DIST_BUCKET_EXIT_2026-11-21_PREREG.md) — SIM, BACKTEST. Research only.
TP {today=min(20×ATR,10), 5, 8, 10, 12, 15 %} × SL {today=min(1.5×ATR,6), 4, 6, 8, 10 %} on the 1,547 DIST fills, with the 1m-path engine of
scripts/DIST_THEORY_EXIT_2026-11-21_sim.py (its work() loader; only the per-fill cell logic is replaced: fixed TP / SL levels, no rule exits).
Cell 0 = (today, today) = that sim's REF (parity checked in the eval).
Output: reports/DIST_BUCKET_EXIT_2026-11-21_data/sim_grid.npz
Usage: venv/bin/python -I scripts/DIST_BUCKET_EXIT_2026-11-21_sim.py [--procs 6]"""
import argparse, os, sys, time, importlib.util
import multiprocessing as mp
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
sp = importlib.util.spec_from_file_location("DTE_SIM", os.path.join(R, "scripts", "DIST_THEORY_EXIT_2026-11-21_sim.py"))
SM = importlib.util.module_from_spec(sp); sys.modules["DTE_SIM"] = SM; sp.loader.exec_module(SM)
D = os.path.join(R, "reports", "DIST_BUCKET_EXIT_2026-11-21_data")
MIN, H1 = 60_000, 3_600_000
TPS = ["T", 5.0, 8.0, 10.0, 12.0, 15.0]
SLS = ["T", 4.0, 6.0, 8.0, 10.0]


def cells():
    return [dict(name=f"TP{tp if tp == 'T' else int(tp)}_SL{sl if sl == 'T' else int(sl)}", fam="G", tp=tp, sl=sl) for tp in TPS for sl in SLS]


def sim_grid(t, o, h, l, c, T, atrp, tick, an, b15, ft, fcum, CL, nb=1):
    E0 = o[0]; n = len(o)
    end_t = T + SM.HOLD * H1
    H = int(np.searchsorted(t, end_t, "left")); H = max(min(H, n), 1)
    t, o, h, l, c = t[:H], o[:H], h[:H], l[:H], c[:H]
    last = H - 1
    cens = (t[last] + nb * MIN) < (end_t - nb * MIN)
    sl0 = min(1.5 * atrp, 6.0); tp0 = min(20 * atrp, 10.0)
    out = []
    for cl in CL:
        slp = sl0 if cl["sl"] == "T" else cl["sl"]; tpp = tp0 if cl["tp"] == "T" else cl["tp"]
        S_ = E0 * (1 + slp / 100); TP_ = E0 * (1 - tpp / 100)
        hs = h >= S_; ht = l <= TP_ - tick
        fs = int(np.argmax(hs)) if hs.any() else 10**9
        ftp = int(np.argmax(ht)) if ht.any() else 10**9
        if fs < 10**9 and fs <= ftp:
            X = max(S_, o[fs]); k = 1; j = fs
        elif ftp < 10**9:
            X = TP_; k = 0; j = ftp
        else:
            X = c[last]; k = 9 if cens else 2; j = last
        out.append((-(X / E0 - 1) * 100, k, int(t[j]) + nb * MIN, slp))
    return out


SM.sim_one = sim_grid
SM.cells = cells


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6)
    A = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    S, F = SM.frame()
    CL = cells()
    jobs = [(p, g, CL) for p, g in F.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
    res = []; t0 = time.time()
    with mp.get_context("fork").Pool(A.procs) as pool:
        for i, (p, out) in enumerate(pool.imap_unordered(SM.work, jobs, chunksize=2)):
            res.append(out)
            if i % 100 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p, f"{time.time() - t0:.0f}s", flush=True)
    Z = {k: np.concatenate([r_[k] for r_ in res]) for k in res[0]}
    o = np.argsort(Z["ix"]); Z = {k: v[o] for k, v in Z.items()}
    np.savez_compressed(os.path.join(D, "sim_grid.npz"), names=np.array([c["name"] for c in CL]), **Z)
    print("done", len(Z["ix"]), "ok", int(Z["ok"].sum()), "1m", int(Z["src1m"].sum()), flush=True)


if __name__ == "__main__":
    main()
