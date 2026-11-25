#!/usr/bin/env python3
"""NOSPOT_TODAY_EXIT (2026-11-25, DECISION_LOG 642, research only) — SIM. The EXHAUSTION trigger on NO-SPOT coins (NOSPOT_SLEEVE NS0 rows,
reports/NOSPOT_SLEEVE_data/X.pkl) re-walked at TODAY's EXH exit (DL 609: TP min(20 x ATR15, 7 %) / SL min(1.5 x ATR15, 6 %) / 24 h) on the
NOSPOT_EXIT 1m engine (nospot_exit_sim.run_rows: stop first on the bar HIGH at max(stop, open), maker TP on the LOW, both → stop, real funding),
with the TP_GRID 2026-11-20 TP-only cell kernel (the with-spot side of the comparison = reports/TP_GRID_2026-11-20_EXH_grid.csv FRENZY rows).
Cells: REF (DL 579 TP min(20 x ATR, 10)) — parity vs NOSPOT_EXIT REF_XE −0.050 on 2,696 · F5 · F7 · TODAY · TODAY_T05 (TP only ≥ 0.05 % through).
1m sources: NOSPOT_EXIT k1m (npz) first, then the EXH_EXIT2 / TP_GRID csv caches (merged per pair). No network.
Run: venv/bin/python -I scripts/NOSPOT_TODAY_EXIT_2026-11-25_sim.py → reports/NOSPOT_TODAY_EXIT_data/sim_ns0.npz"""
import os, sys, time, multiprocessing as mp, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import nospot_exit_sim as S
import nospot_sleeve_lib as L
orig_load = S.load1m
spec = importlib.util.spec_from_file_location("TG", os.path.join(R, "scripts", "TP_GRID_2026-11-20_EXH_sim.py"))
TG = importlib.util.module_from_spec(spec); spec.loader.exec_module(TG)
XS = TG.XS
def load_both(p, days):
    a, oka = orig_load(p, days)
    if a is not None and len(oka) == len(days):
        return a, oka
    b, okb = XS.load1m(p, days)
    if b is None: return a, oka
    if a is None: return b, okb
    t = np.r_[a[0], b[0]]; u, i = np.unique(t, return_index=True)
    return tuple(np.r_[x, y][i] for x, y in zip(a, b)), oka | okb
S.load1m = load_both
OUT = os.path.join(R, "reports", "NOSPOT_TODAY_EXIT_data")
CL = [dict(name="REF", k=20.0, cap=10.0, thr=0.0, hold=24), dict(name="F5", k=np.inf, cap=5.0, thr=0.0, hold=24),
      dict(name="F7", k=np.inf, cap=7.0, thr=0.0, hold=24), dict(name="TODAY", k=20.0, cap=7.0, thr=0.0, hold=24),
      dict(name="TODAY_T05", k=20.0, cap=7.0, thr=0.05, hold=24)]
def work(a):
    p, g = a
    out, _ = S.run_rows(p, g, CL); return out
if __name__ == "__main__":
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    m = L.ns0_mask(X)
    rows = pd.DataFrame(dict(xi=np.nonzero(m)[0], pair=X.pair.values[m], entry_ms=X.entry_ms.values[m].astype("int64"),
                             atrp=X.atrp.values[m], slip=X.slip.values[m]))
    print("NS0 candidate rows", len(rows), rows.pair.nunique(), flush=True)
    jobs = list(rows.groupby("pair")); t0 = time.time(); res = []
    with mp.get_context("fork").Pool(7) as pool:
        for i, out in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            res.append(out)
            if i % 100 == 0: print(i, len(jobs), f"{time.time()-t0:.0f}s", flush=True)
    Z = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "sim_ns0.npz"), names=np.array([c["name"] for c in CL]), **Z)
    print("done", len(Z["xi"]), "ok", int(Z["ok"].sum()), "1m", Z["src1m"][Z["ok"]].mean(), flush=True)
