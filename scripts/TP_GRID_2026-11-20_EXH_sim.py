#!/usr/bin/env python3
"""TP_GRID 2026-11-20 (EXHAUSTION_SHORT take-profit level) — STEP 2 BACKTEST: rows + 1m data + sim.

Research only. Re-uses the EXH_EXIT2 / NOSPOT_EXIT 1m engine (scripts/nospot_exit_sim.run_rows: 1m path, real funding, stop first on the
bar HIGH at max(stop, bar open), maker TP on the bar LOW ≥ 1 tick through, a bar touching both = stop, 24 h time exit at the last close)
with its own cells: today's stop min(1.5 × ATR15, 6 %) and 24 h in EVERY cell; only the take-profit moves.
  REF      = today's TP min(20 × ATR15, 10 %)  (= 10 % for every ATR ≥ 0.5 %)
  F4 … F9  = fixed TP 4 … 9 %
  K{k}_C{X} = min(k × ATR15, X %), k ∈ {1.5, 2, 3, 5}, X ∈ {6, 8, 10}
  F10_T05  = REF with the TP filled only when the low trades ≥ 0.05 % through (queue stress, all TPs: F*_T05 for 6 / 8 too)
Cohorts (rows written to reports/TP_GRID_2026-11-20_EXH_data/rows_<tag>.pkl):
  FRENZY = EXH_EXIT2 rows_EXH (BASELINE_TRADES_EXHAUSTION_2026-11-18 rows with dl582_block False: FRENZY universe, ≥ $20M, spot,
           vol_decay / spot-share / RSI cross, DL 582 filter, one position per pair) — 3,368 fills 2023-07 → 2026-11
  WIDE   = DL 585 extension: BASELINE_TRADES_EXH_WIDE_L4 set C ∧ not in BASE ∧ not DL 582-blocked (the joint_short_sim L4 cohort) — 939
  E      = EXH_EXIT2 rows_E (2023-H1 clean block, 248 candidates; DL 582 + one position applied in the eval)
Missing 1m days come from data.binance.vision daily zips (0 API weight; lead1_data.get stops on 403 / 418 / 429) into
reports/TP_GRID_2026-11-20_EXH_data/k1m/<PAIR>/<YYYY-MM-DD>.csv.
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_EXH_sim.py rows | k1m | sim FRENZY|WIDE|E [--procs 7]"""
import argparse, io, os, sys, time, zipfile
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import exh_exit2_sim as XS      # noqa: E402  (patches nospot_exit_sim's 1m loader + funding / 5m sources)
import exh_exit2_data as XD     # noqa: E402
import lead1_data as LD         # noqa: E402
S = XS.S
RP = os.path.join(R, "reports")
D = os.path.join(RP, "TP_GRID_2026-11-20_EXH_data")
K1_MINE = os.path.join(D, "k1m")
XS.K1_CSV.append(K1_MINE)
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000


def cells():
    C = [dict(name="REF", k=20.0, cap=10.0, thr=0.0)]
    for x in (4, 5, 6, 7, 8, 9):
        C.append(dict(name=f"F{x}", k=np.inf, cap=float(x), thr=0.0))
    for k in (1.5, 2, 3, 5):
        for x in (6, 8, 10):
            C.append(dict(name=f"K{k:g}_C{x}", k=float(k), cap=float(x), thr=0.0))
    for x in (6, 8):
        C.append(dict(name=f"F{x}_T05", k=np.inf, cap=float(x), thr=0.05))
    C.append(dict(name="REF_T05", k=20.0, cap=10.0, thr=0.05))
    for c in C:
        c["hold"] = 24
    return C


def cell_arrays(CL):
    return dict(k=np.array([c["k"] for c in CL], float), cap=np.array([c["cap"] for c in CL], float),
                thr=np.array([c["thr"] for c in CL], float), hold=np.array([c["hold"] for c in CL], np.int64),
                sl=np.zeros(len(CL)))


def sim_trade(T, t, o, h, l, c, atrp, slip, tick, CC, nb):
    """one trade × all TP cells. Returns r (gross short %), k (0 TP · 1 stop · 2 time), xm, worked, rec, best (= 24 h MFE), reach (= 24 h MAE)."""
    E0 = o[0]; n = len(o); nC = len(CC["k"])
    H = np.searchsorted(t, T + CC["hold"] * H1, "left"); H = np.maximum(np.minimum(H, n), 1)
    Hmax = int(H.max())
    tt, oo, hh, ll, cc = t[:Hmax], o[:Hmax], h[:Hmax], l[:Hmax], c[:Hmax]
    sl = min(1.5 * atrp, 6.0)
    with np.errstate(invalid="ignore"):
        tp = np.minimum(CC["k"] * atrp, CC["cap"])
    S0 = E0 * (1 + sl / 100)
    valid = np.arange(Hmax)[None, :] < H[:, None]
    hs1 = hh >= S0
    fs1 = int(np.argmax(hs1)) if hs1.any() else 10**9
    TPp = E0 * (1 - tp / 100)
    thrp = np.where(CC["thr"] > 0, TPp * (1 - CC["thr"] / 100), TPp - tick)
    ht = (ll[None, :] <= thrp[:, None]) & valid
    BIG = 10**9
    fs = np.where(fs1 < H, fs1, BIG); ft = np.where(ht.any(1), ht.argmax(1), BIG)
    isstop = (fs < BIG) & (fs <= ft); istp = (~isstop) & (ft < BIG)
    last = H - 1
    X = np.where(isstop, max(S0, oo[min(fs1, Hmax - 1)]), np.where(istp, TPp, cc[last]))
    col = np.where(isstop, fs, np.where(istp, ft, last))
    k = np.where(isstop, 1, np.where(istp, 0, 2)).astype(np.int8)
    r = -(X / E0 - 1) * 100
    xm = tt[np.minimum(col, Hmax - 1)] + nb * MIN
    H24 = int(min(np.searchsorted(t, T + 24 * H1, "left"), n)); H24 = max(H24, 1)
    mfe = (1 - l[:H24].min() / E0) * 100; mae = (h[:H24].max() / E0 - 1) * 100
    # MFE before the stop (how far it went our way before today's stop hit, within 24 h)
    stop_at = fs1 if fs1 < H24 else H24
    mfe_bs = (1 - l[:max(stop_at, 1)].min() / E0) * 100 if stop_at > 0 else 0.0
    z = np.zeros(nC, np.int8)
    return (r.astype(np.float32), k, xm.astype(np.int64), z, z,
            np.full(nC, mfe, np.float32), np.full(nC, mae if fs1 >= H24 else -mfe_bs, np.float32))


S.sim_trade = sim_trade; S.cell_arrays = cell_arrays


def rows():
    os.makedirs(D, exist_ok=True)
    F = pd.read_pickle(os.path.join(RP, "EXH_EXIT2_data", "rows_EXH.pkl")).reset_index(drop=True)
    F.to_pickle(os.path.join(D, "rows_FRENZY.pkl"))
    L = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXH_WIDE_L4_2026-11-18.csv.gz"))
    bk = set(L[L.set == "BASE"].key); A = L[(L.set == "C") & ~L.key.isin(bk)]
    blk = np.isfinite(A.oi_v24) & (A.ssc > -0.0366) & (A.oi_v24 > 0.128); A = A[~blk].reset_index(drop=True)
    half = pd.to_datetime(A.entry_ms, unit="ms").map(lambda d: f"{d.year}-H{1 if d.month <= 6 else 2}")
    W = pd.DataFrame(dict(xi=np.arange(len(A)), pair=A.pair, entry_ms=A.entry_ms.astype("int64"), atrp=A.atrp, q5=A.q5,
                          slip=XD.slip_of(A.q5), half=half, base_r=A.EX_r, base_kind=A.EX_k, base_exit=A.EX_x.astype("int64"),
                          base_f=A.EX_f))
    W.to_pickle(os.path.join(D, "rows_WIDE.pkl"))
    E = pd.read_pickle(os.path.join(RP, "EXH_EXIT2_data", "rows_E.pkl")).reset_index(drop=True)
    E.to_pickle(os.path.join(D, "rows_E.pkl"))
    print("FRENZY", len(F), "WIDE", len(W), "E", len(E))


def one_1m(pd_):
    p, d = pd_
    fo = os.path.join(K1_MINE, p, d + ".csv")
    if os.path.exists(fo):
        return 0
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    z = LD.get(f"{XD.VIS}/daily/klines/{p}/1m/{p}-1m-{d}.zip")
    if not z:
        open(fo, "w").close(); return 0
    zz = zipfile.ZipFile(io.BytesIO(z)); lines = zz.open(zz.namelist()[0]).read().decode().strip().split("\n")
    if lines and not lines[0][0].isdigit():
        lines = lines[1:]
    open(fo, "w").write("\n".join(["t,o,h,l,c"] + [",".join(ln.split(",")[:5]) for ln in lines])); return 1


def k1m():
    need = set()
    for tag in ("FRENZY", "WIDE", "E"):
        x = pd.read_pickle(os.path.join(D, f"rows_{tag}.pkl"))
        for p, t in zip(x.pair, x.entry_ms):
            for kk in range(3):
                need.add((p, pd.to_datetime(int(t) + kk * DAY, unit="ms").strftime("%Y-%m-%d")))
    nd = sorted(x for x in need if not XD.have(*x) and not os.path.exists(os.path.join(K1_MINE, x[0], x[1] + ".csv")))
    print("need", len(need), "missing", len(nd), flush=True)
    n = 0
    with cf.ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(one_1m, nd)):
            n += r
            if i % 300 == 0:
                print(time.strftime("%H:%M:%S"), i, len(nd), n, flush=True)
    print("fetched", n, "of", len(nd))


def work(args):
    p, rws, CL = args
    out, _ = S.run_rows(p, rws, CL)
    return p, out


def sim(tag, procs):
    X = pd.read_pickle(os.path.join(D, f"rows_{tag}.pkl"))
    CL = cells()
    jobs = [(p, g, CL) for p, g in X.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
    res = []; t0 = time.time()
    with mp.get_context("fork").Pool(procs) as pool:
        for i, (p, out) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            res.append(out)
            if i % 100 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), p, f"{time.time() - t0:.0f}s", flush=True)
    Z = {k: np.concatenate([r_[k] for r_ in res]) for k in res[0]}
    o = np.argsort(Z["xi"]); Z = {k: v[o] for k, v in Z.items()}
    np.savez_compressed(os.path.join(D, f"sim_{tag}.npz"), names=np.array([c["name"] for c in CL]), **Z)
    print("done", tag, len(Z["xi"]), "ok", int(Z["ok"].sum()), "1m share", Z["src1m"][Z["ok"]].mean(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("tag", nargs="?"); ap.add_argument("--procs", type=int, default=7)
    A = ap.parse_args()
    {"rows": rows, "k1m": k1m}.get(A.step, lambda: sim(A.tag, A.procs))()
