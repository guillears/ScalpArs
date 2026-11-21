#!/usr/bin/env python3
"""FUNDING_STOP_2026-11-21 — BACKTEST cohorts: arm A (price stop, today) vs B (price + accrued funding) vs Bp (paid-only), per sleeve.

  FB   = the OOS_REAL_FLATBASE rebuild (reports/OOS_REAL_FLATBASE_data/wl_features.csv), NEAR_BASE kept (dist_base > 2.09, X0_why TP / STOP /
         TIME) = 614, today's stack drops the DL 601 funding-above-base entries (funding_rate_pct > 0.005) — the TP_GRID_2026-11-20_FB cohort.
         5m paths (oos_real_flatbase_recon W.load), entry = cf_entry_px at the cf_entry_t bar, per-trade exit slip slip_x, taker 0.045 /
         maker 0.018 % (rebuild convention), TP 5 / SL 10 / 72 h. Funding: oos_real_flatbase_common.funding_table.
  EXH  = TP_GRID_2026-11-20_EXH rows_FRENZY (3,368 fills: BASELINE_TRADES_EXHAUSTION rows kept by DL 582, one position per pair) on the
         EXH_EXIT2 1m loader (5m fallback flagged), TP min(20 × ATR15, 10) / SL min(1.5 × ATR15, 6) / 24 h, slip = the row's slip (≥ 0.10).
  DIST = the DL 586 kept cohort reports/ENTRY_WL_2026-11-18_data/dist.pkl (1,547) on the DIST_EXIT 1m loader, same exit as EXH.
Walk (scripts/FUNDING_STOP_2026-11-21_common.walk): from the entry bar (a backtest fill is at the bar OPEN), stop first on the HIGH, maker TP
filled only ≥ 0.05 % through (primary), time exit at the last close; EXH / DIST costs = the cohort's own: entry slip + 0.05 taker, exit taker
0.05 + slip (stop / time) or maker 0.02. Funding mark-scaled per settlement. A pair-month without funding data → that fill carries the cohort's
own imputed funding total in BOTH arms and B = A (Δ = 0, counted 'no funding data').
Parity (arm A in each cohort's own TP rule + plain rate × 100 funding over its own window) vs the cohort's recorded result.
Output: reports/FUNDING_STOP_2026-11-21_data/bt_<SLEEVE>.csv
Usage: venv/bin/python -I scripts/FUNDING_STOP_2026-11-21_bt.py FB|EXH|DIST [--procs 6]"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data")
HR, MIN, DAY = 3_600_000, 60_000, 86_400_000


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


C = _imp("FSC", "FUNDING_STOP_2026-11-21_common.py")
G = {}


# ───────────────────────────── FLATBASE (5m rebuild) ─────────────────────────────
def fb_setup():
    import oos_real_flatbase_common as OC
    import oos_real_flatbase_recon as RC
    G.update(OC=OC, RC=RC, F=OC.funding_table())


def fb_rows():
    X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09)].copy().reset_index(drop=True)
    X["rid"] = X.index
    return X


def fb_pair(args):
    pair, g = args
    OC, RC, F = G["OC"], G["RC"], G["F"]
    x = RC.W.load(pair)
    out = []
    if x is None:
        return out
    t, o, h, l, c = [np.asarray(a) for a in x[:5]]
    t = t.astype("int64")
    tidx = {int(tt): k for k, tt in enumerate(t)}
    fx = F.get(pair)
    for r in g.itertuples():
        i = tidx.get(int(r.cf_entry_t))
        if i is None:
            continue
        E = float(r.cf_entry_px); sx = float(r.slip_x); te = int(r.cf_entry_t) + 10_000
        end_t = int(t[i]) + 72 * HR
        tot_i, miss = OC.fund(F, pair, te, end_t)
        if miss or fx is None or not len(fx[0]):
            Fb = np.zeros(len(t)); fs_t = fs_r = None
        else:
            fs_t, fs_r = fx[0], fx[3]
            Fb = C.settle_on_bars(t, o, E, te, fs_t, fs_r, 5, markscale=True)
        d = dict(rid=r.rid, pair=pair, te=int(r.cf_entry_t), E=E, miss=miss, entry_rate=r.funding_rate_pct / 100,
                 interval_h=C.interval_h(fs_t, te) if fs_t is not None else np.nan, tp=5.0, sl=10.0)
        for arm in ("A", "B", "Bp"):
            w = C.walk(t, o, h, l, c, i, E, 5.0, 10.0, end_t, Fb, arm, sx, 0.05, RC.TAKER, RC.TAKER, RC.MAKER, "through")
            extra = 0.0
            if miss:   # imputed total over the arm's own hold (both arms; B = A)
                extra = OC.fund(F, pair, te, int(t[w["j"]]))[0]
            d.update({f"{arm}_net": w["net"] + extra, f"{arm}_why": w["why"], f"{arm}_xt": int(t[w["j"]]), f"{arm}_fund": w["fund"] + extra})
        p = C.walk(t, o, h, l, c, i, E, 5.0, 10.0, end_t, np.zeros(len(t)), "A", sx, 0.0, RC.TAKER, RC.TAKER, RC.MAKER, "touch")
        d.update(par_net=p["net"] + OC.fund(F, pair, te, int(t[p["j"]]) + 300_000)[0], par_why=p["why"], ref_net=r.X0_net, ref_why=r.X0_why)
        out.append(d)
    return out


# ───────────────────────────── EXH / DIST (1m engine loaders) ─────────────────────────────
def ed_setup(sleeve):
    if sleeve == "EXH":
        M = _imp("EXHSIM", "TP_GRID_2026-11-20_EXH_sim.py")     # patches exh_exit2_sim loaders (+ its own k1m dir)
        XS = M.XS
        G.update(load1m=XS.load1m, TB=XS.TB, NS=XS.S)
    else:
        import dist_exit_sim as DE
        G.update(load1m=DE.load1m, TB=DE.NS.TB, NS=DE.NS)


def ed_rows(sleeve):
    if sleeve == "EXH":
        S = pd.read_pickle(os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_EXH_data", "rows_FRENZY.pkl")).reset_index(drop=True)
        z = np.load(os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_EXH_data", "sim_FRENZY.npz"), allow_pickle=True)
        j = list(z["names"]).index("REF")
        ref = pd.DataFrame(dict(xi=z["xi"], ref_r=z["r"][:, j], ref_k=z["k"][:, j], ref_f=z["f"][:, j], ref_ok=z["ok"]))
        S = S.merge(ref, on="xi", how="left")
        return pd.DataFrame(dict(rid=S.index, pair=S.pair.astype(str), entry_ms=S.entry_ms.astype("int64"), atrp=S.atrp.astype(float),
                                 slip=S.slip.astype(float), ref_r=S.ref_r, ref_k=S.ref_k, ref_f=S.ref_f))
    S = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl")).reset_index(drop=True)
    return pd.DataFrame(dict(rid=S.index, pair=S.pair.astype(str), entry_ms=S.entry_ms.astype("int64"), atrp=S.atr15_pct.astype(float),
                             slip=S.slip.astype(float), ref_net=S.net.astype(float), ref_k=S.k))


def ed_pair(args):
    p, g = args
    load1m, TB, NS = G["load1m"], G["TB"], G["NS"]
    days = sorted({int(T) // DAY + kk for T in g.entry_ms.values for kk in range(3)})
    one, okd = load1m(p, days)
    five = None
    ft, fc, months = TB.funding_for(p)
    ft = np.asarray(ft, "int64"); fc = np.asarray(fc, float)
    rates = np.diff(fc) if len(fc) > 1 else np.zeros(0)
    out = []
    for r in g.itertuples():
        T = int(r.entry_ms)
        need = {T // DAY + kk for kk in range(2)}
        nb = 1
        use1 = one is not None and need <= okd
        if use1:
            t1 = one[0]; a = int(np.searchsorted(t1, T)); use1 = a < len(t1) and t1[a] == T
        if use1:
            b = int(np.searchsorted(t1, T + 26 * HR))
            t, o, h, l, c = [x[a:b] for x in one]
        else:
            if five is None:
                five = TB.load(p)
            if five is None:
                continue
            t5 = np.asarray(five[0], "int64"); a = int(np.searchsorted(t5, T))
            if a >= len(t5) or t5[a] != T:
                continue
            b = int(np.searchsorted(t5, T + 26 * HR))
            t, o, h, l, c = t5[a:b], *[np.asarray(five[k][a:b], float) for k in (1, 2, 3, 4)]
            nb = 5
        if len(o) < 2:
            continue
        E = float(o[0]); end_t = T + 24 * HR
        sl = min(1.5 * r.atrp, 6.0); tp = min(20 * r.atrp, 10.0)
        mE = pd.Timestamp(T, unit="ms").strftime("%Y-%m"); mX = pd.Timestamp(end_t, unit="ms").strftime("%Y-%m")
        miss = (mE not in months) or (mX not in months)
        if miss:
            Fb = np.zeros(len(t))
        else:
            Fb = C.settle_on_bars(t, o, E, T, ft, rates, nb, markscale=True)
        tick = NS.tick_of(o, h, l, c)
        d = dict(rid=r.rid, pair=p, te=T, E=E, miss=miss, nb=nb, tp=tp, sl=sl,
                 entry_rate=C.last_rate(ft, rates, T) if not miss else np.nan,
                 interval_h=C.interval_h(ft, T) if not miss else np.nan)
        for arm in ("A", "B", "Bp"):
            w = C.walk(t, o, h, l, c, 0, E, tp, sl, end_t, Fb, arm, r.slip, 0.05, r.slip + 0.05, 0.05, 0.02, "through")
            extra = 0.0
            if miss:
                extra = -(int(t[w["j"]]) - T) / (8 * HR) * TB.FUND_MISS
            d.update({f"{arm}_net": w["net"] + extra, f"{arm}_why": w["why"], f"{arm}_xt": int(t[w["j"]]), f"{arm}_fund": w["fund"] + extra})
        p_ = C.walk(t, o, h, l, c, 0, E, tp, sl, end_t, np.zeros(len(t)), "A", 0.0, 0.0, 0.0, 0.0, 0.0, "tick", tick=tick)
        xm = int(t[p_["j"]]) + nb * MIN
        fpl = ((fc[np.searchsorted(ft, xm, "right")] - fc[np.searchsorted(ft, T, "right")]) * 100) if len(ft) else 0.0
        if miss:
            fpl = -(xm - T) / (8 * HR) * TB.FUND_MISS
        d.update(par_gross=p_["gross"], par_why=p_["why"], par_fund=fpl)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sleeve"); ap.add_argument("--procs", type=int, default=6)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if A.sleeve == "FB":
        fb_setup(); X = fb_rows(); fn = fb_pair
    else:
        ed_setup(A.sleeve); X = ed_rows(A.sleeve); fn = ed_pair
    jobs = list(X.groupby("pair"))
    res = []
    with mp.get_context("fork").Pool(A.procs) as pool:
        for k, o in enumerate(pool.imap_unordered(fn, jobs, chunksize=2)):
            res += o
            if k % 100 == 0:
                print(k, "/", len(jobs), flush=True)
    R = pd.DataFrame(res).merge(X.drop(columns=[c for c in ("pair",) if c in X.columns]), on="rid", how="left", suffixes=("", "_src"))
    R.to_csv(os.path.join(OUT, f"bt_{A.sleeve}.csv"), index=False)
    print(A.sleeve, "rows", len(R), "of", len(X), "· no-funding-data", int(R.miss.sum()))
    if A.sleeve == "FB":
        g = (R.par_net - R.ref_net).abs()
        print("parity vs X0_net: mean |gap|", round(g.mean(), 4), "max", round(g.max(), 4), "exact(<0.001)", int((g < 1e-3).sum()),
              "why match", int((R.par_why == R.ref_why).sum()))
    elif A.sleeve == "EXH":
        km = {0: "TP", 1: "STOP", 2: "TIME"}
        gr = (R.par_gross - R.ref_r).abs(); gf = (R.par_fund - R.ref_f).abs()
        print("parity vs sim_FRENZY REF: gross mean |gap|", round(gr.mean(), 4), "max", round(gr.max(), 4), "· funding mean |gap|", round(gf.mean(), 4),
              "· kind match", int((R.par_why == R.ref_k.map(km)).sum()), "/", len(R))
    else:
        km = {0: "TP", 1: "STOP", 2: "TIME"}
        net = R.par_gross + R.par_fund - R.slip - np.where(R.par_why == "TP", 0, R.slip) - 0.05 - np.where(R.par_why == "TP", 0.02, 0.05)
        g = (net - R.ref_net).abs()
        print("parity vs dist.pkl net: mean |gap|", round(g.mean(), 4), "median", round(g.median(), 4), "max", round(g.max(), 4),
              "· kind match", int((R.par_why == R.ref_k.map(km)).sum()), "/", len(R))


if __name__ == "__main__":
    main()
