#!/usr/bin/env python3
"""FB_THEORY_EXIT_2026-11-21 — BACKTEST sim (PREREG reports/FB_THEORY_EXIT_2026-11-21_PREREG.md). Research only.
Cohort = reports/FUNDING_STOP_2026-11-21_data/bt_FB.csv minus entry_rate > 0.00005 (DL 601) = 545 fills on today's stack.
5m paths = oos_real_flatbase_recon W.load; funding = oos_real_flatbase_common.funding_table, assigned to bars exactly like
FUNDING_STOP_2026-11-21_bt.fb_pair (mark-scaled). REF must reproduce Bp_net / Bp_why (parity printed; gate ≥ 99 %).
Pass 1 = anchors (BASE / PEAK / VWAP / CFHI) → the K_M* control medians; pass 2 = every cell, at the row's slip_x and at a flat 0.20 % slip.
Output: reports/FB_THEORY_EXIT_2026-11-21_data/bt_cells.csv (one row per fill × cell × slip mode) + bt_anchors.csv
Usage: venv/bin/python -I scripts/FB_THEORY_EXIT_2026-11-21_sim_bt.py [--procs 6]"""
import argparse, importlib.util, json, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_data")
HR = 3_600_000


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


FC = _imp("FSC", "FUNDING_STOP_2026-11-21_common.py")
K = _imp("FBT", "FB_THEORY_EXIT_2026-11-21_common.py")
import oos_real_flatbase_common as OC  # noqa: E402
import oos_real_flatbase_recon as RC   # noqa: E402
G = {}


def rows():
    X = pd.read_csv(os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data", "bt_FB.csv"), low_memory=False)
    X = X[~(X.entry_rate > 0.00005)].copy().reset_index(drop=True)
    return X[["rid", "pair", "te", "E", "miss", "slip_x", "on_t", "spike_ts", "cf_entry_t", "cf_entry_px", "Bp_net", "Bp_why", "Bp_xt",
              "dist_base"]].copy()


def load(pair):
    x = RC.W.load(pair)
    if x is None:
        return None
    t, o, h, l, c, v = [np.asarray(a) for a in x[:6]]
    return t.astype("int64"), o.astype(float), h.astype(float), l.astype(float), c.astype(float), v.astype(float)


def anchors_of(t, o, h, l, c, v, on, i):
    if on is None or on < 6 or i <= on:
        return None
    tp = (h[on:i] + l[on:i] + c[on:i]) / 3.0
    q = v[on:i] * tp
    return dict(base=float(c[on - 6]), peak=float(h[on:i].max()), vwap=float((tp * q).sum() / q.sum()) if q.sum() > 0 else np.nan,
                cfhi=float(h[max(i - 48, 0):i].max()))


def p1(args):
    pair, g = args
    x = load(pair)
    out = []
    if x is None:
        return out
    t, o, h, l, c, v = x
    tidx = {int(tt): k for k, tt in enumerate(t)}
    for r in g.itertuples():
        i = tidx.get(int(r.cf_entry_t)); on = tidx.get(int(r.on_t))
        an = anchors_of(t, o, h, l, c, v, on, i) if i is not None else None
        d = dict(rid=r.rid, ok=i is not None, anc=an is not None)
        if an:
            d.update(an)
        out.append(d)
    return out


def p2(args):
    pair, g, AN, kmed = args
    F = G["F"]; CL = K.cells()
    x = load(pair)
    out = []
    if x is None:
        return out
    t, o, h, l, c, v = x
    tidx = {int(tt): k for k, tt in enumerate(t)}
    fx = F.get(pair)
    for r in g.itertuples():
        i = tidx.get(int(r.cf_entry_t))
        if i is None:
            continue
        E = float(r.cf_entry_px); te = int(r.cf_entry_t) + 10_000; end_t = int(t[i]) + 72 * HR
        tot_i, miss = OC.fund(F, pair, te, end_t)
        if miss or fx is None or not len(fx[0]):
            Fb = np.zeros(len(t))
        else:
            Fb = FC.settle_on_bars(t, o, E, te, fx[0], fx[3], 5, markscale=True)
        an = AN.get(r.rid)
        for mode, sx in (("row", float(r.slip_x)), ("s20", 0.20)):
            for cl in CL:
                w = K.run_cell(cl, t, o, h, l, c, i, E, end_t, Fb, an, sx, RC.TAKER, RC.TAKER, RC.MAKER, 5, kmed, int(t[i]))
                if w is None:
                    continue
                extra = OC.fund(F, pair, te, int(t[w["j"]]))[0] if miss else 0.0
                out.append(dict(rid=r.rid, cell=cl["name"], mode=mode, net=w["net"] + extra, why=w["why"], xt=int(t[w["j"]]) + 300_000,
                                fund=w["fund"] + extra))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    X = rows()
    print("cohort", len(X), flush=True)
    jobs = list(X.groupby("pair"))
    res = []
    with mp.get_context("fork").Pool(A.procs) as pool:
        for o in pool.imap_unordered(p1, jobs, chunksize=2):
            res += o
    AN = pd.DataFrame(res)
    AN = X[["rid", "pair", "cf_entry_t", "E", "dist_base"]].merge(AN, on="rid", how="left")
    AN.to_csv(os.path.join(OUT, "bt_anchors.csv"), index=False)
    E = AN.E.values
    kmed = {}
    for key, b in (("PEAK", 0.0), ("VWAP", 0.5), ("CFHI", 0.5)):
        ds = [K.struct_dist(key, b, e, dict(peak=pk, vwap=vw, cfhi=cf)) for e, pk, vw, cf in zip(E, AN.peak, AN.vwap, AN.cfhi)]
        ds = np.array([d for d in ds if d is not None])
        kmed[key] = float(np.median(ds)); kmed[key + "_n"] = int(len(ds))
    json.dump(kmed, open(os.path.join(OUT, "kmed.json"), "w"), indent=1)
    print("anchored", int(AN.anc.fillna(False).sum()), "kmed", kmed, flush=True)
    andict = {int(r.rid): dict(base=r.base, peak=r.peak, vwap=r.vwap, cfhi=r.cfhi) for r in AN.itertuples() if r.anc is True}
    G["F"] = OC.funding_table()
    jobs = [(p, g, {k: andict[k] for k in g.rid if k in andict}, kmed) for p, g in X.groupby("pair")]
    res = []; t0 = time.time()
    with mp.get_context("fork").Pool(A.procs) as pool:
        for k, o in enumerate(pool.imap_unordered(p2, jobs, chunksize=2)):
            res += o
            if k % 100 == 0:
                print(time.strftime("%H:%M:%S"), k, "/", len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    Rz = pd.DataFrame(res)
    Rz.to_csv(os.path.join(OUT, "bt_cells.csv"), index=False)
    ref = Rz[(Rz.cell == "REF") & (Rz["mode"] == "row")].merge(X[["rid", "Bp_net", "Bp_why"]], on="rid")
    gap = (ref.net - ref.Bp_net).abs()
    ok = (gap < 0.01) & (ref.why == ref.Bp_why)
    print(f"PARITY REF vs Bp: {int(ok.sum())} / {len(X)} ({ok.sum() / len(X) * 100:.2f} %) · mean |gap| {gap.mean():.5f} · max {gap.max():.4f}"
          f" · why match {int((ref.why == ref.Bp_why).sum())}", flush=True)


if __name__ == "__main__":
    main()
