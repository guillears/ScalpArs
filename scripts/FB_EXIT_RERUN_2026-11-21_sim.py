#!/usr/bin/env python3
"""FB_EXIT_RERUN_2026-11-21 — capped peak-stop grid (PREREG reports/FB_EXIT_RERUN_2026-11-21_PREREG.md). Research only.
Imports (does not edit) the first study's walker / loaders. Backtest 545 (5m) + real kept fills (KEPT_LIST_FLATBASE_2026-11-21.csv, 1m).
Output: reports/FB_EXIT_RERUN_2026-11-21_data/{bt.csv, real.csv}
Usage: venv/bin/python -I scripts/FB_EXIT_RERUN_2026-11-21_sim.py [--procs 6]"""
import argparse, importlib.util, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
D1 = os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_data")
OUT = os.path.join(ROOT, "reports", "FB_EXIT_RERUN_2026-11-21_data")
HR, MIN = 3_600_000, 60_000


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


K = _imp("FBT", "FB_THEORY_EXIT_2026-11-21_common.py")
SB = _imp("FBS", "FB_THEORY_EXIT_2026-11-21_sim_bt.py")
CELLS = [("REF", None, None), ("PKC12", "PK", 12.0), ("PKC15", "PK", 15.0), ("PKC20", "PK", 20.0), ("C_PK1", "PK", 25.0),
         ("K_S12", "FIX", 12.0), ("K_S15", "FIX", 15.0), ("K_S20", "FIX", 20.0)]


def stop_d(kind, cap, E, peak):
    if kind is None:
        return 10.0
    if kind == "FIX":
        return cap
    if peak is None or not np.isfinite(peak):
        return 10.0
    lv = peak * 1.01
    if lv < E * 1.01:
        return 10.0
    return min((lv / E - 1) * 100, cap)


def p_bt(args):
    pair, g, AN = args
    F = SB.G["F"]
    x = SB.load(pair)
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
        _, miss = SB.OC.fund(F, pair, te, end_t)
        Fb = np.zeros(len(t)) if (miss or fx is None or not len(fx[0])) else SB.FC.settle_on_bars(t, o, E, te, fx[0], fx[3], 5, markscale=True)
        pk = AN.get(int(r.rid))
        for mode, sx in (("row", float(r.slip_x)), ("s20", 0.20)):
            for nm, kind, cap in CELLS:
                d = stop_d(kind, cap, E, pk)
                w = K.run_cell(dict(name=nm, fam="K", sl=d), t, o, h, l, c, i, E, end_t, Fb, None, sx, SB.RC.TAKER, SB.RC.TAKER, SB.RC.MAKER, 5, {}, int(t[i]))
                if w is None:
                    continue
                extra = SB.OC.fund(F, pair, te, int(t[w["j"]]))[0] if miss else 0.0
                out.append(dict(rid=r.rid, pair=pair, te=te, cell=nm, mode=mode, d=d, net=w["net"] + extra, why=w["why"],
                                xt=int(t[w["j"]]) + 300_000))
    return out


def run_bt(procs):
    X = SB.rows()
    AN = pd.read_csv(os.path.join(D1, "bt_anchors.csv"))
    pk = {int(r.rid): (float(r.peak) if r.anc is True or str(r.anc) == "True" else np.nan) for r in AN.itertuples()}
    SB.G["F"] = SB.OC.funding_table()
    jobs = [(p, g, {k: pk.get(int(k), np.nan) for k in g.rid}) for p, g in X.groupby("pair")]
    res = []
    with mp.get_context("fork").Pool(procs) as pool:
        for o in pool.imap_unordered(p_bt, jobs, chunksize=2):
            res += o
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "bt.csv"), index=False)
    ref = R[(R.cell == "REF") & (R["mode"] == "row")].merge(X[["rid", "Bp_net", "Bp_why"]], on="rid")
    ok = ((ref.net - ref.Bp_net).abs() < 0.01) & (ref.why == ref.Bp_why)
    print(f"PARITY REF vs Bp: {int(ok.sum())} / {len(X)}", flush=True)
    old = pd.read_csv(os.path.join(D1, "bt_cells.csv"))
    for nm in ("C_PK1", "K_S12", "K_S15", "K_S20"):
        a = R[(R.cell == nm)].merge(old[old.cell == nm], on=["rid", "mode"], suffixes=("", "_o"))
        print(f"  {nm} vs first study: {int(((a.net - a.net_o).abs() < 1e-6).sum())} / {len(a)} identical", flush=True)


def run_real():
    RL = _imp("FBR", "FB_THEORY_EXIT_2026-11-21_real.py")
    KL = pd.read_csv(os.path.join(ROOT, "reports", "KEPT_LIST_FLATBASE_2026-11-21.csv"))
    X = RL.cohort()
    X = X.merge(KL, on=["pair", "k"], how="inner")
    assert len(X) == len(KL) == 20, len(X)
    rows = []
    for _, r in X.iterrows():
        p = r.pair; te = RL.ms(r.opened_at); sp = RL.ms(r.entry_frenzy_spike_at); E = float(r.entry_price)
        a = min(te - 26 * HR, sp - HR) // MIN * MIN
        raw = RL.k1m(p, a, te + 72 * HR + 10 * MIN)
        g = RL.SC.grid(raw, a, min(te + 72 * HR + 10 * MIN, int(raw["t"][-1]) + MIN))
        T, o, h, l, c = g["t"], g["o"], g["h"], g["l"], g["c"]
        i0 = int(np.searchsorted(T, (te // MIN) * MIN + MIN)); ifill = int(np.searchsorted(T, (te // MIN) * MIN))
        on = int(np.searchsorted(T, sp - 5 * MIN))
        peak = float(h[on:ifill].max())
        fs_t, fs_r = RL.funding(p)
        F = RL.SC.fund_cum(T, o, E, -1, fs_t, fs_r, te)
        end_t = te + 72 * HR
        for nm, kind, cap in CELLS:
            d = stop_d(kind, cap, E, peak)
            w = K.run_cell(dict(name=nm, fam="K", sl=d), T, o, h, l, c, i0, E, end_t, F, None, RL.SLIP, RL.TAKER, RL.TAKER, RL.MAKER, 1, {}, te)
            rows.append(dict(pair=p, k=r.k, block=r.block, te=te, E=E, peak_d=(peak / E - 1) * 100, booked=r.booked, reason=r.close_reason,
                             notional=r.get("notional_value", np.nan), data_end=int(T[-1]), cell=nm, d=d, net=w["net"], why=w["why"],
                             xt=int(T[w["j"]]) + MIN, fund=w["fund"]))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "real.csv"), index=False)
    ref = R[R.cell == "REF"]; post = ref[ref.k >= "2026-10-28"]
    m = ((post.why == "TP") == post.reason.str.contains("TP")).sum()
    print(f"real REF vs booked (>= 10-28): exit kind match {m} / {len(post)} · mean |net - booked| {(post.net - post.booked).abs().mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--skip_bt", action="store_true")
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    run_real()
    if not A.skip_bt:
        run_bt(A.procs)
