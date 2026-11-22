#!/usr/bin/env python3
"""MAX_HOLD_STUDY_2026-11-21 — FRENZY_WILLY_HT and FRENZY_WILLY max-hold grids (research only; no bot code / config / git touched).

Grid FROZEN before any result was read:
  WILLY_HT : cap h ∈ {4, 6, 8, 12, 16, 24 (today), 36}  — today's exit: maker TP +6 net (DL 608) / stop −4 net / 24 h, no late TP.
  WILLY    : cap min ∈ {45, 60, 90 (today), 120, 180}   — today's exit (fast_tp_sim.W_CUR: TP 1.25, high target, late TP 0.25 at 30 min,
             runner, stop −4) with only HOLD changed.
Engines imported, never edited: scripts/HT_THEORY_EXIT_2026-11-21_sim.py (bars, run_ht, lp_net, stop_px, fund_long — the DL 608 study's
engine) and fast_tp_sim.willy_ref. Cohorts = HT_THEORY_EXIT sim.pkl (HT: the HT_EXIT_RERUN set — REAL master / batch + backtest 300;
WILLY: REAL master 85 (23 kept on today's stack) + backtest 545). A time exit = market at the open of the bar at entry + cap
(run_ht) / willy_ref's MAX_HOLD_TIME, taker 0.05 % + slip 0.10 %; funding = the study's fund_long to the exit minute.
36 h for HT is walked only if some fill is still open at 24 h (none is → Δ ≡ 0, stated in the report).
Parity: HT 24 h = HT_EXIT_RERUN G6; WILLY 90 min = sim.pkl REF.
Output: reports/MAX_HOLD_STUDY_2026-11-21_data/willy.pkl
Usage: venv/bin/python -I scripts/MAX_HOLD_STUDY_2026-11-21_willy.py [--procs 8]"""
import argparse
import importlib.util
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
_sp = importlib.util.spec_from_file_location("HTS", os.path.join(HERE, "HT_THEORY_EXIT_2026-11-21_sim.py"))
HTS = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(HTS)
S = HTS.S
OUT = os.path.join(ROOT, "reports", "MAX_HOLD_STUDY_2026-11-21_data")
MIN, HR = 60_000, 3_600_000
HT_CAPS = (4, 6, 8, 12, 16, 24)
W_CAPS = (45, 60, 90, 120, 180)


def ht_one(fl):
    p, te, E = fl["pair"], int(fl["te"]), float(fl["E"])
    sp = int(fl["spike_ms"]) if fl["spike_ms"] == fl["spike_ms"] else te - HR
    a = (min(te, sp) - 60 * MIN) // MIN * MIN
    g = HTS.bars(p, a, te + 25 * HR, fresh=(fl["kind"] == "REAL"))
    if g is None:
        return None
    t = g["t"]
    T0 = (te // MIN) * MIN + (MIN if fl["start"] == "next" else 0)
    i0 = int(np.searchsorted(t, T0))
    if i0 >= len(t) or t[i0] - T0 > 5 * MIN:
        return None
    tick = S.tick_for(p, E)
    tp, sl = HTS.lp_net(E, 6.0, HTS.MAKER, tick), HTS.stop_px(E, 4.0)
    out = {}
    for H in HT_CAPS:
        t_end = T0 + H * HR
        jend = int(np.searchsorted(t, t_end, "left"))
        has_end = jend < len(t) and t[jend] - t_end < 30 * MIN
        r = HTS.run_ht(g, i0, jend, E, tick, tp, sl, 0.10, HTS.TAKER, has_end=has_end)
        if r is None:
            out.update({f"c{H}_net": np.nan, f"c{H}_why": "NA", f"c{H}_hold": np.nan, f"c{H}_fund": np.nan}); continue
        net0, j, kind = r
        xm = int(t[j]) + MIN
        fu = HTS.fund_long(fl, xm)
        out.update({f"c{H}_net": net0 + fu, f"c{H}_why": kind, f"c{H}_hold": (xm - T0) / HR, f"c{H}_fund": fu})
    return out


def willy_one(fl):
    p, te, E = fl["pair"], int(fl["te"]), float(fl["E"])
    sp = int(fl["spike_ms"]) if fl["spike_ms"] == fl["spike_ms"] else te - HR
    a = (min(te, sp) - 60 * MIN) // MIN * MIN
    g = HTS.bars(p, a, te + 4 * HR, fresh=(fl["kind"] == "REAL"))
    if g is None:
        return None
    t = g["t"]
    T0 = (te // MIN) * MIN + (MIN if fl["start"] == "next" else 0)
    i0 = int(np.searchsorted(t, T0))
    if i0 >= len(t) or t[i0] != T0:
        return None
    tick = S.tick_for(p, E)
    live = E / (1 + 0.10 / 100.0)
    H = float(fl["H"]); ra = fl.get("run_atr")
    ra = float(ra) if ra is not None and ra == ra else None
    out = {}
    for cap in W_CAPS:
        j1 = min(len(t), int(np.searchsorted(t, T0 + (cap + 2) * MIN, "left")) + 1)
        if j1 - i0 < cap:
            out.update({f"c{cap}_net": np.nan, f"c{cap}_why": "SHORT", f"c{cap}_hold": np.nan, f"c{cap}_fund": np.nan}); continue
        sl_ = (t[i0:j1], g["o"][i0:j1], g["h"][i0:j1], g["l"][i0:j1], g["c"][i0:j1])
        y = S.willy_ref(*sl_, int(t[i0]), live, H, ra, dict(S.W_CUR, HOLD=int(cap)), tick, slip=0.10)
        if y is None:
            out.update({f"c{cap}_net": np.nan, f"c{cap}_why": "NA", f"c{cap}_hold": np.nan, f"c{cap}_fund": np.nan}); continue
        net0, jj, why, tu, E_ = y
        xm = int(t[i0 + jj]) + MIN
        fu = HTS.fund_long(fl, xm)
        out.update({f"c{cap}_net": net0 + fu, f"c{cap}_why": why, f"c{cap}_hold": (xm - T0) / HR, f"c{cap}_fund": fu})
    return out


def work(fl):
    try:
        o = ht_one(fl) if fl["sleeve"] == "HT" else willy_one(fl)
        return fl["fid"], o
    except Exception as e:  # noqa: BLE001
        return fl["fid"], {"err": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    O = pd.read_pickle(os.path.join(HTS.D, "sim.pkl"))
    O = O[O.REF.notna()].copy()
    cols = ["fid", "sleeve", "pair", "te", "E", "spike_ms", "start", "kind", "H", "run_atr"]
    recs = O[cols].to_dict("records")
    with mp.Pool(A.procs) as pool:
        res = dict(pool.map(work, recs, chunksize=2))
    R = pd.DataFrame([dict(fid=k, **v) for k, v in res.items() if v is not None])
    O = O[[c for c in O.columns if not c.startswith(("WA_", "WB_", "WE_", "WTP", "A_", "N_", "B_X", "C_", "S3", "S5", "S6", "D_", "E_", "TP"))]]
    O = O.merge(R, on="fid", how="left")
    # parity
    GR = pd.read_pickle(os.path.join(ROOT, "reports", "HT_EXIT_RERUN_2026-11-21_grid.pkl"))[["fid", "G6", "G6_k"]]
    P = O[O.sleeve == "HT"].merge(GR, on="fid")
    print("HT parity c24 vs HT_EXIT_RERUN G6: n", len(P), "max|Δ|", float((P.c24_net - P.G6).abs().max()))
    W = O[O.sleeve == "WILLY"]
    print("WILLY parity c90 vs sim.pkl REF: n", int(W.c90_net.notna().sum()), "max|Δ|", float((W.c90_net - W.REF).abs().max()))
    print("err rows", int(O.get("err", pd.Series(dtype=object)).notna().sum()))
    print("HT still open at 24 h:", int((O[O.sleeve == "HT"].c24_why.isin(["TIME", "OPEN"])).sum()))
    O.to_pickle(os.path.join(OUT, "willy.pkl"))


if __name__ == "__main__":
    main()
