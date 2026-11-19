#!/usr/bin/env python3
"""ROUTE A FOLLOW-UP (PREREG reports/ROUTE_A_FOLLOWUP_2026-11-19_PREREG.md) — Q1(a) full-population pricing on 1m klines.
Every W15_R3_V10 trigger (reports/ROUTE_A_TICKS_data/order.csv, 33,399): entry ≈ trigger close + 8 s = open of the 1m bar at the
trigger close × (1 + d) (d = median tick-entry ÷ 1m-open drift, calibrated on the 1,550 tick events) × (1 + slip); pessimistic
variant = close of that first 1m bar. Path from the next 1m bar; exits Q3 / TR / T1 (route_a_ticks_features.sim); real funding.
Also the 1m VWAP proxy `vwap1m` for Q2. One position per coin per exit is applied in route_a_followup_q1.py.
Output reports/ROUTE_A_FOLLOWUP_data/pop1m.pkl. Local files only (0 API weight)."""
import os, pickle, sys
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import route_a_ticks_features as RF   # noqa: E402
from route_a_followup_fetch1m import load_local   # noqa: E402
TB = RF.TB
D = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_data"); SRC = os.path.join(R, "reports", "ROUTE_A_TICKS_data")
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
G = {}


def calibrate():
    """median (first trade ≥ sig + 8 s ÷ open of the 1m bar starting at sig − 1) over the tick events"""
    tk = pickle.load(open(os.path.join(D, "ticks.pkl"), "rb"))["T"]
    t0 = tk[tk.chk == 0]
    r = []
    for x in t0.itertuples():
        f = os.path.join(SRC, "ev", f"{x.pair}_{x.sig_ms}.npz")
        b = np.load(f)["bars"]
        i = np.where(b[:, 0] == x.sig_ms)[0]
        if len(i):
            r.append(x.P0 / b[i[0], 1] - 1)
    r = np.array(r)
    return float(np.median(r)), float(np.mean(r)), len(r)


def work(pair):
    O = G["O"][G["O"].pair == pair]
    x = load_local(pair)
    if x is None or not len(x):
        return [dict(pair=pair, sig_ms=int(s), status="nodata") for s in O.sig_ms]
    ot = x.open_time.values.astype("int64"); o, h, l, c = (x[k].values.astype(float) for k in ("o", "h", "l", "c"))
    v, qv = x.vol.values.astype(float), x.qvol.values.astype(float)
    ft, fc, months = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    out = []
    for e in O.itertuples():
        s = int(e.sig_ms)
        r = dict(pair=pair, sig_ms=s, month=e.month, block=e.block, runner=e.runner, spike_after=e.spike_after,
                 flagged=int(e.spike_after > 0 and e.spike_after <= s))
        i0 = int(np.searchsorted(ot, s))
        if i0 >= len(ot) or ot[i0] != s or i0 < 15:
            r["status"] = "nobar"; out.append(r); continue
        # bars after the entry minute, up to 25 h; stop at the first gap > 10 min (data hole)
        j1 = int(np.searchsorted(ot, s + 25 * H1))
        seg = np.arange(i0 + 1, j1)
        if len(seg):
            gaps = np.where(np.diff(np.r_[ot[i0], ot[seg]]) > 10 * MIN)[0]
            if len(gaps):
                seg = seg[:gaps[0]]
        r["status"] = "ok"
        r["vwap1m"] = (c[i0 - 1] / (qv[i0 - 15:i0].sum() / v[i0 - 15:i0].sum()) - 1) * 100 if v[i0 - 15:i0].sum() > 0 else np.nan
        r["ret15_1m"] = (c[i0 - 1] / o[i0 - 15] - 1) * 100
        r["path_end_h"] = ((ot[seg[-1]] + MIN) - s) / H1 if len(seg) else 0.0
        te = ot[seg] + MIN
        mon = pd.Timestamp(s, unit="ms").strftime("%Y-%m")
        r["fund_missing"] = int(mon not in months)
        for var, E0 in (("p", o[i0] * (1 + G["d"])), ("x", c[i0])):
            ent = s + 8000 if var == "p" else s + MIN
            for k, (tp, sl, hz, tr) in RF.EXITS.items():
                if len(seg) < 2:
                    r[f"{k}_{var}_g"] = np.nan; continue
                gr, kind, xms = RF.sim(o[seg], h[seg], l[seg], c[seg], te, E0, tp, sl, hz, tr)
                if len(ft) and not r["fund_missing"]:
                    a_ = int(np.searchsorted(ft, ent, side="right")); b_ = int(np.searchsorted(ft, int(xms), side="right"))
                    fcost = float(rates[a_:b_].sum())
                else:
                    fcost = 0.01 * (int(xms) - ent) / (8 * H1)
                r[f"{k}_{var}_g"], r[f"{k}_{var}_k"], r[f"{k}_{var}_x"], r[f"{k}_{var}_f"] = gr, kind, int(xms), fcost
                r[f"{k}_{var}_complete"] = bool(kind != 0 or te[-1] >= te[0] + hz)
        out.append(r)
    return out


def init(O, d):
    G["O"] = O; G["d"] = d


def main():
    d_med, d_mean, n = calibrate()
    print(f"entry drift calibration: median {d_med*100:.3f} % mean {d_mean*100:.3f} % (n {n})", flush=True)
    O = pd.read_csv(os.path.join(SRC, "order.csv"))
    rows = []
    with mp.Pool(8, initializer=init, initargs=(O, d_med)) as P:
        for k, r in enumerate(P.imap_unordered(work, sorted(O.pair.unique()))):
            rows += r
            if k % 100 == 0:
                print(k, len(rows), flush=True)
    X = pd.DataFrame(rows)
    pickle.dump(dict(X=X, d=d_med, d_mean=d_mean, n_cal=n), open(os.path.join(D, "pop1m.pkl"), "wb"))
    print(X.status.value_counts().to_string())


if __name__ == "__main__":
    main()
