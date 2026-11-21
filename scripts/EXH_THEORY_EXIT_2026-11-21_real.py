#!/usr/bin/env python3
"""EXH_THEORY_EXIT — real EXH fills (newest export, closed + open marked now) re-priced with the same sim (1m klines vision / fapi, real
funding). Entry = the real fill price; the path starts the minute after the fill. Research only."""
import os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
SM = load("SM", "EXH_THEORY_EXIT_2026-11-21_sim.py"); C = load("SEC", "STATE_EXIT_2026-11-21_common.py")
CACHE = sys.argv[2]
C.K1, C.MET, C.FUN = (os.path.join(CACHE, x) for x in ("k1m", "met", "fund"))
for d_ in (C.K1, C.FUN): os.makedirs(d_, exist_ok=True)
X = pd.read_csv(sys.argv[1], low_memory=False)
X = X[X.entry_strategy == "EXHAUSTION_SHORT"].copy()
os.environ["EXH_CTRL"] = "1"; CL = SM.cells()
want = ["REF", "C_PK0", "W_PK0", "K_SL7", "K_SL8", "K_SL10", "A_R618", "N_BASE", "B_X7_N240", "F_P1"]
CL = [c for c in CL if c["name"] in want]
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
now = int(pd.Timestamp.utcnow().value // 10**6)
rows = []
for r in X.itertuples():
    p = r.pair; te = int(pd.Timestamp(str(r.opened_at)[:19]).value // 10**6); E = float(r.entry_price)
    sp_ = int(pd.Timestamp(str(r.entry_frenzy_spike_at)[:19]).value // 10**6)
    on = sp_ - 300_000
    a = on - 2 * H1; b = min(te + 25 * H1, now)
    for ds in C.days_of(a, b): C.fetch_kday(p, ds)
    C.fetch_funding(p, sorted({pd.Timestamp(a, unit="ms").strftime("%Y-%m"), pd.Timestamp(b, unit="ms").strftime("%Y-%m")}))
    k = C.k1m(p, a, b)
    t, o, h, l, c = (k[x] for x in ("t", "o", "h", "l", "c")); v = k["v"].astype(float)
    i_on = int(np.searchsorted(t, on)); i_e = int(np.searchsorted(t, te // MIN * MIN))
    tp = (h + l + c) / 3
    an = dict(base=float(c[max(i_on - 30, 0)]), peak=float(h[i_on:i_e].max()), vwap=float((tp[i_on:i_e] * v[i_on:i_e]).sum() / v[i_on:i_e].sum()))
    j0 = i_e + 1
    tt, oo, hh, ll, cc = t[j0:].copy(), o[j0:].copy(), h[j0:].copy(), l[j0:].copy(), c[j0:].copy(); oo[0] = E
    ft, fr = C.funding(p); ft = np.asarray(ft, "int64"); fcum = np.r_[0.0, np.cumsum(np.asarray(fr, float))]
    tick = SM.S.tick_of(oo, hh, ll, cc)
    atrp = float(r.entry_exh_atr_pct)
    res = SM.sim_one(tt, oo, hh, ll, cc, int(tt[0]), atrp, tick, an, ft, fcum, CL)
    slip = max(0.10, float(r.entry_exh_model_slip_pct) if r.entry_exh_model_slip_pct == r.entry_exh_model_slip_pct else 0.10)
    d = dict(pair=p, opened=str(r.opened_at)[5:16], status=r.status, booked=r.pnl / (r.quantity * r.entry_price) * 100 if r.status == "CLOSED" else np.nan,
             peak_booked=r.peak_pnl, sl_today=min(1.5 * atrp, 6.0), pump_top_above_entry=(an["peak"] / E - 1) * 100)
    for cl, (rg, kk, xm) in zip(CL, res):
        f = (fcum[np.searchsorted(ft, xm, "right")] - fcum[np.searchsorted(ft, te, "right")]) * 100 if len(ft) else 0.0
        net = rg + f - slip - 0.05 - ((slip + 0.05) if kk != 0 else 0.02)
        d[cl["name"]] = round(net, 2); d[cl["name"] + "_k"] = {0: "TP", 1: "SL", 2: "T", 5: "R"}[kk] + ("*" if kk == 2 and xm >= now - 2 * MIN else "")
    rows.append(d)
T = pd.DataFrame(rows).sort_values("opened")
pd.set_option("display.width", 300)
print(T[["pair", "opened", "status", "booked", "peak_booked", "sl_today", "pump_top_above_entry"] + [x for c_ in want for x in (c_, c_ + "_k")]].round(2).to_string(index=False))
cl_ = T[T.status == "CLOSED"]
print("\nclosed avg:", {w: round(cl_[w].mean(), 2) for w in want}, "N", len(cl_))
T.to_csv(os.path.join(R, "reports", "EXH_THEORY_EXIT_2026-11-21_data", "real_fills.csv"), index=False)
