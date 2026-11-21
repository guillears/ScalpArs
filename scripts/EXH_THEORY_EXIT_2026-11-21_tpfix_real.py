#!/usr/bin/env python3
"""EXH real fills (newest export, closed) + the 3 OPEN EXH shorts from the operator's Open Orders screenshot (ENSO 03:30:33 entry 0.4809;
RARE 14:30:40 and ORCA 15:15:35 UTC — entry ≈ the 1m close of the fill minute, labelled approx) re-priced at today's TP vs fixed TP
5 / 7 / 8 / 9 (same stop, 24 h), with duration. Open trades are marked at the last closed 1m bar. Research only."""
import os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
os.environ["EXH_TPFIX"] = "1"
SM = load("SM", "EXH_THEORY_EXIT_2026-11-21_sim.py"); C = load("SEC", "STATE_EXIT_2026-11-21_common.py")
CACHE = sys.argv[2]; C.K1, C.MET, C.FUN = (os.path.join(CACHE, x) for x in ("k1m", "met", "fund"))
CL = [c for c in SM.cells() if c["name"] in ("REF", "TPF5", "TPF7", "TPF8", "TPF9")]
MIN, H1 = 60_000, 3_600_000
now = int(pd.Timestamp.now("UTC").value // 10**6)
X = pd.read_csv(sys.argv[1], low_memory=False); X = X[(X.entry_strategy == "EXHAUSTION_SHORT") & (X.status == "CLOSED")]
fills = [dict(pair=r.pair, te=int(pd.Timestamp(str(r.opened_at)[:19]).value // 10**6), E=float(r.entry_price), atrp=float(r.entry_exh_atr_pct),
              slip=max(0.10, float(r.entry_exh_model_slip_pct) if r.entry_exh_model_slip_pct == r.entry_exh_model_slip_pct else 0.10),
              status="CLOSED", booked=r.pnl / (r.quantity * r.entry_price) * 100, reason=r.close_reason, tp_stamp=r.entry_exh_tp_pct, sl_stamp=r.entry_exh_sl_pct)
         for r in X.itertuples()]
for p, ts, E in (("ENSOUSDT", "2026-11-21 03:30:33", 0.4809), ("RAREUSDT", "2026-11-21 14:30:40", None), ("ORCAUSDT", "2026-11-21 15:15:35", None)):
    fills.append(dict(pair=p, te=int(pd.Timestamp(ts).value // 10**6), E=E, atrp=4.0, slip=0.10, status="OPEN", booked=np.nan, reason="", tp_stamp=10.0, sl_stamp=6.0))
rows = []
for f in fills:
    p, te = f["pair"], f["te"]
    for ds in C.days_of(te - H1, min(te + 25 * H1, now)):
        fp = os.path.join(C.K1, p, ds + ".npz")
        if ds >= "2026-11-21" and os.path.exists(fp): os.remove(fp)        # today's day: always refetch
        C.fetch_kday(p, ds)
    C.fetch_funding(p, ["2026-11"])
    k = C.k1m(p, te - H1, min(te + 25 * H1, now)); t, o, h, l, c = (k[x] for x in ("t", "o", "h", "l", "c"))
    i_e = int(np.searchsorted(t, te // MIN * MIN))
    E = f["E"] if f["E"] else float(c[i_e]); j0 = i_e + 1
    tt, oo, hh, ll, cc = t[j0:].copy(), o[j0:].copy(), h[j0:].copy(), l[j0:].copy(), c[j0:].copy(); oo[0] = E
    ft, fr = C.funding(p); ft = np.asarray(ft, "int64"); fcum = np.r_[0.0, np.cumsum(np.asarray(fr, float))]
    tick = SM.S.tick_of(oo, hh, ll, cc)
    # the stamped stop / TP (the sim derives them from atrp: back out an atrp that gives the stamped SL, TP stays min(20×atrp, 10))
    atrp = f["sl_stamp"] / 1.5 if f["sl_stamp"] < 6 else max(f["atrp"], 4.0)
    res = SM.sim_one(tt, oo, hh, ll, cc, int(tt[0]), atrp, tick, None, ft, fcum, CL)
    peak = (1 - ll.min() / E) * 100
    d = dict(pair=p, opened=pd.Timestamp(te, unit="ms").strftime("%m-%d %H:%M"), status=f["status"], booked=f["booked"], peak=round(peak, 2),
             sl=round(min(1.5 * atrp, 6.0), 2), tp_today=round(min(20 * atrp, 10.0), 2))
    for cl, (rg, kk, xm) in zip(CL, res):
        fu = (fcum[np.searchsorted(ft, xm, "right")] - fcum[np.searchsorted(ft, te, "right")]) * 100 if len(ft) else 0.0
        net = rg + fu - f["slip"] - 0.05 - ((f["slip"] + 0.05) if kk != 0 else 0.02)
        still = kk == 2 and xm >= int(tt[-1])            # path ended before 24 h → still open (marked now)
        d[cl["name"]] = round(net, 2); d[cl["name"] + "_k"] = "OPEN" if still and te + 24 * H1 > now else {0: "TP", 1: "SL", 2: "TIME", 5: "R"}[kk]
        d[cl["name"] + "_h"] = round((xm - te) / 3.6e6, 1)
    rows.append(d)
T = pd.DataFrame(rows).sort_values("opened"); pd.set_option("display.width", 320)
print(T.to_string(index=False))
cl_ = T[T.status == "CLOSED"]
for nm in ("REF", "TPF5", "TPF7", "TPF8", "TPF9"):
    print(f"CLOSED 7 {nm}: WR {(cl_[nm] > 0).mean() * 100:.0f}% avg {cl_[nm].mean():+.2f} \${cl_[nm].sum() * 30.5:+.0f} dur avg {cl_[nm + '_h'].mean():.1f} h median {cl_[nm + '_h'].median():.1f} h")
T.to_csv(os.path.join(R, "reports", "EXH_THEORY_EXIT_2026-11-21_data", "tpfix_real.csv"), index=False)
