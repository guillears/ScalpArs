#!/usr/bin/env python3
"""DIST_THEORY_EXIT — real DISTRIBUTION_SHORT fills re-priced with the same sim (PREREG reports/DIST_THEORY_EXIT_2026-11-21_PREREG.md).
Every exports_archive/scalpars_orders_*.csv, DISTRIBUTION_SHORT, CLOSED, no probes, dedup (opened_at[:19], pair, direction) keeping the newest
export. 1m klines with volume (data.binance.vision daily zips, 0 weight; recent days fapi, weight header read, stop on 418 / 429) + real
funding, cached in reports/DIST_THEORY_EXIT_2026-11-21_data/real_cache/. Entry = the real fill price; the path starts the minute after the
fill; landmarks from 1m (distribution candle = the 15m bar closing at the 15m boundary before the fill; episode on = entry_frenzy_spike_at − 5
min). A cell still open at the data end is marked to the last close (kind 'open'). Research only.
Usage: venv/bin/python -I scripts/DIST_THEORY_EXIT_2026-11-21_real.py"""
import glob, os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


SM = load("DTE_SIM", "DIST_THEORY_EXIT_2026-11-21_sim.py"); C = load("SEC", "STATE_EXIT_2026-11-21_common.py")
D = os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21_data")
CACHE = os.path.join(D, "real_cache")
C.K1, C.MET, C.FUN = (os.path.join(CACHE, x) for x in ("k1m", "met", "fund"))
for d_ in (C.K1, C.FUN):
    os.makedirs(d_, exist_ok=True)
MIN, H1, M15 = 60_000, 3_600_000, 900_000
USD = 3050.0


def real_fills():
    xs = []
    for f in sorted(glob.glob(os.path.join(R, "exports_archive", "scalpars_orders_*.csv"))):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "entry_strategy" not in d.columns:
            continue
        d = d[d.entry_strategy.astype(str) == "DISTRIBUTION_SHORT"].copy()
        if len(d):
            d["src"] = os.path.basename(f); d["src_ts"] = os.path.getmtime(f); xs.append(d)
    A = pd.concat(xs, ignore_index=True)
    A = A[A.status == "CLOSED"]
    if "is_probe" in A.columns:
        A = A[~A.is_probe.fillna(False).astype(bool)]
    A["key"] = A.opened_at.astype(str).str[:19] + "|" + A.pair + "|" + A.direction.astype(str)
    A = A.sort_values(["src", "src_ts"]).drop_duplicates("key", keep="last")
    return A.sort_values("opened_at")


def main():
    X = real_fills()
    CL = SM.cells()
    now = int(pd.Timestamp.utcnow().value // 10**6)
    rows = []
    for r in X.itertuples():
        p = r.pair; te = int(pd.Timestamp(str(r.opened_at)[:19]).value // 10**6); E = float(r.entry_price)
        sp_ = int(pd.Timestamp(str(r.entry_frenzy_spike_at)[:19]).value // 10**6); on = sp_ - 300_000
        a = on - 2 * H1; b = min(te + 26 * H1, now)
        for ds in C.days_of(a, b):
            C.fetch_kday(p, ds)
        C.fetch_funding(p, sorted({pd.Timestamp(a, unit="ms").strftime("%Y-%m"), pd.Timestamp(b, unit="ms").strftime("%Y-%m")}))
        k = C.grid(C.k1m(p, a, b), a, b)
        t, o, h, l, c, v, qv = (k[x] for x in ("t", "o", "h", "l", "c", "v", "qv"))
        b0 = te // M15 * M15
        i_d = int(np.searchsorted(t, b0 - M15)); i_b = int(np.searchsorted(t, b0))
        i_on = int(np.searchsorted(t, on)); i_e = int(np.searchsorted(t, te // MIN * MIN))
        tpx = (h + l + c) / 3
        an = dict(DH=float(h[i_d:i_b].max()), DO=float(o[i_d]), DL=float(l[i_d:i_b].min()), DQV=float(qv[i_d:i_b].sum()),
                  base=float(c[max(i_on - 30, 0)]), PK=float(h[i_on:i_e].max()),
                  vwap=float((tpx[i_on:i_e] * v[i_on:i_e]).sum() / max(v[i_on:i_e].sum(), 1e-12)))
        # closed 15m bars after the fill boundary
        be, bo, bc, bq = [], [], [], []
        kk = 0
        while True:
            s0 = b0 + kk * M15; s1 = s0 + M15
            if s1 > t[-1] + MIN or s0 > te + 25 * H1:
                break
            i0, i1 = int(np.searchsorted(t, s0)), int(np.searchsorted(t, s1))
            if i1 - i0 == 15:
                be.append(s1); bo.append(o[i0]); bc.append(c[i1 - 1]); bq.append(qv[i0:i1].sum())
            kk += 1
        b15 = (np.array(be, np.int64), np.array(bo), np.array(bc), np.array(bq)) if be else None
        j0 = i_e + 1
        tt, oo, hh, ll, cc = t[j0:].copy(), o[j0:].copy(), h[j0:].copy(), l[j0:].copy(), c[j0:].copy(); oo[0] = E
        ft, fr = C.funding(p); ft = np.asarray(ft, "int64"); fcum = np.r_[0.0, np.cumsum(np.asarray(fr, float))]
        tick = SM.NS.tick_of(oo, hh, ll, cc)
        atrp = float(r.entry_dist_atr_pct)
        res = SM.sim_one(tt, oo, hh, ll, cc, int(tt[0]), atrp, tick, an, b15, ft, fcum, CL)
        ms = r.entry_dist_model_slip_pct
        slip = max(0.10, float(ms) if ms == ms else 0.10)
        booked = r.pnl / (r.quantity * r.entry_price) * 100
        d = dict(pair=p, opened=str(r.opened_at)[5:16], booked=booked, booked_reason=r.close_reason, peak_booked=r.peak_pnl,
                 sl_today=min(1.5 * atrp, 6.0), DH_above=(an["DH"] / E - 1) * 100, PK_above=(an["PK"] / E - 1) * 100,
                 DO_above=(an["DO"] / E - 1) * 100, data_h=(tt[-1] + MIN - tt[0]) / H1)
        for cl, (rg, kd, xm, slp) in zip(CL, res):
            f = (fcum[np.searchsorted(ft, xm, "right")] - fcum[np.searchsorted(ft, te, "right")]) * 100 if len(ft) else 0.0
            net = rg + f - slip - 0.05 - ((slip + 0.05) if kd != 0 else 0.02)
            d[cl["name"]] = round(net, 2)
            d[cl["name"] + "_k"] = {0: "TP", 1: "SL", 2: "T", 5: "R", 9: "open"}[kd]
        rows.append(d)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(D, "real_fills.csv"), index=False)
    pd.set_option("display.width", 320)
    names = [c["name"] for c in CL]
    print(T[["pair", "opened", "booked", "booked_reason", "peak_booked", "sl_today", "DO_above", "DH_above", "PK_above", "data_h", "REF", "REF_k"]].round(2).to_string(index=False))
    S = []
    for nm in names:
        vv = T[nm].values; d_ = vv - T.REF.values
        S.append(dict(cell=nm, N=len(vv), WR=(vv > 0).mean() * 100, avg=vv.mean(), d_mean=d_.mean(), usd=vv.sum() * USD / 100,
                      d_usd=d_.sum() * USD / 100, n_better=int((d_ > 0.01).sum()), n_worse=int((d_ < -0.01).sum()),
                      kinds="/".join(T[nm + "_k"].values),
                      loo_min=min((d_.sum() - x) / (len(d_) - 1) for x in d_) if len(d_) > 1 else np.nan))
    S = pd.DataFrame(S); S.to_csv(os.path.join(D, "real_summary.csv"), index=False)
    print("\n", S.round(2).to_string(index=False))
    print("\nper fill net % (selected)\n", T[["pair", "booked"] + [n for n in names if n in ("REF", "A_R382", "B_X7_N120", "C_DH0", "C_DH1", "C_DHT",
                                                                                              "C_PK0", "R_DO", "R_DH", "W_SL10", "W_SL12", "W_SL15", "M_1", "G_ABS50")]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
