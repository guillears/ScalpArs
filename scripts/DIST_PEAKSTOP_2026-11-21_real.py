#!/usr/bin/env python3
"""DIST_PEAKSTOP — the real DISTRIBUTION_SHORT fills re-walked with every pump-top stop cell (PREREG reports/DIST_PEAKSTOP_2026-11-21_PREREG.md).
Fills: the newest export exports_archive/scalpars_orders_paper_2026-11-21_19-22-18.csv (DISTRIBUTION_SHORT, CLOSED), cross-checked against the
DIST_THEORY_EXIT dedup over every export. Path / landmarks / funding = the DIST_THEORY_EXIT real pipeline (cached 1m in
reports/DIST_THEORY_EXIT_2026-11-21_data/real_cache; recent days fapi with the weight header read). Engine = sim_one family W per-fill stop.
→ reports/DIST_PEAKSTOP_2026-11-21_data/{real_fills.csv, real_summary.csv}. Research only."""
import os, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


RL = _load("DTE_REAL", "DIST_THEORY_EXIT_2026-11-21_real.py")
SIMP = _load("DPS_SIM", "DIST_PEAKSTOP_2026-11-21_sim.py")
SM, C = RL.SM, RL.C
D = os.path.join(R, "reports", "DIST_PEAKSTOP_2026-11-21_data")
EXP = os.path.join(R, "exports_archive", "scalpars_orders_paper_2026-11-21_19-22-18.csv")
MIN, H1, M15 = 60_000, 3_600_000, 900_000
USD = 3050.0
CAPS = SIMP.CAPS
NAMES = ["REF"] + [f"P{c}" for c in CAPS] + [f"Q{c}" for c in CAPS] + [f"W{c}" for c in CAPS] + ["DQ10", "DQ15"]


def stops_for(top, dh, s0, PKabove1):
    out = {}
    for c in CAPS:
        out[f"P{c}"] = min(top, c) if PKabove1 else s0
        out[f"Q{c}"] = max(s0, min(top, c))
        out[f"W{c}"] = float(c)
    for c in (10, 15):
        out[f"DQ{c}"] = max(s0, min(dh, c))
    return out


def main():
    X = pd.read_csv(EXP, low_memory=False)
    X = X[(X.entry_strategy.astype(str) == "DISTRIBUTION_SHORT") & (X.status == "CLOSED")].sort_values("opened_at")
    A = RL.real_fills()
    k1 = set(X.opened_at.astype(str).str[:19] + "|" + X.pair); k2 = set(A.opened_at.astype(str).str[:19] + "|" + A.pair)
    print(f"newest export: {len(X)} closed DIST fills · all-export dedup: {len(A)} · same keys: {k1 == k2}")
    now = int(pd.Timestamp.utcnow().value // 10**6)
    rows = []
    for r in X.itertuples():
        p = r.pair; te = int(pd.Timestamp(str(r.opened_at)[:19]).value // 10**6); E = float(r.entry_price)
        sp_ = int(pd.Timestamp(str(r.entry_frenzy_spike_at)[:19]).value // 10**6); on = sp_ - 300_000
        a = on - 2 * H1; b = min(te + 26 * H1, now)
        for ds in C.days_of(a, b):
            C.fetch_kday(p, ds)
        k = C.grid(C.k1m(p, a, b), a, b)
        t, o, h, l, c = (k[x] for x in ("t", "o", "h", "l", "c"))
        b0 = te // M15 * M15
        i_d = int(np.searchsorted(t, b0 - M15)); i_b = int(np.searchsorted(t, b0))
        i_on = int(np.searchsorted(t, on)); i_e = int(np.searchsorted(t, te // MIN * MIN))
        PK = float(h[i_on:i_e].max()); DH = float(h[i_d:i_b].max())
        j0 = i_e + 1
        tt, oo, hh, ll, cc = t[j0:].copy(), o[j0:].copy(), h[j0:].copy(), l[j0:].copy(), c[j0:].copy(); oo[0] = E
        ft, fr = C.funding(p); ft = np.asarray(ft, "int64"); fcum = np.r_[0.0, np.cumsum(np.asarray(fr, float))]
        tick = SM.NS.tick_of(oo, hh, ll, cc)
        atrp = float(r.entry_dist_atr_pct); s0 = min(1.5 * atrp, 6.0)
        top = (PK * 1.01 / E - 1) * 100; dh = (DH * 1.01 / E - 1) * 100
        st = stops_for(top, dh, s0, PK >= E * 1.01)
        CL = [dict(name="REF", fam="REF")] + [dict(name=n, fam="W", w=st[n]) for n in NAMES[1:]]
        res = SM.sim_one(tt, oo, hh, ll, cc, int(tt[0]), atrp, tick, None, None, ft, fcum, CL)
        ms = r.entry_dist_model_slip_pct
        slip = max(0.10, float(ms) if ms == ms else 0.10)
        booked = r.pnl / (r.quantity * r.entry_price) * 100
        bh = (pd.Timestamp(str(r.closed_at)[:19]).value // 10**6 - te) / H1
        d = dict(pair=p, opened=str(r.opened_at)[:16], booked=booked, booked_reason=r.close_reason, booked_h=bh, funding_usd=r.funding_fee_usd,
                 sl_today=s0, PK_above=(PK / E - 1) * 100, DH_above=(DH / E - 1) * 100, data_h=(tt[-1] + MIN - tt[0]) / H1)
        for cl, (rg, kd, xm, slp) in zip(CL, res):
            f = (fcum[np.searchsorted(ft, xm, "right")] - fcum[np.searchsorted(ft, te, "right")]) * 100 if len(ft) else 0.0
            net = rg + f - slip - 0.05 - ((slip + 0.05) if kd != 0 else 0.02)
            nm = cl["name"]
            d[nm] = round(net, 2); d[nm + "_k"] = {0: "TP", 1: "SL", 2: "T", 9: "open"}[kd]
            d[nm + "_h"] = round((xm - te) / H1, 2); d[nm + "_stop"] = round(slp, 2)
        rows.append(d)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(D, "real_fills.csv"), index=False)
    pd.set_option("display.width", 320)
    print(T[["pair", "opened", "booked", "booked_reason", "booked_h", "sl_today", "PK_above", "DH_above", "data_h", "REF", "REF_k", "REF_h"]].round(2).to_string(index=False))
    S = []
    for nm in NAMES:
        v = T[nm].values; dd = v - T.REF.values; hh_ = T[nm + "_h"].values; kk = T[nm + "_k"].values
        S.append(dict(cell=nm, N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), usd=v.sum() * USD / 100, avg_win=v[v > 0].mean() if (v > 0).any() else np.nan,
                      avg_loss=v[v <= 0].mean() if (v <= 0).any() else np.nan, worst=v.min(), d_mean=dd.mean(), d_usd=dd.sum() * USD / 100,
                      dur_mean=hh_.mean(), dur_med=np.median(hh_), n_tp=int((kk == "TP").sum()), n_sl=int((kk == "SL").sum()),
                      n_time=int((kk == "T").sum()), n_open=int((kk == "open").sum()),
                      loo_min=min((dd.sum() - x) / (len(dd) - 1) for x in dd), loo_max=max((dd.sum() - x) / (len(dd) - 1) for x in dd),
                      per_fill=" · ".join(f"{a_[:-4]} {x:+.2f}{k_[0]}" for a_, x, k_ in zip(T.pair, v, kk))))
    S = pd.DataFrame(S); S.to_csv(os.path.join(D, "real_summary.csv"), index=False)
    print(S.drop(columns="per_fill").round(2).to_string(index=False))
    for nm in NAMES:
        print(nm, S[S.cell == nm].per_fill.iloc[0])
    print("\nstops used (%):")
    print(T[["pair"] + [n + "_stop" for n in NAMES]].to_string(index=False))


if __name__ == "__main__":
    main()
