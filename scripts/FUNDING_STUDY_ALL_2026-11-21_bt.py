#!/usr/bin/env python3
"""FUNDING_STUDY_ALL_2026-11-21 — BACKTEST walker: today's exit WITH real funding + the three frozen rule families (short sleeves).

Research only (no bot code / config / git touched). Rule grid FROZEN here, before any result was read:
  ENTRY  (paid rate at entry = the last SETTLED rate ≤ entry, signed against the position; lag-safe — live stamps the premiumIndex
          estimate, see the parity table):  skip | ½-size when paid rate ≥ X, X ∈ {0.05, 0.10, 0.25, 0.50, 1.00} % per settlement,
          and skip | ½-size when the settlement interval at entry ≤ 1 h.                                         (12 cells)
  BUDGET exit at market when cumulative funding PAID ≥ F % of notional AND the trade is not ≥ G % in our favour:
          F ∈ {0.5, 1.0, 1.5, 2.0, 3.0}, G ∈ {0, 2, 4}. Checked from the settlement bar on: its open, then every bar close.   (15 cells)
  TIME   market exit at the close of the last bar before entry + h, h ∈ {6, 8, 12} (24 h = today's).            (3 cells; not FB)
Sleeves (today's exit, shorts):
  EXH  = TP_GRID_2026-11-20_EXH rows_FRENZY (3,368) + rows_WIDE (939) = 4,307 (the DL 603 / 609 set), TP min(20 × ATR15, 7) (DL 609) /
         SL min(1.5 × ATR15, 6) / 24 h, price stop (EXH keeps the price stop, DL 607).
  DIST = reports/ENTRY_WL_2026-11-18_data/dist.pkl (1,547, the DL 586 kept set), TP min(20 × ATR, 10) / SL min(1.5 × ATR, 6) / 24 h (DIST
         unchanged by DL 609).
  FB   = the OOS_REAL_FLATBASE NEAR_BASE rebuild on today's stack (DL 601 cut applied: 545), TP 5 / SL 10 COUNTING FUNDING PAID (DL 607,
         paid-only line) / 72 h, 5 m bars. ENTRY + BUDGET only (operator: FB already counts funding in its stop).
Walk = scripts/FUNDING_STOP_2026-11-21_common conventions: stop first on the bar HIGH at max(line, open) + slip, maker TP only ≥ 0.05 %
through (fill at the line), time exit at the last close; funding = Σ real settlements in (fill, exit bar], each rate × bar-open ÷ fill.
A new market exit pays taker 0.05 % + the row's slip (≥ 0.10 %); sensitivity at 0.30 % is applied in the report.
Output: reports/FUNDING_STUDY_ALL_2026-11-21_data/bt_<SLEEVE>.pkl (per fill: base + every cell + path label + hourly funding / price)
Usage: venv/bin/python -I scripts/FUNDING_STUDY_ALL_2026-11-21_bt.py EXH|DIST|FB [--procs 6]"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "FUNDING_STUDY_ALL_2026-11-21_data")
HR, MIN, DAY = 3_600_000, 60_000, 86_400_000
X_GRID = (0.05, 0.10, 0.25, 0.50, 1.00)
F_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
G_GRID = (0, 2, 4)
T_GRID = (6, 8, 12)


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


C = _imp("FSC", "FUNDING_STOP_2026-11-21_common.py")
G = {}


def walk_all(t, o, h, l, c, i0, E, tp, sl, end_t, F, slip, entry_cost, taker, maker, paid_stop, time_cells=True, through=0.05):
    """short from bar i0 → dict: base + every exit cell (net, why, hold_h) + path label + hourly snapshots."""
    last = int(np.searchsorted(t, end_t, "left")) - 1
    if last < i0:
        return None
    s_ = slice(i0, last + 1)
    tt, oo, hh, ll, cc, Fj = t[s_], o[s_], h[s_], l[s_], c[s_], F[s_]
    n = len(tt); T0 = int(tt[0])
    line = E * (1 + (sl + (np.minimum(Fj, 0.0) if paid_stop else np.zeros(len(Fj)))) / 100.0)
    Tp = E * (1 - tp / 100.0)
    hs = hh >= line; tl = ll <= Tp * (1 - through / 100.0)
    ks = int(np.argmax(hs)) if hs.any() else n; kt = int(np.argmax(tl)) if tl.any() else n
    if ks < n and ks <= kt:
        k0, why = ks, "STOP"; X = max(line[ks], oo[ks]) * (1 + slip / 100); fee = taker * X / E
    elif kt < n:
        k0, why = kt, "TP"; X = Tp; fee = maker * X / E
    else:
        k0, why = n - 1, "TIME"; X = cc[-1] * (1 + slip / 100); fee = taker * X / E
    gross = (1 - X / E) * 100
    base = dict(net=gross - entry_cost - fee + Fj[k0], gross=gross, fund=Fj[k0], why=why, k=k0, hold=(int(tt[k0]) - T0) / HR + 1 / 60)
    out = dict(base_net=base["net"], base_gross=base["gross"], base_fund=base["fund"], base_why=why, base_hold=base["hold"],
               base_net_nofund=base["net"] - base["fund"])

    def forced(kf, mode, px):
        if kf < k0 or (kf == k0 and mode == "open"):
            Xf = px * (1 + slip / 100)
            return (1 - Xf / E) * 100 - entry_cost - taker * Xf / E + Fj[kf], "FORCED", (int(tt[kf]) - T0) / HR + 1 / 60
        return base["net"], why, base["hold"]

    # NET TP / NET SL families (operator 11-21, DL 607 machinery): paid funding moves the resting TP further (re-priced at each settlement
    # bar) and / or the stop in (paid-only: received funding never moves either line). Each judged with funding in the result.
    paidj = np.maximum(-Fj, 0.0)

    def core(net_tp, net_sl):
        ln = E * (1 + (sl - (paidj if net_sl else 0.0 * paidj)) / 100.0)
        tpl = E * (1 - (tp + (paidj if net_tp else 0.0 * paidj)) / 100.0)
        a = hh >= ln; b_ = ll <= tpl * (1 - through / 100.0)
        ks_ = int(np.argmax(a)) if a.any() else n; kt_ = int(np.argmax(b_)) if b_.any() else n
        if ks_ < n and ks_ <= kt_:
            Xx = max(ln[ks_], oo[ks_]) * (1 + slip / 100); kk, wy, fe = ks_, "STOP", taker
        elif kt_ < n:
            Xx = tpl[kt_]; kk, wy, fe = kt_, "TP", maker
        else:
            Xx = cc[-1] * (1 + slip / 100); kk, wy, fe = n - 1, "TIME", taker
        return (1 - Xx / E) * 100 - entry_cost - fe * Xx / E + Fj[kk], wy, (int(tt[kk]) - T0) / HR + 1 / 60

    variants = (("NTP", True, True), ("PSL", False, False), ("NTP_PSL", True, False)) if paid_stop else \
               (("NTP", True, False), ("NSL", False, True), ("NTPSL", True, True))
    out["chk_core"] = core(False, paid_stop)[0] - base["net"]          # must be 0: the variant walker reproduces today
    for nm, a_, b2 in variants:
        out[nm + "_net"], out[nm + "_why"], out[nm + "_hold"] = core(a_, b2)
    if time_cells:
        for hcut in T_GRID:
            kf = int(np.searchsorted(tt, T0 + hcut * HR, "left")) - 1
            r = forced(kf, "close", cc[kf]) if 0 <= kf < n - 1 else (base["net"], why, base["hold"])
            out[f"T{hcut}_net"], out[f"T{hcut}_why"], out[f"T{hcut}_hold"] = r
    paid = -Fj
    favc = (1 - cc / E) * 100; favo = (1 - oo / E) * 100
    for Fb in F_GRID:
        jb = np.flatnonzero(paid >= Fb - 1e-12)
        for g in G_GRID:
            key = f"B{Fb:g}_{g}"
            if not len(jb):
                r = (base["net"], why, base["hold"])
            else:
                j = int(jb[0])
                if favo[j] < g:
                    r = forced(j, "open", oo[j])
                else:
                    kk = np.flatnonzero(favc[j:] < g)
                    r = forced(j + int(kk[0]), "close", cc[j + int(kk[0])]) if len(kk) else (base["net"], why, base["hold"])
            out[key + "_net"], out[key + "_why"], out[key + "_hold"] = r
    # path label (DL 406): WORKED = +1 % our way before −SL against, within the hold; reach = best before −SL
    fav = (1 - ll / E) * 100; adv = (hh / E - 1) * 100
    up = np.flatnonzero(fav >= 1.0); dn = np.flatnonzero(adv >= sl)
    iu = up[0] if len(up) else n; idn = dn[0] if len(dn) else n
    out["label"] = "WORKED" if iu < idn else ("NEVER" if idn < iu else ("AMBIG" if iu < n else "FLAT"))
    out["reach"] = float(fav[:max(idn, 1)].max()) if n else np.nan
    # hourly snapshots under today's exit (NaN after the exit)
    hrs = np.arange(0, 25 if time_cells else 73)
    ix = np.searchsorted(tt, T0 + hrs * HR, "left") - 1; ix[0] = 0
    ok = (ix >= 0) & (ix <= k0) & (ix < n)
    out["snap_fund"] = np.where(ok, Fj[np.clip(ix, 0, n - 1)], np.nan).astype(np.float32)
    out["snap_px"] = np.where(ok, favc[np.clip(ix, 0, n - 1)], np.nan).astype(np.float32)
    return out


# ───────────────────────────── EXH / DIST (1m) ─────────────────────────────
def ed_setup(sleeve):
    if sleeve == "EXH":
        M = _imp("EXHSIM", "TP_GRID_2026-11-20_EXH_sim.py")
        G.update(load1m=M.XS.load1m, TB=M.XS.TB, NS=M.XS.S, cap=7.0)
    else:
        import dist_exit_sim as DE
        G.update(load1m=DE.load1m, TB=DE.NS.TB, NS=DE.NS, cap=10.0)


def ed_rows(sleeve):
    D = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_EXH_data")
    if sleeve == "EXH":
        out = []
        for tag in ("FRENZY", "WIDE"):
            S = pd.read_pickle(os.path.join(D, f"rows_{tag}.pkl")).reset_index(drop=True)
            out.append(pd.DataFrame(dict(univ=tag, xi=S.xi, pair=S.pair.astype(str), entry_ms=S.entry_ms.astype("int64"),
                                         atrp=S.atrp.astype(float), slip=S.slip.astype(float))))
        X = pd.concat(out, ignore_index=True)
    else:
        S = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl")).reset_index(drop=True)
        X = pd.DataFrame(dict(univ="DIST", xi=S.index, pair=S.pair.astype(str), entry_ms=S.entry_ms.astype("int64"),
                              atrp=S.atr15_pct.astype(float), slip=S.slip.astype(float), ref_net=S.net.astype(float)))
    X["rid"] = X.index
    return X


def ed_pair(args):
    p, g = args
    load1m, TB = G["load1m"], G["TB"]
    days = sorted({int(T) // DAY + kk for T in g.entry_ms.values for kk in range(3)})
    one, okd = load1m(p, days)
    five = None
    ft, fc, months = TB.funding_for(p)
    ft = np.asarray(ft, "int64"); fc = np.asarray(fc, float)
    rates = np.diff(fc) if len(fc) > 1 else np.zeros(0)
    out = []
    for r in g.itertuples():
        T = int(r.entry_ms); need = {T // DAY + kk for kk in range(2)}; nb = 1
        use1 = one is not None and need <= okd
        if use1:
            t1 = one[0]; a = int(np.searchsorted(t1, T)); use1 = a < len(t1) and t1[a] == T
        if use1:
            b = int(np.searchsorted(t1, T + 26 * HR)); t, o, h, l, c = [x[a:b] for x in one]
        else:
            if five is None:
                five = TB.load(p)
            if five is None:
                continue
            t5 = np.asarray(five[0], "int64"); a = int(np.searchsorted(t5, T))
            if a >= len(t5) or t5[a] != T:
                continue
            b = int(np.searchsorted(t5, T + 26 * HR)); t, o, h, l, c = t5[a:b], *[np.asarray(five[k][a:b], float) for k in (1, 2, 3, 4)]
            nb = 5
        if len(o) < 2:
            continue
        E = float(o[0]); sl = min(1.5 * r.atrp, 6.0); tp = min(20 * r.atrp, G["cap"])
        mE = pd.Timestamp(T, unit="ms").strftime("%Y-%m"); mX = pd.Timestamp(T + 24 * HR, unit="ms").strftime("%Y-%m")
        miss = (mE not in months) or (mX not in months) or not len(ft)
        Fb = np.zeros(len(t)) if miss else C.settle_on_bars(t, o, E, T, ft, rates, nb, markscale=True)
        w = walk_all(t, o, h, l, c, 0, E, tp, sl, T + 24 * HR, Fb, r.slip, r.slip + 0.05, 0.05, 0.02, False)
        if w is None:
            continue
        w.update(rid=r.rid, pair=p, te=T, miss=miss, nb=nb, tp=tp, sl=sl,
                 paid_rate=(-C.last_rate(ft, rates, T) * 100) if not miss else np.nan,
                 interval_h=C.interval_h(ft, T) if not miss else np.nan)
        out.append(w)
    return out


# ───────────────────────────── FB (5m rebuild) ─────────────────────────────
def fb_setup():
    import oos_real_flatbase_common as OC
    import oos_real_flatbase_recon as RC
    G.update(OC=OC, RC=RC, F=OC.funding_table())


def fb_rows():
    X = pd.read_csv(os.path.join(ROOT, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    X = X[X.X0_why.isin(["TP", "STOP", "TIME"]) & (X.dist_base > 2.09) & ~(X.funding_rate_pct > 0.005)].copy().reset_index(drop=True)
    X["rid"] = X.index; X["univ"] = "FB"; X["entry_ms"] = X.cf_entry_t.astype("int64")
    return X


def fb_pair(args):
    pair, g = args
    OC, RC, F = G["OC"], G["RC"], G["F"]
    x = RC.W.load(pair)
    out = []
    if x is None:
        return out
    t, o, h, l, c = [np.asarray(a) for a in x[:5]]; t = t.astype("int64")
    tidx = {int(tt): k for k, tt in enumerate(t)}
    fx = F.get(pair)
    for r in g.itertuples():
        i = tidx.get(int(r.cf_entry_t))
        if i is None:
            continue
        E = float(r.cf_entry_px); te = int(r.cf_entry_t) + 10_000; end_t = int(t[i]) + 72 * HR
        tot_i, miss = OC.fund(F, pair, te, end_t)
        if miss or fx is None or not len(fx[0]):
            Fb = np.zeros(len(t)); fs_t = None
        else:
            fs_t = fx[0]; Fb = C.settle_on_bars(t, o, E, te, fs_t, fx[3], 5, markscale=True)
        w = walk_all(t, o, h, l, c, i, E, 5.0, 10.0, end_t, Fb, float(r.slip_x), RC.TAKER, RC.TAKER, RC.MAKER, True, time_cells=False)
        if w is None:
            continue
        w.update(rid=r.rid, pair=pair, te=int(r.cf_entry_t), miss=miss, nb=5, tp=5.0, sl=10.0,
                 paid_rate=-float(r.funding_rate_pct), interval_h=C.interval_h(fs_t, te) if fs_t is not None else np.nan)
        out.append(w)
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
    R = pd.DataFrame(res)
    keep = [c for c in ("rid", "univ", "xi", "ref_net", "X0_net", "X0_why", "slip") if c in X.columns]
    R = R.merge(X[keep], on="rid", how="left")
    R.to_pickle(os.path.join(OUT, f"bt_{A.sleeve}.pkl"))
    print(A.sleeve, "rows", len(R), "of", len(X), "· no-funding-data", int(R.miss.sum()), "· 5m fallback", int((R.nb == 5).sum()))


if __name__ == "__main__":
    main()
