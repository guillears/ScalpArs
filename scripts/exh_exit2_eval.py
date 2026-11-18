#!/usr/bin/env python3
"""EXH_EXIT2 (PREREG reports/EXH_EXIT2_2026-11-18_PREREG.md) — EVAL. Research only, BACKTEST.
Nets (model slip ≥ 0.10 per taker leg / flat 0.20 stress, real funding) · DESIGN grid + plateau + pick · HOLDOUT ship bar · family-wise
day sign-flip max-T null · RND control · saved / deeper + winners cut · path decomposition · half-years / BTC regimes · concentration / LOO ·
Kelly · one-size DCR 0.61 · joint book (joint_short_sim.sim) · stop clustering · E 2023-H1 (DL 582 block + one position).
Writes reports/EXH_EXIT2_2026-11-18_*.csv + reports/EXH_EXIT2_data/eval.pkl.  Usage: venv/bin/python -I scripts/exh_exit2_eval.py"""
import glob, os, pickle, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import exh_exit2_sim as XS   # noqa: E402
import joint_short_sim as JS  # noqa: E402
R = XS.R; RP = os.path.join(R, "reports"); D = XS.D
OUT = os.path.join(RP, "EXH_EXIT2_2026-11-18")
DAY, H1 = 86_400_000, 3_600_000
SIZE = 0.61; NOTIONAL = 6_100.0
CL = XS.cells(); NAMES = [c["name"] for c in CL]; NC = len(CL)
REF = 0; XNT = NAMES.index("X_NOTP")
SELECT = [i for i, c in enumerate(CL) if c["fam"] not in ("REF", "X")]
HO_MS = int(pd.Timestamp("2025-07-01").value // 10**6)
D0, D1 = int(pd.Timestamp("2023-07-01").value // 10**6) // DAY, int(pd.Timestamp("2026-11-18").value // 10**6) // DAY
DH = HO_MS // DAY
HALVES = ["2023-H2", "2024-H1", "2024-H2", "2025-H1", "2025-H2", "2026-H1", "2026-H2"]
DES_H = HALVES[:4]
RNG = np.random.default_rng(20261118)


def neighbours(j):
    c = CL[j]; f = c["fam"]; same = [i for i in SELECT if CL[i]["fam"] == f and i != j]
    def grid(ax1, ax2, k1, k2):
        return [i for i in same if abs(ax1.index(k1(CL[i])) - ax1.index(k1(c))) + abs(ax2.index(k2(CL[i])) - ax2.index(k2(c))) == 1]
    if f == "B":
        a = [2, 3, 4, 5, 6]; return [i for i in same if abs(a.index(CL[i]["arm"]) - a.index(c["arm"])) == 1]
    if f == "L":
        return grid([4, 6, 8], [2, 3], lambda x: x["arm"], lambda x: x["arm"] - x["par"])
    if f == "D1":
        return grid([6, 8], [2, 3, 4], lambda x: x["arm"], lambda x: x["par"])
    if f == "D2":
        return grid([8, 10], [3, 5], lambda x: x["arm"], lambda x: x["par"])
    if f == "S":
        return grid([1, 1.5, 2, 3], [2, 3, 4], lambda x: x["tp"], lambda x: x["sl"])
    if f == "T":
        a = [4, 8, 16]; return [i for i in same if abs(a.index(CL[i]["hold"]) - a.index(c["hold"])) == 1]
    if f == "NP":
        a = [2, 4, 8]; return [i for i in same if abs(a.index(CL[i]["par"]) - a.index(c["par"])) == 1]
    return []


def load(tag):
    X = pd.read_pickle(os.path.join(D, f"rows_{tag}.pkl")).reset_index(drop=True)
    Z = dict(np.load(os.path.join(D, f"sim_{tag}.npz")))
    assert (Z["xi"] == X.xi.values).all() and Z["ok"].all()
    return X, Z


def nets(X, Z, flat=None):
    s = X.slip.values.astype(float)[:, None] if flat is None else flat
    tk = Z["k"] != 0
    base = Z["r"].astype(float) + Z["f"].astype(float)
    return base - s - 0.05 - np.where(tk, s + 0.05, 0.02)


def wilson_floor(k, n, z=1.645):
    if n <= 0:
        return 0.0
    p = k / n; den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0))) / den


def kel(v):
    w = v[v > 0]; lo = v[v <= 0]
    if not len(w) or not len(lo):
        return np.nan, np.nan, np.nan
    aw, al = w.mean(), -lo.mean(); pf = wilson_floor(len(w), len(v))
    return pf - (1 - pf) / (aw / al), pf * 100, al / (aw + al) * 100


def onesize(v, xd, d0, d1):
    r = pd.Series(v * SIZE / 100).groupby(xd).sum().reindex(np.arange(d0, d1), fill_value=0.0)
    eq = np.cumprod(1 + r.values)
    return dict(dcr=(eq[-1] ** (1 / len(r)) - 1) * 100 if eq[-1] > 0 else np.nan,
                maxdd=(1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max() * 100, worst_day=r.min() * 100), r


def boot_ci(v, days, nb=2000, block=10):
    s = pd.Series(v).groupby(days).agg(["sum", "count"])
    dd = np.arange(days.min(), days.max() + 1); s = s.reindex(dd, fill_value=0)
    n = len(dd); nbk = int(np.ceil(n / block)); st = RNG.integers(0, max(n - block, 1), (nb, nbk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :n]
    a = s["sum"].values[idx].sum(1) / np.maximum(s["count"].values[idx].sum(1), 1)
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def blk_days(blk):
    return {"DESIGN": (D0, DH), "HOLDOUT": (DH, D1), "ALL": (D0, D1)}[blk]


def stats(X, Z, NET, N20, idx, j, blk):
    d0, d1 = blk_days(blk)
    v = NET[idx, j]; ref = NET[idx, REF]; dlt = v - ref; xd = Z["xm"][idx, j] // DAY; ed = X.entry_ms.values[idx] // DAY
    k = Z["k"][idx, j]
    kt, pf, be = kel(v); kd, pfd, _ = kel(pd.Series(v).groupby(xd).sum().values)
    os_, _ = onesize(v, xd, d0, d1)
    sv = np.sort(v); sr = np.sort(ref); n95 = int(np.floor(len(v) * 0.95))
    ci = boot_ci(dlt, ed) if j != REF else (np.nan, np.nan)
    pos = dlt.sum()
    pp = pd.Series(dlt).groupby(X.pair.values[idx]).sum(); dd_ = pd.Series(dlt).groupby(ed).sum()
    toppair = pp.idxmax() if len(pp) else ""
    rec = dict(N=len(idx), WR=(v > 0).mean() * 100, avg=v.mean(), avg_win=v[v > 0].mean(), avg_loss=v[v <= 0].mean(), BE=be,
               delta=dlt.mean(), d_lo=ci[0], d_hi=ci[1], ref=ref.mean(), b5=sv[:n95].mean(), ref_b5=sr[:n95].mean(),
               f20=N20[idx, j].mean(), ref_f20=N20[idx, REF].mean(), kelly=kt, WRfloor=pf, kelly_day=kd, WRfloor_day=pfd,
               tp=(k == 0).mean() * 100, sl=(k == 1).mean() * 100, time=(k == 2).mean() * 100, dyn=(k == 3).mean() * 100, np_=(k == 4).mean() * 100,
               hold_h=((Z["xm"][idx, j] - X.entry_ms.values[idx]) / H1).mean(),
               pair_share=pp.max() / pos * 100 if pos > 0 else np.nan, pair_top=toppair,
               day_share=dd_.max() / pos * 100 if pos > 0 else np.nan,
               loo_pair=np.delete(dlt, np.nonzero(X.pair.values[idx] == toppair)[0]).mean() if toppair else np.nan,
               **os_)
    for h in HALVES:
        m = X.half.values[idx] == h
        rec[f"d_{h}"] = dlt[m].mean() if m.any() else np.nan; rec[f"a_{h}"] = v[m].mean() if m.any() else np.nan
        rec[f"n_{h}"] = int(m.sum())
    return rec


def path_group(Z, NET, idx, j):
    v = NET[idx, j]; wk = Z["worked"][idx, j] == 1; k = Z["k"][idx, j]; rec_ = Z["rec"][idx, j] == 1; best = Z["best"][idx, j]
    return np.where(v > 0, np.where((best - 0.09 - v) >= np.maximum(1.0, v), "LEFT_ON_TABLE", "CAPTURED"),
                    np.where(wk, "GAVE_BACK", np.where((k != 0) & (k != 2) & rec_, "STOPPED_RECOVERED",
                                                        np.where((k == 1) | (k == 3), "NEVER_WORKED", "DRIFT_LOSS"))))


def regime_lookup(ms):
    Rg = pickle.load(open(os.path.join(RP, "LEADS_DEEP_2026-11-18_data", "regimes.pkl"), "rb"))
    out = {}
    for nm in ("R1", "R4"):
        tv, vv = Rg[nm]; i = np.searchsorted(tv, ms, "right") - 1
        out[nm] = np.where(i >= 0, vv[np.maximum(i, 0)], np.nan)
    return out


def e_block(XE):
    """DL 582 on E: ssc > −0.0366 ∧ oi_vol > 0.128; OI = last metrics row ≤ entry; 24 h quote volume = the 288 5m bars before entry."""
    oiv = np.full(len(XE), np.nan)
    for i, (p, t) in enumerate(zip(XE.pair, XE.entry_ms)):
        ms = []
        for dd in (t - DAY, t):
            f = os.path.join(D, "metrics", f"{p}.{pd.to_datetime(int(dd), unit='ms').strftime('%Y-%m-%d')}.csv")
            if os.path.exists(f) and os.path.getsize(f) > 10:
                ms.append(pd.read_csv(f, usecols=["create_time", "sum_open_interest_value"]))
        f5 = os.path.join(XS.G, "E", "k5m", p + ".csv")
        if not ms or not os.path.exists(f5):
            continue
        m = pd.concat(ms); m["t"] = pd.to_datetime(m.create_time).astype("int64") // 10**6; m = m[m.t <= t].sort_values("t")
        if not len(m):
            continue
        k5 = pd.read_csv(f5, usecols=["open_time", "qvol"]); k5 = k5[(k5.open_time < t) & (k5.open_time >= t - DAY)]
        if len(k5) < 200:
            continue
        oiv[i] = m.sum_open_interest_value.iloc[-1] / k5.qvol.sum()
    blk = np.isfinite(oiv) & (XE.ssc.values > -0.0366) & (oiv > 0.128)
    return oiv, blk


def one_pos(pair, T, XM):
    o = np.lexsort((T, pair)); keep = np.zeros(len(o), bool); lp = None; busy = -1
    for q, i in enumerate(o):
        if pair[i] != lp:
            lp = pair[i]; busy = -1
        if T[i] < busy:
            continue
        keep[q] = True; busy = XM[i]
    return np.sort(o[keep])


_JT = {}


def joint(XE, ZE, NET, j, blk):
    if "T" not in _JT:
        _JT["T"] = JS.load()[0]
    T = _JT["T"].copy()
    m = (T.sleeve == "EXH").values
    key = dict(zip(zip(XE.pair, XE.entry_ms.astype("int64")), range(len(XE))))
    ii = np.array([key[(p, int(t))] for p, t in zip(T.pair.values[m], T.entry_ms.values[m])])
    T.loc[m, "net"] = NET[ii, j]; T.loc[m, "exit_ms"] = ZE["xm"][ii, j]; T.loc[m, "stop"] = np.isin(ZE["k"][ii, j], (1, 3))
    s = JS.sim(T)
    r = s["r_day"]; days = JS.DAYS
    if blk == "HOLDOUT":
        r = r[days >= DH]
    st = JS.stats(r)
    return dict(dcr=st["dcr"], maxdd=st["maxdd"], worst_day=st["worst_day"], worst7=st["worst7"], N_exh=int(s["taken"][m].sum())), s, T


def main():
    X, Z = load("EXH")
    NET = nets(X, Z); N20 = nets(X, Z, flat=0.20)
    blkv = np.where(X.entry_ms.values >= HO_MS, "HOLDOUT", "DESIGN")
    rg = regime_lookup(X.entry_ms.values); X["R1"] = rg["R1"]; X["R4"] = rg["R4"]
    print("REF avg", NET[:, REF].mean().round(3), "WR", (NET[:, REF] > 0).mean().round(3), "1m share", Z["src1m"].mean(), flush=True)
    # 1. grid
    rows = []
    for blk in ("DESIGN", "HOLDOUT", "ALL"):
        idx = np.arange(len(X)) if blk == "ALL" else np.nonzero(blkv == blk)[0]
        for j in range(NC):
            rows.append(dict(cell=NAMES[j], fam=CL[j]["fam"], blk=blk, **stats(X, Z, NET, N20, idx, j, blk)))
    GR = pd.DataFrame(rows)
    # 2. selection on DESIGN
    Gd = GR[GR.blk == "DESIGN"].set_index("cell")
    sel = []
    for j in SELECT:
        g = Gd.loc[NAMES[j]]; nb = neighbours(j)
        hp = sum(g[f"d_{h}"] > 0 for h in DES_H)
        npos = np.mean([Gd.loc[NAMES[i], "delta"] > 0 for i in nb]) if nb else 0.0
        smooth = np.mean([g.delta] + [Gd.loc[NAMES[i], "delta"] for i in nb])
        sel.append(dict(cell=NAMES[j], fam=CL[j]["fam"], delta=g.delta, d_lo=g.d_lo, d_hi=g.d_hi, halves_pos=hp, plateau_pos=npos,
                        smooth=smooth, neighbours=",".join(NAMES[i] for i in nb),
                        survivor=bool(g.delta > 0 and hp >= 3 and npos >= 2 / 3)))
    SS = pd.DataFrame(sel).sort_values("smooth", ascending=False)
    sv = SS[SS.survivor]
    pick = sv.iloc[0].cell if len(sv) else None
    print("survivors", list(sv.cell), "pick", pick, flush=True)
    # 3. family-wise null: day-level sign flip of paired Δ, max-T (studentised), full span
    ed = X.entry_ms.values // DAY; ud, inv = np.unique(ed, return_inverse=True)
    Dd = np.zeros((len(SELECT), len(ud)))
    for q, j in enumerate(SELECT):
        np.add.at(Dd[q], inv, NET[:, j] - NET[:, REF])
    nd = len(ud)
    def tstat(M):
        return M.mean(-1) / (M.std(-1, ddof=1) / np.sqrt(nd) + 1e-12)
    tobs = tstat(Dd)
    mx = np.zeros(2000)
    for b in range(2000):
        sg = RNG.choice((-1.0, 1.0), nd)
        mx[b] = tstat(Dd * sg[None, :]).max()
    NU = pd.DataFrame(dict(cell=[NAMES[j] for j in SELECT], t_obs=tobs, p_fw=[(mx >= t).mean() for t in tobs]))
    # 4. RND control
    XR, ZR = load("RND"); NR = nets(XR, ZR)
    blkr = np.where(XR.entry_ms.values >= HO_MS, "HOLDOUT", "DESIGN")
    ct = []
    for blk in ("DESIGN", "HOLDOUT", "ALL"):
        ii = np.arange(len(XR)) if blk == "ALL" else np.nonzero(blkr == blk)[0]
        jj = np.arange(len(X)) if blk == "ALL" else np.nonzero(blkv == blk)[0]
        for j in range(NC):
            ct.append(dict(blk=blk, cell=NAMES[j], N_rnd=len(ii), rnd_avg=NR[ii, j].mean(), rnd_ref=NR[ii, REF].mean(),
                           rnd_delta=(NR[ii, j] - NR[ii, REF]).mean(), exh_avg=NET[jj, j].mean(), exh_delta=(NET[jj, j] - NET[jj, REF]).mean(),
                           rnd_WR=(NR[ii, j] > 0).mean() * 100))
    CT = pd.DataFrame(ct)
    # 5. path decomposition
    fam_best = SS.groupby("fam").head(1).cell.tolist()
    show = list(dict.fromkeys(["REF"] + ([pick] if pick else []) + fam_best))
    pg = []
    for nm in show:
        j = NAMES.index(nm)
        for blk in ("ALL", "HOLDOUT"):
            idx = np.arange(len(X)) if blk == "ALL" else np.nonzero(blkv == blk)[0]
            g = path_group(Z, NET, idx, j); v = NET[idx, j]; tl = v[v <= 0].sum()
            for grp in ["CAPTURED", "LEFT_ON_TABLE", "GAVE_BACK", "NEVER_WORKED", "STOPPED_RECOVERED", "DRIFT_LOSS"]:
                m = g == grp
                pg.append(dict(cell=nm, blk=blk, group=grp, N=int(m.sum()), share=m.mean() * 100, sum_pct=v[m].sum(),
                               usd=v[m].sum() * NOTIONAL / 100, loss_share=v[m].sum() / tl * 100 if grp not in ("CAPTURED", "LEFT_ON_TABLE") else np.nan,
                               reach_med=np.median(Z["reach"][idx[m], j]) if m.any() else np.nan))
    PG = pd.DataFrame(pg)
    # REF stopped fills: how many went our way first
    st = Z["k"][:, REF] == 1
    reach = Z["reach"][st, REF]
    gb = dict(N_stopped=int(st.sum()), worked=float((Z["worked"][st, REF] == 1).mean() * 100),
              **{f"reach_ge{a}": float((reach >= a).mean() * 100) for a in (1, 2, 3, 4, 5, 6)}, reach_med=float(np.median(reach)))
    # 6. saved / deeper vs own base and vs REF by REF outcome; winners cut
    sd = []
    for j in SELECT:
        c = CL[j]; idx = np.arange(len(X))
        b = XNT if c["fam"] == "D2" else REF
        if c["mech"] in ("be", "lock", "trail", "np") or c["fam"] == "T":
            if c["fam"] == "T":
                fired = Z["k"][:, j] == 2
                fired &= (Z["xm"][:, j] < Z["xm"][:, REF])
            else:
                fired = np.isin(Z["k"][:, j], (3, 4))
            dr = Z["r"][:, j].astype(float) - Z["r"][:, b].astype(float)
            bad = int((np.abs(dr[~fired]) > 1e-4).sum())
            dn = NET[:, j] - NET[:, b]
            saved = fired & (dn > 0); cost = fired & ~saved
            sd.append(dict(cell=NAMES[j], kind="mech_vs_base", base=NAMES[b], N=len(idx), unfired=int((~fired).sum()), unfired_nonzero=bad,
                           fired=int(fired.sum()), saved_N=int(saved.sum()), saved_sum=dn[saved].sum(), cost_N=int(cost.sum()), cost_sum=dn[cost].sum(),
                           net_avg=dn.mean()))
        kr = Z["k"][:, REF]; d = NET[:, j] - NET[:, REF]
        for nm, m in (("REF stopped", kr == 1), ("REF TP", kr == 0), ("REF time", kr == 2)):
            sv_ = m & (d > 1e-9); dp = m & (d < -1e-9); eq_ = m & (np.abs(d) <= 1e-9)
            sd.append(dict(cell=NAMES[j], kind="vs_REF:" + nm, base="REF", N=int(m.sum()), saved_N=int(sv_.sum()), saved_sum=d[sv_].sum(),
                           cost_N=int(dp.sum()), cost_sum=d[dp].sum(), same_N=int(eq_.sum()), net_avg=d[m].mean() if m.any() else np.nan,
                           usd=d[m].sum() * NOTIONAL / 100))
        w = (kr == 0) & (Z["r"][:, REF] >= 9.99)
        cut = w & (Z["k"][:, j] != 0) & (d < -1e-9)
        sd.append(dict(cell=NAMES[j], kind="winners_cut_10pct", base="REF", N=int(w.sum()), cost_N=int(cut.sum()), cost_sum=d[cut].sum(),
                       usd=d[cut].sum() * NOTIONAL / 100, net_avg=d[cut].mean() if cut.any() else np.nan))
    SD = pd.DataFrame(sd)
    # 7. regimes (sign granularity), full span + per block
    rgr = []
    for nm in show:
        j = NAMES.index(nm)
        for reg in ("R1", "R4"):
            for s_ in (-1, 0, 1):
                for blk in ("ALL", "DESIGN", "HOLDOUT"):
                    m = (X[reg].values == s_) & ((blkv == blk) if blk != "ALL" else True)
                    if m.sum() == 0:
                        continue
                    rgr.append(dict(cell=nm, regime=reg, sign=s_, blk=blk, N=int(m.sum()), ref=NET[m, REF].mean(), avg=NET[m, j].mean(),
                                    delta=(NET[m, j] - NET[m, REF]).mean()))
    RG = pd.DataFrame(rgr)
    # 8. joint book + stop clustering
    jt = []; jsims = {}
    for nm in show:
        j = NAMES.index(nm)
        for blk in ("ALL", "HOLDOUT"):
            a, s, _ = joint(X, Z, NET, j, blk); jt.append(dict(cell=nm, blk=blk, **a))
            if blk == "ALL":
                jsims[nm] = s["r_day"]
    JT = pd.DataFrame(jt)
    _, rref = onesize(NET[:, REF], Z["xm"][:, REF] // DAY, D0, D1)
    worst = rref.sort_values().index[:10]
    cl = []
    for nm in show:
        j = NAMES.index(nm); xd = Z["xm"][:, j] // DAY; stp = np.isin(Z["k"][:, j], (1, 3))
        _, rr = onesize(NET[:, j], xd, D0, D1)
        spd = pd.Series(stp).groupby(xd).sum()
        hr = (Z["xm"][:, j] // H1)
        same_h = pd.Series(stp).groupby(hr).sum()
        cl.append(dict(cell=nm, stops_on_ref_worst10=int(spd.reindex(worst, fill_value=0).sum()),
                       pnl_on_ref_worst10=rr.reindex(worst).sum() * 100, worst_day=rr.min() * 100, worst_day_date=str(pd.to_datetime(rr.idxmin() * DAY, unit="ms").date()),
                       stops_day_mean=spd[spd > 0].mean(), stops_day_p99=spd.quantile(0.99), stops_day_max=spd.max(), same_hour_max=same_h.max()))
    CLU = pd.DataFrame(cl)
    # 9. E 2023-H1
    XE, ZE = load("E"); NE = nets(XE, ZE)
    oiv, blkE = e_block(XE)
    keepE = np.nonzero(~blkE)[0]
    keepE = keepE[one_pos(XE.pair.values[keepE], XE.entry_ms.values[keepE], ZE["xm"][keepE, REF])]
    erow = []
    for nm in show + [n for n in NAMES if n not in show]:
        j = NAMES.index(nm); v = NE[keepE, j]; d = v - NE[keepE, REF]
        erow.append(dict(cell=nm, N=len(v), avg=v.mean(), WR=(v > 0).mean() * 100, delta=d.mean()))
    EE = pd.DataFrame(erow)
    einfo = dict(cands=len(XE), oi_scored=int(np.isfinite(oiv).sum()), blocked=int(blkE.sum()), kept=len(keepE))
    print("E", einfo, flush=True)
    for nm, df in (("grid", GR), ("select", SS), ("null", NU), ("control", CT), ("paths", PG), ("saved", SD), ("regime", RG),
                   ("joint", JT), ("cluster", CLU), ("E", EE)):
        df.to_csv(f"{OUT}_{nm}.csv", index=False)
    pickle.dump(dict(pick=pick, show=show, gb=gb, einfo=einfo, jsims=jsims, names=NAMES, keepE=keepE, null_max=mx), open(os.path.join(D, "eval.pkl"), "wb"))
    pd.set_option("display.width", 260)
    print(SS.round(3).to_string())
    print(gb)


if __name__ == "__main__":
    main()
