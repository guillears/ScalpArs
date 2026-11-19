#!/usr/bin/env python3
"""NOSPOT_BTCBEAR stage 3: spot-coin EXHAUSTION split by the same switches (1m, REF + TP4 SL10) · joint book via joint_short_sim
(all four + candidate; in-sample / haircut 30 / 50 %) · one-size DCR alone · clustering vs EXH on ON days · replays 11-10 → 11-18 with the
BTC regime at each signal · chart · trade rows.  Usage: venv/bin/python -I scripts/nospot_btcbear_stage3.py"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_btcbear_lib as B   # noqa: E402
L = B.L; S = B.S; EV = B.EV
import joint_short_sim as JS     # noqa: E402
OUT = os.path.join(B.R, "reports", "NOSPOT_BTCBEAR_2026-11-18")
CL = S.cells(); NAMES = [c["name"] for c in CL]
DAY = B.DAY
CANDS = [("a", "R4", "A_TP4_SL10"), ("a", "R4", "T_48h"), ("a", "R4", "REF_XE"), ("d", "R4", "A_TP4_SL10"), ("e", "R10", "A_TP5_SL10"),
         ("c", "R4", "A_TP4_SL10")]


def main():
    RS = B.regime_series(); Y, Z = B.load(); NET = B.nets(Y, Z); net, net20, _ = NET
    T = Y.entry_ms.values.astype(np.int64); blk = Y.blk.values; pair = Y.pair.values
    REG = {d: B.at(RS, d, T) for d in B.DEFS}
    # ---------- spot-coin EXHAUSTION
    f = os.path.join(B.DD, "exh_1m.pkl")
    if not os.path.exists(f):
        Ex = pd.read_csv(os.path.join(B.R, "reports", "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
        Ex = Ex[~Ex.dl582_block.astype(bool)].reset_index(drop=True)
        Ex["slip"] = np.maximum(0.10, L.slip_model(Ex.q5.values.astype(float))); Ex["xi"] = np.arange(len(Ex))
        cx = [c for c in CL if c["name"] in ("REF_XE", "A_TP4_SL10")]
        R_ = []
        for p, g in Ex.groupby("pair"):
            o, _ = S.run_rows(p, g[["xi", "entry_ms", "atrp", "slip"]], cx); R_.append(o)
        ex = {k: np.concatenate([o[k] for o in R_]) for k in R_[0]}
        ok = ex["ok"]; xi = ex["xi"][ok]
        s = Ex.slip.values[xi][:, None]; kk = ex["k"][ok]; tk = kk != 0
        nv = ex["r"][ok] + ex["f"][ok] - s - 0.05 - np.where(tk, s + 0.05, 0.02)
        E = Ex.iloc[xi][["pair", "entry_ms", "half"]].reset_index(drop=True)
        E["net_REF"] = nv[:, 0]; E["net_TP4"] = nv[:, 1]; E["xm_REF"] = ex["xm"][ok][:, 0]; E["k_REF"] = kk[:, 0]
        E.to_pickle(f)
    E = pd.read_pickle(f); E["blk"] = np.where(E.entry_ms < B.HOLD0, "DESIGN", "HOLDOUT")
    sp = []
    for d in B.DEFS:
        rg = B.at(RS, d, E.entry_ms.values)
        for bk in ("DESIGN", "HOLDOUT", "ALL"):
            for state, sv in (("ON", -1), ("OFF", 1)):
                m = (rg == sv) & ((E.blk.values == bk) if bk != "ALL" else True)
                sp.append(dict(defn=d, blk=bk, state=state, N=int(m.sum()), REF=E.net_REF.values[m].mean(), TP4_SL10=E.net_TP4.values[m].mean(),
                               WR_REF=(E.net_REF.values[m] > 0).mean() * 100))
    pd.DataFrame(sp).to_csv(OUT + "_spot.csv", index=False)
    # ---------- joint book
    TJ, _ = JS.load(); TJ = TJ[TJ.sleeve.isin(["EXH", "FB", "L4", "DIST"])].reset_index(drop=True)
    hidx = int(B.HOLD0 // DAY - JS.DAYS[0])
    def run(Tc):
        Tc = Tc.sort_values(["entry_ms", "prio"]).reset_index(drop=True); s = JS.sim(Tc)
        full = JS.stats(s["r_day"]); ho = JS.stats(s["r_day"][hidx:])
        m = s["taken"]
        return Tc, s, dict(N=int(m.sum()), tpd=m.sum() / JS.ND, dcr=full["dcr"], maxdd=full["maxdd"], worst=full["worst_day"],
                           dcr_H=ho["dcr"], maxdd_H=ho["maxdd"], worst_H=ho["worst_day"])
    jt = []
    T0, s0, r0 = run(TJ); jt.append(dict(book="all four (EXH + FB + L4 + DIST)", **r0))
    TRADES = {}
    for d, c, e in CANDS:
        j = NAMES.index(e)
        ii = B.one_pos(Y, Z, np.nonzero(Y[c].values & (REG[d] == -1))[0], j)
        k = Z["k"][ii, j]; tk = k != 0
        C = pd.DataFrame(dict(sleeve="NSB", pair=pair[ii], entry_ms=T[ii], exit_ms=Z["xm"][ii, j], net=net[ii, j], nt=1 + tk,
                              stop=(k == 1) | (k == 3), depth=JS.depth_of(Y.q5.values[ii])))
        C = C[(C.entry_ms >= JS.D0ms) & (C.entry_ms < JS.D1ms)]
        TRADES[(d, c, e)] = (ii, j)
        r1 = np.zeros(JS.ND); xd = C.exit_ms.values // DAY - JS.DAYS[0]; okk = (xd >= 0) & (xd < JS.ND)
        np.add.at(r1, xd[okk], JS.X * C.net.values[okk] / 100)
        alone = JS.stats(r1); aloneH = JS.stats(r1[hidx:])
        for hc in (0.0, 0.3, 0.5):
            Cc = C.copy(); Cc["net"] = Cc.net - hc * C.net.mean(); Cc["prio"] = 5
            Tc, s, r = run(pd.concat([TJ, Cc], ignore_index=True))
            mN = (Tc.sleeve.values == "NSB") & s["taken"]
            jt.append(dict(book=f"+ {d} · {c} · {e}" + (f" (haircut {int(hc * 100)} %)" if hc else ""), cand_taken=int(mN.sum()),
                           cand_tpd=mN.sum() / JS.ND, cand_avg_taken=Tc.net.values[mN].mean(),
                           alone_dcr=alone["dcr"], alone_dcr_H=aloneH["dcr"], alone_worst=alone["worst_day"], **r))
            if hc == 0 and (d, c, e) == CANDS[0]:
                SJ = (Tc, s)
    JT = pd.DataFrame(jt); JT.to_csv(OUT + "_joint.csv", index=False)
    # ---------- clustering vs EXH on ON days (source cell)
    Tc, s = SJ; m = s["taken"] & s["inwin"]
    by = pd.DataFrame(dict(d=s["xd"][m], sl=Tc.sleeve.values[m], p=s["pnl"][m], stop=Tc.stop.values[m], prev=s["prev"][s["xd"][m]]))
    by["r"] = by.p / by.prev * 100
    dd = by.pivot_table(index="d", columns="sl", values="r", aggfunc="sum").fillna(0)
    st = by[by.stop].pivot_table(index="d", columns="sl", values="p", aggfunc="size").fillna(0)
    cl = dict(corr_NSB_EXH=dd["NSB"].corr(dd["EXH"]) if "EXH" in dd else np.nan,
              corr_NSB_rest=dd["NSB"].corr(dd.drop(columns="NSB").sum(axis=1)),
              days_NSB_stops=int((st.get("NSB", 0) > 0).sum()), days_both_stop=int(((st.get("NSB", 0) > 0) & (st.drop(columns="NSB").sum(axis=1) > 0)).sum()),
              nsb_stops_p99=np.percentile(st["NSB"][st["NSB"] > 0], 99), nsb_stops_max=st["NSB"].max())
    wd = np.argsort(s["r_day"])[:8]; wrows = []
    for d_ in wd:
        g = by[by.d == d_]
        wrows.append(dict(date=str(pd.to_datetime((JS.DAYS[0] + d_) * DAY, unit="ms").date()), day_pct=s["r_day"][d_] * 100,
                          parts=" · ".join(f"{k} {v.r.sum():+.1f} % ({int(v.stop.sum())}/{len(v)} stops)" for k, v in g.groupby("sl"))))
    pd.DataFrame(wrows).to_csv(OUT + "_worstdays.csv", index=False)
    pd.Series(cl).to_csv(OUT + "_cluster.csv")
    # ---------- replays
    LD = os.path.join(L.ND, "live"); S.TB.FUND_SRC.append(os.path.join(LD, "funding"))
    cx = [c for c in CL if c["name"] in ("REF_XE", "A_TP4_SL10", "T_48h", "A_TP5_SL10")]
    rp = []
    for p in ["PIPPINUSDT", "BLESSUSDT", "JELLYJELLYUSDT", "PHAROSUSDT"]:
        fr = os.path.join(LD, "rows", p + ".pkl")
        if not os.path.exists(fr):
            continue
        Rw = pd.read_pickle(fr); Rw = Rw[Rw.entry_ms >= L.TS("2026-11-10")].reset_index(drop=True)
        Rw["slip"] = np.maximum(0.10, L.slip_model(Rw.q5.values.astype(float)))
        M = L.rule_masks(Rw); masks = {"NS0": L.ns0_mask(Rw), "R1": M["R1_OI_UNWIND"][0], "R4": M["R4_TOP_DISTRIB"][0], "R10": M["R10_OI_FROM_MAX"][0]}
        z = np.load(os.path.join(S.D, "live1m", p + ".npz")); one = (z["t"], z["o"], z["h"], z["l"], z["c"])
        for c, mm in masks.items():
            if not mm.any():
                rp.append(dict(pair=p, cohort=c, entry="no signal")); continue
            rr = pd.DataFrame(dict(xi=np.nonzero(mm)[0], entry_ms=Rw.entry_ms.values[mm], atrp=Rw.atrp.values[mm], slip=Rw.slip.values[mm]))
            o, _ = S.run_rows(p, rr, cx, one1m=one); busy = -1
            for i in range(len(rr)):
                if rr.entry_ms.values[i] < busy or not o["ok"][i]:
                    continue
                busy = o["xm"][i, 0]; te = int(rr.entry_ms.values[i]); s_ = rr.slip.values[i]
                rec = dict(pair=p, cohort=c, entry=str(pd.Timestamp(te, unit="ms"))[:16], open=bool(o["xm"][i, 0] > z["t"][-1]),
                           btc_5m_label=RS["a_label"][1][np.searchsorted(RS["a_label"][0], te, side="right") - 1])
                for dname in B.DEFS:
                    rec[f"reg_{dname}"] = {-1: "BEAR", 1: "bull", 0: "chop"}.get(B.at(RS, dname, [te])[0], "?")
                for q, cdef in enumerate(cx):
                    k_ = o["k"][i, q]; tk_ = k_ != 0
                    rec[cdef["name"]] = f'{["TP", "SL", "TIME", "DYN"][k_]} {o["r"][i, q] + o["f"][i, q] - s_ - 0.05 - (s_ + 0.05 if tk_ else 0.02):+.2f}'
                rp.append(rec)
    pd.DataFrame(rp).to_csv(OUT + "_replays.csv", index=False)
    # ---------- trade rows
    tr = []
    for (d, c, e), (ii, j) in TRADES.items():
        g = EV.path_group(Z, NET, ii, j)
        tr.append(pd.DataFrame(dict(defn=d, cohort=c, cell=e, pair=pair[ii], entry_ms=T[ii], exit_ms=Z["xm"][ii, j], blk=blk[ii],
                                    half=Y.half.values[ii], kind=Z["k"][ii, j], net=np.round(net[ii, j], 3), net20=np.round(net20[ii, j], 3),
                                    path_group=g, **{f"reg_{x}": REG[x][ii] for x in B.DEFS})))
    TR = pd.concat(tr); TR.to_csv(OUT + "_trades.csv.gz", index=False)
    # ---------- chart
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    d0, d1 = B.SPAN0 // DAY, B.SPAN1 // DAY; days = np.arange(d0, d1); dts = pd.to_datetime(days * DAY, unit="ms")
    j = NAMES.index("A_TP4_SL10")
    for d, axx in (("a", ax[0]), ("d", ax[1])):
        for state, sv, col in (("BTC bear (ON)", -1, "tab:red"), ("BTC bull (OFF)", 1, "tab:blue"), ("always on", None, "grey")):
            base = Y["R4"].values if sv is None else Y["R4"].values & (REG[d] == sv)
            ii = B.one_pos(Y, Z, np.nonzero(base)[0], j)
            r = pd.Series(net[ii, j] * L.SIZE / 100).groupby(Z["xm"][ii, j] // DAY).sum().reindex(days, fill_value=0.0)
            axx.plot(dts, np.cumprod(1 + r.values), color=col, label=state, lw=1.3)
        du = B.duty(RS, d, d0, d1).reindex(days, fill_value=0)
        if d == "d":
            on = du.values > 0.5
            for k0 in np.nonzero(on & ~np.r_[False, on[:-1]])[0]:
                k1 = k0 + np.argmax(~on[k0:]) if (~on[k0:]).any() else len(on)
                axx.axvspan(dts[k0], dts[min(k1, len(dts) - 1)], color="tab:red", alpha=0.08)
        else:
            axx.fill_between(dts, 0.02, 0.02 + 0.3 * du.rolling(7).mean().values, color="tab:red", alpha=0.25, transform=axx.get_xaxis_transform(),
                             label="share of day BTC-bear (7d avg)")
        axx.axvline(pd.Timestamp("2025-07-01"), color="k", ls="--", lw=0.8)
        axx.set_yscale("log"); axx.set_title(f"R4 TOP_DISTRIB · TP4 / SL10 · one size 0.61 — switch ({d}) {B.DEFNAME[d]}", fontsize=9)
        axx.legend(fontsize=8); axx.grid(alpha=0.3)
    # anchored paths
    PP = {}
    for fp in glob.glob(os.path.join(S.D, "path", "*.npz")):
        zz = np.load(fp); PP.update({int(a): b for a, b in zip(zz["xi"], zz["p"])})
    xi = Z["xi"]; pos = {int(x): i for i, x in enumerate(xi)}
    tt = np.arange(0, 48 * 12 + 1) / 12
    for c, ls in (("R4", "-"), ("NS0", ":")):
        for state, sv, col in (("bear", -1, "tab:red"), ("bull", 1, "tab:blue")):
            ii = B.one_pos(Y, Z, np.nonzero(Y[c].values & (REG["a"] == sv))[0], NAMES.index("REF_XE"))
            M_ = np.array([PP[int(xi[i])] for i in ii if int(xi[i]) in PP], float)
            if not len(M_):
                continue
            med = np.nanmedian(M_, 0); ax[2].plot(tt, med, color=col, ls=ls, label=f"{c} BTC-5m {state} median (N {len(M_)})")
            if c == "R4":
                ax[2].fill_between(tt, np.nanpercentile(M_, 25, 0), np.nanpercentile(M_, 75, 0), color=col, alpha=0.10)
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xlabel("hours after entry"); ax[2].set_ylabel("short P&L % (close, gross)")
    ax[2].set_title("Anchored short paths from entry: BTC 5m bear vs bull (R4 band = IQR)", fontsize=9); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    ax[2].set_ylim(-12, 15)
    plt.tight_layout(); plt.savefig(OUT + "_chart.png", dpi=110)
    print(JT.round(3).to_string()); print(cl)


if __name__ == "__main__":
    main()
