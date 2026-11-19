#!/usr/bin/env python3
"""NOSPOT_HYBRID — EXTRAS: full ship-bar table for HYB (both exits), V1 and the HYB + T1 filter; the family-wise null of the theory-filter
family (block flag shuffled within the HYB trades); replays (PHAROS / PIEVERSE / PIPPIN / BLESS / JELLYJELLY); chart (anchored paths
winners vs losers + one-size equity); trade rows. Usage: venv/bin/python -I scripts/nospot_hybrid_extra.py"""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_exit_eval as E   # noqa: E402
import nospot_exit_sim as S    # noqa: E402
import nospot_sleeve_lib as L  # noqa: E402
import nospot_hybrid_build as HB  # noqa: E402
OD = os.path.join(L.R, "reports", "NOSPOT_HYBRID_2026-11-18_data")
OUT = os.path.join(L.R, "reports", "NOSPOT_HYBRID_2026-11-18")
DAY = L.DAY
JX = E.NAMES.index("REF_XE"); J4 = E.NAMES.index("A_TP4_SL10")
EX = {"XE": JX, "TP4SL10": J4}
rng = np.random.default_rng(3)
CUT = -0.01097


def bar_row(name, Y, Z, NET, idx, j, extra=None):
    net = NET[0]
    r = dict(cell=name)
    h = idx[Y.blk.values[idx] == "HOLDOUT"]
    st = E.stats(Y, Z, NET, h, j, "HOLDOUT"); sa = E.stats(Y, Z, NET, idx, j, "ALL")
    lo, hi = E.boot_ci(net[h, j], Z["xm"][h, j] // DAY)
    hv = [sa[f"h_{x}"] for x, _, _ in L.HALVES if sa[f"h_{x}_N"] >= 10]
    r.update(N_hold=st["N"], tpd_hold=st["tpd"], WR=st["WR"], avg_hold=st["avg"], ci_lo=lo, ci_hi=hi,
             halves_pos=f"{sum(x > 0 for x in hv)}/{len(hv)}", f20=st["f20"], b5=st["b5"], kelly=st["kelly"], kelly_day=st["kelly_day"],
             pair_share=st["pair_share"], day_share=st["day_share"], windows=st["ndays"], dcr_hold=st["dcr"], maxdd_hold=st["maxdd"],
             worst_day=st["worst_day"], avg_des=E.stats(Y, Z, NET, idx[Y.blk.values[idx] == "DESIGN"], j, "DESIGN").get("avg"))
    if extra:
        r.update(extra)
    return r


def main():
    Y = pd.read_pickle(os.path.join(OD, "Y.pkl"))
    X, _, Z = E.load()
    NET = E.nets(Y, Z); net = NET[0]
    C = pickle.load(open(os.path.join(OD, "core.pkl"), "rb")); SEL, sh = C["SEL"], C["shared"]
    W = pd.read_pickle(os.path.join(OD, "W.pkl"))
    NU = pd.read_csv(os.path.join(OD, "null.csv")).set_index(["cohort", "exit"])
    JT = pd.read_csv(os.path.join(OD, "joint.csv")); JT = JT[~JT.depth_cap].set_index("config")
    base = JT.loc["all four (EXH+FB+L4+DIST)"]
    # ---- ship-bar table
    rows = []
    for c, e, jn in (("HYB", "XE", "+ HYB XE"), ("HYB", "TP4SL10", "+ HYB TP4SL10"), ("V1_pos1h", "XE", "+ V1_pos1h XE (secondary)")):
        idx = SEL[(c, e)]
        rows.append(bar_row(f"{c} · {e}", Y, Z, NET, idx, EX[e],
                            dict(p_fw=NU.loc[(c, e), "p_fw"], joint_dcr=JT.loc[jn, "dcr"], joint_hold_dcr=JT.loc[jn, "hold_dcr"],
                                 joint_maxdd=JT.loc[jn, "maxdd"])))
        if c == "HYB":
            nn = idx[~sh[e]]
            jn2 = jn + " non-DIST-shared only"
            rows.append(bar_row(f"{c} · {e} · not DIST-shared", Y, Z, NET, nn, EX[e],
                                dict(joint_dcr=JT.loc[jn2, "dcr"], joint_hold_dcr=JT.loc[jn2, "hold_dcr"], joint_maxdd=JT.loc[jn2, "maxdd"])))
    t1 = np.nan_to_num(W.glA.values < 1.0, nan=0).astype(bool)
    kept = W.xi_row.values[~t1]
    jn = "+ HYB XE filter: T1 shorts crowded (glA<1) block"
    # theory-filter family null: shuffle each candidate's block flag within the trades, stat = kept avg − all avg (ALL / HOLDOUT)
    md_p4 = np.nanmedian(W.pos4.values[W.blk.values == "DESIGN"]); md_dd = np.nanmedian(W.dd_hi.values[W.blk.values == "DESIGN"])
    f0 = lambda a: np.nan_to_num(a, nan=0).astype(bool)
    with np.errstate(invalid="ignore"):
        fam = {"T1": f0(W.glA.values < 1), "T2": f0(W.f_last.values < 0), "T3": f0(W.oi_chg1h.values > 0), "T4": f0(W.pos4.values > md_p4),
               "T5": f0(W.btc24h.values > 0), "T6": f0(W.dd_hi.values > md_dd)}
    y = W.y_xe.values; hold = W.blk.values == "HOLDOUT"
    obs = {k: (y[~b].mean() - y.mean(), y[~b & hold].mean() - y[hold].mean()) for k, b in fam.items()}
    mx = np.zeros((2000, 2))
    for s in range(2000):
        best = [-9, -9]
        for k, b in fam.items():
            p = rng.permutation(b)
            best[0] = max(best[0], y[~p].mean() - y.mean()); best[1] = max(best[1], y[~p & hold].mean() - y[hold].mean())
        mx[s] = best
    FN = pd.DataFrame([dict(cand=k, lift_all=o[0], p_fw_all=(mx[:, 0] >= o[0]).mean(), lift_hold=o[1], p_fw_hold=(mx[:, 1] >= o[1]).mean())
                       for k, o in obs.items()])
    FN.to_csv(os.path.join(OD, "filter_null.csv"), index=False)
    print(FN.round(3).to_string())
    rows.append(bar_row("HYB · XE + T1 (block glA < 1)", Y, Z, NET, kept, JX,
                        dict(p_fw=FN.set_index("cand").loc["T1", "p_fw_hold"], joint_dcr=JT.loc[jn, "dcr"], joint_hold_dcr=JT.loc[jn, "hold_dcr"],
                             joint_maxdd=JT.loc[jn, "maxdd"])))
    kept4 = kept  # same entries under TP4 (one-position re-check not needed for a read of the exit flip)
    rows.append(bar_row("HYB · TP4SL10 entries + T1 (same entries)", Y, Z, NET, kept4, J4))
    BR = pd.DataFrame(rows)
    BR["joint_base_dcr"] = base.dcr; BR["joint_base_hold_dcr"] = base.hold_dcr; BR["joint_base_maxdd"] = base.maxdd
    BR.to_csv(os.path.join(OD, "shipbar.csv"), index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(BR.round(3).to_string())
    # T1 kept: overlap with DIST
    shx = W.shared.values
    print("T1 kept shared with DIST:", (shx & ~t1).sum(), "of", (~t1).sum(), "avg shared", y[shx & ~t1].mean(), "not", y[~shx & ~t1].mean(),
          "HOLDOUT not-shared", y[~shx & ~t1 & hold].mean(), (~shx & ~t1 & hold).sum())

    # ---- replays
    rp = []
    PAIRS = ["PHAROSUSDT", "PIEVERSEUSDT", "PIPPINUSDT", "BLESSUSDT", "JELLYJELLYUSDT"]
    for e, j in EX.items():
        idx = SEL[("HYB", e)]
        for p in PAIRS:
            m = Y.pair.values[idx] == p
            ii = idx[m]
            rp.append(dict(src="history (X.pkl)", pair=p, exit=e, kind="summary", N=int(m.sum()), avg=net[ii, j].mean() if m.any() else np.nan,
                           sum=net[ii, j].sum(), WR=(net[ii, j] > 0).mean() * 100 if m.any() else np.nan,
                           ns0_N=int(((Y.pair.values == p) & Y.NS0.values).sum())))
            for i in ii[Y.entry_ms.values[ii] >= L.TS("2026-11-01")]:
                rp.append(dict(src="history (X.pkl)", pair=p, exit=e, kind="trade", entry=str(pd.Timestamp(int(Y.entry_ms.values[i]), unit="ms"))[:16],
                               exit_kind=["TP", "SL", "TIME", "DYN"][Z["k"][i, j]], net=net[i, j], pos4=Y.pos4.values[i], glA=Y.glA.values[i],
                               shared_DIST=bool(sh[e][list(idx).index(i)])))
    # live rows (11-14 → 11-18): every 15m RSI-cross row; pos4 computed for all; hybrid = vd ≥ 0.6366 ∧ pos4 ≤ cut
    LV = pd.read_pickle(os.path.join(OD, "live_rows.pkl"))
    LD = os.path.join(L.ND, "live")
    HB.TB.K5M_SRC.append(os.path.join(LD, "k5m")); HB.TB.FUND_SRC.append(os.path.join(LD, "funding")); S.TB.FUND_SRC.append(os.path.join(LD, "funding"))
    MI = HB.NB.met_index(sorted(set(HB.NB.MET_DIRS + [os.path.join(L.ND, "metrics"), os.path.join(LD, "metrics")])))
    cells = [c for c in S.cells() if c["name"] in ("REF_XE", "A_TP4_SL10")]
    for p, g in LV[LV.rsx15.astype(bool)].groupby("pair"):
        _, f = HB.work((p, g.index.values, g.entry_ms.values, MI.get(p, [])))
        z = np.load(os.path.join(L.R, "reports", "NOSPOT_EXIT_data", "live1m", p + ".npz")); one = (z["t"], z["o"], z["h"], z["l"], z["c"])
        rr = pd.DataFrame(dict(xi=g.index.values, entry_ms=g.entry_ms.values, atrp=g.atrp.values,
                               slip=np.maximum(0.10, L.slip_model(g.q5.values.astype(float)))))
        o, _ = S.run_rows(p, rr, cells, one1m=one)
        for q in range(len(g)):
            vd = float(g.vd.values[q]); p4 = f["pos4"][q]
            rec = dict(src="live rows 11-14→11-18", pair=p, kind="RSI-cross row", entry=str(pd.Timestamp(int(g.entry_ms.values[q]), unit="ms"))[:16],
                       vd=vd, pos4=p4, glA=float(g.glA.values[q]), hyb=bool(vd >= 0.6366 and p4 <= CUT), hyb_if_vd_passed=bool(p4 <= CUT),
                       open=bool(o["ok"][q] and o["xm"][q, 0] > z["t"][-1]))
            for cq, cd in enumerate(cells):
                if o["ok"][q]:
                    k_ = o["k"][q, cq]; tk = k_ != 0; s_ = rr.slip.values[q]
                    rec[cd["name"]] = f'{["TP", "SL", "TIME", "DYN"][k_]} {o["r"][q, cq] + o["f"][q, cq] - s_ - 0.05 - (s_ + 0.05 if tk else 0.02):+.2f}'
            rp.append(rec)
    RP = pd.DataFrame(rp); RP.to_csv(OUT + "_replays.csv", index=False)
    print(RP.round(3).to_string())

    # ---- chart
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    P = {}
    import glob
    for f in glob.glob(os.path.join(L.R, "reports", "NOSPOT_EXIT_data", "path", "*.npz")):
        zz = np.load(f)
        for a_, b_ in zip(zz["xi"], zz["p"]):
            P[int(a_)] = b_.astype(float)
    idx = SEL[("HYB", "XE")]; v = net[idx, JX]; wk = Z["worked"][idx, JX] == 1
    g3 = np.where(v > 0, "RUNNER", np.where(wk, "GAVE_BACK", "NEVER_WORKED"))
    tt = np.arange(289) * 5 / 60
    cols = {"RUNNER": "#2a7f62", "GAVE_BACK": "#d08c2c", "NEVER_WORKED": "#b8433b"}
    for grp, col in cols.items():
        ps = np.array([P[int(Y.xi.values[i])][:289] for i in idx[g3 == grp] if int(Y.xi.values[i]) in P])
        md = np.nanmedian(ps, 0); q1, q3 = np.nanpercentile(ps, [25, 75], 0)
        ax[0].plot(tt, md, color=col, lw=2, label=f"{grp} (n={len(ps)})"); ax[0].fill_between(tt, q1, q3, color=col, alpha=0.12)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xlabel("hours after entry"); ax[0].set_ylabel("short P&L % (close path, gross)")
    ax[0].set_title("HYB · XE: anchored paths by outcome (median, IQR)"); ax[0].legend(fontsize=8); ax[0].set_ylim(-15, 15)
    # winners vs losers, deep cut vs shallow and glA<1 vs ≥1 (the theory reads)
    for lab, m, col, ls in (("glA ≥ 1 (kept by T1)", ~t1, "#2a7f62", "-"), ("glA < 1 (blocked by T1)", t1, "#b8433b", "-")):
        ps = np.array([P[int(r)][:289] for r in W.xi.values[m] if int(r) in P])
        ax[1].plot(tt, np.nanmean(ps, 0), color=col, ls=ls, lw=2, label=f"{lab} mean (n={len(ps)})")
        ax[1].plot(tt, np.nanmedian(ps, 0), color=col, ls=":", lw=1.5, label=f"{lab} median")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("hours after entry"); ax[1].set_title("HYB · XE entries: crowd long vs crowd short")
    ax[1].legend(fontsize=8)
    for lab, ii, j, col in (("HYB · XE", SEL[("HYB", "XE")], JX, "#1f4e9c"), ("HYB · TP4/SL10", SEL[("HYB", "TP4SL10")], J4, "#7d5ba6"),
                            ("HYB · XE + T1", kept, JX, "#2a7f62"), ("NS0 · XE", SEL[("NS0", "XE")], JX, "#888888"),
                            ("NS0 not HYB · XE", SEL[("NS0_notHYB", "XE")], JX, "#b8433b")):
        r_ = pd.Series(net[ii, j] * L.SIZE / 100).groupby(Z["xm"][ii, j] // DAY).sum()
        d0, d1 = L.TS("2023-07-01") // DAY, L.TS("2026-11-18") // DAY
        r_ = r_.reindex(np.arange(d0, d1), fill_value=0.0)
        eq = np.cumprod(1 + r_.values)
        ax[2].plot(pd.to_datetime(np.arange(d0, d1) * DAY, unit="ms"), eq, color=col, lw=1.6, label=lab)
    ax[2].set_yscale("log"); ax[2].axvline(pd.Timestamp("2025-07-01"), color="k", ls="--", lw=0.8)
    import matplotlib.dates as mdates
    ax[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax[2].set_title("one-size equity (0.61 × equity per fill, no slot cap)"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT + "_chart.png", dpi=110)
    # ---- trade rows
    tr = []
    for (c, e), ii in SEL.items():
        if c not in ("HYB", "V1_pos1h"):
            continue
        j = EX[e]
        d = pd.DataFrame(dict(cohort=c, exit=e, pair=Y.pair.values[ii], entry=pd.to_datetime(Y.entry_ms.values[ii], unit="ms"),
                              entry_ms=Y.entry_ms.values[ii], exit_ms=Z["xm"][ii, j], blk=Y.blk.values[ii], half=Y.half.values[ii],
                              exit_kind=np.array(["TP", "SL", "TIME", "DYN"])[Z["k"][ii, j]], gross=Z["r"][ii, j], funding=Z["f"][ii, j],
                              slip=Y.slip.values[ii], net=net[ii, j], net20=NET[1][ii, j], worked=Z["worked"][ii, j], reach=Z["reach"][ii, j],
                              shared_DIST=sh[e] if c == "HYB" else np.nan))
        for v_ in ("pos4", "pos1", "acc4", "pos_now", "glA", "f_last", "oi_chg1h", "dd_hi", "vd", "rsi1h", "atrp", "q5", "R1reg", "adx15"):
            d[v_] = Y[v_].values[ii]
        tr.append(d)
    TR = pd.concat(tr); TR.to_csv(OUT + "_trades.csv.gz", index=False)
    print("trades", len(TR))


if __name__ == "__main__":
    main()
