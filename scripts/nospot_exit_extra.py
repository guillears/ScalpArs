#!/usr/bin/env python3
"""NOSPOT_EXIT — extra reads after the selection (nothing re-fit):
  · small-TP addendum S: breakeven WR vs WR, /day, DCR, loser clustering (stops per day, same-hour stop count across pairs, worst day),
    +1 % hit rate by cohort and on the random control; dedicated SL 3 % rows
  · post-selection robustness of each pick (stop 6 / 12 %, 48 h hold)
  · EXHAUSTION spot-coin cross-check (BASELINE_TRADES_EXHAUSTION, 1m) — REF vs the picks
  · replays PIPPIN / BLESS / JELLYJELLY / PHAROS (live 1m, NS0 / R1 / R4 entries from the NOSPOT_SLEEVE live rows)
  · chart (anchored paths with exit levels + one-size equity curves), trade rows CSV.
Usage: venv/bin/python -I scripts/nospot_exit_extra.py"""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_sleeve_lib as L   # noqa: E402
import nospot_exit_sim as S     # noqa: E402
import nospot_exit_eval as EV   # noqa: E402
OUT = EV.OUT; D = S.D; DAY = L.DAY
NAMES = EV.NAMES; CL = EV.CL; REF = EV.REF


def main():
    X, Y, Z = EV.load(); NET = EV.nets(Y, Z); net = NET[0]
    E = pickle.load(open(os.path.join(D, "eval.pkl"), "rb")); SEL = E["SEL"]; picks = E["picks"]
    blkv = Y.blk.values; T = Y.entry_ms.values.astype(np.int64)
    # 1. small-TP family S
    rows = []
    for c in EV.COH + ["RND"]:
        for j in [i for i, cc in enumerate(CL) if cc["fam"] == "S"]:
            idx = SEL[(c, j)]
            for blk in ("DESIGN", "HOLDOUT", "ALL"):
                ii = idx if blk == "ALL" else idx[blkv[idx] == blk]
                if c == "RND":
                    ii = ii[::5]
                st = EV.stats(Y, Z, NET, ii, j, blk)
                k = Z["k"][ii, j]; xm = Z["xm"][ii, j]; stp = k == 1
                spd = np.bincount((xm[stp] // DAY - xm.min() // DAY).astype(int)) if stp.any() else np.zeros(1)
                act = spd[spd > 0] if (spd > 0).any() else np.zeros(1)
                hr = pd.Series(1, index=xm[stp] // 3_600_000).groupby(level=0).size() if stp.any() else pd.Series([0])
                rows.append(dict(cohort=c, cell=NAMES[j], blk=blk, N=st.get("N"), tpd=st.get("tpd"), WR=st.get("WR"), BE=st.get("BE"),
                                 avg=st.get("avg"), avg_win=st.get("avg_win"), avg_loss=st.get("avg_loss"), f20=st.get("f20"),
                                 hit_tp=(k == 0).mean() * 100, stop=(k == 1).mean() * 100, time=(k == 2).mean() * 100,
                                 dcr=st.get("dcr"), maxdd=st.get("maxdd"), worst_day=st.get("worst_day"),
                                 stops_day_mean=act.mean(), stops_day_p99=np.percentile(act, 99), stops_day_max=act.max(),
                                 same_hour_stops_p99=np.percentile(hr.values, 99), same_hour_stops_max=hr.max(),
                                 kelly=st.get("kelly"), kelly_day=st.get("kelly_day")))
    SM = pd.DataFrame(rows); SM.to_csv(OUT + "_smalltp.csv", index=False)
    # +1 % hit rate before −Y (path WORKED at each S stop, 24 h) per cohort, REF one-pos set
    hit = []
    for c in EV.COH + ["RND"]:
        idx = SEL[(c, REF)] if c != "RND" else SEL[(c, REF)][::5]
        for sl in (2, 3, 4, 6):
            j = NAMES.index(f"S_TP1_SL{sl}_24h")
            for blk in ("DESIGN", "HOLDOUT", "ALL"):
                ii = idx if blk == "ALL" else idx[blkv[idx] == blk]
                hit.append(dict(cohort=c, sl=sl, blk=blk, N=len(ii), hit1_before_stop=(Z["k"][ii, j] == 0).mean() * 100,
                                worked=Z["worked"][ii, j].mean() * 100))
    HT = pd.DataFrame(hit); HT.to_csv(OUT + "_hit1.csv", index=False)
    # 2. robustness + EXH cross-check + replays (picks; else the best DESIGN cell per cohort for the descriptive reads)
    SS = pd.read_csv(OUT + "_select.csv")
    shown = list(picks) if picks else []
    for c in EV.COH:
        b = SS[SS.cohort == c].sort_values("avg", ascending=False).iloc[0]
        if (c, b.cell) not in shown:
            shown.append((c, b.cell))
    cells_x = [CL[REF]] + [CL[NAMES.index(n)] for n in dict.fromkeys(p[1] for p in shown)]
    rob = []
    for c, n in shown:
        j = NAMES.index(n); base = CL[j]; var = []
        for sl in (6, 12):
            d = dict(base); d["sl"] = sl; d["name"] = f"{n}|SL{sl}"; var.append(d)
        d = dict(base); d["hold"] = 48; d["name"] = f"{n}|48h"; var.append(d)
        idx = SEL[(c, j)]
        R_ = []
        for p, g in pd.Series(idx).groupby(Y.pair.values[idx]):
            rr = pd.DataFrame(dict(xi=g.values, entry_ms=T[g.values], atrp=Y.atrp.values[g.values], slip=Y.slip.values[g.values]))
            o, _ = S.run_rows(p, rr, var); R_.append(o)
        rr_ = {k: np.concatenate([o[k] for o in R_]) for k in R_[0]}
        s = Y.slip.values[rr_["xi"]][:, None]; tk = rr_["k"] != 0
        nv = rr_["r"] + rr_["f"] - s - 0.05 - np.where(tk, s + 0.05, 0.02)
        hb = blkv[rr_["xi"]]
        for q, v in enumerate(var):
            for blk in ("DESIGN", "HOLDOUT"):
                m = (hb == blk) & rr_["ok"]
                rob.append(dict(cohort=c, cell=v["name"], blk=blk, N=int(m.sum()), avg=nv[m, q].mean()))
        for blk in ("DESIGN", "HOLDOUT"):
            ii = idx[blkv[idx] == blk]
            rob.append(dict(cohort=c, cell=n + " (as selected)", blk=blk, N=len(ii), avg=net[ii, j].mean()))
    RB = pd.DataFrame(rob); RB.to_csv(OUT + "_robust.csv", index=False)
    # EXH cross-check
    Ex = pd.read_csv(os.path.join(L.R, "reports", "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    Ex = Ex[~Ex.dl582_block.astype(bool)].reset_index(drop=True)
    Ex["slip"] = np.maximum(0.10, L.slip_model(Ex.q5.values.astype(float))); Ex["xi"] = np.arange(len(Ex))
    R_ = []
    for p, g in Ex.groupby("pair"):
        o, _ = S.run_rows(p, g[["xi", "entry_ms", "atrp", "slip"]], cells_x); R_.append(o)
    ex = {k: np.concatenate([o[k] for o in R_]) for k in R_[0]}
    ok = ex["ok"]; xi = ex["xi"][ok]
    s = Ex.slip.values[xi][:, None]; kk = ex["k"][ok]; tk = kk != 0
    nv = ex["r"][ok] + ex["f"][ok] - s - 0.05 - np.where(tk, s + 0.05, 0.02)
    hv = Ex.half.values[xi]; xc = []
    for q, cdef in enumerate(cells_x):
        rec = dict(cell=cdef["name"], N=int(ok.sum()), src1m=ex["src1m"][ok].mean() * 100, avg=nv[:, q].mean(), WR=(nv[:, q] > 0).mean() * 100,
                   tp=(kk[:, q] == 0).mean() * 100, ref_csv_r=Ex.r.values[xi].mean(), sim_r=ex["r"][ok][:, q].mean())
        for h in sorted(set(hv)):
            rec[h] = nv[hv == h, q].mean()
        xc.append(rec)
    XC = pd.DataFrame(xc); XC.to_csv(OUT + "_exh_crosscheck.csv", index=False)
    # replays
    LD = os.path.join(L.ND, "live")
    S.TB.FUND_SRC.append(os.path.join(LD, "funding"))
    rp = []
    for p in ["PIPPINUSDT", "BLESSUSDT", "JELLYJELLYUSDT", "PHAROSUSDT"]:
        f = os.path.join(LD, "rows", p + ".pkl")
        if not os.path.exists(f):
            rp.append(dict(pair=p, cohort="-", entry="no rows (no spike in the replay window)")); continue
        Rw = pd.read_pickle(f); Rw = Rw[Rw.entry_ms >= pd.Timestamp("2026-11-10").value // 10**6].reset_index(drop=True)
        Rw["slip"] = np.maximum(0.10, L.slip_model(Rw.q5.values.astype(float)))
        M = L.rule_masks(Rw); masks = {"NS0": L.ns0_mask(Rw), "R1": M["R1_OI_UNWIND"][0], "R4": M["R4_TOP_DISTRIB"][0]}
        z = np.load(os.path.join(D, "live1m", p + ".npz")); one = (z["t"], z["o"], z["h"], z["l"], z["c"])
        for c, m in masks.items():
            if not m.any():
                rp.append(dict(pair=p, cohort=c, entry="no signal")); continue
            rr = pd.DataFrame(dict(xi=np.nonzero(m)[0], entry_ms=Rw.entry_ms.values[m], atrp=Rw.atrp.values[m], slip=Rw.slip.values[m]))
            o, _ = S.run_rows(p, rr, cells_x, one1m=one)
            xm_ref = o["xm"][:, 0]; busy = -1
            for i in range(len(rr)):
                if rr.entry_ms.values[i] < busy or not o["ok"][i]:
                    continue
                busy = xm_ref[i]
                s_ = rr.slip.values[i]
                rec = dict(pair=p, cohort=c, entry=str(pd.Timestamp(int(rr.entry_ms.values[i]), unit="ms"))[:16],
                           open=bool(xm_ref[i] > z["t"][-1]))
                for q, cdef in enumerate(cells_x):
                    k_ = o["k"][i, q]; tk_ = k_ != 0
                    rec[cdef["name"]] = f'{["TP", "SL", "TIME", "DYN"][k_]} {o["r"][i, q] + o["f"][i, q] - s_ - 0.05 - (s_ + 0.05 if tk_ else 0.02):+.2f}'
                rp.append(rec)
    RP = pd.DataFrame(rp); RP.to_csv(OUT + "_replays.csv", index=False)
    # 3. trade rows (cohort trades at REF + every shown cell)
    tr = []
    for c, n in [(c, "REF_XE") for c in EV.COH] + shown + [(c, n2) for c in EV.COH for n2 in ("S_TP1_SL3_24h", "S_TP1.5_SL3_24h", "S_TP1_SL3_4h", "S_TP1.5_SL3_4h")]:
        j = NAMES.index(n); idx = SEL[(c, j)]
        g = EV.path_group(Z, NET, idx, j)
        tr.append(pd.DataFrame(dict(cohort=c, cell=n, pair=Y.pair.values[idx], entry_ms=T[idx], exit_ms=Z["xm"][idx, j], blk=blkv[idx],
                                    half=Y.half.values[idx], kind=np.array(["TP", "SL", "TIME", "DYN"])[Z["k"][idx, j]],
                                    gross=Z["r"][idx, j].round(3), funding=Z["f"][idx, j].round(4), slip=Y.slip.values[idx].round(3),
                                    net=net[idx, j].round(3), net20=NET[1][idx, j].round(3), src1m=Z["src1m"][idx], path_group=g,
                                    reach=Z["reach"][idx, j].round(2), btc_R1=Y.R1reg.values[idx], btc_R4=Y.R4reg.values[idx])))
    TR = pd.concat(tr).drop_duplicates(["cohort", "cell", "pair", "entry_ms"])
    TR.to_csv(OUT + "_trades.csv.gz", index=False, compression="gzip")
    # 4. chart: anchored mean / median short path (5-min, 48 h) per cohort + RND sample, exit levels; equity curves
    import glob
    P = {}
    for f in glob.glob(os.path.join(D, "path", "*.npz")):
        z = np.load(f); P.update({int(a): b for a, b in zip(z["xi"], z["p"])})
    xi_of = Z["xi"]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    hrs = np.arange(577) / 12
    for c, col in (("NS0", "#555"), ("R1", "#1f5fa8"), ("R4", "#8a4fb0")):
        idx = SEL[(c, REF)]
        A = np.array([P[int(x)] for x in xi_of[idx] if int(x) in P], float); A[~np.isfinite(A)] = np.nan
        ax[0].plot(hrs, np.nanmean(A, 0), color=col, lw=1.8, label=f"{c} mean (N {len(A)})")
        ax[0].plot(hrs, np.nanmedian(A, 0), color=col, lw=1.2, ls="--", label=f"{c} median")
        q = np.nanpercentile(A, [25, 75], 0)
        if c == "NS0":
            ax[0].fill_between(hrs, q[0], q[1], color=col, alpha=0.08, label="NS0 25–75 %")
    for lv, lab in ((1, "TP 1 %"), (3, "arm / TP 3 %"), (10, "TP 10 %"), (-3, "SL 3 %"), (-6, "SL 6 %"), (-10, "SL 10 %")):
        ax[0].axhline(lv, color="#c33" if lv < 0 else "#2a7", lw=0.7, ls=":")
        ax[0].text(47.5, lv, lab, fontsize=7, ha="right", va="bottom")
    ax[0].set_xlim(0, 48); ax[0].set_ylim(-15, 15); ax[0].set_xlabel("hours after entry"); ax[0].set_ylabel("short P&L % (close-to-entry)")
    ax[0].set_title("Anchored short paths, cohort entries (BACKTEST, 1m→5-min)"); ax[0].legend(fontsize=7, loc="lower left")
    # winners vs losers of the best shown cell on its cohort
    c0, n0 = shown[0]; j0 = NAMES.index(n0); idx = SEL[(c0, j0)]
    v = net[idx, j0]
    for m, col, lab in ((v > 0, "#2a7", "winners"), (v <= 0, "#c33", "losers")):
        A = np.array([P[int(x)] for x in xi_of[idx[m]] if int(x) in P], float)
        if len(A):
            ax[1].plot(hrs, np.nanmean(A, 0), color=col, lw=1.8, label=f"{lab} mean (N {len(A)})")
            ax[1].plot(hrs, np.nanmedian(A, 0), color=col, lw=1, ls="--")
    cd = CL[j0]
    for lv in [x for x in (cd.get("tp"), -cd.get("sl", np.nan), cd.get("arm")) if x is not None and np.isfinite(x)]:
        ax[1].axhline(lv, color="#888", lw=0.7, ls=":")
    ax[1].set_xlim(0, 48); ax[1].set_ylim(-15, 15); ax[1].set_title(f"{c0} · {n0}: winners vs losers (dotted = exit levels)")
    ax[1].legend(fontsize=7); ax[1].set_xlabel("hours after entry")
    # equity curves (one size 0.61, full span, exit day)
    d0, d1 = L.cal_days("ALL")[1:]
    for (c, n), col in zip([(c, "REF_XE") for c in EV.COH] + shown[:3] + [("NS0", "S_TP1_SL3_24h"), ("R1", "S_TP1_SL3_24h")],
                           ["#999", "#6a8fc2", "#b49ad0", "#000", "#1f5fa8", "#8a4fb0", "#d08a2a", "#c33"]):
        j = NAMES.index(n); idx = SEL[(c, j)]
        r = pd.Series(net[idx, j] * L.SIZE / 100).groupby(Z["xm"][idx, j] // DAY).sum().reindex(np.arange(d0, d1), fill_value=0)
        eq = np.cumprod(1 + r.values)
        ax[2].plot(pd.to_datetime(np.arange(d0, d1) * DAY, unit="ms"), np.maximum(eq, 1e-6), color=col, lw=1.2, label=f"{c} · {n}")
    ax[2].axvline(pd.Timestamp("2025-07-01"), color="k", lw=0.6, ls="--"); ax[2].set_yscale("log"); ax[2].tick_params(axis="x", labelrotation=30, labelsize=7)
    ax[2].set_title("One-size equity (0.61 × equity per fill, log) — DESIGN | HOLDOUT"); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT + "_chart.png", dpi=110)
    pd.set_option("display.width", 250)
    print(SM[SM.blk != "ALL"].round(2).to_string())
    print(HT.round(1).to_string()); print(RB.round(3).to_string()); print(XC.round(3).to_string()); print(RP.to_string())


if __name__ == "__main__":
    main()
