#!/usr/bin/env python3
"""DIST_PEAKSTOP — EVAL (PREREG reports/DIST_PEAKSTOP_2026-11-21_PREREG.md). Research only, BACKTEST.
Parity (REF vs sim_BT REF, P15 vs C_PK1) · distance-to-top distribution · every cell × DESIGN / HOLDOUT / ALL (N, WR, avg, $, avg win / loss,
worst, DCR, max DD, worst day, duration, TP / STOP / TIME) · DESIGN pick among P ∪ Q → HOLDOUT gates ①–⑧ · live-stopped split · Kelly / risk.
→ reports/DIST_PEAKSTOP_2026-11-21_data/{eval.csv, stopsplit.csv, gates.csv, dist_to_top.csv, perfill.csv.gz, kelly.csv}"""
import os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(n, f):
    sp = importlib.util.spec_from_file_location(n, os.path.join(R, "scripts", f)); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


EV = _load("DTE_EVAL", "DIST_THEORY_EXIT_2026-11-21_eval.py")
D0 = os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21_data")
D = os.path.join(R, "reports", "DIST_PEAKSTOP_2026-11-21_data")
DAY, H1 = 86_400_000, 3_600_000
USD, X_EQ = 3050.0, 0.61
CAPS = (8, 10, 12, 15)


def wilson_lo(p, n, z=1.645):
    if n == 0:
        return np.nan
    c = p + z * z / (2 * n); a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / (1 + z * z / n)


def kelly(p, w, l):
    b = w / abs(l)
    return p - (1 - p) / b


def main():
    S = pd.read_pickle(os.path.join(R, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl"))
    Y = dict(np.load(os.path.join(D, "sim.npz"), allow_pickle=False))
    Z = dict(np.load(os.path.join(D0, "sim_BT.npz"), allow_pickle=False))
    names = [str(x) for x in Y["names"]]; ci = {n: j for j, n in enumerate(names)}
    zn = [str(x) for x in Z["names"]]
    # ---- parity
    for mine, theirs in (("REF", "REF"), ("P15", "C_PK1")):
        a, b = ci[mine], zn.index(theirs)
        m = (np.abs(Y["r"][:, a] - Z["r"][:, b]) <= 0.01) & (Y["k"][:, a] == Z["k"][:, b])
        print(f"PARITY {mine} vs {theirs}: {m.sum()}/{len(m)} = {m.mean() * 100:.2f} %")
        if m.mean() < 0.999:
            print("PARITY FAIL — stop"); sys.exit(1)
    slip = S.slip.values.astype(float)
    r = Y["r"].astype(float); k = Y["k"]; f = Y["f"].astype(float)
    Nn = EV.nets(slip, r, k, f); N2 = EV.nets(slip, r, k, f, flat=0.20)
    T = S.entry_ms.values.astype(np.int64); dayn = T // DAY
    per = np.where(T < EV.SPAN["DESIGN"][1], "DESIGN", "HOLDOUT"); hv = EV.half_of(T); pair = S.pair.astype(str).values
    ok = np.isfinite(Nn).all(1)
    print("finite in every cell:", int(ok.sum()), "of", len(S))
    dur = (Y["xm"] - T[:, None]) / H1
    stp = Y["stops"]; s0 = np.minimum(1.5 * S.atr15_pct.values.astype(float), 6.0)
    # ---- distance to top (entry-time, no outcomes)
    top = (Y["PK"] / Y["E0_bt"] - 1) * 100; dh = (Y["DH"] / Y["E0_bt"] - 1) * 100
    rows = []
    for p in ("DESIGN", "HOLDOUT", "ALL"):
        m = (per == p) if p != "ALL" else np.ones(len(T), bool)
        for nm, v in (("pump top above entry %", top[m]), ("distribution-candle high above entry %", dh[m]), ("today's stop %", s0[m])):
            v = v[np.isfinite(v)]
            rows.append(dict(block=p, what=nm, N=len(v), p10=np.percentile(v, 10), p25=np.percentile(v, 25), median=np.median(v),
                             p75=np.percentile(v, 75), p90=np.percentile(v, 90), mean=v.mean(), share_lt1=(v < 1).mean() * 100,
                             share_gt_stop=np.nan))
        t_ = top[m]; okp = np.isfinite(t_)
        rows.append(dict(block=p, what="share top+1% above today's stop", N=int(okp.sum()), share_gt_stop=((t_[okp] + 1.01) > s0[m][okp]).mean() * 100))
        for c in CAPS:
            rows.append(dict(block=p, what=f"share top+1% above cap {c}", N=int(okp.sum()), share_gt_stop=(((1 + t_[okp] / 100) * 1.01 - 1) * 100 > c).mean() * 100))
    DT = pd.DataFrame(rows); DT.to_csv(os.path.join(D, "dist_to_top.csv"), index=False)
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 400)
    print(DT.round(2).to_string(index=False))
    # ---- cells
    E = []; SP = []
    for j, nm in enumerate(names):
        for p in ("DESIGN", "HOLDOUT", "ALL"):
            m = ok & ((per == p) if p != "ALL" else True)
            sp = EV.SPAN[p] if p != "ALL" else (EV.SPAN["DESIGN"][0], EV.SPAN["HOLDOUT"][1])
            v = Nn[m, j]; v0 = Nn[m, 0]; d = v - v0; d2 = N2[m, j] - N2[m, 0]
            lo, hi = EV.boot(d, dayn[m], sp) if j else (0.0, 0.0)
            hp = pd.Series(d).groupby(hv[m]).sum()
            ps = pd.Series(d).groupby(pair[m]).sum(); ds = pd.Series(d).groupby(dayn[m]).sum(); tot = d.sum()
            dc, dd, wd = EV.risk(v, Y["xm"][m, j], sp); dc0, dd0, wd0 = EV.risk(v0, Y["xm"][m, 0], sp)
            kk = k[m, j]; du = dur[m, j]
            # live-stopped split
            st0 = k[m, 0] == 1
            sv = st0 & (d > 1e-9); dp = st0 & (d < -1e-9); ns = ~st0 & (np.abs(d) > 1e-9)
            stops_day = pd.Series((kk == 1).astype(int)).groupby(dayn[m]).sum()
            E.append(dict(cell=nm, period=p, N=int(m.sum()), changed=int((np.abs(d) > 1e-9).sum()), WR=(v > 0).mean() * 100, avg=v.mean(),
                          usd=v.sum() * USD / 100, avg_win=v[v > 0].mean(), avg_loss=v[v <= 0].mean(), worst=v.min(),
                          dcr=dc, maxdd=dd, worst_day=wd, dcr_ref=dc0, maxdd_ref=dd0, worst_day_ref=wd0,
                          dur_mean=du.mean(), dur_med=np.median(du), n_tp=int((kk == 0).sum()), n_sl=int((kk == 1).sum()), n_time=int((kk == 2).sum()),
                          d_mean=d.mean(), ci_lo=lo, ci_hi=hi, halves_pos=f"{int((hp > 0).sum())}/{len(hp)}", halves_frac=(hp > 0).mean(),
                          d020=d2.mean(), top_pair_share=(ps.max() / tot * 100) if tot > 0 else np.nan,
                          top_day_share=(ds.max() / tot * 100) if tot > 0 else np.nan,
                          ref_stopped=int(st0.sum()), saved_n=int(sv.sum()), saved_usd=d[sv].sum() * USD / 100, deeper_n=int(dp.sum()),
                          deeper_usd=d[dp].sum() * USD / 100, nonstopped_changed=int(ns.sum()), nonstopped_usd=d[ns].sum() * USD / 100,
                          max_stops_day=int(stops_day.max()) if len(stops_day) else 0,
                          med_stop=float(np.median(stp[m, j])) if j else float(np.median(s0[m])),
                          avg_stop_on_sl=float(np.mean(-v[kk == 1])) if (kk == 1).any() else np.nan))
    E = pd.DataFrame(E)
    # shuffled-control means
    for c in CAPS:
        for p in ("DESIGN", "HOLDOUT", "ALL"):
            sub = E[(E.period == p) & E.cell.str.match(fr"S{c}_\d")]
            row = sub.mean(numeric_only=True).to_dict(); row.update(cell=f"S{c}_mean", period=p, halves_pos="")
            E = pd.concat([E, pd.DataFrame([row])], ignore_index=True)
    E.to_csv(os.path.join(D, "eval.csv"), index=False)
    cols = ["cell", "N", "changed", "WR", "avg", "usd", "avg_win", "avg_loss", "worst", "d_mean", "ci_lo", "ci_hi", "halves_pos", "d020",
            "dcr_ref", "dcr", "maxdd", "worst_day", "dur_mean", "dur_med", "n_tp", "n_sl", "n_time", "med_stop"]
    for p in ("DESIGN", "HOLDOUT", "ALL"):
        print(f"\n===== {p}"); print(E[(E.period == p) & ~E.cell.str.match(r"S\d+_\d")][cols].round(2).to_string(index=False))
    print("\n===== live-stopped split")
    sc = ["cell", "period", "ref_stopped", "saved_n", "saved_usd", "deeper_n", "deeper_usd", "nonstopped_changed", "nonstopped_usd"]
    print(E[~E.cell.str.match(r"S\d+_\d")][sc].round(0).to_string(index=False))
    # assert Q / W / S never touch non-stopped fills
    for nm in names:
        if nm[0] in "QWS" or nm.startswith("DQ"):
            assert (E[(E.cell == nm)].nonstopped_changed == 0).all(), nm
    # ---- pick + gates
    pk = E[(E.period == "DESIGN") & E.cell.str.match(r"^[PQ]\d+$")].sort_values("d_mean", ascending=False)
    print("\nDESIGN ranking (P ∪ Q):\n", pk[["cell", "d_mean", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    G = []
    for cand in list(pk.cell) + ["W8", "W10", "W12", "W15"]:
        h = E[(E.period == "HOLDOUT") & (E.cell == cand)].iloc[0]
        cap = int(cand[1:])
        w = E[(E.period == "HOLDOUT") & (E.cell == f"W{cap}")].iloc[0]; s = E[(E.period == "HOLDOUT") & (E.cell == f"S{cap}_mean")].iloc[0]
        g = [h.d_mean > 0 and h.ci_lo > 0, h.halves_frac >= 2 / 3, h.d020 > 0,
             (h.top_pair_share < 50 and h.top_day_share < 50) if h.d_mean > 0 else False,
             h.saved_usd > -(h.deeper_usd + min(h.nonstopped_usd, 0)), h.dcr >= h.dcr_ref,
             (h.d_mean > w.d_mean and h.d_mean > s.d_mean) if cand[0] != "W" else True, h.maxdd >= h.maxdd_ref - 5]
        G.append(dict(cell=cand, is_pick=cand == pk.cell.iloc[0], design_d=E[(E.period == "DESIGN") & (E.cell == cand)].d_mean.iloc[0],
                      hold_d=h.d_mean, ci_lo=h.ci_lo, ci_hi=h.ci_hi, halves=h.halves_pos, d020=h.d020, pair_share=h.top_pair_share,
                      day_share=h.top_day_share, saved=h.saved_usd, deeper=h.deeper_usd, nonstopped=h.nonstopped_usd, dcr_ref=h.dcr_ref, dcr=h.dcr,
                      maxdd_ref=h.maxdd_ref, maxdd=h.maxdd, W_same_cap=w.d_mean, S_same_cap=s.d_mean,
                      gates="".join("✓" if x else "✗" for x in g), verdict="PASS" if all(g) else "FAIL"))
    G = pd.DataFrame(G); G.to_csv(os.path.join(D, "gates.csv"), index=False)
    print("\n===== HOLDOUT gates (pick first; the rest descriptive)\n", G.round(3).to_string(index=False))
    # ---- Kelly / risk
    K = []
    for nm in ("REF", "P10", "P12", "Q8", "Q10", "Q12", "Q15", "W12"):
        for p in ("DESIGN", "HOLDOUT", "ALL"):
            m = ok & ((per == p) if p != "ALL" else True)
            v = Nn[m, ci[nm]]; n = len(v); wr = (v > 0).mean(); w_ = v[v > 0].mean(); l_ = v[v <= 0].mean()
            lo = wilson_lo(wr, n)
            K.append(dict(cell=nm, period=p, N=n, WR=wr * 100, WR_floor95=lo * 100, be_WR=abs(l_) / (w_ + abs(l_)) * 100, avg_win=w_, avg_loss=l_,
                          kelly_obs=kelly(wr, w_, l_), kelly_floor=kelly(lo, w_, l_),
                          per_stop_eq_pct=np.mean(-v[k[m, ci[nm]] == 1]) * X_EQ if (k[m, ci[nm]] == 1).any() else np.nan,
                          worst_trade_eq_pct=v.min() * X_EQ))
    K = pd.DataFrame(K); K.to_csv(os.path.join(D, "kelly.csv"), index=False)
    print("\n===== Kelly\n", K.round(3).to_string(index=False))
    PF = pd.DataFrame(dict(pair=pair, entry=pd.to_datetime(T, unit="ms"), period=per, half=hv, top_pct=top, dh_pct=dh, s0=s0))
    for j, nm in enumerate(names):
        if nm.startswith("S"):
            continue
        PF[nm] = np.round(Nn[:, j], 3); PF[nm + "_k"] = k[:, j]; PF[nm + "_h"] = np.round(dur[:, j], 2); PF[nm + "_stop"] = np.round(stp[:, j], 2)
    PF.to_csv(os.path.join(D, "perfill.csv.gz"), index=False)


if __name__ == "__main__":
    main()
