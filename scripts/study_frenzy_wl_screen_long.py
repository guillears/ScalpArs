#!/usr/bin/env python3
"""FRENZY W/L screen 2026-10-19 — FRENZY_LONG special focus (operator priority): sleeve expectancy / size, Pattern-W cell search,
blocker role. Read-only; writes reports/FRENZY_WL_SCREEN_2026-10-19_long_*.csv. argv[1] = real-fill pool pickle, argv[2] = journal pickle.

Blocks: ① live B23 export (booked) · ② master per batch (STACK 2026-10-19a stack_pct = today's +1.5/−3) · ③ backtest yr6 seeds 1–3
(pct_new at +1.5/−3, days < 2026-10-03 for the cell work so the real era is out of sample; sleeve table shows through 10-11 too).
Pattern-W bar (CLAUDE.md): N ≥ 30 ∧ WR ≥ 70 % ∧ avg ≥ +0.10 % ∧ Total > 0, cross-batch, then backtest confirmation at the same frozen cut;
BE-compatibility: ≥ 60 % of the cell's losses peaked ≥ +0.20 %.
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_yr6wl_frenzy_common as C
import study_frenzy_wl_screen as W

POOL, JPK = sys.argv[1], sys.argv[2]
OUTP = os.path.join(ROOT, "reports", "FRENZY_WL_SCREEN_2026-10-19_long")
BLOCKER_NOTIONAL = None   # set from the real DL-289+ fills below


def summ(y, day, label, notional=None, peak=None):
    y = np.asarray(y, float); day = np.asarray(day)
    lo, hi, p = C.day_boot(y, day)
    ds = pd.Series(y).groupby(day).sum()
    w, l = y[y > 0], y[y <= 0]
    streak = mx = 0
    for v in y:
        streak = streak + 1 if v <= 0 else 0; mx = max(mx, streak)
    r = dict(block=label, N=len(y), days=len(set(day)), WR=round(100 * (y > 0).mean(), 1), BE_WR=round(100 * C.breakeven_wr(y), 1) if len(w) and len(l) else np.nan,
             avg=round(y.mean(), 3), day_ci=f"[{lo:+.3f}, {hi:+.3f}]", P_avg_pos=round(p, 3), sum_pts=round(y.sum(), 2),
             worst_fill=round(y.min(), 2), worst_day_pts=round(ds.min(), 2), max_loss_streak=mx)
    if notional is not None:
        r["usd_per_fill_at_blocker"] = round(y.mean() / 100 * notional[0], 2); r["worst_day_usd_blocker"] = round(ds.min() / 100 * notional[0], 1)
        r["usd_per_fill_at_2x"] = round(y.mean() / 100 * notional[1], 2); r["worst_day_usd_2x"] = round(ds.min() / 100 * notional[1], 1)
    if peak is not None:
        pk = np.asarray(peak, float); lm = y <= 0
        r["losses_peak_ge_0.20"] = round((pk[lm] >= 0.20).mean(), 2) if lm.any() else np.nan
    return r


def main():
    D = pd.read_pickle(POOL)
    L = D[D.sleeve == "FRENZY_LONG"].sort_values("ts").reset_index(drop=True)
    blk = L[L.ts >= "2026-10-14 15:00"]           # DL 289 blocker size live from 2026-10-14
    bn = float(pd.to_numeric(blk.notional_value, errors="coerce").median())
    old2x = 1413.75                                 # yr6 fixed-book LONG notional at 2× (invest 1.0 · lev 0.1) — the pre-289 size
    NOT = (bn, old2x)
    rows = []
    # ① live / ② master per batch / all real
    rows.append(summ(L[L.src == "live"].y, L[L.src == "live"].day, "① live B23 (booked)", NOT, L[L.src == "live"].peak_pnl))
    for era, g in L[L.src == "master"].groupby("era"):
        rows.append(summ(g.y, g.day, f"② master {era}", NOT, g.peak_pnl))
    m = L[L.src == "master"]
    rows.append(summ(m.y, m.day, "② master ALL (today's +1.5/−3)", NOT, m.peak_pnl))
    rows.append(summ(m.pnl_percentage, m.day, "② master ALL as traded (mixed exits, reference only)", NOT, m.peak_pnl))
    rows.append(summ(L.y, L.day, "①+② real ALL", NOT, L.peak_pnl))
    # ③ backtest
    BA = C.load(all_seeds=True); BA = BA[BA.sleeve == "FRENZY_LONG"]
    for seed in (1, 2, 3):
        S = BA[BA.seed == seed]
        for reg in C.REGIMES:
            g = S[S.regime == reg]
            if seed == 1 or reg == C.REGIMES[0]:
                pass
            rows.append(summ(g.y, g.day, f"③ yr6 s{seed} {reg}", NOT, g.peak_pnl))
        rows.append(summ(S.y, S.day, f"③ yr6 s{seed} ALL (2025-01 → 2026-10-11)", NOT, S.peak_pnl))
        t = S[S.top50]
        rows.append(summ(t.y, t.day, f"③ yr6 s{seed} ALL top-50 pair-days", NOT, t.peak_pnl))
    T1 = pd.DataFrame(rows)
    T1.to_csv(OUTP + "_sleeve.csv", index=False)
    print(T1.to_string())

    # concurrency (yr6 seed 1, all FRENZY sleeves, fixed $2,900 book usd_new)
    B1 = C.load(); B1["tc"] = pd.to_datetime(B1.closed_at, format="mixed")
    LB = B1[B1.sleeve == "FRENZY_LONG"]; OT = B1[B1.sleeve.isin(["FRENZY_LITE", "FRENZY_WILLY"])]
    ov = []
    for _, r in LB.iterrows():
        o = OT[(OT.ts < r.tc) & (OT.tc > r.ts) & (OT.pair != r.pair)]
        ov.append(len(o))
    LB = LB.assign(n_overlap=ov)
    conc = dict(long_fills=len(LB), share_with_lite_or_willy_open=round((LB.n_overlap > 0).mean(), 3))
    fz = B1[B1.sleeve.isin(["FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"])].copy()
    fz["usd_blocker"] = np.where(fz.sleeve == "FRENZY_LONG", fz.usd_new * bn / old2x, fz.usd_new)
    dd = fz.groupby("day").agg(usd_2x=("usd_new", "sum"), usd_blocker=("usd_blocker", "sum"))
    lday = fz[fz.sleeve == "FRENZY_LONG"].groupby("day").usd_new.sum()
    conc.update(worst_frenzy_day_usd_long2x=round(dd.usd_2x.min(), 0), worst_frenzy_day_usd_long_blocker=round(dd.usd_blocker.min(), 0),
                worst_long_day_usd_2x=round(lday.min(), 0), worst_long_day_usd_blocker=round(lday.min() * bn / old2x, 1),
                long_2y_usd_2x=round(LB.usd_new.sum(), 0), long_2y_usd_blocker=round(LB.usd_new.sum() * bn / old2x, 0),
                blocker_notional=round(bn, 1), old_2x_notional=old2x)
    pd.DataFrame([conc]).to_csv(OUTP + "_concurrency.csv", index=False)
    print(conc)

    # ── Pattern-W cell search ───────────────────────────────────────────────────────────────────────────────────────────
    B = W.load_bt(); B = B[B.sleeve == "FRENZY_LONG"].reset_index(drop=True)
    dis = B[B.regime == "2025 BULL"].reset_index(drop=True)
    cand = W.candidates(dis)
    cells = []
    for c, x in cand.items():
        if x.notna().mean() < 0.6:
            continue
        for lab, cut, z, sc, rule in W.zones_1d(x):
            # both sides of each cut are cells (a winner cell can be either side)
            for side, zz in (("zone", z), ("complement", sc & ~z)):
                cells.append((c, lab, cut, side, rule, zz))
    y = dis.y.values; day = dis.day.values

    def passes(yz):
        return len(yz) >= 30 and (yz > 0).mean() >= 0.70 and yz.mean() >= 0.10 and yz.sum() > 0

    Zm = np.array([cl[5] for cl in cells]).T.astype(float)
    def n_pass(yv):
        n = Zm.sum(0); s = Zm.T @ yv; w = Zm.T @ (yv > 0).astype(float)
        return int(((n >= 30) & (w / np.maximum(n, 1) >= 0.70) & (s / np.maximum(n, 1) >= 0.10) & (s > 0)).sum())
    obs = n_pass(y)
    Yp = C.block_draws(y, day, 300, 5)
    nul = np.array([n_pass(Yp[:, k]) for k in range(Yp.shape[1])])
    out = []
    for c, lab, cut, side, rule, zz in cells:
        yz = y[zz]
        if not passes(yz):
            continue
        r = dict(var=c, gran=lab, cut=cut, side=side, disc_N=len(yz), disc_days=len(set(day[zz])), disc_WR=round(100 * (yz > 0).mean(), 1), disc_avg=round(yz.mean(), 3),
                 disc_BEcompat=round((dis.peak_pnl.values[zz][yz <= 0] >= 0.2).mean(), 2) if (yz <= 0).any() else np.nan)
        for reg in ["2026-H1 BEAR", "2026-H2 BULL"]:
            S = B[B.regime == reg]
            z2, s2 = W.apply_rule(W.numcol(S[c]), rule)
            if side == "complement":
                z2 = s2 & ~z2
            yy = S.y.values[z2]
            r[f"{reg[:7]}_N"] = len(yy); r[f"{reg[:7]}_WR"] = round(100 * (yy > 0).mean(), 1) if len(yy) else np.nan
            r[f"{reg[:7]}_avg"] = round(yy.mean(), 3) if len(yy) else np.nan; r[f"{reg[:7]}_pass"] = passes(yy)
        if c in L.columns:
            zr, sr = W.apply_rule(W.numcol(L[c]), rule)
            if side == "complement":
                zr = sr & ~zr
            yy = L.y.values[zr]
            r.update(real_N=len(yy), real_WR=round(100 * (yy > 0).mean(), 1) if len(yy) else np.nan, real_avg=round(yy.mean(), 3) if len(yy) else np.nan,
                     real_cov=round(sr.mean(), 2))
        out.append(r)
    PW = pd.DataFrame(out)
    PW.to_csv(OUTP + "_patternW_cells.csv", index=False)
    print(f"Pattern-W cells passing in 2025 discovery: {obs} of {len(cells)} · null mean {nul.mean():.1f} · 95th {np.percentile(nul, 95):.0f} · p {(nul >= obs).mean():.3f}")
    if len(PW):
        print(PW.sort_values("disc_avg", ascending=False).to_string())
    pd.DataFrame([dict(cells_tested=len(cells), pass_obs=obs, null_mean=round(nul.mean(), 1), null_95=int(np.percentile(nul, 95)), p=round((nul >= obs).mean(), 3),
                       confirmed_both_2026=int(((PW.get("2026-H1_pass", pd.Series(dtype=bool)) == True) & (PW.get("2026-H2_pass", pd.Series(dtype=bool)) == True)).sum()) if len(PW) else 0)]
                 ).to_csv(OUTP + "_patternW_null.csv", index=False)

    # real-fill side: best cells at N ≥ 10 (Pattern-W N bar cannot be met: sleeve N = 29), then the backtest at the frozen cut
    candR = W.candidates(L); yR = L.y.values; rr = []
    for c, x in candR.items():
        if x.notna().sum() < 12:
            continue
        for lab, cut, z, sc, rule in W.zones_1d(x):
            for side, zz in (("zone", z), ("complement", sc & ~z)):
                yz = yR[zz]
                if len(yz) >= 10 and (yz > 0).mean() >= 0.70 and yz.mean() >= 0.10:
                    r = dict(var=c, gran=lab, cut=cut, side=side, real_N=len(yz), real_days=len(set(L.day.values[zz])), real_WR=round(100 * (yz > 0).mean(), 1),
                             real_avg=round(yz.mean(), 3), real_sum=round(yz.sum(), 2))
                    if c in B.columns:
                        for reg in C.REGIMES + ["ALL"]:
                            S = B if reg == "ALL" else B[B.regime == reg]
                            z2, s2 = W.apply_rule(W.numcol(S[c]), rule)
                            if side == "complement":
                                z2 = s2 & ~z2
                            yy = S.y.values[z2]
                            r[f"BT_{reg[:7]}"] = f"{len(yy)}·{100 * (yy > 0).mean():.0f}%·{yy.mean():+.3f}" if len(yy) else "0"
                            if reg == "ALL":
                                r["BT_ALL_avg"] = round(yy.mean(), 3) if len(yy) else np.nan; r["BT_ALL_WR"] = round(100 * (yy > 0).mean(), 1) if len(yy) else np.nan
                                r["BT_ALL_N"] = len(yy)
                                r["BT_ALL_patternW"] = passes(yy)
                    else:
                        r["BT_ALL"] = "not in backtest"
                    rr.append(r)
    RR = pd.DataFrame(rr)
    RR.to_csv(OUTP + "_real_cells.csv", index=False)
    print(len(RR), "real cells N≥10 WR≥70 avg≥0.10;", int(RR.get("BT_ALL_patternW", pd.Series(dtype=bool)).fillna(False).sum()) if len(RR) else 0, "also pass Pattern-W in backtest")
    if len(RR):
        print(RR.sort_values("real_avg", ascending=False).head(40).to_string())

    # blocker role in the journals
    J = pd.read_pickle(JPK)
    bl = J[(J.e == "BLOCK") & J.gate.astype(str).str.contains("PAIR_HELD")]
    print(bl[["t", "pair", "gate", "mode"]].to_string())
    bl[["t", "pair", "gate", "mode"]].to_csv(OUTP + "_journal_pairheld.csv", index=False)
    print("journal span", J.t.min(), J.t.max())

    # which LONG/WIDE holders did the blocking in the yr6 LONG/WIDE-off A/B (reports/YR6_LWOFF_AB_lite_bonly.csv)? — a LONG entry
    # filter on high ADX would remove exactly these holders → price the LITE it would release
    bo = pd.read_csv(os.path.join(ROOT, "reports", "YR6_LWOFF_AB_lite_bonly.csv"), low_memory=False)
    bo["ts"] = pd.to_datetime(bo.opened_at, format="mixed")
    A = C.load(); Hh = A[A.sleeve.isin(["FRENZY_LONG", "FRENZY_WIDE"])].copy(); Hh["tc"] = pd.to_datetime(Hh.closed_at, format="mixed")
    rows = []
    for _, r in bo.iterrows():
        h = Hh[(Hh.pair == r.pair) & (Hh.ts <= r.ts) & (Hh.tc > r.ts)]
        rows.append(dict(pair=r.pair, lite_opened=str(r.ts)[:16], holder=h.sleeve.iloc[0] if len(h) else None,
                         holder_adx=float(h.entry_adx.iloc[0]) if len(h) else np.nan, holder_pct=float(h.y.iloc[0]) if len(h) else np.nan,
                         lite_pct=float(r.pct_new)))
    HB = pd.DataFrame(rows); HB.to_csv(OUTP + "_blocked_lite_by_holder_adx.csv", index=False)
    HB = HB[HB.holder.notna()]
    for nm, mm in [("holder ADX > 29.45", HB.holder_adx > 29.45), ("holder ADX <= 29.45", HB.holder_adx <= 29.45)]:
        g = HB[mm]; print(nm, len(g), "LITE WR", round(100 * (g.lite_pct > 0).mean()), "avg", round(g.lite_pct.mean(), 3), "sum", round(g.lite_pct.sum(), 2))


if __name__ == "__main__":
    main()
