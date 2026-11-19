#!/usr/bin/env python3
"""ROUTE A FOLLOW-UP (PREREG reports/ROUTE_A_FOLLOWUP_2026-11-19_PREREG.md) — Q1(b) selection lift · Q2 VWAP-hold selector ·
Q3 small-TP scalp grid with the random-entry family-wise null. Reads reports/ROUTE_A_TICKS_data/eval.pkl (Route A scores, weights,
tick-priced Q3 / TR / T1) and reports/ROUTE_A_FOLLOWUP_data/ticks.pkl (route_a_followup_ticks.py).
Writes reports/ROUTE_A_FOLLOWUP_2026-11-19_{q1b_lift,q2_selector,q2_dose,q2_auc,q3_grid,q3_null}.csv + data/eval_q23.pkl."""
import os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
from route_a_ticks_eval import net   # noqa: E402  (Route A cost model for Q3 / TR / T1)
import route_a_followup_common as C   # noqa: E402
SRC = os.path.join(R, "reports", "ROUTE_A_TICKS_data"); D = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_data")
OUT = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_2026-11-19")
CHK = (0, 60, 120, 300); EXN = ("Q3", "TR", "T1"); HALVES = ["24-H2", "25-H1", "25-H2", "26-H1"]
NB = 1000


def load():
    res = pickle.load(open(os.path.join(SRC, "eval.pkl"), "rb"))
    E = res["E"].copy()                 # events, already-flagged excluded, w / day / half / R1 / R5
    ts = res["trade_sets"]
    key = lambda d: set(zip(d.pair, d.sig_ms))
    E["score"] = np.nan; E["top20"] = 0; E["oof"] = 0
    for cs in CHK:
        g, s = res["scores"][cs]
        sc = dict(zip(zip(g.pair, g.sig_ms), s["BOTH"]))
        h = ts[(cs, "HOLDOUT", "all")]; sc.update(dict(zip(zip(h.pair, h.sig_ms), h.score)))
        top = key(ts[(cs, "DESIGN-OOF", "top20")]) | key(ts[(cs, "HOLDOUT", "top20")])
        m = E.chk == cs
        kk = list(zip(E.loc[m, "pair"], E.loc[m, "sig_ms"]))
        E.loc[m, "score"] = [sc.get(k, np.nan) for k in kk]
        E.loc[m, "top20"] = [int(k in top) for k in kk]
    E["oof"] = (np.isfinite(E.score) & ((E.block == "HOLDOUT") | (E.half != "24-H1"))).astype(int)
    for ex in EXN:
        E[ex + "_n10"] = net(E, ex, 0.10); E[ex + "_n30"] = net(E, ex, 0.30)
        E.loc[~E[ex + "_complete"].astype(bool), [ex + "_n10", ex + "_n30"]] = np.nan
    return res, E


def span_days(res):
    return res["cal"]


# ───────────────────────────── Q1(b) ─────────────────────────────
def q1b(E):
    rows = []
    for cs in CHK:
        for blk, sub in (("HOLDOUT", E[(E.block == "HOLDOUT")]), ("DESIGN-OOF", E[(E.block == "DESIGN") & (E.oof == 1)])) + \
                tuple((h, E[(E.block == "DESIGN") & (E.oof == 1) & (E.half == h)]) for h in HALVES):
            d = sub[sub.chk == cs]
            for ex in EXN:
                x = d[ex + "_n10"].values; w = d.w.values; dy = d.day.values; t = d.top20.values == 1
                ok = np.isfinite(x)
                a_all = C.wmean(x, w); a_top = C.wmean(x[t], w[t]); a_bot = C.wmean(x[~t], w[~t])
                lo1, hi1, p1 = C.day_boot_diff(x[t & ok], w[t & ok], dy[t & ok], x[ok], w[ok], dy[ok], NB)
                lo2, hi2, p2 = C.day_boot_diff(x[t & ok], w[t & ok], dy[t & ok], x[~t & ok], w[~t & ok], dy[~t & ok], NB)
                rows.append(dict(chk=cs, period=blk, exit=ex, N=int(ok.sum()), N_top=int((t & ok).sum()), days=len(np.unique(dy)),
                                 all=a_all, top20=a_top, bottom80=a_bot, lift_vs_all=a_top - a_all, ci_all_lo=lo1, ci_all_hi=hi1,
                                 lift_vs_bottom=a_top - a_bot, ci_bot_lo=lo2, ci_bot_hi=hi2, p_bot_le0=p2,
                                 all_ci_lo=C.day_boot(x, w, dy, NB)[0], all_ci_hi=C.day_boot(x, w, dy, NB)[1]))
    Q = pd.DataFrame(rows); Q.to_csv(OUT + "_q1b_lift.csv", index=False)
    return Q


# ───────────────────────────── Q2 ─────────────────────────────
def zfit(x, w):
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99); z = np.clip(x, lo, hi)
    mu = C.wmean(z, w); sd = np.sqrt(C.wmean((z - mu) ** 2, w))
    return lambda v: (np.clip(v, lo, hi) - mu) / sd


def q2(E, T, cal):
    E = E.merge(T[["pair", "sig_ms", "chk", "path1", "path2", "path3", "path5"]], on=["pair", "sig_ms", "chk"], how="left")
    sel_rows, dose_rows, auc_rows = [], [], []
    E["S_VW"] = 0; E["S_VA"] = 0; E["va"] = np.nan; E["vw_q"] = np.nan
    for cs in CHK:
        m = E.chk == cs; dz = E[m & (E.block == "DESIGN")]
        zv = zfit(dz.vwap_gap.values, dz.w.values); za = zfit(dz.tps_accel.values, dz.w.values)
        E.loc[m, "va"] = zv(E.loc[m, "vwap_gap"].values) + za(E.loc[m, "tps_accel"].values)
        dz = E[m & (E.block == "DESIGN")]
        t_vw = C.wquantile(dz.vwap_gap.values, dz.w.values, 0.80); t_va = C.wquantile(dz.va.values, dz.w.values, 0.80)
        cuts = [C.wquantile(dz.vwap_gap.values, dz.w.values, q) for q in (0.2, 0.4, 0.6, 0.8)]
        E.loc[m, "S_VW"] = (E.loc[m, "vwap_gap"] >= t_vw).astype(int); E.loc[m, "S_VA"] = (E.loc[m, "va"] >= t_va).astype(int)
        E.loc[m, "vw_q"] = np.searchsorted(cuts, E.loc[m, "vwap_gap"].values, side="right") + 1
        for blk in ("DESIGN", "HOLDOUT"):
            d = E[m & (E.block == blk)]
            # AUCs (population-weighted) of raw vwap_gap and the S_VA score
            for lab in ("runner", "path1", "path2", "path3", "path5"):
                y = d[lab].values
                ok = np.isfinite(y.astype(float))
                auc_rows.append(dict(chk=cs, block=blk, label=lab, N=int(ok.sum()), base_rate=C.wmean(y[ok], d.w.values[ok]) * 100,
                                     auc_vwap_gap=C.wauc(d.vwap_gap.values[ok], y[ok].astype(int), d.w.values[ok]),
                                     auc_va=C.wauc(d.va.values[ok], y[ok].astype(int), d.w.values[ok]),
                                     auc_tps_accel=C.wauc(d.tps_accel.values[ok], y[ok].astype(int), d.w.values[ok])))
            span = cal[blk]
            for sname, mask in (("all", np.ones(len(d), bool)), ("S_VW", d.S_VW.values == 1), ("S_VA", d.S_VA.values == 1),
                                ("rest_VW", d.S_VW.values == 0)):
                x0 = d[mask]
                for ex in EXN:
                    for slip in ("10", "30"):
                        x = x0[f"{ex}_n{slip}"].values; w = x0.w.values
                        if not np.isfinite(x).any():
                            continue
                        lo, hi = C.day_boot(x, w, x0.day.values, NB)
                        ok = np.isfinite(x)
                        sel_rows.append(dict(chk=cs, block=blk, select=sname, exit=ex, slip=int(slip) / 100, N=int(ok.sum()),
                                             days=x0.day.nunique(), trades_day=w[ok].sum() / span, WR=C.wmean(x > 0, w) * 100,
                                             avg=C.wmean(x, w), ci_lo=lo, ci_hi=hi, best5_out=C.best5_out(x, w),
                                             loo_coin=C.loo(x[ok], w[ok], x0.pair.values[ok]) if slip == "10" else np.nan,
                                             loo_day=C.loo(x[ok], w[ok], x0.day.values[ok]) if slip == "10" else np.nan,
                                             runner_pct=C.wmean(x0.runner.values, w) * 100,
                                             **{f"path{X}_pct": C.wmean(x0[f"path{X}"].values, w) * 100 for X in (1, 2, 3, 5)}))
            for q_ in range(1, 6):
                x0 = d[d.vw_q == q_]
                o = dict(chk=cs, block=blk, quintile=q_, N=len(x0), vwap_gap_med=x0.vwap_gap.median(),
                         runner_pct=C.wmean(x0.runner.values, x0.w.values) * 100,
                         **{f"path{X}_pct": C.wmean(x0[f"path{X}"].values, x0.w.values) * 100 for X in (1, 2, 3, 5)})
                for ex in EXN:
                    o[ex + "_net"] = C.wmean(x0[ex + "_n10"].values, x0.w.values)
                dose_rows.append(o)
    S = pd.DataFrame(sel_rows); Dd = pd.DataFrame(dose_rows); A = pd.DataFrame(auc_rows)
    # Spearman of quintile vs net (DESIGN) per chk × exit
    sp = []
    for (cs, blk), g in Dd.groupby(["chk", "block"]):
        for ex in EXN:
            sp.append(dict(chk=cs, block=blk, exit=ex, spearman=pd.Series(g.quintile.values).rank().corr(pd.Series(g[ex + "_net"].values).rank())))
    Dd = Dd.merge(pd.DataFrame(sp).pivot_table(index=["chk", "block"], columns="exit", values="spearman").add_prefix("rho_").reset_index(),
                  on=["chk", "block"])
    S.to_csv(OUT + "_q2_selector.csv", index=False); Dd.to_csv(OUT + "_q2_dose.csv", index=False); A.to_csv(OUT + "_q2_auc.csv", index=False)
    return E, S, Dd, A


# ───────────────────────────── Q3 ─────────────────────────────
def q3(E, Tk, null, cells, cal):
    J = E[["pair", "sig_ms", "chk", "w", "day", "block", "half", "oof", "top20", "S_VW", "runner"]].merge(Tk, on=["pair", "sig_ms", "chk"])
    names = [f"TP{tp}/SL{sl}/H{hd}" for tp, sl, hd in cells]
    sels = {"all": np.ones(len(J), bool), "top20": (J.top20.values == 1) & (J.oof.values == 1), "S_VW": J.S_VW.values == 1}
    rows, nrows = [], []
    for sname, smask in sels.items():
        for blk in ("DESIGN", "HOLDOUT"):
            mb = smask & (J.block.values == blk)
            if sname == "top20" and blk == "DESIGN":
                span = cal["DESIGN"] * 4 / 5
            else:
                span = cal[blk]
            for cs in CHK:
                mm = mb & (J.chk.values == cs)
                d = J[mm]
                if not len(d):
                    continue
                w = d.w.values; dy = d.day.values
                N10 = np.vstack(d["net_0.10"].values); N20 = np.vstack(d["net_0.20"].values); N30 = np.vstack(d["net_0.30"].values)
                B10 = np.vstack(d["bnd_0.10"].values); K10 = np.vstack(d["kind_0.10"].values); W10 = np.vstack(d["where_0.10"].values)
                O1m = np.vstack(d["net1m_obs"].values)
                u, inv = np.unique(dy, return_inverse=True)
                sw = np.bincount(inv, w)
                SX = np.vstack([np.bincount(inv, N10[:, c] * w) for c in range(len(cells))]).T     # days × cells
                pk = C.RNG.integers(0, len(u), (NB, len(u)))
                BS = SX[pk].sum(1) / sw[pk].sum(1)[:, None]                                          # NB × cells
                for c, nm in enumerate(names):
                    tp, sl, hd = cells[c]
                    x = N10[:, c]
                    kd = C.kelly_dcr(x, w, w.sum() / span)
                    o = dict(select=sname, block=blk, chk=cs, cell=nm, tp=tp, sl=sl, hold=hd, N=len(d), days=len(u),
                             trades_day=w.sum() / span, avg=C.wmean(x, w), ci_lo=np.percentile(BS[:, c], 2.5),
                             ci_hi=np.percentile(BS[:, c], 97.5), avg_s20=C.wmean(N20[:, c], w), avg_s30=C.wmean(N30[:, c], w),
                             avg_tpfirst=C.wmean(B10[:, c], w), avg_1m=C.wmean(O1m[:, c], w),
                             tp_pct=C.wmean(K10[:, c] == 1, w) * 100, sl_pct=C.wmean(K10[:, c] == 2, w) * 100,
                             time_pct=C.wmean(K10[:, c] == 3, w) * 100, tick_decided_pct=C.wmean(W10[:, c] == 1, w) * 100, **kd)
                    o["dcr"] = kd["dcr"]
                    if o["avg"] > 0:
                        o["loo_coin"] = C.loo(x, w, d.pair.values); o["loo_day"] = C.loo(x, w, dy); o["best5_out"] = C.best5_out(x, w)
                    rows.append(o)
            # family-wise random-entry null (DESIGN only), 1m convention, statistic = max weighted mean over 4 chk × 48 cells
            if blk == "DESIGN":
                obs = []; NL = []
                for ci_, cs in enumerate(CHK):
                    d = J[mb & (J.chk.values == cs)]
                    if not len(d):
                        continue
                    w = d.w.values
                    O1m = np.vstack(d["net1m_obs"].values); ok = np.isfinite(O1m)
                    obs.append(np.nansum(O1m * w[:, None], 0) / (ok * w[:, None]).sum(0))
                    nl = np.stack([null[(p, s)] for p, s in zip(d.pair, d.sig_ms)])          # events × 300 × cells
                    nl = np.roll(nl, ci_ * 75, axis=1)                                         # different random minute per chk
                    okn = np.isfinite(nl)
                    NL.append(np.nansum(nl * w[:, None, None], 0) / (okn * w[:, None, None]).sum(0))   # 300 × cells
                obs = np.concatenate(obs); NLc = np.concatenate(NL, axis=1)
                nmax = NLc.max(1); omax = obs.max()
                nmean = NLc.mean(0)
                for k, nm in enumerate([f"C{c // 60}:{n}" for c in CHK for n in names]):
                    nrows.append(dict(select=sname, cell=nm, obs_1m=obs[k], null_mean=nmean[k], null_p95=np.percentile(NLc[:, k], 95),
                                      excess=obs[k] - nmean[k], p_cell=(NLc[:, k] >= obs[k]).mean()))
                nrows.append(dict(select=sname, cell="FAMILY max", obs_1m=omax, null_mean=nmax.mean(), null_p95=np.percentile(nmax, 95),
                                  excess=omax - nmax.mean(), p_cell=(nmax >= omax).mean()))
    G = pd.DataFrame(rows); Nn = pd.DataFrame(nrows)
    G.to_csv(OUT + "_q3_grid.csv", index=False); Nn.to_csv(OUT + "_q3_null.csv", index=False)
    return J, G, Nn


def main():
    res, E = load()
    cal = res["cal"]
    tk = pickle.load(open(os.path.join(D, "ticks.pkl"), "rb"))
    Q = q1b(E)
    E2, S, Dd, A = q2(E, tk["T"], cal)
    J, G, Nn = q3(E2, tk["T"], tk["null"], tk["cells"], cal)
    pickle.dump(dict(Q1b=Q, S=S, Dd=Dd, A=A, G=G, Nn=Nn, E=E2), open(os.path.join(D, "eval_q23.pkl"), "wb"))
    pd.set_option("display.width", 250)
    print(Q[(Q.period.isin(["HOLDOUT", "DESIGN-OOF"]))].round(2).to_string())


if __name__ == "__main__":
    main()
