#!/usr/bin/env python3
"""ROUTE A — EVALUATION (PREREG reports/ROUTE_A_TICKS_2026-11-19_PREREG.md).
Per-feature AUC + family-wise label-shuffle null · TICK / CANDLE / BOTH logistic models with half-year time-split CV, label-shuffle
null (TICK) and tick-block permutation null (ΔAUC) · PATH label · HOLDOUT once · priced long (top-20 % BOTH score vs all triggers),
population-weighted · regime in window units. Writes reports/ROUTE_A_TICKS_2026-11-19_*.csv and data/eval.pkl."""
import os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
from route_a_ticks_features import TICK, CAND   # noqa: E402
D = os.path.join(R, "reports", "ROUTE_A_TICKS_data"); OUT = os.path.join(R, "reports", "ROUTE_A_TICKS_2026-11-19")
DAY = 86_400_000
CHK = (0, 60, 120, 300)
SIGN = dict(tps_accel=1, tps_burst=1, buy_cnt=1, buy_qv=1, buy_qv_last=1, big_lift=1, med_trade=1, top10_share=1, retrace=-1,
            run_ratio=1, last_conc=1, spread_bps=-1, uptick_share=1, vwap_gap=1)
M_TICK = ["tps_accel", "buy_qv", "buy_qv_last", "retrace", "last_conc", "spread_bps"]
M_CAND = ["ret15", "volx15", "ret6h_base", "close_pos", "logq24", "since_burst_h"]
M_BOTH = M_TICK + M_CAND
HALVES = ["24-H1", "24-H2", "25-H1", "25-H2", "26-H1"]
RNG = np.random.default_rng(20261119)
EXN = ("Q3", "TR", "T1")
DRY = os.environ.get("ROUTE_A_DRYRUN") == "1"
NFW, NMOD = (5, 3) if DRY else (1000, 200)


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logit_fit(X, y, C=1.0, it=50):
    X1 = np.c_[np.ones(len(X)), X]; w = np.zeros(X1.shape[1]); lam = 1.0 / C
    for _ in range(it):
        z = np.clip(X1 @ w, -30, 30); p = 1 / (1 + np.exp(-z))
        g = X1.T @ (p - y) + lam * np.r_[0, w[1:]]
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


class Prep:
    def __init__(self, Xtr):
        self.lo = np.nanpercentile(Xtr, 1, axis=0); self.hi = np.nanpercentile(Xtr, 99, axis=0)
        Z = np.clip(Xtr, self.lo, self.hi); self.med = np.nanmedian(Z, axis=0)
        Z = np.where(np.isnan(Z), self.med, Z); self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-12

    def __call__(self, X):
        Z = np.clip(X, self.lo, self.hi); Z = np.where(np.isnan(Z), self.med, Z)
        return (Z - self.mu) / self.sd


def fit_score(Xtr, ytr, Xte):
    P = Prep(Xtr); w = logit_fit(P(Xtr), ytr)
    return np.c_[np.ones(len(Xte)), P(Xte)] @ w


def oof(df, cols, y):
    """half-year expanding time-split; returns OOF score (NaN in 24-H1)"""
    s = np.full(len(df), np.nan); X = df[cols].values.astype(float); h = df.half.values
    for k in range(1, len(HALVES)):
        tr = np.isin(h, HALVES[:k]); te = h == HALVES[k]
        if te.any() and len(np.unique(y[tr])) == 2:
            s[te] = fit_score(X[tr], y[tr], X[te])
    return s


def net(d, ex, slip=0.10):
    g = d[ex + "_g"].values / 100; k = d[ex + "_k"].values; f = d[ex + "_f"].values; s = slip / 100
    out_px = np.where(k == 2, 1 + g, (1 + g) * (1 - s)); fee = np.where(k == 2, 0.02, 0.05)
    return ((out_px / (1 + s)) - 1) * 100 - 0.05 - fee - f


def wstats(x, w, d):
    x = np.asarray(x, float); w = np.asarray(w, float)
    if len(x) == 0:
        return dict(N=0)
    avg = np.average(x, weights=w); days = d.day.values
    u, inv = np.unique(days, return_inverse=True)
    sx = np.bincount(inv, x * w); sw = np.bincount(inv, w)
    bs = []
    for _ in range(1000):
        pk = RNG.integers(0, len(u), len(u))
        bs.append(sx[pk].sum() / sw[pk].sum())
    o = np.argsort(x)[::-1]; cut = int(np.ceil(len(x) * 0.05))
    keep = o[cut:]
    loo_c = min(np.average(x[d.pair.values != p], weights=w[d.pair.values != p]) for p in np.unique(d.pair.values)) if d.pair.nunique() > 1 else np.nan
    loo_d = min(np.average(x[days != dd], weights=w[days != dd]) for dd in u) if len(u) > 1 else np.nan
    return dict(N=len(x), days=len(u), coins=d.pair.nunique(), WR=np.average(x > 0, weights=w) * 100, avg=avg,
                med=float(np.median(x)), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                best5_out=np.average(x[keep], weights=w[keep]) if len(keep) else np.nan, loo_coin_min=loo_c, loo_day_min=loo_d)


def main():
    F = pickle.load(open(os.path.join(D, "features.pkl"), "rb"))
    E = F[F.kind == "event"].copy()
    if DRY:   # pipeline test only: labels scrambled, few null draws (no result is read)
        E["runner"] = RNG.permutation(E.runner.values); E["worked"] = RNG.permutation(E.worked.values)
    BL = F[F.kind == "baseline"].copy()
    dt = pd.to_datetime(E.sig_ms, unit="ms")
    E["half"] = dt.dt.year.astype(str).str[2:] + np.where(dt.dt.month <= 6, "-H1", "-H2")
    E["day"] = E.entry_ms // DAY
    Tp = pd.read_csv(os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data", "tape_daily.csv"), index_col=0)
    for k in ("R1", "R5"):
        E[k] = E.day.map(Tp[k])
    r1 = Tp["R1"].fillna(-1); wid = (r1 != r1.shift()).cumsum(); E["r1_win"] = E.day.map(wid)
    r5 = Tp["R5"].fillna(-1); wid5 = (r5 != r5.shift()).cumsum(); E["r5_win"] = E.day.map(wid5)
    # population weights: month × class count ÷ sampled (at C0)
    smp = E[E.chk == 0].groupby(["month", "runner"]).size().rename("smp_n")
    E = E.join(smp, on=["month", "runner"]); E["w"] = E.pop_n / E.smp_n
    trig_rate = pd.read_csv(os.path.join(D, "order.csv"))
    cal = {"DESIGN": (trig_rate[trig_rate.block == "DESIGN"].sig_ms.max() - trig_rate[trig_rate.block == "DESIGN"].sig_ms.min()) / DAY,
           "HOLDOUT": (trig_rate[trig_rate.block == "HOLDOUT"].sig_ms.max() - trig_rate[trig_rate.block == "HOLDOUT"].sig_ms.min()) / DAY}
    res = dict(cal=cal)
    # ── coverage
    cov = E.groupby(["chk", "block", "runner"]).size().unstack([1, 2])
    cov["already_flagged"] = E.groupby("chk").already_flagged.sum()
    res["coverage"] = cov; print(cov.to_string())
    E = E[E.already_flagged == 0].copy()
    # ── 1. per-feature AUC + family-wise null (labels shuffled within month, per event)
    Dd = E[E.block == "DESIGN"]
    ev_keys = Dd[["pair", "sig_ms", "month", "runner"]].drop_duplicates(["pair", "sig_ms"]).reset_index(drop=True)
    ev_keys["eid"] = np.arange(len(ev_keys))
    Dd = Dd.merge(ev_keys[["pair", "sig_ms", "eid"]], on=["pair", "sig_ms"])
    rows = []; ranks = {}
    for cs in CHK:
        g = Dd[Dd.chk == cs]
        for f in TICK + CAND:
            a = auc(g[f].values, g.runner.values); aw = auc(g[f].values, g.worked.values)
            sg = SIGN.get(f, 0)
            rows.append(dict(chk=cs, feature=f, side="tick" if f in TICK else "candle", sign_prereg=sg, N=g[f].notna().sum(),
                             auc_runner=a, auc_runner_signed=(a if sg >= 0 else 1 - a) if sg else np.nan,
                             auc_worked=aw, med_runner=g.loc[g.runner == 1, f].median(), med_fizzle=g.loc[g.runner == 0, f].median(),
                             med_baseline=BL[BL.block == "DESIGN"][f].median() if cs == 0 else np.nan))
            if f in TICK:
                v = g[f].values.astype(float); ok = np.isfinite(v)
                ranks[(cs, f)] = (pd.Series(v[ok]).rank().values, g.eid.values[ok])
    A = pd.DataFrame(rows)
    obs_fw = (A[A.side == "tick"].auc_runner - 0.5).abs().max()
    lab0 = ev_keys.runner.values; mon = ev_keys.month.values
    null = []
    for it in range(NFW):
        lab = lab0.copy()
        for mo in np.unique(mon):
            ii = np.where(mon == mo)[0]; lab[ii] = lab0[RNG.permutation(ii)]
        best = 0
        for (cs, f), (rk, eid) in ranks.items():
            y = lab[eid]; n1 = y.sum(); n0 = len(y) - n1
            a = (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            best = max(best, abs(a - 0.5))
        null.append(best)
    null = np.array(null)
    A["p_fw"] = [(null >= abs(x - 0.5)).mean() if s == "tick" else np.nan for x, s in zip(A.auc_runner, A.side)]
    res.update(A=A, fw_obs=obs_fw, fw_null95=np.percentile(null, 95), fw_p=(null >= obs_fw).mean())
    A.to_csv(OUT + "_feature_auc.csv", index=False)
    print(f"family-wise: max|AUC-0.5| {obs_fw:.3f} null95 {np.percentile(null,95):.3f} p {(null>=obs_fw).mean():.3f}")
    # ── 2. models, OOF on DESIGN
    MR = []; scores = {}
    for cs in CHK:
        g = E[(E.block == "DESIGN") & (E.chk == cs)].sort_values("sig_ms").reset_index(drop=True)
        for tgt in ("runner", "worked"):
            y = g[tgt].values.astype(int)
            s = {nm: oof(g, cols, y) for nm, cols in (("TICK", M_TICK), ("CANDLE", M_CAND), ("BOTH", M_BOTH))}
            ok = np.isfinite(s["BOTH"])
            a = {nm: auc(v[ok], y[ok]) for nm, v in s.items()}
            # nulls (200 draws each)
            nl_t, nl_d = [], []
            mon = g.month.values
            for it in range(NMOD):
                yy = y.copy()
                for mo in np.unique(mon):
                    ii = np.where(mon == mo)[0]; yy[ii] = y[RNG.permutation(ii)]
                st = oof(g, M_TICK, yy); nl_t.append(auc(st[ok], yy[ok]))
                gp = g.copy(); perm = np.arange(len(g))
                for mo in np.unique(mon):
                    ii = np.where(mon == mo)[0]; perm[ii] = RNG.permutation(ii)
                gp[M_TICK] = g[M_TICK].values[perm]
                sb = oof(gp, M_BOTH, y); nl_d.append(auc(sb[ok], y[ok]) - a["CANDLE"])
            nl_t = np.array(nl_t); nl_d = np.array(nl_d)
            dA = a["BOTH"] - a["CANDLE"]
            MR.append(dict(chk=cs, target=tgt, N_oof=int(ok.sum()), auc_TICK=a["TICK"], auc_CANDLE=a["CANDLE"], auc_BOTH=a["BOTH"],
                           dAUC=dA, p_TICK=(nl_t >= a["TICK"]).mean(), null95_TICK=np.percentile(nl_t, 95),
                           p_dAUC=(nl_d >= dA).mean(), null95_dAUC=np.percentile(nl_d, 95),
                           **{f"auc_BOTH_{h}": auc(s["BOTH"][g.half.values == h], y[g.half.values == h]) for h in HALVES[1:]}))
            if tgt == "runner":
                scores[cs] = (g, s)
            print(MR[-1], flush=True)
    MRd = pd.DataFrame(MR); MRd.to_csv(OUT + "_models.csv", index=False); res["models"] = MRd
    # ── 3. HOLDOUT once
    HR = []; hold = {}
    for cs in CHK:
        g = E[(E.block == "DESIGN") & (E.chk == cs)]; h = E[(E.block == "HOLDOUT") & (E.chk == cs)].reset_index(drop=True)
        for tgt in ("runner", "worked"):
            o = dict(chk=cs, target=tgt, N=len(h), runners=int(h.runner.sum()))
            for nm, cols in (("TICK", M_TICK), ("CANDLE", M_CAND), ("BOTH", M_BOTH)):
                sc = fit_score(g[cols].values.astype(float), g[tgt].values.astype(int), h[cols].values.astype(float))
                o["auc_" + nm] = auc(sc, h[tgt].values)
                if tgt == "runner" and nm == "BOTH":
                    hold[cs] = (h, sc)
            HR.append(o)
    HRd = pd.DataFrame(HR); HRd.to_csv(OUT + "_holdout.csv", index=False); res["holdout"] = HRd; print(HRd.round(3).to_string())
    # ── 4. priced
    PR = []; trade_sets = {}
    for cs in CHK:
        g, s = scores[cs]; sb = s["BOTH"]; ok = np.isfinite(sb)
        go = g[ok].copy(); go["score"] = sb[ok]
        thr = np.quantile(np.repeat(go.score.values, np.maximum(1, np.round(go.w.values / go.w.min()).astype(int))), 0.80)
        h, sh = hold[cs]; h = h.copy(); h["score"] = sh
        allD = g.copy()
        sets = {("DESIGN-OOF", "all"): go, ("DESIGN-OOF", "top20"): go[go.score >= thr], ("DESIGN-all", "all"): allD,
                ("HOLDOUT", "all"): h, ("HOLDOUT", "top20"): h[h.score >= thr],
                ("DESIGN-OOF", "runner(hindsight)"): go[go.runner == 1], ("DESIGN-OOF", "fizzle(hindsight)"): go[go.runner == 0]}
        for (blk, sel), d in sets.items():
            for ex in EXN:
                if not len(d) or ex + "_g" not in d.columns:
                    continue
                dd = d[d[ex + "_complete"].astype(bool)]
                x = net(dd, ex); w = dd.w.values
                st = wstats(x, w, dd)
                span = cal["HOLDOUT"] if blk == "HOLDOUT" else (cal["DESIGN"] * (4 / 5) if blk == "DESIGN-OOF" else cal["DESIGN"])
                st.update(chk=cs, block=blk, select=sel, exit=ex, trades_day=w.sum() / span,
                          slip30=np.average(net(dd, ex, 0.30), weights=w) if len(dd) else np.nan,
                          slip50=np.average(net(dd, ex, 0.50), weights=w) if len(dd) else np.nan,
                          gross=np.average(dd[ex + "_g"].values, weights=w) if len(dd) else np.nan,
                          share_runner=np.average(dd.runner.values, weights=w) * 100 if len(dd) else np.nan,
                          worked=np.average(dd.worked.values, weights=w) * 100 if len(dd) else np.nan)
                for rv in ("R1", "R5"):
                    for sg in (1, 0):
                        m_ = (dd[rv] == sg).values
                        if m_.sum():
                            st[f"{rv}_{sg}_avg"] = np.average(x[m_], weights=w[m_]); st[f"{rv}_{sg}_N"] = int(m_.sum())
                            st[f"{rv}_{sg}_win"] = dd[m_][rv.lower() + "_win"].nunique()
                PR.append(st)
            trade_sets[(cs, blk, sel)] = d
    PRd = pd.DataFrame(PR); PRd.to_csv(OUT + "_priced.csv", index=False); res["priced"] = PRd
    # ── 5. EXPLORATORY (added after the pre-registered reads): score quintiles (dose-response) and runner / fizzle inside the top 20 %
    QR = []
    for cs in CHK:
        g, s = scores[cs]; ok = np.isfinite(s["BOTH"]); d = g[ok].copy(); d["sc"] = s["BOTH"][ok]
        d["q"] = pd.qcut(d.sc, 5, labels=False)
        for q_, x in d.groupby("q"):
            o = dict(chk=cs, view="quintile", group=f"Q{q_ + 1}", N=len(x), runner_pct=np.average(x.runner, weights=x.w) * 100,
                     worked_pct=np.average(x.worked, weights=x.w) * 100, reach_med=x.reach.median())
            for ex in EXN:
                o[ex + "_net"] = np.average(net(x, ex), weights=x.w)
            QR.append(o)
        for nm in ("all", "top20"):
            dd = trade_sets[(cs, "DESIGN-OOF", nm)]
            for lab in (1, 0):
                x = dd[dd.runner == lab]
                o = dict(chk=cs, view="class_in_" + nm, group="runner" if lab else "fizzle", N=len(x), reach_med=x.reach.median(),
                         ret15_med=x.ret15.median(), vwap_gap_med=x.vwap_gap.median())
                for ex in EXN:
                    o[ex + "_net"] = np.average(net(x, ex), weights=x.w)
                QR.append(o)
    pd.DataFrame(QR).to_csv(OUT + "_dose_response.csv", index=False)
    res["trade_sets"] = trade_sets; res["scores"] = scores; res["E"] = E; res["BL"] = BL
    pickle.dump(res, open(os.path.join(D, "eval.pkl"), "wb"))
    pd.set_option("display.width", 300)
    print(PRd[["chk", "block", "select", "exit", "N", "trades_day", "WR", "gross", "avg", "ci_lo", "ci_hi", "best5_out", "slip30",
               "share_runner", "worked"]].round(2).to_string())


if __name__ == "__main__":
    main()
