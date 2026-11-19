#!/usr/bin/env python3
"""ENTRY_WL (PREREG reports/ENTRY_WL_2026-11-18_PREREG.md) — DESIGN phase. Coverage · 1D (sign / tercile / quintile) + every 2D pair
(sign × sign, tercile × tercile) on RUNNER vs GAVE_BACK (primary) and booked net / WORKED / REACH (secondary) · family-wise null
(200 label shuffles + 200 time rotations) · AUC · survivors + DESIGN checks · theory logistic fit. Freezes the forward list to
reports/ENTRY_WL_2026-11-18_data/<sleeve>_frozen.json. Prints NO HOLDOUT number.
Usage: venv/bin/python -I scripts/entry_wl_design.py exh|dist"""
import json, os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import entry_wl_lib as W  # noqa: E402

NSH = 200
RNG = np.random.default_rng(20261118)


def side_of(L, test):
    """test = dict(kind, i, j) → (bad mask, valid mask, name)."""
    if test["kind"] == "1D":
        li = L[test["i"]]
        if test["j"] == -1:
            a, b = li["mask"], li["valid"] & ~li["mask"]; na, nb = li["lev"], "rest"
        else:
            lj = L[test["j"]]; a, b = li["mask"], lj["mask"]; na, nb = li["lev"], lj["lev"]
        valid = li["valid"]
        bad = a if test["m1"] < test["m2"] else b
        nm = f"{li['var']} {na if test['m1'] < test['m2'] else nb}"
        return bad, valid, nm
    li, lj = L[test["i"]], L[test["j"]]
    cell = li["mask"] & lj["mask"]; valid = li["valid"] & lj["valid"]
    bad = cell if test["m1"] < test["m2"] else (valid & ~cell)
    nm = f"{'' if test['m1'] < test['m2'] else 'NOT '}[{li['var']} {li['lev']} ∧ {lj['var']} {lj['lev']}]"
    return bad, valid, nm


def checks(D, bad, valid, rows, tag):
    """Δ = avg net (kept = valid ∧ ¬bad) − avg net (bad), on fills `rows`; plus the DESIGN robustness items."""
    r = rows & valid
    v = D.net.values; v20 = D.net20.values; days = D.day.values
    kb, kk = r & bad, r & ~bad
    o = dict(tag=tag, N_bad=int(kb.sum()), N_kept=int(kk.sum()), avg_bad=v[kb].mean(), avg_kept=v[kk].mean())
    o["delta"] = o["avg_kept"] - o["avg_bad"]
    y = D.y_rg.values
    o["run_share_bad"] = np.nanmean(y[kb]); o["run_share_kept"] = np.nanmean(y[kk])
    lab = D.label.values
    for L_ in ("RUNNER", "GAVE_BACK", "NEVER_WORKED", "OTHER"):
        o[f"bad_{L_}"] = (lab[kb] == L_).mean() * 100
    lo, hi, lo90, hi90 = W.day_boot_diff(v[r], bad[r], days[r])
    o["ci_lo"], o["ci_hi"] = lo, hi
    hv = D.half.values
    hs = []
    for h in W.HALVES:
        m = r & (hv == h)
        if (m & bad).sum() >= 5 and (m & ~bad).sum() >= 5:
            hs.append(v[m & ~bad].mean() - v[m & bad].mean())
    o["halves_pos"] = f"{sum(x > 0 for x in hs)}/{len(hs)}"
    o["halves"] = " ".join(f"{x:+.2f}" for x in hs)
    for nm, col in (("bd1", "btc_d1_regime"), ("b1d", "btc_ret1d"), ("b4h", "btc_ret4h")):
        if col not in D:
            continue
        s = np.sign(D[col].values)
        parts = []
        for sg in (-1, 1):
            m = r & (s == sg)
            if (m & bad).sum() >= W.MIN_N and (m & ~bad).sum() >= W.MIN_N:
                parts.append(f"{'−' if sg < 0 else '+'}:{v[m & ~bad].mean() - v[m & bad].mean():+.2f}")
            else:
                parts.append(f"{'−' if sg < 0 else '+'}:n<30")
        o[f"reg_{nm}"] = " ".join(parts)
    # concentration of the bad side's excess loss (vs the kept average)
    ex = v[kb] - o["avg_kept"]
    tot = ex.sum()
    if tot < 0:
        pp = pd.Series(ex).groupby(D.pair.values[kb]).sum(); dd = pd.Series(ex).groupby(days[kb]).sum()
        o["top_pair"] = pp.idxmin(); o["pair_share"] = pp.min() / tot * 100; o["day_share"] = dd.min() / tot * 100
    else:
        o["top_pair"] = ""; o["pair_share"] = np.nan; o["day_share"] = np.nan
    o["n_days_bad"] = len(np.unique(days[kb]))
    # best-5 %-out on both sides
    def trim(a):
        a = np.sort(a)[::-1]; return a[int(np.ceil(0.05 * len(a))):].mean()
    o["delta_b5"] = trim(v[kk]) - trim(v[kb])
    o["avg_bad20"] = v20[kb].mean(); o["delta20"] = v20[kk].mean() - v20[kb].mean()
    return o


def main(sleeve):
    D = W.load(sleeve)
    log = []; P = lambda *a: (log.append(" ".join(str(x) for x in a)), print(*a, flush=True))
    dm = (D.block == "DESIGN").values
    V = W.variables(D)
    # ---- coverage
    cov = []
    for c in V:
        x = pd.to_numeric(D[c], errors="coerce")
        cov.append(dict(var=c, theory=W.theory_of(c), market_wide=W.is_mkt(c), cov_all=x.notna().mean() * 100,
                        cov_design=x[dm].notna().mean() * 100, n_unique=x.nunique()))
    COV = pd.DataFrame(cov)
    L = W.levels(D, V, dm)
    tested = pd.Series([l["gran"] for l in L]).groupby([l["var"] for l in L]).agg(lambda s: ",".join(sorted(set(s))))
    COV["granularities"] = COV["var"].map(tested).fillna("— (not testable)")
    COV.to_csv(os.path.join(W.D_OUT, f"{sleeve}_coverage.csv"), index=False)
    missing_theory = [v for v in W.THEORY_VARS if v not in V]
    P(f"== {sleeve}: N {len(D)} DESIGN {dm.sum()} · vars {len(V)} · levels {len(L)} · theory vars not available: {missing_theory}")
    P("labels DESIGN:", D[dm].label.value_counts().to_dict())
    # ---- screens
    rows_p = dm & np.isfinite(D.y_rg.values)
    S_p = W.Screen(D, L, rows_p)
    S_a = W.Screen(D, L, dm)
    ys = dict(primary=(S_p, D.y_rg.values[rows_p]), net=(S_a, D.net.values[dm]), worked=(S_a, D.worked.values[dm].astype(float)),
              reach=(S_a, D.reach.values[dm]))
    res = {}; nulls = {}
    tord = np.argsort(D.entry_ms.values[rows_p])
    for nm, (S, y) in ys.items():
        o1, o2 = S.stats(y)
        res[nm] = (o1, o2)
        t0 = time.time()
        sh = np.array([S.maxabs(RNG.permutation(y)) for _ in range(NSH)])
        rot = np.array([])
        if nm == "primary":
            ys_ = y[tord]; n = len(ys_); rr = []
            for _ in range(NSH):
                off = RNG.integers(int(0.1 * n), int(0.9 * n))
                yy = np.empty(n); yy[tord] = np.roll(ys_, off); rr.append(S.maxabs(yy))
            rot = np.array(rr)
        bar = max(np.percentile(sh, 95), np.percentile(rot, 95) if len(rot) else 0)
        nulls[nm] = dict(shuffle95=np.percentile(sh, 95), rot95=np.percentile(rot, 95) if len(rot) else np.nan, bar=bar,
                         n_tests=int(np.isfinite(o1[:, 0]).sum() + np.isfinite(o2[:, 0]).sum()),
                         best=float(np.nanmax(np.abs(np.r_[o1[:, 0], o2[:, 0]]))))
        P(f"[{nm}] tests {nulls[nm]['n_tests']} · best |t| {nulls[nm]['best']:.2f} · null95 shuffle {nulls[nm]['shuffle95']:.2f}"
          f" rot {nulls[nm]['rot95']:.2f} · bar {bar:.2f} · {time.time() - t0:.0f}s")
    # ---- tables of top tests (primary + net)
    def table(nm, top=25):
        S = ys[nm][0]; o1, o2 = res[nm]; rows = []
        for k, (i, j) in enumerate(S.C1):
            if np.isfinite(o1[k, 0]):
                li = L[i]
                rows.append(dict(kind="1D", i=i, j=j, var=li["var"], gran=li["gran"],
                                 a=li["lev"], b=("rest" if j == -1 else L[j]["lev"]), t=o1[k, 0], m1=o1[k, 1], m2=o1[k, 2],
                                 n1=o1[k, 3], n2=o1[k, 4], theory=W.theory_of(li["var"]), mkt=W.is_mkt(li["var"])))
        for k in np.nonzero(np.isfinite(o2[:, 0]))[0]:
            i, j = S.I[k], S.J[k]
            rows.append(dict(kind="2D", i=i, j=j, var=f"{L[i]['var']} × {L[j]['var']}", gran=L[i]["gran"], a=f"{L[i]['lev']} ∧ {L[j]['lev']}",
                             b="rest", t=o2[k, 0], m1=o2[k, 1], m2=o2[k, 2], n1=o2[k, 3], n2=o2[k, 4],
                             theory=W.theory_of(L[i]["var"]) + "|" + W.theory_of(L[j]["var"]), mkt=W.is_mkt(L[i]["var"]) or W.is_mkt(L[j]["var"])))
        T = pd.DataFrame(rows); T["abs_t"] = T.t.abs(); T = T.sort_values("abs_t", ascending=False)
        T["pass_null"] = T.abs_t > nulls[nm]["bar"]
        return T
    TT = {nm: table(nm) for nm in ys}
    for nm in ys:
        T = TT[nm]
        T.head(300).drop(columns=["i", "j"]).to_csv(os.path.join(W.D_OUT, f"{sleeve}_top_{nm}.csv"), index=False)
        P(f"\n[{nm}] top 1D:"); P(T[T.kind == "1D"].head(15)[["var", "gran", "a", "b", "t", "m1", "m2", "n1", "n2", "theory"]].round(3).to_string(index=False))
        P(f"[{nm}] top 2D:"); P(T[T.kind == "2D"].head(10)[["var", "gran", "a", "t", "m1", "m2", "n1", "n2"]].round(3).to_string(index=False))
        P(f"[{nm}] passing the family-wise bar: {int(T.pass_null.sum())}")
    # ---- market-wide variables in WINDOW (day) units: collapse each side to day-level RUNNER shares
    MW = []
    yb0 = D.y_rg.values; days = D.day.values
    for k, (i, j) in enumerate(ys["primary"][0].C1):
        li = L[i]
        if not W.is_mkt(li["var"]):
            continue
        a = li["mask"] & rows_p; b = (li["valid"] & ~li["mask"] & rows_p) if j == -1 else (L[j]["mask"] & rows_p)
        da = pd.Series(yb0[a]).groupby(days[a]).mean(); db = pd.Series(yb0[b]).groupby(days[b]).mean()
        if len(da) < 8 or len(db) < 8:
            continue
        t = (da.mean() - db.mean()) / np.sqrt(da.var() / len(da) + db.var() / len(db))
        ta = pd.Series(yb0[a]).groupby(days[a]).size(); top = ta.max() / ta.sum() * 100
        MW.append(dict(var=li["var"], gran=li["gran"], a=li["lev"], b="rest" if j == -1 else L[j]["lev"], trades_a=int(a.sum()),
                       days_a=len(da), days_b=len(db), share_a_days=da.mean(), share_b_days=db.mean(), t_days=t,
                       t_trades=res["primary"][0][k, 0], top_day_share_a=top))
    MW = pd.DataFrame(MW)
    if len(MW):
        MW = MW.assign(abs_=MW.t_days.abs()).sort_values("abs_", ascending=False).drop(columns="abs_")
        MW.to_csv(os.path.join(W.D_OUT, f"{sleeve}_marketwide_days.csv"), index=False)
        P("\nmarket-wide variables, WINDOW (day) units, top 12:"); P(MW.head(12).round(3).to_string(index=False))
    # ---- AUC per variable (DESIGN, RUNNER vs GAVE_BACK)
    yb = D.y_rg.values
    A = []
    for c in V:
        x = pd.to_numeric(D[c], errors="coerce").astype(float).values
        a = W.auc(x[rows_p], yb[rows_p])
        if np.isfinite(a):
            A.append(dict(var=c, auc=a, dev=abs(a - 0.5), theory=W.theory_of(c), mkt=W.is_mkt(c)))
    AU = pd.DataFrame(A).sort_values("dev", ascending=False)
    AU.to_csv(os.path.join(W.D_OUT, f"{sleeve}_auc_design.csv"), index=False)
    P("\nAUC (DESIGN, RUNNER vs GAVE_BACK) top 15:"); P(AU.head(15).round(3).to_string(index=False))
    # ---- forward list
    Tp = TT["primary"]
    surv = Tp[Tp.pass_null]
    fwd = []
    if len(surv):
        for _, r in surv.head(10).iterrows():
            fwd.append(dict(r[["kind", "i", "j", "m1", "m2"]]) | dict(src="null-survivor", name=""))
    th = Tp[(Tp.kind == "1D") & (Tp.theory != "")].drop_duplicates("var").head(5)
    for _, r in th.iterrows():
        if not any((f["kind"] == "1D" and f["i"] == r.i and f["j"] == r.j) for f in fwd):
            fwd.append(dict(r[["kind", "i", "j", "m1", "m2"]]) | dict(src="theory-led", name=""))
    # also the best booked-P&L read (secondary) and best REACH read, for the record
    for nm in ("net", "reach"):
        r = TT[nm][(TT[nm].kind == "1D")].iloc[0]
        # for net/reach, 'bad' side = lower mean
        fwd.append(dict(r[["kind", "i", "j", "m1", "m2"]]) | dict(src=f"best-{nm}", name=""))
    CH = []
    for f in fwd:
        f["i"] = int(f["i"]); f["j"] = int(f["j"]); f["m1"] = float(f["m1"]); f["m2"] = float(f["m2"])
        bad, valid, nm = side_of(L, f); f["name"] = nm
        lv = L[f["i"]]; f["var"] = lv["var"]; f["gran"] = lv["gran"]; f["lev_i"] = lv["lev"]
        f["lo"] = float(lv.get("lo", np.nan)); f["hi"] = float(lv.get("hi", np.nan))
        if f["kind"] == "2D":
            lj = L[f["j"]]; f["var2"] = lj["var"]; f["lev_j"] = lj["lev"]; f["lo2"] = float(lj.get("lo", np.nan)); f["hi2"] = float(lj.get("hi", np.nan))
        elif f["j"] >= 0:
            lj = L[f["j"]]; f["lev_j"] = lj["lev"]; f["lo2"] = float(lj.get("lo", np.nan)); f["hi2"] = float(lj.get("hi", np.nan))
        c = checks(D, bad, valid, dm, nm) | dict(src=f["src"])
        CH.append(c)
    CH = pd.DataFrame(CH)
    CH.to_csv(os.path.join(W.D_OUT, f"{sleeve}_design_checks.csv"), index=False)
    P("\nDESIGN checks (bad side = lower RUNNER share / lower mean):")
    P(CH[["src", "tag", "N_bad", "N_kept", "run_share_bad", "run_share_kept", "avg_bad", "avg_kept", "delta", "ci_lo", "ci_hi", "halves_pos",
          "halves", "reg_b1d", "reg_b4h", "pair_share", "day_share", "n_days_bad", "delta_b5", "delta20"]].round(3).to_string(index=False))
    # ---- theory logistic (DESIGN fit)
    tv = [v for v in W.THEORY_VARS if v in V and pd.to_numeric(D[v], errors="coerce")[dm].notna().mean() >= 0.5]
    Xm = np.column_stack([pd.to_numeric(D[v], errors="coerce").astype(float).values for v in tv])
    med = np.nanmedian(Xm[dm], 0); Xm = np.where(np.isfinite(Xm), Xm, med)
    lo_, hi_ = np.nanpercentile(Xm[dm], 1, 0), np.nanpercentile(Xm[dm], 99, 0); Xm = np.clip(Xm, lo_, hi_)
    mu, sd = Xm[dm].mean(0), Xm[dm].std(0) + 1e-12; Z = (Xm - mu) / sd
    w = W.logistic_fit(Z[rows_p], yb[rows_p], lam=float(rows_p.sum()) * 0.05)
    sc = W.logistic_score(Z, w)
    a_in = W.auc(sc[rows_p], yb[rows_p])
    # 5-fold time-blocked CV inside DESIGN (honest DESIGN AUC)
    di = np.nonzero(rows_p)[0]; di = di[np.argsort(D.entry_ms.values[di])]; folds = np.array_split(di, 5); cv = np.full(len(D), np.nan)
    for k in range(5):
        tr = np.setdiff1d(di, folds[k]); wk = W.logistic_fit(Z[tr], yb[tr], lam=len(tr) * 0.05); cv[folds[k]] = W.logistic_score(Z[folds[k]], wk)
    a_cv = W.auc(cv[rows_p], yb[rows_p])
    coef = pd.DataFrame(dict(var=tv, coef=w[1:])).assign(abs_=lambda d: d.coef.abs()).sort_values("abs_", ascending=False)
    P(f"\ntheory logistic ({len(tv)} vars): DESIGN in-sample AUC {a_in:.3f} · time-blocked 5-fold CV AUC {a_cv:.3f}")
    P(coef.head(12).round(3).to_string(index=False))
    model = dict(vars=tv, med=med.tolist(), lo=lo_.tolist(), hi=hi_.tolist(), mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(),
                 auc_in=a_in, auc_cv=a_cv)
    # score tercile edges on DESIGN (for the model-based action)
    model["q_edges"] = np.quantile(cv[rows_p & np.isfinite(cv)], [1 / 3, 2 / 3]).tolist()
    model["q_edges_all"] = np.quantile(sc[dm], [1 / 3, 2 / 3]).tolist()
    frozen = dict(frozen_utc=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), sleeve=sleeve, nulls=nulls, forward=fwd, model=model,
                  n_levels=len(L))
    json.dump(frozen, open(os.path.join(W.D_OUT, f"{sleeve}_frozen.json"), "w"), indent=1, default=float)
    pd.to_pickle(dict(L=[{k: v for k, v in l.items()} for l in L], TT=TT, AU=AU, COV=COV, CH=CH, coef=coef),
                 os.path.join(W.D_OUT, f"{sleeve}_design.pkl"))
    open(os.path.join(W.D_OUT, f"{sleeve}_design.log"), "w").write("\n".join(log))
    P("frozen →", frozen["frozen_utc"])


if __name__ == "__main__":
    main(sys.argv[1])
