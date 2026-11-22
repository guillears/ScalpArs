#!/usr/bin/env python3
"""ROUTE_B_PEAK_STUDY 2026-11-21 — POST-HOC (not pre-registered; labelled as such in the report):
  ① price-only CONTROLS scored exactly like the layer-1 variables (does order flow add anything beyond "price just made a new extreme /
     a fast move"?) — NEWEXT15 (a new in-trade favourable extreme within the last 15 min) · FASTMOVE (3-min favourable move ≥ the sleeve's
     in-trade q90) · OI_FALL (OI 15-min change ≤ 0 WITHOUT the new-extreme condition) · CAPLIQ_NOT_FAST (CAPITULATION_LIQ on minutes that
     are NOT a FASTMOVE).
  ② the pre-registered exit counterfactual for the screen cells with precision ≥ 60 % on ≥ 5 episodes (illustrative, UNVERIFIED — all of
     them sit below their family-wise null).
Usage: venv/bin/python -I scripts/route_b_peak_1121_posthoc.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_b_peak_1121_eval as EV  # noqa: E402


def feat(tr, fn):
    """recompute one screen feature (same construction as EV.screen) over the trade's full panel."""
    for suf in ("_d10", "_d3", "_rel"):
        if fn.endswith(suf):
            base = tr["base"][fn[: -len(suf)]]
            if suf == "_d10":
                return base - EV.lagged(base, 10)
            if suf == "_d3":
                return base - EV.lagged(base, 3)
            med = EV.run_median(np.where(tr["ins"], base, np.nan), tr["i0"])
            with np.errstate(divide="ignore", invalid="ignore"):
                return base / med
    return tr["base"][fn]


def main():
    D = pd.read_pickle(os.path.join(EV.DATA, "panels.pkl"))
    R = pd.read_pickle(os.path.join(EV.DATA, "results.pkl"))
    T = [x for x in (EV.build_trade(pn) for pn in D["panels"] if pn.get("ok")) if x is not None]
    T = [tr for tr in T if tr["mfe"] >= EV.ELIG_MFE and tr["has_book"][tr["dec"]].any()]
    groups = {}
    for sl in ("EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FRENZY_WILLY_HT"):
        for co in ("REAL", "SHADOW"):
            groups[(sl, co)] = [tr for tr in T if tr["m"]["sleeve"] == sl and tr["m"]["cohort"] == co]
    groups[("POOLED_SHORTS", "REAL+SHADOW")] = [tr for tr in T if tr["s"] < 0]
    # ① controls
    rows = []
    for (sl, co), G in groups.items():
        if not G:
            continue
        r3 = np.concatenate([(tr["fav"] - EV.lagged(tr["fav"], 3))[tr["dec"]] for tr in G])
        q90 = np.nanquantile(r3, 0.9)
        for tr in G:
            n, i0 = tr["n"], tr["i0"]
            rmx = EV.run_max(np.where(tr["ins"], tr["fav"], np.nan), i0)
            newext = np.zeros(n, bool)
            for i in range(i0, n):
                w = tr["fav"][max(i0, i - 14): i + 1]
                newext[i] = len(w) and np.nanmax(w) >= rmx[i] - 1e-12
            fast = (tr["fav"] - EV.lagged(tr["fav"], 3)) >= q90
            oi = tr["base"]["oi"]
            tr["V"]["C_NEWEXT15"] = newext
            tr["V"]["C_FASTMOVE"] = fast
            tr["V"]["C_OI_FALL"] = (oi - EV.lagged(oi, 15)) <= 0
            tr["V"]["C_CAPLIQ_NOT_FAST"] = tr["V"]["CAPITULATION_LIQ"] & ~fast
            for k in ("C_NEWEXT15", "C_FASTMOVE", "C_CAPLIQ_NOT_FAST"):
                tr["avail"][k] = np.ones(n, bool) if k != "C_CAPLIQ_NOT_FAST" else tr["avail"]["CAPITULATION_LIQ"]
            tr["avail"]["C_OI_FALL"] = ~np.isnan(oi - EV.lagged(oi, 15))
        for v in ("C_NEWEXT15", "C_FASTMOVE", "C_OI_FALL", "C_CAPLIQ_NOT_FAST", "OI_NO_NEW_MONEY", "CAPITULATION_LIQ"):
            x = EV.layer1(G, v, nshuf=300)
            x.update(sleeve=sl, cohort=co)
            rows.append(x)
    C = pd.DataFrame(rows)
    # ② CF for the screen cells ≥ 60 % on ≥ 5 episodes
    S = R["SCR"]
    pick = S[S.ranked & (S.precision >= 0.60)]
    cf_rows = []
    for (sl, co), cells in pick.groupby(["sleeve", "cohort_"]):
        G = groups.get((sl, co), [])
        if not G:
            continue
        # pooled medians for the 2D cells, same as EV.screen
        cache = {}

        def pooled(fn):
            if fn not in cache:
                cache[fn] = [feat(tr, fn) for tr in G]
            return cache[fn]

        def med(fn):
            arr = np.concatenate([x[tr["dec"]] for x, tr in zip(pooled(fn), G)])
            return np.nanmedian(arr)

        for c in cells.itertuples():
            if c.dim == "1D":
                xs = pooled(c.feature)
                fires = [(x <= c.thr) if c.cond.startswith("≤") else (x >= c.thr) for x in xs]
                avail = [np.isfinite(x) for x in xs]
            else:
                a, b = c.feature.split(" × ")
                sa, sb = c.cond.split("/")
                ma, mb = med(a), med(b)
                fires, avail = [], []
                for xa, xb in zip(pooled(a), pooled(b)):
                    ga, gb = xa >= ma, xb >= mb
                    fires.append(((ga if sa == "hi" else ~ga)) & ((gb if sb == "hi" else ~gb)))
                    avail.append(np.isfinite(xa) & np.isfinite(xb))
            name = f"S:{c.feature} {c.cond}"
            for tr, f, av in zip(G, fires, avail):
                tr["V"][name] = np.asarray(f, bool) & av
                tr["avail"][name] = av
            CF = EV.cf(G, name)
            loo = [CF.d.drop(i).mean() for i in CF.index] if len(CF) > 1 else [np.nan]
            cf_rows.append(dict(sleeve=sl, cohort=co, cell=name, dim=c.dim, hits=c.hits, fa=c.false_alarms, precision=c.precision,
                                fw_null_q95=c.fw_null_q95, n_trades=len(CF), cf_fired=int((CF.d != 0).sum()), mean_d=CF.d.mean(),
                                saved=CF.d[CF.d > 0].sum(), chopped=-CF.d[CF.d < 0].sum(), loo_min=np.nanmin(loo), loo_max=np.nanmax(loo),
                                today_mean=CF.today.mean(), new_mean=CF.new.mean()))
    CFT = pd.DataFrame(cf_rows)
    C.to_csv(os.path.join(EV.REP, "ROUTE_B_PEAK_STUDY_2026-11-21_controls.csv"), index=False)
    CFT.to_csv(os.path.join(EV.REP, "ROUTE_B_PEAK_STUDY_2026-11-21_cf.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    print(C[["sleeve", "cohort", "var", "hits", "false_alarms", "precision", "null_q95", "p_null", "recall", "minute_rate_peak",
             "minute_rate_cont", "windows"]].round(3).to_string())
    if len(CFT):
        print(CFT.groupby(["sleeve", "cohort", "dim"]).agg(cells=("cell", "size"), fired_med=("cf_fired", "median"),
                                                          d_mean=("mean_d", "mean"), d_pos=("mean_d", lambda x: (x > 0).mean()),
                                                          d_best=("mean_d", "max"), d_worst=("mean_d", "min")).round(3).to_string())
        print(CFT.sort_values("mean_d", ascending=False).head(12)[["sleeve", "cohort", "cell", "hits", "fa", "precision", "n_trades",
                                                                   "cf_fired", "mean_d", "saved", "chopped", "loo_min", "loo_max"]].round(2).to_string())


if __name__ == "__main__":
    main()
