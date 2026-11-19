#!/usr/bin/env python3
"""FRENZY LONG SEARCH — EARLIER-DETECTOR eval (PREREG addendum B): 135 cells, family-wise random-timing null, K = 10 book, earliness,
move left, false positives, regime signs (window units) for the top cells. Writes _early_cells.csv, _early_regime.csv, data/early_eval.pkl."""
import os, pickle, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_eval as EV, frenzy_long_search_flag_eval as FE   # noqa: E402
D = EV.D; OUT = EV.OUT; DAY = EV.DAY
EXN = ("Q3", "Q5", "TR", "T1", "T4")
RNG = np.random.default_rng(11)


def k10(d, ex):
    d = d.sort_values("entry_ms"); ox = []; keep = []
    for e_, x_ in zip(d.entry_ms.values, d[ex + "_x"].values):
        ox = [y for y in ox if y > e_]
        if len(ox) < 10:
            ox.append(x_); keep.append(True)
        else:
            keep.append(False)
    return d[np.array(keep)]


def dcr(d, x, f, ex, d0, d1):
    days = np.arange(d0, d1 + 1)
    pnl = pd.Series(x * f / 100).groupby(d[ex + "_x"].values // DAY).sum().reindex(days).fillna(0).values
    eq = np.cumprod(1 + np.maximum(pnl, -0.99))
    return (eq[-1] ** (1 / len(days)) - 1) * 100, (eq / np.maximum.accumulate(eq) - 1).min() * 100


def main():
    T = pd.read_csv(os.path.join(D, "tape_daily.csv"), index_col=0); A = pd.read_csv(OUT + "_analogs.csv", index_col=0)
    X = FE.tag(pickle.load(open(os.path.join(D, "early_rows.pkl"), "rb")), T, A)
    pool = pickle.load(open(os.path.join(D, "early_pool.pkl"), "rb"))
    d0, d1 = int(X.day.min()), int(X.day.max()); ncal = d1 - d0 + 1
    rows = []; keep = {}
    for cell, g in X.groupby("cell"):
        for ex in EXN:
            d = EV.dedupe(g, ex); x = FE.netf(d, ex)
            s = EV.stats(x, d)
            hv = pd.Series(x).groupby(d.half.values).mean()
            b = k10(d, ex); xb = FE.netf(b, ex); sb = EV.stats(xb, b)
            fk = max(0.0, sb["kelly_floor"]) if sb["kelly_floor"] == sb["kelly_floor"] else 0.0
            sp = d[d.spike_after > 0]
            o = dict(cell=cell, exit=ex, N=s["N"], trades_day=s["N"] / ncal, coins_day=d.groupby("day").pair.nunique().sum() / ncal,
                     WR=s["WR"], avg=s["avg"], med=s["med"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], best5_out=s["best5_out"],
                     kelly_floor=s["kelly_floor"], slip20=FE.netf(d, ex, 0.20).mean(), halves_pos=f"{int((hv > 0).sum())}/{len(hv)}",
                     DESIGN=x[(d.block == "DESIGN").values].mean(), HOLD_A=x[(d.block == "HOLD-A").values].mean(),
                     HOLD_B=x[(d.block == "HOLD-B").values].mean(),
                     t_design=EV.cl_t(x[(d.block == "DESIGN").values], d[d.block == "DESIGN"].day.values),
                     fp_rate=d.fp.mean() * 100, share_became_frenzy=(d.spike_after > 0).mean() * 100,
                     lead_min_med=((sp.spike_after - sp.sig_ms) / 60000).median() if len(sp) else np.nan,
                     gain_vs_flag_entry_med=((sp.px_ratio_flag - 1) * 100).median() if len(sp) else np.nan,
                     K10_N=sb["N"], K10_trades_day=sb["N"] / ncal, K10_avg=sb["avg"], K10_best5_out=sb["best5_out"],
                     K10_kelly_floor=sb["kelly_floor"], K10_DESIGN=xb[(b.block == "DESIGN").values].mean(),
                     K10_HOLD_A=xb[(b.block == "HOLD-A").values].mean(), K10_HOLD_B=xb[(b.block == "HOLD-B").values].mean())
            o["K10_dcr_018"], o["K10_dd_018"] = dcr(b, xb, 0.18, ex, d0, d1)
            o["K10_dcr_kelly"] = dcr(b, xb, fk, ex, d0, d1)[0] if fk > 0 else 0.0
            rows.append(o); keep[(cell, ex)] = d.assign(net=x)
    C = pd.DataFrame(rows); C.to_csv(OUT + "_early_cells.csv", index=False)
    # family-wise random-timing null (DESIGN, no dedupe beyond the 6 h debounce)
    pool["key"] = pool.pair + "|" + pool.month
    pool = pool.sort_values("key").reset_index(drop=True)
    kk = pool.key.values; st = {}; ln = {}
    u, first, cnt = np.unique(kk, return_index=True, return_counts=True)
    st = dict(zip(u, first)); ln = dict(zip(u, cnt)); pidx = st
    pn = {ex: FE.netf(pool, ex) for ex in EXN}
    obs = C.t_design.max()
    Xd = X[X.block == "DESIGN"].copy(); Xd["key"] = Xd.pair + "|" + Xd.month
    Xd = Xd[Xd.key.isin(pidx.keys())]
    Xd["pst"] = Xd.key.map(st).astype(int); Xd["pln"] = Xd.key.map(ln).astype(int)
    groups = {c: g for c, g in Xd.groupby("cell")}
    null = []
    for it in range(200):
        best = -9
        for c, g in groups.items():
            pick = g.pst.values + (RNG.random(len(g)) * g.pln.values).astype(int)
            for ex in EXN:
                best = max(best, EV.cl_t(pn[ex][pick], g.day.values))
        null.append(best)
    null = np.array(null); p_fw = float((null >= obs).mean())
    # mean of the random-timing pool (what a random bar on the same coins/months earns)
    rnd = {ex: pn[ex].mean() for ex in EXN}
    # regime for the top 5 by K10 DCR at 0.18 (then by K10 avg)
    top = C.sort_values(["K10_dcr_kelly", "K10_dcr_018"], ascending=False).head(5)
    reg = []
    for _, r in top.iterrows():
        d = keep[(r.cell, r.exit)]
        for rv in ("R1", "RF", "R6", "R5", "analog"):
            for sg in (1, 0):
                s_ = d[d[rv] == sg]
                if len(s_):
                    reg.append(dict(cell=r.cell, exit=r.exit, regime=rv, side=sg, N=len(s_), days=s_.day.nunique(),
                                    r1_windows=s_.r1_win.nunique(), avg=s_.net.mean(), t=EV.cl_t(s_.net.values, s_.day.values)))
    RG = pd.DataFrame(reg); RG.to_csv(OUT + "_early_regime.csv", index=False)
    pickle.dump(dict(C=C, null=null, obs=obs, p_fw=p_fw, rnd=rnd, RG=RG), open(os.path.join(D, "early_eval.pkl"), "wb"))
    pd.set_option("display.width", 320)
    print(f"family-wise: obs max t {obs:.2f} null p95 {np.percentile(null, 95):.2f} p {p_fw:.3f}; random-bar pool avg", {k: round(v, 3) for k, v in rnd.items()})
    cols = ["cell", "exit", "trades_day", "avg", "med", "ci_lo", "DESIGN", "HOLD_A", "HOLD_B", "best5_out", "halves_pos", "fp_rate",
            "share_became_frenzy", "lead_min_med", "gain_vs_flag_entry_med", "K10_trades_day", "K10_avg", "K10_HOLD_A", "K10_HOLD_B", "K10_kelly_floor", "K10_dcr_018", "K10_dcr_kelly"]
    print(C.sort_values("K10_dcr_018", ascending=False)[cols].head(25).round(2).to_string())
    print("cells with avg>0:", (C.avg > 0).sum(), "of", len(C), "; K10 avg>0:", (C.K10_avg > 0).sum(), "; K10 avg>0 in DESIGN&A&B:", ((C.K10_DESIGN > 0) & (C.K10_HOLD_A > 0) & (C.K10_HOLD_B > 0)).sum())
    print(C.groupby("exit")[["avg", "K10_avg"]].mean().round(3))
    print(RG.round(2).to_string())


if __name__ == "__main__":
    main()
