#!/usr/bin/env python3
"""FRENZY LONG SEARCH — FLAG / FAST eval (PREREG addendum A). Writes reports/FRENZY_LONG_SEARCH_2026-11-19_flag_cells.csv,
_flag_regime.csv, _flag_lag.csv, _fast_cells.csv and FRENZY_LONG_SEARCH_data/flag_eval.pkl."""
import os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_eval as EV   # noqa: E402
D = EV.D; OUT = EV.OUT; DAY = EV.DAY
EXS = ["Q3", "Q5", "TR", "T1", "T4", "H72"]


def netf(d, ex, slip=0.10):
    g = d[ex + "_g"].values / 100; k = d[ex + "_k"].values; f = d[ex + "_f"].values; s = slip / 100
    out_px = np.where(k == 2, 1 + g, (1 + g) * (1 - s)); fee = np.where(k == 2, 0.02, 0.05)
    return ((out_px / (1 + s)) - 1) * 100 - 0.05 - fee - f


def tag(df, T, A):
    df["day"] = df.entry_ms // DAY
    dt = pd.to_datetime(df.entry_ms, unit="ms")
    df["month"] = dt.dt.strftime("%Y-%m")
    df["half"] = dt.dt.year.astype(str).str[2:] + np.where(dt.dt.month <= 6, "-H1", "-H2")
    df["block"] = np.where(df.entry_ms < pd.Timestamp("2025-01-01").value // 10**6, "HOLD-A",
                           np.where(df.entry_ms < pd.Timestamp("2026-07-01").value // 10**6, "DESIGN", "HOLD-B"))
    for k in ("R1", "R6", "R5", "RF"):
        df[k] = df.day.map(T[k])
    r1 = T["R1"].fillna(-1); wid = (r1 != r1.shift()).cumsum(); df["r1_win"] = df.day.map(wid)
    df["analog"] = df.day.isin(set(A.index[A.analog])).astype(int)
    return df


def cells(df, ex_list, groups, label):
    rows = []
    for ex in ex_list:
        ok = df[ex + "_complete"].values
        for gname, gmask in groups:
            d = df[ok & gmask]
            d = EV.dedupe(d, ex)
            x = netf(d, ex)
            for blk, sel in [("ALL", np.ones(len(d), bool))] + [(b, (d.block == b).values) for b in ("DESIGN", "HOLD-A", "HOLD-B")]:
                s = EV.stats(x[sel], d[sel], full=(blk == "ALL"))
                s.update(set=label, group=gname, exit=ex, block=blk)
                if blk == "ALL" and len(d):
                    s["slip20"] = netf(d, ex, 0.20).mean()
                    hv = pd.Series(x).groupby(d.half.values).mean()
                    for hk, hvv in hv.items():
                        s["h_" + hk] = hvv
                    s["halves_pos"] = f"{int((hv > 0).sum())}/{len(hv)}"
                rows.append(s)
    return pd.DataFrame(rows)


def main():
    T = pd.read_csv(os.path.join(D, "tape_daily.csv"), index_col=0)
    A = pd.read_csv(OUT + "_analogs.csv", index_col=0)
    F = tag(pickle.load(open(os.path.join(D, "flag_rows.pkl"), "rb")), T, A)
    F["mv_bucket"] = pd.cut(F.moved, [-1e9, 10, 20, 1e9], labels=["<10", "10-20", ">20"]).astype(str)
    # ── detection lag table
    q = F[["moved", "lag_min", "late_min", "gain1h"]].describe(percentiles=[.1, .25, .5, .75, .9]).T
    q.to_csv(OUT + "_flag_lag.csv")
    lag_by = F.groupby("mv_bucket").agg(N=("pair", "size"), lag_med=("lag_min", "median"), late_med=("late_min", "median"),
                                        share_late=("late_min", lambda s: (s > 0).mean() * 100))
    groups = [("ALL", np.ones(len(F), bool))] + [("moved " + b, (F.mv_bucket == b).values) for b in ("<10", "10-20", ">20")] + \
             [("at spike close", (F.late_min <= 0).values), ("shortlisted later", (F.late_min > 0).values)]
    C = cells(F, EXS, groups, "FLAG")
    C.to_csv(OUT + "_flag_cells.csv", index=False)
    # ── regime splits on the primary (TR) and Q3, all + moved<10, window units
    reg = []
    for ex in ("TR", "Q3", "H72"):
        for gname, gm in groups[:2]:
            d = EV.dedupe(F[F[ex + "_complete"] & gm], ex); x = netf(d, ex); d = d.assign(net=x)
            for rv in ("R1", "R6", "R5", "RF", "analog"):
                for sg in (1, 0):
                    s_ = d[d[rv] == sg]
                    if not len(s_):
                        continue
                    hv = s_.groupby("half").net.mean()
                    reg.append(dict(exit=ex, group=gname, regime=rv, side=sg, N=len(s_), days=s_.day.nunique(), r1_windows=s_.r1_win.nunique(),
                                    avg=s_.net.mean(), WR=(s_.net > 0).mean() * 100, t=EV.cl_t(s_.net.values, s_.day.values),
                                    halves_pos=f"{int((hv > 0).sum())}/{len(hv)}"))
    RG = pd.DataFrame(reg); RG.to_csv(OUT + "_flag_regime.csv", index=False)
    # ── hindsight bound vs reachable
    hb = []
    for ex in ("Q3", "Q5", "TR"):
        g = F[ex + "_g"]; gh = F["HB_" + ex + "_g"]
        for b in ("<10", "10-20", ">20", "ALL"):
            m = (F.mv_bucket == b) if b != "ALL" else np.ones(len(F), bool)
            hb.append(dict(exit=ex, moved=b, N=int(m.sum()), reach_gross=g[m].mean(), hindsight_gross=gh[m].mean()))
    HB = pd.DataFrame(hb)
    # ── FAST
    S = pickle.load(open(os.path.join(D, "fast_rows.pkl"), "rb"))
    S = tag(S, T, A)
    fg = []
    for det in ("D3", "D5"):
        for bf in (None, True, False):
            m = (S.det == det).values & (np.ones(len(S), bool) if bf is None else (S.became_frenzy == bf).values)
            fg.append((f"{det}" + ("" if bf is None else (" became Frenzy (hindsight)" if bf else " no Frenzy (hindsight)")), m))
    FC = cells(S, EXS, fg, "FAST")
    FC.to_csv(OUT + "_fast_cells.csv", index=False)
    # paired: same episode, FAST D3 entry vs FLAG entry (became_frenzy rows)
    P = S[(S.det == "D3") & S.became_frenzy].merge(F[["pair", "spike_ms", "entry_ms", "moved", "TR_g", "Q3_g", "TR_k"] +
                                                     [e + "_f" for e in ("TR", "Q3")] + [e + "_complete" for e in ("TR", "Q3")]],
                                                    on=["pair", "spike_ms"], suffixes=("_fast", "_flag"))
    P["lead_min"] = (P.entry_ms_flag - P.entry_ms_fast) / 60000
    pickle.dump(dict(C=C, RG=RG, HB=HB, FC=FC, P=P, lag=q, lag_by=lag_by, F=F.drop(columns=["path"]), paths=F[["path", "TR_g", "R1", "mv_bucket"]]),
                open(os.path.join(D, "flag_eval.pkl"), "wb"))
    pd.set_option("display.width", 250)
    print(q.round(2).to_string()); print(lag_by.round(1).to_string())
    a = C[C.block == "ALL"]
    print(a[["group", "exit", "N", "days", "WR", "avg", "med", "ci_lo", "ci_hi", "slip20", "best5_out", "kelly_floor", "halves_pos"]].round(2).to_string())
    print(C.pivot_table(index=["group", "exit"], columns="block", values="avg").round(2).to_string())
    print(HB.round(2).to_string())
    print(RG[RG.exit == "TR"].round(2).to_string())
    fa = FC[FC.block == "ALL"]
    print(fa[["group", "exit", "N", "days", "WR", "avg", "med", "ci_lo", "ci_hi", "slip20", "best5_out", "halves_pos"]].round(2).to_string())
    print("paired fast-vs-flag:", len(P), "median lead min", P.lead_min.median(), "fast TR gross", P.TR_g_fast.mean() if "TR_g_fast" in P else None,
          "flag TR gross", P.TR_g_flag.mean() if "TR_g_flag" in P else None)


if __name__ == "__main__":
    main()
