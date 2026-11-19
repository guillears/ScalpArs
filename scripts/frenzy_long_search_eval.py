#!/usr/bin/env python3
"""FRENZY LONG SEARCH — EVAL (PREREG reports/FRENZY_LONG_SEARCH_2026-11-19_PREREG.md).
Reads reports/FRENZY_LONG_SEARCH_data/rows/*.pkl (build) + tape_daily.csv (tape) and writes
reports/FRENZY_LONG_SEARCH_2026-11-19_cells.csv (every family × checkpoint × exit × block), _regime.csv (sign splits, window units),
_primary_trades.csv.gz (trades of the primary cells with path labels) and FRENZY_LONG_SEARCH_data/eval.pkl (tables for the report)."""
import glob, os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data")
OUT = os.path.join(R, "reports", "FRENZY_LONG_SEARCH_2026-11-19")
DAY = 86_400_000
HOLDS = (24, 72, 168); STOPS = ("x", "20", "10"); TPS = ("x", "20", "10", "5")
EXITS = [f"H{h}_S{s}_T{t}" for h in HOLDS for s in STOPS for t in TPS]
PRIMARY = {"SURVIVOR": "H72_S20_Tx", "CARRY": "H72_Sx_Tx", "SURV_CARRY": "H72_S20_Tx"}
CKS = {"SURVIVOR": (24, 48, 72), "CARRY": (6, 24, 48), "SURV_CARRY": (24, 48)}
RNG = np.random.default_rng(20261119)


def load():
    fs = sorted(glob.glob(os.path.join(D, "rows", "*.pkl")))
    df = pd.concat([pickle.load(open(f, "rb")) for f in fs], ignore_index=True)
    df["day"] = df.entry_ms // DAY
    dt = pd.to_datetime(df.entry_ms, unit="ms")
    df["month"] = dt.dt.strftime("%Y-%m")
    df["half"] = dt.dt.year.astype(str).str[2:] + np.where(dt.dt.month <= 6, "-H1", "-H2")
    df["block"] = np.where(df.entry_ms < pd.Timestamp("2025-01-01").value // 10**6, "HOLD-A",
                           np.where(df.entry_ms < pd.Timestamp("2026-07-01").value // 10**6, "DESIGN", "HOLD-B"))
    T = pd.read_csv(os.path.join(D, "tape_daily.csv"), index_col=0)
    for k in ("R1", "R6", "R5", "RF", "btc_gap_prev", "breadth_prev", "spikes7_prev", "btc_r1_prev", "btc_r7_prev"):
        df[k] = df.day.map(T[k])
    A = pd.read_csv(OUT + "_analogs.csv", index_col=0)
    df["analog"] = df.day.isin(set(A.index[A.analog])).astype(int)
    # R1 window id (runs of the R1 calendar)
    r1 = T["R1"].fillna(-1)
    wid = (r1 != r1.shift()).cumsum()
    df["r1_win"] = df.day.map(wid)
    # families
    df["SURVIVOR"] = (df.gain >= 20) & (df.vs_vwap >= 0) & (df.off_high >= -15)
    df["CARRY"] = df.fund8 <= -0.10
    df["SURV_CARRY"] = df.SURVIVOR & df.CARRY
    df["BASE"] = True
    return df, T


def net(d, ex, slip=0.10):
    g = d[ex + "_g"].values / 100; k = d[ex + "_k"].values; f = d[ex + "_f"].values
    s = slip / 100
    out_px = np.where(k == 2, 1 + g, (1 + g) * (1 - s))
    fee_out = np.where(k == 2, 0.02, 0.05)
    return ((out_px / (1 + s)) - 1) * 100 - 0.05 - fee_out - f


def dedupe(d, ex):
    """one open position per pair inside a cell: skip an entry while the pair's previous trade is still open"""
    d = d.sort_values(["pair", "entry_ms"])
    keep = np.ones(len(d), bool)
    pairs = d.pair.values; ent = d.entry_ms.values; xs = d[ex + "_x"].values
    last_p, last_x = None, -1
    for i in range(len(d)):
        if pairs[i] != last_p:
            last_p, last_x = pairs[i], xs[i]; continue
        if ent[i] < last_x:
            keep[i] = False
        else:
            last_x = xs[i]
    return d[keep]


def cl_t(x, days):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan
    m = x.mean()
    s = pd.Series(x - m).groupby(np.asarray(days)).sum().values
    se = np.sqrt((s ** 2).sum()) / len(x)
    return m / se if se > 0 else np.nan


def day_ci(x, days, B=600):
    x = np.asarray(x, float); days = np.asarray(days)
    if len(x) < 5:
        return (np.nan, np.nan)
    u, inv = np.unique(days, return_inverse=True)
    sums = np.bincount(inv, x); cnts = np.bincount(inv)
    bs = []
    for _ in range(B):
        k = RNG.integers(0, len(u), len(u))
        bs.append(sums[k].sum() / cnts[k].sum())
    return tuple(np.percentile(bs, [2.5, 97.5]))


def wilson_lo(p, n, z=1.645):
    if n == 0:
        return np.nan
    den = 1 + z * z / n
    c = p + z * z / (2 * n); r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / den


def stats(x, d, full=True):
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return dict(N=0)
    w = x[x > 0]; l = x[x <= 0]
    wr = len(w) / n
    aw = w.mean() if len(w) else 0; al = -l.mean() if len(l) else 0
    be = al / (aw + al) if aw + al > 0 else np.nan
    nd = d.day.nunique()
    o = dict(N=n, days=nd, pairs=d.pair.nunique(), WR=wr * 100, avg=x.mean(), med=np.median(x), avg_win=aw, avg_loss=-al,
             be_wr=be * 100 if be == be else np.nan, t=cl_t(x, d.day.values))
    if full:
        lo, hi = day_ci(x, d.day.values); o["ci_lo"], o["ci_hi"] = lo, hi
        p_lo = wilson_lo(wr, nd)
        b = aw / al if al > 0 else np.nan
        o["wr_floor"] = p_lo * 100
        o["kelly_floor"] = p_lo - (1 - p_lo) / b if b and b == b else np.nan
        k5 = int(np.ceil(n * 0.05))
        o["best5_out"] = np.sort(x)[:n - k5].mean() if n > k5 else np.nan
        tot = x.sum()
        ds = pd.Series(x).groupby(d.day.values).sum(); ps = pd.Series(x).groupby(d.pair.values).sum()
        o["top_day_share"] = ds.max() / tot * 100 if tot > 0 else np.nan
        o["top_pair_share"] = ps.max() / tot * 100 if tot > 0 else np.nan
        o["lopo_min"] = min((tot - v) / (n - c) for v, c in zip(ps.values, pd.Series(1, index=d.pair.values).groupby(level=0).sum().reindex(ps.index).values) if n - c > 0)
        ms = pd.Series(x).groupby(d.month.values)
        o["lomo_min"] = min((tot - g.sum()) / (n - len(g)) for _, g in ms if n - len(g) > 0)
    return o


def main():
    df, T = load()
    ok = {ex: (df[ex + "_complete"]) for ex in EXITS}
    rows = []
    store = {}
    for fam in ("SURVIVOR", "CARRY", "SURV_CARRY", "BASE"):
        cks = CKS.get(fam, (6, 24, 48, 72))
        for ck in cks:
            for ex in EXITS:
                d = df[(df.ck == ck) & df[fam] & ok[ex]]
                d = dedupe(d, ex)
                x = net(d, ex)
                prim = PRIMARY.get(fam) == ex
                for blk, sel in [("ALL", np.ones(len(d), bool))] + [(b, (d.block == b).values) for b in ("DESIGN", "HOLD-A", "HOLD-B")]:
                    s = stats(x[sel], d[sel], full=prim or blk == "ALL")
                    s.update(fam=fam, ck=ck, exit=ex, block=blk, primary=prim)
                    if blk == "ALL":
                        s["slip20"] = net(d, ex, 0.20).mean() if len(d) else np.nan
                        s["slip50"] = net(d, ex, 0.50).mean() if len(d) else np.nan
                        hv = pd.Series(x).groupby(d.half.values).mean()
                        for hk, hvv in hv.items():
                            s["h_" + hk] = hvv
                        s["halves_pos"] = f"{int((hv > 0).sum())}/{len(hv)}"
                        s["fund_mean"] = d[ex + "_f"].mean() if len(d) else np.nan
                    rows.append(s)
                if prim or (fam == "BASE" and ex in PRIMARY.values()):
                    store[(fam, ck, ex)] = (d.assign(net=x, net20=net(d, ex, 0.20), net50=net(d, ex, 0.50)))
    C = pd.DataFrame(rows)
    C.to_csv(OUT + "_cells.csv", index=False)

    # ── family-wise null: permute the condition inside month among BASE entries of the same checkpoint (DESIGN)
    prim_cells = [(f, ck) for f in ("SURVIVOR", "CARRY", "SURV_CARRY") for ck in CKS[f]]

    def max_t(cond_map):
        best = -9
        for f, ck in prim_cells:
            ex = PRIMARY[f]
            d = cond_map[(f, ck)]
            d = d[d.block == "DESIGN"]
            if len(d) < 20:
                continue
            d = dedupe(d, ex)
            best = max(best, cl_t(net(d, ex), d.day.values))
        return best
    base = {}
    for f, ck in prim_cells:
        ex = PRIMARY[f]
        b = df[(df.ck == ck) & ok[ex]].copy()
        base[(f, ck)] = b
    obs = max_t({k: v[v[k[0]]] for k, v in base.items()})
    null = []
    for it in range(200):
        cm = {}
        for (f, ck), b in base.items():
            perm = b.groupby("month")[f].transform(lambda s: pd.Series(RNG.permutation(s.values), index=s.index))
            cm[(f, ck)] = b[perm.astype(bool)]
        null.append(max_t(cm))
    null = np.array(null)
    p_fw = float((null >= obs).mean())
    print(f"family-wise: observed max t {obs:.2f} · null p95 {np.percentile(null, 95):.2f} · p {p_fw:.3f}")

    # ── regime splits (sign granularity, window units) on primary cells + BASE at the same exit
    reg = []
    for (fam, ck, ex), d in store.items():
        for rv in ("R1", "R6", "R5", "RF", "analog"):
            for sgn in (1, 0):
                s_ = d[d[rv] == sgn]
                if len(s_) == 0:
                    continue
                hv = s_.groupby("half").net.mean()
                reg.append(dict(fam=fam, ck=ck, exit=ex, regime=rv, side=sgn, N=len(s_), days=s_.day.nunique(),
                                r1_windows=s_.r1_win.nunique(), avg=s_.net.mean(), WR=(s_.net > 0).mean() * 100,
                                t=cl_t(s_.net.values, s_.day.values), halves_pos=f"{int((hv > 0).sum())}/{len(hv)}",
                                halves=" ".join(f"{k}:{v:+.2f}({(s_.half == k).sum()})" for k, v in hv.items())))
    RG = pd.DataFrame(reg); RG.to_csv(OUT + "_regime.csv", index=False)

    # ── path labels on the primary trades
    P = []
    for (fam, ck, ex), d in store.items():
        if fam == "BASE":
            continue
        d = d.copy()
        stop = 20.0 if "_S20" in ex else 10.0
        d["WORKED"] = d["reach_s20" if stop == 20 else "reach_s10"] >= 1.0
        d["REACH"] = d["reach_s20" if stop == 20 else "reach_s10"]
        k = d[ex + "_k"].values
        hz = int(ex.split("_")[0][1:])
        path_end = np.array([p[24 + hz] if 24 + hz < len(p) else np.nan for p in d.path.values])
        lab = np.where(d.net > 0, np.where(d.REACH >= 2 * d.net + 5, "LEFT_ON_TABLE", "CAPTURED"),
                       np.where((k == 1) & (path_end > 0), "STOPPED_RECOVERED",
                                np.where(d.REACH >= 5, "GAVE_BACK", np.where(d.WORKED, "DRIFT_LOSS", "NEVER_WORKED"))))
        d["label"] = lab
        d["fam"] = fam
        P.append(d)
    P = pd.concat(P, ignore_index=True)
    keep_cols = [c for c in P.columns if not c.startswith("H") and c != "path"]
    P[keep_cols].to_csv(OUT + "_primary_trades.csv.gz", index=False)
    pickle.dump(dict(C=C, RG=RG, null=null, obs=obs, p_fw=p_fw, store={k: v for k, v in store.items()}, P=P),
                open(os.path.join(D, "eval.pkl"), "wb"))
    pd.set_option("display.width", 250)
    pc = C[C.primary & (C.block == "ALL")]
    print(pc[["fam", "ck", "exit", "N", "days", "WR", "avg", "ci_lo", "ci_hi", "slip20", "best5_out", "kelly_floor", "halves_pos", "fund_mean"]].round(2).to_string())
    print(C[C.primary].pivot_table(index=["fam", "ck"], columns="block", values="avg").round(2))


if __name__ == "__main__":
    main()
