#!/usr/bin/env python3
"""HT_WILLY 2026-11-20 (PREREG reports/HT_WILLY_2026-11-20_PREREG.md) — EVAL of ht_willy_sim / ht_willy_real outputs. Research only.
Writes reports/HT_WILLY_2026-11-20_*.csv / .json and prints every table the write-up uses.
Usage: venv/bin/python -I scripts/ht_willy_eval.py"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "best_case"))
import ht_willy_sim as W  # noqa: E402
import one_size as OS  # noqa: E402
import path_labels as PL  # noqa: E402
RP = W.RP; D = W.D; OUT = os.path.join(RP, "HT_WILLY_2026-11-20")
X1 = 0.61; USD = 6100.0
MIN = 60_000
RNG = np.random.default_rng(11)
SUM = {}


def yr_block(ts):
    return np.where(pd.to_datetime(np.asarray(ts, "int64"), unit="ms").year == 2025, "DESIGN", "HOLDOUT")


def half(ts):
    t = pd.to_datetime(np.asarray(ts, "int64"), unit="ms")
    return np.array([f"{y % 100}-H{1 if m <= 6 else 2}" for y, m in zip(t.year, t.month)])


def day(ts):
    return pd.to_datetime(np.asarray(ts, "int64"), unit="ms").strftime("%Y-%m-%d").values


def deoverlap_x(ts, pair, xs):
    keep = np.zeros(len(ts), bool); last = {}
    for i in np.argsort(ts, kind="stable"):
        if ts[i] >= last.get(pair[i], -1):
            keep[i] = True; last[pair[i]] = xs[i]
    return keep


def boot_ci(d, days, n=5000, q=(2.5, 97.5)):
    d = np.asarray(d, float); days = np.asarray(days)
    u, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, d, len(u)); c = np.bincount(inv, None, len(u))
    idx = RNG.integers(0, len(u), (n, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return np.percentile(m, q)


def dcr(nets, xms, t0, t1):
    nets = np.asarray(nets, float)
    if not len(nets):
        return np.nan
    ca = pd.to_datetime(np.asarray(xms, "int64"), unit="ms")
    return OS.dcr(pd.Series(ca), OS.frac(nets, X1), np.ones(len(nets), bool), pd.Timestamp(int(t0), unit="ms"), pd.Timestamp(int(t1), unit="ms"))


def stats(nets):
    nets = np.asarray(nets, float)
    return dict(N=len(nets), WR=round(100 * (nets > 0).mean(), 1) if len(nets) else np.nan, avg=round(nets.mean(), 3) if len(nets) else np.nan,
                usd=round(nets.sum() * USD / 100, 0))


def fmt(d):
    return " · ".join(f"{k} {v}" for k, v in d.items())


# ───────────────────────────── cohorts ─────────────────────────────
def cohorts():
    H = pd.read_pickle(os.path.join(D, "sim_HT.pkl")); Wl = pd.read_pickle(os.path.join(D, "sim_WILLY.pkl"))
    for Z in (H, Wl):
        Z["blk"] = yr_block(Z.ts); Z["half"] = half(Z.ts); Z["day"] = day(Z.ts)
    H["dipok"] = H.dip >= W.DIP_MIN
    Hd = H[H.dipok].copy()
    Hd["keep"] = deoverlap_x(Hd.ts.values, Hd.pair.values, Hd.ref_x.values)
    Hall = H.copy(); Hall["keep"] = deoverlap_x(Hall.ts.values, Hall.pair.values, Hall.ref_x.values)
    Wl["keep"] = deoverlap_x(Wl.ts.values, Wl.pair.values, Wl.ref_x.values)
    C = dict(HT=Hd[Hd.keep].reset_index(drop=True), HT_nodip=Hall[Hall.keep].reset_index(drop=True), WILLY=Wl[Wl.keep].reset_index(drop=True))
    print("== COHORTS")
    print(f"HT baseline {len(H)} → dip ≥ 1.4: {int(H.dipok.sum())} → one per coin per holding period (REF exit): {len(C['HT'])} "
          f"| episodes {C['HT'].groupby(['pair', 'spike']).ngroups} · days {C['HT'].day.nunique()} · pairs {C['HT'].pair.nunique()}")
    print(f"HT no-dip sensitivity: {len(C['HT_nodip'])}")
    print(f"WILLY baseline {len(Wl)} → de-overlapped (90 min) {len(C['WILLY'])} | episodes {C['WILLY'].groupby(['pair', 'spike']).ngroups} · days {C['WILLY'].day.nunique()}")
    SUM["cohorts"] = {k: len(v) for k, v in C.items()}
    return C, H


def ref_table(C):
    print("\n== REF (today's exit) per block")
    rows = []
    for sl in ("WILLY", "HT", "HT_nodip"):
        Z = C[sl]
        for b, g in [("ALL", Z)] + list(Z.groupby("blk")) + list(Z.groupby("half")):
            if not len(g):
                continue
            s = stats(g.ref_net); s["DCR"] = round(dcr(g.ref_net, g.ref_x, g.ts.min(), g.ref_x.max()), 3)
            s["days"] = g.day.nunique(); rows.append(dict(sleeve=sl, block=b, **s))
            print(sl, b, fmt(s))
    pd.DataFrame(rows).to_csv(OUT + "_ref.csv", index=False)


# ───────────────────────────── decomposition ─────────────────────────────
def decomp(Z, sl):
    fn = getattr(PL.label_from_bars, "__wrapped__", PL.label_from_bars)
    res = []
    for _, r in Z.iterrows():
        bars = pd.DataFrame(dict(open_time=r.pb_t, h=r.pb_h.astype(float), l=r.pb_l.astype(float)))
        # backtest entries fill at the bar OPEN, so bar 0 is after the fill: bound it by its own extremes (stamped) — AMBIG only if it holds both lines
        pk = (float(r.pb_h[0]) / float(r.E) - 1) * 100 - PL.FEE_RT; tr = (float(r.pb_l[0]) / float(r.E) - 1) * 100 - PL.FEE_RT
        f = fn(bars, int(r.ts), int(r.ref_x) - MIN, float(r.E), "LONG", float(r.ref_net), 1.0, 4.0, now_ms=2_000_000_000_000,
               stopped=(r.ref_kind == "SL"), peak=pk, trough=tr)
        g = PL.finish(f, float(r.ref_net)); res.append(dict(f, **g))
    R = pd.DataFrame(res); R["net"] = Z.ref_net.values; R["blk"] = Z.blk.values
    print(f"\n== PATH DECOMPOSITION {sl} (REF exit; X = +1 %, Y = 4 %)")
    rows = []
    for b, g in [("ALL", R)] + list(R.groupby("blk")):
        t = g.groupby("group").net.agg(["size", "sum"]); t["usd"] = (t["sum"] * USD / 100).round(0)
        for k, v in t.iterrows():
            rows.append(dict(sleeve=sl, block=b, group=k, N=int(v["size"]), sum_pct=round(v["sum"], 1), usd=v["usd"]))
        print(b, {k: (int(v["size"]), int(v["usd"])) for k, v in t.iterrows()})
    lab = R.label.value_counts().to_dict()
    print("entry label", lab, "| reach median", R.reach.median())
    return pd.DataFrame(rows), R


# ───────────────────────────── part 1: grid ─────────────────────────────
def grid(C, REAL):
    Z = C["HT"]
    G = np.vstack(Z.g_net.values).astype(float)
    ref = Z.ref_net.values; ok = np.isfinite(G).all(1)
    print(f"\n== HT EXIT GRID — {ok.sum()} of {len(Z)} fills priced on every cell")
    Z = Z[ok].reset_index(drop=True); G = G[ok]; ref = ref[ok]
    Dl = G - ref[:, None]
    ci_ref = W.CELLS.index(W.REF_CELL)
    assert np.allclose(Dl[:, ci_ref], 0, atol=1e-6)
    des = (Z.blk == "DESIGN").values; hol = ~des
    rows = []
    for ci, cl in enumerate(W.CELLS):
        r = dict(cell=cl)
        for b, m in (("DESIGN", des), ("HOLDOUT", hol), ("ALL", np.ones(len(Z), bool))):
            r[f"{b}_N"] = int(m.sum()); r[f"{b}_avg"] = round(G[m, ci].mean(), 3); r[f"{b}_d"] = round(Dl[m, ci].mean(), 3)
            r[f"{b}_chg"] = int((np.abs(Dl[m, ci]) > 1e-3).sum())
        rows.append(r)
    T = pd.DataFrame(rows)
    # family-wise day sign-flip null on DESIGN (40 non-REF cells)
    cells = [c for c in W.CELLS if c != W.REF_CELL]; cix = [W.CELLS.index(c) for c in cells]
    dd = Z.day.values[des]; u, inv = np.unique(dd, return_inverse=True)
    DS = np.vstack([np.bincount(inv, Dl[des, ci], len(u)) for ci in cix]).T   # days × cells
    def tstat(M):
        return M.mean(0) / (M.std(0, ddof=1) / np.sqrt(len(M)) + 1e-12)
    t_obs = tstat(DS)
    mx = []
    for _ in range(2000):
        s = RNG.choice([-1.0, 1.0], len(u))[:, None]
        mx.append(np.nanmax(tstat(DS * s)))
    null95 = float(np.percentile(mx, 95))
    T["DESIGN_t"] = np.nan
    for c, tv in zip(cells, t_obs):
        T.loc[T.cell == c, "DESIGN_t"] = round(tv, 2)
    elig = T[(T.cell != W.REF_CELL) & (T.DESIGN_chg >= 30)]
    pick = elig.sort_values("DESIGN_d", ascending=False).iloc[0] if (elig.DESIGN_d > 0).any() else None
    T.to_csv(OUT + "_grid.csv", index=False)
    print(T.sort_values("DESIGN_d", ascending=False).head(12).to_string(index=False))
    print("null 95th (max t over 40 cells, DESIGN days):", round(null95, 2))
    out = dict(null95=null95, pick=None)
    if pick is None:
        print("NO PICK: no eligible cell with DESIGN Δ > 0"); SUM["grid"] = out; return out, T, Z, G
    cl = pick.cell; ci = W.CELLS.index(cl)
    d = Dl[:, ci]
    ci_h = boot_ci(d[hol], Z.day.values[hol]); ci_d = boot_ci(d[des], Z.day.values[des])
    halves = {h: round(d[(Z.half == h).values].mean(), 3) for h in sorted(Z.half.unique())}
    ps = pd.Series(d).groupby(Z.pair.values).sum()
    top = ps.sort_values(ascending=False)
    top_share = float(top.iloc[0] / d.sum()) if d.sum() > 0 else np.nan
    loo = min((d[Z.pair.values != p].mean() for p in ps.index))
    out.update(pick=cl, design_d=float(d[des].mean()), design_ci=list(ci_d), design_t=float(pick.DESIGN_t), holdout_d=float(d[hol].mean()),
               holdout_ci=list(ci_h), halves=halves, top_pair=top.index[0], top_share=top_share, loo_min=float(loo))
    print(f"PICK {cl}: DESIGN Δ {d[des].mean():+.3f} {np.round(ci_d, 3)} t {pick.DESIGN_t} vs null {null95:.2f} | HOLDOUT Δ {d[hol].mean():+.3f} "
          f"CI {np.round(ci_h, 3)} | halves {halves} | top pair {top.index[0]} {top_share:.0%} | LOO min {loo:+.3f}")
    SUM["grid"] = out
    return out, T, Z, G


def stop_split(Z, G):
    ref = Z.ref_net.values; stopped = (Z.ref_kind == "SL").values
    print("\n== LIVE-STOPPED SPLIT (stop-only cells, TP 5 / 24 h)")
    rows = []
    for s in (3, 5, 6):
        ci = W.CELLS.index(f"T5_S{s}_H24"); d = G[:, ci] - ref
        if s > 4:
            bad = np.abs(d[~stopped]) > 1e-6
            assert not bad.any(), f"wider stop changed {bad.sum()} REF-not-stopped fills"
        sv = stopped & (G[:, ci] > ref + 1e-9); dp = stopped & (G[:, ci] < ref - 1e-9)
        new = (~stopped) & (np.abs(d) > 1e-9)
        for b in ("DESIGN", "HOLDOUT"):
            m = (Z.blk == b).values
            r = dict(cell=f"T5_S{s}_H24", block=b, N=int(m.sum()), stopped=int((stopped & m).sum()),
                     saved=int((sv & m).sum()), saved_sum=round(d[sv & m].sum(), 1), deeper=int((dp & m).sum()), deeper_sum=round(d[dp & m].sum(), 1),
                     new_stops=int((new & m).sum()), new_sum=round(d[new & m].sum(), 1), d_avg=round(d[m].mean(), 3))
            r["CI"] = list(np.round(boot_ci(d[m], Z.day.values[m]), 3)); rows.append(r); print(r)
    pd.DataFrame(rows).to_csv(OUT + "_stopsplit.csv", index=False)


def own_deoverlap(H):
    """sensitivity: each cell de-overlapped under its own exit (dip-kept rows)."""
    Hd = H[H.dipok].reset_index(drop=True)
    G = np.vstack(Hd.g_net.values).astype(float); X = np.vstack(Hd.g_x.values)
    rows = []
    for ci, cl in enumerate(W.CELLS):
        ok = np.isfinite(G[:, ci])
        k = deoverlap_x(Hd.ts.values[ok], Hd.pair.values[ok], X[ok, ci])
        g = Hd[ok][k]; v = G[ok, ci][k]
        r = dict(cell=cl)
        for b in ("DESIGN", "HOLDOUT"):
            m = (g.blk == b).values; r[f"{b}_N"] = int(m.sum()); r[f"{b}_avg"] = round(v[m].mean(), 3)
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(OUT + "_grid_owndeov.csv", index=False)
    return T


def before_after(Z, G, cells, label):
    rows = []
    ref = Z.ref_net.values
    for cl in cells:
        ci = W.CELLS.index(cl); a = G[:, ci]; xa = np.vstack(Z.g_x.values)[:, ci]
        for b in ("DESIGN", "HOLDOUT"):
            m = (Z.blk == b).values
            t0, t1 = Z.ts[m].min(), max(Z.ref_x[m].max(), xa[m].max())
            sb, sa = stats(ref[m]), stats(a[m]); chg = np.abs(a[m] - ref[m]) > 1e-3
            rows.append(dict(set=label, cell=cl, block=b, N=sb["N"], WR=f"{sb['WR']} → {sa['WR']}", avg=f"{sb['avg']:+.3f} → {sa['avg']:+.3f}",
                             d=round(a[m].mean() - ref[m].mean(), 3), CI=list(np.round(boot_ci(a[m] - ref[m], Z.day.values[m]), 3)),
                             usd=f"{sb['usd']:+,.0f} → {sa['usd']:+,.0f}",
                             DCR=f"{dcr(ref[m], Z.ref_x[m], t0, t1):+.3f} → {dcr(a[m], xa[m], t0, t1):+.3f}",
                             changed=f"{int(chg.sum())} ({int((chg & (ref[m] <= 0)).sum())} saved-side / {int((chg & (ref[m] > 0)).sum())} cut-side)"))
    return pd.DataFrame(rows)


def real_grid(REAL, cells):
    R = REAL[(REAL.sleeve == "HT")].copy()
    R["post587"] = pd.to_datetime(R.opened_at) >= pd.Timestamp("2026-11-18T22:56")   # DL 587 commit 2026-11-18 19:56 −03 = 22:56 UTC
    print("\n== REAL HT FILLS (current batch): replica parity")
    par = []
    for _, r in R.iterrows():
        kind_b = "TP" if "TP" in str(r.booked_reason) else ("SL" if "STOP" in str(r.booked_reason) else "OTHER")
        dm = abs((int(r.ref_x) - MIN) - int(r.booked_x)) / MIN
        par.append(dict(pair=r.pair, opened=r.opened_at, post587=r.post587, booked=round(r.booked, 3), booked_kind=kind_b, ref=round(r.ref_net, 3),
                        ref_kind=r.ref_kind, kind_ok=kind_b == r.ref_kind, exit_min_diff=round(dm, 1)))
    P = pd.DataFrame(par); print(P.to_string(index=False))
    Rp = R[R.post587]
    G = np.vstack(Rp.g_net.values).astype(float); ref = Rp.ref_net.values
    rows = []
    for cl in cells:
        ci = W.CELLS.index(cl); a = G[:, ci]; xa = np.vstack(Rp.g_x.values)[:, ci]
        chg = np.abs(a - ref) > 1e-3
        t0 = Rp.ts.min(); t1 = max(Rp.ref_x.max(), xa.max())
        rows.append(dict(set="CURRENT HT (post-587)", cell=cl, N=len(Rp), WR=f"{100 * (ref > 0).mean():.0f} → {100 * (a > 0).mean():.0f}",
                         avg=f"{ref.mean():+.3f} → {a.mean():+.3f}", usd=f"{ref.sum() * 61:+,.0f} → {a.sum() * 61:+,.0f}",
                         DCR=f"{dcr(ref, Rp.ref_x, t0, t1):+.3f} → {dcr(a, xa, t0, t1):+.3f}",
                         changed=f"{int(chg.sum())} ({int((chg & (a > ref + 1e-3)).sum())} better / {int((chg & (a < ref - 1e-3)).sum())} worse)",
                         open_marked=int(sum(k == "OPEN" for k in Rp.g_kind.map(lambda v: v[ci])))))
    T = pd.DataFrame(rows); print(T.to_string(index=False))
    return P, T


# ───────────────────────────── part 2: random entries ─────────────────────────────
def rnd_rows(Z, sl, ctrls, label):
    rows = []
    for cn in ctrls:
        col = cn + "_net"
        if col not in Z:
            continue
        ok = Z[col].map(len) >= 10
        g = Z[ok]
        rm = g[col].map(lambda v: float(np.mean(v))).values
        rk = np.array([float((np.asarray(v) < x).mean() + 0.5 * (np.asarray(v) == x).mean()) for v, x in zip(g[col], g.ref_net)])
        d = g.ref_net.values - rm
        for b, m in [("ALL", np.ones(len(g), bool))] + [(x, (g.blk == x).values) for x in ("DESIGN", "HOLDOUT")] + \
                    [(h, (g.half == h).values) for h in sorted(g.half.unique())]:
            if m.sum() < 3:
                continue
            ci = boot_ci(d[m], g.day.values[m])
            rows.append(dict(set=label, sleeve=sl, control=cn, block=b, N=int(m.sum()), days=int(pd.Series(g.day.values[m]).nunique()),
                             real=round(g.ref_net.values[m].mean(), 3), random=round(rm[m].mean(), 3), d=round(d[m].mean(), 3),
                             lo=round(ci[0], 3), hi=round(ci[1], 3), rank=round(rk[m].mean(), 3),
                             real_WR=round(100 * (g.ref_net.values[m] > 0).mean(), 1),
                             rnd_WR=round(100 * np.mean([np.mean(np.asarray(v) > 0) for v in g[col].values[m]]), 1)))
    return rows


def old12_split(Z, sl):
    pre, post = [], []
    for _, r in Z.iterrows():
        f = r.C_OLD12_f; v = r.C_OLD12_net
        if not len(v):
            continue
        s = f[:, 1] < 0
        pre += list(v[s]); post += list(v[~s])
    pre, post = np.array(pre), np.array(post)
    out = dict(sleeve=sl, pre_N=len(pre), pre_avg=round(pre.mean(), 3), post_N=len(post), post_avg=round(post.mean(), 3),
               all_avg=round(np.r_[pre, post].mean(), 3), real=round(Z.ref_net.mean(), 3))
    print("C_OLD12 split", out)
    return out


def part2(C, REAL):
    print("\n== PART 2: REAL vs RANDOM (d = real − mean of 50 random, same exit)")
    rows = []
    rows += rnd_rows(C["WILLY"], "WILLY", ("C_LOC", "C_EP", "C_OLD12"), "BACKTEST")
    rows += rnd_rows(C["HT"], "HT", ("C_LOC", "C_EP", "C_EPD", "C_OLD12"), "BACKTEST")
    rows += rnd_rows(C["HT_nodip"], "HT_nodip", ("C_LOC", "C_EP", "C_EPD"), "BACKTEST")
    Rw = REAL[REAL.sleeve == "WILLY"].copy(); Rh = REAL[(REAL.sleeve == "HT") & (pd.to_datetime(REAL.opened_at) >= pd.Timestamp("2026-11-18T22:56"))].copy()
    for Z in (Rw, Rh):
        Z["blk"] = "REAL"; Z["half"] = "REAL"; Z["day"] = day(Z.ts)
    rows += rnd_rows(Rw, "WILLY", ("C_LOC", "C_EP"), "MASTER B19-B27")
    rows += rnd_rows(Rh, "HT", ("C_LOC", "C_EP", "C_EPD"), "CURRENT (post-587)")
    T = pd.DataFrame(rows)
    T = T[~((T.set != "BACKTEST") & (T.block != "ALL"))]
    T.to_csv(OUT + "_random.csv", index=False)
    with pd.option_context("display.width", 250):
        print(T.to_string(index=False))
    sp = [old12_split(C["WILLY"], "WILLY"), old12_split(C["HT"], "HT")]
    SUM["old12"] = sp
    return T


def tmap(Z, sl, design_only=True):
    """EXPLORATORY: C_EP draws (DESIGN only) by timing features; real fills beside."""
    g = Z[Z.blk == "DESIGN"] if design_only else Z
    rows = []
    for _, r in g.iterrows():
        f = r.C_EP_f; v = r.C_EP_net
        for k in range(len(v)):
            rows.append(dict(src=r.row, since=f[k, 1], below_run=f[k, 2], below_H=f[k, 3], ret_sp=f[k, 4], net=v[k], kind="RND"))
        rows.append(dict(src=r.row, since=r.f_since, below_run=r.f_below_run, below_H=r.f_below_H, ret_sp=r.f_ret_sp, net=r.ref_net, kind="REAL"))
    M = pd.DataFrame(rows)
    bins = dict(since=[-1, 15, 30, 60, 120, 240, 1e9], below_run=[-1e9, 1, 2, 4, 7, 12, 1e9], below_H=[-1e9, 0, 1.4, 4, 8, 15, 1e9],
                ret_sp=[-1e9, -5, 0, 5, 10, 20, 1e9])
    out = []
    for v, b in bins.items():
        M[v + "_b"] = pd.cut(M[v], b)
        t = M.groupby([v + "_b", "kind"], observed=True).net.agg(["size", "mean"]).unstack("kind")
        nf = M[M.kind == "RND"].groupby(v + "_b", observed=True).src.nunique()
        for k_, rw in t.iterrows():
            out.append(dict(sleeve=sl, var=v, bucket=str(k_), rnd_N=rw.get(("size", "RND")), rnd_fills=nf.get(k_, 0),
                            rnd_avg=round(rw.get(("mean", "RND"), np.nan), 3), real_N=rw.get(("size", "REAL")), real_avg=round(rw.get(("mean", "REAL"), np.nan), 3)))
    T = pd.DataFrame(out)
    print(f"\n== TIMING MAP (EXPLORATORY, DESIGN C_EP draws) {sl}")
    print(T.to_string(index=False))
    return T, M


def main():
    import exh_exit2_sim  # noqa: F401
    C, H = cohorts()
    REAL = pd.read_pickle(os.path.join(D, "real.pkl"))
    ref_table(C)
    dec = []
    for sl in ("HT", "WILLY"):
        t, R = decomp(C[sl], sl); dec.append(t)
        C[sl]["group"] = R.group.values; C[sl]["path_label"] = R.label.values; C[sl]["reach"] = R.reach.values
    pd.concat(dec).to_csv(OUT + "_decomp.csv", index=False)
    g, T, Zg, G = grid(C, REAL)
    stop_split(Zg, G)
    od = own_deoverlap(H)
    print("\nown-deoverlap sensitivity (top by DESIGN avg):"); print(od.sort_values("DESIGN_avg", ascending=False).head(8).to_string(index=False))
    cells = [W.REF_CELL] + ([g["pick"]] if g["pick"] else []) + ["OLD2", "T5_S6_H24", "T4_S4_H24", "T6_S4_H24", "T5_S4_H12"]
    cells = list(dict.fromkeys(cells))
    BA = before_after(Zg, G, cells[1:], "BACKTEST HT")
    with pd.option_context("display.width", 250):
        print("\n== BEFORE → AFTER (backtest)"); print(BA.to_string(index=False))
    BA.to_csv(OUT + "_ba_backtest.csv", index=False)
    P, RT = real_grid(REAL, cells[1:])
    P.to_csv(OUT + "_real_parity.csv", index=False); RT.to_csv(OUT + "_ba_real.csv", index=False)
    T2 = part2(C, REAL)
    maps = []
    for sl in ("WILLY", "HT"):
        t, M = tmap(C[sl], sl); maps.append(t)
    pd.concat(maps).to_csv(OUT + "_timing_map_DESIGN.csv", index=False)
    for sl in ("WILLY", "HT"):
        C[sl].drop(columns=[c for c in C[sl].columns if c.startswith(("pb_", "cpath", "rnd_paths")) or c.endswith(("_net", "_f")) or c.startswith("g_")]).to_csv(OUT + f"_cohort_{sl}.csv", index=False)
    pd.to_pickle(C, os.path.join(D, "cohorts.pkl"))
    json.dump(SUM, open(OUT + "_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
