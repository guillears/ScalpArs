#!/usr/bin/env python3
"""DIST_ENTRY_TIMING (PREREG reports/DIST_ENTRY_TIMING_2026-11-20_PREREG.md) — SIM + EVAL. READ-ONLY on the bot.
Every entry (T0 / T−1 / T−2 / T+1 / T+2) through dist_exit_sim.run_rows with the REF_B cell only (unchanged engine), net = dist_exit_eval
net_mat, one position per pair = dist_exit_eval.one_pos. Variants V0 · VE1 · VE2 (primary) · P1 · P2 · LAT1 · LAT2 (descriptive).
Outputs reports/DIST_ENTRY_TIMING_data/{sim.pkl, res.pkl, trades.csv.gz} + reports/DIST_ENTRY_TIMING_2026-11-20.png; prints every table.
Usage: venv/bin/python -I scripts/dist_entry_timing_eval.py [--procs 5] [--resim]"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_exit_sim as DS       # noqa: E402
import dist_exit_eval as EV      # noqa: E402
import nospot_sleeve_lib as NL   # noqa: E402
R = DS.R; OUT = os.path.join(R, "reports", "DIST_ENTRY_TIMING_data")
NEWK1 = os.path.join(OUT, "k1m")
M1, DAY = 60_000, 86_400_000
TS = lambda s: pd.Timestamp(s).value // 10**6
DES = (TS("2023-07-01"), TS("2025-07-01")); HOL = (TS("2025-07-01"), TS("2026-11-18")); FULL = (DES[0], HOL[1])
HALVES = EV.HALVES
REF = [c for c in DS.cells() if c["name"] == "REF_B"]
SIZE = EV.SIZE


def load1m(p, days):
    """= dist_exit_sim.load1m + this study's k1m dir (npz with an extra qv key) as the last source."""
    xs = []; ok = set()
    for d in days:
        ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d")
        f = os.path.join(DS.K1[0], p, ds + ".csv")
        if os.path.exists(f) and os.path.getsize(f) > 20:
            x = pd.read_csv(f)
            if len(x):
                xs.append(x[["t", "o", "h", "l", "c"]]); ok.add(d); continue
        for kd in (DS.K1[1], DS.K1[2], NEWK1):
            f = os.path.join(kd, p, ds + ".npz")
            if os.path.exists(f):
                z = np.load(f); xs.append(pd.DataFrame({k: z[k] for k in ("t", "o", "h", "l", "c")})); ok.add(d); break
    if not xs:
        return None, ok
    x = pd.concat(xs).drop_duplicates("t").sort_values("t")
    t = x.t.values.astype("int64"); t = np.where(t > 10**14, t // 1000, t)
    return (t, x.o.values.astype(float), x.h.values.astype(float), x.l.values.astype(float), x.c.values.astype(float)), ok


DS.load1m = load1m


def work(a):
    p, rows = a
    out, _ = DS.run_rows(p, rows, REF, keep_path=None)
    return rows.index.values, out


def simulate(E, procs):
    jobs = [(p, E[E.pair == p]) for p in E.pair.unique()]
    jobs.sort(key=lambda j: -len(j[1]))
    n = len(E); res = dict(r=np.full(n, np.nan), k=np.full(n, -1), xm=np.zeros(n, np.int64), f=np.zeros(n), ok=np.zeros(n, bool),
                           src1m=np.zeros(n, bool))
    with mp.get_context("fork").Pool(procs) as pool:
        for i, (ix, out) in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            for k in ("r", "k", "xm", "f"):
                res[k][ix] = out[k][:, 0]
            res["ok"][ix] = out["ok"]; res["src1m"][ix] = out["src1m"]
            if i % 200 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), flush=True)
    for k, v in res.items():
        E[k] = v
    tk = E.k.values != 0
    E["net"] = E.r - E.slip - np.where(tk, E.slip, 0.0) - 0.05 - np.where(tk, 0.05, 0.02) + E.f
    E.loc[(~E.ok) | (E.k == 9) | (E.k < 0), "net"] = np.nan
    return E


def build_entries(C):
    slip = np.maximum(0.10, NL.slip_model(C.q5.values.astype(float)))
    base = dict(pair=C.pair.values, cid=C.index.values, T=C.entry_ms.values.astype(np.int64), slip=slip, keep=False,
                trig0=C.trig0_pool.values, ep=C.ep_on_ms.values.astype(np.int64))
    parts = []
    for off, at, msk, nm in ((0, C.atr15_pct.values, C.trig0_pool.values, "T0"), (-1, C.atr1.values, C.trig1.values, "Tm1"),
                             (-2, C.atr2.values, C.trig2.values, "Tm2"), (1, C.atr15_pct.values, C.trig0_pool.values, "Tp1"),
                             (2, C.atr15_pct.values, C.trig0_pool.values, "Tp2")):
        d = pd.DataFrame(base)[msk].copy(); d["atrp"] = np.asarray(at, float)[msk]; d["at"] = nm
        d["entry_ms"] = d["T"] + off * M1
        parts.append(d)
    E = pd.concat(parts, ignore_index=True)
    return E


def variant(E, nm):
    """→ the candidate entries of a variant (one per candle), before one position."""
    g = {a: E[E["at"] == a].set_index("cid") for a in ("T0", "Tm1", "Tm2", "Tp1", "Tp2")}
    if nm in ("V0", "LAT1", "LAT2", "P1", "P2"):
        a = dict(V0="T0", LAT1="Tp1", LAT2="Tp2", P1="Tm1", P2="Tm2")[nm]
        return g[a].reset_index()
    order = ["Tm1", "T0"] if nm == "VE1" else ["Tm2", "Tm1", "T0"]
    out = []; seen = set()
    for a in order:
        d = g[a][~g[a].index.isin(seen)]
        out.append(d.reset_index()); seen |= set(d.index)
    return pd.concat(out, ignore_index=True)


def kept(V):
    pair = V.pair.values.astype(str); T = V.entry_ms.values.astype(np.int64)
    m = EV.one_pos(pair, T, V.xm.values.astype(np.int64), np.isfinite(V.net.values))
    return V[m].copy()


def daily(v, span):
    lo = span[0] // DAY; nd = span[1] // DAY - lo
    d = (v.entry_ms.values // DAY - lo).astype(int); ok = (d >= 0) & (d < nd)
    return np.bincount(d[ok], weights=v.net.values[ok], minlength=nd)


def boot_sum(x, nb=4000, block=10, seed=1120):
    rng = np.random.default_rng(seed); nd = len(x); nblk = int(np.ceil(nd / block))
    st = rng.integers(0, nd - block + 1, (nb, nblk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :nd]
    s = x[idx].sum(1)
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def stats(v, span):
    m = (v.entry_ms >= span[0]) & (v.entry_ms < span[1]); x = v[m]
    if not len(x):
        return dict(N=0)
    d, dd, worst = EV.dcr(x.net.values, x.xm.values.astype(np.int64), span)
    return dict(N=len(x), days=x.entry_ms.floordiv(DAY).nunique(), eps=(x.pair + x.ep.astype(str)).nunique(), WR=(x.net > 0).mean() * 100,
                avg=x.net.mean(), sum=x.net.sum(), DCR=d, maxDD=dd)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=5); ap.add_argument("--resim", action="store_true")
    A = ap.parse_args()
    C = pd.read_pickle(os.path.join(OUT, "cand.pkl"))
    fs = os.path.join(OUT, "sim.pkl")
    if A.resim or not os.path.exists(fs):
        E = simulate(build_entries(C), A.procs); E.to_pickle(fs)
    E = pd.read_pickle(fs)
    print("entries", E["at"].value_counts().to_dict(), "sim ok", int(E.ok.sum()), "1m", int(E.src1m.sum()))
    res = {}
    # ---------------- parity V0 vs REF_B
    V0 = kept(variant(E, "V0"))
    TR = pd.read_csv(os.path.join(R, "reports", "DIST_EXIT_2026-11-18_trades.csv.gz")); TR = TR[TR.cell == "REF_B"]
    a = set(zip(V0.pair, V0.entry_ms)); b = set(zip(TR.pair, TR.entry_ms))
    J = V0.merge(TR[["pair", "entry_ms", "net", "kind"]], on=["pair", "entry_ms"], suffixes=("", "_ref"))
    dn = (J.net - J.net_ref).abs()
    par = dict(V0=len(V0), REF_B=len(TR), both=len(a & b), only_V0=len(a - b), only_REF=len(b - a), net_match_001=int((dn < 0.01).sum()),
               kind_match=int((J.k == J.kind).sum()), max_abs_dnet=float(dn.max()), V0_sum=V0.net.sum(), REF_sum=TR.net.sum())
    res["parity"] = par; print("PARITY", par)
    # ---------------- variants
    VAR = ["V0", "VE1", "VE2", "P1", "P2", "LAT1", "LAT2"]
    K = {nm: kept(variant(E, nm)) for nm in VAR}
    rows = []
    for nm in VAR:
        for bn, sp in (("DESIGN", DES), ("HOLDOUT", HOL), ("FULL", FULL)):
            s = stats(K[nm], sp); s.update(var=nm, block=bn)
            dlt = daily(K[nm], sp) - daily(K["V0"], sp)
            s["dsum"] = dlt.sum(); s["ci_lo"], s["ci_hi"] = boot_sum(dlt)
            rows.append(s)
    T1 = pd.DataFrame(rows); res["table"] = T1
    print(T1.round(3).to_string())
    # ---------------- halves
    hv = []
    for nm in VAR[1:]:
        for h, a_, b_ in HALVES:
            sp = (TS(a_), TS(b_)); s0 = stats(K["V0"], sp); s1 = stats(K[nm], sp)
            hv.append(dict(var=nm, half=h, N0=s0.get("N", 0), N1=s1.get("N", 0), avg0=s0.get("avg", np.nan), avg1=s1.get("avg", np.nan),
                           dsum=s1.get("sum", 0) - s0.get("sum", 0)))
    H = pd.DataFrame(hv); res["halves"] = H; print(H.round(3).to_string())
    # ---------------- false triggers / matched
    ft = []; mt = []
    for nm in ("VE1", "VE2", "P1", "P2"):
        x = K[nm]; early = x["at"].isin(["Tm1", "Tm2"])
        for bn, sp in (("DESIGN", DES), ("HOLDOUT", HOL), ("FULL", FULL)):
            m = (x.entry_ms >= sp[0]) & (x.entry_ms < sp[1])
            for lab, mm in (("early_true", early & x.trig0), ("early_FALSE", early & ~x.trig0), ("T0_fallback", ~early)):
                y = x[m & mm]
                ft.append(dict(var=nm, block=bn, group=lab, N=len(y), WR=(y.net > 0).mean() * 100 if len(y) else np.nan,
                               avg=y.net.mean() if len(y) else np.nan, sum=y.net.sum()))
    for k_, a_ in ((1, "Tm1"), (2, "Tm2")):
        e = E[E["at"] == a_].set_index("cid"); z = E[E["at"] == "T0"].set_index("cid")
        ii = e.index.intersection(z.index)
        d = pd.DataFrame(dict(pair=z.loc[ii, "pair"], T=z.loc[ii, "T"], nete=e.loc[ii, "net"], n0=z.loc[ii, "net"], E0=np.nan)).dropna(subset=["nete", "n0"])
        Cc = C.loc[d.index]
        d["entry_gain_pct"] = (Cc[f"E{k_}"] / Cc.E0 - 1) * 100       # + = the early entry price is higher = better for the short
        for bn, sp in (("DESIGN", DES), ("HOLDOUT", HOL), ("FULL", FULL)):
            y = d[(d["T"] >= sp[0]) & (d["T"] < sp[1])]
            mt.append(dict(k=k_, block=bn, N=len(y), avg_entry_gain=y.entry_gain_pct.mean(), med_entry_gain=y.entry_gain_pct.median(),
                           avg_net_early=y.nete.mean(), avg_net_T0=y.n0.mean(), paired_d=(y.nete - y.n0).mean(),
                           share_better=((y.nete - y.n0) > 0).mean() * 100))
    FT = pd.DataFrame(ft); MT = pd.DataFrame(mt); res["false"] = FT; res["matched"] = MT
    print(FT.round(3).to_string()); print(MT.round(3).to_string())
    # latency matched (same candles, T0 vs T+1 / T+2), entry price drift
    lt = []
    for a_ in ("Tp1", "Tp2"):
        e = E[E["at"] == a_].set_index("cid"); z = E[E["at"] == "T0"].set_index("cid"); ii = e.index.intersection(z.index)
        d = pd.DataFrame(dict(T=z.loc[ii, "T"], nete=e.loc[ii, "net"], n0=z.loc[ii, "net"])).dropna()
        for bn, sp in (("DESIGN", DES), ("HOLDOUT", HOL), ("FULL", FULL)):
            y = d[(d["T"] >= sp[0]) & (d["T"] < sp[1])]
            lt.append(dict(at=a_, block=bn, N=len(y), avg_T0=y.n0.mean(), avg_late=y.nete.mean(), paired_d=(y.nete - y.n0).mean()))
    LT = pd.DataFrame(lt); res["lat_matched"] = LT; print(LT.round(3).to_string())
    # ---------------- concentration + LOO (FULL and HOLDOUT)
    cc = []
    for nm in VAR[1:]:
        for bn, sp in (("HOLDOUT", HOL), ("FULL", FULL), ("DESIGN", DES)):
            f = lambda v: v[(v.entry_ms >= sp[0]) & (v.entry_ms < sp[1])].groupby("pair").net.sum()
            pp = f(K[nm]).subtract(f(K["V0"]), fill_value=0.0)
            tot = pp.sum(); top = pp.idxmax() if tot >= 0 else pp.idxmin()
            loo = tot - pp
            cc.append(dict(var=nm, block=bn, dsum=tot, top_pair=top, top_d=pp[top], top_share=pp[top] / tot * 100 if tot != 0 else np.nan,
                           n_pairs_changed=int((pp.abs() > 1e-9).sum()), loo_min=loo.min(), loo_min_pair=loo.idxmin(),
                           loo_drop_top=tot - pp[top]))
    CC = pd.DataFrame(cc); res["conc"] = CC; print(CC.round(3).to_string())
    res["K"] = K
    pd.to_pickle(res, os.path.join(OUT, "res.pkl"))
    allk = pd.concat([v.assign(var=nm) for nm, v in K.items()], ignore_index=True)
    allk["block"] = np.where(allk.entry_ms < HOL[0], "DESIGN", "HOLDOUT")
    allk[["var", "pair", "cid", "T", "at", "entry_ms", "xm", "k", "r", "f", "slip", "atrp", "net", "trig0", "src1m", "block"]].to_csv(
        os.path.join(OUT, "trades.csv.gz"), index=False, compression="gzip")
    chart(C, E, K)


def chart(C, E, K):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    V0 = K["V0"]; VE2 = K["VE2"]
    fal = VE2[VE2["at"].isin(["Tm1", "Tm2"]) & ~VE2.trig0]
    MINS = np.arange(-5, 241)
    def paths(df):
        P = []
        for p, g in df.groupby("pair"):
            days = sorted({int(T) // DAY + k for T in g["T"].values for k in (-1, 0, 1)})
            one, _ = load1m(p, days)
            if one is None:
                continue
            t, o, h, l, c = one
            for T in g["T"].values.astype(np.int64):
                ts = T + MINS * M1
                j = np.searchsorted(t, ts)
                ok = (j < len(t)) & (t[np.minimum(j, len(t) - 1)] == ts)
                if not ok[3:8].all():                     # need the bars at T−2 … T+2
                    continue
                P.append(np.where(ok, o[np.minimum(j, len(t) - 1)], np.nan))   # bar OPEN at each minute = price at that instant
        return np.array(P)
    P0 = paths(V0); PF = paths(fal)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for axi, (P, ttl) in enumerate(((P0, f"kept T0 fills (V0 = REF_B), N={len(P0)}"), (PF, f"FALSE early triggers (VE2), N={len(PF)}"))):
        for anc, col, lab in ((-2, "#1f77b4", "enter T−2 min"), (-1, "#2ca02c", "enter T−1 min"), (0, "#000000", "enter T (backtest)"),
                              (1, "#d62728", "enter T+1 min (latency)")):
            ia = int(np.nonzero(MINS == anc)[0][0])
            pnl = -(P / P[:, [ia]] - 1) * 100
            m = np.nanmean(pnl, 0); m[:ia] = np.nan
            ax[axi].plot(MINS, m, color=col, lw=1.4, label=lab)
        ax[axi].axvline(0, color="grey", lw=0.6, ls=":"); ax[axi].axhline(0, color="grey", lw=0.6)
        ax[axi].set_xscale("symlog", linthresh=10); ax[axi].set_xlabel("minutes from the signal close T (symlog)")
        ax[axi].set_ylabel("avg short P&L % (gross, no exit)"); ax[axi].set_title(ttl, fontsize=10); ax[axi].legend(fontsize=8)
    fig.suptitle("DIST — anchored average short P&L path by entry moment (BACKTEST, 1m bars, 2023-07 → 2026-11)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(R, "reports", "DIST_ENTRY_TIMING_2026-11-20.png"), dpi=110)
    print("chart N", len(P0), len(PF))


if __name__ == "__main__":
    main()
