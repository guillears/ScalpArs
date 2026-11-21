#!/usr/bin/env python3
"""DIST_AFTERCRASH (2026-11-21) — EVAL per reports/DIST_AFTERCRASH_2026-11-21_PREREG.md. Research only.
Inputs: reports/DIST_AFTERCRASH_2026-11-21_data/features.csv.gz + paths.npy (features script) and the Exit B REF net % per fill
(reports/DIST_THEORY_EXIT_2026-11-21_data/perfill.csv.gz). Real fills: exports_archive DISTRIBUTION_SHORT CLOSED, 1m klines
(data.binance.vision / fapi with the weight header, via STATE_EXIT_2026-11-21_common) cached in the study's data dir.
Outputs: reports/DIST_AFTERCRASH_2026-11-21_data/{tables.md, gates.csv, real_fills.csv}, reports/DIST_AFTERCRASH_2026-11-21.png
Usage: venv/bin/python -I scripts/DIST_AFTERCRASH_2026-11-21_eval.py
"""
import glob, os, sys, importlib.util
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "DIST_AFTERCRASH_2026-11-21_data")
SIZE = 0.61
H4, MIN, M15, H1 = 4 * 3_600_000, 60_000, 900_000, 3_600_000
rng = np.random.default_rng(20261121)
OUT = []


def say(s=""):
    print(s); OUT.append(s)


def load():
    F = pd.read_csv(os.path.join(D, "features.csv.gz"))
    P = pd.read_csv(os.path.join(R, "reports", "DIST_THEORY_EXIT_2026-11-21_data", "perfill.csv.gz"))
    P["entry_ms"] = pd.to_datetime(P.entry).values.astype("datetime64[ms]").astype("int64")
    F = F.merge(P[["pair", "entry_ms", "REF", "REF_k"]], on=["pair", "entry_ms"], how="left", validate="1:1")
    assert F.REF.notna().all() and len(F) == 1547
    F["win"] = F.REF > 0
    F["w4"] = F.entry_ms // H4
    paths = np.load(os.path.join(D, "paths.npy"))
    return F, paths


def be_wr(x):
    w, l = x[x > 0], x[x <= 0]
    return abs(l.mean()) / (w.mean() + abs(l.mean())) * 100 if len(w) and len(l) else np.nan


def stats(g):
    x = g.REF.values
    if not len(x):
        return dict(N=0)
    neg = g[g.REF < 0]
    tl = neg.REF.sum()
    topw = (neg.groupby("w4").REF.sum().min() / tl * 100) if tl < 0 else 0.0
    topp = (neg.groupby("pair").REF.sum().min() / tl * 100) if tl < 0 else 0.0
    return dict(N=len(x), WR=(x > 0).mean() * 100, avg=x.mean(), BE=be_wr(x), days=g.day.nunique(), win4h=g.w4.nunique(),
                top_pair=(g.pair.value_counts().iloc[0] / len(g) * 100), topw_loss=topw, topp_loss=topp)


def fmt_row(lbl, s):
    if s.get("N", 0) == 0:
        return f"| {lbl} | 0 | | | | | | |"
    return (f"| {lbl} | {s['N']} | {s['WR']:.0f} % | {s['avg']:+.2f} | {s['BE']:.0f} % | {s['days']} / {s['win4h']} | "
            f"{s['top_pair']:.0f} % |")


def bucket(v, edges):
    """edges ascending e.g. [-20,-15,-10,-5]; bucket k: (edges[k-1], edges[k]]"""
    lab = []
    for k in range(len(edges) + 1):
        lo = "-inf" if k == 0 else f"{edges[k-1]:g}"
        hi = f"{edges[k]:g}" if k < len(edges) else "inf"
        lab.append(f"({lo}, {hi}]" if k < len(edges) else f"> {lo}")
    b = np.searchsorted(np.array(edges, float), v, side="left")
    return b, lab


def dose_table(F, var, edges, title):
    say(f"\n### {title} — `{var}`")
    say("| bucket | N | WR | avg net % | bucket BE WR | days / 4h windows | top pair share |")
    say("|---|---|---|---|---|---|---|")
    b, lab = bucket(F[var].values, edges)
    for blk in ("ALL", "DESIGN", "HOLDOUT", "BTC bull (HOL)", "BTC bear (HOL)", "BTC bull (ALL)", "BTC bear (ALL)"):
        if blk == "ALL":
            m = np.ones(len(F), bool)
        elif blk in ("DESIGN", "HOLDOUT"):
            m = (F.block == blk).values
        else:
            reg = 1 if "bull" in blk else -1
            m = (F.btc_d1_regime == reg).values & ((F.block == "HOLDOUT").values if "HOL" in blk else True)
        say(f"| **{blk}** | | | | | | |")
        for k, L in enumerate(lab):
            say(fmt_row(L, stats(F[m & (b == k)])))


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def boot_upper(g, nb=5000):
    """window-clustered bootstrap of the block's mean: one-sided 95 % upper bound and P(mean >= 0)."""
    W = g.groupby("w4").REF.agg(["sum", "count"])
    s, c = W["sum"].values, W["count"].values
    n = len(W)
    idx = rng.integers(0, n, (nb, n))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(np.quantile(m, 0.95)), float((m >= 0).mean())


def dcr(F, keep, d0, d1):
    days = np.arange(d0, d1 + 1)
    r = pd.Series(0.0, index=days)
    g = F[keep].groupby("day").REF.sum() * SIZE / 100
    r.loc[g.index] += g.values
    return (np.prod(1 + r.values) ** (1 / len(days)) - 1) * 100


CUTS = [("r15", -10), ("r15", -15), ("r15", -20), ("r30", -10), ("r30", -15), ("r30", -20), ("r60", -10), ("r60", -15),
        ("r60", -20), ("drop60", -10), ("drop60", -15), ("drop60", -20), ("dd24", -20), ("dd24", -30)]


def gates(F, BE_SLEEVE, volhi):
    say("\n## Gate table — locked Pattern-C → FILTER expectancy bar (ALL fills; DESIGN / HOLDOUT shown)")
    say(f"Sleeve breakeven WR (all 1,547 kept fills, Exit B) = **{BE_SLEEVE:.1f} %**.")
    say("| block (would be skipped) | N (D/H) | WR | avg % | rest avg % | DES avg / HOL avg | 95 % upper (4h-window boot) | P(avg≥0) | 4h windows | top window / pair share of loss | t vs rest | legs passed |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    cuts = [(v, th, F[v] <= th) for v, th in CUTS] + [("r15≤-10 ∧ volx15 top tercile", None, (F.r15 <= -10) & (F.volx15 >= volhi))]
    for v, th, m in cuts:
        lbl = f"{v} ≤ {th}" if th is not None else v
        g, rest = F[m], F[~m]
        s = stats(g)
        if s["N"] < 2:
            continue
        up, p0 = boot_upper(g)
        des, hol = g[g.block == "DESIGN"], g[g.block == "HOLDOUT"]
        t = welch_t(g.REF.values, rest.REF.values)
        legs = dict(WR=s["WR"] < BE_SLEEVE, conf=up < 0, win=s["win4h"] >= 8, conc=max(s["topw_loss"], s["topp_loss"]) < 50, N=s["N"] >= 15)
        npass = sum(legs.values())
        say(f"| {lbl} | {s['N']} ({len(des)}/{len(hol)}) | {s['WR']:.0f} % | {s['avg']:+.2f} | {rest.REF.mean():+.2f} | "
            f"{des.REF.mean() if len(des) else float('nan'):+.2f} / {hol.REF.mean() if len(hol) else float('nan'):+.2f} | {up:+.2f} | {p0:.2f} | "
            f"{s['win4h']} | {s['topw_loss']:.0f} % / {s['topp_loss']:.0f} % | {t:+.2f} | {npass}/5 "
            f"({', '.join(k for k, ok in legs.items() if ok) or '—'}) |")
        rows.append(dict(cut=lbl, **s, up95=up, p_ge0=p0, rest_avg=rest.REF.mean(), des_avg=des.REF.mean() if len(des) else np.nan,
                         hol_avg=hol.REF.mean() if len(hol) else np.nan, n_des=len(des), n_hol=len(hol), t=t, legs=npass))
    G = pd.DataFrame(rows); G.to_csv(os.path.join(D, "gates.csv"), index=False)
    return G, cuts


def null_test(F, cuts):
    say("\n## Family-wise shuffled null (variable shuffled within half-years, 2,000×, max |t| over the 15 family cuts)")
    halves = F.half.values
    y = F.REF.values
    obs = {}
    for v, th, m in cuts:
        obs[(v, th)] = welch_t(y[m.values], y[~m.values])
    vars_needed = sorted({v for v, th, m in cuts if th is not None} | {"volx15"})
    X = {v: F[v].values for v in vars_needed}
    groups = [np.nonzero(halves == h)[0] for h in np.unique(halves)]
    mx = []
    for _ in range(2000):
        perm = np.arange(len(F))
        for g in groups:
            perm[g] = rng.permutation(g)
        ts = []
        for v, th, m in cuts:
            if th is None:
                mm = (X["r15"][perm] <= -10) & (X["volx15"][perm] >= np.nanquantile(X["volx15"], 2 / 3))
            else:
                mm = X[v][perm] <= th
            if mm.sum() >= 2:
                ts.append(abs(welch_t(y[mm], y[~mm])))
        mx.append(np.nanmax(ts))
    q95 = float(np.quantile(mx, 0.95))
    best = max(obs.items(), key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1)
    say(f"Family null 95 % max |t| = **{q95:.2f}**. Observed: " + " · ".join(f"{v}≤{th}: {t:+.2f}" if th is not None else f"liq-proxy: {t:+.2f}"
                                                                         for (v, th), t in obs.items()))
    say(f"Largest |t| = {best[0]} {best[1]:+.2f} → {'beats' if abs(best[1]) > q95 else 'does NOT beat'} the null. "
        "(A positive t means the 'after-crash' block did BETTER than the rest.)")
    return q95


def dcr_table(F, m, lbl):
    say(f"\n## DCR before → after at ONE size (0.61 × equity per fill), skipping `{lbl}`")
    say("| period | days | fills before → after | DCR before | DCR after | Δ | Δ after 50 % haircut |")
    say("|---|---|---|---|---|---|---|")
    for blk in ("ALL", "DESIGN", "HOLDOUT"):
        mm = np.ones(len(F), bool) if blk == "ALL" else (F.block == blk).values
        G = F[mm]
        d0, d1 = int(G.day.min()), int(G.day.max())
        a = dcr(G, np.ones(len(G), bool), d0, d1); b = dcr(G, ~m[mm], d0, d1)
        say(f"| {blk} | {d1 - d0 + 1} | {len(G)} → {int((~m[mm]).sum())} | {a:+.3f} % | {b:+.3f} % | {b - a:+.3f} pp | {(b - a) * 0.5:+.3f} pp |")


def loo(F, m):
    g = F[m]
    say("\n## Leave-one-out on the primary block (r15 ≤ −15): avg net % with each 4h window removed")
    vals = [g[g.w4 != w].REF.mean() for w in g.w4.unique()]
    say(f"min {min(vals):+.2f} · max {max(vals):+.2f} · full {g.REF.mean():+.2f} (N windows {len(vals)})")


def chart(F, paths):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = (np.arange(paths.shape[1]) - 24) * 5
    crash = (F.r15 <= -15).values
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    cols = {"winners": "#2a7f62", "losers": "#c0392b"}
    for a, (msk, title) in zip(ax, ((crash, f"After-crash: signal 15m ≤ −15 % (N={crash.sum()})"),
                                    (~crash, f"Rest of the kept cohort (N={(~crash).sum()})"))):
        for nm, wm in (("winners", F.win.values), ("losers", ~F.win.values)):
            sel = msk & wm
            med = np.nanmedian(paths[sel], 0)
            q1, q3 = np.nanquantile(paths[sel], [.25, .75], axis=0)
            a.plot(x, med, color=cols[nm], lw=2, label=f"{nm} (N={sel.sum()}) median")
            a.fill_between(x, q1, q3, color=cols[nm], alpha=.12)
        a.axvline(0, color="k", lw=.8); a.axhline(0, color="grey", lw=.6, ls=":")
        a.set_title(title, fontsize=10); a.set_xlabel("minutes from entry (5m closes; entry = open of the next 5m bar)")
        a.legend(fontsize=8, loc="lower left")
    ax[0].set_ylabel("price vs entry, % (short wins when below 0 after entry)")
    fig.suptitle("DIST shorts: price path around entry — after-crash vs rest, winners vs losers (median, IQR band; backtest 2023-07 → 2026-11)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "reports", "DIST_AFTERCRASH_2026-11-21.png"), dpi=130)


def real_fills():
    sp = importlib.util.spec_from_file_location("SEC", os.path.join(R, "scripts", "STATE_EXIT_2026-11-21_common.py"))
    C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
    C.K1 = os.path.join(D, "k1m"); os.makedirs(C.K1, exist_ok=True)
    xs = []
    for f in sorted(glob.glob(os.path.join(R, "exports_archive", "scalpars_orders_*.csv"))):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "entry_strategy" not in d.columns:
            continue
        d = d[(d.entry_strategy.astype(str) == "DISTRIBUTION_SHORT") & (d.status == "CLOSED")].copy()
        if len(d):
            d["src"] = os.path.basename(f); xs.append(d)
    A = pd.concat(xs, ignore_index=True)
    if "is_probe" in A.columns:
        A = A[~A.is_probe.fillna(False).astype(bool)]
    A["key"] = A.opened_at.astype(str).str[:19] + "|" + A.pair + "|" + A.direction.astype(str)
    A = A.sort_values("src").drop_duplicates("key", keep="last").sort_values("opened_at")
    rows = []
    for r in A.itertuples():
        te = int(pd.Timestamp(str(r.opened_at)[:19]).value // 10**6)
        sig = te // M15 * M15          # the signal 15m bar closes at this boundary
        a, b = sig - 25 * H1, sig
        for ds in C.days_of(a, b):
            C.fetch_kday(r.pair, ds)
        k = C.grid(C.k1m(r.pair, a, b), a, b)
        t, h, c, qv = k["t"], k["h"], k["c"], k["qv"]
        i = int(np.searchsorted(t, sig - MIN))           # last closed 1m bar (opens 1 min before the boundary)
        cj = c[i]
        f = dict(pair=r.pair, opened=str(r.opened_at)[:16], pnl_pct=float(r.pnl_percentage), reason=r.close_reason,
                 stamp_bar_ret=float(getattr(r, "entry_dist_bar_ret_pct", np.nan)))
        for nm, n in (("r15", 15), ("r30", 30), ("r60", 60), ("r240", 240)):
            f[nm] = (cj / c[i - n] - 1) * 100
        f["drop60"] = (cj / np.nanmax(h[i - 59:i + 1]) - 1) * 100
        f["dd24"] = (cj / np.nanmax(h[i - 1439:i + 1]) - 1) * 100
        q15 = pd.Series(qv).rolling(15).sum().values
        prior = q15[i - 15:i - 15 - 1440:-15]
        prior = prior[np.isfinite(prior)]
        f["volx15"] = q15[i] / np.median(prior) if len(prior) > 20 else np.nan
        rows.append(f)
    X = pd.DataFrame(rows); X.to_csv(os.path.join(D, "real_fills.csv"), index=False)
    say("\n## Real fills (every DISTRIBUTION_SHORT CLOSED row, exports_archive, dedup; 1m klines closed at the signal boundary)")
    say("| opened (UTC) | pair | P&L % | exit | r15 % (stamp) | r30 % | r60 % | below 60-min high % | below 24 h high % | volx15 | primary block? |")
    say("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in X.itertuples():
        say(f"| {r.opened} | {r.pair} | {r.pnl_pct:+.2f} | {r.reason} | {r.r15:+.1f} ({r.stamp_bar_ret:+.1f}) | {r.r30:+.1f} | {r.r60:+.1f} | "
            f"{r.drop60:+.1f} | {r.dd24:+.1f} | {r.volx15:.0f}× | {'YES' if r.r15 <= -15 else 'no'} |")
    return X


def main():
    F, paths = load()
    BE_SLEEVE = be_wr(F.REF.values)
    volhi = float(np.nanquantile(F.volx15, 2 / 3))
    s = stats(F)
    say(f"# DIST_AFTERCRASH tables (generated)\n\nCohort: {s['N']} fills · WR {s['WR']:.1f} % · avg {s['avg']:+.3f} % · sleeve BE WR {BE_SLEEVE:.1f} % · "
        f"volx15 top-tercile edge {volhi:.0f}×")
    say("\n## Dose-response")
    E = [-20, -15, -10, -5]
    dose_table(F, "r15", E, "Primary: return of the 15 min before entry (= the signal candle)")
    for v in ("r30", "r60", "drop60"):
        dose_table(F, v, E, "Secondary")
    dose_table(F, "dd24", [-30, -20, -10], "Secondary: below the 24 h high")
    say("\n### r240 sign (4 h return)")
    for blk in ("ALL", "DESIGN", "HOLDOUT"):
        m = np.ones(len(F), bool) if blk == "ALL" else (F.block == blk).values
        say(f"{blk}: <0 " + fmt_row("", stats(F[m & (F.r240 < 0).values])) + "  ≥0 " + fmt_row("", stats(F[m & (F.r240 >= 0).values])))
    say("\n### Liquidation proxy: r15 ≤ −10 split by volx15 (top tercile vs rest)")
    say("| cell | N | WR | avg | BE | days / windows | top pair |\n|---|---|---|---|---|---|---|")
    for lbl, m in (("r15≤−10 ∧ volx15 high", (F.r15 <= -10) & (F.volx15 >= volhi)), ("r15≤−10 ∧ volx15 not high", (F.r15 <= -10) & (F.volx15 < volhi)),
                   ("r15>−10 ∧ volx15 high", (F.r15 > -10) & (F.volx15 >= volhi)), ("r15>−10 ∧ volx15 not high", (F.r15 > -10) & (F.volx15 < volhi))):
        say(fmt_row(lbl, stats(F[m])))
    say("\n### Primary block (r15 ≤ −15) — exit mix")
    g = F[F.r15 <= -15]
    say(" · ".join(f"{k}: {v}" for k, v in g.REF_k.value_counts().items()) + f" | rest: " +
        " · ".join(f"{k}: {v / (F.r15 > -15).sum() * 100:.0f} %" for k, v in F[F.r15 > -15].REF_k.value_counts().items()))
    say(f"Primary block pairs: " + ", ".join(f"{p} {n}" for p, n in g.pair.value_counts().head(8).items()))
    G, cuts = gates(F, BE_SLEEVE, volhi)
    null_test(F, cuts)
    prim = (F.r15 <= -15).values
    dcr_table(F, prim, "r15 ≤ −15")
    loo(F, prim)
    chart(F, paths)
    # event-study numbers for the write-up
    x = (np.arange(paths.shape[1]) - 24) * 5
    for nm, msk in (("after-crash", prim), ("rest", ~prim)):
        md = np.nanmedian(paths[msk], 0)
        say(f"\nmedian path {nm}: " + " · ".join(f"{xx:+d}m {md[k]:+.1f}" for k, xx in enumerate(x) if xx in (-60, -30, -15, 0, 15, 30, 60, 120)))
    try:
        real_fills()
    except Exception as e:  # network is optional; report it
        say(f"\nReal-fill kline fetch failed: {e!r}")
    with open(os.path.join(D, "tables.md"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
