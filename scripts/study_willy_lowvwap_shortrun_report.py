#!/usr/bin/env python3
"""Report for the WILLY LOW-VWAP / SHORT-RUN zone study (2026-10-16). Reads reports/WILLY_LOWVWAP_SHORTRUN_rows.csv
(built by scripts/study_willy_lowvwap_shortrun.py) + the live export + 1m klines of the 8 live fills (scratch dir, 8 public
requests, weight 8). Writes reports/WILLY_LOWVWAP_SHORTRUN_tables.md and reports/WILLY_LOWVWAP_SHORTRUN_tests.csv.
Research only."""
import os, sys, glob, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
ROWS = os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_rows.csv")
OUT_MD = os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_tables.md")
OUT_T = os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_tests.csv")
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-16_13-10-21 (1).csv")
K1M_LIVE = sys.argv[1] if len(sys.argv) > 1 else None
VW, RUN = 0.10, 10.0
TAKER = 0.045
ZONES = {"ZONE": lambda d: (d.entry_frenzy_vs_vwap_pct <= VW) & (d.entry_frenzy_run_pct <= RUN),
         "VWAP_LEG": lambda d: d.entry_frenzy_vs_vwap_pct <= VW,
         "RUN_LEG": lambda d: d.entry_frenzy_run_pct <= RUN}


def day_boot(d, day, n=4000, seed=7):
    d = np.asarray(d, float); day = np.asarray(day)
    g = pd.DataFrame(dict(d=d, day=day)).groupby("day").d.agg(["sum", "size"])
    if len(g) < 3:
        return np.nan, np.nan, np.nan
    s, c = g["sum"].values, g["size"].values
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float((m > 0).mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def be_wr(y):
    w, l = y[y > 0], y[y <= 0]
    return np.nan if not len(w) or not len(l) else 100 * abs(l.mean()) / (w.mean() + abs(l.mean()))


def loss_share(y, key):
    """largest single key's share of the cohort's NET loss (Σ < 0); NaN if Σ ≥ 0."""
    s = pd.Series(np.asarray(y, float)).groupby(np.asarray(key)).sum(); tot = s.sum()
    if tot >= 0 or not len(s):
        return np.nan, None
    return float(s.min() / tot), s.idxmin()


def lomo(d, month):
    d = np.asarray(d, float); month = np.asarray(month); full = d.mean()
    v = [d[month != m].mean() for m in sorted(set(month)) if (month != m).sum()]
    return (min(v), max(v), len(v), all(np.sign(x) == np.sign(full) for x in v)) if v else (np.nan, np.nan, 0, False)


def f(x, k=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (f"{x:+.{k}f}" if isinstance(x, (float, np.floating)) else str(x))


def md(df):
    cols = list(df.columns)
    o = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        o.append("| " + " | ".join(f(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |")
    return "\n".join(o)


def zone_tests(g, zname, label):
    """g = one seed × era (or pooled) × pricing block, today's exit in y, modifier in y_mod."""
    z = ZONES[zname](g).values
    y, ym = g.y.values, g.y_mod.values
    Z, Rr = g[z], g[~z]
    n = len(g)
    be = be_wr(y)
    pz, lo, hi = day_boot(Z.y, Z.day) if len(Z) else (np.nan,) * 3
    p_neg = 1 - pz if np.isfinite(pz) else np.nan
    dsh, dkey = loss_share(Z.y, Z.day) if len(Z) else (np.nan, None)
    psh, pkey = loss_share(Z.y, Z.pair) if len(Z) else (np.nan, None)
    wrz = 100 * (Z.y > 0).mean() if len(Z) else np.nan
    bar = dict(b1_wr_below_be=bool(wrz < be), b2_p_neg_ge95=bool(p_neg >= 0.95), b3_days_ge8=bool(Z.day.nunique() >= 8),
               b3_conc_lt50=bool(not (dsh >= 0.5) and not (psh >= 0.5)), b4_n_ge15=bool(len(Z) >= 15))
    dref = np.where(z, -y, 0.0)                      # refuse the zone: every refused fill's P&L leaves (winners count)
    pr, rlo, rhi = day_boot(dref, g.day)
    lr = lomo(dref, g.month)
    dmod = np.where(z, ym - y, 0.0)
    pm, mlo, mhi = day_boot(dmod, g.day)
    lm = lomo(dmod, g.month)
    return dict(block=label, zone=zname, N=n, BE_WR=be,
                zN=len(Z), z_days=Z.day.nunique(), z_WR=wrz, z_avg=Z.y.mean() if len(Z) else np.nan, z_sum=Z.y.sum(),
                z_ci=f"[{lo:+.3f}, {hi:+.3f}]", z_P_neg=p_neg,
                rN=len(Rr), r_WR=100 * (Rr.y > 0).mean() if len(Rr) else np.nan, r_avg=Rr.y.mean() if len(Rr) else np.nan, r_sum=Rr.y.sum(),
                top_day_loss_share=dsh, top_day=dkey, top_pair_loss_share=psh, top_pair=pkey, **bar,
                BAR_PASS=all(bar.values()),
                d_refuse=dref.mean(), d_refuse_P=pr, d_refuse_ci=f"[{rlo:+.3f}, {rhi:+.3f}]",
                refused_winners=int((Z.y > 0).sum()), refused_losers=int((Z.y <= 0).sum()),
                lomo_refuse_min=lr[0], lomo_refuse_max=lr[1], lomo_refuse_same_sign=lr[3],
                d_tp100=dmod.mean(), d_tp100_P=pm, d_tp100_ci=f"[{mlo:+.3f}, {mhi:+.3f}]",
                tp100_saved=int((dmod > 1e-9).sum()), tp100_killed=int((dmod < -1e-9).sum()),
                lomo_tp100_min=lm[0], lomo_tp100_max=lm[1], lomo_tp100_same_sign=lm[3])


def live_block(kdir):
    L = pd.read_csv(LIVE, low_memory=False)
    L = L[(L.entry_strategy == "FRENZY_WILLY") & (L.status == "CLOSED")].copy()
    L["usd_pt"] = L.pnl.astype(float) / L.pnl_percentage.astype(float)
    L["zone"] = ZONES["ZONE"](L)
    rows = []
    for r in L.itertuples():
        E = float(r.entry_price); te = pd.Timestamp(r.opened_at[:19])
        fs = glob.glob(os.path.join(kdir, f"{r.pair}_*.json")) if kdir else []
        why_t = pct_t = why_m = pct_m = None
        if fs:
            k = pd.DataFrame(json.load(open(fs[0])), columns=["t", "o", "h", "l", "c", "v"] + list(range(6)))
            k["t"] = pd.to_datetime(k.t.astype("int64"), unit="ms")
            k = k[(k.t > te.floor("min")) & (k.t < te + pd.Timedelta(minutes=90))]   # entry minute skipped (walker convention)

            def net(p):
                q = float(p) / E
                return (q - 1) * 100 - TAKER - TAKER * q

            def walk(tp):
                for b in k.itertuples():
                    age = (b.t - te).total_seconds() / 60
                    line = tp if age + 1 <= 30 else 0.25          # a bar that starts before 30 min uses the early line
                    if net(b.l) <= -4:
                        return "STOP", min(net(b.o), -4.0) - 0.2
                    if net(b.h) >= line:
                        return ("TP" if line == tp else "LATE_TP"), line
                return "CAP", net(k.c.iloc[-1]) if len(k) else np.nan
            why_t, pct_t = walk(1.25)
            why_m, pct_m = walk(1.0)
        rows.append(dict(pair=r.pair, opened=r.opened_at[5:16], zone=bool(r.zone), vs_vwap=r.entry_frenzy_vs_vwap_pct, run=r.entry_frenzy_run_pct,
                         as_traded_why=r.close_reason, as_traded_pct=float(r.pnl_percentage), as_traded_usd=float(r.pnl),
                         peak=float(r.peak_pnl), usd_pt=r.usd_pt, k1m_today_why=why_t, k1m_today_pct=pct_t, k1m_tp100_why=why_m, k1m_tp100_pct=pct_m))
    T = pd.DataFrame(rows)
    # zone with TP +1.0: zone fills re-priced on the 1m path (TP lines booked at the line = resting maker); others as traded
    T["refused_pct"] = np.where(T.zone, 0.0, T.as_traded_pct)
    # a zone fill keeps its real result unless TP +1.0 changes its 1m outcome (then the re-priced value; TP lines = resting maker)
    chg = T.zone & ((T.k1m_tp100_why != T.k1m_today_why) | ~np.isclose(T.k1m_tp100_pct.astype(float), T.k1m_today_pct.astype(float)))
    T["tp100_pct"] = np.where(chg, T.k1m_tp100_pct, T.as_traded_pct)
    return T


def per_trade_block(T, extra_cols):
    for c in ["as_traded", "refused", "tp100"] + extra_cols:
        T[c + "_usd"] = T[c + "_pct"] * T.usd_pt if c != "as_traded" else T.as_traded_usd
    return T


def main():
    R = pd.read_csv(ROWS, low_memory=False)
    B = R[(R.block == "BT") & (R.why != "UNRES") & (R.mod_why != "UNRES")].copy()
    B["tick_priced"] = B.tick_priced.astype(bool)
    out, tests = [], []
    out.append(f"# Tables — WILLY low-VWAP / short-run zone (2026-10-16)\n\nZONE = entry_frenzy_vs_vwap_pct ≤ {VW} ∧ entry_frenzy_run_pct ≤ {RUN}. "
               "Today's WILLY exit (resting TP +1.25 at the line · stop −4 booked min(print, −4) − 0.2 · 90-min cap · late TP +0.25 after 30 min). "
               "y = % per fill, fee-net, leverage-invariant. TP100 = zone fills at TP +1.0 (late TP unchanged).\n")
    out.append(f"BT fills used: {len(B)} (3 seeds; UNRES dropped: {int(((R.block=='BT')&((R.why=='UNRES')|(R.mod_why=='UNRES'))).sum())}).\n")
    # per seed × era × pricing
    blocks = []
    for seed in (1, 2, 3):
        S = B[B.seed == seed]
        for pr, P in (("tick", S[S.tick_priced]), ("1m", S[~S.tick_priced])):
            for era in ("2025", "2026-H1", "2026-H2"):
                blocks.append((f"s{seed}|{pr}|{era}", P[P.era == era]))
            blocks.append((f"s{seed}|{pr}|2026", P[P.era != "2025"]))
            blocks.append((f"s{seed}|{pr}|ALL", P))
        blocks.append((f"s{seed}|tick|2025-top50", S[S.tick_priced & (S.era == "2025") & S.top50.astype(bool)]))
    for label, g in blocks:
        if len(g) < 5:
            continue
        for zn in ZONES:
            tests.append(zone_tests(g, zn, label))
    T = pd.DataFrame(tests)
    T.to_csv(OUT_T, index=False)
    show = ["block", "N", "BE_WR", "zN", "z_days", "z_WR", "z_avg", "z_sum", "z_ci", "z_P_neg", "rN", "r_WR", "r_avg", "BAR_PASS",
            "d_refuse", "d_refuse_P", "refused_winners", "refused_losers", "d_tp100", "d_tp100_P", "tp100_saved", "tp100_killed"]
    for zn in ZONES:
        for seedtag in ("s1", "s2", "s3"):
            sub = T[(T.zone == zn) & T.block.str.startswith(seedtag + "|")]
            out.append(f"\n## {zn} — seed {seedtag[1:]} (tick = verdict; 1m = refute-only)\n")
            out.append(md(sub[show]))
    # expectancy-bar detail (seed 1 tick)
    out.append("\n## Expectancy filter bar detail — ZONE, seed 1, tick-priced\n")
    det = ["block", "zN", "z_WR", "BE_WR", "b1_wr_below_be", "z_P_neg", "b2_p_neg_ge95", "z_days", "b3_days_ge8", "top_day_loss_share", "top_day",
           "top_pair_loss_share", "top_pair", "b3_conc_lt50", "b4_n_ge15", "BAR_PASS"]
    out.append(md(T[(T.zone == "ZONE") & T.block.str.startswith("s1|tick")][det]))
    out.append("\n## Leave-one-month-out — seed 1 tick (Δ per WILLY fill, min / max over dropped months)\n")
    lc = ["block", "zone", "d_refuse", "lomo_refuse_min", "lomo_refuse_max", "lomo_refuse_same_sign", "d_tp100", "lomo_tp100_min", "lomo_tp100_max", "lomo_tp100_same_sign"]
    out.append(md(T[T.block.str.startswith("s1|tick")][lc]))
    # per-pair concentration (seed 1, tick, ZONE, all eras)
    S1 = B[(B.seed == 1) & B.tick_priced].copy(); S1["zone"] = ZONES["ZONE"](S1)
    Zp = S1[S1.zone].groupby("pair").y.agg(["size", "sum", "mean"]).sort_values("sum")
    out.append(f"\n## Per-pair — ZONE, seed 1, tick, all eras ({S1.zone.sum()} fills, {len(Zp)} pairs, Σ {S1[S1.zone].y.sum():+.2f})\n")
    out.append("Worst 8 pairs: " + " · ".join(f"{p} {int(r['size'])}·{r['sum']:+.2f}" for p, r in Zp.head(8).iterrows()))
    out.append("\nBest 8 pairs: " + " · ".join(f"{p} {int(r['size'])}·{r['sum']:+.2f}" for p, r in Zp.tail(8)[::-1].iterrows()))
    # descriptive bucket table (NOT for re-fitting)
    S1["vb"] = pd.cut(S1.entry_frenzy_vs_vwap_pct, [-1e9, -1, -0.25, 0.1, 0.5, 1e9], labels=["≤−1", "−1…−0.25", "−0.25…0.10", "0.10…0.5", ">0.5"])
    S1["rb"] = pd.cut(S1.entry_frenzy_run_pct, [0, 7, 10, 15, 1e9], labels=["≤7", "7…10", "10…15", ">15"])
    out.append("\n## Descriptive buckets (seed 1, tick, all eras; avg y · N) — description only, the verdict uses the frozen cuts\n")
    pv = S1.pivot_table(index="vb", columns="rb", values="y", aggfunc=lambda s: f"{s.mean():+.2f}·{len(s)}", observed=False)
    out.append(md(pv.reset_index().astype(str)))
    for era in ("2025", "2026-H1", "2026-H2"):
        e = S1[S1.era == era]
        pv = e.pivot_table(index="vb", columns="rb", values="y", aggfunc=lambda s: f"{s.mean():+.2f}·{len(s)}", observed=False)
        out.append(f"\n{era}:\n\n" + md(pv.reset_index().astype(str)))
    # exit-reason mix zone vs rest
    out.append("\n## Exit mix (seed 1, tick) — zone vs rest\n")
    mix = S1.groupby(["era", "zone"]).why.value_counts(normalize=True).unstack().fillna(0).round(3).reset_index()
    out.append(md(mix.astype(str)))
    # MASTER
    M = R[R.block == "MASTER"].copy()
    M["zone"] = ZONES["ZONE"](M)
    M["usd_pt"] = M.pnl.astype(float) / M.pnl_percentage.astype(float)
    M["as_traded_pct"] = M.pnl_percentage.astype(float); M["as_traded_usd"] = M.pnl.astype(float)
    M["today_pct"] = M.y; M["refused_pct"] = np.where(M.zone, 0.0, M.y); M["tp100_pct"] = np.where(M.zone, M.y_mod, M.y)
    for c in ("today", "refused", "tp100"):
        M[c + "_usd"] = M[c + "_pct"] * M.usd_pt
    M["batch"] = M.era
    M["refused_astraded_pct"] = np.where(M.zone, 0.0, M.as_traded_pct); M["refused_astraded_usd"] = np.where(M.zone, 0.0, M.as_traded_usd)
    out.append("\n## MASTER — per trade (paths on ticks; $ at the as-traded $ per point)\n")
    mc = ["batch", "pair", "opened_at", "zone", "entry_frenzy_vs_vwap_pct", "entry_frenzy_run_pct", "src", "as_traded_pct", "as_traded_usd", "why",
          "today_pct", "today_usd", "refused_pct", "refused_usd", "mod_why", "tp100_pct", "tp100_usd"]
    out.append(md(M.sort_values("te")[mc]))
    agg = []
    for b, g in list(M.groupby("batch")) + [("ALL", M)]:
        agg.append(dict(batch=b, N=len(g), zone_N=int(g.zone.sum()), zone_WR=100 * (g[g.zone].today_pct > 0).mean() if g.zone.any() else np.nan,
                        as_traded_pct=g.as_traded_pct.sum(), as_traded_usd=g.as_traded_usd.sum(), refused_astraded_pct=g.refused_astraded_pct.sum(),
                        refused_astraded_usd=g.refused_astraded_usd.sum(), today_pct=g.today_pct.sum(), today_usd=g.today_usd.sum(),
                        refused_pct=g.refused_pct.sum(), refused_usd=g.refused_usd.sum(), tp100_pct=g.tp100_pct.sum(), tp100_usd=g.tp100_usd.sum()))
    out.append("\nMASTER totals (Σ % and Σ $; B18 has no WILLY fill — WILLY went live 2026-10-08 18:30, B19's start):\n")
    out.append(md(pd.DataFrame(agg)))
    # LIVE
    L = live_block(K1M_LIVE)
    for c in ("refused", "tp100", "k1m_today"):
        L[c + "_usd"] = L[c + "_pct"] * L.usd_pt
    out.append("\n## CURRENT LIVE batch — per trade (as traded = real fills; TP100 = zone fills re-priced on public 1m klines, refute-only)\n")
    lcols = ["pair", "opened", "zone", "vs_vwap", "run", "as_traded_why", "as_traded_pct", "as_traded_usd", "peak", "k1m_today_why", "k1m_today_pct",
             "refused_pct", "refused_usd", "k1m_tp100_why", "tp100_pct", "tp100_usd"]
    out.append(md(L[lcols]))
    la = dict(N=len(L), zone_N=int(L.zone.sum()), as_traded_pct=L.as_traded_pct.sum(), as_traded_usd=L.as_traded_usd.sum(),
              k1m_today_pct=L.k1m_today_pct.sum(), k1m_today_usd=L.k1m_today_usd.sum(),
              refused_pct=L.refused_pct.sum(), refused_usd=L.refused_usd.sum(), tp100_pct=L.tp100_pct.sum(), tp100_usd=L.tp100_usd.sum())
    out.append("\nLIVE totals:\n\n" + md(pd.DataFrame([la])))
    L.to_csv(os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_live.csv"), index=False)
    M[mc + ["usd_pt"]].to_csv(os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_master.csv"), index=False)
    open(OUT_MD, "w").write("\n".join(out) + "\n")
    print("wrote", OUT_MD, OUT_T)


if __name__ == "__main__":
    main()
