#!/usr/bin/env python3
"""MOMENTUM LONG checklist items ③ (uniform-degradation), ④ (tape context), the candidate conditions through the locked
expectancy bar, and the sizing angle (2026-10-08). Read-only; caches only.
Out: reports/study_ml_checklist_tables.md (+ csv side files study_ml_checklist_{cohort_month,tape_month,tape_batch,sizing}.csv)
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
import study_ml_checklist_scan as SC
import study_ml_b18_common as C

OUT = []
def P(s=""):
    OUT.append(s); print(s)


def fmt(d, ns=1):
    if len(d) == 0:
        return "—"
    return f"{len(d)/ns:.0f}·{(d.pct>0).mean()*100:.0f}%·{d.pct.mean():+.3f}"


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    cols = [str(c) for c in df.columns]
    s = "| " + " | ".join([str(df.index.name or "")] + cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for i, r in df.iterrows():
        s += "| " + " | ".join([str(i)] + [str(x) for x in r.values]) + " |\n"
    return s


# ───────────────────────────── ③ uniform degradation ─────────────────────────────
TAGS = ["t_cell", "t_pvr", "t_sprint", "t_pattern", "t_heat", "t_chop", "t_negflank", "t_hot", "t_regime", "t_order"]


def item3(D):
    Y = D[D.src == "yr5"].copy(); Y["mo"] = Y.month.replace({"2026-10": "2026-09"})   # Oct = 2 days → folded into Sep
    M = D[(D.src == "master")].copy()
    months = sorted(Y.mo.unique())
    sl = Y.groupby("mo").pct.mean()
    P("\n## ③ Uniform-degradation test\n")
    P("### 3a · yr5 cohort × month (avg %/fill; N per seed in brackets); sleeve row first\n")
    rows = {"SLEEVE": {m: f"{sl[m]:+.3f} ({(Y.mo==m).sum()/3:.0f})" for m in months}}
    stats = []
    for t in TAGS:
        for lv, g in Y.groupby(t):
            if len(g) / 3 < 15:
                continue
            gm = g.groupby("mo").pct.agg(["mean", "count"])
            rows[f"{t}={lv}"] = {m: (f"{gm.loc[m,'mean']:+.3f} ({gm.loc[m,'count']/3:.0f})" if m in gm.index and gm.loc[m, "count"] >= 6 else "·") for m in months}
            rest = Y[Y[t] != lv].groupby("mo").pct.mean()
            ok = gm["count"] >= 6
            a, b = gm["mean"][ok], rest.reindex(gm.index[ok])
            r = np.corrcoef(a, b)[0, 1] if ok.sum() >= 4 else np.nan
            same = (np.sign(a) == np.sign(sl.reindex(a.index))).mean() if ok.sum() else np.nan
            h1 = g[g.half == "H1"].pct.mean(); h2 = g[g.half == "H2"].pct.mean()
            neg_m = sl[sl < 0].index
            loss_share = -g[g.mo.isin(neg_m)].pct.sum() / -Y[Y.mo.isin(neg_m)].pct.sum()
            fill_share = (g.mo.isin(neg_m)).sum() / Y.mo.isin(neg_m).sum()
            stats.append(dict(cohort=f"{t}={lv}", N_seed=len(g) / 3, avg=g.pct.mean(), WR=(g.pct > 0).mean() * 100, H1=h1, H2=h2,
                              r_vs_rest=r, months_same_sign_as_sleeve=same, months_used=int(ok.sum()), loss_share_neg_months=loss_share,
                              fill_share_neg_months=fill_share, avg_master_exB1=M[(M.era != "B1") & (M[t] == lv)].pct.mean(),
                              N_master=int(((M.era != "B1") & (M[t] == lv)).sum())))
    T = pd.DataFrame(rows).T; T.index.name = "cohort"
    P(md(T))
    S = pd.DataFrame(stats).set_index("cohort")
    S.to_csv("reports/study_ml_checklist_cohort_month.csv")
    P("### 3b · per cohort: co-movement with the rest of the sleeve across yr5 months, halves, loss share\n")
    P("r_vs_rest = Pearson r of the cohort's monthly avg vs the rest-of-sleeve monthly avg (months with ≥ 6 fills). "
      "loss/fill share = share of the sleeve's loss / fills in the negative months.\n")
    P(md(S, 3))
    # summary of co-movement
    good = S[S.months_used >= 6]
    P(f"cohorts with ≥ 6 months: {len(good)} · median r vs rest {good.r_vs_rest.median():+.2f} · r > 0: {int((good.r_vs_rest>0).sum())}/{len(good)} · "
      f"negative in BOTH halves: {int(((good.H1<0)&(good.H2<0)).sum())}/{len(good)} · positive in both: {int(((good.H1>0)&(good.H2>0)).sum())}/{len(good)}")
    # variance decomposition on cohort × month cell means (each tag separately)
    P("\n### 3c · two-way decomposition (fills): share of explained variance from MONTH vs COHORT vs interaction, per tag\n")
    rows = []
    rng = np.random.default_rng(4)
    for t in TAGS:
        d = Y[[t, "mo", "pct", "day"]].dropna(); d = d[d.groupby(t)[t].transform("size") >= 45]
        if d[t].nunique() < 2:
            continue
        gm = d.pct.mean(); tot = ((d.pct - gm) ** 2).sum()
        mo_eff = d.groupby("mo").pct.transform("mean"); co_eff = d.groupby(t).pct.transform("mean"); cell = d.groupby([t, "mo"]).pct.transform("mean")
        ss_mo = ((mo_eff - gm) ** 2).sum(); ss_co = ((co_eff - gm) ** 2).sum(); ss_cell = ((cell - gm) ** 2).sum()
        ss_int = max(ss_cell - ss_mo - ss_co, 0)
        # permutation: shuffle cohort labels within day (keeps the month/day structure) → null for cohort + interaction
        nul = []
        for _ in range(300):
            lab = d.groupby("day")[t].transform(lambda s: rng.permutation(s.values))
            ce = d.pct.groupby(lab).transform("mean"); cc = d.pct.groupby([lab, d.mo]).transform("mean")
            nul.append((((ce - gm) ** 2).sum(), max(((cc - gm) ** 2).sum() - ss_mo - ((ce - gm) ** 2).sum(), 0)))
        nul = np.array(nul)
        rows.append(dict(tag=t, levels=d[t].nunique(), month_pct=ss_mo / tot * 100, cohort_pct=ss_co / tot * 100, interaction_pct=ss_int / tot * 100,
                         p_cohort=(nul[:, 0] >= ss_co).mean(), p_interaction=(nul[:, 1] >= ss_int).mean()))
    V = pd.DataFrame(rows).set_index("tag"); P(md(V, 3))
    # live batches
    P("\n### 3d · live batches (master kept ex-probe, stack_pct) by cohort — N·WR·avg\n")
    M["batch"] = M.era
    order = ["BASE", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B12", "B13", "B14", "B15", "B16", "B17", "B18"]
    rows = {}
    for t in ["t_cell", "t_pvr", "t_negflank", "t_hot", "t_pattern", "t_regime"]:
        for lv, g in M.groupby(t):
            if len(g) < 8:
                continue
            rows[f"{t}={lv}"] = {b: fmt(g[g.batch == b]) for b in order}
    rows["SLEEVE kept"] = {b: fmt(M[M.batch == b]) for b in order}
    A = pd.read_csv("reports/study_ml_checklist_astraded.csv")
    rows["SLEEVE as-traded"] = {b: (f"{(A.era==b).sum()}·{(A[A.era==b].pct_traded>0).mean()*100:.0f}%·{A[A.era==b].pct_traded.mean():+.3f}" if (A.era == b).any() else "—") for b in order}
    T = pd.DataFrame(rows).T; T.index.name = "cohort"; P(md(T))
    return S, V


# ───────────────────────────── ④ tape context ─────────────────────────────
def tape_series():
    b = pd.read_csv("reports/backtest_cache/k5m_full/BTCUSDT.csv", usecols=["open_time", "h", "l", "c"]).drop_duplicates("open_time")
    b["t"] = pd.to_datetime(b.open_time, unit="ms"); b = b.set_index("t").sort_index()
    e = pd.read_csv("reports/backtest_cache/k5m_full/ETHUSDT.csv", usecols=["open_time", "c"]).drop_duplicates("open_time")
    e["t"] = pd.to_datetime(e.open_time, unit="ms"); e = e.set_index("t").sort_index().c
    h = b.c.resample("1h").last()
    d1 = b.c.resample("1D").last(); ema20 = d1.ewm(span=20, adjust=False).mean()
    eff = (h - h.shift(72)).abs() / h.diff().abs().rolling(72).sum()
    A = pd.read_pickle("reports/NEGFLANK_2D_altindex.pkl"); A.index = pd.to_datetime(A.index, unit="ms")
    g = pd.read_pickle("reports/backtest_cache/gvr_year.pkl"); g.index = pd.to_datetime(g.index, unit="ms")
    return dict(btc=b.c, eth=e, h=h, d1=d1, above=(d1 > ema20), eff=eff, alt=A, gvr=g.gvr, rv=(np.log(h).diff()))


def tape_for(TS, a, z):
    btc = TS["btc"][a:z]; eth = TS["eth"][a:z]
    if len(btc) < 12:
        return {}
    alt = TS["alt"][a:z]; day_alt = alt.alt_med_ret24.resample("1D").last().dropna()
    btc_d = TS["btc"].resample("1D").last().pct_change() * 100
    dom = (btc_d.reindex(day_alt.index) - day_alt).mean() if len(day_alt) else np.nan
    days = max((z - a).total_seconds() / 86400, 1e-9)
    return dict(days=days, btc_ret=(btc.iloc[-1] / btc.iloc[0] - 1) * 100, btc_ret_per_day=(btc.iloc[-1] / btc.iloc[0] - 1) * 100 / days,
                eth_ret=(eth.iloc[-1] / eth.iloc[0] - 1) * 100 if len(eth) else np.nan,
                btc_rv_daily=TS["rv"][a:z].std() * np.sqrt(24) * 100, btc_above_1d_ema20=TS["above"][a:z].mean() * 100 if len(TS["above"][a:z]) else np.nan,
                btc_eff72=TS["eff"][a:z].mean(), alt_med_ret24=alt.alt_med_ret24.mean(), alt_up_share=alt.alt_up_share24.mean() * 100,
                dom_proxy_daily=dom, gvr=TS["gvr"][a:z].mean(),
                btc_off30=((btc / TS["btc"].rolling(8640, min_periods=2000).max()[a:z] - 1) * 100).mean())


def item4(D):
    TS = tape_series()
    Y = D[D.src == "yr5"].copy(); Y["mo"] = Y.month.replace({"2026-10": "2026-09"})
    P("\n## ④ Tape context — winning vs losing periods\n")
    rows = []
    for mo, g in Y.groupby("mo"):
        a = pd.Timestamp(mo + "-01"); z = (a + pd.offsets.MonthBegin(1)) if mo != "2026-09" else pd.Timestamp("2026-10-03")
        if mo == "2026-01":
            a = pd.Timestamp("2026-01-04")
        r = dict(period=mo, ml_N_seed=len(g) / 3, ml_WR=(g.pct > 0).mean() * 100, ml_avg=g.pct.mean(),
                 ml_seed_min=g.groupby("seed").pct.mean().min(), ml_seed_max=g.groupby("seed").pct.mean().max())
        r.update(tape_for(TS, a, z)); rows.append(r)
    T = pd.DataFrame(rows).set_index("period"); T.to_csv("reports/study_ml_checklist_tape_month.csv")
    P("### 4a · yr5 months (ML at today's rules) vs tape\n"); P(md(T.drop(columns=["days"]), 2))
    num = [c for c in T.columns if c not in ("ml_N_seed", "ml_WR", "ml_avg", "ml_seed_min", "ml_seed_max", "days")]
    P("Spearman ρ across the 9 months, ML avg vs tape (descriptive; 9 points): " +
      "; ".join(f"{c} {T[c].rank().corr(T.ml_avg.rank()):+.2f}" for c in num))
    # live batches
    PM = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False, usecols=["era", "opened_at"])
    PM["o"] = pd.to_datetime(PM.opened_at.astype(str).str[:19].str.replace("T", " "))
    span = PM.groupby("era").o.agg(["min", "max"])
    BASE_SPAN = (pd.Timestamp("2026-06-15"), pd.Timestamp("2026-07-10 23:59"))
    span.loc["BASE"] = BASE_SPAN
    A = pd.read_csv("reports/study_ml_checklist_astraded.csv")
    M = D[(D.src == "master")]
    order = ["BASE", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18"]
    rows = []
    for e in order:
        if e not in span.index:
            continue
        a, z = span.loc[e, "min"], span.loc[e, "max"]
        k = M[M.era == e]; at = A[A.era == e]; yy = Y[(Y.o >= a) & (Y.o <= z)]
        r = dict(batch=e, start=str(a)[:10], end=str(z)[:10], kept_N=len(k), kept_avg=k.pct.mean() if len(k) else np.nan,
                 traded_N=len(at), traded_avg=at.pct_traded.mean() if len(at) else np.nan,
                 yr5_N_seed=len(yy) / 3, yr5_avg=yy.pct.mean() if len(yy) else np.nan)
        r.update(tape_for(TS, a, z)); rows.append(r)
    B = pd.DataFrame(rows).set_index("batch"); B.to_csv("reports/study_ml_checklist_tape_batch.csv")
    P("\n### 4b · live batches BASE…B18 vs tape (kept = today's stack on live fills; traded = as traded; yr5 = replay in the same hours)\n")
    P(md(B.drop(columns=["days"]), 2))
    bb = B[(B.traded_N >= 5)]
    P(f"Spearman ρ across batches with ≥ 5 as-traded fills (n={len(bb)}), as-traded avg vs tape: " +
      "; ".join(f"{c} {bb[c].rank().corr(bb.traded_avg.rank()):+.2f}" for c in num if c in bb))
    # day-level: yr5 day-mean ML vs day tape (the WINDOW-unit test of ④)
    dm = Y.groupby(Y.o.dt.floor("D")).pct.agg(["mean", "count"])
    btc_d = TS["btc"].resample("1D").last().pct_change() * 100
    alt_d = TS["alt"].alt_med_ret24.resample("1D").last()
    eff_d = TS["eff"].resample("1D").mean(); rv_d = TS["rv"].resample("1D").std() * np.sqrt(24) * 100
    X = pd.DataFrame({"ml": dm["mean"], "n": dm["count"], "btc_prevday_ret": btc_d.shift(1), "btc_ret_7d": TS["btc"].resample("1D").last().pct_change(7).shift(1) * 100,
                      "alt_prevday_med24": alt_d.shift(1), "dom_prevday": (btc_d - alt_d).shift(1), "btc_eff72_prevday": eff_d.shift(1), "btc_rv_prevday": rv_d.shift(1),
                      "btc_above_1d_ema20": TS["above"].shift(1).astype(float)}).dropna()
    P("\n### 4c · yr5 DAY-level (prior-day tape, known before the day starts; Spearman, circular day-shift null 2000)\n")
    rng = np.random.default_rng(8); rr = []
    for c in X.columns[2:]:
        x, y = X[c].rank().values, X.ml.rank().values; rho = np.corrcoef(x, y)[0, 1]; k = len(x)
        null = np.array([np.corrcoef(x, np.roll(y, s))[0, 1] for s in rng.integers(7, k - 7, 2000)])
        q = (X[c].map({0.0: "low", 1.0: "high"}) if X[c].nunique() <= 2 else pd.qcut(X[c], 3, labels=["low", "mid", "high"]))
        rr.append(dict(var=c, days=k, rho=rho, p=((np.abs(null) >= abs(rho)).sum() + 1) / 2001,
                       ml_low=X.ml[q == "low"].mean(), ml_mid=X.ml[q == "mid"].mean(), ml_high=X.ml[q == "high"].mean()))
    P(md(pd.DataFrame(rr).set_index("var"), 3))
    return T, B


# ───────────────────────────── candidates through the locked bar ─────────────────────────────
def bar(d, ns, be):
    s = C.stats(d.assign(unit=d.day, usd=d.usd_today), nseeds=ns)
    if s.get("N", 0) == 0:
        return s
    s["BE"] = be
    s["pass"] = (s["WR"] < be) and (s["p_neg"] >= 0.95) and (s["days"] >= 8) and (s["N"] >= 15) and (s["maxday"] < 50) and (s["maxpair"] < 50)
    return s


def candidates(D):
    Y = D[D.src == "yr5"].copy(); M = D[(D.src == "master") & (D.era != "B1")].copy()
    n = lambda s: pd.to_numeric(s, errors="coerce")
    for d in (Y, M):
        d["day"] = d.o.dt.strftime("%Y-%m-%d")
    CAND = {
        # block cohorts (pre-registered / existing lines)
        "NEGFLANK (BTC 1h slope ≤ −0.05) [pre-reg 10-05]": lambda d: d.t_negflank,
        "BTC_HOT_MATURE [pre-reg Jul-16]": lambda d: d.t_hot,
        # post-hoc coarsenings of this scan's top family (sign granularity, round cuts)
        "BTC 1h slope ≤ 0 (sign)": lambda d: n(d.k_btc_slope1h) <= 0,
        "BTC 24h return ≤ 0 (sign)": lambda d: n(d.k_btc_ret24h) <= 0,
        "BTC 4h slope ≤ 0 (sign)": lambda d: n(d.k_btc_4h_slope) <= 0,
        "NOT trend-aligned: ¬(BTC 1h slope > 0 ∧ pair 4h EMA20>EMA50)": lambda d: ~((n(d.k_btc_slope1h) > 0) & (n(d.k_pair_gap4h_20_50) > 0)),
        "NOT trend-aligned-2: ¬(BTC 1h > 0 ∧ BTC 4h > 0 ∧ pair 4h > 0)": lambda d: ~((n(d.k_btc_slope1h) > 0) & (n(d.k_btc_4h_slope) > 0) & (n(d.k_pair_gap4h_20_50) > 0)),
        "BTC eff72 ≤ 0.21 (all but top quintile, post-hoc)": lambda d: n(d.k_btc_eff72) <= 0.211,
        "alts out-performing BTC 24h (dom proxy ≤ 0)": lambda d: n(d.k_btc_dom24) <= 0,
    }
    P("\n## Candidates through the locked expectancy bar (BLOCK cohort judged; yr5 N per seed; day-clustered)\n")
    rows = []
    for lab, f in CAND.items():
        for coh, d, ns in (("yr5 all", Y, 3), ("yr5 ex-washed", Y[~Y.washed], 3), ("master ex-B1", M, 1), ("master ex-B1 ex-washed", M[~M.washed], 1)):
            be = C.be_wr(d)
            z = f(d).fillna(False).astype(bool)
            blk, keep = d[z], d[~z]
            s = bar(blk, ns, be)
            h = {h_: blk[blk.half == h_].pct.mean() for h_ in ("H1", "H2")} if coh.startswith("yr5") else {}
            seeds = blk.groupby("seed").pct.mean().round(3).to_dict() if coh.startswith("yr5") else {}
            mo = blk.assign(mo=blk.month).groupby("mo").pct.mean()
            kmo = keep.assign(mo=keep.month).groupby("mo").pct.mean()
            rows.append(dict(candidate=lab, cohort=coh, N=s.get("N"), WR=s.get("WR"), BE=be, avg=s.get("avg"), p_neg=s.get("p_neg"),
                             days=s.get("days"), maxday=s.get("maxday"), maxpair=s.get("maxpair"), PASS=s.get("pass"),
                             keep_N=len(keep) / ns, keep_avg=keep.pct.mean(), keep_p_pos=1 - C.boot_day(keep.assign(unit=keep.day))[2],
                             blk_H1=h.get("H1"), blk_H2=h.get("H2"), seeds=seeds,
                             months_blk_lt_keep=f"{int((mo.reindex(kmo.index) < kmo).sum())}/{int(mo.reindex(kmo.index).notna().sum())}"))
    R = pd.DataFrame(rows); R.to_csv("reports/study_ml_checklist_candidates.csv", index=False)
    P(md(R.set_index("candidate"), 3))
    return R


# ───────────────────────────── sizing angle ─────────────────────────────
def maxdd(x):
    c = np.cumsum(x); peak = np.maximum.accumulate(np.concatenate([[0], c]))[1:]
    return (c - peak).min()


def sizing(D):
    Y = D[D.src == "yr5"].sort_values("o"); M = D[(D.src == "master") & (D.era != "B1")].sort_values("o")
    P("\n## Sizing angle (pure risk control, not a verdict): ML-only $ on a FIXED $3,000 book, BASE_NOTIONAL_FRAC 4.875 (yr5 convention)\n")
    rows = []
    for lab, col in (("1× everywhere", "usd1"), ("yr5 as-run (UNMATCHED / CALM3D 2×)", "usd_asrun"), ("TODAY (UNMATCHED 1.5×, de-mux 1×, CALM3D 1×)", "usd_today")):
        for s, g in Y.groupby("seed"):
            rows.append(dict(sizing=lab, cohort=f"yr5 seed {s}", N=len(g), sum_usd=g[col].sum(), maxdd_usd=maxdd(g[col].values),
                             worst_month=g.groupby("month")[col].sum().min(), H1=g[g.half == "H1"][col].sum(), H2=g[g.half == "H2"][col].sum()))
        for coh, g in (("master ex-B1", M), ("master ex-B1 ex-washed", M[~M.washed])):
            rows.append(dict(sizing=lab, cohort=coh, N=len(g), sum_usd=g[col].sum(), maxdd_usd=maxdd(g[col].values),
                             worst_month=g.groupby("month")[col].sum().min(), H1=np.nan, H2=np.nan))
    for frac in (0.5,):
        for s, g in Y.groupby("seed"):
            rows.append(dict(sizing=f"TODAY × {frac}", cohort=f"yr5 seed {s}", N=len(g), sum_usd=g.usd_today.sum() * frac, maxdd_usd=maxdd(g.usd_today.values * frac),
                             worst_month=g.groupby("month").usd_today.sum().min() * frac, H1=g[g.half == "H1"].usd_today.sum() * frac, H2=g[g.half == "H2"].usd_today.sum() * frac))
    S = pd.DataFrame(rows); S.to_csv("reports/study_ml_checklist_sizing.csv", index=False)
    agg = S.assign(grp=np.where(S.cohort.str.startswith("yr5"), "yr5 (mean of 3 seeds)", S.cohort)).groupby(["sizing", "grp"])[["N", "sum_usd", "maxdd_usd", "worst_month", "H1", "H2"]].mean()
    P(md(agg.reset_index().set_index("sizing"), 0))
    # share of today's yr5 $ by cell size and the avg % at each size
    g = Y.groupby("m_today").agg(N=("pct", "size"), avg=("pct", "mean"), usd=("usd_today", "sum"))
    g["N"] /= 3; g["usd"] /= 3
    P("yr5 by TODAY's size multiplier (per seed):\n"); P(md(g, 3))
    g = M.groupby("m_today").agg(N=("pct", "size"), avg=("pct", "mean"), usd=("usd_today", "sum"))
    P("master ex-B1 by TODAY's size multiplier:\n"); P(md(g, 3))
    return S


def main():
    D = SC.load()
    P("# MOMENTUM LONG checklist — tables (scripts/study_ml_checklist_cohorts.py)\n")
    item3(D); item4(D); candidates(D); sizing(D)
    open("reports/study_ml_checklist_tables.md", "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
