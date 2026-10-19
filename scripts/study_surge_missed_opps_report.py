#!/usr/bin/env python3
"""⚡ SURGE missed-opps study — tables (reads the csvs written by study_surge_missed_opps.py sim). Read-only.
Statistics: per-fill avg % (net of fees, TODAY's SURGE exit), WR, Σ$ at today's live size, day-block bootstrap (resample UTC days of the
trigger; 5,000 draws), per period (2025 BULL · 2026-H1 BEAR · 2026-H2 BULL), evidence counted in WINDOWS (triggers that filled).
Every relaxation is read on its INCREMENTAL cohort (fills the variant adds) and on the paired Δ vs TODAY (adds AND removes)."""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
USD = 22.1
CUT = pd.Timestamp("2026-10-05").value // 10**6
H1 = pd.Timestamp("2026-01-04").value // 10**6; H2 = pd.Timestamp("2026-07-01").value // 10**6


def period(t):
    return np.where(t < H1, "2025 BULL", np.where(t < H2, "2026-H1 BEAR", "2026-H2 BULL"))


def boot_mean(x, days, n=5000, seed=7):
    x = np.asarray(x, float); days = np.asarray(days)
    u = np.unique(days)
    if len(u) < 3:
        return np.nan, np.nan, np.nan
    idx = {d: np.where(days == d)[0] for d in u}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.choice(u, len(u))
        ii = np.concatenate([idx[d] for d in s])
        out.append(x[ii].mean())
    out = np.array(out)
    return np.percentile(out, 2.5), np.percentile(out, 97.5), (out > 0).mean()


def boot_delta(dv, n=5000, seed=7):
    """dv: per-day Σ Δ% (variant − TODAY). → CI of the total Δ (sum over days resampled) and P(Δ>0)."""
    x = np.asarray(dv, float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    s = np.random.default_rng(seed).choice(x, (n, len(x))).sum(axis=1)
    return np.percentile(s, 2.5), np.percentile(s, 97.5), (s > 0).mean()


def line(g, label):
    if not len(g):
        return dict(cohort=label, N=0)
    lo, hi, p = boot_mean(g.pnl.values, (g.trig // 86_400_000).values)
    w = g[g.pnl > 0].pnl; l = g[g.pnl <= 0].pnl
    be = abs(l.mean()) / (w.mean() + abs(l.mean())) * 100 if len(w) and len(l) else np.nan
    return dict(cohort=label, N=len(g), windows=g.trig.nunique(), days=(g.trig // 86_400_000).nunique(), WR=round((g.pnl > 0).mean() * 100, 1),
                BE_WR=round(be, 1), avg=round(g.pnl.mean(), 3), ci=f"[{lo:+.2f}, {hi:+.2f}]", P_gt0=round(p, 3), sum_pct=round(g.pnl.sum(), 1),
                usd=round(g.pnl.sum() * USD), top_pair_share=_top_share(g, "pair"), top_day_share=_top_share(g.assign(d=g.trig // 86_400_000), "d"),
                **{f"avg {k}": round(v.pnl.mean(), 3) for k, v in g.groupby("period")}, **{f"N {k}": len(v) for k, v in g.groupby("period")})


def _top_share(g, col):
    s = g.groupby(col).pnl.sum()
    tot = s.sum()
    if tot == 0:
        return np.nan
    big = s.max() if tot > 0 else s.min()
    return round(big / tot, 2)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:+.3f}" if isinstance(v, float) and abs(v) < 10 and not float(v).is_integer() else str(v)) for v in r) + " |")
    return "\n".join(out)


def main():
    F = pd.read_csv(os.path.join(REP, "SURGE_MISSED_OPPS_fills.csv"))
    F = F[F.trig < CUT]   # backtest block ends where the batch / live block (study_surge_missed_live.py) starts — disjoint blocks
    F["period"] = period(F.trig.values)
    print("src mix:", F.groupby("variant").src.value_counts().unstack(fill_value=0).to_string())
    T = F[F.variant == "TODAY"]
    rows, inc, dl = [], [], []
    key = lambda d: set(zip(d.pair, d.t_entry))
    tk = key(T)
    for v, g in F.groupby("variant", sort=False):
        rows.append(dict(variant=v, **line(g, v)))
        if v == "TODAY":
            continue
        gk = key(g)
        add = g[[k not in tk for k in zip(g.pair, g.t_entry)]]
        rem = T[[k not in gk for k in zip(T.pair, T.t_entry)]]
        inc.append(dict(variant=v, **line(add, "ADDED")))
        if len(rem):
            inc.append(dict(variant=v, **line(rem, "REMOVED (displaced)")))
        # paired Δ per day
        d = pd.concat([g.assign(s=1), T.assign(s=-1)])
        d["day"] = d.trig // 86_400_000
        per = d.groupby("day").apply(lambda x: (x.pnl * x.s).sum(), include_groups=False)
        lo, hi, p = boot_delta(per.values)
        dp = {}
        for k in ("2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"):
            dp[f"Δ$ {k}"] = round(((g[g.period == k].pnl.sum()) - (T[T.period == k].pnl.sum())) * USD)
        dl.append(dict(variant=v, added=len(add), removed=len(rem), d_sum_pct=round(g.pnl.sum() - T.pnl.sum(), 1),
                       d_usd=round((g.pnl.sum() - T.pnl.sum()) * USD), d_usd_ci=f"[{lo * USD:+,.0f}, {hi * USD:+,.0f}]", P_gt0=round(p, 3), **dp))
    R = pd.DataFrame(rows); I = pd.DataFrame(inc); D = pd.DataFrame(dl)
    R.to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_variants.csv"), index=False)
    I.to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_incremental.csv"), index=False)
    D.to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_delta.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(R.to_string()); print(I.to_string()); print(D.to_string())
    # per-trigger fill-count distribution (TODAY) and pass counts at triggers
    U = pd.read_csv(os.path.join(REP, "SURGE_MISSED_OPPS_universe.csv"))
    U = U[U.trig < CUT]
    Ut = U[U.status == "TRIGGER"]
    pc = Ut.groupby("trig").pair_status.value_counts().unstack(fill_value=0)
    print("per-trigger status means:\n", pc.mean().round(2))
    print("PASS count distribution:\n", pc.get("PASS", pd.Series(0, index=pc.index)).value_counts().sort_index())
    print("TODAY fills per filled trigger:\n", T.groupby("trig").size().value_counts().sort_index())
    print("spaced-out triggers:", (U.status == "SPACED").sum())
    # refused cohorts priced
    Rf = pd.read_csv(os.path.join(REP, "SURGE_MISSED_OPPS_refused.csv"))
    Rf["period"] = period(Rf.trig.values); Rf = Rf[np.isfinite(Rf.pnl) & (Rf.trig < CUT)]
    rr = []
    for k, g in [("NOT_LEADER (ATR≥1.5, gap≥0? any)", Rf[Rf.pair_status == "NOT_LEADER"]),
                 ("NOT_LEADER ∧ gap≥0 ∧ pmove>0", Rf[(Rf.pair_status == "NOT_LEADER") & (Rf.gap >= 0) & (Rf.pmove > 0)]),
                 ("EMA_GAP (leader, gap<0)", Rf[Rf.pair_status == "EMA_GAP"]),
                 ("ATR_LOW 1.0–1.5 ∧ leader ∧ gap≥0", Rf[(Rf.pair_status == "ATR_LOW") & (Rf.pmove > Rf.btc_move) & (Rf.gap >= 0)]),
                 ("ATR_LOW 1.0–1.5 (any)", Rf[Rf.pair_status == "ATR_LOW"])]:
        rr.append(line(g, k))
    print(pd.DataFrame(rr).to_string())
    pd.DataFrame(rr).to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_refused_summary.csv"), index=False)
    Sp = pd.read_csv(os.path.join(REP, "SURGE_MISSED_OPPS_spaced.csv"))
    Sp["period"] = period(Sp.trig.values); Sp = Sp[np.isfinite(Sp.pnl) & (Sp.trig < CUT)]
    print("SPACED-OUT would-be picks:", pd.DataFrame([line(Sp, "spaced picks")]).to_string())


if __name__ == "__main__":
    main()
