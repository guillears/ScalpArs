#!/usr/bin/env python3
"""WILLY late joiners (2026-10-22) — ANALYSIS: Q1 counts, Q2 outcomes per rule (R0 / R1-30 / R1-60 / R2) and routing, Q3 cohort contrast,
lag-matched null, slot-aware trades added, the pre-registered bar. Reads data/episodes.csv, _entries_priced.csv, _live.csv,
_onTime_live.csv. Writes reports/WILLY_LATE_JOIN_2026-10-22_tables.md + _cohort_stats.csv + _q1_lateness.csv (the .md report is
written by hand from these tables). Usage: venv/bin/python scripts/study_willy_latejoin_report.py
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_latejoin_common as C  # noqa: E402

DONE = ("TP", "LATE_TP", "STOP", "CAP")
OUT = []


def P(s=""):
    OUT.append(s); print(s)


def md(df, index=True, floatfmt=".3f"):
    """minimal markdown table (no tabulate dependency)."""
    df = df.to_frame() if isinstance(df, pd.Series) else df
    if index:
        df = df.reset_index()
    f = lambda v: (format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v)))  # noqa: E731
    h = [str(c) for c in df.columns]
    return "\n".join(["| " + " | ".join(h) + " |", "|" + "---|" * len(h)] + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def stats(d, label):
    d = d[d.why.isin(DONE)]
    if not len(d):
        return dict(cohort=label, N=0)
    x = d.pct.values; days = d.day.values
    lo, hi = C.day_boot(x, days)
    ps = d.groupby("pair").pct.sum().sort_values(ascending=False)
    pos = ps[ps > 0].sum()
    top2 = ps.index[:2]
    ex2 = d[~d.pair.isin(top2)].pct
    q = np.quantile(x, 0.95)
    w = np.where(d.route == "HT", C.HT_FRAC, 1.0)
    out = dict(cohort=label, N=len(d), days=d.day.nunique(), pairs=d.pair.nunique(), WR=(x > 0).mean() * 100, avg=x.mean(),
               med=np.median(x), p5=np.quantile(x, 0.05), worst=x.min(), stop_pct=(d.why == "STOP").mean() * 100,
               cap_loss_pct=((d.why == "CAP") & (d.pct < 0)).mean() * 100, ci_lo=lo, ci_hi=hi,
               avg_dollar_w=(x * w).sum() / w.sum(), sum_pct_w=(x * w).sum(),
               top2_share_pos=(ps.iloc[:2].clip(lower=0).sum() / pos * 100) if pos > 0 else np.nan, top2=",".join(top2),
               avg_ex_top2=ex2.mean() if len(ex2) else np.nan, avg_ex_best5=x[x < q].mean() if len(x) > 5 else np.nan,
               min_spike_entry_med=float(np.nanmedian(d.min_spike_to_entry)) if "min_spike_to_entry" in d else np.nan,
               n_full=int((d.route == "FULL").sum()), n_ht=int((d.route == "HT").sum()))
    for e in ("2025", "2026-H1", "2026-H2"):
        de = d[d.era == e]
        out[f"avg_{e}"] = de.pct.mean() if len(de) else np.nan; out[f"N_{e}"] = len(de)
    for r in ("FULL", "HT"):
        dr = d[d.route == r]
        out[f"avg_{r}"] = dr.pct.mean() if len(dr) else np.nan
    return out


def fmt(rows, cols):
    df = pd.DataFrame(rows)
    cols = [c for c in cols if c in df]
    return md(df[cols], index=False)


if __name__ == "__main__":
    ep = pd.read_csv(os.path.join(C.D, "episodes.csv"))
    en = pd.read_csv(C.OUTP + "_entries_priced.csv")
    ep["block"] = [C.block(x) for x in ep.spike_close]; ep["era"] = [C.era(x) for x in ep.spike_close]
    weeks = (ep.spike_close.max() - ep.spike_close.min()) / (7 * C.DAY)

    # ── Q1 ──
    P("## Q1 — lateness L (bars from the spike bar to the first shortlisted-while-flagged bar), all walk episodes ≥ 2025-01-01")
    b = pd.cut(ep.L, [-1, 0, 3, 6, 12, 36, 288, 10**6], labels=["0 (on list at spike)", "1-3 (R0 late A)", "4-6", "7-12", "13-36", "37-288", ">288"])
    t = pd.crosstab(b, ep.era, margins=True)
    P(md(t)); P()
    P("why late (which shortlist leg failed on the spike bar), L 4-12:")
    P(md(ep[(ep.L >= 4) & (ep.L <= 12)].why_late.value_counts())); P()
    q1 = pd.DataFrame(dict(bucket=b.astype(str), era=ep.era)).value_counts().rename("n").reset_index()
    q1.to_csv(C.OUTP + "_q1_lateness.csv", index=False)
    n = len(ep); n412 = ((ep.L >= 4) & (ep.L <= 12)).sum(); n46 = ((ep.L >= 4) & (ep.L <= 6)).sum()
    P(f"episodes {n} · L≤3 {int((ep.L <= 3).sum())} ({(ep.L <= 3).mean() * 100:.1f}%) · L 4-6 {n46} ({n46 / n * 100:.1f}%) · L 4-12 {n412} "
      f"({n412 / n * 100:.1f}%) · L>12 {int((ep.L > 12).sum())} · weeks {weeks:.1f} → L 4-12 per week {n412 / weeks:.1f}")
    P("pump size (max high in the 65 min from the spike bar / base − 1, %), median: " +
      " · ".join(f"{k} {ep[m].pump_1h.median():.1f}" for k, m in (("L≤3", ep.L <= 3), ("L4-12", (ep.L >= 4) & (ep.L <= 12)), ("L>12", ep.L > 12))))
    P()

    # ── Q2 / Q3 ──
    en = en[en.st == "ok"].copy()
    rows = []
    for blk, bm in (("ALL walk", en.block.notna()), ("③ backtest", en.block == "3_backtest"), ("② master (walk)", en.block == "2_master"),
                    ("① live (walk)", en.block == "1_live")):
        for rule in ("R0", "R1-30", "R1-60", "R2"):
            d = en[bm & (en.rule == rule) & en.route.isin(["FULL", "HT"])]
            rows.append(dict(block=blk, rule=rule, **{k: v for k, v in stats(d, rule).items() if k != "cohort"}))
    S = pd.DataFrame(rows); S.to_csv(C.OUTP + "_cohort_stats.csv", index=False)
    P("## Q2/Q3 — routed fills (FULL R<1 + HT R≥1; UNREAD excluded). R0 = on-time; R1/R2 = the late fills each rule ADDS")
    P(fmt(rows, ["block", "rule", "N", "days", "pairs", "WR", "avg", "med", "p5", "worst", "stop_pct", "ci_lo", "ci_hi", "avg_2025",
                 "avg_2026-H1", "avg_2026-H2", "n_full", "avg_FULL", "n_ht", "avg_HT", "avg_dollar_w", "min_spike_entry_med"])); P()
    P("concentration / tails (ALL walk):")
    P(fmt([r for r in rows if r["block"] == "ALL walk"], ["rule", "N", "top2", "top2_share_pos", "avg_ex_top2", "avg_ex_best5", "cap_loss_pct",
                                                          "N_2025", "N_2026-H1", "N_2026-H2"])); P()
    P("routing incl. UNREAD (ALL walk, red bar found):")
    P(md(pd.crosstab(en.rule, en.route))); P()
    P("not priced (why):"); P(md(pd.crosstab(en.rule, en.why.fillna("NA")))); P()

    # ── lag-matched null ──
    late = en[(en.rule == "R1-60") & en.route.isin(["FULL", "HT"]) & en.why.isin(DONE)]
    wL = late.L.value_counts(normalize=True)
    nl = []
    for Lp in range(4, 13):
        d = en[(en.rule == f"NULL{Lp}") & en.route.isin(["FULL", "HT"]) & en.why.isin(DONE)]
        nl.append(dict(L=Lp, N=len(d), WR=(d.pct > 0).mean() * 100, avg=d.pct.mean(), late_N=int((late.L == Lp).sum()),
                       late_avg=late[late.L == Lp].pct.mean()))
    NL = pd.DataFrame(nl)
    P("## Fair null — on-time episodes re-armed at spike + L bars (lag-matched) vs the real late joiners at the same L (R1-60 rule)")
    P(md(NL, index=False))
    P(f"lag-matched null avg (weights = the late joiners' L mix): {(NL.set_index('L').avg * wL).sum():+.3f} · real late joiners "
      f"{late.pct.mean():+.3f} · on-time R0 {en[(en.rule == 'R0') & en.route.isin(['FULL', 'HT']) & en.why.isin(DONE)].pct.mean():+.3f}")
    P()

    # ── slots + per week ──
    P("## Trades added per week (ALL walk) — unconstrained and with the 1 WILLY slot + 1 HT slot taken by earlier fills")
    on = en[(en.rule == "R0") & en.route.isin(["FULL", "HT"]) & en.why.isin(DONE)].copy()
    def slot_sim(df):
        df = df.copy(); df["te"] = df.red_close + C.DELAY_MS; df["tx"] = df.te + df.xmin * C.MIN
        df = df.sort_values("te"); busy = {"FULL": 0, "HT": 0}; kept = []
        for r in df.itertuples():
            ok = r.te >= busy[r.route]
            kept.append(ok)
            if ok:
                busy[r.route] = r.tx
        df["kept"] = kept
        return df
    base = slot_sim(on.assign(late=False)); base_on = int(base.kept.sum())
    for rule in ("R1-30", "R1-60", "R2"):
        lt = en[(en.rule == rule) & en.route.isin(["FULL", "HT"]) & en.why.isin(DONE)].copy()
        allf = slot_sim(pd.concat([on.assign(late=False), lt.assign(late=True)]))
        k = allf[allf.late & allf.kept]; on_k = allf[~allf.late & allf.kept]
        lost = base[base.kept & ~base.set_index(["pair", "red_close"]).index.isin(on_k.set_index(["pair", "red_close"]).index)]
        P(f"{rule}: added {len(lt)} ({len(lt) / weeks:.2f}/wk) · after slots {len(k)} ({len(k) / weeks:.2f}/wk) avg {k.pct.mean():+.3f} · "
          f"on-time fills displaced {base_on - len(on_k)} (their avg {lost.pct.mean() if len(lost) else float('nan'):+.3f}) · "
          f"net Σ change {k.pct.sum() - lost.pct.sum():+.1f} pts")
    P()

    # ── robustness: crash day out, gap-capped, walker calibration vs the 10-19 tick re-price ──
    P("## Robustness (ALL walk, routed): without the 2025-10-10 crash day · stop gaps capped at −4 · per-route")
    rr = []
    for rule in ("R0", "R1-30", "R1-60", "R2"):
        d = en[(en.rule == rule) & en.route.isin(["FULL", "HT"]) & en.why.isin(DONE)]
        dx = d[d.day != "2025-10-10"]
        lo, hi = C.day_boot(dx.pct.values, dx.day.values)
        rr.append(dict(rule=rule, N=len(d), avg=d.pct.mean(), N_ex_crash=len(dx), avg_ex_crash=dx.pct.mean(), ci_lo_ex=lo, ci_hi_ex=hi,
                       avg_gapcap=d.pct.clip(lower=-C.SL).mean(), H1_ex=dx[dx.era == "2026-H1"].pct.mean(), H2_ex=dx[dx.era == "2026-H2"].pct.mean(),
                       full_ex=dx[dx.route == "FULL"].pct.mean(), ht_ex=dx[dx.route == "HT"].pct.mean()))
    P(md(pd.DataFrame(rr), index=False)); P()
    tp = os.path.join(C.R, "reports", "WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv")
    if os.path.exists(tp):
        t = pd.read_csv(tp); t = t[t.st == "ok"]
        r0 = en[(en.rule == "R0") & en.why.isin(DONE)]
        m = r0.merge(t[["pair", "t_red", "pct", "why"]].rename(columns={"t_red": "red_close", "pct": "pct_tick", "why": "why_tick"}), on=["pair", "red_close"])
        P(f"walker calibration — on-time fills also in the 10-19 tick re-price (same red bar): N {len(m)} · this 1m walker {m.pct.mean():+.3f} vs "
          f"tick walker {m.pct_tick.mean():+.3f} (Δ {m.pct.mean() - m.pct_tick.mean():+.3f}; 0.065 of it is the 0.10 vs 0.035 slip) · same exit "
          f"{(m.why == m.why_tick).mean() * 100:.0f}% · 1m STOP where ticks TP: {int(((m.why == 'STOP') & (m.why_tick == 'TP')).sum())}")
        P()
    # ── live / master (engine refusals) ──
    fl = C.OUTP + "_live.csv"
    if os.path.exists(fl):
        lv = pd.read_csv(fl)
        P("## ①② the engine's OWN refusals ('late window — no entry A', CloudWatch 10-08 → 10-22, L ≤ 12), priced")
        P(md(lv[[c for c in ["block", "pair", "spike_close", "L", "rule", "st", "wait_bars", "min_spike_to_entry", "R", "route", "why", "pct"] if c in lv]]
          .assign(spike=lambda x: pd.to_datetime(x.spike_close, unit="ms")).drop(columns="spike_close"), index=False))
        for blk in ("2_master", "1_live"):
            for rule in ("R1-30", "R1-60", "R2"):
                d = lv[(lv.block == blk) & (lv.rule == rule) & lv.route.isin(["FULL", "HT"]) & lv.why.isin(DONE)]
                P(f"{blk} {rule}: N {len(d)} · WR {(d.pct > 0).mean() * 100 if len(d) else np.nan:.0f}% · avg {d.pct.mean() if len(d) else np.nan:+.3f} · Σ {d.pct.sum():+.2f}")
        P()
    fo = C.OUTP + "_onTime_live.csv"
    if os.path.exists(fo):
        ot = pd.read_csv(fo)
        P("## ①② on-time comparator — real FRENZY_WILLY fills (master), booked vs this walker")
        for blk, g in ot.groupby("block"):
            g2 = g[g.walk_why.isin(DONE)]
            P(f"{blk}: N {len(g)} · booked avg {g.booked_pct.mean():+.3f} · stack avg {g.stack_pct.mean():+.3f} · walker avg {g2.walk_pct.mean():+.3f} "
              f"(N {len(g2)}) · |walk − stack| ≤ 0.10 on {(abs(g2.walk_pct - g2.stack_pct) <= 0.10).mean() * 100:.0f}% · same sign {(np.sign(g2.walk_pct) == np.sign(g2.stack_pct)).mean() * 100:.0f}%")
        P()
    # ── parity: walk late joiners vs CloudWatch refusals ──
    if os.path.exists(os.path.join(C.D, "cw_refusals_parsed.csv")):
        cw = pd.read_csv(os.path.join(C.D, "cw_refusals_parsed.csv"))
        lo_, hi_ = cw.log_ms.min(), ep.spike_close.max()
        wk = ep[(ep.spike_close >= lo_ - 3600_000) & (ep.spike_close <= hi_) & (ep.L >= 4) & (ep.L <= 12)]
        cwk = cw[(cw.L <= 12) & (cw.spike_close <= hi_)]
        m = wk.merge(cwk, on=["pair", "spike_close"], how="outer", suffixes=("_walk", "_cw"), indicator=True)
        P("## Parity — walk late joiners (L 4-12) vs the engine's CloudWatch refusals (L ≤ 12), same window")
        P(md(m[["pair", "spike_close", "L_walk", "L_cw", "_merge"]].assign(spike=lambda x: pd.to_datetime(x.spike_close, unit="ms"))
          .drop(columns="spike_close"), index=False))
        P()
    open(C.OUTP + "_tables.md", "w").write("\n".join(OUT))
