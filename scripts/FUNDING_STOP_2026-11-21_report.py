#!/usr/bin/env python3
"""FUNDING_STOP_2026-11-21 — tables for the report (reads real_rows.csv + bt_{FB,EXH,DIST}.csv written by the _real / _bt scripts).
Output: reports/FUNDING_STOP_2026-11-21_tables.md (pasted into reports/FUNDING_STOP_2026-11-21.md) + printed.
One size: X = 0.61 × equity per fill (DL 503; the TP_GRID_2026-11-20 size for all three sleeves); "$" = Δ % × 61 = dollars on a $10,000 account.
Usage: venv/bin/python -I scripts/FUNDING_STOP_2026-11-21_report.py"""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = importlib.util.spec_from_file_location("FSC", os.path.join(ROOT, "scripts", "FUNDING_STOP_2026-11-21_common.py"))
C = importlib.util.module_from_spec(sp); sp.loader.exec_module(C)
D = os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data")
X_SIZE, USD = 0.61, 61.0
FUND_THR = 0.00005
L = []


def P(s=""):
    print(s); L.append(s)


def prep(df):
    df = df.copy()
    df["day"] = pd.to_datetime(df.te, unit="ms").dt.strftime("%Y-%m-%d")
    for a in ("A", "B", "Bp"):
        df[f"{a}_day"] = pd.to_datetime(df[f"{a}_xt"], unit="ms").dt.strftime("%Y-%m-%d")
    df["paid"] = -df.A_fund
    return df


def load():
    R = prep(pd.read_csv(os.path.join(D, "real_rows.csv")))
    R["erate"] = R.entry_rate.where(R.entry_rate.notna(), R.rate_at_entry)
    out = {}
    fb = R[R.sleeve == "FLATBASE_SHORT"]
    out[("FLATBASE_SHORT", "real")] = fb[~(fb.entry_rate > FUND_THR)]
    out[("FLATBASE_SHORT", "real (stack_keep, before DL 601)")] = fb
    out[("EXHAUSTION_SHORT", "real")] = R[R.sleeve == "EXHAUSTION_SHORT"]
    out[("DISTRIBUTION_SHORT", "real")] = R[R.sleeve == "DISTRIBUTION_SHORT"]
    for s, nm in (("FB", "FLATBASE_SHORT"), ("EXH", "EXHAUSTION_SHORT"), ("DIST", "DISTRIBUTION_SHORT")):
        B = prep(pd.read_csv(os.path.join(D, f"bt_{s}.csv")))
        B["erate"] = B.entry_rate
        if s == "FB":
            B = B[~(B.entry_rate > FUND_THR)]
        out[(nm, "backtest")] = B
    return out


def funding_table(S, name):
    rows = []
    for lab, g in S:
        p = g.paid
        r = dict(cohort=lab, N=len(g), hourly_at_entry_pct=(g.interval_h == 1).mean() * 100, paying_pct=(p > 1e-9).mean() * 100,
                 paid_median=p.median(), paid_p90=p.quantile(0.9), paid_worst=p.max(), received_best=-p.min())
        for w in ("TP", "STOP", "TIME"):
            m = g.A_why == w
            r[f"{w} N"] = int(m.sum()); r[f"{w} paid avg"] = p[m].mean() if m.any() else np.nan
        ok = g.erate.notna()
        r["corr(entry rate, received)"] = np.corrcoef(g.erate[ok], g.A_fund[ok])[0, 1] if ok.sum() > 2 and g.erate[ok].std() > 0 else np.nan
        r["spearman"] = pd.Series(g.erate[ok].values).rank().corr(pd.Series(g.A_fund[ok].values).rank()) if ok.sum() > 2 else np.nan
        rows.append(r)
    P(f"\n**{name} — funding paid (% of notional; + = paid by the short) under today's exit (arm A)**\n")
    P(C.md(pd.DataFrame(rows), 2))


def span(g, arm):
    d0 = g.day.min(); d1 = max(g[f"{arm}_day"].max(), g.A_day.max())
    return d0, d1


def arms_table(blocks, name):
    rows, cls_rows = [], []
    for lab, g, d0, d1 in blocks:
        if not len(g):
            continue
        dcrA, ddA = C.day_dcr(g.A_net, g.A_day, d0, d1, X_SIZE)
        rows.append(dict(block=lab, arm="A (today)", N=len(g), WR=(g.A_net > 0).mean() * 100, avg=g.A_net.mean(), sum=g.A_net.sum(),
                         DCR=dcrA, maxDD=ddA, d_avg=np.nan, d_lo=np.nan, d_hi=np.nan, P_gain=np.nan, loo_min=np.nan, loo_max=np.nan))
        for arm in ("B", "Bp"):
            dcr, dd = C.day_dcr(g[f"{arm}_net"], g[f"{arm}_day"], d0, d1, X_SIZE)
            dl = (g[f"{arm}_net"] - g.A_net).values
            m, lo, hi, pg = C.cluster_boot(dl, g.day.values)
            l0, l1 = C.loo(dl)
            rows.append(dict(block=lab, arm="B net" if arm == "B" else "Bp paid-only", N=len(g), WR=(g[f"{arm}_net"] > 0).mean() * 100,
                             avg=g[f"{arm}_net"].mean(), sum=g[f"{arm}_net"].sum(), DCR=dcr, maxDD=dd, d_avg=m, d_lo=lo, d_hi=hi, P_gain=pg,
                             loo_min=l0, loo_max=l1))
            cl = np.array([C.classify(dict(net=a, why=aw, j=ax), dict(net=b, why=bw, j=bx))
                           for a, aw, ax, b, bw, bx in zip(g.A_net, g.A_why, g.A_xt, g[f"{arm}_net"], g[f"{arm}_why"], g[f"{arm}_xt"])])
            # exactness: identical (reason, exit bar) on a non-stop → Δ must be exactly 0
            same_path = (g.A_why.values == g[f"{arm}_why"].values) & (g.A_xt.values == g[f"{arm}_xt"].values) & (g.A_why.values != "STOP")
            viol = int((np.abs(dl[same_path]) > 0).sum())
            r = dict(block=lab, arm=arm, unchanged_N=int((cl == "SAME").sum()), unchanged_max_abs_d=float(np.abs(dl[cl == "SAME"]).max()) if (cl == "SAME").any() else 0.0,
                     exactness_violations=viol)
            for k in ("SAVED", "CUT", "WIDENED", "OTHER"):
                mm = cl == k
                r[f"{k} N"] = int(mm.sum()); r[f"{k} Σ%"] = dl[mm].sum(); r[f"{k} $"] = dl[mm].sum() * USD
            cls_rows.append(r)
    P(f"\n**{name} — arm A vs B (net stop) vs Bp (paid-only) · one size 0.61 × equity · Δ = arm − A per fill, day-block bootstrap 95 % CI**\n")
    P(C.md(pd.DataFrame(rows), 2))
    P(f"\n**{name} — what changed (SAVED = A stopped anyway, B out earlier · CUT = A reached TP / time, B stopped · WIDENED = received funding "
      f"moved B's line out past A's stop bar) · $ = Σ Δ % × 61 (a $10,000 account at one size)**\n")
    P(C.md(pd.DataFrame(cls_rows), 2))


def changed_fills(g, name, top=12):
    dl = g.B_net - g.A_net
    ch = g[np.abs(dl) > 1e-9].copy()
    ch["d"] = dl[np.abs(dl) > 1e-9]
    if not len(ch):
        P(f"\n{name}: no fill changes under arm B.")
        return
    ch = ch.reindex(ch.d.abs().sort_values(ascending=False).index).head(top)
    t = pd.DataFrame(dict(fill=ch.pair + " " + ch.day, A=ch.A_net, A_why=ch.A_why, B=ch.B_net, B_why=ch.B_why, d=ch.d, paid_A=ch.paid,
                          hourly=(ch.interval_h == 1), entry_rate_pct=ch.erate * 100))
    P(f"\n**{name} — the changed fills (largest |Δ| first, up to {top})**\n")
    P(C.md(t, 3))


def concentration(g, name):
    dl = (g.B_net - g.A_net).values
    s = pd.Series(dl).groupby(g.pair.values).sum()
    s = s[s.abs() > 1e-9].sort_values()
    if not len(s):
        P(f"\n{name}: Δ is zero for every pair."); return
    pos, neg = s[s > 0].sum(), s[s < 0].sum()
    P(f"\n**{name} — per-pair Δ (arm B − A, Σ % pts)**: {len(s)} pairs move · Σ+ {pos:+.2f} · Σ− {neg:+.2f} · top gainer share of Σ+ "
      f"{(s.max() / pos * 100) if pos > 0 else float('nan'):.0f} % · top loser share of Σ− {(s.min() / neg * 100) if neg < 0 else float('nan'):.0f} %")
    t = pd.concat([s.head(5), s.tail(5)]).drop_duplicates()
    P("\n" + C.md(pd.DataFrame(dict(pair=t.index, d_sum=t.values)), 2))


def arm_c(g, name):
    m = (g.interval_h == 1) & (g.erate < 0)
    rows = []
    for lab, mm in (("C-skip: hourly ∧ entry rate < 0", m), ("rest", ~m)):
        x = g[mm]
        rows.append(dict(cohort=lab, N=len(x), WR=(x.A_net > 0).mean() * 100 if len(x) else np.nan, avg_A=x.A_net.mean() if len(x) else np.nan,
                         sum_A=x.A_net.sum(), paid_avg=x.paid.mean() if len(x) else np.nan, stops=int((x.A_why == "STOP").sum())))
    P(f"\n**{name} — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**\n")
    P(C.md(pd.DataFrame(rows), 2))


def main():
    S = load()
    for sl in ("FLATBASE_SHORT", "EXHAUSTION_SHORT", "DISTRIBUTION_SHORT"):
        P(f"\n## {sl}")
        coh = [(k[1], v) for k, v in S.items() if k[0] == sl]
        funding_table(coh, sl)
        R = S[(sl, "real")]
        B = S[(sl, "backtest")]
        R0, R1 = ("2026-10-23", "2026-11-21") if sl == "FLATBASE_SHORT" else ("2026-11-18", "2026-11-21")
        blocks = [("REAL", R, R0, R1)]
        if sl == "FLATBASE_SHORT":
            noK = R[~((R.pair == "KERNELUSDT") & (R.day == "2026-11-20"))]
            blocks.append(("REAL w/o KERNEL #25", noK, R0, R1))
            F0 = S[(sl, "real (stack_keep, before DL 601)")]
            blocks.append(("REAL stack_keep (pre-DL 601)", F0, R0, R1))
        des = B[B.day < "2025-07-01"]; hol = B[B.day >= "2025-07-01"]
        for lab, g in (("BT DESIGN", des), ("BT HOLDOUT", hol), ("BT ALL", B)):
            if len(g):
                blocks.append((lab, g, g.day.min(), max(g.A_day.max(), g.B_day.max(), g.Bp_day.max())))
        arms_table(blocks, sl)
        changed_fills(R, f"{sl} real")
        changed_fills(B, f"{sl} backtest")
        concentration(R, f"{sl} real")
        concentration(B, f"{sl} backtest")
        arm_c(R, f"{sl} real"); arm_c(B, f"{sl} backtest")
    with open(os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_tables.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
