#!/usr/bin/env python3
"""B25 prep — block ③ BACKTEST for the FLATBASE_SHORT exit shadows (DL 367 / 368 / 370) vs the live TP −8 / stop 15 / 72 h.

Source (no new simulation, no network): the deterministic S4 · 72 h simulator outputs of scripts/study_flatbase_tp3_stop5.py
(reports/FRENZY_FLATBASE_TP3_STOP5_2026-10-24_trades.csv: TP 3/5/8 × stop 5/10/15) and its TP1 re-run
(reports/FRENZY_FLATBASE_TP1_2026-10-24_trades.csv: TP 1/2/8 × stop 5/10/15); the two live cells (tp8 / stop 15) agree per row.
Rows of the study's backtest periods only (2025 / 2026H1 / 2026H2; the study's 2_master / 1_live rows are blocks ① ② — dropped).
TODAY'S STACK: S4 entry + turnover (24 h volume / mcap) > 1 → trade; ≤ 1 → shadow only; unknown mcap → trade (DL 348). turnover from
reports/FRENZY_FLATBASE_SHORT_2026-10-22_WINLOSS_trades.csv (join pair + dt). Costs 0.20 %, REAL FUNDING INCLUDED (the scout shadows
exclude funding — stated). One deterministic realisation (no seeds). $ at today's ≈ $310 notional per fill (0.2 × 0.25 × base).
Outputs: reports/PREP_B25_BT_FLATBASE_summary.csv · _monthly.csv · _rows.csv
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_b25_bt_common as S  # noqa: E402

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, "reports", "PREP_B25_BT_FLATBASE")
NOTIONAL = 310.0
V = {"LIVE": "tp8|72h|stop15", "TP5": "tp5|72h|stop15", "TP3": "tp3|72h|stop15", "TP5_SL10": "tp5|72h|stop10", "TP5_SL5": "tp5|72h|stop5",
     "TP8_SL10": "tp8|72h|stop10", "TP1": "tp1|72h|stop15", "TP1_SL10": "tp1|72h|stop10", "TP1_SL5": "tp1|72h|stop5"}
K = ["pair", "dt", "per"]
a = pd.read_csv(os.path.join(R, "reports", "FRENZY_FLATBASE_TP3_STOP5_2026-10-24_trades.csv"))
b = pd.read_csv(os.path.join(R, "reports", "FRENZY_FLATBASE_TP1_2026-10-24_trades.csv"))
# a variant that exits earlier can free the pair for a later episode (one short per pair) → the sim writes a row per (pair, dt) per
# variant set; a NaN cell = that variant did not hold that entry. Collapse the split rows (first non-null per column).
a = a.groupby(K, as_index=False, sort=False).first(); b = b.groupby(K, as_index=False, sort=False).first()
bc = [c for c in b.columns if c.startswith(("net|tp1", "reason|tp1"))]
D = a.merge(b[K + bc + ["net|tp8|72h|stop15"]].rename(columns={"net|tp8|72h|stop15": "chk"}), on=K, how="inner", validate="1:1")
_m = D["net|tp8|72h|stop15"].notna() & D.chk.notna()
assert len(D) == len(a) and (D.loc[_m, "net|tp8|72h|stop15"] - D.loc[_m, "chk"]).abs().max() < 1e-9, "live cell differs between the two study runs"
D = D[D.per.isin(["2025", "2026H1", "2026H2"])].copy()
w = pd.read_csv(os.path.join(R, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_WINLOSS_trades.csv"), low_memory=False)[["pair", "dt", "turnover", "mcap_musd"]]
D = D.merge(w.drop_duplicates(["pair", "dt"]), on=["pair", "dt"], how="left")
D["stack"] = np.where(D.turnover.isna(), "unknown→trade", np.where(D.turnover > 1, "turnover>1", "shadow(≤1)"))
D["today"] = D["stack"] != "shadow(≤1)"
D["day"] = D.dt.str[:10]; D["month"] = D.dt.str[:7]
D["half"] = D.fold
for k, c in V.items():
    D[k] = D[f"net|{c}"].astype(float); D[k + "_why"] = D[f"reason|{c}"]
D["live_class"] = D.LIVE_why.map({"stop": "live_stopped", "tp": "live_tp", "time": "live_time"}).fillna("other")

rows, mrows = [], []
for scope in ("today", "all_S4"):
    A = D[D.today] if scope == "today" else D
    for per in ("ALL", "2025H1", "2025H2", "2026H1", "2026H2", "2026"):
        P = A if per == "ALL" else (A[A.per == "2026H1"].pipe(lambda x: pd.concat([x, A[A.per == "2026H2"]])) if per == "2026" else A[A.half == per])
        if not len(P):
            continue
        for k in V:
            Pv = P[P[k].notna()].assign(_x=lambda x: x[k], _usd=lambda x: x[k] / 100 * NOTIONAL)
            r = S.summ(Pv, col="_x", usd={"usd": "_usd"})
            r["only_variant_rows"] = int((P[k].notna() & P.LIVE.isna()).sum()); r["only_live_rows"] = int((P[k].isna() & P.LIVE.notna()).sum())
            Pp = P[P[k].notna() & P.LIVE.notna()]          # paired rows only (never overwrite P: the next variant needs every row)
            dd = (Pp[k] - Pp.LIVE).values
            lo, hi, pn = S.day_boot(dd, Pp.day.values)
            # one-sided Bonferroni LB (DL 367/370: 1 − 0.05/8) via the same day bootstrap
            dg = pd.DataFrame({"d": Pp.day.values, "x": dd}).groupby("d").x.agg(["sum", "count"])
            rng = np.random.default_rng(367); ix = rng.integers(0, len(dg), (4000, len(dg)))
            bm = dg["sum"].values[ix].sum(1) / dg["count"].values[ix].sum(1)
            lb = float(np.percentile(bm, 100 * 0.05 / 8))
            pos = pd.Series(dd, index=Pp.day.values); pday = pos.groupby(level=0).sum()
            top_day_share = float(pday.max() / pday[pday > 0].sum()) if (pday > 0).any() else np.nan
            split = {}
            for cl in ("live_stopped", "live_tp", "live_time"):
                m = (Pp.live_class == cl).values
                x = dd[m]
                split[f"{cl}_n"] = int(m.sum())
                split[f"{cl}_sumD"] = round(float(x.sum()), 1)
                if cl == "live_stopped":
                    split["saved_n"] = int((x > 0.01).sum()); split["saved_sumD"] = round(float(x[x > 0.01].sum()), 1)
                    split["deeper_n"] = int((x < -0.01).sum()); split["deeper_sumD"] = round(float(x[x < -0.01].sum()), 1)
                if cl == "live_tp":
                    vw = Pp[k + "_why"].values[m]
                    split["cut_winners_n"] = int((vw == "stop").sum()); split["cut_winners_sumD"] = round(float(x[vw == "stop"].sum()), 1)
                    split["giveback_n"] = int(((vw != "stop") & (x < -0.01)).sum()); split["giveback_sumD"] = round(float(x[(vw != "stop") & (x < -0.01)].sum()), 1)
            ps, pk = S.loss_share(Pp.assign(_d=dd), "pair", col="_d")
            rows.append(dict(scope=scope, period=per, variant=k, **r, dAvg=round(float(dd.mean()), 4), dCI_lo=round(lo, 4), dCI_hi=round(hi, 4),
                             dP_neg=round(pn, 3), dLB_bonf8=round(lb, 4), top_day_share_posD=round(top_day_share, 2) if np.isfinite(top_day_share) else np.nan,
                             top_pair_share_negD=round(ps, 2) if np.isfinite(ps) else np.nan, **split))
    for (m, k), g in A.melt(id_vars=["month", "LIVE"], value_vars=list(V), var_name="variant", value_name="pct").groupby(["month", "variant"]):
        mrows.append(dict(scope=scope, month=m, variant=k, N=len(g), WR=round(100 * (g.pct > 0).mean(), 1), avg=round(g.pct.mean(), 4),
                          dAvg=round((g.pct - g.LIVE).mean(), 4)))
Sm = pd.DataFrame(rows); Mm = pd.DataFrame(mrows)
Sm.to_csv(OUT + "_summary.csv", index=False); Mm.to_csv(OUT + "_monthly.csv", index=False)
D[["pair", "dt", "per", "half", "turnover", "mcap_musd", "stack", "today", "live_class"] + list(V) + [k + "_why" for k in V]].to_csv(OUT + "_rows.csv", index=False)
cons = []
for (sc, k), g in Mm[Mm.variant != "LIVE"].groupby(["scope", "variant"]):
    g = g[g.N >= 5]
    cons.append(dict(scope=sc, variant=k, months=len(g), months_better=int((g.dAvg > 0).sum())))
pd.DataFrame(cons).to_csv(OUT + "_monthcons.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 300)
print(D.groupby(["stack"]).size())
print(Sm[Sm.period.isin(["ALL", "2026"])][["scope", "period", "variant", "N", "WR", "avg", "ci_lo", "ci_hi", "usd", "dAvg", "dCI_lo", "dCI_hi", "dLB_bonf8",
                                            "top_day_share_posD", "saved_n", "saved_sumD", "deeper_n", "deeper_sumD", "cut_winners_n", "cut_winners_sumD", "giveback_n", "giveback_sumD", "live_time_sumD"]].to_string())
print(Sm[(Sm.scope == "today") & Sm.period.isin(["2025H1", "2025H2", "2026H1", "2026H2"])].pivot(index="variant", columns="period", values="dAvg").round(3).to_string())
print(Sm[(Sm.scope == "today") & Sm.period.isin(["2025H1", "2025H2", "2026H1", "2026H2"])].pivot(index="variant", columns="period", values="avg").round(3).to_string())
print(pd.DataFrame(cons).to_string())
