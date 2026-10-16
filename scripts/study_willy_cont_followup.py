#!/usr/bin/env python3
"""WILLY CONTINUATION — ADDENDUM A follow-up (declared post-hoc in reports/WILLY_CONT_PREREG.txt). F1 survivor robustness, F2 model reads.
Usage: venv/bin/python -W ignore scripts/study_willy_cont_followup.py   → reports/willy_cont/followup.md
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports"); OUT = os.path.join(REP, "willy_cont")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
assert "ADDENDUM A" in open(os.path.join(REP, "WILLY_CONT_PREREG.txt")).read()
import study_yr6wl_frenzy_common as C  # noqa: E402

ERAS = ["2025", "2026-H1", "2026-H2"]
MD = []


def P(*s):
    line = " ".join(str(x) for x in s); print(line, flush=True); MD.append(line)


def tbl(df):
    return C.md_table(df, 3) if len(df) else "(empty)"


R = pd.read_pickle(os.path.join(OUT, "rows.pkl"))
for c in list(R.columns):
    if c.startswith("entry_"):
        R[c] = C.num(R, c)
R["tick"] = R.tick90.astype(bool)
R["q1"] = (R["t_1.25"] <= 30) & R.why_today.eq("TP")
R["q2"] = ~(R["t_1.25"] <= 30) & ~(R["t_stop"] <= 30) & R.y_today.notna() & (R.x_today > 30)
R["C2_30"] = R.why_hold2.eq("TP2").astype(float); R["C2_90"] = R["t_2"].notna().astype(float)
R["y1"] = R.y_hold2 - R.y_today; R["y2"] = R.y_keep - R.y_today
BT = R[R.block == "BT"].copy(); BT["seed"] = BT.seed.astype(int)
S1 = BT[BT.seed == 1]; VT = S1[S1.tick]
X1, X2 = "e_br_up5", "entry_ema_gap_8_13"
C1, C2 = 0.2979, 1.393

# ═════════ F1 survivor robustness ═════════
P("# WILLY CONTINUATION — ADDENDUM A follow-up (post-hoc, declared)\n")
P("## F1 — the Q1 survivor: market-wide red entry bar (≤ 30 % of the top-50 up on the last closed 5m bar) × coin EMA8–13 gap high\n")
Q1 = VT[VT.q1].copy()
Q1["zone"] = (Q1[X1] <= C1) & (Q1[X2] > C2)
rows = []
for e in ERAS + ["ALL"]:
    g = Q1 if e == "ALL" else Q1[Q1.era == e]
    z = g[g.zone]; r = g[~g.zone]
    rows.append(dict(era=e, n_zone=len(z), days_zone=z.day.nunique(), pairs_zone=z.pair.nunique(), C2_30_zone=z.C2_30.mean(), C2_30_rest=r.C2_30.mean(),
                     C3_90_zone=z["t_3"].notna().mean(), hold3_minus_bank_zone=(z.y_hold3 - z.y_today).mean(), y1_zone=z.y1.mean(), y1_rest=r.y1.mean()))
P(tbl(pd.DataFrame(rows)))
z = Q1[Q1.zone]


def lb(k, n):
    from math import comb
    for p in np.linspace(0.5, 1, 5001):                 # one-sided 95 % Clopper-Pearson lower bound: P(X >= k | p) = 0.05
        if sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)) >= 0.05:
            return p
    return np.nan


P(f"\nZone concentration: top pair {z.pair.value_counts().iloc[0]}/{len(z)} ({z.pair.value_counts().index[0]}), top day {z.day.value_counts().iloc[0]}/{len(z)}; "
  f"months {z.month.nunique()}; one-sided 95 % lower bound of C2_30 for {int(z.C2_30.sum())}/{len(z)} = {lb(int(z.C2_30.sum()), len(z)):.3f}")
lomo = []
for mo in sorted(Q1.month.unique()):
    g = Q1[Q1.month != mo]
    lomo.append(g[g.zone].y1.mean() - 0)
P(f"Leave-one-month-out: zone y1 (hold2 − bank) > 0 in {sum(v > 0 for v in lomo)}/{len(lomo)} months left out (min {min(lomo):+.3f})")

P("\nEra-frozen cuts (lower tercile of e_br_up5, upper tercile of the gap, computed on the discovery era's Q1 fills only):\n")
rows = []
for disc, tests in (("2025", ["2026-H1", "2026-H2"]), ("2026", ["2025"]), ("2026-H1", ["2025", "2026-H2"])):
    d = Q1[Q1.era.str.startswith(disc)] if disc == "2026" else Q1[Q1.era == disc]
    c1 = d[X1].quantile(1 / 3); c2 = d[X2].quantile(2 / 3)
    for t in tests:
        g = Q1[Q1.era == t]; m = (g[X1] <= c1) & (g[X2] > c2)
        rows.append(dict(discovery=disc, cut_br=c1, cut_gap=c2, test=t, n_zone=int(m.sum()), C2_30_zone=g[m].C2_30.mean(), C2_30_rest=g[~m].C2_30.mean(),
                         y1_zone=g[m].y1.mean(), delta_per_signal=(g[m].y1.sum()) / len(VT[VT.era == t])))
P(tbl(pd.DataFrame(rows)))

P("\nDose-response: C2_30 (share of +1.25 touches that reach +2 before minute 30) by terciles, seed-1 tick, all eras (N in brackets):\n")
q_b = Q1[X1].quantile([1 / 3, 2 / 3]).values; q_g = Q1[X2].quantile([1 / 3, 2 / 3]).values
Q1["b3"] = np.digitize(Q1[X1], q_b); Q1["g3"] = np.digitize(Q1[X2], q_g)
grid = Q1.groupby(["b3", "g3"]).agg(C2=("C2_30", "mean"), n=("C2_30", "size"), y1=("y1", "mean")).reset_index()
gr = grid.pivot(index="b3", columns="g3", values="C2").round(3).astype(str) + " (" + grid.pivot(index="b3", columns="g3", values="n").astype(str) + ")"
gr.index = [f"breadth T{i + 1}" + (" (lowest)" if i == 0 else "") for i in gr.index]; gr.columns = [f"gap T{i + 1}" + (" (highest)" if i == 2 else "") for i in gr.columns]
P(tbl(gr.reset_index().rename(columns={'index': 'breadth tercile'})))
gy = grid.pivot(index="b3", columns="g3", values="y1").round(3)
P("\nSame grid, y1 = HOLD2 − bank (%/touched fill):\n"); P(tbl(gy.reset_index()))
for nm, v in (("breadth alone (≤ 0.2979)", Q1[X1] <= C1), ("gap alone (> 1.393)", Q1[X2] > C2)):
    P(f"{nm}: C2_30 {Q1[v].C2_30.mean():.3f} (N {int(v.sum())}) vs rest {Q1[~v].C2_30.mean():.3f}; y1 {Q1[v].y1.mean():+.3f} vs {Q1[~v].y1.mean():+.3f}")

O = S1[(~S1.tick) & S1.q1].copy(); O["zone"] = (O[X1] <= C1) & (O[X2] > C2)
P(f"\n1m ruler (seed 1 non-tick touched fills, refute-only, mostly 2025): zone C2_30 {O[O.zone].C2_30.mean():.3f} (N {int(O.zone.sum())}) vs rest "
  f"{O[~O.zone].C2_30.mean():.3f}; y1 zone {O[O.zone].y1.mean():+.3f} vs rest {O[~O.zone].y1.mean():+.3f}")
Ms = R[(R.block != "BT") & R.q1].copy(); Ms["zone"] = (Ms[X1] <= C1) & (Ms[X2] > C2)
P(f"Master + live touched fills: zone {int(Ms.zone.sum())} → reached +2 by min 30: {int(Ms[Ms.zone].C2_30.sum())}; rest C2_30 {Ms[~Ms.zone].C2_30.mean():.3f} (N {int((~Ms.zone).sum())})")

# ═════════ F2 models ═════════


def prep(Xtr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def logit_fit(X, y, lam=1.0, it=50):
    X1_ = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X1_.shape[1])
    L = lam * np.eye(len(w)); L[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X1_ @ w, -30, 30)))
        step = np.linalg.solve(X1_.T @ (X1_ * (p * (1 - p))[:, None]) + L, X1_.T @ (p - y) + L @ w); w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def auc(s, y):
    y = np.asarray(y, bool); r = pd.Series(s).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


cov = pd.read_csv(os.path.join(OUT, "coverage.csv"))
for q, mask, lab, ycol in (("Q2", "q2", "r125_after30", "y2"), ("Q3", None, "C2_90", None)):
    F = VT[VT[mask]] if mask else VT[VT.y_today.notna() & VT.y_hold2.notna()]
    F = F.reset_index(drop=True)
    cols = cov[(cov.q == q) & cov.tested & (cov.block == "COIN")]["var"].tolist()
    cols = [c for c in cols if pd.to_numeric(F[c], errors="coerce").notna().mean() >= 0.5 and pd.to_numeric(F[c], errors="coerce").nunique() > 1]
    X = np.column_stack([pd.to_numeric(F[c], errors="coerce").values.astype(float) for c in cols])
    y = F[lab].values.astype(float)
    P(f"\n## F2 — {q} COIN logistic: the largest standardised coefficients per training era (label {lab}, N {len(F)})\n")
    co = {}
    for nm, tr in (("2025", F.era.eq("2025").values), ("2026", F.era.ne("2025").values)):
        Xtr, _ = prep(X[tr], X[~tr]); w = logit_fit(Xtr, y[tr])[1:]
        co[nm] = pd.Series(w, index=cols)
    CO = pd.DataFrame(co); CO["same_sign"] = np.sign(CO["2025"]) == np.sign(CO["2026"]); CO["min_abs"] = np.where(CO.same_sign, CO[["2025", "2026"]].abs().min(1), 0)
    CO = CO.sort_values("min_abs", ascending=False)
    P(tbl(CO.head(10).reset_index().rename(columns={"index": "var"})))
    top = CO.index[0]; sg = np.sign(CO.loc[top, "2026"])
    P(f"\nTop same-sign variable: {top} (sign {'+' if sg > 0 else '−'}). Single-variable rule, cut = discovery-era median:")
    rows = []
    for disc, te in (("2025", F.era.ne("2025")), ("2026", F.era.eq("2025"))):
        trm = ~te.values
        cut = np.nanmedian(pd.to_numeric(F[top], errors="coerce").values[trm])
        g = F[te.values]; x = pd.to_numeric(g[top], errors="coerce").values
        good = (x > cut) if sg > 0 else (x <= cut)
        rec = dict(discovery=disc, cut=cut, test_N=len(g), lab_good=g[lab][good].mean(), lab_bad=g[lab][~good].mean(), AUC_single=auc(sg * np.nan_to_num(x, nan=cut), g[lab].values))
        if ycol:
            d = np.where(good, g.y_keep.values, g.y_today.values) - g.y_today.values
            lo, hi, pp = C.day_boot(d, g.day.values)
            rec.update(rule_minus_today=d.mean(), P_gt0=pp, kept=int(good.sum()), keep_good=g.y2[good].mean(), keep_bad=g.y2[~good].mean())
        rows.append(rec)
    P(tbl(pd.DataFrame(rows)))
    if ycol:
        P("\nModel-score rule (post-hoc): keep +1.25 at minute 30 iff the COIN-logit predicted P(R125) is in the training top third; else today's late rule:\n")
        rows = []
        for disc, te in (("2025", F.era.ne("2025").values), ("2026", F.era.eq("2025").values)):
            Xtr, Xte = prep(X[~te], X[te]); w = logit_fit(Xtr, y[~te])
            str_ = np.column_stack([np.ones(len(Xtr)), Xtr]) @ w; ste = np.column_stack([np.ones(len(Xte)), Xte]) @ w
            thr = np.quantile(str_, 2 / 3); good = ste > thr
            g = F[te]
            d = np.where(good, g.y_keep.values, g.y_today.values) - g.y_today.values
            lo, hi, pp = C.day_boot(d, g.day.values)
            rows.append(dict(discovery=disc, test_N=len(g), kept=int(good.sum()), R125_kept=g[lab][good].mean(), R125_rest=g[lab][~good].mean(),
                             keep_minus_today_kept=g.y2[good].mean(), rule_minus_today_per_open30=d.mean(), P_gt0=pp,
                             per_signal=d.sum() / len(S1[S1.tick & (S1.era.ne('2025') if disc == '2025' else S1.era.eq('2025'))])))
        P(tbl(pd.DataFrame(rows)))
        P("\nQ2 economics by the state at minute 30 — what recovers is mostly decided by how far below the entry the coin sits:\n")
        F["cl30b"] = pd.cut(F.cl30, [-5, -2, -1, -0.25, 0.25, 5])
        P(tbl(F.groupby("cl30b", observed=True).agg(N=("y2", "size"), R125=(lab, "mean"), today=("y_today", "mean"), keep=("y_keep", "mean"),
                                                     close30=("y_close30", "mean")).reset_index().astype({"cl30b": str})))
open(os.path.join(OUT, "followup.md"), "w").write("\n".join(MD) + "\n")
