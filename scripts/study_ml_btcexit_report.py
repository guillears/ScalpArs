#!/usr/bin/env python3
"""ML BTC-REACTIVE EXIT study (2026-10-26) — tables + verdicts from scripts/study_ml_btcexit_run.py outputs.

PRE-DECLARED VERDICT BAR (fixed 2026-10-26 before any variant was read; primary latency scenario = FAST, K = 128 variants):
  SHIP-GRADE   Bonferroni (K = 128) day-block CI low > 0 on KEEP master (FAST) ∧ Δ > 0 under SLOW ∧ Δ > 0 in ≥ 5 era groups incl. G5
               and G6 ∧ Δ > 0 without the washed-out window ∧ top day and top pair each < 50 % of the net Δ ∧ DAY units: more days
               improved than worsened ∧ ≥ 8 distinct days changed ∧ yr5 Δ ≥ 0 (caveat leg: master leads, a significant opposite yr5 sign
               downgrades to OPERATOR CALL).
  OPERATOR CALL  nominal 95 % day-block CI low > 0 (FAST) but some ship leg fails.
  KEEP OBSERVING Δ > 0 (FAST) with the CI including 0.
  REJECTED       Δ ≤ 0 (FAST) — DECIDED by the gate.
  Frozen observe lines: at most two, chosen by a pre-declared rule — among KEEP OBSERVING / OPERATOR CALL variants with SLOW Δ > 0 and
  G6 Δ > 0, the highest FAST 95 % CI low (one per family X* / TR* / TS*, max two).

    venv/bin/python -W ignore scripts/study_ml_btcexit_report.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_btcexit_run as RN  # noqa: E402
DD = RN.DD

OUT = RN.OUT
SEED = 20261026
K = len(RN.VARIANTS)
GROUPS = DD.GROUP_ORDER[:-1]
NS = (1, 3, 5, 10, 15)
REPS = 20000


def nwr(x):
    x = pd.Series(x).dropna()
    return f"{len(x)}·{(x > 0).mean() * 100:.0f}%·{x.mean():+.3f}" if len(x) else "0"


def _boot_pick(days, reps=REPS, seed=SEED):
    ud, inv = np.unique(np.asarray(days), return_inverse=True)
    rng = np.random.default_rng(seed)
    return ud, inv, rng.integers(0, len(ud), (reps, len(ud)))


def boot(vals, days, alpha=0.05):
    vals = np.asarray(vals, float)
    ud, inv, pick = _boot_pick(days)
    s = np.bincount(inv, weights=vals, minlength=len(ud)); c = np.bincount(inv, minlength=len(ud))
    m = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1)
    return (float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2))), float((m <= 0).mean()))


def auc(score, y):
    """P(score of a winner > score of a loser) — Mann-Whitney with ties = ½."""
    score = np.asarray(score, float); y = np.asarray(y, bool)
    a, b = score[y], score[~y]
    if not len(a) or not len(b):
        return np.nan
    r = pd.Series(np.r_[a, b]).rank().values
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def auc_boot(score, y, days, reps=2000):
    score = np.asarray(score, float); y = np.asarray(y, bool); days = np.asarray(days)
    ud = np.unique(days); rng = np.random.default_rng(SEED)
    ix = {d: np.flatnonzero(days == d) for d in ud}
    vals = []
    for _ in range(reps):
        sel = np.concatenate([ix[d] for d in rng.choice(ud, len(ud))])
        vals.append(auc(score[sel], y[sel]))
    v = np.array(vals); v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def strat_auc(score, y, strata):
    """AUC of `score` WITHIN strata of the pair's own state (weighted by the pairs compared) — BTC's information beyond the pair."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        a = auc(score[m], y[m])
        w = (y[m].sum()) * ((~y[m]).sum())
        if w > 0 and not np.isnan(a):
            num += a * w; den += w
    return num / den if den else np.nan


def strat_auc_boot(score, y, strata, days, reps=2000):
    ud = np.unique(days); rng = np.random.default_rng(SEED)
    ix = {d: np.flatnonzero(days == d) for d in ud}
    vals = []
    for _ in range(reps):
        sel = np.concatenate([ix[d] for d in rng.choice(ud, len(ud))])
        vals.append(strat_auc(score[sel], y[sel], strata[sel]))
    v = np.array(vals); v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def logit_fit(X, y, ridge=1e-3):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (y - p) - ridge * w
        H = -(X.T * (p * (1 - p))) @ X - ridge * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def lodo_auc(X, y, days):
    """leave-one-DAY-out cross-validated AUC of a logistic model (no in-sample fit leaks into the score)."""
    s = np.full(len(y), np.nan)
    for d in np.unique(days):
        te = days == d
        w = logit_fit(X[~te], y[~te].astype(float))
        s[te] = np.c_[np.ones(te.sum()), X[te]] @ w
    return auc(s, y)


def load():
    C = DD.cohort().set_index("key")
    F = pd.read_csv(OUT + "_fills.csv").set_index("key")
    C = C.join(F, rsuffix="_f")
    C["day"] = C.ts.dt.strftime("%Y-%m-%d")
    atr = C.loc[C.block == "MASTER", "atr"]
    lo, hi = np.round(np.percentile(atr, [100 / 3, 200 / 3]), 2)
    C["atr_b"] = np.where(C.atr < lo, "ATR_lo", np.where(C.atr < hi, "ATR_mid", "ATR_hi"))
    C["win"] = C.y > 0
    V = pd.read_csv(OUT + "_variants.csv.gz")
    V["d"] = V.pct - V.base_pct
    return C, V, (lo, hi)


def conc_share(d, by):
    tot = d.sum()
    if tot <= 1e-12:
        return np.nan
    return float(d.groupby(by).sum().max() / tot)


def summarize(V, C, scen, scope_mask):
    S = V[V.scen == scen]
    piv = S.pivot(index="key", columns="variant", values="d")
    chg = S.pivot(index="key", columns="variant", values="changed")
    keys = C.index[scope_mask]
    piv = piv.reindex(keys).fillna(0.0); chg = chg.reindex(keys).fillna(0)
    Cs = C.loc[keys]
    days = Cs.day.values
    ud, inv, pick = _boot_pick(days)
    cnt = np.bincount(inv, minlength=len(ud))
    den = cnt[pick].sum(1)
    a_b = 0.05 / K
    rows = []
    base = Cs.base_pct.values
    for v in RN.VARIANTS:
        d = piv[v].values; ch = chg[v].values.astype(bool)
        s = np.bincount(inv, weights=d, minlength=len(ud))
        m = s[pick].sum(1) / den
        r = dict(variant=v, scen=scen, N=len(d), d=d.mean(), sum_d=d.sum(),
                 lo95=np.percentile(m, 2.5), hi95=np.percentile(m, 97.5), p_le0=(m <= 0).mean(),
                 lo_bonf=np.percentile(m, 100 * a_b / 2), hi_bonf=np.percentile(m, 100 * (1 - a_b / 2)),
                 n_changed=int(ch.sum()))
        win = base > 0
        r["win_cut"] = int((ch & win & (d < 0)).sum()); r["win_cut_sum"] = d[ch & win & (d < 0)].sum()
        r["win_better"] = int((ch & win & (d > 0)).sum()); r["win_better_sum"] = d[ch & win & (d > 0)].sum()
        r["los_saved"] = int((ch & ~win & (d > 0)).sum()); r["los_saved_sum"] = d[ch & ~win & (d > 0)].sum()
        r["los_worse"] = int((ch & ~win & (d < 0)).sum()); r["los_worse_sum"] = d[ch & ~win & (d < 0)].sum()
        ds = pd.Series(d, index=keys)
        dd = ds.groupby(days).sum()
        r["days_changed"] = int((pd.Series(ch, index=keys).groupby(days).sum() > 0).sum())
        r["days_pos"] = int((dd > 1e-9).sum()); r["days_neg"] = int((dd < -1e-9).sum())
        r["top_day_share"] = conc_share(ds, days); r["top_pair_share"] = conc_share(ds, Cs.pair.values)
        r["top_day"] = dd.idxmax() if len(dd) else ""
        for g in GROUPS:
            mm = (Cs.era_group == g).values
            r[f"d_{g}"] = d[mm].mean() if mm.sum() else np.nan
            r[f"base_{g}"] = base[mm].mean() if mm.sum() else np.nan
        r["d_exwashed"] = d[~Cs.washed.astype(bool).values].mean()
        r["d_early"] = d[Cs.era_group.isin(GROUPS[:5]).values].mean()
        r["d_recent"] = d[Cs.era_group.isin(GROUPS[5:]).values].mean()
        r["groups_pos"] = sum(1 for g in GROUPS if r[f"d_{g}"] > 1e-9)
        rows.append(r)
    return pd.DataFrame(rows)


def other_blocks(V, C, scen, block):
    S = V[(V.scen == scen)]
    keys = C.index[C.block == block]
    piv = S.pivot(index="key", columns="variant", values="d").reindex(keys).fillna(0.0)
    return piv.mean(), C.loc[keys, "base_pct"].mean(), len(keys)


def verdict(r, slow_d, yr5):
    if r.d <= 1e-9:
        return "REJECTED (Δ ≤ 0)"
    ship = (r.lo_bonf > 0 and slow_d > 0 and r.groups_pos >= 5 and r[f"d_{GROUPS[5]}"] > 0 and r[f"d_{GROUPS[6]}"] > 0
            and r.d_exwashed > 0 and r.top_day_share < 0.5 and r.top_pair_share < 0.5 and r.days_pos > r.days_neg
            and r.days_changed >= 8)
    if ship:
        if yr5 is not None and yr5[1] < 0:
            return "OPERATOR CALL (ship legs pass; yr5 significantly opposite)"
        return "SHIP-GRADE"
    if r.lo95 > 0:
        return "OPERATOR CALL (nominal CI > 0, a ship leg fails)"
    return "KEEP OBSERVING (Δ > 0, CI incl. 0)"


def yr5_summary():
    fp = OUT + "_yr5_variants.csv.gz"
    if not os.path.exists(fp):
        return None, None
    Y = pd.read_csv(fp); Y["d"] = Y.pct - Y.base_pct
    F = pd.read_csv(OUT + "_yr5_fills.csv").set_index("key")
    F["day"] = pd.to_datetime(F.index.str.split("|").str[1].str[:10])
    rows = []
    for (v, sc), g in Y.groupby(["variant", "scen"]):
        days = F.loc[g.key, "day"].values
        lo, hi, p = boot(g.d.values, days)
        seeds = g.key.str.split("|").str[0]
        rows.append(dict(variant=v, scen=sc, N=len(g), d=g.d.mean(), lo95=lo, hi95=hi, p_le0=p, n_changed=int(g.changed.sum()),
                         seeds="/".join(f"{g.d[seeds == s].mean():+.3f}" for s in sorted(seeds.unique()))))
    return pd.DataFrame(rows), F


def main():
    C, V, (alo, ahi) = load()
    L = []; W = L.append
    M = C[C.block == "MASTER"]
    ref_notional = float(C.loc[C.block == "B25", "notional"].median())

    # ───────── §0 coverage
    W("## A. Cohort, path sources and replica check\n")
    W("| block | fills | BTC path tick | BTC path 1m | pair path tick | pair path 1m | WR (y) | avg y % | BASE replica avg |")
    W("|---|---|---|---|---|---|---|---|---|")
    for b in ("MASTER", "B25", "PROBE"):
        g = C[C.block == b]
        W(f"| {b} | {len(g)} | {(g.btc_src == 'tick').sum()} | {(g.btc_src == '1m').sum()} | {(g.pair_src == 'tick').sum()} | "
          f"{(g.pair_src == '1m').sum()} | {g.win.mean() * 100:.0f}% | {g.y.mean():+.3f} | {g.base_pct.mean():+.3f} |")
    for gname in GROUPS:
        g = M[M.era_group == gname]
        W(f"|   {gname} | {len(g)} | {(g.btc_src == 'tick').sum()} | {(g.btc_src == '1m').sum()} | {(g.pair_src == 'tick').sum()} | "
          f"{(g.pair_src == '1m').sum()} | {g.win.mean() * 100:.0f}% | {g.y.mean():+.3f} | {g.base_pct.mean():+.3f} |")
    W(f"\nBASE replica = the exit-deep evaluator at today's stack (bit-equal to `ML_EXIT_DEEP_2026-10-26_variants.csv` BASE on all 333 fills). "
      f"Distinct master days: {M.day.nunique()}.\n")

    # ───────── §1 descriptive
    W("## B. BTC's move after entry — winners vs losers (KEEP master; W/L = recorded outcome y at 1×)\n")
    W("BTC return since entry in %, at minute N. AUC = P(a winner's BTC move > a loser's), 0.5 = no information; CI = day-block bootstrap "
      "(resample calendar days). `pair net N` = the fill's own net P&L at N (replica path). `open` = fills the BASE replica has not yet "
      "exited at N (the only fills an exit rule at N can touch).\n")
    W("| N | fills (open at N) | BTC winners median [p25, p75] | BTC losers median [p25, p75] | AUC BTC [95 % CI] | AUC pair net N [CI] | "
      "AUC BTC within pair-net terciles [CI] | LODO-CV AUC pair-only → pair+BTC |")
    W("|---|---|---|---|---|---|---|---|")
    drows = []
    for N in NS:
        g = M[M[f"open{N}"].astype(bool) & M[f"btc{N}"].notna()]
        y = g.win.values; days = g.day.values
        b = g[f"btc{N}"].values; pn = g[f"pair_net{N}"].values; pk = g[f"pair_pk{N}"].values
        a_b = auc(b, y); ci_b = auc_boot(b, y, days)
        a_p = auc(pn, y); ci_p = auc_boot(pn, y, days)
        terc = pd.qcut(pn, 3, labels=False, duplicates="drop")
        a_s = strat_auc(b, y, terc); ci_s = strat_auc_boot(b, y, terc, days)
        X1 = np.c_[pn, pk]; X2 = np.c_[pn, pk, b]
        cv1 = lodo_auc(X1, y, days); cv2 = lodo_auc(X2, y, days)
        q = lambda x: f"{np.median(x):+.3f} [{np.percentile(x, 25):+.3f}, {np.percentile(x, 75):+.3f}]"
        W(f"| {N} | {len(g)} ({y.sum()} W / {(~y).sum()} L) | {q(b[y])} | {q(b[~y])} | {a_b:.3f} [{ci_b[0]:.3f}, {ci_b[1]:.3f}] | "
          f"{a_p:.3f} [{ci_p[0]:.3f}, {ci_p[1]:.3f}] | {a_s:.3f} [{ci_s[0]:.3f}, {ci_s[1]:.3f}] | {cv1:.3f} → {cv2:.3f} |")
        drows.append(dict(N=N, n=len(g), auc_btc=a_b, auc_btc_lo=ci_b[0], auc_btc_hi=ci_b[1], auc_pair=a_p, auc_pair_lo=ci_p[0],
                          auc_pair_hi=ci_p[1], auc_btc_within=a_s, auc_btc_within_lo=ci_s[0], auc_btc_within_hi=ci_s[1],
                          lodo_pair=cv1, lodo_pair_btc=cv2, btc_med_w=np.median(b[y]), btc_med_l=np.median(b[~y])))
    pd.DataFrame(drows).to_csv(OUT + "_separation.csv", index=False)
    # all-fills version (incl. already-exited) for the plain description at 15 min
    W("\nAll master fills regardless of exit (descriptive only):\n")
    W("| N | BTC winners mean | BTC losers mean | share of losers with BTC ≤ −0.10 | share of winners with BTC ≤ −0.10 | AUC |")
    W("|---|---|---|---|---|---|")
    for N in NS:
        g = M[M[f"btc{N}"].notna()]
        y = g.win.values; b = g[f"btc{N}"].values
        W(f"| {N} | {b[y].mean():+.3f} | {b[~y].mean():+.3f} | {(b[~y] <= -0.10).mean() * 100:.0f}% | {(b[y] <= -0.10).mean() * 100:.0f}% | {auc(b, y):.3f} |")

    # 2-variable conditional tables
    W("\n### B2. Two-variable conditional table — does BTC add information beyond the pair's own P&L at N? (KEEP master, fills open at N)\n")
    W("Cells: N·WR·avg y %. Rows = the pair's own net P&L at N; columns = BTC since entry at N.\n")
    crow = []
    for N in (3, 5, 10, 15):
        g = M[M[f"open{N}"].astype(bool) & M[f"btc{N}"].notna()]
        pb = pd.cut(g[f"pair_net{N}"], [-99, -0.30, 0.0, 99], labels=["pair ≤ −0.30", "pair (−0.30, 0]", "pair > 0"])
        bb = pd.cut(g[f"btc{N}"], [-99, -0.10, 0.0, 99], labels=["BTC ≤ −0.10", "BTC (−0.10, 0]", "BTC > 0"])
        W(f"**N = {N} min** ({len(g)} fills)\n")
        W("| | BTC ≤ −0.10 | BTC (−0.10, 0] | BTC > 0 | row |"); W("|---|---|---|---|---|")
        for pr in pb.cat.categories:
            cells = []
            for bc in bb.cat.categories:
                s = g[(pb == pr) & (bb == bc)].y
                cells.append(nwr(s))
                crow.append(dict(N=N, pair=pr, btc=bc, n=len(s), wr=(s > 0).mean() if len(s) else np.nan, avg=s.mean() if len(s) else np.nan,
                                 days=g[(pb == pr) & (bb == bc)].day.nunique()))
            W(f"| {pr} | " + " | ".join(cells) + f" | {nwr(g[pb == pr].y)} |")
        W("| col | " + " | ".join(nwr(g[bb == bc].y) for bc in bb.cat.categories) + f" | {nwr(g.y)} |\n")
    pd.DataFrame(crow).to_csv(OUT + "_conditional.csv", index=False)

    # ───────── §2 variants
    scopes = {"MASTER": (C.block == "MASTER").values}
    SUM = {sc: summarize(V, C, sc, scopes["MASTER"]) for sc in RN.SCEN}
    Y5, F5 = yr5_summary()
    y5f = {} if Y5 is None else {(r.variant, r.scen): (r.d, r.hi95 if r.d < 0 else r.lo95, r.lo95, r.hi95) for r in Y5.itertuples()}
    allrows = []
    for sc, S in SUM.items():
        b25, b25base, nb = other_blocks(V, C, sc, "B25")
        pr, prbase, npr = other_blocks(V, C, sc, "PROBE")
        S["d_B25"] = S.variant.map(b25); S["d_PROBE"] = S.variant.map(pr)
        if Y5 is not None:
            S["yr5_d"] = S.variant.map(lambda v: y5f.get((v, sc), (np.nan,))[0])
            S["yr5_lo95"] = S.variant.map(lambda v: y5f.get((v, sc), (np.nan, np.nan, np.nan))[2])
            S["yr5_hi95"] = S.variant.map(lambda v: y5f.get((v, sc), (np.nan, np.nan, np.nan, np.nan))[3])
        S["usd_ticket"] = S.sum_d / 100 * ref_notional
        allrows.append(S)
    A = pd.concat(allrows)
    F_ = A[A.scen == "FAST"].set_index("variant"); SL = A[A.scen == "SLOW"].set_index("variant"); ID = A[A.scen == "IDEAL"].set_index("variant")
    verd = {}
    for v, r in F_.iterrows():
        y5 = None
        if Y5 is not None and not np.isnan(r.get("yr5_d", np.nan)):
            y5 = (r.yr5_d, r.yr5_hi95 if r.yr5_d < 0 else 1.0)
        verd[v] = verdict(r, SL.loc[v, "d"], y5)
    A["verdict_fast"] = A.variant.map(verd)
    A.to_csv(OUT + "_variant_summary.csv", index=False)

    W("\n## C. Exit-rule counterfactuals — grid summary (KEEP master 158, Δ = variant − BASE replica, %/fill at 1×)\n")
    W(f"K = {K} pre-declared variants × 3 latency scenarios. Bonferroni at K = {K} on the primary FAST scenario.\n")
    W("| scenario | variants Δ > 0 | Δ ≤ 0 | nominal CI > 0 | nominal CI < 0 | Bonferroni CI > 0 | best Δ (variant) | median Δ |")
    W("|---|---|---|---|---|---|---|---|")
    for sc in ("IDEAL", "FAST", "SLOW"):
        S = A[A.scen == sc]
        bst = S.loc[S.d.idxmax()]
        W(f"| {sc} | {(S.d > 1e-9).sum()} | {(S.d <= 1e-9).sum()} | {(S.lo95 > 0).sum()} | {(S.hi95 < 0).sum()} | {(S.lo_bonf > 0).sum()} | "
          f"{bst.d:+.3f} ({bst.variant}) | {S.d.median():+.3f} |")
    W("\nBy family (FAST): mean Δ over the family's variants, and the count positive.\n")
    W("| family | variants | mean Δ | positive | nominal CI < 0 | winners cut (Σ over variants, mean per variant) | losers saved (mean per variant) |")
    W("|---|---|---|---|---|---|---|")
    Ff = A[A.scen == "FAST"].copy(); Ff["fam"] = Ff.variant.str.split("_").str[0]
    for fam, g in Ff.groupby("fam", sort=False):
        W(f"| {fam} | {len(g)} | {g.d.mean():+.4f} | {(g.d > 1e-9).sum()} | {(g.hi95 < 0).sum()} | {g.win_cut.mean():.1f} fills, Σ {g.win_cut_sum.mean():+.2f} | "
          f"{g.los_saved.mean():.1f} fills, Σ {g.los_saved_sum.mean():+.2f} |")

    def vrow(v, sc="FAST"):
        r = A[(A.scen == sc) & (A.variant == v)].iloc[0]
        y5 = "" if "yr5_d" not in r or pd.isna(r.yr5_d) else f"{r.yr5_d:+.3f} [{r.yr5_lo95:+.3f}, {r.yr5_hi95:+.3f}]"
        return (f"| `{v}` | {r.d:+.4f} [{r.lo95:+.3f}, {r.hi95:+.3f}] | [{r.lo_bonf:+.3f}, {r.hi_bonf:+.3f}] | {r.p_le0:.2f} | "
                f"{ID.loc[v, 'd']:+.4f} / {SL.loc[v, 'd']:+.4f} | {r.n_changed} | {r.win_cut} ({r.win_cut_sum:+.2f}) | {r.los_saved} ({r.los_saved_sum:+.2f}) | "
                f"{r.los_worse} ({r.los_worse_sum:+.2f}) / {r.win_better} ({r.win_better_sum:+.2f}) | {r.groups_pos}/7 | "
                f"{r[f'd_{GROUPS[5]}']:+.3f} / {r[f'd_{GROUPS[6]}']:+.3f} | {r.d_exwashed:+.3f} | {r.days_pos}+ / {r.days_neg}− of {r.days_changed} | "
                f"{'n/a' if pd.isna(r.top_day_share) else f'{r.top_day_share * 100:.0f}%'} / {'n/a' if pd.isna(r.top_pair_share) else f'{r.top_pair_share * 100:.0f}%'} | "
                f"{r.d_B25:+.3f} | {r.d_PROBE:+.3f} | {y5} | ${r.usd_ticket:+.0f} | {verd[v]} |")
    HDR = ("| variant | Δ FAST [95 % CI] | Bonf. CI | P(Δ≤0) | Δ IDEAL / SLOW | changed | winners cut (Σ Δ) | losers saved (Σ Δ) | "
           "losers worse / winners better | groups + | G5 / G6 Δ | ex-washed Δ | DAYS + / − of changed | top day / pair | B25 Δ | PROBE Δ | yr5 Δ FAST [CI] | "
           "$ Σ at ticket | verdict |")
    W("\n### C1. Top 25 variants by FAST Δ\n")
    W(HDR); W("|---" * 20 + "|")
    for v in F_.sort_values("d", ascending=False).index[:25]:
        W(vrow(v))
    W("\n### C2. Every variant (FAST), grid order\n")
    W(HDR); W("|---" * 20 + "|")
    for v in RN.VARIANTS:
        W(vrow(v))

    # ───────── before → after per era group for the top candidates
    top = list(F_.sort_values("d", ascending=False).index[:6])
    best_by_fam = []
    for fam in ("X", "XG", "XN", "XGN", "TR", "TRG", "TS"):
        g = F_[F_.index.str.split("_").str[0] == fam]
        if len(g):
            best_by_fam.append(g.d.idxmax())
    # pre-declared frozen-line selection (see docstring)
    el = F_[F_.index.map(lambda v: verd[v].startswith(("KEEP", "OPERATOR"))) & (SL.d.reindex(F_.index) > 0) & (F_[f"d_{GROUPS[6]}"] > 1e-9)].copy()
    el["fam"] = np.where(el.index.str.startswith("TS"), "TS", np.where(el.index.str.startswith("TR"), "TR", "X"))
    el["grid"] = [list(RN.VARIANTS).index(v) for v in el.index]
    el = el.sort_values("grid").sort_values(["lo95", "d"], ascending=[False, False], kind="mergesort")   # ties → grid order
    frozen = []
    for v, r in el.iterrows():
        if r.fam not in [el.loc[x, "fam"] for x in frozen] and len(frozen) < 2:
            frozen.append(v)
    tie_note = []
    for fam in ("X", "TS", "TR"):
        g = el[el.fam == fam]
        if not len(g):
            continue
        top_lo = g.lo95.max()
        tied = g[np.isclose(g.lo95, top_lo, atol=1e-12)]
        pick = [v for v in frozen if el.loc[v, "fam"] == fam]
        twins = tied[np.isclose(tied.d, tied.d.max(), atol=1e-12)]
        tie_note.append(f"{fam}: {len(g)} eligible, {len(tied)} tied at the highest CI-low {top_lo:+.4f} "
                        f"({', '.join(f'`{v}`' for v in tied.index)}); tie-break Δ → " + (', '.join(f'`{v}`' for v in twins.index))
                        + (f" (identical twins, resolved by grid order → `{twins.index[0]}`)" if len(twins) > 1 else "")
                        + (f"; picked `{pick[0]}`" if pick else "; family not picked (max two lines)"))
    opcall = [v for v in F_.index if verd[v].startswith("OPERATOR")]
    cand = list(dict.fromkeys(top + best_by_fam + frozen + opcall))
    W("\n## D. Before → after per era group (avg % per fill, BASE replica → variant, FAST)\n")
    W("| variant | " + " | ".join(GROUPS) + " | B25 (9) | PROBE (166) | master 158 |"); W("|---" * (len(GROUPS) + 4) + "|")
    b25b = C[C.block == "B25"].base_pct.mean(); prb = C[C.block == "PROBE"].base_pct.mean()
    gb = []
    for v in cand:
        r = F_.loc[v]
        cells = [f"{r[f'base_{g}']:+.2f}→{r[f'base_{g}'] + r[f'd_{g}']:+.2f}" for g in GROUPS]
        W(f"| `{v}` | " + " | ".join(cells) + f" | {b25b:+.2f}→{b25b + r.d_B25:+.2f} | {prb:+.2f}→{prb + r.d_PROBE:+.2f} | "
          f"{M.base_pct.mean():+.3f}→{M.base_pct.mean() + r.d:+.3f} |")
        gb.append(dict(variant=v, **{f"base_{g}": r[f"base_{g}"] for g in GROUPS}, **{f"after_{g}": r[f"base_{g}"] + r[f"d_{g}"] for g in GROUPS}))
    pd.DataFrame(gb).to_csv(OUT + "_variant_by_group.csv", index=False)

    # per batch for candidates
    S = V[(V.scen == "FAST") & V.variant.isin(cand)].merge(C[["era", "day", "block"]], left_on="key", right_index=True)
    S = S[S.block == "MASTER"]
    pb = S.pivot_table(index="era", columns="variant", values="d", aggfunc="sum").round(3)
    pb.to_csv(OUT + "_variant_by_batch.csv")
    W("\nPer batch Σ Δ (FAST, % points): `ML_BTC_EXIT_2026-10-26_variant_by_batch.csv`.\n")

    # ───────── §3 state splits
    W("## E. State splits for the candidates (FAST, KEEP master; Δ %/fill, n fills in state)\n")
    states = {"STRONG_BULL": M.sbull.values, "not SB": ~M.sbull.values, f"ATR < {alo}": (M.atr_b == "ATR_lo").values,
              f"ATR {alo}–{ahi}": (M.atr_b == "ATR_mid").values, f"ATR ≥ {ahi}": (M.atr_b == "ATR_hi").values,
              "G0–G4 (≤ B11)": M.era_group.isin(GROUPS[:5]).values, "G5 B12–16": (M.era_group == GROUPS[5]).values,
              "G6 B17–24": (M.era_group == GROUPS[6]).values}
    W("| variant | " + " | ".join(f"{k} (n {v.sum()})" for k, v in states.items()) + " |"); W("|---" * (len(states) + 1) + "|")
    Sf = V[V.scen == "FAST"].pivot(index="key", columns="variant", values="d").reindex(M.index).fillna(0)
    srows = []
    for v in cand:
        d = Sf[v].values
        cells = []
        for k, m in states.items():
            lo, hi, _ = boot(d[m], M.day.values[m]) if m.sum() >= 5 else (np.nan, np.nan, np.nan)
            cells.append(f"{d[m].mean():+.3f} [{lo:+.2f}, {hi:+.2f}]")
            srows.append(dict(variant=v, state=k, n=int(m.sum()), d=d[m].mean(), lo95=lo, hi95=hi))
        W(f"| `{v}` | " + " | ".join(cells) + " |")
    pd.DataFrame(srows).to_csv(OUT + "_state_summary.csv", index=False)
    W("\nBASE per state: " + " · ".join(f"{k} {nwr(M.base_pct.values[m])}" for k, m in states.items()) + "\n")

    # ───────── §4 yr5
    W("## F. Backtest cross-check (yr5 ML fills, same engine, caveat only)\n")
    if Y5 is None:
        W("yr5 not run.\n")
    else:
        W(f"yr5 fills: {len(F5)} (the exit-deep yr5 set: chop∧burst excluded); with a BTC path: {F5.btc_src.notna().sum()} "
          f"(tick {(F5.btc_src == 'tick').sum()}, 1m {(F5.btc_src == '1m').sum()}). W/L in yr5 = BASE replica outcome > 0.\n")
        Fb = F5[F5.btc_src.notna()].copy()
        Fb["win"] = Fb.base_pct > 0
        W("| N | yr5 fills open at N | AUC BTC | AUC pair net | AUC BTC within pair-net terciles | LODO-CV AUC pair → pair+BTC |"); W("|---|---|---|---|---|---|")
        for N in NS:
            g = Fb[Fb[f"open{N}"].astype(bool) & Fb[f"btc{N}"].notna()]
            y = g.win.values; days = g.day.dt.strftime("%Y-%m-%d").values
            b = g[f"btc{N}"].values; pn = g[f"pair_net{N}"].values; pk = g[f"pair_pk{N}"].values
            terc = pd.qcut(pn, 3, labels=False, duplicates="drop")
            W(f"| {N} | {len(g)} | {auc(b, y):.3f} | {auc(pn, y):.3f} | {strat_auc(b, y, terc):.3f} | "
              f"{lodo_auc(np.c_[pn, pk], y, days):.3f} → {lodo_auc(np.c_[pn, pk, b], y, days):.3f} |")
        Yf = Y5[Y5.scen == "FAST"].set_index("variant")
        Ymask = pd.read_csv(OUT + "_yr5_variants.csv.gz")
        keep_keys = set(Fb.index)
        Ymask = Ymask[Ymask.key.isin(keep_keys)]
        mf = A[A.scen == "FAST"].set_index("variant").d
        agree = np.mean([np.sign(mf[v]) == np.sign(Yf.loc[v, "d"]) for v in RN.VARIANTS if abs(mf[v]) > 1e-9])
        cor = np.corrcoef(mf.reindex(Yf.index).values, Yf.d.values)[0, 1]
        W(f"\nFAST grid: master vs yr5 Δ sign agreement {agree * 100:.0f}% of non-zero variants; correlation of per-variant Δ {cor:+.2f}. "
          f"yr5 variants Δ > 0: {(Yf.d > 0).sum()}/{len(Yf)}; nominal CI > 0: {(Yf.lo95 > 0).sum()}; CI < 0: {(Yf.hi95 < 0).sum()}.\n")
        W("| candidate | master Δ FAST | yr5 Δ FAST [95 % CI] | yr5 per seed | yr5 changed |"); W("|---|---|---|---|---|")
        for v in cand:
            r = Yf.loc[v]
            W(f"| `{v}` | {mf[v]:+.4f} | {r.d:+.4f} [{r.lo95:+.3f}, {r.hi95:+.3f}] | {r.seeds} | {r.n_changed} |")
        Y5.to_csv(OUT + "_yr5_summary.csv", index=False)

    # ───────── §5 latency
    W("\n## G. Latency realism — the candidates under IDEAL / FAST / SLOW (KEEP master Δ %/fill [95 % CI])\n")
    W("| variant | IDEAL (trigger price, no slip) | FAST (BTC tick: T + 8 s · BTC 1m: close of the bar containing T · slip 0.02) | SLOW (≥ 60 s: tick T + 60 s → next print · 1m next close ≥ T + 60 s · slip 0.10) |"); W("|---|---|---|---|")
    for v in cand:
        cells = []
        for sc in ("IDEAL", "FAST", "SLOW"):
            r = A[(A.scen == sc) & (A.variant == v)].iloc[0]
            cells.append(f"{r.d:+.4f} [{r.lo95:+.3f}, {r.hi95:+.3f}]")
        W(f"| `{v}` | " + " | ".join(cells) + " |")

    # ───────── §H robustness: anchor the changed MASTER fills on the RECORDED outcome y instead of the BASE replica
    W("\n## H. Replica-anchoring check — Δ with the recorded outcome as the 'before' on changed master fills (FAST)\n")
    W("The grid prices Δ = variant − BASE replica. On a changed fill whose replica misses the live/recorded exit, Δ inherits that miss. "
      "Here Δ_y = variant − recorded y on the changed fills (0 elsewhere).\n")
    W("| variant | Δ vs replica | Δ_y vs recorded [95 % CI] | G6 Δ → G6 Δ_y | changed fills with \\|BASE − y\\| > 0.15 |"); W("|---|---|---|---|---|")
    Vf = V[(V.scen == "FAST") & (V.changed == 1)].merge(C[["y", "block", "era_group", "pair", "day"]], left_on="key", right_index=True)
    Vf = Vf[Vf.block == "MASTER"]
    hrows = []
    for v in cand:
        g = Vf[Vf.variant == v]
        dy = pd.Series(0.0, index=M.index); dy.loc[g.key.values] = (g.pct - g.y).values
        lo, hi, _ = boot(dy.values, M.day.values)
        g6 = (M.era_group == GROUPS[6]).values
        bad = g[(g.base_pct - g.y).abs() > 0.15]
        W(f"| `{v}` | {F_.loc[v, 'd']:+.4f} | {dy.mean():+.4f} [{lo:+.3f}, {hi:+.3f}] | {F_.loc[v, f'd_{GROUPS[6]}']:+.3f} → {dy.values[g6].mean():+.3f} | "
          + (", ".join(f"{r.pair} {r.day} (BASE {r.base_pct:+.2f} vs y {r.y:+.2f})" for r in bad.itertuples()) or "none") + " |")
        hrows.append(dict(variant=v, d=F_.loc[v, "d"], d_y=dy.mean(), lo95_y=lo, hi95_y=hi))
    pd.DataFrame(hrows).to_csv(OUT + "_anchor_check.csv", index=False)
    W("\nFrozen observe lines selected by the pre-declared rule: " + (", ".join(f"`{v}`" for v in frozen) or "none") + ".\n")
    W("Selection tie (the CI-low is a percentile of a day-bootstrap; rules that change only a handful of fills, all upward, have a "
      "CI-low of exactly 0, so the 'highest CI-low' leg ties and the pick falls to the tie-break: higher FAST Δ, then grid order): "
      + " · ".join(tie_note) + ".\n")
    W("Changed-fill detail for every candidate (FAST, all blocks): `ML_BTC_EXIT_2026-10-26_changed_fills.csv`.\n")
    Vc = V[(V.scen == "FAST") & (V.changed == 1) & V.variant.isin(cand)].merge(
        C[["y", "block", "era", "pair", "ts"]], left_on="key", right_index=True)
    Vc.to_csv(OUT + "_changed_fills.csv", index=False)

    open(OUT + "_tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print("frozen:", frozen, "opcall:", opcall); print(tie_note)
    print("verdicts:", pd.Series(verd).str.split(" ").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
