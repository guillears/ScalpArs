#!/usr/bin/env python3
"""MOMENTUM LONG checklist item ② (2026-10-08): every MACRO / REGIME variable at SIGN granularity first, then median, terciles,
quintiles — in DAY units (day-clustered bootstrap, ≥ days per bucket), with and without the washed-out window, on yr5 and master.

Per variable and granularity a variable is "ALIVE" (not refuted) when some bucket-vs-rest split has, on yr5 (all OR ex-washed):
  ① |z| above that split's own 95th percentile under the circular day-block shift null (per-mask p < 0.05),
  ② the same Δ sign in both yr5 halves, and ③ master (ex-B1, same cohort wash setting) not contradicting it (same sign, or < 10
  master fills in the bucket = untestable). Refuted = dead at EVERY granularity in BOTH yr5 cohorts.
Chance level: the whole criterion is re-run on 300 circularly shifted outcome vectors (yr5 and master shifted independently) →
the null distribution of the number of ALIVE variables.
Also: day-level Spearman ρ (day-mean ML % vs day-mean variable, yr5, seeds pooled) with a circular day-shift null.
Out: reports/study_ml_checklist_regime_buckets.csv, reports/study_ml_checklist_regime_verdict.csv, _regime_summary.txt
"""
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
import study_ml_checklist_scan as SC

GRAN = ["sign", "median", "tercile", "quintile"]


def bucket_edges(r, gran):
    if gran == "sign":
        return [0.0]
    q = {"median": [0.5], "tercile": [1 / 3, 2 / 3], "quintile": [0.2, 0.4, 0.6, 0.8]}[gran]
    return list(r.quantile(q).values)


def boot_p(pct, day, B=2000, seed=5):
    g = pd.DataFrame({"p": pct, "d": day}).groupby("d").p.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values; k = len(s)
    if k < 2:
        return np.nan, np.nan, np.nan
    i = np.random.default_rng(seed).integers(0, k, (B, k)); m = s[i].sum(1) / c[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m < 0).mean()


def bucket_table(d, var, gran, edges, ns):
    v = d[var].values; ok = ~np.isnan(v); b = np.digitize(v, edges, right=True)
    rows = []
    for k in range(len(edges) + 1):
        z = ok & (b == k)
        if z.sum() == 0:
            continue
        lo, hi, p = boot_p(d.pct.values[z], d.day.values[z])
        rows.append(dict(var=var, gran=gran, bucket=k, lo_edge=(edges[k - 1] if k > 0 else -np.inf), hi_edge=(edges[k] if k < len(edges) else np.inf),
                         N=z.sum() / ns, days=len(set(d.day.values[z])), WR=(d.pct.values[z] > 0).mean() * 100, avg=d.pct.values[z].mean(),
                         ci_lo=lo, ci_hi=hi, p_neg=p, avg_rest=d.pct.values[ok & ~z].mean()))
    return rows


def split_masks(d, var, gran, edges):
    """bucket-vs-rest boolean masks (zone, valid) for every bucket of a granularity."""
    v = d[var].values; ok = ~np.isnan(v); b = np.digitize(v, edges, right=True)
    return [((ok & (b == k)), ok) for k in range(len(edges) + 1)]


def welch(y, Z, V):
    z, _, _ = SC.zstats(Z, V, y[:, None] if y.ndim == 1 else y)
    return z


def main():
    D = SC.load(); CO = SC.cohorts(D)
    mac = [c for c in SC.usable(CO["Y_all"], SC.MACRO)]
    lines = [f"MACRO variables tested: {len(mac)}"]
    # ---------- bucket tables ----------
    rows = []
    for lab, d in CO.items():
        ns = 3 if lab.startswith("Y") else 1
        ref = CO["Y_all"]
        for var in mac:
            if d[var].notna().mean() < 0.5:
                continue
            for g in GRAN:
                r = ref[var].dropna()
                if g == "sign" and not ((r > 0).mean() >= 0.10 and (r <= 0).mean() >= 0.10):
                    continue
                for row in bucket_table(d, var, g, bucket_edges(r, g), ns):
                    row["cohort"] = lab; rows.append(row)
    B = pd.DataFrame(rows); B.to_csv("reports/study_ml_checklist_regime_buckets.csv", index=False)

    # ---------- ALIVE criterion with a calibrated null ----------
    def build(d, ref):
        d = d.sort_values("o").reset_index(drop=True)
        spec, Zs, Vs = [], [], []
        for var in mac:
            if d[var].notna().mean() < 0.5:
                continue
            r = ref[var].dropna()
            for g in GRAN:
                if g == "sign" and not ((r > 0).mean() >= 0.10 and (r <= 0).mean() >= 0.10):
                    continue
                for k, (z, v) in enumerate(split_masks(d, var, g, bucket_edges(r, g))):
                    spec.append((var, g, k)); Zs.append(z); Vs.append(v)
        return d, spec, np.array(Zs).T, np.array(Vs).T

    out = {}
    for ylab, mlab in (("Y_all", "M_all"), ("Y_exw", "M_exw")):
        ref = CO["Y_all"]
        y, spec, Zy, Vy = build(CO[ylab], ref)
        m, specm, Zm, Vm = build(CO[mlab], ref)
        mi = {s: i for i, s in enumerate(specm)}
        h1 = (y.half == "H1").values
        rng = np.random.default_rng(3); n = len(y); nm = len(m)
        # per-split null 95th of |z| on yr5 (2000 shifts)
        ZP = []
        for _ in range(20):
            sh = rng.integers(int(0.05 * n), int(0.95 * n), 100)
            ZP.append(np.abs(np.nan_to_num(welch(np.stack([np.roll(y.pct.values, s) for s in sh], 1), Zy, Vy))))
        q95 = np.quantile(np.concatenate(ZP, 1), 0.95, axis=1)
        valid_n = (Zy.sum(0) >= 45) & ((Vy & ~Zy).sum(0) >= 45)

        def alive(yv, mv):
            z = welch(yv, Zy, Vy)[:, 0]
            zh1 = welch(np.where(h1, yv, 0.0), Zy & h1[:, None], Vy & h1[:, None])[:, 0]
            zh2 = welch(np.where(~h1, yv, 0.0), Zy & ~h1[:, None], Vy & ~h1[:, None])[:, 0]
            sig = (np.abs(np.nan_to_num(z)) > q95) & valid_n & (np.sign(zh1) == np.sign(z)) & (np.sign(zh2) == np.sign(z))
            zm = np.full(len(spec), np.nan); nmz = np.zeros(len(spec))
            for i, s in enumerate(spec):
                j = mi.get(s)
                if j is None:
                    continue
                zz, vv = Zm[:, j], Vm[:, j]; nmz[i] = zz.sum()
                if zz.sum() >= 1 and (vv & ~zz).sum() >= 1:
                    zm[i] = mv[zz].mean() - mv[vv & ~zz].mean()
            ok_m = (nmz < 10) | np.isnan(zm) | (np.sign(zm) == np.sign(z))
            return sig & ok_m, z, zm, nmz, sig

        a, z, zm, nmz, sig = alive(y.pct.values, m.pct.values)
        S = pd.DataFrame(spec, columns=["var", "gran", "bucket"]).assign(z=z, q95=q95, sig_yr5=sig, master_dlt=zm, master_n=nmz, alive=a)
        V = S.groupby(["var", "gran"]).alive.any().unstack().reindex(columns=GRAN)
        V["alive_any"] = V.any(axis=1)
        out[ylab] = (S, V)
        # null: number of alive variables under shifted outcomes
        nulls = []
        for _ in range(300):
            ys = np.roll(y.pct.values, rng.integers(int(0.05 * n), int(0.95 * n)))
            ms = np.roll(m.pct.values, rng.integers(int(0.05 * nm), int(0.95 * nm)))
            an, *_ = alive(ys, ms)
            nulls.append(pd.Series(an).groupby([s[0] for s in spec]).any().sum())
        nulls = np.array(nulls)
        obs = int(V.alive_any.sum())
        lines.append(f"[{ylab} vs {mlab}] ALIVE variables: {obs}/{len(V)} · null (300 shifts) median {np.median(nulls):.0f}, 95th {np.percentile(nulls,95):.0f}, "
                     f"P(null ≥ obs) = {(nulls >= obs).mean():.3f}")
        lines.append("   alive at sign: " + ", ".join(V.index[V["sign"] == True]))
        lines.append("   alive (any granularity): " + ", ".join(V.index[V.alive_any]))
        S.to_csv(f"reports/study_ml_checklist_regime_splits_{ylab}.csv", index=False)
    VV = out["Y_all"][1].add_suffix("_Yall").join(out["Y_exw"][1].add_suffix("_Yexw"), how="outer")
    VV["refuted"] = ~(VV.alive_any_Yall.fillna(False) | VV.alive_any_Yexw.fillna(False))
    VV.to_csv("reports/study_ml_checklist_regime_verdict.csv")
    lines.append(f"refuted at every granularity in both yr5 cohorts: {int(VV.refuted.sum())}/{len(VV)} · not refuted: {', '.join(VV.index[~VV.refuted])}")

    # ---------- day-level Spearman (yr5) ----------
    rows = []
    for lab in ("Y_all", "Y_exw"):
        d = CO[lab]; g = d.groupby("day"); dm = g.pct.mean()
        rng = np.random.default_rng(9)
        for var in mac:
            x = g[var].mean().reindex(dm.index)
            ok = x.notna().values
            if ok.sum() < 30:
                continue
            xr, yr_ = x[ok].rank().values, dm[ok].rank().values
            rho = np.corrcoef(xr, yr_)[0, 1]; k = len(xr)
            null = np.array([np.corrcoef(xr, np.roll(yr_, s))[0, 1] for s in rng.integers(int(0.05 * k), int(0.95 * k), 2000)])
            rows.append(dict(cohort=lab, var=var, days=k, rho=rho, p_two=((np.abs(null) >= abs(rho)).sum() + 1) / 2001))
    R = pd.DataFrame(rows)
    R["p_bonf"] = (R.p_two * R.groupby("cohort")["var"].transform("count")).clip(upper=1)
    R.to_csv("reports/study_ml_checklist_regime_dayrho.csv", index=False)
    for lab in ("Y_all", "Y_exw"):
        r = R[R.cohort == lab].sort_values("p_two")
        lines.append(f"[day-level ρ {lab}] {len(r)} vars · p<0.05: {int((r.p_two<0.05).sum())} (chance {0.05*len(r):.1f}) · Bonferroni<0.05: {int((r.p_bonf<0.05).sum())} · top: " +
                     "; ".join(f"{a} ρ={b:+.2f} p={c:.3f}" for a, b, c in r[["var", "rho", "p_two"]].head(8).values))
    open("reports/study_ml_checklist_regime_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
