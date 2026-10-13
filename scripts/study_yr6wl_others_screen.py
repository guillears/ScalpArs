#!/usr/bin/env python3
"""YR6 winners-vs-losers, AREA = OTHERS — exhaustive separator screen per sleeve × regime with shuffled null + OOS confirmation.

  venv/bin/python scripts/study_yr6wl_others_screen.py [--perms 500] [--sleeves FLIP-short,Spike-Fade,...]

Per sleeve:
  ① per-regime table (N/seed · WR · avg % · day CI · breakeven WR · days) + 2025 top-50 split (fidelity caveat)
  ② coverage of every entry_* column per regime (≥ 80 % → main screen; < 80 % → own read, scored-only)
  ③ discovery screen in EACH regime: 1D macro (DAY units) · 1D pair (PAIR-DAY units) · 2D quadrants of the top-40 columns
     (median splits, PAIR-DAY units, top-40 reselected inside every shuffle) — shuffled-label null, per-test p + family max-|z| rank
  ④ confirmation: survivors (1D p < 0.05; 2D pooled-null p < 0.01) discovered in one bull regime → frozen thresholds applied in the
     other bull regime, day-block bootstrap P(same sign) ≥ 0.90 = CONFIRMED; bear half read separately; expected chance hits stated
  ⑤ concentration (pair / day share of the zone's loss) for every confirmed separator
Outputs: reports/YR6_WL_OTHERS_screen_<sleeve>.csv (every test), reports/YR6_WL_OTHERS_confirm.csv, reports/YR6_WL_OTHERS_screen.txt
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_yr6wl_others_common import (REP, SLEEVES, REGIMES, MIN_UNITS, load, top50_flag, candidate_columns, is_macro,
                                       units, bootstrap_diff, day_ci, zone_specs, apply_zone, screen_family)

ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=500); ap.add_argument("--sleeves", default=",".join(SLEEVES))
ap.add_argument("--top2d", type=int, default=40)
A = ap.parse_args()
MIN_DAYS = 5  # a zone living on < 5 distinct days is < 5 window observations — cannot confirm (window-units rule)
rng = np.random.default_rng(20261013)
D = load()
D["top50"] = top50_flag(D)
OUT = []
CONF = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)


def stats(g):
    ns = g.seed.nunique() if len(g) else 1
    w = g.y[g.y > 0]; l = g.y[g.y <= 0]
    be = abs(l.mean()) / (w.mean() + abs(l.mean())) * 100 if len(w) and len(l) else np.nan
    m, lo, hi = day_ci(g)
    return dict(n_seed=len(g) / 3, wr=(g.y > 0).mean() * 100, avg=m, lo=lo, hi=hi, be=be, days=g.day.nunique(),
                usd=g.usd_new.sum() / 3, pairs=g.pair.nunique())


def fmt(s):
    return (f"{s['n_seed']:6.1f}/seed · {s['days']:3d} d · WR {s['wr']:5.1f}% · avg {s['avg']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] · "
            f"BE-WR {s['be']:4.1f}% · ${s['usd']:+,.0f}/seed")


def expand_cats(g, cats):
    """categorical entry_* → 0/1 dummy columns (level vs rest)."""
    new = []
    for c in cats:
        for v in g[c].dropna().astype(str).value_counts().index[:8]:
            nm = f"{c}=={v}"; g[nm] = np.where(g[c].isna(), np.nan, (g[c].astype(str) == v).astype(float)); new.append(nm)
    return new


def unit_frame(g, cols, macro):
    key = "day" if macro else "pd_key"
    X = g[[key] + cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["y"] = g.y.values
    return X.groupby(key).mean()


def run_1d(g, cols, macro, specs_from=None):
    U = unit_frame(g, cols, macro)
    Y = U.y.values
    tests = []
    for c in cols:
        specs = specs_from[c] if specs_from is not None else zone_specs(U[c], c, macro)
        for sp in specs:
            if sp[1].startswith(">=0") or sp[1].startswith(">=med"):
                continue  # exact complements of <0 / <med — same test, flipped sign
            x = U[c].values
            s = ~np.isnan(x); m = apply_zone(x, sp[2], sp[3])
            if (m & s).sum() >= MIN_UNITS and (s & ~m).sum() >= MIN_UNITS:
                tests.append((sp, m, s))
    if not tests:
        return pd.DataFrame(), None, U
    Zr, p, maxnull, Zn = screen_family(tests, Y, A.perms, rng)
    R = pd.DataFrame([dict(col=t[0][0], zone=t[0][1], op=t[0][2], thr=t[0][3], unit="day" if macro else "pair-day",
                           n_units=int((t[1] & t[2]).sum()), n_scored=int(t[2].sum()), z=z, p=pp) for t, z, pp in zip(tests, Zr, p)])
    R["fw_p"] = [((maxnull >= abs(z)).sum() + 1) / (len(maxnull) + 1) for z in R.z]
    # excess discoveries: real count of |z|>2 vs null mean count
    R.attrs["excess"] = ((np.abs(Zr) > 2).sum(), (np.abs(Zn) > 2).sum(1).mean(), np.percentile((np.abs(Zn) > 2).sum(1), 95))
    return R, Zn, U


def run_2d(g, cols, k):
    """quadrant screen on pair-day units; top-k columns by max 1D |z| (median split), reselected in every shuffle."""
    U = unit_frame(g, cols, False)
    Y = U.y.values; n = len(Y)
    meds = {c: float(np.nanmedian(U[c].values)) for c in cols}
    B = np.array([np.nan_to_num(U[c].values >= meds[c]).astype(float) for c in cols])
    S = np.array([(~np.isnan(U[c].values)).astype(float) for c in cols])
    B = B * S
    ok = [i for i, c in enumerate(cols) if np.unique(U[c].dropna()).size > 1 and 0 < B[i].sum() < S[i].sum()]
    B, S = B[ok], S[ok]; cl = [cols[i] for i in ok]
    perms = np.vstack([Y] + [rng.permutation(Y) for _ in range(A.perms)])
    res, nullz = None, []
    for pi, y in enumerate(perms):
        # 1D median-split z per column → select top-k
        s1 = B @ y; c1 = B.sum(1); sS = S @ y; cS = S.sum(1); q = S @ (y ** 2)
        mu = sS / cS; sd = np.sqrt(np.maximum(q / cS - mu ** 2, 1e-12))
        z1 = (s1 / c1 - (sS - s1) / (cS - c1)) / (sd * np.sqrt(1 / c1 + 1 / (cS - c1)))
        sel = np.argsort(-np.abs(z1))[:k]
        Bs, Ss = B[sel], S[sel]; Ns = Ss - Bs
        SS = Ss[:, None, :] * Ss[None, :, :]                       # scored on both
        nS = SS.sum(2); yS = (SS * y).sum(2); qS = (SS * y ** 2).sum(2)
        mu = yS / np.maximum(nS, 1); sd = np.sqrt(np.maximum(qS / np.maximum(nS, 1) - mu ** 2, 1e-12))
        out = []
        for qi, (Li, Lj) in enumerate(((Bs, Bs), (Bs, Ns), (Ns, Bs), (Ns, Ns))):
            M1 = Li[:, None, :] * Lj[None, :, :]
            n1 = M1.sum(2); s1 = (M1 * y).sum(2); n0 = nS - n1
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (s1 / n1 - (yS - s1) / n0) / (sd * np.sqrt(1 / n1 + 1 / n0))
            z[(n1 < MIN_UNITS) | (n0 < MIN_UNITS)] = np.nan
            iu = np.triu_indices(len(sel), 1)
            out.append((z[iu], n1[iu], iu))
        if pi == 0:
            rows = []
            for qi, (z, n1, iu) in enumerate(out):
                for a, b, zz, nn in zip(iu[0], iu[1], z, n1):
                    if np.isfinite(zz):
                        ca, cb = cl[sel[a]], cl[sel[b]]
                        rows.append(dict(col_a=ca, col_b=cb, quad=["hi·hi", "hi·lo", "lo·hi", "lo·lo"][qi], med_a=meds[ca], med_b=meds[cb],
                                         n_units=int(nn), z=zz, macro_in=is_macro(ca.split("==")[0]) or is_macro(cb.split("==")[0])))
            res = pd.DataFrame(rows)
        else:
            allz = np.concatenate([np.abs(o[0][np.isfinite(o[0])]) for o in out])
            nullz.append(allz)
    nmax = np.array([z.max() if len(z) else 0 for z in nullz])
    pooled = np.sort(np.concatenate(nullz))
    if res is not None and len(res):
        res["p_pooled"] = 1 - np.searchsorted(pooled, np.abs(res.z.values)) / len(pooled)
        res["fw_p"] = [((nmax >= abs(z)).sum() + 1) / (len(nmax) + 1) for z in res.z]
        res.attrs["excess"] = ((np.abs(res.z) > 3).sum(), np.mean([(z > 3).sum() for z in nullz]), np.percentile([(z > 3).sum() for z in nullz], 95))
    return res


def zone_mask_fill(g, col, op, thr, macro):
    x = pd.to_numeric(g[col], errors="coerce")
    if macro:
        x = x.groupby(g.day).transform("mean")
    return apply_zone(x.values, op, thr), ~np.isnan(x.values)


def zone_mask_2d(g, r):
    ma, sa = zone_mask_fill(g, r.col_a, "ge", r.med_a, False); mb, sb = zone_mask_fill(g, r.col_b, "ge", r.med_b, False)
    qa = ma if r.quad[0:2] == "hi" else ~ma; qb = mb if r.quad[3:5] == "hi" else ~mb
    return qa & qb & sa & sb, sa & sb


def read_zone(g, m, s):
    gz, gr = g[m & s], g[s & ~m]
    if len(gz) < 3 or len(gr) < 3:
        return None
    diff, pp = bootstrap_diff(g[s], m[s])
    return dict(nz=len(gz) / 3, dz=gz.day.nunique(), wrz=(gz.y > 0).mean() * 100, avgz=gz.y.mean(), avgr=gr.y.mean(), diff=diff, p_pos=pp)


def concentration(gz):
    l = gz[gz.y < 0]
    if not len(l):
        return "no losses"
    tot = l.y.sum()
    pr = l.groupby("pair").y.sum().sort_values() / tot; dy = l.groupby("day").y.sum().sort_values() / tot
    return (f"top-2 pairs {pr.iloc[:2].sum() * 100:.0f}% ({', '.join(pr.index[:2])}) · top day {dy.iloc[0] * 100:.0f}% "
            f"({dy.index[0].date()}) · top-2 days {dy.iloc[:2].sum() * 100:.0f}%")


for SL in A.sleeves.split(","):
    G = D[D.sleeve == SL].copy()
    P("=" * 140); P(f"SLEEVE {SL} — {len(G)} fills (3 seeds pooled) · {G.day.nunique()} days · {G.pair.nunique()} pairs")
    for rg in REGIMES:
        g = G[G.regime == rg]
        if not len(g):
            P(f"  {rg:13}: no fills"); continue
        P(f"  {rg:13}: {fmt(stats(g))}")
        if rg == "2025 BULL":
            for f, lab in ((True, "top-50"), (False, "non-top-50")):
                h = g[g.top50 == f]
                if len(h): P(f"      2025 {lab:10}: {fmt(stats(h))}")
    P(f"  ALL          : {fmt(stats(G))}")
    # per-seed sign consistency
    ps = G.groupby(["regime", "seed"]).y.mean().unstack().round(3)
    P("  per-seed avg %:\n" + ps.to_string())

    num, cat = candidate_columns(G)
    dums = expand_cats(G, cat)
    cols_all = num + dums
    # coverage
    cov = pd.DataFrame({rg: [pd.to_numeric(G.loc[G.regime == rg, c], errors="coerce").notna().mean() if (G.regime == rg).any() else np.nan
                             for c in cols_all] for rg in REGIMES}, index=cols_all)
    cov.to_csv(os.path.join(REP, f"YR6_WL_OTHERS_coverage_{SL}.csv"))
    low = cov[(cov < 0.8).any(axis=1) & (cov.max(axis=1) > 0)]
    P(f"  coverage: {len(cols_all)} candidate columns ({len(num)} numeric + {len(dums)} categorical dummies from {cat}); "
      f"<80 % in some regime: {len(low)} → {', '.join(f'{c} ' + '/'.join(f'{v:.0%}' if v == v else '–' for v in low.loc[c]) for c in low.index)}")

    ALLR = []
    for rg in REGIMES:
        g = G[G.regime == rg]
        if g.day.nunique() < 2 * MIN_UNITS:
            P(f"  [{rg}] {g.day.nunique()} days — below 2×{MIN_UNITS} day units: macro screen skipped (no power)")
        main = [c for c in cols_all if cov.loc[c, rg] >= 0.8]
        lowc = [c for c in cols_all if 0 < cov.loc[c, rg] < 0.8]
        mac = [c for c in main if is_macro(c.split("==")[0])]; pr = [c for c in main if not is_macro(c.split("==")[0])]
        for fam, cols, macro in (("1D-macro(day)", mac, True), ("1D-pair(pair-day)", pr, False), ("1D-lowcov(scored-only)", lowc, False)):
            if not cols or (macro and g.day.nunique() < 2 * MIN_UNITS):
                continue
            R, Zn, U = run_1d(g, cols, macro)
            if not len(R):
                P(f"  [{rg}] {fam}: no testable zones (≥{MIN_UNITS} units each side)"); continue
            R["family"] = fam; R["regime"] = rg; R["sleeve"] = SL
            ex = R.attrs["excess"]
            P(f"  [{rg}] {fam}: {len(R)} tests on {len(U)} units · |z|>2: real {ex[0]} vs null mean {ex[1]:.1f} (95th {ex[2]:.0f}) · "
              f"best |z| {R.z.abs().max():.2f} fw-p {R.fw_p.min():.3f} · p<0.05: {(R.p < 0.05).sum()} (chance ≈ {0.05 * len(R):.1f})")
            ALLR.append(R)
        if len(pr) + len(mac) >= 4 and len(g) / 3 >= 30:
            R2 = run_2d(g, pr + mac, A.top2d)
            if R2 is not None and len(R2):
                R2["family"] = "2D"; R2["regime"] = rg; R2["sleeve"] = SL
                ex = R2.attrs["excess"]
                P(f"  [{rg}] 2D quadrants: {len(R2)} tests · |z|>3: real {ex[0]} vs null mean {ex[1]:.1f} (95th {ex[2]:.0f}) · best |z| {R2.z.abs().max():.2f} "
                  f"fw-p {R2.fw_p.min():.3f}")
                ALLR.append(R2)
    if not ALLR:
        continue
    T = pd.concat(ALLR, ignore_index=True)
    T.to_csv(os.path.join(REP, f"YR6_WL_OTHERS_screen_{SL}.csv"), index=False)

    # ── discovery → confirmation ──
    bulls = [r for r in ("2025 BULL", "2026-H2 BULL") if (G.regime == r).any()]
    pairs = [(a, b) for a in bulls for b in bulls if a != b] + [("2026-H1 BEAR", None)]
    for disc, conf in pairs:
        S = T[(T.regime == disc)]
        if not len(S):
            continue
        s1 = S[(S.family != "2D") & (S.p < 0.05)]
        s2 = S[(S.family == "2D") & (S.p_pooled < 0.01)] if "p_pooled" in S else S.iloc[0:0]
        surv = pd.concat([s1.sort_values("p").head(40), s2.reindex(s2.z.abs().sort_values(ascending=False).index).head(25)])
        if not len(surv):
            P(f"  {disc} → {conf}: no survivors"); continue
        hits = 0; tested = 0
        P(f"  ── discovery {disc} ({len(surv)} survivors) → confirm in {conf or 'both bull regimes (bear-discovered: read only)'} · bear read alongside")
        for _, r in surv.iterrows():
            gd = G[G.regime == disc]
            if r.family == "2D":
                md, sd_ = zone_mask_2d(gd, r); name = f"{r.col_a}{'≥' if r.quad[:2] == 'hi' else '<'}{r.med_a:.4g} ∧ {r.col_b}{'≥' if r.quad[3:] == 'hi' else '<'}{r.med_b:.4g}"
                mac = bool(r.macro_in)
            else:
                mac = r.unit == "day"; md, sd_ = zone_mask_fill(gd, r.col, r.op, r.thr, mac); name = f"{r.col} {r.zone}"
            zd = read_zone(gd, md, sd_)
            row = dict(sleeve=SL, disc=disc, conf=conf, family=r.family, test=name, z=r.z, p=r.p if r.family != "2D" else r.p_pooled,
                       fw_p=r.fw_p, macro=mac)
            if zd: row.update({f"d_{k}": v for k, v in zd.items()})
            for lab, rr in (("r25", "2025 BULL"), ("rH1", "2026-H1 BEAR"), ("rH2", "2026-H2 BULL")):
                if not (G.regime == rr).any():
                    continue
                gc = G[G.regime == rr]
                mc, sc = zone_mask_2d(gc, r) if r.family == "2D" else zone_mask_fill(gc, r.col, r.op, r.thr, mac)
                zc = read_zone(gc, mc, sc)
                if zc:
                    row.update({f"{lab}_{k}": v for k, v in zc.items()})
                    if rr == conf: row.update({f"c_{k}": v for k, v in zc.items()})
                    if rr == "2026-H1 BEAR": row.update({f"bear_{k}": v for k, v in zc.items()})
            if conf and "c_diff" in row and "d_diff" in row:
                tested += 1
                same = (np.sign(row["c_diff"]) == np.sign(r.z)) and (np.sign(row["d_diff"]) == np.sign(r.z))  # unit-level z sign rules; fill-level disc diff must agree
                pc = row["c_p_pos"] if r.z > 0 else 1 - row["c_p_pos"]
                row["confirmed"] = bool(same and pc >= 0.90 and row["c_nz"] * 3 >= MIN_UNITS and row["c_dz"] >= MIN_DAYS and row.get("d_dz", 0) >= MIN_DAYS)
                row["underpowered"] = bool(row["c_dz"] < MIN_DAYS or row.get("d_dz", 0) < MIN_DAYS)
                row["c_P_same"] = pc
                hits += row["confirmed"]
                if row["confirmed"]:
                    gz = gd[md & sd_]; row["conc_disc"] = concentration(gz)
                    gzc = G[G.regime == conf]; mcc, scc = (zone_mask_2d(gzc, r) if r.family == "2D" else zone_mask_fill(gzc, r.col, r.op, r.thr, mac))
                    row["conc_conf"] = concentration(gzc[mcc & scc])
            CONF.append(row)
        if conf:
            P(f"     confirmed {hits}/{tested} (underpowered: zone on < {MIN_DAYS} days in disc or conf = {sum(1 for c in CONF if c['sleeve'] == SL and c['disc'] == disc and c.get('underpowered'))}) (chance ≈ {0.10 * tested:.1f} at P≥0.90 one-sided if no real effect)")
    C = pd.DataFrame([c for c in CONF if c["sleeve"] == SL])
    if len(C) and "confirmed" in C:
        for _, r in C[C.confirmed.fillna(False).astype(bool)].iterrows():
            P(f"     ✔ [{r.disc}→{r.conf}] {r.family} {r.test}: disc z {r.z:+.2f} p {r.p:.3f} fw {r.fw_p:.3f} | disc zone {r.d_nz:.1f}/seed "
              f"{r.d_dz}d WR {r.d_wrz:.0f}% avg {r.d_avgz:+.3f} vs rest {r.d_avgr:+.3f} | conf zone {r.c_nz:.1f}/seed {r.c_dz}d WR {r.c_wrz:.0f}% "
              f"avg {r.c_avgz:+.3f} vs {r.c_avgr:+.3f} P {r.c_P_same:.2f}"
              + (f" | bear zone {r.bear_nz:.1f}/seed avg {r.bear_avgz:+.3f} vs {r.bear_avgr:+.3f}" if "bear_nz" in r and r.bear_nz == r.bear_nz else "")
              + f"\n         conc disc: {r.conc_disc} · conc conf: {r.conc_conf}")

pd.DataFrame(CONF).to_csv(os.path.join(REP, "YR6_WL_OTHERS_confirm.csv"), index=False)
open(os.path.join(REP, "YR6_WL_OTHERS_screen.txt"), "w").write("\n".join(OUT) + "\n")
print("wrote reports/YR6_WL_OTHERS_screen.txt + _confirm.csv + _screen_<sleeve>.csv + _coverage_<sleeve>.csv")
