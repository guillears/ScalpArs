"""B25-prep MOMENTUM-LONG winners-vs-losers screen (2026-10-26). Reads only local files; no network.

Blocks: KEEP = master non-probe kept by today's stack (y = stack_pct, discovery) · TRAD = master non-probe as traded (robustness) ·
B25 = provisional current batch (descriptive) · PROBE = B1/B2 probes (separate line) · REPLAY = yr6 2-year engine replay MOM-long
(refute-only, stamped entry_* columns).
Steps: ① coverage variable × era group · ② 1D sign / median / tercile zones + quintile tables · ③ exhaustive all-pairs 2D
(median × median, 4 quadrants) · shuffled null (circular shifts of the time-sorted outcome, 1D 1000, 2D 200) with per-test and
family-wise p · OOS: leave-one-era-group-out + chronological halves (thresholds re-fit on the training part) + replay sign ·
④ macro variables in DAY units, per-pair / per-day concentration · ⑤ expectancy bar (filters) · ⑥ Pattern-W bar (multipliers).
Outputs: reports/PREP_B25_ML_WL_*.csv / .txt (the .md is written by study_ml_wl_b25prep_report.py).

    venv/bin/python -W ignore scripts/study_ml_wl_b25prep_analyse.py
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_wl_b25prep_data import (DASHBOARD_MAP, DURING_1015, ROOT, POST, feature_columns, is_macro, load_all)  # noqa: E402

OUT = os.path.join(ROOT, "reports/PREP_B25_ML_WL_")
RNG = np.random.default_rng(20261026)
S1, S2 = 1000, 200
MIN_IN = 8
ERA_ORDER = ["G0 BASE", "G1 B1", "G2 B2", "G3 B3", "G4 B4-B11", "G5 B12-B16", "G6 B17-B24"]


# ------------------------------------------------------------------ helpers
def be_wr(y):
    y = np.asarray(y, float); w, l = y[y > 0], y[y <= 0]
    return abs(l.mean()) / (w.mean() + abs(l.mean())) if len(w) and len(l) else np.nan


def day_boot_mean(y, day, n=4000, seed=1):
    y = np.asarray(y, float); u, inv = np.unique(np.asarray(day), return_inverse=True)
    if len(u) < 2:
        return y.mean(), np.nan, np.nan, np.nan
    s = np.bincount(inv, weights=y); c = np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return y.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs < 0).mean()


def day_boot_delta(y, z, day, n=2000, seed=2):
    y = np.asarray(y, float); z = np.asarray(z, bool); u, inv = np.unique(np.asarray(day), return_inverse=True)
    si = np.bincount(inv, y * z, len(u)); ci = np.bincount(inv, z.astype(float), len(u))
    so = np.bincount(inv, y * ~z, len(u)); co = np.bincount(inv, (~z).astype(float), len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = si[idx].sum(1) / ci[idx].sum(1) - so[idx].sum(1) / co[idx].sum(1)
    bs = bs[np.isfinite(bs)]
    return (bs > 0).mean() if len(bs) else np.nan


def concentration(y, keys):
    """share of the zone's net loss carried by its worst key (day or pair): worst key net Σ / Σ of negative key nets."""
    s = pd.Series(np.asarray(y, float)).groupby(np.asarray(keys)).sum()
    neg = s[s < 0]
    return (neg.min() / neg.sum()) if len(neg) else 0.0


def shifts(y, S, rng):
    n = len(y); k = rng.integers(1, n, size=S)
    return np.stack([np.roll(y, int(s)) for s in k], 1)


def zone_masks(x, thr):
    """returns dict name -> (zone bool array) for a numeric x with frozen thresholds thr."""
    z = {}
    v = np.isfinite(x)
    if thr.get("binary"):
        z["==1"] = v & (x == 1)
        z["==0"] = v & (x == 0)
        return z
    if thr.get("sign"):
        z[">0"] = v & (x > 0); z["<=0"] = v & (x <= 0)
    z[f">med({thr['med']:.4g})"] = v & (x > thr["med"]); z[f"<=med({thr['med']:.4g})"] = v & (x <= thr["med"])
    z[f"<=q33({thr['q33']:.4g})"] = v & (x <= thr["q33"]); z[f">q67({thr['q67']:.4g})"] = v & (x > thr["q67"])
    return z


def fit_thr(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    u = np.unique(x)
    if len(u) < 2:
        return None
    if set(u) <= {0.0, 1.0}:
        return {"binary": True}
    t = {"med": np.median(x), "q33": np.quantile(x, 1 / 3), "q67": np.quantile(x, 2 / 3)}
    t["sign"] = (x > 0).mean() >= 0.05 and (x <= 0).mean() >= 0.05
    return t


def zone_label_to_mask(x, label, thr):
    return zone_masks(x, thr)[label]


def zone_family(label):
    return label.split("(")[0]


def apply_family(x, fam, thr):
    """mask for a zone family with (possibly re-fitted) thresholds thr."""
    for k, v in zone_masks(x, thr).items():
        if zone_family(k) == fam:
            return v
    return None


# ------------------------------------------------------------------ load
t0 = time.time()
D = load_all()
FEATS = feature_columns(D)
master = D[D.src == "master"]
KEEP = master[~master.is_probe & master.stack_keep & master.y_keep.notna()].sort_values("ts").reset_index(drop=True)
TRAD = master[~master.is_probe & master.y_trad.notna()].sort_values("ts").reset_index(drop=True)
B25 = D[(D.era == "B25") & ~D.is_probe].sort_values("ts").reset_index(drop=True)
PROBE = master[master.is_probe].sort_values("ts").reset_index(drop=True)
KEEP["y"] = KEEP.y_keep.astype(float); TRAD["y"] = TRAD.y_trad.astype(float); B25["y"] = B25.y_trad.astype(float)
PROBE["y"] = PROBE.y_trad.astype(float)
BE = be_wr(KEEP.y)
lines = []
P = lines.append


def summ(d, y="y"):
    y_ = d[y].astype(float)
    if not len(d):
        return dict(N=0)
    m, lo, hi, pneg = day_boot_mean(y_, d.day)
    usd = pd.to_numeric(d.get("stack_pnl"), errors="coerce").where(d.get("stack_keep", True)).sum() if "stack_pnl" in d else np.nan
    return dict(N=len(d), days=d.day.nunique(), wins=int((y_ > 0).sum()), WR=(y_ > 0).mean(), avg=m, lo=lo, hi=hi,
                sum_pct=y_.sum(), usd=usd, be_wr=be_wr(y_), avg_win=y_[y_ > 0].mean(), avg_loss=y_[y_ <= 0].mean())


def fmt_s(s):
    if not s.get("N"):
        return "N 0"
    return (f"N {s['N']:3d} · {s['days']:3d} d · WR {s['WR']*100:4.0f}% · avg {s['avg']:+.3f} [{s['lo']:+.3f},{s['hi']:+.3f}] · "
            f"Σ {s['sum_pct']:+7.2f}% · BE-WR {s['be_wr']*100 if np.isfinite(s['be_wr']) else float('nan'):.0f}%")


P("# B25-prep ML W/L — engine output (scripts/study_ml_wl_b25prep_analyse.py)\n")
P("## S0 blocks (MOMENTUM LONG CLOSED; % at 1×; 95% DAY-bootstrap CI)")
for nm, d in [("KEEP master (today's stack)", KEEP), ("TRAD master (as traded)", TRAD), ("B25 provisional", B25),
              ("PROBE (B1/B2, separate)", PROBE)]:
    P(f"  {nm:30s} {fmt_s(summ(d))}")
P(f"  ML breakeven WR on KEEP = {BE*100:.1f}% (avg win {KEEP.y[KEEP.y>0].mean():+.3f} / avg loss {KEEP.y[KEEP.y<=0].mean():+.3f})")
kw = KEEP[~KEEP.washed]; tw = TRAD[~TRAD.washed]
P(f"  KEEP without washed-out Jun-18→Jul-2   {fmt_s(summ(kw))}")
P(f"  KEEP washed-out window only            {fmt_s(summ(KEEP[KEEP.washed]))}")
P(f"  TRAD without washed-out                {fmt_s(summ(tw))}")
P("\n  per era group (KEEP | TRAD):")
for g in ERA_ORDER:
    P(f"   {g:12s} KEEP {fmt_s(summ(KEEP[KEEP.era_group == g]))}")
    P(f"   {'':12s} TRAD {fmt_s(summ(TRAD[TRAD.era_group == g]))}")
P(f"   per-era KEEP: " + " · ".join(f"{e} {len(g)}·{(g.y>0).mean()*100:.0f}%·{g.y.mean():+.2f}" for e, g in KEEP.groupby("era", sort=False)))

# ------------------------------------------------------------------ ① coverage
cov_rows = []
groups = ERA_ORDER + ["B25 (prov.)"]
ALL_NP = D[~D.is_probe]
for c in FEATS:
    x = pd.to_numeric(ALL_NP[c], errors="coerce")
    r = {"variable": c, "macro": is_macro(c)}
    for g in groups:
        m = ALL_NP.era_group == g
        r[f"{g} scored"] = int(x[m].notna().sum()); r[f"{g} N"] = int(m.sum())
    xk = pd.to_numeric(KEEP[c], errors="coerce")
    r["KEEP scored"] = int(xk.notna().sum()); r["KEEP N"] = len(KEEP); r["KEEP cov"] = xk.notna().mean()
    r["KEEP distinct"] = int(xk.nunique())
    cov_rows.append(r)
COV = pd.DataFrame(cov_rows).sort_values(["KEEP cov", "variable"], ascending=[False, True])
COV.to_csv(OUT + "coverage.csv", index=False)
SCREEN = [c for c in FEATS if pd.to_numeric(KEEP[c], errors="coerce").notna().sum() >= 30
          and pd.to_numeric(KEEP[c], errors="coerce").nunique() >= 2]
FULLCOV = [c for c in SCREEN if pd.to_numeric(KEEP[c], errors="coerce").notna().mean() >= 0.80]
LOWCOV = [c for c in SCREEN if c not in FULLCOV]
UNTESTABLE = COV[~COV.variable.isin(SCREEN)]
P(f"\n## S1 coverage: {len(FEATS)} candidate variables · screened {len(SCREEN)} (≥30 scored KEEP fills, ≥2 values) · "
  f"full-coverage (≥80% of KEEP) {len(FULLCOV)} · low-coverage scored-only {len(LOWCOV)} · NOT testable {len(UNTESTABLE)}")

# ------------------------------------------------------------------ ② 1D
y = KEEP.y.values.astype(float); day = KEEP.day.values; n = len(y)
YS1 = shifts(y, S1, RNG)
X = {c: pd.to_numeric(KEEP[c], errors="coerce").values.astype(float) for c in SCREEN}
THR = {c: fit_thr(X[c]) for c in SCREEN}
tests = []
for c in SCREEN:
    if THR[c] is None:
        continue
    v = np.isfinite(X[c])
    for lab, z in zone_masks(X[c], THR[c]).items():
        o = v & ~z
        if z.sum() < max(MIN_IN, 0.05 * v.sum()) or o.sum() < MIN_IN:
            continue
        tests.append((c, lab, z, o))
Zi = np.stack([t[2] for t in tests], 1).astype(float); Zo = np.stack([t[3] for t in tests], 1).astype(float)
ni, no = Zi.sum(0), Zo.sum(0)
obs = (Zi.T @ y) / ni - (Zo.T @ y) / no
null = (Zi.T @ YS1) / ni[:, None] - (Zo.T @ YS1) / no[:, None]
p_test = ((np.abs(null) >= np.abs(obs)[:, None]).sum(1) + 1) / (S1 + 1)
fw_max = np.abs(null).max(0)
p_fw = ((fw_max[None, :] >= np.abs(obs)[:, None]).sum(1) + 1) / (S1 + 1)
R1 = []
for i, (c, lab, z, o) in enumerate(tests):
    yz = y[z]
    R1.append(dict(variable=c, zone=lab, family=zone_family(lab), macro=is_macro(c), lowcov=c in LOWCOV, N_in=int(z.sum()),
                   N_out=int(o.sum()), days_in=len(set(day[z])), WR_in=(yz > 0).mean(), avg_in=yz.mean(),
                   avg_out=y[o].mean(), delta=obs[i], p_null=p_test[i], p_fw=p_fw[i]))
R1 = pd.DataFrame(R1)
P(f"\n## S2 1D: {len(R1)} zone tests on KEEP (N={n}); family-wise 95% |Δ| threshold = {np.quantile(fw_max, 0.95):.3f}; "
  f"tests with p_null<0.05: {(R1.p_null < 0.05).sum()} (chance ≈ {0.05*len(R1):.0f}); p_fw<0.05: {(R1.p_fw < 0.05).sum()}")

# ------------------------------------------------------------------ ③ exhaustive 2D
YS2 = YS1[:, :S2]
V2 = [c for c in FULLCOV if THR[c] is not None]
bins = {}
for c in V2:
    x = X[c]; t = THR[c]
    hi = (x == 1) if t.get("binary") else (x > t["med"])
    bins[c] = (np.isfinite(x), hi)
pairs = list(itertools.combinations(V2, 2))
meta, cols_in, cols_out = [], [], []
for a, b in pairs:
    va, ha = bins[a]; vb, hb = bins[b]; v = va & vb
    for qa, qb in ((1, 1), (1, 0), (0, 1), (0, 0)):
        z = v & (ha == bool(qa)) & (hb == bool(qb))
        o = v & ~z
        if z.sum() < 10 or o.sum() < 10:
            continue
        meta.append((a, qa, b, qb)); cols_in.append(z); cols_out.append(o)
P(f"\n## S3 exhaustive 2D: {len(V2)} full-coverage variables → {len(pairs)} pairs → {len(meta)} valid quadrant tests (median × median)")
obs2 = np.empty(len(meta)); null_max = np.zeros(S2); ptest2 = np.empty(len(meta)); CH = 4000
for s in range(0, len(meta), CH):
    zi = np.stack(cols_in[s:s + CH], 1).astype(np.float32); zo = np.stack(cols_out[s:s + CH], 1).astype(np.float32)
    a_, b_ = zi.sum(0), zo.sum(0)
    ob = (zi.T @ y) / a_ - (zo.T @ y) / b_
    nl = (zi.T @ YS2) / a_[:, None] - (zo.T @ YS2) / b_[:, None]
    obs2[s:s + CH] = ob
    ptest2[s:s + CH] = ((np.abs(nl) >= np.abs(ob)[:, None]).sum(1) + 1) / (S2 + 1)
    null_max = np.maximum(null_max, np.abs(nl).max(0))
pfw2 = np.array([((null_max >= abs(o)).sum() + 1) / (S2 + 1) for o in obs2])
R2 = pd.DataFrame([dict(var_a=a, hi_a=qa, var_b=b, hi_b=qb, N_in=int(cols_in[i].sum()), days_in=len(set(day[cols_in[i]])),
                        WR_in=(y[cols_in[i]] > 0).mean(), avg_in=y[cols_in[i]].mean(), avg_out=y[cols_out[i]].mean(),
                        delta=obs2[i], p_null=ptest2[i], p_fw=pfw2[i], macro_a=is_macro(a), macro_b=is_macro(b))
                   for i, (a, qa, b, qb) in enumerate(meta)])
# null calibration: how many tests at p<0.01 would chance give vs observed
P(f"  2D tests p_null<0.01: {(R2.p_null < 0.01).sum()} (chance ≈ {0.01*len(R2):.0f}) · p_null<0.05: {(R2.p_null < 0.05).sum()} "
  f"(chance ≈ {0.05*len(R2):.0f}) · family-wise 95% |Δ| = {np.quantile(null_max, 0.95):.3f} · p_fw<0.05: {(R2.p_fw < 0.05).sum()}")
# second null: how many p<0.01 tests does a SHUFFLED label give (distribution over 20 shifts)
cnt = []
for j in range(20):
    ysh = YS2[:, j]
    k = 0
    for s in range(0, len(meta), CH):
        zi = np.stack(cols_in[s:s + CH], 1).astype(np.float32); zo = np.stack(cols_out[s:s + CH], 1).astype(np.float32)
        ob = (zi.T @ ysh) / zi.sum(0) - (zo.T @ ysh) / zo.sum(0)
        nl = (zi.T @ YS2) / zi.sum(0)[:, None] - (zo.T @ YS2) / zo.sum(0)[:, None]
        k += int(((((np.abs(nl) >= np.abs(ob)[:, None]).sum(1) + 1) / (S2 + 1)) < 0.01).sum())
    cnt.append(k)
P(f"  shuffled-label replicate: # tests with p<0.01 under 20 shifted outcomes: median {np.median(cnt):.0f}, 95th pct "
  f"{np.percentile(cnt, 95):.0f} (observed {(R2.p_null < 0.01).sum()})")


# ------------------------------------------------------------------ OOS machinery (thresholds re-fit on the training part)
def mask_1d(df, c, fam, thr):
    x = pd.to_numeric(df[c], errors="coerce").values.astype(float)
    return apply_family(x, fam, thr), np.isfinite(x)


def mask_2d(df, a, qa, b, qb, ta, tb):
    xa = pd.to_numeric(df[a], errors="coerce").values.astype(float); xb = pd.to_numeric(df[b], errors="coerce").values.astype(float)
    ha = (xa == 1) if ta.get("binary") else (xa > ta["med"])
    hb = (xb == 1) if tb.get("binary") else (xb > tb["med"])
    v = np.isfinite(xa) & np.isfinite(xb)
    return v & (ha == bool(qa)) & (hb == bool(qb)), v


def delta(df, z, v):
    yy = df.y.values.astype(float)
    if z.sum() < 3 or (v & ~z).sum() < 3:
        return np.nan, int(z.sum())
    return yy[z].mean() - yy[v & ~z].mean(), int(z.sum())


def oos(kind, spec):
    """LOEO over era groups + chronological halves; thresholds re-fit on the training part each time."""
    same = tot = 0; parts = []
    for g in ERA_ORDER:
        tr, te = KEEP[KEEP.era_group != g], KEEP[KEEP.era_group == g]
        if kind == 1:
            c, fam = spec; t = fit_thr(pd.to_numeric(tr[c], errors="coerce").values.astype(float))
            if t is None:
                continue
            ztr, vtr = mask_1d(tr, c, fam, t); zte, vte = mask_1d(te, c, fam, t)
        else:
            a, qa, b, qb = spec
            ta = fit_thr(pd.to_numeric(tr[a], errors="coerce").values.astype(float))
            tb = fit_thr(pd.to_numeric(tr[b], errors="coerce").values.astype(float))
            if ta is None or tb is None:
                continue
            ztr, vtr = mask_2d(tr, a, qa, b, qb, ta, tb); zte, vte = mask_2d(te, a, qa, b, qb, ta, tb)
        if ztr is None or zte is None:
            continue
        dtr, _ = delta(tr, ztr, vtr); dte, nz = delta(te, zte, vte)
        if np.isfinite(dtr) and np.isfinite(dte):
            tot += 1; same += int(np.sign(dtr) == np.sign(dte)); parts.append(f"{g.split()[0]}{dte:+.2f}({nz})")
        else:
            parts.append(f"{g.split()[0]}·({nz})")
    half = KEEP.ts.iloc[len(KEEP) // 2]
    tr, te = KEEP[KEEP.ts < half], KEEP[KEEP.ts >= half]
    if kind == 1:
        c, fam = spec; t = fit_thr(pd.to_numeric(tr[c], errors="coerce").values.astype(float))
        h = (np.nan, 0, np.nan)
        if t is not None:
            ztr, vtr = mask_1d(tr, c, fam, t); zte, vte = mask_1d(te, c, fam, t)
            if ztr is not None and zte is not None:
                h = (delta(tr, ztr, vtr)[0], delta(te, zte, vte)[1], delta(te, zte, vte)[0])
    else:
        a, qa, b, qb = spec
        ta = fit_thr(pd.to_numeric(tr[a], errors="coerce").values.astype(float))
        tb = fit_thr(pd.to_numeric(tr[b], errors="coerce").values.astype(float))
        h = (np.nan, 0, np.nan)
        if ta is not None and tb is not None:
            ztr, vtr = mask_2d(tr, a, qa, b, qb, ta, tb); zte, vte = mask_2d(te, a, qa, b, qb, ta, tb)
            h = (delta(tr, ztr, vtr)[0], delta(te, zte, vte)[1], delta(te, zte, vte)[0])
    return same, tot, " ".join(parts), h


# replay (refute-only)
REP = None
try:
    from study_yr6wl_momentum_common import load_replay
    REP = load_replay(["MOM-long"])
    from study_ml_wl_b25prep_data import add_derived
    REP = add_derived(REP)
    # merge the prior-study per-fill features for the replay blocks (same definitions as master): 10-15 impulse + Oct-15 legs
    from study_ml_wl_b25prep_data import IMPULSE_1015, IMP_SCORE
    rk = list(zip(REP.pair, REP.opened_at.astype(str).str[:19], REP.seed.astype(int)))
    f_ = pd.read_csv(os.path.join(ROOT, "reports/ML_WL_1015_features.csv"))
    f_ = f_[~f_.block.isin(["master", "live"])]
    f_.index = list(zip(f_.pair, f_.opened_at.astype(str).str[:19], f_.seed.astype(int)))
    f_ = f_[~f_.index.duplicated()]
    for c in IMPULSE_1015:
        REP[c] = f_[c].reindex(rk).values
    s_ = pd.read_csv(os.path.join(ROOT, "reports/ML_IMPULSE_fill_features.csv"))
    s_ = s_[~s_.block.isin(["master", "live"])]
    s_.index = list(zip(s_.pair, pd.to_datetime(s_.o_ms, unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S"), s_.seed.astype(int)))
    s_ = s_[~s_.index.duplicated()]
    for c in IMP_SCORE:
        REP["imp_" + c] = s_[c].reindex(rk).values
    P(f"  replay loaded: {len(REP)} MOM-long fills · 10-15 impulse matched {REP.btc_r15.notna().mean():.0%} · "
      f"Oct-15 legs matched {REP.imp_SCORE.notna().mean():.0%}")
except Exception as e:  # noqa: BLE001
    P(f"  (replay not loaded: {e})")


def replay_sign(kind, spec, thr):
    if REP is None:
        return ""
    out = []
    for r in ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]:
        d = REP[REP.regime == r]
        try:
            if kind == 1:
                c, fam = spec
                if c not in d or pd.to_numeric(d[c], errors="coerce").notna().mean() < 0.5:
                    out.append(f"{r[:7]} n/a"); continue
                z, v = mask_1d(d, c, fam, thr[0])
            else:
                a, qa, b, qb = spec
                if a not in d or b not in d or pd.to_numeric(d[a], errors="coerce").notna().mean() < 0.5 \
                        or pd.to_numeric(d[b], errors="coerce").notna().mean() < 0.5:
                    out.append(f"{r[:7]} n/a"); continue
                z, v = mask_2d(d, a, qa, b, qb, thr[0], thr[1])
            dl, nz = delta(d, z, v)
            P_ = day_boot_delta(d.y.values[v], z[v], d.day.values[v]) if np.isfinite(dl) else np.nan
            out.append(f"{r[:7]} {dl:+.3f}(n{nz},P>0 {P_:.2f})")
        except Exception:
            out.append(f"{r[:7]} err")
    return " · ".join(out)


# ------------------------------------------------------------------ candidates + bars
def bar_eval(z, label):
    """expectancy bar (filter) and Pattern-W bar (multiplier) on KEEP for zone z."""
    yz = y[z]; dz = day[z]; pz = KEEP.pair.values[z]
    m, lo, hi, pneg = day_boot_mean(yz, dz)
    usd = pd.to_numeric(KEEP.stack_pnl, errors="coerce").values[z].sum()
    r = dict(N=int(z.sum()), days=len(set(dz)), windows=KEEP.win_id.values[z].size and len(set(KEEP.win_id.values[z])),
             WR=(yz > 0).mean(), avg=m, ci_lo=lo, ci_hi=hi, P_avg_neg=pneg, sum_pct=yz.sum(), usd=usd,
             top_day_loss=concentration(yz, dz), top_pair_loss=concentration(yz, pz))
    r["FILTER_bar"] = all([r["WR"] < BE, pneg >= 0.95, r["days"] >= 8, r["N"] >= 15, r["top_day_loss"] < 0.5,
                           r["top_pair_loss"] < 0.5])
    r["filter_fails"] = ",".join(k for k, ok in [("WR≥BE", r["WR"] < BE), ("P<95%", pneg >= 0.95), ("days<8", r["days"] >= 8),
                                                 ("N<15", r["N"] >= 15), ("day≥50%", r["top_day_loss"] < 0.5),
                                                 ("pair≥50%", r["top_pair_loss"] < 0.5)] if not ok)
    r["W_bar"] = all([r["N"] >= 30, r["WR"] >= 0.70, m >= 0.10, usd > 0])
    r["w_fails"] = ",".join(k for k, ok in [("N<30", r["N"] >= 30), ("WR<70", r["WR"] >= 0.70), ("avg<0.10", m >= 0.10),
                                            ("$≤0", usd > 0)] if not ok)
    rest = ~z
    r["sleeve_after_avg"] = y[rest].mean(); r["sleeve_after_WR"] = (y[rest] > 0).mean()
    r["removed_sum_pct"] = yz.sum()
    return r


def era_before_after(z):
    rows = []
    for g in ERA_ORDER:
        m = (KEEP.era_group == g).values
        yb = y[m]; ya = y[m & ~z]; yz = y[m & z]
        rows.append(f"{g.split()[0]}: {len(yb)}·{(yb>0).mean()*100:.0f}%·{yb.mean():+.2f} → {len(ya)}·"
                    f"{(ya>0).mean()*100 if len(ya) else float('nan'):.0f}%·{ya.mean() if len(ya) else float('nan'):+.2f} "
                    f"[zone {len(yz)}·{(yz>0).mean()*100 if len(yz) else float('nan'):.0f}%·{yz.mean() if len(yz) else float('nan'):+.2f}]")
    return " | ".join(rows)


def b25_zone(kind, spec, thr):
    if not len(B25):
        return ""
    if kind == 1:
        z, v = mask_1d(B25, spec[0], spec[1], thr[0])
    else:
        z, v = mask_2d(B25, *spec, thr[0], thr[1])
    if z is None:
        return "n/a"
    yz = B25.y.values[z]
    return f"B25 zone {len(yz)}/{int(v.sum())} scored: " + (", ".join(f"{p}{q:+.2f}" for p, q in zip(B25.pair.values[z], yz)) or "none")


cands = []
# 1D survivors: p_null < 0.05 and |Δ| ≥ 0.10; plus every dashboard variable's best zone (reported regardless)
s1 = R1[(R1.p_null < 0.05) & (R1.delta.abs() >= 0.10)].copy()
for r in s1.itertuples():
    cands.append((1, (r.variable, r.family), r.zone, r.p_null, r.p_fw, r.delta))
s2 = R2[(R2.p_null < 0.01) & (R2.delta.abs() >= 0.20)].sort_values("p_null").head(150)
for r in s2.itertuples():
    cands.append((2, (r.var_a, r.hi_a, r.var_b, r.hi_b), f"{r.var_a}{'>' if r.hi_a else '<='}med ∧ {r.var_b}{'>' if r.hi_b else '<='}med",
                  r.p_null, r.p_fw, r.delta))
CR = []
for kind, spec, lab, pn, pf, dl in cands:
    if kind == 1:
        c, fam = spec; z, v = mask_1d(KEEP, c, fam, THR[c]); thr = (THR[c],)
    else:
        a, qa, b, qb = spec; z, v = mask_2d(KEEP, a, qa, b, qb, THR[a], THR[b]); thr = (THR[a], THR[b])
    be_ = bar_eval(z, lab)
    same, tot, parts, h = oos(kind, spec)
    zt = None
    # TRAD view with the same frozen thresholds
    if kind == 1:
        zt, vt = mask_1d(TRAD, spec[0], spec[1], THR[spec[0]])
    else:
        zt, vt = mask_2d(TRAD, *spec, THR[spec[0]], THR[spec[2]])
    trad_d = delta(TRAD, zt, vt)[0]
    # ex-washed
    nw = ~KEEP.washed.values
    yzw = y[z & nw]
    macro = is_macro(spec[0]) or (kind == 2 and is_macro(spec[2]))
    CR.append(dict(kind=kind, zone=f"{spec[0]} {lab}" if kind == 1 else lab, side="LOSER" if dl < 0 else "WINNER",
                   macro=macro, delta=dl, p_null=pn, p_fw=pf, **be_, oos_same=same, oos_tot=tot, oos_parts=parts,
                   half_train=h[0], half_test_n=h[1], half_test=h[2], trad_delta=trad_d,
                   exwash_N=len(yzw), exwash_WR=(yzw > 0).mean() if len(yzw) else np.nan,
                   exwash_avg=yzw.mean() if len(yzw) else np.nan,
                   replay=replay_sign(kind, spec, thr), b25=b25_zone(kind, spec, thr), eras=era_before_after(z)))
CR = pd.DataFrame(CR)
CR["oos_ok"] = (CR.oos_tot >= 4) & (CR.oos_same >= np.ceil(0.75 * CR.oos_tot)) & (np.sign(CR.half_test) == np.sign(CR.delta)) \
    & (np.sign(CR.trad_delta) == np.sign(CR.delta))
CR.to_csv(OUT + "candidates.csv", index=False)
R1.to_csv(OUT + "screen1d.csv", index=False)
R2.sort_values("p_null").to_csv(OUT + "screen2d.csv", index=False)

# ------------------------------------------------------------------ quintile tables for every dashboard + every screened variable
Q = []
for c in SCREEN:
    x = pd.to_numeric(KEEP[c], errors="coerce")
    v = x.notna()
    if x[v].nunique() < 5:
        for val, g in KEEP[v].groupby(x[v]):
            Q.append(dict(variable=c, bucket=f"=={val:g}", lo=val, hi=val, N=len(g), days=g.day.nunique(), WR=(g.y > 0).mean(),
                          avg=g.y.mean()))
        continue
    qs = pd.qcut(x[v].rank(method="first"), 5, labels=False)
    for k in range(5):
        g = KEEP[v][qs == k]; xx = x[v][qs == k]
        Q.append(dict(variable=c, bucket=f"Q{k+1}", lo=xx.min(), hi=xx.max(), N=len(g), days=g.day.nunique(),
                      WR=(g.y > 0).mean(), avg=g.y.mean()))
pd.DataFrame(Q).to_csv(OUT + "quintiles.csv", index=False)

# ------------------------------------------------------------------ dashboard variables: W vs L medians + sign + best zone
dash_vars = ["entry_rsi", "entry_adx", "d_adx_dir", "d_rsi_dir", "entry_gap", "entry_ema_gap_5_8", "entry_ema_gap_8_13",
             "entry_ema5_stretch", "d_abs_ema20_slope", "entry_ema20_slope", "entry_adx_delta", "entry_range_position",
             "entry_atr_pct", "entry_pair_ema20_ema50_gap_pct", "entry_pair_volume_24h_usd", "entry_dist_from_ema13_pct",
             "entry_btc_1h_slope", "d_abs_btc_ema20_slope", "entry_btc_ema20_slope", "entry_btc_rsi", "entry_btc_adx",
             "d_btc_adx_dir", "d_btc_rsi_dir", "d_btc_rsi_30m_dir", "entry_bull_pct", "entry_btc_trend_gap_pct",
             "entry_global_volume_ratio", "entry_btc_dist_from_ema13_pct", "d_fan_ratio", "entry_btc_atr_pct",
             "entry_price_vs_ema5_pct"]
DV = []
for c in dash_vars:
    x = pd.to_numeric(KEEP[c], errors="coerce"); w = KEEP.y > 0
    r1 = R1[R1.variable == c]
    best = r1.loc[r1.p_null.idxmin()] if len(r1) else None
    DV.append(dict(variable=c, scored=int(x.notna().sum()), med_W=x[w].median(), med_L=x[~w].median(),
                   mean_W=x[w].mean(), mean_L=x[~w].mean(),
                   best_zone=best.zone if best is not None else "", best_N=best.N_in if best is not None else np.nan,
                   best_WR=best.WR_in if best is not None else np.nan, best_avg=best.avg_in if best is not None else np.nan,
                   best_delta=best.delta if best is not None else np.nan, best_p=best.p_null if best is not None else np.nan,
                   best_pfw=best.p_fw if best is not None else np.nan, macro=is_macro(c)))
DV = pd.DataFrame(DV); DV.to_csv(OUT + "dashboard_vars.csv", index=False)

# ------------------------------------------------------------------ post-entry descriptive
PE = []
for c in POST + DURING_1015:
    x = pd.to_numeric(KEEP[c], errors="coerce"); w = KEEP.y > 0
    PE.append(dict(variable=c, scored=int(x.notna().sum()), med_W=x[w].median(), med_L=x[~w].median()))
PE = pd.DataFrame(PE); PE.to_csv(OUT + "post_entry.csv", index=False)

# ------------------------------------------------------------------ replay agreement base rate (is "replay agrees" better than chance?)
import re  # noqa: E402
RX = re.compile(r"([+-]\d+\.\d+)\(n(\d+),P>0 ([\d.]+)\)")


def replay_agree(s, sign):
    m = RX.findall(str(s))
    if len(m) < 3:
        return np.nan, np.nan
    same = sum(1 for d, _, _ in m if np.sign(float(d)) == sign)
    strong = sum(1 for d, _, p in m if np.sign(float(d)) == sign and (float(p) >= 0.8 if sign > 0 else float(p) <= 0.2))
    return same, strong


CR[["replay_same", "replay_strong"]] = [replay_agree(s, np.sign(d)) for s, d in zip(CR.replay, CR.delta)]
CR.to_csv(OUT + "candidates.csv", index=False)
base = []
samp = R2.sample(min(300, len(R2)), random_state=7)
for r in samp.itertuples():
    s = replay_sign(2, (r.var_a, r.hi_a, r.var_b, r.hi_b), (THR[r.var_a], THR[r.var_b]))
    base.append(replay_agree(s, np.sign(r.delta)))
base = pd.DataFrame(base, columns=["same", "strong"]).dropna()
cand2 = CR[(CR.kind == 2)].dropna(subset=["replay_same"])
P(f"\n## S3b replay agreement (refute-only): random 2D tests (n={len(base)}) — all-3-regimes same sign as master "
  f"{(base.same == 3).mean():.0%}, ≥2 strong (P≥0.80) {(base.strong >= 2).mean():.0%} · 2D candidates (n={len(cand2)}) — all-3 "
  f"{(cand2.replay_same == 3).mean():.0%}, ≥2 strong {(cand2.replay_strong >= 2).mean():.0%}")
pd.DataFrame(dict(base_all3=[(base.same == 3).mean()], base_strong2=[(base.strong >= 2).mean()],
                  cand_all3=[(cand2.replay_same == 3).mean()], cand_strong2=[(cand2.replay_strong >= 2).mean()],
                  n_base=[len(base)], n_cand=[len(cand2)])).to_csv(OUT + "replay_baserate.csv", index=False)

# ------------------------------------------------------------------ text dump
P("\n## S4 candidates (1D p<0.05 ∧ |Δ|≥0.10; 2D p<0.01 ∧ |Δ|≥0.20, top 150) — OOS = LOEO same-sign ≥75% of ≥4 groups ∧ "
  "chrono 2nd half same sign ∧ TRAD same sign")
P(f"  1D candidates {int((CR.kind==1).sum())} (OOS ok {int(((CR.kind==1)&CR.oos_ok).sum())}) · 2D candidates "
  f"{int((CR.kind==2).sum())} (OOS ok {int(((CR.kind==2)&CR.oos_ok).sum())})")
P(f"  FILTER bar passes: {int(CR.FILTER_bar.sum())} (of which OOS ok {int((CR.FILTER_bar & CR.oos_ok).sum())}) · "
  f"W bar passes: {int(CR.W_bar.sum())} (of which OOS ok {int((CR.W_bar & CR.oos_ok).sum())})")
for side in ["LOSER", "WINNER"]:
    sub = CR[CR.side == side].sort_values(["oos_ok", "p_null"], ascending=[False, True]).head(25)
    P(f"\n  -- top {side} zones --")
    for r in sub.itertuples():
        P(f"  [{'OOS✓' if r.oos_ok else 'oos✗'}|{'FILTER✓' if r.FILTER_bar else 'filter✗:'+r.filter_fails if side=='LOSER' else ('W✓' if r.W_bar else 'W✗:'+r.w_fails)}] "
          f"{r.zone} {'(MACRO)' if r.macro else ''} N {r.N} · {r.days}d · WR {r.WR*100:.0f}% · avg {r.avg:+.3f} "
          f"[{r.ci_lo:+.2f},{r.ci_hi:+.2f}] · Δ {r.delta:+.3f} p {r.p_null:.3f} pfw {r.p_fw:.2f} · LOEO {r.oos_same}/{r.oos_tot} "
          f"· half2 {r.half_test:+.2f}(n{r.half_test_n}) · TRAD Δ {r.trad_delta:+.2f} · exwash {r.exwash_N}·{r.exwash_avg:+.2f} "
          f"· topday {r.top_day_loss:.0%} toppair {r.top_pair_loss:.0%}")
        P(f"        eras: {r.eras}")
        P(f"        replay: {r.replay} · {r.b25}")
P(f"\n## S5 dashboard variables (KEEP medians W vs L, best 1D zone)")
for r in DV.itertuples():
    P(f"  {r.variable:34s} n {r.scored:3d} med W {r.med_W:+.4g} / L {r.med_L:+.4g} · best {r.best_zone} N {r.best_N} WR "
      f"{r.best_WR*100 if np.isfinite(r.best_WR) else float('nan'):.0f}% avg {r.best_avg:+.3f} Δ {r.best_delta:+.3f} p {r.best_p:.3f} pfw {r.best_pfw:.2f}")
P("\n## S6 post-entry (descriptive only — NOT entry separators)")
for r in PE.itertuples():
    P(f"  {r.variable:12s} n {r.scored:3d} median W {r.med_W:+.3f} / L {r.med_L:+.3f}")
P(f"\n## S7 concentration / uniformity of KEEP losses")
lo_ = KEEP[KEEP.y <= 0]
P(f"  losers {len(lo_)} · top pair {lo_.groupby('pair').y.sum().idxmin()} {lo_.groupby('pair').y.sum().min()/lo_.y.sum():.0%} of loss · "
  f"worst day {lo_.groupby('day').y.sum().min()/lo_.y.sum():.0%} · losing days share {(KEEP.groupby('day').y.sum()<0).mean():.0%} · "
  f"2-min scan windows {KEEP.win_id.nunique()} on {KEEP.day.nunique()} days")
P(f"\nruntime {time.time()-t0:.0f}s")
open(OUT + "engine.txt", "w").write("\n".join(lines))
print("\n".join(lines[:60]))
print("...")
print(f"wrote {OUT}engine.txt / candidates.csv / screen1d.csv / screen2d.csv / quintiles.csv / coverage.csv / dashboard_vars.csv")
