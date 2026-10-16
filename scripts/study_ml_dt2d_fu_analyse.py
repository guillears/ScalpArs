#!/usr/bin/env python3
"""ML_DOWNTREND_2D follow-up (2026-10-16) — Test A (2D, new coin RSI/ADX 1h·4h + price-vs-EMA grid) and Test B (limited 3-way
DT × new X × 7 theory variables), per reports/ML_DOWNTREND_2D_PREREG_ADDENDUM.txt. Same machinery as study_ml_dt2d_analyse.py
(cuts frozen on the 2025 zone, day-clustered z, consistency C, 200 day-block shuffles, walk-forward). Family-wise nulls on the
SAME shuffles: new-2D alone, 3-way alone, parent+new-2D union, and the full union (parent 697 + new-2D + 3-way).
Input reports/ML_DT2D_FU_features.pkl. Outputs reports/ML_DT2D_FU_tables.txt + reports/ML_DT2D_FU_screen.csv.
    venv/bin/python -W ignore scripts/study_ml_dt2d_fu_analyse.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr6wl_momentum_common import day_boot, day_boot_delta, breakeven_wr   # noqa: E402

ERAS = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
NSHUF = 200
RNG = np.random.default_rng(20261016)
NOTIONAL = 700.0
OUT_TXT = os.path.join(ROOT, "reports/ML_DT2D_FU_tables.txt")
OUT_CSV = os.path.join(ROOT, "reports/ML_DT2D_FU_screen.csv")
LINES = []


def P(s=""):
    print(s)
    LINES.append(s)


D = pd.read_pickle(os.path.join(ROOT, "reports/ML_DT2D_FU_features.pkl"))
D["btc1h"] = pd.to_numeric(D.entry_btc_1h_slope, errors="coerce").fillna(D.k_btc_slope1h)
D = D.sort_values(["block", "ts"]).reset_index(drop=True)
num = lambda c: pd.to_numeric(D[c], errors="coerce").values.astype(float)

# ─────────── cells ───────────
Z25 = D[(D.block == ERAS[0]) & (D.DT == 1)]
PAR_VARS = list(pd.read_csv(os.path.join(ROOT, "reports/ML_DT2D_screen.csv"))["var"].unique())
NEW_LVL = ["n_rsi1h", "n_adx1h", "n_rsi4h", "n_adx4h"]
NEW_DIR = ["n_rsi1h_dir", "n_adx1h_dir", "n_rsi4h_dir", "n_adx4h_dir"]
NEW_PX = [c for c in D.columns if c.startswith("n_px_")]
OLD_PX_FLAGS = ["entry_price_vs_ema5_pct", "entry_dist_from_ema13_pct", "k_px_e20", "k_px_e200"]
NEW_VARS = NEW_LVL + NEW_DIR + NEW_PX


def qcells(c, fam):
    x = pd.to_numeric(Z25[c], errors="coerce").dropna()
    u = np.sort(x.unique())
    out = []
    if len(u) <= 3:
        return [(fam, f"{c}=={v:g}", ("eq", c, v, v)) for v in u]
    q1, md, q2 = np.quantile(x, [1 / 3, 0.5, 2 / 3])
    for nm, lo, hi in [("≤med", -np.inf, md), (">med", md, np.inf), ("T1", -np.inf, q1), ("T2", q1, q2), ("T3", q2, np.inf),
                       ("≤q2", -np.inf, q2), (">q1", q1, np.inf)]:
        out.append((fam, f"{c} {nm} [{lo:.4g},{hi:.4g}]", ("rng", c, lo, hi)))
    return out


CELLS = []
for c in PAR_VARS:
    CELLS += qcells(c, "parent")
NPAR = len(CELLS)
for c in NEW_VARS:
    CELLS += qcells(c, "new2D")
for c in NEW_DIR + NEW_PX + OLD_PX_FLAGS:
    CELLS += [("new2D", f"{c} ≤0 (below/falling)", ("rng", c, -np.inf, 0.0)), ("new2D", f"{c} >0 (above/rising)", ("rng", c, 0.0, np.inf))]

# 3-way: X halves × theory sides
VMED = float(pd.to_numeric(Z25.entry_pair_volume_ratio, errors="coerce").median())
THEORY = {"T1deep": ("g1h_z", -np.inf, -3.0), "T2widen": ("k_g1h_chg4h", -np.inf, 0.0), "T3belowE200": ("k_px_e200", -np.inf, 0.0),
          "T4btcDown": ("btc1h", -np.inf, 0.0), "T5coin4hDown": ("k_gap4h", -np.inf, 0.0), "T6adxFall": ("d_adx_dir", -np.inf, 0.0),
          "T7lowVol": ("entry_pair_volume_ratio", -np.inf, VMED)}
for c in NEW_VARS:
    if c in NEW_LVL:
        md = float(pd.to_numeric(Z25[c], errors="coerce").median())
        halves = [(f"{c}≤{md:.4g}", ("rng", c, -np.inf, md)), (f"{c}>{md:.4g}", ("rng", c, md, np.inf))]
    else:
        halves = [(f"{c}≤0", ("rng", c, -np.inf, 0.0)), (f"{c}>0", ("rng", c, 0.0, np.inf))]
    for xn, xs in halves:
        for tn, (tc, lo, hi) in THEORY.items():
            CELLS.append(("3way", f"{xn} ∧ {tn}=BAD", ("and", xs, ("rng", tc, lo, hi))))
            CELLS.append(("3way", f"{xn} ∧ {tn}=good", ("and", xs, ("not", ("rng", tc, lo, hi)))))
FAM = np.array([c[0] for c in CELLS])
P(f"# ML_DOWNTREND_2D follow-up tables (2026-10-16) — addendum sha ba21014b…\n")
P(f"cells: parent {NPAR} · new2D {(FAM == 'new2D').sum()} · 3way {(FAM == '3way').sum()} · total {len(CELLS)}; T7 cut {VMED:.4f}")


def member(d, spec):
    k = spec[0]
    if k == "and":
        a1, s1 = member(d, spec[1]); a2, s2 = member(d, spec[2])
        return a1 & a2 & s1 & s2, s1 & s2
    if k == "not":
        a, s = member(d, spec[1])
        return ~a & s, s
    _, c, lo, hi = spec
    x = pd.to_numeric(d[c], errors="coerce").values.astype(float)
    sc = np.isfinite(x)
    inx = (x == lo) if k == "eq" else ((x > lo) & (x <= hi))
    return inx & sc, sc


def build(d):
    z = (d.DT == 1).values
    zs = d.DT.notna().values
    IN = np.zeros((len(CELLS), len(d)), bool); OUT = np.zeros_like(IN); OUTR = np.zeros_like(IN)
    for i, cell in enumerate(CELLS):
        a, sc = member(d, cell[2])
        IN[i] = z & a
        OUT[i] = z & sc & ~a
        OUTR[i] = zs & ~(z & a) & (sc | ~z)
    return IN, OUT, OUTR


def dstats(IN, OUT, Y, dayi, nd):
    INf, OUTf = IN.astype(float), OUT.astype(float)
    nin, nout = INf.sum(1), OUTf.sum(1)
    C, K = len(IN), Y.shape[1]
    SI = np.zeros((C, K)); SO = np.zeros((C, K)); Pp = np.zeros((C, K)); PA = np.zeros((C, K)); PB = np.zeros((C, K))
    QA = np.zeros(C); QB = np.zeros(C); QAB = np.zeros(C); din = np.zeros(C); dout = np.zeros(C)
    with np.errstate(invalid="ignore", divide="ignore"):
        for dd in range(nd):
            ix = np.where(dayi == dd)[0]
            a = INf[:, ix]; b = OUTf[:, ix]; yk = Y[ix]
            si = a @ yk; so = b @ yk
            ci = a.sum(1); co = b.sum(1)
            SI += si; SO += so
            p = si / nin[:, None] - so / nout[:, None]
            ra, rb = ci / nin, co / nout
            Pp += p ** 2; PA += p * ra[:, None]; PB += p * rb[:, None]
            QA += ra ** 2; QB += rb ** 2; QAB += ra * rb
            din += ci > 0; dout += co > 0
        mi, mo = SI / nin[:, None], SO / nout[:, None]
        var = Pp - 2 * (mi * PA - mo * PB) + (mi ** 2) * QA[:, None] - 2 * mi * mo * QAB[:, None] + (mo ** 2) * QB[:, None]
        se = np.sqrt(np.clip(var, 0, None) * nd / max(nd - 1, 1))
        Dl = mi - mo
        Z = Dl / se
    return Dl, Z, nin, nout, din, dout


def shuffled(d):
    y = d.y.values.astype(float)
    day = d.day.values
    u, first = np.unique(day, return_index=True)
    blocks = [np.where(day == u[i])[0] for i in np.argsort(first)]
    Y = np.empty((len(y), NSHUF + 1)); Y[:, 0] = y
    for k in range(1, NSHUF + 1):
        perm = RNG.permutation(len(blocks))
        Y[:, k] = np.concatenate([y[blocks[j]] for j in perm])
    return Y


RES = {}
for e in ERAS:
    d = D[D.block == e].sort_values("ts", kind="stable").reset_index(drop=True)
    _, dayi = np.unique(d.day.values, return_inverse=True)
    nd = dayi.max() + 1
    IN, OUT, OUTR = build(d)
    Y = shuffled(d)
    Di, Zi, nin, nout, din, dout = dstats(IN, OUT, Y, dayi, nd)
    Dr, Zr, _, _, _, _ = dstats(IN, OUTR, Y, dayi, nd)
    nz = (d.DT == 1).sum()
    valid = (nin >= 0.05 * nz) & (nout >= 0.05 * nz) & (din >= 8) & (dout >= 8)
    RES[e] = dict(Di=Di, Zi=Zi, Dr=Dr, Zr=Zr, nin=nin, din=din, valid=valid, d=d, IN=IN)
    print("screened", e, flush=True)
valid = np.logical_and.reduce([RES[e]["valid"] for e in ERAS])


def consistency(Zs):
    S = np.stack(Zs)
    pos = np.nanmin(S, 0); neg = np.nanmax(S, 0)
    return np.where(pos > 0, pos, np.where(neg < 0, neg, 0.0))


C = {"in": consistency([RES[e]["Zi"] for e in ERAS]), "rest": consistency([RES[e]["Zr"] for e in ERAS])}
FAMS = {"new2D": FAM == "new2D", "3way": FAM == "3way", "parent": FAM == "parent",
        "union2D (parent+new2D)": FAM != "3way", "UNION all": np.ones(len(CELLS), bool)}
NULLMAX = {}
P("\n## §1 family-wise reads (C = signed min |z| over the 3 eras, same sign; 200 day-block shuffles, same shuffles for every family)")
P("| family | stat | valid cells | cells |C|≥1.96 obs | null median / 95th | p(count) | observed max|C| | null 95th max|C| | fw-p of the best cell |")
P("|---|---|---|---|---|---|---|---|---|")
for fam, fm in FAMS.items():
    for lab in ("in", "rest"):
        m = fm & valid
        obs = C[lab][m, 0]
        nm = np.nanmax(np.abs(C[lab][m, 1:]), 0)
        NULLMAX[(fam, lab)] = nm
        cnt = int((np.abs(obs) >= 1.96).sum()); cn = (np.abs(C[lab][m, 1:]) >= 1.96).sum(0)
        mx = np.nanmax(np.abs(obs))
        P(f"| {fam} | {lab} | {int(m.sum())} | {cnt} | {np.median(cn):.0f} / {np.percentile(cn, 95):.0f} | "
          f"{((cn >= cnt).sum() + 1) / (NSHUF + 1):.3f} | {mx:.2f} | {np.percentile(nm, 95):.2f} | {(nm >= mx).mean():.3f} |")


def fwp(i, lab, fam):
    return float((NULLMAX[(fam, lab)] >= abs(C[lab][i, 0])).mean()) if valid[i] else np.nan


# real blocks
REAL = {"master as-traded": ("master", "y", None), "master stack_keep": ("master", "y_stack", "stack_keep"),
        "fresh": ("fresh", "y", None), "LIVE batch": ("fresh", "y", "is_live")}


def realblock(lab):
    b, yc, kc = REAL[lab]
    d = D[D.block == b]
    if kc:
        d = d[d[kc].fillna(False).astype(bool)]
    return d.assign(yy=d[yc].astype(float)).dropna(subset=["yy"]).reset_index(drop=True)


REALB = {}
for lab in REAL:
    d = realblock(lab)
    IN, OUT, OUTR = build(d)
    y = d.yy.values
    with np.errstate(invalid="ignore", divide="ignore"):
        REALB[lab] = dict(n=IN.sum(1), mi=(IN * y).sum(1) / IN.sum(1), mo=(OUT * y).sum(1) / OUT.sum(1),
                          mr=(OUTR * y).sum(1) / OUTR.sum(1), wr=(IN * (y > 0)).sum(1) / IN.sum(1))

rows = []
for i, cell in enumerate(CELLS):
    r = dict(family=cell[0], cell=cell[1], valid=bool(valid[i]))
    for e in ERAS:
        R = RES[e]; k = e[:7]
        r[f"{k}_n"] = int(R["nin"][i]); r[f"{k}_d"] = int(R["din"][i])
        r[f"{k}_mean"] = float((R["IN"][i] * R["d"].y.values).sum() / max(R["nin"][i], 1))
        r[f"{k}_Din"] = R["Di"][i, 0]; r[f"{k}_Zin"] = R["Zi"][i, 0]; r[f"{k}_Drest"] = R["Dr"][i, 0]; r[f"{k}_Zrest"] = R["Zr"][i, 0]
    r["C_in"] = C["in"][i, 0]; r["C_rest"] = C["rest"][i, 0]
    own = cell[0] if cell[0] != "parent" else "parent"
    for lab in ("in", "rest"):
        r[f"fw_{lab}_own"] = fwp(i, lab, own); r[f"fw_{lab}_union"] = fwp(i, lab, "UNION all")
    for lab in REAL:
        Rb = REALB[lab]
        r[f"{lab}_n"] = int(Rb["n"][i]); r[f"{lab}_mean"] = Rb["mi"][i]; r[f"{lab}_wr"] = Rb["wr"][i]
        r[f"{lab}_Din"] = Rb["mi"][i] - Rb["mo"][i]; r[f"{lab}_Drest"] = Rb["mi"][i] - Rb["mr"][i]
    rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(OUT_CSV, index=False)

# parity with the parent screen (same cuts, same machinery → identical observed z on the parent cells)
PS = pd.read_csv(os.path.join(ROOT, "reports/ML_DT2D_screen.csv"))
mm = S[S.family == "parent"].merge(PS[["cell", "2025 BU_Zin", "C_in"]], on="cell", suffixes=("", "_par"))
P(f"\nparity vs parent screen: {len(mm)} of {len(PS)} parent cells matched by name · max |Δz 2025| "
  f"{(mm['2025 BU_Zin'] - mm['2025 BU_Zin_par']).abs().max():.2e} · max |ΔC_in| {(mm.C_in - mm.C_in_par).abs().max():.2e}")

# ─────────── walk-forward ───────────
P("\n## §2 walk-forward (discover 2025 |z| ≥ 1.96 ∧ |Δ| ≥ 0.05 → same sign 2026-H1 AND 2026-H2), count vs shuffled null")
A, B, Cc = RES[ERAS[0]], RES[ERAS[1]], RES[ERAS[2]]
for fam in ("new2D", "3way"):
    for lab, kD, kZ in (("in", "Di", "Zi"), ("rest", "Dr", "Zr")):
        fm = (FAM == fam) & valid
        def conf(k):
            dk = fm & (np.abs(A[kZ][:, k]) >= 1.96) & (np.abs(A[kD][:, k]) >= 0.05)
            return dk, dk & (np.sign(B[kD][:, k]) == np.sign(A[kD][:, k])) & (np.sign(Cc[kD][:, k]) == np.sign(A[kD][:, k]))
        dk, ck = conf(0)
        nulls = np.array([conf(k)[1].sum() for k in range(1, NSHUF + 1)])
        ok = S.index[ck]
        sg = np.sign(S.loc[ok, f"2025 BU_D{lab}"])
        pts = ((np.sign(S.loc[ok, f"master as-traded_D{lab}"]) == sg) & (np.sign(S.loc[ok, f"fresh_D{lab}"]) == sg)).sum()
        P(f"- {fam} [{lab}]: discovered {int(dk.sum())} of {int(fm.sum())} (luck ≈ {0.05 * fm.sum():.0f}) → confirmed {int(ck.sum())} · "
          f"null median {np.median(nulls):.0f} / 95th {np.percentile(nulls, 95):.0f} · p {((nulls >= ck.sum()).sum() + 1) / (NSHUF + 1):.3f} · "
          f"of which same sign in master AND fresh {int(pts)}")
        if ck.sum():
            for _, r in S.loc[ok].assign(a=S.loc[ok, f"C_{lab}"].abs()).sort_values("a", ascending=False).head(12).iterrows():
                P(f"    · {r.cell}: Δ{lab} {r[f'2025 BU_D{lab}']:+.3f}/{r[f'2026-H1_D{lab}']:+.3f}/{r[f'2026-H2_D{lab}']:+.3f} · "
                  f"C {r[f'C_{lab}']:+.2f} fw(own) {r[f'fw_{lab}_own']:.2f} fw(union) {r[f'fw_{lab}_union']:.2f} · master {r[f'master as-traded_D{lab}']:+.3f} "
                  f"(n {r['master as-traded_n']}) · stack {r[f'master stack_keep_D{lab}']:+.3f} · fresh {r[f'fresh_D{lab}']:+.3f} (n {r['fresh_n']}) · "
                  f"LIVE n {r['LIVE batch_n']} {r['LIVE batch_mean']:+.3f}")


def f3(v, p=3):
    return "—" if not np.isfinite(v) else f"{v:+.{p}f}"


P("\n## §3 top cells by |C| per family (valid), all blocks")
P("| family | cell | 2025 n·avg·Δin (z) | 2026-H1 | 2026-H2 | C_in · fw own / union | C_rest · fw union | master n·avg·Δin | stack_keep Δin | fresh n·avg·Δin | LIVE n·avg |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
TOP = {}
for fam in ("new2D", "3way"):
    sv = S[(S.family == fam) & S.valid]
    t = sv.reindex(sv.C_in.abs().sort_values(ascending=False).index).head(10)
    TOP[fam] = t
    for _, r in t.iterrows():
        P(f"| {fam} | {r.cell} | " + " | ".join(f"{r[f'{e[:7]}_n']}·{r[f'{e[:7]}_mean']:+.2f}·{r[f'{e[:7]}_Din']:+.2f} ({r[f'{e[:7]}_Zin']:+.1f})" for e in ERAS)
          + f" | {r.C_in:+.2f} · {r.fw_in_own:.2f} / {r.fw_in_union:.2f} | {r.C_rest:+.2f} · {r.fw_rest_union:.2f} | "
          f"{r['master as-traded_n']}·{f3(r['master as-traded_mean'], 2)}·{f3(r['master as-traded_Din'], 2)} | {f3(r['master stack_keep_Din'], 2)} | "
          f"{r['fresh_n']}·{f3(r['fresh_mean'], 2)}·{f3(r['fresh_Din'], 2)} | {r['LIVE batch_n']}·{f3(r['LIVE batch_mean'], 2)} |")
P("\n### top 5 by |C_rest| (zone∧X vs everything else) per family")
for fam in ("new2D", "3way"):
    sv = S[(S.family == fam) & S.valid]
    for _, r in sv.reindex(sv.C_rest.abs().sort_values(ascending=False).index).head(5).iterrows():
        P(f"- {fam} · {r.cell}: Δrest {r['2025 BU_Drest']:+.3f}/{r['2026-H1_Drest']:+.3f}/{r['2026-H2_Drest']:+.3f} · C {r.C_rest:+.2f} · "
          f"fw own {r.fw_rest_own:.2f} / union {r.fw_rest_union:.2f} · master {f3(r['master as-traded_Drest'])} · fresh {f3(r['fresh_Drest'])}")

# all-block direction-consistency count (the operator's question, literally): same sign in 3 eras + master + stack + fresh
P("\n## §4 cells pointing the same way in ALL of: 3 eras (any z) + master as-traded + stack_keep + fresh (inside-zone Δ), vs null-free luck")
for fam in ("new2D", "3way"):
    sv = S[(S.family == fam) & S.valid].copy()
    sg = np.sign(sv["2025 BU_Din"])
    allsame = sv[(np.sign(sv["2026-H1_Din"]) == sg) & (np.sign(sv["2026-H2_Din"]) == sg) & (np.sign(sv["master as-traded_Din"]) == sg)
                 & (np.sign(sv["master stack_keep_Din"]) == sg) & (np.sign(sv["fresh_Din"]) == sg)]
    P(f"- {fam}: {len(allsame)} of {len(sv)} valid cells (pure-luck share for 6 independent signs ≈ 2/64 → ≈ {len(sv) * 2 / 64:.0f}); "
      f"of those with |C_in| ≥ 1.0: {int((allsame.C_in.abs() >= 1.0).sum())}")
    for _, r in allsame.reindex(allsame.C_in.abs().sort_values(ascending=False).index).head(6).iterrows():
        P(f"    · {r.cell}: C_in {r.C_in:+.2f} (fw union {r.fw_in_union:.2f}) · Δ {r['2025 BU_Din']:+.2f}/{r['2026-H1_Din']:+.2f}/{r['2026-H2_Din']:+.2f} · "
          f"master {r['master as-traded_Din']:+.2f} (n {r['master as-traded_n']}) · fresh {r['fresh_Din']:+.2f} (n {r['fresh_n']})")

# ─────────── three blocks + before/after for the top cells ───────────
SPEC = {c[1]: c[2] for c in CELLS}


def fmt(y, day):
    y = np.asarray(y, float)
    if not len(y):
        return "0"
    m, lo, hi, _ = day_boot(y, day, n=2000)
    return f"{len(y)}·{np.unique(day).size} d·{(y > 0).mean() * 100:.0f} %·{m:+.3f} [{lo:+.2f},{hi:+.2f}]"


def blocks():
    out = [(e, D[D.block == e].assign(yy=lambda x: x.y.astype(float))) for e in ERAS]
    for lab in REAL:
        out.append((lab, realblock(lab)))
    return out


def deep(name, sign):
    spec = SPEC[name]
    rule = "refuse DT ∧ X" if sign < 0 else "refuse DT ∧ ¬X (keep only the cell inside the zone)"
    P(f"\n### {name}  — Δin sign {'−' if sign < 0 else '+'} → before/after rule: {rule}")
    P("| block | DT∧X | DT∧¬X | Δ in-zone [95 %] | REST (not DT) | before N·avg·Σ% | refused (W/L)·Σ% | after N·avg·Σ% | Δ avg/fill | Σ$ recorded before→after | Σ$ @ $700 1× before→after |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, d in blocks():
        a, sc = member(d, spec)
        z = (d.DT == 1).values
        x, nx = z & a, z & sc & ~a
        di = day_boot_delta(d.yy.values[x | nx], x[x | nx], d.day.values[x | nx], n=2000) if x.any() and nx.any() else (np.nan,) * 4
        ref = x if sign < 0 else nx
        y = d.yy.values
        usd = (pd.to_numeric(d.usd, errors="coerce") if lab in ("fresh", "LIVE batch") else
               pd.to_numeric(d.pnl, errors="coerce") if lab.startswith("master") else pd.Series(np.nan, index=d.index)).values
        if lab == "master stack_keep":
            usd = np.full(len(d), np.nan)
        aft = y[~ref]
        P(f"| {lab} | {fmt(y[x], d.day.values[x])} | {fmt(y[nx], d.day.values[nx])} | {f3(di[0])} [{f3(di[1], 2)},{f3(di[2], 2)}] | "
          f"{fmt(y[~z], d.day.values[~z])} | {len(y)}·{y.mean():+.3f}·{y.sum():+.1f} | {int(ref.sum())} ({int((y[ref] > 0).sum())}/{int((y[ref] <= 0).sum())})·{y[ref].sum():+.2f} | "
          f"{len(aft)}·{(aft.mean() if len(aft) else np.nan):+.3f}·{aft.sum():+.1f} | {(aft.mean() if len(aft) else np.nan) - y.mean():+.3f} | "
          f"{np.nansum(usd):+.0f} → {np.nansum(usd[~ref]):+.0f} | {y.sum() * NOTIONAL / 100:+.0f} → {aft.sum() * NOTIONAL / 100:+.0f} |")
    m = realblock("master as-traded")
    a, sc = member(m, spec)
    z = (m.DT == 1).values
    ref = (z & a) if sign < 0 else (z & sc & ~a)
    m = m.assign(ref=ref)
    P("\nmaster per batch (as-traded Σ% before → after · $ recorded before → after · refused W/L): " + " · ".join(
        f"{e} {g.yy.sum():+.2f}→{g.yy[~g.ref].sum():+.2f} (${g.pnl.sum():+.0f}→${g.pnl[~g.ref].sum():+.0f}, {int((g.yy[g.ref] > 0).sum())}/{int((g.yy[g.ref] <= 0).sum())})"
        for e, g in m.groupby("era", sort=False) if g.ref.any()) + " · other batches untouched")
    P("\nper trade — refused fills (master + fresh; fresh rows marked LIVE where live):")
    P("| block | batch | opened | pair | g1h % | X value | P&L % | $ recorded |")
    P("|---|---|---|---|---|---|---|---|")
    xc = spec[1][1] if spec[0] == "and" else spec[1]
    for lab in ("master as-traded", "fresh"):
        d = realblock(lab)
        a, sc = member(d, spec)
        z = (d.DT == 1).values
        ref = (z & a) if sign < 0 else (z & sc & ~a)
        for r in d[ref].sort_values("ts").itertuples():
            bt = r.era if lab.startswith("master") else ("LIVE" if r.is_live else "paper")
            usd = r.pnl if lab.startswith("master") else r.usd
            P(f"| {lab.split()[0]} | {bt} | {r.ts:%m-%d %H:%M} | {r.pair} | {r.g1h_z:+.2f} | {float(getattr(r, xc)):+.3f} | {r.yy:+.3f} | {float(usd):+.1f} |")


P("\n## §5 three blocks side by side + before/after (two-sided: refused winners count against) for the top cells")
picked = []
for fam in ("new2D", "3way"):
    for _, r in TOP[fam].head(2).iterrows():
        picked.append((r.cell, np.sign(r.C_in)))
for _, r in S[(S.family == "new2D") & S.valid].iterrows():
    pass
for name, sg in picked:
    deep(name, sg)

with open(OUT_TXT, "w") as f:
    f.write("\n".join(LINES) + "\n")
print("wrote", OUT_TXT, OUT_CSV)
