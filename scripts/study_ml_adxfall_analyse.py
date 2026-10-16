#!/usr/bin/env python3
"""ML_ADXFALL_2D (2026-10-16) — analysis per reports/ML_ADXFALL_2D_PREREG.txt. Research only.
Zone = ADX falling at entry (AF_stamp primary, AF_closed secondary). Machinery = study_ml_dt2d_fu_analyse.py (cuts frozen on the
2025 zone, day-clustered z, consistency C, 200 day-block shuffles, walk-forward) with the zone swapped for ADX-falling.
    venv/bin/python -W ignore scripts/study_ml_adxfall_analyse.py [--zone stamp|closed]
Outputs reports/ML_ADXFALL_tables_<zone>.txt + reports/ML_ADXFALL_screen_<zone>.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr6wl_momentum_common import day_boot, day_boot_delta, breakeven_wr   # noqa: E402

ZONE = sys.argv[sys.argv.index("--zone") + 1] if "--zone" in sys.argv else "stamp"
ZC = {"stamp": "AF_stamp", "closed": "AF_closed"}[ZONE]
ERAS = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
NSHUF = 200
RNG = np.random.default_rng(20261016)
NOTIONAL = 700.0
OUT_TXT = os.path.join(ROOT, f"reports/ML_ADXFALL_tables_{ZONE}.txt")
OUT_CSV = os.path.join(ROOT, f"reports/ML_ADXFALL_screen_{ZONE}.csv")
LINES = []


def P(s=""):
    print(s)
    LINES.append(s)


D = pd.read_pickle(os.path.join(ROOT, "reports/ML_ADXFALL_features.pkl"))
D["btc1h"] = pd.to_numeric(D.entry_btc_1h_slope, errors="coerce").fillna(D.k_btc_slope1h)
D["Z"] = D[ZC]
D["month"] = pd.to_datetime(D.ts).dt.to_period("M").astype(str)
D = D.sort_values(["block", "ts"]).reset_index(drop=True)

# ─────────── primaries (frozen in the prereg) ───────────
PRIM = {
    "P1 RSI high": ("rng", "entry_rsi", 60.1367, np.inf),
    "P2 RSI rolling over": ("rng", "d_rsi_dir", -np.inf, 0.0),
    "P3 stretched >EMA5": ("rng", "entry_price_vs_ema5_pct", 0.149, np.inf),
    "P4 stretched >EMA13": ("rng", "entry_dist_from_ema13_pct", 0.372, np.inf),
    "P5 EMA5-20 gap narrowing": ("rng", "c_gap5_20_chg", -np.inf, 0.0),
    "P6 ADX level high": ("rng", "entry_adx", 21.4623, np.inf),
    "P7 DI spread shrinking": ("rng", "c_dis_chg", -np.inf, 0.0),
    "P8 volume fading": ("rng", "c_vol_tr3", -np.inf, 1.0),
    "P9 coin 1h trend down": ("rng", "k_slope1h", -np.inf, 0.0),
    "P10 coin 4h trend down": ("rng", "k_gap4h", -np.inf, 0.0),
    "P11 BTC 15m down": ("rng", "m_btc_r15", -np.inf, 0.0),
    "P12 move old": ("rng", "c_run_e20", 8.0, np.inf),
}

# ─────────── screen cells ───────────
Z25 = D[(D.block == ERAS[0]) & (D.Z == 1)]
PAR = [v for v in pd.read_csv(os.path.join(ROOT, "reports/ML_DT2D_screen.csv"))["var"].unique() if v not in ("d_adx_dir", "entry_adx_delta")]
NEW = [c for c in D.columns if c.startswith("n_")]
CC = [c for c in D.columns if c.startswith("c_") and c not in ("c_adx_delta",)] if ZONE == "closed" else [c for c in D.columns if c.startswith("c_")]
CC = [c for c in CC if c in ("c_adx", "c_adx_delta", "c_adx_delta3", "c_pdi", "c_ndi", "c_dis", "c_dis_chg", "c_rsi", "c_rsi_dir",
                             "c_vol_r20", "c_vol_tr3", "c_run_e20", "c_gap5_20", "c_gap5_20_chg", "c_ret6", "c_ret12", "c_px_e5")]
VARS = list(dict.fromkeys(PAR + NEW + CC + ["s_dis", "s_gap520_chg"]))


def qcells(c):
    x = pd.to_numeric(Z25[c], errors="coerce").dropna()
    if not len(x):
        return []
    u = np.sort(x.unique())
    if len(u) <= 3:
        return [(c, f"{c}=={v:g}", ("eq", c, v, v)) for v in u]
    q1, md, q2 = np.quantile(x, [1 / 3, 0.5, 2 / 3])
    out = [(c, f"{c} {nm} [{lo:.4g},{hi:.4g}]", ("rng", c, lo, hi)) for nm, lo, hi in
           [("≤med", -np.inf, md), (">med", md, np.inf), ("T1", -np.inf, q1), ("T2", q1, q2), ("T3", q2, np.inf),
            ("≤q2", -np.inf, q2), (">q1", q1, np.inf)]]
    sh = (x <= 0).mean()
    if 0.10 <= sh <= 0.90:
        out += [(c, f"{c} ≤0", ("rng", c, -np.inf, 0.0)), (c, f"{c} >0", ("rng", c, 0.0, np.inf))]
    return out


CELLS = [("PRIMARY", k, v) for k, v in PRIM.items()]
NPRIM = len(CELLS)
for c in VARS:
    CELLS += qcells(c)
FAMV = np.array([c[0] for c in CELLS])
ISPRIM = np.arange(len(CELLS)) < NPRIM
P(f"# ML_ADXFALL_2D tables — zone = {ZC} (2026-10-16; prereg reports/ML_ADXFALL_2D_PREREG.txt)\n")
P(f"variables {len(VARS)} · cells: primaries {NPRIM} · screen {len(CELLS) - NPRIM}")


def member(d, spec):
    k, c, lo, hi = spec
    x = pd.to_numeric(d[c], errors="coerce").values.astype(float)
    sc = np.isfinite(x)
    inx = (x == lo) if k == "eq" else ((x > lo) & (x <= hi))
    return inx & sc, sc


def build(d):
    z = (d.Z == 1).values
    zs = d.Z.notna().values
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


def f3(v, p=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:+.{p}f}"


# ─────────── §0 zone baseline ───────────
REAL = {"master as-traded": ("master", "y", None), "master stack_keep": ("master", "y_stack", "stack_keep"),
        "fresh": ("fresh", "y", None), "LIVE batch": ("fresh", "y", "is_live")}


def realblock(lab):
    b, yc, kc = REAL[lab]
    d = D[D.block == b]
    if kc:
        d = d[d[kc].fillna(False).astype(bool)]
    return d.assign(yy=d[yc].astype(float)).dropna(subset=["yy"]).reset_index(drop=True)


def blocks():
    out = [(e, D[D.block == e].assign(yy=lambda x: x.y.astype(float)).reset_index(drop=True)) for e in ERAS]
    for lab in REAL:
        out.append((lab, realblock(lab)))
    return out


def fmt(y, day):
    y = np.asarray(y, float)
    if not len(y):
        return "0"
    m, lo, hi, _ = day_boot(y, day, n=2000)
    return f"{len(y)}·{np.unique(day).size} d·{(y > 0).mean() * 100:.0f} %·{m:+.3f} [{lo:+.2f},{hi:+.2f}]"


BE = {}
P("\n## §0 zone baseline (ADX falling vs rising) and sleeve breakeven WR per block")
P("| block | all fills | ADX falling | ADX rising | Δ [95 %] | sleeve breakeven WR |")
P("|---|---|---|---|---|---|")
for lab, d in blocks():
    z = (d.Z == 1).values
    dl = day_boot_delta(d.yy.values, z, d.day.values, n=2000) if z.any() and (~z).any() else (np.nan,) * 4
    BE[lab] = breakeven_wr(d.yy.values)
    P(f"| {lab} | {fmt(d.yy.values, d.day.values)} | {fmt(d.yy.values[z], d.day.values[z])} | {fmt(d.yy.values[~z], d.day.values[~z])} | "
      f"{f3(dl[0])} [{f3(dl[1], 2)},{f3(dl[2], 2)}] | {BE[lab] * 100:.1f} % |")

# ─────────── screen machinery per era ───────────
RES = {}
YS = {}
for e in ERAS:
    d = D[D.block == e].sort_values("ts", kind="stable").reset_index(drop=True)
    _, dayi = np.unique(d.day.values, return_inverse=True)
    nd = dayi.max() + 1
    IN, OUT, OUTR = build(d)
    Y = shuffled(d)
    YS[e] = (d, Y, IN, OUT, OUTR)
    Di, Zi, nin, nout, din, dout = dstats(IN, OUT, Y, dayi, nd)
    Dr, Zr, _, _, _, _ = dstats(IN, OUTR, Y, dayi, nd)
    nz = (d.Z == 1).sum()
    valid = (nin >= 0.05 * nz) & (nout >= 0.05 * nz) & (din >= 8) & (dout >= 8)
    RES[e] = dict(Di=Di, Zi=Zi, Dr=Dr, Zr=Zr, nin=nin, din=din, valid=valid, d=d, IN=IN)
    print("screened", e, flush=True)
# pooled 2026 (H1 ∪ H2) for walk-forward direction (b): same shuffles concatenated
d26 = pd.concat([YS[ERAS[1]][0], YS[ERAS[2]][0]], ignore_index=True)
Y26 = np.vstack([YS[ERAS[1]][1], YS[ERAS[2]][1]])
IN26 = np.hstack([YS[ERAS[1]][2], YS[ERAS[2]][2]]); OUT26 = np.hstack([YS[ERAS[1]][3], YS[ERAS[2]][3]])
OUTR26 = np.hstack([YS[ERAS[1]][4], YS[ERAS[2]][4]])
_, dayi26 = np.unique(d26.day.values, return_inverse=True)
D26i, Z26i, *_ = dstats(IN26, OUT26, Y26, dayi26, dayi26.max() + 1)
D26r, Z26r, *_ = dstats(IN26, OUTR26, Y26, dayi26, dayi26.max() + 1)
valid = np.logical_and.reduce([RES[e]["valid"] for e in ERAS])


def consistency(Zs):
    S = np.stack(Zs)
    pos = np.nanmin(S, 0); neg = np.nanmax(S, 0)
    return np.where(pos > 0, pos, np.where(neg < 0, neg, 0.0))


C = {"in": consistency([RES[e]["Zi"] for e in ERAS]), "rest": consistency([RES[e]["Zr"] for e in ERAS])}
STOUF = {"in": sum(RES[e]["Zi"] for e in ERAS) / np.sqrt(3), "rest": sum(RES[e]["Zr"] for e in ERAS) / np.sqrt(3)}

# real blocks per cell
REALB = {}
for lab in REAL:
    d = realblock(lab)
    IN, OUT, OUTR = build(d)
    y = d.yy.values
    with np.errstate(invalid="ignore", divide="ignore"):
        REALB[lab] = dict(n=IN.sum(1), no=OUT.sum(1), mi=(IN * y).sum(1) / IN.sum(1), mo=(OUT * y).sum(1) / OUT.sum(1),
                          mr=(OUTR * y).sum(1) / OUTR.sum(1), wr=(IN * (y > 0)).sum(1) / IN.sum(1))

rows = []
for i, cell in enumerate(CELLS):
    r = dict(var=cell[0], cell=cell[1], primary=bool(ISPRIM[i]), valid=bool(valid[i]))
    for e in ERAS:
        R = RES[e]; k = e[:7]
        r[f"{k}_n"] = int(R["nin"][i]); r[f"{k}_d"] = int(R["din"][i])
        r[f"{k}_mean"] = float((R["IN"][i] * R["d"].y.values).sum() / max(R["nin"][i], 1))
        r[f"{k}_Din"] = R["Di"][i, 0]; r[f"{k}_Zin"] = R["Zi"][i, 0]; r[f"{k}_Drest"] = R["Dr"][i, 0]; r[f"{k}_Zrest"] = R["Zr"][i, 0]
    r["C_in"] = C["in"][i, 0]; r["C_rest"] = C["rest"][i, 0]; r["S_in"] = STOUF["in"][i, 0]; r["S_rest"] = STOUF["rest"][i, 0]
    for lab in REAL:
        Rb = REALB[lab]
        r[f"{lab}_n"] = int(Rb["n"][i]); r[f"{lab}_nout"] = int(Rb["no"][i]); r[f"{lab}_mean"] = Rb["mi"][i]; r[f"{lab}_wr"] = Rb["wr"][i]
        r[f"{lab}_Din"] = Rb["mi"][i] - Rb["mo"][i]; r[f"{lab}_Drest"] = Rb["mi"][i] - Rb["mr"][i]
    rows.append(r)
S = pd.DataFrame(rows)

# ─────────── §1 primaries ───────────
P("\n## §1 PRE-REGISTERED PRIMARIES (predicted bad side X; pass = Δin < 0 in all 3 eras ∧ Stouffer z ≤ −2.64 ∧ master not opposite (≥ 10 each side))")
P("| primary | 2025 n·avg·Δin (z) | 2026-H1 | 2026-H2 | Stouffer z in / rest | master n·avg·Δin (n¬X) | stack_keep Δin | fresh n·avg·Δin | LIVE n·avg | verdict |")
P("|---|---|---|---|---|---|---|---|---|---|")
for i in range(NPRIM):
    r = S.iloc[i]
    neg3 = all(r[f"{e[:7]}_Din"] < 0 for e in ERAS)
    mref = (r["master as-traded_Din"] > 0) and r["master as-traded_n"] >= 10 and r["master as-traded_nout"] >= 10
    ok = neg3 and r.S_in <= -2.64 and not mref
    why = "PASS" if ok else "fail: " + ", ".join(x for x, c in (("sign", not neg3), ("z", r.S_in > -2.64), ("master opp.", mref)) if c)
    P(f"| {r.cell} | " + " | ".join(f"{r[f'{e[:7]}_n']}·{r[f'{e[:7]}_mean']:+.2f}·{r[f'{e[:7]}_Din']:+.3f} ({r[f'{e[:7]}_Zin']:+.1f})" for e in ERAS)
      + f" | {r.S_in:+.2f} / {r.S_rest:+.2f} | {r['master as-traded_n']}·{f3(r['master as-traded_mean'], 2)}·{f3(r['master as-traded_Din'], 3)} ({r['master as-traded_nout']}) | "
      f"{f3(r['master stack_keep_Din'], 3)} | {r['fresh_n']}·{f3(r['fresh_mean'], 2)}·{f3(r['fresh_Din'], 2)} | {r['LIVE batch_n']}·{f3(r['LIVE batch_mean'], 2)} | {why} |")

# ─────────── §2 family-wise null ───────────
SCR = ~ISPRIM
NULLMAX = {}
P("\n## §2 exhaustive screen — family-wise reads (C = signed min |z| over 3 eras, same sign; 200 day-block shuffles)")
P("| family | stat | valid cells | cells |C|≥1.96 obs | null median / 95th | p(count) | observed max|C| | null 95th max|C| | fw-p best |")
P("|---|---|---|---|---|---|---|---|---|---|")
for fam, fm in (("primaries", ISPRIM), ("screen", SCR), ("all", np.ones(len(CELLS), bool))):
    for lab in ("in", "rest"):
        m = fm & valid
        obs = C[lab][m, 0]
        nm = np.nanmax(np.abs(C[lab][m, 1:]), 0)
        NULLMAX[(fam, lab)] = nm
        cnt = int((np.abs(obs) >= 1.96).sum()); cn = (np.abs(C[lab][m, 1:]) >= 1.96).sum(0)
        mx = np.nanmax(np.abs(obs))
        P(f"| {fam} | {lab} | {int(m.sum())} | {cnt} | {np.median(cn):.0f} / {np.percentile(cn, 95):.0f} | "
          f"{((cn >= cnt).sum() + 1) / (NSHUF + 1):.3f} | {mx:.2f} | {np.percentile(nm, 95):.2f} | {(nm >= mx).mean():.3f} |")
for lab in ("in", "rest"):
    S[f"fw_{lab}"] = [float((NULLMAX[("all", lab)] >= abs(C[lab][i, 0])).mean()) if valid[i] else np.nan for i in range(len(CELLS))]

# ─────────── §3 walk-forward both directions ───────────
P("\n## §3 walk-forward (discover |z| ≥ 1.96 ∧ |Δ| ≥ 0.05; (a) 2025 → same sign H1 AND H2; (b) 2026 pooled → same sign 2025)")
A, B, Cc = RES[ERAS[0]], RES[ERAS[1]], RES[ERAS[2]]
WF = {}
for lab, kD, kZ, D26, Z26 in (("in", "Di", "Zi", D26i, Z26i), ("rest", "Dr", "Zr", D26r, Z26r)):
    fm = valid & ~ISPRIM

    def conf_a(k):
        dk = fm & (np.abs(A[kZ][:, k]) >= 1.96) & (np.abs(A[kD][:, k]) >= 0.05)
        return dk, dk & (np.sign(B[kD][:, k]) == np.sign(A[kD][:, k])) & (np.sign(Cc[kD][:, k]) == np.sign(A[kD][:, k]))

    def conf_b(k):
        dk = fm & (np.abs(Z26[:, k]) >= 1.96) & (np.abs(D26[:, k]) >= 0.05)
        return dk, dk & (np.sign(A[kD][:, k]) == np.sign(D26[:, k]))

    for nm, fn in (("(a) 2025→2026", conf_a), ("(b) 2026→2025", conf_b)):
        dk, ck = fn(0)
        nulls = np.array([fn(k)[1].sum() for k in range(1, NSHUF + 1)])
        p95 = np.percentile(nulls, 95)
        WF[(lab, nm)] = (ck, ck.sum() > p95)
        ok = S.index[ck]
        sg = np.sign(S.loc[ok, f"2025 BU_D{lab}"])
        pts = ((np.sign(S.loc[ok, f"master as-traded_D{lab}"]) == sg) & (np.sign(S.loc[ok, f"fresh_D{lab}"]) == sg)).sum()
        P(f"- [{lab}] {nm}: discovered {int(dk.sum())} of {int(fm.sum())} → confirmed {int(ck.sum())} · null median {np.median(nulls):.0f} / "
          f"95th {p95:.0f} · p {((nulls >= ck.sum()).sum() + 1) / (NSHUF + 1):.3f} · of which same sign in master AND fresh {int(pts)}")
        for _, r in S.loc[ok].assign(a=S.loc[ok, f"C_{lab}"].abs()).sort_values("a", ascending=False).head(8).iterrows():
            P(f"    · {r.cell}: Δ{lab} {r[f'2025 BU_D{lab}']:+.3f}/{r[f'2026-H1_D{lab}']:+.3f}/{r[f'2026-H2_D{lab}']:+.3f} · C {r[f'C_{lab}']:+.2f} "
              f"fw {r[f'fw_{lab}']:.2f} · master {f3(r[f'master as-traded_D{lab}'])} (n {r['master as-traded_n']}) · stack {f3(r[f'master stack_keep_D{lab}'])} · "
              f"fresh {f3(r[f'fresh_D{lab}'])} (n {r['fresh_n']})")
S.to_csv(OUT_CSV, index=False)

# ─────────── §4 top screen cells ───────────
P("\n## §4 top 12 screen cells by |C_in| (valid), all blocks")
P("| cell | 2025 n·avg·Δin (z) | 2026-H1 | 2026-H2 | C_in · fw | C_rest · fw | master n·avg·Δin | stack_keep Δin | fresh n·avg·Δin | LIVE n·avg |")
P("|---|---|---|---|---|---|---|---|---|---|")
sv = S[(~S.primary) & S.valid]
TOPIN = sv.reindex(sv.C_in.abs().sort_values(ascending=False).index).head(12)
for _, r in TOPIN.iterrows():
    P(f"| {r.cell} | " + " | ".join(f"{r[f'{e[:7]}_n']}·{r[f'{e[:7]}_mean']:+.2f}·{r[f'{e[:7]}_Din']:+.3f} ({r[f'{e[:7]}_Zin']:+.1f})" for e in ERAS)
      + f" | {r.C_in:+.2f} · {r.fw_in:.2f} | {r.C_rest:+.2f} · {r.fw_rest:.2f} | {r['master as-traded_n']}·{f3(r['master as-traded_mean'], 2)}·{f3(r['master as-traded_Din'], 3)} | "
      f"{f3(r['master stack_keep_Din'], 3)} | {r['fresh_n']}·{f3(r['fresh_mean'], 2)}·{f3(r['fresh_Din'], 2)} | {r['LIVE batch_n']}·{f3(r['LIVE batch_mean'], 2)} |")
P("\n### top 6 by |C_rest| (zone∧X vs everything else)")
for _, r in sv.reindex(sv.C_rest.abs().sort_values(ascending=False).index).head(6).iterrows():
    P(f"- {r.cell}: Δrest {r['2025 BU_Drest']:+.3f}/{r['2026-H1_Drest']:+.3f}/{r['2026-H2_Drest']:+.3f} · C {r.C_rest:+.2f} · fw {r.fw_rest:.2f} · "
      f"master {f3(r['master as-traded_Drest'])} · stack {f3(r['master stack_keep_Drest'])} · fresh {f3(r['fresh_Drest'])}")
P("\n### all-block direction agreement (3 eras + master + stack_keep + fresh, inside-zone Δ, any z)")
sg = np.sign(sv["2025 BU_Din"])
allsame = sv[(np.sign(sv["2026-H1_Din"]) == sg) & (np.sign(sv["2026-H2_Din"]) == sg) & (np.sign(sv["master as-traded_Din"]) == sg)
             & (np.sign(sv["master stack_keep_Din"]) == sg) & (np.sign(sv["fresh_Din"]) == sg)]
P(f"- {len(allsame)} of {len(sv)} valid cells (luck for 6 independent signs ≈ 2/64 → ≈ {len(sv) * 2 / 64:.0f}); with |C_in| ≥ 1.0: "
  f"{int((allsame.C_in.abs() >= 1.0).sum())}; with fw_in ≤ 0.05: {int((allsame.fw_in <= 0.05).sum())}")
for _, r in allsame.reindex(allsame.C_in.abs().sort_values(ascending=False).index).head(8).iterrows():
    P(f"    · {r.cell}: C_in {r.C_in:+.2f} (fw {r.fw_in:.2f}) · Δ {r['2025 BU_Din']:+.3f}/{r['2026-H1_Din']:+.3f}/{r['2026-H2_Din']:+.3f} · "
      f"master {r['master as-traded_Din']:+.3f} (n {r['master as-traded_n']}) · stack {f3(r['master stack_keep_Din'])} · fresh {r['fresh_Din']:+.3f} (n {r['fresh_n']})")

# ─────────── §5 deep reads ───────────
SPEC = {c[1]: c[2] for c in CELLS}


def conc(y, key):
    tot = y.sum()
    if tot >= 0 or not len(y):
        return np.nan
    s = pd.Series(y).groupby(np.asarray(key)).sum()
    return float((-s.min()) / (-tot))


def deep(name, sign):
    spec = SPEC[name]
    rule = "refuse ADXfall ∧ X" if sign < 0 else "refuse ADXfall ∧ ¬X"
    P(f"\n### {name} — bad side = {'X' if sign < 0 else '¬X'} → before/after rule: {rule}")
    P("| block | ADXfall∧X | ADXfall∧¬X | Δ in-zone [95 %] | ADX rising | refused cohort: WR vs BE · P(avg<0) · days · top pair / day share | "
      "selectivity kept−refused [P>0] | before N·avg·Σ% | refused W/L·Σ% | after N·avg·Σ% | Σ$ recorded before→after | Σ$ @ $700 1× before→after |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, d in blocks():
        a, sc = member(d, spec)
        z = (d.Z == 1).values
        x, nx = z & a, z & sc & ~a
        y, day = d.yy.values, d.day.values
        di = day_boot_delta(y[x | nx], x[x | nx], day[x | nx], n=2000) if x.any() and nx.any() else (np.nan,) * 4
        ref = x if sign < 0 else nx
        if ref.any() and (~ref).any():
            sel = day_boot_delta(y, ~ref, day, n=2000)
            m, lo, hi, pp = day_boot(y[ref], day[ref], n=2000)
            exp = (f"{(y[ref] > 0).mean() * 100:.0f} % vs {BE[lab] * 100:.0f} % · {1 - pp:.2f} · {np.unique(day[ref]).size} · "
                   f"{f3(conc(y[ref], d.pair.values[ref]), 2)} / {f3(conc(y[ref], day[ref]), 2)}")
        else:
            sel = (np.nan,) * 4; exp = "—"
        usd = (pd.to_numeric(d.usd, errors="coerce") if lab in ("fresh", "LIVE batch") else
               pd.to_numeric(d.pnl, errors="coerce") if lab == "master as-traded" else pd.Series(np.nan, index=d.index)).values
        aft = y[~ref]
        P(f"| {lab} | {fmt(y[x], day[x])} | {fmt(y[nx], day[nx])} | {f3(di[0])} [{f3(di[1], 2)},{f3(di[2], 2)}] | {fmt(y[~z], day[~z])} | {exp} | "
          f"{f3(sel[0])} [{sel[3]:.2f}] | {len(y)}·{y.mean():+.3f}·{y.sum():+.1f} | {int((y[ref] > 0).sum())}/{int((y[ref] <= 0).sum())}·{y[ref].sum():+.2f} | "
          f"{len(aft)}·{(aft.mean() if len(aft) else np.nan):+.3f}·{aft.sum():+.1f} | {np.nansum(usd):+.0f} → {np.nansum(usd[~ref]):+.0f} | "
          f"{y.sum() * NOTIONAL / 100:+.0f} → {aft.sum() * NOTIONAL / 100:+.0f} |")
    # LOMO on the pooled backtest
    bt = D[D.block.isin(ERAS)].reset_index(drop=True)
    a, sc = member(bt, spec)
    z = (bt.Z == 1).values
    x, nx = z & a, z & sc & ~a
    if not nx.any():                                   # zone-alone reference: compare with ADX rising
        nx = ~z
    ds = []
    for mo in sorted(bt.month.unique()):
        k = (bt.month != mo).values
        ds.append(bt.y.values[x & k].mean() - bt.y.values[nx & k].mean())
    ds = np.array(ds)
    P(f"\nLOMO (pooled backtest, {len(ds)} months left out one at a time): Δin range {ds.min():+.3f} … {ds.max():+.3f}; "
      f"sign flips vs full {int((np.sign(ds) != np.sign(bt.y.values[x].mean() - bt.y.values[nx].mean())).sum())}")
    m = realblock("master as-traded")
    a, sc = member(m, spec)
    z = (m.Z == 1).values
    ref = (z & a) if sign < 0 else (z & sc & ~a)
    m = m.assign(ref=ref)
    P("master per batch (as-traded Σ% before → after · $ recorded before → after · refused W/L): " + " · ".join(
        f"{e} {g.yy.sum():+.2f}→{g.yy[~g.ref].sum():+.2f} (${g.pnl.sum():+.0f}→${g.pnl[~g.ref].sum():+.0f}, {int((g.yy[g.ref] > 0).sum())}/{int((g.yy[g.ref] <= 0).sum())})"
        for e, g in m.groupby("era", sort=False) if g.ref.any()) + " · other batches untouched")
    P("\nper trade — refused fills (master + fresh; LIVE marked):")
    P("| block | batch | opened | pair | ADX Δ stamp | X value | P&L % | $ recorded |")
    P("|---|---|---|---|---|---|---|---|")
    xc = spec[1]
    for lab in ("master as-traded", "fresh"):
        d = realblock(lab)
        a, sc = member(d, spec)
        z = (d.Z == 1).values
        ref = (z & a) if sign < 0 else (z & sc & ~a)
        for r in d[ref].sort_values("ts").itertuples():
            bt_ = r.era if lab.startswith("master") else ("LIVE" if r.is_live else "paper")
            usd = r.pnl if lab.startswith("master") else r.usd
            P(f"| {lab.split()[0]} | {bt_} | {r.ts:%m-%d %H:%M} | {r.pair} | {r.d_adx_dir:+.2f} | {float(getattr(r, xc)):+.3f} | {r.yy:+.3f} | {float(usd):+.1f} |")


P("\n## §5 deep reads (three blocks, two-sided) — the zone alone, the best primary, and the top screen cells (labelled, even if failing)")
# the zone alone (refuse all ADX-falling) as reference
CELLS.append(("ref", "ZONE ALONE (X = everything)", ("rng", "entry_rsi", -np.inf, np.inf)))
SPEC["ZONE ALONE (X = everything)"] = CELLS[-1][2]
deep("ZONE ALONE (X = everything)", -1)
pr = S[S.primary].copy()
best = pr.loc[pr.S_in.idxmin()]
deep(best.cell, -1)
surv = S[(~S.primary) & S.valid & ((S.fw_in <= 0.05) | (S.fw_rest <= 0.05))]
picks = list(surv.cell)
seen = set()
for _, r in TOPIN.iterrows():                          # top 2 distinct variables by |C_in| (mirror cells skipped)
    if r["var"] not in seen and len(seen) < 2:
        seen.add(r["var"]); picks.append(r.cell)
TOPR = sv.reindex(sv.C_rest.abs().sort_values(ascending=False).index)
for _, r in TOPR.iterrows():                           # + top 2 distinct variables by |C_rest|
    if r["var"] not in seen and len(seen) < 4:
        seen.add(r["var"]); picks.append(r.cell)
for nm in picks[:6]:
    r = S[S.cell == nm].iloc[0]
    deep(nm, np.sign(r.C_in) if r.C_in != 0 else (np.sign(r.C_rest) if r.C_rest != 0 else -1))
P(f"\nscreen survivors at fw ≤ 0.05: {len(surv)}")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(LINES) + "\n")
print("wrote", OUT_TXT, OUT_CSV)
