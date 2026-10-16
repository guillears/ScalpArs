#!/usr/bin/env python3
"""ML_DOWNTREND_2D (2026-10-16) — analysis. Pre-registration reports/ML_DOWNTREND_2D_PREREG.txt (sha in the deliverable).
Input reports/ML_DT2D_features.pkl (study_ml_dt2d_build.py). Outputs reports/ML_DT2D_tables.txt + reports/ML_DT2D_screen.csv.
    venv/bin/python -W ignore scripts/study_ml_dt2d_analyse.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr6wl_momentum_common import NON_SIGNAL, day_boot, breakeven_wr, is_macro   # noqa: E402

ERAS = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
NSHUF = 200
RNG = np.random.default_rng(20261016)
OUT_TXT = os.path.join(ROOT, "reports/ML_DT2D_tables.txt")
OUT_CSV = os.path.join(ROOT, "reports/ML_DT2D_screen.csv")
LINES = []


def P(s=""):
    print(s)
    LINES.append(s)


D = pd.read_pickle(os.path.join(ROOT, "reports/ML_DT2D_features.pkl"))
D["DTb"] = D.DT.astype(float)
D["btc1h"] = pd.to_numeric(D.entry_btc_1h_slope, errors="coerce").fillna(D.k_btc_slope1h)
D["month"] = D.ts.dt.strftime("%Y-%m")
D = D.sort_values(["block", "ts"]).reset_index(drop=True)


def blk(name):
    return D[D.block == name].copy()


def stats(y, day):
    y = np.asarray(y, float)
    if len(y) == 0:
        return "0"
    m, lo, hi, pp = day_boot(y, day, n=2000)
    return f"{len(y)} · {np.unique(day).size} d · {np.mean(y > 0) * 100:.0f} % · {m:+.3f} [{lo:+.2f},{hi:+.2f}]"


def num(d, c):
    return pd.to_numeric(d[c], errors="coerce").values.astype(float)


# ───────────────────────────── §0 zone per block ─────────────────────────────
P("# ML_DOWNTREND_2D tables (2026-10-16)\n")
P("## §0 the zone itself (DT = coin 1h EMA20 ≤ EMA200; stamp, rebuild where missing) — N · days · WR · avg % [day-boot 95 %]")
REAL = {"master as-traded": ("master", "y", None), "master stack_keep": ("master", "y_stack", "stack_keep"),
        "fresh (line cohort)": ("fresh", "y", None)}
BE = {}
for e in ERAS:
    d = blk(e)
    BE[e] = breakeven_wr(d.y)
    P(f"- {e}: DT {stats(d.y[d.DT == 1], d.day[d.DT == 1])} · rest {stats(d.y[d.DT == 0], d.day[d.DT == 0])} · "
      f"BE {BE[e] * 100:.1f} %")
for lab, (b, ycol, kc) in REAL.items():
    d = blk(b)
    if kc:
        d = d[d[kc].astype(bool)]
    BE[lab] = breakeven_wr(d[ycol])
    P(f"- {lab}: DT {stats(d[ycol][d.DT == 1], d.day[d.DT == 1])} · rest {stats(d[ycol][d.DT == 0], d.day[d.DT == 0])} · "
      f"BE {BE[lab] * 100:.1f} % · unscored {int(d.DT.isna().sum())}")
m = blk("master")
P("- master per batch (as-traded, DT / rest): " + " · ".join(
    f"{e} {int((g.DT == 1).sum())}:{g.y[g.DT == 1].mean():+.2f}/{int((g.DT == 0).sum())}:{g.y[g.DT == 0].mean():+.2f}"
    for e, g in m.groupby("era", sort=False)))

# ───────────────────────────── variables ─────────────────────────────
EXCL = set(NON_SIGNAL) | {"entry_pair_1h_ema20_200_gap_pct", "entry_quality_score", "entry_fee", "entry_slippage_pct",
                          "entry_desired_notional", "entry_liquidity_cap_notional", "entry_price"}
cand = [c for c in D.columns if (c.startswith(("entry_", "d_", "k_", "m_")) and c not in EXCL
                                  and c not in ("k_g1h",))]
cand += ["g1h_z", "btc1h"]
VARS, BLIND = [], []
for c in cand:
    x = pd.to_numeric(D[c], errors="coerce")
    covs = {e: x[D.block == e].notna().mean() for e in ERAS}
    nun = x[D.block == ERAS[0]].nunique()
    if min(covs.values()) >= 0.9 and nun >= 2:
        VARS.append(c)
    elif max(covs.values()) > 0:
        BLIND.append((c, {k: round(v, 2) for k, v in covs.items()}))
P(f"\nscreen variables: {len(VARS)}; blind spots (replay coverage < 90 % in some era, or constant): {len(BLIND)}")
for c, cv in BLIND:
    P(f"  - BLIND {c}: {cv}")

# label-blind cuts on the 2025 replay ZONE fills
Z25 = D[(D.block == ERAS[0]) & (D.DT == 1)]
CELLS = []      # (name, var, kind, lo, hi) membership: lo < x ≤ hi ; categorical: x == v
for c in VARS:
    x = pd.to_numeric(Z25[c], errors="coerce").dropna()
    u = np.sort(x.unique())
    if len(u) <= 3:
        for v in u:
            CELLS.append((f"{c}=={v:g}", c, "eq", v, v))
        continue
    q1, md, q2 = np.quantile(x, [1 / 3, 0.5, 2 / 3])
    for nm, lo, hi in [("≤med", -np.inf, md), (">med", md, np.inf), ("T1", -np.inf, q1), ("T2", q1, q2), ("T3", q2, np.inf),
                       ("≤q2", -np.inf, q2), (">q1", q1, np.inf)]:
        CELLS.append((f"{c} {nm} [{lo:.4g},{hi:.4g}]", c, "rng", lo, hi))
P(f"cells: {len(CELLS)}")


def member(d, cell):
    _, c, kind, lo, hi = cell
    x = pd.to_numeric(d[c], errors="coerce").values.astype(float)
    sc = np.isfinite(x)
    if kind == "eq":
        inx = x == lo
    else:
        inx = (x > lo) & (x <= hi)
    return inx & sc, sc


def build(d):
    """→ IN, OUT (inside-zone split), INR, OUTR (zone∧X vs rest) boolean matrices over d rows."""
    z = (d.DT == 1).values
    zs = d.DT.notna().values
    n = len(d)
    IN = np.zeros((len(CELLS), n), bool); OUT = np.zeros_like(IN); OUTR = np.zeros_like(IN)
    for i, cell in enumerate(CELLS):
        a, sc = member(d, cell)
        IN[i] = z & a
        OUT[i] = z & sc & ~a
        OUTR[i] = zs & ~(z & a) & (sc | ~z)       # rest = every scored fill not in (DT∧X); X-unscored zone fills excluded
    return IN, OUT, OUTR


def dstats(IN, OUT, Y, dayi, nd):
    """Δ = mean(IN) − mean(OUT) and day-clustered z (cluster-robust variance of the difference) for every cell × every outcome
    column of Y (n × K). Accumulated day by day (no scipy)."""
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
    """day-block shuffles of y in time order (seeds of a day stay together): n × (K+1), column 0 = observed."""
    y = d.y.values.astype(float)
    day = d.day.values
    u, first = np.unique(day, return_index=True)
    order = np.argsort(first)
    blocks = [np.where(day == u[i])[0] for i in order]
    Y = np.empty((len(y), NSHUF + 1)); Y[:, 0] = y
    for k in range(1, NSHUF + 1):
        perm = RNG.permutation(len(blocks))
        Y[:, k] = np.concatenate([y[blocks[j]] for j in perm])
    return Y


# order each era by time so block shuffles lay outcomes over the time sequence
RES = {}
for e in ERAS:
    d = blk(e).sort_values("ts", kind="stable").reset_index(drop=True)
    _, dayi = np.unique(d.day.values, return_inverse=True)
    nd = dayi.max() + 1
    IN, OUT, OUTR = build(d)
    Y = shuffled(d)
    Di, Zi, nin, nout, din, dout = dstats(IN, OUT, Y, dayi, nd)
    Dr, Zr, _, noutr, _, doutr = dstats(IN, OUTR, Y, dayi, nd)
    nz = (d.DT == 1).sum()
    valid = (nin >= 0.05 * nz) & (nout >= 0.05 * nz) & (din >= 8) & (dout >= 8)
    RES[e] = dict(Di=Di, Zi=Zi, Dr=Dr, Zr=Zr, nin=nin, din=din, valid=valid, nz=nz, d=d, IN=IN)
    print("screened", e, flush=True)

valid = np.logical_and.reduce([RES[e]["valid"] for e in ERAS])


def consistency(Zs):
    """Zs: list of (cells × K) z arrays → signed min-|z| when all same sign else 0."""
    S = np.stack(Zs)                                 # eras × cells × K
    pos = np.nanmin(S, 0); neg = np.nanmax(S, 0)
    return np.where(pos > 0, pos, np.where(neg < 0, neg, 0.0))


C_in = consistency([RES[e]["Zi"] for e in ERAS])
C_rest = consistency([RES[e]["Zr"] for e in ERAS])
out = {}
for lab, C in (("in", C_in), ("rest", C_rest)):
    obs = C[:, 0]
    nullmax = np.nanmax(np.where(valid[:, None], np.abs(C[:, 1:]), np.nan), 0)
    fw = np.array([(nullmax >= abs(o)).mean() if valid[i] else np.nan for i, o in enumerate(obs)])
    cnt_obs = int((np.abs(obs[valid]) >= 1.96).sum())
    cnt_null = (np.abs(C[valid, 1:]) >= 1.96).sum(0)
    out[lab] = dict(obs=obs, fw=fw, nullmax=nullmax, cnt_obs=cnt_obs, cnt_null=cnt_null)

# per-cell real-block reads
REALB = {}
for lab, (b, ycol, kc) in REAL.items():
    d = blk(b)
    if kc:
        d = d[d[kc].astype(bool)]
    d = d.assign(y=d[ycol].astype(float)).reset_index(drop=True)
    IN, OUT, OUTR = build(d)
    y = d.y.values
    with np.errstate(invalid="ignore", divide="ignore"):
        REALB[lab] = dict(nin=IN.sum(1), din_=np.array([d.day[IN[i]].nunique() for i in range(len(CELLS))]),
                          mi=(IN * y).sum(1) / IN.sum(1), mo=(OUT * y).sum(1) / OUT.sum(1), mr=(OUTR * y).sum(1) / OUTR.sum(1),
                          wr=(IN * (y > 0)).sum(1) / IN.sum(1))

rows = []
for i, cell in enumerate(CELLS):
    r = dict(cell=cell[0], var=cell[1], macro=is_macro(cell[1]) or cell[1].startswith(("m_btc", "m_br", "m_disp", "m_agg", "k_btc")),
             valid=bool(valid[i]))
    for e in ERAS:
        R = RES[e]
        r[f"{e[:7]}_n"] = int(R["nin"][i]); r[f"{e[:7]}_d"] = int(R["din"][i])
        r[f"{e[:7]}_mean"] = float((R["IN"][i] * R["d"].y.values).sum() / max(R["nin"][i], 1))
        r[f"{e[:7]}_Din"] = R["Di"][i, 0]; r[f"{e[:7]}_Zin"] = R["Zi"][i, 0]
        r[f"{e[:7]}_Drest"] = R["Dr"][i, 0]; r[f"{e[:7]}_Zrest"] = R["Zr"][i, 0]
    r["C_in"] = out["in"]["obs"][i]; r["fw_in"] = out["in"]["fw"][i]
    r["C_rest"] = out["rest"]["obs"][i]; r["fw_rest"] = out["rest"]["fw"][i]
    for lab in REAL:
        Rb = REALB[lab]
        r[f"{lab}_n"] = int(Rb["nin"][i]); r[f"{lab}_d"] = int(Rb["din_"][i]); r[f"{lab}_mean"] = Rb["mi"][i]
        r[f"{lab}_wr"] = Rb["wr"][i]; r[f"{lab}_Din"] = Rb["mi"][i] - Rb["mo"][i]; r[f"{lab}_Drest"] = Rb["mi"][i] - Rb["mr"][i]
    rows.append(r)
S = pd.DataFrame(rows)
S.to_csv(OUT_CSV, index=False)

P("\n## §2 exhaustive screen DT × X (inside-zone split and zone∧X vs rest), 3 replay eras, day-block-shuffled null ×200")
for lab in ("in", "rest"):
    o = out[lab]
    P(f"- [{lab}] valid cells {int(valid.sum())} · cells with |C| ≥ 1.96 (same sign + |z| ≥ 1.96 in ALL three eras): "
      f"{o['cnt_obs']} vs null median {np.median(o['cnt_null']):.0f} / 95th {np.percentile(o['cnt_null'], 95):.0f} "
      f"(p {((o['cnt_null'] >= o['cnt_obs']).sum() + 1) / (NSHUF + 1):.3f}) · null 95th of max|C| {np.percentile(o['nullmax'], 95):.2f} · "
      f"observed max|C| {np.nanmax(np.abs(o['obs'][valid])):.2f} · family-wise p<.05 cells {int((o['fw'][valid] < 0.05).sum())}")

# walk-forward: discover 2025 → confirm H1, H2 → master / fresh pointing
P("\n## §3 walk-forward (discover 2025 inside-zone |z| ≥ 1.96 ∧ |Δ| ≥ 0.05 → same sign in 2026-H1 AND 2026-H2)")
sv = S[S.valid]
disc = sv[(sv["2025 BU_Zin"].abs() >= 1.96) & (sv["2025 BU_Din"].abs() >= 0.05)]
conf = disc[(np.sign(disc["2026-H1_Din"]) == np.sign(disc["2025 BU_Din"])) & (np.sign(disc["2026-H2_Din"]) == np.sign(disc["2025 BU_Din"]))]
P(f"- discovered {len(disc)} of {len(sv)} valid cells (luck ≈ {0.05 * len(sv):.0f}); confirmed same sign in both 2026 eras {len(conf)}")
ptr = conf[(np.sign(conf["master as-traded_Din"]) == np.sign(conf["2025 BU_Din"])) &
           (np.sign(conf["fresh (line cohort)_Din"]) == np.sign(conf["2025 BU_Din"]))]
P(f"- … and same sign in master as-traded AND the fresh cohort: {len(ptr)}")
# null for the walk-forward count: same pipeline on shuffled outcome columns
wf_null = []
for k in range(1, NSHUF + 1):
    a = RES["2025 BULL"]; b = RES["2026-H1 BEAR"]; c = RES["2026-H2 BULL"]
    dk = valid & (np.abs(a["Zi"][:, k]) >= 1.96) & (np.abs(a["Di"][:, k]) >= 0.05)
    ck = dk & (np.sign(b["Di"][:, k]) == np.sign(a["Di"][:, k])) & (np.sign(c["Di"][:, k]) == np.sign(a["Di"][:, k]))
    wf_null.append(int(ck.sum()))
wf_null = np.array(wf_null)
P(f"- null (day-block shuffles) count of confirmed cells: median {np.median(wf_null):.0f} · 95th {np.percentile(wf_null, 95):.0f} "
  f"· p(obs {len(conf)}) {((wf_null >= len(conf)).sum() + 1) / (NSHUF + 1):.3f}")


def cellrow(r):
    f = lambda v, p=3: "—" if not np.isfinite(v) else f"{v:+.{p}f}"
    return (f"| {r.cell} | " + " | ".join(f"{int(r[f'{e[:7]}_n'])}·{f(r[f'{e[:7]}_mean'], 2)}·Δ{f(r[f'{e[:7]}_Din'], 2)} (z{r[f'{e[:7]}_Zin']:+.1f})"
                                        for e in ERAS)
            + f" | {r['C_in']:+.2f} / fw {r['fw_in']:.2f} | "
            + " | ".join(f"{int(r[f'{l}_n'])}·{f(r[f'{l}_mean'], 2)}·Δ{f(r[f'{l}_Din'], 2)}" for l in REAL) + " |")


HDR = ("| cell (inside DT) | 2025 n·avg·Δin | 2026-H1 | 2026-H2 | C_in / fw-p | master as-traded n·avg·Δin | master stack_keep | fresh |\n"
       "|---|---|---|---|---|---|---|---|")
P("\n### confirmed walk-forward cells (sorted by |C_in|)")
P(HDR)
for _, r in ptr.reindex(ptr.C_in.abs().sort_values(ascending=False).index).head(25).iterrows():
    P(cellrow(r))
P("\n### top 15 valid cells by |C_in| (whatever master / fresh say)")
P(HDR)
for _, r in sv.reindex(sv.C_in.abs().sort_values(ascending=False).index).head(15).iterrows():
    P(cellrow(r))
P("\n### top 10 valid cells by |C_rest| (zone∧X vs everything else)")
P("| cell | 2025 n·avg·Δrest (z) | 2026-H1 | 2026-H2 | C_rest / fw | master Δrest | stack_keep Δrest | fresh Δrest |\n|---|---|---|---|---|---|---|---|")
for _, r in sv.reindex(sv.C_rest.abs().sort_values(ascending=False).index).head(10).iterrows():
    P(f"| {r.cell} | " + " | ".join(f"{int(r[f'{e[:7]}_n'])}·{r[f'{e[:7]}_mean']:+.2f}·Δ{r[f'{e[:7]}_Drest']:+.2f} (z{r[f'{e[:7]}_Zrest']:+.1f})" for e in ERAS)
      + f" | {r.C_rest:+.2f} / {r.fw_rest:.2f} | " + " | ".join(f"{int(r[f'{l}_n'])}·Δ{r[f'{l}_Drest']:+.2f}" for l in REAL) + " |")

with open(OUT_TXT, "w") as f:
    f.write("\n".join(LINES) + "\n")
print("wrote", OUT_TXT, OUT_CSV)
