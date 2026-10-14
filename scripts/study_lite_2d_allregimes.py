#!/usr/bin/env python3
"""LITE 2D separator that holds in ALL THREE regimes (operator 2026-10-14: "is there not a 2D, considering the first half of 2026?").
yr6 two-year replay re-priced to today (ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv), FRENZY_LITE seed 1, y = pct_new. Every pair of numeric
entry_* columns (≥ 95 % scored, ≥ 10 distinct values) split at each column's two-year median → 4 quadrant zones. A zone qualifies when
zone − rest < 0 in 2025 BULL, 2026-H1 BEAR AND 2026-H2 BULL, each regime with ≥ 15 zone fills. Score = the weakest (least negative)
regime gap. Null: permute outcomes within each regime (200×), same search, count qualifying zones and the best score → is the observed
count / best score beyond chance? Then the master real-fill check for the top zones. Output reports/LITE_2D_ALLREGIMES.txt."""
import os, itertools, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(14)
F = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv"), low_memory=False)
F = F[(F.sleeve == "FRENZY_LITE") & (F.seed == 1)].reset_index(drop=True)
y = F.pct_new.astype(float).values
REG = ["2025 BULL", "2026-H1 BEAR", "2026-H2 BULL"]
rid = F.regime.map({r: i for i, r in enumerate(REG)}).values
cols = []
for c in F.columns:
    if not c.startswith("entry_"):
        continue
    v = pd.to_numeric(F[c], errors="coerce")
    if v.notna().mean() >= 0.95 and v.nunique() >= 10:
        cols.append(c)
med = {c: pd.to_numeric(F[c], errors="coerce").median() for c in cols}
HI = {c: (pd.to_numeric(F[c], errors="coerce") > med[c]).values for c in cols}
OK = {c: pd.to_numeric(F[c], errors="coerce").notna().values for c in cols}
zones = []
for a, b in itertools.combinations(cols, 2):
    ok = OK[a] & OK[b]
    for qa in (False, True):
        for qb in (False, True):
            zones.append((a, qa, b, qb, ok & (HI[a] == qa) & (HI[b] == qb), ok))
print(f"{len(cols)} columns · {len(zones)} quadrant zones")


def search(yy):
    out = []
    for a, qa, b, qb, z, ok in zones:
        gaps, ns = [], []
        good = True
        for r in range(3):
            m = rid == r
            zz, rr = z & m, ok & ~z & m
            n = zz.sum()
            if n < 15 or rr.sum() < 15:
                good = False; break
            g = yy[zz].mean() - yy[rr].mean()
            if g >= 0:
                good = False; break
            gaps.append(g); ns.append(n)
        if good:
            out.append((max(gaps), a, qa, b, qb, gaps, ns))
    return out


obs = search(y)
best = min((o[0] for o in obs), default=0.0)
null_n, null_best = [], []
for _ in range(200):
    yy = y.copy()
    for r in range(3):
        m = np.flatnonzero(rid == r); yy[m] = rng.permutation(yy[m])
    s = search(yy); null_n.append(len(s)); null_best.append(min((o[0] for o in s), default=0.0))
null_n, null_best = np.array(null_n), np.array(null_best)
L = [__doc__.splitlines()[0], f"{len(cols)} columns ({', '.join(c.replace('entry_', '') for c in cols)})", "",
     f"qualifying zones (negative in all 3 regimes, ≥ 15 fills each): observed {len(obs)} · null median {np.median(null_n):.0f} "
     f"(p95 {np.percentile(null_n, 95):.0f}) → P(null ≥ observed) {np.mean(null_n >= len(obs)):.3f}",
     f"best weakest-regime gap: observed {best:+.3f} · null median {np.median(null_best):+.3f} (p5 {np.percentile(null_best, 5):+.3f}) → "
     f"P(null ≤ observed) {np.mean(null_best <= best):.3f}", "", "Top 15 zones (weakest-regime gap first):"]
M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
M = M[(M.entry_strategy == "FRENZY_LITE") & (M.stack_keep == True)]   # noqa: E712
for g0, a, qa, b, qb, gaps, ns in sorted(obs, key=lambda o: o[0])[:15]:
    va, vb = pd.to_numeric(M[a], errors="coerce"), pd.to_numeric(M[b], errors="coerce")
    mz = ((va > med[a]) == qa) & ((vb > med[b]) == qb) & va.notna() & vb.notna()
    mp = pd.to_numeric(M.stack_pct, errors="coerce"); mu = pd.to_numeric(M.stack_pnl, errors="coerce")
    L.append(f"  {a.replace('entry_', '')} {'>' if qa else '≤'} {med[a]:.3g} ∧ {b.replace('entry_', '')} {'>' if qb else '≤'} {med[b]:.3g}: "
             f"gaps {' / '.join(f'{x:+.2f}' for x in gaps)} (n {'/'.join(map(str, ns))}) · master zone {int(mz.sum())} fills "
             f"{(mp[mz] > 0).sum()}W {(mp[mz] <= 0).sum()}L ${mu[mz].sum():+,.0f}")
txt = "\n".join(L); open(os.path.join(ROOT, "reports", "LITE_2D_ALLREGIMES.txt"), "w").write(txt + "\n"); print(txt)
