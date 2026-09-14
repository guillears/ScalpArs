#!/usr/bin/env python3
"""
sweep_fades.py — the sweep_separators.py machinery applied to the SPIKE_FADE sleeve (Sep-14, 2026;
operator: "you sure you checked every single variable we have available?"). Same rules: every numeric
entry_* column (+ optional BTC macro features merged from a features CSV) × 3 granularities (sign /
median / outer-tercile, thresholds anchored on era A — no lookahead) × 2 eras, ranked by cross-era
direction-consistency; then every 2D quadrant pair; then a LABEL-SHUFFLE CALIBRATION (pnl_percentage
permuted within era, N_SHUF times) that reports how many 1D/2D survivors chance alone produces.
Survivors are a SCREEN, never a ship. 3D is deliberately NOT run: at N≈54 the 8 cells hold ~7 fills.

  ./venv/bin/python scripts/sweep_fades.py [extra_orders.csv] [--features F.csv] [--split 2026-08-11] [--raw]
"""
import sys, argparse, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')
ap = argparse.ArgumentParser(); ap.add_argument('extra', nargs='*'); ap.add_argument('--features', default=None)
ap.add_argument('--split', default='2026-08-11'); ap.add_argument('--raw', action='store_true', help='ignore stack_keep (raw pool)')
ap.add_argument('--shuffles', type=int, default=200); args = ap.parse_args()
m = pd.read_csv('reports/MASTER_POOL_stacked.csv', low_memory=False)
fr = [m] + [pd.read_csv(p, low_memory=False).assign(stack_keep=True) for p in args.extra]
df = pd.concat(fr, ignore_index=True, sort=False)
df = df[(df.entry_strategy == 'SPIKE_FADE') & (df.status.fillna('CLOSED') == 'CLOSED')].copy()
df['key'] = df.opened_at.astype(str).str[:19] + '|' + df.pair.astype(str); df = df.drop_duplicates('key')
if not args.raw: df = df[df.stack_keep.astype(str).str.lower() != 'false']
if args.features:
    F = pd.read_csv(args.features, low_memory=False); F['key'] = F.opened_at.astype(str).str[:19] + '|' + F.pair.astype(str)
    fcols = ['ret_15m','ret_1h','ret_4h','ret_24h','ret_72h','ret_7d','dist_e50_1h','dist_e20_1h','rsi_1h','off_24h_high','rv_24h','above_e20_6h','slope_e20_1h']
    F = F.drop_duplicates('key')[['key'] + [c for c in fcols if c in F]].rename(columns={c: f'entry_btcX_{c}' for c in fcols})
    df = df.merge(F, on='key', how='left')
for c in df.columns:
    if c.startswith('entry_'): df[c] = pd.to_numeric(df[c], errors='coerce')
df['pnl_percentage'] = pd.to_numeric(df.pnl_percentage, errors='coerce')
df['era'] = np.where(df.opened_at < args.split, 'A', 'B')
nA, nB = (df.era == 'A').sum(), (df.era == 'B').sum()
print(f"FADE SWEEP: N={len(df)} ({'raw' if args.raw else 'screened'}) | era A <{args.split}: {nA} · era B: {nB} | WR {100*(df.pnl_percentage>0).mean():.0f}% avg {df.pnl_percentage.mean():+.3f}")
EXCLUDE = {'entry_price','entry_fee','entry_desired_notional','entry_liquidity_cap_notional','entry_gap_expand_marginal','entry_btc_regime_started_at'}
cands = [c for c in df.columns if c.startswith('entry_') and c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])
         and df[c].notna().sum() >= 0.8*len(df) and df[c].nunique() > 3]
print(f"candidate dimensions: {len(cands)} (stamped entry_* numeric + {sum(c.startswith('entry_btcX_') for c in cands)} BTC macro features)")
y = df.pnl_percentage.values; era = df.era.values
refA = {c: df[df.era=='A'][c].dropna() for c in cands}
# ---- precompute 1D masks (label-independent) ----
tests1 = []
for c in cands:
    x = df[c].values
    for gran, thr in [('sign', 0.0), ('median', refA[c].median()), ('tercile', refA[c].quantile(2/3))]:
        if gran == 'sign' and not ((x > 0).any() and (x <= 0).any()): continue
        mask = x > thr; valid = ~np.isnan(x)
        ok = all(((mask & valid & (era==e)).sum() >= 3) and ((~mask & valid & (era==e)).sum() >= 3) for e in 'AB')
        if ok: tests1.append((c, gran, thr, mask, valid))
def run1(yv):
    out = []
    for c, gran, thr, mask, valid in tests1:
        d = []
        for e in 'AB':
            s = valid & (era==e); d.append(yv[s & mask].mean() - yv[s & ~mask].mean())
        a, b = d; cons = (a*b > 0) and min(abs(a), abs(b)) > 0.05
        out.append((c, gran, thr, a, b, cons, min(abs(a),abs(b)) if cons else 0))
    return pd.DataFrame(out, columns=['col','gran','thr','dA','dB','consistent','strength'])
res = run1(y); cons = res[res.consistent].sort_values('strength', ascending=False)
print(f"\n1D: {res.col.nunique()} dims × {len(res)} tests | cross-era consistent: {len(cons)}")
print(f"{'dimension':<44}{'gran':<9}{'thr':>8}{'ΔavgA':>8}{'ΔavgB':>8}{'str':>6}")
for _, r in cons.head(20).iterrows(): print(f"{r.col:<44}{r.gran:<9}{r.thr:>8.2f}{r.dA:>+8.3f}{r.dB:>+8.3f}{r.strength:>6.2f}")
# ---- 2D quadrants ----
top = list(dict.fromkeys(cons.col.tolist() + cands))[:40]
med = {c: refA[c].median() for c in top}
tests2 = []
for i in range(len(top)):
    for j in range(i+1, len(top)):
        c1, c2 = top[i], top[j]; x1, x2 = df[c1].values, df[c2].values; valid = ~np.isnan(x1) & ~np.isnan(x2)
        for q1 in (True, False):
            for q2 in (True, False):
                mask = ((x1 > med[c1]) == q1) & ((x2 > med[c2]) == q2) & valid
                if all(((mask & (era==e)).sum() >= 5) and ((~mask & valid & (era==e)).sum() >= 5) for e in 'AB'):
                    tests2.append((c1, q1, c2, q2, mask, valid))
def run2(yv):
    out = []
    for c1, q1, c2, q2, mask, valid in tests2:
        d = []
        for e in 'AB':
            s = valid & (era==e); d.append(yv[s & mask].mean() - yv[s & ~mask].mean())
        a, b = d
        if a*b > 0 and min(abs(a), abs(b)) > 0.10: out.append((c1, q1, c2, q2, a, b, min(abs(a),abs(b))))
    return pd.DataFrame(out, columns=['c1','q1','c2','q2','dA','dB','strength'])
r2 = run2(y)
print(f"\n2D: {len(top)} dims → {len(tests2)} quadrant cells tested (min 5/quadrant/era) | consistent (|Δ|>0.10 both eras): {len(r2)}")
lab = lambda c, q: f"{c.replace('entry_','')}{'>' if q else '≤'}med"
for _, r in r2.sort_values('strength', ascending=False).head(15).iterrows(): print(f"  {lab(r.c1,r.q1)+' × '+lab(r.c2,r.q2):<72}{r.dA:>+7.2f}{r.dB:>+7.2f}  str {r.strength:.2f}")
# ---- shuffle calibration ----
rng = np.random.default_rng(3); n1, n2, s1, s2 = [], [], [], []
for _ in range(args.shuffles):
    yp = y.copy()
    for e in 'AB':
        idx = np.where(era==e)[0]; yp[idx] = yp[rng.permutation(idx)]
    a = run1(yp); b = run2(yp); n1.append(int(a.consistent.sum())); n2.append(len(b))
    s1.append(a.strength.max()); s2.append(b.strength.max() if len(b) else 0)
n1, n2, s1, s2 = map(np.array, (n1, n2, s1, s2))
print(f"\nSHUFFLE CALIBRATION ({args.shuffles} label permutations within era):")
print(f"  1D consistent survivors — observed {len(cons)} | chance median {np.median(n1):.0f} (95th pct {np.percentile(n1,95):.0f}) | P(chance ≥ observed) = {(n1>=len(cons)).mean():.2f}")
print(f"  1D top strength        — observed {cons.strength.max() if len(cons) else 0:.3f} | chance 95th pct {np.percentile(s1,95):.3f} | P(chance ≥ obs) = {(s1>=(cons.strength.max() if len(cons) else 0)).mean():.2f}")
print(f"  2D consistent survivors — observed {len(r2)} | chance median {np.median(n2):.0f} (95th pct {np.percentile(n2,95):.0f}) | P(chance ≥ observed) = {(n2>=len(r2)).mean():.2f}")
print(f"  2D top strength        — observed {r2.strength.max() if len(r2) else 0:.3f} | chance 95th pct {np.percentile(s2,95):.3f} | P(chance ≥ obs) = {(s2>=(r2.strength.max() if len(r2) else 0)).mean():.2f}")
print("\n3D deliberately not run: 8 cells over ~54 fills (~7 per cell) cannot separate anything from noise.")
