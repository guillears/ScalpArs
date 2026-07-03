#!/usr/bin/env python3
"""
sweep_separators.py — EXHAUSTIVE cross-period separator sweep (Jul 3, 2026).

Born from an operator reprimand: a sleeve verdict was once given after hand-testing
~8 dimensions when the orders schema stores 60+. This script removes analyst selection
from the loop: for a given sleeve it tests EVERY numeric entry_* column at THREE
granularities (sign split / median split / outer-tercile split) in BOTH eras
(baseline vs fresh), and ranks candidates by cross-period direction-consistency.

RULES OF USE (locked in CLAUDE.md):
- Any sleeve-level verdict ("no separator", "disable", "demote") MUST cite this sweep.
- A dimension is "refuted" only if NO granularity is direction-consistent.
- Consistency here is a SCREEN, not a ship: survivors still face the locked promotion
  gates (N, WR, avg, theory, haircut) before any config change.

Usage: ./venv/bin/python scripts/sweep_separators.py [FS|ML|MS] [era_split_date]
"""
import sys, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')

POOL = "reports/SCREENED_BASELINE.csv"
SLEEVE = sys.argv[1] if len(sys.argv) > 1 else 'FS'
SPLIT = sys.argv[2] if len(sys.argv) > 2 else '2026-06-30'

df = pd.read_csv(POOL)
df = df[df.screen_sleeve == {'FS':'FLIP_SHORT','ML':'MOM_LONG','MS':'MOM_SHORT'}[SLEEVE]].copy()
df['era'] = np.where(df.opened_at < SPLIT, 'A', 'B')
if df.era.nunique() < 2:
    sys.exit(f"only one era in pool for {SLEEVE} at split {SPLIT}")

EXCLUDE = {'id','leverage','investment','notional_value','quantity','entry_price','exit_price',
           'current_price','pnl','pnl_percentage','peak_pnl','trough_pnl','cell_multiplier',
           'cell_lev_multiplier','entry_fee','exit_fee','total_fee','pnl_current_sizing'}
cands = [c for c in df.columns
         if (c.startswith('entry_')) and c not in EXCLUDE
         and pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 0.8*len(df)
         and df[c].nunique() > 3]

def split_stats(d, mask, col):
    hi, lo = d[mask], d[~mask]
    if len(hi) < 3 or len(lo) < 3: return None
    return dict(n_hi=len(hi), n_lo=len(lo),
                d_avg=hi.pnl_percentage.mean() - lo.pnl_percentage.mean())

results = []
for col in cands:
    for gran, maskf in [
        ('sign',    lambda s: s > 0),
        ('median',  lambda s: s > s.median()),
        ('tercile', lambda s: s > s.quantile(2/3)),
    ]:
        if gran == 'sign' and not ((df[col] > 0).any() and (df[col] <= 0).any()):
            continue
        rows = {}
        ok = True
        for era in ['A','B']:
            d = df[df.era == era].dropna(subset=[col])
            # median/tercile computed on era A only (no lookahead into fresh era)
            ref = df[(df.era == 'A')][col].dropna()
            if gran == 'sign':   mask = d[col] > 0
            elif gran == 'median': mask = d[col] > ref.median()
            else:                mask = d[col] > ref.quantile(2/3)
            st = split_stats(d, mask, col)
            if st is None: ok = False; break
            rows[era] = st
        if not ok: continue
        a, b = rows['A']['d_avg'], rows['B']['d_avg']
        consistent = (a * b > 0) and min(abs(a), abs(b)) > 0.05
        results.append(dict(col=col, gran=gran, dA=a, dB=b, consistent=consistent,
                            strength=min(abs(a), abs(b)) if consistent else 0,
                            nA=rows['A']['n_hi']+rows['A']['n_lo'], nB=rows['B']['n_hi']+rows['B']['n_lo']))

res = pd.DataFrame(results)
n_dims = res.col.nunique(); n_tests = len(res)
cons = res[res.consistent].sort_values('strength', ascending=False)
print(f"SWEEP {SLEEVE}: {n_dims} dimensions x {n_tests} tests | era A (<{SPLIT}) vs B | consistent: {len(cons)}")
print(f"{'dimension':<42}{'gran':<9}{'ΔavgA':>8}{'ΔavgB':>8}{'strength':>9}")
for _, r in cons.head(15).iterrows():
    print(f"{r.col:<42}{r.gran:<9}{r.dA:>+8.3f}{r.dB:>+8.3f}{r.strength:>9.3f}")
if not len(cons):
    print("(no cross-period-consistent separator at any granularity — a 'none exists' verdict is now defensible)")

# ── Stage 2 (Jul 3, operator-directed): 2D interaction sweep — every dimension PAIR,
# 4 quadrants (median x median, thresholds anchored on era A), quadrant-vs-rest delta,
# cross-era consistency, min 5 trades per quadrant per era. WARNING baked in: with
# ~thousands of tests, expect false survivors — this RANKS candidates for the locked
# gates (N/WR/avg/theory/haircut); it never ships anything by itself.
print("\n── 2D INTERACTION SWEEP ──")
top_dims = list(dict.fromkeys(res[res.consistent].col.tolist() + cands))[:36]
refA = {c: df[df.era=='A'][c].dropna().median() for c in top_dims}
res2 = []
for i in range(len(top_dims)):
    for j in range(i+1, len(top_dims)):
        c1, c2 = top_dims[i], top_dims[j]
        d = df.dropna(subset=[c1, c2])
        for q1 in (True, False):
            for q2 in (True, False):
                ok = True; deltas = []
                for era in ['A','B']:
                    de = d[d.era==era]
                    m = ((de[c1] > refA[c1]) == q1) & ((de[c2] > refA[c2]) == q2)
                    if m.sum() < 5 or (~m).sum() < 5: ok = False; break
                    deltas.append(de[m].pnl_percentage.mean() - de[~m].pnl_percentage.mean())
                if not ok: continue
                a, b = deltas
                if a*b > 0 and min(abs(a),abs(b)) > 0.10:
                    res2.append(dict(pair=f"{c1.replace('entry_','')}{'>' if q1 else '<='}med x {c2.replace('entry_','')}{'>' if q2 else '<='}med",
                                     dA=a, dB=b, strength=min(abs(a),abs(b))))
r2 = pd.DataFrame(res2)
n_pairs_tested = len(top_dims)*(len(top_dims)-1)//2*4
print(f"pairs tested: ~{n_pairs_tested} quadrant cells | consistent (|Δ|>0.10 both eras, minN 5/quadrant/era): {len(r2)}")
if len(r2):
    for _, r in r2.sort_values('strength', ascending=False).head(12).iterrows():
        print(f"  {r['pair']:<70}{r.dA:>+7.2f}{r.dB:>+7.2f}  str {r.strength:.2f}")
