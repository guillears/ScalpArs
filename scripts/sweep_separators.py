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
       ./venv/bin/python scripts/sweep_separators.py <ENTRY_STRATEGY> [era_split_date]   e.g. SURGE_LONG · SPIKE_FADE · FRENZY_LITE ·
                                                       FRENZY_WILLY · FLATBASE_SHORT · BULLRUN_LONG · FLIP:FAN_RATIO_GATE · MOMENTUM:LONG
       ./venv/bin/python scripts/sweep_separators.py ALL                                    every bot sleeve with ≥ 6 real fills

🔁 2026-10-27 (DECISION_LOG 398, operator: "all SLEEVES should have it! why this one dont?"): every sleeve has a mode. FS / ML / MS keep
the SCREENED_BASELINE pool (unchanged). Any other name = the sleeve's REAL fills (master over backtest): reports/MASTER_POOL_stacked.csv
rows of that entry_strategy kept by TODAY's stack (stack_keep, P&L = stack_pct — re-priced on today's rules) + the current batch's
closed bot fills from every orders export (opened ≥ the batch start, as traded — the batch IS today's stack), dedup (opened_at, pair,
direction), MANUAL out. Era split default = the cohort's median open date (both eras populated); a column below 80 % coverage is NOT
tested and is listed as a blind spot (never silently dropped). N < 30 → printed as a screen on a thin cohort.
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "reports", "SCREENED_BASELINE.csv")
MASTER = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
BATCH_START = "2026-10-23T18:30:00"     # current batch B25 (paper) — fills after it are not in the master yet
LEGACY = {'FS': 'FLIP_SHORT', 'ML': 'MOM_LONG', 'MS': 'MOM_SHORT'}
MIN_FILLS = 6
EXCLUDE = {'id', 'leverage', 'investment', 'notional_value', 'quantity', 'entry_price', 'exit_price',
           'current_price', 'pnl', 'pnl_percentage', 'peak_pnl', 'trough_pnl', 'cell_multiplier',
           'cell_lev_multiplier', 'entry_fee', 'exit_fee', 'total_fee', 'pnl_current_sizing'}
_TRUE = ("True", "true", "1", "1.0")


def _exports():
    return sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_*.csv")), key=os.path.getmtime)


_CSV = {}
def _read(path, **kw):
    """read_csv memoised on (path, mtime, size) — the scout reads the master + every export once per run, not once per sleeve."""
    st = os.stat(path)
    k = (path, st.st_mtime_ns, st.st_size, tuple(sorted((a, str(b)) for a, b in kw.items())))
    if k not in _CSV:
        if len(_CSV) > 64:
            _CSV.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CSV[k] = pd.read_csv(path, low_memory=False, **kw)
    return _CSV[k].copy()


def load_real(sleeve, master=MASTER, exports=None, batch_start=None):
    """→ the sleeve's real closed fills: master (stack_keep, pnl_percentage := stack_pct) + current-batch exports (as traded), dedup
    (opened_at, pair, direction), the newest export wins. 'MOMENTUM:LONG' / 'MOMENTUM:SHORT' = entry_strategy MOMENTUM of that direction."""
    es, dr = (sleeve.split(":", 1) if sleeve.startswith("MOMENTUM:") else (sleeve, ""))
    fr, mkeys = [], set()
    if master and os.path.exists(master):
        m = _read(master)
        _k = m.opened_at.astype(str).str.replace(" ", "T").str[:19]
        mkeys = set(zip(_k, m.pair.astype(str), m.direction.astype(str)))      # EVERY master row (kept or blocked by today's stack)
        if batch_start is None and len(m):
            batch_start = max(BATCH_START, str(_k.max()))                       # the batch = fills after the master's last one
        m = m[(m.entry_strategy.astype(str) == es) & m.stack_keep.astype(str).isin(_TRUE)].copy()
        if "stack_pct" in m:
            m["pnl_percentage"] = pd.to_numeric(m.stack_pct, errors="coerce")
        fr.append(m.assign(_src="master", _rank=-1))
    for i, f in enumerate(_exports() if exports is None else exports):
        try:
            x = _read(f)
        except Exception:
            continue
        if "entry_strategy" not in x or "opened_at" not in x:
            continue
        _xk = x.opened_at.astype(str).str.replace(" ", "T").str[:19]
        x = x[(x.entry_strategy.astype(str) == es) & (_xk >= (batch_start or BATCH_START))
              & ~pd.Series([k in mkeys for k in zip(_xk, x.pair.astype(str), x.direction.astype(str))], index=x.index)]
        if len(x):
            fr.append(x.assign(_src="batch", _rank=i))
    if not fr:
        return pd.DataFrame()
    d = pd.concat(fr, ignore_index=True)
    d["opened_at"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d = d.sort_values("_rank", kind="stable").drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    d["pnl_percentage"] = pd.to_numeric(d.pnl_percentage, errors="coerce")
    d = d[(d.status.astype(str) == "CLOSED") & d.pnl_percentage.notna()]
    if dr:
        d = d[d.direction.astype(str) == dr]
    return d.reset_index(drop=True)


def sleeves(master=MASTER, exports=None):
    """every bot sleeve name seen in the master or the newest exports (MOMENTUM split by direction; MANUAL out)."""
    fr = []
    if os.path.exists(master):
        fr.append(_read(master)[["entry_strategy", "direction"]])
    for f in (_exports() if exports is None else exports)[-3:]:
        try:
            fr.append(_read(f)[["entry_strategy", "direction"]])
        except Exception:
            pass
    d = pd.concat(fr).dropna() if fr else pd.DataFrame(columns=["entry_strategy", "direction"])
    return sorted({("MOMENTUM:" + r.direction) if r.entry_strategy == "MOMENTUM" else str(r.entry_strategy)
                   for r in d.itertuples() if r.entry_strategy != "MANUAL"})


MAX_LEVELS = 6


def candidates(df, detail=False):
    """numeric entry_* columns scored on ≥ 80 % of fills (> 3 distinct values) → threshold splits; LEVEL columns (≤ MAX_LEVELS distinct
    values: flags, pattern matches, regimes, text) scored on ≥ 80 % → level-vs-rest splits; the rest are listed, never dropped silently:
    blind = scored on < 80 % (not tested), empty = never stamped on this sleeve, other = text with too many levels.
    detail=False → (cands, blind) for the legacy callers; True → (cands, levels, blind, empty, other)."""
    for c in [c for c in df.columns if c.startswith('entry_')]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            n = pd.to_numeric(df[c], errors='coerce')
            if n.notna().any() and n.notna().sum() >= 0.8 * df[c].notna().sum():
                df[c] = n
    ent = [c for c in df.columns if c.startswith('entry_') and c not in EXCLUDE]
    empty = sorted(c for c in ent if df[c].notna().sum() == 0)
    num = [c for c in ent if c not in empty and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 3]
    lev = [c for c in ent if c not in empty and c not in num and 2 <= df[c].nunique() <= MAX_LEVELS]
    other = sorted(c for c in ent if c not in empty and c not in num and c not in lev and df[c].nunique() > MAX_LEVELS)
    cands = [c for c in num if df[c].notna().sum() >= 0.8 * len(df)]
    levels = [c for c in lev if df[c].notna().sum() >= 0.8 * len(df)]
    blind = sorted(c for c in num + lev if c not in cands and c not in levels)
    return (cands, levels, blind, empty, other) if detail else (cands, blind)


def _split_stats(d, mask):
    hi, lo = d[mask], d[~mask]
    if len(hi) < 3 or len(lo) < 3:
        return None
    return dict(n_hi=len(hi), n_lo=len(lo), d_avg=hi.pnl_percentage.mean() - lo.pnl_percentage.mean())


def sweep_levels(df, levels):
    """level-vs-rest for flag / categorical columns (each level of each column, cross-era consistency as sweep_1d)."""
    out = []
    for col in levels:
        for v in sorted(df[col].dropna().unique(), key=str):
            rows, ok = {}, True
            for era in ('A', 'B'):
                d = df[df.era == era].dropna(subset=[col])
                st = _split_stats(d, d[col] == v)
                if st is None:
                    ok = False
                    break
                rows[era] = st
            if not ok:
                continue
            a, b = rows['A']['d_avg'], rows['B']['d_avg']
            consistent = (a * b > 0) and min(abs(a), abs(b)) > 0.05
            out.append(dict(col=col, gran=f"={v}"[:14], dA=a, dB=b, consistent=consistent, strength=min(abs(a), abs(b)) if consistent else 0,
                            nA=rows['A']['n_hi'] + rows['A']['n_lo'], nB=rows['B']['n_hi'] + rows['B']['n_lo']))
    return pd.DataFrame(out, columns=["col", "gran", "dA", "dB", "consistent", "strength", "nA", "nB"])


def sweep_1d(df, cands, levels=()):
    results = []
    for col in cands:
        for gran in ('sign', 'median', 'tercile'):
            if gran == 'sign' and not ((df[col] > 0).any() and (df[col] <= 0).any()):
                continue
            rows, ok = {}, True
            ref = df[df.era == 'A'][col].dropna()   # median / tercile computed on era A only (no lookahead into the fresh era)
            for era in ('A', 'B'):
                d = df[df.era == era].dropna(subset=[col])
                mask = d[col] > (0 if gran == 'sign' else ref.median() if gran == 'median' else ref.quantile(2 / 3))
                st = _split_stats(d, mask)
                if st is None:
                    ok = False
                    break
                rows[era] = st
            if not ok:
                continue
            a, b = rows['A']['d_avg'], rows['B']['d_avg']
            consistent = (a * b > 0) and min(abs(a), abs(b)) > 0.05
            results.append(dict(col=col, gran=gran, dA=a, dB=b, consistent=consistent,
                                strength=min(abs(a), abs(b)) if consistent else 0,
                                nA=rows['A']['n_hi'] + rows['A']['n_lo'], nB=rows['B']['n_hi'] + rows['B']['n_lo']))
    r = pd.DataFrame(results, columns=["col", "gran", "dA", "dB", "consistent", "strength", "nA", "nB"])
    if levels:
        lv = sweep_levels(df, levels)
        r = pd.concat([r, lv], ignore_index=True) if len(lv) else r
    return r


def sweep_2d(df, res, cands):
    """every dimension PAIR (top 36), 4 quadrants (median × median, thresholds anchored on era A), quadrant-vs-rest Δ, cross-era
    consistency, min 5 trades per quadrant per era. With thousands of tests expect false survivors — it RANKS, never ships."""
    top = list(dict.fromkeys([c for c in res.loc[res.consistent.astype(bool)].col.tolist() if c in cands] + cands))[:36]   # numeric only
    refA = {c: df[df.era == 'A'][c].dropna().median() for c in top}
    out = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c1, c2 = top[i], top[j]
            d = df.dropna(subset=[c1, c2])
            for q1 in (True, False):
                for q2 in (True, False):
                    ok, deltas = True, []
                    for era in ('A', 'B'):
                        de = d[d.era == era]
                        m = ((de[c1] > refA[c1]) == q1) & ((de[c2] > refA[c2]) == q2)
                        if m.sum() < 5 or (~m).sum() < 5:
                            ok = False
                            break
                        deltas.append(de[m].pnl_percentage.mean() - de[~m].pnl_percentage.mean())
                    if not ok:
                        continue
                    a, b = deltas
                    if a * b > 0 and min(abs(a), abs(b)) > 0.10:
                        out.append(dict(pair=f"{c1.replace('entry_', '')}{'>' if q1 else '<='}med x {c2.replace('entry_', '')}{'>' if q2 else '<='}med",
                                        dA=a, dB=b, strength=min(abs(a), abs(b))))
    return pd.DataFrame(out, columns=["pair", "dA", "dB", "strength"]), len(top) * (len(top) - 1) // 2 * 4


def prepare(sleeve, split=None, master=MASTER, exports=None):
    """→ (df with era, split, note) or (None, None, reason)."""
    if sleeve in LEGACY:
        df = pd.read_csv(POOL)
        df = df[df.screen_sleeve == LEGACY[sleeve]].copy()
        split = split or '2026-06-30'
        note = f"{sleeve}: SCREENED_BASELINE pool {len(df)} fills"
    else:
        df = load_real(sleeve, master, exports)
        if len(df) < MIN_FILLS:
            return None, None, f"{sleeve}: {len(df)} real fills — too few to sweep (need ≥ {MIN_FILLS})"
        split = split or pd.to_datetime(df.opened_at).median().strftime("%Y-%m-%dT%H:%M:%S")   # same 'T' format as opened_at (review)
        note = (f"{sleeve}: {len(df)} REAL fills (master on today's stack {int((df._src == 'master').sum())} · current batch "
                f"{int((df._src == 'batch').sum())}) · era split {split}" + (" · ⚠ N < 30: a screen on a thin cohort" if len(df) < 30 else ""))
    df['era'] = np.where(df.opened_at.astype(str) < split, 'A', 'B')
    if df.era.nunique() < 2:
        return None, None, f"only one era in pool for {sleeve} at split {split}"
    return df, split, note


def run(sleeve, split=None, two_d=True, master=MASTER, exports=None):
    """→ dict(note, n, split, cands, blind, res, cons, res2, n2) or dict(note) when not sweepable. Pure on its inputs."""
    df, split, note = prepare(sleeve, split, master, exports)
    if df is None:
        return dict(note=note, n=0)
    cands, levels, blind, empty, other = candidates(df, detail=True)
    res = sweep_1d(df, cands, levels)
    res["consistent"] = res.consistent.astype(bool)
    cons = res.loc[res.consistent.astype(bool)].sort_values('strength', ascending=False)
    r2, n2 = sweep_2d(df, res, cands) if two_d else (pd.DataFrame(), 0)
    return dict(note=note, n=len(df), split=split, df=df, cands=cands, levels=levels, blind=blind, empty=empty, other=other,
                res=res, cons=cons, res2=r2, n2=n2)


def main(argv):
    sleeve = argv[1] if len(argv) > 1 else 'FS'
    split = argv[2] if len(argv) > 2 else None
    if sleeve == "ALL":   # every bot sleeve, one sweep each
        for nm in sleeves():
            print(f"\n════════ {nm} ════════", flush=True)
            try:
                main([argv[0], nm] + argv[2:3])
            except Exception as e:   # one sleeve's failure never hides the others
                print(f"⚠ {nm}: sweep failed — {type(e).__name__}: {str(e)[:160]}")
        return
    R = run(sleeve, split)
    print(R["note"])
    if not R["n"]:
        return
    df, res, cons = R["df"], R["res"], R["cons"]
    print(f"coverage: {len(R['cands'])} numeric + {len(R['levels'])} level (flag / category) entry_* columns tested · BLIND SPOTS "
          f"(< 80 % of fills scored, NOT tested): " + (", ".join(f"{c[6:]} {df[c].notna().mean():.0%}" for c in R['blind']) if R['blind'] else "none")
          + f" · never stamped on this sleeve: {len(R['empty'])}" + (f" · text with > {MAX_LEVELS} levels (not tested): "
          + ", ".join(c[6:] for c in R['other']) if R['other'] else ""))
    print(f"SWEEP {sleeve}: {res.col.nunique()} dimensions x {len(res)} tests | era A (<{R['split']}) vs B | consistent: {len(cons)}")
    print(f"{'dimension':<42}{'gran':<9}{'ΔavgA':>8}{'ΔavgB':>8}{'strength':>9}")
    for _, r in cons.head(15).iterrows():
        print(f"{r.col:<42}{r.gran:<9}{r.dA:>+8.3f}{r.dB:>+8.3f}{r.strength:>9.3f}")
    if not len(cons):
        print("(no cross-period-consistent separator at any granularity — a 'none exists' verdict is now defensible)")
    print("\n── 2D INTERACTION SWEEP ──")
    r2 = R["res2"]
    print(f"pairs tested: ~{R['n2']} quadrant cells | consistent (|Δ|>0.10 both eras, minN 5/quadrant/era): {len(r2)}")
    if len(r2):
        for _, r in r2.sort_values('strength', ascending=False).head(12).iterrows():
            print(f"  {r['pair']:<70}{r.dA:>+7.2f}{r.dB:>+7.2f}  str {r.strength:.2f}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main(sys.argv)
