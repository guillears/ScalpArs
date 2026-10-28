"""🔎 Scout — SEPARATOR SWEEP for EVERY sleeve (DECISION_LOG 398, operator 2026-10-27: "all SLEEVES should have it! why this one
dont?" · "add it to ALL sleeves, to scout, to watchlist"). Alert-only; changes nothing.

WHAT   scripts/sweep_separators.py (the sleeve-kill checklist's automated screen, CLAUDE.md) on every bot sleeve's REAL fills — the master
       on today's stack (stack_keep, stack_pct) + the current batch (as traded); FS / ML / MS also on their SCREENED_BASELINE pool. Per
       sleeve: N (master · batch) · era split · columns tested · BLIND SPOTS (< 80 % scored, not tested) · 1D tests and cross-era-consistent
       survivors vs a SHUFFLED NULL (P&L permuted within each era, same tests — survivors expected by luck) · 2D quadrant survivors (EVERY column pair, DECISION_LOG 402) vs
       their null · the top 1D survivors. A survivor count not clearly above its null = nothing found. Survivors are a SCREEN for the
       locked gates (expectancy bar, window units, theory, haircut), never a ship list.
COST   local CSVs only, no network. 1D null ×150, 2D + 3D null ×200 shuffles — so each sleeve's row is CACHED in
       reports/.scout_sweep_cache.json under a fingerprint of its cohort (N, last fill, Σ P&L, columns, split, code version): a sleeve is
       re-swept only when its fills changed.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_separators as SW  # noqa: E402

DL = 398
CACHE = os.path.join(SW.ROOT, "reports", ".scout_sweep_cache.json")
VERSION = 5   # bump when the sweep / null logic changes (the sources of both scripts are hashed into the key too)
NULL_1D, NULL_2D = 150, 200  # each level is judged at p ≤ 0.05 / 3: 2 / 151 · 3 / 201 still pass (at 100 shuffles only 1 / 101 = beat EVERY shuffle) — 2D / 3D are ms per shuffle (DECISION_LOG 402, 403; the 3D null runs on the 2D shuffles), 1D ~0.5 s on the biggest sleeve
P_BAR = 0.05
LEVELS = 3                 # 1D + 2D + 3D-from-2D families → each judged at P_BAR / LEVELS (Bonferroni over the levels, DECISION_LOG 402, 403)
BUDGET_S = 240             # the opportunity scout's time budget for re-sweeping changed sleeves (the rest: "not swept this run")


def _day_block_perm(s, rng):
    """within each era: the UTC days' P&L blocks in a random order, laid back onto the fills in time order — same-day fills keep their
    shared outcome together (WINDOW-UNITS rule: a trade-level shuffle understates the null's spread), the feature → P&L link breaks."""
    out = s.pnl_percentage.to_numpy().copy()
    for era in ("A", "B"):
        idx = np.where((s.era == era).to_numpy())[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(s.opened_at.to_numpy()[idx], kind="stable")]
        day = s.opened_at.astype(str).str[:10].to_numpy()[idx]
        blocks = [out[idx][day == d] for d in dict.fromkeys(day)]
        order = rng.permutation(len(blocks))
        out[idx] = np.concatenate([blocks[i] for i in order])
    s = s.copy()
    s["pnl_percentage"] = out
    return s


def _null(df, cands, k, two_d, seed=398, levels=(), k2=None):
    """under DAY-BLOCK shuffles of pnl_percentage within each era (pure) → (1D survivor counts, 2D counts or None, 1D best strengths)
    + (2D best strengths, 3D-from-2D best strengths) when two_d. ONE set of shuffles feeds all three levels: 1D on the first k, 2D and
    3D on the first k2 (default k; at NULL_1D 150 / NULL_2D 200 shuffles 151–200 run 2D + 3D only) —
    the 2D sweep never needs the 1D result, so a 2D null never re-runs the slow 1D pass."""
    rng = np.random.default_rng(seed)
    k2 = (k if k2 is None else k2) if two_d else 0
    c1, c2, mx, mx2, mx3 = [], [], [], [], []
    for i in range(max(k, k2)):
        s = _day_block_perm(df, rng)
        if i < k:
            r = SW.sweep_1d(s, cands, levels)
            c1.append(int(r.consistent.astype(bool).sum()) if len(r) else 0)
            mx.append(float(r.strength.max()) if len(r) else 0.0)
        if i < k2:
            n_, b_ = SW.sweep_2d(s, None, cands, stats_only=True)
            c2.append(n_)
            mx2.append(b_)
            mx3.append(SW.sweep_3d(s, None, cands, stats_only=True)[1])   # the WHOLE 3D path: this shuffle's own top 2D cells (403)
    return (c1, (c2 if two_d else None), mx) + ((mx2, mx3) if two_d else ())


def _src_hash():
    h = hashlib.sha1()
    for f in (SW.__file__, __file__):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _fingerprint(name, master, exports):
    """the cohort AND its values (a COALESCE stamp backfill changes values, not N) + both scripts' source + the null sizes."""
    df, split, note = SW.prepare(name, None, master, exports)
    if df is None:
        return None
    cols = sorted(c for c in df.columns if c.startswith("entry_")) + ["pnl_percentage", "opened_at", "pair"]
    vh = int(pd.util.hash_pandas_object(df[cols].astype(str), index=False).sum() % (1 << 61))
    key = json.dumps([VERSION, _src_hash(), NULL_1D, NULL_2D, name, len(df), split, vh])
    return hashlib.sha1(key.encode()).hexdigest()


def _load_cache(cache_path):
    try:
        with open(cache_path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def sleeve_row_cached(name, master=SW.MASTER, exports=None, cache_path=CACHE, allow_compute=True):
    """the cached row when the cohort fingerprint matches; else re-swept (allow_compute) or None (the caller's budget ran out)."""
    cache = _load_cache(cache_path) if cache_path else {}
    fp = _fingerprint(name, master, exports)
    if fp is not None and cache.get(name, {}).get("fp") == fp:
        return cache[name]["row"]
    if fp is not None and not allow_compute:
        return None
    r = sleeve_row(name, master, exports)
    if fp is not None and cache_path:
        cache = _load_cache(cache_path)
        import time as _t
        cache[name] = dict(fp=fp, row=r, t=_t.time())   # t = last swept: run() re-sweeps changed sleeves oldest-first (review: no starvation)
        try:
            tmp = f"{cache_path}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(cache, fh, default=float)
            os.replace(tmp, cache_path)
        except Exception:
            pass
    return r


def sleeve_row(name, master=SW.MASTER, exports=None, null_1d=NULL_1D, null_2d=NULL_2D):
    """→ dict for one sleeve (pure on its inputs)."""
    R = SW.run(name, None, True, master, exports)
    if not R["n"]:
        return dict(name=name, note=R["note"], n=0)
    df, res = R["df"], R["res"]
    n1 = len(res); s1 = int(res.consistent.astype(bool).sum()) if n1 else 0
    s2 = len(R["res2"])
    if n1:
        _nl = _null(df, R["cands"], null_1d, bool(null_2d), levels=R["levels"], k2=null_2d)
        c1, mx = _nl[0], _nl[2]
        c2, mx2, mx3 = (_nl[1], _nl[3], _nl[4]) if null_2d else (None, [], [])
    else:
        c1, mx, c2, mx2, mx3 = [], [], None, [], []
    best = float(res.strength.max()) if n1 else 0.0
    pmax = (1 + sum(x >= best for x in mx)) / (len(mx) + 1) if mx else None   # FAMILY-WISE: the best split vs luck's best split
    best2 = float(R["res2"].strength.max()) if s2 else 0.0
    pmax2 = ((1 + sum(x >= best2 for x in mx2)) / (len(mx2) + 1) if best2 > 0 else 1.0) if (mx2 and (best2 > 0 or max(mx2) > 0)) else None
    r3 = R.get("res3", pd.DataFrame())
    s3 = len(r3)
    best3 = float(r3.strength.max()) if s3 else 0.0
    pmax3 = ((1 + sum(x >= best3 for x in mx3)) / (len(mx3) + 1) if best3 > 0 else 1.0) if (mx3 and (best3 > 0 or max(mx3) > 0)) else None
    nl1 = float(np.mean(c1)) if c1 else 0.0
    p1 = (1 + sum(x >= s1 for x in c1)) / (len(c1) + 1) if c1 else None     # empirical one-sided p (survivor counts are heavy-tailed)
    nl2 = float(np.mean(c2)) if c2 else None
    p2 = (1 + sum(x >= s2 for x in c2)) / (len(c2) + 1) if c2 else None
    top = R["cons"].head(3)
    return dict(name=name, note=R["note"], n=R["n"], nm=int((df._src == "master").sum()) if "_src" in df else R["n"],
                nb=int((df._src == "batch").sum()) if "_src" in df else 0, split=R["split"], cols=len(R["cands"]), lev=len(R["levels"]),
                empty=len(R["empty"]), other=len(R["other"]), blind=R["blind"],
                bcov={c: float(df[c].notna().mean()) for c in R["blind"]}, tests=n1, s1=s1, null1=nl1, p1=p1, k1=len(c1), best=best,
                null_best=(float(np.quantile(mx, 0.95)) if mx else None), pmax=pmax,
                s2=s2, n2=R["n2"], null2=nl2, p2=p2, k2=len(c2 or []), best2=best2,
                null_best2=(float(np.quantile(mx2, 0.95)) if mx2 else None), pmax2=pmax2,
                top2=[(x.pair, x.dA, x.dB) for x in R["res2"].head(2).itertuples()] if s2 else [],
                s3=s3, n3=R.get("n3", 0), best3=best3, null_best3=(float(np.quantile(mx3, 0.95)) if mx3 else None), pmax3=pmax3,
                k3=len(mx3), top3=[(x.cell, x.third, x.dA, x.dB) for x in r3.head(1).itertuples()] if s3 else [],
                top=[(r.col[6:], r.gran, r.dA, r.dB) for r in top.itertuples()])


def verdict(r):
    if not r["n"]:
        return f"{(r.get('note') or 'too few fills').split(': ', 1)[-1]} — keep collecting"
    if r["tests"] == 0:
        return "too thin to split into two eras (each era needs ≥ 3 fills per side) — keep collecting"
    bar = P_BAR / LEVELS
    pm, b, nb = r.get("pmax"), r.get("best"), r.get("null_best")
    pm2, b2, nb2 = r.get("pmax2"), r.get("best2"), r.get("null_best2")
    t1 = f"1D best {b or 0:.2f} % vs luck {nb if nb is not None else 0:.2f} (p {'–' if pm is None else f'{pm:.3f}'})"
    t2 = (f"2D best {b2 or 0:.2f} % vs luck {nb2 if nb2 is not None else 0:.2f} (p {'–' if pm2 is None else f'{pm2:.3f}'})"
          if pm2 is not None else ("2D: null not run" if not r.get("k2") else
                                   "2D: no consistent quadrant, real or shuffled (cells < 5 fills per era, or none agrees across eras)"))
    pm3, b3, nb3 = r.get("pmax3"), r.get("best3"), r.get("null_best3")
    t3 = (f"3D best {b3 or 0:.2f} % vs luck {nb3 if nb3 is not None else 0:.2f} (p {'–' if pm3 is None else f'{pm3:.3f}'})"
          if pm3 is not None else ("3D: null not run" if not r.get("k3") else
                                   "3D: no consistent 2D cell, or no third variable splits one consistently (≥ 5 per half per era, "
                                   "same sign both eras), real or shuffled"))
    hit = [lab for lab, p_, v in (("1D", pm, b), ("2D", pm2, b2), ("3D", pm3, b3)) if p_ is not None and p_ <= bar and v]
    t2 = f"{t2} · {t3}"
    if hit:
        return (f"{t1} · {t2} → ABOVE LUCK at {' + '.join(hit)} (each level at p ≤ {bar:.3f}): the top split(s) are CANDIDATES for the "
                "locked gates (W/L read at the review)")
    return f"{t1} · {t2} → not above luck at any level (each at p ≤ {bar:.3f}; screen only)"


def run(names=None, master=SW.MASTER, exports=None, cache_path=CACHE, budget_s=None):
    """budget_s: stop re-sweeping changed sleeves after this many seconds (cached rows are always shown); None = no limit."""
    import time
    t_start = time.time()
    L = [f"## 🔎 SEPARATOR SWEEP — every sleeve (DECISION_LOG {DL}, alert-only)", "",
         "scripts/sweep_separators.py on each sleeve's REAL fills (master on today's stack + the current batch; FS / ML / MS = the "
         "SCREENED_BASELINE pool). Survivors = cross-era-consistent splits (sign / median / tercile, thresholds from era A); null = the same "
         f"tests with P&L shuffled in UTC-DAY blocks within each era (same-day fills move together — window units; 1D × {NULL_1D}; 2D × {NULL_2D}). "
         f"2D = EVERY pair of numeric columns × 4 median quadrants (DECISION_LOG 402). READ = FAMILY-WISE PER LEVEL: the sleeve's strongest "
         f"split (min |Δ| over both eras) against the strongest split of each shuffle at the same level — p ≤ {P_BAR}/{LEVELS} = {P_BAR / LEVELS:.3f} per level (Bonferroni over 1D + 2D + 3D) = above luck. 3D = FROM 2D (DECISION_LOG 403): the 10 strongest 2D cells, each split inside by every other variable; its null re-picks the top cells on each shuffle. A survivor COUNT is shown but "
         "is a weak statistic (collinear columns, a weak 'both eras agree' bar — luck passes ~a third of the tests). A screen for the locked gates — never a ship list; "
         "every 'no separator' / switch-off claim cites this table + the sleeve's own run.", "",
         "| Sleeve | N (master · batch) | Era split | Cols tested (numeric + flag/category · never stamped) | Blind spots (< 80 %) | 1D tests · survivors · null (count p) | 2D cells · survivors · null | 3D-from-2D tests (cells × other vars) · survivors · top | Top 1D (ΔA / ΔB %) | Read (family-wise: best split vs luck's best) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    names = names if names is not None else [n for n in SW.sleeves(master, exports)] + ["FS", "ML", "MS"]
    # review (DECISION_LOG 402): the budget is spent OLDEST-SWEPT FIRST (never-swept first), so a sleeve late in the list cannot be
    # starved run after run by earlier ones that keep changing; the table keeps the fixed order below.
    _c = _load_cache(cache_path) if cache_path else {}
    got = {}
    for nm in sorted(names, key=lambda n: float((_c.get(n) or {}).get("t") or 0.0)):
        try:
            got[nm] = sleeve_row_cached(nm, master, exports, cache_path,
                                        allow_compute=(budget_s is None or time.time() - t_start < budget_s))
        except Exception as e:
            got[nm] = e
    for nm in names:
        r = got.get(nm)
        if isinstance(r, Exception):
            e = r
            L.append(f"| {nm} | – | – | – | – | – | – | – | – | ⚠ failed: {type(e).__name__}: {str(e)[:80].replace("|", "/")} |")
            continue
        if r is None:
            L.append(f"| {nm} | changed since the last sweep | – | – | – | – | – | – | – | not swept this run (time budget) — "
                     "`venv/bin/python scripts/scout_sweep_all.py` refreshes it |")
            continue
        if not r["n"]:
            L.append(f"| {nm} | {r['note'].split(': ', 1)[-1]} | – | – | – | – | – | – | – | {verdict(r)} |")
            continue
        bl = ", ".join(f"{c[6:]} {v:.0%}" for c, v in list(r["bcov"].items())[:6]) + (f" +{len(r['blind']) - 6}" if len(r["blind"]) > 6 else "") if r["blind"] else "none"
        top = " · ".join(f"{c} {g} {a:+.2f}/{b:+.2f}" for c, g, a, b in r["top"]) or "–"
        nl2 = "–" if r["null2"] is None else f"{r['null2']:.0f} (×{r.get('k2', NULL_2D)})"
        n2 = (f"{r['n2']} · {r['s2']} · {nl2}" + (" · top " + " · ".join(f"{p_} {a:+.2f}/{b:+.2f}" for p_, a, b in r.get("top2", [])) if r.get("top2") else ""))
        n3 = (f"{r.get('n3', 0)} · {r.get('s3', 0)}" + (" · top " + " · ".join(f"[{c}] {t} {a:+.2f}/{b:+.2f}" for c, t, a, b in r.get("top3", []))
                                                       if r.get("top3") else ""))
        L.append(f"| {nm} | {r['n']} ({r['nm']} · {r['nb']}){' ⚠ N<30' if r['n'] < 30 else ''} | {str(r['split'])[:10]} | "
                 f"{r['cols']} + {r.get('lev', 0)} · {r.get('empty', 0)} | {bl} | "
                 f"{r['tests']} · {r['s1']} · {r['null1']:.1f} (p {'–' if r.get('p1') is None else format(r['p1'], '.3f')}) | {n2} | {n3} | {top} | {verdict(r)} |")
    L += ["", "_Full per-sleeve output: `venv/bin/python scripts/sweep_separators.py <SLEEVE>` (or ALL). Blind-spot columns need their own "
          "read (scored subset) — never counted as 'rest'._", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run(sys.argv[1:] or None)))
