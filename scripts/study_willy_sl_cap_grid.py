#!/usr/bin/env python3
"""WILLY fixed-stop × time-cap grid (operator 2026-10-13: "why not SL −2 or a 30-min exit, or both?"). Same cohorts / paths as
scripts/study_willy_late_sl.py (reports/WILLY_LATE_SL_rows.csv supplies the fills): TP +1.25 always; stop −2 / −2.5 / −3 / −4 from entry ×
cap 20 / 30 / 45 / 60 / 90 min. Base = −4 / 90 (today). Output reports/WILLY_SL_CAP_GRID.txt."""
import os, sys, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_timecap as WT  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
TP = 1.25
GRID = [(s, c) for s in (2.0, 2.5, 3.0, 4.0) for c in (20, 30, 45, 60, 90)]
SPLIT = pd.Timestamp("2026-07-01")
rng = np.random.default_rng(7)


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), 90, tc)
        rec = dict(r)
        for s, c in GRID:
            v = FX.evaluate(P, TP, s, c)
            rec[f"S{s:g}C{c}"] = v["pct"]
        out.append(rec)
    return out


def day_p(d, day):
    g = pd.DataFrame(dict(d=d, day=day)).groupby("day").d.agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    idx = rng.integers(0, len(g), size=(4000, len(g)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float((m > 0).mean())


def main():
    R0 = pd.read_csv(os.path.join(ROOT, "reports", "WILLY_LATE_SL_rows.csv"), low_memory=False)
    R0 = R0[R0.base_why != "UNRES"]
    items = R0[["cohort", "pair", "te", "E", "ts", "day", "usd_per_pt"]].to_dict("records")
    pairs = sorted({r["pair"] for r in items}); grp = {p: k % 6 for k, p in enumerate(pairs)}
    with Pool(6) as P:
        R = pd.DataFrame(sum(P.map(job, [[r for r in items if grp[r["pair"]] == k] for k in range(6)]), []))
    R.to_csv(os.path.join(ROOT, "reports", "WILLY_SL_CAP_GRID_rows.csv"), index=False)
    L = ["WILLY fixed stop × time cap (TP +1.25 always; base = −4 / 90 min = today) — own paths, ticks → 1m tail", ""]
    for coh in ("BACKTEST", "B19", "B20"):
        D = R[R.cohort == coh]
        if coh != "BACKTEST":
            M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
            M = M[(M.entry_strategy == "FRENZY_WILLY") & (M.stack_keep == True)]   # noqa: E712
            u = {(str(a)[:19].replace("T", " "), p): s / c for a, p, s, c in zip(M.opened_at, M.pair, M.stack_pnl, M.stack_pct) if c}
            D = D.assign(usd_per_pt=[u.get((str(t)[:19], p), np.nan) for t, p in zip(D.ts, D.pair)])
        base = D["S4C90"]
        L.append(f"{coh}: {len(D)} fills · base −4/90: WR {(base > 0).mean() * 100:.0f} % · avg {base.mean():+.3f} % · ${(base * D.usd_per_pt).sum():+,.0f}")
        L.append(f"  {'stop/cap':9} {'WR':>4} {'avg %':>7} {'Δavg':>7} {'BE WR':>6} {'P(Δ>0)':>6} {'H1 avg':>7} {'H2 avg':>7} {'$':>8} {'Δ$':>8}")
        rows = []
        for s, c in GRID:
            v = D[f"S{s:g}C{c}"]; d = v - base
            ok = v.notna()
            w = v[ok & (v > 0)].mean(); l_ = -v[ok & (v <= 0)].mean()
            be = l_ / (w + l_) * 100 if w == w and l_ == l_ else np.nan
            p = day_p(d[ok].values, D.day[ok].values) if coh == "BACKTEST" else np.nan
            h1 = v[pd.to_datetime(D.ts, format="mixed") < SPLIT].mean(); h2 = v[pd.to_datetime(D.ts, format="mixed") >= SPLIT].mean()
            rows.append((f"−{s:g} / {c}", (v[ok] > 0).mean() * 100, v.mean(), d.mean(), be, p, h1, h2, (v * D.usd_per_pt).sum(), (d * D.usd_per_pt).sum()))
        for r in sorted(rows, key=lambda x: -x[2]):
            L.append(f"  {r[0]:9} {r[1]:4.0f} {r[2]:+7.3f} {r[3]:+7.3f} {r[4]:6.1f} {r[5]:6.2f} {r[6]:+7.3f} {r[7]:+7.3f} {r[8]:+8,.0f} {r[9]:+8,.0f}")
        L.append("")
    txt = "\n".join(L)
    open(os.path.join(ROOT, "reports", "WILLY_SL_CAP_GRID.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
