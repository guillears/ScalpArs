#!/usr/bin/env python3
"""Calibration: match replay fills against REAL closed fills (master pool, stack_keep=True,
full-size only) inside a window. Match key = (pair, direction, |Δopened| ≤ tol min).
Reports recall (real fills reproduced), precision (replay fills that happened live), and
the P&L / entry-price gap on matched pairs, per sleeve."""
import os, sys, argparse
import pandas as pd, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "reports", "backtest_cache", "replay")
ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
ap.add_argument("--tol-min", type=float, default=20)
ap.add_argument("--sleeves", default="")   # comma list to restrict (e.g. MOMENTUM,FLIP)
A = ap.parse_args()

def sleeve_of(strat, direction):
    s = (strat or "MOMENTUM")
    if s.startswith("FLIP"): return "FLIP-short"
    if s == "MOMENTUM": return f"MOM-{direction.lower()}"
    if s in ("BULLRUN_LONG",): return "BullRun-Long"
    if s in ("BEARRUN_SHORT",): return "BearRun-Short"
    return s
rep = pd.concat([pd.read_csv(os.path.join(OUT, f"{t}_orders.csv"), low_memory=False) for t in A.tags], ignore_index=True)
rep = rep[rep.status == "CLOSED"].copy()
rep["opened_at"] = pd.to_datetime(rep["opened_at"], format="mixed")
rep = rep[(rep.opened_at >= A.start) & (rep.opened_at < A.end)]
rep["sleeve"] = [sleeve_of(s if isinstance(s, str) else None, d) for s, d in zip(rep.entry_strategy, rep.direction)]
src = rep.get("cell_multiplier_source")
rep["is_probe"] = src.fillna("").astype(str).str.contains("PROBE") if src is not None else False

m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
m["opened_at"] = pd.to_datetime(m["opened_at"], format="mixed")
m = m[(m.opened_at >= A.start) & (m.opened_at < A.end) & (m.stack_keep == True) & (~m.is_probe.fillna(False).astype(bool))].copy()
m["sleeve"] = [sleeve_of(s if isinstance(s, str) else None, d) for s, d in zip(m.entry_strategy, m.direction)]
if A.sleeves:
    keep = set(A.sleeves.split(","))
    m = m[m.sleeve.isin(keep)]; rep = rep[rep.sleeve.isin(keep)]
rep_ns = rep[~rep.is_probe]

def match(a, b):
    used = set(); pairs = []
    for i, r in a.iterrows():
        c = b[(b.pair == r.pair) & (b.direction == r.direction)]
        c = c[abs((c.opened_at - r.opened_at).dt.total_seconds()) <= A.tol_min * 60]
        c = c[~c.index.isin(used)]
        if len(c):
            j = (abs((c.opened_at - r.opened_at).dt.total_seconds())).idxmin()
            used.add(j); pairs.append((i, j))
    return pairs

pairs = match(m, rep_ns)
mi = {i for i, _ in pairs}; ri = {j for _, j in pairs}
print(f"window {A.start}→{A.end} tol ±{A.tol_min:.0f} min | REAL stack-kept full-size: {len(m)} | REPLAY full-size: {len(rep_ns)} (+{int(rep.is_probe.sum())} probe)")
rows = []
for sl in sorted(set(m.sleeve) | set(rep_ns.sleeve)):
    mm = m[m.sleeve == sl]; rr = rep_ns[rep_ns.sleeve == sl]
    hit = mm.index.isin(list(mi)).sum(); phit = rr.index.isin(list(ri)).sum()
    rows.append((sl, len(mm), hit, f"{hit/len(mm)*100:.0f}%" if len(mm) else "-", len(rr), phit,
                 f"{phit/len(rr)*100:.0f}%" if len(rr) else "-",
                 f"{mm.pnl.sum():+.0f}", f"{rr.pnl.sum():+.0f}",
                 f"{(mm.pnl>0).mean()*100:.0f}%" if len(mm) else "-", f"{(rr.pnl>0).mean()*100:.0f}%" if len(rr) else "-"))
print(pd.DataFrame(rows, columns=["sleeve","real N","matched","recall","replay N","matched","precision","real $","replay $","real WR","replay WR"]).to_string(index=False))
if pairs:
    d = pd.DataFrame([{"pair": m.loc[i,"pair"], "dir": m.loc[i,"direction"], "sleeve": m.loc[i,"sleeve"],
                       "real_open": m.loc[i,"opened_at"], "dt_min": (rep_ns.loc[j,"opened_at"]-m.loc[i,"opened_at"]).total_seconds()/60,
                       "real_px": m.loc[i,"entry_price"], "rep_px": rep_ns.loc[j,"entry_price"],
                       "real_pnl%": m.loc[i,"pnl_percentage"], "rep_pnl%": rep_ns.loc[j,"pnl_percentage"],
                       "real_exit": m.loc[i,"close_reason"], "rep_exit": rep_ns.loc[j,"close_reason"]} for i, j in pairs])
    d["px_gap_bps"] = (d.rep_px / d.real_px - 1) * 1e4
    print("\nmatched pairs:"); print(d.round(3).to_string(index=False))
    print(f"\nmatched: mean Δt {d.dt_min.mean():+.1f} min | mean entry gap {d.px_gap_bps.mean():+.1f} bps | "
          f"real pnl% mean {d['real_pnl%'].mean():+.3f} vs replay {d['rep_pnl%'].mean():+.3f}")
miss = m[~m.index.isin(list(mi))]
if len(miss):
    print("\nREAL fills NOT reproduced:"); print(miss[["opened_at","pair","direction","sleeve","entry_order_type","pnl","entry_pair_rank"]].to_string(index=False))
extra = rep_ns[~rep_ns.index.isin(list(ri))]
if len(extra):
    print("\nREPLAY fills with no live counterpart:"); print(extra[["opened_at","pair","direction","sleeve","entry_order_type","pnl","close_reason"]].head(60).to_string(index=False))
