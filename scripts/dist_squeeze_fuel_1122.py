#!/usr/bin/env python3
"""DIST_SQUEEZE_FUEL 2026-11-22 — theory (e) "crowded shorts → fast squeeze stop" on DISTRIBUTION_SHORT (pre-registration:
reports/DIST_SQUEEZE_FUEL_2026-11-22_PREREG.md). Backtest cohort = DIST_WL trades, today's armed rule (rsi1h ≥ 74.07), Exit B
(LIVE_*: LIVE_k 0 TP · 1 SL · 2 time · 3 unfinished, excluded — scripts/dist_wl_analyze.py).
Unscored ≠ rest: each variable is read on the rows where it is scored; the crowd score only on rows scored on all five.
Outputs: _tables.csv (per variable, terciles + PREREG pass rule), _signs.csv (sign granularity), _cuts.csv, _before_after.csv
(block each crowded side: N · WR · avg · DCR at 0.61 × equity · max DD). Research only.
Usage: venv/bin/python -I scripts/dist_squeeze_fuel_1122.py"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports", "DIST_SQUEEZE_FUEL_2026-11-22")
T = pd.read_csv(os.path.join(ROOT, "reports", "DIST_WL_2026-11-18_trades.csv.gz"))
T = T[(T.in_LIVE.astype(bool)) & (T.LIVE_k != 3) & (T.rsi1h >= 74.07)].copy()
T["fast"] = ((T.LIVE_k == 1) & ((T.LIVE_x - T.entry_ms) <= 60 * 60_000)).astype(float)
T["net"] = T.net_LIVE.astype(float)
T["H"] = np.where(T.block == "DESIGN", "DESIGN", "HOLDOUT")
LOW = ["fund_last", "prem_last", "top_pos_ls", "top_acct_ls", "glob_ls", "taker_1h"]
HIGH = ["oi_chg1h"]
D = T[T.H == "DESIGN"]
q1 = {v: D[v].quantile(1 / 3) for v in LOW + HIGH}
q2 = {v: D[v].quantile(2 / 3) for v in LOW + HIGH}
cut = {v: q1[v] for v in LOW} | {v: q2[v] for v in HIGH}
for v in LOW + HIGH:   # zone = the crowded tercile; NaN = unscored (neither zone nor rest)
    terc = pd.cut(T[v], [-np.inf, q1[v], q2[v], np.inf], labels=["lo", "mid", "hi"])   # (lo ≤ q1] (q1, q2] (q2, ∞)
    T["t_" + v] = terc
    T["z_" + v] = (terc == ("lo" if v in LOW else "hi")).where(T[v].notna())
full = T[LOW[:5]].notna().all(axis=1)
T["score"] = T[["z_" + v for v in LOW[:5]]].astype(float).sum(axis=1).where(full)
T["z_score3"] = (T.score >= 3).where(full)
ZS = ["z_" + v for v in LOW + HIGH] + ["z_score3"]
D = T[T.H == "DESIGN"]   # re-read with the zone / tercile columns


def welch(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def side(g, z):
    m = g[z]
    return g[m == True], g[m == False]   # noqa: E712 (NaN rows drop out)


rng = np.random.default_rng(1122)
Dd = T[T.H == "DESIGN"].reset_index(drop=True)
null = []
for _ in range(200):
    p = pd.Series(rng.permutation(Dd.net.values))
    null.append(np.nanmax([abs(welch(p[Dd[z] == True], p[Dd[z] == False])) for z in ZS]))   # noqa: E712
NULL = float(np.quantile(null, 0.95))
rows = []
for z in ZS:
    v = z[2:]
    r = dict(var=v, cut=cut.get(v, 3))
    for h in ("DESIGN", "HOLDOUT"):
        g = T[T.H == h]; a, b = side(g, z)
        r[f"{h}_scored_pct"] = (g[z].notna().mean()) * 100
        r[f"{h}_n_zone"] = len(a); r[f"{h}_fast_zone"] = a.fast.mean() * 100; r[f"{h}_fast_rest"] = b.fast.mean() * 100
        r[f"{h}_net_zone"] = a.net.mean(); r[f"{h}_net_rest"] = b.net.mean(); r[f"{h}_t"] = welch(a.net, b.net)
        if v != "score3":
            r[f"{h}_terc"] = " / ".join(f"{g.net[g['t_' + v] == k].mean():+.2f}" for k in ["lo", "mid", "hi"])
    if v != "score3":
        tm = [D.net[D["t_" + v] == k].mean() for k in ["lo", "mid", "hi"]]
        r["design_monotone"] = bool(np.all(np.diff(tm) > 0) or np.all(np.diff(tm) < 0))
    else:
        sm = D.groupby(T.loc[D.index, "score"]).net.mean().values
        r["design_monotone"] = bool(np.all(np.diff(sm) > 0) or np.all(np.diff(sm) < 0))
    a = side(T[T.H == "DESIGN"], z)[0]
    pl = a.groupby("pair").net.sum()   # top-pair share = the worst losing pair ÷ all losing pairs' loss, crowded side, DESIGN
    r["top_pair_loss_share"] = (-pl[pl < 0].min() / -pl[pl < 0].sum()) if (pl < 0).any() else 0.0
    rows.append(r)
R = pd.DataFrame(rows)
R["pass"] = ((R.DESIGN_fast_zone > R.DESIGN_fast_rest) & (R.DESIGN_net_zone < R.DESIGN_net_rest) & (R.DESIGN_t.abs() > NULL)
             & (R.HOLDOUT_fast_zone > R.HOLDOUT_fast_rest) & (R.HOLDOUT_net_zone < R.HOLDOUT_net_rest) & R.design_monotone
             & (R.top_pair_loss_share < 0.5))
print(f"cohort {len(T)} (DESIGN {len(Dd)}, HOLDOUT {len(T) - len(Dd)}) · base fast-stop {T.fast.mean() * 100:.1f} % · net {T.net.mean():+.3f} · theory null |t| {NULL:.2f}")
pd.set_option("display.width", 280)
print(R.round(3).to_string(index=False))
print("score dose (net, fast-stop share), rows scored on all five:")
for h in ("DESIGN", "HOLDOUT"):
    g = T[(T.H == h) & full]
    print(h, g.groupby("score").agg(n=("net", "size"), net=("net", "mean"), fast=("fast", "mean")).round(3).to_dict("index"))
# sign granularity (PREREG)
S = []
for v, c in [("fund_last", 0.0), ("prem_last", 0.0), ("top_pos_ls", 1.0), ("glob_ls", 1.0)]:
    for h in ("DESIGN", "HOLDOUT"):
        g = T[(T.H == h) & T[v].notna()]; a, b = g[g[v] < c], g[g[v] >= c]
        S.append(dict(var=f"{v} < {c:g}", block=h, n_zone=len(a), fast_zone=a.fast.mean() * 100, fast_rest=b.fast.mean() * 100,
                      net_zone=a.net.mean(), net_rest=b.net.mean(), t=welch(a.net, b.net)))
S = pd.DataFrame(S)
print(S.round(3).to_string(index=False))
# before → after (block the crowded side; unscored rows are kept, as live would fail-open)
X = 0.61
days = pd.date_range(pd.to_datetime(T.entry_ms, unit="ms").min().normalize(), pd.to_datetime(T.LIVE_x, unit="ms").max().normalize())


def book(g):
    d = pd.to_datetime(g.LIVE_x, unit="ms").dt.normalize()
    r = (g.net * X / 100).groupby(d.values).sum().reindex(days, fill_value=0.0)
    eq = (1 + r).cumprod()
    return dict(N=len(g), WR=(g.net > 0).mean() * 100, avg=g.net.mean(), dcr=((1 + r).prod() ** (1 / len(days)) - 1) * 100,
                max_dd=(eq / eq.cummax() - 1).min() * 100)


BA = [dict(rule="today", **book(T))]
for z in ["z_fund_last", "z_top_pos_ls", "z_score3"]:
    blk = T[z] == True   # noqa: E712
    BA.append(dict(rule="block " + z[2:], **book(T[~blk]), blocked_n=int(blk.sum()), blocked_avg=T.net[blk].mean(),
                   blocked_WR=(T.net[blk] > 0).mean() * 100))
BA = pd.DataFrame(BA)
print(BA.round(3).to_string(index=False))
R.to_csv(REP + "_tables.csv", index=False)
S.to_csv(REP + "_signs.csv", index=False)
BA.to_csv(REP + "_before_after.csv", index=False)
pd.Series(cut).to_csv(REP + "_cuts.csv")
