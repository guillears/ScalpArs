#!/usr/bin/env python3
"""🧭 Post-Exit Regret per sleeve × exit name (DECISION_LOG 421, operator 2026-10-30: "when we review the different exit strategies for each sleeve … a deep review on the Post-Exit Regret").
The dashboard table groups by exit name across every sleeve; this runs the SAME row computation (services/post_exit_regret.py) inside each sleeve, master and current batch apart, with the exits
that have no post-exit data listed, then lists EVERY TRADE of the sleeve with its own post-exit path (operator: "on each transaction"). It shows what price did in the tracking window after a close, never a counterfactual exit: a tempting peak still has to pass the live-stopped, two-sided test.
No network, no config or DB change (export_glob keeps its usual copies of new exports in exports_archive/). The master block is today's stack (stack_keep, no probes) unless --all; Close % is as traded and
the master's Stack % sits beside it. A sleeve with both directions is split (MOMENTUM_LONG / MOMENTUM_SHORT); each exit version gets its own table (Flatbase old / new, LITE +2 / +3).
Usage: venv/bin/python scripts/post_exit_regret_by_sleeve.py --since "<batch start UTC>" [SLEEVE] [--all]"""
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
sys.path.insert(0, os.path.dirname(_D))
import export_glob as EG                                                                   # noqa: E402
from services.post_exit_regret import post_exit_regret_rows, post_exit_untracked           # noqa: E402

FLATBASE_NEW_AT = pd.Timestamp("2026-10-28 21:37")   # the DL 408 commit + 10 min (+5 / −10 live), same as the dashboard split
WINDOW_MAX = 46                      # minutes: the tracker runs 45; longer = a path that overran (restart), flagged
MASTER = os.path.join(os.path.dirname(_D), "reports", "MASTER_POOL_stacked.csv")
TIME_COLS = ("opened_at", "closed_at", "post_exit_regime_flip_at")


def to_orders(d):
    """rows → attribute objects the row computation reads; a NaN becomes None. PURE."""
    d = d.copy()
    for c in TIME_COLS:
        if c in d:
            d[c] = pd.to_datetime(d[c], format="mixed", errors="coerce")
    return [NS(**{k: (None if pd.isna(v) else v) for k, v in r.items()}) for r in d.to_dict("records")]


def table(d):
    """one sleeve's fills → markdown lines. PURE."""
    o = to_orders(d)
    L = ["| Exit | N | Dir | Close % | Post peak % | at min | Post trough % | at min | Peak first | Final % |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in post_exit_regret_rows(o):
        L.append(f"| {r['reason']} | {r['count']} | {r['longs']}L/{r['shorts']}S | {r['avg_close_pnl']:+.2f} | {r['avg_post_peak']:+.2f} | {r['avg_peak_min']:.0f} | "
                 f"{r['avg_post_trough']:+.2f} | {r['avg_trough_min']:.0f} | {r['peak_first_pct']:.0f} % | {r['avg_final_pnl']:+.2f} |")
    u = post_exit_untracked(o)
    n = sum(bool(_over(r)) for r in d.to_dict("records"))
    if n:
        L.append(f"⚠ {n} trade(s) tracked past 45 min are in these averages: their Final is not a 45-min reading.")
    if u:
        L.append("No post-exit data: " + " · ".join(f"{x['reason']} {x['n']}" for x in u))
    return L


def _over(r):
    m = pd.to_numeric(pd.Series([r.get("post_exit_peak_minutes"), r.get("post_exit_trough_minutes")]), errors="coerce").max()
    return " ⚠ past 45 min" if pd.notna(m) and m > WINDOW_MAX else ""


def trades(d):
    """one sleeve's fills, each on its own row, newest first → markdown lines: the exit, then what price did after it. PURE."""
    def n(v, f="{:+.2f}"):
        v = pd.to_numeric(v, errors="coerce")
        return "–" if pd.isna(v) else f.format(float(v))
    stacked = "stack_pct" in d
    L = ["| Opened UTC | Pair | Dir | Exit | Close % | Stack % | Peak in trade % | +1 min | +5 | +15 | +30 | Post peak % | at min | Post trough % | at min | Final % |", "|" + "---|" * 16]
    for r in d.sort_values("opened_at", ascending=False).to_dict("records"):
        L.append(f"| {str(r.get('opened_at'))[:16]} | {r.get('pair')} | {r.get('direction')} | {r.get('close_reason')} | {n(r.get('pnl_percentage'))} | {n(r.get('stack_pct')) if stacked else '–'} | {n(r.get('peak_pnl'))} | "
                 f"{n(r.get('post_exit_pnl_at_1min'))} | {n(r.get('post_exit_pnl_at_5min'))} | {n(r.get('post_exit_pnl_at_15min'))} | {n(r.get('post_exit_pnl_at_30min'))} | "
                 f"{n(r.get('post_exit_peak_pnl'))} | {n(r.get('post_exit_peak_minutes'), '{:.0f}')} | {n(r.get('post_exit_trough_pnl'))} | {n(r.get('post_exit_trough_minutes'), '{:.0f}')} | {n(r.get('post_exit_final_pnl'))}{_over(r)} |")
    return L


def load_batch(since):
    fr = []
    for f in EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            fr.append(pd.read_csv(f, low_memory=False).assign(_m=os.path.getmtime(f)))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    d = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable")
    d = d[d.status.astype(str) == "CLOSED"].assign(_k=lambda x: x.opened_at.astype(str).str.replace(" ", "T").str[:19])
    d = d.drop_duplicates(["_k", "pair", "direction"], keep="last")
    return d[pd.to_datetime(d.opened_at, format="mixed") >= pd.Timestamp(since)].drop(columns=["_m", "_k"])


def split_sleeves(d):
    """→ [(label, fills)], a strategy with both directions split in two. PURE."""
    out = []
    for es, g in d.groupby(d.entry_strategy.astype(str)):
        if g.direction.astype(str).nunique() > 1:
            out += [(f"{es}_{k}", h) for k, h in g.groupby(g.direction.astype(str))]
        else:
            out.append((es, g))
    return out


def versions(label, g):
    """→ [(exit version label, fills)]: fills that ran different targets never share an average. PURE."""
    if label == "FLATBASE_SHORT":
        o, c = pd.to_datetime(g.opened_at, format="mixed"), pd.to_datetime(g.closed_at, format="mixed", errors="coerce")
        new, old = o >= FLATBASE_NEW_AT, c < FLATBASE_NEW_AT
        parts = (("today's exit +5 / −10", g[new]), ("older exit +8 / −15", g[old]), ("open across the change (mixed)", g[~new & ~old]))
    elif label in ("FRENZY_LITE", "FRENZY_LITE_RETRY") and "frenzy_lite_tp_used" in g:
        t = pd.to_numeric(g.frenzy_lite_tp_used, errors="coerce")
        parts = (("ran the +2 target", g[t == 2.0]), ("ran +3 (or the older exit)", g[t != 2.0]))
    else:
        return [("", g)]
    return [(n, x) for n, x in parts if len(x)]


def master_view(d, keep_all=False):
    """→ (rows to read, one line on what was left out). PURE."""
    if keep_all:
        return d, "raw pool (--all): every row, blocked and probes included"
    k = d.stack_keep.astype(str).isin(("True", "true", "1", "1.0")) & ~d.is_probe.astype(str).isin(("True", "true", "1", "1.0"))
    out = d[~k].groupby(d.entry_strategy.astype(str)).size().sort_values(ascending=False)
    return d[k], "today's stack, no probes · left out: " + " · ".join(f"{a} {b}" for a, b in out.items())


def main(argv):
    a = list(argv[1:])
    if "--since" not in a:
        sys.exit('--since "<batch start UTC>" is required (the current batch\'s first fill window)')
    i = a.index("--since")
    since = a[i + 1]
    del a[i:i + 2]
    keep_all = "--all" in a
    a = [x for x in a if x != "--all"]
    only = a[0] if a else None
    m, note = master_view(pd.read_csv(MASTER, low_memory=False), keep_all)
    for title, d in ((f"Current batch (opened ≥ {since} UTC, as booked)", load_batch(since)), (f"Master ({note})", m)):
        print(f"\n# {title}\n")
        if not len(d):
            print("none")
            continue
        groups = split_sleeves(d)
        hit = [g for g in groups if only is None or g[0] == only]
        if not hit:
            print(f"no fills for {only}; sleeves: " + ", ".join(n for n, _ in groups))
        for es, g in hit:
            print(f"\n## {es} ({len(g)} closed)\n")
            for v, x in versions(es, g):
                if v:
                    print(f"\n### {v} ({len(x)})\n")
                print("\n".join(table(x)))
            print("\nEach trade:\n")
            print("\n".join(trades(g)))


if __name__ == "__main__":
    main(sys.argv)
