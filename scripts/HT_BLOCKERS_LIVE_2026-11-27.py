#!/usr/bin/env python3
"""HT_BLOCKERS_LIVE_2026-11-27 — the live HT refusals priced on HT's exit (research only). Source: the decisions exports' BLOCKED rows
(scout_blocked_filters.load_blocked) joined to the blocked-trade pricer's store reports/SCOUT_BLOCKED_FILTERS.csv (TP +6 net / SL −4 /
24 h, refusal + 8 s, 0.10 % slip). Out: reports/HT_BLOCKERS_LIVE_2026-11-27.csv (one row per refusal) + _summary.txt.
Usage: venv/bin/python scripts/HT_BLOCKERS_LIVE_2026-11-27.py [last_fill_utc]"""
import os, sys
import pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [os.path.join(R, "scripts"), R]
import scout_blocked_filters as B   # noqa: E402

LAST = sys.argv[1] if len(sys.argv) > 1 else "2026-11-21 02:30:14"
d = B.load_blocked()
d = d[d.sleeve == "FRENZY_WILLY_HT"].merge(pd.read_csv(os.path.join(R, "reports", "SCOUT_BLOCKED_FILTERS.csv"))[["key", "pct", "why"]], on="key", how="left")
d["ts"] = pd.to_datetime(d.t_ms, unit="ms")
d["vs_vwap"] = [s.get("entry_frenzy_vs_vwap_pct") for s in d.snap]
d["dip"] = [s.get("entry_frenzy_willy_dip_pct") for s in d.snap]
out = d[["ts", "pair", "lines", "sole", "pct", "why", "vs_vwap", "dip"]]
out.to_csv(os.path.join(R, "reports", "HT_BLOCKERS_LIVE_2026-11-27.csv"), index=False)


def row(g, lab):
    p = g[g.pct.notna()]
    return (f"{lab}: {len(g)} refused · {len(p)} priced · TP {int((p.why == 'TP').sum())} / STOP {int((p.why == 'STOP').sum())} · "
            f"WR {(p.pct > 0).mean():.0%} · avg {p.pct.mean():+.2f} · Σ {p.pct.sum():+.1f} · {p.ts.dt.date.nunique()} days")


L = [f"HT refusals since arming ({d.ts.min():%Y-%m-%d}) and since the last HT fill ({LAST} UTC)"]
for lab, g in (("since arming", d), ("since last fill", d[d.ts > LAST])):
    L.append(f"## {lab}")
    L += [row(x, ln) for ln, x in g.groupby("lines")] + [row(g, "ALL")]
s = d[(d.ts > LAST) & (d.lines == "WILLY_FAILED_SPIKE") & d.pct.notna()]
L.append(f"FAILED_SPIKE since last fill by depth: shallow (−1…0) {row(s[s.vs_vwap > -1], 'n')} · deeper {row(s[s.vs_vwap <= -1], 'n')}")
open(os.path.join(R, "reports", "HT_BLOCKERS_LIVE_2026-11-27_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
